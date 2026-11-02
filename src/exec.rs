//! The executor. It evaluates planned statements through the kernel table.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::check::VarId;
use crate::error::{Error, ErrorKind, Span};
use crate::kernels::{Dispatch, KernelClass, KernelId, Registry, Variant};
use crate::lower::{LStmt, Leaf, Lowered};
use crate::ops::Op;
use crate::plan::PlanNode;
use crate::value::{Arg, Array, Scalar, ScalarType, TileBuf, Value, ValueType};

#[cfg(test)]
mod tests;

/// The tile row count of `Tiling::default()`.
// 1024 rows is 22% slower than untiled, and 4096 rows is up to 6% slower on the scalar and AVX2
// variants. The time does not improve measurably beyond 16384 rows, which keeps one `i64` tile
// at 128 KiB. See `docs/measurements/2026-11-02-tiled-regions.md`.
pub const DEFAULT_TILE_ROWS: usize = 16384;

/// How the executor runs regions, the subtrees of a plan made of at least two elementwise
/// kernels with array outputs. `Off` runs every kernel over whole arrays. `Rows(t)` runs each
/// region tile by tile over `t` rows at a time, with the values and errors of `Off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiling {
    Off,
    Rows(NonZeroUsize),
}

impl Default for Tiling {
    /// `Rows(DEFAULT_TILE_ROWS)`.
    fn default() -> Tiling {
        Tiling::Rows(NonZeroUsize::new(DEFAULT_TILE_ROWS).expect("the default tile has rows"))
    }
}

/// What a variable holds: its value, or the runtime error its binding raised. Values are
/// `Arc`-backed, so cloning one never copies data.
pub type Slot = Result<Value, Error>;

/// The variable environment. A read of a variable that holds an error raises that error.
pub type Env = HashMap<VarId, Slot>;

/// A lowered program with one plan per statement, parallel to `lowered.stmts`.
pub struct Plan {
    pub lowered: Lowered,
    pub nodes: Vec<PlanNode>,
}

/// Totals of one kernel over a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelStat {
    pub calls: u64,
    pub bytes: u64,
    pub time: Duration,
}

/// Per-kernel totals, keyed by kernel name.
#[derive(Clone, Debug, Default)]
pub struct Stats(BTreeMap<String, KernelStat>);

impl Stats {
    pub fn write(&self, w: &mut dyn Write) {
        for (name, s) in &self.0 {
            let micros = s.time.as_secs_f64() * 1e6;
            // Statistics are best effort, a failing sink must not fail the program.
            let _ = writeln!(
                w,
                "stats {name} calls={} bytes={} time={micros:.1}",
                s.calls, s.bytes
            );
        }
    }
}

/// What the executor observes while running.
pub struct Observe<'a> {
    pub trace: bool,
    pub stats: Option<&'a mut Stats>,
    /// Incremented once per region run that completed tiled, not through the whole fallback.
    pub tiled_regions: &'a AtomicU64,
}

/// Where statement output goes.
pub struct Sinks<'a> {
    pub out: &'a mut dyn Write,
    /// Trace lines go here, and `print` output too when `print_to_diag` is set.
    pub diag: &'a mut dyn Write,
    pub print_to_diag: bool,
    /// Expression statements are outputs that write to `out` when set. Otherwise they are
    /// evaluated and dropped, with their runtime errors.
    pub show: bool,
}

/// Runs every statement of `plan` in order. A kernel failure becomes a runtime error with
/// the span of its statement, except a failed `load_*`, which is an I/O error. A `Bind` that is not an output and fails with a runtime error
/// stores the error in its variable, and the error is raised by the first output that reads
/// the variable, directly or through other bindings. Outputs are `Print`, `Show` when
/// `sinks.show` is set, and a `Bind` marked as an output, and the first one whose value is an
/// error fails the run. Errors of other kinds fail the run at once. `tiling` decides how
/// regions run and changes no value and no error.
pub fn run_plan(
    reg: &Registry,
    dispatch: &Dispatch,
    plan: &Plan,
    env: &mut Env,
    tiling: Tiling,
    mut obs: Observe<'_>,
    sinks: &mut Sinks<'_>,
) -> Result<(), Error> {
    for (stmt, node) in plan.lowered.stmts.iter().zip(&plan.nodes) {
        let span = stmt.span();
        let result = Machine {
            reg,
            dispatch,
            leaves: &plan.lowered.leaves,
            env,
            span,
            trace: obs.trace.then_some(&mut *sinks.diag),
            stats: obs.stats.as_deref_mut(),
            tiling,
            tiled_regions: obs.tiled_regions,
        }
        .eval(node);
        match stmt {
            LStmt::Bind { var, output, .. } => {
                let slot = match result {
                    // Only a runtime error is a value. A failed write is not.
                    Err(e) if *output || e.kind != ErrorKind::Runtime => return Err(e),
                    slot => slot,
                };
                env.insert(*var, slot);
            }
            LStmt::Print { .. } => {
                let value = result?;
                let w = if sinks.print_to_diag {
                    &mut *sinks.diag
                } else {
                    &mut *sinks.out
                };
                writeln!(w, "{value}").map_err(|e| Error::io(&e))?;
            }
            LStmt::Show { .. } => match result {
                Ok(value) if sinks.show => {
                    writeln!(sinks.out, "{value}").map_err(|e| Error::io(&e))?;
                }
                // Without the echo an expression statement is no output, and its runtime
                // error is dropped like its value.
                Err(e) if sinks.show || e.kind != ErrorKind::Runtime => return Err(e),
                Ok(_) | Err(_) => {}
            },
        }
    }
    Ok(())
}

struct Machine<'a> {
    reg: &'a Registry,
    dispatch: &'a Dispatch,
    leaves: &'a [Leaf],
    env: &'a Env,
    span: Span,
    trace: Option<&'a mut dyn Write>,
    stats: Option<&'a mut Stats>,
    tiling: Tiling,
    tiled_regions: &'a AtomicU64,
}

/// Where a region kernel takes an argument from.
#[derive(Clone, Copy, Debug)]
enum Src {
    /// A region input, by position in `Region::inputs`.
    Input(usize),
    /// The output of a region kernel, by position in `Region::kernels`.
    Kernel(usize),
}

/// A kernel of a region with the sources of its arguments in input order.
struct RegionKernel {
    id: KernelId,
    args: Vec<Src>,
    /// The element type of its output.
    elem: ScalarType,
}

/// A region of a plan: its inputs in the order the whole evaluation reaches them, and its
/// kernels in post-order, so that every kernel comes after the kernels it reads and the root
/// comes last.
struct Region<'p> {
    inputs: Vec<&'p PlanNode>,
    kernels: Vec<RegionKernel>,
}

/// A region run that completed tiled.
struct TiledRun {
    /// The inputs' values, in the order of `Region::inputs`.
    inputs: Vec<Value>,
    /// The rows of every array input and every kernel output.
    rows: usize,
    /// Per region kernel, its time summed over the tiles. Empty unless observing.
    times: Vec<Duration>,
    /// The root's value.
    value: Value,
}

impl Machine<'_> {
    fn observing(&self) -> bool {
        self.trace.is_some() || self.stats.is_some()
    }

    fn eval(&mut self, node: &PlanNode) -> Result<Value, Error> {
        match node {
            PlanNode::Leaf(id) => {
                match &self.leaves[usize::try_from(id.0).expect("u32 fits in usize")] {
                    Leaf::Const(v) => Ok(v.clone()),
                    // A stored error keeps the span of the statement that raised it. A read of
                    // an unbound variable is a lowering bug, never a value of the program.
                    Leaf::Var(v) => match self.env.get(v) {
                        Some(slot) => slot.clone(),
                        None => panic!("variable {v:?} has no value"),
                    },
                }
            }
            // Single-op array kernels take a scalar operand directly and repeat it themselves.
            PlanNode::Bcast(inner) => self.eval(inner),
            PlanNode::Cond(c, t, e) => match self.eval(c)? {
                Value::Scalar(Scalar::Bool(b)) => self.eval(if b { t } else { e }),
                // The checker types every condition as bool, so another value is a typing or
                // lowering bug, never a value of the program.
                other => panic!("if condition is {}, not bool", other.ty()),
            },
            PlanNode::Kernel(id, args) => {
                // A region root is a region kernel with a region kernel argument. Evaluation
                // reaches a region's root before its other kernels, so the region is maximal.
                if let Tiling::Rows(t) = self.tiling
                    && self.region_kernel(node).is_some()
                    && args.iter().any(|a| self.region_kernel(a).is_some())
                {
                    return self.eval_region(node, t.get());
                }
                let args = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>, _>>()?;
                self.call_whole(*id, &args)
            }
        }
    }

    /// Calls kernel `id` once over whole arguments and observes the call.
    fn call_whole(&mut self, id: KernelId, args: &[Value]) -> Result<Value, Error> {
        let start = self.observing().then(Instant::now);
        let result = self.dispatch.call(self.reg, id, args).map_err(|m| {
            // A file that cannot be loaded is an I/O failure, not a runtime value.
            let kind = match self.reg.get(id).op {
                Some(Op::Load(_)) => ErrorKind::Io,
                _ => ErrorKind::Runtime,
            };
            Error {
                kind,
                message: m,
                span: Some(self.span),
            }
        })?;
        if let Some(start) = start {
            self.observe(
                id,
                self.dispatch.variant_used(id),
                args,
                &result,
                start.elapsed(),
            )?;
        }
        Ok(result)
    }

    /// The kernel and arguments of `node` when it is a region kernel: a kernel of class
    /// `Elementwise` with an array output.
    fn region_kernel<'p>(&self, node: &'p PlanNode) -> Option<(KernelId, &'p [PlanNode])> {
        let PlanNode::Kernel(id, args) = node else {
            return None;
        };
        let def = self.reg.get(*id);
        (def.class == KernelClass::Elementwise && matches!(def.output, ValueType::Array(_)))
            .then_some((*id, args.as_slice()))
    }

    /// Adds the region kernels of the subtree `node` to `region` in post-order, and the
    /// subtrees just below them to its inputs, left to right. Returns where `node`'s value
    /// comes from.
    fn collect<'p>(&self, node: &'p PlanNode, region: &mut Region<'p>) -> Src {
        let Some((id, args)) = self.region_kernel(node) else {
            region.inputs.push(node);
            return Src::Input(region.inputs.len() - 1);
        };
        let args = args.iter().map(|a| self.collect(a, region)).collect();
        let ValueType::Array(elem) = self.reg.get(id).output else {
            unreachable!("a region kernel has an array output");
        };
        region.kernels.push(RegionKernel { id, args, elem });
        Src::Kernel(region.kernels.len() - 1)
    }

    /// Evaluates the region rooted at `node` in tiles of `t` rows. When the tiled run fails
    /// anywhere, it evaluates the region whole instead.
    fn eval_region(&mut self, node: &PlanNode, t: usize) -> Result<Value, Error> {
        let mut region = Region {
            inputs: Vec::new(),
            kernels: Vec::new(),
        };
        self.collect(node, &mut region);
        let run = match self.run_tiled(&region, t) {
            Ok(run) => run,
            // The whole evaluation raises the error that the program raises without tiling.
            // It takes the inputs the attempt evaluated instead of evaluating them again,
            // which is the same because evaluation is pure.
            Err(inputs) => return self.eval_region_whole(node, &mut inputs.into_iter()),
        };
        self.tiled_regions.fetch_add(1, Ordering::Relaxed);
        if self.observing() {
            for (k, time) in region.kernels.iter().zip(&run.times) {
                // Bytes and row count are those of the whole arguments and result, as
                // `observe` reports them for a whole call.
                let read = k
                    .args
                    .iter()
                    .map(|src| match *src {
                        Src::Input(i) => match &run.inputs[i] {
                            Value::Array(a) => a.byte_len(),
                            Value::Scalar(_) | Value::Str(_) => 0,
                        },
                        Src::Kernel(j) => run.rows * region.kernels[j].elem.width(),
                    })
                    .sum();
                let write = run.rows * k.elem.width();
                let used = self.dispatch.variant_used(k.id);
                self.record(k.id, used, run.rows, read, write, *time)?;
            }
        }
        Ok(run.value)
    }

    /// Evaluates the region kernels of the subtree `node` whole, each with one call, in the
    /// order of a whole evaluation. The region's inputs come from `inputs` in their order,
    /// which must hold every input up to the first that failed.
    fn eval_region_whole(
        &mut self,
        node: &PlanNode,
        inputs: &mut std::vec::IntoIter<Slot>,
    ) -> Result<Value, Error> {
        let Some((id, args)) = self.region_kernel(node) else {
            // The whole evaluation stops at the first failed input, so it never asks for an
            // input after it.
            return inputs
                .next()
                .expect("the tiled attempt evaluated this input");
        };
        let args = args
            .iter()
            .map(|a| self.eval_region_whole(a, inputs))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_whole(id, &args)
    }

    /// Evaluates the inputs of `region`, then runs its kernels tile by tile over `t` rows,
    /// writing the root's tiles into its output array. Fails when an input fails, an input is
    /// a string, the array inputs differ in length or none is an array, an allocation fails,
    /// or a tile fails, and then returns the inputs evaluated so far, ending with the failed
    /// one if any.
    fn run_tiled(&mut self, region: &Region<'_>, t: usize) -> Result<TiledRun, Vec<Slot>> {
        let mut slots = Vec::with_capacity(region.inputs.len());
        for n in &region.inputs {
            let slot = self.eval(n);
            let failed = slot.is_err();
            slots.push(slot);
            if failed {
                return Err(slots);
            }
        }
        let inputs: Vec<Value> = slots
            .into_iter()
            .map(|s| s.expect("checked above"))
            .collect();
        let fail = |values: Vec<Value>| Err(values.into_iter().map(Ok).collect());
        let mut rows = None;
        for v in &inputs {
            match v {
                Value::Array(a) if rows.is_some_and(|n| n != a.len()) => return fail(inputs),
                Value::Array(a) => rows = Some(a.len()),
                Value::Scalar(_) => {}
                Value::Str(_) => return fail(inputs),
            }
        }
        let Some(n) = rows else {
            return fail(inputs);
        };
        let (root, inner) = region.kernels.split_last().expect("a region has kernels");
        let observing = self.observing();
        // A whole call times the allocation of its output, so the region's allocations are
        // charged to the root below, from this point to the end of the output's construction.
        let alloc_start = observing.then(Instant::now);
        // One tile buffer per inner kernel, allocated once per run and overwritten by each
        // tile. A tile never holds more than `t` rows, or `n` when that is fewer.
        let Ok(mut bufs) = inner
            .iter()
            .map(|k| TileBuf::new(k.elem, t.min(n)))
            .collect::<Result<Vec<_>, _>>()
        else {
            return fail(inputs);
        };
        let mut times = if observing {
            vec![Duration::ZERO; region.kernels.len()]
        } else {
            Vec::new()
        };
        let mut args: Vec<Arg> = Vec::new();
        let w = root.elem.width();
        let built = Array::build_bytes(n, root.elem, |out| {
            let mut start = 0;
            // An empty region still runs one empty tile, so that every kernel is called once
            // as in the whole evaluation.
            loop {
                let len = t.min(n - start);
                for (j, k) in region.kernels.iter().enumerate() {
                    args.clear();
                    args.extend(k.args.iter().map(|src| match *src {
                        Src::Input(i) => match &inputs[i] {
                            Value::Array(a) => Arg::Rows(a.slice(start, len)),
                            Value::Scalar(s) => Arg::Splat(s.clone()),
                            Value::Str(_) => unreachable!("string inputs fall back above"),
                        },
                        Src::Kernel(i) => Arg::Rows(bufs[i].view(len)),
                    }));
                    let dst = match bufs.get_mut(j) {
                        Some(buf) => buf.rows_mut(len),
                        None => &mut out[start * w..(start + len) * w],
                    };
                    let begin = observing.then(Instant::now);
                    self.dispatch.elementwise_tile(self.reg, k.id, &args, dst)?;
                    if let Some(begin) = begin {
                        times[j] += begin.elapsed();
                    }
                }
                // The views of this tile's buffers go before the next tile overwrites them.
                args.clear();
                start += len;
                if start >= n {
                    return Ok(());
                }
            }
        });
        // What the region took beyond its tile calls is the allocation of the tile buffers and
        // of the output, and it goes to the root, whose whole call would have allocated.
        if let Some(start) = alloc_start {
            let tiles: Duration = times.iter().sum();
            if let Some(last) = times.last_mut() {
                *last += start.elapsed().saturating_sub(tiles);
            }
        }
        match built {
            Ok(value) => Ok(TiledRun {
                inputs,
                rows: n,
                times,
                value: Value::Array(value),
            }),
            Err(_) => fail(inputs),
        }
    }

    fn observe(
        &mut self,
        id: KernelId,
        used: Variant,
        args: &[Value],
        result: &Value,
        time: Duration,
    ) -> Result<(), Error> {
        let lens = args.iter().filter_map(|a| match a {
            Value::Array(a) => Some((a.len(), a.byte_len())),
            _ => None,
        });
        let n = lens.clone().map(|(n, _)| n).max().unwrap_or(0);
        let read: usize = lens.map(|(_, b)| b).sum();
        let write = match result {
            Value::Array(a) => a.byte_len(),
            _ => 0,
        };
        self.record(id, used, n, read, write, time)
    }

    /// Adds one call of kernel `id` to the statistics and writes its trace line, for `n` rows,
    /// `read` bytes of array arguments, and `write` bytes of array result.
    fn record(
        &mut self,
        id: KernelId,
        used: Variant,
        n: usize,
        read: usize,
        write: usize,
        time: Duration,
    ) -> Result<(), Error> {
        let name = &self.reg.get(id).name;
        if let Some(stats) = self.stats.as_deref_mut() {
            let s = stats.0.entry(name.clone()).or_default();
            s.calls += 1;
            s.bytes += u64::try_from(read + write).expect("byte count fits in u64");
            s.time += time;
        }
        if let Some(w) = self.trace.as_deref_mut() {
            let secs = time.as_secs_f64();
            // Byte counts stay far below 2^53, so the conversion is exact.
            let total = (read + write) as f64;
            let gbps = if secs > 0.0 { total / secs / 1e9 } else { 0.0 };
            writeln!(
                w,
                "trace {name} {used} n={n} read={read} write={write} time={:.1} gbps={gbps:.2}",
                secs * 1e6,
            )
            .map_err(|e| Error::io(&e))?;
        }
        Ok(())
    }
}
