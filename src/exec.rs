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
use crate::plan::{PlanGraph, PlanId, PlanNode, shared_flags};
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

/// A lowered program with its plan graph, one plan root per statement, parallel to
/// `lowered.stmts`, and the shared flag of every plan node.
pub struct Plan {
    pub lowered: Lowered,
    pub graph: PlanGraph,
    pub roots: Vec<PlanId>,
    /// Per `PlanId`, whether a run evaluates the node at most once and reuses its value, see
    /// `shared_flags`.
    pub shared: Vec<bool>,
}

impl Plan {
    /// The plan of `lowered` with the shared flags of `graph` and `roots`.
    pub fn new(reg: &Registry, lowered: Lowered, graph: PlanGraph, roots: Vec<PlanId>) -> Plan {
        let shared = shared_flags(reg, &graph, &roots);
        Plan {
            lowered,
            graph,
            roots,
            shared,
        }
    }
}

/// How the executor runs a plan. Neither setting changes a value or an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strategy {
    pub tiling: Tiling,
    /// Whether a shared plan node, see `Plan::shared`, is evaluated once per run and its
    /// value reused. When off, every path to it evaluates it.
    pub sharing: bool,
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
    /// Incremented once per evaluation of a shared node that reused its value.
    pub shared_hits: &'a AtomicU64,
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
/// error fails the run. Errors of other kinds fail the run at once. `how` decides how regions
/// run and whether shared nodes are evaluated once, and changes no value and no error. With
/// sharing, a shared node's value is kept from its first successful evaluation to the end of
/// the run. A failed evaluation is not kept, so every statement that needs the node
/// evaluates it again.
pub fn run_plan(
    reg: &Registry,
    dispatch: &Dispatch,
    plan: &Plan,
    env: &mut Env,
    how: Strategy,
    mut obs: Observe<'_>,
    sinks: &mut Sinks<'_>,
) -> Result<(), Error> {
    // Without a shared node the memo stays empty and is never read.
    let sharing = how.sharing && plan.shared.contains(&true);
    let shared: &[bool] = if sharing { &plan.shared } else { &[] };
    let mut memo: Vec<Option<Value>> = if sharing {
        vec![None; plan.graph.len()]
    } else {
        Vec::new()
    };
    for (stmt, &root) in plan.lowered.stmts.iter().zip(&plan.roots) {
        let span = stmt.span();
        let result = Machine {
            reg,
            dispatch,
            graph: &plan.graph,
            leaves: &plan.lowered.leaves,
            env,
            span,
            trace: obs.trace.then_some(&mut *sinks.diag),
            stats: obs.stats.as_deref_mut(),
            tiling: how.tiling,
            tiled_regions: obs.tiled_regions,
            shared,
            memo: &mut memo,
            shared_hits: obs.shared_hits,
        }
        .eval(root);
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
    graph: &'a PlanGraph,
    leaves: &'a [Leaf],
    env: &'a Env,
    span: Span,
    trace: Option<&'a mut dyn Write>,
    stats: Option<&'a mut Stats>,
    tiling: Tiling,
    tiled_regions: &'a AtomicU64,
    /// Per `PlanId`, whether the node is shared. Empty when nothing is shared.
    shared: &'a [bool],
    /// Per `PlanId`, the value of a shared node after its first successful evaluation in this
    /// run. Empty when nothing is shared.
    memo: &'a mut [Option<Value>],
    shared_hits: &'a AtomicU64,
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
struct Region {
    inputs: Vec<PlanId>,
    kernels: Vec<RegionKernel>,
    /// Whether every input that is not trivial comes before the first kernel in post-order,
    /// so that the whole evaluation reaches all of them before it calls any region kernel.
    ordered: bool,
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

impl<'a> Machine<'a> {
    fn observing(&self) -> bool {
        self.trace.is_some() || self.stats.is_some()
    }

    fn shared(&self, node: PlanId) -> bool {
        self.shared.get(node.index()).copied().unwrap_or(false)
    }

    /// Whether `node` is shared and its value is in the memo.
    fn memoized(&self, node: PlanId) -> bool {
        self.memo.get(node.index()).is_some_and(Option::is_some)
    }

    fn eval(&mut self, node: PlanId) -> Result<Value, Error> {
        let graph = self.graph;
        match graph.node(node) {
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
            PlanNode::Bcast(inner) => self.eval(*inner),
            PlanNode::Cond(c, t, e) => match self.eval(*c)? {
                Value::Scalar(Scalar::Bool(b)) => self.eval(if b { *t } else { *e }),
                // The checker types every condition as bool, so another value is a typing or
                // lowering bug, never a value of the program.
                other => panic!("if condition is {}, not bool", other.ty()),
            },
            PlanNode::Kernel(id, args) => {
                if !self.shared(node) {
                    return self.eval_kernel(node, *id, args);
                }
                // A hit calls no kernel, so it writes no trace line and adds no statistics.
                if let Some(v) = &self.memo[node.index()] {
                    self.shared_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(v.clone());
                }
                // A failure is not kept, so the next statement that needs the node raises it
                // again with its own span.
                let result = self.eval_kernel(node, *id, args);
                if let Ok(v) = &result {
                    self.memo[node.index()] = Some(v.clone());
                }
                result
            }
        }
    }

    /// Evaluates the kernel node `node`, a call of kernel `id` on `args`, as a region root
    /// or with one whole call.
    fn eval_kernel(&mut self, node: PlanId, id: KernelId, args: &[PlanId]) -> Result<Value, Error> {
        // A region root is a region kernel with a region kernel argument. Evaluation
        // reaches a region's root before its other kernels, so the region is maximal.
        if let Tiling::Rows(t) = self.tiling
            && self.region_kernel(node).is_some()
            && args.iter().any(|a| self.region_kernel(*a).is_some())
        {
            return self.eval_region(node, t.get());
        }
        let args = args
            .iter()
            .map(|a| self.eval(*a))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_whole(id, &args)
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
    /// `Elementwise` with an array output that is not shared.
    fn region_kernel(&self, node: PlanId) -> Option<(KernelId, &'a [PlanId])> {
        // A shared node runs through `eval` and the memo, so it is a region input that bounds
        // the region above it.
        if self.shared(node) {
            return None;
        }
        let PlanNode::Kernel(id, args) = self.graph.node(node) else {
            return None;
        };
        let def = self.reg.get(*id);
        (def.class == KernelClass::Elementwise && matches!(def.output, ValueType::Array(_)))
            .then_some((*id, args.as_slice()))
    }

    /// Adds the region kernels of the subtree `node` to `region` in post-order, and the
    /// subtrees just below them to its inputs, left to right. Clears `region.ordered` when an
    /// input that is not trivial follows a kernel. Returns where `node`'s value comes from.
    /// A node reached twice is added twice, as in the plan as a tree.
    fn collect(&self, node: PlanId, region: &mut Region) -> Src {
        let Some((id, args)) = self.region_kernel(node) else {
            // A shared input whose value is in the memo can neither block nor fail, so it is
            // as trivial as a leaf. So is a shared input that is an earlier input of this
            // region too: whole evaluation reaches the earlier one first, and either stops at
            // its failure or finds the value in the memo here. Any other shared input is not
            // trivial.
            let reused =
                self.memoized(node) || (self.shared(node) && region.inputs.contains(&node));
            if !trivial(self.graph, node) && !reused && !region.kernels.is_empty() {
                region.ordered = false;
            }
            region.inputs.push(node);
            return Src::Input(region.inputs.len() - 1);
        };
        let args = args.iter().map(|a| self.collect(*a, region)).collect();
        let ValueType::Array(elem) = self.reg.get(id).output else {
            unreachable!("a region kernel has an array output");
        };
        region.kernels.push(RegionKernel { id, args, elem });
        Src::Kernel(region.kernels.len() - 1)
    }

    /// Evaluates the region rooted at `node` in tiles of `t` rows. When the tiled run fails
    /// anywhere, or tiling would evaluate an input that is not trivial before a region kernel
    /// that the whole evaluation calls first, it evaluates the region whole instead.
    fn eval_region(&mut self, node: PlanId, t: usize) -> Result<Value, Error> {
        let mut region = Region {
            inputs: Vec::new(),
            kernels: Vec::new(),
            ordered: true,
        };
        self.collect(node, &mut region);
        // A tiled run evaluates every input before any kernel. Such an input after a kernel
        // may block or run long where the whole evaluation raises that kernel's error first.
        // Trivial inputs after a kernel can at most hold a stored error, which the fallback
        // raises in whole-evaluation order.
        if !region.ordered {
            return self.eval_region_whole(node, &mut Vec::new().into_iter());
        }
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
    /// which holds the slots of a prefix of the inputs, of which only the last may have
    /// failed. The inputs after that prefix are evaluated when reached.
    fn eval_region_whole(
        &mut self,
        node: PlanId,
        inputs: &mut std::vec::IntoIter<Slot>,
    ) -> Result<Value, Error> {
        let Some((id, args)) = self.region_kernel(node) else {
            // The whole evaluation stops at the first failed input, so it never asks for an
            // input after it.
            return match inputs.next() {
                Some(slot) => slot,
                None => self.eval(node),
            };
        };
        let args = args
            .iter()
            .map(|a| self.eval_region_whole(*a, inputs))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_whole(id, &args)
    }

    /// Evaluates the inputs of `region`, then runs its kernels tile by tile over `t` rows,
    /// writing the root's tiles into its output array. Fails when an input fails, an input is
    /// a string, the array inputs differ in length or none is an array, an allocation fails,
    /// or a tile fails, and then returns the inputs evaluated so far, ending with the failed
    /// one if any.
    fn run_tiled(&mut self, region: &Region, t: usize) -> Result<TiledRun, Vec<Slot>> {
        let mut slots = Vec::with_capacity(region.inputs.len());
        for &n in &region.inputs {
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

/// Whether evaluating `node` can neither block nor do unbounded work: a leaf, or a broadcast
/// of a leaf. Such a node can still return a stored error.
fn trivial(graph: &PlanGraph, node: PlanId) -> bool {
    match graph.node(node) {
        PlanNode::Leaf(_) => true,
        PlanNode::Bcast(inner) => matches!(graph.node(*inner), PlanNode::Leaf(_)),
        PlanNode::Cond(..) | PlanNode::Kernel(..) => false,
    }
}
