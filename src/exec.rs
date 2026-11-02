//! The executor. It evaluates planned statements through the kernel table.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::check::VarId;
use crate::error::{Error, ErrorKind, Span};
use crate::kernels::{Dispatch, KernelId, Registry, Variant};
use crate::lower::{LStmt, Leaf, Lowered};
use crate::ops::Op;
use crate::plan::{PlanGraph, PlanId, PlanNode, shared_flags};
use crate::value::{Scalar, Value};

mod region;

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

/// How the executor runs a plan. No setting changes a value or an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strategy {
    pub tiling: Tiling,
    /// Whether a shared plan node, see `Plan::shared`, is evaluated once per run and its
    /// value reused. When off, every path to it evaluates it.
    pub sharing: bool,
    /// Whether a tiled region evaluates the right operand of an `and_bool` or `or_bool` only
    /// over the rows of a tile from the first to the last row where the left operand leaves
    /// the result open, true for `and` and false for `or`, when every region kernel of that
    /// operand is total, see `KernelDef::total`. It has no effect without tiling.
    pub selective: bool,
}

/// Totals of one kernel over a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelStat {
    pub calls: u64,
    /// Bytes of array arguments and array results.
    pub bytes: u64,
    /// Bytes of array arguments.
    pub read: u64,
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
                "stats {name} calls={} bytes={} read={} time={micros:.1}",
                s.calls, s.bytes, s.read
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
    /// Incremented once per guard and tile in which the guard's right operand ran over fewer
    /// rows than the guard's, see `Strategy::selective`, for the region runs that completed
    /// tiled.
    pub narrowed_tiles: &'a AtomicU64,
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
            selective: how.selective,
            tiled_regions: obs.tiled_regions,
            narrowed_tiles: obs.narrowed_tiles,
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
    /// See `Strategy::selective`.
    selective: bool,
    tiled_regions: &'a AtomicU64,
    narrowed_tiles: &'a AtomicU64,
    /// Per `PlanId`, whether the node is shared. Empty when nothing is shared.
    shared: &'a [bool],
    /// Per `PlanId`, the value of a shared node after its first successful evaluation in this
    /// run. Empty when nothing is shared.
    memo: &'a mut [Option<Value>],
    shared_hits: &'a AtomicU64,
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
            s.read += u64::try_from(read).expect("byte count fits in u64");
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
