//! The executor. It evaluates planned statements through the kernel table.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::time::{Duration, Instant};

use crate::check::VarId;
use crate::error::{Error, Span};
use crate::kernels::{Dispatch, KernelId, Registry, Variant};
use crate::lower::{LStmt, Leaf, Lowered};
use crate::plan::PlanNode;
use crate::value::{Scalar, Value};

/// The variable environment. Values are `Arc`-backed, so cloning one never copies data.
pub type Env = HashMap<VarId, Value>;

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
}

/// Where statement output goes.
pub struct Sinks<'a> {
    pub out: &'a mut dyn Write,
    /// Trace lines go here, and `print` output too when `print_to_diag` is set.
    pub diag: &'a mut dyn Write,
    pub print_to_diag: bool,
    /// Expression statements write to `out` when set, otherwise they are evaluated and dropped.
    pub show: bool,
}

/// Runs every statement of `plan` in order. A kernel failure becomes a runtime error with
/// the span of its statement.
pub fn run_plan(
    reg: &Registry,
    dispatch: &Dispatch,
    plan: &Plan,
    env: &mut Env,
    mut obs: Observe<'_>,
    sinks: &mut Sinks<'_>,
) -> Result<(), Error> {
    for (stmt, node) in plan.lowered.stmts.iter().zip(&plan.nodes) {
        let span = stmt.span();
        let value = Machine {
            reg,
            dispatch,
            leaves: &plan.lowered.leaves,
            env,
            span,
            trace: obs.trace.then_some(&mut *sinks.diag),
            stats: obs.stats.as_deref_mut(),
        }
        .eval(node)?;
        match stmt {
            LStmt::Bind { var, .. } => {
                env.insert(*var, value);
            }
            LStmt::Print { .. } => {
                let w = if sinks.print_to_diag {
                    &mut *sinks.diag
                } else {
                    &mut *sinks.out
                };
                writeln!(w, "{value}").map_err(|e| Error::io(&e))?;
            }
            LStmt::Show { .. } => {
                if sinks.show {
                    writeln!(sinks.out, "{value}").map_err(|e| Error::io(&e))?;
                }
            }
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
}

impl Machine<'_> {
    fn eval(&mut self, node: &PlanNode) -> Result<Value, Error> {
        match node {
            PlanNode::Leaf(id) => {
                match &self.leaves[usize::try_from(id.0).expect("u32 fits in usize")] {
                    Leaf::Const(v) => Ok(v.clone()),
                    Leaf::Var(v) => self
                        .env
                        .get(v)
                        .cloned()
                        .ok_or_else(|| Error::runtime("variable has no value", Some(self.span))),
                }
            }
            // Single-op array kernels take a scalar operand directly and repeat it themselves.
            PlanNode::Bcast(inner) => self.eval(inner),
            PlanNode::Cond(c, t, e) => match self.eval(c)? {
                Value::Scalar(Scalar::Bool(b)) => self.eval(if b { t } else { e }),
                other => Err(Error::runtime(
                    format!("if condition is {}, not bool", other.ty()),
                    Some(self.span),
                )),
            },
            PlanNode::Kernel(id, args) => {
                let args = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>, _>>()?;
                let observing = self.trace.is_some() || self.stats.is_some();
                let start = observing.then(Instant::now);
                let result = self
                    .dispatch
                    .call(self.reg, *id, &args)
                    .map_err(|m| Error::runtime(m, Some(self.span)))?;
                if let Some(start) = start {
                    self.observe(
                        *id,
                        self.dispatch.variant_used(*id),
                        &args,
                        &result,
                        start.elapsed(),
                    )?;
                }
                Ok(result)
            }
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
        let name = &self.reg.get(id).name;
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
