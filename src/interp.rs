//! The library API: an interpreter with persistent bindings, and compile-once programs for
//! embedding in a host.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::check::{Globals, TStmt, VarId, check, check_unread};
use crate::error::Error;
use crate::exec::{Env, Observe, Plan, Sinks, Stats, Strategy, Tiling, run_plan};
use crate::kernels::{Backend, Dispatch, FeatureSet, Isa, Registry};
use crate::lower::{Mode, lower};
use crate::plan::{PlanContext, PlanShapes, Planner, SumRewrite, render, render_leaf};
use crate::syntax::parse;
use crate::value::{Value, ValueType};

#[cfg(test)]
mod tests;

/// Interpreter settings. The default has everything off except tiling, which runs with
/// `DEFAULT_TILE_ROWS` rows, and detects the instruction set.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// The instruction set to run kernels with, `None` detects the best one.
    pub isa: Option<Isa>,
    /// How the instruction set's kernels are written, see `Registry::resolve`.
    pub backend: Backend,
    /// Print one plan line per statement to the diagnostic stream before execution.
    pub show_plan: bool,
    /// Print one line per kernel over its whole arguments to the diagnostic stream. A kernel
    /// of a tiled region runs as several tile calls and still prints one line, whose `n` is
    /// the rows it ran over. A shared value computed once and read again prints one line.
    pub trace: bool,
    /// Collect per-kernel totals, written by `Interpreter::write_stats`.
    pub stats: bool,
    /// CPU features to treat as absent, so no kernel entry that needs one is selected and
    /// an instruction set whose level needs one is unavailable.
    pub disable: FeatureSet,
    /// Plan without fused kernel patterns and without the integer sum rewrites, so every plan
    /// uses single-op kernels only and keeps each sum over the array the program sums.
    pub no_fusion: bool,
    /// How chains of elementwise kernels run, see `Tiling`. It changes no value and no error.
    pub tiling: Tiling,
    /// Evaluate a plan node that a run reaches several times once per path, instead of once
    /// per run, see `Strategy::sharing`. It changes no value and no error, only kernel calls.
    pub no_sharing: bool,
    /// Run the right operand of `and` and `or` in a tile over all rows, instead of only over
    /// the rows the left operand leaves open, see `Strategy::selective`. It changes no value
    /// and no error, only the rows kernels run over.
    pub no_selective: bool,
    /// Run a reduction over a chain of elementwise kernels after the chain's whole result,
    /// instead of updating it with each tile of the chain, see `Strategy::fuse_reductions`.
    /// It changes no value and no error.
    pub no_reduction_fusion: bool,
    /// Lower `argsort(a, b)` into two single-key sorts and gathers instead of one composite
    /// kernel. It changes no value, and no error except which operand's error surfaces when
    /// both fail.
    pub no_composite_sort: bool,
}

impl Options {
    fn strategy(&self) -> Strategy {
        Strategy {
            tiling: self.tiling,
            sharing: !self.no_sharing,
            selective: !self.no_selective,
            fuse_reductions: !self.no_reduction_fusion,
        }
    }
}

pub struct Interpreter {
    opts: Options,
    dispatch: Dispatch,
    reg: Registry,
    planner: Planner,
    globals: Globals,
    env: Env,
    stats: Stats,
    /// Region runs that completed tiled, see `tiled_regions`.
    tiled_regions: AtomicU64,
    /// Guard evaluations that narrowed, see `narrowed_tiles`.
    narrowed_tiles: AtomicU64,
    /// Per `KernelId`, fused region runs with that root that completed tiled, see
    /// `fused_reductions`.
    fused_reductions: Vec<AtomicU64>,
    /// Evaluations of shared nodes that reused a value, see `shared_hits`.
    shared_hits: AtomicU64,
}

/// An `input` declaration of a program.
struct InputDecl {
    name: String,
    var: VarId,
    ty: ValueType,
}

/// A checked, lowered, and planned program.
struct Compiled {
    plan: Plan,
    inputs: Vec<InputDecl>,
    /// One `plan <line>: <render>` text per statement, empty unless `show_plan` is set.
    plan_lines: Vec<String>,
    /// Per statement, the integer sum rewrites that matched while planning it.
    sum_rewrites: Vec<BTreeSet<SumRewrite>>,
}

/// A program compiled once and executed many times. It holds no interpreter state.
pub struct Prepared {
    plan: Plan,
    inputs: Vec<InputDecl>,
    /// The `render_leaf` text of every leaf, by `LeafId`.
    leaf_texts: Vec<String>,
}

impl Interpreter {
    /// Fails with `ErrorKind::Usage` when the requested instruction set is not available.
    pub fn new(opts: Options) -> Result<Interpreter, Error> {
        let isa = match opts.isa {
            Some(isa) if !isa.is_available_with(opts.disable) => {
                return Err(Error::usage(format!(
                    "instruction set '{}' is not available on this CPU",
                    isa.name()
                )));
            }
            Some(isa) => isa,
            None => Isa::detect_with(opts.disable),
        };
        let reg = Registry::new();
        let planner = Planner::new(&reg, !opts.no_fusion);
        let dispatch = reg.resolve(isa, opts.backend, opts.disable);
        let fused_reductions = reg.iter().map(|_| AtomicU64::new(0)).collect();
        Ok(Interpreter {
            opts,
            dispatch,
            reg,
            planner,
            globals: Globals::new(),
            env: Env::new(),
            stats: Stats::default(),
            tiled_regions: AtomicU64::new(0),
            narrowed_tiles: AtomicU64::new(0),
            fused_reductions,
            shared_hits: AtomicU64::new(0),
        })
    }

    /// Runs a whole program. Every statement is checked and planned before any executes.
    /// `print` writes to `out`, plan and trace lines go to `diag`. Expression statements are no
    /// outputs: they are evaluated and dropped with their runtime errors. An `input`, or a read,
    /// of a name whose binding holds an error is a usage error.
    pub fn run(
        &mut self,
        src: &str,
        out: &mut dyn Write,
        diag: &mut dyn Write,
    ) -> Result<(), Error> {
        self.eval(src, Mode::Run, out, diag)
    }

    /// Like `run`, but lets stay bound and expression statements print their value to `out`.
    pub fn eval_repl(
        &mut self,
        src: &str,
        out: &mut dyn Write,
        diag: &mut dyn Write,
    ) -> Result<(), Error> {
        self.eval(src, Mode::Repl, out, diag)
    }

    fn eval(
        &mut self,
        src: &str,
        mode: Mode,
        out: &mut dyn Write,
        diag: &mut dyn Write,
    ) -> Result<(), Error> {
        // The checker declares each `input` as a fresh variable, so the host's binding is
        // looked up in the globals as they were before this program.
        let before = self.globals.clone();
        let mut globals = before.clone();
        let c = self.compile(&mut globals, src, mode)?;
        let mut aliases = Vec::new();
        for decl in &c.inputs {
            let slot = before
                .lookup(&decl.name)
                .and_then(|(var, _)| self.env.get(&var));
            let v = match slot {
                None => return Err(Error::usage(format!("input '{}' is not bound", decl.name))),
                Some(Err(e)) => {
                    return Err(Error::usage(format!(
                        "input '{}' is bound to a value that failed: {}",
                        decl.name, e.message
                    )));
                }
                Some(Ok(v)) => v,
            };
            if v.ty() != decl.ty {
                return Err(Error::usage(format!(
                    "input '{}' is declared {} but bound as {}",
                    decl.name,
                    decl.ty,
                    v.ty()
                )));
            }
            aliases.push((decl.var, Ok(v.clone())));
        }
        // Plan lines go out before anything is committed, so a failing sink changes nothing.
        for line in &c.plan_lines {
            writeln!(diag, "{line}").map_err(|e| Error::io(&e))?;
        }
        // Everything is checked, so the new names become visible now.
        let var_count = before.var_count();
        self.globals = globals;
        self.env.extend(aliases);
        // A binding of an earlier program that holds an error is not handed on as data. Its
        // slot holds a usage error while this program runs, so a statement that reads it
        // fails like an input of it, and a read in an untaken branch does not. Every error
        // in the environment comes from an earlier program because this one has not run.
        let mut failed = Vec::new();
        for (var, slot) in self.env.iter_mut() {
            if let Err(e) = slot {
                let usage = Error::usage(format!(
                    "'{}' is bound to a value that failed: {}",
                    before.name(*var),
                    e.message
                ));
                failed.push((*var, std::mem::replace(e, usage)));
            }
        }
        let mut sinks = Sinks {
            out,
            diag,
            print_to_diag: false,
            show: mode == Mode::Repl,
        };
        let obs = Observe {
            trace: self.opts.trace,
            stats: self.opts.stats.then_some(&mut self.stats),
            tiled_regions: &self.tiled_regions,
            narrowed_tiles: &self.narrowed_tiles,
            shared_hits: &self.shared_hits,
            fused_reductions: &self.fused_reductions,
        };
        let result = run_plan(
            &self.reg,
            &self.dispatch,
            &c.plan,
            &mut self.env,
            self.opts.strategy(),
            obs,
            &mut sinks,
        );
        // The original runtime errors go back, whether or not the run succeeded.
        for (var, e) in failed {
            self.env.insert(var, Err(e));
        }
        if result.is_err() {
            // A failed input exposes no new names and keeps old bindings. Variables created
            // by it are dropped from the environment because their ids will be reused.
            self.globals = before;
            self.env.retain(|var, _| var.0 < var_count);
        }
        result
    }

    /// Parses, checks, lowers, and plans `src` as a program with no prior bindings.
    fn compile(&self, globals: &mut Globals, src: &str, mode: Mode) -> Result<Compiled, Error> {
        let ast = parse(src)?;
        let typed = check(&ast, &self.reg, globals)?;
        check_unread(&ast, mode == Mode::Repl)?;
        let inputs = typed
            .stmts
            .iter()
            .filter_map(|s| match s {
                TStmt::Input { var, ty, .. } => Some(InputDecl {
                    name: globals.name(*var).to_string(),
                    var: *var,
                    ty: *ty,
                }),
                _ => None,
            })
            .collect();
        let lowered = lower(
            &typed,
            globals,
            &self.reg,
            mode,
            !self.opts.no_composite_sort,
        )?;
        // One context for the whole program, so a term repeated across statements is planned
        // once and its plan is one node of the plan graph.
        let mut cx = PlanContext::default();
        let mut roots = Vec::new();
        let mut plan_lines = Vec::new();
        let mut sum_rewrites = Vec::new();
        for stmt in &lowered.stmts {
            let span = stmt.span();
            let (root, fired) = self.planner.plan_with_sum_rewrites(
                &self.reg,
                &lowered.graph,
                stmt.term(),
                span,
                &mut cx,
            )?;
            if self.opts.show_plan {
                let text = render(&self.reg, &cx.graph, root, &lowered.leaves, globals);
                plan_lines.push(format!("plan {}: {text}", span.line));
            }
            roots.push(root);
            sum_rewrites.push(fired);
        }
        Ok(Compiled {
            plan: Plan::new(&self.reg, lowered, cx.graph, roots),
            inputs,
            plan_lines,
            sum_rewrites,
        })
    }

    /// Binds `name` to `value` at the value's type. A later `input` of the name refers to it.
    pub fn bind(&mut self, name: &str, value: Value) {
        let var = self.globals.declare(name, value.ty());
        self.env.insert(var, Ok(value));
    }

    /// The value bound to `name`. `None` when nothing is bound to it, or when its binding
    /// holds the runtime error that computing it raised.
    pub fn get(&self, name: &str) -> Option<Value> {
        let (var, _) = self.globals.lookup(name)?;
        self.env.get(&var)?.as_ref().ok().cloned()
    }

    /// Writes the per-kernel totals collected under `Options::stats`.
    pub fn write_stats(&self, diag: &mut dyn Write) {
        self.stats.write(diag);
    }

    /// The number of region runs that completed tiled, not through the whole fallback, since
    /// the interpreter was created. It counts runs of `run`, `eval_repl`, and
    /// `Prepared::execute` with this interpreter, whether or not statistics are collected.
    pub fn tiled_regions(&self) -> u64 {
        self.tiled_regions.load(Ordering::Relaxed)
    }

    /// The number of guard evaluations, one per `and` or `or` and tile, in which the right
    /// operand ran over fewer rows of the tile than the guard, see `Strategy::selective`,
    /// summed over the region runs that completed tiled since the interpreter was created. It
    /// counts runs of `run`, `eval_repl`, and `Prepared::execute` with this interpreter,
    /// whether or not statistics are collected, and stays 0 under `Options::no_selective`.
    pub fn narrowed_tiles(&self) -> u64 {
        self.narrowed_tiles.load(Ordering::Relaxed)
    }

    /// Per root kernel name, the number of fused region runs, reductions updated tile by tile
    /// with a chain of elementwise kernels, that completed tiled since the interpreter was
    /// created, see `Strategy::fuse_reductions`. Only kernels with runs appear. It counts runs
    /// of `run`, `eval_repl`, and `Prepared::execute` with this interpreter, whether or not
    /// statistics are collected, and each run counts in `tiled_regions` too.
    pub fn fused_reductions(&self) -> BTreeMap<String, u64> {
        self.reg
            .iter()
            .zip(&self.fused_reductions)
            .filter_map(|((_, def), count)| {
                let count = count.load(Ordering::Relaxed);
                (count > 0).then(|| (def.name.clone(), count))
            })
            .collect()
    }

    /// The number of evaluations of shared plan nodes that reused the value of an earlier
    /// evaluation in the same run instead of calling kernels, since the interpreter was
    /// created. It counts runs of `run`, `eval_repl`, and `Prepared::execute` with this
    /// interpreter, and stays 0 under `Options::no_sharing`.
    pub fn shared_hits(&self) -> u64 {
        self.shared_hits.load(Ordering::Relaxed)
    }

    /// Compiles `src` for repeated execution. Names must be declared by the program itself,
    /// and `input` declarations give the types of what `Prepared::execute` is supplied.
    pub fn prepare(&self, src: &str) -> Result<Prepared, Error> {
        self.prepare_to(src, &mut std::io::sink())
    }

    /// Like `prepare`, and with `show_plan` writes the plan lines to `diag`. The plan is
    /// written here and never by `Prepared::execute`.
    pub fn prepare_to(&self, src: &str, diag: &mut dyn Write) -> Result<Prepared, Error> {
        self.prepare_with_sum_rewrites(src, diag).map(|(p, _)| p)
    }

    /// Like `prepare_to`, and also returns per planned statement, in source order, the
    /// integer sum rewrites that matched while planning it, see
    /// `Planner::plan_with_sum_rewrites`. This is a test hook.
    pub fn prepare_with_sum_rewrites(
        &self,
        src: &str,
        diag: &mut dyn Write,
    ) -> Result<(Prepared, Vec<BTreeSet<SumRewrite>>), Error> {
        let mut globals = Globals::new();
        let c = self.compile(&mut globals, src, Mode::Run)?;
        for line in &c.plan_lines {
            writeln!(diag, "{line}").map_err(|e| Error::io(&e))?;
        }
        let leaf_texts = c
            .plan
            .lowered
            .leaves
            .iter()
            .map(|l| render_leaf(l, &globals))
            .collect();
        let prepared = Prepared {
            plan: c.plan,
            inputs: c.inputs,
            leaf_texts,
        };
        Ok((prepared, c.sum_rewrites))
    }
}

impl Prepared {
    /// The declared inputs as (name, type), in source order.
    pub fn inputs(&self) -> impl Iterator<Item = (&str, ValueType)> {
        self.inputs.iter().map(|d| (d.name.as_str(), d.ty))
    }

    /// Whether `other` has the statements of this program at the same lines, each with a plan
    /// that `render` prints as this one's, so that the plan lines of `prepare_to` with
    /// `show_plan` would be equal. It takes time linear in the sizes of the plan graphs,
    /// where rendering is exponential in the nesting of shared nodes. This is a test hook.
    pub fn same_plans(&self, other: &Prepared) -> bool {
        let (a, b) = (&self.plan, &other.plan);
        let lines = |p: &Plan| {
            p.lowered
                .stmts
                .iter()
                .map(|s| s.span().line)
                .collect::<Vec<_>>()
        };
        if lines(a) != lines(b) {
            return false;
        }
        let mut shapes = PlanShapes::default();
        let sa = shapes.number(&a.graph, &self.leaf_texts);
        let sb = shapes.number(&b.graph, &other.leaf_texts);
        a.roots
            .iter()
            .zip(&b.roots)
            .all(|(ra, rb)| sa[ra.index()] == sb[rb.index()])
    }

    /// Runs the program on `inputs` and returns its exports in source order. Every declared
    /// input must be supplied with its declared type. `print` writes to `diag`, as do trace
    /// lines when the interpreter traces. Statistics are not collected. An export is an output
    /// at its statement, so a runtime error in it fails the program there, and no later
    /// `print` runs.
    pub fn execute(
        &self,
        interp: &Interpreter,
        inputs: &[(&str, Value)],
        diag: &mut dyn Write,
    ) -> Result<Vec<(String, Value)>, Error> {
        for (i, (name, _)) in inputs.iter().enumerate() {
            if !self.inputs.iter().any(|d| d.name == *name) {
                return Err(Error::usage(format!(
                    "program does not declare input '{name}'"
                )));
            }
            if inputs[..i].iter().any(|(n, _)| n == name) {
                return Err(Error::usage(format!("input '{name}' is supplied twice")));
            }
        }
        let mut env = Env::new();
        for decl in &self.inputs {
            let Some((_, v)) = inputs.iter().find(|(n, _)| *n == decl.name) else {
                return Err(Error::usage(format!(
                    "input '{}' is not supplied",
                    decl.name
                )));
            };
            if v.ty() != decl.ty {
                return Err(Error::usage(format!(
                    "input '{}' is declared {} but supplied as {}",
                    decl.name,
                    decl.ty,
                    v.ty()
                )));
            }
            env.insert(decl.var, Ok(v.clone()));
        }
        let mut sink = std::io::sink();
        let mut sinks = Sinks {
            out: &mut sink,
            diag,
            print_to_diag: true,
            show: false,
        };
        let obs = Observe {
            trace: interp.opts.trace,
            stats: None,
            tiled_regions: &interp.tiled_regions,
            narrowed_tiles: &interp.narrowed_tiles,
            shared_hits: &interp.shared_hits,
            fused_reductions: &interp.fused_reductions,
        };
        run_plan(
            &interp.reg,
            &interp.dispatch,
            &self.plan,
            &mut env,
            interp.opts.strategy(),
            obs,
            &mut sinks,
        )?;
        self.plan
            .lowered
            .exports
            .iter()
            .map(|(name, var)| {
                // `run_plan` raised the error of a failed export at its statement already.
                let v = env.remove(var).ok_or_else(|| {
                    Error::runtime(format!("export '{name}' has no value"), None)
                })??;
                Ok((name.clone(), v))
            })
            .collect()
    }
}
