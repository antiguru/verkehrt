//! The library API: an interpreter with persistent bindings, and compile-once programs for
//! embedding in a host.

use std::collections::BTreeSet;
use std::io::Write;

use crate::check::{Globals, TStmt, VarId, check, check_unread};
use crate::error::Error;
use crate::exec::{Env, Observe, Plan, Sinks, Stats, run_plan};
use crate::kernels::{Backend, Dispatch, FeatureSet, Isa, Registry};
use crate::lower::{LStmt, Mode, lower};
use crate::plan::{Planner, SumRewrite, render};
use crate::syntax::parse;
use crate::value::{Value, ValueType};

/// Interpreter settings. The default has everything off and detects the instruction set.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// The instruction set to run kernels with, `None` detects the best one.
    pub isa: Option<Isa>,
    /// How the instruction set's kernels are written, see `Registry::resolve`.
    pub backend: Backend,
    /// Print one plan line per statement to the diagnostic stream before execution.
    pub show_plan: bool,
    /// Print one line per kernel call to the diagnostic stream.
    pub trace: bool,
    /// Collect per-kernel totals, written by `Interpreter::write_stats`.
    pub stats: bool,
    /// CPU features to treat as absent, so no kernel entry that needs one is selected and
    /// an instruction set whose level needs one is unavailable.
    pub disable: FeatureSet,
    /// Plan without fused kernel patterns and without the integer sum rewrites, so every plan
    /// uses single-op kernels only and keeps each sum over the array the program sums.
    pub no_fusion: bool,
}

pub struct Interpreter {
    opts: Options,
    dispatch: Dispatch,
    reg: Registry,
    planner: Planner,
    globals: Globals,
    env: Env,
    stats: Stats,
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
        Ok(Interpreter {
            opts,
            dispatch,
            reg,
            planner,
            globals: Globals::new(),
            env: Env::new(),
            stats: Stats::default(),
        })
    }

    /// Runs a whole program. Every statement is checked and planned before any executes.
    /// `print` writes to `out`, plan and trace lines go to `diag`.
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
                .and_then(|(var, ty)| self.env.get(&var).map(|slot| (ty, slot)));
            let Some((ty, slot)) = slot else {
                return Err(Error::usage(format!("input '{}' is not bound", decl.name)));
            };
            // A binding that holds an error has no value to take the type from, so its
            // declared type stands in.
            let bound_ty = match slot {
                Ok(v) => v.ty(),
                Err(_) => ty,
            };
            if bound_ty != decl.ty {
                return Err(Error::usage(format!(
                    "input '{}' is declared {} but bound as {bound_ty}",
                    decl.name, decl.ty,
                )));
            }
            // An input of a failed binding holds its error, which an output reading the
            // input raises.
            aliases.push((decl.var, slot.clone()));
        }
        // Plan lines go out before anything is committed, so a failing sink changes nothing.
        for line in &c.plan_lines {
            writeln!(diag, "{line}").map_err(|e| Error::io(&e))?;
        }
        // Everything is checked, so the new names become visible now.
        let var_count = before.var_count();
        self.globals = globals;
        self.env.extend(aliases);
        let mut sinks = Sinks {
            out,
            diag,
            print_to_diag: false,
            show: mode == Mode::Repl,
        };
        let obs = Observe {
            trace: self.opts.trace,
            stats: self.opts.stats.then_some(&mut self.stats),
        };
        let result = run_plan(
            &self.reg,
            &self.dispatch,
            &c.plan,
            &mut self.env,
            obs,
            &mut sinks,
        );
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
        let lowered = lower(&typed, globals, &self.reg, mode)?;
        let mut nodes = Vec::new();
        let mut plan_lines = Vec::new();
        let mut sum_rewrites = Vec::new();
        for stmt in &lowered.stmts {
            let (LStmt::Bind { term, .. } | LStmt::Print { term, .. } | LStmt::Show { term, .. }) =
                stmt;
            let span = stmt.span();
            let (node, fired) = self.planner.plan_with_sum_rewrites(&self.reg, term)?;
            if self.opts.show_plan {
                let text = render(&self.reg, &node, &lowered.leaves, globals);
                plan_lines.push(format!("plan {}: {text}", span.line));
            }
            nodes.push(node);
            sum_rewrites.push(fired);
        }
        Ok(Compiled {
            plan: Plan { lowered, nodes },
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
        let c = self.compile(&mut Globals::new(), src, Mode::Run)?;
        for line in &c.plan_lines {
            writeln!(diag, "{line}").map_err(|e| Error::io(&e))?;
        }
        let prepared = Prepared {
            plan: c.plan,
            inputs: c.inputs,
        };
        Ok((prepared, c.sum_rewrites))
    }
}

impl Prepared {
    /// The declared inputs as (name, type), in source order.
    pub fn inputs(&self) -> impl Iterator<Item = (&str, ValueType)> {
        self.inputs.iter().map(|d| (d.name.as_str(), d.ty))
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
        };
        run_plan(
            &interp.reg,
            &interp.dispatch,
            &self.plan,
            &mut env,
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
