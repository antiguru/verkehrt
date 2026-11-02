//! The reference evaluator of oracle 3: a tree-walking interpreter of the typed program
//! (`check::TProgram`) that works one scalar at a time.
//!
//! It shares no code with the pipeline. It reads the program through `syntax::parse` and
//! `check::check`, and it uses the plain value types (`Scalar`, `Value`, `Array`) for inputs
//! and results, but no kernel, lowering, lifting, hoisting, planning, or execution code.
//! Every primitive is written here from the documented semantics.
//!
//! # Semantics
//!
//! * **Demand.** The language computes only values that reach a `print` or an export. Lets,
//!   block lets, and function arguments are therefore thunks, evaluated when first read and
//!   then shared. A value that nothing reads never runs, so its errors are not raised, and an
//!   expression statement is not evaluated either.
//! * **Conditionals.** An `if` on a scalar `bool` evaluates the condition and then only the
//!   branch it picks. Inside a lambda that holds for a uniform condition, one that does not
//!   depend on a lambda parameter. An `if` on a `bool[]`, and an `if` inside a lambda whose
//!   condition depends on a parameter, computes both branches and selects per element.
//! * **Lambdas** apply per element, with each parameter bound to one element's scalar and
//!   captured scalars read from the enclosing scope. The base spec defines a lambda by lifting
//!   it over whole arrays, so a part of the body that does not depend on a parameter runs once
//!   even when the arrays are empty. Over empty arrays the body therefore runs once with
//!   phantom parameters (`Val::Phantom`), which raises the errors of its uniform parts, such
//!   as `div_const` by a zero divisor.
//! * **Functions** are called directly with lazy arguments.
//! * **Arithmetic** wraps on integers, follows IEEE 754 on floats with `min` and `max` as
//!   `minimum` and `maximum`, and casts follow Rust `as`, see `scalar`.
//! * **Reductions** widen `add` on narrow integers and fold floats in the canonical order of
//!   the base spec, see `prims::canonical`. Float `min` and `max` give the same bits in any
//!   order, apart from the sign and payload of NaN, which the comparison ignores.
//!
//! # Errors
//!
//! Whether an error occurs is fixed, and so is the output that raises it: the first print or
//! export whose evaluation fails. Which of several errors within that output surfaces may
//! differ between plans. The evaluator therefore evaluates every operand of an operation even
//! after one fails, runs a lambda body with phantom parameters even when its arrays failed,
//! and collects the category (see `oracles::Category`) of every error it meets in the failing
//! output. It stops after that output.
//!
//! The line an error reports is not fixed. The span rule of the errors-as-data spec
//! (`docs/superpowers/specs/2026-11-02-verkehrt-errors-as-data-design.md`) gives an error the
//! span of the statement that computed it, and lowering decides which statements compute
//! separately: a binding read by a later statement reports its own line, and a single-use let
//! substituted into its reader reports the reader's line. The evaluator stays lazy and does
//! not mirror lowering. Instead each collected error carries its chain, the line of every
//! statement from the demanding output to the failing expression: the output itself and each
//! top-level let or export whose thunk was forced on the way. Block lets and arguments add no
//! line, because they belong to their statement. The pipeline agrees when it fails at a line
//! on the chain of a collected error of its category, and its printed output equals the
//! evaluator's, because the prints before the failing output appear in full.
//!
//! # Unsupported
//!
//! Every primitive that the generator produces is supported, so generation for oracle 3
//! excludes nothing. Strings, `load_*`, and arrays longer than `prims::MAX_ROWS` make a case
//! unsupported, which fails it, because the generator produces none of them and a silent skip
//! would hide a gap in the oracle.

mod prims;
mod scalar;

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::fmt::Write as _;
use std::rc::Rc;

use verkehrt::check::{self, Globals, TExpr, TExprKind, TFn, TLambda, TStmt, VarId};
use verkehrt::kernels::Registry;
use verkehrt::ops::Op;
use verkehrt::syntax::parse;
use verkehrt::value::{Array, Scalar, ScalarType, Value, ValueType};

use crate::oracles::Category;
use prims::elem;

/// An evaluated value.
#[derive(Clone, Debug)]
enum Val {
    Scalar(Scalar),
    Array(ScalarType, Rc<Vec<Scalar>>),
    /// A lambda parameter over empty arrays, or a value computed from one. It has no value,
    /// only the errors of the computation around it matter.
    Phantom,
}

/// An error that may occur, with the lines of the statements it may be reported at.
#[derive(Clone, Debug)]
pub struct Possible {
    pub category: Category,
    pub message: String,
    /// The statements on the chain from the demanding output to the failing expression: the
    /// output and each top-level let or export whose thunk was forced on the way.
    pub lines: BTreeSet<u32>,
}

/// Why evaluation stopped.
#[derive(Clone, Debug)]
enum Stop {
    /// Errors that may occur, at most one per category. Errors of one category met on
    /// several chains keep the first message and the lines of every chain.
    Errors(Vec<Possible>),
    /// The case uses something the evaluator does not support.
    Unsupported(String),
}

impl Stop {
    fn error(message: String) -> Stop {
        Stop::Errors(vec![Possible {
            category: Category::of(&message),
            message,
            lines: BTreeSet::new(),
        }])
    }

    fn merge(self, other: Stop) -> Stop {
        match (self, other) {
            // A real error is a result the pipeline must match, so it wins over a part the
            // evaluator does not support.
            (Stop::Errors(a), Stop::Unsupported(_)) | (Stop::Unsupported(_), Stop::Errors(a)) => {
                Stop::Errors(a)
            }
            (Stop::Unsupported(r), Stop::Unsupported(_)) => Stop::Unsupported(r),
            (Stop::Errors(mut a), Stop::Errors(b)) => {
                for e in b {
                    match a.iter_mut().find(|x| x.category == e.category) {
                        Some(x) => x.lines.extend(e.lines),
                        None => a.push(e),
                    }
                }
                Stop::Errors(a)
            }
        }
    }

    /// Adds the statement on `line` to the chain of every error.
    fn through(mut self, line: u32) -> Stop {
        if let Stop::Errors(es) = &mut self {
            for e in es {
                e.lines.insert(line);
            }
        }
        self
    }
}

fn merge_into(acc: &mut Option<Stop>, e: Stop) {
    *acc = Some(match acc.take() {
        None => e,
        Some(a) => a.merge(e),
    });
}

/// Inside a lambda body, the variables whose values depend on a lambda parameter: the
/// parameters, block lets computed from them, and the parameters of a function called with
/// such arguments. `None` outside lambdas.
type Lambda = Option<Rc<HashSet<VarId>>>;

/// Whether `e` depends on one of the variables `dep`.
fn depends(g: &Globals, e: &TExpr, dep: &HashSet<VarId>) -> bool {
    match &e.kind {
        TExprKind::Lit(_) | TExprKind::Str(_) => false,
        TExprKind::Var(v) => dep.contains(v),
        TExprKind::ArrayLit(es) | TExprKind::Prim(_, es) => es.iter().any(|x| depends(g, x, dep)),
        // A call depends only when its body reads a parameter whose argument depends, as
        // lifting types a call by its body.
        TExprKind::Call(id, args) => {
            let f = g.func(*id);
            depends(g, &f.body, &dependent_params(g, f, args, dep))
        }
        TExprKind::Map(x, lam) | TExprKind::Filter(x, lam) => {
            depends(g, x, dep) || depends(g, &lam.body, dep)
        }
        TExprKind::ZipMap(x, y, lam) => {
            depends(g, x, dep) || depends(g, y, dep) || depends(g, &lam.body, dep)
        }
        TExprKind::If(c, t, f) => [c, t, f].into_iter().any(|x| depends(g, x, dep)),
        TExprKind::Block(lets, result) => {
            let mut dep = dep.clone();
            for (v, x) in lets {
                if depends(g, x, &dep) {
                    dep.insert(*v);
                }
            }
            depends(g, result, &dep)
        }
    }
}

/// The parameters of `f` whose arguments `args` depend on one of the variables `dep`.
fn dependent_params(g: &Globals, f: &TFn, args: &[TExpr], dep: &HashSet<VarId>) -> HashSet<VarId> {
    f.params
        .iter()
        .zip(args)
        .filter(|(_, a)| depends(g, a, dep))
        .map(|((p, _), _)| *p)
        .collect()
}

/// The state of a lazily evaluated binding.
enum Slot<'p> {
    /// The expression, its scope, and its lambda context.
    Pending(&'p TExpr, Env<'p>, Lambda),
    Forcing,
    Done(Result<Val, Stop>),
}

/// A lazily evaluated binding, and the line of its statement if it is a top-level let or an
/// export, which forcing it adds to the chain of its errors.
struct Thunk<'p>(RefCell<Slot<'p>>, Option<u32>);

impl<'p> Thunk<'p> {
    fn done(v: Val) -> Rc<Thunk<'p>> {
        Rc::new(Thunk(RefCell::new(Slot::Done(Ok(v))), None))
    }

    fn pending(e: &'p TExpr, env: Env<'p>, lambda: Lambda) -> Rc<Thunk<'p>> {
        Rc::new(Thunk(RefCell::new(Slot::Pending(e, env, lambda)), None))
    }

    /// A top-level let or export on `line`.
    fn statement(e: &'p TExpr, env: Env<'p>, line: u32) -> Rc<Thunk<'p>> {
        Rc::new(Thunk(RefCell::new(Slot::Pending(e, env, None)), Some(line)))
    }
}

/// A scope: the bindings visible at a point, innermost first.
#[derive(Clone, Default)]
struct Env<'p>(Option<Rc<Frame<'p>>>);

struct Frame<'p> {
    var: VarId,
    value: Rc<Thunk<'p>>,
    next: Env<'p>,
}

impl<'p> Env<'p> {
    fn bind(&self, var: VarId, value: Rc<Thunk<'p>>) -> Env<'p> {
        Env(Some(Rc::new(Frame {
            var,
            value,
            next: self.clone(),
        })))
    }

    fn get(&self, var: VarId) -> Rc<Thunk<'p>> {
        let mut at = &self.0;
        while let Some(frame) = at {
            if frame.var == var {
                return Rc::clone(&frame.value);
            }
            at = &frame.next.0;
        }
        panic!("variable {var:?} is not in scope")
    }
}

struct Evaluator<'p> {
    globals: &'p Globals,
}

impl<'p> Evaluator<'p> {
    fn force(&self, t: &Thunk<'p>) -> Result<Val, Stop> {
        let slot = {
            let mut slot = t.0.borrow_mut();
            match &*slot {
                Slot::Done(r) => return r.clone(),
                Slot::Forcing => panic!("a binding reads itself"),
                Slot::Pending(..) => std::mem::replace(&mut *slot, Slot::Forcing),
            }
        };
        let Slot::Pending(e, env, lambda) = slot else {
            unreachable!("matched above")
        };
        let r = self.eval(e, &env, &lambda);
        let r = match t.1 {
            Some(line) => r.map_err(|s| s.through(line)),
            None => r,
        };
        *t.0.borrow_mut() = Slot::Done(r.clone());
        r
    }

    /// Evaluates every expression, also after one fails, so that all their errors count.
    fn all(
        &self,
        es: impl IntoIterator<Item = &'p TExpr>,
        env: &Env<'p>,
        lambda: &Lambda,
    ) -> Result<Vec<Val>, Stop> {
        let mut vals = Vec::new();
        let mut stop = None;
        for e in es {
            match self.eval(e, env, lambda) {
                Ok(v) => vals.push(v),
                Err(s) => merge_into(&mut stop, s),
            }
        }
        match stop {
            None => Ok(vals),
            Some(s) => Err(s),
        }
    }

    fn eval(&self, e: &'p TExpr, env: &Env<'p>, lambda: &Lambda) -> Result<Val, Stop> {
        match &e.kind {
            TExprKind::Lit(s) => Ok(Val::Scalar(s.clone())),
            TExprKind::Str(_) => Err(Stop::Unsupported("a string".to_string())),
            TExprKind::ArrayLit(items) => {
                let rows = self
                    .all(items, env, lambda)?
                    .into_iter()
                    .map(|v| match v {
                        Val::Scalar(s) => s,
                        other => panic!("array literal element {other:?}"),
                    })
                    .collect();
                Ok(Val::Array(elem(e.ty), Rc::new(rows)))
            }
            TExprKind::Var(v) => self.force(&env.get(*v)),
            TExprKind::Prim(op, args) => {
                let vals = self.all(args, env, lambda)?;
                prims::apply(*op, &vals, e.ty)
            }
            TExprKind::Call(id, args) => {
                let f = self.globals.func(*id);
                let mut inner = Env::default();
                for ((p, _), a) in f.params.iter().zip(args) {
                    inner = inner.bind(*p, Thunk::pending(a, env.clone(), lambda.clone()));
                }
                let body_lambda = lambda
                    .as_ref()
                    .map(|dep| Rc::new(dependent_params(self.globals, f, args, dep)));
                self.eval(&f.body, &inner, &body_lambda)
            }
            TExprKind::Map(x, lam) => {
                let x = self
                    .eval(x, env, lambda)
                    .map_err(|s| self.with_body(s, lam, env))?;
                let rows = self.apply(lam, &[array_rows(&x)], env)?;
                Ok(Val::Array(elem(e.ty), Rc::new(rows)))
            }
            TExprKind::ZipMap(x, y, lam) => {
                let vals = self
                    .all([&**x, &**y], env, lambda)
                    .map_err(|s| self.with_body(s, lam, env))?;
                let (xs, ys) = (array_rows(&vals[0]), array_rows(&vals[1]));
                if xs.len() != ys.len() {
                    let mismatch =
                        Stop::error(format!("length mismatch: {} vs {}", xs.len(), ys.len()));
                    return Err(self.with_body(mismatch, lam, env));
                }
                let rows = self.apply(lam, &[xs, ys], env)?;
                Ok(Val::Array(elem(e.ty), Rc::new(rows)))
            }
            TExprKind::Filter(x, lam) => {
                let x = self
                    .eval(x, env, lambda)
                    .map_err(|s| self.with_body(s, lam, env))?;
                let xs = array_rows(&x);
                let mask = self.apply(lam, &[xs], env)?;
                let kept = xs
                    .iter()
                    .zip(mask)
                    .filter(|(_, m)| scalar::bool_of(m))
                    .map(|(v, _)| v.clone())
                    .collect();
                Ok(Val::Array(elem(e.ty), Rc::new(kept)))
            }
            TExprKind::If(c, t, f) => {
                let lazy = match lambda {
                    None => c.ty == ValueType::Scalar(ScalarType::Bool),
                    Some(dep) => !depends(self.globals, c, dep),
                };
                if lazy {
                    let taken = match self.eval(c, env, lambda)? {
                        Val::Scalar(b) => scalar::bool_of(&b),
                        other => panic!("if condition {other:?}"),
                    };
                    self.eval(if taken { t } else { f }, env, lambda)
                } else {
                    let vals = self.all([&**c, &**t, &**f], env, lambda)?;
                    prims::apply(Op::Select, &vals, e.ty)
                }
            }
            TExprKind::Block(lets, result) => {
                let mut inner = env.clone();
                let mut inner_lambda = lambda.clone();
                for (v, x) in lets {
                    inner = inner.bind(*v, Thunk::pending(x, inner.clone(), inner_lambda.clone()));
                    if let Some(dep) = &inner_lambda
                        && depends(self.globals, x, dep)
                    {
                        let mut more = HashSet::clone(dep);
                        more.insert(*v);
                        inner_lambda = Some(Rc::new(more));
                    }
                }
                self.eval(result, &inner, &inner_lambda)
            }
        }
    }

    /// The lambda body once with phantom parameters, which raises the errors of its uniform
    /// parts, see the module doc.
    fn phantom(&self, lam: &'p TLambda, env: &Env<'p>) -> Result<(), Stop> {
        let mut inner = env.clone();
        for p in &lam.params {
            inner = inner.bind(*p, Thunk::done(Val::Phantom));
        }
        self.eval(&lam.body, &inner, &params_of(lam)).map(|_| ())
    }

    /// `stop`, which keeps the lambda's arrays from existing, with the errors of the lambda
    /// body's uniform parts, which lifting computes regardless.
    fn with_body(&self, stop: Stop, lam: &'p TLambda, env: &Env<'p>) -> Stop {
        match self.phantom(lam, env) {
            Ok(()) => stop,
            Err(body) => stop.merge(body),
        }
    }

    /// The lambda's body for each row of the equally long columns, with the parameters bound
    /// to the row's elements.
    fn apply(
        &self,
        lam: &'p TLambda,
        cols: &[&[Scalar]],
        env: &Env<'p>,
    ) -> Result<Vec<Scalar>, Stop> {
        let rows = cols[0].len();
        if rows == 0 {
            // The lifted body runs once over the empty arrays, see the module doc.
            return self.phantom(lam, env).map(|()| Vec::new());
        }
        let mut out = Vec::with_capacity(rows);
        let mut stop = None;
        let lambda = params_of(lam);
        for i in 0..rows {
            let mut inner = env.clone();
            for (p, col) in lam.params.iter().zip(cols) {
                inner = inner.bind(*p, Thunk::done(Val::Scalar(col[i].clone())));
            }
            match self.eval(&lam.body, &inner, &lambda) {
                Ok(Val::Scalar(s)) => out.push(s),
                Ok(other) => panic!("lambda body gave {other:?}"),
                Err(s) => merge_into(&mut stop, s),
            }
        }
        match stop {
            None => Ok(out),
            Some(s) => Err(s),
        }
    }
}

/// The lambda context of a lambda's body, in which its parameters are the dependent variables.
fn params_of(lam: &TLambda) -> Lambda {
    Some(Rc::new(lam.params.iter().copied().collect()))
}

fn array_rows(v: &Val) -> &[Scalar] {
    match v {
        Val::Array(_, rows) => rows,
        other => panic!("expected an array, got {other:?}"),
    }
}

fn from_value(v: &Value) -> Val {
    match v {
        Value::Scalar(s) => Val::Scalar(s.clone()),
        Value::Array(a) => Val::Array(a.elem(), Rc::new((0..a.len()).map(|i| a.get(i)).collect())),
        Value::Str(_) => panic!("no string inputs are generated"),
    }
}

fn to_value(v: &Val) -> Value {
    macro_rules! build {
        ($rows:expr, $variant:ident) => {
            Array::from_vec(
                $rows
                    .iter()
                    .map(|s| match s {
                        Scalar::$variant(x) => *x,
                        other => panic!("{other:?} in a {} array", stringify!($variant)),
                    })
                    .collect(),
            )
        };
    }
    match v {
        Val::Scalar(s) => Value::Scalar(s.clone()),
        Val::Array(t, rows) => Value::Array(match t {
            ScalarType::I8 => build!(rows, I8),
            ScalarType::I16 => build!(rows, I16),
            ScalarType::I32 => build!(rows, I32),
            ScalarType::I64 => build!(rows, I64),
            ScalarType::U8 => build!(rows, U8),
            ScalarType::U16 => build!(rows, U16),
            ScalarType::U32 => build!(rows, U32),
            ScalarType::U64 => build!(rows, U64),
            ScalarType::F32 => build!(rows, F32),
            ScalarType::F64 => build!(rows, F64),
            ScalarType::Bool => {
                let bools: Vec<bool> = rows.iter().map(scalar::bool_of).collect();
                Array::from_bools(&bools)
            }
        }),
        Val::Phantom => panic!("a phantom value escaped its lambda"),
    }
}

/// The first statement whose evaluation failed.
pub struct Failed {
    /// The line of the statement's span.
    pub line: u32,
    /// Every error that may surface in the statement, one per category, each with the lines
    /// of its chain, which include `line`.
    pub errors: Vec<Possible>,
}

/// What the evaluator computed for a program.
pub struct Reference {
    /// The output of the statements before the first failing one.
    pub printed: String,
    /// The exports in source order, or the failing statement.
    pub result: Result<Vec<(String, Value)>, Failed>,
}

/// The typed program of `source`, checked with `reg`.
fn typed(reg: &Registry, source: &str) -> Result<(check::TProgram, Globals), String> {
    let ast = parse(source).map_err(|e| format!("does not parse: {e}"))?;
    let mut globals = Globals::new();
    let program =
        check::check(&ast, reg, &mut globals).map_err(|e| format!("does not check: {e}"))?;
    Ok((program, globals))
}

/// The input `var` from `inputs`.
fn input<'p>(globals: &Globals, inputs: &[(String, Value)], var: VarId) -> Rc<Thunk<'p>> {
    let name = globals.name(var);
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("input {name} is not supplied"));
    Thunk::done(from_value(value))
}

/// Evaluates `source` on `inputs`. Fails with the reason when the program uses something the
/// evaluator does not support. `reg` only serves type checking.
pub fn evaluate(
    reg: &Registry,
    source: &str,
    inputs: &[(String, Value)],
) -> Result<Reference, String> {
    let (program, globals) = typed(reg, source)?;
    let ev = Evaluator { globals: &globals };
    let mut env = Env::default();
    let mut printed = String::new();
    let mut exports = Vec::new();
    for s in &program.stmts {
        let line = stmt_line(s);
        let outcome = match s {
            TStmt::Input { var, .. } => {
                env = env.bind(*var, input(&globals, inputs, *var));
                Ok(())
            }
            TStmt::Let { var, value, .. } => {
                env = env.bind(*var, Thunk::statement(value, env.clone(), line));
                Ok(())
            }
            TStmt::Export { var, value, .. } => {
                let t = Thunk::statement(value, env.clone(), line);
                env = env.bind(*var, Rc::clone(&t));
                ev.force(&t)
                    .map(|v| exports.push((globals.name(*var).to_string(), to_value(&v))))
            }
            TStmt::Print { value, .. } => ev
                .eval(value, &env, &None)
                .map(|v| {
                    writeln!(printed, "{}", to_value(&v)).expect("writing to a string");
                })
                .map_err(|stop| stop.through(line)),
            // Its value reaches no print or export.
            TStmt::Expr { .. } => Ok(()),
        };
        match outcome {
            Ok(()) => {}
            Err(Stop::Unsupported(reason)) => return Err(reason),
            Err(Stop::Errors(errors)) => {
                return Ok(Reference {
                    printed,
                    result: Err(Failed { line, errors }),
                });
            }
        }
    }
    Ok(Reference {
        printed,
        result: Ok(exports),
    })
}

/// Whether forcing the binding named `name` alone fails: its value evaluated in its scope, with
/// the bindings before it lazy and no output evaluated. `None` when no top-level let and no
/// block let outside lambdas and function bodies has that name. Fails with the reason when
/// the binding uses something the evaluator does not support.
pub fn fails_alone(
    reg: &Registry,
    source: &str,
    inputs: &[(String, Value)],
    name: &str,
) -> Result<Option<bool>, String> {
    let (program, globals) = typed(reg, source)?;
    let ev = Evaluator { globals: &globals };
    let mut env = Env::default();
    for s in &program.stmts {
        let found = match s {
            TStmt::Input { var, .. } => {
                env = env.bind(*var, input(&globals, inputs, *var));
                None
            }
            TStmt::Let { var, value, .. } | TStmt::Export { var, value, .. } => {
                let found = if globals.name(*var) == name {
                    Some(ev.eval(value, &env, &None))
                } else {
                    ev.find(value, &env, name)
                };
                env = env.bind(*var, Thunk::pending(value, env.clone(), None));
                found
            }
            TStmt::Print { value, .. } | TStmt::Expr { value, .. } => ev.find(value, &env, name),
        };
        match found {
            None => {}
            Some(Ok(_)) => return Ok(Some(false)),
            Some(Err(Stop::Errors(_))) => return Ok(Some(true)),
            Some(Err(Stop::Unsupported(reason))) => return Err(reason),
        }
    }
    Ok(None)
}

impl<'p> Evaluator<'p> {
    /// The value of the block let named `name` in `e`, evaluated in its scope, or `None` when
    /// `e` has no such let outside lambdas and function bodies.
    fn find(&self, e: &'p TExpr, env: &Env<'p>, name: &str) -> Option<Result<Val, Stop>> {
        let first = |es: &[&'p TExpr]| es.iter().find_map(|x| self.find(x, env, name));
        match &e.kind {
            TExprKind::Lit(_) | TExprKind::Str(_) | TExprKind::Var(_) => None,
            TExprKind::ArrayLit(es) | TExprKind::Prim(_, es) | TExprKind::Call(_, es) => {
                es.iter().find_map(|x| self.find(x, env, name))
            }
            // A lambda body reads its parameters, which a binding alone does not have.
            TExprKind::Map(x, _) | TExprKind::Filter(x, _) => first(&[x]),
            TExprKind::ZipMap(x, y, _) => first(&[x, y]),
            TExprKind::If(c, t, f) => first(&[c, t, f]),
            TExprKind::Block(lets, result) => {
                let mut inner = env.clone();
                for (v, x) in lets {
                    if self.globals.name(*v) == name {
                        return Some(self.eval(x, &inner, &None));
                    }
                    if let Some(r) = self.find(x, &inner, name) {
                        return Some(r);
                    }
                    inner = inner.bind(*v, Thunk::pending(x, inner.clone(), None));
                }
                self.find(result, &inner, name)
            }
        }
    }
}

fn stmt_line(s: &TStmt) -> u32 {
    let (TStmt::Let { span, .. }
    | TStmt::Input { span, .. }
    | TStmt::Export { span, .. }
    | TStmt::Print { span, .. }
    | TStmt::Expr { span, .. }) = s;
    span.line
}

#[cfg(test)]
mod tests {
    use verkehrt::kernels::Registry;

    use super::evaluate;

    /// A call whose body ignores its parameter is uniform even with a dependent argument, so
    /// the lambda `if` on it is lazy and the branch not taken never divides by zero.
    #[test]
    fn call_ignoring_a_dependent_argument_is_uniform() {
        let src = "fn f(q: i64) -> bool { true }\n\
                   let x = [1, 2, 3];\n\
                   print(map(x, |v| if f(v) { v } else { div_const(v, 0) }));\n";
        let r = evaluate(&Registry::new(), src, &[]).expect("supported");
        assert!(r.result.is_ok(), "the program fails");
        assert_eq!(r.printed, "[1, 2, 3]\n");
    }
}
