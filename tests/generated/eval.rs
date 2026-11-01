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
//! Whether an error occurs is fixed, and so is the statement that raises it: the first
//! statement whose evaluation fails. Which of several errors within that statement surfaces
//! may differ between plans. The evaluator therefore evaluates every operand of an operation
//! even after one fails, runs a lambda body with phantom parameters even when its arrays
//! failed, and collects the category (see `oracles::Category`) of every error it meets in the
//! failing statement. It stops after that statement. The pipeline agrees when it fails at the
//! same line with one of the collected categories, and its output is a prefix of the
//! evaluator's, because output printed before an error may be incomplete.
//!
//! # Unsupported
//!
//! Every primitive that the generator produces is supported, so generation for oracle 3
//! excludes nothing. Strings, `load_*`, and arrays longer than `prims::MAX_ROWS` make a case
//! unsupported, and oracle 3 skips it. The generator produces none of them.

mod prims;
mod scalar;

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::rc::Rc;

use verkehrt::check::{self, Globals, TExpr, TExprKind, TLambda, TStmt, VarId};
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

/// Why evaluation stopped.
#[derive(Clone, Debug)]
enum Stop {
    /// Errors that may occur, with their messages, at most one per category.
    Errors(Vec<(Category, String)>),
    /// The case uses something the evaluator does not support.
    Unsupported(String),
}

impl Stop {
    fn error(message: String) -> Stop {
        Stop::Errors(vec![(Category::of(&message), message)])
    }

    fn merge(self, other: Stop) -> Stop {
        match (self, other) {
            (Stop::Unsupported(r), _) | (_, Stop::Unsupported(r)) => Stop::Unsupported(r),
            (Stop::Errors(mut a), Stop::Errors(b)) => {
                for e in b {
                    if !a.iter().any(|(c, _)| *c == e.0) {
                        a.push(e);
                    }
                }
                Stop::Errors(a)
            }
        }
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
fn depends(e: &TExpr, dep: &HashSet<VarId>) -> bool {
    match &e.kind {
        TExprKind::Lit(_) | TExprKind::Str(_) => false,
        TExprKind::Var(v) => dep.contains(v),
        // A function body sees only its parameters, so a call depends through its arguments.
        TExprKind::ArrayLit(es) | TExprKind::Prim(_, es) | TExprKind::Call(_, es) => {
            es.iter().any(|x| depends(x, dep))
        }
        TExprKind::Map(x, lam) | TExprKind::Filter(x, lam) => {
            depends(x, dep) || depends(&lam.body, dep)
        }
        TExprKind::ZipMap(x, y, lam) => {
            depends(x, dep) || depends(y, dep) || depends(&lam.body, dep)
        }
        TExprKind::If(c, t, f) => [c, t, f].into_iter().any(|x| depends(x, dep)),
        TExprKind::Block(lets, result) => {
            let mut dep = dep.clone();
            for (v, x) in lets {
                if depends(x, &dep) {
                    dep.insert(*v);
                }
            }
            depends(result, &dep)
        }
    }
}

/// The state of a lazily evaluated binding.
enum Slot<'p> {
    /// The expression, its scope, and its lambda context.
    Pending(&'p TExpr, Env<'p>, Lambda),
    Forcing,
    Done(Result<Val, Stop>),
}

struct Thunk<'p>(RefCell<Slot<'p>>);

impl<'p> Thunk<'p> {
    fn done(v: Val) -> Rc<Thunk<'p>> {
        Rc::new(Thunk(RefCell::new(Slot::Done(Ok(v)))))
    }

    fn pending(e: &'p TExpr, env: Env<'p>, lambda: Lambda) -> Rc<Thunk<'p>> {
        Rc::new(Thunk(RefCell::new(Slot::Pending(e, env, lambda))))
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
                let body_lambda = lambda.as_ref().map(|dep| {
                    let params = f.params.iter().zip(args);
                    Rc::new(
                        params
                            .filter(|(_, a)| depends(a, dep))
                            .map(|((p, _), _)| *p)
                            .collect(),
                    )
                });
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
                    Some(dep) => !depends(c, dep),
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
                        && depends(x, dep)
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
    /// Every error that may surface in the statement, with its message, one per category.
    pub errors: Vec<(Category, String)>,
}

/// What the evaluator computed for a program.
pub struct Reference {
    /// The output of the statements before the first failing one.
    pub printed: String,
    /// The exports in source order, or the failing statement.
    pub result: Result<Vec<(String, Value)>, Failed>,
}

/// Evaluates `source` on `inputs`. Fails with the reason when the program uses something the
/// evaluator does not support. `reg` only serves type checking.
///
/// With `eager_lets`, every top-level let is evaluated at its statement. Until #40 (errors as
/// data), the pipeline computes top-level lets it does not inline, and this mode only serves
/// to recognize that divergence.
pub fn evaluate(
    reg: &Registry,
    source: &str,
    inputs: &[(String, Value)],
    eager_lets: bool,
) -> Result<Reference, String> {
    let ast = parse(source).map_err(|e| format!("does not parse: {e}"))?;
    let mut globals = Globals::new();
    let program =
        check::check(&ast, reg, &mut globals).map_err(|e| format!("does not check: {e}"))?;
    let ev = Evaluator { globals: &globals };
    let mut env = Env::default();
    let mut printed = String::new();
    let mut exports = Vec::new();
    for s in &program.stmts {
        let outcome = match s {
            TStmt::Input { var, .. } => {
                let name = globals.name(*var);
                let (_, value) = inputs
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("input {name} is not supplied"));
                env = env.bind(*var, Thunk::done(from_value(value)));
                Ok(())
            }
            TStmt::Let { var, value, .. } => {
                let t = Thunk::pending(value, env.clone(), None);
                env = env.bind(*var, Rc::clone(&t));
                if eager_lets {
                    ev.force(&t).map(|_| ())
                } else {
                    Ok(())
                }
            }
            TStmt::Export { var, value, .. } => {
                let t = Thunk::pending(value, env.clone(), None);
                env = env.bind(*var, Rc::clone(&t));
                ev.force(&t)
                    .map(|v| exports.push((globals.name(*var).to_string(), to_value(&v))))
            }
            TStmt::Print { value, .. } => ev.eval(value, &env, &None).map(|v| {
                writeln!(printed, "{}", to_value(&v)).expect("writing to a string");
            }),
            // Its value reaches no print or export.
            TStmt::Expr { .. } => Ok(()),
        };
        match outcome {
            Ok(()) => {}
            Err(Stop::Unsupported(reason)) => return Err(reason),
            Err(Stop::Errors(errors)) => {
                return Ok(Reference {
                    printed,
                    result: Err(Failed {
                        line: stmt_line(s),
                        errors,
                    }),
                });
            }
        }
    }
    Ok(Reference {
        printed,
        result: Ok(exports),
    })
}

fn stmt_line(s: &TStmt) -> u32 {
    let (TStmt::Let { span, .. }
    | TStmt::Input { span, .. }
    | TStmt::Export { span, .. }
    | TStmt::Print { span, .. }
    | TStmt::Expr { span, .. }) = s;
    span.line
}
