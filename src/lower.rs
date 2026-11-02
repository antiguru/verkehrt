//! Lowering from the typed AST to first-order array terms.
//!
//! Function calls and block lets are inlined by substitution, lambdas are lifted so they run
//! once over whole arrays, scalar operands next to arrays get a broadcast, single-use lets are
//! inlined, and repeated subterms are hoisted so every statement's term is a tree without
//! shared structure, except inside the branches of a conditional, which must not run
//! unconditionally. An `if` on a scalar condition stays a lazy conditional, any other `if`
//! becomes `select`. Terms are nodes of one hash-consed `TermGraph` per program, so a value
//! read many times is one node.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::check::{Globals, TExpr, TExprKind, TLambda, TProgram, TStmt, VarId};
use crate::error::{Error, Span};
use crate::kernels::Registry;
use crate::ops::Op;
use crate::value::{ScalarType, Value, ValueType};

mod graph;
pub(crate) mod higher;
mod leaves;
mod passes;
#[cfg(test)]
mod tests;

use higher::{Build, Higher};
use leaves::{Leaves, array_of};
use passes::{hoist_repeated, inline_single_use};

pub use graph::{TermGraph, TermId};

/// How top-level lets are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Single-use lets are inlined into their user.
    Run,
    /// Every let stays a binding because later inputs may refer to it.
    Repl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LeafId(pub u32);

/// A value a term reads without computing it.
#[derive(Clone, Debug)]
pub enum Leaf {
    Const(Value),
    /// A materialized binding or an input.
    Var(VarId),
}

/// A node of a first-order term, with its children as ids in the `TermGraph`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TermKind {
    Op(Op, Vec<TermId>),
    Leaf(LeafId),
    /// A scalar term repeated to the length of the array operands next to it.
    Bcast(TermId),
    /// Evaluates the scalar `bool` condition, then only the branch it picks. Both branches
    /// have the type of the term.
    Cond(TermId, TermId, TermId),
}

#[derive(Clone, Debug)]
pub enum LStmt {
    /// Binds `var` to the value of `term`. An `output` bind is demanded: a runtime error of
    /// its term fails the program at this statement. Any other bind holds the error in `var`
    /// until an output reads it.
    Bind {
        var: VarId,
        term: TermId,
        span: Span,
        output: bool,
    },
    Print {
        term: TermId,
        span: Span,
    },
    Show {
        term: TermId,
        span: Span,
    },
}

impl LStmt {
    /// The span of the source statement, which runtime errors of the statement carry.
    pub fn span(&self) -> Span {
        let (LStmt::Bind { span, .. } | LStmt::Print { span, .. } | LStmt::Show { span, .. }) =
            self;
        *span
    }

    /// The term of the statement.
    pub fn term(&self) -> TermId {
        let (LStmt::Bind { term, .. } | LStmt::Print { term, .. } | LStmt::Show { term, .. }) =
            self;
        *term
    }
}

#[derive(Clone, Debug)]
pub struct Lowered {
    pub leaves: Vec<Leaf>,
    /// The terms of all statements.
    pub graph: TermGraph,
    pub stmts: Vec<LStmt>,
    /// Exported names in source order. Each variable has an `output` `Bind` in `stmts`, which
    /// places the export among the outputs in statement order.
    pub exports: Vec<(String, VarId)>,
}

/// The lowercase name of an op, as used in diagnostics and test output.
pub fn op_name(op: Op) -> String {
    match op {
        Op::Reduce(r) => format!("reduce_{}", format!("{r:?}").to_lowercase()),
        Op::Scan(r) => format!("scan_{}", format!("{r:?}").to_lowercase()),
        Op::ScanBy(r) => format!("scan_by_{}", format!("{r:?}").to_lowercase()),
        Op::ReduceInto(r) => format!("reduce_into_{}", format!("{r:?}").to_lowercase()),
        Op::DivConst => "div_const".into(),
        Op::GroupId => "group_id".into(),
        Op::Cast(t) => format!("cast_{}", t.name()),
        Op::Load(t) => format!("load_{}", t.name()),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// The type error for `op` having no kernel on operands of types `tys`.
pub(crate) fn no_kernel(op: Op, tys: &[ValueType], span: Span) -> Error {
    let shown: Vec<String> = tys.iter().map(ToString::to_string).collect();
    Error::ty(
        format!("no kernel for '{}' on {}", op_name(op), shown.join(", ")),
        span,
    )
}

fn is_array(ty: ValueType) -> bool {
    matches!(ty, ValueType::Array(_))
}

/// Lowers a checked program. Hoisted temporaries are declared in `globals` as `_t0`, `_t1`, ...
pub fn lower(
    prog: &TProgram,
    globals: &mut Globals,
    reg: &Registry,
    mode: Mode,
) -> Result<Lowered, Error> {
    let mut lw = Lowerer {
        g: globals,
        reg,
        leaves: Leaves::default(),
        graph: TermGraph::default(),
        env: HashMap::new(),
        lifted: Vec::new(),
    };
    let mut stmts = Vec::new();
    let mut exports = Vec::new();
    let mut exported = HashSet::new();
    for s in &prog.stmts {
        match s {
            // The value is supplied at execution, every use is a `Leaf::Var`.
            TStmt::Input { .. } => {}
            TStmt::Let { var, value, span } => {
                let term = lw.expr(value)?;
                stmts.push(LStmt::Bind {
                    var: *var,
                    term,
                    span: *span,
                    // A top-level let of the REPL stays bound for later inputs, so it is
                    // demanded where it stands.
                    output: mode == Mode::Repl,
                });
            }
            TStmt::Export { var, value, span } => {
                let term = lw.expr(value)?;
                exports.push((lw.g.name(*var).to_string(), *var));
                exported.insert(*var);
                stmts.push(LStmt::Bind {
                    var: *var,
                    term,
                    span: *span,
                    output: true,
                });
            }
            TStmt::Print { value, span } => {
                stmts.push(LStmt::Print {
                    term: lw.expr(value)?,
                    span: *span,
                });
            }
            TStmt::Expr { value, span } => {
                stmts.push(LStmt::Show {
                    term: lw.expr(value)?,
                    span: *span,
                });
            }
        }
    }
    let Lowerer {
        mut leaves,
        mut graph,
        ..
    } = lw;
    if mode == Mode::Run {
        stmts = inline_single_use(stmts, &leaves, &mut graph, &exported);
    }
    let stmts = stmts
        .into_iter()
        .flat_map(|s| hoist_repeated(s, &mut leaves, &mut graph, globals))
        .collect();
    Ok(Lowered {
        leaves: leaves.leaves,
        graph,
        stmts,
        exports,
    })
}

struct Lowerer<'a> {
    g: &'a Globals,
    reg: &'a Registry,
    leaves: Leaves,
    graph: TermGraph,
    /// Terms substituted for parameters and block locals. Other variables are top-level
    /// bindings and stay leaves.
    env: HashMap<VarId, TermId>,
    /// The first parameter array of each lambda body being lifted around the current
    /// expression, innermost last.
    lifted: Vec<TermId>,
}

impl Lowerer<'_> {
    fn leaf(&mut self, leaf: Leaf, ty: ValueType) -> TermId {
        let id = match leaf {
            Leaf::Const(v) => self.leaves.constant(v),
            Leaf::Var(v) => self.leaves.var(v),
        };
        self.graph.intern(TermKind::Leaf(id), ty)
    }

    /// Builds an operation over lowered operands. The kernel lookup on the operands as given
    /// decides which scalars next to an array operand are broadcast: those at `Either`
    /// positions, while a `Scalar` input keeps its scalar. When no kernel accepts the operands
    /// as given, every scalar next to an array is broadcast and the lookup runs on the lifted
    /// types, unless the op has a kernel with a `Scalar` input. Then the error reports the
    /// operands as given, because the lifted types would not be the ones the user wrote.
    /// The rule is `Registry::lookup_lifted`, which the checker applies to lambda bodies too.
    fn op(&mut self, op: Op, args: Vec<TermId>, span: Span) -> Result<TermId, Error> {
        let tys: Vec<ValueType> = args.iter().map(|&a| self.graph.ty(a)).collect();
        let lifted = self
            .reg
            .lookup_lifted(op, &tys)
            .map_err(|tys| no_kernel(op, &tys, span))?;
        let args = args
            .into_iter()
            .zip(lifted.bcast)
            .map(|(a, bcast)| match self.graph.ty(a) {
                ValueType::Scalar(t) if bcast => {
                    self.graph.intern(TermKind::Bcast(a), ValueType::Array(t))
                }
                _ => a,
            })
            .collect();
        let ty = self.reg.get(lifted.kernel).output;
        Ok(self.graph.intern(TermKind::Op(op, args), ty))
    }

    /// Builds an operation over operands exactly as given, without broadcasting.
    fn op_exact(&mut self, op: Op, args: Vec<TermId>, span: Span) -> Result<TermId, Error> {
        let tys: Vec<ValueType> = args.iter().map(|&a| self.graph.ty(a)).collect();
        let Some(id) = self.reg.lookup(op, &tys) else {
            return Err(no_kernel(op, &tys, span));
        };
        let ty = self.reg.get(id).output;
        Ok(self.graph.intern(TermKind::Op(op, args), ty))
    }

    fn exprs(&mut self, es: &[TExpr]) -> Result<Vec<TermId>, Error> {
        es.iter().map(|e| self.expr(e)).collect()
    }

    fn expr(&mut self, e: &TExpr) -> Result<TermId, Error> {
        let span = e.span;
        match &e.kind {
            TExprKind::Lit(s) => Ok(self.leaf(Leaf::Const(Value::Scalar(s.clone())), e.ty)),
            TExprKind::Str(s) => Ok(self.leaf(Leaf::Const(Value::Str(Arc::clone(s))), e.ty)),
            TExprKind::ArrayLit(items) => self.array_lit(e, items),
            TExprKind::Var(v) => Ok(match self.env.get(v) {
                Some(&t) => t,
                None => self.leaf(Leaf::Var(*v), e.ty),
            }),
            TExprKind::Prim(op, args) => {
                let args = self.exprs(args)?;
                self.op(*op, args, span)
            }
            TExprKind::Call(id, args) => {
                let g = self.g;
                let f = g.func(*id);
                let args = self.exprs(args)?;
                let params: Vec<VarId> = f.params.iter().map(|(v, _)| *v).collect();
                self.with_bindings(&params, args, |lw| lw.expr(&f.body))
            }
            TExprKind::Map(x, lam) => {
                let x = self.expr(x)?;
                self.lift(Higher::Map, lam, vec![x], span)
            }
            TExprKind::ZipMap(x, y, lam) => {
                let (x, y) = (self.expr(x)?, self.expr(y)?);
                self.lift(Higher::ZipMap, lam, vec![x, y], span)
            }
            TExprKind::Filter(x, lam) => {
                let x = self.expr(x)?;
                self.lift(Higher::Filter, lam, vec![x], span)
            }
            TExprKind::If(c, t, f) => {
                let (c, mut t, mut f) = (self.expr(c)?, self.expr(t)?, self.expr(f)?);
                // A scalar condition is lazy, also in a lifted lambda, where a condition that
                // depends on a parameter is an array and stays a `select`. A uniform condition
                // there may pick a uniform scalar branch next to an array branch, so the scalar
                // one is filled to the lambda's length, which comes from its first parameter
                // and not from the other branch, so the branch not taken never runs.
                let bool_cond = self.graph.ty(c) == ValueType::Scalar(ScalarType::Bool);
                if bool_cond && let Some(&param) = self.lifted.last() {
                    let fill = |lw: &mut Self, branch: TermId| -> Result<TermId, Error> {
                        let len = lw.op(Op::Length, vec![param], span)?;
                        lw.op(Op::Fill, vec![len, branch], span)
                    };
                    match (is_array(self.graph.ty(t)), is_array(self.graph.ty(f))) {
                        (true, false) => f = fill(self, f)?,
                        (false, true) => t = fill(self, t)?,
                        _ => {}
                    }
                }
                if bool_cond && self.graph.ty(t) == self.graph.ty(f) {
                    let ty = self.graph.ty(t);
                    return Ok(self.graph.intern(TermKind::Cond(c, t, f), ty));
                }
                self.op(Op::Select, vec![c, t, f], span)
            }
            TExprKind::Block(lets, result) => {
                let mut saved = Vec::new();
                let mut r = Ok(());
                for (v, value) in lets {
                    match self.expr(value) {
                        Ok(t) => saved.push((*v, self.env.insert(*v, t))),
                        Err(err) => {
                            r = Err(err);
                            break;
                        }
                    }
                }
                let out = r.and_then(|()| self.expr(result));
                self.restore(saved);
                out
            }
        }
    }

    /// Array literals of constants become one constant array.
    fn array_lit(&mut self, e: &TExpr, items: &[TExpr]) -> Result<TermId, Error> {
        let ValueType::Array(elem) = e.ty else {
            unreachable!("the checker types an array literal as an array")
        };
        let mut scalars = Vec::new();
        for it in items {
            let t = self.expr(it)?;
            let constant = match *self.graph.kind(t) {
                TermKind::Leaf(id) => match self.leaves.get(id) {
                    Leaf::Const(Value::Scalar(s)) => Some(s.clone()),
                    _ => None,
                },
                _ => None,
            };
            let Some(s) = constant else {
                return Err(Error::ty(
                    "array literal elements must be constants",
                    it.span,
                ));
            };
            scalars.push(s);
        }
        let arr = array_of(elem, &scalars);
        Ok(self.leaf(Leaf::Const(Value::Array(arr)), e.ty))
    }

    /// Runs `f` with `params` bound to `args`, then restores the previous bindings. The
    /// restore matters because the same parameters are bound again on a nested or repeated use.
    fn with_bindings<T>(
        &mut self,
        params: &[VarId],
        args: Vec<TermId>,
        f: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let saved: Vec<_> = params
            .iter()
            .zip(args)
            .map(|(p, a)| (*p, self.env.insert(*p, a)))
            .collect();
        let r = f(self);
        self.restore(saved);
        r
    }

    fn restore(&mut self, saved: Vec<(VarId, Option<TermId>)>) {
        for (v, old) in saved.into_iter().rev() {
            match old {
                Some(t) => self.env.insert(v, t),
                None => self.env.remove(&v),
            };
        }
    }

    /// Applies a lambda once to whole arrays, as the higher-order call `kind` on the arrays
    /// `xs` does.
    fn lift(
        &mut self,
        kind: Higher,
        lam: &TLambda,
        xs: Vec<TermId>,
        span: Span,
    ) -> Result<TermId, Error> {
        let params = higher::params(self, kind, xs.clone(), span)?;
        self.lifted.push(params[0]);
        let body = self.with_bindings(&lam.params, params.clone(), |lw| lw.expr(&lam.body));
        self.lifted.pop();
        higher::result(self, kind, &xs, &params, body?, span)
    }
}

impl Build for Lowerer<'_> {
    type V = TermId;

    fn ty(&self, v: &TermId) -> ValueType {
        self.graph.ty(*v)
    }

    fn apply(&mut self, op: Op, args: Vec<TermId>, span: Span) -> Result<TermId, Error> {
        self.op(op, args, span)
    }

    fn apply_exact(&mut self, op: Op, args: Vec<TermId>, span: Span) -> Result<TermId, Error> {
        self.op_exact(op, args, span)
    }
}
