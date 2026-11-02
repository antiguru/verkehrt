//! Lowering from the typed AST to first-order array terms.
//!
//! Function calls and block lets are inlined by substitution, lambdas are lifted so they run
//! once over whole arrays, scalar operands next to arrays get a broadcast, single-use lets are
//! inlined, and repeated subterms are hoisted so every statement's term is a tree without
//! shared structure, except inside the branches of a conditional, which must not run
//! unconditionally. An `if` on a scalar condition stays a lazy conditional, any other `if`
//! becomes `select`.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::check::{Globals, TExpr, TExprKind, TLambda, TProgram, TStmt, VarId};
use crate::error::{Error, Span};
use crate::kernels::Registry;
use crate::ops::Op;
use crate::value::{Array, Scalar, ScalarType, Value, ValueType};

pub(crate) mod higher;
#[cfg(test)]
mod tests;

use higher::{Build, Higher};

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TermKind {
    Op(Op, Vec<Term>),
    Leaf(LeafId),
    /// A scalar term repeated to the length of the array operands next to it.
    Bcast(Box<Term>),
    /// Evaluates the scalar `bool` condition, then only the branch it picks. Both branches
    /// have the type of the term.
    Cond(Box<Term>, Box<Term>, Box<Term>),
}

/// A first-order term. Equality and hashing ignore `span`.
#[derive(Clone, Debug)]
pub struct Term {
    pub kind: TermKind,
    pub ty: ValueType,
    pub span: Span,
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        self.kind == other.kind && self.ty == other.ty
    }
}

impl Eq for Term {}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.ty.hash(state);
    }
}

#[derive(Clone, Debug)]
pub enum LStmt {
    /// Binds `var` to the value of `term`. An `output` bind is demanded: a runtime error of
    /// its term fails the program at this statement. Any other bind holds the error in `var`
    /// until an output reads it.
    Bind {
        var: VarId,
        term: Term,
        span: Span,
        output: bool,
    },
    Print {
        term: Term,
        span: Span,
    },
    Show {
        term: Term,
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
}

#[derive(Clone, Debug)]
pub struct Lowered {
    pub leaves: Vec<Leaf>,
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

/// The interned leaves. Variables share one leaf each, as do equal scalar and string
/// constants, so that structurally equal terms compare equal.
struct Leaves(Vec<Leaf>);

impl Leaves {
    fn push(&mut self, leaf: Leaf) -> LeafId {
        self.0.push(leaf);
        LeafId(u32::try_from(self.0.len() - 1).expect("leaf count fits in u32"))
    }

    fn id_at(i: usize) -> LeafId {
        LeafId(u32::try_from(i).expect("leaf count fits in u32"))
    }

    fn var(&mut self, v: VarId) -> LeafId {
        match self
            .0
            .iter()
            .position(|l| matches!(l, Leaf::Var(w) if *w == v))
        {
            Some(i) => Leaves::id_at(i),
            None => self.push(Leaf::Var(v)),
        }
    }

    fn constant(&mut self, value: Value) -> LeafId {
        // Arrays are never compared, they are large and each literal is its own value.
        if !matches!(value, Value::Array(_)) {
            let found = self.0.iter().position(|l| match l {
                Leaf::Const(c) => same_const(c, &value),
                Leaf::Var(_) => false,
            });
            if let Some(i) = found {
                return Leaves::id_at(i);
            }
        }
        self.push(Leaf::Const(value))
    }

    fn get(&self, id: LeafId) -> &Leaf {
        &self.0[usize::try_from(id.0).expect("u32 fits in usize")]
    }
}

/// Bitwise-faithful equality for scalar and string constants. Display distinguishes `-0.0`
/// from `0.0`, which `==` does not.
fn same_const(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => x.ty() == y.ty() && x.to_string() == y.to_string(),
        (Value::Str(x), Value::Str(y)) => x == y,
        _ => false,
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
        leaves: Leaves(Vec::new()),
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
    let Lowerer { mut leaves, .. } = lw;
    if mode == Mode::Run {
        stmts = inline_single_use(stmts, &leaves, &exported);
    }
    let stmts = stmts
        .into_iter()
        .flat_map(|s| hoist_repeated(s, &mut leaves, globals))
        .collect();
    Ok(Lowered {
        leaves: leaves.0,
        stmts,
        exports,
    })
}

struct Lowerer<'a> {
    g: &'a Globals,
    reg: &'a Registry,
    leaves: Leaves,
    /// Terms substituted for parameters and block locals. Other variables are top-level
    /// bindings and stay leaves.
    env: HashMap<VarId, Term>,
    /// The first parameter array of each lambda body being lifted around the current
    /// expression, innermost last.
    lifted: Vec<Term>,
}

impl Lowerer<'_> {
    fn leaf(&mut self, leaf: Leaf, ty: ValueType, span: Span) -> Term {
        let id = match leaf {
            Leaf::Const(v) => self.leaves.constant(v),
            Leaf::Var(v) => self.leaves.var(v),
        };
        Term {
            kind: TermKind::Leaf(id),
            ty,
            span,
        }
    }

    /// Builds an operation over lowered operands. The kernel lookup on the operands as given
    /// decides which scalars next to an array operand are broadcast: those at `Either`
    /// positions, while a `Scalar` input keeps its scalar. When no kernel accepts the operands
    /// as given, every scalar next to an array is broadcast and the lookup runs on the lifted
    /// types, unless the op has a kernel with a `Scalar` input. Then the error reports the
    /// operands as given, because the lifted types would not be the ones the user wrote.
    /// The rule is `Registry::lookup_lifted`, which the checker applies to lambda bodies too.
    fn op(&self, op: Op, args: Vec<Term>, span: Span) -> Result<Term, Error> {
        let tys: Vec<ValueType> = args.iter().map(|a| a.ty).collect();
        let lifted = self
            .reg
            .lookup_lifted(op, &tys)
            .map_err(|tys| no_kernel(op, &tys, span))?;
        let args = args
            .into_iter()
            .zip(lifted.bcast)
            .map(|(a, bcast)| match a.ty {
                ValueType::Scalar(t) if bcast => {
                    let operand_span = a.span;
                    Term {
                        kind: TermKind::Bcast(Box::new(a)),
                        ty: ValueType::Array(t),
                        span: operand_span,
                    }
                }
                _ => a,
            })
            .collect();
        Ok(Term {
            kind: TermKind::Op(op, args),
            ty: self.reg.get(lifted.kernel).output,
            span,
        })
    }

    /// Builds an operation over operands exactly as given, without broadcasting.
    fn op_exact(&self, op: Op, args: Vec<Term>, span: Span) -> Result<Term, Error> {
        let tys: Vec<ValueType> = args.iter().map(|a| a.ty).collect();
        let Some(id) = self.reg.lookup(op, &tys) else {
            return Err(no_kernel(op, &tys, span));
        };
        Ok(Term {
            kind: TermKind::Op(op, args),
            ty: self.reg.get(id).output,
            span,
        })
    }

    fn exprs(&mut self, es: &[TExpr]) -> Result<Vec<Term>, Error> {
        es.iter().map(|e| self.expr(e)).collect()
    }

    fn expr(&mut self, e: &TExpr) -> Result<Term, Error> {
        let span = e.span;
        match &e.kind {
            TExprKind::Lit(s) => Ok(self.leaf(Leaf::Const(Value::Scalar(s.clone())), e.ty, span)),
            TExprKind::Str(s) => Ok(self.leaf(Leaf::Const(Value::Str(Arc::clone(s))), e.ty, span)),
            TExprKind::ArrayLit(items) => self.array_lit(e, items),
            TExprKind::Var(v) => Ok(match self.env.get(v) {
                Some(t) => t.clone(),
                None => self.leaf(Leaf::Var(*v), e.ty, span),
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
                if c.ty == ValueType::Scalar(ScalarType::Bool)
                    && let Some(param) = self.lifted.last().cloned()
                {
                    let fill = |lw: &Self, branch: Term| -> Result<Term, Error> {
                        let len = lw.op(Op::Length, vec![param.clone()], span)?;
                        lw.op(Op::Fill, vec![len, branch], span)
                    };
                    match (is_array(t.ty), is_array(f.ty)) {
                        (true, false) => f = fill(self, f)?,
                        (false, true) => t = fill(self, t)?,
                        _ => {}
                    }
                }
                if c.ty == ValueType::Scalar(ScalarType::Bool) && t.ty == f.ty {
                    let ty = t.ty;
                    return Ok(Term {
                        kind: TermKind::Cond(Box::new(c), Box::new(t), Box::new(f)),
                        ty,
                        span,
                    });
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
    fn array_lit(&mut self, e: &TExpr, items: &[TExpr]) -> Result<Term, Error> {
        let ValueType::Array(elem) = e.ty else {
            unreachable!("the checker types an array literal as an array")
        };
        let mut scalars = Vec::new();
        for it in items {
            let t = self.expr(it)?;
            let constant = match t.kind {
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
        Ok(self.leaf(Leaf::Const(Value::Array(arr)), e.ty, e.span))
    }

    /// Runs `f` with `params` bound to `args`, then restores the previous bindings. The
    /// restore matters because the same parameters are bound again on a nested or repeated use.
    fn with_bindings<T>(
        &mut self,
        params: &[VarId],
        args: Vec<Term>,
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

    fn restore(&mut self, saved: Vec<(VarId, Option<Term>)>) {
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
        xs: Vec<Term>,
        span: Span,
    ) -> Result<Term, Error> {
        let params = higher::params(self, kind, xs.clone(), span)?;
        self.lifted.push(params[0].clone());
        let body = self.with_bindings(&lam.params, params.clone(), |lw| lw.expr(&lam.body));
        self.lifted.pop();
        higher::result(self, kind, &xs, &params, body?, span)
    }
}

impl Build for Lowerer<'_> {
    type V = Term;

    fn ty(v: &Term) -> ValueType {
        v.ty
    }

    fn apply(&self, op: Op, args: Vec<Term>, span: Span) -> Result<Term, Error> {
        self.op(op, args, span)
    }

    fn apply_exact(&self, op: Op, args: Vec<Term>, span: Span) -> Result<Term, Error> {
        self.op_exact(op, args, span)
    }
}

fn array_of(elem: ScalarType, items: &[Scalar]) -> Array {
    macro_rules! build {
        ($variant:ident, $ty:ty) => {
            Array::from_vec(
                items
                    .iter()
                    .map(|s| match s {
                        Scalar::$variant(v) => *v,
                        _ => unreachable!("the checker unifies array element types"),
                    })
                    .collect::<Vec<$ty>>(),
            )
        };
    }
    match elem {
        ScalarType::I8 => build!(I8, i8),
        ScalarType::I16 => build!(I16, i16),
        ScalarType::I32 => build!(I32, i32),
        ScalarType::I64 => build!(I64, i64),
        ScalarType::U8 => build!(U8, u8),
        ScalarType::U16 => build!(U16, u16),
        ScalarType::U32 => build!(U32, u32),
        ScalarType::U64 => build!(U64, u64),
        ScalarType::F32 => build!(F32, f32),
        ScalarType::F64 => build!(F64, f64),
        ScalarType::Bool => {
            let bools: Vec<bool> = items
                .iter()
                .map(|s| matches!(s, Scalar::Bool(true)))
                .collect();
            Array::from_bools(&bools)
        }
    }
}

fn leaf_vars(t: &Term, leaves: &Leaves, f: &mut impl FnMut(VarId)) {
    match &t.kind {
        TermKind::Leaf(id) => {
            if let Leaf::Var(v) = leaves.get(*id) {
                f(*v);
            }
        }
        TermKind::Bcast(a) => leaf_vars(a, leaves, f),
        TermKind::Op(_, args) => args.iter().for_each(|a| leaf_vars(a, leaves, f)),
        TermKind::Cond(c, t, e) => [c, t, e].into_iter().for_each(|a| leaf_vars(a, leaves, f)),
    }
}

fn stmt_term_mut(s: &mut LStmt) -> &mut Term {
    match s {
        LStmt::Bind { term, .. } | LStmt::Print { term, .. } | LStmt::Show { term, .. } => term,
    }
}

fn substitute(t: &Term, pending: &HashMap<VarId, Term>, leaves: &Leaves) -> Term {
    let kind = match &t.kind {
        TermKind::Leaf(id) => match leaves.get(*id) {
            Leaf::Var(v) if pending.contains_key(v) => return pending[v].clone(),
            _ => t.kind.clone(),
        },
        TermKind::Bcast(a) => TermKind::Bcast(Box::new(substitute(a, pending, leaves))),
        TermKind::Op(op, args) => TermKind::Op(
            *op,
            args.iter()
                .map(|a| substitute(a, pending, leaves))
                .collect(),
        ),
        // A let used only in a branch moves into it and runs only when the branch does.
        TermKind::Cond(c, t, e) => {
            let sub = |a: &Term| Box::new(substitute(a, pending, leaves));
            TermKind::Cond(sub(c), sub(t), sub(e))
        }
    };
    Term {
        kind,
        ty: t.ty,
        span: t.span,
    }
}

/// Substitutes each compound let with exactly one use into its user and drops the let.
/// A let bound to a leaf is already as cheap as it gets and stays the materialized binding,
/// and an export counts as an extra use so it never goes away.
fn inline_single_use(stmts: Vec<LStmt>, leaves: &Leaves, exported: &HashSet<VarId>) -> Vec<LStmt> {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for s in &stmts {
        let (LStmt::Bind { term, .. } | LStmt::Print { term, .. } | LStmt::Show { term, .. }) = s;
        leaf_vars(term, leaves, &mut |v| *uses.entry(v).or_default() += 1);
    }
    let mut pending: HashMap<VarId, Term> = HashMap::new();
    let mut out = Vec::new();
    for mut s in stmts {
        let t = stmt_term_mut(&mut s);
        *t = substitute(t, &pending, leaves);
        if let LStmt::Bind { var, term, .. } = &s
            && !exported.contains(var)
            && uses.get(var) == Some(&1)
            && !matches!(term.kind, TermKind::Leaf(_))
        {
            pending.insert(*var, term.clone());
            continue;
        }
        out.push(s);
    }
    out
}

/// Counts every distinct compound subterm that is evaluated unconditionally. A repeated
/// subterm is descended into once, so its children are not counted as repeated just because
/// their parent is. Branches of a conditional are not counted, so nothing inside them is
/// hoisted into code that always runs.
fn count_ops(t: &Term, counts: &mut HashMap<Term, usize>) {
    match &t.kind {
        TermKind::Leaf(_) => {}
        TermKind::Bcast(a) => count_ops(a, counts),
        TermKind::Op(..) | TermKind::Cond(..) => {
            let c = counts.entry(t.clone()).or_default();
            *c += 1;
            if *c == 1 {
                match &t.kind {
                    TermKind::Op(_, args) => args.iter().for_each(|a| count_ops(a, counts)),
                    TermKind::Cond(cond, ..) => count_ops(cond, counts),
                    TermKind::Leaf(_) | TermKind::Bcast(_) => {}
                }
            }
        }
    }
}

struct Hoister<'a> {
    /// The span of the statement, which every hoisted bind takes, so that a runtime error in
    /// one carries the line of the statement it serves.
    span: Span,
    counts: HashMap<Term, usize>,
    done: HashMap<Term, Term>,
    binds: Vec<LStmt>,
    leaves: &'a mut Leaves,
    g: &'a mut Globals,
}

impl Hoister<'_> {
    fn go(&mut self, t: &Term) -> Term {
        match &t.kind {
            TermKind::Leaf(_) => t.clone(),
            TermKind::Bcast(a) => Term {
                kind: TermKind::Bcast(Box::new(self.go(a))),
                ty: t.ty,
                span: t.span,
            },
            TermKind::Op(..) | TermKind::Cond(..) => {
                let shared = self.counts[t] >= 2;
                if shared && let Some(leaf) = self.done.get(t) {
                    return leaf.clone();
                }
                let kind = match &t.kind {
                    TermKind::Op(op, args) => {
                        TermKind::Op(*op, args.iter().map(|a| self.go(a)).collect())
                    }
                    // Branches stay as they are, see `count_ops`.
                    TermKind::Cond(c, a, b) => {
                        TermKind::Cond(Box::new(self.go(c)), a.clone(), b.clone())
                    }
                    TermKind::Leaf(_) | TermKind::Bcast(_) => unreachable!("matched above"),
                };
                let rebuilt = Term {
                    kind,
                    ty: t.ty,
                    span: t.span,
                };
                if !shared {
                    return rebuilt;
                }
                let var = self.g.fresh_temp();
                // The statement reads the temporary unconditionally, so it raises the
                // temporary's error if it is an output.
                self.binds.push(LStmt::Bind {
                    var,
                    term: rebuilt,
                    span: self.span,
                    output: false,
                });
                let id = self.leaves.var(var);
                let leaf = Term {
                    kind: TermKind::Leaf(id),
                    ty: t.ty,
                    span: t.span,
                };
                self.done.insert(t.clone(), leaf.clone());
                leaf
            }
        }
    }
}

/// Hoists the compound subterms that occur at least twice into binds placed before the
/// statement, so the statement's own term has no shared structure outside the branches of
/// conditionals.
fn hoist_repeated(mut s: LStmt, leaves: &mut Leaves, g: &mut Globals) -> Vec<LStmt> {
    let mut counts = HashMap::new();
    count_ops(stmt_term_mut(&mut s), &mut counts);
    let mut h = Hoister {
        span: s.span(),
        counts,
        done: HashMap::new(),
        binds: Vec::new(),
        leaves,
        g,
    };
    let t = stmt_term_mut(&mut s);
    *t = h.go(t);
    let mut out = h.binds;
    out.push(s);
    out
}
