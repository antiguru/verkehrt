//! The type checker. It turns the AST into a typed AST and enforces the language's typing
//! rules: kernel lookup through the registry, the lambda liftability rules, and the ban on
//! recursion. Lambdas, calls, and `if` stay intact because lowering handles them next.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Span};
use crate::kernels::Registry;
use crate::ops::{Op, ReduceOp};
use crate::syntax::ast::{self, BinOp, ExprKind};
use crate::value::{ScalarType, ValueType};

mod typed;
pub use typed::*;

#[cfg(test)]
mod tests;

/// Top-level state that persists across REPL inputs: names, their variables, and functions.
#[derive(Clone, Debug, Default)]
pub struct Globals {
    /// Name of every variable ever created, indexed by `VarId`.
    var_names: Vec<String>,
    names: HashMap<String, (VarId, ValueType)>,
    fns: Vec<TFn>,
    fn_names: HashMap<String, FnId>,
    temps: u32,
}

impl Globals {
    pub fn new() -> Globals {
        Globals::default()
    }

    /// Creates a variable that is not bound to any name. Used for locals and parameters.
    pub fn fresh_var(&mut self, name: &str) -> VarId {
        let id = VarId(u32::try_from(self.var_names.len()).expect("variable count fits in u32"));
        self.var_names.push(name.to_string());
        id
    }

    /// Creates a temporary named `_t0`, `_t1`, and so on.
    pub fn fresh_temp(&mut self) -> VarId {
        let name = format!("_t{}", self.temps);
        self.temps += 1;
        self.fresh_var(&name)
    }

    /// Binds `name` to a new variable of type `ty`, shadowing an earlier binding.
    /// A host uses this to provide a value that the program did not define.
    pub fn declare(&mut self, name: &str, ty: ValueType) -> VarId {
        let var = self.fresh_var(name);
        self.names.insert(name.to_string(), (var, ty));
        var
    }

    /// The number of variables created so far. Ids below it are in use.
    pub fn var_count(&self) -> u32 {
        u32::try_from(self.var_names.len()).expect("variable count fits in u32")
    }

    pub fn name(&self, var: VarId) -> &str {
        &self.var_names[to_index(var.0)]
    }

    pub fn lookup(&self, name: &str) -> Option<(VarId, ValueType)> {
        self.names.get(name).copied()
    }

    pub fn lookup_fn(&self, name: &str) -> Option<FnId> {
        self.fn_names.get(name).copied()
    }

    pub fn func(&self, id: FnId) -> &TFn {
        &self.fns[to_index(id.0)]
    }
}

fn to_index(i: u32) -> usize {
    usize::try_from(i).expect("u32 fits in usize")
}

/// Type checks a program. On error `env` is left unchanged.
pub fn check(prog: &ast::Program, reg: &Registry, env: &mut Globals) -> Result<TProgram, Error> {
    let mut work = env.clone();
    let stmts = Checker::new(reg, &mut work).run(prog)?;
    *env = work;
    Ok(TProgram { stmts })
}

enum Builtin {
    Op(Op, usize),
    Reduce,
    Map,
    ZipMap,
    Filter,
}

/// Maps a call name to its builtin. User functions may not use these names.
fn builtin(name: &str) -> Option<Builtin> {
    let (op, arity) = match name {
        "neg" => (Op::Neg, 1),
        "abs" => (Op::Abs, 1),
        "min" => (Op::Min, 2),
        "max" => (Op::Max, 2),
        "sqrt" => (Op::Sqrt, 1),
        "and" => (Op::And, 2),
        "or" => (Op::Or, 2),
        "not" => (Op::Not, 1),
        "select" => (Op::Select, 3),
        "sum" => (Op::Reduce(ReduceOp::Add), 1),
        "count" => (Op::Count, 1),
        "any" => (Op::Any, 1),
        "all" => (Op::All, 1),
        "argmin" => (Op::ArgMin, 1),
        "argmax" => (Op::ArgMax, 1),
        "compress" => (Op::Compress, 2),
        "where" => (Op::Where, 1),
        "iota" => (Op::Iota, 1),
        "fill" => (Op::Fill, 2),
        "rand_i64" => (Op::RandI64, 2),
        "rand_f64" => (Op::RandF64, 2),
        "length" => (Op::Length, 1),
        "reduce" => return Some(Builtin::Reduce),
        "map" => return Some(Builtin::Map),
        "zip_map" => return Some(Builtin::ZipMap),
        "filter" => return Some(Builtin::Filter),
        _ => {
            if let Some(t) = name.strip_prefix("cast_").and_then(ScalarType::from_name) {
                (Op::Cast(t), 1)
            } else {
                let t = name.strip_prefix("load_").and_then(ScalarType::from_name)?;
                (Op::Load(t), 1)
            }
        }
    };
    Some(Builtin::Op(op, arity))
}

fn reduce_op(name: &str) -> Option<ReduceOp> {
    Some(match name {
        "add" => ReduceOp::Add,
        "mul" => ReduceOp::Mul,
        "min" => ReduceOp::Min,
        "max" => ReduceOp::Max,
        "and" => ReduceOp::And,
        "or" => ReduceOp::Or,
        "xor" => ReduceOp::Xor,
        _ => return None,
    })
}

fn reduce_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Add => "add",
        ReduceOp::Mul => "mul",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::And => "and",
        ReduceOp::Or => "or",
        ReduceOp::Xor => "xor",
    }
}

fn binop(op: BinOp) -> (Op, &'static str) {
    match op {
        BinOp::Add => (Op::Add, "+"),
        BinOp::Sub => (Op::Sub, "-"),
        BinOp::Mul => (Op::Mul, "*"),
        BinOp::Div => (Op::Div, "/"),
        BinOp::Eq => (Op::Eq, "=="),
        BinOp::Ne => (Op::Ne, "!="),
        BinOp::Lt => (Op::Lt, "<"),
        BinOp::Le => (Op::Le, "<="),
        BinOp::Gt => (Op::Gt, ">"),
        BinOp::Ge => (Op::Ge, ">="),
    }
}

/// Ops that apply independently per element, the only ones a lambda body may use.
fn is_elementwise(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Neg
            | Op::Abs
            | Op::Min
            | Op::Max
            | Op::Sqrt
            | Op::Eq
            | Op::Ne
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::And
            | Op::Or
            | Op::Not
            | Op::Select
            | Op::Cast(_)
    )
}

/// Binary ops whose operands must share an element type.
fn is_binary_elementwise(op: Op) -> bool {
    is_elementwise(op)
        && !matches!(
            op,
            Op::Neg | Op::Abs | Op::Sqrt | Op::Not | Op::Select | Op::Cast(_)
        )
}

fn elem(ty: ValueType) -> Option<ScalarType> {
    match ty {
        ValueType::Scalar(t) | ValueType::Array(t) => Some(t),
        ValueType::Str => None,
    }
}

fn value_type(t: ast::TypeExpr) -> ValueType {
    if t.array {
        ValueType::Array(t.elem)
    } else {
        ValueType::Scalar(t.elem)
    }
}

fn join_types(tys: &[ValueType]) -> String {
    tys.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Why a typed expression may not run inside a lambda, if it may not.
fn lift_violation(g: &Globals, e: &TExpr) -> Option<String> {
    let all = |es: &[TExpr]| es.iter().find_map(|x| lift_violation(g, x));
    match &e.kind {
        TExprKind::Lit(_) | TExprKind::Var(_) => None,
        TExprKind::Str(_) => Some("a string literal".into()),
        TExprKind::ArrayLit(_) => Some("an array literal".into()),
        TExprKind::Prim(op, args) => {
            if is_elementwise(*op) {
                all(args)
            } else {
                Some(format!("'{}'", op_name(*op)))
            }
        }
        TExprKind::Call(id, args) => all(args).or_else(|| lift_violation(g, &g.func(*id).body)),
        TExprKind::Map(..) => Some("'map'".into()),
        TExprKind::ZipMap(..) => Some("'zip_map'".into()),
        TExprKind::Filter(..) => Some("'filter'".into()),
        TExprKind::If(c, t, f) => [c, t, f].into_iter().find_map(|x| lift_violation(g, x)),
        TExprKind::Block(lets, r) => lets
            .iter()
            .find_map(|(_, x)| lift_violation(g, x))
            .or_else(|| lift_violation(g, r)),
    }
}

fn op_name(op: Op) -> String {
    match op {
        Op::Add => "+".into(),
        Op::Sub => "-".into(),
        Op::Mul => "*".into(),
        Op::Div => "/".into(),
        Op::Eq => "==".into(),
        Op::Ne => "!=".into(),
        Op::Lt => "<".into(),
        Op::Le => "<=".into(),
        Op::Gt => ">".into(),
        Op::Ge => ">=".into(),
        Op::Cast(t) => format!("cast_{}", t.name()),
        Op::Load(t) => format!("load_{}", t.name()),
        Op::Reduce(r) => format!("reduce({})", reduce_name(r)),
        other => format!("{other:?}").to_lowercase(),
    }
}

type Scope = HashMap<String, (VarId, ValueType)>;

struct Checker<'a> {
    reg: &'a Registry,
    g: &'a mut Globals,
    /// Innermost last. Holds parameters and block-local lets.
    scopes: Vec<Scope>,
    /// Function bodies see only their parameters and own lets.
    local_only: bool,
    in_lambda: bool,
    /// Parameters of the lambda being checked, the only variables it may use besides scalars.
    lambda_vars: Vec<VarId>,
    /// Functions called from the body being checked.
    calls: Vec<FnId>,
    /// Functions called from lambdas, whose bodies must satisfy the lambda rules.
    deferred: Vec<(FnId, Span)>,
}

impl<'a> Checker<'a> {
    fn new(reg: &'a Registry, g: &'a mut Globals) -> Checker<'a> {
        Checker {
            reg,
            g,
            scopes: Vec::new(),
            local_only: false,
            in_lambda: false,
            lambda_vars: Vec::new(),
            calls: Vec::new(),
            deferred: Vec::new(),
        }
    }

    fn run(mut self, prog: &ast::Program) -> Result<Vec<TStmt>, Error> {
        let new_fns = self.declare_fns(prog)?;
        let mut graph = Vec::new();
        for (id, def) in &new_fns {
            let body = self.check_fn_body(*id, def)?;
            self.g.fns[to_index(id.0)].body = body;
            graph.push((*id, std::mem::take(&mut self.calls)));
        }
        self.check_cycles(&new_fns, &graph)?;
        let mut stmts = Vec::new();
        // A host supplies inputs and receives exports by name, so each name may occur once.
        let (mut inputs, mut exports) = (HashSet::new(), HashSet::new());
        for item in &prog.items {
            let (seen, kind, name, span) = match item {
                ast::Item::Input(name, _, span) => (&mut inputs, "input", name, *span),
                ast::Item::Export(l) => (&mut exports, "export", &l.name, l.span),
                _ => continue,
            };
            if !seen.insert(name.as_str()) {
                return Err(Error::ty(
                    format!("{kind} '{name}' is declared twice"),
                    span,
                ));
            }
        }
        for item in &prog.items {
            if let Some(s) = self.check_item(item)? {
                stmts.push(s);
            }
        }
        for (id, span) in std::mem::take(&mut self.deferred) {
            if let Some(why) = lift_violation(self.g, &self.g.func(id).body.clone()) {
                let name = &self.g.func(id).name;
                return Err(Error::ty(
                    format!("'{name}' cannot be used inside a lambda, its body uses {why}"),
                    span,
                ));
            }
        }
        Ok(stmts)
    }

    /// Registers every function of the program first, so bodies may refer to later functions.
    fn declare_fns<'p>(
        &mut self,
        prog: &'p ast::Program,
    ) -> Result<Vec<(FnId, &'p ast::FnDef)>, Error> {
        let mut out: Vec<(FnId, &ast::FnDef)> = Vec::new();
        for item in &prog.items {
            let ast::Item::Fn(def) = item else { continue };
            if builtin(&def.name).is_some() {
                return Err(Error::ty(
                    format!("cannot define function '{}', it is a builtin", def.name),
                    def.span,
                ));
            }
            if out.iter().any(|(_, d)| d.name == def.name) {
                return Err(Error::ty(
                    format!("function '{}' is defined twice", def.name),
                    def.span,
                ));
            }
            let params = def
                .params
                .iter()
                .map(|(n, t)| (self.g.fresh_var(n), value_type(*t)))
                .collect();
            let ret = value_type(def.ret);
            let id = FnId(u32::try_from(self.g.fns.len()).expect("function count fits in u32"));
            // The body is a placeholder until it is checked.
            let body = TExpr {
                kind: TExprKind::Str("".into()),
                ty: ret,
                span: def.span,
            };
            self.g.fns.push(TFn {
                name: def.name.clone(),
                params,
                ret,
                body,
            });
            self.g.fn_names.insert(def.name.clone(), id);
            out.push((id, def));
        }
        Ok(out)
    }

    fn check_fn_body(&mut self, id: FnId, def: &ast::FnDef) -> Result<TExpr, Error> {
        let f = self.g.func(id);
        let (ret, params) = (f.ret, f.params.clone());
        self.local_only = true;
        self.scopes = vec![
            def.params
                .iter()
                .zip(&params)
                .map(|((n, _), (v, t))| (n.clone(), (*v, *t)))
                .collect(),
        ];
        let body = self.check_block(&def.body);
        self.scopes.clear();
        self.local_only = false;
        let body = body?;
        if body.ty != ret {
            return Err(Error::ty(
                format!(
                    "function '{}' returns {} but is declared to return {ret}",
                    def.name, body.ty
                ),
                def.span,
            ));
        }
        Ok(body)
    }

    fn check_cycles(
        &self,
        new_fns: &[(FnId, &ast::FnDef)],
        graph: &[(FnId, Vec<FnId>)],
    ) -> Result<(), Error> {
        // 0 unvisited, 1 on the stack, 2 done.
        let mut state: HashMap<FnId, u8> = HashMap::new();
        fn visit(
            id: FnId,
            graph: &[(FnId, Vec<FnId>)],
            state: &mut HashMap<FnId, u8>,
        ) -> Option<FnId> {
            match state.get(&id) {
                Some(1) => return Some(id),
                Some(_) => return None,
                None => {}
            }
            state.insert(id, 1);
            // Functions from earlier inputs are absent from the graph and cannot call back.
            if let Some((_, callees)) = graph.iter().find(|(f, _)| *f == id) {
                for c in callees {
                    if let Some(hit) = visit(*c, graph, state) {
                        return Some(hit);
                    }
                }
            }
            state.insert(id, 2);
            None
        }
        for (id, _) in new_fns {
            if let Some(hit) = visit(*id, graph, &mut state) {
                let def = new_fns
                    .iter()
                    .find(|(f, _)| *f == hit)
                    .expect("cycles are among new functions")
                    .1;
                return Err(Error::ty(
                    format!("recursive function '{}'", def.name),
                    def.span,
                ));
            }
        }
        Ok(())
    }

    fn check_item(&mut self, item: &ast::Item) -> Result<Option<TStmt>, Error> {
        Ok(Some(match item {
            ast::Item::Fn(_) => return Ok(None),
            ast::Item::Let(l) => {
                let value = self.check_let_value(l)?;
                let var = self.g.declare(&l.name, value.ty);
                TStmt::Let {
                    var,
                    value,
                    span: l.span,
                }
            }
            ast::Item::Export(l) => {
                let value = self.check_let_value(l)?;
                let var = self.g.declare(&l.name, value.ty);
                TStmt::Export {
                    var,
                    value,
                    span: l.span,
                }
            }
            ast::Item::Input(name, t, span) => {
                let ty = value_type(*t);
                let var = self.g.declare(name, ty);
                TStmt::Input {
                    var,
                    ty,
                    span: *span,
                }
            }
            ast::Item::Print(e, span) => TStmt::Print {
                value: self.check_expr(e)?,
                span: *span,
            },
            ast::Item::Expr(e, span) => TStmt::Expr {
                value: self.check_expr(e)?,
                span: *span,
            },
        }))
    }

    /// Checks a let's value against its annotation, if any.
    fn check_let_value(&mut self, l: &ast::LetStmt) -> Result<TExpr, Error> {
        let value = self.check_expr(&l.value)?;
        if let Some(t) = l.ty {
            let declared = value_type(t);
            if declared != value.ty {
                return Err(Error::ty(
                    format!(
                        "'{}' is declared {declared} but its value is {}",
                        l.name, value.ty
                    ),
                    l.span,
                ));
            }
        }
        Ok(value)
    }

    fn lookup_var(&self, name: &str) -> Option<(VarId, ValueType)> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name).copied())
            .or_else(|| {
                if self.local_only {
                    None
                } else {
                    self.g.lookup(name)
                }
            })
    }

    fn check_block(&mut self, b: &ast::Block) -> Result<TExpr, Error> {
        self.scopes.push(Scope::new());
        let r = self.check_block_inner(b);
        self.scopes.pop();
        r
    }

    fn check_block_inner(&mut self, b: &ast::Block) -> Result<TExpr, Error> {
        let mut lets = Vec::new();
        for l in &b.lets {
            let value = self.check_let_value(l)?;
            let var = self.g.fresh_var(&l.name);
            self.scopes
                .last_mut()
                .expect("a block scope is open")
                .insert(l.name.clone(), (var, value.ty));
            lets.push((var, value));
        }
        let result = self.check_expr(&b.result)?;
        if lets.is_empty() {
            return Ok(result);
        }
        Ok(TExpr {
            ty: result.ty,
            span: result.span,
            kind: TExprKind::Block(lets, Box::new(result)),
        })
    }

    fn check_expr(&mut self, e: &ast::Expr) -> Result<TExpr, Error> {
        let span = e.span;
        let mk = |kind, ty| TExpr { kind, ty, span };
        match &e.kind {
            ExprKind::Lit(s) => Ok(mk(TExprKind::Lit(s.clone()), ValueType::Scalar(s.ty()))),
            ExprKind::Str(_) => Err(Error::ty(
                "a string is only allowed as the argument of load_*",
                span,
            )),
            ExprKind::ArrayLit(items) => self.check_array_lit(items, span),
            ExprKind::Name(n) => {
                let Some((var, ty)) = self.lookup_var(n) else {
                    return Err(Error::ty(format!("unknown name '{n}'"), span));
                };
                if self.in_lambda
                    && !self.lambda_vars.contains(&var)
                    && matches!(ty, ValueType::Array(_))
                {
                    return Err(Error::ty(format!("lambda captures array '{n}'"), span));
                }
                Ok(mk(TExprKind::Var(var), ty))
            }
            ExprKind::Call(name, args) => self.check_call(name, args, span),
            ExprKind::Binary(op, l, r) => {
                let (op, sym) = binop(*op);
                let args = vec![self.check_expr(l)?, self.check_expr(r)?];
                self.prim(op, sym, args, span, true)
            }
            ExprKind::If(c, t, f) => {
                let cond = self.check_expr(c)?;
                if !matches!(
                    cond.ty,
                    ValueType::Scalar(ScalarType::Bool) | ValueType::Array(ScalarType::Bool)
                ) {
                    return Err(Error::ty(
                        format!("if condition must be bool or bool[], got {}", cond.ty),
                        c.span,
                    ));
                }
                let (t, f) = (self.check_block(t)?, self.check_block(f)?);
                if t.ty != f.ty {
                    return Err(Error::ty(
                        format!("if branch types differ: {} and {}", t.ty, f.ty),
                        span,
                    ));
                }
                // A scalar condition picks one branch, so the result has the branch type.
                // A mask condition selects per element, so the registry's `select` kernel
                // decides the result type, an array even when both branches are scalars.
                let ty = if cond.ty == ValueType::Scalar(ScalarType::Bool) {
                    t.ty
                } else {
                    let tys = [cond.ty, t.ty, f.ty];
                    let Some(id) = self.reg.lookup(Op::Select, &tys) else {
                        return Err(Error::ty(
                            format!("no kernel for 'if' on {}", join_types(&tys)),
                            span,
                        ));
                    };
                    self.reg.get(id).output
                };
                Ok(mk(
                    TExprKind::If(Box::new(cond), Box::new(t), Box::new(f)),
                    ty,
                ))
            }
            ExprKind::Lambda(..) => Err(Error::ty(
                "a lambda is only allowed as an argument of map, zip_map, or filter",
                span,
            )),
            ExprKind::Block(b) => self.check_block(b),
        }
    }

    fn check_array_lit(&mut self, items: &[ast::Expr], span: Span) -> Result<TExpr, Error> {
        if self.in_lambda {
            return Err(Error::ty(
                "an array literal cannot be used inside a lambda",
                span,
            ));
        }
        let Some(first) = items.first() else {
            return Err(Error::ty(
                "an array literal needs at least one element",
                span,
            ));
        };
        let mut elems = Vec::new();
        for it in items {
            elems.push(self.check_expr(it)?);
        }
        let ValueType::Scalar(t) = elems[0].ty else {
            return Err(Error::ty("array elements must be scalars", first.span));
        };
        for (el, it) in elems.iter().zip(items) {
            if el.ty != ValueType::Scalar(t) {
                return Err(Error::ty(
                    format!(
                        "array elements must all have the same type, found {} and {}",
                        ValueType::Scalar(t),
                        el.ty
                    ),
                    it.span,
                ));
            }
        }
        Ok(TExpr {
            kind: TExprKind::ArrayLit(elems),
            ty: ValueType::Array(t),
            span,
        })
    }

    /// Resolves an op on typed arguments through the registry.
    fn prim(
        &self,
        op: Op,
        sym: &str,
        args: Vec<TExpr>,
        span: Span,
        infix: bool,
    ) -> Result<TExpr, Error> {
        let tys: Vec<ValueType> = args.iter().map(|a| a.ty).collect();
        let Some(id) = self.reg.lookup(op, &tys) else {
            if is_binary_elementwise(op)
                && let [a, b] = tys[..]
                && let (Some(x), Some(y)) = (elem(a), elem(b))
                && x != y
            {
                let shown = if infix {
                    format!("{a} {sym} {b}")
                } else {
                    format!("{sym}({a}, {b})")
                };
                return Err(Error::ty(format!("no implicit conversion: {shown}"), span));
            }
            return Err(Error::ty(
                format!("no kernel for '{sym}' on {}", join_types(&tys)),
                span,
            ));
        };
        Ok(TExpr {
            kind: TExprKind::Prim(op, args),
            ty: self.reg.get(id).output,
            span,
        })
    }

    fn check_call(&mut self, name: &str, args: &[ast::Expr], span: Span) -> Result<TExpr, Error> {
        match builtin(name) {
            Some(Builtin::Op(op, arity)) => {
                if self.in_lambda && !is_elementwise(op) {
                    return Err(Error::ty(
                        format!("'{name}' cannot be used inside a lambda"),
                        span,
                    ));
                }
                if let Op::Load(_) = op
                    && (self.local_only || self.in_lambda)
                {
                    return Err(Error::ty(
                        format!("'{name}' is only allowed at top level"),
                        span,
                    ));
                }
                if args.len() != arity {
                    return Err(Error::ty(
                        format!("'{name}' takes {arity} arguments, got {}", args.len()),
                        span,
                    ));
                }
                let targs = if let Op::Load(_) = op {
                    let ExprKind::Str(s) = &args[0].kind else {
                        return Err(Error::ty(
                            format!("'{name}' takes a string literal"),
                            args[0].span,
                        ));
                    };
                    vec![TExpr {
                        kind: TExprKind::Str(s.as_str().into()),
                        ty: ValueType::Str,
                        span: args[0].span,
                    }]
                } else {
                    self.check_args(args)?
                };
                self.prim(op, name, targs, span, false)
            }
            Some(Builtin::Reduce) => {
                if self.in_lambda {
                    return Err(Error::ty(
                        format!("'{name}' cannot be used inside a lambda"),
                        span,
                    ));
                }
                let [x, op] = args else {
                    return Err(Error::ty(
                        format!("'reduce' takes 2 arguments, got {}", args.len()),
                        span,
                    ));
                };
                let ExprKind::Name(n) = &op.kind else {
                    return Err(Error::ty(
                        "reduce takes an operation name such as add or max",
                        op.span,
                    ));
                };
                let Some(r) = reduce_op(n) else {
                    return Err(Error::ty(
                        format!("unknown reduce operation '{n}'"),
                        op.span,
                    ));
                };
                let x = self.check_expr(x)?;
                self.prim(Op::Reduce(r), "reduce", vec![x], span, false)
            }
            Some(Builtin::Map) => self.check_higher(name, args, 1, span),
            Some(Builtin::ZipMap) => self.check_higher(name, args, 2, span),
            Some(Builtin::Filter) => self.check_higher(name, args, 1, span),
            None => {
                let Some(id) = self.g.lookup_fn(name) else {
                    return Err(Error::ty(format!("unknown function '{name}'"), span));
                };
                let targs = self.check_args(args)?;
                self.call_user(id, targs, span)
            }
        }
    }

    fn check_args(&mut self, args: &[ast::Expr]) -> Result<Vec<TExpr>, Error> {
        args.iter().map(|a| self.check_expr(a)).collect()
    }

    fn call_user(&mut self, id: FnId, args: Vec<TExpr>, span: Span) -> Result<TExpr, Error> {
        let f = self.g.func(id);
        let want: Vec<ValueType> = f.params.iter().map(|(_, t)| *t).collect();
        let got: Vec<ValueType> = args.iter().map(|a| a.ty).collect();
        if want != got {
            return Err(Error::ty(
                format!(
                    "function '{}' expects ({}), got ({})",
                    f.name,
                    join_types(&want),
                    join_types(&got)
                ),
                span,
            ));
        }
        let ty = f.ret;
        self.calls.push(id);
        if self.in_lambda {
            self.deferred.push((id, span));
        }
        Ok(TExpr {
            kind: TExprKind::Call(id, args),
            ty,
            span,
        })
    }

    /// Checks `map`, `zip_map`, and `filter`. The last argument is the function, the first
    /// `arrays` arguments are the arrays it maps over.
    fn check_higher(
        &mut self,
        name: &str,
        args: &[ast::Expr],
        arrays: usize,
        span: Span,
    ) -> Result<TExpr, Error> {
        if self.in_lambda {
            return Err(Error::ty(
                format!("'{name}' cannot be used inside a lambda"),
                span,
            ));
        }
        if args.len() != arrays + 1 {
            return Err(Error::ty(
                format!(
                    "'{name}' takes {} arguments, got {}",
                    arrays + 1,
                    args.len()
                ),
                span,
            ));
        }
        let mut xs = Vec::new();
        let mut elems = Vec::new();
        for a in &args[..arrays] {
            let x = self.check_expr(a)?;
            let ValueType::Array(t) = x.ty else {
                return Err(Error::ty(
                    format!("'{name}' needs an array, got {}", x.ty),
                    a.span,
                ));
            };
            elems.push(t);
            xs.push(x);
        }
        let lambda = self.check_lambda_arg(&args[arrays], &elems)?;
        let ValueType::Scalar(ret) = lambda.body.ty else {
            return Err(Error::ty(
                format!("'{name}' function must return a scalar"),
                args[arrays].span,
            ));
        };
        let mut xs = xs.into_iter().map(Box::new);
        let (kind, ty) = match name {
            "map" => (
                TExprKind::Map(xs.next().expect("one array"), lambda),
                ValueType::Array(ret),
            ),
            "zip_map" => (
                TExprKind::ZipMap(
                    xs.next().expect("two arrays"),
                    xs.next().expect("two arrays"),
                    lambda,
                ),
                ValueType::Array(ret),
            ),
            _ => {
                if ret != ScalarType::Bool {
                    return Err(Error::ty(
                        format!("filter predicate must return bool, got {}", ret.name()),
                        args[arrays].span,
                    ));
                }
                let x = xs.next().expect("one array");
                let ty = x.ty;
                (TExprKind::Filter(x, lambda), ty)
            }
        };
        Ok(TExpr { kind, ty, span })
    }

    /// Checks a lambda or a function name used as a lambda. Parameters get the scalar element types.
    fn check_lambda_arg(
        &mut self,
        arg: &ast::Expr,
        elems: &[ScalarType],
    ) -> Result<TLambda, Error> {
        match &arg.kind {
            ExprKind::Lambda(params, body) => {
                if params.len() != elems.len() {
                    return Err(Error::ty(
                        format!(
                            "lambda needs {} parameter(s), got {}",
                            elems.len(),
                            params.len()
                        ),
                        arg.span,
                    ));
                }
                let mut scope = Scope::new();
                let mut vars = Vec::new();
                for (p, t) in params.iter().zip(elems) {
                    let v = self.g.fresh_var(p);
                    scope.insert(p.clone(), (v, ValueType::Scalar(*t)));
                    vars.push(v);
                }
                self.scopes.push(scope);
                let saved = (self.in_lambda, self.lambda_vars.len());
                self.in_lambda = true;
                self.lambda_vars.extend(&vars);
                let body = self.check_expr(body);
                self.in_lambda = saved.0;
                self.lambda_vars.truncate(saved.1);
                self.scopes.pop();
                Ok(TLambda {
                    params: vars,
                    body: Box::new(body?),
                })
            }
            ExprKind::Name(n) => {
                let Some(id) = self.g.lookup_fn(n) else {
                    return Err(Error::ty(format!("unknown function '{n}'"), arg.span));
                };
                let f = self.g.func(id);
                if f.params.len() != elems.len() {
                    return Err(Error::ty(
                        format!(
                            "function '{n}' takes {} parameter(s), expected {}",
                            f.params.len(),
                            elems.len()
                        ),
                        arg.span,
                    ));
                }
                let names: Vec<(String, ValueType)> = f
                    .params
                    .iter()
                    .map(|(v, t)| (self.g.name(*v).to_string(), *t))
                    .collect();
                let mut vars = Vec::new();
                let mut targs = Vec::new();
                for ((pname, pty), t) in names.iter().zip(elems) {
                    if *pty != ValueType::Scalar(*t) {
                        return Err(Error::ty(
                            format!(
                                "function '{n}' parameter '{pname}' is {pty}, expected {}",
                                ValueType::Scalar(*t)
                            ),
                            arg.span,
                        ));
                    }
                    let v = self.g.fresh_var(pname);
                    vars.push(v);
                    targs.push(TExpr {
                        kind: TExprKind::Var(v),
                        ty: *pty,
                        span: arg.span,
                    });
                }
                // The function's own liftability is checked once all bodies exist.
                self.calls.push(id);
                self.deferred.push((id, arg.span));
                let ty = self.g.func(id).ret;
                let body = TExpr {
                    kind: TExprKind::Call(id, targs),
                    ty,
                    span: arg.span,
                };
                Ok(TLambda {
                    params: vars,
                    body: Box::new(body),
                })
            }
            _ => Err(Error::ty("expected a lambda or a function name", arg.span)),
        }
    }
}
