//! Binding-time analysis of lambda bodies. Lifting substitutes a lambda's parameters with
//! the argument arrays, so an expression that depends on a parameter becomes an array
//! (varying) while every other expression stays a scalar (uniform). This module computes
//! the type each expression of a lambda body has after lifting and resolves every op on
//! those types with `Registry::lookup_lifted`, the rule lowering uses, and builds the
//! operations around the body with `lower::higher`, so every lambda the checker accepts
//! lowers.

use std::collections::HashMap;

use crate::check::{FnId, Globals, TExpr, TExprKind, TLambda, VarId, join_types, op_name};
use crate::error::{Error, Span};
use crate::kernels::Registry;
use crate::lower::higher::{self, Build, Higher};
use crate::lower::no_kernel;
use crate::ops::Op;
use crate::value::ValueType;

/// Checks lambda bodies on their lifted types. User functions are instantiated once per
/// pattern of lifted argument types.
pub(super) struct Lifting<'a> {
    reg: &'a Registry,
    g: &'a Globals,
    /// The lifted result type of a function body per lifted argument types, or the error
    /// in the body, located in the body.
    memo: HashMap<(FnId, Vec<ValueType>), Result<ValueType, Error>>,
}

impl<'a> Lifting<'a> {
    pub(super) fn new(reg: &'a Registry, g: &'a Globals) -> Lifting<'a> {
        Lifting {
            reg,
            g,
            memo: HashMap::new(),
        }
    }

    /// Checks that the higher-order call `kind` at `span` lowers when it applies `lam` to
    /// arrays of the types `xs`. The body must already satisfy the lambda rules, and every
    /// function it calls must exist, be non-recursive, and be liftable.
    pub(super) fn lambda(
        &mut self,
        kind: Higher,
        lam: &TLambda,
        xs: &[ValueType],
        span: Span,
    ) -> Result<(), Error> {
        let params = higher::params(self, kind, xs.to_vec(), span)?;
        let mut env: HashMap<VarId, ValueType> = lam
            .params
            .iter()
            .copied()
            .zip(params.iter().copied())
            .collect();
        let body = self.expr(&lam.body, &mut env)?;
        higher::result(self, kind, xs, &params, body, span)?;
        Ok(())
    }

    /// The lifted type of `e`. `env` holds the lifted types of parameters and block lets.
    /// Any other variable is a captured scalar, which stays as it is.
    fn expr(&mut self, e: &TExpr, env: &mut HashMap<VarId, ValueType>) -> Result<ValueType, Error> {
        match &e.kind {
            TExprKind::Lit(_) => Ok(e.ty),
            TExprKind::Var(v) => Ok(env.get(v).copied().unwrap_or(e.ty)),
            TExprKind::Prim(op, args) => {
                let tys = self.exprs(args, env)?;
                self.op(*op, &op_name(*op), &tys, e.span)
            }
            // Lowering turns every `if` of a lifted body into `select`, computing both
            // branches, because the condition is per element.
            TExprKind::If(c, t, f) => {
                let tys = [self.expr(c, env)?, self.expr(t, env)?, self.expr(f, env)?];
                self.op(Op::Select, "if", &tys, e.span)
            }
            TExprKind::Block(lets, result) => {
                for (v, value) in lets {
                    let ty = self.expr(value, env)?;
                    env.insert(*v, ty);
                }
                self.expr(result, env)
            }
            TExprKind::Call(id, args) => {
                let tys = self.exprs(args, env)?;
                self.call(*id, tys, e.span)
            }
            TExprKind::Str(_)
            | TExprKind::ArrayLit(_)
            | TExprKind::Map(..)
            | TExprKind::ZipMap(..)
            | TExprKind::Filter(..) => {
                unreachable!("the lambda rules reject this in a lambda body")
            }
        }
    }

    fn exprs(
        &mut self,
        es: &[TExpr],
        env: &mut HashMap<VarId, ValueType>,
    ) -> Result<Vec<ValueType>, Error> {
        es.iter().map(|x| self.expr(x, env)).collect()
    }

    /// The output type of `op` on lifted operand types, or the registry's "no kernel" error.
    fn op(&self, op: Op, name: &str, tys: &[ValueType], span: Span) -> Result<ValueType, Error> {
        match self.reg.lookup_lifted(op, tys) {
            Ok(l) => Ok(self.reg.get(l.kernel).output),
            Err(shown) => Err(Error::ty(
                format!("no kernel for '{name}' on {}", join_types(&shown)),
                span,
            )),
        }
    }

    /// The lifted result type of calling `id` at `span` with arguments of lifted types
    /// `args`. An error in the body is reported at `span` and names the function and where
    /// in its body the error is.
    fn call(&mut self, id: FnId, args: Vec<ValueType>, span: Span) -> Result<ValueType, Error> {
        let g = self.g;
        let f = g.func(id);
        let key = (id, args);
        let r = match self.memo.get(&key) {
            Some(r) => r.clone(),
            None => {
                // Lowering inlines the body with each parameter bound to its argument, and
                // a function body sees only its parameters and own lets.
                let mut env = f
                    .params
                    .iter()
                    .map(|(v, _)| *v)
                    .zip(key.1.iter().copied())
                    .collect();
                // Recursion is rejected before, so this terminates.
                let r = self.expr(&f.body, &mut env);
                self.memo.insert(key, r.clone());
                r
            }
        };
        r.map_err(|e| {
            let at = e
                .span
                .map(|s| format!(" at {}:{}", s.line, s.col))
                .unwrap_or_default();
            Error::ty(format!("{} (in '{}'{at})", e.message, f.name), span)
        })
    }
}

impl Build for Lifting<'_> {
    type V = ValueType;

    fn ty(v: &ValueType) -> ValueType {
        *v
    }

    fn apply(&self, op: Op, args: Vec<ValueType>, span: Span) -> Result<ValueType, Error> {
        match self.reg.lookup_lifted(op, &args) {
            Ok(l) => Ok(self.reg.get(l.kernel).output),
            Err(shown) => Err(no_kernel(op, &shown, span)),
        }
    }

    fn apply_exact(&self, op: Op, args: Vec<ValueType>, span: Span) -> Result<ValueType, Error> {
        match self.reg.lookup(op, &args) {
            Some(id) => Ok(self.reg.get(id).output),
            None => Err(no_kernel(op, &args, span)),
        }
    }
}
