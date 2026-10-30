//! The typed AST produced by the checker. Lambdas, calls, and `if` stay intact for lowering.

use std::sync::Arc;

use crate::error::Span;
use crate::ops::Op;
use crate::value::{Scalar, ValueType};

/// A variable, unique within one `Globals`. Names live in `Globals::name`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub u32);

/// A user function, an index into the functions held by `Globals`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FnId(pub u32);

#[derive(Clone, Debug, PartialEq)]
pub struct TProgram {
    /// Functions live in `Globals`.
    pub stmts: Vec<TStmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TFn {
    pub name: String,
    pub params: Vec<(VarId, ValueType)>,
    pub ret: ValueType,
    pub body: TExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TStmt {
    Let { var: VarId, value: TExpr, span: Span },
    /// Declares a top-level name of the type with no value. Top level only.
    Input { var: VarId, ty: ValueType, span: Span },
    /// Binds a result like `Let`. Top level only.
    Export { var: VarId, value: TExpr, span: Span },
    Print { value: TExpr, span: Span },
    Expr { value: TExpr, span: Span },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TExpr {
    pub kind: TExprKind,
    pub ty: ValueType,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TExprKind {
    Lit(Scalar),
    Str(Arc<str>),
    ArrayLit(Vec<TExpr>),
    Var(VarId),
    Prim(Op, Vec<TExpr>),
    Call(FnId, Vec<TExpr>),
    Map(Box<TExpr>, TLambda),
    ZipMap(Box<TExpr>, Box<TExpr>, TLambda),
    Filter(Box<TExpr>, TLambda),
    If(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    Block(Vec<(VarId, TExpr)>, Box<TExpr>),
}

/// A lambda whose parameters are scalars. The body is elementwise by construction.
#[derive(Clone, Debug, PartialEq)]
pub struct TLambda {
    pub params: Vec<VarId>,
    pub body: Box<TExpr>,
}
