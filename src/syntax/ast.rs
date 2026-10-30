use crate::error::Span;
use crate::value::{Scalar, ScalarType};

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Fn(FnDef),
    Let(LetStmt),
    /// `input name: type;`
    Input(String, TypeExpr, Span),
    /// `export name = expr;`, always with `ty: None`.
    Export(LetStmt),
    Print(Expr, Span),
    Expr(Expr, Span),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<(String, TypeExpr)>,
    pub ret: TypeExpr,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LetStmt {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeExpr {
    pub elem: ScalarType,
    pub array: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub lets: Vec<LetStmt>,
    pub result: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Lit(Scalar),
    Str(String),
    ArrayLit(Vec<Expr>),
    Name(String),
    Call(String, Vec<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    If(Box<Expr>, Block, Block),
    Lambda(Vec<String>, Box<Expr>),
    Block(Block),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub fn is_comparison(self) -> bool {
        !matches!(self, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
    }
}
