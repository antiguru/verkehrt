use super::ast::*;
use super::lexer::{Tok, Token, lex};
use crate::error::{Error, Span};
use crate::value::{Scalar, ScalarType};

/// Parses a whole program. Every statement ends with `;`.
pub fn parse(src: &str) -> Result<Program, Error> {
    let mut p = Parser { toks: lex(src)?, pos: 0 };
    let mut items = Vec::new();
    while p.peek() != &Tok::Eof {
        items.push(p.item()?);
    }
    Ok(Program { items })
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Ident(s) => format!("`{s}`"),
        Tok::Num { text, .. } => format!("`{text}`"),
        Tok::Str(_) => "a string".into(),
        Tok::Eof => "end of input".into(),
        other => format!("`{}`", symbol(other)),
    }
}

fn symbol(t: &Tok) -> &'static str {
    match t {
        Tok::LParen => "(",
        Tok::RParen => ")",
        Tok::LBrace => "{",
        Tok::RBrace => "}",
        Tok::LBracket => "[",
        Tok::RBracket => "]",
        Tok::Comma => ",",
        Tok::Semi => ";",
        Tok::Colon => ":",
        Tok::Arrow => "->",
        Tok::Pipe => "|",
        Tok::Assign => "=",
        Tok::Plus => "+",
        Tok::Minus => "-",
        Tok::Star => "*",
        Tok::Slash => "/",
        Tok::EqEq => "==",
        Tok::Ne => "!=",
        Tok::Lt => "<",
        Tok::Le => "<=",
        Tok::Gt => ">",
        Tok::Ge => ">=",
        _ => "?",
    }
}

fn binop(t: &Tok) -> Option<BinOp> {
    Some(match t {
        Tok::Plus => BinOp::Add,
        Tok::Minus => BinOp::Sub,
        Tok::Star => BinOp::Mul,
        Tok::Slash => BinOp::Div,
        Tok::EqEq => BinOp::Eq,
        Tok::Ne => BinOp::Ne,
        Tok::Lt => BinOp::Lt,
        Tok::Le => BinOp::Le,
        Tok::Gt => BinOp::Gt,
        Tok::Ge => BinOp::Ge,
        _ => return None,
    })
}

/// Converts a numeric literal to a scalar of its suffix type (`i64` or `f64` by default).
/// `neg` prepends a minus sign so that the range check sees the signed value.
fn literal(
    text: &str,
    float: bool,
    suffix: Option<&str>,
    neg: bool,
    span: Span,
) -> Result<Scalar, Error> {
    let ty = match suffix {
        Some(s) => ScalarType::from_name(s)
            .filter(|t| *t != ScalarType::Bool)
            .ok_or_else(|| Error::parse(format!("unknown literal suffix `{s}`"), span))?,
        None if float => ScalarType::F64,
        None => ScalarType::I64,
    };
    if float && !ty.is_float() {
        return Err(Error::parse(
            format!("float literal `{text}` cannot have integer suffix `{}`", ty.name()),
            span,
        ));
    }
    let s = if neg { format!("-{text}") } else { text.to_string() };
    let out_of_range =
        || Error::parse(format!("literal `{s}` is out of range for {}", ty.name()), span);
    macro_rules! int {
        ($variant:ident) => {
            s.parse().map(Scalar::$variant).map_err(|_| out_of_range())
        };
    }
    match ty {
        ScalarType::I8 => int!(I8),
        ScalarType::I16 => int!(I16),
        ScalarType::I32 => int!(I32),
        ScalarType::I64 => int!(I64),
        ScalarType::U8 => int!(U8),
        ScalarType::U16 => int!(U16),
        ScalarType::U32 => int!(U32),
        ScalarType::U64 => int!(U64),
        // Rust parses overflow to infinity, which no literal may denote.
        ScalarType::F32 => s
            .parse::<f32>()
            .ok()
            .filter(|v| v.is_finite())
            .map(Scalar::F32)
            .ok_or_else(out_of_range),
        ScalarType::F64 => s
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(Scalar::F64)
            .ok_or_else(out_of_range),
        ScalarType::Bool => unreachable!("bool suffix rejected above"),
    }
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if t.tok != Tok::Eof {
            self.pos += 1;
        }
        t
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn unexpected(&self, expected: &str) -> Error {
        Error::parse(format!("expected {expected}, found {}", describe(self.peek())), self.span())
    }

    fn expect(&mut self, t: Tok) -> Result<Span, Error> {
        if self.peek() == &t {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(&format!("`{}`", symbol(&t))))
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<Span, Error> {
        if self.is_kw(kw) { Ok(self.bump().span) } else { Err(self.unexpected(&format!("`{kw}`"))) }
    }

    fn ident(&mut self, what: &str) -> Result<String, Error> {
        match self.peek() {
            Tok::Ident(s) if !is_reserved(s) => {
                let s = s.clone();
                self.bump();
                Ok(s)
            }
            _ => Err(self.unexpected(what)),
        }
    }

    fn item(&mut self) -> Result<Item, Error> {
        let span = self.span();
        if self.is_kw("fn") {
            return self.fn_def().map(Item::Fn);
        }
        if self.is_kw("let") {
            return self.let_stmt().map(Item::Let);
        }
        if self.is_kw("export") {
            self.bump();
            let name = self.ident("an export name")?;
            self.expect(Tok::Assign)?;
            let value = self.expr()?;
            self.expect(Tok::Semi)?;
            return Ok(Item::Export(LetStmt { name, ty: None, value, span }));
        }
        if self.is_kw("input") {
            self.bump();
            let name = self.ident("an input name")?;
            self.expect(Tok::Colon)?;
            let ty = self.type_expr()?;
            self.expect(Tok::Semi)?;
            return Ok(Item::Input(name, ty, span));
        }
        // `print(expr);` is a statement. `print` is only special when a `(` follows.
        if self.is_kw("print") && self.peek_at(1) == &Tok::LParen {
            self.bump();
            self.bump();
            let e = self.expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Item::Print(e, span));
        }
        let e = self.expr()?;
        self.expect(Tok::Semi)?;
        Ok(Item::Expr(e, span))
    }

    fn type_expr(&mut self) -> Result<TypeExpr, Error> {
        let span = self.span();
        let Tok::Ident(n) = self.peek().clone() else {
            return Err(self.unexpected("a type"));
        };
        let elem = ScalarType::from_name(&n)
            .ok_or_else(|| Error::parse(format!("unknown type `{n}`"), span))?;
        self.bump();
        let array = if self.peek() == &Tok::LBracket {
            self.bump();
            self.expect(Tok::RBracket)?;
            true
        } else {
            false
        };
        Ok(TypeExpr { elem, array })
    }

    fn fn_def(&mut self) -> Result<FnDef, Error> {
        let span = self.expect_kw("fn")?;
        let name = self.ident("a function name")?;
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        while self.peek() != &Tok::RParen {
            let pname = self.ident("a parameter name")?;
            self.expect(Tok::Colon)?;
            params.push((pname, self.type_expr()?));
            if self.peek() != &Tok::RParen {
                self.expect(Tok::Comma)?;
            }
        }
        self.expect(Tok::RParen)?;
        self.expect(Tok::Arrow)?;
        let ret = self.type_expr()?;
        let body = self.block()?;
        Ok(FnDef { name, params, ret, body, span })
    }

    fn let_stmt(&mut self) -> Result<LetStmt, Error> {
        let span = self.expect_kw("let")?;
        let name = self.ident("a binding name")?;
        let ty = if self.peek() == &Tok::Colon {
            self.bump();
            Some(self.type_expr()?)
        } else {
            None
        };
        self.expect(Tok::Assign)?;
        let value = self.expr()?;
        self.expect(Tok::Semi)?;
        Ok(LetStmt { name, ty, value, span })
    }

    /// `{ let a = ...; ...; expr }`
    fn block(&mut self) -> Result<Block, Error> {
        self.expect(Tok::LBrace)?;
        let mut lets = Vec::new();
        while self.is_kw("let") {
            lets.push(self.let_stmt()?);
        }
        let result = Box::new(self.expr()?);
        self.expect(Tok::RBrace)?;
        Ok(Block { lets, result })
    }

    /// One operand, then a run of the same binary operator folded left.
    /// A different operator after the run, or a second comparison, is an error.
    fn expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.operand()?;
        let Some(op) = binop(self.peek()) else {
            return Ok(lhs);
        };
        while let Some(next) = binop(self.peek()) {
            if next != op {
                return Err(self.paren_error());
            }
            // A binary node carries the operator's span, so type errors point at the operator.
            let span = self.bump().span;
            let rhs = self.operand()?;
            lhs = Expr { kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)), span };
            // Comparisons never chain, so any operator after one comparison is an error.
            if op.is_comparison() {
                if binop(self.peek()).is_some() {
                    return Err(self.paren_error());
                }
                break;
            }
        }
        Ok(lhs)
    }

    fn paren_error(&self) -> Error {
        Error::parse(
            "mixing operators needs parentheses, for example `a + (b * c)`, and comparisons cannot be chained",
            self.span(),
        )
    }

    fn operand(&mut self) -> Result<Expr, Error> {
        let t = self.bump();
        let span = t.span;
        let kind = match t.tok {
            Tok::Num { text, float, suffix } => {
                ExprKind::Lit(literal(&text, float, suffix.as_deref(), false, span)?)
            }
            // A minus directly adjacent to a number is a negative literal.
            Tok::Minus => {
                let adjacent = self.span() == Span { line: span.line, col: span.col + 1 };
                match self.peek().clone() {
                    Tok::Num { text, float, suffix } if adjacent => {
                        self.bump();
                        ExprKind::Lit(literal(&text, float, suffix.as_deref(), true, span)?)
                    }
                    _ => {
                        return Err(Error::parse(
                            "no unary minus, write a negative literal like `-1.0` or use `neg(x)`",
                            span,
                        ));
                    }
                }
            }
            Tok::Str(s) => ExprKind::Str(s),
            Tok::LBracket => {
                let mut elems = Vec::new();
                while self.peek() != &Tok::RBracket {
                    elems.push(self.expr()?);
                    if self.peek() != &Tok::RBracket {
                        self.expect(Tok::Comma)?;
                    }
                }
                self.expect(Tok::RBracket)?;
                ExprKind::ArrayLit(elems)
            }
            Tok::LParen => {
                let inner = self.expr()?;
                self.expect(Tok::RParen)?;
                return Ok(inner);
            }
            Tok::LBrace => {
                self.pos -= 1;
                ExprKind::Block(self.block()?)
            }
            Tok::Pipe => {
                let mut params = Vec::new();
                while self.peek() != &Tok::Pipe {
                    params.push(self.ident("a lambda parameter")?);
                    if self.peek() != &Tok::Pipe {
                        self.expect(Tok::Comma)?;
                    }
                }
                self.expect(Tok::Pipe)?;
                ExprKind::Lambda(params, Box::new(self.expr()?))
            }
            Tok::Ident(w) => match w.as_str() {
                "true" => ExprKind::Lit(Scalar::Bool(true)),
                "false" => ExprKind::Lit(Scalar::Bool(false)),
                "if" => {
                    let cond = self.expr()?;
                    let then = self.block()?;
                    self.expect_kw("else")?;
                    let els = self.block()?;
                    ExprKind::If(Box::new(cond), then, els)
                }
                _ if is_reserved(&w) => {
                    return Err(Error::parse(format!("unexpected keyword `{w}`"), span));
                }
                _ if self.peek() == &Tok::LParen => {
                    self.bump();
                    let mut args = Vec::new();
                    while self.peek() != &Tok::RParen {
                        args.push(self.expr()?);
                        if self.peek() != &Tok::RParen {
                            self.expect(Tok::Comma)?;
                        }
                    }
                    self.expect(Tok::RParen)?;
                    ExprKind::Call(w, args)
                }
                _ => ExprKind::Name(w),
            },
            other => {
                return Err(Error::parse(
                    format!("expected an expression, found {}", describe(&other)),
                    span,
                ));
            }
        };
        Ok(Expr { kind, span })
    }
}

fn is_reserved(s: &str) -> bool {
    matches!(s, "fn" | "let" | "input" | "export" | "if" | "else" | "true" | "false")
}
