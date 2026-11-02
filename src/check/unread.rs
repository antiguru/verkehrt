//! The unread-binding rule: a `let`, a function parameter, or a lambda parameter that nothing
//! reads is a type error, so a broken statement cannot sit unread in a program.

use std::collections::HashMap;

use crate::error::{Error, Span};
use crate::syntax::ast::{self, ExprKind};

/// One binding site and whether anything has read it.
struct Binding<'a> {
    name: &'a str,
    span: Span,
    read: bool,
    /// Whether the rule applies. Inputs, exports, `_` names, and REPL top-level lets opt out.
    checked: bool,
}

#[derive(Default)]
struct Walker<'a> {
    bindings: Vec<Binding<'a>>,
    /// Innermost last, mapping a name to its binding.
    scopes: Vec<HashMap<&'a str, usize>>,
}

impl<'a> Walker<'a> {
    fn bind(&mut self, name: &'a str, span: Span, checked: bool) {
        let id = self.bindings.len();
        self.bindings.push(Binding {
            name,
            span,
            read: false,
            checked: checked && !name.starts_with('_'),
        });
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .insert(name, id);
    }

    /// Marks the innermost binding of `name` as read. A name with no binding here is
    /// defined by an earlier REPL input or is a type error reported elsewhere.
    fn read(&mut self, name: &str) {
        if let Some(id) = self.scopes.iter().rev().find_map(|s| s.get(name)) {
            self.bindings[*id].read = true;
        }
    }

    fn block(&mut self, b: &'a ast::Block) {
        self.scopes.push(HashMap::new());
        for l in &b.lets {
            // The value comes before the binding, so `let x = x` reads the old `x`.
            self.expr(&l.value);
            self.bind(&l.name, l.span, true);
        }
        self.expr(&b.result);
        self.scopes.pop();
    }

    fn expr(&mut self, e: &'a ast::Expr) {
        match &e.kind {
            ExprKind::Lit(_) | ExprKind::Str(_) => {}
            ExprKind::Name(n) => self.read(n),
            ExprKind::ArrayLit(es) | ExprKind::Call(_, es) => es.iter().for_each(|x| self.expr(x)),
            ExprKind::Binary(_, a, b) => {
                self.expr(a);
                self.expr(b);
            }
            ExprKind::If(c, t, f) => {
                self.expr(c);
                self.block(t);
                self.block(f);
            }
            ExprKind::Lambda(params, body) => {
                self.scopes.push(HashMap::new());
                for p in params {
                    self.bind(p, e.span, true);
                }
                self.expr(body);
                self.scopes.pop();
            }
            ExprKind::Block(b) => self.block(b),
        }
    }
}

/// Rejects the first binding of `prog` that nothing reads. A read anywhere counts, also inside
/// a lambda or in one branch of an `if`. A name that starts with `_` opts out. With `repl` set,
/// top-level lets are exempt because a later input may read them.
pub fn check_unread(prog: &ast::Program, repl: bool) -> Result<(), Error> {
    // Function bodies see only their own parameters and lets, so each gets its own walker
    // and only the top-level items share one.
    let mut top = Walker::default();
    top.scopes.push(HashMap::new());
    let mut fns = Vec::new();
    for item in &prog.items {
        match item {
            ast::Item::Fn(def) => {
                let mut f = Walker::default();
                f.scopes.push(HashMap::new());
                for (p, _) in &def.params {
                    f.bind(p, def.span, true);
                }
                f.block(&def.body);
                fns.push(f);
            }
            ast::Item::Let(l) => {
                top.expr(&l.value);
                top.bind(&l.name, l.span, !repl);
            }
            // An export is an output, so it counts as a read of its own binding.
            ast::Item::Export(l) => {
                top.expr(&l.value);
                top.bind(&l.name, l.span, false);
            }
            ast::Item::Input(name, _, span) => top.bind(name, *span, false),
            ast::Item::Print(e, _) | ast::Item::Expr(e, _) => top.expr(e),
        }
    }
    fns.push(top);
    let unread = fns
        .iter()
        .flat_map(|w| &w.bindings)
        .filter(|b| b.checked && !b.read)
        .min_by_key(|b| (b.span.line, b.span.col));
    match unread {
        Some(b) => Err(Error::ty(
            format!("'{}' is never read, prefix it with '_' if intended", b.name),
            b.span,
        )),
        None => Ok(()),
    }
}
