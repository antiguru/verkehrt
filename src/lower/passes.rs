use std::collections::{HashMap, HashSet};

use crate::check::{Globals, VarId};
use crate::error::Span;

use super::leaves::Leaves;
use super::{LStmt, Leaf, Term, TermKind};

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
pub(super) fn inline_single_use(
    stmts: Vec<LStmt>,
    leaves: &Leaves,
    exported: &HashSet<VarId>,
) -> Vec<LStmt> {
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
pub(super) fn hoist_repeated(mut s: LStmt, leaves: &mut Leaves, g: &mut Globals) -> Vec<LStmt> {
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
