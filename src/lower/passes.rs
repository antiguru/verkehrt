use std::collections::{HashMap, HashSet};

use crate::check::{Globals, VarId};
use crate::error::Span;

use super::leaves::Leaves;
use super::{LStmt, Leaf, TermGraph, TermId, TermKind, step};

/// The nodes reachable from `root` along the edges that `children` lists, parents before
/// children. Interning gives every child a smaller id than its parents, so descending ids
/// are a topological order.
fn reachable(
    graph: &TermGraph,
    root: TermId,
    children: impl Fn(&TermKind, &mut Vec<TermId>),
) -> Vec<TermId> {
    let mut seen = HashSet::from([root]);
    let mut stack = vec![root];
    let mut out = Vec::new();
    let mut kids = Vec::new();
    while let Some(t) = stack.pop() {
        step();
        out.push(t);
        kids.clear();
        children(graph.kind(t), &mut kids);
        for &k in &kids {
            if seen.insert(k) {
                stack.push(k);
            }
        }
    }
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// Every child edge, branches of conditionals included, with multiplicity.
fn all_children(kind: &TermKind, out: &mut Vec<TermId>) {
    match kind {
        TermKind::Leaf(_) => {}
        TermKind::Bcast(a) => out.push(*a),
        TermKind::Op(_, args) => out.extend(args),
        TermKind::Cond(c, t, e) => out.extend([*c, *t, *e]),
    }
}

/// The number of paths from `root` to each node reachable from it, saturated at 2. This is
/// the number of occurrences of the node in the term as a tree.
fn path_counts(graph: &TermGraph, root: TermId) -> HashMap<TermId, usize> {
    let mut counts = HashMap::from([(root, 1)]);
    let mut kids = Vec::new();
    for t in reachable(graph, root, all_children) {
        let c = counts[&t];
        kids.clear();
        all_children(graph.kind(t), &mut kids);
        for &k in &kids {
            let e = counts.entry(k).or_default();
            *e = (*e + c).min(2);
        }
    }
    counts
}

fn stmt_term_mut(s: &mut LStmt) -> &mut TermId {
    match s {
        LStmt::Bind { term, .. } | LStmt::Print { term, .. } | LStmt::Show { term, .. } => term,
    }
}

/// Replaces variable leaves in the terms of one statement.
struct Substitution<'a> {
    graph: &'a mut TermGraph,
    pending: &'a HashMap<VarId, TermId>,
    leaves: &'a Leaves,
    /// The nodes already rebuilt for the statement, with their new ids.
    memo: HashMap<TermId, TermId>,
}

impl Substitution<'_> {
    /// Rebuilds `t` with every variable leaf in `pending` replaced by its term.
    fn go(&mut self, t: TermId) -> TermId {
        if let Some(&n) = self.memo.get(&t) {
            return n;
        }
        step();
        let kind = match self.graph.kind(t).clone() {
            TermKind::Leaf(id) => {
                let n = match self.leaves.get(id) {
                    Leaf::Var(v) if self.pending.contains_key(v) => self.pending[v],
                    _ => t,
                };
                self.memo.insert(t, n);
                return n;
            }
            TermKind::Bcast(a) => TermKind::Bcast(self.go(a)),
            TermKind::Op(op, args) => {
                TermKind::Op(op, args.into_iter().map(|a| self.go(a)).collect())
            }
            // A let used only in a branch moves into it and runs only when the branch does.
            TermKind::Cond(c, a, b) => TermKind::Cond(self.go(c), self.go(a), self.go(b)),
        };
        let n = self.graph.intern(kind, self.graph.ty(t));
        self.memo.insert(t, n);
        n
    }
}

/// Substitutes each compound let with exactly one use into its user and drops the let.
/// A let bound to a leaf is already as cheap as it gets and stays the materialized binding,
/// and an export counts as an extra use so it never goes away.
pub(super) fn inline_single_use(
    stmts: Vec<LStmt>,
    leaves: &Leaves,
    graph: &mut TermGraph,
    exported: &HashSet<VarId>,
) -> Vec<LStmt> {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for s in &stmts {
        for (t, c) in path_counts(graph, s.term()) {
            if let TermKind::Leaf(id) = graph.kind(t)
                && let Leaf::Var(v) = leaves.get(*id)
            {
                let e = uses.entry(*v).or_default();
                *e = (*e + c).min(2);
            }
        }
    }
    let mut pending: HashMap<VarId, TermId> = HashMap::new();
    let mut out = Vec::new();
    for mut s in stmts {
        let t = stmt_term_mut(&mut s);
        if !pending.is_empty() {
            *t = Substitution {
                graph,
                pending: &pending,
                leaves,
                memo: HashMap::new(),
            }
            .go(*t);
        }
        if let LStmt::Bind { var, term, .. } = &s
            && !exported.contains(var)
            && uses.get(var) == Some(&1)
            && !matches!(graph.kind(*term), TermKind::Leaf(_))
        {
            pending.insert(*var, *term);
            continue;
        }
        out.push(s);
    }
    out
}

/// The edges that hoisting counts: operands of an op, the operand of a broadcast, and the
/// condition of a conditional, whose branches are not counted.
fn unconditional_children(kind: &TermKind, out: &mut Vec<TermId>) {
    match kind {
        TermKind::Leaf(_) => {}
        TermKind::Bcast(a) => out.push(*a),
        TermKind::Op(_, args) => out.extend(args),
        TermKind::Cond(c, ..) => out.push(*c),
    }
}

/// Counts every distinct compound subterm that is evaluated unconditionally. A repeated
/// subterm is descended into once, so its children are not counted as repeated just because
/// their parent is. Branches of a conditional are not counted, so nothing inside them is
/// hoisted into code that always runs.
///
/// On the graph, the count of a node is the number of edges into it from the distinct op and
/// conditional nodes reachable from `root`. A broadcast is transparent and passes its own
/// count on to its operand, as the tree walk descends into it at every occurrence.
fn count_ops(graph: &TermGraph, root: TermId) -> HashMap<TermId, usize> {
    let mut counts = HashMap::from([(root, 1)]);
    for t in reachable(graph, root, unconditional_children) {
        match graph.kind(t) {
            TermKind::Leaf(_) => {}
            TermKind::Bcast(a) => {
                let c = counts[&t];
                *counts.entry(*a).or_default() += c;
            }
            TermKind::Op(_, args) => {
                for a in args {
                    *counts.entry(*a).or_default() += 1;
                }
            }
            TermKind::Cond(c, ..) => *counts.entry(*c).or_default() += 1,
        }
    }
    counts
}

struct Hoister<'a> {
    /// The span of the statement, which every hoisted bind takes, so that a runtime error in
    /// one carries the line of the statement it serves.
    span: Span,
    counts: HashMap<TermId, usize>,
    done: HashMap<TermId, TermId>,
    binds: Vec<LStmt>,
    leaves: &'a mut Leaves,
    graph: &'a mut TermGraph,
    g: &'a mut Globals,
}

impl Hoister<'_> {
    fn go(&mut self, t: TermId) -> TermId {
        step();
        let ty = self.graph.ty(t);
        match self.graph.kind(t).clone() {
            TermKind::Leaf(_) => t,
            TermKind::Bcast(a) => {
                let a = self.go(a);
                self.graph.intern(TermKind::Bcast(a), ty)
            }
            kind @ (TermKind::Op(..) | TermKind::Cond(..)) => {
                let shared = self.counts[&t] >= 2;
                if shared && let Some(&leaf) = self.done.get(&t) {
                    return leaf;
                }
                let kind = match kind {
                    TermKind::Op(op, args) => {
                        TermKind::Op(op, args.into_iter().map(|a| self.go(a)).collect())
                    }
                    // Branches stay as they are, see `count_ops`.
                    TermKind::Cond(c, a, b) => TermKind::Cond(self.go(c), a, b),
                    TermKind::Leaf(_) | TermKind::Bcast(_) => unreachable!("matched above"),
                };
                let rebuilt = self.graph.intern(kind, ty);
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
                let leaf = self.graph.intern(TermKind::Leaf(id), ty);
                self.done.insert(t, leaf);
                leaf
            }
        }
    }
}

/// Hoists the compound subterms that occur at least twice into binds placed before the
/// statement, so the statement's own term has no shared structure outside the branches of
/// conditionals.
pub(super) fn hoist_repeated(
    mut s: LStmt,
    leaves: &mut Leaves,
    graph: &mut TermGraph,
    g: &mut Globals,
) -> Vec<LStmt> {
    let counts = count_ops(graph, s.term());
    let mut h = Hoister {
        span: s.span(),
        counts,
        done: HashMap::new(),
        binds: Vec::new(),
        leaves,
        graph,
        g,
    };
    let t = stmt_term_mut(&mut s);
    *t = h.go(*t);
    let mut out = h.binds;
    out.push(s);
    out
}
