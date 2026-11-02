//! Kernel selection. Each statement's term goes into an egglog e-graph with exact rewrites,
//! the integer sum rewrites, and one rewrite per fused kernel pattern, and the cheapest cover
//! by kernels is extracted.

mod egglog;
mod sexpr;

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use crate::check::Globals;
use crate::error::{Error, Span};
use crate::kernels::{KernelId, Registry};
use crate::lower::{Leaf, LeafId, TermGraph, TermId, TermKind, step};
use crate::value::{ScalarType, Value};

#[cfg(test)]
mod tests;

/// Equality saturation iterations per statement. When saturation has not finished by then,
/// extraction uses the e-graph as it stands, which is correct but possibly less fused.
pub const SATURATION_ITERATIONS: usize = 8;

/// The id of a node in a `PlanGraph`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanId(pub u32);

impl PlanId {
    pub(crate) fn index(self) -> usize {
        usize::try_from(self.0).expect("u32 fits in usize")
    }
}

/// A node of a plan: a kernel call over plan nodes, a leaf, a broadcast, or a conditional.
/// Children are ids into the `PlanGraph` that holds the node.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PlanNode {
    /// A kernel applied to its arguments in the kernel's input order. An argument for a
    /// `Scalar` input is the scalar itself, never a `Bcast`.
    Kernel(KernelId, Vec<PlanId>),
    Leaf(LeafId),
    /// A scalar operand of a single-op array kernel, repeated to the array length.
    Bcast(PlanId),
    /// Evaluates the scalar `bool` condition, then only the branch it picks.
    Cond(PlanId, PlanId, PlanId),
}

/// The plans of one program as a graph of hash-consed nodes. Equal nodes have one id, so
/// a subplan repeated within a statement or across statements is stored once. A child
/// always has a smaller id than its parents.
#[derive(Clone, Debug, Default)]
pub struct PlanGraph {
    nodes: Vec<PlanNode>,
    index: HashMap<PlanNode, PlanId>,
}

impl PlanGraph {
    pub fn node(&self, id: PlanId) -> &PlanNode {
        &self.nodes[id.index()]
    }

    /// The number of distinct nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The id of `node`, added to the graph if no equal node is in it. Panics when a child
    /// is not in the graph.
    pub fn intern(&mut self, node: PlanNode) -> PlanId {
        if let Some(&id) = self.index.get(&node) {
            return id;
        }
        let id = PlanId(u32::try_from(self.nodes.len()).expect("plan node count fits in u32"));
        let children: &[PlanId] = match &node {
            PlanNode::Kernel(_, args) => args,
            PlanNode::Leaf(_) => &[],
            PlanNode::Bcast(c) => std::slice::from_ref(c),
            PlanNode::Cond(c, t, e) => &[*c, *t, *e],
        };
        assert!(
            children.iter().all(|c| *c < id),
            "a plan node's children must be in the graph"
        );
        self.nodes.push(node.clone());
        self.index.insert(node, id);
        id
    }
}

/// The arithmetic that an integer sum rewrite moves out of a sum, see `SumRewrite`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SumRule {
    /// `sum(x + k)` to `sum(x) + n * k`.
    Add,
    /// `sum(x - k)` to `sum(x) - n * k`.
    Sub,
    /// `sum(k - x)` to `n * k - sum(x)`.
    SubFrom,
    /// `sum(x * k)` to `sum(x) * k`.
    Mul,
}

impl SumRule {
    pub const ALL: [SumRule; 4] = [SumRule::Add, SumRule::Sub, SumRule::SubFrom, SumRule::Mul];
}

/// An integer sum rewrite on one element type. Rewrites order by type name, then rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SumRewrite {
    pub ty: ScalarType,
    pub rule: SumRule,
}

impl Ord for SumRewrite {
    fn cmp(&self, other: &SumRewrite) -> std::cmp::Ordering {
        (self.ty.name(), self.rule).cmp(&(other.ty.name(), other.rule))
    }
}

impl PartialOrd for SumRewrite {
    fn partial_cmp(&self, other: &SumRewrite) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for SumRewrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = match self.rule {
            SumRule::Add => "sum(x + k)",
            SumRule::Sub => "sum(x - k)",
            SumRule::SubFrom => "sum(k - x)",
            SumRule::Mul => "sum(x * k)",
        };
        write!(f, "{shape} on {}", self.ty.name())
    }
}

/// Selects kernels for terms. Holds the egglog prelude generated from one registry, which
/// must be the registry passed to `plan`.
pub struct Planner {
    prelude: String,
}

impl Planner {
    /// Builds the prelude. With `fusion` off no fused pattern and no integer sum rewrite
    /// applies, so every plan uses single-op kernels only and keeps each sum over its array.
    /// Panics on a malformed fused pattern, a bug in the kernel tables.
    pub fn new(reg: &Registry, fusion: bool) -> Planner {
        let prelude =
            egglog::prelude(reg, fusion).unwrap_or_else(|e| panic!("kernel patterns: {e}"));
        Planner { prelude }
    }

    /// The cheapest cover of the term at `root` of `graph` by kernels, added to `cx.graph`.
    /// Every op node of the term is covered, either by its single-op kernel or as part of a
    /// fused kernel. The condition and branches of a conditional are planned on their own,
    /// so no kernel spans a conditional's boundary. A term that `cx` planned before returns
    /// its earlier plan. `graph` must be the term graph of every term planned with `cx`.
    /// Errors are internal planner failures and carry `span`, the span of the statement
    /// whose term is planned.
    pub fn plan(
        &self,
        reg: &Registry,
        graph: &TermGraph,
        root: TermId,
        span: Span,
        cx: &mut PlanContext,
    ) -> Result<PlanId, Error> {
        self.plan_with_sum_rewrites(reg, graph, root, span, cx)
            .map(|(id, _)| id)
    }

    /// Like `plan`, and also returns the integer sum rewrites that matched at least once
    /// while planning the term, the parts of its conditionals included. A matched rewrite
    /// added its right-hand side to the e-graph, whether or not extraction chose it. A term
    /// planned before returns the rewrites of its first planning. This is a test hook, the
    /// plan is the one `plan` returns.
    pub fn plan_with_sum_rewrites(
        &self,
        reg: &Registry,
        graph: &TermGraph,
        root: TermId,
        span: Span,
        cx: &mut PlanContext,
    ) -> Result<(PlanId, BTreeSet<SumRewrite>), Error> {
        let id = self.plan_into(reg, graph, span, cx, root)?;
        Ok((id, cx.planned[&root].1.clone()))
    }

    /// `plan` of `term`. On success `cx.planned` holds the plan of `term` with the integer
    /// sum rewrites that matched while planning it and its parts. A term planned before
    /// returns its earlier plan. A failure is not memoized.
    fn plan_into(
        &self,
        reg: &Registry,
        graph: &TermGraph,
        span: Span,
        cx: &mut PlanContext,
        term: TermId,
    ) -> Result<PlanId, Error> {
        if let Some((id, _)) = cx.planned.get(&term) {
            return Ok(*id);
        }
        step();
        let mut fired = BTreeSet::new();
        // Plans a part of `term` and adds the rewrites of the part to `fired`.
        let mut part = |ctx: &mut PlanContext, x: TermId| {
            let id = self.plan_into(reg, graph, span, ctx, x)?;
            fired.extend(ctx.planned[&x].1.iter().copied());
            Ok::<_, Error>(id)
        };
        let id = if let TermKind::Cond(c, t, e) = *graph.kind(term) {
            let node = PlanNode::Cond(part(cx, c)?, part(cx, t)?, part(cx, e)?);
            cx.graph.intern(node)
        } else {
            let fail = |msg: String| Error::runtime(format!("planner: {msg}"), Some(span));
            let (commands, root, holes) = egglog::emit(reg, graph, term).map_err(fail)?;
            let holes = holes
                .iter()
                .map(|&h| part(cx, h))
                .collect::<Result<Vec<_>, _>>()?;
            let (dag, extracted, matched) =
                egglog::extract(&self.prelude, &commands, &root, SATURATION_ITERATIONS)
                    .map_err(fail)?;
            fired.extend(
                matched
                    .iter()
                    .filter_map(|name| egglog::sum_rewrite_named(name)),
            );
            to_plan(reg, &dag, extracted, &holes, &mut cx.graph).map_err(fail)?
        };
        cx.planned.insert(term, (id, fired));
        Ok(id)
    }
}

/// What planning one program carries from statement to statement: the plan graph that
/// holds every plan, and the plan of each term planned so far.
#[derive(Debug, Default)]
pub struct PlanContext {
    pub graph: PlanGraph,
    /// Per term planned so far, its plan and the integer sum rewrites that matched while
    /// planning it and its parts.
    planned: HashMap<TermId, (PlanId, BTreeSet<SumRewrite>)>,
}

/// Maps the term `root` of an extracted egglog `dag` into `graph`, each distinct DAG node
/// once. `holes` are the plans of the conditionals that `egglog::emit` replaced by holes.
fn to_plan(
    reg: &Registry,
    dag: &::egglog::TermDag,
    root: ::egglog::TermId,
    holes: &[PlanId],
    graph: &mut PlanGraph,
) -> Result<PlanId, String> {
    let mut walk = DagWalk {
        reg,
        dag,
        holes,
        graph,
        done: HashMap::new(),
    };
    walk.node(root)
}

/// The state of `to_plan`.
struct DagWalk<'a> {
    reg: &'a Registry,
    dag: &'a ::egglog::TermDag,
    holes: &'a [PlanId],
    graph: &'a mut PlanGraph,
    /// The plan of each DAG node mapped so far.
    done: HashMap<::egglog::TermId, PlanId>,
}

impl DagWalk<'_> {
    fn node(&mut self, t: ::egglog::TermId) -> Result<PlanId, String> {
        use ::egglog::Term;
        use ::egglog::ast::Literal;
        if let Some(&id) = self.done.get(&t) {
            return Ok(id);
        }
        step();
        let dag = self.dag;
        let bad = || format!("unexpected extracted term `{}`", dag.to_string(t));
        let Term::App(head, args) = dag.get(t) else {
            return Err(bad());
        };
        // The integer literal that is the only argument of a leaf or a hole.
        let int = || match args.as_slice() {
            [a] => match dag.get(*a) {
                Term::Lit(Literal::Int(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        };
        let id = match head.as_str() {
            egglog::LEAF => {
                let n = int().and_then(|n| u32::try_from(n).ok()).ok_or_else(bad)?;
                self.graph.intern(PlanNode::Leaf(LeafId(n)))
            }
            egglog::HOLE => {
                let n = int()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(bad)?;
                *self.holes.get(n).ok_or_else(bad)?
            }
            egglog::BCAST => {
                let [inner] = args.as_slice() else {
                    return Err(bad());
                };
                let inner = self.node(*inner)?;
                self.graph.intern(PlanNode::Bcast(inner))
            }
            _ => {
                let name = head.strip_prefix(egglog::KERNEL_PREFIX).ok_or_else(bad)?;
                let id = self
                    .reg
                    .by_name(name)
                    .ok_or_else(|| format!("unknown kernel `{name}`"))?;
                if self.reg.get(id).inputs.len() != args.len() {
                    return Err(bad());
                }
                let args = args
                    .iter()
                    .map(|&a| self.node(a))
                    .collect::<Result<_, _>>()?;
                self.graph.intern(PlanNode::Kernel(id, args))
            }
        };
        self.done.insert(t, id);
        Ok(id)
    }
}

/// Renders the plan at `id` of `graph` as an s-expression of kernel names. Variable leaves
/// show their name, scalar constants their value, constant arrays `<type[len]>`, a broadcast
/// shows the scalar inside it, and a conditional is `(if C then T else E)`. A node reached
/// twice is rendered twice, so the text is that of the plan as a tree.
pub fn render(
    reg: &Registry,
    graph: &PlanGraph,
    id: PlanId,
    leaves: &[Leaf],
    globals: &Globals,
) -> String {
    match graph.node(id) {
        PlanNode::Kernel(k, args) => {
            let mut out = format!("({}", reg.get(*k).name);
            for a in args {
                out.push(' ');
                out.push_str(&render(reg, graph, *a, leaves, globals));
            }
            out.push(')');
            out
        }
        PlanNode::Leaf(id) => match &leaves[usize::try_from(id.0).expect("u32 fits in usize")] {
            Leaf::Var(v) => globals.name(*v).to_string(),
            Leaf::Const(Value::Scalar(s)) => s.to_string(),
            Leaf::Const(Value::Array(a)) => format!("<{}[{}]>", a.elem().name(), a.len()),
            Leaf::Const(Value::Str(s)) => format!("{s:?}"),
        },
        PlanNode::Bcast(inner) => render(reg, graph, *inner, leaves, globals),
        PlanNode::Cond(c, t, e) => format!(
            "(if {} then {} else {})",
            render(reg, graph, *c, leaves, globals),
            render(reg, graph, *t, leaves, globals),
            render(reg, graph, *e, leaves, globals),
        ),
    }
}
