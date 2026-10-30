//! Kernel selection. Each statement's term goes into an egglog e-graph with exact rewrites
//! and one rewrite per fused kernel pattern, and the cheapest cover by kernels is extracted.

mod egglog;
mod sexpr;

use crate::check::Globals;
use crate::error::Error;
use crate::kernels::{KernelId, Registry};
use crate::lower::{Leaf, LeafId, Term};
use crate::value::Value;

use sexpr::Sexp;

#[cfg(test)]
mod tests;

/// Equality saturation iterations per statement. When saturation has not finished by then,
/// extraction uses the e-graph as it stands, which is correct but possibly less fused.
pub const SATURATION_ITERATIONS: usize = 8;

/// A plan: a tree of kernel calls over leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanNode {
    /// A kernel applied to its arguments in the kernel's input order. An argument for a
    /// `Scalar` input is the scalar itself, never a `Bcast`.
    Kernel(KernelId, Vec<PlanNode>),
    Leaf(LeafId),
    /// A scalar operand of a single-op array kernel, repeated to the array length.
    Bcast(Box<PlanNode>),
}

/// Selects kernels for terms. Holds the egglog prelude generated from one registry, which
/// must be the registry passed to `plan`.
pub struct Planner {
    prelude: String,
}

impl Planner {
    /// Builds the prelude. Panics on a malformed fused pattern, a bug in the kernel tables.
    pub fn new(reg: &Registry) -> Planner {
        let prelude = egglog::prelude(reg).unwrap_or_else(|e| panic!("kernel patterns: {e}"));
        Planner { prelude }
    }

    /// The cheapest cover of `term` by kernels. Every op node of the term is covered, either
    /// by its single-op kernel or as part of a fused kernel. Errors are internal planner
    /// failures and carry the term's span.
    pub fn plan(&self, reg: &Registry, term: &Term) -> Result<PlanNode, Error> {
        let fail = |msg: String| Error::runtime(format!("planner: {msg}"), Some(term.span));
        let (commands, root) = egglog::emit(reg, term).map_err(fail)?;
        let extracted = egglog::extract(&self.prelude, &commands, &root, SATURATION_ITERATIONS)
            .map_err(fail)?;
        let sexp = sexpr::parse(&extracted).map_err(fail)?;
        to_node(reg, &sexp).map_err(fail)
    }
}

/// Maps an extracted egglog term back to a plan.
fn to_node(reg: &Registry, sexp: &Sexp) -> Result<PlanNode, String> {
    let bad = || format!("unexpected extracted term `{sexp}`");
    let Sexp::List(items) = sexp else {
        return Err(bad());
    };
    let Some((Sexp::Atom(head), args)) = items.split_first() else {
        return Err(bad());
    };
    match (head.as_str(), args) {
        (egglog::LEAF, [Sexp::Atom(n)]) => {
            let n = n.parse::<u32>().map_err(|_| bad())?;
            Ok(PlanNode::Leaf(LeafId(n)))
        }
        (egglog::BCAST, [inner]) => Ok(PlanNode::Bcast(Box::new(to_node(reg, inner)?))),
        _ => {
            let name = head.strip_prefix(egglog::KERNEL_PREFIX).ok_or_else(bad)?;
            let id = reg.by_name(name).ok_or_else(|| format!("unknown kernel `{name}`"))?;
            if reg.get(id).inputs.len() != args.len() {
                return Err(bad());
            }
            let args = args.iter().map(|a| to_node(reg, a)).collect::<Result<_, _>>()?;
            Ok(PlanNode::Kernel(id, args))
        }
    }
}

/// Renders a plan as an s-expression of kernel names. Variable leaves show their name,
/// scalar constants their value, constant arrays `<type[len]>`, and a broadcast shows the
/// scalar inside it.
pub fn render(reg: &Registry, node: &PlanNode, leaves: &[Leaf], globals: &Globals) -> String {
    match node {
        PlanNode::Kernel(id, args) => {
            let mut out = format!("({}", reg.get(*id).name);
            for a in args {
                out.push(' ');
                out.push_str(&render(reg, a, leaves, globals));
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
        PlanNode::Bcast(inner) => render(reg, inner, leaves, globals),
    }
}
