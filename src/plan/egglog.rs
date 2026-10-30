//! The egglog program text: the prelude generated from the registry, the per-statement
//! program, and running it to an extracted term.

use std::fmt::Write;

use crate::kernels::{InputSpec, KernelDef, Registry};
use crate::lower::{Term, TermKind};
use crate::ops::Op;
use crate::value::ValueType;

use super::sexpr::{self, Sexp};

/// Prefix of every kernel constructor, so kernel names never collide with egglog keywords.
pub const KERNEL_PREFIX: &str = "K_";
/// Constructor of a leaf, applied to the `LeafId` as an i64.
pub const LEAF: &str = "Leaf";
/// Constructor of a broadcast scalar operand.
pub const BCAST: &str = "Bcast";
/// Constructor of a conditional planned on its own, applied to its index in the holes that
/// `emit` returns. No rewrite looks into it.
pub const HOLE: &str = "Hole";
/// Relation holding the e-classes whose value is a materialized array, as opposed to a
/// broadcast scalar or a scalar.
const ARR: &str = "Arr";

fn kernel_ctor(name: &str) -> String {
    format!("{KERNEL_PREFIX}{name}")
}

/// The datatype with one constructor per kernel, the `Arr` relation, then the exact
/// rewrites and one rewrite per fused kernel pattern.
pub fn prelude(reg: &Registry) -> Result<String, String> {
    let mut out = format!(
        "(datatype Expr\n  ({LEAF} i64 :cost 0)\n  ({HOLE} i64 :cost 0)\n  ({BCAST} Expr :cost 0)\n"
    );
    for (_, def) in reg.iter() {
        let args = " Expr".repeat(def.inputs.len());
        writeln!(out, "  ({}{args} :cost {})", kernel_ctor(&def.name), def.cost)
            .expect("writing to a String cannot fail");
    }
    out.push_str(")\n");
    writeln!(out, "(relation {ARR} (Expr))").expect("writing to a String cannot fail");
    for (_, def) in reg.iter() {
        if def.commutative && def.op.is_some() && def.inputs.len() == 2 {
            let k = kernel_ctor(&def.name);
            writeln!(out, "(rewrite ({k} ?a ?b) ({k} ?b ?a))")
                .expect("writing to a String cannot fail");
        }
    }
    for (from, to) in [(Op::Gt, Op::Lt), (Op::Lt, Op::Gt), (Op::Ge, Op::Le), (Op::Le, Op::Ge)] {
        for (_, def) in reg.iter().filter(|(_, d)| d.op == Some(from)) {
            // Rules exist only for pairs in the registry. The partner takes the same operand
            // types in swapped order, which are equal here, so equal inputs identify it.
            let partner = reg.iter().map(|(_, d)| d).find(|d| {
                d.op == Some(to) && d.inputs == def.inputs && d.output == def.output
            });
            if let Some(p) = partner {
                let (k, pk) = (kernel_ctor(&def.name), kernel_ctor(&p.name));
                writeln!(out, "(rewrite ({k} ?a ?b) ({pk} ?b ?a))")
                    .expect("writing to a String cannot fail");
            }
        }
    }
    for (_, def) in reg.iter() {
        if let Some(rule) = fused_rule(def)? {
            out.push_str(&rule);
            out.push('\n');
        }
    }
    Ok(out)
}

/// The rewrite from a fused kernel's pattern to the kernel, or `None` for a single-op kernel.
fn fused_rule(def: &KernelDef) -> Result<Option<String>, String> {
    let Some(pattern) = &def.pattern else {
        return Ok(None);
    };
    if pattern.vars.len() != def.inputs.len() {
        return Err(format!("pattern of {} does not bind every input", def.name));
    }
    let lhs = sexpr::parse(&pattern.lhs)
        .map_err(|e| format!("pattern of {}: {e}", def.name))?
        .map_heads(&|h| if h == BCAST { h.to_string() } else { kernel_ctor(h) });
    let mut rhs = vec![Sexp::Atom(kernel_ctor(&def.name))];
    rhs.extend(pattern.vars.iter().map(|v| Sexp::Atom(v.clone())));
    // Single-op elementwise kernels take an array or a broadcast in the same position, so
    // without a guard `sum(x * 2.0)` would match `dot` with a broadcast where dot needs
    // an array.
    let guards: Vec<String> = pattern
        .vars
        .iter()
        .zip(&def.inputs)
        .filter(|(_, spec)| matches!(spec, InputSpec::Array(_)))
        .map(|(v, _)| format!("({ARR} {v})"))
        .collect();
    let when = if guards.is_empty() {
        String::new()
    } else {
        format!(" :when ({})", guards.join(" "))
    };
    Ok(Some(format!("(rewrite {lhs} {}{when})", Sexp::List(rhs))))
}

/// The per-statement commands for `term`: one `let` per node in post-order, with an `Arr`
/// fact for every node that is an array and not a broadcast. Each op is written as the
/// kernel that `Registry::lookup` resolves for its argument types, and each conditional as a
/// hole. Returns the commands, the name bound to the root, and the conditionals in hole order.
pub fn emit(reg: &Registry, term: &Term) -> Result<(String, String, Vec<Term>), String> {
    // Facts per original node suffice because every rewrite adds a single constructor over
    // existing e-classes, so saturation never creates an e-class without an original node.
    let mut out = String::new();
    let mut next = 0;
    let mut holes = Vec::new();
    let root = emit_node(reg, term, &mut out, &mut next, &mut holes)?;
    Ok((out, root, holes))
}

fn emit_node(
    reg: &Registry,
    term: &Term,
    out: &mut String,
    next: &mut usize,
    holes: &mut Vec<Term>,
) -> Result<String, String> {
    let expr = match &term.kind {
        TermKind::Leaf(id) => format!("({LEAF} {})", id.0),
        TermKind::Bcast(inner) => {
            format!("({BCAST} {})", emit_node(reg, inner, out, next, holes)?)
        }
        TermKind::Cond(..) => {
            holes.push(term.clone());
            format!("({HOLE} {})", holes.len() - 1)
        }
        TermKind::Op(op, args) => {
            let tys: Vec<_> = args.iter().map(|a| a.ty).collect();
            let id = reg
                .lookup(*op, &tys)
                .ok_or_else(|| format!("no kernel for {op:?} on {tys:?}"))?;
            let mut expr = format!("({}", kernel_ctor(&reg.get(id).name));
            for a in args {
                expr.push(' ');
                expr.push_str(&emit_node(reg, a, out, next, holes)?);
            }
            expr.push(')');
            expr
        }
    };
    let name = format!("n{next}");
    *next += 1;
    writeln!(out, "(let {name} {expr})").expect("writing to a String cannot fail");
    if matches!(term.ty, ValueType::Array(_)) && !matches!(term.kind, TermKind::Bcast(_)) {
        writeln!(out, "({ARR} {name})").expect("writing to a String cannot fail");
    }
    Ok(name)
}

/// Runs `prelude` and the statement `commands`, saturates for `iterations`, and returns the
/// cheapest term extracted for `root` as text.
pub fn extract(
    prelude: &str,
    commands: &str,
    root: &str,
    iterations: usize,
) -> Result<String, String> {
    let program = format!("{prelude}{commands}(run {iterations})\n(extract {root})\n");
    let mut eg = ::egglog::EGraph::default();
    let outputs = eg
        .parse_and_run_program(None, &program)
        .map_err(|e| format!("egglog: {e}"))?;
    outputs
        .iter()
        .rev()
        .find_map(|o| match o {
            ::egglog::CommandOutput::ExtractBest(dag, _, t) => Some(dag.to_string(*t)),
            _ => None,
        })
        .ok_or_else(|| "egglog produced no extraction".to_string())
}
