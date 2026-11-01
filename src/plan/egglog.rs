//! The egglog program text: the prelude generated from the registry, the per-statement
//! program, and running it to an extracted term.

use std::fmt::Write;

use crate::kernels::{InputSpec, KernelDef, Registry};
use crate::lower::{Term, TermKind};
use crate::ops::{Op, ReduceOp};
use crate::value::{ScalarType, ValueType};

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
/// Relation holding the array e-classes that are not themselves a hole. A hole nested deeper
/// inside such an e-class still counts. A plan may evaluate one of them twice, which a hole's
/// cost of 0 would hide.
const PLAIN: &str = "Plain";

fn kernel_ctor(name: &str) -> String {
    format!("{KERNEL_PREFIX}{name}")
}

/// The datatype with one constructor per kernel, the `Arr` and `Plain` relations, then the
/// exact rewrites and, when `fusion` is set, the integer sum rewrites of `sum_rules` and one
/// rewrite per fused kernel pattern.
pub fn prelude(reg: &Registry, fusion: bool) -> Result<String, String> {
    let mut out = format!(
        "(datatype Expr\n  ({LEAF} i64 :cost 0)\n  ({HOLE} i64 :cost 0)\n  ({BCAST} Expr :cost 0)\n"
    );
    for (_, def) in reg.iter() {
        let args = " Expr".repeat(def.inputs.len());
        writeln!(
            out,
            "  ({}{args} :cost {})",
            kernel_ctor(&def.name),
            def.cost
        )
        .expect("writing to a String cannot fail");
    }
    out.push_str(")\n");
    writeln!(out, "(relation {ARR} (Expr))").expect("writing to a String cannot fail");
    writeln!(out, "(relation {PLAIN} (Expr))").expect("writing to a String cannot fail");
    for (_, def) in reg.iter() {
        if def.commutative && def.op.is_some() && def.inputs.len() == 2 {
            let k = kernel_ctor(&def.name);
            writeln!(out, "(rewrite ({k} ?a ?b) ({k} ?b ?a))")
                .expect("writing to a String cannot fail");
        }
    }
    for (from, to) in [
        (Op::Gt, Op::Lt),
        (Op::Lt, Op::Gt),
        (Op::Ge, Op::Le),
        (Op::Le, Op::Ge),
    ] {
        for (_, def) in reg.iter().filter(|(_, d)| d.op == Some(from)) {
            // Rules exist only for pairs in the registry. The partner takes the same operand
            // types in swapped order, which are equal here, so equal inputs identify it.
            let partner = reg
                .iter()
                .map(|(_, d)| d)
                .find(|d| d.op == Some(to) && d.inputs == def.inputs && d.output == def.output);
            if let Some(p) = partner {
                let (k, pk) = (kernel_ctor(&def.name), kernel_ctor(&p.name));
                writeln!(out, "(rewrite ({k} ?a ?b) ({pk} ?b ?a))")
                    .expect("writing to a String cannot fail");
            }
        }
    }
    if fusion {
        for t in ScalarType::ARRAY_TYPES {
            out.push_str(&sum_rules(reg, t));
        }
        for (_, def) in reg.iter() {
            if let Some(rule) = fused_rule(def)? {
                out.push_str(&rule);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// The rewrites that move a broadcast scalar `k` out of a sum of element type `t` over an
/// array `x`: `sum(x + k)` to `sum(x) + n * k`, `sum(x - k)` to `sum(x) - n * k`,
/// `sum(k - x)` to `n * k - sum(x)`, and `sum(x * k)` to `sum(x) * k`, where `n` is
/// `length(x)` cast to `t`. Every operation of a right-hand side is a scalar kernel of
/// type `t`. Empty for a float type, for a sum whose result is not of type `t`, and for a
/// type that lacks one of the kernels. `x` is never a conditional, because the right-hand
/// side evaluates `x` twice.
fn sum_rules(reg: &Registry, t: ScalarType) -> String {
    // Integer arithmetic wraps, so it is a ring modulo 2^w, in which these identities hold
    // bit for bit. Float arithmetic does not associate.
    if !t.is_int() {
        return String::new();
    }
    let (arr, sc, i64_sc) = (
        ValueType::Array(t),
        ValueType::Scalar(t),
        ValueType::Scalar(ScalarType::I64),
    );
    // The constructor of the kernel that `op` resolves to on `args`, if it returns `out`.
    let kernel = |op: Op, args: &[ValueType], out: ValueType| {
        reg.lookup(op, args)
            .filter(|id| reg.get(*id).output == out)
            .map(|id| kernel_ctor(&reg.get(id).name))
    };
    // A sum that widens, as for `i32` to `i64`, adds elements that wrapped in the narrow
    // type, which no product in either type reproduces.
    let sum = kernel(Op::Reduce(ReduceOp::Add), &[arr], sc);
    let len = kernel(Op::Length, &[arr], i64_sc);
    // `length` is `i64`. A cast to a narrower type wraps to `n mod 2^w`, and the product
    // with `k` is the same modulo 2^w, so the count stays exact.
    let count = match t {
        ScalarType::I64 => len.map(|l| format!("({l} ?x)")),
        _ => len
            .zip(kernel(Op::Cast(t), &[i64_sc], sc))
            .map(|(l, c)| format!("({c} ({l} ?x))")),
    };
    let (
        Some(sum),
        Some(count),
        Some(add),
        Some(sub),
        Some(mul),
        Some(add_s),
        Some(sub_s),
        Some(mul_s),
    ) = (
        sum,
        count,
        kernel(Op::Add, &[arr, arr], arr),
        kernel(Op::Sub, &[arr, arr], arr),
        kernel(Op::Mul, &[arr, arr], arr),
        kernel(Op::Add, &[sc, sc], sc),
        kernel(Op::Sub, &[sc, sc], sc),
        kernel(Op::Mul, &[sc, sc], sc),
    )
    else {
        return String::new();
    };
    // The guard keeps `x` an array, so `k` is the only broadcast. A hole costs 0, so
    // extraction would not see that a conditional `x` runs twice, and the guard excludes an
    // `x` that is itself a hole. Each right-hand side names
    // `x` and `k` in the order of its left-hand side, so the plan evaluates them in that
    // order unless commutativity swaps them.
    let when = format!(":when (({PLAIN} ?x))");
    [
        format!("(rewrite ({sum} ({add} ?x ({BCAST} ?k))) ({add_s} ({sum} ?x) ({mul_s} {count} ?k)) {when})"),
        format!("(rewrite ({sum} ({sub} ?x ({BCAST} ?k))) ({sub_s} ({sum} ?x) ({mul_s} {count} ?k)) {when})"),
        format!("(rewrite ({sum} ({sub} ({BCAST} ?k) ?x)) ({sub_s} ({mul_s} ?k {count}) ({sum} ?x)) {when})"),
        format!("(rewrite ({sum} ({mul} ?x ({BCAST} ?k))) ({mul_s} ({sum} ?x) ?k) {when})"),
    ]
    .iter()
    .map(|r| format!("{r}\n"))
    .collect()
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
        .map_heads(&|h| {
            if h == BCAST {
                h.to_string()
            } else {
                kernel_ctor(h)
            }
        });
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
/// fact for every node that is an array and not a broadcast, and a `Plain` fact for every
/// such node that is not a conditional. Each op is written as the
/// kernel that `Registry::lookup` resolves for its argument types, and each conditional as a
/// hole. Returns the commands, the name bound to the root, and the conditionals in hole order.
pub fn emit(reg: &Registry, term: &Term) -> Result<(String, String, Vec<Term>), String> {
    // Facts per original node suffice because every rewrite either adds a single constructor
    // over existing e-classes, or, as the integer sum rewrites do, creates e-classes that
    // hold scalars only. So every array e-class has an original node, and no rule that
    // guards an operand with `Arr` or `Plain` matches a created e-class as an array. No
    // rewrite puts a hole into another e-class, so a `Plain` e-class never holds a hole.
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
        if !matches!(term.kind, TermKind::Cond(..)) {
            writeln!(out, "({PLAIN} {name})").expect("writing to a String cannot fail");
        }
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
