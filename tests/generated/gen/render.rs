//! Source text for generated expressions and functions.

use std::collections::BTreeMap;

use verkehrt::ops::{Op, ReduceOp};
use verkehrt::value::ValueType;

use super::features::reads;
use super::{E, FnDef, Gen, Higher, K, Lam};

pub(super) fn reduce_name(r: ReduceOp) -> &'static str {
    match r {
        ReduceOp::Add => "add",
        ReduceOp::Mul => "mul",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::And => "and",
        ReduceOp::Or => "or",
        ReduceOp::Xor => "xor",
    }
}

pub(super) fn infix(op: Op) -> Option<&'static str> {
    Some(match op {
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
        Op::Div => "/",
        Op::Eq => "==",
        Op::Ne => "!=",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
        _ => return None,
    })
}

pub(super) fn call_name(op: Op) -> String {
    match op {
        Op::Neg => "neg".into(),
        Op::Abs => "abs".into(),
        Op::Min => "min".into(),
        Op::Max => "max".into(),
        Op::Sqrt => "sqrt".into(),
        Op::And => "and".into(),
        Op::Or => "or".into(),
        Op::Xor => "xor".into(),
        Op::Not => "not".into(),
        Op::Select => "select".into(),
        Op::Cast(t) => format!("cast_{}", t.name()),
        Op::Reduce(_) => "reduce".into(),
        Op::Scan(_) => "scan".into(),
        Op::ScanBy(_) => "scan_by".into(),
        Op::ReduceInto(_) => "reduce_into".into(),
        Op::Count => "count".into(),
        Op::Any => "any".into(),
        Op::All => "all".into(),
        Op::ArgMin => "argmin".into(),
        Op::ArgMax => "argmax".into(),
        Op::Compress => "compress".into(),
        Op::Where => "where".into(),
        Op::Prev => "prev".into(),
        Op::Iota => "iota".into(),
        Op::Fill => "fill".into(),
        Op::Gather => "gather".into(),
        Op::Slice => "slice".into(),
        Op::Histogram => "histogram".into(),
        Op::DivConst => "div_const".into(),
        Op::ArgSort => "argsort".into(),
        Op::GroupId => "group_id".into(),
        Op::RandI64 => "rand_i64".into(),
        Op::RandF64 => "rand_f64".into(),
        Op::Length => "length".into(),
        other => unreachable!("{other:?} is not generated"),
    }
}

/// A binder as written in the source. Names are unique per program, so a name that no read
/// mentions gets the `_` prefix that the unread-binding rule requires.
fn binder(name: &str, uses: &BTreeMap<String, usize>) -> String {
    if uses.contains_key(name) {
        name.to_string()
    } else {
        format!("_{name}")
    }
}

pub(super) fn type_name(t: ValueType) -> String {
    t.to_string()
}

impl Gen<'_, '_> {
    pub(super) fn render_fn(&self, f: &FnDef) -> String {
        let uses = reads(&f.body);
        let params: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", binder(n, &uses), type_name(*t)))
            .collect();
        format!(
            "fn {}({}) -> {} {{ {} }}\n",
            f.name,
            params.join(", "),
            type_name(f.ret),
            self.render_inner(&f.body)
        )
    }

    /// The contents of a block: its lets and result, or the expression itself.
    pub(super) fn render_inner(&self, e: &E) -> String {
        match &e.k {
            K::Block(lets, r) => {
                let mut uses = reads(r);
                for (_, v) in lets {
                    for (n, k) in reads(v) {
                        *uses.entry(n).or_default() += k;
                    }
                }
                let mut s = String::new();
                for (n, v) in lets {
                    s.push_str(&format!("let {} = {}; ", binder(n, &uses), self.render(v)));
                }
                s.push_str(&self.render(r));
                s
            }
            _ => self.render(e),
        }
    }

    pub(super) fn render(&self, e: &E) -> String {
        let list = |args: &[E]| {
            args.iter()
                .map(|a| self.render(a))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match &e.k {
            K::Lit(s) => s.clone(),
            K::Var(n, _) => n.clone(),
            K::ArrayLit(items) => format!("[{}]", list(items)),
            K::Op(op, args, sugar) => {
                if let Some(sym) = infix(*op) {
                    return format!(
                        "({} {sym} {})",
                        self.render(&args[0]),
                        self.render(&args[1])
                    );
                }
                match op {
                    Op::Reduce(ReduceOp::Add) if *sugar => format!("sum({})", list(args)),
                    Op::Reduce(r) | Op::Scan(r) | Op::ScanBy(r) | Op::ReduceInto(r) => {
                        format!("{}({}, {})", call_name(*op), list(args), reduce_name(*r))
                    }
                    _ => format!("{}({})", call_name(*op), list(args)),
                }
            }
            K::Call(i, args) => format!("{}({})", self.fns[*i].name, list(args)),
            K::If(c, a, b) => format!(
                "(if {} {{ {} }} else {{ {} }})",
                self.render(c),
                self.render_inner(a),
                self.render_inner(b)
            ),
            K::Block(..) => format!("{{ {} }}", self.render_inner(e)),
            K::Higher(kind, arrays, lam) => {
                let name = match kind {
                    Higher::Map => "map",
                    Higher::ZipMap => "zip_map",
                    Higher::Filter => "filter",
                };
                let f = match lam {
                    Lam::Lambda(params, body) => {
                        let uses = reads(body);
                        let params: Vec<String> = params.iter().map(|p| binder(p, &uses)).collect();
                        format!("|{}| {}", params.join(", "), self.render(body))
                    }
                    Lam::Fn(i) => self.fns[*i].name.clone(),
                };
                format!("{name}({}, {f})", list(arrays))
            }
        }
    }
}
