//! The features of a generated program, for the histogram.

use std::collections::{BTreeMap, BTreeSet};

use verkehrt::ops::{Op, ReduceOp};

use super::{CMPS, E, Gen, Higher, K, Lam, is_array, is_elementwise};

/// The feature name of an op in the histogram.
pub(super) fn feature(op: Op) -> &'static str {
    match op {
        Op::Add => "op +",
        Op::Sub => "op -",
        Op::Mul => "op *",
        Op::Div => "op /",
        Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => "op comparison",
        Op::Neg => "op neg",
        Op::Abs => "op abs",
        Op::Min | Op::Max => "op min/max",
        Op::Sqrt => "op sqrt",
        Op::And | Op::Or | Op::Xor | Op::Not => "op and/or/xor/not",
        Op::Select => "op select",
        Op::Cast(_) => "op cast",
        Op::Reduce(ReduceOp::Add) => "reduce add",
        Op::Reduce(ReduceOp::Mul) => "reduce mul",
        Op::Reduce(ReduceOp::Min | ReduceOp::Max) => "reduce min/max",
        Op::Reduce(_) => "reduce and/or/xor",
        Op::Scan(_) => "scan",
        Op::ScanBy(_) => "scan_by",
        Op::ReduceInto(_) => "reduce_into",
        Op::Count => "count",
        Op::Any | Op::All => "any/all",
        Op::ArgMin | Op::ArgMax => "argmin/argmax",
        Op::Compress => "compress",
        Op::Where => "where",
        Op::Prev => "prev",
        Op::Iota => "iota",
        Op::Fill => "fill",
        Op::Gather => "gather",
        Op::Slice => "slice",
        Op::Histogram => "histogram",
        Op::DivConst => "div_const",
        Op::ArgSort => "argsort",
        Op::GroupId => "group_id",
        Op::RandI64 | Op::RandF64 => "rand",
        Op::Length => "length",
        _ => "op other",
    }
}

pub(super) fn walk(
    e: &E,
    lambda: bool,
    out: &mut BTreeSet<&'static str>,
    uses: &mut BTreeMap<String, usize>,
) {
    match &e.k {
        K::Lit(_) => {}
        K::Var(n, captured) => {
            *uses.entry(n.clone()).or_default() += 1;
            if *captured {
                out.insert("lambda captures a scalar");
            }
        }
        K::ArrayLit(items) => {
            out.insert("array literal");
            items.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::Op(op, args, _) => {
            out.insert(feature(*op));
            if is_elementwise(*op)
                && args.iter().any(|x| !is_array(x.ty))
                && args.iter().any(|x| is_array(x.ty))
            {
                out.insert("scalar broadcast");
            }
            if lambda {
                out.insert("op in lambda");
            }
            match (op, args.first().map(|a| &a.k)) {
                (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Mul, xs, _))) => {
                    out.insert(
                        if matches!((&xs[0].k, &xs[1].k), (K::Var(a, _), K::Var(b, _)) if a == b) {
                            "shape sum(x * x)"
                        } else {
                            "shape sum(x * y)"
                        },
                    );
                }
                (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Select, _, _))) => {
                    out.insert("shape sum(select(..))");
                }
                (Op::Count, Some(K::Op(c, _, _))) if CMPS.contains(c) => {
                    out.insert("shape count(x cmp c)");
                }
                (Op::Compress, _) if matches!(&args[1].k, K::Op(c, _, _) if CMPS.contains(c)) => {
                    out.insert("shape compress(x, x cmp c)");
                }
                _ => {}
            }
            args.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::Call(_, args) => {
            out.insert(if lambda {
                "fn call in lambda"
            } else {
                "fn call"
            });
            args.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::If(c, a, b) => {
            out.insert(if lambda {
                "if in lambda"
            } else if is_array(c.ty) {
                "if on mask"
            } else {
                "if on scalar"
            });
            walk(c, lambda, out, uses);
            walk(a, lambda, out, uses);
            walk(b, lambda, out, uses);
        }
        K::Block(lets, r) => {
            out.insert("block");
            for (_, v) in lets {
                walk(v, lambda, out, uses);
            }
            walk(r, lambda, out, uses);
        }
        K::Higher(kind, arrays, lam) => {
            out.insert(match kind {
                Higher::Map => "map",
                Higher::ZipMap => "zip_map",
                Higher::Filter => "filter",
            });
            arrays.iter().for_each(|x| walk(x, lambda, out, uses));
            match lam {
                Lam::Lambda(_, body) => walk(body, true, out, uses),
                Lam::Fn(_) => {
                    out.insert("fn as lambda");
                }
            }
        }
    }
}

impl Gen<'_, '_> {
    /// The features of the program: what its expressions use, and how often its top-level
    /// lets are used.
    /// Also returns the number of reads of each variable.
    pub(super) fn features(
        &self,
        stmts: &[(&'static str, String, E)],
    ) -> (BTreeSet<&'static str>, BTreeMap<String, usize>) {
        let mut out = BTreeSet::new();
        let mut uses: BTreeMap<String, usize> = BTreeMap::new();
        for f in &self.fns {
            out.insert(if f.liftable {
                "fn liftable"
            } else {
                "fn with arrays"
            });
            walk(&f.body, f.liftable, &mut out, &mut uses);
        }
        for (_, _, e) in stmts {
            walk(e, false, &mut out, &mut uses);
        }
        for (kind, name, _) in stmts {
            if *kind == "let" {
                out.insert(match uses.get(name).copied().unwrap_or(0) {
                    0 => "let exported at the end",
                    1 => "let used once",
                    _ => "let used more than once",
                });
            }
        }
        (out, uses)
    }
}
