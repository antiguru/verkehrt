//! The features of a generated program, for the histogram.

use std::collections::{BTreeMap, BTreeSet};

use verkehrt::ops::{Op, ReduceOp};

use super::{CMPS, E, FnDef, Gen, Higher, K, Lam, is_array, is_elementwise};

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

/// Counts the reads of each variable in `e` and collects its features, syntactically: every
/// subexpression counts, function bodies are not entered.
pub(super) fn walk(
    e: &E,
    lambda: bool,
    out: &mut BTreeSet<&'static str>,
    uses: &mut BTreeMap<String, usize>,
) {
    let mut w = Walk::new(None);
    w.visit(e, lambda);
    out.append(&mut w.out);
    for (n, k) in w.uses {
        *uses.entry(n).or_default() += k;
    }
}

/// The variables `e` reads, syntactically.
fn reads(e: &E) -> BTreeMap<String, usize> {
    let mut uses = BTreeMap::new();
    walk(e, false, &mut BTreeSet::new(), &mut uses);
    uses
}

/// A walk over expressions that collects features and counts reads. With the functions, it
/// follows only code that runs: a call enters the function body, and a block let that nothing
/// reads is skipped, because lowering computes only what reaches a print or an export.
struct Walk<'f> {
    fns: Option<&'f [FnDef]>,
    out: BTreeSet<&'static str>,
    uses: BTreeMap<String, usize>,
    called: BTreeSet<usize>,
}

impl<'f> Walk<'f> {
    fn new(fns: Option<&'f [FnDef]>) -> Walk<'f> {
        Walk {
            fns,
            out: BTreeSet::new(),
            uses: BTreeMap::new(),
            called: BTreeSet::new(),
        }
    }

    /// Enters the body of function `i` if this walk follows calls.
    fn call(&mut self, i: usize, lambda: bool) {
        let Some(fns) = self.fns else { return };
        self.called.insert(i);
        let f = &fns[i];
        self.out.insert(if f.liftable {
            "fn liftable called"
        } else {
            "fn with arrays called"
        });
        self.visit(&f.body, lambda);
    }

    fn visit(&mut self, e: &E, lambda: bool) {
        match &e.k {
            K::Lit(_) => {}
            K::Var(n, captured) => {
                *self.uses.entry(n.clone()).or_default() += 1;
                if *captured {
                    self.out.insert("lambda captures a scalar");
                }
            }
            K::ArrayLit(items) => {
                self.out.insert("array literal");
                items.iter().for_each(|x| self.visit(x, lambda));
            }
            K::Op(op, args, _) => {
                self.out.insert(feature(*op));
                if is_elementwise(*op)
                    && args.iter().any(|x| !is_array(x.ty))
                    && args.iter().any(|x| is_array(x.ty))
                {
                    self.out.insert("scalar broadcast");
                }
                if lambda {
                    self.out.insert("op in lambda");
                }
                match (op, args.first().map(|a| &a.k)) {
                    (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Mul, xs, _))) => {
                        self.out.insert(
                            if matches!((&xs[0].k, &xs[1].k), (K::Var(a, _), K::Var(b, _)) if a == b)
                            {
                                "shape sum(x * x)"
                            } else {
                                "shape sum(x * y)"
                            },
                        );
                    }
                    (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Select, _, _))) => {
                        self.out.insert("shape sum(select(..))");
                    }
                    (Op::Count, Some(K::Op(c, _, _))) if CMPS.contains(c) => {
                        self.out.insert("shape count(x cmp c)");
                    }
                    (Op::Compress, _) if matches!(&args[1].k, K::Op(c, _, _) if CMPS.contains(c)) =>
                    {
                        self.out.insert("shape compress(x, x cmp c)");
                    }
                    _ => {}
                }
                args.iter().for_each(|x| self.visit(x, lambda));
            }
            K::Call(i, args) => {
                self.out.insert(if lambda {
                    "fn call in lambda"
                } else {
                    "fn call"
                });
                args.iter().for_each(|x| self.visit(x, lambda));
                self.call(*i, lambda);
            }
            K::If(c, a, b) => {
                self.out.insert(if lambda {
                    "if in lambda"
                } else if is_array(c.ty) {
                    "if on mask"
                } else {
                    "if on scalar"
                });
                self.visit(c, lambda);
                self.visit(a, lambda);
                self.visit(b, lambda);
            }
            K::Block(lets, r) => {
                // A let is live when the result or a later live let reads it.
                let mut live: BTreeMap<String, usize> = reads(r);
                let mut keep = vec![self.fns.is_none(); lets.len()];
                for (i, (n, v)) in lets.iter().enumerate().rev() {
                    if live.contains_key(n) {
                        keep[i] = true;
                        live.append(&mut reads(v));
                    }
                }
                if keep.contains(&true) {
                    self.out.insert("block");
                }
                if keep.contains(&false) {
                    self.out.insert("block let kept unread");
                }
                for ((_, v), k) in lets.iter().zip(keep) {
                    if k {
                        self.visit(v, lambda);
                    }
                }
                self.visit(r, lambda);
            }
            K::Higher(kind, arrays, lam) => {
                self.out.insert(match kind {
                    Higher::Map => "map",
                    Higher::ZipMap => "zip_map",
                    Higher::Filter => "filter",
                });
                arrays.iter().for_each(|x| self.visit(x, lambda));
                match lam {
                    Lam::Lambda(_, body) => self.visit(body, true),
                    Lam::Fn(i) => {
                        self.out.insert("fn as lambda");
                        self.call(*i, true);
                    }
                }
            }
        }
    }
}

impl Gen<'_, '_> {
    /// The features of the code that runs: the statements and, through their calls, the
    /// function bodies. Also returns the number of reads of each variable in that code.
    pub(super) fn features(
        &self,
        stmts: &[(&'static str, String, E)],
    ) -> (BTreeSet<&'static str>, BTreeMap<String, usize>) {
        let mut w = Walk::new(Some(&self.fns));
        for (_, _, e) in stmts {
            w.visit(e, false);
        }
        if !self.fns.is_empty() {
            w.out.insert("fn defined");
        }
        if !w.called.is_empty() {
            w.out.insert("fn called");
        }
        if (0..self.fns.len()).any(|i| !w.called.contains(&i)) {
            w.out.insert("fn defined but uncalled");
        }
        for (kind, name, _) in stmts {
            if *kind == "let" {
                w.out.insert(match w.uses.get(name).copied().unwrap_or(0) {
                    0 => "let exported at the end",
                    1 => "let used once",
                    _ => "let used more than once",
                });
            }
        }
        (w.out, w.uses)
    }
}
