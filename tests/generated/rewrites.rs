//! Counts the integer sum rewrites that fired, from the plan lines that the fused and the
//! unfused planner print for one program.
//!
//! A rewrite leaves its right-hand side in the plan, for example
//! `(add_i64_s (reduce_add_i64 x) (mul_i64_s (length_i64 x) k))`. A program can also write
//! that shape itself, and the unfused planner, which drops the rewrites, keeps it. So a
//! rewrite fired in a statement when the fused plan holds more of its right-hand sides than
//! the unfused plan does. Fused kernels that absorb a right-hand side's sum can only lower
//! the count, so the gate may miss a firing but never counts one that did not happen.

use std::collections::BTreeMap;
use std::fmt;

use verkehrt::kernels::Registry;
use verkehrt::ops::{Op, ReduceOp};
use verkehrt::value::{ScalarType, ValueType};

/// One rewrite, by the arithmetic it moves out of the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// `sum(x + k)` to `sum(x) + n * k`.
    Add,
    /// `sum(x - k)` to `sum(x) - n * k`.
    Sub,
    /// `sum(k - x)` to `n * k - sum(x)`.
    SubFrom,
    /// `sum(x * k)` to `sum(x) * k`.
    Mul,
}

impl Rule {
    const ALL: [Rule; 4] = [Rule::Add, Rule::Sub, Rule::SubFrom, Rule::Mul];
}

/// A rewrite on one element type. Rewrites order by type name, then rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rewrite {
    pub ty: ScalarType,
    pub rule: Rule,
}

impl Ord for Rewrite {
    fn cmp(&self, other: &Rewrite) -> std::cmp::Ordering {
        (self.ty.name(), self.rule).cmp(&(other.ty.name(), other.rule))
    }
}

impl PartialOrd for Rewrite {
    fn partial_cmp(&self, other: &Rewrite) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rewrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = match self.rule {
            Rule::Add => "sum(x + k)",
            Rule::Sub => "sum(x - k)",
            Rule::SubFrom => "sum(k - x)",
            Rule::Mul => "sum(x * k)",
        };
        write!(f, "{shape} on {}", self.ty.name())
    }
}

/// Every rewrite the planner emits: the four rules on each integer type whose sum keeps the
/// element type.
pub fn expected(reg: &Registry) -> Vec<Rewrite> {
    ScalarType::ARRAY_TYPES
        .into_iter()
        .filter(|t| {
            t.is_int()
                && reg
                    .lookup(Op::Reduce(ReduceOp::Add), &[ValueType::Array(*t)])
                    .is_some_and(|id| reg.get(id).output == ValueType::Scalar(*t))
        })
        .flat_map(|ty| Rule::ALL.into_iter().map(move |rule| Rewrite { ty, rule }))
        .collect()
}

/// The rewrites of `rewrites` that fired in some statement, with the number of statements in
/// which each fired. `fused` and `unfused` are the plan lines of one program, one per
/// statement in the same order.
pub fn fired(rewrites: &[Rewrite], fused: &str, unfused: &str) -> BTreeMap<Rewrite, usize> {
    let mut out = BTreeMap::new();
    for (f, u) in fused.lines().zip(unfused.lines()) {
        let (f, u) = (count(rewrites, f), count(rewrites, u));
        for (rw, n) in f {
            if n > u.get(&rw).copied().unwrap_or(0) {
                *out.entry(rw).or_default() += 1;
            }
        }
    }
    out
}

/// The right-hand sides in one plan line `plan <line>: <term>`, counted per rewrite.
fn count(rewrites: &[Rewrite], line: &str) -> BTreeMap<Rewrite, usize> {
    let term = line.split_once(": ").map_or(line, |(_, t)| t);
    let mut tokens = tokenize(term).into_iter().peekable();
    let tree = parse(&mut tokens);
    let mut out = BTreeMap::new();
    visit(&tree, &mut |node| {
        for rw in rewrites {
            if matches_rhs(*rw, node) {
                *out.entry(*rw).or_default() += 1;
            }
        }
    });
    out
}

#[derive(Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn tokenize(s: &str) -> Vec<String> {
    s.replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn parse(tokens: &mut std::iter::Peekable<std::vec::IntoIter<String>>) -> Sexp {
    match tokens.next().as_deref() {
        Some("(") => {
            let mut items = Vec::new();
            while tokens.peek().is_some_and(|t| t != ")") {
                items.push(parse(tokens));
            }
            tokens.next();
            Sexp::List(items)
        }
        Some(atom) => Sexp::Atom(atom.to_string()),
        None => Sexp::List(Vec::new()),
    }
}

fn visit(e: &Sexp, f: &mut impl FnMut(&Sexp)) {
    f(e);
    if let Sexp::List(items) = e {
        items.iter().for_each(|i| visit(i, f));
    }
}

/// The head and arguments of a kernel call.
fn call(e: &Sexp) -> Option<(&str, &[Sexp])> {
    match e {
        Sexp::List(items) => match items.split_first() {
            Some((Sexp::Atom(h), args)) => Some((h.as_str(), args)),
            _ => None,
        },
        Sexp::Atom(_) => None,
    }
}

fn is_call(e: &Sexp, head: &str) -> bool {
    call(e).is_some_and(|(h, _)| h == head)
}

/// Whether `node` is the right-hand side of `rw`. Scalar `+` and `*` commute, so their
/// operands match in either order.
fn matches_rhs(rw: Rewrite, node: &Sexp) -> bool {
    let t = rw.ty.name();
    let Some((head, [a, b])) = call(node) else {
        return false;
    };
    let sum = |e: &Sexp| is_call(e, &format!("reduce_add_{t}"));
    let length = |e: &Sexp| is_call(e, &format!("length_{t}"));
    // `length` is `i64`, which other types cast to the element type.
    let count = |e: &Sexp| {
        length(e)
            || call(e).is_some_and(|(h, args)| {
                h == format!("cast_i64_{t}_s") && args.len() == 1 && length(&args[0])
            })
    };
    let times_count = |e: &Sexp| {
        call(e).is_some_and(|(h, args)| {
            h == format!("mul_{t}_s") && args.len() == 2 && (count(&args[0]) || count(&args[1]))
        })
    };
    match rw.rule {
        Rule::Add => {
            head == format!("add_{t}_s")
                && ((sum(a) && times_count(b)) || (times_count(a) && sum(b)))
        }
        Rule::Sub => head == format!("sub_{t}_s") && sum(a) && times_count(b),
        Rule::SubFrom => head == format!("sub_{t}_s") && times_count(a) && sum(b),
        // The product of a count with a sum `k` is part of another rule's right-hand side.
        Rule::Mul => {
            head == format!("mul_{t}_s") && ((sum(a) && !count(b)) || (sum(b) && !count(a)))
        }
    }
}

#[test]
fn fired_counts_rewritten_plans_only() {
    let reg = Registry::new();
    let rws = expected(&reg);
    let names: Vec<String> = rws.iter().map(ToString::to_string).collect();
    assert_eq!(
        names,
        [
            "sum(x + k) on i64",
            "sum(x - k) on i64",
            "sum(k - x) on i64",
            "sum(x * k) on i64",
            "sum(x + k) on u64",
            "sum(x - k) on u64",
            "sum(k - x) on u64",
            "sum(x * k) on u64",
        ]
    );
    let fused = "plan 1: (add_i64_s (reduce_add_i64 a) (mul_i64_s (length_i64 a) 3))\n\
                 plan 2: (sub_u64_s (mul_u64_s 3 (cast_i64_u64_s (length_u64 a))) (reduce_add_u64 a))\n\
                 plan 3: (mul_i64_s (reduce_add_i64 a) (length_i64 a))\n\
                 plan 4: (mul_i64_s (reduce_add_i64 a) k)\n";
    let unfused = "plan 1: (reduce_add_i64 (add_i64 a 3))\n\
                   plan 2: (reduce_add_u64 (sub_u64 3 a))\n\
                   plan 3: (mul_i64_s (reduce_add_i64 a) (length_i64 a))\n\
                   plan 4: (mul_i64_s (reduce_add_i64 a) k)\n";
    let got: Vec<String> = fired(&rws, fused, unfused)
        .into_iter()
        .map(|(rw, n)| format!("{rw}: {n}"))
        .collect();
    // Plans 3 and 4 hold the same shapes in both plans, written by the program.
    assert_eq!(got, ["sum(x + k) on i64: 1", "sum(k - x) on u64: 1"]);
}
