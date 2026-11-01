//! Counts the integer sum rewrites that fired, from the plan lines that the fused and the
//! unfused planner print for one program.
//!
//! A rewrite leaves its right-hand side in the plan, for example
//! `(add_i64_s (reduce_add_i64 x) (mul_i64_s (length_i64 x) k))`. A program can also write
//! that shape itself, and the unfused planner, which drops the rewrites, keeps it. So a
//! rewrite fired in a statement when the fused plan holds more of its right-hand sides than
//! the unfused plan does. A right-hand side matches only when its count is the length of
//! the array it sums, compared structurally. Fused kernels that absorb a right-hand side's
//! sum, and the shapes the matcher leaves out, can only lower the count, so the gate may miss
//! a firing but never counts one that did not happen.

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
    // This mirrors only the output-type gating of the planner's `sum_rules`. A type lacking
    // another kernel would appear here and fail the gate, which is how it gets noticed.
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

#[derive(Debug, PartialEq)]
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

/// The argument of `e` when it is a call to `head` with one argument.
fn arg_of<'a>(e: &'a Sexp, head: &str) -> Option<&'a Sexp> {
    match call(e) {
        Some((h, [x])) if h == head => Some(x),
        _ => None,
    }
}

/// The array that `e` sums, for element type `t`.
fn summed<'a>(e: &'a Sexp, t: &str) -> Option<&'a Sexp> {
    arg_of(e, &format!("reduce_add_{t}"))
}

/// The array whose length `e` counts, for element type `t`.
fn counted<'a>(e: &'a Sexp, t: &str) -> Option<&'a Sexp> {
    let length = format!("length_{t}");
    // `length` is `i64`, which other types cast to the element type.
    arg_of(e, &length)
        .or_else(|| arg_of(e, &format!("cast_i64_{t}_s")).and_then(|c| arg_of(c, &length)))
}

/// Whether `s` sums an array `x` and `p` is the length of the same `x` times some scalar, in
/// either operand order, with `x` compared structurally.
fn sum_and_count(s: &Sexp, p: &Sexp, t: &str) -> bool {
    let Some(x) = summed(s, t) else {
        return false;
    };
    match call(p) {
        Some((h, [a, b])) if h == format!("mul_{t}_s") => {
            counted(a, t) == Some(x) || counted(b, t) == Some(x)
        }
        _ => false,
    }
}

/// Whether `node` is the right-hand side of `rw`. Scalar `+` and `*` commute, so their
/// operands match in either order.
fn matches_rhs(rw: Rewrite, node: &Sexp) -> bool {
    let t = rw.ty.name();
    let Some((head, [a, b])) = call(node) else {
        return false;
    };
    match rw.rule {
        Rule::Add => {
            head == format!("add_{t}_s") && (sum_and_count(a, b, t) || sum_and_count(b, a, t))
        }
        Rule::Sub => head == format!("sub_{t}_s") && sum_and_count(a, b, t),
        Rule::SubFrom => head == format!("sub_{t}_s") && sum_and_count(b, a, t),
        // The product of a count with a sum `k` is part of another rule's right-hand side, so
        // a product of a sum with any count is left out. That misses `sum(x * length(y))`,
        // which only lowers the count.
        Rule::Mul => {
            head == format!("mul_{t}_s")
                && ((summed(a, t).is_some() && counted(b, t).is_none())
                    || (summed(b, t).is_some() && counted(a, t).is_none()))
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
                 plan 4: (mul_i64_s (reduce_add_i64 a) k)\n\
                 plan 5: (add_i64_s (reduce_add_i64 z) (mul_i64_s (reduce_add_i64 x) (length_i64 y)))\n\
                 plan 6: (sub_u64_s (reduce_add_u64 z) (mul_u64_s (cast_i64_u64_s (length_u64 y)) 3))\n";
    let unfused = "plan 1: (reduce_add_i64 (add_i64 a 3))\n\
                   plan 2: (reduce_add_u64 (sub_u64 3 a))\n\
                   plan 3: (mul_i64_s (reduce_add_i64 a) (length_i64 a))\n\
                   plan 4: (mul_i64_s (reduce_add_i64 a) k)\n\
                   plan 5: (add_i64_s (reduce_add_i64 z) (reduce_add_i64 (mul_i64 x (length_i64 y))))\n\
                   plan 6: (sub_u64_s (reduce_add_u64 z) w)\n";
    let got: Vec<String> = fired(&rws, fused, unfused)
        .into_iter()
        .map(|(rw, n)| format!("{rw}: {n}"))
        .collect();
    // Plans 3 and 4 hold the same shapes in both plans, written by the program. Plan 5 is
    // `sum(z) + sum(x * length(y))`, whose count is not the length of the summed `z`. Plan 6
    // is the same for a cast count, and its unfused plan lacks the shape too, so only the
    // structural comparison keeps it from counting.
    assert_eq!(got, ["sum(x + k) on i64: 1", "sum(k - x) on u64: 1"]);
}
