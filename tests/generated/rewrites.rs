//! Counts the integer sum rewrites that fired, from the rewrites that the planner reports
//! as matched for each statement.

use std::collections::{BTreeMap, BTreeSet};

use verkehrt::interp::{Interpreter, Options};
use verkehrt::kernels::{Isa, Registry};
use verkehrt::ops::{Op, ReduceOp};
use verkehrt::plan::{SumRewrite, SumRule};
use verkehrt::value::{ScalarType, ValueType};

/// Every rewrite the planner emits: the four rules on each integer type whose sum keeps the
/// element type.
pub fn expected(reg: &Registry) -> Vec<SumRewrite> {
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
        .flat_map(|ty| {
            SumRule::ALL
                .into_iter()
                .map(move |rule| SumRewrite { ty, rule })
        })
        .collect()
}

/// The rewrites that fired in some statement, with the number of statements in which each
/// fired. `stmts` holds the rewrites the planner reports per statement of one program.
pub fn fired(stmts: &[BTreeSet<SumRewrite>]) -> BTreeMap<SumRewrite, usize> {
    let mut out = BTreeMap::new();
    for rw in stmts.iter().flatten() {
        *out.entry(*rw).or_default() += 1;
    }
    out
}

/// The rewrites that fired per statement of `src`, planned with fusion unless `no_fusion`.
fn plan_fired(src: &str, no_fusion: bool) -> Vec<Vec<String>> {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        no_fusion,
        ..Options::default()
    })
    .expect("scalar is always available");
    let (_, stmts) = interp
        .prepare_with_sum_rewrites(src, &mut std::io::sink())
        .expect("the program compiles");
    stmts
        .iter()
        .map(|s| s.iter().map(ToString::to_string).collect())
        .collect()
}

/// One statement per rewrite, each the source shape of exactly that rewrite.
const ONE_PER_RULE: &str = "input a: i64[]; input k: i64; input b: u64[]; input m: u64;\n\
    print(sum(a + k)); print(sum(a - k)); print(sum(k - a)); print(sum(a * k));\n\
    print(sum(b + m)); print(sum(b - m)); print(sum(m - b)); print(sum(b * m));\n";

#[test]
fn expected_lists_both_wrapping_integer_types() {
    let names: Vec<String> = expected(&Registry::new())
        .iter()
        .map(ToString::to_string)
        .collect();
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
}

#[test]
fn each_rewrite_reports_itself_only() {
    let got = plan_fired(ONE_PER_RULE, false);
    let want: Vec<Vec<String>> = expected(&Registry::new())
        .iter()
        .map(|rw| vec![rw.to_string()])
        .collect();
    assert_eq!(got, want);
}

#[test]
fn unfused_plans_report_no_rewrite() {
    let got = plan_fired(ONE_PER_RULE, true);
    assert_eq!(got, vec![Vec::<String>::new(); 8]);
}

#[test]
fn rewritten_shape_counts_only_the_rewrite_that_ran() {
    // The `Mul` rewrite turns the second sum into `sum(b) * length(a)`, which next to
    // `sum(a)` is the right-hand side of `Add`, although `Add` never matched.
    let got = plan_fired(
        "input a: i64[]; input b: i64[];\nprint(sum(a) + sum(b * length(a)));\n",
        false,
    );
    assert_eq!(got, [["sum(x * k) on i64"]]);
}
