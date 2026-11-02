use super::*;
use crate::lower::budget;
use crate::plan::{PlanId, PlanNode};

/// Prepares `src` within `limit` steps of the passes over the term graph and of planning,
/// and returns the program with the steps taken. A pass that lost its memo panics on the
/// budget.
fn prepared_within(src: &str, limit: usize) -> (Prepared, usize) {
    let interp = Interpreter::new(Options::default()).unwrap();
    let (prepared, taken) = budget::run(limit, || interp.prepare(src));
    (prepared.unwrap_or_else(|e| panic!("{src}: {e}")), taken)
}

#[test]
fn nested_calls_prepare_to_a_linear_plan_graph() {
    // The parameter is read four times, so the plan as a tree has 4^16 copies of the
    // innermost argument, far beyond the step budget.
    let depth = 16;
    let mut src = String::from(
        "fn f(q: i64[]) -> i64[] { let b = filter(q, |p| (p > 3)); ((b + b) + (b + b)) }\n\
         input a: i64[];\nprint(sum(",
    );
    src.push_str(&"f(".repeat(depth));
    src.push('a');
    src.push_str(&")".repeat(depth));
    src.push_str("));\n");
    let c = 64;
    let (p, steps) = prepared_within(&src, c * depth);
    let n = p.plan.graph.len();
    assert!(
        n <= c * depth,
        "{n} plan nodes for depth {depth}, {steps} steps"
    );
}

#[test]
fn nested_lazy_calls_prepare_to_a_linear_plan_graph() {
    // The taken branch reads the parameter twice and stays unhoisted, so the plan as a
    // tree has 2^24 copies of the innermost argument.
    let depth = 24;
    let mut src = String::from(
        "fn f(q: f64[], s: f64) -> f64[] { if s > 0.0 { q + q } else { q } }\n\
         input a: f64[];\ninput s: f64;\nprint(sum(",
    );
    src.push_str(&"f(".repeat(depth));
    src.push('a');
    src.push_str(&", s)".repeat(depth));
    src.push_str("));\n");
    let c = 64;
    let (p, steps) = prepared_within(&src, c * depth);
    let n = p.plan.graph.len();
    assert!(
        n <= c * depth,
        "{n} plan nodes for depth {depth}, {steps} steps"
    );
}

#[test]
fn block_let_chain_in_a_branch_prepares_to_a_linear_plan_graph() {
    // Each let reads the previous one twice inside a lazy branch, which hoisting leaves
    // alone, so one emitted term holds the whole chain and its extraction as a tree has
    // 2^24 leaves.
    let depth = 24;
    let mut src = String::from("input a0: i64[];\ninput s: i64;\nprint(sum(if s > 0 { ");
    for i in 1..=depth {
        let p = i - 1;
        src.push_str(&format!("let a{i} = a{p} + a{p}; "));
    }
    src.push_str(&format!("a{depth} }} else {{ a0 }}));\n"));
    let c = 16;
    let (p, steps) = prepared_within(&src, c * depth);
    let n = p.plan.graph.len();
    assert!(
        n <= c * depth,
        "{n} plan nodes for depth {depth}, {steps} steps"
    );
}

#[test]
fn a_subplan_that_only_planning_repeats_is_one_node() {
    // The integer sum rewrites turn both sums into plans over `(reduce_add_i64 r)`, which
    // no term of the program contains, as in ClickBench q29.
    let src = "input r: i64[];\nprint(sum(r + 1));\nprint(sum(r + 2));\n";
    let (p, _) = prepared_within(src, 1024);
    let interp = Interpreter::new(Options::default()).unwrap();
    let sum = interp.reg.by_name("reduce_add_i64").unwrap();
    let graph = &p.plan.graph;
    let sums = (0..graph.len())
        .map(|i| PlanId(u32::try_from(i).unwrap()))
        .filter(|&id| matches!(graph.node(id), PlanNode::Kernel(k, _) if *k == sum))
        .count();
    assert_eq!(sums, 1);
    assert_ne!(p.plan.roots[0], p.plan.roots[1]);
}

#[test]
fn a_memoized_plan_reports_its_sum_rewrites() {
    // The second statement of each pair hits the planning memo, for its root in the first
    // pair and for the conditional in the second.
    let src = "input a: i64[];\ninput s: i64;\n\
               print(sum(a + 3));\nprint(sum(a + 3));\n\
               print(1 + (if s > 0 { sum(a + 3) } else { 0 }));\n\
               print(2 + (if s > 0 { sum(a + 3) } else { 0 }));\n";
    let interp = Interpreter::new(Options::default()).unwrap();
    let (_, fired) = interp
        .prepare_with_sum_rewrites(src, &mut std::io::sink())
        .unwrap();
    let want = BTreeSet::from([SumRewrite {
        ty: crate::value::ScalarType::I64,
        rule: crate::plan::SumRule::Add,
    }]);
    assert_eq!(fired, vec![want; 4]);
}

#[test]
fn identical_statements_share_one_plan_root() {
    let src = "input a: i64[];\nprint(sum(a + 1));\nprint(sum(a + 1));\n";
    let (p, _) = prepared_within(src, 1024);
    let roots = &p.plan.roots;
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0], roots[1]);
}
