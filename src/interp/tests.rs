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

/// The ids of the plan nodes of `p` that call the kernel `name`.
fn kernel_nodes(interp: &Interpreter, p: &Prepared, name: &str) -> Vec<PlanId> {
    let graph = &p.plan.graph;
    (0..graph.len())
        .map(|i| PlanId(u32::try_from(i).unwrap()))
        .filter(|&id| {
            matches!(graph.node(id), PlanNode::Kernel(k, _) if interp.reg.get(*k).name == name)
        })
        .collect()
}

#[test]
fn shared_flags_count_uses_along_paths() {
    let interp = Interpreter::new(Options::default()).unwrap();
    let flag = |src: &str, name: &str| {
        let p = interp.prepare(src).unwrap();
        let ids = kernel_nodes(&interp, &p, name);
        assert_eq!(ids.len(), 1, "{name} in\n{src}");
        p.plan.shared[ids[0].index()]
    };
    // Two statements with one root: the reduction has two uses, and the selection below it
    // has one, because the shared reduction runs once.
    let twice = "input a: i64[];\n\
                 print(sum(compress(a, a > 1)));\nprint(sum(compress(a, a > 1)));\n";
    assert!(flag(twice, "reduce_add_i64"));
    assert!(!flag(twice, "filter_gt_i64"));
    // A conditional is never shared, so the reduction in its branch has two uses through
    // the conditional's one edge.
    let cond = "input a: i64[];\ninput s: i64;\n\
                print(if s > 0 { sum(compress(a, a > 1)) } else { 0 });\n\
                print(if s > 0 { sum(compress(a, a > 1)) } else { 0 });\n";
    assert!(flag(cond, "reduce_add_i64"));
    // A subgraph with a load is never shared.
    let load = "print(sum(load_i64(\"x.bin\")));\nprint(sum(load_i64(\"x.bin\")));\n";
    assert!(!flag(load, "reduce_add_i64"));
    assert!(!flag(load, "load_i64"));
    // One use is not shared.
    let once = "input a: i64[];\nprint(sum(compress(a, a > 1)));\n";
    assert!(!flag(once, "reduce_add_i64"));
}

#[test]
fn same_plans_agrees_with_the_plan_lines() {
    let planner = |no_fusion| {
        Interpreter::new(Options {
            show_plan: true,
            no_fusion,
            ..Options::default()
        })
        .unwrap()
    };
    let (fused, unfused) = (planner(false), planner(true));
    let prepared = |interp: &Interpreter, src: &str| {
        let mut text = Vec::new();
        let p = interp.prepare_to(src, &mut text).unwrap();
        (p, String::from_utf8(text).unwrap())
    };
    for src in [
        // Fusion changes the plan.
        "input a: f64[];\ninput b: f64[];\nprint(sum(a * b));\n",
        "input a: i64[];\nprint(sum(a + 3));\n",
        // Fusion changes nothing: broadcasts, constants, and conditionals.
        "input a: i64[];\nprint(a + 1);\nprint([1, 2] * 2);\n",
        "input a: f64[];\ninput s: i64;\nprint(if s > 0 { a + 1.5 } else { a });\n",
        "input a: i64[];\nprint(length(a));\n",
    ] {
        let (pf, tf) = prepared(&fused, src);
        let (pu, tu) = prepared(&unfused, src);
        assert_eq!(pf.same_plans(&pu), tf == tu, "{src}\n{tf}\n{tu}");
        assert!(pf.same_plans(&prepared(&fused, src).0), "{src}");
    }
    // Programs whose statements sit at other lines differ.
    let (one, _) = prepared(&fused, "input a: i64[];\nprint(a + 1);\n");
    let (other, _) = prepared(&fused, "input a: i64[];\n\nprint(a + 1);\n");
    assert!(!one.same_plans(&other));
}

/// With `no_composite_sort`, `argsort(a, b)` plans as two single-key sorts, two gathers and
/// one length check. The sort of `b` is one node of the plan graph, so it has no copy for
/// the second gather: two `argsort_i64` nodes are the sorts of `b` and of the gathered `a`.
#[test]
fn expanded_composite_sort_shares_the_sort_of_the_minor_key() {
    let off = Interpreter::new(Options {
        no_composite_sort: true,
        ..Options::default()
    })
    .unwrap();
    let src = "input a: i64[];\ninput b: i64[];\nprint(argsort(a, b));\n";
    let expanded = off.prepare(src).unwrap();
    assert_eq!(kernel_nodes(&off, &expanded, "argsort_i64").len(), 2);
    assert_eq!(kernel_nodes(&off, &expanded, "gather_i64").len(), 2);
    assert_eq!(kernel_nodes(&off, &expanded, "same_len_i64").len(), 1);
    assert!(kernel_nodes(&off, &expanded, "argsort2_i64_i64").is_empty());
    // The default plans the composite kernel and no expansion.
    let on = Interpreter::new(Options::default()).unwrap();
    let composite = on.prepare(src).unwrap();
    assert_eq!(kernel_nodes(&on, &composite, "argsort2_i64_i64").len(), 1);
    assert!(kernel_nodes(&on, &composite, "argsort_i64").is_empty());
}
