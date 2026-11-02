use super::*;
use crate::check::{Globals, check};
use crate::kernels::{InputSpec, Registry};
use crate::lower::{LStmt, Lowered, Mode, budget, lower};
use crate::syntax::parse;
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Pins down the egglog 3 surface the planner relies on: per-constructor `:cost` in a
/// `datatype`, `rewrite`, a `let` global, `run`, and `extract` returning `ExtractBest`.
#[test]
fn egglog_smoke() {
    let program = "
        (datatype E (A i64 :cost 1) (Slow E :cost 10) (Fast E :cost 2))
        (rewrite (Slow ?x) (Fast ?x))
        (let root (Slow (A 7)))
        (run 3)
        (extract root)
    ";
    let mut eg = ::egglog::EGraph::default();
    let outputs = eg
        .parse_and_run_program(None, program)
        .expect("program runs");
    let best = outputs
        .iter()
        .find_map(|o| match o {
            ::egglog::CommandOutput::ExtractBest(dag, cost, term) => {
                Some((dag.to_string(*term), *cost))
            }
            _ => None,
        })
        .expect("extract produces a term");
    // The i64 literal inside `A` costs 1 on top of the declared constructor costs.
    assert_eq!(best, ("(Fast (A 7))".to_string(), 4));
}

const XY: &str = "let x = [1.0, 2.0, 3.0]; let y = [4.0, 5.0, 6.0]; ";

/// Plans every `print` of `XY` followed by `src` in `Mode::Run`, with fused patterns when
/// `fusion` is set. Returns the registry, the plan graph, the plan of each `print` in
/// source order, and its rendering.
fn plan_nodes(src: &str, fusion: bool) -> (Registry, PlanGraph, Vec<(PlanId, String)>) {
    let src = format!("{XY}{src}");
    let prog = parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let reg = Registry::new();
    let mut g = Globals::new();
    let typed = check(&prog, &reg, &mut g).unwrap_or_else(|e| panic!("{src}: {e}"));
    let lowered = lower(&typed, &mut g, &reg, Mode::Run).unwrap_or_else(|e| panic!("{src}: {e}"));
    let planner = Planner::new(&reg, fusion);
    let mut cx = PlanContext::default();
    let plans = lowered
        .stmts
        .iter()
        .filter_map(|s| match s {
            LStmt::Print { term, span } => Some((*term, *span)),
            _ => None,
        })
        .map(|(term, span)| {
            let id = planner
                .plan(&reg, &lowered.graph, term, span, &mut cx)
                .unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_broadcasts_fit(&reg, &cx.graph, id);
            let rendered = render(&reg, &cx.graph, id, &lowered.leaves, &g);
            (id, rendered)
        })
        .collect();
    (reg, cx.graph, plans)
}

/// Plans like `plan_nodes`, returning the root kernel name (`-` for a bare leaf) and the
/// rendered plan of each `print`.
fn plan_prints(src: &str, fusion: bool) -> Vec<(String, String)> {
    let (reg, graph, plans) = plan_nodes(src, fusion);
    plans
        .into_iter()
        .map(|(id, rendered)| {
            let root = match graph.node(id) {
                PlanNode::Kernel(k, _) => reg.get(*k).name.clone(),
                _ => "-".to_string(),
            };
            (root, rendered)
        })
        .collect()
}

/// Asserts that a broadcast only feeds an `Either` input, which is where single-op array
/// kernels accept one, and that `Array` and `Scalar` inputs never receive a broadcast.
fn assert_broadcasts_fit(reg: &Registry, graph: &PlanGraph, id: PlanId) {
    match graph.node(id) {
        PlanNode::Leaf(_) => {}
        PlanNode::Bcast(inner) => assert_broadcasts_fit(reg, graph, *inner),
        PlanNode::Cond(c, t, e) => [c, t, e]
            .into_iter()
            .for_each(|x| assert_broadcasts_fit(reg, graph, *x)),
        PlanNode::Kernel(k, args) => {
            let def = reg.get(*k);
            for (spec, &arg) in def.inputs.iter().zip(args) {
                if matches!(graph.node(arg), PlanNode::Bcast(_)) {
                    assert!(
                        matches!(spec, InputSpec::Either(_)),
                        "{} got a broadcast",
                        def.name
                    );
                }
                assert_broadcasts_fit(reg, graph, arg);
            }
        }
    }
}

/// Runs every `plan` directive in `testdata/*.dd`. The input is a source that follows
/// `XY`. The output has one line per `print` in source order, `<root kernel>: <plan>`,
/// where the root is `-` for a bare leaf or conditional. The argument `fusion=off` plans
/// without fused patterns.
#[test]
fn datadriven_plans() {
    crate::testutil::run_dir(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/plan/testdata"),
        |tc| {
            crate::testutil::reject_unknown_args(tc, &["fusion"]);
            // `fusion=off` plans without fused patterns, the default plans with them.
            let fusion = match tc.args.get("fusion").map(|v| v[0].as_str()) {
                None | Some("on") => true,
                Some("off") => false,
                Some(v) => panic!("fusion takes on or off, not {v}"),
            };
            match tc.directive.as_str() {
                "plan" => plan_prints(&tc.input, fusion)
                    .into_iter()
                    .map(|(root, plan)| format!("{root}: {plan}\n"))
                    .collect::<String>(),
                d => panic!("unknown directive {d}"),
            }
        },
    );
}

/// Collects the name of every kernel in the plan at `id`.
fn kernel_names(reg: &Registry, graph: &PlanGraph, id: PlanId, out: &mut BTreeSet<String>) {
    match graph.node(id) {
        PlanNode::Leaf(_) => {}
        PlanNode::Bcast(inner) => kernel_names(reg, graph, *inner, out),
        PlanNode::Cond(c, t, e) => [c, t, e]
            .into_iter()
            .for_each(|x| kernel_names(reg, graph, *x, out)),
        PlanNode::Kernel(k, args) => {
            out.insert(reg.get(*k).name.clone());
            args.iter().for_each(|a| kernel_names(reg, graph, *a, out));
        }
    }
}

/// One program per element type, with the natural source of every fused kernel: dot, sumsq,
/// and count, filter, and conditional sum for each of the six comparisons.
fn fused_sources() -> String {
    let mut src = String::new();
    // Element type, literal suffix, and a literal for each of the two array elements.
    for (ty, suffix, one, two) in [
        ("f64", "", "1.0", "2.0"),
        ("f32", "f32", "1.0", "2.0"),
        ("i64", "i64", "1", "2"),
        ("u64", "u64", "1", "2"),
        ("i32", "i32", "1", "2"),
    ] {
        let (a, b) = (format!("a_{ty}"), format!("b_{ty}"));
        let (c, e) = (format!("{one}{suffix}"), format!("{two}{suffix}"));
        writeln!(
            src,
            "let {a} = [{one}{suffix}, {two}{suffix}]; let {b} = [{two}{suffix}, {one}{suffix}];"
        )
        .unwrap();
        writeln!(src, "print(sum({a} * {b})); print(sum({a} * {a}));").unwrap();
        for cmp in ["==", "!=", "<", "<=", ">", ">="] {
            writeln!(
                src,
                "print(count({a} {cmp} {c})); print(filter({a}, |v| v {cmp} {c})); \
                 print(sum(map({a}, |v| if v {cmp} {c} {{ v }} else {{ {e} }})));"
            )
            .unwrap();
        }
    }
    src
}

/// With fusion off no plan holds a fused kernel. With fusion on the same programs produce
/// every fused kernel of the registry, so the off check is not vacuous.
#[test]
fn fusion_off_plans_use_single_op_kernels_only() {
    let src = fused_sources();
    let (reg, off_graph, off) = plan_nodes(&src, false);
    for (id, rendered) in &off {
        let mut names = BTreeSet::new();
        kernel_names(&reg, &off_graph, *id, &mut names);
        for name in names {
            let def = reg.get(reg.by_name(&name).expect("kernel exists"));
            assert!(
                def.pattern.is_none(),
                "fusion off planned {name}: {rendered}"
            );
        }
    }
    let (on_reg, on_graph, on) = plan_nodes(&src, true);
    let mut seen = BTreeSet::new();
    for (id, _) in &on {
        kernel_names(&on_reg, &on_graph, *id, &mut seen);
    }
    let missing: Vec<_> = on_reg
        .iter()
        .map(|(_, d)| d)
        .filter(|d| d.pattern.is_some() && !seen.contains(&d.name))
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "fusion on never planned fused kernels {missing:?}"
    );
}

/// The lazy-branch form of #39 at `depth`: a block let bound to a lazy conditional,
/// compressed through a comparison with its own reduction, then nested lazy conditionals
/// whose taken branch reads the previous value three times and whose other branch reads it
/// once. As a tree the term repeats the innermost value `4^depth` times.
fn lazy_branch_program(depth: usize) -> (Registry, Lowered) {
    let mut src = String::from(
        "input a: i64[]; input s: i64;\nprint({ let b7 = if false { a } else { a }; \
         let c0 = compress(b7, (b7 >= reduce(b7, and))); ",
    );
    for i in 1..=depth {
        let p = i - 1;
        src.push_str(&format!(
            "let c{i} = if s > {i} {{ compress(c{p}, (c{p} >= reduce(c{p}, and))) }} \
             else {{ c{p} }}; "
        ));
    }
    src.push_str(&format!("c{depth} }});\n"));
    let prog = parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let reg = Registry::new();
    let mut g = Globals::new();
    let typed = check(&prog, &reg, &mut g).unwrap_or_else(|e| panic!("{src}: {e}"));
    let lowered = lower(&typed, &mut g, &reg, Mode::Run).unwrap_or_else(|e| panic!("{src}: {e}"));
    (reg, lowered)
}

/// Emitting every term the planner emits for the lazy-branch form, each distinct
/// conditional part and hole once, writes a number of `let` commands and holes linear in
/// the depth, and each emission has one hole at most. Without the emission memo the
/// repeated holes are emitted again.
#[test]
fn lazy_branch_nesting_emits_linearly() {
    let depth = 16;
    let (reg, lowered) = lazy_branch_program(depth);
    let graph = &lowered.graph;
    let (mut lets, mut holes) = (0, 0);
    let mut seen = BTreeSet::new();
    let mut todo: Vec<_> = lowered.stmts.iter().map(LStmt::term).collect();
    let c = 16;
    let ((), steps) = budget::run(c * depth, || {
        while let Some(t) = todo.pop() {
            if !seen.insert(t) {
                continue;
            }
            if let TermKind::Cond(k, a, b) = *graph.kind(t) {
                todo.extend([k, a, b]);
                continue;
            }
            let (commands, _, hs) = egglog::emit(&reg, graph, t).unwrap();
            lets += commands.matches("(let ").count();
            assert!(hs.len() <= 1, "{} holes in one emission", hs.len());
            holes += hs.len();
            todo.extend(hs);
        }
    });
    assert!(lets <= c * depth, "{lets} lets for depth {depth}");
    assert!(holes <= depth + 1, "{holes} holes for depth {depth}");
    assert!(steps <= lets, "{steps} emission steps for {lets} lets");
}

/// Planning the lazy-branch form at `depth` plans each distinct conditional part and hole
/// once and maps each extracted DAG node once, so the planner takes a number of steps
/// linear in the depth, and the plan graph has a number of nodes linear in the depth.
/// Without the planning memo it plans a hole once per occurrence, `3^depth` times.
fn assert_lazy_branch_plans_linearly(depth: usize) {
    let (reg, lowered) = lazy_branch_program(depth);
    let planner = Planner::new(&reg, true);
    let mut cx = PlanContext::default();
    let c = 32;
    for s in &lowered.stmts {
        let (plan, steps) = budget::run(c * depth, || {
            planner.plan(&reg, &lowered.graph, s.term(), s.span(), &mut cx)
        });
        plan.unwrap_or_else(|e| panic!("{e} after {steps} steps"));
    }
    let n = cx.graph.len();
    assert!(n <= c * depth, "{n} plan nodes for depth {depth}");
}

#[test]
fn lazy_branch_nesting_plans_in_linear_steps() {
    assert_lazy_branch_plans_linearly(8);
}

#[test]
fn deep_lazy_branch_nesting_plans_in_linear_steps() {
    assert_lazy_branch_plans_linearly(16);
}

/// A conditional repeated inside one emitted term is one hole, planned once, so a fused
/// pattern that reads a variable twice matches across the repeats. Each conditional
/// branch here reads the inner conditional `y` twice.
#[test]
fn repeated_conditional_in_a_branch_is_one_hole() {
    let plans = plan_prints(
        "let n = length(x);\n\
         print(if n > 1 { let y = if n > 2 { x } else { x + 1.0 }; sum(y * y) } else { 0.0 });\n\
         print(if n > 1 { let y = if n > 2 { x } else { x + 1.0 }; filter(y, |v| v > 1.5) } \
         else { [0.0] });\n",
        true,
    );
    let rendered: Vec<&str> = plans.iter().map(|(_, r)| r.as_str()).collect();
    assert_eq!(
        rendered,
        [
            "(if (lt_i64_s 1 n) then (sumsq_f64 (if (lt_i64_s 2 n) then x else (add_f64 x 1.0))) \
             else 0.0)",
            "(if (lt_i64_s 1 n) then (filter_gt_f64 (if (lt_i64_s 2 n) then x else \
             (add_f64 x 1.0)) 1.5) else <f64[1]>)",
        ]
    );
}
