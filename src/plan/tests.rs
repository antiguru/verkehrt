use super::*;
use crate::check::{Globals, check};
use crate::kernels::{InputSpec, Registry};
use crate::lower::{LStmt, Mode, lower};
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
/// `fusion` is set. Returns the registry, the plan of each `print` in source order, and
/// its rendering.
fn plan_nodes(src: &str, fusion: bool) -> (Registry, Vec<(PlanNode, String)>) {
    let src = format!("{XY}{src}");
    let prog = parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let reg = Registry::new();
    let mut g = Globals::new();
    let typed = check(&prog, &reg, &mut g).unwrap_or_else(|e| panic!("{src}: {e}"));
    let lowered = lower(&typed, &mut g, &reg, Mode::Run).unwrap_or_else(|e| panic!("{src}: {e}"));
    let planner = Planner::new(&reg, fusion);
    let plans = lowered
        .stmts
        .iter()
        .filter_map(|s| match s {
            LStmt::Print { term, .. } => Some(term),
            _ => None,
        })
        .map(|term| {
            let node = planner
                .plan(&reg, term)
                .unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_broadcasts_fit(&reg, &node);
            let rendered = render(&reg, &node, &lowered.leaves, &g);
            (node, rendered)
        })
        .collect();
    (reg, plans)
}

/// Plans like `plan_nodes`, returning the root kernel name (`-` for a bare leaf) and the
/// rendered plan of each `print`.
fn plan_prints(src: &str, fusion: bool) -> Vec<(String, String)> {
    let (reg, plans) = plan_nodes(src, fusion);
    plans
        .into_iter()
        .map(|(node, rendered)| {
            let root = match &node {
                PlanNode::Kernel(id, _) => reg.get(*id).name.clone(),
                _ => "-".to_string(),
            };
            (root, rendered)
        })
        .collect()
}

/// Asserts that a broadcast only feeds an `Either` input, which is where single-op array
/// kernels accept one, and that `Array` and `Scalar` inputs never receive a broadcast.
fn assert_broadcasts_fit(reg: &Registry, node: &PlanNode) {
    match node {
        PlanNode::Leaf(_) => {}
        PlanNode::Bcast(inner) => assert_broadcasts_fit(reg, inner),
        PlanNode::Cond(c, t, e) => [c, t, e]
            .into_iter()
            .for_each(|x| assert_broadcasts_fit(reg, x)),
        PlanNode::Kernel(id, args) => {
            let def = reg.get(*id);
            for (spec, arg) in def.inputs.iter().zip(args) {
                if matches!(arg, PlanNode::Bcast(_)) {
                    assert!(
                        matches!(spec, InputSpec::Either(_)),
                        "{} got a broadcast",
                        def.name
                    );
                }
                assert_broadcasts_fit(reg, arg);
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

/// Collects the name of every kernel in `node`.
fn kernel_names(reg: &Registry, node: &PlanNode, out: &mut BTreeSet<String>) {
    match node {
        PlanNode::Leaf(_) => {}
        PlanNode::Bcast(inner) => kernel_names(reg, inner, out),
        PlanNode::Cond(c, t, e) => [c, t, e]
            .into_iter()
            .for_each(|x| kernel_names(reg, x, out)),
        PlanNode::Kernel(id, args) => {
            out.insert(reg.get(*id).name.clone());
            args.iter().for_each(|a| kernel_names(reg, a, out));
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
    let (reg, off) = plan_nodes(&src, false);
    for (node, rendered) in &off {
        let mut names = BTreeSet::new();
        kernel_names(&reg, node, &mut names);
        for name in names {
            let def = reg.get(reg.by_name(&name).expect("kernel exists"));
            assert!(
                def.pattern.is_none(),
                "fusion off planned {name}: {rendered}"
            );
        }
    }
    let (on_reg, on) = plan_nodes(&src, true);
    let mut seen = BTreeSet::new();
    for (node, _) in &on {
        kernel_names(&on_reg, node, &mut seen);
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
