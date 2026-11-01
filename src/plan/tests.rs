use super::*;
use crate::check::{Globals, check};
use crate::kernels::{InputSpec, Registry};
use crate::lower::{LStmt, Mode, lower};
use crate::syntax::parse;

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

/// Plans every `print` of `XY` followed by `src` in `Mode::Run`. Returns the root kernel
/// name (`-` for a bare leaf) and the rendered plan of each, in source order.
fn plan_prints(src: &str) -> Vec<(String, String)> {
    let src = format!("{XY}{src}");
    let prog = parse(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let reg = Registry::new();
    let mut g = Globals::new();
    let typed = check(&prog, &reg, &mut g).unwrap_or_else(|e| panic!("{src}: {e}"));
    let lowered = lower(&typed, &mut g, &reg, Mode::Run).unwrap_or_else(|e| panic!("{src}: {e}"));
    let planner = Planner::new(&reg);
    lowered
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
            let root = match &node {
                PlanNode::Kernel(id, _) => reg.get(*id).name.clone(),
                _ => "-".to_string(),
            };
            (root, render(&reg, &node, &lowered.leaves, &g))
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
/// where the root is `-` for a bare leaf or conditional.
#[test]
fn datadriven_plans() {
    datadriven::walk(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/plan/testdata"),
        |f| {
            f.run(|tc| match tc.directive.as_str() {
                "plan" => plan_prints(&tc.input)
                    .into_iter()
                    .map(|(root, plan)| format!("{root}: {plan}\n"))
                    .collect::<String>(),
                d => panic!("unknown directive {d}"),
            })
        },
    );
}
