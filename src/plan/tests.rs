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
    let outputs = eg.parse_and_run_program(None, program).expect("program runs");
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
            let node = planner.plan(&reg, term).unwrap_or_else(|e| panic!("{src}: {e}"));
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
        PlanNode::Kernel(id, args) => {
            let def = reg.get(*id);
            for (spec, arg) in def.inputs.iter().zip(args) {
                if matches!(arg, PlanNode::Bcast(_)) {
                    assert!(matches!(spec, InputSpec::Either(_)), "{} got a broadcast", def.name);
                }
                assert_broadcasts_fit(reg, arg);
            }
        }
    }
}

/// The plan of the single `print` in `src`.
fn plan_print(src: &str) -> (String, String) {
    let mut plans = plan_prints(src);
    assert_eq!(plans.len(), 1, "{src}");
    plans.pop().unwrap()
}

#[test]
fn sum_of_product_is_dot() {
    assert_eq!(plan_print("print(sum(x * y));").1, "(dot_f64 x y)");
}

#[test]
fn sum_of_swapped_product_is_dot() {
    let (kernel, rendered) = plan_print("print(sum(y * x));");
    assert_eq!(kernel, "dot_f64");
    assert!(rendered == "(dot_f64 y x)" || rendered == "(dot_f64 x y)", "{rendered}");
}

#[test]
fn sum_of_square_is_sumsq() {
    assert_eq!(plan_print("print(sum(x * x));").1, "(sumsq_f64 x)");
}

#[test]
fn count_of_comparison_is_fused() {
    let plan = plan_print("print(count(x > 0.5));");
    assert_eq!(plan, ("count_gt_f64".into(), "(count_gt_f64 x 0.5)".into()));
}

#[test]
fn flipped_comparison_is_fused() {
    let plan = plan_print("print(count(0.5 < x));");
    assert_eq!(plan, ("count_gt_f64".into(), "(count_gt_f64 x 0.5)".into()));
}

#[test]
fn flipped_non_strict_comparison_is_fused() {
    let plan = plan_print("print(count(0.5 <= x));");
    assert_eq!(plan, ("count_ge_f64".into(), "(count_ge_f64 x 0.5)".into()));
}

#[test]
fn filter_with_comparison_is_fused() {
    assert_eq!(
        plan_print("print(filter(x, |v| v > 0.5));"),
        ("filter_gt_f64".into(), "(filter_gt_f64 x 0.5)".into())
    );
}

#[test]
fn conditional_sum_is_fused() {
    assert_eq!(
        plan_print("print(sum(map(x, |v| if v > 0.5 { v } else { 0.0 })));"),
        ("condsum_gt_f64".into(), "(condsum_gt_f64 x 0.5 0.0)".into())
    );
}

#[test]
fn single_use_let_fuses_across_binding() {
    assert_eq!(plan_print("let t = x * y; print(sum(t));").1, "(dot_f64 x y)");
}

#[test]
fn shared_let_is_a_leaf() {
    let plans = plan_prints("let t = x * y; print(sum(t)); print(t);");
    assert_eq!(plans[0], ("reduce_add_f64".into(), "(reduce_add_f64 t)".into()));
    assert_eq!(plans[1], ("-".into(), "t".into()));
}

#[test]
fn unfused_sum_uses_single_op_kernels() {
    assert_eq!(plan_print("print(sum(x + y));").1, "(reduce_add_f64 (add_f64 x y))");
}

#[test]
fn single_op_kernel_keeps_broadcast_operand() {
    assert_eq!(plan_print("print(x * 2.0);").1, "(mul_f64 x 2.0)");
}

#[test]
fn constant_array_renders_by_type_and_length() {
    // Also pins the `Arr` guard: `dot_f64` needs two arrays, so it must not cover a
    // broadcast operand.
    let rendered = plan_print("print(sum([1.0, 2.0] * 2.0));").1;
    assert_eq!(rendered, "(reduce_add_f64 (mul_f64 <f64[2]> 2.0))");
}
