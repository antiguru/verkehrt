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
    let mut eg = egglog::EGraph::default();
    let outputs = eg.parse_and_run_program(None, program).expect("program runs");
    let best = outputs
        .iter()
        .find_map(|o| match o {
            egglog::CommandOutput::ExtractBest(dag, cost, term) => {
                Some((dag.to_string(*term), *cost))
            }
            _ => None,
        })
        .expect("extract produces a term");
    // The i64 literal inside `A` costs 1 on top of the declared constructor costs.
    assert_eq!(best, ("(Fast (A 7))".to_string(), 4));
}
