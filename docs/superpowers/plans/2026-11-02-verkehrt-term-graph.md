# Term graph and shared evaluation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make compilation linear in repeated reads by lowering and planning into hash-consed graphs (#39), and compute each repeated pure plan subtree of a program once per execution, so q29's ninety identical `reduce_add_i64` passes become one.

**Architecture:** Lowering (`src/lower.rs`) builds a program-wide `TermGraph` interned on kind and type, and inlining and hoisting run on it with the rules they have today. The planner (`src/plan.rs`, `src/plan/egglog.rs`) emits one egglog fact per distinct node, plans each distinct root or conditional once per program, and builds a `PlanGraph` interned across statements by walking egglog's `TermDag`. The executor (`src/exec.rs`) evaluates `PlanId`s and memoizes the successful values of shared nodes, kernel nodes with two or more uses and no load below them, for one run.

**Tech Stack:** Rust, egglog 3, the datadriven tests, the `tests/generated/` oracles, criterion benches, `examples/acceptance_stats.rs`.

**Spec:** `docs/superpowers/specs/2026-11-02-verkehrt-term-graph-design.md`

## Global Constraints

* Values, errors (kind, message, span), output order, demand, and lazy branches are exactly those of today, on every ISA, backend, plan, and tiling.
* Tasks 1 to 3 leave `src/*/testdata/*.dd`, `tests/programs/*.dd`, `tests/cli.rs`, and the `acceptance_plans` output byte for byte unchanged. Any diff stops the task and is reported with the mechanism, not accepted.
* Task 4 may change only kernel-call counts in statistics and trace expectations.
* Hoisting stays per statement and never moves a node out of a lazy branch. Temporary numbering and bind order stay as today.
* A failed evaluation is never memoized. Subgraphs with a `load_*` kernel are never shared. The memo lives for one `run_plan` call.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit, with `CARGO_INCREMENTAL=0`. The debug CI run of the generated test stays under about 20 s (13.7 s today).
* No `#[path]`, no `mod.rs` (a module is `name.rs` plus `name/`), `#[expect]` not `#[allow]`, no `unsafe` without a SAFETY note. Comments have no em-dashes or structuring semicolons, and existing comments move with their code. Docs name no commit hashes.

## Review Focus

* Hoisting counts on the graph: a `Bcast` passes its count to its operand, a `Cond` contributes only its condition edge, and `add(a, a)` counts `a` twice. Compare with `count_ops` on the existing `lets.dd` and `conditionals.dd` cases.
* Single-use inlining counts paths, not distinct parents: `fn f(q) { q + q }` applied to a let must keep the let.
* The planning memo returns the sum rewrites recorded for the memoized id, so `prepare_with_sum_rewrites` reports what it reports today.
* The memo stores only `Ok` values. Two statements that share a failing subtree must each report their own span.
* A shared node is never a region kernel. A shared elementwise kernel feeding a region runs once, and the region still tiles.
* A `Leaf::Var` read during a run never changes: check that no path binds a variable twice in one `run_plan`.

---

### Task 1: Split `src/lower.rs` by responsibility

Pure move, no behavior change. The term graph adds to `src/lower.rs`, which has 732 lines today.

**Files:**
* Modify: `src/lower.rs`
* Create: `src/lower/leaves.rs` (`Leaves`, `same_const`, `array_of`), `src/lower/passes.rs` (`leaf_vars`, `stmt_term_mut`, `substitute`, `inline_single_use`, `count_ops`, `Hoister`, `hoist_repeated`)

**Interfaces:**
* Produces: the same public surface of `crate::lower` (`Mode`, `LeafId`, `Leaf`, `TermKind`, `Term`, `LStmt`, `Lowered`, `op_name`, `no_kernel`, `lower`). Moved items become `pub(super)` or `pub(crate)` as needed and nothing else changes.

- [ ] **Step 1:** Move the items listed above verbatim with their doc comments and inline comments. Declare the modules in `src/lower.rs` next to `pub(crate) mod higher;`.
- [ ] **Step 2:** Run `cargo test --release` and clippy. `git diff --stat` must show only moves (line counts balance). Commit "Split lowering into leaves and passes".

### Task 2: Lower into a term graph and emit it once per node

**Files:**
* Modify: `src/lower.rs`, `src/lower/leaves.rs`, `src/lower/passes.rs`, `src/lower/higher.rs` (`Build`), `src/check/lifting.rs` (`Build` impl signatures), `src/plan.rs`, `src/plan/egglog.rs`, `src/interp.rs` (pass the graph and statement span to the planner)
* Create: `src/lower/graph.rs` (`TermGraph`, `TermId`)
* Modify: `src/lower/tests.rs`, `src/plan/tests.rs` (helpers read the graph, assertions unchanged)

**Interfaces:**
* Produces: `pub struct TermId(pub u32)`, `pub struct TermGraph` with `kind(&self, TermId) -> &TermKind`, `ty(&self, TermId) -> ValueType`, `len(&self) -> usize`, and a crate-internal `intern(&mut self, TermKind, ValueType) -> TermId`. `TermKind` children are `TermId`s. `Lowered { leaves, graph: TermGraph, stmts, exports }`, and `LStmt` variants hold `term: TermId`. The `Term` struct is removed.
* Produces: `Planner::plan(&self, reg, graph: &TermGraph, root: TermId, span: Span) -> Result<PlanNode, Error>` and the matching `plan_with_sum_rewrites`. Planner errors carry `span`. `PlanNode` is still a tree in this task.
* Produces: `egglog::emit(reg, graph, root) -> Result<(String, String, Vec<TermId>), String>` with one `let` per distinct reachable node and distinct holes.

- [ ] **Step 1:** Capture the baseline: run `cargo test --release --test acceptance acceptance_plans -- --ignored --nocapture` before changing code (it needs no data), save its output as `plans-before.txt` outside the repository, and keep it for Tasks 3 and 5.
- [ ] **Step 2:** Write failing tests in `src/lower/tests.rs`: the block-let chain at depth 64 and the function form `f(f(...f(a)))` at depth 32 (parameter read four times, as in the generated-program record) lower with a graph of at most `c * depth` nodes for a small constant `c` the test states. Add a test in `src/plan/tests.rs` that the lazy-branch form of the #39 comment (`let b7 = if false { a } else { a }; let b11 = compress(b7, (b7 >= reduce(b7, and)));` under 16 nested lazy ifs) emits a number of `(let ` commands and holes linear in the nesting, calling `egglog::emit` directly.
- [ ] **Step 3:** Add `TermGraph` and switch `Lowerer` to ids: `env` and `lifted` hold ids, `leaf`, `op`, and `op_exact` intern. Change `Build::ty` to take `&self` and `apply`/`apply_exact` to take `&mut self`, and adapt `Lifting`. Index `Leaves` with hash maps keyed as `same_const` compares, assigning ids in order of first request.
- [ ] **Step 4:** Port the passes: `inline_single_use` counts uses as path counts saturated at 2 (statement roots count one), `substitute` rebuilds with a per-statement memo from old to new id, and hoisting counts edges from distinct reachable `Op` and `Cond` nodes, with `Bcast` passing its count to its operand and `Cond` contributing only its condition edge. Keep the visiting order of `Hoister::go`.
- [ ] **Step 5:** Port `egglog::emit` with a memo from `TermId` to emitted name and distinct holes, and `plan_into` with a per-statement memo from hole `TermId` to its planned `PlanNode`. Pass the statement span into `plan`.
- [ ] **Step 6:** Run the whole suite. Every expectation must pass unchanged. Commit "Lower into a hash-consed term graph".

### Task 3: Plan into a graph and lift the generator's sharing cap

**Files:**
* Modify: `src/plan.rs` (`PlanGraph`, `PlanId`, planning context, `render`), `src/plan/egglog.rs` (`extract` returns the `TermDag` walk), `src/exec.rs` (`Plan`, `Machine` and regions on `PlanId`), `src/interp.rs` (one planning context per compiled program)
* Modify: `src/plan/tests.rs`, `src/exec/tests.rs` (`eval_node` and the hand-built plan in `kernel_failing_in_a_later_tile_raises_the_untiled_error` build a `PlanGraph`)
* Modify: `tests/generated/gen.rs`, `tests/generated/main.rs`, `docs/measurements/2026-11-01-generated-programs.md`

**Interfaces:**
* Consumes: `TermGraph`, `TermId`, `Planner::plan` from Task 2.
* Produces: `pub struct PlanId(pub u32)`, `pub struct PlanGraph` with `node(&self, PlanId) -> &PlanNode` and `len`, interning across a program. `PlanNode` variants take `PlanId` children. `exec::Plan { lowered, graph: PlanGraph, roots: Vec<PlanId> }`. A planning context (name left to the builder) holding the `PlanGraph` and a program-wide memo from `TermId` to `(PlanId, BTreeSet<SumRewrite>)`. `render(reg, graph, id, leaves, globals)` with unchanged text.

- [ ] **Step 1:** Write failing tests: the function form at depth 32 and a function `fn f(q: f64[]) -> f64[] { if s > 0.0 { q + q } else { q } }` nested 24 deep compile end to end (`Interpreter::prepare`) with a plan graph linear in the depth, and two identical statements share one root `PlanId`.
- [ ] **Step 2:** Replace the text round trip in `egglog::extract` with a walk of the `TermDag` from `ExtractBest`, memoized per DAG node, mapping `Leaf` and `Hole` literals, `Bcast`, and `K_` constructors into the plan graph. Keep `sexpr` for fused patterns.
- [ ] **Step 3:** Build one planning context per program in `Interpreter::compile`, memoize `plan_into` results program-wide, and return the memoized sum rewrites on a hit.
- [ ] **Step 4:** Move the executor and tiled regions to `PlanId`: `eval`, `region_kernel`, `collect`, `eval_region`, `eval_region_whole`, and `Region::inputs`.
- [ ] **Step 5:** Run `acceptance_plans` again and diff against `plans-before.txt` from Task 2, which must be empty. Run the whole suite unchanged.
- [ ] **Step 6:** Remove `MAX_SHARE_DEPTH`, the replacement by a leaf, and the "statement over the sharing depth cap" feature. Keep `share_depth` and record the feature "sharing depth over 3". Add the gate `MIN_DEEP_SHARE` at a third of the count the fixed CI seed gives. Time the debug CI run, which must stay under about 20 s, and if it does not, stop and report BLOCKED with the time. Update the generated-program record: the cap is gone, the gate, the new CI time. Commit "Plan into a hash-consed graph".

### Task 4: Evaluate shared plan nodes once

**Files:**
* Modify: `src/exec.rs` (shared flags, memo, hit counter), `src/plan.rs` (shared flag computation), `src/interp.rs` (`Options::no_sharing`, `Interpreter::shared_hits`), `src/main.rs` (`--no-sharing`, `--trace` help), `src/columns.rs` (`sharing_from_env`), `benches/acceptance.rs` and `examples/acceptance_stats.rs` (`VERKEHRT_SHARING`)
* Modify: `src/exec/tests.rs`, `tests/programs.rs` (`stats_total_per_kernel` with `no_sharing`, plus a sibling), `tests/cli.rs` if a flag test fits
* Modify: `tests/generated/oracles.rs`, `tests/generated/main.rs`, `docs/measurements/2026-11-01-generated-programs.md`

**Interfaces:**
* Consumes: `PlanGraph`, `Plan { graph, roots }` from Task 3.
* Produces: `Plan::shared` (flags per `PlanId`): a `Kernel` node with no `load_*` kernel in its subgraph and at least two uses, counting parent edges and statement roots. `Options::no_sharing: bool` (default `false`), `Interpreter::shared_hits(&self) -> u64` counted across `run`, `eval_repl`, and `Prepared::execute`. `pub fn sharing_from_env() -> bool` in `columns`, reading `VERKEHRT_SHARING` (`off` disables, unset enables, anything else panics as `tiling_from_env` does).

- [ ] **Step 1:** Write failing tests in `src/exec/tests.rs`: five q29-style sums over one column plan to one shared `reduce_add` that stats count once, with the expected hit count. Two statements sharing a subtree that fails each report their own span, and match the `no_sharing` run by kind, message, and span. A shared node only in an untaken branch never runs (no stats entry). A shared elementwise kernel feeding two regions runs once and `tiled_regions` counts both regions. Two `eval_repl` inputs with the same expression each run it. A shared subgraph with a `load_*` kernel runs at each use.
- [ ] **Step 2:** Compute the shared flags after planning and store them in `Plan`. In `Machine::eval`, return a memoized value for a shared node, otherwise evaluate it and store only `Ok`. Count hits. A hit writes no trace line and adds no statistics. Allocate the memo per `run_plan` call.
- [ ] **Step 3:** Add `Options::no_sharing`, `--no-sharing`, `sharing_from_env`, and its use in the bench and the stats example. Update the `--trace` help text. Switch `stats_total_per_kernel` to `no_sharing` and add a sibling asserting `calls=1` with sharing.
- [ ] **Step 4:** Add oracle 5 in `tests/generated/oracles.rs`: on the scalar interpreter, each plan with sharing on and with `no_sharing` gives equal printed output and exports, and equal errors by kind, message, and span. Add `MIN_SHARED_HITS` at a third of the hits the fixed CI seed gives. Update the oracle list in the module doc and the generated-program record. Time the debug CI run (under about 20 s).
- [ ] **Step 5:** Run the whole suite, a release generated search of 5,000 cases, clippy, and fmt. Commit "Evaluate shared plan nodes once per run".

### Task 5: Measure the term graph and shared evaluation

**Files:**
* Create: `docs/measurements/2026-11-02-term-graph.md` and `docs/measurements/2026-11-02-term-graph/` (evidence files)
* Modify: `docs/superpowers/specs/2026-11-02-verkehrt-term-graph-design.md` (an outcome paragraph under "Evidence", as the tiled-regions spec has)

- [ ] **Step 1:** Commit `plans-before.txt` from Task 2 and a fresh `acceptance_plans` output as evidence, and state that they are identical.
- [ ] **Step 2:** Bench q29 with `VERKEHRT_SHARING=off` and with sharing, five variants, two runs each, recording the 1-minute load at the start and end of each file and requiring it at or below 4 before each run. Record per-kernel stats of q29 both ways with `acceptance_stats` (best of five fresh runs).
- [ ] **Step 3:** Run the full acceptance benches (both suites, five variants) with sharing on and off. Name every query and variant that moved beyond the 10% noise band, and repeat any that did.
- [ ] **Step 4:** Time the function form of the generated-program record at depths 5 and 7 in the release CLI, and record time and peak RSS against the 0.52 s and 4.84 s it records.
- [ ] **Step 5:** Write the record following `docs/measurements/2026-11-02-tiled-regions.md`: setup, evidence files, the before and after table, the q29 plan and kernel shares, the DuckDB (7.24 ms) and corgi (33.14 ms) ratios reused from `docs/measurements/2026-10-31-clickbench-now.md`, and the compile-time table. Every number comes from an evidence file. Add the outcome paragraph to the spec. Commit "Measure the term graph and shared evaluation".
