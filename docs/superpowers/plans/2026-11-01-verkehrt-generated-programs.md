# Generated programs implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate random well-typed programs and check each with three oracles: cross-variant, fused against unfused, and a reference evaluator.

**Architecture:** An integration test `tests/generated/main.rs` with modules for the generator (`gen.rs`), the oracles (`oracles.rs`), and the reference evaluator (`eval.rs`). The library gains one option, which plans without fused patterns. Found bugs get fixes and datadriven regression cases in `tests/programs/generated.dd`.

**Tech Stack:** Rust, proptest, the public `syntax`, `check`, and `Interpreter` APIs, and the datadriven harness from #19.

**Spec:** `docs/superpowers/specs/2026-11-01-verkehrt-generated-programs-design.md`

## Global Constraints

* Every generated program type-checks. A generator bug that produces a type error is a test failure, not a skipped case.
* Floats compare by bits. Agreement is exact.
* The CI run uses a fixed seed and stays well under 30 s in debug. A local long run is selected by environment variables (for example `VERKEHRT_GEN_CASES` and `VERKEHRT_GEN_SEED`).
* Each bug found is fixed in its own commit with a regression case in `tests/programs/generated.dd`, or recorded as an issue and excluded from generation with a pointer.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. No `#[path]`. Comments have no em-dashes or structuring semicolons.

## Review Focus

* A generator that quietly produces only trivial programs. Measure and report the distribution of features per run.
* Oracles that agree because they share the code under test. For example, an evaluator that calls library kernels, or an "unfused" plan that still fuses.
* Errors compared too loosely (any error equals any error) or too strictly (message text that legitimately differs by plan).
* Nondeterminism: the program generator must be reproducible from its seed, and `rand_*` programs must use explicit seeds.

---

### Task 1: Planner option without fused patterns

**Files:**
* Modify: `src/plan.rs` (`Planner` built with or without fused rewrites), `src/plan/egglog.rs` (prelude without the fused rewrites), `src/interp.rs` (`Options` field, default fused)
* Test: `src/plan/testdata/` (a `plan` directive argument, for example `plan fusion=off`, showing `sum(x * y)` planned as `reduce_add_f64 (mul_f64 x y)`)

- [ ] **Step 1:** Add the option. With fusion off, the prelude omits every fused-pattern rewrite and keeps the exact rewrites. Add datadriven cases for `dot`, `sumsq`, a fused comparison count, and a fused filter, each with fusion on and off.
- [ ] **Step 2:** Run all tests, then commit "Add a planner option without fused patterns".

### Task 2: Generator with oracles 1 and 2

**Files:**
* Create: `tests/generated/main.rs`, `tests/generated/gen.rs`, `tests/generated/oracles.rs`

- [ ] **Step 1:** Write the strategy. It produces a program's source text plus input values, built from typed expressions so that the program checks. Cover the features listed in the spec's Generator goal. Print a feature histogram from the test, which is visible with `--nocapture`.
- [ ] **Step 2:** Oracle 1 runs the program on every available variant and compares with scalar. Oracle 2 runs it with fusion on and off. Both compare output text and error kind.
- [ ] **Step 3:** Run a long local search, for example 20,000 cases. For each failure, shrink it, fix the bug in its own commit with a `generated.dd` case, or file an issue and exclude it. Commit the generator as "Generate programs and compare variants and fusion".

### Task 3: Reference evaluator (oracle 3)

**Files:**
* Create: `tests/generated/eval.rs`
* Modify: `tests/generated/oracles.rs`

- [ ] **Step 1:** Evaluate `check::TProgram` directly, one scalar at a time: elementwise operators with broadcasting, lambdas applied per element, function calls, blocks, conditionals, reductions in the canonical float order of the base spec (`docs/superpowers/specs/2026-10-30-verkehrt-design.md`, the reduction order section), and the tier 2 primitives.
  * List each unsupported primitive in the module doc, and exclude it from generation for this oracle.
- [ ] **Step 2:** Add oracle 3 to the test, then run a long local search and handle failures as in Task 2.
- [ ] **Step 3:** Commit "Check generated programs against a reference evaluator".

### Task 4: Report

**Files:**
* Create: `docs/measurements/2026-11-01-generated-programs.md` (a short findings record, not a benchmark)

- [ ] **Step 1:** Record the runs: cases, seeds, the feature histogram, each bug found with its fix commit or issue, and the primitives excluded from oracle 3. Commit "Record the generated-program findings".
