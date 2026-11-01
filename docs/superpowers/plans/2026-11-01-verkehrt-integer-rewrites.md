# Exact integer sum rewrites implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plan `sum` over an integer array with a broadcast scalar operand as one `reduce_add` plus scalar arithmetic, and re-measure ClickBench q29.

**Architecture:** The egglog prelude in `src/plan/egglog.rs` gains rewrites from `reduce_add_T (op_T x (Bcast k))` to scalar kernels over `reduce_add_T x` and `length_T x`, generated from the registry for every integer type `T` whose `sum` returns `T` (today `i64` and `u64`) and whose kernels exist. `i32` and `u8` are excluded because they sum into a wider type after wrapping each element. They sit behind the same flag as the fused patterns, so the unfused planner and oracle 2 of `tests/generated/` compare plans with and without them. The generator gains the source shapes and a gate that each rewrite fires in the CI run.

**Tech Stack:** Rust, egglog, the datadriven planner tests, the `tests/generated/` oracles, criterion.

**Spec:** `docs/superpowers/specs/2026-11-01-verkehrt-integer-rewrites-design.md`

## Global Constraints

* Rewrites must preserve every result bit. Integers only, never floats.
* Rewrites exist only for an integer type `T` whose `sum` returns `T`, today `i64` and `u64`. For `u64` the count is `cast(length(x))` to `u64`, which wraps.
* A rewrite is emitted only when every kernel on its right-hand side exists in the registry for that type.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays well under 30 s.
* No `#[path]`, no `mod.rs`, `#[expect]` not `#[allow]`. Comments have no em-dashes or structuring semicolons.
* Docs name no commit hashes (squash merges).

## Review Focus

* A right-hand side with the wrong result type or a missing wrap. The count is cast to the element type first, and the product and sum are computed in that type. A test pins an overflowing case for each width.
* `k - x` ordering: `sum(k - x)` is `n * k - sum(x)`, not `sum(x) - n * k`.
* The `Arr` invariant in `emit`: new e-classes are scalars, and no fused rule may match them as arrays.
* Oracle 2 must actually see the difference. The unfused planner must drop these rewrites, and a test shows one program planned both ways.
* Generator gating that is vacuous, for example a gate that counts source shapes instead of plans in which the rewrite fired.

---

### Task 1: Rewrites, planner tests, and generator coverage

**Files:**
* Modify: `src/plan/egglog.rs` (prelude rewrites, the `Arr` invariant comment), `src/interp.rs` (`no_fusion` doc covers the rewrites)
* Modify: `src/plan/testdata/` (new `sum_rewrites.dd`)
* Modify: `tests/generated/gen.rs` and `gen/` (shapes), `tests/generated/main.rs` (gate)

- [ ] **Step 1:** Add datadriven `plan` cases in `src/plan/testdata/sum_rewrites.dd` for `sum(x + k)`, `sum(k + x)`, `sum(x - k)`, `sum(k - x)`, `sum(x * k)`, and `sum(k * x)` on `i64`, `u64`, `i32`, and `u8`, each with `fusion=off` beside it, plus one `f64` case where no rewrite fires. Add `run` cases in `tests/programs/` with values that overflow each width, so the rewritten plan's output is pinned. Run them and watch the plan cases fail.
- [ ] **Step 2:** Generate the rewrites in `prelude` when `fusion` is set. For `reduce_add_T` over `add_T`, `sub_T` (both operand orders), and `mul_T` with one `Bcast` operand, emit the right-hand side in scalar kernels (`add_T_s`, `sub_T_s`, `mul_T_s`, `length_T`, and the `i64` to `T` scalar cast for non-`i64` types), skipping a type when any of them is missing. Update the comment above `emit` that states the single-constructor invariant. Run the tests.
- [ ] **Step 3:** Make the generator produce `sum` over `x ± k`, `k - x`, and `x * k` with broadcast scalars, including overflowing values. Add a gate to the CI run that fails when any of the rewrites fired in no plan, counting fired rewrites from the plans the oracles build. Run a 5,000-case release search.
- [ ] **Step 4:** Commit "Move broadcast scalars out of integer sums".

### Task 2: Measure q29

**Files:**
* Create: `docs/measurements/2026-11-01-integer-rewrites.md` and its evidence directory

- [ ] **Step 1:** Re-run q29 with criterion as in `docs/measurements/2026-10-31-clickbench-now.md` (same command, five variants, load recorded), the per-kernel stats, and the plan. Check that `clickbench_matches_duckdb` passes with the prepared data.
- [ ] **Step 2:** Write the record: the before and after table with the DuckDB and corgi ratios, the plan, the kernel share, and what remains for #39 (ninety `reduce_add` passes over one column). Describe the build, do not name commits. Commit "Measure q29 with the integer sum rewrites".
