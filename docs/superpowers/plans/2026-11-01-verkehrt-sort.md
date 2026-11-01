# Plan 5: sort and high-cardinality grouping implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `argsort` fast without changing its output, then add ClickBench q04, q08, q09, q15, q32, and q35 and measure them with the sort-bound queries from #20.

**Architecture:** The body of `argsort` (`ArgSort<T>` in `src/kernels/scalar/index.rs`) gets an algorithm chosen by measurement. The plan 4 body stays as a test reference. Its whole-kernel auto entries compile the new body per level, as today. The new queries are `.vk` programs that the acceptance suite discovers.

**Tech Stack:** Rust, criterion (`benches/tier1.rs`, group `tier2`, bench `argsort-i64`), proptest (`tests/isa_variants.rs`), the acceptance harness.

**Spec:** `docs/superpowers/specs/2026-11-01-verkehrt-sort-design.md`

## Global Constraints

* `argsort` output must equal the plan 4 output on every input: stable ascending positions as `i64`, with signed keys in signed order.
* Allocation failure is an error (`Array::try_build`, `try_zeroed`, `try_reserve_exact`), including scratch buffers.
* No hash grouping, `find`, top-k primitive, `f64` argsort, or changes to `histogram`/`reduce_into`.
* Every algorithm choice is backed by a measurement that is committed as raw output.
* No unsafe beyond what exists, and `SAFETY` comments on any that is added. Use `#[expect]`, not `#[allow]`. No `#[path]`. Comments have no em-dashes or structuring semicolons.
* Host: Zen 5, boost off, `powersave`, 3.0 GHz under load per the user. Differences within about 10% between builds are noise.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit.

## Review Focus

* Ties and duplicates. Inputs with many equal keys, all-equal arrays, and alternating patterns must keep position order within equal keys.
* Signed extremes. `i64::MIN`, `i64::MAX`, `-1`, and `0` in one array (and the same for `i32`) must sort in signed order, which is where a radix sign-bit flip goes wrong.
* Sizes around algorithm switch points. Inputs of length 0, 1, 2, and the sizes where a small-input path hands over must give the reference output.
* Allocation failure in scratch buffers, which must surface as an error.

---

### Task 1: Reference oracle and sort tests

**Files:**
* Modify: `src/kernels/scalar/index.rs` (keep the plan 4 body as `fn argsort_reference<T: Elem + Ord>(xs: &[T]) -> Vec<i64>`, `#[cfg(test)]` or `pub(crate)` as needed by the tests)
* Test: `src/kernels/tests.rs` (or a `tests.rs` submodule next to `index.rs` if it grows large)

- [ ] **Step 1:** Add a property test, `argsort_matches_reference`, for each of `i64`, `u64`, `i32`, and `u8`. It generates arrays of length 0 to 5000 from strategies that cover the Review Focus classes (few distinct values, signed extremes, sorted, reverse sorted, random) and asserts that the `argsort_<t>` kernel output equals `argsort_reference`.
- [ ] **Step 2:** Add fixed tests for the signed extremes and for an all-equal array of 100,000 rows, which must return `0..n`.
- [ ] **Step 3:** Run them against the current body (expect PASS, since the body is the reference). Commit "Pin argsort to its plan 4 output".

### Task 2: Fast argsort

**Files:**
* Modify: `src/kernels/scalar/index.rs` (`ArgSort<T>::run`), plus a new helper module next to it if the algorithm needs more than a screenful
* Create: `docs/measurements/2026-11-01-argsort.md` and `docs/measurements/2026-11-01-argsort/` (raw output)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (the `argsort` catalog row and the strain-case list, only as the measurement supports)

- [ ] **Step 1:** Measure the plan 4 baseline. Run `cargo bench --bench tier1 -- tier2/argsort` and record it. Add bench rows for `argsort` on `u64`, `i32`, and `u8` if they are missing, with the same input shape as `argsort-i64`.
- [ ] **Step 2:** Prototype at least two candidates from the spec, for example an LSD radix on (key, position) and a sort of keys gathered into a contiguous buffer. Measure each at 10M rows per type and at 1M rows (the ClickBench size). Keep each prototype's raw output.
- [ ] **Step 3:** Implement the winner in `ArgSort<T>::run`. A radix sort flips the sign bit of signed keys before the passes. The Task 1 tests must pass unchanged, along with `tests/isa_variants.rs`, which compares every variant with scalar.
- [ ] **Step 4:** Check the auto entries. The whole-kernel auto entries compile the body under each level's features. Measure scalar, `x86-64-v3-auto`, and `x86-64-v4-auto`, and record whether vectorization helps. Add intrinsics only if a measurement shows a gain beyond noise.
- [ ] **Step 5:** Write `docs/measurements/2026-11-01-argsort.md`: candidates, numbers, the choice and why, per-variant times, and the ratio against the plan 4 baseline. Update the base spec's `argsort` row and strain-case entry only as supported. Commit "Replace argsort with <algorithm>".

### Task 3: Many-group ClickBench queries

**Files:**
* Create: `tests/acceptance/clickbench/q04.vk`, `q08.vk`, `q09.vk`, `q15.vk`, `q32.vk`, `q35.vk`

This task runs on a new stack layer, `acceptance/clickbench-groups`, which the controller creates before dispatching it.

- [ ] **Step 1:** Write each query from the SQL in its corgi header (`~/dev/repos/data/verkehrt/corgi-wip/corgi/algorithms/clickbench/qNN.col`), with the grouping idiom of `q30.vk`. COUNT(DISTINCT) is the number of run starts after sorting by (group, value).
- [ ] **Step 2:** Add a synthetic non-empty and tie-heavy test in `tests/acceptance/main.rs` for any query whose real answer cannot distinguish a wrong tie order, following `q41_pages_past_ten_thousand_groups`.
- [ ] **Step 3:** Run the acceptance suite with `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH` set, then commit "Add ClickBench q04, q08, q09, q15, q32, and q35".

### Task 4: Measurement

**Files:**
* Create: `docs/measurements/2026-11-01-clickbench-groups.md` and its raw-output directory

- [ ] **Step 1:** Time q04, q08, q09, q15, q32, q35, q30, q31, q40, and q41 with the acceptance bench, corgi's ClickBench bench, and DuckDB's one-thread times, using the method of `docs/measurements/2026-10-31-clickbench-now.md`. Collect per-kernel stats with `examples/acceptance_stats.rs`.
- [ ] **Step 2:** Compare q30, q31, q40, and q41 with their #20 times, and name the dominating kernel of each query.
- [ ] **Step 3:** Write the document. Record gaps for queries that sorting cannot bring within reach of DuckDB (candidates for hash grouping or a top-k primitive). Commit "Measure the many-group ClickBench queries".
