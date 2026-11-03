# Composite-key argsort and top-k implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `argsort(a, b)`, one stable LSD radix sort over two key columns, and fused top-k kernels for `slice` over `argsort`, then rewrite q32, q31, q30, q15 and q35 where the measurement keeps the new form (#31, the rest of #9).

**Architecture:** The `argsort` kernels move into their own module `src/kernels/scalar/sort.rs`. `argsort2_<a>_<b>` is a whole kernel with op `ArgSort` that range-reduces both key images, packs them into one composite image, and sorts 16-byte records when the image fits 96 bits and positions fit `u32`, 24-byte records otherwise. `Options::no_composite_sort` makes lowering expand `argsort(a, b)` into today's two-sort idiom with a `same_len` check, which is the reference for a new generated-program oracle. `topk_<t>` and `topk2_<a>_<b>` are fused whole kernels with the patterns `slice(argsort(...), s, e)`, so `--no-fusion` and oracle 2 already cover them.

**Tech Stack:** Rust, proptest, the `tests/generated/` oracles, `tests/fused_oracle.rs`, criterion (`benches/tier1.rs`, `benches/acceptance.rs`), `examples/acceptance_stats.rs`.

**Spec:** `docs/superpowers/specs/2026-11-03-verkehrt-composite-keys-design.md`

## Global Constraints

* `argsort(a, b)` returns the positions in ascending order of (`a`, `b`, position) for `i64`, `u64`, `i32` and `u8` columns in any combination. Tie order is part of the output and must be identical on every ISA, backend, layout, and length. A length mismatch fails with `length mismatch: {len(a)} vs {len(b)}`, an empty pair gives an empty array, and a scratch reservation failure is an error.
* The one-argument `argsort` keeps its output, its code path, and its catalog row. The pure-move task changes no behavior.
* A top-k kernel returns exactly `slice(argsort(...), s, e)` and fails with the unfused plan's error: the argsort's argument errors first, then `slice {s}..{e} out of range for length {n}`.
* `no_composite_sort` changes no value. Its expansion is `gather(o1, argsort(gather(same_len(a, length(b)), o1)))` with `o1 = argsort(b)` as one shared term.
* `CARGO_INCREMENTAL=0`. `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays under about 20 s.
* No `#[path]`, no `mod.rs` (a module is `name.rs` plus `name/`), `#[expect]` not `#[allow]`, no `unsafe`. Comments have no em-dashes or structuring semicolons, and existing comments move with their code. Docs name no commit hashes, and every performance claim cites a committed evidence file.

## Review Focus

* Stability and tie order: few distinct values, equal major keys with differing minor keys and the reverse, all rows equal, both signs of signed keys, type extremes, and lengths at 1024 and 1025 (the comparison and radix paths).
* Layout choice: composite widths of 0, 64, 96 and 97 bits, a constant column on either side, and positions that force the 24-byte layout must all give the reference permutation. The range reduction must never wrap (the maximum minus the minimum image of a `u64` column can be `u64::MAX`).
* Off switch: the expansion shares `o1`, raises the kernel's mismatch message, and leaves no `argsort2` in any plan with `no_composite_sort`.
* Top-k: `e` at 0, `TOPK_MAX`, `TOPK_MAX + 1` and `n`, `s = e`, `s > e`, `e > n`, descending input (worst case), and errors ordered as in the unfused plan.
* Gates: the existing minimums of `tests/generated/main.rs` still hold after the generator change, and the new minimums are a third of what the CI seed gives.

---

### Task 1: Move `argsort` into its own kernel module

Pure move, no behavior change. The composite and top-k kernels join the sort code, which today sits in `src/kernels/scalar/index.rs` among `gather`, `slice`, `histogram`, and `reduce_into`, with its tests in `src/kernels/scalar/index/tests.rs` (all argsort tests) and the radix code in `src/kernels/scalar/index/radix.rs`.

**Files:**
* Modify: `src/kernels/scalar/index.rs` (remove `ArgSort`, `argsort_reference`, the `argsort!` macro and its registration calls, `mod radix;`, `mod tests;`), `src/kernels/scalar.rs` (`pub mod sort;`), and `Registry::new` in `src/kernels.rs`, which calls `scalar::index::register` (also call `scalar::sort::register`, keeping the kernel registration order of `argsort_*` before `histogram` if any test or output depends on it)
* Create: `src/kernels/scalar/sort.rs` (the moved items and `register`), `src/kernels/scalar/sort/radix.rs` (moved from `index/radix.rs`), `src/kernels/scalar/sort/tests.rs` (moved from `index/tests.rs`)
* Delete: `src/kernels/scalar/index/radix.rs`, `src/kernels/scalar/index/tests.rs`

**Interfaces:**
* Produces: `crate::kernels::scalar::sort::{ArgSort, register}` and `sort::radix::{RadixKey, Digits, Pos, argsort_into, radix_into, scratch}` with their current visibility adjusted only as the move needs. Nothing new is exported from the crate. Kernel names, ids order, and `--plan` output are unchanged.

- [ ] **Step 1:** Move the items with their doc and inline comments. Fix `use` paths (`crate::kernels::scalar::index::radix::RadixKey` becomes `crate::kernels::scalar::sort::radix::RadixKey`). Keep `index.rs`'s module doc accurate by dropping `argsort` from its list, and give `sort.rs` a module doc.
- [ ] **Step 2:** Confirm that the test count of `cargo test --release` is unchanged, that `git diff --stat -M` shows the two files as renames plus small edits, and that `cargo run --release -- run tests/cli/dot.vk --plan` prints the same plan. Run clippy and fmt. Commit "Move argsort into its own kernel module".

### Task 2: The composite sort kernel and `argsort(a, b)`

Correctness-critical (tie order is output): review with `opus`.

**Files:**
* Modify: `src/check.rs` (`"argsort"` becomes `Builtin::Op(Op::ArgSort, 1..=2)`, like `group_id`), `src/kernels/scalar/sort.rs` (`ArgSort2<A, B>` body, `argsort2!` macros for the 16 pairs, `argsort2_reference` under `#[cfg(test)]`), `src/kernels/scalar/sort/radix.rs` or a new `src/kernels/scalar/sort/composite.rs` (the composite radix sort, declared with `mod composite;`)
* Modify: `src/kernels/scalar/sort/tests.rs`, or create `src/kernels/scalar/sort/composite/tests.rs` if the composite code gets its own module
* Modify: `tests/isa_variants.rs` (`WHOLE_AUTO_PREFIXES` covers `argsort2_` if `argsort_` does not), `tests/generated/eval/prims.rs` (two-argument `argsort` in the reference evaluator), `tests/generated/gen.rs` (`Catalog::new` skips `ArgSort` signatures with two inputs, with a comment that Task 4 enables them), `src/check/testdata` or `tests/programs` (type-check cases: `argsort(a, b)` accepted, three arguments and a float column rejected)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (catalog row for `argsort(a, b)` with a measurement placeholder that Task 6 fills)

**Interfaces:**
* Consumes: `RadixKey::image`, `scratch`, the `scatter` helper and `Pos` from Task 1.
* Produces: kernels `argsort2_<a>_<b>` for `a`, `b` in `i64`, `u64`, `i32`, `u8`, class `Whole`, op `Some(Op::ArgSort)`, inputs two `InputSpec::Array`, output `i64[]`, cost `width(a) + width(b) + 8`, `total: false`, `pattern: None`. `pub(crate) fn argsort2_into<A: RadixKey, B: RadixKey>(a: &[A], b: &[B], out: &mut [i64]) -> Result<(), String>` with the doc contract of the spec's "Semantics".

- [ ] **Step 1:** Write `argsort2_reference` (positions sorted by `(a[i], b[i], i)` with the standard library) and failing property tests for all 16 type pairs, with the boundary pools of the existing `argsort` tests, pools of few distinct values, lengths from a sweep around 1024 and 1025, and inputs built to give composite widths of 0, 64, 96 and 97 bits. Add fixed tests: empty pair, one row, (1, 2) against (2, 1), equal major keys with descending minor keys, a `u64` column holding 0 and `u64::MAX` (range reduction must not wrap), the length mismatch message, and a scratch allocation failure (follow `argsort_scratch_allocation_failure_is_an_error`).
- [ ] **Step 2:** Implement. One pass computes each column's minimum and maximum image, the reduced widths give the composite width, and the layout is 16 bytes (`u64` high, `u32` low, `u32` position) when the width is at most 96 and `n` fits `u32`, else 24 bytes. Inputs of at most 1024 rows use a comparison sort by `(a, b, position)`. Reuse the counting pass, the skip of constant digits, and the forward scatter of `radix_into`. Reserve scratch with `try_reserve_exact`.
- [ ] **Step 3:** Change the checker arity, add the reference evaluator case, and exclude the two-input `ArgSort` signatures from `Catalog::new`. Run `cargo test --release` (including `tests/isa_variants.rs` and the generated CI run, whose output must not change) and confirm that `argsort(a, b)` runs through the CLI. Commit "Add a two-column argsort".

### Task 3: The composite sort off switch

**Files:**
* Modify: `src/lower.rs` (the expansion where a two-argument `ArgSort` is lowered), `src/interp.rs` (`Options::no_composite_sort`, passing it to `lower`), `src/main.rs` (`--no-composite-sort`), `src/columns.rs` (`composite_sort_from_env`), `benches/acceptance.rs` and `examples/acceptance_stats.rs` (read `VERKEHRT_COMPOSITE_SORT`, document it in their module docs)
* Modify: `src/lower/tests.rs`, `tests/cli.rs`

**Interfaces:**
* Produces: `Options::no_composite_sort: bool` defaulting to false (doc: lower `argsort(a, b)` into two single-key sorts and gathers, which changes no value), `lower(prog, globals, reg, mode, composite_sort: bool)` or an equivalent options argument, `--no-composite-sort`, and `pub fn composite_sort_from_env() -> bool` (`off` turns the composite sort off, unset leaves it on, anything else panics, like `selective_from_env`).

- [ ] **Step 1:** Write failing tests: with `no_composite_sort`, `--plan` of `argsort(a, b)` contains `argsort_` twice, `gather_` twice, `same_len_` once, and no `argsort2_`, and the plan graph holds one node for `argsort(b)`; values equal the composite kernel's on fixed inputs with ties; a length mismatch raises `length mismatch: {len(a)} vs {len(b)}` in both modes with the same span; `--no-composite-sort` prints the same output as the default in a CLI test.
- [ ] **Step 2:** Implement the expansion with `Op::SameLen`, `Op::Length`, `Op::Gather` and the one-argument `Op::ArgSort` through the lowerer's existing op constructors, so the term graph shares `o1`. Wire the option, the flag, and the environment reader. Run the suite, clippy, fmt. Commit "Add an off switch for the two-column argsort".

### Task 4: Oracle 8 and generator coverage

**Files:**
* Modify: `tests/generated/gen.rs` (remove the Task 2 exclusion), `tests/generated/gen/tier2.rs` (a two-key branch in `indices`, and the arity balance of `generic` for `ArgSort`), `tests/generated/gen/features.rs` (a feature `argsort2` for the two-argument form), `tests/generated/oracles.rs` (oracle 8), `tests/generated/main.rs` (gate)
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (oracle list, gate, CI time)

**Interfaces:**
* Consumes: `Options::no_composite_sort`, `Prepared::same_plans`.

- [ ] **Step 1:** Enable the two-input signatures. `generic` picks among 4 one-key and 16 two-key `ArgSort` signatures, so pick the arity first (designer's suggestion: one key two times in three) and then the signature, to keep one-key sorts common. Add a two-key branch to `indices` with a small weight. Map two-argument `argsort` to the feature `argsort2`.
- [ ] **Step 2:** Add oracle 8: on the scalar interpreter with `Tiling::Rows(3)`, plan each program with and without `no_composite_sort`. Skip the second run when `Prepared::same_plans` holds, otherwise compare output and exports exactly and errors as oracle 2 does (failing output and kind), with the reason in the module doc. Count the programs whose plans differed.
- [ ] **Step 3:** Run the CI seed. If any existing gate falls below its minimum, change only the new weights until all hold, and if that fails, stop and report BLOCKED with the counts. Add `MIN_COMPOSITE_PLANS`, a third of the programs of the CI seed in which oracle 8 compared different plans, and fail the CI run below it. Time the debug CI run (under about 20 s, else BLOCKED with the time). Run a release search of 5,000 cases. Update the generated-program record with oracle 8, the gate, the counts, and the CI time. Commit "Check the two-column argsort against its expansion in the generated programs".

### Task 5: Fused top-k kernels

**Files:**
* Modify: `src/kernels/scalar/sort.rs` (or create `src/kernels/scalar/sort/topk.rs` with `topk/tests.rs`), registration of `topk_<t>` and `topk2_<a>_<b>` with `FusedPattern`s
* Modify: `tests/fused_oracle.rs` (top-k family), `tests/isa_variants.rs` (prefix `topk` in `WHOLE_AUTO_PREFIXES` if needed), `src/kernels/tests.rs` (if a test enumerates fused kernels or costs), `tests/generated/gen/tier2.rs` (a `slice(argsort(...), s, e)` shape with in-range bounds in `slice`), `tests/generated/main.rs` (gate), `benches/tier1.rs` (`tier2/topk-*` rows)
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (the top-k gate)

**Interfaces:**
* Consumes: `argsort_into`, `argsort2_into`, `run_slice`'s bounds check (extract its check into a shared `pub(crate) fn slice_bounds(start: i64, end: i64, len: usize) -> Result<(usize, usize), String>` in `src/kernels/scalar/index.rs` so both use one message).
* Produces: `const TOPK_MAX: usize = 64` with a doc that cites the bench, kernels `topk_<t>` (inputs `Array(t)`, `Scalar(i64)`, `Scalar(i64)`) and `topk2_<a>_<b>` (inputs `Array(a)`, `Array(b)`, `Scalar(i64)`, `Scalar(i64)`), class `Whole`, op `None`, output `i64[]`, patterns `(slice_i64 (argsort_<t> ?x) ?s ?e)` with vars `[?x, ?s, ?e]` and `(slice_i64 (argsort2_<a>_<b> ?x ?y) ?s ?e)` with vars `[?x, ?y, ?s, ?e]`, cost below the replaced argsort's.

- [ ] **Step 1:** Feasibility first. Register one `topk_i64` with a body that runs the full sort and slices, and check that `--plan` of `slice(argsort(x), 0, 10)` shows `topk_i64` and that `--no-fusion` shows `slice_i64 (argsort_i64 x)`. If the prelude rejects the pattern or extraction never picks it, stop and report BLOCKED with the planner output.
- [ ] **Step 2:** Write failing tests against `slice` of the full argsort, for every kernel: `e` in 0, 1, 10, `TOPK_MAX`, `TOPK_MAX + 1` and `n`, `s` in 0 and `e`, ascending, descending, and all-equal inputs, ties across the cut at `e`, each bounds error by message, and for `topk2` a length mismatch that must win over a bounds error. Add the top-k family to `tests/fused_oracle.rs` with a pool that draws in-range bounds most of the time.
- [ ] **Step 3:** Implement the selection: for `e <= TOPK_MAX`, keep the best `e` rows by (composite or single image, position) in a bounded heap or sorted buffer, order them, and return rows `s..e`, otherwise sort fully and copy. Bench `tier2/topk-*` at 80,000 and 1M `i64` rows with `e = 10` and `e = TOPK_MAX` on uniform and descending input against `argsort` plus `slice`, and lower `TOPK_MAX` if the selection loses on descending input. Record the numbers in the commit message and keep the raw output for Task 6.
- [ ] **Step 4:** Add the `slice(argsort(...), s, e)` shape to the generator, `MIN_TOPK_PLANS` (a third of the CI seed's plans that contain a `topk` kernel), and keep the existing minimums as in Task 4. Time the debug CI run. Update the generated-program record. Commit "Add fused top-k kernels".

### Task 6: Rewrite the queries and measure

**Files:**
* Modify: `tests/acceptance/clickbench/q32.vk`, `q31.vk`, `q30.vk` (group by `o = argsort(major, minor)` and gathers through `o`), `q15.vk`, `q35.vk` (`top = slice(argsort(0 - counts, gk), 0, min(ng, 10))`), and `q32.vk`'s count order if top-k applies there
* Create: `docs/measurements/2026-11-03-composite-keys.md`, `docs/measurements/2026-11-03-composite-keys/` (evidence files, `programs/clickbench/` with today's versions of the five files for `VERKEHRT_PROGRAMS`)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (fill the catalog rows for `argsort(a, b)` and top-k), `docs/superpowers/specs/2026-11-03-verkehrt-composite-keys-design.md` ("Outcome" paragraph in "Evidence")

- [ ] **Step 1:** Copy today's five programs into the evidence directory, then rewrite them. Keep each query's comments accurate and its output order unchanged. Run `cargo test --release` with `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH` set, so `clickbench_matches_duckdb` checks the rewrites, and run `acceptance_plans` before and after, committing both outputs.
- [ ] **Step 2:** Per-kernel statistics of the five queries in both forms (`VERKEHRT_PROGRAMS` for the old form), and the rewritten q32 with `VERKEHRT_COMPOSITE_SORT=off`, with the load at the start and end of each file.
- [ ] **Step 3:** Bench q32, q15 and q35 in three interleaved pairs (old, new, new, old, old, new) with `VERKEHRT_BENCH_SAMPLES=30` and `VERKEHRT_BENCH_SECS=10`, five variants each, and q31 and q30 in one pair, repeated once if they move beyond the band. Keep a rewrite only if it is faster beyond the run-to-run spread on most variants in every pass, otherwise restore the old file exactly and say so.
- [ ] **Step 4:** Write the record following `docs/measurements/2026-11-01-group-id-queries.md`: setup, evidence files, results with the five variants, decisions per query, per-kernel shares, the composite kernel against its off switch, the top-k bench from Task 5, the `acceptance_plans` diff, DuckDB and corgi ratios from the existing reference figures, and gaps, including the in-situ gap with the spec's probe numbers labeled as the designer's. Fill the catalog rows and the spec's "Outcome" paragraph. Commit "Measure the two-column argsort and top-k".
