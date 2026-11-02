# Fused reductions over tiled regions implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a reduction be the root of a tiled region, so each tile of an elementwise chain feeds the reduction's accumulator and the chain's result is never written in full (#30), with the values and errors of whole evaluation, and measure the effect on ClickBench q03.

**Architecture:** The executor's region code (`src/exec/region.rs`) recognizes a kernel of class `Reduction` with at least one region-kernel argument as a fused region root. It evaluates the region's inputs as today, runs the elementwise kernels per tile into tile buffers through `Dispatch::elementwise_tile`, and feeds each tile to the accumulator from `Dispatch::reducer`, in tiles rounded up to a multiple of 16 rows so the `Reducer` contract reproduces the whole call's bits. Any failure discards the attempt and evaluates region and root whole. Statistics and trace equal the unfused run's apart from time, `Options::no_reduction_fusion` turns the feature off, and `Interpreter::fused_reductions` counts completed runs per root kernel. The planner and plans stay unchanged, and `rem` is deferred.

**Tech Stack:** Rust, the `tests/generated/` oracles, criterion benches, `examples/acceptance_stats.rs`.

**Spec:** `docs/superpowers/specs/2026-11-02-verkehrt-map-reduce-fusion-design.md`

## Global Constraints

* Values and errors (kind, message, span) equal the unfused run byte for byte, on every ISA, backend, plan, tiling, sharing, and selective setting. Floats included, compared by bits. Any failure of a fused attempt, `finish` included, falls back to whole evaluation of the region and the root.
* Region inputs, their order, and the `ordered` rule of tiled regions are unchanged. Shared nodes stay region inputs. A fused region's tiles are `t` rounded up to a multiple of 16 (down on overflow), every `update` but the last gets a multiple of 16 rows, and an empty array gets no `update`. Plain regions keep `t` rows.
* No change to the planner, plans, `--plan` output, kernel entries, or the `acceptance_plans` output. The default tile row count stays 16384.
* With fusion on or off, call counts, row counts, `bytes`, and `read` in statistics and the trace lines' fields other than time are equal. `--no-tiling` turns fusion off, `--no-fusion` keeps its meaning.
* `CARGO_INCREMENTAL=0`. `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays under about 20 s (16.4 s today per `docs/measurements/2026-11-01-generated-programs.md`).
* No `#[path]`, no `mod.rs` (a module is `name.rs` plus `name/`), `#[expect]` not `#[allow]`, no `unsafe` without a SAFETY note. Comments have no em-dashes or structuring semicolons, and existing comments move with their code. Docs name no commit hashes.

## Review Focus

* Tile rows and float bits: `Tiling::Rows(1)`, `Rows(3)`, and `Rows(17)` run fused regions in 16, 16, and 32-row tiles, the last tile may be short, an empty array runs one empty elementwise tile and no `update`, and float sums and products equal the unfused bits on every variant the host supports.
* Fallback: a failing `update`, a failing `finish` (`reduce_min` of an empty chain), a `div_const` with a zero divisor inside the region under a guard, and a kernel failing after earlier tiles were fed must raise the unfused error by kind, message, and span, reusing the inputs the attempt evaluated, with no accumulator state escaping.
* Root detection: a reduction with no region-kernel argument stays a whole call, a shared root goes through the memo and fuses on its first evaluation, a shared elementwise argument is an input, `dot` with two region-kernel arguments or one input argument, a scalar argument passes as `Arg::Splat`, and a non-trivial input after a region kernel in post-order makes the region run whole.
* Observation: the root records one call with the region's rows, `read` of rows times its array argument widths, no `write`, and its `update`, `finish`, and tile-buffer allocation time. The topmost elementwise kernels report the same bytes as unfused. Trace order is inputs, elementwise kernels in post-order, root. `tiled_regions` and `fused_reductions` grow only for runs that complete tiled.
* Selective evaluation inside a fused region: a guard as the root's argument (`count(and(...))`) fills its whole tile before `update`, nested guards narrow as today, and `narrowed_tiles` counts the same with fusion on and off.

---

### Task 1: Split the per-tile runner out of `src/exec/region.rs`

Pure move, no behavior change. Fusion adds to the region code in `src/exec/region.rs` (517 lines) and its tests in `src/exec/region/tests.rs` (925 lines).

**Files:**
* Modify: `src/exec/region.rs`, `src/exec/region/tests.rs`
* Create: `src/exec/region/tiles.rs` (`SCAN_BLOCK`, `Tiles` and its `impl`, `open_span`, `dst`), declared in `src/exec/region.rs` with `mod tiles;`
* Create: `src/exec/region/tiles/tests.rs`, declared in `src/exec/region/tiles.rs` with `#[cfg(test)] mod tests;`

**Interfaces:**
* Produces: the same public surface of `crate::exec`. Items that `tiles.rs` and `region.rs` share (`Src`, `Region`, `RegionKernel`, `Tiles` fields) become `pub(super)` as needed, with nothing new exported from the crate.

- [ ] **Step 1:** Move the items listed above to `src/exec/region/tiles.rs` with their doc comments and inline comments. `run_tiled` and `guards` stay in `src/exec/region.rs`.
- [ ] **Step 2:** Move the selective-evaluation tests (from `tile_rows` through `selective_evaluation_without_tiles_changes_nothing`, including `the_open_span_scan_matches_a_row_by_row_scan`) and the helpers only they use to `src/exec/region/tiles/tests.rs`. Helpers that the remaining region tests also use stay in `src/exec/region/tests.rs` as `pub(super)` items. Test bodies do not change.
- [ ] **Step 3:** Confirm the test count of `cargo test --release` is unchanged and `git diff --stat` shows only moves and visibility edits. Run clippy and fmt. Commit "Move the per-tile runner into its own module".

### Task 2: Fused reductions in the executor

Correctness-critical: review with `opus`.

**Files:**
* Modify: `src/kernels.rs` (`REDUCE_BLOCK`), `src/kernels/scalar/reduce.rs` (`LANES` defined as `REDUCE_BLOCK`)
* Modify: `src/exec.rs` (`Strategy::fuse_reductions`, `Observe::fused_reductions`, `Machine` field, `eval_kernel` dispatching to the fused path), `src/exec/region.rs` (root detection, `collect` for a reduction root, the fallback through `eval_region_whole`), `src/exec/region/tiles.rs` (topmost elementwise kernels write into tile buffers in a fused run)
* Create: `src/exec/region/fused.rs` (the fused run and the root's observation, as a child `impl Machine` block), declared in `src/exec/region.rs` with `mod fused;`, and `src/exec/region/fused/tests.rs`, declared in `fused.rs` with `#[cfg(test)] mod tests;`
* Modify: `src/interp.rs` (`Options::no_reduction_fusion`, `Interpreter::fused_reductions`, wiring in `run`, `eval_repl`, and `Prepared::execute`), `src/main.rs` (`--no-reduction-fusion`), `src/lib.rs` if an export is needed, `tests/cli.rs` (one `--no-reduction-fusion` test)

**Interfaces:**
* Consumes: `Dispatch::reducer`, `Reducer`, `Dispatch::elementwise_tile`, `TileBuf`, `Array::slice`, `Region`, `Src`, `Tiles` from Task 1.
* Produces: `crate::kernels::REDUCE_BLOCK: usize = 16` (doc: the row multiple of every `Reducer::update` but the last), `Strategy::fuse_reductions: bool` (doc: a kernel of class `Reduction` with a region-kernel argument runs as the root of a tiled region, with the values and errors of a whole call, no effect without tiling), `Options::no_reduction_fusion: bool` defaulting to false, `Interpreter::fused_reductions(&self) -> BTreeMap<String, u64>` (per root kernel name, fused runs that completed tiled, only nonzero entries), and `--no-reduction-fusion`.

- [ ] **Step 1:** Write failing tests in the fused test module, each run with `Tiling::Rows(t)` for t in 1, 3, 16, 17, 48 and with fusion on and off, and also against `Tiling::Off`, comparing values by bits and stats lines field by field apart from time: `sum` over a two-kernel chain, over one elementwise kernel, over a `slice` view starting at row 7, over an empty array, and over a length that is a multiple of the tile; `min`, `max`, `argmin`, `argmax` over chains with ties and a NaN in a later tile; `count`, `any`, `all` over a mask chain whose root argument is a guard; `dot` with two region-kernel arguments and with one input argument; a root with a scalar argument; a shared root read by two statements; a reduction over a shared elementwise node; a reduction over a leaf. Assert `fused_reductions` per root name, that `tiled_regions` grows by the same count, that both stay unchanged with fusion off or `Tiling::Off`, and the trace order inputs, elementwise kernels in post-order, root.
- [ ] **Step 2:** Write failing float and error tests. Float: `sum` and product over `mul` and `add` chains of `f64` and `f32` with mixed magnitudes, signed zeros, and NaN, on every variant `crate::exec::tests::variants()` returns, fused with tiles of 16, 48, and 16384 rows against the unfused whole call, by bits. Errors: `reduce_min` of an empty chain, `div_const` with a zero divisor inside a fused region under an `and` guard with selective evaluation on, a probe kernel failing in a later tile after earlier tiles were fed (reuse the `FailAt` pattern of `src/exec/region/tests.rs`), a non-trivial input after a region kernel in post-order, and `dot` over arrays of different lengths. Each must equal the unfused error by kind, message, and span.
- [ ] **Step 3:** Implement. In `eval_kernel`, with `Tiling::Rows(t)` and `fuse_reductions`, a node whose kernel has class `Reduction` and at least one argument for which `region_kernel` is `Some` evaluates as a fused region. `collect` gathers the elementwise subtrees below the root's arguments and the root's other arguments as inputs, in post-order, with the `ordered` rule unchanged. The fused run evaluates inputs, checks equal array lengths, allocates one tile buffer per elementwise kernel, runs tiles of `t` rounded up to a multiple of `REDUCE_BLOCK` (down on overflow) through the existing per-tile recursion with guards, calls `update` with the root's arguments over each tile (tile buffer rows, input slices, `Arg::Splat` for scalars), skips `update` for an empty array after one empty elementwise tile, and calls `finish`. Any failure returns the evaluated inputs, and the fallback evaluates the root's arguments through `eval_region_whole` and calls the root with `call_whole`. Count the run in `tiled_regions` and `fused_reductions` only when it completes.
- [ ] **Step 4:** Observation: record the elementwise kernels as a plain region does, and the root with one call, the region's rows, `read` of rows times the widths of its array arguments, no `write`, and the time of its `update` and `finish` calls plus the tile-buffer allocation. Update the `Options::trace` doc and the `--trace` help only if their wording no longer holds.
- [ ] **Step 5:** Add `Options::no_reduction_fusion`, `--no-reduction-fusion`, the per-kernel counter, and a CLI test that `--no-reduction-fusion` gives the same output as the default on `sum` over a chain. Run the suite, clippy, fmt. Commit "Fuse reductions into tiled regions".

### Task 3: Fused oracle and shape gates in the generated programs

**Files:**
* Modify: `tests/generated/oracles.rs`, `tests/generated/main.rs`
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (oracle list, gates, CI time)

**Interfaces:**
* Consumes: `Options::no_reduction_fusion`, `Interpreter::fused_reductions`.

- [ ] **Step 1:** Add oracle 7: on the scalar interpreter with `Tiling::Rows(3)`, each plan run with fusion and with `no_reduction_fusion` gives equal printed output and exports, and equal errors by kind, message, and span. Reuse the existing tiled run as the fused side, so the oracle adds one run per plan. Report a mismatch with the seed and program like the other oracles. Update the module doc's oracle list and state that oracles 1 to 6 run with fusion on.
- [ ] **Step 2:** Map root kernel names to the shape classes of the spec (integer sums and products, float sums and products, `min` and `max`, `argmin` and `argmax`, bool reductions, bitwise folds, fused multi-argument kernels) in the test, accumulate them across workers, and print them in the summary. Add `MIN_FUSED_REDUCTIONS`, a third of the total the fixed CI seed gives, and `FUSED_CLASSES`, the classes the CI seed exercises, and fail the CI run when the total falls below the minimum or a listed class has no fused run. Do not change the generator's weights, and name any class the CI seed does not produce in the report.
- [ ] **Step 3:** Time the debug CI run, which must stay under about 20 s, and if it does not, stop and report BLOCKED with the time. Run a release search of 5,000 cases and record its per-class counts. Update the generated-program record with the oracle, the gates, the classes the CI seed lacks, and the CI time. Commit "Check fused against unfused reductions in the generated programs".

### Task 4: Measure fused reductions

**Files:**
* Modify: `src/columns.rs` (`reduction_fusion_from_env`), `benches/acceptance.rs`, `examples/acceptance_stats.rs` (read `VERKEHRT_REDUCTION_FUSION`, print `fused_reductions`)
* Create: `docs/measurements/2026-11-02-map-reduce-fusion.md` and `docs/measurements/2026-11-02-map-reduce-fusion/` (evidence files and probe programs)
* Modify: `docs/superpowers/specs/2026-11-02-verkehrt-map-reduce-fusion-design.md` ("Outcome" paragraph in "Evidence")

- [ ] **Step 1:** Add `reduction_fusion_from_env`, which reads `VERKEHRT_REDUCTION_FUSION` (`off` turns fusion off, unset leaves it on, anything else panics) like `selective_from_env`, and use it in the acceptance bench and the stats example. The stats example prints `fused_reductions` after `narrowed_tiles` and documents the variable. Commit "Read the reduction fusion setting in the acceptance tools".
- [ ] **Step 2:** Run `acceptance_plans` and confirm it equals `docs/measurements/2026-11-02-term-graph/plans-after.txt` apart from the timing line, and commit it as evidence. Record per-kernel stats of q03, TPC-H q06, and TPC-H q01 with fusion off and on, default tiling, `x86-64-v3` and `x86-64-v4`. Copy the three probe programs of the spec's "Evidence" into the evidence directory, run them through `acceptance_stats` the same way (the example reads programs from `tests/acceptance/`, so copy each there for the run and delete it afterwards, never commit it there), and record their stats. Check that only q03 has nonzero `fused_reductions` among the three queries.
- [ ] **Step 3:** Bench q03 in at least three interleaved pairs (off, on, on, off, off, on), five variants each, with the 1-minute load at or below 4 and recorded at the start and end of each file, and TPC-H q06 and q01 in one pair as controls. State the q03 ratio per pair and its spread, and compare the stats against the spec's estimate of about 700 us saved from `sub_i64`.
- [ ] **Step 4:** Run the full acceptance benches, both suites, five variants, fusion off and on. State every query that moved beyond the 10% noise band and repeat those with a second pair.
- [ ] **Step 5:** Write the record following `docs/measurements/2026-11-02-selective-predicates.md`: setup, evidence files, the unchanged plans, the stats and probes, the q03 pairs, the controls, the full suites, the DuckDB and corgi ratios of q03 and TPC-H q06 from the existing reference figures, and the division cost that remains in q03 as the input for the `rem` or power-of-two follow-up. Add an "Outcome" paragraph to the spec's "Evidence" section with the numbers. Commit "Measure fused reductions".
