# Selective predicate evaluation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In tiled regions, run the right operand of an `and` only over the rows of a tile from the first to the last row where the left operand is true, and the right operand of an `or` likewise for false rows, so the comparisons after a selective predicate read fewer column bytes (#49), with the values and errors of whole evaluation.

**Architecture:** `KernelDef` gains `total`, set at every registration site, true for every elementwise kernel except the `div_const` kernels. The executor's tiled run (`src/exec/region.rs` after Task 1) evaluates a region per tile as a recursion over row ranges: a guard, an `and_bool` or `or_bool` region kernel whose right argument is a region kernel subtree of total kernels, scans its left result for the first and last open row, runs the right subtree and itself over that range through `Dispatch::elementwise_tile`, and fills the rows outside it. Region inputs, the ordering rule, and the fallback to whole evaluation stay as they are. Statistics and trace report the rows each kernel ran over, and `Options::no_selective` turns the feature off.

**Tech Stack:** Rust, the `tests/generated/` oracles, criterion benches, `examples/acceptance_stats.rs`.

**Spec:** `docs/superpowers/specs/2026-11-02-verkehrt-selective-predicates-design.md`

## Global Constraints

* Values and errors (kind, message, span) equal whole evaluation byte for byte, on every ISA, backend, plan, tiling, and sharing setting. Narrowing skips only kernels with `total` set. Any tile failure still falls back to whole evaluation.
* Region inputs, their order, and the `ordered` rule of tiled regions are unchanged. Shared nodes stay region inputs and are never skipped.
* No change to the planner, plans, `--plan` output, or kernel entries. The tile row count stays 16384.
* Call counts are the same with and without narrowing: one call per region kernel per region run, also when every tile skipped it.
* With `no_selective`, with `Tiling::Off`, or when nothing narrows, statistics and trace equal today's except for the new `read` field of the stats line.
* `CARGO_INCREMENTAL=0`. `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays under about 20 s (16.3 to 16.4 s today per `docs/measurements/2026-11-01-generated-programs.md`).
* No `#[path]`, no `mod.rs` (a module is `name.rs` plus `name/`), `#[expect]` not `#[allow]`, no `unsafe` without a SAFETY note. Comments have no em-dashes or structuring semicolons, and existing comments move with their code. Docs name no commit hashes.

## Review Focus

* A right subtree containing a kernel that is not total (`div_const` with a zero divisor) under a left operand that is false in every row: no narrowing, and the error equals the untiled error.
* Fill values and offsets: false outside the range for `and`, true for `or`, for a guard that is the region root writing into the output array, a guard that is an inner kernel writing into a tile buffer read by `select`, and nested guards, whose ranges are relative to the tile and to the parent range, over an input view starting at a nonzero row.
* Guard eligibility: a left operand that is a region input array guards, a scalar operand never does, and a right operand that is a region input (including a shared node) is not narrowed.
* Observation: a kernel skipped in every tile still records one call with zero rows. Row counts and `read` bytes are those of the tile calls, a guard's scans and fills go to its time, and `narrowed_tiles` counts only region runs that completed tiled.
* A kernel failing in a later tile after earlier tiles narrowed: the attempt is discarded, no partial output escapes, and the error equals the untiled error by kind, message, and span.

---

### Task 1: Split the tiled regions out of `src/exec.rs`

Pure move, no behavior change. Selective evaluation adds to the region code in `src/exec.rs` (656 lines) and to `src/exec/tests.rs` (999 lines).

**Files:**
* Modify: `src/exec.rs`, `src/exec/tests.rs`
* Create: `src/exec/region.rs` (`Src`, `RegionKernel`, `Region`, `TiledRun`, `trivial`, and the `impl Machine` block with `region_kernel`, `collect`, `eval_region`, `eval_region_whole`, `run_tiled`), declared in `src/exec.rs` with `mod region;`
* Create: `src/exec/region/tests.rs`, declared in `src/exec/region.rs` with `#[cfg(test)] mod tests;`

**Interfaces:**
* Produces: the same public surface of `crate::exec`. `Machine` fields and methods that `region.rs` needs become `pub(super)` or stay private through a child `impl` block, the builder's choice, with nothing new exported from the crate.

- [ ] **Step 1:** Move the region items listed above to `src/exec/region.rs`, with their doc comments and inline comments. `Machine::eval_kernel` keeps calling `eval_region` and `call_whole`, and `record` and `observe` stay in `src/exec.rs`.
- [ ] **Step 2:** Move the tests that exercise tiling (from `tilings` users such as `mask_chain` through `a_shared_input_after_a_region_kernel_runs_whole`, and the probe kernels and plan builder they use) to `src/exec/region/tests.rs`. Helpers that the remaining tests in `src/exec/tests.rs` also use stay there as `pub(super)` items, and region tests reach them as `crate::exec::tests::...`. Test bodies do not change.
- [ ] **Step 3:** Confirm the test count of `cargo test --release` is unchanged and `git diff --stat` shows only moves and visibility edits. Run clippy and fmt. Commit "Move tiled regions into their own module".

### Task 2: Selective evaluation in tiled regions

**Files:**
* Modify: `src/kernels.rs` (`KernelDef::total`), every `KernelDef` literal: `src/kernels/scalar/elementwise.rs`, `reduce.rs`, `group.rs`, `scan.rs`, `index.rs`, `construct.rs`, `select.rs`, `fused.rs`, `src/kernels/tests.rs`, `src/exec/tests.rs`, `src/exec/region/tests.rs`
* Modify: `src/exec.rs` (`Strategy::selective`, `KernelStat::read`, stats line, `Observe::narrowed_tiles`, `record` taking a row count), `src/exec/region.rs` (guards, per-tile range recursion, row counts), `src/interp.rs` (`Options::no_selective`, `Interpreter::narrowed_tiles`, `Options::trace` doc), `src/main.rs` (`--no-selective`, `--trace` help), `src/lib.rs` if an export is needed
* Modify: `src/exec/region/tests.rs`, `src/kernels/tests.rs`, `tests/cli.rs` (one `--no-selective` test)

**Interfaces:**
* Consumes: `Dispatch::elementwise_tile`, `TileBuf`, `Array::slice`, `Region`, `Src` from Task 1.
* Produces: `KernelDef::total: bool` (doc: the kernel's entries never fail on arguments it accepts whose arrays have equal lengths), `Strategy::selective: bool`, `Options::no_selective: bool` defaulting to false, `Interpreter::narrowed_tiles(&self) -> u64` counted as `tiled_regions` is across `run`, `eval_repl`, and `Prepared::execute`, `KernelStat::read: u64`, and the stats line `stats NAME calls=C bytes=B read=R time=T`.

- [ ] **Step 1:** Add `total` to `KernelDef` and set it at every literal. In `src/kernels/scalar/elementwise.rs` it is true except for the kernels `register_div_const` adds (array and `_s`), and it is false at every other site and in the test kernels. Write kernel tests: every elementwise array kernel with `total` runs on every variant the host supports over rows of extreme values (0, 1, the type's minimum and maximum, NaN and infinities for floats, both bool values) without error, and the `div_const` kernels are the only elementwise kernels with `total` false.
- [ ] **Step 2:** Write failing executor tests in `src/exec/region/tests.rs`, each run with `Tiling::Rows(n)` for n in 1, 3, 16 and selective on and off, comparing values and call counts: an `and` chain whose left operand is false in whole tiles, false at both ends of a tile, and true throughout; an `or` chain with true rows; nested guards; a guard at the region root and one feeding `select`; a left operand that is a region input array; a right operand that is a region input; a chain over a `slice` view starting at row 7; an empty array. Assert `narrowed_tiles` grows where a range is shorter than its tile and stays 0 with selective off or `Tiling::Off`, and assert the row count a test computes for one skipped comparison from its `read` bytes. Error tests: `and(x > 5, div_const(y, z) > 0)` with every `x` at most 5 and `z` zero equals the untiled error, and a kernel failing in a later tile after earlier tiles narrowed equals the untiled error by kind, message, and span.
- [ ] **Step 3:** Implement guards and the per-tile recursion in `src/exec/region.rs`. A guard is a region kernel with op `And` or `Or` and output `bool[]` whose left argument is an array (a region kernel or an array input) and whose right argument is `Src::Kernel` of a subtree whose region kernels are all `total`, determined once per region run. Per tile, evaluate the left operand over the guard's range, scan forward and backward for the first and last open row, evaluate the right subtree and the guard only over that range with inputs sliced at the tile start plus the range offset, and fill the rest with 0 for `and` and 1 for `or`. An empty range calls no kernel of the right subtree. Increment `narrowed_tiles` per guard and tile whose range is shorter than the guard's range, only when the region run completes tiled. With `Strategy::selective` false the run is today's.
- [ ] **Step 4:** Observation: accumulate per region kernel the rows of its tile calls, record one call per region run with that row count, `read` as rows times the widths of its array arguments, and `write` as rows times its output width, and add a guard's scan and fill time to its own. Add `read` to `KernelStat` and the stats line. Keep the existing tests that compare tiled and untiled stats and trace lines comparing with selective off. Update the `--trace` help and the `Options::trace` doc: `n` is the rows the kernel ran over.
- [ ] **Step 5:** Add `Options::no_selective`, `--no-selective`, the counter, and a CLI test that `--no-selective` gives the same output as the default on a mask chain. Run the suite, clippy, fmt. Commit "Evaluate the right operand of and and or over the open rows of a tile".

### Task 3: Selective oracle and gate in the generated programs

**Files:**
* Modify: `tests/generated/oracles.rs`, `tests/generated/main.rs`
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (oracle list, gate, CI time)

**Interfaces:**
* Consumes: `Options::no_selective`, `Interpreter::narrowed_tiles`.

- [ ] **Step 1:** Add oracle 6: on the scalar interpreter with `Tiling::Rows(3)`, each plan run with selective evaluation and with `no_selective` gives equal printed output and exports, and equal errors by kind, message, and span. Report a mismatch with the seed and program like the other oracles. Update the module doc's oracle list, and state that oracles 1 to 5 run with selective evaluation on.
- [ ] **Step 2:** Add `MIN_NARROWED_TILES`, a third of the `narrowed_tiles` the fixed CI seed gives in oracle 6, and fail the CI run below it, as `MIN_SHARED_HITS` does. Print the count in the summary line. If the generator rarely produces `and` or `or` over two array comparisons, say so in the report instead of changing its weights.
- [ ] **Step 3:** Time the debug CI run, which must stay under about 20 s, and if it does not, stop and report BLOCKED with the time. Run a release search of 5,000 cases. Update the generated-program record with the oracle, the gate, and the CI time. Commit "Check selective against non-selective runs in the generated programs".

### Task 4: Measure selective evaluation

**Files:**
* Modify: `src/columns.rs` (`selective_from_env`), `benches/acceptance.rs`, `examples/acceptance_stats.rs` (read `VERKEHRT_SELECTIVE`, print `narrowed_tiles`)
* Create: `docs/measurements/2026-11-02-selective-predicates.md` and `docs/measurements/2026-11-02-selective-predicates/` (evidence files)

- [ ] **Step 1:** Add `selective_from_env`, which reads `VERKEHRT_SELECTIVE` (`off` turns it off, unset leaves it on, anything else panics) like `sharing_from_env`, and use it in the acceptance bench and the stats example. The stats example prints `narrowed_tiles` after `tiled_regions` and documents the variable. Commit "Read the selective setting in the acceptance tools".
- [ ] **Step 2:** Record the `--plan` output of q40, q41, q42, and TPC-H q06 and the operand order of each guard. Record per-kernel stats of the four queries with `VERKEHRT_SELECTIVE=off` and on, default tiling, `x86-64-v4`, into evidence files. Check the rows the mask comparisons evaluate (from `read` bytes) against the spec's simulation table, which predicts 0.497, 0.512, 0.531, and 0.998 of comparison rows at 16384-row tiles, and explain any difference.
- [ ] **Step 3:** Bench q40, q41, q42, and TPC-H q06, five variants, selective off and on, with the 1-minute load at or below 4 and recorded at the start and end of each file, and a second pair for every query that leaves the 10% band. State whether comparison time fell in proportion to the rows evaluated.
- [ ] **Step 4:** Run the full acceptance benches, both suites, five variants, selective off and on. State every query that moved beyond the 10% noise band, and repeat those with a second pair.
- [ ] **Step 5:** Write the record following `docs/measurements/2026-11-02-tiled-regions.md`: setup, evidence files, the plans' guard order, the rows and `read` bytes of the comparisons against the simulation, before and after tables, the full suites, the DuckDB and corgi ratios from the existing reference figures, and whether the evidence supports selection vectors for TPC-H q06 as the next step, as the spec's "Mechanism" section states. Add an "Outcome" paragraph to the spec's "Evidence" section with the numbers. Commit "Measure selective evaluation".
