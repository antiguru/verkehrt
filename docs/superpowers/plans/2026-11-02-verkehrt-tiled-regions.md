# Tiled elementwise regions implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run chains of elementwise kernels tile by tile so their intermediates stay in cache, with results and errors identical to whole-array execution.

**Architecture:** The executor (`src/exec.rs`) finds regions, maximal subtrees of two or more elementwise array kernels, while evaluating a plan. It evaluates the region's inputs whole, checks their lengths, and runs the region's kernels per tile through `Dispatch::elementwise_tile`, writing the root's tiles into one output array. Any failure discards the attempt and evaluates the region whole. The planner and plans stay unchanged.

**Tech Stack:** Rust, the `tests/generated/` oracles, criterion benches, `examples/acceptance_stats.rs`.

**Spec:** `docs/superpowers/specs/2026-11-02-verkehrt-tiled-regions-design.md`

## Global Constraints

* A region's value equals the whole evaluation's value byte for byte, and a failing region raises the whole evaluation's error, kind, message, and span.
* No change to the planner, plans, `--plan` output, or kernel entries.
* `Cond` never belongs to a region.
* Stats count one call per kernel per region run with whole-array bytes and tile-summed time. Trace lines keep their format.
* `--no-fusion` does not turn tiling off. `--no-tiling` does.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays well under 30 s.
* No `#[path]`, no `mod.rs`, `#[expect]` not `#[allow]`, no `unsafe` without a SAFETY note. Comments have no em-dashes or structuring semicolons. Docs name no commit hashes.

## Review Focus

* An array input that is a view starting at a nonzero row, and a region whose inputs include the same array twice.
* A region of length 0, and a length that is an exact multiple of the tile row count.
* A region under a `Cond` branch that is not taken must not run, and a region inside a taken branch must tile.
* A region whose root feeds a non-elementwise kernel (`compress`, `sum`) and a non-elementwise kernel feeding a region input: the boundary nodes run whole exactly once.
* A failing kernel in the second tile of a region whose first tile succeeded: the error must equal the untiled error, and no partial output may escape.

---

### Task 1: Tiled regions in the executor

**Files:**
* Modify: `src/exec.rs` (regions, tiled evaluation, observation), `src/interp.rs` (`Options`, tiled-region counter), `src/main.rs` (`--no-tiling`, `--trace` help), `src/lib.rs` if a new public type needs exporting
* Create: `src/exec/tests.rs` (declared with `#[cfg(test)] mod tests;`)
* Modify: `tests/cli.rs` if a CLI test covers flags

**Interfaces:**
* Produces: `pub enum Tiling { Off, Rows(NonZeroUsize) }` exported from the crate root, `pub const DEFAULT_TILE_ROWS: usize = 4096`, `Options::tiling: Tiling` defaulting to `Tiling::Rows(DEFAULT_TILE_ROWS)`, and `Interpreter::tiled_regions(&self) -> u64`, the number of region runs that completed tiled (not through the fallback) since the interpreter was created, counted whether or not stats are on. `Prepared::execute` paths count into the same counter if they run through an interpreter, otherwise the builder exposes an equivalent on `Prepared` and the report names it.

- [ ] **Step 1:** Write failing unit tests in `src/exec/tests.rs` that run the same program tiled with `Tiling::Rows(n)` for n in 1, 3, 16 and with `Tiling::Off`, and compare values: a five-comparison `and`/`or` mask chain like TPC-H q06's over 100 rows, an arithmetic chain with a broadcast scalar, a chain over a `slice` view starting at row 7, a chain over an empty array, a chain whose length is a multiple of the tile row count, and a chain under the taken and untaken branch of an `if`. Add an error test: a chain containing a kernel that fails only on a row in a later tile, compared to the untiled error by kind, message, and span. Assert `tiled_regions` grows for tiled runs and stays 0 with `Tiling::Off`.
- [ ] **Step 2:** Implement region detection and tiled evaluation in `Machine::eval`. A region root is a `PlanNode::Kernel` with an elementwise array kernel that has at least one elementwise array kernel child. Evaluate region inputs whole, check equal array lengths, then run the kernels per tile with `Dispatch::elementwise_tile`, slicing inputs with `Array::slice` and passing scalars as `Arg::Splat`. Write the root's tiles into an output allocated once. On any error, evaluate the region with the existing whole path.
- [ ] **Step 3:** Observation: accumulate per-kernel tile times and emit one stat update and one trace line per kernel per region run, with whole-array byte counts, after the region's inputs. Update the `--trace` help text to say a line is one kernel over its whole arguments, which may have run as several tile calls.
- [ ] **Step 4:** Add `Options::tiling`, `--no-tiling`, and the counter. Run the suite, clippy, fmt. Commit "Run elementwise chains in tiles".

### Task 2: Tiling oracle and gate in the generated programs

**Files:**
* Modify: `tests/generated/oracles.rs`, `tests/generated/main.rs`, `tests/generated/eval.rs` if it builds interpreters
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (oracle list, gate)

**Interfaces:**
* Consumes: `Tiling`, `Options::tiling`, `Interpreter::tiled_regions` (or the `Prepared` equivalent Task 1 names).

- [ ] **Step 1:** Add oracle 4: run each case with `Tiling::Rows(3)` and with `Tiling::Off` on the fused plan, and require equal printed output, equal exports, and equal errors including message and span. Report a mismatch with the seed and program like the other oracles.
- [ ] **Step 2:** Add a gate `MIN_TILED_REGIONS` counting tiled region runs over the CI cases, set at about a third of the count the default CI run observes, and fail the run below it, as the rewrite gate does. If the generator rarely produces chains of two elementwise kernels over arrays, raise their weight and say so in the report.
- [ ] **Step 3:** Run the generated test in debug (must stay well under 30 s) and a release search of 5,000 cases. Update the generated-programs record with the oracle and the gate. Commit "Check tiled against untiled runs in the generated programs".

### Task 3: Measure the tiled regions

**Files:**
* Modify: `examples/acceptance_stats.rs` (TPC-H support: a suite argument or `VERKEHRT_TPCH`, its signed column list from `tests/acceptance/main.rs`)
* Modify: `src/exec.rs` or `src/interp.rs` (`DEFAULT_TILE_ROWS` if the measurement picks another value)
* Create: `docs/measurements/2026-11-02-tiled-regions.md` and `docs/measurements/2026-11-02-tiled-regions/` (evidence files)

- [ ] **Step 1:** Extend `acceptance_stats` to run TPC-H queries. Record per-kernel stats of q40, q41, q42, and TPC-H q06 with `--no-tiling` behavior (`Tiling::Off`) and with tiling, into evidence files.
- [ ] **Step 2:** Bench q40, q41, q42, and TPC-H q06 with tile row counts 1024, 4096, and 16384 (`VERKEHRT_BENCH_*` settings as the record states), pick the default, and set `DEFAULT_TILE_ROWS`.
- [ ] **Step 3:** Run the full acceptance benches (`benches/acceptance.rs`, both suites, five variants) with tiling on and off, recording load before and after each run. State every query that moved beyond the 10% noise band.
- [ ] **Step 4:** Write the record following `docs/measurements/2026-10-31-clickbench-now.md`: setup, evidence files, before and after tables, the tile choice, and the DuckDB and corgi ratios from the existing reference figures. Commit "Measure the tiled regions".
