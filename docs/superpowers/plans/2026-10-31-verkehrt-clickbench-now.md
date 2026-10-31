# ClickBench queries on today's primitives implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ClickBench q00, q03, q19, q27, q29, q30, q31, q40, and q41 to the acceptance suite, matching DuckDB byte for byte, and time them against corgi and DuckDB.

**Architecture:** Each query is one `.vk` program in `tests/acceptance/clickbench/`, which the acceptance test (`tests/acceptance/main.rs`, `run_suite`) and bench (`benches/acceptance.rs`) discover by directory listing. The only library change is the column loader's sign flip-back and `_ends` inputs (`src/columns.rs`), plus possibly one harness output kind.

**Tech Stack:** Verkehrt `.vk` programs, the acceptance harness, criterion, corgi's ClickBench bench, DuckDB timings from `prepare.py`.

**Spec:** `docs/superpowers/specs/2026-10-31-verkehrt-clickbench-now-design.md`

## Global Constraints

* Data: `VERKEHRT_CLICKBENCH=~/dev/repos/data/verkehrt/clickbench`. Expected answers are `expected/qNN.txt`. Corgi's versions are in `~/dev/repos/data/verkehrt/corgi-wip/corgi/algorithms/clickbench/qNN.col`, and their header comments give the SQL as `prepare.py` runs it (including the scaled `HAVING` and the extra `ORDER BY` tiebreaks).
* Each program starts with `// sql:` lines copying the SQL from the corgi header and an `// output: [...]` line with the kinds, like the existing programs.
* A program's output matches `expected/qNN.txt` byte for byte. A query that cannot match is not committed. Instead, record the gap in the task report with the missing primitive or semantics.
* No new primitives, kernels, or language features.
* Comments in programs explain non-obvious steps in full sentences. No em-dashes or structuring semicolons.
* Run `CARGO_INCREMENTAL=0 VERKEHRT_CLICKBENCH=... cargo test --release --test acceptance` for the suite. Run `cargo test --release` and `cargo clippy --all-targets -- -D warnings` before each commit.

## Review Focus

* Empty answers (q19, q41): the program must compute the full query and still produce an empty output, and must not fail on empty intermediate arrays (`reduce` of an empty array, a bucket count of zero), as `q07_program` in `tests/acceptance/main.rs` guards.
* Ties in `ORDER BY ... LIMIT`: the expected answers break ties by the extra keys in the `sql:` lines. The program must reproduce that order, not just the top values.
* Float formatting: the `f` and `g` kinds must print exactly as `prepare.py` writes them.

---

### Task 1: Simple and empty-answer queries

**Files:**
* Create: `tests/acceptance/clickbench/q00.vk`, `q19.vk`, `q29.vk`, `q41.vk`

- [ ] **Step 1:** Write the four programs from the spec's query notes. For q29, write ninety separate sums as the SQL does.
- [ ] **Step 2:** Run the acceptance test and confirm all ClickBench queries match.
- [ ] **Step 3:** Commit "Add ClickBench q00, q19, q29, and q41".

### Task 2: Exact average and string lengths

**Files:**
* Create: `tests/acceptance/clickbench/q03.vk`, `q27.vk`
* Possibly modify: `tests/acceptance/harness.rs` and `tests/acceptance/main.rs` (only to read an output kind that `prepare.py` writes, with a unit test like `harness_formats_like_prepare_py`)

- [ ] **Step 1:** Write q03. The sum of `UserID` exceeds `i64`, so split each id at 2^32 (`div_const` and the remainder) and sum the halves separately, as corgi's `q03.col` describes. Then combine them in `f64` so the printed value equals `expected/q03.txt`. If no exact combination exists with today's primitives, record the gap instead.
- [ ] **Step 2:** Write q27. URL lengths come from the URL column's `ends` offsets (see `src/columns.rs` for how a `_bytes` input and its ends are loaded). Group by `CounterID` shifted by its minimum with `histogram` and `reduce_into`, keep the groups whose count exceeds 1000, and order by average length descending, then `CounterID`, limit 25.
- [ ] **Step 3:** Run the acceptance test, then commit "Add ClickBench q03 and q27".

### Task 3: Sort-based grouping

**Files:**
* Create: `tests/acceptance/clickbench/q30.vk`, `q31.vk`, `q40.vk`

- [ ] **Step 1:** Write q40: filter, then group by (`URLHash`, `EventDate`). Sort with two stable `argsort` passes (minor key first), find run starts with `prev` and a comparison, count per run with `scan_by` or with differences of run positions, order by count descending and then the keys, and apply `LIMIT 10 OFFSET 100`.
- [ ] **Step 2:** Write q30 (group by `SearchEngineID`, `ClientIP`, with count, `SUM(IsRefresh)`, and `AVG(ResolutionWidth)` per group) and q31 (group by `WatchID`, `ClientIP`) the same way, with `WHERE SearchPhrase <> ''` from the phrase lengths.
- [ ] **Step 3:** Run the acceptance test, then commit "Add ClickBench q30, q31, and q40".

### Task 4: Measurement

**Files:**
* Create: `docs/measurements/2026-10-31-clickbench-now.md`, `docs/measurements/2026-10-31-clickbench-now/` (raw output)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (only the acceptance workload list, if it enumerates queries)

- [ ] **Step 1:** Time every committed query with the method of `docs/measurements/2026-10-31-tier2.md`: the acceptance bench for Verkehrt, corgi's ClickBench bench at the pinned revision, and DuckDB's one-thread times from `duckdb.tsv`. Record the load average and the host setup (Zen 5, boost off, `powersave`, 3.0 GHz under load per the user). Keep the raw output.
- [ ] **Step 2:** For each query, capture `--stats` and name the kernel that dominates its time.
- [ ] **Step 3:** Write the document: a table of best Verkehrt variant, corgi, and DuckDB per query with ratios, the dominating kernel, and a "Gaps" section listing each query that could not be written, with the missing primitive. Treat differences within the build-to-build shift of about 10% as noise.
- [ ] **Step 4:** Commit "Measure the new ClickBench queries".
