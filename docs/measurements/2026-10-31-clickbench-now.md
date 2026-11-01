# ClickBench queries on today's primitives: Measurements

This document times all 14 ClickBench acceptance queries against corgi and DuckDB on one thread, and names the kernel that dominates each new query.
Nine queries are new in this round (q00, q03, q19, q27, q29, q30, q31, q40, q41), and the other five (q01, q02, q06, q07, q42) are re-measured so the table is complete.
Of the nine, q19 and q27 are clearly faster than DuckDB and corgi, q00 is comparable with neither (Verkehrt reads the length of an already-loaded column, DuckDB runs `COUNT(*)`, and corgi prints 0.00 ms, below its print resolution), and the other six are level with or slower than at least one reference.
The slower ones point at two cost centres: the sort inside grouping (q30, q31, q40, q41, issue #9) and elementwise maps that materialize before a reduction (q03, q29, issue #30).
The method follows `2026-10-31-tier2.md`, and every number below comes from a file in `2026-10-31-clickbench-now/`.
The last sections list the gaps and the candidates for plan 5.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | this branch after the q40 slice-first fix, plus the `acceptance_stats` example. The final review fixes then changed q03, q27 and q41, which were re-measured on the programs in this PR. The other later changes give the loader a per-suite list of signed columns and add tests, with no effect on the ClickBench inputs |
| corgi revision | `2751e04e8501d47b33219be23049bf4afa7a4428`, checkout `~/dev/repos/data/verkehrt/corgi-wip` |
| DuckDB | one-thread times from `duckdb.tsv` of the prepared data directory, written by `prepare.py` |
| Data | `hits_0.parquet`, 1,000,000 rows, the data directory of the tier 2 round |

Other work shares the host, so each run group records `/proc/loadavg` at its start and end.
The load ranged from 1.9 to 3.3 across all groups, lower than the 1.6 to 18 of the tier 2 round.
Differences within about 10% between runs or builds count as noise, and the text claims no change inside that band.

### Evidence files

* `bench-a.txt`: criterion log for q00, q01, q02, q03, q06, q07, q19. Load 2.19 at the start, 3.29 at the end.
* `bench-b.txt`: criterion log for q27, q29, q30, q31. Load 3.29 to 2.89.
* `bench-c.txt`: criterion log for q40, q41, q42. Load 2.89 to 2.40.
* `corgi.txt`: corgi's bench for the 14 queries at the pinned revision. Load 2.28 to 2.34.
* `bench-d.txt`: criterion log for q03, q27 and q41, re-measured on the programs in this PR. Load 3.08 to 4.16.
* `stats.txt`: per-kernel totals of each query, from the first build. Load 1.87 to 2.08. Its q03, q27 and q41 sections describe the older programs.
* `stats-rerun.txt`: per-kernel totals of q03, q27 and q41 on the programs in this PR. Load 3.75 to 3.63.
* `query-plans.txt`: the `acceptance_plans` output on the programs in this PR, the plan of every acceptance program.

## How the systems were timed

* Verkehrt: criterion with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time (`benches/acceptance.rs`, run as `cargo bench --bench acceptance -- 'clickbench/q(..|..)/'` with `VERKEHRT_CLICKBENCH` set). The timed region is `Prepared::execute` only. Tables give the criterion mean in ms. The variants are `scalar`, `x86-64-v3-auto`, `x86-64-v3-intrinsics`, `x86-64-v4-auto`, and `x86-64-v4-intrinsics`.
* corgi: `cargo bench --bench clickbench -- q00 q01 ...` in corgi's checkout with `CORGI_CLICKBENCH` set. It prints the best of at least 3 runs (up to 50 within 1.5 s) of `Program::run` with the input cloned per run. Its output has two decimals, so q00 prints 0.00 ms.
* DuckDB: the `one_thread_ms` column of `duckdb.tsv`.
* Per-kernel totals: `examples/acceptance_stats.rs`, run as `cargo run --release --example acceptance_stats -- QUERY` with `VERKEHRT_CLICKBENCH` set. The CLI's `run --stats` cannot supply column inputs, and `Prepared::execute` does not collect statistics. The example binds the loaded columns as globals so that `Interpreter::run` collects them, uses the CLI defaults (best detected ISA, intrinsics backend), and prints the run with the least total kernel time of five fresh runs. The stats timer wraps every kernel call, so the per-kernel times include the output allocation and its first-touch page faults, and their sum differs from the criterion mean of `x86-64-v4-intrinsics` (the variant the example runs) in both directions. For the nine new queries the difference ranges from -16% (q03) to +8% (q40). The short old queries differ more: q01 +49% (135.7 against 91.3 us), q02 +91% (334.6 against 175.1 us), q06 +18% (216.8 against 183.5 us), and q07 +62% (527.3 against 326.0 us). The shares of the nine new queries come from runs that differ from criterion by at most 16%, and the shares of any sub-millisecond query should be read as rough, because the gap there is large and its cause was not measured.

Criterion's mean is slower than a best-of time by a few percent, so the corgi and DuckDB columns favor those systems slightly against Verkehrt.

## Results

Times are criterion means in ms for Verkehrt and best-of in ms for corgi, and ratios below 1.0 mean Verkehrt is faster.
The ratio columns use the best Verkehrt variant of the row.
The corgi ratio for q00 is not computed because corgi's time prints as 0.00 ms, and Verkehrt's is 0.000175 ms. The DuckDB ratio for q00 is not a like-for-like comparison: Verkehrt's timed region reads the length of a column that is already loaded, while DuckDB's time covers running `COUNT(*)`.

| Query | scalar | v3-auto | v3-intrinsics | v4-auto | v4-intrinsics | corgi | DuckDB 1t | best variant | best / DuckDB | best / corgi |
|---|---|---|---|---|---|---|---|---|---|---|
| q00 (new) | 0.00018 | 0.00018 | 0.00018 | 0.00018 | 0.00018 | 0.00 | 0.34 | v4-auto | not comparable | n/a |
| q01 | 0.223 | 0.096 | 0.096 | 0.091 | 0.091 | 0.38 | 1.39 | v4-auto | 0.066 | 0.24 |
| q02 | 0.340 | 0.182 | 0.184 | 0.184 | 0.175 | 0.34 | 1.11 | v4-intrinsics | 0.158 | 0.51 |
| q03 (new) | 3.73 | 3.05 | 3.06 | 2.99 | 3.20 | 1.03 | 1.96 | v4-auto | 1.52 | 2.90 |
| q06 | 0.515 | 0.192 | 0.192 | 0.183 | 0.184 | 2.55 | 1.67 | v4-auto | 0.110 | 0.07 |
| q07 | 0.848 | 0.689 | 0.467 | 0.680 | 0.326 | 1.16 | 2.00 | v4-intrinsics | 0.163 | 0.28 |
| q19 (new) | 0.727 | 0.775 | 0.350 | 0.645 | 0.273 | 1.10 | 0.60 | v4-intrinsics | 0.455 | 0.25 |
| q27 (new) | 8.92 | 5.82 | 6.94 | 7.06 | 6.35 | 11.46 | 10.74 | v3-auto | 0.542 | 0.51 |
| q29 (new) | 42.34 | 31.38 | 31.46 | 31.82 | 32.07 | 33.14 | 7.24 | v3-auto | 4.33 | 0.95 |
| q30 (new) | 12.15 | 10.25 | 9.64 | 9.98 | 8.70 | 5.18 | 8.27 | v4-intrinsics | 1.05 | 1.68 |
| q31 (new) | 12.36 | 11.08 | 10.39 | 10.64 | 9.13 | 4.89 | 9.62 | v4-intrinsics | 0.95 | 1.87 |
| q40 (new) | 7.97 | 6.03 | 5.68 | 5.57 | 4.92 | 5.36 | 4.31 | v4-intrinsics | 1.14 | 0.92 |
| q41 (new) | 9.02 | 7.42 | 6.99 | 6.91 | 6.36 | 4.93 | 3.83 | v4-intrinsics | 1.66 | 1.29 |
| q42 | 4.47 | 2.87 | 2.73 | 2.52 | 2.26 | 8.29 | 7.08 | v4-intrinsics | 0.32 | 0.27 |

Against DuckDB, the new queries other than q00 split into faster (q19 at 0.46, q27 at 0.54), level (q31 at 0.95 and q30 at 1.05, inside the noise band), and slower (q40 at 1.14, q03 at 1.52, q41 at 1.66, q29 at 4.33).
Against corgi, q19 and q27 are faster, q00 is not comparable (corgi prints 0.00 ms), q40 (0.92) and q29 (0.95) are level, and q41 (1.29), q30 (1.68), q03 (2.90) and q31 (1.87) are slower.
All five variants of q00 are equal (0.175 to 0.179 us). Among the four non-scalar variants, q03 is within 7% and q29 within 3%, inside the noise band, so the ISA level and the backend change nothing measurable there.
For q07, q19, q30, q31, q40, q41 and q42, `x86-64-v4-intrinsics` is the best variant, with 8% (q41) to 58% (q19) less time than `x86-64-v4-auto`. For q27, `x86-64-v3-auto` is the best (5.82 ms), and the spread across its four non-scalar variants (5.82 to 7.06 ms) was not investigated.

The five re-measured queries agree with the tier 2 round for q01, q02, q06 and corgi's times, and differ for q07 and one q42 row.
q07 is faster in four variants: `x86-64-v4-intrinsics` 0.467 to 0.326 ms, `x86-64-v3-intrinsics` 0.620 to 0.467 ms, `x86-64-v3-auto` 0.824 to 0.689 ms, and `scalar` 0.956 to 0.848 ms, while `x86-64-v4-auto` is unchanged (0.683 and 0.680 ms).
q42 `x86-64-v4-intrinsics` went from 2.599 to 2.260 ms, and its other variants moved by 8% or less.
The four q07 changes (-30%, -25%, -16%, -11%) and the q42 row (-13%) all exceed the 10% noise band, and this document did not investigate their cause, so it claims no change from them.
The build differs from the tier 2 round's (the kernel sources changed in the x86-64 level commits and others), while the programs of q07 and q42 are unchanged since the tier 2 merge (`6033fe9`) and the data is the same.

## What dominates each new query

The kernel and its share come from `stats.txt`, and for q03, q27 and q41 from `stats-rerun.txt`, where the share is of the summed kernel time of one warm run.
Plans are in `query-plans.txt`.

| Query | Dominating kernel | Share | Reading |
|---|---|---|---|
| q00 | `length_i64` | 100% | 0.2 us, no work beyond reading the column's length. |
| q03 | `sub_i64`, `div_const_i64`, `mul_i64` | 32%, 30%, 30% | Three full-column elementwise passes (the split at 2^32), each about 0.8 to 0.9 ms. No single kernel dominates. `reduce_add_i64` is 7%. |
| q19 | `filter_eq_i64` | 100% | One pass over `UserID`, and the answer is empty. |
| q27 | `sub_i64` | 26% | Four calls (the row lengths from `URL_ends`, the shift of the keys by their minimum, and the two small negated sort keys). Then `compress_i64` (23%), `prev_i64` (21%), `histogram` (12%), and `reduce_into_add_i64` (12%). The two `argsort_i64` calls run on the groups that pass HAVING and take 1.4 us together. |
| q29 | `add_i64` | 75% | 89 calls, one materialized `ResolutionWidth + k` per sum. `reduce_add_i64` is the other 25% (90 calls). |
| q30 | `argsort_i64` | 49% | Three sorts (minor key, major key, group counts). Then `compress_i64` (15%), `prev_i64` (13%), `sub_i64` (11%). |
| q31 | `argsort_i64` | 50% | As q30. |
| q40 | `argsort_i64` | 44% | The same three sorts. The filter's `eq_i64`, `le_i64`, and `and_bool` together are 37% (1.95 ms of 5.30 ms). |
| q41 | `argsort_i64` | 54% | As q40. The filter kernels are 29% (1.92 ms of 6.53 ms). |

For q29, the 89 `add_i64` and 90 `reduce_add_i64` calls move 1.42 GB and 0.72 GB, and the 31.9 ms kernel total is 67 GB/s over those bytes.
DuckDB's 7.24 ms would be 296 GB/s over the same bytes if it ran the passes as written, which is far above the 67 GB/s that Verkehrt reaches on the same bytes with the 8 MB column in L3, so DuckDB most likely does not run them as written (an inference from the byte counts, not checked in DuckDB).
Corgi runs the passes as written too and takes 33.14 ms, so q29 is at parity with corgi.
The program keeps the ninety passes because the spec requires it, so this row measures passes and not an aggregate rewrite.

The sort-bound queries (q30, q31, q40, q41) match the spec's expectation that `argsort` dominates.
Corgi is faster on q30 and q31 (1.7x and 1.9x) and on q41 (1.3x), and Verkehrt is level on q40.
This document did not read corgi's per-op profile, so it does not attribute corgi's lead.
Verkehrt sorts the key columns twice with a stable sort to build a two-column key and sorts the group counts a third time.
This document did not measure how much of the `argsort` time is the second pass or the scalar sort itself, so it cannot say which a radix sort or a composite key would remove.

## Grouping by plan

* Sort-bound grouping, plan 5 and issue #9 (`argsort`): q30 (49%), q31 (50%), q40 (44%), q41 (54%). The tier 2 round already recorded `argsort` as a scalar sort at 0.10 GB/s.
* Mask chains, issue #16: q40 and q41 (the five-column filter, 37% and 29% of kernel time). q42 has the same mask shape.
* Elementwise maps that materialize before a reduction, issue #30 ("Fuse elementwise maps into reductions"): q29 (89 materialized `add_i64` outputs, each followed by `reduce_add_i64`) and q03 (`div_const`, `mul`, and `sub` materialized before `reduce_add`, plus a missing `rem` primitive).
* Not slower than a reference by more than noise: q19, q27, q30 and q31 (against DuckDB), q40 (against corgi), and q29 (against corgi).

## Gaps

No ClickBench query of the nine was left unwritten, so no query needs a missing primitive to match DuckDB.
These caveats stand on the committed programs:

* q27 has no `f64` `argsort`, so it sorts on an exact integer key: the integer part of the average and its fractional part scaled by 2^52, with two stable sorts. The key is exact for every average of at least 1, so the missing `f64` kernel costs q27 a second sort and no correctness. The second sort runs on the groups that pass HAVING, and both sorts together take 1.4 us in `stats-rerun.txt`.
* q19 and q41 return an empty answer on this data, so their programs match DuckDB trivially on output. Their timings cover the full computation (the filter in q19, the filter and three sorts in q41), and synthetic-data tests in `tests/acceptance/main.rs` run the committed programs on non-empty inputs.
* The aggregates of the empty-input case are unguarded, as in the tier 2 round (see its "Known limit: no NULL").
* The column loader gained two things that the data format needs and no language feature: it flips back the sign bit of the columns `prepare.py` stores flipped, and it reads `X_ends` inputs for string columns' row ends. q27, q30 and q31 depend on `_ends`.

## Plan 5 candidates

* Sort kernel for `argsort`: it is 44% to 54% of the four grouping queries and the largest single cost in them.
* A grouping path that avoids two stable sorts for a two-column key. This document measured only that `argsort` as a whole dominates.
* Fused predicate kernel for the five-column filter of q40, q41, and q42: 29% to 37% of the first two. Only worth doing if issue #16 includes mask chains of this width.
* Fuse elementwise maps into reductions (issue #30), which covers q03's split and q29's `sum(x + k)` passes.
