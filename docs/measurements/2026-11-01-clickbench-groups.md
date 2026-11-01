# Many-group ClickBench queries: Measurements

This document times the six ClickBench queries added in plan 5 (q04, q08, q09, q15, q32, q35) against corgi and DuckDB on one thread, and re-times the four sort-bound queries of the #20 round (q30, q31, q40, q41) on the radix `argsort`.
Of the six new queries, q35 is faster than corgi and slower than DuckDB, q32 is level with DuckDB and slower than corgi, q04, q08 and q09 are level with corgi and 2.9 to 4.9 times slower than DuckDB, and q15 is slower than both.
In all six, `argsort_i64` is the largest kernel at 53% to 89% of kernel time, and `gather_i64` is the second largest in q08, q09, q32 and q35 at 11% to 31%.
Compared with the #20 round, q30, q31, q40 and q41 take 28% to 48% less time on the best variant, and `argsort_i64` is no longer the largest kernel in q30, q40 and q41 and is about 30% or less of kernel time in any of the four (it was 44% to 54%).
The comparison with #20 spans two sessions on the same host, and the last sections list the gaps.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt commit | `62bdfd7` (branch `acceptance/clickbench-groups`) |
| corgi revision | `2751e04e8501d47b33219be23049bf4afa7a4428`, checkout `~/dev/repos/data/verkehrt/corgi-wip`, the same as in the #20 round |
| DuckDB | one-thread times from `duckdb.tsv` of the prepared data directory |
| Data | `hits_0.parquet`, 1,000,000 rows, the data directory of the #20 round |

Other work shares the host, so each run group records `/proc/loadavg` at its start and end.
The load ranged from 1.78 to 4.42 across all groups, and the highest value is the end of `bench-b.txt`.
Differences within about 10% between runs or builds count as noise, and the text claims no change inside that band.

### Evidence files

The raw output is in `docs/measurements/2026-11-01-clickbench-groups/`.

* `bench-a.txt`: criterion log for q04, q08, q09, q15, q32. Load 1.78 at the start, 2.41 at the end.
* `bench-b.txt`: criterion log for q35, q30, q31, q40, q41. Load 2.30 to 4.42.
* `corgi.txt`: corgi's bench for the ten queries at the pinned revision. Load 3.89 to 3.60.
* `stats.txt`: per-kernel totals of each query. Load 3.17 to 2.84.
* `query-plans.txt`: the `acceptance_plans` output, the plan of every acceptance program.

The #20 numbers come from `docs/measurements/2026-10-31-clickbench-now/` (`bench-*.txt`, `corgi.txt`, `stats.txt`).

## How the systems were timed

The method is that of `2026-10-31-clickbench-now.md`.

* Verkehrt: criterion with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time (`benches/acceptance.rs`, run as `cargo bench --bench acceptance -- 'clickbench/q(..|..)/'` with `VERKEHRT_CLICKBENCH` set). The timed region is `Prepared::execute` only. Tables give the criterion mean in ms.
* corgi: `cargo bench --bench clickbench -- q04 q08 ...` in corgi's checkout with `CORGI_CLICKBENCH` set. It prints the best of at least 3 runs of `Program::run` with the input cloned per run, to two decimals.
* DuckDB: the `one_thread_ms` column of `duckdb.tsv`.
* Per-kernel totals: `cargo run --release --example acceptance_stats -- QUERY` with `VERKEHRT_CLICKBENCH` set. It prints the run with the least total kernel time of five fresh runs, with the CLI's defaults (best detected ISA, intrinsics backend). The stats timer wraps every kernel call, so per-kernel times include output allocation and first-touch page faults.

The summed kernel time differs from the criterion mean of `x86-64-v4-intrinsics` by -4% to +2% for the six new queries (q04 25.65 against 25.73 ms, q08 46.97 against 48.42, q09 56.02 against 54.86, q15 31.75 against 32.57, q32 68.93 against 72.11, q35 28.81 against 28.97).
For the four queries of the #20 round it is higher by +14% to +25% (q30 5.87 against 4.71 ms, q31 6.32 against 5.55, q40 4.02 against 3.53, q41 3.77 against 3.14).
The shares of the four short queries are therefore rough, and this document did not measure the cause of the gap.
Criterion's mean is slower than a best-of time by a few percent, so the corgi and DuckDB columns favor those systems slightly against Verkehrt.

## Results

Times are criterion means in ms for Verkehrt and best-of in ms for corgi, and ratios below 1.0 mean Verkehrt is faster.
The ratio columns use the best Verkehrt variant of the row.

| Query | scalar | v3-auto | v3-intrinsics | v4-auto | v4-intrinsics | corgi | DuckDB 1t | best variant | best / DuckDB | best / corgi |
|---|---|---|---|---|---|---|---|---|---|---|
| q04 (new) | 26.32 | 26.14 | 26.21 | 27.12 | 25.73 | 27.04 | 6.91 | v4-intrinsics | 3.72 | 0.95 |
| q08 (new) | 49.66 | 49.02 | 49.62 | 49.56 | 48.42 | 48.88 | 9.85 | v4-intrinsics | 4.92 | 0.99 |
| q09 (new) | 58.61 | 56.51 | 57.70 | 54.86 | 54.86 | 53.81 | 19.09 | v4-auto | 2.87 | 1.02 |
| q15 (new) | 34.22 | 33.41 | 33.72 | 32.52 | 32.57 | 28.49 | 9.22 | v4-auto | 3.53 | 1.14 |
| q32 (new) | 71.75 | 74.36 | 76.53 | 68.36 | 72.11 | 60.29 | 71.29 | v4-auto | 0.96 | 1.13 |
| q35 (new) | 30.46 | 31.47 | 31.32 | 29.14 | 28.97 | 40.12 | 16.02 | v4-intrinsics | 1.81 | 0.72 |
| q30 | 8.10 | 7.21 | 5.53 | 5.81 | 4.71 | 5.20 | 8.27 | v4-intrinsics | 0.57 | 0.91 |
| q31 | 8.77 | 7.95 | 6.52 | 6.54 | 5.55 | 4.86 | 9.62 | v4-intrinsics | 0.58 | 1.14 |
| q40 | 6.61 | 5.03 | 4.16 | 4.21 | 3.53 | 5.31 | 4.31 | v4-intrinsics | 0.82 | 0.67 |
| q41 | 6.09 | 4.55 | 3.75 | 3.75 | 3.14 | 4.96 | 3.83 | v4-intrinsics | 0.82 | 0.63 |

Against DuckDB, q32 (0.96) is level, q35 (1.81) is slower by under 2x, and q04, q15, q08 and q09 are slower by 3.7x, 3.5x, 4.9x and 2.9x.
Against corgi, q04 (0.95), q08 (0.99) and q09 (1.02) are level, q15 (1.14) and q32 (1.13) are slower by a little more than the noise band, and q35 (0.72) is faster.
This document did not read corgi's per-op profile, so it does not attribute corgi's lead on q15 and q32 or Verkehrt's lead on q35.
The q04, q08 and q09 times are the same on all five variants within 10% (q04 25.73 to 27.12, q08 48.42 to 49.66, q09 54.86 to 58.61), so the ISA level and the backend change nothing there.
q15 (5%) and q35 (9%) spread across variants inside the noise band with no consistent order. q32 spreads by 12% (68.36 to 76.53 ms), which exceeds the 10% band, and this document did not repeat the runs to tell noise from a variant effect, so it claims neither.
The four queries of the #20 round still gain from the ISA levels, with `x86-64-v4-intrinsics` the best variant and 37% to 48% below `scalar` (q30 8.10 to 4.71 ms, q31 8.77 to 5.55, q40 6.61 to 3.53, q41 6.09 to 3.14).

## Change against #20

The #20 times are the five-variant rows of `2026-10-31-clickbench-now.md`, measured in an earlier session on the same host with Verkehrt at `8c695db`.
The comparison therefore spans sessions and builds, and only changes above the 10% noise band are claimed.
The corgi column is a drift check, not part of the comparison: corgi's times for q30, q31, q40 and q41 were 5.18, 4.89, 5.36 and 4.93 ms then and are 5.20, 4.86, 5.31 and 4.96 ms now, all within 1%.

| Query | variant | #20 ms | now ms | change |
|---|---|---|---|---|
| q30 | v4-intrinsics (best) | 8.70 | 4.71 | -46% |
| q30 | scalar | 12.15 | 8.10 | -33% |
| q31 | v4-intrinsics (best) | 9.13 | 5.55 | -39% |
| q31 | scalar | 12.36 | 8.77 | -29% |
| q40 | v4-intrinsics (best) | 4.92 | 3.53 | -28% |
| q40 | scalar | 7.97 | 6.61 | -17% |
| q41 | v4-intrinsics (best) | 5.99 | 3.14 | -48% |
| q41 | scalar | 8.88 | 6.09 | -31% |

All twenty variant rows (four queries, five variants) moved by about -17% or more (the smallest is q40 `x86-64-v3-auto`, 6.03 to 5.03 ms, -16.6%), none inside the noise band.
The `argsort_i64` kernel time in `stats.txt` fell from 4495 to 1323 us in q30, from 4715 to 1916 us in q31, from 2339 to 941 us in q40, and from 3575 to 783 us in q41.
The drop in `argsort_i64` is 3172, 2799, 1398 and 2792 us, against drops in the criterion mean of 3.99, 3.58, 1.39 and 2.85 ms.
The two sets of numbers are the same size, so the sort accounts for most of the change.
The remaining difference is at most 0.8 ms (q30), and this document did not attribute it.
The kernel time outside `argsort_i64` supports the cross-session reading, because it is unchanged within noise: q30 4609 to 4544 us, q31 4738 to 4405 us, q40 2963 to 3079 us, and q41 3038 to 2983 us.
`bench-b.txt` also ran at a higher load (up to 4.42) than the #20 runs (up to 3.3), so if load slowed the runs at all, it makes the speedup a conservative reading.

Against the references, q30 (0.57) and q31 (0.58) are 1.7x faster than DuckDB, and q40 and q41 (both 0.82) are faster by 18%.
In the #20 round, q30 and q31 were level with DuckDB, and q40 and q41 were 1.14x and 1.56x slower.
Against corgi, q40 (0.67) and q41 (0.63) are faster, q30 (0.91) is level, and q31 (1.14) is slower by a little more than the noise band.

## What dominates each query

The kernel and its share come from `stats.txt`, where the share is of the summed kernel time of one warm run.
Plans are in `query-plans.txt`.

| Query | Dominating kernel | Share | Next kernels | Follow-up |
|---|---|---|---|---|
| q04 | `argsort_i64` (1 call) | 89% (22.9 of 25.7 ms) | `gather_i64` 5%, `ne_i64` 2% | Sort itself (see gaps) |
| q08 | `argsort_i64` (3 calls, two over 1M rows) | 69% (32.4 of 47.0 ms) | `gather_i64` (6 calls) 18%, `prev_i64` 3% | Composite-key grouping, hash grouping (unverified), top-k |
| q09 | `argsort_i64` (3 calls, two over 1M rows) | 59% (32.8 of 56.0 ms) | `gather_i64` (13 calls) 23%, `reduce_into_add_i64` 6% | Composite-key grouping, hash grouping (unverified), top-k |
| q15 | `argsort_i64` (2 calls) | 73% (23.2 of 31.7 ms) | `reduce_into_add_i64` 5%, `gather_i64` 4% | Hash grouping (unverified), top-k |
| q32 | `argsort_i64` (3 calls, all over 1M rows) | 53% (36.6 of 68.9 ms) | `gather_i64` (13 calls) 31%, `reduce_into_add_i64` 6% | Composite-key grouping, hash grouping (unverified), top-k |
| q35 | `argsort_i64` (2 calls) | 64% (18.5 of 28.8 ms) | `gather_i64` 11%, `reduce_into_add_i64` 5% | Hash grouping (unverified), top-k |
| q30 | none: `compress_i64` 24%, `argsort_i64` 23% | 24%, 23% | `prev_i64` 21%, `sub_i64` 16% | Spread, see below |
| q31 | `argsort_i64` | 30% | `compress_i64` 23%, `prev_i64` 16%, `sub_i64` 15% | Spread, see below |
| q40 | `eq_i64` (6 calls, five over the 1M-row filter columns) | 33% | `argsort_i64` 23%, `compress_i64` 15%, `le_i64` 11% | #16 |
| q41 | `eq_i64` (5 calls, four over the filter columns) | 30% | `argsort_i64` 21%, `compress_i64` 17%, `le_i64` 12% | #16 |

q04, q15 and q35 group by one key: one `argsort_i64` over the 1M-row column, then (q15, q35) a second sort of the group counts.
The byte counts in `stats.txt` show that this second sort is small: in q15 the two calls handle 17.3 MB together and the first alone handles 16 MB, and in q35 the two handle 17.1 MB.
q08 and q09 group by two keys (plans 8 and 11 of q08): one stable sort over `UserID` and a second over the region key `r1`, so two sorts over 1M rows, then a small sort of the per-region counts (about 20 KB of argsort bytes).
q32 sorts three times over 1M rows (`ClientIP`, `k1`, and the 1M group counts, whose `slice_i64` input is 8 MB).
Its 1,000,000 groups for 1M rows follow from the compress and slice byte counts in `stats.txt`, so every count is 1, `0 - counts` is constant, and the radix sort runs only its counting pass and an identity fill, with no scatter passes.
A release CLI probe at 1M rows measured a full 8-pass sort at 15.8 to 17.0 ms, a constant key at 2.5 to 2.8 ms, and one active pass at 2.9 to 3.2 ms.
That puts the count sort at roughly 7% of q32's argsort time and about 4% of the query, from a single probe that is not committed as raw output.
Byte share is not a sound time estimate for radix sort, because cost follows the number of active passes and not the bytes.
The `gather_i64` calls in q08, q09 and q32 carry each payload column through both permutations (plans 9, 12 and 13 of q08, and plans 12, 17 and 18 of q32), so every payload is gathered twice, and they take 8.2, 13.1 and 21.2 ms.
q08, q09 and q32 therefore attribute to composite-key grouping: one sort over a combined key instead of two, and one gather per payload instead of two.
Composing the two permutations once would also cut the gather count without a new sort, from 8 to 6 for q09 and q32, but not for q08, where `r1` is needed as the second sort key anyway.

In q40 and q41 the mask chain is now the largest block.
The kernels `eq_i64`, `le_i64` and `and_bool` that build the five-column filter take 2036 us in q40 and 1885 us in q41, which is 51% and 50% of kernel time (37% and 29% in #20), and issue #16 (mask chains) covers them.
In q30 no kernel exceeds 24%, and in q31 `argsort_i64` is the largest at 30% with `compress_i64` at 23%.
`compress_i64` (six calls, 1.4 ms each query) is the largest in q30, and `prev_i64`, `sub_i64`, `lt_i64` and `compress_i64` also serve the non-empty-phrase filter on `SearchPhrase_ends` (plan 12).
This document did not split those kernel times between the filter and the later steps, so it attributes q30 and q31 to no single follow-up.

## Gaps

* **DuckDB on the counting queries.** q04, q08, q09, q15 and q35 are 1.8x to 4.9x slower than DuckDB (q35 is 1.81x, the other four 2.9x to 4.9x), and the sort alone is slower than DuckDB's whole query (q04: `argsort_i64` 22.9 ms, DuckDB 6.91 ms). The group counts come from sorting a 1M-row key, and DuckDB most likely uses a hash table here (an inference, not checked in DuckDB), which the spec excludes from this layer. These five queries are the candidates for hash grouping. q32 is level with DuckDB (0.96) and needs nothing from it.
* **In-situ `argsort_i64` against the kernel's own scaling.** q04 spends 22.9 ms in one `argsort_i64` call on the real `UserID` column, and q15's `UserID` sort repeats the cost (q15's two calls take 23.2 ms in total). Stats overhead does not explain it, because the q04 kernel sum of 25.65 ms matches criterion's 25.73 ms. The pass count does not explain it either: the prototype input of `2026-11-01-argsort.md` (uniform `i64`, 13.5 ms per 1M rows for `lsd8`) already needs all 8 passes. The prototype is the wrong baseline, and the final kernel at 10M rows scales to about 15 to 16.6 ms per 1M rows (`final-i64-u64.txt`), so the gap is about 40%, not 70%. The cause is not measured. A fresh-process probe with fresh scratch took about 16 ms on spread 1M-row keys, which matches the 10M scaling, so first-touch page faults do not explain it. The key distribution is untested: q15's bytes imply about 80K distinct `UserID`s, about 12.5 rows per key.
* **Permutation gathers.** `gather_i64` costs 11% to 31% of q08, q09, q32 and q35, and the two-permutation payload gathers of q08, q09 and q32 are the part composite-key grouping or composing the permutations once would remove (composing cuts the gathers of q09 and q32 from 8 to 6, but not q08, which needs `r1` as the second sort key anyway). Nothing here measures how much that would save.
* **Top-k.** q15 and q35 sort all group counts to take ten of them. The byte counts say the count sort is small there (about 7% of the argsort bytes in q15 and 6% in q35), so a top-k primitive is unlikely to matter for these two on this data. q32 has 1M groups, but every count is 1, so its count sort runs no scatter passes and takes roughly 7% of its argsort time in a single probe, and a top-k primitive is unlikely to matter there either. The q15 and q35 statements come from byte counts, not timings, and byte share is not a sound time estimate for radix sort.
* **Elementwise maps before a reduction, issue #30.** The count column `add_i64 (mul_i64 gid 0) 1` is materialized before `reduce_into_add_i64` in q09, q15, q32 and q35, where `add_i64` and `mul_i64` together take 1.5% (q32) to 4.6% (q15) of kernel time.
* **Stats for the short queries.** The summed kernel time of q30, q31, q40 and q41 is 14% to 25% above the criterion mean, so their shares are rough.
* **Output checks.** This round timed the programs and did not re-check their output against `expected/`. Correctness rests on the acceptance tests of the commits that added the queries.
