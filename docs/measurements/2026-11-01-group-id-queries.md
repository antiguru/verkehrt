# Hash-grouped ClickBench queries: Measurements

This document times the six ClickBench queries that group by `group_id` (q04, q08, q09, q15, q32, q35) against their sort-based versions, corgi and DuckDB on one thread.
It also tries `group_id` on q30, q40 and q41 and keeps it on q41, and decides on q31 from its kernel counts.
q04, q08, q09, q15 and q35 are 2.8x to 4.5x faster with `group_id` than with sorting on `x86-64-v4-intrinsics` (q04 26.0 to 5.7 ms), and q32 is 1.24x slower, so q32 is reverted to its sort-based version.
Of the other four queries, q41 is kept as `group_id`: it is 5.4% to 8.8% faster in all fifteen cells of three interleaved passes of longer runs, against a spread across passes of at most 2.4%, which meets the rule that `group_id` stays when it is faster beyond the run-to-run spread on most variants in every pass. q30 is up to 11% faster with a spread as large as the effect in one pass, and q40 is 8% to 16% slower, so both stay sort-based, and q31 is left sort-based without a trial.
Percentage changes are time reductions, 1 - new/old, with "new" the `group_id` time and "old" the sort-based time, so 10% faster means the hash time is 0.90 of the sort time. The q32 and q40 comparisons use hash/sort time ratios instead.
The sort and hash versions run in one session on one build, and the last sections list the gaps.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt | this branch after the one-pass group count, plus `VERKEHRT_PROGRAMS`, `VERKEHRT_BENCH_SAMPLES` and `VERKEHRT_BENCH_SECS` overrides in `benches/acceptance.rs` |
| Sort-based programs | `tests/acceptance/clickbench/qNN.vk` as #33 leaves them, before this branch moves any query to `group_id`. The first runs read them from a scratch directory through an earlier, flat form of `VERKEHRT_PROGRAMS`, and the long runs read the repository files |
| Tried `group_id` programs | `2026-11-01-group-id-queries/programs/clickbench/q30.vk`, `q40.vk`, `q41.vk`, read by the long runs through `VERKEHRT_PROGRAMS` |
| corgi revision | `2751e04e8501d47b33219be23049bf4afa7a4428`, checkout `~/dev/repos/data/verkehrt/corgi-wip`, the same as in `2026-11-01-clickbench-groups.md` |
| DuckDB | one-thread times from `duckdb.tsv` of the prepared data directory |
| Data | `hits_0.parquet`, 1,000,000 rows, the data directory of the earlier rounds |

The method is that of `2026-11-01-clickbench-groups.md`: criterion with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time on `Prepared::execute` only, corgi's best of at least 3 runs, and the `one_thread_ms` column of DuckDB.
`VERKEHRT_PROGRAMS=DIR` makes the acceptance bench read the programs of each suite from `DIR/SUITE`, so two versions of a query run in the same build and session without a second copy under `tests/acceptance/`.
`VERKEHRT_BENCH_SAMPLES` and `VERKEHRT_BENCH_SECS` replace the sample size and the measurement time, and the long runs below use 30 samples and 10 s against the 10 samples and 3 s of the other runs.
Differences within about 10% count as noise, and a change is a time reduction, 1 - new/old.
Every table row in this document is a criterion mean in ms, from the files listed below, and a "best" column is the minimum over all runs and variants of that form.
Other work shares the host, so each run group records `/proc/loadavg` at its start and end, and the load ranged from 1.69 to 2.90 over all groups.

### Evidence files

The raw output is in `docs/measurements/2026-11-01-group-id-queries/`.

* `bench-hash.txt`: criterion log for the `group_id` versions of q04, q08, q09, q15, q32, q35. Load 1.69 at the start, 2.90 at the end (this run includes the build).
* `bench-sort.txt`: criterion log for the sort-based versions of the same six. Load 2.90 to 2.41.
* `bench-sortbound.txt`: criterion log for q30, q31, q40, q41 as they stood before this task, all sort-based. Load 2.41 to 2.35.
* `bench-try-hash.txt`, `bench-try-sort.txt`: q30, q40, q41 as `group_id` and as sort-based, run back to back. Load 1.87 to 2.33.
* `bench-try-hash-2.txt`, `bench-try-sort-2.txt`: the repeat for q30 and q41. Load 2.23 to 2.39.
* `long-hash-N.txt`, `long-sort-N.txt` (N = 1 to 3): q30 and q41 as `group_id` and as sort-based with 30 samples and 10 s, interleaved in the order hash 1, sort 1, hash 2, sort 2, hash 3, sort 3. Load at the start and end of each file: 1.64 to 4.89 across the six, the highest at the end of `long-hash-1.txt` (4.89) and in the passes 3 (4.40, 4.01).
* `corgi.txt`: corgi's bench for the ten queries. Load 2.12 to 2.10.
* `stats.txt`: per-kernel totals. q04, q08, q09, q15, q32, q35 are the `group_id` versions, q30, q31, q40, q41 the sort-based versions, and the last two sections are the sort-based q32 and q04 run by path. Load 2.26 to 2.40.
* `stats-q41-hash.txt`: per-kernel totals of the tried `group_id` q41. Load 2.09.

The `group_id` versions of q30, q40 and q41 were tried, q30 and q40 were reverted and q41 was kept, and `programs/clickbench/` in the evidence directory holds them, so the timings can be reproduced with `VERKEHRT_PROGRAMS` set to the `programs` directory. The sort-based versions are the programs of #33 under `tests/acceptance/clickbench/`.

## Results

Times are criterion means in ms on `x86-64-v4-intrinsics`, the variant the CLI picks, and "best" is the lowest mean over the five variants (scalar, `x86-64-v3` and `x86-64-v4`, each auto and intrinsics where they apply).
Ratios below 1.0 mean Verkehrt is faster, and the ratios use the `x86-64-v4-intrinsics` time of the kept form.

| Query | hash v4-intr | hash best | sort v4-intr | sort best | hash / sort (v4-intr) | corgi | DuckDB 1t | kept | kept / corgi | kept / DuckDB |
|---|---|---|---|---|---|---|---|---|---|---|
| q04 | 5.74 | 5.74 | 26.01 | 25.16 | 0.22 | 26.77 | 6.91 | hash | 0.21 | 0.83 |
| q08 | 15.23 | 15.23 | 48.67 | 47.61 | 0.31 | 49.19 | 9.85 | hash | 0.31 | 1.55 |
| q09 | 20.37 | 20.29 | 57.34 | 53.80 | 0.36 | 52.76 | 19.09 | hash | 0.39 | 1.07 |
| q15 | 11.02 | 11.01 | 34.47 | 32.22 | 0.32 | 28.46 | 9.22 | hash | 0.39 | 1.20 |
| q35 | 10.63 | 10.60 | 29.46 | 27.90 | 0.36 | 39.15 | 16.02 | hash | 0.27 | 0.66 |
| q32 | 92.77 | 89.42 | 74.95 | 65.07 | 1.24 | 60.45 | 71.29 | sort | 1.24 | 1.05 |
| q41 | 2.80 | 2.80 | 3.18 | 3.10 | 0.88 | 5.03 | 3.83 | hash | 0.63 | 0.83 |
| q30 | 5.34 | 5.27 | 5.95 | 5.88 | 0.90 | 5.29 | 8.27 | sort | 1.12 | 0.72 |
| q40 | 4.16 | 4.16 | 3.59 | 3.55 | 1.16 | 5.30 | 4.31 | sort | 0.68 | 0.83 |
| q31 | not tried | not tried | 6.86 | 6.86 | n/a | 4.93 | 9.62 | sort | 1.39 | 0.71 |

The q41 row is the kept version (`group_id`), and the q41 Kept column ignores the longer runs, which are the basis of the decision below. The q41, q30 and q40 hash and sort columns come from `bench-try-*` (the second run for q41 and q30, the first for q40).
The q31 sort time comes from `bench-sortbound.txt`, and the q30 and q41 sort-based times are 5.88 and 3.10 there, within 1% of the `bench-try-sort` values.
q09's sort-based `scalar` mean (53.80) is below its other variants (55.66 to 57.77), which is spread inside the noise band, so the sort best for q09 is the least meaningful number in the table.
q32's sort-based variants spread by 15% (65.07 to 75.14), more than the noise band, and this document did not repeat the runs to tell noise from a variant effect.
The corgi and DuckDB columns are best-of times, so they favor those systems slightly against criterion's mean.

### Five variants

Criterion means in ms, in the order scalar, `v3-auto`, `v3-intrinsics`, `v4-auto`, `v4-intrinsics`.

| Query | form | scalar | v3-auto | v3-intr | v4-auto | v4-intr |
|---|---|---|---|---|---|---|
| q04 | hash | 6.57 | 5.77 | 5.77 | 5.75 | 5.74 |
| q04 | sort | 26.88 | 25.16 | 25.50 | 25.75 | 26.01 |
| q08 | hash | 17.62 | 15.52 | 15.55 | 16.09 | 15.23 |
| q08 | sort | 50.18 | 47.61 | 48.48 | 47.92 | 48.67 |
| q09 | hash | 21.93 | 20.66 | 20.68 | 20.29 | 20.37 |
| q09 | sort | 53.80 | 55.66 | 57.77 | 56.82 | 57.34 |
| q15 | hash | 12.05 | 11.01 | 11.01 | 11.01 | 11.02 |
| q15 | sort | 34.45 | 33.10 | 33.08 | 32.22 | 34.47 |
| q35 | hash | 11.55 | 10.60 | 10.60 | 10.66 | 10.63 |
| q35 | sort | 29.32 | 27.90 | 27.92 | 29.47 | 29.46 |
| q32 | hash | 90.85 | 90.27 | 89.42 | 92.51 | 92.77 |
| q32 | sort | 75.14 | 65.07 | 70.09 | 70.16 | 74.95 |
| q41 | hash (run 1, run 2) | 5.60, 5.64 | 3.87, 3.87 | 3.51, 3.48 | 3.42, 3.40 | 2.84, 2.80 |
| q41 | sort (run 1, run 2) | 6.03, 6.04 | 4.17, 4.21 | 3.75, 3.77 | 3.78, 3.75 | 3.17, 3.18 |
| q30 | hash (run 1, run 2) | 8.43, 8.90 | 7.16, 6.76 | 5.99, 6.00 | 6.37, 6.41 | 5.27, 5.34 |
| q30 | sort (run 1, run 2) | 9.41, 9.14 | 7.10, 7.49 | 6.65, 6.68 | 7.21, 7.14 | 5.88, 5.95 |
| q40 | hash | 7.14 | 5.12 | 4.72 | 4.77 | 4.16 |
| q40 | sort | 6.59 | 4.57 | 4.16 | 4.24 | 3.59 |

The five queries that keep `group_id` (q04, q08, q09, q15, q35) are 2.4x to 4.5x faster on every variant (the smallest ratio is q09 at `scalar`, 21.93 against 53.80 ms, 0.41), far outside the noise band.
In the hash versions of these five, the spread across variants is 14% for q04, 16% for q08, 8% for q09, 9% for q15 and 9% for q35, and in q04, q15 and q35 almost all of it is between `scalar` and the rest.

## Decisions

### q32: Reverted

q32 groups 1,000,000 rows into about 1,000,000 distinct (WatchID, ClientIP) pairs, and the hash version is slower than the sort version on all five variants: 89.4 to 92.8 ms against 65.1 to 75.1 ms, and 1.24x slower on `x86-64-v4-intrinsics`.
The cause is in `stats.txt`.
The hash version spends 36.0 ms in `group_id2_i64_i64` and still runs three `argsort_i64` calls over the 1M groups (35.6 ms) and 9 `gather_i64` calls (7.7 ms), summing to 86.2 ms.
The sort version spends 38.3 ms in three `argsort_i64` calls and 21.4 ms in 13 `gather_i64` calls, summing to 70.8 ms.
Grouping by hash saves 13.7 ms in `gather_i64`, 2.7 ms in `argsort_i64` and 4.2 ms in other kernels, 20.6 ms in total, and the 36.0 ms hash pass on mostly distinct keys costs more than that, which matches `docs/measurements/2026-11-01-group-id.md` (slower than sorting on mostly distinct keys at 1M rows).
The file is restored to the exact version of #33, and the diff against it is empty. The same holds for q30, q40 and q41.

### q30 and q41: Longer runs

The first runs (10 samples, 3 s) put `group_id` at 10.4% to 11.9% faster on `x86-64-v4-intrinsics` for q41 and 10.4% for q30, and between 1% slower and 12% faster on the other variants of q30 and 6% to 9.5% faster on those of q41, which is at the edge of the 10% band.
Per-variant percentages move by a point or two between metrics, so the decision rests on a longer measurement instead.
The long runs are three interleaved passes of hash and sort (30 samples, 10 s per variant), and the rule is to keep `group_id` only if it is faster beyond the run-to-run spread on most variants in every pass.
The tables give criterion means in ms and the time reduction of each pass.

q41:

| variant | hash 1 | sort 1 | red. 1 | hash 2 | sort 2 | red. 2 | hash 3 | sort 3 | red. 3 |
|---|---|---|---|---|---|---|---|---|---|
| scalar | 5.709 | 6.082 | 6.1% | 5.740 | 6.067 | 5.4% | 5.739 | 6.088 | 5.7% |
| v3-auto | 4.207 | 4.542 | 7.4% | 4.259 | 4.555 | 6.5% | 4.219 | 4.561 | 7.5% |
| v3-intrinsics | 3.455 | 3.755 | 8.0% | 3.499 | 3.764 | 7.0% | 3.495 | 3.764 | 7.2% |
| v4-auto | 3.453 | 3.768 | 8.4% | 3.493 | 3.813 | 8.4% | 3.478 | 3.804 | 8.6% |
| v4-intrinsics | 2.931 | 3.126 | 6.2% | 2.933 | 3.186 | 7.9% | 2.921 | 3.201 | 8.8% |

q30:

| variant | hash 1 | sort 1 | red. 1 | hash 2 | sort 2 | red. 2 | hash 3 | sort 3 | red. 3 |
|---|---|---|---|---|---|---|---|---|---|
| scalar | 8.513 | 9.494 | 10.3% | 8.969 | 8.919 | -0.6% | 8.943 | 9.475 | 5.6% |
| v3-auto | 7.503 | 8.330 | 9.9% | 7.449 | 8.333 | 10.6% | 7.912 | 8.365 | 5.4% |
| v3-intrinsics | 6.048 | 6.828 | 11.4% | 6.080 | 6.842 | 11.1% | 6.487 | 6.852 | 5.3% |
| v4-auto | 6.321 | 7.095 | 10.9% | 6.324 | 7.030 | 10.0% | 6.738 | 7.129 | 5.5% |
| v4-intrinsics | 5.284 | 5.780 | 8.6% | 5.232 | 5.783 | 9.5% | 5.254 | 5.894 | 10.9% |

q41 is faster with `group_id` in all fifteen cells, by 5.4% to 8.8%.
Its spread across the three passes is at most 1.3% for the hash version and 2.4% for the sort version, so the gain is beyond the run-to-run spread in every cell, and it saves 0.2 to 0.4 ms of a 3 to 6 ms query.
q30 is 10% or more faster in seven of fifteen cells (three variants in each of passes 1 and 2, and v4-intrinsics in pass 3), and 5.3% to 5.6% on the other four variants in pass 3.
The hash version's spread across passes is 5% to 7% on four variants, as large as the effect in pass 3, so q30 is not faster beyond the run-to-run spread on most variants in every pass.
q41 therefore is kept as `group_id` under the rule, and `tests/acceptance/clickbench/q41.vk` is the tried version from `programs/clickbench/q41.vk`. q30 stays sort-based, because its gain is not beyond the spread in every pass, and its file is the version of #33.
For q41 the kernel totals show where the gain comes from: `argsort_i64` takes 129 us in the hash version (three calls over 336 KB) against 743 us in the sort version, and `group_id2_i64_i64` costs 494 us over 56,737 rows with about 7,000 groups.
The total kernel time falls from 3842 to 3354 us (13%), and the mask chain, which this change does not touch, is 54% of the remaining time.
q30 filters 1M rows to about 69,000 and has about 22,800 groups (from the `slice_i64` bytes in `stats.txt`), so the key sorts shrink and the group sort over 22,800 groups remains.

### q40: Reverted

The `group_id` q40 is slower on all five variants (one run, 10 samples and 3 s), by 15.9% on `x86-64-v4-intrinsics` (4.16 against 3.59 ms) and by 8.3%, 12.0%, 13.4% and 12.5% on scalar, `v3-auto`, `v3-intrinsics` and `v4-auto`, measured as the hash time over the sort time minus one.
q40 filters 1M rows to about 50,000 and has about 23,600 groups, nearly half as many groups as rows, and `group_id` also leaves the two group-key sorts in place.

### q31: Not tried

q31 groups about 69,350 filtered rows into about 69,360 slice bytes of groups (`slice_i64 bytes=554912` in the sort-based `stats.txt`, 69,364 elements for an `order` that is also `ng` long), so nearly every row is its own group.
`group_id` does not shrink the sorts in that case, because the group sort over `ng` is the same size as the row sort, and `docs/measurements/2026-11-01-group-id.md` shows `group_id` slower than sorting on mostly distinct keys.
q31 therefore stays sort-based on the evidence of q32 and the group-id document, and this document did not time a `group_id` q31.

## What dominates each query

The kernel and its share come from `stats.txt`, where the share is of the summed kernel time of one warm run of the kept form.
The summed kernel time is within 7% of the criterion mean on `x86-64-v4-intrinsics` for the six long queries (q04 +5%, q08 -2%, q09 -6%, q15 +1%, q35 +1%, sort-based q32 -5%).

| Query | Kept | Dominating kernel | Share | Next kernels | Follow-up |
|---|---|---|---|---|---|
| q04 | hash | `group_id_i64` | 98% (5.90 of 6.02 ms) | `reduce_max_i64` 2% | The hash pass itself, a named gap |
| q08 | hash | `group_id2_i64_i64` | 53% (7.87 of 14.92 ms) | `reduce_into_min_i64` 30%, `group_id_i64` 14% | Key recovery by minimum reduction, a named gap |
| q09 | hash | `group_id2_i64_i64` | 38% (7.38 of 19.19 ms) | `reduce_into_min_i64` 24%, `reduce_into_add_i64` 16%, `group_id_i64` 11% | Key recovery by minimum reduction and fused aggregation, a named gap |
| q15 | hash | `group_id_i64` | 53% (5.91 of 11.14 ms) | `reduce_into_min_i64` 20%, `argsort_i64` 13%, `histogram` 12% | Key recovery by minimum reduction, a named gap |
| q35 | hash | `group_id_i64` | 54% (5.80 of 10.76 ms) | `reduce_into_min_i64` 21%, `histogram` 12%, `argsort_i64` 10% | Key recovery by minimum reduction, a named gap |
| q32 | sort | `argsort_i64` (3 calls over 1M rows) | 54% (38.3 of 70.8 ms) | `gather_i64` (13 calls) 30%, `reduce_into_add_i64` 5% | Composite-key grouping, top-k, and hash grouping at this group count, see below |
| q41 | hash | `eq_i64`, `le_i64`, `and_bool` (mask chain) | 51% (1.97 of 3.84 ms) | `argsort_i64` 19%, `compress_i64` 17% | #16 (shares are of the sort-based version in `stats.txt`) |
| q40 | sort | `eq_i64`, `le_i64`, `and_bool` (mask chain) | 50% (1.99 of 4.00 ms) | `argsort_i64` 24%, `compress_i64` 15% | #16 |
| q30 | sort | none: `compress_i64` 24%, `argsort_i64` 24% | 24%, 24% | `sub_i64` 18%, `prev_i64` 17% | A named gap, see below |
| q31 | sort | `argsort_i64` | 31% (2.0 of 6.4 ms) | `compress_i64` 22%, `prev_i64` 16%, `sub_i64` 15% | Composite keys, top-k, hash grouping |

In q08, q09, q15 and q35 the `reduce_into_min_i64` calls recover each group's key from the row column, 2.3 ms per call over 1M rows, and take 20% to 30% of kernel time.
The `group_id` table already holds each key, but the primitive returns ids only, so the program reads the key again from the rows.
This document did not measure what a primitive that also returns the first row of each group would save, so the follow-up is a gap and not a claim.
The elementwise map before a reduction of issue #30 no longer appears in these programs, because `histogram` replaced the `add_i64 (mul_i64 gid 0) 1` count column.
q32 keeps the sort-based form, so its row is the sort version's, and the `argsort_i64` and `gather_i64` calls there are the payload gathers through two permutations that composite-key grouping would remove.
This document does not measure what composite-key grouping or top-k would save in q32, and the hash version shows that the three sorts over 1M groups stay when only the grouping step changes.
In q30, no kernel exceeds 24%, and the filter on `SearchPhrase_ends` (`prev_i64`, `sub_i64`, `lt_i64` and then `compress_i64` of four columns) take 37% of kernel time together (2.03 of 5.48 ms in `stats.txt`), and the byte counts put most of it in the 1M-row calls (16 of the 18.2 MB of `prev_i64`).
This document did not split the `compress_i64` time between the filter and later steps and does not name an issue for the filter.

## Change against the earlier round

The sort-based times of `2026-11-01-clickbench-groups.md` (the same host, an earlier session) and the sort-based times here agree within 10% for q04, q08, q15 and q35 on `x86-64-v4-intrinsics` (25.73 against 26.01, 48.42 against 48.67, 32.57 against 34.47, 28.97 against 29.46), and q09 and q32 differ by 4.5% (54.86 against 57.34) and 4% (72.11 against 74.95).
corgi's times differ from that round by at most 2.4% (q35 40.12 to 39.15, q09 53.81 to 52.76, q32 60.29 to 60.45), so the host did not drift visibly.
The q30, q31, q40 and q41 sort-based times are 5.88 to 5.95, 6.86, 3.55 to 3.59 and 3.10 to 3.18 ms, and the earlier round had 4.71, 5.55, 3.53 and 3.14.
q30 and q31 are 25% and 24% slower than the earlier round, q40 and q41 are level, and this document did not find the cause.
The q30 and q31 kernel sums in `stats.txt` (5.48 and 6.41 ms) are close to those of the earlier `stats.txt` (5.87 and 6.32 ms), so the stats runs do not show the slowdown that criterion does.
The ratios between the sort-based and hash-based forms use only data from one session and are not affected by this.

## Gaps

* **q32 and mostly distinct keys.** q32 gains nothing from `group_id` at 1M groups, and the sort of the groups by key and count stays at 35.6 ms in either form. A top-k that avoids sorting 1M groups to take ten might change that, and this document did not test it.
* **Key recovery.** `reduce_into_min_i64` takes 20% to 30% of kernel time in q08, q09, q15 and q35 to re-read keys the hash table already holds. The saving from a primitive that returns a representative row per group is not measured.
* **Hash pass cost.** `group_id` is 98% of q04 and 53% to 54% of q15 and q35, so the table build is the next cost. The `docs/measurements/2026-11-01-group-id.md` document owns that kernel, and this one does not look inside it.
* **DuckDB on the many-group queries.** q08 (1.55x) and q15 (1.20x) are slower than DuckDB and q09 is level (1.07x). q04 (0.83) and q35 (0.66) are faster, and q32 (1.05 on `v4-intrinsics`, 0.91 on the best sort variant) is level. This document did not look at why DuckDB leads on q08 and q15.
* **Against corgi.** q32 (1.24 on `v4-intrinsics`) and q30 (1.12) and q31 (1.39) are slower than corgi, and this document did not read corgi's per-op profile.
* **q30 below the threshold.** q30 is up to 11% faster with `group_id` with a spread across passes as large as the effect, so it stays sort-based. q41 is kept as `group_id` and its numbers come from the longer runs only.
* **q30 and q31 slowdown.** Both are 24% to 25% above the earlier round on the same variant, and the cause is not measured.
* **Variant spread in q32.** The sort-based q32 spreads by 15% across variants and was not repeated.
* **Output checks.** The six queries pass the acceptance tests (`cargo test --release`, 17 acceptance tests pass, including `clickbench_matches_duckdb`) after the reverts, and q41 is the `group_id` version. The `group_id` q30 and q40 were checked against the same suite before timing and were then reverted.
