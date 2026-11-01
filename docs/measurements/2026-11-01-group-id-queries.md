# Hash-grouped ClickBench queries: Measurements

This document times the six ClickBench queries that group by `group_id` (q04, q08, q09, q15, q32, q35) against their sort-based versions, corgi and DuckDB on one thread.
It also tries `group_id` on q30, q40 and q41, and decides on q31 from its kernel counts.
q04, q08, q09, q15 and q35 are 2.8x to 4.5x faster with `group_id` than with sorting on `x86-64-v4-intrinsics` (q04 26.0 to 5.7 ms), and q32 is 1.24x slower, so q32 is reverted to its sort-based version.
Of the other four queries, `group_id` is 10% to 12% faster on q30 and q41 on the default variant only, which is at the edge of the noise band, and 16% slower on q40, so all three stay sort-based, and q31 is left sort-based without a trial.
The sort and hash versions run in one session on one build, and the last sections list the gaps.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt | branch `grouping/hashed` at `bc70ec7`, plus a `VERKEHRT_PROGRAMS` override in `benches/acceptance.rs` |
| Sort-based programs | `tests/acceptance/clickbench/qNN.vk` of `bb01630`, run from a directory outside the repository through `VERKEHRT_PROGRAMS` |
| corgi revision | `2751e04e8501d47b33219be23049bf4afa7a4428`, checkout `~/dev/repos/data/verkehrt/corgi-wip`, the same as in `2026-11-01-clickbench-groups.md` |
| DuckDB | one-thread times from `duckdb.tsv` of the prepared data directory |
| Data | `hits_0.parquet`, 1,000,000 rows, the data directory of the earlier rounds |

The method is that of `2026-11-01-clickbench-groups.md`: criterion with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time on `Prepared::execute` only, corgi's best of at least 3 runs, and the `one_thread_ms` column of DuckDB.
`VERKEHRT_PROGRAMS` makes the acceptance bench read its `.vk` files from another directory, so the sort-based versions run in the same build and session as the hash-based ones without a second copy in the repository.
Differences within about 10% count as noise.
Every table row in this document is a criterion mean in ms, from the files listed below.
Other work shares the host, so each run group records `/proc/loadavg` at its start and end, and the load ranged from 1.69 to 2.90 over all groups.

### Evidence files

The raw output is in `docs/measurements/2026-11-01-group-id-queries/`.

* `bench-hash.txt`: criterion log for the `group_id` versions of q04, q08, q09, q15, q32, q35. Load 1.69 at the start, 2.90 at the end (this run includes the build).
* `bench-sort.txt`: criterion log for the sort-based versions of the same six. Load 2.90 to 2.41.
* `bench-sortbound.txt`: criterion log for q30, q31, q40, q41 as they stood before this task, all sort-based. Load 2.41 to 2.35.
* `bench-try-hash.txt`, `bench-try-sort.txt`: q30, q40, q41 as `group_id` and as sort-based, run back to back. Load 1.87 to 2.33.
* `bench-try-hash-2.txt`, `bench-try-sort-2.txt`: the repeat for q30 and q41. Load 2.23 to 2.39.
* `corgi.txt`: corgi's bench for the ten queries. Load 2.12 to 2.10.
* `stats.txt`: per-kernel totals. q04, q08, q09, q15, q32, q35 are the `group_id` versions, q30, q31, q40, q41 the sort-based versions, and the last two sections are the sort-based q32 and q04 run by path. Load 2.26 to 2.40.
* `stats-q41-hash.txt`: per-kernel totals of the tried `group_id` q41. Load 2.09.

The `group_id` versions of q30, q40 and q41 were tried and reverted, so no file in the repository holds them. The sort-based versions of the queries from `bb01630` are the ones the repository holds.

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
| q41 | 2.80 | 2.80 | 3.18 | 3.10 | 0.88 | 5.03 | 3.83 | sort | 0.63 | 0.83 |
| q30 | 5.34 | 5.27 | 5.95 | 5.88 | 0.90 | 5.29 | 8.27 | sort | 1.12 | 0.72 |
| q40 | 4.16 | 4.16 | 3.59 | 3.55 | 1.16 | 5.30 | 4.31 | sort | 0.68 | 0.83 |
| q31 | not tried | not tried | 6.86 | 6.86 | n/a | 4.93 | 9.62 | sort | 1.39 | 0.71 |

The q41, q30 and q40 hash and sort columns come from `bench-try-*` (the second run for q41 and q30, the first for q40).
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

### q32: reverted

q32 groups 1,000,000 rows into about 1,000,000 distinct (WatchID, ClientIP) pairs, and the hash version is slower than the sort version on all five variants: 89.4 to 92.8 ms against 65.1 to 75.1 ms, and 1.24x slower on `x86-64-v4-intrinsics`.
The cause is in `stats.txt`.
The hash version spends 36.0 ms in `group_id2_i64_i64` and still runs three `argsort_i64` calls over the 1M groups (35.6 ms) and 9 `gather_i64` calls (7.7 ms), summing to 86.2 ms.
The sort version spends 38.3 ms in three `argsort_i64` calls and 21.4 ms in 13 `gather_i64` calls, summing to 70.8 ms.
Grouping by hash saves 14 ms of gathers and 4 ms in other kernels, and the 36 ms hash pass on mostly distinct keys costs more than that, which matches `docs/measurements/2026-11-01-group-id.md` (slower than sorting on mostly distinct keys at 1M rows).
The file is restored to the exact `bb01630` version, and the diff against `bb01630` is empty. The same holds for q30, q40 and q41.

### q41 and q30: reverted at the edge of the band

The `group_id` q41 is 10.4% and 11.9% faster than the sort version on `x86-64-v4-intrinsics` in two runs (2.84 against 3.17 ms, 2.80 against 3.18 ms), and the other variants are 6% to 9% faster (for example `v4-auto` 3.42 against 3.78 and 3.40 against 3.75 ms), inside the band.
The `group_id` q30 is 10.4% faster on `x86-64-v4-intrinsics` in both runs (5.27 against 5.88 ms, 5.34 against 5.95 ms), and across the other variants it ranges from 1% slower to 12% faster (`v3-auto` 7.16 against 7.10 ms in run 1 and 6.76 against 7.49 ms in run 2).
Both clear 10% on one variant by 0.4 to 1.9 points and neither clears it on the rest, so neither is faster beyond noise, and both are reverted to the `bb01630` form.
The saving is 0.3 to 0.6 ms of a 3 to 6 ms query.
For q41 the kernel totals show where the gain comes from: `argsort_i64` takes 129 us in the hash version (three calls over 336 KB) against 743 us in the sort version, and `group_id2_i64_i64` costs 494 us over 56,737 rows with about 7,000 groups.
The total kernel time falls from 3842 to 3354 us (13%), and the mask chain, which this change does not touch, is 54% of the remaining time.
q30 filters 1M rows to about 69,000 and has about 22,800 groups (from the `slice_i64` bytes in `stats.txt`), so the key sorts shrink and the group sort over 22,800 groups remains.
More runs or a longer measurement time would settle whether the 10% to 12% on the default variant is real, and this document did not run them.

### q40: reverted

The `group_id` q40 is slower on all five variants, 16% on `x86-64-v4-intrinsics` (4.16 against 3.59 ms) and 8% to 12% on the others.
q40 filters 1M rows to about 50,000 and has about 23,600 groups, nearly half as many groups as rows, and `group_id` also leaves the two group-key sorts in place.

### q31: not tried

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
| q41 | sort | `eq_i64`, `le_i64`, `and_bool` (mask chain) | 51% (1.97 of 3.84 ms) | `argsort_i64` 19%, `compress_i64` 17% | #16 |
| q40 | sort | `eq_i64`, `le_i64`, `and_bool` (mask chain) | 50% (1.99 of 4.00 ms) | `argsort_i64` 24%, `compress_i64` 15% | #16 |
| q30 | sort | none: `compress_i64` 24%, `argsort_i64` 24% | 24%, 24% | `sub_i64` 18%, `prev_i64` 17% | A named gap, see below |
| q31 | sort | `argsort_i64` | 31% (2.0 of 6.4 ms) | `compress_i64` 22%, `prev_i64` 16%, `sub_i64` 15% | Composite keys, top-k, hash grouping |

In q08, q09, q15 and q35 the `reduce_into_min_i64` calls recover each group's key from the row column, 2.3 ms per call over 1M rows, and take 20% to 30% of kernel time.
The `group_id` table already holds each key, but the primitive returns ids only, so the program reads the key again from the rows.
This document did not measure what a primitive that also returns the first row of each group would save, so the follow-up is a gap and not a claim.
The elementwise map before a reduction of issue #30 no longer appears in these programs, because `histogram` replaced the `add_i64 (mul_i64 gid 0) 1` count column.
q32 keeps the sort-based form, so its row is the sort version's, and the `argsort_i64` and `gather_i64` calls there are the payload gathers through two permutations that composite-key grouping would remove.
This document does not measure what composite-key grouping or top-k would save in q32, and the hash version shows that the three sorts over 1M groups stay when only the grouping step changes.
In q30, no kernel exceeds 24%, and the filter on `SearchPhrase_ends` (`prev_i64`, `sub_i64`, `lt_i64` and then `compress_i64` of four columns) adds up to roughly 37% of kernel time before the first sort.
This document did not split the `compress_i64` time between the filter and later steps and does not name an issue for the filter.

## Change against the earlier round

The sort-based times of `2026-11-01-clickbench-groups.md` (the same host, an earlier session) and the sort-based times here agree within 10% for q04, q08, q15 and q35 on `x86-64-v4-intrinsics` (25.73 against 26.01, 48.42 against 48.67, 32.57 against 34.47, 28.97 against 29.46), and q09 and q32 differ by 4.5% (54.86 against 57.34) and 4% (72.11 against 74.95).
corgi's times differ from that round by at most 2.2% (q09 53.81 to 52.76, q35 40.12 to 39.15, q32 60.29 to 60.45), so the host did not drift visibly.
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
* **q30 and q41 at the edge.** `group_id` is 10% to 12% faster on the default variant only, and more runs or a longer measurement time would settle whether it is real.
* **q30 and q31 slowdown.** Both are 24% to 25% above the earlier round on the same variant, and the cause is not measured.
* **Variant spread in q32.** The sort-based q32 spreads by 15% across variants and was not repeated.
* **Output checks.** The six queries pass the acceptance tests (`cargo test --release`, 17 acceptance tests pass, including `clickbench_matches_duckdb`) after the reverts. The `group_id` q30, q40 and q41 were checked against the same suite before timing and were then reverted.
