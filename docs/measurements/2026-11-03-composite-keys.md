# Two-column argsort and top-k: Measurements

This document measures five ClickBench queries rewritten to use the two-column `argsort(a, b)` and the fused top-k kernels, against their programs before the rewrite.
q15 and q35 take their ten best groups with `slice(argsort(0 - counts, gk), 0, min(ng, 10))` and are 6.1% to 9.3% faster in all thirty cells of three interleaved passes, against a spread across passes of at most 3.3%, so both rewrites are kept.
q32, q31 and q30 group by `argsort(major, minor)` and one gather set instead of two sorts and two gather sets, and none meets the keep rule, so their files return to the version before the rewrite.
q32 is 3.7% to 10.3% faster on the means of three passes, but its third pass gains less than the spread of the old runs, q30 gains 1.7% to 4.0% on the means, inside its spread, and q31 is 2.0% to 7.5% slower.
The in-situ cost of the composite kernel on q32 is 41.1 ms against 34.3 ms for the two single-key sorts it replaces, even though its image fits the 16-byte record and it makes 12 passes instead of 16, and the cause is not established.
Percentage changes are time reductions, 1 - new/old, with "new" the rewritten query and "old" the program before the rewrite, and the ratios in the tables are new/old.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 2 to 3 GHz (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt | the composite-keys branch after the top-k kernels, one build for every run |
| Old programs | `2026-11-03-composite-keys/programs-before/clickbench/`, byte copies of the five files before the rewrite, read through `VERKEHRT_PROGRAMS` |
| New programs | `2026-11-03-composite-keys/programs/clickbench/q15.vk` and `q35.vk` (kept) and `programs-rejected/clickbench/q30.vk`, `q31.vk`, `q32.vk` (reverted) |
| corgi and DuckDB | the one-thread figures of `2026-11-01-group-id-queries.md`, taken on another day |
| Data | `hits_0.parquet`, 1,000,000 rows, the data directory of the earlier rounds |

The method is that of `2026-11-01-group-id-queries.md`: criterion on `Prepared::execute` only, with `VERKEHRT_BENCH_SAMPLES=30` and `VERKEHRT_BENCH_SECS=10`, five variants per query, and old and new in interleaved passes (old, new, new, old, old, new).
Every bench file covers all five queries, so q31 and q30 also ran in three passes instead of the one pair of the plan.
`VERKEHRT_PROGRAMS=DIR` makes the bench and, since this task, `examples/acceptance_stats` read the programs of a suite from `DIR/SUITE`, so before and after run in one build.
The example could not read other programs before, and the old-form statistics need it.
Differences within about 10% count as noise, and a rewrite is kept only if it is faster beyond the run-to-run spread on most variants in every pass.
Other work shares the host, and its load swung between 2.5 and 60 during this task, so every file records `/proc/loadavg` at its start and end and a run that finished under high load was discarded and repeated.

### Evidence files

The raw output is in `docs/measurements/2026-11-03-composite-keys/`.

* `bench-old-1.txt`, `bench-new-1.txt`, `bench-new-2.txt`, `bench-old-2.txt`, `bench-old-3.txt`, `bench-new-3.txt`: criterion logs, in the order of the interleaved passes (old 1, new 1, new 2, old 2, old 3, new 3). Load at the start and end: old 1 2.14 to 2.62, new 1 2.57 to 5.48, new 2 3.64 to 4.04, old 2 2.70 to 7.80, old 3 5.60 to 4.00, new 3 3.76 to 5.62.
* `discarded-bench-old-2-load-14.txt`, `discarded-bench-old-2-load-43.txt`, `discarded-bench-old-2-load-14b.txt`: three discarded attempts at old 2 that ended at load 14.33, 43.22 and 13.93, with q32 up to 147 ms against 62 to 73 ms in the kept passes. A fourth attempt (ended at load 11.12) and the first new 2 (ended at load 16.14) were overwritten by their repeats and are not kept, so `bench-new-2.txt` is the repeat.
* `stats-before.txt`, `stats-after.txt`, `stats-q32-composite-off.txt`: per-kernel totals, best of five runs. `stats-after.txt` holds all five rewrites, including the three reverted ones. Load 2.49 to 2.81 across the three files.
* `plans-before.txt`, `plans-after.txt`: `acceptance_plans` output before and after the five rewrites. Load 1.95 and 8.18 at the start.
* `probe-q32-columns.txt`: extents, composite width, record layout and active radix passes of the two q32 sort columns. Load 6.59.
* `probe-q32-sort-timing.txt`: kernel time of `argsort(WatchID, ClientIP)`, `argsort(WatchID)` and `argsort(ClientIP)` on the real columns, on uniform columns with the same extents, and on the real columns shuffled independently. Load 7.34 to 7.23.

## Results

Times are criterion means in ms on `x86-64-v4-intrinsics`, the variant the CLI picks, averaged over the three passes, and the spread is the largest old time over the smallest across the passes.
The per-pass ratios are new/old for each pass in the order 1, 2, 3.

| Query | old | new | new / old | old spread | per-pass ratios | corgi | DuckDB 1t | kept | kept / corgi | kept / DuckDB |
|---|---|---|---|---|---|---|---|---|---|---|
| q15 | 10.53 | 9.67 | 0.918 | 3.3% | 0.916, 0.932, 0.907 | 28.46 | 9.22 | new | 0.34 | 1.05 |
| q35 | 10.28 | 9.47 | 0.922 | 1.4% | 0.912, 0.922, 0.930 | 39.15 | 16.02 | new | 0.24 | 0.59 |
| q32 | 67.53 | 61.42 | 0.909 | 6.3% | 0.924, 0.864, 0.942 | 60.45 | 71.29 | old | 1.12 | 0.95 |
| q31 | 4.89 | 5.26 | 1.075 | 8.2% | 1.075, 1.087, 1.063 | 4.93 | 9.62 | old | 0.99 | 0.51 |
| q30 | 4.29 | 4.21 | 0.982 | 5.8% | 0.961, 1.011, 0.973 | 5.29 | 8.27 | old | 0.81 | 0.52 |

The kept columns use the old time for the reverted queries.
The corgi and DuckDB figures are best-of times from another day, so they favor those systems slightly against a criterion mean, and the ratios are indicative.

### Five variants

Criterion means in ms over the three passes, in the order scalar, `v3-auto`, `v3-intrinsics`, `v4-auto`, `v4-intrinsics`, with new/old ratios below each pair.

| Query | form | scalar | v3-auto | v3-intr | v4-auto | v4-intr |
|---|---|---|---|---|---|---|
| q15 | old | 12.10 | 10.71 | 10.73 | 10.55 | 10.53 |
| q15 | new | 11.08 | 9.85 | 9.82 | 9.64 | 9.67 |
| q15 | new / old | 0.916 | 0.919 | 0.915 | 0.914 | 0.918 |
| q35 | old | 11.58 | 10.47 | 10.47 | 10.30 | 10.28 |
| q35 | new | 10.77 | 9.66 | 9.66 | 9.47 | 9.47 |
| q35 | new / old | 0.931 | 0.923 | 0.923 | 0.920 | 0.922 |
| q32 | old | 71.01 | 68.78 | 68.53 | 66.99 | 67.53 |
| q32 | new | 67.83 | 63.33 | 61.48 | 64.49 | 61.42 |
| q32 | new / old | 0.955 | 0.921 | 0.897 | 0.963 | 0.909 |
| q31 | old | 8.35 | 7.18 | 5.65 | 6.13 | 4.89 |
| q31 | new | 8.52 | 7.46 | 6.01 | 6.45 | 5.26 |
| q31 | new / old | 1.020 | 1.039 | 1.064 | 1.052 | 1.075 |
| q30 | old | 7.73 | 6.63 | 5.10 | 5.57 | 4.29 |
| q30 | new | 7.60 | 6.47 | 4.90 | 5.44 | 4.21 |
| q30 | new / old | 0.983 | 0.975 | 0.960 | 0.977 | 0.982 |

The old spread across passes is 1.4% to 3.3% for q15 and q35, 5.6% to 14.2% for q32, 4.5% to 8.2% for q31 and 4.2% to 6.4% for q30.

### Decisions

* **q15 and q35, kept.** The new form is faster in all fifteen cells of each query, by 6.8% to 9.3% for q15 and 6.1% to 8.8% for q35, against a spread of the old runs of at most 3.3%. The statistics show `argsort` (1.00 and 0.77 ms) and `topk` (0.04 ms) replaced by one `topk2` kernel (0.12 and 0.11 ms), and the summed kernel times fall by 1.1 and 0.7 ms. The criterion gains, 0.86 and 0.81 ms, are of the same size, and they sit at the edge of the 10% noise band by size but not by repeatability, since every cell moves the same way.
* **q32, restored.** The new form is faster in 14 of 15 cells, with a median ratio of 0.93, but the keep rule fails in pass 3, where the ratios are 0.997, 0.951, 0.911, 0.969 and 0.942 against an old spread of 5.6% to 14.2% across passes. The statistics show only 1.4 ms of 62.8 ms saved (see the gap below), against the 6.1 ms criterion gain of the three-pass means. The benchmark says the rewrite is probably faster by about 7% and the statistics say 2%, and the repeats do not settle it, so the rule restores the old file. The rewrite is kept as `programs-rejected/clickbench/q32.vk`.
* **q31, restored.** The new form is 1.4% to 11.9% slower in every cell of all three passes (2.0% to 7.5% on the means), and the statistics show `argsort2_i64_i64` at 2.43 ms against 1.73 ms for the two `argsort_i64` it replaces, with 0.2 ms saved in `gather_i64`.
* **q30, restored.** The new form is faster in 14 of 15 cells, by up to 5.2%, and 1.1% slower in one, and 1.7% to 4.0% faster on the means, against an old spread of 4.2% to 6.4%, so the gain is not beyond the spread.

### Per-kernel statistics

Summed kernel times in ms, best of five runs, from `stats-before.txt` and `stats-after.txt`, each query in the old and the rewritten form.
`other` is every kernel that is neither `argsort`, `topk`, `gather`, a reduction (`reduce_*`, `histogram`, `count`) nor `group_id`.

| Query | form | total | argsort | topk | gather | reductions | group_id | other |
|---|---|---|---|---|---|---|---|---|
| q32 | old | 62.77 | 32.87 | 0.37 | 20.12 | 4.46 | 0.00 | 4.96 |
| q32 | new | 61.37 | 40.94 | 0.39 | 9.33 | 4.84 | 0.00 | 5.88 |
| q31 | old | 5.09 | 1.73 | 0.03 | 0.44 | 0.19 | 0.00 | 2.69 |
| q31 | new | 5.78 | 2.43 | 0.03 | 0.25 | 0.20 | 0.00 | 2.87 |
| q30 | old | 4.60 | 1.20 | 0.02 | 0.41 | 0.24 | 0.00 | 2.72 |
| q30 | new | 4.56 | 1.35 | 0.03 | 0.20 | 0.25 | 0.00 | 2.73 |
| q15 | old | 10.96 | 1.00 | 0.04 | 0.06 | 3.76 | 6.07 | 0.03 |
| q15 | new | 9.85 | 0.00 | 0.12 | 0.00 | 3.64 | 6.06 | 0.03 |
| q35 | old | 10.55 | 0.77 | 0.04 | 0.06 | 3.70 | 5.95 | 0.03 |
| q35 | new | 9.82 | 0.00 | 0.11 | 0.00 | 3.69 | 6.00 | 0.03 |

The `topk` column of the old q15 and q35 is a single-key `topk_i64` that the planner already chose for the final slice of the old programs, so "old" includes one fused top-k.
The `topk` column of the new queries is `topk2_i64_i64`, and q32 keeps a single-key `topk_i64` for its count order in both forms because its stable tie rule needs no second key.

### The composite kernel against its off switch

`VERKEHRT_COMPOSITE_SORT=off` lowers `argsort(a, b)` to two single-key sorts and a composed gather set, and `stats-q32-composite-off.txt` runs the rewritten q32 that way.
The total is 61.04 ms against 61.37 ms with the kernel, level within the noise band, with `argsort` at 34.69 ms against 40.94 ms and `gather_i64` at 15.05 ms against 9.33 ms.
The composite kernel therefore saves 5.7 ms of gathers and costs 6.3 ms more in the sort than two single-key sorts, so its net gain over its own expansion is zero on q32.
Both forms beat the old program's 62.77 ms by 1.4 to 1.7 ms, which means the rewrite's gain comes from sharing one gather set, not from the kernel.

### Top-k bench

`2026-11-03-topk-bench.md` benches the fused kernels against the unfused plan on uniform and descending keys at 80,000 and 1M rows, and the fused kernels win every measured cell by 1.7x to 28.6x.
The bench is not repeated here.
In situ the gain is smaller because the sorts it removes are a small share of q15 and q35: `argsort` plus `topk` took 1.04 ms and 0.81 ms before, and `topk2` takes 0.12 and 0.11 ms.
That saves 0.9 and 0.7 ms of kernel time, and the summed totals fall 1.1 and 0.7 ms of 10.96 and 10.55 ms in the statistics, 10.1% and 6.9%, against criterion gains of 8.2% and 7.8%.

### The `acceptance_plans` diff

`plans-before.txt` and `plans-after.txt` differ only in the five rewritten queries.
In q32, q31 and q30 the two `argsort_i64` and the second set of gathers become one `argsort2_i64_i64` and one gather set.
In q15 and q35 the old `argsort_i64` of the keys and `topk_i64` over a gathered count vector become one `topk2_i64_i64` over `(0 - counts, gk)`.
No other query's plan changes, which a per-query comparison of the two files confirms.
`plans-after.txt` still shows the three reverted rewrites, because it was taken before the restore, and the committed programs of those queries are the old ones.

## Gaps

* **The in-situ gap of q32.** The designer's probe, quoted from the spec's "Evidence" section, put the grouping stage of q32 at 48.0 to 53.0 ms for the two-sort form and 34.7 to 37.1 ms for a single sort over 16-byte records and four gathers, a saving of 13.1 to 17.4 ms, and the sort alone at 27.4 ms against 35.3 ms. The kernel measures 40.9 ms for the sort in situ and 32.9 ms for the two single-key sorts, and the statistics show the kernel 8.1 ms above the two sorts (40.9 against 32.9 ms) and the gathers 10.8 ms below (9.3 against 20.1 ms), so the net saving is 1.4 ms of kernel time. The measured facts are in `probe-q32-columns.txt` and `probe-q32-sort-timing.txt`. The probe columns are as the designer assumed: WatchID spans 62 bits and ClientIP 32 bits as radix images, so the composite image is 94 bits, the kernel picks the 16-byte record (`R16`), and the sort makes 12 active passes against 8 and 8 for the two single sorts. The kernel nevertheless takes 3.4 ms per pass in situ, against 2.0 ms per pass for `argsort(WatchID)` and 2.2 ms for `argsort(ClientIP)`, so each composite pass costs 1.5 to 1.7 times a single-key pass of the same record size.
* **What the timing probe separates.** On uniform columns with the same extents the kernel takes 37.0 ms (3.1 ms per pass) and the single sorts 16.1 and 13.3 ms. On the real columns shuffled independently the kernel takes 40.2 ms and the single sorts 16.3 and 16.8 ms. The real ClientIP has 68,330 distinct values in 1M rows, and both the real and the shuffled column are slower than the uniform one, so skew costs 3.2 to 4.7 ms in either sort, against a gap of 6.8 ms between the kernel and the two sorts on the real columns, and does not explain it. The per-pass cost is high on uniform columns as well, so the cost is in the composite kernel, not in the data. The designer's copy of `radix_into` for 16-byte records was faster per pass than this kernel (27.4 ms for the sort alone), and the difference between the probe and the kernel is not measured.
* **Hypothesis, not measured.** The kernel computes the 128-bit composite image per row with a shift and mask in the counting pass over all 12 digits and again in every scatter pass, and `R16::digit` branches on the shift to pick the low or high word. Either could cost per pass what the designer's probe, which kept a fixed record, did not. No per-pass profile of the kernel was taken, so this stays a hypothesis, and the gap goes to its own follow-up issue.
* **The #9 gap.** The spec's probe also reproduced the in-situ slowdown of `argsort_i64` on real UserID columns, and q32's `argsort(WatchID)` and `argsort(ClientIP)` on the real columns take 16.3 and 18.0 ms against 16.1 and 13.3 ms on uniform columns, so ClientIP shows it at 35% and WatchID, which is unique, does not. That agrees with the designer's inference about runs of equal digits and does not test it.
* **q31 and q30 passes.** The plan's single pair for q31 and q30 became three passes, because every file covers all five queries, so the three-pass spread is the larger sample and no repeat was needed.
* **Load.** Four runs of old 2 and two of new 2 ended at load 7 to 43 and are discarded, and the kept old 2 started at load 2.70 and ended at 7.80. Its q32 times run 3% to 12% above old 1 and old 3, the same direction as drift on the host, so the q32 spread across passes of up to 14% may be host load rather than the build, and the q32 decision rests on that.
* **Unverified claims.** The corgi and DuckDB ratios divide this run's times by figures from another day.
