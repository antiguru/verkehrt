# Term graph and shared evaluation: measurements

This document measures the term graph in lowering, the plan graph in planning, and shared evaluation in the executor (issue #39).
ClickBench q29 drops from 8.13 ms to 0.123 ms in its best variant, about 66 times faster (both times from run b, the lower of the two runs without sharing, and 71 times from run a in the q29 table), because the ninety identical `reduce_add_i64` passes become one.
No other acceptance query changes plan, kernel calls, or tiled regions. In the full suites, run once per mode, the 20 queries other than q29 and q03 stay inside the 10% noise band. q03 moved beyond it in both directions and stays unresolved after two repeats (see "The full suites").
The function form of the generated-program case, which took 4.84 s and 1.8 GB at depth 7, takes 0.58 to 0.60 s and 24 MB, and grows linearly with depth.
The before numbers come from the same build with shared evaluation off (`VERKEHRT_SHARING=off`), and the DuckDB and corgi figures are reused from an earlier record and were not re-run.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | the branch with the term graph, plan graph and shared evaluation. `Options::no_sharing` turns shared evaluation off, the CLI takes `--no-sharing`, and the acceptance bench and the stats example read `VERKEHRT_SHARING=off`. Lowering and planning build the graphs in both modes, so "off" differs only in the executor |
| Before | the same build with `VERKEHRT_SHARING=off` |
| corgi and DuckDB | the figures of `2026-10-31-clickbench-now.md` (q29), reused and not re-run |
| Data | `hits_0.parquet` (1,000,000 rows) and the TPC-H data, the prepared data directories of the earlier rounds |

The method is the one in the "How the systems were timed" section of `2026-10-31-clickbench-now.md`.
Verkehrt times are criterion means with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time.
Per-kernel totals come from `acceptance_stats` with the default ISA and backend (best of five fresh runs).
Before each timed run the 1-minute load had to be at or below 4, and every evidence file records the load at its start and end.
Differences within about 10% count as noise.
The stats example also prints `tiled_regions`, the number of region runs that completed tiled in the best run.

### Evidence files

All files are in `2026-11-02-term-graph/`. Loads are the 1-minute average at the start and the end, and a bench can add to the end value, though not always (q29 runs ended lower than they started).

* `plans-before.txt` and `plans-after.txt`: the `acceptance_plans` output of all 22 programs before the change and on the final tree. They differ only in the harness's `finished in` timing line (11.25 s and 12.02 s). No load was recorded.
* `bench-q29-off-a.txt`, `bench-q29-off-b.txt`, `bench-q29-on-a.txt`, `bench-q29-on-b.txt`: q29, five variants, two runs each, in the order off a, on a, off b, on b. Loads: off a 2.95 to 2.78, on a 2.78 to 2.70, off b 2.70 to 3.81, on b 3.81 to 2.88.
* `stats-q29-off.txt` and `stats-q29-on.txt`: per-kernel totals of q29. Loads 2.88 to 2.69 and 2.69 to 2.53.
* `stats-all-sharing-off.txt` and `stats-all-sharing-on.txt`: per-kernel totals and tiled region counts of every acceptance query, one block per query with its own load lines. No query started or ended above 2.46.
* `bench-full-off.txt` and `bench-full-on.txt`: every acceptance query, five variants. Load 3.77 to 2.66 and 3.75 to 4.99.
* `bench-full-off-discarded.txt`: the first run without sharing, discarded because the load rose to 20.08 during it. Load 2.35 to 20.08.
* `bench-q03-off.txt`, `bench-q03-on.txt`, `bench-q03-off-2.txt`, `bench-q03-on-2.txt`: two more pairs for q03. Loads: 2.48 to 2.67, 2.67 to 24.32, 3.68 to 2.64, 2.64 to 2.59.
* `compile-time.txt`: the function form of the generated-program case at depths 2 to 7, three runs each, with and without `--no-sharing`. Load 2.31 to 2.24.

The first full run without sharing ended at a load of 20.08, and in it at least one variant of 13 queries other than q29 ran more than 10% slower than in the sharing run, so it was discarded and repeated. Its log is kept as `bench-full-off-discarded.txt` and is not used in any table.
The first `bench-q03-on.txt` ended at a load of 24.32, which no evidence file explains, and it is kept and discussed below. `bench-full-on.txt` ended at 4.99, above the cap of 4 that applies before a run, and started right after the discarded run with a 5-minute load of 15.25.

## Plans are unchanged

`plans-before.txt` is the plan output of the tree before the term graph, and `plans-after.txt` is the output of the final tree.
`diff` shows one differing line in 310, the timing line of the test harness, so every plan of every acceptance query is identical.
The executor, not the planner, therefore causes every difference in the tables below.

## q29

q29 is ninety exports `sum(ResolutionWidth + k)`, and after the integer sum rewrites every plan contains the same `(reduce_add_i64 ResolutionWidth)`, as in `2026-11-01-integer-rewrites.md`.
The 90 plans in `plans-after.txt` are the same as before, for example:

```
plan 6: (reduce_add_i64 ResolutionWidth)
plan 7: (add_i64_s (reduce_add_i64 ResolutionWidth) (mul_i64_s (length_i64 ResolutionWidth) 1))
plan 95: (add_i64_s (reduce_add_i64 ResolutionWidth) (mul_i64_s (length_i64 ResolutionWidth) 89))
```

Criterion means in ms before (sharing off) and in us after (sharing on), from the four `bench-q29-*` files.

| Variant | off, run a (ms) | off, run b (ms) | on, run a (us) | on, run b (us) | off a / on a |
|---|---|---|---|---|---|
| scalar | 15.43 | 15.34 | 201.3 | 201.0 | 77 |
| x86-64-v3-auto | 8.94 | 8.23 | 123.3 | 125.2 | 72 |
| x86-64-v3-intrinsics | 8.79 | 8.21 | 124.8 | 124.6 | 70 |
| x86-64-v4-auto | 8.93 | 8.13 | 124.5 | 124.0 | 72 |
| x86-64-v4-intrinsics | 8.83 | 8.13 | 123.5 | 123.2 | 71 |

The two runs without sharing differ by up to 10% on the vector variants (8.93 against 8.13 ms on `x86-64-v4-auto`, 9.9%), at the edge of the noise band, and the two runs with sharing differ by at most 2%.
The best variant is `x86-64-v3-intrinsics` at 8.79 ms in run a and `x86-64-v4-intrinsics` at 8.13 ms in run b without sharing, and `x86-64-v3-auto` at 123.3 us and `x86-64-v4-intrinsics` at 123.2 us in the two runs with sharing.
The ratios below use run a, 8.79 ms and 123.3 us, and are ratios of Verkehrt over the reference, so values below 1.0 mean Verkehrt is faster.

| Reference | time (ms) | off: best / reference | on: best / reference |
|---|---|---|---|
| DuckDB 1t | 7.24 | 1.21 | 0.017 |
| corgi | 33.14 | 0.27 | 0.0037 |

The DuckDB and corgi times are the best-of figures of `2026-10-31-clickbench-now.md`, and criterion means run a few percent above a best-of time.
Without sharing, q29 is 21% slower than DuckDB (11% in `2026-11-01-integer-rewrites.md`, which measured 8.03 ms on a quieter pair of runs), and with sharing it is 59 times faster than that DuckDB figure and 269 times faster than corgi's.
The roadmap goal of q29 below 1 ms is met with a factor of 8 to spare.

### Kernel shares

Per-kernel totals from `stats-q29-off.txt` and `stats-q29-on.txt`, in us.

| Kernel | calls off | time off | share off | calls on | time on | share on |
|---|---|---|---|---|---|---|
| `reduce_add_i64` | 90 | 8075.9 | 99.4% | 1 | 224.1 | 93.0% |
| `mul_i64_s` | 89 | 30.3 | 0.4% | 89 | 8.7 | 3.6% |
| `add_i64_s` | 89 | 10.1 | 0.1% | 89 | 7.5 | 3.1% |
| `length_i64` | 89 | 8.2 | 0.1% | 1 | 0.7 | 0.3% |
| total | | 8124.5 | | | 241.0 | |

Sharing removes 89 `reduce_add_i64` calls and 88 `length_i64` calls, and the 89 scalar kernels remain.
Both runs report no tiled region, so tiling plays no part.

Without sharing the 90 reductions take 89.7 us each on average (8075.9 / 90), and the criterion mean agrees (8.13 to 8.94 ms over 90 passes).
With sharing the single call takes 224.1 us in `stats-q29-on.txt` and 228.9 us in `stats-all-sharing-on.txt`, 2.5 times the unshared average, and the stats total of 241.0 us is about twice the criterion mean of 123 to 125 us for the whole run.
The criterion mean of 123.3 us bounds the single reduction at about 107 us, because it includes the 89 scalar kernels (about 16 us in the stats) on top of it.
The stats example times a fresh interpreter, and criterion reruns the same prepared program, so a cold first pass over the 8 MB column in the stats run would fit both figures.
Cold cache is only a hypothesis, because the five fresh stats runs share one process, each reads the same column, and the best of five is still 224 us. The unshared run's first call is also in its 90-call average, so cold and warm are not separated, and why the stats timer reads the shared call 2.5 times higher than criterion implies is an open question.
The headline speedup uses criterion, where the shared and unshared runs are comparable.

## The full suites

Ratios of the best variant over all 22 queries, from `bench-full-off.txt` and `bench-full-on.txt` (one run each, five variants).
The two right columns are the largest and smallest on / off ratio over the five variants.

| Query | off best (ms) | on best (ms) | on / off | largest variant ratio | smallest variant ratio |
|---|---|---|---|---|---|
| q00 | 0.000213 | 0.000206 | 0.97 | 0.98 | 0.94 |
| q01 | 0.097 | 0.091 | 0.94 | 1.00 | 0.90 |
| q02 | 0.186 | 0.184 | 0.99 | 1.00 | 0.97 |
| q03 | 2.157 | 1.464 | 0.68 | 1.29 | 0.68 |
| q04 | 5.697 | 5.745 | 1.01 | 1.04 | 0.99 |
| q06 | 0.191 | 0.185 | 0.97 | 1.01 | 0.92 |
| q07 | 0.329 | 0.326 | 0.99 | 1.02 | 0.98 |
| q08 | 15.503 | 15.988 | 1.03 | 1.03 | 1.01 |
| q09 | 19.743 | 19.658 | 1.00 | 1.05 | 0.98 |
| q15 | 10.806 | 10.825 | 1.00 | 1.03 | 0.98 |
| q19 | 0.258 | 0.244 | 0.95 | 0.99 | 0.95 |
| q27 | 6.021 | 5.625 | 0.93 | 1.00 | 0.93 |
| q29 | 8.350 | 0.124 | 0.015 | 0.015 | 0.013 |
| q30 | 4.247 | 4.294 | 1.01 | 1.02 | 0.99 |
| q31 | 4.813 | 4.886 | 1.02 | 1.03 | 1.00 |
| q32 | 67.362 | 66.085 | 0.98 | 1.07 | 0.98 |
| q35 | 10.390 | 10.614 | 1.02 | 1.06 | 1.00 |
| q40 | 3.233 | 3.339 | 1.03 | 1.03 | 1.00 |
| q41 | 2.704 | 2.704 | 1.00 | 1.02 | 0.99 |
| q42 | 2.099 | 2.093 | 1.00 | 1.03 | 0.98 |
| TPC-H q01 | 15.977 | 15.578 | 0.98 | 0.99 | 0.96 |
| TPC-H q06 | 1.594 | 1.597 | 1.00 | 1.01 | 1.00 |

Two queries have variants beyond the 10% band in this run, q29 and q03.
q29 is the intended change, and all five variants are faster by 67 to 76 times.
q03 has five variants outside the band, four of them slower with sharing (1.18, 1.29, 1.18, 1.24) and one faster (0.68), and no other query has any.
q03's plan has no repeated node and its kernel calls are identical in both modes (`stats-all-sharing-*.txt`), so the work is the same in both modes, and repeats were needed to see whether the time differs.
Two further pairs of q03 follow, and the result stays unresolved:

| Run | scalar | v3-auto | v3-intrinsics | v4-auto | v4-intrinsics | on / off per variant |
|---|---|---|---|---|---|---|
| full, off | 3.057 | 2.231 | 2.222 | 2.163 | 2.157 | |
| full, on | 3.613 | 2.868 | 2.626 | 2.688 | 1.464 | 1.18, 1.29, 1.18, 1.24, 0.68 |
| pair 1, off | 2.286 | 1.441 | 1.430 | 1.400 | 1.408 | |
| pair 1, on | 3.547 | 2.704 | 3.098 | 2.609 | 3.240 | 1.55, 1.88, 2.17, 1.86, 2.30 |
| pair 2, off | 3.379 | 2.570 | 2.582 | 2.540 | 2.549 | |
| pair 2, on | 3.341 | 2.578 | 2.590 | 2.540 | 2.523 | 0.99, 1.00, 1.00, 1.00, 0.99 |

The same code without sharing runs between 1.40 and 2.54 ms on `x86-64-v4-auto` across the three runs, and with sharing between 1.46 and 3.24 ms on `x86-64-v4-intrinsics`, so q03's run-to-run spread is larger than the band in both modes.
Sharing was slower in two of the three pairs (1.18 to 1.29 and 1.55 to 2.30) and level in one (0.99 to 1.00).
Pair 2 is the one pair with level times, and its 1-minute loads were 3.68 to 2.64 (off) and 2.64 to 2.59 (on), against 2.48 to 2.67 and 2.67 to 24.32 in pair 1. The 24.32 at the end of pair 1's sharing run is unexplained, so that pair is not trusted.
The kernel totals of q03 are level (2671.0 us off and 2588.0 us on in `stats-all-sharing-*.txt`).
The cause of q03's 1.4 to 3.4 ms spread for the same code is unknown, and the evidence neither shows that sharing slows q03 nor rules it out. The identical plan and kernel calls are the argument that sharing does not change q03's work, and a decisive answer needs more interleaved pairs, which were not run.

All other 20 queries stay within 10% in all five variants, with per-variant ratios from 0.90 (q01) to 1.07 (q32).
The spec's guard on programs without sharing, that the planning memo and the shared-node bookkeeping cost nothing outside the noise band, holds on the acceptance suites.

## Tiled regions

A shared node is a region boundary, and a shared input that sits to the right of a region kernel disables tiling in the statement that first computes it.
Only q29 repeats nodes, so no other acceptance query should lose a tiled region.
`stats-all-sharing-off.txt` and `stats-all-sharing-on.txt` give `tiled_regions` per query, and the counts are identical in both modes for all 22 queries.

| Queries | tiled regions off and on |
|---|---|
| q00, q01, q02, q04, q06, q07, q08, q15, q19, q29, q35 | 0 |
| q03, q09, q41, TPC-H q06 | 1 |
| q27, q32, q40, q42 | 2 |
| q30, q31 | 3 |
| TPC-H q01 | 4 |

Kernel call counts are also identical in both modes for every query except q29, whose `reduce_add_i64` drops from 90 calls to 1 and `length_i64` from 89 to 1.
No acceptance query loses a tiled region with sharing.

The generated-program CI seed (`tests/generated`, seed `0x5eed0001`, 64 cases) shows the mechanism.
The figures 402 and 129 are reused from `2026-11-01-generated-programs.md` and were not re-measured here: region runs that completed tiled fell from 402 to 129 with shared evaluation, which that record gives as the only change between the two counts.
The mechanism is that a shared node is a region boundary, so a shared input right of a region kernel disables tiling in the statement that first computes it, and the generator's deep chains of lets that read the previous one several times hit this often.
The acceptance table above is the evidence that no measured query loses tiling, and this document did not check whether the fall for generated programs also reflects less repeated work.

## Compile time

The function form of the generated-program case, `fn f(q: i64[]) -> i64[] { let b = filter(q, |p| (p > 3)); ((b + b) + (b + b)) }` called as `f(f(...f(a)))` with `let a = iota(10)`, wrapped as `print(length(...))`, in the release CLI.
The first column is the figure `2026-11-01-generated-programs.md` records before the term graph, and the others are the mean of three runs from `compile-time.txt` with sharing on.

| Depth | before: time | before: peak RSS | now: time | now: peak RSS |
|---|---|---|---|---|
| 2 | 0.20 s | 23 MB | 0.19 s | 23 MB |
| 3 | 0.29 s | 24 MB | 0.27 s | 24 MB |
| 4 | 0.38 s | 24 MB | 0.35 s | 24 MB |
| 5 | 0.52 s | 33 MB | 0.43 s | 24 MB |
| 6 | 1.06 s | 230 MB | 0.51 s | 24 MB |
| 7 | 4.84 s | 1.8 GB | 0.59 s | 24 MB |

Time grows by about 0.08 s per level and memory stays at 24 MB, where it grew by a factor of 7 to 8 per level at depths 6 and 7 before.
Runs with `--no-sharing` take about the same time (0.59 s at depth 7, 0.43 against 0.45 s at depth 5) and the same memory, because lowering and planning build the graphs in both modes.
The before figures come from an earlier run whose exact wrapper around the call is not recorded, so the differences of 0.01 to 0.03 s at depths 2 to 4 are not a result, and the differences at depths 5 to 7 are.

## What remains

q29 is no longer dominated by repeated passes, and its remaining 93% in `reduce_add_i64` is one pass over the 8 MB column.
The single call reads 2.5 times slower in the stats than the unshared per-call average, and the cause is an open question.
Extraction still charges a shared node once per use (issue #24), and fusing maps into reductions is issue #30.
The shared executor keeps every memoized value until the end of a run, and no acceptance query measures the memory cost of that.
