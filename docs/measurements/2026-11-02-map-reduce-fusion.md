# Fused reductions: measurements

This document measures fused reductions (issue #30), where a reduction over an elementwise chain is the root of a tiled region and each tile goes straight into the reduction's accumulator.
ClickBench q03 has a fused region (one run of `reduce_add_i64` per query run, the only one of the three queries whose statistics were recorded, and the only one the spec's plan survey finds), and it is the only query that moves beyond the 10% noise band in the full suites.
On `x86-64-v4`, fusion cuts q03's total kernel time from 2334.0 to 1517.7 us (0.65 times), and the region root `sub_i64` falls from 928.6 to 320.0 us, which is the effect the spec estimated (about 700 us from `sub_i64`).
The criterion means of q03 are lower with fusion in all 30 variant pairs of six off and on runs, but the size of the gain is unresolved: the best variant is 0.49 to 0.89 times its off time, because the off runs fall into two modes, about 2.5 ms and about 1.4 ms, and the on runs stay at 1.25 ms.
The stats of the other kernels of q03 disagree with the criterion in one place: `div_const_i64`, which no fusion touches, fell from 1008.6 to 829.6 us in the first pair of stats runs, and the evidence shows that its time varies from 804 to 1471 us across runs of both modes, so that fall is not an effect of fusion.
After fusion the division by 2^32 is 55% of q03's kernel time (53% to 63% across the runs), and this record gives it as the input for the `rem` and power-of-two `div_const` follow-up.
The before numbers come from the same build with `VERKEHRT_REDUCTION_FUSION=off`, and the DuckDB and corgi figures are reused from earlier records and were not re-run.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | the branch with fused reductions and the acceptance tools reading `VERKEHRT_REDUCTION_FUSION`. Tiling is the default (16384 rows) in every run |
| Before | the same build with `VERKEHRT_REDUCTION_FUSION=off`, which sets `Options::no_reduction_fusion`, so only fusion differs |
| Reducers that fuse | entries marked `Split::Free`. On aarch64 the NEON float and integer `reduce_min` and `reduce_max` are `Split::Bound` and run unfused, which does not matter on this x86 host and none of these runs exercises it |
| corgi and DuckDB | the figures of `2026-10-31-clickbench-now.md` (q03) and `2026-10-31-tier2.md` (TPC-H q06), reused and not re-run |
| Data | `hits_0.parquet` (1,000,000 rows) and the TPC-H data (1,199,969 `lineitem` rows), the prepared data directories of the earlier rounds |

The method is the one in `2026-11-02-selective-predicates.md`.
Verkehrt times are criterion means with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time, over the five variants (`scalar`, and the auto and intrinsics variants of `x86-64-v3` and `x86-64-v4`).
Per-kernel totals come from `acceptance_stats` on `x86-64-v3` and `x86-64-v4` (best of five fresh runs by total kernel time).
Before each run a script waited until the 1-minute load was at most 4, and every evidence file records the load at its start and end.
The 15-minute load was 5 to 10 at the start of most files, because other work shares the host, and the 1-minute gate does not cover that history.
Differences within about 10% count as noise.

### Evidence files

All files are in `2026-11-02-map-reduce-fusion/`. Loads are the 1-minute average at the start and the end, and a bench run adds up to 1 to the end value.

* `plans.txt`: the output of `acceptance_plans`, which equals `2026-11-02-term-graph/plans-after.txt` apart from the cargo header lines and the timing line.
* `stats-off.txt` and `stats-on.txt`: per-kernel totals of q03, TPC-H q06 and TPC-H q01 on `x86-64-v3` and `x86-64-v4`, with `tiled_regions`, `narrowed_tiles` and `fused_reductions`. Load 2.39 to 2.41 and 2.41 to 2.38.
* `probe-a.vk`, `probe-b.vk`, `probe-c.vk`: the three probe programs of the spec. They were copied into `tests/acceptance/clickbench/` for the runs and deleted afterwards.
* `stats-probes-off.txt` and `stats-probes-on.txt`: the same statistics for the probes. Load 2.38 to 2.35 and 2.35 to 2.35.
* `stats-heap-off.txt`, `stats-heap-on.txt`, `stats-mmap-off.txt`, `stats-mmap-on.txt`: q03 with `MALLOC_MMAP_THRESHOLD_` set to 1073741824 (8 MB arrays come from the heap) and to 65536 (every large array is a fresh mapping). Loads 2.01 to 2.01 in three of them and 2.01 to 2.09 in `stats-mmap-on.txt`, with a 15-minute load of 10.1 to 10.2.
* `bench-q03-p1-off.txt`, `bench-q03-p1-on.txt`, `bench-q03-p2-on.txt`, `bench-q03-p2-off.txt`, `bench-q03-p3-off.txt`, `bench-q03-p3-on.txt`: three interleaved pairs of q03 (off, on, on, off, off, on). Loads 2.00 to 3.81, 3.81 to 2.82, 2.82 to 2.53, 2.53 to 2.38, 2.38 to 2.28, 2.28 to 2.19.
* `bench-q03-p4-off.txt`, `bench-q03-p4-on.txt`, `bench-q03-p5-on.txt`, `bench-q03-p5-off.txt`: two more pairs, because the first three disagreed in size. Loads 1.98 to 2.22, 2.22 to 2.73, 2.73 to 2.53, 2.53 to 2.49.
* `bench-tpch-off.txt` and `bench-tpch-on.txt`: the control pair, TPC-H q01 and q06. Load 2.19 to 2.15 and 2.15 to 2.40.
* `bench-full-off.txt` and `bench-full-on.txt`: every acceptance query, five variants. Load 2.49 to 5.94 and 3.93 to 3.88. The off file ended at 5.94, above the cap that applies before a run, and the next file started at 3.93 after waiting for the gate.
* `correctness.txt`: `cargo test --release --test acceptance` with both data directories set, which passes (17 passed, 0 failed, 1 ignored) with fusion on. Load 1.88 to 2.38.

## The plans

The plans are unchanged.
`diff` of `plans.txt` against `2026-11-02-term-graph/plans-after.txt` shows no difference outside the cargo header and the test timing line, so the planner, the plans and `--plan` output are as before.
q03's plan has `div_const_i64 UserID 4294967296` as its own statement (`hi`, read by two statements) and `reduce_add_i64 (sub_i64 UserID (mul_i64 hi 4294967296))` for the sum of `lo`, and the fused region is that sum with `sub_i64` and `mul_i64` below it.
`fused_reductions` is `reduce_add_i64=1` for q03 in `stats-on.txt` on both ISAs and empty for TPC-H q06 and q01, so q03 is the only one of the three with a fused region.
With fusion off it is empty everywhere.
The plan survey of the spec found the same, and this record did not run the other 19 ClickBench queries for `fused_reductions`, so that claim for them rests on the spec's reading of the plans and on the criterion ratios below.

Call counts, `bytes`, `read` and `tiled_regions` equal the unfused run's in `stats-off.txt` and `stats-on.txt` for every kernel of all three queries (checked with `diff` after removing the `time` field).
In the probes only `tiled_regions` differs, for `probe-b`: it is 0 off and 1 on, because with fusion off a whole division under a sum is not a tiled region and with fusion on it is a fused region of one kernel.

## Kernel totals of q03

Times in us, from `stats-off.txt` and `stats-on.txt`.
`reduce_add_i64` covers both calls (the sum of `hi` and the fused root), and "other" is the remainder of the total, which holds the scalar kernels and `length_i64`.

| Kernel | v4 off | v4 on | change | v3 off | v3 on | change |
|---|---|---|---|---|---|---|
| `div_const_i64` | 1008.6 | 829.6 | -179.0 | 1382.2 | 865.3 | -516.9 |
| `sub_i64` (region root, elementwise) | 928.6 | 320.0 | -608.6 | 934.1 | 334.8 | -599.3 |
| `mul_i64` (inner) | 192.9 | 204.6 | +11.7 | 288.9 | 208.3 | -80.6 |
| `reduce_add_i64` | 181.5 | 153.2 | -28.3 | 199.6 | 177.3 | -22.3 |
| other | 22.4 | 10.3 | -12.1 | 10.4 | 9.3 | -1.1 |
| total | 2334.0 | 1517.7 | -816.3 | 2815.2 | 1595.0 | -1220.2 |

The spec estimated that fusion saves most of the difference between a region root that writes a full array and an inner kernel, about 700 us of 2359 us (30%).
The root `sub_i64` falls by 608.6 us on `x86-64-v4` and 599.3 us on `x86-64-v3`, about 600 us, 90 to 100 us below the estimate on the root alone, and the reduction's own time falls by 28 and 22 us.
The total falls by 816.3 us (35%) and 1220.2 us (43%), more than the estimate, but 179.0 and 516.9 us of that is the change of `div_const_i64`, which fusion does not touch (next section).
Without the `div_const_i64` change the total falls by 637.3 us (27% of the off total) on v4 and 703.3 us (25%) on v3, which brackets the estimate of 700 us.
The root's time with fusion (320.0 us) is still above the 134 to 141 us it takes in the probes (below), a gap this record has not explained.
In q03 the root reads `hi`, an 8 MB array that `div_const_i64` wrote earlier, besides `UserID`, and the probes read only `UserID`, which may account for it, but the evidence does not separate that from other causes.

### The `div_const_i64` change

`div_const_i64` is the first kernel q03 runs, its input `UserID` is a global, and it is a whole call in both modes, so fusion changes neither its work nor its arguments.
Its time still fell by 18% on v4 and 37% on v3 in the first pair of stats runs, which is the unexplained change of the task 2 probe.
`stats-heap-*.txt` and `stats-mmap-*.txt` repeat q03 with the allocator pinned to one behavior, and the times of `div_const_i64` across all q03 runs in this record are:

| Run | v4 off | v4 on | v3 off | v3 on |
|---|---|---|---|---|
| default allocator (`stats-off.txt`, `stats-on.txt`) | 1008.6 | 829.6 | 1382.2 | 865.3 |
| 8 MB arrays from the heap (`stats-heap-*.txt`) | 816.2 | 803.7 | 877.2 | 1136.3 |
| every large array a fresh mapping (`stats-mmap-*.txt`) | 1470.7 | 1107.2 | 1170.2 | 1134.8 |

The time ranges from 803.7 to 1470.7 us over the twelve runs and does not separate the modes: with the heap pinned the two modes are level on v4 (816.2 and 803.7) and the on run is slower on v3 (877.2 against 1136.3), and with mappings pinned the off run is slower on both.
So the evidence rules out a stable effect of fusion on `div_const_i64`, and the 18% and 37% falls of the default pair are two draws from a spread that is as wide in each mode.
The cause of the spread is not established.
The first-touch page faults of its fresh 8 MB output are a candidate by inference, because the stats timer wraps the output allocation, but the pinned runs do not show that setting alone removes the spread.
Under the pinned heap the root `sub_i64` stays at 916.2 us (v4, off) and 981.4 us (v3, off), against 329.3 and 348.9 us on, so the root's cost with fusion off is not page faults of a fresh mapping alone.

## The probes

Each probe sums an expression over the same `UserID` column, run on both ISAs with fusion off and on.
Times in us from `stats-probes-off.txt` and `stats-probes-on.txt` (v4, with v3 in parentheses), and the spec's single probe runs on this tree (x86-64-v4, load 1.67 to 1.78, not committed) are the "spec" column.

| Probe | Kernel | off | on | spec (off) |
|---|---|---|---|---|
| A, `sum(UserID - (div_const(UserID, 2^32) * 2^32))` | `div_const_i64` (inner) | 654.6 (680.6) | 657.6 (664.6) | 639.1 |
| | `mul_i64` (inner) | 94.3 (163.1) | 91.3 (169.3) | 93.9 |
| | `sub_i64` (root) | 469.8 (478.6) | 133.8 (149.3) | 441.1 |
| | `reduce_add_i64` | 89.8 (93.5) | 76.1 (83.4) | 90.4 |
| | total | 1308.5 (1415.8) | 958.8 (1066.6) | |
| B, `sum(div_const(UserID, 2^32))` | `div_const_i64` | 745.8 (776.0) | 671.2 (659.3) | 765.1 |
| | `reduce_add_i64` | 88.5 (95.9) | 74.1 (76.4) | |
| | total | 834.3 (871.9) | 745.3 (735.7) | |
| C, `sum((UserID * 3) - UserID)` | `mul_i64` (inner) | 283.4 (384.3) | 314.8 (429.2) | 273.6 |
| | `sub_i64` (root) | 441.5 (472.8) | 140.4 (138.2) | 466.6 |
| | `reduce_add_i64` | 90.1 (94.0) | 76.5 (78.6) | |
| | total | 815.0 (951.1) | 531.7 (646.0) | |

The probes reproduce the spec's single runs with fusion off to within 7% on every kernel.
In probes A and C the full-output root costs 441.5 to 478.6 us and falls to 133.8 to 149.3 us with fusion, a saving of 301 to 336 us on v4 for 1,000,000 rows, and probe A's total falls by 27% (v4) and 25% (v3).
That is below q03's 600 us at the root, so the 440 to 930 us range in the spec's estimate is covered only at its lower end by the probes, and q03's root is the expensive case.
Probe B shows what a fused division alone saves: the whole division and the sum fall from 834.3 to 745.3 us (11%), and the fused division reads 671.2 us against 745.8 us whole.
In probe C the inner `mul_i64` rises from 283.4 to 314.8 us on v4 while the root falls by 301 us, and the evidence does not say why the multiply rises. A candidate, by inference only, is that the first kernel of the region absorbs a cold read of `UserID` that the root paid before.

## The pairs of q03

Criterion means in ms, off then on, and the ratio on over off, for five variants.
The run order was off then on in pairs 1, 3 and 4 and on then off in pairs 2 and 5, and the evidence files are named by pair.
The last column group is the full suite run (`bench-full-off.txt`, `bench-full-on.txt`), q03 only.

| variant | p1 off | p1 on | p1 | p2 off | p2 on | p2 | p3 off | p3 on | p3 |
|---|---|---|---|---|---|---|---|---|---|
| scalar | 3.370 | 2.123 | 0.63 | 3.341 | 2.145 | 0.64 | 2.296 | 2.122 | 0.92 |
| x86-64-v3-auto | 2.590 | 1.334 | 0.51 | 2.575 | 1.338 | 0.52 | 1.450 | 1.313 | 0.91 |
| x86-64-v3-intrinsics | 2.565 | 1.336 | 0.52 | 1.445 | 1.337 | 0.92 | 1.462 | 1.310 | 0.90 |
| x86-64-v4-auto | 2.516 | 1.252 | 0.50 | 2.527 | 1.248 | 0.49 | 1.417 | 1.246 | 0.88 |
| x86-64-v4-intrinsics | 2.543 | 1.254 | 0.49 | 2.490 | 1.244 | 0.50 | 1.405 | 1.248 | 0.89 |

| variant | p4 off | p4 on | p4 | p5 off | p5 on | p5 | full off | full on | full |
|---|---|---|---|---|---|---|---|---|---|
| scalar | 3.028 | 2.123 | 0.70 | 2.289 | 2.118 | 0.93 | 3.033 | 2.121 | 0.70 |
| x86-64-v3-auto | 2.253 | 1.317 | 0.58 | 2.588 | 1.310 | 0.51 | 2.246 | 1.317 | 0.59 |
| x86-64-v3-intrinsics | 2.237 | 1.315 | 0.59 | 2.612 | 1.311 | 0.50 | 2.241 | 1.312 | 0.59 |
| x86-64-v4-auto | 2.525 | 1.252 | 0.50 | 2.512 | 1.250 | 0.50 | 2.172 | 1.246 | 0.57 |
| x86-64-v4-intrinsics | 2.184 | 1.251 | 0.57 | 1.403 | 1.245 | 0.89 | 1.409 | 1.247 | 0.89 |

The on runs are stable: 1.244 to 1.254 ms on each v4 variant, 1.310 to 1.338 ms on each v3 variant and 2.118 to 2.145 ms on scalar across the six runs, a spread of about 1% on v4 and 2% on v3.
The off runs are not.
Each off measurement lands in a fast mode (1.403 to 1.462 ms on the four vector variants, 2.289 to 2.296 ms on scalar) or a slow one (2.172 to 2.612 ms, and 3.028 to 3.370 ms on scalar), 9 of 30 in the fast mode, and the mode changes between variants within one file (pair 2 has `x86-64-v3-intrinsics` at 1.445 ms among slow neighbours).
The ratio is 0.49 to 0.70 against a slow-mode off and 0.88 to 0.93 against a fast-mode off.
The pairs therefore disagree in size, and this record calls the size unresolved, between about 0.5 and 0.9 times.
The direction does not disagree, because the on mean is below the off mean in all 30 pairs, and 26 of the 30 are beyond the 10% band (the four others are 0.90 to 0.93).
The fast off mode (1.40 ms on v4) is still 12% above the on runs (1.25 ms), so a fast-mode off run does not close the gap.
Earlier records saw the same spread without fusion (1.4 to 3.4 ms in `2026-11-02-term-graph.md` and `2026-11-02-selective-predicates.md`, both left unresolved), and this record does not explain the two off modes either.
The stats runs are another measurement (kernel time with the timer on, in one process of five runs), so they cannot say which criterion mode they correspond to, and the per-kernel evidence says nothing about what the fast mode is.

The per-kernel stats are the primary evidence, and they say fusion removes about 600 us at the root of q03's sum and 27% to 35% of its kernel time on v4, which agrees in direction with every pair, and the kernel-time ratio of 0.65 on v4 lies inside the slow-mode range of the pairs (0.49 to 0.70).

## Controls

TPC-H q01 and q06 have no fused region (`fused_reductions` is empty in `stats-on.txt`), so they show what fusion costs where it does not apply.
Criterion means in ms from `bench-tpch-off.txt` and `bench-tpch-on.txt`, one pair.

| Query | variant | off | on | ratio |
|---|---|---|---|---|
| TPC-H q06 | scalar | 6.124 | 6.142 | 1.00 |
| TPC-H q06 | x86-64-v3-auto | 2.954 | 2.921 | 0.99 |
| TPC-H q06 | x86-64-v3-intrinsics | 2.506 | 2.495 | 1.00 |
| TPC-H q06 | x86-64-v4-auto | 2.391 | 2.385 | 1.00 |
| TPC-H q06 | x86-64-v4-intrinsics | 1.702 | 1.708 | 1.00 |
| TPC-H q01 | scalar | 24.028 | 26.262 | 1.09 |
| TPC-H q01 | x86-64-v3-auto | 19.966 | 20.383 | 1.02 |
| TPC-H q01 | x86-64-v3-intrinsics | 20.512 | 19.302 | 0.94 |
| TPC-H q01 | x86-64-v4-auto | 19.554 | 19.663 | 1.01 |
| TPC-H q01 | x86-64-v4-intrinsics | 17.413 | 16.404 | 0.94 |

All ten stay inside the band, from 0.94 to 1.09, and the full suite run gives 0.98 to 1.03 for q06 and 0.96 to 1.03 for q01.
The per-kernel totals of q06 are 2234.5 us off and 2173.2 us on v4 and 2769.8 and 2772.1 us on v3, and q01's fall (18474.5 to 17924.9 us on v4, 20140.6 to 18647.4 us on v3) is a spread of one stats run of a query that has no fused region, so it is no effect of fusion.

## The full suites

Ratios of the best variant over all 22 queries, from `bench-full-off.txt` and `bench-full-on.txt`, one off and on pair.
Ratios are on over off, and values below 1.0 mean fusion is faster.

* q03: best variant 1.409 ms off and 1.246 ms on (0.88), and variant ratios from 0.57 to 0.89. All five variants are more than 10% faster, with the unresolved size described above. The best variant ratio of 0.88 is the fast off mode.
* The other 21 queries: no variant of any of them leaves the band in either direction (variant ratios 0.92 to 1.08). The best-variant ratio runs from 0.92 (q01) to 1.07 (q27), and the extremes are q27 (1.04 to 1.08 across variants, best variant 5.418 ms off and 5.809 ms on), q09 (0.96 to 1.07) and q42 (0.98 to 1.04).

So q03 is the only query beyond the band, and it was repeated with four more pairs, as the section above describes.
No query got slower beyond the band.
`q27` was unstable across pairs in the tiled-regions record, it stays inside the band here (1.04 to 1.08), and it was not repeated.
The 21 other queries have no fused region by the plans (see "The plans"), so their ratios are the noise of the host and not an effect of fusion, which this record did not test beyond the plan reading.

## Against DuckDB and corgi

The best variant of q03 is `x86-64-v4-intrinsics` in the on columns and the fast mode of `x86-64-v4-intrinsics` in one off column, and TPC-H q06 is `x86-64-v4-intrinsics` in both.
Ratios below 1.0 mean Verkehrt is faster.
The Verkehrt times are the first pair (`bench-q03-p1-*.txt` and `bench-tpch-*.txt`), the off column of q03 also gives the fast-mode value of pair 3 (1.405 ms) in parentheses.

| Query | DuckDB 1t (ms) | corgi (ms) | off (ms) | on (ms) | off / DuckDB | on / DuckDB | off / corgi | on / corgi |
|---|---|---|---|---|---|---|---|---|
| q03 | 1.96 | 1.03 | 2.543 (1.405) | 1.254 | 1.30 (0.72) | 0.64 | 2.47 (1.36) | 1.22 |
| TPC-H q06 | 4.74 | 4.36 | 1.702 | 1.708 | 0.36 | 0.36 | 0.39 | 0.39 |

The DuckDB and corgi times are reused, not re-run: q03's from `2026-10-31-clickbench-now.md` and TPC-H q06's from `2026-10-31-tier2.md`.
Verkehrt's times in those records come from builds before tiling and the term graph, so the ratios are not a before and after of this layer alone.
With fusion on, q03 is faster than DuckDB (0.64) and still slower than corgi (1.22), and TPC-H q06 is unchanged because it has no fused region.
The change this layer contributes to q03 is the difference between the off and on columns, 0.66 against DuckDB and 1.25 against corgi for the slow-mode off value, and 0.08 and 0.14 for the fast one.

## The division that remains in q03

With fusion on, q03's `div_const_i64` takes 829.6 us of the 1517.7 us kernel total on v4 (54.7%), and 865.3 of 1595.0 us on v3 (54.3%).
Across the six on runs of this record (`stats-on.txt`, `stats-heap-on.txt`, `stats-mmap-on.txt`) its share ranges from 53% to 63%, with times of 803.7 to 1136.3 us.
The scalar `div_const_i64_s` adds 3.1 us.
The other kernels take 685.0 us on v4 (45%): `sub_i64` 320.0 us (21%), `mul_i64` 204.6 us (13%), the two `reduce_add_i64` calls 153.2 us (10%), and 7 us of the rest.
There is only one division in q03's plan that runs per row, the `hi` of the whole array, so a `rem` for `lo` would add a second one.

The division costs 0.8 ns per row on v4, where the multiply costs 0.2 ns per row in q03 (`mul_i64` 204.6 us for 1,000,000 rows) and 0.09 to 0.3 ns per row in the probes.
Probe A's division inside a fused region costs 657.6 us on v4 (664.6 us on v3), and it replaces `mul_i64` and `sub_i64` (91.3 and 133.8 us in probe A, 204.6 and 320.0 us in q03).
Two inferences follow from these numbers, and this record did not test either.
A `rem` kernel that costs one division per row (about 650 us as a region kernel) would not beat the 525 us that `mul_i64` and `sub_i64` take in q03 now, so after fusion `rem` saves time only if the division itself gets cheaper.
A power-of-two path in `div_const_i64` for 2^32, a shift with a sign correction, would cost about what a multiply does if it compiled to similar code, and q03 would then save roughly 530 to 630 us of its 1517.7 us (35% to 42%), with the division at 200 to 300 us and the spread of the measurement of 804 to 1471 us in the way of a closer estimate.
A power-of-two path also makes `rem` cheap to build as the same shift and a subtract, and the choice between `rem` and the power-of-two path is the follow-up's.

The 525 us of `mul_i64` and `sub_i64` is also the input for issue #53, register-resident chains.
Both kernels write to a tile buffer in cache, and with fusion on they run over 16384-row tiles of 128 KB (an 8-byte row) each, so the stores are cache hits, and no number here measures what a chain that skips the buffer would save.
`sub_i64` at 320 us reads `hi` and `UserID` from memory in q03, and the probes' roots take 134 to 140 us over one input array, so part of the 320 us is likely the second input array and not the kernel (inference).

## What remains

Fusion removes the full-size output of the region root, and the evidence agrees with the spec on size: about 600 us at q03's root and 27% to 35% of its kernel time on v4, a 1.25 ms criterion mean on every variant of q03 against 1.4 to 2.6 ms before.
The size of the criterion gain stays unresolved because q03's off time has two modes that this record did not explain, and the on runs hold one mode.
The division by 2^32 is more than half of q03's kernel time, and the change of its time in the first stats pair is not an effect of fusion.
The `rem` and power-of-two decision, issue #53 and the open question of the off modes are the follow-ups this record leaves.
