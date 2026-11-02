# Selective predicate evaluation: measurements

This document measures range narrowing in tiled regions (issue #49) on the mask-chain queries ClickBench q40, q41 and q42 and TPC-H q06.
Narrowing lowers the best variant of q40, q41 and q42 by 17%, 20% and 20% (criterion means, 0.83, 0.80 and 0.80 times), and all 15 variant pairs of the three queries run more than 10% faster in each of three pairs of runs.
The comparisons evaluate 0.497, 0.512 and 0.531 of the whole-array rows on those three and 0.998 on TPC-H q06, which matches the spec's simulation, and TPC-H q06 does not change beyond the noise band (1.00 to 1.05 times).
On q06 the narrowing finds almost nothing to skip and still scans the left operands, a small cost: `and_bool` rises from 252 to 330 us in the stats and the criterion means rise by up to 5%. Part of the stats rise is timing itself, since with statistics on a guard times its scan, its right operand and its fill separately in every tile.
Comparison time fell with the rows evaluated on the three ClickBench queries (0.52 to 0.60 of selective-off time for 0.50 to 0.53 of the rows), which is the premise the spec sets for selection vectors.
Across both suites, 15 of 110 query and variant pairs run more than 10% faster and 2 run more than 10% slower in the first pair, and neither slower pair is a result of narrowing on this evidence: `q03` on `x86-64-v4-intrinsics` is unresolved because its repeat disagrees, and TPC-H q01 on `x86-64-v3-auto` reproduces at 1.11 and 1.12 on a query where no tile narrowed.
The before numbers come from the same build with `VERKEHRT_SELECTIVE=off`, and the DuckDB and corgi figures are reused from earlier records and were not re-run.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | the branch with selective evaluation and the acceptance tools reading `VERKEHRT_SELECTIVE`. Tiling is the default (16384 rows) in every run |
| Before | the same build with `VERKEHRT_SELECTIVE=off`, which sets `Options::no_selective`, so only narrowing differs |
| corgi and DuckDB | the figures of `2026-10-31-clickbench-now.md` (q40, q41, q42) and `2026-10-31-tier2.md` (TPC-H q06), reused and not re-run. This record took them from the table of `2026-11-02-tiled-regions.md` |
| Data | `hits_0.parquet` (1,000,000 rows) and the TPC-H data (1,199,969 `lineitem` rows), the prepared data directories of the earlier rounds |

The method is the one in `2026-11-02-tiled-regions.md`.
Verkehrt times are criterion means with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time.
Per-kernel totals come from `acceptance_stats` on `x86-64-v4` with the default backend (best of five fresh runs by total kernel time).
Before each timed run the 1-minute load had to be at or below 4, and every evidence file records the load at its start and end.
The 15-minute load was 9.13 at the start of `stats-off.txt`, 8.89 at the start of `bench-four-off.txt`, and 3.62 at the start of `stats-other-off.txt`, and the 1-minute gate does not cover that history, so absolute times may differ from a quiet host's while the off and on runs of a pair stay adjacent.
Differences within about 10% count as noise.

### Evidence files

All files are in `2026-11-02-selective-predicates/`. Loads are the 1-minute average at the start and the end, and the bench itself adds up to 1 to the end value.

* `plans.txt`, `q40.vk`, `q41.vk`, `q42.vk`, `q06.vk`: the `--plan` output of reduced forms of the four mask expressions. The CLI binds no column data (see `2026-10-31-clickbench-now.md`), so each column is a distinct `iota(100) + k` stand-in, which keeps the term graph from merging equal columns. The reduced forms contain only the mask expression, so they show operand order and not the rest of each plan.
* `stats-off.txt` and `stats-on.txt`: per-kernel totals with `read` and `narrowed_tiles` for the four queries. Load 2.33 to 2.36 and 2.36 to 2.33.
* `stats-other-off.txt` and `stats-other-on.txt`: the same for `q03` and TPC-H q01 on `x86-64-v3` and `x86-64-v4`. Load 2.64 to 2.62 and 2.62 to 2.57.
* `bench-four-off.txt` and `bench-four-on.txt`: first pair, the four queries, five variants. Load 2.03 to 2.10 and 2.10 to 2.57.
* `bench-four2-on.txt` and `bench-four2-off.txt`: second pair, run in the reverse order. Load 2.33 to 3.26 and 3.26 to 2.56.
* `bench-full-off.txt` and `bench-full-on.txt`: every acceptance query, five variants. Load 2.52 to 2.28 and 2.28 to 2.82.
* `bench-repeat-on.txt` and `bench-repeat-off.txt`: a second pair for `q03` and TPC-H q01, run in the reverse order. Load 2.62 to 2.91 and 2.91 to 2.85.
* `correctness.txt`: `cargo test --release --test acceptance` with both data directories set, which passes (17 passed, 0 failed, 1 ignored) with selective evaluation on. Load 1.76 to 1.80.

The off and on files of a pair use the same build and run back to back, and no run waited for the load to fall because it was at most 3.3 at every start.
An off file's end load is its successor's start load in the first pair of `bench-full` and in `bench-four`, so those load values are the same reading.

## The plans and the guard order

The plans keep the source operand order of each mask, so the first predicate of each query is the left operand that narrows the rest.

| Query | root guard, left operand then right operand | first comparison |
|---|---|---|
| q40 | `and(and(CounterID == 62, and(date >= lo, date <= hi)), and(and(IsRefresh == 0, or(TraficSourceID == -1, TraficSourceID == 6)), RefererHash == h))` | `CounterID == 62` |
| q41 | `and(and(CounterID == 62, and(date >= lo, date <= hi)), and(and(IsRefresh == 0, DontCountHits == 0), URLHash == h))` | `CounterID == 62` |
| q42 | `and(and(CounterID == 62, and(date >= lo, date <= hi)), and(IsRefresh == 0, DontCountHits == 0))` | `CounterID == 62` |
| TPC-H q06 | `and(and(shipdate >= 8766, shipdate < 9131), and(and(discount >= 5, discount <= 7), quantity < 24))` | `l_shipdate >= 8766` |

The ClickBench comparisons appear in the plan as `eq_i64` and `le_i64`, with `date >= lo` as `le_i64 lo date`.
In the three ClickBench plans `CounterID == 62` guards every other comparison, and the nested `and` and `or` guards narrow within the range their parent gave them.
In TPC-H q06 the date range comes first, and the spec's simulation found that rows passing it are spread across the tiles, so it leaves almost nothing to narrow.
The plans printed here are from reduced forms, so the operand order of the full programs is inferred from them and not printed.

## Rows evaluated against the simulation

The rows the mask comparisons ran over follow from the `read` bytes of `eq_i64`, `le_i64` and `lt_i64` in `stats-off.txt` and `stats-on.txt`, at 8 bytes per row.
The selective-off count is the number of comparisons times the table rows times 8 (56,000,000 bytes for q40's seven comparisons).
Two calls do not belong to the mask and are removed: the `iota(n) == 0` of q40's grouping code (one `eq_i64` call, 400,896 bytes, whole in both runs), and q42's one `lt_i64` of 11,520 bytes after the mask.

| Query | comparison `read` bytes, selective off | selective on | fraction | simulation (range narrowing, 16384 rows) |
|---|---|---|---|---|
| q40 | 56,000,000 | 27,848,800 | 0.4973 | 0.497 |
| q41 | 48,000,000 | 24,552,536 | 0.5115 | 0.512 |
| q42 | 40,000,000 | 21,242,392 | 0.5311 | 0.531 |
| TPC-H q06 | 47,998,760 | 47,918,064 | 0.9983 | 0.998 |

The fractions agree with the simulation to the third decimal, so there is no difference to explain.
The `narrowed_tiles` counts are 110 (q40), 78 (q41), 73 (q42) and 235 (TPC-H q06), so q06's tiles narrow often, by a few rows each time.
The `and_bool` and `or_bool` kernels read fewer bytes too: `and_bool` reads 4,138,102 bytes in q40 against 10,000,000 selective-off, because a guard runs only over its range.
`eq_i64` reads 21,628,496 bytes in q40 against 40,400,896, and the stats record the calls as 6 in both runs, so call counts are unchanged.

## Comparison time against rows evaluated

The table gives, from the same two stats files, the time in us of the mask comparison kernels (`eq_i64` and `le_i64`, plus `lt_i64` in TPC-H q06 only, so q42's 11,520-byte `lt_i64` call is not in the column, while q40's `iota(n) == 0` call of 400,896 bytes is in it), and the guards `and_bool` and `or_bool`.

| Query | comparison rows | comparison time off | on | on / off | `and_bool` off | `and_bool` on | total kernel time off | on |
|---|---|---|---|---|---|---|---|---|
| q40 | 0.497 | 1486.3 | 767.2 | 0.52 | 308.9 | 269.9 | 3867.9 | 3062.0 |
| q41 | 0.512 | 1349.2 | 706.2 | 0.52 | 285.2 | 291.2 | 3164.3 | 2540.8 |
| q42 | 0.531 | 1073.7 | 640.3 | 0.60 | 218.0 | 223.2 | 2478.9 | 2063.7 |
| TPC-H q06 | 0.998 | 970.3 | 1023.8 | 1.06 | 251.9 | 330.2 | 1960.8 | 2075.1 |

The comparison time fell in proportion to the rows evaluated on q40 and q41 (0.52 for 0.50 and 0.51 of the rows) and by somewhat less on q42 (0.60 for 0.53), so the premise of the spec holds on the three ClickBench queries.
The `le_i64` calls fall as the rows do in q40 (330.2 to 138.4 us for 0.414 of the bytes), q41 (318.1 to 129.5) and q42 (309.8 to 136.9), and `eq_i64` falls by less than its bytes: q42's 763.9 to 503.4 us is 0.66 of the time for 0.61 of the bytes.
This record did not measure why q42's `eq_i64` falls by less, and the evidence does not separate a per-call cost of the shorter calls from the whole-length first comparison.
The spec estimated that narrowing saves about a fifth of the kernel time of q40, q41 and q42, and the comparisons, `and_bool` and `or_bool` together fall by 19.8%, 20.1% and 17.3% of the selective-off total kernel time.
The total kernel time falls to 0.79, 0.80 and 0.83 of the off run, and the criterion means of the best variant fall to 0.83, 0.80 and 0.80, which agree within 4 points.

TPC-H q06 is the control for the per-tile cost, because narrowing skips 0.17% of its comparison rows but still scans the left operand of its guards in 235 narrowed guard evaluations.
Its guard time rises from 251.9 to 330.2 us (`and_bool`) and its comparison time by 6% (970.3 to 1023.8 us), which is inside the 10% band, and its criterion means rise by 1% to 5% in each of three pairs (below).
`and_bool` takes 269.9 us on against 308.9 us off in q40 and is level in q41 (291.2 against 285.2) and q42 (223.2 against 218.0), so on the ClickBench queries the guards are not slower beyond the band.
The `and_bool` rise on q06 is 78 us in one stats run, and the criterion rise of 3% to 5% on the best variant is inside the band, so q06 pays a small cost, part of which in the stats is the extra timing of each guard's scan and fill, for finding almost nothing to skip.

## Results

Times are criterion means in ms for the five variants, selective off then on, from the first pair (`bench-four-off.txt`, `bench-four-on.txt`).
The ratio is on over off, and values below 1.0 mean narrowing is faster.
The three ratio columns are the first pair, the second pair (`bench-four2-off.txt`, `bench-four2-on.txt`) and the full suite run (`bench-full-off.txt`, `bench-full-on.txt`).

| Query | variant | off (ms) | on (ms) | ratio, pair 1 | ratio, pair 2 | ratio, full run |
|---|---|---|---|---|---|---|
| q40 | scalar | 6.613 | 5.030 | 0.76 | 0.76 | 0.76 |
| q40 | x86-64-v3-auto | 4.552 | 3.642 | 0.80 | 0.80 | 0.81 |
| q40 | x86-64-v3-intrinsics | 4.102 | 3.226 | 0.79 | 0.77 | 0.80 |
| q40 | x86-64-v4-auto | 3.954 | 3.324 | 0.84 | 0.85 | 0.86 |
| q40 | x86-64-v4-intrinsics | 3.236 | 2.653 | 0.82 | 0.81 | 0.83 |
| q41 | scalar | 5.761 | 4.324 | 0.75 | 0.75 | 0.77 |
| q41 | x86-64-v3-auto | 3.767 | 3.049 | 0.81 | 0.79 | 0.81 |
| q41 | x86-64-v3-intrinsics | 3.378 | 2.669 | 0.79 | 0.76 | 0.79 |
| q41 | x86-64-v4-auto | 3.331 | 2.742 | 0.82 | 0.82 | 0.84 |
| q41 | x86-64-v4-intrinsics | 2.674 | 2.189 | 0.82 | 0.79 | 0.80 |
| q42 | scalar | 4.479 | 3.337 | 0.74 | 0.75 | 0.74 |
| q42 | x86-64-v3-auto | 2.750 | 2.155 | 0.78 | 0.76 | 0.78 |
| q42 | x86-64-v3-intrinsics | 2.607 | 1.940 | 0.74 | 0.75 | 0.73 |
| q42 | x86-64-v4-auto | 2.409 | 1.953 | 0.81 | 0.81 | 0.81 |
| q42 | x86-64-v4-intrinsics | 2.064 | 1.610 | 0.78 | 0.78 | 0.80 |
| TPC-H q06 | scalar | 5.662 | 5.692 | 1.01 | 1.00 | 1.01 |
| TPC-H q06 | x86-64-v3-auto | 2.898 | 2.944 | 1.02 | 1.01 | 1.02 |
| TPC-H q06 | x86-64-v3-intrinsics | 2.461 | 2.491 | 1.01 | 1.00 | 1.03 |
| TPC-H q06 | x86-64-v4-auto | 2.338 | 2.403 | 1.03 | 1.02 | 1.04 |
| TPC-H q06 | x86-64-v4-intrinsics | 1.611 | 1.651 | 1.03 | 1.03 | 1.05 |

All 15 variants of the three ClickBench queries leave the 10% band in the first pair, and every one reproduces in the second pair and in the full run (ratios 0.73 to 0.86).
The gain is largest on the scalar and AVX2 variants (0.73 to 0.81) and smallest on the AVX-512 variants (0.78 to 0.86). A possible reason is that AVX-512 comparisons cost less per row, so less is left to save, but this record did not test that.
TPC-H q06 stays inside the band on every variant in all three runs, at 1.00 to 1.05, with the AVX-512 variants consistently the highest, which is the small per-tile cost described above.

The best variant of each of the four queries is `x86-64-v4-intrinsics` in both columns.
Ratios below 1.0 mean Verkehrt is faster, and the before and after columns use that variant's first-pair means.

| Query | DuckDB 1t (ms) | corgi (ms) | off / DuckDB | on / DuckDB | off / corgi | on / corgi |
|---|---|---|---|---|---|---|
| q40 | 4.31 | 5.36 | 0.75 | 0.62 | 0.60 | 0.50 |
| q41 | 3.83 | 4.93 | 0.70 | 0.57 | 0.54 | 0.44 |
| q42 | 7.08 | 8.29 | 0.29 | 0.23 | 0.25 | 0.19 |
| TPC-H q06 | 4.74 | 4.36 | 0.34 | 0.35 | 0.37 | 0.38 |

The DuckDB and corgi times are reused, not re-run: the ClickBench figures come from `2026-10-31-clickbench-now.md` and the TPC-H q06 figures from `2026-10-31-tier2.md`.
Verkehrt's times in those records come from builds before tiling and the term graph, so the ratios here are not a before and after of this layer alone, and the off column of this table is already faster than the build of those records.
The change this layer contributes to each ratio is the difference between the off and on columns: 0.13, 0.13 and 0.06 against DuckDB and 0.10, 0.10 and 0.06 against corgi for q40, q41 and q42, and a rise of 0.01 against both for TPC-H q06.

## The full suites

Ratios of the best variant over all 22 queries, from `bench-full-off.txt` and `bench-full-on.txt` (first pair).
`narrowed_tiles` was recorded only for `q03` and TPC-H q01 among the queries outside the four mask chains, so this record does not say which other queries contain a narrowing guard.

| Query | off best (ms) | on best (ms) | on / off | variants more than 10% faster | variants more than 10% slower |
|---|---|---|---|---|---|
| q40 | 3.170 | 2.638 | 0.83 | 5 of 5 | 0 |
| q41 | 2.677 | 2.135 | 0.80 | 5 of 5 | 0 |
| q42 | 2.047 | 1.631 | 0.80 | 5 of 5 | 0 |
| q03 | 1.410 | 2.180 | 1.55 | 0 | 1 (unresolved) |
| TPC-H q01 | 16.704 | 16.913 | 1.01 | 0 | 1 |
| TPC-H q06 | 1.594 | 1.670 | 1.05 | 0 | 0 |
| other 16 queries | | | 0.95 to 1.07 | 0 | 0 |

The best-variant ratio of the other 16 queries runs from 0.95 (q08) to 1.07 (q27), and no variant of them leaves the band, though three variants of q27 sit at 1.07 to 1.10.
Fifteen of 110 pairs run more than 10% faster, all of them the 15 variants of q40, q41 and q42.
Two pairs run more than 10% slower in the first pair, and the second pair repeated both.

* `q03` on `x86-64-v4-intrinsics`: 1.410 ms off against 2.185 ms on in the first pair (1.55), but 2.566 off against 2.544 on (0.99) in the repeat. The other four variants of q03 are level (1.00 to 1.01 in the first pair), the off figure of 1.410 ms sits 36% below the other q03 variants of its pair and 45% below the repeat's off figure, and the two pairs disagree. This record calls it unresolved and does not explain it away. `stats-other-on.txt` shows `narrowed_tiles` 0 for q03, so no tile narrowed and the executor's path matches the off run's.
* TPC-H q01 on `x86-64-v3-auto`: 19.680 off against 22.063 on (1.12), and 19.718 against 21.920 in the repeat (1.11). It reproduces, and the other four variants stay within 1.05 in both pairs. `narrowed_tiles` is 0 for q01 on both builds in `stats-other-on.txt`, so no narrowing ran, and the kernel totals on `x86-64-v3` point the other way (18,553.2 us on against 20,917.9 us off in one stats run). The evidence does not say what causes the criterion difference, which sits 1 to 2 points beyond the band, and it is not established as an effect of narrowing.

`q27` was unstable across pairs in the tiled-regions record, and here it stays inside the band (1.01 to 1.10 across variants in one pair) and was not repeated.

## What remains

The measured premise is that comparison time falls with the rows evaluated, and it holds on q40, q41 and q42 for ranges that are contiguous because `CounterID == 62` keeps long runs of rows.
The spec's rule makes selection vectors for TPC-H q06 the next step on this evidence, and the evidence supports that step as a premise and not as a result: q06's comparison rows are 0.998 of the whole and its comparison time did not fall, as the simulation predicted.
If a selection-vector comparison on q06 cost time per row like the ClickBench comparisons do (0.52 to 0.60 of the time for 0.50 to 0.53 of the rows, so 1.02 to 1.12 times per row), its 0.40 of the rows would take about 0.41 to 0.45 of 970 us, which saves about 530 to 570 us of its 1961 us kernel total.
That is an upper bound by inference, because the ClickBench rows are contiguous ranges that call the existing kernel entries on a slice, while selection vectors read scattered rows, and this record measured no gathered comparison.
The spec's simulation puts the q06 rows that pass the shipdate range across the tiles, so the cost of a gather or compress before each comparison decides whether the 0.40 turns into time, and this record did not measure it.
Beyond q06, `eq_i64` on q40, q41 and q42 still reads each column for its range, and `compress_i64`, the sorts and the second reads of `EventDate` and `TraficSourceID` (issue #39) are untouched by this layer.
