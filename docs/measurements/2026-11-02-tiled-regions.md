# Tiled elementwise regions: measurements

This document measures the effect of running chains of elementwise kernels in tiles on the mask-chain queries of issue #16, and picks the default tile size.
Tiling lowers the best variant of TPC-H q06 from 1.90 to 1.60 ms (0.84 times, from one pair of runs) and ClickBench q40, q41 and q42 by 4% to 7%, which is inside the noise band for those three.
The default tile is 16384 rows, because 1024 rows is 22% slower than untiled and the time does not improve measurably beyond 16384 rows.
Across both suites, 21 of 110 query and variant pairs run more than 10% faster in the first pair of runs, 18 of them reproduce in a second pair (q03, q30, q31, and three variants of TPC-H q01), the two AVX-512 variants of TPC-H q06 were not repeated, and none run more than 10% slower in the first pair.
The largest gains are on queries that are not mask chains (ClickBench q30, q31 and q03, TPC-H q01).
The before numbers come from the same build with tiling off (`VERKEHRT_TILE_ROWS=off`), and the DuckDB and corgi figures are reused from earlier records and were not re-run.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | the branch with tiled regions, in which `Options::tiling` chooses `Tiling::Off` or `Tiling::Rows(n)`. The acceptance bench and the stats example read `VERKEHRT_TILE_ROWS` (`off` or a row count) to override the default. The tile size sweep ran with the default of 4096 rows replaced by the environment variable, and the full runs ran on the build whose default is 16384 rows |
| Before | the same build with `VERKEHRT_TILE_ROWS=off`, so only the tiling differs |
| corgi and DuckDB | the figures of `2026-10-31-clickbench-now.md` (q40, q41, q42) and `2026-10-31-tier2.md` (TPC-H q06), reused and not re-run |
| Data | `hits_0.parquet` (1,000,000 rows) and the TPC-H data, the prepared data directories of the earlier rounds |

The method is the one in the "How the systems were timed" section of `2026-10-31-clickbench-now.md`.
Verkehrt times are criterion means with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time.
Per-kernel totals come from `acceptance_stats` on `x86-64-v4` with the default backend (best of five fresh runs).
Before each timed run the 1-minute load had to be at or below 4, and every evidence file records `uptime` at its start and end.
Differences within about 10% count as noise.

### Evidence files

All files are in `2026-11-02-tiled-regions/`. Loads are the 1-minute average at the start and the end, and the bench itself adds about 1 to the end value.

* `stats.txt`: per-kernel totals of q40, q41, q42 and TPC-H q06 untiled and with 16384-row tiles. Load 2.49 to 2.38.
* `stats-other.txt`: the same for q03, q30, q31 and TPC-H q01. Load 2.95 to 2.68.
* `bench-tile-off.txt`, `bench-tile-1024.txt`, `bench-tile-4096.txt`, `bench-tile-8192.txt`, `bench-tile-16384.txt`, `bench-tile-32768.txt`, `bench-tile-65536.txt`: the tile size sweep over q40, q41, q42 and TPC-H q06, five variants each. Loads at the start and end: off 2.08 to 3.58, 1024 3.60 to 3.48, 4096 2.04 to 4.07, 8192 2.56 to 3.96, 16384 3.48 to 2.48, 32768 3.96 to 2.38, 65536 3.51 to 3.18.
* `bench-full-off.txt` and `bench-full-default.txt`: every acceptance query, five variants, untiled and with the default tile. Load 2.54 to 2.78 and 2.49 to 4.30.
* `bench-repeat-off.txt` and `bench-repeat-default.txt`: a second pair for q03, q30, q31 and TPC-H q01. Load 2.58 to 3.12 and 2.15 to 2.58.
* `bench-q27-off.txt` and `bench-q27-default.txt`: a second pair for ClickBench q27. Load 2.65 to 2.59 and 2.59 to 2.94.
* `correctness.txt`: `cargo test --release --test acceptance` with both data directories set, which passes (17 passed, 0 failed, 1 ignored) on the default tile. No load was recorded for this run.

The first untiled sweep, two 65536 sweeps, and the first q27 pair were repeated because the load rose above 6 during them, and their logs are not kept.

## Where the time goes

A comparison reads an `i64` column at 8 bytes per row and writes a `bool` mask at 1 byte per row, so a chain of comparisons and `and_bool` kernels writes and rereads one mask per step when it runs on whole arrays.
Tiling keeps those masks in 16384-row buffers, and the stats show where that pays.
The totals are in us, the first number untiled and the second tiled.

| Query | `eq_i64` | `le_i64` or `lt_i64` | `and_bool` | those three, share of total | total kernel time |
|---|---|---|---|---|---|
| q40 | 1282.1 to 1196.0 | 451.2 to 343.0 | 261.1 to 126.6 | 49.2% to 44.5% | 4054.5 to 3744.1 |
| q41 | 1154.1 to 1035.2 | 445.0 to 316.5 | 271.9 to 123.6 | 54.6% to 49.5% | 3424.4 to 2981.8 |
| q42 | 861.9 to 768.3 | 476.6 to 315.8 | 228.4 to 90.8 | 57.4% to 49.8% | 2729.1 to 2359.2 |
| TPC-H q06 | none | 890.8 to 639.8 (`le`), 586.7 to 378.8 (`lt`) | 306.1 to 112.1 | 70.5% to 60.7% | 2530.1 to 1863.3 |

The `and_bool` kernels get 2.1 to 2.7 times faster in q40, q41, q42 and q06, and `le_i64` and `lt_i64` drop by 24% to 35%, which fits the masks staying in cache but was not tested.
`eq_i64` drops by only 7% to 11%. The spec predicted that tiling would remove about half of the chain's bytes moved, and the kernel totals fall by 8% to 14% on the ClickBench queries against a chain share of 49% to 57%, so the prediction did not hold. A comparison reads 8 bytes per row and writes 1 byte per row, so the mask traffic that tiling removes is a small part of the bytes of the chain, and the column reads dominate. This document did not measure bytes moved or cache misses, so it does not establish why `eq_i64` gains least, and whether the repeated reads of `EventDate` and `TraficSourceID` are served from cache inside a tile is unknown.
q40 and q41 keep their sort (`argsort_i64` is 1010 us of q40 and does not change) and their `compress_i64` calls, which are not part of any region.
The kernel total falls by 8%, 13%, 14% and 26%, and the criterion means of the best variant fall by less, 4%, 6%, 7% and 16%, which fits that the stats timer and criterion do not measure the same span.

## Tile size

Criterion means in ms of `x86-64-v4-intrinsics`, and the ratio to untiled over all 20 query and variant rows of the sweep.

| tile rows | q40 | q41 | q42 | TPC-H q06 | geomean ratio | worst ratio | best ratio |
|---|---|---|---|---|---|---|---|
| off | 3.73 | 2.97 | 2.29 | 2.12 | 1.000 | 1.00 | 1.00 |
| 1024 | 4.54 | 3.88 | 3.03 | 2.36 | 1.225 | 1.33 | 1.08 |
| 4096 | 3.39 | 2.82 | 2.17 | 1.70 | 0.977 | 1.06 | 0.80 |
| 8192 | 3.41 | 2.83 | 2.14 | 1.61 | 0.943 | 1.03 | 0.76 |
| 16384 | 3.33 | 2.70 | 2.05 | 1.60 | 0.924 | 1.01 | 0.75 |
| 32768 | 3.35 | 2.75 | 2.05 | 1.60 | 0.921 | 1.00 | 0.76 |
| 65536 | 3.31 | 2.73 | 2.08 | 1.66 | 0.923 | 0.99 | 0.78 |

The 1-minute gate does not cover a recent busy period: the 4096, 1024 and 16384 sweeps started with a 15-minute load of 9.00, 8.38 and 7.76, the untiled sweep with 3.31, and the sweeps ran once each in that order, not interleaved.
The untiled run of this sweep ran above the untiled run of `bench-full-off.txt` on this variant by 7% for q40 (3.73 against 3.48 ms), 6% for q41, 3% for q42 and 12% for TPC-H q06 (2.12 against 1.90 ms), so the sweep's ratios to untiled are too low by up to that much, for example 0.75 for q06 against 0.84 in the full run.
The comparison between tile sizes is better controlled but ran once per size without interleaving, so differences below the 10% band, such as 4096 against 16384, are indicative only.
1024 rows loses on every row (1.08 to 1.33 times untiled), and 4096 rows is slower than or level with untiled on the scalar and `x86-64-v3` variants of q40 and q42 (worst ratio 1.06).
From 16384 rows the geomean stays between 0.921 and 0.924, and each row differs by at most 4% between 16384, 32768 and 65536 rows.
The default is 16384 rows. The evidence cannot separate it from 8192 (geomean 0.943 against 0.924, the larger gap on q41 and q42 AVX-512 rows at 5% to 6%), but 16384 rows is the smallest size with every row within 4% of the larger sizes, so one `i64` tile is 128 KiB and a region holds only a few tiles of intermediates at once.
The sweep did not test sizes between the listed ones, so the true optimum on the plateau is not known.

## Results

Times are criterion means in ms for the five variants, untiled then tiled, from `bench-full-off.txt` and `bench-full-default.txt`.
The ratio is tiled over untiled, and values below 1.0 mean tiling is faster.

| Query | variant | untiled | tiled | tiled / untiled |
|---|---|---|---|---|
| q40 | scalar | 6.578 | 6.641 | 1.01 |
| q40 | x86-64-v3-auto | 4.559 | 4.475 | 0.98 |
| q40 | x86-64-v3-intrinsics | 4.211 | 4.153 | 0.99 |
| q40 | x86-64-v4-auto | 4.253 | 3.985 | 0.94 |
| q40 | x86-64-v4-intrinsics | 3.479 | 3.347 | 0.96 |
| q41 | scalar | 5.611 | 5.631 | 1.00 |
| q41 | x86-64-v3-auto | 3.846 | 3.840 | 1.00 |
| q41 | x86-64-v3-intrinsics | 3.562 | 3.549 | 1.00 |
| q41 | x86-64-v4-auto | 3.451 | 3.281 | 0.95 |
| q41 | x86-64-v4-intrinsics | 2.808 | 2.650 | 0.94 |
| q42 | scalar | 4.478 | 4.424 | 0.99 |
| q42 | x86-64-v3-auto | 2.733 | 2.768 | 1.01 |
| q42 | x86-64-v3-intrinsics | 2.694 | 2.634 | 0.98 |
| q42 | x86-64-v4-auto | 2.465 | 2.379 | 0.97 |
| q42 | x86-64-v4-intrinsics | 2.222 | 2.067 | 0.93 |
| TPC-H q06 | scalar | 5.766 | 5.619 | 0.97 |
| TPC-H q06 | x86-64-v3-auto | 3.014 | 2.916 | 0.97 |
| TPC-H q06 | x86-64-v3-intrinsics | 2.674 | 2.587 | 0.97 |
| TPC-H q06 | x86-64-v4-auto | 2.671 | 2.342 | 0.88 |
| TPC-H q06 | x86-64-v4-intrinsics | 1.899 | 1.595 | 0.84 |

Among these four queries only the two AVX-512 variants of TPC-H q06 leave the noise band (0.88 and 0.84), from one pair of runs each.
The three ClickBench mask-chain queries move by 7% at most, and their scalar and AVX2 variants move by 2% at most.
The stats come from the AVX-512 build only, so this document does not say why the other builds gain less.

The best variant of each of the four queries is `x86-64-v4-intrinsics` in both columns.
Ratios below 1.0 mean Verkehrt is faster, and the before and after columns use that variant.

| Query | DuckDB 1t (ms) | corgi (ms) | untiled / DuckDB | tiled / DuckDB | untiled / corgi | tiled / corgi |
|---|---|---|---|---|---|---|
| q40 | 4.31 | 5.36 | 0.81 | 0.78 | 0.65 | 0.62 |
| q41 | 3.83 | 4.93 | 0.73 | 0.69 | 0.57 | 0.54 |
| q42 | 7.08 | 8.29 | 0.31 | 0.29 | 0.27 | 0.25 |
| TPC-H q06 | 4.74 | 4.36 | 0.40 | 0.34 | 0.44 | 0.37 |

The q40, q41 and q42 reference times are the one-thread DuckDB and best-of corgi figures of `2026-10-31-clickbench-now.md`, and the TPC-H q06 figures are those of `2026-10-31-tier2.md`.
Verkehrt's own times for q40 and q41 are lower than in `2026-10-31-clickbench-now.md` because other changes landed in between, so the ratios here are not a before and after of this layer alone.
The tiled-over-untiled change of each ratio is what this layer contributes, and it is 0.02 to 0.04 for the three ClickBench queries and 0.06 to 0.07 for TPC-H q06.

## The full suites

Ratios of the best variant over all 22 queries, from `bench-full-off.txt` and `bench-full-default.txt`.
No variant of any query ran more than 10% slower with tiling in the first pair (the second q27 pair has one variant at 1.18, discussed below).

| Query | untiled best | tiled best | tiled / untiled | variants more than 10% faster |
|---|---|---|---|---|
| q03 | 2.542 | 2.171 | 0.85 | 5 of 5 |
| q30 | 5.598 | 4.363 | 0.78 | 5 of 5 |
| q31 | 6.511 | 5.122 | 0.79 | 5 of 5 |
| TPC-H q01 | 18.054 | 14.696 | 0.81 | 4 of 5 |
| TPC-H q06 | 1.899 | 1.595 | 0.84 | 2 of 5 |
| other 17 queries | | | 0.93 to 1.09 | 0 of 85 |

Every ClickBench query other than q03, q30, q31 and the three mask chains is within 5% of untiled in its best variant, except q27 (1.09).
The q27 ratio is not a tiling effect on this evidence: in the second pair (`bench-q27-off.txt`, `bench-q27-default.txt`) the ratios run from 0.82 to 1.18 across variants, and the same variant moves by more than 10% between pairs.
The faster queries reproduce in the second pair: q03 0.82 to 0.89, q30 0.76 to 0.83, q31 0.76 to 0.82, and TPC-H q01 0.82 to 0.87 on the AVX variants.
TPC-H q01 on `scalar` (0.87 then 0.97) and `x86-64-v3-auto` (1.01 then 0.85) disagree between pairs, so only its other three variants count as faster.

`stats-other.txt` shows the kernels behind the larger gains, though not why they are faster.
In q30 `sub_i64` falls from 1062.5 to 351.0 us with the same three calls, and in q03 `sub_i64` falls from 884.5 to 322.6 us and `mul_i64` from 788.9 to 205.5 us, while `div_const_i64` rises from 800.7 to 1351.8 us.
The stats timer includes each kernel's output allocation, so a region's full-size output allocation lands on its last kernel, which fits the rise of `div_const_i64`, but this document did not measure allocation separately.
Total kernel time falls from 2686.8 to 2067.6 us in q03, 5899.0 to 4682.6 us in q30, 6620.3 to 5380.5 us in q31, and 17667.7 to 13990.5 us in TPC-H q01.

## What remains

On the mask-chain queries the chain is no longer the largest cost, but it is still 44% to 61% of kernel time because `eq_i64`, `le_i64` and `lt_i64` read their columns from memory once per use.
In q40 `EventDate` and `TraficSourceID` are each read twice, and a shared read (issue #39) or a fused predicate over the tile (issue #30) would remove that traffic, but neither was measured here.
The sort and the `compress_i64` calls of q40 and q41 stay outside any region and dominate what is left, 1010 us of `argsort_i64` and 581 us of `compress_i64` in q40.
