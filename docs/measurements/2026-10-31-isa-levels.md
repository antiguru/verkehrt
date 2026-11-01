# The x86 levels: Casts, `mul`, byte compress, and the suites against the old ISAs

This document measures the change from the `avx2` and `avx512` instruction sets to the psABI levels `x86-64-v3` and `x86-64-v4`.
The levels widen the feature strings: `x86-64-v3` adds BMI1, BMI2, FMA, F16C, LZCNT, MOVBE and more to AVX2, and `x86-64-v4` adds AVX-512 CD and DQ to F, BW, and VL.
The measurements answer three questions: whether the DQ instructions now appear in the `auto` entries of the casts and `mul_i64`, what the byte `compress` costs without VBMI2, and which kernels or queries got slower under the wider strings.
Every number comes from one run group on the x86 host below, with the baseline build `6033fe9` (main before this branch) and the new build run interleaved in the same session.
Six rows are listed under "Regressions" and one scalar-entry shift beside them. Every row except `compress_u8` and TPC-H q06 at `x86-64-v3-auto` lies inside the build-to-build shift that scalar entries show (see "Build-to-build shift"), and the cause of none is established.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2, AVX-512 F, BW, CD, DQ, VL, VBMI2 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, not recorded during the runs, the sandbox cannot read cpufreq) |
| Kernel, rustc | 7.2.11+deb14-amd64, 1.99.0 |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Baseline | `6033fe9`, worktree built with `cargo build --release` and `cargo bench --no-run`, ISAs named `avx2` and `avx512`, AVX-512 string `avx512f,avx512bw,avx512vl,avx512vbmi2` |
| New build | this branch before its final review fixes, ISAs `x86-64-v3` and `x86-64-v4`. Those fixes rename the per-level functions from `avx2` and `avx512` to `v3` and `v4`, register the VBMI2 entry through a constructor, and change comments, tests and docs, but no kernel loop |
| Data | ClickBench `hits_0` and TPC-H SF 0.2 column files from `~/dev/repos/data/verkehrt/{clickbench,tpch}`, as in `2026-10-31-tier2.md` |

Other work shared the host, so each run records its load average (1 minute, 5 minute, 15 minute) at start and end in the raw files.
The 1-minute load at the start of the bench runs ranged from 1.96 to 9.52, and the 65 cast and compress runs started at 3.23 to 3.51.
Ratios compare the baseline and new build runs of the same group, which ran minutes apart, and absolute times vary with load.

### How the runs were made

* Casts, `mul`, and compress: the program of `2026-10-31-isa-levels/casts.vk` (and `compress-u8.vk`) run with `verkehrt run FILE --stats --isa ISA`, five times per configuration, in rounds that interleave baseline `scalar`, new `scalar`, `avx2`, `x86-64-v3`, `avx512`, `x86-64-v4` (and for compress, the `--disable avx512vbmi2` run). The tables give the median of the per-kernel `time` of the `--stats` line, with the minimum and maximum of the five runs. The raw output of every run is `casts-and-compress-raw.txt`.
* `tier1` and `tier2` groups, and the acceptance queries: the criterion settings of `2026-10-31-tier2.md` (`sample_size(10)`, 0.5 s warm-up, 3 s measurement), baseline and new run alternately, at least twice per row (`bench-{base,new}-{tier1,tier2,acceptance}-{a,b}.txt`). The rows flagged by the first two passes got two further passes (`bench-*-flagged-{c,d}.txt` for `tier1/argmin` and four tier 2 groups, `bench-*-acceptance-{c,d}.txt` for the ClickBench q01, q02, q42 and TPC-H rows). The crossover sweep ran twice (`bench-*-crossover-{a,b}.txt`).
* Comparison: `comparison-tier1-tier2-acceptance.txt` lists every row with the criterion means of all passes of each side in order, and the ratio of the best new mean over the best baseline mean. The baseline IDs `avx2-*` and `avx512-*` map to `x86-64-v3-*` and `x86-64-v4-*`. "Within the spread" below means the ranges of the baseline and new means overlap, or the difference is under 3%.
* Assembly: `objdump -d --no-show-raw-insn -C` of the new build's release `tier1` bench binary, locating the entries `auto::x86_64_v4::elementwise::<Unary<S, D, CastOp, false>>` and `<Binary<i64, i64, MulOp, false>>`. `assembly-verdicts.txt` has the instruction counts, the vector loop excerpts, and the full conversion loop of the two float to integer casts (instruction subset, with the loop bounds and counts). The disassembly listings themselves are not in the repository.

The timed region of the `--stats` runs includes the output allocation and its zeroing, so the casts and `mul` are memory-bound at 10M rows and differences of a few percent are noise.

## Casts and `mul_i64`

The program (`casts.vk`) generates 10M rows with `rand_i64`, `cast_u64`, and two `rand_f64` arrays, so the `f64` to `i64` inputs span both signs and the `f64` to `u64` inputs are non-negative, all within range.
Each cast and `mul` result is consumed by a `sum` or `reduce(..., min)`, and `let m = x * x` is used twice so that `mul_i64` stays its own kernel instead of fusing into `sumsq`.
All 30 cast runs printed one identical output across both builds and all configurations, and so did all 35 compress runs (checked over `casts-and-compress-raw.txt`).

Kernel time in ms, median with the five-run range in parentheses.

| Kernel | baseline scalar | baseline `avx2` | baseline `avx512` | new scalar | `x86-64-v3` | `x86-64-v4` |
|---|---|---|---|---|---|---|
| `cast_i64_f64` | 8.35 (8.14 to 8.62) | 8.15 (8.12 to 8.34) | 8.31 (7.91 to 8.37) | 7.92 (7.76 to 8.36) | 7.77 (7.35 to 7.84) | 7.46 (7.03 to 7.72) |
| `cast_u64_f64` | 7.71 (7.15 to 8.43) | 7.56 (7.39 to 7.94) | 7.50 (7.32 to 7.80) | 7.40 (7.34 to 7.44) | 7.48 (7.14 to 7.60) | 7.12 (6.92 to 7.88) |
| `cast_f64_i64` | 9.87 (9.28 to 11.47) | 9.68 (9.45 to 9.88) | 9.91 (9.63 to 10.50) | 9.38 (8.99 to 9.47) | 9.51 (9.16 to 10.31) | 10.84 (10.59 to 12.03) |
| `cast_f64_u64` | 13.27 (13.22 to 13.68) | 12.43 (12.04 to 12.63) | 11.11 (11.04 to 11.54) | 13.21 (13.13 to 13.47) | 12.01 (11.82 to 13.36) | 12.49 (11.83 to 14.53) |
| `mul_i64` | 7.75 (7.60 to 7.91) | 7.57 (7.28 to 7.72) | 7.02 (6.96 to 7.78) | 7.63 (7.40 to 8.38) | 7.14 (6.87 to 7.90) | 7.36 (7.22 to 7.80) |

### Assembly verdicts at `x86-64-v4`

| Kernel | Verdict | Key instructions in the hot loop |
|---|---|---|
| `cast_i64_f64` | Vector loop | `vcvtqq2pd` with memory source on `zmm`, 4 per iteration (32 rows), `vmovupd zmm` stores |
| `cast_u64_f64` | Vector loop | `vcvtuqq2pd` on `zmm`, 4 per iteration (32 rows) |
| `cast_f64_i64` | Scalar loop | 32 `vcvttsd2si` per iteration of the unrolled loop (one per row, 46 in the symbol), assembled into `zmm` with `vinserti64x4` and stored with 4 `vmovdqu64 zmm`. `vcvttpd2qq` is not emitted |
| `cast_f64_u64` | Scalar loop | 32 `vcvttsd2usi` per iteration of the unrolled loop (the scalar form, 46 in the symbol), assembled into `zmm`. `vcvttpd2uqq` is not emitted |
| `mul_i64` | Vector loop | `vpmullq` on `zmm` with one memory operand and the other source in `zmm0`, 4 per iteration (32 rows) |

At `x86-64-v3` the `auto` entries have no DQ instruction available.
`cast_i64_f64` and `cast_f64_i64` there are scalar conversions per row (26 `vcvtsi2sd` and 6 `vcvttsd2si` in the symbols), `cast_f64_u64` is scalar (`vcvttsd2si` with fixups), `cast_u64_f64` is vectorized by the magic-constant split on `ymm` (`vsubpd`, `vaddpd`), and `mul_i64` is the `vpmuludq` emulation on `ymm` (36 `vpmuludq`).
The `tier2` document recorded the same verdicts for the old strings, with scalar conversions for all three float to integer and `i64` to `f64` casts and `vpmuludq` emulation for `mul` at both ISAs.

### What changed

* `cast_i64_f64` at `x86-64-v4` is a vector loop (`vcvtqq2pd`), and its times (7.03 to 7.72 ms) are 6% below the new scalar entry's median (7.46 against 7.92 ms) with non-overlapping five-run ranges (7.76 to 8.36 ms for scalar). That margin is smaller than the build-to-build shift of 6% to 14% seen on scalar entries the branch did not touch (see "Build-to-build shift"), and the baseline scalar entry of the same kernel is 5% slower than the new one (8.35 against 7.92 ms) with no code change. It therefore shows no gain beyond that shift and stays a strain case at both levels (scalar loop at `x86-64-v3`).
* `cast_u64_f64` at `x86-64-v4` now uses `vcvtuqq2pd`, but its time is within the spread of the scalar entry (6.92 to 7.88 against 7.34 to 7.44 ms), so the hardware instruction gives no measurable gain over the magic-constant split.
* `mul_i64` at `x86-64-v4` now uses `vpmullq`, and its time (7.22 to 7.80 ms) is within the spread of the new scalar entry (7.40 to 8.38 ms) and of the baseline emulation (6.96 to 7.78 ms). The kernel is memory-bound, so the instruction change is not visible in the interpreter.
* `cast_f64_i64` and `cast_f64_u64` stay scalar loops at both levels, and both are slower at `x86-64-v4` than at the old `avx512` (see "Regressions"). Why the `auto` body does not compile to `vcvttpd2qq` was not investigated.

## Byte `compress` and VBMI2

`compress-u8.vk` runs two byte compresses at 50% selectivity over 10M rows (`filter` on a `u8` array and `compress` on a `u8` array with an `i64` comparison mask), so each `--stats` line covers two calls.

| Configuration | `compress_u8` ms, median (range) |
|---|---|
| baseline scalar | 18.78 (18.28 to 19.45) |
| new scalar | 18.55 (18.14 to 19.98) |
| baseline `avx2` (`auto` entry) | 13.78 (13.22 to 14.37) |
| `x86-64-v3` (`auto` entry) | 17.16 (16.48 to 17.72) |
| baseline `avx512` (`vpcompressb` entry) | 2.43 (2.27 to 3.25) |
| `x86-64-v4` (`vpcompressb` entry) | 3.07 (2.24 to 3.62) |
| `x86-64-v4 --disable avx512vbmi2` (`auto` entry) | 11.85 (11.32 to 12.38) |

The new `x86-64-v4` VBMI2 entry runs at 3.07 ms (2.24 to 3.62) and its `auto` fallback with `--disable avx512vbmi2` at 11.85 ms (11.32 to 12.38), so disabling VBMI2 costs a factor of 3.86 on the medians and leaves the kernel at 0.64 of the new scalar entry's median (18.55 ms).
The `x86-64-v4` entry with VBMI2 is within the spread of the baseline `avx512` entry (the ranges 2.24 to 3.62 and 2.27 to 3.25 ms overlap).
The `x86-64-v3` row is the `auto` entry in both builds, since AVX2 has no byte compress, and the new one is 24% slower on the medians (17.16 against 13.78 ms, a regression, see below).
The disassembly of the two entries is the same (see the regressions table).

## Tier 1 and tier 2 suites

All 150 rows of the `tier1` and `tier2` groups and the acceptance queries are in `comparison-tier1-tier2-acceptance.txt`, with 108 rows that exclude the native loops.
Of those 108 rows, 80 are within 3% of the baseline (ratio 0.97 to 1.03 of the best means), 21 are faster by more than 3%, and 7 are slower by more than 3%.
The 7 slower rows are four rows of the regressions table (`argmin`, `gather-i64`, TPC-H q06, and ClickBench q42, all at `x86-64-v3-auto`), ClickBench q01 `scalar`, and the two TPC-H q01 rows at `x86-64-v4`, where the ranges of the baseline and new means overlap and the rows are within the spread.
The crossover sweep (68 rows, two passes each) has one row beyond 15% in either direction (`crossover/mul/interp/2048`, 0.7 to 1.6 microseconds), and no claim rests on it (`comparison-crossover.txt`).

### Faster, with the cause unmeasured

* `filter-1pct`, `filter-50pct`, `filter-99pct` intrinsics are 0.78 to 0.94 of the baseline at both levels, and ClickBench q07 intrinsics is 0.77 (`x86-64-v3`) and 0.72 (`x86-64-v4`) of the baseline. The native baseline loop of the same bench (`native-base`) moved by 0.76 to 0.86 between the builds in the three `filter` rows, and the scalar rows did not move, so the bench binary shows a build-to-build shift in code that this branch did not touch. The cause was not found, and the gains are not attributed to the levels. The shift applies in both directions (see "Build-to-build shift").
* `div-const-i64` at `x86-64-v4` is 0.96 of the baseline (8.47 to 8.66 against 8.89 to 9.07 ms) and `native-avx512` for `div-const-i64` ran at 5.56 ms against 10.0 ms. The second is a native loop in the bench, compiled under the v4 string with DQ, and it is not a kernel result.

### Build-to-build shift

Scalar entries and native loops that this branch did not change also moved between the two builds, which bounds what a row can show.
Over the 28 scalar rows of `comparison-tier1-tier2-acceptance.txt` the ratio of the best new mean to the best baseline mean ranges from 0.943 (`tier2/reduce-into-add-i64`) over 0.950 (`div-const-u64`), 0.955 (`prev-i64`), and 0.958 (`tier1/min`) to 1.139 (ClickBench q01), and the `cast_i64_f64` scalar entry is 5% faster in the new build (8.35 against 7.92 ms medians).
The `native-base` loops of the three `filter-*` rows, which this branch did not change, moved by 0.76 to 0.86. `div-const-i64/native-avx512` (0.56) is left out, because its feature string changed to the v4 string with DQ.
This document therefore treats a shift of -6% to +14% as the build-to-build shift, in both directions: a gain below 6% is not a gain, and a loss below 14% is not shown to come from the levels.

### Regressions

A row is listed when the new means of all passes are slower than the baseline means of all passes (no overlap) by more than 3%, at the level whose name replaced the old ISA.
Each row keeps its numbers and a verdict, and every row except `compress_u8` (+24%) and TPC-H q06 (+14%, see its verdict) lies inside the -6% to +14% range of the build-to-build shift, so none is established as caused by the levels.
The `x86-64-v4` `argmin` row (ratio 1.0296, +2.96%, which the comparison file rounds to 1.030, 6.31 to 6.43 ms against 6.49 to 6.67 ms) is below the 3% threshold and is not listed.

| Row | Baseline ms (passes) | New ms (passes) | Change | Verdict |
|---|---|---|---|---|
| `tier1/argmin` `x86-64-v3-auto` | 6.31, 6.31, 6.38, 6.30 | 6.77, 6.77, 6.86, 6.79 | +7% | Entry code identical, consistent with build/layout shift. The `update` entry for `argmin` on `f64` has an identical instruction stream in both builds (0x1a1 bytes, the disassemblies differ only in jump and call addresses). Scalar row unchanged (6.29 to 6.43 against 6.31 to 6.43) |
| `tier2/gather-i64` `x86-64-v3-auto` | 47.3, 47.5, 48.4, 47.4 | 49.8, 50.4, 50.4, 50.3 | +5% to +6% | Within the observed build-to-build shift. Scalar row loop behind a vectorized bounds pass in both builds. Entry grew from 0x7b7 to 0x7e4 bytes with the same instruction mix within three instructions. The `intrinsics` entry did not move (37.8 to 39.1 against 37.7 to 38.3) |
| `acceptance/tpch/q06` `x86-64-v3-auto` | 3.05, 3.14, 3.08, 3.11 | 3.51, 3.52, 3.50, 3.49 | +14% | Unexplained. The scalar q06 row in the same runs is unchanged (ratio 1.000 in `comparison-tier1-tier2-acceptance.txt`), so the shift on scalar code does not cover it, and the +14% upper bound of that shift comes from ClickBench q01 `scalar`, whose cause was not investigated either. Not disassembled. The plan is `dot_i64` over two `compress_i64` outputs behind comparison and `and_bool` kernels, run with the `auto` entries. The `x86-64-v3-intrinsics` row is faster (2.60 to 2.65 against 2.73 to 2.78) |
| `acceptance/clickbench/q42` `x86-64-v3-auto` | 2.86, 2.80, 2.88, 2.88 | 2.99, 3.10, 3.05, 3.07 | +6% to +8% | Within the observed build-to-build shift. Not disassembled. The scalar row moved by +2% in the same runs (4.44 to 4.53 against 4.55 to 4.56 ms) |
| `compress_u8` `x86-64-v3` (`auto`, casts-and-compress run) | 13.22 to 14.37 (median 13.78) | 16.48 to 17.72 (median 17.16) | +24% on the medians | Entry code identical, cause not established. The `auto` entry for `u8` is the same in both builds: both symbols are 0x46c bytes, and the disassemblies differ only in rip-relative and jump targets. The called helpers were not compared. The change exceeds the shift seen on scalar code, so this row is the least explained |
| `cast_f64_i64` and `cast_f64_u64` `x86-64-v4` | 9.63 to 10.50 and 11.04 to 11.54 | 10.59 to 12.03 and 11.83 to 14.53 | +9% and +12% (medians) | Scalar loop in both builds. Inside the observed shift. At `x86-64-v4` `cast_f64_i64` is also 16% slower than the new scalar entry (10.84 against 9.38 ms medians) |

Rows that moved but are not level regressions or are within the spread:

* ClickBench q01 `scalar`: 0.2232 to 0.2237 ms in the baseline and 0.2543 to 0.2548 ms in the new build, +14%, in all four passes. The scalar entry is not a level, and the same query at both levels did not move (`x86-64-v3` 0.0956 to 0.1023 against 0.0958 to 0.1026 ms). The cause was not investigated.
* `tier1/argmin` `x86-64-v4-auto`: 6.31 to 6.43 ms in the baseline and 6.49 to 6.67 ms in the new build, ratio 1.0296 (+2.96%), below the threshold and inside the shift.
* TPC-H q01 `x86-64-v4-auto` and `-intrinsics`: the baseline passes range 17.98 to 20.79 ms for `auto` (a 16% span) and 16.18 to 18.33 ms for the intrinsics variant (13%), and the new passes 19.52 to 20.82 ms and 18.02 to 18.84 ms. The sides overlap, so this is within the spread.
* ClickBench q02 `x86-64-v4-auto`: 0.175 to 0.184 ms against 0.175 to 0.189 ms, overlapping, within the spread.
* `tier2/scan-by-add-i64` `x86-64-v4-intrinsics`: 7.91 to 8.29 ms against 7.88 to 8.64 ms, within the spread.
* The ClickBench q42 `x86-64-v4-auto` (+2%) and `x86-64-v4-intrinsics` (+2%) rows are within the spread.

No kernel got a narrower feature string than its level, as the design requires, and none of the regressions blocks the change.
The `compress_u8` `x86-64-v3` row, outside the build-to-build shift, is tracked as issue #25. The TPC-H q06 `x86-64-v3-auto` row is unexplained, and a per-kernel trace of q06 is issue #16. The other rows lie inside the build-to-build shift and have no issue.

## Acceptance queries

The `x86-64-v4-intrinsics` times with all passes, in ms, baseline first: q01 0.090 to 0.099 and 0.091 to 0.098, q02 0.179 to 0.184 and 0.174 to 0.190, ClickBench q06 0.185 to 0.202 and 0.184 to 0.202, q07 0.454 to 0.472 and 0.327, q42 2.22 to 2.43 and 2.27 to 2.37, TPC-H q01 16.2 to 18.3 and 18.0 to 18.8, TPC-H q06 2.05 to 2.26 and 1.94 to 2.13.
In the baseline, `avx512-auto` was the fastest variant for q01 and q02 and tied with the intrinsics variant for ClickBench q06, while the intrinsics variant was the fastest for q07, q42, and both TPC-H queries.
The speedup on q07 comes from the filter intrinsics named above, the cause of which is unmeasured.
Every `x86-64-v4-intrinsics` query is within the spread of its baseline `avx512` time or faster.

## Verdict summary

* `cast_i64_f64`: vector loop at `x86-64-v4` (`vcvtqq2pd`), no gain beyond the build-to-build shift, so it stays a strain case. Scalar loop at `x86-64-v3`.
* `cast_u64_f64`: vector loop at both levels. At `x86-64-v4` it uses `vcvtuqq2pd`, no measurable gain over the magic-constant split.
* `cast_f64_i64`, `cast_f64_u64`: scalar loop at both levels, slower at `x86-64-v4` than at the old `avx512`.
* `mul_i64`: `vpmullq` vector loop at `x86-64-v4`, `vpmuludq` emulation at `x86-64-v3`, no measurable time difference.
* Byte `compress`: the VBMI2 entry is 3.86x faster than the `auto` fallback at `x86-64-v4` (medians), and the `x86-64-v3` `auto` entry is 24% slower than before with an unchanged entry.
