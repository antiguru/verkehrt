# Tier 1 kernels on the Apple M5: Measurements

This document compares Verkehrt's tier 1 kernels on aarch64 against native Rust loops at 10M elements, adds the float `min` and `argmin` reductions, and sweeps the vector length to find where dispatch overhead stops dominating.
The spec's target is throughput within 1.5x of a native Rust baseline for `sum`, `sum(x * y)`, `count(x > c)`, and `filter(x, |v| v > c)`, and this document applies the same 1.5x bound to `min` and `argmin`.
The NEON intrinsics backend is new on this branch for the selections (`filter`, `compress`, `where`) and the float extreme reductions, and it is measured here for `filter`, `min`, and `argmin`.
The first intrinsics attempt (commit `bbef3bb` "Add NEON intrinsics for float extreme reductions" for the extremes, `13a5e98` "Add NEON intrinsics for compress and fused filter" for the selections) failed the target for `filter` and `argmin`, and three later commits (`1ca388a` "Step the NEON selections four vectors at a time", `eb1b552` "Split the NEON argmin and argmax into four accumulators", `1b46b65` "Keep the NEON four-vector compress step free of calls") changed the loops, so both sets of numbers appear below.
Every number in the final tables comes from one run of the commands in the setup section, so run-to-run variation is not quantified beyond the reruns noted in the text.

## Setup

| Item | Value |
|---|---|
| CPU | Apple M5 Max (`sysctl -n machdep.cpu.brand_string`) |
| OS | macOS 27.2 (build 26B123) |
| rustc | 1.99.1 (58cf8f9c3 2026-10-13), Homebrew |
| Branch and commit | `kernels-neon`, benches at `1b46b65` "Keep the NEON four-vector compress step free of calls" plus `1a6263d` "Document the bench baselines' signed-zero and NaN limits" |
| Power | AC power, battery 100% charged, `powermode 2` (high power mode), sleep prevented by `caffeinate`, read with `pmset -g` and `pmset -g batt` just before the runs |
| Other work | None building. `top -l 1 -n 5 -o cpu` listed no process above 0.0% CPU (Chrome helpers, `localspeechrecog`), and the one-minute load average was 2.12 when read just before the final runs, and 1.51 before the earlier first-attempt run |
| SME | `hw.optional.arm.FEAT_SME` is 1, so SME is present. It is out of scope, and no kernel uses it |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |

`pmset -g batt` printed `Now drawing from 'AC Power'` and `-InternalBattery-0 (id=36503651) 100%; charged; 0:00 remaining present: true`.
The clock frequency was not recorded, so the numbers carry whatever clock the machine ran at, and ratios against native baselines in the same run share those conditions.

Commands:

```
CARGO_INCREMENTAL=0 cargo bench --locked --bench tier1
CARGO_INCREMENTAL=0 cargo bench --locked --bench corgi
```

The two benches ran separately and in the foreground.
Criterion ran with `sample_size(10)`, a 0.5 s warm-up, and a 3 s measurement time for the 10M workloads, and with `sample_size(20)`, 0.1 s warm-up, and 0.3 s measurement time for the crossover sweep.
All times and ratios come from criterion's median estimates.
Raw output: `bench-tier1-final.txt` (includes the crossover sweep) and `bench-corgi-final.txt`, kept in the session scratchpad and not committed.

## What the variants are

* `scalar`: the interpreter with `--isa scalar`. The scalar entries are portable Rust compiled for the baseline aarch64 target, which includes NEON, so the compiler may vectorize them. This is the baseline build, not a no-SIMD build.
* `neon-intrinsics`: the interpreter with `--isa neon --backend intrinsics`. The NEON `auto` slot is empty by decision, so this is the only NEON variant. It has entries for the selections and the float `min`, `max`, `argmin`, and `argmax`.
* For `sum`, `dot`, and `count` there is no NEON intrinsics entry, so `neon-intrinsics` resolves to the scalar entry and equals `scalar`. It is not benched there, and the crossover sweep for `sum` and `mul` runs the scalar entry for the same reason.
* `native-base`: plain Rust loops in `benches/tier1.rs`, compiled for the baseline target. No `native-neon` flavor exists, because the baseline target already includes NEON.
* Native `sum` and `dot` use the canonical 16-accumulator order with no FMA, and the bench asserts that every variant returns the same bits before timing.
* Native `min` keeps 16 running minima. Native `argmin` keeps 16 (value, index) pairs, replaces a pair only on a strictly smaller row, and merges by smaller value and then smaller index, which gives the first index. The bench asserts it against a plain forward loop with a strict `<`. Both baselines have no signed-zero order and no NaN rule, so they agree with Verkehrt only on data without NaN and `-0.0`, which the bench data is.
* Native `filter` is a predicated store into a zeroed `Vec` of capacity `n`, a single pass with no count pass.
* The interpreter timing is `Prepared::execute` only, prepared once. It includes cloning the input `Value`s, output allocation, and plan dispatch.
* Inputs are `rand_f64(10_000_000, 1)` and `rand_f64(10_000_000, 2)`. The first minimum is 2.5056049013372217e-8 at index 5618431. The comparison constants are 0.5 for `count` and 0.99, 0.5, and 0.01 for `filter`, with 99,695 (0.997%), 4,998,931 (49.989%), and 9,899,130 (98.991%) matching rows.
* `corgi`: the same workload in corgi at the revision DDIR pins (`benches/corgi.rs`), with a column as one row holding the whole column as a list. `min` is `input fold_min`, asserted equal to Verkehrt's result. Corgi has no `argmin` or `argmax` operator (its operator table in `src/frontend/mod.rs` has `fold_add`, `fold_mul`, `fold_min`, `fold_max`, `fold_all`, `fold_any`), so its `argmin` column is n/a (not in corgi). A composition of `fold_min`, `eq`, and `filter` would be a different algorithm and is not benched.
* Corgi's `sum` and `dot` differ from Verkehrt's in the last bits (sum 4998770.953740556 against 4998770.953739915, dot 2499047.034215136 against 2499047.034215335), and the `count`, `min`, and `filter` results are exact.

## Before: The plan 2 numbers

The user provided these on 2026-10-31, from the plan 2 benches on this M5 before the NEON intrinsics work.

| Workload | Verkehrt | native-base | ratio | corgi |
|---|---|---|---|---|
| sum | 0.834 ms | 0.821 ms | 1.02 | 5.47 ms |
| dot | 1.424 ms | 1.416 ms | 1.01 | 11.1 ms |
| count | 0.866 ms | 0.865 ms | 1.00 | 1.33 ms |
| filter 1% | 3.44 ms | 2.75 ms | 1.25 | 4.93 ms |
| filter 50% | 3.91 ms | 2.79 ms | 1.40 | 5.10 ms |
| filter 99% | 4.24 ms | 2.86 ms | 1.48 | 5.21 ms |
| copy (read and write) | 1.113 ms (133.9 GiB/s) | | | |

These "Verkehrt" figures are the scalar build, since no NEON entry existed yet.
`min` and `argmin` were not benched then.

## Memory bandwidth reference

A plain `copy_from_slice` of 10M `f64` (reads 80 MB, writes 80 MB) takes 1.104 ms in the final run, which criterion reports as 135.0 GiB/s of combined read and write traffic.
A read-only bandwidth was not measured separately.

## Results at 10M elements

The target column is the intrinsics ratio to `native-base` against 1.5x. "Intr / scalar" is the `neon-intrinsics` median divided by the `scalar` median, so below 1 means the intrinsics are faster.

### sum, dot, and count

| Workload | scalar | native-base | scalar / native | corgi | corgi / native |
|---|---|---|---|---|---|
| sum | 0.783 ms | 0.782 ms | 1.00 | 5.432 ms | 6.94 |
| dot | 1.438 ms | 1.413 ms | 1.02 | 10.977 ms | 7.77 |
| count | 0.820 ms | 0.817 ms | 1.00 | 1.309 ms | 1.60 |

`neon-intrinsics` equals `scalar` here (no entry), so these rows have no separate intrinsics result.
The scalar ratios match the before table (1.02, 1.01, 1.00) within 0.02.

### min (`reduce(x, min)`)

| Variant | Median | Ratio to native | Target (1.5x) | Intr / scalar |
|---|---|---|---|---|
| scalar | 4.554 ms | 5.48 | - | - |
| neon-intrinsics | 0.801 ms | 0.96 | pass | 0.18 |
| native-base | 0.831 ms | 1.00 | - | - |
| corgi | 1.167 ms | 1.40 | - | - |

### argmin

| Variant | Median | Ratio to native | Target (1.5x) | Intr / scalar |
|---|---|---|---|---|
| scalar | 4.358 ms | 3.39 | - | - |
| neon-intrinsics | 1.196 ms | 0.93 | pass | 0.27 |
| native-base | 1.286 ms | 1.00 | - | - |
| corgi | n/a (not in corgi) | - | - | - |

The scalar `argmin` median of 4.358 ms is high against three other measurements of the same variant on this machine: 3.578 ms and 3.616 ms in two isolated reruns of the `tier1/argmin` group (`bench-argmin-rerun.txt`), and 3.608 ms in the first-attempt run.
The scalar ratio and the intrinsics-over-scalar ratio therefore range over 2.77 to 3.39 and 0.27 to 0.33 depending on which run is used.
The intrinsics (1.199 and 1.197 ms in the reruns) and native (1.290 and 1.292 ms) medians were stable across runs.

### filter

| Selectivity | scalar | neon-intrinsics | native-base | corgi | Scalar / native | Intr / native | Target (1.5x) | Intr / scalar | Corgi / intr |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 3.379 ms | 3.051 ms | 2.774 ms | 4.788 ms | 1.22 | 1.10 | pass | 0.90 | 1.57 |
| 50% | 3.881 ms | 3.453 ms | 2.796 ms | 5.085 ms | 1.39 | 1.23 | pass | 0.89 | 1.47 |
| 99% | 4.206 ms | 3.946 ms | 2.839 ms | 5.169 ms | 1.48 | 1.39 | pass | 0.94 | 1.31 |

All three corgi `filter` results equal Verkehrt's rows bitwise, and the bench asserts that before timing.
For `f64` rows the NEON `filter` is 6 to 11% faster than scalar (0.89 to 0.94), and it is 1.10 to 1.39 times `native-base`.
During the optimization, a scratch harness measured the same loops at 0.51 to 0.59 times scalar for `i32` and `f32` rows and at 0.88 to 0.92 for `i64` rows (median of 41 runs, 10M rows).
That harness is not committed, so the committed bench does not reproduce these figures, and it measures `f64` only.

### Summary against the target

| Workload | scalar / native | neon-intrinsics / native | Result |
|---|---|---|---|
| sum | 1.00 | (scalar, no entry) | pass |
| dot | 1.02 | (scalar, no entry) | pass |
| count | 1.00 | (scalar, no entry) | pass |
| min | 5.48 | 0.96 | pass |
| argmin | 3.39 | 0.93 | pass |
| filter 1% | 1.22 | 1.10 | pass |
| filter 50% | 1.39 | 1.23 | pass |
| filter 99% | 1.48 | 1.39 | pass |

Every workload meets the 1.5x target in the final run.
The `filter` ratios are the closest to it, with 99% at 1.39, and the scalar `filter` at 99% (1.48) is itself under the bound by 0.02.
The `filter` intrinsics gain over scalar is small for `f64` rows, as the previous section states.

## First intrinsics attempt and what changed

The first measurement of the NEON entries at `bbef3bb` (selections from `13a5e98`, extremes from `bbef3bb`) failed the target for `filter` and `argmin`.

| Workload | scalar | neon-intrinsics (bbef3bb) | native-base | Intr / native | Intr / scalar | Target (1.5x) |
|---|---|---|---|---|---|---|
| filter 1% | 3.383 ms | 4.876 ms | 2.770 ms | 1.76 | 1.44 | FAIL |
| filter 50% | 3.902 ms | 5.122 ms | 2.789 ms | 1.84 | 1.31 | FAIL |
| filter 99% | 4.211 ms | 5.681 ms | 2.838 ms | 2.00 | 1.35 | FAIL |
| min | 4.584 ms | 0.813 ms | 0.825 ms | 0.99 | 0.18 | pass |
| argmin | 3.608 ms | 2.332 ms | 1.296 ms | 1.80 | 0.65 | FAIL |

The `filter` numbers were reproduced in a second run at 50% (5.120 ms against 5.122 ms).
A profile of the 50% group, taken with samply during the optimization, put 33.1% of its 5363 samples on the transfer after the popcount in the select loop, 32.4% on the vector-to-register transfer in the count loop, and 13.4% on the bound check that waits on the popcount.
An instruction listing showed 19 instructions per 2-row step in the `f64` select loop.
Each 2-row step paid a horizontal mask reduction, a vector-to-register transfer, and a popcount round trip, and the time per step was about 1 ns whatever the lane count.
A scratch harness measured the first attempt at 1.22 to 1.44 times scalar for `f64` and `i64` `filter` and at 0.70 to 0.80 for `f32` and `i32`, so the 4-lane types won and the 2-lane types lost.
The ratios come from the uncommitted scratch harness (median of 41 runs, 10M rows), and the profile comes from the committed `tier1/filter-50pct/neon-intrinsics` group at the first attempt (`2d97663` "Add min, argmin, and an aarch64 crossover to the benches"). Neither was repeated after the fix.

The fix (`1ca388a`) steps the selections four vectors at a time (8 `f64` or 16 `f32` rows per step) with a prefix table instead of a popcount, and the argmin fix (`eb1b552`) uses four (value, index) accumulators.
A run of the committed `tier1` bench after these commits agrees with this document's final run within 2% for `filter` (3.05, 3.46, and 3.95 ms against 3.05, 3.45, and 3.95 ms), and its `argmin` figure of 1.20 ms matches this run's 1.196 ms.

## Fragility of the hot loop

The speed of the four-vector step depends on the compiler inlining everything into the loop.
After the argmin commit (`eb1b552`), the 64-bit `filter` and `compress` entries measured 6.2 ms, 1.5x to 1.8x slower than scalar, and the release binary showed a `bl core::array::map` call inside the select loop with the four vectors spilled to the stack around it.
The call came from `[0,1,2,3].map(..)` in `compress_quad`, and commit `1b46b65` removed the closures and `array::map` from the step in favor of `#[inline(always)]` helpers.
After that commit, a script that lists the call targets of the `aarch64::{compress,extreme}` functions found only calls outside the hot loops, in the example binary and the bench binary.
Nothing in the test suite guards this property, so a later edit can reintroduce a call without a test failing, and listing the call targets of the release binary is the check to run after edits.

## Count pass

The `f64` and `i64` `filter` reads the 80 MB input twice, once in a count pass that sizes the output exactly and once in the select pass, while `native-base` filters in one pass with no count.
An estimate from the measured `sum` and `count` bandwidth puts the count pass at about 0.8 ms, and this document's `filter` gap to native is 0.28 ms at 1% and 1.11 ms at 99%, so the gap is not uniformly of that size.
This bench does not measure the count pass separately, and the estimate is derived from the `sum` and `count` medians, not measured on the count pass itself.
A faster count would need a vector accumulator in the shared `count_rows` that x86 also uses, and the x86 effect cannot be measured on this machine, so the branch leaves it alone.

## Corgi

Corgi is slower than `native-base` on every workload measured here, by 1.60x for `count`, 1.40x for `min`, 6.94x for `sum`, 7.77x for `dot`, and 1.73 to 1.82x for `filter` (4.788, 5.085, and 5.169 ms against 2.774, 2.796, and 2.839 ms).
Against the NEON intrinsics it is 1.57, 1.47, and 1.31 times slower on `filter` at 1%, 50%, and 99%, and 1.46 times slower on `min` (1.167 ms against 0.801 ms).
Corgi's figures match the before table within 3% (sum 5.43 against 5.47 ms, dot 10.98 against 11.1 ms, count 1.31 against 1.33 ms, filter 4.79, 5.08, and 5.17 against 4.93, 5.10, and 5.21 ms).
These numbers do not show why corgi is slower, and its operator passes were not profiled.

## Crossover

The sweep compares the interpreter on `--isa neon --backend intrinsics` with the baseline-target native loop.
`sum` is `sum(x)` and `mul` is `x * y` returning a new array, so its timings include output allocation, and neither has a NEON intrinsics entry, so the interpreter runs the scalar entry for both.
`min` is `reduce(x, min)`, which has a NEON intrinsics entry, and it is swept on aarch64 only.
The crossover length is the smallest `n` from which the interpreter stays within 1.5x of native for all larger `n` in the sweep.
Inputs are prefixes of the same 2^20-element arrays (8 MB each at the largest size), so the small sizes run from cache.

| n | sum interp (ns) | sum native (ns) | sum ratio | mul interp (ns) | mul native (ns) | mul ratio | min interp (ns) | min native (ns) | min ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2^4 | 125 | 1.2 | 102.49 | 159 | 9.6 | 16.59 | 109 | 11 | 9.61 |
| 2^5 | 121 | 1.7 | 72.48 | 162 | 12 | 13.54 | 110 | 11 | 9.93 |
| 2^6 | 126 | 3.1 | 40.13 | 173 | 20 | 8.77 | 113 | 11 | 9.79 |
| 2^7 | 128 | 5.9 | 21.65 | 197 | 31 | 6.26 | 115 | 14 | 7.96 |
| 2^8 | 133 | 11 | 12.32 | 211 | 33 | 6.33 | 122 | 24 | 5.05 |
| 2^9 | 142 | 21 | 6.88 | 248 | 59 | 4.21 | 134 | 40 | 3.31 |
| 2^10 | 161 | 41 | 3.90 | 335 | 113 | 2.98 | 165 | 75 | 2.19 |
| 2^11 | 206 | 82 | 2.52 | 505 | 202 | 2.50 | 234 | 144 | 1.63 |
| 2^12 | 291 | 167 | 1.75 | 835 | 390 | 2.14 | 361 | 285 | 1.27 |
| 2^13 | 460 | 334 | 1.38 | 1708 | 983 | 1.74 | 593 | 575 | 1.03 |
| 2^14 | 884 | 675 | 1.31 | 4577 | 3231 | 1.42 | 1107 | 1130 | 0.98 |
| 2^15 | 2075 | 1944 | 1.07 | 8344 | 6429 | 1.30 | 2088 | 2364 | 0.88 |
| 2^16 | 4046 | 4014 | 1.01 | 16845 | 12212 | 1.38 | 4075 | 4608 | 0.88 |
| 2^17 | 7942 | 8068 | 0.98 | 33823 | 27354 | 1.24 | 8230 | 8992 | 0.92 |
| 2^18 | 17938 | 17913 | 1.00 | 70510 | 52122 | 1.35 | 18271 | 18754 | 0.97 |
| 2^19 | 37592 | 36309 | 1.04 | 141350 | 106290 | 1.33 | 37496 | 39422 | 0.95 |
| 2^20 | 74793 | 74617 | 1.00 | 271430 | 209110 | 1.30 | 75406 | 79849 | 0.94 |

The crossover length is 2^13 = 8192 for `sum` (1.38 there, at most 1.31 from 2^14, and at most 1.07 from 2^15), 2^14 = 16384 for `mul` (1.42, then at most 1.38), and 2^12 = 4096 for `min` (1.27 there, and at most 1.27 from there on).
The `min` interpreter is below native from 2^14 up (0.88 to 0.98), and the `sum` interpreter, which runs the scalar entry, equals native within 4% from 2^16 up.
For `sum` the interpreter time is flat at 121 to 128 ns from 16 to 128 elements, `mul` at 159 to 197 ns, and `min` at 109 to 115 ns, so a fixed per-call cost of roughly 110 to 160 ns dominates there.
At 2^20 elements the `mul` interpreter takes 1.30 times the native time, an allocation-inclusive gap that this sweep does not break down.
Each row is one run of 20 samples, so ratios within a few percent of each other, such as the non-monotonic `mul` ratios at 2^15 to 2^19, are not shown to differ.

## Reading the results

On the M5 the scalar build, which the compiler vectorizes with NEON, is within 2% of `native-base` for `sum`, `dot`, and `count`.
The NEON intrinsics bring `min` from 5.48x to 0.96x of native and `argmin` from 3.39x to 0.93x, and they bring `filter` from 1.22 to 1.48x down to 1.10 to 1.39x.
The `filter` gain for `f64` rows is 6 to 11% over scalar and still leaves a 10 to 39% gap to native.
The first intrinsics attempt lost to scalar on `filter` and failed the target, and the loops that replaced it depend on inlining that no test checks.
SME is present on this chip and out of scope for this branch.
