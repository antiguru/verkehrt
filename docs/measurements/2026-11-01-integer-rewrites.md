# Integer sum rewrites: q29 measurements

This document re-measures ClickBench q29 after the planner learned to rewrite `sum(x + k)` on `i64` and `u64` as `sum(x) + length(x) * k`.
q29 sums `ResolutionWidth + k` for ninety constants, and the rewrite replaces eighty-nine materialized `add_i64` passes with scalar arithmetic.
The best variant went from 31.38 ms to 8.03 ms, which is 3.9 times faster, and its ratio to DuckDB went from 4.33 to 1.11.
The before numbers, and the corgi and DuckDB figures, are reused from `2026-10-31-clickbench-now.md` and were not re-measured.
What remains is ninety `reduce_add_i64` passes over one column.

## Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395 (Zen 5, 32 logical CPUs), AVX2 and AVX-512 |
| Frequency scaling | boost off, governor `powersave`, 3.0 GHz under load (user-reported, the sandbox cannot read cpufreq) |
| Profile | release (LTO, `codegen-units = 4`), `CARGO_INCREMENTAL=0` |
| Verkehrt build | the branch with the exact integer sum rewrites: broadcast scalars move out of `i64` and `u64` sums, and conditionals are excluded from the rewrite |
| Before | the numbers of `2026-10-31-clickbench-now.md` (q29 row and `stats.txt`), taken on an earlier main. The intervening changes leave q29's kernels unchanged: `reduce_add_i64` is 7988.9 us before and 8184.5 us after. |
| corgi and DuckDB | the figures of `2026-10-31-clickbench-now.md` (corgi 33.14 ms at its pinned revision, DuckDB one-thread 7.24 ms), reused and not re-run |
| Data | `hits_0.parquet`, 1,000,000 rows, the same prepared data directory |

The method is the one in the "How the systems were timed" section of `2026-10-31-clickbench-now.md`.
Verkehrt times are criterion means with `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time, run as `cargo bench --bench acceptance -- 'clickbench/q29/'`.
Per-kernel totals come from `acceptance_stats` (best of five fresh runs).
Differences within about 10% count as noise.

### Evidence files

All files are in `2026-11-01-integer-rewrites/`.

* `bench-a.txt`: criterion log for q29, five variants. Load 2.79 at the start, 2.59 at the end.
* `bench-b.txt`: the same command, second run. Load 2.59 to 2.52.
* `stats.txt`: per-kernel totals of q29. Load 2.52 to 2.40.
* `query-plans.txt`: the `acceptance_plans` output of every acceptance program. The q29 section is the block under `== clickbench/q29`. No load was recorded for this run.
* `correctness.txt`: `cargo test --release --test acceptance clickbench_matches_duckdb`, which passes (1 passed, 0 failed) with the prepared data. No load was recorded for this run.

## Results

Times are criterion means in ms.
The two after columns are the runs of `bench-a.txt` and `bench-b.txt`.
The before column is from `2026-10-31-clickbench-now.md`.

| Variant | before | after, run a | after, run b | before / after (run a) |
|---|---|---|---|---|
| scalar | 42.34 | 15.33 | 15.34 | 2.8 |
| x86-64-v3-auto | 31.38 | 8.26 | 8.18 | 3.8 |
| x86-64-v3-intrinsics | 31.46 | 8.26 | 8.24 | 3.8 |
| x86-64-v4-auto | 31.82 | 8.03 | 8.07 | 4.0 |
| x86-64-v4-intrinsics | 32.07 | 8.03 | 8.04 | 4.0 |

The two runs differ by at most 1% in every variant, well inside the noise band.
The best variant before was `x86-64-v3-auto` (31.38 ms), and after it is `x86-64-v4-intrinsics` (8.03 and 8.04 ms), tied with `x86-64-v4-auto` (8.03 and 8.07 ms).
The four non-scalar variants span 8.03 to 8.26 ms in run a, a 3% spread, so the ISA level and backend change nothing measurable.

| Reference | time (ms) | before: best / reference | after: best / reference |
|---|---|---|---|
| DuckDB 1t | 7.24 | 4.33 | 1.11 |
| corgi | 33.14 | 0.95 | 0.24 |

Ratios below 1.0 mean Verkehrt is faster, and the after ratios use the 8.03 ms of run a.
Against corgi, Verkehrt is now 4.1 times faster, because corgi still runs the ninety passes as written.
Against DuckDB, q29 is 11% slower, just outside the 10% noise band. The DuckDB figure is a best-of time from the earlier round, and criterion means run a few percent above a best-of time, so this document claims neither a gap nor parity.

## The q29 plan

The first sum (`k = 0`) plans as a bare `reduce_add_i64`, and each of the other eighty-nine plans as one reduction plus scalar arithmetic.
The 90 plans in `query-plans.txt` are plan 6 to plan 95, and these are the first, second and last:

```
plan 6: (reduce_add_i64 ResolutionWidth)
plan 7: (add_i64_s (reduce_add_i64 ResolutionWidth) (mul_i64_s (length_i64 ResolutionWidth) 1))
plan 95: (add_i64_s (reduce_add_i64 ResolutionWidth) (mul_i64_s (length_i64 ResolutionWidth) 89))
```

No plan contains a column-length `add_i64` any more.

## Kernel shares

From `stats.txt`, the total kernel time of the least-time run is 8208.5 us.

| Kernel | calls | time (us) | share |
|---|---|---|---|
| `reduce_add_i64` | 90 | 8184.5 | 99.7% |
| `mul_i64_s` | 89 | 10.3 | 0.1% |
| `add_i64_s` | 89 | 8.8 | 0.1% |
| `length_i64` | 89 | 4.9 | 0.1% |

Before, `add_i64` was 75% of 31.9 ms and `reduce_add_i64` 25%.
The `reduce_add_i64` time is now about 8.2 ms for 90 calls, which is 90.9 us per pass, or 88 GB/s over the 720 MB of the 8 MB column read 90 times.
The criterion mean (8.03 ms) is 2% below the kernel total, so the stats timer and criterion agree here.

## What remains

The ninety `reduce_add_i64` calls read the same `ResolutionWidth` column ninety times and return the same value each time, because every rewritten sum contains the same `reduce_add_i64 ResolutionWidth` and differs only in the scalar term.
The planner does not yet recognise the repeated reduction, so the program pays 99.7% of its kernel time on ninety identical passes.
Sharing one reduction across the ninety plans (issue #39) would leave one pass of about 0.09 ms plus the scalar work, an estimate from the per-pass time above that this document did not measure.
That would put q29 below DuckDB's 7.24 ms, but only a measurement after #39 can confirm it.
