# Verkehrt plan 5: sort and high-cardinality grouping

This spec makes `argsort` fast and uses it to add the ClickBench queries that group or count distinct values over many keys. It is plan 5 of the base spec `2026-10-30-verkehrt-design.md` and is tracked by issues #9 and #10. It follows the ClickBench round of #20, whose measurement (`docs/measurements/2026-10-31-clickbench-now.md`) sets its targets.

## Why

Grouping by a key without a small range sorts the rows, and `argsort` is the bottleneck. Plan 4 implemented `argsort` as `sort_unstable_by_key` over row positions, with the position as the tiebreak. It takes about 790 ms per 10M `i64` rows on every ISA, and it is a catalogued strain case. In the #20 measurement, `argsort_i64` takes 44% to 54% of the time of q30, q31, q40, and q41. Those queries are level with DuckDB on one thread or slower, and up to 1.9× slower than corgi. Corgi's notes attribute most of its own time on the many-group ClickBench queries to sorting, and its radix sort on two integer columns beats DuckDB's hash table on q32.

## Goals

* **Fast `argsort`.** `argsort` on `i64`, `u64`, `i32`, and `u8` beats the plan 4 baseline at 10M rows on the x86 benchmark host. Its output stays exactly the same. The implementer chooses the algorithm by measurement, and the choice and the numbers behind it are recorded. Candidates, listed as options only:
  * LSD radix sort on (key, position) pairs, using `histogram`-style counting and a scatter
  * sorting packed (key, position) words
  * a comparison sort over keys gathered into a contiguous buffer, instead of indirect loads
  * SIMD sorting networks for small runs
  The keys come from a bounded domain, so radix is the likely winner, per the user, but the measurement decides.
* **Identical output by construction.** `argsort` is stable, so its output is fully determined by the input: any correct stable sort produces the same permutation. Every variant (scalar, auto, intrinsics, each ISA) therefore returns identical output if it is correct, and the existing cross-variant property tests check that.
* **New acceptance queries.** ClickBench q04, q08, q09, q15, q32, and q35 join the acceptance suite, each matching DuckDB byte for byte. All of them group or count distinct values over `UserID`, `WatchID`, or `ClientIP` with sorting, as q30, q31, and q40 already do.
* **Measurement.** The new queries and the four sort-bound queries from #20 (q30, q31, q40, q41) are timed against corgi and DuckDB on one thread, with the method of the earlier measurement documents. `argsort` is benched at 10M rows for each type against the plan 4 baseline. The base spec's strain-case list and catalog row for `argsort` change only where the measurement supports it.

## Non-goals

* **No hash grouping.** Sorting serves every query in scope, and corgi's results suggest it is competitive. If the measurement shows a query that sorting cannot bring within reach of DuckDB, that query is recorded as a gap for hash grouping.
* **No `find` (a sorted search that returns ranges) and no joins.** The TPC-H join queries (q03, q05, q10) also need string predicates, so they belong with the string tier (#21) and a joins plan.
* **No top-k primitive** (`sort_limit`), unless the measurement shows that `argsort` plus `slice` dominates a query where a partial sort would change the outcome. In that case the gap is recorded, not built.
* **No `f64` argsort.** It remains a surface gap (#11). The #20 round showed that q27's exact order needs none.
* **No changes to `histogram` and `reduce_into`.** Making them single-pass (#10) is independent of sorting, and stays a separate item.
* **No string queries.** ClickBench q24 to q26 sort on strings, which belong with the string tier.

## Constraints

* Stability and output identity are absolute. A faster sort that is not stable, or that differs from the plan 4 output on any input, is wrong. The plan 4 implementation stays available as the reference in tests.
* Allocation failure surfaces as an error, as everywhere else (`Array::try_build` or `try_zeroed`). A radix sort's scratch buffers count.
* `argsort` keeps its signature: one array in, `i64` positions out. Signed keys sort in signed order, which for radix means flipping the sign bit of the key before the passes.
* A scalar loop on a supported ISA remains acceptable only as a recorded strain case. If the chosen algorithm has no vector variant, the catalog row says so, with the measurement behind it.
* Correctness in general, not just on the data. The queries must be right for every input, including empty filters and ties, as in the #20 round.

## Left to the builder

* The algorithm, the radix width, and whether small inputs switch to another method. Each choice comes from a measurement that is recorded.
* Whether vector variants (auto or intrinsics) pay for the counting or scatter passes.
* How each query is written with today's primitives. The grouping idiom of q30, q31, and q40 (stable sorts, then run starts from `prev` and a comparison) is the starting point.
