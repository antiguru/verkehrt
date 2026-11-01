# Verkehrt: grouping without sorting

This spec adds a primitive that assigns dense group ids to the rows of a key column without sorting, and uses it for the many-group ClickBench queries. It follows plan 5 (`2026-11-01-verkehrt-sort-design.md`) and addresses issue #31.

## Why

After plan 5's radix `argsort`, sorting still dominates the counting queries: `argsort_i64` takes 59% to 89% of the time of q04, q08, q09, q15, and q35 (`docs/measurements/2026-11-01-clickbench-groups.md`). These queries run 1.8× to 4.9× slower than DuckDB on one thread. In q04 the sort alone (22.9 ms) exceeds DuckDB's whole query (6.91 ms). Grouping needs only equality, not order, and a hash table finds equal keys in one pass. Two-column keys pay twice under sorting, once per sort pass and once per payload gather, as q08, q09, and q32 show. A two-column hash key avoids that.

## Goals

* **The `group_id` primitive.** `group_id(k)` returns, for each row of the key column `k`, the dense id of its key: the first distinct key seen gets 0, the next new key gets 1, and so on, as `i64`. `group_id(k1, k2)` does the same for the pair of two equally long columns. Both are defined for `i64`, `u64`, `i32`, and `u8` keys. The two-column form accepts any combination of those types. The number of groups is `reduce(ids, max) + 1` on non-empty input.
* **Identical output by construction.** The ids depend only on the order in which distinct keys first appear, never on the hash function, the table layout, or the ISA. Every variant therefore returns identical output if it is correct, and the existing cross-variant tests check that.
* **Rewritten queries.** q04, q08, q09, q15, q32, and q35 are rewritten with `group_id` wherever that is faster, and they still match DuckDB byte for byte. The output order of every query is unchanged. Groups come out in first-occurrence order, so the queries order groups by key before the count sort, as the SQL tiebreaks require, by sorting the groups (far fewer than the rows) instead of the rows. The same check applies to q30, q31, q40, and q41.
* **Measurement.** `group_id` is benched against the sort-based grouping idiom on the same inputs at 1M and 10M rows, with uniform random keys and with the ClickBench key distributions. The rewritten queries are timed against their sort-based versions, corgi, and DuckDB on one thread. The base spec's catalog gains a row for `group_id`.

## Non-goals

* **No changes to `argsort`, `histogram`, or `reduce_into`.**
* **No fused hash aggregation.** For example, a primitive that groups and sums in one pass. `group_id` followed by `reduce_into` is the composition. A fused kernel is a later step if the measurement shows the second pass matters.
* **No string keys.** Those belong with the string tier (#21), which will reuse this primitive's design.
* **No more than two key columns.** A third column is a later extension if a query needs one.

## Constraints

* Allocation failure surfaces as an error, including the hash table's storage.
* The hash table must handle adversarial key sets without unbounded probing. A deterministic, well-mixed hash (for example a multiply-xorshift of the key) with linear or quadratic probing and a load factor bound is enough, because the output does not depend on the hash. The design must not depend on the absence of collisions.
* `group_id` on an empty column returns an empty `i64` array. The two-column form rejects columns of different lengths with the length mismatch error that other binary kernels use.
* A scalar loop on a supported ISA remains acceptable only as a recorded strain case. If `group_id` has no vector variant, the catalog row says so, with the measurement behind it.
* Correctness in general, not only on this data: ties, empty filters, and both signs of signed keys.

## Left to the builder

* The hash function, the table layout and probing scheme, and the growth policy. Each choice is measured and recorded.
* Whether `u8` keys use a direct 256-entry table instead of hashing.
* How the type checker and the planner register the new op. It follows `histogram` and `argsort` in `src/ops.rs` and `src/check.rs`.
* Which queries switch to `group_id`. A query keeps the sort-based form when that is faster, and the measurement records why.
