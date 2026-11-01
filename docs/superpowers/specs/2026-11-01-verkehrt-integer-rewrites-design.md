# Verkehrt: exact integer rewrites for sums

This spec adds planner rewrites that move a broadcast scalar out of an integer sum. It is the first part of issue #30, split out because it needs no new kernels and closes the largest gap to DuckDB in the ClickBench suite. The general fusion of elementwise maps into reductions stays in #30 and follows the term graph (#39).

## Why

ClickBench q29 sums `ResolutionWidth + k` for ninety constants `k`, and Verkehrt runs each as a materialized `add_i64` followed by `reduce_add_i64`. That takes 31.9 ms of kernel time against DuckDB's 7.24 ms (`docs/measurements/2026-10-31-clickbench-now.md`). Integer arithmetic wraps, so it forms a ring modulo 2^w, and `sum(x + k) = sum(x) + length(x) * k` holds bit for bit. The base spec lists integer distributivity as a candidate exact rewrite for this reason. Float arithmetic does not associate, so floats are excluded.

## Goals

* **Rewrites.** For every integer type `T` whose `sum` returns `T` (today `i64` and `u64`) and that has the kernels involved, the planner rewrites:
  * `sum(x + k)` to `sum(x) + length(x) * k`,
  * `sum(x - k)` to `sum(x) - length(x) * k`,
  * `sum(k - x)` to `length(x) * k - sum(x)`,
  * `sum(x * k)` to `sum(x) * k`,

  where `x` is an array and `k` a broadcast scalar. Commutativity, which the planner already has, covers `k + x` and `k * x`. Extraction picks the rewritten form by its lower cost. `i32` and `u8` are excluded because they sum into a wider type after wrapping each element, which no product of the count and `k` reproduces.
* **Visible result.** q29 is re-measured against DuckDB and corgi on the same host and method as `2026-10-31-clickbench-now.md`. The expected outcome is about 8 ms, because ninety `reduce_add_i64` passes remain. Sharing the one `sum(x)` across statements is #39's job.
* **Oracle coverage.** The unfused planner of #35 also drops these rewrites, so oracle 2 compares plans with and without them. The generator produces `sum` over `x ± k`, `k - x`, and `x * k` often enough that every rewrite fires in each CI run, and the test fails if one does not.
* **Planner tests.** Datadriven `plan` cases show each rewrite and the unfused plan beside it, and a float case shows that no rewrite fires.

## Non-goals

* No float rewrites, under any flag.
* No general map-into-reduce fusion and no `rem` primitive. Those stay in #30.
* No sharing across statements.
* No folding of the constant count columns (`add_i64 (mul_i64 gid 0) 1`) of q09, q15, q32, and q35. Constants enter the e-graph as opaque leaves, so a rewrite cannot see a zero. Writing those programs with `fill` or `histogram` is a separate question.

## Constraints

* Wrapping holds for every intermediate: `length(x) * k` wraps exactly as the summed elements would. The planner tests include values that overflow, and the generator draws such values.
* `length` returns `i64` for every element type, so for `u64` the count enters as `cast(length(x))` to `u64`. A cast to an integer type of width `w` gives `n mod 2^w`, and `(n mod 2^w) * k` equals `n * k` modulo `2^w`, so the rewrite stays exact. Unsigned types follow the same ring argument.
* The rewrites create new scalar e-classes. Today `emit` relies on every rewrite adding a single constructor over existing e-classes, so that each e-class has an `Arr` fact if it is an array. The new right-hand sides are scalars, which need no `Arr` fact, and the comment that states the invariant changes to say so.
* A rewrite applies only when every kernel on its right-hand side exists for the type, so no plan can fail extraction.

## Left to the builder

* Whether the rewrites are generated from the registry, like the comparison flips, or listed by type.
* The planner option's name, if the flag that drops fused patterns is renamed to cover these rewrites too.
