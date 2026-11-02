# Tiled elementwise regions: design

This layer addresses #16, the cost of filter predicates such as the nine-kernel mask chain of TPC-H q06.
It is roadmap layer 4, stacked on errors as data (`docs/superpowers/specs/2026-11-02-verkehrt-errors-as-data-design.md`).
The executor runs every kernel of a plan over whole arrays, so each comparison and each `and` writes a full intermediate array and the next kernel reads it back from memory.
This design keeps the plans as they are and changes how the executor runs chains of elementwise kernels: in tiles of rows small enough that the intermediates stay in cache.
The sections below give the evidence, the semantics, and the parts left to the builder.

## Evidence

A per-kernel run of `examples/acceptance_stats` on the ClickBench data (x86 host, release build, load 1.8) shows where the time of the mask-chain queries goes.
In q40, `eq_i64`, `le_i64`, and `and_bool` take 1274, 434, and 258 us of 3972 us of kernel time, 50%.
In q41 the same three take 1083, 451, and 320 us of 3442 us, 54%, and in q42 853, 433, and 198 us of 2686 us, 55%.
Each comparison reads a column of 8 bytes per row and writes 1 byte per row, and q40 reads `EventDate` and `TraficSourceID` twice, so the chain moves about twice the bytes of reading each column once.
The layer's measurement task commits these numbers as evidence together with TPC-H q06, which the stats example cannot load today.

Outcome: the measurement record `docs/measurements/2026-11-02-tiled-regions.md` found 4% to 7% on q40, q41 and q42, inside the noise band, and 16% on the best variant of TPC-H q06. The estimate above counts the second reads of `EventDate` and `TraficSourceID` as chain traffic, but tiling does not remove them, and sharing them is #39. The traffic that tiling removes is the 1-byte masks between kernels, which is small next to the 8-byte column reads. The design below stands, and the record gives the numbers.

## Why tiling and not a fused predicate kernel

A fused kernel per predicate shape does not scale, because each query combines a different number of comparisons with `and` and `or`, and the planner matches fused kernels as fixed egglog patterns.
Tiling covers every chain of elementwise kernels, not only predicates, and reuses every existing kernel entry and its intrinsics unchanged.
The kernel layer already has the entry point it needs: `Dispatch::elementwise_tile` runs one elementwise kernel over one tile, and nothing in the executor calls it yet.
Elementwise kernels compute each output row from the same row of their inputs, so a tiled run yields the same bytes as a whole-array run, and bit identity across plans, ISAs, and backends holds without new argument.
Fusing the tiled region into a following selection or reduction, such as `compress` or `sum`, is left to #30.

## Regions

A region is a maximal subtree of a statement's plan made of `PlanNode::Kernel` nodes whose kernel has class `KernelClass::Elementwise` and an array output, with at least two kernels.
Its inputs are the subtrees just below it: leaves, broadcasts, and kernels of other classes, which the executor evaluates whole as it does today.
A `Cond` is never part of a region, so a region never spans a conditional's boundary and lazy branches stay lazy.
A single elementwise kernel is not a region and runs whole, because tiling it saves nothing.
An input is trivial when evaluating it can neither block nor do unbounded work: a leaf, or a broadcast of a leaf.
A region tiles only when every input that is not trivial comes before the region's first kernel in the post-order of the whole evaluation, which visits the subtree left to right.
Otherwise the executor evaluates the region whole, because tiling would evaluate such an input before a kernel whose error the whole evaluation raises first, and the input may block, for example on a load of a FIFO.
The executor finds regions while evaluating, so the planner, the plan, and `--plan` output stay unchanged.

## Semantics

A region's value is the value that evaluating its subtree whole would produce, byte for byte.
Its inputs are evaluated first, left to right in the order the whole evaluation reaches them, and then the region runs tile by tile over rows `[k * T, min((k + 1) * T, n))`, where `T` is the tile row count.
By the rule of the previous section the whole evaluation also reaches every input that is not trivial before any region kernel, so tiling evaluates no such input that the whole evaluation does not.
Trivial inputs after a kernel can at most return a stored error, which the fallback below raises in the order of the whole evaluation.
Before tiling, the executor checks that every array input of the region has the same length, as `Dispatch::call` would for each kernel.
Results must not depend on `T`, and the tests run with small tile row counts to cover many tiles, tile tails, and empty arrays.

Errors must be the same as without tiling, kind, message, and span.
When an input fails, the length check fails, or any tile fails, the executor discards the tiled attempt and evaluates the region whole, which raises the error the whole evaluation raises.
The fallback reuses the inputs the attempt evaluated and evaluates the rest when it reaches them, and kernels are pure, so it yields the same values and the same error.
The cost falls only on failing programs.
The discarded attempt writes no trace lines, because a region's lines print only when its tiled run completes.

## Options and flags

`Options` gains a way to turn tiling off and to set the tile row count, and the CLI gains `--no-tiling`.
The default tile row count is a constant that the measurement task picks from at least three candidates on q40, q41, q42, and TPC-H q06, and the record gives the numbers behind the pick.
`--no-fusion` keeps its meaning, planning without fused patterns, and does not turn tiling off.

## Observation

Statistics count a region's kernels as if they had run whole: one call per kernel per region run, the bytes of the whole arguments and result, and the time summed over the tiles, with the allocation of the region's output and tile buffers added to the root.
Trace lines keep their format, one per kernel per region run, with the time summed over the tiles.
The region's inputs print first, then the region's kernels in post-order, which is the order of the whole evaluation, because every input that can call a kernel comes before the region's first kernel.
A trace line therefore no longer means one entry call, and the `--trace` help text says so.

## Tests

* Unit tests in the executor compare tiled and untiled runs of chains over arrays that span several tiles, with tile tails, empty arrays, broadcast scalars, array views that start at any row, and a failing kernel inside a region.
* A generated-programs oracle runs every case tiled with a small tile row count and untiled, and requires equal printed output, equal exports, and equal errors, kind, message, and span.
* The CI gate counts the regions the generated programs tile and requires a minimum, as the rewrite gate does for the sum rewrites.

## Measurement

The record follows `docs/measurements/2026-10-31-clickbench-now.md`, criterion with the five variants, load recorded, the 10% noise band, and every number from a committed evidence file.
It covers q40, q41, q42, and TPC-H q06 before and after, the tile row count choice, and the full acceptance suites to show no query got slower beyond the noise band.
It compares with DuckDB and corgi from the existing reference figures.

## Non-goals

* Fusing tiles into a selection or a reduction, which is #30.
* Sharing common subexpressions such as a column read twice, which is #39.
* Multithreading.

## Left to the builder

The tile buffers' layout and reuse, whether intermediates allocate per tile or once per region, and how the region walk is structured are the builder's choice, guided by the measurement.
A tiled run must not hold more than a few tiles of intermediates per region node at once.
