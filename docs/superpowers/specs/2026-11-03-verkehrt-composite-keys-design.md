# Composite-key argsort and top-k: design

This layer addresses #31 (composite-key grouping and top-k after the radix argsort) and what remains of #9 (make argsort fast), as roadmap layer 8.
Since #31 was filed, `group_id` (#34) took over the grouping of q04, q08, q09, q15 and q35, so most of the issue's targets changed shape, and the designer re-measured before choosing.
The measurement shows one large gain left, a single radix sort over two key columns for q32, and one smaller gain, a top-k over the sorted group order for q15 and q35, which needs the two-key sort to express its order.
This design adds a two-column form of `argsort` with an off switch that expands it into today's two-sort form, and fused top-k kernels for `slice` over `argsort`.
The sections below give the evidence, the scope and the rejected options, the semantics, the kernels, the off switch, the tests, and the measurement, with each open decision marked as the designer's call.

## Evidence

The designer measured on this tree before writing the spec: per-kernel statistics of the target queries from `examples/acceptance_stats` (x86-64-v4 intrinsics, best of five, two runs), and a standalone probe crate that copies `radix_into` for `i64` keys and reads the ClickBench columns.
The probe output and its notes are in the designer's scratch directory and no record holds them yet, so the measurement task repeats the claims below as committed evidence.
Load was 1.9 to 2.9 for the first statistics run and the first two probe runs, 4.3 to 6.2 for the later probe runs, and 9.6 to 11.2 for the second statistics run, whose totals still agree with the first within 4%.
All times are summed kernel times or probe medians in ms, and the 10% noise band of earlier records applies.

| Query | Total | `argsort` | `gather` | reductions | `group_id` | other |
|---|---|---|---|---|---|---|
| q04 | 5.98 | 0.00 (0.0%) | 0.00 (0.0%) | 0.12 (2.0%) | 5.86 (97.9%) | 0.00 (0.1%) |
| q08 | 15.06 | 0.04 (0.3%) | 0.01 (0.1%) | 4.83 (32.1%) | 10.17 (67.5%) | 0.01 (0.1%) |
| q09 | 19.96 | 0.05 (0.2%) | 0.02 (0.1%) | 9.75 (48.9%) | 10.12 (50.7%) | 0.02 (0.1%) |
| q15 | 11.10 | 1.42 (12.8%) | 0.06 (0.5%) | 3.70 (33.3%) | 5.90 (53.1%) | 0.03 (0.3%) |
| q32 | 67.33 | 36.16 (53.7%) | 21.07 (31.3%) | 4.31 (6.4%) | 0.00 (0.0%) | 5.79 (8.6%) |
| q35 | 10.79 | 1.15 (10.6%) | 0.08 (0.7%) | 3.74 (34.6%) | 5.80 (53.8%) | 0.03 (0.3%) |
| q31 | 5.26 | 1.89 (36.0%) | 0.46 (8.8%) | 0.20 (3.8%) | 0.00 (0.0%) | 2.71 (51.5%) |

Reductions are the `reduce_*`, `reduce_into_*`, `histogram` and `count` kernels.
q08 and q09 now sort only their about 400 groups, so sorting and gathering take 0.3% to 0.4% of their kernel time and the composite-key item of #31 has nothing left to remove there.
q32 still groups 1M rows by sorting on (WatchID, ClientIP), with 85% of its kernel time in `argsort_i64` and `gather_i64`, and q31 and q30 use the same idiom on about 69,000 filtered rows.
q15 and q35 sort their 80,000 and 68,000 groups twice to take ten, which costs 12.8% and 10.6%.

The probe timed q32's grouping stage, which produces the sorted `k`, `ip`, `r` and `w` columns, in four runs of 15 interleaved rounds.
Today's form (two sorts and eight 1M-row gathers) took 48.0 to 53.0 ms, and a single LSD radix sort over 16-byte records of the major key image, the minor key image minus its minimum, and a `u32` position, followed by four gathers, took 34.7 to 37.1 ms, 13.1 to 17.4 ms less in every run.
The sort alone took 27.4 ms against 35.3 ms for the two sorts, so the saving comes from both the sort and the four gathers it removes.
Against q32's summed kernel time of 64.8 to 67.3 ms the estimate is 19% to 27%, a range because the probe's scalar gather (2.0 ms per call) is faster than the kernel's (about 2.6 ms per call in the statistics).
The same records widened to 24 bytes (two `u64` images and a position), needed when the keys do not fit 96 bits, took 47.2 to 49.1 ms, level with today's form.

The other candidates for q32 fell short.
Composing the two permutations, so that each payload is gathered once, saved 3.0 to 4.6 ms (5% to 7% of q32).
A composite sort on (key, position) pairs that re-keys after the minor passes by loading the major key through the position took 41.9 to 44.5 ms.
Writing the sorted keys from the sort's last pass saved another 0.8 to 2.5 ms over the 16-byte form, but needs a kernel with several outputs, which the kernel model lacks.

For q15 and q35, the probe ran on the real group columns (first-occurrence keys and counts) in two runs of 60 rounds.
Today's two sorts and gathers took 1.46 to 1.50 ms (q15) and 1.23 to 1.31 ms (q35), a single 24-byte composite sort of (minus count, key) took 1.53 to 1.62 and 1.30 to 1.43 ms, and a scan that keeps the best ten took 0.09 to 0.11 and 0.07 to 0.10 ms, with the same ten positions.
The kernels a top-k replaces take 1.50 ms of 11.10 in q15 and 1.25 ms of 10.79 in q35, so a top-k saves about 12.6% and 10.7%, at the edge of the noise band.
A composite sort without top-k is level with today on these two queries, so the gain there comes from top-k alone.
q32's count sort runs over a constant key (comment on #31, 2.5 to 2.8 ms), and a 1M-row selection scan costs about 1 ms by scaling the 80,000-row probe, so top-k saves under 3% of q32 (inference).

**The in-situ gap of #9.** `docs/measurements/2026-11-01-clickbench-groups.md` found q04's `argsort_i64` on the real UserID column about 40% slower than the kernel on uniform keys, with the cause not measured.
The probe reproduces it: UserID sorts in 21.7 to 23.4 ms and uniform keys in 12.8 to 15.1 ms, both with 8 passes, and a shuffled UserID still takes 20.2 to 21.5 ms.
Per pass, uniform keys take 1.1 ms and UserID 2.0 to 2.3 ms, and the bucket-offset increments alone take 0.44 against 1.1 to 1.3 ms, while a shuffled UserID is fast until pass 2, when equal keys have become adjacent.
The designer infers that runs of equal digits serialize the offset increment through one counter, and a two-way interleaved scatter with separate offsets did not help (2.3 to 2.4 ms per pass).
After #34 no target query sorts a long column with such runs except q32's ClientIP passes (about 3 ms over uniform passes, about 5% of q32), so the gap is below the 10% gate here and goes to its own issue.

## Scope and order

Designer's call: this layer builds the two-column `argsort`, then fused top-k kernels on top of it, and rewrites q32, q31, q30, q15 and q35 where the measurement keeps the new form.
The two-column sort is the only option with a gain well beyond the noise band (q32), and top-k reaches the band on q15 and q35 only once the composite sort lets those queries state their order in one call.
Top-k comes second because it depends on the composite sort for its main cases, and because its gain is small enough that the measurement task may drop the q15 and q35 rewrites.
The rest of this section names the options left out and why.

* **Composed permutations.** 5% to 7% of q32, below the gate, and the composite sort removes the same gathers and more.
* **A sort with several outputs.** 0.8 to 2.5 ms over the composite sort, and kernels return one value.
* **Hash grouping (#31's third bullet).** Done by #34: q04 runs at 0.83 of DuckDB, q35 at 0.66 (`docs/measurements/2026-11-01-group-id-queries.md`). The remaining costs there, `group_id` itself and key recovery by `reduce_into(.., min)`, are not sorting.
* **The in-situ gap.** Cause located (see "Evidence"), below the gate on every target query. A follow-up issue records it.
* **`find` for TPC-H joins (#9's comment).** No acceptance program uses a join (`tests/acceptance/tpch/` holds q01 and q06), so nothing measures it.
* **Three or more key columns, float and bool keys, descending order as a flag.** No query needs them. Descending order is `0 - x` on integer keys, as today.

## Semantics

`argsort(a, b)` returns the positions of the rows of the equally long columns `a` and `b` in ascending order of `a`, ties broken by ascending `b`, then by position, as `i64`.
It equals `gather(o1, argsort(gather(a, o1)))` with `o1 = argsort(b)` on every input of equal lengths, because both forms order rows by (`a`, `b`, position) and positions are distinct.
Each column is `i64`, `u64`, `i32` or `u8`, and any combination is allowed, as in `group_id(k1, k2)`.
Columns of different lengths fail with `length mismatch: {len(a)} vs {len(b)}`, the message of `group_id(k1, k2)`, and an empty pair gives an empty array.
The one-argument `argsort(x)` is unchanged.

`slice(argsort(...), s, e)` keeps its meaning, and the fused top-k kernels compute it without sorting every row.
They return the same positions, and fail with the same error as the unfused plan, which checks the argsort's arguments first and then the bounds with `slice {s}..{e} out of range for length {n}`.

## The composite sort kernel

`argsort2_<a>_<b>` is a whole kernel with op `ArgSort` and two array inputs, registered for the 16 type pairs next to `argsort_<t>`, with auto entries through `KernelImpl::whole`.
Its body maps each key to its unsigned image with the existing `RadixKey::image`, subtracts each column's minimum image, and packs the two reduced images into one composite image of `bits(a) + bits(b)` bits, minor bits low, where `bits(x)` is the bit length of the column's maximum reduced image.
Subtracting the minimum preserves order and is what lets ClientIP's full `i32` range fit 32 bits, which the sign-flipped `i64` image does not.
The sort is LSD with byte digits over the composite image, one counting pass for all digits, and the existing skip of digits equal in every row, so the code follows `radix_into` and its scatter.

Designer's call: two record layouts, chosen per call.
When the composite image fits 96 bits and every position fits `u32`, records are 16 bytes (`u64` high part, `u32` low part, `u32` position), the C16 form of the probe and the same size as today's 64-bit pairs.
Otherwise records are 24 bytes (the two reduced images as `u64` each and the position), which the probe measured level with today's two-sort form on q32's columns.
Inputs of at most 1024 rows use a comparison sort by (`a`, `b`, position), the 64-bit threshold of `argsort`, which is not measured for pairs, and the builder may lower it with evidence.

Peak scratch is two record buffers, 32 bytes per row in the 16-byte layout, the same as `argsort` on 64-bit keys, and 48 bytes per row in the 24-byte layout.
The min and max pass reads both columns once more before the counting pass.
Scratch is reserved with `try_reserve_exact`, and a failed reservation is an error of the call, as in `argsort`.
The kernel has no intrinsics, and its auto entries compile the same body, as for `argsort` (`docs/measurements/2026-11-01-argsort.md` found no gain from target features there).

## Bit identity

The output is a permutation fixed by the total order (`a`, `b`, position) on integer keys, so it does not depend on the layout, the threshold, the ISA, or the backend.
The radix path is stable because each scatter visits records in order, and the comparison path is stable because the position is part of its key.
`tests/isa_variants.rs` covers the new kernels through its `argsort_` prefix and must gain `argsort2_` and `topk` prefixes if its matching needs them.
The property tests below compare every kernel with a reference that sorts by (`a`, `b`, position) with the standard library, so tie order is checked, not assumed.

## Off switch

`Options` gains `no_composite_sort`, the CLI `--no-composite-sort`, and the acceptance bench and the stats example read `VERKEHRT_COMPOSITE_SORT=off`, as they read `VERKEHRT_SELECTIVE`.
Designer's call: with `no_composite_sort` set, lowering expands every two-argument `argsort(a, b)` into `gather(o1, argsort(gather(same_len(a, length(b)), o1)))` with `o1 = argsort(b)`, which is today's two-sort idiom plus the length check.
The `same_len` operand raises the kernel's message `length mismatch: {len(a)} vs {len(b)}`, and the term graph shares `o1`, so the expansion runs each single-key sort once.
Because the expansion is a different plan that evaluates the operands in a different order, it matches the composite kernel's output exactly but, when both operands fail, may surface the other operand's error, as fusion does under oracle 2.

The off switch is the oracle's reference and the measurement's control, so it must not disappear when the kernel exists.
`--no-fusion` does not expand `argsort(a, b)`, because the composite kernel is a single-op kernel and not a fused pattern.
Top-k kernels are fused patterns, so `--no-fusion` and the existing oracle 2 already plan without them, and they need no switch of their own.
With `--no-composite-sort`, a two-key top-k cannot match, because the expansion leaves no `argsort2` under the `slice`.

## Top-k

`topk_<t>` and `topk2_<a>_<b>` are fused whole kernels with no op and the patterns `(slice_i64 (argsort_<t> ?x) ?s ?e)` and `(slice_i64 (argsort2_<a>_<b> ?x ?y) ?s ?e)`, with inputs (`x`, [`y`,] `s`, `e`), `s` and `e` scalar `i64`.
Their cost must be below the cost of the `argsort` they replace, so that extraction prefers them, since `slice` costs 0.
The body checks the argsort's arguments (length mismatch), then the bounds as `run_slice` does, and then, when `e` is at most `TOPK_MAX`, keeps the best `e` rows by (images, position) in a bounded heap or sorted buffer and returns rows `s..e` of that order.
When `e` exceeds `TOPK_MAX`, it runs the full sort and copies `s..e`, so a large slice is never slower than today beyond the copy.

Designer's call: `TOPK_MAX` is 64.
The probe measured only `e` = 10, and the scan's worst case is an input in descending order, where every row enters the buffer.
The top-k task benches that worst case at `e` = `TOPK_MAX` against the full sort at 80,000 and 1M rows, and lowers `TOPK_MAX` if the scan loses.
The fused patterns are the first over whole kernels with scalar operands, so the top-k task first confirms that the planner matches `slice` of `argsort` in a plan, and stops with BLOCKED if the prelude rejects the pattern or extraction never chooses it.

## Tests

* **Kernel property tests.** `argsort2_<a>_<b>` against the reference sort by (`a`, `b`, position) for all 16 type pairs, with arrays drawn from boundary pools (type extremes, both signs, few distinct values so ties are common), lengths around 1024, and composite widths at 0, 64, 96 and 97 bits so both layouts and the all-constant path run. Fixed tests for empty input, one row, (1, 2) against (2, 1), a length mismatch message, constant columns, and a scratch allocation failure, following the existing `argsort` tests.
* **Top-k tests.** Each `topk` kernel against `slice` of the full argsort for `e` in 0, 1, 10, `TOPK_MAX`, `TOPK_MAX + 1` and `n`, `s` in 0 and `e`, descending, ascending and all-equal inputs, and each bounds error and the length mismatch by message. `tests/fused_oracle.rs` gains a top-k family, with a pool that draws in-range bounds most of the time.
* **Lowering tests.** With `no_composite_sort`, `--plan` shows the expansion and no `argsort2` kernel, `o1` is one shared node, and values and the length mismatch error equal the composite kernel's on fixed inputs. A CLI test checks that `--no-composite-sort` prints the same output.
* **Oracle 8, composite against expanded.** On the scalar interpreter with three-row tiles, each plan run with the composite sort gives the output and exports of the run with `no_composite_sort`, and fails at the same output with the same error kind, compared as oracle 2 compares them (see "Off switch"). Programs whose plans are equal skip the second run, as oracle 2 does with `Prepared::same_plans`.
* **Generator and gates.** The generator emits `argsort(a, b)` wherever it emits `argsort` today, through the registry signatures and as a sort permutation in `indices`, and emits `slice(argsort(...), s, e)` with in-range bounds as a `slice` shape. The reference evaluator gains the two-argument `argsort`. The CI run requires a minimum of programs in which oracle 8 compared different plans and a minimum of plans that contain a `topk` kernel, each a third of what the fixed CI seed gives. The existing minimums must still hold after the generator change, and the generator task adjusts the new weights, not the old minimums, to keep them.

## Measurement

The record `docs/measurements/2026-11-03-composite-keys.md` follows `docs/measurements/2026-11-01-group-id-queries.md`: criterion with the five variants, `sample_size(10)`, a 0.5 s warm-up and a 3 s measurement time, the 1-minute load recorded at the start and end of every evidence file, the 10% noise band, and every number from a committed evidence file.
The before programs are today's files, read through `VERKEHRT_PROGRAMS` from a copy in the evidence directory, so before and after run in one build and session.
q32, q15 and q35 run in three interleaved pairs with 30 samples and 10 s, because q15 and q35 sit at the band's edge, and q31 and q30 in one pair, repeated if they move beyond the band.
A rewritten query is kept only if it is faster beyond the run-to-run spread on most variants in every pass, the rule of the group-id record, and otherwise its file returns to today's version.

The record also gives the per-kernel statistics of each query before and after, the composite kernel against the off switch (`VERKEHRT_COMPOSITE_SORT=off`) on the kept q32, the top-k worst-case bench, the `acceptance_plans` diff (only rewritten queries change), and the DuckDB and corgi ratios from the existing reference figures.
The probe crate is not in the repository, so the record re-measures the spec's estimates with the kernels and states the in-situ gap as a follow-up, quoting the spec's probe numbers as the designer's and labeling them so.
The base spec's catalog gains rows for the two-column `argsort` and for top-k.
`clickbench_matches_duckdb` must pass on the kept programs.

## Non-goals

* The in-situ gap fix, a kernel with several outputs, and `find`.
* Rewriting q08 and q09, whose sorts cover about 400 groups.
* Planner or cost-model changes beyond the top-k patterns, and DAG-aware extraction (#24), which decides whether a shared `argsort` under a `slice` is sorted once.

## Left to the builder

* The exact packing code, whether the 16-byte layout covers the 64-bit case or a third 16-byte (`u64`, `u32`) layout serves it, and the generic structure over the 16 type pairs.
* Whether top-k keeps a heap or a sorted buffer, as long as the worst-case bench holds.
* Where in lowering the expansion lives, as long as the term graph shares `o1`.
