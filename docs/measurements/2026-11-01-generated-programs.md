# Generated programs: Findings

This document records what the random program generator of `tests/generated` found, and what it cannot tell.
The generator builds typed programs from the kernel registry and checks each one against six oracles: every ISA variant against scalar, fused against unfused plans, a reference evaluator written from the specs, tiled against untiled runs, shared against unshared runs, and selective against non-selective runs.
The runs found five issues: two pipeline bugs fixed in this layer (hoisted bind spans and the lambda `if`), two lowering and planning costs recorded as #39, and one semantic divergence recorded as #40, which errors as data has since resolved.
No oracle reported a disagreement between kernels, but that result carries no correctness claim, because the coverage limits below are narrow.
This is a findings record, not a benchmark, and the timings only size the CI run.

## Setup

| Item | Value |
|---|---|
| Host | 32 cores, release builds for the searches |
| Workers | independent proptest runners, worker `i` uses seed `SEED + i` |
| Reproduction | `VERKEHRT_GEN_THREADS=1 VERKEHRT_GEN_SEED=<worker seed> VERKEHRT_GEN_CASES=<worker cases>` |
| CI run | fixed seed `0x5eed0001`, 64 cases on 4 workers, 16.1 to 16.5 s in debug with selective evaluation, oracle 6, and the mask chains, 16.3 to 16.4 s with shared evaluation and oracle 5, 16.1 to 16.4 s with the generated deep-sharing chains before them, and 15.2 to 15.6 s before the sharing cap was removed, measured on the 32-core host |
| Variants | scalar, plus x86-64-v3 and v4, each with auto and intrinsics (5 on this host) |

Oracle 1 requires every variant to match scalar exactly, including the error kind, message, and span.
Oracle 2 requires the fused and unfused plans to agree on printed text, export bits, and the error kind.
It compares neither the error line nor the message category, because a statement can read several bindings that hold errors from different statements, and fusion and the comparison flip change which operand a kernel evaluates first.
Oracle 3 checks the line and category of both plans' errors instead.
Oracle 3 compares the scalar outcome against a reference evaluator that calls no kernel, lowering, planner, or executor code.
It shares the front end with the pipeline: the parser and `check::check`, including the lifted typing of lambda bodies through the kernel registry, and it prints values through the library's `Value` display.
Floats compare by bits except that all NaNs are equal, per the base spec.
A value that reaches no print or export is not computed, so errors in it are not raised, which is the ruling the evaluator follows.
Oracle 4 runs each plan on the scalar interpreter with `Tiling::Rows(3)` and with `Tiling::Off`.
It requires equal printed text and export bits, and equal errors including the kind, message, and span, which is stricter than oracle 2 because a tiled run falls back to whole evaluation on any error.
Oracle 1 runs every variant with the same three-row tiles, so the tiled kernels of every ISA are compared too.
The CI run gates on region runs that complete tiled: at least 54 must do so, a third of the 163 the fixed CI seed gave when the gate was set. The seed gave 402 once the generator emitted deep-sharing chains, and gives 129 since shared evaluation, which is the only change between the two counts. A region below a shared node runs once instead of once per use, and a shared node is a region input, never a region kernel, so it shortens or removes the regions around it.
The 5000-case release run at seed 4200000, with oracle 1 tiled and oracle 4 on the unfused plan, had no oracle failure and 13645 region runs that completed tiled.
Oracle 1 runs every variant with three-row tiles, so a region kernel's vector body does not run on generated values there, and `tests/isa_variants.rs` covers the vector loops of the kernels at lengths up to 300.
Oracle 5 runs each plan on the scalar interpreter with three-row tiles, with shared evaluation and with `Options::no_sharing`.
It requires equal printed text and export bits, and equal errors including the kind, message, and span, because reusing a value changes only kernel calls and a failed evaluation is never reused.
Oracles 1 to 4 run with shared evaluation on.
The CI run gates on evaluations of shared nodes that reuse a value in oracle 5: at least 111 must do so, a third of the 334 the fixed CI seed gives. The 3000-case release run at seed 5300000 had no oracle failure, with 14092 evaluations that reused a value and 5344 region runs that completed tiled.
Oracle 6 runs each plan on the scalar interpreter with three-row tiles, with selective evaluation and with `Options::no_selective`.
It requires equal printed text and export bits, and equal errors including the kind, message, and span, because narrowing skips only kernels that cannot fail and a failing tile falls back to whole evaluation.
Oracles 1 to 5 run with selective evaluation on.
The CI run gates on tiles in which a guard narrowed its right operand in oracle 6: at least 32 must do so, a third of the 96 the fixed CI seed gives. Without the generated mask chains the seed gave none. The count varies by seed, with 90 to 1428 over 34 other seeds of 64 cases, so a fresh seed falling under the gate is unlikely. The 5000-case release run at seed 6100000 had no oracle failure, with 42343 narrowed tiles and 9247 region runs that completed tiled.
With three-row tiles the 64-byte block scan of a guard is never reached, and the unit tests in `src/exec/region/tests.rs` cover it.
Oracle 5 is weak on failures: a run that kept failed evaluations, and so reported an earlier statement's span, passed 1000 cases at seed 5400000, because generated programs rarely share a failing subtree between a statement whose error is dropped and a later output. The unit test `statements_sharing_a_failing_subtree_report_their_own_spans` in `src/exec/tests.rs` catches that change.
Whether fusion changed a plan is decided by `Prepared::same_plans`, which numbers the plan graphs by the tree that the plan text prints, in linear time. Rendering the plan text expands shared nodes, so the oracles render it only for failure messages. On 664 programs, the CI seed and 600 cases at seed 5400000, `same_plans` agreed with comparing the rendered text in every case.

## Runs

The first table lists the runs of oracles 1 and 2.
Only the last row used the final generator, and earlier rows used earlier versions of it.
The oracle 2 column names the comparison of errors: "kind only" compared the error kind, and "strict" compared the kind, the statement line, and the category. Oracle 2 has since stopped comparing the line and the category, see the oracle 3 runs below.

| Seed | Cases | Params | Oracle 2 | Result |
|---|---|---|---|---|
| 200000, 300000, 500000 | 6000 each | default | kind only | no failure |
| 400000 | 4000 | depth 5, 9 statements | kind only | no failure |
| 600000 | 6000 | default | strict | no failure |
| 700000 | 3000 | default | strict | no failure |
| 800000 | 6000 | default, final generator | strict | no failure |

About 15,000 further cases with seeds 1, 1000, 20000, and 100000 ran on still earlier generator versions with oracle 2 comparing the error kind only, also without oracle failures.
In the seed 500000 run, 5308 of 6000 programs (88%) ran to completion.
The rest stopped with runtime errors on empty reductions, bucket keys, `gather` indices, `slice` bounds, length mismatches, and `div_const` by zero, which exercises error agreement.
A separate count during the final review fixes, at seed 1200000, found that fusion changed the plan of 2375 of 6000 programs (40%).

The second table lists the runs that include oracle 3, with the oracle 3 version each used.

* **First:** one error union over the whole program, no line rule, and a stopgap that forced every top-level let.
* **Line rule:** the failing statement's line and categories, with a stopgap that forced every top-level let.
* **Narrow stopgap:** the line rule with a stopgap that forces only the let on the pipeline's failing line.
* **Strict:** no stopgap, and the chain rule described under the reference evaluator below, after #40 landed. The generator also emits unread bindings that fail.
* **Strict, both plans:** the strict rule applied to the fused and the unfused plan, with printed output on failure equal to the evaluator's instead of a prefix of it. Oracle 2 no longer compares the error line.

| Seed | Cases | Oracle 3 | Skipped for #40 | Result |
|---|---|---|---|---|
| 900000 | 3000 | first | 6 | no failure |
| 1000000 | 1000 | line rule, with the lazy lambda `if` in the evaluator but before the lowering fix | not recorded | no failure, did not exercise the lambda `if` bug |
| 1100000 | 6000 | line rule | 73 | failed 4 workers on the stopgap |
| 1100000 | 6000 | narrow stopgap | 57 | no failure |
| 1200000 | 6000 | narrow stopgap | 71 | no failure |
| 1300000 | 4000 | narrow stopgap, with calls typed by their body | 44 | no failure |
| 1400000 | 5000 | strict | none, no skip path remains | no failure |
| 1400000 | 5000 | strict, both plans | none | no failure, latest run |

Oracle 3 skips no case. A program the evaluator does not support fails the case, because the generator produces none.

### Feature histogram

Programs containing each feature out of 6000, from seed 800000 with the final generator.
The histogram counts only code that runs, so it skips uncalled function bodies and unread block lets.

```
any/all 700, argmin/argmax 560, argsort 1001, array literal 1723, block 4064,
block let kept unread 1480, compress 1897, count 301, div_const 2524, fill 696,
filter 2144, fn as lambda 178, fn call 3506, fn call in lambda 195, fn called 3555,
fn defined 4539, fn defined but uncalled 2717, fn liftable called 2483,
fn with arrays called 1639, gather 1760, group_id 1419, histogram 119,
if in lambda 842, if on mask 1883, if on scalar 3055, iota 682,
lambda captures a scalar 951, length 2195, let exported at the end 3399,
let used more than once 1237, let used once 1484, map 3223, op * 2362, op + 1283,
op - 1289, op / 428, op abs 897, op and/or/xor/not 2856, op cast 5526,
op comparison 5008, op in lambda 3919, op min/max 2206, op neg 907, op select 2066,
op sqrt 454, prev 1638, rand 170, reduce add 2619, reduce and/or/xor 2355,
reduce min/max 2175, reduce mul 1246, reduce_into 1593, scalar broadcast 4455,
scan 1293, scan_by 1229, shape compress(x, x cmp c) 1594, shape count(x cmp c) 272,
shape sum(select(..)) 1051, shape sum(x * x) 749, shape sum(x * y) 985, slice 827,
statement over the sharing depth cap 101, where 393, zip_map 2189
```

In the same run, 5344 programs ran to completion.
Calls ran in 3555 of the 4539 programs that define functions (78%), and in 2717 programs at least one defined function is never called.
Dead code from constant conditions such as `if true { .. } else { .. }` still counts in the histogram.

## Coverage limits

The runs show that the harness can find bugs, but they bound what it has covered.
Each limit below weakens any statement that the runs found nothing.

* **Thin per-pair coverage.** Per (operation, type) pair coverage was not measured. The histogram bounds it: `histogram`, which has one key type, appears in 119 of 6000 programs (2%), and `op sqrt`, over two float types, in 454 (about 4% per type if split evenly). Operations spread over more types, such as `scan_by` over seven operators and several element types, are estimated at about 1 to 2% per pair. A bug specific to one pair therefore needs thousands of cases to surface.
* **Mutation probes caught 2 of 3 planted kernel bugs.** The probes were reverted after each run.
  * A fused `CountCmp::update` that skipped the last row above 40 rows failed oracle 2 within 640 cases.
  * A `scan_by_add` tail loop on v4 that ignored segment starts for tiles of 8 or more rows failed oracle 1, once unread lets were exported.
  * A second `scan_by_add` mutation (`m |= m << 3`) is wrong only when a segment starts on lane 0 of a later 8-row chunk and the next 7 rows do not start. It was not caught in 3,000 cases, and the kernel property tests in `tests/isa_variants.rs` fit data-pattern bugs of this kind better.
* **Lowering bugs reach only oracle 3.** Swapping the branches of a lambda `if` in lowering passed oracles 1 and 2, because every variant and both fusion settings share the lowering. Only the reference evaluator failed on it, in the 48-case CI run.
* **Deep sharing is rare.** Until lowering and planning shared repeated terms (#39), a statement whose shared bindings nested deeper than 3 became a leaf. That cap hit 49 of 6000 programs at seed 500000 and 101 of 6000 at seed 800000. The cap is gone, and the histogram counts such programs as "sharing depth over 3". Random expressions reach that depth in about 2% of programs (70 of 3000 at seed 5100000), so 6% of statements get a block of 5 or 6 chained lets, each reading the previous one twice through a binary op or three times through a lazy conditional. With it, 571 of 3000 programs at seed 5200000 (19%) have the feature. The fixed CI seed gives 19 of 64, and the CI run requires at least 6, a third of that.
* **Guards over two array comparisons are rare.** Random expressions seldom put `and` or `or` over two comparisons of arrays with a computed scalar in a region that runs tiled, because a scalar computed after the first kernel keeps the region from tiling. So 6% of statements become a chain of `and` and `or`, partly nested, over comparisons of arrays of one length with variables and literals, the first against a literal near an end of the type's range. The chain is demanded as the mask, as `compress` by it, or as the sum of its rows. The histogram counts it as "selective mask chain", 916 of 5000 programs at seed 6100000.
* **The chain rule accepts several lines.** Oracle 3 accepts an error at any statement on its chain, so it cannot tell whether lowering computed a binding in the right statement, only that the reported line is one the span rule allows.
* **Shared code.** All three oracles run the same parser and checker, and the evaluator types lambda bodies through the same registry and prints through the same `Value` display. The generator draws its operations from that registry too. Bugs in parsing, checking, or value formatting are invisible to all three oracles.
* **Untaken lazy branches.** The generator rarely puts a failing value, such as a zero `div_const` divisor, in a lambda branch that is not taken. The 1000-case run of seed 1000000 did not find the lambda `if` bug, and a hand-written probe did.
* **Value-dependent primitives are mostly built valid.** Most `gather`, `slice`, `histogram`, and `reduce_into` operands are generated in range, and only a small share is arbitrary.
* **Types and primitives not generated.** The scalar-only types `i8`, `i16`, `u16`, and `u32` are absent, because lambdas over them cannot lift. `load_*` and strings are absent too.
* **Shrinking is bounded.** Shrinking has a 120 s budget, and minimal cases from the probes still had 4 to 6 statements.

## Bugs found

### Exponential term growth in lowering (#39)

Lowering inlines block lets and function arguments by substituting deep clones.
A binding read k times and nested d levels deep therefore grows the term as k^d before the hoisting pass recovers any sharing.
Top-level lets do not show it, because they stay separate statements.
The function form `fn f(q: i64[]) -> i64[] { let b = filter(q, |p| (p > 3)); ((b + b) + (b + b)) }` called as `f(f(...f(a)))` took 4.84 s and 1.8 GB at depth 7 in the release CLI. The block form, nested `filter` blocks that read each binding four times, took 4.85 s and 1.8 GB, and the same program with top-level lets took 0.59 s and 24 MB.
The table below is the function form.

| Depth (function form) | Time | Peak RSS |
|---|---|---|
| 2 | 0.20 s | 23 MB |
| 3 | 0.29 s | 24 MB |
| 4 | 0.38 s | 24 MB |
| 5 | 0.52 s | 33 MB |
| 6 | 1.06 s | 230 MB |
| 7 | 4.84 s | 1.8 GB |

The generator found this as an out-of-memory failure, when 32 workers exceeded a 32 GB cap.
The fix is a hash-consed term graph in lowering, tracked as #39 and not part of this change.
The generator worked around it with the cap described above until lowering built a term graph and planning built a plan graph, and the cap has since been removed.

### Slow planning inside lazy branches

Hoisting leaves the branches of a lazy (scalar-condition) `if` alone, so each branch is planned on its own and `emit_node` writes repeated nodes again.
One generated statement took 7.5 s to plan on a 4-element input.
A reduced form, `let b7 = if false { a } else { a }; let b11 = compress(b7, (b7 >= reduce(b7, and)));` followed by 1 to 4 nested lazy ifs over `b11`, costs 0.57, 0.85, 1.12, and 1.42 s, against 0.06 s without the scalar-`if` binding.
Under 32 parallel workers the slowest cases reached 12 to 36 s.
This is a separate symptom of the same substitution, and it is recorded on #39.
The generator's cap did not exclude it, because the cost stays in seconds.
The planner now emits each distinct node once and plans each distinct conditional once per program into a plan graph, and `deep_lazy_branch_nesting_plans_in_linear_steps` in `src/plan/tests.rs` plans 16 nested lazy ifs of this form within a step budget linear in the depth.

### Hoisted binds reported the wrong span

A runtime error in a hoisted subterm carried the span of the subterm, so a let inlined into a later export reported the let's line and column.
The stricter line comparison of oracle 3 found it.
Hoisted binds now carry the span of their statement, with the regression case `hoisted_bind_error_carries_statement_line` in `tests/programs/generated.dd`.

### Lambda `if` on a uniform condition was not lazy

Lowering selected between the branches for every lambda `if`, so `map(x, |v| if true { v } else { div_const(v, 0) })` failed on the branch that is not taken.
The base spec makes an `if` on a uniform condition lazy.
A lifted scalar condition now lowers to the lazy conditional, and a uniform scalar branch next to an array branch is filled to the length of the lambda's first parameter, so the branch not taken never runs.
The change alters generated plans, where a `Cond` with a `fill` replaces a `select`, and oracles 1 and 2 passed on 12,000 cases after it.
The regression case is `lambda_if_on_uniform_condition_is_lazy` in `tests/programs/generated.dd`, and the lowering datadriven case was updated.

### Unread top-level lets are computed (#40)

The pipeline evaluates every top-level `let` that lowering does not inline into its single reader, while the ruling says a value that reaches no print or export is not computed.
The two disagree on three shapes, each confirmed with the release CLI, which fails with `reduce_min of an empty array` where the evaluator prints a value:

* a let that nothing reads;
* a let read only through a function parameter that the body ignores;
* a let read twice, both times in an untaken lazy branch.

The generator hit the second shape at CI worker seed 1592590339.
The fix is demand-driven semantics through errors as data plus a checker rule for unread bindings, tracked as #40.
Until it landed, oracle 3 skipped a case when evaluating the pipeline's failing top-level let at its statement reproduced the failure, and a first, simpler stopgap failed 4 workers of a 6000-case run at seed 1100000.
With errors as data, a failing binding holds its error and only an output that reads it raises, so the stopgap and its skip path are gone.

Errors as data moved the line an error reports, and the strict line comparison rejected some generated cases because of it.
A let with several readers is computed as its own statement, so the pipeline reported the let's line, while the evaluator reported the line of the reading output.
Both lines are correct under the span rule of the errors-as-data spec, which gives an error the line of the statement that computed it.
The evaluator therefore records the chain of statements from the demanding output to the failing expression, and oracle 3 accepts any line on it, as described below.
The regression test `oracle_3_accepts_lines_on_the_chain` in `tests/generated/oracles.rs` pins this shape.

The final review found a second shape, which the generator had not produced.
A print that reads two failing lets reports the line of whichever let its kernel evaluates first, and the fused plan `count_gt_f64 x c` evaluates its operands in the opposite order of the unfused `count (lt_f64 c x)`, so the two plans report different lines.
When the two lets fail with different categories, such as an empty `reduce` and an out-of-range `gather`, the plans also report different categories.
Both errors are on the evaluator's chains, so oracle 2 no longer compares lines or categories, and oracle 3 checks both plans instead.
The regression tests `plans_may_report_different_failing_lets` and `plans_may_report_different_categories` pin this shape.
The 5000-case run with both plans under oracle 3 still compared categories in oracle 2.

## Reference evaluator

The evaluator excludes no primitives: every primitive the generator produces is supported, including float reductions in the canonical order of the base spec.
Strings, `load_*`, and arrays longer than 2^22 rows make a case unsupported, which fails it, because the generator produces none of them.
Evaluation is demand-driven: lets, block lets, and arguments are memoized thunks, and an expression statement is not evaluated.
The evaluator collects the error categories of the first failing output and stops there.
Each collected error carries its chain: the line of the output and of each top-level let or export whose thunk failed while the output forced it.
Oracle 3 accepts a runtime error whose category was collected, whose line is on that category's chain, and whose printed output equals the evaluator's.
It checks the scalar outcomes of the fused and the unfused plan.
The evaluator stays lazy and does not mirror which bindings lowering inlines, so the chain covers every statement that lowering may compute the failing binding in.

## Failing unread bindings

The generator emits `_`-prefixed bindings that nothing reads, some of which fail whenever they are computed.
Before about 12% of top-level statements it adds a failing let, outside lambdas about 12% of blocks get a failing block let, and a quarter of the top-level lets nothing reads keep the `_` prefix instead of being exported.
A failing binding takes the `min` of an empty slice, gathers an index one past the end, or compares an array with one longer by one element, mostly on an array in scope.
The CI run gates on these bindings: at least 8 of its 64 programs must contain one that fails when the evaluator forces it alone, and the fixed CI seed gave 23 when the gate was set and gives 16 since the generator emits deep-sharing chains.
In the 5000-case run at seed 1400000, 1974 programs (39%) had such a binding, 1755 had a failing unread top-level let, and 445 a failing unread block let, while 4492 programs ran to completion.

