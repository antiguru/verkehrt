# Generated programs: Findings

This document records what the random program generator of `tests/generated` found, and what it cannot tell.
The generator builds typed programs from the kernel registry and checks each one against three oracles: every ISA variant against scalar, fused against unfused plans, and a reference evaluator written from the specs.
The runs found five issues: two pipeline bugs fixed in this layer (hoisted bind spans and the lambda `if`), two lowering and planning costs recorded as #39, and one semantic divergence recorded as #40.
No oracle reported a disagreement between kernels, but that result carries no correctness claim, because the coverage limits below are narrow.
This is a findings record, not a benchmark, and the timings only size the CI run.

## Setup

| Item | Value |
|---|---|
| Host | 32 cores, release builds for the searches |
| Workers | independent proptest runners, worker `i` uses seed `SEED + i` |
| Reproduction | `VERKEHRT_GEN_THREADS=1 VERKEHRT_GEN_SEED=<worker seed> VERKEHRT_GEN_CASES=<worker cases>` |
| CI run | fixed seed `0x5eed0001`, 48 cases on 4 workers, 7.5 s to 8.1 s in release and 10.1 s in debug (latest measurement) |
| Variants | scalar, plus x86-64-v3 and v4, each with auto and intrinsics (5 on this host) |

Oracle 1 requires every variant to match scalar exactly, including the error kind, message, and span.
Oracle 2 requires the fused and unfused plans to agree on printed text, export bits, and the error kind, statement line, and message category.
Oracle 3 compares the scalar outcome against a reference evaluator that imports only the typed program from `check`, the parser, and the value types, and calls no kernel, lowering, planner, or executor code.
Floats compare by bits except that all NaNs are equal, per the base spec.
A value that reaches no print or export is not computed, so errors in it are not raised, which is the ruling the evaluator follows.

## Runs

The first table lists the runs of oracles 1 and 2.
Only the last row used the final generator, and earlier rows used earlier versions of it.

| Seed | Cases | Params | Result |
|---|---|---|---|
| 200000, 300000, 500000 | 6000 each | default | no failure |
| 400000 | 4000 | depth 5, 9 statements | no failure |
| 600000 | 6000 | default, strict oracle 2 | no failure |
| 700000 | 3000 | default | no failure |
| 800000 | 6000 | default, final generator | no failure |

About 15,000 further cases with seeds 1, 1000, 20000, and 100000 ran on still earlier generator versions, also without oracle failures.
In the seed 500000 run, fusion changed the plan in about 39% of programs, and about 88% ran to completion.
The rest stopped with runtime errors on empty reductions, bucket keys, `gather` indices, `slice` bounds, length mismatches, and `div_const` by zero, which exercises error agreement.

The second table lists the runs that include oracle 3.

| Seed | Cases | Skipped for #40 | Result |
|---|---|---|---|
| 900000 | 3000 | 6 | no failure |
| 1100000 | 6000 | 73, then 57 on rerun | the first run failed 4 workers on the first stopgap, and the rerun passed after the fix |
| 1200000 | 6000 | 71 | no failure |
| 1300000 | 4000 | 44 | no failure, latest run |

Seed 1000000 ran 1000 cases before the lambda `if` fix and passed, but it did not exercise that bug.
Oracle 3 skips cases only for the #40 stopgap, because the evaluator supports every generated primitive.

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

* **Thin per-pair coverage.** Each (operation, type) pair appears in only about 1 to 2% of programs, so a bug specific to one pair needs thousands of cases to surface.
* **Mutation probes caught 2 of 3 planted kernel bugs.** The probes were reverted after each run.
  * A fused `CountCmp::update` that skipped the last row above 40 rows failed oracle 2 within 640 cases.
  * A `scan_by_add` tail loop on v4 that ignored segment starts for tiles of 8 or more rows failed oracle 1, once unread lets were exported.
  * A second `scan_by_add` mutation (`m |= m << 3`) is wrong only when a segment starts on lane 0 of a later 8-row chunk and the next 7 rows do not start. It was not caught in 3,000 cases, and the kernel property tests in `tests/isa_variants.rs` fit data-pattern bugs of this kind better.
* **Lowering bugs reach only oracle 3.** Swapping the branches of a lambda `if` in lowering passed oracles 1 and 2, because every variant and both fusion settings share the lowering. Only the reference evaluator failed on it, in the 48-case CI run.
* **The #39 cap replaces statements.** A statement whose shared bindings nest deeper than `MAX_SHARE_DEPTH = 3` becomes a leaf. The cap hit 49 of 6000 programs at seed 500000 and 101 of 6000 at seed 800000, the latest count, after the generator started to call functions more often. The generator under-covers deep sharing until #39 lands.
* **The #40 stopgap skips cases.** Oracle 3 skips about 1.2% of cases (44 of 4000 to 73 of 6000 in the runs above), which hides its checks on those programs until #40 lands.
* **Untaken lazy branches.** The generator rarely puts a failing value, such as a zero `div_const` divisor, in a lambda branch that is not taken. The 1000-case run of seed 1000000 did not find the lambda `if` bug, and a hand-written probe did.
* **Value-dependent primitives are mostly built valid.** Most `gather`, `slice`, `histogram`, and `reduce_into` operands are generated in range, and only a small share is arbitrary.
* **Types and primitives not generated.** The scalar-only types `i8`, `i16`, `u16`, and `u32` are absent, because lambdas over them cannot lift. `load_*` and strings are absent too.
* **Shrinking is bounded.** Shrinking has a 120 s budget, and minimal cases from the probes still had 4 to 6 statements.

## Bugs found

### Exponential term growth in lowering (#39)

Lowering inlines block lets and function arguments by substituting deep clones.
A binding read k times and nested d levels deep therefore grows the term as k^d before the hoisting pass recovers any sharing.
Top-level lets do not show it, because they stay separate statements.
A depth 7 program of nested `filter` blocks that read each binding four times took 4.84 s and 1.8 GB in the release CLI, where the same program with top-level lets took 0.59 s and 24 MB.

| Depth | Time | Peak RSS |
|---|---|---|
| 2 | 0.20 s | 23 MB |
| 3 | 0.29 s | 24 MB |
| 4 | 0.38 s | 24 MB |
| 5 | 0.52 s | 33 MB |
| 6 | 1.06 s | 230 MB |
| 7 | 4.84 s | 1.8 GB |

The generator found this as an out-of-memory failure, when 32 workers exceeded a 32 GB cap.
The fix is a hash-consed term graph in lowering, tracked as #39 and not part of this change.
The generator works around it with the cap described above, with a comment that points to #39.

### Slow planning inside lazy branches

Hoisting leaves the branches of a lazy (scalar-condition) `if` alone, so each branch is planned on its own and `emit_node` writes repeated nodes again.
One generated statement took 7.5 s to plan on a 4-element input.
A reduced form, `let b7 = if false { a } else { a }; let b11 = compress(b7, (b7 >= reduce(b7, and)));` followed by 1 to 4 nested lazy ifs over `b11`, costs 0.57, 0.85, 1.12, and 1.42 s, against 0.06 s without the scalar-`if` binding.
Under 32 parallel workers the slowest cases reached 12 to 36 s.
This is a separate symptom of the same substitution, and it is recorded on #39.
The generator's cap does not exclude it, because the cost stays in seconds.

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
The design direction is demand-driven semantics through errors as data plus a checker rule for unread bindings, filed as #40 above this layer.
Until then, oracle 3 skips a case only when evaluating the pipeline's failing top-level let at its statement reproduces the failure.
A first, simpler stopgap was too narrow and too wide at once, and a 6000-case run at seed 1100000 failed 4 workers on it, so the stopgap now requires all of:

* the pipeline fails at line L;
* the lazy evaluation succeeds or fails only after L;
* evaluating only the top-level let on line L gives an error at L with the pipeline's category and the pipeline's output.

## Reference evaluator

The evaluator excludes no primitives: every primitive the generator produces is supported, including float reductions in the canonical order of the base spec.
Strings, `load_*`, and arrays longer than 2^22 rows make a case unsupported and skipped, and the generator produces none of them.
No case was skipped for those reasons.
Evaluation is demand-driven: lets, block lets, and arguments are memoized thunks, and an expression statement is not evaluated.
The evaluator collects the error categories of the first failing statement and stops output there, and oracle 3 accepts a runtime error at that line whose category was collected and whose output is a prefix of the evaluator's.
