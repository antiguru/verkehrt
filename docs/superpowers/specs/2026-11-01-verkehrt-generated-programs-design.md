# Verkehrt: generated programs with independent oracles

This spec adds a generator of random well-typed programs and three oracles that check each program from different sides. It is tracked as issue #35 and comes before #30, so the fusion work lands with a fused-against-unfused oracle in place.

## Why

Most bugs so far sat above the kernels, in lowering, lambda lifting, uniformity analysis, hoisting, and planner rewrites. Examples are `div_const` inside lambdas, the `i8` multiply disagreement, and q27's sort key, which was right on the data but wrong at scale. Hand-written cases rarely combine those features, and random programs combine them constantly. The existing cross-variant tests compare kernels on generated inputs, but they run one kernel at a time, and every variant of a program shares one plan, so a wrong plan or a wrong lowering passes them.

## Goals

* **Generator.** A proptest strategy that builds programs bottom-up from typed expressions, so every program type-checks. It covers literals, input arrays of several scalar types, lets (single-use and multi-use, so inlining and hoisting both fire), elementwise operators with scalar broadcasting, `map`, `filter`, and `zip_map` with lambdas that capture outer scalars, user functions called inside and outside lambdas, scalar conditionals and mask `if`, the reductions, and the tier 2 primitives. Arrays are small, so a case runs in milliseconds. Nesting depth and program length are parameters, so a long run can generate larger programs. Shrinking reduces a failing program to a minimal one.
* **Oracle 1, cross-variant.** The program runs on every instruction set and backend the CPU supports and must give identical output, using the harness of #19's `run` directive.
* **Oracle 2, fused against unfused.** The same program planned without fused patterns must give identical output. This needs a planner option that drops the fused-pattern rewrites, exposed in `Options` for tests.
* **Oracle 3, reference evaluator.** A naive tree-walking evaluator over the typed program (`check::TProgram`) must agree with the full pipeline. It does no lowering, lifting, hoisting, or planning: lambdas are applied per element, user functions are called directly, and blocks are evaluated in order. It encodes the documented semantics: wrapping integers, `div_const`, the canonical float summation order of the base spec, and the tier 2 rulings. It is the only oracle that catches lowering and lifting bugs.
* **Errors.** An error that the semantics require, such as a length mismatch or `reduce` of an empty array, must be the same error kind in every oracle. Divergence in whether an error occurs is a failure. The language computes only values that reach a `print` or an export, so an error inside a value nothing reads is not raised, and the reference evaluator follows the same rule.
* **CI and local runs.** The test runs with a fixed seed and a small case count in `cargo test`, well under 30 s in debug. A local run with an environment variable for the case count and a random seed searches longer. Each failure found is fixed and added as a datadriven `run` case (#19) in a `generated.dd` file, so the regression corpus grows from real bugs.

## Non-goals

* No generation of programs that fail type checking. The checker has its own datadriven tests.
* No string programs, since strings are barely in the surface yet.
* No performance testing.
* The reference evaluator does not need to be fast or complete for every primitive on day one. A primitive it does not support is excluded from generation for oracle 3 and listed in its module doc, and the list shrinks over time.

## Constraints

* The generator and evaluator live under `tests/generated/` as an integration test that uses the public API (`syntax::parse`, `check::check`, `Interpreter`, `Options`). No test-only code goes into the library except the planner option.
* Found bugs are fixed in their own commits, each with its datadriven regression case, so the history shows what the generator found. A bug whose fix is out of scope is recorded as an issue, and its case is excluded from generation with a pointer to the issue.
* The evaluator's float reductions follow the canonical order exactly, or float reductions are excluded from oracle 3 with that recorded. Bit-identical agreement is required, not approximate agreement.

## Left to the builder

* The shape of the strategy, the weights, and the size limits.
* The planner option's name and how `Options` exposes it.
* How oracle results are compared (rendered output text or values), as long as floats compare by bits.
