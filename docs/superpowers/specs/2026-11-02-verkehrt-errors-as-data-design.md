# Verkehrt: errors as data

This spec makes runtime errors depend only on what a program outputs, not on how lowering shares its bindings. It is tracked as issue #40 and lands on top of the generated-program oracles (#35), whose oracle 3 found the problem and carries a stopgap until this lands.

## Why

Whether a program fails depends today on how often a variable is used. A top-level `let` that lowering does not inline is computed as its own statement, so `let v = reduce(fill(0, 1.0), min); print(1);` fails with an empty-reduction error even though nothing reads `v`. A single-use or block `let` is substituted into its reader instead, and it is computed only when that reader runs. A `let` read only in the untaken branch of a scalar `if` fails if it has two readers and succeeds if it has one. The result depends on an optimization, which is what the base spec's ordering rule forbids.

## Goals

* **Demand.** A semantic runtime error is raised only when its value reaches an output. Outputs are `print` statements and exports, plus the echo of an expression statement in the REPL. A value that no output reads, directly or through other bindings, never raises, whatever lowering does with it.
* **Errors as data.** Execution stays eager. When a binding's computation fails with a runtime error, the binding holds that error instead of a value, and the error is raised when an output reads the binding, directly or through another binding. The error keeps the span of the statement that computed it, so a binding read through another statement reports its own line and not the reader's. Lowering substitutes a single-use `let` into its reader, and an error inside such a `let` reports the reader's line. Lowering is the same for every plan and variant, so this does not break bit identity.
* **REPL.** In the REPL a top-level `let` stays bound for later inputs, so the REPL treats it as demanded: a failing top-level `let` raises at its statement, as today. The unread-binding rule below exempts top-level `let`s in the REPL for the same reason.
* **Output order.** Outputs are produced in source order. An export is an output at its own statement. The first output whose value is an error fails the program with that error, and the prints before it appear in full.
* **No unread bindings.** The checker rejects a `let` (top-level or block), a function parameter, or a lambda parameter that nothing reads. A name that starts with `_` opts out. A broken statement then cannot sit unread in a program, and laziness hides errors only in untaken branches.
* **Lazy branches stay lazy.** A scalar `if` evaluates only its taken branch, as the base spec says. This includes an `if` inside a lambda whose condition does not depend on a lambda parameter, which #35 already lowers to a lazy branch.
* **Oracles.** Oracle 3 of #35 loses its stopgap and compares strictly. Because of the span rule above, it accepts the line of any statement on the chain from the demanding output to the failing expression, as long as the error category matches. The generator emits `_`-prefixed bindings for values it does not read, and some of them deliberately fail, so deferral is exercised.

## Non-goals

* No per-row errors in the SQL sense, where one bad row does not fail a query. That is the `Result` analogue of the nullable scalars on #13.
* No nullable or optional values. They are #13's next step and build on this.
* No rule for unused functions or unused inputs. An input is supplied by the caller, and a function nothing calls never runs.
* No warning infrastructure. The unread-binding rule is an error, like Go's.

## Constraints

* **Error classes.**
  * Semantic errors are the errors the language defines: length mismatches, empty `min`, `max`, `argmin`, and `argmax`, gather indices out of range, `div_const` by zero or a divisor that is too large, bucket keys out of range, and negative counts or lengths. They follow the demand rule and are part of the bit-identity guarantee.
  * Panics are everything outside the semantics: allocation failure (#14) and broken internal invariants. They abort, do not follow the demand rule, and are not part of the guarantee. A panic must never stand in for a semantic error.
  * Usage and I/O errors (missing inputs, a failed write) are not runtime values and keep failing immediately.
* **Which error.** The base spec leaves unspecified which error is reported when several would occur. That stays, with one refinement: whether the program fails is fixed by demand, and all variants of one plan report the same error. Which of a statement's errors surfaces may differ between plans, as in Peyton Jones et al., "A semantics for imprecise exceptions" (PLDI 1999). Once kernels fuse across statements, the same latitude extends to the statements they fuse.
* **Element-wise errors in a selected `if`.** An `if` whose condition depends on a lambda parameter computes both branches and selects per element, so an error in either branch over any element is demanded whenever the `if` is. The base spec already says this, and this spec keeps it.
* **Cost.** A binding holds a value or an error, so the change is one branch per statement and no change to kernels or plans.
* **Tests.** Every rule above gets datadriven cases: checker cases for the unread-binding rule and the `_` opt-out, and program cases for an unread failing binding, one read only in an untaken branch, one read in a taken branch (raising with the span where it arose), and prints before a failing output.

## Left to the builder

* How the environment represents a failed binding, and how the error travels through reads.
* The checker's message wording and where the rule lives in the checker.
* How the generator chooses which unread bindings fail.
