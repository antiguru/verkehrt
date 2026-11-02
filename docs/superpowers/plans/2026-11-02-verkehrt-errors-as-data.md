# Errors as data implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make runtime errors depend only on what a program outputs: failed bindings hold their error, outputs raise it, and the checker rejects unread bindings.

**Architecture:** The checker gains an unread-binding rule with a `_` opt-out. The executor's environment maps each variable to a value or a runtime error. A `Bind` statement stores its error, and a read of the binding propagates the stored error with its original span. `Print`, exports, and REPL top-level bindings are outputs and raise. Exports become positional, so an export fails at its own statement. Oracle 3 of `tests/generated/` drops its #40 stopgap and the generator emits deliberately failing `_` bindings.

**Tech Stack:** Rust, the datadriven tests in `tests/programs/` and `src/*/testdata/`, the `tests/generated/` oracles.

**Spec:** `docs/superpowers/specs/2026-11-02-verkehrt-errors-as-data-design.md`

## Global Constraints

* Only `ErrorKind::Runtime` errors become data. Usage and I/O errors fail immediately. A panic never stands in for a semantic error.
* Outputs are `print`, exports (each at its own statement), and in the REPL expression statements and top-level `let`s. Outputs run in source order, and the first failing output fails the program with every earlier print intact.
* The unread-binding rule covers top-level and block `let`s, function parameters, and lambda parameters. A name starting with `_` opts out. Top-level `let`s are exempt in the REPL. An export counts as read.
* No change to kernels, plans, or the planner.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit. The debug CI run of the generated test stays well under 30 s.
* No `#[path]`, no `mod.rs`, `#[expect]` not `#[allow]`. Comments have no em-dashes or structuring semicolons. Docs name no commit hashes.

## Review Focus

* A read of a failed binding inside the untaken branch of a `Cond` must not raise. Only the taken branch's reads count.
* The REPL: after a failed input, the environment rollback still drops the new variables, and a failing top-level `let` raises at its statement.
* `Prepared::execute`: an export that holds an error fails at its position, so a later `print` does not appear in `diag`.
* The checker rule must see reads through closures: a lambda reading an outer `let`, a function body reading its parameter only inside a lambda, and a `let` read only by an export.
* Oracle 3 must stay strict: no skip path may remain, and the generator gate must count failing `_` bindings actually generated, not shapes offered.

---

### Task 1: Unread-binding rule

**Files:**
* Modify: `src/check.rs` (or a new `src/check/unread.rs` submodule), `src/interp.rs` (pass the mode to the checker)
* Modify: checker datadriven cases under `src/check/` testdata or `tests/programs/errors.dd`
* Modify: every `.vk` and `.dd` program the rule rejects, and `tests/generated/gen.rs` plus `gen/` so generated programs never leave a binding unread without a `_` prefix

- [ ] **Step 1:** Add datadriven cases: an unread top-level `let` (type error naming the binding, at its span), an unread block `let`, an unread function parameter, an unread lambda parameter, each with a `_`-prefixed variant that checks. Add cases where the read is through a lambda, through an export, and only in one branch of an `if`, all of which check. Add a REPL case where an unread top-level `let` is accepted. Run them and watch them fail.
- [ ] **Step 2:** Implement the rule in the checker. The checker needs to know the mode, so `check` gains a parameter (or a flag on `Globals`) that `Interpreter::compile` sets from `Mode`. The error is `ErrorKind::Type` with a message of the form `'name' is never read, prefix it with '_' if intended`.
- [ ] **Step 3:** Run the whole suite and fix every program the rule rejects, preferring a `_` prefix over deleting the binding, unless the binding is plainly dead. Change the generator so every binding it does not read gets a `_` prefix. Run a 2,000-case release search of the generated test.
- [ ] **Step 4:** Commit "Reject bindings nothing reads".

### Task 2: Failed bindings hold their error

**Files:**
* Modify: `src/exec.rs` (environment slots, outputs), `src/interp.rs` (REPL demand, exports, `get`), `src/lower.rs` (positional exports)
* Modify: `tests/programs/errors.dd` (or a new `tests/programs/demand.dd`), the base spec's error section in `docs/superpowers/specs/2026-10-30-verkehrt-design.md` if it states eager failure

- [ ] **Step 1:** Add `run` cases: an unread failing `_` binding with a later `print` that succeeds; a failing binding read only in the untaken branch of a scalar `if`; a failing binding with two readers read in a taken branch, raising with the binding's line; prints before a failing `print`, which appear in full; a failing export followed by a `print` that must not appear (through `Prepared::execute`, so this one is a Rust test in `src/interp` tests or `tests/`); a REPL case where a failing top-level `let` raises at its line. Run them and watch them fail.
- [ ] **Step 2:** Represent a failed binding in the environment. Pick the representation (for example `Result<Value, Error>` slots or a wrapper type) and keep `Interpreter::get` and `bind` working for hosts. A `Bind` statement whose evaluation fails with `ErrorKind::Runtime` stores the error. `Leaf::Var` of a stored error returns that error unchanged, so its span is the binding's statement. Other error kinds still fail immediately.
- [ ] **Step 3:** Make exports positional: lowering records, for each export, the statement it binds, so `run_plan` treats that statement as an output and fails there when it holds an error. In `Mode::Repl`, every top-level `Bind` is an output. Keep the REPL rollback on failure.
- [ ] **Step 4:** Run the suite. Update the base spec's error text if it says bindings fail eagerly, citing the new spec. Commit "Keep runtime errors in the bindings that raise them".

### Task 3: Strict oracle 3 and failing unread bindings

**Files:**
* Modify: `tests/generated/oracles.rs`, `tests/generated/eval.rs`, `tests/generated/gen.rs` and `gen/`, `tests/generated/main.rs` (gate)
* Modify: `docs/measurements/2026-11-01-generated-programs.md` (note that the stopgap is gone), `tests/programs/generated.dd` if it pins stopgap behavior

- [ ] **Step 1:** Remove the #40 stopgap: the `eager_let` parameter of `eval::evaluate`, `eager_reproduces`, and the skip result in oracle 3. Oracle 3 then either agrees or fails.
- [ ] **Step 2:** Make the reference evaluator report, with each collected error category, the line of every statement on the chain from the demanding output to the failing expression (each forced `let` or export thunk adds its line). `agrees` accepts the pipeline's line if it is on the chain of an error with the same category. Document the rule in the module comment of `eval.rs`, citing the span rule of the spec.
- [ ] **Step 3:** Make the generator emit `_`-prefixed top-level and block bindings that nothing reads, some of which fail (empty `min`, a gather out of range, a length mismatch). Add a gate to the CI run that fails when fewer than a handful of generated cases contain a failing unread binding, counted from cases where the reference evaluator, forcing that binding alone, fails. Run a 5,000-case release search and record the result in the findings record.
- [ ] **Step 4:** Commit "Compare oracle 3 strictly now that errors follow demand".
