# Datadriven tests implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the planner, checker, lowering, and golden-program tests to `datadriven` files, without losing any case.

**Architecture:** Each test module gets a small harness function that walks its `testdata` directory with `datadriven::walk` and dispatches on the directive. Inputs and expected outputs move to data files. Rust-only tests (structural checks, value tests) stay in place.

**Tech Stack:** Rust, `datadriven` 0.9 (dev-dependency), rewrite mode via `REWRITE=1`.

**Spec:** `docs/superpowers/specs/2026-11-01-verkehrt-datadriven-design.md`

## Global Constraints

* No case is lost. Before converting a module, count its cases. After converting, count the blocks in its data files plus the Rust tests that remain, and record both counts in the commit message.
* Expected output must not change during the conversion. Rewrite mode on the converted files produces no diff.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit.
* Comments have no em-dashes or structuring semicolons. No `#[path]`. Large test modules go in `<module>/tests.rs`.

## Review Focus

* A converted case that silently asserts less than before, for example a full error message reduced to its first line.
* Cases that depend on test order or shared state, which a file-driven harness could reorder.
* Program cases whose output differs across instruction sets, which the new cross-variant check would expose. Report them as findings, do not hide them.

---

### Task 1: Planner tests

**Files:**
* Modify: `Cargo.toml` (dev-dependency `datadriven = "0.9"`), `src/plan/tests.rs`
* Create: `src/plan/testdata/*`

- [ ] **Step 1:** Add the dependency and a harness `#[test] fn datadriven_plans()`.
  * The `plan` directive takes a source (the `XY` prefix stays available, as `prefix=xy` or a default) and prints one line per `print` with the root kernel name and the rendered plan. Use the current `plan_prints` and `plan_print` logic.
  * The harness keeps calling `assert_broadcasts_fit` on every plan.
- [ ] **Step 2:** Move every case that asserts plan text into data files, grouped by topic. Keep `egglog_smoke` and any test that checks something other than the rendered plan as Rust tests.
- [ ] **Step 3:** Run the tests, then `REWRITE=1` on the data files, and confirm there is no diff. Commit "Move the planner tests to datadriven files".

### Task 2: Checker and lowering tests

**Files:**
* Modify: `src/check/tests.rs`, `src/lower/tests.rs`
* Create: `src/check/testdata/*`, `src/lower/testdata/*`

- [ ] **Step 1:** Checker: a `type` directive prints the type of the last binding, or `error: <message> at <line>:<col>`. Every case pins the full message and span, so an assertion that used to check a substring now checks the whole text.
- [ ] **Step 2:** Lowering: a `lower` directive with `mode=run|repl` prints the current `dump` output.
- [ ] **Step 3:** Run the tests, then `REWRITE=1`, and confirm there is no diff apart from the deliberate widening of substring checks to full messages. List those in the commit message. Commit "Move the checker and lowering tests to datadriven files".

### Task 3: Golden programs

**Files:**
* Modify: `tests/programs.rs`
* Create: `tests/programs/*.txt` (data files), and delete each `.vk`/`.out` pair once its case is in a data file

- [ ] **Step 1:**
  * A `run` directive executes the program on every instruction set the CPU supports (`Isa::ALL` filtered by `is_available`) with both backends, and prints the output. It fails if any variant's output differs from the scalar run.
  * An `error` directive prints the runtime error kind, message and line.
- [ ] **Step 2:** Convert every `.vk`/`.out` pair and every test in `tests/programs.rs` that checks program output or a runtime error. Keep the library-API tests (`Prepared`, `Options`) as Rust tests.
- [ ] **Step 3:** Run the tests, then `REWRITE=1`, and confirm there is no diff. Commit "Move the golden programs to datadriven files".
