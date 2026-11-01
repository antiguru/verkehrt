# Verkehrt: datadriven tests for text-in, text-out cases

This spec moves the tests that compare a source text with a rendered output to the `datadriven` crate. It is tracked as issue #19.

## Why

Many tests hand-write an assertion over rendered text, and each one carries its own boilerplate:
* `src/plan/tests.rs` asserts plan strings such as `(dot_f64 x y)`.
* `src/check/tests.rs` asserts `last_type(src)` or a substring of the type error.
* `src/lower/tests.rs` asserts the dump of the lowered statements.
* `tests/programs` pairs each `.vk` program with an `.out` file through a hand-rolled harness.

`datadriven` expresses each case as a directive block in a file, with the input above a `----` separator and the expected output below it. A rewrite mode regenerates the expected output, so a change in rendering shows up as a reviewable diff instead of edits across many assertions. The planner work of #30 will add many small cases, so it gains the most.

## Goals

* The `datadriven` crate (0.9) as a dev-dependency.
* **Planner.** Cases become `plan` blocks: the source in, one rendered plan per `print` out. The existing structural checks, such as `assert_broadcasts_fit`, keep running inside the harness for every case.
* **Checker.** Cases become `type` blocks: the type of the last binding, or the full error with its span. Every case then pins the complete message and span, not a substring.
* **Lowering.** Cases that assert the dump become `lower` blocks, with an argument for the mode (`mode=run` or `mode=repl`). Tests that check something other than the dump stay as Rust tests.
* **Programs.** `tests/programs/*.vk` and `.out` become `run` blocks in a few files grouped by topic. The harness runs each case on every instruction set and backend the CPU supports, and requires identical output, which adds a cross-variant check at the program level. An `error` directive covers the runtime errors that `tests/programs.rs` asserts today.
* Case counts are preserved. Each converted case appears in a data file, and the conversion deletes no case. Rewrite mode produces no diff on the converted files.

## Non-goals

* The property tests (`tests/isa_variants.rs`, `tests/fused_oracle.rs`, the kernel oracles) and the kernel unit tests stay as they are. They check values over generated inputs.
* The acceptance suite stays as it is. It compares with DuckDB's files, which `prepare.py` writes.
* No change to what is tested, only to how cases are written. A case whose expected output changes during the conversion is a bug in the conversion.

## Constraints

* Rewrite mode is the crate's own mechanism, run with `REWRITE=1 cargo test`. Expected output that rewrite mode changed is committed only after reading the diff, as with `cargo insta accept`.
* Data files live next to the code they test, for example `src/plan/testdata/` and `tests/programs/`. The harness for unit tests stays inside the crate so it can use private APIs.
* Output must be deterministic. If a case's output depends on the instruction set (it should not), the harness reports it as a failure.

## Left to the builder

* The file layout and grouping of cases, and the directive names beyond `plan`, `type`, `lower`, `run`, and `error`.
* How the harness prints types, errors, and plans, as long as the rendering matches what the current assertions check.
