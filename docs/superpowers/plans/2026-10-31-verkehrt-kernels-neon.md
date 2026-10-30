# Verkehrt NEON kernels implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NEON intrinsics for the kernels the compiler does not vectorize on aarch64 (selections and float min/max/argmin/argmax), bit-identical to scalar, tested in CI on Apple Silicon and measured on the user's M5 (and optionally M1).

**Architecture:** First an architecture layer removes the x86-specific call sites from shared code and moves the autovectorization macro to shared code, parameterized by its feature string. On aarch64 the autovectorization backend is the baseline build: NEON is mandatory there, so the scalar entries are already NEON-vectorized, and the `neon/auto` slot stays empty and falls back to scalar (decision recorded below). The intrinsics backend then gets hand-written NEON entries for compress, `where`, the fused filter, and the float extreme reductions.

**Tech stack:** Rust 1.95+, `std::arch::aarch64`, `criterion`, `proptest`.

**Spec:** `docs/superpowers/specs/2026-10-30-verkehrt-design.md`

**Plan series:** Plan 3 of 3. Plans 1 (interpreter) and 2 (x86 kernels, PR #2) are merged. Issue #3 tracks the selection sizing trade-off and is out of scope here.

## Decisions taken before this plan

* NEON auto (user decision, option 1): on aarch64 the auto backend is the baseline build. No `neon/auto` entries exist, `Variant { isa: Neon, backend: Auto }` resolves to scalar, and the trace prints `isa=scalar` for those kernels. Intrinsics are measured against the scalar entry, which on aarch64 is NEON-autovectorized code.
* Keeping a pivot cheap: the `neon/auto` slot stays in `Variant`, and the auto macro lives in shared code with the feature string as a parameter. Pivoting to an auto backend with optional features (for example `"dotprod,fp16,rdm"`) is then one macro invocation plus an `is_aarch64_feature_detected!` check, and `tests/isa_variants.rs` covers the new entries without changes.
* Split of work: Task 1 runs on the x86 development machine (a refactor that x86 tests and the macOS CI job verify). Tasks 2 to 4 run in a separate agent session on the M5 MacBook, because NEON intrinsics need a fast native edit, test, and measure loop. See the hand-off section.

## Global constraints

* Edition 2024, `rust-version = "1.95"`. CI runs `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and tests on ubuntu and macOS (Apple Silicon) with stable and 1.95. Build with `CARGO_INCREMENTAL=0`.
* Modules use `name.rs` plus a `name/` directory, never `name/mod.rs`. Large test modules go in `name/tests.rs`.
* `unsafe` appears only in `src/kernels/dispatch.rs`, under `src/kernels/x86/` (and `src/kernels/x86.rs`), under `src/kernels/aarch64/` (and `src/kernels/aarch64.rs`), in `src/kernels/auto.rs` for the shared auto macro, and in `src/kernels/select_loops.rs` for the shared selection loops macro. Every `unsafe` block or `unsafe fn` carries a `// SAFETY:` comment naming the CPU features and pointer or length assumptions it relies on.
* NEON entries use `#[target_feature(enable = "neon")]`. `Isa::Neon.is_available()` is true on every aarch64 target.
* Every variant is bit-identical to scalar for non-NaN results, a NaN result only has to be NaN. Float `min`/`max` are IEEE 754-2019 `minimum`/`maximum` (NaN propagates, `-0.0 < +0.0`). The AArch64 `FMIN`/`FMAX` instructions (`vminq_*`/`vmaxq_*`) have exactly these semantics, `FMINNM`/`FMAXNM` do not and must not be used. No FMA.
* Intrinsics entries check the lengths they rely on for memory safety themselves (the `Dispatch` tile paths pass arguments unchecked, see the `ReducerFn` and `SelectionFn` docs).
* Selections size their output exactly with a count pass and append through `append_chunks` (see the spec's tile section). NEON entries follow the same pattern as the x86 intrinsics.
* Comments: no em-dashes, no structuring semicolons. Doc comments state contracts, reasoning goes inline. Never drop existing comments.
* Commits end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -A`.

## Review focus

* An aarch64 build must not compile any x86 module and vice versa, and an `x86_64` build must still register exactly the entries it did before Task 1. Test in Task 1 (entry-count snapshot per variant).
* NEON selection tails: lengths 0 to 17 and around 2, 4, 8, 16 lanes, misaligned views, dense-prefix and empty-suffix masks. Covered by `tests/isa_variants.rs` once Task 1 includes `Isa::Neon`.
* NaN and signed zero through NEON `reduce_min`/`reduce_max`/`argmin`/`argmax`, including a NaN in a later tile than the extreme and ties across vector lanes (first index wins). Test in Task 3.
* `--isa neon --backend auto` on aarch64 resolves to scalar and the trace says `isa=scalar`. Test in Task 1 (cfg aarch64, runs in macOS CI).

## File structure

| File | Responsibility |
|---|---|
| `src/kernels/arch.rs` | The architecture layer: `add_elementwise`, `add_reduction`, `add_selection`, `add_compress`, `add_where`, `add_filter`, `add_extreme`, dispatching to `x86` or `aarch64` by `cfg`, no-op elsewhere. The only place shared code reaches architecture code. |
| `src/kernels/auto.rs` | The shared `auto_backend!` macro, parameterized by module name, display name, and feature string. |
| `src/kernels/x86.rs` | Unchanged role, now reached through `arch` and invoking the shared macro. |
| `src/kernels/aarch64.rs`, `src/kernels/aarch64/compress.rs`, `src/kernels/aarch64/extreme.rs` | NEON intrinsics entries. |
| `tests/isa_variants.rs` | Includes `Isa::Neon`. |
| `benches/tier1.rs` | Adds min and argmin workloads, generalizes the crossover sweep to every architecture. |
| `docs/measurements/2026-10-31-aarch64-kernels.md` | M5 (and optionally M1) measurements. |

---

### Task 1: Architecture layer (x86 development machine)

**Files:**
* Create: `src/kernels/arch.rs`, `src/kernels/auto.rs`, `src/kernels/aarch64.rs`
* Modify: `src/kernels.rs` (the `#[cfg(target_arch = "x86_64")] x86::add_*` lines in `KernelImpl::elementwise/reduction/selection` and the `mod x86` declaration), `src/kernels/scalar/select.rs` (the `x86::add_compress` and `x86::add_where` call sites), `src/kernels/scalar/fused.rs` (the `x86::add_filter` call site), `src/kernels/x86.rs`, `src/kernels/x86/auto.rs` (removed, its macro moves), `tests/isa_variants.rs`, `docs/superpowers/specs/2026-10-30-verkehrt-design.md`

**Interfaces:**
* Produces in `src/kernels/arch.rs`, all `pub(crate)`:
  * `fn add_elementwise<B: ElementwiseBody>(impls: &mut [Option<KernelImpl>])`, `fn add_reduction<R: ReductionBody>(...)`, `fn add_selection<S: SelectionBody>(...)`: forward to `x86::add_*` on `x86_64`, to `aarch64::add_*` on `aarch64`, and do nothing elsewhere.
  * `fn add_compress<T: Elem, const BOOL: bool>(...)`, `fn add_where(...)`, `fn add_filter<T: Elem, const C: usize>(...)`: same forwarding.
  * `fn add_extreme<T: Elem, const MAX: bool>(...)` and `fn add_arg_extreme<T: Elem, const MAX: bool>(...)`: hooks for Task 3, forwarding to `aarch64` on `aarch64` and doing nothing elsewhere (x86 has no extreme intrinsics). Call them from the registration of `reduce_min_*`/`reduce_max_*` and `argmin_*`/`argmax_*` in `src/kernels/scalar/reduce.rs`.
* Produces in `src/kernels/auto.rs`: the `auto_backend!` macro moved from `src/kernels/x86/auto.rs` unchanged in behavior, still taking `($isa:ident, $name:literal, $features:literal)`. `x86.rs` invokes it for `avx2` and `avx512` with the same feature strings as before.
* Produces in `src/kernels/aarch64.rs`: the same `add_*` function set as `x86.rs`, all empty for now except for a doc comment. Task 2 and Task 3 fill them. The module doc states the NEON auto decision (auto is the baseline build, the `neon/auto` slot stays empty, how to pivot).
* After this task, `grep -rn 'x86::' src --include=*.rs` outside `src/kernels/x86*` and `src/kernels/arch.rs` finds nothing, and the only `target_arch` uses outside `arch.rs`, `x86*`, `aarch64*`, and `ops.rs` are the `mod` declarations in `src/kernels.rs`.

- [ ] **Step 1: Write the failing tests.**
  * `src/kernels/tests.rs`: `variant_entry_counts_are_stable` counts, per `Variant`, the kernels that have an entry in that exact slot, and asserts the counts on `x86_64` equal the counts before this task (record them first by running the test with a print against the unchanged code, then paste them as literals). On `aarch64` the expected counts are the scalar count and zero for every other slot (until Task 2).
  * `tests/isa_variants.rs`: iterate over `Isa::ALL` except `Scalar` instead of `[Avx2, Avx512]`, skipping unavailable ISAs as today. The `variant_used` assertions that expect auto entries apply to x86 ISAs only. Add, under `#[cfg(target_arch = "aarch64")]`, `neon_auto_resolves_to_scalar`: `Registry::new().resolve(Isa::Neon, Backend::Auto).variant_used(id) == Variant::SCALAR` for `add_f64`, `reduce_add_f64`, `dot_f64`, `compress_f64`.
  * `tests/programs.rs` or `tests/cli.rs`, under `#[cfg(target_arch = "aarch64")]`: running `sum(x * y)` with `--isa neon --backend auto --trace` prints `isa=scalar` for `dot_f64`.
- [ ] **Step 2: Run** the suite. Expected: `variant_entry_counts_are_stable` passes on x86 before the refactor (it pins current behavior), the aarch64 tests compile out on x86.
- [ ] **Step 3: Implement** the layer, move the macro, replace the call sites.
- [ ] **Step 4: Run** `cargo fmt --all`, clippy, full `cargo test --locked`, `cargo test --release --test isa_variants`, and the aarch64 type check `CC_aarch64_apple_darwin=<stub> cargo check --locked --target aarch64-apple-darwin --all-targets` (the stub compiler only satisfies the `libmimalloc-sys` build script, see the plan 2 final review). Push the branch and confirm the macOS CI jobs pass, since they are the first aarch64 run.
- [ ] **Step 5: Update the spec**: in "Execution model", state that on aarch64 the auto backend is the baseline build (NEON is mandatory), so `neon/auto` resolves to scalar, and that an auto backend with optional aarch64 features is a possible later pivot.
- [ ] **Step 6: Commit** `Add an architecture layer and share the auto backend macro`.

---

### Task 2: NEON intrinsics for compress, `where`, and fused filter (M5 session)

**Files:**
* Create: `src/kernels/aarch64/compress.rs` (and `src/kernels/aarch64/compress/` submodules if it grows)
* Modify: `src/kernels/aarch64.rs`, `tests/isa_variants.rs` (the intrinsics `variant_used` assertions for Neon)

**Interfaces:**
* Consumes: Task 1 hooks, `SelectionFn`, `ArrayBuilder`, `append_chunks`, the scalar entries' argument parsing helpers (as the x86 intrinsics use them, see `src/kernels/x86/compress.rs`).
* Produces: `Variant { isa: Neon, backend: Intrinsics }` entries for `compress_{f64,f32,i64,i32,u8,bool}`, `where`, and `filter_<cmp>_{f64,f32,i64,i32}`.
* Technique: AArch64 has no compress instruction. Per 128-bit vector, turn the lane mask into a small integer (2 bits for 64-bit lanes, 4 bits for 32-bit lanes) and index a precomputed byte-shuffle table, then `vqtbl1q_u8` the vector and store it whole, advancing the output by the popcount. For `u8`/`bool` (16 lanes), process each 8-byte half with an 8-bit mask, a 256-entry table of 8-byte shuffles, and `vtbl1_u8`/`vqtbl1q_u8`, storing 8 bytes and advancing by the popcount. The fused filter compares with `vcgtq_*`/`vcleq_*`/`vceqq_*` (floats with the same NaN behavior as Rust `PartialOrd`: an ordered compare is false on NaN, `!=` is the negation of `==`) and selects from the original bits. Count pass: the same compares with lane popcounts (`vaddvq`).
* Store bounds: a whole-vector store at output position `k` needs `k + LANES <= dst.len()` as in the x86 entries, otherwise finish with the portable predicated loop.

- [ ] **Step 1: Run** `CARGO_INCREMENTAL=0 cargo test --release --test isa_variants` and confirm Neon is exercised (it prints skips for x86 ISAs). It is the test for this task, extend its `variant_used` intrinsics assertions so these kernels resolve to `neon/intrinsics`.
- [ ] **Step 2: Implement** the entries, compress first, then `where`, then filter.
- [ ] **Step 3: Run** fmt, clippy, full `cargo test --locked`, release `isa_variants`. Expected: all pass.
- [ ] **Step 4: Commit** `Add NEON intrinsics for compress and fused filter`.

---

### Task 3: NEON intrinsics for float min, max, argmin, argmax (M5 session)

**Files:**
* Create: `src/kernels/aarch64/extreme.rs`
* Modify: `src/kernels/aarch64.rs`, `src/kernels/dispatch/tests.rs` or `tests/isa_variants.rs`

**Interfaces:**
* Consumes: the `add_extreme`/`add_arg_extreme` hooks from Task 1, `ReducerFn`, `Reducer`.
* Produces: `neon/intrinsics` reducers for `reduce_min_{f64,f32}`, `reduce_max_{f64,f32}`, `argmin_{f64,f32}`, `argmax_{f64,f32}`.
* Technique: `vminq_f64`/`vmaxq_f64` (and `_f32`) accumulate per lane, reduced across lanes with `vminvq`/`vmaxvq` at `finish` (all `FMIN` family, never `FMINNM`). NaN propagates by the instruction semantics. For argmin/argmax, keep a value vector and an index vector per lane, update with a strict compare and `vbslq`, and track the first NaN index separately (once a NaN is seen, the result is that index). At `finish`, pick the lane with the extreme value and, among equal values, the smallest index, honoring `-0.0 < +0.0`. The reducer counts consumed rows so indices are global across tiles.
* Empty input keeps the existing errors.

- [ ] **Step 1: Write the failing tests**: in `tests/isa_variants.rs` or a dedicated test, the tiled cases from plan 2 Task 2 (first NaN in an early tile with a lower value later, ties across a vector lane boundary and a tile boundary, `-0.0`/`+0.0` pairs) run against the `neon/intrinsics` reducer and must equal scalar. The generic `isa_variants` sweep covers the rest once entries exist.
- [ ] **Step 2: Implement** the reducers.
- [ ] **Step 3: Run** fmt, clippy, full suite, release `isa_variants`.
- [ ] **Step 4: Commit** `Add NEON intrinsics for float extreme reductions`.

---

### Task 4: Measurements on the M5 (M5 session)

**Files:**
* Create: `docs/measurements/2026-10-31-aarch64-kernels.md`
* Modify: `benches/tier1.rs`

**Interfaces:**
* Consumes: everything above.

Bench changes:
* Add workloads `min` (`reduce(x, min)`) and `argmin` on the same 10M `f64` data, with native baselines (a canonical-free plain loop is fine for min since min is order-independent, for argmin the same first-index rule).
* Generalize the crossover sweep so it runs on aarch64 too, comparing the best available variant against the baseline-target native loop.
* Variants on aarch64: `scalar` (the NEON-autovectorized baseline build), `neon-intrinsics` (where entries exist, otherwise it equals scalar and is reported as such), `native-base`, and corgi.

Document:
* Machine facts: chip (`sysctl -n machdep.cpu.brand_string`), macOS version, `rustc -V`, power state (plugged in, Low Power Mode off, recorded by the user or with `pmset -g`), and whether other work was running.
* The "before" numbers: the plan 2 benches on the M5 before Task 2, provided by the user on 2026-10-31:

  | Workload | Verkehrt | native-base | ratio | corgi |
  |---|---|---|---|---|
  | sum | 0.834 ms | 0.821 ms | 1.02 | 5.47 ms |
  | dot | 1.424 ms | 1.416 ms | 1.01 | 11.1 ms |
  | count | 0.866 ms | 0.865 ms | 1.00 | 1.33 ms |
  | filter 1% | 3.44 ms | 2.75 ms | 1.25 | 4.93 ms |
  | filter 50% | 3.91 ms | 2.79 ms | 1.40 | 5.10 ms |
  | filter 99% | 4.24 ms | 2.86 ms | 1.48 | 5.21 ms |
  | copy (read and write) | 1.113 ms (133.9 GiB/s) | | | |

* The "after" tables per workload: scalar, neon-intrinsics, native, corgi, ratios to native, pass/fail against 1.5x, intrinsics against scalar.
* The crossover table.
* State only what the numbers show. Note SME on the M5 as present and out of scope.

- [ ] **Step 1: Ask the user** to put the machine on power, turn Low Power Mode off, and close heavy applications before the run.
- [ ] **Step 2: Extend** the bench and run `CARGO_INCREMENTAL=0 cargo bench --locked --bench tier1` and `--bench corgi`.
- [ ] **Step 3: Write** the document.
- [ ] **Step 4: Commit** `Measure NEON kernels on the M5`, push the branch, open a draft pull request against `main`.

---

## Hand-off to the M5 session

The M5 session starts without this conversation. Give it this section and the plan path.

* Repository: `https://github.com/antiguru/verkehrt`, branch `kernels-neon` (pushed after Task 1). `git fetch && git switch kernels-neon`, then work in a worktree per the user's global instructions.
* Read first: this plan (`docs/superpowers/plans/2026-10-31-verkehrt-kernels-neon.md`), the spec, and `docs/measurements/2026-10-30-x86-kernels.md` for how measurements are written.
* Execute Tasks 2 to 4 with superpowers:subagent-driven-development. Task 1 is done when this hand-off happens; confirm with `git log` that the commit `Add an architecture layer and share the auto backend macro` is on the branch.
* Reference implementations for the NEON entries: `src/kernels/x86/compress.rs`, `src/kernels/x86/compress/avx2.rs`, `src/kernels/x86/compress/avx512.rs` (argument parsing, count pass, `append_chunks`, store bounds, SAFETY style).
* Commands: `CARGO_INCREMENTAL=0 cargo test --locked`, `CARGO_INCREMENTAL=0 cargo test --release --locked --test isa_variants`, `CARGO_INCREMENTAL=0 cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, benches as in Task 4.
* Expected state at hand-off: on the M5, `isa_variants` reports the x86 ISAs as skipped and exercises no Neon entries yet (the `neon/auto` slot is empty by decision, and `neon/intrinsics` is filled by Task 2).
* Finish: push `kernels-neon`, open a draft pull request against `main` (never mark it ready), and report the measurement summary.
