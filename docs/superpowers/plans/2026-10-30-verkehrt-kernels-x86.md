# Verkehrt x86 kernels implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AVX2 and AVX-512 variants of the tier 1 kernels in two backends, an autovectorization backend (portable bodies compiled per ISA) and an intrinsics backend (hand-written where it pays), all bit-identical to the scalar reference, measured against native Rust and corgi, with the kernel interface reshaped so tile-at-a-time execution needs no kernel rewrite.

**Architecture:** First the kernel table changes from whole-array functions to class-specific slice-level entries (elementwise into a destination, reductions as state/update/finish, selections appending to a builder), and a `Dispatch` table resolved once per interpreter replaces per-call feature detection. Then every elementwise, reduction, selection, and fused kernel gets an autovectorization-backend variant for AVX2 and AVX-512 by compiling its portable body under `#[target_feature]`. A generic property test checks every variant against scalar. Hand-written intrinsics form a second backend that sits beside the autovectorized one instead of replacing it, so every intrinsic kernel has to justify itself against autovectorization in the benchmarks. The plan writes intrinsics for compress and the fused filter, which cannot autovectorize.

**Tech stack:** Rust 1.95+, `std::arch` (`x86_64`), `criterion` 0.8 for benchmarks, `corgi` from `frankmcsherry/WIP` as a benchmark-only dependency, `proptest`.

**Spec:** `docs/superpowers/specs/2026-10-30-verkehrt-design.md`

**Plan series:** Plan 2 of 3. Plan 1 (core interpreter, scalar kernels) is merged. Plan 3 adds NEON variants, tested in CI on Apple Silicon and benchmarked on the user's Macs.

## Global constraints

* Edition 2024, `rust-version = "1.95"`. The `[lints.clippy]` table in `Cargo.toml` applies, CI runs `cargo clippy --locked --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check`. Build with `CARGO_INCREMENTAL=0`.
* Modules use `name.rs` plus a `name/` directory, never `name/mod.rs`. Large test modules go in `name/tests.rs`, declared with `#[cfg(test)] mod tests;`.
* `unsafe` appears only in `src/kernels/dispatch.rs` (calling resolved entries) and under `src/kernels/x86/`. Every `unsafe` block or `unsafe fn` carries a `// SAFETY:` comment naming the CPU features and pointer or alignment assumptions it relies on.
* A kernel variant is identified by `Variant { isa: Isa, backend: Backend }` with `pub enum Backend { Auto, Intrinsics }`. `Isa::Scalar` has only the scalar entry. The user picks the ISA (`--isa`) and the backend (`--backend auto|intrinsics`, default `intrinsics`). With `intrinsics`, a kernel uses its intrinsics entry, else its auto entry, else scalar. With `auto`, it uses its auto entry, else scalar. The trace prints the variant actually used as `isa=avx512 backend=auto`, or `isa=scalar` for scalar.
* ISA variants are installed only for an `Isa` whose `is_available()` is true. Feature strings: AVX2 is `#[target_feature(enable = "avx2")]`. AVX-512 is `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]`, matching `Isa::is_available`.
* Every ISA variant computes bit-identical results to the scalar kernel for every non-NaN result. A NaN result only has to be NaN. This includes the canonical 16-accumulator float order and IEEE `minimum`/`maximum`. No FMA.
* The scalar kernels stay the reference. They are not rewritten for speed.
* Comments: no em-dashes, no structuring semicolons. Doc comments state contracts, reasoning goes inline. Never drop existing comments.
* Commits end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -A`.
* Deferred, by design of this plan: writing outputs into a uniquely owned input buffer (the spec allows it, measure first), the tiled executor itself (this plan only makes kernels tile-capable), SIMD variants of construction kernels (`iota`, `fill`, `rand_*`), and the plan-1 follow-ups listed for the embedding plan (depth limit, tying `Prepared` to its registry, adopting host buffers).

## Review focus

* Misaligned and offset array views (a slice starting at row 1 to 7 of an aligned buffer) reaching an ISA variant must give the same result as aligned input. Test in Task 3.
* Lengths around vector widths (0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65) through every ISA variant, where tail handling bugs live. Test in Task 3.
* `--isa avx512` on a CPU without AVX-512 stays a usage error, and `--isa avx2` with a kernel that has no AVX2 variant falls back to scalar with the trace saying `isa=scalar`. Test in Task 1 (fallback) and kept passing after Task 3.
* A tiled call sequence (several elementwise calls into one destination, reducer updates of multiples of 16, selection appends with increasing `base`) must give exactly the one-shot result for every kernel, through the registered entries. Test in Tasks 1 and 2.
* Float NaN, `-0.0`, and infinities through SIMD `min`, `max`, `argmin`, `argmax`, and casts must keep the scalar semantics. Test in Task 3 (generators include them).

## File structure

| File | Responsibility |
|---|---|
| `src/kernels.rs` | `Registry`, `KernelDef`, `KernelImpl` and the class-specific entry types, lookup. |
| `src/kernels/dispatch.rs` | `Dispatch`: per-interpreter resolved table, the whole-array drivers per class, the only caller of entries. |
| `src/value.rs` | Adds `ArrayBuilder` (growable aligned output) and `Arg` (tile operand). |
| `src/kernels/scalar/*.rs` | Scalar entries reshaped to the new signatures. Behavior unchanged. |
| `src/kernels/x86.rs`, `src/kernels/x86/auto.rs` | Auto backend: `#[target_feature]` compilation of portable bodies, registration of AVX2 and AVX-512 entries. |
| `src/kernels/x86/compress.rs` | Intrinsics backend: hand-written compress and fused-filter entries. |
| `tests/isa_variants.rs` | Generic property test: every installed variant against scalar. |
| `benches/tier1.rs`, `benches/native.rs` (module of the bench) | Criterion benchmarks of tier 1 programs per ISA, native baselines, crossover sweep. |
| `benches/corgi.rs` | The same workloads in corgi. |
| `docs/measurements/2026-10-30-x86-kernels.md` | Measured results, ratios, crossover lengths. |

---

### Task 1: Class-specific kernel entries and the dispatch table, elementwise first

**Files:**
* Create: `src/kernels/dispatch.rs`
* Modify: `src/kernels.rs`, `src/value.rs`, `src/kernels/scalar/elementwise.rs`, `src/exec.rs`, `src/interp.rs`, and every scalar module only where the type change forces it
* Test: `src/kernels/tests.rs`, `src/kernels/dispatch/tests.rs`

**Interfaces:**
* Produces, in `src/value.rs`:
  * `pub enum Arg { Rows(Array), Splat(Scalar) }`, the operand of one tile. `Rows` is a view, so a tile is `array.slice(start, len)`.
  * `pub struct ArrayBuilder`, a growable 64-byte aligned buffer of one `ScalarType`: `fn new(elem: ScalarType) -> ArrayBuilder`, `fn with_capacity(elem, rows) -> ArrayBuilder`, `fn extend<T: Elem>(&mut self, rows: &[T])` (panics on a type mismatch, bool takes `u8`), `fn len(&self) -> usize`, `fn finish(self) -> Array` (no copy).
* Produces, in `src/kernels.rs`:
  * `pub type ElementwiseFn = unsafe fn(args: &[Arg], out: &mut [u8]) -> Result<(), String>`. Contract: `out` is the destination rows of the output type as bytes, every `Rows` argument has exactly `out.len() / width` rows, and the caller guarantees the CPU supports the entry's ISA.
  * `pub type ReducerFn = unsafe fn() -> Box<dyn Reducer>`, with `pub trait Reducer: Send { fn update(&mut self, args: &[Arg]) -> Result<(), String>; fn finish(self: Box<Self>) -> Result<Scalar, String>; }`. Contract: every `update` except the last receives a row count that is a multiple of 16, and `finish` follows the last `update` (an empty input means no `update` at all).
  * `pub type SelectionFn = unsafe fn(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String>`. Contract: appends the selected rows of this tile, `base` is the tile's first row index in the whole array.
  * `pub type WholeFn = fn(&[Value]) -> Result<Value, String>`, the former `KernelFn`, for class `Whole`.
  * `#[derive(Clone, Copy)] pub enum KernelImpl { Elementwise(ElementwiseFn), Reduction(ReducerFn), Selection(SelectionFn), Whole(WholeFn) }`.
  * `pub enum Backend { Auto, Intrinsics }` and `pub struct Variant { pub isa: Isa, pub backend: Backend }` with `Variant::SCALAR`, `fn index(self) -> usize` over the 7 slots (scalar, then auto and intrinsics for each of AVX2, AVX-512, NEON), and `Display` as `isa=scalar` or `isa=avx2 backend=auto`.
  * `KernelDef.impls: [Option<KernelImpl>; 7]` indexed by `Variant::index`. `Registry::add` asserts that every `Some` entry matches `def.class` and that the scalar slot is `Some`.
  * `Registry::select_impl` is removed. `pub fn resolve(&self, isa: Isa, backend: Backend) -> Dispatch` picks, per kernel, by the rule in Global constraints, installing ISA entries only when `isa.is_available()`, and records the variant picked. Feature detection runs once per `resolve`.
  * `Options` gains `backend: Backend` (default `Intrinsics`), and the CLI gains `--backend auto|intrinsics` on `run` and `repl` (unknown value is a usage error, exit 2).
* Produces, in `src/kernels/dispatch.rs`:
  * `pub struct Dispatch` (`Send + Sync`), `fn variant_used(&self, id: KernelId) -> Variant`, `fn call(&self, reg: &Registry, id: KernelId, args: &[Value]) -> Result<Value, String>`: the whole-array driver. It checks arity and argument kinds against `KernelDef.inputs`, unifies array lengths (`length mismatch: n vs m`), then: elementwise allocates the output (array or one-row scalar for `_s` kernels) and calls the entry once, reduction creates a reducer, updates once with all rows if any, and finishes, selection creates an `ArrayBuilder` and calls once with `base` 0, whole calls the function. The `unsafe` call carries `// SAFETY: resolve installs an entry only for an ISA whose is_available() returned true.`
* Consumes: everything in plan 1.

Scope of this task: the types, `Dispatch`, the executor and interpreter switched to `Dispatch` (resolved once in `Interpreter::new` and stored in the interpreter, `Prepared::execute` uses the interpreter's), and the elementwise scalar kernels (arith, compare, logic, select, casts, all `_s` kernels) moved to `ElementwiseFn`. Reductions, selections, fused, and construction kernels keep working by temporarily wrapping their existing whole-array functions as `KernelImpl::Whole` with their class unchanged, so `Registry::add` must allow `Whole` entries for any class until Task 2 (mark the allowance with a `// TODO(task 2)` comment that Task 2 removes).

- [ ] **Step 1: Write the failing tests.**
  * `src/kernels/dispatch/tests.rs`: `tiled_elementwise_matches_whole` calls the registered scalar entry of `add_f64`, `gt_f64`, `select_f64`, and `cast_f64_i32` through `Dispatch` internals over tiles `0..37`, `37..100`, `100..1000` of 1000-row inputs (one argument a `Splat`) into one destination, and asserts the bytes equal one `Dispatch::call` over all rows. `call_checks_lengths`: `add_f64` on 2 and 3 rows gives `Err` containing `length mismatch`. `resolve_falls_back_to_scalar`: `Registry::new().resolve(Isa::Avx2, Backend::Intrinsics)` reports `variant_used == Variant::SCALAR` for `add_f64` (no variants exist yet). `tests/cli.rs`: `--backend bogus` exits 2.
  * `tests/programs.rs` keeps passing unchanged, and the trace test still sees `isa=scalar`.
  * `array_builder_is_aligned_and_typed`: extend an `ArrayBuilder` of `f64` with 3 then 70 rows, `finish`, check rows and 64-byte alignment. Extending with `i64` rows panics.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test`. Expected: compile failures for the missing types.
- [ ] **Step 3: Implement** the types, `Dispatch`, the elementwise migration, and the executor/interpreter switch. `Dispatch::call` replaces every former `select_impl` call.
- [ ] **Step 4: Run** `CARGO_INCREMENTAL=0 cargo test` and clippy. Expected: all pass.
- [ ] **Step 5: Commit** `Introduce class-specific kernel entries and a per-interpreter dispatch table`.

---

### Task 2: Reductions, selections, and fused kernels on the new entries

**Files:**
* Modify: `src/kernels/scalar/reduce.rs`, `src/kernels/scalar/select.rs`, `src/kernels/scalar/fused.rs`, `src/kernels.rs` (remove the Task 1 allowance)
* Test: `src/kernels/tests.rs`, `src/kernels/dispatch/tests.rs`, `tests/fused_oracle.rs`

**Interfaces:**
* Consumes: Task 1 types.
* Produces: every Reduction-class kernel (`reduce_*`, `count`, `any`, `all`, `argmin_*`, `argmax_*`, `dot_*`, `sumsq_*`, `count_*_*`, `condsum_*_*`) as `KernelImpl::Reduction`, every Selection-class kernel (`compress_*`, `where`, `filter_*_*`) as `KernelImpl::Selection`. Construction kernels (`iota`, `fill_*`, `rand_*`, `load_*`, `length_*`, `same_len_*`) stay `Whole`. After this task `Registry::add` asserts the variant matches the class with no exception.
* Reducer state rules: float sums and products keep `CanonicalSum`/`CanonicalProduct` as their state, so tiles of multiples of 16 reproduce the one-shot bits. `argmin`/`argmax` state carries the running best index and value plus a `seen_nan` index, and the tile's `base` offset is the count of rows already consumed. `reduce_min`/`reduce_max` and `argmin`/`argmax` with no rows at `finish` return the existing empty-array errors.

- [ ] **Step 1: Write the failing tests.**
  * `tiled_reductions_match_whole`: for every Reduction-class kernel in the registry (iterate `Registry::iter`, build random arguments from `KernelDef.inputs` with 1000 rows), feeding the scalar reducer tiles of 64, 320, 608 rows and a final tile of 8 equals one update of all rows, by bits (NaN rule).
  * `tiled_selections_match_whole`: same for every Selection-class kernel with tiles `0..37`, `37..100`, `100..1000` and the matching `base`.
  * `canonical_sum_golden_values`: a fixed 35-element `f64` input (write the values literally in the test) whose canonical sum is a literal expected bit pattern, computed once by the current implementation, reviewed by hand against the spec's bullets, and pasted as `f64::from_bits(0x...)`. Same for a 35-element `f32` input whose `f32` accumulation differs from `f64` accumulation rounded to `f32`: assert both the golden value and that the `f64`-accumulated result differs.
  * `tests/fused_oracle.rs` keeps passing through `Dispatch::call`.
- [ ] **Step 2: Run** the tests. Expected: the new tests fail (entries are still `Whole`).
- [ ] **Step 3: Implement** the reducers and selection entries and remove the allowance.
- [ ] **Step 4: Run** the full suite and clippy. Expected: all pass.
- [ ] **Step 5: Commit** `Run reductions and selections as tile-capable entries`.

---

### Task 3: The autovectorization backend for AVX2 and AVX-512

**Files:**
* Create: `src/kernels/x86.rs`, `src/kernels/x86/auto.rs`, `tests/isa_variants.rs`
* Modify: `src/kernels.rs` (`Registry::new` calls `x86::register` under `#[cfg(target_arch = "x86_64")]`), `src/kernels/scalar/*.rs` only to expose the portable bodies as `#[inline(always)]` generic functions

**Interfaces:**
* Consumes: Tasks 1 and 2.
* Produces: for every Elementwise, Reduction, and Selection kernel with an array input, a `Variant { isa: Avx2, backend: Auto }` and a `Variant { isa: Avx512, backend: Auto }` entry in `impls`. Each entry is an `unsafe fn` with `#[target_feature]` (feature strings in Global constraints) whose body calls the same `#[inline(always)]` portable body the scalar entry uses, so the compiler vectorizes it for that ISA. `_s` kernels and Whole kernels get no variants.
* Mechanism (my call): a `macro_rules!` in `x86.rs` that, given a kernel's portable body path and type parameters, emits the two `#[target_feature]` functions and returns the `KernelImpl` pair. Reducers: the variant's `ReducerFn` returns a reducer type whose `update` is a `#[target_feature]` function called through `unsafe` with the SAFETY argument that the constructor is only reachable through `Dispatch` after detection.

- [ ] **Step 1: Write the failing test** `tests/isa_variants.rs`:
  * For each `isa` in `[Avx2, Avx512]` that `is_available()` on this machine (print `skipping <isa>: not available` otherwise) and each backend in `[Auto, Intrinsics]`, resolve a `Dispatch` for that pair and one for scalar.
  * For every kernel whose entry in that exact variant slot exists (so a kernel resolving to a fallback is not tested twice), generate arguments from `KernelDef.inputs`: lengths from `[0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 300]`, each array argument as a view at offset 0 and at a random offset 1 to 7 into a larger buffer. Floats mix finite values, `±0.0`, `±inf`, and NaN. Integers cover the full range. Bool rows are 0/1. Comparison scalars come from the same generator.
  * Assert `dispatch_isa.call(...)` equals `dispatch_scalar.call(...)`: arrays row by row, floats by bits except that a NaN in the scalar result only requires NaN, errors equal as strings.
  * Also assert that `Registry::new().resolve(isa, Backend::Auto).variant_used(id) == Variant { isa, backend: Auto }` for `add_f64`, `reduce_add_f64`, `dot_f64`, `compress_f64`, and that `Backend::Intrinsics` falls back to the auto entry for them while no intrinsics exist.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --test isa_variants`. Expected: FAIL (no variants installed, the `variant_used` assertion fails).
- [ ] **Step 3: Implement** `x86.rs` and `auto.rs`.
- [ ] **Step 4: Run** the full suite, clippy, and `CARGO_INCREMENTAL=0 cargo build --release`. Also run `cargo run --release -- run --isa avx512 --trace` on a 1M-element `sum(x * y)` program and check the trace line says `dot_f64 isa=avx512 backend=auto`. Expected: all pass. Note the release build time before and after in the report.
- [ ] **Step 5: Commit** `Add the autovectorization backend for AVX2 and AVX-512`.

---

### Task 4: Benchmarks against native Rust and the measurements document

**Files:**
* Create: `benches/tier1.rs` (with `[[bench]] name = "tier1" harness = false` in `Cargo.toml`), `docs/measurements/2026-10-30-x86-kernels.md`
* Modify: `Cargo.toml` (`cargo add --dev criterion@0.8`)

**Interfaces:**
* Consumes: `Interpreter::prepare`, `Prepared::execute`, `Options { isa, .. }`.
* Produces: criterion groups `tier1/<program>/<variant>` and `crossover/<program>/<n>`, and the measurements document.

Workloads, all at `n = 10_000_000` with data from `rand_f64(n, seed)` semantics (generate in the bench with the same SplitMix64 mapping, or through the interpreter once):
* `sum`: `input x: f64[]; export s = sum(x);`
* `dot`: `input x: f64[]; input y: f64[]; export s = sum(x * y);`
* `count`: `input x: f64[]; input c: f64; export n = count(x > c);` with `c = 0.5`. If scalar inputs are not supported by `input`, use a literal.
* `filter` at selectivity 1%, 50%, 99%: `export f = filter(x, |v| v > c);` with `c` = 0.99, 0.5, 0.01.

Variants per workload: interpreter with scalar, and for `avx2` and `avx512` (skip unavailable ones) both `--backend auto` and `--backend intrinsics`, and a native baseline per ISA. Until Task 6 the intrinsics backend falls back to auto, so Task 4 records it only where it differs, and Task 6 fills it in. A native baseline is a plain Rust loop in the bench file, compiled under the same `#[target_feature]` as the ISA it is compared with: `sum` and `dot` use the canonical 16-accumulator order so they compute the same bits (assert equality once), `count` is a counting loop, `filter` is a predicated-push loop into a `Vec` with capacity `n`. The interpreter measurement is `Prepared::execute` only, prepared once outside the timed loop.

Crossover sweep: `sum(x)` and `x * y` for `n` in powers of two from 16 to 2^20, interpreter (best ISA) against the native baseline. The crossover length is the smallest `n` at which the interpreter is within 1.5x of native.

The measurements document records: machine (CPU model from `/proc/cpuinfo`, kernel, rustc version), the exact command, a table per workload with median time, GB/s, and ratio to the native baseline of the same ISA for each backend, a pass/fail column against the spec's 1.5x target (sum, dot, count, filter on AVX2 and AVX-512), and the crossover table. Ratios are computed from criterion's median estimates.

- [ ] **Step 1: Write the benches.** Before timing, each workload asserts that every interpreter variant and the native baseline agree on the result (bits for floats, rows for filter).
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo bench --bench tier1` with no other heavy process running. Expected: completes, prints criterion estimates.
- [ ] **Step 3: Write** the measurements document from the run.
- [ ] **Step 4: Commit** `Benchmark tier 1 kernels against native Rust` with the bench, `Cargo.toml`, `Cargo.lock`, and the document.

---

### Task 5: The same workloads in corgi

**Files:**
* Create: `benches/corgi.rs` (`[[bench]] name = "corgi" harness = false`)
* Modify: `Cargo.toml` (dev-dependency `corgi = { git = "https://github.com/frankmcsherry/WIP", rev = "2751e04e8501d47b33219be23049bf4afa7a4428" }`, the rev DDIR pins), `docs/measurements/2026-10-30-x86-kernels.md`

**Interfaces:**
* Consumes: corgi's public API (`corgi::frontend::program::Program` or `compile_ml` plus `eval_graph`, read `corgi/NOTES.md` and `corgi/programs/*.col` in that repository for the surface syntax).

Write `sum`, `dot`, `count`, and `filter` (three selectivities) in corgi's ML surface over one row holding the whole column (or the form corgi's own benchmarks use for column work), on the same data. Time only evaluation, with the program compiled once. Assert each result equals Verkehrt's, allowing a different float sum in the last bits because corgi sums in row order (record the difference instead of failing). Add a corgi column to each table in the measurements document.

- [ ] **Step 1: Add the dependency and write the bench.** If corgi cannot express a workload, or the dependency does not build, write down what failed and continue with the rest. Report the gap as a concern instead of working around it.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo bench --bench corgi`.
- [ ] **Step 3: Update** the measurements document.
- [ ] **Step 4: Commit** `Benchmark tier 1 workloads in corgi`.

---

### Task 6: The intrinsics backend: compress and fused filter

**Files:**
* Create: `src/kernels/x86/compress.rs`
* Modify: `src/kernels/x86.rs` (install these entries in the `Backend::Intrinsics` slots, beside the auto entries), `docs/measurements/2026-10-30-x86-kernels.md`

**Interfaces:**
* Consumes: Task 3 registration, `SelectionFn`, `ArrayBuilder`.
* Produces: `Variant { isa: Avx512 | Avx2, backend: Intrinsics }` `SelectionFn` entries for `compress_*`, `where`, and `filter_*_*`. The auto entries stay.
  * AVX-512: for 32- and 64-bit element types, compare into a mask register (for `filter`) or load the bool mask bytes and turn them into a mask (for `compress`), then compress with `_mm512_maskz_compress_*` and store the selected lanes. `u8` and `bool` use the VBMI2 byte compress. `where` compresses an index vector.
  * AVX2: for 32- and 64-bit element types, a precomputed permutation table indexed by the lane mask (`_mm256_movemask_*`) with `_mm256_permutevar8x32_*`, storing the full vector and advancing by the popcount. `u8` and `bool` get no AVX2 intrinsics entry, so they fall back to auto, and the measurements document lists them as a strain case.
  * Writes may go to a stack buffer that is then appended with `ArrayBuilder::extend`, or straight into reserved builder capacity. Either way each `unsafe` block states its SAFETY argument.

- [ ] **Step 1: Run** `tests/isa_variants.rs` first and confirm it covers every selection kernel (it does through Task 3). It is the test for this task.
- [ ] **Step 2: Implement** the entries.
- [ ] **Step 3: Run** the full suite and clippy. Expected: all pass, including `tests/isa_variants.rs` on AVX2 and AVX-512.
- [ ] **Step 4: Re-run** `CARGO_INCREMENTAL=0 cargo bench --bench tier1` for the filter workloads and update the measurements document with auto against intrinsics numbers per ISA. An intrinsics entry that is not faster than its auto entry is listed as such, so the controller can rule on removing it. List every tier 1 workload still above 1.5x of native with the likely cause from the measurement (memory-bound, allocation, a missing instruction), so the controller can rule on it.
- [ ] **Step 5: Commit** `Add the intrinsics backend for compress and fused filter`.
