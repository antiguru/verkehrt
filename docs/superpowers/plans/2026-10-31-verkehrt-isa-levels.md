# x86 instruction-set levels implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redefine the x86 instruction sets as the psABI levels x86-64-v3 and x86-64-v4, let intrinsics entries declare features beyond their level, and add a disable set for testing fallbacks.

**Architecture:** A new `src/kernels/features.rs` owns `Feature`, `FeatureSet`, the level feature sets, and the one predicate `runnable`. Each kernel's variant table becomes a `KernelTable` whose slots carry the entry and its declared features, and a pure `select` function picks the variant that `Registry::resolve` installs. `Options.disable` and `--disable` remove features from what detection reports.

**Tech Stack:** Rust 2024, `std::arch::is_x86_feature_detected!`, `#[target_feature]`, proptest, assert_cmd, criterion.

**Spec:** `docs/superpowers/specs/2026-10-31-verkehrt-isa-levels-design.md` (extends `docs/superpowers/specs/2026-10-30-verkehrt-design.md`).

## Global Constraints

* Level strings, verbatim from the spec. v3: `avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave` plus `cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3`. `lahfsahf` is dropped, because rustc 1.99 rejects it as unstable, and the spec's level table records the drop. v4: the v3 string plus `avx512f,avx512bw,avx512cd,avx512dq,avx512vl`. A feature that Rust cannot name in `#[target_feature]` or that `is_x86_feature_detected!` cannot check is dropped and recorded in the spec's level table.
* Names: `Isa::X86_64V3` named `x86-64-v3`, `Isa::X86_64V4` named `x86-64-v4`. No aliases for `avx2` and `avx512`.
* The only declared extra today: the AVX-512 byte `compress` entry (`compress_u8`, `compress_bool`) declares `avx512vbmi2`.
* Results stay bit-identical to the scalar entries for every level, backend, and disable set.
* No unsafe beyond what exists. Every `SAFETY` comment that justifies calling an ISA entry cites `features::runnable`.
* `#[expect(lint)]`, never `#[allow(lint)]`, for any suppression added or touched.
* Large test modules go in `<module>/tests.rs` with a plain `#[cfg(test)] mod tests;`, never `#[path]`.
* Comments: no em-dashes, no structuring semicolons. Doc comments state the contract.
* Tasks 1 to 6 run on the x86-64 benchmark host (Zen 5, with AVX-512 and VBMI2), so every v4 path runs there. The selection and level tests use synthetic feature sets and carry no `cfg` gate, so they also run on aarch64 and on CI runners without AVX-512.
* Build with `CARGO_INCREMENTAL=0`. The full check is `cargo test --release` plus `cargo clippy --all-targets -- -D warnings`.
* Markdown: asterisk lists, evidence-based claims, headers capitalized only on the first word.

## Review Focus

* A forced level that the disable set drops must fail with the usage error, never run scalar silently. Task 4 tests it.
* An entry whose declared feature is disabled must never be installed, also through `Backend::Auto` and on a CPU without that feature. Task 3's `select` tests cover it with synthetic sets.
* A `#[target_feature]` string in the x86 sources that differs from its level, or names an undeclared extra, would make dispatch unsound without any test failing on the host. Task 3's source-scan test pins it.
* `--disable` with a name valid on the other architecture: accepted everywhere, and the stderr note fires. Task 4 tests it.
* `Isa::detect` on a CPU with v4 but without VBMI2 must return v4 (before this change it returned AVX2). Task 3's `best_level` test covers it with a synthetic set.

---

### Task 1: Rename the x86 ISAs

Pure rename, no change in meaning. `Isa::Avx2` becomes `Isa::X86_64V3`, `Isa::Avx512` becomes `Isa::X86_64V4`, and their names become `x86-64-v3` and `x86-64-v4`. The feature strings and `is_available` stay as they are until Task 3.

**Files:**
* Modify: `src/ops.rs` (`Isa`), `src/kernels.rs` (`Variant::index` doc and match), `src/kernels/x86.rs` and `src/kernels/x86/**` (constant names such as `AVX2_AUTO` become `V3_AUTO`, module names in `auto_backend!` invocations become `x86_64_v3` and `x86_64_v4`, doc text)
* Modify: `src/kernels/tests.rs`, `src/kernels/dispatch/tests.rs`, `tests/isa_variants.rs`, `tests/programs.rs`, `tests/cli.rs`, `benches/tier1.rs`, `benches/acceptance.rs`

**Interfaces:**
* Produces: `Isa::{Scalar, X86_64V3, X86_64V4, Neon}`, `Isa::name()` returning `"scalar"`, `"x86-64-v3"`, `"x86-64-v4"`, `"neon"`. Bench IDs use the new names.

- [ ] **Step 1: Add the failing CLI test** `old_isa_names_are_rejected` in `tests/cli.rs`: `run --isa avx512 x.vk` and `run --isa avx2 x.vk` each fail with stderr containing `unknown instruction set 'avx512'` (resp. `'avx2'`). Add `new_isa_names_parse`: `Isa::from_name("x86-64-v3") == Some(Isa::X86_64V3)` and likewise for v4, in `src/kernels/tests.rs`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --release --test cli old_isa_names` and expect FAIL.
- [ ] **Step 3: Rename** across the files above. Use `grep -rn "Avx2\|Avx512\|\"avx2\"\|\"avx512\"\|AVX2_\|AVX512_" src tests benches` to find every site, and update comments that name the ISAs. Leave `docs/measurements/` untouched.
- [ ] **Step 4: Run** `CARGO_INCREMENTAL=0 cargo test --release` and `cargo clippy --all-targets -- -D warnings`, expect PASS.
- [ ] **Step 5: Commit** "Rename the x86 instruction sets to x86-64-v3 and x86-64-v4".

### Task 2: Features module

Adds the feature vocabulary and the level predicate without wiring them into dispatch.

**Files:**
* Create: `src/kernels/features.rs`, `src/kernels/features/tests.rs`
* Modify: `src/kernels.rs` (declare `pub mod features;`, re-export `Feature`, `FeatureSet`)

**Interfaces:**
* Produces, in `crate::kernels::features`:
  * `pub enum Feature` with one variant per feature of the v4 string, plus `Avx512vbmi2` and `Neon`. `Feature::ALL: [Feature; N]`, `fn name(self) -> &'static str` (the `target_feature` name, for example `"sse4.1"`, `"avx512vbmi2"`), `fn from_name(&str) -> Option<Feature>`.
  * `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)] pub struct FeatureSet(u64)` with `pub const EMPTY`, `pub const fn of(features: &[Feature]) -> FeatureSet`, `fn contains(self, Feature) -> bool`, `fn contains_all(self, other: FeatureSet) -> bool`, `fn union(self, other) -> FeatureSet`, `fn minus(self, other) -> FeatureSet`, `fn iter(self) -> impl Iterator<Item = Feature>`, `#[cfg(test)] fn parse(s: &str) -> Option<FeatureSet>` (a comma-separated `target_feature` string, used only by tests), and `fn detect() -> FeatureSet`, which on x86-64 checks every x86 `Feature` with `is_x86_feature_detected!`, on aarch64 reports `Neon` (`is_aarch64_feature_detected!("neon")`), and elsewhere returns `EMPTY`.
  * `pub fn level(isa: Isa) -> FeatureSet`: `EMPTY` for `Scalar`, `{neon}` for `Neon`, the v3 set for `X86_64V3`, the v4 set for `X86_64V4`.
  * `pub fn runnable(isa: Isa, declared: FeatureSet, available: FeatureSet) -> bool`: true when `available` contains `level(isa).union(declared)`. It has no `cfg!` check: detection carries the architecture, because on aarch64 `detect` never reports an x86 feature and on x86-64 it never reports `Neon`. `Scalar` is therefore always runnable.
  * `pub fn best_level(available: FeatureSet) -> Isa`: the first of `X86_64V4`, `X86_64V3`, `Neon` that is `runnable` with no declared features, else `Scalar`.

- [ ] **Step 1: Write the failing tests** in `src/kernels/features/tests.rs`:
  * `names_round_trip`: `Feature::from_name(f.name()) == Some(f)` for every `f` in `Feature::ALL`, and `Feature::from_name("avx512") == None`.
  * `v4_contains_v3`: `level(X86_64V4).contains_all(level(X86_64V3))`, and `level(X86_64V4).minus(level(X86_64V3))` is exactly `{avx512f, avx512bw, avx512cd, avx512dq, avx512vl}`.
  * `dropping_a_level_feature_drops_the_level` (no `cfg` gate): with `all = FeatureSet::of(&Feature::ALL)`, `best_level(all) == X86_64V4`, `best_level(all.minus(of(&[Avx512dq]))) == X86_64V3`, `best_level(all.minus(of(&[Avx2, Neon]))) == Scalar`, and `best_level(of(&[Neon])) == Neon`.
  * `v4_without_vbmi2_is_v4`: `best_level(all.minus(of(&[Avx512vbmi2]))) == X86_64V4`.
  * `declared_features_gate_runnable`: `runnable(X86_64V4, of(&[Avx512vbmi2]), all)` is true, and false on `all.minus(of(&[Avx512vbmi2]))`. `runnable(Scalar, EMPTY, EMPTY)` is true, and `runnable(X86_64V3, EMPTY, of(&[Neon]))` is false.
  * `detect_is_architecture_specific`: on x86-64, `FeatureSet::detect()` does not contain `Neon`, and on aarch64 it contains `Neon` and no x86 feature.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --release features` and expect a compile failure.
- [ ] **Step 3: Implement** the interfaces above. The level sets are `const` `FeatureSet`s built with `FeatureSet::of`. Nothing outside the tests uses the module yet, so mark the unused items with `#[cfg_attr(not(test), expect(dead_code))]`, which Task 3 removes. A plain `expect` would be unfulfilled in the test build, where the tests use them.
- [ ] **Step 4: Run** the tests, `cargo test --release`, and clippy. Expect PASS.
- [ ] **Step 5: Commit** "Add the CPU feature vocabulary and level predicate".

### Task 3: Levels, declared features, and selection

This is the soundness-critical step. The `#[target_feature]` strings, `Isa::is_available`, and dispatch selection switch to the levels and declared features in one commit, so no intermediate state installs an entry on a CPU without its features.

**Files:**
* Create: `src/kernels/x86/compress/avx512_vbmi2.rs`
* Modify: `src/kernels.rs` (`KernelTable`, `KernelDef::impls`, `select`, `Registry::resolve`, `Registry::add`), `src/kernels/arch.rs`, `src/kernels/auto.rs`, `src/kernels/x86.rs`, `src/kernels/x86/compress.rs`, `src/kernels/x86/compress/avx2.rs`, `src/kernels/x86/compress/avx512.rs`, `src/kernels/x86/gather.rs`, `src/kernels/x86/scan.rs`, `src/kernels/aarch64.rs`, `src/kernels/scalar/*.rs` (the table type), `src/kernels/dispatch.rs` (`SAFETY` comments), `src/ops.rs` (`Isa::is_available`, `Isa::detect`), every `Registry::resolve` caller in `src`, `tests`, and `benches`
* Test: `src/kernels/tests.rs` (or `src/kernels/features/tests.rs` for the source scan)

**Interfaces:**
* Consumes: everything Task 2 produces.
* Produces:
  * `pub struct KernelTable` in `src/kernels.rs`, replacing `[Option<KernelImpl>; Variant::COUNT]` everywhere. Each slot holds `Option<(KernelImpl, FeatureSet)>`. Methods: `pub fn scalar_only(imp: KernelImpl) -> KernelTable` (also used by `KernelImpl::scalar_only`, which returns a `KernelTable`), `pub fn get(&self, v: Variant) -> Option<KernelImpl>`, `pub fn declared(&self, v: Variant) -> FeatureSet`, `pub(crate) fn set(&mut self, v: Variant, imp: KernelImpl)`, `pub(crate) fn set_declared(&mut self, v: Variant, imp: KernelImpl, declared: FeatureSet)`, `pub fn entries(&self) -> impl Iterator<Item = KernelImpl>`.
  * `KernelDef::impls: KernelTable`. The `arch::add_*` functions take `&mut KernelTable`.
  * `pub fn select(table: &KernelTable, isa: Isa, backend: Backend, available: FeatureSet) -> Variant` in `src/kernels.rs`. It is pure, and its rule is the spec's: the intrinsics slot when `backend` is `Intrinsics`, the slot is set, and `runnable(isa, declared, available)` holds. Otherwise the auto slot when it is set and `runnable(isa, EMPTY, available)` holds. Otherwise `Variant::SCALAR`.
  * `pub fn resolve(&self, isa: Isa, backend: Backend, disable: FeatureSet) -> Dispatch`, which computes `FeatureSet::detect().minus(disable)` once and calls `select` per kernel.
  * `Isa::is_available(self) -> bool` is `runnable(self, EMPTY, FeatureSet::detect())`. `Isa::detect()` is `best_level(FeatureSet::detect())`.

- [ ] **Step 1: Write the failing tests**:
  * `select_falls_back_by_declared_feature` (pure, runs on every CPU): build a `KernelTable` with a scalar entry, an auto entry at `X86_64V4`, and an intrinsics entry at `X86_64V4` declaring `avx512vbmi2`, using the existing scalar test bodies as stand-in fn pointers. With `all = FeatureSet::of(&Feature::ALL)`: `select(.., X86_64V4, Intrinsics, all)` is the intrinsics variant, `select(.., X86_64V4, Intrinsics, all.minus(vbmi2))` is the `X86_64V4` auto variant, `select(.., X86_64V4, Auto, all)` is the auto variant, `select(.., X86_64V4, Intrinsics, all.minus(avx512dq))` is `Variant::SCALAR`, and `select(.., X86_64V4, Intrinsics, FeatureSet::EMPTY)` is `Variant::SCALAR`.
  * `detected_level_is_available`: `Isa::detect().is_available()` holds, and `Isa::detect() == best_level(FeatureSet::detect())`.
  * `byte_compress_declares_vbmi2` (x86-64 only, because only x86-64 builds the entries): in `Registry::new()`, `compress_u8` and `compress_bool` have `declared(V4 intrinsics) == of(&[Avx512vbmi2])` and a set `X86_64V4` auto slot (so the fallback lands on auto, not scalar), and every other slot of every kernel declares `EMPTY`.
  * `target_feature_strings_match_levels` (no `cfg` gate): walk every `.rs` file under `src/` from `env!("CARGO_MANIFEST_DIR")` at test time, extract each string of `target_feature(enable = "...")` and the feature-string argument of every invocation of a macro that forwards one to `#[target_feature]` (`auto_backend!` and `selection_loops!`, skipping the `macro_rules!` definitions, whose argument is `$features`), and parse it with `FeatureSet::parse`. Each must equal `level(X86_64V3)`, `level(X86_64V4)`, `level(X86_64V4).union(of(&[Avx512vbmi2]))` (only in `src/kernels/x86/compress/avx512_vbmi2.rs`), or `level(Neon)` (only under `src/kernels/aarch64`). Any other string fails, including one in a file added later. The test skips its own source file, which contains the pattern in its extractor. The VBMI2 string may appear only in `src/kernels/x86/compress/avx512_vbmi2.rs`. Assert at least one string matched each of the four, so a broken extractor fails.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --release select_falls_back byte_compress target_feature_strings` and expect a compile failure.
- [ ] **Step 3: Introduce `KernelTable`** and port every table site (`grep -rn "Option<KernelImpl>; Variant::COUNT\|impls\[" src tests benches`). Port `Registry::add`'s checks to `KernelTable`.
- [ ] **Step 4: Split the AVX-512 selection loops by feature string.** Today `selection_loops!("AVX-512", "avx512f,avx512bw,avx512vl,avx512vbmi2")` at `src/kernels/x86/compress/avx512.rs:300` compiles `compress`, `where`, and `filter` for every row type under one string. Invoke the macro twice. One instantiation uses the v4 string and serves the 32-bit and 64-bit `compress` entries, `where`, and every `filter` entry. The other uses the v4 string plus `avx512vbmi2` and serves only `compress_u8` and `compress_bool`. Move the `u8` `Lanes` impl, whose `compress` step issues `vpcompressb`, and the VBMI2 instantiation into `src/kernels/x86/compress/avx512_vbmi2.rs`, so that no v4-only entry can reach a VBMI2-compiled function. `compress_entries` picks the module by row type. The invariant: every function that can execute a VBMI2 instruction is compiled with v4 plus `avx512vbmi2` and is reachable only from an entry that declares `avx512vbmi2`, and every other AVX-512 function is compiled with the v4 string alone. Check in the release disassembly that `vpcompressb` stays inlined in the byte `compress_chunk` loop.
- [ ] **Step 5: Switch the feature strings.** `auto_backend!` invocations, the `selection_loops!` invocations, and every intrinsics `#[target_feature]` use the full level string from the Global Constraints, and the byte `compress` code uses the v4 string plus `avx512vbmi2`. Register the byte `compress` entries with `set_declared(.., FeatureSet::of(&[Feature::Avx512vbmi2]))`.
- [ ] **Step 6: Implement `select`, the new `resolve`, `is_available`, and `detect`**, and pass `FeatureSet::EMPTY` at every existing `resolve` call site. Remove the `cfg_attr(not(test), expect(dead_code))` attributes from Task 2, the `#[allow(unreachable_patterns)]` in the old `is_available`, and change the `allow(clippy::...)` attributes on `add_extreme` and `add_arg_extreme` in `src/kernels/arch.rs` to `expect`.
- [ ] **Step 7: Rewrite the `SAFETY` comments** in `src/kernels/dispatch.rs`, `src/kernels/auto.rs`, and `src/kernels/x86/**` so that each cites `features::runnable` as the check that `resolve` applied through `select`, instead of restating "`is_available` returned true".
- [ ] **Step 8: Run** `CARGO_INCREMENTAL=0 cargo test --release` and clippy. `tests/isa_variants.rs` must pass at both levels on the host. Expect PASS.
- [ ] **Step 9: Commit** "Define the x86 levels as psABI v3 and v4 with per-entry features".

### Task 4: Disable set in Options and on the CLI

**Files:**
* Modify: `src/interp.rs` (`Options`, `Interpreter::new`), `src/ops.rs` (`Isa::is_available_with`, `Isa::detect_with`), `src/main.rs` (`--disable`, stderr note), `README.md` (the CLI usage, if it lists flags)
* Test: `tests/cli.rs`, `tests/programs.rs`

**Interfaces:**
* Consumes: `FeatureSet`, `Feature::from_name`, `Registry::resolve(isa, backend, disable)` from Tasks 2 and 3.
* Produces: `Options::disable: FeatureSet` (default empty, so `Options` stays `Copy` and `Default`). `Isa::is_available_with(self, disable: FeatureSet) -> bool` is `runnable(self, EMPTY, FeatureSet::detect().minus(disable))`, and `Isa::is_available` calls it with `EMPTY`. `Isa::detect_with(disable: FeatureSet) -> Isa`, likewise.

- [ ] **Step 1: Write the failing tests**:
  * `tests/cli.rs` `unknown_feature_is_usage_error`: `run --disable avx513 x.vk` fails, and stderr contains `unknown feature 'avx513'` and `avx512vbmi2` (the list of valid names).
  * `tests/cli.rs` `disabling_a_level_feature_rejects_the_forced_level`: when `Isa::X86_64V4.is_available()`, `run --isa x86-64-v4 --disable avx512dq` on a valid program fails with stderr containing `instruction set 'x86-64-v4' is not available`. Otherwise print a skip message.
  * `tests/cli.rs` `undetected_disable_prints_a_note`: pick a feature from `Feature::ALL` that `FeatureSet::detect()` does not report (on aarch64 that is any of them), run a valid program with `--disable <it>`, and check stderr contains `note: feature '<it>' is not detected on this CPU, disabling it has no effect` and the program still succeeds. Skip with a message when the CPU reports every feature.
  * `tests/programs.rs` `disable_vbmi2_keeps_v4`: when `Isa::X86_64V4.is_available()`, an `Interpreter` with `Options { disable: of(&[Avx512vbmi2]), trace: true, ..Default::default() }` running a program that calls `compress` on a `u8` array prints the same output as the scalar interpreter, and its trace line for `compress_u8` contains `isa=x86-64-v4 backend=auto`.
- [ ] **Step 2: Run** them and expect FAIL.
- [ ] **Step 3: Implement.** `Interpreter::new` rejects a forced `isa` for which `is_available_with(opts.disable)` is false, with the existing usage error text, and otherwise uses `Isa::detect_with(opts.disable)` and passes `opts.disable` to `resolve`. `--disable` is a repeatable `clap` argument parsed with `Feature::from_name`. Its error is `unknown feature '<name>', expected one of: <names joined by ", ">`. `main` prints the note for each disabled feature missing from `FeatureSet::detect()` before running.
- [ ] **Step 4: Run** `cargo test --release` and clippy. Expect PASS.
- [ ] **Step 5: Commit** "Add a feature disable set to Options and the CLI".

### Task 5: Cross-variant tests over disable configurations

**Files:**
* Modify: `tests/isa_variants.rs`, `src/kernels/tests.rs` (where it iterates ISAs with `is_available`)

**Interfaces:**
* Consumes: `Registry::resolve(isa, backend, disable)`, `KernelTable::declared`, `Isa::is_available_with`, `features::runnable`.

- [ ] **Step 1: Replace `available_isas`** with a fixed list of configurations `(Isa, FeatureSet)`: `(X86_64V3, EMPTY)`, `(X86_64V4, EMPTY)`, `(X86_64V4, of(&[Avx512vbmi2]))`, `(Neon, EMPTY)`. A configuration with `!isa.is_available_with(disable)` is skipped with `println!("skipping {}: not available", ..)`, and the disable set is printed with the configuration name.
- [ ] **Step 2: Change `check_variant`** to take the disable set and pass it to `resolve`. A kernel whose slot is set but whose declared features are not all available (`!runnable(isa, declared, FeatureSet::detect().minus(disable))`) is not property-tested in that slot. Instead the test asserts that `dispatch.variant_used(id)` is the same level's auto variant, and counts it in a separate "fell back" total that is printed. Under `(X86_64V4, {avx512vbmi2})` on a CPU with VBMI2, assert that the fell-back total is 2 (`compress_u8`, `compress_bool`).
- [ ] **Step 3: Update `has_auto_entries`** and `every_array_kernel_has_auto_variants` to the new names. Run `CARGO_INCREMENTAL=0 cargo test --release --test isa_variants -- --nocapture` and check the printed counts for all three x86 configurations.
- [ ] **Step 4: Run** `cargo test --release` and clippy. Expect PASS.
- [ ] **Step 5: Commit** "Test every kernel variant under the disable configurations".

### Task 6: Measurement and documentation

Run on the benchmark host (AMD Ryzen AI MAX+ 395, Zen 5, boost off). The baseline is `6033fe9`, built in a separate worktree at `.claude/worktrees/isa-baseline` with `git worktree add --detach .claude/worktrees/isa-baseline 6033fe9`, and run interleaved with the new build in the same session. Remove that worktree afterwards with `git worktree remove`.

**Files:**
* Create: `docs/measurements/2026-10-31-isa-levels.md`, `docs/measurements/2026-10-31-isa-levels/` (raw output), `docs/measurements/2026-10-31-isa-levels/casts.vk` (the measurement program)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (ISA names in current-state text, the catalog rows and the strain-case list where measurements support a change), `docs/superpowers/specs/2026-10-31-verkehrt-isa-levels-design.md` (the level table, only if Task 3 dropped a feature), `README.md` (ISA names)

- [ ] **Step 1: Casts and `mul_i64`.** Write `casts.vk` with `let n = 10000000;`, `let x = rand_i64(n, 1);`, `let u = cast_u64(x);`, and `let f = (rand_f64(n, 2) - 0.5) * 1.0e12;` and `let g = rand_f64(n, 3) * 1.0e12;`, so the `f64` to `i64` inputs span both signs and the `f64` to `u64` inputs are non-negative, all within range. Adjust the syntax to what the checker accepts and record the final program. Then `cast_f64` of `i64` and of `u64`, `cast_i64` and `cast_u64` of `f64`, and `mul` on `i64`, each consumed by a `sum` so nothing is dead. Run it with `--stats` five times per configuration: baseline `--isa scalar`, `--isa avx2`, `--isa avx512`, new `--isa scalar`, `--isa x86-64-v3`, `--isa x86-64-v4`, interleaving baseline and new runs. Keep the raw output.
- [ ] **Step 2: Assembly verdicts.** For each of those five kernels, disassemble the `x86-64-v4` auto entry in the new release build (`objdump -d --no-show-raw-insn` on the bench or test binary, locating the symbol by name) and record "vector loop" or "scalar loop" with the key instructions (for example `vcvtqq2pd`, `vcvttpd2qq`, `vpmullq`).
- [ ] **Step 3: Byte `compress` fallback.** Time `compress_u8` at `x86-64-v4` with and without `--disable avx512vbmi2` the same way.
- [ ] **Step 4: Suites.** Run `cargo bench --bench tier1` and `cargo bench --bench acceptance` on both builds, interleaved, and keep the raw output. Compare the baseline `avx2` and `avx512` IDs with the new `x86-64-v3` and `x86-64-v4` IDs.
- [ ] **Step 5: Write `docs/measurements/2026-10-31-isa-levels.md`**: method, host, the tables from Steps 1 to 4, and the assembly verdicts. A kernel or query slower at the new level than at the old ISA by more than its run-to-run spread is listed under a "Regressions" heading with its numbers and verdict. Do not file issues, the controller files them from that list.
- [ ] **Step 6: Update the base spec.** Replace `avx2` and `avx512` in current-state text with the new names. Change a catalog row or the strain-case list only where Steps 1 to 4 support it: a cast leaves the strain-case list only if its verdict is "vector loop" and it beats scalar. Update `README.md`'s ISA names.
- [ ] **Step 7: Commit** "Measure the x86 levels" with the documents and raw output.
