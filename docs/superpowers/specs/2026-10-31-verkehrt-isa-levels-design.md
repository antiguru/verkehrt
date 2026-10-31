# Verkehrt x86 instruction-set levels

This spec redefines the two x86 instruction sets as the psABI microarchitecture levels x86-64-v3 and x86-64-v4. It also adds per-entry CPU feature requirements, which `resolve` checks at startup. It extends `2026-10-30-verkehrt-design.md` and is tracked as issue #8. It comes before plan 5, because plan 5 (sort and hash) wants AVX-512DQ and possibly further extensions, so the dispatch model should be settled first.

## Why

Today `avx512` means the hand-picked set `avx512f,avx512bw,avx512vl,avx512vbmi2`, and `avx2` means `avx2` alone. AVX-512 is a family of extensions, not one instruction set, and that set has two defects. First, it lacks AVX-512DQ, so the `auto` entries cannot emit `vpmullq` or the 64-bit integer to float conversions. The `i64` to `f64`, `f64` to `i64`, and `f64` to `u64` casts are scalar loops on both x86 ISAs, which the assembly verdicts in `docs/measurements/2026-10-31-tier2/kernel-and-acceptance-results.txt` record. Second, it requires VBMI2, which only one kernel uses (byte `compress`, through `vpcompressb`), so a CPU without VBMI2, such as Skylake-X, falls back to AVX2 for every kernel.

The psABI levels give each name a fixed, documented meaning that compilers and distributions already use. Features beyond a level then belong to the entries that need them, not to the level, so a missing extension costs one kernel its intrinsics entry rather than costing every kernel its level.

## Goals

* `Isa::Avx2` becomes `Isa::X86_64V3` with the name `x86-64-v3`, and `Isa::Avx512` becomes `Isa::X86_64V4` with the name `x86-64-v4`. The old names are removed from `--isa`, `Isa::from_name`, and the bench IDs, with no aliases. Measurement documents written before this change keep the old names as history, and the base spec, the README, and new measurements use the new names.
* `auto` entries compile once per level, with the level's full feature string as `#[target_feature]`.
* An intrinsics entry may declare features beyond its level. `resolve` installs it only when the CPU has its level and every declared feature. Otherwise it falls back along the existing order: the level's `auto` entry, then scalar.
* `Options` gains a set of features to treat as absent, and the CLI gains `--disable <feature>` (repeatable). Tests use it to exercise the fallback paths on a CPU that has every feature.
* The DQ-dependent kernels are re-measured, and the catalog and the strain-case list are updated with the results.
* Results stay bit-identical across levels, backends, and disable sets, which the existing cross-variant property tests check.

## Non-goals

* No new kernels. A kernel that could use a newly available extension stays as is, unless the re-measurement shows it already vectorizes under v4.
* No change to aarch64 beyond what the shared types require. NEON stays one level with no declared extras, and SVE and optional NEON extensions are out of scope. The aarch64 baseline differs by target today: `rustc --print cfg` shows that `aarch64-apple-darwin` enables `neon` plus `dotprod`, `fp16`, `fhm`, `rdm`, `rcpc`, `lse`, `crc`, `aes`, `sha2`, `sha3`, and more at compile time, while `aarch64-unknown-linux-gnu` enables only `neon`. The scalar and NEON entries are therefore M1-level code on macOS and plain NEON on Linux, and the aarch64 measurements describe the macOS build. `Feature` is designed to take aarch64 features later, so a plan that wants CRC32, PMULL, or dotprod on Linux declares them per entry the same way.
* No x86-64-v2 level, and no level between v3 and v4. A CPU without AVX2 runs the scalar entries, as today.
* No fix for issue #7 (the `auto`-entry ISA guard), which stays separate.

## Anti-goals

* A disable set must never make an entry run on a CPU that lacks its features. The set only removes features from what detection reports, it never adds any, so it cannot make dispatch unsound.
* Detection must not depend on the build machine. Each level and feature is checked at run time with `is_x86_feature_detected!`, which also covers OS support for the wider registers.

## Level definitions

The feature strings follow the x86-64 psABI and use Rust's `target_feature` names.

| Level | `#[target_feature]` string |
|---|---|
| `x86-64-v3` | `avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave` plus the x86-64-v2 features `cmpxchg16b,lahfsahf,popcnt,sse3,sse4.1,sse4.2,ssse3` |
| `x86-64-v4` | the v3 string plus `avx512f,avx512bw,avx512cd,avx512dq,avx512vl` |

A level is available when the CPU reports every feature of its string and the disable set contains none of them. Disabling a level feature therefore drops the whole level, for example `--disable avx512dq` makes `Isa::detect` return `x86-64-v3`. The builder may drop features from the strings that Rust cannot name or that `is_x86_feature_detected!` cannot check, and records each dropped one in the spec's level table.

Enabling `fma` is safe for bit identity. Rust never contracts a separate multiply and add into a fused one without an explicit `mul_add`, and the `auto` bodies contain none. The existing property tests confirm it.

## Features

A `Feature` enum names every feature that a level contains or an entry may declare, with `Feature::name` returning the `target_feature` name (`"avx512vbmi2"`). A `FeatureSet` is a `Copy` bit set of features, so `Options` stays `Copy`. `FeatureSet::detect()` reads every feature once at startup.

`Feature` is the same on every target, so `Options` and `--disable` behave the same everywhere (my call). On aarch64 no x86 feature is ever detected, so disabling one changes nothing, just as disabling a feature that an x86 CPU lacks changes nothing. The CLI prints a note on stderr for each disabled feature that detection did not report, so a disable without effect cannot pass for a fallback test.

Today exactly one intrinsics entry declares an extra feature. The AVX-512 byte `compress` entry, which serves `compress` on `u8` and bool, declares `avx512vbmi2`. Its `#[target_feature]` string is the v4 string plus `avx512vbmi2`. Every other intrinsics entry (the 32-bit and 64-bit `compress`, `where`, the `filter_<cmp>` entries, `gather`, `scan_by_add`) fits in its level and declares nothing. The builder confirms this by checking each intrinsic's required features in the Intel intrinsics guide or `core::arch` documentation, and lists any other entry that needs an extra.

## Registration and resolve

Each slot of a kernel's variant table holds the entry and the features it declares beyond its level, empty for every `auto` and scalar entry. `Registry::resolve` takes the disable set in addition to the ISA and the backend, and computes the available features itself as what `FeatureSet::detect` reports minus the disable set. No caller can pass features the CPU lacks, so `resolve` stays a safe function and remains the soundness gate for `Dispatch` (my call). The selection rule is a pure function from a variant table, the ISA, the backend, and an available set to the chosen variant. It installs nothing, so tests may call it with synthetic sets. For each kernel it picks, in order:

1. the intrinsics entry of the level, when the backend is `Intrinsics`, the level is available, and every declared feature is available
2. the `auto` entry of the level, when the level is available
3. the scalar entry

A requested level that is unavailable resolves every kernel to scalar in `Registry::resolve`, as today. `Interpreter::new` keeps rejecting a forced level that is unavailable with a usage error, and the disable set counts: `--isa x86-64-v4 --disable avx512dq` fails with "instruction set 'x86-64-v4' is not available", which keeps a mistyped disable visible instead of silently running scalar. Without `--isa`, `Isa::detect` applies the disable set and picks the best remaining level. The variant that `Dispatch` records, and that `--trace` and `--stats` print, is the entry that was installed, so a fallback caused by a missing feature shows as the level's `auto` variant.

One predicate decides whether an entry may run: `runnable(isa, declared, available)`, true when every feature of the level of `isa` and every feature in `declared` is in `available`. The selection function calls it for each candidate, and `Isa::is_available` is the same predicate with an empty `declared` set. Every `SAFETY` comment in `dispatch.rs`, `auto.rs`, and the x86 intrinsics points at this predicate instead of restating the check. `Dispatch` keeps calling only entries that `resolve` installed.

## Options and CLI

`Options` gains `disable: FeatureSet`, empty by default. The CLI gains `--disable <feature>`, repeatable, which accepts the names `Feature::name` returns and rejects others with an error that lists the valid names. `Isa::detect` and `Interpreter` both apply the disable set, so `--disable avx512vbmi2` alone keeps `x86-64-v4` and moves only byte `compress` to its `auto` entry.

## Measurement

The re-measurement runs on the benchmark host (AMD Ryzen AI MAX+ 395, Zen 5, boost off) with the same methodology as `docs/measurements/2026-10-31-tier2.md`. It covers:

* the `i64` to `f64`, `f64` to `i64`, `f64` to `u64`, and `u64` to `f64` casts and `mul_i64`, comparing the scalar, `x86-64-v3`, and `x86-64-v4` entries, with an assembly verdict per kernel (vector loop or scalar loop)
* byte `compress` with and without `--disable avx512vbmi2`, to show the fallback cost
* the tier 1 and tier 2 bench suites and the acceptance queries at both levels, to catch kernels whose code generation changed under the wider feature strings

The baseline is the merge of #6 (`6033fe9`), built in a separate worktree and run in the same session, interleaved with the new build, so both see the same host state. Its `avx2` and `avx512` entries are compared with the new `x86-64-v3` and `x86-64-v4` entries.

A kernel or acceptance query that is slower at the new level than at the old ISA by more than the run-to-run spread is recorded in the measurement document with its assembly verdict and filed as an issue. It does not block the change, and no kernel gets a narrower feature string than its level (my call). Correctness and a well-defined level take precedence over a per-kernel speed difference, and each regression stays visible as an issue.

The results go into a new `docs/measurements/2026-10-31-isa-levels.md` with the raw output as evidence beside it. The base spec's catalog rows and strain-case list change only where a measurement supports the change, for example a cast leaves the strain-case list only when its assembly shows a vector loop and it beats scalar.

## Testing

* The cross-variant property tests in `tests/isa_variants.rs` and `src/kernels/tests.rs` iterate a fixed list of configurations and compare every entry to scalar bit for bit: `x86-64-v3`, `x86-64-v4`, and `x86-64-v4` with `avx512vbmi2` disabled (my call). A configuration the CPU cannot run is skipped with a message.
* A pure test checks that removing a level feature from a synthetic available set drops the level: with `avx512dq` removed, the best available level is `x86-64-v3`, and with `avx2` removed, it is scalar.
* A test of the pure selection function checks the fallback order with synthetic feature sets: an entry whose declared feature is absent resolves to the level's `auto` entry, and an entry whose level is absent resolves to scalar. It installs no entry, so it runs on every CPU.
* A test runs byte `compress` with `avx512vbmi2` disabled and checks that it resolves to the `auto` variant and matches scalar. It skips with a message on a CPU without v4, like the existing ISA tests.
* A CLI test checks that `--disable` rejects an unknown name, that `--isa avx512` is now an error, and that `--isa x86-64-v4 --disable avx512dq` fails with the unavailable error on a CPU with v4.

## Left to the builder

* The exact names and layout of `Feature`, `FeatureSet`, and the table slot type.
* How the level feature strings are shared between `#[target_feature]` attributes, which need string literals, and `FeatureSet`, as long as one definition is the source of truth and a test checks that they agree.
* Whether `Isa::is_available` takes the disable set as an argument or a new function does.
