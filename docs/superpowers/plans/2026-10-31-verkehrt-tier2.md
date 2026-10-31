# Verkehrt tier 2 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `u64` kernels and the tier 2 primitives (`prev`, `scan`, `scan_by`, `gather`, `slice`, `histogram`, `reduce_into`, `div_const`, `argsort`) with scalar and `auto` entries, plus an acceptance harness that runs TPC-H Q1 and Q6 and ClickBench q01, q02, q06, q07, and q42 against DuckDB's answers and measures them against corgi and DuckDB.

**Architecture:** Two kernel-table extensions come first: a `Scan` class whose accumulator writes output rows and carries state across tiles, and ISA variants for `Whole` kernels through a `WholeBody` trait and the shared `auto_backend!` macro. The new primitives then land in groups, each end to end through the builtin table, the type checker, the registry, and the property tests. The acceptance harness binds corgi's column files to `input` declarations, and the measurement work compares the queries with corgi's own benches and DuckDB's recorded times.

**Tech stack:** Rust 1.95+, `egglog`, `criterion`, `proptest`. Data preparation uses `uv`, DuckDB through Python, and corgi's `prepare.py` scripts.

**Spec:** `docs/superpowers/specs/2026-10-31-verkehrt-tier2-design.md`, which extends `docs/superpowers/specs/2026-10-30-verkehrt-design.md`. Executors read both.

**Plan series:** Plan 4. Plans 1 to 3 (interpreter, x86 kernels, NEON kernels) are merged. Plan 5 adds grouping on keys with many distinct values.

## Decisions taken before this plan

* The builder prepares the acceptance data itself, in `~/dev/repos/data/verkehrt/` (user decision). Nothing from that directory is committed.
* Results stay bit-identical across ISAs and backends (user discussion, recorded in the spec's semantics section).
* Float `add` and `mul` in `scan`, `scan_by`, and `reduce_into`, `add` and `mul` on `u8` there, and float `argsort` are rejected for now and recorded as surface gaps (spec, "Deferred: surface gaps").
* The operator of `scan`, `scan_by`, and `reduce_into` is always the last argument and is an operation name, as in `reduce(x, max)`.
* Kernel names follow the existing scheme: `prev_<t>`, `scan_<op>_<t>`, `scan_by_<op>_<t>`, `gather_<t>`, `slice_<t>`, `histogram`, `reduce_into_<op>_<t>`, `div_const_<t>`, and `argsort_<t>`, with `<op>` the lowercase operation name and `<t>` the element type name.

## Global constraints

* Edition 2024, `rust-version = "1.95"`. CI runs `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and tests on ubuntu and macOS (Apple Silicon) with stable and 1.95. Build with `CARGO_INCREMENTAL=0`.
* Modules use `name.rs` plus a `name/` directory, never `name/mod.rs`. Large test modules go in `name/tests.rs`, declared with a plain `#[cfg(test)] mod tests;`.
* `unsafe` appears only in `src/kernels/dispatch.rs`, `src/kernels/auto.rs`, `src/kernels/select_loops.rs`, and the `x86` and `aarch64` kernel modules. Every `unsafe` block or `unsafe fn` carries a `// SAFETY:` comment naming the CPU features and pointer or length assumptions it relies on.
* The library has no global mutable state, an interpreter instance stays `Send`.
* Every variant is bit-identical to scalar for non-NaN results, a NaN result only has to be NaN. No FMA.
* Out-of-range indices and keys, `div_const` by zero, invalid `slice` bounds, and a bucket count whose allocation fails are runtime errors, never panics or aborts.
* Acceptance query programs compute the SQL's answer for any data with the same schema: bucket counts come from the data or the key type's range, never from this dataset's key range.
* Verkehrt does not copy corgi's scripts or programs. Query programs transcribe the `# sql:` header of the matching corgi query file into their own `// sql:` comment.
* Comments: no em-dashes, no structuring semicolons. Doc comments state contracts, reasoning goes inline. Never drop existing comments.
* Markdown in the repository: asterisk lists, paragraphs of 4 to 5 sentences, evidence-based claims only.
* Commits end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -A`.

## Review focus

* `div_const(i64::MIN, -1)` and `div_const(i32::MIN, -1)` return the dividend (wrapping), and negative dividends and divisors truncate toward zero. Test in Task 7.
* A filter that selects no rows, feeding `reduce(keys, max)` for a bucket count, must give the SQL's empty answer, not the empty-array error of `reduce_max`. Query programs guard with a lazy `if length(k) > 0`. Test in Task 9 (synthetic data where the filter keeps nothing).
* `histogram(keys, n)` and `reduce_into` with a huge `n` (for example `1 << 62`) return an allocation error instead of aborting. Test in Task 5.
* `slice` of a `slice` composes offsets: `slice(slice(x, 2, 8), 1, 3)` is rows 3 and 4 of `x`. Test in Task 4.
* Keys equal to `n - 1` are accepted, keys equal to `n` and negative keys are rejected, for `histogram`, `reduce_into`, and index `length(xs) - 1` against `length(xs)` and `-1` for `gather`. Tests in Tasks 4 and 5.

## File structure

* `src/kernels.rs`: `KernelClass::Scan`, `ScanFn`, `Scanner`, `ScanBody`, `WholeBody`, the `WholeFn` change, and the `KernelImpl` table builders (Task 1).
* `src/kernels/auto.rs`, `src/kernels/arch.rs`, `src/kernels/x86.rs`, `src/kernels/aarch64.rs`: auto entries and registration hooks for scans and whole bodies (Task 1).
* `src/kernels/dispatch.rs`: `call` for scans, `Dispatch::scanner` (Task 1).
* `src/columns.rs` (new), `src/interp.rs`: column-file loading and `Prepared::inputs` (Task 9).
* `src/kernels/scalar/scan.rs` (new): `prev`, `scan`, `scan_by` bodies and registration (Tasks 1 and 3).
* `src/kernels/scalar/index.rs` (new): `gather`, `slice`, `histogram`, `reduce_into`, `argsort` bodies and registration (Tasks 4, 5, 6).
* `src/kernels/scalar/elementwise.rs`: `div_const` (Task 7).
* `src/ops.rs`, `src/check.rs`, `src/lower.rs`: new ops, builtins with an operation-name argument, `op_name` (Tasks 1 to 7).
* `tests/isa_variants.rs`: scan class and whole kernels with variants, index-aware argument generation (Tasks 1, 4, 5).
* `tests/programs/`: one program per primitive group plus the sorted group-by and string-length programs (Tasks 1 to 8).
* `scripts/prepare-acceptance.sh` (new): data preparation (Task 8).
* `tests/acceptance.rs`, `tests/acceptance/` (new): harness, synthetic check, query programs (Tasks 10, 10).
* `benches/acceptance.rs` (new), `benches/tier1.rs`: benches (Task 11).
* `docs/measurements/2026-10-31-tier2.md` (new), the base spec's catalog, `README.md` (Task 12).

---

### Task 1: Scan class, whole-kernel variants, and `prev`

**Files:**
* Modify: `src/kernels.rs`, `src/kernels/auto.rs`, `src/kernels/arch.rs`, `src/kernels/x86.rs`, `src/kernels/aarch64.rs`, `src/kernels/dispatch.rs`, `src/kernels/scalar.rs`, `src/kernels/scalar/construct.rs` (only where `WholeFn` coercion needs it), `src/ops.rs`, `src/check.rs`, `src/lower.rs`
* Create: `src/kernels/scalar/scan.rs`, `tests/programs/prev.vk`, `tests/programs/prev.out`
* Test: `src/kernels/tests.rs`, `src/kernels/dispatch/tests.rs`, `tests/isa_variants.rs`, `tests/programs.rs`

**Interfaces:**
* Produces, in `src/kernels.rs`:
  * `pub type ScanFn = unsafe fn() -> Box<dyn Scanner>;`
  * `pub trait Scanner: Send { fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String>; }`, where `out` holds exactly the output rows of this tile and the state carries to the next `update`. Tiles may have any length, and the output must not depend on how rows split into tiles.
  * `pub trait ScanBody: Scanner + Sized + 'static { fn new() -> Self; }`
  * `pub trait WholeBody: 'static { fn run(args: &[Value]) -> Result<Value, String>; }`
  * `pub type WholeFn = unsafe fn(&[Value]) -> Result<Value, String>;` (was a safe `fn`, existing whole kernels coerce).
  * `KernelClass::Scan`, `KernelImpl::Scan(ScanFn)`, and `KernelImpl::scan::<S: ScanBody>()` and `KernelImpl::whole::<W: WholeBody>()` returning `[Option<KernelImpl>; Variant::COUNT]` with scalar plus auto entries, like `KernelImpl::reduction`.
* Produces in `src/kernels/arch.rs`: `add_scan::<S: ScanBody>` and `add_whole::<W: WholeBody>`, forwarding like `add_reduction`.
* Produces in `src/kernels/dispatch.rs`: `Dispatch::call` handles `Scan` by preparing args like a reduction, allocating the output array, and calling `update` once with all rows (not at all for zero rows). `pub fn scanner(&self, reg: &Registry, id: KernelId) -> Result<Box<dyn Scanner>, String>`, like `reducer`.
* Produces `Op::Prev` and the builtin `prev(x, fill)` with kernels `prev_<t>` for `f64`, `f32`, `i64`, `i32`, `u8`, `bool`, inputs `[Array(t), Scalar(t)]`, output `Array(t)`. `prev` is not elementwise, so it is rejected inside lambdas.

- [ ] **Step 1: Write the failing tests.**
  * `src/kernels/tests.rs`: `prev_shifts_rows_and_fills_row_zero` with `prev([1, 2, 3], 9)` on `i64` giving `[9, 1, 2]`, `prev([], 9)` giving `[]`, and `prev` on `bool` giving 0 or 1 bytes only.
  * `src/kernels/dispatch/tests.rs`: `scanner_tiles_match_whole_call` feeding `prev_i64` over 37 rows in tiles of 5, 16, and 16 through `Dispatch::scanner` and comparing with `Dispatch::call`.
  * `tests/isa_variants.rs`: extend `check_case` with a `KernelClass::Scan` arm that runs the scan tiled at the existing cuts (a new `scan_tiled`, like `reduce_tiled`) and whole kernels with ISA entries through `call`, and extend `every_array_kernel_has_auto_variants` so `Scan` kernels and `Whole` kernels built with `KernelImpl::whole` must have auto entries on x86. The test decides which whole kernels need variants from a list of name prefixes kept in the test (`gather_`, `histogram`, `reduce_into_`), not from the impls table, so a kernel registered with `scalar_only()` by mistake fails it. Later tasks extend the list.
  * `tests/programs/prev.vk` with `print(prev([1, 2, 3], 0));` and `print(x - prev(x, 0));` over `let x = iota(5);`, and its `.out`.
  * `src/lower/tests.rs`: `prev_keeps_its_scalar_operand` asserting `prev(x, 0)` lowers to `Op(Prev, [x, Leaf(c)])` with no `Bcast`, and an existing elementwise case such as `x + 1` still lowering with `Bcast`.
  * `src/check/tests.rs`: `prev_is_rejected_inside_a_lambda` with `map(x, |v| prev(v, 0))` failing with an error naming `prev`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: the new tests fail to compile or fail.
- [ ] **Step 3: Implement** the class, the auto entries in `auto_backend!` (a scanner wrapper like `OnIsa` for reducers, and a whole entry like `elementwise`), the arch hooks, dispatch, `Op::Prev`, the builtin, `op_name`, and `prev_<t>` in `src/kernels/scalar/scan.rs` registered from `Registry::new`. Change `Lowerer::op` (`src/lower.rs`) to look up the kernel on the raw argument types first and wrap in `Bcast` only the scalar operands at `Either` positions, because today it broadcasts every scalar next to an array, so a kernel with a `Scalar(t)` input never resolves. This keeps the `PlanNode::Kernel` contract that an argument for a `Scalar` input is the scalar itself. Confirm `--plan` shows `prev_i64` for the program.
- [ ] **Step 4: Run** `cargo fmt --all`, `CARGO_INCREMENTAL=0 cargo clippy --locked --workspace --all-targets -- -D warnings`, `CARGO_INCREMENTAL=0 cargo test --locked`, and `CARGO_INCREMENTAL=0 cargo test --release --locked --test isa_variants`. Expected: all pass, and the `isa_variants` output counts the new auto entries.
- [ ] **Step 5: Commit** `Add a scan kernel class, whole-kernel variants, and prev`.

### Task 2: `u64` kernels

**Files:**
* Modify: `src/value.rs` (`ScalarType::V1_ARRAY`), `src/kernels/scalar/elementwise.rs`, `src/kernels/scalar/reduce.rs`, `src/kernels/scalar/select.rs`, `src/kernels/scalar/fused.rs`, `src/kernels/scalar/construct.rs`, the x86 and aarch64 intrinsics registration for compress and the fused filter
* Test: `src/kernels/tests.rs`, `tests/isa_variants.rs`, `tests/fused_oracle.rs`

**Interfaces:**
* Produces: every kernel family that has an `i64` kernel gets a `u64` kernel with the same inputs and the `u64` element type: the elementwise arithmetic and comparisons, casts from and to every array type, `reduce` with `add`, `mul`, `min`, `max`, `argmin`, `argmax`, the selections (`compress`, fused filters), the fused reductions (`dot`, `count_<cmp>`), `fill`, `load_u64`, and `prev` from Task 1. `sum` of `u64` returns `u64`. Comparisons are unsigned, arithmetic wraps, and casts follow Rust `as`.
* `ScalarType::V1_ARRAY` gains `U64`, with its doc updated, and Task 12 updates the base spec's "Scope of the first version" line.

- [ ] **Step 1: Write the failing tests.**
  * `u64_compares_unsigned`: `[1 << 63, 1]` with `> 1` gives `[true, false]`, which a signed compare gets wrong.
  * `u64_casts_follow_as`: `cast_f64` of `u64::MAX` gives `18446744073709551615.0`, `cast_u64` of `-1.0` gives 0 and of `1e30` gives `u64::MAX`, and `cast_i64` of `1 << 63` gives `i64::MIN`.
  * `u64_sum_and_extremes`: `sum`, `reduce(x, min)`, `reduce(x, max)`, and `argmax` over `[3, 1 << 63, 7]`.
  * `every_i64_kernel_has_a_u64_twin`: for every registered kernel whose name ends in `_i64`, a kernel with `_u64` in its place exists, except kernels that take `i64` as an index or count, which the test lists by name.
  * `tests/isa_variants.rs` and `tests/fused_oracle.rs` pick up the new kernels through the registry. Add `u64` to their type strategies.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: FAIL.
- [ ] **Step 3: Implement** by adding `u64` to the macros and type lists that instantiate `i64` kernels. The intrinsics filters need unsigned compares: AVX-512 has `vpcmpuq`, AVX2 needs the sign-bias trick (flip bit 63 of both operands, then `vpcmpgtq`), and NEON has `vcgtq_u64`.
- [ ] **Step 4: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass, and the entry-count test in `tests/isa_variants.rs` is updated to the new counts with the increase explained in the commit message.
- [ ] **Step 5: Commit** `Add u64 kernels`.

### Task 3: `scan` and `scan_by`

**Files:**
* Modify: `src/kernels/scalar/scan.rs`, `src/ops.rs`, `src/check.rs`, `src/lower.rs`
* Create: `tests/programs/scan.vk`, `tests/programs/scan.out`
* Test: `src/kernels/tests.rs`, `src/check/tests.rs`, `tests/programs.rs`

**Interfaces:**
* Consumes: `ScanBody`, `KernelImpl::scan` (Task 1), the `u64` kernels (Task 2).
* Produces `Op::Scan(ReduceOp)` and `Op::ScanBy(ReduceOp)`, builtins `scan(x, op)` and `scan_by(x, starts, op)`, and `op_name` giving `scan_<op>` and `scan_by_<op>`.
* Kernels `scan_<op>_<t>` with input `[Array(t)]` and `scan_by_<op>_<t>` with inputs `[Array(t), Array(bool)]`, for:
  * `i64` and `u64`: `add`, `mul`, `min`, `max`, output the element type.
  * `i32`: `add` and `mul` with output `i64`, `min` and `max` with output `i32`.
  * `u8`: `add` and `mul` with output `u64`, `min` and `max` with output `u8`.
  * `f64`, `f32`: `min`, `max`, output the element type, IEEE `minimum` and `maximum`.
  * `bool`: `and`, `or`, `xor`, mirroring `register_bool` in `scalar/reduce.rs`. Integer `and`, `or`, and `xor` have no `reduce` kernel today, so they are out of scope here and listed as a surface gap in Task 12.
* The builtin table generalizes the existing `Builtin::Reduce` handling: the operation name is always the last argument, and the same `reduce_op` parser and error messages apply.

- [ ] **Step 1: Write the failing tests.**
  * `scan_add_i64_is_a_prefix_sum`: `[1, 2, 3, 4]` gives `[1, 3, 6, 10]`, and `[i64::MAX, 1]` wraps to `[i64::MAX, i64::MIN]`.
  * `scan_add_i32_widens`: `[i32::MAX, 1]` gives the `i64` array `[2147483647, 2147483648]`.
  * `scan_max_f64_propagates_nan`: `[1.0, NaN, 3.0]` gives `[1.0, NaN, NaN]`, and `[-0.0, 0.0]` under `max` gives `[-0.0, 0.0]` with the signs checked by `to_bits`.
  * `scan_by_restarts_at_starts`: values `[1, 2, 3, 4, 5]`, starts `[false, false, true, false, true]`, `add`, gives `[1, 3, 3, 7, 5]`, and row 0 starts a segment even when `starts[0]` is false.
  * `scan_last_equals_sum`: for `i64` and `i32` random inputs, the last row of `scan(x, add)` equals `sum(x)`.
  * `scan_add_u8_widens`: `[255, 1]` gives the `u64` array `[255, 256]`.
  * `src/check/tests.rs`: `scan_add_f64_has_no_kernel` gives the registry's "no kernel" type error, and `scan(x, foo)` gives the unknown-operation error.
  * `tests/programs/scan.vk` covering `scan` and `scan_by` once each, with `.out`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: FAIL.
- [ ] **Step 3: Implement** the ops, builtins, and kernels. Scanners carry the running value (and for `scan_by`, nothing else, since row 0 of a tile continues the previous segment unless `starts` says otherwise).
- [ ] **Step 4: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass, the tiled scan check passes for every new kernel.
- [ ] **Step 5: Commit** `Add scan and scan_by`.

### Task 4: `gather` and `slice`

**Files:**
* Create: `src/kernels/scalar/index.rs`, `tests/programs/gather.vk`, `tests/programs/gather.out`
* Modify: `src/kernels/scalar.rs`, `src/kernels.rs` (`Registry::new`), `src/ops.rs`, `src/check.rs`, `tests/isa_variants.rs`
* Test: `src/kernels/tests.rs`, `tests/isa_variants.rs`

**Interfaces:**
* Consumes: `WholeBody`, `KernelImpl::whole` (Task 1).
* Produces `Op::Gather` and `Op::Slice`, builtins `gather(xs, idx)` and `slice(x, start, end)`.
* `gather_<t>`: class `Whole`, built with `KernelImpl::whole`, inputs `[Array(t), Array(i64)]`, output `Array(t)`, for `f64`, `f32`, `i64`, `u64`, `i32`, `u8`, and `bool`.
* `slice_<t>`: class `Whole`, scalar entry only, inputs `[Array(t), Scalar(i64), Scalar(i64)]`, output `Array(t)`, implemented with `Array::slice` so it does not copy.

- [ ] **Step 1: Write the failing tests.**
  * `gather_picks_rows`: `gather([10, 20, 30], [2, 0, 2])` gives `[30, 10, 30]`, and an empty `idx` gives `[]`.
  * `gather_rejects_out_of_range`: index `3` and index `-1` into a 3-row array are runtime errors naming the index, and index `2` is accepted.
  * `slice_is_a_view`: `slice(iota(10), 2, 8)` gives rows 2 to 7, `slice(slice(x, 2, 8), 1, 3)` gives `[3, 4]`, and `slice(x, 0, 0)` and `slice(x, 10, 10)` give `[]`.
  * `slice_rejects_bad_bounds`: `start > end`, `end > length`, and a negative `start` are runtime errors.
  * `tests/isa_variants.rs`: the case strategy draws a raw index per row, and `args_for` maps it into `0..n` once the length `n` is known, keyed on the kernel name `gather_*`, with a minority of cases forcing one index to `n` or `-1`, so both the value path and the error path are compared.
  * `tests/programs/gather.vk` with `.out`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: FAIL.
- [ ] **Step 3: Implement.** `gather` checks all indices before writing (one pass computing whether any index is out of range), so the error does not depend on the order of rows. The second pass clamps each index to `len - 1`, which the first pass made a no-op, so the loop has no early exit and the compiler can vectorize it.
- [ ] **Step 4: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass.
- [ ] **Step 5: Commit** `Add gather and slice`.

### Task 5: `histogram` and `reduce_into`

**Files:**
* Modify: `src/kernels/scalar/index.rs`, `src/ops.rs`, `src/check.rs`, `tests/isa_variants.rs`
* Create: `tests/programs/buckets.vk`, `tests/programs/buckets.out`
* Test: `src/kernels/tests.rs`, `src/check/tests.rs`, `tests/isa_variants.rs`

**Interfaces:**
* Consumes: `WholeBody`, `KernelImpl::whole` (Task 1), the operation-name builtin handling (Task 3).
* Produces `Op::Histogram` and `Op::ReduceInto(ReduceOp)`, builtins `histogram(keys, n)` and `reduce_into(keys, vals, n, op)`, `op_name` giving `reduce_into_<op>`.
* `histogram`: class `Whole`, with variants, inputs `[Array(i64), Scalar(i64)]`, output `Array(i64)`.
* `reduce_into_<op>_<t>`: class `Whole`, with variants, inputs `[Array(i64), Array(t), Scalar(i64)]`, with the same type and operator coverage and output types as `scan` in Task 3. Empty buckets hold the operator's identity: `0` for `add`, `1` for `mul`, the type's largest value for `min` (`+inf` for floats), its smallest for `max` (`-inf` for floats), all ones for `and`, `0` for `or` and `xor`.
* No planner pattern rewrites `reduce_into(keys, fill(length(keys), 1), n, add)` to `histogram`: egglog leaves do not expose a constant's value, so the pattern could not require the fill value to be 1. Task 12 lists it as a candidate that needs a constant relation in the prelude.

- [ ] **Step 1: Write the failing tests.**
  * `histogram_counts_keys`: keys `[0, 2, 2, 3]`, `n = 5`, gives `[1, 0, 2, 1, 0]`, and empty keys with `n = 0` give `[]`.
  * `bucket_keys_are_range_checked`: key `n` and key `-1` are runtime errors for both kernels, key `n - 1` is accepted, and a negative `n` is a runtime error.
  * `huge_bucket_count_is_an_error`: `histogram([0], 1 << 62)` returns an error, does not abort.
  * `reduce_into_sums_per_bucket`: keys `[1, 0, 1]`, values `[5, 7, 9]`, `n = 3`, `add`, gives `[7, 14, 0]`, and `i32` values give an `i64` result.
  * `reduce_into_empty_buckets_hold_identity`: `min` on `f64` gives `+inf` in an empty bucket, `max` on `i64` gives `i64::MIN`.
  * `reduce_into_with_one_key_matches_reduce`: with every key 0 and `n = 1`, every supported operator gives `reduce(vals, op)` (for `min` and `max` on non-empty input).
  * `reduce_into_checks_lengths`: keys and values of different lengths are a runtime error.
  * `src/check/tests.rs`: `reduce_into_add_f64_has_no_kernel`.
  * `tests/isa_variants.rs`: as in Task 4, `args_for` draws the bucket count in `1..=64` and maps raw keys into `0..n`, with a minority out of range, for `histogram` and `reduce_into_*`.
  * `tests/programs/buckets.vk` with `.out`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: FAIL.
- [ ] **Step 3: Implement.** Allocate buckets through `Array::try_build` so a failed allocation is an error. Check keys in one pass before accumulating, as `gather` does.
- [ ] **Step 4: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass.
- [ ] **Step 5: Commit** `Add histogram and reduce_into`.

### Task 6: Sorted group-by and string-length programs

**Files:**
* Create: `tests/programs/group_sorted.vk`, `tests/programs/group_sorted.out`, `tests/programs/string_len.vk`, `tests/programs/string_len.out`

**Interfaces:**
* Consumes: `prev`, `scan`, `scan_by`, `gather`, `reduce_into`, `compress` (Tasks 1, 3, 4, 5, tier 1).

- [ ] **Step 1: Write** `group_sorted.vk` over sorted keys `k = [1, 1, 2, 2, 2, 5]` and values `v = [1, 2, 3, 4, 5, 6]`, with `n = length(k)`:
  * run starts `s = or(k != prev(k, 0), iota(n) == 0)`, which needs no first key,
  * run ends `e = or(k != gather(k, min(iota(n) + 1, n - 1)), iota(n) == n - 1)`,
  * per-run totals two ways, `compress(scan_by(v, s, add), e)` and `reduce_into(scan(cast_i64(s), add) - 1, v, count(s), add)`, both printing `[3, 12, 6]`,
  * the distinct keys `compress(k, s)` printing `[1, 2, 5]`.
  Write `string_len.vk`: an `ends` array `[3, 3, 7, 7, 8]` gives lengths `ends - prev(ends, 0)` = `[3, 0, 4, 0, 1]` and the non-empty mask `[true, false, true, false, true]`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked --test programs`. Expected: PASS once the `.out` files hold the printed results, checked by hand against the values above.
- [ ] **Step 3: Commit** `Add sorted group-by and string-length programs`.

### Task 7: `div_const` and `argsort`

**Files:**
* Modify: `src/kernels/scalar/elementwise.rs`, `src/kernels/scalar/index.rs`, `src/ops.rs`, `src/check.rs`, `src/lower.rs` (if `DivConst` joins the elementwise ops)
* Create: `tests/programs/div_sort.vk`, `tests/programs/div_sort.out`
* Test: `src/kernels/tests.rs`, `src/check/tests.rs`, `src/lower/tests.rs`

**Interfaces:**
* Produces `Op::DivConst` and `Op::ArgSort`, builtins `div_const(x, c)` and `argsort(x)`.
* `div_const_<t>` for `i64`, `u64`, `i32`, `u8`: class `Elementwise`, the dividend takes an array or a scalar like other elementwise operands, the divisor only a scalar (a broadcast), output `t`. `DivConst` counts as elementwise for lambda lifting, so `map(x, |v| div_const(v, 60))` works, and an array divisor is a type error naming `div_const`. Declare the divisor as `InputSpec::Scalar(t)`, which the lowering change of Task 1 keeps unbroadcast. The elementwise entry receives it as `Arg::Splat`.
* `argsort_<t>` for `i64`, `u64`, `i32`, `u8`: class `Whole`, scalar entry only, input `[Array(t)]`, output `Array(i64)`, stable ascending.

- [ ] **Step 1: Write the failing tests.**
  * `div_const_truncates_toward_zero`: `[7, -7, 0]` by `2` gives `[3, -3, 0]`, by `-2` gives `[-3, 3, 0]`.
  * `div_const_wraps_at_min`: `i64::MIN` by `-1` gives `i64::MIN`, and `i32::MIN` by `-1` gives `i32::MIN`.
  * `div_const_matches_rust_division`: a property test over random dividends and every divisor in `-300..=300` except 0, plus `i64::MIN`, `i64::MAX`, and their neighbours, against `wrapping_div`, for `i64`, `i32`, `u64` (divisors in `1..=300` plus `u64::MAX`), and `u8` (divisors in `1..=255`).
  * `div_const_by_zero_is_an_error`.
  * `argsort_is_stable`: `[3, 1, 3, 1]` gives `[1, 3, 0, 2]`, and an empty array gives `[]`.
  * `src/lower/tests.rs`: `div_const_lifts_in_a_lambda` for `map(x, |v| div_const(v, 60))`.
  * `src/check/tests.rs`: `div_const_rejects_an_array_divisor` and `argsort_f64_has_no_kernel`.
  * `tests/programs/div_sort.vk`, including `argsort(0 - c)` for descending order, with `.out`.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked`. Expected: FAIL.
- [ ] **Step 3: Implement.** `div_const` computes its multiply-high and shift constants once per call from the divisor, with special cases where they do not apply (divisor 1, -1, and powers of two are the usual ones). The property test is the check that the constants are right.
- [ ] **Step 4: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass, `div_const_<t>` has auto entries on x86.
- [ ] **Step 5: Commit** `Add div_const and argsort`.

### Task 8: Acceptance data

**Files:**
* Create: `scripts/prepare-acceptance.sh`

**Interfaces:**
* Produces, under `~/dev/repos/data/verkehrt/`:
  * `corgi-wip/`: a clone of `https://github.com/frankmcsherry/WIP` at `2751e04e8501d47b33219be23049bf4afa7a4428`.
  * `hits_0.parquet`: from `https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet`.
  * `clickbench/` and `tpch/`: the `prepare.py` outputs, each with `*.u64`, `*.ends`, `*.bytes`, `expected/*.txt`, and `duckdb.tsv`.

- [ ] **Step 1: Write** `scripts/prepare-acceptance.sh DATA_DIR`: `set -euo pipefail`, clone the repository if `DATA_DIR/corgi-wip` is missing and check out the pinned revision, download `hits_0.parquet` if missing, then run `uv run --with duckdb --with numpy python3 corgi-wip/corgi/benches/clickbench/prepare.py DATA_DIR/clickbench DATA_DIR/hits_0.parquet` and `uv run --with duckdb --with numpy python3 corgi-wip/corgi/benches/tpch/prepare.py DATA_DIR/tpch 0.2`, and print the DuckDB version with `uv run --with duckdb python3 -c 'import duckdb; print(duckdb.__version__)'`. Every step is idempotent.
- [ ] **Step 2: Run** `bash scripts/prepare-acceptance.sh ~/dev/repos/data/verkehrt`. Expected: both data directories exist, `ls ~/dev/repos/data/verkehrt/clickbench/expected/` lists `q01.txt` through `q42.txt`, `ls ~/dev/repos/data/verkehrt/tpch/expected/` lists `q01.txt` and `q06.txt` among others. If the sandbox refuses the clone or the network, stop and return BLOCKED with the exact error.
- [ ] **Step 3: Record** the DuckDB version, the total size (`du -sh`), and the row counts in the report for Task 12.
- [ ] **Step 4: Commit** `Add the acceptance data preparation script`.

### Task 9: Acceptance harness

**Files:**
* Create: `src/columns.rs`, `tests/acceptance.rs`, `tests/acceptance/harness.rs` (declared as `mod harness;` from `tests/acceptance.rs`)
* Modify: `src/interp.rs`, `src/lib.rs`

**Interfaces:**
* Produces in the library, because the bench in Task 11 needs the same loading and may not include test files:
  * `Prepared::inputs(&self) -> impl Iterator<Item = (&str, ValueType)>`: the declared inputs in source order.
  * `verkehrt::columns::load_inputs(dir: &Path, prepared: &Prepared) -> Result<Vec<(String, Value)>, String>`: for every declared input, reads `dir/<name>.u64` as `u64[]` or `i64[]`, whichever the input declares, or `dir/<column>.bytes` as `u8[]` when the input is named `<column>_bytes` with type `u8[]`, checking that the `.bytes` file has one byte per row of `dir/<column>.ends`. Any other input type is an error naming the input. The doc comment states this file convention.
* Produces in `tests/acceptance/harness.rs`:
  * `fn parse_kinds(program: &str) -> Vec<char>`: the kind letters of the `// output:` comment.
  * `fn format_rows(kinds: &[char], exports: &[(String, Value)]) -> Result<String, String>`: one line per row, tab-separated, with `u` and `i` as decimal integers, `f` as `{:.4}`, and `s` as lowercase hex of the one byte of a `u8` row, with scalar exports counting as one row and all exports required to have the same row count.
  * `fn run_query(dir: &Path, program_path: &Path) -> Result<String, String>`: prepares the program, loads its columns with `load_inputs`, executes, and formats.
* Acceptance tests read `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH` and print a skip message and return when the variable is unset.

- [ ] **Step 1: Write the failing tests.**
  * `src/columns.rs` unit tests: `load_inputs_reads_u64_and_bytes` over files written under `std::env::temp_dir()` in a directory named after the test and the process id (unit tests have no `CARGO_TARGET_TMPDIR`), and `load_inputs_rejects_multibyte_strings` where `.ends` is `[1, 3]`.
  * `harness_formats_like_prepare_py`: `format_rows` on kinds `[s, u, f]` with exports `u8[] [65, 78]`, `i64[] [3, 4]`, `f64[] [0.5, 1.0 / 3.0]` gives `"41\t3\t0.5000\n4e\t4\t0.3333\n"`.
  * `harness_runs_a_synthetic_query`: writes column files for a 4-row table to a temporary directory under the cargo target directory (`env!("CARGO_TARGET_TMPDIR")`), runs a small program through `run_query`, and compares with expected text computed in Rust.
  * `harness_empty_filter_gives_empty_output`: a synthetic q07-shaped program, whose bucket count is `if length(k) > 0 { reduce(k, max) + 1 } else { 0 }`, over data whose filter keeps no row gives the empty string, as DuckDB's `expected` file would be. The same program without the `if` fails with the empty-array error of `reduce_max`, and the test asserts that too.
- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --locked --test acceptance`. Expected: FAIL.
- [ ] **Step 3: Implement** the helpers.
- [ ] **Step 4: Run** `CARGO_INCREMENTAL=0 cargo test --locked --test acceptance`. Expected: PASS, with skip messages for the two suites.
- [ ] **Step 5: Commit** `Add the acceptance harness`.

### Task 10: Acceptance queries

**Files:**
* Create: `tests/acceptance/tpch/q01.vk`, `tests/acceptance/tpch/q06.vk`, `tests/acceptance/clickbench/q01.vk`, `tests/acceptance/clickbench/q02.vk`, `tests/acceptance/clickbench/q06.vk`, `tests/acceptance/clickbench/q07.vk`, `tests/acceptance/clickbench/q42.vk`
* Modify: `tests/acceptance.rs`

**Interfaces:**
* Consumes: everything above. The SQL and its constants come from the `# sql:` lines of `~/dev/repos/data/verkehrt/corgi-wip/corgi/algorithms/{tpch,clickbench}/<query>.col`, and the output kinds from its `# output:` line.

- [ ] **Step 1: Write the failing test** `acceptance_queries_match_duckdb`: for each suite whose variable is set, for each `.vk` file in its directory, `run_query` output equals `expected/<query>.txt` byte for byte, and a mismatch prints the query name, the first differing line, and both lines.
- [ ] **Step 2: Write the programs**, as the spec's query table describes. Q1 combines the flags as `cast_i64(rf) * 256 + cast_i64(ls)` with `n = 65536` (the range of two `u8` keys), splits the key back with `div_const`, and orders by key through `where(counts > 0)`. q07 and q42 derive their bucket counts from the filtered data and guard only those scalars with a lazy `if` (`histogram`, `where`, `gather`, `argsort`, and `slice` all handle empty inputs), and q42 applies `OFFSET 1000 LIMIT 10` with `slice` and `min(length, ...)` bounds. q07 orders with one stable `argsort(0 - c)` over keys that `where` already listed in ascending order.
- [ ] **Step 3: Run** `VERKEHRT_CLICKBENCH=~/dev/repos/data/verkehrt/clickbench VERKEHRT_TPCH=~/dev/repos/data/verkehrt/tpch CARGO_INCREMENTAL=0 cargo test --release --locked --test acceptance`. Expected: PASS for all seven. A float mismatch in the last printed digit is a real mismatch, investigate how DuckDB computes that average before changing anything.
- [ ] **Step 4: Record plans.** Add an ignored test `acceptance_plans` that prepares each program with `Options { show_plan: true, .. }` through `Interpreter::prepare_to` and prints the plans, run it with `-- --ignored --nocapture` and the variables set, and save the output for the Task 12 report.
- [ ] **Step 5: Commit** `Add the seven acceptance queries`.

### Task 11: Benches

**Files:**
* Create: `benches/acceptance.rs`
* Modify: `benches/tier1.rs`, `Cargo.toml` (`[[bench]] name = "acceptance" harness = false`)

**Interfaces:**
* Consumes: `Prepared::inputs` and `verkehrt::columns::load_inputs` (Task 9).

- [ ] **Step 1: Write** `benches/tier1.rs` groups at 10M rows, each with the interpreter per variant (scalar, and on x86 AVX2 and AVX-512 auto) and a native Rust loop compiled for the same ISA, like the existing tier 1 groups: `scan-add-i64`, `gather-i64` (random indices into a 10M array, with an AVX2 and AVX-512 native baseline using `_mm256_i64gather_epi64` and `_mm512_i64gather_epi64`, not a scalar indexed load), `histogram` (keys uniform in `0..1024`), `reduce-into-add-i64` (keys uniform in `0..1024`), and `div-const-i64` (divisor 60).
- [ ] **Step 2: Write** `benches/acceptance.rs`: one criterion group per query, timing `execute` only with the program prepared and the columns loaded once, sample size 10, skipping suites whose variable is unset.
- [ ] **Step 3: Run** `CARGO_INCREMENTAL=0 cargo bench --locked --bench tier1 -- 'scan|gather|histogram|reduce-into'` and the acceptance bench with both variables set. In the same session, run corgi's `CORGI_CLICKBENCH=~/dev/repos/data/verkehrt/clickbench CARGO_INCREMENTAL=0 cargo bench --bench clickbench -- q01 q02 q06 q07 q42` and `CORGI_TPCH=~/dev/repos/data/verkehrt/tpch CARGO_INCREMENTAL=0 cargo bench --bench tpch -- q01 q06` from `~/dev/repos/data/verkehrt/corgi-wip/corgi`, after reading its bench source to note how it times a query. Save all outputs in the scratchpad for Task 12.
- [ ] **Step 4: Decide intrinsics.** The native loop is often the same portable body, so "within 1.5x of native" cannot show that a kernel failed to vectorize. For each new kernel, record three numbers at 10M: the best `auto` time over the scalar-entry time, its GB/s against the `add_i64` streaming kernel at 10M from the same run, and its ratio to the native baseline. Then:
  * A kernel whose `auto` entry runs at about the scalar entry's speed (ratio above 0.9) did not vectorize. It gets the intrinsics entry the spec names (log-step shifts for the integer `scan` and `scan_by`, `vpgatherqq` for `gather`, `vpconflictq` or replicated sub-histograms for `histogram` and `reduce_into` on AVX-512), or, where the spec names none or the intrinsics do not beat the scalar entry, a strain-case row in Task 12 saying it is a scalar loop on that ISA.
  * `div_const_i64` and `div_const_u64` need a 64-bit multiply-high, which neither AVX2 nor AVX-512 has, so they are expected to be strain cases like `mul_i64`. Measure and record either way.
  * A kernel more than 1.5x slower than native after that gets an entry in the measurement document naming the gap.
  Every intrinsics entry is covered by `tests/isa_variants.rs`. Record each decision with its numbers.
- [ ] **Step 5: Run** fmt, clippy, `cargo test --locked`, release `isa_variants`. Expected: all pass.
- [ ] **Step 6: Commit** `Add tier 2 and acceptance benches`, and one commit per intrinsics entry from Step 4 named after the kernel.

### Task 12: Measurements, catalog, and README

**Files:**
* Create: `docs/measurements/2026-10-31-tier2.md`
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (tier 2 catalog and strain cases), `README.md`

- [ ] **Step 1: Ask the user** to pin the x86 governor and decide on boost before the final run, as the benchmark-host notes require, and record the setting they report.
- [ ] **Step 2: Write** the measurement document in the form of `docs/measurements/2026-10-30-x86-kernels.md`: setup (machine, clock, corgi revision, DuckDB version, data sizes and row counts), how each system was timed, the tier 2 kernel table against native with the intrinsics decisions, and the acceptance table with Verkehrt per ISA, corgi, DuckDB one-thread, and the ratios. Name the first gap to close for plan 5 and any fused-kernel candidate visible in the traces.
- [ ] **Step 3: Update** the base spec's tier 2 catalog: one row per new primitive with its x86 and aarch64 instructions or strain-case note, the strain cases (gather on NEON, bucketed reductions, the scalar `argsort`, and every kernel that Task 11 Step 4 found to be a scalar loop on an ISA, expected to include the integer `scan` without intrinsics and `div_const` on 64-bit types), the `u64` cast and compare costs on AVX2, the "Scope of the first version" line updated for `u64`, and a "Surface gaps" list matching the tier 2 spec's deferred section plus integer `and`, `or`, and `xor` (no reduce or elementwise kernels exist) and the `histogram` pattern for `reduce_into` with a fill of 1.
- [ ] **Step 4: Update** `README.md`: the primitive list where it names primitives, and a short results paragraph for the acceptance queries pointing at the measurement document, in the style of the existing results paragraphs.
- [ ] **Step 5: Commit** `Measure tier 2 kernels and the acceptance queries`, push the branch, and open a draft pull request against `main`, marked as posted by Claude Code.

## Hand-off: M5 session

After the pull request is open, a session on the M5 runs the NEON part, following the plan 3 pattern.

* Check out the branch, run `bash scripts/prepare-acceptance.sh ~/dev/repos/data/verkehrt` there (or the M5 user's chosen directory), and run the full test suite including release `isa_variants` and the acceptance test.
* Run the tier 2 groups of `tier1` and the acceptance bench, and corgi's two benches on the same data, on AC power in high power mode.
* Write NEON intrinsics only for groups more than 1.5x native, as in Task 11 Step 4. `gather` has no NEON instruction, so its result is recorded as a strain case whatever it measures.
* Add an aarch64 section to `docs/measurements/2026-10-31-tier2.md` and commit to the same branch.
