# Grouping without sorting implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `group_id` primitive (dense ids in first-occurrence order, one or two key columns), bench it against sort-based grouping, and rewrite the many-group ClickBench queries with it where that is faster.

**Architecture:** `group_id` is a new op (`Op::GroupId`) with whole kernels `group_id_<t>` and `group_id2_<t1>_<t2>`, registered like `argsort` in `src/kernels/scalar/index.rs` (or a new sibling module) with auto entries through `KernelImpl::whole`. The checker maps the source name `group_id` with one or two arguments to it.

**Tech Stack:** Rust, criterion (`benches/tier1.rs`, group `tier2`), proptest, the acceptance harness.

**Spec:** `docs/superpowers/specs/2026-11-01-verkehrt-group-id-design.md`

## Global Constraints

* Output: for each row, the dense `i64` id of its key, numbered by first occurrence starting at 0. It is independent of the hash function and the ISA.
* Key types are `i64`, `u64`, `i32`, and `u8`, and the two-column form takes any pair of them. Empty input gives an empty `i64` array. A length mismatch in the two-column form gives the existing length mismatch error.
* Allocation failure is an error, including the hash table.
* No changes to `argsort`, `histogram`, or `reduce_into`. No fused aggregation, no string keys, no third key column.
* Every performance claim is backed by committed raw output. Differences within about 10% are noise. Host: Zen 5, boost off, `powersave`, 3.0 GHz under load per the user.
* No unsafe beyond what exists, and `SAFETY` comments on any that is added. Use `#[expect]`, not `#[allow]`. No `#[path]`. Comments have no em-dashes or structuring semicolons.
* `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` pass before every commit.

## Review Focus

* Hash collisions. Keys chosen to collide under the chosen hash must still get distinct ids, and probing must terminate.
* Table growth. Inputs that cross every growth threshold, with the ids after a rehash still in first-occurrence order.
* Signed keys and two-column keys where one column alone is not unique, for example (1, 2) and (2, 1) as distinct pairs, and (x, y) repeated far apart.
* Empty input and a single row.

---

### Task 1: The `group_id` primitive

**Files:**
* Modify: `src/ops.rs` (`Op::GroupId`), `src/check.rs` (source name `group_id`, one or two arguments), and `src/kernels/scalar/index.rs` (bodies and registration), or a new module `src/kernels/scalar/group.rs` with its own `group/tests.rs`
* Modify: `tests/isa_variants.rs` (`WHOLE_AUTO_PREFIXES` gains `group_id`), `src/kernels/tests.rs` (kernel counts, if they are asserted)
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (catalog row for `group_id`, with a measurement placeholder that Task 2 fills)

- [ ] **Step 1:** Write a reference oracle, `group_id_reference`, that assigns first-occurrence ids with a `std::collections::HashMap`. Write property tests comparing the kernel with it for every key type and for pairs, covering the Review Focus classes. Add fixed tests for empty input, a single row, (1, 2) against (2, 1), a length mismatch, and a program-level test through the interpreter (`group_id(x)` and `group_id(a, b)` type-check, and the wrong arity or types give type errors).
- [ ] **Step 2:** Implement the bodies with an open-addressing table: a deterministic mixing hash, a power-of-two capacity at load factor at most 1/2, and fallible growth. Then check the Review Focus classes.
- [ ] **Step 3:** Run `tests/isa_variants.rs`. The auto entries must match scalar. Then commit "Add the group_id primitive".

### Task 2: Bench `group_id` against sort-based grouping

**Files:**
* Modify: `benches/tier1.rs` (rows `group-id-i64`, `group-id2-i64-i64`, and the sort idiom on the same inputs: two `argsort` passes and a run-start mask)
* Create: `docs/measurements/2026-11-01-group-id.md` and its raw-output directory
* Modify: `docs/superpowers/specs/2026-10-30-verkehrt-design.md` (fill the catalog row)

- [ ] **Step 1:** Bench at 1M and 10M rows, with keys uniform over the full range, with about 80K distinct values in 1M rows (the ClickBench `UserID` shape), and with about 10 distinct values. Measure scalar and both levels' auto entries.
- [ ] **Step 2:** If a design choice is open (hash function, probing, a `u8` direct table), measure the alternatives and record the choice.
- [ ] **Step 3:** Write the document, fill the catalog row, and commit "Measure group_id against sort-based grouping".

### Task 3: Rewrite the many-group queries

**Files:**
* Modify: `tests/acceptance/clickbench/q04.vk`, `q08.vk`, `q09.vk`, `q15.vk`, `q32.vk`, `q35.vk`, and, if faster, `q30.vk`, `q31.vk`, `q40.vk`, `q41.vk`

- [ ] **Step 1:** Rewrite each query with `group_id` followed by `reduce_into` and `histogram`. Recover each group's key with `reduce_into(ids, k, ng, min)`, which works because every row of a group has the same key. Order the groups by key with an `argsort` over the groups only, then apply the count sort. COUNT(DISTINCT v) per group becomes `group_id(g, v)`, followed by a count of the distinct pairs per group.
- [ ] **Step 2:** The acceptance suite and all synthetic tests in `tests/acceptance/main.rs` must pass unchanged. Keep a query's sort-based form when `group_id` is not faster, which Task 4 confirms.
- [ ] **Step 3:** Commit "Group the many-group ClickBench queries by hash".

### Task 4: Measurement

**Files:**
* Create: `docs/measurements/2026-11-01-group-id-queries.md` and its raw-output directory

- [ ] **Step 1:** Time the rewritten queries, and their sort-based versions from the commit before Task 3, against corgi and DuckDB on one thread, with the method of `docs/measurements/2026-11-01-clickbench-groups.md`. Collect per-kernel stats.
- [ ] **Step 2:** Record each query's dominating kernel and the follow-up it points to: #30 for maps before reductions, #16 for mask chains, top-k, or fused aggregation.
- [ ] **Step 3:** Revert any query whose rewrite is not faster beyond noise, and record why. Commit "Measure the hash-grouped ClickBench queries".
