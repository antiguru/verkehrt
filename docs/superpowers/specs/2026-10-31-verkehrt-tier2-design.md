# Verkehrt tier 2: design

This document extends `docs/superpowers/specs/2026-10-30-verkehrt-design.md`, the base spec, which stays binding for everything not changed here.
It covers plan 4: the tier 2 primitives of the problem catalog, and a set of external acceptance workloads that measure the language against corgi and DuckDB.

## Intent

Tier 1 showed that whole-array kernels can match native loops for sums, dot products, counts, and filters.
Tier 2 adds the primitives that relational work is built from: scans, gathers, and grouping.
They are also the first primitives where the "good code" map has known holes, because NEON has no gather, x86 gathers are slow on many cores, and scatters into the same bucket conflict.
Finding and recording those holes is as much the goal as the kernels themselves.

The user chose corgi's ClickBench and TPC-H query suites as the acceptance workloads, starting with TPC-H Q1 and the ClickBench group-by queries.
Each query that Verkehrt can express is then a three-way comparison, against corgi and against DuckDB on one thread, on the same data and with answers checked against DuckDB's.
This replaces judging tier 2 only by its own catalog, and it keeps later tiers honest, because every ClickBench query that stays out of reach names the primitive that is missing.

## Goals

* The tier 2 primitives below, each with a scalar reference kernel, an `auto` entry on AVX2 and AVX-512, and the property tests of the base spec.
* Intrinsics only where measurement shows a gap against the `auto` backend or the native baseline, as in plans 2 and 3.
* An acceptance harness that runs Verkehrt programs over the column files written by corgi's `prepare.py` scripts and checks the results against DuckDB's expected answers.
* Seven acceptance queries pass that check: TPC-H Q1 and Q6, and ClickBench q01, q02, q06, q07, and q42.
* A measurement document with Verkehrt, corgi, and DuckDB times for each acceptance query, and the strain cases found along the way.

## Non-goals

* Grouping on keys with many distinct values, which needs a sort or a hash table. It is plan 5, and it unlocks ClickBench q15, q30 to q32, q35, q40, and q41.
* A byte-string type. The string predicates that the acceptance queries need are expressed over offset columns (see below), and tier 4 adds real strings.
* Float `add` and `mul` in scans and bucketed reductions, and the other gaps listed under "Deferred: surface gaps", which a later phase closes.
* `reverse`, `rotate`, and packed bitmasks from the tier 2 catalog, also deferred (my call).
* Tile-at-a-time execution, though every new kernel must keep it possible.
* Running the acceptance queries in CI, because the data does not belong in the repository.

## Anti-goals

* Rewriting a query so it matches the data instead of the SQL. A query program must compute the SQL's answer for any data with the same schema, not only for `hits_0` or scale factor 0.2. Bucket counts derived from the data at run time are fine, and constants that encode this dataset's key range are not.
* A tier 2 kernel that is a scalar loop on a supported ISA without a catalog entry saying so. Gather on NEON and the bucketed reductions are expected strain cases, and they must be measured and recorded, not hidden.
* Results that differ between ISAs, which the base spec already rules out.

## Primitives

All index and key arguments are `i64[]`, as in the base spec.
The new kernel class "scan" covers primitives whose output row `i` depends on input rows `0` to `i`: the kernel carries a state across tiles, the way a reducer does, so tiling stays possible.

| Primitive | Class | Meaning |
|---|---|---|
| `scan(x, op)` | scan | Inclusive prefix: `out[i] = x[0] op x[1] op ... op x[i]`. |
| `scan_by(x, starts, op)` | scan | Segmented inclusive prefix. `starts` is `bool[]` of the same length, and a true row starts a new segment. Row 0 always starts a segment. |
| `prev(x, fill)` | scan | `out[0] = fill` and `out[i] = x[i - 1]`. Neighbour comparisons and differences use it, for example `x - prev(x, 0)`. |
| `gather(xs, idx)` | whole-array | `out[i] = xs[idx[i]]`. Any index outside `0..length(xs)` is a runtime error. |
| `histogram(keys, n)` | whole-array | `i64[]` of length `n`, where `out[k]` counts the rows with key `k`. Any key outside `0..n` is a runtime error. |
| `reduce_into(keys, vals, n, op)` | whole-array | `T[]` of length `n`, where `out[k]` is `reduce` with `op` over the values whose key is `k`. Buckets with no rows hold the identity of `op`. Same key range error as `histogram`. |
| `div_const(x, c)` | elementwise | Integer division by a scalar, truncating toward zero as Rust `/` does. `c == 0` is a runtime error. The kernel computes multiply-high plus shift constants once per call. |
| `argsort(x)` | whole-array | `i64[]` permutation that sorts `x` ascending, stable. Integer element types only in plan 4. |
| `slice(x, start, end)` | whole-array | Zero-copy view of rows `start` to `end - 1`. It requires `0 <= start <= end <= length(x)` and is otherwise a runtime error. |

Kernels cover the first version's element types where the operation is defined (my call):
* `gather`, `prev`, and `slice` move rows without arithmetic, so they cover `f64`, `f32`, `i64`, `i32`, `u8`, and `bool`.
* `scan`, `scan_by`, and `reduce_into` cover `i64`, `i32`, and `u8` with every `reduce` operator that has an integer kernel, and `f64` and `f32` with `min` and `max` only.
* `div_const` covers `i64`, `i32`, and `u8`.
* `argsort` covers `i64`, `i32`, and `u8`.
* `histogram` and the key arguments take `i64[]` only.

`scan`, `scan_by`, and `reduce_into` take the same operators as `reduce`.
`reduce_into` with `add` widens its result like `sum`, so `i32` values give `i64` buckets, and the empty-bucket identity of `min` and `max` is the type's largest and smallest value, with `+inf` and `-inf` for floats.
`histogram(keys, n)` is the same result as `reduce_into(keys, fill(length(keys), 1), n, add)`, and the planner may cover either form with either kernel through the usual pattern mechanism (my call).

`argsort` is in plan 4 only because ClickBench q07 sorts its handful of result groups for `ORDER BY`.
Plan 4 implements it with a scalar stable sort and records it in the catalog as a strain case, and tier 3 replaces it with sorting networks or a radix sort built from `histogram` and `scatter`.
Descending order and multi-key order are written by the user: `argsort(0 - c)` sorts descending, and a stable sort on the secondary key followed by a stable sort on the primary key gives a two-key order.

Group-by on sorted keys, which the catalog lists for tier 2, needs no primitive of its own.
It is `scan_by` over run starts found with `prev`, followed by `compress` on the run ends, and plan 4 includes a test program for it.

## Semantics

Integer scans and integer bucketed sums wrap, so every association order gives the same bits, and a kernel may use any order.
Float `min` and `max` follow IEEE `minimum` and `maximum`, which are associative and commutative, so they also allow any order, with NaN payloads free as in the base spec.

Float `add` and `mul` do not associate, so a vectorized scan or a bucketed sum needs a canonical order that every ISA computes.
Plan 4 rejects float `add` and `mul` in `scan`, `scan_by`, and `reduce_into` with the usual "no kernel" type error, and a later phase closes the gap (see "Deferred: surface gaps").
The rejection is temporary scope, not a statement that these combinations are out of the language, because a primitive that works for some operator and type pairs and not others is a wrinkle in the surface.

Results stay bit-identical rather than equal within a tolerance.
A tolerance would let results differ between ISAs, which the base spec's anti-goals rule out, and it would break the intended host: Differential Dataflow cancels a retraction against the original update only when both compute the same bits.
Bit-identity is also what makes fusion legal without per-kernel error analysis, since a fused kernel then matches the reference exactly.

Out-of-range indices and keys, a zero divisor, and invalid `slice` bounds are runtime errors, which the base spec's ordering rules allow to surface in any order.
`gather` on an empty `idx` returns an empty array, and `histogram` and `reduce_into` with `n = 0` and empty keys return empty arrays.

## Acceptance workloads

### Data

Corgi's repository (`github.com/frankmcsherry/WIP` at the revision the benches pin) ships `benches/clickbench/prepare.py` and `benches/tpch/prepare.py`.
They read ClickBench's `hits_0.parquet` (1M rows) or generate TPC-H data at a given scale factor through DuckDB, then write one raw file per column and DuckDB's answer for each query to `expected/<query>.txt`.
Integer columns are written as little-endian `u64` to `<column>.u64`, and string columns as `<column>.ends` (`u64` cumulative end offsets) plus `<column>.bytes`.
Verkehrt reads those files as they are: `load_i64` reads every unsigned column the acceptance queries use, and a string column whose values are all one byte long, such as TPC-H's `l_returnflag` and `l_linestatus`, is a `u8[]` in its `.bytes` file.
The repository has no license file, so Verkehrt does not copy its scripts or corgi programs, and `prepare.py` runs from a corgi checkout at the pinned revision (TPC-H at scale factor 0.2, as corgi's measurements use).
The expected answers come from the SQL in each corgi query file's `# sql:` header, which corgi made deterministic and rewrote to integers (prices in cents, discounts in percent, dates as day numbers, tie-breaking `ORDER BY` columns).
Each Verkehrt program transcribes that SQL into its own `// sql:` comment and takes its constants from it, because the SQL text belongs to the public benchmarks while the corgi programs below the header are corgi's own (my call).

The builder prepares the data, so acceptance queries can be debugged in a loop instead of handed off.
It clones corgi's repository at the pinned revision, downloads `hits_0.parquet` from `datasets.clickhouse.com`, and runs both `prepare.py` scripts with `uv run --with duckdb --with numpy`, so nothing is installed globally (DuckDB downloads its `tpch` extension on first use).
Everything lands in a data directory outside the repository that the user chose, which the plan names, and the measurement document records the corgi revision, the DuckDB version, and the file sizes.

A string predicate such as `SearchPhrase <> ''` is `(ends - prev(ends, 0)) != 0` over the `.ends` file.
None of the seven acceptance queries needs this, but plan 5's queries do, and plan 4 adds a test that shows it.

### Query programs

Each acceptance query is one Verkehrt program, at `tests/acceptance/<suite>/<query>.vk`, written as the SQL is.
Its first comment lines carry the SQL and the output kinds, in the same form as corgi's files (`// sql:` and `// output:` with corgi's kind letters `u`, `i`, `f`, `s`).
The program declares one `input` per column it reads, named after the column file, and exports one array per output column, in SQL column order.
Programs derive bucket counts from the data at run time, for example `reduce(keys, max) + 1`, and never hard-code a key range.

The seven queries map onto the primitives as follows.

| Query | Shape | Primitives beyond tier 1 |
|---|---|---|
| TPC-H Q1 | Filter on date, group by two one-byte flags (a few groups), four sums, a mean, a count, ordered by the keys | `histogram`, `reduce_into`, `div_const` (to split the combined key) |
| TPC-H Q6 | Filter chain, then one sum | none |
| ClickBench q01 | Count where `AdvEngineID <> 0` | none |
| ClickBench q02 | A sum, a count, a mean | none |
| ClickBench q06 | Min and max of `EventDate` | none |
| ClickBench q07 | Filter, group by `AdvEngineID`, count, order by count descending then key | `histogram`, `argsort` |
| ClickBench q42 | Filter chain, group by minute (`EventTime / 60 * 60`), order by minute, `OFFSET 1000 LIMIT 10` | `div_const`, `histogram`, `slice` |

A group key with a small range becomes a bucket index by subtracting its minimum, Q1 combines its two flags as `rf * 256 + ls`, and `where(counts > 0)` lists the non-empty buckets in key order.

### Harness

The harness is a Rust test, `tests/acceptance.rs`, and a criterion bench, `benches/acceptance.rs`.
Both read the data directory from `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH`, and both skip a suite with a message when its variable is unset, so `cargo test` stays green in CI.
The test prepares each program, binds the column files to its inputs, executes it, formats the exports row by row the way `prepare.py`'s `canonical` function does (integers as decimal, `f` as `{:.4f}`, `s` as hex of the bytes), and compares with `expected/<query>.txt` byte for byte.
The bench times `execute` only, with inputs loaded and the program prepared once, which is how the corgi bench in `benches/corgi.rs` times corgi.

CI checks the harness itself on small synthetic column files generated by a test, so the formatting and binding code is covered without the real data (my call).

### Comparison

Corgi's own `cargo bench --bench clickbench` and `--bench tpch`, run from the corgi checkout on the same data directory and in the same session as Verkehrt's bench, give corgi's times.
`prepare.py` records DuckDB's one-thread and all-thread times in `duckdb.tsv`, and the comparison uses the one-thread column, since Verkehrt is single-threaded.
The measurement document reports, per query and per ISA, the Verkehrt median, corgi's time, DuckDB's time, and the ratios, along with the machine setup in the form plans 2 and 3 used.
Corgi's numbers come from its own harness, so the document states how corgi times a query, as read from its bench source, before comparing.

## Kernel and catalog work

* Every new primitive gets a scalar reference kernel and a property test against it over the base spec's length and alignment cases, extended with key and index edge cases: the first and last bucket, all rows in one bucket, every out-of-range case, and empty inputs.
* The `auto` backend covers every new kernel on AVX2 and AVX-512, through the shared `auto_backend!` macro.
* Intrinsics candidates are `gather` (`vpgatherqq` against scalar loads, a measured strain case), `histogram` and `reduce_into` (conflict handling with `vpconflictq` on AVX-512, or replicated sub-histograms), and the integer `scan` (log-step shifts). Each is written only if the bench shows a gap.
* NEON work on the M5 follows the plan 3 pattern: a hand-off section in the plan, measurements by a session on the Mac, and NEON intrinsics only where measurement shows a gap.
* Fused kernels are not required. If a pattern such as `reduce_into(compress(keys, m), compress(vals, m), n, add)` dominates an acceptance query, the measurement document names it as a candidate, and adding it is the builder's call.
* The base spec's tier 2 catalog gains the new primitives' rows and the strain cases: gather on NEON, bucketed reductions, float scans and bucketed float sums, and the scalar `argsort`.
* `tier1.rs` gains native-baseline benches for `scan`, `gather`, `histogram`, and `reduce_into` at 10M rows, so each new kernel is measured against a plain Rust loop, as the tier 1 kernels were.

## Deferred: surface gaps

The user's rule is that Verkehrt's surface should not have many wrinkles: a primitive should accept every operator and element type for which it has a defined, ISA-independent result.
Plan 4 leaves gaps for scope, and each gets closed in a later phase or gets an explicit reason why it stays closed.
The base spec's catalog lists the open gaps, so the surface's holes are visible in one place.

* **Float `add` and `mul` in `reduce_into`.** A candidate canonical order extends the 16-accumulator reduction order: row `i` goes to stripe `i mod 16` below `b = n - (n mod 16)` and to stripe 0 from `b` on, each stripe accumulates its bucket's values in index order, and the 16 stripes combine pairwise as in the reduction. With every key equal, this gives exactly the bits of `sum`. The same 16 replicated sub-histograms are a standard way to avoid scatter conflicts, so the order may cost little beyond the memory for 16 copies of the buckets.
* **Float `add` and `mul` in `scan` and `scan_by`.** A candidate canonical order defines `out[i]` as the reduction order applied to the prefix `x[0..=i]`, so `out[n - 1]` equals `sum(x)`. A kernel can compute it by keeping the 16 stripe accumulators and combining them for each output row, which costs a pairwise tree per row and needs measurement against a sequential scan.
* **Float element types in `argsort`.** These need a total order that says where NaN and the two zeros sort, which tier 3 decides along with `sort`.
* **`reverse`, `rotate`, and packed bitmasks** from the tier 2 catalog.

## Success for plan 4

* All seven acceptance queries match DuckDB's expected output on `hits_0` and on TPC-H at scale factor 0.2.
* Every tier 2 primitive has scalar and `auto` entries that pass the bit-identity property tests on x86 and, through CI's macOS runners, on NEON.
* The measurement document gives Verkehrt, corgi, and DuckDB times for each acceptance query on the x86 host, and on the M5 after its session runs.
* No speed target gates plan 4, because these are the first measurements of grouping in Verkehrt. The document records how far each query is from DuckDB and corgi, and the first gap to close becomes input to plan 5.

## Left to the builder

* The internal representation of the scan class and its state.
* How `div_const` computes its constants, as long as results match Rust `/` for every input.
* The bucket-conflict strategy inside `histogram` and `reduce_into`.
* Whether `slice` needs a kernel or is a view operation in the executor.
* Exact names of the test and bench helpers.
