<p align="center"><img src="docs/logo.svg" width="120" alt="verkehrt"></p>

<h1 align="center">Verkehrt</h1>

<p align="center">An interpreted array language whose primitives are SIMD kernels, selected by equality saturation over kernel patterns.</p>

## The idea

Verkehrt is an array language built backwards from the machine, which is what the German word means: reversed, inverted, the wrong way round.
Instead of compiling arbitrary programs and hoping for good code, it offers only primitives that map onto good SIMD instruction sequences.
Each primitive is a kernel, written by hand with intrinsics or compiled from portable Rust under each instruction set's target features.
The interpreter only dispatches between kernels over whole arrays, so the dispatch cost amortizes over the data, and it never generates machine code.
The primitive set is the object of study, because it maps what a user can express that also runs well.

Kernels are patterns, not functions the user calls by name.
Each kernel declares the expression it implements in the language's own primitives, for example `dot: sum(?x * ?y)` on `f64[]`.
The interpreter builds an e-graph of each statement with [egglog](https://github.com/egraphs-good/egglog), applies rewrites that preserve every result bit, and covers the e-graph with kernel patterns at minimum cost.
This is equality saturation in reverse: the extraction targets are a fixed library of kernels, so `sum(x * y)` runs a fused dot-product kernel, and there is no user-facing `dot`.

```mermaid
flowchart LR
  src[Source] --> parse[Parse] --> check[Type check] --> lower[Lower]
  lower --> egglog[egglog: rewrite and cover] --> plan[Plan] --> exec[Execute]
  exec -->|kernel call| table[Kernel table]
```

Lowering inlines functions and single-use `let` bindings, and it lifts lambdas into whole-array operations, so the e-graph only sees first-order array terms.
A function passed to `map` or `filter` is never called once per element: `|v| (v * 2.0) + 1.0` becomes two array operations, and `if` inside a lambda becomes a select.
A program that cannot be lifted to whole arrays is rejected with an error, and all type errors are reported before any kernel runs.
Because floats use one canonical summation order and no kernel uses fused multiply-add, a fused kernel computes the same bits as the unfused primitives, which is what makes fusion legal.

## Example

The program below matches the style of the test programs in `tests/programs/`.
It generates two random arrays, then computes a dot product, a count, and a filter over them.
The intermediate `t` has one use, so lowering inlines it into `sum`.
The lambda passed to `filter` is lifted into a whole-array comparison, so the planner sees `compress(x, x > 0.99)`.
Running it with `--plan` and `--trace` shows how the planner covered each statement.

```
let x = rand_f64(1000000, 1);
let y = rand_f64(1000000, 2);
let t = x * y;
print(sum(t));
print(count(x > 0.5));
let big = filter(x, |v| v > 0.99);
print(length(big));
```

`--plan` prints the cover chosen for each statement, labeled by source line, and `--trace` prints one line per kernel call to stderr.
Statements 2, 3, and 6 have no plan of their own, because their bindings are inlined into their single consumers.
`x` has three uses, so it is materialized once and stays a leaf for each consumer.
In the output below, trace lines on stderr interleave with the printed results on stdout.

```
$ cargo run --release -- run example.vk --plan --trace
plan 1: (rand_f64 1000000 1)
plan 4: (dot_f64 x (rand_f64 1000000 2))
plan 5: (count_gt_f64 x 0.5)
plan 7: (length_f64 (filter_gt_f64 x 0.99))
trace rand_f64 isa=scalar n=0 read=0 write=8000000 time=2761.9 gbps=2.90
trace rand_f64 isa=scalar n=0 read=0 write=8000000 time=2645.6 gbps=3.02
trace dot_f64 isa=avx512 backend=auto n=1000000 read=16000000 write=0 time=284.4 gbps=56.27
249724.39002046228
trace count_gt_f64 isa=avx512 backend=auto n=1000000 read=8000000 write=0 time=155.3 gbps=51.51
499521
trace filter_gt_f64 isa=avx512 backend=intrinsics n=1000000 read=8000000 write=78808 time=416.3 gbps=19.41
trace length_f64 isa=scalar n=9851 read=78808 write=0 time=0.7 gbps=107.96
9851
```

The plan shows three fused kernels the program never names: `dot_f64` for `sum(x * y)`, `count_gt_f64` for `count(x > 0.5)`, and `filter_gt_f64` for the lifted filter.
Each trace line gives the kernel, the variant that ran, the element count, bytes read and written, the time in microseconds, and the throughput in GB/s.
The output above comes from a single run on an AVX-512 machine, so the times are illustrative only.
`--stats` prints per-kernel totals at exit, and `verkehrt repl` starts an interactive session with the same flags.

## Backends

Every kernel has a scalar reference entry, and each instruction set adds up to two backends on top of it.
The `auto` backend compiles the kernel's portable body under the instruction set's target features and covers every kernel.
The `intrinsics` backend holds hand-written kernels and exists only where one was written, currently compress, `where`, and the fused filter.
`--backend auto|intrinsics` picks the backend, and the default `intrinsics` falls back to `auto`, then to scalar, so every hand-written kernel is measured against the compiler's output for the same semantics.
`--isa scalar|avx2|avx512|neon` forces an instruction set, otherwise the interpreter detects the best one at startup.

| Architecture | Instruction set | Status |
|---|---|---|
| x86-64 | AVX2 | `auto` and `intrinsics` |
| x86-64 | AVX-512 (F, BW, VL, VBMI2) | `auto` and `intrinsics` |
| aarch64 | NEON | In progress, kernels run their scalar entries until NEON entries land |

Every variant must return the scalar kernel's result bit for bit, and a NaN result only has to be NaN.
The property test `tests/isa_variants.rs` checks this for every installed variant, over aligned and misaligned views and lengths around the vector widths.
CI runs the test suite on Linux and on macOS runners, which are Apple Silicon, so the NEON entries will be checked there.
The benchmarks then show whether each variant is also fast.

## Results

The measurements compare the interpreter against plain Rust loops compiled for the same instruction set, at 10 million `f64` elements.
For `sum`, `sum(x * y)`, and `count(x > 0.5)`, the AVX2 and AVX-512 variants run within 1.04x of the native loop.
For `filter(x, |v| v > c)` with the default `intrinsics` backend, the largest ratio is 1.40x, for AVX2 at 99% selectivity, and the `auto` backend reaches 1.51x at 50% selectivity, above the spec's 1.5x target.
Each call has a fixed cost of roughly 270 ns, so the interpreter stays within 1.5x of native from about 2^17 elements for `sum` and 2^13 elements for `x * y`, good to about one power of two.

These numbers come from one AMD Ryzen AI MAX+ 395 with boost off and other work running, and most rows are single runs.
The clock moved between 2.0 and 3.0 GHz under load, so absolute throughput is below what the CPU reaches with boost on, while ratios share the same conditions.
The `auto` and baseline medians drifted by up to 1.36x between builds of unchanged code, and the cause was not measured.
Read [docs/measurements/2026-10-30-x86-kernels.md](docs/measurements/2026-10-30-x86-kernels.md) for the full tables, the setup, and the caveats before quoting any number.

## Building and testing

Verkehrt is a Rust crate with edition 2024 and a minimum supported Rust version of 1.95.
The interpreter is a library, and the `verkehrt` binary is a thin command-line tool on top of it.
The binary has two subcommands, `run` for a program file and `repl` for an interactive session.
Release builds use LTO, so they take longer than debug builds.

```
CARGO_INCREMENTAL=0 cargo build --release
CARGO_INCREMENTAL=0 cargo run --release -- run tests/programs/dot.vk
CARGO_INCREMENTAL=0 cargo run --release -- repl
```

The test suite covers the parser, type checker, planner, kernels, and the example programs in `tests/programs/`.
The ISA variant test is worth running in release mode too, since the `auto` backend depends on what the optimizer vectorizes.
The benchmarks compare against native loops and corgi, which they pull from Git as a dev-dependency.

```
CARGO_INCREMENTAL=0 cargo test
CARGO_INCREMENTAL=0 cargo test --release --test isa_variants
CARGO_INCREMENTAL=0 cargo bench --locked --bench tier1
CARGO_INCREMENTAL=0 cargo bench --locked --bench corgi
```

## Design documents

* [Design](docs/superpowers/specs/2026-10-30-verkehrt-design.md): intent, semantics, kernel selection, and the problem catalog.
* [Core interpreter plan](docs/superpowers/plans/2026-10-30-verkehrt-core.md): parser, type checker, lowering, egglog planning, and scalar kernels.
* [x86 kernels plan](docs/superpowers/plans/2026-10-30-verkehrt-kernels-x86.md): AVX2 and AVX-512 kernels in both backends, benchmarks, and crossover measurements.
* [x86 measurements](docs/measurements/2026-10-30-x86-kernels.md): results, setup, and caveats.

## Prior art

[Corgi](https://github.com/frankmcsherry/WIP) is Frank McSherry's columnar interpreter, and the benchmarks run the same workloads in corgi at the revision that DDIR pins.
Verkehrt's embedding API prepares a program once and executes it many times, as DDIR does with corgi graphs.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
