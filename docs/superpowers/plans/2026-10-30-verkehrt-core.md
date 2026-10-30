# Verkehrt core interpreter implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Verkehrt interpreter (library plus `verkehrt run` and `verkehrt repl`) that parses, type-checks, lifts, plans with egglog kernel patterns, and executes tier 1 programs on scalar reference kernels, including the fused tier 1 kernels.

**Architecture:** Source goes through lexer, parser, type checker, and lowering (function inlining, lambda lifting, broadcast insertion, single-use `let` inlining, CSE hoisting) into first-order terms. Each statement's term is sent to egglog as text, where commutativity, comparison flipping, and fused-kernel rules run, and the extracted term is parsed back into a plan of kernel calls. The executor walks plans and dispatches through a kernel registry that is also the type checker's source of truth.

**Tech stack:** Rust (edition 2024), `egglog` 3.x, `bytemuck` 1.x (safe aligned buffers), `clap` 4.x, `rustyline` 19.x, `proptest` 1.x and `assert_cmd` 2.x for tests.

**Spec:** `docs/superpowers/specs/2026-10-30-verkehrt-design.md`

**Plan series:** This is plan 1 of 3. Plan 2 adds AVX2 and AVX-512 kernels, kernel property tests against the scalar reference, benchmarks against native Rust baselines, and crossover measurements. Plan 3 is the NEON batch with a Mac hand-off. This plan ships only scalar kernels, so `--isa avx2` falls back to scalar and the trace says so.

## Global constraints

* Edition 2024, one package with `src/lib.rs` (library) and `src/main.rs` (thin CLI). Build and test with `CARGO_INCREMENTAL=0`.
* No `unsafe` in this plan. Aligned buffers use `bytemuck` casts over a `#[repr(C, align(64))]` block type.
* No global mutable state. `Interpreter` must be `Send` (asserted by a test).
* Array kernels exist for `f64`, `f32`, `i64`, `i32`, `u8`, and `bool` only. Scalar-scalar kernels exist for every scalar type (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`), so values such as the `u64` result of `sum(u8[])` stay usable (my call, cheap because scalar kernels are plain Rust).
* Integer arithmetic wraps. No integer division. Casts follow Rust `as`.
* Float `sum`, `reduce(add)`, and `reduce(mul)` use the canonical 16-accumulator order from the spec ("Semantics decisions"). No kernel uses FMA (no `mul_add`, and no reliance on the compiler contracting, which Rust does not do by default).
* Float `min`/`max` are IEEE 754-2019 `minimum`/`maximum`: NaN propagates, `-0.0 < +0.0`.
* Reductions widen: `sum`/`reduce(add)` over `i32` return `i64`, over `u8` return `u64`. Other reductions keep the element type.
* `rand_i64(n, seed)` element `i` is `splitmix64(seed.wrapping_add(i))` as `i64`. `rand_f64` maps `(bits >> 11) as f64 * 2^-53`. SplitMix64 constants: add `0x9E3779B97F4A7C15`, multiply `0xBF58476D1CE4E5B9` after `^ >> 30`, multiply `0x94D049BB133111EB` after `^ >> 27`, finish with `^ >> 31`.
* Indices, counts, lengths are `i64`.
* Errors carry a source location (line and column). Runtime errors carry the span of the statement that failed.
* Comments: no em-dashes, no structuring semicolons. Doc comments state contracts, reasoning goes inline.
* Deferred from the spec, by design of this plan: tuples and `let (a, b)` (no v1 primitive returns a tuple), output-buffer reuse at reference count 1 (plan 2, it only matters for SIMD throughput), near-miss reporting (spec defers it).

## Review focus

* Empty arrays flowing through every primitive (an all-false filter, then `sum`, `argmin`, `print`) must give identity results, the specified errors, or `[]`, never a panic. Test in Task 4.
* Arrays of different lengths in one elementwise operation must give a runtime error naming the statement's line, not a panic. Test in Task 3 (kernel) and Task 9 (location).
* NaN, `-0.0`, and infinities through `min`, `max`, `argmin`, `sum`, and float-to-int casts must follow the spec. Tests in Tasks 3 and 4.
* Common mistakes (`x * 2` on `f64[]`, `a + b * c`, `a < b < c`, an unknown name, a lambda capturing an array) must give a located error before anything runs. Tests in Tasks 2, 5, and 6.
* `load_f64` on a missing file or a file whose size is not a multiple of 8 must give a runtime error, not a panic. Test in Task 4.

## File structure

| File | Responsibility |
|---|---|
| `src/lib.rs` | Module declarations, re-exports of `Interpreter`, `Options`, `Isa`, `Value`, `Error`. |
| `src/error.rs` | `Error`, `ErrorKind`, `Span`. |
| `src/value.rs` | Scalar and value types, aligned buffers, array views. |
| `src/syntax/lexer.rs`, `src/syntax/ast.rs`, `src/syntax/parser.rs` | Tokens, surface AST, recursive-descent parser. |
| `src/ops.rs` | `Op` and `ReduceOp`, the first-order operation vocabulary shared by checker, lowering, and registry. |
| `src/kernels/mod.rs` | `Registry`, `KernelDef`, `KernelId`, `InputSpec`, `Isa`, `KernelFn`. |
| `src/kernels/scalar/elementwise.rs` | Single-op elementwise, comparison, logic, select, cast kernels. |
| `src/kernels/scalar/reduce.rs` | Canonical sum and product, reductions, count/any/all, argmin/argmax. |
| `src/kernels/scalar/select.rs` | `compress`, `where`. |
| `src/kernels/scalar/construct.rs` | `iota`, `fill`, `rand_*`, `load_*`, `length`. |
| `src/kernels/scalar/fused.rs` | Fused kernels and their egglog patterns. |
| `src/check/mod.rs`, `src/check/typed.rs` | Type checker and the typed AST it produces. |
| `src/lower.rs` | Inlining, lifting, broadcast insertion, `let` inlining, CSE, kernel-existence check. |
| `src/plan/mod.rs`, `src/plan/egglog.rs`, `src/plan/sexpr.rs` | Plan types, egglog program emission and extraction, s-expression parsing. |
| `src/exec.rs` | Plan executor, trace, stats. |
| `src/interp.rs` | `Interpreter` and `Options`, the library entry point. |
| `src/main.rs` | CLI. |
| `tests/programs/*.vk` and `tests/programs.rs` | Golden-output program tests. |
| `tests/cli.rs` | CLI tests. |

---

### Task 1: Crate, errors, and value model

**Files:**
* Create: `Cargo.toml`, `.gitignore` (`/target`), `src/lib.rs`, `src/error.rs`, `src/value.rs`, `src/main.rs` (stub `fn main() {}`)
* Test: inline `#[cfg(test)] mod tests` in `src/value.rs`

**Interfaces:**
* Produces:
  * `pub struct Span { pub line: u32, pub col: u32 }`
  * `pub enum ErrorKind { Parse, Type, Runtime, Io, Usage }`, `pub struct Error { pub kind: ErrorKind, pub message: String, pub span: Option<Span> }`, `Display` as `error[type] at 3:14: message` (omit ` at ..` without span). Constructors `Error::parse(msg, span)`, `Error::ty(msg, span)`, `Error::runtime(msg, Option<Span>)`.
  * `pub enum ScalarType { I8, I16, I32, I64, U8, U16, U32, U64, F32, F64, Bool }` with `fn width(self) -> usize` (bool is 1), `fn name(self) -> &'static str` (`"f64"` etc.), `fn is_float`, `fn is_int`, `fn is_signed`, `fn from_name(&str) -> Option<Self>`, `const V1_ARRAY: [ScalarType; 6]`.
  * `pub enum ValueType { Scalar(ScalarType), Array(ScalarType), Str }`, `Display` as `f64`, `f64[]`, `str`.
  * `pub enum Scalar { I8(i8), ..., F64(f64), Bool(bool) }` with `fn ty(&self) -> ScalarType`.
  * `pub trait Elem: bytemuck::Pod + Copy + Send + Sync + 'static { const TYPE: ScalarType; }` for every numeric Rust type. Bool arrays are stored as `u8` with values 0 and 1.
  * `pub struct Array` (view: `Arc` buffer, element offset, length, `ScalarType`), with `fn elem(&self)`, `fn len(&self)`, `fn byte_len(&self)`, `fn slice(&self, start: usize, len: usize) -> Array` (zero-copy, panics out of range), `fn as_slice<T: Elem>(&self) -> &[T]` (panics on type mismatch), `fn as_bool_bytes(&self) -> &[u8]` (panics unless `Bool`), `fn from_vec<T: Elem>(v: Vec<T>) -> Array`, `fn from_bools(v: &[bool]) -> Array`, `fn build<T: Elem>(len: usize, f: impl FnOnce(&mut [T])) -> Array`, `fn build_bool(len: usize, f: impl FnOnce(&mut [u8])) -> Array`, `fn get(&self, i: usize) -> Scalar`.
  * `pub enum Value { Scalar(Scalar), Array(Array), Str(Arc<str>) }` with `fn ty(&self) -> ValueType` and `Display` (format below).

- [ ] **Step 1: Create the crate.** `cargo init --lib --name verkehrt`, add `src/main.rs`, then `cargo add bytemuck --features derive`, `cargo add egglog@3`, `cargo add clap --features derive`, `cargo add rustyline`, `cargo add --dev proptest assert_cmd`.

- [ ] **Step 2: Write the failing tests** in `src/value.rs`:

```rust
#[test]
fn buffers_are_64_byte_aligned() {
    for len in [0usize, 1, 7, 100] {
        let a = Array::from_vec((0..len as i64).collect::<Vec<_>>());
        assert_eq!(a.as_slice::<i64>().as_ptr() as usize % 64, 0);
    }
}

#[test]
fn slice_is_zero_copy_view() {
    let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
    let s = a.slice(1, 2);
    assert_eq!(s.as_slice::<f64>(), &[2.0, 3.0]);
    assert_eq!(s.as_slice::<f64>().as_ptr(), a.as_slice::<f64>()[1..].as_ptr());
}

#[test]
fn bools_are_bytes() {
    let a = Array::from_bools(&[true, false, true]);
    assert_eq!(a.as_bool_bytes(), &[1, 0, 1]);
    assert_eq!(a.ty(), ValueType::Array(ScalarType::Bool));
}

#[test]
fn display_formats() {
    assert_eq!(Value::Scalar(Scalar::F64(1.0)).to_string(), "1.0");
    assert_eq!(Value::Scalar(Scalar::Bool(true)).to_string(), "true");
    assert_eq!(Value::Array(Array::from_vec(vec![1i64, 2])).to_string(), "[1, 2]");
    assert_eq!(Value::Array(Array::from_vec(Vec::<f64>::new())).to_string(), "[]");
    let long = Value::Array(Array::from_vec((0..25i64).collect::<Vec<_>>()));
    assert_eq!(long.to_string(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... 15 more]");
}
```

(`Array::ty` is shorthand used here: add `fn ty(&self) -> ValueType` on `Array` too.)

- [ ] **Step 3: Run** `CARGO_INCREMENTAL=0 cargo test --lib value`. Expected: compile failure, types missing.

- [ ] **Step 4: Implement** `src/error.rs` and `src/value.rs` per the interfaces. Storage: `Vec<Block>` with `#[derive(Clone, Copy, Pod, Zeroable)] #[repr(C, align(64))] struct Block([u8; 64])`, typed access through `bytemuck::cast_slice`. Float display uses `{:?}` (so `1.0`, `NaN`, `-0.0`, `inf`). Arrays with at most 20 elements print fully, longer ones print the first 10 then `... N more`.

- [ ] **Step 5: Run** the tests. Expected: PASS.

- [ ] **Step 6: Commit** `git add Cargo.toml Cargo.lock .gitignore src && git commit -m "Add value model with aligned array views"`

---

### Task 2: Lexer, AST, and parser

**Files:**
* Create: `src/syntax/mod.rs`, `src/syntax/lexer.rs`, `src/syntax/ast.rs`, `src/syntax/parser.rs`
* Test: `src/syntax/tests.rs` (declared `#[cfg(test)] mod tests;` in `src/syntax/mod.rs`)

**Interfaces:**
* Consumes: `Span`, `Error`, `Scalar`, `ScalarType`.
* Produces:
  * `pub fn parse(src: &str) -> Result<ast::Program, Error>`
  * `ast::Program { pub items: Vec<Item> }`, `enum Item { Fn(FnDef), Let(LetStmt), Print(Expr, Span), Expr(Expr, Span) }`
  * `FnDef { name: String, params: Vec<(String, TypeExpr)>, ret: TypeExpr, body: Block, span: Span }`
  * `LetStmt { name: String, ty: Option<TypeExpr>, value: Expr, span: Span }`
  * `TypeExpr { elem: ScalarType, array: bool }`
  * `Block { lets: Vec<LetStmt>, result: Box<Expr> }`
  * `Expr { kind: ExprKind, span: Span }`, `enum ExprKind { Lit(Scalar), Str(String), ArrayLit(Vec<Expr>), Name(String), Call(String, Vec<Expr>), Binary(BinOp, Box<Expr>, Box<Expr>), If(Box<Expr>, Block, Block), Lambda(Vec<String>, Box<Expr>), Block(Block) }`
  * `enum BinOp { Add, Sub, Mul, Div, Eq, Ne, Lt, Le, Gt, Ge }`

Grammar decisions (my calls where the spec is silent):
* Statements end with `;`. `print(expr);` is a statement, not an expression. Function bodies are blocks `{ let a = ...; ...; expr }`.
* Integer literals default to `i64`, float literals to `f64`. Suffixes pick other types: `1i32`, `7u8`, `2.0f32`, and so on for every scalar type. `true`/`false` are `bool`.
* No unary minus operator. A `-` immediately followed by a numeric literal in operand position is a negative literal (`-1.0`). `neg(x)` negates an expression.
* `and`, `or`, `not`, `min`, `max`, and the rest are ordinary calls, not infix operators.
* A sequence of the same arithmetic operator folds left. Mixing two different binary operators without parentheses, or any chain of two comparisons, is a parse error whose message tells the user to add parentheses.
* `b"` is reserved: lexing it is a parse error ("byte strings are not supported yet").

- [ ] **Step 1: Write the failing tests** in `src/syntax/tests.rs`. Cases, each asserting the AST shape (helper `fn expr(src) -> ExprKind` parsing `src` as a single expression statement) or the error kind and message fragment:
  * `a + b + c` parses as `Binary(Add, Binary(Add, a, b), c)`.
  * `a - b - c` parses left-nested.
  * `a + b * c` fails with `ErrorKind::Parse` at `1:7`, message contains `parentheses`.
  * `a < b < c` fails with `ErrorKind::Parse`, message contains `parentheses`.
  * `a + (b * c)` parses.
  * `x - 1.0` is `Binary(Sub, x, Lit(F64(1.0)))`, and `f(-1.0)` has argument `Lit(F64(-1.0))`.
  * `2.0f32`, `7u8`, `1i32` give `Scalar::F32(2.0)`, `Scalar::U8(7)`, `Scalar::I32(1)`.
  * `filter(x, |v| v > 0.0)` gives `Call("filter", [Name x, Lambda(["v"], Binary(Gt, ...))])`.
  * `if c { 1.0 } else { 2.0 }` gives `If`.
  * Whole program: `fn dot(x: f64[], y: f64[]) -> f64 { sum(x * y) }\nlet d = dot(a, b);\nprint(d);` gives three items with the right kinds and the `FnDef` param types `f64[]`.
  * `let x: f64[] = load_f64("x.bin");` gives `LetStmt` with `ty` `f64[]` and a `Str` argument.
  * `b"abc"` fails with `ErrorKind::Parse`.
  * Missing `;` reports the line and column of the next token.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --lib syntax`. Expected: FAIL to compile.

- [ ] **Step 3: Implement** lexer (tokens with `Span`), AST, and recursive-descent parser. Binary expressions: parse one operand, then if a binary operator follows, collect operands while the same operator repeats, and error if a different operator follows.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/syntax src/lib.rs && git commit -m "Add lexer and parser with explicit operator rules"`

---

### Task 3: Kernel registry and single-op elementwise kernels

**Files:**
* Create: `src/ops.rs`, `src/kernels/mod.rs`, `src/kernels/scalar/mod.rs`, `src/kernels/scalar/elementwise.rs`
* Test: `src/kernels/tests.rs`

**Interfaces:**
* Consumes: `Value`, `Array`, `Scalar`, `ScalarType`, `ValueType`.
* Produces:
  * `pub enum Op { Add, Sub, Mul, Div, Neg, Abs, Min, Max, Sqrt, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Not, Select, Cast(ScalarType), Reduce(ReduceOp), Count, Any, All, ArgMin, ArgMax, Compress, Where, Iota, Fill, RandI64, RandF64, Load(ScalarType), Length }`, `pub enum ReduceOp { Add, Mul, Min, Max, And, Or, Xor }` (both `Copy, Eq, Hash, Debug`).
  * `pub enum Isa { Scalar, Avx2, Avx512, Neon }`, `Isa::detect() -> Isa` (best available), `Isa::is_available(self) -> bool`, `Isa::from_name(&str) -> Option<Isa>`, `fn name(self)`.
  * `pub type KernelFn = fn(&[Value]) -> Result<Value, String>` (the `String` is the runtime error message, the executor adds the span).
  * `pub enum InputSpec { Array(ScalarType), Scalar(ScalarType), Either(ScalarType), Str }`. `Either` accepts an array or a scalar, used for elementwise operands, where at least one operand must be an array for an array kernel.
  * `pub struct KernelDef { pub name: String, pub op: Option<Op>, pub inputs: Vec<InputSpec>, pub output: ValueType, pub cost: u64, pub commutative: bool, pub pattern: Option<FusedPattern>, pub impls: [Option<KernelFn>; 4] }` (`impls` indexed by `Isa as usize`, `Scalar` always `Some`). `FusedPattern` is defined in Task 7. Until then declare `pub type FusedPattern = ();` and leave every `pattern` as `None`.
  * `pub struct KernelId(pub u32)`, `pub struct Registry`, `Registry::new() -> Registry` (builds all kernels, no global state), `fn lookup(&self, op: Op, args: &[ValueType]) -> Option<KernelId>`, `fn get(&self, id: KernelId) -> &KernelDef`, `fn by_name(&self, name: &str) -> Option<KernelId>`, `fn iter(&self) -> impl Iterator<Item = (KernelId, &KernelDef)>`, `fn select_impl(&self, id: KernelId, isa: Isa) -> (KernelFn, Isa)` (falls back to scalar, returns the variant actually used).
* Naming: single-op kernel names are `{op}_{type}` (`add_f64`, `gt_f64`, `select_f64`, `not_bool`), with scalar-only kernels named `{op}_{type}_s` (`add_u64_s`) and casts `cast_{from}_{to}` (`cast_i32_f64`). Names must be valid egglog identifiers.
* Lookup rule: an operation whose argument types include an array resolves to the array kernel (only for the v1 array types), an all-scalar operation resolves to the `_s` kernel (every scalar type).
* Cost (bytes per element, all operands treated as arrays): binary elementwise `2w + w_out`, unary `w + w_out`, select `1 + 2w + w`, scalar-only kernels `0`.

Type rules for this task: `Add Sub Mul Min Max` numeric, `Div Sqrt` float, `Neg Abs` signed int and float, `Eq Ne` all types, `Lt Le Gt Ge` numeric (output `bool` of the same shape), `And Or Not` bool, `Select(bool, T, T)` all types, `Cast(to)` from any numeric or bool to any numeric (no cast to bool). Array casts exist between the six v1 types.

- [ ] **Step 1: Write the failing tests** in `src/kernels/tests.rs`, calling kernels through `Registry::new()`, `lookup`, and the scalar impl:
  * `lookup(Add, [f64[], f64[]])` is `add_f64`, `lookup(Add, [f64[], f64])` is also `add_f64`, `lookup(Add, [u64, u64])` is `add_u64_s`, `lookup(Add, [i8[], i8[]])` is `None`, `lookup(Add, [f64[], f32[]])` is `None`, `lookup(Div, [i64[], i64[]])` is `None`, `lookup(Neg, [u8[]])` is `None`.
  * `add_f64([1.0, 2.0], 10.0)` is `[11.0, 12.0]`, and with the scalar first is the same.
  * `add_f64` on lengths 2 and 3 returns `Err` whose message contains `length mismatch` and both lengths.
  * `add_i32([i32::MAX], [1])` is `[i32::MIN]` (wraps), `add_u8([255], [1])` is `[0]`.
  * `min_f64([NaN, 1.0, -0.0], [1.0, NaN, 0.0])` is `[NaN, NaN, -0.0]` (check `-0.0` with `to_bits`), `max_f64([-0.0], [0.0])` is `[0.0]` with positive sign.
  * `abs_i64([i64::MIN])` is `[i64::MIN]`.
  * `cast_f64_i32([NaN, 1e10, -1e10, -1.9])` is `[0, i32::MAX, i32::MIN, -1]`, `cast_i64_i32([1 << 32 + 5])` wraps to `[5]`, `cast_bool_i64([true, false])` is `[1, 0]`.
  * `gt_f64([1.0, NaN], 0.0)` is `[true, false]`, `select_f64([true, false], [1.0, 2.0], [3.0, 4.0])` is `[1.0, 4.0]`, `select_f64` with a broadcast else `0.0` works.
  * Every `KernelDef` name is unique and matches `^[a-z][a-z0-9_]*$`.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --lib kernels`. Expected: FAIL to compile.

- [ ] **Step 3: Implement** `src/ops.rs`, `src/kernels/mod.rs`, and `src/kernels/scalar/elementwise.rs`. Generate per-type kernels with `macro_rules!` over a generic `fn binary<T: Elem, U: Elem>(args, f: impl Fn(T, T) -> U)` helper that handles the array-array, array-scalar, and scalar-array forms and the length check. `Isa::detect` uses `is_x86_feature_detected!("avx512f")` (plus `avx512bw`, `avx512vl`, `avx512vbmi2`) and `"avx2"` under `cfg(target_arch = "x86_64")`, and `Neon` under `cfg(target_arch = "aarch64")`.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/ops.rs src/kernels src/lib.rs && git commit -m "Add kernel registry and scalar elementwise kernels"`

---

### Task 4: Reduction, selection, and construction kernels

**Files:**
* Create: `src/kernels/scalar/reduce.rs`, `src/kernels/scalar/select.rs`, `src/kernels/scalar/construct.rs`
* Modify: `src/kernels/mod.rs` (register them), `src/kernels/tests.rs`

**Interfaces:**
* Consumes: Task 3 registry types.
* Produces:
  * `pub fn canonical_sum<T: Float>(xs: &[T]) -> T` and `pub fn canonical_product<T: Float>(xs: &[T]) -> T` in `reduce.rs`, implementing the spec's order exactly (identity `0.0` resp. `1.0` per accumulator). Plan 2 SIMD kernels are tested against these. `Float` is a small local trait implemented for `f32` and `f64`.
  * `pub fn splitmix64(x: u64) -> u64` in `construct.rs`.
  * Kernels: `reduce_{add,mul,min,max}_{t}` (numeric), `reduce_{and,or,xor}_bool`, `count`, `any`, `all` (on `bool[]`, `count` returns `i64`), `argmin_{t}`, `argmax_{t}`, `compress_{t}` (`T[]`, `bool[]`), `where` (`bool[]` to `i64[]`), `iota` (`i64` to `i64[]`), `fill_{t}` (`i64`, `T`), `rand_i64`, `rand_f64`, `load_{t}` (`str` to `T[]`, numeric v1 types), `length_{t}`.
  * Costs: reductions `w`, `count`/`any`/`all` `1`, argmin/argmax `w`, `compress` `2w + 1`, `where` `9`, `iota`/`rand_*` `8`, `fill` `w`, `load` `w`, `length` `0`.

- [ ] **Step 1: Write the failing tests:**
  * `canonical_sum` on 35 elements equals a hand-written reference that follows the spec's bullets literally (accumulators `acc[j]` over `j, j+16`, below `b = 32`, pairwise `+8, +4, +2, +1` combine, then elements 32..35 added to `acc[0]`), compared with `to_bits`. Use values like `1e16, 1.0, -1e16, ...` so the order changes the result, and assert that a naive left fold gives a different value for the same input.
  * `canonical_sum(&[])` is `0.0`, `canonical_product(&[])` is `1.0`.
  * `reduce_add_i32([i32::MAX, 1])` is `Scalar::I64(2147483648)`, `reduce_add_u8([200, 100])` is `Scalar::U64(300)`, `reduce_add_i64([i64::MAX, 1])` wraps to `i64::MIN`.
  * `reduce_min_f64([1.0, NaN, 0.0])` is NaN, `reduce_min_f64([0.0, -0.0])` is `-0.0` by bits, `reduce_min_f64([])` is `Err` containing `empty`.
  * `argmin_f64([3.0, 1.0, 1.0])` is `1`, `argmin_f64([3.0, NaN, NaN])` is `1`, `argmin_f64([0.0, -0.0])` is `1`, `argmax_f64([])` is `Err`.
  * `count([true, false, true])` is `2`, `any([])` is `false`, `all([])` is `true`, `reduce_and_bool([])` is `true`, `reduce_xor_bool([true, true])` is `false`.
  * `compress_f64([1.0, 2.0, 3.0], [false, true, true])` is `[2.0, 3.0]`, an all-false mask gives `[]`, a length mismatch is `Err`. `where([false, true, true])` is `[1, 2]`.
  * `iota(3)` is `[0, 1, 2]`, `iota(-1)` is `Err`, `fill_f64(2, 1.5)` is `[1.5, 1.5]`.
  * `splitmix64(0)` is `0xE220A8397B1DCDAF`. `rand_i64(3, 7)` element 1 equals `splitmix64(8) as i64`. `rand_f64(1000, 1)` values all lie in `[0, 1)`.
  * `load_f64` reads a temp file written from `[1.0, 2.5]` as little-endian bytes. A missing file is `Err` containing the path. A 9-byte file is `Err` containing `multiple of 8`.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --lib kernels`. Expected: the new tests fail.

- [ ] **Step 3: Implement** the three files and register the kernels.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/kernels && git commit -m "Add reduction, selection, and construction kernels"`

---

### Task 5: Type checker

**Files:**
* Create: `src/check/mod.rs`, `src/check/typed.rs`
* Test: `src/check/tests.rs`

**Interfaces:**
* Consumes: `ast::Program`, `Registry::lookup`, `Op`, `ReduceOp`.
* Produces:
  * `pub fn check(prog: &ast::Program, reg: &Registry, env: &mut Globals) -> Result<TProgram, Error>`. `Globals` holds top-level bindings (name to `VarId` and `ValueType`) and function definitions, and persists across REPL inputs.
  * `pub struct VarId(pub u32)`, `pub struct FnId(pub u32)`.
  * `TProgram { pub stmts: Vec<TStmt> }` (functions live in `Globals`), `pub struct TFn { pub name: String, pub params: Vec<(VarId, ValueType)>, pub ret: ValueType, pub body: TExpr }`.
  * `enum TStmt { Let { var: VarId, value: TExpr, span: Span }, Print { value: TExpr, span: Span }, Expr { value: TExpr, span: Span } }`.
  * `TExpr { pub kind: TExprKind, pub ty: ValueType, pub span: Span }`, `enum TExprKind { Lit(Scalar), Str(Arc<str>), ArrayLit(Vec<TExpr>), Var(VarId), Prim(Op, Vec<TExpr>), Call(FnId, Vec<TExpr>), Map(Box<TExpr>, TLambda), ZipMap(Box<TExpr>, Box<TExpr>, TLambda), Filter(Box<TExpr>, TLambda), If(Box<TExpr>, Box<TExpr>, Box<TExpr>), Block(Vec<(VarId, TExpr)>, Box<TExpr>) }`, `struct TLambda { pub params: Vec<VarId>, pub body: Box<TExpr> }`.

Rules:
* Builtin calls map to `Op`: `neg abs min max sqrt and or not sum count any all argmin argmax compress where iota fill rand_i64 rand_f64 length`, `cast_{t}`, `load_{t}`, `reduce(x, op)` with `op` one of the identifiers `add mul min max and or xor`, and `sum(x)` as `Reduce(Add)`. Binary operators map to `Op` directly.
* Result types come from `Registry::lookup` on the argument types. A missing kernel is a type error at the call: `no kernel for '*' on i8[], i8[]`. Mismatched operand types without a kernel are reported as `no implicit conversion: f64[] * i64` when the element types differ.
* `if` requires a `bool` or `bool[]` condition and equal branch types, and becomes `If`.
* Array literals need at least one element, all of the same scalar type.
* `map(x, f)`, `zip_map(x, y, f)`, `filter(x, f)`: `f` is a lambda or a top-level function name. Its parameters get the element types of the arrays. The body is checked with scalar parameter types. `filter` requires a `bool` result. A lambda body may contain only literals, its parameters, captured scalar variables, elementwise `Prim` ops (arithmetic, comparison, logic, `Select`, casts), `if`, and calls to functions whose bodies satisfy the same rule. Anything else is an error naming the construct, for example `lambda captures array 'x'`, `'sum' cannot be used inside a lambda`.
* Function calls check argument types against the signature. A call cycle among functions (including direct self-calls) is a type error `recursive function 'f'`.
* `print` and `load_*` are only allowed at top level, not in function bodies (`load_*` takes a string literal).
* Redefining a top-level name shadows it (new `VarId`).
* Function bodies see only their parameters and their own `let`s, not top-level bindings (my call, it keeps inlining a pure substitution).
* `Globals` provides `fn name(&self, var: VarId) -> &str`. Temporaries created by lowering are named `_t0`, `_t1`, and so on.

- [ ] **Step 1: Write the failing tests** using `parse` then `check`:
  * `let x = [1.0, 2.0]; let y = x * 2.0;` gives `y` type `f64[]`.
  * `let x = [1.0]; let y = x * 2;` fails with `ErrorKind::Type` at the `*` span, message contains `no implicit conversion`.
  * `let x = [1i8]; let y = x * x;` fails, message contains `no kernel for '*' on i8[]`.
  * `let a = 2u64; let b = a + a;` checks (`u64` scalar kernels exist).
  * `let x = [1u8, 2u8]; let s = sum(x);` gives `s` type `u64`.
  * `let x = [1.0]; let m = map(x, |v| (v * 2.0) + 1.0);` gives `f64[]`.
  * `let x = [1.0]; let y = [2.0]; let m = map(x, |v| v * sum(y));` fails, message contains `'sum' cannot be used inside a lambda`.
  * `let x = [1.0]; let y = [2.0]; let m = map(x, |v| v * y);` fails, message contains `captures array 'y'`.
  * `let t = 0.5; let f = filter([1.0], |v| v > t);` checks.
  * `let f = filter([1.0], |v| v * 2.0);` fails, message contains `bool`.
  * `fn f(a: f64) -> f64 { f(a) }` fails with `recursive function 'f'`.
  * `fn sq(v: f64) -> f64 { v * v } let m = map([1.0], sq);` checks.
  * `let m = reduce([1.0], max);` gives `f64`, `reduce([1.0], foo)` fails naming `foo`.
  * Unknown name `z` fails with the span of `z`.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --lib check`. Expected: FAIL.

- [ ] **Step 3: Implement** the checker.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/check src/lib.rs && git commit -m "Add type checker with lambda liftability rules"`

---

### Task 6: Lowering to first-order terms

**Files:**
* Create: `src/lower/mod.rs`
* Test: `src/lower/tests.rs` (declared `#[cfg(test)] mod tests;`)

**Interfaces:**
* Consumes: `TProgram`, `Globals`, `Registry`.
* Produces:
  * `pub enum Mode { Run, Repl }`.
  * `pub fn lower(prog: &TProgram, globals: &Globals, reg: &Registry, mode: Mode) -> Result<Lowered, Error>`.
  * `pub struct LeafId(pub u32)`, `pub enum Leaf { Const(Value), Var(VarId) }`.
  * `pub struct Term { pub kind: TermKind, pub ty: ValueType, pub span: Span }`, `pub enum TermKind { Op(Op, Vec<Term>), Leaf(LeafId), Bcast(Box<Term>) }`. `Term` implements structural `Eq` and `Hash` ignoring `span`.
  * `pub struct Lowered { pub leaves: Vec<Leaf>, pub stmts: Vec<LStmt> }`, `pub enum LStmt { Bind { var: VarId, term: Term, span: Span }, Print { term: Term, span: Span }, Show { term: Term, span: Span } }`.

Steps of lowering, in order:
1. Inline function calls by substituting argument terms for parameters, and inline `Block` lets by substitution.
2. Lift lambdas: substitute each parameter with the array argument term. `map` yields the substituted body. `zip_map` substitutes both. `filter(x, f)` yields `Op(Compress, [x, f(x)])`. `If` becomes `Op(Select, ...)`. A body that does not reference its parameter becomes `Op(Fill, [Op(Length, [x]), body])`.
3. Insert `Bcast` around every scalar-typed operand of an operation that has an array operand, then re-type each operation with `Registry::lookup` on the lifted argument types. A missing kernel is a type error at the operation's span (this catches lifted lambdas on types without array kernels).
4. Turn literals, string literals, and array literals into `Leaf::Const`, and references to materialized bindings into `Leaf::Var`.
5. In `Mode::Run`, a top-level `let` with exactly one use is substituted into its user and dropped. A `let` with zero or several uses stays a `Bind`. In `Mode::Repl`, every `let` stays a `Bind`.
6. Within each statement, a compound subterm (not a `Leaf`, not a `Bcast` of a `Leaf`) that occurs at least twice is hoisted into a fresh `Bind` placed before the statement and replaced by a `Leaf::Var`.

- [ ] **Step 1: Write the failing tests.** Helper `fn lowered(src: &str, mode: Mode) -> Lowered` and a printer `fn show(t: &Term) -> String` rendering ops lowercase with leaves as `L0, L1, ...` and broadcasts as `bcast(...)`:
  * `let x = [1.0]; let y = [2.0]; print(sum(x * y));` in `Run` gives one `Bind` per array and a `Print` of `reduce_add(mul(L_x, L_y))`.
  * `let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t));` gives the same `Print` term, with no `Bind` for `t`.
  * Same program with `print(sum(t)); print(t);` keeps `t` as a `Bind`.
  * Same program in `Mode::Repl` keeps `t` as a `Bind`.
  * `let x = [1.0]; print(filter(x, |v| v > 0.5));` gives `compress(L_x, gt(L_x, bcast(L_c)))`.
  * `let x = [1.0]; print(map(x, |v| 1.0));` gives `fill(length(L_x), L_c)`.
  * `let x = [1.0]; print(map(x, |v| if v > 0.0 { v } else { 0.0 - v }));` gives `select(gt(L_x, bcast(..)), L_x, sub(bcast(..), L_x))`.
  * `fn sq(v: f64) -> f64 { v * v } let x = [1.0]; print(sum(map(x, sq)));` gives `reduce_add(mul(L_x, L_x))`.
  * `let x = [1.0]; print(sum(x + x) + sum(x + x));` hoists `add(L_x, L_x)` into a `Bind`, and the `Print` term references it twice as a leaf.
  * `let x = [1i8]; print(map(x, |v| v * v));` fails with `ErrorKind::Type` and `no kernel`, because the scalar `i8` kernel exists but the array one does not.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --lib lower`. Expected: FAIL.

- [ ] **Step 3: Implement** `lower`.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/lower src/lib.rs && git commit -m "Add lowering with lifting, let inlining, and CSE hoisting"`

---

### Task 7: Fused kernels with oracle property tests

**Files:**
* Create: `src/kernels/scalar/fused.rs`
* Modify: `src/kernels/mod.rs` (replace the `FusedPattern` placeholder, register fused kernels)
* Test: `tests/fused_oracle.rs`

**Interfaces:**
* Consumes: Task 3 and 4 kernels.
* Produces:
  * `pub struct FusedPattern { pub lhs: String, pub vars: Vec<String> }`. `lhs` is an egglog pattern over single-op kernel names with `?`-variables and `(Bcast ?c)` for broadcast operands. `vars` lists the variables in the fused kernel's argument order. The rule emitted later is `(rewrite <lhs> (<name> <vars...>))`.
  * `pub fn reference(reg: &Registry, id: KernelId, args: &[Value]) -> Result<Value, String>`: evaluates the fused kernel's pattern through the single-op scalar kernels. This is the oracle.
  * Fused kernels, for each `t` in `f64 f32 i64 i32` unless noted:
    * `dot_{t}`: lhs `(reduce_add_{t} (mul_{t} ?x ?y))`, vars `?x ?y`, cost `2w`.
    * `sumsq_{t}`: lhs `(reduce_add_{t} (mul_{t} ?x ?x))`, vars `?x`, cost `w`.
    * `count_{cmp}_{t}` for `cmp` in `eq ne lt le gt ge`: lhs `(count ({cmp}_{t} ?x (Bcast ?c)))`, vars `?x ?c`, cost `w`.
    * `filter_{cmp}_{t}`: lhs `(compress_{t} ?x ({cmp}_{t} ?x (Bcast ?c)))`, vars `?x ?c`, cost `2w`.
    * `condsum_{cmp}_{t}`: lhs `(reduce_add_{t} (select_{t} ({cmp}_{t} ?x (Bcast ?c)) ?x (Bcast ?e)))`, vars `?x ?c ?e`, cost `w`.
  * Integer `dot_i32` and `sumsq_i32` multiply wrapping in `i32` and accumulate widened to `i64`, exactly like the unfused pair.

- [ ] **Step 1: Write the failing test** `tests/fused_oracle.rs`: for every kernel whose `pattern` is `Some`, a proptest generates arrays of length 0 to 300 (float generators mix finite values, `±0.0`, `±inf`, and NaN, integer generators cover the full range) plus scalars, runs the fused scalar impl and `reference`, and asserts equal results. Floats compare with `to_bits`, except that when the reference is NaN the fused result only has to be NaN. One extra test asserts that at least 5 kernel families (`dot sumsq count filter condsum`) are registered with patterns.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --test fused_oracle`. Expected: FAIL to compile.

- [ ] **Step 3: Implement** the fused kernels (single loops that compute the same per-element values as the unfused kernels and feed the same canonical accumulation) and `reference`.

- [ ] **Step 4: Run** the test. Expected: PASS.

- [ ] **Step 5: Commit** `git add src/kernels tests/fused_oracle.rs && git commit -m "Add fused tier 1 kernels with oracle property tests"`

---

### Task 8: Planner on egglog

**Files:**
* Create: `src/plan/mod.rs`, `src/plan/egglog.rs`, `src/plan/sexpr.rs`
* Test: `src/plan/tests.rs`

**Interfaces:**
* Consumes: `Term`, `Lowered`, `Registry`, `FusedPattern`.
* Produces:
  * `pub enum PlanNode { Kernel(KernelId, Vec<PlanNode>), Leaf(LeafId), Bcast(Box<PlanNode>) }`.
  * `pub struct Planner` (owns the generated egglog prelude, built from the registry), `Planner::new(reg: &Registry) -> Planner`, `fn plan(&self, reg: &Registry, term: &Term) -> Result<PlanNode, Error>`.
  * `pub fn render(reg: &Registry, node: &PlanNode, leaves: &[Leaf], globals: &Globals) -> String`: renders kernel names and leaves as s-expressions, used by `--plan`, for example `(dot_f64 x y)`, with variable names for `Leaf::Var` (via `Globals::name`), literal values for `Leaf::Const` scalars, `<f64[3]>` for constant arrays, and the scalar inside a `Bcast` rendered bare.
  * `pub const SATURATION_ITERATIONS: usize = 8;`

Egglog program per statement (my call: drive egglog through its text interface, the most stable API surface):
* Prelude, generated once: `(datatype Expr (Leaf i64 :cost 0) (Bcast Expr :cost 0) (<kernel> Expr... :cost <cost>) ...)` with one constructor per kernel in the registry, single-op and fused. Rules: for each `commutative` single-op kernel, `(rewrite (k ?a ?b) (k ?b ?a))`. For each `t`, `(rewrite (gt_t ?a ?b) (lt_t ?b ?a))` and the reverse, same for `ge`/`le`. For each fused kernel, `(rewrite <lhs> (<name> <vars>))`.
* Per statement: a fresh `egglog::EGraph`, the prelude, `(let root <term>)` where each `Op` node is written with the kernel name that `Registry::lookup` gives for its argument types and each leaf as `(Leaf n)`, then `(run 8)` and `(extract root)`.
* Constructors are emitted with a `K_` prefix (`K_dot_f64`) so kernel names can never collide with egglog keywords. Fused patterns are written without the prefix and prefixed during emission.
* Parse the extracted s-expression with `sexpr.rs`, strip the prefix, and map constructor names back through `Registry::by_name`.

- [ ] **Step 1: Spike the egglog API.** Write a test `egglog_smoke` that runs a tiny program text (a datatype with two constructors, one rewrite, `(run 3)`, `(extract ...)`) through egglog 3 and asserts the extracted string. Check the exact datatype and cost syntax and how extraction output is returned against the egglog 3 crate docs before writing more. If egglog cannot express per-constructor costs or extraction as needed, stop and report BLOCKED with what failed, because the spec allows falling back to `egg` but that decision belongs to the user.

- [ ] **Step 2: Write the failing planner tests** (lower a program in `Mode::Run`, plan the `Print` term, `render` it):
  * `sum(x * y)` renders `(dot_f64 x y)`.
  * `sum(y * x)` renders `(dot_f64 y x)` or `(dot_f64 x y)`. Assert the kernel is `dot_f64`.
  * `sum(x * x)` renders `(sumsq_f64 x)`, not `dot_f64`.
  * `count(x > 0.5)` selects `count_gt_f64` (assert the root kernel name, the scalar argument renders as `0.5`).
  * `count(0.5 < x)` also selects `count_gt_f64` (through comparison flipping).
  * `filter(x, |v| v > 0.5)` selects `filter_gt_f64`.
  * `sum(map(x, |v| if v > 0.5 { v } else { 0.0 }))` selects `condsum_gt_f64`.
  * `let t = x * y; print(sum(t));` selects `dot_f64`.
  * `let t = x * y; print(sum(t)); print(t);` plans `sum(t)` as `reduce_add_f64` over leaf `t`.
  * `sum(x + y)` renders `(reduce_add_f64 (add_f64 x y))` (no fused kernel exists).
  * These tests only plan and execute nothing.

- [ ] **Step 3: Run** `CARGO_INCREMENTAL=0 cargo test --lib plan`. Expected: FAIL.

- [ ] **Step 4: Implement** the planner.

- [ ] **Step 5: Run** the tests. Expected: PASS.

- [ ] **Step 6: Commit** `git add src/plan src/lib.rs && git commit -m "Add egglog planner covering terms with kernel patterns"`

---

### Task 9: Executor and the `Interpreter` library API

**Files:**
* Create: `src/exec.rs`, `src/interp.rs`
* Modify: `src/lib.rs` (re-exports)
* Test: `tests/programs.rs`, `tests/programs/*.vk` with `*.out` expected stdout

**Interfaces:**
* Consumes: everything above.
* Produces:
  * `pub struct Options { pub isa: Option<Isa>, pub show_plan: bool, pub trace: bool, pub stats: bool }` (`Default`: all off, ISA detected).
  * `pub struct Interpreter`, `Interpreter::new(opts: Options) -> Result<Interpreter, Error>` (`ErrorKind::Usage` when the requested ISA is not available on this CPU), `fn run(&mut self, src: &str, out: &mut dyn Write, diag: &mut dyn Write) -> Result<(), Error>` (whole program, `Mode::Run`), `fn eval_repl(&mut self, src: &str, out: &mut dyn Write, diag: &mut dyn Write) -> Result<(), Error>` (`Mode::Repl`, an expression statement prints its value to `out`), `fn bind(&mut self, name: &str, value: Value)`, `fn get(&self, name: &str) -> Option<Value>`, `fn write_stats(&self, diag: &mut dyn Write)`.

Behavior:
* `run` parses, checks, lowers, and plans every statement before executing any, so all type errors come before any kernel runs. Then statements execute in source order.
* The executor evaluates `PlanNode`s: leaves from `Lowered.leaves` and the variable environment, `Bcast` evaluates to its scalar, kernels through `Registry::select_impl`. A kernel `Err(msg)` becomes `Error::runtime(msg, Some(statement span))`.
* `--plan` (`show_plan`): before execution, one line per statement to `diag`: `plan <line>: <render>`.
* `--trace`: one line per kernel call to `diag`: `trace <kernel> isa=<variant actually used> n=<largest input length> read=<bytes> write=<bytes> time=<µs> gbps=<(read+write)/time>`.
* `--stats`: totals per kernel (calls, bytes, time) written by `write_stats`.

- [ ] **Step 1: Write the failing tests.** `tests/programs.rs` runs every `tests/programs/*.vk` through `Interpreter::run` with default options and compares stdout to the matching `.out` file. Add these programs:
  * `dot.vk`: `let x = [1.0, 2.0, 3.0]; let y = [4.0, 5.0, 6.0]; print(sum(x * y));`, output `32.0`.
  * `filter.vk`: `filter`, `count`, `where` on a literal array, including an all-false filter whose `sum` prints `0.0` and whose `print` prints `[]`.
  * `reduce.vk`: `reduce` with `max`, `min`, `mul`, `sum` of `u8` printing a value above 255, `argmin` with a tie.
  * `lambda.vk`: `map` with an `if`, `zip_map`, a named function passed to `map`, a captured scalar.
  * `rand.vk`: `print(length(rand_f64(10, 1)));` and `print(rand_i64(2, 0));` with values precomputed from `splitmix64`.
  * Further tests in `tests/programs.rs`, not files:
    * A program with a length mismatch on line 3 returns `ErrorKind::Runtime` with `span.line == 3`.
    * A program with a type error on line 4 and a `print` on line 1 returns `ErrorKind::Type` and writes nothing to `out`.
    * With `show_plan`, `sum(x * y)` writes a `diag` line containing `dot_f64`.
    * With `trace`, `diag` contains `isa=scalar`.
    * `Options { isa: Some(Isa::Neon), .. }` on x86 returns `ErrorKind::Usage`.
    * `eval_repl("let x = [1.0];")` then `eval_repl("sum(x);")` prints `1.0` to `out`.
    * `bind("x", Value::Array(...))` then `run("print(sum(x));")` uses the bound array without copying (assert the data pointer of `get("x")` equals the original).
    * `fn assert_send<T: Send>() {}` called with `Interpreter`.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --test programs`. Expected: FAIL.

- [ ] **Step 3: Implement** `exec.rs` and `interp.rs`. `bind` makes the name visible to the checker as a global of the value's type.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Commit** `git add src tests && git commit -m "Add executor and Interpreter API with plan and trace output"`

---

### Task 10: Command-line tool

**Files:**
* Modify: `src/main.rs`
* Test: `tests/cli.rs`

**Interfaces:**
* Consumes: `Interpreter`, `Options`, `Isa::from_name`.
* Produces: `verkehrt run FILE [--isa scalar|avx2|avx512|neon] [--plan] [--trace] [--stats]` and `verkehrt repl` with the same flags. Program output on stdout, plan, trace, stats, and errors on stderr. Exit code 0 on success, 1 on a program error, 2 on usage errors (clap's default).

REPL: `rustyline` editor with prompt `vk> `. Each line is passed to `eval_repl`. Errors print and the session continues. `:quit` or end of input exits. `:stats` prints stats.

- [ ] **Step 1: Write the failing tests** with `assert_cmd`:
  * `verkehrt run tests/programs/dot.vk` prints `32.0` and exits 0.
  * `verkehrt run --plan tests/programs/dot.vk` prints `32.0` on stdout and stderr contains `dot_f64`.
  * A temp file with `let x = [1.0] * 2;` exits 1 and stderr contains `error[type] at 1:`.
  * `verkehrt run --isa bogus x.vk` exits 2.
  * `verkehrt repl` with stdin `let x = [1.0, 2.0];\nsum(x);\n` prints `3.0` in stdout.

- [ ] **Step 2: Run** `CARGO_INCREMENTAL=0 cargo test --test cli`. Expected: FAIL.

- [ ] **Step 3: Implement** `main.rs` with `clap` derive. When stdin is not a terminal, the REPL reads lines from stdin without `rustyline`, so the test works.

- [ ] **Step 4: Run** the full suite `CARGO_INCREMENTAL=0 cargo test` and `CARGO_INCREMENTAL=0 cargo clippy --all-targets -- -D warnings`. Expected: all pass, no warnings.

- [ ] **Step 5: Commit** `git add src/main.rs tests/cli.rs && git commit -m "Add verkehrt run and repl commands"`
