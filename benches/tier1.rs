//! Tier 1 and tier 2 kernels against native Rust loops, and the dispatch crossover sweep.
//!
//! Groups are `tier1/<program>/<variant>`, `tier2/<program>/<variant>`, and
//! `crossover/<program>/<n>`. Interpreter variants
//! measure `Prepared::execute` only. Native baselines are plain loops compiled for the baseline
//! target and, per ISA, under the same `#[target_feature]` set as the ISA's kernels.
//! Throughput is input bytes read, so criterion's bytes/s are the input bandwidth.

use std::hint::black_box;
use std::io::sink;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use verkehrt::kernels::{Backend, Isa};
use verkehrt::value::{Array, Elem, Scalar, Value};
use verkehrt::{Interpreter, Options, Prepared};

const N: usize = 10_000_000;
const LANES: usize = 16;

/// A native baseline compiled for one target. The ISA flavors exist on x86-64 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Native {
    /// Baseline target, comparable with the scalar kernels.
    Base,
    #[cfg(target_arch = "x86_64")]
    X86_64V3,
    #[cfg(target_arch = "x86_64")]
    X86_64V4,
}

impl Native {
    fn label(self) -> &'static str {
        match self {
            Native::Base => "native-base",
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V3 => "native-x86-64-v3",
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V4 => "native-x86-64-v4",
        }
    }

    fn available(self) -> bool {
        match self {
            Native::Base => true,
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V3 => Isa::X86_64V3.is_available(),
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V4 => Isa::X86_64V4.is_available(),
        }
    }

    /// Every flavor compiled for this target, available or not.
    fn all() -> Vec<Native> {
        vec![
            Native::Base,
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V3,
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V4,
        ]
    }
}

mod checked {
    use super::Native;

    /// A `Native` flavor that this CPU can run. The field is private to this module, so
    /// `Native::check` is the only way to obtain one, and it checks `available` first.
    #[derive(Clone, Copy, Debug)]
    pub struct Checked(Native);

    impl Native {
        /// The flavor as `Checked` when this CPU can run it.
        pub fn check(self) -> Option<Checked> {
            self.available().then_some(Checked(self))
        }
    }

    impl Checked {
        pub fn native(self) -> Native {
            self.0
        }
    }
}

use checked::Checked;

/// The canonical 16-accumulator sum, with the pairwise `+8 +4 +2 +1` finish and the tail
/// folded into accumulator 0 in index order. The 16-element chunks let it vectorize.
#[inline(always)]
fn sum_body(x: &[f64]) -> f64 {
    let (chunks, tail) = x.as_chunks::<LANES>();
    let mut acc = [0.0f64; LANES];
    for c in chunks {
        for j in 0..LANES {
            acc[j] += c[j];
        }
    }
    finish(acc, tail.iter().copied())
}

/// The canonical dot product, one product per accumulator, no fused multiply-add.
#[inline(always)]
fn dot_body(x: &[f64], y: &[f64]) -> f64 {
    let (xc, xt) = x.as_chunks::<LANES>();
    let (yc, yt) = y.as_chunks::<LANES>();
    let mut acc = [0.0f64; LANES];
    for (a, b) in xc.iter().zip(yc) {
        for j in 0..LANES {
            acc[j] += a[j] * b[j];
        }
    }
    finish(acc, xt.iter().zip(yt).map(|(a, b)| a * b))
}

#[inline(always)]
fn finish(mut acc: [f64; LANES], tail: impl Iterator<Item = f64>) -> f64 {
    let mut step = LANES / 2;
    while step > 0 {
        for j in 0..step {
            acc[j] += acc[j + step];
        }
        step /= 2;
    }
    tail.fold(acc[0], |a, x| a + x)
}

#[inline(always)]
fn count_body(x: &[f64], c: f64) -> i64 {
    let mut n = 0i64;
    for &v in x {
        n += i64::from(v > c);
    }
    n
}

/// Predicated compaction: every element is stored, the output index advances on a match.
#[inline(always)]
fn filter_body(x: &[f64], c: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; x.len()];
    let mut k = 0usize;
    for &v in x {
        out[k] = v;
        k += usize::from(v > c);
    }
    out.truncate(k);
    out
}

/// Only the crossover sweep multiplies.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn mul_body(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(a, b)| a * b).collect()
}

/// The minimum of the rows, one running minimum per accumulator so that it vectorizes. Minimum
/// is order-independent on the NaN-free bench data, so no canonical order is needed. The
/// comparison has no signed-zero order (`-0.0 == +0.0`), so it agrees with Verkehrt only on
/// data without NaN and `-0.0`.
#[inline(always)]
fn min_body(x: &[f64]) -> f64 {
    let (chunks, tail) = x.as_chunks::<LANES>();
    let mut acc = [f64::INFINITY; LANES];
    for c in chunks {
        for j in 0..LANES {
            acc[j] = if c[j] < acc[j] { c[j] } else { acc[j] };
        }
    }
    let mut m = f64::INFINITY;
    for v in acc.into_iter().chain(tail.iter().copied()) {
        m = if v < m { v } else { m };
    }
    m
}

/// The index of the first minimum, with one running (value, index) pair per accumulator. A
/// lane replaces its pair only on a strictly smaller row, so it keeps its first minimum. The
/// merge takes the smaller value and, on equal values, the smaller index, which gives the
/// first index overall. The comparison has no signed-zero order (`-0.0 == +0.0`) and a NaN
/// never wins, so it agrees with Verkehrt only on data without NaN and `-0.0`, which the bench
/// data is.
#[inline(always)]
fn argmin_body(x: &[f64]) -> i64 {
    let (chunks, tail) = x.as_chunks::<LANES>();
    let mut val = [f64::INFINITY; LANES];
    let mut idx = [0usize; LANES];
    for (k, c) in chunks.iter().enumerate() {
        for j in 0..LANES {
            let lt = c[j] < val[j];
            val[j] = if lt { c[j] } else { val[j] };
            idx[j] = if lt { k * LANES + j } else { idx[j] };
        }
    }
    let mut best = (f64::INFINITY, usize::MAX);
    let base = chunks.len() * LANES;
    let pairs = val
        .into_iter()
        .zip(idx)
        .chain(tail.iter().copied().enumerate().map(|(j, v)| (v, base + j)));
    for (v, i) in pairs {
        if v < best.0 || (v == best.0 && i < best.1) {
            best = (v, i);
        }
    }
    i64::try_from(best.1).unwrap()
}

/// Defines a module with `base`, `v3`, and `v4` functions around one shared body. The
/// `v3` and `v4` functions are compiled for the x86-64-v3 and x86-64-v4 levels and
/// exist on x86-64 only.
macro_rules! native_fns {
    ($name:ident, $body:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        mod $name {
            use super::*;
            pub fn base($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3")]
            pub fn v3($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
            pub fn v4($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
        }
    };
}

native_fns!(n_sum, sum_body, (x: &[f64]) -> f64);
native_fns!(n_dot, dot_body, (x: &[f64], y: &[f64]) -> f64);
native_fns!(n_count, count_body, (x: &[f64], c: f64) -> i64);
native_fns!(n_filter, filter_body, (x: &[f64], c: f64) -> Vec<f64>);
native_fns!(n_min, min_body, (x: &[f64]) -> f64);
native_fns!(n_argmin, argmin_body, (x: &[f64]) -> i64);
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
native_fns!(n_mul, mul_body, (x: &[f64], y: &[f64]) -> Vec<f64>);

/// Calls the flavor of a function module that a `Checked` names. The availability check
/// happened when the `Checked` was made, once per bench group before any timing, so it stays
/// out of timed closures.
macro_rules! call_native {
    ($nat:expr, $m:ident ( $($arg:expr),* )) => {{
        let checked: Checked = $nat;
        match checked.native() {
            Native::Base => $m::base($($arg),*),
            // SAFETY: a `Checked` comes only from `Native::check`, which returned it because
            // `available` found the x86-64-v3 level through `features::runnable`. The
            // arguments are ordinary safe references, so only the CPU features are required.
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V3 => unsafe { $m::v3($($arg),*) },
            // SAFETY: as above, `Native::check` returned the `Checked` because `available`
            // found the x86-64-v4 level through `features::runnable`. The arguments are
            // ordinary safe references.
            #[cfg(target_arch = "x86_64")]
            Native::X86_64V4 => unsafe { $m::v4($($arg),*) },
        }
    }};
}

/// An interpreter variant.
struct Interp {
    label: String,
    interp: Interpreter,
    /// Whether it uses the intrinsics backend. On x86-64 the selections, the 64-bit
    /// `gather`s, and the AVX-512 `scan_by_add` on 64-bit integers have intrinsics entries,
    /// so only the `filter`, `gather`, and `scan_by` groups measure these variants. On
    /// aarch64 the selections and the float extremes (`min`, `argmin`) have them. Elsewhere
    /// the intrinsics backend resolves to the scalar entries, so those groups skip it.
    intrinsics: bool,
}

fn interpreters() -> Vec<Interp> {
    let mut v = vec![Interp {
        label: "scalar".into(),
        interp: Interpreter::new(Options {
            isa: Some(Isa::Scalar),
            backend: Backend::Auto,
            ..Options::default()
        })
        .unwrap(),
        intrinsics: false,
    }];
    for (isa, name) in [(Isa::X86_64V3, "x86-64-v3"), (Isa::X86_64V4, "x86-64-v4")] {
        if !isa.is_available() {
            continue;
        }
        for (backend, suffix) in [(Backend::Auto, "auto"), (Backend::Intrinsics, "intrinsics")] {
            v.push(Interp {
                label: format!("{name}-{suffix}"),
                interp: Interpreter::new(Options {
                    isa: Some(isa),
                    backend,
                    ..Options::default()
                })
                .unwrap(),
                intrinsics: backend == Backend::Intrinsics,
            });
        }
    }
    // On aarch64 the scalar variant is the baseline build, which the compiler vectorizes with
    // NEON. The auto slot of NEON is empty, so the only NEON variant is the intrinsics one.
    #[cfg(target_arch = "aarch64")]
    if Isa::Neon.is_available() {
        v.push(Interp {
            label: "neon-intrinsics".into(),
            interp: Interpreter::new(Options {
                isa: Some(Isa::Neon),
                backend: Backend::Intrinsics,
                ..Options::default()
            })
            .unwrap(),
            intrinsics: true,
        });
    }
    v
}

/// The flavors this CPU can run.
fn natives() -> Vec<Checked> {
    Native::all()
        .into_iter()
        .filter_map(Native::check)
        .collect()
}

/// `n` uniform doubles in `[0, 1)`, produced by the interpreter's own `rand_f64`.
fn rand_f64(n: usize, seed: i64) -> Array {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    })
    .unwrap();
    let p = interp
        .prepare(&format!("export r = rand_f64({n}, {seed});"))
        .unwrap();
    let mut out = p.execute(&interp, &[], &mut sink()).unwrap();
    match out.pop().unwrap().1 {
        Value::Array(a) => a,
        other => panic!("expected an array, got {other}"),
    }
}

fn run(p: &Prepared, i: &Interpreter, inputs: &[(&str, Value)]) -> Value {
    p.execute(i, inputs, &mut sink()).unwrap().pop().unwrap().1
}

fn f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Scalar(Scalar::F64(x)) => *x,
        other => panic!("expected f64, got {other}"),
    }
}

fn i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Scalar(Scalar::I64(x)) => *x,
        other => panic!("expected i64, got {other}"),
    }
}

fn f64_rows(v: &Value) -> &[f64] {
    match v {
        Value::Array(a) => a.as_slice::<f64>(),
        other => panic!("expected an array, got {other}"),
    }
}

fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn configure(g: &mut BenchmarkGroup<'_, WallTime>) {
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
}

fn tier1(c: &mut Criterion) {
    let xa = rand_f64(N, 1);
    let ya = rand_f64(N, 2);
    let x = xa.as_slice::<f64>();
    let y = ya.as_slice::<f64>();
    let xv = Value::Array(xa.clone());
    let yv = Value::Array(ya.clone());
    let all_interps = interpreters();
    // `sum`, `dot`, and `count` resolve the intrinsics backend to their auto entries, or to
    // scalar on aarch64, so the intrinsics variants stay out of those groups.
    let interps: Vec<&Interp> = all_interps.iter().filter(|it| !it.intrinsics).collect();
    // `min` and `argmin` have intrinsics entries on aarch64 only.
    let ext_interps: Vec<&Interp> = all_interps
        .iter()
        .filter(|it| !it.intrinsics || cfg!(target_arch = "aarch64"))
        .collect();
    let nats = natives();
    let bytes = u64::try_from(N * size_of::<f64>()).unwrap();

    // The canonical tail: a length that is not a multiple of 16 for sum and dot.
    {
        let n = 10_000_007;
        let (ta, tb) = (rand_f64(n, 3), rand_f64(n, 4));
        let (tx, ty) = (ta.as_slice::<f64>(), tb.as_slice::<f64>());
        assert_ne!(n % LANES, 0);
        let want_sum = n_sum::base(tx).to_bits();
        let want_dot = n_dot::base(tx, ty).to_bits();
        for nat in &nats {
            assert_eq!(call_native!(*nat, n_sum(tx)).to_bits(), want_sum);
            assert_eq!(call_native!(*nat, n_dot(tx, ty)).to_bits(), want_dot);
        }
        let ins = [
            ("x", Value::Array(ta.clone())),
            ("y", Value::Array(tb.clone())),
        ];
        for it in &interps {
            let ps = it
                .interp
                .prepare("input x: f64[]; export s = sum(x);")
                .unwrap();
            let pd = it
                .interp
                .prepare("input x: f64[]; input y: f64[]; export s = sum(x * y);")
                .unwrap();
            let got_sum = f64_scalar(&run(&ps, &it.interp, &ins[..1]));
            assert_eq!(got_sum.to_bits(), want_sum, "tail sum {}", it.label);
            let got_dot = f64_scalar(&run(&pd, &it.interp, &ins));
            assert_eq!(got_dot.to_bits(), want_dot, "tail dot {}", it.label);
        }
    }

    // sum
    {
        let src = "input x: f64[]; export s = sum(x);";
        let want = n_sum::base(x);
        for n in &nats {
            assert_eq!(call_native!(*n, n_sum(x)).to_bits(), want.to_bits());
        }
        let mut g = c.benchmark_group("tier1/sum");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        for it in &interps {
            let p = it.interp.prepare(src).unwrap();
            let ins = [("x", xv.clone())];
            let got = f64_scalar(&run(&p, &it.interp, &ins));
            assert_eq!(got.to_bits(), want.to_bits(), "sum {}", it.label);
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_sum(black_box(x))))
            });
        }
        g.finish();
    }

    // dot
    {
        let src = "input x: f64[]; input y: f64[]; export s = sum(x * y);";
        let want = n_dot::base(x, y);
        for n in &nats {
            assert_eq!(call_native!(*n, n_dot(x, y)).to_bits(), want.to_bits());
        }
        let mut g = c.benchmark_group("tier1/dot");
        configure(&mut g);
        g.throughput(Throughput::Bytes(2 * bytes));
        for it in &interps {
            let p = it.interp.prepare(src).unwrap();
            let ins = [("x", xv.clone()), ("y", yv.clone())];
            let got = f64_scalar(&run(&p, &it.interp, &ins));
            assert_eq!(got.to_bits(), want.to_bits(), "dot {}", it.label);
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_dot(black_box(x), black_box(y))))
            });
        }
        g.finish();
    }

    // count
    {
        let cut = 0.5f64;
        let src = format!("input x: f64[]; export n = count(x > {cut:?});");
        let want = n_count::base(x, cut);
        for n in &nats {
            assert_eq!(call_native!(*n, n_count(x, cut)), want);
        }
        let mut g = c.benchmark_group("tier1/count");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        for it in &interps {
            let p = it.interp.prepare(&src).unwrap();
            let ins = [("x", xv.clone())];
            assert_eq!(
                i64_scalar(&run(&p, &it.interp, &ins)),
                want,
                "count {}",
                it.label
            );
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_count(black_box(x), cut)))
            });
        }
        g.finish();
    }

    // min
    {
        let src = "input x: f64[]; export m = reduce(x, min);";
        let want = n_min::base(x);
        for n in &nats {
            assert_eq!(call_native!(*n, n_min(x)).to_bits(), want.to_bits());
        }
        let mut g = c.benchmark_group("tier1/min");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        for it in &ext_interps {
            let p = it.interp.prepare(src).unwrap();
            let ins = [("x", xv.clone())];
            let got = f64_scalar(&run(&p, &it.interp, &ins));
            assert_eq!(got.to_bits(), want.to_bits(), "min {}", it.label);
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_min(black_box(x))))
            });
        }
        g.finish();
    }

    // argmin, the first index of the minimum
    {
        let src = "input x: f64[]; export i = argmin(x);";
        let want = n_argmin::base(x);
        // The plain forward loop with a strict comparison also returns the first index.
        let (mut best, mut first) = (f64::INFINITY, 0usize);
        for (i, &v) in x.iter().enumerate() {
            if v < best {
                best = v;
                first = i;
            }
        }
        assert_eq!(want, i64::try_from(first).unwrap(), "argmin reference");
        for n in &nats {
            assert_eq!(call_native!(*n, n_argmin(x)), want);
        }
        eprintln!("argmin: first minimum {best:?} at index {want}");
        let mut g = c.benchmark_group("tier1/argmin");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        for it in &ext_interps {
            let p = it.interp.prepare(src).unwrap();
            let ins = [("x", xv.clone())];
            assert_eq!(
                i64_scalar(&run(&p, &it.interp, &ins)),
                want,
                "argmin {}",
                it.label
            );
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_argmin(black_box(x))))
            });
        }
        g.finish();
    }

    // filter at three selectivities
    for (name, cut) in [
        ("filter-1pct", 0.99f64),
        ("filter-50pct", 0.5),
        ("filter-99pct", 0.01),
    ] {
        let src = format!("input x: f64[]; export f = filter(x, |v| v > {cut:?});");
        let want = n_filter::base(x, cut);
        // The data is uniform in [0, 1), so `x > cut` keeps about `1 - cut` of the rows.
        let matched = u32::try_from(want.len()).unwrap();
        let frac = f64::from(matched) / f64::from(u32::try_from(N).unwrap());
        assert!(
            (frac - (1.0 - cut)).abs() < 0.005,
            "{name}: {matched} matches, fraction {frac}"
        );
        eprintln!("{name}: {matched} of {N} rows match ({:.3}%)", frac * 100.0);
        for n in &nats {
            assert_eq!(bits(&call_native!(*n, n_filter(x, cut))), bits(&want));
        }
        let mut g = c.benchmark_group(format!("tier1/{name}"));
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        for it in &all_interps {
            let p = it.interp.prepare(&src).unwrap();
            let ins = [("x", xv.clone())];
            let out = run(&p, &it.interp, &ins);
            assert_eq!(bits(f64_rows(&out)), bits(&want), "{name} {}", it.label);
            g.bench_function(&it.label, |b| {
                b.iter(|| run(&p, &it.interp, black_box(&ins)))
            });
        }
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_filter(black_box(x), cut)))
            });
        }
        g.finish();
    }

    // Memory bandwidth reference: read 80 MB and write 80 MB.
    {
        let mut dst = vec![0.0f64; N];
        let mut g = c.benchmark_group("tier1/copy");
        configure(&mut g);
        g.throughput(Throughput::Bytes(2 * bytes));
        g.bench_function("copy-read-write", |b| {
            b.iter(|| {
                dst.copy_from_slice(black_box(x));
                black_box(&mut dst);
            })
        });
        g.finish();
    }
}

/// Buckets of the `histogram` and `reduce_into` groups.
const BUCKETS: usize = 1024;
/// Divisor of the `div_const` groups.
const DIVISOR: i64 = 60;

/// `n` pseudo-random words from splitmix64, mapped through `f`. Deterministic per seed.
fn splitmix<T>(n: usize, seed: u64, f: impl Fn(u64) -> T) -> Vec<T> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            f(z ^ (z >> 31))
        })
        .collect()
}

#[inline(always)]
fn add_i64_body(x: &[i64], y: &[i64]) -> Vec<i64> {
    x.iter().zip(y).map(|(a, b)| a.wrapping_add(*b)).collect()
}

/// The inclusive prefix sum, wrapping like `scan(x, add)`.
#[inline(always)]
fn scan_add_body(x: &[i64]) -> Vec<i64> {
    let mut acc = 0i64;
    x.iter()
        .map(|&v| {
            acc = acc.wrapping_add(v);
            acc
        })
        .collect()
}

/// The count per key with Rust's bounds-checked indexing, which panics where `histogram`
/// returns an error.
#[inline(always)]
fn histogram_body(keys: &[i64], n: usize) -> Vec<i64> {
    let mut h = vec![0i64; n];
    for &k in keys {
        h[k as usize] += 1;
    }
    h
}

/// The wrapping sum per key, see `histogram_body`.
#[inline(always)]
fn reduce_into_add_body(keys: &[i64], vals: &[i64], n: usize) -> Vec<i64> {
    let mut h = vec![0i64; n];
    for (&k, &v) in keys.iter().zip(vals) {
        let b = &mut h[k as usize];
        *b = b.wrapping_add(v);
    }
    h
}

/// Division by the constant `DIVISOR`, which the compiler lowers to a multiply-high.
#[inline(always)]
fn div_const_body(x: &[i64]) -> Vec<i64> {
    x.iter().map(|v| v / DIVISOR).collect()
}

native_fns!(n_add_i64, add_i64_body, (x: &[i64], y: &[i64]) -> Vec<i64>);
native_fns!(n_scan_add, scan_add_body, (x: &[i64]) -> Vec<i64>);
native_fns!(n_histogram, histogram_body, (keys: &[i64], n: usize) -> Vec<i64>);
native_fns!(
    n_reduce_into_add,
    reduce_into_add_body,
    (keys: &[i64], vals: &[i64], n: usize) -> Vec<i64>
);
native_fns!(n_div_const, div_const_body, (x: &[i64]) -> Vec<i64>);

/// Whether every index lies in `0..n`, checked without an early exit as the `gather` kernel
/// does before it reads a row.
#[inline(always)]
fn indices_in_range(idx: &[i64], n: usize) -> bool {
    let n = n as u64;
    !idx.iter().fold(false, |b, &i| b | (i.cast_unsigned() >= n))
}

/// `gather` with the same bounds pass as the kernel. The baseline target uses indexed loads,
/// the ISA flavors use the hardware gathers, four rows per `vpgatherqq` on AVX2 and eight on
/// AVX-512, so they are not the shared portable body.
mod n_gather {
    use super::indices_in_range;

    pub fn base(xs: &[i64], idx: &[i64]) -> Vec<i64> {
        assert!(indices_in_range(idx, xs.len()), "gather index out of range");
        idx.iter().map(|&i| xs[i as usize]).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3"
    )]
    pub fn v3(xs: &[i64], idx: &[i64]) -> Vec<i64> {
        use std::arch::x86_64::{
            __m256i, _mm256_i64gather_epi64, _mm256_loadu_si256, _mm256_storeu_si256,
        };
        assert!(indices_in_range(idx, xs.len()), "gather index out of range");
        let mut out = vec![0i64; idx.len()];
        let (ic, it) = idx.as_chunks::<4>();
        let (oc, ot) = out.as_chunks_mut::<4>();
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            // SAFETY: the function is compiled for x86-64-v3, which its caller checked. The load
            // and store touch exactly the four `i64` of `i` and `o`, unaligned accesses are
            // allowed. Every index lies in `0..xs.len()` by the assertion above, so each
            // gathered address `xs + 8 * i` is a row of `xs`.
            unsafe {
                let v = _mm256_loadu_si256(i.as_ptr().cast::<__m256i>());
                let g = _mm256_i64gather_epi64::<8>(xs.as_ptr(), v);
                _mm256_storeu_si256(o.as_mut_ptr().cast::<__m256i>(), g);
            }
        }
        for (o, &i) in ot.iter_mut().zip(it) {
            *o = xs[i as usize];
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    pub fn v4(xs: &[i64], idx: &[i64]) -> Vec<i64> {
        use std::arch::x86_64::{_mm512_i64gather_epi64, _mm512_loadu_si512, _mm512_storeu_si512};
        assert!(indices_in_range(idx, xs.len()), "gather index out of range");
        let mut out = vec![0i64; idx.len()];
        let (ic, it) = idx.as_chunks::<8>();
        let (oc, ot) = out.as_chunks_mut::<8>();
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            // SAFETY: the function is compiled for x86-64-v4, which its caller checked. The
            // load and store touch exactly the eight `i64` of `i` and `o`, unaligned accesses
            // are allowed. Every index lies in `0..xs.len()` by the assertion above, so each
            // gathered address `xs + 8 * i` is a row of `xs`.
            unsafe {
                let v = _mm512_loadu_si512(i.as_ptr().cast());
                let g = _mm512_i64gather_epi64::<8>(v, xs.as_ptr());
                _mm512_storeu_si512(o.as_mut_ptr().cast(), g);
            }
        }
        for (o, &i) in ot.iter_mut().zip(it) {
            *o = xs[i as usize];
        }
        out
    }
}

fn rows_of<T: Elem>(v: &Value) -> &[T] {
    match v {
        Value::Array(a) => a.as_slice::<T>(),
        other => panic!("expected an array, got {other}"),
    }
}

/// Adds every interpreter variant of `interps` to `g`, after checking its answer to `src`
/// on `ins` against `want`.
fn interp_variants<T: Elem + PartialEq + std::fmt::Debug>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    interps: &[&Interp],
    src: &str,
    ins: &[(&str, Value)],
    want: &[T],
) {
    for it in interps {
        let p = it.interp.prepare(src).unwrap();
        let out = run(&p, &it.interp, ins);
        assert!(rows_of::<T>(&out) == want, "{src}: {}", it.label);
        g.bench_function(&it.label, |b| {
            b.iter(|| run(&p, &it.interp, black_box(ins)))
        });
    }
}

/// The tier 2 kernels at `N` rows. Each group has the scalar and auto variants, the
/// intrinsics variants where the kernel has intrinsics entries, and, for the kernels the spec
/// names, native loops per ISA. Throughput is input bytes read, as in `tier1`.
fn tier2(c: &mut Criterion) {
    let all_interps = interpreters();
    // Among the tier 2 kernels only `gather` and, on AVX-512, `scan_by_add` have intrinsics
    // entries on x86-64, so only their groups measure them. AVX2 `scan_by_add` has none, so
    // its intrinsics variant would rerun the auto entry and is left out.
    let auto: Vec<&Interp> = all_interps.iter().filter(|it| !it.intrinsics).collect();
    let all: Vec<&Interp> = all_interps.iter().collect();
    let scan_by: Vec<&Interp> = all_interps
        .iter()
        .filter(|it| !it.intrinsics || it.label == "x86-64-v4-intrinsics")
        .collect();
    let nats = natives();
    let bytes = u64::try_from(N * size_of::<i64>()).unwrap();
    let x = splitmix(N, 1, u64::cast_signed);
    let y = splitmix(N, 2, u64::cast_signed);
    let keys = splitmix(N, 3, |r| (r % BUCKETS as u64).cast_signed());
    let idx = splitmix(N, 4, |r| (r % N as u64).cast_signed());
    let starts = splitmix(N, 5, |r| r % 16 == 0);
    let xv = Value::Array(Array::from_vec(x.clone()));
    let yv = Value::Array(Array::from_vec(y.clone()));
    let kv = Value::Array(Array::from_vec(keys.clone()));
    let iv = Value::Array(Array::from_vec(idx.clone()));
    let sv = Value::Array(Array::from_bools(&starts));
    let xu: Vec<u64> = x.iter().map(|v| v.cast_unsigned()).collect();
    let xuv = Value::Array(Array::from_vec(xu.clone()));

    // The streaming reference: two rows read, one written, no dependency between rows.
    {
        let want = n_add_i64::base(&x, &y);
        for n in &nats {
            assert_eq!(call_native!(*n, n_add_i64(&x, &y)), want);
        }
        let mut g = c.benchmark_group("tier2/add-i64");
        configure(&mut g);
        g.throughput(Throughput::Bytes(2 * bytes));
        let src = "input x: i64[]; input y: i64[]; export z = x + y;";
        let ins = [("x", xv.clone()), ("y", yv.clone())];
        interp_variants(&mut g, &auto, src, &ins, &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_add_i64(black_box(&x), black_box(&y))))
            });
        }
        g.finish();
    }

    // scan
    {
        let want = n_scan_add::base(&x);
        for n in &nats {
            assert_eq!(call_native!(*n, n_scan_add(&x)), want);
        }
        let mut g = c.benchmark_group("tier2/scan-add-i64");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        let src = "input x: i64[]; export s = scan(x, add);";
        interp_variants(&mut g, &auto, src, &[("x", xv.clone())], &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_scan_add(black_box(&x))))
            });
        }
        g.finish();
    }

    // gather, random indices into the whole array
    {
        let want = n_gather::base(&x, &idx);
        for n in &nats {
            assert_eq!(call_native!(*n, n_gather(&x, &idx)), want);
        }
        let mut g = c.benchmark_group("tier2/gather-i64");
        configure(&mut g);
        // One index and one source row per output row.
        g.throughput(Throughput::Bytes(2 * bytes));
        let src = "input x: i64[]; input i: i64[]; export g = gather(x, i);";
        let ins = [("x", xv.clone()), ("i", iv.clone())];
        interp_variants(&mut g, &all, src, &ins, &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_gather(black_box(&x), black_box(&idx))))
            });
        }
        g.finish();
    }

    // histogram, keys uniform in 0..BUCKETS
    {
        let want = n_histogram::base(&keys, BUCKETS);
        for n in &nats {
            assert_eq!(call_native!(*n, n_histogram(&keys, BUCKETS)), want);
        }
        let mut g = c.benchmark_group("tier2/histogram");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        let src = format!("input k: i64[]; export h = histogram(k, {BUCKETS});");
        interp_variants(&mut g, &auto, &src, &[("k", kv.clone())], &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_histogram(black_box(&keys), BUCKETS)))
            });
        }
        g.finish();
    }

    // reduce_into, keys uniform in 0..BUCKETS
    {
        let want = n_reduce_into_add::base(&keys, &x, BUCKETS);
        for n in &nats {
            assert_eq!(
                call_native!(*n, n_reduce_into_add(&keys, &x, BUCKETS)),
                want
            );
        }
        let mut g = c.benchmark_group("tier2/reduce-into-add-i64");
        configure(&mut g);
        g.throughput(Throughput::Bytes(2 * bytes));
        let src = format!(
            "input k: i64[]; input v: i64[]; export r = reduce_into(k, v, {BUCKETS}, add);"
        );
        let ins = [("k", kv.clone()), ("v", xv.clone())];
        interp_variants(&mut g, &auto, &src, &ins, &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| {
                    call_native!(
                        *n,
                        n_reduce_into_add(black_box(&keys), black_box(&x), BUCKETS)
                    )
                })
            });
        }
        g.finish();
    }

    // div_const
    {
        let want = n_div_const::base(&x);
        for n in &nats {
            assert_eq!(call_native!(*n, n_div_const(&x)), want);
        }
        let mut g = c.benchmark_group("tier2/div-const-i64");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        let src = format!("input x: i64[]; export q = div_const(x, {DIVISOR});");
        interp_variants(&mut g, &auto, &src, &[("x", xv.clone())], &want);
        for n in &nats {
            g.bench_function(n.native().label(), |b| {
                b.iter(|| call_native!(*n, n_div_const(black_box(&x))))
            });
        }
        g.finish();
    }

    // The other new kernels, interpreter variants only, for the auto over scalar ratio.
    let xs_ins = [("x", xv.clone())];
    let xy_ins = [("x", xv.clone()), ("y", yv.clone())];
    let mut sorted: Vec<i64> = (0..N as i64).collect();
    sorted.sort_by_key(|&i| x[i as usize]);
    let others: [(&str, &str, &[(&str, Value)], Vec<i64>, u64); 5] = [
        (
            "prev-i64",
            "input x: i64[]; export p = prev(x, 0);",
            &xs_ins,
            std::iter::once(0)
                .chain(x[..N - 1].iter().copied())
                .collect(),
            bytes,
        ),
        (
            "and-i64",
            "input x: i64[]; input y: i64[]; export z = and(x, y);",
            &xy_ins,
            x.iter().zip(&y).map(|(a, b)| a & b).collect(),
            2 * bytes,
        ),
        (
            "xor-i64",
            "input x: i64[]; input y: i64[]; export z = xor(x, y);",
            &xy_ins,
            x.iter().zip(&y).map(|(a, b)| a ^ b).collect(),
            2 * bytes,
        ),
        (
            "not-i64",
            "input x: i64[]; export z = not(x);",
            &xs_ins,
            x.iter().map(|a| !a).collect(),
            bytes,
        ),
        (
            "argsort-i64",
            "input x: i64[]; export o = argsort(x);",
            &xs_ins,
            sorted,
            bytes,
        ),
    ];
    for (name, src, ins, want, b) in &others {
        let mut g = c.benchmark_group(format!("tier2/{name}"));
        configure(&mut g);
        g.throughput(Throughput::Bytes(*b));
        interp_variants(&mut g, &auto, src, ins, want);
        g.finish();
    }
    {
        let mut acc = 0i64;
        let want: Vec<i64> = x
            .iter()
            .zip(&starts)
            .enumerate()
            .map(|(i, (&v, &s))| {
                acc = if s || i == 0 { v } else { acc.wrapping_add(v) };
                acc
            })
            .collect();
        let mut g = c.benchmark_group("tier2/scan-by-add-i64");
        configure(&mut g);
        // The rows and one byte of segment starts per row.
        g.throughput(Throughput::Bytes(bytes + u64::try_from(N).unwrap()));
        let src = "input x: i64[]; input s: bool[]; export r = scan_by(x, s, add);";
        let ins = [("x", xv.clone()), ("s", sv.clone())];
        interp_variants(&mut g, &scan_by, src, &ins, &want);
        g.finish();
    }
    {
        let du = DIVISOR.cast_unsigned();
        let want: Vec<u64> = xu.iter().map(|v| v / du).collect();
        let mut g = c.benchmark_group("tier2/div-const-u64");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        let src = format!("input x: u64[]; export q = div_const(x, {du}u64);");
        interp_variants(&mut g, &auto, &src, &[("x", xuv.clone())], &want);
        g.finish();
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn crossover_group<'a>(
    c: &'a mut Criterion,
    name: &str,
    bytes: u64,
) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(name);
    g.sample_size(20);
    g.warm_up_time(Duration::from_millis(100));
    g.measurement_time(Duration::from_millis(300));
    g.throughput(Throughput::Bytes(bytes));
    g
}

/// The sweep compares the best ISA variant with native code on x86-64 and aarch64, so other
/// targets skip it.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn crossover(_c: &mut Criterion) {
    eprintln!("crossover: skipped, the sweep needs x86-64 or aarch64");
}

/// The ISA and backend of the interpreter in the sweep and the native flavor it is compared
/// with. On x86-64 that is the best of x86-64-v4 and x86-64-v3 with the `auto` backend.
#[cfg(target_arch = "x86_64")]
fn crossover_target() -> (Isa, Backend, Checked) {
    let (best, nat) = if Isa::X86_64V4.is_available() {
        (Isa::X86_64V4, Native::X86_64V4)
    } else {
        (Isa::X86_64V3, Native::X86_64V3)
    };
    assert!(best.is_available(), "the benchmark needs x86-64-v3");
    let nat = nat.check().expect("the benchmark needs x86-64-v3");
    (best, Backend::Auto, nat)
}

/// On aarch64 it is NEON with the intrinsics backend against the baseline-target native loop,
/// which already is NEON. `sum` and `mul` have no intrinsics entry, so for them the
/// interpreter runs the scalar entry. `min` has a NEON intrinsics entry.
#[cfg(target_arch = "aarch64")]
fn crossover_target() -> (Isa, Backend, Checked) {
    assert!(Isa::Neon.is_available(), "the benchmark needs neon");
    let nat = Native::Base
        .check()
        .expect("the baseline native always runs");
    (Isa::Neon, Backend::Intrinsics, nat)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn crossover(c: &mut Criterion) {
    let (best, backend, nat) = crossover_target();
    let interp = Interpreter::new(Options {
        isa: Some(best),
        backend,
        ..Options::default()
    })
    .unwrap();
    let sum = interp
        .prepare("input x: f64[]; export s = sum(x);")
        .unwrap();
    let mul = interp
        .prepare("input x: f64[]; input y: f64[]; export z = x * y;")
        .unwrap();
    // Only aarch64 has an intrinsics entry for the minimum, so only it sweeps `min`.
    let min = cfg!(target_arch = "aarch64").then(|| {
        interp
            .prepare("input x: f64[]; export m = reduce(x, min);")
            .unwrap()
    });
    let xa = rand_f64(1 << 20, 1);
    let ya = rand_f64(1 << 20, 2);
    for p in 4..=20u32 {
        let n = 1usize << p;
        let xs = xa.slice(0, n);
        let ys = ya.slice(0, n);
        let (x, y) = (xs.as_slice::<f64>(), ys.as_slice::<f64>());
        let ins1 = [("x", Value::Array(xs.clone()))];
        let ins2 = [
            ("x", Value::Array(xs.clone())),
            ("y", Value::Array(ys.clone())),
        ];
        let bytes = u64::try_from(n * size_of::<f64>()).unwrap();

        let want_sum = call_native!(nat, n_sum(x));
        assert_eq!(
            f64_scalar(&run(&sum, &interp, &ins1)).to_bits(),
            want_sum.to_bits()
        );
        let mut g = crossover_group(c, "crossover/sum", bytes);
        g.bench_with_input(criterion::BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| run(&sum, &interp, black_box(&ins1)))
        });
        g.bench_with_input(criterion::BenchmarkId::new("native", n), &n, |b, _| {
            b.iter(|| call_native!(nat, n_sum(black_box(x))))
        });
        g.finish();

        let want_mul = call_native!(nat, n_mul(x, y));
        assert_eq!(bits(f64_rows(&run(&mul, &interp, &ins2))), bits(&want_mul));
        let mut gm = crossover_group(c, "crossover/mul", 2 * bytes);
        gm.bench_with_input(criterion::BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| run(&mul, &interp, black_box(&ins2)))
        });
        gm.bench_with_input(criterion::BenchmarkId::new("native", n), &n, |b, _| {
            b.iter(|| call_native!(nat, n_mul(black_box(x), black_box(y))))
        });
        gm.finish();

        if let Some(min) = &min {
            let want_min = call_native!(nat, n_min(x));
            assert_eq!(
                f64_scalar(&run(min, &interp, &ins1)).to_bits(),
                want_min.to_bits()
            );
            let mut gn = crossover_group(c, "crossover/min", bytes);
            gn.bench_with_input(criterion::BenchmarkId::new("interp", n), &n, |b, _| {
                b.iter(|| run(min, &interp, black_box(&ins1)))
            });
            gn.bench_with_input(criterion::BenchmarkId::new("native", n), &n, |b, _| {
                b.iter(|| call_native!(nat, n_min(black_box(x))))
            });
            gn.finish();
        }
    }
}

criterion_group!(benches, tier1, tier2, crossover);
criterion_main!(benches);
