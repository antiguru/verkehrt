//! Tier 1 kernels against native Rust loops, and the dispatch crossover sweep.
//!
//! Groups are `tier1/<program>/<variant>` and `crossover/<program>/<n>`. Interpreter variants
//! measure `Prepared::execute` only. Native baselines are plain loops compiled for the baseline
//! target and, per ISA, under the same `#[target_feature]` set as the ISA's kernels.
//! Throughput is input bytes read, so criterion's bytes/s are the input bandwidth.

use std::hint::black_box;
use std::io::sink;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use verkehrt::kernels::{Backend, Isa};
use verkehrt::value::{Array, Scalar, Value};
use verkehrt::{Interpreter, Options, Prepared};

const N: usize = 10_000_000;
const LANES: usize = 16;

/// A native baseline compiled for one target. The ISA flavors exist on x86-64 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Native {
    /// Baseline target, comparable with the scalar kernels.
    Base,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Native {
    fn label(self) -> &'static str {
        match self {
            Native::Base => "native-base",
            #[cfg(target_arch = "x86_64")]
            Native::Avx2 => "native-avx2",
            #[cfg(target_arch = "x86_64")]
            Native::Avx512 => "native-avx512",
        }
    }

    fn available(self) -> bool {
        match self {
            Native::Base => true,
            #[cfg(target_arch = "x86_64")]
            Native::Avx2 => Isa::Avx2.is_available(),
            #[cfg(target_arch = "x86_64")]
            Native::Avx512 => Isa::Avx512.is_available(),
        }
    }

    /// Every flavor compiled for this target, available or not.
    fn all() -> Vec<Native> {
        vec![
            Native::Base,
            #[cfg(target_arch = "x86_64")]
            Native::Avx2,
            #[cfg(target_arch = "x86_64")]
            Native::Avx512,
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
#[inline(always)]
fn mul_body(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(a, b)| a * b).collect()
}

/// The minimum of the rows, one running minimum per accumulator so that it vectorizes. Minimum
/// is order-independent on the NaN-free bench data, so no canonical order is needed.
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
/// first index overall. The bench data is NaN-free.
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

/// Defines a module with `base`, `avx2`, and `avx512` functions around one shared body. The
/// `avx2` and `avx512` functions exist on x86-64 only.
macro_rules! native_fns {
    ($name:ident, $body:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        mod $name {
            use super::*;
            pub fn base($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            pub fn avx2($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            pub fn avx512($($arg: $ty),*) -> $ret {
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
            // `available` detected avx2. The arguments are ordinary safe references, so only
            // the CPU feature is required.
            #[cfg(target_arch = "x86_64")]
            Native::Avx2 => unsafe { $m::avx2($($arg),*) },
            // SAFETY: as above, `Native::check` returned the `Checked` because `available`
            // detected avx512f, avx512bw, avx512vl, and avx512vbmi2. The arguments are
            // ordinary safe references.
            #[cfg(target_arch = "x86_64")]
            Native::Avx512 => unsafe { $m::avx512($($arg),*) },
        }
    }};
}

/// An interpreter variant.
struct Interp {
    label: String,
    interp: Interpreter,
    /// Whether it uses the intrinsics backend. On x86-64 only the selections have intrinsics
    /// entries, so only the `filter` groups measure these variants. On aarch64 the selections
    /// and the float extremes (`min`, `argmin`) have them. Elsewhere the intrinsics backend
    /// resolves to the scalar entries, so those groups skip it.
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
    for (isa, name) in [(Isa::Avx2, "avx2"), (Isa::Avx512, "avx512")] {
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
/// with. On x86-64 that is the best of AVX-512 and AVX2 with the `auto` backend.
#[cfg(target_arch = "x86_64")]
fn crossover_target() -> (Isa, Backend, Checked) {
    let (best, nat) = if Isa::Avx512.is_available() {
        (Isa::Avx512, Native::Avx512)
    } else {
        (Isa::Avx2, Native::Avx2)
    };
    assert!(best.is_available(), "the benchmark needs avx2");
    let nat = nat.check().expect("the benchmark needs avx2");
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

criterion_group!(benches, tier1, crossover);
criterion_main!(benches);
