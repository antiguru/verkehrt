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
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use verkehrt::kernels::{Backend, Isa};
use verkehrt::value::{Array, Scalar, Value};
use verkehrt::{Interpreter, Options, Prepared};

const N: usize = 10_000_000;
const LANES: usize = 16;

/// A native baseline compiled for one target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Native {
    /// Baseline x86-64, comparable with the scalar kernels.
    Base,
    Avx2,
    Avx512,
}

impl Native {
    fn label(self) -> &'static str {
        match self {
            Native::Base => "native-base",
            Native::Avx2 => "native-avx2",
            Native::Avx512 => "native-avx512",
        }
    }

    fn available(self) -> bool {
        match self {
            Native::Base => true,
            Native::Avx2 => Isa::Avx2.is_available(),
            Native::Avx512 => Isa::Avx512.is_available(),
        }
    }
}

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

#[inline(always)]
fn mul_body(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(a, b)| a * b).collect()
}

/// Defines a module with `base`, `avx2`, and `avx512` functions around one shared body.
macro_rules! native_fns {
    ($name:ident, $body:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        mod $name {
            use super::*;
            pub fn base($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
            #[target_feature(enable = "avx2")]
            pub fn avx2($($arg: $ty),*) -> $ret {
                $body($($arg),*)
            }
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
native_fns!(n_mul, mul_body, (x: &[f64], y: &[f64]) -> Vec<f64>);

/// Calls the `Native` flavor of a function module. The availability check is not repeated
/// here so that it stays out of timed closures, callers check it once up front.
macro_rules! call_native {
    ($nat:expr, $m:ident ( $($arg:expr),* )) => {{
        match $nat {
            Native::Base => $m::base($($arg),*),
            // SAFETY: every `Native` that reaches this macro comes from `natives()` or the
            // `available` assert in `crossover`, both before any timing, so avx2 was detected.
            // The arguments are ordinary safe references, so only the CPU feature is required.
            Native::Avx2 => unsafe { $m::avx2($($arg),*) },
            // SAFETY: as above, `available` checked avx512f, avx512bw, avx512vl, and
            // avx512vbmi2. The arguments are ordinary safe references.
            Native::Avx512 => unsafe { $m::avx512($($arg),*) },
        }
    }};
}

/// An interpreter variant.
struct Interp {
    label: String,
    interp: Interpreter,
    /// Whether it uses the intrinsics backend. Only the selections have intrinsics entries,
    /// so only the `filter` groups measure these variants.
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
    v
}

fn natives() -> Vec<Native> {
    // Only available flavors are returned, which is what `call_native!` relies on.
    [Native::Base, Native::Avx2, Native::Avx512]
        .into_iter()
        .filter(|n| n.available())
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
    // Every workload but `filter` resolves the intrinsics backend to its auto entries.
    let interps: Vec<&Interp> = all_interps.iter().filter(|it| !it.intrinsics).collect();
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
            g.bench_function(n.label(), |b| {
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
            g.bench_function(n.label(), |b| {
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
            g.bench_function(n.label(), |b| {
                b.iter(|| call_native!(*n, n_count(black_box(x), cut)))
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
            g.bench_function(n.label(), |b| {
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

fn crossover(c: &mut Criterion) {
    let (best, nat) = if Isa::Avx512.is_available() {
        (Isa::Avx512, Native::Avx512)
    } else {
        (Isa::Avx2, Native::Avx2)
    };
    assert!(
        best.is_available() && nat.available(),
        "the benchmark needs avx2"
    );
    let interp = Interpreter::new(Options {
        isa: Some(best),
        backend: Backend::Auto,
        ..Options::default()
    })
    .unwrap();
    let sum = interp
        .prepare("input x: f64[]; export s = sum(x);")
        .unwrap();
    let mul = interp
        .prepare("input x: f64[]; input y: f64[]; export z = x * y;")
        .unwrap();
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
        g.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| run(&sum, &interp, black_box(&ins1)))
        });
        g.bench_with_input(BenchmarkId::new("native", n), &n, |b, _| {
            b.iter(|| call_native!(nat, n_sum(black_box(x))))
        });
        g.finish();

        let want_mul = call_native!(nat, n_mul(x, y));
        assert_eq!(bits(f64_rows(&run(&mul, &interp, &ins2))), bits(&want_mul));
        let mut gm = crossover_group(c, "crossover/mul", 2 * bytes);
        gm.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| run(&mul, &interp, black_box(&ins2)))
        });
        gm.bench_with_input(BenchmarkId::new("native", n), &n, |b, _| {
            b.iter(|| call_native!(nat, n_mul(black_box(x), black_box(y))))
        });
        gm.finish();
    }
}

criterion_group!(benches, tier1, crossover);
criterion_main!(benches);
