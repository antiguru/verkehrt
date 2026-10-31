//! The tier 1 workloads in corgi, the closest prior art, for comparison with `tier1.rs`.
//!
//! Corgi has no flat column type. A column is one row holding the whole column as a list, the
//! form its own benchmarks use (`one_row` in corgi's `benches/idioms.rs`), so reductions and
//! `filter` run per row over the list. Floats are stored as total-order keys and summed in row
//! order. Inputs are the same `rand_f64(10_000_000, 1 or 2)` arrays as `tier1.rs`. Timing covers
//! `Program::run` only, with the program compiled once. The input `Value` is cloned per
//! iteration, an `Arc` bump, and the output is dropped inside the timed closure.

use std::hint::black_box;
use std::io::sink;
use std::time::Duration;

use corgi::{Bounds, Program, Value as CValue};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use verkehrt::kernels::Isa;
use verkehrt::value::{Array, Scalar, Value};
use verkehrt::{Interpreter, Options};

const N: usize = 10_000_000;

/// `n` uniform doubles in `[0, 1)`, produced by the interpreter's own `rand_f64`, as in `tier1.rs`.
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

/// The Verkehrt (scalar ISA) result of a program over the given inputs.
fn reference(src: &str, inputs: &[(&str, Value)]) -> Value {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    })
    .unwrap();
    let p = interp.prepare(src).unwrap();
    p.execute(&interp, inputs, &mut sink())
        .unwrap()
        .pop()
        .unwrap()
        .1
}

fn f64_of(v: Value) -> f64 {
    match v {
        Value::Scalar(Scalar::F64(x)) => x,
        other => panic!("expected f64, got {other}"),
    }
}

fn i64_of(v: Value) -> i64 {
    match v {
        Value::Scalar(Scalar::I64(x)) => x,
        other => panic!("expected i64, got {other}"),
    }
}

/// A corgi column of one row holding `inner` as a list.
fn one_row(inner: CValue) -> CValue {
    CValue::List(Bounds::offsets(vec![N]), Box::new(inner))
}

fn compile(src: &str) -> Program {
    Program::compile_ml(src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn configure(g: &mut BenchmarkGroup<'_, WallTime>) {
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
}

/// The single `f64` of a one-row corgi result.
fn corgi_f64(v: &CValue) -> f64 {
    let xs = v.as_f64("corgi result").unwrap();
    assert_eq!(xs.len(), 1);
    xs[0]
}

/// The single `i64` of a one-row corgi result.
fn corgi_i64(v: &CValue) -> i64 {
    let xs = v.as_i64("corgi result").unwrap();
    assert_eq!(xs.len(), 1);
    xs[0]
}

/// The elements of a one-row corgi list of floats.
fn corgi_rows(v: CValue) -> Vec<f64> {
    let (bounds, inner) = v.into_list("corgi result").unwrap();
    assert_eq!(bounds.to_vec().len(), 1);
    inner.as_f64("corgi rows").unwrap()
}

fn bench(g: &mut BenchmarkGroup<'_, WallTime>, p: &Program, input: &CValue) {
    g.bench_function("corgi", |b| {
        b.iter(|| black_box(p.run(black_box(input.clone()))))
    });
}

fn corgi(c: &mut Criterion) {
    let xa = rand_f64(N, 1);
    let ya = rand_f64(N, 2);
    let x = xa.as_slice::<f64>();
    let xv = Value::Array(xa.clone());
    let yv = Value::Array(ya.clone());
    let cx = one_row(CValue::f64(x.to_vec()));
    let cxy = one_row(CValue::Prod(vec![
        CValue::f64(x.to_vec()),
        CValue::f64(ya.as_slice::<f64>().to_vec()),
    ]));
    let bytes = u64::try_from(N * size_of::<f64>()).unwrap();

    // sum. Corgi sums in row order, so the last bits may differ from Verkehrt's canonical order.
    // The difference is recorded, not asserted.
    {
        let want = f64_of(reference(
            "input x: f64[]; export s = sum(x);",
            &[("x", xv.clone())],
        ));
        let p = compile("input fold_add");
        let got = corgi_f64(&p.run(cx.clone()));
        eprintln!(
            "sum: verkehrt {want:?} corgi {got:?} bit xor {:#x} abs diff {:e}",
            want.to_bits() ^ got.to_bits(),
            (want - got).abs()
        );
        let mut g = c.benchmark_group("corgi/sum");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        bench(&mut g, &p, &cx);
        g.finish();
    }

    // dot
    {
        let want = f64_of(reference(
            "input x: f64[]; input y: f64[]; export s = sum(x * y);",
            &[("x", xv.clone()), ("y", yv.clone())],
        ));
        let p = compile("input map (r -> (r.0, r.1) mul) fold_add");
        let got = corgi_f64(&p.run(cxy.clone()));
        eprintln!(
            "dot: verkehrt {want:?} corgi {got:?} bit xor {:#x} abs diff {:e}",
            want.to_bits() ^ got.to_bits(),
            (want - got).abs()
        );
        let mut g = c.benchmark_group("corgi/dot");
        configure(&mut g);
        g.throughput(Throughput::Bytes(2 * bytes));
        bench(&mut g, &p, &cxy);
        g.finish();
    }

    // count
    {
        let cut = 0.5f64;
        let want = i64_of(reference(
            &format!("input x: f64[]; export n = count(x > {cut:?});"),
            &[("x", xv.clone())],
        ));
        let p = compile(&format!("input map (v -> (v, {cut:?}) gt) fold_add"));
        assert_eq!(corgi_i64(&p.run(cx.clone())), want, "count");
        let mut g = c.benchmark_group("corgi/count");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        bench(&mut g, &p, &cx);
        g.finish();
    }

    // min. Corgi has `fold_min`, so it runs the same reduction as Verkehrt's `reduce(x, min)`.
    // The data has no NaN, so the minimum is the same f64 in both.
    {
        let want = f64_of(reference(
            "input x: f64[]; export m = reduce(x, min);",
            &[("x", xv.clone())],
        ));
        let p = compile("input fold_min");
        let got = corgi_f64(&p.run(cx.clone()));
        assert_eq!(got.to_bits(), want.to_bits(), "min");
        let mut g = c.benchmark_group("corgi/min");
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        bench(&mut g, &p, &cx);
        g.finish();
    }

    // argmin is not benched. Corgi's operator table (`src/frontend/mod.rs`) has `fold_min`,
    // `fold_max`, and `fold_add` but no argmin or argmax, and the first-index rule would need a
    // composition of `fold_min`, `eq`, and `filter` that is a different algorithm.

    // filter at three selectivities, compared bitwise against Verkehrt's rows
    for (name, cut) in [
        ("filter-1pct", 0.99f64),
        ("filter-50pct", 0.5),
        ("filter-99pct", 0.01),
    ] {
        let want = match reference(
            &format!("input x: f64[]; export f = filter(x, |v| v > {cut:?});"),
            &[("x", xv.clone())],
        ) {
            Value::Array(a) => a.as_slice::<f64>().to_vec(),
            other => panic!("expected an array, got {other}"),
        };
        let p = compile(&format!("input map (e -> ((e, {cut:?}) gt, e)) filter"));
        let got = corgi_rows(p.run(cx.clone()));
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}"
        );
        eprintln!("{name}: {} rows match", want.len());
        let mut g = c.benchmark_group(format!("corgi/{name}"));
        configure(&mut g);
        g.throughput(Throughput::Bytes(bytes));
        bench(&mut g, &p, &cx);
        g.finish();
    }
}

criterion_group!(benches, corgi);
criterion_main!(benches);
