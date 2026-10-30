use super::*;
use crate::kernels::scalar::construct::splitmix64;
use crate::kernels::{Backend, Isa, KernelClass, Reducer};
use crate::value::ScalarType;

/// Finite, distinct values so that equal bytes mean equal results.
fn f64s(n: usize, scale: f64) -> Array {
    Array::from_vec((0..n).map(|i| (i as f64 - 400.0) * scale).collect())
}

fn bools(n: usize) -> Array {
    Array::from_bools(&(0..n).map(|i| i % 3 == 0).collect::<Vec<_>>())
}

fn bytes_of(v: &Value) -> Vec<u8> {
    let Value::Array(a) = v else {
        panic!("expected an array, got {v:?}");
    };
    (0..a.len())
        .flat_map(|i| match a.elem() {
            ScalarType::F64 => a.as_slice::<f64>()[i].to_ne_bytes().to_vec(),
            ScalarType::I32 => a.as_slice::<i32>()[i].to_ne_bytes().to_vec(),
            ScalarType::Bool => vec![a.as_bool_bytes()[i]],
            other => panic!("unexpected element type {other:?}"),
        })
        .collect()
}

/// The arguments of the tile `start..end`: zero-copy windows of the arrays, scalars as is.
fn tile_args(args: &[Value], start: usize, end: usize) -> Vec<Arg> {
    args.iter()
        .map(|a| match a {
            Value::Array(a) => Arg::Rows(a.slice(start, end - start)),
            Value::Scalar(s) => Arg::Splat(s.clone()),
            Value::Str(_) => unreachable!(),
        })
        .collect()
}

/// Runs the registered scalar entry over the tiles `0..37`, `37..100`, `100..n` into one
/// destination, and compares it with one `Dispatch::call` over all rows.
fn assert_tiled_matches_whole(name: &str, args: &[Value], n: usize) {
    let reg = Registry::new();
    let id = reg.by_name(name).unwrap();
    let KernelImpl::Elementwise(f) = reg.get(id).impls[Variant::SCALAR.index()].unwrap() else {
        panic!("{name} has no elementwise scalar entry");
    };
    let ValueType::Array(out_ty) = reg.get(id).output else {
        panic!("{name} is not an array kernel");
    };
    let w = out_ty.width();
    let tiled = Array::build_bytes(n, out_ty, |out| {
        for (start, end) in [(0, 37), (37, 100), (100, n)] {
            let tile = tile_args(args, start, end);
            // SAFETY: the scalar entry is a safe function and needs no CPU feature.
            unsafe { f(&tile, &mut out[start * w..end * w]) }?;
        }
        Ok(())
    })
    .unwrap();
    let whole = reg
        .resolve(Isa::Scalar, Backend::Auto)
        .call(&reg, id, args)
        .unwrap();
    assert_eq!(bytes_of(&Value::Array(tiled)), bytes_of(&whole), "{name}");
}

#[test]
fn tiled_elementwise_matches_whole() {
    let n = 1000;
    let splat = Value::Scalar(Scalar::F64(1.5));
    assert_tiled_matches_whole("add_f64", &[Value::Array(f64s(n, 0.25)), splat.clone()], n);
    assert_tiled_matches_whole(
        "gt_f64",
        &[Value::Array(f64s(n, 0.25)), Value::Array(f64s(n, -0.5))],
        n,
    );
    assert_tiled_matches_whole(
        "select_f64",
        &[Value::Array(bools(n)), Value::Array(f64s(n, 2.0)), splat],
        n,
    );
    assert_tiled_matches_whole("cast_f64_i32", &[Value::Array(f64s(n, 0.75))], n);
}

#[test]
fn call_checks_lengths() {
    let reg = Registry::new();
    let id = reg.by_name("add_f64").unwrap();
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto);
    let err = dispatch
        .call(
            &reg,
            id,
            &[Value::Array(f64s(2, 1.0)), Value::Array(f64s(3, 1.0))],
        )
        .unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");
}

#[test]
fn call_checks_arity_and_kinds() {
    let reg = Registry::new();
    let id = reg.by_name("add_f64").unwrap();
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto);
    let one = [Value::Array(f64s(2, 1.0))];
    let err = dispatch.call(&reg, id, &one).unwrap_err();
    assert!(err.contains("expected 2 arguments"), "{err}");
    let mixed = [Value::Array(f64s(2, 1.0)), Value::Scalar(Scalar::I64(1))];
    let kind_err = dispatch.call(&reg, id, &mixed).unwrap_err();
    assert!(kind_err.contains("expected f64"), "{kind_err}");
}

#[test]
fn resolve_falls_back_to_scalar() {
    let reg = Registry::new();
    let id = reg.by_name("add_f64").unwrap();
    // No ISA variants exist yet, so every request lands on scalar.
    let dispatch = reg.resolve(Isa::Avx2, Backend::Intrinsics);
    assert_eq!(dispatch.variant_used(id), Variant::SCALAR);
}

#[test]
fn dispatch_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Dispatch>();
}

/// A random scalar of type `ty` for stream `seed` at position `i`. Floats span several
/// magnitudes so that the summation order matters, integers stay small so that comparisons
/// against a random scalar select some rows and not others.
fn random_scalar(ty: ScalarType, seed: u64, i: usize) -> Scalar {
    let r = splitmix64(seed.wrapping_add(u64::try_from(i).unwrap()));
    let unit = (r >> 11) as f64 * (-53f64).exp2();
    let scaled = (unit - 0.5) * 10f64.powi(i32::try_from(i % 5).unwrap());
    let small = i64::try_from(r % 200).unwrap() - 100;
    match ty {
        ScalarType::F64 => Scalar::F64(scaled),
        ScalarType::F32 => Scalar::F32(scaled as f32),
        ScalarType::I64 => Scalar::I64(small),
        ScalarType::I32 => Scalar::I32(i32::try_from(small).unwrap()),
        ScalarType::U8 => Scalar::U8(u8::try_from(r % 200).unwrap()),
        ScalarType::Bool => Scalar::Bool(r & 1 == 1),
        other => panic!("no random data for {other:?}"),
    }
}

/// `n` random rows of type `ty`. A float array gets a NaN at row `nan_at` when given.
fn random_array(ty: ScalarType, n: usize, seed: u64, nan_at: Option<usize>) -> Array {
    let rows = (0..n).map(|i| random_scalar(ty, seed, i));
    let is_nan_row = |i: usize| nan_at == Some(i);
    match ty {
        ScalarType::F64 => Array::from_vec(
            rows.enumerate()
                .map(|(i, s)| match s {
                    Scalar::F64(_) if is_nan_row(i) => f64::NAN,
                    Scalar::F64(v) => v,
                    _ => unreachable!(),
                })
                .collect(),
        ),
        ScalarType::F32 => Array::from_vec(
            rows.enumerate()
                .map(|(i, s)| match s {
                    Scalar::F32(_) if is_nan_row(i) => f32::NAN,
                    Scalar::F32(v) => v,
                    _ => unreachable!(),
                })
                .collect(),
        ),
        ScalarType::I64 => Array::from_vec(
            rows.map(|s| match s {
                Scalar::I64(v) => v,
                _ => unreachable!(),
            })
            .collect(),
        ),
        ScalarType::I32 => Array::from_vec(
            rows.map(|s| match s {
                Scalar::I32(v) => v,
                _ => unreachable!(),
            })
            .collect(),
        ),
        ScalarType::U8 => Array::from_vec(
            rows.map(|s| match s {
                Scalar::U8(v) => v,
                _ => unreachable!(),
            })
            .collect(),
        ),
        ScalarType::Bool => {
            Array::from_bools(&rows.map(|s| s == Scalar::Bool(true)).collect::<Vec<_>>())
        }
        other => panic!("no random data for {other:?}"),
    }
}

/// Random arguments for a kernel's inputs, with `n` rows per array.
fn random_args(inputs: &[InputSpec], n: usize, nan_at: Option<usize>) -> Vec<Value> {
    inputs
        .iter()
        .enumerate()
        .map(|(k, spec)| {
            let seed = 1_000_003 * (u64::try_from(k).unwrap() + 1);
            match *spec {
                InputSpec::Array(ty) => Value::Array(random_array(ty, n, seed, nan_at)),
                InputSpec::Scalar(ty) => Value::Scalar(random_scalar(ty, seed, 7)),
                other => panic!("no random data for {other:?}"),
            }
        })
        .collect()
}

/// Equal bits, except that any NaN equals any NaN.
fn same_scalar(a: &Scalar, b: &Scalar) -> bool {
    match (a, b) {
        (Scalar::F64(x), Scalar::F64(y)) => {
            (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
        }
        (Scalar::F32(x), Scalar::F32(y)) => {
            (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
        }
        _ => a == b,
    }
}

fn same_array(a: &Array, b: &Array) -> bool {
    a.elem() == b.elem()
        && a.len() == b.len()
        && (0..a.len()).all(|i| same_scalar(&a.get(i), &b.get(i)))
}

#[test]
fn tiled_reductions_match_whole() {
    let reg = Registry::new();
    let mut checked = 0;
    for (_, def) in reg
        .iter()
        .filter(|(_, d)| d.class == KernelClass::Reduction)
    {
        let KernelImpl::Reduction(make) = def.impls[Variant::SCALAR.index()].unwrap() else {
            panic!("{} has no reduction scalar entry", def.name);
        };
        for nan_at in [None, Some(700)] {
            let args = random_args(&def.inputs, 1000, nan_at);
            let run = |tiles: &[(usize, usize)]| {
                // SAFETY: the scalar entry is a safe function and needs no CPU feature.
                let mut reducer: Box<dyn Reducer> = unsafe { make() };
                for &(start, end) in tiles {
                    reducer.update(&tile_args(&args, start, end)).unwrap();
                }
                reducer.finish()
            };
            let whole = run(&[(0, 1000)]);
            // Every tile but the last is a multiple of 16 rows.
            let tiled = run(&[(0, 64), (64, 384), (384, 992), (992, 1000)]);
            let ok = match (&whole, &tiled) {
                (Ok(w), Ok(t)) => same_scalar(w, t),
                (w, t) => w == t,
            };
            assert!(ok, "{} nan_at={nan_at:?}: {whole:?} vs {tiled:?}", def.name);
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} reductions checked");
}

#[test]
fn tiled_selections_match_whole() {
    let reg = Registry::new();
    let mut checked = 0;
    for (_, def) in reg
        .iter()
        .filter(|(_, d)| d.class == KernelClass::Selection)
    {
        let KernelImpl::Selection(f) = def.impls[Variant::SCALAR.index()].unwrap() else {
            panic!("{} has no selection scalar entry", def.name);
        };
        let ValueType::Array(out_ty) = def.output else {
            panic!("{} does not produce an array", def.name);
        };
        for nan_at in [None, Some(700)] {
            let args = random_args(&def.inputs, 1000, nan_at);
            let run = |tiles: &[(usize, usize)]| {
                let mut out = ArrayBuilder::new(out_ty);
                for &(start, end) in tiles {
                    // SAFETY: the scalar entry is a safe function and needs no CPU feature.
                    unsafe { f(&tile_args(&args, start, end), start, &mut out) }.unwrap();
                }
                out.finish()
            };
            let whole = run(&[(0, 1000)]);
            let tiled = run(&[(0, 37), (37, 100), (100, 1000)]);
            assert!(same_array(&whole, &tiled), "{} nan_at={nan_at:?}", def.name);
            checked += 1;
        }
    }
    assert!(checked >= 2 * 13, "only {checked} selections checked");
}

#[test]
fn call_rejects_another_registry() {
    let reg = Registry::new();
    let other = Registry::new();
    let id = reg.by_name("reduce_add_f64").unwrap();
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto);
    let args = [Value::Array(f64s(4, 1.0))];
    assert!(dispatch.call(&reg, id, &args).is_ok());
    let err = dispatch.call(&other, id, &args).unwrap_err();
    assert!(err.contains("different registry"), "{err}");
}
