use super::*;
use crate::kernels::{Backend, Isa};
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
            let tile: Vec<Arg> = args
                .iter()
                .map(|a| match a {
                    Value::Array(a) => Arg::Rows(a.slice(start, end - start)),
                    Value::Scalar(s) => Arg::Splat(s.clone()),
                    Value::Str(_) => unreachable!(),
                })
                .collect();
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
