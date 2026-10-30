use super::scalar::construct::splitmix64;
use super::scalar::elementwise::{Operand, binary_into};
use super::scalar::reduce::{CanonicalProduct, CanonicalSum, canonical_product, canonical_sum};
use super::*;
use crate::value::{Arg, Array, Scalar};

fn f64s(v: &[f64]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

fn i32s(v: &[i32]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

fn bools(v: &[bool]) -> Value {
    Value::Array(Array::from_bools(v))
}

fn f64_s(v: f64) -> Value {
    Value::Scalar(Scalar::F64(v))
}

/// Calls a kernel by name through the scalar implementation.
fn call(reg: &Registry, name: &str, args: &[Value]) -> Result<Value, String> {
    let id = reg
        .by_name(name)
        .unwrap_or_else(|| panic!("no kernel {name}"));
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto);
    assert_eq!(dispatch.variant_used(id), Variant::SCALAR);
    dispatch.call(reg, id, args)
}

fn arr(v: Value) -> Array {
    match v {
        Value::Array(a) => a,
        other => panic!("expected an array, got {other:?}"),
    }
}

fn arr_ty(t: ScalarType) -> ValueType {
    ValueType::Array(t)
}

fn sc_ty(t: ScalarType) -> ValueType {
    ValueType::Scalar(t)
}

#[test]
fn lookup_resolves_array_and_scalar_kernels() {
    use ScalarType::*;
    let reg = Registry::new();
    let name = |op, args: &[ValueType]| reg.lookup(op, args).map(|id| reg.get(id).name.clone());
    assert_eq!(
        name(Op::Add, &[arr_ty(F64), arr_ty(F64)]).as_deref(),
        Some("add_f64")
    );
    assert_eq!(
        name(Op::Add, &[arr_ty(F64), sc_ty(F64)]).as_deref(),
        Some("add_f64")
    );
    assert_eq!(
        name(Op::Add, &[sc_ty(F64), arr_ty(F64)]).as_deref(),
        Some("add_f64")
    );
    assert_eq!(
        name(Op::Add, &[sc_ty(U64), sc_ty(U64)]).as_deref(),
        Some("add_u64_s")
    );
    assert_eq!(name(Op::Add, &[arr_ty(I8), arr_ty(I8)]), None);
    assert_eq!(name(Op::Add, &[arr_ty(F64), arr_ty(F32)]), None);
    assert_eq!(name(Op::Add, &[arr_ty(F64), sc_ty(F32)]), None);
    assert_eq!(name(Op::Div, &[arr_ty(I64), arr_ty(I64)]), None);
    assert_eq!(name(Op::Neg, &[arr_ty(U8)]), None);
    assert_eq!(
        name(Op::Gt, &[arr_ty(F64), sc_ty(F64)]).as_deref(),
        Some("gt_f64")
    );
    assert_eq!(name(Op::Not, &[arr_ty(Bool)]).as_deref(), Some("not_bool"));
    assert_eq!(name(Op::Not, &[sc_ty(Bool)]).as_deref(), Some("not_bool_s"));
    assert_eq!(
        name(Op::Select, &[arr_ty(Bool), arr_ty(F64), sc_ty(F64)]).as_deref(),
        Some("select_f64")
    );
    assert_eq!(
        name(Op::Select, &[sc_ty(Bool), sc_ty(F64), sc_ty(F64)]).as_deref(),
        Some("select_f64_s")
    );
    assert_eq!(
        name(Op::Select, &[sc_ty(Bool), arr_ty(F64), arr_ty(F64)]).as_deref(),
        Some("select_f64")
    );
    assert_eq!(
        name(Op::Cast(I32), &[arr_ty(F64)]).as_deref(),
        Some("cast_f64_i32")
    );
    assert_eq!(
        name(Op::Cast(F64), &[sc_ty(I8)]).as_deref(),
        Some("cast_i8_f64_s")
    );
    assert_eq!(name(Op::Cast(Bool), &[arr_ty(F64)]), None);
    assert_eq!(name(Op::Cast(F64), &[arr_ty(I8)]), None);
}

#[test]
fn lookup_is_exact_for_non_elementwise_kernels() {
    use ScalarType::*;
    // An empty registry, so the dummy kernels do not collide with the real ones.
    let mut reg = Registry {
        defs: Vec::new(),
        by_name: std::collections::HashMap::new(),
        by_op: std::collections::HashMap::new(),
    };
    fn dummy(_: &[Value]) -> Result<Value, String> {
        Err("unused".into())
    }
    let mut def = |name: &str, op, inputs: Vec<InputSpec>, output| {
        reg.add(KernelDef {
            name: name.into(),
            class: KernelClass::Whole,
            op: Some(op),
            inputs,
            output,
            cost: 0,
            commutative: false,
            pattern: None,
            impls: KernelImpl::Whole(dummy).scalar_only(),
        })
    };
    let fill = def(
        "fill_f64",
        Op::Fill,
        vec![InputSpec::Scalar(I64), InputSpec::Scalar(F64)],
        arr_ty(F64),
    );
    let iota = def("iota", Op::Iota, vec![InputSpec::Scalar(I64)], arr_ty(I64));
    assert_eq!(reg.lookup(Op::Fill, &[sc_ty(I64), sc_ty(F64)]), Some(fill));
    assert_eq!(reg.lookup(Op::Fill, &[sc_ty(I64), sc_ty(F32)]), None);
    assert_eq!(reg.lookup(Op::Fill, &[arr_ty(I64), sc_ty(F64)]), None);
    assert_eq!(reg.lookup(Op::Iota, &[sc_ty(I64)]), Some(iota));
    assert_eq!(reg.lookup(Op::Iota, &[arr_ty(I64)]), None);
}

#[test]
fn add_broadcasts_scalars() {
    let reg = Registry::new();
    let want = f64s(&[11.0, 12.0]);
    assert_eq!(
        call(&reg, "add_f64", &[f64s(&[1.0, 2.0]), f64_s(10.0)]).unwrap(),
        want
    );
    assert_eq!(
        call(&reg, "add_f64", &[f64_s(10.0), f64s(&[1.0, 2.0])]).unwrap(),
        want
    );
}

#[test]
fn length_mismatch_is_an_error() {
    let reg = Registry::new();
    let err = call(
        &reg,
        "add_f64",
        &[f64s(&[1.0, 2.0]), f64s(&[1.0, 2.0, 3.0])],
    )
    .unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");
    assert!(err.contains('2') && err.contains('3'), "{err}");
    let args = [bools(&[true]), f64s(&[1.0, 2.0]), f64s(&[1.0, 2.0])];
    let select_err = call(&reg, "select_f64", &args).unwrap_err();
    assert!(select_err.contains("length mismatch"), "{select_err}");
}

#[test]
fn integer_arithmetic_wraps() {
    let reg = Registry::new();
    let r = call(&reg, "add_i32", &[i32s(&[i32::MAX]), i32s(&[1])]).unwrap();
    assert_eq!(arr(r).as_slice::<i32>(), &[i32::MIN]);
    let u8s = |v: u8| Value::Array(Array::from_vec(vec![v]));
    let add_u8_res = call(&reg, "add_u8", &[u8s(255), u8s(1)]).unwrap();
    assert_eq!(arr(add_u8_res).as_slice::<u8>(), &[0]);
    let abs_i64_res = call(
        &reg,
        "abs_i64",
        &[Value::Array(Array::from_vec(vec![i64::MIN]))],
    )
    .unwrap();
    assert_eq!(arr(abs_i64_res).as_slice::<i64>(), &[i64::MIN]);
    let neg_i32_res = call(&reg, "neg_i32", &[i32s(&[i32::MIN, 5])]).unwrap();
    assert_eq!(arr(neg_i32_res).as_slice::<i32>(), &[i32::MIN, -5]);
}

#[test]
fn scalar_kernels_return_scalars() {
    let reg = Registry::new();
    let u64s = |v| Value::Scalar(Scalar::U64(v));
    let r = call(&reg, "add_u64_s", &[u64s(u64::MAX), u64s(2)]);
    assert_eq!(r.unwrap(), u64s(1));
    let gt_f64_s_res = call(&reg, "gt_f64_s", &[f64_s(1.0), f64_s(0.0)]);
    assert_eq!(gt_f64_s_res.unwrap(), Value::Scalar(Scalar::Bool(true)));
    let cast_bool_i64_s_res = call(
        &reg,
        "cast_bool_i64_s",
        &[Value::Scalar(Scalar::Bool(true))],
    );
    assert_eq!(cast_bool_i64_s_res.unwrap(), Value::Scalar(Scalar::I64(1)));
    let not_bool_s_res = call(&reg, "not_bool_s", &[Value::Scalar(Scalar::Bool(true))]);
    assert_eq!(not_bool_s_res.unwrap(), Value::Scalar(Scalar::Bool(false)));
    // A scalar kernel rejects arrays and an array kernel rejects all-scalar arguments.
    assert!(call(&reg, "add_f64_s", &[f64s(&[1.0]), f64_s(1.0)]).is_err());
    assert!(call(&reg, "add_f64", &[f64_s(1.0), f64_s(1.0)]).is_err());
}

#[test]
fn float_min_max_follow_ieee_minimum_maximum() {
    let reg = Registry::new();
    let r = call(
        &reg,
        "min_f64",
        &[f64s(&[f64::NAN, 1.0, -0.0]), f64s(&[1.0, f64::NAN, 0.0])],
    );
    let r = arr(r.unwrap());
    let out = r.as_slice::<f64>();
    assert!(out[0].is_nan() && out[1].is_nan());
    assert_eq!(out[2].to_bits(), (-0.0f64).to_bits());
    let max_f64_res = call(&reg, "max_f64", &[f64s(&[-0.0, 0.0]), f64s(&[0.0, -0.0])]);
    let max_f64_arr = arr(max_f64_res.unwrap());
    assert!(
        max_f64_arr
            .as_slice::<f64>()
            .iter()
            .all(|v| v.to_bits() == 0.0f64.to_bits())
    );
    let min_f64_res = call(&reg, "min_f64", &[f64s(&[0.0]), f64s(&[-0.0])]);
    assert_eq!(
        arr(min_f64_res.unwrap()).as_slice::<f64>()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    let max_f64_res2 = call(&reg, "max_f64", &[f64s(&[1.0, 2.0]), f64_s(1.5)]);
    assert_eq!(arr(max_f64_res2.unwrap()).as_slice::<f64>(), &[1.5, 2.0]);
}

#[test]
fn casts_follow_rust_as() {
    let reg = Registry::new();
    let r = call(
        &reg,
        "cast_f64_i32",
        &[f64s(&[f64::NAN, 1e10, -1e10, -1.9])],
    );
    assert_eq!(
        arr(r.unwrap()).as_slice::<i32>(),
        &[0, i32::MAX, i32::MIN, -1]
    );
    let big = Value::Array(Array::from_vec(vec![(1i64 << 32) + 5]));
    let cast_i64_i32_res = call(&reg, "cast_i64_i32", &[big]);
    assert_eq!(arr(cast_i64_i32_res.unwrap()).as_slice::<i32>(), &[5]);
    let cast_bool_i64_res = call(&reg, "cast_bool_i64", &[bools(&[true, false])]);
    assert_eq!(arr(cast_bool_i64_res.unwrap()).as_slice::<i64>(), &[1, 0]);
    let cast_i32_f64_res = call(&reg, "cast_i32_f64", &[i32s(&[-3, 4])]);
    assert_eq!(
        arr(cast_i32_f64_res.unwrap()).as_slice::<f64>(),
        &[-3.0, 4.0]
    );
}

#[test]
fn comparisons_and_select() {
    let reg = Registry::new();
    let r = arr(call(&reg, "gt_f64", &[f64s(&[1.0, f64::NAN]), f64_s(0.0)]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[1, 0]);
    let nans = f64s(&[f64::NAN, 1.0]);
    let ne_f64_res = arr(call(&reg, "ne_f64", &[nans.clone(), nans]).unwrap());
    assert_eq!(ne_f64_res.as_bool_bytes(), &[1, 0]);
    let cond = bools(&[true, false]);
    let select_f64_res = call(
        &reg,
        "select_f64",
        &[cond.clone(), f64s(&[1.0, 2.0]), f64s(&[3.0, 4.0])],
    );
    assert_eq!(arr(select_f64_res.unwrap()).as_slice::<f64>(), &[1.0, 4.0]);
    let select_f64_res2 = call(
        &reg,
        "select_f64",
        &[cond.clone(), f64s(&[1.0, 2.0]), f64_s(0.0)],
    );
    assert_eq!(arr(select_f64_res2.unwrap()).as_slice::<f64>(), &[1.0, 0.0]);
    // A scalar condition picks a whole branch.
    let no = Value::Scalar(Scalar::Bool(false));
    let select_f64_res3 = call(&reg, "select_f64", &[no, f64s(&[1.0, 2.0]), f64_s(7.0)]);
    assert_eq!(arr(select_f64_res3.unwrap()).as_slice::<f64>(), &[7.0, 7.0]);
    let select_bool_res = call(
        &reg,
        "select_bool",
        &[cond, bools(&[false, false]), bools(&[true, true])],
    );
    assert_eq!(arr(select_bool_res.unwrap()).as_bool_bytes(), &[0, 1]);
    let (a, b) = (bools(&[true, true, false]), bools(&[true, false, false]));
    let and_bool_res = arr(call(&reg, "and_bool", &[a.clone(), b.clone()]).unwrap());
    assert_eq!(and_bool_res.as_bool_bytes(), &[1, 0, 0]);
    let or_bool_res = arr(call(&reg, "or_bool", &[a, b]).unwrap());
    assert_eq!(or_bool_res.as_bool_bytes(), &[1, 1, 0]);
    let not_bool_res = arr(call(&reg, "not_bool", &[bools(&[true, false])]).unwrap());
    assert_eq!(not_bool_res.as_bool_bytes(), &[0, 1]);
}

#[test]
fn float_only_ops() {
    let reg = Registry::new();
    let r = call(&reg, "div_f64", &[f64s(&[1.0, -1.0]), f64_s(0.0)]);
    assert_eq!(
        arr(r.unwrap()).as_slice::<f64>(),
        &[f64::INFINITY, f64::NEG_INFINITY]
    );
    let sqrt_f64_res = call(&reg, "sqrt_f64", &[f64s(&[4.0, 9.0])]);
    assert_eq!(arr(sqrt_f64_res.unwrap()).as_slice::<f64>(), &[2.0, 3.0]);
    let abs_f64_res = call(&reg, "abs_f64", &[f64s(&[-1.5, 2.0])]);
    assert_eq!(arr(abs_f64_res.unwrap()).as_slice::<f64>(), &[1.5, 2.0]);
}

#[test]
fn wrong_argument_types_are_errors() {
    let reg = Registry::new();
    assert!(call(&reg, "add_f64", &[f64s(&[1.0]), i32s(&[1])]).is_err());
    assert!(call(&reg, "add_f64", &[f64s(&[1.0])]).is_err());
    assert!(call(&reg, "add_f64", &[f64s(&[1.0]), Value::Str("x".into())]).is_err());
}

#[test]
fn kernel_names_are_unique_valid_identifiers() {
    let reg = Registry::new();
    let mut seen = std::collections::HashSet::new();
    for (_, def) in reg.iter() {
        assert!(seen.insert(def.name.clone()), "duplicate {}", def.name);
        let mut chars = def.name.chars();
        assert!(
            chars.next().is_some_and(|c| c.is_ascii_lowercase()),
            "{}",
            def.name
        );
        assert!(
            chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{}",
            def.name
        );
        // Fused kernels have no op, their classes are checked by the oracle tests.
        if def.pattern.is_some() {
            continue;
        }
        let expected = match def.op {
            Some(Op::Reduce(_) | Op::Count | Op::Any | Op::All | Op::ArgMin | Op::ArgMax) => {
                KernelClass::Reduction
            }
            Some(Op::Compress | Op::Where) => KernelClass::Selection,
            Some(
                Op::Iota
                | Op::Fill
                | Op::RandI64
                | Op::RandF64
                | Op::Load(_)
                | Op::Length
                | Op::SameLen,
            ) => KernelClass::Whole,
            _ => KernelClass::Elementwise,
        };
        assert_eq!(def.class, expected, "{}", def.name);
        assert!(def.impls[Variant::SCALAR.index()].is_some());
    }
}

#[test]
fn costs_count_bytes_per_element() {
    let reg = Registry::new();
    let cost = |n: &str| reg.get(reg.by_name(n).unwrap()).cost;
    assert_eq!(cost("add_f64"), 24);
    assert_eq!(cost("add_i32"), 12);
    assert_eq!(cost("gt_f64"), 17);
    assert_eq!(cost("neg_f32"), 8);
    assert_eq!(cost("select_f64"), 1 + 16 + 8);
    assert_eq!(cost("not_bool"), 2);
    assert_eq!(cost("cast_f64_i32"), 12);
    assert_eq!(cost("add_u64_s"), 0);
}

#[test]
fn array_kernels_exist_for_v1_types_only() {
    let reg = Registry::new();
    for t in ScalarType::V1_ARRAY {
        if t != ScalarType::Bool {
            assert!(reg.by_name(&format!("add_{}", t.name())).is_some());
        }
    }
    assert!(reg.by_name("add_i8").is_none());
    assert!(reg.by_name("add_i8_s").is_some());
    assert!(reg.by_name("add_bool").is_none());
}

#[test]
fn resolve_falls_back_to_scalar_without_variants() {
    let reg = Registry::new();
    let id = reg.by_name("add_f64").unwrap();
    for isa in Isa::ALL {
        for backend in [Backend::Auto, Backend::Intrinsics] {
            assert_eq!(reg.resolve(isa, backend).variant_used(id), Variant::SCALAR);
        }
    }
}

#[test]
fn variant_index_and_display() {
    let all: Vec<Variant> = Isa::ALL
        .into_iter()
        .flat_map(|isa| {
            [Backend::Auto, Backend::Intrinsics]
                .into_iter()
                .map(move |backend| Variant { isa, backend })
        })
        .filter(|v| v.isa != Isa::Scalar || v.backend == Backend::Auto)
        .collect();
    let mut indexes: Vec<usize> = all.iter().map(|v| v.index()).collect();
    indexes.sort_unstable();
    assert_eq!(indexes, (0..Variant::COUNT).collect::<Vec<_>>());
    assert_eq!(Variant::SCALAR.to_string(), "isa=scalar");
    let avx2 = Variant {
        isa: Isa::Avx2,
        backend: Backend::Auto,
    };
    assert_eq!(avx2.to_string(), "isa=avx2 backend=auto");
    assert_eq!(Backend::default(), Backend::Intrinsics);
}

#[test]
#[should_panic(expected = "does not fit its class")]
fn add_rejects_entries_of_another_class() {
    fn entry(_: &[Arg], _: &mut [u8]) -> Result<(), String> {
        Ok(())
    }
    let mut reg = Registry::new();
    reg.add(KernelDef {
        name: "bad".into(),
        class: KernelClass::Reduction,
        op: None,
        inputs: vec![],
        output: sc_ty(ScalarType::I64),
        cost: 0,
        commutative: false,
        pattern: None,
        impls: KernelImpl::Elementwise(entry).scalar_only(),
    });
}

#[test]
fn isa_names_round_trip() {
    for isa in Isa::ALL {
        assert_eq!(Isa::from_name(isa.name()), Some(isa));
    }
    assert_eq!(Isa::from_name("sse"), None);
    assert!(Isa::Scalar.is_available());
    assert!(Isa::detect().is_available());
}

#[test]
fn tiled_calls_match_one_call() {
    let n = 1000usize;
    let a: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i).unwrap()) * 0.37 - 5.0)
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i).unwrap()).sin())
        .collect();
    let mut whole = vec![0.0f64; n];
    binary_into(
        Operand::Slice(&a),
        Operand::Slice(&b),
        &mut whole,
        |x, y| x + y,
    );
    let mut tiled = vec![0.0f64; n];
    for r in [0..37, 37..100, 100..1000] {
        let (x, y) = (
            Operand::Slice(&a).range(r.clone()),
            Operand::Slice(&b).range(r.clone()),
        );
        binary_into(x, y, &mut tiled[r], |x, y| x + y);
    }
    assert_eq!(
        bytemuck::cast_slice::<f64, u8>(&whole),
        bytemuck::cast_slice::<f64, u8>(&tiled)
    );
    // The same holds with a splat operand.
    let mut whole_splat = vec![0.0f64; n];
    binary_into(
        Operand::Slice(&a),
        Operand::Splat(2.5),
        &mut whole_splat,
        |x, y| x * y,
    );
    let mut tiled_splat = vec![0.0f64; n];
    for r in [0..37, 37..100, 100..1000] {
        let x = Operand::Slice(&a).range(r.clone());
        binary_into(x, Operand::Splat(2.5), &mut tiled_splat[r], |x, y| x * y);
    }
    assert_eq!(whole_splat, tiled_splat);
}

fn i64_s(v: i64) -> Value {
    Value::Scalar(Scalar::I64(v))
}

fn u8s(v: &[u8]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

fn scalar(v: Result<Value, String>) -> Scalar {
    match v.unwrap() {
        Value::Scalar(s) => s,
        other => panic!("expected a scalar, got {other:?}"),
    }
}

/// Test data that spans magnitudes, so the summation order changes the result.
fn varied(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let unit = (splitmix64(u64::try_from(i).unwrap()) >> 11) as f64 * (-53f64).exp2();
            (unit - 0.5) * 10f64.powi(i32::try_from(i % 9).unwrap() * 2)
        })
        .collect()
}

/// The spec's order, written bullet by bullet.
fn reference_sum(xs: &[f64]) -> f64 {
    let n = xs.len();
    let b = n - n % 16;
    let mut acc = [0.0f64; 16];
    for (j, a) in acc.iter_mut().enumerate() {
        let mut i = j;
        while i < b {
            *a += xs[i];
            i += 16;
        }
    }
    for step in [8, 4, 2, 1] {
        for j in 0..step {
            acc[j] += acc[j + step];
        }
    }
    for &x in &xs[b..] {
        acc[0] += x;
    }
    acc[0]
}

#[test]
fn canonical_sum_follows_the_spec_order() {
    // The large pair sits in one accumulator lane, so the canonical order cancels it
    // before the ones are added, while a left fold absorbs the first ones into 1e16.
    let xs: Vec<f64> = (0..35)
        .map(|i| match i {
            0 => 1e16,
            16 => -1e16,
            _ => 1.0,
        })
        .collect();
    let want = reference_sum(&xs);
    assert_eq!(canonical_sum(&xs).to_bits(), want.to_bits());
    let naive = xs.iter().fold(0.0, |a, &x| a + x);
    assert_ne!(naive.to_bits(), want.to_bits());
    let varied_xs = varied(35);
    assert_eq!(
        canonical_sum(&varied_xs).to_bits(),
        reference_sum(&varied_xs).to_bits()
    );
}

#[test]
fn canonical_empty_is_the_identity() {
    assert_eq!(canonical_sum::<f64>(&[]).to_bits(), 0.0f64.to_bits());
    assert_eq!(canonical_product::<f64>(&[]).to_bits(), 1.0f64.to_bits());
    assert_eq!(canonical_sum::<f32>(&[]).to_bits(), 0.0f32.to_bits());
}

#[test]
fn canonical_sum_f32_accumulates_in_f32() {
    let xs: Vec<f32> = (0..37)
        .map(|i| 1e7 + f32::from(i16::try_from(i).unwrap()))
        .collect();
    let mut acc = [0.0f32; 16];
    for j in 0..32 {
        acc[j % 16] += xs[j];
    }
    for step in [8, 4, 2, 1] {
        for j in 0..step {
            acc[j] += acc[j + step];
        }
    }
    for &x in &xs[32..] {
        acc[0] += x;
    }
    assert_eq!(canonical_sum(&xs).to_bits(), acc[0].to_bits());
}

#[test]
fn canonical_state_update_finish_matches_one_shot() {
    let xs = varied(1000);
    let mut s = CanonicalSum::new();
    s.update(&xs[0..64]);
    s.update(&xs[64..384]);
    s.update(&xs[384..992]);
    assert_eq!(s.finish(&xs[992..]).to_bits(), canonical_sum(&xs).to_bits());
    // The data is order sensitive, so the equality above is not vacuous.
    assert_ne!(
        xs.iter().fold(0.0, |a, &x| a + x).to_bits(),
        canonical_sum(&xs).to_bits()
    );
    let mut p = CanonicalProduct::new();
    let ys: Vec<f64> = xs.iter().map(|x| 1.0 + x.abs().sin() * 0.01).collect();
    p.update(&ys[0..640]);
    p.update(&ys[640..992]);
    assert_eq!(
        p.finish(&ys[992..]).to_bits(),
        canonical_product(&ys).to_bits()
    );
}

#[test]
#[should_panic]
fn canonical_update_rejects_partial_blocks() {
    CanonicalSum::<f64>::new().update(&[1.0; 17]);
}

#[test]
fn float_reduce_kernels_use_the_canonical_order() {
    let reg = Registry::new();
    let xs = varied(100);
    let r = scalar(call(&reg, "reduce_add_f64", &[f64s(&xs)]));
    assert_eq!(r, Scalar::F64(canonical_sum(&xs)));
    let reduce_mul_f64_res = scalar(call(&reg, "reduce_mul_f64", &[f64s(&[2.0, 3.0, 4.0])]));
    assert_eq!(reduce_mul_f64_res, Scalar::F64(24.0));
    let reduce_add_f64_res = scalar(call(&reg, "reduce_add_f64", &[f64s(&[])]));
    assert_eq!(reduce_add_f64_res, Scalar::F64(0.0));
}

#[test]
fn integer_reductions_widen_and_wrap() {
    let reg = Registry::new();
    let r = scalar(call(&reg, "reduce_add_i32", &[i32s(&[i32::MAX, 1])]));
    assert_eq!(r, Scalar::I64(2147483648));
    let reduce_add_u8_res = scalar(call(&reg, "reduce_add_u8", &[u8s(&[200, 100])]));
    assert_eq!(reduce_add_u8_res, Scalar::U64(300));
    let reduce_add_i64_res = scalar(call(
        &reg,
        "reduce_add_i64",
        &[Value::Array(Array::from_vec(vec![i64::MAX, 1]))],
    ));
    assert_eq!(reduce_add_i64_res, Scalar::I64(i64::MIN));
    let reduce_mul_u8_res = scalar(call(&reg, "reduce_mul_u8", &[u8s(&[16, 17])]));
    assert_eq!(reduce_mul_u8_res, Scalar::U8(16));
    let reduce_mul_i32_res = scalar(call(&reg, "reduce_mul_i32", &[i32s(&[])]));
    assert_eq!(reduce_mul_i32_res, Scalar::I32(1));
    let reduce_max_i32_res = scalar(call(&reg, "reduce_max_i32", &[i32s(&[3, 9, -1])]));
    assert_eq!(reduce_max_i32_res, Scalar::I32(9));
    let reduce_min_u8_res = scalar(call(&reg, "reduce_min_u8", &[u8s(&[3, 9, 1])]));
    assert_eq!(reduce_min_u8_res, Scalar::U8(1));
    let e = call(&reg, "reduce_min_i32", &[i32s(&[])]).unwrap_err();
    assert!(e.contains("empty"), "{e}");
    assert!(call(&reg, "reduce_max_u8", &[u8s(&[])]).is_err());
}

#[test]
fn float_min_max_reductions_follow_ieee_minimum() {
    let reg = Registry::new();
    let r = scalar(call(&reg, "reduce_min_f64", &[f64s(&[1.0, f64::NAN, 0.0])]));
    assert!(matches!(r, Scalar::F64(v) if v.is_nan()));
    let reduce_min_f64_res = scalar(call(&reg, "reduce_min_f64", &[f64s(&[0.0, -0.0])]));
    assert!(matches!(reduce_min_f64_res, Scalar::F64(v) if v.to_bits() == (-0.0f64).to_bits()));
    let reduce_max_f64_res = scalar(call(&reg, "reduce_max_f64", &[f64s(&[-0.0, 0.0])]));
    assert!(matches!(reduce_max_f64_res, Scalar::F64(v) if v.to_bits() == 0.0f64.to_bits()));
    let e = call(&reg, "reduce_min_f64", &[f64s(&[])]).unwrap_err();
    assert!(e.contains("empty"), "{e}");
}

#[test]
fn argmin_argmax_tie_and_nan_rules() {
    let reg = Registry::new();
    let idx = |name: &str, v: &[f64]| scalar(call(&reg, name, &[f64s(v)]));
    assert_eq!(idx("argmin_f64", &[3.0, 1.0, 1.0]), Scalar::I64(1));
    assert_eq!(
        idx("argmin_f64", &[3.0, f64::NAN, f64::NAN]),
        Scalar::I64(1)
    );
    assert_eq!(idx("argmin_f64", &[0.0, -0.0]), Scalar::I64(1));
    assert_eq!(idx("argmax_f64", &[-0.0, 0.0]), Scalar::I64(1));
    assert_eq!(idx("argmax_f64", &[1.0, 5.0, 5.0]), Scalar::I64(1));
    assert_eq!(idx("argmax_f64", &[1.0, f64::NAN, 5.0]), Scalar::I64(1));
    assert!(call(&reg, "argmax_f64", &[f64s(&[])]).is_err());
    let r = scalar(call(&reg, "argmax_i32", &[i32s(&[4, 9, 9, 2])]));
    assert_eq!(r, Scalar::I64(1));
    assert!(call(&reg, "argmin_u8", &[u8s(&[])]).is_err());
}

#[test]
fn bool_reductions() {
    let reg = Registry::new();
    let b = |name: &str, v: &[bool]| scalar(call(&reg, name, &[bools(v)]));
    assert_eq!(b("count", &[true, false, true]), Scalar::I64(2));
    assert_eq!(b("any", &[]), Scalar::Bool(false));
    assert_eq!(b("all", &[]), Scalar::Bool(true));
    assert_eq!(b("any", &[false, true]), Scalar::Bool(true));
    assert_eq!(b("all", &[true, false]), Scalar::Bool(false));
    assert_eq!(b("reduce_and_bool", &[]), Scalar::Bool(true));
    assert_eq!(b("reduce_or_bool", &[]), Scalar::Bool(false));
    assert_eq!(b("reduce_xor_bool", &[true, true]), Scalar::Bool(false));
    assert_eq!(
        b("reduce_xor_bool", &[true, true, true]),
        Scalar::Bool(true)
    );
}

#[test]
fn compress_and_where() {
    let reg = Registry::new();
    let r = call(
        &reg,
        "compress_f64",
        &[f64s(&[1.0, 2.0, 3.0]), bools(&[false, true, true])],
    );
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[2.0, 3.0]);
    let compress_f64_res = call(
        &reg,
        "compress_f64",
        &[f64s(&[1.0, 2.0]), bools(&[false, false])],
    );
    assert!(arr(compress_f64_res.unwrap()).is_empty());
    let compress_f64_res2 = call(&reg, "compress_f64", &[f64s(&[1.0, 2.0]), bools(&[true])]);
    assert!(compress_f64_res2.is_err());
    let compress_bool_res = call(
        &reg,
        "compress_bool",
        &[bools(&[true, false, true]), bools(&[true, true, false])],
    );
    assert_eq!(
        arr(compress_bool_res.unwrap()),
        Array::from_bools(&[true, false])
    );
    let compress_u8_res = call(
        &reg,
        "compress_u8",
        &[u8s(&[7, 8, 9]), bools(&[true, false, true])],
    );
    assert_eq!(arr(compress_u8_res.unwrap()).as_slice::<u8>(), &[7, 9]);
    let where_res = call(&reg, "where", &[bools(&[false, true, true])]);
    assert_eq!(arr(where_res.unwrap()).as_slice::<i64>(), &[1, 2]);
}

#[test]
fn iota_fill_and_negative_counts() {
    let reg = Registry::new();
    let r = call(&reg, "iota", &[i64_s(3)]);
    assert_eq!(arr(r.unwrap()).as_slice::<i64>(), &[0, 1, 2]);
    assert!(call(&reg, "iota", &[i64_s(-1)]).is_err());
    let fill_f64_res = call(&reg, "fill_f64", &[i64_s(2), f64_s(1.5)]);
    assert_eq!(arr(fill_f64_res.unwrap()).as_slice::<f64>(), &[1.5, 1.5]);
    let fill_bool_res = call(
        &reg,
        "fill_bool",
        &[i64_s(2), Value::Scalar(Scalar::Bool(true))],
    );
    assert_eq!(
        arr(fill_bool_res.unwrap()),
        Array::from_bools(&[true, true])
    );
    assert!(call(&reg, "fill_f64", &[i64_s(-1), f64_s(1.5)]).is_err());
    assert!(call(&reg, "rand_i64", &[i64_s(-1), i64_s(0)]).is_err());
    assert!(call(&reg, "rand_f64", &[i64_s(-1), i64_s(0)]).is_err());
    let length_f64_res = call(&reg, "length_f64", &[f64s(&[1.0, 2.0])]);
    assert_eq!(scalar(length_f64_res), Scalar::I64(2));
    let length_bool_res = call(&reg, "length_bool", &[bools(&[])]);
    assert_eq!(scalar(length_bool_res), Scalar::I64(0));
}

#[test]
fn splitmix_and_rand() {
    assert_eq!(splitmix64(0), 0xE220A8397B1DCDAF);
    let reg = Registry::new();
    let r = arr(call(&reg, "rand_i64", &[i64_s(3), i64_s(7)]).unwrap());
    assert_eq!(r.as_slice::<i64>()[1], splitmix64(8) as i64);
    let rand_f64_res = arr(call(&reg, "rand_f64", &[i64_s(1000), i64_s(1)]).unwrap());
    assert!(
        rand_f64_res
            .as_slice::<f64>()
            .iter()
            .all(|v| (0.0..1.0).contains(v))
    );
    let want = (splitmix64(1) >> 11) as f64 * (-53f64).exp2();
    assert_eq!(rand_f64_res.as_slice::<f64>()[0].to_bits(), want.to_bits());
}

fn temp_file(name: &str, bytes: &[u8]) -> String {
    let path = std::env::temp_dir().join(format!("verkehrt-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn load_reads_little_endian_files() {
    let reg = Registry::new();
    let bytes: Vec<u8> = [1.0f64, 2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = temp_file("load.f64", &bytes);
    let r = call(&reg, "load_f64", &[Value::Str(path.as_str().into())]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[1.0, 2.5]);
    let load_u8_res = call(&reg, "load_u8", &[Value::Str(path.as_str().into())]);
    assert_eq!(arr(load_u8_res.unwrap()).len(), 16);
    let load_i32_res = call(&reg, "load_i32", &[Value::Str(path.as_str().into())]);
    assert_eq!(arr(load_i32_res.unwrap()).len(), 4);
    std::fs::remove_file(&path).unwrap();

    let missing = "/nonexistent/verkehrt-missing.bin";
    let e = call(&reg, "load_f64", &[Value::Str(missing.into())]).unwrap_err();
    assert!(e.contains(missing), "{e}");

    let odd_path = temp_file("odd.bin", &[0u8; 9]);
    let odd_err = call(&reg, "load_f64", &[Value::Str(odd_path.as_str().into())]).unwrap_err();
    assert!(odd_err.contains("multiple of 8"), "{odd_err}");
    std::fs::remove_file(&odd_path).unwrap();
    assert!(reg.by_name("load_bool").is_none());
}

#[test]
fn new_kernels_resolve_through_lookup_with_classes_and_costs() {
    use ScalarType::*;
    let reg = Registry::new();
    let name = |op, args: &[ValueType]| reg.lookup(op, args).map(|id| reg.get(id).name.clone());
    assert_eq!(
        name(Op::Reduce(crate::ops::ReduceOp::Add), &[arr_ty(I32)]).as_deref(),
        Some("reduce_add_i32")
    );
    assert_eq!(
        name(Op::Fill, &[sc_ty(I64), sc_ty(F64)]).as_deref(),
        Some("fill_f64")
    );
    assert_eq!(name(Op::Iota, &[sc_ty(I64)]).as_deref(), Some("iota"));
    assert_eq!(
        name(Op::Load(F32), &[ValueType::Str]).as_deref(),
        Some("load_f32")
    );
    assert_eq!(name(Op::Fill, &[sc_ty(I64), sc_ty(I8)]), None);
    let def = |n: &str| reg.get(reg.by_name(n).unwrap());
    let cost = |n: &str| def(n).cost;
    assert_eq!(cost("reduce_add_f64"), 8);
    assert_eq!(cost("reduce_add_u8"), 1);
    assert_eq!((cost("count"), cost("any"), cost("all")), (1, 1, 1));
    assert_eq!(cost("argmin_f32"), 4);
    assert_eq!(cost("compress_f64"), 17);
    assert_eq!(cost("compress_bool"), 3);
    assert_eq!(cost("where"), 9);
    assert_eq!(
        (cost("iota"), cost("rand_i64"), cost("rand_f64")),
        (8, 8, 8)
    );
    assert_eq!(
        (cost("fill_i32"), cost("load_f64"), cost("length_u8")),
        (4, 8, 0)
    );
    assert_eq!(def("compress_f64").class, KernelClass::Selection);
    assert_eq!(def("reduce_add_f64").class, KernelClass::Reduction);
    assert_eq!(def("iota").class, KernelClass::Whole);
    assert_eq!(def("reduce_add_i32").output, sc_ty(I64));
    assert_eq!(def("reduce_add_u8").output, sc_ty(U64));
}
