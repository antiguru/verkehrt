use super::scalar::elementwise::{Operand, binary_into};
use super::*;
use crate::value::{Array, Scalar};

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
    let (f, isa) = reg.select_impl(id, Isa::Scalar);
    assert_eq!(isa, Isa::Scalar);
    f(args)
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
    let mut reg = Registry::new();
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
            impls: [Some(dummy as KernelFn), None, None, None],
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
    let err = call(&reg, "select_f64", &args).unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");
}

#[test]
fn integer_arithmetic_wraps() {
    let reg = Registry::new();
    let r = call(&reg, "add_i32", &[i32s(&[i32::MAX]), i32s(&[1])]).unwrap();
    assert_eq!(arr(r).as_slice::<i32>(), &[i32::MIN]);
    let u8s = |v: u8| Value::Array(Array::from_vec(vec![v]));
    let r = call(&reg, "add_u8", &[u8s(255), u8s(1)]).unwrap();
    assert_eq!(arr(r).as_slice::<u8>(), &[0]);
    let r = call(
        &reg,
        "abs_i64",
        &[Value::Array(Array::from_vec(vec![i64::MIN]))],
    )
    .unwrap();
    assert_eq!(arr(r).as_slice::<i64>(), &[i64::MIN]);
    let r = call(&reg, "neg_i32", &[i32s(&[i32::MIN, 5])]).unwrap();
    assert_eq!(arr(r).as_slice::<i32>(), &[i32::MIN, -5]);
}

#[test]
fn scalar_kernels_return_scalars() {
    let reg = Registry::new();
    let u64s = |v| Value::Scalar(Scalar::U64(v));
    let r = call(&reg, "add_u64_s", &[u64s(u64::MAX), u64s(2)]);
    assert_eq!(r.unwrap(), u64s(1));
    let r = call(&reg, "gt_f64_s", &[f64_s(1.0), f64_s(0.0)]);
    assert_eq!(r.unwrap(), Value::Scalar(Scalar::Bool(true)));
    let r = call(
        &reg,
        "cast_bool_i64_s",
        &[Value::Scalar(Scalar::Bool(true))],
    );
    assert_eq!(r.unwrap(), Value::Scalar(Scalar::I64(1)));
    let r = call(&reg, "not_bool_s", &[Value::Scalar(Scalar::Bool(true))]);
    assert_eq!(r.unwrap(), Value::Scalar(Scalar::Bool(false)));
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
    let r = call(&reg, "max_f64", &[f64s(&[-0.0, 0.0]), f64s(&[0.0, -0.0])]);
    let r = arr(r.unwrap());
    assert!(
        r.as_slice::<f64>()
            .iter()
            .all(|v| v.to_bits() == 0.0f64.to_bits())
    );
    let r = call(&reg, "min_f64", &[f64s(&[0.0]), f64s(&[-0.0])]);
    assert_eq!(
        arr(r.unwrap()).as_slice::<f64>()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    let r = call(&reg, "max_f64", &[f64s(&[1.0, 2.0]), f64_s(1.5)]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[1.5, 2.0]);
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
    let r = call(&reg, "cast_i64_i32", &[big]);
    assert_eq!(arr(r.unwrap()).as_slice::<i32>(), &[5]);
    let r = call(&reg, "cast_bool_i64", &[bools(&[true, false])]);
    assert_eq!(arr(r.unwrap()).as_slice::<i64>(), &[1, 0]);
    let r = call(&reg, "cast_i32_f64", &[i32s(&[-3, 4])]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[-3.0, 4.0]);
}

#[test]
fn comparisons_and_select() {
    let reg = Registry::new();
    let r = arr(call(&reg, "gt_f64", &[f64s(&[1.0, f64::NAN]), f64_s(0.0)]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[1, 0]);
    let nans = f64s(&[f64::NAN, 1.0]);
    let r = arr(call(&reg, "ne_f64", &[nans.clone(), nans]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[1, 0]);
    let cond = bools(&[true, false]);
    let r = call(
        &reg,
        "select_f64",
        &[cond.clone(), f64s(&[1.0, 2.0]), f64s(&[3.0, 4.0])],
    );
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[1.0, 4.0]);
    let r = call(
        &reg,
        "select_f64",
        &[cond.clone(), f64s(&[1.0, 2.0]), f64_s(0.0)],
    );
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[1.0, 0.0]);
    // A scalar condition picks a whole branch.
    let no = Value::Scalar(Scalar::Bool(false));
    let r = call(&reg, "select_f64", &[no, f64s(&[1.0, 2.0]), f64_s(7.0)]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[7.0, 7.0]);
    let r = call(
        &reg,
        "select_bool",
        &[cond, bools(&[false, false]), bools(&[true, true])],
    );
    assert_eq!(arr(r.unwrap()).as_bool_bytes(), &[0, 1]);
    let (a, b) = (bools(&[true, true, false]), bools(&[true, false, false]));
    let r = arr(call(&reg, "and_bool", &[a.clone(), b.clone()]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[1, 0, 0]);
    let r = arr(call(&reg, "or_bool", &[a, b]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[1, 1, 0]);
    let r = arr(call(&reg, "not_bool", &[bools(&[true, false])]).unwrap());
    assert_eq!(r.as_bool_bytes(), &[0, 1]);
}

#[test]
fn float_only_ops() {
    let reg = Registry::new();
    let r = call(&reg, "div_f64", &[f64s(&[1.0, -1.0]), f64_s(0.0)]);
    assert_eq!(
        arr(r.unwrap()).as_slice::<f64>(),
        &[f64::INFINITY, f64::NEG_INFINITY]
    );
    let r = call(&reg, "sqrt_f64", &[f64s(&[4.0, 9.0])]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[2.0, 3.0]);
    let r = call(&reg, "abs_f64", &[f64s(&[-1.5, 2.0])]);
    assert_eq!(arr(r.unwrap()).as_slice::<f64>(), &[1.5, 2.0]);
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
        assert_eq!(def.class, KernelClass::Elementwise, "{}", def.name);
        assert!(def.pattern.is_none());
        assert!(def.impls[Isa::Scalar.index()].is_some());
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
fn select_impl_falls_back_to_scalar() {
    let reg = Registry::new();
    let id = reg.by_name("add_f64").unwrap();
    for isa in Isa::ALL {
        assert_eq!(reg.select_impl(id, isa).1, Isa::Scalar);
    }
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
    let mut whole = vec![0.0f64; n];
    binary_into(
        Operand::Slice(&a),
        Operand::Splat(2.5),
        &mut whole,
        |x, y| x * y,
    );
    let mut tiled = vec![0.0f64; n];
    for r in [0..37, 37..100, 100..1000] {
        let x = Operand::Slice(&a).range(r.clone());
        binary_into(x, Operand::Splat(2.5), &mut tiled[r], |x, y| x * y);
    }
    assert_eq!(whole, tiled);
}
