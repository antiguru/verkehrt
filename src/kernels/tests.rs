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
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
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

/// The selection rule `lookup_lifted` replaced, which looked the kernel up a second time
/// when no operand is an array.
fn lookup_lifted_reference(
    reg: &Registry,
    op: Op,
    args: &[ValueType],
) -> Result<Lifted, Vec<ValueType>> {
    let none = || vec![false; args.len()];
    let bcast: Vec<bool> = if !args.iter().any(|t| matches!(t, ValueType::Array(_))) {
        none()
    } else {
        match reg.lookup(op, args) {
            Some(id) => reg
                .get(id)
                .inputs
                .iter()
                .zip(args)
                .map(|(spec, t)| {
                    matches!(spec, InputSpec::Either(_)) && matches!(t, ValueType::Scalar(_))
                })
                .collect(),
            None if reg.has_scalar_input(op) => none(),
            None => args
                .iter()
                .map(|t| matches!(t, ValueType::Scalar(_)))
                .collect(),
        }
    };
    let tys: Vec<ValueType> = args
        .iter()
        .zip(&bcast)
        .map(|(t, b)| match (t, b) {
            (ValueType::Scalar(s), true) => ValueType::Array(*s),
            _ => *t,
        })
        .collect();
    match reg.lookup(op, &tys) {
        Some(kernel) => Ok(Lifted { kernel, bcast }),
        None => Err(tys),
    }
}

#[test]
fn lookup_lifted_matches_the_reference_selection() {
    use ScalarType::*;
    let reg = Registry::new();
    let types = [
        sc_ty(F64),
        sc_ty(I64),
        sc_ty(Bool),
        arr_ty(F64),
        arr_ty(I64),
        arr_ty(Bool),
    ];
    let ops = [
        Op::Add,
        Op::Div,
        Op::Gt,
        Op::Neg,
        Op::Not,
        Op::Cast(F64),
        Op::Fill,
        Op::Iota,
        Op::Compress,
        Op::Gather,
        Op::Reduce(crate::ops::ReduceOp::Add),
        Op::Length,
    ];
    let mut checked = 0;
    for op in ops {
        for a in types {
            assert_eq!(
                reg.lookup_lifted(op, &[a]),
                lookup_lifted_reference(&reg, op, &[a]),
                "{op:?} {a:?}"
            );
            for b in types {
                assert_eq!(
                    reg.lookup_lifted(op, &[a, b]),
                    lookup_lifted_reference(&reg, op, &[a, b]),
                    "{op:?} {a:?} {b:?}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    // Spot checks on the three paths: all scalar, a broadcast scalar, a scalar that stays.
    let all_scalar = reg
        .lookup_lifted(Op::Add, &[sc_ty(F64), sc_ty(F64)])
        .unwrap();
    assert_eq!(all_scalar.bcast, [false, false]);
    let mixed = reg
        .lookup_lifted(Op::Add, &[arr_ty(F64), sc_ty(F64)])
        .unwrap();
    assert_eq!(mixed.bcast, [false, true]);
    assert!(reg.lookup_lifted(Op::Iota, &[arr_ty(I64)]).is_err());
}

#[test]
fn lookup_is_exact_for_non_elementwise_kernels() {
    use ScalarType::*;
    // An empty registry, so the dummy kernels do not collide with the real ones.
    let mut reg = Registry::empty();
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
            Some(Op::Prev | Op::Scan(_) | Op::ScanBy(_)) => KernelClass::Scan,
            Some(
                Op::Iota
                | Op::Fill
                | Op::Gather
                | Op::Slice
                | Op::Histogram
                | Op::ArgSort
                | Op::ReduceInto(_)
                | Op::RandI64
                | Op::RandF64
                | Op::Load(_)
                | Op::Length
                | Op::SameLen,
            ) => KernelClass::Whole,
            _ => KernelClass::Elementwise,
        };
        assert_eq!(def.class, expected, "{}", def.name);
        assert!(def.impls.get(Variant::SCALAR).is_some());
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
    for t in ScalarType::ARRAY_TYPES {
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
    // A `_s` kernel has only its scalar entry.
    let id = reg.by_name("add_f64_s").unwrap();
    for isa in Isa::ALL {
        for backend in [Backend::Auto, Backend::Intrinsics] {
            assert_eq!(
                reg.resolve(isa, backend, FeatureSet::EMPTY)
                    .variant_used(id),
                Variant::SCALAR
            );
        }
    }
    // A kernel with ISA variants uses its scalar entry for `Isa::Scalar` and for an ISA
    // without variants or support.
    let add = reg.by_name("add_f64").unwrap();
    for isa in [Isa::Scalar, Isa::Neon] {
        for backend in [Backend::Auto, Backend::Intrinsics] {
            assert_eq!(
                reg.resolve(isa, backend, FeatureSet::EMPTY)
                    .variant_used(add),
                Variant::SCALAR
            );
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
    let v3 = Variant {
        isa: Isa::X86_64V3,
        backend: Backend::Auto,
    };
    assert_eq!(v3.to_string(), "isa=x86-64-v3 backend=auto");
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
fn new_isa_names_parse() {
    assert_eq!(Isa::from_name("x86-64-v3"), Some(Isa::X86_64V3));
    assert_eq!(Isa::from_name("x86-64-v4"), Some(Isa::X86_64V4));
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

/// The golden bits below pin the canonical order independently of the code that computes
/// it. The input has 35 elements, so `b = 32` and the tail is elements 32, 33, 34.
///
/// Hand check against the spec's bullets:
/// * Accumulator 0 holds `1e16 + -1e16 = 0` (elements 0 and 16), so the large pair cancels
///   exactly. A left fold instead adds 0.1 to 1e16 first, which rounds the small values away,
///   and ends at 49.56.
/// * Accumulators 1 to 15 hold `x[j] + x[j + 16]`, for example `0.1 + 1.7` in accumulator 1.
/// * Pairwise combination then adds accumulator `j + 8` into `j`, then `+ 4`, `+ 2`, `+ 1`.
/// * The tail adds `0.01`, `0.02`, `0.03` to accumulator 0 in order.
///
/// The resulting bits are those of the literal `50.16`. They were computed by an independent
/// lane-by-lane simulation as well as by `canonical_sum`, and the two agree.
#[test]
fn canonical_sum_golden_values() {
    #[rustfmt::skip]
    let xs: [f64; 35] = [
        1e16, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        -1e16, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3,
        0.01, 0.02, 0.03,
    ];
    assert_eq!(
        canonical_sum(&xs).to_bits(),
        f64::from_bits(0x4049_147a_e147_ae14).to_bits()
    );
    assert_eq!(xs.iter().fold(0.0, |a, &x| a + x), 49.56);

    // Accumulator 0 holds 2^24 and 1.0 (elements 0 and 16). In `f32` the sum 2^24 + 1 is not
    // representable and rounds to 2^24 (ties to even), so the 1.0 is lost. The other
    // accumulators hold exact multiples of 0.25, and the three tail elements are 3.0 each.
    // With `f64` accumulation the 1.0 survives, the final rounding to `f32` lands on
    // 16777394 instead of the `f32`-accumulated 16777396, which the assertions pin down.
    #[rustfmt::skip]
    let ys: [f32; 35] = [
        16777216.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5,
        1.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25,
        3.0, 3.0, 3.0,
    ];
    let got = canonical_sum(&ys);
    assert_eq!(got.to_bits(), f32::from_bits(0x4b80_005a).to_bits());
    let wide: Vec<f64> = ys.iter().map(|&y| f64::from(y)).collect();
    let in_f64 = canonical_sum(&wide) as f32;
    assert_eq!(in_f64.to_bits(), f32::from_bits(0x4b80_0059).to_bits());
    assert_ne!(got.to_bits(), in_f64.to_bits());
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

/// The predicated selections reserve and commit in chunks, so check lengths around the chunk
/// boundaries against plain iterator filters, one-shot and in tiles that straddle chunks.
#[test]
fn selections_span_chunks() {
    use super::scalar::select::CHUNK;
    let reg = Registry::new();
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
    for n in [CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 5] {
        let xs: Vec<f64> = (0..n)
            .map(|i| f64::from(u32::try_from(i).unwrap()))
            .collect();
        let mask: Vec<bool> = (0..n)
            .map(|i| splitmix64(u64::try_from(i).unwrap()).is_multiple_of(3))
            .collect();
        let kept: Vec<f64> = xs
            .iter()
            .zip(&mask)
            .filter(|p| *p.1)
            .map(|p| *p.0)
            .collect();
        let idx: Vec<i64> = (0..n)
            .filter(|&i| mask[i])
            .map(|i| i64::try_from(i).unwrap())
            .collect();
        let cut = f64::from(u32::try_from(n / 2).unwrap());
        let above: Vec<f64> = xs.iter().copied().filter(|&v| v > cut).collect();
        let compress = call(&reg, "compress_f64", &[f64s(&xs), bools(&mask)]);
        assert_eq!(arr(compress.unwrap()).as_slice::<f64>(), kept, "n={n}");
        let r#where = call(&reg, "where", &[bools(&mask)]);
        assert_eq!(arr(r#where.unwrap()).as_slice::<i64>(), idx, "n={n}");
        let filter = call(&reg, "filter_gt_f64", &[f64s(&xs), f64_s(cut)]);
        assert_eq!(arr(filter.unwrap()).as_slice::<f64>(), above, "n={n}");
        // Tiles of `CHUNK + 3` rows put chunk boundaries at every offset into a tile.
        let id = reg.by_name("where").unwrap();
        let all = Array::from_bools(&mask);
        let mut out = ArrayBuilder::new(ScalarType::I64);
        for start in (0..n).step_by(CHUNK + 3) {
            let len = (CHUNK + 3).min(n - start);
            let args = [Arg::Rows(all.slice(start, len))];
            dispatch
                .select_tile(&reg, id, &args, start, &mut out)
                .unwrap();
        }
        assert_eq!(out.finish().as_slice::<i64>(), idx, "tiled n={n}");
    }
}

/// A selection counts its survivors and reserves for them plus one vector (64 bytes) of
/// room for vector stores, so the array keeps at most that vector plus the rounding up to
/// whole 64-byte blocks of capacity past its rows, without a shrink copy.
#[test]
fn selection_retains_at_most_one_vector_of_slack() {
    let reg = Registry::new();
    let n = 10_000;
    let xs: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i).unwrap()))
        .collect();
    // 13% survive, and 60%, where the 2x shrink of `ArrayBuilder::finish` does not apply, so
    // only exact sizing keeps the slack within a block.
    for keep in [1300, 6000] {
        let mask: Vec<bool> = (0..n).map(|i| i < keep).collect();
        let kept = arr(call(&reg, "compress_f64", &[f64s(&xs), bools(&mask)]).unwrap());
        assert_eq!(kept.len(), keep);
        let bytes = kept.len() * size_of::<f64>();
        assert!(
            kept.buffer_capacity_bytes() <= (bytes + 64).next_multiple_of(64),
            "{} bytes of capacity for {bytes} bytes of rows",
            kept.buffer_capacity_bytes()
        );
    }
}

/// Once the survivors of a tile run out, every chunk still gets a vector of room past them,
/// so the entries keep their vector loops instead of the predicated scalar loop.
#[test]
fn append_chunks_keeps_a_vector_of_room_past_the_survivors() {
    use super::scalar::select::{CHUNK, append_chunks, predicated};
    let n = 3 * CHUNK + 5;
    let kept = 10;
    let mut out = ArrayBuilder::new(ScalarType::F64);
    let mut rooms = Vec::new();
    append_chunks::<f64>(&mut out, n, kept, |rows, dst| {
        rooms.push((rows.len(), dst.len()));
        let xs = rows.map(|i| (f64::from(u32::try_from(i).unwrap()), i < kept));
        predicated(xs, dst)
    });
    assert_eq!(rooms.len(), 4);
    for (rows, room) in rooms {
        assert!(
            room >= rows.min(8),
            "{room} rows of room for a chunk of {rows}"
        );
    }
    let want: Vec<f64> = (0..10).map(f64::from).collect();
    assert_eq!(out.finish().as_slice::<f64>(), want);
}

/// Survivors packed at the front and none after, through every variant this CPU runs.
#[test]
fn selections_with_a_dense_prefix_and_an_empty_suffix() {
    use super::scalar::select::CHUNK;
    let reg = Registry::new();
    let n = 3 * CHUNK + 5;
    let prefix = CHUNK + 100;
    // Falling values, so `x > cut` keeps exactly the first `prefix` rows.
    let xs: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(n - i).unwrap()))
        .collect();
    let cut = f64::from(u32::try_from(n - prefix).unwrap());
    let mask: Vec<bool> = (0..n).map(|i| i < prefix).collect();
    let idx: Vec<i64> = (0..prefix).map(|i| i64::try_from(i).unwrap()).collect();
    let mut checked = 0;
    for isa in [Isa::Scalar, Isa::X86_64V3, Isa::X86_64V4] {
        if !isa.is_available() {
            continue;
        }
        for backend in [Backend::Auto, Backend::Intrinsics] {
            let dispatch = reg.resolve(isa, backend, FeatureSet::EMPTY);
            let run = |name: &str, args: &[Value]| {
                let id = reg.by_name(name).unwrap();
                arr(dispatch.call(&reg, id, args).unwrap())
            };
            let compressed = run("compress_f64", &[f64s(&xs), bools(&mask)]);
            assert_eq!(
                compressed.as_slice::<f64>(),
                &xs[..prefix],
                "{isa:?} {backend:?}"
            );
            let filtered = run("filter_gt_f64", &[f64s(&xs), f64_s(cut)]);
            assert_eq!(
                filtered.as_slice::<f64>(),
                &xs[..prefix],
                "{isa:?} {backend:?}"
            );
            let indices = run("where", &[bools(&mask)]);
            assert_eq!(indices.as_slice::<i64>(), idx, "{isa:?} {backend:?}");
            checked += 1;
        }
    }
    assert!(checked >= 2);
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
fn prev_shifts_rows_and_fills_row_zero() {
    let reg = Registry::new();
    let i64s = |v: &[i64]| Value::Array(Array::from_vec(v.to_vec()));
    let shifted = call(&reg, "prev_i64", &[i64s(&[1, 2, 3]), i64_s(9)]);
    assert_eq!(arr(shifted.unwrap()).as_slice::<i64>(), &[9, 1, 2]);
    let empty = call(&reg, "prev_i64", &[i64s(&[]), i64_s(9)]);
    assert_eq!(arr(empty.unwrap()).len(), 0);
    let flags = call(
        &reg,
        "prev_bool",
        &[
            bools(&[true, false, true]),
            Value::Scalar(Scalar::Bool(true)),
        ],
    );
    let flags = arr(flags.unwrap());
    assert_eq!(flags.as_bool_bytes(), &[1, 1, 0]);
    // A fill of the wrong type is an error, not a reinterpretation.
    assert!(call(&reg, "prev_i64", &[i64s(&[1]), f64_s(1.0)]).is_err());
}

/// A whole body with auto entries, which no registered kernel is built from yet.
struct Echo;

impl WholeBody for Echo {
    fn run(args: &[Value]) -> Result<Value, String> {
        args.first().cloned().ok_or_else(|| "no argument".into())
    }
}

#[test]
fn whole_tables_have_scalar_and_auto_entries() {
    let impls = KernelImpl::whole::<Echo>();
    assert!(impls.get(Variant::SCALAR).is_some());
    let auto = |isa| Variant {
        isa,
        backend: Backend::Auto,
    };
    let x86 = cfg!(target_arch = "x86_64");
    assert_eq!(impls.get(auto(Isa::X86_64V3)).is_some(), x86);
    assert_eq!(impls.get(auto(Isa::X86_64V4)).is_some(), x86);
    let isa = Isa::detect();
    if let Some(KernelImpl::Whole(f)) = impls.get(auto(isa)) {
        let args = [i64_s(5)];
        // SAFETY: `f` is the auto entry of `Isa::detect()`, whose level `features::runnable`
        // approved for this CPU, and the body is safe Rust.
        let got = unsafe { f(&args) };
        assert_eq!(scalar(got), Scalar::I64(5));
    }
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

/// The number of kernels with an entry in each variant slot, in `Variant::index` order.
#[test]
fn variant_entry_counts_are_stable() {
    let reg = Registry::new();
    let mut counts = [0usize; Variant::COUNT];
    for (_, def) in reg.iter() {
        for v in all_variants() {
            counts[v.index()] += usize::from(def.impls.get(v).is_some());
        }
    }
    // Scalar, then auto and intrinsics for x86-64-v3, x86-64-v4, and NEON. The x86 literals were
    // recorded before the architecture layer existed. The `u64` kernels added 56 scalar and
    // 52 auto entries per x86 ISA, and 7 intrinsics entries per ISA (`compress_u64` and the
    // six `filter_<cmp>_u64`). The bitwise kernels on four integer types and `xor_bool`
    // added 46 scalar and 29 auto entries per x86 ISA, and no intrinsics entries. The 70
    // `scan_<op>_<t>` and `scan_by_<op>_<t>` kernels added 70 scalar and 70 auto entries per
    // x86 ISA. The 7 `gather_<t>` kernels added 7 scalar and 7 auto entries per x86 ISA, and
    // the 7 `slice_<t>` kernels added 7 scalar entries.
    // `histogram` and the 35 `reduce_into_<op>_<t>` kernels added 36 scalar and 36 auto
    // entries per x86 ISA.
    // The 4 `div_const_<t>` and 4 `argsort_<t>` kernels added 12 scalar entries (the array and
    // `_s` forms of `div_const`) and 8 auto entries per x86 ISA.
    // `gather_i64`, `gather_u64`, and `gather_f64` added 3 intrinsics entries per x86 ISA.
    // `scan_by_add_i64` and `scan_by_add_u64` added 2 x86-64-v4 intrinsics entries.
    #[cfg(target_arch = "x86_64")]
    assert_eq!(counts, [749, 439, 39, 439, 43, 0, 0]);
    // NEON intrinsics cover 7 compress kernels, `where`, 30 fused filters, and the 8 float
    // min, max, argmin, and argmax reductions. Its auto slot stays empty by design.
    #[cfg(target_arch = "aarch64")]
    assert_eq!(counts, [749, 0, 0, 0, 0, 0, 46]);
}

fn u64s(v: &[u64]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

fn u64_s(v: u64) -> Value {
    Value::Scalar(Scalar::U64(v))
}

#[test]
fn u64_compares_unsigned() {
    let reg = Registry::new();
    // A signed compare reads `1 << 63` as negative, so it gets this wrong.
    let r = call(&reg, "gt_u64", &[u64s(&[1 << 63, 1]), u64_s(1)]);
    assert_eq!(arr(r.unwrap()).as_bool_bytes(), &[1, 0]);
}

#[test]
fn u64_casts_follow_as() {
    let reg = Registry::new();
    let to_f64 = call(&reg, "cast_u64_f64", &[u64s(&[u64::MAX])]);
    assert_eq!(
        arr(to_f64.unwrap()).as_slice::<f64>(),
        &[18446744073709551615.0]
    );
    let from_f64 = call(&reg, "cast_f64_u64", &[f64s(&[-1.0, 1e30, f64::NAN])]);
    assert_eq!(arr(from_f64.unwrap()).as_slice::<u64>(), &[0, u64::MAX, 0]);
    let to_i64 = call(&reg, "cast_u64_i64", &[u64s(&[1 << 63])]);
    assert_eq!(arr(to_i64.unwrap()).as_slice::<i64>(), &[i64::MIN]);
    let minus_one = Value::Array(Array::from_vec(vec![-1i64]));
    let from_i64 = call(&reg, "cast_i64_u64", &[minus_one]);
    assert_eq!(arr(from_i64.unwrap()).as_slice::<u64>(), &[u64::MAX]);
}

#[test]
fn u64_sum_and_extremes() {
    let reg = Registry::new();
    let x = u64s(&[3, 1 << 63, 7]);
    let sum = scalar(call(&reg, "reduce_add_u64", std::slice::from_ref(&x)));
    assert_eq!(sum, Scalar::U64((1 << 63) + 10));
    let min = scalar(call(&reg, "reduce_min_u64", std::slice::from_ref(&x)));
    assert_eq!(min, Scalar::U64(3));
    let max = scalar(call(&reg, "reduce_max_u64", std::slice::from_ref(&x)));
    assert_eq!(max, Scalar::U64(1 << 63));
    let argmax = scalar(call(&reg, "argmax_u64", std::slice::from_ref(&x)));
    assert_eq!(argmax, Scalar::I64(1));
    let argmin = scalar(call(&reg, "argmin_u64", std::slice::from_ref(&x)));
    assert_eq!(argmin, Scalar::I64(0));
    let mul = scalar(call(&reg, "reduce_mul_u64", &[u64s(&[1 << 63, 2])]));
    assert_eq!(mul, Scalar::U64(0));
}

/// The `_i64` kernels with no `_u64` twin by design.
const NO_U64_TWIN: &[&str] = &[
    // Signed only: unsigned values have no negation or absolute value.
    "neg_i64", "abs_i64",
    // The generator is defined by its `i64` bits, there is no unsigned variant of it.
    "rand_i64",
];

#[test]
fn every_i64_kernel_has_a_u64_twin() {
    let reg = Registry::new();
    let mut missing = vec![];
    for (_, def) in reg.iter() {
        let name = &def.name;
        let Some(stem) = name.strip_suffix("_i64") else {
            continue;
        };
        if !NO_U64_TWIN.contains(&name.as_str()) && reg.by_name(&format!("{stem}_u64")).is_none() {
            missing.push(name.clone());
        }
    }
    assert!(missing.is_empty(), "no u64 twin for {missing:?}");
}

#[test]
fn bitwise_kernels_on_integers() {
    let reg = Registry::new();
    let a = Value::Array(Array::from_vec(vec![12i64, 5]));
    let b = Value::Array(Array::from_vec(vec![10i64, 3]));
    let and = call(&reg, "and_i64", &[a.clone(), b.clone()]);
    assert_eq!(arr(and.unwrap()).as_slice::<i64>(), &[8, 1]);
    let or = call(&reg, "or_i64", &[a.clone(), b.clone()]);
    assert_eq!(arr(or.unwrap()).as_slice::<i64>(), &[14, 7]);
    let xor = call(&reg, "xor_i64", &[a.clone(), b]);
    assert_eq!(arr(xor.unwrap()).as_slice::<i64>(), &[6, 6]);
    let not = call(&reg, "not_i64", &[a]);
    assert_eq!(arr(not.unwrap()).as_slice::<i64>(), &[-13, -6]);
    let not_u8 = call(
        &reg,
        "not_u8",
        &[Value::Array(Array::from_vec(vec![0u8, 255]))],
    );
    assert_eq!(arr(not_u8.unwrap()).as_slice::<u8>(), &[255, 0]);
    // The unsigned twin keeps the top bit as a value bit.
    let not_u64 = call(&reg, "not_u64", &[u64s(&[0, 1 << 63])]);
    assert_eq!(
        arr(not_u64.unwrap()).as_slice::<u64>(),
        &[u64::MAX, u64::MAX >> 1]
    );
}

#[test]
fn bitwise_reductions_have_identities() {
    let reg = Registry::new();
    let x = Value::Array(Array::from_vec(vec![12i32, 10, 6]));
    let r = |name: &str, v: &Value| scalar(call(&reg, name, std::slice::from_ref(v)));
    assert_eq!(r("reduce_and_i32", &x), Scalar::I32(0));
    assert_eq!(r("reduce_or_i32", &x), Scalar::I32(14));
    assert_eq!(r("reduce_xor_i32", &x), Scalar::I32(0));
    let empty = u64s(&[]);
    assert_eq!(r("reduce_and_u64", &empty), Scalar::U64(u64::MAX));
    assert_eq!(r("reduce_or_u64", &empty), Scalar::U64(0));
    assert_eq!(r("reduce_xor_u64", &empty), Scalar::U64(0));
}

fn i64v(v: &[i64]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

/// The array a kernel produces through the scalar implementation.
fn run_arr(name: &str, args: &[Value]) -> Array {
    arr(call(&Registry::new(), name, args).unwrap())
}

#[test]
fn scan_add_i64_is_a_prefix_sum() {
    let r = run_arr("scan_add_i64", &[i64v(&[1, 2, 3, 4])]);
    assert_eq!(r.as_slice::<i64>(), &[1, 3, 6, 10]);
    let wrapped = run_arr("scan_add_i64", &[i64v(&[i64::MAX, 1])]);
    assert_eq!(wrapped.as_slice::<i64>(), &[i64::MAX, i64::MIN]);
}

#[test]
fn scan_add_i32_widens() {
    let r = run_arr("scan_add_i32", &[i32s(&[i32::MAX, 1])]);
    assert_eq!(r.as_slice::<i64>(), &[2147483647, 2147483648]);
}

#[test]
fn scan_add_u8_widens() {
    let r = run_arr("scan_add_u8", &[u8s(&[255, 1])]);
    assert_eq!(r.as_slice::<u64>(), &[255, 256]);
}

#[test]
fn scan_max_f64_propagates_nan() {
    let nan = run_arr("scan_max_f64", &[f64s(&[1.0, f64::NAN, 3.0])]);
    let nan = nan.as_slice::<f64>();
    assert_eq!(nan[0], 1.0);
    assert!(nan[1].is_nan() && nan[2].is_nan());
    let zeros = run_arr("scan_max_f64", &[f64s(&[-0.0, 0.0])]);
    let bits: Vec<u64> = zeros
        .as_slice::<f64>()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(bits, [(-0.0f64).to_bits(), 0.0f64.to_bits()]);
}

#[test]
fn scan_by_restarts_at_starts() {
    let starts = bools(&[false, false, true, false, true]);
    let r = run_arr("scan_by_add_i64", &[i64v(&[1, 2, 3, 4, 5]), starts]);
    assert_eq!(r.as_slice::<i64>(), &[1, 3, 3, 7, 5]);
    // Row 0 starts a segment whether or not `starts[0]` says so.
    let first = run_arr("scan_by_add_i64", &[i64v(&[1, 2]), bools(&[true, false])]);
    assert_eq!(first.as_slice::<i64>(), &[1, 3]);
}

#[test]
fn scan_bool_and_bitwise_kernels() {
    let xor = run_arr("scan_xor_bool", &[bools(&[true, true, true, false])]);
    assert_eq!(xor.as_bool_bytes(), &[1, 0, 1, 1]);
    let and = run_arr("scan_and_i32", &[i32s(&[7, 3, 1, 6])]);
    assert_eq!(and.as_slice::<i32>(), &[7, 3, 1, 0]);
    let mul = run_arr("scan_mul_i64", &[i64v(&[2, 3, 4])]);
    assert_eq!(mul.as_slice::<i64>(), &[2, 6, 24]);
    let min = run_arr("scan_min_i64", &[i64v(&[3, 1, 2])]);
    assert_eq!(min.as_slice::<i64>(), &[3, 1, 1]);
    let starts = bools(&[false, false, true, false]);
    let max = run_arr("scan_by_max_u64", &[u64s(&[5, 1, 9, 2]), starts]);
    assert_eq!(max.as_slice::<u64>(), &[5, 5, 9, 9]);
}

#[test]
fn scan_mul_keeps_the_element_type() {
    let i32s_out = run_arr("scan_mul_i32", &[i32s(&[65536, 65536])]);
    assert_eq!(i32s_out.elem(), ScalarType::I32);
    assert_eq!(i32s_out.as_slice::<i32>(), &[65536, 0]);
    let u8s_out = run_arr("scan_mul_u8", &[u8s(&[16, 16])]);
    assert_eq!(u8s_out.elem(), ScalarType::U8);
    assert_eq!(u8s_out.as_slice::<u8>(), &[16, 0]);
    let by = run_arr(
        "scan_by_mul_u8",
        &[u8s(&[16, 16, 3]), bools(&[false, false, true])],
    );
    assert_eq!(by.as_slice::<u8>(), &[16, 0, 3]);
}

#[test]
fn scan_last_equals_sum() {
    let reg = Registry::new();
    let xs: Vec<i64> = (0..500)
        .map(|i| splitmix64(u64::try_from(i).unwrap()) as i64)
        .collect();
    let scan64 = run_arr("scan_add_i64", &[i64v(&xs)]);
    let sum64 = scalar(call(&reg, "reduce_add_i64", &[i64v(&xs)]));
    assert_eq!(
        Scalar::I64(*scan64.as_slice::<i64>().last().unwrap()),
        sum64
    );
    let ys: Vec<i32> = xs.iter().map(|&x| x as i32).collect();
    let scan32 = run_arr("scan_add_i32", &[i32s(&ys)]);
    let sum32 = scalar(call(&reg, "reduce_add_i32", &[i32s(&ys)]));
    assert_eq!(
        Scalar::I64(*scan32.as_slice::<i64>().last().unwrap()),
        sum32
    );
}

#[test]
fn gather_picks_rows() {
    let picked = run_arr("gather_i64", &[i64v(&[10, 20, 30]), i64v(&[2, 0, 2])]);
    assert_eq!(picked.as_slice::<i64>(), &[30, 10, 30]);
    let empty = run_arr("gather_i64", &[i64v(&[10, 20, 30]), i64v(&[])]);
    assert_eq!(empty.len(), 0);
    let flags = run_arr(
        "gather_bool",
        &[bools(&[true, false, false]), i64v(&[1, 0, 1, 0])],
    );
    assert_eq!(flags.elem(), ScalarType::Bool);
    assert_eq!(flags.as_bool_bytes(), &[0, 1, 0, 1]);
    let floats = run_arr("gather_f64", &[f64s(&[1.5, 2.5]), i64v(&[1, 1, 0])]);
    assert_eq!(floats.as_slice::<f64>(), &[2.5, 2.5, 1.5]);
}

#[test]
fn gather_rejects_out_of_range() {
    let reg = Registry::new();
    let xs = i64v(&[10, 20, 30]);
    let high = call(&reg, "gather_i64", &[xs.clone(), i64v(&[0, 3])]).unwrap_err();
    assert!(high.contains("index 3"), "{high}");
    let low = call(&reg, "gather_i64", &[xs.clone(), i64v(&[-1, 0])]).unwrap_err();
    assert!(low.contains("index -1"), "{low}");
    let last = run_arr("gather_i64", &[xs, i64v(&[2])]);
    assert_eq!(last.as_slice::<i64>(), &[30]);
    // Every index is out of range for an empty source.
    let none = call(&reg, "gather_i64", &[i64v(&[]), i64v(&[0])]).unwrap_err();
    assert!(none.contains("index 0"), "{none}");
}

#[test]
fn slice_is_a_view() {
    let reg = Registry::new();
    let x = arr(call(&reg, "iota", &[i64_s(10)]).unwrap());
    let xv = Value::Array(x.clone());
    let mid = run_arr("slice_i64", &[xv.clone(), i64_s(2), i64_s(8)]);
    assert_eq!(mid.as_slice::<i64>(), &[2, 3, 4, 5, 6, 7]);
    // A view shares the operand's rows.
    assert_eq!(
        mid.as_slice::<i64>().as_ptr(),
        x.as_slice::<i64>()[2..].as_ptr()
    );
    let inner = run_arr("slice_i64", &[Value::Array(mid), i64_s(1), i64_s(3)]);
    assert_eq!(inner.as_slice::<i64>(), &[3, 4]);
    for at in [0, 10] {
        let none = run_arr("slice_i64", &[xv.clone(), i64_s(at), i64_s(at)]);
        assert_eq!(none.len(), 0);
    }
}

#[test]
fn slice_rejects_bad_bounds() {
    let reg = Registry::new();
    let xv = i64v(&[1, 2, 3]);
    for (start, end) in [(2, 1), (0, 4), (-1, 2), (4, 4)] {
        let e = call(&reg, "slice_i64", &[xv.clone(), i64_s(start), i64_s(end)]).unwrap_err();
        assert!(e.contains(&format!("{start}..{end}")), "{e}");
    }
}

#[test]
fn histogram_counts_keys() {
    let h = run_arr("histogram", &[i64v(&[0, 2, 2, 3]), i64_s(5)]);
    assert_eq!(h.as_slice::<i64>(), &[1, 0, 2, 1, 0]);
    let none = run_arr("histogram", &[i64v(&[]), i64_s(0)]);
    assert_eq!(none.len(), 0);
    let empty_buckets = run_arr("histogram", &[i64v(&[]), i64_s(3)]);
    assert_eq!(empty_buckets.as_slice::<i64>(), &[0, 0, 0]);
}

#[test]
fn bucket_keys_are_range_checked() {
    let reg = Registry::new();
    let both = |keys: &[i64], n: i64| {
        let h = call(&reg, "histogram", &[i64v(keys), i64_s(n)]);
        let vals = i64v(&vec![1; keys.len()]);
        let r = call(&reg, "reduce_into_add_i64", &[i64v(keys), vals, i64_s(n)]);
        (h, r)
    };
    // With no buckets every key is out of range.
    for (key, n) in [(3, 3), (-1, 3), (0, 0)] {
        let (h, r) = both(&[0, key], n);
        let first = if n == 0 { 0 } else { key };
        let want = format!("bucket key {first} out of range for {n} buckets");
        assert_eq!(h.unwrap_err(), want);
        assert_eq!(r.unwrap_err(), want);
    }
    let (h, r) = both(&[2, 0, 2], 3);
    assert_eq!(arr(h.unwrap()).as_slice::<i64>(), &[1, 0, 2]);
    assert_eq!(arr(r.unwrap()).as_slice::<i64>(), &[1, 0, 2]);
    let (neg_h, neg_r) = both(&[], -1);
    assert_eq!(neg_h.unwrap_err(), "negative bucket count -1");
    assert_eq!(neg_r.unwrap_err(), "negative bucket count -1");
}

#[test]
fn huge_bucket_count_is_an_error() {
    let reg = Registry::new();
    let hist = call(&reg, "histogram", &[i64v(&[0]), i64_s(1 << 62)]).unwrap_err();
    assert!(hist.contains("too large"), "{hist}");
    // Under the `isize::MAX` byte cap but 8 PiB, so only the allocator can refuse it.
    let big = call(&reg, "histogram", &[i64v(&[0]), i64_s(1 << 50)]).unwrap_err();
    assert!(big.contains("cannot allocate"), "{big}");
    let big_reduce = call(
        &reg,
        "reduce_into_add_i64",
        &[i64v(&[0]), i64v(&[1]), i64_s(1 << 50)],
    )
    .unwrap_err();
    assert!(big_reduce.contains("cannot allocate"), "{big_reduce}");
    let reduced = call(
        &reg,
        "reduce_into_add_i64",
        &[i64v(&[0]), i64v(&[1]), i64_s(1 << 62)],
    )
    .unwrap_err();
    assert!(reduced.contains("too large"), "{reduced}");
}

#[test]
fn reduce_into_sums_per_bucket() {
    let sums = run_arr(
        "reduce_into_add_i64",
        &[i64v(&[1, 0, 1]), i64v(&[5, 7, 9]), i64_s(3)],
    );
    assert_eq!(sums.as_slice::<i64>(), &[7, 14, 0]);
    // Only `add` widens narrow values, as in `reduce` and `scan`.
    let wide = run_arr(
        "reduce_into_add_i32",
        &[i64v(&[0, 0]), i32s(&[i32::MAX, 1]), i64_s(1)],
    );
    assert_eq!(wide.elem(), ScalarType::I64);
    assert_eq!(wide.as_slice::<i64>(), &[i64::from(i32::MAX) + 1]);
    let bits = run_arr(
        "reduce_into_xor_bool",
        &[i64v(&[1, 1, 0]), bools(&[true, true, true]), i64_s(2)],
    );
    assert_eq!(bits.elem(), ScalarType::Bool);
    assert_eq!(bits.as_bool_bytes(), &[1, 0]);
}

#[test]
fn reduce_into_empty_buckets_hold_identity() {
    // Bucket 1 holds the one value and every other bucket is empty.
    let one = |name: &str, vals: Value, n: i64| run_arr(name, &[i64v(&[1]), vals, i64_s(n)]);
    let min_f64 = one("reduce_into_min_f64", f64s(&[2.5]), 3);
    assert_eq!(
        min_f64.as_slice::<f64>(),
        &[f64::INFINITY, 2.5, f64::INFINITY]
    );
    let max_f32 = one(
        "reduce_into_max_f32",
        Value::Array(Array::from_vec(vec![2.5f32])),
        2,
    );
    assert_eq!(max_f32.as_slice::<f32>(), &[f32::NEG_INFINITY, 2.5]);
    let max_i64 = one("reduce_into_max_i64", i64v(&[4]), 2);
    assert_eq!(max_i64.as_slice::<i64>(), &[i64::MIN, 4]);
    let min_u64 = one("reduce_into_min_u64", u64s(&[4]), 2);
    assert_eq!(min_u64.as_slice::<u64>(), &[u64::MAX, 4]);
    let mul_i32 = one("reduce_into_mul_i32", i32s(&[4]), 2);
    assert_eq!(mul_i32.elem(), ScalarType::I32);
    assert_eq!(mul_i32.as_slice::<i32>(), &[1, 4]);
    let and_u8 = one("reduce_into_and_u8", u8s(&[4]), 2);
    assert_eq!(and_u8.as_slice::<u8>(), &[u8::MAX, 4]);
    let or_i32 = one("reduce_into_or_i32", i32s(&[4]), 2);
    assert_eq!(or_i32.as_slice::<i32>(), &[0, 4]);
    let xor_i64 = one("reduce_into_xor_i64", i64v(&[4]), 2);
    assert_eq!(xor_i64.as_slice::<i64>(), &[0, 4]);
    // The identity of bool `and` is true.
    let and_bool = one("reduce_into_and_bool", bools(&[false]), 2);
    assert_eq!(and_bool.as_bool_bytes(), &[1, 0]);
}

/// 40 pseudo-random rows of type `ty`, without NaNs and with duplicates.
fn rows_of_type(ty: ScalarType) -> Value {
    let raw: Vec<u64> = (0..40).map(|i| splitmix64(i) % 7).collect();
    match ty {
        ScalarType::I64 => i64v(&raw.iter().map(|&r| r as i64 - 3).collect::<Vec<_>>()),
        ScalarType::U64 => u64s(&raw),
        ScalarType::I32 => i32s(&raw.iter().map(|&r| r as i32 - 3).collect::<Vec<_>>()),
        ScalarType::U8 => u8s(&raw.iter().map(|&r| r as u8).collect::<Vec<_>>()),
        ScalarType::F64 => f64s(&raw.iter().map(|&r| r as f64 - 3.5).collect::<Vec<_>>()),
        ScalarType::F32 => Value::Array(Array::from_vec(
            raw.iter().map(|&r| r as f32 - 3.5).collect::<Vec<_>>(),
        )),
        ScalarType::Bool => bools(&raw.iter().map(|&r| r % 2 == 0).collect::<Vec<_>>()),
        other => panic!("no rows for {other:?}"),
    }
}

#[test]
fn reduce_into_with_one_key_matches_reduce() {
    let reg = Registry::new();
    let mut checked = 0;
    for (_, def) in reg.iter() {
        let Some(suffix) = def.name.strip_prefix("reduce_into_") else {
            continue;
        };
        let ValueType::Array(out) = def.output else {
            panic!("{} must produce an array", def.name);
        };
        let InputSpec::Array(ty) = def.inputs[1] else {
            panic!("{} takes an array of values", def.name);
        };
        let vals = rows_of_type(ty);
        let keys = i64v(&[0; 40]);
        let got = arr(call(&reg, &def.name, &[keys, vals.clone(), i64_s(1)]).unwrap());
        let want = scalar(call(&reg, &format!("reduce_{suffix}"), &[vals]));
        assert_eq!(got.elem(), out, "{}", def.name);
        assert_eq!(got.len(), 1, "{}", def.name);
        assert_eq!(got.get(0), want, "{}", def.name);
        checked += 1;
    }
    // 4 integer types with 7 operators, 2 float types with 2, and 3 for bool.
    assert_eq!(checked, 4 * 7 + 2 * 2 + 3);
}

#[test]
fn reduce_into_checks_lengths() {
    let reg = Registry::new();
    let e = call(
        &reg,
        "reduce_into_add_i64",
        &[i64v(&[0, 0]), i64v(&[1]), i64_s(1)],
    )
    .unwrap_err();
    assert_eq!(e, "length mismatch: 2 keys for 1 values");
}

fn i32_s(v: i32) -> Value {
    Value::Scalar(Scalar::I32(v))
}

fn u8_s(v: u8) -> Value {
    Value::Scalar(Scalar::U8(v))
}

#[test]
fn div_const_truncates_toward_zero() {
    let by = |d: i64| run_arr("div_const_i64", &[i64v(&[7, -7, 0]), i64_s(d)]);
    assert_eq!(by(2).as_slice::<i64>(), &[3, -3, 0]);
    assert_eq!(by(-2).as_slice::<i64>(), &[-3, 3, 0]);
}

#[test]
fn div_const_wraps_at_min() {
    let r64 = run_arr("div_const_i64", &[i64v(&[i64::MIN]), i64_s(-1)]);
    assert_eq!(r64.as_slice::<i64>(), &[i64::MIN]);
    let r32 = run_arr("div_const_i32", &[i32s(&[i32::MIN]), i32_s(-1)]);
    assert_eq!(r32.as_slice::<i32>(), &[i32::MIN]);
}

/// Dividends for the property test: boundary values, their neighbours, and random words.
fn dividends<T: Copy>(edges: &[u64], word: impl Fn(u64) -> T, seed: u64) -> Vec<T> {
    let mut v: Vec<T> = Vec::new();
    for &e in edges {
        for k in 0..4u64 {
            v.push(word(e.wrapping_add(k)));
            v.push(word(e.wrapping_sub(k)));
        }
    }
    v.extend((0..512).map(|i| word(splitmix64(seed ^ i))));
    v
}

#[test]
fn div_const_matches_rust_division() {
    let reg = Registry::new();
    // i64 and i32, every divisor in -300..=300 but 0, plus the extremes and neighbours.
    let mut d64: Vec<i64> = (-300..=300).filter(|&d| d != 0).collect();
    for k in 0..4 {
        d64.extend([i64::MIN + k, i64::MAX - k]);
    }
    let xs = dividends(&[0, 1 << 63, 1 << 62, 1 << 32], |w| w.cast_signed(), 1);
    for &d in &d64 {
        let r = call(&reg, "div_const_i64", &[i64v(&xs), i64_s(d)]);
        let want: Vec<i64> = xs.iter().map(|x| x.wrapping_div(d)).collect();
        assert_eq!(arr(r.unwrap()).as_slice::<i64>(), want, "i64 / {d}");
    }
    let mut d32: Vec<i32> = (-300..=300).filter(|&d| d != 0).collect();
    for k in 0..4 {
        d32.extend([i32::MIN + k, i32::MAX - k]);
    }
    let xs32 = dividends(
        &[0, 1 << 31, 1 << 30],
        |w| u32::try_from(w & 0xFFFF_FFFF).unwrap().cast_signed(),
        2,
    );
    for &d in &d32 {
        let r = call(&reg, "div_const_i32", &[i32s(&xs32), i32_s(d)]);
        let want: Vec<i32> = xs32.iter().map(|x| x.wrapping_div(d)).collect();
        assert_eq!(arr(r.unwrap()).as_slice::<i32>(), want, "i32 / {d}");
    }
    let mut du: Vec<u64> = (1..=300).collect();
    du.extend([
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) + 1,
        (1 << 63) - 1,
    ]);
    let xsu = dividends(&[0, 1 << 63, 1 << 32], |w| w, 3);
    for &d in &du {
        let r = call(&reg, "div_const_u64", &[u64s(&xsu), u64_s(d)]);
        let want: Vec<u64> = xsu.iter().map(|x| x / d).collect();
        assert_eq!(arr(r.unwrap()).as_slice::<u64>(), want, "u64 / {d}");
    }
    // Every u8 dividend against every u8 divisor.
    let xs8: Vec<u8> = (0..=255).collect();
    for d in 1..=255u8 {
        let r = call(&reg, "div_const_u8", &[u8s(&xs8), u8_s(d)]);
        let want: Vec<u8> = xs8.iter().map(|x| x / d).collect();
        assert_eq!(arr(r.unwrap()).as_slice::<u8>(), want, "u8 / {d}");
    }
}

#[test]
fn div_const_by_zero_is_an_error() {
    let reg = Registry::new();
    let signed = call(&reg, "div_const_i64", &[i64v(&[1]), i64_s(0)]).unwrap_err();
    assert!(signed.contains("division by zero"), "{signed}");
    let unsigned = call(&reg, "div_const_u8", &[u8s(&[]), u8_s(0)]).unwrap_err();
    assert!(unsigned.contains("division by zero"), "{unsigned}");
}

#[test]
fn argsort_is_stable() {
    let r = run_arr("argsort_i64", &[i64v(&[3, 1, 3, 1])]);
    assert_eq!(r.as_slice::<i64>(), &[1, 3, 0, 2]);
    let none = run_arr("argsort_i64", &[i64v(&[])]);
    assert_eq!(none.len(), 0);
    // Signed order, and unsigned order above the sign bit.
    let r32 = run_arr("argsort_i32", &[i32s(&[0, -1, i32::MIN, i32::MAX])]);
    assert_eq!(r32.as_slice::<i64>(), &[2, 1, 0, 3]);
    let r64 = run_arr("argsort_u64", &[u64s(&[1 << 63, 1, 1 << 63, 0])]);
    assert_eq!(r64.as_slice::<i64>(), &[3, 1, 0, 2]);
    let r8 = run_arr("argsort_u8", &[u8s(&[200, 5, 200, 5, 0])]);
    assert_eq!(r8.as_slice::<i64>(), &[4, 1, 3, 0, 2]);
}

/// Every slot of a `KernelTable`, scalar first.
fn all_variants() -> Vec<Variant> {
    let mut all = vec![Variant::SCALAR];
    for isa in [Isa::X86_64V3, Isa::X86_64V4, Isa::Neon] {
        for backend in [Backend::Auto, Backend::Intrinsics] {
            all.push(Variant { isa, backend });
        }
    }
    all
}

#[test]
fn select_falls_back_by_declared_feature() {
    fn entry(_: &[Arg], _: &mut [u8]) -> Result<(), String> {
        Ok(())
    }
    let imp = KernelImpl::Elementwise(entry);
    let v4_auto = Variant {
        isa: Isa::X86_64V4,
        backend: Backend::Auto,
    };
    let v4_intrinsics = Variant {
        isa: Isa::X86_64V4,
        backend: Backend::Intrinsics,
    };
    let vbmi2 = FeatureSet::of(&[Feature::Avx512vbmi2]);
    let mut table = KernelTable::scalar_only(imp);
    table.set(v4_auto, imp);
    table.set_declared(v4_intrinsics, imp, vbmi2);
    let all = FeatureSet::of(&Feature::ALL);
    let pick = |backend, available| select(&table, Isa::X86_64V4, backend, available);
    assert_eq!(pick(Backend::Intrinsics, all), v4_intrinsics);
    assert_eq!(pick(Backend::Intrinsics, all.minus(vbmi2)), v4_auto);
    assert_eq!(pick(Backend::Auto, all), v4_auto);
    let no_dq = all.minus(FeatureSet::of(&[Feature::Avx512dq]));
    assert_eq!(pick(Backend::Intrinsics, no_dq), Variant::SCALAR);
    assert_eq!(
        pick(Backend::Intrinsics, FeatureSet::EMPTY),
        Variant::SCALAR
    );
    // The scalar level resolves to the scalar slot, whatever the backend.
    for backend in [Backend::Auto, Backend::Intrinsics] {
        assert_eq!(select(&table, Isa::Scalar, backend, all), Variant::SCALAR);
    }
}

#[test]
fn detected_level_is_available() {
    assert!(Isa::detect().is_available());
    assert_eq!(Isa::detect(), features::best_level(FeatureSet::detect()));
}

/// Only the x86-64-v4 byte `compress` entries declare a feature beyond their level, and they
/// have an auto entry of the same level to fall back to.
#[cfg(target_arch = "x86_64")]
#[test]
fn byte_compress_declares_vbmi2() {
    let reg = Registry::new();
    let v4_auto = Variant {
        isa: Isa::X86_64V4,
        backend: Backend::Auto,
    };
    let v4_intrinsics = Variant {
        isa: Isa::X86_64V4,
        backend: Backend::Intrinsics,
    };
    let vbmi2 = FeatureSet::of(&[Feature::Avx512vbmi2]);
    let mut declaring = 0;
    for (_, def) in reg.iter() {
        let byte = def.name == "compress_u8" || def.name == "compress_bool";
        for v in all_variants() {
            let want = if byte && v == v4_intrinsics {
                vbmi2
            } else {
                FeatureSet::EMPTY
            };
            assert_eq!(def.impls.declared(v), want, "{} {v}", def.name);
        }
        if byte {
            assert!(def.impls.get(v4_intrinsics).is_some(), "{}", def.name);
            assert!(def.impls.get(v4_auto).is_some(), "{}", def.name);
            declaring += 1;
        }
    }
    assert_eq!(declaring, 2);
}
