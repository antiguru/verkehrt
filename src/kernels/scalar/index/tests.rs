use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use super::argsort_reference;
use crate::kernels::{Backend, FeatureSet, Isa, Registry, Variant};
use crate::value::{Array, Elem, ScalarType, Value};

/// Longest generated array, past the sizes where a small-input path would hand over.
const MAX_LEN: usize = 5000;
/// Proptest cases per element type, each drawn from every input class.
const CASES: u32 = 64;

/// Runs `argsort_<ty>` through the scalar implementation.
fn argsort<T: Elem>(ty: ScalarType, xs: &[T]) -> Vec<i64> {
    let reg = Registry::new();
    let id = reg
        .by_name(&format!("argsort_{}", ty.name()))
        .expect("argsort kernel");
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
    assert_eq!(dispatch.variant_used(id), Variant::SCALAR);
    let args = [Value::Array(Array::from_vec(xs.to_vec()))];
    match dispatch.call(&reg, id, &args).expect("argsort succeeds") {
        Value::Array(a) => a.as_slice::<i64>().to_vec(),
        other => panic!("expected an array, got {other:?}"),
    }
}

fn check<T: Elem + Ord>(ty: ScalarType, xs: &[T]) -> Result<(), TestCaseError> {
    prop_assert_eq!(argsort(ty, xs), argsort_reference(xs));
    Ok(())
}

/// Arrays of length `0..=MAX_LEN` in the classes of the review focus: few distinct values,
/// signed extremes, sorted, reverse sorted, and random.
fn arrays<T>(
    random: impl Strategy<Value = T> + Clone + 'static,
    extremes: Vec<T>,
) -> impl Strategy<Value = Vec<T>>
where
    T: Clone + Ord + std::fmt::Debug + 'static,
{
    let len = 0..=MAX_LEN;
    let few = {
        let random = random.clone();
        (
            vec(random, 1..=4),
            vec(any::<prop::sample::Index>(), len.clone()),
        )
            .prop_map(|(pool, picks)| {
                picks
                    .into_iter()
                    .map(|p| p.get(&pool).clone())
                    .collect::<Vec<_>>()
            })
    };
    let extreme = {
        let extremes = Just(extremes);
        (extremes, vec(any::<prop::sample::Index>(), len.clone())).prop_map(|(pool, picks)| {
            picks
                .into_iter()
                .map(|p| p.get(&pool).clone())
                .collect::<Vec<_>>()
        })
    };
    let sorted = vec(random.clone(), len.clone()).prop_map(|mut v| {
        v.sort();
        v
    });
    let reversed = vec(random.clone(), len.clone()).prop_map(|mut v| {
        v.sort_by(|a, b| b.cmp(a));
        v
    });
    let plain = vec(random, len);
    prop_oneof![few, extreme, sorted, reversed, plain]
}

fn run<T>(ty: ScalarType, strategy: impl Strategy<Value = Vec<T>>)
where
    T: Elem + Ord + std::fmt::Debug,
{
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    runner.run(&strategy, |xs| check(ty, &xs)).unwrap();
}

#[test]
fn argsort_matches_reference_i64() {
    run(
        ScalarType::I64,
        arrays(any::<i64>(), vec![i64::MIN, i64::MAX, -1, 0, 1]),
    );
}

#[test]
fn argsort_matches_reference_u64() {
    run(
        ScalarType::U64,
        arrays(any::<u64>(), vec![0, 1, u64::MAX, 1 << 63, (1 << 63) - 1]),
    );
}

#[test]
fn argsort_matches_reference_i32() {
    run(
        ScalarType::I32,
        arrays(any::<i32>(), vec![i32::MIN, i32::MAX, -1, 0, 1]),
    );
}

#[test]
fn argsort_matches_reference_u8() {
    run(
        ScalarType::U8,
        arrays(any::<u8>(), vec![0, 1, 127, 128, 255]),
    );
}

#[test]
fn argsort_signed_extremes_sort_in_signed_order() {
    let i64s = [i64::MAX, 0, i64::MIN, -1, i64::MAX, i64::MIN, 0, -1];
    assert_eq!(argsort(ScalarType::I64, &i64s), [2, 5, 3, 7, 1, 6, 0, 4]);
    let i32s = [i32::MAX, 0, i32::MIN, -1, i32::MAX, i32::MIN, 0, -1];
    assert_eq!(argsort(ScalarType::I32, &i32s), [2, 5, 3, 7, 1, 6, 0, 4]);
    // Unsigned keys above the sign bit stay above it.
    let u64s = [u64::MAX, 1 << 63, 0, (1 << 63) - 1, u64::MAX];
    assert_eq!(argsort(ScalarType::U64, &u64s), [2, 3, 1, 0, 4]);
    let u8s = [255u8, 128, 0, 127, 255];
    assert_eq!(argsort(ScalarType::U8, &u8s), [2, 3, 1, 0, 4]);
}

#[test]
fn argsort_of_all_equal_rows_is_the_identity() {
    let n = 100_000;
    let want: Vec<i64> = (0..).take(n).collect();
    assert_eq!(argsort(ScalarType::I64, &vec![7i64; n]), want);
    assert_eq!(argsort(ScalarType::U64, &vec![7u64; n]), want);
    assert_eq!(argsort(ScalarType::I32, &vec![-7i32; n]), want);
    assert_eq!(argsort(ScalarType::U8, &vec![7u8; n]), want);
}

#[test]
fn argsort_of_tiny_arrays_matches_reference() {
    for len in 0..=3usize {
        let xs: Vec<i64> = (0..len).map(|i| i64::from(len > 1 && i == 0)).collect();
        assert_eq!(argsort(ScalarType::I64, &xs), argsort_reference(&xs));
    }
}
