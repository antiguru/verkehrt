use std::ops::RangeInclusive;

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use super::argsort_reference;
use super::radix::{RadixKey, radix_into, scratch};
use crate::kernels::scalar::construct::splitmix64;
use crate::kernels::{Backend, FeatureSet, Isa, Registry, Variant};
use crate::value::{Array, Elem, ScalarType, Value};

/// Longest generated array, past the sizes where a small-input path would hand over.
const MAX_LEN: usize = 5000;
/// Proptest cases per element type, each drawn from every input class.
const CASES: u32 = 256;

/// A sortable key type that tests can build from the low bits of a `u64`.
trait Key: Elem + Ord + std::fmt::Debug + 'static {
    const TY: ScalarType;
    const BYTES: usize;
    const SIGNED: bool;
    /// Keeps the low `8 * BYTES` bits of `v`.
    fn from_bits(v: u64) -> Self;
}

macro_rules! key {
    ($t:ty, $ty:expr, $signed:expr) => {
        impl Key for $t {
            const TY: ScalarType = $ty;
            const BYTES: usize = std::mem::size_of::<$t>();
            const SIGNED: bool = $signed;
            fn from_bits(v: u64) -> Self {
                // Truncation is the point, and sign extension in `v` keeps negatives negative.
                v as $t
            }
        }
    };
}

key!(i64, ScalarType::I64, true);
key!(u64, ScalarType::U64, false);
key!(i32, ScalarType::I32, true);
key!(u8, ScalarType::U8, false);

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

fn check<T: Key>(xs: &[T]) -> Result<(), TestCaseError> {
    prop_assert_eq!(argsort(T::TY, xs), argsort_reference(xs));
    Ok(())
}

/// Values at and around every digit boundary: `2^(8k)` and `2^(8k) - 1` for each byte `k`,
/// `2^(11k) +- 1` for 11-bit digits, and the negatives of all of them for signed types.
fn boundary_pool<T: Key>() -> Vec<T> {
    let bits = 8 * T::BYTES;
    let mut raw = vec![0u64, 1];
    for k in 0..T::BYTES {
        raw.push(1 << (8 * k));
        raw.push((1u64 << (8 * k)).wrapping_sub(1));
    }
    for k in (0..).take_while(|k| 11 * k < bits) {
        raw.push((1u64 << (11 * k)).wrapping_add(1));
        raw.push((1u64 << (11 * k)).wrapping_sub(1));
    }
    if T::SIGNED {
        raw.extend(raw.clone().into_iter().map(u64::wrapping_neg));
    }
    raw.into_iter().map(T::from_bits).collect()
}

fn pick<T: Clone>(pool: Vec<T>, picks: Vec<Index>) -> Vec<T> {
    picks.into_iter().map(|p| p.get(&pool).clone()).collect()
}

/// Arrays in the classes of the review focus: few distinct values, extremes, sorted,
/// reverse sorted, random, small ranges, keys that differ in exactly one byte, and digit
/// boundary values. Lengths are biased small so that short arrays are well covered.
fn arrays<T: Key>(extremes: Vec<T>) -> impl Strategy<Value = Vec<T>> {
    let random = || any::<u64>().prop_map(T::from_bits);
    // Each class is built twice, with lengths up to 300 and up to `MAX_LEN`.
    let sized = |class: &dyn Fn(RangeInclusive<usize>) -> BoxedStrategy<Vec<T>>| {
        prop_oneof![class(0..=300), class(0..=MAX_LEN)].boxed()
    };
    let few = sized(&|len| {
        (vec(random(), 1..=4), vec(any::<Index>(), len))
            .prop_map(|(p, i)| pick(p, i))
            .boxed()
    });
    let extreme = sized(&|len| {
        (Just(extremes.clone()), vec(any::<Index>(), len))
            .prop_map(|(p, i)| pick(p, i))
            .boxed()
    });
    let sorted = sized(&|len| {
        vec(random(), len)
            .prop_map(|mut v| {
                v.sort();
                v
            })
            .boxed()
    });
    let reversed = sized(&|len| {
        vec(random(), len)
            .prop_map(|mut v| {
                v.sort_by(|a, b| b.cmp(a));
                v
            })
            .boxed()
    });
    let plain = sized(&|len| vec(random(), len).boxed());
    let small = sized(&|len| {
        vec(0u64..300, len)
            .prop_map(|v| v.into_iter().map(T::from_bits).collect())
            .boxed()
    });
    let small_signed = sized(&|len| {
        // Sign-extended, so unsigned types see values near their top.
        vec(-300i64..300, len)
            .prop_map(|v| {
                v.into_iter()
                    .map(|x| T::from_bits(x.cast_unsigned()))
                    .collect()
            })
            .boxed()
    });
    // Keys share a random base and differ in exactly one random byte.
    let one_byte = sized(&|len| {
        (any::<u64>(), 0..T::BYTES, vec(any::<u8>(), len))
            .prop_map(|(base, k, ds)| {
                ds.into_iter()
                    .map(|d| T::from_bits(base ^ (u64::from(d) << (8 * k))))
                    .collect()
            })
            .boxed()
    });
    let boundary = sized(&|len| {
        (Just(boundary_pool::<T>()), vec(any::<Index>(), len))
            .prop_map(|(p, i)| pick(p, i))
            .boxed()
    });
    prop_oneof![
        few,
        extreme,
        sorted,
        reversed,
        plain,
        small,
        small_signed,
        one_byte,
        boundary
    ]
}

fn run<T: Key>(extremes: Vec<T>) {
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    runner.run(&arrays(extremes), |xs| check(&xs)).unwrap();
}

#[test]
fn argsort_matches_reference_i64() {
    run::<i64>(vec![i64::MIN, i64::MAX, -1, 0, 1]);
}

#[test]
fn argsort_matches_reference_u64() {
    run::<u64>(vec![0, 1, u64::MAX, 1 << 63, (1 << 63) - 1]);
}

#[test]
fn argsort_matches_reference_i32() {
    run::<i32>(vec![i32::MIN, i32::MAX, -1, 0, 1]);
}

#[test]
fn argsort_matches_reference_u8() {
    run::<u8>(vec![0, 1, 127, 128, 255]);
}

/// Every length around the places where a small-input path or a pass structure could change
/// behavior: `0..=300` and `2^k +- 1` up to `MAX_LEN`.
fn sweep_lengths() -> Vec<usize> {
    let mut lens: Vec<usize> = (0..=300).collect();
    for k in 0..13 {
        for len in [(1usize << k) - 1, 1 << k, (1 << k) + 1] {
            if len <= MAX_LEN {
                lens.push(len);
            }
        }
    }
    lens.sort_unstable();
    lens.dedup();
    lens
}

fn sweep<T: Key>() {
    let pool = boundary_pool::<T>();
    let pool_len = u64::try_from(pool.len()).unwrap();
    for len in sweep_lengths() {
        let seed = u64::try_from(len).unwrap();
        let rows = || (0..seed).map(|i| splitmix64(seed.wrapping_mul(31).wrapping_add(i)));
        // Three values with nonzero bytes in every digit.
        let few: Vec<T> = rows()
            .map(|h| T::from_bits(0x0101_0101_0101_0101 * (h % 3 + 1)))
            .collect();
        let random: Vec<T> = rows().map(T::from_bits).collect();
        let bounds: Vec<T> = rows()
            .map(|h| pool[usize::try_from(h % pool_len).unwrap()])
            .collect();
        for xs in [few, random, bounds] {
            assert_eq!(argsort(T::TY, &xs), argsort_reference(&xs), "len {len}");
        }
    }
}

#[test]
fn argsort_matches_reference_across_lengths() {
    sweep::<i64>();
    sweep::<u64>();
    sweep::<i32>();
    sweep::<u8>();
}

#[test]
fn argsort_signed_extremes_sort_in_signed_order() {
    let i64s = [i64::MAX, 0, i64::MIN, -1, i64::MAX, i64::MIN, 0, -1];
    assert_eq!(argsort(ScalarType::I64, &i64s), [2, 5, 3, 7, 1, 6, 0, 4]);
    let i32s = [i32::MAX, 0, i32::MIN, -1, i32::MAX, i32::MIN, 0, -1];
    assert_eq!(argsort(ScalarType::I32, &i32s), [2, 5, 3, 7, 1, 6, 0, 4]);
    // Neighbors of the extremes, mixed with positives whose low bytes equal those of the
    // negatives (`-2` and `0xfe`, `i64::MIN + 1` and `1`).
    let near64 = [0xfe, -2, i64::MIN + 1, 1, i64::MAX, -1, 0xff, i64::MIN];
    assert_eq!(argsort(ScalarType::I64, &near64), [7, 2, 1, 5, 3, 0, 6, 4]);
    let near32 = [0xfe, -2, i32::MIN + 1, 1, i32::MAX, -1, 0xff, i32::MIN];
    assert_eq!(argsort(ScalarType::I32, &near32), [7, 2, 1, 5, 3, 0, 6, 4]);
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
    // Constants with nonzero bytes in every digit, so no pass sees a zero digit.
    assert_eq!(argsort(ScalarType::I64, &vec![-1i64; n]), want);
    assert_eq!(
        argsort(ScalarType::U64, &vec![0x0101_0101_0101_0101u64; n]),
        want
    );
    assert_eq!(argsort(ScalarType::I32, &vec![-7i32; n]), want);
    assert_eq!(argsort(ScalarType::U8, &vec![7u8; n]), want);
}

/// The radix path on its own, with 32-bit positions and with the 64-bit positions that only
/// inputs above `u32::MAX` rows reach through the kernel, at every generated length, including
/// those below the small-input switch point.
fn radix_paths<T: Key + RadixKey>(extremes: Vec<T>) {
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    runner
        .run(&arrays(extremes), |xs| {
            let want = argsort_reference(&xs);
            let mut narrow = vec![0; xs.len()];
            radix_into::<T, u32>(&xs, &mut narrow).map_err(TestCaseError::fail)?;
            prop_assert_eq!(&narrow, &want);
            let mut wide = vec![0; xs.len()];
            radix_into::<T, u64>(&xs, &mut wide).map_err(TestCaseError::fail)?;
            prop_assert_eq!(&wide, &want);
            Ok(())
        })
        .unwrap();
}

#[test]
fn radix_paths_match_reference() {
    radix_paths::<i64>(vec![i64::MIN, i64::MAX, -1, 0, 1]);
    radix_paths::<u64>(vec![0, 1, u64::MAX, 1 << 63, (1 << 63) - 1]);
    radix_paths::<i32>(vec![i32::MIN, i32::MAX, -1, 0, 1]);
    radix_paths::<u8>(vec![0, 1, 127, 128, 255]);
}

#[test]
fn argsort_scratch_allocation_failure_is_an_error() {
    // No allocator grants this many pairs, so the reservation fails instead of aborting.
    let err = scratch::<(u64, u32)>(usize::MAX / 32, true).unwrap_err();
    assert!(err.contains("cannot allocate"), "{err}");
    assert!(scratch::<(u64, u32)>(usize::MAX, false).unwrap().is_empty());
}
