use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use super::super::argsort2_reference;
use super::super::radix::{RadixKey, scratch};
use super::super::tests::{Key, MAX_LEN, boundary_pool, pick, sweep_lengths};
use super::{R16, R24, Record, Reduced, radix2_into};
use crate::kernels::scalar::construct::splitmix64;
use crate::kernels::{Backend, FeatureSet, Isa, Registry, Variant};
use crate::value::{Array, Value};

/// Proptest cases per type pair, each drawn from every column class.
const CASES: u32 = 128;

/// Runs `argsort2_<a>_<b>` through the scalar implementation.
fn argsort2<A: Key, B: Key>(a: &[A], b: &[B]) -> Result<Vec<i64>, String> {
    let reg = Registry::new();
    let id = reg
        .by_name(&format!("argsort2_{}_{}", A::TY.name(), B::TY.name()))
        .expect("argsort2 kernel");
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
    assert_eq!(dispatch.variant_used(id), Variant::SCALAR);
    let args = [
        Value::Array(Array::from_vec(a.to_vec())),
        Value::Array(Array::from_vec(b.to_vec())),
    ];
    match dispatch.call(&reg, id, &args)? {
        Value::Array(r) => Ok(r.as_slice::<i64>().to_vec()),
        other => panic!("expected an array, got {other:?}"),
    }
}

/// The radix sort with record type `R`, whatever the length of the input.
fn radix<A: RadixKey, B: RadixKey, R: Record>(a: &[A], b: &[B]) -> Result<Vec<i64>, String> {
    let mut out = vec![0; a.len()];
    radix2_into::<A, B, R>(a, b, &Reduced::of(a, b), &mut out)?;
    Ok(out)
}

/// Compares the kernel and every record layout that holds the composite image with the
/// reference, so both the comparison and the radix paths run at every length.
fn verify<A: Key + RadixKey, B: Key + RadixKey>(a: &[A], b: &[B]) -> Result<(), String> {
    let want = argsort2_reference(a, b);
    let mut got = vec![("kernel", argsort2(a, b)?)];
    got.push(("r24 u32", radix::<A, B, R24<u32>>(a, b)?));
    got.push(("r24 u64", radix::<A, B, R24<u64>>(a, b)?));
    if Reduced::of(a, b).width <= R16::BITS {
        got.push(("r16", radix::<A, B, R16>(a, b)?));
    }
    for (path, out) in got {
        if out != want {
            let at = out.iter().zip(&want).position(|(x, y)| x != y);
            return Err(format!(
                "{path}: {}[{}] x {}[{}] differs from the reference at {at:?}",
                A::TY.name(),
                a.len(),
                B::TY.name(),
                b.len(),
            ));
        }
    }
    Ok(())
}

/// The minimum, the maximum, and their neighbors of `T`, for either signedness.
fn extremes<T: Key>() -> Vec<T> {
    let top = 1u64 << (8 * T::BYTES - 1);
    [0, 1, u64::MAX, top, top - 1]
        .into_iter()
        .map(T::from_bits)
        .collect()
}

/// A column of `len` rows: few distinct values, type extremes, digit boundary values,
/// random values, or small values of both signs.
fn column<T: Key>(len: usize) -> BoxedStrategy<Vec<T>> {
    let random = || any::<u64>().prop_map(T::from_bits);
    prop_oneof![
        (vec(random(), 1..=4), vec(any::<Index>(), len)).prop_map(|(p, i)| pick(p, i)),
        (Just(extremes::<T>()), vec(any::<Index>(), len)).prop_map(|(p, i)| pick(p, i)),
        (Just(boundary_pool::<T>()), vec(any::<Index>(), len)).prop_map(|(p, i)| pick(p, i)),
        vec(random(), len),
        // Sign-extended, so unsigned types see values near their top.
        vec(-300i64..300, len).prop_map(|v| {
            v.into_iter()
                .map(|x| T::from_bits(x.cast_unsigned()))
                .collect()
        }),
    ]
    .boxed()
}

fn run<A: Key + RadixKey, B: Key + RadixKey>() {
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    // Lengths are biased small so that short pairs are well covered.
    let pairs = prop_oneof![0..=300usize, 0..=MAX_LEN]
        .prop_flat_map(|len| (column::<A>(len), column::<B>(len)));
    runner
        .run(&pairs, |(a, b)| verify(&a, &b).map_err(TestCaseError::fail))
        .unwrap();
}

macro_rules! pair_tests {
    ($($name:ident: $a:ty, $b:ty;)*) => {
        $(
            #[test]
            fn $name() {
                run::<$a, $b>();
            }
        )*
    };
}

pair_tests! {
    argsort2_matches_reference_i64_i64: i64, i64;
    argsort2_matches_reference_i64_u64: i64, u64;
    argsort2_matches_reference_i64_i32: i64, i32;
    argsort2_matches_reference_i64_u8: i64, u8;
    argsort2_matches_reference_u64_i64: u64, i64;
    argsort2_matches_reference_u64_u64: u64, u64;
    argsort2_matches_reference_u64_i32: u64, i32;
    argsort2_matches_reference_u64_u8: u64, u8;
    argsort2_matches_reference_i32_i64: i32, i64;
    argsort2_matches_reference_i32_u64: i32, u64;
    argsort2_matches_reference_i32_i32: i32, i32;
    argsort2_matches_reference_i32_u8: i32, u8;
    argsort2_matches_reference_u8_i64: u8, i64;
    argsort2_matches_reference_u8_u64: u8, u64;
    argsort2_matches_reference_u8_i32: u8, i32;
    argsort2_matches_reference_u8_u8: u8, u8;
}

/// Every length of `sweep_lengths`, which includes 1024 and 1025 where the comparison sort
/// hands over to the radix sort, with columns of few values, random values, and boundary
/// values in each combination.
fn sweep<A: Key + RadixKey, B: Key + RadixKey>() {
    for len in sweep_lengths() {
        let seed = u64::try_from(len).unwrap();
        let rows = |salt: u64| {
            (0..seed).map(move |i| splitmix64(seed.wrapping_mul(31).wrapping_add(i) ^ salt))
        };
        fn few<T: Key>(hs: impl Iterator<Item = u64>) -> Vec<T> {
            // Three values with nonzero bytes in every digit.
            hs.map(|h| T::from_bits(0x0101_0101_0101_0101 * (h % 3 + 1)))
                .collect()
        }
        fn bounds<T: Key>(hs: impl Iterator<Item = u64>) -> Vec<T> {
            let pool = boundary_pool::<T>();
            let pool_len = u64::try_from(pool.len()).unwrap();
            hs.map(|h| pool[usize::try_from(h % pool_len).unwrap()])
                .collect()
        }
        let random = |salt| rows(salt).map(A::from_bits).collect::<Vec<A>>();
        let random_b = |salt| rows(salt).map(B::from_bits).collect::<Vec<B>>();
        let cases: [(Vec<A>, Vec<B>); 5] = [
            (few(rows(1)), few(rows(2))),
            (few(rows(3)), random_b(4)),
            (random(5), few(rows(6))),
            (bounds(rows(7)), bounds(rows(8))),
            (random(9), random_b(10)),
        ];
        for (a, b) in cases {
            verify(&a, &b).unwrap_or_else(|e| panic!("len {len}: {e}"));
        }
    }
}

macro_rules! sweep_all {
    ($($a:ty, $b:ty;)*) => {
        $(sweep::<$a, $b>();)*
    };
}

#[test]
fn argsort2_matches_reference_across_lengths() {
    sweep_all! {
        i64, i64; i64, u64; i64, i32; i64, u8;
        u64, i64; u64, u64; u64, i32; u64, u8;
        i32, i64; i32, u64; i32, i32; i32, u8;
        u8, i64; u8, u64; u8, i32; u8, u8;
    }
}

/// Asserts the composite width of `a` and `b`, then the output of every path.
fn at_width<A: Key + RadixKey, B: Key + RadixKey>(a: &[A], b: &[B], width: u32) {
    assert_eq!(Reduced::of(a, b).width, width);
    verify(a, b).unwrap();
}

/// `n` rows drawn from `pool` by a fixed hash, with every pool value in the first rows, so
/// each column's range is that of the pool.
fn drawn<T: Copy>(pool: &[T], n: u64, salt: u64) -> Vec<T> {
    let pool_len = u64::try_from(pool.len()).unwrap();
    (0..n)
        .map(|i| {
            let h = if i < pool_len {
                i
            } else {
                splitmix64(i ^ salt)
            };
            pool[usize::try_from(h % pool_len).unwrap()]
        })
        .collect()
}

#[test]
fn argsort2_layouts_at_composite_widths() {
    for n in [1024, 1025, 3000] {
        // Width 0: both columns constant, so the radix sort has no active digit.
        at_width(&drawn(&[-5i64], n, 1), &drawn(&[9u8], n, 2), 0);
        at_width(&drawn(&[u64::MAX], n, 1), &drawn(&[i32::MIN], n, 2), 0);
        // Width 64: a full-range `u64` with a constant on either side, and two full-range
        // `i32` columns.
        let full64 = [0, 1 << 63, u64::MAX, 7, u64::MAX - 7];
        let full32 = [i32::MIN, -1, 0, 1, i32::MAX];
        at_width(&drawn(&full64, n, 3), &drawn(&[3i32], n, 4), 64);
        at_width(&drawn(&[3i32], n, 3), &drawn(&full64, n, 4), 64);
        at_width(&drawn(&full32, n, 5), &drawn(&full32, n, 6), 64);
        // Width 96: the widest 16-byte layout, a full-range major and minor key.
        at_width(&drawn(&full64, n, 7), &drawn(&full32, n, 8), 96);
        at_width(&drawn(&full32, n, 7), &drawn(&full64, n, 8), 96);
        let full_i64 = [i64::MIN, -1, 0, 1, i64::MAX];
        at_width(&drawn(&full_i64, n, 9), &drawn(&full32, n, 10), 96);
        // Width 97: one bit past the 16-byte layout, on either side.
        let bits33 = [0i64, 1, (1 << 33) - 1, 1 << 32, -5 + (1 << 33)];
        at_width(&drawn(&full64, n, 11), &drawn(&bits33, n, 12), 97);
        at_width(&drawn(&bits33, n, 11), &drawn(&full64, n, 12), 97);
        // Width 128: two full-range 64-bit columns.
        at_width(&drawn(&full_i64, n, 13), &drawn(&full64, n, 14), 128);
    }
}

#[test]
fn argsort2_of_an_empty_pair_is_empty() {
    assert_eq!(argsort2::<i64, u8>(&[], &[]).unwrap(), Vec::<i64>::new());
    assert_eq!(radix::<i64, u8, R16>(&[], &[]).unwrap(), Vec::<i64>::new());
    assert_eq!(Reduced::of::<i64, u8>(&[], &[]).width, 0);
}

#[test]
fn argsort2_of_one_row_is_zero() {
    assert_eq!(argsort2(&[i64::MIN], &[u64::MAX]).unwrap(), [0]);
}

#[test]
fn argsort2_orders_by_the_major_key_first() {
    assert_eq!(argsort2(&[1i64, 2], &[2i64, 1]).unwrap(), [0, 1]);
    assert_eq!(argsort2(&[2i64, 1], &[1i64, 2]).unwrap(), [1, 0]);
}

#[test]
fn argsort2_breaks_major_ties_by_the_minor_key_then_position() {
    assert_eq!(
        argsort2(&[5i32; 5], &[4u8, 3, 2, 1, 0]).unwrap(),
        [4, 3, 2, 1, 0]
    );
    assert_eq!(
        argsort2(&[5i32, 4, 5, 4, 5], &[1u8, 1, 0, 1, 0]).unwrap(),
        [1, 3, 2, 4, 0]
    );
    // The same at radix lengths, with descending minor keys under equal major keys.
    let n = 3000;
    let a: Vec<i32> = (0..n).map(|i| i % 3).collect();
    let b: Vec<u8> = (0..n)
        .map(|i| u8::try_from(255 - i % 256).unwrap())
        .collect();
    verify(&a, &b).unwrap();
}

#[test]
fn argsort2_with_a_u64_column_of_zero_and_max_does_not_wrap() {
    // The reduced range of `a` is `u64::MAX`, the widest a column can have.
    for n in [6, 3000] {
        let a = drawn(&[u64::MAX, 0], n, 1);
        let b = drawn(&[2u8, 1, 0], n, 2);
        assert_eq!(Reduced::of(&a, &b).width, 66);
        verify(&a, &b).unwrap();
    }
    assert_eq!(
        argsort2(&[u64::MAX, 0, u64::MAX, 0], &[0u8, 1, 0, 0]).unwrap(),
        [3, 1, 0, 2]
    );
}

#[test]
fn argsort2_with_constant_columns_matches_reference() {
    let n = 3000;
    let varied = drawn(&[-3i64, 9, i64::MIN, 0], n, 1);
    verify(&varied, &vec![7u64; n_usize(n)]).unwrap();
    verify(&vec![-1i32; n_usize(n)], &varied).unwrap();
    let want: Vec<i64> = (0..).take(n_usize(n)).collect();
    assert_eq!(
        argsort2(&vec![-1i64; n_usize(n)], &vec![0x0101u64; n_usize(n)]).unwrap(),
        want
    );
}

fn n_usize(n: u64) -> usize {
    usize::try_from(n).unwrap()
}

#[test]
fn argsort2_length_mismatch_is_an_error() {
    assert_eq!(
        argsort2(&[1i64, 2], &[1u8]).unwrap_err(),
        "length mismatch: 2 vs 1"
    );
    assert_eq!(
        argsort2::<u8, i32>(&[], &[1]).unwrap_err(),
        "length mismatch: 0 vs 1"
    );
}

#[test]
fn argsort2_records_have_the_specified_sizes() {
    assert_eq!(size_of::<R16>(), 16);
    assert_eq!(size_of::<R24<u32>>(), 24);
    assert_eq!(size_of::<R24<u64>>(), 24);
}

#[test]
fn argsort2_scratch_allocation_failure_is_an_error() {
    // No allocator grants this many records, so the reservation fails instead of aborting.
    let narrow = scratch::<R16>(usize::MAX / 32, true)
        .err()
        .expect("reservation fails");
    assert!(narrow.contains("cannot allocate"), "{narrow}");
    let wide = scratch::<R24<u64>>(usize::MAX / 48, true)
        .err()
        .expect("reservation fails");
    assert!(wide.contains("cannot allocate"), "{wide}");
    assert!(scratch::<R16>(usize::MAX, false).unwrap().is_empty());
}
