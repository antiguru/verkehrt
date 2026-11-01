use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::ops::RangeInclusive;

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestRunner};

use super::{GroupKey, INITIAL_SLOTS, JUMP_BITS, MUL, hashed_into, mix, table};
use crate::kernels::scalar::construct::splitmix64;
use crate::kernels::{Backend, FeatureSet, Isa, Registry, Variant};
use crate::value::{Array, Elem, ScalarType, Value};

/// Longest generated column, past several growth thresholds of the table.
const MAX_LEN: usize = 5000;
/// Proptest cases per key type, each drawn from every input class.
const CASES: u32 = 128;
/// Proptest cases per pair of key types.
const PAIR_CASES: u32 = 32;

/// The oracle: first-occurrence ids from a `HashMap`.
fn group_id_reference<K: Hash + Eq + Copy>(keys: &[K]) -> Vec<i64> {
    let mut ids: HashMap<K, i64> = HashMap::new();
    keys.iter()
        .map(|&k| {
            let next = i64::try_from(ids.len()).unwrap();
            match ids.entry(k) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => *e.insert(next),
            }
        })
        .collect()
}

/// A key type that tests can build from the low bits of a `u64`.
trait Key: Elem + Eq + Hash + std::fmt::Debug + 'static {
    const TY: ScalarType;
    /// Keeps the low bits of `v`.
    fn from_bits(v: u64) -> Self;
}

macro_rules! key {
    ($t:ty, $ty:expr) => {
        impl Key for $t {
            const TY: ScalarType = $ty;
            fn from_bits(v: u64) -> Self {
                // Truncation is the point, and sign extension in `v` keeps negatives negative.
                v as $t
            }
        }
    };
}

key!(i64, ScalarType::I64);
key!(u64, ScalarType::U64);
key!(i32, ScalarType::I32);
key!(u8, ScalarType::U8);

/// Runs the kernel `name` through its scalar implementation.
fn call(name: &str, args: &[Value]) -> Result<Vec<i64>, String> {
    let reg = Registry::new();
    let id = reg
        .by_name(name)
        .unwrap_or_else(|| panic!("no kernel {name}"));
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
    assert_eq!(dispatch.variant_used(id), Variant::SCALAR);
    match dispatch.call(&reg, id, args)? {
        Value::Array(a) => {
            assert_eq!(a.elem(), ScalarType::I64);
            Ok(a.as_slice::<i64>().to_vec())
        }
        other => panic!("expected an array, got {other:?}"),
    }
}

fn col<T: Elem>(xs: &[T]) -> Value {
    Value::Array(Array::from_vec(xs.to_vec()))
}

/// `group_id_<t>` on `xs`.
fn group_id<T: Key>(xs: &[T]) -> Vec<i64> {
    call(&format!("group_id_{}", T::TY.name()), &[col(xs)]).expect("group_id succeeds")
}

/// `group_id2_<a>_<b>` on `xs` and `ys`.
fn group_id2<A: Key, B: Key>(xs: &[A], ys: &[B]) -> Result<Vec<i64>, String> {
    let name = format!("group_id2_{}_{}", A::TY.name(), B::TY.name());
    call(&name, &[col(xs), col(ys)])
}

fn pairs<A: Copy, B: Copy>(xs: &[A], ys: &[B]) -> Vec<(A, B)> {
    xs.iter().copied().zip(ys.iter().copied()).collect()
}

fn pick<T: Clone>(pool: &[T], picks: Vec<Index>) -> Vec<T> {
    picks.into_iter().map(|p| p.get(pool).clone()).collect()
}

/// Values around both signs and the extremes: 0, 1, -1, the minimum and maximum, values
/// that differ only in the sign bit, and values that differ only in the high half.
fn signed_pool<T: Key>() -> Vec<T> {
    let raw: [u64; 12] = [
        0,
        1,
        u64::MAX,
        1 << 63,
        (1 << 63) - 1,
        (1 << 63) | 1,
        1 << 31,
        (1 << 31) - 1,
        u64::MAX << 31,
        1 << 32,
        (1 << 32) | 1,
        0xFFFF_FFFF,
    ];
    raw.into_iter().map(T::from_bits).collect()
}

/// Key columns in the classes of the review focus: all distinct (random), few distinct
/// values, both signs and extremes, small ranges of both signs, and keys repeated far apart.
/// Lengths up to `MAX_LEN` cross several growth thresholds.
fn columns<T: Key>() -> impl Strategy<Value = Vec<T>> {
    let random = || any::<u64>().prop_map(T::from_bits);
    // Each class is built twice, with lengths up to 300 and up to `MAX_LEN`.
    let sized = |class: &dyn Fn(RangeInclusive<usize>) -> BoxedStrategy<Vec<T>>| {
        prop_oneof![class(0..=300), class(0..=MAX_LEN)].boxed()
    };
    let plain = sized(&|len| vec(random(), len).boxed());
    let few = sized(&|len| {
        (vec(random(), 1..=8), vec(any::<Index>(), len))
            .prop_map(|(p, i)| pick(&p, i))
            .boxed()
    });
    let signed = sized(&|len| {
        vec(any::<Index>(), len)
            .prop_map(|i| pick(&signed_pool::<T>(), i))
            .boxed()
    });
    let small = sized(&|len| {
        vec(-200i64..200, len)
            .prop_map(|v| {
                v.into_iter()
                    .map(|x| T::from_bits(x.cast_unsigned()))
                    .collect()
            })
            .boxed()
    });
    // A distinct prefix followed by the same keys in another order, so every key recurs
    // after the table has grown.
    let far = sized(&|len| {
        (vec(random(), len), any::<u64>())
            .prop_map(|(mut v, seed)| {
                let half = v.len() / 2;
                for i in half..v.len() {
                    let j = splitmix64(seed ^ u64::try_from(i).unwrap())
                        % u64::try_from(half.max(1)).unwrap();
                    v[i] = v[usize::try_from(j).unwrap().min(half.saturating_sub(1))];
                }
                v
            })
            .boxed()
    });
    prop_oneof![plain, few, signed, small, far]
}

fn run_single<T: Key>() {
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    runner
        .run(&columns::<T>(), |xs| {
            prop_assert_eq!(group_id(&xs), group_id_reference(&xs));
            Ok(())
        })
        .unwrap();
}

#[test]
fn group_id_matches_reference_i64() {
    run_single::<i64>();
}

#[test]
fn group_id_matches_reference_u64() {
    run_single::<u64>();
}

#[test]
fn group_id_matches_reference_i32() {
    run_single::<i32>();
}

#[test]
fn group_id_matches_reference_u8() {
    run_single::<u8>();
}

/// Pairs of equally long columns. Besides independent columns, the second column can be a
/// function of the first (swapped, so (1, 2) and (2, 1) both occur), and both columns can
/// draw from few values, so pairs repeat while neither column alone is unique.
fn pair_columns<A: Key, B: Key>() -> impl Strategy<Value = (Vec<A>, Vec<B>)> {
    let independent = (0..=MAX_LEN).prop_flat_map(|n| (columns_of::<A>(n), columns_of::<B>(n)));
    let swapped = vec((0u64..6, 0u64..6), 0..=MAX_LEN).prop_map(|v| {
        // Each pair also occurs swapped.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (a, b) in v {
            xs.extend([A::from_bits(a), A::from_bits(b)]);
            ys.extend([B::from_bits(b), B::from_bits(a)]);
        }
        (xs, ys)
    });
    let repeated =
        (vec(any::<u64>(), 1..=600), vec(any::<Index>(), 0..=MAX_LEN)).prop_map(|(pool, picks)| {
            // Few distinct values per column, so pairs repeat far apart.
            let xs = picks
                .iter()
                .map(|p| A::from_bits(pool[p.index(pool.len())] % 7))
                .collect();
            let ys = picks
                .iter()
                .map(|p| B::from_bits(pool[p.index(pool.len())].rotate_left(17) % 5))
                .collect();
            (xs, ys)
        });
    prop_oneof![independent, swapped, repeated]
}

/// A column of exactly `n` rows from the first classes of `columns`.
fn columns_of<T: Key>(n: usize) -> BoxedStrategy<Vec<T>> {
    let random = any::<u64>().prop_map(T::from_bits);
    prop_oneof![
        vec(random.clone(), n),
        (vec(random, 1..=8), vec(any::<Index>(), n)).prop_map(|(p, i)| pick(&p, i)),
        vec(any::<Index>(), n).prop_map(|i| pick(&signed_pool::<T>(), i)),
    ]
    .boxed()
}

fn run_pair<A: Key, B: Key>() {
    let mut runner = TestRunner::new(Config {
        cases: PAIR_CASES,
        ..Config::default()
    });
    runner
        .run(&pair_columns::<A, B>(), |(xs, ys)| {
            let got = group_id2(&xs, &ys).map_err(TestCaseError::fail)?;
            prop_assert_eq!(got, group_id_reference(&pairs(&xs, &ys)));
            Ok(())
        })
        .unwrap();
}

/// `run_pair` for first column type `$a` and every second column type.
macro_rules! pair_tests {
    ($name:ident, $a:ty) => {
        #[test]
        fn $name() {
            run_pair::<$a, i64>();
            run_pair::<$a, u64>();
            run_pair::<$a, i32>();
            run_pair::<$a, u8>();
        }
    };
}

pair_tests!(group_id2_matches_reference_i64, i64);
pair_tests!(group_id2_matches_reference_u64, u64);
pair_tests!(group_id2_matches_reference_i32, i32);
pair_tests!(group_id2_matches_reference_u8, u8);

#[test]
fn empty_and_single_row() {
    assert_eq!(group_id::<i64>(&[]), Vec::<i64>::new());
    assert_eq!(group_id::<u8>(&[]), Vec::<i64>::new());
    assert_eq!(group_id::<i32>(&[-5]), [0]);
    assert_eq!(group_id::<u8>(&[200]), [0]);
    assert_eq!(group_id2::<i64, u8>(&[], &[]).unwrap(), Vec::<i64>::new());
    assert_eq!(group_id2::<u64, i32>(&[7], &[-7]).unwrap(), [0]);
}

#[test]
fn swapped_pairs_are_distinct() {
    let xs = [1i64, 2, 1, 2, 1];
    let ys = [2i64, 1, 2, 1, 1];
    assert_eq!(group_id2(&xs, &ys).unwrap(), [0, 1, 0, 1, 2]);
    // Neither column alone separates the pairs.
    assert_eq!(group_id(&xs), [0, 1, 0, 1, 0]);
}

#[test]
fn signed_keys_by_value() {
    let xs = [-1i64, 1, -1, i64::MIN, 0, i64::MAX, i64::MIN, 0, 1];
    assert_eq!(group_id(&xs), [0, 1, 0, 2, 3, 4, 2, 3, 1]);
    // The same bit patterns in a narrower type are different keys from their extension.
    let ys = [-1i32, 1, i32::MIN, 0, -1, i32::MAX];
    assert_eq!(group_id(&ys), [0, 1, 2, 3, 0, 4]);
    let us = [u64::MAX, 1 << 63, u64::MAX, 0, 1 << 63];
    assert_eq!(group_id(&us), [0, 1, 0, 2, 1]);
}

#[test]
fn u8_every_value_twice() {
    // Descending, then ascending, so the ids are not the values.
    let mut xs: Vec<u8> = (0..=255).rev().collect();
    xs.extend(0..=255);
    let want: Vec<i64> = (0..256).chain((0..256).rev()).collect();
    assert_eq!(group_id(&xs), want);
}

#[test]
fn length_mismatch() {
    assert_eq!(
        group_id2::<i64, u8>(&[1, 2, 3], &[1, 2]).unwrap_err(),
        "length mismatch: 3 vs 2"
    );
    assert_eq!(
        group_id2::<u8, i32>(&[], &[1]).unwrap_err(),
        "length mismatch: 0 vs 1"
    );
}

#[test]
fn wrong_arguments() {
    assert_eq!(
        call("group_id_i64", &[col(&[1i64]), col(&[1i64])]).unwrap_err(),
        "expected 1 arguments, got 2"
    );
    assert_eq!(
        call("group_id2_i64_i64", &[col(&[1i64])]).unwrap_err(),
        "expected 2 arguments, got 1"
    );
    assert_eq!(
        call("group_id_i32", &[col(&[1i64])]).unwrap_err(),
        "expected i32[], got i64[]"
    );
}

/// Every length around the growth thresholds of the table: more than half of
/// `INITIAL_SLOTS * 2^k` distinct keys, each followed by all keys again in reverse order,
/// so every id is looked up after the last rehash.
#[test]
fn growth_thresholds_keep_first_occurrence_order() {
    let mut lens = vec![0, 1, 2, 3];
    let mut slots = INITIAL_SLOTS;
    while slots / 2 <= MAX_LEN {
        let half = slots / 2;
        lens.extend([half - 1, half, half + 1]);
        slots *= 2;
    }
    for len in lens {
        let seed = u64::try_from(len).unwrap();
        // `splitmix64` is a bijection, so the keys are distinct.
        let keys: Vec<u64> = (0..seed).map(|i| splitmix64(seed << 32 | i)).collect();
        let mut xs = keys.clone();
        xs.extend(keys.iter().rev());
        let n = i64::try_from(len).unwrap();
        let want: Vec<i64> = (0..n).chain((0..n).rev()).collect();
        assert_eq!(group_id(&xs), want, "len {len}");
        let signed: Vec<i64> = xs.iter().map(|&x| x.cast_signed()).collect();
        assert_eq!(group_id(&signed), want, "len {len}");
        let narrow: Vec<i32> = (0..len)
            .map(|i| i32::try_from(i).unwrap() - 2000)
            .chain((0..len).rev().map(|i| i32::try_from(i).unwrap() - 2000))
            .collect();
        assert_eq!(group_id(&narrow), want, "len {len}");
        let ys: Vec<u8> = xs.iter().map(|&x| x.to_le_bytes()[0]).collect();
        assert_eq!(
            group_id2(&xs, &ys).unwrap(),
            group_id_reference(&pairs(&xs, &ys)),
            "len {len}"
        );
    }
}

/// The same thresholds with every key twice in a row, so under 7/8 of the rows open a group
/// and every growth doubles instead of jumping.
#[test]
fn doubling_growth_keeps_first_occurrence_order() {
    let mut lens = Vec::new();
    let mut slots = INITIAL_SLOTS;
    while slots / 2 <= MAX_LEN {
        let half = slots / 2;
        lens.extend([half - 1, half, half + 1]);
        slots *= 2;
    }
    for len in lens {
        let seed = u64::try_from(len).unwrap();
        let xs: Vec<u64> = (0..seed)
            .flat_map(|i| {
                let k = splitmix64(seed << 32 | i);
                [k, k]
            })
            .collect();
        let want: Vec<i64> = (0..i64::try_from(len).unwrap())
            .flat_map(|i| [i, i])
            .collect();
        assert_eq!(group_id(&xs), want, "len {len}");
    }
}

/// Every length around the thresholds of the `2^JUMP_BITS`-fold growths of a column that is
/// all distinct so far, up to a table of 1M slots, each followed by the keys in reverse. The
/// rows are twice the keys, so a growth just above a threshold meets the cap of the table
/// for twice the rows.
#[test]
fn jump_thresholds_keep_first_occurrence_order() {
    let mut lens = Vec::new();
    let mut slots = INITIAL_SLOTS;
    while slots <= 1 << 20 {
        let half = slots / 2;
        lens.extend([half - 1, half, half + 1]);
        slots <<= JUMP_BITS;
    }
    for len in lens {
        let seed = u64::try_from(len).unwrap();
        let keys: Vec<u64> = (0..seed).map(|i| splitmix64(seed << 32 | i)).collect();
        let mut xs = keys.clone();
        xs.extend(keys.iter().rev());
        let n = i64::try_from(len).unwrap();
        let want: Vec<i64> = (0..n).chain((0..n).rev()).collect();
        assert_eq!(group_id(&xs), want, "len {len}");
    }
}

/// A column whose first rows repeat keys, so that the first growths double. The repeated
/// prefix is diluted by 600K distinct keys, which end in jumps.
#[test]
fn doubling_growth_to_a_large_table() {
    let few = 1000;
    let groups = 600_000;
    let mut xs: Vec<u64> = (0..few).flat_map(|i| [splitmix64(i); 8]).collect();
    xs.extend((few..groups).map(splitmix64));
    assert_eq!(group_id(&xs), group_id_reference(&xs));
}

/// A column that opens many groups at first, so that its table jumps, and then only repeats
/// them.
#[test]
fn distinct_prefix_then_repeats() {
    let keys: Vec<u64> = (0..70_000).map(splitmix64).collect();
    let mut xs = keys.clone();
    for _ in 0..5 {
        xs.extend(keys.iter().rev());
    }
    assert_eq!(group_id(&xs), group_id_reference(&xs));
}

/// The table on hashes that collide everywhere: every key probes from the same slot, so
/// probing must still separate the keys and terminate across growth.
#[test]
fn degenerate_hashes() {
    let constant = |_: u64| 0u64;
    let top = |_: u64| u64::MAX;
    let two = |k: u64| (k & 1) << 63;
    for len in [0, 1, 2, 3, 127, 128, 129, 300, 1100] {
        let seed = u64::try_from(len).unwrap();
        let keys: Vec<u64> = (0..seed)
            .map(|i| splitmix64(i) % (seed / 2 + 1))
            .chain(0..seed)
            .collect();
        let want = group_id_reference(&keys);
        for hash in [&constant as &dyn Fn(u64) -> u64, &top, &two] {
            let mut out = vec![-1; keys.len()];
            hashed_into(keys.iter().copied(), keys.len(), hash, &mut out).unwrap();
            assert_eq!(out, want, "len {len}");
        }
    }
}

/// Runs `hashed_into` on `keys` under the hash with the fixed multiplier `MUL`.
fn fixed_hash_ids<K: GroupKey>(keys: &[K]) -> Vec<i64> {
    let mut out = vec![-1; keys.len()];
    hashed_into(
        keys.iter().copied(),
        keys.len(),
        |k| k.key_hash(MUL),
        &mut out,
    )
    .unwrap();
    out
}

/// Keys whose hash under `MUL` picks slot 0 in every table of up to `2^12` slots, the worst
/// case for linear probing, which a call with a random multiplier does not meet.
#[test]
fn colliding_keys_under_the_real_hash() {
    let bits = 12;
    let i64s: Vec<i64> = (0u64..)
        .filter(|&k| mix(k, MUL) >> (64 - bits) == 0)
        .take(1500)
        .map(u64::cast_signed)
        .collect();
    // The same keys repeated in reverse order, then their negations, which collide less.
    let mut xs = i64s.clone();
    xs.extend(i64s.iter().rev());
    xs.extend(i64s.iter().map(|k| k.wrapping_neg()));
    assert_eq!(fixed_hash_ids(&xs), group_id_reference(&xs));
    assert_eq!(group_id(&xs), group_id_reference(&xs));
    let i32s: Vec<i32> = (0u32..)
        .filter(|&k| mix(u64::from(k), MUL) >> (64 - bits) == 0)
        .take(1500)
        .map(u32::cast_signed)
        .collect();
    let mut ys = i32s.clone();
    ys.extend(i32s.iter().rev());
    assert_eq!(fixed_hash_ids(&ys), group_id_reference(&ys));
    assert_eq!(group_id(&ys), group_id_reference(&ys));
}

/// `mix` is a bijection, so single keys never share a full hash. Pairs can: under a known
/// multiplier, for any `a` and `a2`, the pair `(a2, b ^ mix(a) ^ mix(a2))` has the full hash
/// of `(a, b)`. The multiplier of a call is random, so chosen pairs do not collide there, except for
/// pairs of two, which collide under every multiplier.
#[test]
fn pairs_with_equal_full_hashes() {
    let b = 0x1234_5678_9ABC_DEF0u64;
    let xs: Vec<u64> = (0..1000).collect();
    let ys: Vec<u64> = xs.iter().map(|&a| b ^ mix(0, MUL) ^ mix(a, MUL)).collect();
    let h = (0u64, b).key_hash(MUL);
    assert!(pairs(&xs, &ys).iter().all(|p| p.key_hash(MUL) == h));
    let mut xs2 = xs.clone();
    xs2.extend(xs.iter().rev());
    let mut ys2 = ys.clone();
    ys2.extend(ys.iter().rev());
    let want: Vec<i64> = (0..1000).chain((0..1000).rev()).collect();
    assert_eq!(fixed_hash_ids(&pairs(&xs2, &ys2)), want);
    assert_eq!(group_id2(&xs2, &ys2).unwrap(), want);
}

#[test]
fn table_allocation_failure_is_an_error() {
    let err = table::<u64, u32>(usize::MAX / 4).unwrap_err();
    assert!(err.starts_with("cannot allocate a group_id table"), "{err}");
}

#[test]
fn ids_count_the_groups() {
    // The number of groups is the largest id plus 1.
    let xs: Vec<i32> = (0..3000).map(|i| (i * 7919) % 1013).collect();
    let ids = group_id(&xs);
    assert_eq!(ids.iter().max(), Some(&1012));
}
