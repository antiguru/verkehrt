//! `group_id`: the dense id of each row's key, numbered by first occurrence, from an
//! open-addressing hash table. `u8` keys index a direct table instead.
//!
//! The ids depend only on the order in which distinct keys first appear, never on the hash
//! function or the table layout, so every variant returns the same output.

use std::marker::PhantomData;

use crate::kernels::scalar::construct::width;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::index::radix::Pos;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeBody};
use crate::ops::Op;
use crate::value::{Array, Elem, ScalarType, Value, ValueType};

/// Slots of the first table. The table doubles whenever more than half its slots are taken.
pub(super) const INITIAL_SLOTS: usize = 256;

/// The odd multiplier of `mix`, `2^64` divided by the golden ratio.
const MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// A deterministic, well-mixed hash of `x`. It is a bijection on `u64`, and its high bits,
/// which pick the slot, depend on every bit of `x`.
#[inline(always)]
pub(super) fn mix(x: u64) -> u64 {
    // The xorshift folds the high half into the low half, which the multiply then carries
    // into the high bits.
    (x ^ (x >> 32)).wrapping_mul(MUL)
}

/// A key that `group_id` hashes: one column's value or a pair of them.
pub(super) trait GroupKey: Copy + Eq + Default {
    /// The hash of the key. Equal keys hash equally.
    fn key_hash(self) -> u64;
}

/// A key column type of `group_id`.
pub(super) trait Column: Elem + Eq + Default {
    /// The bits of the value, zero-extended. A signed value hashes on its bit pattern, which
    /// keeps equality, the only property the ids need.
    fn bits(self) -> u64;

    /// Writes the id of each row of `xs` to `out`, which has the length of `xs`. Fails if
    /// the table cannot be allocated.
    #[inline(always)]
    fn ids_into(xs: &[Self], out: &mut [i64]) -> Result<(), String> {
        hashed_into(xs.iter().copied(), xs.len(), Self::key_hash, out)
    }
}

impl<T: Column> GroupKey for T {
    #[inline(always)]
    fn key_hash(self) -> u64 {
        mix(self.bits())
    }
}

impl<A: Column, B: Column> GroupKey for (A, B) {
    #[inline(always)]
    fn key_hash(self) -> u64 {
        mix(mix(self.0.bits()) ^ self.1.bits())
    }
}

impl Column for i64 {
    #[inline(always)]
    fn bits(self) -> u64 {
        self.cast_unsigned()
    }
}

impl Column for u64 {
    #[inline(always)]
    fn bits(self) -> u64 {
        self
    }
}

impl Column for i32 {
    #[inline(always)]
    fn bits(self) -> u64 {
        u64::from(self.cast_unsigned())
    }
}

impl Column for u8 {
    #[inline(always)]
    fn bits(self) -> u64 {
        u64::from(self)
    }

    /// A direct table of the 256 values replaces the hash table.
    #[inline(always)]
    fn ids_into(xs: &[u8], out: &mut [i64]) -> Result<(), String> {
        // -1 marks a value that has not occurred yet.
        let mut ids = [-1i64; 256];
        let mut next = 0;
        for (o, &x) in out.iter_mut().zip(xs) {
            let id = &mut ids[usize::from(x)];
            if *id < 0 {
                *id = next;
                next += 1;
            }
            *o = *id;
        }
        Ok(())
    }
}

/// A zeroed table of `slots` empty slots. Fails if it cannot be allocated.
pub(super) fn table<K: Copy + Default, P: Copy + Default>(
    slots: usize,
) -> Result<Vec<(K, P)>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(slots)
        .map_err(|_| format!("cannot allocate a group_id table of {slots} slots"))?;
    v.resize(slots, <(K, P)>::default());
    Ok(v)
}

/// The slot of hash `h` in a table of `2^(64 - shift)` slots, its high bits.
#[inline(always)]
fn slot(h: u64, shift: u32) -> usize {
    usize::try_from(h >> shift).unwrap_or(0)
}

/// Writes the id of each of the `n` keys of `keys` to `out`, which has length `n`. `hash`
/// maps a key to its hash, which affects only the speed, never the ids. Fails if the table
/// cannot be allocated.
#[inline(always)]
pub(super) fn hashed_into<K: GroupKey>(
    keys: impl Iterator<Item = K>,
    n: usize,
    hash: impl Fn(K) -> u64,
    out: &mut [i64],
) -> Result<(), String> {
    // A tag is an id plus 1, at most `n`, so `u32` tags halve the slot of a narrow key
    // whenever they fit.
    if u32::try_from(n).is_ok() {
        probe_into::<K, u32>(keys, n, hash, out)
    } else {
        probe_into::<K, u64>(keys, n, hash, out)
    }
}

/// The body of `hashed_into` for tags of type `P`, which holds `n`.
///
/// Each slot holds a key and its tag, the id plus 1, and tag 0 marks an empty slot. Probing
/// is linear. The table grows before more than half its slots are taken, so every probe
/// meets an empty slot and terminates whatever the hash. Growth reinserts the keys with
/// their tags, so the ids stay in first-occurrence order.
#[inline(always)]
fn probe_into<K: GroupKey, P: Pos + Eq>(
    keys: impl Iterator<Item = K>,
    n: usize,
    hash: impl Fn(K) -> u64,
    out: &mut [i64],
) -> Result<(), String> {
    // The smallest power of two that keeps `n` keys at most half full, capped at the first
    // table, and at least 2 slots so that the shift stays below 64.
    let mut slots = n
        .saturating_mul(2)
        .checked_next_power_of_two()
        .unwrap_or(INITIAL_SLOTS)
        .clamp(2, INITIAL_SLOTS);
    let mut shift = 64 - slots.trailing_zeros();
    let mut tab = table::<K, P>(slots)?;
    let empty = P::default();
    let mut groups = 0usize;
    let mut next = 0i64;
    for (o, k) in out.iter_mut().zip(keys) {
        let mask = slots - 1;
        let mut i = slot(hash(k), shift);
        loop {
            let (key, tag) = tab[i];
            if tag == empty {
                tab[i] = (k, P::from_index(groups + 1));
                *o = next;
                groups += 1;
                next += 1;
                break;
            }
            if key == k {
                *o = tag.to_i64() - 1;
                break;
            }
            i = (i + 1) & mask;
        }
        if 2 * groups > slots {
            let grown = slots
                .checked_mul(2)
                .ok_or_else(|| format!("cannot allocate a group_id table for {n} rows"))?;
            let new_shift = shift - 1;
            let mut new = table::<K, P>(grown)?;
            // The keys are distinct, so reinsertion only looks for an empty slot.
            for &(key, tag) in tab.iter().filter(|(_, tag)| *tag != empty) {
                let mut j = slot(hash(key), new_shift);
                while new[j].1 != empty {
                    j = (j + 1) & (grown - 1);
                }
                new[j] = (key, tag);
            }
            (tab, slots, shift) = (new, grown, new_shift);
        }
    }
    Ok(())
}

/// The body of `group_id` on one key column of type `T`.
pub(crate) struct GroupId<T>(PhantomData<fn() -> T>);

impl<T: Column> WholeBody for GroupId<T> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs] = args else {
            return Err(arity_error(1, args.len()));
        };
        let xs = array_arg(xs, T::TYPE)?.as_slice::<T>();
        Array::build_bytes(xs.len(), ScalarType::I64, |out| {
            T::ids_into(xs, bytemuck::cast_slice_mut(out))
        })
        .map(Value::Array)
    }
}

/// The body of `group_id` on the pairs of key columns of types `A` and `B`.
pub(crate) struct GroupId2<A, B>(PhantomData<fn() -> (A, B)>);

impl<A: Column, B: Column> WholeBody for GroupId2<A, B> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs, ys] = args else {
            return Err(arity_error(2, args.len()));
        };
        let xs = array_arg(xs, A::TYPE)?.as_slice::<A>();
        let ys = array_arg(ys, B::TYPE)?.as_slice::<B>();
        if xs.len() != ys.len() {
            return Err(format!("length mismatch: {} vs {}", xs.len(), ys.len()));
        }
        Array::build_bytes(xs.len(), ScalarType::I64, |out| {
            hashed_into(
                xs.iter().copied().zip(ys.iter().copied()),
                xs.len(),
                <(A, B)>::key_hash,
                bytemuck::cast_slice_mut(out),
            )
        })
        .map(Value::Array)
    }
}

// The kernel closures must coerce to bare function pointers, so their type arguments are
// constant expressions, never locals.

macro_rules! group_id {
    ($reg:expr, $t:ty, $ty:expr) => {
        $reg.add(KernelDef {
            name: format!("group_id_{}", $ty.name()),
            class: KernelClass::Whole,
            op: Some(Op::GroupId),
            inputs: vec![InputSpec::Array($ty)],
            output: ValueType::Array(ScalarType::I64),
            // One row read and one id written, ignoring the table.
            cost: width($ty) + 8,
            commutative: false,
            pattern: None,
            impls: KernelImpl::whole::<GroupId<$t>>(),
        })
    };
}

macro_rules! group_id2 {
    ($reg:expr, $a:ty, $ta:expr, $b:ty, $tb:expr) => {
        $reg.add(KernelDef {
            name: format!("group_id2_{}_{}", $ta.name(), $tb.name()),
            class: KernelClass::Whole,
            op: Some(Op::GroupId),
            inputs: vec![InputSpec::Array($ta), InputSpec::Array($tb)],
            output: ValueType::Array(ScalarType::I64),
            // One row of each column read and one id written, ignoring the table.
            cost: width($ta) + width($tb) + 8,
            commutative: false,
            pattern: None,
            impls: KernelImpl::whole::<GroupId2<$a, $b>>(),
        })
    };
}

/// `group_id2` with first column type `$a` and every second column type.
macro_rules! group_id2_all {
    ($reg:expr, $a:ty, $ta:expr) => {
        group_id2!($reg, $a, $ta, i64, ScalarType::I64);
        group_id2!($reg, $a, $ta, u64, ScalarType::U64);
        group_id2!($reg, $a, $ta, i32, ScalarType::I32);
        group_id2!($reg, $a, $ta, u8, ScalarType::U8);
    };
}

pub(crate) fn register(reg: &mut Registry) {
    group_id!(reg, i64, ScalarType::I64);
    group_id!(reg, u64, ScalarType::U64);
    group_id!(reg, i32, ScalarType::I32);
    group_id!(reg, u8, ScalarType::U8);
    // Every pair of key types, so the checker accepts any combination.
    group_id2_all!(reg, i64, ScalarType::I64);
    group_id2_all!(reg, u64, ScalarType::U64);
    group_id2_all!(reg, i32, ScalarType::I32);
    group_id2_all!(reg, u8, ScalarType::U8);
}

#[cfg(test)]
mod tests;
