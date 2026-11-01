//! `group_id`: the dense id of each row's key, numbered by first occurrence, from an
//! open-addressing hash table. `u8` keys index a direct table instead.
//!
//! The ids depend only on the order in which distinct keys first appear, never on the hash
//! function or the table layout, so every variant returns the same output.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::marker::PhantomData;

use crate::kernels::scalar::construct::width;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::index::radix::Pos;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeBody};
use crate::ops::Op;
use crate::value::{Array, Elem, ScalarType, Value, ValueType};

/// Cap on the slots of the first table (smaller for fewer than 128 rows), which grows whenever an insert leaves it more than half full.
pub(super) const INITIAL_SLOTS: usize = 256;

/// Doublings of one growth when nearly every row so far has opened a group. A growth
/// happens when the table is just over half full, so the new table has under 16 slots per
/// group, whatever the rest of the column holds. Measured in
/// `docs/measurements/2026-11-01-group-id.md`.
pub(super) const JUMP_BITS: u32 = 3;

/// A fixed odd multiplier for `mix`, `2^64` divided by the golden ratio. `group_id` draws a
/// random multiplier per call instead, and the tests use this one to build colliding keys.
#[cfg(test)]
pub(super) const MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// A random odd multiplier for one `group_id` call. Chosen keys cannot be made to collide
/// without knowing it, except for links between two pairs, and the ids do not depend on it.
fn multiplier() -> u64 {
    RandomState::new().hash_one(0u64) | 1
}

/// A well-mixed hash of `x` under the odd multiplier `mul`. It is a bijection on `u64`, and
/// its high bits, which pick the slot, depend on every bit of `x`.
#[inline(always)]
pub(super) fn mix(x: u64, mul: u64) -> u64 {
    // The xorshift folds the high half into the low half, which the multiply then carries
    // into the high bits.
    (x ^ (x >> 32)).wrapping_mul(mul)
}

/// A key that `group_id` hashes: one column's value or a pair of them.
pub(super) trait GroupKey: Copy + Eq + Default {
    /// The hash of the key under the odd multiplier `mul`. Equal keys hash equally.
    fn key_hash(self, mul: u64) -> u64;
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
        let mul = multiplier();
        hashed_into(xs.iter().copied(), xs.len(), |k| k.key_hash(mul), out)
    }
}

impl<T: Column> GroupKey for T {
    #[inline(always)]
    fn key_hash(self, mul: u64) -> u64 {
        mix(self.bits(), mul)
    }
}

impl<A: Column, B: Column> GroupKey for (A, B) {
    #[inline(always)]
    fn key_hash(self, mul: u64) -> u64 {
        mix(mix(self.0.bits(), mul) ^ self.1.bits(), mul)
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

/// The table that replaces the full `tab` of `slots` slots, with its slot count and hash
/// shift. A table that is nearly all distinct keys (`distinct`) grows `2^JUMP_BITS`-fold
/// instead of doubling, which saves rehashing at every size. No growth goes past `max_slots`,
/// which is `nextpow2(2n)` for n rows and so under 4n. After its first growth a table also has
/// at most 16 slots per group.
/// Fails if the new table cannot be allocated.
#[cold]
#[inline(never)]
fn grow<K: GroupKey, P: Pos + Eq>(
    tab: &[(K, P)],
    slots: usize,
    shift: u32,
    distinct: bool,
    max_slots: usize,
    hash: &impl Fn(K) -> u64,
) -> Result<(Vec<(K, P)>, usize, u32), String> {
    let empty = P::default();
    let jump = if distinct { JUMP_BITS } else { 1 };
    let grown = slots
        .checked_mul(1 << jump)
        .ok_or("cannot allocate a group_id table")?
        .min(max_slots);
    let new_shift = shift - (grown / slots).trailing_zeros();
    let mut new = table::<K, P>(grown)?;
    // The keys are distinct, so reinsertion only looks for an empty slot.
    for &(key, tag) in tab.iter().filter(|(_, tag)| *tag != empty) {
        let mut j = slot(hash(key), new_shift);
        while new[j].1 != empty {
            j = (j + 1) & (grown - 1);
        }
        new[j] = (key, tag);
    }
    Ok((new, grown, new_shift))
}

/// The body of `hashed_into` for tags of type `P`, which holds `n`.
///
/// Each slot holds a key and its tag, the id plus 1, and tag 0 marks an empty slot. Probing
/// is linear. The table grows right after an insert leaves it more than half full, so it is
/// at most half full when each probe starts. Every probe therefore meets an empty slot and
/// terminates whatever the hash. Growth reinserts the keys with their tags, so the ids stay
/// in first-occurrence order.
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
    // The table that holds `n` keys at most half full, which a growth never exceeds.
    let max_slots = n
        .saturating_mul(2)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    let mut shift = 64 - slots.trailing_zeros();
    let mut tab = table::<K, P>(slots)?;
    let empty = P::default();
    let mut groups = 0usize;
    let mut next = 0i64;
    for (row, (o, k)) in out.iter_mut().zip(keys).enumerate() {
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
            // Nearly every row so far opened a group, so the column is likely close to
            // all distinct.
            let distinct = 8 * groups >= 7 * (row + 1);
            (tab, slots, shift) = grow(&tab, slots, shift, distinct, max_slots, &hash)
                .map_err(|e| format!("{e} for {n} rows"))?;
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
        let mul = multiplier();
        Array::build_bytes(xs.len(), ScalarType::I64, |out| {
            hashed_into(
                xs.iter().copied().zip(ys.iter().copied()),
                xs.len(),
                |k| k.key_hash(mul),
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
