//! The stable argsort behind `argsort`: an LSD radix sort on (key, position) pairs with byte
//! digits, and a comparison sort for short inputs.
//!
//! The choices come from the measurement in `docs/measurements/2026-11-01-argsort.md`.

use crate::value::Elem;

/// Digit width in bits, which `Digits::digit` fixes at a byte.
// Byte digits beat 11-bit digits at 10M rows on every key type, and were within noise of
// them at 1M rows.
const RADIX_BITS: u32 = 8;
/// Buckets per digit.
const RADIX: usize = 1 << RADIX_BITS;
/// The most digits of any key type, those of a 64-bit key.
const MAX_PASSES: usize = 8;

/// The unsigned image of a key, split into byte digits.
pub(crate) trait Digits: Copy + Default {
    /// Number of byte digits, at most `MAX_PASSES`.
    const PASSES: usize;
    /// The byte of `self` at bit offset `shift`, a multiple of 8 below the width.
    fn digit(self, shift: u32) -> usize;
}

macro_rules! digits {
    ($u:ty) => {
        impl Digits for $u {
            const PASSES: usize = size_of::<$u>();
            #[inline(always)]
            fn digit(self, shift: u32) -> usize {
                usize::from(self.wrapping_shr(shift).to_le_bytes()[0])
            }
        }
    };
}

digits!(u64);
digits!(u32);
digits!(u8);

/// A key type that `argsort` sorts.
pub(crate) trait RadixKey: Elem + Ord {
    /// The unsigned image.
    type U: Digits;
    /// Inputs of at most this many rows sort by comparison.
    const SMALL: usize;
    /// The unsigned image, with the same order as `self`.
    fn image(self) -> Self::U;
}

// Each `SMALL` is the crossover between the comparison sort and the radix sort, measured on
// random keys.

impl RadixKey for i64 {
    type U = u64;
    const SMALL: usize = 1024;
    #[inline(always)]
    fn image(self) -> u64 {
        // Flipping the sign bit maps signed order onto unsigned order.
        self.cast_unsigned() ^ (1 << 63)
    }
}

impl RadixKey for u64 {
    type U = u64;
    const SMALL: usize = 1024;
    #[inline(always)]
    fn image(self) -> u64 {
        self
    }
}

impl RadixKey for i32 {
    type U = u32;
    const SMALL: usize = 256;
    #[inline(always)]
    fn image(self) -> u32 {
        self.cast_unsigned() ^ (1 << 31)
    }
}

impl RadixKey for u8 {
    type U = u8;
    const SMALL: usize = 32;
    #[inline(always)]
    fn image(self) -> u8 {
        self
    }
}

/// A row position in the scratch pairs. `u32` halves the pair of a 32-bit key, so it is used
/// whenever every position fits.
pub(crate) trait Pos: Copy + Default {
    /// Position `i`, which the caller guarantees to fit.
    fn from_index(i: usize) -> Self;
    /// The position as an `i64` output row.
    fn to_i64(self) -> i64;
}

impl Pos for u32 {
    #[inline(always)]
    fn from_index(i: usize) -> u32 {
        debug_assert!(u32::try_from(i).is_ok(), "position {i} exceeds u32");
        u32::try_from(i).unwrap_or(u32::MAX)
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
}

impl Pos for u64 {
    #[inline(always)]
    fn from_index(i: usize) -> u64 {
        u64::try_from(i).expect("usize fits in u64")
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self.cast_signed()
    }
}

/// Writes the stable ascending positions of `xs` to `out`, which has the length of `xs`.
/// Fails if a scratch buffer cannot be allocated.
#[inline(always)]
pub(crate) fn argsort_into<T: RadixKey>(xs: &[T], out: &mut [i64]) -> Result<(), String> {
    if xs.len() <= T::SMALL {
        comparison_into(xs, out);
        Ok(())
    } else if u32::try_from(xs.len()).is_ok() {
        radix_into::<T, u32>(xs, out)
    } else {
        radix_into::<T, u64>(xs, out)
    }
}

/// The comparison sort for short inputs, where the radix sort's fixed cost dominates.
#[inline(always)]
fn comparison_into<T: RadixKey>(xs: &[T], out: &mut [i64]) {
    for (o, i) in out.iter_mut().zip(0..) {
        *o = i;
    }
    // Ties break on the position, so the unstable sort is stable, and it needs no buffer
    // beyond the output.
    out.sort_unstable_by_key(|&i| {
        let row = usize::try_from(i.cast_unsigned()).unwrap_or(0);
        (xs[row], i)
    });
}

/// A zeroed scratch buffer of `n` pairs, or an empty one if it is not `needed`.
pub(super) fn scratch<E: Copy + Default>(n: usize, needed: bool) -> Result<Vec<E>, String> {
    let mut v = Vec::new();
    if needed {
        v.try_reserve_exact(n)
            .map_err(|_| format!("cannot allocate argsort scratch for {n} rows"))?;
        v.resize(n, E::default());
    }
    Ok(v)
}

/// Moves each item of `src` to the next free slot of its digit's bucket in `dst`, where
/// `counts` holds the number of items per digit. Visiting the items in order keeps equal
/// digits in their order, which makes the sort stable.
#[inline(always)]
fn scatter<S, D>(
    src: impl Iterator<Item = S>,
    counts: &[usize; RADIX],
    dst: &mut [D],
    digit: impl Fn(&S) -> usize,
    value: impl Fn(S) -> D,
) {
    // The first free slot of each bucket, the exclusive prefix sums of `counts`.
    let mut off = [0; RADIX];
    let mut sum = 0;
    for (o, &c) in off.iter_mut().zip(counts) {
        *o = sum;
        sum += c;
    }
    for s in src {
        let slot = &mut off[digit(&s)];
        dst[*slot] = value(s);
        *slot += 1;
    }
}

/// The LSD radix sort of `xs` into `out`, with positions of type `P`, which holds every
/// position of `xs`.
#[inline(always)]
pub(super) fn radix_into<T: RadixKey, P: Pos>(xs: &[T], out: &mut [i64]) -> Result<(), String> {
    let n = xs.len();
    let Some(&first) = xs.first() else {
        return Ok(());
    };
    // One counting pass for all digits.
    let mut counts = [[0usize; RADIX]; MAX_PASSES];
    for &x in xs {
        let k = x.image();
        for (c, shift) in counts.iter_mut().zip((0..).step_by(8)).take(T::U::PASSES) {
            c[k.digit(shift)] += 1;
        }
    }
    // A digit that is equal in every row leaves the order unchanged, so its pass is skipped.
    let mut active = [0u32; MAX_PASSES];
    let mut passes = 0;
    for (c, shift) in counts.iter().zip((0..).step_by(8)).take(T::U::PASSES) {
        if c[first.image().digit(shift)] != n {
            active[passes] = shift;
            passes += 1;
        }
    }
    let count = |shift: u32| &counts[usize::try_from(shift / RADIX_BITS).unwrap_or(0)];
    let key = |&(k, _): &(T::U, P)| k;
    match active[..passes] {
        [] => {
            for (o, i) in out.iter_mut().zip(0..) {
                *o = i;
            }
        }
        [shift] => scatter(
            xs.iter().zip(0i64..),
            count(shift),
            out,
            |&(x, _)| x.image().digit(shift),
            |(_, i)| i,
        ),
        [head, ref middle @ .., last] => {
            // Two buffers alternate between the first pass, which reads `xs`, and the last,
            // which writes `out`.
            let mut a: Vec<(T::U, P)> = scratch(n, true)?;
            let mut b: Vec<(T::U, P)> = scratch(n, !middle.is_empty())?;
            scatter(
                xs.iter().zip(0..),
                count(head),
                &mut a,
                |&(x, _)| x.image().digit(head),
                |(x, i)| (x.image(), P::from_index(i)),
            );
            for &shift in middle {
                scatter(
                    a.iter().copied(),
                    count(shift),
                    &mut b,
                    |p| key(p).digit(shift),
                    |p| p,
                );
                std::mem::swap(&mut a, &mut b);
            }
            scatter(
                a.iter().copied(),
                count(last),
                out,
                |p| key(p).digit(last),
                |(_, i)| i.to_i64(),
            );
        }
    }
    Ok(())
}
