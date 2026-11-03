//! The stable argsort behind `argsort(a, b)`: an LSD radix sort with byte digits over the
//! composite image of two range-reduced key columns, and a comparison sort for short inputs.
//!
//! The layouts come from the probe in
//! `docs/superpowers/specs/2026-11-03-verkehrt-composite-keys-design.md`.

use crate::kernels::scalar::sort::radix::{
    Digits, Pos, RADIX, RADIX_BITS, RadixKey, scatter, scratch,
};

/// Inputs of at most this many rows sort by comparison.
// The 64-bit threshold of `argsort`, which is not measured for pairs.
const SMALL: usize = 1024;
/// The most digits of a composite image, those of 128 bits.
const MAX_PASSES: usize = 16;

/// The range reduction of two key columns `a` and `b`. The composite image of a row is its
/// reduced `a` image above its reduced `b` image, in `width` bits, with the same order as
/// (`a`, `b`).
pub(super) struct Reduced {
    /// The minimum image of `a`.
    min_a: u64,
    /// The minimum image of `b`.
    min_b: u64,
    /// The bit length of the largest reduced `b` image.
    bits_b: u32,
    /// The bit length of the largest composite image, at most 128.
    pub(super) width: u32,
}

impl Reduced {
    /// The reduction of the equally long columns `a` and `b`. An empty pair has width 0.
    pub(super) fn of<A: RadixKey, B: RadixKey>(a: &[A], b: &[B]) -> Reduced {
        let (min_a, max_a) = extent(a);
        let (min_b, max_b) = extent(b);
        // The maximum is at least the minimum, so the difference never wraps. It may be
        // `u64::MAX`, whose bit length is 64, so the length comes from the leading zeros and
        // never from adding 1 to the range.
        let bits = |range: u64| u64::BITS - range.leading_zeros();
        let bits_b = bits(max_b - min_b);
        Reduced {
            min_a,
            min_b,
            bits_b,
            width: bits(max_a - min_a) + bits_b,
        }
    }

    /// The composite image of a row whose key images are `x` and `y`, which lie within the
    /// columns' extents.
    #[inline(always)]
    fn key(&self, x: u64, y: u64) -> u128 {
        // The reduced `a` image has at most `width - bits_b` bits, so the shift drops none.
        (u128::from(x - self.min_a) << self.bits_b) | u128::from(y - self.min_b)
    }
}

/// The minimum and maximum image of `xs`, or (0, 0) if it is empty.
#[inline(always)]
fn extent<T: RadixKey>(xs: &[T]) -> (u64, u64) {
    if xs.is_empty() {
        return (0, 0);
    }
    xs.iter().fold((u64::MAX, 0), |(lo, hi), &x| {
        let k: u64 = x.image().into();
        (lo.min(k), hi.max(k))
    })
}

/// The byte of the composite image `k` at bit offset `shift`, a multiple of 8 below 128.
#[inline(always)]
fn digit(k: u128, shift: u32) -> usize {
    usize::from(k.wrapping_shr(shift).to_le_bytes()[0])
}

/// The low 64 bits of `k`.
#[inline(always)]
fn low64(k: u128) -> u64 {
    u64::try_from(k & u128::from(u64::MAX)).unwrap_or(u64::MAX)
}

/// A scratch record of the radix sort: a composite image and a row position.
pub(super) trait Record: Copy + Default {
    /// The widest composite image the record holds, in bits.
    const BITS: u32;
    /// The record of composite image `k` of at most `BITS` bits at position `i`, which the
    /// caller guarantees to fit the record's position type.
    fn new(k: u128, i: usize) -> Self;
    /// The byte of the composite image at bit offset `shift`, a multiple of 8 below `BITS`.
    fn digit(&self, shift: u32) -> usize;
    /// The position as an `i64` output row.
    fn pos(&self) -> i64;
}

/// The 16-byte record: the high 64 and low 32 bits of a composite image of at most 96 bits,
/// and a `u32` position.
#[derive(Clone, Copy, Default)]
pub(super) struct R16 {
    hi: u64,
    lo: u32,
    pos: u32,
}

impl Record for R16 {
    const BITS: u32 = 96;
    #[inline(always)]
    fn new(k: u128, i: usize) -> R16 {
        debug_assert!(k >> Self::BITS == 0, "composite image exceeds 96 bits");
        R16 {
            hi: low64(k >> 32),
            lo: u32::try_from(k & u128::from(u32::MAX)).unwrap_or(u32::MAX),
            pos: u32::from_index(i),
        }
    }
    #[inline(always)]
    fn digit(&self, shift: u32) -> usize {
        if shift < 32 {
            self.lo.digit(shift)
        } else {
            self.hi.digit(shift - 32)
        }
    }
    #[inline(always)]
    fn pos(&self) -> i64 {
        self.pos.to_i64()
    }
}

/// The 24-byte record: the high and low 64 bits of a composite image, and a position of type
/// `P`.
#[derive(Clone, Copy, Default)]
pub(super) struct R24<P> {
    hi: u64,
    lo: u64,
    pos: P,
}

impl<P: Pos> Record for R24<P> {
    const BITS: u32 = 128;
    #[inline(always)]
    fn new(k: u128, i: usize) -> R24<P> {
        R24 {
            hi: low64(k >> 64),
            lo: low64(k),
            pos: P::from_index(i),
        }
    }
    #[inline(always)]
    fn digit(&self, shift: u32) -> usize {
        if shift < 64 {
            self.lo.digit(shift)
        } else {
            self.hi.digit(shift - 64)
        }
    }
    #[inline(always)]
    fn pos(&self) -> i64 {
        self.pos.to_i64()
    }
}

/// Writes the positions of the rows of `a` and `b` to `out`, which has the length of `a`, in
/// ascending order of `a`, ties broken by ascending `b`, then by position. Fails with
/// `length mismatch: {len(a)} vs {len(b)}` if the columns differ in length, and fails if a
/// scratch buffer cannot be allocated.
#[inline(always)]
pub(crate) fn argsort2_into<A: RadixKey, B: RadixKey>(
    a: &[A],
    b: &[B],
    out: &mut [i64],
) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    let n = a.len();
    if n <= SMALL {
        comparison2_into(a, b, out);
        return Ok(());
    }
    let r = Reduced::of(a, b);
    // The 16-byte record needs both a narrow image and narrow positions. Either record gives
    // the same output, so the choice only sets the scratch size.
    let narrow = u32::try_from(n).is_ok();
    if narrow && r.width <= R16::BITS {
        radix2_into::<A, B, R16>(a, b, &r, out)
    } else if narrow {
        radix2_into::<A, B, R24<u32>>(a, b, &r, out)
    } else {
        radix2_into::<A, B, R24<u64>>(a, b, &r, out)
    }
}

/// The comparison sort for short inputs, where the radix sort's fixed cost dominates.
#[inline(always)]
fn comparison2_into<A: RadixKey, B: RadixKey>(a: &[A], b: &[B], out: &mut [i64]) {
    for (o, i) in out.iter_mut().zip(0..) {
        *o = i;
    }
    // Ties break on the position, so the unstable sort is stable, and it needs no buffer
    // beyond the output.
    out.sort_unstable_by_key(|&i| {
        let row = usize::try_from(i.cast_unsigned()).unwrap_or(0);
        (a[row], b[row], i)
    });
}

/// The LSD radix sort of the equally long `a` and `b` into `out`, with records of type `R`.
/// `r` is the reduction of `a` and `b`, whose width the record holds, and `R` holds every
/// position.
#[inline(always)]
pub(super) fn radix2_into<A: RadixKey, B: RadixKey, R: Record>(
    a: &[A],
    b: &[B],
    r: &Reduced,
    out: &mut [i64],
) -> Result<(), String> {
    debug_assert!(
        r.width <= R::BITS,
        "the record cannot hold the composite image"
    );
    let n = a.len();
    let keys = || {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| r.key(x.image().into(), y.image().into()))
    };
    let Some(first) = keys().next() else {
        return Ok(());
    };
    // Digits above the width are zero in every row, so only the low ones are counted.
    let digits = usize::try_from(r.width.div_ceil(RADIX_BITS)).unwrap_or(MAX_PASSES);
    // One counting pass for all digits.
    let mut counts = [[0usize; RADIX]; MAX_PASSES];
    for k in keys() {
        for (c, shift) in counts.iter_mut().zip((0..).step_by(8)).take(digits) {
            c[digit(k, shift)] += 1;
        }
    }
    // A digit that is equal in every row leaves the order unchanged, so its pass is skipped.
    let mut active = [0u32; MAX_PASSES];
    let mut passes = 0;
    for (c, shift) in counts.iter().zip((0..).step_by(8)).take(digits) {
        if c[digit(first, shift)] != n {
            active[passes] = shift;
            passes += 1;
        }
    }
    let count = |shift: u32| &counts[usize::try_from(shift / RADIX_BITS).unwrap_or(0)];
    match active[..passes] {
        [] => {
            for (o, i) in out.iter_mut().zip(0..) {
                *o = i;
            }
        }
        [shift] => scatter(
            keys().zip(0i64..),
            count(shift),
            out,
            |&(k, _)| digit(k, shift),
            |(_, i)| i,
        ),
        [head, ref middle @ .., last] => {
            // Two buffers alternate between the first pass, which reads the columns, and the
            // last, which writes `out`.
            let mut x: Vec<R> = scratch(n, true)?;
            let mut y: Vec<R> = scratch(n, !middle.is_empty())?;
            scatter(
                keys().zip(0..),
                count(head),
                &mut x,
                |&(k, _)| digit(k, head),
                |(k, i)| R::new(k, i),
            );
            for &shift in middle {
                scatter(
                    x.iter().copied(),
                    count(shift),
                    &mut y,
                    |p| p.digit(shift),
                    |p| p,
                );
                std::mem::swap(&mut x, &mut y);
            }
            scatter(
                x.iter().copied(),
                count(last),
                out,
                |p| p.digit(last),
                |p| p.pos(),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
