//! The NEON intrinsics backend of the float extreme reductions: `reduce_min`, `reduce_max`,
//! `argmin`, and `argmax` on `f64` and `f32`.
//!
//! Each reducer parses its argument with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails, and reads the rows only through the checked slice. The vector
//! loops use `FMIN` and `FMAX` (`vminq`, `vmaxq`, `vminvq`, `vmaxvq`), which are IEEE
//! 754-2019 `minimum` and `maximum`: NaN propagates and `-0.0 < +0.0`. Those are the scalar
//! `min2` and `max2`, so on non-NaN rows the vector result has the scalar bits. The rows past
//! the last whole vector of a tile take the scalar rules directly.
//!
//! `argmin` and `argmax` keep a value vector and an index vector per lane. A lane takes the
//! new row's index when `FMIN` (or `FMAX`) of the lane's value and the row differs in bits
//! from the lane's value, which is exactly when the row is strictly smaller (larger) in the
//! order with `-0.0 < +0.0`, so each lane keeps the first index of its extreme. A NaN row
//! turns its lane's value into NaN for good, so a NaN lane after the loop means that the rows
//! hold a NaN, and a scan of the rows finds the first one.

use std::arch::aarch64::*;

use crate::kernels::scalar::elementwise::{arity_error, rows_arg, scalar_from};
use crate::kernels::scalar::reduce::{Accum, Extreme, Ordered};
use crate::kernels::{Reducer, ReducerFn};
use crate::value::{Arg, Elem, Scalar, ScalarType};

/// The NEON entry of `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true) on rows of type
/// `T`, where it exists.
pub(crate) fn extreme_entry<T: Elem, const MAX: bool>() -> Option<ReducerFn> {
    // Each storage type has its own `ScalarType`, so an arm instantiates the entry for
    // exactly `T`.
    match T::TYPE {
        ScalarType::F64 => Some(extreme::<f64, MAX>),
        ScalarType::F32 => Some(extreme::<f32, MAX>),
        _ => None,
    }
}

/// The NEON entry of `argmin` (`MAX` false) or `argmax` (`MAX` true) on rows of type `T`,
/// where it exists.
pub(crate) fn arg_extreme_entry<T: Elem, const MAX: bool>() -> Option<ReducerFn> {
    // As in `extreme_entry`, an arm instantiates the entry for exactly `T`.
    match T::TYPE {
        ScalarType::F64 => Some(arg_extreme::<f64, MAX>),
        ScalarType::F32 => Some(arg_extreme::<f32, MAX>),
        _ => None,
    }
}

/// The most lanes of any `FloatLanes` type, the length of the arrays that `FloatLanes::lanes`
/// fills.
const MAX_LANES: usize = 4;

/// Rows per chunk of an `argmin` or `argmax` update. Lane indices count from the chunk's first
/// row, so they stay below `CHUNK`, which fits the 32-bit index lanes of `f32`. Each chunk's
/// winner gets the chunk's offset as a `usize`, so updates of any length keep exact indices.
const CHUNK: usize = 1 << 30;

/// A float row type that NEON reduces `LANES` rows, one 128-bit vector, at a time.
///
/// The methods are `unsafe` because they need NEON, and `load` reads through a raw pointer.
pub(super) trait FloatLanes: Ordered {
    const LANES: usize;

    /// A vector of `LANES` rows.
    type V: Copy;

    /// A vector of `LANES` row indices.
    type I: Copy;

    /// The `LANES` rows at `src`, which must be valid for reading them.
    unsafe fn load(src: *const Self) -> Self::V;

    /// `minimum` (`MAX` false) or `maximum` (`MAX` true) per lane.
    unsafe fn extreme<const MAX: bool>(a: Self::V, b: Self::V) -> Self::V;

    /// `minimum` (`MAX` false) or `maximum` (`MAX` true) across the lanes.
    unsafe fn across<const MAX: bool>(v: Self::V) -> Self;

    /// The indices `0..LANES`.
    unsafe fn first_indices() -> Self::I;

    /// `i` advanced by `LANES`.
    unsafe fn next_indices(i: Self::I) -> Self::I;

    /// The lanes of `old_i` where `new` has the bits of `old`, and of `new_i` elsewhere.
    unsafe fn keep_unchanged(old: Self::V, new: Self::V, old_i: Self::I, new_i: Self::I)
    -> Self::I;

    /// The values and indices of the lanes, in the first `LANES` entries.
    unsafe fn lanes(v: Self::V, i: Self::I) -> ([Self; MAX_LANES], [usize; MAX_LANES]);
}

// SAFETY (all `FloatLanes` methods below): each method is compiled for, and needs, NEON,
// which its callers in this file have because they are compiled for the same feature. Only
// `load` reads memory, through the pointer range that the trait states, and `lanes` writes
// only to local arrays of exactly one vector.
macro_rules! float_lanes {
    (
        $t:ty, $lanes:literal, $v:ty, $i:ty, $u:ty, $ld:ident, $min:ident, $max:ident,
        $minv:ident, $maxv:ident, $as_bits:ident, $ceq:ident, $bsl:ident, $dup:ident,
        $add:ident, $st:ident, $ld_i:ident, $st_i:ident, [$($lane:literal),*]
    ) => {
        impl FloatLanes for $t {
            const LANES: usize = $lanes;
            type V = $v;
            type I = $i;

            // SAFETY: the caller guarantees NEON and `LANES` readable rows at `src`.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn load(src: *const $t) -> $v {
                // SAFETY: the caller guarantees `LANES` readable rows (16 bytes) at `src`,
                // and `ld1` needs only element alignment, which a `*const $t` from a slice
                // has.
                unsafe { $ld(src) }
            }

            // SAFETY: the caller guarantees NEON. The method reads no memory.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn extreme<const MAX: bool>(a: $v, b: $v) -> $v {
                if MAX { $max(a, b) } else { $min(a, b) }
            }

            // SAFETY: the caller guarantees NEON. The method reads no memory.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn across<const MAX: bool>(v: $v) -> $t {
                if MAX { $maxv(v) } else { $minv(v) }
            }

            // SAFETY: the caller guarantees NEON. The method reads no memory.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn first_indices() -> $i {
                let lanes: [$u; $lanes] = [$($lane),*];
                // SAFETY: `lanes` is a local array of exactly one vector.
                unsafe { $ld_i(lanes.as_ptr()) }
            }

            // SAFETY: the caller guarantees NEON. The method reads no memory.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn next_indices(i: $i) -> $i {
                $add(i, $dup($lanes))
            }

            // SAFETY: the caller guarantees NEON. The method reads no memory.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn keep_unchanged(old: $v, new: $v, old_i: $i, new_i: $i) -> $i {
                // The compare is on the bits, so `-0.0` and `+0.0` differ.
                $bsl($ceq($as_bits(old), $as_bits(new)), old_i, new_i)
            }

            // SAFETY: the caller guarantees NEON. The method writes only local arrays.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn lanes(v: $v, i: $i) -> ([$t; MAX_LANES], [usize; MAX_LANES]) {
                let mut values = [<$t>::default(); MAX_LANES];
                let mut indices: [$u; MAX_LANES] = [0; MAX_LANES];
                // SAFETY: both arrays hold `MAX_LANES >= LANES` elements, room for one
                // vector, and `st1` needs only element alignment.
                unsafe {
                    $st(values.as_mut_ptr(), v);
                    $st_i(indices.as_mut_ptr(), i);
                }
                let index = |x: $u| usize::try_from(x).expect("a lane index fits in usize");
                (values, indices.map(index))
            }
        }
    };
}

float_lanes!(
    f64,
    2,
    float64x2_t,
    uint64x2_t,
    u64,
    vld1q_f64,
    vminq_f64,
    vmaxq_f64,
    vminvq_f64,
    vmaxvq_f64,
    vreinterpretq_u64_f64,
    vceqq_u64,
    vbslq_u64,
    vdupq_n_u64,
    vaddq_u64,
    vst1q_f64,
    vld1q_u64,
    vst1q_u64,
    [0, 1]
);
float_lanes!(
    f32,
    4,
    float32x4_t,
    uint32x4_t,
    u32,
    vld1q_f32,
    vminq_f32,
    vmaxq_f32,
    vminvq_f32,
    vmaxvq_f32,
    vreinterpretq_u32_f32,
    vceqq_u32,
    vbslq_u32,
    vdupq_n_u32,
    vaddq_u32,
    vst1q_f32,
    vld1q_u32,
    vst1q_u32,
    [0, 1, 2, 3]
);

/// The rows of the only argument, checked like the scalar entry checks them.
fn rows<T: Elem>(args: &[Arg]) -> Result<&[T], String> {
    let [a] = args else {
        return Err(arity_error(1, args.len()));
    };
    rows_arg::<T>(a, T::TYPE)
}

/// Whether `a` comes strictly before `b` in the order of `argmin` (`MAX` false) or `argmax`
/// (`MAX` true), with `-0.0 < +0.0`. Both must not be NaN.
#[inline]
fn better<T: Ordered, const MAX: bool>(a: T, b: T) -> bool {
    if MAX { a.greater(b) } else { a.less(b) }
}

/// The NEON reducer of `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true). It folds each
/// tile to one value and hands that and the rest of the tile to the scalar accumulator, whose
/// result does not depend on the order of the rows.
struct NeonExtreme<T, const MAX: bool> {
    acc: Extreme<T, MAX>,
}

impl<T: FloatLanes, const MAX: bool> NeonExtreme<T, MAX> {
    /// Consumes the rows of one tile.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    unsafe fn feed(&mut self, xs: &[T]) -> Result<(), String> {
        let lanes = T::LANES;
        // Four accumulators hide the latency of `FMIN`, which the result does not see
        // because `minimum` and `maximum` are associative and commutative.
        let mut quads = xs.chunks_exact(4 * lanes);
        let mut singles = quads.remainder().chunks_exact(lanes);
        let rest = singles.remainder();
        // SAFETY: the caller guarantees NEON. Every load reads one vector at the start or at
        // a multiple of `LANES` rows into a chunk from `chunks_exact`, whose length is a
        // multiple of `LANES` covering that vector.
        let folded = unsafe {
            let mut acc: Option<T::V> = None;
            if let Some(first) = quads.next() {
                let p0 = first.as_ptr();
                let mut a = [0, 1, 2, 3].map(|k| T::load(p0.add(k * lanes)));
                for q in quads.by_ref() {
                    let p = q.as_ptr();
                    for (k, ak) in a.iter_mut().enumerate() {
                        *ak = T::extreme::<MAX>(*ak, T::load(p.add(k * lanes)));
                    }
                }
                let ab = T::extreme::<MAX>(a[0], a[1]);
                acc = Some(T::extreme::<MAX>(ab, T::extreme::<MAX>(a[2], a[3])));
            }
            for s in singles.by_ref() {
                let v = T::load(s.as_ptr());
                acc = Some(acc.map_or(v, |a| T::extreme::<MAX>(a, v)));
            }
            acc.map(|a| T::across::<MAX>(a))
        };
        if let Some(x) = folded {
            self.acc.feed(1, |_| x)?;
        }
        self.acc.feed_slice(rest)
    }
}

impl<T: FloatLanes, const MAX: bool> Reducer for NeonExtreme<T, MAX> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let xs = rows::<T>(args)?;
        // SAFETY: a `NeonExtreme` is only obtained through the unsafe `extreme`, whose caller
        // guarantees NEON (`Dispatch` calls it only after `resolve` checked `is_available`).
        // `feed` reads only the rows of the checked slice `xs`.
        unsafe { self.feed(xs) }
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        Ok(scalar_from(self.acc.finish()?, T::TYPE))
    }
}

/// The reducer constructor of `reduce_min` or `reduce_max` on rows of type `T`.
// SAFETY: the caller guarantees that the CPU supports NEON, and the returned reducer relies
// on that for every `update`.
unsafe fn extreme<T: FloatLanes, const MAX: bool>() -> Box<dyn Reducer> {
    Box::new(NeonExtreme::<T, MAX> {
        acc: Extreme::new(),
    })
}

/// The NEON reducer of `argmin` (`MAX` false) or `argmax` (`MAX` true), with the state of the
/// scalar `ArgExtreme`: the rows consumed by earlier updates, the first extreme row with its
/// index, and the index of the first NaN.
struct NeonArgExtreme<T, const MAX: bool> {
    consumed: usize,
    best: Option<(usize, T)>,
    seen_nan: Option<usize>,
}

impl<T: FloatLanes, const MAX: bool> NeonArgExtreme<T, MAX> {
    /// Takes row `index` with value `x` when it is strictly better than the best so far.
    /// Callers offer rows in increasing index order, so ties keep the first index.
    #[inline]
    fn offer(&mut self, index: usize, x: T) {
        if self.best.is_none_or(|(_, b)| better::<T, MAX>(x, b)) {
            self.best = Some((index, x));
        }
    }

    /// Consumes the rows of one tile.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    unsafe fn feed(&mut self, xs: &[T]) {
        let base = self.consumed;
        self.consumed += xs.len();
        for (c, chunk) in xs.chunks(CHUNK).enumerate() {
            // The first NaN decides the result, so later rows cannot change it.
            if self.seen_nan.is_some() {
                return;
            }
            // SAFETY: the caller guarantees NEON.
            unsafe { self.feed_chunk(base + c * CHUNK, chunk) };
        }
    }

    /// Consumes at most `CHUNK` rows whose first row has index `base`.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    unsafe fn feed_chunk(&mut self, base: usize, xs: &[T]) {
        debug_assert!(xs.len() <= CHUNK);
        let lanes = T::LANES;
        let mut vectors = xs.chunks_exact(lanes);
        let rest_start = xs.len() - vectors.remainder().len();
        if let Some(first) = vectors.next() {
            // SAFETY: the caller guarantees NEON. Each load reads one chunk of exactly
            // `LANES` rows from `chunks_exact`.
            let (values, indices) = unsafe {
                let mut best = T::load(first.as_ptr());
                let mut best_i = T::first_indices();
                let mut cur_i = best_i;
                for v in vectors {
                    // `cur_i` stays below `rest_start <= CHUNK`, so it fits the index lanes.
                    cur_i = T::next_indices(cur_i);
                    let next = T::extreme::<MAX>(best, T::load(v.as_ptr()));
                    best_i = T::keep_unchanged(best, next, best_i, cur_i);
                    best = next;
                }
                T::lanes(best, best_i)
            };
            let (values, indices) = (&values[..lanes], &indices[..lanes]);
            if values.iter().any(|x| x.is_nan_()) {
                // A NaN lane holds a NaN row, see the module doc.
                let first_nan = xs.iter().position(|x| x.is_nan_());
                self.seen_nan = Some(base + first_nan.expect("a NaN lane comes from a NaN row"));
                return;
            }
            // The lane with the extreme value, the smallest index among equal values. The
            // order of `better` puts `-0.0` before `+0.0`, so the zeros are not equal here.
            let mut win = 0;
            for j in 1..lanes {
                let (x, w) = (values[j], values[win]);
                if better::<T, MAX>(x, w) || (!better::<T, MAX>(w, x) && indices[j] < indices[win])
                {
                    win = j;
                }
            }
            // The rows before have smaller indices and the rest of the chunk larger ones.
            self.offer(base + indices[win], values[win]);
        }
        for (j, &x) in xs[rest_start..].iter().enumerate() {
            if x.is_nan_() {
                self.seen_nan = Some(base + rest_start + j);
                return;
            }
            self.offer(base + rest_start + j, x);
        }
    }
}

impl<T: FloatLanes, const MAX: bool> Reducer for NeonArgExtreme<T, MAX> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let xs = rows::<T>(args)?;
        // SAFETY: a `NeonArgExtreme` is only obtained through the unsafe `arg_extreme`, whose
        // caller guarantees NEON (`Dispatch` calls it only after `resolve` checked
        // `is_available`). `feed` reads only the rows of the checked slice `xs`.
        unsafe { self.feed(xs) };
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        // The result and errors of the scalar `ArgExtreme::finish`.
        let what = if MAX { "argmax" } else { "argmin" };
        let index = self
            .seen_nan
            .or(self.best.map(|(i, _)| i))
            .ok_or_else(|| format!("{what} of an empty array"))?;
        Ok(Scalar::I64(
            i64::try_from(index).map_err(|e| e.to_string())?,
        ))
    }
}

/// The reducer constructor of `argmin` or `argmax` on rows of type `T`.
// SAFETY: the caller guarantees that the CPU supports NEON, and the returned reducer relies
// on that for every `update`.
unsafe fn arg_extreme<T: FloatLanes, const MAX: bool>() -> Box<dyn Reducer> {
    Box::new(NeonArgExtreme::<T, MAX> {
        consumed: 0,
        best: None,
        seen_nan: None,
    })
}
