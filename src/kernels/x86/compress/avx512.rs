//! AVX-512 selection entries. The keep mask of a vector comes from a compare into a mask
//! register (`filter`) or from a byte test of the bool mask (`compress`, `where`), and
//! `vpcompress` packs the kept lanes to the front of the vector, which is then stored whole.
//! `u8` and bool use the VBMI2 byte compress.

use std::arch::x86_64::*;

use super::ones;
use crate::kernels::scalar::elementwise::tag;
use crate::kernels::scalar::fused::{CMPS, Cmp, filter_args};
use crate::kernels::scalar::select::{
    append_chunks, check_row_indices, compress_args, predicated, where_args,
};
use crate::value::{Arg, ArrayBuilder, Elem};

/// A row type that AVX-512 selects `LANES` rows at a time.
///
/// The methods are `unsafe` because they read and write through raw pointers and need the
/// CPU features of `Isa::Avx512`. Each states the memory its pointers must cover.
pub(super) trait Lanes: Elem {
    const LANES: usize;
    /// A vector of `LANES` rows.
    type Vector: Copy;

    /// The keep mask of the `LANES` bool bytes at `mask`, bit `j` set when byte `j` is
    /// nonzero. `mask` must be valid for reading `LANES` bytes.
    unsafe fn keep(mask: *const u8) -> u64;

    /// The `LANES` rows at `src`, which must be valid for reading them.
    unsafe fn load(src: *const Self) -> Self::Vector;

    /// Stores the lanes of `v` whose bit in `keep` is set, in lane order, to the front of the
    /// `LANES` rows at `dst`, and returns their count. Bits of `keep` at `LANES` and above are
    /// ignored. The other rows at `dst` get unspecified values. `dst` must be valid for
    /// writing `LANES` rows.
    unsafe fn compress(v: Self::Vector, keep: u64, dst: *mut Self) -> usize;
}

/// A row type whose comparisons AVX-512 evaluates `LANES` rows at a time.
pub(super) trait Compare: Lanes + PartialOrd {
    /// The mask of the lanes of `v` with `v[j] <cmp> c`, for comparison number `C` of `CMPS`,
    /// with the predicate of `Cmp::test` (IEEE 754 for floats, signed for integers).
    unsafe fn compare<const C: usize>(v: Self::Vector, c: Self) -> u64;
}

// SAFETY (all `Lanes` and `Compare` methods below): each method is compiled for, and needs,
// the AVX-512 features of `Isa::Avx512`, which its callers in this file have because they
// are compiled for the same features. Pointer requirements are the trait's, and the unsafe
// blocks rely on exactly those.
macro_rules! lanes_64 {
    ($t:ty) => {
        impl Lanes for $t {
            const LANES: usize = 8;
            type Vector = __m512i;

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::keep` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 8 readable bytes at `mask`, and `loadl` reads
                // 8 bytes with no alignment requirement.
                let v = unsafe { _mm_loadl_epi64(mask.cast()) };
                u64::from(_mm_test_epi8_mask(v, v))
            }

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::load` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn load(src: *const $t) -> __m512i {
                // SAFETY: the caller guarantees 8 readable rows (64 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm512_loadu_si512(src.cast()) }
            }

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::compress` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn compress(v: __m512i, keep: u64, dst: *mut $t) -> usize {
                let k = keep.to_le_bytes()[0];
                // The compress moves bits, so float rows keep their exact bits, NaN payloads
                // included.
                let packed = _mm512_maskz_compress_epi64(k, v);
                // SAFETY: the caller guarantees 8 writable rows (64 bytes) at `dst`, and
                // `storeu` has no alignment requirement.
                unsafe { _mm512_storeu_si512(dst.cast(), packed) };
                ones(u64::from(k))
            }
        }
    };
}

macro_rules! lanes_32 {
    ($t:ty) => {
        impl Lanes for $t {
            const LANES: usize = 16;
            type Vector = __m512i;

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::keep` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 16 readable bytes at `mask`, and `loadu` has
                // no alignment requirement.
                let v = unsafe { _mm_loadu_si128(mask.cast()) };
                u64::from(_mm_test_epi8_mask(v, v))
            }

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::load` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn load(src: *const $t) -> __m512i {
                // SAFETY: the caller guarantees 16 readable rows (64 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm512_loadu_si512(src.cast()) }
            }

            // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
            // ranges that `Lanes::compress` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
            unsafe fn compress(v: __m512i, keep: u64, dst: *mut $t) -> usize {
                let b = keep.to_le_bytes();
                let k = u16::from_le_bytes([b[0], b[1]]);
                // The compress moves bits, see `lanes_64`.
                let packed = _mm512_maskz_compress_epi32(k, v);
                // SAFETY: the caller guarantees 16 writable rows (64 bytes) at `dst`, and
                // `storeu` has no alignment requirement.
                unsafe { _mm512_storeu_si512(dst.cast(), packed) };
                ones(u64::from(k))
            }
        }
    };
}

lanes_64!(f64);
lanes_64!(i64);
lanes_32!(f32);
lanes_32!(i32);

impl Lanes for u8 {
    const LANES: usize = 64;
    type Vector = __m512i;

    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Lanes::keep` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn keep(mask: *const u8) -> u64 {
        // SAFETY: the caller guarantees 64 readable bytes at `mask`, and `loadu` has no
        // alignment requirement.
        let v = unsafe { _mm512_loadu_si512(mask.cast()) };
        _mm512_test_epi8_mask(v, v)
    }

    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Lanes::load` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn load(src: *const u8) -> __m512i {
        // SAFETY: the caller guarantees 64 readable bytes at `src`, and `loadu` has no
        // alignment requirement.
        unsafe { _mm512_loadu_si512(src.cast()) }
    }

    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Lanes::compress` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn compress(v: __m512i, keep: u64, dst: *mut u8) -> usize {
        // The VBMI2 byte compress. Bool rows are 0 or 1 in the input, and the compress only
        // moves them.
        let packed = _mm512_maskz_compress_epi8(keep, v);
        // SAFETY: the caller guarantees 64 writable bytes at `dst`, and `storeu` has no
        // alignment requirement.
        unsafe { _mm512_storeu_si512(dst.cast(), packed) };
        ones(keep)
    }
}

impl Compare for f64 {
    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn compare<const C: usize>(v: __m512i, c: f64) -> u64 {
        let (v, c) = (_mm512_castsi512_pd(v), _mm512_set1_pd(c));
        // Ordered predicates are false on NaN like `<`, and `!=` is the unordered NEQ, which
        // is true on NaN like Rust's `!=`.
        let m = match CMPS[C].1 {
            Cmp::Eq => _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(v, c),
            Cmp::Ne => _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(v, c),
            Cmp::Lt => _mm512_cmp_pd_mask::<_CMP_LT_OQ>(v, c),
            Cmp::Le => _mm512_cmp_pd_mask::<_CMP_LE_OQ>(v, c),
            Cmp::Gt => _mm512_cmp_pd_mask::<_CMP_GT_OQ>(v, c),
            Cmp::Ge => _mm512_cmp_pd_mask::<_CMP_GE_OQ>(v, c),
        };
        u64::from(m)
    }
}

impl Compare for f32 {
    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn compare<const C: usize>(v: __m512i, c: f32) -> u64 {
        let (v, c) = (_mm512_castsi512_ps(v), _mm512_set1_ps(c));
        // The predicates of the f64 case.
        let m = match CMPS[C].1 {
            Cmp::Eq => _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(v, c),
            Cmp::Ne => _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(v, c),
            Cmp::Lt => _mm512_cmp_ps_mask::<_CMP_LT_OQ>(v, c),
            Cmp::Le => _mm512_cmp_ps_mask::<_CMP_LE_OQ>(v, c),
            Cmp::Gt => _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, c),
            Cmp::Ge => _mm512_cmp_ps_mask::<_CMP_GE_OQ>(v, c),
        };
        u64::from(m)
    }
}

impl Compare for i64 {
    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn compare<const C: usize>(v: __m512i, c: i64) -> u64 {
        let c = _mm512_set1_epi64(c);
        // Signed compares. `NLE` is `>` and `NLT` is `>=`.
        let m = match CMPS[C].1 {
            Cmp::Eq => _mm512_cmp_epi64_mask::<_MM_CMPINT_EQ>(v, c),
            Cmp::Ne => _mm512_cmp_epi64_mask::<_MM_CMPINT_NE>(v, c),
            Cmp::Lt => _mm512_cmp_epi64_mask::<_MM_CMPINT_LT>(v, c),
            Cmp::Le => _mm512_cmp_epi64_mask::<_MM_CMPINT_LE>(v, c),
            Cmp::Gt => _mm512_cmp_epi64_mask::<_MM_CMPINT_NLE>(v, c),
            Cmp::Ge => _mm512_cmp_epi64_mask::<_MM_CMPINT_NLT>(v, c),
        };
        u64::from(m)
    }
}

impl Compare for i32 {
    // SAFETY: the caller guarantees the features of `Isa::Avx512` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
    unsafe fn compare<const C: usize>(v: __m512i, c: i32) -> u64 {
        let c = _mm512_set1_epi32(c);
        // The signed predicates of the i64 case.
        let m = match CMPS[C].1 {
            Cmp::Eq => _mm512_cmp_epi32_mask::<_MM_CMPINT_EQ>(v, c),
            Cmp::Ne => _mm512_cmp_epi32_mask::<_MM_CMPINT_NE>(v, c),
            Cmp::Lt => _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(v, c),
            Cmp::Le => _mm512_cmp_epi32_mask::<_MM_CMPINT_LE>(v, c),
            Cmp::Gt => _mm512_cmp_epi32_mask::<_MM_CMPINT_NLE>(v, c),
            Cmp::Ge => _mm512_cmp_epi32_mask::<_MM_CMPINT_NLT>(v, c),
        };
        u64::from(m)
    }
}

/// Writes the rows of `x` whose `mask` byte is nonzero to the front of `dst` and returns
/// their count. Panics unless `mask` has the rows of `x` and `dst` has room for them.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
fn compress_chunk<T: Lanes>(x: &[T], mask: &[u8], dst: &mut [T]) -> usize {
    // The pointer arithmetic below relies on these lengths, which the entry's caller does
    // not guarantee, so they are checked here.
    assert!(
        mask.len() == x.len() && dst.len() >= x.len(),
        "compress chunk of {} rows with {} mask rows into {} rows",
        x.len(),
        mask.len(),
        dst.len()
    );
    let n = x.len();
    let (mut i, mut k) = (0, 0);
    while i + T::LANES <= n {
        // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` of `x` and `mask` are in bounds.
        // Each step adds `LANES` to `i` and at most `LANES` to `k`, so `k <= i`, and rows
        // `k..k + LANES` of `dst` are in bounds because `dst.len() >= n`. This function has
        // the features of `Lanes`.
        unsafe {
            let keep = T::keep(mask.as_ptr().add(i));
            let v = T::load(x.as_ptr().add(i));
            k += T::compress(v, keep, dst.as_mut_ptr().add(k));
        }
        i += T::LANES;
    }
    let rest = x[i..].iter().zip(&mask[i..]).map(|(&v, &m)| (v, m != 0));
    k + predicated(rest, &mut dst[k..])
}

/// Writes the rows `v` of `x` with `v <cmp> c` to the front of `dst`, for comparison number
/// `C` of `CMPS`, and returns their count. Panics unless `dst` has room for the rows of `x`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
fn filter_chunk<T: Compare, const C: usize>(x: &[T], c: T, dst: &mut [T]) -> usize {
    // As in `compress_chunk`, the length the pointer arithmetic relies on is checked here.
    assert!(
        dst.len() >= x.len(),
        "filter chunk of {} rows into {} rows",
        x.len(),
        dst.len()
    );
    let n = x.len();
    let (mut i, mut k) = (0, 0);
    while i + T::LANES <= n {
        // SAFETY: as in `compress_chunk`, rows `i..i + LANES` of `x` and rows `k..k + LANES`
        // of `dst` are in bounds because `i + LANES <= n`, `k <= i`, and `dst.len() >= n`.
        // This function has the features of `Compare`.
        unsafe {
            let v = T::load(x.as_ptr().add(i));
            let keep = T::compare::<C>(v, c);
            k += T::compress(v, keep, dst.as_mut_ptr().add(k));
        }
        i += T::LANES;
    }
    let cmp = CMPS[C].1;
    let rest = x[i..].iter().map(|&v| (v, cmp.test(v, c)));
    k + predicated(rest, &mut dst[k..])
}

/// Writes `first + i` for every row `i` of `mask` whose byte is nonzero to the front of
/// `dst` and returns their count. Panics unless `dst` has room for a row per mask byte. The
/// caller guarantees that `first + mask.len()` fits in i64.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
fn where_chunk(mask: &[u8], first: i64, dst: &mut [i64]) -> usize {
    // As in `compress_chunk`, the length the pointer arithmetic relies on is checked here.
    assert!(
        dst.len() >= mask.len(),
        "where chunk of {} rows into {} rows",
        mask.len(),
        dst.len()
    );
    let n = mask.len();
    let lanes = <i64 as Lanes>::LANES;
    let step = _mm512_set1_epi64(8);
    // Lane `j` holds the index of row `i + j`. The adds cannot overflow because the indices
    // of the chunk fit in i64.
    let mut idx = _mm512_add_epi64(
        _mm512_set1_epi64(first),
        _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
    );
    let (mut i, mut k) = (0, 0);
    while i + lanes <= n {
        // SAFETY: as in `compress_chunk`, bytes `i..i + 8` of `mask` and rows `k..k + 8` of
        // `dst` are in bounds because `i + 8 <= n`, `k <= i`, and `dst.len() >= n`. This
        // function has the features of `Lanes`.
        unsafe {
            let keep = i64::keep(mask.as_ptr().add(i));
            k += i64::compress(idx, keep, dst.as_mut_ptr().add(k));
        }
        idx = _mm512_add_epi64(idx, step);
        i += lanes;
    }
    let at = first + i64::try_from(i).expect("row index fits in i64");
    let rest = (at..).zip(&mask[i..]).map(|(r, &m)| (r, m != 0));
    k + predicated(rest, &mut dst[k..])
}

/// The AVX-512 `compress` entry on rows of storage type `T` and type `tag::<T, BOOL>()`.
// SAFETY: the caller guarantees that the CPU supports the features above, which `Dispatch`
// does by calling only entries that `resolve` installed after `is_available` returned true.
// Memory safety does not depend on the caller: `compress_args` checks that the rows and the
// mask have the same length, and `compress_chunk` checks the lengths it relies on again.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
pub(super) unsafe fn compress<T: Lanes, const BOOL: bool>(
    args: &[Arg],
    _base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let (x, mask) = compress_args::<T>(args, tag::<T, BOOL>())?;
    append_chunks(out, x.len(), |start, dst| {
        let end = start + dst.len();
        compress_chunk(&x[start..end], &mask[start..end], dst)
    });
    Ok(())
}

/// The AVX-512 `filter_<cmp>` entry on rows of type `T`, for comparison number `C` of
/// `CMPS`.
// SAFETY: as for `compress`, the caller guarantees the CPU features, and `filter_chunk`
// checks the lengths it relies on.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
pub(super) unsafe fn filter<T: Compare, const C: usize>(
    args: &[Arg],
    _base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let (x, c) = filter_args::<T>(args)?;
    append_chunks(out, x.len(), |start, dst| {
        filter_chunk::<T, C>(&x[start..start + dst.len()], c, dst)
    });
    Ok(())
}

/// The AVX-512 `where` entry.
// SAFETY: as for `compress`, the caller guarantees the CPU features, and `where_chunk`
// checks the lengths it relies on.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
pub(super) unsafe fn r#where(
    args: &[Arg],
    base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let mask = where_args(args)?;
    check_row_indices(base, mask.len());
    append_chunks(out, mask.len(), |start, dst| {
        let first = i64::try_from(base + start).expect("row index fits in i64");
        where_chunk(&mask[start..start + dst.len()], first, dst)
    });
    Ok(())
}
