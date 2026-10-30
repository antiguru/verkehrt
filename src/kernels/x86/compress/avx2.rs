//! AVX2 selection entries for 32- and 64-bit rows. The keep mask of a vector comes from a
//! compare and `movemask` (`filter`) or from a byte compare of the bool mask (`compress`,
//! `where`). It indexes a table of `vpermd` permutations that move the kept lanes to the
//! front, and the permuted vector is stored whole. AVX2 has no byte permutation of this kind
//! across 32 lanes, so `u8` and bool have no entry here.

use std::arch::x86_64::*;

use super::ones;
use crate::kernels::scalar::elementwise::tag;
use crate::kernels::scalar::fused::{CMPS, Cmp, count_cmp, filter_args};
use crate::kernels::scalar::select::{
    append_chunks, check_row_indices, compress_args, count_nonzero, predicated, where_args,
};
use crate::value::{Arg, ArrayBuilder, Elem};

/// `PERM8[m]` lists the lanes of 8 32-bit lanes whose bit in `m` is set, in order, then
/// zeros. As a `vpermd` index it packs the kept lanes to the front.
static PERM8: [[u32; 8]; 256] = perm8();

/// `PERM4[m]` does the same for 4 64-bit lanes, as pairs of 32-bit lanes.
static PERM4: [[u32; 8]; 16] = perm4();

const fn perm8() -> [[u32; 8]; 256] {
    let mut table = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut lane, mut out) = (0u32, 0);
        while lane < 8 {
            if m >> lane & 1 == 1 {
                table[m][out] = lane;
                out += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    table
}

const fn perm4() -> [[u32; 8]; 16] {
    let mut table = [[0u32; 8]; 16];
    let mut m = 0;
    while m < 16 {
        let (mut lane, mut out) = (0u32, 0);
        while lane < 4 {
            if m >> lane & 1 == 1 {
                table[m][2 * out] = 2 * lane;
                table[m][2 * out + 1] = 2 * lane + 1;
                out += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    table
}

/// A row type that AVX2 selects `LANES` rows at a time, see the AVX-512 `Lanes`.
///
/// The methods are `unsafe` because they read and write through raw pointers and need the
/// AVX2 feature. Each states the memory its pointers must cover.
pub(super) trait Lanes: Elem {
    const LANES: usize;

    /// The keep mask of the `LANES` bool bytes at `mask`, bit `j` set when byte `j` is
    /// nonzero. `mask` must be valid for reading `LANES` bytes.
    unsafe fn keep(mask: *const u8) -> u64;

    /// The `LANES` rows at `src`, which must be valid for reading them.
    unsafe fn load(src: *const Self) -> __m256i;

    /// Stores the lanes of `v` whose bit in `keep` is set, in lane order, to the front of the
    /// `LANES` rows at `dst`, and returns their count. Bits of `keep` at `LANES` and above are
    /// ignored. The other rows at `dst` get unspecified values. `dst` must be valid for
    /// writing `LANES` rows.
    unsafe fn compress(v: __m256i, keep: u64, dst: *mut Self) -> usize;
}

/// A row type whose comparisons AVX2 evaluates `LANES` rows at a time.
pub(super) trait Compare: Lanes + PartialOrd {
    /// The mask of the lanes of `v` with `v[j] <cmp> c`, for comparison number `C` of `CMPS`,
    /// with the predicate of `Cmp::test`.
    unsafe fn compare<const C: usize>(v: __m256i, c: Self) -> u64;
}

/// The bits of the zero bytes among the low `lanes` bytes of `v`, inverted: bit `j` is set
/// when byte `j` is nonzero.
#[inline]
#[target_feature(enable = "avx2")]
fn nonzero_bytes(v: __m128i, lanes: usize) -> u64 {
    let zero = _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_setzero_si128()));
    let all = (1u64 << lanes) - 1;
    !u64::from(zero.cast_unsigned()) & all
}

/// Stores `v` permuted by `perm` at `dst`.
///
/// # Safety
///
/// `dst` must be valid for writing 32 bytes, and the CPU must support AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn permute_store<T>(v: __m256i, perm: &[u32; 8], dst: *mut T) {
    // SAFETY: `perm` is a reference to 32 readable bytes, the caller guarantees 32 writable
    // bytes at `dst`, and `loadu` and `storeu` have no alignment requirement.
    unsafe {
        let p = _mm256_loadu_si256(perm.as_ptr().cast());
        _mm256_storeu_si256(dst.cast(), _mm256_permutevar8x32_epi32(v, p));
    }
}

// SAFETY (all `Lanes` methods below): each method is compiled for, and needs, AVX2, which
// its callers in this file have because they are compiled for the same feature. Pointer
// requirements are the trait's, and the unsafe blocks rely on exactly those.
macro_rules! lanes_64 {
    ($t:ty) => {
        impl Lanes for $t {
            const LANES: usize = 4;

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::keep`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 4 readable bytes at `mask`, and `loadu_si32`
                // reads 4 bytes with no alignment requirement.
                nonzero_bytes(unsafe { _mm_loadu_si32(mask.cast()) }, 4)
            }

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::load`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn load(src: *const $t) -> __m256i {
                // SAFETY: the caller guarantees 4 readable rows (32 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm256_loadu_si256(src.cast()) }
            }

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::compress`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn compress(v: __m256i, keep: u64, dst: *mut $t) -> usize {
                let m = keep & 0xf;
                let perm = &PERM4[usize::try_from(m).expect("4 bits fit in usize")];
                // The permutation moves bits, so float rows keep their exact bits.
                // SAFETY: the caller guarantees 4 writable rows (32 bytes) at `dst`.
                unsafe { permute_store(v, perm, dst) };
                ones(m)
            }
        }
    };
}

macro_rules! lanes_32 {
    ($t:ty) => {
        impl Lanes for $t {
            const LANES: usize = 8;

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::keep`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 8 readable bytes at `mask`, and `loadl` reads
                // 8 bytes with no alignment requirement.
                nonzero_bytes(unsafe { _mm_loadl_epi64(mask.cast()) }, 8)
            }

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::load`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn load(src: *const $t) -> __m256i {
                // SAFETY: the caller guarantees 8 readable rows (32 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm256_loadu_si256(src.cast()) }
            }

            // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Lanes::compress`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn compress(v: __m256i, keep: u64, dst: *mut $t) -> usize {
                let m = keep & 0xff;
                let perm = &PERM8[usize::try_from(m).expect("8 bits fit in usize")];
                // The permutation moves bits, see `lanes_64`.
                // SAFETY: the caller guarantees 8 writable rows (32 bytes) at `dst`.
                unsafe { permute_store(v, perm, dst) };
                ones(m)
            }
        }
    };
}

lanes_64!(f64);
lanes_64!(i64);
lanes_32!(f32);
lanes_32!(i32);

/// The `movemask` bits of a compare result, as a keep mask.
#[inline]
#[target_feature(enable = "avx2")]
fn bits(movemask: i32) -> u64 {
    u64::from(movemask.cast_unsigned())
}

impl Compare for f64 {
    // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Compare::compare`
    // states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare<const C: usize>(v: __m256i, c: f64) -> u64 {
        let (v, c) = (_mm256_castsi256_pd(v), _mm256_set1_pd(c));
        // Ordered predicates are false on NaN like `<`, and `!=` is the unordered NEQ, which
        // is true on NaN like Rust's `!=`.
        let r = match CMPS[C].1 {
            Cmp::Eq => _mm256_cmp_pd::<_CMP_EQ_OQ>(v, c),
            Cmp::Ne => _mm256_cmp_pd::<_CMP_NEQ_UQ>(v, c),
            Cmp::Lt => _mm256_cmp_pd::<_CMP_LT_OQ>(v, c),
            Cmp::Le => _mm256_cmp_pd::<_CMP_LE_OQ>(v, c),
            Cmp::Gt => _mm256_cmp_pd::<_CMP_GT_OQ>(v, c),
            Cmp::Ge => _mm256_cmp_pd::<_CMP_GE_OQ>(v, c),
        };
        bits(_mm256_movemask_pd(r))
    }
}

impl Compare for f32 {
    // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Compare::compare`
    // states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare<const C: usize>(v: __m256i, c: f32) -> u64 {
        let (v, c) = (_mm256_castsi256_ps(v), _mm256_set1_ps(c));
        // The predicates of the f64 case.
        let r = match CMPS[C].1 {
            Cmp::Eq => _mm256_cmp_ps::<_CMP_EQ_OQ>(v, c),
            Cmp::Ne => _mm256_cmp_ps::<_CMP_NEQ_UQ>(v, c),
            Cmp::Lt => _mm256_cmp_ps::<_CMP_LT_OQ>(v, c),
            Cmp::Le => _mm256_cmp_ps::<_CMP_LE_OQ>(v, c),
            Cmp::Gt => _mm256_cmp_ps::<_CMP_GT_OQ>(v, c),
            Cmp::Ge => _mm256_cmp_ps::<_CMP_GE_OQ>(v, c),
        };
        bits(_mm256_movemask_ps(r))
    }
}

impl Compare for i64 {
    // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Compare::compare`
    // states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare<const C: usize>(v: __m256i, c: i64) -> u64 {
        let c = _mm256_set1_epi64x(c);
        let mask = |r: __m256i| bits(_mm256_movemask_pd(_mm256_castsi256_pd(r)));
        // AVX2 has only signed `==` and `>`. The other predicates are their negations or
        // take the operands swapped.
        match CMPS[C].1 {
            Cmp::Eq => mask(_mm256_cmpeq_epi64(v, c)),
            Cmp::Ne => mask(_mm256_cmpeq_epi64(v, c)) ^ 0xf,
            Cmp::Lt => mask(_mm256_cmpgt_epi64(c, v)),
            Cmp::Le => mask(_mm256_cmpgt_epi64(v, c)) ^ 0xf,
            Cmp::Gt => mask(_mm256_cmpgt_epi64(v, c)),
            Cmp::Ge => mask(_mm256_cmpgt_epi64(c, v)) ^ 0xf,
        }
    }
}

impl Compare for i32 {
    // SAFETY: the caller guarantees AVX2 and the pointer ranges that `Compare::compare`
    // states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare<const C: usize>(v: __m256i, c: i32) -> u64 {
        let c = _mm256_set1_epi32(c);
        let mask = |r: __m256i| bits(_mm256_movemask_ps(_mm256_castsi256_ps(r)));
        // As for i64.
        match CMPS[C].1 {
            Cmp::Eq => mask(_mm256_cmpeq_epi32(v, c)),
            Cmp::Ne => mask(_mm256_cmpeq_epi32(v, c)) ^ 0xff,
            Cmp::Lt => mask(_mm256_cmpgt_epi32(c, v)),
            Cmp::Le => mask(_mm256_cmpgt_epi32(v, c)) ^ 0xff,
            Cmp::Gt => mask(_mm256_cmpgt_epi32(v, c)),
            Cmp::Ge => mask(_mm256_cmpgt_epi32(c, v)) ^ 0xff,
        }
    }
}

/// Writes the rows of `x` whose `mask` byte is nonzero to the front of `dst` and returns
/// their count. Panics unless `mask` has the rows of `x`, or when `dst` has no room for a
/// predicated write of the survivors, see `predicated`.
#[target_feature(enable = "avx2")]
fn compress_chunk<T: Lanes>(x: &[T], mask: &[u8], dst: &mut [T]) -> usize {
    // The pointer arithmetic below relies on equal lengths, which the entry's caller does not
    // guarantee, so they are checked here.
    assert!(
        mask.len() == x.len(),
        "compress chunk of {} rows with {} mask rows",
        x.len(),
        mask.len()
    );
    let n = x.len();
    let (mut i, mut k) = (0, 0);
    // A vector step stores a whole vector, so it runs only while one fits in `dst`, and the
    // predicated loop finishes the rest.
    while i + T::LANES <= n && k + T::LANES <= dst.len() {
        // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` of `x` and `mask` are in bounds,
        // and `k + LANES <= dst.len()`, so rows `k..k + LANES` of `dst` are in bounds. This
        // function has AVX2, the feature of `Lanes`.
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

/// The number of rows `v` of `x` with `v <cmp> c`, for comparison number `C` of `CMPS`, with
/// the compares of `filter_chunk`.
#[target_feature(enable = "avx2")]
fn count_rows<T: Compare, const C: usize>(x: &[T], c: T) -> usize {
    let n = x.len();
    let (mut i, mut kept) = (0, 0);
    while i + T::LANES <= n {
        // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` of `x` are in bounds. This
        // function has AVX2, the feature of `Compare`.
        kept += unsafe { ones(T::compare::<C>(T::load(x.as_ptr().add(i)), c)) };
        i += T::LANES;
    }
    kept + count_cmp(&x[i..], CMPS[C].1, c)
}

/// Writes the rows `v` of `x` with `v <cmp> c` to the front of `dst`, for comparison number
/// `C` of `CMPS`, and returns their count. Panics when `dst` has no room for a predicated
/// write of the survivors, see `predicated`.
#[target_feature(enable = "avx2")]
fn filter_chunk<T: Compare, const C: usize>(x: &[T], c: T, dst: &mut [T]) -> usize {
    let n = x.len();
    let (mut i, mut k) = (0, 0);
    // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
    while i + T::LANES <= n && k + T::LANES <= dst.len() {
        // SAFETY: as in `compress_chunk`, rows `i..i + LANES` of `x` and rows `k..k + LANES`
        // of `dst` are in bounds by the loop condition. This function has AVX2, the feature of `Compare`.
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
/// `dst` and returns their count. Panics when `dst` has no room for a predicated write of
/// the survivors, see `predicated`. The caller guarantees that `first + mask.len()` fits in
/// i64.
#[target_feature(enable = "avx2")]
fn where_chunk(mask: &[u8], first: i64, dst: &mut [i64]) -> usize {
    let n = mask.len();
    let lanes = <i64 as Lanes>::LANES;
    let step = _mm256_set1_epi64x(4);
    // Lane `j` holds the index of row `i + j`. The index of every row of the chunk fits in
    // i64, so every lane that is stored holds its exact index. Lanes past the last row, after
    // the final add, may wrap, which is harmless because they are never stored and SIMD adds
    // wrap without undefined behavior.
    let mut idx = _mm256_add_epi64(_mm256_set1_epi64x(first), _mm256_setr_epi64x(0, 1, 2, 3));
    let (mut i, mut k) = (0, 0);
    // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
    while i + lanes <= n && k + lanes <= dst.len() {
        // SAFETY: as in `compress_chunk`, bytes `i..i + lanes` of `mask` and rows
        // `k..k + lanes` of `dst` are in bounds by the loop condition. This function has
        // AVX2, the feature of `Lanes`.
        unsafe {
            let keep = i64::keep(mask.as_ptr().add(i));
            k += i64::compress(idx, keep, dst.as_mut_ptr().add(k));
        }
        idx = _mm256_add_epi64(idx, step);
        i += lanes;
    }
    let at = first + i64::try_from(i).expect("row index fits in i64");
    let rest = (at..).zip(&mask[i..]).map(|(r, &m)| (r, m != 0));
    k + predicated(rest, &mut dst[k..])
}

/// The AVX2 `compress` entry on rows of storage type `T` and type `tag::<T, BOOL>()`.
// SAFETY: the caller guarantees that the CPU supports AVX2, which `Dispatch` does by
// calling only entries that `resolve` installed after `is_available` returned true. Memory
// safety does not depend on the caller: `compress_args` checks that the rows and the mask
// have the same length, and `compress_chunk` checks the lengths it relies on again.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress<T: Lanes, const BOOL: bool>(
    args: &[Arg],
    _base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let (x, mask) = compress_args::<T>(args, tag::<T, BOOL>())?;
    // The count pass sizes the output exactly. It is the portable count, compiled for this
    // ISA.
    append_chunks(out, x.len(), count_nonzero(mask), |rows, dst| {
        compress_chunk(&x[rows.clone()], &mask[rows], dst)
    });
    Ok(())
}

/// The AVX2 `filter_<cmp>` entry on rows of type `T`, for comparison number `C` of
/// `CMPS`.
// SAFETY: as for `compress`, the caller guarantees AVX2, and the chunk functions check the
// lengths they rely on.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn filter<T: Compare, const C: usize>(
    args: &[Arg],
    _base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let (x, c) = filter_args::<T>(args)?;
    append_chunks(out, x.len(), count_rows::<T, C>(x, c), |rows, dst| {
        filter_chunk::<T, C>(&x[rows], c, dst)
    });
    Ok(())
}

/// The AVX2 `where` entry.
// SAFETY: as for `compress`, the caller guarantees AVX2, and `where_chunk` checks the
// lengths it relies on.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn r#where(
    args: &[Arg],
    base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    let mask = where_args(args)?;
    check_row_indices(base, mask.len());
    append_chunks(out, mask.len(), count_nonzero(mask), |rows, dst| {
        let first = i64::try_from(base + rows.start).expect("row index fits in i64");
        where_chunk(&mask[rows], first, dst)
    });
    Ok(())
}
