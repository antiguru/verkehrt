//! AVX-512 selection entries. The keep mask of a vector comes from a compare into a mask
//! register (`filter`) or from a byte test of the bool mask (`compress`, `where`), and
//! `vpcompress` packs the kept lanes to the front of the vector, which is then stored whole.
//! Everything here is compiled for the x86-64-v4 level alone. `u8` and bool need the VBMI2
//! byte compress, so they live in `avx512_vbmi2`, and no row type here has a byte `Lanes`.

use std::arch::x86_64::*;

use crate::kernels::scalar::fused::{CMPS, Cmp};
use crate::kernels::select_loops::{ones, selection_loops};
use crate::value::Elem;

/// A row type that AVX-512 selects `LANES` rows at a time.
///
/// The methods are `unsafe` because they read and write through raw pointers and need the
/// CPU features of the `Isa::X86_64V4` level. Each states the memory its pointers must cover.
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
// the features of the `Isa::X86_64V4` level, which its callers in this file have because they
// are compiled for the same features. Pointer requirements are the trait's, and the unsafe
// blocks rely on exactly those.
macro_rules! lanes_64 {
    ($t:ty) => {
        impl Lanes for $t {
            const LANES: usize = 8;
            type Vector = __m512i;

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::keep` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 8 readable bytes at `mask`, and `loadl` reads
                // 8 bytes with no alignment requirement.
                let v = unsafe { _mm_loadl_epi64(mask.cast()) };
                u64::from(_mm_test_epi8_mask(v, v))
            }

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::load` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
            unsafe fn load(src: *const $t) -> __m512i {
                // SAFETY: the caller guarantees 8 readable rows (64 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm512_loadu_si512(src.cast()) }
            }

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::compress` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
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

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::keep` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees 16 readable bytes at `mask`, and `loadu` has
                // no alignment requirement.
                let v = unsafe { _mm_loadu_si128(mask.cast()) };
                u64::from(_mm_test_epi8_mask(v, v))
            }

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::load` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
            unsafe fn load(src: *const $t) -> __m512i {
                // SAFETY: the caller guarantees 16 readable rows (64 bytes) at `src`, and
                // `loadu` has no alignment requirement.
                unsafe { _mm512_loadu_si512(src.cast()) }
            }

            // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
            // ranges that `Lanes::compress` states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
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
lanes_64!(u64);
lanes_32!(f32);
lanes_32!(i32);

impl Compare for f64 {
    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
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
    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
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
    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
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

impl Compare for u64 {
    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    unsafe fn compare<const C: usize>(v: __m512i, c: u64) -> u64 {
        let c = _mm512_set1_epi64(c.cast_signed());
        // Unsigned compares (`vpcmpuq`), with the predicates of `i64`.
        let m = match CMPS[C].1 {
            Cmp::Eq => _mm512_cmp_epu64_mask::<_MM_CMPINT_EQ>(v, c),
            Cmp::Ne => _mm512_cmp_epu64_mask::<_MM_CMPINT_NE>(v, c),
            Cmp::Lt => _mm512_cmp_epu64_mask::<_MM_CMPINT_LT>(v, c),
            Cmp::Le => _mm512_cmp_epu64_mask::<_MM_CMPINT_LE>(v, c),
            Cmp::Gt => _mm512_cmp_epu64_mask::<_MM_CMPINT_NLE>(v, c),
            Cmp::Ge => _mm512_cmp_epu64_mask::<_MM_CMPINT_NLT>(v, c),
        };
        u64::from(m)
    }
}

impl Compare for i32 {
    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` and the pointer
    // ranges that `Compare::compare` states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
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

/// The `i64` rows `first + j` of lanes `j`, see `selection_loops`.
#[inline]
#[target_feature(
    enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
fn first_indices(first: i64) -> __m512i {
    _mm512_add_epi64(
        _mm512_set1_epi64(first),
        _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

/// `idx` advanced by the 8 `i64` lanes of a vector, see `selection_loops`.
#[inline]
#[target_feature(
    enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
fn next_indices(idx: __m512i) -> __m512i {
    _mm512_add_epi64(idx, _mm512_set1_epi64(8))
}

selection_loops!(
    "AVX-512",
    "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
);
