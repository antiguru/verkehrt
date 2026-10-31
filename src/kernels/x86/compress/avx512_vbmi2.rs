//! The AVX-512 `compress` entry on `u8` and bool rows. The keep mask comes from a byte test of
//! the bool mask, and the VBMI2 byte compress `vpcompressb` packs the kept bytes to the front
//! of the vector, which is then stored whole.
//!
//! VBMI2 is not part of the x86-64-v4 level, so everything here is compiled for the v4 string
//! plus `avx512vbmi2`, and the only entry, `compress` on `u8`, is registered declaring
//! `avx512vbmi2`. `Lanes` is private to this module and only `u8` implements it, so no entry
//! that is compiled for the v4 level alone can reach a function compiled here.

use std::arch::x86_64::*;

use crate::kernels::select_loops::{ones, selection_loops};
use crate::value::Elem;

/// A row type that AVX-512 with VBMI2 selects `LANES` rows at a time, see the `Lanes` trait
/// of `avx512`, whose contract this one repeats.
///
/// The methods are `unsafe` because they read and write through raw pointers and need the
/// CPU features of the `Isa::X86_64V4` level plus `avx512vbmi2`. Each states the memory its
/// pointers must cover.
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

// SAFETY (all `Lanes` methods below): each method is compiled for, and needs, the features of
// the `Isa::X86_64V4` level plus `avx512vbmi2`, which its callers in this file have because
// they are compiled for the same features. Pointer requirements are the trait's, and the
// unsafe blocks rely on exactly those.
impl Lanes for u8 {
    const LANES: usize = 64;
    type Vector = __m512i;

    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` plus `avx512vbmi2` and
    // the pointer ranges that `Lanes::keep` states. Nothing assumes more than element
    // alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi2"
    )]
    unsafe fn keep(mask: *const u8) -> u64 {
        // SAFETY: the caller guarantees 64 readable bytes at `mask`, and `loadu` has no
        // alignment requirement.
        let v = unsafe { _mm512_loadu_si512(mask.cast()) };
        _mm512_test_epi8_mask(v, v)
    }

    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` plus `avx512vbmi2` and
    // the pointer ranges that `Lanes::load` states. Nothing assumes more than element
    // alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi2"
    )]
    unsafe fn load(src: *const u8) -> __m512i {
        // SAFETY: the caller guarantees 64 readable bytes at `src`, and `loadu` has no
        // alignment requirement.
        unsafe { _mm512_loadu_si512(src.cast()) }
    }

    // SAFETY: the caller guarantees the features of `Isa::X86_64V4` plus `avx512vbmi2` and
    // the pointer ranges that `Lanes::compress` states. Nothing assumes more than element
    // alignment.
    #[inline]
    #[target_feature(
        enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi2"
    )]
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

selection_loops!(
    compress: "AVX-512 VBMI2",
    "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi2"
);
