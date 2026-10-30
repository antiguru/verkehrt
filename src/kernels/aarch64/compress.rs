//! The NEON intrinsics backend of the selections: `compress_*`, `where`, and the fused
//! `filter_*_*`.
//!
//! Each entry parses its arguments with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails. It counts the survivors of the tile first, then appends through
//! `append_chunks`, which reserves that many rows plus one vector. Within a chunk it selects
//! a vector of rows at a time while a whole vector store fits in the chunk's spare rows, and
//! finishes the rest with the portable predicated loop.
//!
//! AArch64 has no compress instruction. The keep mask of a 128-bit vector comes from a
//! compare (`filter`) or from a byte test of the bool mask (`compress`, `where`), reduced to
//! one bit per lane. It indexes a table of byte shuffles that move the kept lanes to the
//! front, `tbl` applies the shuffle, and the shuffled vector is stored whole. `u8` and bool
//! rows have 16 lanes, too many for one table, so each 8-byte half takes its own 8-bit
//! shuffle and store.

use std::arch::aarch64::*;

use crate::kernels::SelectionFn;
use crate::kernels::scalar::elementwise::tag;
use crate::kernels::scalar::fused::{CMPS, Cmp, count_cmp, filter_args};
use crate::kernels::scalar::select::{
    append_chunks, check_row_indices, compress_args, count_nonzero, predicated, where_args,
};
use crate::value::{Arg, ArrayBuilder, Elem, ScalarType};

/// The NEON entry of `compress` on rows of storage type `T`, whose semantic type is
/// `tag::<T, BOOL>()`, where it exists.
pub(crate) fn compress_entry<T: Elem, const BOOL: bool>() -> Option<SelectionFn> {
    // Each storage type has its own `ScalarType`, so an arm instantiates the entry for
    // exactly `T`. Only `u8` is instantiated with `BOOL` set.
    match T::TYPE {
        ScalarType::F64 => Some(compress::<f64, BOOL>),
        ScalarType::F32 => Some(compress::<f32, BOOL>),
        ScalarType::I64 => Some(compress::<i64, BOOL>),
        ScalarType::I32 => Some(compress::<i32, BOOL>),
        ScalarType::U8 => Some(compress::<u8, BOOL>),
        _ => None,
    }
}

/// The NEON entry of `where`.
pub(crate) fn where_entry() -> SelectionFn {
    r#where
}

/// The NEON entry of `filter_<cmp>` on rows of type `T`, for comparison number `C` of
/// `CMPS`, where it exists.
pub(crate) fn filter_entry<T: Elem, const C: usize>() -> Option<SelectionFn> {
    // As in `compress_entry`, an arm instantiates the entry for exactly `T`.
    match T::TYPE {
        ScalarType::F64 => Some(filter::<f64, C>),
        ScalarType::F32 => Some(filter::<f32, C>),
        ScalarType::I64 => Some(filter::<i64, C>),
        ScalarType::I32 => Some(filter::<i32, C>),
        _ => None,
    }
}

/// The number of set bits, as a row count.
#[inline(always)]
fn ones(bits: u64) -> usize {
    usize::try_from(bits.count_ones()).expect("a bit count fits in usize")
}

/// `shuffles::<M, B, W>()[m]` lists the bytes of the `W`-byte lanes of a `B`-byte vector
/// whose bit in `m` is set, in order, then `0xff`. As a `tbl` index it packs the kept lanes
/// to the front and zeros the rest. `M` is `1 << (B / W)`.
const fn shuffles<const M: usize, const B: usize, const W: usize>() -> [[u8; B]; M] {
    let mut table = [[0xffu8; B]; M];
    let lanes = B / W;
    let mut m = 0;
    while m < M {
        let (mut lane, mut out) = (0, 0);
        while lane < lanes {
            if m >> lane & 1 == 1 {
                let mut b = 0;
                while b < W {
                    // `lane * W + b < B <= 16`, so the low byte is the whole index.
                    table[m][out * W + b] = (lane * W + b).to_le_bytes()[0];
                    b += 1;
                }
                out += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    table
}

/// Shuffles that pack the kept lanes of 2 64-bit lanes, indexed by a 2-bit keep mask.
static SHUFFLE_64: [[u8; 16]; 4] = shuffles::<4, 16, 8>();

/// Shuffles that pack the kept lanes of 4 32-bit lanes, indexed by a 4-bit keep mask.
static SHUFFLE_32: [[u8; 16]; 16] = shuffles::<16, 16, 4>();

/// Shuffles that pack the kept bytes of 8 bytes, indexed by an 8-bit keep mask.
static SHUFFLE_8: [[u8; 8]; 256] = shuffles::<256, 8, 1>();

/// The weight of each byte lane in a bit mask, `1 << j` in byte `j`.
const BYTE_WEIGHTS: u64 = 0x8040_2010_0804_0201;

/// The keep mask of the 8 bytes of `v`, bit `j` set when byte `j` is nonzero.
#[inline]
#[target_feature(enable = "neon")]
fn byte_bits(v: uint8x8_t) -> u64 {
    // Each nonzero byte becomes its distinct bit, so the wrapping byte sum is their OR.
    let weighted = vand_u8(vtst_u8(v, v), vcreate_u8(BYTE_WEIGHTS));
    u64::from(vaddv_u8(weighted))
}

/// The bits of the 64-bit compare result `m`, bit `j` set when lane `j` is all ones.
#[inline]
#[target_feature(enable = "neon")]
fn bits_64(m: uint64x2_t) -> u64 {
    let weights = vcombine_u64(vcreate_u64(1), vcreate_u64(2));
    vaddvq_u64(vandq_u64(m, weights))
}

/// The bits of the 32-bit compare result `m`, bit `j` set when lane `j` is all ones.
#[inline]
#[target_feature(enable = "neon")]
fn bits_32(m: uint32x4_t) -> u64 {
    // `vcreate_u32` puts the low 32 bits in lane 0.
    let weights = vcombine_u32(vcreate_u32(1 | 2 << 32), vcreate_u32(4 | 8 << 32));
    u64::from(vaddvq_u32(vandq_u32(m, weights)))
}

/// The 16 bytes at `src` as a vector.
///
/// # Safety
///
/// `src` must be valid for reading 16 bytes, and the CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load<T>(src: *const T) -> uint8x16_t {
    // SAFETY: the caller guarantees 16 readable bytes at `src`, and `ld1` of bytes has no
    // alignment requirement.
    unsafe { vld1q_u8(src.cast()) }
}

/// Stores `v` shuffled by `shuffle` at `dst`.
///
/// # Safety
///
/// `dst` must be valid for writing 16 bytes, and the CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn shuffle_store<T>(v: uint8x16_t, shuffle: &[u8; 16], dst: *mut T) {
    // SAFETY: `shuffle` is a reference to 16 readable bytes, the caller guarantees 16
    // writable bytes at `dst`, and `ld1` and `st1` of bytes have no alignment requirement.
    unsafe {
        let s = vld1q_u8(shuffle.as_ptr());
        vst1q_u8(dst.cast(), vqtbl1q_u8(v, s));
    }
}

/// A row type that NEON selects `LANES` rows, one 128-bit vector, at a time.
///
/// The methods are `unsafe` because they read and write through raw pointers and need NEON.
/// Each states the memory its pointers must cover.
pub(super) trait Lanes: Elem {
    const LANES: usize;

    /// The keep mask of the `LANES` bool bytes at `mask`, bit `j` set when byte `j` is
    /// nonzero. `mask` must be valid for reading `LANES` bytes.
    unsafe fn keep(mask: *const u8) -> u64;

    /// Stores the lanes of `v` whose bit in `keep` is set, in lane order, to the front of the
    /// `LANES` rows at `dst`, and returns their count. Bits of `keep` at `LANES` and above are
    /// ignored. The other rows at `dst` get unspecified values. `dst` must be valid for
    /// writing `LANES` rows.
    unsafe fn compress(v: uint8x16_t, keep: u64, dst: *mut Self) -> usize;
}

/// A row type whose comparisons NEON evaluates `LANES` rows at a time.
pub(super) trait Compare: Lanes + PartialOrd {
    /// The mask of the lanes of `v` with `v[j] <cmp> c`, for comparison number `C` of `CMPS`,
    /// with the predicate of `Cmp::test`.
    unsafe fn compare<const C: usize>(v: uint8x16_t, c: Self) -> u64;
}

// SAFETY (all `Lanes` methods below): each method is compiled for, and needs, NEON, which
// its callers in this file have because they are compiled for the same feature. Pointer
// requirements are the trait's, and the unsafe blocks rely on exactly those.
macro_rules! lanes {
    ($t:ty, $lanes:literal, $word:ty, $table:ident) => {
        impl Lanes for $t {
            const LANES: usize = $lanes;

            // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::keep`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn keep(mask: *const u8) -> u64 {
                // SAFETY: the caller guarantees `LANES` readable bytes at `mask`, the size of
                // the word read here, and `read_unaligned` has no alignment requirement.
                let word = unsafe { mask.cast::<$word>().read_unaligned() };
                // The bytes past `LANES` are zero, so their bits stay clear.
                byte_bits(vcreate_u8(u64::from(word)))
            }

            // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::compress`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn compress(v: uint8x16_t, keep: u64, dst: *mut $t) -> usize {
                let m = keep & ((1 << $lanes) - 1);
                let shuffle = &$table[usize::try_from(m).expect("a lane mask fits in usize")];
                // The shuffle moves bytes, so float rows keep their exact bits.
                // SAFETY: the caller guarantees `LANES` writable rows (16 bytes) at `dst`.
                unsafe { shuffle_store(v, shuffle, dst) };
                ones(m)
            }
        }
    };
}

lanes!(f64, 2, u16, SHUFFLE_64);
lanes!(i64, 2, u16, SHUFFLE_64);
lanes!(f32, 4, u32, SHUFFLE_32);
lanes!(i32, 4, u32, SHUFFLE_32);

impl Lanes for u8 {
    const LANES: usize = 16;

    // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::keep` states.
    // Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn keep(mask: *const u8) -> u64 {
        // SAFETY: the caller guarantees 16 readable bytes at `mask`.
        let v = unsafe { load(mask) };
        byte_bits(vget_low_u8(v)) | byte_bits(vget_high_u8(v)) << 8
    }

    // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::compress`
    // states. Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compress(v: uint8x16_t, keep: u64, dst: *mut u8) -> usize {
        // Each half packs its kept bytes with an 8-byte shuffle, and the high half is stored
        // right after the survivors of the low half. Bool rows are 0 or 1 in the input, and
        // the shuffle only moves them.
        let (lo, hi) = (keep & 0xff, keep >> 8 & 0xff);
        let index = |m: u64| usize::try_from(m).expect("8 bits fit in usize");
        let low = ones(lo);
        // SAFETY: the shuffles are references to 8 readable bytes, and `ld1` and `st1` of
        // bytes have no alignment requirement. The caller guarantees 16 writable bytes at
        // `dst`. The first store writes bytes `0..8`, and the second bytes `low..low + 8`
        // with `low <= 8`, so both stay inside them.
        unsafe {
            let s_lo = vld1_u8(SHUFFLE_8[index(lo)].as_ptr());
            vst1_u8(dst, vtbl1_u8(vget_low_u8(v), s_lo));
            let s_hi = vld1_u8(SHUFFLE_8[index(hi)].as_ptr());
            vst1_u8(dst.add(low), vtbl1_u8(vget_high_u8(v), s_hi));
        }
        low + ones(hi)
    }
}

impl Compare for f64 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16_t, c: f64) -> u64 {
        let (v, c) = (vreinterpretq_f64_u8(v), vdupq_n_f64(c));
        // The NEON float compares are ordered, false on NaN like `<`. `!=` is the negation
        // of `==`, which is true on NaN like Rust's `!=`.
        match CMPS[C].1 {
            Cmp::Eq => bits_64(vceqq_f64(v, c)),
            Cmp::Ne => bits_64(vceqq_f64(v, c)) ^ 0b11,
            Cmp::Lt => bits_64(vcltq_f64(v, c)),
            Cmp::Le => bits_64(vcleq_f64(v, c)),
            Cmp::Gt => bits_64(vcgtq_f64(v, c)),
            Cmp::Ge => bits_64(vcgeq_f64(v, c)),
        }
    }
}

impl Compare for f32 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16_t, c: f32) -> u64 {
        let (v, c) = (vreinterpretq_f32_u8(v), vdupq_n_f32(c));
        // The predicates of the f64 case.
        match CMPS[C].1 {
            Cmp::Eq => bits_32(vceqq_f32(v, c)),
            Cmp::Ne => bits_32(vceqq_f32(v, c)) ^ 0b1111,
            Cmp::Lt => bits_32(vcltq_f32(v, c)),
            Cmp::Le => bits_32(vcleq_f32(v, c)),
            Cmp::Gt => bits_32(vcgtq_f32(v, c)),
            Cmp::Ge => bits_32(vcgeq_f32(v, c)),
        }
    }
}

impl Compare for i64 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16_t, c: i64) -> u64 {
        let (v, c) = (vreinterpretq_s64_u8(v), vdupq_n_s64(c));
        // The signed compares, and `!=` as the negation of `==`.
        match CMPS[C].1 {
            Cmp::Eq => bits_64(vceqq_s64(v, c)),
            Cmp::Ne => bits_64(vceqq_s64(v, c)) ^ 0b11,
            Cmp::Lt => bits_64(vcltq_s64(v, c)),
            Cmp::Le => bits_64(vcleq_s64(v, c)),
            Cmp::Gt => bits_64(vcgtq_s64(v, c)),
            Cmp::Ge => bits_64(vcgeq_s64(v, c)),
        }
    }
}

impl Compare for i32 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16_t, c: i32) -> u64 {
        let (v, c) = (vreinterpretq_s32_u8(v), vdupq_n_s32(c));
        // As for i64.
        match CMPS[C].1 {
            Cmp::Eq => bits_32(vceqq_s32(v, c)),
            Cmp::Ne => bits_32(vceqq_s32(v, c)) ^ 0b1111,
            Cmp::Lt => bits_32(vcltq_s32(v, c)),
            Cmp::Le => bits_32(vcleq_s32(v, c)),
            Cmp::Gt => bits_32(vcgtq_s32(v, c)),
            Cmp::Ge => bits_32(vcgeq_s32(v, c)),
        }
    }
}

/// Writes the rows of `x` whose `mask` byte is nonzero to the front of `dst` and returns
/// their count. Panics unless `mask` has the rows of `x`, or when `dst` has no room for a
/// predicated write of the survivors, see `predicated`.
#[target_feature(enable = "neon")]
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
        // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` (16 bytes) of `x` and `LANES`
        // bytes of `mask` are in bounds, and `k + LANES <= dst.len()`, so rows
        // `k..k + LANES` of `dst` are in bounds. This function has NEON, the feature of
        // `Lanes` and `load`.
        unsafe {
            let keep = T::keep(mask.as_ptr().add(i));
            let v = load(x.as_ptr().add(i));
            k += T::compress(v, keep, dst.as_mut_ptr().add(k));
        }
        i += T::LANES;
    }
    let rest = x[i..].iter().zip(&mask[i..]).map(|(&v, &m)| (v, m != 0));
    k + predicated(rest, &mut dst[k..])
}

/// The number of rows `v` of `x` with `v <cmp> c`, for comparison number `C` of `CMPS`, with
/// the compares of `filter_chunk`.
#[target_feature(enable = "neon")]
fn count_rows<T: Compare, const C: usize>(x: &[T], c: T) -> usize {
    let n = x.len();
    let (mut i, mut kept) = (0, 0);
    while i + T::LANES <= n {
        // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` (16 bytes) of `x` are in bounds.
        // This function has NEON, the feature of `Compare` and `load`.
        kept += unsafe { ones(T::compare::<C>(load(x.as_ptr().add(i)), c)) };
        i += T::LANES;
    }
    kept + count_cmp(&x[i..], CMPS[C].1, c)
}

/// Writes the rows `v` of `x` with `v <cmp> c` to the front of `dst`, for comparison number
/// `C` of `CMPS`, and returns their count. Panics when `dst` has no room for a predicated
/// write of the survivors, see `predicated`.
#[target_feature(enable = "neon")]
fn filter_chunk<T: Compare, const C: usize>(x: &[T], c: T, dst: &mut [T]) -> usize {
    let n = x.len();
    let (mut i, mut k) = (0, 0);
    // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
    while i + T::LANES <= n && k + T::LANES <= dst.len() {
        // SAFETY: as in `compress_chunk`, rows `i..i + LANES` of `x` and rows `k..k + LANES`
        // of `dst` are in bounds by the loop condition. This function has NEON, the feature
        // of `Compare` and `load`.
        unsafe {
            let v = load(x.as_ptr().add(i));
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
#[target_feature(enable = "neon")]
fn where_chunk(mask: &[u8], first: i64, dst: &mut [i64]) -> usize {
    let n = mask.len();
    let lanes = <i64 as Lanes>::LANES;
    let step = vdupq_n_s64(2);
    // Lane `j` holds the index of row `i + j`. The index of every row of the chunk fits in
    // i64, so every lane that is stored holds its exact index. Lanes past the last row, after
    // the final add, may wrap, which is harmless because they are never stored and SIMD adds
    // wrap without undefined behavior.
    let mut idx = vaddq_s64(
        vdupq_n_s64(first),
        vcombine_s64(vcreate_s64(0), vcreate_s64(1)),
    );
    let (mut i, mut k) = (0, 0);
    // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
    while i + lanes <= n && k + lanes <= dst.len() {
        // SAFETY: as in `compress_chunk`, bytes `i..i + lanes` of `mask` and rows
        // `k..k + lanes` of `dst` are in bounds by the loop condition. This function has
        // NEON, the feature of `Lanes`.
        unsafe {
            let keep = i64::keep(mask.as_ptr().add(i));
            k += i64::compress(vreinterpretq_u8_s64(idx), keep, dst.as_mut_ptr().add(k));
        }
        idx = vaddq_s64(idx, step);
        i += lanes;
    }
    let at = first + i64::try_from(i).expect("row index fits in i64");
    let rest = (at..).zip(&mask[i..]).map(|(r, &m)| (r, m != 0));
    k + predicated(rest, &mut dst[k..])
}

/// The NEON `compress` entry on rows of storage type `T` and type `tag::<T, BOOL>()`.
// SAFETY: the caller guarantees that the CPU supports NEON, which `Dispatch` does by calling
// only entries that `resolve` installed after `is_available` returned true. Memory safety
// does not depend on the caller: `compress_args` checks that the rows and the mask have the
// same length, and `compress_chunk` checks the lengths it relies on again.
#[target_feature(enable = "neon")]
unsafe fn compress<T: Lanes, const BOOL: bool>(
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

/// The NEON `filter_<cmp>` entry on rows of type `T`, for comparison number `C` of `CMPS`.
// SAFETY: as for `compress`, the caller guarantees NEON, and the chunk functions check the
// lengths they rely on.
#[target_feature(enable = "neon")]
unsafe fn filter<T: Compare, const C: usize>(
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

/// The NEON `where` entry.
// SAFETY: as for `compress`, the caller guarantees NEON, and `where_chunk` checks the
// lengths it relies on.
#[target_feature(enable = "neon")]
unsafe fn r#where(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String> {
    let mask = where_args(args)?;
    check_row_indices(base, mask.len());
    append_chunks(out, mask.len(), count_nonzero(mask), |rows, dst| {
        let first = i64::try_from(base + rows.start).expect("row index fits in i64");
        where_chunk(&mask[rows], first, dst)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks every entry of a shuffle table of `W`-byte lanes by applying it to the lane
    /// bytes `0..B` the way `tbl` does and comparing with the kept lanes in order.
    fn check_table<const B: usize>(table: &[[u8; B]], w: usize) {
        let lanes = B / w;
        assert_eq!(table.len(), 1 << lanes);
        for (m, shuffle) in table.iter().enumerate() {
            let kept: Vec<u8> = (0..lanes)
                .filter(|lane| m >> lane & 1 == 1)
                .flat_map(|lane| lane * w..(lane + 1) * w)
                .map(|b| u8::try_from(b).unwrap())
                .collect();
            let applied: Vec<u8> = shuffle
                .iter()
                .map(|&s| if usize::from(s) < B { s } else { 0 })
                .collect();
            assert_eq!(&applied[..kept.len()], &kept[..], "mask {m:#b}");
            assert!(
                shuffle[kept.len()..].iter().all(|&s| s == 0xff),
                "mask {m:#b}"
            );
        }
    }

    #[test]
    fn shuffle_tables_pack_kept_lanes() {
        check_table(&SHUFFLE_64, 8);
        check_table(&SHUFFLE_32, 4);
        check_table(&SHUFFLE_8, 1);
    }
}
