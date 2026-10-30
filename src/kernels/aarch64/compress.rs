//! The NEON intrinsics backend of the selections: `compress_*`, `where`, and the fused
//! `filter_*_*`.
//!
//! Each entry parses its arguments with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails. It counts the survivors of the tile first, then appends through
//! `append_chunks`, which reserves that many rows plus one vector. Within a chunk it selects
//! a vector of rows at a time while a whole vector store fits in the chunk's spare rows, and
//! finishes the rest with the portable predicated loop. The loops and entries are shared
//! with the other ISAs, see `crate::kernels::select_loops`.
//!
//! AArch64 has no compress instruction. The keep mask of a 128-bit vector comes from a
//! compare (`filter`) or from a byte test of the bool mask (`compress`, `where`), reduced to
//! one bit per lane. It indexes a table of byte shuffles that move the kept lanes to the
//! front, `tbl` applies the shuffle, and the shuffled vector is stored whole. `u8` and bool
//! rows have 16 lanes, too many for one table, so each 8-byte half takes its own 8-bit
//! shuffle and store. The 64-bit and 32-bit rows step four vectors at a time, with one keep
//! mask for all of them and one shuffle and store per vector, see `lanes!`.

use std::arch::aarch64::*;

use crate::kernels::SelectionFn;
use crate::kernels::scalar::fused::{CMPS, Cmp};
use crate::kernels::select_loops::{ones, selection_loops};
use crate::value::{Elem, ScalarType};

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

/// `PREFIX[m]` holds in byte `j` the number of set bits among the low `2 * (j + 1)` bits of
/// the 8-bit mask `m`. Byte 3 is the count of all set bits of `m`.
static PREFIX: [u32; 256] = prefix_counts();

const fn prefix_counts() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut m = 0;
    while m < 256 {
        let mut j = 0;
        while j < 4 {
            let low = m & ((4 << (2 * j)) - 1);
            table[m] |= low.count_ones() << (8 * j);
            j += 1;
        }
        m += 1;
    }
    table
}

/// The `PREFIX` entry of the low 8 bits of `keep`.
#[inline(always)]
fn prefix_entry(keep: u64) -> u32 {
    PREFIX[usize::try_from(keep & 0xff).expect("a byte fits in usize")]
}

/// The shuffle of `table` for the lane mask `m`, which must index it.
#[inline(always)]
fn lane_shuffle(table: &[[u8; 16]], m: u64) -> &[u8; 16] {
    &table[usize::try_from(m).expect("a lane mask fits in usize")]
}

/// Byte `j` of a `PREFIX` entry, as a row count.
#[inline(always)]
fn prefix_byte(p: u32, j: usize) -> usize {
    usize::from(p.to_le_bytes()[j])
}

/// The weight of each byte lane in a bit mask, `1 << j` in byte `j`.
const BYTE_WEIGHTS: u64 = 0x8040_2010_0804_0201;

/// The bits of the 8 byte masks `m`, bit `j` set when byte `j` is all ones. Every byte must
/// be all ones or zero.
#[inline]
#[target_feature(enable = "neon")]
fn lane_bits_8(m: uint8x8_t) -> u64 {
    // Each set byte becomes its distinct bit, so the wrapping byte sum is their OR.
    u64::from(vaddv_u8(vand_u8(m, vcreate_u8(BYTE_WEIGHTS))))
}

/// The bits of the 16 byte masks `m`, bit `j` set when byte `j` is all ones. Every byte must
/// be all ones or zero.
#[inline]
#[target_feature(enable = "neon")]
fn lane_bits_16(m: uint8x16_t) -> u64 {
    lane_bits_8(vget_low_u8(m)) | lane_bits_8(vget_high_u8(m)) << 8
}

/// The keep mask of the 8 bytes of `v`, bit `j` set when byte `j` is nonzero.
#[inline]
#[target_feature(enable = "neon")]
fn byte_bits(v: uint8x8_t) -> u64 {
    lane_bits_8(vtst_u8(v, v))
}

/// The keep mask of the 16 bytes of `v`, bit `j` set when byte `j` is nonzero.
#[inline]
#[target_feature(enable = "neon")]
fn byte_bits_16(v: uint8x16_t) -> u64 {
    lane_bits_16(vtstq_u8(v, v))
}

/// The bits of the 64-bit compare results `m`, bit `2 * i + j` set when lane `j` of `m[i]` is
/// all ones.
#[inline]
#[target_feature(enable = "neon")]
fn bits_64(m: [uint64x2_t; 4]) -> u64 {
    // Each step keeps the low half of every lane of two vectors, in order, which narrows
    // the all ones or zero lanes to bytes without a transfer per vector.
    let half = |a: uint64x2_t, b: uint64x2_t| {
        vuzp1q_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b))
    };
    let (lo, hi) = (half(m[0], m[1]), half(m[2], m[3]));
    let words = vuzp1q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi));
    lane_bits_8(vmovn_u16(words))
}

/// The bits of the 32-bit compare results `m`, bit `4 * i + j` set when lane `j` of `m[i]` is
/// all ones.
#[inline]
#[target_feature(enable = "neon")]
fn bits_32(m: [uint32x4_t; 4]) -> u64 {
    // As in `bits_64`, each step keeps the low half of every lane of two vectors, in order.
    let half = |a: uint32x4_t, b: uint32x4_t| {
        vuzp1q_u16(vreinterpretq_u16_u32(a), vreinterpretq_u16_u32(b))
    };
    let (lo, hi) = (half(m[0], m[1]), half(m[2], m[3]));
    lane_bits_16(vuzp1q_u8(
        vreinterpretq_u8_u16(lo),
        vreinterpretq_u8_u16(hi),
    ))
}

/// The 16 bytes at `src` as a vector.
///
/// # Safety
///
/// `src` must be valid for reading 16 bytes, and the CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_bytes<T>(src: *const T) -> uint8x16_t {
    // SAFETY: the caller guarantees 16 readable bytes at `src`, and `ld1` of bytes has no
    // alignment requirement.
    unsafe { vld1q_u8(src.cast()) }
}

/// The 64 bytes at `src` as four vectors, in order.
///
/// # Safety
///
/// `src` must be valid for reading 64 bytes, and the CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_quad<T>(src: *const T) -> uint8x16x4_t {
    // SAFETY: the caller guarantees 64 readable bytes at `src`, and `ld1` of bytes has no
    // alignment requirement.
    unsafe { vld1q_u8_x4(src.cast()) }
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

/// Stores the kept lanes of the four vectors `v`, in order, to the front of the 64 bytes at
/// `dst`, and returns their count. `keep` has one bit per lane, the lanes of `v.0` first,
/// and bits at `4 * PER` and above are ignored. `PER` is the number of lanes of one vector,
/// 2 or 4, and `table` the shuffles of that lane width.
///
/// # Safety
///
/// `dst` must be valid for writing 64 bytes, and the CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn compress_quad<T, const PER: usize>(
    v: uint8x16x4_t,
    keep: u64,
    table: &[[u8; 16]],
    dst: *mut T,
) -> usize {
    // The survivors before each vector. With 2 lanes per vector one `PREFIX` entry counts
    // all 8 bits. With 4 lanes the low byte counts the first two vectors and the high byte
    // the last two, which start after all survivors of the low byte.
    // The steps below are plain code and `#[inline(always)]` helpers, no closures or
    // `array::map`: the compiler once left an `array::map` here as a call in the loop, with
    // the vectors spilled around it, which made the step slower than one vector at a time.
    let (at1, at2, at3, total) = if PER == 2 {
        let p = prefix_entry(keep);
        (
            prefix_byte(p, 0),
            prefix_byte(p, 1),
            prefix_byte(p, 2),
            prefix_byte(p, 3),
        )
    } else {
        let (lo, hi) = (prefix_entry(keep), prefix_entry(keep >> 8));
        let low = prefix_byte(lo, 3);
        (
            prefix_byte(lo, 1),
            low,
            low + prefix_byte(hi, 1),
            low + prefix_byte(hi, 3),
        )
    };
    let field = (1u64 << PER) - 1;
    // Each store writes a whole vector of `PER` rows at the survivors before it, which are
    // at most the rows of the vectors before it, so it ends inside the `4 * PER` rows at
    // `dst`. A later store overwrites the unspecified tail of the one before.
    // SAFETY: by the bound above every store stays inside the 64 writable bytes at `dst`
    // that the caller guarantees, and the CPU supports NEON by the caller's guarantee.
    unsafe {
        shuffle_store(v.0, lane_shuffle(table, keep & field), dst);
        shuffle_store(v.1, lane_shuffle(table, keep >> PER & field), dst.add(at1));
        shuffle_store(
            v.2,
            lane_shuffle(table, keep >> (2 * PER) & field),
            dst.add(at2),
        );
        shuffle_store(
            v.3,
            lane_shuffle(table, keep >> (3 * PER) & field),
            dst.add(at3),
        );
    }
    total
}

/// A row type that NEON selects `LANES` rows, a `Vector` of one or more 128-bit registers, at
/// a time.
///
/// The methods are `unsafe` because they read and write through raw pointers and need NEON.
/// Each states the memory its pointers must cover.
pub(super) trait Lanes: Elem {
    const LANES: usize;

    /// `LANES` rows in registers.
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

/// A row type whose comparisons NEON evaluates `LANES` rows at a time.
pub(super) trait Compare: Lanes + PartialOrd {
    /// The mask of the lanes of `v` with `v[j] <cmp> c`, for comparison number `C` of `CMPS`,
    /// with the predicate of `Cmp::test`.
    unsafe fn compare<const C: usize>(v: Self::Vector, c: Self) -> u64;
}

// The 64-bit and 32-bit row types take four vectors, 64 bytes, per step. A step pays for
// one transfer of the keep mask to a general register and one loop iteration, so with one
// vector of 2 or 4 lanes that fixed cost dominated and the 64-bit rows were slower than the
// compiler's scalar loop. Four vectors spread it over 8 or 16 rows, and 64 bytes are the
// spare rows that `append_chunks` reserves past the survivors.
// SAFETY (all `Lanes` methods below): each method is compiled for, and needs, NEON, which
// its callers in this file have because they are compiled for the same feature. Pointer
// requirements are the trait's, and the unsafe blocks rely on exactly those.
macro_rules! lanes {
    ($t:ty, $lanes:literal, $per:literal, $table:ident) => {
        impl Lanes for $t {
            const LANES: usize = $lanes;
            type Vector = uint8x16x4_t;

            // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::keep`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn keep(mask: *const u8) -> u64 {
                if $lanes == 8 {
                    // SAFETY: the caller guarantees `LANES` (8) readable bytes at `mask`, the
                    // size of the word read here, and `read_unaligned` has no alignment
                    // requirement.
                    let word = unsafe { mask.cast::<u64>().read_unaligned() };
                    byte_bits(vcreate_u8(word))
                } else {
                    // SAFETY: the caller guarantees `LANES` (16) readable bytes at `mask`.
                    byte_bits_16(unsafe { load_bytes(mask) })
                }
            }

            // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::load`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn load(src: *const $t) -> uint8x16x4_t {
                // SAFETY: the caller guarantees `LANES` readable rows (64 bytes) at `src`.
                unsafe { load_quad(src) }
            }

            // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::compress`
            // states. Nothing assumes more than element alignment.
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn compress(v: uint8x16x4_t, keep: u64, dst: *mut $t) -> usize {
                // The shuffles move bytes, so float rows keep their exact bits.
                // SAFETY: the caller guarantees `LANES` writable rows (64 bytes) at `dst`.
                unsafe { compress_quad::<$t, $per>(v, keep, &$table, dst) }
            }
        }
    };
}

lanes!(f64, 8, 2, SHUFFLE_64);
lanes!(i64, 8, 2, SHUFFLE_64);
lanes!(f32, 16, 4, SHUFFLE_32);
lanes!(i32, 16, 4, SHUFFLE_32);

impl Lanes for u8 {
    const LANES: usize = 16;
    type Vector = uint8x16_t;

    // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::keep` states.
    // Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn keep(mask: *const u8) -> u64 {
        // SAFETY: the caller guarantees 16 readable bytes at `mask`.
        byte_bits_16(unsafe { load_bytes(mask) })
    }

    // SAFETY: the caller guarantees NEON and the pointer ranges that `Lanes::load` states.
    // Nothing assumes more than element alignment.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(src: *const u8) -> uint8x16_t {
        // SAFETY: the caller guarantees 16 readable bytes at `src`.
        unsafe { load_bytes(src) }
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

/// Applies `$cmp` with comparison number `$c` of `CMPS` to each of the four vectors of `$v`
/// against the splat `$s`, reinterpreted by `$as`, and gathers the lane bits with `$bits`.
/// `!=` is the negation of `==`, which is true on NaN like Rust's `!=`.
macro_rules! compare_quad {
    ($c:ident, $v:ident, $s:ident, $as:ident, $bits:ident, $all:literal,
     $eq:ident, $lt:ident, $le:ident, $gt:ident, $ge:ident) => {{
        let one = |x: uint8x16_t| {
            let x = $as(x);
            match CMPS[$c].1 {
                Cmp::Eq | Cmp::Ne => $eq(x, $s),
                Cmp::Lt => $lt(x, $s),
                Cmp::Le => $le(x, $s),
                Cmp::Gt => $gt(x, $s),
                Cmp::Ge => $ge(x, $s),
            }
        };
        let bits = $bits([one($v.0), one($v.1), one($v.2), one($v.3)]);
        match CMPS[$c].1 {
            Cmp::Ne => bits ^ $all,
            _ => bits,
        }
    }};
}

impl Compare for f64 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16x4_t, c: f64) -> u64 {
        let s = vdupq_n_f64(c);
        // The NEON float compares are ordered, false on NaN like `<`. `!=` is the negation
        // of `==`, which is true on NaN like Rust's `!=`.
        compare_quad!(
            C,
            v,
            s,
            vreinterpretq_f64_u8,
            bits_64,
            0xff,
            vceqq_f64,
            vcltq_f64,
            vcleq_f64,
            vcgtq_f64,
            vcgeq_f64
        )
    }
}

impl Compare for f32 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16x4_t, c: f32) -> u64 {
        let s = vdupq_n_f32(c);
        // The predicates of the f64 case.
        compare_quad!(
            C,
            v,
            s,
            vreinterpretq_f32_u8,
            bits_32,
            0xffff,
            vceqq_f32,
            vcltq_f32,
            vcleq_f32,
            vcgtq_f32,
            vcgeq_f32
        )
    }
}

impl Compare for i64 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16x4_t, c: i64) -> u64 {
        let s = vdupq_n_s64(c);
        // The signed compares, and `!=` as the negation of `==`.
        compare_quad!(
            C,
            v,
            s,
            vreinterpretq_s64_u8,
            bits_64,
            0xff,
            vceqq_s64,
            vcltq_s64,
            vcleq_s64,
            vcgtq_s64,
            vcgeq_s64
        )
    }
}

impl Compare for i32 {
    // SAFETY: the caller guarantees NEON. The method reads no memory.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare<const C: usize>(v: uint8x16x4_t, c: i32) -> u64 {
        let s = vdupq_n_s32(c);
        // As for i64.
        compare_quad!(
            C,
            v,
            s,
            vreinterpretq_s32_u8,
            bits_32,
            0xffff,
            vceqq_s32,
            vcltq_s32,
            vcleq_s32,
            vcgtq_s32,
            vcgeq_s32
        )
    }
}

/// The `i64` rows `first + j` of lanes `j`, see `selection_loops`.
#[inline]
#[target_feature(enable = "neon")]
fn first_indices(first: i64) -> uint8x16x4_t {
    let at = |j: i64| {
        let lanes = vcombine_s64(vcreate_s64(0), vcreate_s64(1));
        vreinterpretq_u8_s64(vaddq_s64(vdupq_n_s64(first.wrapping_add(j)), lanes))
    };
    uint8x16x4_t(at(0), at(2), at(4), at(6))
}

/// `idx` advanced by the 8 `i64` lanes of a step, see `selection_loops`.
#[inline]
#[target_feature(enable = "neon")]
fn next_indices(idx: uint8x16x4_t) -> uint8x16x4_t {
    let step =
        |v: uint8x16_t| vreinterpretq_u8_s64(vaddq_s64(vreinterpretq_s64_u8(v), vdupq_n_s64(8)));
    uint8x16x4_t(step(idx.0), step(idx.1), step(idx.2), step(idx.3))
}

selection_loops!("NEON", "neon");

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
    fn prefix_table_counts_low_bits() {
        for (m, &p) in PREFIX.iter().enumerate() {
            for j in 0..4 {
                let low = m & ((4 << (2 * j)) - 1);
                assert_eq!(
                    prefix_byte(p, j),
                    ones(u64::try_from(low).unwrap()),
                    "{m:#b}"
                );
            }
        }
    }

    #[test]
    fn shuffle_tables_pack_kept_lanes() {
        check_table(&SHUFFLE_64, 8);
        check_table(&SHUFFLE_32, 4);
        check_table(&SHUFFLE_8, 1);
    }
}
