//! The x86-64-v4 intrinsics entry of `scan_by_add` on 64-bit integer rows, a segmented
//! log-step prefix sum.
//!
//! Each vector of eight rows takes three steps. Step `d` (1, 2, 4) adds the lane `d` below to
//! every lane that has no segment start in its last `d` lanes, then ORs the start flags `d`
//! lanes up. The lanes before the vector's first start then add the carry, the output of the
//! previous row. The scalar loop instead chains a select and an add through every row. The
//! sum wraps, so the order of the additions does not change the bits, and the entry returns
//! the scalar entry's result. AVX2 has no entry: its lane shifts cross the 128-bit halves,
//! and the prototype ran no faster than the scalar loop.

use std::arch::x86_64::*;
use std::marker::PhantomData;

use crate::kernels::scalar::scan::tile_rows;
use crate::kernels::{ScanFn, Scanner};
use crate::value::{Arg, Elem, ScalarType};

/// The x86-64-v4 entry of `scan_by_add` on rows of storage type `T`, `None` where it has none.
pub(crate) fn scan_by_add_entry<T: Elem>() -> Option<ScanFn> {
    matches!(T::TYPE, ScalarType::I64 | ScalarType::U64).then_some(scan_by_add::<T> as ScanFn)
}

/// The state of `scan_by_add` on 64-bit integer rows of type `T`, whose semantic type is
/// `T::TYPE`. Only `scan_by_add` creates one.
struct SegmentedSum<T> {
    /// The bits of the output of the last row of the tiles consumed so far, `None` before
    /// the first row.
    acc: Option<u64>,
    _rows: PhantomData<fn() -> T>,
}

/// The scanner constructor.
// SAFETY: the caller guarantees the features of the x86-64-v4 level (`Dispatch` calls the
// entry only after `resolve` found `features::runnable` true for it through `select`), and
// every `update` of the returned scanner relies on that.
unsafe fn scan_by_add<T: Elem>() -> Box<dyn Scanner> {
    Box::new(SegmentedSum::<T> {
        acc: None,
        _rows: PhantomData,
    })
}

impl<T: Elem> Scanner for SegmentedSum<T> {
    fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        // SAFETY: a `SegmentedSum` comes only from the unsafe `scan_by_add`, whose caller
        // guarantees the features of the x86-64-v4 level. `update_avx512` checks the arguments
        // and makes no pointer assumptions beyond those its own unsafe block states.
        unsafe { update_avx512::<T>(self, args, out) }
    }
}

/// One tile of `scan_by_add`.
// SAFETY: the caller guarantees the features of the x86-64-v4 level, the features the
// `target_feature` attribute enables. The pointer assumptions are stated at the unsafe block.
#[target_feature(
    enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
unsafe fn update_avx512<T: Elem>(
    s: &mut SegmentedSum<T>,
    args: &[Arg],
    out: &mut [u8],
) -> Result<(), String> {
    let (xs, out, starts) = tile_rows::<T, T, true>(args, out, T::TYPE)?;
    let Some(starts) = starts else {
        return Err("scan_by: missing segment starts".to_string());
    };
    // The rows are 64-bit integers, and a wrapping sum of their bits is their sum.
    let xs: &[u64] = bytemuck::cast_slice(xs);
    let out: &mut [u64] = bytemuck::cast_slice_mut(out);
    // Row 0 of the array lifts its value, which for a sum is the value added to zero, so a
    // zero carry needs no forced start.
    let mut acc = s.acc.unwrap_or(0);
    let (xc, xt) = xs.as_chunks::<8>();
    let (sc, st) = starts.as_chunks::<8>();
    let (oc, ot) = out.as_chunks_mut::<8>();
    // `tile_rows` checked that the three have the same row count, so the chunks pair up.
    let zero = _mm512_setzero_si512();
    let lane7 = _mm512_set1_epi64(7);
    let mut carry = _mm512_set1_epi64(acc.cast_signed());
    for ((x, f), o) in xc.iter().zip(sc).zip(oc.iter_mut()) {
        // SAFETY: avx512f, avx512bw, and avx512vl are enabled for this function. The loads
        // read the eight start bytes of `f` and the eight rows of `x`, the store writes the
        // eight rows of `o`, all unaligned accesses that the intrinsics permit.
        unsafe {
            let fb = _mm_loadl_epi64(f.as_ptr().cast::<__m128i>());
            // The low eight bits flag the eight start bytes, the upper eight bytes are zero.
            let mut m = _mm_test_epi8_mask(fb, fb) as u8;
            let mut v = _mm512_loadu_si512(x.as_ptr().cast());
            v = _mm512_mask_add_epi64(v, !m, v, _mm512_alignr_epi64::<7>(v, zero));
            m |= m << 1;
            v = _mm512_mask_add_epi64(v, !m, v, _mm512_alignr_epi64::<6>(v, zero));
            m |= m << 2;
            v = _mm512_mask_add_epi64(v, !m, v, _mm512_alignr_epi64::<4>(v, zero));
            m |= m << 4;
            v = _mm512_mask_add_epi64(v, !m, v, carry);
            _mm512_storeu_si512(o.as_mut_ptr().cast(), v);
            carry = _mm512_permutexvar_epi64(lane7, v);
        }
    }
    if let Some(o) = oc.last() {
        acc = o[7];
    }
    for ((&x, &f), o) in xt.iter().zip(st).zip(ot.iter_mut()) {
        acc = if f != 0 { x } else { acc.wrapping_add(x) };
        *o = acc;
    }
    if !xs.is_empty() {
        s.acc = Some(acc);
    }
    Ok(())
}
