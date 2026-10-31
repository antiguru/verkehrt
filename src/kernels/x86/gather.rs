//! The intrinsics backend of `gather` on 64-bit rows, hand-written for the x86-64-v3 level
//! (AVX2) and the x86-64-v4 level (AVX-512) beside the auto entries.
//!
//! Each entry checks its arguments and every index with the scalar body's `Gather::checked`,
//! so it fails exactly when the scalar entry fails. After that check every index addresses a
//! source row, and the entry loads the rows with the hardware gather, four rows per
//! `vpgatherqq` on AVX2 and eight on AVX-512. The rows the vectors leave over are copied one
//! at a time. A gather moves the 64 bits of a row unchanged, so the entries serve every
//! 64-bit row type and return the scalar entry's bits.

use std::arch::x86_64::*;

use crate::kernels::WholeFn;
use crate::kernels::scalar::index::Gather;
use crate::value::{Array, Elem, Value};

/// The x86-64-v3 and the x86-64-v4 entry of `gather` on rows of storage type `T`, whose
/// semantic type is `tag::<T, BOOL>()`, `None` where it has none.
pub(crate) fn gather_entries<T: Elem, const BOOL: bool>() -> (Option<WholeFn>, Option<WholeFn>) {
    if size_of::<T>() == 8 && !BOOL {
        (Some(avx2::<T>), Some(avx512::<T>))
    } else {
        (None, None)
    }
}

/// The checked source rows and indices of a `gather` call on 64-bit rows of type `T`, the
/// rows as their bits.
#[inline(always)]
fn checked<T: Elem>(args: &[Value]) -> Result<(&[u64], &[i64]), String> {
    let (xs, idx) = Gather::<T, false>::checked(args)?;
    Ok((bytemuck::cast_slice(xs), idx))
}

/// Copies `xs[i]` for each index `i` of `idx` to `out`, one row at a time. Every index lies
/// in `0..xs.len()`.
#[inline(always)]
fn rest(xs: &[u64], idx: &[i64], out: &mut [u64]) {
    for (o, &i) in out.iter_mut().zip(idx) {
        *o = xs[usize::try_from(i).expect("a checked index")];
    }
}

/// The AVX2 entry of `gather` on 64-bit rows of type `T`.
// SAFETY: the caller guarantees the features of the x86-64-v3 level (`Dispatch` calls the
// entry only after `resolve` found `features::runnable` true for it through `select`). The
// pointer assumptions are stated at the unsafe block.
#[target_feature(
    enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3"
)]
unsafe fn avx2<T: Elem>(args: &[Value]) -> Result<Value, String> {
    let (xs, idx) = checked::<T>(args)?;
    Array::build_bytes(idx.len(), T::TYPE, |out| {
        let out: &mut [u64] = bytemuck::cast_slice_mut(out);
        let (ic, it) = idx.as_chunks::<4>();
        let (oc, ot) = out.as_chunks_mut::<4>();
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            // SAFETY: avx2 is enabled for this function. The load reads the four indices of
            // `i` and the store writes the four rows of `o`, both unaligned accesses that
            // the intrinsics permit. `checked` verified every index to lie in
            // `0..xs.len()`, so each gathered address `xs + 8 * i` is a row of `xs`.
            unsafe {
                let v = _mm256_loadu_si256(i.as_ptr().cast::<__m256i>());
                let g = _mm256_i64gather_epi64::<8>(xs.as_ptr().cast::<i64>(), v);
                _mm256_storeu_si256(o.as_mut_ptr().cast::<__m256i>(), g);
            }
        }
        rest(xs, it, ot);
        Ok(())
    })
    .map(Value::Array)
}

/// The AVX-512 entry of `gather` on 64-bit rows of type `T`.
// SAFETY: the caller guarantees the features of the x86-64-v4 level (`Dispatch` calls the
// entry only after `resolve` found `features::runnable` true for it through `select`). The
// pointer assumptions are stated at the unsafe block.
#[target_feature(
    enable = "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
unsafe fn avx512<T: Elem>(args: &[Value]) -> Result<Value, String> {
    let (xs, idx) = checked::<T>(args)?;
    Array::build_bytes(idx.len(), T::TYPE, |out| {
        let out: &mut [u64] = bytemuck::cast_slice_mut(out);
        let (ic, it) = idx.as_chunks::<8>();
        let (oc, ot) = out.as_chunks_mut::<8>();
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            // SAFETY: avx512f is enabled for this function. The load reads the eight indices
            // of `i` and the store writes the eight rows of `o`, both unaligned accesses that
            // the intrinsics permit. `checked` verified every index to lie in
            // `0..xs.len()`, so each gathered address `xs + 8 * i` is a row of `xs`.
            unsafe {
                let v = _mm512_loadu_si512(i.as_ptr().cast());
                let g = _mm512_i64gather_epi64::<8>(v, xs.as_ptr().cast::<i64>());
                _mm512_storeu_si512(o.as_mut_ptr().cast(), g);
            }
        }
        rest(xs, it, ot);
        Ok(())
    })
    .map(Value::Array)
}
