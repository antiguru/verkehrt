//! The intrinsics backend of the selections: `compress_*`, `where`, and the fused
//! `filter_*_*`, hand-written for AVX2 and AVX-512 beside their auto entries.
//!
//! Each entry parses its arguments with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails. It counts the survivors of the tile first, then appends through
//! `append_chunks`, which reserves exactly that many rows. Within a chunk it selects a vector
//! of rows at a time while a whole vector store fits in the chunk's spare rows, and finishes
//! the rest with the portable predicated loop.

use crate::kernels::SelectionFn;
use crate::value::{Elem, ScalarType};

mod avx2;
mod avx512;

/// The AVX2 and the AVX-512 entry of a kernel, `None` where it has none.
type Entries = (Option<SelectionFn>, Option<SelectionFn>);

/// The AVX2 and AVX-512 entries of `compress` on rows of storage type `T`, whose semantic
/// type is `tag::<T, BOOL>()`, where they exist.
pub(crate) fn compress_entries<T: Elem, const BOOL: bool>() -> Entries {
    // Each storage type has its own `ScalarType`, so an arm instantiates the entries for
    // exactly `T`. Only `u8` is instantiated with `BOOL` set.
    match T::TYPE {
        ScalarType::F64 => (
            Some(avx2::compress::<f64, BOOL>),
            Some(avx512::compress::<f64, BOOL>),
        ),
        ScalarType::F32 => (
            Some(avx2::compress::<f32, BOOL>),
            Some(avx512::compress::<f32, BOOL>),
        ),
        ScalarType::I64 => (
            Some(avx2::compress::<i64, BOOL>),
            Some(avx512::compress::<i64, BOOL>),
        ),
        ScalarType::I32 => (
            Some(avx2::compress::<i32, BOOL>),
            Some(avx512::compress::<i32, BOOL>),
        ),
        // AVX2 has no byte compress, so `u8` and bool keep the auto entry there.
        ScalarType::U8 => (None, Some(avx512::compress::<u8, BOOL>)),
        _ => (None, None),
    }
}

/// The AVX2 and AVX-512 entries of `where`.
pub(crate) fn where_entries() -> (SelectionFn, SelectionFn) {
    (avx2::r#where, avx512::r#where)
}

/// The AVX2 and AVX-512 entries of `filter_<cmp>` on rows of type `T`, for comparison
/// number `C` of `CMPS`, where they exist.
pub(crate) fn filter_entries<T: Elem, const C: usize>() -> Entries {
    // As in `compress_entries`, an arm instantiates the entries for exactly `T`.
    match T::TYPE {
        ScalarType::F64 => (Some(avx2::filter::<f64, C>), Some(avx512::filter::<f64, C>)),
        ScalarType::F32 => (Some(avx2::filter::<f32, C>), Some(avx512::filter::<f32, C>)),
        ScalarType::I64 => (Some(avx2::filter::<i64, C>), Some(avx512::filter::<i64, C>)),
        ScalarType::I32 => (Some(avx2::filter::<i32, C>), Some(avx512::filter::<i32, C>)),
        _ => (None, None),
    }
}

/// The number of set bits, as a row count.
#[inline(always)]
fn ones(bits: u64) -> usize {
    usize::try_from(bits.count_ones()).expect("a bit count fits in usize")
}
