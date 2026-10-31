//! The intrinsics backend of the selections: `compress_*`, `where`, and the fused
//! `filter_*_*`, hand-written for the x86-64-v3 level (AVX2) and the x86-64-v4 level
//! (AVX-512) beside their auto entries. The x86-64-v4 byte `compress` additionally needs
//! VBMI2, so it lives in `avx512_vbmi2`, compiled for the v4 string plus `avx512vbmi2`, and
//! its entries declare that feature.
//!
//! Each entry parses its arguments with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails. It counts the survivors of the tile first, then appends through
//! `append_chunks`, which reserves that many rows plus one vector. Within a chunk it selects
//! a vector of rows at a time while a whole vector store fits in the chunk's spare rows, and
//! finishes the rest with the portable predicated loop. The loops and entries are shared
//! with the other ISAs, see `crate::kernels::select_loops`.

use crate::kernels::{FeatureSet, SelectionFn};
use crate::value::{Elem, ScalarType};

mod avx2;
mod avx512;
mod avx512_vbmi2;

/// The x86-64-v3 and the x86-64-v4 entry of a kernel, `None` where it has none.
type Entries = (Option<SelectionFn>, Option<SelectionFn>);

/// The x86-64-v3 and x86-64-v4 entries of `compress` on rows of storage type `T`, whose
/// semantic type is `tag::<T, BOOL>()`, where they exist. The x86-64-v4 entry comes with the
/// features it declares beyond its level.
pub(crate) fn compress_entries<T: Elem, const BOOL: bool>()
-> (Option<SelectionFn>, Option<(SelectionFn, FeatureSet)>) {
    // The 32-bit and 64-bit entries need nothing beyond the level.
    const NO_EXTRA: FeatureSet = FeatureSet::EMPTY;
    // Each storage type has its own `ScalarType`, so an arm instantiates the entries for
    // exactly `T`. Only `u8` is instantiated with `BOOL` set.
    match T::TYPE {
        ScalarType::F64 => (
            Some(avx2::compress::<f64, BOOL>),
            Some((avx512::compress::<f64, BOOL>, NO_EXTRA)),
        ),
        ScalarType::F32 => (
            Some(avx2::compress::<f32, BOOL>),
            Some((avx512::compress::<f32, BOOL>, NO_EXTRA)),
        ),
        ScalarType::I64 => (
            Some(avx2::compress::<i64, BOOL>),
            Some((avx512::compress::<i64, BOOL>, NO_EXTRA)),
        ),
        ScalarType::U64 => (
            Some(avx2::compress::<u64, BOOL>),
            Some((avx512::compress::<u64, BOOL>, NO_EXTRA)),
        ),
        ScalarType::I32 => (
            Some(avx2::compress::<i32, BOOL>),
            Some((avx512::compress::<i32, BOOL>, NO_EXTRA)),
        ),
        // AVX2 has no byte compress, so `u8` and bool keep the auto entry there. The AVX-512
        // byte compress is `vpcompressb`, a VBMI2 instruction, so its constructor pairs the
        // entry with the feature it declares.
        ScalarType::U8 => (None, Some(avx512_vbmi2::compress_entry::<BOOL>())),
        _ => (None, None),
    }
}

/// The x86-64-v3 and x86-64-v4 entries of `where`.
pub(crate) fn where_entries() -> (SelectionFn, SelectionFn) {
    (avx2::r#where, avx512::r#where)
}

/// The x86-64-v3 and x86-64-v4 entries of `filter_<cmp>` on rows of type `T`, for comparison
/// number `C` of `CMPS`, where they exist.
pub(crate) fn filter_entries<T: Elem, const C: usize>() -> Entries {
    // As in `compress_entries`, an arm instantiates the entries for exactly `T`.
    match T::TYPE {
        ScalarType::F64 => (Some(avx2::filter::<f64, C>), Some(avx512::filter::<f64, C>)),
        ScalarType::F32 => (Some(avx2::filter::<f32, C>), Some(avx512::filter::<f32, C>)),
        ScalarType::I64 => (Some(avx2::filter::<i64, C>), Some(avx512::filter::<i64, C>)),
        ScalarType::U64 => (Some(avx2::filter::<u64, C>), Some(avx512::filter::<u64, C>)),
        ScalarType::I32 => (Some(avx2::filter::<i32, C>), Some(avx512::filter::<i32, C>)),
        _ => (None, None),
    }
}
