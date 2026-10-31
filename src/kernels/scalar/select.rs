//! Scalar reference selection kernels. Output length depends on the data, so the slice
//! functions append to a growing output and a tiled executor calls them once per tile.

use std::marker::PhantomData;
use std::ops::Range;

use crate::kernels::scalar::elementwise::{arity_error, rows_arg, tag};
use crate::kernels::{
    InputSpec, KernelClass, KernelDef, KernelImpl, Registry, SelectionBody, Variant,
};
use crate::ops::Op;
use crate::value::{Arg, Array, ArrayBuilder, Elem, ScalarType, ValueType};

/// Rows per reservation of `append_chunks`. A chunk's spare rows stay in cache between the
/// zero fill of the builder and the selection's writes.
pub(crate) const CHUNK: usize = 2048;

/// Rows of `T` in 64 bytes, the widest vector store of any selection entry.
pub(crate) const fn vector_rows<T>() -> usize {
    64 / size_of::<T>()
}

/// Appends the survivors of `n` rows to `out`, `CHUNK` rows at a time, where `kept` is the
/// number of survivors among the `n` rows. `chunk(rows, dst)` selects from `rows`, writes
/// the survivors in order to the front of `dst`, and returns their count. It may write any
/// value to the rest of `dst`. `dst` has `min(rows.len(), r + vector_rows::<T>())` rows,
/// with `r` the survivors still to come, so a predicated write of every row and a full
/// vector store at any survivor count stay inside it.
///
/// The output buffer is reserved once for exactly the rows it can reach, so a selection
/// into an empty builder neither grows by reallocation nor leaves more than one vector plus
/// the rounding up to whole blocks of unused capacity. Panics when a chunk returns more
/// survivors than `dst` holds, which a wrong `kept` can cause.
#[inline(always)]
pub(crate) fn append_chunks<T: Elem>(
    out: &mut ArrayBuilder,
    n: usize,
    kept: usize,
    mut chunk: impl FnMut(Range<usize>, &mut [T]) -> usize,
) {
    // The spare rows of a chunk end at most one vector past the last survivor, and never
    // past the rows of the tile. With only one row past it, a chunk after the survivors ran
    // out would leave the vector loops of the entries for their predicated scalar loops.
    let slack = vector_rows::<T>();
    out.reserve(kept.saturating_add(slack).min(n));
    let (mut start, mut done) = (0, 0);
    while start < n {
        let rows = CHUNK.min(n - start);
        let room = rows.min(kept.saturating_sub(done).saturating_add(slack));
        let dst = out.spare_rows::<T>(room);
        let got = chunk(start..start + rows, dst);
        out.commit(got);
        done += got;
        start += rows;
    }
}

/// The number of nonzero bytes of `mask`, the survivors of `compress` and `where`.
#[inline(always)]
pub(crate) fn count_nonzero(mask: &[u8]) -> usize {
    mask.iter().filter(|&&m| m != 0).count()
}

/// Writes every row of `rows` to `dst` at the survivor count so far, which then advances by
/// the row's keep flag, and returns the survivor count. There is no branch on the data, so
/// the time does not depend on the selectivity. `dst` needs a slot per survivor, plus one
/// when a row that is not kept follows the last survivor, and panics otherwise.
#[inline(always)]
pub(crate) fn predicated<T: Copy>(rows: impl Iterator<Item = (T, bool)>, dst: &mut [T]) -> usize {
    let mut kept = 0;
    for (v, keep) in rows {
        dst[kept] = v;
        kept += usize::from(keep);
    }
    kept
}

/// Appends the rows of `x` whose `mask` byte is nonzero. Both slices must have the same length.
#[inline(always)]
pub fn compress_append<T: Elem>(x: &[T], mask: &[u8], out: &mut ArrayBuilder) {
    assert_eq!(x.len(), mask.len(), "mask rows must match the data rows");
    append_chunks(out, x.len(), count_nonzero(mask), |rows, dst| {
        predicated(
            x[rows.clone()]
                .iter()
                .zip(&mask[rows])
                .map(|(&v, &m)| (v, m != 0)),
            dst,
        )
    });
}

/// Appends `base + i` for every row `i` of `mask` whose byte is nonzero. `base` is the
/// index of the first row of this tile in the whole array. Panics when `base + mask.len()`
/// exceeds `i64::MAX`, even if no row is kept.
#[inline(always)]
pub fn where_append(mask: &[u8], base: usize, out: &mut ArrayBuilder) {
    check_row_indices(base, mask.len());
    append_chunks(out, mask.len(), count_nonzero(mask), |rows, dst| {
        let first = i64::try_from(base + rows.start).expect("row index fits in i64");
        predicated((first..).zip(&mask[rows]).map(|(i, &m)| (i, m != 0)), dst)
    });
}

/// Panics unless the row indices `base..base + len` all fit in i64. Every row index of a
/// tile fits when the end does, so a `where` entry counts up from `base` without a check per
/// row.
pub(crate) fn check_row_indices(base: usize, len: usize) {
    base.checked_add(len)
        .and_then(|end| i64::try_from(end).ok())
        .expect("row index fits in i64");
}

/// An array of type `ty` from its storage rows. `T` is the storage type of `ty`.
pub(crate) fn array_of<T: Elem>(rows: Vec<T>, ty: ScalarType) -> Array {
    if ty == ScalarType::Bool {
        Array::build_bool(rows.len(), |o| {
            o.copy_from_slice(bytemuck::cast_slice(&rows))
        })
    } else {
        Array::from_vec(rows)
    }
}

/// The rows and the mask of a `compress` tile on rows of type `ty`, whose storage type is
/// `T`. Fails on a wrong arity, wrong argument kinds, or row counts that differ, so the two
/// slices it returns have the same length.
#[inline(always)]
pub(crate) fn compress_args<T: Elem>(
    args: &[Arg],
    ty: ScalarType,
) -> Result<(&[T], &[u8]), String> {
    let [x, mask] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, mask) = (
        rows_arg::<T>(x, ty)?,
        rows_arg::<u8>(mask, ScalarType::Bool)?,
    );
    if x.len() != mask.len() {
        return Err(format!(
            "length mismatch: {} values vs {} mask rows",
            x.len(),
            mask.len()
        ));
    }
    Ok((x, mask))
}

/// The `compress` entry for rows of type `ty`, whose storage type is `T`. The tile's base
/// does not matter because the survivors are rows, not indices.
#[inline(always)]
fn compress_tile<T: Elem>(
    args: &[Arg],
    out: &mut ArrayBuilder,
    ty: ScalarType,
) -> Result<(), String> {
    let (x, mask) = compress_args::<T>(args, ty)?;
    compress_append(x, mask, out);
    Ok(())
}

/// The mask of a `where` tile. Fails on a wrong arity or argument kind.
#[inline(always)]
pub(crate) fn where_args(args: &[Arg]) -> Result<&[u8], String> {
    let [mask] = args else {
        return Err(arity_error(1, args.len()));
    };
    rows_arg::<u8>(mask, ScalarType::Bool)
}

/// The `where` entry. `base` makes the tile's row indices global.
#[inline(always)]
fn where_tile(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String> {
    where_append(where_args(args)?, base, out);
    Ok(())
}

/// The body of `compress` on rows of storage type `T`, whose semantic type is
/// `tag::<T, BOOL>()`.
struct Compress<T, const BOOL: bool>(PhantomData<fn() -> T>);

impl<T: Elem, const BOOL: bool> SelectionBody for Compress<T, BOOL> {
    #[inline(always)]
    fn run(args: &[Arg], _base: usize, out: &mut ArrayBuilder) -> Result<(), String> {
        compress_tile::<T>(args, out, tag::<T, BOOL>())
    }
}

/// The body of `where`.
struct Where;

impl SelectionBody for Where {
    #[inline(always)]
    fn run(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String> {
        where_tile(args, base, out)
    }
}

/// Registers `compress_{name}` for rows of storage type `T` and semantic type
/// `tag::<T, BOOL>()`.
fn compress<T: Elem, const BOOL: bool>(reg: &mut Registry) {
    let ty = tag::<T, BOOL>();
    reg.add(KernelDef {
        name: format!("compress_{}", ty.name()),
        class: KernelClass::Selection,
        op: Some(Op::Compress),
        inputs: vec![InputSpec::Array(ty), InputSpec::Array(ScalarType::Bool)],
        output: ValueType::Array(ty),
        // Read the value and the mask byte, write the survivors, plus a byte of
        // branch and compaction overhead per row.
        cost: 2 * u64::try_from(ty.width()).expect("width fits in u64") + 1,
        commutative: false,
        pattern: None,
        impls: compress_impls::<T, BOOL>(),
    });
}

/// The entries of `compress` on rows of storage type `T`: the scalar and auto entries of its
/// body, plus intrinsics entries where they exist.
fn compress_impls<T: Elem, const BOOL: bool>() -> [Option<KernelImpl>; Variant::COUNT] {
    let mut impls = KernelImpl::selection::<Compress<T, BOOL>>();
    crate::kernels::arch::add_compress::<T, BOOL>(&mut impls);
    impls
}

/// The entries of `where`, see `compress_impls`.
fn where_impls() -> [Option<KernelImpl>; Variant::COUNT] {
    let mut impls = KernelImpl::selection::<Where>();
    crate::kernels::arch::add_where(&mut impls);
    impls
}

pub(crate) fn register(reg: &mut Registry) {
    compress::<f64, false>(reg);
    compress::<f32, false>(reg);
    compress::<i64, false>(reg);
    compress::<i32, false>(reg);
    compress::<u8, false>(reg);
    // Bool arrays are stored as `u8`.
    compress::<u8, true>(reg);
    reg.add(KernelDef {
        name: "where".into(),
        class: KernelClass::Selection,
        op: Some(Op::Where),
        inputs: vec![InputSpec::Array(ScalarType::Bool)],
        output: ValueType::Array(ScalarType::I64),
        cost: 9,
        commutative: false,
        pattern: None,
        impls: where_impls(),
    });
}
