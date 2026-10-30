//! Scalar reference selection kernels. Output length depends on the data, so the slice
//! functions append to a growing output and a tiled executor calls them once per tile.

use std::marker::PhantomData;

use crate::kernels::scalar::elementwise::{arity_error, rows_arg, tag};
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, SelectionBody};
use crate::ops::Op;
use crate::value::{Arg, Array, ArrayBuilder, Elem, ScalarType, ValueType};

/// Appends the rows of `x` whose `mask` byte is nonzero. Both slices must have the same length.
#[inline(always)]
pub fn compress_append<T: Copy>(x: &[T], mask: &[u8], out: &mut Vec<T>) {
    assert_eq!(x.len(), mask.len(), "mask rows must match the data rows");
    out.extend(
        x.iter()
            .zip(mask)
            .filter(|&(_, &m)| m != 0)
            .map(|(&v, _)| v),
    );
}

/// Appends `base + i` for every row `i` of `mask` whose byte is nonzero. `base` is the
/// index of the first row of this tile in the whole array.
#[inline(always)]
pub fn where_append(mask: &[u8], base: usize, out: &mut Vec<i64>) {
    out.extend(
        mask.iter()
            .enumerate()
            .filter(|&(_, &m)| m != 0)
            .map(|(i, _)| i64::try_from(base + i).expect("row index fits in i64")),
    );
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

/// The `compress` entry for rows of type `ty`, whose storage type is `T`. The tile's base
/// does not matter because the survivors are rows, not indices.
#[inline(always)]
fn compress_tile<T: Elem>(
    args: &[Arg],
    out: &mut ArrayBuilder,
    ty: ScalarType,
) -> Result<(), String> {
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
    let mut kept = Vec::new();
    compress_append(x, mask, &mut kept);
    out.extend(&kept);
    Ok(())
}

/// The `where` entry. `base` makes the tile's row indices global.
#[inline(always)]
fn where_tile(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String> {
    let [mask] = args else {
        return Err(arity_error(1, args.len()));
    };
    let mask = rows_arg::<u8>(mask, ScalarType::Bool)?;
    let mut kept = Vec::new();
    where_append(mask, base, &mut kept);
    out.extend(&kept);
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
        impls: KernelImpl::selection::<Compress<T, BOOL>>(),
    });
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
        impls: KernelImpl::selection::<Where>(),
    });
}
