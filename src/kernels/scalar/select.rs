//! Scalar reference selection kernels. Output length depends on the data, so the slice
//! functions append to a growing output and a tiled executor calls them once per tile.

use crate::kernels::scalar::elementwise::{arity_error, slice_of};
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::{InputSpec, KernelClass, KernelDef, Registry};
use crate::ops::Op;
use crate::value::{Array, Elem, ScalarType, Value, ValueType};

/// Appends the rows of `x` whose `mask` byte is nonzero. Both slices must have the same length.
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

fn run_compress<T: Elem>(args: &[Value], ty: ScalarType) -> Result<Value, String> {
    let [x, mask] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, mask) = (array_arg(x, ty)?, array_arg(mask, ScalarType::Bool)?);
    if x.len() != mask.len() {
        return Err(format!(
            "length mismatch: {} values vs {} mask rows",
            x.len(),
            mask.len()
        ));
    }
    let mut out = Vec::new();
    compress_append(slice_of::<T>(x), mask.as_bool_bytes(), &mut out);
    Ok(Value::Array(array_of(out, ty)))
}

fn run_where(args: &[Value]) -> Result<Value, String> {
    let [mask] = args else {
        return Err(arity_error(1, args.len()));
    };
    let mask = array_arg(mask, ScalarType::Bool)?;
    let mut out = Vec::new();
    where_append(mask.as_bool_bytes(), 0, &mut out);
    Ok(Value::Array(Array::from_vec(out)))
}

/// Registers `compress_{name}`. The closure needs constant arguments, see `reduce.rs`.
macro_rules! compress {
    ($reg:expr, $t:ty, $ty:expr) => {
        $reg.add(KernelDef {
            name: format!("compress_{}", $ty.name()),
            class: KernelClass::Selection,
            op: Some(Op::Compress),
            inputs: vec![InputSpec::Array($ty), InputSpec::Array(ScalarType::Bool)],
            output: ValueType::Array($ty),
            // Read the value and the mask byte, write the survivors, plus a byte of
            // branch and compaction overhead per row.
            cost: 2 * u64::try_from($ty.width()).expect("width fits in u64") + 1,
            commutative: false,
            pattern: None,
            impls: [Some(|args| run_compress::<$t>(args, $ty)), None, None, None],
        })
    };
}

pub(crate) fn register(reg: &mut Registry) {
    compress!(reg, f64, ScalarType::F64);
    compress!(reg, f32, ScalarType::F32);
    compress!(reg, i64, ScalarType::I64);
    compress!(reg, i32, ScalarType::I32);
    compress!(reg, u8, ScalarType::U8);
    // Bool arrays are stored as `u8`.
    compress!(reg, u8, ScalarType::Bool);
    reg.add(KernelDef {
        name: "where".into(),
        class: KernelClass::Selection,
        op: Some(Op::Where),
        inputs: vec![InputSpec::Array(ScalarType::Bool)],
        output: ValueType::Array(ScalarType::I64),
        cost: 9,
        commutative: false,
        pattern: None,
        impls: [Some(run_where), None, None, None],
    });
}
