//! Scalar reference scan kernels. A scan keeps state across the tiles of one array, so a tiled
//! executor feeds it tile by tile and the output never depends on where the tiles split.

use std::marker::PhantomData;

use crate::kernels::scalar::elementwise::{arity_error, out_rows, rows_arg, splat_arg, tag};
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, ScanBody, Scanner};
use crate::ops::Op;
use crate::value::{Arg, Elem, ValueType};

/// `prev(x, fill)`: row `i` is row `i - 1` of `x` and row 0 is `fill`. `T` is the storage type
/// of the rows and `BOOL` marks the bool instantiation on `u8`.
pub struct Prev<T, const BOOL: bool> {
    /// The last row of the tiles consumed so far, `None` before the first row.
    last: Option<T>,
    _rows: PhantomData<fn() -> T>,
}

impl<T: Elem, const BOOL: bool> Scanner for Prev<T, BOOL> {
    #[inline(always)]
    fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        let [x, fill] = args else {
            return Err(arity_error(2, args.len()));
        };
        let ty = tag::<T, BOOL>();
        let xs = rows_arg::<T>(x, ty)?;
        let fill = splat_arg::<T>(fill, ty)?;
        let out = out_rows::<T>(out)?;
        if out.len() != xs.len() {
            return Err(format!(
                "length mismatch: {} rows for a destination of {} rows",
                xs.len(),
                out.len()
            ));
        }
        let Some((first, rest)) = out.split_first_mut() else {
            return Ok(());
        };
        *first = self.last.unwrap_or(fill);
        rest.copy_from_slice(&xs[..xs.len() - 1]);
        self.last = xs.last().copied();
        Ok(())
    }
}

impl<T: Elem, const BOOL: bool> ScanBody for Prev<T, BOOL> {
    fn new() -> Self {
        Prev {
            last: None,
            _rows: PhantomData,
        }
    }
}

/// Registers `prev_<t>` for rows of storage type `T`.
fn register_prev<T: Elem, const BOOL: bool>(reg: &mut Registry) {
    let ty = tag::<T, BOOL>();
    reg.add(KernelDef {
        name: format!("prev_{}", ty.name()),
        class: KernelClass::Scan,
        op: Some(Op::Prev),
        inputs: vec![InputSpec::Array(ty), InputSpec::Scalar(ty)],
        output: ValueType::Array(ty),
        // One row read and one row written.
        cost: 2 * u64::try_from(ty.width()).expect("width fits in u64"),
        commutative: false,
        pattern: None,
        impls: KernelImpl::scan::<Prev<T, BOOL>>(),
    });
}

pub(crate) fn register(reg: &mut Registry) {
    register_prev::<f64, false>(reg);
    register_prev::<f32, false>(reg);
    register_prev::<i64, false>(reg);
    register_prev::<u64, false>(reg);
    register_prev::<i32, false>(reg);
    register_prev::<u8, false>(reg);
    register_prev::<u8, true>(reg);
}
