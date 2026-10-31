//! Scalar reference `gather` and `slice` kernels.

use std::marker::PhantomData;

use crate::kernels::scalar::construct::add_whole;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeBody};
use crate::ops::Op;
use crate::value::{Array, Elem, Scalar, ScalarType, Value, ValueType};

/// An `i64` scalar argument.
fn i64_arg(v: &Value) -> Result<i64, String> {
    match v {
        Value::Scalar(Scalar::I64(n)) => Ok(*n),
        other => Err(format!("expected i64, got {}", other.ty())),
    }
}

/// The body of `gather` on rows of storage type `T`. `BOOL` selects element type `Bool`,
/// whose storage type is `u8`, over `T::TYPE`.
pub(crate) struct Gather<T, const BOOL: bool>(PhantomData<fn() -> T>);

impl<T: Elem, const BOOL: bool> Gather<T, BOOL> {
    const ELEM: ScalarType = if BOOL { ScalarType::Bool } else { T::TYPE };

    fn rows(a: &Array) -> &[T] {
        if BOOL {
            bytemuck::cast_slice(a.as_bool_bytes())
        } else {
            a.as_slice::<T>()
        }
    }
}

impl<T: Elem, const BOOL: bool> WholeBody for Gather<T, BOOL> {
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs, idx] = args else {
            return Err(arity_error(2, args.len()));
        };
        let xs = Self::rows(array_arg(xs, Self::ELEM)?);
        let idx = array_arg(idx, ScalarType::I64)?.as_slice::<i64>();
        let n = u64::try_from(xs.len()).expect("length fits in u64");
        // The error must not depend on the order of the rows, so every index is checked
        // before any row is written. Negative indices wrap to values above `n`. The fold has
        // no early exit, so it vectorizes.
        let bad = idx.iter().fold(false, |b, &i| b | (i.cast_unsigned() >= n));
        if bad {
            let i = idx
                .iter()
                .find(|&&i| i.cast_unsigned() >= n)
                .expect("a bad index exists");
            return Err(format!("gather index {i} out of range for length {n}"));
        }
        // The clamp is a no-op after the check above, and keeps the loop free of an exit.
        // The source is only empty when `idx` is, so the loop body never runs on it.
        let last = xs.len().saturating_sub(1);
        Array::build_bytes(idx.len(), Self::ELEM, |out| {
            let out: &mut [T] = bytemuck::cast_slice_mut(out);
            for (o, &i) in out.iter_mut().zip(idx) {
                let i = usize::try_from(i.cast_unsigned()).unwrap_or(last).min(last);
                *o = xs[i];
            }
            Ok(())
        })
        .map(Value::Array)
    }
}

/// `slice` on arrays of type `ty`, a view of rows `start..end`.
fn run_slice(args: &[Value], ty: ScalarType) -> Result<Value, String> {
    let [xs, start, end] = args else {
        return Err(arity_error(3, args.len()));
    };
    let xs = array_arg(xs, ty)?;
    let (start, end) = (i64_arg(start)?, i64_arg(end)?);
    let len = xs.len();
    let bounds = usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .filter(|&(s, e)| s <= e && e <= len);
    let Some((s, e)) = bounds else {
        return Err(format!(
            "slice {start}..{end} out of range for length {len}"
        ));
    };
    Ok(Value::Array(xs.slice(s, e - s)))
}

fn width(ty: ScalarType) -> u64 {
    u64::try_from(ty.width()).expect("width fits in u64")
}

// The kernel closures must coerce to bare function pointers, so their type arguments are
// constant expressions, never locals.

macro_rules! gather {
    ($reg:expr, $t:ty, $bool:literal, $ty:expr) => {
        $reg.add(KernelDef {
            name: format!("gather_{}", $ty.name()),
            class: KernelClass::Whole,
            op: Some(Op::Gather),
            inputs: vec![InputSpec::Array($ty), InputSpec::Array(ScalarType::I64)],
            output: ValueType::Array($ty),
            // One index and one source row read, one row written.
            cost: 8 + 2 * width($ty),
            commutative: false,
            pattern: None,
            impls: KernelImpl::whole::<Gather<$t, $bool>>(),
        })
    };
}

macro_rules! slice {
    ($reg:expr, $ty:expr) => {
        add_whole(
            $reg,
            Op::Slice,
            format!("slice_{}", $ty.name()),
            vec![
                InputSpec::Array($ty),
                InputSpec::Scalar(ScalarType::I64),
                InputSpec::Scalar(ScalarType::I64),
            ],
            ValueType::Array($ty),
            // A view copies nothing.
            0,
            |args| run_slice(args, $ty),
        )
    };
}

pub(crate) fn register(reg: &mut Registry) {
    gather!(reg, f64, false, ScalarType::F64);
    gather!(reg, f32, false, ScalarType::F32);
    gather!(reg, i64, false, ScalarType::I64);
    gather!(reg, u64, false, ScalarType::U64);
    gather!(reg, i32, false, ScalarType::I32);
    gather!(reg, u8, false, ScalarType::U8);
    // Bool arrays are stored as `u8`.
    gather!(reg, u8, true, ScalarType::Bool);
    slice!(reg, ScalarType::F64);
    slice!(reg, ScalarType::F32);
    slice!(reg, ScalarType::I64);
    slice!(reg, ScalarType::U64);
    slice!(reg, ScalarType::I32);
    slice!(reg, ScalarType::U8);
    slice!(reg, ScalarType::Bool);
}
