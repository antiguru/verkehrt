//! Scalar reference `argsort` kernels, the stable ascending positions of an array.

use std::marker::PhantomData;

use crate::kernels::scalar::construct::width;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::scalar::sort::radix::RadixKey;
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeBody};
use crate::ops::Op;
#[cfg(test)]
use crate::value::Elem;
use crate::value::{Array, ScalarType, Value, ValueType};

/// The body of `argsort` on rows of type `T`, the positions of the rows in stable ascending
/// order.
pub(crate) struct ArgSort<T>(PhantomData<fn() -> T>);

impl<T: RadixKey> WholeBody for ArgSort<T> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs] = args else {
            return Err(arity_error(1, args.len()));
        };
        let xs = array_arg(xs, T::TYPE)?.as_slice::<T>();
        Array::build_bytes(xs.len(), ScalarType::I64, |out| {
            radix::argsort_into(xs, bytemuck::cast_slice_mut(out))
        })
        .map(Value::Array)
    }
}

/// The plan 4 body of `argsort`, kept as the oracle that pins the kernel's output: the stable
/// ascending positions of `xs`.
#[cfg(test)]
fn argsort_reference<T: Elem + Ord>(xs: &[T]) -> Vec<i64> {
    let mut out: Vec<i64> = (0..).take(xs.len()).collect();
    // Ties break on the position, so the unstable sort is stable.
    out.sort_unstable_by_key(|&i| {
        let row = usize::try_from(i.cast_unsigned()).unwrap_or(0);
        (xs[row], i)
    });
    out
}

pub(super) mod radix;
#[cfg(test)]
mod tests;

macro_rules! argsort {
    ($reg:expr, $t:ty, $ty:expr) => {
        $reg.add(KernelDef {
            name: format!("argsort_{}", $ty.name()),
            class: KernelClass::Whole,
            op: Some(Op::ArgSort),
            inputs: vec![InputSpec::Array($ty)],
            output: ValueType::Array(ScalarType::I64),
            // One row read and one position written, ignoring the sort passes.
            cost: width($ty) + 8,
            commutative: false,
            pattern: None,
            total: false,
            impls: KernelImpl::whole::<ArgSort<$t>>(),
        })
    };
}

pub(crate) fn register(reg: &mut Registry) {
    argsort!(reg, i64, ScalarType::I64);
    argsort!(reg, u64, ScalarType::U64);
    argsort!(reg, i32, ScalarType::I32);
    argsort!(reg, u8, ScalarType::U8);
}
