//! Scalar reference `argsort` kernels, the stable ascending positions of an array or of the
//! rows of two equally long arrays.

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

/// The body of `argsort` on the rows of two columns of types `A` and `B`, the positions of the
/// rows in ascending order of (`a`, `b`, position).
pub(crate) struct ArgSort2<A, B>(PhantomData<fn() -> (A, B)>);

impl<A: RadixKey, B: RadixKey> WholeBody for ArgSort2<A, B> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs, ys] = args else {
            return Err(arity_error(2, args.len()));
        };
        let xs = array_arg(xs, A::TYPE)?.as_slice::<A>();
        let ys = array_arg(ys, B::TYPE)?.as_slice::<B>();
        Array::build_bytes(xs.len(), ScalarType::I64, |out| {
            composite::argsort2_into(xs, ys, bytemuck::cast_slice_mut(out))
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

/// The oracle of `argsort(a, b)`: the positions of the rows of `a` and `b` in ascending order
/// of (`a`, `b`, position).
#[cfg(test)]
fn argsort2_reference<A: Elem + Ord, B: Elem + Ord>(a: &[A], b: &[B]) -> Vec<i64> {
    assert_eq!(a.len(), b.len(), "the reference takes equally long columns");
    let mut out: Vec<i64> = (0..).take(a.len()).collect();
    // Ties break on the position, so the unstable sort is stable.
    out.sort_unstable_by_key(|&i| {
        let row = usize::try_from(i.cast_unsigned()).unwrap_or(0);
        (a[row], b[row], i)
    });
    out
}

mod composite;
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

macro_rules! argsort2 {
    ($reg:expr, $a:ty, $ta:expr, $b:ty, $tb:expr) => {
        $reg.add(KernelDef {
            name: format!("argsort2_{}_{}", $ta.name(), $tb.name()),
            class: KernelClass::Whole,
            op: Some(Op::ArgSort),
            inputs: vec![InputSpec::Array($ta), InputSpec::Array($tb)],
            output: ValueType::Array(ScalarType::I64),
            // One row of each column read and one position written, ignoring the sort passes.
            cost: width($ta) + width($tb) + 8,
            commutative: false,
            pattern: None,
            total: false,
            impls: KernelImpl::whole::<ArgSort2<$a, $b>>(),
        })
    };
}

/// `argsort2` with first column type `$a` and every second column type.
macro_rules! argsort2_all {
    ($reg:expr, $a:ty, $ta:expr) => {
        argsort2!($reg, $a, $ta, i64, ScalarType::I64);
        argsort2!($reg, $a, $ta, u64, ScalarType::U64);
        argsort2!($reg, $a, $ta, i32, ScalarType::I32);
        argsort2!($reg, $a, $ta, u8, ScalarType::U8);
    };
}

pub(crate) fn register(reg: &mut Registry) {
    argsort!(reg, i64, ScalarType::I64);
    argsort!(reg, u64, ScalarType::U64);
    argsort!(reg, i32, ScalarType::I32);
    argsort!(reg, u8, ScalarType::U8);
    // Every pair of key types, so the checker accepts any combination.
    argsort2_all!(reg, i64, ScalarType::I64);
    argsort2_all!(reg, u64, ScalarType::U64);
    argsort2_all!(reg, i32, ScalarType::I32);
    argsort2_all!(reg, u8, ScalarType::U8);
}
