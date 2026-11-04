//! Scalar reference `argsort` kernels, the stable ascending positions of an array or of the
//! rows of two equally long arrays.

use std::marker::PhantomData;

use crate::kernels::scalar::construct::width;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::scalar::sort::radix::RadixKey;
use crate::kernels::{
    FusedPattern, InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeBody,
};
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
        // Checked before the output is allocated, with the message of `argsort2_into`.
        if xs.len() != ys.len() {
            return Err(format!("length mismatch: {} vs {}", xs.len(), ys.len()));
        }
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
mod topk;

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

/// Registers `topk_<t>`, the fused `slice(argsort(x), s, e)`.
fn topk<T: RadixKey>(reg: &mut Registry) {
    let t = T::TYPE;
    reg.add(KernelDef {
        name: format!("topk_{}", t.name()),
        class: KernelClass::Whole,
        op: None,
        inputs: vec![
            InputSpec::Array(t),
            InputSpec::Scalar(ScalarType::I64),
            InputSpec::Scalar(ScalarType::I64),
        ],
        output: ValueType::Array(ScalarType::I64),
        // One row read, below the argsort's row read and position written.
        cost: width(t),
        commutative: false,
        pattern: Some(FusedPattern {
            lhs: format!("(slice_i64 (argsort_{} ?x) ?s ?e)", t.name()),
            vars: ["?x", "?s", "?e"].map(String::from).to_vec(),
        }),
        total: false,
        impls: KernelImpl::whole::<topk::TopK<T>>(),
    });
}

/// Registers `topk2_<a>_<b>`, the fused `slice(argsort(x, y), s, e)`.
fn topk2<A: RadixKey, B: RadixKey>(reg: &mut Registry) {
    let (ta, tb) = (A::TYPE, B::TYPE);
    reg.add(KernelDef {
        name: format!("topk2_{}_{}", ta.name(), tb.name()),
        class: KernelClass::Whole,
        op: None,
        inputs: vec![
            InputSpec::Array(ta),
            InputSpec::Array(tb),
            InputSpec::Scalar(ScalarType::I64),
            InputSpec::Scalar(ScalarType::I64),
        ],
        output: ValueType::Array(ScalarType::I64),
        // One row of each column read, below the argsort's reads and position written.
        cost: width(ta) + width(tb),
        commutative: false,
        pattern: Some(FusedPattern {
            lhs: format!(
                "(slice_i64 (argsort2_{}_{} ?x ?y) ?s ?e)",
                ta.name(),
                tb.name()
            ),
            vars: ["?x", "?y", "?s", "?e"].map(String::from).to_vec(),
        }),
        total: false,
        impls: KernelImpl::whole::<topk::TopK2<A, B>>(),
    });
}

/// `topk2` with first column type `A` and every second column type.
fn topk2_all<A: RadixKey>(reg: &mut Registry) {
    topk2::<A, i64>(reg);
    topk2::<A, u64>(reg);
    topk2::<A, i32>(reg);
    topk2::<A, u8>(reg);
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
    topk::<i64>(reg);
    topk::<u64>(reg);
    topk::<i32>(reg);
    topk::<u8>(reg);
    topk2_all::<i64>(reg);
    topk2_all::<u64>(reg);
    topk2_all::<i32>(reg);
    topk2_all::<u8>(reg);
}
