//! Scalar reference `gather`, `slice`, `histogram`, and `reduce_into` kernels.

use std::marker::PhantomData;

use crate::kernels::scalar::construct::add_whole;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::scalar::scan::ScanOp;
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

/// The rows of `a` as storage type `T`, which is `u8` for a bool array. The caller has checked
/// the element type of `a` against `T`.
fn rows_of<T: Elem>(a: &Array) -> &[T] {
    if a.elem() == ScalarType::Bool {
        bytemuck::cast_slice(a.as_bool_bytes())
    } else {
        a.as_slice::<T>()
    }
}

/// The body of `gather` on rows of storage type `T`. `BOOL` selects element type `Bool`,
/// whose storage type is `u8`, over `T::TYPE`.
pub(crate) struct Gather<T, const BOOL: bool>(PhantomData<fn() -> T>);

impl<T: Elem, const BOOL: bool> Gather<T, BOOL> {
    const ELEM: ScalarType = if BOOL { ScalarType::Bool } else { T::TYPE };

    fn rows(a: &Array) -> &[T] {
        rows_of(a)
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

/// The bucket count `n` and the keys, checked to lie in `0..n`. A negative `n` and a key outside
/// `0..n` are errors.
fn checked_keys<'a>(keys: &'a Value, n: &Value) -> Result<(&'a [i64], usize), String> {
    let keys = array_arg(keys, ScalarType::I64)?.as_slice::<i64>();
    let buckets = i64_arg(n)?;
    let Ok(count) = usize::try_from(buckets) else {
        return Err(format!("negative bucket count {buckets}"));
    };
    // The error must not depend on the order of the keys, so every key is checked before any
    // bucket is written. Negative keys wrap to values above `n`. The fold has no early exit,
    // so it vectorizes.
    let bound = buckets.cast_unsigned();
    let bad = keys
        .iter()
        .fold(false, |b, &k| b | (k.cast_unsigned() >= bound));
    if bad {
        let k = keys
            .iter()
            .find(|&&k| k.cast_unsigned() >= bound)
            .expect("a bad key exists");
        return Err(format!("bucket key {k} out of range for {buckets} buckets"));
    }
    Ok((keys, count))
}

/// The bucket of a key that `checked_keys` accepted. The clamp is a no-op after the check and
/// keeps the loop free of an exit, a bucket exists whenever a key does.
fn bucket(k: i64, count: usize) -> usize {
    usize::try_from(k.cast_unsigned())
        .unwrap_or(0)
        .min(count.saturating_sub(1))
}

/// The body of `histogram`, the number of occurrences of each key in `0..n`.
pub(crate) struct Histogram;

impl WholeBody for Histogram {
    fn run(args: &[Value]) -> Result<Value, String> {
        let [keys, n] = args else {
            return Err(arity_error(2, args.len()));
        };
        let (keys, count) = checked_keys(keys, n)?;
        Array::try_build::<i64>(count, |out| {
            for &k in keys {
                let b = &mut out[bucket(k, count)];
                *b = b.wrapping_add(1);
            }
        })
        .map(Value::Array)
    }
}

/// The body of `reduce_into`, the fold `O` of the values per key in `0..n`. Rows of storage type
/// `T` fold into buckets of `O::Out`, and a bucket without rows holds `O::identity`.
pub(crate) struct ReduceInto<T, O>(PhantomData<fn() -> (T, O)>);

impl<T: Elem, O: ScanOp<T>> WholeBody for ReduceInto<T, O> {
    fn run(args: &[Value]) -> Result<Value, String> {
        let [keys, vals, n] = args else {
            return Err(arity_error(3, args.len()));
        };
        let vals = rows_of::<T>(array_arg(vals, O::IN_TYPE)?);
        let (keys, count) = checked_keys(keys, n)?;
        if keys.len() != vals.len() {
            return Err(format!(
                "length mismatch: {} keys for {} values",
                keys.len(),
                vals.len()
            ));
        }
        Array::build_bytes(count, O::OUT_TYPE, |out| {
            let out: &mut [O::Out] = bytemuck::cast_slice_mut(out);
            out.fill(O::identity());
            for (&k, &x) in keys.iter().zip(vals) {
                let b = &mut out[bucket(k, count)];
                *b = O::step(*b, x);
            }
            Ok(())
        })
        .map(Value::Array)
    }
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
    reg.add(KernelDef {
        name: "histogram".into(),
        class: KernelClass::Whole,
        op: Some(Op::Histogram),
        inputs: vec![
            InputSpec::Array(ScalarType::I64),
            InputSpec::Scalar(ScalarType::I64),
        ],
        output: ValueType::Array(ScalarType::I64),
        // One key read, and one bucket read and written.
        cost: 8 + 2 * 8,
        commutative: false,
        pattern: None,
        impls: KernelImpl::whole::<Histogram>(),
    });
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
