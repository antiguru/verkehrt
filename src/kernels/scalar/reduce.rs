//! Scalar reference reductions.
//!
//! Float `add` and `mul` use the canonical 16-accumulator order from the spec, as a state,
//! update, and finish triple so a tiled executor can feed tiles whose lengths are multiples
//! of 16 and still produce the one-shot result bit for bit. Every SIMD variant is tested
//! against these functions.

use crate::kernels::scalar::elementwise::{Num, arity_error, scalar_from, slice_of};
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, WholeFn};
use crate::ops::{Op, ReduceOp};
use crate::value::{Array, Elem, ScalarType, Value, ValueType};

/// Number of independent accumulators in the canonical order.
const LANES: usize = 16;

/// Float element types of the canonical reductions.
pub trait Float: Num {
    const ZERO: Self;
    const ONE: Self;
}

impl Float for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
}

/// Adds whole blocks of 16 into the accumulators, one element per accumulator.
fn update_lanes<T: Float>(acc: &mut [T; LANES], block: &[T], op: fn(T, T) -> T) {
    assert!(
        block.len().is_multiple_of(LANES),
        "canonical update needs a multiple of {LANES} elements, got {}",
        block.len()
    );
    for chunk in block.as_chunks::<LANES>().0 {
        for (a, &x) in acc.iter_mut().zip(chunk) {
            *a = op(*a, x);
        }
    }
}

/// Combines the accumulators pairwise (`+8`, `+4`, `+2`, `+1`), then folds the tail into
/// accumulator 0 in index order.
fn finish_lanes<T: Float>(mut acc: [T; LANES], tail: &[T], op: fn(T, T) -> T) -> T {
    assert!(
        tail.len() < LANES,
        "canonical tail must be shorter than {LANES}, got {}",
        tail.len()
    );
    let mut step = LANES / 2;
    while step > 0 {
        for j in 0..step {
            acc[j] = op(acc[j], acc[j + step]);
        }
        step /= 2;
    }
    tail.iter().fold(acc[0], |a, &x| op(a, x))
}

/// Canonical float summation state. Accumulators start at `+0.0`.
#[derive(Clone, Debug)]
pub struct CanonicalSum<T> {
    acc: [T; LANES],
}

impl<T: Float> CanonicalSum<T> {
    pub fn new() -> Self {
        CanonicalSum {
            acc: [T::ZERO; LANES],
        }
    }

    /// Consumes a block. Panics unless `block.len()` is a multiple of 16.
    pub fn update(&mut self, block: &[T]) {
        update_lanes(&mut self.acc, block, T::add);
    }

    /// The sum. Panics unless `tail.len() < 16`.
    pub fn finish(self, tail: &[T]) -> T {
        finish_lanes(self.acc, tail, T::add)
    }
}

impl<T: Float> Default for CanonicalSum<T> {
    fn default() -> Self {
        CanonicalSum::new()
    }
}

/// Canonical float product state. The same order as `CanonicalSum` with identity `1.0`.
#[derive(Clone, Debug)]
pub struct CanonicalProduct<T> {
    acc: [T; LANES],
}

impl<T: Float> CanonicalProduct<T> {
    pub fn new() -> Self {
        CanonicalProduct {
            acc: [T::ONE; LANES],
        }
    }

    /// Consumes a block. Panics unless `block.len()` is a multiple of 16.
    pub fn update(&mut self, block: &[T]) {
        update_lanes(&mut self.acc, block, T::mul);
    }

    /// The product. Panics unless `tail.len() < 16`.
    pub fn finish(self, tail: &[T]) -> T {
        finish_lanes(self.acc, tail, T::mul)
    }
}

impl<T: Float> Default for CanonicalProduct<T> {
    fn default() -> Self {
        CanonicalProduct::new()
    }
}

/// The canonical sum of all of `xs`.
pub fn canonical_sum<T: Float>(xs: &[T]) -> T {
    let b = xs.len() - xs.len() % LANES;
    let mut s = CanonicalSum::new();
    s.update(&xs[..b]);
    s.finish(&xs[b..])
}

/// The canonical product of all of `xs`.
pub fn canonical_product<T: Float>(xs: &[T]) -> T {
    let b = xs.len() - xs.len() % LANES;
    let mut p = CanonicalProduct::new();
    p.update(&xs[..b]);
    p.finish(&xs[b..])
}

/// Element types with a total "which is smaller" order for `argmin` and `argmax`.
/// Floats rank `-0.0` below `+0.0`. NaN has no rank, the callers handle it first.
pub trait Ordered: Num {
    fn is_nan_(self) -> bool;
    fn less(self, o: Self) -> bool;
    fn greater(self, o: Self) -> bool;
}

macro_rules! impl_ordered_int {
    ($($t:ty),*) => {$(
        impl Ordered for $t {
            fn is_nan_(self) -> bool { false }
            fn less(self, o: Self) -> bool { self < o }
            fn greater(self, o: Self) -> bool { self > o }
        }
    )*};
}

macro_rules! impl_ordered_float {
    ($($t:ty),*) => {$(
        impl Ordered for $t {
            fn is_nan_(self) -> bool { self.is_nan() }
            fn less(self, o: Self) -> bool {
                self < o || (self == o && self.is_sign_negative() && o.is_sign_positive())
            }
            fn greater(self, o: Self) -> bool {
                self > o || (self == o && self.is_sign_positive() && o.is_sign_negative())
            }
        }
    )*};
}

impl_ordered_int!(i64, i32, u8);
impl_ordered_float!(f64, f32);

/// Element types with kernels for every numeric reduction.
pub trait Reducible: Ordered {
    /// The result type of `add`. Narrow integers widen.
    type Sum: Elem;
    fn sum(xs: &[Self]) -> Self::Sum;
    fn product(xs: &[Self]) -> Self;
}

macro_rules! impl_reducible_float {
    ($($t:ty),*) => {$(
        impl Reducible for $t {
            type Sum = $t;
            fn sum(xs: &[Self]) -> $t { canonical_sum(xs) }
            fn product(xs: &[Self]) -> $t { canonical_product(xs) }
        }
    )*};
}

macro_rules! impl_reducible_int {
    ($($t:ty => $w:ty),*) => {$(
        impl Reducible for $t {
            type Sum = $w;
            fn sum(xs: &[Self]) -> $w {
                xs.iter().fold(0, |a: $w, &x| a.wrapping_add(<$w>::from(x)))
            }
            fn product(xs: &[Self]) -> $t {
                xs.iter().fold(1, |a: $t, &x| a.wrapping_mul(x))
            }
        }
    )*};
}

impl_reducible_float!(f64, f32);
impl_reducible_int!(i64 => i64, i32 => i64, u8 => u64);

/// Folds with `f`, which is a `minimum` or `maximum`. An empty array is an error.
fn extreme<T: Ordered>(xs: &[T], f: fn(T, T) -> T, what: &str) -> Result<T, String> {
    let (&first, rest) = xs
        .split_first()
        .ok_or_else(|| format!("{what} of an empty array"))?;
    Ok(rest.iter().fold(first, |a, &x| f(a, x)))
}

/// The index of the first NaN, otherwise the first index of the extreme value under `better`.
fn arg_extreme<T: Ordered>(xs: &[T], better: fn(T, T) -> bool, what: &str) -> Result<i64, String> {
    if xs.is_empty() {
        return Err(format!("{what} of an empty array"));
    }
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x.is_nan_() {
            best = i;
            break;
        }
        if better(x, xs[best]) {
            best = i;
        }
    }
    i64::try_from(best).map_err(|e| e.to_string())
}

/// The array argument of a one-argument reduction, checked against `ty`.
pub(crate) fn array_arg(v: &Value, ty: ScalarType) -> Result<&Array, String> {
    match v {
        Value::Array(a) if a.elem() == ty => Ok(a),
        other => Err(format!("expected {}[], got {}", ty.name(), other.ty())),
    }
}

/// Runs a reduction of one array argument. `T` is the storage type of `in_ty` and `R` that of
/// `out_ty`.
fn reduce_run<T: Elem, R: Elem>(
    args: &[Value],
    in_ty: ScalarType,
    out_ty: ScalarType,
    f: impl FnOnce(&[T]) -> Result<R, String>,
) -> Result<Value, String> {
    let [a] = args else {
        return Err(arity_error(1, args.len()));
    };
    let a = array_arg(a, in_ty)?;
    Ok(Value::Scalar(scalar_from(f(slice_of::<T>(a))?, out_ty)))
}

fn add_reduction(
    reg: &mut Registry,
    op: Op,
    name: String,
    in_ty: ScalarType,
    out_ty: ScalarType,
    f: WholeFn,
) {
    reg.add(KernelDef {
        name,
        class: KernelClass::Reduction,
        op: Some(op),
        inputs: vec![InputSpec::Array(in_ty)],
        output: ValueType::Scalar(out_ty),
        cost: u64::try_from(in_ty.width()).expect("width fits in u64"),
        commutative: false,
        pattern: None,
        impls: KernelImpl::Whole(f).scalar_only(),
    });
}

// The kernel closures must coerce to bare function pointers, so their type arguments are
// constant expressions such as `T::TYPE`, never locals.

fn register_numeric<T: Reducible>(reg: &mut Registry) {
    let n = T::TYPE.name();
    let sum_ty = T::Sum::TYPE;
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Add),
        format!("reduce_add_{n}"),
        T::TYPE,
        sum_ty,
        |args| reduce_run::<T, T::Sum>(args, T::TYPE, T::Sum::TYPE, |xs| Ok(T::sum(xs))),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Mul),
        format!("reduce_mul_{n}"),
        T::TYPE,
        T::TYPE,
        |args| reduce_run::<T, T>(args, T::TYPE, T::TYPE, |xs| Ok(T::product(xs))),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Min),
        format!("reduce_min_{n}"),
        T::TYPE,
        T::TYPE,
        |args| {
            reduce_run::<T, T>(args, T::TYPE, T::TYPE, |xs| {
                extreme(xs, T::min2, "reduce_min")
            })
        },
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Max),
        format!("reduce_max_{n}"),
        T::TYPE,
        T::TYPE,
        |args| {
            reduce_run::<T, T>(args, T::TYPE, T::TYPE, |xs| {
                extreme(xs, T::max2, "reduce_max")
            })
        },
    );
    add_reduction(
        reg,
        Op::ArgMin,
        format!("argmin_{n}"),
        T::TYPE,
        ScalarType::I64,
        |args| {
            reduce_run::<T, i64>(args, T::TYPE, ScalarType::I64, |xs| {
                arg_extreme(xs, T::less, "argmin")
            })
        },
    );
    add_reduction(
        reg,
        Op::ArgMax,
        format!("argmax_{n}"),
        T::TYPE,
        ScalarType::I64,
        |args| {
            reduce_run::<T, i64>(args, T::TYPE, ScalarType::I64, |xs| {
                arg_extreme(xs, T::greater, "argmax")
            })
        },
    );
}

/// Bool reductions read the 0/1 byte storage.
fn register_bool(reg: &mut Registry) {
    const B: ScalarType = ScalarType::Bool;
    for (name, op) in [
        ("reduce_and_bool", Op::Reduce(ReduceOp::And)),
        ("reduce_or_bool", Op::Reduce(ReduceOp::Or)),
        ("reduce_xor_bool", Op::Reduce(ReduceOp::Xor)),
    ] {
        let f: WholeFn = match op {
            Op::Reduce(ReduceOp::And) => |args| {
                reduce_run::<u8, u8>(args, B, B, |xs| Ok(u8::from(xs.iter().all(|&x| x != 0))))
            },
            Op::Reduce(ReduceOp::Or) => |args| {
                reduce_run::<u8, u8>(args, B, B, |xs| Ok(u8::from(xs.iter().any(|&x| x != 0))))
            },
            _ => |args| {
                reduce_run::<u8, u8>(args, B, B, |xs| {
                    Ok(xs.iter().fold(0, |a, &x| a ^ u8::from(x != 0)))
                })
            },
        };
        add_reduction(reg, op, name.to_string(), B, B, f);
    }
    add_reduction(reg, Op::Count, "count".into(), B, ScalarType::I64, |args| {
        reduce_run::<u8, i64>(args, B, ScalarType::I64, |xs| {
            i64::try_from(xs.iter().filter(|&&x| x != 0).count()).map_err(|e| e.to_string())
        })
    });
    add_reduction(reg, Op::Any, "any".into(), B, B, |args| {
        reduce_run::<u8, u8>(args, B, B, |xs| Ok(u8::from(xs.iter().any(|&x| x != 0))))
    });
    add_reduction(reg, Op::All, "all".into(), B, B, |args| {
        reduce_run::<u8, u8>(args, B, B, |xs| Ok(u8::from(xs.iter().all(|&x| x != 0))))
    });
}

pub(crate) fn register(reg: &mut Registry) {
    register_numeric::<f64>(reg);
    register_numeric::<f32>(reg);
    register_numeric::<i64>(reg);
    register_numeric::<i32>(reg);
    register_numeric::<u8>(reg);
    register_bool(reg);
}
