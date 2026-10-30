//! Scalar reference reductions.
//!
//! Float `add` and `mul` use the canonical 16-accumulator order from the spec, as a state,
//! update, and finish triple so a tiled executor can feed tiles whose lengths are multiples
//! of 16 and still produce the one-shot result bit for bit. Every SIMD variant is tested
//! against these functions.

use std::marker::PhantomData;

use crate::kernels::scalar::elementwise::{Num, arity_error, rows_arg, scalar_from};
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Reducer, ReducerFn, Registry};
use crate::ops::{Op, ReduceOp};
use crate::value::{Arg, Array, Elem, Scalar, ScalarType, Value, ValueType};

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

/// Adds `n` elements `f(0)..f(n)` into the accumulators, element `i` into accumulator
/// `i mod 16`. `n` must be a multiple of 16. This is the one loop over a tile's rows that
/// every canonical reduction shares, with `f` reading and combining the operands of a row.
#[inline(always)]
fn update_lanes_with<T: Float>(
    acc: &mut [T; LANES],
    n: usize,
    f: impl Fn(usize) -> T,
    op: fn(T, T) -> T,
) {
    assert!(
        n.is_multiple_of(LANES),
        "canonical update needs a multiple of {LANES} elements, got {n}"
    );
    for start in (0..n).step_by(LANES) {
        for (j, a) in acc.iter_mut().enumerate() {
            *a = op(*a, f(start + j));
        }
    }
}

/// Adds whole blocks of 16 into the accumulators, one element per accumulator.
fn update_lanes<T: Float>(acc: &mut [T; LANES], block: &[T], op: fn(T, T) -> T) {
    update_lanes_with(acc, block.len(), |i| block[i], op);
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

/// The running state of one reduction over rows of `T`, behind the reducers below.
///
/// `feed` consumes `n` rows, where `f(i)` is the value of row `i` of the tile. A reducer
/// whose result depends on the grouping of rows (the canonical float order) accepts only
/// multiples of 16 rows in every `feed` but the last.
pub trait Accum<T>: Send + Sized {
    /// The storage type of the result.
    type Out: Elem;

    fn new() -> Self;

    fn feed(&mut self, n: usize, f: impl Fn(usize) -> T) -> Result<(), String>;

    fn finish(self) -> Result<Self::Out, String>;
}

/// The canonical float state for `add` (`MUL` false) or `mul` (`MUL` true). The accumulators
/// are the state, and the tail of fewer than 16 rows is kept until `finish`, so tiles of
/// multiples of 16 rows reproduce the one-shot bits.
#[derive(Clone, Debug)]
pub struct CanonicalAcc<T, const MUL: bool> {
    acc: [T; LANES],
    tail: [T; LANES],
    tail_len: usize,
}

impl<T: Float, const MUL: bool> CanonicalAcc<T, MUL> {
    const IDENTITY: T = if MUL { T::ONE } else { T::ZERO };

    #[inline(always)]
    fn op(a: T, b: T) -> T {
        if MUL { Num::mul(a, b) } else { Num::add(a, b) }
    }
}

impl<T: Float, const MUL: bool> Accum<T> for CanonicalAcc<T, MUL> {
    type Out = T;

    fn new() -> Self {
        CanonicalAcc {
            acc: [Self::IDENTITY; LANES],
            tail: [Self::IDENTITY; LANES],
            tail_len: 0,
        }
    }

    #[inline(always)]
    fn feed(&mut self, n: usize, f: impl Fn(usize) -> T) -> Result<(), String> {
        if self.tail_len != 0 {
            return Err("canonical update after a partial block".into());
        }
        let blocks = n - n % LANES;
        update_lanes_with(&mut self.acc, blocks, &f, Self::op);
        self.tail_len = n - blocks;
        for (j, slot) in self.tail[..self.tail_len].iter_mut().enumerate() {
            *slot = f(blocks + j);
        }
        Ok(())
    }

    fn finish(self) -> Result<T, String> {
        Ok(finish_lanes(
            self.acc,
            &self.tail[..self.tail_len],
            Self::op,
        ))
    }
}

/// Wrapping integer sum in the widened type `W`.
pub struct IntSum<W>(W);

/// Wrapping integer product.
pub struct IntProduct<T>(T);

macro_rules! impl_int_accum {
    ($($t:ty => $w:ty),*) => {$(
        impl Accum<$t> for IntSum<$w> {
            type Out = $w;
            fn new() -> Self { IntSum(0) }
            #[inline(always)]
            fn feed(&mut self, n: usize, f: impl Fn(usize) -> $t) -> Result<(), String> {
                for i in 0..n {
                    self.0 = self.0.wrapping_add(<$w>::from(f(i)));
                }
                Ok(())
            }
            fn finish(self) -> Result<$w, String> { Ok(self.0) }
        }

        impl Accum<$t> for IntProduct<$t> {
            type Out = $t;
            fn new() -> Self { IntProduct(1) }
            #[inline(always)]
            fn feed(&mut self, n: usize, f: impl Fn(usize) -> $t) -> Result<(), String> {
                for i in 0..n {
                    self.0 = self.0.wrapping_mul(f(i));
                }
                Ok(())
            }
            fn finish(self) -> Result<$t, String> { Ok(self.0) }
        }
    )*};
}

impl_int_accum!(i64 => i64, i32 => i64, u8 => u64);

/// `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true). An empty array is an error.
pub struct Extreme<T, const MAX: bool> {
    cur: Option<T>,
}

impl<T: Ordered, const MAX: bool> Accum<T> for Extreme<T, MAX> {
    type Out = T;

    fn new() -> Self {
        Extreme { cur: None }
    }

    #[inline(always)]
    fn feed(&mut self, n: usize, f: impl Fn(usize) -> T) -> Result<(), String> {
        for i in 0..n {
            let x = f(i);
            self.cur = Some(match self.cur {
                None => x,
                Some(a) if MAX => a.max2(x),
                Some(a) => a.min2(x),
            });
        }
        Ok(())
    }

    fn finish(self) -> Result<T, String> {
        self.cur.ok_or_else(|| {
            format!(
                "reduce_{} of an empty array",
                if MAX { "max" } else { "min" }
            )
        })
    }
}

/// `argmin` (`MAX` false) or `argmax` (`MAX` true): the index of the first NaN, otherwise the
/// first index of the extreme value. `consumed` is the count of rows seen so far, which turns
/// the tile-local row index into the index in the whole array.
pub struct ArgExtreme<T, const MAX: bool> {
    consumed: usize,
    best: Option<(usize, T)>,
    seen_nan: Option<usize>,
}

impl<T: Ordered, const MAX: bool> Accum<T> for ArgExtreme<T, MAX> {
    type Out = i64;

    fn new() -> Self {
        ArgExtreme {
            consumed: 0,
            best: None,
            seen_nan: None,
        }
    }

    #[inline(always)]
    fn feed(&mut self, n: usize, f: impl Fn(usize) -> T) -> Result<(), String> {
        let base = self.consumed;
        self.consumed += n;
        for i in 0..n {
            // The first NaN decides the result, so later rows cannot change it.
            if self.seen_nan.is_some() {
                break;
            }
            let x = f(i);
            if x.is_nan_() {
                self.seen_nan = Some(base + i);
                break;
            }
            let better = match self.best {
                None => true,
                Some((_, b)) if MAX => x.greater(b),
                Some((_, b)) => x.less(b),
            };
            if better {
                self.best = Some((base + i, x));
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<i64, String> {
        let what = if MAX { "argmax" } else { "argmin" };
        let index = self
            .seen_nan
            .or(self.best.map(|(i, _)| i))
            .ok_or_else(|| format!("{what} of an empty array"))?;
        i64::try_from(index).map_err(|e| e.to_string())
    }
}

/// `reduce_and` and `all`, over the 0/1 byte storage of bool.
pub struct BoolAnd(bool);

/// `reduce_or` and `any`.
pub struct BoolOr(bool);

/// `reduce_xor`.
pub struct BoolXor(bool);

/// `count`.
pub struct BoolCount(usize);

macro_rules! impl_bool_accum {
    ($name:ident, $init:expr, $out:ty, |$acc:ident, $x:ident| $step:expr, |$s:ident| $done:expr) => {
        impl Accum<u8> for $name {
            type Out = $out;
            fn new() -> Self {
                $name($init)
            }
            #[inline(always)]
            fn feed(&mut self, n: usize, f: impl Fn(usize) -> u8) -> Result<(), String> {
                for i in 0..n {
                    let $x = f(i) != 0;
                    let $acc = &mut self.0;
                    $step;
                }
                Ok(())
            }
            fn finish(self) -> Result<$out, String> {
                let $s = self.0;
                $done
            }
        }
    };
}

impl_bool_accum!(BoolAnd, true, u8, |a, x| *a &= x, |s| Ok(u8::from(s)));
impl_bool_accum!(BoolOr, false, u8, |a, x| *a |= x, |s| Ok(u8::from(s)));
impl_bool_accum!(BoolXor, false, u8, |a, x| *a ^= x, |s| Ok(u8::from(s)));
impl_bool_accum!(BoolCount, 0, i64, |a, x| *a += usize::from(x), |s| {
    i64::try_from(s).map_err(|e| e.to_string())
});

/// Element types with kernels for every numeric reduction.
pub trait Reducible: Ordered {
    /// The result type of `add`. Narrow integers widen.
    type Sum: Elem;
    type AddAcc: Accum<Self, Out = Self::Sum> + 'static;
    type MulAcc: Accum<Self, Out = Self> + 'static;
}

macro_rules! impl_reducible_float {
    ($($t:ty),*) => {$(
        impl Reducible for $t {
            type Sum = $t;
            type AddAcc = CanonicalAcc<$t, false>;
            type MulAcc = CanonicalAcc<$t, true>;
        }
    )*};
}

macro_rules! impl_reducible_int {
    ($($t:ty => $w:ty),*) => {$(
        impl Reducible for $t {
            type Sum = $w;
            type AddAcc = IntSum<$w>;
            type MulAcc = IntProduct<$t>;
        }
    )*};
}

impl_reducible_float!(f64, f32);
impl_reducible_int!(i64 => i64, i32 => i64, u8 => u64);

/// The array argument of a one-argument reduction, checked against `ty`.
pub(crate) fn array_arg(v: &Value, ty: ScalarType) -> Result<&Array, String> {
    match v {
        Value::Array(a) if a.elem() == ty => Ok(a),
        other => Err(format!("expected {}[], got {}", ty.name(), other.ty())),
    }
}

/// The reducer of one array argument: `A` accumulates rows of storage type `T`, and the
/// result becomes a scalar of type `out_ty`.
struct Fold<T, A> {
    acc: A,
    in_ty: ScalarType,
    out_ty: ScalarType,
    _rows: PhantomData<fn() -> T>,
}

impl<T: Elem, A: Accum<T>> Reducer for Fold<T, A> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [a] = args else {
            return Err(arity_error(1, args.len()));
        };
        let xs = rows_arg::<T>(a, self.in_ty)?;
        self.acc.feed(xs.len(), |i| xs[i])
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        Ok(scalar_from(self.acc.finish()?, self.out_ty))
    }
}

/// A new reducer over one array of type `in_ty` with result type `out_ty`. `T` is the storage
/// type of `in_ty`.
fn fold<T: Elem, A: Accum<T> + 'static>(in_ty: ScalarType, out_ty: ScalarType) -> Box<dyn Reducer> {
    Box::new(Fold::<T, A> {
        acc: A::new(),
        in_ty,
        out_ty,
        _rows: PhantomData,
    })
}

fn add_reduction(
    reg: &mut Registry,
    op: Op,
    name: String,
    in_ty: ScalarType,
    out_ty: ScalarType,
    make: ReducerFn,
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
        impls: KernelImpl::Reduction(make).scalar_only(),
    });
}

// The reducer constructors must coerce to bare function pointers, so their type arguments are
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
        || fold::<T, T::AddAcc>(T::TYPE, T::Sum::TYPE),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Mul),
        format!("reduce_mul_{n}"),
        T::TYPE,
        T::TYPE,
        || fold::<T, T::MulAcc>(T::TYPE, T::TYPE),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Min),
        format!("reduce_min_{n}"),
        T::TYPE,
        T::TYPE,
        || fold::<T, Extreme<T, false>>(T::TYPE, T::TYPE),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Max),
        format!("reduce_max_{n}"),
        T::TYPE,
        T::TYPE,
        || fold::<T, Extreme<T, true>>(T::TYPE, T::TYPE),
    );
    add_reduction(
        reg,
        Op::ArgMin,
        format!("argmin_{n}"),
        T::TYPE,
        ScalarType::I64,
        || fold::<T, ArgExtreme<T, false>>(T::TYPE, ScalarType::I64),
    );
    add_reduction(
        reg,
        Op::ArgMax,
        format!("argmax_{n}"),
        T::TYPE,
        ScalarType::I64,
        || fold::<T, ArgExtreme<T, true>>(T::TYPE, ScalarType::I64),
    );
}

/// Bool reductions read the 0/1 byte storage.
fn register_bool(reg: &mut Registry) {
    const B: ScalarType = ScalarType::Bool;
    const I: ScalarType = ScalarType::I64;
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::And),
        "reduce_and_bool".into(),
        B,
        B,
        || fold::<u8, BoolAnd>(B, B),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Or),
        "reduce_or_bool".into(),
        B,
        B,
        || fold::<u8, BoolOr>(B, B),
    );
    add_reduction(
        reg,
        Op::Reduce(ReduceOp::Xor),
        "reduce_xor_bool".into(),
        B,
        B,
        || fold::<u8, BoolXor>(B, B),
    );
    add_reduction(reg, Op::Count, "count".into(), B, I, || {
        fold::<u8, BoolCount>(B, I)
    });
    add_reduction(reg, Op::Any, "any".into(), B, B, || {
        fold::<u8, BoolOr>(B, B)
    });
    add_reduction(reg, Op::All, "all".into(), B, B, || {
        fold::<u8, BoolAnd>(B, B)
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
