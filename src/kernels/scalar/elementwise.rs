//! Scalar reference implementations of the single-op elementwise kernels.
//!
//! Every kernel entry is a thin wrapper over a slice function (`unary_into`, `binary_into`,
//! `select_into`) that fills exactly `out.len()` rows. A tiled executor calls the same
//! entry once per tile with `Arg::Rows` windows. The scalar-only `_s` kernels share the
//! entry of their array kernel, because a one-row destination with `Splat` operands is the
//! same computation.

use std::marker::PhantomData;
use std::ops::Range;

use crate::kernels::{ElementwiseBody, InputSpec, KernelClass, KernelDef, KernelImpl, Registry};
use crate::ops::Op;
use crate::value::{Arg, Array, Elem, Scalar, ScalarType, ValueType};

/// An elementwise operand: a slice of rows or one value repeated for every row.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a, T> {
    Slice(&'a [T]),
    Splat(T),
}

impl<'a, T: Copy> Operand<'a, T> {
    /// The row count, or `None` for a splat.
    pub fn rows(&self) -> Option<usize> {
        match self {
            Operand::Slice(s) => Some(s.len()),
            Operand::Splat(_) => None,
        }
    }

    /// The operand restricted to a row range. Panics when the range exceeds a slice.
    pub fn range(&self, r: Range<usize>) -> Operand<'a, T> {
        match *self {
            Operand::Slice(s) => Operand::Slice(&s[r]),
            Operand::Splat(v) => Operand::Splat(v),
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        match self {
            Operand::Slice(s) => s[i],
            Operand::Splat(v) => *v,
        }
    }
}

/// Writes `f(a[i], b[i])` for every row of `out`. Slice operands must have `out.len()` rows.
#[inline(always)]
pub fn binary_into<T: Elem, U: Elem>(
    a: Operand<T>,
    b: Operand<T>,
    out: &mut [U],
    f: impl Fn(T, T) -> U,
) {
    let n = out.len();
    match (a, b) {
        (Operand::Slice(a), Operand::Slice(b)) => {
            assert!(
                a.len() == n && b.len() == n,
                "operand rows must match the destination"
            );
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = f(x, y);
            }
        }
        (Operand::Slice(a), Operand::Splat(y)) => {
            assert_eq!(a.len(), n, "operand rows must match the destination");
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f(x, y);
            }
        }
        (Operand::Splat(x), Operand::Slice(b)) => {
            assert_eq!(b.len(), n, "operand rows must match the destination");
            for (o, &y) in out.iter_mut().zip(b) {
                *o = f(x, y);
            }
        }
        (Operand::Splat(x), Operand::Splat(y)) => out.fill(f(x, y)),
    }
}

/// Writes `f(a[i])` for every row of `out`. A slice operand must have `out.len()` rows.
#[inline(always)]
pub fn unary_into<T: Elem, U: Elem>(a: Operand<T>, out: &mut [U], f: impl Fn(T) -> U) {
    match a {
        Operand::Slice(a) => {
            assert_eq!(
                a.len(),
                out.len(),
                "operand rows must match the destination"
            );
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f(x);
            }
        }
        Operand::Splat(x) => out.fill(f(x)),
    }
}

/// Writes `t[i]` where `cond[i]` is nonzero and `e[i]` otherwise, for every row of `out`.
/// `cond` holds bool storage (0 or 1). Slice operands must have `out.len()` rows.
#[inline(always)]
pub fn select_into<T: Elem>(cond: Operand<u8>, t: Operand<T>, e: Operand<T>, out: &mut [T]) {
    for op in [cond.rows(), t.rows(), e.rows()].into_iter().flatten() {
        assert_eq!(op, out.len(), "operand rows must match the destination");
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = if cond.at(i) != 0 { t.at(i) } else { e.at(i) };
    }
}

/// Typed rows of an array. Bool arrays are viewed as their `u8` storage, so `T` is `u8` then.
pub(crate) fn slice_of<T: Elem>(a: &Array) -> &[T] {
    if a.elem() == ScalarType::Bool {
        bytemuck::cast_slice(a.as_bool_bytes())
    } else {
        a.as_slice::<T>()
    }
}

/// The scalar's storage value. The caller checked that `s` has the storage width of `T`.
pub(crate) fn scalar_to<T: Elem>(s: &Scalar) -> T {
    let mut buf = [0u8; 8];
    match s {
        Scalar::I8(v) => buf[..1].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::I16(v) => buf[..2].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::I32(v) => buf[..4].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::I64(v) => buf[..8].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::U8(v) => buf[..1].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::U16(v) => buf[..2].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::U32(v) => buf[..4].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::U64(v) => buf[..8].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::F32(v) => buf[..4].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::F64(v) => buf[..8].copy_from_slice(bytemuck::bytes_of(v)),
        Scalar::Bool(v) => buf[0] = u8::from(*v),
    }
    bytemuck::pod_read_unaligned(&buf[..size_of::<T>()])
}

/// The scalar of type `ty` for a storage value. `ty` must have the storage width of `U`.
pub(crate) fn scalar_from<U: Elem>(u: U, ty: ScalarType) -> Scalar {
    let b = bytemuck::bytes_of(&u);
    match ty {
        ScalarType::I8 => Scalar::I8(bytemuck::pod_read_unaligned(b)),
        ScalarType::I16 => Scalar::I16(bytemuck::pod_read_unaligned(b)),
        ScalarType::I32 => Scalar::I32(bytemuck::pod_read_unaligned(b)),
        ScalarType::I64 => Scalar::I64(bytemuck::pod_read_unaligned(b)),
        ScalarType::U8 => Scalar::U8(bytemuck::pod_read_unaligned(b)),
        ScalarType::U16 => Scalar::U16(bytemuck::pod_read_unaligned(b)),
        ScalarType::U32 => Scalar::U32(bytemuck::pod_read_unaligned(b)),
        ScalarType::U64 => Scalar::U64(bytemuck::pod_read_unaligned(b)),
        ScalarType::F32 => Scalar::F32(bytemuck::pod_read_unaligned(b)),
        ScalarType::F64 => Scalar::F64(bytemuck::pod_read_unaligned(b)),
        ScalarType::Bool => Scalar::Bool(b[0] != 0),
    }
}

/// Reads an argument as an operand of scalar type `ty`.
fn operand<T: Elem>(arg: &Arg, ty: ScalarType) -> Result<Operand<'_, T>, String> {
    debug_assert_eq!(T::TYPE.width(), ty.width());
    match arg {
        Arg::Rows(a) if a.elem() == ty => Ok(Operand::Slice(slice_of(a))),
        Arg::Splat(s) if s.ty() == ty => Ok(Operand::Splat(scalar_to(s))),
        Arg::Rows(a) => Err(format!("expected {0} or {0}[], got {1}", ty.name(), a.ty())),
        Arg::Splat(s) => Err(format!("expected {0} or {0}[], got {0}", s.ty().name())),
    }
}

/// The rows of an argument that must be an array of scalar type `ty`. `T` is its storage type.
pub(crate) fn rows_arg<T: Elem>(arg: &Arg, ty: ScalarType) -> Result<&[T], String> {
    debug_assert_eq!(T::TYPE.width(), ty.width());
    match arg {
        Arg::Rows(a) if a.elem() == ty => Ok(slice_of(a)),
        Arg::Rows(a) => Err(format!("expected {}[], got {}", ty.name(), a.ty())),
        Arg::Splat(s) => Err(format!("expected {}[], got {}", ty.name(), s.ty().name())),
    }
}

/// The value of an argument that must be a scalar of type `ty`. `T` is its storage type.
pub(crate) fn splat_arg<T: Elem>(arg: &Arg, ty: ScalarType) -> Result<T, String> {
    debug_assert_eq!(T::TYPE.width(), ty.width());
    match arg {
        Arg::Splat(s) if s.ty() == ty => Ok(scalar_to(s)),
        Arg::Splat(s) => Err(format!("expected {}, got {}", ty.name(), s.ty().name())),
        Arg::Rows(a) => Err(format!("expected {}, got {}", ty.name(), a.ty())),
    }
}

/// The destination bytes as rows of `U`. Fails when the bytes are not aligned for `U` or
/// not a whole number of rows.
fn out_rows<U: Elem>(out: &mut [u8]) -> Result<&mut [U], String> {
    bytemuck::try_cast_slice_mut(out).map_err(|e| format!("bad destination: {e}"))
}

pub(crate) fn arity_error(want: usize, got: usize) -> String {
    format!("expected {want} arguments, got {got}")
}

#[inline(always)]
fn run_unary<T: Elem, U: Elem>(
    args: &[Arg],
    out: &mut [u8],
    in_ty: ScalarType,
    f: impl Fn(T) -> U,
) -> Result<(), String> {
    let [a] = args else {
        return Err(arity_error(1, args.len()));
    };
    unary_into(operand::<T>(a, in_ty)?, out_rows::<U>(out)?, f);
    Ok(())
}

#[inline(always)]
fn run_binary<T: Elem, U: Elem>(
    args: &[Arg],
    out: &mut [u8],
    in_ty: ScalarType,
    f: impl Fn(T, T) -> U,
) -> Result<(), String> {
    let [a, b] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (a, b) = (operand::<T>(a, in_ty)?, operand::<T>(b, in_ty)?);
    binary_into(a, b, out_rows::<U>(out)?, f);
    Ok(())
}

#[inline(always)]
fn run_select<T: Elem>(args: &[Arg], out: &mut [u8], ty: ScalarType) -> Result<(), String> {
    let [c, t, e] = args else {
        return Err(arity_error(3, args.len()));
    };
    let c = operand::<u8>(c, ScalarType::Bool)?;
    let (t, e) = (operand::<T>(t, ty)?, operand::<T>(e, ty)?);
    select_into(c, t, e, out_rows::<T>(out)?);
    Ok(())
}

/// Registers the array kernel `name` (when every type is a v1 array type) and the
/// scalar-only kernel `{name}_s`. Both run body `B`. Only the array kernel gets ISA variants,
/// because a `_s` kernel computes one row.
fn add_entry<B: ElementwiseBody>(
    reg: &mut Registry,
    op: Op,
    name: String,
    inputs: &[ScalarType],
    out: ScalarType,
    commutative: bool,
) {
    let row_bytes: usize = inputs.iter().map(|t| t.width()).sum::<usize>() + out.width();
    let v1 = inputs
        .iter()
        .chain([&out])
        .all(|t| ScalarType::V1_ARRAY.contains(t));
    let def = |kernel_name, input_specs, output, cost, impls| KernelDef {
        name: kernel_name,
        class: KernelClass::Elementwise,
        op: Some(op),
        inputs: input_specs,
        output,
        cost,
        commutative,
        pattern: None,
        impls,
    };
    if v1 {
        let cost = u64::try_from(row_bytes).expect("row width fits in u64");
        let inputs = inputs.iter().map(|t| InputSpec::Either(*t)).collect();
        let impls = KernelImpl::elementwise::<B>();
        reg.add(def(
            name.clone(),
            inputs,
            ValueType::Array(out),
            cost,
            impls,
        ));
    }
    let inputs = inputs.iter().map(|t| InputSpec::Scalar(*t)).collect();
    let impls = KernelImpl::elementwise_scalar::<B>().scalar_only();
    reg.add(def(
        format!("{name}_s"),
        inputs,
        ValueType::Scalar(out),
        0,
        impls,
    ));
}

/// A one-operand row operation. It is a type, so that a kernel body can name it and every
/// ISA variant inlines the same operation.
pub(crate) trait UnaryOp<T, U>: 'static {
    fn apply(a: T) -> U;
}

/// A two-operand row operation, see `UnaryOp`.
pub(crate) trait BinaryOp<T, U>: 'static {
    fn apply(a: T, b: T) -> U;
}

/// The body that writes `F(a[i])` as rows of `U`, for an operand of storage type `T` whose
/// semantic type is `tag::<T, BOOL>()`.
pub(crate) struct Unary<T, U, F, const BOOL: bool>(PhantomData<fn() -> (T, U, F)>);

impl<T: Elem, U: Elem, F: UnaryOp<T, U>, const BOOL: bool> ElementwiseBody
    for Unary<T, U, F, BOOL>
{
    #[inline(always)]
    fn run(args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        run_unary::<T, U>(args, out, tag::<T, BOOL>(), F::apply)
    }
}

/// The body that writes `F(a[i], b[i])`, see `Unary`.
pub(crate) struct Binary<T, U, F, const BOOL: bool>(PhantomData<fn() -> (T, U, F)>);

impl<T: Elem, U: Elem, F: BinaryOp<T, U>, const BOOL: bool> ElementwiseBody
    for Binary<T, U, F, BOOL>
{
    #[inline(always)]
    fn run(args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        run_binary::<T, U>(args, out, tag::<T, BOOL>(), F::apply)
    }
}

/// The body of `select` on rows of storage type `T`, see `Unary`.
pub(crate) struct Select<T, const BOOL: bool>(PhantomData<fn() -> T>);

impl<T: Elem, const BOOL: bool> ElementwiseBody for Select<T, BOOL> {
    #[inline(always)]
    fn run(args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        run_select::<T>(args, out, tag::<T, BOOL>())
    }
}

/// Registers a one-operand kernel of operation `F` on input type `tag::<T, BOOL>()`.
fn unary<T: Elem, U: Elem, F: UnaryOp<T, U>, const BOOL: bool>(
    reg: &mut Registry,
    op: Op,
    name: String,
    out: ScalarType,
) {
    add_entry::<Unary<T, U, F, BOOL>>(reg, op, name, &[tag::<T, BOOL>()], out, false);
}

/// Registers a two-operand kernel, see `unary`.
fn binary<T: Elem, U: Elem, F: BinaryOp<T, U>, const BOOL: bool>(
    reg: &mut Registry,
    op: Op,
    name: String,
    commutative: bool,
    out: ScalarType,
) {
    let ty = tag::<T, BOOL>();
    add_entry::<Binary<T, U, F, BOOL>>(reg, op, name, &[ty, ty], out, commutative);
}

/// Numeric element types. Integers wrap, floats follow IEEE 754.
pub trait Num: Elem + PartialOrd {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn min2(self, o: Self) -> Self;
    fn max2(self, o: Self) -> Self;
}

macro_rules! impl_int_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            #[inline(always)]
            fn add(self, o: Self) -> Self { self.wrapping_add(o) }
            #[inline(always)]
            fn sub(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline(always)]
            fn mul(self, o: Self) -> Self { self.wrapping_mul(o) }
            #[inline(always)]
            fn min2(self, o: Self) -> Self { Ord::min(self, o) }
            #[inline(always)]
            fn max2(self, o: Self) -> Self { Ord::max(self, o) }
        }
    )*};
}

macro_rules! impl_float_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            #[inline(always)]
            fn add(self, o: Self) -> Self { self + o }
            #[inline(always)]
            fn sub(self, o: Self) -> Self { self - o }
            #[inline(always)]
            fn mul(self, o: Self) -> Self { self * o }
            /// IEEE 754-2019 `minimum`: NaN propagates and `-0.0 < +0.0`.
            #[inline(always)]
            fn min2(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    <$t>::NAN
                } else if self == o {
                    // Equal values differ only for the zeros, where the negative one is smaller.
                    if self.is_sign_negative() { self } else { o }
                } else if self < o {
                    self
                } else {
                    o
                }
            }
            /// IEEE 754-2019 `maximum`: NaN propagates and `+0.0 > -0.0`.
            #[inline(always)]
            fn max2(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() {
                    <$t>::NAN
                } else if self == o {
                    if self.is_sign_positive() { self } else { o }
                } else if self > o {
                    self
                } else {
                    o
                }
            }
        }
    )*};
}

impl_int_num!(i8, i16, i32, i64, u8, u16, u32, u64);
impl_float_num!(f32, f64);

/// Signed integers and floats.
trait Signed: Num {
    fn neg(self) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_int_signed {
    ($($t:ty),*) => {$(
        impl Signed for $t {
            #[inline(always)]
            fn neg(self) -> Self { self.wrapping_neg() }
            // `MIN.abs()` wraps to `MIN`.
            #[inline(always)]
            fn abs(self) -> Self { self.wrapping_abs() }
        }
    )*};
}

macro_rules! impl_float_signed {
    ($($t:ty),*) => {$(
        impl Signed for $t {
            #[inline(always)]
            fn neg(self) -> Self { -self }
            #[inline(always)]
            fn abs(self) -> Self { <$t>::abs(self) }
        }
    )*};
}

impl_int_signed!(i8, i16, i32, i64);
impl_float_signed!(f32, f64);

trait Float: Signed {
    fn div(self, o: Self) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_float {
    ($($t:ty),*) => {$(
        impl Float for $t {
            #[inline(always)]
            fn div(self, o: Self) -> Self { self / o }
            #[inline(always)]
            fn sqrt(self) -> Self { <$t>::sqrt(self) }
        }
    )*};
}

impl_float!(f32, f64);

/// Defines the operation type `$name` for every element type `$t` with the bounds in brackets.
macro_rules! op {
    (unary $name:ident, $t:ident: [$($bound:tt)*], $u:ty, |$a:ident| $e:expr) => {
        pub(crate) struct $name;

        impl<$t: $($bound)*> UnaryOp<$t, $u> for $name {
            #[inline(always)]
            fn apply($a: $t) -> $u {
                $e
            }
        }
    };
    (binary $name:ident, $t:ident: [$($bound:tt)*], $u:ty, |$a:ident, $b:ident| $e:expr) => {
        pub(crate) struct $name;

        impl<$t: $($bound)*> BinaryOp<$t, $u> for $name {
            #[inline(always)]
            fn apply($a: $t, $b: $t) -> $u {
                $e
            }
        }
    };
}

op!(binary AddOp, T: [Num], T, |a, b| a.add(b));
op!(binary SubOp, T: [Num], T, |a, b| a.sub(b));
op!(binary MulOp, T: [Num], T, |a, b| a.mul(b));
op!(binary MinOp, T: [Num], T, |a, b| a.min2(b));
op!(binary MaxOp, T: [Num], T, |a, b| a.max2(b));
op!(binary LtOp, T: [Num], u8, |a, b| u8::from(a < b));
op!(binary LeOp, T: [Num], u8, |a, b| u8::from(a <= b));
op!(binary GtOp, T: [Num], u8, |a, b| u8::from(a > b));
op!(binary GeOp, T: [Num], u8, |a, b| u8::from(a >= b));
op!(unary NegOp, T: [Signed], T, |a| a.neg());
op!(unary AbsOp, T: [Signed], T, |a| a.abs());
op!(binary DivOp, T: [Float], T, |a, b| a.div(b));
op!(unary SqrtOp, T: [Float], T, |a| a.sqrt());
op!(binary EqOp, T: [Elem + PartialEq], u8, |a, b| u8::from(a == b));
op!(binary NeOp, T: [Elem + PartialEq], u8, |a, b| u8::from(a != b));
// The bool operations work on the 0/1 storage, so `T` is always `u8`.
op!(binary AndOp, T: [BoolStorage], u8, |a, b| a.byte() & b.byte());
op!(binary OrOp, T: [BoolStorage], u8, |a, b| a.byte() | b.byte());
// Bool storage is 0 or 1.
op!(unary NotOp, T: [BoolStorage], u8, |a| a.byte() ^ 1);

/// The storage type of bool, `u8`.
trait BoolStorage: Elem {
    fn byte(self) -> u8;
}

impl BoolStorage for u8 {
    #[inline(always)]
    fn byte(self) -> u8 {
        self
    }
}

/// Conversion of storage type `Self` to `U` by Rust `as`.
trait CastTo<U> {
    fn cast(self) -> U;
}

/// `as` from `$f` to each listed type.
macro_rules! impl_cast_to {
    ($f:ty => $($t:ty),*) => {$(
        impl CastTo<$t> for $f {
            #[allow(trivial_numeric_casts, clippy::unnecessary_cast)]
            #[inline(always)]
            fn cast(self) -> $t {
                self as $t
            }
        }
    )*};
}

/// `as` from each listed type to every numeric type.
macro_rules! impl_cast_from {
    ($($f:ty),*) => {$(
        impl_cast_to!($f => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
    )*};
}

impl_cast_from!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

/// Casts follow Rust `as`: float to int saturates and maps NaN to zero, int to int truncates.
/// Bool sources use their `u8` storage.
pub(crate) struct CastOp;

impl<T: CastTo<U> + 'static, U> UnaryOp<T, U> for CastOp {
    #[inline(always)]
    fn apply(a: T) -> U {
        a.cast()
    }
}

// The kernel bodies are types, so element types and operations are type arguments, and every
// ISA variant instantiates the same body as the scalar entry.

fn register_num<T: Num>(reg: &mut Registry) {
    let n = T::TYPE.name();
    let (t, b) = (T::TYPE, ScalarType::Bool);
    binary::<T, T, AddOp, false>(reg, Op::Add, format!("add_{n}"), true, t);
    binary::<T, T, SubOp, false>(reg, Op::Sub, format!("sub_{n}"), false, t);
    binary::<T, T, MulOp, false>(reg, Op::Mul, format!("mul_{n}"), true, t);
    binary::<T, T, MinOp, false>(reg, Op::Min, format!("min_{n}"), true, t);
    binary::<T, T, MaxOp, false>(reg, Op::Max, format!("max_{n}"), true, t);
    binary::<T, u8, LtOp, false>(reg, Op::Lt, format!("lt_{n}"), false, b);
    binary::<T, u8, LeOp, false>(reg, Op::Le, format!("le_{n}"), false, b);
    binary::<T, u8, GtOp, false>(reg, Op::Gt, format!("gt_{n}"), false, b);
    binary::<T, u8, GeOp, false>(reg, Op::Ge, format!("ge_{n}"), false, b);
}

fn register_signed<T: Signed>(reg: &mut Registry) {
    let n = T::TYPE.name();
    unary::<T, T, NegOp, false>(reg, Op::Neg, format!("neg_{n}"), T::TYPE);
    unary::<T, T, AbsOp, false>(reg, Op::Abs, format!("abs_{n}"), T::TYPE);
}

fn register_float<T: Float>(reg: &mut Registry) {
    let n = T::TYPE.name();
    binary::<T, T, DivOp, false>(reg, Op::Div, format!("div_{n}"), false, T::TYPE);
    unary::<T, T, SqrtOp, false>(reg, Op::Sqrt, format!("sqrt_{n}"), T::TYPE);
}

/// The semantic type of storage type `T`. Bool is stored as `u8`, so `BOOL` selects it.
pub(crate) fn tag<T: Elem, const BOOL: bool>() -> ScalarType {
    if BOOL { ScalarType::Bool } else { T::TYPE }
}

/// Equality and select exist for every type, with `BOOL` marking the bool instantiation on `u8`.
fn register_any<T: Elem + PartialEq, const BOOL: bool>(reg: &mut Registry) {
    let ty = tag::<T, BOOL>();
    let n = ty.name();
    let b = ScalarType::Bool;
    binary::<T, u8, EqOp, BOOL>(reg, Op::Eq, format!("eq_{n}"), true, b);
    binary::<T, u8, NeOp, BOOL>(reg, Op::Ne, format!("ne_{n}"), true, b);
    add_entry::<Select<T, BOOL>>(
        reg,
        Op::Select,
        format!("select_{n}"),
        &[ScalarType::Bool, ty, ty],
        ty,
        false,
    );
}

fn register_bool(reg: &mut Registry) {
    register_any::<u8, true>(reg);
    let b = ScalarType::Bool;
    binary::<u8, u8, AndOp, true>(reg, Op::And, "and_bool".into(), true, b);
    binary::<u8, u8, OrOp, true>(reg, Op::Or, "or_bool".into(), true, b);
    // Bool storage is 0 or 1.
    unary::<u8, u8, NotOp, true>(reg, Op::Not, "not_bool".into(), b);
}

/// Casts follow Rust `as`: float to int saturates and maps NaN to zero, int to int truncates.
/// Bool sources use their `u8` storage.
macro_rules! cast_one {
    ($reg:expr, $f:ty, $fty:ident, $t:ty, $tty:ident) => {
        unary::<$f, $t, CastOp, { matches!(ScalarType::$fty, ScalarType::Bool) }>(
            $reg,
            Op::Cast(ScalarType::$tty),
            format!(
                "cast_{}_{}",
                ScalarType::$fty.name(),
                ScalarType::$tty.name()
            ),
            ScalarType::$tty,
        )
    };
}

/// One source type cast to every numeric type.
macro_rules! cast_row {
    ($reg:expr, $f:ty, $fty:ident) => {
        cast_one!($reg, $f, $fty, i8, I8);
        cast_one!($reg, $f, $fty, i16, I16);
        cast_one!($reg, $f, $fty, i32, I32);
        cast_one!($reg, $f, $fty, i64, I64);
        cast_one!($reg, $f, $fty, u8, U8);
        cast_one!($reg, $f, $fty, u16, U16);
        cast_one!($reg, $f, $fty, u32, U32);
        cast_one!($reg, $f, $fty, u64, U64);
        cast_one!($reg, $f, $fty, f32, F32);
        cast_one!($reg, $f, $fty, f64, F64);
    };
}

fn register_casts(reg: &mut Registry) {
    cast_row!(reg, i8, I8);
    cast_row!(reg, i16, I16);
    cast_row!(reg, i32, I32);
    cast_row!(reg, i64, I64);
    cast_row!(reg, u8, U8);
    cast_row!(reg, u16, U16);
    cast_row!(reg, u32, U32);
    cast_row!(reg, u64, U64);
    cast_row!(reg, f32, F32);
    cast_row!(reg, f64, F64);
    cast_row!(reg, u8, Bool);
}

/// Adds every elementwise kernel to the registry.
pub(crate) fn register(reg: &mut Registry) {
    register_num::<i8>(reg);
    register_num::<i16>(reg);
    register_num::<i32>(reg);
    register_num::<i64>(reg);
    register_num::<u8>(reg);
    register_num::<u16>(reg);
    register_num::<u32>(reg);
    register_num::<u64>(reg);
    register_num::<f32>(reg);
    register_num::<f64>(reg);
    register_signed::<i8>(reg);
    register_signed::<i16>(reg);
    register_signed::<i32>(reg);
    register_signed::<i64>(reg);
    register_signed::<f32>(reg);
    register_signed::<f64>(reg);
    register_float::<f32>(reg);
    register_float::<f64>(reg);
    register_any::<i8, false>(reg);
    register_any::<i16, false>(reg);
    register_any::<i32, false>(reg);
    register_any::<i64, false>(reg);
    register_any::<u8, false>(reg);
    register_any::<u16, false>(reg);
    register_any::<u32, false>(reg);
    register_any::<u64, false>(reg);
    register_any::<f32, false>(reg);
    register_any::<f64, false>(reg);
    register_bool(reg);
    register_casts(reg);
}
