//! Scalar reference implementations of the single-op elementwise kernels.
//!
//! Every kernel entry is a thin wrapper over a slice function (`unary_into`, `binary_into`,
//! `select_into`) that fills exactly `out.len()` rows. A tiled executor calls the same
//! entry once per tile with `Arg::Rows` windows. The scalar-only `_s` kernels share the
//! entry of their array kernel, because a one-row destination with `Splat` operands is the
//! same computation.

use std::ops::Range;

use crate::kernels::{ElementwiseFn, InputSpec, KernelClass, KernelDef, KernelImpl, Registry};
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

    fn at(&self, i: usize) -> T {
        match self {
            Operand::Slice(s) => s[i],
            Operand::Splat(v) => *v,
        }
    }
}

/// Writes `f(a[i], b[i])` for every row of `out`. Slice operands must have `out.len()` rows.
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

/// The destination bytes as rows of `U`. Fails when the bytes are not aligned for `U` or
/// not a whole number of rows.
fn out_rows<U: Elem>(out: &mut [u8]) -> Result<&mut [U], String> {
    bytemuck::try_cast_slice_mut(out).map_err(|e| format!("bad destination: {e}"))
}

pub(crate) fn arity_error(want: usize, got: usize) -> String {
    format!("expected {want} arguments, got {got}")
}

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
/// scalar-only kernel `{name}_s`.
#[allow(clippy::too_many_arguments)]
fn add_entry(
    reg: &mut Registry,
    op: Op,
    name: String,
    inputs: &[ScalarType],
    out: ScalarType,
    commutative: bool,
    entry: ElementwiseFn,
) {
    let row_bytes: usize = inputs.iter().map(|t| t.width()).sum::<usize>() + out.width();
    let v1 = inputs
        .iter()
        .chain([&out])
        .all(|t| ScalarType::V1_ARRAY.contains(t));
    let def = |kernel_name, input_specs, output, cost| KernelDef {
        name: kernel_name,
        class: KernelClass::Elementwise,
        op: Some(op),
        inputs: input_specs,
        output,
        cost,
        commutative,
        pattern: None,
        impls: KernelImpl::Elementwise(entry).scalar_only(),
    };
    if v1 {
        let cost = u64::try_from(row_bytes).expect("row width fits in u64");
        let inputs = inputs.iter().map(|t| InputSpec::Either(*t)).collect();
        reg.add(def(name.clone(), inputs, ValueType::Array(out), cost));
    }
    let inputs = inputs.iter().map(|t| InputSpec::Scalar(*t)).collect();
    reg.add(def(format!("{name}_s"), inputs, ValueType::Scalar(out), 0));
}

/// Registers a one-operand kernel. `$f` is a closure literal, expanded into the entry
/// because an `ElementwiseFn` is a bare function pointer and cannot capture it.
macro_rules! unary {
    ($reg:expr, $op:expr, $name:expr, $in:expr, $out:expr, $t:ty, $o:ty, $f:expr) => {
        add_entry(
            $reg,
            $op,
            $name,
            &[$in],
            $out,
            false,
            |args: &[Arg], out: &mut [u8]| run_unary::<$t, $o>(args, out, $in, $f),
        )
    };
}

/// Registers a two-operand kernel, see `unary!`.
macro_rules! binary {
    ($reg:expr, $op:expr, $name:expr, $comm:expr, $in:expr, $out:expr, $t:ty, $o:ty, $f:expr) => {
        add_entry(
            $reg,
            $op,
            $name,
            &[$in, $in],
            $out,
            $comm,
            |args: &[Arg], out: &mut [u8]| run_binary::<$t, $o>(args, out, $in, $f),
        )
    };
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
            fn add(self, o: Self) -> Self { self.wrapping_add(o) }
            fn sub(self, o: Self) -> Self { self.wrapping_sub(o) }
            fn mul(self, o: Self) -> Self { self.wrapping_mul(o) }
            fn min2(self, o: Self) -> Self { Ord::min(self, o) }
            fn max2(self, o: Self) -> Self { Ord::max(self, o) }
        }
    )*};
}

macro_rules! impl_float_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            fn add(self, o: Self) -> Self { self + o }
            fn sub(self, o: Self) -> Self { self - o }
            fn mul(self, o: Self) -> Self { self * o }
            /// IEEE 754-2019 `minimum`: NaN propagates and `-0.0 < +0.0`.
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
            fn neg(self) -> Self { self.wrapping_neg() }
            // `MIN.abs()` wraps to `MIN`.
            fn abs(self) -> Self { self.wrapping_abs() }
        }
    )*};
}

macro_rules! impl_float_signed {
    ($($t:ty),*) => {$(
        impl Signed for $t {
            fn neg(self) -> Self { -self }
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
            fn div(self, o: Self) -> Self { self / o }
            fn sqrt(self) -> Self { <$t>::sqrt(self) }
        }
    )*};
}

impl_float!(f32, f64);

// The kernel macros embed their type arguments in closures that must coerce to bare function
// pointers, so those arguments are constant expressions such as `T::TYPE`, never locals.

fn register_num<T: Num>(reg: &mut Registry) {
    let n = T::TYPE.name();
    binary!(
        reg,
        Op::Add,
        format!("add_{n}"),
        true,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.add(b)
    );
    binary!(
        reg,
        Op::Sub,
        format!("sub_{n}"),
        false,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.sub(b)
    );
    binary!(
        reg,
        Op::Mul,
        format!("mul_{n}"),
        true,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.mul(b)
    );
    binary!(
        reg,
        Op::Min,
        format!("min_{n}"),
        true,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.min2(b)
    );
    binary!(
        reg,
        Op::Max,
        format!("max_{n}"),
        true,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.max2(b)
    );
    binary!(
        reg,
        Op::Lt,
        format!("lt_{n}"),
        false,
        T::TYPE,
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a < b)
    );
    binary!(
        reg,
        Op::Le,
        format!("le_{n}"),
        false,
        T::TYPE,
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a <= b)
    );
    binary!(
        reg,
        Op::Gt,
        format!("gt_{n}"),
        false,
        T::TYPE,
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a > b)
    );
    binary!(
        reg,
        Op::Ge,
        format!("ge_{n}"),
        false,
        T::TYPE,
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a >= b)
    );
}

fn register_signed<T: Signed>(reg: &mut Registry) {
    let n = T::TYPE.name();
    unary!(
        reg,
        Op::Neg,
        format!("neg_{n}"),
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T| a.neg()
    );
    unary!(
        reg,
        Op::Abs,
        format!("abs_{n}"),
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T| a.abs()
    );
}

fn register_float<T: Float>(reg: &mut Registry) {
    let n = T::TYPE.name();
    binary!(
        reg,
        Op::Div,
        format!("div_{n}"),
        false,
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T, b: T| a.div(b)
    );
    unary!(
        reg,
        Op::Sqrt,
        format!("sqrt_{n}"),
        T::TYPE,
        T::TYPE,
        T,
        T,
        |a: T| a.sqrt()
    );
}

/// The semantic type of storage type `T`. Bool is stored as `u8`, so `BOOL` selects it.
fn tag<T: Elem, const BOOL: bool>() -> ScalarType {
    if BOOL { ScalarType::Bool } else { T::TYPE }
}

/// Equality and select exist for every type, with `BOOL` marking the bool instantiation on `u8`.
fn register_any<T: Elem + PartialEq, const BOOL: bool>(reg: &mut Registry) {
    let n = tag::<T, BOOL>().name();
    binary!(
        reg,
        Op::Eq,
        format!("eq_{n}"),
        true,
        tag::<T, BOOL>(),
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a == b)
    );
    binary!(
        reg,
        Op::Ne,
        format!("ne_{n}"),
        true,
        tag::<T, BOOL>(),
        ScalarType::Bool,
        T,
        u8,
        |a: T, b: T| u8::from(a != b)
    );
    add_entry(
        reg,
        Op::Select,
        format!("select_{n}"),
        &[ScalarType::Bool, tag::<T, BOOL>(), tag::<T, BOOL>()],
        tag::<T, BOOL>(),
        false,
        |args: &[Arg], out: &mut [u8]| run_select::<T>(args, out, tag::<T, BOOL>()),
    );
}

fn register_bool(reg: &mut Registry) {
    register_any::<u8, true>(reg);
    binary!(
        reg,
        Op::And,
        "and_bool".into(),
        true,
        ScalarType::Bool,
        ScalarType::Bool,
        u8,
        u8,
        |a: u8, b: u8| a & b
    );
    binary!(
        reg,
        Op::Or,
        "or_bool".into(),
        true,
        ScalarType::Bool,
        ScalarType::Bool,
        u8,
        u8,
        |a: u8, b: u8| a | b
    );
    // Bool storage is 0 or 1.
    unary!(
        reg,
        Op::Not,
        "not_bool".into(),
        ScalarType::Bool,
        ScalarType::Bool,
        u8,
        u8,
        |a: u8| a ^ 1
    );
}

/// Casts follow Rust `as`: float to int saturates and maps NaN to zero, int to int truncates.
/// Bool sources use their `u8` storage.
#[allow(trivial_numeric_casts, clippy::unnecessary_cast)]
macro_rules! cast_one {
    ($reg:expr, $f:ty, $fty:ident, $t:ty, $tty:ident) => {
        add_entry(
            $reg,
            Op::Cast(ScalarType::$tty),
            format!(
                "cast_{}_{}",
                ScalarType::$fty.name(),
                ScalarType::$tty.name()
            ),
            &[ScalarType::$fty],
            ScalarType::$tty,
            false,
            |args: &[Arg], out: &mut [u8]| {
                run_unary::<$f, $t>(args, out, ScalarType::$fty, |a| a as $t)
            },
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

#[allow(trivial_numeric_casts, clippy::unnecessary_cast)]
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
