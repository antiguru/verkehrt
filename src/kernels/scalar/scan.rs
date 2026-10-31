//! Scalar reference scan kernels. A scan keeps state across the tiles of one array, so a tiled
//! executor feeds it tile by tile and the output never depends on where the tiles split.

use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, BitXor, Not};

use crate::kernels::scalar::elementwise::{Num, arity_error, out_rows, rows_arg, splat_arg, tag};
use crate::kernels::scalar::index::ReduceInto;
use crate::kernels::{InputSpec, KernelClass, KernelDef, KernelImpl, Registry, ScanBody, Scanner};
use crate::ops::{Op, ReduceOp};
use crate::value::{Arg, Elem, ScalarType, ValueType};

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

/// The fold of one scan over rows of storage type `T`. The first row of a segment is `lift`ed
/// and every later row is `step`ped into the running value, so a scan needs no identity.
/// `reduce_into` has empty buckets and starts every bucket at the `identity` instead.
pub trait ScanOp<T: Elem>: Send + 'static {
    /// The storage type of the output rows.
    type Out: Elem;

    /// The semantic type of the input rows, whose storage type is `T`.
    const IN_TYPE: ScalarType;

    /// The semantic type of the output rows, whose storage type is `Out`.
    const OUT_TYPE: ScalarType;

    fn lift(x: T) -> Self::Out;

    fn step(acc: Self::Out, x: T) -> Self::Out;

    /// The value that `step` leaves unchanged on the left, so that `step(identity, x)` equals
    /// `lift(x)` for every `x` and an empty bucket reads as the fold of no rows.
    fn identity() -> Self::Out;
}

/// The extreme values of an ordered element type: the smallest and largest value of an
/// integer, `-inf` and `+inf` for a float. They are the identities of `max` and `min`.
pub trait Extremes: Num {
    const LOWEST: Self;
    const HIGHEST: Self;
}

macro_rules! impl_extremes {
    ($($t:ty => $lo:expr, $hi:expr);*) => {$(
        impl Extremes for $t {
            const LOWEST: $t = $lo;
            const HIGHEST: $t = $hi;
        }
    )*};
}

impl_extremes!(
    i64 => i64::MIN, i64::MAX;
    u64 => u64::MIN, u64::MAX;
    i32 => i32::MIN, i32::MAX;
    u8 => u8::MIN, u8::MAX;
    f64 => f64::NEG_INFINITY, f64::INFINITY;
    f32 => f32::NEG_INFINITY, f32::INFINITY
);

/// Wrapping sum in the widened type `W`, as `IntSum` in the reductions.
pub struct ScanSum<W>(PhantomData<fn() -> W>);

/// Wrapping product in the element type, as `reduce(x, mul)`.
pub struct ScanProduct;

/// `min` (`MAX` false) or `max` (`MAX` true), with the IEEE semantics of `reduce_min` and
/// `reduce_max` on floats.
pub struct ScanExtreme<const MAX: bool>;

/// Bitwise `and` over integers, or logical `and` over the 0/1 bytes of bool when `BOOL`.
pub struct ScanAnd<const BOOL: bool>;

/// Bitwise `or`, see `ScanAnd`.
pub struct ScanOr<const BOOL: bool>;

/// Bitwise `xor`, see `ScanAnd`.
pub struct ScanXor<const BOOL: bool>;

impl<T: Elem + Into<W>, W: Num> ScanOp<T> for ScanSum<W> {
    type Out = W;
    const IN_TYPE: ScalarType = T::TYPE;
    const OUT_TYPE: ScalarType = W::TYPE;

    #[inline(always)]
    fn lift(x: T) -> W {
        x.into()
    }

    #[inline(always)]
    fn step(acc: W, x: T) -> W {
        Num::add(acc, x.into())
    }

    #[inline(always)]
    fn identity() -> W {
        W::zeroed()
    }
}

impl<T: Num + From<u8>> ScanOp<T> for ScanProduct {
    type Out = T;
    const IN_TYPE: ScalarType = T::TYPE;
    const OUT_TYPE: ScalarType = T::TYPE;

    #[inline(always)]
    fn lift(x: T) -> T {
        x
    }

    #[inline(always)]
    fn step(acc: T, x: T) -> T {
        Num::mul(acc, x)
    }

    #[inline(always)]
    fn identity() -> T {
        T::from(1)
    }
}

impl<T: Extremes, const MAX: bool> ScanOp<T> for ScanExtreme<MAX> {
    type Out = T;
    const IN_TYPE: ScalarType = T::TYPE;
    const OUT_TYPE: ScalarType = T::TYPE;

    #[inline(always)]
    fn lift(x: T) -> T {
        x
    }

    #[inline(always)]
    fn step(acc: T, x: T) -> T {
        if MAX { acc.max2(x) } else { acc.min2(x) }
    }

    #[inline(always)]
    fn identity() -> T {
        if MAX { T::LOWEST } else { T::HIGHEST }
    }
}

macro_rules! impl_bit_scan {
    ($name:ident, $bound:ident, |$a:ident, $x:ident| $step:expr, $identity:expr) => {
        impl<T: Elem + $bound<Output = T> + Not<Output = T> + From<u8>, const BOOL: bool> ScanOp<T>
            for $name<BOOL>
        {
            type Out = T;
            const IN_TYPE: ScalarType = if BOOL { ScalarType::Bool } else { T::TYPE };
            const OUT_TYPE: ScalarType = if BOOL { ScalarType::Bool } else { T::TYPE };

            #[inline(always)]
            fn lift(x: T) -> T {
                x
            }

            #[inline(always)]
            fn step($a: T, $x: T) -> T {
                $step
            }

            #[inline(always)]
            fn identity() -> T {
                $identity
            }
        }
    };
}

// All ones, which for bool is the single true bit of the 0/1 byte.
impl_bit_scan!(
    ScanAnd,
    BitAnd,
    |a, x| a & x,
    if BOOL { T::from(1) } else { !T::zeroed() }
);
impl_bit_scan!(ScanOr, BitOr, |a, x| a | x, T::zeroed());
impl_bit_scan!(ScanXor, BitXor, |a, x| a ^ x, T::zeroed());

/// The checked rows of one tile: the input, the output, and the segment starts if any.
pub(crate) type Tile<'a, T, U> = (&'a [T], &'a mut [U], Option<&'a [u8]>);

/// Reads the arguments of one tile. `BY` selects `scan_by`, which takes a `bool` operand of
/// segment starts after the values.
#[inline(always)]
fn tile<'a, T: Elem, O: ScanOp<T>, const BY: bool>(
    args: &'a [Arg],
    out: &'a mut [u8],
) -> Result<Tile<'a, T, O::Out>, String> {
    tile_rows::<T, O::Out, BY>(args, out, O::IN_TYPE)
}

/// `tile` for input rows of semantic type `in_ty` with storage type `T` and output rows of
/// storage type `U`. It fails exactly when `tile` fails for a fold with these types.
#[inline(always)]
pub(crate) fn tile_rows<'a, T: Elem, U: Elem, const BY: bool>(
    args: &'a [Arg],
    out: &'a mut [u8],
    in_ty: ScalarType,
) -> Result<Tile<'a, T, U>, String> {
    let (xs, starts) = match args {
        [x] if !BY => (rows_arg::<T>(x, in_ty)?, None),
        [x, s] if BY => (
            rows_arg::<T>(x, in_ty)?,
            Some(rows_arg::<u8>(s, ScalarType::Bool)?),
        ),
        _ => return Err(arity_error(1 + usize::from(BY), args.len())),
    };
    let out = out_rows::<U>(out)?;
    if out.len() != xs.len() || starts.is_some_and(|s| s.len() != xs.len()) {
        return Err(format!(
            "length mismatch: {} rows for a destination of {} rows",
            xs.len(),
            out.len()
        ));
    }
    Ok((xs, out, starts))
}

/// A scan of rows of storage type `T` with the fold `O`. `BY` makes it `scan_by`, whose
/// segments restart at the true rows of the second operand.
pub struct Scan<T: Elem, O: ScanOp<T>, const BY: bool> {
    /// The output of the last row of the tiles consumed so far, `None` before the first row.
    /// Row 0 of the array starts a segment whatever `starts` says, and a later tile continues
    /// the segment of its predecessor unless `starts` says otherwise.
    acc: Option<O::Out>,
    _rows: PhantomData<fn() -> (T, O)>,
}

impl<T: Elem, O: ScanOp<T>, const BY: bool> Scanner for Scan<T, O, BY> {
    #[inline(always)]
    fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        let (xs, out, starts) = tile::<T, O, BY>(args, out)?;
        let mut rows = xs.iter().zip(out.iter_mut()).enumerate();
        let mut acc = match self.acc {
            Some(acc) => acc,
            None => {
                let Some((_, (&x, o))) = rows.next() else {
                    return Ok(());
                };
                *o = O::lift(x);
                *o
            }
        };
        for (i, (&x, o)) in rows {
            acc = match starts {
                Some(s) if s[i] != 0 => O::lift(x),
                _ => O::step(acc, x),
            };
            *o = acc;
        }
        self.acc = Some(acc);
        Ok(())
    }
}

impl<T: Elem, O: ScanOp<T>, const BY: bool> ScanBody for Scan<T, O, BY> {
    fn new() -> Self {
        Scan {
            acc: None,
            _rows: PhantomData,
        }
    }
}

/// Registers `scan_<op>_<t>`, `scan_by_<op>_<t>`, and `reduce_into_<op>_<t>` for the fold `O`
/// over rows of storage type `T`.
fn register_scan<T: Elem, O: ScanOp<T>>(reg: &mut Registry, r: ReduceOp) {
    let op = format!("{r:?}").to_lowercase();
    let n = O::IN_TYPE.name();
    let width = |t: ScalarType| u64::try_from(t.width()).expect("width fits in u64");
    // One row read and one row written, and the starts byte for `scan_by`.
    let cost = width(O::IN_TYPE) + width(O::OUT_TYPE);
    reg.add(KernelDef {
        name: format!("scan_{op}_{n}"),
        class: KernelClass::Scan,
        op: Some(Op::Scan(r)),
        inputs: vec![InputSpec::Array(O::IN_TYPE)],
        output: ValueType::Array(O::OUT_TYPE),
        cost,
        commutative: false,
        pattern: None,
        impls: KernelImpl::scan::<Scan<T, O, false>>(),
    });
    let mut by_impls = KernelImpl::scan::<Scan<T, O, true>>();
    // Only wrapping sums that keep the element type have a segmented intrinsics entry.
    if r == ReduceOp::Add && O::OUT_TYPE == O::IN_TYPE {
        crate::kernels::arch::add_scan_by_add::<T>(&mut by_impls);
    }
    reg.add(KernelDef {
        name: format!("scan_by_{op}_{n}"),
        class: KernelClass::Scan,
        op: Some(Op::ScanBy(r)),
        inputs: vec![
            InputSpec::Array(O::IN_TYPE),
            InputSpec::Array(ScalarType::Bool),
        ],
        output: ValueType::Array(O::OUT_TYPE),
        cost: cost + 1,
        commutative: false,
        pattern: None,
        impls: by_impls,
    });
    reg.add(KernelDef {
        name: format!("reduce_into_{op}_{n}"),
        class: KernelClass::Whole,
        op: Some(Op::ReduceInto(r)),
        inputs: vec![
            InputSpec::Array(ScalarType::I64),
            InputSpec::Array(O::IN_TYPE),
            InputSpec::Scalar(ScalarType::I64),
        ],
        output: ValueType::Array(O::OUT_TYPE),
        // One key and one value read, and one bucket read and written.
        cost: 8 + width(O::IN_TYPE) + 2 * width(O::OUT_TYPE),
        commutative: false,
        pattern: None,
        impls: KernelImpl::whole::<ReduceInto<T, O>>(),
    });
}

/// The scans of an integer type `T` whose sums widen to `W`, as `reduce(x, add)`.
fn register_int<
    T: Extremes
        + Into<W>
        + BitAnd<Output = T>
        + BitOr<Output = T>
        + BitXor<Output = T>
        + Not<Output = T>
        + From<u8>,
    W: Num + From<u8>,
>(
    reg: &mut Registry,
) {
    register_scan::<T, ScanSum<W>>(reg, ReduceOp::Add);
    register_scan::<T, ScanProduct>(reg, ReduceOp::Mul);
    register_scan::<T, ScanExtreme<false>>(reg, ReduceOp::Min);
    register_scan::<T, ScanExtreme<true>>(reg, ReduceOp::Max);
    register_scan::<T, ScanAnd<false>>(reg, ReduceOp::And);
    register_scan::<T, ScanOr<false>>(reg, ReduceOp::Or);
    register_scan::<T, ScanXor<false>>(reg, ReduceOp::Xor);
}

/// Floats have no scan for `add` and `mul`, whose results would depend on the tiling.
fn register_float<T: Extremes>(reg: &mut Registry) {
    register_scan::<T, ScanExtreme<false>>(reg, ReduceOp::Min);
    register_scan::<T, ScanExtreme<true>>(reg, ReduceOp::Max);
}

/// Bool scans read the 0/1 byte storage.
fn register_bool(reg: &mut Registry) {
    register_scan::<u8, ScanAnd<true>>(reg, ReduceOp::And);
    register_scan::<u8, ScanOr<true>>(reg, ReduceOp::Or);
    register_scan::<u8, ScanXor<true>>(reg, ReduceOp::Xor);
}

pub(crate) fn register(reg: &mut Registry) {
    register_int::<i64, i64>(reg);
    register_int::<u64, u64>(reg);
    register_int::<i32, i64>(reg);
    register_int::<u8, u64>(reg);
    register_float::<f64>(reg);
    register_float::<f32>(reg);
    register_bool(reg);
    register_prev::<f64, false>(reg);
    register_prev::<f32, false>(reg);
    register_prev::<i64, false>(reg);
    register_prev::<u64, false>(reg);
    register_prev::<i32, false>(reg);
    register_prev::<u8, false>(reg);
    register_prev::<u8, true>(reg);
}
