//! The operations on single scalars that every elementwise primitive applies per element.
//!
//! Integers wrap. Floats follow IEEE 754, with `min` and `max` as the 2019 `minimum` and
//! `maximum`. Casts follow Rust `as`. Bool operations are logical, integer ones bitwise.

use std::cmp::Ordering;

use verkehrt::ops::Op;
use verkehrt::value::{Scalar, ScalarType};

/// Applies `$e` to two scalars of one integer type, binding their values to `$x` and `$y`.
macro_rules! int_pair {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {
        match ($a, $b) {
            (Scalar::I8(x), Scalar::I8(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::I8($e)
            }
            (Scalar::I16(x), Scalar::I16(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::I16($e)
            }
            (Scalar::I32(x), Scalar::I32(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::I32($e)
            }
            (Scalar::I64(x), Scalar::I64(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::I64($e)
            }
            (Scalar::U8(x), Scalar::U8(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::U8($e)
            }
            (Scalar::U16(x), Scalar::U16(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::U16($e)
            }
            (Scalar::U32(x), Scalar::U32(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::U32($e)
            }
            (Scalar::U64(x), Scalar::U64(y)) => {
                let ($x, $y) = (*x, *y);
                Scalar::U64($e)
            }
            (a, b) => panic!("no integer operation on {a:?} and {b:?}"),
        }
    };
}

/// IEEE 754-2019 `minimum` (`MAX` false) or `maximum` (`MAX` true) on one float type.
macro_rules! ieee_extreme {
    ($name:ident, $t:ty) => {
        fn $name(x: $t, y: $t, max: bool) -> $t {
            if x.is_nan() || y.is_nan() {
                <$t>::NAN
            } else if x == y {
                // Only the zeros are equal and distinct, and `-0.0` ranks below `+0.0`.
                if x.is_sign_negative() == max { y } else { x }
            } else if (x < y) != max {
                x
            } else {
                y
            }
        }
    };
}

ieee_extreme!(extreme_f32, f32);
ieee_extreme!(extreme_f64, f64);

/// The order of two scalars of one type. Floats order by IEEE comparison, so NaN is unordered
/// and the zeros are equal.
pub fn compare(a: &Scalar, b: &Scalar) -> Option<Ordering> {
    match (a, b) {
        (Scalar::I8(x), Scalar::I8(y)) => Some(x.cmp(y)),
        (Scalar::I16(x), Scalar::I16(y)) => Some(x.cmp(y)),
        (Scalar::I32(x), Scalar::I32(y)) => Some(x.cmp(y)),
        (Scalar::I64(x), Scalar::I64(y)) => Some(x.cmp(y)),
        (Scalar::U8(x), Scalar::U8(y)) => Some(x.cmp(y)),
        (Scalar::U16(x), Scalar::U16(y)) => Some(x.cmp(y)),
        (Scalar::U32(x), Scalar::U32(y)) => Some(x.cmp(y)),
        (Scalar::U64(x), Scalar::U64(y)) => Some(x.cmp(y)),
        (Scalar::F32(x), Scalar::F32(y)) => x.partial_cmp(y),
        (Scalar::F64(x), Scalar::F64(y)) => x.partial_cmp(y),
        (Scalar::Bool(x), Scalar::Bool(y)) => Some(x.cmp(y)),
        _ => panic!("cannot compare {a:?} and {b:?}"),
    }
}

/// Whether the scalar is a float NaN.
pub fn is_nan(a: &Scalar) -> bool {
    match a {
        Scalar::F32(x) => x.is_nan(),
        Scalar::F64(x) => x.is_nan(),
        _ => false,
    }
}

/// Whether `a` ranks strictly below `b` for `argmin` and `argmax`: the comparison order, with
/// `-0.0` below `+0.0`. Neither may be NaN.
pub fn ranks_below(a: &Scalar, b: &Scalar) -> bool {
    let negative_zero_first = match (a, b) {
        (Scalar::F32(x), Scalar::F32(y)) => x.is_sign_negative() && y.is_sign_positive(),
        (Scalar::F64(x), Scalar::F64(y)) => x.is_sign_negative() && y.is_sign_positive(),
        _ => false,
    };
    match compare(a, b) {
        Some(Ordering::Less) => true,
        Some(Ordering::Equal) => negative_zero_first,
        _ => false,
    }
}

/// `op` on two scalars of one type. The comparisons return bool.
pub fn binary(op: Op, a: &Scalar, b: &Scalar) -> Scalar {
    let ordered = |want: fn(Ordering) -> bool| Scalar::Bool(compare(a, b).is_some_and(want));
    match op {
        Op::Add => match (a, b) {
            (Scalar::F32(x), Scalar::F32(y)) => Scalar::F32(x + y),
            (Scalar::F64(x), Scalar::F64(y)) => Scalar::F64(x + y),
            _ => int_pair!(a, b, |x, y| x.wrapping_add(y)),
        },
        Op::Sub => match (a, b) {
            (Scalar::F32(x), Scalar::F32(y)) => Scalar::F32(x - y),
            (Scalar::F64(x), Scalar::F64(y)) => Scalar::F64(x - y),
            _ => int_pair!(a, b, |x, y| x.wrapping_sub(y)),
        },
        Op::Mul => match (a, b) {
            (Scalar::F32(x), Scalar::F32(y)) => Scalar::F32(x * y),
            (Scalar::F64(x), Scalar::F64(y)) => Scalar::F64(x * y),
            _ => int_pair!(a, b, |x, y| x.wrapping_mul(y)),
        },
        Op::Div => match (a, b) {
            (Scalar::F32(x), Scalar::F32(y)) => Scalar::F32(x / y),
            (Scalar::F64(x), Scalar::F64(y)) => Scalar::F64(x / y),
            _ => panic!("no division on {a:?} and {b:?}"),
        },
        Op::Min | Op::Max => {
            let max = op == Op::Max;
            match (a, b) {
                (Scalar::F32(x), Scalar::F32(y)) => Scalar::F32(extreme_f32(*x, *y, max)),
                (Scalar::F64(x), Scalar::F64(y)) => Scalar::F64(extreme_f64(*x, *y, max)),
                _ => {
                    let a_wins = (compare(a, b) == Some(Ordering::Less)) != max;
                    // Integers are totally ordered, so on equality either operand is the result.
                    if a_wins { a.clone() } else { b.clone() }
                }
            }
        }
        Op::Lt => ordered(Ordering::is_lt),
        Op::Le => ordered(Ordering::is_le),
        Op::Gt => ordered(Ordering::is_gt),
        Op::Ge => ordered(Ordering::is_ge),
        Op::Eq => ordered(Ordering::is_eq),
        // NaN is unequal to everything, itself included.
        Op::Ne => Scalar::Bool(!compare(a, b).is_some_and(Ordering::is_eq)),
        Op::And => match (a, b) {
            (Scalar::Bool(x), Scalar::Bool(y)) => Scalar::Bool(*x && *y),
            _ => int_pair!(a, b, |x, y| x & y),
        },
        Op::Or => match (a, b) {
            (Scalar::Bool(x), Scalar::Bool(y)) => Scalar::Bool(*x || *y),
            _ => int_pair!(a, b, |x, y| x | y),
        },
        Op::Xor => match (a, b) {
            (Scalar::Bool(x), Scalar::Bool(y)) => Scalar::Bool(x != y),
            _ => int_pair!(a, b, |x, y| x ^ y),
        },
        // Truncating division that wraps, so `MIN / -1` is `MIN`. The caller rejects zero.
        Op::DivConst => int_pair!(a, b, |x, y| x.wrapping_div(y)),
        _ => panic!("{op:?} is not a binary scalar operation"),
    }
}

/// `op` on one scalar.
pub fn unary(op: Op, a: &Scalar) -> Scalar {
    match (op, a) {
        (Op::Neg, Scalar::I8(x)) => Scalar::I8(x.wrapping_neg()),
        (Op::Neg, Scalar::I16(x)) => Scalar::I16(x.wrapping_neg()),
        (Op::Neg, Scalar::I32(x)) => Scalar::I32(x.wrapping_neg()),
        (Op::Neg, Scalar::I64(x)) => Scalar::I64(x.wrapping_neg()),
        (Op::Neg, Scalar::F32(x)) => Scalar::F32(-x),
        (Op::Neg, Scalar::F64(x)) => Scalar::F64(-x),
        (Op::Abs, Scalar::I8(x)) => Scalar::I8(x.wrapping_abs()),
        (Op::Abs, Scalar::I16(x)) => Scalar::I16(x.wrapping_abs()),
        (Op::Abs, Scalar::I32(x)) => Scalar::I32(x.wrapping_abs()),
        (Op::Abs, Scalar::I64(x)) => Scalar::I64(x.wrapping_abs()),
        (Op::Abs, Scalar::F32(x)) => Scalar::F32(x.abs()),
        (Op::Abs, Scalar::F64(x)) => Scalar::F64(x.abs()),
        (Op::Sqrt, Scalar::F32(x)) => Scalar::F32(x.sqrt()),
        (Op::Sqrt, Scalar::F64(x)) => Scalar::F64(x.sqrt()),
        (Op::Not, Scalar::Bool(x)) => Scalar::Bool(!x),
        (Op::Not, Scalar::I8(x)) => Scalar::I8(!x),
        (Op::Not, Scalar::I16(x)) => Scalar::I16(!x),
        (Op::Not, Scalar::I32(x)) => Scalar::I32(!x),
        (Op::Not, Scalar::I64(x)) => Scalar::I64(!x),
        (Op::Not, Scalar::U8(x)) => Scalar::U8(!x),
        (Op::Not, Scalar::U16(x)) => Scalar::U16(!x),
        (Op::Not, Scalar::U32(x)) => Scalar::U32(!x),
        (Op::Not, Scalar::U64(x)) => Scalar::U64(!x),
        (Op::Cast(t), _) => cast(a, t),
        _ => panic!("no unary {op:?} on {a:?}"),
    }
}

/// `a` converted to `to` by Rust `as`: float to integer truncates toward zero and saturates
/// with NaN as 0, integer to integer wraps, and bool is 0 or 1.
pub fn cast(a: &Scalar, to: ScalarType) -> Scalar {
    macro_rules! to {
        ($v:expr) => {
            match to {
                ScalarType::I8 => Scalar::I8($v as i8),
                ScalarType::I16 => Scalar::I16($v as i16),
                ScalarType::I32 => Scalar::I32($v as i32),
                ScalarType::I64 => Scalar::I64($v as i64),
                ScalarType::U8 => Scalar::U8($v as u8),
                ScalarType::U16 => Scalar::U16($v as u16),
                ScalarType::U32 => Scalar::U32($v as u32),
                ScalarType::U64 => Scalar::U64($v as u64),
                ScalarType::F32 => Scalar::F32($v as f32),
                ScalarType::F64 => Scalar::F64($v as f64),
                ScalarType::Bool => panic!("no cast to bool"),
            }
        };
    }
    match *a {
        Scalar::I8(v) => to!(v),
        Scalar::I16(v) => to!(v),
        Scalar::I32(v) => to!(v),
        Scalar::I64(v) => to!(v),
        Scalar::U8(v) => to!(v),
        Scalar::U16(v) => to!(v),
        Scalar::U32(v) => to!(v),
        Scalar::U64(v) => to!(v),
        Scalar::F32(v) => to!(v),
        Scalar::F64(v) => to!(v),
        Scalar::Bool(v) => to!(u8::from(v)),
    }
}

/// Zero of a numeric type, false for bool.
pub fn zero(t: ScalarType) -> Scalar {
    match t {
        ScalarType::Bool => Scalar::Bool(false),
        _ => cast(&Scalar::U8(0), t),
    }
}

/// One of a numeric type.
pub fn one(t: ScalarType) -> Scalar {
    cast(&Scalar::U8(1), t)
}

/// Every bit set, true for bool.
pub fn all_ones(t: ScalarType) -> Scalar {
    match t {
        ScalarType::Bool => Scalar::Bool(true),
        _ => unary(Op::Not, &zero(t)),
    }
}

/// The largest value of a type, `+inf` for floats.
pub fn highest(t: ScalarType) -> Scalar {
    match t {
        ScalarType::I8 => Scalar::I8(i8::MAX),
        ScalarType::I16 => Scalar::I16(i16::MAX),
        ScalarType::I32 => Scalar::I32(i32::MAX),
        ScalarType::I64 => Scalar::I64(i64::MAX),
        ScalarType::U8 => Scalar::U8(u8::MAX),
        ScalarType::U16 => Scalar::U16(u16::MAX),
        ScalarType::U32 => Scalar::U32(u32::MAX),
        ScalarType::U64 => Scalar::U64(u64::MAX),
        ScalarType::F32 => Scalar::F32(f32::INFINITY),
        ScalarType::F64 => Scalar::F64(f64::INFINITY),
        ScalarType::Bool => Scalar::Bool(true),
    }
}

/// The smallest value of a type, `-inf` for floats.
pub fn lowest(t: ScalarType) -> Scalar {
    match t {
        ScalarType::I8 => Scalar::I8(i8::MIN),
        ScalarType::I16 => Scalar::I16(i16::MIN),
        ScalarType::I32 => Scalar::I32(i32::MIN),
        ScalarType::I64 => Scalar::I64(i64::MIN),
        ScalarType::U8 => Scalar::U8(u8::MIN),
        ScalarType::U16 => Scalar::U16(u16::MIN),
        ScalarType::U32 => Scalar::U32(u32::MIN),
        ScalarType::U64 => Scalar::U64(u64::MIN),
        ScalarType::F32 => Scalar::F32(f32::NEG_INFINITY),
        ScalarType::F64 => Scalar::F64(f64::NEG_INFINITY),
        ScalarType::Bool => Scalar::Bool(false),
    }
}

/// Whether an integer scalar is zero.
pub fn is_zero(a: &Scalar) -> bool {
    compare(a, &zero(a.ty())) == Some(Ordering::Equal)
}

/// The value of an `i64` scalar.
pub fn i64_of(a: &Scalar) -> i64 {
    match a {
        Scalar::I64(v) => *v,
        other => panic!("expected i64, got {other:?}"),
    }
}

/// The value of an integer scalar, which every integer type converts to exactly.
pub fn int_value(a: &Scalar) -> i128 {
    match *a {
        Scalar::I8(v) => i128::from(v),
        Scalar::I16(v) => i128::from(v),
        Scalar::I32(v) => i128::from(v),
        Scalar::I64(v) => i128::from(v),
        Scalar::U8(v) => i128::from(v),
        Scalar::U16(v) => i128::from(v),
        Scalar::U32(v) => i128::from(v),
        Scalar::U64(v) => i128::from(v),
        Scalar::Bool(v) => i128::from(v),
        Scalar::F32(_) | Scalar::F64(_) => panic!("expected an integer, got {a:?}"),
    }
}

/// The value of a bool scalar.
pub fn bool_of(a: &Scalar) -> bool {
    match a {
        Scalar::Bool(v) => *v,
        other => panic!("expected bool, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::extreme_f64;

    #[test]
    fn ieee_extremes_order_zeros_and_propagate_nan() {
        let bits = |x: f64| x.to_bits();
        for (a, b) in [(-0.0, 0.0), (0.0, -0.0)] {
            assert_eq!(bits(extreme_f64(a, b, false)), bits(-0.0), "min({a}, {b})");
            assert_eq!(bits(extreme_f64(a, b, true)), bits(0.0), "max({a}, {b})");
        }
        for max in [false, true] {
            assert!(extreme_f64(f64::NAN, 1.0, max).is_nan());
            assert!(extreme_f64(1.0, f64::NAN, max).is_nan());
            assert!(extreme_f64(f64::NAN, -0.0, max).is_nan());
        }
        assert_eq!(extreme_f64(1.0, 2.0, false), 1.0);
        assert_eq!(extreme_f64(1.0, 2.0, true), 2.0);
    }
}
