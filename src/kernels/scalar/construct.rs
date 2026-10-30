//! Scalar reference construction, input, and length kernels.

use crate::kernels::scalar::elementwise::{arity_error, scalar_to};
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::scalar::select::array_of;
use crate::kernels::{InputSpec, KernelClass, KernelDef, Registry};
use crate::ops::Op;
use crate::value::{Array, Elem, Scalar, ScalarType, Value, ValueType};

/// SplitMix64 finalizer applied to `x + 0x9E3779B97F4A7C15`. Counter-based, so element `i`
/// of a random array needs no state from element `i - 1`.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `2^-53`, which maps the top 53 bits of a random word into `[0, 1)` exactly.
const UNIT_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// An `i64` scalar argument.
fn i64_arg(v: &Value) -> Result<i64, String> {
    match v {
        Value::Scalar(Scalar::I64(n)) => Ok(*n),
        other => Err(format!("expected i64, got {}", other.ty())),
    }
}

/// A row count argument. Negative counts are errors.
fn count_arg(v: &Value) -> Result<usize, String> {
    let n = i64_arg(v)?;
    usize::try_from(n).map_err(|_| format!("negative length {n}"))
}

fn run_iota(args: &[Value]) -> Result<Value, String> {
    let [n] = args else {
        return Err(arity_error(1, args.len()));
    };
    let n = count_arg(n)?;
    Ok(Value::Array(Array::build::<i64>(n, |out| {
        for (i, o) in out.iter_mut().enumerate() {
            *o = i64::try_from(i).expect("row index fits in i64");
        }
    })))
}

/// `T` is the storage type of `ty`.
fn run_fill<T: Elem>(args: &[Value], ty: ScalarType) -> Result<Value, String> {
    let [n, v] = args else {
        return Err(arity_error(2, args.len()));
    };
    let n = count_arg(n)?;
    let v = match v {
        Value::Scalar(s) if s.ty() == ty => scalar_to::<T>(s),
        other => return Err(format!("expected {}, got {}", ty.name(), other.ty())),
    };
    Ok(Value::Array(array_of(vec![v; n], ty)))
}

/// Element `i` is `f(splitmix64(seed + i))`.
fn run_rand<T: Elem>(args: &[Value], f: fn(u64) -> T) -> Result<Value, String> {
    let [n, seed] = args else {
        return Err(arity_error(2, args.len()));
    };
    let n = count_arg(n)?;
    let seed = i64_arg(seed)?.cast_unsigned();
    Ok(Value::Array(Array::build::<T>(n, |out| {
        for (i, o) in out.iter_mut().enumerate() {
            let i = u64::try_from(i).expect("row index fits in u64");
            *o = f(splitmix64(seed.wrapping_add(i)));
        }
    })))
}

/// Reads the whole file at the path argument and decodes it as little-endian `T` rows with
/// `decode`, which receives exactly `size_of::<T>()` bytes.
fn run_load<T: Elem>(args: &[Value], decode: fn(&[u8]) -> T) -> Result<Value, String> {
    let [path] = args else {
        return Err(arity_error(1, args.len()));
    };
    let Value::Str(path) = path else {
        return Err(format!("expected str, got {}", path.ty()));
    };
    let bytes = std::fs::read(&**path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let w = size_of::<T>();
    if bytes.len() % w != 0 {
        return Err(format!(
            "{path} has {} bytes, which is not a multiple of {w}",
            bytes.len()
        ));
    }
    let rows: Vec<T> = bytes.chunks_exact(w).map(decode).collect();
    Ok(Value::Array(Array::from_vec(rows)))
}

fn run_length(args: &[Value], ty: ScalarType) -> Result<Value, String> {
    let [a] = args else {
        return Err(arity_error(1, args.len()));
    };
    let a = array_arg(a, ty)?;
    let n = i64::try_from(a.len()).map_err(|e| e.to_string())?;
    Ok(Value::Scalar(Scalar::I64(n)))
}

#[allow(clippy::too_many_arguments)]
fn add_whole(
    reg: &mut Registry,
    op: Op,
    name: String,
    inputs: Vec<InputSpec>,
    output: ValueType,
    cost: u64,
    f: crate::kernels::KernelFn,
) {
    reg.add(KernelDef {
        name,
        class: KernelClass::Whole,
        op: Some(op),
        inputs,
        output,
        cost,
        commutative: false,
        pattern: None,
        impls: [Some(f), None, None, None],
    });
}

fn width(ty: ScalarType) -> u64 {
    u64::try_from(ty.width()).expect("width fits in u64")
}

// The kernel closures must coerce to bare function pointers, so their type arguments are
// constant expressions, never locals.

macro_rules! fill {
    ($reg:expr, $t:ty, $ty:expr) => {
        add_whole(
            $reg,
            Op::Fill,
            format!("fill_{}", $ty.name()),
            vec![InputSpec::Scalar(ScalarType::I64), InputSpec::Scalar($ty)],
            ValueType::Array($ty),
            width($ty),
            |args| run_fill::<$t>(args, $ty),
        )
    };
}

macro_rules! load {
    ($reg:expr, $t:ty, $ty:expr) => {
        add_whole(
            $reg,
            Op::Load($ty),
            format!("load_{}", $ty.name()),
            vec![InputSpec::Str],
            ValueType::Array($ty),
            width($ty),
            |args| {
                run_load::<$t>(args, |c| {
                    <$t>::from_le_bytes(c.try_into().expect("chunks have the type's width"))
                })
            },
        )
    };
}

macro_rules! length {
    ($reg:expr, $ty:expr) => {
        add_whole(
            $reg,
            Op::Length,
            format!("length_{}", $ty.name()),
            vec![InputSpec::Array($ty)],
            ValueType::Scalar(ScalarType::I64),
            0,
            |args| run_length(args, $ty),
        )
    };
}

pub(crate) fn register(reg: &mut Registry) {
    let i64_s = || InputSpec::Scalar(ScalarType::I64);
    add_whole(
        reg,
        Op::Iota,
        "iota".into(),
        vec![i64_s()],
        ValueType::Array(ScalarType::I64),
        8,
        run_iota,
    );
    fill!(reg, f64, ScalarType::F64);
    fill!(reg, f32, ScalarType::F32);
    fill!(reg, i64, ScalarType::I64);
    fill!(reg, i32, ScalarType::I32);
    fill!(reg, u8, ScalarType::U8);
    // Bool arrays are stored as `u8`.
    fill!(reg, u8, ScalarType::Bool);
    add_whole(
        reg,
        Op::RandI64,
        "rand_i64".into(),
        vec![i64_s(), i64_s()],
        ValueType::Array(ScalarType::I64),
        8,
        |args| run_rand::<i64>(args, |bits| bits.cast_signed()),
    );
    add_whole(
        reg,
        Op::RandF64,
        "rand_f64".into(),
        vec![i64_s(), i64_s()],
        ValueType::Array(ScalarType::F64),
        8,
        // The shifted value has 53 bits, so the conversion to f64 is exact.
        |args| run_rand::<f64>(args, |bits| (bits >> 11) as f64 * UNIT_53),
    );
    load!(reg, f64, ScalarType::F64);
    load!(reg, f32, ScalarType::F32);
    load!(reg, i64, ScalarType::I64);
    load!(reg, i32, ScalarType::I32);
    load!(reg, u8, ScalarType::U8);
    length!(reg, ScalarType::F64);
    length!(reg, ScalarType::F32);
    length!(reg, ScalarType::I64);
    length!(reg, ScalarType::I32);
    length!(reg, ScalarType::U8);
    length!(reg, ScalarType::Bool);
}
