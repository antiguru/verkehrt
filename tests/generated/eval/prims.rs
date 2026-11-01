//! The primitives on evaluated operands: elementwise operations with broadcasting, the
//! reductions in the canonical float order, the scans, and the tier 2 primitives.

use std::collections::HashMap;
use std::rc::Rc;

use verkehrt::ops::{Op, ReduceOp};
use verkehrt::value::{Scalar, ScalarType, ValueType};

use super::scalar::{
    all_ones, binary, bool_of, cast, compare, highest, i64_of, int_value, is_nan, is_zero, lowest,
    one, ranks_below, unary, zero,
};
use super::{Stop, Val};

/// The longest array the evaluator builds. A longer one makes the case unsupported, because it
/// would only cost memory and time.
const MAX_ROWS: usize = 1 << 22;

/// The element type of a value type.
pub fn elem(ty: ValueType) -> ScalarType {
    match ty {
        ValueType::Scalar(t) | ValueType::Array(t) => t,
        ValueType::Str => panic!("no strings are evaluated"),
    }
}

fn array(t: ScalarType, rows: Vec<Scalar>) -> Val {
    Val::Array(t, Rc::new(rows))
}

fn rows(v: &Val) -> &[Scalar] {
    match v {
        Val::Array(_, rows) => rows,
        other => panic!("expected an array, got {other:?}"),
    }
}

fn elem_of(v: &Val) -> ScalarType {
    match v {
        Val::Array(t, _) => *t,
        other => panic!("expected an array, got {other:?}"),
    }
}

fn scalar(v: &Val) -> &Scalar {
    match v {
        Val::Scalar(s) => s,
        other => panic!("expected a scalar, got {other:?}"),
    }
}

fn len_i64(n: usize) -> i64 {
    i64::try_from(n).expect("lengths fit i64")
}

fn index(i: usize) -> Scalar {
    Scalar::I64(len_i64(i))
}

/// The row count of a new array of type `t`, from the program's value `n`. `negative` words
/// the error of a negative count.
fn count(n: i64, t: ScalarType, negative: &str) -> Result<usize, Stop> {
    let Ok(rows) = usize::try_from(n) else {
        return Err(Stop::error(format!("{negative} {n}")));
    };
    let fits = rows
        .checked_mul(t.width())
        .is_some_and(|b| isize::try_from(b).is_ok());
    if !fits {
        return Err(Stop::error(format!(
            "array of {rows} elements is too large"
        )));
    }
    if rows > MAX_ROWS {
        return Err(Stop::Unsupported(format!("an array of {rows} elements")));
    }
    Ok(rows)
}

fn mismatch(a: usize, b: usize) -> Stop {
    Stop::error(format!("length mismatch: {a} vs {b}"))
}

/// `op` on one scalar per operand.
fn per_element(op: Op, args: &[&Scalar]) -> Scalar {
    match args {
        [a] => unary(op, a),
        [c, t, f] if op == Op::Select => {
            if bool_of(c) {
                (*t).clone()
            } else {
                (*f).clone()
            }
        }
        [a, b] => binary(op, a, b),
        _ => panic!("{op:?} on {} operands", args.len()),
    }
}

/// An elementwise op. Scalar operands next to arrays repeat to the arrays' length, which
/// must be equal. A phantom operand, see `Val::Phantom`, gives a phantom result.
fn elementwise(op: Op, args: &[Val], ty: ValueType) -> Result<Val, Stop> {
    let mut stop = None;
    if op == Op::DivConst {
        match &args[1] {
            Val::Scalar(d) if is_zero(d) => {
                stop = Some(Stop::error("div_const: division by zero".to_string()));
            }
            Val::Scalar(_) => {}
            // The checker takes the divisor as a scalar operand, which inside a lambda must
            // not depend on a parameter, so no phantom divisor should reach this point.
            other => {
                return Err(Stop::Unsupported(format!("div_const by {other:?}")));
            }
        }
    }
    let mut len = None;
    for a in args {
        if let Val::Array(_, r) = a {
            match len {
                Some(n) if n != r.len() => {
                    let e = mismatch(n, r.len());
                    stop = Some(match stop {
                        None => e,
                        Some(s) => s.merge(e),
                    });
                    break;
                }
                _ => len = Some(r.len()),
            }
        }
    }
    if let Some(s) = stop {
        return Err(s);
    }
    if args.iter().any(|a| matches!(a, Val::Phantom)) {
        return Ok(Val::Phantom);
    }
    let at = |i: usize| -> Vec<&Scalar> {
        args.iter()
            .map(|a| match a {
                Val::Scalar(s) => s,
                Val::Array(_, r) => &r[i],
                Val::Phantom => unreachable!("handled above"),
            })
            .collect()
    };
    Ok(match len {
        None => Val::Scalar(per_element(op, &at(0))),
        Some(n) => array(elem(ty), (0..n).map(|i| per_element(op, &at(i))).collect()),
    })
}

/// The canonical reduction order of the base spec: accumulator `j` of 16 folds the elements
/// `j, j + 16, ...` below the last multiple of 16 in index order, the accumulators combine
/// pairwise at distances 8, 4, 2, and 1, and the tail folds into accumulator 0 in index order.
fn canonical<T: Copy>(xs: &[T], identity: T, op: impl Fn(T, T) -> T) -> T {
    let body = xs.len() - xs.len() % 16;
    let mut acc = [identity; 16];
    for (i, &x) in xs[..body].iter().enumerate() {
        acc[i % 16] = op(acc[i % 16], x);
    }
    for distance in [8, 4, 2, 1] {
        for j in 0..distance {
            acc[j] = op(acc[j], acc[j + distance]);
        }
    }
    xs[body..].iter().fold(acc[0], |a, &x| op(a, x))
}

/// The canonical float sum or product of `xs` of float type `t`, accumulated in `t`.
fn canonical_float(r: ReduceOp, xs: &[Scalar], t: ScalarType) -> Scalar {
    let mul = r == ReduceOp::Mul;
    match t {
        ScalarType::F32 => {
            let v: Vec<f32> = xs
                .iter()
                .map(|s| match s {
                    Scalar::F32(x) => *x,
                    other => panic!("expected f32, got {other:?}"),
                })
                .collect();
            Scalar::F32(if mul {
                canonical(&v, 1.0, |a, b| a * b)
            } else {
                canonical(&v, 0.0, |a, b| a + b)
            })
        }
        ScalarType::F64 => {
            let v: Vec<f64> = xs
                .iter()
                .map(|s| match s {
                    Scalar::F64(x) => *x,
                    other => panic!("expected f64, got {other:?}"),
                })
                .collect();
            Scalar::F64(if mul {
                canonical(&v, 1.0, |a, b| a * b)
            } else {
                canonical(&v, 0.0, |a, b| a + b)
            })
        }
        other => panic!("no canonical order on {other:?}"),
    }
}

/// The identity of the fold `r` in type `t`, which an empty bucket of `reduce_into` holds.
fn identity(r: ReduceOp, t: ScalarType) -> Scalar {
    match r {
        ReduceOp::Add | ReduceOp::Or | ReduceOp::Xor => zero(t),
        ReduceOp::Mul => one(t),
        ReduceOp::And => all_ones(t),
        ReduceOp::Min => highest(t),
        ReduceOp::Max => lowest(t),
    }
}

fn fold_op(r: ReduceOp) -> Op {
    match r {
        ReduceOp::Add => Op::Add,
        ReduceOp::Mul => Op::Mul,
        ReduceOp::Min => Op::Min,
        ReduceOp::Max => Op::Max,
        ReduceOp::And => Op::And,
        ReduceOp::Or => Op::Or,
        ReduceOp::Xor => Op::Xor,
    }
}

/// `x` in the fold's output type `out`. Only `add` on narrow integers widens.
fn widen(x: &Scalar, out: ScalarType) -> Scalar {
    if x.ty() == out {
        x.clone()
    } else {
        cast(x, out)
    }
}

/// One step of the fold `r` with output type `out`.
fn step(r: ReduceOp, acc: &Scalar, x: &Scalar, out: ScalarType) -> Scalar {
    binary(fold_op(r), acc, &widen(x, out))
}

fn reduce(r: ReduceOp, input: &Val, out: ScalarType) -> Result<Val, Stop> {
    let xs = rows(input);
    let float = matches!(elem_of(input), ScalarType::F32 | ScalarType::F64);
    let result = match r {
        ReduceOp::Add | ReduceOp::Mul if float => canonical_float(r, xs, elem_of(input)),
        ReduceOp::Min | ReduceOp::Max => {
            let Some((first, rest)) = xs.split_first() else {
                let name = if r == ReduceOp::Min { "min" } else { "max" };
                return Err(Stop::error(format!("reduce_{name} of an empty array")));
            };
            rest.iter()
                .fold(first.clone(), |acc, x| binary(fold_op(r), &acc, x))
        }
        _ => xs
            .iter()
            .fold(identity(r, out), |acc, x| step(r, &acc, x, out)),
    };
    Ok(Val::Scalar(result))
}

/// The index of the first NaN, otherwise of the first extreme value.
fn arg_extreme(x: &Val, max: bool) -> Result<Val, Stop> {
    let xs = rows(x);
    if let Some(i) = xs.iter().position(is_nan) {
        return Ok(Val::Scalar(index(i)));
    }
    let mut best: Option<usize> = None;
    for (i, v) in xs.iter().enumerate() {
        let better = match best {
            None => true,
            Some(b) if max => ranks_below(&xs[b], v),
            Some(b) => ranks_below(v, &xs[b]),
        };
        if better {
            best = Some(i);
        }
    }
    match best {
        Some(i) => Ok(Val::Scalar(index(i))),
        None => Err(Stop::error(format!(
            "{} of an empty array",
            if max { "argmax" } else { "argmin" }
        ))),
    }
}

/// `scan`, or `scan_by` when `starts` is given: row 0 and every row that starts a segment
/// hold the row itself, every other row the fold of its predecessor's output and the row.
fn scan(r: ReduceOp, x: &Val, starts: Option<&Val>, out: ScalarType) -> Result<Val, Stop> {
    let xs = rows(x);
    let starts = starts.map(rows);
    if let Some(s) = starts
        && s.len() != xs.len()
    {
        return Err(mismatch(xs.len(), s.len()));
    }
    let mut result: Vec<Scalar> = Vec::with_capacity(xs.len());
    for (i, v) in xs.iter().enumerate() {
        let restart = i == 0 || starts.is_some_and(|s| bool_of(&s[i]));
        let next = if restart {
            widen(v, out)
        } else {
            step(r, &result[i - 1], v, out)
        };
        result.push(next);
    }
    Ok(array(out, result))
}

/// The bucket of each key, checked against the bucket count `n`. A negative count and a key
/// out of range are both possible errors.
fn buckets(keys: &[Scalar], n: i64) -> Result<(Vec<usize>, usize), Stop> {
    let negative = (n < 0).then(|| Stop::error(format!("negative bucket count {n}")));
    let mut ids = Vec::with_capacity(keys.len());
    for k in keys {
        let k = i64_of(k);
        if !(0..n).contains(&k) {
            let e = Stop::error(format!("bucket key {k} out of range for {n} buckets"));
            return Err(match negative {
                None => e,
                Some(s) => s.merge(e),
            });
        }
        ids.push(usize::try_from(k).expect("checked above"));
    }
    if let Some(s) = negative {
        return Err(s);
    }
    let count = count(n, ScalarType::I64, "negative bucket count")?;
    Ok((ids, count))
}

/// `reduce_into(keys, vals, n, r)`: the fold of the values per key, each bucket starting from
/// the identity of the fold. Both a bucket error and a length mismatch are possible errors.
fn reduce_into(r: ReduceOp, keys: &Val, vals: &Val, n: i64, out: ScalarType) -> Result<Val, Stop> {
    let (keys, vals) = (rows(keys), rows(vals));
    let checked = buckets(keys, n);
    let lengths = if keys.len() == vals.len() {
        Ok(())
    } else {
        Err(mismatch(keys.len(), vals.len()))
    };
    let (ids, count) = match (checked, lengths) {
        (Ok(b), Ok(())) => b,
        (Err(a), Err(b)) => return Err(a.merge(b)),
        (Err(e), _) | (_, Err(e)) => return Err(e),
    };
    let mut result = vec![identity(r, out); count];
    for (b, v) in ids.into_iter().zip(vals) {
        result[b] = step(r, &result[b], v, out);
    }
    Ok(array(out, result))
}

fn histogram(keys: &Val, n: i64) -> Result<Val, Stop> {
    let (ids, count) = buckets(rows(keys), n)?;
    let mut result = vec![0i64; count];
    for b in ids {
        result[b] = result[b].wrapping_add(1);
    }
    Ok(array(
        ScalarType::I64,
        result.into_iter().map(Scalar::I64).collect(),
    ))
}

/// SplitMix64 of `x`: the finalizer applied to `x + 0x9E3779B97F4A7C15`.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `rand_i64` and `rand_f64`: element `i` is SplitMix64 of `seed + i`, as `i64` bits or with
/// its top 53 bits scaled by `2^-53`.
fn rand(n: i64, seed: i64, out: ScalarType) -> Result<Val, Stop> {
    let n = count(n, out, "negative length")?;
    let seed = seed.cast_unsigned();
    let unit = 1.0 / 9_007_199_254_740_992.0;
    let result = (0..n)
        .map(|i| {
            let bits = splitmix64(seed.wrapping_add(u64::try_from(i).expect("fits u64")));
            match out {
                ScalarType::I64 => Scalar::I64(bits.cast_signed()),
                // The top 53 bits convert to `f64` exactly.
                ScalarType::F64 => Scalar::F64((bits >> 11) as f64 * unit),
                other => panic!("no random {other:?}"),
            }
        })
        .collect();
    Ok(array(out, result))
}

/// `group_id`: the dense id of each row's key, numbered by first occurrence. The key is a
/// row of each column, and the columns must be equally long.
fn group_id(cols: &[Val]) -> Result<Val, Stop> {
    let cols: Vec<&[Scalar]> = cols.iter().map(rows).collect();
    let n = cols[0].len();
    if let Some(c) = cols.iter().find(|c| c.len() != n) {
        return Err(mismatch(n, c.len()));
    }
    let mut ids: HashMap<Vec<i128>, i64> = HashMap::new();
    let result = (0..n)
        .map(|i| {
            let key: Vec<i128> = cols.iter().map(|c| int_value(&c[i])).collect();
            let next = len_i64(ids.len());
            Scalar::I64(*ids.entry(key).or_insert(next))
        })
        .collect();
    Ok(array(ScalarType::I64, result))
}

/// Applies the primitive `op` to evaluated operands. `ty` is the type of the expression.
pub fn apply(op: Op, args: &[Val], ty: ValueType) -> Result<Val, Stop> {
    let out = elem(ty);
    let int = |i: usize| i64_of(scalar(&args[i]));
    match op {
        _ if is_elementwise(op) => elementwise(op, args, ty),
        Op::Reduce(r) => reduce(r, &args[0], out),
        Op::Scan(r) => scan(r, &args[0], None, out),
        Op::ScanBy(r) => scan(r, &args[0], Some(&args[1]), out),
        Op::ReduceInto(r) => reduce_into(r, &args[0], &args[1], int(2), out),
        Op::Count => Ok(Val::Scalar(Scalar::I64(len_i64(
            rows(&args[0]).iter().filter(|b| bool_of(b)).count(),
        )))),
        Op::Any => Ok(Val::Scalar(Scalar::Bool(
            rows(&args[0]).iter().any(bool_of),
        ))),
        Op::All => Ok(Val::Scalar(Scalar::Bool(
            rows(&args[0]).iter().all(bool_of),
        ))),
        Op::ArgMin => arg_extreme(&args[0], false),
        Op::ArgMax => arg_extreme(&args[0], true),
        Op::Length => Ok(Val::Scalar(Scalar::I64(len_i64(rows(&args[0]).len())))),
        Op::Compress => {
            let (xs, mask) = (rows(&args[0]), rows(&args[1]));
            if xs.len() != mask.len() {
                return Err(mismatch(xs.len(), mask.len()));
            }
            let kept = xs
                .iter()
                .zip(mask)
                .filter(|(_, m)| bool_of(m))
                .map(|(x, _)| x.clone())
                .collect();
            Ok(array(out, kept))
        }
        Op::Where => {
            let kept = rows(&args[0])
                .iter()
                .enumerate()
                .filter(|(_, m)| bool_of(m))
                .map(|(i, _)| index(i))
                .collect();
            Ok(array(ScalarType::I64, kept))
        }
        Op::Prev => {
            let xs = rows(&args[0]);
            let fill = scalar(&args[1]);
            let shifted = xs
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        fill.clone()
                    } else {
                        xs[i - 1].clone()
                    }
                })
                .collect();
            Ok(array(out, shifted))
        }
        Op::Iota => {
            let n = count(int(0), ScalarType::I64, "negative length")?;
            Ok(array(ScalarType::I64, (0..n).map(index).collect()))
        }
        Op::Fill => {
            let n = count(int(0), out, "negative length")?;
            Ok(array(out, vec![scalar(&args[1]).clone(); n]))
        }
        Op::Gather => {
            let (xs, idx) = (rows(&args[0]), rows(&args[1]));
            let n = len_i64(xs.len());
            let mut picked = Vec::with_capacity(idx.len());
            for i in idx {
                let i = i64_of(i);
                if !(0..n).contains(&i) {
                    return Err(Stop::error(format!(
                        "gather index {i} out of range for length {n}"
                    )));
                }
                picked.push(xs[usize::try_from(i).expect("checked above")].clone());
            }
            Ok(array(out, picked))
        }
        Op::Slice => {
            let xs = rows(&args[0]);
            let (start, end) = (int(1), int(2));
            let n = len_i64(xs.len());
            if !(0 <= start && start <= end && end <= n) {
                return Err(Stop::error(format!(
                    "slice {start}..{end} out of range for length {n}"
                )));
            }
            let (s, e) = (
                usize::try_from(start).expect("checked above"),
                usize::try_from(end).expect("checked above"),
            );
            Ok(array(out, xs[s..e].to_vec()))
        }
        Op::Histogram => histogram(&args[0], int(1)),
        Op::ArgSort => {
            let xs = rows(&args[0]);
            let mut order: Vec<usize> = (0..xs.len()).collect();
            // A stable sort keeps equal keys in position order.
            order.sort_by(|&a, &b| compare(&xs[a], &xs[b]).expect("argsort keys are ordered"));
            Ok(array(
                ScalarType::I64,
                order.into_iter().map(index).collect(),
            ))
        }
        Op::GroupId => group_id(args),
        Op::RandI64 | Op::RandF64 => rand(int(0), int(1), out),
        Op::Load(_) => Err(Stop::Unsupported("load".to_string())),
        _ => panic!("{op:?} is not a primitive of a typed program"),
    }
}

/// Whether the op applies per element, the only kind a lambda body may use.
fn is_elementwise(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::DivConst
            | Op::Neg
            | Op::Abs
            | Op::Min
            | Op::Max
            | Op::Sqrt
            | Op::Eq
            | Op::Ne
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Not
            | Op::Select
            | Op::Cast(_)
    )
}
