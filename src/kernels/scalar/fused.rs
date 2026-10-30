//! Scalar reference implementations of the fused tier 1 kernels, and the oracle that
//! evaluates their patterns through the single-op kernels.
//!
//! A fused kernel is one loop that computes the same per-element values as the unfused
//! chain and feeds them to the same accumulation (the canonical 16-accumulator sum for
//! floats, wrapping widened addition for integers). Results are bit-identical to the chain.

use std::collections::HashMap;

use crate::kernels::scalar::elementwise::{Num, arity_error, scalar_from, scalar_to, slice_of};
use crate::kernels::scalar::reduce::{CanonicalSum, Float, Reducible, array_arg};
use crate::kernels::scalar::select::array_of;
use crate::kernels::{
    FusedPattern, InputSpec, Isa, KernelClass, KernelDef, KernelFn, KernelId, Registry,
};
use crate::value::{Elem, ScalarType, Value, ValueType};

/// Number of accumulators of the canonical sum, which is also its block size.
const BLOCK: usize = 16;

/// The comparisons that fuse with `count`, `filter`, and `condsum`.
#[derive(Clone, Copy)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Kernel name part and comparison, in the order of the const generic index `C`.
const CMPS: [(&str, Cmp); 6] = [
    ("eq", Cmp::Eq),
    ("ne", Cmp::Ne),
    ("lt", Cmp::Lt),
    ("le", Cmp::Le),
    ("gt", Cmp::Gt),
    ("ge", Cmp::Ge),
];

impl Cmp {
    /// The same predicate as the single-op comparison kernel, so NaN follows IEEE 754.
    fn test<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
}

/// Element types whose per-element values can be summed without materializing them.
trait Fusable: Reducible {
    /// The sum of `f(0)..f(n)` exactly as `reduce_add` over the materialized array.
    fn sum_of(n: usize, f: impl Fn(usize) -> Self) -> Self::Sum;
}

/// The canonical sum of `f(0)..f(n)`. Blocks of 16 values go through a stack buffer to
/// `CanonicalSum::update` and the tail of fewer than 16 goes to `finish`.
fn canonical_of<T: Float>(n: usize, f: impl Fn(usize) -> T) -> T {
    let blocks = n - n % BLOCK;
    let mut sum = CanonicalSum::new();
    let mut buf = [T::ZERO; BLOCK];
    for start in (0..blocks).step_by(BLOCK) {
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = f(start + j);
        }
        sum.update(&buf);
    }
    let tail = n - blocks;
    for (j, slot) in buf.iter_mut().enumerate().take(tail) {
        *slot = f(blocks + j);
    }
    sum.finish(&buf[..tail])
}

macro_rules! impl_fusable_float {
    ($($t:ty),*) => {$(
        impl Fusable for $t {
            fn sum_of(n: usize, f: impl Fn(usize) -> Self) -> $t { canonical_of(n, f) }
        }
    )*};
}

macro_rules! impl_fusable_int {
    ($($t:ty => $w:ty),*) => {$(
        impl Fusable for $t {
            fn sum_of(n: usize, f: impl Fn(usize) -> Self) -> $w {
                (0..n).fold(0, |a: $w, i| a.wrapping_add(<$w>::from(f(i))))
            }
        }
    )*};
}

impl_fusable_float!(f64, f32);
impl_fusable_int!(i64 => i64, i32 => i64);

/// The rows of an array argument of element type `ty`.
fn rows<T: Elem>(v: &Value, ty: ScalarType) -> Result<&[T], String> {
    Ok(slice_of::<T>(array_arg(v, ty)?))
}

/// A scalar argument of type `ty`.
fn scalar<T: Elem>(v: &Value, ty: ScalarType) -> Result<T, String> {
    match v {
        Value::Scalar(s) if s.ty() == ty => Ok(scalar_to(s)),
        other => Err(format!("expected {}, got {}", ty.name(), other.ty())),
    }
}

fn same_len(a: usize, b: usize) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("length mismatch: {a} vs {b}"))
    }
}

fn sum_value<T: Fusable>(sum: T::Sum) -> Value {
    Value::Scalar(scalar_from(sum, <T::Sum as Elem>::TYPE))
}

fn run_dot<T: Fusable>(args: &[Value]) -> Result<Value, String> {
    let [x, y] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, y) = (rows::<T>(x, T::TYPE)?, rows::<T>(y, T::TYPE)?);
    same_len(x.len(), y.len())?;
    Ok(sum_value::<T>(T::sum_of(x.len(), |i| Num::mul(x[i], y[i]))))
}

fn run_sumsq<T: Fusable>(args: &[Value]) -> Result<Value, String> {
    let [x] = args else {
        return Err(arity_error(1, args.len()));
    };
    let x = rows::<T>(x, T::TYPE)?;
    Ok(sum_value::<T>(T::sum_of(x.len(), |i| Num::mul(x[i], x[i]))))
}

fn run_count<T: Elem + PartialOrd>(args: &[Value], cmp: Cmp) -> Result<Value, String> {
    let [x, c] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, c) = (rows::<T>(x, T::TYPE)?, scalar::<T>(c, T::TYPE)?);
    let n = x.iter().filter(|&&v| cmp.test(v, c)).count();
    let n = i64::try_from(n).map_err(|e| e.to_string())?;
    Ok(Value::Scalar(scalar_from(n, ScalarType::I64)))
}

fn run_filter<T: Elem + PartialOrd>(args: &[Value], cmp: Cmp) -> Result<Value, String> {
    let [x, c] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, c) = (rows::<T>(x, T::TYPE)?, scalar::<T>(c, T::TYPE)?);
    let out: Vec<T> = x.iter().copied().filter(|&v| cmp.test(v, c)).collect();
    Ok(Value::Array(array_of(out, T::TYPE)))
}

fn run_condsum<T: Fusable>(args: &[Value], cmp: Cmp) -> Result<Value, String> {
    let [x, c, e] = args else {
        return Err(arity_error(3, args.len()));
    };
    let x = rows::<T>(x, T::TYPE)?;
    let (c, e) = (scalar::<T>(c, T::TYPE)?, scalar::<T>(e, T::TYPE)?);
    Ok(sum_value::<T>(T::sum_of(x.len(), |i| {
        if cmp.test(x[i], c) { x[i] } else { e }
    })))
}

/// Registers one fused kernel. Cost is `w` times `cost_in_widths`, with `w` the element width.
#[allow(clippy::too_many_arguments)]
fn add_fused(
    reg: &mut Registry,
    name: String,
    class: KernelClass,
    ty: ScalarType,
    inputs: Vec<InputSpec>,
    output: ValueType,
    cost_in_widths: u64,
    lhs: String,
    vars: &[&str],
    f: KernelFn,
) {
    let w = u64::try_from(ty.width()).expect("width fits in u64");
    reg.add(KernelDef {
        name,
        class,
        op: None,
        inputs,
        output,
        cost: cost_in_widths * w,
        commutative: false,
        pattern: Some(FusedPattern {
            lhs,
            vars: vars.iter().map(|v| (*v).to_string()).collect(),
        }),
        impls: [Some(f), None, None, None],
    });
}

/// Registers the comparison kernels for comparison number `C` of `CMPS`. The kernel
/// closures must coerce to bare function pointers, so `T` and `C` are type-level constants.
fn register_cmp<T: Fusable, const C: usize>(reg: &mut Registry) {
    let (cmp_name, _) = CMPS[C];
    let n = T::TYPE.name();
    let cmp = format!("{cmp_name}_{n}");
    let mask = format!("({cmp} ?x (Bcast ?c))");
    let x_c = vec![InputSpec::Array(T::TYPE), InputSpec::Scalar(T::TYPE)];
    add_fused(
        reg,
        format!("count_{cmp_name}_{n}"),
        KernelClass::Reduction,
        T::TYPE,
        x_c.clone(),
        ValueType::Scalar(ScalarType::I64),
        1,
        format!("(count {mask})"),
        &["?x", "?c"],
        |args| run_count::<T>(args, CMPS[C].1),
    );
    add_fused(
        reg,
        format!("filter_{cmp_name}_{n}"),
        KernelClass::Selection,
        T::TYPE,
        x_c,
        ValueType::Array(T::TYPE),
        2,
        format!("(compress_{n} ?x {mask})"),
        &["?x", "?c"],
        |args| run_filter::<T>(args, CMPS[C].1),
    );
    add_fused(
        reg,
        format!("condsum_{cmp_name}_{n}"),
        KernelClass::Reduction,
        T::TYPE,
        vec![
            InputSpec::Array(T::TYPE),
            InputSpec::Scalar(T::TYPE),
            InputSpec::Scalar(T::TYPE),
        ],
        ValueType::Scalar(<T::Sum as Elem>::TYPE),
        1,
        format!("(reduce_add_{n} (select_{n} {mask} ?x (Bcast ?e)))"),
        &["?x", "?c", "?e"],
        |args| run_condsum::<T>(args, CMPS[C].1),
    );
}

fn register_type<T: Fusable>(reg: &mut Registry) {
    let n = T::TYPE.name();
    let sum_ty = ValueType::Scalar(<T::Sum as Elem>::TYPE);
    add_fused(
        reg,
        format!("dot_{n}"),
        KernelClass::Reduction,
        T::TYPE,
        vec![InputSpec::Array(T::TYPE), InputSpec::Array(T::TYPE)],
        sum_ty,
        2,
        format!("(reduce_add_{n} (mul_{n} ?x ?y))"),
        &["?x", "?y"],
        run_dot::<T>,
    );
    add_fused(
        reg,
        format!("sumsq_{n}"),
        KernelClass::Reduction,
        T::TYPE,
        vec![InputSpec::Array(T::TYPE)],
        sum_ty,
        1,
        format!("(reduce_add_{n} (mul_{n} ?x ?x))"),
        &["?x"],
        run_sumsq::<T>,
    );
    register_cmp::<T, 0>(reg);
    register_cmp::<T, 1>(reg);
    register_cmp::<T, 2>(reg);
    register_cmp::<T, 3>(reg);
    register_cmp::<T, 4>(reg);
    register_cmp::<T, 5>(reg);
}

pub(crate) fn register(reg: &mut Registry) {
    register_type::<f64>(reg);
    register_type::<f32>(reg);
    register_type::<i64>(reg);
    register_type::<i32>(reg);
}

/// A parsed pattern expression.
#[derive(Debug)]
enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

/// Parses one s-expression. Atoms are runs of characters other than parentheses and whitespace.
fn parse(text: &str) -> Result<Sx, String> {
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut pos = 0;
    let sx = parse_at(&tokens, &mut pos)?;
    match tokens.get(pos) {
        None => Ok(sx),
        Some(extra) => Err(format!("trailing token `{extra}` in pattern `{text}`")),
    }
}

/// Parses the expression starting at `tokens[*pos]` and advances `pos` past it.
fn parse_at(tokens: &[&str], pos: &mut usize) -> Result<Sx, String> {
    let token = *tokens.get(*pos).ok_or("unexpected end of pattern")?;
    *pos += 1;
    match token {
        ")" => Err("unexpected `)` in pattern".into()),
        "(" => {
            let mut items = Vec::new();
            while *tokens.get(*pos).ok_or("missing `)` in pattern")? != ")" {
                items.push(parse_at(tokens, pos)?);
            }
            *pos += 1;
            Ok(Sx::List(items))
        }
        atom => Ok(Sx::Atom(atom.to_string())),
    }
}

/// Evaluates a pattern through the single-op scalar kernels. `env` binds the `?` variables.
fn eval(reg: &Registry, sx: &Sx, env: &HashMap<&str, &Value>) -> Result<Value, String> {
    match sx {
        Sx::Atom(a) => env
            .get(a.as_str())
            .map(|v| (*v).clone())
            .ok_or_else(|| format!("unbound pattern atom `{a}`")),
        Sx::List(items) => {
            let Some((Sx::Atom(head), rest)) = items.split_first() else {
                return Err("pattern application needs a head name".into());
            };
            let args = rest
                .iter()
                .map(|a| eval(reg, a, env))
                .collect::<Result<Vec<_>, _>>()?;
            if head == "Bcast" {
                // The scalar passes through as a scalar operand, the kernels broadcast it.
                return match args.as_slice() {
                    [v @ Value::Scalar(_)] => Ok(v.clone()),
                    _ => Err("Bcast takes one scalar".into()),
                };
            }
            let id = reg
                .by_name(head)
                .ok_or_else(|| format!("unknown kernel `{head}` in pattern"))?;
            let def = reg.get(id);
            let types: Vec<ValueType> = args.iter().map(Value::ty).collect();
            if !def.accepts(&types) {
                return Err(format!(
                    "kernel {head} does not accept its pattern arguments"
                ));
            }
            let (f, _) = reg.select_impl(id, Isa::Scalar);
            f(&args)
        }
    }
}

/// The oracle. Evaluates the pattern of fused kernel `id` over `args` through the single-op
/// scalar kernels. A fused kernel must return exactly this value.
pub fn reference(reg: &Registry, id: KernelId, args: &[Value]) -> Result<Value, String> {
    let def = reg.get(id);
    let pattern = def
        .pattern
        .as_ref()
        .ok_or_else(|| format!("kernel {} has no fusion pattern", def.name))?;
    if pattern.vars.len() != args.len() {
        return Err(arity_error(pattern.vars.len(), args.len()));
    }
    let env: HashMap<&str, &Value> = pattern.vars.iter().map(String::as_str).zip(args).collect();
    eval(reg, &parse(&pattern.lhs)?, &env)
}
