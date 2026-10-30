//! Scalar reference implementations of the fused tier 1 kernels, and the oracle that
//! evaluates their patterns through the single-op kernels.
//!
//! A fused kernel is one loop that computes the same per-element values as the unfused
//! chain and feeds them to the same accumulation (the canonical 16-accumulator sum for
//! floats, wrapping widened addition for integers). Results are bit-identical to the chain.

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::kernels::scalar::elementwise::{Num, arity_error, rows_arg, scalar_from, splat_arg};
use crate::kernels::scalar::reduce::{Accum, Reducible};
use crate::kernels::{
    Backend, Dispatch, FusedPattern, InputSpec, Isa, KernelClass, KernelDef, KernelId, KernelImpl,
    Reducer, Registry,
};
use crate::value::{Arg, ArrayBuilder, Elem, Scalar, ScalarType, Value, ValueType};

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

fn same_len(a: usize, b: usize) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("length mismatch: {a} vs {b}"))
    }
}

/// The sum of a computed value per row, in the accumulation of `reduce_add` over the
/// materialized values (the canonical float order, wrapping widened integer addition). The
/// values are never stored, `Accum::feed` takes them from a row function.
fn sum_scalar<T: Reducible>(acc: T::AddAcc) -> Result<Scalar, String> {
    Ok(scalar_from(acc.finish()?, <T::Sum as Elem>::TYPE))
}

/// `dot`: the sum of `x[i] * y[i]`.
struct Dot<T: Reducible> {
    acc: T::AddAcc,
}

impl<T: Reducible> Reducer for Dot<T> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [x, y] = args else {
            return Err(arity_error(2, args.len()));
        };
        let (x, y) = (rows_arg::<T>(x, T::TYPE)?, rows_arg::<T>(y, T::TYPE)?);
        same_len(x.len(), y.len())?;
        self.acc.feed(x.len(), |i| Num::mul(x[i], y[i]))
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        sum_scalar::<T>(self.acc)
    }
}

/// `sumsq`: the sum of `x[i] * x[i]`.
struct SumSq<T: Reducible> {
    acc: T::AddAcc,
}

impl<T: Reducible> Reducer for SumSq<T> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [x] = args else {
            return Err(arity_error(1, args.len()));
        };
        let x = rows_arg::<T>(x, T::TYPE)?;
        self.acc.feed(x.len(), |i| Num::mul(x[i], x[i]))
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        sum_scalar::<T>(self.acc)
    }
}

/// `count_<cmp>`: the number of rows with `x[i] <cmp> c`, for comparison number `C` of `CMPS`.
struct CountCmp<T, const C: usize> {
    count: usize,
    _rows: PhantomData<fn() -> T>,
}

impl<T: Elem + PartialOrd, const C: usize> Reducer for CountCmp<T, C> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [x, c] = args else {
            return Err(arity_error(2, args.len()));
        };
        let (x, c) = (rows_arg::<T>(x, T::TYPE)?, splat_arg::<T>(c, T::TYPE)?);
        self.count += x.iter().filter(|&&v| CMPS[C].1.test(v, c)).count();
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        let n = i64::try_from(self.count).map_err(|e| e.to_string())?;
        Ok(scalar_from(n, ScalarType::I64))
    }
}

/// `condsum_<cmp>`: the sum of `x[i]` where `x[i] <cmp> c` and `e` elsewhere.
struct CondSum<T: Reducible, const C: usize> {
    acc: T::AddAcc,
}

impl<T: Reducible, const C: usize> Reducer for CondSum<T, C> {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [x, c, e] = args else {
            return Err(arity_error(3, args.len()));
        };
        let x = rows_arg::<T>(x, T::TYPE)?;
        let (c, e) = (splat_arg::<T>(c, T::TYPE)?, splat_arg::<T>(e, T::TYPE)?);
        self.acc
            .feed(x.len(), |i| if CMPS[C].1.test(x[i], c) { x[i] } else { e })
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        sum_scalar::<T>(self.acc)
    }
}

/// The `filter_<cmp>` entry for comparison `cmp`: appends the rows with `x[i] <cmp> c`.
fn filter_tile<T: Elem + PartialOrd>(
    args: &[Arg],
    out: &mut ArrayBuilder,
    cmp: Cmp,
) -> Result<(), String> {
    let [x, c] = args else {
        return Err(arity_error(2, args.len()));
    };
    let (x, c) = (rows_arg::<T>(x, T::TYPE)?, splat_arg::<T>(c, T::TYPE)?);
    let kept: Vec<T> = x.iter().copied().filter(|&v| cmp.test(v, c)).collect();
    out.extend(&kept);
    Ok(())
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
    imp: KernelImpl,
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
        impls: imp.scalar_only(),
    });
}

/// Registers the comparison kernels for comparison number `C` of `CMPS`. The kernel
/// closures must coerce to bare function pointers, so `T` and `C` are type-level constants.
fn register_cmp<T: Reducible, const C: usize>(reg: &mut Registry) {
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
        KernelImpl::Reduction(|| {
            Box::new(CountCmp::<T, C> {
                count: 0,
                _rows: PhantomData,
            })
        }),
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
        KernelImpl::Selection(|args: &[Arg], _base: usize, out: &mut ArrayBuilder| {
            filter_tile::<T>(args, out, CMPS[C].1)
        }),
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
        KernelImpl::Reduction(|| {
            Box::new(CondSum::<T, C> {
                acc: <T::AddAcc as Accum<T>>::new(),
            })
        }),
    );
}

fn register_type<T: Reducible>(reg: &mut Registry) {
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
        KernelImpl::Reduction(|| {
            Box::new(Dot::<T> {
                acc: <T::AddAcc as Accum<T>>::new(),
            })
        }),
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
        KernelImpl::Reduction(|| {
            Box::new(SumSq::<T> {
                acc: <T::AddAcc as Accum<T>>::new(),
            })
        }),
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
fn eval(
    reg: &Registry,
    dispatch: &Dispatch,
    sx: &Sx,
    env: &HashMap<&str, &Value>,
) -> Result<Value, String> {
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
                .map(|a| eval(reg, dispatch, a, env))
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
            dispatch.call(reg, id, &args)
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
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto);
    eval(reg, &dispatch, &parse(&pattern.lhs)?, &env)
}
