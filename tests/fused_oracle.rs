//! Oracle property tests: every fused kernel must produce bit-identical results to its
//! pattern evaluated through the single-op scalar kernels.

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use verkehrt::kernels::scalar::fused::reference;
use verkehrt::kernels::{InputSpec, Isa, KernelId, Registry};
use verkehrt::value::{Array, Elem, Scalar, ScalarType, Value};

const MAX_LEN: usize = 300;
const CASES: u32 = 64;

/// Two arrays and two scalars of one element type. The arrays share their length.
type Pool = (Array, Array, Scalar, Scalar);

/// How special float values are mixed in.
#[derive(Clone, Copy)]
enum Mix {
    /// Finite values and signed zeros only, so sums stay finite and the bit-exact compare
    /// checks the block and tail accumulation order.
    Finite,
    /// Heavy on NaN and infinities, to check their propagation.
    Special,
}

/// `specials` is `[0.0, -0.0, inf, -inf, NaN]`.
fn floats<T: std::fmt::Debug + Copy + 'static>(
    finite: std::ops::Range<T>,
    specials: [T; 5],
    mix: Mix,
) -> BoxedStrategy<T>
where
    std::ops::Range<T>: Strategy<Value = T>,
{
    match mix {
        Mix::Finite => prop_oneof![
            40 => finite,
            1 => proptest::sample::select(specials[..2].to_vec()),
        ]
        .boxed(),
        Mix::Special => prop_oneof![
            8 => finite,
            5 => proptest::sample::select(specials.to_vec()),
        ]
        .boxed(),
    }
}

/// Lengths weighted toward the block boundaries of the canonical sum.
fn lengths() -> BoxedStrategy<usize> {
    prop_oneof![
        1 => Just(0usize),
        1 => Just(1usize),
        1 => Just(15usize),
        1 => Just(16usize),
        1 => Just(17usize),
        1 => Just(31usize),
        1 => Just(32usize),
        1 => Just(MAX_LEN),
        4 => 0..=MAX_LEN,
    ]
    .boxed()
}

fn pool_of<T: Elem + std::fmt::Debug>(
    elem: BoxedStrategy<T>,
    wrap: fn(T) -> Scalar,
) -> BoxedStrategy<(usize, Pool)> {
    (
        lengths(),
        vec(elem.clone(), MAX_LEN),
        vec(elem.clone(), MAX_LEN),
        elem.clone(),
        elem,
    )
        .prop_map(move |(n, x, y, c, e)| {
            let cut = |v: Vec<T>| Array::from_vec(v).slice(0, n);
            (n, (cut(x), cut(y), wrap(c), wrap(e)))
        })
        .boxed()
}

fn pool_strategy(ty: ScalarType, mix: Mix) -> BoxedStrategy<(usize, Pool)> {
    match ty {
        ScalarType::F64 => pool_of(
            floats(
                -1e3f64..1e3,
                [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                mix,
            ),
            Scalar::F64,
        ),
        ScalarType::F32 => pool_of(
            floats(
                -1e3f32..1e3,
                [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
                mix,
            ),
            Scalar::F32,
        ),
        // Mix the full range with a small range so comparisons hit equal values.
        ScalarType::I64 => pool_of(
            prop_oneof![1 => any::<i64>(), 1 => -3i64..3].boxed(),
            Scalar::I64,
        ),
        ScalarType::I32 => pool_of(
            prop_oneof![1 => any::<i32>(), 1 => -3i32..3].boxed(),
            Scalar::I32,
        ),
        other => panic!("no generator for {other:?}"),
    }
}

fn same_scalar(want: &Scalar, got: &Scalar) -> bool {
    match (want, got) {
        (Scalar::F64(w), Scalar::F64(g)) => {
            if w.is_nan() {
                g.is_nan()
            } else {
                w.to_bits() == g.to_bits()
            }
        }
        (Scalar::F32(w), Scalar::F32(g)) => {
            if w.is_nan() {
                g.is_nan()
            } else {
                w.to_bits() == g.to_bits()
            }
        }
        (w, g) => w == g,
    }
}

fn same(want: &Value, got: &Value) -> bool {
    match (want, got) {
        (Value::Scalar(w), Value::Scalar(g)) => same_scalar(w, g),
        (Value::Array(w), Value::Array(g)) => {
            w.elem() == g.elem()
                && w.len() == g.len()
                && (0..w.len()).all(|i| same_scalar(&w.get(i), &g.get(i)))
        }
        _ => false,
    }
}

/// The element type of the kernel's array input.
fn elem_type(inputs: &[InputSpec]) -> ScalarType {
    match inputs[0] {
        InputSpec::Array(t) => t,
        other => panic!("first input is not an array: {other:?}"),
    }
}

/// Builds the kernel's arguments from the pool, in input order.
fn args_for(inputs: &[InputSpec], pool: &Pool) -> Vec<Value> {
    let (x, y, c, e) = pool;
    let mut arrays = [x, y].into_iter();
    let mut scalars = [c, e].into_iter();
    inputs
        .iter()
        .map(|spec| match spec {
            InputSpec::Array(_) => Value::Array(arrays.next().expect("at most two arrays").clone()),
            InputSpec::Scalar(_) => {
                Value::Scalar(scalars.next().expect("at most two scalars").clone())
            }
            other => panic!("unexpected input {other:?}"),
        })
        .collect()
}

fn check_family(prefix: &str, mixes: &[Mix]) {
    let reg = Registry::new();
    let ids: Vec<KernelId> = reg
        .iter()
        .filter(|(_, d)| d.pattern.is_some() && d.name.starts_with(prefix))
        .map(|(id, _)| id)
        .collect();
    assert!(!ids.is_empty(), "no fused kernels with prefix {prefix}");
    for (id, &mix) in ids
        .iter()
        .flat_map(|id| mixes.iter().map(move |m| (*id, m)))
    {
        let def = reg.get(id);
        let strategy = pool_strategy(elem_type(&def.inputs), mix);
        let mut runner = TestRunner::new(Config {
            cases: CASES,
            failure_persistence: None,
            ..Config::default()
        });
        let fused = def.impls[Isa::Scalar.index()].expect("scalar impl");
        runner
            .run(&strategy, |(n, pool)| {
                let args = args_for(&def.inputs, &pool);
                let got = fused(&args).map_err(TestCaseError::fail)?;
                let want = reference(&reg, id, &args).map_err(TestCaseError::fail)?;
                prop_assert!(
                    same(&want, &got),
                    "{}: n={n} want {want} got {got}",
                    def.name
                );
                prop_assert_eq!(got.ty(), def.output);
                // A finite mix must keep float sums finite, or the bit compare is vacuous.
                if let (Mix::Finite, Value::Scalar(Scalar::F64(g))) = (mix, &got) {
                    prop_assert!(g.is_finite(), "{}: non-finite sum {g}", def.name);
                }
                if let (Mix::Finite, Value::Scalar(Scalar::F32(g))) = (mix, &got) {
                    prop_assert!(g.is_finite(), "{}: non-finite sum {g}", def.name);
                }
                Ok(())
            })
            .unwrap_or_else(|e| panic!("{}: {e}", def.name));
    }
}

#[test]
fn dot_matches_oracle() {
    check_family("dot_", &[Mix::Finite, Mix::Special]);
}

#[test]
fn sumsq_matches_oracle() {
    check_family("sumsq_", &[Mix::Finite, Mix::Special]);
}

#[test]
fn count_matches_oracle() {
    check_family("count_", &[Mix::Special]);
}

#[test]
fn filter_matches_oracle() {
    check_family("filter_", &[Mix::Special]);
}

#[test]
fn condsum_matches_oracle() {
    check_family("condsum_", &[Mix::Finite, Mix::Special]);
}

#[test]
fn at_least_five_families_have_patterns() {
    let reg = Registry::new();
    let families: std::collections::HashSet<&str> = reg
        .iter()
        .filter(|(_, d)| d.pattern.is_some())
        .filter_map(|(_, d)| d.name.split('_').next())
        .collect();
    for f in ["dot", "sumsq", "count", "filter", "condsum"] {
        assert!(families.contains(f), "missing family {f}");
    }
    assert!(families.len() >= 5);
}

#[test]
fn classes_and_costs_follow_the_brief() {
    use verkehrt::kernels::KernelClass;
    let reg = Registry::new();
    let get = |n: &str| reg.get(reg.by_name(n).unwrap_or_else(|| panic!("missing {n}")));
    assert_eq!(get("dot_f64").cost, 16);
    assert_eq!(get("sumsq_i32").cost, 4);
    assert_eq!(get("count_lt_f32").cost, 4);
    assert_eq!(get("filter_ge_i64").cost, 16);
    assert_eq!(get("condsum_ne_f64").cost, 8);
    assert_eq!(get("dot_i32").class, KernelClass::Reduction);
    assert_eq!(get("filter_eq_f64").class, KernelClass::Selection);
    assert_eq!(
        get("condsum_gt_f64").pattern.as_ref().unwrap().lhs,
        "(reduce_add_f64 (select_f64 (gt_f64 ?x (Bcast ?c)) ?x (Bcast ?e)))"
    );
}
