//! Property test: every ISA variant of every kernel returns the scalar kernel's result bit
//! for bit, except that a NaN result only has to be NaN. Kernels are enumerated from the
//! registry, so every new variant is covered without listing it here.

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use verkehrt::kernels::{
    Backend, Dispatch, InputSpec, Isa, KernelClass, KernelDef, KernelId, Registry, Variant,
};
use verkehrt::value::{Arg, Array, ArrayBuilder, Elem, Scalar, ScalarType, Value, ValueType};

/// Row counts of every checked call, around the vector widths of AVX2 and AVX-512.
const LENGTHS: [usize; 15] = [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 300];
/// A row count long enough for every reduction and selection tile below plus a rest.
const TILED_LEN: usize = 1000;
/// Array views start at row 0 or at a random row in `1..=MAX_OFFSET` of a larger buffer.
const MAX_OFFSET: usize = 7;
/// Rows in each generated buffer.
const BUF_LEN: usize = TILED_LEN + MAX_OFFSET;
/// Reduction tiles of 64, 320, and 608 rows, then the rest. Values are tile ends.
const REDUCE_CUTS: [usize; 3] = [64, 384, 992];
/// Selection tiles `0..37`, `37..100`, and `100..n`. Values are tile ends.
const SELECT_CUTS: [usize; 2] = [37, 100];
/// Proptest cases per kernel and variant, each covering every length at both offsets.
const CASES: u32 = 6;

/// One generated input: a buffer of `BUF_LEN` rows and a scalar of the same element type.
/// `splat` picks the scalar for an `Either` input, `pick` replaces the scalar by a row of
/// the first array input of the same type, so comparisons hit equal values.
#[derive(Clone, Debug)]
struct Input {
    rows: Array,
    scalar: Scalar,
    splat: bool,
    pick: Option<Index>,
}

#[derive(Clone, Debug)]
struct Case {
    inputs: Vec<Input>,
    offset: usize,
}

/// `specials` is `[0.0, -0.0, inf, -inf, NaN]`. Without `special`, only the signed zeros
/// join the finite values, so most float sums stay finite and the bit compare is not vacuous.
fn floats<T: std::fmt::Debug + Copy + 'static>(
    finite: std::ops::Range<T>,
    wide: BoxedStrategy<T>,
    specials: [T; 5],
    special: bool,
) -> BoxedStrategy<T>
where
    std::ops::Range<T>: Strategy<Value = T>,
{
    if special {
        prop_oneof![
            12 => finite,
            4 => wide,
            5 => proptest::sample::select(specials.to_vec()),
        ]
        .boxed()
    } else {
        prop_oneof![
            30 => finite,
            4 => wide,
            1 => proptest::sample::select(specials[..2].to_vec()),
        ]
        .boxed()
    }
}

fn input_of<T: Elem + std::fmt::Debug>(
    elem: BoxedStrategy<T>,
    wrap: fn(T) -> Scalar,
) -> BoxedStrategy<Input> {
    (
        vec(elem.clone(), BUF_LEN),
        elem,
        any::<bool>(),
        proptest::option::of(any::<Index>()),
    )
        .prop_map(move |(rows, s, splat, pick)| Input {
            rows: Array::from_vec(rows),
            scalar: wrap(s),
            splat,
            pick,
        })
        .boxed()
}

fn input_strategy(ty: ScalarType, special: bool) -> BoxedStrategy<Input> {
    match ty {
        ScalarType::F64 => input_of(
            floats(
                -1e3f64..1e3,
                (proptest::num::f64::NORMAL | proptest::num::f64::SUBNORMAL).boxed(),
                [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                special,
            ),
            Scalar::F64,
        ),
        ScalarType::F32 => input_of(
            floats(
                -1e3f32..1e3,
                (proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL).boxed(),
                [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
                special,
            ),
            Scalar::F32,
        ),
        ScalarType::I64 => input_of(any::<i64>().boxed(), Scalar::I64),
        ScalarType::I32 => input_of(any::<i32>().boxed(), Scalar::I32),
        ScalarType::U8 => input_of(any::<u8>().boxed(), Scalar::U8),
        ScalarType::Bool => (
            vec(any::<bool>(), BUF_LEN),
            any::<bool>(),
            any::<bool>(),
            proptest::option::of(any::<Index>()),
        )
            .prop_map(|(rows, s, splat, pick)| Input {
                rows: Array::from_bools(&rows),
                scalar: Scalar::Bool(s),
                splat,
                pick,
            })
            .boxed(),
        other => panic!("no generator for {other:?}"),
    }
}

fn spec_type(spec: InputSpec) -> ScalarType {
    match spec {
        InputSpec::Array(t) | InputSpec::Scalar(t) | InputSpec::Either(t) => t,
        InputSpec::Str => panic!("string inputs belong to whole kernels"),
    }
}

fn case_strategy(def: &KernelDef) -> BoxedStrategy<Case> {
    let inputs: Vec<InputSpec> = def.inputs.clone();
    any::<bool>()
        .prop_flat_map(move |special| {
            let per_input: Vec<BoxedStrategy<Input>> = inputs
                .iter()
                .map(|s| input_strategy(spec_type(*s), special))
                .collect();
            (per_input, 1..=MAX_OFFSET)
        })
        .prop_map(|(inputs, offset)| Case { inputs, offset })
        .boxed()
}

/// The arguments for `n` rows starting at row `offset` of each buffer.
fn args_for(def: &KernelDef, case: &Case, offset: usize, n: usize) -> Vec<Value> {
    // An elementwise array kernel needs one array operand, so the first `Either` input is an
    // array when every `Either` input asks for a scalar.
    let all_splat = def
        .inputs
        .iter()
        .zip(&case.inputs)
        .all(|(s, i)| !matches!(s, InputSpec::Either(_)) || i.splat);
    let first_either = def
        .inputs
        .iter()
        .position(|s| matches!(s, InputSpec::Either(_)));
    let is_array = |k: usize| match def.inputs[k] {
        InputSpec::Array(_) => true,
        InputSpec::Either(_) => !case.inputs[k].splat || (all_splat && first_either == Some(k)),
        _ => false,
    };
    (0..def.inputs.len())
        .map(|k| {
            let input = &case.inputs[k];
            if is_array(k) {
                return Value::Array(input.rows.slice(offset, n));
            }
            let ty = spec_type(def.inputs[k]);
            let source = (0..def.inputs.len())
                .find(|&j| is_array(j) && spec_type(def.inputs[j]) == ty)
                .map(|j| &case.inputs[j].rows);
            match (input.pick, source) {
                (Some(i), Some(rows)) if n > 0 => Value::Scalar(rows.get(offset + i.index(n))),
                _ => Value::Scalar(input.scalar.clone()),
            }
        })
        .collect()
}

/// Equal bits, except that a NaN in `want` only requires a NaN in `got`.
fn same_scalar(want: &Scalar, got: &Scalar) -> bool {
    match (want, got) {
        (Scalar::F64(w), Scalar::F64(g)) if w.is_nan() => g.is_nan(),
        (Scalar::F64(w), Scalar::F64(g)) => w.to_bits() == g.to_bits(),
        (Scalar::F32(w), Scalar::F32(g)) if w.is_nan() => g.is_nan(),
        (Scalar::F32(w), Scalar::F32(g)) => w.to_bits() == g.to_bits(),
        (w, g) => w == g,
    }
}

fn same_array(want: &Array, got: &Array) -> bool {
    want.elem() == got.elem()
        && want.len() == got.len()
        && (0..want.len()).all(|i| same_scalar(&want.get(i), &got.get(i)))
}

/// Results agree by `same_scalar` row by row, and errors agree as strings.
fn same(want: &Result<Value, String>, got: &Result<Value, String>) -> bool {
    match (want, got) {
        (Ok(Value::Scalar(w)), Ok(Value::Scalar(g))) => same_scalar(w, g),
        (Ok(Value::Array(w)), Ok(Value::Array(g))) => same_array(w, g),
        (Err(w), Err(g)) => w == g,
        _ => false,
    }
}

fn show(r: &Result<Value, String>) -> String {
    match r {
        Ok(v) => format!("{v}"),
        Err(e) => format!("error: {e}"),
    }
}

/// The arguments of the tile `start..end`: zero-copy windows of the arrays, scalars as is.
fn tile_args(args: &[Value], start: usize, end: usize) -> Vec<Arg> {
    args.iter()
        .map(|a| match a {
            Value::Array(a) => Arg::Rows(a.slice(start, end - start)),
            Value::Scalar(s) => Arg::Splat(s.clone()),
            Value::Str(_) => unreachable!("string inputs belong to whole kernels"),
        })
        .collect()
}

/// Tile boundaries `(start, end)` covering `0..n`, cut at every value of `cuts` below `n`.
fn tiles(n: usize, cuts: &[usize]) -> Vec<(usize, usize)> {
    let mut ends: Vec<usize> = cuts.iter().copied().filter(|&c| c < n).collect();
    ends.push(n);
    let mut start = 0;
    ends.into_iter()
        .map(|end| {
            let t = (start, end);
            start = end;
            t
        })
        .collect()
}

/// The variant's reduction fed tile by tile. An empty input gets no update.
fn reduce_tiled(
    reg: &Registry,
    dispatch: &Dispatch,
    id: KernelId,
    args: &[Value],
    n: usize,
) -> Result<Value, String> {
    let mut reducer = dispatch.reducer(reg, id)?;
    if n > 0 {
        for (start, end) in tiles(n, &REDUCE_CUTS) {
            reducer.update(&tile_args(args, start, end))?;
        }
    }
    reducer.finish().map(Value::Scalar)
}

/// The variant's selection called once per tile with the tile's base.
fn select_tiled(
    reg: &Registry,
    dispatch: &Dispatch,
    id: KernelId,
    args: &[Value],
    n: usize,
) -> Result<Value, String> {
    let ValueType::Array(t) = reg.get(id).output else {
        return Err("a selection must produce an array".into());
    };
    let mut out = ArrayBuilder::new(t);
    for (start, end) in tiles(n, &SELECT_CUTS) {
        dispatch.select_tile(reg, id, &tile_args(args, start, end), start, &mut out)?;
    }
    Ok(Value::Array(out.finish()))
}

/// Compares the variant with scalar for every length at both offsets, one-shot and tiled.
fn check_case(
    reg: &Registry,
    variant: &Dispatch,
    scalar: &Dispatch,
    id: KernelId,
    case: &Case,
) -> Result<(), TestCaseError> {
    let def = reg.get(id);
    for n in LENGTHS.into_iter().chain([TILED_LEN]) {
        for offset in [0, case.offset] {
            let args = args_for(def, case, offset, n);
            let want = scalar.call(reg, id, &args);
            let got = variant.call(reg, id, &args);
            prop_assert!(
                same(&want, &got),
                "{} n={n} offset={offset}: scalar {} vs variant {}",
                def.name,
                show(&want),
                show(&got)
            );
            let tiled = match def.class {
                KernelClass::Reduction => Some(reduce_tiled(reg, variant, id, &args, n)),
                KernelClass::Selection => Some(select_tiled(reg, variant, id, &args, n)),
                _ => None,
            };
            if let Some(tiled) = tiled {
                prop_assert!(
                    same(&want, &tiled),
                    "{} n={n} offset={offset} tiled: scalar {} vs variant {}",
                    def.name,
                    show(&want),
                    show(&tiled)
                );
            }
        }
    }
    Ok(())
}

/// Runs the property for every kernel with an entry in exactly this variant's slot, so a
/// kernel that falls back to another entry is not tested twice. Returns the kernel count.
fn check_variant(reg: &Registry, variant: Variant) -> usize {
    let dispatch = reg.resolve(variant.isa, variant.backend);
    let scalar = reg.resolve(Isa::Scalar, Backend::Auto);
    let mut checked = 0;
    for (id, def) in reg.iter() {
        if def.impls[variant.index()].is_none() {
            continue;
        }
        assert_eq!(dispatch.variant_used(id), variant, "{}", def.name);
        let mut runner = TestRunner::new(Config {
            cases: CASES,
            failure_persistence: None,
            ..Config::default()
        });
        runner
            .run(&case_strategy(def), |case| {
                check_case(reg, &dispatch, &scalar, id, &case)
            })
            .unwrap_or_else(|e| panic!("{} {variant}: {e}", def.name));
        checked += 1;
    }
    checked
}

/// The ISAs under test that this CPU can run, printing a message for each skipped one.
fn available_isas() -> Vec<Isa> {
    [Isa::Avx2, Isa::Avx512]
        .into_iter()
        .filter(|isa| {
            let ok = isa.is_available();
            if !ok {
                println!("skipping {}: not available", isa.name());
            }
            ok
        })
        .collect()
}

#[test]
fn isa_variants_match_scalar() {
    let reg = Registry::new();
    for isa in available_isas() {
        let auto = check_variant(
            &reg,
            Variant {
                isa,
                backend: Backend::Auto,
            },
        );
        let intrinsics = check_variant(
            &reg,
            Variant {
                isa,
                backend: Backend::Intrinsics,
            },
        );
        println!(
            "{}: {auto} auto and {intrinsics} intrinsics kernels checked",
            isa.name()
        );
        assert!(auto > 0, "no auto variants for {}", isa.name());
    }
}

#[test]
fn every_array_kernel_has_auto_variants() {
    let reg = Registry::new();
    for (_, def) in reg.iter() {
        let wanted = match def.class {
            KernelClass::Elementwise => def
                .inputs
                .iter()
                .any(|s| matches!(s, InputSpec::Either(_) | InputSpec::Array(_))),
            KernelClass::Reduction | KernelClass::Selection => true,
            KernelClass::Whole => false,
        };
        for isa in [Isa::Avx2, Isa::Avx512] {
            let variant = Variant {
                isa,
                backend: Backend::Auto,
            };
            let has = def.impls[variant.index()].is_some();
            // Variants are compiled for x86-64 only.
            let expected = wanted && cfg!(target_arch = "x86_64");
            assert_eq!(has, expected, "{} {variant}", def.name);
        }
    }
}

#[test]
fn resolve_picks_auto_and_intrinsics_falls_back_to_it() {
    let reg = Registry::new();
    for isa in available_isas() {
        let auto = Variant {
            isa,
            backend: Backend::Auto,
        };
        let by_auto = reg.resolve(isa, Backend::Auto);
        let by_intrinsics = reg.resolve(isa, Backend::Intrinsics);
        for name in ["add_f64", "reduce_add_f64", "dot_f64", "compress_f64"] {
            let id = reg.by_name(name).unwrap();
            assert_eq!(by_auto.variant_used(id), auto, "{name}");
            // No intrinsics entries exist yet, so the intrinsics backend uses the auto entry.
            assert_eq!(by_intrinsics.variant_used(id), auto, "{name}");
        }
    }
}
