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

/// Row counts of every checked call, around the vector widths of AVX2, AVX-512, and NEON.
const LENGTHS: [usize; 15] = [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 300];
/// A row count long enough for every tile below plus a rest.
const TILED_LEN: usize = 1000;
/// A row count for selections only, longer than two of the chunks that selection entries
/// reserve output in (2048 rows), so chunk boundaries fall inside the tile `100..n`.
const LONG_LEN: usize = 4500;
/// Array views start at row 0 or at a random row in `1..=MAX_OFFSET` of a larger buffer,
/// drawn per input.
const MAX_OFFSET: usize = 7;
/// Rows in each generated buffer.
const BUF_LEN: usize = LONG_LEN + MAX_OFFSET;
/// Reduction tiles of 64, 320, and 608 rows, then the rest. Values are tile ends.
const REDUCE_CUTS: [usize; 3] = [64, 384, 992];
/// Elementwise and selection tiles `0..37`, `37..100`, and `100..n`. Values are tile ends.
const TILE_CUTS: [usize; 2] = [37, 100];
/// Proptest cases per kernel and variant, each covering every length at both offsets.
const CASES: u32 = 6;

/// One generated input: a buffer of `BUF_LEN` rows and a scalar of the same element type.
/// `splat` picks the scalar for an `Either` input, `pick` replaces the scalar by a row of
/// the first array input of the same type, so comparisons hit equal values. `offset` is the
/// view's first row in the misaligned run, drawn per input so that operands are misaligned
/// relative to each other.
#[derive(Clone, Debug)]
struct Input {
    rows: Array,
    scalar: Scalar,
    splat: bool,
    pick: Option<Index>,
    offset: usize,
}

#[derive(Clone, Debug)]
struct Case {
    inputs: Vec<Input>,
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
        1..=MAX_OFFSET,
    )
        .prop_map(move |(rows, s, splat, pick, offset)| Input {
            rows: Array::from_vec(rows),
            scalar: wrap(s),
            splat,
            pick,
            offset,
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
            // A dense prefix of true rows and an empty suffix runs the chunks after the last
            // survivor of a selection, which keep a vector of room past it.
            prop_oneof![
                3 => vec(any::<bool>(), BUF_LEN),
                1 => (0..=BUF_LEN).prop_map(|cut| (0..BUF_LEN).map(|i| i < cut).collect()),
            ],
            any::<bool>(),
            any::<bool>(),
            proptest::option::of(any::<Index>()),
            1..=MAX_OFFSET,
        )
            .prop_map(|(rows, s, splat, pick, offset)| Input {
                rows: Array::from_bools(&rows),
                scalar: Scalar::Bool(s),
                splat,
                pick,
                offset,
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
            per_input
        })
        .prop_map(|inputs| Case { inputs })
        .boxed()
}

/// The arguments for `n` rows, starting at row 0 of each buffer when `aligned`, else at each
/// input's own offset.
fn args_for(def: &KernelDef, case: &Case, aligned: bool, n: usize) -> Vec<Value> {
    let start = |input: &Input| if aligned { 0 } else { input.offset };
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
                return Value::Array(input.rows.slice(start(input), n));
            }
            let ty = spec_type(def.inputs[k]);
            let source = (0..def.inputs.len())
                .find(|&j| is_array(j) && spec_type(def.inputs[j]) == ty)
                .map(|j| &case.inputs[j]);
            match (input.pick, source) {
                (Some(i), Some(src)) if n > 0 => {
                    Value::Scalar(src.rows.get(start(src) + i.index(n)))
                }
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

/// The variant's elementwise kernel called once per tile into one destination. `None` when
/// the kernel does not produce an array from an array operand, which `call` covers alone.
fn elementwise_tiled(
    reg: &Registry,
    dispatch: &Dispatch,
    id: KernelId,
    args: &[Value],
    n: usize,
) -> Option<Result<Value, String>> {
    let ValueType::Array(t) = reg.get(id).output else {
        return None;
    };
    if !args.iter().any(|a| matches!(a, Value::Array(_))) {
        return None;
    }
    let w = t.width();
    let out = Array::build_bytes(n, t, |out| {
        for (start, end) in tiles(n, &TILE_CUTS) {
            let dst = &mut out[start * w..end * w];
            dispatch.elementwise_tile(reg, id, &tile_args(args, start, end), dst)?;
        }
        Ok(())
    });
    Some(out.map(Value::Array))
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
    for (start, end) in tiles(n, &TILE_CUTS) {
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
    let long = (def.class == KernelClass::Selection).then_some(LONG_LEN);
    for n in LENGTHS.into_iter().chain([TILED_LEN]).chain(long) {
        for aligned in [true, false] {
            let args = args_for(def, case, aligned, n);
            let want = scalar.call(reg, id, &args);
            let got = variant.call(reg, id, &args);
            prop_assert!(
                same(&want, &got),
                "{} n={n} aligned={aligned}: scalar {} vs variant {}",
                def.name,
                show(&want),
                show(&got)
            );
            let tiled = match def.class {
                KernelClass::Elementwise => elementwise_tiled(reg, variant, id, &args, n),
                KernelClass::Reduction => Some(reduce_tiled(reg, variant, id, &args, n)),
                KernelClass::Selection => Some(select_tiled(reg, variant, id, &args, n)),
                KernelClass::Whole => None,
            };
            if let Some(tiled) = tiled {
                prop_assert!(
                    same(&want, &tiled),
                    "{} n={n} aligned={aligned} tiled: scalar {} vs variant {}",
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

/// Whether the ISA has auto entries. NEON does not: on aarch64 the auto backend is the
/// baseline build, so the `neon/auto` slot is empty and resolves to scalar.
fn has_auto_entries(isa: Isa) -> bool {
    matches!(isa, Isa::Avx2 | Isa::Avx512)
}

/// The ISAs under test that this CPU can run, printing a message for each skipped one.
fn available_isas() -> Vec<Isa> {
    Isa::ALL
        .into_iter()
        .filter(|isa| *isa != Isa::Scalar)
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
        if has_auto_entries(isa) {
            assert!(auto > 0, "no auto variants for {}", isa.name());
        } else {
            assert_eq!(auto, 0, "unexpected auto variants for {}", isa.name());
        }
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
        for isa in Isa::ALL.into_iter().filter(|isa| *isa != Isa::Scalar) {
            let variant = Variant {
                isa,
                backend: Backend::Auto,
            };
            let has = def.impls[variant.index()].is_some();
            // Variants are compiled for x86-64 only.
            let expected = wanted && has_auto_entries(isa) && cfg!(target_arch = "x86_64");
            assert_eq!(has, expected, "{} {variant}", def.name);
        }
    }
}

#[test]
fn resolve_picks_auto_and_intrinsics_falls_back_to_it() {
    let reg = Registry::new();
    for isa in available_isas()
        .into_iter()
        .filter(|isa| has_auto_entries(*isa))
    {
        let auto = Variant {
            isa,
            backend: Backend::Auto,
        };
        let intrinsics = Variant {
            isa,
            backend: Backend::Intrinsics,
        };
        let by_auto = reg.resolve(isa, Backend::Auto);
        let by_intrinsics = reg.resolve(isa, Backend::Intrinsics);
        for name in ["add_f64", "reduce_add_f64", "dot_f64"] {
            let id = reg.by_name(name).unwrap();
            assert_eq!(by_auto.variant_used(id), auto, "{name}");
            // These kernels have no intrinsics entry, so the intrinsics backend uses the auto
            // entry.
            assert_eq!(by_intrinsics.variant_used(id), auto, "{name}");
        }
        let mut selections = vec!["where".to_string()];
        for t in ["f64", "f32", "i64", "i32"] {
            selections.push(format!("compress_{t}"));
            for cmp in ["eq", "ne", "lt", "le", "gt", "ge"] {
                selections.push(format!("filter_{cmp}_{t}"));
            }
        }
        // AVX2 has no byte compress, so `u8` and bool fall back to auto there.
        let bytes = if isa == Isa::Avx2 { auto } else { intrinsics };
        let expected = selections
            .iter()
            .map(|n| (n.as_str(), intrinsics))
            .chain([("compress_u8", bytes), ("compress_bool", bytes)]);
        for (name, want) in expected {
            let id = reg.by_name(name).unwrap();
            assert_eq!(by_auto.variant_used(id), auto, "{name}");
            assert_eq!(by_intrinsics.variant_used(id), want, "{name}");
        }
    }
}

/// On aarch64 the auto backend is the baseline build, so `neon/auto` resolves to scalar.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_auto_resolves_to_scalar() {
    let reg = Registry::new();
    let dispatch = reg.resolve(Isa::Neon, Backend::Auto);
    for name in ["add_f64", "reduce_add_f64", "dot_f64", "compress_f64"] {
        let id = reg.by_name(name).unwrap();
        assert_eq!(dispatch.variant_used(id), Variant::SCALAR, "{name}");
    }
}

/// NEON has intrinsics entries for the selections and the float extreme reductions. The other
/// kernels have neither an intrinsics nor an auto entry, so the intrinsics backend resolves
/// them to scalar.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_intrinsics_resolve_for_selections() {
    let reg = Registry::new();
    let dispatch = reg.resolve(Isa::Neon, Backend::Intrinsics);
    let intrinsics = Variant {
        isa: Isa::Neon,
        backend: Backend::Intrinsics,
    };
    for name in ["add_f64", "reduce_add_f64", "dot_f64"] {
        let id = reg.by_name(name).unwrap();
        assert_eq!(dispatch.variant_used(id), Variant::SCALAR, "{name}");
    }
    let mut selections = vec!["where".to_string()];
    for t in ["f64", "f32", "i64", "i32", "u8", "bool"] {
        selections.push(format!("compress_{t}"));
    }
    for t in ["f64", "f32", "i64", "i32"] {
        for cmp in ["eq", "ne", "lt", "le", "gt", "ge"] {
            selections.push(format!("filter_{cmp}_{t}"));
        }
    }
    for name in &selections {
        let id = reg.by_name(name).unwrap();
        assert_eq!(dispatch.variant_used(id), intrinsics, "{name}");
    }
}

/// The float extreme reductions have NEON intrinsics entries.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_intrinsics_resolve_for_float_extremes() {
    let reg = Registry::new();
    let dispatch = reg.resolve(Isa::Neon, Backend::Intrinsics);
    for t in ["f64", "f32"] {
        for op in ["reduce_min", "reduce_max", "argmin", "argmax"] {
            let name = format!("{op}_{t}");
            let id = reg.by_name(&name).unwrap();
            assert_eq!(dispatch.variant_used(id), NEON_INTRINSICS, "{name}");
        }
    }
}

#[cfg(target_arch = "aarch64")]
const NEON_INTRINSICS: Variant = Variant {
    isa: Isa::Neon,
    backend: Backend::Intrinsics,
};

/// The float extreme reductions, which have NEON intrinsics entries.
#[cfg(target_arch = "aarch64")]
const EXTREME_OPS: [&str; 4] = ["reduce_min", "reduce_max", "argmin", "argmax"];

/// `values` as an `f64` or an `f32` array. The conversion keeps zeros, infinities, and NaN.
#[cfg(target_arch = "aarch64")]
fn float_array(values: &[f64], f32s: bool) -> Value {
    if f32s {
        #[allow(clippy::cast_possible_truncation)]
        let v: Vec<f32> = values.iter().map(|&x| x as f32).collect();
        Value::Array(Array::from_vec(v))
    } else {
        Value::Array(Array::from_vec(values.to_vec()))
    }
}

/// Runs `op` on `values` as `f64` and as `f32` with the NEON intrinsics entry, one-shot and
/// over the reduction tiles, and checks both against scalar with `same`. Returns the f64 and
/// f32 one-shot results of the NEON entry.
#[cfg(target_arch = "aarch64")]
fn neon_extreme(op: &str, values: &[f64]) -> [Result<Value, String>; 2] {
    let reg = Registry::new();
    let neon = reg.resolve(Isa::Neon, Backend::Intrinsics);
    let scalar = reg.resolve(Isa::Scalar, Backend::Auto);
    [false, true].map(|f32s| {
        let name = format!("{op}_{}", if f32s { "f32" } else { "f64" });
        let id = reg.by_name(&name).unwrap();
        assert_eq!(neon.variant_used(id), NEON_INTRINSICS, "{name}");
        let args = [float_array(values, f32s)];
        let n = values.len();
        let want = scalar.call(&reg, id, &args);
        let got = neon.call(&reg, id, &args);
        assert!(
            same(&want, &got),
            "{name} n={n}: scalar {} vs neon {}",
            show(&want),
            show(&got)
        );
        let tiled = reduce_tiled(&reg, &neon, id, &args, n);
        assert!(
            same(&want, &tiled),
            "{name} n={n} tiled: scalar {} vs neon {}",
            show(&want),
            show(&tiled)
        );
        got
    })
}

/// `n` rows of `fill` with the listed rows replaced.
#[cfg(target_arch = "aarch64")]
fn rows_with(n: usize, fill: f64, set: &[(usize, f64)]) -> Vec<f64> {
    let mut v = vec![fill; n];
    for &(i, x) in set {
        v[i] = x;
    }
    v
}

/// Asserts that both element types gave index `want`.
#[cfg(target_arch = "aarch64")]
fn assert_index(op: &str, values: &[f64], want: i64) {
    for got in neon_extreme(op, values) {
        assert_eq!(got, Ok(Value::Scalar(Scalar::I64(want))), "{op}");
    }
}

/// Asserts that both element types gave a value with the bits of `want`, or a NaN.
#[cfg(target_arch = "aarch64")]
fn assert_value(op: &str, values: &[f64], want: f64) {
    for got in neon_extreme(op, values) {
        let ok = match got {
            Ok(Value::Scalar(Scalar::F64(x))) if want.is_nan() => x.is_nan(),
            Ok(Value::Scalar(Scalar::F64(x))) => x.to_bits() == want.to_bits(),
            #[allow(clippy::cast_possible_truncation)]
            Ok(Value::Scalar(Scalar::F32(x))) if want.is_nan() => x.is_nan(),
            #[allow(clippy::cast_possible_truncation)]
            Ok(Value::Scalar(Scalar::F32(x))) => x.to_bits() == (want as f32).to_bits(),
            _ => false,
        };
        assert!(ok, "{op}: want {want:?}, got {got:?}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_take_the_first_nan_across_tiles() {
    // The first NaN sits in the first tile and a lower and a higher value in later ones.
    let v = rows_with(
        TILED_LEN,
        5.0,
        &[(10, f64::NAN), (500, -100.0), (700, 100.0)],
    );
    assert_index("argmin", &v, 10);
    assert_index("argmax", &v, 10);
    assert_value("reduce_min", &v, f64::NAN);
    assert_value("reduce_max", &v, f64::NAN);
    // A NaN in a later tile than the extreme, and one in the scalar rest of a tile.
    let late = rows_with(
        TILED_LEN,
        5.0,
        &[(3, -100.0), (995, f64::NAN), (998, f64::NAN)],
    );
    assert_index("argmin", &late, 995);
    assert_index("argmax", &late, 995);
    assert_value("reduce_min", &late, f64::NAN);
    let rest = rows_with(19, 5.0, &[(1, -1.0), (18, f64::NAN)]);
    assert_index("argmin", &rest, 18);
    assert_value("reduce_max", &rest, f64::NAN);
    // Two NaNs in one vector, the second in a lower lane of the next.
    let pair = rows_with(40, 5.0, &[(6, f64::NAN), (7, f64::NAN), (8, f64::NAN)]);
    assert_index("argmax", &pair, 6);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_of_all_nan_rows() {
    for n in [1, 2, 5, 17, TILED_LEN] {
        let v = vec![f64::NAN; n];
        assert_index("argmin", &v, 0);
        assert_index("argmax", &v, 0);
        assert_value("reduce_min", &v, f64::NAN);
        assert_value("reduce_max", &v, f64::NAN);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_keep_the_first_index_on_ties() {
    // Rows 1 and 2, and rows 3 and 4, sit in different vectors of both element types, with
    // the later row in the lower lane. Rows 5 and 6 share an f32 vector. Rows 7 and 8, 15
    // and 16, and 13 and 18 sit in the last and the first of the four accumulators of a
    // group, for f64 or f32, with the later row in the earlier accumulator.
    for (a, b) in [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (15, 16),
        (13, 18),
        (63, 64),
        (383, 384),
        (0, 999),
    ] {
        let low = rows_with(TILED_LEN, 5.0, &[(a, 1.0), (b, 1.0)]);
        assert_index("argmin", &low, i64::try_from(a).unwrap());
        assert_value("reduce_min", &low, 1.0);
        let high = rows_with(TILED_LEN, 5.0, &[(a, 9.0), (b, 9.0)]);
        assert_index("argmax", &high, i64::try_from(a).unwrap());
        assert_value("reduce_max", &high, 9.0);
    }
    // Equal rows everywhere.
    for n in [1, 3, 4, 16, 17, TILED_LEN] {
        let v = vec![2.5; n];
        assert_index("argmin", &v, 0);
        assert_index("argmax", &v, 0);
    }
    // Infinities, the vector identities of min and max.
    let inf = vec![f64::INFINITY; 9];
    assert_index("argmin", &inf, 0);
    assert_value("reduce_min", &inf, f64::INFINITY);
    let ninf = vec![f64::NEG_INFINITY; 9];
    assert_index("argmax", &ninf, 0);
    assert_value("reduce_max", &ninf, f64::NEG_INFINITY);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_order_signed_zeros() {
    // Pairs of rows in different lanes, in different vectors, in different accumulators, and
    // across a tile boundary.
    for (a, b) in [(2, 7), (3, 4), (4, 5), (1, 2), (7, 8), (15, 16), (63, 64)] {
        let ia = i64::try_from(a).unwrap();
        let ib = i64::try_from(b).unwrap();
        // +0.0 first: -0.0 is the smaller.
        let pos_neg = rows_with(TILED_LEN, 3.0, &[(a, 0.0), (b, -0.0)]);
        assert_index("argmin", &pos_neg, ib);
        assert_value("reduce_min", &pos_neg, -0.0);
        // -0.0 first: it is still the smaller.
        let neg_pos = rows_with(TILED_LEN, 3.0, &[(a, -0.0), (b, 0.0)]);
        assert_index("argmin", &neg_pos, ia);
        assert_value("reduce_min", &neg_pos, -0.0);
        // The same for max, where +0.0 is the larger.
        let neg_pos_max = rows_with(TILED_LEN, -3.0, &[(a, -0.0), (b, 0.0)]);
        assert_index("argmax", &neg_pos_max, ib);
        assert_value("reduce_max", &neg_pos_max, 0.0);
        let pos_neg_max = rows_with(TILED_LEN, -3.0, &[(a, 0.0), (b, -0.0)]);
        assert_index("argmax", &pos_neg_max, ia);
        assert_value("reduce_max", &pos_neg_max, 0.0);
    }
    // Only zeros, in both orders.
    let zeros = [0.0, 0.0, -0.0, 0.0, -0.0];
    assert_index("argmin", &zeros, 2);
    assert_index("argmax", &zeros, 0);
    assert_value("reduce_min", &zeros, -0.0);
    assert_value("reduce_max", &zeros, 0.0);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_match_scalar_for_short_lengths() {
    // Every length up to two f32 vectors plus a rest, with the extreme at every row, and a
    // few lengths that are not multiples of the lane counts.
    for n in (1..=17).chain([31, 33, 65, 1001]) {
        for at in 0..n {
            let low = rows_with(n, 1.0, &[(at, -1.0)]);
            assert_index("argmin", &low, i64::try_from(at).unwrap());
            assert_value("reduce_min", &low, -1.0);
            let high = rows_with(n, 1.0, &[(at, 7.0)]);
            assert_index("argmax", &high, i64::try_from(at).unwrap());
            assert_value("reduce_max", &high, 7.0);
            let nan = rows_with(n, 1.0, &[(at, f64::NAN)]);
            assert_index("argmin", &nan, i64::try_from(at).unwrap());
            assert_value("reduce_max", &nan, f64::NAN);
        }
        // Distinct values, ascending and descending.
        let up: Vec<f64> = (0..n)
            .map(|i| f64::from(u32::try_from(i).unwrap()))
            .collect();
        let down: Vec<f64> = up.iter().rev().copied().collect();
        // `neon_extreme` checks against scalar, so the results need no further check.
        for op in EXTREME_OPS {
            let _ = neon_extreme(op, &up);
            let _ = neon_extreme(op, &down);
        }
    }
    // A single row.
    assert_index("argmin", &[4.0], 0);
    assert_value("reduce_max", &[-0.0], -0.0);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extremes_of_empty_rows_fail_like_scalar() {
    for op in EXTREME_OPS {
        for got in neon_extreme(op, &[]) {
            assert_eq!(got, Err(format!("{op} of an empty array")), "{op}");
        }
    }
}

/// The NEON extreme reducers check their argument like the scalar ones.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_extreme_reducers_reject_bad_arguments() {
    let reg = Registry::new();
    let neon = reg.resolve(Isa::Neon, Backend::Intrinsics);
    let scalar = reg.resolve(Isa::Scalar, Backend::Auto);
    let ints = Array::from_vec(vec![1i32, 2, 3]);
    let floats = Array::from_vec(vec![1.0f64, 2.0]);
    let bad: [Vec<Arg>; 4] = [
        vec![],
        vec![Arg::Rows(ints.slice(0, 3))],
        vec![Arg::Splat(Scalar::F64(1.0))],
        vec![Arg::Rows(floats.slice(0, 2)), Arg::Rows(floats.slice(0, 2))],
    ];
    for t in ["f64", "f32"] {
        for op in EXTREME_OPS {
            let id = reg.by_name(&format!("{op}_{t}")).unwrap();
            for args in &bad {
                let want = scalar.reducer(&reg, id).unwrap().update(args);
                let got = neon.reducer(&reg, id).unwrap().update(args);
                assert!(want.is_err(), "{op}_{t}");
                assert_eq!(want, got, "{op}_{t}");
            }
        }
    }
}
