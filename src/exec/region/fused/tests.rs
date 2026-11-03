use std::cell::Cell;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use super::fused_tile_rows;
use crate::error::{ErrorKind, Span};
use crate::exec::region::tests::{PROBE_CALLS, probe_registry};
use crate::exec::tests::{Builder, Built, Outcome, calls_only, eval_with, rows, text, variants};
use crate::exec::{Slot, Tiling};
use crate::kernels::{
    Backend, FeatureSet, InputSpec, Isa, KernelClass, KernelDef, KernelImpl, Reducer,
    ReductionBody, Registry, Split,
};
use crate::lower::Leaf;
use crate::value::{Arg, Array, Scalar, ScalarType, Value, ValueType};
use crate::{Interpreter, Options};

/// The tile row counts of the program tests. 1 and 3 round up to 16-row tiles and 17 to
/// 32-row tiles, and 48 is a multiple of the 16-row block.
const ROWS: [usize; 5] = [1, 3, 16, 17, 48];

fn rows_of(t: usize) -> Tiling {
    Tiling::Rows(NonZeroUsize::new(t).unwrap())
}

/// What one run of a program gave.
struct Ran {
    outcome: Outcome,
    /// The trace and stats lines without their times, sorted.
    lines: Vec<String>,
    /// The kernel names of the trace lines in order.
    order: Vec<String>,
    regions: u64,
    narrowed: u64,
    fused: BTreeMap<String, u64>,
}

/// Runs `src` with the given settings on `variant`, keeping its `let`s bound.
fn run(
    src: &str,
    names: &[&str],
    variant: (Isa, Backend),
    tiling: Tiling,
    selective: bool,
    fuse: bool,
) -> Ran {
    let mut interp = Interpreter::new(Options {
        isa: Some(variant.0),
        backend: variant.1,
        trace: true,
        stats: true,
        tiling,
        no_selective: !selective,
        no_reduction_fusion: !fuse,
        ..Options::default()
    })
    .unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    let outcome = interp.eval_repl(src, &mut out, &mut diag).map(|()| {
        let bound = names
            .iter()
            .map(|n| rows(&interp.get(n).unwrap_or_else(|| panic!("{n} is not bound"))))
            .collect();
        (String::from_utf8(out).unwrap(), bound)
    });
    let traced = String::from_utf8(diag.clone()).unwrap();
    let order = traced
        .lines()
        .map(|l| l.split(' ').nth(1).unwrap().to_string())
        .collect();
    interp.write_stats(&mut diag);
    let mut lines: Vec<String> = String::from_utf8(diag)
        .unwrap()
        .lines()
        .map(|l| {
            l.split(' ')
                .filter(|w| !w.starts_with("time=") && !w.starts_with("gbps="))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    lines.sort();
    Ran {
        outcome,
        lines,
        order,
        regions: interp.tiled_regions(),
        narrowed: interp.narrowed_tiles(),
        fused: interp.fused_reductions(),
    }
}

/// Runs `src` on every variant untiled and with every tile row count of `ROWS`, with fusion
/// on and off and with and without selective evaluation. Checks that every run gives the
/// outcome of the untiled scalar run, and that fused runs give the trace order of unfused
/// runs and the trace and stats lines of the untiled run without selective evaluation. With
/// it, fused runs give the kernel calls of unfused runs, and when the fused tiles have the
/// rows of the unfused ones, also their trace and stats lines and narrowed tile count. Fused
/// runs count `fused` per root and `regions_on` tiled regions, unfused tiled runs
/// `regions_off`, and untiled runs none. A root whose entry on a variant is `Split::Bound`
/// does not fuse there, so on that variant fused runs count nothing for it and the tiled
/// regions of unfused runs. Returns the untiled outcome.
fn check(
    src: &str,
    names: &[&str],
    fused: &[(&str, u64)],
    regions_off: u64,
    regions_on: u64,
) -> Outcome {
    let reg = Registry::new();
    let scalar = (Isa::Scalar, Backend::Auto);
    let want = run(src, names, scalar, Tiling::Off, false, true).outcome;
    for variant in variants() {
        // NEON's float `reduce_min` and `reduce_max` are `Split::Bound`, so which roots fuse
        // depends on the variant.
        let dispatch = reg.resolve(variant.0, variant.1, FeatureSet::EMPTY);
        let free = |name: &str| {
            let id = reg
                .by_name(name)
                .unwrap_or_else(|| panic!("no kernel {name}"));
            dispatch.reducer_split(id) == Some(Split::Free)
        };
        let want_fused: BTreeMap<String, u64> = fused
            .iter()
            .filter(|(n, _)| free(n))
            .map(|(n, c)| (n.to_string(), *c))
            .collect();
        let all_free = fused.iter().all(|(n, _)| free(n));
        // The fused region count of a program with a bound root follows from `regions_off`
        // only when no other root fuses.
        assert!(
            all_free || want_fused.is_empty(),
            "{variant:?} mixes bound and free roots in\n{src}"
        );
        let regions_on = if all_free { regions_on } else { regions_off };
        let untiled = run(src, names, variant, Tiling::Off, false, true);
        assert_eq!(
            untiled.outcome, want,
            "{variant:?} untiled outcome of\n{src}"
        );
        assert_eq!(untiled.regions, 0, "{variant:?} untiled regions of\n{src}");
        assert!(untiled.fused.is_empty(), "{variant:?} untiled of\n{src}");
        let unfused = run(src, names, variant, Tiling::Off, false, false);
        assert_eq!(
            unfused.lines, untiled.lines,
            "{variant:?} untiled of\n{src}"
        );
        for t in ROWS {
            for selective in [false, true] {
                let at = format!("{variant:?} rows {t} selective {selective} of\n{src}");
                let on = run(src, names, variant, rows_of(t), selective, true);
                let off = run(src, names, variant, rows_of(t), selective, false);
                assert_eq!(on.outcome, want, "fused outcome, {at}");
                assert_eq!(off.outcome, want, "unfused outcome, {at}");
                assert_eq!(on.order, off.order, "trace order, {at}");
                assert_eq!(calls_only(&on.lines), calls_only(&off.lines), "calls, {at}");
                // A narrowed operand runs over the open rows of each tile, so its rows and
                // the narrowed tile count depend on the tiles' rows.
                if !selective || fused_tile_rows(t) == t {
                    assert_eq!(on.lines, off.lines, "trace and stats, {at}");
                    assert_eq!(on.narrowed, off.narrowed, "narrowed tiles, {at}");
                }
                if !selective {
                    assert_eq!(on.lines, untiled.lines, "untiled trace and stats, {at}");
                    assert_eq!(on.order, untiled.order, "untiled trace order, {at}");
                    assert_eq!(on.narrowed, 0, "{at}");
                }
                assert_eq!(on.fused, want_fused, "fused runs, {at}");
                assert!(off.fused.is_empty(), "unfused runs, {at}");
                assert_eq!(on.regions, regions_on, "fused regions, {at}");
                assert_eq!(off.regions, regions_off, "unfused regions, {at}");
            }
        }
    }
    want
}

/// `check` of a program that binds `s`, returning the rows of `s`.
fn check_s(src: &str, fused: &[(&str, u64)], regions_off: u64, regions_on: u64) -> Vec<String> {
    let (_, mut bound) = check(src, &["s"], fused, regions_off, regions_on).unwrap();
    bound.remove(0)
}

#[test]
fn fused_tile_rows_round_to_the_reduce_block() {
    assert_eq!(fused_tile_rows(1), 16);
    assert_eq!(fused_tile_rows(3), 16);
    assert_eq!(fused_tile_rows(16), 16);
    assert_eq!(fused_tile_rows(17), 32);
    assert_eq!(fused_tile_rows(48), 48);
    assert_eq!(fused_tile_rows(16384), 16384);
    assert_eq!(fused_tile_rows(usize::MAX), usize::MAX / 16 * 16);
}

#[test]
fn sum_over_a_chain() {
    let src = "let x = cast_f64(iota(100));
let s = sum((x * 3.0) + 1.0);";
    let s = check_s(src, &[("reduce_add_f64", 1)], 1, 1);
    assert_eq!(s, ["14950.0"]);
}

#[test]
fn sum_over_one_elementwise_kernel() {
    let src = "let x = cast_f64(iota(100));
let s = sum(x * 0.5);";
    let s = check_s(src, &[("reduce_add_f64", 1)], 0, 1);
    assert_eq!(s, ["2475.0"]);
}

#[test]
fn sum_over_a_view() {
    // The view starts at row 7 of its buffer.
    let src = "let v = slice(iota(120), 7, 107);
let s = sum((cast_f64(v) * 3.0) + 1.0);";
    let s = check_s(src, &[("reduce_add_f64", 1)], 1, 1);
    assert_eq!(s, ["17050.0"]);
}

#[test]
fn sum_over_an_empty_array() {
    let src = "let e = slice(iota(10), 3, 3);
let s = sum((cast_f64(e) * 3.0) + 1.0);";
    let s = check_s(src, &[("reduce_add_f64", 1)], 1, 1);
    assert_eq!(s, ["0.0"]);
}

#[test]
fn sum_over_a_multiple_of_the_tile_rows() {
    let src = "let x = cast_f64(iota(48));
let s = sum((x * 3.0) + 1.0);";
    let s = check_s(src, &[("reduce_add_f64", 1)], 1, 1);
    assert_eq!(s, ["3432.0"]);
}

#[test]
fn integer_dot_and_float_product_over_chains() {
    let dot = "let i = iota(100);
let s = sum((i + 1) * (i * 2));";
    assert_eq!(check_s(dot, &[("dot_i64", 1)], 0, 1), ["666600"]);
    let product = "let x = cast_f64(iota(100));
let s = reduce((x * 0.01) + 1.0, mul);";
    let want = (0..100).fold(1.0f64, |p, x| p * (f64::from(x) * 0.01 + 1.0));
    let s = check_s(product, &[("reduce_mul_f64", 1)], 1, 1);
    // The canonical order differs from a left fold, so only the magnitude is checked here,
    // and the bits against the whole call in `float_sums_and_products_keep_their_bits`.
    let got: f64 = s[0].parse().unwrap();
    assert!((got / want - 1.0).abs() < 1e-12, "{got} vs {want}");
}

#[test]
fn extremes_with_ties_in_later_tiles() {
    // The product is -0.0 at rows 40 and 60 and positive elsewhere, so the minimum ties in
    // two tiles of every fused tile row count. Its negation is +0.0 there.
    let chain = "((x - 40.0) * (x - 60.0)) * ((40.0 - x) * (60.0 - x))";
    let cases = [
        (format!("reduce({chain}, min)"), "reduce_min_f64", "-0.0"),
        (format!("argmin({chain})"), "argmin_f64", "40"),
        (
            format!("reduce(0.0 - ({chain}), max)"),
            "reduce_max_f64",
            "0.0",
        ),
        (format!("argmax(0.0 - ({chain}))"), "argmax_f64", "40"),
    ];
    for (expr, root, want) in cases {
        let src = format!("let x = cast_f64(iota(100));\nlet s = {expr};");
        assert_eq!(check_s(&src, &[(root, 1)], 1, 1), [want], "{src}");
    }
}

#[test]
fn extremes_with_a_nan_in_a_later_tile() {
    // 0 / 0 is NaN at row 70, every other row is -1.0.
    let chain = "(x - 70.0) / (70.0 - x)";
    let cases = [
        (format!("reduce({chain}, min)"), "reduce_min_f64", "NaN"),
        (format!("reduce({chain}, max)"), "reduce_max_f64", "NaN"),
        (format!("argmin({chain})"), "argmin_f64", "70"),
        (format!("argmax({chain})"), "argmax_f64", "70"),
    ];
    for (expr, root, want) in cases {
        let src = format!("let x = cast_f64(iota(100));\nlet s = {expr};");
        assert_eq!(check_s(&src, &[(root, 1)], 1, 1), [want], "{src}");
    }
}

#[test]
fn bool_reductions_over_a_guard() {
    // The guard is the root's argument, so it fills its whole tile before the update, and
    // with selective evaluation its right operand runs over fewer rows.
    let cases = [
        ("count(and(x > 10.0, x < 90.0))", "count", "79"),
        ("any(and(x > 95.0, x < 97.0))", "any", "true"),
        ("all(or(x < 50.0, x > 40.0))", "all", "true"),
        ("all(and(x > 10.0, x < 90.0))", "all", "false"),
    ];
    for (expr, root, want) in cases {
        let src = format!("let x = cast_f64(iota(100));\nlet s = {expr};");
        assert_eq!(check_s(&src, &[(root, 1)], 1, 1), [want], "{src}");
    }
}

#[test]
fn narrowed_tiles_count_the_same_with_fusion() {
    // Rows 95 to 99 leave the guard open, so a tile narrows unless all its rows are among
    // them. Fused tiles of 16 and 48 rows have the rows of unfused ones, and fused tiles of
    // 1 and 17 rows have 16 and 32.
    let src = "let x = cast_f64(iota(100));
let s = any(and(x > 94.0, x < 97.0));";
    let scalar = (Isa::Scalar, Backend::Auto);
    let narrowed = |t, fuse| run(src, &["s"], scalar, rows_of(t), true, fuse).narrowed;
    for t in [16, 48] {
        assert_eq!(narrowed(t, true), narrowed(t, false), "rows {t}");
    }
    // Every tile but the last, rows 96 to 99, narrows.
    assert_eq!(narrowed(16, true), 6);
    assert_eq!(narrowed(48, true), 2);
    assert_eq!(narrowed(1, true), 6);
    assert_eq!(narrowed(17, true), 3);
    // One-row tiles narrow on the closed rows 0 to 94.
    assert_eq!(narrowed(1, false), 95);
}

#[test]
fn dot_over_two_region_kernels_and_over_one() {
    let two = "let x = cast_f64(iota(100));
let s = sum((x + 1.0) * (x * 2.0));";
    assert_eq!(check_s(two, &[("dot_f64", 1)], 0, 1), ["666600.0"]);
    let one = "let x = cast_f64(iota(100));
let s = sum((x + 1.0) * x);";
    assert_eq!(check_s(one, &[("dot_f64", 1)], 0, 1), ["333300.0"]);
}

#[test]
fn a_root_with_a_scalar_argument() {
    // The plan is `count_gt_f64 (mul_f64 x 2.0) 50.0`, whose threshold is a splat.
    let src = "let x = cast_f64(iota(100));
let s = count((x * 2.0) > 50.0);";
    assert_eq!(check_s(src, &[("count_gt_f64", 1)], 0, 1), ["74"]);
}

#[test]
fn a_shared_root_fuses_on_its_first_evaluation() {
    // The sum is one node read by two statements. Its first evaluation fuses and the
    // second reads the memo.
    let src = "let x = cast_f64(iota(100));
let s = sum((x * 3.0) + 1.0);
let t = sum((x * 3.0) + 1.0) + 2.0;";
    let (_, bound) = check(src, &["s", "t"], &[("reduce_add_f64", 1)], 1, 1).unwrap();
    assert_eq!(bound, [text(["14950.0"]), text(["14952.0"])]);
}

#[test]
fn a_reduction_over_a_shared_elementwise_node_runs_whole() {
    // The chain is one node read by two reductions, so it is an input of each, and neither
    // has a region kernel argument.
    let src = "let x = cast_f64(iota(100));
let s = sum((x * 3.0) + 1.0);
let t = reduce((x * 3.0) + 1.0, max);";
    let (_, bound) = check(src, &["s", "t"], &[], 0, 0).unwrap();
    assert_eq!(bound, [text(["14950.0"]), text(["298.0"])]);
}

#[test]
fn a_reduction_over_a_leaf_runs_whole() {
    let src = "let x = cast_f64(iota(100));
let s = sum(x);";
    assert_eq!(check_s(src, &[], 0, 0), ["4950.0"]);
}

#[test]
fn trace_order_is_inputs_kernels_root() {
    // The gather is a region input before every region kernel, so the region fuses, and
    // its line comes first, then the region kernels in post-order, then the root.
    let src = "let i = iota(100);
let x = cast_f64(i);
let s = sum((gather(x, i) * 2.0) + (x * 3.0));";
    let s = check_s(src, &[("reduce_add_f64", 1)], 1, 1);
    assert_eq!(s, ["24750.0"]);
    let scalar = (Isa::Scalar, Backend::Auto);
    let ran = run(src, &["s"], scalar, rows_of(16), true, true);
    assert_eq!(
        ran.order,
        [
            "iota",
            "cast_i64_f64",
            "gather_f64",
            "mul_f64",
            "mul_f64",
            "add_f64",
            "reduce_add_f64"
        ]
    );
}

#[test]
fn prepared_execute_counts_fused_reductions() {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    })
    .unwrap();
    let prepared = interp
        .prepare("input x: f64[]; export s = sum((x * 3.0) + 1.0);")
        .unwrap();
    let x = Value::Array(Array::from_vec((0..1000).map(f64::from).collect()));
    for run in 1..=2 {
        let out = prepared
            .execute(&interp, &[("x", x.clone())], &mut std::io::sink())
            .unwrap();
        assert_eq!(out[0].1.to_string(), "1499500.0");
        let want = BTreeMap::from([("reduce_add_f64".to_string(), run)]);
        assert_eq!(interp.fused_reductions(), want);
        assert_eq!(interp.tiled_regions(), run);
    }
}

/// The values of the float bits tests: mixed magnitudes and signs, signed zeros, and every
/// fifth row of `z` a negative zero. With `nan`, row 700 of `x` holds that NaN.
fn float_leaves(n: usize, product: bool, nan: Option<f64>, f32_rows: bool) -> Vec<Leaf> {
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let i = f64::from(u32::try_from(i).unwrap());
            if product {
                1.0 + ((i * 13.0) % 17.0 - 8.0) * 1e-3
            } else {
                ((i * 37.0) % 101.0 - 50.0) * 10f64.powf(i % 13.0 - 6.0)
            }
        })
        .collect();
    if let Some(nan) = nan {
        x[700] = nan;
    }
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let i = f64::from(u32::try_from(i).unwrap());
            if i % 9.0 == 0.0 {
                -0.0
            } else {
                1.0 - (i % 7.0) * 1e-4
            }
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| {
            let i = f64::from(u32::try_from(i).unwrap());
            if i % 5.0 == 0.0 {
                -0.0
            } else {
                (i % 3.0) * 1e-9
            }
        })
        .collect();
    let leaf = |v: Vec<f64>| {
        let array = if f32_rows {
            Array::from_vec(v.into_iter().map(|r| r as f32).collect())
        } else {
            Array::from_vec(v)
        };
        Leaf::Const(Value::Array(array))
    };
    vec![leaf(x), leaf(y), leaf(z)]
}

/// `reduce (add (mul x y) z)` with the kernels of element type `ty` and reduction `reduce`.
fn float_chain(reg: &Registry, ty: &str, reduce: &str) -> Built {
    let id = |n: String| reg.by_name(&n).unwrap();
    let mut b = Builder::default();
    let (x, y, z) = (b.leaf(0), b.leaf(1), b.leaf(2));
    let product = b.kernel(id(format!("mul_{ty}")), vec![x, y]);
    let sum = b.kernel(id(format!("add_{ty}")), vec![product, z]);
    let root = b.kernel(id(format!("{reduce}_{ty}")), vec![sum]);
    b.finish(root)
}

/// The bits of a float scalar.
fn bits(slot: &Slot) -> u64 {
    match slot {
        Ok(Value::Scalar(Scalar::F64(x))) => x.to_bits(),
        Ok(Value::Scalar(Scalar::F32(x))) => u64::from(x.to_bits()),
        other => panic!("expected a float, got {other:?}"),
    }
}

/// Whether `bits` is the bits of a NaN of element type `ty`.
fn is_nan_bits(ty: &str, bits: u64) -> bool {
    match ty {
        "f64" => f64::from_bits(bits).is_nan(),
        _ => f32::from_bits(u32::try_from(bits).unwrap()).is_nan(),
    }
}

#[test]
fn float_reductions_keep_their_bits() {
    let reg = Registry::new();
    let scalar = (Isa::Scalar, Backend::Auto);
    let reductions = ["reduce_add", "reduce_mul", "reduce_min", "reduce_max"];
    // A NaN with the sign bit set is what x86 arithmetic produces, and what `neg` makes of
    // the default NaN.
    let nans = [None, Some(f64::NAN), Some(-f64::NAN)];
    for ty in ["f64", "f32"] {
        for reduce in reductions {
            for nan in nans {
                // 1000 rows end every tested tiling in a partial tile with a canonical tail.
                let leaves = float_leaves(1000, reduce == "reduce_mul", nan, ty == "f32");
                let node = float_chain(&reg, ty, reduce);
                let root = reg.by_name(&format!("{reduce}_{ty}")).unwrap();
                let whole = eval_with(&reg, &node, &leaves, scalar, Tiling::Off, true, true);
                let reference = bits(&whole.result);
                for variant in variants() {
                    let at = format!("{ty} {reduce} nan {nan:?} {variant:?}");
                    let untiled = eval_with(&reg, &node, &leaves, variant, Tiling::Off, true, true);
                    let want = bits(&untiled.result);
                    // Across variants a NaN result needs only be NaN, within one variant
                    // fusion keeps every bit.
                    if is_nan_bits(ty, reference) {
                        assert!(is_nan_bits(ty, want), "untiled {at}");
                    } else {
                        assert_eq!(want, reference, "untiled {at}");
                    }
                    let dispatch = reg.resolve(variant.0, variant.1, FeatureSet::EMPTY);
                    let free = dispatch.reducer_split(root) == Some(Split::Free);
                    for t in [16, 48, 16384] {
                        let on = eval_with(&reg, &node, &leaves, variant, rows_of(t), true, true);
                        let off = eval_with(&reg, &node, &leaves, variant, rows_of(t), true, false);
                        assert_eq!(bits(&on.result), want, "fused rows {t} {at}");
                        assert_eq!(bits(&off.result), want, "unfused rows {t} {at}");
                        let fused = u64::from(free);
                        assert_eq!((on.fused, on.regions), (fused, 1), "rows {t} {at}");
                        assert_eq!((off.fused, off.regions), (0, 1), "rows {t} {at}");
                    }
                }
            }
        }
    }
}

/// Runs `src` untiled and fused with every tile row count of `ROWS` on every variant, with
/// selective evaluation, and checks that each fails with the untiled scalar error by kind,
/// message, and span, and that no fused run completes. Returns the error.
fn check_error(src: &str) -> crate::error::Error {
    let scalar = (Isa::Scalar, Backend::Auto);
    let want = run(src, &[], scalar, Tiling::Off, true, true)
        .outcome
        .unwrap_err();
    for variant in variants() {
        for t in ROWS {
            for fuse in [true, false] {
                let got = run(src, &[], variant, rows_of(t), true, fuse);
                let e = got.outcome.unwrap_err();
                let at = format!("{variant:?} rows {t} fuse {fuse} of\n{src}");
                assert_eq!(
                    (&e.kind, &e.message, e.span),
                    (&want.kind, &want.message, want.span),
                    "{at}"
                );
                assert!(got.fused.is_empty(), "{at}");
            }
        }
    }
    want
}

#[test]
fn the_minimum_of_an_empty_chain_fails_in_finish_like_untiled() {
    let e = check_error(
        "let e = slice(cast_f64(iota(10)), 3, 3);
print(reduce((e * 2.0) + 1.0, min));",
    );
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert_eq!(e.message, "reduce_min of an empty array");
    assert_eq!(e.span.map(|s| s.line), Some(2));
}

#[test]
fn a_zero_divisor_under_a_guard_fails_like_untiled() {
    // `div_const` is not total, so the guard does not narrow its operand, which fails in
    // the first tile.
    let e = check_error(
        "let x = iota(100);
let z = length(slice(x, 0, 0));
print(count(and(x > 1000, div_const(x, z) > 1)));",
    );
    assert_eq!(e.message, "div_const: division by zero");
    assert_eq!(e.span.map(|s| s.line), Some(3));
}

#[test]
fn dot_over_arrays_of_different_lengths_fails_like_untiled() {
    let e = check_error(
        "let a = cast_f64(iota(40));
let b = cast_f64(iota(41));
print(sum((a + 1.0) * (b * 2.0)));",
    );
    assert_eq!(e.message, "length mismatch: 40 vs 41");
}

/// Sums its `i64` rows and fails on the first row holding `FAIL_AT`, naming that row's
/// position in the update, so a tiled update would report a different position. It also
/// fails on an update without rows, which a whole call never makes.
struct FailSum(i64);

const FAIL_AT: i64 = 90;

impl Reducer for FailSum {
    fn update(&mut self, args: &[Arg]) -> Result<(), String> {
        let [Arg::Rows(a)] = args else {
            return Err("fail_sum takes one array".into());
        };
        if a.is_empty() {
            return Err("fail_sum: an update without rows".into());
        }
        for (i, &x) in a.as_slice::<i64>().iter().enumerate() {
            if x == FAIL_AT {
                return Err(format!("fail_sum: row {i} holds {x}"));
            }
            self.0 += x;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Scalar, String> {
        Ok(Scalar::I64(self.0))
    }
}

impl ReductionBody for FailSum {
    fn new() -> Self {
        FailSum(0)
    }
}

/// `probe_registry` with `fail_sum_i64` added, whose entries have `split`, and its id.
fn fail_registry(split: Split) -> (Registry, crate::kernels::KernelId) {
    let (mut reg, ..) = probe_registry();
    let id = reg.add(KernelDef {
        name: "fail_sum_i64".into(),
        class: KernelClass::Reduction,
        op: None,
        inputs: vec![InputSpec::Array(ScalarType::I64)],
        output: ValueType::Scalar(ScalarType::I64),
        cost: 8,
        commutative: false,
        pattern: None,
        total: false,
        impls: KernelImpl::reduction::<FailSum>(split),
    });
    (reg, id)
}

/// Evaluates `node` over the leaves `0..n` and the scalar 1 untiled and with every tile row
/// count of `ROWS` on every variant, fused and unfused, and checks that each gives the
/// untiled scalar result, calls `Probe` `probes` times, and completes `fused` fused runs
/// when fused and tiled. Returns the untiled scalar result.
fn compare_fused(reg: &Registry, node: &Built, n: i64, fused: u64, probes: usize) -> Slot {
    let leaves = vec![
        Leaf::Const(Value::Array(Array::from_vec((0..n).collect()))),
        Leaf::Const(Value::Scalar(Scalar::I64(1))),
    ];
    let probed = |variant: (Isa, Backend), tiling: Tiling, fuse: bool| {
        PROBE_CALLS.with(|c| c.set(0));
        let got = eval_with(reg, node, &leaves, variant, tiling, true, fuse);
        let calls = PROBE_CALLS.with(Cell::get);
        assert_eq!(
            calls, probes,
            "probe calls of {variant:?} {tiling:?} {fuse}"
        );
        got
    };
    let want = probed((Isa::Scalar, Backend::Auto), Tiling::Off, true).result;
    for variant in variants() {
        for t in ROWS {
            for fuse in [true, false] {
                let got = probed(variant, rows_of(t), fuse);
                assert_eq!(got.result, want, "{variant:?} rows {t} fuse {fuse}");
                let fused = if fuse { fused } else { 0 };
                assert_eq!(got.fused, fused, "{variant:?} rows {t} fuse {fuse}");
            }
        }
    }
    want
}

#[test]
fn a_failing_update_after_earlier_tiles_raises_the_untiled_error() {
    let (reg, fail_sum) = fail_registry(Split::Free);
    let add = reg.by_name("add_i64").unwrap();
    let mut b = Builder::default();
    // fail_sum(x + 1) over 0..100 fails at the row holding 90, row 89 of the whole call and
    // a different row of any tile.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let inc = b.kernel(add, vec![x, one]);
    let root = b.kernel(fail_sum, vec![inc]);
    let node = b.finish(root);
    let e = compare_fused(&reg, &node, 100, 0, 0).unwrap_err();
    assert_eq!(e.message, "fail_sum: row 89 holds 90");
    assert_eq!(e.span, Some(Span { line: 3, col: 1 }));
    let sum = compare_fused(&reg, &node, 80, 1, 0).unwrap();
    assert_eq!(rows(&sum), text([(1..=80).sum::<i64>()]));
    // An empty array runs one empty tile and no update, so the fused run completes.
    let empty = compare_fused(&reg, &node, 0, 1, 0).unwrap();
    assert_eq!(rows(&empty), text([0]));
}

#[test]
fn a_split_bound_reduction_runs_unfused() {
    let (reg, bound_sum) = fail_registry(Split::Bound);
    let add = reg.by_name("add_i64").unwrap();
    let mut b = Builder::default();
    // fail_sum(x + 1) with entries whose bits may depend on the split never fuses, and
    // gives the values and errors of the whole call.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let inc = b.kernel(add, vec![x, one]);
    let root = b.kernel(bound_sum, vec![inc]);
    let node = b.finish(root);
    let sum = compare_fused(&reg, &node, 80, 0, 0).unwrap();
    assert_eq!(rows(&sum), text([(1..=80).sum::<i64>()]));
    let e = compare_fused(&reg, &node, 100, 0, 0).unwrap_err();
    assert_eq!(e.message, "fail_sum: row 89 holds 90");
}

#[test]
fn a_kernel_failing_in_a_later_tile_raises_the_untiled_error() {
    let (reg, fail, probe, add) = probe_registry();
    let sum = reg.by_name("reduce_add_i64").unwrap();
    let mut b = Builder::default();
    // sum(fail_at(probe(x) + 1) + 1): the probe is an input before every region kernel, so
    // the region fuses over 0..80. Over 0..100 `fail_at` fails at row 89 after earlier
    // tiles updated the sum, and the fallback reuses the probe's value.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let probed = b.kernel(probe, vec![x]);
    let inc = b.kernel(add, vec![probed, one]);
    let failed = b.kernel(fail, vec![inc]);
    let top = b.kernel(add, vec![failed, one]);
    let root = b.kernel(sum, vec![top]);
    let node = b.finish(root);
    let e = compare_fused(&reg, &node, 100, 0, 1).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
    let got = compare_fused(&reg, &node, 80, 1, 1).unwrap();
    assert_eq!(rows(&got), text([(2..82).sum::<i64>()]));
}

#[test]
fn an_input_after_a_region_kernel_is_never_evaluated_first() {
    let (reg, fail, probe, add) = probe_registry();
    let dot = reg.by_name("dot_i64").unwrap();
    let mut b = Builder::default();
    // dot(fail_at(x + 1), probe(x)): over 0..100 the whole evaluation fails in `fail_at`
    // before it reaches `probe`, which a fused run would evaluate first. Over 0..80 `probe`
    // runs once, after the left argument, and nothing fuses.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let inc = b.kernel(add, vec![x, one]);
    let failed = b.kernel(fail, vec![inc]);
    let probed = b.kernel(probe, vec![x]);
    let root = b.kernel(dot, vec![failed, probed]);
    let node = b.finish(root);
    let e = compare_fused(&reg, &node, 100, 0, 0).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
    let got = compare_fused(&reg, &node, 80, 0, 1).unwrap();
    assert_eq!(
        rows(&got),
        text([(0..80).map(|v: i64| (v + 1) * v).sum::<i64>()])
    );
}
