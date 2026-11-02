use std::cell::Cell;

use crate::error::{ErrorKind, Span};
use crate::exec::tests::{
    Builder, Built, assert_tiles, compare, eval_node, rows, text, tilings, variants,
};
use crate::exec::{Slot, Tiling};
use crate::kernels::{Backend, ElementwiseBody, InputSpec, Isa, KernelDef, KernelImpl, WholeBody};
use crate::kernels::{KernelClass, KernelId, Registry};
use crate::lower::Leaf;
use crate::value::{Arg, Array, Scalar, ScalarType, Value, ValueType};
use crate::{Interpreter, Options};

#[test]
fn mask_chain() {
    // Five comparisons combined with `and` and `or`, as in TPC-H q06.
    let src = "let x = iota(100);
let d = x * 3;
let m = and(and(x >= 10, x < 90), or(and(d >= 5, d <= 70), x < 24));
let s = sum(compress(x, m));";
    let m = (0..100).map(|x| (10..90).contains(&x) && ((5..=70).contains(&(x * 3)) || x < 24));
    assert_tiles(src, &[("m", text(m)), ("s", text([231]))]);
}

#[test]
fn arithmetic_chain_with_broadcast_scalars() {
    let src = "let x = iota(100);
let r = ((x * 3) + 7) - x;
let f = (fill(100, 0.5) * 3.0) + 1.0;";
    let r = (0..100).map(|x| x * 2 + 7);
    assert_tiles(src, &[("r", text(r)), ("f", text(["2.5"; 100]))]);
}

#[test]
fn chain_over_a_view() {
    // The view starts at row 7 of its buffer, and the mask chain reads it twice.
    let src = "let x = slice(iota(120), 7, 107);
let r = (x * 3) + 1;
let m = and(x > 20, x < 50);";
    let r = (7..107).map(|x| x * 3 + 1);
    let m = (7..107).map(|x| x > 20 && x < 50);
    assert_tiles(src, &[("r", text(r)), ("m", text(m))]);
}

#[test]
fn chain_over_an_empty_array() {
    let src = "let e = slice(iota(10), 3, 3);
let r = (e * 3) + 1;
let m = and(e > 1, e < 5);";
    let none = Vec::<String>::new();
    assert_tiles(src, &[("r", none.clone()), ("m", none)]);
}

#[test]
fn chain_over_a_multiple_of_the_tile_rows() {
    // 48 rows are 48, 16, and 3 tiles of the tested tile row counts.
    let src = "let x = iota(48);
let r = ((x * 3) + 1) - x;";
    assert_tiles(src, &[("r", text((0..48).map(|x| x * 2 + 1)))]);
}

#[test]
fn chain_in_the_taken_branch_tiles() {
    let src = "let x = iota(100);
let r = if length(x) > 0 { (x * 3) + 1 } else { x };";
    assert_tiles(src, &[("r", text((0..100).map(|x| x * 3 + 1)))]);
}

#[test]
fn chain_in_the_untaken_branch_does_not_run() {
    let src = "let x = iota(100);
let r = if length(x) > 1000 { (x * 3) + 1 } else { x };";
    let (outcome, counts) = compare(src, &["r"]);
    assert_eq!(outcome.unwrap().1, [text(0..100)]);
    assert!(counts.iter().all(|&c| c == 0), "{counts:?}");
}

#[test]
fn chain_between_other_kernels() {
    // A selection feeds the region and a reduction reads it, both run once and whole.
    let src = "let x = iota(100);
let c = compress(x, x > 30);
let s = sum((c * 2) + 1);
let t = sum(((compress(x, x > 30) * 2) + 1) - c);";
    let s: i64 = (31..100).map(|x| x * 2 + 1).sum();
    let t: i64 = (31..100).map(|x| x + 1).sum();
    assert_tiles(src, &[("s", text([s])), ("t", text([t]))]);
}

#[test]
fn failing_regions_raise_the_untiled_error() {
    // A length mismatch inside the region, a failing kernel in every tile, and an input
    // that fails before the region runs.
    for src in [
        "let r = (iota(5) + iota(6)) * 2; print(r);",
        "let z = length(slice(iota(3), 0, 0)); let r = div_const(iota(40) * 2, z); print(r);",
        "let r = (gather(iota(40), iota(40) - 1) * 2) + 1; print(r);",
    ] {
        let (outcome, counts) = compare(src, &[]);
        let e = outcome.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Runtime, "{src}");
        assert!(e.span.is_some(), "{src}");
        assert!(counts.iter().all(|&c| c == 0), "{counts:?} for\n{src}");
    }
}

#[test]
fn chains_with_unaligned_tiles_and_vector_tails() {
    // 1003 rows end every tested tiling, and every vector width, in a partial tile.
    let src = "let x = iota(1003);
let a = ((cast_f64(x) * 0.25) + 1.5) - cast_f64(x);
let b = (cast_f32(x) + cast_f32(x)) * cast_f32(x);
let c = ((cast_i32(x) + cast_i32(x)) * cast_i32(x)) - cast_i32(x);
let m = or(and(x > 100, x < 900), x == 1001);";
    let a = (0..1003).map(|x| format!("{:?}", (f64::from(x) * 0.25 + 1.5) - f64::from(x)));
    let b = (0..1003u16).map(|x| format!("{:?}", (f32::from(x) + f32::from(x)) * f32::from(x)));
    let c = (0..1003).map(|x| (x + x) * x - x);
    let m = (0..1003).map(|x| (x > 100 && x < 900) || x == 1001);
    assert_tiles(
        src,
        &[
            ("a", a.collect()),
            ("b", b.collect()),
            ("c", text(c)),
            ("m", text(m)),
        ],
    );
}

/// The kernel names of the trace lines of running `src` with `tiling`, in order.
fn trace_order(src: &str, tiling: Tiling) -> Vec<String> {
    let mut interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        trace: true,
        tiling,
        ..Options::default()
    })
    .unwrap();
    let mut diag = Vec::new();
    interp.run(src, &mut std::io::sink(), &mut diag).unwrap();
    String::from_utf8(diag)
        .unwrap()
        .lines()
        .map(|l| l.split(' ').nth(1).unwrap().to_string())
        .collect()
}

#[test]
fn region_lines_follow_their_inputs_in_post_order() {
    // In the first program the left operand of the root is a region kernel over a whole
    // kernel input, which the whole evaluation reaches before any region kernel, so the
    // region tiles. In the second the whole kernel input follows region kernels, so the
    // region runs whole. Either way the lines keep the order of the whole evaluation.
    let names = |n: &[&str]| n.iter().map(ToString::to_string).collect::<Vec<_>>();
    let cases = [
        (
            "let x = iota(100);
print(sum((gather(x, x) * 2) + ((x * 3) - x)));",
            names(&[
                "iota",
                "gather_i64",
                "mul_i64",
                "mul_i64",
                "sub_i64",
                "add_i64",
                "reduce_add_i64",
            ]),
            1,
        ),
        (
            "let x = iota(100);
print(sum(((x * 3) - x) + (gather(x, x) * 2)));",
            names(&[
                "iota",
                "mul_i64",
                "sub_i64",
                "gather_i64",
                "mul_i64",
                "add_i64",
                "reduce_add_i64",
            ]),
            0,
        ),
    ];
    for (src, want, regions) in cases {
        for tiling in tilings() {
            assert_eq!(trace_order(src, tiling), want, "{tiling:?} of\n{src}");
        }
        let (_, counts) = compare(src, &[]);
        assert!(
            counts.iter().all(|&c| c == regions),
            "{counts:?} for\n{src}"
        );
    }
}

#[test]
fn a_left_kernel_error_wins_over_a_right_input_error() {
    // The whole evaluation fails in `div_const` before it reaches the failing right input.
    // That input follows region kernels, so the region runs whole and never evaluates it,
    // and the trace and stats lines match the untiled run.
    // A failed binding of the same program raises at its statement, so only an earlier
    // program's failed global reaches a region as a variable, see the test below.
    for right in [
        "gather(iota(40), iota(40) - 1)",
        "load_i64(\"/nonexistent/x.bin\")",
    ] {
        let src = format!(
            "let z = length(slice(iota(3), 0, 0));
let r = div_const(iota(40) * 2, z) + {right};
print(r);"
        );
        let (outcome, counts) = compare(&src, &[]);
        let e = outcome.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Runtime, "{src}");
        assert_eq!(e.message, "div_const: division by zero", "{src}");
        assert!(counts.iter().all(|&c| c == 0), "{counts:?} for\n{src}");
    }
}

#[test]
fn a_failed_global_of_an_earlier_program_raises_like_untiled() {
    // `g` fails in the first program. The second program's region reads it after a left
    // `div_const` that fails, so the untiled error is the `div_const` one.
    let outcome = |tiling| {
        let mut interp = Interpreter::new(Options {
            isa: Some(Isa::Scalar),
            tiling,
            ..Options::default()
        })
        .unwrap();
        let (mut out, mut diag) = (Vec::new(), Vec::new());
        // The unread binding fails and stays bound as an error.
        interp
            .run(
                "let _g = gather(iota(40), iota(40) - 1);\nprint(1);\n",
                &mut out,
                &mut diag,
            )
            .unwrap();
        let src = "let z = length(slice(iota(3), 0, 0));
let r = div_const(iota(40) * 2, z) + _g;
print(r);";
        interp.run(src, &mut out, &mut diag).unwrap_err()
    };
    let want = outcome(Tiling::Off);
    assert_eq!(want.kind, ErrorKind::Runtime);
    assert_eq!(want.message, "div_const: division by zero");
    for tiling in tilings().into_iter().skip(1) {
        let got = outcome(tiling);
        assert_eq!((got.kind, got.message), (want.kind, want.message.clone()));
    }
}

#[test]
fn prepared_execute_counts_tiled_regions() {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    })
    .unwrap();
    let prepared = interp
        .prepare("input x: i64[]; export y = (x * 3) + 1;")
        .unwrap();
    let x = Value::Array(Array::from_vec((0..10_000i64).collect()));
    let out = prepared
        .execute(&interp, &[("x", x)], &mut std::io::sink())
        .unwrap();
    let Value::Array(y) = &out[0].1 else {
        panic!("expected an array");
    };
    assert_eq!(y.as_slice::<i64>()[9_999], 29_998);
    assert_eq!(interp.tiled_regions(), 1);
}

/// Copies its `i64` rows and fails on the first row holding `FAIL_AT`, naming that row's
/// position in the call's arguments, so a tiled call would report a different position.
struct FailAt;

const FAIL_AT: i64 = 90;

impl ElementwiseBody for FailAt {
    fn run(args: &[Arg], out: &mut [u8]) -> Result<(), String> {
        let [Arg::Rows(a)] = args else {
            return Err("fail_at takes one array".into());
        };
        let out: &mut [i64] = bytemuck::cast_slice_mut(out);
        for (i, (o, &x)) in out.iter_mut().zip(a.as_slice::<i64>()).enumerate() {
            if x == FAIL_AT {
                return Err(format!("fail_at: row {i} holds {x}"));
            }
            *o = x;
        }
        Ok(())
    }
}
#[test]
fn kernel_failing_in_a_later_tile_raises_the_untiled_error() {
    let mut reg = Registry::new();
    let fail = reg.add(KernelDef {
        name: "fail_at_i64".into(),
        class: KernelClass::Elementwise,
        op: None,
        inputs: vec![InputSpec::Array(ScalarType::I64)],
        output: ValueType::Array(ScalarType::I64),
        cost: 16,
        commutative: false,
        pattern: None,
        total: false,
        impls: KernelImpl::elementwise_scalar::<FailAt>().scalar_only(),
    });
    let add = reg.by_name("add_i64").unwrap();
    let mut b = Builder::default();
    // (fail_at (x + 1)) + 1 over x = 0..100 fails at the row holding 90, row 89 of x.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let inc = b.kernel(add, vec![x, one]);
    let failed = b.kernel(fail, vec![inc]);
    let root = b.kernel(add, vec![failed, one]);
    let node = b.finish(root);
    let leaves = |n: i64| {
        vec![
            Leaf::Const(Value::Array(Array::from_vec((0..n).collect()))),
            Leaf::Const(Value::Scalar(Scalar::I64(1))),
        ]
    };
    let scalar = (Isa::Scalar, Backend::Auto);
    let (untiled, _) = eval_node(&reg, &node, &leaves(100), scalar, Tiling::Off);
    let want = untiled.unwrap_err();
    assert_eq!(want.message, "fail_at: row 89 holds 90");
    assert_eq!(want.span, Some(Span { line: 3, col: 1 }));
    // Without the failing row every tiled run completes and matches the untiled value.
    let (passing, _) = eval_node(&reg, &node, &leaves(80), scalar, Tiling::Off);
    let Ok(Value::Array(whole)) = passing else {
        panic!("expected an array, got {passing:?}");
    };
    for variant in variants() {
        for tiling in tilings() {
            let (got, regions) = eval_node(&reg, &node, &leaves(100), variant, tiling);
            assert_eq!(got.unwrap_err(), want, "{variant:?} {tiling:?}");
            assert_eq!(regions, 0, "{variant:?} {tiling:?}");
        }
        for tiling in tilings().into_iter().skip(1) {
            let (got, regions) = eval_node(&reg, &node, &leaves(80), variant, tiling);
            let Ok(Value::Array(got)) = got else {
                panic!("expected an array, got {got:?}");
            };
            assert_eq!(
                got.as_slice::<i64>(),
                whole.as_slice::<i64>(),
                "{variant:?} {tiling:?}"
            );
            assert_eq!(regions, 1, "{variant:?} {tiling:?}");
        }
    }
}

thread_local! {
    /// The calls of `Probe` on this thread.
    static PROBE_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Returns its one argument and counts the call in `PROBE_CALLS`. Its class is `Whole`, so
/// it is never part of a region, only a region input.
struct Probe;

impl WholeBody for Probe {
    fn run(args: &[Value]) -> Result<Value, String> {
        PROBE_CALLS.with(|c| c.set(c.get() + 1));
        match args {
            [v] => Ok(v.clone()),
            _ => Err("probe takes one array".into()),
        }
    }
}

/// A registry with `fail_at_i64` and `probe_i64` added, and the ids of `fail_at_i64`,
/// `probe_i64`, and `add_i64`.
fn probe_registry() -> (Registry, KernelId, KernelId, KernelId) {
    let mut reg = Registry::new();
    let fail = reg.add(KernelDef {
        name: "fail_at_i64".into(),
        class: KernelClass::Elementwise,
        op: None,
        inputs: vec![InputSpec::Array(ScalarType::I64)],
        output: ValueType::Array(ScalarType::I64),
        cost: 16,
        commutative: false,
        pattern: None,
        total: false,
        impls: KernelImpl::elementwise_scalar::<FailAt>().scalar_only(),
    });
    let probe = reg.add(KernelDef {
        name: "probe_i64".into(),
        class: KernelClass::Whole,
        op: None,
        inputs: vec![InputSpec::Array(ScalarType::I64)],
        output: ValueType::Array(ScalarType::I64),
        cost: 16,
        commutative: false,
        pattern: None,
        total: false,
        impls: KernelImpl::whole::<Probe>(),
    });
    let add = reg.by_name("add_i64").unwrap();
    (reg, fail, probe, add)
}

/// Evaluates `node` over the leaves `0..n` and the scalar 1 with every tiling on every
/// variant, and checks that each run gives the untiled scalar result, tiles `regions`
/// regions when tiled, and calls `Probe` `probes` times. Returns the untiled scalar result.
fn compare_probed(reg: &Registry, node: &Built, n: i64, regions: u64, probes: usize) -> Slot {
    let leaves = vec![
        Leaf::Const(Value::Array(Array::from_vec((0..n).collect()))),
        Leaf::Const(Value::Scalar(Scalar::I64(1))),
    ];
    let probed = |variant: (Isa, Backend), tiling: Tiling| {
        PROBE_CALLS.with(|c| c.set(0));
        let (got, count) = eval_node(reg, node, &leaves, variant, tiling);
        let calls = PROBE_CALLS.with(Cell::get);
        assert_eq!(calls, probes, "probe calls of {variant:?} {tiling:?}");
        (got.map(|v| rows(&v)), count)
    };
    let (want, _) = probed((Isa::Scalar, Backend::Auto), Tiling::Off);
    for variant in variants() {
        for tiling in tilings() {
            let (got, count) = probed(variant, tiling);
            assert_eq!(got, want, "{variant:?} {tiling:?}");
            let regions = if tiling == Tiling::Off { 0 } else { regions };
            assert_eq!(count, regions, "{variant:?} {tiling:?}");
        }
    }
    eval_node(
        reg,
        node,
        &leaves,
        (Isa::Scalar, Backend::Auto),
        Tiling::Off,
    )
    .0
}

#[test]
fn an_input_after_a_failing_kernel_is_never_evaluated() {
    let (reg, fail, probe, add) = probe_registry();
    let mut b = Builder::default();
    // fail_at(x + 1) + probe(x): over 0..100 the whole evaluation fails in `fail_at` before
    // it reaches `probe`, which a tiled run would evaluate first. Over 0..80 `probe` runs
    // once, after the left operand, and the region runs whole.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let inc = b.kernel(add, vec![x, one]);
    let failed = b.kernel(fail, vec![inc]);
    let probed = b.kernel(probe, vec![x]);
    let root = b.kernel(add, vec![failed, probed]);
    let node = b.finish(root);
    let e = compare_probed(&reg, &node, 100, 0, 0).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
    compare_probed(&reg, &node, 80, 0, 1).unwrap();
}

#[test]
fn an_input_before_the_first_region_kernel_tiles() {
    let (reg, fail, probe, add) = probe_registry();
    let mut b = Builder::default();
    // fail_at(probe(x) + 1) + 1: the whole evaluation reaches `probe` before any region
    // kernel, so the region tiles over 0..80. Over 0..100 the tiled run fails, and its
    // fallback reuses the value of `probe` instead of evaluating it again.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let probed = b.kernel(probe, vec![x]);
    let inc = b.kernel(add, vec![probed, one]);
    let failed = b.kernel(fail, vec![inc]);
    let root = b.kernel(add, vec![failed, one]);
    let node = b.finish(root);
    let got = compare_probed(&reg, &node, 80, 1, 1).unwrap();
    assert_eq!(rows(&got), text(2..82));
    let e = compare_probed(&reg, &node, 100, 0, 1).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
}

#[test]
fn a_kernel_reached_twice_runs_once_outside_any_region() {
    let (reg, _, probe, add) = probe_registry();
    let mul = reg.by_name("mul_i64").unwrap();
    // (y + 1) * (y + 1) with the sum one node, where `y` is `x` or `probe(x)`.
    let square = |probed: bool| {
        let mut b = Builder::default();
        let (x, one) = (b.leaf(0), b.bcast(1));
        let y = if probed { b.kernel(probe, vec![x]) } else { x };
        let inc = b.kernel(add, vec![y, one]);
        let root = b.kernel(mul, vec![inc, inc]);
        b.finish(root)
    };
    let want = text((1..81).map(|v: i64| v * v));
    // The sum has two uses, so it is shared and never a region kernel. The product is then
    // the only elementwise kernel above it, which is no region, and the sum runs once.
    let leaves = compare_probed(&reg, &square(false), 80, 0, 0).unwrap();
    assert_eq!(rows(&leaves), want);
    // Below the shared sum, `probe` runs once too.
    let probed = compare_probed(&reg, &square(true), 80, 0, 1).unwrap();
    assert_eq!(rows(&probed), want);
}

#[test]
fn a_shared_input_reached_again_in_its_region_tiles() {
    let (reg, _, probe, add) = probe_registry();
    let (mul, sub) = (
        reg.by_name("mul_i64").unwrap(),
        reg.by_name("sub_i64").unwrap(),
    );
    let mut b = Builder::default();
    // (s + x) * (s - x) with s = probe(x) + 1 shared. The region's first input is s, before
    // any region kernel, and its second s is an earlier input reached again, which whole
    // evaluation finds in the memo, so the region tiles and `probe` runs once.
    let (x, one) = (b.leaf(0), b.bcast(1));
    let probed = b.kernel(probe, vec![x]);
    let s = b.kernel(add, vec![probed, one]);
    let left = b.kernel(add, vec![s, x]);
    let right = b.kernel(sub, vec![s, x]);
    let root = b.kernel(mul, vec![left, right]);
    let got = compare_probed(&reg, &b.finish(root), 80, 1, 1).unwrap();
    assert_eq!(rows(&got), text((0..80).map(|v: i64| 2 * v + 1)));
}

#[test]
fn a_shared_input_after_a_region_kernel_runs_whole() {
    let (reg, _, probe, add) = probe_registry();
    let mul = reg.by_name("mul_i64").unwrap();
    let mut b = Builder::default();
    // ((x * 3) + s) * s with s = probe(x) + 1 shared. Whole evaluation calls `x * 3` before
    // it first reaches s, which has not run, so the region runs whole. The second s is a
    // memo hit, and `probe` runs once.
    let (x, one, three) = (b.leaf(0), b.bcast(1), b.bcast(2));
    let probed = b.kernel(probe, vec![x]);
    let s = b.kernel(add, vec![probed, one]);
    let tripled = b.kernel(mul, vec![x, three]);
    let inner = b.kernel(add, vec![tripled, s]);
    let root = b.kernel(mul, vec![inner, s]);
    let node = b.finish(root);
    let leaves = vec![
        Leaf::Const(Value::Array(Array::from_vec((0..80i64).collect()))),
        Leaf::Const(Value::Scalar(Scalar::I64(1))),
        Leaf::Const(Value::Scalar(Scalar::I64(3))),
    ];
    let want = text((0..80).map(|v: i64| (4 * v + 1) * (v + 1)));
    for variant in variants() {
        for tiling in tilings() {
            PROBE_CALLS.with(|c| c.set(0));
            let (got, regions) = eval_node(&reg, &node, &leaves, variant, tiling);
            assert_eq!(rows(&got.unwrap()), want, "{variant:?} {tiling:?}");
            assert_eq!(regions, 0, "{variant:?} {tiling:?}");
            assert_eq!(PROBE_CALLS.with(Cell::get), 1, "{variant:?} {tiling:?}");
        }
    }
}

#[test]
fn a_mask_chain_over_leaves_tiles() {
    // and(x >= 10, x < 90) as in q40: every input is a leaf or a broadcast leaf.
    let reg = Registry::new();
    let id = |n| reg.by_name(n).unwrap();
    let mut b = Builder::default();
    let col = b.leaf(0);
    let mut cmp = |name, c| {
        let k = b.bcast(c);
        b.kernel(id(name), vec![col, k])
    };
    let (ge, lt) = (cmp("ge_i64", 1), cmp("lt_i64", 2));
    let root = b.kernel(id("and_bool"), vec![ge, lt]);
    let node = b.finish(root);
    let leaves = vec![
        Leaf::Const(Value::Array(Array::from_vec((0..100i64).collect()))),
        Leaf::Const(Value::Scalar(Scalar::I64(10))),
        Leaf::Const(Value::Scalar(Scalar::I64(90))),
    ];
    let want = text((0..100).map(|x| (10..90).contains(&x)));
    for variant in variants() {
        for tiling in tilings().into_iter().skip(1) {
            let (got, regions) = eval_node(&reg, &node, &leaves, variant, tiling);
            assert_eq!(rows(&got.unwrap()), want, "{variant:?} {tiling:?}");
            assert_eq!(regions, 1, "{variant:?} {tiling:?}");
        }
    }
}
