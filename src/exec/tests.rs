use std::num::NonZeroUsize;

use super::*;
use std::cell::Cell;

use crate::kernels::{
    Backend, ElementwiseBody, FeatureSet, InputSpec, Isa, KernelDef, KernelImpl, WholeBody,
};
use crate::lower::LeafId;
use crate::{Interpreter, Options};

/// The tilings every test compares, untiled first. 3 and 37 rows put tile boundaries at
/// rows that are not multiples of any vector width, so tiles start unaligned and end in
/// vector tails.
fn tilings() -> Vec<Tiling> {
    let rows = |n| Tiling::Rows(NonZeroUsize::new(n).unwrap());
    vec![Tiling::Off, rows(1), rows(3), rows(16), rows(37)]
}

/// Every instruction set this host runs, with both backends, scalar first.
fn variants() -> Vec<(Isa, Backend)> {
    Isa::ALL
        .into_iter()
        .filter(|isa| isa.is_available_with(FeatureSet::EMPTY))
        .flat_map(|isa| [(isa, Backend::Auto), (isa, Backend::Intrinsics)])
        .collect()
}

/// The rows of a value as text, every row in full. Floats print as `{:?}`, so distinct
/// values print differently.
fn rows(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => (0..a.len()).map(|i| a.get(i).to_string()).collect(),
        other => vec![other.to_string()],
    }
}

/// What a program run produced: the printed output and the rows of every binding of the
/// names asked for, or the error.
type Outcome = Result<(String, Vec<Vec<String>>), Error>;

/// Runs `src` with `tiling` on the instruction set and backend of `variant`, keeping its
/// `let`s bound. Returns the outcome with the rows of `names`, the trace and stats lines
/// without their times, and the tiled region count.
fn run(
    src: &str,
    names: &[&str],
    variant: (Isa, Backend),
    tiling: Tiling,
) -> (Outcome, Vec<String>, u64) {
    let mut interp = Interpreter::new(Options {
        isa: Some(variant.0),
        backend: variant.1,
        trace: true,
        stats: true,
        tiling,
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
    interp.write_stats(&mut diag);
    // Times differ between runs. The lines compare as a sorted list, and a test checks their
    // order separately, see `region_lines_follow_their_inputs_in_post_order`.
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
    (outcome, lines, interp.tiled_regions())
}

/// Runs `src` under every tiling on every variant and checks that each gives the outcome
/// of the untiled scalar run and the trace and stats lines of the untiled run of its
/// variant. Returns the untiled outcome and the tiled region count of each tiled run.
fn compare(src: &str, names: &[&str]) -> (Outcome, Vec<u64>) {
    let (want, _, _) = run(src, names, (Isa::Scalar, Backend::Auto), Tiling::Off);
    let mut counts = Vec::new();
    for variant in variants() {
        let mut runs = tilings()
            .into_iter()
            .map(|t| (t, run(src, names, variant, t)));
        let (_, (off, off_lines, off_regions)) = runs.next().unwrap();
        assert_eq!(off, want, "{variant:?} untiled outcome of\n{src}");
        assert_eq!(off_regions, 0, "untiled runs tile no region");
        for (t, (got, lines, regions)) in runs {
            assert_eq!(got, want, "{variant:?} {t:?} outcome of\n{src}");
            assert_eq!(
                lines, off_lines,
                "{variant:?} {t:?} trace and stats of\n{src}"
            );
            counts.push(regions);
        }
    }
    (want, counts)
}

/// Checks that `src` binds every name of `want` to its rows under every tiling, and tiles
/// a region in each tiled run.
fn assert_tiles(src: &str, want: &[(&str, Vec<String>)]) {
    let names: Vec<&str> = want.iter().map(|(n, _)| *n).collect();
    let (outcome, counts) = compare(src, &names);
    let (_, bound) = outcome.unwrap();
    for ((name, rows), got) in want.iter().zip(bound) {
        assert_eq!(&got, rows, "{name} of\n{src}");
    }
    assert!(counts.iter().all(|&c| c > 0), "{counts:?} for\n{src}");
}

fn text<T: ToString>(rows: impl IntoIterator<Item = T>) -> Vec<String> {
    rows.into_iter().map(|r| r.to_string()).collect()
}

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

/// Evaluates `node` over `leaves` with `tiling` on the instruction set and backend of
/// `variant`, and returns the result and the tiled region count.
fn eval_node(
    reg: &Registry,
    node: &PlanNode,
    leaves: &[Leaf],
    variant: (Isa, Backend),
    tiling: Tiling,
) -> (Slot, u64) {
    let dispatch = reg.resolve(variant.0, variant.1, FeatureSet::EMPTY);
    let tiled = AtomicU64::new(0);
    let env = Env::new();
    let result = Machine {
        reg,
        dispatch: &dispatch,
        leaves,
        env: &env,
        span: Span { line: 3, col: 1 },
        trace: None,
        stats: None,
        tiling,
        tiled_regions: &tiled,
    }
    .eval(node);
    (result, tiled.load(Ordering::Relaxed))
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
        impls: KernelImpl::elementwise_scalar::<FailAt>().scalar_only(),
    });
    let add = reg.by_name("add_i64").unwrap();
    let leaf = |i| Box::new(PlanNode::Leaf(LeafId(i)));
    let one = || PlanNode::Bcast(leaf(1));
    // (fail_at (x + 1)) + 1 over x = 0..100 fails at the row holding 90, row 89 of x.
    let node = PlanNode::Kernel(
        add,
        vec![
            PlanNode::Kernel(fail, vec![PlanNode::Kernel(add, vec![*leaf(0), one()])]),
            one(),
        ],
    );
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
        impls: KernelImpl::whole::<Probe>(),
    });
    let add = reg.by_name("add_i64").unwrap();
    (reg, fail, probe, add)
}

/// Evaluates `node` over the leaves `0..n` and the scalar 1 with every tiling on every
/// variant, and checks that each run gives the untiled scalar result, tiles `regions`
/// regions when tiled, and calls `Probe` `probes` times. Returns the untiled scalar result.
fn compare_probed(reg: &Registry, node: &PlanNode, n: i64, regions: u64, probes: usize) -> Slot {
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
    let leaf = |i| PlanNode::Leaf(LeafId(i));
    // fail_at(x + 1) + probe(x): over 0..100 the whole evaluation fails in `fail_at` before
    // it reaches `probe`, which a tiled run would evaluate first. Over 0..80 `probe` runs
    // once, after the left operand, and the region runs whole.
    let node = PlanNode::Kernel(
        add,
        vec![
            PlanNode::Kernel(
                fail,
                vec![PlanNode::Kernel(
                    add,
                    vec![leaf(0), PlanNode::Bcast(Box::new(leaf(1)))],
                )],
            ),
            PlanNode::Kernel(probe, vec![leaf(0)]),
        ],
    );
    let e = compare_probed(&reg, &node, 100, 0, 0).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
    compare_probed(&reg, &node, 80, 0, 1).unwrap();
}

#[test]
fn an_input_before_the_first_region_kernel_tiles() {
    let (reg, fail, probe, add) = probe_registry();
    let leaf = |i| PlanNode::Leaf(LeafId(i));
    let one = || PlanNode::Bcast(Box::new(leaf(1)));
    // fail_at(probe(x) + 1) + 1: the whole evaluation reaches `probe` before any region
    // kernel, so the region tiles over 0..80. Over 0..100 the tiled run fails, and its
    // fallback reuses the value of `probe` instead of evaluating it again.
    let node = PlanNode::Kernel(
        add,
        vec![
            PlanNode::Kernel(
                fail,
                vec![PlanNode::Kernel(
                    add,
                    vec![PlanNode::Kernel(probe, vec![leaf(0)]), one()],
                )],
            ),
            one(),
        ],
    );
    let got = compare_probed(&reg, &node, 80, 1, 1).unwrap();
    assert_eq!(rows(&got), text(2..82));
    let e = compare_probed(&reg, &node, 100, 0, 1).unwrap_err();
    assert_eq!(e.message, "fail_at: row 89 holds 90");
}

#[test]
fn a_mask_chain_over_leaves_tiles() {
    // and(x >= 10, x < 90) as in q40: every input is a leaf or a broadcast leaf.
    let reg = Registry::new();
    let id = |n| reg.by_name(n).unwrap();
    let leaf = |i| PlanNode::Leaf(LeafId(i));
    let cmp =
        |name, c| PlanNode::Kernel(id(name), vec![leaf(0), PlanNode::Bcast(Box::new(leaf(c)))]);
    let node = PlanNode::Kernel(id("and_bool"), vec![cmp("ge_i64", 1), cmp("lt_i64", 2)]);
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
