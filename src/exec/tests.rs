use std::num::NonZeroUsize;

use super::*;
use crate::kernels::{Backend, ElementwiseBody, FeatureSet, InputSpec, Isa, KernelDef, KernelImpl};
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
    // Times differ between runs, and the order of a region's lines relative to its inputs'
    // lines may differ from the whole evaluation, so the lines compare as a sorted list.
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
/// of the untiled scalar run and, with `same_trace`, the trace and stats lines of the
/// untiled run of its variant. Returns the untiled outcome and the tiled region count of
/// each tiled run.
fn compare_traced(src: &str, names: &[&str], same_trace: bool) -> (Outcome, Vec<u64>) {
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
            if same_trace {
                assert_eq!(
                    lines, off_lines,
                    "{variant:?} {t:?} trace and stats of\n{src}"
                );
            }
            counts.push(regions);
        }
    }
    (want, counts)
}

/// `compare_traced` with equal trace and stats lines.
fn compare(src: &str, names: &[&str]) -> (Outcome, Vec<u64>) {
    compare_traced(src, names, true)
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
    // The right operand of the root is a region kernel over a whole kernel input.
    let src = "let x = iota(100);
print(sum(((x * 3) - x) + (gather(x, x) * 2)));";
    let names = |n: &[&str]| n.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        trace_order(src, Tiling::Off),
        names(&[
            "iota",
            "mul_i64",
            "sub_i64",
            "gather_i64",
            "mul_i64",
            "add_i64",
            "reduce_add_i64"
        ]),
    );
    for tiling in tilings().into_iter().skip(1) {
        assert_eq!(
            trace_order(src, tiling),
            names(&[
                "iota",
                "gather_i64",
                "mul_i64",
                "sub_i64",
                "mul_i64",
                "add_i64",
                "reduce_add_i64"
            ]),
            "{tiling:?}",
        );
    }
}

#[test]
fn a_left_kernel_error_wins_over_a_right_input_error() {
    // The whole evaluation fails in `div_const` before it reaches the failing right input.
    // The tiled attempt evaluates that input first and fails there, and its fallback must
    // still raise the `div_const` error. The attempt's extra trace lines of the right input
    // are allowed, so traces are not compared.
    for right in [
        "gather(iota(40), iota(40) - 1)",
        "load_i64(\"/nonexistent/x.bin\")",
    ] {
        let src = format!(
            "let z = length(slice(iota(3), 0, 0));
let r = div_const(iota(40) * 2, z) + {right};
print(r);"
        );
        let (outcome, counts) = compare_traced(&src, &[], false);
        let e = outcome.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Runtime, "{src}");
        assert_eq!(e.message, "div_const: division by zero", "{src}");
        assert!(counts.iter().all(|&c| c == 0), "{counts:?} for\n{src}");
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
