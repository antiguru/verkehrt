use std::num::NonZeroUsize;

use super::*;

use crate::kernels::{Backend, FeatureSet, Isa};
use crate::lower::LeafId;
use crate::value::Array;
use crate::{Interpreter, Options};

/// The tilings every test compares, untiled first. 3 and 37 rows put tile boundaries at
/// rows that are not multiples of any vector width, so tiles start unaligned and end in
/// vector tails.
pub(super) fn tilings() -> Vec<Tiling> {
    let rows = |n| Tiling::Rows(NonZeroUsize::new(n).unwrap());
    vec![Tiling::Off, rows(1), rows(3), rows(16), rows(37)]
}

/// Every instruction set this host runs, with both backends, scalar first.
pub(super) fn variants() -> Vec<(Isa, Backend)> {
    Isa::ALL
        .into_iter()
        .filter(|isa| isa.is_available_with(FeatureSet::EMPTY))
        .flat_map(|isa| [(isa, Backend::Auto), (isa, Backend::Intrinsics)])
        .collect()
}

/// The rows of a value as text, every row in full. Floats print as `{:?}`, so distinct
/// values print differently.
pub(super) fn rows(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => (0..a.len()).map(|i| a.get(i).to_string()).collect(),
        other => vec![other.to_string()],
    }
}

/// What a program run produced: the printed output and the rows of every binding of the
/// names asked for, or the error.
pub(super) type Outcome = Result<(String, Vec<Vec<String>>), Error>;

/// What `run` gave: the outcome with the rows of the names asked for, the trace and stats
/// lines without their times, the tiled region count, and the narrowed tile count.
pub(super) struct Ran {
    pub(super) outcome: Outcome,
    pub(super) lines: Vec<String>,
    pub(super) regions: u64,
    pub(super) narrowed: u64,
}

/// Runs `src` with `tiling`, with or without selective evaluation, on the instruction set
/// and backend of `variant`, keeping its `let`s bound.
pub(super) fn run(
    src: &str,
    names: &[&str],
    variant: (Isa, Backend),
    tiling: Tiling,
    selective: bool,
) -> Ran {
    let mut interp = Interpreter::new(Options {
        isa: Some(variant.0),
        backend: variant.1,
        trace: true,
        stats: true,
        tiling,
        no_selective: !selective,
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
    Ran {
        outcome,
        lines,
        regions: interp.tiled_regions(),
        narrowed: interp.narrowed_tiles(),
    }
}

/// Trace and stats lines without the fields that count rows and bytes, which keeps the
/// kernel, its variant, and its calls.
pub(super) fn calls_only(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|l| {
            l.split(' ')
                .filter(|w| {
                    !["n=", "read=", "write=", "bytes="]
                        .iter()
                        .any(|p| w.starts_with(p))
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Runs `src` under every tiling on every variant, with and without selective evaluation,
/// and checks that each gives the outcome of the untiled scalar run. Without selective
/// evaluation each run gives the trace and stats lines of the untiled run of its variant,
/// and with it the same kernel calls and tiled regions. Returns the untiled outcome, the
/// tiled region count of each tiled run, and the narrowed tile count of each selective run.
pub(super) fn compare_narrowed(src: &str, names: &[&str]) -> (Outcome, Vec<u64>, Vec<u64>) {
    let scalar = (Isa::Scalar, Backend::Auto);
    let want = run(src, names, scalar, Tiling::Off, false).outcome;
    let mut counts = Vec::new();
    let mut narrowed = Vec::new();
    for variant in variants() {
        let off = run(src, names, variant, Tiling::Off, false);
        assert_eq!(off.outcome, want, "{variant:?} untiled outcome of\n{src}");
        assert_eq!(off.regions, 0, "untiled runs tile no region");
        // Selective evaluation needs tiles, so without them it changes nothing.
        let untiled = run(src, names, variant, Tiling::Off, true);
        assert_eq!(
            untiled.outcome, want,
            "{variant:?} untiled outcome of\n{src}"
        );
        assert_eq!(
            untiled.lines, off.lines,
            "{variant:?} untiled lines of\n{src}"
        );
        assert_eq!(untiled.narrowed, 0, "untiled runs narrow nothing");
        for t in tilings().into_iter().skip(1) {
            let got = run(src, names, variant, t, false);
            assert_eq!(got.outcome, want, "{variant:?} {t:?} outcome of\n{src}");
            assert_eq!(
                got.lines, off.lines,
                "{variant:?} {t:?} trace and stats of\n{src}"
            );
            assert_eq!(got.narrowed, 0, "{variant:?} {t:?} without selective");
            let selective = run(src, names, variant, t, true);
            assert_eq!(
                selective.outcome, want,
                "{variant:?} {t:?} selective outcome of\n{src}"
            );
            assert_eq!(
                calls_only(&selective.lines),
                calls_only(&off.lines),
                "{variant:?} {t:?} selective calls of\n{src}"
            );
            assert_eq!(
                selective.regions, got.regions,
                "{variant:?} {t:?} selective regions of\n{src}"
            );
            counts.push(got.regions);
            narrowed.push(selective.narrowed);
        }
    }
    (want, counts, narrowed)
}

/// `compare_narrowed` without the narrowed tile counts.
pub(super) fn compare(src: &str, names: &[&str]) -> (Outcome, Vec<u64>) {
    let (want, counts, _) = compare_narrowed(src, names);
    (want, counts)
}

/// Checks that `src` binds every name of `want` to its rows under every tiling, and tiles
/// a region in each tiled run.
pub(super) fn assert_tiles(src: &str, want: &[(&str, Vec<String>)]) {
    let names: Vec<&str> = want.iter().map(|(n, _)| *n).collect();
    let (outcome, counts) = compare(src, &names);
    let (_, bound) = outcome.unwrap();
    for ((name, rows), got) in want.iter().zip(bound) {
        assert_eq!(&got, rows, "{name} of\n{src}");
    }
    assert!(counts.iter().all(|&c| c > 0), "{counts:?} for\n{src}");
}

pub(super) fn text<T: ToString>(rows: impl IntoIterator<Item = T>) -> Vec<String> {
    rows.into_iter().map(|r| r.to_string()).collect()
}

/// A hand-built plan: its graph and its root.
pub(super) struct Built {
    graph: PlanGraph,
    root: PlanId,
}

/// Builds a plan into one graph node by node, interning as it goes.
#[derive(Default)]
pub(super) struct Builder(PlanGraph);

impl Builder {
    pub(super) fn leaf(&mut self, i: u32) -> PlanId {
        self.0.intern(PlanNode::Leaf(LeafId(i)))
    }

    /// A broadcast of leaf `i`.
    pub(super) fn bcast(&mut self, i: u32) -> PlanId {
        let leaf = self.leaf(i);
        self.0.intern(PlanNode::Bcast(leaf))
    }

    pub(super) fn kernel(&mut self, id: KernelId, args: Vec<PlanId>) -> PlanId {
        self.0.intern(PlanNode::Kernel(id, args))
    }

    pub(super) fn finish(self, root: PlanId) -> Built {
        Built {
            graph: self.0,
            root,
        }
    }
}

/// Evaluates `node` over `leaves` with `tiling` and selective evaluation on the instruction
/// set and backend of `variant`, with its shared nodes evaluated once, and returns the result
/// and the tiled region count.
pub(super) fn eval_node(
    reg: &Registry,
    node: &Built,
    leaves: &[Leaf],
    variant: (Isa, Backend),
    tiling: Tiling,
) -> (Slot, u64) {
    let (result, regions, _) = eval_selective(reg, node, leaves, variant, tiling, true);
    (result, regions)
}

/// `eval_node` with or without selective evaluation, also returning the narrowed tile count.
pub(super) fn eval_selective(
    reg: &Registry,
    node: &Built,
    leaves: &[Leaf],
    variant: (Isa, Backend),
    tiling: Tiling,
    selective: bool,
) -> (Slot, u64, u64) {
    let dispatch = reg.resolve(variant.0, variant.1, FeatureSet::EMPTY);
    let tiled = AtomicU64::new(0);
    let narrowed = AtomicU64::new(0);
    let hits = AtomicU64::new(0);
    let env = Env::new();
    let shared = shared_flags(reg, &node.graph, &[node.root]);
    let mut memo = vec![None; node.graph.len()];
    let result = Machine {
        reg,
        dispatch: &dispatch,
        graph: &node.graph,
        leaves,
        env: &env,
        span: Span { line: 3, col: 1 },
        trace: None,
        stats: None,
        tiling,
        selective,
        tiled_regions: &tiled,
        narrowed_tiles: &narrowed,
        shared: &shared,
        memo: &mut memo,
        shared_hits: &hits,
    }
    .eval(node.root);
    (
        result,
        tiled.load(Ordering::Relaxed),
        narrowed.load(Ordering::Relaxed),
    )
}

/// What a run with statistics gave: the outcome with the rows of the names asked for, the
/// calls per kernel, the shared hit count, and the tiled region count.
struct Counted {
    outcome: Outcome,
    calls: BTreeMap<String, u64>,
    hits: u64,
    regions: u64,
}

/// Runs the REPL inputs `srcs` in order on one scalar interpreter with statistics, with or
/// without sharing, and counts what they did. The outcome is that of the first failing
/// input, or of the last.
fn counted(srcs: &[&str], names: &[&str], no_sharing: bool, tiling: Tiling) -> Counted {
    let mut interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        stats: true,
        tiling,
        no_sharing,
        ..Options::default()
    })
    .unwrap();
    let mut out = Vec::new();
    let mut result = Ok(());
    for src in srcs {
        result = interp.eval_repl(src, &mut out, &mut std::io::sink());
        if result.is_err() {
            break;
        }
    }
    let outcome = result.map(|()| {
        let bound = names
            .iter()
            .map(|n| rows(&interp.get(n).unwrap_or_else(|| panic!("{n} is not bound"))))
            .collect();
        (String::from_utf8(out).unwrap(), bound)
    });
    let mut stats = Vec::new();
    interp.write_stats(&mut stats);
    let calls = String::from_utf8(stats)
        .unwrap()
        .lines()
        .map(|l| {
            let mut words = l.split(' ').skip(1);
            let name = words.next().unwrap().to_string();
            let calls = words.next().unwrap().strip_prefix("calls=").unwrap();
            (name, calls.parse().unwrap())
        })
        .collect();
    Counted {
        outcome,
        calls,
        hits: interp.shared_hits(),
        regions: interp.tiled_regions(),
    }
}

/// The calls of kernel `name` in `c`, 0 when it never ran.
fn calls(c: &Counted, name: &str) -> u64 {
    c.calls.get(name).copied().unwrap_or(0)
}

#[test]
fn sums_over_one_column_run_one_reduction() {
    // q29's shape: after the integer sum rewrite every plan holds `reduce_add_i64 x` and
    // `length_i64 x`, one shared node each across the five statements.
    let src = "let x = iota(1000);
print(sum(x + 1));
print(sum(x + 2));
print(sum(x + 3));
print(sum(x + 4));
print(sum(x + 5));";
    let shared = counted(&[src], &[], false, Tiling::default());
    let unshared = counted(&[src], &[], true, Tiling::default());
    assert_eq!(shared.outcome, unshared.outcome);
    let (printed, _) = shared.outcome.as_ref().unwrap();
    assert_eq!(printed, "500500\n501500\n502500\n503500\n504500\n");
    assert_eq!(calls(&shared, "reduce_add_i64"), 1);
    assert_eq!(calls(&unshared, "reduce_add_i64"), 5);
    assert_eq!(calls(&shared, "length_i64"), 1);
    assert_eq!(calls(&unshared, "length_i64"), 5);
    // Four reused reductions and four reused lengths.
    assert_eq!(shared.hits, 8);
    assert_eq!(unshared.hits, 0);
}

#[test]
fn statements_sharing_a_failing_subtree_report_their_own_spans() {
    // The expression statement fails and its error is dropped. The failure is not kept, so
    // the print evaluates the shared gather again and reports its own line.
    let src = |index: &str| {
        format!(
            "let x = iota(40);
sum(gather(x, {index}));
print(sum(gather(x, {index})) + 1);"
        )
    };
    let failing = src("x - 1");
    let run = |no_sharing| {
        let mut interp = Interpreter::new(Options {
            isa: Some(Isa::Scalar),
            stats: true,
            no_sharing,
            ..Options::default()
        })
        .unwrap();
        let e = interp
            .run(&failing, &mut std::io::sink(), &mut std::io::sink())
            .unwrap_err();
        (e, interp.shared_hits())
    };
    let (shared, hits) = run(false);
    let (unshared, _) = run(true);
    assert_eq!(shared.kind, ErrorKind::Runtime);
    assert_eq!(shared.span.map(|s| s.line), Some(3));
    assert_eq!(
        (&shared.kind, &shared.message, shared.span),
        (&unshared.kind, &unshared.message, unshared.span)
    );
    assert_eq!(hits, 0);
    // The same statements over an index that succeeds share the gather.
    let passing = counted(&[&src("x")], &[], false, Tiling::default());
    assert_eq!(calls(&passing, "gather_i64"), 1);
    assert_eq!(passing.hits, 1);
}

#[test]
fn a_shared_node_only_in_untaken_branches_never_runs() {
    let src = |n: i64| {
        format!(
            "let x = iota(10);
print(if length(x) > {n} {{ sum(x * x) }} else {{ 0 }});
print(if length(x) > {} {{ sum(x * x) }} else {{ 1 }});",
            n + 1
        )
    };
    let untaken = counted(&[&src(100)], &[], false, Tiling::default());
    assert_eq!(untaken.outcome.as_ref().unwrap().0, "0\n1\n");
    // The sum plans to `sumsq_i64 x`. The length in both conditions runs once.
    assert_eq!(calls(&untaken, "sumsq_i64"), 0);
    assert_eq!(calls(&untaken, "length_i64"), 1);
    assert_eq!(untaken.hits, 1);
    // Taken, the sum runs in the first statement and is reused in the second.
    let taken = counted(&[&src(0)], &[], false, Tiling::default());
    assert_eq!(taken.outcome.as_ref().unwrap().0, "285\n285\n");
    assert_eq!(calls(&taken, "sumsq_i64"), 1);
    assert_eq!(taken.hits, 2);
}

#[test]
fn a_shared_elementwise_kernel_feeding_two_regions_runs_once() {
    // `(x * 3) + 1` is one node read by two statements. It runs once, whole, and each
    // statement's region above it tiles with it as an input.
    let src = "let x = iota(100);
let r = (((x * 3) + 1) - x) * 2;
let q = (((x * 3) + 1) + x) * 5;";
    let r = text((0..100).map(|x| (2 * x + 1) * 2));
    let q = text((0..100).map(|x| (4 * x + 1) * 5));
    let (outcome, counts) = compare(src, &["r", "q"]);
    assert_eq!(outcome.unwrap().1, [r, q]);
    assert!(counts.iter().all(|&c| c == 2), "{counts:?}");
    let shared = counted(&[src], &["r", "q"], false, Tiling::default());
    let unshared = counted(&[src], &["r", "q"], true, Tiling::default());
    assert_eq!(shared.outcome, unshared.outcome);
    assert_eq!(shared.regions, 2);
    assert_eq!(shared.hits, 1);
    // The shared sum and its product once, and each region's product.
    assert_eq!(calls(&shared, "mul_i64"), 3);
    assert_eq!(calls(&shared, "add_i64"), 2);
    assert_eq!(calls(&unshared, "mul_i64"), 4);
    assert_eq!(calls(&unshared, "add_i64"), 3);
}

#[test]
fn a_shared_input_computed_by_an_earlier_statement_is_trivial() {
    // `(x * 7) + 2` runs in the first statement. In the second, whole evaluation calls
    // `x * 3` before it reaches the sum, which is in the memo by then, so the region tiles.
    let src = "let x = iota(100);
let a = (((x * 7) + 2) - x) * 2;
let b = ((x * 3) + ((x * 7) + 2)) * 5;";
    let a = text((0..100).map(|x| (6 * x + 2) * 2));
    let b = text((0..100).map(|x| (10 * x + 2) * 5));
    let (outcome, counts) = compare(src, &["a", "b"]);
    assert_eq!(outcome.unwrap().1, [a, b]);
    assert!(counts.iter().all(|&c| c == 2), "{counts:?}");
}

#[test]
fn repl_inputs_never_share() {
    let inputs = [
        "let x = iota(10);",
        "print(sum(x * x));",
        "print(sum(x * x));",
    ];
    let c = counted(&inputs, &[], false, Tiling::default());
    assert_eq!(c.outcome.as_ref().unwrap().0, "285\n285\n");
    // The sum plans to `sumsq_i64 x`.
    assert_eq!(calls(&c, "sumsq_i64"), 2);
    assert_eq!(c.hits, 0);
}

#[test]
fn a_shared_subgraph_with_a_load_runs_at_each_use() {
    let path = std::env::temp_dir().join(format!("verkehrt-{}-shared.i64", std::process::id()));
    let bytes: Vec<u8> = (1..=4i64).flat_map(i64::to_le_bytes).collect();
    std::fs::write(&path, bytes).unwrap();
    let src = format!(
        "print(sum(load_i64({p:?}) * 2));\nprint(sum(load_i64({p:?}) * 2));",
        p = path.to_str().unwrap()
    );
    let c = counted(&[&src], &[], false, Tiling::default());
    std::fs::remove_file(&path).unwrap();
    assert_eq!(c.outcome.as_ref().unwrap().0, "20\n20\n");
    assert_eq!(calls(&c, "load_i64"), 2);
    assert_eq!(calls(&c, "reduce_add_i64"), 2);
    assert_eq!(c.hits, 0);
}

#[test]
fn prepared_execute_counts_shared_hits() {
    let interp = Interpreter::new(Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    })
    .unwrap();
    let prepared = interp
        .prepare("input x: i64[]; export a = sum(x + 1); export b = sum(x + 2);")
        .unwrap();
    let x = Value::Array(Array::from_vec((0..10i64).collect()));
    for run in 1..=2 {
        let out = prepared
            .execute(&interp, &[("x", x.clone())], &mut std::io::sink())
            .unwrap();
        assert_eq!(out[0].1.to_string(), "55");
        assert_eq!(out[1].1.to_string(), "65");
        // Each run starts with an empty memo and reuses the reduction and the length once.
        assert_eq!(interp.shared_hits(), 2 * run);
    }
}
