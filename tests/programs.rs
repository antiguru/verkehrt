//! Golden programs in `tests/programs/*.dd` and the library API contract. Nothing here depends on trace timings.

use verkehrt::error::ErrorKind;
use verkehrt::kernels::{Backend, Feature, FeatureSet, Isa, Variant};
use verkehrt::value::{Array, Value};
use verkehrt::{Interpreter, Options, Prepared};

fn run_with(opts: Options, src: &str) -> (Result<(), verkehrt::error::Error>, String, String) {
    let mut interp = Interpreter::new(opts).expect("interpreter");
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    let r = interp.run(src, &mut out, &mut diag);
    (
        r,
        String::from_utf8(out).unwrap(),
        String::from_utf8(diag).unwrap(),
    )
}

fn scalar_opts() -> Options {
    Options {
        isa: Some(Isa::Scalar),
        ..Options::default()
    }
}

fn f64s(v: &[f64]) -> Value {
    Value::Array(Array::from_vec(v.to_vec()))
}

/// Every instruction set the CPU supports, each with both backends. Scalar has one variant,
/// and comes first so that it is the reference.
fn variants() -> Vec<Variant> {
    Isa::ALL
        .into_iter()
        .filter(|isa| isa.is_available())
        .flat_map(|isa| {
            let backends: &[Backend] = if isa == Isa::Scalar {
                &[Backend::Intrinsics]
            } else {
                &[Backend::Auto, Backend::Intrinsics]
            };
            backends
                .iter()
                .map(move |&backend| Variant { isa, backend })
        })
        .collect()
}

/// What a program did: its output, then the error that stopped it, if any, as
/// `<kind> error at <line>:<col>: <message>`.
fn outcome(variant: Variant, src: &str) -> (String, Option<String>) {
    let opts = Options {
        isa: Some(variant.isa),
        backend: variant.backend,
        ..Options::default()
    };
    let (r, out, _) = run_with(opts, src);
    let err = r.err().map(|e| match e.span {
        Some(s) => format!("{:?} error at {}:{}: {}", e.kind, s.line, s.col, e.message),
        None => format!("{:?} error: {}", e.kind, e.message),
    });
    (out, err)
}

/// Runs `src` on every variant and returns the outcome that all of them share. Panics, naming
/// the variants, if one differs from the scalar run.
fn run_everywhere(src: &str) -> (String, Option<String>) {
    let variants = variants();
    let want = outcome(variants[0], src);
    let differing: Vec<String> = variants[1..]
        .iter()
        .filter(|v| outcome(**v, src) != want)
        .map(|v| format!("[{v}]"))
        .collect();
    assert!(
        differing.is_empty(),
        "{} differ from [{}] on:\n{src}\nscalar: {want:?}",
        differing.join(" "),
        variants[0],
    );
    want
}

/// Runs the `run` and `error` directives on every file in `tests/programs`. The input is a
/// program. `run` prints its output and requires that it succeeds. `error` prints the output
/// written before the error, if any, then the error, and requires that the program fails.
#[test]
fn datadriven_programs() {
    datadriven::walk(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/programs"),
        |f| {
            f.run(|tc| {
                let (out, err) = run_everywhere(&tc.input);
                match (tc.directive.as_str(), err) {
                    ("run", None) => out,
                    ("run", Some(e)) => panic!("{e}\nafter output:\n{out}"),
                    ("error", Some(e)) if out.is_empty() => format!("{e}\n"),
                    ("error", Some(e)) => format!("{out}{e}\n"),
                    ("error", None) => panic!("the program succeeded with output:\n{out}"),
                    (d, _) => panic!("unknown directive {d}"),
                }
            })
        },
    );
}

#[test]
fn show_plan_prints_fused_kernel() {
    let src = "let x = [1.0, 2.0];\nlet y = [3.0, 4.0];\nprint(sum(x * y));\n";
    let (r, out, diag) = run_with(
        Options {
            show_plan: true,
            ..scalar_opts()
        },
        src,
    );
    r.unwrap();
    assert_eq!(out, "11.0\n");
    let line = diag.lines().find(|l| l.contains("dot_f64")).expect(&diag);
    assert!(line.starts_with("plan 3: "), "{line}");
}

#[test]
fn trace_reports_isa_and_bytes() {
    let src = "let x = [1.0, 2.0];\nlet y = [3.0, 4.0];\nprint(sum(x * y));\n";
    let (r, _, diag) = run_with(
        Options {
            trace: true,
            ..scalar_opts()
        },
        src,
    );
    r.unwrap();
    let line = diag.lines().find(|l| l.starts_with("trace ")).expect(&diag);
    assert!(line.contains("isa=scalar"), "{line}");
    assert!(line.contains("n=2 read=32 write=0 time="), "{line}");
    assert!(line.contains("gbps="), "{line}");
}

#[test]
fn stats_total_per_kernel() {
    let mut interp = Interpreter::new(Options {
        stats: true,
        ..scalar_opts()
    })
    .unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp
        .run(
            "let x = [1.0];\nprint(sum(x));\nprint(sum(x));\n",
            &mut out,
            &mut diag,
        )
        .unwrap();
    let mut stats = Vec::new();
    interp.write_stats(&mut stats);
    let stats = String::from_utf8(stats).unwrap();
    assert!(stats.contains("calls=2"), "{stats}");
}

// An ISA of another architecture is never available on the host.
#[cfg(target_arch = "aarch64")]
const FOREIGN_ISA: Isa = Isa::X86_64V3;
#[cfg(not(target_arch = "aarch64"))]
const FOREIGN_ISA: Isa = Isa::Neon;

#[test]
fn unavailable_isa_is_a_usage_error() {
    let e = Interpreter::new(Options {
        isa: Some(FOREIGN_ISA),
        ..Options::default()
    })
    .err()
    .expect("an ISA of another architecture is unavailable");
    assert_eq!(e.kind, ErrorKind::Usage);
}

#[test]
fn disable_vbmi2_keeps_v4() {
    if !Isa::X86_64V4.is_available() {
        eprintln!("skipped: x86-64-v4 is not available on this CPU");
        return;
    }
    let src = "let a = [1u8, 2u8, 3u8, 4u8];\nprint(compress(a, a > 2u8));\n";
    let (scalar_result, expected, _) = run_with(scalar_opts(), src);
    scalar_result.unwrap();
    let (r, out, diag) = run_with(
        Options {
            disable: FeatureSet::of(&[Feature::Avx512vbmi2]),
            trace: true,
            ..Options::default()
        },
        src,
    );
    r.unwrap();
    assert_eq!(out, expected);
    let line = diag
        .lines()
        .find(|l| l.starts_with("trace compress_u8"))
        .expect(&diag);
    assert!(line.contains("isa=x86-64-v4 backend=auto"), "{line}");
}

#[test]
fn repl_expression_prints_its_value() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp
        .eval_repl("let x = [1.0];", &mut out, &mut diag)
        .unwrap();
    assert!(out.is_empty());
    interp.eval_repl("sum(x);", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "1.0\n");
}

#[test]
fn bound_arrays_are_not_copied() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let Value::Array(a) = f64s(&[1.0, 2.0, 3.0]) else {
        unreachable!()
    };
    let ptr = a.as_slice::<f64>().as_ptr();
    interp.bind("x", Value::Array(a));
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run("print(sum(x));", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "6.0\n");
    let Some(Value::Array(got)) = interp.get("x") else {
        panic!("x is bound")
    };
    assert_eq!(got.as_slice::<f64>().as_ptr(), ptr);
    assert!(interp.get("nope").is_none());
}

#[test]
fn input_must_be_bound_in_run() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    let e = interp
        .run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag)
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    assert_eq!(e.message, "input 'x' is not bound");
    interp.bind("x", Value::Array(Array::from_vec(vec![1i64])));
    let second_err = interp
        .run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag)
        .unwrap_err();
    assert_eq!(second_err.kind, ErrorKind::Usage);
    interp.bind("x", f64s(&[1.0, 2.0]));
    interp
        .run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag)
        .unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "3.0\n");
}

#[test]
fn exports_behave_like_let_in_run() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp
        .run("export s = sum([1.0, 2.0]);", &mut out, &mut diag)
        .unwrap();
    assert_eq!(interp.get("s").unwrap().to_string(), "3.0");
}

#[test]
fn handles_are_thread_safe() {
    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send::<Interpreter>();
    assert_send_sync::<Prepared>();
}

#[test]
fn prepare_once_execute_many() {
    let interp = Interpreter::new(Options {
        show_plan: true,
        ..scalar_opts()
    })
    .unwrap();
    let mut plan = Vec::new();
    let src = "input x: f64[]; input y: f64[]; export d = sum(x * y);";
    let prepared = interp.prepare_to(src, &mut plan).unwrap();
    let plan = String::from_utf8(plan).unwrap();
    assert_eq!(plan.matches("dot_f64").count(), 1, "{plan}");
    for (x, y, want) in [
        ([1.0, 2.0], [3.0, 4.0], 11.0),
        ([1.0, 1.0], [2.0, 2.0], 4.0),
    ] {
        let mut diag = Vec::new();
        let got = prepared
            .execute(&interp, &[("x", f64s(&x)), ("y", f64s(&y))], &mut diag)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "d");
        assert_eq!(got[0].1.to_string(), format!("{want:?}"));
        assert!(diag.is_empty(), "plan is written at prepare only");
    }
}

#[test]
fn execute_checks_inputs() {
    let interp = Interpreter::new(Options::default()).unwrap();
    let p = interp
        .prepare("input x: f64[]; export s = sum(x);")
        .unwrap();
    let mut diag = Vec::new();
    let e = p.execute(&interp, &[], &mut diag).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    let ints = Value::Array(Array::from_vec(vec![1i64]));
    let second_err = p.execute(&interp, &[("x", ints)], &mut diag).unwrap_err();
    assert_eq!(second_err.kind, ErrorKind::Usage);
}

#[test]
fn prepare_rejects_undeclared_names_and_returns_exports_in_order() {
    let interp = Interpreter::new(Options::default()).unwrap();
    assert_eq!(
        interp.prepare("export s = sum(x);").err().unwrap().kind,
        ErrorKind::Type
    );
    let p = interp
        .prepare("input x: f64[]; export b = sum(x); export a = count(x > 0.0); print(b);")
        .unwrap();
    let mut diag = Vec::new();
    let got = p
        .execute(&interp, &[("x", f64s(&[1.0, -1.0]))], &mut diag)
        .unwrap();
    let names: Vec<_> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["b", "a"]);
    assert_eq!(String::from_utf8(diag).unwrap(), "0.0\n");
}

#[test]
fn execute_rejects_undeclared_and_duplicate_inputs() {
    let interp = Interpreter::new(Options::default()).unwrap();
    let p = interp
        .prepare("input x: f64[]; export s = sum(x);")
        .unwrap();
    let mut diag = Vec::new();
    let e = p
        .execute(
            &interp,
            &[("x", f64s(&[1.0])), ("n", f64s(&[1.0]))],
            &mut diag,
        )
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    assert_eq!(e.message, "program does not declare input 'n'");
    let second_err = p
        .execute(
            &interp,
            &[("x", f64s(&[1.0])), ("x", f64s(&[2.0]))],
            &mut diag,
        )
        .unwrap_err();
    assert_eq!(second_err.kind, ErrorKind::Usage);
}

#[test]
fn failed_repl_input_keeps_old_bindings() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp
        .eval_repl("let x = [1.0, 2.0];", &mut out, &mut diag)
        .unwrap();
    let e = interp
        .eval_repl(
            "let x = [1.0] + [1.0, 2.0]; let z = [3.0];",
            &mut out,
            &mut diag,
        )
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert_eq!(interp.get("x").unwrap().to_string(), "[1.0, 2.0]");
    assert!(interp.get("z").is_none());
    interp.eval_repl("x;", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "[1.0, 2.0]\n");
    // Variable ids of the failed input are reused, which must not resurrect stale values.
    interp
        .eval_repl("let z = [9.0];", &mut Vec::new(), &mut diag)
        .unwrap();
    assert_eq!(interp.get("z").unwrap().to_string(), "[9.0]");
}

#[test]
fn if_with_mask_condition_in_export() {
    let interp = Interpreter::new(Options::default()).unwrap();
    let p = interp
        .prepare("input m: bool[]; export y = if m { 1.0 } else { 2.0 };")
        .unwrap();
    let mask = Value::Array(Array::from_bools(&[false, true]));
    let got = p.execute(&interp, &[("m", mask)], &mut Vec::new()).unwrap();
    assert_eq!(got[0].1, f64s(&[2.0, 1.0]));
}

#[test]
fn zip_map_length_checks_do_not_prevent_fusion() {
    let src =
        "let x = [1.0, 2.0];\nlet y = [3.0, 4.0];\nprint(sum(zip_map(x, y, |a, b| a * b)));\n";
    let (r, out, diag) = run_with(
        Options {
            show_plan: true,
            ..scalar_opts()
        },
        src,
    );
    r.unwrap();
    assert_eq!(out, "11.0\n");
    assert!(diag.contains("dot_f64"), "{diag}");
}

#[test]
fn scalar_condition_if_plans_only_the_taken_branch() {
    let src = "let e = fill(0, 0.0);\nprint(if length(e) > 0 { reduce(e, max) } else { 0.0 });\n";
    let (r, _, diag) = run_with(
        Options {
            show_plan: true,
            ..scalar_opts()
        },
        src,
    );
    r.unwrap();
    let line = diag
        .lines()
        .find(|l| l.starts_with("plan 2: "))
        .expect(&diag);
    // The comparison may come out as `gt` or as the swapped `lt`, both cost the same.
    assert!(line.starts_with("plan 2: (if ("), "{line}");
    assert!(
        line.ends_with(" then (reduce_max_f64 e) else 0.0)"),
        "{line}"
    );
}

/// The printed output is in `programs/arith.dd`. The test checks the bytes of the exported array.
#[test]
fn xor_on_bool_exports_bool_bytes() {
    let src = "export r = xor([true, true, false], [true, false, false]);\n";
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run(src, &mut out, &mut diag).unwrap();
    let Some(Value::Array(a)) = interp.get("r") else {
        panic!("expected an array");
    };
    assert_eq!(a.as_bool_bytes(), &[0, 1, 0]);
}
