//! Golden program tests and the library API contract. Nothing here depends on trace timings.

use std::fs;

use verkehrt::error::ErrorKind;
use verkehrt::kernels::{Feature, FeatureSet, Isa};
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

#[test]
fn golden_programs() {
    let mut count = 0;
    for entry in fs::read_dir("tests/programs").expect("programs dir") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "vk") {
            continue;
        }
        let src = fs::read_to_string(&path).unwrap();
        let want = fs::read_to_string(path.with_extension("out")).unwrap();
        let (r, out, _) = run_with(Options::default(), &src);
        r.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(out, want, "{}", path.display());
        count += 1;
    }
    assert!(count >= 5, "found only {count} programs");
}

#[test]
fn group_id_length_mismatch_is_a_runtime_error() {
    let src = "let a = [1, 2, 3];\nlet b = [1, 2];\nprint(group_id(a, b));\n";
    let (r, out, _) = run_with(Options::default(), src);
    let e = r.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert!(e.message.contains("length mismatch: 3 vs 2"), "{e}");
    assert_eq!(e.span.unwrap().line, 3);
    assert_eq!(out, "");
}

#[test]
fn runtime_error_carries_statement_line() {
    let src = "let x = [1.0, 2.0];\nlet y = [1.0, 2.0, 3.0];\nprint(sum(x + y));\n";
    let (r, out, _) = run_with(Options::default(), src);
    let e = r.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert_eq!(e.span.unwrap().line, 3);
    assert_eq!(out, "");
}

#[test]
fn type_errors_come_before_any_output() {
    let src = "print(1.0);\nlet x = [1.0];\nlet y = [1];\nprint(x + y);\n";
    let (r, out, _) = run_with(Options::default(), src);
    let e = r.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Type);
    assert_eq!(e.span.unwrap().line, 4);
    assert_eq!(out, "");
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

fn run_ok(src: &str) -> String {
    let (r, out, _) = run_with(Options::default(), src);
    r.unwrap_or_else(|e| panic!("{src}: {e}"));
    out
}

fn run_err(src: &str) -> verkehrt::error::Error {
    let (r, out, _) = run_with(Options::default(), src);
    assert_eq!(out, "", "{src}");
    r.unwrap_err()
}

#[test]
fn if_with_mask_condition_is_an_array() {
    let src = "let m = [true, false, true];\nlet y = if m { 1.0 } else { 2.0 };\nprint(y);\nprint(y + 1.0);\n";
    assert_eq!(run_ok(src), "[1.0, 2.0, 1.0]\n[2.0, 3.0, 2.0]\n");
    let masked_src =
        "let m = [true, false, true];\nlet y: f64 = if m { 1.0 } else { 2.0 };\nprint(sum(y));\n";
    let e = run_err(masked_src);
    assert_eq!(e.kind, ErrorKind::Type, "{e}");
    assert_eq!(e.span.map(|s| (s.line, s.col)), Some((2, 1)), "{e}");
}

#[test]
fn if_with_mask_condition_in_function_and_export() {
    let src = "fn f(m: bool[]) -> f64[] { if m { 1.0 } else { 2.0 } }\n\
               let m = [true, false];\nprint(f(m) + 1.0);\n";
    assert_eq!(run_ok(src), "[2.0, 3.0]\n");
    let interp = Interpreter::new(Options::default()).unwrap();
    let p = interp
        .prepare("input m: bool[]; export y = if m { 1.0 } else { 2.0 };")
        .unwrap();
    let mask = Value::Array(Array::from_bools(&[false, true]));
    let got = p.execute(&interp, &[("m", mask)], &mut Vec::new()).unwrap();
    assert_eq!(got[0].1, f64s(&[2.0, 1.0]));
}

#[test]
fn zip_map_checks_lengths_whatever_the_body() {
    for (body, want) in [
        ("|a, b| a", "[1.0, 2.0, 3.0]"),
        ("|a, b| b", "[4.0, 5.0, 6.0]"),
        ("|a, b| 1.0", "[1.0, 1.0, 1.0]"),
        ("|a, b| a + b", "[5.0, 7.0, 9.0]"),
    ] {
        let ok = format!(
            "let x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nprint(zip_map(x, y, {body}));\n"
        );
        assert_eq!(run_ok(&ok), format!("{want}\n"), "{ok}");
        let bad =
            format!("let x = [1.0, 2.0, 3.0];\nlet y = [4.0];\nprint(zip_map(x, y, {body}));\n");
        let e = run_err(&bad);
        assert_eq!(e.kind, ErrorKind::Runtime, "{bad}: {e}");
        assert!(e.message.contains("length mismatch"), "{bad}: {e}");
        assert_eq!(e.span.unwrap().line, 3, "{bad}: {e}");
    }
    // The length checks do not prevent fusion.
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
fn huge_lengths_are_runtime_errors() {
    for src in [
        "print(length(iota(9223372036854775807)));",
        "print(length(fill(4611686018427387904, 1.0)));",
        "print(length(rand_i64(2305843009213693952, 1)));",
        "print(length(rand_f64(9223372036854775807, 1)));",
    ] {
        let e = run_err(src);
        assert_eq!(e.kind, ErrorKind::Runtime, "{src}: {e}");
        assert!(e.message.contains("is too large"), "{src}: {e}");
    }
}

#[test]
fn scalar_condition_if_evaluates_only_the_taken_branch() {
    for (src, want) in [
        (
            "let e = fill(0, 0.0);\nprint(if length(e) > 0 { reduce(e, max) } else { 0.0 });\n",
            "0.0\n",
        ),
        ("print(if true { [1.0] } else { [1.0, 2.0] });\n", "[1.0]\n"),
        (
            "print(if false { [1.0] } else { [1.0, 2.0] });\n",
            "[1.0, 2.0]\n",
        ),
        // A subterm repeated inside a branch stays inside it.
        (
            "let e = fill(0, 0.0);\nprint(if length(e) > 0 { reduce(e, max) + reduce(e, max) } else { 0.0 });\n",
            "0.0\n",
        ),
        // A conditional nested in an operation and in a function body.
        (
            "fn safe_max(e: f64[]) -> f64 { if length(e) > 0 { reduce(e, max) } else { 0.0 } }\n\
             let x = [1.0, 2.0];\nprint(safe_max(fill(0, 1.0)) + safe_max(x));\nprint(x * safe_max(x));\n",
            "2.0\n[2.0, 4.0]\n",
        ),
        (
            "let x = [1.0, 2.0];\nprint(sum(x * if length(x) > 1 { x } else { [0.0] }));\n",
            "5.0\n",
        ),
    ] {
        assert_eq!(run_ok(src), want, "{src}");
    }
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

/// Runs `src` and returns its output.
fn output_of(src: &str) -> String {
    let (r, out, _) = run_with(Options::default(), src);
    r.unwrap_or_else(|e| panic!("{src}: {e}"));
    out
}

#[test]
fn bitwise_on_integers() {
    let src = "print(and(12, 10));\nprint(or(12, 10));\nprint(xor(12, 10));\n\
               print(not(0u8));\nprint(not(0));\n\
               let a = [12, 5];\nlet b = [10, 3];\n\
               print(and(a, b));\nprint(or(a, b));\nprint(xor(a, b));\nprint(not(a));\n\
               print(not([0u8, 255u8]));\n\
               print(and([12u64, 5u64], 10u64));\nprint(xor([12i32], [10i32]));\n";
    assert_eq!(
        output_of(src),
        "8\n14\n6\n255\n-1\n[8, 1]\n[14, 7]\n[6, 6]\n[-13, -6]\n[255, 0]\n[8, 0]\n[6]\n"
    );
}

#[test]
fn xor_on_bool() {
    let src = "export r = xor([true, true, false], [true, false, false]);\nprint(r);\n";
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run(src, &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "[false, true, false]\n");
    let Some(Value::Array(a)) = interp.get("r") else {
        panic!("expected an array");
    };
    assert_eq!(a.as_bool_bytes(), &[0, 1, 0]);
}

#[test]
fn reduce_bitwise_on_integers() {
    let src = "let x = [12, 10, 6];\n\
               print(reduce(x, and));\nprint(reduce(x, or));\nprint(reduce(x, xor));\n\
               let e = filter(x, |v| v > 100);\n\
               print(reduce(e, and));\nprint(reduce(e, or));\nprint(reduce(e, xor));\n\
               let u = filter([1u8], |v| v > 100u8);\n\
               print(reduce(u, and));\n";
    assert_eq!(output_of(src), "0\n14\n0\n-1\n0\n0\n255\n");
}
