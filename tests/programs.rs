//! Golden program tests and the library API contract. Nothing here depends on trace timings.

use std::fs;

use verkehrt::error::ErrorKind;
use verkehrt::kernels::Isa;
use verkehrt::value::{Array, Value};
use verkehrt::{Interpreter, Options, Prepared};

fn run_with(opts: Options, src: &str) -> (Result<(), verkehrt::error::Error>, String, String) {
    let mut interp = Interpreter::new(opts).expect("interpreter");
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    let r = interp.run(src, &mut out, &mut diag);
    (r, String::from_utf8(out).unwrap(), String::from_utf8(diag).unwrap())
}

fn scalar_opts() -> Options {
    Options { isa: Some(Isa::Scalar), ..Options::default() }
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
    let (r, out, diag) = run_with(Options { show_plan: true, ..scalar_opts() }, src);
    r.unwrap();
    assert_eq!(out, "11.0\n");
    let line = diag.lines().find(|l| l.contains("dot_f64")).expect(&diag);
    assert!(line.starts_with("plan 3: "), "{line}");
}

#[test]
fn trace_reports_isa_and_bytes() {
    let src = "let x = [1.0, 2.0];\nlet y = [3.0, 4.0];\nprint(sum(x * y));\n";
    let (r, _, diag) = run_with(Options { trace: true, ..scalar_opts() }, src);
    r.unwrap();
    let line = diag.lines().find(|l| l.starts_with("trace ")).expect(&diag);
    assert!(line.contains("isa=scalar"), "{line}");
    assert!(line.contains("n=2 read=32 write=0 time="), "{line}");
    assert!(line.contains("gbps="), "{line}");
}

#[test]
fn stats_total_per_kernel() {
    let mut interp = Interpreter::new(Options { stats: true, ..scalar_opts() }).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run("let x = [1.0];\nprint(sum(x));\nprint(sum(x));\n", &mut out, &mut diag).unwrap();
    let mut stats = Vec::new();
    interp.write_stats(&mut stats);
    let stats = String::from_utf8(stats).unwrap();
    assert!(stats.contains("calls=2"), "{stats}");
}

#[test]
fn unavailable_isa_is_a_usage_error() {
    let e = Interpreter::new(Options { isa: Some(Isa::Neon), ..Options::default() })
        .err()
        .expect("neon is unavailable on x86");
    assert_eq!(e.kind, ErrorKind::Usage);
}

#[test]
fn repl_expression_prints_its_value() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.eval_repl("let x = [1.0];", &mut out, &mut diag).unwrap();
    assert!(out.is_empty());
    interp.eval_repl("sum(x);", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "1.0\n");
}

#[test]
fn bound_arrays_are_not_copied() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let Value::Array(a) = f64s(&[1.0, 2.0, 3.0]) else { unreachable!() };
    let ptr = a.as_slice::<f64>().as_ptr();
    interp.bind("x", Value::Array(a));
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run("print(sum(x));", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "6.0\n");
    let Some(Value::Array(got)) = interp.get("x") else { panic!("x is bound") };
    assert_eq!(got.as_slice::<f64>().as_ptr(), ptr);
    assert!(interp.get("nope").is_none());
}

#[test]
fn input_must_be_bound_in_run() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    let e = interp.run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    assert_eq!(e.message, "input 'x' is not bound");
    interp.bind("x", Value::Array(Array::from_vec(vec![1i64])));
    let e = interp.run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    interp.bind("x", f64s(&[1.0, 2.0]));
    interp.run("input x: f64[];\nprint(sum(x));", &mut out, &mut diag).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "3.0\n");
}

#[test]
fn exports_behave_like_let_in_run() {
    let mut interp = Interpreter::new(Options::default()).unwrap();
    let (mut out, mut diag) = (Vec::new(), Vec::new());
    interp.run("export s = sum([1.0, 2.0]);", &mut out, &mut diag).unwrap();
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
    let interp = Interpreter::new(Options { show_plan: true, ..scalar_opts() }).unwrap();
    let mut plan = Vec::new();
    let src = "input x: f64[]; input y: f64[]; export d = sum(x * y);";
    let prepared = interp.prepare_to(src, &mut plan).unwrap();
    let plan = String::from_utf8(plan).unwrap();
    assert_eq!(plan.matches("dot_f64").count(), 1, "{plan}");
    for (x, y, want) in [([1.0, 2.0], [3.0, 4.0], 11.0), ([1.0, 1.0], [2.0, 2.0], 4.0)] {
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
    let p = interp.prepare("input x: f64[]; export s = sum(x);").unwrap();
    let mut diag = Vec::new();
    let e = p.execute(&interp, &[], &mut diag).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
    let ints = Value::Array(Array::from_vec(vec![1i64]));
    let e = p.execute(&interp, &[("x", ints)], &mut diag).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Usage);
}

#[test]
fn prepare_rejects_undeclared_names_and_returns_exports_in_order() {
    let interp = Interpreter::new(Options::default()).unwrap();
    assert_eq!(interp.prepare("export s = sum(x);").err().unwrap().kind, ErrorKind::Type);
    let p = interp
        .prepare("input x: f64[]; export b = sum(x); export a = count(x > 0.0); print(b);")
        .unwrap();
    let mut diag = Vec::new();
    let got = p.execute(&interp, &[("x", f64s(&[1.0, -1.0]))], &mut diag).unwrap();
    let names: Vec<_> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["b", "a"]);
    assert_eq!(String::from_utf8(diag).unwrap(), "0.0\n");
}
