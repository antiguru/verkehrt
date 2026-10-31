use super::*;
use crate::error::{Error, ErrorKind, Span};
use crate::syntax::parse;
use crate::value::{ScalarType, ValueType};

const F64: ValueType = ValueType::Scalar(ScalarType::F64);
const F64A: ValueType = ValueType::Array(ScalarType::F64);

fn run(src: &str) -> Result<(TProgram, Globals), Error> {
    let prog = parse(src).unwrap();
    let reg = Registry::new();
    let mut g = Globals::new();
    check(&prog, &reg, &mut g).map(|p| (p, g))
}

/// The type of the last `let` or `export` in the program.
fn last_type(src: &str) -> ValueType {
    let (p, _) = run(src).unwrap_or_else(|e| panic!("{src}: {e}"));
    match p.stmts.last().unwrap() {
        TStmt::Let { value, .. } | TStmt::Export { value, .. } => value.ty,
        other => panic!("unexpected {other:?}"),
    }
}

fn type_err(src: &str, needle: &str) -> Error {
    let e = run(src).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Type, "{e}");
    assert!(
        e.message.contains(needle),
        "{:?} lacks {needle:?}",
        e.message
    );
    e
}

#[test]
fn elementwise_array_scalar() {
    assert_eq!(last_type("let x = [1.0, 2.0]; let y = x * 2.0;"), F64A);
}

#[test]
fn no_implicit_conversion_at_operator() {
    let e = type_err("let x = [1.0]; let y = x * 2;", "no implicit conversion");
    assert_eq!(e.span, Some(Span { line: 1, col: 26 }));
    assert!(e.message.contains("f64[] * i64"), "{}", e.message);
}

#[test]
fn no_kernel() {
    type_err("let x = [1i8]; let y = x * x;", "no kernel for '*' on i8[]");
}

#[test]
fn and_on_f64_has_no_kernel() {
    type_err(
        "let x = [1.0]; let y = and(x, x);",
        "no kernel for 'and' on f64[]",
    );
}

#[test]
fn scalar_kernels_for_every_type() {
    assert_eq!(
        last_type("let a = 2u64; let b = a + a;"),
        ValueType::Scalar(ScalarType::U64)
    );
}

#[test]
fn reduction_widens() {
    assert_eq!(
        last_type("let x = [1u8, 2u8]; let s = sum(x);"),
        ValueType::Scalar(ScalarType::U64)
    );
}

#[test]
fn map_lambda() {
    assert_eq!(
        last_type("let x = [1.0]; let m = map(x, |v| (v * 2.0) + 1.0);"),
        F64A
    );
}

#[test]
fn lambda_rejects_reduction() {
    type_err(
        "let x = [1.0]; let y = [2.0]; let m = map(x, |v| v * sum(y));",
        "'sum' cannot be used inside a lambda",
    );
}

#[test]
fn prev_is_rejected_inside_a_lambda() {
    type_err(
        "let x = [1, 2]; let m = map(x, |v| prev(v, 0));",
        "'prev' cannot be used inside a lambda",
    );
}

#[test]
fn lambda_rejects_array_capture() {
    type_err(
        "let x = [1.0]; let y = [2.0]; let m = map(x, |v| v * y);",
        "captures array 'y'",
    );
}

#[test]
fn lambda_captures_scalar() {
    assert_eq!(
        last_type("let t = 0.5; let f = filter([1.0], |v| v > t);"),
        F64A
    );
}

#[test]
fn filter_needs_bool() {
    type_err("let f = filter([1.0], |v| v * 2.0);", "bool");
}

#[test]
fn direct_recursion() {
    type_err("fn f(a: f64) -> f64 { f(a) }", "recursive function 'f'");
}

#[test]
fn mutual_recursion() {
    type_err(
        "fn f(a: f64) -> f64 { g(a) } fn g(a: f64) -> f64 { f(a) }",
        "recursive function",
    );
}

#[test]
fn forward_reference_without_cycle() {
    assert_eq!(
        last_type("fn f(a: f64) -> f64 { g(a) } fn g(a: f64) -> f64 { a } let r = f(1.0);"),
        F64
    );
}

#[test]
fn map_with_function_name() {
    assert_eq!(
        last_type("fn sq(v: f64) -> f64 { v * v } let m = map([1.0], sq);"),
        F64A
    );
}

#[test]
fn lambda_calls_non_liftable_function() {
    type_err(
        "fn s(a: f64) -> f64 { sum([a]) } let m = map([1.0], |v| s(v));",
        "'s' cannot be used inside a lambda",
    );
}

#[test]
fn reduce_op_names() {
    assert_eq!(last_type("let m = reduce([1.0], max);"), F64);
    type_err("let m = reduce([1.0], foo);", "foo");
}

#[test]
fn unknown_name_span() {
    let e = type_err("let a = 1.0;\nlet y = a + z;", "unknown name 'z'");
    assert_eq!(e.span, Some(Span { line: 2, col: 13 }));
}

#[test]
fn input_and_export() {
    assert_eq!(last_type("input x: f64[]; export s = sum(x);"), F64);
}

#[test]
fn input_no_implicit_conversion() {
    type_err("input x: f64[]; let y = x * 2;", "no implicit conversion");
}

#[test]
fn export_name_visible_later() {
    assert_eq!(last_type("export a = 1.0; let b = a + a;"), F64);
}

#[test]
fn let_annotation_must_match() {
    assert_eq!(last_type("let x: f64[] = [1.0];"), F64A);
    type_err("let x: f64 = [1.0];", "f64");
}

#[test]
fn if_rules() {
    assert_eq!(last_type("let r = if 1.0 < 2.0 { 1.0 } else { 2.0 };"), F64);
    type_err("let r = if 1.0 { 1.0 } else { 2.0 };", "bool");
    type_err("let r = if true { 1.0 } else { 2 };", "branch");
    assert_eq!(
        last_type("let m = [true]; let r = if m { 1.0 } else { 2.0 };"),
        F64A
    );
    type_err(
        "let m = [true]; let r = if m { [1i8] } else { [2i8] };",
        "no kernel for 'if'",
    );
}

#[test]
fn array_literal_rules() {
    type_err("let a = [1.0, 2];", "same");
}

#[test]
fn function_scope_and_shadowing() {
    type_err(
        "let t = 1.0; fn f(a: f64) -> f64 { a + t }",
        "unknown name 't'",
    );
    type_err("fn sum(a: f64) -> f64 { a }", "builtin");
    let (p, g) = run("let a = 1.0; let a = [2.0];").unwrap();
    let vars: Vec<VarId> = p
        .stmts
        .iter()
        .map(|s| match s {
            TStmt::Let { var, .. } => *var,
            _ => unreachable!(),
        })
        .collect();
    assert_ne!(vars[0], vars[1]);
    assert_eq!(g.name(vars[0]), "a");
}

#[test]
fn load_and_print_top_level_only() {
    assert_eq!(last_type("let a = load_f64(\"x.bin\");"), F64A);
    type_err("fn f(a: f64) -> f64 { sum(load_f64(\"x\")) }", "top level");
    type_err(
        "let a = map([1.0], |v| sum(load_f64(\"x\")));",
        "inside a lambda",
    );
}

#[test]
fn globals_persist_and_declare() {
    let reg = Registry::new();
    let mut g = Globals::new();
    let x = g.declare("x", F64A);
    check(&parse("let y = x;").unwrap(), &reg, &mut g).unwrap();
    let p = check(&parse("let z = y;").unwrap(), &reg, &mut g).unwrap();
    assert!(matches!(&p.stmts[0], TStmt::Let { value, .. } if value.ty == F64A));
    assert_eq!(g.name(x), "x");
    // A failed check leaves the globals untouched.
    assert!(check(&parse("let w = nope;").unwrap(), &reg, &mut g).is_err());
    assert!(g.lookup("w").is_none());
}

#[test]
fn builtins_result_types() {
    let i = ValueType::Scalar(ScalarType::I64);
    assert_eq!(last_type("let n = length([1.0]);"), i);
    assert_eq!(
        last_type("let n = where([true]);"),
        ValueType::Array(ScalarType::I64)
    );
    assert_eq!(last_type("let n = count([true]);"), i);
    assert_eq!(last_type("let n = cast_f64([1]);"), F64A);
    assert_eq!(last_type("let n = fill(3, 1.0);"), F64A);
    assert_eq!(last_type("let n = compress([1.0], [true]);"), F64A);
    type_err("let n = min([1.0]);", "takes 2 arguments");
}

#[test]
fn zip_map_types() {
    assert_eq!(
        last_type("let m = zip_map([1.0], [2.0], |a, b| a + b);"),
        F64A
    );
    type_err("let m = zip_map([1.0], [2.0], |a| a);", "parameter");
}

#[test]
fn duplicate_input_and_export_are_rejected() {
    let e = type_err(
        "input x: f64[]; input x: i64[];",
        "input 'x' is declared twice",
    );
    assert_eq!(e.span, Some(Span { line: 1, col: 17 }));
    let second_err = type_err(
        "export a = 1.0; export a = 2.0;",
        "export 'a' is declared twice",
    );
    assert_eq!(second_err.span, Some(Span { line: 1, col: 17 }));
    // An input and an export are different namespaces of the host interface.
    run("input a: f64; export b = a; let b = 1.0;").unwrap();
}

#[test]
fn scan_add_f64_has_no_kernel() {
    type_err(
        "let x = [1.0, 2.0]; let s = scan(x, add);",
        "no kernel for 'scan' on f64[]",
    );
}

#[test]
fn scan_rejects_unknown_operations() {
    type_err(
        "let x = [1, 2]; let s = scan(x, foo);",
        "unknown scan operation 'foo'",
    );
    type_err(
        "let x = [1, 2]; let s = scan_by(x, [true, false], foo);",
        "unknown scan_by operation 'foo'",
    );
}

#[test]
fn scan_is_rejected_inside_a_lambda() {
    type_err(
        "let x = [1, 2]; let m = map(x, |v| scan(v, add));",
        "'scan' cannot be used inside a lambda",
    );
}

#[test]
fn reduce_into_add_f64_has_no_kernel() {
    type_err(
        "let k = [0, 1]; let v = [1.0, 2.0]; let s = reduce_into(k, v, 2, add);",
        "no kernel for 'reduce_into' on i64[], f64[], i64",
    );
}
