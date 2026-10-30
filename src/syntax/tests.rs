use super::ast::*;
use super::parse;
use crate::error::{ErrorKind, Span};
use crate::value::{Scalar, ScalarType};

/// Parses `src` as a single expression statement.
fn expr(src: &str) -> ExprKind {
    let p = parse(&format!("{src};")).unwrap();
    assert_eq!(p.items.len(), 1);
    match p.items.into_iter().next().unwrap() {
        Item::Expr(e, _) => e.kind,
        other => panic!("not an expression item: {other:?}"),
    }
}

fn err(src: &str) -> crate::error::Error {
    let e = parse(src).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Parse);
    e
}

fn name(n: &str) -> ExprKind {
    ExprKind::Name(n.into())
}

const NOSPAN: Span = Span { line: 0, col: 0 };

fn bin(op: BinOp, l: ExprKind, r: ExprKind) -> ExprKind {
    let wrap = |k| Box::new(Expr { kind: k, span: NOSPAN });
    ExprKind::Binary(op, wrap(l), wrap(r))
}

/// Rewrites every span to a fixed value so shapes compare without positions.
fn strip(k: &ExprKind) -> ExprKind {
    let s = |e: &Expr| Box::new(Expr { kind: strip(&e.kind), span: NOSPAN });
    match k {
        ExprKind::Binary(op, l, r) => ExprKind::Binary(*op, s(l), s(r)),
        ExprKind::Call(n, args) => ExprKind::Call(
            n.clone(),
            args.iter().map(|a| Expr { kind: strip(&a.kind), span: NOSPAN }).collect(),
        ),
        ExprKind::Lambda(ps, b) => ExprKind::Lambda(ps.clone(), s(b)),
        other => other.clone(),
    }
}

#[test]
fn same_operator_folds_left() {
    assert_eq!(
        strip(&expr("a + b + c")),
        bin(BinOp::Add, bin(BinOp::Add, name("a"), name("b")), name("c"))
    );
    assert_eq!(
        strip(&expr("a - b - c")),
        bin(BinOp::Sub, bin(BinOp::Sub, name("a"), name("b")), name("c"))
    );
}

#[test]
fn mixed_operators_need_parentheses() {
    let e = err("a + b * c");
    assert_eq!(e.span, Some(Span { line: 1, col: 7 }));
    assert!(e.message.contains("parentheses"), "{}", e.message);
}

#[test]
fn comparison_chains_need_parentheses() {
    let e = err("a < b < c;");
    assert!(e.message.contains("parentheses"), "{}", e.message);
}

#[test]
fn parenthesized_mix_parses() {
    assert_eq!(
        strip(&expr("a + (b * c)")),
        bin(BinOp::Add, name("a"), bin(BinOp::Mul, name("b"), name("c")))
    );
}

#[test]
fn negative_literals() {
    assert_eq!(
        strip(&expr("x - 1.0")),
        bin(BinOp::Sub, name("x"), ExprKind::Lit(Scalar::F64(1.0)))
    );
    match expr("f(-1.0)") {
        ExprKind::Call(n, args) => {
            assert_eq!(n, "f");
            assert_eq!(args[0].kind, ExprKind::Lit(Scalar::F64(-1.0)));
        }
        k => panic!("{k:?}"),
    }
}

#[test]
fn suffixed_literals() {
    assert_eq!(expr("2.0f32"), ExprKind::Lit(Scalar::F32(2.0)));
    assert_eq!(expr("7u8"), ExprKind::Lit(Scalar::U8(7)));
    assert_eq!(expr("1i32"), ExprKind::Lit(Scalar::I32(1)));
    assert_eq!(expr("1"), ExprKind::Lit(Scalar::I64(1)));
    assert_eq!(expr("3u64"), ExprKind::Lit(Scalar::U64(3)));
    assert_eq!(expr("3f32"), ExprKind::Lit(Scalar::F32(3.0)));
    assert_eq!(expr("-128i8"), ExprKind::Lit(Scalar::I8(-128)));
    assert_eq!(expr("true"), ExprKind::Lit(Scalar::Bool(true)));
}

#[test]
fn bad_literals() {
    err("1.5i32;");
    err("300u8;");
    err("-1u8;");
    err("1bool;");
    err("1x;");
    err("1.0f32 1;");
    err("340282350000000000000000000000000000000000.0f32;");
}

#[test]
fn lambda_call() {
    match expr("filter(x, |v| v > 0.0)") {
        ExprKind::Call(n, args) => {
            assert_eq!(n, "filter");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].kind, name("x"));
            match &args[1].kind {
                ExprKind::Lambda(ps, body) => {
                    assert_eq!(ps, &vec!["v".to_string()]);
                    assert_eq!(
                        strip(&body.kind),
                        bin(BinOp::Gt, name("v"), ExprKind::Lit(Scalar::F64(0.0)))
                    );
                }
                k => panic!("{k:?}"),
            }
        }
        k => panic!("{k:?}"),
    }
}

#[test]
fn if_expression() {
    assert!(matches!(expr("if c { 1.0 } else { 2.0 }"), ExprKind::If(..)));
}

#[test]
fn whole_program() {
    let p = parse("fn dot(x: f64[], y: f64[]) -> f64 { sum(x * y) }\nlet d = dot(a, b);\nprint(d);")
        .unwrap();
    assert_eq!(p.items.len(), 3);
    let f64a = TypeExpr { elem: ScalarType::F64, array: true };
    match &p.items[0] {
        Item::Fn(f) => {
            assert_eq!(f.name, "dot");
            assert_eq!(f.params, vec![("x".to_string(), f64a), ("y".to_string(), f64a)]);
            assert_eq!(f.ret, TypeExpr { elem: ScalarType::F64, array: false });
            assert!(f.body.lets.is_empty());
        }
        i => panic!("{i:?}"),
    }
    assert!(matches!(&p.items[1], Item::Let(l) if l.name == "d" && l.ty.is_none()));
    assert!(matches!(&p.items[2], Item::Print(e, _) if e.kind == name("d")));
}

#[test]
fn block_with_lets() {
    let p = parse("fn f(x: f64) -> f64 { let a = x; let b = a; b }").unwrap();
    match &p.items[0] {
        Item::Fn(f) => {
            assert_eq!(f.body.lets.len(), 2);
            assert_eq!(f.body.result.kind, name("b"));
        }
        i => panic!("{i:?}"),
    }
}

#[test]
fn typed_let_with_string() {
    let p = parse("let x: f64[] = load_f64(\"x.bin\");").unwrap();
    match &p.items[0] {
        Item::Let(l) => {
            assert_eq!(l.ty, Some(TypeExpr { elem: ScalarType::F64, array: true }));
            match &l.value.kind {
                ExprKind::Call(n, args) => {
                    assert_eq!(n, "load_f64");
                    assert_eq!(args[0].kind, ExprKind::Str("x.bin".into()));
                }
                k => panic!("{k:?}"),
            }
        }
        i => panic!("{i:?}"),
    }
}

#[test]
fn byte_strings_rejected() {
    let e = err("b\"abc\";");
    assert!(e.message.contains("byte strings"), "{}", e.message);
}

#[test]
fn input_and_export() {
    let p = parse("input x: f64[]; export s = sum(x);").unwrap();
    assert_eq!(p.items.len(), 2);
    match &p.items[0] {
        Item::Input(n, t, _) => {
            assert_eq!(n, "x");
            assert_eq!(*t, TypeExpr { elem: ScalarType::F64, array: true });
        }
        i => panic!("{i:?}"),
    }
    assert!(matches!(&p.items[1], Item::Export(l) if l.name == "s" && l.ty.is_none()));
}

#[test]
fn missing_semicolon_reports_next_token() {
    let e = err("let a = 1\nlet b = 2;");
    assert_eq!(e.span, Some(Span { line: 2, col: 1 }));
}

#[test]
fn array_literal_and_comments() {
    let p = parse("[1.0, 2.0]; // trailing\n// whole line\n").unwrap();
    match &p.items[0] {
        Item::Expr(e, _) => assert!(matches!(&e.kind, ExprKind::ArrayLit(v) if v.len() == 2)),
        i => panic!("{i:?}"),
    }
}

#[test]
fn binary_node_span_is_operator_span() {
    let p = parse("a + (b * c);").unwrap();
    let Item::Expr(outer, _) = &p.items[0] else { panic!() };
    assert_eq!(outer.span, Span { line: 1, col: 3 });
    let ExprKind::Binary(_, l, r) = &outer.kind else { panic!() };
    assert_eq!(l.span, Span { line: 1, col: 1 });
    assert_eq!(r.span, Span { line: 1, col: 8 });

    let p = parse("x * 2;").unwrap();
    let Item::Expr(e, _) = &p.items[0] else { panic!() };
    assert_eq!(e.span, Span { line: 1, col: 3 });
}
