use super::*;
use crate::check::check;
use crate::error::ErrorKind;
use crate::syntax::parse;

fn lowered_with(src: &str, mode: Mode) -> Result<(Lowered, Globals), Error> {
    let prog = parse(src).unwrap();
    let reg = Registry::new();
    let mut g = Globals::new();
    let typed = check(&prog, &reg, &mut g).unwrap_or_else(|e| panic!("{src}: {e}"));
    lower(&typed, &mut g, &reg, mode).map(|l| (l, g))
}

fn lowered(src: &str, mode: Mode) -> (Lowered, Globals) {
    lowered_with(src, mode).unwrap_or_else(|e| panic!("{src}: {e}"))
}

/// Ops in lowercase, variable leaves as `L_name`, constants as `L_c`, broadcasts as `bcast(..)`.
fn show(t: &Term, l: &Lowered, g: &Globals) -> String {
    match &t.kind {
        TermKind::Leaf(id) => match &l.leaves[id.0 as usize] {
            Leaf::Var(v) => format!("L_{}", g.name(*v).trim_start_matches('_')),
            Leaf::Const(_) => "L_c".to_string(),
        },
        TermKind::Bcast(a) => format!("bcast({})", show(a, l, g)),
        TermKind::Op(op, args) => {
            let args: Vec<String> = args.iter().map(|a| show(a, l, g)).collect();
            format!("{}({})", op_name(*op), args.join(", "))
        }
    }
}

/// One line per statement: `bind NAME = TERM`, `print TERM`, or `show TERM`.
fn dump(l: &Lowered, g: &Globals) -> Vec<String> {
    l.stmts
        .iter()
        .map(|s| match s {
            LStmt::Bind { var, term, .. } => format!("bind {} = {}", g.name(*var), show(term, l, g)),
            LStmt::Print { term, .. } => format!("print {}", show(term, l, g)),
            LStmt::Show { term, .. } => format!("show {}", show(term, l, g)),
        })
        .collect()
}

fn run_dump(src: &str) -> Vec<String> {
    let (l, g) = lowered(src, Mode::Run);
    dump(&l, &g)
}

#[test]
fn array_lets_stay_binds() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; print(sum(x * y));"),
        ["bind x = L_c", "bind y = L_c", "print reduce_add(mul(L_x, L_y))"]
    );
}

#[test]
fn single_use_let_is_inlined() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t));"),
        ["bind x = L_c", "bind y = L_c", "print reduce_add(mul(L_x, L_y))"]
    );
}

#[test]
fn multi_use_let_stays() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t)); print(t);"),
        [
            "bind x = L_c",
            "bind y = L_c",
            "bind t = mul(L_x, L_y)",
            "print reduce_add(L_t)",
            "print L_t"
        ]
    );
}

#[test]
fn repl_keeps_every_let() {
    let (l, g) = lowered("let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t));", Mode::Repl);
    assert_eq!(
        dump(&l, &g),
        [
            "bind x = L_c",
            "bind y = L_c",
            "bind t = mul(L_x, L_y)",
            "print reduce_add(L_t)"
        ]
    );
}

#[test]
fn inlining_chains() {
    assert_eq!(
        run_dump("let x = [1.0]; let a = x + x; let b = a * x; print(sum(b));"),
        ["bind x = L_c", "print reduce_add(mul(add(L_x, L_x), L_x))"]
    );
}

#[test]
fn filter_compresses_by_lifted_predicate() {
    assert_eq!(
        run_dump("let x = [1.0]; print(filter(x, |v| v > 0.5));"),
        ["bind x = L_c", "print compress(L_x, gt(L_x, bcast(L_c)))"]
    );
}

#[test]
fn constant_body_is_filled() {
    assert_eq!(
        run_dump("let x = [1.0]; print(map(x, |v| 1.0));"),
        ["bind x = L_c", "print fill(length(L_x), L_c)"]
    );
}

#[test]
fn if_becomes_select() {
    assert_eq!(
        run_dump("let x = [1.0]; print(map(x, |v| if v > 0.0 { v } else { 0.0 - v }));"),
        [
            "bind x = L_c",
            "print select(gt(L_x, bcast(L_c)), L_x, sub(bcast(L_c), L_x))"
        ]
    );
}

#[test]
fn named_function_inlines_through_map() {
    assert_eq!(
        run_dump("fn sq(v: f64) -> f64 { v * v } let x = [1.0]; print(sum(map(x, sq)));"),
        ["bind x = L_c", "print reduce_add(mul(L_x, L_x))"]
    );
}

#[test]
fn function_called_twice_and_nested() {
    assert_eq!(
        run_dump(
            "fn sq(v: f64[]) -> f64[] { v * v } fn quad(v: f64[]) -> f64[] { sq(sq(v)) } \
             let x = [1.0]; let y = [2.0]; print(quad(x) + y);"
        ),
        [
            "bind x = L_c",
            "bind y = L_c",
            "bind _t0 = mul(L_x, L_x)",
            "print add(mul(L_t0, L_t0), L_y)"
        ]
    );
}

#[test]
fn block_lets_are_substituted() {
    assert_eq!(
        run_dump("fn f(v: f64[]) -> f64 { let s = sum(v); s + 1.0 } let x = [1.0]; print(f(x));"),
        ["bind x = L_c", "print add(reduce_add(L_x), L_c)"]
    );
}

#[test]
fn repeated_subterm_is_hoisted() {
    let (l, g) = lowered("let x = [1.0]; print(sum(x + x) + sum(x + x));", Mode::Run);
    assert_eq!(
        dump(&l, &g),
        [
            "bind x = L_c",
            "bind _t0 = reduce_add(add(L_x, L_x))",
            "print add(L_t0, L_t0)"
        ]
    );
    let LStmt::Bind { var, .. } = &l.stmts[1] else { panic!("expected a bind") };
    assert_eq!(g.name(*var), "_t0");
}

#[test]
fn inner_term_shared_only_through_parent_is_not_hoisted_twice() {
    let (l, g) = lowered("let x = [1.0]; let y = [2.0]; print(sum(x * y) + sum(x * y) + sum(x));", Mode::Run);
    assert_eq!(
        dump(&l, &g),
        [
            "bind x = L_c",
            "bind y = L_c",
            "bind _t0 = reduce_add(mul(L_x, L_y))",
            "print add(add(L_t0, L_t0), reduce_add(L_x))"
        ]
    );
}

#[test]
fn inner_term_also_used_elsewhere_is_hoisted_separately() {
    let (l, g) = lowered("let x = [1.0]; print(sum(x + x) + sum(x + x) + sum(sqrt(x + x)));", Mode::Run);
    assert_eq!(
        dump(&l, &g),
        [
            "bind x = L_c",
            "bind _t0 = add(L_x, L_x)",
            "bind _t1 = reduce_add(L_t0)",
            "print add(add(L_t1, L_t1), reduce_add(sqrt(L_t0)))"
        ]
    );
}

#[test]
fn map_over_i8_has_no_array_kernel() {
    let e = lowered_with("let x = [1i8]; print(map(x, |v| v * v));", Mode::Run).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Type);
    assert!(e.message.contains("no kernel"), "{}", e.message);
    assert_eq!(e.span, Some(Span { line: 1, col: 35 }));
}

#[test]
fn input_and_export() {
    let (l, g) = lowered("input x: f64[]; export d = sum(x * x);", Mode::Run);
    assert_eq!(dump(&l, &g), ["bind d = reduce_add(mul(L_x, L_x))"]);
    let (var, _) = g.lookup("d").unwrap();
    assert_eq!(l.exports, [("d".to_string(), var)]);
    let (x, _) = g.lookup("x").unwrap();
    assert!(l.leaves.iter().any(|leaf| matches!(leaf, Leaf::Var(v) if *v == x)));
}

#[test]
fn export_is_never_inlined() {
    let (l, g) = lowered("let x = [1.0]; export e = x + x; print(sum(e));", Mode::Run);
    assert_eq!(
        dump(&l, &g),
        ["bind x = L_c", "bind e = add(L_x, L_x)", "print reduce_add(L_e)"]
    );
}

#[test]
fn zero_use_let_stays() {
    assert_eq!(
        run_dump("let x = [1.0]; let t = x + x; print(sum(x));"),
        ["bind x = L_c", "bind t = add(L_x, L_x)", "print reduce_add(L_x)"]
    );
}

#[test]
fn scalar_capture_is_broadcast() {
    assert_eq!(
        run_dump("let x = [1.0]; let s = sum(x); print(map(x, |v| v * s));"),
        ["bind x = L_c", "print mul(L_x, bcast(reduce_add(L_x)))"]
    );
}

#[test]
fn term_equality_ignores_span() {
    let a = Term {
        kind: TermKind::Leaf(LeafId(0)),
        ty: ValueType::Array(ScalarType::F64),
        span: Span { line: 1, col: 1 },
    };
    let b = Term { span: Span { line: 9, col: 9 }, ..a.clone() };
    assert_eq!(a, b);
    let mut set = HashSet::new();
    set.insert(a);
    assert!(set.contains(&b));
}

#[test]
fn zip_map_reads_each_array_through_a_length_check() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; print(zip_map(x, y, |a, b| a));"),
        ["bind x = L_c", "bind y = L_c", "print samelen(L_x, length(L_y))"]
    );
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; print(zip_map(x, y, |a, b| a + b));"),
        [
            "bind x = L_c",
            "bind y = L_c",
            "print add(samelen(L_x, length(L_y)), samelen(L_y, length(L_x)))"
        ]
    );
}
