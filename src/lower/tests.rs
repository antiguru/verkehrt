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
        TermKind::Cond(c, t, e) => format!(
            "if({}, {}, {})",
            show(c, l, g),
            show(t, l, g),
            show(e, l, g)
        ),
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
            LStmt::Bind { var, term, .. } => {
                format!("bind {} = {}", g.name(*var), show(term, l, g))
            }
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
        [
            "bind x = L_c",
            "bind y = L_c",
            "print reduce_add(mul(L_x, L_y))"
        ]
    );
}

#[test]
fn single_use_let_is_inlined() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t));"),
        [
            "bind x = L_c",
            "bind y = L_c",
            "print reduce_add(mul(L_x, L_y))"
        ]
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
    let (l, g) = lowered(
        "let x = [1.0]; let y = [2.0]; let t = x * y; print(sum(t));",
        Mode::Repl,
    );
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
fn prev_keeps_its_scalar_operand() {
    assert_eq!(
        run_dump("let x = [1, 2]; print(prev(x, 0));"),
        ["bind x = L_c", "print prev(L_x, L_c)"]
    );
    // Elementwise kernels still broadcast their scalar operands.
    assert_eq!(
        run_dump("let x = [1, 2]; print(x + 1);"),
        ["bind x = L_c", "print add(L_x, bcast(L_c))"]
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
    let LStmt::Bind { var, .. } = &l.stmts[1] else {
        panic!("expected a bind")
    };
    assert_eq!(g.name(*var), "_t0");
}

#[test]
fn inner_term_shared_only_through_parent_is_not_hoisted_twice() {
    let (l, g) = lowered(
        "let x = [1.0]; let y = [2.0]; print(sum(x * y) + sum(x * y) + sum(x));",
        Mode::Run,
    );
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
    let (l, g) = lowered(
        "let x = [1.0]; print(sum(x + x) + sum(x + x) + sum(sqrt(x + x)));",
        Mode::Run,
    );
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
fn input_and_export() {
    let (l, g) = lowered("input x: f64[]; export d = sum(x * x);", Mode::Run);
    assert_eq!(dump(&l, &g), ["bind d = reduce_add(mul(L_x, L_x))"]);
    let (var, _) = g.lookup("d").unwrap();
    assert_eq!(l.exports, [("d".to_string(), var)]);
    let (x, _) = g.lookup("x").unwrap();
    assert!(
        l.leaves
            .iter()
            .any(|leaf| matches!(leaf, Leaf::Var(v) if *v == x))
    );
}

#[test]
fn export_is_never_inlined() {
    let (l, g) = lowered("let x = [1.0]; export e = x + x; print(sum(e));", Mode::Run);
    assert_eq!(
        dump(&l, &g),
        [
            "bind x = L_c",
            "bind e = add(L_x, L_x)",
            "print reduce_add(L_e)"
        ]
    );
}

#[test]
fn zero_use_let_stays() {
    assert_eq!(
        run_dump("let x = [1.0]; let t = x + x; print(sum(x));"),
        [
            "bind x = L_c",
            "bind t = add(L_x, L_x)",
            "print reduce_add(L_x)"
        ]
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
    let b = Term {
        span: Span { line: 9, col: 9 },
        ..a.clone()
    };
    assert_eq!(a, b);
    let mut set = HashSet::new();
    set.insert(a);
    assert!(set.contains(&b));
}

#[test]
fn zip_map_reads_each_array_through_a_length_check() {
    assert_eq!(
        run_dump("let x = [1.0]; let y = [2.0]; print(zip_map(x, y, |a, b| a));"),
        [
            "bind x = L_c",
            "bind y = L_c",
            "print samelen(L_x, length(L_y))"
        ]
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

#[test]
fn lambda_parameter_shadows_outer_variable() {
    assert_eq!(
        run_dump("let v = [1.0]; let x = [2.0]; print(sum(v)); print(map(x, |v| v * 2.0));"),
        [
            "bind v = L_c",
            "bind x = L_c",
            "print reduce_add(L_v)",
            "print mul(L_x, bcast(L_c))"
        ]
    );
}

#[test]
fn function_local_shadows_outer_variable() {
    assert_eq!(
        run_dump(
            "fn f(a: f64[]) -> f64 { let x = sum(a); x + 1.0 } let x = [1.0]; let y = [2.0]; print(f(y) + sum(x));"
        ),
        [
            "bind x = L_c",
            "bind y = L_c",
            "print add(add(reduce_add(L_y), L_c), reduce_add(L_x))"
        ]
    );
}

#[test]
fn single_use_let_inlines_into_call_argument() {
    assert_eq!(
        run_dump("fn f(a: f64[]) -> f64 { sum(a) } let x = [1.0]; let t = x + x; print(f(t));"),
        ["bind x = L_c", "print reduce_add(add(L_x, L_x))"]
    );
}

#[test]
fn single_use_let_inlines_into_lambda() {
    assert_eq!(
        run_dump("let x = [1.0]; let s = sum(x) * 2.0; print(map(x, |v| v + s));"),
        [
            "bind x = L_c",
            "print add(L_x, bcast(mul(reduce_add(L_x), L_c)))"
        ]
    );
}

#[test]
fn scalar_subexpression_gets_one_broadcast() {
    assert_eq!(
        run_dump("let x = [1.0]; let a = 2.0; let b = 3.0; print(x * (a + b));"),
        [
            "bind x = L_c",
            "bind a = L_c",
            "bind b = L_c",
            "print mul(L_x, bcast(add(L_a, L_b)))"
        ]
    );
}

#[test]
fn single_use_let_inlines_across_print() {
    assert_eq!(
        run_dump("let x = [1.0]; let t = x + x; print(sum(x)); print(sum(t));"),
        [
            "bind x = L_c",
            "print reduce_add(L_x)",
            "print reduce_add(add(L_x, L_x))"
        ]
    );
}

#[test]
fn array_literal_of_non_constant_is_type_error() {
    let e = lowered_with("let x = [1.0]; print([sum(x)]);", Mode::Run).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Type);
    assert_eq!(e.message, "array literal elements must be constants");
    assert_eq!(e.span, Some(Span { line: 1, col: 23 }));
}

#[test]
fn scalar_condition_if_is_lazy() {
    assert_eq!(
        run_dump("let x = [1.0]; print(if sum(x) > 0.0 { x } else { x + x });"),
        [
            "bind x = L_c",
            "print if(gt(reduce_add(L_x), L_c), L_x, add(L_x, L_x))"
        ]
    );
}

#[test]
fn mask_condition_if_and_lambda_if_select() {
    assert_eq!(
        run_dump("let x = [1.0]; print(if x > 0.0 { 1.0 } else { 0.0 });"),
        [
            "bind x = L_c",
            "print select(gt(L_x, bcast(L_c)), bcast(L_c), bcast(L_c))"
        ]
    );
    // A condition on a captured scalar is still per element inside a lambda.
    assert_eq!(
        run_dump(
            "let x = [1.0]; let s = sum(x); print(map(x, |v| if s > 0.0 { v } else { 0.0 }));"
        ),
        [
            "bind x = L_c",
            "print select(bcast(gt(reduce_add(L_x), L_c)), L_x, bcast(L_c))"
        ]
    );
}

#[test]
fn repeated_subterm_in_branch_is_not_hoisted() {
    assert_eq!(
        run_dump("let x = [1.0]; print(if sum(x) > 0.0 { sum(x + x) + sum(x + x) } else { 0.0 });"),
        [
            "bind x = L_c",
            "print if(gt(reduce_add(L_x), L_c), add(reduce_add(add(L_x, L_x)), reduce_add(add(L_x, L_x))), L_c)"
        ]
    );
    // Repeats in the condition run unconditionally and are hoisted.
    assert_eq!(
        run_dump("let x = [1.0]; print(if sum(x) > sum(x) { sum(x) } else { 0.0 });"),
        [
            "bind x = L_c",
            "bind _t0 = reduce_add(L_x)",
            "print if(gt(L_t0, L_t0), reduce_add(L_x), L_c)"
        ]
    );
}

#[test]
fn repeated_conditional_is_hoisted_whole() {
    assert_eq!(
        run_dump("let x = [1.0]; let c = if sum(x) > 0.0 { 1.0 } else { 2.0 }; print(c + c);"),
        [
            "bind x = L_c",
            "bind c = if(gt(reduce_add(L_x), L_c), L_c, L_c)",
            "print add(L_c, L_c)"
        ]
    );
    assert_eq!(
        run_dump(
            "fn f(x: f64[]) -> f64 { if sum(x) > 0.0 { 1.0 } else { 2.0 } } let x = [1.0]; print(f(x) + f(x));"
        ),
        [
            "bind x = L_c",
            "bind _t0 = if(gt(reduce_add(L_x), L_c), L_c, L_c)",
            "print add(L_t0, L_t0)"
        ]
    );
}

#[test]
fn single_use_let_inlines_into_branch() {
    assert_eq!(
        run_dump("let x = [1.0]; let t = sum(x + x); print(if true { t } else { 0.0 });"),
        [
            "bind x = L_c",
            "print if(L_c, reduce_add(add(L_x, L_x)), L_c)"
        ]
    );
}

#[test]
fn div_const_lifts_in_a_lambda() {
    // The divisor stays a scalar, only the dividend is an array operand.
    assert_eq!(
        run_dump("let x = [100, 200]; print(map(x, |v| div_const(v, 60)));"),
        ["bind x = L_c", "print div_const(L_x, L_c)"]
    );
}

#[test]
fn div_const_with_a_captured_divisor_lifts_in_a_lambda() {
    assert_eq!(
        run_dump("let x = [100, 200]; let c = 60; print(map(x, |v| div_const(v, c)));"),
        ["bind x = L_c", "bind c = L_c", "print div_const(L_x, L_c)"]
    );
}

#[test]
fn div_const_through_functions_lifts_in_a_lambda() {
    assert_eq!(
        run_dump(
            "fn f(a: i64, b: i64) -> i64 { div_const(a, b) } \
             let x = [100, 200]; print(map(x, |v| f(v, 3)));"
        ),
        ["bind x = L_c", "print div_const(L_x, L_c)"]
    );
    assert_eq!(
        run_dump(
            "fn g(a: i64, b: i64) -> i64 { div_const(a, b) } \
             fn f(a: i64, b: i64) -> i64 { g(a + 1, b * 2) } \
             let x = [100, 200]; print(map(x, |v| f(v, 3)));"
        ),
        [
            "bind x = L_c",
            "print div_const(add(L_x, bcast(L_c)), mul(L_c, L_c))"
        ]
    );
}

/// Every lambda the checker accepts must also lower, so the checker and lowering cannot
/// disagree. Bodies combine operands that read the parameter (`v`, `v + 1`, `pv` from a
/// block let) with operands that do not (a literal, a captured scalar, a uniform expression,
/// `cu` from a block let, a captured `bool` condition), directly and through user functions,
/// under `map`, `filter`, and `zip_map` with a second array of each element type.
#[test]
fn checked_lambdas_lower() {
    let types = [
        ("i64", "3"),
        ("f64", "3.0"),
        ("i8", "3i8"),
        ("u64", "3u64"),
        ("i32", "3i32"),
        ("u8", "3u8"),
    ];
    let binary = [
        "{a} + {b}",
        "{a} - {b}",
        "{a} * {b}",
        "{a} / {b}",
        "cast_i64({a} < {b})",
        "cast_i64({a} == {b})",
        "min({a}, {b})",
        "max({a}, {b})",
        "and({a}, {b})",
        "xor({a}, {b})",
        "div_const({a}, {b})",
        "f({a}, {b})",
        "h({a}, {b})",
        "if {a} > {b} {{ {a} }} else {{ {b} }}",
        "if flag {{ {a} }} else {{ {b} }}",
        "select({a} > {b}, {a}, {b})",
    ];
    let unary = ["neg({a})", "abs({a})", "sqrt({a})", "not({a})"];
    // Higher-order calls: a name, the call with `{body}` for the lambda body, and the
    // element type of a second array.
    let mut calls = vec![
        ("map", "map(x, |v| {body})".to_string(), None),
        ("filter", "filter(x, |v| ({body}) > c)".to_string(), None),
    ];
    for (u, ulit) in types {
        calls.push((
            "zip_map",
            format!("zip_map(x, [{ulit}, {ulit}], |v, q| {{body}})"),
            Some(u),
        ));
    }
    let reg = Registry::new();
    // Accepted programs per element type and call.
    let mut accepted: HashMap<(&str, &str, Option<&str>), usize> = HashMap::new();
    let mut rejected = 0;
    for (ty, lit) in types {
        let one = lit.replacen('3', "1", 1);
        let atoms = [
            "v".to_string(),
            format!("(v + {one})"),
            "pv".to_string(),
            lit.to_string(),
            "c".to_string(),
            format!("(c * {lit})"),
            "cu".to_string(),
        ];
        // The functions are defined only where used, since `div_const` lacks some types.
        let fns = format!(
            "fn f(a: {ty}, b: {ty}) -> {ty} {{ div_const(a, b) }} \
             fn h(a: {ty}, b: {ty}) -> {ty} {{ f(a, b) + a }}"
        );
        let mut bodies = Vec::new();
        for shape in binary {
            for a in &atoms {
                for b in &atoms {
                    bodies.push(shape.replace("{a}", a).replace("{b}", b));
                }
            }
        }
        for shape in unary {
            for a in &atoms {
                bodies.push(shape.replace("{a}", a));
            }
        }
        for body in &bodies {
            let prelude = if body.contains("f(") || body.contains("h(") {
                fns.as_str()
            } else {
                ""
            };
            // Block lets are bound only where used, so a let without a kernel does not
            // reject every body of its type.
            let mut lets = String::new();
            if body.contains("pv") {
                lets.push_str(&format!("let pv = v * {lit}; "));
            }
            if body.contains("cu") {
                lets.push_str(&format!("let cu = c + {one}; "));
            }
            let body = format!("{{ {lets}{body} }}");
            for (name, call, second) in &calls {
                let call = call.replace("{body}", &body);
                let src = format!(
                    "{prelude} let x = [{lit}, {one}]; let c = {lit}; let flag = true; \
                     print({call});"
                );
                let mut g = Globals::new();
                let Ok(typed) = check(&parse(&src).unwrap(), &reg, &mut g) else {
                    rejected += 1;
                    continue;
                };
                *accepted.entry((ty, *name, *second)).or_default() += 1;
                if let Err(e) = lower(&typed, &mut g, &reg, Mode::Run) {
                    panic!("checked but failed to lower: {src}: {e}");
                }
            }
        }
    }
    // Every element type has accepted programs under every call, except where an `i8` array
    // is involved. `i8` has scalar kernels only, so a varying body has no kernel, and
    // `zip_map` and the fill of a uniform body need `length` on the array.
    for (ty, _) in types {
        for (name, _, second) in &calls {
            let expected = ty != "i8" && *second != Some("i8");
            let n = accepted.get(&(ty, *name, *second)).copied().unwrap_or(0);
            assert_eq!(n > 0, expected, "{ty} {name} {second:?}: {n} accepted");
        }
    }
    assert!(rejected > 1000, "{rejected} rejected");
}

#[test]
fn no_kernel_error_for_a_scalar_input_names_the_operands_as_given() {
    // `div_const` has a `Scalar` input, so a scalar next to an array is not broadcast to
    // find a kernel and the message must show `f64`, not the lifted `f64[]`.
    let g = Globals::new();
    let reg = Registry::new();
    let mut lw = Lowerer {
        g: &g,
        reg: &reg,
        leaves: Leaves(Vec::new()),
        env: HashMap::new(),
        lambda_depth: 0,
    };
    let span = Span { line: 1, col: 1 };
    let xs = lw.leaf(Leaf::Var(VarId(0)), ValueType::Array(ScalarType::I64), span);
    let d = lw.leaf(
        Leaf::Var(VarId(1)),
        ValueType::Scalar(ScalarType::F64),
        span,
    );
    let e = lw.op(Op::DivConst, vec![xs, d], span).unwrap_err();
    assert_eq!(e.message, "no kernel for 'div_const' on i64[], f64");
}
