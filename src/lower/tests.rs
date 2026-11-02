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

/// Renders the dump of `src` lowered in `mode`, or the lowering error with its span as
/// `error: <message> at <line>:<col>`. The program must check, and any lowering error
/// must be a type error.
fn lower_output(src: &str, mode: Mode) -> String {
    match lowered_with(src, mode) {
        Ok((l, g)) => dump(&l, &g)
            .iter()
            .map(|line| format!("{line}\n"))
            .collect(),
        Err(e) => {
            assert_eq!(e.kind, ErrorKind::Type, "{e}");
            match e.span {
                Some(s) => format!("error: {} at {}:{}\n", e.message, s.line, s.col),
                None => format!("error: {}\n", e.message),
            }
        }
    }
}

/// Runs the `lower` directive on every file in `testdata`. The `mode` argument is `run` or
/// `repl`, and the output is one line per statement from `dump`.
#[test]
fn datadriven_lowering() {
    crate::testutil::run_dir(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/lower/testdata"),
        |tc| match tc.directive.as_str() {
            "lower" => {
                crate::testutil::reject_unknown_args(tc, &["mode"]);
                let mode = match tc.args["mode"][0].as_str() {
                    "run" => Mode::Run,
                    "repl" => Mode::Repl,
                    m => panic!("unknown mode {m}"),
                };
                lower_output(&tc.input, mode)
            }
            d => panic!("unknown directive {d}"),
        },
    );
}

/// The names of the `Bind` statements of `src` lowered in `mode`, each with whether it is an
/// output, in statement order.
fn bind_outputs(src: &str, mode: Mode) -> Vec<(String, bool)> {
    let (l, g) = lowered(src, mode);
    l.stmts
        .iter()
        .filter_map(|s| match s {
            LStmt::Bind { var, output, .. } => Some((g.name(*var).to_string(), *output)),
            LStmt::Print { .. } | LStmt::Show { .. } => None,
        })
        .collect()
}

#[test]
fn exports_and_repl_lets_are_outputs() {
    let src = "input x: f64[];\nlet a = x + x;\nprint(sum(a));\nprint(sum(a));\n\
               export d = sum(x * x) + sum(x * x);\n";
    let binds = |mode| bind_outputs(src, mode);
    let names = |v: &[(&str, bool)]| -> Vec<(String, bool)> {
        v.iter().map(|(n, o)| (n.to_string(), *o)).collect()
    };
    // The hoisted temporary of the export is no output, the export after it is.
    assert_eq!(
        binds(Mode::Run),
        names(&[("a", false), ("_t0", false), ("d", true)])
    );
    assert_eq!(
        binds(Mode::Repl),
        names(&[("a", true), ("_t0", false), ("d", true)])
    );
}

#[test]
fn input_and_export() {
    // The dump of this case is in `testdata/lets.dd`.
    let (l, g) = lowered("input x: f64[]; export d = sum(x * x);", Mode::Run);
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
        lifted: Vec::new(),
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
