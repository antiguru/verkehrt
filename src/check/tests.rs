use super::*;
use crate::error::{Error, ErrorKind};
use crate::syntax::parse;
use crate::value::{ScalarType, ValueType};

const F64A: ValueType = ValueType::Array(ScalarType::F64);

fn run(src: &str) -> Result<(TProgram, Globals), Error> {
    let prog = parse(src).unwrap();
    let reg = Registry::new();
    let mut g = Globals::new();
    check(&prog, &reg, &mut g).map(|p| (p, g))
}

/// Renders the type of the last `let` or `export` in `src`, or the type error with its
/// span as `error: <message> at <line>:<col>`. Any other error kind is a failure.
fn type_of(src: &str) -> String {
    match run(src) {
        Ok((p, _)) => match p.stmts.last().unwrap() {
            TStmt::Let { value, .. } | TStmt::Export { value, .. } => value.ty.to_string(),
            other => panic!("unexpected {other:?}"),
        },
        Err(e) => {
            assert_eq!(e.kind, ErrorKind::Type, "{e}");
            match e.span {
                Some(s) => format!("error: {} at {}:{}", e.message, s.line, s.col),
                None => format!("error: {}", e.message),
            }
        }
    }
}

/// Runs the checker, then the unread-binding rule in the given mode, and renders the first
/// error as `error: <message> at <line>:<col>`, or `ok`.
fn unread(src: &str, repl: bool) -> String {
    let prog = parse(src).unwrap();
    let mut g = Globals::new();
    let res = check(&prog, &Registry::new(), &mut g).and_then(|_| check_unread(&prog, repl));
    match res {
        Ok(()) => "ok".into(),
        Err(e) => {
            assert_eq!(e.kind, ErrorKind::Type, "{e}");
            let s = e.span.expect("the rule reports a span");
            format!("error: {} at {}:{}", e.message, s.line, s.col)
        }
    }
}

/// Runs the datadriven cases in `testdata`. The `type` directive takes a program and renders
/// one line from `type_of`. The `unread` directive takes a program and an optional
/// `mode=repl` and renders the outcome of the unread-binding rule.
#[test]
fn datadriven_types() {
    crate::testutil::run_dir(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/check/testdata"),
        |tc| match tc.directive.as_str() {
            "type" => {
                crate::testutil::reject_unknown_args(tc, &[]);
                format!("{}\n", type_of(&tc.input))
            }
            "unread" => {
                crate::testutil::reject_unknown_args(tc, &["mode"]);
                let repl = match tc.args.get("mode").map(|v| v[0].as_str()) {
                    None | Some("run") => false,
                    Some("repl") => true,
                    Some(m) => panic!("unknown mode {m}"),
                };
                format!("{}\n", unread(&tc.input, repl))
            }
            d => panic!("unknown directive {d}"),
        },
    );
}

#[test]
fn function_scope_and_shadowing() {
    // The errors of this case are in `testdata/functions.dd`.
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
