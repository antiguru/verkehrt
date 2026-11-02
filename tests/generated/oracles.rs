//! The oracles. Each program is planned twice on the scalar interpreter, with fused patterns
//! and without, and each plan runs on every variant the CPU supports.
//!
//! * Oracle 1, cross-variant: every variant gives the scalar outcome of the same plan. The
//!   variants share the plan, so the outcomes must agree exactly, including the error message
//!   and its location.
//! * Oracle 2, fused against unfused: the unfused plan gives the fused plan's output and
//!   exports, and fails at the same output with the same error kind. Which error surfaces may
//!   differ, because a statement can read several bindings that hold errors from different
//!   statements, and fusion and the comparison flip change which operand a kernel evaluates
//!   first. So neither the line, the category, see `Category`, nor the message is compared,
//!   and the message also differs because a fused kernel words a length mismatch its own way.
//!   Oracle 3 checks the line and category of both plans' errors against the reference
//!   evaluator.
//! * Oracle 3, reference evaluator: the scalar outcome of the fused plan equals what the
//!   evaluator of `eval` computes from the typed program. On success the output and the
//!   exports must be equal. Whether an error occurs and which output raises it are fixed, so
//!   on failure the error must be a runtime error in one of the categories the evaluator met
//!   in its failing output. Its line must be on the chain of an error of that category, a
//!   statement from the output to the failing expression, because an error reports the line
//!   of the statement that lowering computed it in, see `eval`. Which of several errors
//!   within that output surfaces may differ. The prints before the failing output appear in
//!   full, so the printed output must equal the evaluator's. Oracle 3 applies to the scalar
//!   outcomes of both plans.
//!
//! Exports compare by bits, except that every NaN equals every NaN, because the base spec
//! lets the sign and payload of a NaN result differ between variants. Printed output is
//! compared as text, which writes every NaN as `NaN`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use verkehrt::error::{ErrorKind, Span};
use verkehrt::kernels::{Backend, Isa, Registry, Variant};
use verkehrt::plan::SumRewrite;
use verkehrt::value::{Scalar, Value, ValueType};
use verkehrt::{Interpreter, Options, Prepared};

use crate::eval::{self, Reference};
use crate::r#gen::Case;
use crate::rewrites;

/// An export: its name, type, comparable bits, and the value as printed.
#[derive(Debug)]
struct Export {
    name: String,
    ty: ValueType,
    bits: Vec<Bits>,
    shown: String,
}

/// One element of a value. NaNs of every sign and payload are one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bits {
    Nan,
    Of(u64),
}

impl PartialEq for Export {
    fn eq(&self, other: &Export) -> bool {
        self.name == other.name && self.ty == other.ty && self.bits == other.bits
    }
}

#[derive(Debug, PartialEq)]
struct Failure {
    kind: ErrorKind,
    message: String,
    span: Option<Span>,
}

/// What one plan did on one variant.
#[derive(Debug, PartialEq)]
struct Outcome {
    printed: String,
    result: Result<Vec<Export>, Failure>,
}

/// What an error means, independent of the kernel that reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Category {
    LengthMismatch,
    GatherIndex,
    SliceBounds,
    BucketKey,
    NegativeBucketCount,
    NegativeLength,
    /// A reduction without an identity on an empty array, with the reduction's name.
    Empty(String),
    DivisionByZero,
    TooLarge,
    /// Any other message, compared verbatim.
    Other(String),
}

impl Category {
    pub fn of(message: &str) -> Category {
        if message.starts_with("length mismatch") {
            Category::LengthMismatch
        } else if message.starts_with("gather index") {
            Category::GatherIndex
        } else if message.starts_with("slice ") && message.contains("out of range") {
            Category::SliceBounds
        } else if message.starts_with("bucket key") {
            Category::BucketKey
        } else if message.starts_with("negative bucket count") {
            Category::NegativeBucketCount
        } else if message.starts_with("negative length") {
            Category::NegativeLength
        } else if let Some(what) = message.strip_suffix(" of an empty array") {
            Category::Empty(what.to_string())
        } else if message == "div_const: division by zero" {
            Category::DivisionByZero
        } else if message.contains("is too large") {
            Category::TooLarge
        } else {
            Category::Other(message.to_string())
        }
    }
}

impl Outcome {
    /// The parts that oracle 2 compares: output, exports, and the error's kind.
    fn fusion_view(&self) -> (&str, Result<&[Export], ErrorKind>) {
        (
            &self.printed,
            self.result.as_ref().map(Vec::as_slice).map_err(|f| f.kind),
        )
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "printed: {:?}", self.printed)?;
        match &self.result {
            Ok(exports) => {
                for e in exports {
                    write!(f, "\n  export {}: {} = {}", e.name, e.ty, e.shown)?;
                }
            }
            Err(e) => write!(f, "\n  error {:?} at {:?}: {}", e.kind, e.span, e.message)?,
        }
        Ok(())
    }
}

fn bits(v: &Value) -> Vec<Bits> {
    let one = |s: Scalar| match s {
        Scalar::F64(x) if x.is_nan() => Bits::Nan,
        Scalar::F32(x) if x.is_nan() => Bits::Nan,
        Scalar::F64(x) => Bits::Of(x.to_bits()),
        Scalar::F32(x) => Bits::Of(u64::from(x.to_bits())),
        Scalar::I8(x) => Bits::Of(i64::from(x).cast_unsigned()),
        Scalar::I16(x) => Bits::Of(i64::from(x).cast_unsigned()),
        Scalar::I32(x) => Bits::Of(i64::from(x).cast_unsigned()),
        Scalar::I64(x) => Bits::Of(x.cast_unsigned()),
        Scalar::U8(x) => Bits::Of(u64::from(x)),
        Scalar::U16(x) => Bits::Of(u64::from(x)),
        Scalar::U32(x) => Bits::Of(u64::from(x)),
        Scalar::U64(x) => Bits::Of(x),
        Scalar::Bool(x) => Bits::Of(u64::from(x)),
    };
    match v {
        Value::Scalar(s) => vec![one(s.clone())],
        Value::Array(a) => (0..a.len()).map(|i| one(a.get(i))).collect(),
        Value::Str(_) => unreachable!("programs export no strings"),
    }
}

/// Every instruction set the CPU supports, each with both backends. Scalar comes first, it
/// is the reference.
fn variants() -> Vec<Variant> {
    let variants: Vec<Variant> = Isa::ALL
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
        .collect();
    assert_eq!(variants[0].isa, Isa::Scalar, "scalar is the reference");
    variants
}

/// The interpreters for planning and for running plans, and the registry that the reference
/// evaluator type-checks with.
pub struct Engine {
    fused: Interpreter,
    unfused: Interpreter,
    runners: Vec<(Variant, Interpreter)>,
    reg: Registry,
    /// The integer sum rewrites the fused planner emits.
    rewrites: Vec<SumRewrite>,
}

/// What a checked case did, for the statistics.
pub struct Verdict {
    /// The error that stopped the program, with its message.
    pub error: Option<(ErrorKind, String)>,
    /// Whether the fused plan differs from the unfused one in some statement.
    pub fusion_changed_plan: bool,
    /// The integer sum rewrites that fired, with the number of statements in which each did.
    pub rewrites_fired: BTreeMap<SumRewrite, usize>,
    /// Whether the case has an unread binding that the generator made to fail and that fails
    /// when the reference evaluator forces it alone.
    pub failing_unread: bool,
}

impl Engine {
    pub fn new() -> Engine {
        let planner = |no_fusion| {
            Interpreter::new(Options {
                isa: Some(Isa::Scalar),
                show_plan: true,
                no_fusion,
                ..Options::default()
            })
            .expect("scalar is always available")
        };
        let runners = variants()
            .into_iter()
            .map(|v| {
                let interp = Interpreter::new(Options {
                    isa: Some(v.isa),
                    backend: v.backend,
                    ..Options::default()
                })
                .expect("the variant is available");
                (v, interp)
            })
            .collect();
        let reg = Registry::new();
        Engine {
            fused: planner(false),
            unfused: planner(true),
            runners,
            rewrites: rewrites::expected(&reg),
            reg,
        }
    }

    pub fn variant_count(&self) -> usize {
        self.runners.len()
    }

    /// The integer sum rewrites the fused planner emits.
    pub fn rewrites(&self) -> &[SumRewrite] {
        &self.rewrites
    }

    /// Plans `case` with and without fusion, runs both plans on every variant, evaluates it
    /// with the reference evaluator, and applies the three oracles. A program that does not
    /// compile is a generator bug.
    pub fn check(&self, case: &Case) -> Result<Verdict, String> {
        let (fused, fused_plan, fused_rewrites) = prepare(&self.fused, case)?;
        let (unfused, unfused_plan, _) = prepare(&self.unfused, case)?;
        // A plan holds kernel ids of the registry, which every interpreter builds alike, so a
        // plan from the scalar planner runs on each variant's interpreter. Every variant of a
        // program shares one plan in normal use too.
        let mut reference = Vec::new();
        for (plan, label) in [(&fused, "fused"), (&unfused, "unfused")] {
            let outcomes: Vec<Outcome> = self
                .runners
                .iter()
                .map(|(_, interp)| run(interp, plan, case))
                .collect();
            if let Some(i) = (1..outcomes.len()).find(|&i| outcomes[i] != outcomes[0]) {
                return Err(format!(
                    "oracle 1: [{}] differs from [{}] on the {label} plan\n{}\nscalar: {}\n[{}]: {}",
                    self.runners[i].0,
                    self.runners[0].0,
                    describe(case),
                    outcomes[0],
                    self.runners[i].0,
                    outcomes[i],
                ));
            }
            reference.push(outcomes.into_iter().next().expect("scalar runs"));
        }
        let (f, u) = (&reference[0], &reference[1]);
        if f.fusion_view() != u.fusion_view() {
            return Err(format!(
                "oracle 2: the unfused plan differs from the fused plan\n{}\nfused plan:\n{fused_plan}\
                 unfused plan:\n{unfused_plan}fused: {f}\nunfused: {u}",
                describe(case),
            ));
        }
        // The generator produces nothing unsupported, so a skip here would quietly turn the
        // oracle off.
        let unsupported = |reason: String| {
            format!(
                "oracle 3: the reference evaluator does not support the program: {reason}\n{}",
                describe(case),
            )
        };
        let r = eval::evaluate(&self.reg, &case.source, &case.inputs).map_err(unsupported)?;
        for (o, label, plan) in [(f, "fused", &fused_plan), (u, "unfused", &unfused_plan)] {
            if !agrees(&r, o) {
                return Err(format!(
                    "oracle 3: the reference evaluator differs from the {label} plan\n{}\n\
                     {label} plan:\n{plan}pipeline: {o}\nreference: {}",
                    describe(case),
                    show_reference(&r),
                ));
            }
        }
        let mut failing_unread = false;
        for name in &case.failing_unread {
            let fails = eval::fails_alone(&self.reg, &case.source, &case.inputs, name)
                .map_err(unsupported)?;
            failing_unread |= fails == Some(true);
        }
        Ok(Verdict {
            error: f.result.as_ref().err().map(|e| (e.kind, e.message.clone())),
            fusion_changed_plan: fused_plan != unfused_plan,
            rewrites_fired: rewrites::fired(&fused_rewrites),
            failing_unread,
        })
    }
}

fn export(name: &str, v: &Value) -> Export {
    Export {
        name: name.to_string(),
        ty: v.ty(),
        bits: bits(v),
        shown: v.to_string(),
    }
}

/// Oracle 3: whether the pipeline's outcome `o` is one the reference evaluator allows.
fn agrees(r: &Reference, o: &Outcome) -> bool {
    match (&r.result, &o.result) {
        (Ok(exports), Ok(got)) => {
            let want: Vec<Export> = exports.iter().map(|(n, v)| export(n, v)).collect();
            r.printed == o.printed && want == *got
        }
        (Err(failed), Err(f)) => {
            let category = Category::of(&f.message);
            f.kind == ErrorKind::Runtime
                && f.span.is_some_and(|s| {
                    failed
                        .errors
                        .iter()
                        .any(|e| e.category == category && e.lines.contains(&s.line))
                })
                && r.printed == o.printed
        }
        _ => false,
    }
}

fn show_reference(r: &Reference) -> String {
    let mut s = format!("printed: {:?}", r.printed);
    match &r.result {
        Ok(exports) => {
            for (name, v) in exports {
                s.push_str(&format!("\n  export {name}: {} = {v}", v.ty()));
            }
        }
        Err(failed) => {
            for e in &failed.errors {
                s.push_str(&format!(
                    "\n  possible error at line {}, chain {:?}: {}",
                    failed.line, e.lines, e.message
                ));
            }
        }
    }
    s
}

/// The program, its inputs, and a self-contained reproduction for a `run` case.
pub fn describe(case: &Case) -> String {
    let inputs: Vec<String> = case
        .inputs
        .iter()
        .map(|(n, v)| format!("  {n} = {v}"))
        .collect();
    format!(
        "program:\n{}inputs:\n{}\nreproduction:\n{}",
        case.source,
        inputs.join("\n"),
        case.repro
    )
}

/// The prepared program, its plan lines, and per statement the integer sum rewrites that
/// fired.
fn prepare(
    interp: &Interpreter,
    case: &Case,
) -> Result<(Prepared, String, Vec<BTreeSet<SumRewrite>>), String> {
    let mut plan = Vec::new();
    let (prepared, fired) = interp
        .prepare_with_sum_rewrites(&case.source, &mut plan)
        .map_err(|e| {
            format!(
                "generator bug, the program does not compile: {e}\n{}",
                describe(case)
            )
        })?;
    Ok((
        prepared,
        String::from_utf8(plan).expect("plans are text"),
        fired,
    ))
}

fn run(interp: &Interpreter, plan: &Prepared, case: &Case) -> Outcome {
    let inputs: Vec<(&str, Value)> = case
        .inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.clone()))
        .collect();
    let mut printed = Vec::new();
    let result = plan
        .execute(interp, &inputs, &mut printed)
        .map(|exports| {
            exports
                .into_iter()
                .map(|(name, v)| Export {
                    name,
                    ty: v.ty(),
                    bits: bits(&v),
                    shown: v.to_string(),
                })
                .collect()
        })
        .map_err(|e| Failure {
            kind: e.kind,
            message: e.message,
            span: e.span,
        });
    Outcome {
        printed: String::from_utf8(printed).expect("output is text"),
        result,
    }
}

#[test]
fn exports_compare_by_bits_and_nans_alike() {
    let f = |x: f64| bits(&Value::Scalar(Scalar::F64(x)));
    assert_ne!(f(0.0), f(-0.0), "signed zeros differ");
    assert_ne!(f(1.0), f(1.0 + f64::EPSILON), "neighbours differ");
    assert_eq!(f(f64::NAN), f(-f64::NAN), "NaN signs are alike");
    assert_eq!(
        f(f64::NAN),
        f(f64::from_bits(0x7ff8_0000_0000_0001)),
        "NaN payloads are alike"
    );
}

#[test]
fn error_categories_ignore_wording_only() {
    let c = Category::of;
    assert_eq!(c("length mismatch: 3 vs 2"), Category::LengthMismatch);
    assert_eq!(
        c("length mismatch: 3 values vs 2 mask rows"),
        Category::LengthMismatch
    );
    assert_eq!(
        c("gather index 7 out of range for length 5"),
        Category::GatherIndex
    );
    assert_eq!(
        c("slice 1..0 out of range for length 0"),
        Category::SliceBounds
    );
    assert_eq!(
        c("bucket key -1 out of range for 4 buckets"),
        Category::BucketKey
    );
    assert_eq!(c("div_const: division by zero"), Category::DivisionByZero);
    assert_ne!(
        c("reduce_max of an empty array"),
        c("argmax of an empty array")
    );
    assert_ne!(
        c("gather index 7 out of range for length 5"),
        c("reduce_min of an empty array")
    );
    assert_eq!(c("planner: x"), Category::Other("planner: x".into()));
}

/// A let read twice is computed at its own statement, so the pipeline reports the let's line,
/// which is on the chain from the reading print, and oracle 3 accepts it. A line off the
/// chain or another category is rejected.
#[test]
fn oracle_3_accepts_lines_on_the_chain() {
    let case = Case {
        source: "let e = fill(0, 1.0);\n\
                 let v = reduce(e, min);\n\
                 print(1);\n\
                 print(v + v);\n"
            .to_string(),
        inputs: Vec::new(),
        repro: String::new(),
        features: Default::default(),
        failing_unread: Vec::new(),
    };
    let engine = Engine::new();
    engine.check(&case).expect("the oracles agree");
    let r = eval::evaluate(&engine.reg, &case.source, &[]).expect("supported");
    let failed = r.result.as_ref().expect_err("the program fails");
    assert_eq!(failed.line, 4);
    let lines: Vec<u32> = failed.errors[0].lines.iter().copied().collect();
    assert_eq!(
        lines,
        [2, 4],
        "the failing let and the print, not the let it reads"
    );
    let at = |line, message: &str| Outcome {
        printed: "1\n".to_string(),
        result: Err(Failure {
            kind: ErrorKind::Runtime,
            message: message.to_string(),
            span: Some(Span { line, col: 1 }),
        }),
    };
    assert!(agrees(&r, &at(2, "reduce_min of an empty array")));
    assert!(agrees(&r, &at(4, "reduce_min of an empty array")));
    assert!(!agrees(&r, &at(1, "reduce_min of an empty array")));
    assert!(!agrees(&r, &at(3, "reduce_min of an empty array")));
    assert!(!agrees(&r, &at(2, "length mismatch: 1 vs 2")));
}

/// Two failing lets read by one print. The fused plan `count_gt_f64 x c` evaluates `x` first
/// and reports line 2, the unfused plan `count (lt_f64 c x)` evaluates `c` first and reports
/// line 1. Both lines are on the reference's chains, so the oracles accept both plans.
#[test]
fn plans_may_report_different_failing_lets() {
    let case = Case {
        source: "let c = reduce(gather([1.0, 2.0], fill(2, 9)), max);\n\
                 let x = gather([1.0, 2.0], fill(2, 5));\n\
                 print(count(c < x));\n\
                 print(c);\n\
                 print(x);\n"
            .to_string(),
        inputs: Vec::new(),
        repro: String::new(),
        features: Default::default(),
        failing_unread: Vec::new(),
    };
    let engine = Engine::new();
    engine.check(&case).expect("the oracles agree");
    let line = |interp: &Interpreter| {
        let (prepared, _) = prepare(interp, &case).expect("compiles");
        let outcome = run(interp, &prepared, &case);
        let failure = outcome.result.expect_err("the program fails");
        assert_eq!(Category::of(&failure.message), Category::GatherIndex);
        failure.span.map(|s| s.line)
    };
    assert_eq!(line(&engine.fused), Some(2));
    assert_eq!(line(&engine.unfused), Some(1));
}

/// Two failing lets of different categories read by one print. The fused plan evaluates `x`
/// first and reports its out-of-range gather, the unfused plan evaluates `c` first and reports
/// its empty reduction. Both are errors the reference allows, so the oracles accept both plans.
#[test]
fn plans_may_report_different_categories() {
    let case = Case {
        source: "let c = reduce(fill(0, 1.0), min);\n\
                 let x = gather([1.0, 2.0], fill(2, 5));\n\
                 print(count(c < x));\n\
                 print(c);\n\
                 print(x);\n"
            .to_string(),
        inputs: Vec::new(),
        repro: String::new(),
        features: Default::default(),
        failing_unread: Vec::new(),
    };
    let engine = Engine::new();
    engine.check(&case).expect("the oracles agree");
    let failure = |interp: &Interpreter| {
        let (prepared, _) = prepare(interp, &case).expect("compiles");
        let outcome = run(interp, &prepared, &case);
        let failure = outcome.result.expect_err("the program fails");
        (failure.span.map(|s| s.line), Category::of(&failure.message))
    };
    assert_eq!(failure(&engine.fused), (Some(2), Category::GatherIndex));
    assert_eq!(
        failure(&engine.unfused),
        (Some(1), Category::Empty("reduce_min".into()))
    );
}

/// A failing binding that nothing reads fails when forced alone, also as a block let, while
/// the program succeeds.
#[test]
fn unread_failing_bindings_fail_alone() {
    let reg = Registry::new();
    let src = "let x = [1, 2];\n\
               let _v = gather(x, fill(1, length(x)));\n\
               print({ let _b = x == fill(length(x) + 1, 0); x });\n";
    let r = eval::evaluate(&reg, src, &[]).expect("supported");
    assert!(r.result.is_ok(), "nothing demands the failing bindings");
    let fails = |name| eval::fails_alone(&reg, src, &[], name).expect("supported");
    assert_eq!(fails("_v"), Some(true));
    assert_eq!(fails("_b"), Some(true));
    assert_eq!(fails("x"), Some(false));
    assert_eq!(fails("_c"), None);
}
