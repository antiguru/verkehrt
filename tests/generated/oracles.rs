//! The oracles. Each program is planned twice on the scalar interpreter, with fused patterns
//! and without, and each plan runs on every variant the CPU supports.
//!
//! * Oracle 1, cross-variant: every variant gives the scalar outcome of the same plan. The
//!   variants share the plan, so the outcomes must agree exactly, including the error message
//!   and its location.
//! * Oracle 2, fused against unfused: the unfused plan gives the fused plan's output, exports,
//!   and error. An error must have the same kind, come from the same statement (each statement
//!   sits on its own line, so the line of its span), and fall in the same category, see
//!   `Category`. The message text is not compared, because a fused kernel words a length
//!   mismatch its own way.
//! * Oracle 3, reference evaluator: the scalar outcome of the fused plan equals what the
//!   evaluator of `eval` computes from the typed program. On success the output and the
//!   exports must be equal. Whether an error occurs and which statement raises it are fixed,
//!   so on failure the error must be a runtime error at the line of the evaluator's failing
//!   statement, in one of the categories the evaluator met there. Which of several errors
//!   within that statement surfaces may differ, and output printed before an error may be
//!   incomplete, so the pipeline's output must only be a prefix of the evaluator's.
//!
//! Exports compare by bits, except that every NaN equals every NaN, because the base spec
//! lets the sign and payload of a NaN result differ between variants. Printed output is
//! compared as text, which writes every NaN as `NaN`.

use std::fmt;

use verkehrt::error::{ErrorKind, Span};
use verkehrt::kernels::{Backend, Isa, Registry, Variant};
use verkehrt::value::{Scalar, Value, ValueType};
use verkehrt::{Interpreter, Options, Prepared};

use crate::eval::{self, Reference};
use crate::r#gen::Case;

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

/// The parts of an error that oracle 2 compares.
#[derive(Debug, PartialEq, Eq)]
struct FusionError {
    kind: ErrorKind,
    line: Option<u32>,
    category: Category,
}

impl Outcome {
    /// The parts that oracle 2 compares: output, exports, and the error's kind, statement,
    /// and category.
    fn fusion_view(&self) -> (&str, Result<&[Export], FusionError>) {
        (
            &self.printed,
            self.result
                .as_ref()
                .map(Vec::as_slice)
                .map_err(|f| FusionError {
                    kind: f.kind,
                    line: f.span.map(|s| s.line),
                    category: Category::of(&f.message),
                }),
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
}

/// What a checked case did, for the statistics.
pub struct Verdict {
    /// The error that stopped the program, with its message.
    pub error: Option<(ErrorKind, String)>,
    /// Whether the fused plan differs from the unfused one in some statement.
    pub fusion_changed_plan: bool,
    /// Why oracle 3 skipped the case, if it did.
    pub reference_skipped: Option<String>,
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
        Engine {
            fused: planner(false),
            unfused: planner(true),
            runners,
            reg: Registry::new(),
        }
    }

    pub fn variant_count(&self) -> usize {
        self.runners.len()
    }

    /// Plans `case` with and without fusion, runs both plans on every variant, evaluates it
    /// with the reference evaluator, and applies the three oracles. A program that does not
    /// compile is a generator bug.
    pub fn check(&self, case: &Case) -> Result<Verdict, String> {
        let (fused, fused_plan) = prepare(&self.fused, case)?;
        let (unfused, unfused_plan) = prepare(&self.unfused, case)?;
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
        let evaluate = |eager| eval::evaluate(&self.reg, &case.source, &case.inputs, eager);
        let reference_skipped = match evaluate(false) {
            Err(reason) => Some(reason),
            Ok(r) if agrees(&r, f) => None,
            // Until #40 (errors as data), the pipeline computes top-level lets it does not
            // inline. Such a case is skipped only when eager evaluation of the top-level lets
            // reproduces the pipeline's failure exactly.
            Ok(r) if eager_reproduces(&r, f, &evaluate(true)) => Some(
                "until #40, the pipeline computes a top-level let it does not inline".to_string(),
            ),
            Ok(r) => {
                return Err(format!(
                    "oracle 3: the reference evaluator differs from the pipeline\n{}\n\
                     fused plan:\n{fused_plan}pipeline: {f}\nreference: {}",
                    describe(case),
                    show_reference(&r),
                ));
            }
        };
        Ok(Verdict {
            error: f.result.as_ref().err().map(|e| (e.kind, e.message.clone())),
            fusion_changed_plan: fused_plan != unfused_plan,
            reference_skipped,
        })
    }
}

/// Whether the pipeline's failure `o` is the one eager evaluation of top-level lets gives, at
/// the same line and with the same output, while the lazy evaluation `lazy` succeeds or fails
/// only at a later statement.
fn eager_reproduces(lazy: &Reference, o: &Outcome, eager: &Result<Reference, String>) -> bool {
    let (Err(f), Ok(eager)) = (&o.result, eager) else {
        return false;
    };
    let Err(failed) = &eager.result else {
        return false;
    };
    // The two evaluations share every statement before the lazy failure except the lets, so
    // an eager failure before it comes from a let that the lazy evaluation reads later.
    if lazy.result.as_ref().is_err_and(|l| l.line <= failed.line) {
        return false;
    }
    let category = Category::of(&f.message);
    f.kind == ErrorKind::Runtime
        && f.span.map(|s| s.line) == Some(failed.line)
        && failed.errors.iter().any(|(c, _)| *c == category)
        && eager.printed == o.printed
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
        // Every printed line ends with a newline, so a prefix of the text is one of lines.
        (Err(failed), Err(f)) => {
            let category = Category::of(&f.message);
            f.kind == ErrorKind::Runtime
                && f.span.map(|s| s.line) == Some(failed.line)
                && failed.errors.iter().any(|(c, _)| *c == category)
                && r.printed.starts_with(&o.printed)
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
            for (_, message) in &failed.errors {
                s.push_str(&format!(
                    "\n  possible error at line {}: {message}",
                    failed.line
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

fn prepare(interp: &Interpreter, case: &Case) -> Result<(Prepared, String), String> {
    let mut plan = Vec::new();
    let prepared = interp.prepare_to(&case.source, &mut plan).map_err(|e| {
        format!(
            "generator bug, the program does not compile: {e}\n{}",
            describe(case)
        )
    })?;
    Ok((prepared, String::from_utf8(plan).expect("plans are text")))
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
