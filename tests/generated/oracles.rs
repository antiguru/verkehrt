//! The oracles. Each program is planned twice on the scalar interpreter, with fused patterns
//! and without, and each plan runs on every variant the CPU supports.
//!
//! * Oracle 1, cross-variant: every variant gives the scalar outcome of the same plan. The
//!   variants share the plan, so the outcomes must agree exactly, including the error message
//!   and its location.
//! * Oracle 2, fused against unfused: the unfused plan gives the fused plan's output, exports,
//!   and error kind. The message of an error may legitimately differ, because a fused kernel
//!   reports a length mismatch in its own words, and the base spec lets runtime errors
//!   surface in any order. So only whether an error occurs and its kind are compared.
//!
//! Exports compare by bits, except that every NaN equals every NaN, because the base spec
//! lets the sign and payload of a NaN result differ between variants. Printed output is
//! compared as text, which writes every NaN as `NaN`.

use std::fmt;

use verkehrt::error::{ErrorKind, Span};
use verkehrt::kernels::{Backend, Isa, Variant};
use verkehrt::value::{Scalar, Value, ValueType};
use verkehrt::{Interpreter, Options, Prepared};

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

impl Outcome {
    /// The parts that oracle 2 compares: output, exports, and the error kind.
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

/// The interpreters for planning and for running plans.
pub struct Engine {
    fused: Interpreter,
    unfused: Interpreter,
    runners: Vec<(Variant, Interpreter)>,
}

/// What a checked case did, for the statistics.
pub struct Verdict {
    /// The error that stopped the program, with its message.
    pub error: Option<(ErrorKind, String)>,
    /// Whether the fused plan differs from the unfused one in some statement.
    pub fusion_changed_plan: bool,
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
        }
    }

    pub fn variant_count(&self) -> usize {
        self.runners.len()
    }

    /// Plans `case` with and without fusion, runs both plans on every variant, and applies
    /// oracles 1 and 2. A program that does not compile is a generator bug.
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
        Ok(Verdict {
            error: f.result.as_ref().err().map(|e| (e.kind, e.message.clone())),
            fusion_changed_plan: fused_plan != unfused_plan,
        })
    }
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
