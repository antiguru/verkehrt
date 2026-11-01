//! The program generator. It builds a program top-down from typed expressions, choosing
//! only operations whose kernels the registry resolves on the operand types, so every program
//! type-checks. Inside lambdas it also tracks the lifted type of each expression, a scalar
//! when the expression is uniform and an array when it depends on a parameter, and resolves
//! each operation on those types the way the checker's lifting does.
//!
//! Every decision reads the next number from a choice sequence, and a missing number reads as
//! 0, which always selects the simplest option. The inputs, each function, and each statement
//! read their own sequence (see `Plan`). Proptest shrinks by dropping whole sequences, which
//! drops a function or statement, and by dropping and lowering numbers, which simplifies one.
//!
//! Generation covers the element types that array kernels exist for. The scalar-only types
//! (`i8`, `i16`, `u16`, `u32`) are left out, because a lambda over them cannot lift.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use verkehrt::kernels::{InputSpec, KernelClass, Registry};
use verkehrt::ops::{Op, ReduceOp};
use verkehrt::value::{Array, Scalar, ScalarType, Value, ValueType};

/// Size limits of generated programs.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Nesting depth of expressions, beyond which only leaves are generated.
    pub max_depth: usize,
    /// The most top-level statements after the inputs.
    pub max_stmts: usize,
    /// The most user functions.
    pub max_fns: usize,
    /// The longest input array.
    pub max_len: usize,
}

/// A generated program with the values of its inputs.
pub struct Case {
    pub source: String,
    pub inputs: Vec<(String, Value)>,
    /// The program with every input replaced by a `let` of a literal, for a `run` case.
    pub repro: String,
    /// The features the program uses, for the histogram.
    pub features: BTreeSet<&'static str>,
}

/// The choices of a program: one sequence for the inputs, one per function, and one per
/// statement. A function or statement reads only its own sequence, so shrinking can drop one
/// without changing the others.
#[derive(Clone, Debug)]
pub struct Plan {
    pub inputs: Vec<u32>,
    pub fns: Vec<Vec<u32>>,
    pub stmts: Vec<Vec<u32>>,
}

/// A sequence of choices. A read past the end is 0.
pub struct Choices<'a> {
    data: &'a [u32],
    pos: usize,
}

impl<'a> Choices<'a> {
    pub fn new(data: &'a [u32]) -> Choices<'a> {
        Choices { data, pos: 0 }
    }

    fn next(&mut self) -> u32 {
        let v = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        v
    }

    /// A number in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "a choice needs an option");
        usize::try_from(self.next()).expect("u32 fits in usize") % n
    }

    /// True with `percent` probability, false on a 0 choice.
    fn flip(&mut self, percent: usize) -> bool {
        self.below(100) >= 100 - percent
    }

    /// An index into `weights`, by weight. A 0 choice picks the first option of nonzero weight.
    fn weighted(&mut self, weights: &[usize]) -> usize {
        let total: usize = weights.iter().sum();
        let mut v = self.below(total);
        for (i, w) in weights.iter().enumerate() {
            if v < *w {
                return i;
            }
            v -= w;
        }
        unreachable!("the value is below the total weight")
    }

    fn pick<'b, T>(&mut self, xs: &'b [T]) -> &'b T {
        &xs[self.below(xs.len())]
    }
}

const TYPES: [ScalarType; 7] = [
    ScalarType::F64,
    ScalarType::I64,
    ScalarType::Bool,
    ScalarType::F32,
    ScalarType::I32,
    ScalarType::U8,
    ScalarType::U64,
];

const INTS: [ScalarType; 4] = [
    ScalarType::I64,
    ScalarType::I32,
    ScalarType::U8,
    ScalarType::U64,
];

const CMPS: [Op; 6] = [Op::Lt, Op::Gt, Op::Le, Op::Ge, Op::Eq, Op::Ne];

/// The deepest nesting of bindings read more than once, see `Gen::share_depth`.
const MAX_SHARE_DEPTH: usize = 3;

/// The length of an array, as far as generation knows it. Arrays of one class have equal
/// lengths at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Len {
    Known(usize),
    Class(u32),
}

/// A typed expression.
#[derive(Clone, Debug)]
struct E {
    k: K,
    ty: ValueType,
    /// The lifted type inside a lambda, the type itself elsewhere.
    lt: ValueType,
    /// The length class of an array.
    len: Option<Len>,
}

#[derive(Clone, Debug)]
enum K {
    Lit(String),
    /// A variable, and whether a lambda body captures it from outside.
    Var(String, bool),
    ArrayLit(Vec<E>),
    /// A primitive, and for `Reduce(Add)` whether it is written as `sum`.
    Op(Op, Vec<E>, bool),
    Call(usize, Vec<E>),
    If(Box<E>, Box<E>, Box<E>),
    Block(Vec<(String, E)>, Box<E>),
    Higher(Higher, Vec<E>, Lam),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Higher {
    Map,
    ZipMap,
    Filter,
}

#[derive(Clone, Debug)]
enum Lam {
    Lambda(Vec<String>, Box<E>),
    Fn(usize),
}

/// The length of a function's array result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetLen {
    Scalar,
    /// The length class of the array parameters.
    Params,
    Known(usize),
    Fresh,
}

/// Keys for `histogram` and `reduce_into`, see `Gen::keys`.
struct Keys {
    keys: Option<E>,
    n: Option<E>,
    out: Len,
    binding: Option<(String, E)>,
}

struct FnDef {
    name: String,
    params: Vec<(String, ValueType)>,
    ret: ValueType,
    body: E,
    /// Scalar parameters and an elementwise body, so lambdas may call it.
    liftable: bool,
    ret_len: RetLen,
}

#[derive(Clone, Debug)]
struct Var {
    name: String,
    ty: ValueType,
    lt: ValueType,
    len: Option<Len>,
    /// Declared inside the lambda being generated, so not a capture.
    local: bool,
}

struct Ctx {
    vars: Vec<Var>,
    lambda: bool,
    depth: usize,
    /// Functions below this index may be called.
    fns: usize,
}

/// A kernel signature the generator may use.
#[derive(Clone, Debug)]
struct Sig {
    op: Op,
    inputs: Vec<InputSpec>,
    out: ValueType,
    class: KernelClass,
}

fn elem(t: ValueType) -> ScalarType {
    match t {
        ValueType::Scalar(s) | ValueType::Array(s) => s,
        ValueType::Str => unreachable!("no strings are generated"),
    }
}

fn is_array(t: ValueType) -> bool {
    matches!(t, ValueType::Array(_))
}

fn spec_elem(s: InputSpec) -> Option<ScalarType> {
    match s {
        InputSpec::Array(t) | InputSpec::Scalar(t) | InputSpec::Either(t) => Some(t),
        InputSpec::Str => None,
    }
}

/// Ops that apply per element, the only ones a lambda body may use.
fn is_elementwise(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Neg
            | Op::Abs
            | Op::Min
            | Op::Max
            | Op::Sqrt
            | Op::Eq
            | Op::Ne
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Not
            | Op::Select
            | Op::Cast(_)
    )
}

fn is_reduction(op: Op) -> bool {
    matches!(
        op,
        Op::Reduce(_) | Op::Count | Op::Any | Op::All | Op::ArgMin | Op::ArgMax | Op::Length
    )
}

fn reduce_name(r: ReduceOp) -> &'static str {
    match r {
        ReduceOp::Add => "add",
        ReduceOp::Mul => "mul",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::And => "and",
        ReduceOp::Or => "or",
        ReduceOp::Xor => "xor",
    }
}

fn infix(op: Op) -> Option<&'static str> {
    Some(match op {
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
        Op::Div => "/",
        Op::Eq => "==",
        Op::Ne => "!=",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
        _ => return None,
    })
}

fn call_name(op: Op) -> String {
    match op {
        Op::Neg => "neg".into(),
        Op::Abs => "abs".into(),
        Op::Min => "min".into(),
        Op::Max => "max".into(),
        Op::Sqrt => "sqrt".into(),
        Op::And => "and".into(),
        Op::Or => "or".into(),
        Op::Xor => "xor".into(),
        Op::Not => "not".into(),
        Op::Select => "select".into(),
        Op::Cast(t) => format!("cast_{}", t.name()),
        Op::Reduce(_) => "reduce".into(),
        Op::Scan(_) => "scan".into(),
        Op::ScanBy(_) => "scan_by".into(),
        Op::ReduceInto(_) => "reduce_into".into(),
        Op::Count => "count".into(),
        Op::Any => "any".into(),
        Op::All => "all".into(),
        Op::ArgMin => "argmin".into(),
        Op::ArgMax => "argmax".into(),
        Op::Compress => "compress".into(),
        Op::Where => "where".into(),
        Op::Prev => "prev".into(),
        Op::Iota => "iota".into(),
        Op::Fill => "fill".into(),
        Op::Gather => "gather".into(),
        Op::Slice => "slice".into(),
        Op::Histogram => "histogram".into(),
        Op::DivConst => "div_const".into(),
        Op::ArgSort => "argsort".into(),
        Op::GroupId => "group_id".into(),
        Op::RandI64 => "rand_i64".into(),
        Op::RandF64 => "rand_f64".into(),
        Op::Length => "length".into(),
        other => unreachable!("{other:?} is not generated"),
    }
}

/// The feature name of an op in the histogram.
fn feature(op: Op) -> &'static str {
    match op {
        Op::Add => "op +",
        Op::Sub => "op -",
        Op::Mul => "op *",
        Op::Div => "op /",
        Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => "op comparison",
        Op::Neg => "op neg",
        Op::Abs => "op abs",
        Op::Min | Op::Max => "op min/max",
        Op::Sqrt => "op sqrt",
        Op::And | Op::Or | Op::Xor | Op::Not => "op and/or/xor/not",
        Op::Select => "op select",
        Op::Cast(_) => "op cast",
        Op::Reduce(ReduceOp::Add) => "reduce add",
        Op::Reduce(ReduceOp::Mul) => "reduce mul",
        Op::Reduce(ReduceOp::Min | ReduceOp::Max) => "reduce min/max",
        Op::Reduce(_) => "reduce and/or/xor",
        Op::Scan(_) => "scan",
        Op::ScanBy(_) => "scan_by",
        Op::ReduceInto(_) => "reduce_into",
        Op::Count => "count",
        Op::Any | Op::All => "any/all",
        Op::ArgMin | Op::ArgMax => "argmin/argmax",
        Op::Compress => "compress",
        Op::Where => "where",
        Op::Prev => "prev",
        Op::Iota => "iota",
        Op::Fill => "fill",
        Op::Gather => "gather",
        Op::Slice => "slice",
        Op::Histogram => "histogram",
        Op::DivConst => "div_const",
        Op::ArgSort => "argsort",
        Op::GroupId => "group_id",
        Op::RandI64 | Op::RandF64 => "rand",
        Op::Length => "length",
        _ => "op other",
    }
}

fn type_name(t: ValueType) -> String {
    t.to_string()
}

/// Literal texts per type, the simplest first. Negative literals are parenthesized.
fn literals(t: ScalarType) -> &'static [&'static str] {
    match t {
        ScalarType::F64 => &[
            "0.0",
            "1.0",
            "2.5",
            "(-1.0)",
            "0.5",
            "(-0.0)",
            "3.0",
            "100.0",
            "(-7.25)",
            "0.1",
            "1000000.0",
        ],
        ScalarType::F32 => &[
            "0.0f32",
            "1.0f32",
            "2.5f32",
            "(-1.0f32)",
            "0.5f32",
            "(-0.0f32)",
            "3.0f32",
            "100.0f32",
            "(-7.25f32)",
            "0.1f32",
        ],
        ScalarType::I64 => &[
            "0",
            "1",
            "2",
            "(-1)",
            "3",
            "7",
            "(-5)",
            "100",
            "9223372036854775807",
            "(-9223372036854775808)",
        ],
        ScalarType::I32 => &[
            "0i32",
            "1i32",
            "2i32",
            "(-1i32)",
            "3i32",
            "7i32",
            "(-5i32)",
            "100i32",
            "2147483647i32",
            "(-2147483648i32)",
        ],
        ScalarType::U8 => &["0u8", "1u8", "2u8", "3u8", "7u8", "100u8", "255u8"],
        ScalarType::U64 => &[
            "0u64",
            "1u64",
            "2u64",
            "3u64",
            "7u64",
            "100u64",
            "18446744073709551615u64",
        ],
        ScalarType::Bool => &["false", "true"],
        other => unreachable!("{other:?} is not generated"),
    }
}

/// The generator for one program.
struct Gen<'a, 'r> {
    c: Choices<'a>,
    reg: &'r Registry,
    p: Params,
    sigs: &'r [Sig],
    fns: Vec<FnDef>,
    names: usize,
    classes: u32,
}

/// The kernel signatures the generator draws from: single-op kernels on generated types,
/// without `load_*` and the internal `SameLen`.
pub struct Catalog {
    reg: Registry,
    sigs: Vec<Sig>,
}

impl Catalog {
    pub fn new() -> Catalog {
        let reg = Registry::new();
        let sigs = reg
            .iter()
            .filter_map(|(_, d)| {
                let op = d.op?;
                if d.pattern.is_some() || matches!(op, Op::Load(_) | Op::SameLen) {
                    return None;
                }
                let types_ok = d
                    .inputs
                    .iter()
                    .all(|s| spec_elem(*s).is_some_and(|t| TYPES.contains(&t)))
                    && TYPES.contains(&elem(d.output));
                types_ok.then(|| Sig {
                    op,
                    inputs: d.inputs.clone(),
                    out: d.output,
                    class: d.class,
                })
            })
            .collect();
        Catalog { reg, sigs }
    }
}

/// Generates a program from `choices`.
pub fn generate(cat: &Catalog, plan: &Plan, p: Params) -> Case {
    let mut g = Gen {
        c: Choices::new(&plan.inputs),
        reg: &cat.reg,
        p,
        sigs: &cat.sigs,
        fns: Vec::new(),
        names: 0,
        classes: 0,
    };
    g.program(plan)
}

impl<'a> Gen<'a, '_> {
    fn fresh(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}{}", self.names - 1)
    }

    fn fresh_class(&mut self) -> Len {
        self.classes += 1;
        Len::Class(self.classes - 1)
    }

    fn ty(&mut self) -> ScalarType {
        *self.c.pick(&TYPES)
    }

    fn program(&mut self, plan: &'a Plan) -> Case {
        let mut head = String::new();
        for choices in plan.fns.iter().take(self.p.max_fns) {
            self.c = Choices::new(choices);
            let f = self.function();
            head.push_str(&self.render_fn(&f));
            self.fns.push(f);
        }

        // Inputs share one length, apart from an occasional other one.
        self.c = Choices::new(&plan.inputs);
        let main_len = self.input_len();
        let mut cx = Ctx {
            vars: Vec::new(),
            lambda: false,
            depth: 0,
            fns: self.fns.len(),
        };
        let mut decls = String::new();
        let mut lets = String::new();
        let mut inputs = Vec::new();
        let n_arrays = 1 + self.c.below(4);
        for _ in 0..n_arrays {
            let t = self.ty();
            let len = if self.c.flip(15) {
                self.input_len()
            } else {
                main_len
            };
            let name = self.fresh("a");
            let value = self.input_value(t, len, true);
            decls.push_str(&format!("input {name}: {}[];\n", t.name()));
            lets.push_str(&format!("let {name} = {};\n", literal_of(&value)));
            cx.vars.push(Var {
                name: name.clone(),
                ty: ValueType::Array(t),
                lt: ValueType::Array(t),
                len: Some(Len::Known(len)),
                local: false,
            });
            inputs.push((name, value));
        }
        let n_scalars = self.c.weighted(&[3, 2, 1]);
        for _ in 0..n_scalars {
            let t = self.ty();
            let name = self.fresh("s");
            let value = self.input_value(t, 1, false);
            decls.push_str(&format!("input {name}: {};\n", t.name()));
            lets.push_str(&format!("let {name} = {};\n", literal_of(&value)));
            cx.vars.push(Var {
                name: name.clone(),
                ty: ValueType::Scalar(t),
                lt: ValueType::Scalar(t),
                len: None,
                local: false,
            });
            inputs.push((name, value));
        }

        let mut body = String::new();
        let mut stmts: Vec<(&'static str, String, E)> = Vec::new();
        let mut too_large = 0;
        let n_stmts = plan.stmts.len().clamp(1, self.p.max_stmts);
        for i in 0..n_stmts {
            self.c = Choices::new(plan.stmts.get(i).map_or(&[], Vec::as_slice));
            let kind = if i + 1 == n_stmts {
                // The last statement has an observable result.
                ["print", "export"][self.c.below(2)]
            } else {
                ["let", "print", "export"][self.c.weighted(&[5, 2, 2])]
            };
            // A statement often has the type of a variable, which a leaf then reads.
            let want = if !cx.vars.is_empty() && self.c.flip(50) {
                self.c.pick(&cx.vars).ty
            } else {
                self.want_type()
            };
            let mut e = self.expr(&mut cx, want, None);
            // Lowering inlines function calls and block lets by deep-cloning substitution, so
            // bindings read more than once that nest deeply grow the term exponentially, to
            // gigabytes at seven levels. Such a statement is replaced by a leaf.
            // See #39, a term graph for lowering.
            if self.share_depth(&e, &HashMap::new()) > MAX_SHARE_DEPTH {
                too_large += 1;
                e = self.leaf(&cx, want, None);
            }
            match kind {
                "let" => {
                    let name = self.fresh("v");
                    body.push_str(&format!("let {name} = {};\n", self.render(&e)));
                    cx.vars.push(Var {
                        name: name.clone(),
                        ty: e.ty,
                        lt: e.ty,
                        len: e.len,
                        local: false,
                    });
                    stmts.push(("let", name, e));
                }
                "print" => {
                    body.push_str(&format!("print({});\n", self.render(&e)));
                    stmts.push(("print", String::new(), e));
                }
                _ => {
                    let name = self.fresh("x");
                    body.push_str(&format!("export {name} = {};\n", self.render(&e)));
                    stmts.push(("export", name, e));
                }
            }
        }

        let (mut features, uses) = self.features(&stmts);
        if too_large > 0 {
            features.insert("statement over the sharing depth cap");
        }
        // Only what reaches a print or an export runs, so a let that nothing reads is
        // exported, which keeps the lets that are read once single-use.
        for (kind, name, _) in &stmts {
            if *kind == "let" && !uses.contains_key(name) {
                let x = self.fresh("x");
                body.push_str(&format!("export {x} = {name};\n"));
            }
        }
        Case {
            source: format!("{head}{decls}{body}"),
            repro: format!("{head}{lets}{body}"),
            inputs,
            features,
        }
    }

    fn want_type(&mut self) -> ValueType {
        let t = self.ty();
        if self.c.flip(50) {
            ValueType::Scalar(t)
        } else {
            ValueType::Array(t)
        }
    }

    fn input_len(&mut self) -> usize {
        let len = match self.c.weighted(&[30, 30, 25, 5, 10]) {
            0 => 1 + self.c.below(8),
            1 => 9 + self.c.below(32),
            2 => 41 + self.c.below(100),
            3 => 0,
            _ => 141 + self.c.below(160),
        };
        len.min(self.p.max_len)
    }

    /// The value of an input of element type `t`, an array of `len` or a scalar.
    fn input_value(&mut self, t: ScalarType, len: usize, array: bool) -> Value {
        let style = self.c.below(5);
        let mut rng = SplitMix(u64::from(self.c.next()));
        let n = if array { len } else { 1 };
        let mut bits: Vec<u64> = (0..n).map(|_| element(t, style, &mut rng)).collect();
        // Sorted keys run in segments, as group keys do.
        if style == 4 && t != ScalarType::Bool {
            bits.sort_unstable();
        }
        if !array {
            return Value::Scalar(scalar_of(t, bits[0]));
        }
        let arr = match t {
            ScalarType::F64 => Array::from_vec(bits.drain(..).map(f64::from_bits).collect()),
            ScalarType::F32 => Array::from_vec(
                bits.drain(..)
                    .map(|b| f32::from_bits(u32::try_from(b).expect("f32 bits")))
                    .collect(),
            ),
            ScalarType::I64 => Array::from_vec(bits.drain(..).map(u64::cast_signed).collect()),
            ScalarType::I32 => Array::from_vec(
                bits.drain(..)
                    .map(|b| u32::try_from(b).expect("i32 bits").cast_signed())
                    .collect(),
            ),
            ScalarType::U8 => Array::from_vec(
                bits.drain(..)
                    .map(|b| u8::try_from(b).expect("u8 bits"))
                    .collect::<Vec<u8>>(),
            ),
            ScalarType::U64 => Array::from_vec(bits),
            ScalarType::Bool => {
                Array::from_bools(&bits.iter().map(|b| *b != 0).collect::<Vec<_>>())
            }
            other => unreachable!("{other:?} is not generated"),
        };
        Value::Array(arr)
    }

    fn function(&mut self) -> FnDef {
        let name = self.fresh("f");
        let liftable = self.c.flip(60);
        let n_params = 1 + self.c.below(3);
        let class = self.fresh_class();
        let mut params = Vec::new();
        let mut vars = Vec::new();
        for i in 0..n_params {
            let t = self.ty();
            // An array function has an array first parameter.
            let ty = if !liftable && (i == 0 || self.c.flip(40)) {
                ValueType::Array(t)
            } else {
                ValueType::Scalar(t)
            };
            let pname = self.fresh("q");
            let len = is_array(ty).then_some(class);
            vars.push(Var {
                name: pname.clone(),
                ty,
                // A liftable body is generated as if every parameter varied.
                lt: if liftable { ValueType::Array(t) } else { ty },
                len,
                local: true,
            });
            params.push((pname, ty));
        }
        let ret = if liftable {
            ValueType::Scalar(self.ty())
        } else {
            self.want_type()
        };
        let mut cx = Ctx {
            vars,
            lambda: liftable,
            depth: 0,
            fns: self.fns.len(),
        };
        let body = self.expr(&mut cx, ret, None);
        let ret_len = match body.len {
            None => RetLen::Scalar,
            Some(l) if l == class => RetLen::Params,
            Some(Len::Known(n)) => RetLen::Known(n),
            Some(Len::Class(_)) => RetLen::Fresh,
        };
        FnDef {
            name,
            params,
            ret,
            body,
            liftable,
            ret_len,
        }
    }

    /// An expression of type `want`, with length class `len` when given.
    fn expr(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> E {
        if cx.depth < self.p.max_depth {
            cx.depth += 1;
            let e = if cx.lambda {
                self.lambda_expr(cx, elem(want))
            } else {
                match want {
                    ValueType::Scalar(t) => self.scalar_expr(cx, t),
                    ValueType::Array(t) => self.array_expr(cx, t, len),
                    ValueType::Str => unreachable!("no strings are generated"),
                }
            };
            cx.depth -= 1;
            if let Some(e) = e {
                debug_assert_eq!(e.ty, want, "generated {e:?}");
                debug_assert!(len.is_none() || e.len == len, "length of {e:?}");
                return e;
            }
        }
        self.leaf(cx, want, len)
    }

    fn lambda_expr(&mut self, cx: &mut Ctx, t: ScalarType) -> Option<E> {
        match self.c.weighted(&[3, 6, 1, 2, 1, 1]) {
            0 => None,
            1 => self.generic(cx, ValueType::Scalar(t), None, is_elementwise),
            2 => self.cond(cx, ValueType::Scalar(t), None),
            3 => self.call_fn(cx, ValueType::Scalar(t), None),
            4 => self.block(cx, ValueType::Scalar(t), None),
            _ => self.div_const(cx, ValueType::Scalar(t), None),
        }
    }

    fn scalar_expr(&mut self, cx: &mut Ctx, t: ScalarType) -> Option<E> {
        let want = ValueType::Scalar(t);
        match self.c.weighted(&[3, 4, 5, 3, 1, 3, 1, 1]) {
            0 => None,
            1 => self.generic(cx, want, None, is_elementwise),
            2 => self.generic(cx, want, None, is_reduction),
            3 => self.fused_scalar(cx, t),
            4 => self.cond(cx, want, None),
            5 => self.call_fn(cx, want, None),
            6 => self.block(cx, want, None),
            _ => self.div_const(cx, want, None),
        }
    }

    fn array_expr(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let want = ValueType::Array(t);
        match self.c.weighted(&[3, 5, 4, 1, 1, 5, 3, 1, 1, 2, 1]) {
            0 => None,
            1 => self.generic(cx, want, len, is_elementwise),
            2 => self.higher(cx, t, len),
            3 => self.mask_if(cx, t, len),
            4 => self.cond(cx, want, len),
            5 => self.tier2(cx, t, len),
            6 => self.call_fn(cx, want, len),
            7 => self.block(cx, want, len),
            8 => self.array_lit(t, len),
            9 => self.fused_array(cx, t, len),
            _ => self.div_const(cx, want, len),
        }
    }

    /// A variable, a literal, or an array derived from a variable of the right length.
    fn leaf(&mut self, cx: &Ctx, want: ValueType, len: Option<Len>) -> E {
        match want {
            ValueType::Scalar(t) => {
                let vars: Vec<Var> = cx.vars.iter().filter(|v| v.ty == want).cloned().collect();
                // A lambda body mostly reads its parameters and own lets, so it varies.
                let local: Vec<Var> = vars.iter().filter(|v| v.local).cloned().collect();
                let percent = if cx.lambda { 80 } else { 65 };
                if !vars.is_empty() && self.c.flip(percent) {
                    let pool = if !local.is_empty() && self.c.flip(75) {
                        &local
                    } else {
                        &vars
                    };
                    let v = self.c.pick(pool).clone();
                    return self.var(cx, &v);
                }
                // Without a local of the type, a lambda body reads another local through a
                // cast or a comparison.
                let others: Vec<Var> = cx.vars.iter().filter(|v| v.local).cloned().collect();
                if cx.lambda && !others.is_empty() && self.c.flip(70) {
                    let v = self.c.pick(&others).clone();
                    let x = self.var(cx, &v);
                    return self.derive(cx, x, t);
                }
                self.lit(t)
            }
            ValueType::Array(t) => {
                let fits = |v: &Var| is_array(v.ty) && (len.is_none() || v.len == len);
                let exact: Vec<Var> = cx
                    .vars
                    .iter()
                    .filter(|v| v.ty == want && fits(v))
                    .cloned()
                    .collect();
                let any: Vec<Var> = cx.vars.iter().filter(|v| fits(v)).cloned().collect();
                let choice = self.c.weighted(&[
                    usize::from(!exact.is_empty()) * 6,
                    usize::from(!any.is_empty()) * 3,
                    usize::from(any.is_empty()),
                ]);
                match choice {
                    0 => {
                        let local: Vec<Var> = exact.iter().filter(|v| v.local).cloned().collect();
                        let pool = if !local.is_empty() && self.c.flip(75) {
                            &local
                        } else {
                            &exact
                        };
                        let v = self.c.pick(pool).clone();
                        self.var(cx, &v)
                    }
                    1 => {
                        let v = self.c.pick(&any).clone();
                        let x = self.var(cx, &v);
                        self.derive(cx, x, t)
                    }
                    _ => {
                        // Every length class a caller asks for has a variable in scope, apart
                        // from known lengths, which a fill can build.
                        let n = match len {
                            None => 1 + self.c.below(6),
                            Some(Len::Known(n)) => n,
                            Some(Len::Class(_)) => panic!("no variable of length {len:?}"),
                        };
                        let fill = self.lit(t);
                        let nl = self.lit_i64(n);
                        self.op(cx, Op::Fill, vec![nl, fill], Some(Len::Known(n)))
                            .expect("fill on a known length")
                    }
                }
            }
            ValueType::Str => unreachable!("no strings are generated"),
        }
    }

    fn var(&self, cx: &Ctx, v: &Var) -> E {
        E {
            k: K::Var(v.name.clone(), cx.lambda && !v.local),
            ty: v.ty,
            lt: v.lt,
            len: v.len,
        }
    }

    fn lit(&mut self, t: ScalarType) -> E {
        let text = *self.c.pick(literals(t));
        E {
            k: K::Lit(text.to_string()),
            ty: ValueType::Scalar(t),
            lt: ValueType::Scalar(t),
            len: None,
        }
    }

    fn lit_i64(&self, n: usize) -> E {
        E {
            k: K::Lit(n.to_string()),
            ty: ValueType::Scalar(ScalarType::I64),
            lt: ValueType::Scalar(ScalarType::I64),
            len: None,
        }
    }

    /// `x` with element type `t`, of the same length if an array: a cast, a comparison with
    /// a literal for `bool`, or `x` itself.
    fn derive(&mut self, cx: &Ctx, x: E, t: ScalarType) -> E {
        let s = elem(x.ty);
        if s == t {
            return x;
        }
        let len = x.len;
        let e = if t == ScalarType::Bool {
            let cmp = *self.c.pick(&CMPS);
            let c = self.lit(s);
            let cmp = if s == ScalarType::Bool && !matches!(cmp, Op::Eq | Op::Ne) {
                Op::Eq
            } else {
                cmp
            };
            self.op(cx, cmp, vec![x, c], len)
        } else {
            self.op(cx, Op::Cast(t), vec![x], len)
        };
        e.expect("casts and comparisons exist for every generated type")
    }

    /// An op node on `args`, typed by the registry, or `None` when no kernel applies. Inside a
    /// lambda it also resolves the op on the lifted types. `len` overrides the length of an
    /// array result, which otherwise is the length of the first array argument.
    fn op(&self, cx: &Ctx, op: Op, args: Vec<E>, len: Option<Len>) -> Option<E> {
        let tys: Vec<ValueType> = args.iter().map(|a| a.ty).collect();
        let id = self.reg.lookup(op, &tys)?;
        let ty = self.reg.get(id).output;
        let lt = if cx.lambda {
            let lts: Vec<ValueType> = args.iter().map(|a| a.lt).collect();
            let l = self.reg.lookup_lifted(op, &lts).ok()?;
            self.reg.get(l.kernel).output
        } else {
            ty
        };
        let len = if is_array(ty) {
            len.or_else(|| args.iter().find_map(|a| a.len))
        } else {
            None
        };
        Some(E {
            k: K::Op(op, args, false),
            ty,
            lt,
            len,
        })
    }

    /// Binds `e` to a block-local name unless it is a variable, so a caller can mention it
    /// more than once. `finish` wraps the result in the block.
    fn bind(&mut self, cx: &mut Ctx, e: E) -> (E, Option<(String, E)>) {
        if matches!(e.k, K::Var(..)) {
            return (e, None);
        }
        let name = self.fresh("b");
        let v = Var {
            name: name.clone(),
            ty: e.ty,
            lt: e.lt,
            len: e.len,
            local: true,
        };
        cx.vars.push(v.clone());
        (self.var(cx, &v), Some((name, e)))
    }

    /// Binds an array unless a variable of its length class is in scope, so later operands
    /// can be generated at that length.
    fn anchor(&mut self, cx: &mut Ctx, e: E) -> (E, Option<(String, E)>) {
        if e.len.is_none() || cx.vars.iter().any(|v| v.len == e.len) {
            return (e, None);
        }
        self.bind(cx, e)
    }

    fn finish(&self, cx: &mut Ctx, binding: Option<(String, E)>, r: Option<E>) -> Option<E> {
        let Some((name, value)) = binding else {
            return r;
        };
        let at = cx
            .vars
            .iter()
            .rposition(|v| v.name == name)
            .expect("the binding is in scope");
        cx.vars.remove(at);
        let r = r?;
        Some(E {
            ty: r.ty,
            lt: r.lt,
            len: r.len,
            k: K::Block(vec![(name, value)], Box::new(r)),
        })
    }

    /// Whether an operand ignores the length of the others, rarely and only when the caller
    /// does not require a length.
    fn mismatch(&mut self, len: Option<Len>) -> bool {
        len.is_none() && self.c.flip(2)
    }

    /// An op of a kernel with output `want` chosen among the ops `ops` allows. Array operands
    /// share the length class of the first one.
    fn generic(
        &mut self,
        cx: &mut Ctx,
        want: ValueType,
        len: Option<Len>,
        ops: fn(Op) -> bool,
    ) -> Option<E> {
        let sigs: Vec<&Sig> = self
            .sigs
            .iter()
            .filter(|s| {
                s.out == want
                    && ops(s.op)
                    && s.op != Op::DivConst
                    // Ops whose result length is new need a free length.
                    && (len.is_none() || !matches!(s.op, Op::Compress | Op::Where))
                    && !matches!(
                        s.op,
                        Op::Iota
                            | Op::Fill
                            | Op::RandI64
                            | Op::RandF64
                            | Op::Slice
                            | Op::Gather
                            | Op::Histogram
                            | Op::ReduceInto(_)
                    )
            })
            .collect();
        if sigs.is_empty() {
            return None;
        }
        let sig = (*self.c.pick(&sigs)).clone();
        // Elementwise kernels with an array result take an array or a scalar at each `Either`
        // position, at least one an array.
        let mut array_at: Vec<bool> = sig
            .inputs
            .iter()
            .map(|s| match s {
                InputSpec::Array(_) => true,
                InputSpec::Either(_) => self.c.flip(70),
                _ => false,
            })
            .collect();
        if sig.class == KernelClass::Elementwise && is_array(want) && !array_at.contains(&true) {
            array_at[0] = true;
        }
        let mut args = Vec::new();
        let mut bindings = Vec::new();
        let mut class = len;
        for (spec, arr) in sig.inputs.iter().zip(&array_at) {
            let t = spec_elem(*spec).expect("no string inputs");
            if *arr {
                // Reductions and length take any array. Rarely a later operand ignores the
                // length, which is a length mismatch error unless the lengths happen to agree.
                let req = if is_reduction(sig.op) || self.mismatch(len) {
                    None
                } else {
                    class
                };
                let a = self.expr(cx, ValueType::Array(t), req);
                if class.is_none() && !is_reduction(sig.op) {
                    let (a, b) = self.anchor(cx, a);
                    class = a.len;
                    bindings.push(b);
                    args.push(a);
                } else {
                    args.push(a);
                }
            } else {
                args.push(self.expr(cx, ValueType::Scalar(t), None));
            }
        }
        let fresh = matches!(sig.op, Op::Compress | Op::Where).then(|| self.fresh_class());
        let mut r = self.op(cx, sig.op, args, fresh);
        if let Some(e) = &mut r
            && sig.op == Op::Reduce(ReduceOp::Add)
            && let K::Op(_, _, sugar) = &mut e.k
        {
            *sugar = self.c.flip(50);
        }
        for b in bindings.into_iter().rev() {
            r = self.finish(cx, b, r);
        }
        r
    }

    /// A conditional on a scalar `bool`. Inside a lambda it lifts to a select.
    fn cond(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let c = self.expr(cx, ValueType::Scalar(ScalarType::Bool), None);
        let a = self.expr(cx, want, len);
        // The other branch may share the length of the first, if a variable has it.
        let anchored = a.len.filter(|l| cx.vars.iter().any(|v| v.len == Some(*l)));
        let same = if self.c.flip(50) { anchored } else { None };
        let b = self.expr(cx, want, len.or(same));
        let lt = if cx.lambda {
            let l = self
                .reg
                .lookup_lifted(Op::Select, &[c.lt, a.lt, b.lt])
                .ok()?;
            self.reg.get(l.kernel).output
        } else {
            want
        };
        let out_len = match (a.len, b.len) {
            (Some(x), Some(y)) if x == y => Some(x),
            (None, None) => None,
            _ => Some(self.fresh_class()),
        };
        Some(E {
            k: K::If(Box::new(c), Box::new(a), Box::new(b)),
            ty: want,
            lt,
            len: out_len,
        })
    }

    /// An `if` on a `bool[]` mask, selecting per element.
    fn mask_if(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let m = self.expr(cx, ValueType::Array(ScalarType::Bool), len);
        let (m, binding) = self.anchor(cx, m);
        let branch = if self.c.flip(50) {
            ValueType::Array(t)
        } else {
            ValueType::Scalar(t)
        };
        let blen = is_array(branch).then_some(m.len).flatten();
        let a = self.expr(cx, branch, blen);
        let b = self.expr(cx, branch, blen);
        let id = self.reg.lookup(Op::Select, &[m.ty, a.ty, b.ty])?;
        let ty = self.reg.get(id).output;
        let r = E {
            len: m.len,
            k: K::If(Box::new(m), Box::new(a), Box::new(b)),
            ty,
            lt: ty,
        };
        self.finish(cx, binding, Some(r))
    }

    /// A call of a user function returning `want`.
    fn call_fn(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let cands: Vec<usize> = (0..cx.fns)
            .filter(|&i| {
                let f = &self.fns[i];
                f.ret == want
                    && (!cx.lambda || f.liftable)
                    && match (len, f.ret_len) {
                        (None, _) => true,
                        (Some(_), RetLen::Params) => true,
                        (Some(Len::Known(n)), RetLen::Known(m)) => n == m,
                        _ => false,
                    }
            })
            .collect();
        if cands.is_empty() {
            return None;
        }
        let i = *self.c.pick(&cands);
        let params: Vec<ValueType> = self.fns[i].params.iter().map(|(_, t)| *t).collect();
        let ret_len = self.fns[i].ret_len;
        let mut class = if ret_len == RetLen::Params { len } else { None };
        let mut args = Vec::new();
        let mut bindings = Vec::new();
        for t in params {
            if is_array(t) {
                let a = self.expr(cx, t, class);
                if class.is_none() {
                    let (a, b) = self.anchor(cx, a);
                    class = a.len;
                    bindings.push(b);
                    args.push(a);
                } else {
                    args.push(a);
                }
            } else {
                args.push(self.expr(cx, t, None));
            }
        }
        let lt = if cx.lambda {
            let lts: Vec<ValueType> = args.iter().map(|a| a.lt).collect();
            self.lift_call(i, &lts)?
        } else {
            want
        };
        let rlen = match ret_len {
            RetLen::Scalar => None,
            RetLen::Params => class,
            RetLen::Known(n) => Some(Len::Known(n)),
            RetLen::Fresh => Some(self.fresh_class()),
        };
        let mut r = Some(E {
            k: K::Call(i, args),
            ty: want,
            lt,
            len: rlen,
        });
        for b in bindings.into_iter().rev() {
            r = self.finish(cx, b, r);
        }
        r
    }

    /// How deeply bindings that are read more than once nest in `e`: block lets read twice or
    /// more, and function parameters their body reads twice or more. `env` holds the depth of
    /// each variable bound so far.
    fn share_depth(&self, e: &E, env: &HashMap<String, usize>) -> usize {
        let max = |es: &[E]| {
            es.iter()
                .map(|x| self.share_depth(x, env))
                .max()
                .unwrap_or(0)
        };
        let reads = |body: &E| {
            let mut uses = BTreeMap::new();
            walk(body, false, &mut BTreeSet::new(), &mut uses);
            uses
        };
        match &e.k {
            K::Lit(_) | K::ArrayLit(_) => 0,
            K::Var(n, _) => env.get(n).copied().unwrap_or(0),
            K::Op(_, args, _) => max(args),
            K::If(c, a, b) => max(&[(**c).clone(), (**a).clone(), (**b).clone()]),
            K::Block(lets, r) => {
                let mut uses = reads(r);
                for (_, v) in lets {
                    walk(v, false, &mut BTreeSet::new(), &mut uses);
                }
                let mut inner = env.clone();
                let mut deepest = 0;
                for (n, v) in lets {
                    let d = self.share_depth(v, &inner);
                    deepest = deepest.max(d);
                    let shared = uses.get(n).copied().unwrap_or(0) >= 2;
                    inner.insert(n.clone(), d + usize::from(shared));
                }
                deepest.max(self.share_depth(r, &inner))
            }
            K::Call(i, args) => {
                let f = &self.fns[*i];
                let uses = reads(&f.body);
                let inner: HashMap<String, usize> = f
                    .params
                    .iter()
                    .zip(args)
                    .map(|((n, _), a)| {
                        let shared = uses.get(n).copied().unwrap_or(0) >= 2;
                        (n.clone(), self.share_depth(a, env) + usize::from(shared))
                    })
                    .collect();
                max(args).max(self.share_depth(&f.body, &inner))
            }
            K::Higher(_, arrays, lam) => {
                let body = match lam {
                    Lam::Lambda(_, body) => self.share_depth(body, env),
                    Lam::Fn(i) => self.share_depth(&self.fns[*i].body, &HashMap::new()),
                };
                max(arrays).max(body)
            }
        }
    }

    /// The lifted result type of calling function `i` on arguments of lifted types `lts`, the
    /// way the checker instantiates a function body per pattern of lifted argument types.
    fn lift_call(&self, i: usize, lts: &[ValueType]) -> Option<ValueType> {
        let f = &self.fns[i];
        let mut env: HashMap<String, ValueType> = f
            .params
            .iter()
            .map(|(n, _)| n.clone())
            .zip(lts.iter().copied())
            .collect();
        self.lift(&f.body, &mut env)
    }

    fn lift(&self, e: &E, env: &mut HashMap<String, ValueType>) -> Option<ValueType> {
        let resolve = |op: Op, lts: &[ValueType]| {
            self.reg
                .lookup_lifted(op, lts)
                .ok()
                .map(|l| self.reg.get(l.kernel).output)
        };
        match &e.k {
            K::Lit(_) => Some(e.ty),
            K::Var(n, _) => Some(env.get(n).copied().unwrap_or(e.ty)),
            K::Op(op, args, _) => {
                let lts = args
                    .iter()
                    .map(|a| self.lift(a, env))
                    .collect::<Option<Vec<_>>>()?;
                resolve(*op, &lts)
            }
            K::If(c, a, b) => {
                let lts = [self.lift(c, env)?, self.lift(a, env)?, self.lift(b, env)?];
                resolve(Op::Select, &lts)
            }
            K::Block(lets, r) => {
                for (n, v) in lets {
                    let t = self.lift(v, env)?;
                    env.insert(n.clone(), t);
                }
                self.lift(r, env)
            }
            K::Call(i, args) => {
                let lts = args
                    .iter()
                    .map(|a| self.lift(a, env))
                    .collect::<Option<Vec<_>>>()?;
                self.lift_call(*i, &lts)
            }
            K::ArrayLit(_) | K::Higher(..) => None,
        }
    }

    /// A block of lets around an expression of type `want`.
    fn block(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let n = 1 + self.c.below(2);
        let mut lets = Vec::new();
        for _ in 0..n {
            let t = if cx.lambda {
                ValueType::Scalar(self.ty())
            } else {
                self.want_type()
            };
            let value = self.expr(cx, t, None);
            let name = self.fresh("b");
            cx.vars.push(Var {
                name: name.clone(),
                ty: value.ty,
                lt: value.lt,
                len: value.len,
                local: true,
            });
            lets.push((name, value));
        }
        let r = self.expr(cx, want, len);
        cx.vars.truncate(cx.vars.len() - n);
        // Only what reaches a print or an export runs, so most lets that nothing reads are
        // dropped. A few stay, so dead lets are covered too.
        let mut uses = BTreeMap::new();
        walk(&r, cx.lambda, &mut BTreeSet::new(), &mut uses);
        for (_, v) in &lets {
            walk(v, cx.lambda, &mut BTreeSet::new(), &mut uses);
        }
        let mut kept = Vec::new();
        for (name, value) in lets {
            if uses.contains_key(&name) || self.c.flip(20) {
                kept.push((name, value));
            }
        }
        if kept.is_empty() {
            return Some(r);
        }
        Some(E {
            ty: r.ty,
            lt: r.lt,
            len: r.len,
            k: K::Block(kept, Box::new(r)),
        })
    }

    /// `div_const` by a scalar. The divisor is a nonzero literal, rarely any scalar, which
    /// may be zero, an error. Inside a lambda the divisor stays uniform.
    fn div_const(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let t = elem(want);
        if !INTS.contains(&t) {
            return None;
        }
        let x = self.expr(cx, want, len);
        // Inside a lambda a divisor that depends on the parameter does not lift, and `op`
        // rejects it.
        let d = if self.c.flip(5) {
            self.expr(cx, ValueType::Scalar(t), None)
        } else {
            let lits: Vec<&str> = literals(t)
                .iter()
                .copied()
                .filter(|l| !matches!(*l, "0" | "0i32" | "0u8" | "0u64"))
                .collect();
            let text = *self.c.pick(&lits);
            E {
                k: K::Lit(text.into()),
                ty: ValueType::Scalar(t),
                lt: ValueType::Scalar(t),
                len: None,
            }
        };
        self.op(cx, Op::DivConst, vec![x, d], None)
    }

    /// `map`, `zip_map`, or `filter` with a lambda or a liftable function.
    fn higher(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let kind = match self.c.weighted(&[5, 3, usize::from(len.is_none()) * 2]) {
            0 => Higher::Map,
            1 => Higher::ZipMap,
            _ => Higher::Filter,
        };
        let ret = if kind == Higher::Filter {
            ScalarType::Bool
        } else {
            t
        };
        let arity = if kind == Higher::ZipMap { 2 } else { 1 };
        // A liftable function of the right shape may stand in for the lambda.
        let fns: Vec<usize> = (0..cx.fns)
            .filter(|&i| {
                let f = &self.fns[i];
                f.liftable
                    && f.params.len() == arity
                    && f.ret == ValueType::Scalar(ret)
                    && (kind != Higher::Filter || f.params[0].1 == ValueType::Scalar(t))
            })
            .collect();
        let by_fn = (!fns.is_empty() && self.c.flip(50)).then(|| *self.c.pick(&fns));
        let elems: Vec<ScalarType> = match by_fn {
            Some(i) => self.fns[i].params.iter().map(|(_, p)| elem(*p)).collect(),
            None if kind == Higher::Filter => vec![t],
            None => (0..arity).map(|_| self.ty()).collect(),
        };
        let x = self.expr(cx, ValueType::Array(elems[0]), len);
        let (x, binding) = if arity == 2 {
            self.anchor(cx, x)
        } else {
            (x, None)
        };
        let mut arrays = vec![x];
        if arity == 2 {
            let ylen = if self.mismatch(len) {
                None
            } else {
                arrays[0].len
            };
            let y = self.expr(cx, ValueType::Array(elems[1]), ylen);
            arrays.push(y);
        }
        let lam = match by_fn {
            Some(i) => Some(Lam::Fn(i)),
            None => self.lambda(cx, &elems, ret),
        };
        let r = lam.map(|lam| {
            let out_len = if kind == Higher::Filter {
                Some(self.fresh_class())
            } else {
                arrays[0].len
            };
            let ty = ValueType::Array(t);
            E {
                k: K::Higher(kind, arrays, lam),
                ty,
                lt: ty,
                len: out_len,
            }
        });
        self.finish(cx, binding, r)
    }

    /// A lambda over parameters of element types `elems` returning `ret`. It captures the
    /// scalars in scope.
    fn lambda(&mut self, cx: &Ctx, elems: &[ScalarType], ret: ScalarType) -> Option<Lam> {
        let mut vars: Vec<Var> = cx
            .vars
            .iter()
            .filter(|v| !is_array(v.ty))
            .map(|v| Var {
                local: false,
                lt: v.ty,
                ..v.clone()
            })
            .collect();
        let mut names = Vec::new();
        for t in elems {
            let name = self.fresh("p");
            vars.push(Var {
                name: name.clone(),
                ty: ValueType::Scalar(*t),
                lt: ValueType::Array(*t),
                len: None,
                local: true,
            });
            names.push(name);
        }
        let mut lcx = Ctx {
            vars,
            lambda: true,
            depth: cx.depth,
            fns: cx.fns,
        };
        let body = self.expr(&mut lcx, ValueType::Scalar(ret), None);
        Some(Lam::Lambda(names, Box::new(body)))
    }

    /// An array literal. Lowering takes only literals as its elements.
    fn array_lit(&mut self, t: ScalarType, len: Option<Len>) -> Option<E> {
        let n = match len {
            None => 1 + self.c.below(6),
            Some(Len::Known(n)) if (1..=8).contains(&n) => n,
            Some(_) => return None,
        };
        let items: Vec<E> = (0..n).map(|_| self.lit(t)).collect();
        Some(E {
            k: K::ArrayLit(items),
            ty: ValueType::Array(t),
            lt: ValueType::Array(t),
            len: Some(Len::Known(n)),
        })
    }

    /// `length` of an array of class `len`, or the literal for a known length.
    fn length_of(&mut self, cx: &Ctx, len: Len) -> Option<E> {
        let vars: Vec<Var> = cx
            .vars
            .iter()
            .filter(|v| v.len == Some(len))
            .cloned()
            .collect();
        if vars.is_empty() || (matches!(len, Len::Known(_)) && self.c.flip(30)) {
            return match len {
                Len::Known(n) => Some(self.lit_i64(n)),
                Len::Class(_) => None,
            };
        }
        let v = self.c.pick(&vars).clone();
        let x = self.var(cx, &v);
        self.op(cx, Op::Length, vec![x], None)
    }

    /// A length to build an array of: of class `len` when given, else a small literal or the
    /// length of an array in scope.
    fn new_len(&mut self, cx: &Ctx, len: Option<Len>) -> Option<(E, Len)> {
        if let Some(l) = len {
            return Some((self.length_of(cx, l)?, l));
        }
        let classes: Vec<Len> = cx.vars.iter().filter_map(|v| v.len).collect();
        if !classes.is_empty() && self.c.flip(60) {
            let l = *self.c.pick(&classes);
            return Some((self.length_of(cx, l)?, l));
        }
        let n = self.c.below(21);
        Some((self.lit_i64(n), Len::Known(n)))
    }

    /// The tier 2 primitives with a result of `t[]`.
    fn tier2(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let want = ValueType::Array(t);
        let i64s = t == ScalarType::I64;
        let free = len.is_none();
        let weights = [
            2,                                         // scan
            2,                                         // scan_by
            2,                                         // prev
            usize::from(i64s) * 2,                     // argsort
            usize::from(i64s) * 2,                     // group_id
            usize::from(free) * 2,                     // compress
            usize::from(free && i64s),                 // where
            usize::from(i64s),                         // iota
            1,                                         // fill
            usize::from(i64s || t == ScalarType::F64), // rand
            3,                                         // gather
            usize::from(free) * 2,                     // slice
            usize::from(i64s) * 2,                     // histogram
            3,                                         // reduce_into
        ];
        match self.c.weighted(&weights) {
            0 => self.generic(cx, want, len, |op| matches!(op, Op::Scan(_))),
            1 => self.scan_by(cx, want, len),
            2 => self.generic(cx, want, len, |op| op == Op::Prev),
            3 => self.generic(cx, want, len, |op| op == Op::ArgSort),
            4 => self.generic(cx, want, len, |op| op == Op::GroupId),
            5 => self.generic(cx, want, len, |op| op == Op::Compress),
            6 => self.generic(cx, want, len, |op| op == Op::Where),
            7 => {
                let (n, l) = self.new_len(cx, len)?;
                self.op(cx, Op::Iota, vec![n], Some(l))
            }
            8 => {
                let (n, l) = self.new_len(cx, len)?;
                let v = self.expr(cx, ValueType::Scalar(t), None);
                self.op(cx, Op::Fill, vec![n, v], Some(l))
            }
            9 => {
                let (n, l) = self.new_len(cx, len)?;
                // Programs are reproducible, so every seed is explicit.
                let n_seed = self.c.below(1000);
                let seed = self.lit_i64(n_seed);
                let op = if i64s { Op::RandI64 } else { Op::RandF64 };
                self.op(cx, op, vec![n, seed], Some(l))
            }
            10 => self.gather(cx, t, len),
            11 => self.slice(cx, t),
            12 => self.histogram(cx, len),
            _ => self.reduce_into(cx, t, len),
        }
    }

    /// `scan_by`, often with segments that start where a key changes, `k != prev(k, c)`.
    fn scan_by(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let is_scan_by = |op| matches!(op, Op::ScanBy(_));
        if !self.c.flip(40) {
            return self.generic(cx, want, len, is_scan_by);
        }
        let sigs: Vec<Sig> = self
            .sigs
            .iter()
            .filter(|s| s.out == want && is_scan_by(s.op))
            .cloned()
            .collect();
        if sigs.is_empty() {
            return None;
        }
        let sig = self.c.pick(&sigs).clone();
        let vt = spec_elem(sig.inputs[0]).expect("values are an array");
        let x = self.expr(cx, ValueType::Array(vt), len);
        let (x, outer) = self.anchor(cx, x);
        let kt = *self.c.pick(&INTS);
        let k = self.expr(cx, ValueType::Array(kt), x.len);
        let (k, inner) = self.bind(cx, k);
        let fill = self.lit(kt);
        let starts = self
            .op(cx, Op::Prev, vec![k.clone(), fill], None)
            .and_then(|p| self.op(cx, Op::Ne, vec![k, p], None));
        let r = starts.and_then(|s| self.op(cx, sig.op, vec![x, s], None));
        let r = self.finish(cx, inner, r);
        self.finish(cx, outer, r)
    }

    /// An `i64[]` index array into an array of class `of`, with indices in range: a sort
    /// permutation, positions of a mask, or `iota`. Rarely any `i64[]`, which may be out of
    /// range, an error. With `len` the index array has that class.
    fn indices(&mut self, cx: &mut Ctx, of: Len, len: Option<Len>) -> Option<E> {
        let i64a = ValueType::Array(ScalarType::I64);
        let fixed = len.is_some();
        match self.c.weighted(&[3, 2, usize::from(!fixed) * 2, 1]) {
            0 => {
                let k = *self.c.pick(&INTS);
                let y = self.expr(cx, ValueType::Array(k), Some(of));
                self.op(cx, Op::ArgSort, vec![y], None)
            }
            1 => {
                let n = self.length_of(cx, of)?;
                self.op(cx, Op::Iota, vec![n], Some(of))
            }
            2 => {
                let m = self.expr(cx, ValueType::Array(ScalarType::Bool), Some(of));
                let fresh = self.fresh_class();
                self.op(cx, Op::Where, vec![m], Some(fresh))
            }
            _ => Some(self.expr(cx, i64a, len)),
        }
    }

    fn gather(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let x = self.expr(cx, ValueType::Array(t), len);
        let (x, binding) = self.anchor(cx, x);
        let of = x.len.expect("an array has a length");
        let r = self
            .indices(cx, of, len)
            .and_then(|idx| {
                let l = idx.len;
                self.op(cx, Op::Gather, vec![x, idx], l)
            })
            .filter(|e| len.is_none() || e.len == len);
        self.finish(cx, binding, r)
    }

    fn slice(&mut self, cx: &mut Ctx, t: ScalarType) -> Option<E> {
        let x = self.expr(cx, ValueType::Array(t), None);
        let (x, binding) = self.bind(cx, x);
        let of = x.len.expect("an array has a length");
        let r = match self.c.weighted(&[3, 3, 1, 1]) {
            0 => {
                let end = self.length_of(cx, of);
                end.and_then(|end| self.op(cx, Op::Slice, vec![x, self.lit_i64(0), end], None))
            }
            1 => match of {
                Len::Known(n) => {
                    let a = self.c.below(n + 1);
                    let b = a + self.c.below(n - a + 1);
                    let (sa, sb) = (self.lit_i64(a), self.lit_i64(b));
                    self.op(cx, Op::Slice, vec![x, sa, sb], Some(Len::Known(b - a)))
                }
                Len::Class(_) => None,
            },
            2 => {
                // An error on an empty array.
                let end = self.length_of(cx, of);
                let fresh = self.fresh_class();
                end.and_then(|end| {
                    self.op(cx, Op::Slice, vec![x, self.lit_i64(1), end], Some(fresh))
                })
            }
            _ => {
                let i64s = ValueType::Scalar(ScalarType::I64);
                let a = self.expr(cx, i64s, None);
                let b = self.expr(cx, i64s, None);
                let fresh = self.fresh_class();
                self.op(cx, Op::Slice, vec![x, a, b], Some(fresh))
            }
        };
        self.finish(cx, binding, r)
    }

    /// Keys in `0..n` with `n` and the length class of the result: the group ids of an array
    /// and its length, or a `bool[]` as keys with `n` of 2. Rarely any keys and `n`, which may
    /// be out of range, an error. The caller finishes the binding around its result.
    fn keys(&mut self, cx: &mut Ctx, len: Option<Len>) -> Option<Keys> {
        match self.c.weighted(&[4, usize::from(len.is_none()), 1]) {
            0 => {
                let kt = *self.c.pick(&INTS);
                let k = self.expr(cx, ValueType::Array(kt), len);
                let (k, binding) = self.anchor(cx, k);
                let of = k.len.expect("an array has a length");
                let ids = self.op(cx, Op::GroupId, vec![k], None);
                let n = self.length_of(cx, of);
                Some(Keys {
                    keys: ids,
                    n,
                    out: of,
                    binding,
                })
            }
            1 => {
                let m = self.expr(cx, ValueType::Array(ScalarType::Bool), None);
                let keys = self.op(cx, Op::Cast(ScalarType::I64), vec![m], None);
                Some(Keys {
                    keys,
                    n: Some(self.lit_i64(2)),
                    out: Len::Known(2),
                    binding: None,
                })
            }
            _ => {
                let keys = self.expr(cx, ValueType::Array(ScalarType::I64), None);
                let n = self.c.below(9);
                Some(Keys {
                    keys: Some(keys),
                    n: Some(self.lit_i64(n)),
                    out: Len::Known(n),
                    binding: None,
                })
            }
        }
    }

    fn histogram(&mut self, cx: &mut Ctx, len: Option<Len>) -> Option<E> {
        let k = self.keys(cx, len)?;
        let r = match (k.keys, k.n) {
            (Some(keys), Some(n)) if len.is_none_or(|l| l == k.out) => {
                self.op(cx, Op::Histogram, vec![keys, n], Some(k.out))
            }
            _ => None,
        };
        self.finish(cx, k.binding, r)
    }

    fn reduce_into(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let want = ValueType::Array(t);
        let sigs: Vec<Sig> = self
            .sigs
            .iter()
            .filter(|s| s.out == want && matches!(s.op, Op::ReduceInto(_)))
            .cloned()
            .collect();
        if sigs.is_empty() {
            return None;
        }
        let sig = self.c.pick(&sigs).clone();
        let vt = spec_elem(sig.inputs[1]).expect("values are an array");
        let k = self.keys(cx, len)?;
        let r = match (k.keys, k.n) {
            (Some(keys), Some(n)) if len.is_none_or(|l| l == k.out) => {
                let (keys, binding) = self.anchor(cx, keys);
                let vals = self.expr(cx, ValueType::Array(vt), keys.len);
                let r = self.op(cx, sig.op, vec![keys, vals, n], Some(k.out));
                self.finish(cx, binding, r)
            }
            _ => None,
        };
        self.finish(cx, k.binding, r)
    }

    /// Shapes the fused kernels match: `sum(x * y)`, `sum(x * x)`, `count(x < c)`, and
    /// `sum(select(x < c, x, e))`.
    fn fused_scalar(&mut self, cx: &mut Ctx, t: ScalarType) -> Option<E> {
        let want = ValueType::Scalar(t);
        let sums: Vec<ScalarType> = TYPES
            .iter()
            .copied()
            .filter(|s| {
                *s != ScalarType::Bool
                    && self
                        .reg
                        .lookup(Op::Reduce(ReduceOp::Add), &[ValueType::Array(*s)])
                        .is_some_and(|id| self.reg.get(id).output == want)
            })
            .collect();
        if sums.is_empty() && t != ScalarType::I64 {
            return None;
        }
        let shape = self.c.weighted(&[
            usize::from(!sums.is_empty()) * 2,
            usize::from(!sums.is_empty()),
            usize::from(t == ScalarType::I64) * 2,
            usize::from(!sums.is_empty()) * 2,
        ]);
        let sugar = self.c.flip(70);
        let (e, binding) = match shape {
            0 | 1 => {
                let s = *self.c.pick(&sums);
                let x = self.expr(cx, ValueType::Array(s), None);
                let (x, binding) = if shape == 1 {
                    self.bind(cx, x)
                } else {
                    self.anchor(cx, x)
                };
                let y = if shape == 1 {
                    x.clone()
                } else {
                    self.expr(cx, ValueType::Array(s), x.len)
                };
                let prod = self.op(cx, Op::Mul, vec![x, y], None);
                let e = prod.and_then(|p| self.op(cx, Op::Reduce(ReduceOp::Add), vec![p], None));
                (e, binding)
            }
            2 => {
                let s = *self.c.pick(&TYPES[..2]);
                let s = if self.c.flip(50) { s } else { self.ty() };
                let x = self.expr(cx, ValueType::Array(s), None);
                let m = self.cmp_scalar(cx, x);
                (m.and_then(|m| self.op(cx, Op::Count, vec![m], None)), None)
            }
            _ => {
                let s = *self.c.pick(&sums);
                let x = self.expr(cx, ValueType::Array(s), None);
                let (x, binding) = self.bind(cx, x);
                let m = self.cmp_scalar(cx, x.clone());
                let other = self.expr(cx, ValueType::Scalar(s), None);
                let sel = m.and_then(|m| self.op(cx, Op::Select, vec![m, x, other], None));
                let e = sel.and_then(|v| self.op(cx, Op::Reduce(ReduceOp::Add), vec![v], None));
                (e, binding)
            }
        };
        let e = e.map(|mut e| {
            if let K::Op(Op::Reduce(ReduceOp::Add), _, s) = &mut e.k {
                *s = sugar;
            }
            e
        });
        self.finish(cx, binding, e)
    }

    /// The filter shapes the fused kernels match: `compress(x, x < c)` and
    /// `filter(x, |v| v < c)`.
    fn fused_array(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        if len.is_some() || t == ScalarType::Bool {
            return None;
        }
        let x = self.expr(cx, ValueType::Array(t), None);
        let fresh = Some(self.fresh_class());
        if self.c.flip(50) {
            let (x, binding) = self.bind(cx, x);
            let m = self.cmp_scalar(cx, x.clone());
            let r = m.and_then(|m| self.op(cx, Op::Compress, vec![x, m], fresh));
            return self.finish(cx, binding, r);
        }
        let p = self.fresh("p");
        let pv = Var {
            name: p.clone(),
            ty: ValueType::Scalar(t),
            lt: ValueType::Array(t),
            len: None,
            local: true,
        };
        let mut lcx = Ctx {
            vars: cx
                .vars
                .iter()
                .filter(|v| !is_array(v.ty))
                .map(|v| Var {
                    local: false,
                    lt: v.ty,
                    ..v.clone()
                })
                .chain([pv.clone()])
                .collect(),
            lambda: true,
            depth: cx.depth,
            fns: cx.fns,
        };
        let pe = self.var(&lcx, &pv);
        let body = self.cmp_scalar(&mut lcx, pe)?;
        let ty = ValueType::Array(t);
        Some(E {
            k: K::Higher(
                Higher::Filter,
                vec![x],
                Lam::Lambda(vec![p], Box::new(body)),
            ),
            ty,
            lt: ty,
            len: fresh,
        })
    }

    /// `x` compared with a scalar, on either side.
    fn cmp_scalar(&mut self, cx: &mut Ctx, x: E) -> Option<E> {
        let s = elem(x.ty);
        let cmp = if s == ScalarType::Bool {
            *self.c.pick(&[Op::Eq, Op::Ne])
        } else {
            *self.c.pick(&CMPS)
        };
        let c = self.expr(cx, ValueType::Scalar(s), None);
        let args = if self.c.flip(25) {
            vec![c, x]
        } else {
            vec![x, c]
        };
        self.op(cx, cmp, args, None)
    }

    fn render_fn(&self, f: &FnDef) -> String {
        let params: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| format!("{n}: {}", type_name(*t)))
            .collect();
        format!(
            "fn {}({}) -> {} {{ {} }}\n",
            f.name,
            params.join(", "),
            type_name(f.ret),
            self.render_inner(&f.body)
        )
    }

    /// The contents of a block: its lets and result, or the expression itself.
    fn render_inner(&self, e: &E) -> String {
        match &e.k {
            K::Block(lets, r) => {
                let mut s = String::new();
                for (n, v) in lets {
                    s.push_str(&format!("let {n} = {}; ", self.render(v)));
                }
                s.push_str(&self.render(r));
                s
            }
            _ => self.render(e),
        }
    }

    fn render(&self, e: &E) -> String {
        let list = |args: &[E]| {
            args.iter()
                .map(|a| self.render(a))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match &e.k {
            K::Lit(s) => s.clone(),
            K::Var(n, _) => n.clone(),
            K::ArrayLit(items) => format!("[{}]", list(items)),
            K::Op(op, args, sugar) => {
                if let Some(sym) = infix(*op) {
                    return format!(
                        "({} {sym} {})",
                        self.render(&args[0]),
                        self.render(&args[1])
                    );
                }
                match op {
                    Op::Reduce(ReduceOp::Add) if *sugar => format!("sum({})", list(args)),
                    Op::Reduce(r) | Op::Scan(r) | Op::ScanBy(r) | Op::ReduceInto(r) => {
                        format!("{}({}, {})", call_name(*op), list(args), reduce_name(*r))
                    }
                    _ => format!("{}({})", call_name(*op), list(args)),
                }
            }
            K::Call(i, args) => format!("{}({})", self.fns[*i].name, list(args)),
            K::If(c, a, b) => format!(
                "(if {} {{ {} }} else {{ {} }})",
                self.render(c),
                self.render_inner(a),
                self.render_inner(b)
            ),
            K::Block(..) => format!("{{ {} }}", self.render_inner(e)),
            K::Higher(kind, arrays, lam) => {
                let name = match kind {
                    Higher::Map => "map",
                    Higher::ZipMap => "zip_map",
                    Higher::Filter => "filter",
                };
                let f = match lam {
                    Lam::Lambda(params, body) => {
                        format!("|{}| {}", params.join(", "), self.render(body))
                    }
                    Lam::Fn(i) => self.fns[*i].name.clone(),
                };
                format!("{name}({}, {f})", list(arrays))
            }
        }
    }

    /// The features of the program: what its expressions use, and how often its top-level
    /// lets are used.
    /// Also returns the number of reads of each variable.
    fn features(
        &self,
        stmts: &[(&'static str, String, E)],
    ) -> (BTreeSet<&'static str>, BTreeMap<String, usize>) {
        let mut out = BTreeSet::new();
        let mut uses: BTreeMap<String, usize> = BTreeMap::new();
        for f in &self.fns {
            out.insert(if f.liftable {
                "fn liftable"
            } else {
                "fn with arrays"
            });
            walk(&f.body, f.liftable, &mut out, &mut uses);
        }
        for (_, _, e) in stmts {
            walk(e, false, &mut out, &mut uses);
        }
        for (kind, name, _) in stmts {
            if *kind == "let" {
                out.insert(match uses.get(name).copied().unwrap_or(0) {
                    0 => "let exported at the end",
                    1 => "let used once",
                    _ => "let used more than once",
                });
            }
        }
        (out, uses)
    }
}

fn walk(e: &E, lambda: bool, out: &mut BTreeSet<&'static str>, uses: &mut BTreeMap<String, usize>) {
    match &e.k {
        K::Lit(_) => {}
        K::Var(n, captured) => {
            *uses.entry(n.clone()).or_default() += 1;
            if *captured {
                out.insert("lambda captures a scalar");
            }
        }
        K::ArrayLit(items) => {
            out.insert("array literal");
            items.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::Op(op, args, _) => {
            out.insert(feature(*op));
            if is_elementwise(*op)
                && args.iter().any(|x| !is_array(x.ty))
                && args.iter().any(|x| is_array(x.ty))
            {
                out.insert("scalar broadcast");
            }
            if lambda {
                out.insert("op in lambda");
            }
            match (op, args.first().map(|a| &a.k)) {
                (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Mul, xs, _))) => {
                    out.insert(
                        if matches!((&xs[0].k, &xs[1].k), (K::Var(a, _), K::Var(b, _)) if a == b) {
                            "shape sum(x * x)"
                        } else {
                            "shape sum(x * y)"
                        },
                    );
                }
                (Op::Reduce(ReduceOp::Add), Some(K::Op(Op::Select, _, _))) => {
                    out.insert("shape sum(select(..))");
                }
                (Op::Count, Some(K::Op(c, _, _))) if CMPS.contains(c) => {
                    out.insert("shape count(x cmp c)");
                }
                (Op::Compress, _) if matches!(&args[1].k, K::Op(c, _, _) if CMPS.contains(c)) => {
                    out.insert("shape compress(x, x cmp c)");
                }
                _ => {}
            }
            args.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::Call(_, args) => {
            out.insert(if lambda {
                "fn call in lambda"
            } else {
                "fn call"
            });
            args.iter().for_each(|x| walk(x, lambda, out, uses));
        }
        K::If(c, a, b) => {
            out.insert(if lambda {
                "if in lambda"
            } else if is_array(c.ty) {
                "if on mask"
            } else {
                "if on scalar"
            });
            walk(c, lambda, out, uses);
            walk(a, lambda, out, uses);
            walk(b, lambda, out, uses);
        }
        K::Block(lets, r) => {
            out.insert("block");
            for (_, v) in lets {
                walk(v, lambda, out, uses);
            }
            walk(r, lambda, out, uses);
        }
        K::Higher(kind, arrays, lam) => {
            out.insert(match kind {
                Higher::Map => "map",
                Higher::ZipMap => "zip_map",
                Higher::Filter => "filter",
            });
            arrays.iter().for_each(|x| walk(x, lambda, out, uses));
            match lam {
                Lam::Lambda(_, body) => walk(body, true, out, uses),
                Lam::Fn(_) => {
                    out.insert("fn as lambda");
                }
            }
        }
    }
}

/// A SplitMix64 generator, for input values from a seed.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const F64_SPECIAL: [f64; 14] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MIN_POSITIVE,
    5e-324,
    f64::MAX,
    f64::MIN,
    0.1,
    3.0,
];

const F32_SPECIAL: [f32; 14] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MIN_POSITIVE,
    1e-45,
    f32::MAX,
    f32::MIN,
    0.1,
    3.0,
];

/// The bits of one element of type `t` in a value style: 0 small values, 1 special and
/// small values, 2 any bits, 3 and 4 few distinct values (sorted for 4 by the caller), and
/// for `bool` mostly false in 2, mostly true in 3, and rarely true in 4.
fn element(t: ScalarType, style: usize, rng: &mut SplitMix) -> u64 {
    let small = |r: &mut SplitMix, n: u64| r.below(n);
    match t {
        ScalarType::F64 => match style {
            0 => (f64::from(u32::try_from(small(rng, 9)).expect("small")) * 0.5 - 1.0).to_bits(),
            1 if rng.below(2) == 0 => {
                F64_SPECIAL[usize::try_from(rng.below(14)).expect("small")].to_bits()
            }
            1 => {
                (f64::from(u32::try_from(small(rng, 200)).expect("small")) * 0.25 - 20.0).to_bits()
            }
            2 => rng.next(),
            _ => [1.0f64, 2.0, -0.0][usize::try_from(rng.below(3)).expect("small")].to_bits(),
        },
        ScalarType::F32 => u64::from(
            match style {
                0 => f32::from(u8::try_from(small(rng, 9)).expect("small")) * 0.5 - 1.0,
                1 if rng.below(2) == 0 => {
                    F32_SPECIAL[usize::try_from(rng.below(14)).expect("small")]
                }
                1 => f32::from(u8::try_from(small(rng, 200)).expect("small")) * 0.25 - 20.0,
                2 => f32::from_bits(u32::try_from(rng.next() >> 32).expect("high half")),
                _ => [1.0f32, 2.0, -0.0][usize::try_from(rng.below(3)).expect("small")],
            }
            .to_bits(),
        ),
        ScalarType::Bool => match style {
            0 | 1 => rng.below(2),
            2 => u64::from(rng.below(8) == 0),
            3 => u64::from(rng.below(8) != 0),
            _ => u64::from(rng.below(16) == 0),
        },
        _ => {
            let width = t.width() * 8;
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1 << width) - 1
            };
            let signed = t.is_signed();
            let v: i64 = match style {
                0 => rng.below(9).cast_signed() - if signed { 4 } else { 0 },
                1 if rng.below(2) == 0 => {
                    let extremes: [i64; 4] = if signed {
                        let m = 1i64 << (width - 1);
                        [m.wrapping_neg(), m.wrapping_sub(1), -1, 0]
                    } else {
                        [
                            0,
                            -1,
                            1,
                            1i64.wrapping_shl(u32::try_from(width - 1).expect("width")),
                        ]
                    };
                    extremes[usize::try_from(rng.below(4)).expect("small")]
                }
                1 => rng.below(1000).cast_signed() - if signed { 500 } else { 0 },
                2 => rng.next().cast_signed(),
                _ => rng.below(4).cast_signed(),
            };
            v.cast_unsigned() & mask
        }
    }
}

fn scalar_of(t: ScalarType, bits: u64) -> Scalar {
    match t {
        ScalarType::F64 => Scalar::F64(f64::from_bits(bits)),
        ScalarType::F32 => Scalar::F32(f32::from_bits(u32::try_from(bits).expect("f32 bits"))),
        ScalarType::I64 => Scalar::I64(bits.cast_signed()),
        ScalarType::I32 => Scalar::I32(u32::try_from(bits).expect("i32 bits").cast_signed()),
        ScalarType::U8 => Scalar::U8(u8::try_from(bits).expect("u8 bits")),
        ScalarType::U64 => Scalar::U64(bits),
        ScalarType::Bool => Scalar::Bool(bits != 0),
        other => unreachable!("{other:?} is not generated"),
    }
}

/// Source text for a scalar. NaN and the infinities are divisions, so a NaN loses its
/// payload and sign in a reproduction.
fn scalar_literal(s: &Scalar) -> String {
    fn float(v: f64, suffix: &str) -> String {
        if v.is_nan() {
            return format!("(0.0{suffix} / 0.0{suffix})");
        }
        if v.is_infinite() {
            let sign = if v > 0.0 { "" } else { "-" };
            return format!("({sign}1.0{suffix} / 0.0{suffix})");
        }
        // `Display` writes every digit without an exponent, which the lexer needs.
        let mut t = format!("{v}");
        if !t.contains('.') {
            t.push_str(".0");
        }
        format!("({t}{suffix})")
    }
    match s {
        Scalar::F64(v) => float(*v, ""),
        Scalar::F32(v) => {
            // The shortest decimal of the f32 converts back to the same f32.
            let text = format!("{v}");
            let mut t = text;
            if v.is_nan() || v.is_infinite() {
                return float(f64::from(*v), "f32");
            }
            if !t.contains('.') {
                t.push_str(".0");
            }
            format!("({t}f32)")
        }
        Scalar::I64(v) => format!("({v})"),
        Scalar::I32(v) => format!("({v}i32)"),
        Scalar::U8(v) => format!("{v}u8"),
        Scalar::U64(v) => format!("{v}u64"),
        Scalar::Bool(v) => format!("{v}"),
        other => unreachable!("{other:?} is not generated"),
    }
}

/// Source text for a value, an array literal or `fill(0, ..)` for an empty array.
fn literal_of(value: &Value) -> String {
    match value {
        Value::Scalar(s) => scalar_literal(s),
        Value::Array(a) if a.is_empty() => {
            let zero = scalar_literal(&scalar_of(a.elem(), 0));
            format!("fill(0, {zero})")
        }
        Value::Array(a) => {
            let items: Vec<Scalar> = (0..a.len()).map(|i| a.get(i)).collect();
            let finite = |s: &Scalar| match s {
                Scalar::F64(v) => v.is_finite(),
                Scalar::F32(v) => v.is_finite(),
                _ => true,
            };
            let list = |xs: &[Scalar]| {
                let texts: Vec<String> = xs.iter().map(scalar_literal).collect();
                format!("[{}]", texts.join(", "))
            };
            if items.iter().all(finite) {
                return list(&items);
            }
            // Array literal elements are literals, so NaN and the infinities come from
            // dividing two literals elementwise, and every other element is divided by one,
            // which is exact.
            let (num, den): (Vec<Scalar>, Vec<Scalar>) = items
                .iter()
                .map(|s| match s {
                    Scalar::F64(v) if v.is_nan() => (Scalar::F64(0.0), Scalar::F64(0.0)),
                    Scalar::F64(v) if v.is_infinite() => {
                        (Scalar::F64(v.signum()), Scalar::F64(0.0))
                    }
                    Scalar::F64(v) => (Scalar::F64(*v), Scalar::F64(1.0)),
                    Scalar::F32(v) if v.is_nan() => (Scalar::F32(0.0), Scalar::F32(0.0)),
                    Scalar::F32(v) if v.is_infinite() => {
                        (Scalar::F32(v.signum()), Scalar::F32(0.0))
                    }
                    Scalar::F32(v) => (Scalar::F32(*v), Scalar::F32(1.0)),
                    _ => unreachable!("only floats are not finite"),
                })
                .unzip();
            format!("({} / {})", list(&num), list(&den))
        }
        Value::Str(_) => unreachable!("no strings are generated"),
    }
}
