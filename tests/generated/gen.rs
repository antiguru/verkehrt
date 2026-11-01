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
use verkehrt::value::{ScalarType, Value, ValueType};

mod choices;
mod features;
mod inputs;
mod render;
mod tier2;

use choices::Choices;
use features::walk;
use inputs::{literal_of, literals};

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
}
