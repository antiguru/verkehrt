//! The kernel registry. It is the source of truth for which primitives exist on which
//! types (an op on a type without a kernel is a type error) and the dispatch table.

use std::collections::HashMap;
use std::sync::Arc;

pub use crate::ops::Isa;
use crate::ops::Op;
use crate::value::{Arg, ArrayBuilder, Scalar, ScalarType, Value, ValueType};

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod arch;
mod auto;
pub mod dispatch;
pub mod scalar;
mod select_loops;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use dispatch::Dispatch;

#[cfg(test)]
mod tests;

/// An elementwise entry. `out` is the destination rows of the output type as bytes, every
/// `Rows` argument has exactly `out.len() / width` rows, and the caller guarantees that the
/// CPU supports the entry's ISA. The `String` is the runtime error message, the executor
/// adds the span.
///
/// `Rows` windows and `out` are aligned for their element type only, because an array view
/// can start at any row. An entry must not assume 32 or 64 byte alignment. A `Bool` output
/// must be written as 0 or 1, because later kernels read bool bytes as exactly those values.
pub type ElementwiseFn = unsafe fn(args: &[Arg], out: &mut [u8]) -> Result<(), String>;

/// Creates the accumulator of a reduction. The caller guarantees that the CPU supports the
/// entry's ISA.
///
/// The `Rows` windows that the accumulator's `update` receives are aligned for their element
/// type only, as for `ElementwiseFn`. A `Bool` result must be 0 or 1. The arguments of
/// `update` are not checked by the caller, because `Dispatch::reducer` hands the accumulator
/// out for tiles, so an `update` that relies on equal row counts or argument kinds for memory
/// safety must check them itself.
pub type ReducerFn = unsafe fn() -> Box<dyn Reducer>;

/// The running state of one reduction.
///
/// Every `update` except the last receives a row count that is a multiple of 16, and `finish`
/// follows the last `update`. An empty input means no `update` at all, and `finish` then
/// returns the identity, or the empty-array error for `reduce_min`, `reduce_max`, `argmin`,
/// and `argmax`. Index results (`argmin`, `argmax`) count rows over all updates, so a tile's
/// local index is offset by the rows consumed before it.
pub trait Reducer: Send {
    /// Consumes one tile. `args` may arrive with unchecked kinds and row counts, see
    /// `ReducerFn`.
    fn update(&mut self, args: &[Arg]) -> Result<(), String>;
    fn finish(self: Box<Self>) -> Result<Scalar, String>;
}

/// Creates the state of a scan, which maps an array to an array of the same length where a
/// row may depend on the rows before it. The caller guarantees that the CPU supports the
/// entry's ISA.
///
/// The `Rows` windows that the scanner's `update` receives are aligned for their element type
/// only, as for `ElementwiseFn`, and its arguments are not checked by the caller, as for
/// `ReducerFn`.
pub type ScanFn = unsafe fn() -> Box<dyn Scanner>;

/// The running state of one scan.
///
/// `update` consumes one tile and writes the output rows of exactly that tile to `out`, the
/// bytes of the tile's rows of the output type. The state carries to the next `update`.
/// Tiles may have any length, and the output must not depend on how the rows split into
/// tiles. A `Bool` output must be written as 0 or 1. An empty input means no `update` at all.
pub trait Scanner: Send {
    /// Consumes one tile. `args` may arrive with unchecked kinds and row counts, and `out`
    /// with a row count that differs from the arguments', see `ScanFn`. An `update` that
    /// relies on them for memory safety must check them itself.
    fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String>;
}

/// A selection entry. It appends the selected rows of this tile to `out`, and `base` is the
/// tile's first row index in the whole array. The caller guarantees that the CPU supports
/// the entry's ISA.
///
/// Every `Rows` argument has the same number of rows, and the windows are aligned for their
/// element type only, as for `ElementwiseFn`. Rows appended to a `Bool` builder must be
/// exactly 0 or exactly 1. The output is the concatenation of the tiles' appends, so a
/// selection must not depend on how the rows are split into tiles.
pub type SelectionFn =
    unsafe fn(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String>;

/// The portable body of an elementwise kernel. It is a type rather than a function pointer
/// so that each ISA variant can instantiate the same body under its own `#[target_feature]`.
/// `run` has the contract of `ElementwiseFn` and needs no CPU feature.
pub trait ElementwiseBody: 'static {
    fn run(args: &[Arg], out: &mut [u8]) -> Result<(), String>;
}

/// The portable body of a reduction: a `Reducer` that starts from `new`, see `ElementwiseBody`.
pub trait ReductionBody: Reducer + Sized + 'static {
    fn new() -> Self;
}

/// The portable body of a scan: a `Scanner` that starts from `new`, see `ElementwiseBody`.
pub trait ScanBody: Scanner + Sized + 'static {
    fn new() -> Self;
}

/// The portable body of a whole kernel. `run` has the contract of `WholeFn` and needs no CPU
/// feature, see `ElementwiseBody`.
pub trait WholeBody: 'static {
    fn run(args: &[Value]) -> Result<Value, String>;
}

/// The portable body of a selection. `run` has the contract of `SelectionFn` and needs no
/// CPU feature, see `ElementwiseBody`.
pub trait SelectionBody: 'static {
    fn run(args: &[Arg], base: usize, out: &mut ArrayBuilder) -> Result<(), String>;
}

fn scalar_elementwise<B: ElementwiseBody>(args: &[Arg], out: &mut [u8]) -> Result<(), String> {
    B::run(args, out)
}

fn scalar_reducer<R: ReductionBody>() -> Box<dyn Reducer> {
    Box::new(R::new())
}

fn scalar_selection<S: SelectionBody>(
    args: &[Arg],
    base: usize,
    out: &mut ArrayBuilder,
) -> Result<(), String> {
    S::run(args, base, out)
}

fn scalar_scanner<S: ScanBody>() -> Box<dyn Scanner> {
    Box::new(S::new())
}

fn scalar_whole<W: WholeBody>(args: &[Value]) -> Result<Value, String> {
    W::run(args)
}

/// A kernel that sees its whole arguments at once, for class `Whole`. It checks its own
/// arguments. The caller guarantees that the CPU supports the entry's ISA, which a plain
/// function does not need.
pub type WholeFn = unsafe fn(&[Value]) -> Result<Value, String>;

/// The entry of a kernel variant. The variant of the entry matches `KernelDef::class`.
#[derive(Clone, Copy)]
pub enum KernelImpl {
    Elementwise(ElementwiseFn),
    Reduction(ReducerFn),
    Selection(SelectionFn),
    Scan(ScanFn),
    Whole(WholeFn),
}

impl KernelImpl {
    /// Whether the entry fits a kernel of `class`.
    fn fits(self, class: KernelClass) -> bool {
        matches!(
            (self, class),
            (KernelImpl::Elementwise(_), KernelClass::Elementwise)
                | (KernelImpl::Reduction(_), KernelClass::Reduction)
                | (KernelImpl::Selection(_), KernelClass::Selection)
                | (KernelImpl::Scan(_), KernelClass::Scan)
                | (KernelImpl::Whole(_), KernelClass::Whole)
        )
    }

    /// A table with only the scalar slot set.
    pub fn scalar_only(self) -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = [None; Variant::COUNT];
        impls[Variant::SCALAR.index()] = Some(self);
        impls
    }

    /// The scalar entry of an elementwise body.
    pub(crate) fn elementwise_scalar<B: ElementwiseBody>() -> KernelImpl {
        KernelImpl::Elementwise(scalar_elementwise::<B>)
    }

    /// The table of an elementwise body: its scalar entry, and its ISA variants on targets
    /// that compile them.
    pub(crate) fn elementwise<B: ElementwiseBody>() -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = KernelImpl::elementwise_scalar::<B>().scalar_only();
        arch::add_elementwise::<B>(&mut impls);
        impls
    }

    /// The table of a reduction body, see `elementwise`.
    pub(crate) fn reduction<R: ReductionBody>() -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = KernelImpl::Reduction(scalar_reducer::<R>).scalar_only();
        arch::add_reduction::<R>(&mut impls);
        impls
    }

    /// The table of a selection body, see `elementwise`.
    pub(crate) fn selection<S: SelectionBody>() -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = KernelImpl::Selection(scalar_selection::<S>).scalar_only();
        arch::add_selection::<S>(&mut impls);
        impls
    }

    /// The table of a scan body, see `elementwise`.
    pub(crate) fn scan<S: ScanBody>() -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = KernelImpl::Scan(scalar_scanner::<S>).scalar_only();
        arch::add_scan::<S>(&mut impls);
        impls
    }

    /// The table of a whole body, see `elementwise`. A whole kernel whose body is a loop over
    /// its arguments gets auto variants this way, and one that only forwards to other kernels
    /// stays `scalar_only`.
    // The tests build one, no registered kernel does yet.
    #[allow(dead_code)]
    pub(crate) fn whole<W: WholeBody>() -> [Option<KernelImpl>; Variant::COUNT] {
        let mut impls = KernelImpl::Whole(scalar_whole::<W>).scalar_only();
        arch::add_whole::<W>(&mut impls);
        impls
    }
}

/// How an ISA variant of a kernel is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Plain Rust that the compiler vectorizes for the ISA.
    Auto,
    /// Hand-written intrinsics.
    #[default]
    Intrinsics,
}

/// One implementation of a kernel: an instruction set and how it is written.
/// `Isa::Scalar` has only the scalar entry, whatever its backend says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub isa: Isa,
    pub backend: Backend,
}

impl Variant {
    /// The number of slots in `KernelDef::impls`.
    pub const COUNT: usize = 7;

    pub const SCALAR: Variant = Variant {
        isa: Isa::Scalar,
        backend: Backend::Auto,
    };

    /// Position in a `KernelDef::impls` table: scalar, then auto and intrinsics for each of
    /// AVX2, AVX-512, and NEON.
    pub fn index(self) -> usize {
        let base = match self.isa {
            Isa::Scalar => return 0,
            Isa::Avx2 => 1,
            Isa::Avx512 => 3,
            Isa::Neon => 5,
        };
        base + match self.backend {
            Backend::Auto => 0,
            Backend::Intrinsics => 1,
        }
    }
}

impl std::fmt::Display for Variant {
    /// `isa=scalar`, or `isa=avx2 backend=auto`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "isa={}", self.isa.name())?;
        if self.isa != Isa::Scalar {
            let backend = match self.backend {
                Backend::Auto => "auto",
                Backend::Intrinsics => "intrinsics",
            };
            write!(f, " backend={backend}")?;
        }
        Ok(())
    }
}

/// The expression a fused kernel implements, as an egglog pattern over single-op kernel names.
/// The planner later emits `(rewrite <lhs> (<kernel name> <vars...>))` for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedPattern {
    /// Pattern text with `?`-variables. `(Bcast ?c)` marks a scalar broadcast operand.
    pub lhs: String,
    /// The variables in the fused kernel's argument order.
    pub vars: Vec<String>,
}

/// What a kernel accepts for one argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputSpec {
    Array(ScalarType),
    Scalar(ScalarType),
    /// An array or a scalar of the type. Used for elementwise operands.
    Either(ScalarType),
    Str,
}

impl InputSpec {
    pub fn accepts(self, ty: ValueType) -> bool {
        match (self, ty) {
            (InputSpec::Array(s), ValueType::Array(t)) => s == t,
            (InputSpec::Scalar(s), ValueType::Scalar(t)) => s == t,
            (InputSpec::Either(s), ValueType::Array(t) | ValueType::Scalar(t)) => s == t,
            (InputSpec::Str, ValueType::Str) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelClass {
    Elementwise,
    Reduction,
    Selection,
    Scan,
    Whole,
}

pub struct KernelDef {
    pub name: String,
    pub class: KernelClass,
    pub op: Option<Op>,
    pub inputs: Vec<InputSpec>,
    pub output: ValueType,
    /// Bytes moved per element, with every operand treated as an array.
    pub cost: u64,
    pub commutative: bool,
    pub pattern: Option<FusedPattern>,
    /// Indexed by `Variant::index`. The scalar entry is always `Some`.
    pub impls: [Option<KernelImpl>; Variant::COUNT],
}

impl KernelDef {
    /// Whether the kernel applies to arguments of these types.
    /// Elementwise kernels with an `Either` operand are array kernels, so they need at
    /// least one array argument. All-scalar arguments resolve to the scalar-only kernel.
    pub fn accepts(&self, args: &[ValueType]) -> bool {
        self.inputs.len() == args.len()
            && self
                .inputs
                .iter()
                .zip(args)
                .all(|(spec, ty)| spec.accepts(*ty))
            && (self.class != KernelClass::Elementwise
                || !self
                    .inputs
                    .iter()
                    .any(|s| matches!(s, InputSpec::Either(_)))
                || args.iter().any(|t| matches!(t, ValueType::Array(_))))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelId(pub u32);

pub struct Registry {
    /// Identifies this registry by pointer, so that a `Dispatch` can tell which registry it was
    /// resolved from.
    token: Arc<()>,
    defs: Vec<KernelDef>,
    by_name: HashMap<String, KernelId>,
    by_op: HashMap<Op, Vec<KernelId>>,
}

impl Registry {
    /// Builds every kernel. There is no global kernel state, each call returns an independent
    /// registry.
    pub fn new() -> Registry {
        let mut reg = Registry::empty();
        scalar::elementwise::register(&mut reg);
        scalar::reduce::register(&mut reg);
        scalar::select::register(&mut reg);
        scalar::scan::register(&mut reg);
        scalar::construct::register(&mut reg);
        scalar::fused::register(&mut reg);
        reg
    }

    /// A registry without kernels.
    fn empty() -> Registry {
        Registry {
            token: Arc::new(()),
            defs: Vec::new(),
            by_name: HashMap::new(),
            by_op: HashMap::new(),
        }
    }

    /// Adds a kernel. Panics on a duplicate name, a missing scalar implementation, or an
    /// entry that does not match the class, because all are bugs in the kernel tables.
    pub(crate) fn add(&mut self, def: KernelDef) -> KernelId {
        assert!(
            def.impls[Variant::SCALAR.index()].is_some(),
            "kernel {} lacks a scalar impl",
            def.name
        );
        for imp in def.impls.iter().flatten() {
            assert!(
                imp.fits(def.class),
                "kernel {} has an entry that does not fit its class",
                def.name
            );
        }
        let id = KernelId(u32::try_from(self.defs.len()).expect("kernel count fits in u32"));
        let previous = self.by_name.insert(def.name.clone(), id);
        assert!(previous.is_none(), "duplicate kernel name {}", def.name);
        if let Some(op) = def.op {
            self.by_op.entry(op).or_default().push(id);
        }
        self.defs.push(def);
        id
    }

    /// Resolves an op on argument types to a kernel. Elementwise ops with an array
    /// argument resolve to the array kernel and all-scalar arguments to the `_s` kernel.
    /// Every other op resolves by exact argument types.
    pub fn lookup(&self, op: Op, args: &[ValueType]) -> Option<KernelId> {
        self.by_op
            .get(&op)?
            .iter()
            .copied()
            .find(|id| self.get(*id).accepts(args))
    }

    pub fn get(&self, id: KernelId) -> &KernelDef {
        &self.defs[usize::try_from(id.0).expect("u32 fits in usize")]
    }

    pub fn by_name(&self, name: &str) -> Option<KernelId> {
        self.by_name.get(name).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (KernelId, &KernelDef)> {
        self.defs.iter().enumerate().map(|(i, d)| {
            (
                KernelId(u32::try_from(i).expect("kernel count fits in u32")),
                d,
            )
        })
    }

    /// Picks one entry per kernel for `isa` and `backend` and records the variant picked.
    /// With `Intrinsics` a kernel uses its intrinsics entry, else its auto entry, else scalar.
    /// With `Auto` it uses its auto entry, else scalar. ISA entries are installed only when
    /// the CPU supports the ISA, and feature detection runs once here.
    pub fn resolve(&self, isa: Isa, backend: Backend) -> Dispatch {
        let usable = isa != Isa::Scalar && isa.is_available();
        let preference: &[Backend] = match backend {
            Backend::Intrinsics => &[Backend::Intrinsics, Backend::Auto],
            Backend::Auto => &[Backend::Auto],
        };
        let entries = self
            .defs
            .iter()
            .map(|def| {
                let picked = usable
                    .then(|| {
                        preference.iter().find_map(|&b| {
                            let variant = Variant { isa, backend: b };
                            def.impls[variant.index()].map(|imp| (imp, variant))
                        })
                    })
                    .flatten();
                picked.unwrap_or_else(|| {
                    let imp = def.impls[Variant::SCALAR.index()].expect("checked in add");
                    (imp, Variant::SCALAR)
                })
            })
            .collect();
        Dispatch::new(Arc::clone(&self.token), entries)
    }
}

impl Default for Registry {
    fn default() -> Registry {
        Registry::new()
    }
}
