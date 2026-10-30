//! The kernel registry. It is the source of truth for which primitives exist on which
//! types (an op on a type without a kernel is a type error) and the dispatch table.

use std::collections::HashMap;

pub use crate::ops::Isa;
use crate::ops::Op;
use crate::value::{ScalarType, Value, ValueType};

pub mod scalar;

#[cfg(test)]
mod tests;

/// Runs a kernel. The `String` is the runtime error message, the executor adds the span.
pub type KernelFn = fn(&[Value]) -> Result<Value, String>;

/// Defined by the fusion task. Until then no kernel has a pattern.
pub type FusedPattern = ();

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
    /// Indexed by `Isa::index`. The scalar entry is always `Some`.
    pub impls: [Option<KernelFn>; 4],
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
    defs: Vec<KernelDef>,
    by_name: HashMap<String, KernelId>,
    by_op: HashMap<Op, Vec<KernelId>>,
}

impl Registry {
    /// Builds every kernel. There is no global state, each call returns an independent registry.
    pub fn new() -> Registry {
        let mut reg = Registry {
            defs: Vec::new(),
            by_name: HashMap::new(),
            by_op: HashMap::new(),
        };
        scalar::elementwise::register(&mut reg);
        scalar::reduce::register(&mut reg);
        scalar::select::register(&mut reg);
        scalar::construct::register(&mut reg);
        reg
    }

    /// Adds a kernel. Panics on a duplicate name or a missing scalar implementation,
    /// because both are bugs in the kernel tables.
    pub(crate) fn add(&mut self, def: KernelDef) -> KernelId {
        assert!(
            def.impls[Isa::Scalar.index()].is_some(),
            "kernel {} lacks a scalar impl",
            def.name
        );
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

    /// The implementation for `isa`, falling back to scalar when there is none or the CPU
    /// cannot run it. Returns the instruction set actually used.
    pub fn select_impl(&self, id: KernelId, isa: Isa) -> (KernelFn, Isa) {
        let def = self.get(id);
        if isa.is_available()
            && let Some(f) = def.impls[isa.index()]
        {
            return (f, isa);
        }
        (
            def.impls[Isa::Scalar.index()].expect("checked in add"),
            Isa::Scalar,
        )
    }
}

impl Default for Registry {
    fn default() -> Registry {
        Registry::new()
    }
}
