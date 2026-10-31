use crate::kernels::FeatureSet;
use crate::kernels::features;
use crate::value::ScalarType;

/// Associative folds that `Op::Reduce`, `Op::Scan`, `Op::ScanBy`, and `Op::ReduceInto` can apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// A primitive operation. Each op resolves to a kernel through `Registry::lookup`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Min,
    Max,
    Sqrt,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    Not,
    Select,
    /// Cast to the given scalar type.
    Cast(ScalarType),
    Reduce(ReduceOp),
    /// The running fold of a whole array, one output row per input row.
    Scan(ReduceOp),
    /// `Scan` that restarts at every row where the `bool` operand is true.
    ScanBy(ReduceOp),
    Count,
    Any,
    All,
    ArgMin,
    ArgMax,
    Compress,
    Where,
    /// Each row's predecessor, with the second operand in row 0.
    Prev,
    Iota,
    Fill,
    /// The rows of the first operand at the positions of the `i64` second operand.
    Gather,
    /// The rows `start..end` of the first operand as a view, with `start` and `end` the
    /// `i64` scalar operands.
    Slice,
    /// The number of occurrences of each key in `0..n`, with the `i64` scalar operand `n`.
    Histogram,
    /// Truncating division of the first operand by the scalar second operand, which is never
    /// zero. It wraps, so `MIN / -1` is `MIN`.
    DivConst,
    /// The positions of the rows of the operand in stable ascending order, as `i64`.
    ArgSort,
    /// The fold of the values per key in `0..n`, an empty bucket holding the identity of the
    /// fold. The operands are the keys, the values, and the `i64` scalar `n`.
    ReduceInto(ReduceOp),
    RandI64,
    RandF64,
    /// Load a file as an array of the given element type.
    Load(ScalarType),
    Length,
    /// The array operand itself when its length equals the `i64` operand, a length
    /// mismatch error otherwise. Lowering inserts it, it has no source syntax.
    SameLen,
}

/// An instruction set a kernel implementation can target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    Scalar,
    X86_64V3,
    X86_64V4,
    Neon,
}

impl Isa {
    pub const ALL: [Isa; 4] = [Isa::Scalar, Isa::X86_64V3, Isa::X86_64V4, Isa::Neon];

    /// The best instruction set this CPU supports.
    pub fn detect() -> Isa {
        Isa::detect_with(FeatureSet::EMPTY)
    }

    /// The best instruction set this CPU supports with the features of `disable` treated as
    /// absent.
    pub fn detect_with(disable: FeatureSet) -> Isa {
        features::best_level(FeatureSet::detect().minus(disable))
    }

    /// Whether this CPU supports every feature of the instruction set's level, see
    /// `features::runnable`.
    pub fn is_available(self) -> bool {
        self.is_available_with(FeatureSet::EMPTY)
    }

    /// Like `is_available`, with the features of `disable` treated as absent.
    pub fn is_available_with(self, disable: FeatureSet) -> bool {
        features::runnable(self, FeatureSet::EMPTY, FeatureSet::detect().minus(disable))
    }

    pub fn from_name(name: &str) -> Option<Isa> {
        Isa::ALL.into_iter().find(|isa| isa.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::X86_64V3 => "x86-64-v3",
            Isa::X86_64V4 => "x86-64-v4",
            Isa::Neon => "neon",
        }
    }
}
