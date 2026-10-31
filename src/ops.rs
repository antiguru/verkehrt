use crate::value::ScalarType;

/// Associative folds that `Op::Reduce`, `Op::Scan`, and `Op::ScanBy` can apply.
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
    Avx2,
    Avx512,
    Neon,
}

impl Isa {
    pub const ALL: [Isa; 4] = [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon];

    /// The best instruction set this CPU supports.
    pub fn detect() -> Isa {
        [Isa::Avx512, Isa::Avx2, Isa::Neon]
            .into_iter()
            .find(|isa| isa.is_available())
            .unwrap_or(Isa::Scalar)
    }

    /// Whether this CPU can run code compiled for the instruction set.
    pub fn is_available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512vbmi2")
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => true,
            // The remaining variants belong to other architectures.
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn from_name(name: &str) -> Option<Isa> {
        Isa::ALL.into_iter().find(|isa| isa.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Neon => "neon",
        }
    }
}
