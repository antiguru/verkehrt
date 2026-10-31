//! The CPU feature vocabulary and the instruction set level predicate. A level is a set of
//! `target_feature` names that an `Isa` requires, and an entry may declare extra features on
//! top of its level. Detection carries the architecture, so a level never needs a `cfg!`
//! check: no x86 feature is reported on aarch64 and `Neon` is never reported on x86-64.

use super::Isa;

macro_rules! features {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        /// A CPU feature, named by its Rust `target_feature` name.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Feature {
            $($variant),+
        }

        impl Feature {
            /// Every feature, in declaration order. The position is the feature's bit in a
            /// `FeatureSet`.
            pub const ALL: [Feature; [$(Feature::$variant),+].len()] = [$(Feature::$variant),+];

            /// The `target_feature` name of the feature.
            pub fn name(self) -> &'static str {
                match self {
                    $(Feature::$variant => $name),+
                }
            }

            /// The feature with this `target_feature` name.
            pub fn from_name(name: &str) -> Option<Feature> {
                match name {
                    $($name => Some(Feature::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

features! {
    Avx => "avx",
    Avx2 => "avx2",
    Bmi1 => "bmi1",
    Bmi2 => "bmi2",
    F16c => "f16c",
    Fma => "fma",
    Lzcnt => "lzcnt",
    Movbe => "movbe",
    Xsave => "xsave",
    Cmpxchg16b => "cmpxchg16b",
    Popcnt => "popcnt",
    Sse3 => "sse3",
    Sse41 => "sse4.1",
    Sse42 => "sse4.2",
    Ssse3 => "ssse3",
    Avx512f => "avx512f",
    Avx512bw => "avx512bw",
    Avx512cd => "avx512cd",
    Avx512dq => "avx512dq",
    Avx512vl => "avx512vl",
    Avx512vbmi2 => "avx512vbmi2",
    Neon => "neon",
}

/// A set of CPU features, one bit per `Feature::ALL` position in a `u64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FeatureSet(u64);

// A feature past the 64th would overflow the bit set.
const _: () = assert!(Feature::ALL.len() <= 64);

impl FeatureSet {
    pub const EMPTY: FeatureSet = FeatureSet(0);

    /// The set of exactly these features.
    pub const fn of(features: &[Feature]) -> FeatureSet {
        let mut bits = 0u64;
        let mut i = 0;
        while i < features.len() {
            bits |= 1 << features[i] as u64;
            i += 1;
        }
        FeatureSet(bits)
    }

    pub fn contains(self, feature: Feature) -> bool {
        self.0 & (1 << feature as u64) != 0
    }

    /// Whether every feature of `other` is in `self`.
    pub fn contains_all(self, other: FeatureSet) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn union(self, other: FeatureSet) -> FeatureSet {
        FeatureSet(self.0 | other.0)
    }

    /// The features of `self` that are not in `other`.
    pub fn minus(self, other: FeatureSet) -> FeatureSet {
        FeatureSet(self.0 & !other.0)
    }

    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL.into_iter().filter(move |f| self.contains(*f))
    }

    /// Parses a comma-separated list of `target_feature` names, `None` on an unknown name.
    #[cfg(test)]
    pub fn parse(s: &str) -> Option<FeatureSet> {
        s.split(',')
            .filter(|name| !name.is_empty())
            .map(Feature::from_name)
            .try_fold(FeatureSet::EMPTY, |set, f| {
                f.map(|f| set.union(FeatureSet::of(&[f])))
            })
    }

    /// The features this CPU supports. Only features of the compile-time architecture are
    /// reported, and other architectures report none.
    pub fn detect() -> FeatureSet {
        Feature::ALL
            .into_iter()
            .filter(|f| detect_one(*f))
            .fold(FeatureSet::EMPTY, |set, f| set.union(FeatureSet::of(&[f])))
    }
}

/// Whether the CPU supports `feature`. The detection macros need literal names, so this
/// matches on the variant.
#[cfg(target_arch = "x86_64")]
fn detect_one(feature: Feature) -> bool {
    use std::arch::is_x86_feature_detected as d;
    match feature {
        Feature::Avx => d!("avx"),
        Feature::Avx2 => d!("avx2"),
        Feature::Bmi1 => d!("bmi1"),
        Feature::Bmi2 => d!("bmi2"),
        Feature::F16c => d!("f16c"),
        Feature::Fma => d!("fma"),
        Feature::Lzcnt => d!("lzcnt"),
        Feature::Movbe => d!("movbe"),
        Feature::Xsave => d!("xsave"),
        Feature::Cmpxchg16b => d!("cmpxchg16b"),
        Feature::Popcnt => d!("popcnt"),
        Feature::Sse3 => d!("sse3"),
        Feature::Sse41 => d!("sse4.1"),
        Feature::Sse42 => d!("sse4.2"),
        Feature::Ssse3 => d!("ssse3"),
        Feature::Avx512f => d!("avx512f"),
        Feature::Avx512bw => d!("avx512bw"),
        Feature::Avx512cd => d!("avx512cd"),
        Feature::Avx512dq => d!("avx512dq"),
        Feature::Avx512vl => d!("avx512vl"),
        Feature::Avx512vbmi2 => d!("avx512vbmi2"),
        Feature::Neon => false,
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_one(feature: Feature) -> bool {
    feature == Feature::Neon && std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_one(_feature: Feature) -> bool {
    false
}

/// The features of the x86-64-v3 level, without `lahfsahf`, which is unstable in Rust.
const V3: FeatureSet = FeatureSet::of(&[
    Feature::Avx,
    Feature::Avx2,
    Feature::Bmi1,
    Feature::Bmi2,
    Feature::F16c,
    Feature::Fma,
    Feature::Lzcnt,
    Feature::Movbe,
    Feature::Xsave,
    Feature::Cmpxchg16b,
    Feature::Popcnt,
    Feature::Sse3,
    Feature::Sse41,
    Feature::Sse42,
    Feature::Ssse3,
]);

/// The features of the x86-64-v4 level: v3 plus the AVX-512 foundation subsets.
const V4: FeatureSet = V3.union(FeatureSet::of(&[
    Feature::Avx512f,
    Feature::Avx512bw,
    Feature::Avx512cd,
    Feature::Avx512dq,
    Feature::Avx512vl,
]));

/// The features that an entry of this instruction set requires.
pub fn level(isa: Isa) -> FeatureSet {
    match isa {
        Isa::Scalar => FeatureSet::EMPTY,
        Isa::Neon => FeatureSet::of(&[Feature::Neon]),
        Isa::X86_64V3 => V3,
        Isa::X86_64V4 => V4,
    }
}

/// Whether an entry of `isa` that declares `declared` extra features may run when the CPU
/// supports `available`. `Scalar` is always runnable.
pub fn runnable(isa: Isa, declared: FeatureSet, available: FeatureSet) -> bool {
    available.contains_all(level(isa).union(declared))
}

/// The best instruction set level that `available` fully covers.
pub fn best_level(available: FeatureSet) -> Isa {
    [Isa::X86_64V4, Isa::X86_64V3, Isa::Neon]
        .into_iter()
        .find(|isa| runnable(*isa, FeatureSet::EMPTY, available))
        .unwrap_or(Isa::Scalar)
}

#[cfg(test)]
mod tests;
