//! Input values and literal texts: the value styles, the seeded element generator, and
//! source text for values in a reproduction.

use verkehrt::value::{Array, Scalar, ScalarType, Value};

use super::Gen;

/// Literal texts per type, the simplest first. Negative literals are parenthesized.
pub(super) fn literals(t: ScalarType) -> &'static [&'static str] {
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

/// A SplitMix64 generator, for input values from a seed.
pub(super) struct SplitMix(u64);

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

pub(super) const F64_SPECIAL: [f64; 14] = [
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

pub(super) const F32_SPECIAL: [f32; 14] = [
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
pub(super) fn element(t: ScalarType, style: usize, rng: &mut SplitMix) -> u64 {
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

pub(super) fn scalar_of(t: ScalarType, bits: u64) -> Scalar {
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
pub(super) fn scalar_literal(s: &Scalar) -> String {
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
pub(super) fn literal_of(value: &Value) -> String {
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

impl Gen<'_, '_> {
    pub(super) fn input_len(&mut self) -> usize {
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
    pub(super) fn input_value(&mut self, t: ScalarType, len: usize, array: bool) -> Value {
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
}
