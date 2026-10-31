use std::fmt;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl ScalarType {
    /// Element types that array kernels exist for: the v1 types and `u64`.
    pub const V1_ARRAY: [ScalarType; 7] = [
        ScalarType::F64,
        ScalarType::F32,
        ScalarType::I64,
        ScalarType::I32,
        ScalarType::U8,
        ScalarType::U64,
        ScalarType::Bool,
    ];

    const ALL: [ScalarType; 11] = [
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::U8,
        ScalarType::U16,
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::Bool,
    ];

    /// Storage width in bytes. Bool is stored as one byte.
    pub fn width(self) -> usize {
        match self {
            ScalarType::I8 | ScalarType::U8 | ScalarType::Bool => 1,
            ScalarType::I16 | ScalarType::U16 => 2,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::U64 | ScalarType::F64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::U8 => "u8",
            ScalarType::U16 => "u16",
            ScalarType::U32 => "u32",
            ScalarType::U64 => "u64",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::Bool => "bool",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    pub fn is_int(self) -> bool {
        !self.is_float() && self != ScalarType::Bool
    }

    /// Floats are signed, bool is not.
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::F32
                | ScalarType::F64
        )
    }

    pub fn from_name(name: &str) -> Option<ScalarType> {
        ScalarType::ALL.into_iter().find(|t| t.name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar(ScalarType),
    Array(ScalarType),
    Str,
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Scalar(t) => f.write_str(t.name()),
            ValueType::Array(t) => write!(f, "{}[]", t.name()),
            ValueType::Str => f.write_str("str"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Scalar {
    pub fn ty(&self) -> ScalarType {
        match self {
            Scalar::I8(_) => ScalarType::I8,
            Scalar::I16(_) => ScalarType::I16,
            Scalar::I32(_) => ScalarType::I32,
            Scalar::I64(_) => ScalarType::I64,
            Scalar::U8(_) => ScalarType::U8,
            Scalar::U16(_) => ScalarType::U16,
            Scalar::U32(_) => ScalarType::U32,
            Scalar::U64(_) => ScalarType::U64,
            Scalar::F32(_) => ScalarType::F32,
            Scalar::F64(_) => ScalarType::F64,
            Scalar::Bool(_) => ScalarType::Bool,
        }
    }
}

impl Scalar {
    /// The scalar of type `ty` stored in the first `ty.width()` bytes of `bytes`. Bool reads
    /// as nonzero. Panics when `bytes` is shorter than the type's width.
    pub fn from_bytes(ty: ScalarType, bytes: &[u8]) -> Scalar {
        let b = &bytes[..ty.width()];
        match ty {
            ScalarType::I8 => Scalar::I8(bytemuck::pod_read_unaligned(b)),
            ScalarType::I16 => Scalar::I16(bytemuck::pod_read_unaligned(b)),
            ScalarType::I32 => Scalar::I32(bytemuck::pod_read_unaligned(b)),
            ScalarType::I64 => Scalar::I64(bytemuck::pod_read_unaligned(b)),
            ScalarType::U8 => Scalar::U8(bytemuck::pod_read_unaligned(b)),
            ScalarType::U16 => Scalar::U16(bytemuck::pod_read_unaligned(b)),
            ScalarType::U32 => Scalar::U32(bytemuck::pod_read_unaligned(b)),
            ScalarType::U64 => Scalar::U64(bytemuck::pod_read_unaligned(b)),
            ScalarType::F32 => Scalar::F32(bytemuck::pod_read_unaligned(b)),
            ScalarType::F64 => Scalar::F64(bytemuck::pod_read_unaligned(b)),
            ScalarType::Bool => Scalar::Bool(b[0] != 0),
        }
    }
}

impl fmt::Display for Scalar {
    /// Floats use `{:?}` so they always show a decimal point (`1.0`, `NaN`, `-0.0`, `inf`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::I8(v) => write!(f, "{v}"),
            Scalar::I16(v) => write!(f, "{v}"),
            Scalar::I32(v) => write!(f, "{v}"),
            Scalar::I64(v) => write!(f, "{v}"),
            Scalar::U8(v) => write!(f, "{v}"),
            Scalar::U16(v) => write!(f, "{v}"),
            Scalar::U32(v) => write!(f, "{v}"),
            Scalar::U64(v) => write!(f, "{v}"),
            Scalar::F32(v) => write!(f, "{v:?}"),
            Scalar::F64(v) => write!(f, "{v:?}"),
            Scalar::Bool(v) => write!(f, "{v}"),
        }
    }
}

/// A Rust numeric type that can be stored in an `Array`.
/// Bool arrays are stored as `u8` with values 0 and 1 and have no `Elem` impl of their own.
pub trait Elem: Pod + Copy + Send + Sync + 'static {
    const TYPE: ScalarType;
}

macro_rules! impl_elem {
    ($($t:ty => $v:ident),*) => {
        $(impl Elem for $t {
            const TYPE: ScalarType = ScalarType::$v;
        })*
    };
}

impl_elem!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    f32 => F32, f64 => F64
);

/// One cache line. The block alignment gives every buffer a 64-byte aligned start
/// without `unsafe`, because `Vec<Block>` allocates with the block's alignment.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Block([u8; 64]);

/// An immutable, reference-counted view into a 64-byte aligned buffer.
#[derive(Clone)]
pub struct Array {
    buf: Arc<Vec<Block>>,
    /// Offset into the buffer, in elements.
    offset: usize,
    len: usize,
    elem: ScalarType,
}

impl Array {
    pub fn elem(&self) -> ScalarType {
        self.elem
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len * self.elem.width()
    }

    pub fn ty(&self) -> ValueType {
        ValueType::Array(self.elem)
    }

    /// Zero-copy sub-view. Panics when `start + len` exceeds the array length.
    pub fn slice(&self, start: usize, len: usize) -> Array {
        assert!(
            start.checked_add(len).is_some_and(|end| end <= self.len),
            "slice {start}+{len} out of range for length {}",
            self.len
        );
        Array {
            buf: Arc::clone(&self.buf),
            offset: self.offset + start,
            len,
            elem: self.elem,
        }
    }

    fn bytes(&self) -> &[u8] {
        let w = self.elem.width();
        let all: &[u8] = bytemuck::cast_slice(self.buf.as_slice());
        &all[self.offset * w..(self.offset + self.len) * w]
    }

    /// Typed view of the elements. Panics when `T` does not match the element type.
    pub fn as_slice<T: Elem>(&self) -> &[T] {
        assert_eq!(T::TYPE, self.elem, "as_slice type mismatch");
        bytemuck::cast_slice(self.bytes())
    }

    /// Bool storage, one byte per element holding 0 or 1. Panics unless the array is `Bool`.
    pub fn as_bool_bytes(&self) -> &[u8] {
        assert_eq!(
            self.elem,
            ScalarType::Bool,
            "as_bool_bytes on non-bool array"
        );
        self.bytes()
    }

    /// The byte size of `len` elements of `elem`. Fails with "array of N elements is too
    /// large" when it exceeds `isize::MAX`, the largest allocation Rust permits.
    pub fn byte_size(len: usize, elem: ScalarType) -> Result<usize, String> {
        len.checked_mul(elem.width())
            .filter(|&b| isize::try_from(b).is_ok())
            .ok_or_else(|| format!("array of {len} elements is too large"))
    }

    /// Panics when the size exceeds `isize::MAX` bytes or cannot be allocated, see
    /// `try_zeroed`.
    fn zeroed(len: usize, elem: ScalarType) -> Vec<Block> {
        Array::try_zeroed(len, elem).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A zeroed buffer for `len` elements of `elem`. Fails when the size exceeds `isize::MAX`
    /// bytes, see `byte_size`, or when the allocator cannot provide it. `Vec::try_reserve_exact`
    /// reports the failure where `vec!` would abort the process.
    fn try_zeroed(len: usize, elem: ScalarType) -> Result<Vec<Block>, String> {
        let blocks = Array::byte_size(len, elem)?.div_ceil(64);
        let mut buf: Vec<Block> = Vec::new();
        buf.try_reserve_exact(blocks)
            .map_err(|_| format!("cannot allocate an array of {len} elements"))?;
        buf.resize(blocks, Block([0; 64]));
        Ok(buf)
    }

    pub fn from_vec<T: Elem>(v: Vec<T>) -> Array {
        Array::build(v.len(), |out: &mut [T]| out.copy_from_slice(&v))
    }

    pub fn from_bools(v: &[bool]) -> Array {
        Array::build_bool(v.len(), |out| {
            for (o, b) in out.iter_mut().zip(v) {
                *o = u8::from(*b);
            }
        })
    }

    /// Like `build`, but fails instead of panicking when `len` elements exceed `isize::MAX`
    /// bytes or the allocation fails. For lengths that come from program values.
    // Inlined so that `f` compiles in the caller's context: a kernel body that fills its
    // output here inside a `#[target_feature]` entry is then vectorized for that entry's ISA.
    #[inline(always)]
    pub fn try_build<T: Elem>(len: usize, f: impl FnOnce(&mut [T])) -> Result<Array, String> {
        let mut blocks = Array::try_zeroed(len, T::TYPE)?;
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(blocks.as_mut_slice());
        f(bytemuck::cast_slice_mut(
            &mut bytes[..len * T::TYPE.width()],
        ));
        Ok(Array {
            buf: Arc::new(blocks),
            offset: 0,
            len,
            elem: T::TYPE,
        })
    }

    /// Allocates a zeroed aligned buffer of `len` elements and lets `f` fill it. Panics
    /// when `len` elements exceed `isize::MAX` bytes or cannot be allocated.
    pub fn build<T: Elem>(len: usize, f: impl FnOnce(&mut [T])) -> Array {
        Array::try_build(len, f).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Like `build` for `Bool`. The caller must write only 0 or 1.
    pub fn build_bool(len: usize, f: impl FnOnce(&mut [u8])) -> Array {
        let mut blocks = Array::zeroed(len, ScalarType::Bool);
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(blocks.as_mut_slice());
        f(&mut bytes[..len]);
        Array {
            buf: Arc::new(blocks),
            offset: 0,
            len,
            elem: ScalarType::Bool,
        }
    }

    /// Like `build`, but `f` fills the raw bytes of the `len` elements of `elem` and may fail.
    /// The buffer is zeroed and 64-byte aligned. Fails instead of panicking when the size
    /// exceeds `isize::MAX` bytes or the allocation fails. For `Bool`, `f` must write only 0 or 1.
    // Inlined for the reason given at `try_build`.
    #[inline(always)]
    pub fn build_bytes(
        len: usize,
        elem: ScalarType,
        f: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Array, String> {
        let size = Array::byte_size(len, elem)?;
        let mut blocks = Array::try_zeroed(len, elem)?;
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(blocks.as_mut_slice());
        f(&mut bytes[..size])?;
        Ok(Array {
            buf: Arc::new(blocks),
            offset: 0,
            len,
            elem,
        })
    }

    /// The bytes the underlying buffer can hold without reallocating, for tests of how much
    /// capacity an array retains.
    #[cfg(test)]
    pub(crate) fn buffer_capacity_bytes(&self) -> usize {
        self.buf.capacity() * size_of::<Block>()
    }

    /// Element `i` as a scalar. Panics when out of range.
    pub fn get(&self, i: usize) -> Scalar {
        assert!(
            i < self.len,
            "index {i} out of range for length {}",
            self.len
        );
        match self.elem {
            ScalarType::I8 => Scalar::I8(self.as_slice::<i8>()[i]),
            ScalarType::I16 => Scalar::I16(self.as_slice::<i16>()[i]),
            ScalarType::I32 => Scalar::I32(self.as_slice::<i32>()[i]),
            ScalarType::I64 => Scalar::I64(self.as_slice::<i64>()[i]),
            ScalarType::U8 => Scalar::U8(self.as_slice::<u8>()[i]),
            ScalarType::U16 => Scalar::U16(self.as_slice::<u16>()[i]),
            ScalarType::U32 => Scalar::U32(self.as_slice::<u32>()[i]),
            ScalarType::U64 => Scalar::U64(self.as_slice::<u64>()[i]),
            ScalarType::F32 => Scalar::F32(self.as_slice::<f32>()[i]),
            ScalarType::F64 => Scalar::F64(self.as_slice::<f64>()[i]),
            ScalarType::Bool => Scalar::Bool(self.as_bool_bytes()[i] != 0),
        }
    }
}

/// One operand of a kernel entry for one tile: rows of an array or one value repeated for
/// every row. `Rows` is a view, so a tile is `array.slice(start, len)`.
#[derive(Clone, Debug)]
pub enum Arg {
    Rows(Array),
    Splat(Scalar),
}

/// A growable 64-byte aligned buffer of rows of one element type. Selection kernels append
/// the rows they keep, and `finish` turns the buffer into an `Array` without copying.
pub struct ArrayBuilder {
    blocks: Vec<Block>,
    /// Rows appended so far.
    len: usize,
    /// Rows past `len` handed out by the last `spare_rows` that `commit` may still append.
    spare: usize,
    elem: ScalarType,
}

impl ArrayBuilder {
    pub fn new(elem: ScalarType) -> ArrayBuilder {
        ArrayBuilder::with_capacity(elem, 0)
    }

    /// Reserves room for `rows` rows. Panics when that exceeds `isize::MAX` bytes.
    pub fn with_capacity(elem: ScalarType, rows: usize) -> ArrayBuilder {
        let bytes = Array::byte_size(rows, elem).unwrap_or_else(|e| panic!("{e}"));
        ArrayBuilder {
            blocks: Vec::with_capacity(bytes.div_ceil(64)),
            len: 0,
            spare: 0,
            elem,
        }
    }

    /// Appends rows. Panics when `T` does not match the element type, where bool takes `u8`.
    /// Ends any reservation of `spare_rows`.
    pub fn extend<T: Elem>(&mut self, rows: &[T]) {
        self.window::<T>("extend", rows.len(), 0)
            .copy_from_slice(rows);
        self.len += rows.len();
    }

    /// Reserves capacity for at least `rows` more rows without writing them, so that later
    /// appends up to that size do not move the buffer. Panics when that exceeds `isize::MAX`
    /// bytes.
    pub fn reserve(&mut self, rows: usize) {
        let end = self
            .len
            .checked_add(rows)
            .ok_or_else(|| format!("array of more than {} elements is too large", usize::MAX))
            .and_then(|rows| Array::byte_size(rows, self.elem))
            .unwrap_or_else(|e| panic!("{e}"));
        self.blocks
            .reserve(end.div_ceil(64).saturating_sub(self.blocks.len()));
    }

    /// The `n` rows past the end, for a selection to write its survivors into before it
    /// appends a prefix of them with `commit`. The rows hold unspecified values, and they are
    /// aligned for `T` like every row of the builder. Panics when `T` does not match the
    /// element type, where bool takes `u8`, or when the rows would exceed `isize::MAX` bytes.
    /// A `Bool` builder must only be committed rows of 0 or 1.
    pub fn spare_rows<T: Elem>(&mut self, n: usize) -> &mut [T] {
        self.window::<T>("spare_rows", n, n)
    }

    /// Appends the first `n` rows of the last `spare_rows` and ends that reservation. Panics
    /// when `n` exceeds the rows it handed out, or when an `extend` or `commit` came between.
    pub fn commit(&mut self, n: usize) {
        assert!(
            n <= self.spare,
            "commit of {n} rows exceeds the {} spare rows",
            self.spare
        );
        self.len += n;
        self.spare = 0;
    }

    /// Grows the buffer to hold `n` rows past the end and returns them, with `spare` rows of
    /// them reserved for `commit`. Panics on a type mismatch, see `extend`.
    fn window<T: Elem>(&mut self, what: &str, n: usize, spare: usize) -> &mut [T] {
        let matches =
            T::TYPE == self.elem || (self.elem == ScalarType::Bool && T::TYPE == ScalarType::U8);
        assert!(
            matches,
            "{what} with {} rows on a {} builder",
            T::TYPE.name(),
            self.elem.name()
        );
        self.spare = spare;
        let w = self.elem.width();
        let start = self.len * w;
        let end = self
            .len
            .checked_add(n)
            .ok_or_else(|| format!("array of more than {} elements is too large", usize::MAX))
            .and_then(|rows| Array::byte_size(rows, self.elem))
            .unwrap_or_else(|e| panic!("{e}"));
        // `Vec::resize` grows amortized and zero fills new blocks. Rows that an earlier
        // `spare_rows` handed out but `commit` did not append may hold any value, and
        // `finish` zeroes the bytes past the end.
        let blocks = end.div_ceil(64);
        if self.blocks.len() < blocks {
            self.blocks.resize(blocks, Block([0; 64]));
        }
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(self.blocks.as_mut_slice());
        bytemuck::cast_slice_mut(&mut bytes[start..end])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The element type of the rows.
    pub fn elem(&self) -> ScalarType {
        self.elem
    }

    /// Turns the buffer into an array without copying. The bytes past the last row are zero.
    pub fn finish(mut self) -> Array {
        let end = self.len * self.elem.width();
        self.blocks.truncate(end.div_ceil(64));
        // Selections reserve their rows plus one vector, so this only catches other uses that
        // over-reserve. Shrinking copies the rows, because the system allocator cannot resize
        // a 64-byte aligned block in place. The buffer shrinks whenever more than half of it
        // is unused, so an array never holds more than twice its blocks, and the copy costs
        // at most the output size.
        if self.blocks.capacity() / 2 > self.blocks.len() {
            self.blocks.shrink_to_fit();
        }
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(self.blocks.as_mut_slice());
        bytes[end..].fill(0);
        Array {
            buf: Arc::new(self.blocks),
            offset: 0,
            len: self.len,
            elem: self.elem,
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array<{}>{self}", self.elem.name())
    }
}

impl PartialEq for Array {
    /// Element-wise on `Scalar` equality, so float NaN is unequal to itself.
    fn eq(&self, other: &Array) -> bool {
        self.elem == other.elem
            && self.len == other.len
            && (0..self.len).all(|i| self.get(i) == other.get(i))
    }
}

impl fmt::Display for Array {
    /// Prints up to 20 elements, otherwise the first 10 and a count of the rest.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shown = if self.len <= 20 { self.len } else { 10 };
        f.write_str("[")?;
        for i in 0..shown {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", self.get(i))?;
        }
        if shown < self.len {
            write!(f, ", ... {} more", self.len - shown)?;
        }
        f.write_str("]")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Scalar(Scalar),
    Array(Array),
    Str(Arc<str>),
}

impl Value {
    pub fn ty(&self) -> ValueType {
        match self {
            Value::Scalar(s) => ValueType::Scalar(s.ty()),
            Value::Array(a) => a.ty(),
            Value::Str(_) => ValueType::Str,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Scalar(s) => s.fmt(f),
            Value::Array(a) => a.fmt(f),
            Value::Str(s) => f.write_str(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_64_byte_aligned() {
        for len in [0usize, 1, 7, 100] {
            let a = Array::from_vec((0..len as i64).collect::<Vec<_>>());
            assert_eq!(a.as_slice::<i64>().as_ptr() as usize % 64, 0);
        }
    }

    #[test]
    fn slice_is_zero_copy_view() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let s = a.slice(1, 2);
        assert_eq!(s.as_slice::<f64>(), &[2.0, 3.0]);
        assert_eq!(
            s.as_slice::<f64>().as_ptr(),
            a.as_slice::<f64>()[1..].as_ptr()
        );
    }

    #[test]
    fn bools_are_bytes() {
        let a = Array::from_bools(&[true, false, true]);
        assert_eq!(a.as_bool_bytes(), &[1, 0, 1]);
        assert_eq!(a.ty(), ValueType::Array(ScalarType::Bool));
    }

    #[test]
    fn display_formats() {
        assert_eq!(Value::Scalar(Scalar::F64(1.0)).to_string(), "1.0");
        assert_eq!(Value::Scalar(Scalar::Bool(true)).to_string(), "true");
        assert_eq!(
            Value::Array(Array::from_vec(vec![1i64, 2])).to_string(),
            "[1, 2]"
        );
        assert_eq!(
            Value::Array(Array::from_vec(Vec::<f64>::new())).to_string(),
            "[]"
        );
        let long = Value::Array(Array::from_vec((0..25i64).collect::<Vec<_>>()));
        assert_eq!(
            long.to_string(),
            "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... 15 more]"
        );
    }

    #[test]
    fn byte_size_is_capped_at_isize_max() {
        assert_eq!(Array::byte_size(3, ScalarType::F64), Ok(24));
        let half = usize::try_from(isize::MAX).unwrap() / 2 + 1;
        assert_eq!(Array::byte_size(half, ScalarType::U8), Ok(half));
        assert!(Array::byte_size(half, ScalarType::I16).is_err());
        let e = Array::byte_size(usize::MAX, ScalarType::F64).unwrap_err();
        assert_eq!(e, format!("array of {} elements is too large", usize::MAX));
        assert!(Array::try_build::<i64>(usize::MAX, |_| {}).is_err());
    }

    #[test]
    fn array_builder_is_aligned_and_typed() {
        let mut b = ArrayBuilder::new(ScalarType::F64);
        b.extend(&[1.0f64, 2.0, 3.0]);
        b.extend(&(0..70).map(f64::from).collect::<Vec<_>>());
        assert_eq!(b.len(), 73);
        let a = b.finish();
        assert_eq!(a.len(), 73);
        assert_eq!(a.as_slice::<f64>()[..4], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(a.as_slice::<f64>()[72], 69.0);
        assert_eq!(a.as_slice::<f64>().as_ptr() as usize % 64, 0);
        let mut bools = ArrayBuilder::new(ScalarType::Bool);
        bools.extend(&[1u8, 0]);
        assert_eq!(bools.finish().as_bool_bytes(), &[1, 0]);
    }

    #[test]
    #[should_panic(expected = "extend with i64 rows on a f64 builder")]
    fn array_builder_rejects_other_types() {
        ArrayBuilder::new(ScalarType::F64).extend(&[1i64]);
    }

    #[test]
    fn array_builder_spare_rows_commit_a_prefix() {
        let mut b = ArrayBuilder::new(ScalarType::I64);
        b.extend(&[7i64]);
        let spare = b.spare_rows::<i64>(70);
        assert_eq!(spare.len(), 70);
        assert_eq!(spare.as_ptr() as usize % 8, 0);
        for (i, s) in spare.iter_mut().enumerate() {
            *s = i64::try_from(i).unwrap() + 100;
        }
        b.commit(3);
        assert_eq!(b.len(), 4);
        // Spare rows that were not committed are overwritten by the next reservation.
        b.spare_rows::<i64>(2)[0] = -1;
        b.commit(1);
        b.extend(&[9i64]);
        let a = b.finish();
        assert_eq!(a.as_slice::<i64>(), &[7, 100, 101, 102, -1, 9]);
        assert_eq!(a.as_slice::<i64>().as_ptr() as usize % 64, 0);
        // `finish` zeroes the rows past the end, so the garbage of the spare rows is gone.
        let all: &[u8] = bytemuck::cast_slice(a.buf.as_slice());
        assert_eq!(all.len(), 64);
        assert!(all[6 * 8..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn array_builder_spare_rows_of_bool_take_u8() {
        let mut b = ArrayBuilder::new(ScalarType::Bool);
        b.spare_rows::<u8>(4).copy_from_slice(&[1, 0, 1, 1]);
        b.commit(2);
        assert_eq!(b.finish().as_bool_bytes(), &[1, 0]);
    }

    #[test]
    #[should_panic(expected = "commit of 3 rows exceeds the 2 spare rows")]
    fn array_builder_commit_is_bounded_by_the_spare_rows() {
        let mut b = ArrayBuilder::new(ScalarType::F64);
        b.spare_rows::<f64>(2);
        b.commit(3);
    }

    #[test]
    #[should_panic(expected = "commit of 1 rows exceeds the 0 spare rows")]
    fn array_builder_extend_ends_the_reservation() {
        let mut b = ArrayBuilder::new(ScalarType::F64);
        b.spare_rows::<f64>(2);
        b.extend(&[1.0f64]);
        b.commit(1);
    }

    #[test]
    fn value_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Value>();
    }

    #[test]
    fn scalar_type_helpers() {
        assert_eq!(ScalarType::from_name("f64"), Some(ScalarType::F64));
        assert_eq!(ScalarType::from_name("nope"), None);
        assert_eq!(ScalarType::Bool.width(), 1);
        assert_eq!(ValueType::Array(ScalarType::F64).to_string(), "f64[]");
        assert_eq!(ValueType::Str.to_string(), "str");
    }
}
