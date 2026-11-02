use crate::check::VarId;
use crate::value::{Array, Scalar, ScalarType, Value};

use super::{Leaf, LeafId};

/// The interned leaves. Variables share one leaf each, as do equal scalar and string
/// constants, so that structurally equal terms compare equal.
pub(super) struct Leaves(pub(super) Vec<Leaf>);

impl Leaves {
    pub(super) fn push(&mut self, leaf: Leaf) -> LeafId {
        self.0.push(leaf);
        LeafId(u32::try_from(self.0.len() - 1).expect("leaf count fits in u32"))
    }

    pub(super) fn id_at(i: usize) -> LeafId {
        LeafId(u32::try_from(i).expect("leaf count fits in u32"))
    }

    pub(super) fn var(&mut self, v: VarId) -> LeafId {
        match self
            .0
            .iter()
            .position(|l| matches!(l, Leaf::Var(w) if *w == v))
        {
            Some(i) => Leaves::id_at(i),
            None => self.push(Leaf::Var(v)),
        }
    }

    pub(super) fn constant(&mut self, value: Value) -> LeafId {
        // Arrays are never compared, they are large and each literal is its own value.
        if !matches!(value, Value::Array(_)) {
            let found = self.0.iter().position(|l| match l {
                Leaf::Const(c) => same_const(c, &value),
                Leaf::Var(_) => false,
            });
            if let Some(i) = found {
                return Leaves::id_at(i);
            }
        }
        self.push(Leaf::Const(value))
    }

    pub(super) fn get(&self, id: LeafId) -> &Leaf {
        &self.0[usize::try_from(id.0).expect("u32 fits in usize")]
    }
}

/// Bitwise-faithful equality for scalar and string constants. Display distinguishes `-0.0`
/// from `0.0`, which `==` does not.
fn same_const(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => x.ty() == y.ty() && x.to_string() == y.to_string(),
        (Value::Str(x), Value::Str(y)) => x == y,
        _ => false,
    }
}

pub(super) fn array_of(elem: ScalarType, items: &[Scalar]) -> Array {
    macro_rules! build {
        ($variant:ident, $ty:ty) => {
            Array::from_vec(
                items
                    .iter()
                    .map(|s| match s {
                        Scalar::$variant(v) => *v,
                        _ => unreachable!("the checker unifies array element types"),
                    })
                    .collect::<Vec<$ty>>(),
            )
        };
    }
    match elem {
        ScalarType::I8 => build!(I8, i8),
        ScalarType::I16 => build!(I16, i16),
        ScalarType::I32 => build!(I32, i32),
        ScalarType::I64 => build!(I64, i64),
        ScalarType::U8 => build!(U8, u8),
        ScalarType::U16 => build!(U16, u16),
        ScalarType::U32 => build!(U32, u32),
        ScalarType::U64 => build!(U64, u64),
        ScalarType::F32 => build!(F32, f32),
        ScalarType::F64 => build!(F64, f64),
        ScalarType::Bool => {
            let bools: Vec<bool> = items
                .iter()
                .map(|s| matches!(s, Scalar::Bool(true)))
                .collect();
            Array::from_bools(&bools)
        }
    }
}
