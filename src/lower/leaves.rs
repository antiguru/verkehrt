use std::collections::HashMap;
use std::sync::Arc;

use crate::check::VarId;
use crate::value::{Array, Scalar, ScalarType, Value};

use super::{Leaf, LeafId};

/// The interned leaves. Variables share one leaf each, as do equal scalar and string
/// constants, so that structurally equal terms compare equal. Ids are assigned in order of
/// first request.
#[derive(Default)]
pub(super) struct Leaves {
    pub(super) leaves: Vec<Leaf>,
    vars: HashMap<VarId, LeafId>,
    consts: HashMap<ConstKey, LeafId>,
}

/// The key of a scalar or string constant, equal exactly when `same_const` holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ConstKey {
    Scalar(ScalarType, String),
    Str(Arc<str>),
}

impl Leaves {
    pub(super) fn push(&mut self, leaf: Leaf) -> LeafId {
        self.leaves.push(leaf);
        LeafId(u32::try_from(self.leaves.len() - 1).expect("leaf count fits in u32"))
    }

    pub(super) fn var(&mut self, v: VarId) -> LeafId {
        if let Some(&id) = self.vars.get(&v) {
            return id;
        }
        let id = self.push(Leaf::Var(v));
        self.vars.insert(v, id);
        id
    }

    pub(super) fn constant(&mut self, value: Value) -> LeafId {
        // Arrays are never compared, they are large and each literal is its own value.
        let Some(key) = const_key(&value) else {
            return self.push(Leaf::Const(value));
        };
        if let Some(&id) = self.consts.get(&key) {
            return id;
        }
        let id = self.push(Leaf::Const(value));
        self.consts.insert(key, id);
        id
    }

    pub(super) fn get(&self, id: LeafId) -> &Leaf {
        &self.leaves[usize::try_from(id.0).expect("u32 fits in usize")]
    }
}

/// Bitwise-faithful equality for scalar and string constants, as a key. Display distinguishes
/// `-0.0` from `0.0`, which `==` does not. Arrays have no key.
fn const_key(v: &Value) -> Option<ConstKey> {
    match v {
        Value::Scalar(x) => Some(ConstKey::Scalar(x.ty(), x.to_string())),
        Value::Str(s) => Some(ConstKey::Str(Arc::clone(s))),
        Value::Array(_) => None,
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
