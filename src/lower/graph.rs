//! The hash-consed term graph that lowering builds and the planner reads.

use std::collections::HashMap;

use crate::value::ValueType;

use super::TermKind;

/// A node of a `TermGraph`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(pub u32);

impl TermId {
    pub(crate) fn index(self) -> usize {
        usize::try_from(self.0).expect("u32 fits in usize")
    }
}

/// An arena of first-order terms. A node is its kind and its type, and nodes are interned on
/// both, so two structurally equal terms of one graph have the same id. A node's children
/// always have smaller ids than the node.
#[derive(Clone, Debug, Default)]
pub struct TermGraph {
    nodes: Vec<(TermKind, ValueType)>,
    index: HashMap<(TermKind, ValueType), TermId>,
}

impl TermGraph {
    /// The kind of `id`.
    pub fn kind(&self, id: TermId) -> &TermKind {
        &self.nodes[id.index()].0
    }

    /// The type of `id`.
    pub fn ty(&self, id: TermId) -> ValueType {
        self.nodes[id.index()].1
    }

    /// The number of distinct nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no node.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The id of the node with `kind` and `ty`, adding it when the graph has none. The
    /// children of `kind` must be nodes of this graph.
    pub(crate) fn intern(&mut self, kind: TermKind, ty: ValueType) -> TermId {
        let key = (kind, ty);
        if let Some(&id) = self.index.get(&key) {
            return id;
        }
        let id = TermId(u32::try_from(self.nodes.len()).expect("term count fits in u32"));
        self.nodes.push(key.clone());
        self.index.insert(key, id);
        id
    }
}
