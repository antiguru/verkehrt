//! The tier 2 primitives, built so their indices, keys, and bounds are mostly valid.

use verkehrt::ops::Op;
use verkehrt::value::{ScalarType, ValueType};

use super::{Ctx, E, Gen, INTS, Len, Sig, spec_elem};

/// Keys for `histogram` and `reduce_into`, see `Gen::keys`.
pub(super) struct Keys {
    keys: Option<E>,
    n: Option<E>,
    out: Len,
    binding: Option<(String, E)>,
}

impl Gen<'_, '_> {
    /// A length to build an array of: of class `len` when given, else a small literal or the
    /// length of an array in scope.
    pub(super) fn new_len(&mut self, cx: &Ctx, len: Option<Len>) -> Option<(E, Len)> {
        if let Some(l) = len {
            return Some((self.length_of(cx, l)?, l));
        }
        let classes: Vec<Len> = cx.vars.iter().filter_map(|v| v.len).collect();
        if !classes.is_empty() && self.c.flip(60) {
            let l = *self.c.pick(&classes);
            return Some((self.length_of(cx, l)?, l));
        }
        let n = self.c.below(21);
        Some((self.lit_i64(n), Len::Known(n)))
    }

    /// The tier 2 primitives with a result of `t[]`.
    pub(super) fn tier2(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let want = ValueType::Array(t);
        let i64s = t == ScalarType::I64;
        let free = len.is_none();
        let weights = [
            2,                                         // scan
            2,                                         // scan_by
            2,                                         // prev
            usize::from(i64s) * 2,                     // argsort
            usize::from(i64s) * 2,                     // group_id
            usize::from(free) * 2,                     // compress
            usize::from(free && i64s),                 // where
            usize::from(i64s),                         // iota
            1,                                         // fill
            usize::from(i64s || t == ScalarType::F64), // rand
            3,                                         // gather
            usize::from(free) * 2,                     // slice
            usize::from(i64s) * 2,                     // histogram
            3,                                         // reduce_into
        ];
        match self.c.weighted(&weights) {
            0 => self.generic(cx, want, len, |op| matches!(op, Op::Scan(_))),
            1 => self.scan_by(cx, want, len),
            2 => self.generic(cx, want, len, |op| op == Op::Prev),
            3 => self.generic(cx, want, len, |op| op == Op::ArgSort),
            4 => self.generic(cx, want, len, |op| op == Op::GroupId),
            5 => self.generic(cx, want, len, |op| op == Op::Compress),
            6 => self.generic(cx, want, len, |op| op == Op::Where),
            7 => {
                let (n, l) = self.new_len(cx, len)?;
                self.op(cx, Op::Iota, vec![n], Some(l))
            }
            8 => {
                let (n, l) = self.new_len(cx, len)?;
                let v = self.expr(cx, ValueType::Scalar(t), None);
                self.op(cx, Op::Fill, vec![n, v], Some(l))
            }
            9 => {
                let (n, l) = self.new_len(cx, len)?;
                // Programs are reproducible, so every seed is explicit.
                let n_seed = self.c.below(1000);
                let seed = self.lit_i64(n_seed);
                let op = if i64s { Op::RandI64 } else { Op::RandF64 };
                self.op(cx, op, vec![n, seed], Some(l))
            }
            10 => self.gather(cx, t, len),
            11 => self.slice(cx, t),
            12 => self.histogram(cx, len),
            _ => self.reduce_into(cx, t, len),
        }
    }

    /// `scan_by`, often with segments that start where a key changes, `k != prev(k, c)`.
    pub(super) fn scan_by(&mut self, cx: &mut Ctx, want: ValueType, len: Option<Len>) -> Option<E> {
        let is_scan_by = |op| matches!(op, Op::ScanBy(_));
        if !self.c.flip(40) {
            return self.generic(cx, want, len, is_scan_by);
        }
        let sigs: Vec<Sig> = self
            .sigs
            .iter()
            .filter(|s| s.out == want && is_scan_by(s.op))
            .cloned()
            .collect();
        if sigs.is_empty() {
            return None;
        }
        let sig = self.c.pick(&sigs).clone();
        let vt = spec_elem(sig.inputs[0]).expect("values are an array");
        let x = self.expr(cx, ValueType::Array(vt), len);
        let (x, outer) = self.anchor(cx, x);
        let kt = *self.c.pick(&INTS);
        let k = self.expr(cx, ValueType::Array(kt), x.len);
        let (k, inner) = self.bind(cx, k);
        let fill = self.lit(kt);
        let starts = self
            .op(cx, Op::Prev, vec![k.clone(), fill], None)
            .and_then(|p| self.op(cx, Op::Ne, vec![k, p], None));
        let r = starts.and_then(|s| self.op(cx, sig.op, vec![x, s], None));
        let r = self.finish(cx, inner, r);
        self.finish(cx, outer, r)
    }

    /// An `i64[]` index array into an array of class `of`, with indices in range: a sort
    /// permutation, positions of a mask, or `iota`. Rarely any `i64[]`, which may be out of
    /// range, an error. With `len` the index array has that class.
    pub(super) fn indices(&mut self, cx: &mut Ctx, of: Len, len: Option<Len>) -> Option<E> {
        let i64a = ValueType::Array(ScalarType::I64);
        let fixed = len.is_some();
        match self.c.weighted(&[3, 2, usize::from(!fixed) * 2, 1]) {
            0 => {
                let k = *self.c.pick(&INTS);
                let y = self.expr(cx, ValueType::Array(k), Some(of));
                self.op(cx, Op::ArgSort, vec![y], None)
            }
            1 => {
                let n = self.length_of(cx, of)?;
                self.op(cx, Op::Iota, vec![n], Some(of))
            }
            2 => {
                let m = self.expr(cx, ValueType::Array(ScalarType::Bool), Some(of));
                let fresh = self.fresh_class();
                self.op(cx, Op::Where, vec![m], Some(fresh))
            }
            _ => Some(self.expr(cx, i64a, len)),
        }
    }

    pub(super) fn gather(&mut self, cx: &mut Ctx, t: ScalarType, len: Option<Len>) -> Option<E> {
        let x = self.expr(cx, ValueType::Array(t), len);
        let (x, binding) = self.anchor(cx, x);
        let of = x.len.expect("an array has a length");
        let r = self
            .indices(cx, of, len)
            .and_then(|idx| {
                let l = idx.len;
                self.op(cx, Op::Gather, vec![x, idx], l)
            })
            .filter(|e| len.is_none() || e.len == len);
        self.finish(cx, binding, r)
    }

    pub(super) fn slice(&mut self, cx: &mut Ctx, t: ScalarType) -> Option<E> {
        let x = self.expr(cx, ValueType::Array(t), None);
        let (x, binding) = self.bind(cx, x);
        let of = x.len.expect("an array has a length");
        let r = match self.c.weighted(&[3, 3, 1, 1]) {
            0 => {
                let end = self.length_of(cx, of);
                end.and_then(|end| self.op(cx, Op::Slice, vec![x, self.lit_i64(0), end], None))
            }
            1 => match of {
                Len::Known(n) => {
                    let a = self.c.below(n + 1);
                    let b = a + self.c.below(n - a + 1);
                    let (sa, sb) = (self.lit_i64(a), self.lit_i64(b));
                    self.op(cx, Op::Slice, vec![x, sa, sb], Some(Len::Known(b - a)))
                }
                Len::Class(_) => None,
            },
            2 => {
                // An error on an empty array.
                let end = self.length_of(cx, of);
                let fresh = self.fresh_class();
                end.and_then(|end| {
                    self.op(cx, Op::Slice, vec![x, self.lit_i64(1), end], Some(fresh))
                })
            }
            _ => {
                let i64s = ValueType::Scalar(ScalarType::I64);
                let a = self.expr(cx, i64s, None);
                let b = self.expr(cx, i64s, None);
                let fresh = self.fresh_class();
                self.op(cx, Op::Slice, vec![x, a, b], Some(fresh))
            }
        };
        self.finish(cx, binding, r)
    }

    /// Keys in `0..n` with `n` and the length class of the result: the group ids of an array
    /// and its length, or a `bool[]` as keys with `n` of 2. Rarely any keys and `n`, which may
    /// be out of range, an error. The caller finishes the binding around its result.
    pub(super) fn keys(&mut self, cx: &mut Ctx, len: Option<Len>) -> Option<Keys> {
        match self.c.weighted(&[4, usize::from(len.is_none()), 1]) {
            0 => {
                let kt = *self.c.pick(&INTS);
                let k = self.expr(cx, ValueType::Array(kt), len);
                let (k, binding) = self.anchor(cx, k);
                let of = k.len.expect("an array has a length");
                let ids = self.op(cx, Op::GroupId, vec![k], None);
                let n = self.length_of(cx, of);
                Some(Keys {
                    keys: ids,
                    n,
                    out: of,
                    binding,
                })
            }
            1 => {
                let m = self.expr(cx, ValueType::Array(ScalarType::Bool), None);
                let keys = self.op(cx, Op::Cast(ScalarType::I64), vec![m], None);
                Some(Keys {
                    keys,
                    n: Some(self.lit_i64(2)),
                    out: Len::Known(2),
                    binding: None,
                })
            }
            _ => {
                let keys = self.expr(cx, ValueType::Array(ScalarType::I64), None);
                let n = self.c.below(9);
                Some(Keys {
                    keys: Some(keys),
                    n: Some(self.lit_i64(n)),
                    out: Len::Known(n),
                    binding: None,
                })
            }
        }
    }

    pub(super) fn histogram(&mut self, cx: &mut Ctx, len: Option<Len>) -> Option<E> {
        let k = self.keys(cx, len)?;
        let r = match (k.keys, k.n) {
            (Some(keys), Some(n)) if len.is_none_or(|l| l == k.out) => {
                self.op(cx, Op::Histogram, vec![keys, n], Some(k.out))
            }
            _ => None,
        };
        self.finish(cx, k.binding, r)
    }

    pub(super) fn reduce_into(
        &mut self,
        cx: &mut Ctx,
        t: ScalarType,
        len: Option<Len>,
    ) -> Option<E> {
        let want = ValueType::Array(t);
        let sigs: Vec<Sig> = self
            .sigs
            .iter()
            .filter(|s| s.out == want && matches!(s.op, Op::ReduceInto(_)))
            .cloned()
            .collect();
        if sigs.is_empty() {
            return None;
        }
        let sig = self.c.pick(&sigs).clone();
        let vt = spec_elem(sig.inputs[1]).expect("values are an array");
        let k = self.keys(cx, len)?;
        let r = match (k.keys, k.n) {
            (Some(keys), Some(n)) if len.is_none_or(|l| l == k.out) => {
                let (keys, binding) = self.anchor(cx, keys);
                let vals = self.expr(cx, ValueType::Array(vt), keys.len);
                let r = self.op(cx, sig.op, vec![keys, vals, n], Some(k.out));
                self.finish(cx, binding, r)
            }
            _ => None,
        };
        self.finish(cx, k.binding, r)
    }
}
