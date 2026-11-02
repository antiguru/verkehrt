//! Expressions that fail whenever they are computed. The generator binds them to names that
//! nothing reads, so only demand keeps their errors from being raised, see the demand rule of
//! the errors-as-data spec.

use verkehrt::ops::{Op, ReduceOp};
use verkehrt::value::ScalarType;

use super::{Ctx, E, Gen, Len, Var, elem, is_array};

impl Gen<'_, '_> {
    /// An expression that fails whenever it is computed: a `min` of an empty array, a
    /// `gather` index out of range, or a length mismatch. It is built on an array in scope
    /// when there is one. Only called outside lambdas.
    pub(super) fn failing(&mut self, cx: &Ctx) -> E {
        debug_assert!(!cx.lambda, "failing expressions are built outside lambdas");
        let arrays: Vec<Var> = cx.vars.iter().filter(|v| is_array(v.ty)).cloned().collect();
        let x = if !arrays.is_empty() && self.c.flip(75) {
            let v = self.c.pick(&arrays).clone();
            self.var(cx, &v)
        } else {
            let t = self.ty();
            let n = 1 + self.c.below(4);
            let (nl, v) = (self.lit_i64(n), self.lit(t));
            self.op(cx, Op::Fill, vec![nl, v], Some(Len::Known(n)))
                .expect("fill on every generated type")
        };
        let first = self.c.below(3);
        // A shape without a kernel for the element type falls back to the next one.
        (0..3)
            .find_map(|i| match (first + i) % 3 {
                0 => self.empty_min(cx, x.clone()),
                1 => self.gather_past_end(cx, x.clone()),
                _ => self.longer_by_one(cx, x.clone()),
            })
            .unwrap_or_else(|| {
                let (n, v) = (self.lit_i64(0), self.lit(ScalarType::F64));
                let empty = self
                    .op(cx, Op::Fill, vec![n, v], Some(Len::Known(0)))
                    .expect("fill on f64");
                self.op(cx, Op::Reduce(ReduceOp::Min), vec![empty], None)
                    .expect("min on f64")
            })
    }

    /// `reduce(slice(x, 0, 0), min)`.
    fn empty_min(&self, cx: &Ctx, x: E) -> Option<E> {
        let (a, b) = (self.lit_i64(0), self.lit_i64(0));
        let empty = self.op(cx, Op::Slice, vec![x, a, b], Some(Len::Known(0)))?;
        self.op(cx, Op::Reduce(ReduceOp::Min), vec![empty], None)
    }

    /// `gather(x, fill(1, length(x)))`, whose one index is past the end of `x`.
    fn gather_past_end(&self, cx: &Ctx, x: E) -> Option<E> {
        let n = self.op(cx, Op::Length, vec![x.clone()], None)?;
        let one = self.lit_i64(1);
        let idx = self.op(cx, Op::Fill, vec![one, n], Some(Len::Known(1)))?;
        self.op(cx, Op::Gather, vec![x, idx], Some(Len::Known(1)))
    }

    /// `x == fill(length(x) + 1, c)`, a comparison of arrays of different lengths.
    fn longer_by_one(&mut self, cx: &Ctx, x: E) -> Option<E> {
        let n = self.op(cx, Op::Length, vec![x.clone()], None)?;
        let one = self.lit_i64(1);
        let more = self.op(cx, Op::Add, vec![n, one], None)?;
        let c = self.lit(elem(x.ty));
        let len = self.fresh_class();
        let other = self.op(cx, Op::Fill, vec![more, c], Some(len))?;
        self.op(cx, Op::Eq, vec![x, other], None)
    }
}
