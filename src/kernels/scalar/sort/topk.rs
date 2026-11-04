//! The fused top-k kernels `topk_<t>` and `topk2_<a>_<b>`, which compute
//! `slice(argsort(...), s, e)` without sorting every row when `e` is small.

use std::marker::PhantomData;

use crate::kernels::WholeBody;
use crate::kernels::scalar::construct::i64_arg;
use crate::kernels::scalar::elementwise::arity_error;
use crate::kernels::scalar::index::slice_bounds;
use crate::kernels::scalar::reduce::array_arg;
use crate::kernels::scalar::sort::radix::RadixKey;
use crate::kernels::scalar::sort::{ArgSort, ArgSort2};
use crate::value::{Array, ScalarType, Value};

/// The largest `e` that the selection handles, above it the kernel sorts every row. The
/// bench `docs/measurements/2026-11-03-topk-bench.md` measures the selection against the full
/// sort up to this value.
// The selection beat the full sort on descending keys, where every row enters the buffer, at
// 80,000 and 1M rows for `e` up to 128, and tied at 256 for one column at 80,000 rows, so 64
// stays inside the measured range.
pub(crate) const TOPK_MAX: usize = 64;

/// The rows that the selection buffers beyond the kept ones, at least, before it trims.
const BATCH: usize = 256;

/// The `e` smallest rows of `keys` by (key, position), in that order. Rows with equal keys
/// keep their position order, so the result is the first `e` entries of the stable argsort.
fn select<K: Ord + Copy>(keys: impl Iterator<Item = K>, e: usize) -> Vec<(K, i64)> {
    if e == 0 {
        return Vec::new();
    }
    // Rows collect unsorted in a buffer that a partial selection trims to the `e` best
    // whenever it fills. The buffer holds at least `BATCH` rows beyond the `e` kept, so that a
    // trim is amortized over many rows when `e` is small. That costs a constant number of
    // steps per row even when every row enters, as in a descending input, where inserting
    // into a sorted buffer shifts up to `e` rows per row.
    let cap = e + e.max(BATCH);
    let mut buf: Vec<(K, i64)> = Vec::with_capacity(cap);
    // The key of the worst kept row once a trim has run.
    let mut worst: Option<K> = None;
    for (k, i) in keys.zip(0..) {
        // Positions grow, so a row that ties the worst kept row loses to it.
        if worst.is_some_and(|w| k >= w) {
            continue;
        }
        buf.push((k, i));
        if buf.len() == cap {
            buf.select_nth_unstable(e - 1);
            buf.truncate(e);
            worst = Some(buf[e - 1].0);
        }
    }
    if buf.len() > e {
        buf.select_nth_unstable(e - 1);
        buf.truncate(e);
    }
    // Positions are distinct, so the unstable sort gives the stable order.
    buf.sort_unstable();
    buf
}

/// Rows `s..e` of the stable argsort of `n` rows, with the keys of `keys` and the full sort
/// of `full`. Selects when the slice is valid and at most `TOPK_MAX` of fewer than `n` rows,
/// and otherwise runs `full`, so an invalid slice fails after the sort as it does unfused.
fn topk<K: Ord + Copy, I: Iterator<Item = K>>(
    n: usize,
    start: &Value,
    end: &Value,
    keys: impl FnOnce() -> I,
    full: impl FnOnce() -> Result<Value, String>,
) -> Result<Value, String> {
    let (start, end) = (i64_arg(start)?, i64_arg(end)?);
    let bounds = slice_bounds(start, end, n);
    let (s, e) = match bounds {
        Ok((s, e)) if e <= TOPK_MAX && e < n => (s, e),
        _ => {
            let Value::Array(sorted) = full()? else {
                return Err("argsort did not return an array".into());
            };
            let (s, e) = bounds?;
            return Ok(Value::Array(sorted.slice(s, e - s)));
        }
    };
    let best = select(keys(), e);
    Array::build_bytes(e - s, ScalarType::I64, |out| {
        let out: &mut [i64] = bytemuck::cast_slice_mut(out);
        for (o, &(_, i)) in out.iter_mut().zip(&best[s..]) {
            *o = i;
        }
        Ok(())
    })
    .map(Value::Array)
}

/// The body of `topk_<t>`, rows `s..e` of the stable ascending positions of a column of
/// type `T`.
pub(crate) struct TopK<T>(PhantomData<fn() -> T>);

impl<T: RadixKey> WholeBody for TopK<T> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs, s, e] = args else {
            return Err(arity_error(3, args.len()));
        };
        let xs = array_arg(xs, T::TYPE)?.as_slice::<T>();
        topk(
            xs.len(),
            s,
            e,
            || xs.iter().map(|&x| -> u64 { x.image().into() }),
            || ArgSort::<T>::run(&args[..1]),
        )
    }
}

/// The body of `topk2_<a>_<b>`, rows `s..e` of the positions of the rows of two columns in
/// ascending order of (`a`, `b`, position).
pub(crate) struct TopK2<A, B>(PhantomData<fn() -> (A, B)>);

impl<A: RadixKey, B: RadixKey> WholeBody for TopK2<A, B> {
    #[inline(always)]
    fn run(args: &[Value]) -> Result<Value, String> {
        let [xs, ys, s, e] = args else {
            return Err(arity_error(4, args.len()));
        };
        let xs = array_arg(xs, A::TYPE)?.as_slice::<A>();
        let ys = array_arg(ys, B::TYPE)?.as_slice::<B>();
        // The argsort's argument check comes before the bounds, with its message.
        if xs.len() != ys.len() {
            return Err(format!("length mismatch: {} vs {}", xs.len(), ys.len()));
        }
        topk(
            xs.len(),
            s,
            e,
            || {
                xs.iter().zip(ys).map(|(&x, &y)| {
                    let (x, y): (u64, u64) = (x.image().into(), y.image().into());
                    (x, y)
                })
            },
            || ArgSort2::<A, B>::run(&args[..2]),
        )
    }
}

#[cfg(test)]
mod tests;
