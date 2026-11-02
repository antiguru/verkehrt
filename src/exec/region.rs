//! Tiled regions: the subtrees of a plan made of at least two elementwise kernels with array
//! outputs, evaluated tile by tile.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{Machine, Slot};
use crate::error::Error;
use crate::kernels::{KernelClass, KernelId};
use crate::plan::{PlanGraph, PlanId, PlanNode};
use crate::value::{Arg, Array, ScalarType, TileBuf, Value, ValueType};

#[cfg(test)]
mod tests;

/// Where a region kernel takes an argument from.
#[derive(Clone, Copy, Debug)]
enum Src {
    /// A region input, by position in `Region::inputs`.
    Input(usize),
    /// The output of a region kernel, by position in `Region::kernels`.
    Kernel(usize),
}

/// A kernel of a region with the sources of its arguments in input order.
struct RegionKernel {
    id: KernelId,
    args: Vec<Src>,
    /// The element type of its output.
    elem: ScalarType,
}

/// A region of a plan: its inputs in the order the whole evaluation reaches them, and its
/// kernels in post-order, so that every kernel comes after the kernels it reads and the root
/// comes last.
struct Region {
    inputs: Vec<PlanId>,
    kernels: Vec<RegionKernel>,
    /// Whether every input that is not trivial comes before the first kernel in post-order,
    /// so that the whole evaluation reaches all of them before it calls any region kernel.
    ordered: bool,
    /// The shared nodes among `inputs`.
    shared_inputs: HashSet<PlanId>,
}

/// A region run that completed tiled.
struct TiledRun {
    /// The inputs' values, in the order of `Region::inputs`.
    inputs: Vec<Value>,
    /// The rows of every array input and every kernel output.
    rows: usize,
    /// Per region kernel, its time summed over the tiles. Empty unless observing.
    times: Vec<Duration>,
    /// The root's value.
    value: Value,
}

impl<'a> Machine<'a> {
    /// The kernel and arguments of `node` when it is a region kernel: a kernel of class
    /// `Elementwise` with an array output that is not shared.
    pub(super) fn region_kernel(&self, node: PlanId) -> Option<(KernelId, &'a [PlanId])> {
        // A shared node runs through `eval` and the memo, so it is a region input that bounds
        // the region above it.
        if self.shared(node) {
            return None;
        }
        let PlanNode::Kernel(id, args) = self.graph.node(node) else {
            return None;
        };
        let def = self.reg.get(*id);
        (def.class == KernelClass::Elementwise && matches!(def.output, ValueType::Array(_)))
            .then_some((*id, args.as_slice()))
    }

    /// Adds the region kernels of the subtree `node` to `region` in post-order, and the
    /// subtrees just below them to its inputs, left to right. Clears `region.ordered` when an
    /// input that is not trivial follows a kernel. Returns where `node`'s value comes from.
    /// A node reached twice is added twice, as in the plan as a tree.
    fn collect(&self, node: PlanId, region: &mut Region) -> Src {
        let Some((id, args)) = self.region_kernel(node) else {
            // A shared input whose value is in the memo can neither block nor fail, so it is
            // as trivial as a leaf. So is a shared input that is an earlier input of this
            // region too: whole evaluation reaches the earlier one first, and either stops at
            // its failure or finds the value in the memo here. Any other shared input is not
            // trivial.
            let shared = self.shared(node);
            let reused = self.memoized(node) || (shared && region.shared_inputs.contains(&node));
            if !trivial(self.graph, node) && !reused && !region.kernels.is_empty() {
                region.ordered = false;
            }
            if shared {
                region.shared_inputs.insert(node);
            }
            region.inputs.push(node);
            return Src::Input(region.inputs.len() - 1);
        };
        let args = args.iter().map(|a| self.collect(*a, region)).collect();
        let ValueType::Array(elem) = self.reg.get(id).output else {
            unreachable!("a region kernel has an array output");
        };
        region.kernels.push(RegionKernel { id, args, elem });
        Src::Kernel(region.kernels.len() - 1)
    }

    /// Evaluates the region rooted at `node` in tiles of `t` rows. When the tiled run fails
    /// anywhere, or tiling would evaluate an input that is not trivial before a region kernel
    /// that the whole evaluation calls first, it evaluates the region whole instead.
    pub(super) fn eval_region(&mut self, node: PlanId, t: usize) -> Result<Value, Error> {
        let mut region = Region {
            inputs: Vec::new(),
            kernels: Vec::new(),
            ordered: true,
            shared_inputs: HashSet::new(),
        };
        self.collect(node, &mut region);
        // A tiled run evaluates every input before any kernel. Such an input after a kernel
        // may block or run long where the whole evaluation raises that kernel's error first.
        // Trivial inputs after a kernel can at most hold a stored error, which the fallback
        // raises in whole-evaluation order.
        if !region.ordered {
            return self.eval_region_whole(node, &mut Vec::new().into_iter());
        }
        let run = match self.run_tiled(&region, t) {
            Ok(run) => run,
            // The whole evaluation raises the error that the program raises without tiling.
            // It takes the inputs the attempt evaluated instead of evaluating them again,
            // which is the same because evaluation is pure.
            Err(inputs) => return self.eval_region_whole(node, &mut inputs.into_iter()),
        };
        self.tiled_regions.fetch_add(1, Ordering::Relaxed);
        if self.observing() {
            for (k, time) in region.kernels.iter().zip(&run.times) {
                // Bytes and row count are those of the whole arguments and result, as
                // `observe` reports them for a whole call.
                let read = k
                    .args
                    .iter()
                    .map(|src| match *src {
                        Src::Input(i) => match &run.inputs[i] {
                            Value::Array(a) => a.byte_len(),
                            Value::Scalar(_) | Value::Str(_) => 0,
                        },
                        Src::Kernel(j) => run.rows * region.kernels[j].elem.width(),
                    })
                    .sum();
                let write = run.rows * k.elem.width();
                let used = self.dispatch.variant_used(k.id);
                self.record(k.id, used, run.rows, read, write, *time)?;
            }
        }
        Ok(run.value)
    }

    /// Evaluates the region kernels of the subtree `node` whole, each with one call, in the
    /// order of a whole evaluation. The region's inputs come from `inputs` in their order,
    /// which holds the slots of a prefix of the inputs, of which only the last may have
    /// failed. The inputs after that prefix are evaluated when reached.
    fn eval_region_whole(
        &mut self,
        node: PlanId,
        inputs: &mut std::vec::IntoIter<Slot>,
    ) -> Result<Value, Error> {
        let Some((id, args)) = self.region_kernel(node) else {
            // The whole evaluation stops at the first failed input, so it never asks for an
            // input after it.
            return match inputs.next() {
                Some(slot) => slot,
                None => self.eval(node),
            };
        };
        let args = args
            .iter()
            .map(|a| self.eval_region_whole(*a, inputs))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_whole(id, &args)
    }

    /// Evaluates the inputs of `region`, then runs its kernels tile by tile over `t` rows,
    /// writing the root's tiles into its output array. Fails when an input fails, an input is
    /// a string, the array inputs differ in length or none is an array, an allocation fails,
    /// or a tile fails, and then returns the inputs evaluated so far, ending with the failed
    /// one if any.
    fn run_tiled(&mut self, region: &Region, t: usize) -> Result<TiledRun, Vec<Slot>> {
        let mut slots = Vec::with_capacity(region.inputs.len());
        for &n in &region.inputs {
            let slot = self.eval(n);
            let failed = slot.is_err();
            slots.push(slot);
            if failed {
                return Err(slots);
            }
        }
        let inputs: Vec<Value> = slots
            .into_iter()
            .map(|s| s.expect("checked above"))
            .collect();
        let fail = |values: Vec<Value>| Err(values.into_iter().map(Ok).collect());
        let mut rows = None;
        for v in &inputs {
            match v {
                Value::Array(a) if rows.is_some_and(|n| n != a.len()) => return fail(inputs),
                Value::Array(a) => rows = Some(a.len()),
                Value::Scalar(_) => {}
                Value::Str(_) => return fail(inputs),
            }
        }
        let Some(n) = rows else {
            return fail(inputs);
        };
        let (root, inner) = region.kernels.split_last().expect("a region has kernels");
        let observing = self.observing();
        // A whole call times the allocation of its output, so the region's allocations are
        // charged to the root below, from this point to the end of the output's construction.
        let alloc_start = observing.then(Instant::now);
        // One tile buffer per inner kernel, allocated once per run and overwritten by each
        // tile. A tile never holds more than `t` rows, or `n` when that is fewer.
        let Ok(mut bufs) = inner
            .iter()
            .map(|k| TileBuf::new(k.elem, t.min(n)))
            .collect::<Result<Vec<_>, _>>()
        else {
            return fail(inputs);
        };
        let mut times = if observing {
            vec![Duration::ZERO; region.kernels.len()]
        } else {
            Vec::new()
        };
        let mut args: Vec<Arg> = Vec::new();
        let w = root.elem.width();
        let built = Array::build_bytes(n, root.elem, |out| {
            let mut start = 0;
            // An empty region still runs one empty tile, so that every kernel is called once
            // as in the whole evaluation.
            loop {
                let len = t.min(n - start);
                for (j, k) in region.kernels.iter().enumerate() {
                    args.clear();
                    args.extend(k.args.iter().map(|src| match *src {
                        Src::Input(i) => match &inputs[i] {
                            Value::Array(a) => Arg::Rows(a.slice(start, len)),
                            Value::Scalar(s) => Arg::Splat(s.clone()),
                            Value::Str(_) => unreachable!("string inputs fall back above"),
                        },
                        Src::Kernel(i) => Arg::Rows(bufs[i].view(len)),
                    }));
                    let dst = match bufs.get_mut(j) {
                        Some(buf) => buf.rows_mut(len),
                        None => &mut out[start * w..(start + len) * w],
                    };
                    let begin = observing.then(Instant::now);
                    self.dispatch.elementwise_tile(self.reg, k.id, &args, dst)?;
                    if let Some(begin) = begin {
                        times[j] += begin.elapsed();
                    }
                }
                // The views of this tile's buffers go before the next tile overwrites them.
                args.clear();
                start += len;
                if start >= n {
                    return Ok(());
                }
            }
        });
        // What the region took beyond its tile calls is the allocation of the tile buffers and
        // of the output, and it goes to the root, whose whole call would have allocated.
        if let Some(start) = alloc_start {
            let tiles: Duration = times.iter().sum();
            if let Some(last) = times.last_mut() {
                *last += start.elapsed().saturating_sub(tiles);
            }
        }
        match built {
            Ok(value) => Ok(TiledRun {
                inputs,
                rows: n,
                times,
                value: Value::Array(value),
            }),
            Err(_) => fail(inputs),
        }
    }
}

/// Whether evaluating `node` can neither block nor do unbounded work: a leaf, or a broadcast
/// of a leaf. Such a node can still return a stored error.
fn trivial(graph: &PlanGraph, node: PlanId) -> bool {
    match graph.node(node) {
        PlanNode::Leaf(_) => true,
        PlanNode::Bcast(inner) => matches!(graph.node(*inner), PlanNode::Leaf(_)),
        PlanNode::Cond(..) | PlanNode::Kernel(..) => false,
    }
}
