//! Tiled regions: the subtrees of a plan made of at least two elementwise kernels with array
//! outputs, evaluated tile by tile.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{Machine, Slot};
use crate::error::Error;
use crate::kernels::{Dispatch, KernelClass, KernelId, Registry};
use crate::ops::Op;
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
    /// Per region kernel, the rows of its tile calls summed over the tiles.
    rows: Vec<usize>,
    /// Per region kernel, its time summed over the tiles. Empty unless observing.
    times: Vec<Duration>,
    /// Guard evaluations, one per guard and tile, whose right operand ran over fewer rows
    /// than the guard's range.
    narrowed: u64,
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
        self.narrowed_tiles
            .fetch_add(run.narrowed, Ordering::Relaxed);
        if self.observing() {
            for ((k, time), &rows) in region.kernels.iter().zip(&run.times).zip(&run.rows) {
                // Bytes and row count are those of the rows the kernel ran over. Without
                // narrowing they are those of the whole arguments and result, as `observe`
                // reports them for a whole call.
                let width: usize = k
                    .args
                    .iter()
                    .map(|src| match *src {
                        Src::Input(i) => match &run.inputs[i] {
                            Value::Array(a) => a.elem().width(),
                            Value::Scalar(_) | Value::Str(_) => 0,
                        },
                        Src::Kernel(j) => region.kernels[j].elem.width(),
                    })
                    .sum();
                let write = rows * k.elem.width();
                let used = self.dispatch.variant_used(k.id);
                self.record(k.id, used, rows, rows * width, write, *time)?;
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
        let guards = if self.selective {
            self.guards(region, &inputs)
        } else {
            vec![None; region.kernels.len()]
        };
        // A whole call times the allocation of its output, so the region's allocations are
        // charged to the root below, from this point to the end of the output's construction.
        let alloc_start = observing.then(Instant::now);
        // One tile buffer per inner kernel, allocated once per run and overwritten by each
        // tile. A tile never holds more than `t` rows, or `n` when that is fewer.
        let Ok(bufs) = inner
            .iter()
            .map(|k| TileBuf::new(k.elem, t.min(n)))
            .collect::<Result<Vec<_>, _>>()
        else {
            return fail(inputs);
        };
        let mut tiles = Tiles {
            reg: self.reg,
            dispatch: self.dispatch,
            region,
            inputs: &inputs,
            guards: &guards,
            bufs,
            start: 0,
            args: Vec::new(),
            rows: vec![0; region.kernels.len()],
            times: if observing {
                vec![Duration::ZERO; region.kernels.len()]
            } else {
                Vec::new()
            },
            narrowed: 0,
        };
        let built = Array::build_bytes(n, root.elem, |out| {
            // An empty region still runs one empty tile, so that every kernel outside a
            // narrowed operand is called once as in the whole evaluation.
            loop {
                let len = t.min(n - tiles.start);
                tiles.run(inner.len(), 0, len, out)?;
                tiles.start += len;
                if tiles.start >= n {
                    return Ok(());
                }
            }
        });
        let Tiles {
            rows: kernel_rows,
            mut times,
            narrowed,
            ..
        } = tiles;
        // What the region took beyond its tile calls is the allocation of the tile buffers and
        // of the output, and it goes to the root, whose whole call would have allocated.
        if let Some(start) = alloc_start {
            let called: Duration = times.iter().sum();
            if let Some(last) = times.last_mut() {
                *last += start.elapsed().saturating_sub(called);
            }
        }
        match built {
            Ok(value) => Ok(TiledRun {
                inputs,
                rows: kernel_rows,
                times,
                narrowed,
                value: Value::Array(value),
            }),
            Err(_) => fail(inputs),
        }
    }

    /// Per kernel of `region` over the values `inputs`, whether it is a guard, and if so the
    /// value of its left operand's rows that leaves its result open, true for `and_bool` and
    /// false for `or_bool`. A guard is a total `and_bool` or `or_bool` whose left operand is
    /// an array and whose right operand is a region kernel whose subtree holds only total
    /// region kernels.
    fn guards(&self, region: &Region, inputs: &[Value]) -> Vec<Option<bool>> {
        // Post-order puts every kernel after the kernels it reads, so one pass finds the
        // totality of every subtree.
        let mut total: Vec<bool> = Vec::with_capacity(region.kernels.len());
        let mut guards = Vec::with_capacity(region.kernels.len());
        for k in &region.kernels {
            let def = self.reg.get(k.id);
            total.push(
                def.total
                    && k.args.iter().all(|src| match *src {
                        Src::Input(_) => true,
                        Src::Kernel(j) => total[j],
                    }),
            );
            let open = match def.op {
                Some(Op::And) => true,
                Some(Op::Or) => false,
                _ => {
                    guards.push(None);
                    continue;
                }
            };
            let left_array = |src: Src| match src {
                Src::Input(i) => matches!(inputs[i], Value::Array(_)),
                Src::Kernel(_) => true,
            };
            let guard = def.total
                && k.elem == ScalarType::Bool
                && matches!(k.args[..], [left, Src::Kernel(right)]
                    if left_array(left) && total[right]);
            guards.push(guard.then_some(open));
        }
        guards
    }
}

/// The rows of a block that the scan for a guard's open rows skips at once.
const SCAN_BLOCK: usize = 64;

/// The state of a tiled region run while it runs the region's kernels over one tile.
struct Tiles<'r> {
    reg: &'r Registry,
    dispatch: &'r Dispatch,
    region: &'r Region,
    /// The inputs' values, in the order of `Region::inputs`.
    inputs: &'r [Value],
    /// Per region kernel, see `Machine::guards`.
    guards: &'r [Option<bool>],
    /// One tile buffer per inner kernel, where a kernel writes the rows of the tile it runs
    /// over at their positions in the tile.
    bufs: Vec<TileBuf>,
    /// The first row of the tile in the region's arrays.
    start: usize,
    /// The arguments of a tile call, kept to reuse their allocation.
    args: Vec<Arg>,
    /// Per region kernel, the rows of its tile calls so far.
    rows: Vec<usize>,
    /// Per region kernel, its time so far. Empty unless observing.
    times: Vec<Duration>,
    /// See `TiledRun::narrowed`.
    narrowed: u64,
}

impl Tiles<'_> {
    /// Runs the subtree of region kernel `j` over the rows `lo..hi` of the tile, writing the
    /// root's rows into `out`, the region's output. A guard runs its right operand and itself
    /// only over the rows from the first to the last row of `lo..hi` where its left operand
    /// leaves the result open, and fills its other rows with the result they have whatever
    /// the right operand.
    fn run(&mut self, j: usize, lo: usize, hi: usize, out: &mut [u8]) -> Result<(), String> {
        let k = &self.region.kernels[j];
        let Some(open) = self.guards[j] else {
            for src in &k.args {
                if let Src::Kernel(i) = *src {
                    self.run(i, lo, hi, out)?;
                }
            }
            return self.call(j, lo, hi, out);
        };
        let [left, Src::Kernel(right)] = k.args[..] else {
            unreachable!("a guard's right operand is a region kernel");
        };
        if let Src::Kernel(l) = left {
            self.run(l, lo, hi, out)?;
        }
        // The scan and the fill go to the guard's time.
        let scan = (!self.times.is_empty()).then(Instant::now);
        let (a, b) = self.open_rows(left, lo, hi, open);
        self.charge(j, scan);
        if b - a < hi - lo {
            self.narrowed += 1;
        }
        // An empty range calls no kernel of the right operand.
        if a < b {
            self.run(right, a, b, out)?;
            self.call(j, a, b, out)?;
        }
        let filling = (!self.times.is_empty()).then(Instant::now);
        // Bool storage is 0 or 1, so `false & x` is 0 and `true | x` is 1 for every right row
        // `x`, and the filled rows equal what the guard computes from them.
        let fill = u8::from(!open);
        dst(&mut self.bufs, self.region, self.start, j, lo, a, out).fill(fill);
        dst(&mut self.bufs, self.region, self.start, j, b, hi, out).fill(fill);
        self.charge(j, filling);
        Ok(())
    }

    /// Calls region kernel `j` over the rows `lo..hi` of the tile.
    fn call(&mut self, j: usize, lo: usize, hi: usize, out: &mut [u8]) -> Result<(), String> {
        let k = &self.region.kernels[j];
        let mut args = std::mem::take(&mut self.args);
        args.clear();
        args.extend(k.args.iter().map(|src| self.rows_of(*src, lo, hi)));
        let dst = dst(&mut self.bufs, self.region, self.start, j, lo, hi, out);
        let begin = (!self.times.is_empty()).then(Instant::now);
        self.dispatch.elementwise_tile(self.reg, k.id, &args, dst)?;
        self.charge(j, begin);
        self.rows[j] += hi - lo;
        // The views of this tile's buffers go before the next tile overwrites them.
        args.clear();
        self.args = args;
        Ok(())
    }

    /// The rows `lo..hi` of the tile of `src`, or its scalar.
    fn rows_of(&self, src: Src, lo: usize, hi: usize) -> Arg {
        match src {
            Src::Input(i) => match &self.inputs[i] {
                Value::Array(a) => Arg::Rows(a.slice(self.start + lo, hi - lo)),
                Value::Scalar(s) => Arg::Splat(s.clone()),
                Value::Str(_) => unreachable!("string inputs fall back above"),
            },
            Src::Kernel(i) => Arg::Rows(self.bufs[i].view(hi).slice(lo, hi - lo)),
        }
    }

    /// The rows from the first to the last row of `lo..hi` where the bool array `left` is
    /// `open`, or an empty range when there is none.
    fn open_rows(&self, left: Src, lo: usize, hi: usize, open: bool) -> (usize, usize) {
        let Arg::Rows(rows) = self.rows_of(left, lo, hi) else {
            unreachable!("a guard's left operand is an array");
        };
        match open_span(rows.as_bool_bytes(), open) {
            Some((first, end)) => (lo + first, lo + end),
            None => (lo, lo),
        }
    }

    /// Adds the time since `begin` to region kernel `j`.
    fn charge(&mut self, j: usize, begin: Option<Instant>) {
        if let Some(begin) = begin {
            self.times[j] += begin.elapsed();
        }
    }
}

/// The positions of the first row of `rows` that is `open` and one past the last, or `None`
/// when no row is. A row is true when its byte is not 0.
fn open_span(rows: &[u8], open: bool) -> Option<(usize, usize)> {
    // Whole blocks without an open row are skipped with a fold over the block, which
    // compiles to vector instructions, and the rest is scanned row by row.
    let closed = |block: &&[u8; SCAN_BLOCK]| {
        if open {
            block.iter().fold(0, |acc, &b| acc | b) == 0
        } else {
            block.iter().fold(u8::MAX, |acc, &b| acc.min(b)) != 0
        }
    };
    let is_open = |b: &u8| (*b != 0) == open;
    let head = SCAN_BLOCK
        * rows
            .as_chunks::<SCAN_BLOCK>()
            .0
            .iter()
            .take_while(closed)
            .count();
    let first = head + rows[head..].iter().position(is_open)?;
    // The block holding the first open row is not closed, so the skipped tail ends after it.
    let tail = SCAN_BLOCK
        * rows[first..]
            .as_rchunks::<SCAN_BLOCK>()
            .1
            .iter()
            .rev()
            .take_while(closed)
            .count();
    let last = first
        + rows[first..rows.len() - tail]
            .iter()
            .rposition(is_open)
            .expect("the first open row is open");
    Some((first, last + 1))
}

/// The bytes of the rows `lo..hi` of the tile that starts at row `start`, in the tile buffer
/// of region kernel `j`, or in the region's output `out` when `j` is the root.
fn dst<'b>(
    bufs: &'b mut [TileBuf],
    region: &Region,
    start: usize,
    j: usize,
    lo: usize,
    hi: usize,
    out: &'b mut [u8],
) -> &'b mut [u8] {
    let w = region.kernels[j].elem.width();
    match bufs.get_mut(j) {
        Some(buf) => &mut buf.rows_mut(hi)[lo * w..],
        None => &mut out[(start + lo) * w..(start + hi) * w],
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
