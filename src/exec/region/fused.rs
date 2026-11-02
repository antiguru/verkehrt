//! Fused regions: a reduction with region kernel arguments runs as the root of a tiled region
//! and consumes each tile through its accumulator, so the results of its region kernel
//! arguments never exist in full.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::tiles::Tiles;
use super::{Region, Src, TiledRun, arg_width, fail};
use crate::error::Error;
use crate::exec::{Machine, Slot};
use crate::kernels::{KernelId, REDUCE_BLOCK};
use crate::plan::PlanId;
use crate::value::{TileBuf, Value};

#[cfg(test)]
mod tests;

/// A fused region run that completed tiled.
struct FusedRun {
    /// The run of the region kernels, with the root's value.
    run: TiledRun,
    /// The region's row count.
    rows: usize,
    /// The time of the root's `update` and `finish` calls and of the allocation of the tile
    /// buffers. Zero unless observing.
    root_time: Duration,
}

/// The tile row count of a fused region under `Tiling::Rows(t)`: `t` rounded up to a multiple
/// of `REDUCE_BLOCK`, or down when rounding up overflows.
fn fused_tile_rows(t: usize) -> usize {
    t.checked_next_multiple_of(REDUCE_BLOCK)
        .unwrap_or(t / REDUCE_BLOCK * REDUCE_BLOCK)
}

impl Machine<'_> {
    /// Evaluates the call of reduction `id` on `args`, the root of a fused region, see
    /// `fused_root`, with the region's kernels in tiles of `t` rows rounded by
    /// `fused_tile_rows`. When the tiled run fails anywhere, `finish` included, it evaluates
    /// the region and the root whole instead. Returns `None`, having evaluated nothing, when
    /// tiling would evaluate an input that is not trivial before a region kernel that the
    /// whole evaluation calls first.
    pub(in crate::exec) fn eval_fused(
        &mut self,
        id: KernelId,
        args: &[PlanId],
        t: usize,
    ) -> Option<Result<Value, Error>> {
        let (region, srcs) = self.collect_fused(args);
        // As for a plain region, such an input after a kernel may block or run long where
        // the whole evaluation raises that kernel's error first. The caller then evaluates
        // the arguments as without fusion, where each may still be a plain region.
        if !region.ordered {
            return None;
        }
        let fused = match self.run_fused(&region, id, &srcs, fused_tile_rows(t)) {
            Ok(fused) => fused,
            // The whole evaluation raises the error that the program raises without fusion,
            // with the inputs the attempt evaluated, see `eval_region`.
            Err(inputs) => return Some(self.eval_fused_whole(id, args, inputs)),
        };
        Some(self.finish_fused(&region, id, &srcs, fused))
    }

    /// Counts and observes the completed run `fused` of `region` with root `id` on `srcs`,
    /// and returns the root's value.
    fn finish_fused(
        &mut self,
        region: &Region,
        id: KernelId,
        srcs: &[Src],
        fused: FusedRun,
    ) -> Result<Value, Error> {
        self.tiled_regions.fetch_add(1, Ordering::Relaxed);
        self.narrowed_tiles
            .fetch_add(fused.run.narrowed, Ordering::Relaxed);
        self.fused_reductions[usize::try_from(id.0).expect("u32 fits in usize")]
            .fetch_add(1, Ordering::Relaxed);
        self.record_kernels(region, &fused.run)?;
        if self.observing() {
            // A whole call of the root reads its array arguments in full and writes no
            // array, see `observe`.
            let read = fused.rows * arg_width(region, &fused.run.inputs, srcs);
            let used = self.dispatch.variant_used(id);
            self.record(id, used, fused.rows, read, 0, fused.root_time)?;
        }
        Ok(fused.run.value)
    }

    /// Evaluates the inputs of `region`, then runs its kernels tile by tile over `t` rows,
    /// a multiple of `REDUCE_BLOCK`, into tile buffers, and updates the accumulator of
    /// reduction `id` with its arguments `srcs` over each tile that has rows. Fails when
    /// `eval_inputs` fails, an allocation fails, a tile fails, `update` fails, or `finish`
    /// fails, and then returns the inputs evaluated so far, ending with the failed one if any.
    fn run_fused(
        &mut self,
        region: &Region,
        id: KernelId,
        srcs: &[Src],
        t: usize,
    ) -> Result<FusedRun, Vec<Slot>> {
        let (inputs, n) = self.eval_inputs(region)?;
        let observing = self.observing();
        let guards = if self.selective {
            self.guards(region, &inputs)
        } else {
            vec![None; region.kernels.len()]
        };
        // A whole call of a region kernel times the allocation of its output, so the
        // allocation of the tile buffers that replace those outputs goes to the root.
        let alloc_start = observing.then(Instant::now);
        // One tile buffer per kernel, the topmost ones included, allocated once per run and
        // overwritten by each tile.
        let Ok(bufs) = region
            .kernels
            .iter()
            .map(|k| TileBuf::new(k.elem, t.min(n)))
            .collect::<Result<Vec<_>, _>>()
        else {
            return fail(inputs);
        };
        let Ok(mut reducer) = self.dispatch.reducer(self.reg, id) else {
            return fail(inputs);
        };
        let mut root_time = alloc_start.map_or(Duration::ZERO, |start| start.elapsed());
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
        let mut args = Vec::with_capacity(srcs.len());
        let mut feed = || -> Result<(), String> {
            // An empty region still runs one empty tile, so that every kernel outside a
            // narrowed operand is called once as in the whole evaluation.
            loop {
                let len = t.min(n - tiles.start);
                for src in srcs {
                    if let Src::Kernel(j) = *src {
                        tiles.run(j, 0, len, &mut [])?;
                    }
                }
                // A whole call never updates a reduction with no rows, and only an empty
                // region has a tile without rows.
                if len > 0 {
                    args.extend(srcs.iter().map(|src| tiles.rows_of(*src, 0, len)));
                    let begin = observing.then(Instant::now);
                    let updated = reducer.update(&args);
                    if let Some(begin) = begin {
                        root_time += begin.elapsed();
                    }
                    // The views of this tile's buffers go before the next tile overwrites
                    // them.
                    args.clear();
                    updated?;
                }
                tiles.start += len;
                if tiles.start >= n {
                    return Ok(());
                }
            }
        };
        let fed = feed();
        let Tiles {
            rows: kernel_rows,
            times,
            narrowed,
            ..
        } = tiles;
        let Ok(()) = fed else {
            return fail(inputs);
        };
        let begin = observing.then(Instant::now);
        let finished = reducer.finish();
        if let Some(begin) = begin {
            root_time += begin.elapsed();
        }
        let Ok(value) = finished else {
            return fail(inputs);
        };
        Ok(FusedRun {
            run: TiledRun {
                inputs,
                rows: kernel_rows,
                times,
                narrowed,
                value: Value::Scalar(value),
            },
            rows: n,
            root_time,
        })
    }
}
