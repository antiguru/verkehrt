//! The per-tile runner of a tiled region: it runs the region's kernels over one tile at a time,
//! narrowing the guards' right operands to the rows their left operands leave open.

use std::time::{Duration, Instant};

use super::{Region, Src};
use crate::kernels::{Dispatch, Registry};
use crate::value::{Arg, TileBuf, Value};

#[cfg(test)]
mod tests;

/// The rows of a block that the scan for a guard's open rows skips at once.
const SCAN_BLOCK: usize = 64;

/// The state of a tiled region run while it runs the region's kernels over one tile.
pub(super) struct Tiles<'r> {
    pub(super) reg: &'r Registry,
    pub(super) dispatch: &'r Dispatch,
    pub(super) region: &'r Region,
    /// The inputs' values, in the order of `Region::inputs`.
    pub(super) inputs: &'r [Value],
    /// Per region kernel, see `Machine::guards`.
    pub(super) guards: &'r [Option<bool>],
    /// One tile buffer per inner kernel, where a kernel writes the rows of the tile it runs
    /// over at their positions in the tile.
    pub(super) bufs: Vec<TileBuf>,
    /// The first row of the tile in the region's arrays.
    pub(super) start: usize,
    /// The arguments of a tile call, kept to reuse their allocation.
    pub(super) args: Vec<Arg>,
    /// Per region kernel, the rows of its tile calls so far.
    pub(super) rows: Vec<usize>,
    /// Per region kernel, its time so far. Empty unless observing.
    pub(super) times: Vec<Duration>,
    /// See `TiledRun::narrowed`.
    pub(super) narrowed: u64,
}

impl Tiles<'_> {
    /// Runs the subtree of region kernel `j` over the rows `lo..hi` of the tile, writing the
    /// root's rows into `out`, the region's output. A guard runs its right operand and itself
    /// only over the rows from the first to the last row of `lo..hi` where its left operand
    /// leaves the result open, and fills its other rows with the result they have whatever
    /// the right operand.
    pub(super) fn run(
        &mut self,
        j: usize,
        lo: usize,
        hi: usize,
        out: &mut [u8],
    ) -> Result<(), String> {
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
