//! The choice sequence that every generator decision reads.

/// A sequence of choices. A read past the end is 0.
pub(super) struct Choices<'a> {
    data: &'a [u32],
    pos: usize,
}

impl<'a> Choices<'a> {
    pub(super) fn new(data: &'a [u32]) -> Choices<'a> {
        Choices { data, pos: 0 }
    }

    pub(super) fn next(&mut self) -> u32 {
        let v = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        v
    }

    /// A number in `0..n`.
    pub(super) fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "a choice needs an option");
        usize::try_from(self.next()).expect("u32 fits in usize") % n
    }

    /// True with `percent` probability, false on a 0 choice.
    pub(super) fn flip(&mut self, percent: usize) -> bool {
        self.below(100) >= 100 - percent
    }

    /// An index into `weights`, by weight. A 0 choice picks the first option of nonzero weight.
    pub(super) fn weighted(&mut self, weights: &[usize]) -> usize {
        let total: usize = weights.iter().sum();
        let mut v = self.below(total);
        for (i, w) in weights.iter().enumerate() {
            if v < *w {
                return i;
            }
            v -= w;
        }
        unreachable!("the value is below the total weight")
    }

    pub(super) fn pick<'b, T>(&mut self, xs: &'b [T]) -> &'b T {
        &xs[self.below(xs.len())]
    }
}
