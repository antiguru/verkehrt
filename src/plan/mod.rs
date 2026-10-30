//! Kernel selection. Each statement's term goes into an egglog e-graph with exact rewrites
//! and one rewrite per fused kernel pattern, and the cheapest cover by kernels is extracted.

#[cfg(test)]
mod tests;
