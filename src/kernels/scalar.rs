//! Scalar reference kernels. They define the semantics every other implementation must match.

pub mod construct;
pub mod elementwise;
pub mod fused;
pub mod reduce;
pub mod scan;
pub mod select;
