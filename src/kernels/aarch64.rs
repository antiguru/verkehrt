//! AArch64 variants of the kernels, installed through the architecture layer, see
//! `crate::kernels::arch`.
//!
//! The auto backend on aarch64 is the baseline build, because NEON is mandatory on every
//! aarch64 target and the compiler already vectorizes the portable bodies for it. So no
//! `neon/auto` entries exist, the `neon/auto` slot stays empty, and `Registry::resolve` falls
//! back to the scalar entry for it. A later pivot to an auto backend with optional features
//! (for example `"dotprod,fp16,rdm"`) is one invocation of `crate::kernels::auto::auto_backend`
//! plus an `is_aarch64_feature_detected!` check in `Isa::is_available`.
//!
//! The entries installed here run only after `Registry::resolve` checked `Isa::is_available`
//! for their ISA, which is what makes calling them sound.

use crate::kernels::{ElementwiseBody, KernelImpl, ReductionBody, SelectionBody};
use crate::value::Elem;

/// Adds the ISA variants of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic
/// type is `tag::<T, BOOL>()`.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true) on
/// rows of type `T`.
pub(crate) fn add_extreme<T: Elem, const MAX: bool>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `argmin` (`MAX` false) or `argmax` (`MAX` true) on rows of
/// type `T`.
pub(crate) fn add_arg_extreme<T: Elem, const MAX: bool>(_impls: &mut [Option<KernelImpl>]) {}
