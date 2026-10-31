//! AArch64 variants of the kernels, installed through the architecture layer, see
//! `crate::kernels::arch`.
//!
//! The auto backend on aarch64 is the baseline build, because NEON is mandatory on every
//! aarch64 target and the compiler may vectorize the portable bodies for it. So no
//! `neon/auto` entries exist, the `neon/auto` slot stays empty, and `Registry::resolve` falls
//! back to the scalar entry for it. A later pivot to an auto backend with optional features
//! (for example `"dotprod,fp16,rdm"`) is one invocation of `crate::kernels::auto::auto_backend`
//! plus an `is_aarch64_feature_detected!` check in `Isa::is_available`.
//!
//! `compress` holds the hand-written NEON intrinsics entries of the selections, and
//! `extreme` those of the float min, max, argmin, and argmax reductions.
//!
//! The entries installed here run only after `Registry::resolve` checked `Isa::is_available`
//! for their ISA, which is what makes calling them sound.

use crate::kernels::{
    Backend, ElementwiseBody, Isa, KernelImpl, ReductionBody, ScanBody, SelectionBody, Variant,
    WholeBody,
};
use crate::value::Elem;

mod compress;
mod extreme;

const NEON_INTRINSICS: Variant = Variant {
    isa: Isa::Neon,
    backend: Backend::Intrinsics,
};

/// Adds the ISA variants of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a scan body to its table.
pub(crate) fn add_scan<S: ScanBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the ISA variants of a whole body to its table.
pub(crate) fn add_whole<W: WholeBody>(_impls: &mut [Option<KernelImpl>]) {}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic
/// type is `tag::<T, BOOL>()`.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(impls: &mut [Option<KernelImpl>]) {
    impls[NEON_INTRINSICS.index()] =
        compress::compress_entry::<T, BOOL>().map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(impls: &mut [Option<KernelImpl>]) {
    impls[NEON_INTRINSICS.index()] = Some(KernelImpl::Selection(compress::where_entry()));
}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(impls: &mut [Option<KernelImpl>]) {
    impls[NEON_INTRINSICS.index()] = compress::filter_entry::<T, C>().map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true) on
/// rows of type `T`.
pub(crate) fn add_extreme<T: Elem, const MAX: bool>(impls: &mut [Option<KernelImpl>]) {
    impls[NEON_INTRINSICS.index()] = extreme::extreme_entry::<T, MAX>().map(KernelImpl::Reduction);
}

/// Adds the intrinsics entries of `argmin` (`MAX` false) or `argmax` (`MAX` true) on rows of
/// type `T`.
pub(crate) fn add_arg_extreme<T: Elem, const MAX: bool>(impls: &mut [Option<KernelImpl>]) {
    impls[NEON_INTRINSICS.index()] =
        extreme::arg_extreme_entry::<T, MAX>().map(KernelImpl::Reduction);
}
