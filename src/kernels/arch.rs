//! The architecture layer. Shared registration code calls the functions here, and each one
//! forwards to the module of the compiled architecture (`x86` or `aarch64`) or does nothing
//! on other targets. This keeps `target_arch` out of the registration code.

#[cfg(target_arch = "aarch64")]
use crate::kernels::aarch64;
#[cfg(target_arch = "x86_64")]
use crate::kernels::x86;
use crate::kernels::{
    ElementwiseBody, KernelTable, ReductionBody, ScanBody, SelectionBody, WholeBody,
};
use crate::value::Elem;

/// Adds the ISA variants of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_elementwise::<B>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_elementwise::<B>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the ISA variants of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_reduction::<R>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_reduction::<R>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the ISA variants of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_selection::<S>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_selection::<S>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the ISA variants of a scan body to its table.
pub(crate) fn add_scan<S: ScanBody>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_scan::<S>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_scan::<S>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the ISA variants of a whole body to its table.
pub(crate) fn add_whole<W: WholeBody>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_whole::<W>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_whole::<W>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic
/// type is `tag::<T, BOOL>()`.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_compress::<T, BOOL>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_compress::<T, BOOL>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_where(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_where(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_filter::<T, C>(impls);
    #[cfg(target_arch = "aarch64")]
    aarch64::add_filter::<T, C>(impls);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = impls;
}

/// Adds the intrinsics entries of `gather` on rows of storage type `T`, whose semantic type
/// is `tag::<T, BOOL>()`. aarch64 has none.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(clippy::extra_unused_type_parameters, clippy::needless_pass_by_ref_mut)
)]
pub(crate) fn add_gather<T: Elem, const BOOL: bool>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_gather::<T, BOOL>(impls);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = impls;
}

/// Adds the intrinsics entries of `scan_by_add` on rows of storage type `T`, whose sum keeps
/// the element type. aarch64 has none.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(clippy::extra_unused_type_parameters, clippy::needless_pass_by_ref_mut)
)]
pub(crate) fn add_scan_by_add<T: Elem>(impls: &mut KernelTable) {
    #[cfg(target_arch = "x86_64")]
    x86::add_scan_by_add::<T>(impls);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = impls;
}

/// Adds the intrinsics entries of `reduce_min` (`MAX` false) or `reduce_max` (`MAX` true) on
/// rows of type `T`. x86-64 has none.
#[cfg_attr(
    not(target_arch = "aarch64"),
    expect(clippy::extra_unused_type_parameters, clippy::needless_pass_by_ref_mut)
)]
pub(crate) fn add_extreme<T: Elem, const MAX: bool>(impls: &mut KernelTable) {
    #[cfg(target_arch = "aarch64")]
    aarch64::add_extreme::<T, MAX>(impls);
    #[cfg(not(target_arch = "aarch64"))]
    let _ = impls;
}

/// Adds the intrinsics entries of `argmin` (`MAX` false) or `argmax` (`MAX` true) on rows of
/// type `T`. x86-64 has none.
#[cfg_attr(
    not(target_arch = "aarch64"),
    expect(clippy::extra_unused_type_parameters, clippy::needless_pass_by_ref_mut)
)]
pub(crate) fn add_arg_extreme<T: Elem, const MAX: bool>(impls: &mut KernelTable) {
    #[cfg(target_arch = "aarch64")]
    aarch64::add_arg_extreme::<T, MAX>(impls);
    #[cfg(not(target_arch = "aarch64"))]
    let _ = impls;
}
