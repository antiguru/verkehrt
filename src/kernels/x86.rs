//! x86-64 variants of the kernels. `auto` instantiates the shared auto backend macro to compile
//! the portable bodies for AVX2 and AVX-512, `compress` holds the hand-written intrinsics
//! entries of the selections, `gather` those of `gather` on 64-bit rows, and `scan` the
//! AVX-512 entries of `scan_by_add` on 64-bit integers.
//!
//! The entries installed here run only after `Registry::resolve` checked `Isa::is_available`
//! for their ISA, which is what makes calling them sound.

use crate::kernels::{
    Backend, ElementwiseBody, Isa, KernelImpl, ReductionBody, ScanBody, SelectionBody, Variant,
    WholeBody,
};
use crate::value::Elem;

mod compress;
mod gather;
mod scan;

/// The auto entries of the x86-64 ISAs, see `crate::kernels::auto`.
mod auto {
    use crate::kernels::auto::auto_backend;

    auto_backend!(x86_64_v3, "x86-64-v3", "avx2");
    auto_backend!(x86_64_v4, "x86-64-v4", "avx512f,avx512bw,avx512vl,avx512vbmi2");
}

const V3_AUTO: Variant = Variant {
    isa: Isa::X86_64V3,
    backend: Backend::Auto,
};

const V4_AUTO: Variant = Variant {
    isa: Isa::X86_64V4,
    backend: Backend::Auto,
};

const V3_INTRINSICS: Variant = Variant {
    isa: Isa::X86_64V3,
    backend: Backend::Intrinsics,
};

const V4_INTRINSICS: Variant = Variant {
    isa: Isa::X86_64V4,
    backend: Backend::Intrinsics,
};

/// Adds the AVX2 and AVX-512 auto entries of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(impls: &mut [Option<KernelImpl>]) {
    impls[V3_AUTO.index()] = Some(KernelImpl::Elementwise(auto::x86_64_v3::elementwise::<B>));
    impls[V4_AUTO.index()] = Some(KernelImpl::Elementwise(auto::x86_64_v4::elementwise::<B>));
}

/// Adds the AVX2 and AVX-512 auto entries of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(impls: &mut [Option<KernelImpl>]) {
    impls[V3_AUTO.index()] = Some(KernelImpl::Reduction(auto::x86_64_v3::reducer::<R>));
    impls[V4_AUTO.index()] = Some(KernelImpl::Reduction(auto::x86_64_v4::reducer::<R>));
}

/// Adds the AVX2 and AVX-512 auto entries of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(impls: &mut [Option<KernelImpl>]) {
    impls[V3_AUTO.index()] = Some(KernelImpl::Selection(auto::x86_64_v3::selection::<S>));
    impls[V4_AUTO.index()] = Some(KernelImpl::Selection(auto::x86_64_v4::selection::<S>));
}

/// Adds the AVX2 and AVX-512 auto entries of a scan body to its table.
pub(crate) fn add_scan<S: ScanBody>(impls: &mut [Option<KernelImpl>]) {
    impls[V3_AUTO.index()] = Some(KernelImpl::Scan(auto::x86_64_v3::scanner::<S>));
    impls[V4_AUTO.index()] = Some(KernelImpl::Scan(auto::x86_64_v4::scanner::<S>));
}

/// Adds the AVX2 and AVX-512 auto entries of a whole body to its table.
pub(crate) fn add_whole<W: WholeBody>(impls: &mut [Option<KernelImpl>]) {
    impls[V3_AUTO.index()] = Some(KernelImpl::Whole(auto::x86_64_v3::whole::<W>));
    impls[V4_AUTO.index()] = Some(KernelImpl::Whole(auto::x86_64_v4::whole::<W>));
}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic type
/// is `tag::<T, BOOL>()`. AVX2 has none for `u8` and bool.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::compress_entries::<T, BOOL>();
    impls[V3_INTRINSICS.index()] = avx2.map(KernelImpl::Selection);
    impls[V4_INTRINSICS.index()] = avx512.map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::where_entries();
    impls[V3_INTRINSICS.index()] = Some(KernelImpl::Selection(avx2));
    impls[V4_INTRINSICS.index()] = Some(KernelImpl::Selection(avx512));
}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::filter_entries::<T, C>();
    impls[V3_INTRINSICS.index()] = avx2.map(KernelImpl::Selection);
    impls[V4_INTRINSICS.index()] = avx512.map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `gather` on rows of storage type `T`, whose semantic type is
/// `tag::<T, BOOL>()`. Only 64-bit rows have them.
pub(crate) fn add_gather<T: Elem, const BOOL: bool>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = gather::gather_entries::<T, BOOL>();
    impls[V3_INTRINSICS.index()] = avx2.map(KernelImpl::Whole);
    impls[V4_INTRINSICS.index()] = avx512.map(KernelImpl::Whole);
}

/// Adds the intrinsics entries of `scan_by_add` on rows of storage type `T`, whose sum keeps
/// the element type. Only `i64` and `u64` have one, and only on AVX-512.
pub(crate) fn add_scan_by_add<T: Elem>(impls: &mut [Option<KernelImpl>]) {
    impls[V4_INTRINSICS.index()] = scan::scan_by_add_entry::<T>().map(KernelImpl::Scan);
}
