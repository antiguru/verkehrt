//! x86-64 variants of the kernels. `auto` instantiates the shared auto backend macro to compile
//! the portable bodies for AVX2 and AVX-512, `compress` holds the hand-written intrinsics
//! entries of the selections, and `gather` those of `gather` on 64-bit rows.
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

/// The auto entries of the x86-64 ISAs, see `crate::kernels::auto`.
mod auto {
    use crate::kernels::auto::auto_backend;

    auto_backend!(avx2, "AVX2", "avx2");
    auto_backend!(avx512, "AVX-512", "avx512f,avx512bw,avx512vl,avx512vbmi2");
}

const AVX2_AUTO: Variant = Variant {
    isa: Isa::Avx2,
    backend: Backend::Auto,
};

const AVX512_AUTO: Variant = Variant {
    isa: Isa::Avx512,
    backend: Backend::Auto,
};

const AVX2_INTRINSICS: Variant = Variant {
    isa: Isa::Avx2,
    backend: Backend::Intrinsics,
};

const AVX512_INTRINSICS: Variant = Variant {
    isa: Isa::Avx512,
    backend: Backend::Intrinsics,
};

/// Adds the AVX2 and AVX-512 auto entries of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(impls: &mut [Option<KernelImpl>]) {
    impls[AVX2_AUTO.index()] = Some(KernelImpl::Elementwise(auto::avx2::elementwise::<B>));
    impls[AVX512_AUTO.index()] = Some(KernelImpl::Elementwise(auto::avx512::elementwise::<B>));
}

/// Adds the AVX2 and AVX-512 auto entries of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(impls: &mut [Option<KernelImpl>]) {
    impls[AVX2_AUTO.index()] = Some(KernelImpl::Reduction(auto::avx2::reducer::<R>));
    impls[AVX512_AUTO.index()] = Some(KernelImpl::Reduction(auto::avx512::reducer::<R>));
}

/// Adds the AVX2 and AVX-512 auto entries of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(impls: &mut [Option<KernelImpl>]) {
    impls[AVX2_AUTO.index()] = Some(KernelImpl::Selection(auto::avx2::selection::<S>));
    impls[AVX512_AUTO.index()] = Some(KernelImpl::Selection(auto::avx512::selection::<S>));
}

/// Adds the AVX2 and AVX-512 auto entries of a scan body to its table.
pub(crate) fn add_scan<S: ScanBody>(impls: &mut [Option<KernelImpl>]) {
    impls[AVX2_AUTO.index()] = Some(KernelImpl::Scan(auto::avx2::scanner::<S>));
    impls[AVX512_AUTO.index()] = Some(KernelImpl::Scan(auto::avx512::scanner::<S>));
}

/// Adds the AVX2 and AVX-512 auto entries of a whole body to its table.
pub(crate) fn add_whole<W: WholeBody>(impls: &mut [Option<KernelImpl>]) {
    impls[AVX2_AUTO.index()] = Some(KernelImpl::Whole(auto::avx2::whole::<W>));
    impls[AVX512_AUTO.index()] = Some(KernelImpl::Whole(auto::avx512::whole::<W>));
}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic type
/// is `tag::<T, BOOL>()`. AVX2 has none for `u8` and bool.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::compress_entries::<T, BOOL>();
    impls[AVX2_INTRINSICS.index()] = avx2.map(KernelImpl::Selection);
    impls[AVX512_INTRINSICS.index()] = avx512.map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::where_entries();
    impls[AVX2_INTRINSICS.index()] = Some(KernelImpl::Selection(avx2));
    impls[AVX512_INTRINSICS.index()] = Some(KernelImpl::Selection(avx512));
}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = compress::filter_entries::<T, C>();
    impls[AVX2_INTRINSICS.index()] = avx2.map(KernelImpl::Selection);
    impls[AVX512_INTRINSICS.index()] = avx512.map(KernelImpl::Selection);
}

/// Adds the intrinsics entries of `gather` on rows of storage type `T`, whose semantic type is
/// `tag::<T, BOOL>()`. Only 64-bit rows have them.
pub(crate) fn add_gather<T: Elem, const BOOL: bool>(impls: &mut [Option<KernelImpl>]) {
    let (avx2, avx512) = gather::gather_entries::<T, BOOL>();
    impls[AVX2_INTRINSICS.index()] = avx2.map(KernelImpl::Whole);
    impls[AVX512_INTRINSICS.index()] = avx512.map(KernelImpl::Whole);
}
