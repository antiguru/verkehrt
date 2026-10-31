//! x86-64 variants of the kernels. `auto` instantiates the shared auto backend macro to compile
//! the portable bodies for the x86-64-v3 and x86-64-v4 levels, `compress` holds the
//! hand-written intrinsics entries of the selections, `gather` those of `gather` on 64-bit
//! rows, and `scan` the x86-64-v4 entries of `scan_by_add` on 64-bit integers.
//!
//! Every entry is compiled for the full feature string of its level, plus the features it
//! declares. The entries installed here run only after `Registry::resolve` checked
//! `features::runnable` for them through `select`, which is what makes calling them sound.

use crate::kernels::{
    Backend, ElementwiseBody, Isa, KernelImpl, KernelTable, ReductionBody, ScanBody, SelectionBody,
    Variant, WholeBody,
};
use crate::value::Elem;

mod compress;
mod gather;
mod scan;

/// The auto entries of the x86-64 ISAs, see `crate::kernels::auto`.
mod auto {
    use crate::kernels::auto::auto_backend;

    auto_backend!(
        x86_64_v3,
        "x86-64-v3",
        "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3"
    );
    auto_backend!(
        x86_64_v4,
        "x86-64-v4",
        "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    );
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

/// Adds the x86-64-v3 and x86-64-v4 auto entries of an elementwise body to its table.
pub(crate) fn add_elementwise<B: ElementwiseBody>(impls: &mut KernelTable) {
    impls.set(
        V3_AUTO,
        KernelImpl::Elementwise(auto::x86_64_v3::elementwise::<B>),
    );
    impls.set(
        V4_AUTO,
        KernelImpl::Elementwise(auto::x86_64_v4::elementwise::<B>),
    );
}

/// Adds the x86-64-v3 and x86-64-v4 auto entries of a reduction body to its table.
pub(crate) fn add_reduction<R: ReductionBody>(impls: &mut KernelTable) {
    impls.set(
        V3_AUTO,
        KernelImpl::Reduction(auto::x86_64_v3::reducer::<R>),
    );
    impls.set(
        V4_AUTO,
        KernelImpl::Reduction(auto::x86_64_v4::reducer::<R>),
    );
}

/// Adds the x86-64-v3 and x86-64-v4 auto entries of a selection body to its table.
pub(crate) fn add_selection<S: SelectionBody>(impls: &mut KernelTable) {
    impls.set(
        V3_AUTO,
        KernelImpl::Selection(auto::x86_64_v3::selection::<S>),
    );
    impls.set(
        V4_AUTO,
        KernelImpl::Selection(auto::x86_64_v4::selection::<S>),
    );
}

/// Adds the x86-64-v3 and x86-64-v4 auto entries of a scan body to its table.
pub(crate) fn add_scan<S: ScanBody>(impls: &mut KernelTable) {
    impls.set(V3_AUTO, KernelImpl::Scan(auto::x86_64_v3::scanner::<S>));
    impls.set(V4_AUTO, KernelImpl::Scan(auto::x86_64_v4::scanner::<S>));
}

/// Adds the x86-64-v3 and x86-64-v4 auto entries of a whole body to its table.
pub(crate) fn add_whole<W: WholeBody>(impls: &mut KernelTable) {
    impls.set(V3_AUTO, KernelImpl::Whole(auto::x86_64_v3::whole::<W>));
    impls.set(V4_AUTO, KernelImpl::Whole(auto::x86_64_v4::whole::<W>));
}

/// Adds the intrinsics entries of `compress` on rows of storage type `T`, whose semantic type
/// is `tag::<T, BOOL>()`, with the features each declares. x86-64-v3 has none for `u8` and
/// bool.
pub(crate) fn add_compress<T: Elem, const BOOL: bool>(impls: &mut KernelTable) {
    let (v3, v4) = compress::compress_entries::<T, BOOL>();
    if let Some(entry) = v3 {
        impls.set(V3_INTRINSICS, KernelImpl::Selection(entry));
    }
    if let Some((entry, declared)) = v4 {
        impls.set_declared(V4_INTRINSICS, KernelImpl::Selection(entry), declared);
    }
}

/// Adds the intrinsics entries of `where`.
pub(crate) fn add_where(impls: &mut KernelTable) {
    let (v3, v4) = compress::where_entries();
    impls.set(V3_INTRINSICS, KernelImpl::Selection(v3));
    impls.set(V4_INTRINSICS, KernelImpl::Selection(v4));
}

/// Adds the intrinsics entries of `filter_<cmp>` on rows of type `T`, for comparison number
/// `C` of `CMPS`.
pub(crate) fn add_filter<T: Elem, const C: usize>(impls: &mut KernelTable) {
    let (v3, v4) = compress::filter_entries::<T, C>();
    if let Some(entry) = v3 {
        impls.set(V3_INTRINSICS, KernelImpl::Selection(entry));
    }
    if let Some(entry) = v4 {
        impls.set(V4_INTRINSICS, KernelImpl::Selection(entry));
    }
}

/// Adds the intrinsics entries of `gather` on rows of storage type `T`, whose semantic type is
/// `tag::<T, BOOL>()`. Only 64-bit rows have them.
pub(crate) fn add_gather<T: Elem, const BOOL: bool>(impls: &mut KernelTable) {
    let (v3, v4) = gather::gather_entries::<T, BOOL>();
    if let Some(entry) = v3 {
        impls.set(V3_INTRINSICS, KernelImpl::Whole(entry));
    }
    if let Some(entry) = v4 {
        impls.set(V4_INTRINSICS, KernelImpl::Whole(entry));
    }
}

/// Adds the intrinsics entries of `scan_by_add` on rows of storage type `T`, whose sum keeps
/// the element type. Only `i64` and `u64` have one, and only on x86-64-v4.
pub(crate) fn add_scan_by_add<T: Elem>(impls: &mut KernelTable) {
    if let Some(entry) = scan::scan_by_add_entry::<T>() {
        impls.set(V4_INTRINSICS, KernelImpl::Scan(entry));
    }
}
