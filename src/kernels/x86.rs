//! x86-64 variants of the kernels. `auto` compiles the portable bodies for AVX2 and AVX-512.
//!
//! The entries installed here run only after `Registry::resolve` checked `Isa::is_available`
//! for their ISA, which is what makes calling them sound.

use crate::kernels::{
    Backend, ElementwiseBody, Isa, KernelImpl, ReductionBody, SelectionBody, Variant,
};

mod auto;

const AVX2_AUTO: Variant = Variant {
    isa: Isa::Avx2,
    backend: Backend::Auto,
};

const AVX512_AUTO: Variant = Variant {
    isa: Isa::Avx512,
    backend: Backend::Auto,
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
