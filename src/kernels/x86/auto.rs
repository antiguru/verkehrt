//! The autovectorization backend: each entry compiles a portable body under
//! `#[target_feature]`, so the compiler vectorizes the inlined body for that ISA. The bodies
//! are plain Rust without FMA or reassociation, so the variants compute what the scalar
//! entries compute.

/// Emits module `$isa` with the auto entries for one instruction set. `$features` must be
/// the feature string that `Isa::is_available` checks for `$name`.
macro_rules! auto_backend {
    ($isa:ident, $name:literal, $features:literal) => {
        #[doc = concat!("Auto entries compiled for ", $name, ".")]
        pub(crate) mod $isa {
            use crate::kernels::{ElementwiseBody, Reducer, ReductionBody, SelectionBody};
            use crate::value::{Arg, ArrayBuilder, Scalar};

            /// The elementwise entry of body `B`.
            // SAFETY: the caller guarantees that the CPU supports the features above, which
            // `Dispatch` does by calling only entries that `resolve` installed after
            // `is_available` returned true. The body is safe Rust with bounds checks, so it
            // makes no pointer or alignment assumptions of its own.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn elementwise<B: ElementwiseBody>(
                args: &[Arg],
                out: &mut [u8],
            ) -> Result<(), String> {
                B::run(args, out)
            }

            /// The selection entry of body `S`.
            // SAFETY: as for `elementwise`, the caller guarantees the CPU features and the
            // body is safe Rust.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn selection<S: SelectionBody>(
                args: &[Arg],
                base: usize,
                out: &mut ArrayBuilder,
            ) -> Result<(), String> {
                S::run(args, base, out)
            }

            /// One update of reduction body `R`, compiled for the ISA.
            // SAFETY: as for `elementwise`, the caller guarantees the CPU features and the
            // body is safe Rust.
            #[target_feature(enable = $features)]
            unsafe fn update<R: ReductionBody>(r: &mut R, args: &[Arg]) -> Result<(), String> {
                r.update(args)
            }

            /// A reducer whose updates run compiled for the ISA. The type is private to this
            /// module and only `reducer` creates one, so holding one means that the caller
            /// of `reducer` guaranteed the CPU features.
            struct OnIsa<R>(R);

            impl<R: ReductionBody> Reducer for OnIsa<R> {
                fn update(&mut self, args: &[Arg]) -> Result<(), String> {
                    // SAFETY: an `OnIsa` is only obtained through the unsafe `reducer`, whose
                    // caller guarantees the CPU features (`Dispatch` calls it only after
                    // `resolve` checked `is_available`). `update` makes no pointer or
                    // alignment assumptions.
                    unsafe { update(&mut self.0, args) }
                }

                fn finish(self: Box<Self>) -> Result<Scalar, String> {
                    Box::new(self.0).finish()
                }
            }

            /// The reducer constructor of body `R`.
            // SAFETY: the caller guarantees that the CPU supports the features above, and
            // the returned reducer relies on that for every `update`.
            pub(crate) unsafe fn reducer<R: ReductionBody>() -> Box<dyn Reducer> {
                Box::new(OnIsa(R::new()))
            }
        }
    };
}

auto_backend!(avx2, "AVX2", "avx2");
auto_backend!(avx512, "AVX-512", "avx512f,avx512bw,avx512vl,avx512vbmi2");
