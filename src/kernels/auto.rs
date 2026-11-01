//! The autovectorization backend, shared by all architectures: each entry compiles a portable
//! body under `#[target_feature]`, so the compiler vectorizes the inlined body for that ISA.
//! The bodies are plain Rust without FMA or reassociation, so the variants compute what the
//! scalar entries compute.

/// Emits module `$isa` with the auto entries for one instruction set. `$features` must be
/// the full feature string of the level of `$name`, because auto entries declare no extra
/// features and `select` checks exactly that level.
// Only x86-64 instantiates it today, see the `aarch64` module doc for the NEON pivot.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_macros))]
macro_rules! auto_backend {
    ($isa:ident, $name:literal, $features:literal) => {
        #[doc = concat!("Auto entries compiled for ", $name, ".")]
        pub(crate) mod $isa {
            use crate::kernels::{
                ElementwiseBody, Reducer, ReductionBody, ScanBody, Scanner, SelectionBody,
                WholeBody,
            };
            use crate::value::{Arg, ArrayBuilder, Scalar, Value};

            /// The elementwise entry of body `B`.
            // SAFETY: the caller guarantees that the CPU supports the features above, which
            // `Dispatch` does by calling only entries that `resolve` installed after `select`
            // found `features::runnable` true for the level. The body is safe Rust with bounds
            // checks, so it makes no pointer or alignment assumptions of its own.
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

            /// The whole entry of body `W`.
            // SAFETY: as for `elementwise`, the caller guarantees the CPU features and the
            // body is safe Rust.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn whole<W: WholeBody>(args: &[Value]) -> Result<Value, String> {
                W::run(args)
            }

            /// One update of scan body `S`, compiled for the ISA.
            // SAFETY: as for `elementwise`, the caller guarantees the CPU features and the
            // body is safe Rust.
            #[target_feature(enable = $features)]
            unsafe fn scan_update<S: ScanBody>(
                s: &mut S,
                args: &[Arg],
                out: &mut [u8],
            ) -> Result<(), String> {
                s.update(args, out)
            }

            /// A scanner whose updates run compiled for the ISA, see `OnIsa`.
            struct ScanOnIsa<S>(S);

            impl<S: ScanBody> Scanner for ScanOnIsa<S> {
                fn update(&mut self, args: &[Arg], out: &mut [u8]) -> Result<(), String> {
                    // SAFETY: a `ScanOnIsa` is only obtained through the unsafe `scanner`,
                    // whose caller guarantees the CPU features (`Dispatch` calls it only
                    // after `resolve` checked `features::runnable` through `select`).
                    // `update` makes no pointer or alignment assumptions.
                    unsafe { scan_update(&mut self.0, args, out) }
                }
            }

            /// The scanner constructor of body `S`.
            // SAFETY: the caller guarantees that the CPU supports the features above, and
            // the returned scanner relies on that for every `update`.
            pub(crate) unsafe fn scanner<S: ScanBody>() -> Box<dyn Scanner> {
                Box::new(ScanOnIsa(S::new()))
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
                    // `resolve` checked `features::runnable` through `select`). `update`
                    // makes no pointer or alignment assumptions.
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

#[cfg_attr(not(target_arch = "x86_64"), allow(unused_imports))]
pub(crate) use auto_backend;
