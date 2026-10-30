//! The selection loops of the intrinsics backends, shared by all architectures: `compress_*`,
//! `where`, and the fused `filter_*_*`.
//!
//! Each entry parses its arguments with the scalar entry's helpers, so it fails exactly when
//! the scalar entry fails. It counts the survivors of the tile first, then appends through
//! `append_chunks`, which reserves that many rows plus one vector. Within a chunk it selects
//! a vector of rows at a time while a whole vector store fits in the chunk's spare rows, and
//! finishes the rest with the portable predicated loop. The ISA modules provide the vector
//! steps, see `selection_loops`.

/// The number of set bits, as a row count.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
#[inline(always)]
pub(crate) fn ones(bits: u64) -> usize {
    usize::try_from(bits.count_ones()).expect("a bit count fits in usize")
}

/// Emits the chunk loops and the `compress`, `filter`, and `where` entries of one ISA,
/// compiled for `$features`, the feature string that `Isa::is_available` checks for `$name`.
///
/// The invoking module provides the vector steps, all compiled for `$features`:
/// * a trait `Lanes: Elem` with `const LANES: usize` and the unsafe methods `keep(mask:
///   *const u8) -> u64`, `load(src: *const Self) -> V`, and `compress(v: V, keep: u64, dst:
///   *mut Self) -> usize` for its vector type `V`, with the contracts of the AVX2 `Lanes`,
/// * a trait `Compare: Lanes + PartialOrd` with `unsafe fn compare<const C: usize>(v: V, c:
///   Self) -> u64`, the keep mask of `v[j] <cmp> c` with the predicate of `Cmp::test`,
/// * `fn first_indices(first: i64) -> V`, the vector of `i64` rows `first + j` for each lane
///   `j`, and `fn next_indices(idx: V) -> V`, `idx` advanced by `<i64 as Lanes>::LANES`.
// Only x86-64 and aarch64 instantiate it.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_macros)
)]
macro_rules! selection_loops {
    ($name:literal, $features:literal) => {
        /// Writes the rows of `x` whose `mask` byte is nonzero to the front of `dst` and
        /// returns their count. Panics unless `mask` has the rows of `x`, or when `dst` has no
        /// room for a predicated write of the survivors, see `predicated`.
        #[target_feature(enable = $features)]
        fn compress_chunk<T: Lanes>(x: &[T], mask: &[u8], dst: &mut [T]) -> usize {
            // The pointer arithmetic below relies on equal lengths, which the entry's caller
            // does not guarantee, so they are checked here.
            assert!(
                mask.len() == x.len(),
                "compress chunk of {} rows with {} mask rows",
                x.len(),
                mask.len()
            );
            let n = x.len();
            let (mut i, mut k) = (0, 0);
            // A vector step stores a whole vector, so it runs only while one fits in `dst`,
            // and the predicated loop finishes the rest.
            while i + T::LANES <= n && k + T::LANES <= dst.len() {
                // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` of `x` and `mask` are in
                // bounds, and `k + LANES <= dst.len()`, so rows `k..k + LANES` of `dst` are
                // in bounds. This function has the features of `Lanes`.
                unsafe {
                    let keep = T::keep(mask.as_ptr().add(i));
                    let v = T::load(x.as_ptr().add(i));
                    k += T::compress(v, keep, dst.as_mut_ptr().add(k));
                }
                i += T::LANES;
            }
            let rest = x[i..].iter().zip(&mask[i..]).map(|(&v, &m)| (v, m != 0));
            k + $crate::kernels::scalar::select::predicated(rest, &mut dst[k..])
        }

        /// The number of rows `v` of `x` with `v <cmp> c`, for comparison number `C` of
        /// `CMPS`, with the compares of `filter_chunk`.
        #[target_feature(enable = $features)]
        fn count_rows<T: Compare, const C: usize>(x: &[T], c: T) -> usize {
            use $crate::kernels::scalar::fused::{CMPS, count_cmp};
            let n = x.len();
            let (mut i, mut kept) = (0, 0);
            while i + T::LANES <= n {
                // SAFETY: `i + LANES <= n`, so rows `i..i + LANES` of `x` are in bounds. This
                // function has the features of `Compare`.
                kept += unsafe {
                    $crate::kernels::select_loops::ones(T::compare::<C>(
                        T::load(x.as_ptr().add(i)),
                        c,
                    ))
                };
                i += T::LANES;
            }
            kept + count_cmp(&x[i..], CMPS[C].1, c)
        }

        /// Writes the rows `v` of `x` with `v <cmp> c` to the front of `dst`, for comparison
        /// number `C` of `CMPS`, and returns their count. Panics when `dst` has no room for a
        /// predicated write of the survivors, see `predicated`.
        #[target_feature(enable = $features)]
        fn filter_chunk<T: Compare, const C: usize>(x: &[T], c: T, dst: &mut [T]) -> usize {
            let n = x.len();
            let (mut i, mut k) = (0, 0);
            // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
            while i + T::LANES <= n && k + T::LANES <= dst.len() {
                // SAFETY: as in `compress_chunk`, rows `i..i + LANES` of `x` and rows
                // `k..k + LANES` of `dst` are in bounds by the loop condition. This function
                // has the features of `Compare`.
                unsafe {
                    let v = T::load(x.as_ptr().add(i));
                    let keep = T::compare::<C>(v, c);
                    k += T::compress(v, keep, dst.as_mut_ptr().add(k));
                }
                i += T::LANES;
            }
            let cmp = $crate::kernels::scalar::fused::CMPS[C].1;
            let rest = x[i..].iter().map(|&v| (v, cmp.test(v, c)));
            k + $crate::kernels::scalar::select::predicated(rest, &mut dst[k..])
        }

        /// Writes `first + i` for every row `i` of `mask` whose byte is nonzero to the front
        /// of `dst` and returns their count. Panics when `dst` has no room for a predicated
        /// write of the survivors, see `predicated`. The caller guarantees that
        /// `first + mask.len()` fits in i64.
        #[target_feature(enable = $features)]
        fn where_chunk(mask: &[u8], first: i64, dst: &mut [i64]) -> usize {
            let n = mask.len();
            let lanes = <i64 as Lanes>::LANES;
            // Lane `j` holds the index of row `i + j`. The index of every row of the chunk
            // fits in i64, so every lane that is stored holds its exact index. Lanes past the
            // last row, after the final add, may wrap, which is harmless because they are
            // never stored and SIMD adds wrap without undefined behavior.
            let mut idx = first_indices(first);
            let (mut i, mut k) = (0, 0);
            // As in `compress_chunk`, vector steps run while a whole vector fits in `dst`.
            while i + lanes <= n && k + lanes <= dst.len() {
                // SAFETY: as in `compress_chunk`, bytes `i..i + lanes` of `mask` and rows
                // `k..k + lanes` of `dst` are in bounds by the loop condition. This function
                // has the features of `Lanes`.
                unsafe {
                    let keep = i64::keep(mask.as_ptr().add(i));
                    k += i64::compress(idx, keep, dst.as_mut_ptr().add(k));
                }
                idx = next_indices(idx);
                i += lanes;
            }
            let at = first + i64::try_from(i).expect("row index fits in i64");
            let rest = (at..).zip(&mask[i..]).map(|(r, &m)| (r, m != 0));
            k + $crate::kernels::scalar::select::predicated(rest, &mut dst[k..])
        }

        #[doc = concat!(
            "The ", $name,
            " `compress` entry on rows of storage type `T` and type `tag::<T, BOOL>()`."
        )]
        // SAFETY: the caller guarantees that the CPU supports the features above, which
        // `Dispatch` does by calling only entries that `resolve` installed after
        // `is_available` returned true. Memory safety does not depend on the caller:
        // `compress_args` checks that the rows and the mask have the same length, and
        // `compress_chunk` checks the lengths it relies on again.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn compress<T: Lanes, const BOOL: bool>(
            args: &[$crate::value::Arg],
            _base: usize,
            out: &mut $crate::value::ArrayBuilder,
        ) -> Result<(), String> {
            use $crate::kernels::scalar::select::{append_chunks, compress_args, count_nonzero};
            let (x, mask) =
                compress_args::<T>(args, $crate::kernels::scalar::elementwise::tag::<T, BOOL>())?;
            // The count pass sizes the output exactly. It is the portable count, compiled for
            // this ISA.
            append_chunks(out, x.len(), count_nonzero(mask), |rows, dst| {
                compress_chunk(&x[rows.clone()], &mask[rows], dst)
            });
            Ok(())
        }

        #[doc = concat!(
            "The ", $name,
            " `filter_<cmp>` entry on rows of type `T`, for comparison number `C` of `CMPS`."
        )]
        // SAFETY: as for `compress`, the caller guarantees the features above, and the chunk
        // functions check the lengths they rely on.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn filter<T: Compare, const C: usize>(
            args: &[$crate::value::Arg],
            _base: usize,
            out: &mut $crate::value::ArrayBuilder,
        ) -> Result<(), String> {
            let (x, c) = $crate::kernels::scalar::fused::filter_args::<T>(args)?;
            $crate::kernels::scalar::select::append_chunks(
                out,
                x.len(),
                count_rows::<T, C>(x, c),
                |rows, dst| filter_chunk::<T, C>(&x[rows], c, dst),
            );
            Ok(())
        }

        #[doc = concat!("The ", $name, " `where` entry.")]
        // SAFETY: as for `compress`, the caller guarantees the features above, and
        // `where_chunk` checks the lengths it relies on.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn r#where(
            args: &[$crate::value::Arg],
            base: usize,
            out: &mut $crate::value::ArrayBuilder,
        ) -> Result<(), String> {
            use $crate::kernels::scalar::select::{
                append_chunks, check_row_indices, count_nonzero, where_args,
            };
            let mask = where_args(args)?;
            check_row_indices(base, mask.len());
            append_chunks(out, mask.len(), count_nonzero(mask), |rows, dst| {
                let first = i64::try_from(base + rows.start).expect("row index fits in i64");
                where_chunk(&mask[rows], first, dst)
            });
            Ok(())
        }
    };
}

#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_imports)
)]
pub(crate) use selection_loops;
