use super::TOPK_MAX;
use crate::kernels::scalar::construct::splitmix64;
use crate::kernels::scalar::fused::reference;
use crate::kernels::scalar::sort::tests::Key;
use crate::kernels::scalar::sort::{argsort_reference, argsort2_reference};
use crate::kernels::{Backend, FeatureSet, Isa, Registry};
use crate::value::{Array, Scalar, Value};

/// Runs the kernel `name` through its scalar entry and through its pattern's expansion, which
/// must agree, and returns the result.
fn call(name: &str, cols: &[Array], s: i64, e: i64) -> Result<Vec<i64>, String> {
    let reg = Registry::new();
    let id = reg.by_name(name).expect("top-k kernel");
    let mut args: Vec<Value> = cols.iter().cloned().map(Value::Array).collect();
    args.push(Value::Scalar(Scalar::I64(s)));
    args.push(Value::Scalar(Scalar::I64(e)));
    let dispatch = reg.resolve(Isa::Scalar, Backend::Auto, FeatureSet::EMPTY);
    let got = dispatch.call(&reg, id, &args);
    let want = reference(&reg, id, &args);
    match (&got, &want) {
        (Ok(Value::Array(g)), Ok(Value::Array(w))) => {
            assert_eq!(g.as_slice::<i64>(), w.as_slice::<i64>(), "{name} {s}..{e}");
        }
        (Err(g), Err(w)) => assert_eq!(g, w, "{name} {s}..{e}"),
        _ => panic!("{name} {s}..{e}: kernel {got:?} against pattern {want:?}"),
    }
    got.map(|v| match v {
        Value::Array(a) => a.as_slice::<i64>().to_vec(),
        other => panic!("expected an array, got {other:?}"),
    })
}

/// Rows `s..e` of `sorted`, or the error of `slice`.
fn sliced(sorted: &[i64], s: i64, e: i64) -> Result<Vec<i64>, String> {
    crate::kernels::scalar::index::slice_bounds(s, e, sorted.len())
        .map(|(s, e)| sorted[s..e].to_vec())
}

/// Inputs of `n` rows in the classes of the review focus: ascending, descending, all equal,
/// few distinct values, and random.
fn columns<T: Key>(n: usize, seed: u64) -> Vec<Vec<T>> {
    let n64 = u64::try_from(n).expect("length fits");
    let at = |i: u64| T::from_bits(splitmix64(seed + i));
    let few = |i: u64| T::from_bits(splitmix64(seed + i) % 3);
    vec![
        (0..n64).map(T::from_bits).collect(),
        (0..n64).rev().map(T::from_bits).collect(),
        vec![T::from_bits(7); n],
        (0..n64).map(few).collect(),
        (0..n64).map(at).collect(),
        // Ties across every cut: runs of four equal keys, descending.
        (0..n64).map(|i| T::from_bits(n64 / 4 - i / 4)).collect(),
    ]
}

/// The row counts and cuts of the review focus around `n` rows.
fn cuts(n: usize) -> Vec<(i64, i64)> {
    let to = |v: usize| i64::try_from(v).expect("bound fits");
    let mut es = vec![0, 1, 10, TOPK_MAX, TOPK_MAX + 1, TOPK_MAX + 2, n, n + 1];
    es.sort_unstable();
    es.dedup();
    let mut cuts = vec![];
    for e in es {
        for s in [0, 1, e / 2, e, e + 1] {
            cuts.push((to(s), to(e)));
        }
    }
    cuts.push((-1, 3));
    cuts.push((0, -1));
    cuts
}

const LENS: [usize; 9] = [0, 1, 5, 64, 65, 66, 100, 1025, 1500];

fn single<T: Key>() {
    let name = format!("topk_{}", T::TY.name());
    for n in LENS {
        for (seed, xs) in (0..).zip(columns::<T>(n, 1)) {
            let sorted = argsort_reference(&xs);
            let col = Array::from_vec(xs);
            for (s, e) in cuts(n) {
                assert_eq!(
                    call(&name, std::slice::from_ref(&col), s, e),
                    sliced(&sorted, s, e),
                    "{name} n={n} class={seed} {s}..{e}"
                );
            }
        }
    }
}

fn pair<A: Key, B: Key>() {
    let name = format!("topk2_{}_{}", A::TY.name(), B::TY.name());
    for n in [0, 1, 5, 65, 100, 300] {
        let (aa, bb) = (columns::<A>(n, 1), columns::<B>(n, 2));
        // Every class of the major key against every class of the minor key.
        for (ca, a) in (0..).zip(&aa) {
            for (cb, b) in (0..).zip(&bb) {
                let sorted = argsort2_reference(a, b);
                let cols = [Array::from_vec(a.clone()), Array::from_vec(b.clone())];
                for (s, e) in cuts(n) {
                    assert_eq!(
                        call(&name, &cols, s, e),
                        sliced(&sorted, s, e),
                        "{name} n={n} classes={ca},{cb} {s}..{e}"
                    );
                }
            }
        }
    }
}

macro_rules! topk_tests {
    ($($single:ident $t:ty, $pairs:ident [$($b:ty),+];)+) => {
        $(
            #[test]
            fn $single() {
                single::<$t>();
            }

            #[test]
            fn $pairs() {
                $(pair::<$t, $b>();)+
            }
        )+
    };
}

topk_tests! {
    topk_i64_matches_slice_of_argsort i64, topk2_i64_matches_slice_of_argsort2 [i64, u64, i32, u8];
    topk_u64_matches_slice_of_argsort u64, topk2_u64_matches_slice_of_argsort2 [i64, u64, i32, u8];
    topk_i32_matches_slice_of_argsort i32, topk2_i32_matches_slice_of_argsort2 [i64, u64, i32, u8];
    topk_u8_matches_slice_of_argsort u8, topk2_u8_matches_slice_of_argsort2 [i64, u64, i32, u8];
}

#[test]
fn topk_ties_across_the_cut_keep_position_order() {
    // Keys 2, 1, 1, 1, 0: the cut at 2 falls inside the run of ones.
    let xs = Array::from_vec(vec![2i64, 1, 1, 1, 0]);
    assert_eq!(
        call("topk_i64", std::slice::from_ref(&xs), 0, 2),
        Ok(vec![4, 1])
    );
    assert_eq!(
        call("topk_i64", std::slice::from_ref(&xs), 1, 3),
        Ok(vec![1, 2])
    );
    assert_eq!(call("topk_i64", &[xs], 2, 4), Ok(vec![2, 3]));
}

#[test]
fn topk_bounds_errors_name_the_slice() {
    let xs = Array::from_vec(vec![3i64, 1, 2]);
    let err = |s, e| call("topk_i64", std::slice::from_ref(&xs), s, e).unwrap_err();
    assert_eq!(err(2, 1), "slice 2..1 out of range for length 3");
    assert_eq!(err(0, 4), "slice 0..4 out of range for length 3");
    assert_eq!(err(-1, 2), "slice -1..2 out of range for length 3");
    assert_eq!(err(0, -1), "slice 0..-1 out of range for length 3");
    assert_eq!(
        err(0, i64::MAX),
        format!("slice 0..{} out of range for length 3", i64::MAX)
    );
}

#[test]
fn topk2_length_mismatch_wins_over_a_bounds_error() {
    let a = Array::from_vec(vec![1i64, 2, 3]);
    let b = Array::from_vec(vec![1i64, 2, 3, 4]);
    let got = call("topk2_i64_i64", &[a, b], 9, 2);
    assert_eq!(got, Err("length mismatch: 3 vs 4".into()));
}

#[test]
fn topk_kernels_cost_less_than_the_argsort_they_replace() {
    let reg = Registry::new();
    for (_, def) in reg.iter().filter(|(_, d)| d.name.starts_with("topk")) {
        let argsort = def.name.replacen("topk", "argsort", 1);
        let id = reg
            .by_name(&argsort)
            .unwrap_or_else(|| panic!("missing {argsort}"));
        assert!(def.cost < reg.get(id).cost, "{}", def.name);
    }
}
