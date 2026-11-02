use super::super::tests::probe_registry;
use crate::error::{ErrorKind, Span};
use crate::exec::Tiling;
use crate::exec::tests::{
    Builder, compare_narrowed, eval_node, eval_selective, rows, run, text, tilings, variants,
};
use crate::kernels::{Backend, Isa};
use crate::lower::Leaf;
use crate::value::{Array, Scalar, Value};

/// The tile row counts of the tiled runs of `tilings`, in order.
fn tile_rows() -> Vec<usize> {
    tilings()
        .into_iter()
        .filter_map(|t| match t {
            Tiling::Rows(n) => Some(n.get()),
            Tiling::Off => None,
        })
        .collect()
}

/// The tiles of `t` rows in which the rows from the first to the last row of `open` span
/// fewer rows than the tile, which a guard whose left operand leaves exactly the rows of
/// `open` open narrows.
fn narrowed(open: &[bool], t: usize) -> u64 {
    let tiles = open.chunks(t).filter(|tile| {
        match (tile.iter().position(|&o| o), tile.iter().rposition(|&o| o)) {
            (Some(first), Some(last)) => last + 1 - first < tile.len(),
            _ => true,
        }
    });
    u64::try_from(tiles.count()).unwrap()
}

/// Checks `src` like `compare_narrowed`, that it binds every name of `want` to its rows,
/// and that every selective run narrows `expect(t)` tiles for its tile row count `t`.
fn assert_narrows(src: &str, want: &[(&str, Vec<String>)], expect: impl Fn(usize) -> u64) {
    let names: Vec<&str> = want.iter().map(|(n, _)| *n).collect();
    let (outcome, counts, narrowed) = compare_narrowed(src, &names);
    let (_, bound) = outcome.unwrap();
    for ((name, rows), got) in want.iter().zip(bound) {
        assert_eq!(&got, rows, "{name} of\n{src}");
    }
    assert!(counts.iter().all(|&c| c > 0), "{counts:?} for\n{src}");
    let tiles = tile_rows();
    let expected: Vec<u64> = variants()
        .iter()
        .flat_map(|_| tiles.iter().map(|&t| expect(t)))
        .collect();
    assert_eq!(narrowed, expected, "narrowed tiles of\n{src}");
}

#[test]
fn and_narrows_to_the_rows_its_left_operand_keeps() {
    // The left operand is false in whole tiles below row 40, so they skip the right operand.
    let src = "let x = iota(100);
let m = and(x >= 40, x != 70);";
    let open: Vec<bool> = (0..100).map(|x| x >= 40).collect();
    let m = (0..100).map(|x| x >= 40 && x != 70);
    assert_narrows(src, &[("m", text(m))], |t| narrowed(&open, t));
}

#[test]
fn and_narrows_a_tile_false_at_both_ends() {
    // The left operand is a guard itself, true in rows 40 to 59 only, so a tile around them
    // is false at both ends. Its own left operand narrows too.
    let src = "let x = iota(100);
let m = and(and(x >= 40, x < 60), x != 50);";
    let inner: Vec<bool> = (0..100).map(|x| x >= 40).collect();
    let outer: Vec<bool> = (0..100).map(|x| (40..60).contains(&x)).collect();
    let m = (0..100).map(|x| (40..60).contains(&x) && x != 50);
    assert_narrows(src, &[("m", text(m))], |t| {
        narrowed(&inner, t) + narrowed(&outer, t)
    });
}

#[test]
fn and_with_a_left_operand_true_throughout_narrows_nothing() {
    let src = "let x = iota(100);
let m = and(x >= 0, x != 70);";
    let m = (0..100).map(|x| x != 70);
    assert_narrows(src, &[("m", text(m))], |_| 0);
}

#[test]
fn or_narrows_to_the_rows_its_left_operand_leaves_false() {
    let src = "let x = iota(100);
let m = or(x < 30, x > 80);";
    let open: Vec<bool> = (0..100).map(|x| x >= 30).collect();
    let m = (0..100).map(|x| !(30..=80).contains(&x));
    assert_narrows(src, &[("m", text(m))], |t| narrowed(&open, t));
}

#[test]
fn nested_guards_narrow_within_their_parent_range() {
    // The inner guard runs over the rows the outer one gives it, and narrows them further.
    let src = "let x = iota(100);
let m = and(x >= 20, and(x < 70, or(x < 30, x > 60)));
let r = or(x < 10, and(x > 50, x != 75));";
    let m = (0..100).map(|x| (20..70).contains(&x) && !(30..=60).contains(&x));
    let r = (0..100).map(|x| x < 10 || (x > 50 && x != 75));
    let (outcome, counts, narrowed) = compare_narrowed(src, &["m", "r"]);
    assert_eq!(outcome.unwrap().1, [text(m), text(r)]);
    assert!(counts.iter().all(|&c| c == 2), "{counts:?}");
    assert!(narrowed.iter().all(|&c| c > 0), "{narrowed:?}");
}

#[test]
fn a_guard_feeding_select_fills_its_tile_buffer() {
    // The guard is an inner kernel whose output is a tile buffer that `select` reads.
    let src = "let x = iota(100);
let s = map(x, |v| if and(v > 20, v < 60) { v } else { 0 });
let o = map(x, |v| if or(v < 20, v > 60) { 0 } else { v });";
    let open: Vec<bool> = (0..100).map(|x| x > 20).collect();
    let open_or: Vec<bool> = (0..100).map(|x| x >= 20).collect();
    let s = (0..100).map(|x| if x > 20 && x < 60 { x } else { 0 });
    let o = (0..100).map(|x| if (20..=60).contains(&x) { x } else { 0 });
    assert_narrows(src, &[("s", text(s)), ("o", text(o))], |t| {
        narrowed(&open, t) + narrowed(&open_or, t)
    });
}

#[test]
fn a_region_input_array_guards_as_a_left_operand() {
    let src = "let x = iota(100);
let b = x > 30;
let m = and(b, x < 60);";
    let open: Vec<bool> = (0..100).map(|x| x > 30).collect();
    let m = (0..100).map(|x| x > 30 && x < 60);
    assert_narrows(src, &[("m", text(m))], |t| narrowed(&open, t));
}

#[test]
fn a_right_operand_that_is_a_region_input_is_not_narrowed() {
    // In the second program the right operand is a shared node, a region input too.
    let src = "let x = iota(100);
let b = x > 30;
let m = and(x < 60, b);";
    let m = (0..100).map(|x| x > 30 && x < 60);
    assert_narrows(src, &[("m", text(m))], |_| 0);
    let shared = "let x = iota(100);
let s = sum(compress(x, and(x < 60, (x * 2) > 50)));
let o = sum(compress(x, or(x > 90, (x * 2) > 50)));";
    let s: i64 = (26..60).sum();
    let o: i64 = (26..100).sum();
    assert_narrows(shared, &[("s", text([s])), ("o", text([o]))], |_| 0);
}

#[test]
fn a_guard_over_a_view_narrows() {
    // The view starts at row 7 of its buffer.
    let src = "let x = slice(iota(120), 7, 107);
let m = and(x > 20, x < 50);";
    let open: Vec<bool> = (7..107).map(|x| x > 20).collect();
    let m = (7..107).map(|x| x > 20 && x < 50);
    assert_narrows(src, &[("m", text(m))], |t| narrowed(&open, t));
}

#[test]
fn a_guard_over_an_empty_array_narrows_nothing() {
    let src = "let e = slice(iota(10), 3, 3);
let m = and(e > 1, e < 5);
let o = or(e > 2, e < 4);";
    let none = Vec::<String>::new();
    assert_narrows(src, &[("m", none.clone()), ("o", none)], |_| 0);
}

#[test]
fn statistics_count_the_rows_a_narrowed_kernel_ran_over() {
    // The right operand `ne_i64` runs over rows 40 to 99 in every tiling, reading 8 bytes
    // of the column and writing 1 per row. The guard reads two bytes per row.
    let src = "let x = iota(100);
let m = and(x > 39, x != 70);";
    let line = |lines: &[String], name: &str| {
        lines
            .iter()
            .find(|l| l.starts_with(&format!("stats {name} ")))
            .unwrap_or_else(|| panic!("no stats line of {name} in {lines:?}"))
            .clone()
    };
    let read_rows = |l: &str, width: u64| {
        let read = l.split(' ').find_map(|w| w.strip_prefix("read=")).unwrap();
        read.parse::<u64>().unwrap() / width
    };
    for variant in variants() {
        for t in tilings().into_iter().skip(1) {
            let at = format!("{variant:?} {t:?}");
            let on = run(src, &["m"], variant, t, true);
            let ne = line(&on.lines, "ne_i64");
            assert_eq!(read_rows(&ne, 8), 60, "{at}: {ne}");
            assert_eq!(ne, "stats ne_i64 calls=1 bytes=540 read=480", "{at}");
            let and = line(&on.lines, "and_bool");
            assert_eq!(and, "stats and_bool calls=1 bytes=180 read=120", "{at}");
            let lt = line(&on.lines, "lt_i64");
            assert_eq!(lt, "stats lt_i64 calls=1 bytes=900 read=800", "{at}");
            assert!(
                on.lines
                    .iter()
                    .any(|l| l.starts_with("trace ne_i64 ")
                        && l.ends_with(" n=60 read=480 write=60")),
                "{at}: {:?}",
                on.lines
            );
            let off = run(src, &["m"], variant, t, false);
            let whole = line(&off.lines, "ne_i64");
            assert_eq!(whole, "stats ne_i64 calls=1 bytes=900 read=800", "{at}");
        }
    }
}

#[test]
fn a_right_operand_that_can_fail_is_not_skipped() {
    // `x` is at most 5 in every row, so narrowing would skip the right operand everywhere
    // and hide the division by zero of the whole evaluation.
    let src = "let x = iota(40) - 100;
let y = iota(40);
let z = length(slice(iota(3), 0, 0));
print(and(x > 5, div_const(y, z) > 0));";
    let (outcome, counts, narrowed) = compare_narrowed(src, &[]);
    let e = outcome.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Runtime);
    assert_eq!(e.message, "div_const: division by zero");
    assert_eq!(e.span, Some(Span { line: 4, col: 1 }));
    assert!(counts.iter().all(|&c| c == 0), "{counts:?}");
    assert!(narrowed.iter().all(|&c| c == 0), "{narrowed:?}");
}

#[test]
fn a_kernel_failing_after_narrowed_tiles_raises_the_untiled_error() {
    let (reg, fail, _, _) = probe_registry();
    let id = |n| reg.by_name(n).unwrap();
    let mut b = Builder::default();
    // select(and(10 < x, x < 20), fail_at(x), x): the guard narrows every tile not inside
    // rows 11 to 19, and `fail_at` fails at row 90 over 0..100, after tiles that narrowed.
    let (x, ten, twenty) = (b.leaf(0), b.bcast(1), b.bcast(2));
    let above = b.kernel(id("lt_i64"), vec![ten, x]);
    let below = b.kernel(id("lt_i64"), vec![x, twenty]);
    let guard = b.kernel(id("and_bool"), vec![above, below]);
    let failed = b.kernel(fail, vec![x]);
    let root = b.kernel(id("select_i64"), vec![guard, failed, x]);
    let node = b.finish(root);
    let leaves = |n: i64| {
        vec![
            Leaf::Const(Value::Array(Array::from_vec((0..n).collect()))),
            Leaf::Const(Value::Scalar(Scalar::I64(10))),
            Leaf::Const(Value::Scalar(Scalar::I64(20))),
        ]
    };
    let scalar = (Isa::Scalar, Backend::Auto);
    let (untiled, _) = eval_node(&reg, &node, &leaves(100), scalar, Tiling::Off);
    let want = untiled.unwrap_err();
    assert_eq!(want.message, "fail_at: row 90 holds 90");
    assert_eq!(want.span, Some(Span { line: 3, col: 1 }));
    let want_rows = text(0..80);
    for variant in variants() {
        for tiling in tilings().into_iter().skip(1) {
            for selective in [false, true] {
                let at = format!("{variant:?} {tiling:?} selective {selective}");
                let attempt = eval_selective(&reg, &node, &leaves(100), variant, tiling, selective);
                assert_eq!(attempt.0.unwrap_err(), want, "{at}");
                assert_eq!((attempt.1, attempt.2), (0, 0), "{at}");
                // Over 0..80 the run completes, and the selective one narrows.
                let (got, regions, narrowed) =
                    eval_selective(&reg, &node, &leaves(80), variant, tiling, selective);
                assert_eq!(rows(&got.unwrap()), want_rows, "{at}");
                assert_eq!(regions, 1, "{at}");
                assert_eq!(narrowed > 0, selective, "{at}");
            }
        }
    }
}

#[test]
fn the_open_span_scan_matches_a_row_by_row_scan() {
    // Runs of open rows at offsets around the 64-row blocks the scan skips at once.
    let naive = |rows: &[u8], open: bool| {
        let is_open = |b: &u8| (*b != 0) == open;
        let first = rows.iter().position(is_open)?;
        Some((first, rows.iter().rposition(is_open).unwrap() + 1))
    };
    for n in [0, 1, 63, 64, 65, 127, 128, 129, 200, 300] {
        let mut patterns: Vec<Vec<u8>> = vec![vec![0; n], vec![1; n]];
        for a in (0..n).step_by(7) {
            for b in (a..=n).step_by(11) {
                patterns.push((0..n).map(|i| u8::from((a..b).contains(&i))).collect());
                patterns.push((0..n).map(|i| u8::from(i == a || i + 1 == b)).collect());
            }
        }
        for rows in patterns {
            for open in [true, false] {
                assert_eq!(
                    super::open_span(&rows, open),
                    naive(&rows, open),
                    "{open} in {rows:?}"
                );
            }
        }
    }
}

#[test]
fn guards_over_tiles_longer_than_the_scan_blocks_narrow() {
    // 100-row tiles over 300 rows put whole 64-row blocks without open rows at the start and
    // the end of a tile, which the scan skips at once: rows 100 to 163 for `h`, rows 136 to
    // 199 for `t`, and whole tiles for `a`.
    let src = "let x = iota(300);
let a = and(and(x >= 110, x < 190), x != 150);
let o = or(x < 30, x != 250);
let h = and(x >= 180, x != 190);
let t = and(x < 130, x != 5);";
    let inner: Vec<bool> = (0..300).map(|x| x >= 110).collect();
    let outer: Vec<bool> = (0..300).map(|x| (110..190).contains(&x)).collect();
    let open_or: Vec<bool> = (0..300).map(|x| x >= 30).collect();
    let a = text((0..300).map(|x| (110..190).contains(&x) && x != 150));
    let o = text((0..300).map(|x| x < 30 || x != 250));
    let h = text((0..300).map(|x| x >= 180 && x != 190));
    let t = text((0..300).map(|x| x < 130 && x != 5));
    let open_h: Vec<bool> = (0..300).map(|x| x >= 180).collect();
    let open_t: Vec<bool> = (0..300).map(|x| x < 130).collect();
    let tiling = Tiling::Rows(std::num::NonZeroUsize::new(100).unwrap());
    let want = [inner, outer, open_or, open_h, open_t]
        .iter()
        .map(|open| narrowed(open, 100))
        .sum::<u64>();
    assert!(want > 0);
    for variant in variants() {
        let names = ["a", "o", "h", "t"];
        let off = run(src, &names, variant, tiling, false);
        let on = run(src, &names, variant, tiling, true);
        let (_, bound) = on.outcome.as_ref().unwrap();
        let rows = [a.clone(), o.clone(), h.clone(), t.clone()];
        assert_eq!(bound, &rows, "{variant:?}");
        assert_eq!(on.outcome, off.outcome, "{variant:?}");
        assert_eq!(on.regions, 4, "{variant:?}");
        assert_eq!(on.narrowed, want, "{variant:?}");
        assert_eq!(off.narrowed, 0, "{variant:?}");
    }
}

#[test]
fn selective_evaluation_without_tiles_changes_nothing() {
    let src = "let x = iota(100);
let m = and(and(x >= 40, x < 60), x != 50);";
    for variant in variants() {
        let off = run(src, &["m"], variant, Tiling::Off, false);
        let on = run(src, &["m"], variant, Tiling::Off, true);
        assert_eq!(on.outcome, off.outcome, "{variant:?}");
        assert_eq!(on.lines, off.lines, "{variant:?}");
        assert_eq!((on.regions, on.narrowed), (0, 0), "{variant:?}");
    }
}
