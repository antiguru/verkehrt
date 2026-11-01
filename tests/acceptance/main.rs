//! Acceptance of the SQL query programs against DuckDB's answers on the corgi data. The
//! suites read `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH`, the directories `prepare.py` wrote,
//! and skip when the variable is unset.

mod harness;

use std::fs;
use std::path::{Path, PathBuf};

use verkehrt::value::{Array, Value};

use verkehrt::columns::CLICKBENCH_SIGNED;

use harness::{format_general, format_rows, run_query};

/// A fresh directory under the cargo target directory, named after the test.
fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_u64(dir: &Path, file: &str, v: &[u64]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(dir.join(file), bytes).unwrap();
}

#[test]
fn harness_formats_like_prepare_py() {
    let exports = vec![
        (
            "a".to_string(),
            Value::Array(Array::from_vec(vec![65u8, 78])),
        ),
        (
            "b".to_string(),
            Value::Array(Array::from_vec(vec![3i64, 4])),
        ),
        (
            "c".to_string(),
            Value::Array(Array::from_vec(vec![0.5f64, 1.0 / 3.0])),
        ),
    ];
    let got = format_rows(&['s', 'u', 'f'], &exports).unwrap();
    assert_eq!(got, "41\t3\t0.5000\n4e\t4\t0.3333\n");
}

#[test]
fn harness_formats_g_like_prepare_py() {
    let one = |x: f64| {
        let exports = vec![("a".to_string(), Value::Array(Array::from_vec(vec![x])))];
        format_rows(&['g'], &exports).unwrap()
    };
    assert_eq!(one(1.94819487789e18), "1.94819487789e+18\n");
    assert_eq!(one(-0.00123), "-1.23000000000e-03\n");
    assert_eq!(one(0.0), "0.00000000000e+00\n");
    assert_eq!(one(1.5e100), "1.50000000000e+100\n");
}

#[test]
fn harness_rejects_non_finite_g() {
    for x in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let err = format_general(x).unwrap_err();
        assert!(err.contains("general float"), "{err}");
    }
}

#[test]
fn harness_runs_a_synthetic_query() {
    let dir = scratch("synthetic");
    write_u64(&dir, "a.u64", &[5, 6, 7, 8]);
    write_u64(&dir, "g.ends", &[1, 2, 3, 4]);
    fs::write(dir.join("g.bytes"), b"ABAB").unwrap();
    let program = dir.join("q.vk");
    fs::write(
        &program,
        "// sql: SELECT a, g FROM t WHERE g = 'A'\n\
         // output: [u,s]\n\
         input a: u64[];\n\
         input g_bytes: u8[];\n\
         let m = g_bytes == 65u8;\n\
         export a_out = compress(a, m);\n\
         export g_out = compress(g_bytes, m);\n",
    )
    .unwrap();
    let mut expected = String::new();
    for (a, g) in [(5u64, b'A'), (6, b'B'), (7, b'A'), (8, b'B')] {
        if g == b'A' {
            expected.push_str(&format!("{a}\t{g:02x}\n"));
        }
    }
    assert_eq!(run_query(&dir, &program, &[]).unwrap(), expected);
}

/// A q07-shaped program: a group-by whose bucket count comes from the data.
fn q07_program(guard: bool) -> String {
    let nb = if guard {
        "if length(k) > 0 { reduce(k, max) + 1 } else { 0 }"
    } else {
        "reduce(k, max) + 1"
    };
    format!(
        "// output: [u]\n\
         input a: u64[];\n\
         let k = cast_i64(compress(a, a > 100u64));\n\
         let nb = {nb};\n\
         export c = reduce_into(k, k, nb, add);\n"
    )
}

#[test]
fn harness_empty_filter_gives_empty_output() {
    let dir = scratch("empty-filter");
    write_u64(&dir, "a.u64", &[1, 2, 3, 4]);
    let program = dir.join("q.vk");
    fs::write(&program, q07_program(true)).unwrap();
    assert_eq!(run_query(&dir, &program, &[]).unwrap(), "");
    fs::write(&program, q07_program(false)).unwrap();
    let err = run_query(&dir, &program, &[]).unwrap_err();
    assert!(err.contains("empty array"), "{err}");
}

/// Runs every program of `queries` against `var`'s directory. `signed` lists the columns that
/// directory stores with the sign bit flipped.
fn run_suite(var: &str, queries: &str, signed: &[&str]) {
    let Ok(dir) = std::env::var(var) else {
        eprintln!("skipping {queries}: {var} is not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let programs = PathBuf::from("tests/acceptance").join(queries);
    let mut entries: Vec<PathBuf> = fs::read_dir(&programs)
        .unwrap_or_else(|e| panic!("program directory {}: {e}", programs.display()))
        .map(|d| d.unwrap().path())
        .collect();
    entries.sort();
    let mut failures = Vec::new();
    let mut ran = 0;
    for path in entries {
        if path.extension().is_none_or(|e| e != "vk") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let expected = fs::read_to_string(dir.join("expected").join(format!("{name}.txt")))
            .unwrap_or_else(|e| panic!("expected answer of {name}: {e}"));
        ran += 1;
        match run_query(&dir, &path, signed) {
            Ok(got) if got == expected => {}
            Ok(got) => failures.push(format!(
                "{name}: answer differs from DuckDB's, {}",
                first_diff(&expected, &got)
            )),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(ran > 0, "no .vk programs in {}", programs.display());
    eprintln!("{queries}: {ran} queries");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The first differing line of two answers, for the failure message.
fn first_diff(expected: &str, got: &str) -> String {
    let (e, g): (Vec<_>, Vec<_>) = (expected.lines().collect(), got.lines().collect());
    for i in 0..e.len().max(g.len()) {
        if e.get(i) != g.get(i) {
            return format!(
                "line {}: expected {:?}, got {:?}",
                i + 1,
                e.get(i),
                g.get(i)
            );
        }
    }
    "no differing line".to_string()
}

#[test]
fn clickbench_matches_duckdb() {
    run_suite("VERKEHRT_CLICKBENCH", "clickbench", CLICKBENCH_SIGNED);
}

#[test]
fn tpch_matches_duckdb() {
    run_suite("VERKEHRT_TPCH", "tpch", &[]);
}

/// Prints the plan of every query program, for the measurement report. Run with
/// `-- --ignored --nocapture`.
#[test]
#[ignore = "prints plans for the measurement report"]
fn acceptance_plans() {
    let interp = verkehrt::Interpreter::new(verkehrt::Options {
        show_plan: true,
        ..Default::default()
    })
    .unwrap();
    for suite in ["tpch", "clickbench"] {
        let mut entries: Vec<PathBuf> = fs::read_dir(PathBuf::from("tests/acceptance").join(suite))
            .unwrap()
            .map(|d| d.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "vk"))
            .collect();
        entries.sort();
        for path in entries {
            let program = fs::read_to_string(&path).unwrap();
            println!("== {suite}/{}", path.file_stem().unwrap().to_string_lossy());
            let mut plan = Vec::new();
            interp.prepare_to(&program, &mut plan).unwrap();
            print!("{}", String::from_utf8(plan).unwrap());
        }
    }
}

#[test]
fn harness_flips_signed_columns_back() {
    let dir = scratch("signed");
    // -1, 5, and i64::MIN as `prepare.py` stores them, with bit 63 flipped.
    let flip = 1u64 << 63;
    write_u64(&dir, "UserID.u64", &[u64::MAX ^ flip, 5 ^ flip, 0]);
    // An unlisted column keeps its words as stored.
    write_u64(&dir, "CounterID.u64", &[1, 2, 3]);
    let program = dir.join("q.vk");
    fs::write(
        &program,
        "// output: [i,u]\n\
         input UserID: i64[];\n\
         input CounterID: i64[];\n\
         export u = UserID;\n\
         export c = CounterID;\n",
    )
    .unwrap();
    assert_eq!(
        run_query(&dir, &program, CLICKBENCH_SIGNED).unwrap(),
        format!("-1\t1\n5\t2\n{}\t3\n", i64::MIN)
    );
}

#[test]
fn harness_reads_ends_inputs() {
    let dir = scratch("ends");
    // Rows of length 3, 0, and 4.
    write_u64(&dir, "URL.ends", &[3, 3, 7]);
    let program = dir.join("q.vk");
    fs::write(
        &program,
        "// output: [u]\n\
         input URL_ends: i64[];\n\
         export l = URL_ends - prev(URL_ends, 0);\n",
    )
    .unwrap();
    assert_eq!(run_query(&dir, &program, &[]).unwrap(), "3\n0\n4\n");
}

/// Stores a signed column the way `prepare.py` does, with bit 63 flipped.
fn write_signed(dir: &Path, file: &str, v: &[i64]) {
    let words: Vec<u64> = v.iter().map(|x| x.cast_unsigned() ^ (1 << 63)).collect();
    write_u64(dir, file, &words);
}

/// The real q19 on a column with matches and near misses, since the data has no match.
#[test]
fn q19_finds_the_constant_user_id() {
    const ID: i64 = 435090932899640449;
    let dir = scratch("q19");
    write_signed(
        &dir,
        "UserID.u64",
        &[ID, ID - 1, -5, ID, ID + 1, i64::MIN, ID],
    );
    let program = PathBuf::from("tests/acceptance/clickbench/q19.vk");
    assert_eq!(
        run_query(&dir, &program, CLICKBENCH_SIGNED).unwrap(),
        format!("{ID}\n{ID}\n{ID}\n")
    );
}

/// The real q41 on 10012 one-row groups that pass the filter, plus rows each filter term
/// excludes. The excluded rows reuse width 10000, so a leak shows as a count of 2.
#[test]
fn q41_pages_past_ten_thousand_groups() {
    const HASH: i64 = 2868770270353813622;
    let dir = scratch("q41");
    // One row per column value: (CounterID, EventDate, IsRefresh, DontCountHits, URLHash, width).
    let mut rows: Vec<(i64, i64, i64, i64, i64, i64)> = (0..10012)
        .map(|i| (62, 15887 + i % 31, 0, 0, HASH, i))
        .collect();
    rows.extend([
        (63, 15900, 0, 0, HASH, 10000),
        (62, 15886, 0, 0, HASH, 10000),
        (62, 15918, 0, 0, HASH, 10000),
        (62, 15900, 1, 0, HASH, 10000),
        (62, 15900, 0, 1, HASH, 10000),
        (62, 15900, 0, 0, HASH + 1, 10000),
    ]);
    let col = |f: fn(&(i64, i64, i64, i64, i64, i64)) -> i64| -> Vec<u64> {
        rows.iter().map(|r| f(r).cast_unsigned()).collect()
    };
    write_u64(&dir, "CounterID.u64", &col(|r| r.0));
    write_u64(&dir, "EventDate.u64", &col(|r| r.1));
    write_u64(&dir, "IsRefresh.u64", &col(|r| r.2));
    write_u64(&dir, "DontCountHits.u64", &col(|r| r.3));
    let hashes: Vec<i64> = rows.iter().map(|r| r.4).collect();
    write_signed(&dir, "URLHash.u64", &hashes);
    write_u64(&dir, "WindowClientWidth.u64", &col(|r| r.5));
    write_u64(&dir, "WindowClientHeight.u64", &vec![7; rows.len()]);
    let program = PathBuf::from("tests/acceptance/clickbench/q41.vk");
    let expected: String = (10000..10010).map(|w| format!("{w}\t7\t1\n")).collect();
    assert_eq!(
        run_query(&dir, &program, CLICKBENCH_SIGNED).unwrap(),
        expected
    );
}

/// Runs the real ClickBench program `name` on the given columns, `(file, values, signed)`.
fn run_clickbench(name: &str, columns: &[(&str, &[i64], bool)]) -> String {
    let dir = scratch(name);
    for (file, values, signed) in columns {
        if *signed {
            write_signed(&dir, file, values);
        } else {
            let words: Vec<u64> = values.iter().map(|x| x.cast_unsigned()).collect();
            write_u64(&dir, file, &words);
        }
    }
    let program = PathBuf::from(format!("tests/acceptance/clickbench/{name}.vk"));
    run_query(&dir, &program, CLICKBENCH_SIGNED).unwrap()
}

/// The real q04 on repeated ids at both ends of the signed range, and on an empty column.
#[test]
fn q04_counts_distinct_ids() {
    let ids = [i64::MAX, 3, i64::MIN, 3, -1, i64::MAX, i64::MIN, 0, 3];
    assert_eq!(run_clickbench("q04", &[("UserID.u64", &ids, true)]), "5\n");
    assert_eq!(run_clickbench("q04", &[("UserID.u64", &[], true)]), "0\n");
}

/// The real q15 with ties among counts, signed ids, and more groups than the limit. Ties go
/// to the smaller signed id.
#[test]
fn q15_breaks_count_ties_by_signed_id() {
    let ids = [
        5,
        i64::MAX,
        0,
        -7,
        3,
        i64::MAX,
        8,
        i64::MIN,
        5,
        1,
        -7,
        2,
        3,
        i64::MAX,
        4,
        6,
    ];
    let got = run_clickbench("q15", &[("UserID.u64", &ids, true)]);
    // Eleven groups, so the largest single id (8) is cut.
    let expected = format!(
        "{}\t3\n-7\t2\n3\t2\n5\t2\n{}\t1\n0\t1\n1\t1\n2\t1\n4\t1\n6\t1\n",
        i64::MAX,
        i64::MIN
    );
    assert_eq!(got, expected);
    assert_eq!(run_clickbench("q15", &[("UserID.u64", &[], true)]), "");
}

/// The real q35 on addresses at the i32 edge, where `ClientIP - 3` leaves the i32 range, and
/// on an empty column. DuckDB computes `ClientIP - 1` on an INTEGER column and raises an
/// overflow error at `i32::MIN - 1`, so the answer for those addresses is a Verkehrt-only
/// choice (i64 arithmetic) and not a match with DuckDB.
#[test]
fn q35_computes_offsets_without_overflow() {
    let (lo, hi) = (i64::from(i32::MIN), i64::from(i32::MAX));
    let ips = [hi, 5, lo, 0, hi, lo];
    let got = run_clickbench("q35", &[("ClientIP.u64", &ips, true)]);
    let expected = format!(
        "{lo}\t{}\t{}\t{}\t2\n{hi}\t{}\t{}\t{}\t2\n0\t-1\t-2\t-3\t1\n5\t4\t3\t2\t1\n",
        lo - 1,
        lo - 2,
        lo - 3,
        hi - 1,
        hi - 2,
        hi - 3
    );
    assert_eq!(got, expected);
    assert_eq!(run_clickbench("q35", &[("ClientIP.u64", &[], true)]), "");
}

/// The real q08 on users that straddle region boundaries. Region 0 starts with user 0, which
/// equals the `prev` fill of row 0. Region 2 starts with user 20, which region 1 ends with,
/// and region 3 repeats user 10 from region 1. A distinct count that ignored region
/// boundaries would lose those users. Every region but the last has 2 distinct users, so the
/// ties go to the smaller region.
#[test]
fn q08_counts_distinct_users_per_region() {
    let (min, max) = (i64::MIN, i64::MAX);
    // (RegionID, UserID)
    let rows = [
        (3, 10),
        (1, 10),
        (0, 0),
        (2, 30),
        (3, min),
        (1, 10),
        (0, 5),
        (2, 20),
        (3, min),
        (1, 20),
        (2, 30),
        (0, 0),
        (4, max),
    ];
    let region: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let user: Vec<i64> = rows.iter().map(|r| r.1).collect();
    let got = run_clickbench(
        "q08",
        &[
            ("RegionID.u64", &region, false),
            ("UserID.u64", &user, true),
        ],
    );
    assert_eq!(got, "0\t2\n1\t2\n2\t2\n3\t2\n4\t1\n");
    let empty = run_clickbench(
        "q08",
        &[("RegionID.u64", &[], false), ("UserID.u64", &[], true)],
    );
    assert_eq!(empty, "");
}

/// The real q09 with distinct counts at the same region boundaries as q08 (region 0 starts
/// with user 0, region 2 starts with the user region 1 ends with). The counts are unequal and
/// their order (regions 1, 3, 2, 0) differs from both the region order and the distinct-user
/// order (regions 2, 1, 3, 0), so the sort is shown to be by count.
#[test]
fn q09_aggregates_per_region() {
    let min = i64::MIN;
    // (RegionID, AdvEngineID, ResolutionWidth, UserID)
    let rows: [(i64, i64, i64, i64); 14] = [
        (3, 0, 10, min),
        (1, 2, 100, 10),
        (0, 1, 1000, 0),
        (2, 1, 50, 20),
        (3, 5, 40, min),
        (1, 3, 200, 10),
        (2, 0, 70, 30),
        (0, 2, 1001, 0),
        (3, 0, 20, min),
        (1, 0, 300, 10),
        (2, 4, 90, 40),
        (1, 1, 400, 10),
        (3, 1, 31, 10),
        (1, 0, 500, 20),
    ];
    let col = |f: fn(&(i64, i64, i64, i64)) -> i64| -> Vec<i64> { rows.iter().map(f).collect() };
    let columns = [
        ("RegionID.u64", col(|r| r.0), false),
        ("AdvEngineID.u64", col(|r| r.1), false),
        ("ResolutionWidth.u64", col(|r| r.2), false),
        ("UserID.u64", col(|r| r.3), true),
    ];
    let args: Vec<(&str, &[i64], bool)> = columns
        .iter()
        .map(|(f, v, s)| (*f, v.as_slice(), *s))
        .collect();
    assert_eq!(
        run_clickbench("q09", &args),
        "1\t6\t5\t300.0000\t2\n3\t6\t4\t25.2500\t2\n2\t5\t3\t70.0000\t3\n0\t3\t2\t1000.5000\t1\n"
    );
    let empty: Vec<(&str, &[i64], bool)> =
        columns.iter().map(|(f, _, s)| (*f, &[][..], *s)).collect();
    assert_eq!(run_clickbench("q09", &empty), "");
}

/// The real q32 with ties among counts and signed keys. Ties go to the smaller (WatchID,
/// ClientIP) pair in signed order, including two groups of the same WatchID whose ClientIPs
/// differ in sign. An empty input gives an empty answer.
#[test]
fn q32_breaks_count_ties_by_signed_key() {
    // (WatchID, ClientIP, IsRefresh, ResolutionWidth)
    let rows: [(i64, i64, i64, i64); 10] = [
        (5, -1, 1, 100),
        (7, 4, 0, 50),
        (5, 2, 0, 300),
        (-3, 2, 0, 10),
        (7, -9, 0, 70),
        (5, -1, 0, 201),
        (-3, -1, 1, 20),
        (7, 4, 1, 60),
        (-3, 2, 1, 11),
        (7, -9, 0, 80),
    ];
    let col = |f: fn(&(i64, i64, i64, i64)) -> i64| -> Vec<i64> { rows.iter().map(f).collect() };
    let (w, ip, r, wd) = (col(|r| r.0), col(|r| r.1), col(|r| r.2), col(|r| r.3));
    let got = run_clickbench(
        "q32",
        &[
            ("WatchID.u64", &w, true),
            ("ClientIP.u64", &ip, true),
            ("IsRefresh.u64", &r, false),
            ("ResolutionWidth.u64", &wd, false),
        ],
    );
    assert_eq!(
        got,
        "-3\t2\t2\t1\t10.5000\n5\t-1\t2\t1\t150.5000\n7\t-9\t2\t0\t75.0000\n7\t4\t2\t1\t55.0000\n\
         -3\t-1\t1\t1\t20.0000\n5\t2\t1\t0\t300.0000\n"
    );
    let empty = run_clickbench(
        "q32",
        &[
            ("WatchID.u64", &[], true),
            ("ClientIP.u64", &[], true),
            ("IsRefresh.u64", &[], false),
            ("ResolutionWidth.u64", &[], false),
        ],
    );
    assert_eq!(empty, "");
}
