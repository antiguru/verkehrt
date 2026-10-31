//! Acceptance of the SQL query programs against DuckDB's answers on the corgi data. The
//! suites read `VERKEHRT_CLICKBENCH` and `VERKEHRT_TPCH`, the directories `prepare.py` wrote,
//! and skip when the variable is unset.

mod harness;

use std::fs;
use std::path::{Path, PathBuf};

use verkehrt::value::{Array, Value};

use harness::{format_rows, run_query};

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
    assert_eq!(run_query(&dir, &program).unwrap(), expected);
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
    assert_eq!(run_query(&dir, &program).unwrap(), "");
    fs::write(&program, q07_program(false)).unwrap();
    let err = run_query(&dir, &program).unwrap_err();
    assert!(err.contains("empty array"), "{err}");
}

fn run_suite(var: &str, queries: &str) {
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
        match run_query(&dir, &path) {
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
    run_suite("VERKEHRT_CLICKBENCH", "clickbench");
}

#[test]
fn tpch_matches_duckdb() {
    run_suite("VERKEHRT_TPCH", "tpch");
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
