//! The acceptance query programs on the corgi data, per interpreter variant.
//!
//! Groups are `acceptance/<suite>/<query>/<variant>`. The suites read `VERKEHRT_CLICKBENCH`
//! and `VERKEHRT_TPCH`, the directories the acceptance tests read, and skip when the variable
//! is unset. Each query is prepared and its columns loaded once per variant, so timing covers
//! `Prepared::execute` only. The acceptance tests check the answers.

use std::fs;
use std::io::sink;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use verkehrt::columns::{CLICKBENCH_SIGNED, load_inputs};
use verkehrt::kernels::{Backend, Isa};
use verkehrt::value::Value;
use verkehrt::{Interpreter, Options};

/// The scalar variant, and the auto and intrinsics variants of each x86-64 ISA this CPU has.
/// On aarch64 the NEON intrinsics variant follows the scalar one.
fn interpreters() -> Vec<(String, Interpreter)> {
    let new = |isa, backend| {
        Interpreter::new(Options {
            isa: Some(isa),
            backend,
            ..Options::default()
        })
        .unwrap()
    };
    let mut v = vec![("scalar".to_string(), new(Isa::Scalar, Backend::Auto))];
    for (isa, name) in [(Isa::X86_64V3, "x86-64-v3"), (Isa::X86_64V4, "x86-64-v4")] {
        if !isa.is_available() {
            continue;
        }
        for (backend, suffix) in [(Backend::Auto, "auto"), (Backend::Intrinsics, "intrinsics")] {
            v.push((format!("{name}-{suffix}"), new(isa, backend)));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if Isa::Neon.is_available() {
        v.push((
            "neon-intrinsics".to_string(),
            new(Isa::Neon, Backend::Intrinsics),
        ));
    }
    v
}

/// The `.vk` programs of a suite, sorted by name.
fn programs(suite: &str) -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/acceptance")
        .join(suite);
    let mut v: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("program directory {}: {e}", dir.display()))
        .map(|d| d.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vk"))
        .collect();
    v.sort();
    v
}

/// `signed` lists the suite's columns stored with the sign bit flipped.
fn suite(c: &mut Criterion, var: &str, suite: &str, signed: &[&str]) {
    let Some(dir) = std::env::var_os(var).map(PathBuf::from) else {
        eprintln!("acceptance/{suite}: skipped, {var} is not set");
        return;
    };
    let interps = interpreters();
    for path in programs(suite) {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let program = fs::read_to_string(&path).unwrap();
        let mut g = c.benchmark_group(format!("acceptance/{suite}/{name}"));
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(500));
        g.measurement_time(Duration::from_secs(3));
        for (label, interp) in &interps {
            let prepared = interp.prepare(&program).unwrap();
            let inputs = load_inputs(&dir, &prepared, signed).unwrap();
            let named: Vec<(&str, Value)> = inputs
                .iter()
                .map(|(n, v)| (n.as_str(), v.clone()))
                .collect();
            // A failing query fails before timing starts.
            prepared.execute(interp, &named, &mut sink()).unwrap();
            g.bench_function(label, |b| {
                b.iter(|| prepared.execute(interp, &named, &mut sink()).unwrap())
            });
        }
        g.finish();
    }
}

fn acceptance(c: &mut Criterion) {
    suite(c, "VERKEHRT_CLICKBENCH", "clickbench", CLICKBENCH_SIGNED);
    suite(c, "VERKEHRT_TPCH", "tpch", &[]);
}

criterion_group!(benches, acceptance);
criterion_main!(benches);
