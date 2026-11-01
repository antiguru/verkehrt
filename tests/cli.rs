//! End-to-end tests of the `verkehrt` binary.

use assert_cmd::Command;
use verkehrt::kernels::{Feature, FeatureSet, Isa};

fn verkehrt() -> Command {
    Command::cargo_bin("verkehrt").unwrap()
}

/// The captured stderr of a finished command, lossily decoded.
fn stderr_of(a: &assert_cmd::assert::Assert) -> String {
    String::from_utf8_lossy(&a.get_output().stderr).into_owned()
}

#[test]
fn run_prints_result() {
    verkehrt()
        .args(["run", "tests/programs/dot.vk"])
        .assert()
        .success()
        .stdout("32.0\n");
}

#[test]
fn plan_goes_to_stderr() {
    let a = verkehrt()
        .args(["run", "--plan", "tests/programs/dot.vk"])
        .assert()
        .success()
        .stdout("32.0\n");
    assert!(stderr_of(&a).contains("dot_f64"));
}

#[test]
fn program_error_exits_1() {
    let dir = std::env::temp_dir().join(format!("verkehrt-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("bad.vk");
    std::fs::write(&file, "let x = [1.0] * 2;\n").unwrap();
    let a = verkehrt().arg("run").arg(&file).assert().code(1);
    assert!(stderr_of(&a).contains("error[type] at 1:"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unknown_feature_is_usage_error() {
    let a = verkehrt()
        .args(["run", "--disable", "avx513", "x.vk"])
        .assert()
        .failure();
    let err = stderr_of(&a);
    assert!(err.contains("unknown feature 'avx513'"), "{err}");
    assert!(err.contains("avx512vbmi2"), "{err}");
}

#[test]
fn disabling_a_level_feature_rejects_the_forced_level() {
    if !Isa::X86_64V4.is_available() {
        eprintln!("skipped: x86-64-v4 is not available on this CPU");
        return;
    }
    let a = verkehrt()
        .args([
            "run",
            "--isa",
            "x86-64-v4",
            "--disable",
            "avx512dq",
            "tests/programs/dot.vk",
        ])
        .assert()
        .failure();
    let err = stderr_of(&a);
    assert!(
        err.contains("instruction set 'x86-64-v4' is not available"),
        "{err}"
    );
}

#[test]
fn undetected_disable_prints_a_note() {
    let detected = FeatureSet::detect();
    let Some(feature) = Feature::ALL.into_iter().find(|f| !detected.contains(*f)) else {
        eprintln!("skipped: this CPU reports every feature");
        return;
    };
    let name = feature.name();
    let a = verkehrt()
        .args(["run", "--disable", name, "tests/programs/dot.vk"])
        .assert()
        .success()
        .stdout("32.0\n");
    let err = stderr_of(&a);
    assert!(
        err.contains(&format!(
            "note: feature '{name}' is not detected on this CPU, disabling it has no effect"
        )),
        "{err}"
    );
}

#[test]
fn unknown_isa_is_usage_error() {
    verkehrt()
        .args(["run", "--isa", "bogus", "x.vk"])
        .assert()
        .code(2);
}

#[test]
fn old_isa_names_are_rejected() {
    for name in ["avx512", "avx2"] {
        let a = verkehrt()
            .args(["run", "--isa", name, "x.vk"])
            .assert()
            .failure();
        let expected = format!("unknown instruction set '{name}'");
        assert!(stderr_of(&a).contains(&expected), "{}", stderr_of(&a));
    }
}

#[test]
fn unknown_backend_is_usage_error() {
    verkehrt()
        .args(["run", "--backend", "bogus", "x.vk"])
        .assert()
        .code(2);
}

#[test]
fn repl_reads_stdin() {
    verkehrt()
        .arg("repl")
        .write_stdin("let x = [1.0, 2.0];\nsum(x);\n")
        .assert()
        .success()
        .stdout("3.0\n");
}

#[test]
fn repl_continues_after_error() {
    let a = verkehrt()
        .arg("repl")
        .write_stdin("let x = [1.0] * 2;\nsum([1.0, 2.0]);\n")
        .assert()
        .success()
        .stdout("3.0\n");
    assert!(stderr_of(&a).contains("error[type]"));
}

/// On aarch64 the auto backend is the baseline build, so `--backend auto` runs scalar entries.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_auto_runs_the_scalar_entry() {
    let a = verkehrt()
        .args([
            "run",
            "--isa",
            "neon",
            "--backend",
            "auto",
            "--trace",
            "tests/programs/dot.vk",
        ])
        .assert()
        .success()
        .stdout("32.0\n");
    let err = stderr_of(&a);
    let line = err.lines().find(|l| l.contains("dot_f64")).expect(&err);
    assert!(line.contains("isa=scalar"), "{line}");
}
