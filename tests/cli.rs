//! End-to-end tests of the `verkehrt` binary.

use assert_cmd::Command;

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
fn unknown_isa_is_usage_error() {
    verkehrt()
        .args(["run", "--isa", "bogus", "x.vk"])
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
