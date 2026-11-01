//! Helpers for the datadriven unit test harnesses.

use datadriven::TestCase;
use std::cell::Cell;
use std::path::Path;

/// Runs `case` on every test case in the files of `dir`. Panics if `dir` holds no non-hidden
/// file, or if no case ran while `RUN` is unset. `walk` itself silently runs nothing for a
/// missing or empty directory, which would let a harness pass without testing anything. A
/// `RUN` value is exempt from the count because one value applies to every harness, and it
/// legitimately matches nothing in most of them.
pub fn run_dir(dir: &str, mut case: impl FnMut(&mut TestCase) -> String) {
    let has_file = std::fs::read_dir(Path::new(dir))
        .unwrap_or_else(|e| panic!("cannot read test data directory {dir}: {e}"))
        .filter_map(Result::ok)
        .any(|e| !e.file_name().to_string_lossy().starts_with('.'));
    assert!(has_file, "test data directory {dir} holds no files");
    let cases = Cell::new(0usize);
    datadriven::walk(dir, |f| {
        f.run(|tc| {
            cases.set(cases.get() + 1);
            case(tc)
        })
    });
    if std::env::var_os("RUN").is_none() {
        assert!(cases.get() > 0, "no test case ran in {dir}");
    }
}

/// Panics if the directive carries an argument that is not in `allowed`.
pub fn reject_unknown_args(tc: &TestCase, allowed: &[&str]) {
    for arg in tc.args.keys() {
        assert!(
            allowed.contains(&arg.as_str()),
            "directive {} takes no argument {arg}",
            tc.directive
        );
    }
}
