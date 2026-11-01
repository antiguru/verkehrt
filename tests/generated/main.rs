//! Generated programs checked by independent oracles, see `oracles`.
//!
//! Planning dominates the time of a case, so independent searches run on worker threads.
//! Worker `i` of `w` runs its share of the cases from seed `VERKEHRT_GEN_SEED + i`, so a
//! failure reproduces alone with `VERKEHRT_GEN_THREADS=1`, the worker's seed, and the
//! worker's case count, which the failure message names.
//!
//! The test runs a fixed seed, a fixed worker count, and a small case count. A longer local
//! search sets `VERKEHRT_GEN_CASES`, and `VERKEHRT_GEN_SEED` to repeat a seed, which is
//! otherwise random when the case count is set. `VERKEHRT_GEN_THREADS` sets the worker count,
//! which defaults to the available parallelism in a local search. `VERKEHRT_GEN_DEPTH`,
//! `VERKEHRT_GEN_STMTS`, and `VERKEHRT_GEN_FNS` raise the size limits, and
//! `VERKEHRT_GEN_SHRINK_SECS` bounds the shrinking of a failure. The feature histogram
//! is printed at the end, visible with `--nocapture`, and `VERKEHRT_GEN_SHOW` prints every
//! program with its outcome.

mod eval;
mod r#gen;
mod oracles;
mod rewrites;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use proptest::collection::vec;
use proptest::prelude::{Strategy, any};
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestError, TestRng, TestRunner};
use verkehrt::error::ErrorKind;

use r#gen::{Catalog, Params, Plan as GenPlan};
use oracles::Engine;

/// Cases of the CI run, measured to stay well under 30 s in a debug build.
const CI_CASES: u64 = 64;
const CI_SEED: u64 = 0x5eed_0001;
/// Workers of the CI run, fixed so that every machine runs the same programs.
const CI_THREADS: u64 = 4;
/// How long a worker shrinks a failing case.
const SHRINK_SECS: u32 = 120;
/// The longest choice sequence of a function or statement. Missing choices read as 0, the
/// simplest option, which makes the rest of the expression trivial, so sequences are not
/// very short.
const CHOICES: usize = 256;

fn env_num(name: &str) -> Option<u64> {
    let v = std::env::var(name).ok()?;
    Some(
        v.parse()
            .unwrap_or_else(|_| panic!("{name} must be a number, got {v:?}")),
    )
}

fn env_size(name: &str, default: usize) -> usize {
    env_num(name).map_or(default, |v| usize::try_from(v).expect("size fits usize"))
}

#[derive(Default)]
struct Stats {
    cases: usize,
    features: BTreeMap<&'static str, usize>,
    outcomes: BTreeMap<String, usize>,
    fusion_changed_plan: usize,
    /// Per integer sum rewrite, the statements in which it fired.
    rewrites_fired: BTreeMap<rewrites::Rewrite, usize>,
    source_bytes: usize,
    slowest: (Duration, String),
}

impl Stats {
    fn merge(&mut self, other: Stats) {
        self.cases += other.cases;
        for (f, n) in other.features {
            *self.features.entry(f).or_default() += n;
        }
        for (o, n) in other.outcomes {
            *self.outcomes.entry(o).or_default() += n;
        }
        self.fusion_changed_plan += other.fusion_changed_plan;
        for (r, n) in other.rewrites_fired {
            *self.rewrites_fired.entry(r).or_default() += n;
        }
        self.source_bytes += other.source_bytes;
        if other.slowest.0 > self.slowest.0 {
            self.slowest = other.slowest;
        }
    }
}

/// One worker's search: `cases` programs from `seed`. Returns its statistics and the failure
/// message, if a case failed.
fn search(seed: u64, cases: u32, params: Params, shrink_secs: u32) -> (Stats, Option<String>) {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    let config = Config {
        cases,
        failure_persistence: None,
        // Most shrink steps change the program completely and pass, so shrinking needs many,
        // and each plans a program twice, so the time bounds it.
        max_shrink_iters: u32::MAX - 1,
        max_shrink_time: shrink_secs * 1000,
        ..Config::default()
    };
    let mut runner = TestRunner::new_with_rng(
        config,
        TestRng::from_seed(RngAlgorithm::ChaCha, &seed_bytes),
    );
    let catalog = Catalog::new();
    let engine = Engine::new();
    let show = std::env::var_os("VERKEHRT_GEN_SHOW").is_some();
    let stats = RefCell::new(Stats::default());
    let choices = || vec(any::<u32>(), CHOICES / 4..=CHOICES);
    // Proptest shrinks the fields of a tuple in order, so whole statements go first.
    let strategy = (
        vec(choices(), 1..=params.max_stmts),
        vec(choices(), 0..=params.max_fns),
        vec(any::<u32>(), 16..=64),
    )
        .prop_map(|(stmts, fns, inputs)| GenPlan { inputs, fns, stmts });
    let result = runner.run(&strategy, |plan| {
        let case = r#gen::generate(&catalog, &plan, params);
        if show {
            // Printed before the check, so a case that hangs or exhausts memory is visible.
            eprintln!("[worker {seed}]\n{}", case.source);
        }
        let start = Instant::now();
        let verdict = engine.check(&case).map_err(TestCaseError::fail)?;
        let took = start.elapsed();
        if show {
            eprintln!("[worker {seed}] => {:?}\n", verdict.error);
        }
        if verdict
            .error
            .as_ref()
            .is_some_and(|e| e.0 == ErrorKind::Usage)
        {
            return Err(TestCaseError::fail(format!(
                "generator bug, the inputs do not fit the program\n{}",
                oracles::describe(&case)
            )));
        }
        let mut stats = stats.borrow_mut();
        stats.cases += 1;
        if took > stats.slowest.0 {
            stats.slowest = (took, case.source.clone());
        }
        stats.source_bytes += case.source.len();
        for f in &case.features {
            *stats.features.entry(f).or_default() += 1;
        }
        // Errors group by message with the numbers left out.
        let outcome = match &verdict.error {
            None => "ok".to_string(),
            Some((kind, message)) => {
                let shape: String = message.chars().filter(|c| !c.is_ascii_digit()).collect();
                format!("{kind:?}: {shape}")
            }
        };
        *stats.outcomes.entry(outcome).or_default() += 1;
        if let Some(reason) = verdict.reference_skipped {
            *stats
                .outcomes
                .entry(format!("oracle 3 skipped: {reason}"))
                .or_default() += 1;
        }
        stats.fusion_changed_plan += usize::from(verdict.fusion_changed_plan);
        for (r, n) in verdict.rewrites_fired {
            *stats.rewrites_fired.entry(r).or_default() += n;
        }
        Ok(())
    });
    let failure = match result {
        Ok(()) => None,
        Err(TestError::Fail(reason, plan)) => Some(format!(
            "VERKEHRT_GEN_THREADS=1 VERKEHRT_GEN_SEED={seed} VERKEHRT_GEN_CASES={cases}, \
             minimal case of {} functions and {} statements: {reason}",
            plan.fns.len(),
            plan.stmts.len()
        )),
        Err(TestError::Abort(reason)) => Some(format!("seed {seed}: aborted: {reason}")),
    };
    (stats.into_inner(), failure)
}

#[test]
fn generated_programs() {
    let local = env_num("VERKEHRT_GEN_CASES");
    let cases = local.unwrap_or(CI_CASES);
    let seed = env_num("VERKEHRT_GEN_SEED").unwrap_or_else(|| match local {
        // A long run without a seed searches somewhere new.
        Some(_) => u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time is after the epoch")
                .as_micros()
                % u128::from(u32::MAX),
        )
        .expect("below u32::MAX"),
        None => CI_SEED,
    });
    let threads = env_num("VERKEHRT_GEN_THREADS").unwrap_or_else(|| match local {
        Some(_) => std::thread::available_parallelism().map_or(1, |n| {
            u64::try_from(n.get()).expect("the worker count fits u64")
        }),
        None => CI_THREADS,
    });
    let shrink_secs = env_num("VERKEHRT_GEN_SHRINK_SECS")
        .map_or(SHRINK_SECS, |v| u32::try_from(v).expect("seconds fit u32"));
    let params = Params {
        max_depth: env_size("VERKEHRT_GEN_DEPTH", 4),
        max_stmts: env_size("VERKEHRT_GEN_STMTS", 6),
        max_fns: env_size("VERKEHRT_GEN_FNS", 3),
        max_len: 300,
    };
    eprintln!(
        "generated programs: {cases} cases on {threads} workers, VERKEHRT_GEN_SEED={seed}, \
         {params:?}"
    );

    let begin = Instant::now();
    let results: Vec<(Stats, Option<String>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                // The first workers take the remainder.
                let share = cases / threads + u64::from(i < cases % threads);
                let share = u32::try_from(share).expect("cases fit u32");
                let seed = seed.wrapping_add(i);
                std::thread::Builder::new()
                    .name(format!("generated-{i}"))
                    .spawn_scoped(s, move || search(seed, share, params, shrink_secs))
                    .expect("spawn worker")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    let mut stats = Stats::default();
    let mut failures = Vec::new();
    for (s, f) in results {
        stats.merge(s);
        failures.extend(f);
    }
    let engine = Engine::new();
    eprintln!(
        "{} programs on {} variants in {:.1?}, {} source bytes on average, fusion changed the \
         plan of {}",
        stats.cases,
        engine.variant_count(),
        begin.elapsed(),
        stats.source_bytes / stats.cases.max(1),
        stats.fusion_changed_plan,
    );
    eprintln!(
        "slowest case, {:.0?} for planning twice and running:\n{}",
        stats.slowest.0, stats.slowest.1
    );
    for (outcome, n) in &stats.outcomes {
        eprintln!("  outcome {n:>7} {outcome}");
    }
    for (feature, n) in &stats.features {
        eprintln!("  feature {feature:<32} {n:>7}");
    }
    for r in engine.rewrites() {
        let n = stats.rewrites_fired.get(r).copied().unwrap_or(0);
        eprintln!("  rewrite {:<32} {n:>7}", r.to_string());
    }
    if !failures.is_empty() {
        panic!("{}", failures.join("\n\n"));
    }
    // Oracle 2 compares nothing if fusion never changes a plan.
    assert!(
        stats.cases < 20 || stats.fusion_changed_plan > 0,
        "fusion changed no plan in {} programs",
        stats.cases
    );
    // Oracle 2 checks an integer sum rewrite only in a plan where it fired.
    let unfired: Vec<String> = engine
        .rewrites()
        .iter()
        .filter(|r| !stats.rewrites_fired.contains_key(r))
        .map(ToString::to_string)
        .collect();
    assert!(
        stats.cases < 20 || unfired.is_empty(),
        "the rewrites {unfired:?} fired in no plan of {} programs",
        stats.cases
    );
}
