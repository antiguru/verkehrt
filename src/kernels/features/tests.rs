use std::ops::Range;
use std::path::{Path, PathBuf};

use super::Feature::*;
use super::*;

fn all() -> FeatureSet {
    FeatureSet::of(&Feature::ALL)
}

#[test]
fn names_round_trip() {
    for f in Feature::ALL {
        assert_eq!(Feature::from_name(f.name()), Some(f));
    }
    assert_eq!(Feature::from_name("avx512"), None);
}

#[test]
fn parse_reads_comma_separated_names() {
    assert_eq!(
        FeatureSet::parse("sse4.1,neon"),
        Some(FeatureSet::of(&[Sse41, Neon]))
    );
    assert_eq!(FeatureSet::parse("sse4.1,bogus"), None);
}

#[test]
fn v4_contains_v3() {
    assert!(level(Isa::X86_64V4).contains_all(level(Isa::X86_64V3)));
    assert_eq!(
        level(Isa::X86_64V4).minus(level(Isa::X86_64V3)),
        FeatureSet::of(&[Avx512f, Avx512bw, Avx512cd, Avx512dq, Avx512vl])
    );
}

#[test]
fn dropping_a_level_feature_drops_the_level() {
    let all = all();
    assert_eq!(best_level(all), Isa::X86_64V4);
    assert_eq!(
        best_level(all.minus(FeatureSet::of(&[Avx512dq]))),
        Isa::X86_64V3
    );
    assert_eq!(
        best_level(all.minus(FeatureSet::of(&[Avx2, Neon]))),
        Isa::Scalar
    );
    assert_eq!(best_level(FeatureSet::of(&[Neon])), Isa::Neon);
}

#[test]
fn v4_without_vbmi2_is_v4() {
    assert_eq!(
        best_level(all().minus(FeatureSet::of(&[Avx512vbmi2]))),
        Isa::X86_64V4
    );
}

#[test]
fn declared_features_gate_runnable() {
    let vbmi2 = FeatureSet::of(&[Avx512vbmi2]);
    assert!(runnable(Isa::X86_64V4, vbmi2, all()));
    assert!(!runnable(Isa::X86_64V4, vbmi2, all().minus(vbmi2)));
    assert!(runnable(Isa::Scalar, FeatureSet::EMPTY, FeatureSet::EMPTY));
    assert!(!runnable(
        Isa::X86_64V3,
        FeatureSet::EMPTY,
        FeatureSet::of(&[Neon])
    ));
}

#[test]
fn detect_is_architecture_specific() {
    let detected = FeatureSet::detect();
    #[cfg(target_arch = "x86_64")]
    assert!(!detected.contains(Neon));
    #[cfg(target_arch = "aarch64")]
    {
        assert!(detected.contains(Neon));
        assert_eq!(detected.minus(FeatureSet::of(&[Neon])), FeatureSet::EMPTY);
    }
    let _ = detected;
}

/// The x86-64-v3 feature string of the global constraints.
const V3_STRING: &str =
    "avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3";

#[test]
fn levels_match_constraint_strings() {
    assert_eq!(level(Isa::X86_64V3), FeatureSet::parse(V3_STRING).unwrap());
    let v4 = format!("{V3_STRING},avx512f,avx512bw,avx512cd,avx512dq,avx512vl");
    assert_eq!(level(Isa::X86_64V4), FeatureSet::parse(&v4).unwrap());
}

/// The `.rs` files under `dir`, recursively.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The byte ranges of the bodies of the `macro_rules!` definitions in `src`, found by brace
/// matching.
fn macro_bodies(src: &str) -> Vec<Range<usize>> {
    src.match_indices("macro_rules!")
        .map(|(at, _)| {
            let open = at + src[at..].find('{').expect("a macro body");
            let mut depth = 0;
            for (i, c) in src[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return open..open + i + 1;
                        }
                    }
                    _ => {}
                }
            }
            panic!("unbalanced macro body at byte {at}");
        })
        .collect()
}

/// The feature strings in `src`: each `target_feature(enable = ...)` argument, and the last
/// argument of each invocation of a macro that forwards it to `#[target_feature]`. An
/// argument that is a metavariable (`$features`) inside a `macro_rules!` definition is
/// checked at the invocations instead, and any other argument that is not a string literal
/// fails.
fn feature_strings(src: &str) -> Vec<String> {
    let bodies = macro_bodies(src);
    let mut args: Vec<(usize, &str)> = Vec::new();
    for (at, m) in src.match_indices("target_feature(") {
        let rest = &src[at + m.len()..];
        let end = rest.find(')').expect("a closing parenthesis");
        let arg = rest[..end]
            .trim()
            .strip_prefix("enable")
            .and_then(|r| r.trim_start().strip_prefix('='))
            .unwrap_or_else(|| panic!("unknown target_feature form at byte {at}"));
        args.push((at, arg.trim()));
    }
    for name in ["auto_backend!(", "selection_loops!("] {
        for (at, m) in src.match_indices(name) {
            let rest = &src[at + m.len()..];
            let end = rest.find(')').expect("a closing parenthesis");
            // The last argument, a string literal with commas inside or a metavariable.
            let list = rest[..end].trim().trim_end_matches(',').trim_end();
            let last = match list.strip_suffix('"') {
                Some(open) => &list[open.rfind('"').expect("an opening quote")..],
                None => list.rsplit(',').next().unwrap().trim(),
            };
            args.push((at, last));
        }
    }
    args.into_iter()
        .filter_map(|(at, arg)| {
            if let Some(s) = arg.strip_prefix('"') {
                Some(s.strip_suffix('"').expect("a string literal").to_string())
            } else {
                assert!(
                    arg.starts_with('$') && bodies.iter().any(|b| b.contains(&at)),
                    "feature argument {arg} at byte {at} is not a string literal"
                );
                None
            }
        })
        .collect()
}

/// Every feature string that reaches `#[target_feature]` in `src/` is a level string, so a
/// level's `FeatureSet` and the code compiled for it agree. The VBMI2 string is confined to
/// the module that only VBMI2-declaring entries reach.
#[test]
fn target_feature_strings_match_levels() {
    const VBMI2_FILE: &str = "src/kernels/x86/compress/avx512_vbmi2.rs";
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // This file holds the patterns in its extractor.
    let own = root.join(file!());
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    let v3 = level(Isa::X86_64V3);
    let v4 = level(Isa::X86_64V4);
    let v4_vbmi2 = v4.union(FeatureSet::of(&[Avx512vbmi2]));
    let neon = level(Isa::Neon);
    // Matches of v3, v4, v4 plus VBMI2, and NEON.
    let mut counts = [0usize; 4];
    for path in files {
        if path == own {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_str()
            .unwrap()
            .replace('\\', "/");
        let src = std::fs::read_to_string(&path).unwrap();
        for s in feature_strings(&src) {
            let set =
                FeatureSet::parse(&s).unwrap_or_else(|| panic!("{rel}: unknown feature in {s:?}"));
            let slot = if set == v3 {
                0
            } else if set == v4 {
                1
            } else if set == v4_vbmi2 && rel == VBMI2_FILE {
                2
            } else if set == neon && rel.starts_with("src/kernels/aarch64") {
                3
            } else {
                panic!("{rel}: feature string {s:?} is not a level string of this file");
            };
            counts[slot] += 1;
        }
    }
    assert!(
        counts.iter().all(|&c| c > 0),
        "matches per level: {counts:?}"
    );
}
