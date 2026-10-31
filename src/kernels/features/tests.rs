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

/// The macros that may forward a metavariable to `#[target_feature]`. Their invocations are
/// checked in `feature_strings`.
const FORWARDING_MACROS: [&str; 2] = ["auto_backend", "selection_loops"];

/// The index of the delimiter that closes the one at `open` in `src`, found by depth
/// matching. String literals are skipped, because the feature lists hold no delimiters but
/// an argument may.
fn closing(src: &str, open: usize) -> usize {
    let mut depth = 0;
    let mut in_string = false;
    for (i, c) in src[open..].char_indices() {
        match c {
            '"' => in_string = !in_string,
            '(' | '{' | '[' if !in_string => depth += 1,
            ')' | '}' | ']' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return open + i;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced delimiter at byte {open}");
}

/// The text after `at` without leading whitespace, and its byte offset in `src`.
fn skip_whitespace(src: &str, at: usize) -> usize {
    at + (src[at..].len() - src[at..].trim_start().len())
}

/// The name and byte range of the body of each `macro_rules!` definition in `src`, found by
/// delimiter matching.
fn macro_bodies(src: &str) -> Vec<(&str, Range<usize>)> {
    src.match_indices("macro_rules!")
        .map(|(at, m)| {
            let name_at = skip_whitespace(src, at + m.len());
            let name_end = name_at
                + src[name_at..]
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .expect("a macro name");
            let open = skip_whitespace(src, name_end);
            (&src[name_at..name_end], open..closing(src, open) + 1)
        })
        .collect()
}

/// The identifier at the start of `s`, after whitespace, empty if none.
fn leading_ident(s: &str) -> &str {
    let s = s.trim_start();
    &s[..s
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(s.len())]
}

/// The feature strings in `src`, each with the name of the item it applies to: the function
/// that follows a `target_feature` attribute, or the first argument of a macro invocation when
/// it is an identifier, else empty. Each `target_feature(enable = ...)` argument, and the last
/// argument of each invocation of a macro that forwards it to `#[target_feature]`. An
/// argument that is a metavariable (`$features`) is allowed only inside the definition of a
/// forwarding macro, whose invocations are checked instead. Any other argument that is not a
/// string literal fails.
fn feature_strings(src: &str) -> Vec<(String, String)> {
    let bodies = macro_bodies(src);
    // The byte offset, the item name, and the feature argument.
    let mut args: Vec<(usize, String, &str)> = Vec::new();
    for (at, m) in src.match_indices("target_feature") {
        let open = skip_whitespace(src, at + m.len());
        if !src[open..].starts_with('(') {
            // The word in a comment, a string, or an identifier such as `is_x86_feature_detected`.
            continue;
        }
        let close = closing(src, open);
        // The next function after the attribute. Attributes and doc comments may sit between.
        let fn_at = src[close..]
            .match_indices("fn ")
            .map(|(i, _)| close + i)
            .find(|&i| !src[..i].ends_with(|c: char| c.is_alphanumeric() || c == '_'))
            .unwrap_or_else(|| panic!("no function after target_feature at byte {at}"));
        let item = leading_ident(&src[fn_at + 3..]).to_string();
        let arg = src[open + 1..close]
            .trim()
            .strip_prefix("enable")
            .and_then(|r| r.trim_start().strip_prefix('='))
            .unwrap_or_else(|| panic!("unknown target_feature form at byte {at}"));
        args.push((at, item, arg.trim()));
    }
    for name in FORWARDING_MACROS {
        for (at, m) in src.match_indices(&format!("{name}!")) {
            let open = skip_whitespace(src, at + m.len());
            if !src[open..].starts_with(['(', '{', '[']) {
                continue;
            }
            // The last argument, a string literal with commas inside or a metavariable.
            let list = src[open + 1..closing(src, open)]
                .trim()
                .trim_end_matches(',')
                .trim_end();
            let last = match list.strip_suffix('"') {
                Some(head) => &list[head.rfind('"').expect("an opening quote")..],
                None => list.rsplit(',').next().unwrap().trim(),
            };
            let item = leading_ident(&src[open + 1..]).to_string();
            args.push((at, item, last));
        }
    }
    args.into_iter()
        .filter_map(|(at, item, arg)| {
            if let Some(s) = arg.strip_prefix('"') {
                Some((
                    item,
                    s.strip_suffix('"').expect("a string literal").to_string(),
                ))
            } else {
                assert!(
                    arg.starts_with('$')
                        && bodies.iter().any(|(name, body)| {
                            body.contains(&at) && FORWARDING_MACROS.contains(name)
                        }),
                    "feature argument {arg} at byte {at} is not a string literal and not in a \
                     forwarding macro"
                );
                None
            }
        })
        .collect()
}

/// The level that registers the code at `item` of the file `rel`, as the slot in the counts of
/// `target_feature_strings_match_levels`: 0 for x86-64-v3, 1 for x86-64-v4, 2 for x86-64-v4
/// plus VBMI2, 3 for NEON. `None` for code that registers no level.
fn registered_slot(rel: &str, item: &str) -> Option<usize> {
    match (rel, item) {
        ("src/kernels/x86/compress/avx2.rs", _) => Some(0),
        ("src/kernels/x86/compress/avx512.rs" | "src/kernels/x86/scan.rs", _) => Some(1),
        ("src/kernels/x86/compress/avx512_vbmi2.rs", _) => Some(2),
        ("src/kernels/x86/gather.rs" | "benches/tier1.rs", "v3") => Some(0),
        ("src/kernels/x86/gather.rs" | "benches/tier1.rs", "v4") => Some(1),
        ("src/kernels/x86.rs", "x86_64_v3") => Some(0),
        ("src/kernels/x86.rs", "x86_64_v4") => Some(1),
        (rel, _) if rel.starts_with("src/kernels/aarch64/") => Some(3),
        _ => None,
    }
}

/// Every feature string that reaches `#[target_feature]` in `src/` and `benches/` is the string
/// of the level that registers the code it compiles, so a level's `FeatureSet` and the code
/// installed for it agree. The v3 string is accepted only in the code registered as v3, the v4
/// string only in v4 code, the v4 plus VBMI2 string only in `avx512_vbmi2.rs`, and `neon` only
/// under `aarch64`. Files that hold both levels attribute each string to the function or
/// macro invocation it applies to, see `registered_slot`.
#[test]
fn target_feature_strings_match_levels() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // This file holds the patterns in its extractor.
    let own = root.join(file!());
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    rust_files(&root.join("benches"), &mut files);
    let v3 = level(Isa::X86_64V3);
    let v4 = level(Isa::X86_64V4);
    let v4_vbmi2 = v4.union(FeatureSet::of(&[Avx512vbmi2]));
    let neon = level(Isa::Neon);
    let slot_sets = [v3, v4, v4_vbmi2, neon];
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
        for (item, s) in feature_strings(&src) {
            let set =
                FeatureSet::parse(&s).unwrap_or_else(|| panic!("{rel}: unknown feature in {s:?}"));
            let slot = registered_slot(&rel, &item)
                .unwrap_or_else(|| panic!("{rel}: item {item:?} registers no level for {s:?}"));
            assert_eq!(
                set, slot_sets[slot],
                "{rel}: item {item:?} is registered at slot {slot} (v3, v4, v4 plus VBMI2, \
                 neon) but its feature string is {s:?}"
            );
            counts[slot] += 1;
        }
    }
    assert!(
        counts.iter().all(|&c| c > 0),
        "matches per level: {counts:?}"
    );
}
