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
