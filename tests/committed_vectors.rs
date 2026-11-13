//! The test vectors committed in tests/data (see its README): every shown
//! frame must match its published MD5. Runs without any download.

mod common;

#[test]
fn committed_vectors() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let paths = common::vectors_in(&dir);
    assert!(paths.len() >= 12, "tests/data is incomplete");
    let results = common::run_all(&paths);
    let failed: Vec<_> = results.iter().filter(|o| !o.passed()).collect();
    for o in &failed {
        eprintln!(
            "FAIL {}: {}/{} frames; {}",
            o.name,
            o.matched,
            o.expected,
            o.failure.as_deref().unwrap_or("")
        );
    }
    assert!(failed.is_empty());
}

/// Film grain synthesis can be switched off: the grain vectors then come
/// out as decoded (not what their MD5s hash), frame for frame; a stream
/// without grain parameters is not affected.
#[test]
fn film_grain_can_be_switched_off() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let decode = |name: &str, grain: bool| -> Vec<String> {
        let mut dec = av1::Decoder::new();
        dec.set_apply_grain(grain);
        common::packets(&dir.join(name))
            .iter()
            .filter_map(|p| dec.decode(p).unwrap())
            .map(|f| common::frame_md5(&f))
            .collect()
    };
    for name in [
        "av1-1-b8-23-film_grain-50.ivf",
        "av1-1-b10-23-film_grain-50.ivf",
    ] {
        let expected = common::expected_md5s(&dir.join(name));
        let with = decode(name, true);
        let without = decode(name, false);
        assert_eq!(with, expected, "{name}: grain on is the reference");
        assert_eq!(without.len(), with.len(), "{name}");
        assert!(
            without.iter().zip(&with).all(|(a, b)| a != b),
            "{name}: every frame carries grain"
        );
    }
    let plain = "av1-1-b8-05-mv.ivf";
    assert_eq!(decode(plain, false), decode(plain, true));
}
