//! The decoder on several threads (docs/threading.md) gives what it gives
//! on one: the same frames for the committed vectors at every thread
//! count, run after run, and for damaged streams the same frames and the
//! same errors, without ever hanging.

mod common;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

/// What decoding one temporal unit gave: the MD5 of each frame shown, or
/// the error, or that it panicked.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Unit {
    Frames(Vec<String>),
    Error(String),
    Panic,
}

fn vectors() -> Vec<PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let v = common::vectors_in(&dir);
    assert!(v.len() >= 12, "tests/data is incomplete");
    v
}

/// Decodes a stream on `threads` threads, every unit's outcome in order. A
/// decoder goes on after an error, as a player does.
fn decode(packets: &[Vec<u8>], threads: usize) -> Vec<Unit> {
    let mut dec = av1::Decoder::new();
    dec.set_threads(threads);
    dec.set_max_pixels(1 << 22);
    packets
        .iter()
        .map(|p| {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.decode_all(p)));
            match r {
                Ok(Ok(frames)) => Unit::Frames(frames.iter().map(common::frame_md5).collect()),
                Ok(Err(e)) => Unit::Error(e.to_string()),
                Err(_) => Unit::Panic,
            }
        })
        .collect()
}

/// [`decode`] on a thread of its own, given up on after `limit`: a decoder
/// whose threads wait for each other forever fails the test instead of
/// stopping it.
fn decode_within(packets: &[Vec<u8>], threads: usize, limit: Duration, what: &str) -> Vec<Unit> {
    let (tx, rx) = mpsc::channel();
    let packets = packets.to_vec();
    std::thread::spawn(move || {
        let _ = tx.send(decode(&packets, threads));
    });
    match rx.recv_timeout(limit) {
        Ok(units) => units,
        Err(_) => panic!("{what}: no result on {threads} threads after {limit:?}: a hang"),
    }
}

const LIMIT: Duration = Duration::from_secs(120);

/// `AV1_THREAD_SOAK=n` repeats the repeated tests `n` times as often.
fn soak() -> usize {
    std::env::var("AV1_THREAD_SOAK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

#[test]
fn every_thread_count_decodes_the_same_frames() {
    for path in vectors() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let packets = common::packets(&path);
        let expected = common::expected_md5s(&path);
        let one = decode(&packets, 1);
        let shown: Vec<&String> = one
            .iter()
            .flat_map(|u| match u {
                Unit::Frames(f) => f.iter(),
                other => panic!("{name}: {other:?} on one thread"),
            })
            .collect();
        assert!(
            shown.iter().copied().eq(expected.iter()),
            "{name}: one thread is not the reference"
        );
        for threads in [2, 3, 4, 8, 16] {
            let got = decode_within(&packets, threads, LIMIT, &name);
            assert!(got == one, "{name}: {threads} threads differ from one");
        }
    }
}

#[test]
fn threaded_runs_repeat() {
    // The interleaving of the threads differs from run to run; the frames
    // must not. The streams with the most rows and the most kinds of
    // prediction, many times over.
    let all = vectors();
    for name in [
        "av1-1-b8-05-mv.ivf",
        "av1-1-b8-06-mfmv.ivf",
        "av1-1-b8-03-sizeup.mkv",
        "av1-1-b8-01-size-226x226.ivf",
    ] {
        let path = all
            .iter()
            .find(|p| p.ends_with(name))
            .unwrap_or_else(|| panic!("{name} is not in tests/data"));
        let packets = common::packets(path);
        let one = decode(&packets, 1);
        for run in 0..25 * soak() {
            for threads in [2, 4, 8] {
                let got = decode_within(&packets, threads, LIMIT, name);
                assert!(got == one, "{name}: run {run} on {threads} threads differs");
            }
        }
    }
}

/// A small generator for the damage (xorshift): the same damage every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as usize
    }
}

/// The stream with one kind of damage, chosen by `rng`: bits flipped in a
/// unit, a unit cut short, or both in different units.
fn damage(packets: &[Vec<u8>], rng: &mut Rng) -> Vec<Vec<u8>> {
    let mut s = packets.to_vec();
    let n = s.len();
    let kind = rng.next() % 3;
    if kind != 1 {
        // Past the first unit more often than in it: a stream whose key
        // frame is intact goes furthest into the decoder.
        let k = if rng.next().is_multiple_of(4) {
            0
        } else {
            rng.next() % n
        };
        let p = &mut s[k];
        for _ in 0..1 + rng.next() % 4 {
            if !p.is_empty() {
                let i = rng.next() % p.len();
                p[i] ^= 1 << (rng.next() % 8);
            }
        }
    }
    if kind != 0 {
        let k = rng.next() % n;
        let len = s[k].len();
        s[k].truncate(rng.next() % (len + 1));
    }
    s
}

#[test]
fn damaged_streams_end_alike_on_every_thread_count() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut errors = 0;
    let mut cases = 0;
    for path in vectors() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let packets = common::packets(&path);
        // Streams of many frames take a prefix: the damage is what matters.
        let packets = &packets[..packets.len().min(8)];
        for case in 0..24 * soak() {
            let s = damage(packets, &mut rng);
            let what = format!("{name}, damage {case}");
            let one = decode_within(&s, 1, LIMIT, &what);
            assert!(!one.contains(&Unit::Panic), "{what}: a panic on one thread");
            errors += one.iter().filter(|u| matches!(u, Unit::Error(_))).count();
            cases += 1;
            for threads in [2, 4, 8] {
                let got = decode_within(&s, threads, LIMIT, &what);
                assert!(
                    got == one,
                    "{what}: {threads} threads differ from one:\n{got:?}\n{one:?}"
                );
            }
        }
    }
    // The damage did reach the decoder.
    assert!(
        errors * 4 >= cases,
        "{errors} errors in {cases} damaged streams"
    );
}
