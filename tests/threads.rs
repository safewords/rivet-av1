//! The decoder on several threads (docs/threading.md) gives what it gives
//! on one: the same frames for the committed vectors at every thread
//! count, run after run, and for damaged streams the same frames and the
//! same errors, without ever hanging. And with several frames in flight
//! (`queue`, `poll`, `wait`) every unit's result is what decoding it in
//! place gives.

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

/// Decodes a stream with up to `flight` frames in flight on `threads`
/// threads: every unit queued, results taken as they come and, at the end,
/// waited for. Every unit's outcome, in order.
fn decode_in_flight(packets: &[Vec<u8>], threads: usize, flight: usize) -> Vec<Unit> {
    let unit = |r: av1::Result<Vec<av1::Frame>>| match r {
        Ok(frames) => Unit::Frames(frames.iter().map(common::frame_md5).collect()),
        Err(e) => Unit::Error(e.to_string()),
    };
    let r = std::panic::catch_unwind(|| {
        let mut dec = av1::Decoder::new();
        dec.set_threads(threads);
        dec.set_frames_in_flight(flight);
        dec.set_max_pixels(1 << 22);
        let mut out = Vec::new();
        for p in packets {
            dec.queue(p);
            while let Some(r) = dec.poll() {
                out.push(unit(r));
            }
        }
        while let Some(r) = dec.wait() {
            out.push(unit(r));
        }
        assert_eq!(dec.queued(), 0);
        out
    });
    r.unwrap_or_else(|_| vec![Unit::Panic])
}

/// `run` on a thread of its own, given up on after `limit`: a decoder
/// whose threads wait for each other forever fails the test instead of
/// stopping it.
fn within(
    limit: Duration,
    what: &str,
    run: impl FnOnce() -> Vec<Unit> + Send + 'static,
) -> Vec<Unit> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run());
    });
    match rx.recv_timeout(limit) {
        Ok(units) => units,
        Err(_) => panic!("{what}: no result after {limit:?}: a hang"),
    }
}

fn decode_within(packets: &[Vec<u8>], threads: usize, limit: Duration, what: &str) -> Vec<Unit> {
    let packets = packets.to_vec();
    within(limit, &format!("{what}, {threads} threads"), move || {
        decode(&packets, threads)
    })
}

fn decode_in_flight_within(
    packets: &[Vec<u8>],
    threads: usize,
    flight: usize,
    what: &str,
) -> Vec<Unit> {
    let packets = packets.to_vec();
    let what = format!("{what}, {flight} frames in flight on {threads} threads");
    within(LIMIT, &what, move || {
        decode_in_flight(&packets, threads, flight)
    })
}

/// Threads and frames in flight to hold to one thread.
const FLIGHTS: [(usize, usize); 6] = [(2, 2), (3, 4), (4, 2), (4, 4), (8, 4), (8, 8)];

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

#[test]
fn frames_in_flight_show_the_same_frames() {
    for path in vectors() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let packets = common::packets(&path);
        let one = decode(&packets, 1);
        // One frame in flight, and one thread, are decoding in place.
        assert!(
            decode_in_flight(&packets, 1, 4) == one,
            "{name}: queued on one thread"
        );
        assert!(
            decode_in_flight(&packets, 4, 1) == one,
            "{name}: queued, one in flight"
        );
        for (threads, flight) in FLIGHTS {
            let got = decode_in_flight_within(&packets, threads, flight, &name);
            assert!(
                got == one,
                "{name}: {flight} frames in flight on {threads} threads differ from one thread"
            );
        }
    }
}

#[test]
fn flights_repeat() {
    let all = vectors();
    for name in [
        "av1-1-b8-05-mv.ivf",
        "av1-1-b8-06-mfmv.ivf",
        "av1-1-b8-03-sizeup.mkv",
        "av1-1-b8-22-svc-L1T2.ivf",
    ] {
        let path = all
            .iter()
            .find(|p| p.ends_with(name))
            .unwrap_or_else(|| panic!("{name} is not in tests/data"));
        let packets = common::packets(path);
        let one = decode(&packets, 1);
        for run in 0..15 * soak() {
            for (threads, flight) in [(2, 2), (4, 4), (8, 4), (8, 8)] {
                let got = decode_in_flight_within(&packets, threads, flight, name);
                assert!(
                    got == one,
                    "{name}: run {run}, {flight} in flight on {threads}"
                );
            }
        }
    }
}

#[test]
fn damaged_streams_end_alike_in_flight() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut errors = 0;
    let mut cases = 0;
    for path in vectors() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let packets = common::packets(&path);
        let packets = &packets[..packets.len().min(10)];
        for case in 0..24 * soak() {
            let s = damage(packets, &mut rng);
            let what = format!("{name}, damage {case}");
            let one = decode_within(&s, 1, LIMIT, &what);
            errors += one.iter().filter(|u| matches!(u, Unit::Error(_))).count();
            cases += 1;
            for (threads, flight) in [(2, 2), (4, 4), (8, 8)] {
                let got = decode_in_flight_within(&s, threads, flight, &what);
                assert!(
                    got == one,
                    "{what}: {flight} frames in flight on {threads} threads differ from one \
                     thread:\n{got:?}\n{one:?}"
                );
            }
        }
    }
    assert!(
        errors * 4 >= cases,
        "{errors} errors in {cases} damaged streams"
    );
}

#[test]
fn a_bound_on_the_samples_in_flight_changes_no_frame() {
    let all = vectors();
    let path = all
        .iter()
        .find(|p| p.ends_with("av1-1-b8-05-mv.ivf"))
        .expect("the mv vector");
    let packets = common::packets(path);
    let one = decode(&packets, 1);
    // 352x288: room for one frame, for two and a half, for all eight.
    for samples in [1, 352 * 288 * 5 / 2, 352 * 288 * 8] {
        let packets = packets.clone();
        let got = within(LIMIT, "a bound on the samples in flight", move || {
            let mut dec = av1::Decoder::new();
            dec.set_threads(8);
            dec.set_frames_in_flight(8);
            dec.set_samples_in_flight(samples);
            let mut out = Vec::new();
            let mut unit = |r: av1::Result<Vec<av1::Frame>>| {
                out.push(Unit::Frames(
                    r.expect("an intact stream")
                        .iter()
                        .map(common::frame_md5)
                        .collect(),
                ))
            };
            for p in &packets {
                dec.queue(p);
                while let Some(r) = dec.poll() {
                    unit(r);
                }
            }
            while let Some(r) = dec.wait() {
                unit(r);
            }
            out
        });
        assert!(got == one, "{samples} samples in flight");
    }
}

#[test]
fn queued_units_and_decoding_in_place_do_not_mix() {
    let all = vectors();
    let path = all
        .iter()
        .find(|p| p.ends_with("av1-1-b8-05-mv.ivf"))
        .expect("the mv vector");
    let packets = common::packets(path);
    let mut dec = av1::Decoder::new();
    dec.set_threads(4);
    dec.set_frames_in_flight(4);
    dec.queue(&packets[0]);
    dec.queue(&packets[1]);
    assert_eq!(dec.queued(), 2);
    assert!(matches!(
        dec.decode_all(&packets[2]),
        Err(av1::Error::InvalidInput(_))
    ));
    let first = dec.wait().expect("a queued unit").expect("the first unit");
    assert_eq!(dec.queued(), 1);
    // A reset forgets the rest; the stream starts over.
    dec.reset();
    assert_eq!(dec.queued(), 0);
    assert!(dec.poll().is_none() && dec.wait().is_none());
    let again = dec.decode_all(&packets[0]).expect("the first unit");
    assert_eq!(
        first.iter().map(common::frame_md5).collect::<Vec<_>>(),
        again.iter().map(common::frame_md5).collect::<Vec<_>>()
    );
}
