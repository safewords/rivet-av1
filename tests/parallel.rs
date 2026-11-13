//! A registered [`av1::parallel::Runner`] (what an embedder without
//! `std::thread::spawn` provides) decodes the committed vectors exactly:
//! one that starts every thread it is asked for, and one that starts none
//! and leaves the caller's own call to finish the work. A test binary of
//! its own: the runner is registered once for the process.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Runs the extra calls on scoped threads, or (`ALONE`) not at all.
struct Switchable;

static ALONE: AtomicBool = AtomicBool::new(false);
static RUNS: AtomicUsize = AtomicUsize::new(0);
static EXTRA_CALLS: AtomicUsize = AtomicUsize::new(0);

impl av1::parallel::Runner for Switchable {
    fn run(&self, n: usize, job: &(dyn Fn(usize) + Sync)) {
        RUNS.fetch_add(1, Ordering::Relaxed);
        if ALONE.load(Ordering::Relaxed) {
            return job(0);
        }
        std::thread::scope(|s| {
            for k in 1..n {
                s.spawn(move || {
                    EXTRA_CALLS.fetch_add(1, Ordering::Relaxed);
                    job(k)
                });
            }
            job(0);
        });
    }
}

fn decode_committed(threads: usize) -> usize {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut frames = 0;
    for path in common::vectors_in(&dir) {
        let expected = common::expected_md5s(&path);
        let mut dec = av1::Decoder::new();
        dec.set_threads(threads);
        let mut n = 0;
        for p in common::packets(&path) {
            if let Some(f) = dec.decode(&p).unwrap() {
                assert_eq!(
                    expected.get(n),
                    Some(&common::frame_md5(&f)),
                    "{}: frame {n}",
                    path.display()
                );
                n += 1;
            }
        }
        assert_eq!(n, expected.len(), "{}", path.display());
        frames += n;
    }
    frames
}

#[test]
fn a_registered_runner_decodes_exactly() {
    static RUNNER: Switchable = Switchable;
    assert!(av1::parallel::set_runner(&RUNNER));
    assert!(!av1::parallel::set_runner(&RUNNER), "the first stays");

    let frames = decode_committed(4);
    assert!(frames > 0);
    let runs = RUNS.load(Ordering::Relaxed);
    assert!(runs > 0, "the decoder went through the runner");
    assert!(EXTRA_CALLS.load(Ordering::Relaxed) > 0);

    // A runner whose threads are not there: the caller's call does it all.
    ALONE.store(true, Ordering::Relaxed);
    let extra = EXTRA_CALLS.load(Ordering::Relaxed);
    assert_eq!(decode_committed(4), frames);
    assert!(RUNS.load(Ordering::Relaxed) > runs);
    assert_eq!(EXTRA_CALLS.load(Ordering::Relaxed), extra);
}
