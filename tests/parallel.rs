//! A registered [`av1::parallel::Runner`] (what an embedder without
//! `std::thread::spawn` provides) decodes the committed vectors exactly:
//! one that starts every thread it is asked for, and one that starts none
//! and leaves the caller's own call to finish the work. A test binary of
//! its own: the runner is registered once for the process.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Runs the extra calls on scoped threads, or (`ALONE`) not at all; and
/// (`STARTS`) starts threads for a job that is not waited for.
struct Switchable;

static ALONE: AtomicBool = AtomicBool::new(false);
static RUNS: AtomicUsize = AtomicUsize::new(0);
static EXTRA_CALLS: AtomicUsize = AtomicUsize::new(0);
static STARTS: AtomicBool = AtomicBool::new(false);
static STARTED_CALLS: AtomicUsize = AtomicUsize::new(0);

/// The job started and not yet over: its address, the calls added for it,
/// and how many of them have not returned.
static STARTED: Mutex<(usize, usize, usize)> = Mutex::new((0, 0, 0));

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

    fn start(&self, n: usize, job: Arc<dyn Fn(usize) + Send + Sync>) -> usize {
        if !STARTS.load(Ordering::Relaxed) {
            return 0;
        }
        let address = Arc::as_ptr(&job) as *const () as usize;
        let mut st = STARTED.lock().unwrap();
        if st.2 == 0 {
            // No call of an earlier job is left: this one is new.
            *st = (address, 0, 0);
        } else if st.0 != address {
            return 0;
        }
        // Two threads at first, the rest when asked again: as threads that
        // come up late would be.
        let upto = if st.1 == 0 { 2.min(n - 1) } else { n - 1 };
        let added = upto.saturating_sub(st.1);
        for k in st.1 + 1..=st.1 + added {
            let job = job.clone();
            std::thread::spawn(move || {
                STARTED_CALLS.fetch_add(1, Ordering::Relaxed);
                job(k);
                STARTED.lock().unwrap().2 -= 1;
            });
        }
        st.1 += added;
        st.2 += added;
        added
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

    // Several frames in flight. A runner that cannot start a job nobody
    // waits for: every unit is decoded in place, through `run`.
    ALONE.store(false, Ordering::Relaxed);
    assert_eq!(decode_committed_in_flight(4, 4), frames);
    assert_eq!(STARTED_CALLS.load(Ordering::Relaxed), 0);
    // One that can: the frames are decoded on the threads it starts.
    STARTS.store(true, Ordering::Relaxed);
    for _ in 0..3 {
        assert_eq!(decode_committed_in_flight(4, 4), frames);
    }
    assert!(STARTED_CALLS.load(Ordering::Relaxed) > 0);
    // Every decoder ended its sessions: no started call is left, once
    // the threads are back from their last one.
    let since = std::time::Instant::now();
    while STARTED.lock().unwrap().2 != 0 {
        assert!(
            since.elapsed().as_secs() < 30,
            "a started call never returned"
        );
        std::thread::yield_now();
    }
}

/// [`decode_committed`] with the units queued and `flight` frames in
/// flight.
fn decode_committed_in_flight(threads: usize, flight: usize) -> usize {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut frames = 0;
    for path in common::vectors_in(&dir) {
        let expected = common::expected_md5s(&path);
        let mut dec = av1::Decoder::new();
        dec.set_threads(threads);
        dec.set_frames_in_flight(flight);
        let mut n = 0;
        let mut check = |shown: Vec<av1::Frame>| {
            // (As `decode`: a unit's last frame.)
            if let Some(f) = shown.last() {
                assert_eq!(
                    expected.get(n),
                    Some(&common::frame_md5(f)),
                    "{}: frame {n}",
                    path.display()
                );
                n += 1;
            }
        };
        for p in common::packets(&path) {
            dec.queue(&p);
            while let Some(r) = dec.poll() {
                check(r.unwrap());
            }
        }
        while let Some(r) = dec.wait() {
            check(r.unwrap());
        }
        assert_eq!(n, expected.len(), "{}", path.display());
        frames += n;
    }
    frames
}
