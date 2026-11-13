//! Where the threads come from.
//!
//! With [`Decoder::set_threads`](crate::Decoder::set_threads) above one,
//! the decoder hands the tiles of a frame and the bands of its post-filters
//! to several threads at once. By default those are scoped threads of the
//! standard library, started for the job and joined after it.
//!
//! Some targets have no `std::thread::spawn`: on `wasm32-unknown-unknown` a
//! thread is another instance of the module on the same shared memory,
//! which only the embedder (the page's JavaScript) can create. Such an
//! embedder registers a [`Runner`] once with [`set_runner`] — typically a
//! pool of threads it started itself and keeps — and the decoder runs its
//! parallel work through it instead.

use std::sync::OnceLock;

/// Runs one job on several threads at once.
pub trait Runner: Send + Sync {
    /// Calls `job(0)` on the calling thread and `job(k)` for `k` in `1..n`
    /// on other threads, each `k` at most once and at the same time as the
    /// others, and returns when every call it made has returned.
    ///
    /// The job is a loop that takes work off a shared queue until none is
    /// left, so a runner may leave out any of `1..n` (a thread that is not
    /// there yet, or busy): the calls it does make finish the work between
    /// them. It must not return while a call is still running — the job
    /// borrows from the caller's stack.
    fn run(&self, n: usize, job: &(dyn Fn(usize) + Sync));
}

static RUNNER: OnceLock<&'static dyn Runner> = OnceLock::new();

/// Registers the process's [`Runner`]. The first registration stays;
/// returns whether this was it.
pub fn set_runner(runner: &'static dyn Runner) -> bool {
    RUNNER.set(runner).is_ok()
}

/// Runs `job(k)` for `k` in `0..n` at once, `job(0)` on the calling thread:
/// through the registered [`Runner`], else on scoped threads.
pub(crate) fn run(n: usize, job: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        return job(0);
    }
    match RUNNER.get() {
        Some(r) => r.run(n, job),
        None => std::thread::scope(|s| {
            for k in 1..n {
                s.spawn(move || job(k));
            }
            job(0);
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn scoped_threads_call_every_index_once() {
        let calls: Vec<AtomicUsize> = (0..6).map(|_| AtomicUsize::new(0)).collect();
        let caller = std::thread::current().id();
        run(6, &|k| {
            calls[k].fetch_add(1, Ordering::Relaxed);
            assert_eq!(k == 0, std::thread::current().id() == caller);
        });
        assert!(calls.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn one_worker_is_the_caller_alone() {
        let n = AtomicUsize::new(0);
        run(1, &|k| {
            assert_eq!(k, 0);
            n.fetch_add(1, Ordering::Relaxed);
        });
        run(0, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 2);
    }
}
