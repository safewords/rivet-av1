//! Where the threads come from.
//!
//! With [`Decoder::set_threads`](crate::Decoder::set_threads) above one,
//! the decoder hands the rows or tiles of a frame and the bands of its
//! post-filters to several threads at once. By default those are threads
//! of the standard library: a decoder keeps the ones it has started for
//! the next frame and ends them when it is dropped (the encoder starts
//! scoped threads for each job and joins them after it).
//!
//! Some targets have no `std::thread::spawn`: on `wasm32-unknown-unknown` a
//! thread is another instance of the module on the same shared memory,
//! which only the embedder (the page's JavaScript) can create. Such an
//! embedder registers a [`Runner`] once with [`set_runner`] — typically a
//! pool of threads it started itself and keeps — and the decoder runs its
//! parallel work through it instead.

use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

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

    /// Has other threads call `job(k)`, each with a `k` of its own in
    /// `1..n`, and returns without waiting for them: how many calls it has
    /// added. 0, what the provided method returns, if this runner cannot
    /// run a job that outlives the call (or has no thread free for it
    /// now).
    ///
    /// A decoder with several frames in flight
    /// ([`Decoder::set_frames_in_flight`](crate::Decoder::set_frames_in_flight))
    /// is decoding while its caller is away between two calls: this is how
    /// it gets the threads to do it on. It may ask again with the same job
    /// (the same `Arc`) and `n`, for threads that were not there the first
    /// time: the runner then adds calls, with further `k`, up to `n - 1`
    /// in all. Every call of the job returns when that decoder says so
    /// (before it decodes anything in place, when it is reset or dropped);
    /// until every call the runner has added has been made and has
    /// returned, it need not run anything else ([`Runner::run`] may call
    /// the job it is given on the calling thread alone, and `start` of
    /// another job add nothing). Without this a decoder decodes its frames
    /// one at a time, through [`Runner::run`].
    fn start(&self, n: usize, job: Arc<dyn Fn(usize) + Send + Sync>) -> usize {
        let _ = (n, job);
        0
    }
}

/// [`Runner::start`]: through the registered [`Runner`], else on the
/// threads of the decoder that is decoding on this thread
/// ([`Pool::enter`]), else not at all.
pub(crate) fn start(n: usize, job: Arc<dyn Fn(usize) + Send + Sync>) -> usize {
    if n <= 1 {
        return 0;
    }
    if let Some(r) = RUNNER.get() {
        return r.start(n, job);
    }
    match ENTERED.with(|e| e.borrow().clone()) {
        Some(pool) => pool.start(n, job),
        None => 0,
    }
}

static RUNNER: std::sync::OnceLock<&'static dyn Runner> = std::sync::OnceLock::new();

/// Registers the process's [`Runner`]. The first registration stays;
/// returns whether this was it.
pub fn set_runner(runner: &'static dyn Runner) -> bool {
    RUNNER.set(runner).is_ok()
}

/// Runs `job(k)` for `k` in `0..n` at once, `job(0)` on the calling thread:
/// through the registered [`Runner`], else on the threads of the decoder
/// that is decoding on this thread ([`Pool::enter`]), else on scoped
/// threads.
pub(crate) fn run(n: usize, job: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        return job(0);
    }
    if let Some(r) = RUNNER.get() {
        return r.run(n, job);
    }
    let entered = ENTERED.with(|e| e.borrow().clone());
    if let Some(pool) = entered
        && pool.run(n, job)
    {
        return;
    }
    std::thread::scope(|s| {
        for k in 1..n {
            s.spawn(move || job(k));
        }
        job(0);
    })
}

/// A job the pool's threads are in.
#[derive(Clone)]
enum Job {
    /// As [`run`] is given it, its borrow of the caller's stack erased so
    /// that threads which outlive the call can hold it.
    Borrowed(*const (dyn Fn(usize) + Sync)),
    /// As [`start`] is given it.
    Owned(Arc<dyn Fn(usize) + Send + Sync>),
}

// SAFETY: the pointee of `Borrowed` is `Sync`, so calling it from another
// thread is sound; that it is still there when called is `Inner::run`'s to
// ensure. (`Owned` is `Send` by itself.)
unsafe impl Send for Job {}

#[derive(Default)]
struct State {
    threads: Vec<JoinHandle<()>>,
    /// The job in progress, and the indices `next..limit` nobody has taken.
    job: Option<Job>,
    next: usize,
    limit: usize,
    /// Threads inside the job.
    running: usize,
    /// A thread's call of the job panicked.
    panicked: bool,
    /// The pool is being dropped: the threads return.
    shutdown: bool,
}

#[derive(Default)]
struct Inner {
    state: Mutex<State>,
    /// The threads wait here for a job.
    work: Condvar,
    /// `run` waits here for the threads to leave the job.
    idle: Condvar,
}

/// The threads a decoder keeps from one frame to the next: started when
/// first needed, parked between jobs, ended and joined when it is dropped.
#[derive(Default)]
pub(crate) struct Pool {
    inner: Arc<Inner>,
}

thread_local! {
    /// The pool of the decoder decoding on this thread, if any.
    static ENTERED: RefCell<Option<Arc<Inner>>> = const { RefCell::new(None) };
}

/// While it lives, [`run`] on this thread uses the pool it was made from
/// (or, should that pool be dropped meanwhile, scoped threads).
pub(crate) struct Entered {
    before: Option<Arc<Inner>>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        let before = self.before.take();
        ENTERED.with(|e| *e.borrow_mut() = before);
    }
}

impl Pool {
    /// Makes this pool the one [`run`] uses on the calling thread, until
    /// the guard is dropped.
    pub(crate) fn enter(&self) -> Entered {
        let before = ENTERED.with(|e| e.borrow_mut().replace(self.inner.clone()));
        Entered { before }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let threads = {
            let mut st = self.inner.lock();
            st.shutdown = true;
            std::mem::take(&mut st.threads)
        };
        self.inner.work.notify_all();
        for t in threads {
            // A thread that panicked did so inside a job, which reported it.
            let _ = t.join();
        }
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        // Nothing panics with the lock held; a poisoned lock's state is as
        // good as any.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What a pool thread does: takes an index of the job in progress,
    /// calls the job with it, waits for the next job.
    fn thread(self: Arc<Self>) {
        let mut st = self.lock();
        loop {
            if st.shutdown {
                return;
            }
            match st.job.clone() {
                Some(job) if st.next < st.limit => {
                    let k = st.next;
                    st.next += 1;
                    st.running += 1;
                    drop(st);
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &job {
                        // SAFETY: `run` put the job there and does not
                        // return (nor unwind) before `running` is back to
                        // zero, which this thread's call keeps it from
                        // being: the job is alive for the call.
                        Job::Borrowed(job) => (unsafe { &**job })(k),
                        Job::Owned(job) => job(k),
                    }));
                    let owned = matches!(job, Job::Owned(_));
                    drop(job);
                    st = self.lock();
                    st.running -= 1;
                    st.panicked |= r.is_err();
                    if st.running == 0 {
                        if owned && st.next >= st.limit {
                            // Nobody waits for a job that was started: the
                            // pool is free when every call of it has been
                            // made and has returned.
                            st.job = None;
                        }
                        self.idle.notify_all();
                    }
                }
                _ => st = self.work.wait(st).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    /// Makes sure the pool has a thread for every index of a job on `n`
    /// threads but the caller's, as far as threads can be had; returns how
    /// many it has.
    fn grow(self: &Arc<Self>, st: &mut State, n: usize) -> usize {
        while st.threads.len() + 1 < n {
            let inner = self.clone();
            let spawned = std::thread::Builder::new()
                .name("av1-decoder".into())
                .spawn(move || inner.thread());
            match spawned {
                Ok(t) => st.threads.push(t),
                Err(_) => break,
            }
        }
        st.threads.len()
    }

    /// [`start`] on this pool's threads: how many will call the job, 0 if
    /// the pool cannot take it (it is in the middle of a job, or has no
    /// thread to give).
    fn start(self: &Arc<Self>, n: usize, job: Arc<dyn Fn(usize) + Send + Sync>) -> usize {
        let mut st = self.lock();
        if st.shutdown {
            return 0;
        }
        // The calls added before, if this is the job already started.
        let before = match &st.job {
            None => 0,
            Some(Job::Owned(j)) if Arc::ptr_eq(j, &job) => st.limit - 1,
            Some(_) => return 0,
        };
        // As many calls as there are threads to make them: each thread
        // is in one call for as long as the job lasts.
        let calls = self.grow(&mut st, n).min(n - 1);
        if calls <= before {
            return 0;
        }
        if before == 0 {
            st.job = Some(Job::Owned(job));
            st.next = 1;
            st.panicked = false;
        }
        st.limit = calls + 1;
        drop(st);
        self.work.notify_all();
        calls - before
    }

    /// [`run`] on this pool's threads. Returns `false`, having called
    /// nothing, if the pool cannot take the job (it is in the middle of
    /// one, or has no thread to give).
    fn run(self: &Arc<Self>, n: usize, job: &(dyn Fn(usize) + Sync)) -> bool {
        {
            let mut st = self.lock();
            if st.job.is_some() || st.shutdown {
                return false;
            }
            if self.grow(&mut st, n) == 0 {
                return false;
            }
            // SAFETY: only the lifetime changes. The job is not called
            // after this function returns: `Leave`, below, waits for every
            // thread that took an index to come back, on return and on
            // unwinding alike, and clears the job before anything else can
            // take one.
            let job = unsafe {
                std::mem::transmute::<
                    *const (dyn Fn(usize) + Sync + '_),
                    *const (dyn Fn(usize) + Sync + 'static),
                >(job)
            };
            st.job = Some(Job::Borrowed(job));
            st.next = 1;
            st.limit = n;
            st.panicked = false;
        }
        self.work.notify_all();
        /// Ends the job: no further index is given out, and the threads
        /// that have one are waited for.
        struct Leave<'a>(&'a Inner);
        impl Drop for Leave<'_> {
            fn drop(&mut self) {
                let mut st = self.0.lock();
                st.limit = 0;
                while st.running > 0 {
                    st = self.0.idle.wait(st).unwrap_or_else(|e| e.into_inner());
                }
                st.job = None;
            }
        }
        let leave = Leave(self);
        // The caller is index 0. The job is a loop over a shared queue, so
        // when this returns there is nothing left to start.
        job(0);
        drop(leave);
        if self.lock().panicked {
            panic!("a decoder thread panicked");
        }
        true
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
    fn a_pool_keeps_its_threads_between_jobs() {
        let pool = Pool::default();
        let ids = Mutex::new(std::collections::HashSet::new());
        let caller = std::thread::current().id();
        for round in 0..50 {
            let _entered = pool.enter();
            let calls: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
            // Every index is taken: the job holds index 0 until the others
            // have been.
            run(4, &|k| {
                calls[k].fetch_add(1, Ordering::Relaxed);
                assert_eq!(k == 0, std::thread::current().id() == caller);
                ids.lock().unwrap().insert(std::thread::current().id());
                while k == 0 && calls.iter().any(|c| c.load(Ordering::Relaxed) == 0) {
                    std::thread::yield_now();
                }
            });
            assert!(
                calls.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "round {round}"
            );
        }
        // The caller and the pool's three threads, whichever of them took
        // the indices (a thread back early takes another): no others.
        let ids = ids.lock().unwrap().len();
        assert!((2..=4).contains(&ids), "{ids} threads over fifty jobs");
        assert_eq!(pool.inner.lock().threads.len(), 3);
        // Not entered: scoped threads again.
        let n = AtomicUsize::new(0);
        run(3, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 3);
        assert_eq!(pool.inner.lock().threads.len(), 3);
    }

    #[test]
    fn a_started_job_runs_while_the_caller_is_away() {
        let pool = Pool::default();
        let _entered = pool.enter();
        let over = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let job: Arc<dyn Fn(usize) + Send + Sync> = {
            let (over, calls) = (over.clone(), calls.clone());
            Arc::new(move |k| {
                calls.lock().unwrap().push(k);
                while !over.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }
            })
        };
        assert_eq!(start(3, job.clone()), 2);
        // The same job again: nothing to add; then two more threads.
        assert_eq!(start(3, job.clone()), 0);
        assert_eq!(start(5, job.clone()), 2);
        // Another job, or a job to wait for: the pool is taken (the
        // latter then runs on scoped threads).
        assert_eq!(start(3, Arc::new(|_| {})), 0);
        let n = AtomicUsize::new(0);
        run(3, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 3);
        // Every call is made, each index once, and when they have all
        // returned the pool is free again.
        while calls.lock().unwrap().len() < 4 {
            std::thread::yield_now();
        }
        over.store(true, Ordering::Relaxed);
        let mut ks = calls.lock().unwrap().clone();
        ks.sort_unstable();
        assert_eq!(ks, [1, 2, 3, 4]);
        while pool.inner.lock().job.is_some() {
            std::thread::yield_now();
        }
        assert_eq!(start(2, Arc::new(|_| {})), 1);
    }

    #[test]
    fn a_job_inside_a_job_does_not_take_the_pool() {
        let pool = Pool::default();
        let _entered = pool.enter();
        let n = AtomicUsize::new(0);
        let outer = AtomicUsize::new(0);
        run(2, &|k| {
            outer.fetch_add(1, Ordering::Relaxed);
            // The pool is busy with the outer job: scoped threads.
            let _entered = pool.enter();
            run(2, &|_| {
                n.fetch_add(1, Ordering::Relaxed);
            });
            // (The caller stays until the pool's thread has come.)
            while k == 0 && outer.load(Ordering::Relaxed) < 2 {
                std::thread::yield_now();
            }
        });
        assert_eq!(n.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn a_panic_in_a_pool_thread_reaches_the_caller() {
        let pool = Pool::default();
        let started = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _entered = pool.enter();
            run(3, &|k| {
                started.fetch_add(1, Ordering::Relaxed);
                // The caller waits for the others to have started, so that
                // one of them is there to panic.
                while k == 0 && started.load(Ordering::Relaxed) < 2 {
                    std::thread::yield_now();
                }
                assert!(k == 0, "index {k} panics");
            });
        }));
        assert!(r.is_err());
        // The pool is usable afterwards.
        let _entered = pool.enter();
        let n = AtomicUsize::new(0);
        run(3, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert!(n.load(Ordering::Relaxed) >= 1);
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
