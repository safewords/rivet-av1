//! Arrays that several threads work on at once, each on the part it has
//! borrowed.
//!
//! A frame's sample planes and its per-4x4 mode info are single
//! allocations that the stages of decoding work on at the same time: one
//! thread parses a superblock row while others reconstruct the rows parsed
//! before it. Rust's borrows cannot say "these rows are mine, those are
//! yours" for an allocation whose rows change hands as the frame
//! progresses, so [`Shared`] says it at run time: it is a window onto the
//! allocation, a range of elements it may read and a range it may write,
//! and a window is only ever made ([`Lender::window`]) if no other window
//! in existence could conflict with it. Everything else in the decoder is
//! safe code on top of this type; `docs/threading.md` says who borrows
//! what, and when.
//!
//! # The invariant
//!
//! For any two windows of one allocation that exist at the same time, the
//! write range of each is disjoint from both ranges of the other. Hence a
//! `&mut T` this type hands out (through `&mut self`, inside the window's
//! write range) is the only reference to its element anywhere, and a `&T`
//! (through `&self`, inside the read range) is to an element that only this
//! window could write, which needs `&mut self`. That is the whole argument
//! for every `unsafe` block below.
//!
//! [`Lender::window`] and [`Shared::narrow`] check the invariant under the
//! allocation's lock and panic if it would not hold: a scheduling mistake
//! (two stages given the same rows) stops the decoder instead of racing.
//! Registering and dropping a window lock the same mutex, so whatever one
//! thread wrote through a window it has dropped is visible to the thread
//! that borrows those elements next.
//!
//! An array that nothing shares is the window onto all of it that
//! `Shared::from(vec)` makes, and indexing that costs what indexing the
//! `Vec` did.

use std::ops::{Index, IndexMut, Range, RangeFrom};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// One window on loan: its identity and its two ranges.
struct Loan {
    id: u64,
    read: Range<usize>,
    write: Range<usize>,
}

/// The windows of an allocation in existence.
#[derive(Default)]
struct Loans {
    next_id: u64,
    out: Vec<Loan>,
}

fn overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

impl Loans {
    /// Whether a window with these ranges could exist beside the others
    /// (`except` is the window being replaced, if any).
    fn admits(&self, read: &Range<usize>, write: &Range<usize>, except: Option<u64>) -> bool {
        self.out.iter().all(|l| {
            Some(l.id) == except
                || !(overlap(write, &l.read) || overlap(write, &l.write) || overlap(read, &l.write))
        })
    }
}

/// The allocation behind a set of windows: a `Vec<T>` taken apart, put
/// back together and dropped with the last window or lender.
struct Store<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    loans: Mutex<Loans>,
}

// SAFETY: a `Store` owns its elements like the `Vec` it was made from.
// They are reached only through windows, under the module's invariant.
unsafe impl<T: Send + Sync> Send for Store<T> {}
unsafe impl<T: Send + Sync> Sync for Store<T> {}

impl<T> Drop for Store<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr`, `len` and `cap` are those of the `Vec` this store
        // was made from (`Shared::from`), which was forgotten there; no
        // window is left, since each holds an `Arc` of the store.
        drop(unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) });
    }
}

/// A window onto a shared array: the elements `read` may be read through
/// it and the elements `write` (a part of `read`) written. Indices are
/// those of the whole array, whatever the window.
pub(crate) struct Shared<T> {
    ptr: NonNull<T>,
    read_lo: usize,
    read_len: usize,
    write_lo: usize,
    write_len: usize,
    /// `None` for the empty window of `Shared::default()`.
    store: Option<Arc<Store<T>>>,
    id: u64,
}

// SAFETY: a window is a reference to elements of the store: shared to the
// ones it may only read, exclusive to the ones it may write (the module's
// invariant). Sending it sends those references; sharing it shares `&T`.
unsafe impl<T: Send + Sync> Send for Shared<T> {}
unsafe impl<T: Send + Sync> Sync for Shared<T> {}

/// What makes windows onto one shared array.
pub(crate) struct Lender<T> {
    store: Arc<Store<T>>,
}

impl<T> Clone for Lender<T> {
    fn clone(&self) -> Self {
        Lender {
            store: self.store.clone(),
        }
    }
}

impl<T> Lender<T> {
    /// The length of the array.
    pub(crate) fn len(&self) -> usize {
        self.store.len
    }

    /// A window that reads `read` and writes `write`, which must lie
    /// inside `read` (or be empty).
    ///
    /// # Panics
    ///
    /// If a window in existence writes any of `read`, or reads or writes
    /// any of `write`: the caller's schedule is wrong.
    #[track_caller]
    pub(crate) fn window(&self, read: Range<usize>, write: Range<usize>) -> Shared<T> {
        let s = &self.store;
        assert!(read.start <= read.end && read.end <= s.len);
        assert!(write.is_empty() || (read.start <= write.start && write.end <= read.end));
        let mut loans = s.loans.lock().unwrap_or_else(|e| e.into_inner());
        if !loans.admits(&read, &write, None) {
            // Not with the lock held: the other windows are still given
            // back through it.
            drop(loans);
            panic!(
                "a shared array's window (read {read:?}, write {write:?}) conflicts with one on loan"
            );
        }
        let id = loans.next_id;
        loans.next_id += 1;
        loans.out.push(Loan {
            id,
            read: read.clone(),
            write: write.clone(),
        });
        Shared {
            ptr: s.ptr,
            read_lo: read.start,
            read_len: read.len(),
            write_lo: write.start,
            write_len: write.len(),
            store: Some(s.clone()),
            id,
        }
    }

    /// The window onto all of the array, to read and write: what nothing
    /// else may hold a window beside.
    #[track_caller]
    pub(crate) fn whole(&self) -> Shared<T> {
        self.window(0..self.store.len, 0..self.store.len)
    }
}

impl<T> From<Vec<T>> for Shared<T> {
    /// The array `v`, as the one window onto all of it.
    fn from(v: Vec<T>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (len, cap) = (v.len(), v.capacity());
        // A `Vec`'s pointer is never null.
        let ptr = NonNull::new(v.as_mut_ptr()).expect("a Vec's pointer");
        let store = Arc::new(Store {
            ptr,
            len,
            cap,
            loans: Mutex::new(Loans {
                next_id: 1,
                out: vec![Loan {
                    id: 0,
                    read: 0..len,
                    write: 0..len,
                }],
            }),
        });
        Shared {
            ptr,
            read_lo: 0,
            read_len: len,
            write_lo: 0,
            write_len: len,
            store: Some(store),
            id: 0,
        }
    }
}

impl<T> Default for Shared<T> {
    /// A window onto nothing.
    fn default() -> Self {
        Shared {
            ptr: NonNull::dangling(),
            read_lo: 0,
            read_len: 0,
            write_lo: 0,
            write_len: 0,
            store: None,
            id: 0,
        }
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        if let Some(s) = &self.store {
            // A poisoned lock still holds the loans: a window given back
            // while another thread unwinds must not make things worse.
            let mut loans = s.loans.lock().unwrap_or_else(|e| e.into_inner());
            loans.out.retain(|l| l.id != self.id);
        }
    }
}

impl<T: Clone> Clone for Shared<T> {
    /// A copy of the array, as a new array of its own. For the window onto
    /// all of an array only.
    fn clone(&self) -> Self {
        Shared::from(self.as_slice().to_vec())
    }
}

impl<T> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("len", &self.len())
            .field("read", &(self.read_lo..self.read_lo + self.read_len))
            .field("write", &(self.write_lo..self.write_lo + self.write_len))
            .finish()
    }
}

impl<T> Shared<T> {
    /// The length of the whole array (not of the window).
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.len)
    }

    /// What makes further windows onto this array (once this one is
    /// dropped or narrowed out of their way).
    pub(crate) fn lender(&self) -> Lender<T> {
        Lender {
            store: self.store.clone().expect("a window onto an array"),
        }
    }

    /// Gives up part of the window: from now on it reads `read` and writes
    /// `write`, each inside what it had.
    #[track_caller]
    pub(crate) fn narrow(&mut self, read: Range<usize>, write: Range<usize>) {
        let (r0, w0) = (self.read_range(), self.write_range());
        assert!(read.is_empty() || (r0.start <= read.start && read.end <= r0.end));
        assert!(write.is_empty() || (w0.start <= write.start && write.end <= w0.end));
        assert!(write.is_empty() || (read.start <= write.start && write.end <= read.end));
        let s = self.store.as_ref().expect("a window onto an array");
        let mut loans = s.loans.lock().unwrap_or_else(|e| e.into_inner());
        // A narrower window conflicts with nothing the wider one did not.
        debug_assert!(loans.admits(&read, &write, Some(self.id)));
        let l = loans
            .out
            .iter_mut()
            .find(|l| l.id == self.id)
            .expect("this window's loan");
        l.read = read.clone();
        l.write = write.clone();
        drop(loans);
        self.read_lo = read.start;
        self.read_len = read.len();
        self.write_lo = write.start;
        self.write_len = write.len();
    }

    /// The elements this window may read.
    pub(crate) fn read_range(&self) -> Range<usize> {
        self.read_lo..self.read_lo + self.read_len
    }

    /// The elements this window may write.
    pub(crate) fn write_range(&self) -> Range<usize> {
        self.write_lo..self.write_lo + self.write_len
    }

    /// The whole array. For the window onto all of it only.
    #[inline]
    #[track_caller]
    pub(crate) fn as_slice(&self) -> &[T] {
        assert!(self.read_lo == 0 && self.read_len == self.len());
        // SAFETY: the window reads every element of the store, `len` of
        // them at `ptr`, and no other window writes them (the invariant).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.read_len) }
    }

    /// The whole array, to write. For the window onto all of it only.
    #[cfg(test)]
    #[inline]
    #[track_caller]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(self.write_lo == 0 && self.write_len == self.len());
        // SAFETY: the window writes every element of the store, `len` of
        // them at `ptr`, and no other window reads or writes them (the
        // invariant); `&mut self` makes this the only reference through
        // this window.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.write_len) }
    }

    /// The pointer to element 0 of the array and the range of elements
    /// that may be read and written through it, for the kernels that work
    /// on a plane by address. Reading or writing outside the range is
    /// undefined behaviour, as is using the pointer after `self` is
    /// borrowed again or dropped.
    #[inline]
    pub(crate) fn as_mut_ptr_range(&mut self) -> (*mut T, Range<usize>) {
        (self.ptr.as_ptr(), self.write_range())
    }
}

impl<T> Index<usize> for Shared<T> {
    type Output = T;

    #[inline(always)]
    #[track_caller]
    fn index(&self, i: usize) -> &T {
        assert!(
            i.wrapping_sub(self.read_lo) < self.read_len,
            "index outside the window's read range"
        );
        // SAFETY: `i` is in the read range, which lies inside the store's
        // `len` elements at `ptr`; no other window writes it (the
        // invariant), and this one writes only through `&mut self`.
        unsafe { &*self.ptr.as_ptr().add(i) }
    }
}

impl<T> IndexMut<usize> for Shared<T> {
    #[inline(always)]
    #[track_caller]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(
            i.wrapping_sub(self.write_lo) < self.write_len,
            "index outside the window's write range"
        );
        // SAFETY: `i` is in the write range, which lies inside the store's
        // `len` elements at `ptr`; no other window reads or writes it (the
        // invariant), and `&mut self` makes this the only reference
        // through this window.
        unsafe { &mut *self.ptr.as_ptr().add(i) }
    }
}

impl<T> Index<Range<usize>> for Shared<T> {
    type Output = [T];

    #[inline(always)]
    #[track_caller]
    fn index(&self, r: Range<usize>) -> &[T] {
        assert!(
            r.start <= r.end && self.read_lo <= r.start && r.end <= self.read_lo + self.read_len,
            "range outside the window's read range"
        );
        // SAFETY: as for one element, for each element of `r`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(r.start), r.end - r.start) }
    }
}

impl<T> IndexMut<Range<usize>> for Shared<T> {
    #[inline(always)]
    #[track_caller]
    fn index_mut(&mut self, r: Range<usize>) -> &mut [T] {
        assert!(
            r.start <= r.end && self.write_lo <= r.start && r.end <= self.write_lo + self.write_len,
            "range outside the window's write range"
        );
        // SAFETY: as for one element, for each element of `r`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(r.start), r.end - r.start) }
    }
}

impl<T> Index<RangeFrom<usize>> for Shared<T> {
    type Output = [T];

    /// From `r.start` to the end of the window's read range.
    #[inline(always)]
    #[track_caller]
    fn index(&self, r: RangeFrom<usize>) -> &[T] {
        &self[r.start..self.read_lo + self.read_len]
    }
}

impl<T> IndexMut<RangeFrom<usize>> for Shared<T> {
    /// From `r.start` to the end of the window's write range.
    #[inline(always)]
    #[track_caller]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut [T] {
        let end = self.write_lo + self.write_len;
        &mut self[r.start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_whole_array_indexes_like_its_vec() {
        let mut a: Shared<u16> = Shared::from((0..10u16).collect::<Vec<_>>());
        assert_eq!(a.len(), 10);
        assert_eq!(a[3], 3);
        a[3] = 30;
        a[4..6].copy_from_slice(&[40, 50]);
        assert_eq!(&a[2..7], &[2, 30, 40, 50, 6]);
        assert_eq!(&a[8..], &[8, 9]);
        a[8..].fill(1);
        assert_eq!(a.as_slice(), &[0, 1, 2, 30, 40, 50, 6, 7, 1, 1]);
        let b = a.clone();
        a.as_mut_slice().fill(0);
        assert_eq!(b[3], 30, "a clone is an array of its own");
        let e: Shared<u16> = Shared::default();
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn windows_see_their_ranges_at_the_arrays_indices() {
        let a: Shared<u32> = Shared::from(vec![0; 100]);
        let lender = a.lender();
        drop(a);
        // A writer of 40..60 that reads from 20 on, readers below it.
        let mut w = lender.window(20..100, 40..60);
        let r1 = lender.window(0..40, 0..0);
        let r2 = lender.window(10..30, 0..0);
        w[40] = 7;
        w[41..43].fill(8);
        assert_eq!(w[20], 0);
        assert_eq!(w[58..].len(), 42, "to the end of the read range");
        assert_eq!(w[58..].iter_mut().len(), 2, "to the end of the write range");
        assert_eq!(r1[39], 0);
        assert_eq!(r2[10..30].len(), 20);
        // The writer gives up 40..50; a reader may now have them.
        w.narrow(20..100, 50..60);
        let r3 = lender.window(0..50, 0..0);
        assert_eq!((r3[40], r3[41], r3[42]), (7, 8, 8));
        drop((w, r1, r2, r3));
        let all = lender.whole();
        assert_eq!(all.as_slice()[40..43], [7, 8, 8]);
    }

    fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
        std::panic::catch_unwind(f).is_err()
    }

    #[test]
    fn conflicting_windows_are_refused() {
        let a: Shared<u8> = Shared::from(vec![0; 64]);
        let lender = a.lender();
        // The whole array is on loan (`a`): nothing else is admitted.
        assert!(panics({
            let l = lender.clone();
            move || drop(l.window(0..1, 0..0))
        }));
        drop(a);
        let w = lender.window(16..48, 16..32);
        for (read, write) in [
            (0..17, 0..0),    // reads what `w` writes
            (31..40, 0..0),   // the same, at its end
            (0..64, 40..48),  // writes what `w` reads
            (0..64, 20..24),  // writes what `w` writes
            (32..64, 47..48), // one element `w` reads
        ] {
            let l = lender.clone();
            assert!(
                panics(move || drop(l.window(read.clone(), write.clone()))),
                "admitted beside a writer of 16..32 reading 16..48"
            );
        }
        // Beside it: readers of what it only reads, a writer of the rest.
        let r = lender.window(32..48, 0..0);
        let w2 = lender.window(0..16, 0..16);
        let w3 = lender.window(48..64, 48..64);
        drop((w, r, w2, w3));
        // A write range must lie in the read range.
        let l = lender.clone();
        assert!(panics(move || drop(l.window(0..8, 4..12))));
        // Nothing was left on loan by the refusals.
        drop(lender.whole());
    }

    #[test]
    fn indexing_outside_the_window_panics() {
        let a: Shared<u8> = Shared::from(vec![0; 64]);
        let lender = a.lender();
        drop(a);
        let mut w = lender.window(8..40, 16..32);
        assert!(panics(std::panic::AssertUnwindSafe(|| {
            let _ = w[7];
        })));
        assert!(panics(std::panic::AssertUnwindSafe(|| {
            let _ = w[40];
        })));
        assert!(panics(std::panic::AssertUnwindSafe(|| {
            let _ = &w[8..41];
        })));
        assert!(panics(std::panic::AssertUnwindSafe(|| w[15] = 1)));
        assert!(panics(std::panic::AssertUnwindSafe(|| w[32] = 1)));
        assert!(panics(std::panic::AssertUnwindSafe(|| w[16..33].fill(1))));
        assert!(panics(std::panic::AssertUnwindSafe(|| {
            let _ = w.as_slice();
        })));
        assert!(panics(std::panic::AssertUnwindSafe(|| w
            .as_mut_slice()
            .fill(1))));
        assert!(panics(std::panic::AssertUnwindSafe(
            || w.narrow(8..40, 16..33)
        )));
        w[16] = 1;
        w[31] = 1;
        assert_eq!((w[8], w[39], w[16]), (0, 0, 1));
    }

    #[test]
    fn rows_change_hands_between_threads() {
        // A writer fills rows in order and gives each up; a reader sums
        // every row given up so far. The sums are those of a sequential
        // run, whatever the interleaving.
        const ROWS: usize = 64;
        const W: usize = 257;
        for _ in 0..50 {
            let a: Shared<u32> = Shared::from(vec![0; ROWS * W]);
            let lender = a.lender();
            drop(a);
            let done = std::sync::atomic::AtomicUsize::new(0);
            let total = std::thread::scope(|s| {
                s.spawn(|| {
                    let mut w = lender.window(0..ROWS * W, 0..ROWS * W);
                    for r in 0..ROWS {
                        for (i, v) in w[r * W..(r + 1) * W].iter_mut().enumerate() {
                            *v = (r * W + i) as u32;
                        }
                        w.narrow((r + 1) * W..ROWS * W, (r + 1) * W..ROWS * W);
                        done.store(r + 1, std::sync::atomic::Ordering::Release);
                    }
                });
                let reader = s.spawn(|| {
                    let mut sum = 0u64;
                    let mut next = 0;
                    while next < ROWS {
                        let d = done.load(std::sync::atomic::Ordering::Acquire);
                        if d > next {
                            let r = lender.window(next * W..d * W, 0..0);
                            sum += r[next * W..d * W].iter().map(|&v| v as u64).sum::<u64>();
                            next = d;
                        } else {
                            std::thread::yield_now();
                        }
                    }
                    sum
                });
                reader.join().expect("reader")
            });
            let n = (ROWS * W) as u64;
            assert_eq!(total, n * (n - 1) / 2);
        }
    }

    #[test]
    fn the_array_is_dropped_once() {
        let counter = Arc::new(());
        let a: Shared<Arc<()>> = Shared::from(vec![counter.clone(); 5]);
        assert_eq!(Arc::strong_count(&counter), 6);
        let lender = a.lender();
        drop(a);
        let w = lender.window(0..5, 0..0);
        drop(lender);
        assert_eq!(Arc::strong_count(&counter), 6, "a window keeps the array");
        drop(w);
        assert_eq!(Arc::strong_count(&counter), 1);
    }
}
