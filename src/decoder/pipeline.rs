//! Frames as tasks: a one-tile frame is parsed by one thread, its
//! superblock rows reconstructed behind the parsing and filtered behind
//! their reconstruction by the others; and several such frames can be in
//! flight at once, each waiting for the frames it refers to.
//!
//! `docs/threading.md` derives the rules from the specification; this is
//! their bookkeeping. A frame in flight is a [`Job`]. Its work is *tasks*:
//!
//! - **parse** the frame: set up its state from its references' saved
//!   state, then the tile's superblock rows in order ([`Pass::Parse`]),
//!   each row's record handed on as it completes;
//! - **reconstruct** row `r` from its record ([`Pass::Recon`]), a
//!   superblock at a time behind the row above;
//! - **deblock** row `r` (7.14), once it and the rows before it are
//!   reconstructed and the rows before it deblocked;
//! - **CDEF** row `r` (7.15), once the rows it reads are deblocked;
//! - **restore** stripe `s` (7.17), once the rows it reads have been
//!   through CDEF.
//!
//! Threads take tasks from the oldest frame that has one ready
//! ([`Engine`]). A task holds the rows it works on as windows of the
//! frame's arrays ([`Shared`]): no two tasks that run at the same time are
//! given rows of which either writes what the other touches, and the
//! windows check that they were not.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

use crate::consts::*;
use crate::decoder::postfilter::{self, LfLevels, SharedPlane};
use crate::decoder::record::{Pass, RowRecord};
use crate::decoder::shared::{Lender, Shared};
use crate::decoder::state::{FrameBuf, Mi, PlaneBuf, Pool, RefData};
use crate::decoder::tile::TileDecoder;
use crate::decoder::{CoefficientReader, FrameCtx, LrPlane};
use crate::header::FrameHeader;
use crate::obu::SequenceHeader;

/// The rows the loop filter of a superblock row reaches above it, by
/// plane: the widest luma filter reads 7 samples across an edge (7.14.6.2
/// with `filterSize` 16), the widest chroma filter 3.
const LF_REACH: [usize; 3] = [8, 4, 4];
/// The rows CDEF reads above and below an 8x8 block (7.15.3:
/// `Cdef_Directions` reaches two samples).
const CDEF_REACH: usize = 2;
/// The rows of the deblocked frame loop restoration reads above and below
/// a stripe (7.17.6).
const LR_REACH: usize = 2;

/// The threads an engine has a place for: more are not given tasks.
pub(crate) const MAX_WORKERS: usize = 64;

/// The frame's shape, and which stages it has.
#[derive(Clone, Copy)]
struct Geometry {
    /// Superblock rows and columns of the tile (the frame).
    rows: usize,
    cols: usize,
    /// The superblock size in 4x4 units and in luma samples.
    sb4: usize,
    sb_px: usize,
    mi_rows: usize,
    mi_cols: usize,
    /// Row stride of the per-4x4 arrays.
    ms: usize,
    num_planes: usize,
    ssx: usize,
    ssy: usize,
    /// Whether the post-filters run here, row by row. (Not with
    /// super-resolution, whose upscaling stands between CDEF and loop
    /// restoration, 7.16: such a frame is filtered whole, when its rows
    /// are reconstructed.)
    filters: bool,
    lf: bool,
    cdef: bool,
    /// Loop restoration stripes (64 luma rows, the first 56), or 0 when no
    /// plane is restored; and the planes that are.
    stripes: usize,
    restored: [bool; 3],
    frame_height: usize,
    cdef_stride: usize,
}

impl Geometry {
    fn of(f: &FrameCtx) -> Self {
        let sb4 = if f.seq.use_128x128_superblock { 32 } else { 16 };
        let h = &f.hdr;
        let filters = !h.use_superres;
        let lr = filters && h.uses_lr;
        let frame_height = h.frame_height as usize;
        Geometry {
            rows: f.mi_rows.div_ceil(sb4),
            cols: f.mi_cols.div_ceil(sb4),
            sb4,
            sb_px: sb4 * MI_SIZE,
            mi_rows: f.mi_rows,
            mi_cols: f.mi_cols,
            ms: f.ms,
            num_planes: f.num_planes,
            ssx: f.ssx,
            ssy: f.ssy,
            filters,
            lf: filters && (h.loop_filter_level[0] != 0 || h.loop_filter_level[1] != 0),
            cdef: filters && postfilter::cdef_applies(f),
            stripes: if lr {
                (frame_height + 8).div_ceil(64)
            } else {
                0
            },
            restored: std::array::from_fn(|p| {
                lr && p < f.num_planes && h.frame_restoration_type[p] != RESTORE_NONE
            }),
            frame_height,
            cdef_stride: f.cdef_stride,
        }
    }

    fn ss(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.ssx, self.ssy)
        }
    }

    /// The elements of superblock rows `r0..r1` in a per-4x4 array of
    /// `len` elements whose rows are every `1 << sy`-th mode info row:
    /// whole rows, with the margin below the frame for its last row
    /// (blocks overhanging the frame write there).
    fn units(&self, r0: usize, r1: usize, sy: usize, len: usize) -> Range<usize> {
        let start = (((r0 * self.sb4) >> sy) * self.ms).min(len);
        let end = if r1 >= self.rows {
            len
        } else {
            ((r1 * self.sb4) >> sy) * self.ms
        };
        start..end.max(start)
    }

    /// What is left of such an array after superblock row `r`.
    fn units_after(&self, r: usize, sy: usize, len: usize) -> Range<usize> {
        if r + 1 < self.rows {
            self.units(r + 1, self.rows, sy, len)
        } else {
            len..len
        }
    }

    /// The sample rows of superblock row `r` in a plane `h` rows high: to
    /// the plane's end for the frame's last row, whose blocks' predictions
    /// overhang the frame into the padding.
    fn band(&self, plane: usize, r: usize, h: usize) -> Range<usize> {
        let (_, sy) = self.ss(plane);
        let top = ((r * self.sb_px) >> sy).min(h);
        let bottom = if r + 1 >= self.rows {
            h
        } else {
            (((r + 1) * self.sb_px) >> sy).min(h)
        };
        top..bottom
    }

    /// The mode info rows of superblock row `r` inside the frame.
    fn mi_band(&self, r: usize) -> Range<usize> {
        r * self.sb4..((r + 1) * self.sb4).min(self.mi_rows)
    }

    /// The columns of superblock `k` in a plane, inside the frame's mode
    /// info: the samples of the row's last line that a row below may ask
    /// for.
    fn sb_columns(&self, plane: usize, k: usize) -> Range<usize> {
        let (sx, _) = self.ss(plane);
        let end = ((k + 1) * self.sb_px).min(self.mi_cols * MI_SIZE);
        (k * self.sb_px) >> sx..end >> sx
    }

    /// The superblock rows that hold the luma rows of stripe `s`
    /// (`StripeStartY..=StripeEndY` of 7.17.1, inside the frame).
    fn stripe_bands(&self, s: usize) -> Range<usize> {
        let first = (s * 64).saturating_sub(8);
        let last = (s * 64 + 55).min(self.frame_height - 1);
        first / self.sb_px..(last / self.sb_px + 1).min(self.rows)
    }

    /// The CDEF indices of superblock row `r` (one for every 64x64).
    fn cdef_idx(&self, r: usize, len: usize) -> Range<usize> {
        let a = ((r * self.sb4) >> 4) * self.cdef_stride;
        let b = (((r + 1) * self.sb4) >> 4) * self.cdef_stride;
        a.min(len)..b.min(len)
    }
}

/// A piece of a frame's work, for a thread to carry out.
#[derive(Clone, Copy, Debug)]
enum Task {
    Parse,
    Reconstruct(usize),
    Deblock(usize),
    Cdef(usize),
    Restore(usize),
}

/// The bookkeeping of a frame's rows, once its parsing has set it up.
struct Rows {
    g: Geometry,
    /// The records of the rows parsed and not yet taken to reconstruct.
    parsed: Vec<Option<RowRecord>>,
    /// Rows parsed so far; whether there will be no more.
    parsed_rows: usize,
    parse_done: bool,
    /// The parsing thread's CDEF indices and loop restoration units, as
    /// far as the rows parsed: a filtering task copies the ones it needs.
    cdef_idx: Vec<i8>,
    lr: [LrPlane; 3],
    /// The next row to reconstruct (rows are taken in order), the
    /// superblocks reconstructed in each row, and the rows reconstructed
    /// from the first on without a gap.
    next_recon: usize,
    progress: Vec<usize>,
    recon_done: Vec<bool>,
    recon_rows: usize,
    /// Each row's last line of samples as reconstructed, by plane: valid
    /// for the superblocks `progress` counts.
    lines: Vec<[Vec<u16>; 3]>,
    /// The rows deblocked (in order, one at a time), and whether one is
    /// being.
    lf_rows: usize,
    lf_busy: bool,
    /// The next row for CDEF, and the rows through it without a gap.
    next_cdef: usize,
    cdef_done: Vec<bool>,
    cdef_rows: usize,
    /// The next stripe to restore, and the stripes restored.
    next_stripe: usize,
    stripes_done: usize,
    /// Records to write the next rows into.
    spare: Vec<RowRecord>,
}

impl Rows {
    fn new(g: Geometry, f: &FrameCtx) -> Self {
        Rows {
            g,
            parsed: (0..g.rows).map(|_| None).collect(),
            parsed_rows: 0,
            parse_done: false,
            cdef_idx: f.cdef_idx.clone(),
            lr: f.lr.clone(),
            next_recon: 0,
            progress: vec![0; g.rows],
            recon_done: vec![false; g.rows],
            recon_rows: 0,
            lines: (0..g.rows)
                .map(|_| {
                    std::array::from_fn(|p| {
                        if p < g.num_planes {
                            vec![0; f.cur.planes[p].w]
                        } else {
                            Vec::new()
                        }
                    })
                })
                .collect(),
            lf_rows: 0,
            lf_busy: false,
            next_cdef: 0,
            cdef_done: vec![false; g.rows],
            cdef_rows: 0,
            next_stripe: 0,
            stripes_done: 0,
            spare: Vec::new(),
        }
    }

    /// The rows of `CurrFrame`, from the first on, that the stages before
    /// CDEF are done with: reconstructed and, in a frame that is
    /// deblocked, deblocked. (The loop filter of row `r` leaves the last
    /// `LF_REACH` sample rows of row `r` to that of row `r + 1`.)
    fn cur_rows(&self) -> usize {
        if self.g.lf {
            self.lf_rows
        } else {
            self.recon_rows
        }
    }

    /// The rows, from the first on, of the frame loop restoration starts
    /// from (`CdefFrame`, or `CurrFrame` for a frame without CDEF) that
    /// are final to their last sample row.
    fn cdef_final_rows(&self) -> usize {
        if self.g.cdef {
            self.cdef_rows
        } else if self.cur_rows() == self.g.rows {
            self.g.rows
        } else {
            self.cur_rows().saturating_sub(1)
        }
    }

    // The readiness of each stage's next task. A stage's tasks become
    // ready in order, so the next one says it for the stage.

    /// The stripe's rows through CDEF. (The two deblocked rows it reads
    /// below them lie in the same superblock row: 64 s + 56 is not a
    /// multiple of 64.)
    fn stripe_ready(&self, s: usize) -> bool {
        s < self.g.stripes && self.cdef_final_rows() >= self.g.stripe_bands(s).end
    }

    /// The row deblocked to its last sample row, and the first rows of the
    /// row below it.
    fn cdef_ready(&self, r: usize) -> bool {
        self.g.cdef && r < self.g.rows && self.cur_rows() >= (r + 2).min(self.g.rows)
    }

    fn deblock_ready(&self) -> bool {
        self.g.lf && !self.lf_busy && self.lf_rows < self.recon_rows
    }

    /// A task whose rows are ready, taken: the stage furthest along first,
    /// which keeps the rows in flight few. `references` says whether the
    /// frame's reference frames are complete: no row is predicted before.
    fn take(&mut self, references: bool) -> Option<Task> {
        if self.stripe_ready(self.next_stripe) {
            self.next_stripe += 1;
            return Some(Task::Restore(self.next_stripe - 1));
        }
        if self.cdef_ready(self.next_cdef) {
            self.next_cdef += 1;
            return Some(Task::Cdef(self.next_cdef - 1));
        }
        if self.deblock_ready() {
            self.lf_busy = true;
            return Some(Task::Deblock(self.lf_rows));
        }
        if references && self.next_recon < self.parsed_rows {
            self.next_recon += 1;
            return Some(Task::Reconstruct(self.next_recon - 1));
        }
        None
    }

    /// How many tasks [`Self::take`] would give before anything changes.
    fn ready(&self, references: bool) -> usize {
        let mut n = 0;
        let mut s = self.next_stripe;
        while self.stripe_ready(s) {
            n += 1;
            s += 1;
        }
        let mut r = self.next_cdef;
        while self.cdef_ready(r) {
            n += 1;
            r += 1;
        }
        n += self.deblock_ready() as usize;
        if references {
            n += self.parsed_rows - self.next_recon;
        }
        n
    }

    /// Records that `task` is done (and has given its rows back).
    fn finish(&mut self, task: Task) {
        let g = &self.g;
        match task {
            Task::Parse => self.parse_done = true,
            Task::Reconstruct(r) => {
                self.recon_done[r] = true;
                while self.recon_rows < g.rows && self.recon_done[self.recon_rows] {
                    self.recon_rows += 1;
                }
            }
            Task::Deblock(r) => {
                debug_assert!(self.lf_busy && self.lf_rows == r);
                self.lf_busy = false;
                self.lf_rows = r + 1;
            }
            Task::Cdef(r) => {
                self.cdef_done[r] = true;
                while self.cdef_rows < g.rows && self.cdef_done[self.cdef_rows] {
                    self.cdef_rows += 1;
                }
            }
            Task::Restore(_) => self.stripes_done += 1,
        }
    }

    /// Whether every task of the frame is done.
    fn complete(&self) -> bool {
        let g = &self.g;
        self.parse_done
            && self.recon_rows == g.rows
            && (!g.lf || self.lf_rows == g.rows)
            && (!g.cdef || self.cdef_rows == g.rows)
            && self.stripes_done == g.stripes
    }
}

/// One of a frame's sample buffers, lent out by rows.
struct Planes {
    /// Each plane's lender, stride, width and height.
    p: Vec<(Lender<u16>, usize, usize, usize)>,
}

impl Planes {
    /// Takes the samples of `buf`.
    fn take(buf: FrameBuf) -> Self {
        Planes {
            p: buf
                .planes
                .into_iter()
                .map(|p| (p.data.lender(), p.stride, p.w, p.h))
                .collect(),
        }
    }

    /// The buffer whole again: every window must have been dropped.
    fn whole(&self) -> FrameBuf {
        FrameBuf {
            planes: self
                .p
                .iter()
                .map(|(lender, stride, w, h)| PlaneBuf {
                    data: lender.whole(),
                    stride: *stride,
                    w: *w,
                    h: *h,
                })
                .collect(),
        }
    }

    /// A buffer of the same planes with no samples, for a thread to put
    /// windows in.
    fn shells(&self) -> FrameBuf {
        FrameBuf {
            planes: self
                .p
                .iter()
                .map(|&(_, stride, w, h)| PlaneBuf {
                    data: Shared::default(),
                    stride,
                    w,
                    h,
                })
                .collect(),
        }
    }

    fn height(&self, plane: usize) -> usize {
        self.p[plane].3
    }

    /// The window of plane `plane` that reads the sample rows before
    /// `read_to` and reads and writes the rows `write` (both clamped to
    /// the plane).
    #[track_caller]
    fn rows(&self, plane: usize, read_to: usize, write: Range<usize>) -> Shared<u16> {
        let (lender, stride, _, h) = &self.p[plane];
        lender.window(
            read_to.min(*h) * stride,
            write.start.min(*h) * stride..write.end.min(*h) * stride,
        )
    }
}

/// What a frame's parsing task starts from: its headers and its tile, as
/// the decoder had them when the frame was submitted.
pub(crate) struct Input {
    pub(crate) seq: Arc<SequenceHeader>,
    pub(crate) hdr: FrameHeader,
    /// The tile's data.
    pub(crate) data: Vec<u8>,
    pub(crate) coeff_reader: CoefficientReader,
    /// `RefOrderHint` when the header was parsed (7.19 compares with it).
    pub(crate) ref_order_hints: [u32; 8],
    /// Threads for the post-filters of a frame that is filtered whole.
    pub(crate) threads: usize,
    /// Where the frame's sample planes come from.
    pub(crate) pool: Arc<Pool>,
}

/// A frame's arrays once its parsing task has set them up, lent out by
/// rows.
struct Pipe {
    g: Geometry,
    /// The frame's parameters, for each thread to make its own state from.
    template: FrameCtx,
    levels: LfLevels,
    mi: Lender<Mi>,
    lf_tx_sizes: Vec<Lender<u8>>,
    /// `CurrFrame`, `CdefFrame` (if the frame has CDEF) and `LrFrame` (the
    /// planes that are restored; the others have no samples).
    cur: Planes,
    cdef: Option<Planes>,
    lr: Option<Planes>,
    threads: usize,
}

/// A frame in flight.
pub(crate) struct Job {
    id: u64,
    /// The reference slots as the frame's header found them, and of those
    /// the frames whose saved state its parsing reads (`primary_ref_frame`
    /// and, for the motion field, the references) and whose samples its
    /// prediction reads.
    refs: [Option<Arc<RefData>>; 8],
    parse_needs: Vec<Arc<RefData>>,
    predict_needs: Vec<Arc<RefData>>,
    /// This frame as a reference: its saved state is set when it is
    /// parsed, its samples when it is complete.
    pub(crate) out: Arc<RefData>,
    /// Its luma samples: what its buffers are in proportion to.
    samples: u64,
    input: Mutex<Option<Input>>,
    pipe: OnceLock<Pipe>,
}

impl Job {
    /// The frame of `input`, whose header named the references `refs`
    /// (checked: `Decoder::check_frame`) and which will be the reference
    /// `out`.
    pub(crate) fn new(refs: &[Option<Arc<RefData>>; 8], out: Arc<RefData>, input: Input) -> Self {
        let h = &input.hdr;
        let mut predict_needs: Vec<Arc<RefData>> = Vec::new();
        if !h.frame_is_intra {
            for &idx in &h.ref_frame_idx[..REFS_PER_FRAME] {
                let r = refs[idx].as_ref().expect("checked with the header");
                if !predict_needs.iter().any(|p| Arc::ptr_eq(p, r)) {
                    predict_needs.push(r.clone());
                }
            }
        }
        let mut parse_needs = predict_needs.clone();
        if h.primary_ref_frame != PRIMARY_REF_NONE {
            let r = refs[h.ref_frame_idx[h.primary_ref_frame]]
                .as_ref()
                .expect("checked with the header");
            if !parse_needs.iter().any(|p| Arc::ptr_eq(p, r)) {
                parse_needs.push(r.clone());
            }
        }
        Job {
            id: 0,
            refs: refs.clone(),
            parse_needs,
            predict_needs,
            samples: h.upscaled_width as u64 * h.frame_height as u64,
            out,
            input: Mutex::new(Some(input)),
            pipe: OnceLock::new(),
        }
    }

    fn pipe(&self) -> &Pipe {
        self.pipe.get().expect("a frame whose parsing has begun")
    }

    /// The mode info of the superblock rows before `r1`, to read.
    #[track_caller]
    fn mi_rows(&self, r1: usize) -> Shared<Mi> {
        let p = self.pipe();
        p.mi.window(p.g.units(0, r1, 0, p.mi.len()).end, 0..0)
    }
}

/// Whether a frame can be a [`Job`]: a frame of one tile that does not
/// predict from its own samples beyond a block's edge. (Intra block copy
/// reads any part of the frame decoded so far, 7.11.3.2 with
/// `use_intrabc`: its rows cannot be reconstructed apart.)
pub(crate) fn applies(h: &FrameHeader) -> bool {
    h.tile_info.cols == 1 && h.tile_info.rows == 1 && !h.allow_intrabc
}

/// What the engine keeps of a frame in flight.
struct Entry {
    id: u64,
    job: Arc<Job>,
    /// The parsing task has been taken.
    parse_taken: bool,
    /// Set by the parsing task once it has set the frame up.
    rows: Option<Rows>,
    /// The frame's references are complete (kept once seen).
    references: bool,
    /// Tasks being carried out.
    running: usize,
    /// The frame is abandoned: its parsing failed, a thread panicked, or
    /// the decoder gave it up. Nothing more of it is started.
    abort: bool,
    /// Whether the tile's padding checked out.
    tiles_ok: bool,
    /// Every task is done and the frame's samples are being put together;
    /// then, that they are: the frame is complete.
    finishing: bool,
    done: bool,
}

impl Entry {
    /// Nothing more will happen to the frame.
    fn settled(&self) -> bool {
        self.done || (self.abort && self.running == 0)
    }

    fn parse_ready(&self) -> bool {
        !self.parse_taken
            && !self.abort
            && self
                .job
                .parse_needs
                .iter()
                .all(|r| r.syntax.get().is_some())
    }

    fn references(&mut self) -> bool {
        if !self.references {
            self.references = self
                .job
                .predict_needs
                .iter()
                .all(|r| r.frame.get().is_some());
        }
        self.references
    }

    fn take(&mut self) -> Option<Task> {
        if self.abort || self.done {
            return None;
        }
        let references = self.references();
        if let Some(rows) = &mut self.rows {
            return rows.take(references);
        }
        if self.parse_ready() {
            self.parse_taken = true;
            return Some(Task::Parse);
        }
        None
    }

    fn ready(&mut self) -> usize {
        if self.abort || self.done {
            return 0;
        }
        let references = self.references();
        match &self.rows {
            Some(rows) => rows.ready(references),
            None => self.parse_ready() as usize,
        }
    }
}

/// What the threads of an engine keep under one lock.
#[derive(Default)]
struct Global {
    /// The frames in flight, oldest first.
    jobs: VecDeque<Entry>,
    next_id: u64,
    /// The threads waiting for a task, the one that parked last at the
    /// end: it is woken first (its caches are the warmest).
    parked: Vec<usize>,
    /// By thread: whether it has been woken (and is no longer in
    /// `parked`), and what for.
    woken: Vec<Woken>,
    /// Threads woken for a task that have not taken one yet.
    waking: usize,
    /// Threads started to take tasks while the decoder's caller is away
    /// are doing so: a session. Its job (what was given to
    /// `parallel::start`, to ask for more threads with), its number, how
    /// many threads were promised for it and how many are in it.
    session: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    session_number: u64,
    session_promised: usize,
    session_threads: usize,
    /// The frames that are settled and gone from `jobs`, until the decoder
    /// asks: whether the tile's padding checked out, or `None` for a frame
    /// that failed or was abandoned.
    settled: Vec<(u64, Option<bool>)>,
    /// A frame failed, this one: the frames after it were abandoned with
    /// it (they were submitted as if it had decoded) and those submitted
    /// since are too, until the decoder has seen to it
    /// (`Engine::abandon_from`).
    failed: Option<u64>,
}

impl Global {
    fn entry(&mut self, id: u64) -> &mut Entry {
        self.jobs
            .iter_mut()
            .find(|e| e.id == id)
            .expect("a frame in flight")
    }

    /// A task that is ready, taken: of the oldest frame that has one.
    fn take(&mut self) -> Option<(Arc<Job>, Task)> {
        for e in self.jobs.iter_mut() {
            if let Some(t) = e.take() {
                e.running += 1;
                return Some((e.job.clone(), t));
            }
        }
        None
    }

    fn ready(&mut self) -> usize {
        self.jobs.iter_mut().map(Entry::ready).sum()
    }
}

/// Why a waiting thread was woken.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Woken {
    /// It has not been.
    No,
    /// A task is ready for it (unless another thread gets there first).
    Task,
    /// A frame is settled, or the engine ends.
    News,
}

/// When a thread stops taking tasks.
#[derive(Clone, Copy)]
enum Until {
    /// Frame `id` is settled.
    Settled(u64),
    /// There is room for one more frame in flight: fewer than `frames`
    /// are, and with it their luma samples would not exceed `samples`
    /// (or none is in flight at all).
    Room {
        frames: usize,
        samples: u64,
        more: u64,
    },
    /// The session with this number is over (`Engine::end_session`).
    SessionOver(u64),
}

struct Inner {
    global: Mutex<Global>,
    /// By thread: where it waits for a task.
    park: Vec<Condvar>,
    /// Where a row waits for the row above.
    wave: Condvar,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, Global> {
        // A thread that panicked marks its frame abandoned on its way out
        // (`Bail`); the state it leaves is still what the others need.
        self.global.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wakes one thread for each task that is ready and that no thread
    /// already woken is on its way to.
    fn wake_for_tasks(&self, g: &mut Global) {
        let ready = g.ready();
        while g.waking < ready {
            let Some(w) = g.parked.pop() else {
                break;
            };
            g.woken[w] = Woken::Task;
            g.waking += 1;
            self.park[w].notify_one();
        }
    }

    /// Takes a frame to which nothing more will happen out of flight,
    /// leaving its outcome for the decoder, and tells whoever may be
    /// waiting for it.
    fn settle(&self, g: &mut Global, id: u64) {
        if let Some(at) = g.jobs.iter().position(|e| e.id == id && e.settled()) {
            let e = g.jobs.remove(at).expect("the frame found");
            g.settled.push((id, e.done.then_some(e.tiles_ok)));
            self.wake_all(g);
        }
    }

    /// Frame `id` failed: no frame after it can be what it was submitted
    /// as. They are abandoned, so that nobody waits for a frame that waits
    /// for this one.
    fn fail(&self, g: &mut Global, id: u64) {
        g.failed = Some(g.failed.map_or(id, |f| f.min(id)));
        let mut idle = Vec::new();
        for e in g.jobs.iter_mut().filter(|e| e.id >= id) {
            e.abort = true;
            if e.running == 0 {
                idle.push(e.id);
            }
        }
        for i in idle {
            self.settle(g, i);
        }
        self.wake_all(g);
    }

    /// Wakes every waiting thread: something they may be waiting for other
    /// than a task has happened (a frame settled, or the engine ends).
    fn wake_all(&self, g: &mut Global) {
        for w in std::mem::take(&mut g.parked) {
            g.woken[w] = Woken::News;
            self.park[w].notify_one();
        }
        self.wave.notify_all();
    }
}

/// The frames in flight and the threads' way to them.
pub(crate) struct Engine {
    inner: Arc<Inner>,
}

impl Default for Engine {
    fn default() -> Self {
        Engine {
            inner: Arc::new(Inner {
                global: Mutex::new(Global {
                    woken: vec![Woken::No; MAX_WORKERS],
                    ..Global::default()
                }),
                park: (0..MAX_WORKERS).map(|_| Condvar::new()).collect(),
                wave: Condvar::new(),
            }),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.abandon_from(0);
        self.end_session();
    }
}

impl Engine {
    /// Makes sure threads are taking tasks while the calling thread is
    /// away, up to `threads - 1` of them ([`crate::parallel::start`]):
    /// a *session*, which lasts until [`Engine::end_session`]. Asked again
    /// during a session, asks for the threads still missing. Returns
    /// whether there is a session.
    pub(crate) fn session(&self, threads: usize) -> bool {
        let n = threads.clamp(1, MAX_WORKERS);
        let mut g = self.inner.lock();
        if g.session.is_some() && g.session_promised + 1 >= n {
            return true;
        }
        let job = match &g.session {
            Some(job) => job.clone(),
            None => {
                if g.session_threads > 0 {
                    // The last session's threads are still on their way
                    // out.
                    return false;
                }
                g.session_number += 1;
                let (inner, number) = (self.inner.clone(), g.session_number);
                let job: Arc<dyn Fn(usize) + Send + Sync> =
                    Arc::new(move |k| inner.work(k, Until::SessionOver(number)));
                g.session = Some(job.clone());
                g.session_promised = 0;
                job
            }
        };
        // (Whatever lock `start` takes is taken inside this one, here and
        // nowhere the other way round.)
        g.session_promised += crate::parallel::start(n, job);
        if g.session_promised == 0 {
            g.session = None;
        }
        g.session.is_some()
    }

    /// Ends the session, if there is one: when this returns its threads
    /// have stopped taking tasks (the frames in flight stay, for the
    /// calling thread or the next session).
    pub(crate) fn end_session(&self) {
        let inner = &self.inner;
        let mut g = inner.lock();
        g.session = None;
        g.session_promised = 0;
        inner.wake_all(&mut g);
        while g.session_threads > 0 {
            g = inner.park[0].wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Puts a frame in flight; returns its number.
    pub(crate) fn submit(&self, mut job: Job) -> u64 {
        let mut g = self.inner.lock();
        let id = g.next_id;
        g.next_id += 1;
        job.id = id;
        if g.failed.is_some() {
            // Submitted as if a frame that failed had decoded: abandoned
            // on arrival.
            g.settled.push((id, None));
            return id;
        }
        g.jobs.push_back(Entry {
            id,
            job: Arc::new(job),
            parse_taken: false,
            rows: None,
            references: false,
            running: 0,
            abort: false,
            tiles_ok: true,
            finishing: false,
            done: false,
        });
        self.inner.wake_for_tasks(&mut g);
        id
    }

    /// Decodes a frame on up to `threads` threads, the caller's among
    /// them, and returns when it is settled: whether the tile's padding
    /// checked out, or `None` if the tile did not parse.
    pub(crate) fn decode(&self, job: Job, threads: usize) -> Option<bool> {
        let id = self.submit(job);
        let inner = &self.inner;
        if inner.lock().session.is_some() {
            // A session's threads are at it already.
            inner.work(0, Until::Settled(id));
        } else {
            crate::parallel::run(threads.clamp(1, MAX_WORKERS), &|k| {
                inner.work(k, Until::Settled(id));
            });
        }
        let mut g = inner.lock();
        if g.failed == Some(id) {
            // Seen to by the caller, which decodes the frame again.
            g.failed = None;
        }
        drop(g);
        self.outcome(id).expect("a settled frame's outcome")
    }

    /// Whether frame `id` is settled, and then how: whether its tile's
    /// padding checked out, or `None` for a frame whose tile did not
    /// parse. Asked once: the answer is not kept.
    pub(crate) fn outcome(&self, id: u64) -> Option<Option<bool>> {
        let mut g = self.inner.lock();
        let at = g.settled.iter().position(|s| s.0 == id)?;
        Some(g.settled.swap_remove(at).1)
    }

    /// Whether frame `id` is settled (its outcome stays to be asked for).
    pub(crate) fn settled(&self, id: u64) -> Option<Option<bool>> {
        let g = self.inner.lock();
        g.settled.iter().find(|s| s.0 == id).map(|s| s.1)
    }

    /// Takes tasks, as the calling thread, until frame `id` is settled.
    pub(crate) fn wait_for(&self, id: u64) {
        self.inner.work(0, Until::Settled(id));
    }

    /// Takes tasks, as the calling thread, until there is room for a frame
    /// of `more` luma samples in flight: fewer than `frames` are in
    /// flight, and with it they would have no more than `samples` luma
    /// samples between them (or none is in flight).
    pub(crate) fn wait_for_room(&self, frames: usize, samples: u64, more: u64) {
        self.inner.work(
            0,
            Until::Room {
                frames: frames.max(1),
                samples,
                more,
            },
        );
    }

    /// Abandons frame `id` and every frame submitted after it: when this
    /// returns no thread is working on any of them and they are forgotten.
    pub(crate) fn abandon_from(&self, id: u64) {
        let inner = &self.inner;
        let mut g = inner.lock();
        for e in g.jobs.iter_mut().filter(|e| e.id >= id) {
            e.abort = true;
        }
        inner.wake_all(&mut g);
        loop {
            // The ones nobody is working on go at once; the others when
            // their tasks end (`Inner::work` settles them).
            let idle: Vec<u64> = g
                .jobs
                .iter()
                .filter(|e| e.id >= id && e.running == 0)
                .map(|e| e.id)
                .collect();
            for i in idle {
                inner.settle(&mut g, i);
            }
            if !g.jobs.iter().any(|e| e.id >= id) {
                break;
            }
            g.woken[0] = Woken::No;
            g.parked.push(0);
            while g.woken[0] == Woken::No {
                g = inner.park[0].wait(g).unwrap_or_else(|e| e.into_inner());
            }
            if g.woken[0] == Woken::Task {
                g.waking -= 1;
            }
        }
        g.settled.retain(|s| s.0 < id);
        if g.failed.is_some_and(|f| f >= id) {
            g.failed = None;
        }
    }
}

/// Marks a frame abandoned if the thread working on it unwinds, so that no
/// other thread waits for what this one was doing.
struct Bail<'a> {
    inner: &'a Inner,
    id: u64,
}

impl Drop for Bail<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let mut g = self.inner.lock();
            if let Some(e) = g.jobs.iter_mut().find(|e| e.id == self.id) {
                e.abort = true;
                e.running -= 1;
            }
            self.inner.settle(&mut g, self.id);
            self.inner.wake_all(&mut g);
        }
    }
}

/// A thread's own state between tasks: for each frame it has worked on, a
/// frame state without arrays (given windows task by task) and shells for
/// the filters' frames; and buffers that outlive a task.
#[derive(Default)]
struct Local {
    frames: Vec<(u64, FrameCtx, Scratch)>,
    above_line: [Vec<u16>; 3],
    pred_bufs: [Vec<i32>; 2],
}

/// A thread's buffers for the frames the filters write: the planes'
/// shapes, and the windows of the task in hand.
struct Scratch {
    cdef: FrameBuf,
    lr: FrameBuf,
}

impl Local {
    /// This thread's state for frame `job`, made at its first task there.
    /// (States of frames long gone are dropped as new ones come.)
    fn frame(&mut self, job: &Job) -> usize {
        if let Some(i) = self.frames.iter().position(|f| f.0 == job.id) {
            return i;
        }
        if self.frames.len() >= 8 {
            self.frames.remove(0);
        }
        let p = job.pipe();
        let scratch = Scratch {
            cdef: p.cdef.as_ref().map(Planes::shells).unwrap_or_default(),
            lr: p.lr.as_ref().map(Planes::shells).unwrap_or_default(),
        };
        self.frames.push((job.id, p.template.view(), scratch));
        self.frames.len() - 1
    }
}

impl Inner {
    /// Carries out tasks until `until`, as thread `me`.
    fn work(&self, me: usize, until: Until) {
        let mut local = Local::default();
        let mut g = self.lock();
        if let Until::SessionOver(n) = until {
            // A thread that comes up when its session is over leaves at
            // once: it counts for nothing.
            if g.session.is_none() || g.session_number != n {
                return;
            }
            g.session_threads += 1;
        }
        loop {
            // A task, or a wait for one.
            let (job, task) = loop {
                // (This thread's states of frames no longer in flight go:
                // they hold those frames' references.)
                local.frames.retain(|f| g.jobs.iter().any(|e| e.id == f.0));
                let over = match until {
                    Until::Settled(id) => !g.jobs.iter().any(|e| e.id == id),
                    Until::Room {
                        frames,
                        samples,
                        more,
                    } => {
                        g.jobs.is_empty()
                            || (g.jobs.len() < frames
                                && g.jobs.iter().map(|e| e.job.samples).sum::<u64>() + more
                                    <= samples)
                    }
                    Until::SessionOver(n) => g.session.is_none() || g.session_number != n,
                };
                if over {
                    if matches!(until, Until::SessionOver(_)) {
                        g.session_threads -= 1;
                        self.park[0].notify_one();
                    }
                    return;
                }
                if let Some(t) = g.take() {
                    // More may be ready than this thread can take.
                    self.wake_for_tasks(&mut g);
                    break t;
                }
                g.woken[me] = Woken::No;
                g.parked.push(me);
                while g.woken[me] == Woken::No {
                    g = self.park[me].wait(g).unwrap_or_else(|e| e.into_inner());
                }
                // Woken for a task (which may have gone to another thread
                // since) or for news.
                if g.woken[me] == Woken::Task {
                    g.waking -= 1;
                }
            };
            drop(g);
            let bail = Bail {
                inner: self,
                id: job.id,
            };
            let outcome = match task {
                Task::Parse => self.parse(&job),
                Task::Reconstruct(r) => {
                    let i = local.frame(&job);
                    self.reconstruct(&job, &mut local, i, r);
                    Ok(())
                }
                Task::Deblock(r) => {
                    let i = local.frame(&job);
                    deblock(&job, &mut local.frames[i].1, r);
                    Ok(())
                }
                Task::Cdef(r) => {
                    let i = local.frame(&job);
                    let (_, f, scratch) = &mut local.frames[i];
                    self.cdef(&job, f, scratch, r);
                    Ok(())
                }
                Task::Restore(s) => {
                    let i = local.frame(&job);
                    let (_, f, scratch) = &mut local.frames[i];
                    self.restore(&job, f, scratch, s);
                    Ok(())
                }
            };
            std::mem::forget(bail);
            // The task gave its rows back; from here they are whoever's
            // comes next.
            g = self.lock();
            let e = g.entry(job.id);
            let failed = outcome.is_err();
            if !failed && let Some(rows) = &mut e.rows {
                rows.finish(task);
            }
            let complete =
                !failed && !e.abort && !e.finishing && e.rows.as_ref().is_some_and(Rows::complete);
            if complete {
                // The frame's samples are put together outside the lock,
                // by this thread, which still counts as working on it.
                e.finishing = true;
                let rows = e.rows.take().expect("a complete frame's rows");
                drop(g);
                finish(&job, rows);
                g = self.lock();
                g.entry(job.id).done = true;
            }
            g.entry(job.id).running -= 1;
            if failed {
                // The tile does not parse. The decoder decodes the frame
                // again as one thread does, which reports it.
                self.fail(&mut g, job.id);
            }
            self.settle(&mut g, job.id);
            self.wake_for_tasks(&mut g);
        }
    }

    /// The parsing task: sets the frame up, parses the tile's superblock
    /// rows in order and publishes each, then leaves the frame's saved
    /// state for the frames that refer to it.
    fn parse(&self, job: &Arc<Job>) -> crate::Result<()> {
        let input = job
            .input
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("a frame is parsed once");
        let Input {
            seq,
            hdr,
            data,
            coeff_reader,
            ref_order_hints,
            threads,
            pool,
        } = input;
        let mut f = crate::decoder::build_frame(&job.refs, seq, hdr, &pool);
        let g = Geometry::of(&f);
        let template = f.view();
        // The frames the filters write: all of `CdefFrame` (CDEF copies
        // what it does not filter), and of `LrFrame` the planes that are
        // restored.
        let cdef = g.cdef.then(|| FrameBuf {
            planes: f
                .cur
                .planes
                .iter()
                .map(|p| pool.plane(p.w, p.h, 0))
                .collect(),
        });
        let lr = (g.stripes > 0).then(|| FrameBuf {
            planes: f
                .cur
                .planes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let h = if g.restored[i] { p.h } else { 0 };
                    pool.plane(p.w, h, 0)
                })
                .collect(),
        });
        let rows = Rows::new(g, &f);
        // The samples are the tasks' from here on; the mode info and the
        // transform sizes stay with this thread, which gives up each row
        // as it completes it.
        let pipe = Pipe {
            g,
            template,
            levels: LfLevels::new(&f.hdr),
            mi: f.mi.lender(),
            lf_tx_sizes: f.lf_tx_sizes[..g.num_planes]
                .iter()
                .map(Shared::lender)
                .collect(),
            cur: Planes::take(std::mem::take(&mut f.cur)),
            cdef: cdef.map(Planes::take),
            lr: lr.map(Planes::take),
            threads,
        };
        let _ = job.pipe.set(pipe);
        self.lock().entry(job.id).rows = Some(rows);
        let mut td = TileDecoder::new(&mut f, &data, 0, 0);
        td.coeff_reader = coeff_reader;
        td.pass = Pass::Parse;
        td.begin_tile();
        for r in 0..g.rows {
            td.parse_sb_row(r * g.sb4)?;
            // The row's mode info is final: readable by all from here on,
            // written by nobody.
            let len = td.f.mi.len();
            td.f.mi.narrow(len, g.units_after(r, 0, len));
            for p in 0..g.num_planes {
                let a = &mut td.f.lf_tx_sizes[p];
                let len = a.len();
                a.narrow(len, g.units_after(r, g.ss(p).1, len));
            }
            let mut gl = self.lock();
            let e = gl.entry(job.id);
            if e.abort {
                // Given up by the decoder: the rest is not wanted.
                return Ok(());
            }
            let rows = e.rows.as_mut().expect("the frame's rows");
            publish_filter_parameters(&g, td.f, rows, r);
            let next = rows.spare.pop().unwrap_or_default();
            rows.parsed[r] = Some(std::mem::replace(&mut td.rec, next));
            rows.parsed_rows = r + 1;
            self.wake_for_tasks(&mut gl);
        }
        let tiles_ok = td.sd.trailing_ok();
        let cdf = td.cdf;
        // `frame_end_update_cdf()`: the frame's CDFs are its tile's.
        if !f.hdr.disable_frame_end_update_cdf {
            f.cdfs = cdf;
        }
        let syntax = crate::decoder::frame_syntax(&mut f, &ref_order_hints);
        let _ = job.out.syntax.set(syntax);
        self.lock().entry(job.id).tiles_ok = tiles_ok;
        Ok(())
    }
}

/// Copies the CDEF indices and loop restoration units that superblock row
/// `r` coded from the parsing thread's frame state to where the filtering
/// tasks find them.
fn publish_filter_parameters(g: &Geometry, f: &FrameCtx, rows: &mut Rows, r: usize) {
    if g.cdef || !g.filters {
        let idx = g.cdef_idx(r, f.cdef_idx.len());
        rows.cdef_idx[idx.clone()].copy_from_slice(&f.cdef_idx[idx]);
    }
    for plane in 0..g.num_planes {
        if f.hdr.frame_restoration_type[plane] == RESTORE_NONE {
            continue;
        }
        let (unit_rows, _) = crate::decoder::lr_units_of_sb(f, plane, r * g.sb4, 0, g.sb4);
        let (from, to) = (&f.lr[plane], &mut rows.lr[plane]);
        let units = unit_rows.start * from.unit_cols..unit_rows.end * from.unit_cols;
        if units.is_empty() {
            continue;
        }
        to.lr_type[units.clone()].copy_from_slice(&from.lr_type[units.clone()]);
        to.wiener[units.clone()].copy_from_slice(&from.wiener[units.clone()]);
        to.sgr_set[units.clone()].copy_from_slice(&from.sgr_set[units.clone()]);
        to.sgr_xqd[units.clone()].copy_from_slice(&from.sgr_xqd[units]);
    }
}

/// Puts a frame whose tasks are all done together: its samples, as the
/// reference later frames wait for.
fn finish(job: &Job, rows: Rows) {
    let p = job.pipe();
    let g = &p.g;
    // Every window is back: the arrays are whole again.
    let mut cur = p.cur.whole();
    let out = if g.filters {
        // `LrFrame`: each plane from the last stage that wrote it.
        let mut out = match &p.cdef {
            Some(c) => c.whole(),
            None => cur,
        };
        if let Some(lr) = &p.lr {
            for (i, (o, l)) in out.planes.iter_mut().zip(lr.whole().planes).enumerate() {
                if g.restored[i] {
                    *o = l;
                }
            }
        }
        out
    } else {
        // Super-resolution: the post-filters over the whole frame, as the
        // decode frame wrapup has them, on the frame state put back
        // together.
        let mut f = p.template.view();
        f.mi = p.mi.whole();
        for (a, lender) in f.lf_tx_sizes.iter_mut().zip(&p.lf_tx_sizes) {
            *a = lender.whole();
        }
        std::mem::swap(&mut f.cur, &mut cur);
        f.cdef_idx = rows.cdef_idx;
        f.lr = rows.lr;
        crate::decoder::filter_frame(&mut f, p.threads)
    };
    let _ = job.out.frame.set(out);
}

impl Inner {
    /// Reconstructs superblock row `r` from its record.
    fn reconstruct(&self, job: &Job, local: &mut Local, i: usize, r: usize) {
        let p = job.pipe();
        let g = &p.g;
        let rec = self
            .lock()
            .entry(job.id)
            .rows
            .as_mut()
            .expect("the frame's rows")
            .parsed[r]
            .take()
            .expect("a parsed row's record");
        let view = &mut local.frames[i].1;
        // The row's own samples, and the mode info up to it (prediction
        // looks at the blocks above and to the left: 7.11.2.8, 7.11.3.10).
        view.mi = job.mi_rows(r + 1);
        let mut td = TileDecoder::new(&mut *view, &[], 0, 0);
        td.pass = Pass::Recon;
        td.above_line = std::mem::take(&mut local.above_line);
        td.pred_bufs = std::mem::take(&mut local.pred_bufs);
        for pl in 0..g.num_planes {
            let band = g.band(pl, r, p.cur.height(pl));
            td.band_top[pl] = band.start;
            td.f.cur.planes[pl].data = p.cur.rows(pl, 0, band);
            td.above_line[pl].resize(td.f.cur.planes[pl].w, 0);
        }
        td.rec = rec;
        td.rewind();
        let done = self.reconstruct_superblocks(job, &mut td, r);
        local.above_line = std::mem::take(&mut td.above_line);
        local.pred_bufs = std::mem::take(&mut td.pred_bufs);
        let mut rec = std::mem::take(&mut td.rec);
        drop(td);
        view.mi = Shared::default();
        for pl in view.cur.planes.iter_mut() {
            pl.data = Shared::default();
        }
        if done {
            rec.clear();
            let mut gl = self.lock();
            if let Some(rows) = &mut gl.entry(job.id).rows {
                rows.spare.push(rec);
            }
        }
    }

    /// The superblocks of row `r`, each behind the row above. Returns
    /// `false` if the frame was abandoned meanwhile.
    fn reconstruct_superblocks(&self, job: &Job, td: &mut TileDecoder, r: usize) -> bool {
        let g = &job.pipe().g;
        let mi_row = r * g.sb4;
        // Superblocks of the row above whose last line is in `above_line`.
        let mut pulled = 0;
        for k in 0..g.cols {
            if r > 0 {
                // The row above must have finished the superblock above
                // and to the right; its line comes across under the same
                // lock.
                let need = (k + 2).min(g.cols);
                let mut gl = self.lock();
                loop {
                    let e = gl.entry(job.id);
                    if e.abort {
                        return false;
                    }
                    let rows = e.rows.as_ref().expect("the frame's rows");
                    if rows.progress[r - 1] >= need {
                        break;
                    }
                    gl = self.wave.wait(gl).unwrap_or_else(|e| e.into_inner());
                }
                let rows = gl.entry(job.id).rows.as_ref().expect("the frame's rows");
                while pulled < need {
                    for p in 0..g.num_planes {
                        let x = g.sb_columns(p, pulled);
                        td.above_line[p][x.clone()].copy_from_slice(&rows.lines[r - 1][p][x]);
                    }
                    pulled += 1;
                }
            }
            td.clear_block_decoded_flags(mi_row, k * g.sb4, g.sb4);
            for i in td.rec.sb_blocks(k) {
                // Reconstruction reports nothing: every error of a tile is
                // its syntax's.
                td.replay_block(i).expect("reconstruction does not fail");
            }
            let mut gl = self.lock();
            let e = gl.entry(job.id);
            let Some(rows) = e.rows.as_mut() else {
                return false;
            };
            if r + 1 < g.rows {
                for p in 0..g.num_planes {
                    let x = g.sb_columns(p, k);
                    let pl = &td.f.cur.planes[p];
                    let at = (g.band(p, r, pl.h).end - 1) * pl.stride;
                    rows.lines[r][p][x.clone()]
                        .copy_from_slice(pl.data.mine_range(at + x.start..at + x.end));
                }
            }
            rows.progress[r] = k + 1;
            drop(gl);
            self.wave.notify_all();
        }
        true
    }

    /// CDEF (7.15) of superblock row `r`: `CdefFrame`'s rows from
    /// `CurrFrame`'s.
    fn cdef(&self, job: &Job, f: &mut FrameCtx, scratch: &mut Scratch, r: usize) {
        let p = job.pipe();
        let g = &p.g;
        let planes = p.cdef.as_ref().expect("a frame with CDEF");
        {
            let mut gl = self.lock();
            let rows = gl.entry(job.id).rows.as_ref().expect("the frame's rows");
            let idx = g.cdef_idx(r, f.cdef_idx.len());
            f.cdef_idx[idx.clone()].copy_from_slice(&rows.cdef_idx[idx]);
        }
        // Whether a block is filtered is its skip flags (7.15.1).
        f.mi = job.mi_rows(r + 1);
        for pl in 0..g.num_planes {
            let band = g.band(pl, r, p.cur.height(pl));
            // (Everything above the row is final too: read "to" is as
            // good as "around", and cheaper to check.)
            f.cur.planes[pl].data = p.cur.rows(pl, band.end + CDEF_REACH, 0..0);
            // `CdefFrame` starts as a copy of `CurrFrame` (7.15).
            let out = &mut scratch.cdef.planes[pl];
            out.data = planes.rows(pl, 0, band.clone());
            let samples = band.start * out.stride..band.end * out.stride;
            out.data[samples.clone()].copy_from_slice(&f.cur.planes[pl].data[samples]);
        }
        let out: Vec<SharedPlane> = scratch
            .cdef
            .planes
            .iter_mut()
            .map(SharedPlane::new)
            .collect();
        let mi = g.mi_band(r);
        postfilter::cdef_rows(f, &out, mi.start, mi.end);
        f.mi = Shared::default();
        for pl in 0..g.num_planes {
            f.cur.planes[pl].data = Shared::default();
            scratch.cdef.planes[pl].data = Shared::default();
        }
    }

    /// Loop restoration (7.17) of stripe `s`: `LrFrame`'s rows, for the
    /// planes that are restored, from `CdefFrame`'s and (beyond the
    /// stripe) `CurrFrame`'s.
    fn restore(&self, job: &Job, f: &mut FrameCtx, scratch: &mut Scratch, s: usize) {
        let p = job.pipe();
        let g = &p.g;
        let planes = p.lr.as_ref().expect("a frame with loop restoration");
        for pl in 0..g.num_planes {
            if !g.restored[pl] {
                continue;
            }
            let Some((_, _, y0, y1, unit_row)) = postfilter::lr_stripe(f, pl, s) else {
                continue;
            };
            let (y0, y1) = (y0 as usize, y1 as usize + 1);
            {
                // The stripe's units: one row of them (7.17.1).
                let mut gl = self.lock();
                let rows = gl.entry(job.id).rows.as_ref().expect("the frame's rows");
                let (from, to) = (&rows.lr[pl], &mut f.lr[pl]);
                let units = unit_row * from.unit_cols..(unit_row + 1) * from.unit_cols;
                to.lr_type[units.clone()].copy_from_slice(&from.lr_type[units.clone()]);
                to.wiener[units.clone()].copy_from_slice(&from.wiener[units.clone()]);
                to.sgr_set[units.clone()].copy_from_slice(&from.sgr_set[units.clone()]);
                to.sgr_xqd[units.clone()].copy_from_slice(&from.sgr_xqd[units]);
            }
            f.cur.planes[pl].data = p.cur.rows(pl, y1 + LR_REACH, 0..0);
            if let Some(c) = &p.cdef {
                scratch.cdef.planes[pl].data = c.rows(pl, y1, 0..0);
            }
            let stride = f.cur.planes[pl].stride;
            // `LrFrame` starts as a copy of `CdefFrame` (7.17).
            let out = &mut scratch.lr.planes[pl];
            out.data = planes.rows(pl, 0, y0..y1);
            let samples = y0 * stride..y1 * stride;
            let from = if p.cdef.is_some() {
                &scratch.cdef
            } else {
                &f.cur
            };
            out.data[samples.clone()].copy_from_slice(&from.planes[pl].data[samples]);
            let out = SharedPlane::new(out);
            postfilter::restore_stripe(f, &f.cur, from, out, pl, s);
            f.cur.planes[pl].data = Shared::default();
            scratch.lr.planes[pl].data = Shared::default();
            if p.cdef.is_some() {
                scratch.cdef.planes[pl].data = Shared::default();
            }
        }
    }
}

/// The loop filter (7.14) of superblock row `r`: its vertical edges, then
/// its horizontal edges, the one along its top among them.
fn deblock(job: &Job, f: &mut FrameCtx, r: usize) {
    let p = job.pipe();
    let g = &p.g;
    // An edge is decided by the blocks and transforms on both its sides.
    f.mi = job.mi_rows(r + 1);
    for pl in 0..g.num_planes {
        let sizes = &p.lf_tx_sizes[pl];
        let parsed = g.units(0, r + 1, g.ss(pl).1, sizes.len()).end;
        f.lf_tx_sizes[pl] = sizes.window(parsed, 0..0);
        let band = g.band(pl, r, p.cur.height(pl));
        let rows = band.start.saturating_sub(LF_REACH[pl])..band.end;
        f.cur.planes[pl].data = p.cur.rows(pl, 0, rows);
    }
    let mi = g.mi_band(r);
    postfilter::loop_filter_rows(f, &p.levels, mi.start, mi.end);
    f.mi = Shared::default();
    for pl in 0..g.num_planes {
        f.lf_tx_sizes[pl] = Shared::default();
        f.cur.planes[pl].data = Shared::default();
    }
}

impl FrameCtx {
    /// A frame state with this frame's parameters and references and none
    /// of its shared arrays: what a thread working on rows of the frame
    /// works in, given the rows.
    fn view(&self) -> FrameCtx {
        FrameCtx {
            seq: self.seq.clone(),
            hdr: self.hdr.clone(),
            bit_depth: self.bit_depth,
            ssx: self.ssx,
            ssy: self.ssy,
            num_planes: self.num_planes,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            ms: self.ms,
            mi: Shared::default(),
            tx_types: Vec::new(),
            lf_tx_sizes: Default::default(),
            palette_colors: Default::default(),
            segment_ids: Vec::new(),
            prev_segment_ids: self.prev_segment_ids.clone(),
            cdef_idx: vec![-1; self.cdef_idx.len()],
            cdef_stride: self.cdef_stride,
            lr: self.lr.clone(),
            motion_field: self.motion_field.clone(),
            cur: FrameBuf {
                planes: self
                    .cur
                    .planes
                    .iter()
                    .map(|p| PlaneBuf {
                        data: Shared::default(),
                        stride: p.stride,
                        w: p.w,
                        h: p.h,
                    })
                    .collect(),
            },
            refs: self.refs.clone(),
            cdfs: self.cdfs.clone(),
            saved_cdfs: None,
            mask: Box::new([0; 128 * 128]),
            tile_num: 0,
            tiles_ok: true,
            filtered: None,
            pool: self.pool.clone(),
        }
    }
}
