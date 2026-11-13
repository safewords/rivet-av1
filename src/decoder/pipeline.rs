//! A one-tile frame on several threads: its superblock rows reconstructed
//! behind the parsing of the tile, and filtered behind their
//! reconstruction.
//!
//! `docs/threading.md` derives the rules from the specification; this is
//! their bookkeeping. The thread that called the decoder parses the tile's
//! superblock rows in order ([`Pass::Parse`]) and hands each row's record
//! on as it completes. Every thread (that one too, once it has parsed the
//! last row) takes *tasks*, each a stage of one row:
//!
//! - **reconstruct** row `r` from its record ([`Pass::Recon`]), a
//!   superblock at a time behind the row above;
//! - **deblock** row `r` (7.14), once it and the rows before it are
//!   reconstructed and the rows before it deblocked;
//! - **CDEF** row `r` (7.15), once the rows it reads are deblocked;
//! - **restore** stripe `s` (7.17), once the rows it reads have been
//!   through CDEF.
//!
//! A task holds the rows it works on as windows of the frame's arrays
//! ([`Shared`]): no two tasks that run at the same time are given rows of
//! which either writes what the other touches, and the windows check that
//! they were not.

use std::ops::Range;
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::Result;
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::decoder::postfilter::{self, LfLevels, SharedPlane};
use crate::decoder::record::{Pass, RowRecord};
use crate::decoder::shared::{Lender, Shared};
use crate::decoder::state::{FrameBuf, Mi, PlaneBuf};
use crate::decoder::tile::TileDecoder;
use crate::decoder::{CoefficientReader, FrameCtx, LrPlane};

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
    /// restoration, 7.16: such a frame is filtered whole, afterwards.)
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

/// A stage of a row, for a thread to carry out.
#[derive(Clone, Copy, Debug)]
enum Task {
    Reconstruct(usize),
    Deblock(usize),
    Cdef(usize),
    Restore(usize),
}

/// What the threads of a frame keep under one lock.
struct State {
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
    /// The next stripe to restore.
    next_stripe: usize,
    /// Tasks being carried out.
    running: usize,
    /// Records to write the next rows into.
    spare: Vec<RowRecord>,
    /// Parsing failed, or a thread panicked: nothing more is started and
    /// nobody waits.
    abort: bool,
}

impl State {
    /// The rows of `CurrFrame`, from the first on, that the stages before
    /// CDEF are done with: reconstructed and, in a frame that is
    /// deblocked, deblocked. (The loop filter of row `r` leaves the last
    /// `LF_REACH` sample rows of row `r` to that of row `r + 1`.)
    fn cur_rows(&self, g: &Geometry) -> usize {
        if g.lf { self.lf_rows } else { self.recon_rows }
    }

    /// The rows, from the first on, of the frame loop restoration starts
    /// from (`CdefFrame`, or `CurrFrame` for a frame without CDEF) that
    /// are final to their last sample row.
    fn cdef_final_rows(&self, g: &Geometry) -> usize {
        if g.cdef {
            self.cdef_rows
        } else if self.cur_rows(g) == g.rows {
            g.rows
        } else {
            self.cur_rows(g).saturating_sub(1)
        }
    }

    /// A task whose rows are ready, taken: the stage furthest along first,
    /// which keeps the rows in flight few.
    fn take(&mut self, g: &Geometry) -> Option<Task> {
        if self.next_stripe < g.stripes {
            let s = self.next_stripe;
            // The stripe's rows through CDEF. (The two deblocked rows it
            // reads below them lie in the same superblock row: 64 s + 56
            // is not a multiple of 64.)
            if self.cdef_final_rows(g) >= g.stripe_bands(s).end {
                self.next_stripe += 1;
                return Some(Task::Restore(s));
            }
        }
        if g.cdef && self.next_cdef < g.rows {
            let r = self.next_cdef;
            // The row deblocked to its last sample row, and the first rows
            // of the row below it.
            if self.cur_rows(g) >= (r + 2).min(g.rows) {
                self.next_cdef += 1;
                return Some(Task::Cdef(r));
            }
        }
        if g.lf && !self.lf_busy && self.lf_rows < self.recon_rows {
            self.lf_busy = true;
            return Some(Task::Deblock(self.lf_rows));
        }
        if self.next_recon < self.parsed_rows {
            let r = self.next_recon;
            self.next_recon += 1;
            return Some(Task::Reconstruct(r));
        }
        None
    }

    /// Whether every task of the frame has been taken.
    fn all_taken(&self, g: &Geometry) -> bool {
        self.parse_done
            && self.next_recon == g.rows
            && (!g.lf || self.lf_rows + self.lf_busy as usize == g.rows)
            && (!g.cdef || self.next_cdef == g.rows)
            && self.next_stripe == g.stripes
    }

    /// Records that `task` is done (and has given its rows back).
    fn finish(&mut self, g: &Geometry, task: Task) {
        self.running -= 1;
        match task {
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
            Task::Restore(_) => {}
        }
    }
}

/// One of a frame's sample buffers, lent out by rows.
struct Planes {
    /// Each plane's lender, stride, width and height.
    p: Vec<(Lender<u16>, usize, usize, usize)>,
}

impl Planes {
    /// Takes the samples of `buf`, leaving it its planes' shapes.
    fn take(buf: &mut FrameBuf) -> Self {
        Planes {
            p: buf
                .planes
                .iter_mut()
                .map(|p| {
                    let lender = p.data.lender();
                    p.data = Shared::default();
                    (lender, p.stride, p.w, p.h)
                })
                .collect(),
        }
    }

    /// Puts the samples back into `buf`: every window must have been
    /// dropped.
    fn restore(self, buf: &mut FrameBuf) {
        for (p, (lender, ..)) in buf.planes.iter_mut().zip(self.p) {
            p.data = lender.whole();
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

    /// The window of plane `plane` that reads the sample rows `read` and
    /// writes `write` (both clamped to the plane).
    #[track_caller]
    fn rows(&self, plane: usize, read: Range<usize>, write: Range<usize>) -> Shared<u16> {
        let (lender, stride, _, h) = &self.p[plane];
        let clamp = |r: Range<usize>| r.start.min(*h) * stride..r.end.min(*h) * stride;
        lender.window(clamp(read), clamp(write))
    }
}

struct Frame<'a> {
    g: Geometry,
    state: Mutex<State>,
    /// Signalled on every change of `state` a thread may be waiting for.
    changed: Condvar,
    /// The frame's parameters, for each thread to make its own state from.
    template: &'a FrameCtx,
    levels: LfLevels,
    mi: Lender<Mi>,
    lf_tx_sizes: Vec<Lender<u8>>,
    /// `CurrFrame`, `CdefFrame` (if the frame has CDEF) and `LrFrame` (the
    /// planes that are restored; the others have no samples).
    cur: Planes,
    cdef: Option<Planes>,
    lr: Option<Planes>,
}

impl Frame<'_> {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A thread that panicked sets `abort` on its way out (`Bail`); the
        // state it leaves is still what the others need to stop.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'s>(&self, st: MutexGuard<'s, State>) -> MutexGuard<'s, State> {
        self.changed.wait(st).unwrap_or_else(|e| e.into_inner())
    }

    /// The mode info of superblock rows `r0..r1`, to read.
    #[track_caller]
    fn mi_rows(&self, r0: usize, r1: usize) -> Shared<Mi> {
        self.mi.window(self.g.units(r0, r1, 0, self.mi.len()), 0..0)
    }
}

/// Stops the frame's other threads if this one unwinds, so that none of
/// them waits for a row that will not come.
struct Bail<'a, 'b>(&'a Frame<'b>);

impl Drop for Bail<'_, '_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.lock().abort = true;
            self.0.changed.notify_all();
        }
    }
}

impl FrameCtx {
    /// A frame state with this frame's parameters and references and none
    /// of its shared arrays: what a thread working on rows of the frame
    /// works in, given the rows (see [`Frame`]).
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
        }
    }
}

/// Whether [`decode_tile`] decodes this frame's tile: a frame of one tile
/// that does not predict from its own samples beyond a block's edge.
/// (Intra block copy reads any part of the frame decoded so far, 7.11.3.2
/// with `use_intrabc`: its rows cannot be reconstructed apart.)
pub(crate) fn applies(f: &FrameCtx) -> bool {
    let ti = &f.hdr.tile_info;
    ti.cols == 1 && ti.rows == 1 && !f.hdr.allow_intrabc
}

/// Decodes the one tile of frame `f` (`data`) on up to `threads` threads:
/// what `TileDecoder::decode_tile` does and, unless the frame has
/// super-resolution, what the post-filters of the decode frame wrapup do
/// after it, whose result (`LrFrame`) is left in `f.filtered`. Returns
/// whether the tile's padding checked out and the tile's CDFs.
pub(crate) fn decode_tile(
    f: &mut FrameCtx,
    data: &[u8],
    threads: usize,
    coeff_reader: CoefficientReader,
) -> Result<(bool, Box<CdfContext>)> {
    let g = Geometry::of(f);
    let template = f.view();
    // The frames the filters write: all of `CdefFrame` (CDEF copies what
    // it does not filter), and of `LrFrame` the planes that are restored.
    let mut cdef_buf = g.cdef.then(|| FrameBuf {
        planes: f
            .cur
            .planes
            .iter()
            .map(|p| PlaneBuf::new(p.w, p.h, 0))
            .collect(),
    });
    let mut lr_buf = (g.stripes > 0).then(|| FrameBuf {
        planes: f
            .cur
            .planes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let h = if g.restored[i] { p.h } else { 0 };
                PlaneBuf::new(p.w, h, 0)
            })
            .collect(),
    });
    // The samples are the tasks' for the duration; the mode info and the
    // transform sizes stay with the parsing thread, which gives up each
    // row as it completes it.
    let frame = Frame {
        g,
        state: Mutex::new(State {
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
            running: 0,
            spare: Vec::new(),
            abort: false,
        }),
        changed: Condvar::new(),
        template: &template,
        levels: LfLevels::new(&f.hdr),
        mi: f.mi.lender(),
        lf_tx_sizes: f.lf_tx_sizes[..g.num_planes]
            .iter()
            .map(Shared::lender)
            .collect(),
        cur: Planes::take(&mut f.cur),
        cdef: cdef_buf.as_mut().map(Planes::take),
        lr: lr_buf.as_mut().map(Planes::take),
    };
    // The parsing walk, taken by the calling thread's job.
    type Parsed = Result<(bool, Box<CdfContext>)>;
    let mut td = TileDecoder::new(f, data, 0, 0);
    td.coeff_reader = coeff_reader;
    td.pass = Pass::Parse;
    let parser = Mutex::new(Some(td));
    let outcome: Mutex<Option<Parsed>> = Mutex::new(None);
    crate::parallel::run(threads.max(1), &|k| {
        let _bail = Bail(&frame);
        if k == 0 {
            let mut td = parser
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("the calling thread's job runs once");
            let r = parse(&frame, &mut td);
            let r = r.map(|()| (td.sd.trailing_ok(), td.cdf));
            *outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        }
        work(&frame);
    });
    let Frame {
        mi,
        lf_tx_sizes,
        cur,
        cdef,
        lr,
        ..
    } = frame;
    drop(template);
    // Every window is back: the arrays are whole again.
    f.mi = Shared::default();
    f.mi = mi.whole();
    for (a, lender) in f.lf_tx_sizes.iter_mut().zip(lf_tx_sizes) {
        *a = Shared::default();
        *a = lender.whole();
    }
    cur.restore(&mut f.cur);
    if let (Some(p), Some(buf)) = (cdef, cdef_buf.as_mut()) {
        p.restore(buf);
    }
    if let (Some(p), Some(buf)) = (lr, lr_buf.as_mut()) {
        p.restore(buf);
    }
    let outcome = outcome
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .expect("the calling thread's job ran");
    if outcome.is_ok() && g.filters {
        // `LrFrame`: each plane from the last stage that wrote it.
        let mut out = match cdef_buf {
            Some(c) => c,
            None => std::mem::take(&mut f.cur),
        };
        if let Some(lr) = lr_buf {
            for (p, (o, l)) in out.planes.iter_mut().zip(lr.planes).enumerate() {
                if g.restored[p] {
                    *o = l;
                }
            }
        }
        f.filtered = Some(out);
    }
    outcome
}

/// The calling thread's part: parses the tile's superblock rows in order
/// and publishes each.
fn parse(frame: &Frame, td: &mut TileDecoder) -> Result<()> {
    let g = &frame.g;
    td.begin_tile();
    for r in 0..g.rows {
        let parsed = td.parse_sb_row(r * g.sb4);
        if parsed.is_err() {
            frame.lock().abort = true;
            frame.changed.notify_all();
            return parsed;
        }
        // The row's mode info is final: readable by all from here on,
        // written by nobody.
        let len = td.f.mi.len();
        td.f.mi.narrow(0..len, g.units_after(r, 0, len));
        for p in 0..g.num_planes {
            let a = &mut td.f.lf_tx_sizes[p];
            let len = a.len();
            a.narrow(0..len, g.units_after(r, g.ss(p).1, len));
        }
        let mut st = frame.lock();
        publish_filter_parameters(g, td.f, &mut st, r);
        let next = st.spare.pop().unwrap_or_default();
        st.parsed[r] = Some(std::mem::replace(&mut td.rec, next));
        st.parsed_rows = r + 1;
        drop(st);
        frame.changed.notify_all();
    }
    frame.lock().parse_done = true;
    frame.changed.notify_all();
    Ok(())
}

/// Copies the CDEF indices and loop restoration units that superblock row
/// `r` coded from the parsing thread's frame state to where the filtering
/// tasks find them.
fn publish_filter_parameters(g: &Geometry, f: &FrameCtx, st: &mut State, r: usize) {
    if g.cdef {
        let rows = g.cdef_idx(r, f.cdef_idx.len());
        st.cdef_idx[rows.clone()].copy_from_slice(&f.cdef_idx[rows]);
    }
    for plane in 0..g.num_planes {
        if !g.restored[plane] {
            continue;
        }
        let (unit_rows, _) = crate::decoder::lr_units_of_sb(f, plane, r * g.sb4, 0, g.sb4);
        let (from, to) = (&f.lr[plane], &mut st.lr[plane]);
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

/// A thread's buffers for the frames the filters write: the planes'
/// shapes, and the windows of the task in hand.
struct Scratch {
    cdef: FrameBuf,
    lr: FrameBuf,
}

/// Every thread's part: carries out tasks until none is left to take.
fn work(frame: &Frame) {
    let g = &frame.g;
    let (st, first) = next_task(frame, frame.lock());
    drop(st);
    let Some(mut task) = first else {
        return;
    };
    // This thread's frame state and walker, made for its first task.
    let mut view = frame.template.view();
    let mut td = TileDecoder::new(&mut view, &[], 0, 0);
    td.pass = Pass::Recon;
    let mut scratch = Scratch {
        cdef: frame.cdef.as_ref().map(Planes::shells).unwrap_or_default(),
        lr: frame.lr.as_ref().map(Planes::shells).unwrap_or_default(),
    };
    loop {
        match task {
            Task::Reconstruct(r) => reconstruct(frame, &mut td, r),
            Task::Deblock(r) => deblock(frame, td.f, r),
            Task::Cdef(r) => cdef(frame, td.f, &mut scratch, r),
            Task::Restore(s) => restore(frame, td.f, &mut scratch, s),
        }
        // The task gave its rows back; from here they are whoever's comes
        // next.
        let mut st = frame.lock();
        st.finish(g, task);
        frame.changed.notify_all();
        let (st, next) = next_task(frame, st);
        drop(st);
        match next {
            Some(t) => task = t,
            None => return,
        }
    }
}

/// Takes the next task, waiting for one to be ready; `None` when every
/// task has been taken (or the frame was abandoned).
fn next_task<'s>(
    frame: &'s Frame,
    mut st: MutexGuard<'s, State>,
) -> (MutexGuard<'s, State>, Option<Task>) {
    let g = &frame.g;
    loop {
        if st.abort {
            return (st, None);
        }
        if let Some(t) = st.take(g) {
            st.running += 1;
            return (st, Some(t));
        }
        if st.all_taken(g) {
            return (st, None);
        }
        // Whatever is not ready waits for the parser or for a task in
        // hand. Were there neither, no wait would ever end: that would be
        // a mistake in `State::take`, and it stops here.
        assert!(
            !st.parse_done || st.running > 0,
            "a frame's rows are stuck: nothing ready and nothing running"
        );
        st = frame.wait(st);
    }
}

/// Reconstructs superblock row `r` from its record.
fn reconstruct(frame: &Frame, td: &mut TileDecoder, r: usize) {
    let g = &frame.g;
    let rec = frame.lock().parsed[r]
        .take()
        .expect("a parsed row's record");
    // The row's own samples, and the mode info of it and of the row above
    // (prediction looks at the blocks above and to the left: 7.11.2.8,
    // 7.11.3.10).
    td.f.mi = frame.mi_rows(r.saturating_sub(1), r + 1);
    for p in 0..g.num_planes {
        let band = g.band(p, r, frame.cur.height(p));
        td.band_top[p] = band.start;
        td.f.cur.planes[p].data = frame.cur.rows(p, band.clone(), band);
        td.above_line[p].resize(td.f.cur.planes[p].w, 0);
    }
    td.rec = rec;
    td.rewind();
    let done = reconstruct_superblocks(frame, td, r);
    td.f.mi = Shared::default();
    for p in td.f.cur.planes.iter_mut() {
        p.data = Shared::default();
    }
    let mut rec = std::mem::take(&mut td.rec);
    if done {
        rec.clear();
        frame.lock().spare.push(rec);
    }
}

/// The superblocks of row `r`, each behind the row above. Returns `false`
/// if the frame was abandoned meanwhile.
fn reconstruct_superblocks(frame: &Frame, td: &mut TileDecoder, r: usize) -> bool {
    let g = &frame.g;
    let mi_row = r * g.sb4;
    // Superblocks of the row above whose last line is in `above_line`.
    let mut pulled = 0;
    for k in 0..g.cols {
        if r > 0 {
            // The row above must have finished the superblock above and to
            // the right; its line comes across under the same lock.
            let need = (k + 2).min(g.cols);
            let mut st = frame.lock();
            while st.progress[r - 1] < need && !st.abort {
                st = frame.wait(st);
            }
            if st.abort {
                return false;
            }
            while pulled < need {
                for p in 0..g.num_planes {
                    let x = g.sb_columns(p, pulled);
                    td.above_line[p][x.clone()].copy_from_slice(&st.lines[r - 1][p][x]);
                }
                pulled += 1;
            }
        }
        td.clear_block_decoded_flags(mi_row, k * g.sb4, g.sb4);
        for i in td.rec.sb_blocks(k) {
            // Reconstruction reports nothing: every error of a tile is its
            // syntax's.
            td.replay_block(i).expect("reconstruction does not fail");
        }
        let mut st = frame.lock();
        if r + 1 < g.rows {
            for p in 0..g.num_planes {
                let x = g.sb_columns(p, k);
                let pl = &td.f.cur.planes[p];
                let at = (g.band(p, r, pl.h).end - 1) * pl.stride;
                st.lines[r][p][x.clone()].copy_from_slice(&pl.data[at + x.start..at + x.end]);
            }
        }
        st.progress[r] = k + 1;
        drop(st);
        frame.changed.notify_all();
    }
    true
}

/// The loop filter (7.14) of superblock row `r`: its vertical edges, then
/// its horizontal edges, the one along its top among them.
fn deblock(frame: &Frame, f: &mut FrameCtx, r: usize) {
    let g = &frame.g;
    // An edge is decided by the blocks and transforms on both its sides.
    let above = r.saturating_sub(1);
    f.mi = frame.mi_rows(above, r + 1);
    for p in 0..g.num_planes {
        let sizes = &frame.lf_tx_sizes[p];
        f.lf_tx_sizes[p] = sizes.window(g.units(above, r + 1, g.ss(p).1, sizes.len()), 0..0);
        let band = g.band(p, r, frame.cur.height(p));
        let rows = band.start.saturating_sub(LF_REACH[p])..band.end;
        f.cur.planes[p].data = frame.cur.rows(p, rows.clone(), rows);
    }
    let mi = g.mi_band(r);
    postfilter::loop_filter_rows(f, &frame.levels, mi.start, mi.end);
    f.mi = Shared::default();
    for p in 0..g.num_planes {
        f.lf_tx_sizes[p] = Shared::default();
        f.cur.planes[p].data = Shared::default();
    }
}

/// CDEF (7.15) of superblock row `r`: `CdefFrame`'s rows from
/// `CurrFrame`'s.
fn cdef(frame: &Frame, f: &mut FrameCtx, scratch: &mut Scratch, r: usize) {
    let g = &frame.g;
    let planes = frame.cdef.as_ref().expect("a frame with CDEF");
    {
        let st = frame.lock();
        let idx = g.cdef_idx(r, f.cdef_idx.len());
        f.cdef_idx[idx.clone()].copy_from_slice(&st.cdef_idx[idx]);
    }
    // Whether a block is filtered is its skip flags (7.15.1).
    f.mi = frame.mi_rows(r, r + 1);
    for p in 0..g.num_planes {
        let band = g.band(p, r, frame.cur.height(p));
        let read = band.start.saturating_sub(CDEF_REACH)..band.end + CDEF_REACH;
        f.cur.planes[p].data = frame.cur.rows(p, read, 0..0);
        // `CdefFrame` starts as a copy of `CurrFrame` (7.15).
        let out = &mut scratch.cdef.planes[p];
        out.data = planes.rows(p, band.clone(), band.clone());
        let samples = band.start * out.stride..band.end * out.stride;
        out.data[samples.clone()].copy_from_slice(&f.cur.planes[p].data[samples]);
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
    for p in 0..g.num_planes {
        f.cur.planes[p].data = Shared::default();
        scratch.cdef.planes[p].data = Shared::default();
    }
}

/// Loop restoration (7.17) of stripe `s`: `LrFrame`'s rows, for the planes
/// that are restored, from `CdefFrame`'s and (beyond the stripe)
/// `CurrFrame`'s.
fn restore(frame: &Frame, f: &mut FrameCtx, scratch: &mut Scratch, s: usize) {
    let g = &frame.g;
    let planes = frame.lr.as_ref().expect("a frame with loop restoration");
    for p in 0..g.num_planes {
        if !g.restored[p] {
            continue;
        }
        let Some((_, _, y0, y1, unit_row)) = postfilter::lr_stripe(f, p, s) else {
            continue;
        };
        let (y0, y1) = (y0 as usize, y1 as usize + 1);
        {
            // The stripe's units: one row of them (7.17.1).
            let st = frame.lock();
            let (from, to) = (&st.lr[p], &mut f.lr[p]);
            let units = unit_row * from.unit_cols..(unit_row + 1) * from.unit_cols;
            to.lr_type[units.clone()].copy_from_slice(&from.lr_type[units.clone()]);
            to.wiener[units.clone()].copy_from_slice(&from.wiener[units.clone()]);
            to.sgr_set[units.clone()].copy_from_slice(&from.sgr_set[units.clone()]);
            to.sgr_xqd[units.clone()].copy_from_slice(&from.sgr_xqd[units]);
        }
        let around = y0.saturating_sub(LR_REACH)..y1 + LR_REACH;
        f.cur.planes[p].data = frame.cur.rows(p, around, 0..0);
        if let Some(c) = &frame.cdef {
            scratch.cdef.planes[p].data = c.rows(p, y0..y1, 0..0);
        }
        let stride = f.cur.planes[p].stride;
        // `LrFrame` starts as a copy of `CdefFrame` (7.17).
        let out = &mut scratch.lr.planes[p];
        out.data = planes.rows(p, y0..y1, y0..y1);
        let samples = y0 * stride..y1 * stride;
        let from = if frame.cdef.is_some() {
            &scratch.cdef
        } else {
            &f.cur
        };
        out.data[samples.clone()].copy_from_slice(&from.planes[p].data[samples]);
        let out = SharedPlane::new(out);
        postfilter::restore_stripe(f, &f.cur, from, out, p, s);
        f.cur.planes[p].data = Shared::default();
        scratch.lr.planes[p].data = Shared::default();
        if frame.cdef.is_some() {
            scratch.cdef.planes[p].data = Shared::default();
        }
    }
}
