//! A one-tile frame on several threads: superblock rows reconstructed
//! behind the parsing of the tile.
//!
//! `docs/threading.md` derives the rules; in short: the thread that called
//! the decoder parses the tile's superblock rows in order
//! ([`Pass::Parse`]), and hands each row's record on as it completes; the
//! other threads take the rows in order and reconstruct them
//! ([`Pass::Recon`]). A row's reconstruction reads the frame's samples in
//! one place outside its own rows, the row of samples above it, which
//! intra prediction's edge takes from the above-right superblock at the
//! furthest (7.11.2): so superblock `c` of a row waits for the row above
//! to have finished superblock `c + 1`, and takes that row's last line of
//! samples as a copy, through the lock the rows keep their progress under.
//! Its own samples are its alone ([`Shared`]).

use std::sync::{Condvar, Mutex, MutexGuard};

use crate::Result;
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::decoder::record::{Pass, RowRecord};
use crate::decoder::shared::{Lender, Shared};
use crate::decoder::state::{FrameBuf, Mi, PlaneBuf};
use crate::decoder::tile::TileDecoder;
use crate::decoder::{CoefficientReader, FrameCtx};

/// The frame's shape in superblock rows.
#[derive(Clone, Copy)]
struct Geometry {
    /// Superblock rows and columns of the tile (the frame).
    rows: usize,
    cols: usize,
    /// The superblock size in 4x4 units and in luma samples.
    sb4: usize,
    sb_px: usize,
    mi_cols: usize,
    /// Row stride of the per-4x4 arrays.
    ms: usize,
    num_planes: usize,
    ssx: usize,
    ssy: usize,
}

impl Geometry {
    fn of(f: &FrameCtx) -> Self {
        let sb4 = if f.seq.use_128x128_superblock { 32 } else { 16 };
        Geometry {
            rows: f.mi_rows.div_ceil(sb4),
            cols: f.mi_cols.div_ceil(sb4),
            sb4,
            sb_px: sb4 * MI_SIZE,
            mi_cols: f.mi_cols,
            ms: f.ms,
            num_planes: f.num_planes,
            ssx: f.ssx,
            ssy: f.ssy,
        }
    }

    fn ss(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.ssx, self.ssy)
        }
    }

    /// The per-4x4 elements of superblock rows `r0..r1`: whole rows of the
    /// arrays, with the margin below the frame for its last row (blocks
    /// overhanging the frame write there).
    fn mi_range(&self, r0: usize, r1: usize, len: usize) -> std::ops::Range<usize> {
        let end = if r1 >= self.rows {
            len
        } else {
            r1 * self.sb4 * self.ms
        };
        r0 * self.sb4 * self.ms..end
    }

    /// The samples of superblock row `r` in a plane of `stride` by `h`:
    /// whole rows, to the plane's end for the frame's last row (the
    /// padding the last blocks' predictions overhang into).
    fn band(&self, plane: usize, r: usize, stride: usize, h: usize) -> std::ops::Range<usize> {
        let (_, sy) = self.ss(plane);
        let top = (r * self.sb_px) >> sy;
        let bottom = if r + 1 >= self.rows {
            h
        } else {
            ((r + 1) * self.sb_px) >> sy
        };
        top * stride..bottom * stride
    }

    /// The columns of superblock `k` in a plane, inside the frame's mode
    /// info: the samples of the row's last line that a row below may ask
    /// for.
    fn sb_columns(&self, plane: usize, k: usize) -> std::ops::Range<usize> {
        let (sx, _) = self.ss(plane);
        let end = ((k + 1) * self.sb_px).min(self.mi_cols * MI_SIZE);
        (k * self.sb_px) >> sx..end >> sx
    }
}

/// What the threads of a frame keep under one lock.
struct State {
    /// The records of the rows parsed and not yet taken to reconstruct.
    parsed: Vec<Option<RowRecord>>,
    /// Rows parsed so far; whether there will be no more.
    parsed_rows: usize,
    parse_done: bool,
    /// The next row to reconstruct (they are taken in order).
    next_recon: usize,
    /// Superblocks reconstructed in each row.
    progress: Vec<usize>,
    /// Each row's last line of samples, by plane: valid for the
    /// superblocks `progress` counts.
    lines: Vec<[Vec<u16>; 3]>,
    /// Records to write the next rows into.
    spare: Vec<RowRecord>,
    /// Parsing failed, or a thread panicked: nothing more is started and
    /// nobody waits.
    abort: bool,
}

struct Frame<'a> {
    g: Geometry,
    state: Mutex<State>,
    /// Signalled on every change of `state` a thread may be waiting for.
    changed: Condvar,
    /// The frame's parameters, for each thread to make its own state from.
    template: &'a FrameCtx,
    mi: Lender<Mi>,
    planes: Vec<(Lender<u16>, usize, usize, usize)>,
}

impl Frame<'_> {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A thread that panicked sets `abort` on its way out (`Bail`); the
        // state it leaves is still what the others need to stop.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
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
    /// of its arrays: what a thread reconstructing rows of the frame works
    /// in, given the rows (see [`Frame`]).
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
            cdef_idx: Vec::new(),
            cdef_stride: self.cdef_stride,
            lr: Default::default(),
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
/// what `TileDecoder::decode_tile` does, and the same samples, mode info
/// and CDFs. Returns whether the tile's padding checked out and the tile's
/// CDFs.
pub(crate) fn decode_tile(
    f: &mut FrameCtx,
    data: &[u8],
    threads: usize,
    coeff_reader: CoefficientReader,
) -> Result<(bool, Box<CdfContext>)> {
    let g = Geometry::of(f);
    let template = f.view();
    // The planes are the reconstructing threads' for the duration; the
    // mode info stays with the parsing thread, which gives up each row as
    // it completes it.
    let planes: Vec<_> = f
        .cur
        .planes
        .iter_mut()
        .map(|p| {
            let lender = p.data.lender();
            p.data = Shared::default();
            (lender, p.stride, p.w, p.h)
        })
        .collect();
    let frame = Frame {
        g,
        state: Mutex::new(State {
            parsed: (0..g.rows).map(|_| None).collect(),
            parsed_rows: 0,
            parse_done: false,
            next_recon: 0,
            progress: vec![0; g.rows],
            lines: (0..g.rows)
                .map(|_| {
                    std::array::from_fn(|p| {
                        if p < g.num_planes {
                            vec![0; planes[p].2]
                        } else {
                            Vec::new()
                        }
                    })
                })
                .collect(),
            spare: Vec::new(),
            abort: false,
        }),
        changed: Condvar::new(),
        template: &template,
        mi: f.mi.lender(),
        planes,
    };
    // The parsing walk, taken by the calling thread's job.
    type Parsed = Result<(bool, Box<CdfContext>)>;
    let mut td = TileDecoder::new(f, data, 0, 0);
    td.coeff_reader = coeff_reader;
    td.pass = Pass::Parse;
    let parser = Mutex::new(Some(td));
    let outcome: Mutex<Option<Parsed>> = Mutex::new(None);
    crate::parallel::run(threads.min(g.rows + 1).max(1), &|k| {
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
        reconstruct(&frame);
    });
    let Frame { mi, planes, .. } = frame;
    drop(template);
    // Every window is back: the arrays are whole again.
    f.mi = Shared::default();
    f.mi = mi.whole();
    for (p, (lender, ..)) in f.cur.planes.iter_mut().zip(planes) {
        p.data = lender.whole();
    }
    outcome
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .expect("the calling thread's job ran")
}

/// The calling thread's part: parses the tile's superblock rows in order
/// and publishes each.
fn parse(frame: &Frame, td: &mut TileDecoder) -> Result<()> {
    let g = &frame.g;
    td.begin_tile();
    let len = td.f.mi.len();
    for r in 0..g.rows {
        let parsed = td.parse_sb_row(r * g.sb4);
        if parsed.is_err() {
            let mut st = frame.lock();
            st.abort = true;
            frame.changed.notify_all();
            return parsed;
        }
        // The row's mode info is final: readable by all from here on,
        // written by nobody.
        let rest = if r + 1 < g.rows {
            g.mi_range(r + 1, g.rows, len)
        } else {
            len..len
        };
        td.f.mi.narrow(0..len, rest);
        let mut st = frame.lock();
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

/// Takes the next parsed row to reconstruct, waiting for one; `None` when
/// there will be no more (or the frame was abandoned).
fn take_row(frame: &Frame) -> Option<(usize, RowRecord)> {
    let mut st = frame.lock();
    loop {
        if st.abort {
            return None;
        }
        if st.next_recon < st.parsed_rows {
            let r = st.next_recon;
            st.next_recon += 1;
            return Some((r, st.parsed[r].take().expect("a parsed row's record")));
        }
        if st.parse_done {
            return None;
        }
        st = frame.changed.wait(st).unwrap_or_else(|e| e.into_inner());
    }
}

/// Every thread's part: reconstructs parsed rows until none is left to
/// take.
fn reconstruct(frame: &Frame) {
    let Some(mut next) = take_row(frame) else {
        return;
    };
    // This thread's frame state and walker, made for its first row.
    let mut view = frame.template.view();
    let mut td = TileDecoder::new(&mut view, &[], 0, 0);
    td.pass = Pass::Recon;
    loop {
        let (r, rec) = next;
        let rec = reconstruct_row(frame, &mut td, r, rec);
        // The row is given back: its samples and mode info are whoever's
        // comes next.
        td.f.mi = Shared::default();
        for p in td.f.cur.planes.iter_mut() {
            p.data = Shared::default();
        }
        if let Some(mut rec) = rec {
            rec.clear();
            frame.lock().spare.push(rec);
        }
        match take_row(frame) {
            Some(n) => next = n,
            None => return,
        }
    }
}

/// Reconstructs superblock row `r` from its record. Returns the record,
/// or `None` if the frame was abandoned meanwhile.
fn reconstruct_row(
    frame: &Frame,
    td: &mut TileDecoder,
    r: usize,
    rec: RowRecord,
) -> Option<RowRecord> {
    let g = &frame.g;
    // The row's own samples, and the mode info of it and of the row above
    // (prediction looks at the blocks above and to the left: 7.11.2,
    // 7.11.3.10).
    let mi_len = frame.mi.len();
    let mi_rows = g.mi_range(r.saturating_sub(1), r + 1, mi_len);
    td.f.mi = frame.mi.window(mi_rows, 0..0);
    for (p, (lender, stride, _, h)) in frame.planes.iter().enumerate() {
        let band = g.band(p, r, *stride, *h);
        td.band_top[p] = band.start / stride;
        td.f.cur.planes[p].data = lender.window(band.clone(), band);
        td.above_line[p].resize(frame.planes[p].2, 0);
    }
    td.rec = rec;
    td.rewind();
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
                st = frame.changed.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.abort {
                return None;
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
                let (_, sy) = g.ss(p);
                let y = (((r + 1) * g.sb_px) >> sy) - 1;
                st.lines[r][p][x.clone()]
                    .copy_from_slice(&pl.data[y * pl.stride + x.start..y * pl.stride + x.end]);
            }
        }
        st.progress[r] = k + 1;
        drop(st);
        frame.changed.notify_all();
    }
    Some(std::mem::take(&mut td.rec))
}
