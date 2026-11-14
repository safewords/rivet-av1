//! Rate-distortion search, by trial coding.
//!
//! A decision — the partition of a block, the modes of a block, the
//! transform type of a transform block — is made by coding each candidate
//! with the tile walker itself in *counting* mode (the symbol coder adds up
//! each symbol's cost under the current CDFs instead of writing it, and
//! adapts nothing), measuring the distortion of the reconstruction against
//! the source, and keeping the candidate with the least
//! `distortion + lambda * bits`. Between candidates the region's state is
//! put back from a snapshot ([`Snap`]): the mode info, the contexts, the
//! reconstructed samples.
//!
//! Decisions nest: a partition candidate codes its sub-blocks, whose own
//! decisions are searched in turn. Every decision a caller is handed is
//! appended to a log; a search keeps the log segment of its best candidate
//! and *replays* it while its caller codes that candidate for real, so the
//! sub-decisions are not searched again. The final coding of a superblock
//! is a replay of the whole tree.

use std::collections::VecDeque;

use crate::Result;
use crate::consts::*;
use crate::decoder::state::Mi;
use crate::decoder::tile::{Plan, TileDecoder};
use crate::encoder::inter::ProbeFrom;
use crate::tables::*;

/// One decision, in coding order.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Decision {
    Partition(u8),
    Block(Plan),
    /// The luma transform type of a transform block.
    TxType(u8),
}

/// The search state the encoder keeps beside the tile walker.
#[derive(Default)]
pub(crate) struct Rdo {
    /// Every decision handed out, in coding order.
    pub(crate) log: Vec<Decision>,
    /// Decisions to hand out again instead of searching.
    pub(crate) replay: VecDeque<Decision>,
    /// The decision the next query must return (a candidate on trial).
    pub(crate) forced_partition: Option<usize>,
    pub(crate) forced_plan: Option<Plan>,
    pub(crate) forced_tx_type: Option<usize>,
    /// Trial coding of block candidates: luma transform types are the DCT,
    /// not searched.
    pub(crate) fast_tx: bool,
    /// Transform blocks with coefficients coded since the last reset.
    pub(crate) nonzero: u32,
    /// Snapshots for reuse.
    pool: Vec<Snap>,
    /// The partition candidates on trial, innermost last: the cost of what
    /// each has coded so far against the best candidate before it.
    trials: Vec<Trial>,
    /// The trial (index in `trials`) given up: its cost passed its budget.
    abort: Option<usize>,
    /// The references the blocks of a horizontal or vertical candidate
    /// search (`Tools::rect_refs`): the block whole's, or (it intra) none
    /// but the nearest.
    pub(crate) ref_hint: Option<[i32; 2]>,
}

/// A partition candidate on trial: the costs of the blocks and partitions
/// it has coded (a lower bound of its own cost: the partition symbols
/// between them are left out), and the cost to beat.
struct Trial {
    spent: f64,
    budget: f64,
}

/// The error that unwinds a trial given up (`Rdo::abort` says which); the
/// search that started the trial catches it.
fn abandoned() -> crate::Error {
    crate::Error::Unsupported(String::new())
}

/// The state of a rectangle of the frame, as trial coding may change it.
#[derive(Default)]
pub(crate) struct Snap {
    /// Luma 4x4 rectangle, rows `r0..r1`, columns `c0..c1`.
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    mi: Vec<Mi>,
    tx_types: Vec<u8>,
    seg: Vec<u8>,
    pal: [Vec<[u16; 8]>; 2],
    lf: [Vec<u8>; 3],
    pix: [Vec<u16>; 3],
    above: [Vec<u8>; 6],
    left: [Vec<u8>; 6],
    above_seg: Vec<u8>,
    left_seg: Vec<u8>,
    block_decoded: Vec<bool>,
    cdef: Vec<i8>,
    q: i32,
    delta_lf: [i32; 4],
    read_deltas: bool,
}

/// The 4x4 rectangle of `plane` covering the luma rectangle (exclusive
/// ends, clamped to `rows` x `cols`).
fn plane_rect(
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    ssx: usize,
    ssy: usize,
) -> (usize, usize, usize, usize) {
    (r0 >> ssy, (r1 + ssy) >> ssy, c0 >> ssx, (c1 + ssx) >> ssx)
}

impl TileDecoder<'_, '_> {
    pub(crate) fn rdo(&mut self) -> &mut Rdo {
        &mut self.enc.as_mut().expect("encode mode").rdo
    }

    /// The luma 4x4 rectangle of a `b_size` block at `(r, c)`, clamped to
    /// the per-4x4 arrays.
    fn block_rect(&self, r: usize, c: usize, b_size: usize) -> (usize, usize, usize, usize) {
        let rows = self.f.mi_rows + 32;
        let cols = self.f.ms;
        (
            r,
            (r + NUM_4X4_BLOCKS_HIGH[b_size]).min(rows),
            c,
            (c + NUM_4X4_BLOCKS_WIDE[b_size]).min(cols),
        )
    }

    /// Saves the state of a `b_size` block at `(r, c)`.
    pub(crate) fn snap(&mut self, r: usize, c: usize, b_size: usize) -> Snap {
        let mut s = self.rdo().pool.pop().unwrap_or_default();
        let (r0, r1, c0, c1) = self.block_rect(r, c, b_size);
        s.r0 = r0;
        s.r1 = r1;
        s.c0 = c0;
        s.c1 = c1;
        let f = &*self.f;
        let ms = f.ms;
        let grab = |v: &mut Vec<u8>, src: &[u8], r0: usize, r1: usize, c0: usize, c1: usize| {
            v.clear();
            for row in r0..r1 {
                v.extend_from_slice(&src[row * ms + c0..row * ms + c1]);
            }
        };
        s.mi.clear();
        s.pal[0].clear();
        s.pal[1].clear();
        for row in r0..r1 {
            s.mi.extend_from_slice(&f.mi[row * ms + c0..row * ms + c1]);
            if !f.palette_colors[0].is_empty() {
                s.pal[0].extend_from_slice(&f.palette_colors[0][row * ms + c0..row * ms + c1]);
                s.pal[1].extend_from_slice(&f.palette_colors[1][row * ms + c0..row * ms + c1]);
            }
        }
        grab(&mut s.tx_types, &f.tx_types, r0, r1, c0, c1);
        grab(&mut s.seg, &f.segment_ids, r0, r1, c0, c1);
        for p in 0..f.num_planes {
            let (ssx, ssy) = f.plane_ss(p);
            let (pr0, pr1, pc0, pc1) = plane_rect(r0, r1, c0, c1, ssx, ssy);
            let pr1 = pr1.min(f.mi_rows + 32);
            let pc1 = pc1.min(ms);
            grab(
                &mut s.lf[p],
                f.lf_tx_sizes[p].as_slice(),
                pr0,
                pr1,
                pc0,
                pc1,
            );
            let pl = &f.cur.planes[p];
            let (x0, x1) = (pc0 * 4, (pc1 * 4).min(pl.w));
            let (y0, y1) = (pr0 * 4, (pr1 * 4).min(pl.h));
            s.pix[p].clear();
            for y in y0..y1 {
                s.pix[p].extend_from_slice(&pl.data[y * pl.stride + x0..y * pl.stride + x1]);
            }
            let ac = self.above_level_ctx[p].len();
            let lc = self.left_level_ctx[p].len();
            s.above[2 * p].clear();
            s.above[2 * p].extend_from_slice(&self.above_level_ctx[p][pc0..pc1.min(ac)]);
            s.above[2 * p + 1].clear();
            s.above[2 * p + 1].extend_from_slice(&self.above_dc_ctx[p][pc0..pc1.min(ac)]);
            s.left[2 * p].clear();
            s.left[2 * p].extend_from_slice(&self.left_level_ctx[p][pr0..pr1.min(lc)]);
            s.left[2 * p + 1].clear();
            s.left[2 * p + 1].extend_from_slice(&self.left_dc_ctx[p][pr0..pr1.min(lc)]);
        }
        s.above_seg.clear();
        s.above_seg
            .extend_from_slice(&self.above_seg_pred_ctx[c0..c1.min(self.above_seg_pred_ctx.len())]);
        s.left_seg.clear();
        s.left_seg
            .extend_from_slice(&self.left_seg_pred_ctx[r0..r1.min(self.left_seg_pred_ctx.len())]);
        s.block_decoded.clear();
        for p in 0..3 {
            for row in &self.block_decoded[p] {
                s.block_decoded.extend_from_slice(row);
            }
        }
        s.cdef.clear();
        s.cdef.extend_from_slice(&f.cdef_idx);
        s.q = self.current_q_index;
        s.delta_lf = self.delta_lf;
        s.read_deltas = self.read_deltas;
        s
    }

    /// Puts back the state saved in `s`.
    pub(crate) fn restore(&mut self, s: &Snap) {
        self.restore_parts(s, true);
    }

    /// Puts back the state saved in `s`, the samples only with `pixels`
    /// (a probe of a prediction writes nothing the next one reads).
    pub(crate) fn restore_parts(&mut self, s: &Snap, pixels: bool) {
        let (r0, r1, c0, c1) = (s.r0, s.r1, s.c0, s.c1);
        let w = c1 - c0;
        let f = &mut *self.f;
        let ms = f.ms;
        for (i, row) in (r0..r1).enumerate() {
            f.mi[row * ms + c0..row * ms + c1].copy_from_slice(&s.mi[i * w..(i + 1) * w]);
            if !f.palette_colors[0].is_empty() {
                f.palette_colors[0][row * ms + c0..row * ms + c1]
                    .copy_from_slice(&s.pal[0][i * w..(i + 1) * w]);
                f.palette_colors[1][row * ms + c0..row * ms + c1]
                    .copy_from_slice(&s.pal[1][i * w..(i + 1) * w]);
            }
            f.tx_types[row * ms + c0..row * ms + c1]
                .copy_from_slice(&s.tx_types[i * w..(i + 1) * w]);
            f.segment_ids[row * ms + c0..row * ms + c1].copy_from_slice(&s.seg[i * w..(i + 1) * w]);
        }
        for p in 0..f.num_planes {
            let (ssx, ssy) = f.plane_ss(p);
            let (pr0, pr1, pc0, pc1) = plane_rect(r0, r1, c0, c1, ssx, ssy);
            let pr1 = pr1.min(f.mi_rows + 32);
            let pc1 = pc1.min(ms);
            let pw = pc1 - pc0;
            for (i, row) in (pr0..pr1).enumerate() {
                f.lf_tx_sizes[p][row * ms + pc0..row * ms + pc1]
                    .copy_from_slice(&s.lf[p][i * pw..(i + 1) * pw]);
            }
            let pl = &mut f.cur.planes[p];
            let (x0, x1) = (pc0 * 4, (pc1 * 4).min(pl.w));
            let (y0, y1) = (pr0 * 4, (pr1 * 4).min(pl.h));
            let xw = x1 - x0;
            for (i, y) in (y0..y1).enumerate().filter(|_| pixels) {
                pl.data[y * pl.stride + x0..y * pl.stride + x1]
                    .copy_from_slice(&s.pix[p][i * xw..(i + 1) * xw]);
            }
            let n = s.above[2 * p].len();
            self.above_level_ctx[p][pc0..pc0 + n].copy_from_slice(&s.above[2 * p]);
            self.above_dc_ctx[p][pc0..pc0 + n].copy_from_slice(&s.above[2 * p + 1]);
            let n = s.left[2 * p].len();
            self.left_level_ctx[p][pr0..pr0 + n].copy_from_slice(&s.left[2 * p]);
            self.left_dc_ctx[p][pr0..pr0 + n].copy_from_slice(&s.left[2 * p + 1]);
        }
        let n = s.above_seg.len();
        self.above_seg_pred_ctx[c0..c0 + n].copy_from_slice(&s.above_seg);
        let n = s.left_seg.len();
        self.left_seg_pred_ctx[r0..r0 + n].copy_from_slice(&s.left_seg);
        let mut k = 0;
        for p in 0..3 {
            for row in self.block_decoded[p].iter_mut() {
                let n = row.len();
                row.copy_from_slice(&s.block_decoded[k..k + n]);
                k += n;
            }
        }
        f.cdef_idx.copy_from_slice(&s.cdef);
        self.current_q_index = s.q;
        self.delta_lf = s.delta_lf;
        self.read_deltas = s.read_deltas;
    }

    /// Returns a snapshot to the pool.
    pub(crate) fn release(&mut self, s: Snap) {
        self.rdo().pool.push(s);
    }

    /// Sum of squared differences between the source and the
    /// reconstruction over a luma 4x4 rectangle (and the chroma it covers),
    /// inside the frame, each plane's weighted by [`CHROMA_WEIGHT`].
    pub(crate) fn rect_sse(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> u64 {
        let e = self.enc.as_ref().expect("encode mode");
        let f = &*self.f;
        let mut total = 0u64;
        for p in 0..f.num_planes {
            let (ssx, ssy) = f.plane_ss(p);
            let fw = (f.hdr.frame_width as usize + ssx) >> ssx;
            let fh = (f.hdr.frame_height as usize + ssy) >> ssy;
            let x0 = (c0 * 4) >> ssx;
            let x1 = ((c1 * 4) >> ssx).min(fw);
            let y0 = (r0 * 4) >> ssy;
            let y1 = ((r1 * 4) >> ssy).min(fh);
            let pl = &f.cur.planes[p];
            let st = e.stride[p];
            for y in y0..y1 {
                let a = &e.src[p][y * st + x0..y * st + x1.max(x0)];
                let b = &pl.data[y * pl.stride + x0..y * pl.stride + x1.max(x0)];
                total += (crate::dsp::enc::sse_row(a, b) * CHROMA_WEIGHT[p]) >> 2;
            }
        }
        total
    }

    /// The rate-distortion cost of `sse` and `bits` (in 1/256 bit).
    #[inline]
    pub(crate) fn rd_cost(&self, sse: u64, bits: u64) -> f64 {
        sse as f64 + self.enc.as_ref().expect("encode mode").rd_lambda * bits as f64 / 256.0
    }

    /// The decision the encoder hands out next when it is not searching:
    /// a replayed or forced one. Logs it.
    fn next_partition(&mut self) -> Option<usize> {
        let rdo = self.rdo();
        let p = if let Some(d) = rdo.replay.pop_front() {
            match d {
                Decision::Partition(p) => p as usize,
                _ => panic!("replay out of step: {d:?} where a partition was"),
            }
        } else {
            rdo.forced_partition.take()?
        };
        rdo.log.push(Decision::Partition(p as u8));
        Some(p)
    }

    /// Adds the cost of a block or partition just decided to the
    /// innermost trial; gives the trial up (an error the trial's search
    /// catches) when it can no longer win.
    fn spend(&mut self, cost: f64) -> Result<()> {
        let rdo = self.rdo();
        let n = rdo.trials.len();
        if let Some(t) = rdo.trials.last_mut() {
            t.spent += cost;
            if t.spent > t.budget {
                rdo.abort = Some(n - 1);
                return Err(abandoned());
            }
        }
        Ok(())
    }

    /// The partition of a block: replayed, forced, or searched.
    pub(crate) fn enc_partition_rd(
        &mut self,
        r: usize,
        c: usize,
        b_size: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<usize> {
        if let Some(p) = self.next_partition() {
            return Ok(p);
        }
        let tools = self.enc.as_ref().expect("encode mode").tools;
        let mut cands: Vec<usize> = Vec::with_capacity(4);
        if has_rows && has_cols {
            cands.push(PARTITION_NONE);
            let bw = block_width(b_size);
            if tools.partition_rect
                && (tools.rect_min as usize..=tools.rect_max as usize).contains(&bw)
            {
                cands.push(PARTITION_HORZ);
                cands.push(PARTITION_VERT);
            }
            if b_size > BLOCK_8X8 || tools.partition_4x4 {
                cands.push(PARTITION_SPLIT);
            }
        } else if has_cols {
            cands.push(PARTITION_HORZ);
            cands.push(PARTITION_SPLIT);
        } else {
            cands.push(PARTITION_VERT);
            cands.push(PARTITION_SPLIT);
        }
        // Blocks whose chroma would have no size at this subsampling (a
        // vertical half at 4:2:2: twice as tall as wide, chroma four times)
        // are not allowed (6.10.4).
        if self.f.num_planes > 1 {
            cands.retain(|&p| {
                self.f.plane_residual_size(PARTITION_SUBSIZE[p][b_size], 1) != BLOCK_INVALID
            });
        }
        if cands.len() == 1 {
            self.rdo().log.push(Decision::Partition(cands[0] as u8));
            return Ok(cands[0]);
        }
        debug_assert!(self.rdo().replay.is_empty());
        let snap = self.snap(r, c, b_size);
        let saved_b = self.b.clone();
        let counting = self.sd.set_counting(true);
        let bits_start = self.sd.trial_bits();
        let log_start = self.rdo().log.len();
        let mut best: (f64, usize, Vec<Decision>) = (f64::MAX, cands[0], Vec::new());
        let mut none_flat = false;
        // Speed: the split first; when its sub-blocks did not stay whole,
        // the block as one (or two) is not tried.
        let split_first = tools.prune_partition && b_size > BLOCK_8X8;
        if split_first && let Some(k) = cands.iter().position(|&p| p == PARTITION_SPLIT) {
            cands.remove(k);
            cands.insert(0, PARTITION_SPLIT);
        }
        let mut children_whole = true;
        let mut none_refs: Option<[i32; 2]> = None;
        let mut first = true;
        for &p in cands.iter() {
            if !first {
                self.restore(&snap);
            }
            // A flat block that codes with no residual as one block is not
            // split further (speed).
            if p == PARTITION_SPLIT && none_flat && tools.prune_split && !split_first {
                continue;
            }
            if split_first && p != PARTITION_SPLIT && !children_whole {
                continue;
            }
            first = false;
            self.rdo().forced_partition = Some(p);
            self.rdo().nonzero = 0;
            let bits0 = self.sd.trial_bits();
            let depth = self.rdo().trials.len();
            let budget = if tools.abandon_trials {
                best.0
            } else {
                f64::MAX
            };
            self.rdo().trials.push(Trial { spent: 0.0, budget });
            if tools.rect_refs && (p == PARTITION_HORZ || p == PARTITION_VERT) {
                self.rdo().ref_hint = none_refs;
            }
            let res = self.decode_partition(r, c, b_size);
            self.rdo().ref_hint = None;
            if let Err(e) = res {
                let rdo = self.rdo();
                if rdo.abort != Some(depth) {
                    return Err(e);
                }
                // Given up: it costs more than the best already.
                rdo.abort = None;
                rdo.trials.truncate(depth);
                rdo.replay.clear();
                rdo.forced_partition = None;
                rdo.forced_plan = None;
                rdo.forced_tx_type = None;
                rdo.log.truncate(log_start);
                continue;
            }
            self.rdo().trials.truncate(depth);
            let bits = self.sd.trial_bits() - bits0;
            let sse = self.rect_sse(snap.r0, snap.r1, snap.c0, snap.c1);
            let cost = self.rd_cost(sse, bits);
            if p == PARTITION_NONE {
                none_refs = self.rdo().log[log_start..].iter().find_map(|d| match d {
                    Decision::Block(b) if b.is_inter => Some([b.ref_frame, b.ref_frame2]),
                    Decision::Block(_) => Some([INTRA_FRAME, INTRA_FRAME]),
                    _ => None,
                });
                none_flat = self.rdo().nonzero == 0;
                // Or one whose cost per sample is low already.
                let thr = if b_size >= BLOCK_32X32 {
                    tools.split_threshold
                } else {
                    tools.split_threshold_small
                } as f64;
                if thr > 0.0 {
                    let area = (block_width(b_size) * block_height(b_size)) as f64;
                    let lambda = self.enc.as_ref().expect("encode mode").rd_lambda;
                    if cost < thr / 100.0 * lambda * area {
                        none_flat = true;
                    }
                }
            }
            if p == PARTITION_SPLIT && split_first {
                // Whole (or halved): none of the sub-blocks split again.
                children_whole = self.rdo().log[log_start + 1..].iter().all(
                    |d| !matches!(d, Decision::Partition(q) if *q as usize == PARTITION_SPLIT),
                );
            }
            if cost < best.0 {
                let seg = self.rdo().log[log_start..].to_vec();
                best = (cost, p, seg);
            }
            self.rdo().log.truncate(log_start);
        }
        self.restore(&snap);
        self.release(snap);
        self.b = saved_b;
        self.sd.set_counting(counting);
        self.sd.set_trial_bits(bits_start);
        let rdo = self.rdo();
        // The best candidate's own partition decision heads its segment;
        // its caller codes that, then replays the rest.
        rdo.replay.extend(best.2.into_iter().skip(1));
        rdo.log.push(Decision::Partition(best.1 as u8));
        self.spend(best.0)?;
        Ok(best.1)
    }

    /// The modes of a block: replayed, forced, or searched. Sets `plan`.
    ///
    /// The candidates are trialled with the DCT alone (`Rdo::fast_tx`);
    /// the best then with its transform types searched, at each transform
    /// depth.
    pub(crate) fn enc_decide_block_rd(&mut self) -> Result<()> {
        let rdo = self.rdo();
        let plan = if let Some(d) = rdo.replay.pop_front() {
            match d {
                Decision::Block(p) => Some(p),
                _ => panic!("replay out of step: {d:?} where a block was"),
            }
        } else {
            rdo.forced_plan.take()
        };
        if let Some(p) = plan {
            self.plan = p;
            self.rdo().log.push(Decision::Block(p));
            return Ok(());
        }
        let cands = self.block_candidates();
        let (r, c, size) = (self.b.mi_row, self.b.mi_col, self.b.mi_size);
        let snap = self.snap(r, c, size);
        let saved_b = self.b.clone();
        let counting = self.sd.set_counting(true);
        let bits_start = self.sd.trial_bits();
        let log_start = self.rdo().log.len();
        let fast_saved = self.rdo().fast_tx;
        let tools = self.enc.as_ref().expect("encode mode").tools;
        let mut best: (f64, Plan, Vec<Decision>) = (f64::MAX, cands[0], Vec::new());
        let mut first = true;
        let mut trial = |t: &mut Self, p: Plan, fast: bool| -> Result<(f64, bool, Vec<Decision>)> {
            if !first {
                t.restore(&snap);
            }
            first = false;
            t.b = saved_b.clone();
            let rdo = t.rdo();
            rdo.forced_plan = Some(p);
            rdo.nonzero = 0;
            rdo.fast_tx = fast;
            let bits0 = t.sd.trial_bits();
            t.decode_block(r, c, size)?;
            let bits = t.sd.trial_bits() - bits0;
            let sse = t.rect_sse(snap.r0, snap.r1, snap.c0, snap.c1);
            let rdo = t.rdo();
            let zero = rdo.nonzero == 0;
            let seg = rdo.log.split_off(log_start);
            Ok((t.rd_cost(sse, bits), zero, seg))
        };
        let refine = tools.tx_type_rd || tools.tx_size;
        let mut fast_best: (f64, Plan) = (f64::MAX, cands[0]);
        // The best inter candidate coded with a residual, for the inter
        // tools.
        let mut inter_best: (f64, Plan) = (f64::MAX, cands[0]);
        let mut fast_zero = false;
        let mut best_inter_fast = f64::MAX;
        for &p in &cands {
            let (cost, zero, seg) = trial(self, p, refine)?;
            if cost < best.0 {
                best = (cost, p, seg);
            }
            if cost < fast_best.0 && !p.skip {
                fast_best = (cost, p);
                fast_zero = zero;
            }
            let inter_best_here = p.is_inter && cost < best_inter_fast;
            if p.is_inter {
                best_inter_fast = best_inter_fast.min(cost);
            }
            if p.is_inter && !p.skip && cost < inter_best.0 {
                inter_best = (cost, p);
            }
            // No residual: the skip flag says so for less.
            if !p.skip
                && (zero
                    || (p.is_inter && tools.rd_skip && (!tools.rd_skip_best || inter_best_here)))
            {
                let mut q = p;
                q.skip = true;
                q.tx_depth = 0;
                let (cost, _, seg) = trial(self, q, false)?;
                if cost < best.0 {
                    best = (cost, q, seg);
                }
            }
        }
        // The inter tools: the variants of the best inter candidate that
        // probe cheaper, the best of them trialled (or the one a test
        // forces, whatever it costs).
        if inter_best.0 < f64::MAX && tools.inter_tools() {
            let from = ProbeFrom {
                snap: &snap,
                b: &saved_b,
                log_start,
            };
            let v = self.inter_variants(&from, inter_best.1)?;
            let trials = v.better.into_iter().take(tools.inter_tool_trials as usize);
            for (q, forced) in trials
                .map(|q| (q, false))
                .chain(v.forced.map(|q| (q, true)))
            {
                let (cost, zero, seg) = trial(self, q, refine)?;
                // A forced variant is taken, and refined, as if it were the
                // best.
                if cost < best.0 || forced {
                    best = (cost, q, seg);
                }
                if cost < fast_best.0 || forced {
                    fast_best = (cost, q);
                    fast_zero = zero;
                }
                if zero {
                    let mut q = q;
                    q.skip = true;
                    q.tx_depth = 0;
                    let (cost, _, seg) = trial(self, q, false)?;
                    if cost < best.0 {
                        best = (cost, q, seg);
                    }
                }
            }
        }
        // The best coded block: its transform types searched, at each
        // transform depth.
        // (A best coded without coefficients is not refined.)
        if refine && fast_best.0 < f64::MAX && !(tools.fast_tx_refine && fast_zero) {
            let base = fast_best.1;
            let max_depth = if tools.tx_size {
                self.max_tx_depth(size, base.is_inter)
                    .min(tools.tx_depth_max as usize)
            } else {
                0
            };
            if tools.fast_tx_refine {
                // The depths with the DCT, then the transform types at the
                // best of them.
                let mut best_d = (fast_best.0, base.tx_depth);
                for d in 0..=max_depth {
                    if d as u8 == base.tx_depth {
                        continue;
                    }
                    let mut q = base;
                    q.tx_depth = d as u8;
                    let (cost, _, seg) = trial(self, q, true)?;
                    if cost < best_d.0 {
                        best_d = (cost, q.tx_depth);
                    }
                    if cost < best.0 {
                        best = (cost, q, seg);
                    }
                }
                if tools.tx_type_rd {
                    let mut q = base;
                    q.tx_depth = best_d.1;
                    let (cost, _, seg) = trial(self, q, false)?;
                    if cost < best.0 {
                        best = (cost, q, seg);
                    }
                }
            } else {
                for d in 0..=max_depth {
                    let mut q = base;
                    q.tx_depth = d as u8;
                    let (cost, _, seg) = trial(self, q, false)?;
                    if cost < best.0 {
                        best = (cost, q, seg);
                    }
                }
            }
        }
        self.restore(&snap);
        self.release(snap);
        self.b = saved_b;
        self.sd.set_counting(counting);
        self.sd.set_trial_bits(bits_start);
        let rdo = self.rdo();
        rdo.fast_tx = fast_saved;
        self.plan = best.1;
        // The decisions inside the block (transform types) are replayed
        // when the caller codes it.
        let rdo = self.rdo();
        rdo.replay.extend(best.2.into_iter().skip(1));
        rdo.log.push(Decision::Block(best.1));
        self.spend(best.0)?;
        Ok(())
    }

    /// The deepest transform split a block may code: `tx_depth` (intra,
    /// 5.11.15) or the uniform variable-transform depth (inter, 5.11.17).
    pub(crate) fn max_tx_depth(&self, size: usize, inter: bool) -> usize {
        if self.f.hdr.tx_mode != TX_MODE_SELECT || size == BLOCK_4X4 {
            return 0;
        }
        if inter {
            // Down to 4x4 at most, two levels.
            let mut t = MAX_TX_SIZE_RECT[size];
            let mut d = 0;
            while d < MAX_VARTX_DEPTH && t != TX_4X4 {
                t = SPLIT_TX_SIZE[t];
                d += 1;
            }
            d
        } else {
            MAX_TX_DEPTH[size].min(MAX_TX_DEPTH_CODED)
        }
    }
}

/// How much a squared error counts in each plane (in quarters), against
/// luma's: the search's distortion is the weighted sum. Counted as luma,
/// chroma is under-coded for a measure that weighs each plane's PSNR (as
/// the 6:1:1 PSNR-YUV does): a chroma plane has a quarter of the samples
/// and usually a fraction of luma's error, so the same squared error moves
/// its PSNR several times as far. Measured at speed 6 on the five test
/// clips (with rate-distortion optimised quantisation), against weighing
/// every plane alike: U 1.5 and V 3 cost +2.0 % BD-rate PSNR-Y for −3.2 %
/// PSNR-YUV; U and V 2, +2.3 % for −3.1 %; U and V 3, +4.0 % for −3.6 %.
/// V weighs more than U because its error is usually the smaller (by
/// about two in these clips, from 1.0 to 2.6).
pub(crate) const CHROMA_WEIGHT: [u64; 3] = [4, 6, 12];

/// The deepest `tx_depth` the syntax codes.
const MAX_TX_DEPTH_CODED: usize = 2;
