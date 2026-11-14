//! The encoder's decisions, made inside the tile walker.
//!
//! The walker of `decoder::tile` codes the syntax; in encode mode it asks
//! these methods for each decision just before the syntax codes it, with
//! the reconstruction so far in `CurrFrame`: the partition, the modes of a
//! block (from predictions made with the normative predictors), its motion
//! vector (a search on the reference), and the quantised levels of each
//! transform block (from the prediction then in place). With
//! rate-distortion optimisation on (`Tools::rdo`), the decisions are
//! searched by trial coding (`encoder::rdo`); these methods then supply the
//! candidates.

use crate::Result;
use crate::consts::*;
use crate::decoder::predict::block_inter_prediction;
use crate::decoder::state::Mv;
use crate::decoder::tile::{Plan, TileDecoder};
use crate::dsp::enc::{sad, satd};
use crate::encoder::Tools;
use crate::encoder::fwd::{Quantiser, coef_bound, forward_quantise, forward_wht};
use crate::encoder::rdo::{Decision, Rdo};
use crate::symbol::Coder;
use crate::tables::*;

/// The encoder's per-frame state inside a tile.
pub(crate) struct EncCtx {
    /// The source planes, padded like `CurrFrame` (edges replicated).
    pub(crate) src: std::sync::Arc<Vec<Vec<u16>>>,
    pub(crate) stride: Vec<usize>,
    /// The quantised levels chosen for the transform block being coded,
    /// in `Quant`'s layout.
    pub(crate) coefs: Box<[i32; 1024]>,
    /// Lagrange multiplier for the SATD-based preselection (SATD units per
    /// bit).
    pub(crate) lambda: f64,
    /// Lagrange multiplier of the rate-distortion search (squared error
    /// per bit).
    pub(crate) rd_lambda: f64,
    /// The references inter prediction may use (`LAST_FRAME`...); empty
    /// on intra frames.
    pub(crate) refs: Vec<i32>,
    /// Half-width of the full-pel motion search.
    pub(crate) search_range: i32,
    pub(crate) tools: Tools,
    pub(crate) rdo: Rdo,
    /// Scratch: the residual and the forward transform of a block.
    pub(crate) res: Vec<i32>,
    pub(crate) fc: Vec<f64>,
    /// Each 64x64 block's CDEF index, once searched (`cdef_idx` layout).
    pub(crate) cdef_table: Option<Vec<i8>>,
    /// Each loop restoration unit's parameters, once searched.
    pub(crate) lr_plan: Option<std::sync::Arc<crate::encoder::lr::LrPlan>>,
    /// The palette indices of the block being coded (palette blocks).
    pub(crate) palette_map: Box<[[u8; 64]; 64]>,
    /// The last vector the motion search found over each 8x8 block, per
    /// reference (`(ref_frame - LAST_FRAME) * cells + cell`): starting
    /// points for the searches of the blocks around and above it.
    pub(crate) me_hints: Vec<Mv>,
    /// Adaptive quantisation: each superblock's quantiser, coded as a
    /// `delta_qindex` (the frame has `delta_q_present`).
    pub(crate) aq: Option<std::sync::Arc<AqMap>>,
    /// Intra block copy (the frame has `allow_intrabc`): where each 8x8
    /// of the source's luma lies, by content.
    pub(crate) ibc: Option<std::sync::Arc<IbcHash>>,
}

/// The positions of the source's luma 8x8 blocks (every sample position
/// one fits at), by a hash of their samples: an intra block copy's
/// candidates are the earlier places a block's content appears.
#[derive(Debug)]
pub(crate) struct IbcHash {
    /// Hash to positions (`y * width + x`), ascending (raster order).
    map: std::collections::HashMap<u32, Vec<u32>>,
    width: usize,
}

impl IbcHash {
    /// The hash of the 8x8 block of `src` (row stride `stride`) at
    /// `(x, y)`, from the row hashes `rows` (`rows[y * w + x]`, of the 8
    /// samples from `(x, y)`).
    fn combine(rows: &[u32], w: usize, x: usize, y: usize) -> u32 {
        let mut h = 0u32;
        for i in 0..8 {
            h = h
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(rows[(y + i) * w + x]);
        }
        h
    }

    /// The table of a `w` x `h` luma plane `src` (row stride `stride`).
    pub(crate) fn new(src: &[u16], stride: usize, w: usize, h: usize) -> Self {
        let mut map: std::collections::HashMap<u32, Vec<u32>> = Default::default();
        if w >= 8 && h >= 8 {
            let (cw, ch) = (w - 7, h - 7);
            let mut rows = vec![0u32; h * cw];
            for y in 0..h {
                let row = &src[y * stride..y * stride + w];
                for x in 0..cw {
                    let mut r = 0u32;
                    for &v in &row[x..x + 8] {
                        r = r.wrapping_mul(0x0100_0193).wrapping_add(v as u32 + 1);
                    }
                    rows[y * cw + x] = r;
                }
            }
            for y in 0..ch {
                for x in 0..cw {
                    map.entry(Self::combine(&rows, cw, x, y))
                        .or_default()
                        .push((y * w + x) as u32);
                }
            }
        }
        IbcHash { map, width: w }
    }

    /// The hash of `src`'s 8x8 block at `(x, y)`.
    fn hash_at(src: &[u16], stride: usize, x: usize, y: usize) -> u32 {
        let mut rows = [0u32; 8];
        for (i, r) in rows.iter_mut().enumerate() {
            for &v in &src[(y + i) * stride + x..(y + i) * stride + x + 8] {
                *r = r.wrapping_mul(0x0100_0193).wrapping_add(v as u32 + 1);
            }
        }
        Self::combine(&rows, 1, 0, 0)
    }
}

/// Each superblock's quantiser index and the Lagrange multipliers that go
/// with it (`EncCtx::lambda`, `EncCtx::rd_lambda`), in raster order.
#[derive(Debug)]
pub(crate) struct AqMap {
    pub(crate) sb_cols: usize,
    pub(crate) q: Vec<u8>,
    pub(crate) lambda: Vec<(f64, f64)>,
}

impl EncCtx {
    fn src(&self, plane: usize, x: usize, y: usize) -> i32 {
        self.src[plane][y * self.stride[plane] + x] as i32
    }
}

/// Approximate bits of a motion vector difference component:
/// `3 + 2 log2(|d| / 2 + 1)` (a piecewise-linear log2), 1 for none.
fn mv_bits(d: i32) -> f64 {
    if d == 0 {
        1.0
    } else {
        let v = d.unsigned_abs() + 2;
        let e = 31 - v.leading_zeros();
        let frac = (v - (1 << e)) as f64 / (1u32 << e) as f64;
        1.0 + 2.0 * (e as f64 + frac)
    }
}

/// The eight neighbours of a point, as `(dy, dx)`.
const SQUARE8: [(i32, i32); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

thread_local! {
    /// Scratch for inter predictions.
    static PRED: std::cell::RefCell<[Vec<i32>; 2]> = const { std::cell::RefCell::new([Vec::new(), Vec::new()]) };
}

/// A transform block's levels as [`forward_quantise`] gave them, kept for
/// when the same residual is quantised again with the same transform and
/// quantiser: the search codes a block many times over (each candidate
/// that holds it, then the winner again for real, at every level of the
/// partition search above it), and over half of the transforms at the
/// slower speeds are of residuals already transformed. The levels are a
/// function of the residual, the size and type and the quantiser alone,
/// all of which the entry holds, so they are the same whether they are
/// worked out again or not.
#[derive(Default)]
struct QEntry {
    /// Size, type, and the quantiser's `ac_r`, `dc_r` and dead zone (as
    /// bits).
    key: Option<(u8, u8, u64, u64, u64)>,
    res: Vec<i32>,
    levels: Vec<i32>,
    /// The coefficients, `tw` a row.
    coefs: Vec<f64>,
    eob: usize,
}

/// The entries of the cache, picked by where the block lies (a block
/// coded again is mostly coded again soon). Blocks of the 64-sample
/// transforms are not kept: an entry is then 8 KB at most.
const QCACHE_SLOTS: usize = 2048;

thread_local! {
    static QCACHE: std::cell::RefCell<Vec<QEntry>> = std::cell::RefCell::new(
        (0..QCACHE_SLOTS).map(|_| QEntry::default()).collect()
    );
}

/// The luma intra modes the encoder tries, with a rough cost in bits for
/// the greedy (non-searching) decision.
const Y_MODES: [(usize, f64); 13] = [
    (DC_PRED, 1.0),
    (V_PRED, 2.5),
    (H_PRED, 2.5),
    (SMOOTH_PRED, 2.5),
    (PAETH_PRED, 3.0),
    (D45_PRED, 4.0),
    (D135_PRED, 4.0),
    (SMOOTH_V_PRED, 4.0),
    (SMOOTH_H_PRED, 4.0),
    (D67_PRED, 4.5),
    (D113_PRED, 4.5),
    (D157_PRED, 4.5),
    (D203_PRED, 4.5),
];

impl TileDecoder<'_, '_> {
    fn enc(&self) -> &EncCtx {
        self.enc.as_ref().expect("encode mode")
    }

    /// The level the encoder chose at `pos` (0 when decoding).
    #[inline]
    pub(crate) fn enc_level(&self, pos: usize) -> u32 {
        self.enc.as_ref().map_or(0, |e| e.coefs[pos].unsigned_abs())
    }

    /// The signed level the encoder chose at `pos` (0 when decoding).
    #[inline]
    pub(crate) fn enc_coef(&self, pos: usize) -> i32 {
        self.enc.as_ref().map_or(0, |e| e.coefs[pos])
    }

    fn qstep(&self) -> f64 {
        let bdi = ((self.f.bit_depth - 8) >> 1) as usize;
        AC_QLOOKUP[bdi][self.f.hdr.base_q_idx as usize] as f64
            / (1 << (self.f.bit_depth - 8)) as f64
    }

    /// The partition of a block (encode mode), before the syntax codes it.
    pub(crate) fn enc_partition(
        &mut self,
        r: usize,
        c: usize,
        b_size: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<usize> {
        if self.enc().tools.rdo {
            return self.enc_partition_rd(r, c, b_size, has_rows, has_cols);
        }
        if !(has_rows && has_cols) {
            return Ok(PARTITION_SPLIT);
        }
        Ok(self.enc_partition_greedy(r, c, b_size))
    }

    /// The partition of a block whose four quadrants are all inside the
    /// frame, without search: none or split (split when the source has
    /// more detail than the quantiser keeps).
    fn enc_partition_greedy(&mut self, r: usize, c: usize, b_size: usize) -> usize {
        if b_size > BLOCK_32X32 {
            return PARTITION_SPLIT;
        }
        if b_size <= BLOCK_8X8 {
            return PARTITION_NONE;
        }
        let n = block_width(b_size);
        let (x0, y0) = (c * MI_SIZE, r * MI_SIZE);
        let e = self.enc();
        let var = |x: usize, y: usize, s: usize| -> f64 {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for i in 0..s {
                for j in 0..s {
                    let v = e.src(0, x + j, y + i) as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let a = (s * s) as f64;
            sq / a - (sum / a) * (sum / a)
        };
        let whole = var(x0, y0, n);
        let h = n / 2;
        let quads =
            (var(x0, y0, h) + var(x0 + h, y0, h) + var(x0, y0 + h, h) + var(x0 + h, y0 + h, h))
                / 4.0;
        let q = self.qstep();
        if whole > q * q * 0.6 && (quads < whole * 0.75 || whole > q * q * 4.0) {
            PARTITION_SPLIT
        } else {
            PARTITION_NONE
        }
    }

    /// The modes of the block about to be coded (encode mode): sets `plan`.
    pub(crate) fn enc_decide_block(&mut self) -> Result<()> {
        // The block's `Lossless` (its segment's; the encoder codes no
        // segmentation), which mode info would only set after the decision.
        self.b.lossless = self.f.hdr.lossless_array[0];
        if self.enc().tools.rdo {
            return self.enc_decide_block_rd();
        }
        self.enc_decide_block_greedy();
        Ok(())
    }

    /// The prediction arguments `transform_block()` will use for the first
    /// transform block of `plane` at the block's largest transform size:
    /// `(x, y, haveLeft, haveAbove, haveAboveRight, haveBelowLeft, log2W,
    /// log2H, txSz)`.
    #[allow(clippy::type_complexity)]
    fn intra_args(
        &self,
        plane: usize,
    ) -> (usize, usize, bool, bool, bool, bool, usize, usize, usize) {
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let tx_sz = if self.b.lossless {
            TX_4X4
        } else if plane == 0 {
            MAX_TX_SIZE_RECT[self.b.mi_size]
        } else {
            let uv = MAX_TX_SIZE_RECT[self.f.plane_residual_size(self.b.mi_size, plane)];
            match (TX_WIDTH[uv], TX_HEIGHT[uv]) {
                (64, _) | (_, 64) => TX_32X32,
                _ => uv,
            }
        };
        let x = (self.b.mi_col >> sub_x) * MI_SIZE;
        let y = (self.b.mi_row >> sub_y) * MI_SIZE;
        let sb_mask = if self.f.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let row = (y << sub_y) >> MI_SIZE_LOG2;
        let col = (x << sub_x) >> MI_SIZE_LOG2;
        let sbr = (row & sb_mask) >> sub_y;
        let sbc = (col & sb_mask) >> sub_x;
        let step_x = TX_WIDTH[tx_sz] >> 2;
        let step_y = TX_HEIGHT[tx_sz] >> 2;
        let (al, au) = if plane == 0 {
            (self.b.avail_l, self.b.avail_u)
        } else {
            (self.b.avail_l_chroma, self.b.avail_u_chroma)
        };
        let ar = self.block_decoded(plane, sbr as isize - 1, (sbc + step_x) as isize);
        let bl = self.block_decoded(plane, (sbr + step_y) as isize, sbc as isize - 1);
        (
            x,
            y,
            al,
            au,
            ar,
            bl,
            TX_WIDTH_LOG2[tx_sz],
            TX_HEIGHT_LOG2[tx_sz],
            tx_sz,
        )
    }

    /// At the start of superblock `(r, c)`: its multipliers, under adaptive
    /// quantisation.
    pub(crate) fn enc_begin_sb(&mut self, r: usize, c: usize) {
        let e = self.enc.as_mut().expect("encode mode");
        if let Some(aq) = e.aq.as_ref() {
            let i = (r >> 4) * aq.sb_cols + (c >> 4);
            (e.lambda, e.rd_lambda) = aq.lambda[i];
        }
    }

    /// The `delta_qindex` (in units of `delta_q_res`) that takes the
    /// quantiser to this superblock's.
    pub(crate) fn enc_delta_qindex(&self) -> i32 {
        let Some(aq) = self.enc().aq.as_ref() else {
            return 0;
        };
        let i = (self.b.mi_row >> 4) * aq.sb_cols + (self.b.mi_col >> 4);
        let diff = aq.q[i] as i32 - self.current_q_index;
        let res = self.f.hdr.delta_q_res;
        (diff as f64 / (1 << res) as f64).round() as i32
    }

    /// SATD between the source and `CurrFrame` over a region.
    fn region_satd(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
        let e = self.enc();
        let cur = &self.f.cur.planes[plane];
        let st = e.stride[plane];
        satd(
            &e.src[plane][y * st + x..],
            st,
            &cur.data[y * cur.stride + x..],
            cur.stride,
            w.max(4),
            h.max(4),
        )
    }

    /// Predicts `plane` with intra `mode` as `transform_block()` would and
    /// returns the SATD against the source.
    fn try_intra(&mut self, plane: usize, mode: usize) -> u64 {
        let (x, y, al, au, ar, bl, lw, lh, _) = self.intra_args(plane);
        self.predict_intra(plane, x, y, al, au, ar, bl, mode, lw, lh);
        self.region_satd(plane, x, y, 1 << lw, 1 << lh)
    }

    /// The cost in bits of coding luma intra mode `m` for this block, from
    /// the CDFs in force.
    fn y_mode_bits(&self, m: usize) -> f64 {
        let cost = if self.f.hdr.frame_is_intra {
            let above = if self.b.avail_u {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row - 1, self.b.mi_col).y_mode as usize]
            } else {
                0
            };
            let left = if self.b.avail_l {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row, self.b.mi_col - 1).y_mode as usize]
            } else {
                0
            };
            Coder::cost(&self.intra_frame_y_mode_cdf[above][left], m)
        } else {
            Coder::cost(&self.cdf.y_mode[SIZE_GROUP[self.b.mi_size]], m)
        };
        cost as f64 / 256.0
    }

    /// The candidates the rate-distortion search trials for this block:
    /// the best intra modes by SATD (plus their rate), then the best inter
    /// modes and vectors per reference.
    pub(crate) fn block_candidates(&mut self) -> Vec<Plan> {
        let lambda = self.enc().lambda;
        let tools = self.enc().tools;
        let ms = self.b.mi_size;
        self.b.use_filter_intra = false;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        self.b.is_inter = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        // Each luma mode (and filter intra mode: `FILTER_INTRA_MODES` past
        // the plain ones), by SATD and rate.
        let mut ys: Vec<(usize, Option<u8>, f64)> = Vec::with_capacity(Y_MODES.len() + 5);
        for &(m, _) in Y_MODES.iter() {
            self.b.y_mode = m;
            let cost = self.try_intra(0, m) as f64 + lambda * self.y_mode_bits(m);
            ys.push((m, None, cost));
        }
        if self.f.seq.enable_filter_intra
            && block_width(ms).max(block_height(ms)) <= 32
            && !self.b.lossless
        {
            // `use_filter_intra` is coded for every DC_PRED block: its
            // cost is in the plain DC_PRED's rate too.
            let dc_bits = self.y_mode_bits(DC_PRED);
            let off = Coder::cost(&self.cdf.filter_intra[ms], 0) as f64 / 256.0;
            let on = Coder::cost(&self.cdf.filter_intra[ms], 1) as f64 / 256.0;
            if let Some(dc) = ys.iter_mut().find(|y| y.0 == DC_PRED) {
                dc.2 += lambda * off;
            }
            self.b.y_mode = DC_PRED;
            self.b.use_filter_intra = true;
            for fm in 0..5u8 {
                self.b.filter_intra_mode = fm as usize;
                let mode_bits =
                    Coder::cost(&self.cdf.filter_intra_mode, fm as usize) as f64 / 256.0;
                let cost = self.try_intra(0, DC_PRED) as f64 + lambda * (dc_bits + on + mode_bits);
                ys.push((DC_PRED, Some(fm), cost));
            }
            self.b.use_filter_intra = false;
        }
        ys.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut best_uv = DC_PRED;
        if self.b.has_chroma {
            self.b.y_mode = ys[0].0;
            let mut best = f64::MAX;
            for &m in &[DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, PAETH_PRED] {
                self.b.uv_mode = m;
                let cost = (self.try_intra(1, m) + self.try_intra(2, m)) as f64
                    + lambda * if m == DC_PRED { 1.0 } else { 3.0 };
                if cost < best {
                    best = cost;
                    best_uv = m;
                }
            }
        }
        let luma = |y: &(usize, Option<u8>, f64)| Plan {
            y_mode: y.0,
            use_filter_intra: y.1.is_some(),
            filter_intra_mode: y.1.unwrap_or(0),
            ref_frame: INTRA_FRAME,
            ..Default::default()
        };
        let mut out: Vec<(Plan, f64)> = Vec::new();
        // The best by SATD, filter intra modes among them (measured against
        // trialling the best filter intra mode as well as the best plain
        // ones: -0.20 % against -0.10 % BD-rate PSNR-Y at speed 6).
        for y in ys.iter().take(tools.intra_candidates.max(1) as usize) {
            out.push((
                Plan {
                    uv_mode: best_uv,
                    ..luma(y)
                },
                y.2,
            ));
        }
        // A palette, for blocks of few colours (screen content).
        if self.f.hdr.allow_screen_content_tools
            && tools.palette
            && (BLOCK_8X8..=BLOCK_64X64).contains(&ms)
            && block_width(ms) <= 64
            && block_height(ms) <= 64
            && let Some((n, colors)) = self.palette_colors()
        {
            out.push((
                Plan {
                    y_mode: DC_PRED,
                    uv_mode: best_uv,
                    ref_frame: INTRA_FRAME,
                    palette_n: n,
                    palette_colors: colors,
                    ..Default::default()
                },
                ys[0].2,
            ));
        }
        // Chroma from luma, with the best luma mode.
        if tools.cfl
            && let Some((au, av)) = self.cfl_alphas()
        {
            out.push((
                Plan {
                    uv_mode: UV_CFL_PRED,
                    cfl_alpha_u: au,
                    cfl_alpha_v: av,
                    ..luma(&ys[0])
                },
                ys[0].2,
            ));
        }
        // Intra block copy, where the frame allows it.
        if self.f.hdr.allow_intrabc {
            out.extend(self.intrabc_candidates());
        }
        let refs = self.enc().refs.clone();
        if refs.is_empty() || ms < BLOCK_8X8 {
            return out.into_iter().map(|x| x.0).collect();
        }
        let mut inter: Vec<(Plan, f64)> = Vec::new();
        // The searched vector of each reference, for compound candidates.
        let mut searched_mv: Vec<(i32, Mv)> = Vec::new();
        for &rf in &refs {
            self.b.ref_frame = [rf, NONE];
            self.b.is_inter = true;
            self.b.use_intrabc = false;
            self.find_mv_stack(false);
            let n = self.b.num_mv_found;
            let mut cands: Vec<(usize, usize, Mv, f64)> = Vec::new();
            cands.push((NEARESTMV, 0, self.b.ref_stack_mv[0][0], 2.0));
            for idx in 1..n.min(3) {
                cands.push((NEARMV, idx, self.b.ref_stack_mv[idx][0], 3.0 + idx as f64));
            }
            cands.push((GLOBALMV, 0, self.b.global_mvs[0], 2.0));
            let pred_mv = self.b.ref_stack_mv[0][0];
            let searched = self.motion_search(rf, pred_mv);
            searched_mv.push((rf, searched));
            let bits = 3.0 + mv_bits(searched[0] - pred_mv[0]) + mv_bits(searched[1] - pred_mv[1]);
            cands.push((NEWMV, 0, searched, bits));
            for (mode, idx, mv, bits) in cands {
                if inter
                    .iter()
                    .any(|(p, _)| p.ref_frame == rf && p.mv[0] == mv && p.y_mode != NEWMV)
                    && mode != NEWMV
                {
                    continue;
                }
                let cost = self.inter_satd(rf, mv) as f64 + lambda * bits;
                inter.push((
                    Plan {
                        is_inter: true,
                        y_mode: mode,
                        ref_frame: rf,
                        ref_mv_idx: idx,
                        mv: [mv, [0, 0]],
                        ..Default::default()
                    },
                    cost,
                ));
            }
        }
        // Compound: LAST with each other reference, averaged.
        if tools.compound
            && self.f.hdr.reference_select
            && NUM_4X4_BLOCKS_WIDE[ms].min(NUM_4X4_BLOCKS_HIGH[ms]) >= 2
        {
            for &(r2, mv2) in searched_mv.iter().skip(1) {
                let Some(&(_, mv1)) = searched_mv.first() else {
                    break;
                };
                self.b.ref_frame = [LAST_FRAME, r2];
                self.b.is_inter = true;
                self.find_mv_stack(true);
                let nearest = self.b.ref_stack_mv[0];
                let global = self.b.global_mvs;
                for (mode, mvs, bits) in [
                    (NEAREST_NEARESTMV, nearest, 3.0),
                    (GLOBAL_GLOBALMV, global, 4.0),
                    (
                        NEW_NEWMV,
                        [mv1, mv2],
                        6.0 + mv_bits(mv1[0] - nearest[0][0])
                            + mv_bits(mv1[1] - nearest[0][1])
                            + mv_bits(mv2[0] - nearest[1][0])
                            + mv_bits(mv2[1] - nearest[1][1]),
                    ),
                ] {
                    let cost =
                        self.compound_satd(LAST_FRAME, mvs[0], r2, mvs[1]) as f64 + lambda * bits;
                    inter.push((
                        Plan {
                            is_inter: true,
                            y_mode: mode,
                            ref_frame: LAST_FRAME,
                            ref_frame2: r2,
                            ref_mv_idx: 0,
                            mv: mvs,
                            ..Default::default()
                        },
                        cost,
                    ));
                }
            }
        }
        self.b.is_inter = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        inter.sort_by(|a, b| a.1.total_cmp(&b.1));
        // Skip mode, where the frame has it: the cheapest of blocks, always
        // trialled.
        let skip_mode = (self.f.hdr.skip_mode_present
            && block_width(ms) >= 8
            && block_height(ms) >= 8)
            .then(|| Plan {
                skip: true,
                skip_mode: true,
                is_inter: true,
                y_mode: NEAREST_NEARESTMV,
                ref_frame: self.f.hdr.skip_mode_frame[0],
                ref_frame2: self.f.hdr.skip_mode_frame[1],
                ..Default::default()
            });
        // Intra is worth a trial only when its SATD is near the best inter's.
        let best_inter = inter.first().map_or(f64::MAX, |x| x.1);
        let intra_keep = out
            .into_iter()
            .filter(|(_, score)| *score < best_inter * if tools.prune_intra { 1.0 } else { 1.5 })
            .map(|(p, _)| p);
        let mut all: Vec<Plan> = inter
            .iter()
            .take(tools.inter_candidates.max(1) as usize)
            .map(|x| x.0)
            .collect();
        all.extend(skip_mode);
        all.extend(intra_keep);
        all
    }

    /// Whether an intra block copy of this block by `dv` (in eighths, whole
    /// samples) is allowed: `is_mv_valid` (6.10.24) — inside the tile, from
    /// superblocks decoded at least `INTRABC_DELAY_SB64` 64-wide columns
    /// before, along the wavefront it allows — and, when the search runs
    /// in a wavefront of superblock rows two superblocks apart
    /// (`Tools::wavefront`), only from what that has reconstructed.
    fn intrabc_valid(&self, dv: Mv) -> bool {
        if dv[0].abs() >= 1 << 14 || dv[1].abs() >= 1 << 14 || dv[0] & 7 != 0 || dv[1] & 7 != 0 {
            return false;
        }
        let ms = self.b.mi_size;
        let (bw, bh) = (block_width(ms) as i32, block_height(ms) as i32);
        let mut top = (self.b.mi_row * MI_SIZE) as i32 + (dv[0] >> 3);
        let mut left = (self.b.mi_col * MI_SIZE) as i32 + (dv[1] >> 3);
        let bottom = top + bh;
        let right = left + bw;
        if self.b.has_chroma {
            if bw < 8 && self.f.ssx != 0 {
                left -= 4;
            }
            if bh < 8 && self.f.ssy != 0 {
                top -= 4;
            }
        }
        let mi = MI_SIZE as i32;
        if top < self.mi_row_start as i32 * mi
            || left < self.mi_col_start as i32 * mi
            || bottom > self.mi_row_end as i32 * mi
            || right > self.mi_col_end as i32 * mi
        {
            return false;
        }
        let sb128 = self.f.seq.use_128x128_superblock;
        let sb_h = if sb128 { 128 } else { 64 };
        let active_row = (self.b.mi_row * MI_SIZE) as i32 / sb_h;
        let active_col = (self.b.mi_col * MI_SIZE) as i32 >> 6;
        let src_row = (bottom - 1) / sb_h;
        let src_col = (right - 1) >> 6;
        let per_row = ((self.mi_col_end - self.mi_col_start - 1) >> 4) as i32 + 1;
        if src_row * per_row + src_col >= active_row * per_row + active_col - INTRABC_DELAY_SB64 {
            return false;
        }
        let gradient = 1 + INTRABC_DELAY_SB64 + sb128 as i32;
        let wf = gradient * (active_row - src_row);
        if src_row > active_row || src_col >= active_col - INTRABC_DELAY_SB64 + wf {
            return false;
        }
        // The search's wavefront: a row two superblocks behind the one
        // above it has the columns before its own plus one reconstructed
        // there, two more for each row further up.
        !self.enc().tools.wavefront
            || src_row == active_row
            || src_col < active_col + 2 * (active_row - src_row) - 1
    }

    /// The intra block copy candidates of this block (8x8 or more): the
    /// vectors the stack predicts, and the earlier places in the frame
    /// where the block's source appears exactly (found by the hash of its
    /// first 8x8, the nearest first), each that is allowed scored by SATD
    /// against the reconstruction there and its vector's rate; the best
    /// two.
    fn intrabc_candidates(&mut self) -> Vec<(Plan, f64)> {
        let ms = self.b.mi_size;
        let (bw, bh) = (block_width(ms), block_height(ms));
        if bw < 8 || bh < 8 {
            return Vec::new();
        }
        let saved = (self.b.use_intrabc, self.b.is_inter, self.b.ref_frame);
        self.b.use_intrabc = true;
        self.b.is_inter = true;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        self.find_mv_stack(false);
        let pred = self.intrabc_pred_mv();
        let mut dvs: Vec<Mv> = vec![pred, self.b.ref_stack_mv[0][0], self.b.ref_stack_mv[1][0]];
        let (x0, y0) = (self.b.mi_col * MI_SIZE, self.b.mi_row * MI_SIZE);
        if let Some(hash) = self.enc().ibc.clone() {
            let e = self.enc();
            let (src, st) = (&e.src[0], e.stride[0]);
            let key = IbcHash::hash_at(src, st, x0, y0);
            if let Some(list) = hash.map.get(&key) {
                // The positions before this one in raster order, nearest
                // first.
                let here = (y0 * hash.width + x0) as u32;
                let end = list.partition_point(|&p| p < here);
                let mut found = 0;
                for &p in list[..end].iter().rev().take(64) {
                    let (px, py) = (p as usize % hash.width, p as usize / hash.width);
                    let same = (0..bh).all(|i| {
                        src[(py + i) * st + px..(py + i) * st + px + bw]
                            == src[(y0 + i) * st + x0..(y0 + i) * st + x0 + bw]
                    });
                    let dv = [(py as i32 - y0 as i32) * 8, (px as i32 - x0 as i32) * 8];
                    if same && self.intrabc_valid(dv) {
                        dvs.push(dv);
                        found += 1;
                        if found == 4 {
                            break;
                        }
                    }
                }
            }
        }
        let lambda = self.enc().lambda;
        let mut scored: Vec<(Mv, f64)> = Vec::new();
        for dv in dvs {
            if scored.iter().any(|s| s.0 == dv) || !self.intrabc_valid(dv) {
                continue;
            }
            let e = self.enc();
            let cur = &self.f.cur.planes[0];
            let st = e.stride[0];
            let sy = (y0 as i32 + (dv[0] >> 3)) as usize;
            let sx = (x0 as i32 + (dv[1] >> 3)) as usize;
            let cost = satd(
                &e.src[0][y0 * st + x0..],
                st,
                &cur.data[sy * cur.stride + sx..],
                cur.stride,
                bw,
                bh,
            ) as f64;
            let bits = 2.0 + mv_bits((dv[0] - pred[0]) >> 3) + mv_bits((dv[1] - pred[1]) >> 3);
            scored.push((dv, cost + lambda * bits));
        }
        (self.b.use_intrabc, self.b.is_inter, self.b.ref_frame) = saved;
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored
            .into_iter()
            .take(2)
            .map(|(dv, cost)| {
                (
                    Plan {
                        use_intrabc: true,
                        is_inter: true,
                        y_mode: DC_PRED,
                        uv_mode: DC_PRED,
                        ref_frame: INTRA_FRAME,
                        mv: [dv, [0, 0]],
                        ..Default::default()
                    },
                    cost,
                )
            })
            .collect()
    }

    /// A luma palette for this block: its distinct colours when there are
    /// at most eight, else eight found by k-means; `None` for a flat block
    /// or one without sharp edges (where a palette does not pay).
    fn palette_colors(&self) -> Option<(u8, [u16; 8])> {
        let ms = self.b.mi_size;
        let (w, h) = (block_width(ms), block_height(ms));
        let x0 = self.b.mi_col * MI_SIZE;
        let y0 = self.b.mi_row * MI_SIZE;
        let e = self.enc();
        let mut vals: Vec<u16> = Vec::with_capacity(w * h);
        for i in 0..h {
            for j in 0..w {
                vals.push(e.src(0, x0 + j, y0 + i) as u16);
            }
        }
        let mut distinct = vals.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 2 {
            return None;
        }
        let shift = self.f.bit_depth - 8;
        if (distinct[distinct.len() - 1] - distinct[0]) >> shift < 48 {
            return None;
        }
        let mut colors: Vec<u16> = if distinct.len() <= 8 {
            distinct
        } else {
            // k-means, seeded across the range.
            let k = 8usize;
            let (lo, hi) = (distinct[0] as f64, distinct[distinct.len() - 1] as f64);
            let mut c: Vec<f64> = (0..k)
                .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / k as f64)
                .collect();
            for _ in 0..8 {
                let mut sum = vec![0f64; k];
                let mut cnt = vec![0usize; k];
                for &v in &vals {
                    let i = (0..k)
                        .min_by(|&a, &b| {
                            (v as f64 - c[a]).abs().total_cmp(&(v as f64 - c[b]).abs())
                        })
                        .unwrap_or(0);
                    sum[i] += v as f64;
                    cnt[i] += 1;
                }
                for i in 0..k {
                    if cnt[i] > 0 {
                        c[i] = sum[i] / cnt[i] as f64;
                    }
                }
            }
            let mut out: Vec<u16> = c.iter().map(|v| v.round() as u16).collect();
            out.sort_unstable();
            out.dedup();
            out
        };
        colors.truncate(8);
        if colors.len() < 2 {
            return None;
        }
        let mut arr = [0u16; 8];
        arr[..colors.len()].copy_from_slice(&colors);
        Some((colors.len() as u8, arr))
    }

    /// The palette indices of the block about to be coded: each sample's
    /// nearest colour (encode mode; into `EncCtx::palette_map`).
    pub(crate) fn enc_palette_map(&mut self, onscreen_width: usize, onscreen_height: usize) {
        let n = self.b.palette_size_y;
        let colors = self.b.palette_colors_y;
        let x0 = self.b.mi_col * MI_SIZE;
        let y0 = self.b.mi_row * MI_SIZE;
        let e = self.enc.as_mut().expect("encode mode");
        for i in 0..onscreen_height {
            for j in 0..onscreen_width {
                let v = e.src[0][(y0 + i) * e.stride[0] + x0 + j] as i32;
                let mut best = (i32::MAX, 0u8);
                for (k, &c) in colors.iter().take(n).enumerate() {
                    let d = (v - c as i32).abs();
                    if d < best.0 {
                        best = (d, k as u8);
                    }
                }
                e.palette_map[i][j] = best.1;
            }
        }
    }

    /// The chroma-from-luma alphas for this block, fitted by least squares
    /// to the source (the luma reconstruction is not there yet), or `None`
    /// when chroma from luma is not allowed or would be flat.
    fn cfl_alphas(&self) -> Option<(i32, i32)> {
        let ms = self.b.mi_size;
        if !self.b.has_chroma
            || self.b.lossless
            || block_width(ms).max(block_height(ms)) > 32
            || self.f.num_planes < 3
        {
            return None;
        }
        let (ssx, ssy) = (self.f.ssx, self.f.ssy);
        let cs = self.f.plane_residual_size(ms, 1);
        let (w, h) = (block_width(cs), block_height(cs));
        let x0 = (self.b.mi_col >> ssx) * MI_SIZE;
        let y0 = (self.b.mi_row >> ssy) * MI_SIZE;
        let e = self.enc();
        let mut l = vec![0i64; w * h];
        let mut sum = 0i64;
        for i in 0..h {
            for j in 0..w {
                let (lx, ly) = ((x0 + j) << ssx, (y0 + i) << ssy);
                let mut t = 0i64;
                for dy in 0..=ssy {
                    for dx in 0..=ssx {
                        t += e.src(0, lx + dx, ly + dy) as i64;
                    }
                }
                let v = t << (3 - ssx - ssy);
                l[i * w + j] = v;
                sum += v;
            }
        }
        let avg = sum / (w * h) as i64;
        let den: i64 = l.iter().map(|&v| (v - avg) * (v - avg)).sum();
        if den == 0 {
            return None;
        }
        let mut alphas = [0i32; 2];
        for (k, plane) in [1usize, 2].into_iter().enumerate() {
            let mut csum = 0i64;
            for i in 0..h {
                for j in 0..w {
                    csum += e.src(plane, x0 + j, y0 + i) as i64;
                }
            }
            let cavg = csum / (w * h) as i64;
            let mut num = 0i64;
            for i in 0..h {
                for j in 0..w {
                    num += (e.src(plane, x0 + j, y0 + i) as i64 - cavg) * (l[i * w + j] - avg);
                }
            }
            alphas[k] = ((64 * num) as f64 / den as f64).round().clamp(-16.0, 16.0) as i32;
        }
        if alphas == [0, 0] {
            return None;
        }
        Some((alphas[0], alphas[1]))
    }

    /// Chooses the block's modes without search (`plan`), before
    /// `mode_info()` codes them.
    fn enc_decide_block_greedy(&mut self) {
        let lambda = self.enc().lambda;
        let ms = self.b.mi_size;
        self.plan = Default::default();
        self.b.use_filter_intra = false;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        let mut best_y = (DC_PRED, f64::MAX);
        for &(m, bits) in Y_MODES.iter().take(10) {
            self.b.y_mode = m;
            let cost = self.try_intra(0, m) as f64 + lambda * bits;
            if cost < best_y.1 {
                best_y = (m, cost);
            }
        }
        self.b.y_mode = best_y.0;
        let mut best_uv = DC_PRED;
        if self.b.has_chroma {
            let mut best = f64::MAX;
            for &m in &[DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, PAETH_PRED] {
                self.b.uv_mode = m;
                let cost = (self.try_intra(1, m) + self.try_intra(2, m)) as f64
                    + lambda * if m == DC_PRED { 1.0 } else { 3.0 };
                if cost < best {
                    best = cost;
                    best_uv = m;
                }
            }
        }
        self.plan.y_mode = best_y.0;
        self.plan.uv_mode = best_uv;
        self.plan.tx_type = DCT_DCT;
        // Inter: the best of NEWMV (searched), NEARESTMV and GLOBALMV on
        // LAST_FRAME, for blocks of 8x8 and up.
        let mut inter_cost = f64::MAX;
        if !self.enc().refs.is_empty() && ms >= BLOCK_8X8 {
            self.b.ref_frame = [LAST_FRAME, NONE];
            self.b.is_inter = true;
            self.b.use_intrabc = false;
            self.find_mv_stack(false);
            let pred_mv = self.b.ref_stack_mv[0][0];
            let nearest = self.b.ref_stack_mv[0][0];
            let global = self.b.global_mvs[0];
            let mut cands: Vec<(usize, Mv, f64)> =
                vec![(NEARESTMV, nearest, 2.0), (GLOBALMV, global, 2.0)];
            let searched = self.motion_search(LAST_FRAME, pred_mv);
            let bits = 3.0 + mv_bits(searched[0] - pred_mv[0]) + mv_bits(searched[1] - pred_mv[1]);
            cands.push((NEWMV, searched, bits));
            let mut best = (GLOBALMV, global, f64::MAX);
            for (mode, mv, bits) in cands {
                let cost = self.inter_satd(LAST_FRAME, mv) as f64 + lambda * bits;
                if cost < best.2 {
                    best = (mode, mv, cost);
                }
            }
            inter_cost = best.2;
            self.plan.y_mode = best.0;
            self.plan.mv[0] = best.1;
            self.plan.ref_frame = LAST_FRAME;
            self.plan.ref_mv_idx = 0;
            self.b.is_inter = false;
            self.b.ref_frame = [INTRA_FRAME, NONE];
        }
        if inter_cost < best_y.1 + lambda * 2.0 {
            self.plan.is_inter = true;
            self.plan.skip = self.inter_residual_is_zero(self.plan.mv[0]);
        } else {
            self.plan.is_inter = false;
            self.plan.y_mode = best_y.0;
            self.plan.skip = self.intra_residual_is_zero(best_y.0, best_uv);
        }
    }

    /// Whether every transform block of the block, predicted with these
    /// intra modes, quantises to zero.
    fn intra_residual_is_zero(&mut self, y_mode: usize, uv_mode: usize) -> bool {
        let planes = if self.b.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let mode = if plane == 0 { y_mode } else { uv_mode };
            if plane == 0 {
                self.b.y_mode = y_mode;
            } else {
                self.b.uv_mode = uv_mode;
            }
            let (x, y, al, au, ar, bl, lw, lh, tx_sz) = self.intra_args(plane);
            self.predict_intra(plane, x, y, al, au, ar, bl, mode, lw, lh);
            let eob = if plane == 0 {
                self.choose_luma_tx_type(x, y, tx_sz)
            } else {
                let t = self.chroma_intra_tx_type(tx_sz, uv_mode);
                self.quantise_block(plane, x, y, tx_sz, t)
            };
            if eob > 0 {
                return false;
            }
        }
        true
    }

    /// Chooses the luma transform type of an intra block (its prediction in
    /// place) among the DCT and ADST combinations the reduced intra set
    /// allows, by distortion plus estimated rate on the quantised
    /// reconstruction; sets `plan.tx_type` and returns the end of block.
    fn choose_luma_tx_type(&mut self, x: usize, y: usize, tx_sz: usize) -> usize {
        let choice =
            TX_SIZE_SQR_UP[tx_sz] <= TX_16X16 && self.f.hdr.base_q_idx > 0 && !self.b.lossless;
        if !choice {
            self.plan.tx_type = DCT_DCT;
            return self.quantise_block(0, x, y, tx_sz, DCT_DCT);
        }
        let mut best = (DCT_DCT, f64::MAX);
        for t in [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST] {
            let cost = self.trial_cost(x, y, tx_sz, t);
            if cost < best.1 {
                best = (t, cost);
            }
        }
        self.plan.tx_type = best.0;
        self.quantise_block(0, x, y, tx_sz, best.0)
    }

    /// Distortion plus lambda times estimated bits of coding the luma
    /// residual at (x, y) with transform type `t`.
    fn trial_cost(&mut self, x: usize, y: usize, tx_sz: usize, t: usize) -> f64 {
        let eob = self.quantise_block(0, x, y, tx_sz, t);
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let tw = w.min(32);
        let th = h.min(32);
        let bd = self.f.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let q = self.f.hdr.base_q_idx as i32;
        let dc_q = DC_QLOOKUP[bdi][clip3(0, 255, q + self.f.hdr.delta_q_y_dc) as usize] as i64;
        let ac_q = AC_QLOOKUP[bdi][q as usize] as i64;
        let denom: i64 = match tx_sz {
            TX_32X32 | TX_16X32 | TX_32X16 => 2,
            _ => 1,
        };
        let enc = self.enc.as_ref().expect("encode mode");
        let mut dq = vec![0i32; 64 * 64];
        let mut bits = 0f64;
        for i in 0..th {
            for j in 0..tw {
                let l = enc.coefs[i * tw + j] as i64;
                if l != 0 {
                    bits += 2.0 + 2.0 * ((l.unsigned_abs() + 1) as f64).log2();
                    let qq = if i == 0 && j == 0 { dc_q } else { ac_q };
                    dq[i * 64 + j] = (l * qq / denom) as i32;
                }
            }
        }
        let mut rec = vec![0i32; w * h];
        if eob > 0 {
            crate::dsp::itx::inverse_transform_2d(&dq, tx_sz, t, false, bd, &mut rec);
        }
        let cur = &self.f.cur.planes[0];
        let mut sse = 0f64;
        for i in 0..h {
            for j in 0..w {
                let res = enc.src(0, x + j, y + i) - cur.get(x + j, y + i) as i32;
                let d = (res - rec[i * w + j]) as f64;
                sse += d * d;
            }
        }
        let step = ac_q as f64 / (8 << (bd - 8)) as f64;
        sse + 0.3 * step * step * (1 << (2 * (bd - 8))) as f64 * bits
    }

    /// The transform type the decoder derives for an intra chroma block.
    fn chroma_intra_tx_type(&self, tx_sz: usize, uv_mode: usize) -> usize {
        if TX_SIZE_SQR_UP[tx_sz] > TX_32X32 || self.b.lossless {
            return DCT_DCT;
        }
        let t = MODE_TO_TXFM[uv_mode];
        let set = if TX_SIZE_SQR_UP[tx_sz] == TX_32X32 {
            0
        } else if self.f.hdr.reduced_tx_set || TX_SIZE_SQR[tx_sz] == TX_16X16 {
            2
        } else {
            1
        };
        if TX_TYPE_IN_SET_INTRA[set][t] != 0 {
            t
        } else {
            DCT_DCT
        }
    }

    /// Whether the block, predicted from LAST_FRAME with `mv`, has an
    /// all-zero residual after quantisation.
    fn inter_residual_is_zero(&mut self, mv: Mv) -> bool {
        let planes = if self.b.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = self.f.plane_ss(plane);
            let x = (self.b.mi_col >> sub_x) * MI_SIZE;
            let y = (self.b.mi_row >> sub_y) * MI_SIZE;
            let w = block_width(self.b.mi_size) >> sub_x;
            let h = block_height(self.b.mi_size) >> sub_y;
            let mut pred = Vec::new();
            self.inter_pred(plane, x, y, w, h, LAST_FRAME, mv, &mut pred);
            let maxv = (1i32 << self.f.bit_depth) - 1;
            let cur = &mut self.f.cur.planes[plane];
            for i in 0..h {
                for j in 0..w {
                    cur.set(x + j, y + i, pred[i * w + j].clamp(0, maxv) as u16);
                }
            }
            let tx_sz = if plane == 0 {
                MAX_TX_SIZE_RECT[self.b.mi_size]
            } else {
                MAX_TX_SIZE_RECT[self.f.plane_residual_size(self.b.mi_size, plane)]
            };
            if self.quantise_block(plane, x, y, tx_sz, DCT_DCT) > 0 {
                return false;
            }
        }
        true
    }

    /// The single-reference prediction from `ref_frame`, unclipped.
    #[allow(clippy::too_many_arguments)]
    fn inter_pred(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        ref_frame: i32,
        mv: Mv,
        pred: &mut Vec<i32>,
    ) {
        let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let r = self.f.refs[ref_idx].as_ref().expect("a reference");
        let (sx, sy, stx, sty) = self.f.scale_mv(
            plane,
            r.upscaled_width as i32,
            r.frame_height as i32,
            x,
            y,
            mv,
        );
        let (ssx, ssy) = self.f.plane_ss(plane);
        let last_x = ((r.upscaled_width as i32 + ssx as i32) >> ssx) - 1;
        let last_y = ((r.frame_height as i32 + ssy as i32) >> ssy) - 1;
        let rv = self.f.rounding_variables(false);
        if pred.len() < w * h {
            pred.resize(w * h, 0);
        }
        let filt = self.f.hdr.interpolation_filter.min(BILINEAR) as u8;
        block_inter_prediction(
            &r.frame().planes[plane],
            last_x,
            last_y,
            sx,
            sy,
            stx,
            sty,
            w,
            h,
            [filt, filt],
            rv,
            &mut pred[..w * h],
        );
    }

    /// Luma SATD of the inter prediction from `ref_frame` with `mv`.
    fn inter_satd(&self, ref_frame: i32, mv: Mv) -> u64 {
        let w = block_width(self.b.mi_size);
        let h = block_height(self.b.mi_size);
        let x = self.b.mi_col * MI_SIZE;
        let y = self.b.mi_row * MI_SIZE;
        PRED.with_borrow_mut(|pred| {
            self.inter_pred(0, x, y, w, h, ref_frame, mv, &mut pred[0]);
            let e = self.enc();
            let st = e.stride[0];
            satd(&e.src[0][y * st + x..], st, &pred[0], w, w, h)
        })
    }

    /// Luma SATD of the average of two single-reference predictions (an
    /// estimate of the compound prediction, for preselection).
    fn compound_satd(&self, r1: i32, mv1: Mv, r2: i32, mv2: Mv) -> u64 {
        let w = block_width(self.b.mi_size);
        let h = block_height(self.b.mi_size);
        let x = self.b.mi_col * MI_SIZE;
        let y = self.b.mi_row * MI_SIZE;
        PRED.with_borrow_mut(|pred| {
            let [p1, p2] = pred;
            self.inter_pred(0, x, y, w, h, r1, mv1, p1);
            self.inter_pred(0, x, y, w, h, r2, mv2, p2);
            for (a, &b) in p1[..w * h].iter_mut().zip(&p2[..w * h]) {
                *a = (*a + b + 1) >> 1;
            }
            let e = self.enc();
            let st = e.stride[0];
            satd(&e.src[0][y * st + x..], st, &p1[..], w, w, h)
        })
    }

    /// The motion search: full-pel, then half- and quarter- (and eighth-)
    /// sample refinement with the normative predictor. Exhaustive in a
    /// window around the predicted and zero vectors, or (`Tools::fast_me`)
    /// descending square steps from the best of the predicted, stacked,
    /// neighbouring and zero vectors.
    fn motion_search(&mut self, ref_frame: i32, pred_mv: Mv) -> Mv {
        let w = block_width(self.b.mi_size);
        let h = block_height(self.b.mi_size);
        let x0 = (self.b.mi_col * MI_SIZE) as i32;
        let y0 = (self.b.mi_row * MI_SIZE) as i32;
        let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let r = self.f.refs[ref_idx].as_ref().expect("a reference");
        let rp = &r.frame().planes[0];
        let last_x = r.upscaled_width as i32 - 1;
        let last_y = r.frame_height as i32 - 1;
        let e = self.enc.as_ref().expect("encode mode");
        let fast = e.tools.fast_me;
        // The 8x8 cells of the hint grid.
        let cols8 = self.f.mi_cols.div_ceil(2);
        let cells = cols8 * self.f.mi_rows.div_ceil(2);
        let sad = |dx: i32, dy: i32, best: u64| -> u64 {
            let inside = x0 + dx >= 0
                && x0 + dx + w as i32 - 1 <= last_x
                && y0 + dy >= 0
                && y0 + dy + h as i32 - 1 <= last_y;
            if inside {
                let st = e.stride[0];
                let ro = (y0 + dy) as usize * rp.stride + (x0 + dx) as usize;
                return sad(
                    &e.src[0][y0 as usize * st + x0 as usize..],
                    st,
                    &rp.data[ro..],
                    rp.stride,
                    w,
                    h,
                    best,
                );
            }
            let mut s = 0u64;
            let inside = x0 + dx >= 0 && x0 + dx + w as i32 - 1 <= last_x;
            for i in 0..h as i32 {
                let ry = (y0 + i + dy).clamp(0, last_y) as usize;
                let row = rp.row(ry);
                let so = (y0 + i) as usize * e.stride[0] + x0 as usize;
                let srow = &e.src[0][so..so + w];
                if inside {
                    let r0 = (x0 + dx) as usize;
                    s += sad(srow, w, &row[r0..r0 + w], w, w, 1, u64::MAX);
                } else {
                    // The samples left of the reference (which read its
                    // first), those inside it, those right of it (its
                    // last).
                    let (wi, rx0) = (w as i32, x0 + dx);
                    let jl = (-rx0).clamp(0, wi) as usize;
                    let jr = (last_x + 1 - rx0).clamp(jl as i32, wi) as usize;
                    let edge = |part: &[u16], v: u16| -> u64 {
                        part.iter()
                            .map(|&a| (a as i32 - v as i32).unsigned_abs() as u64)
                            .sum()
                    };
                    s += edge(&srow[..jl], row[0]) + edge(&srow[jr..], row[last_x as usize]);
                    if jr > jl {
                        let r0 = (rx0 + jl as i32) as usize;
                        s += sad(&srow[jl..jr], w, &row[r0..], w, jr - jl, 1, u64::MAX);
                    }
                }
                if s > best {
                    return u64::MAX;
                }
            }
            s
        };
        let lambda = e.lambda;
        let cost = |dx: i32, dy: i32, best: f64| -> f64 {
            let bits = mv_bits(dy * 8 - pred_mv[0]) + mv_bits(dx * 8 - pred_mv[1]);
            let rate = lambda * bits;
            if rate >= best {
                return f64::MAX;
            }
            let s = sad(dx, dy, (best - rate).min(1e18) as u64);
            if s == u64::MAX {
                f64::MAX
            } else {
                s as f64 + rate
            }
        };
        let range = e.search_range;
        let mut best = (0i32, 0i32, cost(0, 0, f64::MAX));
        let (pcx, pcy) = ((pred_mv[1] + 4) >> 3, (pred_mv[0] + 4) >> 3);
        let c = cost(pcx, pcy, best.2);
        if c < best.2 {
            best = (pcx, pcy, c);
        }
        if fast {
            // Starting points: the stacked vectors and the vectors found
            // over the 8x8 blocks this one covers and borders.
            let mut starts: [(i32, i32); 16] = [(0, 0); 16];
            let mut n = 0;
            for k in 1..self.b.num_mv_found.min(3) {
                let m = self.b.ref_stack_mv[k][0];
                starts[n] = ((m[1] + 4) >> 3, (m[0] + 4) >> 3);
                n += 1;
            }
            if e.me_hints.len() == cells * 7 {
                let base = (ref_frame - LAST_FRAME) as usize * cells;
                let (r8, c8) = (self.b.mi_row / 2, self.b.mi_col / 2);
                let (h8, w8) = (h.div_ceil(8), w.div_ceil(8));
                let rows8 = self.f.mi_rows.div_ceil(2);
                for (dr, dc) in [
                    (0, 0),
                    (h8 / 2, w8 / 2),
                    (0, w8 - 1),
                    (h8 - 1, 0),
                    (h8 - 1, w8 - 1),
                    (usize::MAX, 0),
                    (0, usize::MAX),
                ] {
                    let rr = r8.wrapping_add(dr);
                    let cc = c8.wrapping_add(dc);
                    if rr < rows8 && cc < cols8 && n < starts.len() {
                        let m = e.me_hints[base + rr * cols8 + cc];
                        starts[n] = ((m[1] + 4) >> 3, (m[0] + 4) >> 3);
                        n += 1;
                    }
                }
            }
            for (k, &(sx, sy)) in starts[..n].iter().enumerate() {
                if (sx, sy) == (0, 0)
                    || (sx, sy) == (pcx, pcy)
                    || starts[..k].contains(&(sx, sy))
                    || sx.abs() > 4 * range
                    || sy.abs() > 4 * range
                {
                    continue;
                }
                let c = cost(sx, sy, best.2);
                if c < best.2 {
                    best = (sx, sy, c);
                }
            }
            // Descending square steps, moving while one improves.
            let lim = 4 * range;
            let mut step = (range / 2).max(1);
            while step >= 1 {
                for _ in 0..8 {
                    let (cx, cy) = (best.0, best.1);
                    for (dy, dx) in SQUARE8 {
                        let (nx, ny) = (cx + dx * step, cy + dy * step);
                        if nx.abs() > lim || ny.abs() > lim {
                            continue;
                        }
                        let c = cost(nx, ny, best.2);
                        if c < best.2 {
                            best = (nx, ny, c);
                        }
                    }
                    if (best.0, best.1) == (cx, cy) {
                        break;
                    }
                }
                step /= 2;
            }
        } else {
            let (cx, cy) = (best.0, best.1);
            // A coarse square search (step 2), then a fine one around the best.
            let mut step_best = best;
            for dy in (-range..=range).step_by(2) {
                let mut dx = -range;
                while dx <= range {
                    // Four candidates at once where they lie inside the
                    // reference: their SADs against the limit the first of
                    // them would have, then taken in order as `cost` takes
                    // them (a SAD over a larger limit is over its own).
                    let (nx, ny) = (cx + dx, cy + dy);
                    if dx + 6 <= range
                        && x0 + nx >= 0
                        && x0 + nx + 6 + w as i32 - 1 <= last_x
                        && y0 + ny >= 0
                        && y0 + ny + h as i32 - 1 <= last_y
                    {
                        let rate = |k: i32| {
                            lambda
                                * (mv_bits(ny * 8 - pred_mv[0])
                                    + mv_bits((nx + 2 * k) * 8 - pred_mv[1]))
                        };
                        let rates = [rate(0), rate(1), rate(2), rate(3)];
                        let low = rates.iter().copied().fold(f64::MAX, f64::min);
                        let sads = if low >= step_best.2 {
                            [u64::MAX; 4]
                        } else {
                            let st = e.stride[0];
                            let ro = (y0 + ny) as usize * rp.stride + (x0 + nx) as usize;
                            crate::dsp::enc::sad4_step2(
                                &e.src[0][y0 as usize * st + x0 as usize..],
                                st,
                                &rp.data[ro..],
                                rp.stride,
                                w,
                                h,
                                (step_best.2 - low).min(1e18) as u64,
                            )
                        };
                        for (k, (&s, &rate)) in sads.iter().zip(&rates).enumerate() {
                            if rate >= step_best.2 || s == u64::MAX {
                                continue;
                            }
                            if s > (step_best.2 - rate).min(1e18) as u64 {
                                continue;
                            }
                            let c = s as f64 + rate;
                            if c < step_best.2 {
                                step_best = (nx + 2 * k as i32, ny, c);
                            }
                        }
                        dx += 8;
                        continue;
                    }
                    let c = cost(nx, ny, step_best.2);
                    if c < step_best.2 {
                        step_best = (nx, ny, c);
                    }
                    dx += 2;
                }
            }
            best = step_best;
            let (cx, cy) = (best.0, best.1);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let c = cost(cx + dx, cy + dy, best.2);
                    if c < best.2 {
                        best = (cx + dx, cy + dy, c);
                    }
                }
            }
        }
        let mut mv: Mv = [best.1 * 8, best.0 * 8];
        if !self.f.hdr.force_integer_mv {
            // Sub-sample refinement, in eighth samples: 4 (half), 2
            // (quarter), then 1 with high precision.
            let mut best_cost = self.inter_satd(ref_frame, mv) as f64
                + lambda * (mv_bits(mv[0] - pred_mv[0]) + mv_bits(mv[1] - pred_mv[1]));
            let min_step = if self.f.hdr.allow_high_precision_mv {
                1
            } else {
                2
            };
            let mut step = 4;
            while step >= min_step {
                let center = mv;
                let try_mv = |t: &Self, dy: i32, dx: i32, mv: &mut Mv, best_cost: &mut f64| {
                    let cand = [center[0] + dy * step, center[1] + dx * step];
                    let c = t.inter_satd(ref_frame, cand) as f64
                        + lambda * (mv_bits(cand[0] - pred_mv[0]) + mv_bits(cand[1] - pred_mv[1]));
                    if c < *best_cost {
                        *best_cost = c;
                        *mv = cand;
                    }
                    c
                };
                if fast {
                    // The cross, then the diagonal between its better
                    // vertical and horizontal neighbours.
                    let up = try_mv(self, -1, 0, &mut mv, &mut best_cost);
                    let down = try_mv(self, 1, 0, &mut mv, &mut best_cost);
                    let left = try_mv(self, 0, -1, &mut mv, &mut best_cost);
                    let right = try_mv(self, 0, 1, &mut mv, &mut best_cost);
                    let dy = if up < down { -1 } else { 1 };
                    let dx = if left < right { -1 } else { 1 };
                    try_mv(self, dy, dx, &mut mv, &mut best_cost);
                } else {
                    for (dy, dx) in SQUARE8 {
                        try_mv(self, dy, dx, &mut mv, &mut best_cost);
                    }
                }
                step /= 2;
            }
        }
        // Remember the vector over the 8x8 blocks this one covers.
        let e = self.enc.as_mut().expect("encode mode");
        if e.me_hints.len() != cells * 7 {
            e.me_hints = vec![[0, 0]; cells * 7];
        }
        let base = (ref_frame - LAST_FRAME) as usize * cells;
        let rows8 = self.f.mi_rows.div_ceil(2);
        for rr in self.b.mi_row / 2..((self.b.mi_row * MI_SIZE + h) / 8).min(rows8) {
            for cc in self.b.mi_col / 2..((self.b.mi_col * MI_SIZE + w) / 8).min(cols8) {
                e.me_hints[base + rr * cols8 + cc] = mv;
            }
        }
        mv
    }

    /// Forward transforms and quantises the residual (source minus
    /// `CurrFrame`) of one transform block into `EncCtx::coefs`; returns the
    /// end of block in the scan of `tx_type`.
    fn quantise_block(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
        tx_type: usize,
    ) -> usize {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let tw = w.min(32);
        let th = h.min(32);
        let mut res = std::mem::take(&mut self.enc.as_mut().expect("encode mode").res);
        res.resize(64 * 64, 0);
        let mut sum_abs = 0u32;
        {
            let e = self.enc();
            let cur = &self.f.cur.planes[plane];
            for i in 0..h {
                let so = (y + i) * e.stride[plane] + x;
                let co = (y + i) * cur.stride + x;
                let srow = &e.src[plane][so..so + w];
                let crow = &cur.data[co..co + w];
                for (r, (&a, &b)) in res[i * w..(i + 1) * w]
                    .iter_mut()
                    .zip(srow.iter().zip(crow))
                {
                    *r = a as i32 - b as i32;
                    sum_abs += r.unsigned_abs();
                }
            }
        }
        if self.b.lossless {
            // Lossless (quantiser index 0): the levels the decoder's
            // Walsh-Hadamard transform takes back to the residual exactly.
            let iscan = crate::decoder::residual::inverse_scan(tx_sz, tx_type);
            let enc = self.enc.as_mut().expect("encode mode");
            let eob = forward_wht(&res[..16], iscan, &mut enc.coefs[..16]);
            enc.res = res;
            self.plane_tx_type = tx_type;
            return eob;
        }
        let bd = self.f.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let (dc_delta, ac_delta) = match plane {
            0 => (self.f.hdr.delta_q_y_dc, 0),
            1 => (self.f.hdr.delta_q_u_dc, self.f.hdr.delta_q_u_ac),
            _ => (self.f.hdr.delta_q_v_dc, self.f.hdr.delta_q_v_ac),
        };
        let q = self.qindex();
        let dc_q = DC_QLOOKUP[bdi][clip3(0, 255, q + dc_delta) as usize] as f64;
        let ac_q = AC_QLOOKUP[bdi][clip3(0, 255, q + ac_delta) as usize] as f64;
        let denom = match tx_sz {
            TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 2.0,
            TX_64X64 | TX_32X64 | TX_64X32 => 4.0,
            _ => 1.0,
        };
        let (dc_r, ac_r) = (denom / dc_q, denom / ac_q);
        // The AC dead zone: where an AC coefficient (in quantiser steps)
        // starts to be coded as 1. Wider in the search's trials when only
        // the blocks coded for real are quantised by rate and distortion
        // (`Tools::rdoq` 1): those levels, which the trials then stand in
        // for, are the sparser (0.70 measured best of 0.62 to 0.78, −0.5 %
        // BD-rate PSNR-Y and −0.4 % PSNR-YUV at speed 6).
        let dz = if self.enc().tools.rdoq == 1 && self.sd.counting() {
            0.70
        } else {
            0.62
        };
        // Every level zero when no coefficient can reach the dead zones.
        let top = sum_abs as f64 * coef_bound(tx_sz, tx_type);
        let enc = self.enc.as_mut().expect("encode mode");
        if top * ac_r < dz && top * dc_r < 0.5 {
            // (Only the first `tw * th`, the block's, are read: the levels
            // below write all of them.)
            enc.coefs[..tw * th].fill(0);
            enc.res = res;
            self.plane_tx_type = tx_type;
            return 0;
        }
        let mut fc = std::mem::take(&mut enc.fc);
        fc.resize(64 * 64, 0.0);
        // The levels, and the end of block in the scan the decoder will use
        // (or what they were when this block was last quantised so), and
        // the coefficients (for `rdoq`).
        let q = Quantiser {
            ac_r,
            dc_r,
            ac_round: 1.0 - dz,
        };
        let iscan = crate::decoder::residual::inverse_scan(tx_sz, tx_type);
        let eob = if TX_SIZE_SQR_UP[tx_sz] == TX_64X64 {
            forward_quantise(
                &res[..w * h],
                tx_sz,
                tx_type,
                q,
                iscan,
                &mut enc.coefs[..],
                &mut fc,
            )
        } else {
            let slot =
                ((y >> 2) * 61 + (x >> 2) * 7 + plane * 3 + tx_sz * 5 + tx_type) % QCACHE_SLOTS;
            QCACHE.with_borrow_mut(|cache| {
                let entry = &mut cache[slot];
                let key = (
                    tx_sz as u8,
                    tx_type as u8,
                    ac_r.to_bits(),
                    dc_r.to_bits(),
                    dz.to_bits(),
                );
                if entry.key == Some(key) && entry.res[..] == res[..w * h] {
                    enc.coefs[..tw * th].copy_from_slice(&entry.levels);
                    for (k, row) in entry.coefs.chunks_exact(tw).enumerate() {
                        fc[k * w..k * w + tw].copy_from_slice(row);
                    }
                    return entry.eob;
                }
                let eob = forward_quantise(
                    &res[..w * h],
                    tx_sz,
                    tx_type,
                    q,
                    iscan,
                    &mut enc.coefs[..],
                    &mut fc,
                );
                entry.key = Some(key);
                entry.res.clear();
                entry.res.extend_from_slice(&res[..w * h]);
                entry.levels.clear();
                entry.levels.extend_from_slice(&enc.coefs[..tw * th]);
                entry.coefs.clear();
                for k in 0..th {
                    entry.coefs.extend_from_slice(&fc[k * w..k * w + tw]);
                }
                entry.eob = eob;
                eob
            })
        };
        enc.res = res;
        enc.fc = fc;
        self.plane_tx_type = tx_type;
        eob
    }

    /// Squared error between the source and the reconstruction over a
    /// sample rectangle of `plane`, inside the frame.
    fn px_sse(&self, plane: usize, x0: usize, y0: usize, w: usize, h: usize) -> u64 {
        let e = self.enc();
        let (ssx, ssy) = self.f.plane_ss(plane);
        let fw = (self.f.hdr.frame_width as usize + ssx) >> ssx;
        let fh = (self.f.hdr.frame_height as usize + ssy) >> ssy;
        let x1 = (x0 + w).min(fw);
        let y1 = (y0 + h).min(fh);
        let pl = &self.f.cur.planes[plane];
        let st = e.stride[plane];
        let mut s = 0u64;
        for y in y0..y1 {
            s += crate::dsp::enc::sse_row(
                &e.src[plane][y * st + x0..y * st + x1.max(x0)],
                &pl.data[y * pl.stride + x0..y * pl.stride + x1.max(x0)],
            );
        }
        s
    }

    /// The luma transform types worth a trial for this transform block.
    fn tx_type_candidates(&self, tx_sz: usize) -> Vec<usize> {
        if self.b.lossless || TX_SIZE_SQR_UP[tx_sz] > TX_32X32 || self.qindex() == 0 {
            return vec![DCT_DCT];
        }
        let set = self.get_tx_set(tx_sz);
        if set == 0 {
            return vec![DCT_DCT];
        }
        const ORDER: [usize; 16] = [
            DCT_DCT,
            ADST_ADST,
            ADST_DCT,
            DCT_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST,
        ];
        let max = self.enc().tools.tx_types as usize;
        ORDER
            .iter()
            .copied()
            .filter(|&t| self.is_tx_type_in_set(set, t))
            .take(max.max(1))
            .collect()
    }

    /// Chooses the levels of the transform block about to be coded (its
    /// prediction is in `CurrFrame`) and returns its end of block. The
    /// luma transform type is searched by trial coding when there is a
    /// choice.
    pub(crate) fn enc_choose_coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> usize {
        if plane == 0
            && let Some(t) = self.rdo().forced_tx_type.take()
        {
            self.enc_tx_type = t;
            return self.quantise_coded(0, start_x, start_y, tx_sz, t, true);
        }
        let derived = self.b.lossless || TX_SIZE_SQR_UP[tx_sz] > TX_32X32;
        let t = if derived {
            DCT_DCT
        } else if plane == 0 {
            if self.enc().tools.tx_type_rd {
                // Replayed, or searched (and logged, for replay).
                let rdo = self.rdo();
                if let Some(Decision::TxType(t)) = rdo.replay.front().copied() {
                    rdo.replay.pop_front();
                    rdo.log.push(Decision::TxType(t));
                    t as usize
                } else {
                    let cands = if self.rdo().fast_tx {
                        vec![DCT_DCT]
                    } else {
                        self.tx_type_candidates(tx_sz)
                    };
                    let t = if cands.len() > 1 {
                        self.search_tx_type(start_x, start_y, tx_sz, &cands)
                    } else {
                        cands[0]
                    };
                    self.rdo().log.push(Decision::TxType(t as u8));
                    t
                }
            } else {
                self.plan.tx_type
            }
        } else {
            self.compute_tx_type(plane, tx_sz, start_x >> 2, start_y >> 2)
        };
        if plane == 0 {
            self.enc_tx_type = t;
        }
        self.quantise_coded(plane, start_x, start_y, tx_sz, t, false)
    }

    /// The levels of a transform block about to be coded: the dead-zone
    /// quantiser's, chosen again by rate and distortion where `Tools::rdoq`
    /// says (`type_search`: coded on trial by the transform type search).
    fn quantise_coded(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
        tx_type: usize,
        type_search: bool,
    ) -> usize {
        let eob = self.quantise_block(plane, x, y, tx_sz, tx_type);
        let on = match self.enc().tools.rdoq {
            0 => false,
            1 => !self.sd.counting(),
            2 => !type_search,
            _ => true,
        };
        if eob == 0 || !on || self.b.lossless {
            return eob;
        }
        self.rdoq(plane, x, y, tx_sz, tx_type, eob)
    }

    /// Codes the luma transform block at `(x, y)` with each candidate type
    /// in counting mode and returns the cheapest.
    fn search_tx_type(&mut self, x: usize, y: usize, tx_sz: usize, cands: &[usize]) -> usize {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let (x4, y4, w4, h4) = (x >> 2, y >> 2, w >> 2, h >> 2);
        let ms = self.f.ms;
        // What coding the block changes: its samples, its contexts, its
        // transform types.
        let pl = &self.f.cur.planes[0];
        let x1 = (x + w).min(pl.w);
        let y1 = (y + h).min(pl.h);
        let mut pix = Vec::with_capacity(w * h);
        for yy in y..y1 {
            pix.extend_from_slice(&pl.data[yy * pl.stride + x..yy * pl.stride + x1]);
        }
        let al: Vec<u8> = self.above_level_ctx[0][x4..x4 + w4].to_vec();
        let ad: Vec<u8> = self.above_dc_ctx[0][x4..x4 + w4].to_vec();
        let ll: Vec<u8> = self.left_level_ctx[0][y4..y4 + h4].to_vec();
        let ld: Vec<u8> = self.left_dc_ctx[0][y4..y4 + h4].to_vec();
        let rows = (y4 + h4).min(self.f.mi_rows + 32);
        let cols = (x4 + w4).min(ms);
        let mut txt = Vec::new();
        for r in y4..rows {
            txt.extend_from_slice(&self.f.tx_types[r * ms + x4..r * ms + cols]);
        }
        let restore = |t: &mut Self| {
            let pl = &mut t.f.cur.planes[0];
            let xw = x1 - x;
            for (i, yy) in (y..y1).enumerate() {
                pl.data[yy * pl.stride + x..yy * pl.stride + x1]
                    .copy_from_slice(&pix[i * xw..(i + 1) * xw]);
            }
            t.above_level_ctx[0][x4..x4 + w4].copy_from_slice(&al);
            t.above_dc_ctx[0][x4..x4 + w4].copy_from_slice(&ad);
            t.left_level_ctx[0][y4..y4 + h4].copy_from_slice(&ll);
            t.left_dc_ctx[0][y4..y4 + h4].copy_from_slice(&ld);
            let cw = cols - x4;
            for (i, r) in (y4..rows).enumerate() {
                t.f.tx_types[r * ms + x4..r * ms + cols]
                    .copy_from_slice(&txt[i * cw..(i + 1) * cw]);
            }
        };
        let counting = self.sd.set_counting(true);
        let bits_start = self.sd.trial_bits();
        let nonzero = self.rdo().nonzero;
        let saved_ptt = self.plane_tx_type;
        let mut best = (cands[0], f64::MAX);
        for &t in cands {
            self.rdo().forced_tx_type = Some(t);
            let bits0 = self.sd.trial_bits();
            let eob = self.coeffs(0, x, y, tx_sz);
            if eob > 0 {
                self.reconstruct_decode(0, x, y, tx_sz, eob, None);
            }
            let bits = self.sd.trial_bits() - bits0;
            let sse = self.px_sse(0, x, y, w, h);
            let cost = self.rd_cost(sse, bits);
            if cost < best.1 {
                best = (t, cost);
            }
            restore(self);
        }
        self.rdo().nonzero = nonzero;
        self.plane_tx_type = saved_ptt;
        self.sd.set_counting(counting);
        self.sd.set_trial_bits(bits_start);
        best.0
    }
}
