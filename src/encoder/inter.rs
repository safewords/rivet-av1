//! The inter coding tools a block's best inter candidate may be coded
//! with instead: switchable interpolation filters, OBMC, local warped
//! motion, inter-intra (smooth and wedge), and the compound predictions
//! other than the average (distance-weighted, wedge, difference-weighted).
//!
//! Each tool changes the block's prediction and the symbols that signal
//! it, not its vectors, so a variant is measured by *probing*: its mode
//! info coded in counting mode and the block predicted, by the decoder's
//! own syntax and prediction (`decode_block_prediction`), and the
//! prediction's SATD against the source added to the Lagrange multiplier
//! times the bits. That is a fraction of a trial (no transform, no
//! residual), so many variants can be probed for the few trials the
//! rate-distortion search then gives the best of them. The walker decides
//! whether a tool can be coded at all (`motion_mode` only where the
//! neighbours allow it, a warp only where it is valid, and so on): a
//! variant that the walker did not code as planned is dropped.

use crate::Result;
use crate::consts::*;
use crate::decoder::predict::wedge_mask;
use crate::decoder::state::Mv;
use crate::decoder::tile::{
    Block, PLAN_AVERAGE, PLAN_DIFFWTD, PLAN_DISTANCE, PLAN_WEDGE, Plan, TileDecoder,
};
use crate::encoder::rdo::Snap;
use crate::encoder::{
    FORCE_DIFFWTD, FORCE_DISTANCE, FORCE_FILTER, FORCE_GLOBAL, FORCE_INTERINTRA, FORCE_OBMC,
    FORCE_WARP, FORCE_WEDGE, FORCE_WEDGE_INTERINTRA,
};
use crate::symbol::Coder;
use crate::tables::*;

/// Where a block's search stands, for probing: the state to put back
/// before each probe.
pub(crate) struct ProbeFrom<'a> {
    pub(crate) snap: &'a Snap,
    pub(crate) b: &'a Block,
    pub(crate) log_start: usize,
}

/// The variants of a block's best inter candidate worth a trial.
pub(crate) struct Variants {
    /// Those that probed cheaper than the candidate, cheapest first.
    pub(crate) better: Vec<Plan>,
    /// The variant `Tools::force_inter_tool` asks for, where the block
    /// can be coded with it.
    pub(crate) forced: Option<Plan>,
}

impl TileDecoder<'_, '_> {
    /// Codes the mode info of `p` and predicts the block with it, from the
    /// state in `from` (counting; nothing is left changed but what the
    /// caller puts back). The prediction's cost: its luma's SATD against
    /// the source (chroma is not predicted) plus the multiplier times the
    /// bits; `None` when `coded` says the walker did not code what `p`
    /// planned.
    fn probe(
        &mut self,
        from: &ProbeFrom,
        p: Plan,
        coded: impl Fn(&Block) -> bool,
    ) -> Result<Option<f64>> {
        let (r, c, size) = (self.b.mi_row, self.b.mi_col, self.b.mi_size);
        self.restore_parts(from.snap, false);
        self.b = from.b.clone();
        self.rdo().forced_plan = Some(p);
        let bits0 = self.sd.trial_bits();
        self.enc.as_mut().expect("encode mode").probing = true;
        let done = self.decode_block_prediction(r, c, size);
        self.enc.as_mut().expect("encode mode").probing = false;
        done?;
        let bits = self.sd.trial_bits() - bits0;
        self.rdo().log.truncate(from.log_start);
        if !coded(&self.b) {
            return Ok(None);
        }
        let satd = self.region_satd(
            0,
            c * MI_SIZE,
            r * MI_SIZE,
            block_width(size),
            block_height(size),
        );
        let lambda = self.enc.as_ref().expect("encode mode").lambda;
        Ok(Some(satd as f64 + lambda * bits as f64 / 256.0))
    }

    /// The variants of `base` (the block's best inter candidate) that the
    /// tools in force allow, probed.
    pub(crate) fn inter_variants(&mut self, from: &ProbeFrom, base: Plan) -> Result<Variants> {
        let tools = self.enc.as_ref().expect("encode mode").tools;
        let force = tools.force_inter_tool;
        let mut out = Variants {
            better: Vec::new(),
            forced: None,
        };
        if base.skip_mode || !base.is_inter {
            return Ok(out);
        }
        let Some(base_cost) = self.probe(from, base, |_| true)? else {
            return Ok(out);
        };
        // Whether the luma vectors have a sub-sample part (row, column): a
        // filter does nothing at a whole sample.
        let lists = 1 + (base.ref_frame2 > INTRA_FRAME) as usize;
        let base_mv = self.b.mv;
        let base_refs = self.b.ref_frame;
        // The cost in bits of each filter in each direction, in the
        // contexts the base probe left.
        let mut filter_bits = [[0f64; 3]; 2];
        for (dir, bits) in filter_bits.iter_mut().enumerate() {
            let ctx = self.interp_filter_ctx(dir);
            for (f, b) in bits.iter_mut().enumerate() {
                *b = Coder::cost(&self.cdf.interp_filter[ctx], f) as f64 / 256.0;
            }
        }
        let frac = self.b.mv[..lists].iter().fold([false; 2], |f, mv| {
            [f[0] || mv[0] & 7 != 0, f[1] || mv[1] & 7 != 0]
        });
        let hdr = &self.f.hdr;
        let seq = &self.f.seq;
        let ms = self.b.mi_size;
        let compound = base.ref_frame2 > INTRA_FRAME;
        let motion_modes = hdr.is_motion_mode_switchable;
        let switchable = hdr.interpolation_filter == SWITCHABLE;
        let dual = seq.enable_dual_filter;
        let interintra = seq.enable_interintra_compound;
        let masked = seq.enable_masked_compound;
        let jnt = seq.enable_jnt_comp;
        let mut found: Vec<(f64, Plan)> = Vec::new();
        let mut keep = |out: &mut Variants, cost: Option<f64>, q: Plan, tool: u8| {
            if let Some(cost) = cost {
                if force != 0 && force == tool && out.forced.is_none() {
                    out.forced = Some(q);
                }
                found.push((cost, q));
            }
        };
        // Global motion (tests): GLOBALMV from the same reference.
        if force == FORCE_GLOBAL && !compound {
            let mut q = base;
            q.y_mode = GLOBALMV;
            q.ref_mv_idx = 0;
            let rf = base.ref_frame as usize;
            let cost = self.probe(from, q, |b| b.y_mode == GLOBALMV && b.motion_mode == SIMPLE)?;
            if self.f.hdr.gm_type[rf] > TRANSLATION {
                keep(&mut out, cost, q, FORCE_GLOBAL);
            }
        }
        // The motion modes: OBMC, then the local warp.
        let mut best_mm = (base_cost, base);
        if !compound && motion_modes {
            if tools.obmc || force == FORCE_OBMC {
                let mut q = base;
                q.motion_mode = OBMC as u8;
                let cost = self.probe(from, q, |b| b.motion_mode == OBMC)?;
                if let Some(cost) = cost
                    && cost < best_mm.0
                {
                    best_mm = (cost, q);
                }
                keep(&mut out, cost, q, FORCE_OBMC);
            }
            if tools.warped_motion || force == FORCE_WARP {
                let mut q = base;
                q.motion_mode = LOCALWARP as u8;
                let cost = self.probe(from, q, |b| b.motion_mode == LOCALWARP && b.local_valid)?;
                keep(&mut out, cost, q, FORCE_WARP);
            }
        }
        // The interpolation filters, of the better of the block's
        // prediction and its OBMC (a warp has none).
        if switchable
            && (tools.switchable_filter || force == FORCE_FILTER)
            && (frac[0] || frac[1] || force == FORCE_FILTER)
        {
            // (Where one direction is at a whole sample, the filters of the
            // other are all there is to choose.)
            let pairs: &[[u8; 2]] = if dual && frac[0] && frac[1] {
                &[
                    [1, 1],
                    [2, 2],
                    [0, 1],
                    [1, 0],
                    [0, 2],
                    [2, 0],
                    [1, 2],
                    [2, 1],
                ]
            } else {
                &[[1, 1], [2, 2]]
            };
            let forced_pair = if pairs.len() > 2 { [2, 1] } else { [2, 2] };
            // The smooth and sharp filters probed, with the better of the
            // block's prediction and its OBMC; with dual filters, the best
            // mixed pair by the predictions alone (the block's vectors as
            // the base probe left them) as well.
            let mut tries: Vec<[u8; 2]> = vec![[1, 1], [2, 2]];
            if pairs.len() > 2
                && let Some(pair) =
                    self.best_filters(&base_mv[..lists], &base_refs[..lists], pairs, &filter_bits)
                && pair[0] != pair[1]
            {
                tries.push(pair);
            }
            if force == FORCE_FILTER && !tries.contains(&forced_pair) {
                tries.push(forced_pair);
            }
            for pair in tries {
                let mut q = best_mm.1;
                q.interp_filter = pair;
                let cost = self.probe(from, q, |b| {
                    b.interp_filter == [pair[0] as u32, pair[1] as u32]
                })?;
                let tool = if pair == forced_pair { FORCE_FILTER } else { 0 };
                keep(&mut out, cost, q, tool);
            }
        }
        // Inter-intra: each smooth mode, then wedges with the best.
        if !compound && interintra && (BLOCK_8X8..=BLOCK_32X32).contains(&ms) {
            let mut best_ii: Option<(f64, Plan)> = None;
            for mode in II_DC_PRED as u8..=II_SMOOTH_PRED as u8 {
                let mut q = base;
                q.interintra = true;
                q.interintra_mode = mode;
                q.motion_mode = SIMPLE as u8;
                let cost = self.probe(from, q, |b| b.interintra && !b.wedge_interintra)?;
                if let Some(cost) = cost
                    && best_ii.is_none_or(|b| cost < b.0)
                {
                    best_ii = Some((cost, q));
                }
                keep(&mut out, cost, q, FORCE_INTERINTRA);
            }
            if let Some((_, ii)) = best_ii
                && WEDGE_BITS[ms] > 0
            {
                // (The state the last probe left: the block's position and
                // neighbours, as the intra prediction reads them.)
                let mut q = ii;
                q.wedge_interintra = true;
                q.wedge_index = self.best_interintra_wedge(&ii);
                let cost = self.probe(from, q, |b| b.wedge_interintra)?;
                keep(&mut out, cost, q, FORCE_WEDGE_INTERINTRA);
            }
        }
        // The compound predictions but the average.
        if compound && base.compound == PLAN_AVERAGE {
            if jnt {
                let mut q = base;
                q.compound = PLAN_DISTANCE;
                let cost = self.probe(from, q, |b| b.compound_type == COMPOUND_DISTANCE)?;
                keep(&mut out, cost, q, FORCE_DISTANCE);
            }
            if masked {
                for mask_type in 0..2 {
                    let mut q = base;
                    q.compound = PLAN_DIFFWTD;
                    q.mask_type = mask_type;
                    let cost = self.probe(from, q, |b| {
                        b.compound_type == COMPOUND_DIFFWTD && b.mask_type == mask_type as usize
                    })?;
                    keep(&mut out, cost, q, FORCE_DIFFWTD);
                }
                if WEDGE_BITS[ms] > 0 {
                    let (index, sign) = self.best_compound_wedge(&base);
                    let mut q = base;
                    q.compound = PLAN_WEDGE;
                    q.wedge_index = index;
                    q.wedge_sign = sign;
                    let cost = self.probe(from, q, |b| b.compound_type == COMPOUND_WEDGE)?;
                    keep(&mut out, cost, q, FORCE_WEDGE);
                }
            }
        }
        found.retain(|x| x.0 < base_cost);
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.better = found.into_iter().map(|x| x.1).collect();
        Ok(out)
    }

    /// The filter pair (`[vertical, horizontal]`) whose luma prediction
    /// of the block, from `refs` with `mvs`, costs least in SATD plus the
    /// multiplier times its bits (`bits[dir][filter]`; the horizontal
    /// direction counted only where it is coded); `None` when that is the
    /// regular filter.
    fn best_filters(
        &self,
        mvs: &[Mv],
        refs: &[i32],
        pairs: &[[u8; 2]],
        bits: &[[f64; 3]; 2],
    ) -> Option<[u8; 2]> {
        let ms = self.b.mi_size;
        let (w, h) = (block_width(ms), block_height(ms));
        let (x, y) = (self.b.mi_col * MI_SIZE, self.b.mi_row * MI_SIZE);
        let e = self.enc.as_ref().expect("encode mode");
        let dual = self.f.seq.enable_dual_filter;
        let st = e.stride[0];
        let mut p = [Vec::new(), Vec::new()];
        let mut best = (f64::MAX, [0u8; 2]);
        for &pair in std::iter::once(&[0u8, 0u8]).chain(pairs) {
            for (i, (&mv, &rf)) in mvs.iter().zip(refs).enumerate() {
                self.inter_pred_with(0, x, y, w, h, rf, mv, pair, &mut p[i]);
            }
            if mvs.len() == 2 {
                let [a, b] = &mut p;
                for (u, &v) in a[..w * h].iter_mut().zip(&b[..w * h]) {
                    *u = (*u + v + 1) >> 1;
                }
            }
            let satd = crate::dsp::enc::satd(&e.src[0][y * st + x..], st, &p[0], w, w, h);
            let rate =
                bits[0][pair[0] as usize] + if dual { bits[1][pair[1] as usize] } else { 0.0 };
            let cost = satd as f64 + e.lambda * rate;
            if cost < best.0 {
                best = (cost, pair);
            }
        }
        (best.1 != [0, 0]).then_some(best.1)
    }

    /// The wedge (index and sign) of the compound block `base` whose blend
    /// of its two luma predictions is nearest the source.
    fn best_compound_wedge(&self, base: &Plan) -> (u8, u8) {
        let ms = self.b.mi_size;
        let (w, h) = (block_width(ms), block_height(ms));
        let (x, y) = (self.b.mi_col * MI_SIZE, self.b.mi_row * MI_SIZE);
        let mut p0 = Vec::new();
        let mut p1 = Vec::new();
        self.inter_pred(0, x, y, w, h, base.ref_frame, base.mv[0], &mut p0);
        self.inter_pred(0, x, y, w, h, base.ref_frame2, base.mv[1], &mut p1);
        let masks: Vec<&[u8]> = (0..32).map(|i| wedge_mask(ms, i & 1, i >> 1)).collect();
        let sse = self.blend_sse(x, y, w, h, &masks, &p0, &p1);
        let i = (0..32)
            .min_by(|&i, &j| sse[i].total_cmp(&sse[j]))
            .unwrap_or(0);
        ((i >> 1) as u8, (i & 1) as u8)
    }

    /// The wedge index of the inter-intra block `ii` (its smooth mode
    /// chosen) whose blend of the intra and inter luma predictions is
    /// nearest the source. Predicts the block's intra luma into the frame.
    fn best_interintra_wedge(&mut self, ii: &Plan) -> u8 {
        let ms = self.b.mi_size;
        let (w, h) = (block_width(ms), block_height(ms));
        let (x, y) = (self.b.mi_col * MI_SIZE, self.b.mi_row * MI_SIZE);
        let mode = match ii.interintra_mode as usize {
            II_DC_PRED => DC_PRED,
            II_V_PRED => V_PRED,
            II_H_PRED => H_PRED,
            _ => SMOOTH_PRED,
        };
        let (px, py, al, au, ar, bl, lw, lh, _) = self.intra_args(0);
        self.predict_intra(0, px, py, al, au, ar, bl, mode, lw, lh);
        let cur = &self.f.cur.planes[0];
        let mut intra = Vec::with_capacity(w * h);
        for i in 0..h {
            for j in 0..w {
                intra.push(cur.data[(y + i) * cur.stride + x + j] as i32);
            }
        }
        let mut inter = Vec::new();
        self.inter_pred(0, x, y, w, h, ii.ref_frame, ii.mv[0], &mut inter);
        let maxv = (1i32 << self.f.bit_depth) - 1;
        for v in inter.iter_mut() {
            *v = (*v).clamp(0, maxv);
        }
        // The mask weighs the intra prediction (sign 0).
        let masks: Vec<&[u8]> = (0..16).map(|i| wedge_mask(ms, 0, i)).collect();
        let sse = self.blend_sse(x, y, w, h, &masks, &intra, &inter);
        (0..16)
            .min_by(|&i, &j| sse[i].total_cmp(&sse[j]))
            .unwrap_or(0) as u8
    }

    /// Squared error against the luma source of `(m a + (64 - m) b) / 64`
    /// for each mask `m` of `masks` (in single precision: it ranks them).
    #[allow(clippy::too_many_arguments)]
    fn blend_sse(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        masks: &[&[u8]],
        a: &[i32],
        b: &[i32],
    ) -> Vec<f32> {
        let e = self.enc.as_ref().expect("encode mode");
        let st = e.stride[0];
        // The error is m (a - b) / 64 - (s - b).
        let mut d = Vec::with_capacity(w * h);
        let mut t = Vec::with_capacity(w * h);
        for i in 0..h {
            let src = &e.src[0][(y + i) * st + x..][..w];
            for j in 0..w {
                let k = i * w + j;
                d.push((a[k] - b[k]) as f32 / 64.0);
                t.push((src[j] as i32 - b[k]) as f32);
            }
        }
        masks
            .iter()
            .map(|m| {
                let mut sum = 0f32;
                for ((&mk, &dk), &tk) in m.iter().zip(&d).zip(&t) {
                    let v = mk as f32 * dk - tk;
                    sum += v * v;
                }
                sum
            })
            .collect()
    }
}
