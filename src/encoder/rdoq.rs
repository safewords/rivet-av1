//! Rate-distortion optimised quantisation.
//!
//! The dead-zone quantiser (`TileDecoder::quantise_block`) rounds every
//! coefficient by the same rule, whatever coding it costs. Here the levels
//! of a transform block are chosen again, each by its distortion plus
//! lambda times its rate, the rate taken from the CDFs in force and the
//! contexts `coeffs()` (5.11.39) will code it in: the candidates of a
//! coefficient are its level rounded to the nearest, one less, and zero.
//! The coefficients are decided in the order they are coded (the reverse
//! of the scan), so the neighbours a coefficient's contexts look at (those
//! later in the scan) are decided before it. Then the end of block: every
//! place the block could end (after a coefficient left non-zero, or
//! nowhere, the block all zero) is costed with the coefficients after it
//! dropped, and the cheapest kept.
//!
//! The distortion of a coefficient is measured in the transform domain,
//! scaled by the energy the inverse transform gives a unit coefficient
//! (the kernels are orthogonal, so the sample-domain error is the sum of
//! the coefficients' errors with that scale). Only the coefficients up to
//! the dead-zone quantiser's end of block are looked at: one it left at
//! zero past its last is too small to be worth a level.
//!
//! It costs about as much as a forward transform per block, and the
//! search codes each block many times on trial, so `Tools::rdoq` says
//! where it runs: in the blocks coded for real only (the search measuring
//! the dead-zone levels, with a wider dead zone that stands in for these
//! sparser ones), or in the trials too.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::consts::*;
use crate::decoder::residual::{
    coeff_base_ctx_of, coeff_base_eob_ctx, coeff_br_ctx_of, get_tx_class, scan_for,
};
use crate::decoder::tile::TileDecoder;
use crate::symbol::Coder;
use crate::tables::*;

/// The sum of squares of the residual the inverse transform makes of a
/// coefficient of 1 (in `Dequant`), for each transform size and type:
/// measured once, on large impulses, over the first few positions.
fn energy(tx_sz: usize, tx_type: usize) -> f64 {
    static CACHE: OnceLock<Vec<AtomicU64>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| (0..TX_SIZES_ALL * 16).map(|_| AtomicU64::new(0)).collect());
    let slot = &cache[tx_sz * 16 + tx_type];
    let v = slot.load(Ordering::Relaxed);
    if v != 0 {
        return f64::from_bits(v);
    }
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    const A: i32 = 1024;
    let mut dq = vec![0i32; 64 * 64];
    let mut rec = vec![0i32; w * h];
    let mut sum = 0f64;
    let mut n = 0;
    for i in 0..h.min(32).min(4) {
        for j in 0..w.min(32).min(4) {
            dq.fill(0);
            dq[i * 64 + j] = A;
            crate::dsp::itx::inverse_transform_2d(&dq, tx_sz, tx_type, false, 8, &mut rec);
            sum += rec.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>();
            n += 1;
        }
    }
    let e = sum / n as f64 / (A as f64 * A as f64);
    slot.store(e.to_bits(), Ordering::Relaxed);
    e
}

/// Per transform block: each coefficient's magnitude in levels and the
/// squared error of a level's worth of difference, in scan order; the
/// levels decided, in `Quant`'s layout; each coefficient's cost.
#[derive(Default)]
struct Scratch {
    mag: Vec<f64>,
    unit: Vec<f64>,
    lv: Vec<u32>,
    cost: Vec<f64>,
    as_last: Vec<f64>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

impl TileDecoder<'_, '_> {
    /// The CDF of `eob_pt` for this transform block.
    fn eob_pt_cdf(&self, eob_multisize: usize, ptype: usize, ectx: usize) -> &[u16] {
        let c = &self.cdf;
        match eob_multisize {
            0 => &c.eob_pt_16[ptype][ectx],
            1 => &c.eob_pt_32[ptype][ectx],
            2 => &c.eob_pt_64[ptype][ectx],
            3 => &c.eob_pt_128[ptype][ectx],
            4 => &c.eob_pt_256[ptype][ectx],
            5 => &c.eob_pt_512[ptype],
            _ => &c.eob_pt_1024[ptype],
        }
    }

    /// Chooses again the levels the dead-zone quantiser left in
    /// `EncCtx::coefs` for the transform block of `plane` at `(x, y)` (its
    /// forward transform in `EncCtx::fc`; `eob` its end of block), by rate
    /// and distortion; returns the new end of block, which is not past
    /// the dead-zone quantiser's (a coefficient it left at zero beyond its
    /// last is too small to be worth coding).
    pub(crate) fn rdoq(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
        tx_type: usize,
        eob: usize,
    ) -> usize {
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
        let w = TX_WIDTH[tx_sz];
        let adj = ADJUSTED_TX_SIZE[tx_sz];
        let bwl = TX_WIDTH_LOG2[adj];
        let tw = 1usize << bwl;
        let height = TX_HEIGHT[adj];
        let tx_class = get_tx_class(tx_type);
        let scan = scan_for(tx_sz, tx_type);
        let ptype = (plane > 0) as usize;
        let tx_sz_ctx = (TX_SIZE_SQR[tx_sz] + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1;
        let eob_multisize = TX_WIDTH_LOG2[tx_sz].min(5) + TX_HEIGHT_LOG2[tx_sz].min(5) - 4;
        let ectx = (tx_class != TX_CLASS_2D) as usize;
        let (x4, y4) = (x >> 2, y >> 2);
        let (w4, h4) = (TX_WIDTH[tx_sz] >> 2, TX_HEIGHT[tx_sz] >> 2);
        let skip_ctx = self.all_zero_ctx(plane, tx_sz, x4, y4, w4, h4);
        let dc_sign_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
        let e = self.enc.as_ref().expect("encode mode");
        // Squared error per 1/256 bit, a chroma plane's error weighing
        // more (`rdo::CHROMA_WEIGHT`, in quarters).
        let lambda = e.rd_lambda / 64.0 / crate::encoder::rdo::CHROMA_WEIGHT[plane] as f64;
        let gain = energy(tx_sz, tx_type);
        let cdf = &self.cdf;
        let base_cdfs = &cdf.coeff_base[tx_sz_ctx][ptype];
        let eob_cdfs = &cdf.coeff_base_eob[tx_sz_ctx][ptype];
        let br_cdfs = &cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype];
        let dc_sign_cdf = &cdf.dc_sign[ptype][dc_sign_ctx];
        let dc_negative = (e.fc[0] < 0.0) as usize;

        // The coefficients up to the dead-zone end of block, in scan
        // order: each one's magnitude in levels, and the squared error of a
        // level's worth of difference.
        let last0 = eob;
        let mut sc = SCRATCH.with_borrow_mut(std::mem::take);
        let Scratch {
            mag,
            unit,
            lv,
            cost,
            as_last,
        } = &mut sc;
        mag.clear();
        unit.clear();
        let (unit_dc, unit_ac) = (
            gain * dc_q * dc_q / (denom * denom),
            gain * ac_q * ac_q / (denom * denom),
        );
        let (inv_dc, inv_ac) = (denom / dc_q, denom / ac_q);
        for &pos in &scan[..last0] {
            let pos = pos as usize;
            let (i, j) = (pos >> bwl, pos & (tw - 1));
            let (inv, u) = if pos == 0 {
                (inv_dc, unit_dc)
            } else {
                (inv_ac, unit_ac)
            };
            mag.push(e.fc[i * w + j].abs() * inv);
            unit.push(u);
        }
        // The levels decided so far, in a map of `Quant`'s layout padded
        // by the four rows and columns the context neighbours reach (so
        // that they need no bounds tests): row stride `tw + 4`.
        let ls = tw + 4;
        lv.clear();
        lv.resize((height + 4) * ls, 0);
        // Per coefficient: the cost of the level chosen, and the extra cost
        // of that level were it the last.
        cost.resize(last0, 0.0);
        as_last.resize(last0, 0.0);
        let near: [usize; 5] = std::array::from_fn(|k| {
            let d = SIG_REF_DIFF_OFFSET[tx_class][k];
            d[0] as usize * ls + d[1] as usize
        });
        let near_br: [usize; 3] = std::array::from_fn(|k| {
            let d = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][k];
            d[0] as usize * ls + d[1] as usize
        });
        let level_rate = |l: u32, base: &[u16], br_ctx: usize, pos: usize| -> u32 {
            let mut r = Coder::cost(base, l.min(3) as usize);
            if l > NUM_BASE_LEVELS {
                let mut rem = l - NUM_BASE_LEVELS - 1;
                for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                    let s = rem.min(BR_CDF_SIZE - 1);
                    r += Coder::cost(&br_cdfs[br_ctx], s as usize);
                    rem -= s;
                    if s < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }
            if l > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
                let gx = l - NUM_BASE_LEVELS - COEFF_BASE_RANGE;
                let len = crate::bits::floor_log2(gx) + 1;
                r += (2 * len - 1) * 256;
            }
            if l > 0 {
                r += if pos == 0 {
                    Coder::cost(dc_sign_cdf, dc_negative)
                } else {
                    256
                };
            }
            r
        };
        // The base-level part of the cost of the last coefficient: the
        // `coeff_base_eob` symbol in place of `coeff_base`.
        let eob_base_rate = |l: u32, c: usize| -> u32 {
            let ctx = coeff_base_eob_ctx(c, bwl, height);
            Coder::cost(&eob_cdfs[ctx], (l.min(3) - 1) as usize)
        };
        for c in (0..last0).rev() {
            let pos = scan[c] as usize;
            let (row, col) = (pos >> bwl, pos & (tw - 1));
            let lp = row * ls + col;
            let m: u32 = near.iter().map(|&o| lv[lp + o].min(3)).sum();
            let base = &base_cdfs[coeff_base_ctx_of(tx_sz, tx_class, row, col, m)];
            let v = mag[c];
            let hi = (v + 0.5).floor().min(((1 << 20) - 1) as f64) as u32;
            let dist = |l: u32| {
                let d = v - l as f64;
                d * d * unit[c]
            };
            let is_last = c + 1 == last0;
            if hi == 0 {
                cost[c] = dist(0) + lambda * Coder::cost(base, 0) as f64;
                as_last[c] = 0.0;
                continue;
            }
            // The context of `coeff_br`, needed only above level 2.
            let br_ctx = if hi > NUM_BASE_LEVELS {
                let mb: u32 = near_br
                    .iter()
                    .map(|&o| lv[lp + o].min(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1))
                    .sum();
                coeff_br_ctx_of(tx_class, row, col, mb)
            } else {
                0
            };
            let mut best = (f64::MAX, 0u32);
            for l in [hi, hi - 1, 0] {
                if is_last && l == 0 {
                    break;
                }
                let mut r = level_rate(l, base, br_ctx, pos);
                if is_last {
                    r = r - Coder::cost(base, l.min(3) as usize) + eob_base_rate(l, c);
                }
                let j = dist(l) + lambda * r as f64;
                if j < best.0 {
                    best = (j, l);
                }
                if l == 0 {
                    break;
                }
            }
            let l = best.1;
            lv[lp] = l;
            cost[c] = best.0;
            as_last[c] = if l > 0 && !is_last {
                lambda * (eob_base_rate(l, c) as f64 - Coder::cost(base, l.min(3) as usize) as f64)
            } else {
                0.0
            };
        }
        // The end of block: the cheapest place to stop.
        let eob_rate = |eob: usize| -> u32 {
            let pt = if eob <= 2 {
                eob
            } else {
                crate::bits::floor_log2(eob as u32 - 1) as usize + 2
            };
            let mut r = Coder::cost(self.eob_pt_cdf(eob_multisize, ptype, ectx), pt - 1);
            if pt >= 3 {
                let rest = eob - ((1 << (pt - 2)) + 1);
                let bit = (rest >> (pt - 3)) & 1;
                r += Coder::cost(&cdf.eob_extra[tx_sz_ctx][ptype][pt - 3], bit);
                r += 256 * (pt as u32 - 3);
            }
            r
        };
        let skip_cdf = &cdf.txb_skip[tx_sz_ctx][skip_ctx];
        // All dropped: the distortion of every coefficient left at zero
        // (but those past `last0`, which every choice leaves at zero).
        let mut dropped: f64 = (0..last0).map(|c| mag[c] * mag[c] * unit[c]).sum();
        let mut best = (dropped + lambda * Coder::cost(skip_cdf, 1) as f64, 0usize);
        let coded = lambda * Coder::cost(skip_cdf, 0) as f64;
        let mut kept = 0f64;
        for c in 0..last0 {
            dropped -= mag[c] * mag[c] * unit[c];
            kept += cost[c];
            let pos = scan[c] as usize;
            if lv[(pos >> bwl) * ls + (pos & (tw - 1))] == 0 {
                continue;
            }
            let j = kept + as_last[c] + dropped + coded + lambda * eob_rate(c + 1) as f64;
            if j < best.0 {
                best = (j, c + 1);
            }
        }
        let eob = best.1;
        let e = self.enc.as_mut().expect("encode mode");
        for (c, &pos) in scan[..last0].iter().enumerate() {
            let pos = pos as usize;
            let (i, j) = (pos >> bwl, pos & (tw - 1));
            let l = if c < eob { lv[i * ls + j] as i32 } else { 0 };
            e.coefs[pos] = if e.fc[i * w + j] < 0.0 { -l } else { l };
        }
        SCRATCH.with_borrow_mut(|s| *s = sc);
        eob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::fwd::forward_2d;

    /// The sample-domain error of a block reconstructed from coarsened
    /// coefficients is the coefficients' error scaled by `energy`, which
    /// is what the distortion of a level is measured with.
    #[test]
    fn energy_scales_coefficient_error_to_sample_error() {
        let mut seed = 0x2545_f491_u32;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for (tx_sz, tx_type) in [
            (TX_4X4, DCT_DCT),
            (TX_8X8, ADST_DCT),
            (TX_16X8, DCT_DCT),
            (TX_16X16, ADST_ADST),
            (TX_32X32, DCT_DCT),
            (TX_8X32, IDTX),
        ] {
            let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
            let res: Vec<i32> = (0..w * h).map(|_| (rand() % 121) as i32 - 60).collect();
            let mut c = vec![0f64; w * h];
            forward_2d(&res, tx_sz, tx_type, &mut c);
            // Every coefficient rounded to a multiple of a step that makes
            // an error of about 5 per sample (well above the inverse
            // transform's own rounding).
            let step = (300.0 / energy(tx_sz, tx_type)).sqrt();
            let mut dq = vec![0i32; 64 * 64];
            let mut coef_err = 0f64;
            for i in 0..h.min(32) {
                for j in 0..w.min(32) {
                    let v = c[i * w + j];
                    let r = ((v / step).round() * step).round();
                    dq[i * 64 + j] = r as i32;
                    coef_err += (v - r) * (v - r);
                }
            }
            let mut rec = vec![0i32; w * h];
            crate::dsp::itx::inverse_transform_2d(&dq, tx_sz, tx_type, false, 8, &mut rec);
            let sample_err: f64 = res
                .iter()
                .zip(&rec)
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let predicted = energy(tx_sz, tx_type) * coef_err;
            assert!(
                (sample_err / predicted - 1.0).abs() < 0.1,
                "size {tx_sz} type {tx_type}: {sample_err} against {predicted}"
            );
        }
    }
}
