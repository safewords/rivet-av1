//! The prediction processes of 7.11: intra (7.11.2), inter (7.11.3),
//! palette (7.11.4) and chroma from luma (7.11.5).

use std::sync::OnceLock;

use crate::consts::*;
use crate::decoder::state::{Mv, PlaneBuf};
use crate::decoder::tile::{TileDecoder, is_directional_mode};
use crate::tables::*;

/// Offset of index 0 in the edge arrays (indices down to -2 are used).
const EDGE_OFF: usize = 16;
const EDGE_LEN: usize = EDGE_OFF + 2 * 129 + 16;

impl TileDecoder<'_, '_> {
    /// `compute_prediction()` (5.11.33).
    pub(crate) fn compute_prediction(&mut self) {
        let sb_mask = if self.f.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let sub_block_mi_row = self.b.mi_row & sb_mask;
        let sub_block_mi_col = self.b.mi_col & sb_mask;
        let planes = 1 + if self.b.has_chroma { 2 } else { 0 };
        for plane in 0..planes {
            let plane_sz = self.f.plane_residual_size(self.b.mi_size, plane);
            let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz];
            let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz];
            let log2w = MI_SIZE_LOG2 + MI_WIDTH_LOG2[plane_sz];
            let log2h = MI_SIZE_LOG2 + MI_HEIGHT_LOG2[plane_sz];
            let sub_x = if plane > 0 { self.f.ssx } else { 0 };
            let sub_y = if plane > 0 { self.f.ssy } else { 0 };
            let base_x = (self.b.mi_col >> sub_x) * MI_SIZE;
            let base_y = (self.b.mi_row >> sub_y) * MI_SIZE;
            let mut cand_row = (self.b.mi_row >> sub_y) << sub_y;
            let mut cand_col = (self.b.mi_col >> sub_x) << sub_x;
            self.b.is_inter_intra = self.b.is_inter && self.b.ref_frame[1] == INTRA_FRAME;
            if self.b.is_inter_intra {
                let mode = match self.b.interintra_mode {
                    II_DC_PRED => DC_PRED,
                    II_V_PRED => V_PRED,
                    II_H_PRED => H_PRED,
                    _ => SMOOTH_PRED,
                };
                let have_above_right = self.block_decoded(
                    plane,
                    (sub_block_mi_row >> sub_y) as isize - 1,
                    ((sub_block_mi_col >> sub_x) + num4x4_w) as isize,
                );
                let have_below_left = self.block_decoded(
                    plane,
                    ((sub_block_mi_row >> sub_y) + num4x4_h) as isize,
                    (sub_block_mi_col >> sub_x) as isize - 1,
                );
                let (have_left, have_above) = if plane == 0 {
                    (self.b.avail_l, self.b.avail_u)
                } else {
                    (self.b.avail_l_chroma, self.b.avail_u_chroma)
                };
                self.predict_intra(
                    plane,
                    base_x,
                    base_y,
                    have_left,
                    have_above,
                    have_above_right,
                    have_below_left,
                    mode,
                    log2w,
                    log2h,
                );
            }
            if self.b.is_inter {
                let mut pred_w = block_width(self.b.mi_size) >> sub_x;
                let mut pred_h = block_height(self.b.mi_size) >> sub_y;
                let mut some_use_intra = false;
                for r in 0..(num4x4_h << sub_y) {
                    for c in 0..(num4x4_w << sub_x) {
                        if self.mi(cand_row + r, cand_col + c).ref_frame[0] as i32 == INTRA_FRAME {
                            some_use_intra = true;
                        }
                    }
                }
                if some_use_intra {
                    pred_w = num4x4_w * 4;
                    pred_h = num4x4_h * 4;
                    cand_row = self.b.mi_row;
                    cand_col = self.b.mi_col;
                }
                let mut r = 0;
                let mut y = 0;
                while y < num4x4_h * 4 {
                    let mut c = 0;
                    let mut x = 0;
                    while x < num4x4_w * 4 {
                        self.predict_inter(
                            plane,
                            base_x + x,
                            base_y + y,
                            pred_w,
                            pred_h,
                            cand_row + r,
                            cand_col + c,
                        );
                        x += pred_w;
                        c += 1;
                    }
                    y += pred_h;
                    r += 1;
                }
            }
        }
    }

    /// The intra prediction process (7.11.2).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_intra(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_above_right: bool,
        have_below_left: bool,
        mode: usize,
        log2w: usize,
        log2h: usize,
    ) {
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let bd = self.f.bit_depth;
        let (max_x, max_y) = if plane == 0 {
            (self.f.mi_cols * MI_SIZE - 1, self.f.mi_rows * MI_SIZE - 1)
        } else {
            (
                ((self.f.mi_cols * MI_SIZE) >> self.f.ssx) - 1,
                ((self.f.mi_rows * MI_SIZE) >> self.f.ssy) - 1,
            )
        };
        let mut above = [0i32; EDGE_LEN];
        let mut left = [0i32; EDGE_LEN];
        {
            let cur = &self.f.cur.planes[plane];
            let px = |xx: usize, yy: usize| cur.get(xx, yy) as i32;
            for i in 0..w + h {
                above[EDGE_OFF + i] = if !have_above && have_left {
                    px(x - 1, y)
                } else if !have_above && !have_left {
                    (1 << (bd - 1)) - 1
                } else {
                    let above_limit = max_x.min(x + if have_above_right { 2 * w } else { w } - 1);
                    px(above_limit.min(x + i), y - 1)
                };
                left[EDGE_OFF + i] = if !have_left && have_above {
                    px(x, y - 1)
                } else if !have_left && !have_above {
                    (1 << (bd - 1)) + 1
                } else {
                    let left_limit = max_y.min(y + if have_below_left { 2 * h } else { h } - 1);
                    px(x - 1, left_limit.min(y + i))
                };
            }
            let corner = if have_above && have_left {
                px(x - 1, y - 1)
            } else if have_above {
                px(x, y - 1)
            } else if have_left {
                px(x - 1, y)
            } else {
                1 << (bd - 1)
            };
            above[EDGE_OFF - 1] = corner;
            left[EDGE_OFF - 1] = corner;
        }
        let mut pred = [0u16; 64 * 64];
        let maxv = (1i32 << bd) - 1;
        if plane == 0 && self.b.use_filter_intra {
            self.recursive_intra(&above, &left, w, h, &mut pred, maxv);
        } else if is_directional_mode(mode) {
            self.directional_intra(
                plane, x, y, have_left, have_above, mode, w, h, max_x, max_y, &mut above,
                &mut left, &mut pred, maxv,
            );
        } else if mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED {
            let wx = sm_weights(log2w);
            let wy = sm_weights(log2h);
            let a = |i: usize| above[EDGE_OFF + i];
            let l = |i: usize| left[EDGE_OFF + i];
            for i in 0..h {
                for j in 0..w {
                    let v = if mode == SMOOTH_PRED {
                        let s = wy[i] * a(j)
                            + (256 - wy[i]) * l(h - 1)
                            + wx[j] * l(i)
                            + (256 - wx[j]) * a(w - 1);
                        round2(s, 9)
                    } else if mode == SMOOTH_V_PRED {
                        round2(wy[i] * a(j) + (256 - wy[i]) * l(h - 1), 8)
                    } else {
                        round2(wx[j] * l(i) + (256 - wx[j]) * a(w - 1), 8)
                    };
                    pred[i * w + j] = v as u16;
                }
            }
        } else if mode == DC_PRED {
            let v = if have_left && have_above {
                let mut sum = 0i32;
                for k in 0..h {
                    sum += left[EDGE_OFF + k];
                }
                for k in 0..w {
                    sum += above[EDGE_OFF + k];
                }
                sum += ((w + h) >> 1) as i32;
                sum / (w + h) as i32
            } else if have_left {
                let mut sum = 0i32;
                for k in 0..h {
                    sum += left[EDGE_OFF + k];
                }
                ((sum + (h >> 1) as i32) >> log2h).clamp(0, maxv)
            } else if have_above {
                let mut sum = 0i32;
                for k in 0..w {
                    sum += above[EDGE_OFF + k];
                }
                ((sum + (w >> 1) as i32) >> log2w).clamp(0, maxv)
            } else {
                1 << (bd - 1)
            };
            pred[..w * h].fill(v as u16);
        } else {
            // PAETH_PRED
            let tl = above[EDGE_OFF - 1];
            for i in 0..h {
                for j in 0..w {
                    let a = above[EDGE_OFF + j];
                    let l = left[EDGE_OFF + i];
                    let base = a + l - tl;
                    let p_left = (base - l).abs();
                    let p_top = (base - a).abs();
                    let p_top_left = (base - tl).abs();
                    let v = if p_left <= p_top && p_left <= p_top_left {
                        l
                    } else if p_top <= p_top_left {
                        a
                    } else {
                        tl
                    };
                    pred[i * w + j] = v as u16;
                }
            }
        }
        let cur = &mut self.f.cur.planes[plane];
        for i in 0..h {
            let row = &mut cur.data[(y + i) * cur.stride + x..(y + i) * cur.stride + x + w];
            row.copy_from_slice(&pred[i * w..(i + 1) * w]);
        }
    }

    /// The recursive intra prediction process (7.11.2.3).
    fn recursive_intra(
        &self,
        above: &[i32],
        left: &[i32],
        w: usize,
        h: usize,
        pred: &mut [u16],
        maxv: i32,
    ) {
        let w4 = w >> 2;
        let h2 = h >> 1;
        let mode = self.b.filter_intra_mode;
        for i2 in 0..h2 {
            for j4 in 0..w4 {
                let mut p = [0i32; 7];
                for (i, pv) in p.iter_mut().enumerate() {
                    *pv = if i < 5 {
                        if i2 == 0 {
                            above[(EDGE_OFF as isize + ((j4 << 2) + i) as isize - 1) as usize]
                        } else if j4 == 0 && i == 0 {
                            left[EDGE_OFF + (i2 << 1) - 1]
                        } else {
                            pred[((i2 << 1) - 1) * w + (j4 << 2) + i - 1] as i32
                        }
                    } else if j4 == 0 {
                        left[EDGE_OFF + (i2 << 1) + i - 5]
                    } else {
                        pred[((i2 << 1) + i - 5) * w + (j4 << 2) - 1] as i32
                    };
                }
                for i1 in 0..2 {
                    for j1 in 0..4 {
                        let mut pr = 0;
                        for (i, &pv) in p.iter().enumerate() {
                            pr += INTRA_FILTER_TAPS[mode][(i1 << 2) + j1][i] * pv;
                        }
                        let v = round2signed(pr, INTRA_FILTER_SCALE_BITS).clamp(0, maxv);
                        pred[((i2 << 1) + i1) * w + (j4 << 2) + j1] = v as u16;
                    }
                }
            }
        }
    }

    /// The directional intra prediction process (7.11.2.4).
    #[allow(clippy::too_many_arguments)]
    fn directional_intra(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        mode: usize,
        w: usize,
        h: usize,
        max_x: usize,
        max_y: usize,
        above: &mut [i32; EDGE_LEN],
        left: &mut [i32; EDGE_LEN],
        pred: &mut [u16],
        maxv: i32,
    ) {
        let angle_delta = if plane == 0 {
            self.b.angle_delta_y
        } else {
            self.b.angle_delta_uv
        };
        let p_angle = MODE_TO_ANGLE[mode] + angle_delta * ANGLE_STEP;
        let mut upsample_above = 0u32;
        let mut upsample_left = 0u32;
        let (wi, hi) = (w as i32, h as i32);
        if self.f.seq.enable_intra_edge_filter {
            if p_angle != 90 && p_angle != 180 {
                if p_angle > 90 && p_angle < 180 && (w + h) >= 24 {
                    // The filter corner process (7.11.2.7).
                    let s = left[EDGE_OFF] * 5 + above[EDGE_OFF - 1] * 6 + above[EDGE_OFF] * 5;
                    let v = round2(s, 4);
                    left[EDGE_OFF - 1] = v;
                    above[EDGE_OFF - 1] = v;
                }
                let filter_type = self.get_filter_type(plane);
                if have_above {
                    let strength = intra_edge_filter_strength(wi, hi, filter_type, p_angle - 90);
                    let num_px = w.min(max_x - x + 1) + if p_angle < 90 { h } else { 0 } + 1;
                    intra_edge_filter(above, num_px, strength);
                }
                if have_left {
                    let strength = intra_edge_filter_strength(wi, hi, filter_type, p_angle - 180);
                    let num_px = h.min(max_y - y + 1) + if p_angle > 180 { w } else { 0 } + 1;
                    intra_edge_filter(left, num_px, strength);
                }
            }
            let filter_type = self.get_filter_type(plane);
            upsample_above = intra_edge_upsample(wi, hi, filter_type, p_angle - 90) as u32;
            let num_px = w + if p_angle < 90 { h } else { 0 };
            if upsample_above != 0 {
                intra_edge_upsample_apply(above, num_px, maxv);
            }
            upsample_left = intra_edge_upsample(wi, hi, filter_type, p_angle - 180) as u32;
            let num_px = h + if p_angle > 180 { w } else { 0 };
            if upsample_left != 0 {
                intra_edge_upsample_apply(left, num_px, maxv);
            }
        }
        let dx = if p_angle < 90 {
            DR_INTRA_DERIVATIVE[p_angle as usize]
        } else if p_angle > 90 && p_angle < 180 {
            DR_INTRA_DERIVATIVE[(180 - p_angle) as usize]
        } else {
            0
        };
        let dy = if p_angle > 90 && p_angle < 180 {
            DR_INTRA_DERIVATIVE[(p_angle - 90) as usize]
        } else if p_angle > 180 {
            DR_INTRA_DERIVATIVE[(270 - p_angle) as usize]
        } else {
            0
        };
        let a = |i: i32| above[(EDGE_OFF as i32 + i) as usize];
        let l = |i: i32| left[(EDGE_OFF as i32 + i) as usize];
        for i in 0..hi {
            for j in 0..wi {
                let v = if p_angle < 90 {
                    let idx = (i + 1) * dx;
                    let base = (idx >> (6 - upsample_above)) + (j << upsample_above);
                    let shift = ((idx << upsample_above) >> 1) & 0x1F;
                    let max_base_x = (wi + hi - 1) << upsample_above;
                    if base < max_base_x {
                        round2(a(base) * (32 - shift) + a(base + 1) * shift, 5)
                    } else {
                        a(max_base_x)
                    }
                } else if p_angle > 90 && p_angle < 180 {
                    let idx = (j << 6) - (i + 1) * dx;
                    let base = idx >> (6 - upsample_above);
                    if base >= -(1 << upsample_above) {
                        let shift = ((idx << upsample_above) >> 1) & 0x1F;
                        round2(a(base) * (32 - shift) + a(base + 1) * shift, 5)
                    } else {
                        let idx = (i << 6) - (j + 1) * dy;
                        let base = idx >> (6 - upsample_left);
                        let shift = ((idx << upsample_left) >> 1) & 0x1F;
                        round2(l(base) * (32 - shift) + l(base + 1) * shift, 5)
                    }
                } else if p_angle > 180 {
                    let idx = (j + 1) * dy;
                    let base = (idx >> (6 - upsample_left)) + (i << upsample_left);
                    let shift = ((idx << upsample_left) >> 1) & 0x1F;
                    round2(l(base) * (32 - shift) + l(base + 1) * shift, 5)
                } else if p_angle == 90 {
                    a(j)
                } else {
                    l(i)
                };
                pred[(i * wi + j) as usize] = v as u16;
            }
        }
    }

    /// The intra filter type process (7.11.2.8).
    fn get_filter_type(&self, plane: usize) -> bool {
        let mut above_smooth = false;
        let mut left_smooth = false;
        let (avail_u, avail_l) = if plane == 0 {
            (self.b.avail_u, self.b.avail_l)
        } else {
            (self.b.avail_u_chroma, self.b.avail_l_chroma)
        };
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        if avail_u {
            let mut r = mi_row - 1;
            let mut c = mi_col;
            if plane > 0 {
                if self.f.ssx != 0 && (mi_col & 1) == 0 {
                    c += 1;
                }
                if self.f.ssy != 0 && (mi_row & 1) != 0 {
                    r -= 1;
                }
            }
            above_smooth = self.is_smooth(r, c, plane);
        }
        if avail_l {
            let mut r = mi_row;
            let mut c = mi_col - 1;
            if plane > 0 {
                if self.f.ssx != 0 && (mi_col & 1) != 0 {
                    c -= 1;
                }
                if self.f.ssy != 0 && (mi_row & 1) == 0 {
                    r += 1;
                }
            }
            left_smooth = self.is_smooth(r, c, plane);
        }
        above_smooth || left_smooth
    }

    fn is_smooth(&self, row: usize, col: usize, plane: usize) -> bool {
        let m = self.mi(row, col);
        let mode = if plane == 0 {
            m.y_mode as usize
        } else {
            if m.ref_frame[0] as i32 > INTRA_FRAME {
                return false;
            }
            m.uv_mode as usize
        };
        mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED
    }

    /// The palette prediction process (7.11.4).
    pub(crate) fn predict_palette(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
    ) {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let palette = match plane {
            0 => self.b.palette_colors_y,
            1 => self.b.palette_colors_u,
            _ => self.b.palette_colors_v,
        };
        let map = if plane == 0 {
            &self.color_map_y
        } else {
            &self.color_map_uv
        };
        let cur = &mut self.f.cur.planes[plane];
        for i in 0..h {
            for j in 0..w {
                let v = palette[map[y * 4 + i][x * 4 + j] as usize];
                cur.set(start_x + j, start_y + i, v);
            }
        }
    }

    /// The predict chroma from luma process (7.11.5).
    pub(crate) fn predict_chroma_from_luma(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let sub_x = self.f.ssx;
        let sub_y = self.f.ssy;
        let alpha = if plane == 1 {
            self.b.cfl_alpha_u
        } else {
            self.b.cfl_alpha_v
        };
        let mut l = [0i32; 32 * 32];
        let mut luma_avg: i32 = 0;
        {
            let luma = &self.f.cur.planes[0];
            for i in 0..h {
                let luma_y = ((start_y + i) << sub_y).min(self.b.max_luma_h - (1 << sub_y));
                for j in 0..w {
                    let luma_x = ((start_x + j) << sub_x).min(self.b.max_luma_w - (1 << sub_x));
                    let mut t = 0i32;
                    for dy in 0..=sub_y {
                        for dx in 0..=sub_x {
                            t += luma.get(luma_x + dx, luma_y + dy) as i32;
                        }
                    }
                    let v = t << (3 - sub_x - sub_y);
                    l[i * w + j] = v;
                    luma_avg += v;
                }
            }
        }
        luma_avg = round2(
            luma_avg,
            (TX_WIDTH_LOG2[tx_sz] + TX_HEIGHT_LOG2[tx_sz]) as u32,
        );
        let maxv = (1i32 << self.f.bit_depth) - 1;
        let cur = &mut self.f.cur.planes[plane];
        for i in 0..h {
            for j in 0..w {
                let dc = cur.get(start_x + j, start_y + i) as i32;
                let scaled = round2signed(alpha * (l[i * w + j] - luma_avg), 6);
                cur.set(
                    start_x + j,
                    start_y + i,
                    (dc + scaled).clamp(0, maxv) as u16,
                );
            }
        }
    }

    /// The inter prediction process (7.11.3.1).
    #[allow(clippy::too_many_arguments)]
    fn predict_inter(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        cand_row: usize,
        cand_col: usize,
    ) {
        let cand = *self.mi(cand_row, cand_col);
        let is_compound = cand.ref_frame[1] as i32 > INTRA_FRAME;
        let rv = self.f.rounding_variables(is_compound);
        if plane == 0 && self.b.motion_mode == LOCALWARP {
            self.warp_estimation();
            if self.b.local_valid {
                let (valid, ..) = setup_shear(&self.b.local_warp_params);
                self.b.local_valid = valid;
            }
        }
        // The prediction buffers, reused from block to block.
        let mut preds = std::mem::take(&mut self.pred_bufs);
        let lists = if is_compound { 2 } else { 1 };
        for p in preds.iter_mut().take(lists) {
            p.clear();
            p.resize(w * h, 0);
        }
        // Set when the prediction went straight into the frame.
        let mut written = false;
        for ref_list in 0..lists {
            let ref_frame = cand.ref_frame[ref_list] as i32;
            let mut global_valid = false;
            let gm_ref = if ref_frame > INTRA_FRAME {
                ref_frame as usize
            } else {
                0
            };
            if (self.b.y_mode == GLOBALMV || self.b.y_mode == GLOBAL_GLOBALMV)
                && self.f.hdr.gm_type[gm_ref] > TRANSLATION
            {
                let (valid, ..) = setup_shear(&self.f.hdr.gm_params[gm_ref]);
                global_valid = valid;
            }
            let use_warp = if w < 8 || h < 8 || self.f.hdr.force_integer_mv {
                0
            } else if self.b.motion_mode == LOCALWARP && self.b.local_valid {
                1
            } else if (self.b.y_mode == GLOBALMV || self.b.y_mode == GLOBAL_GLOBALMV)
                && self.f.hdr.gm_type[gm_ref] > TRANSLATION
                && !self.f.is_scaled(ref_frame)
                && global_valid
            {
                2
            } else {
                0
            };
            let mv = cand.mv[ref_list];
            if use_warp != 0 {
                let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
                let params = if use_warp == 1 {
                    self.b.local_warp_params
                } else {
                    self.f.hdr.gm_params[ref_frame as usize]
                };
                for i8 in 0..=((h - 1) >> 3) {
                    for j8 in 0..=((w - 1) >> 3) {
                        self.block_warp(
                            &params,
                            plane,
                            ref_idx,
                            x,
                            y,
                            i8,
                            j8,
                            w,
                            h,
                            &mut preds[ref_list],
                            rv,
                        );
                    }
                }
            } else if self.b.use_intrabc {
                let (fw, fh, uw) = (
                    self.f.hdr.frame_width as i32,
                    self.f.hdr.frame_height as i32,
                    self.f.hdr.upscaled_width as i32,
                );
                let (sx, sy, stx, sty) = self.f.scale_mv(plane, uw, fh, x, y, mv);
                let _ = fw;
                let last_x = (((self.f.mi_cols * MI_SIZE) as i32
                    + self.f.plane_ss(plane).0 as i32)
                    >> self.f.plane_ss(plane).0)
                    - 1;
                let last_y = (((self.f.mi_rows * MI_SIZE) as i32
                    + self.f.plane_ss(plane).1 as i32)
                    >> self.f.plane_ss(plane).1)
                    - 1;
                let filt = cand.interp_filter;
                let refp = &self.f.cur.planes[plane];
                block_inter_prediction(
                    refp,
                    last_x,
                    last_y,
                    sx,
                    sy,
                    stx,
                    sty,
                    w,
                    h,
                    filt,
                    rv,
                    &mut preds[ref_list],
                );
            } else {
                let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
                let r = self.f.refs[ref_idx]
                    .clone()
                    .expect("reference checked at header");
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
                let one = 1i64 << SCALE_SUBPEL_BITS;
                if !is_compound
                    && !self.b.is_inter_intra
                    && stx == one
                    && sty == one
                    && (sx >> 6) & SUBPEL_MASK as i64 == 0
                    && ((sy & 1023) >> 6) & SUBPEL_MASK as i64 == 0
                {
                    // A whole-sample motion vector into an unscaled
                    // reference, and nothing to blend with: both filters
                    // are `128 * sample` and the rounding gives the sample
                    // back, so the block is the reference's samples (its
                    // edges repeated), copied.
                    let refp = &r.frame.planes[plane];
                    let cur = &mut self.f.cur.planes[plane];
                    let (x0, y0) = ((sx >> 10) as i32, (sy >> 10) as i32);
                    let inside = x0 >= 0 && x0 + w as i32 <= last_x + 1;
                    for i in 0..h {
                        let yy = clip3(0, last_y, y0 + i as i32) as usize;
                        let src = refp.row(yy);
                        let dst = &mut cur.data[(y + i) * cur.stride + x..][..w];
                        if inside {
                            dst.copy_from_slice(&src[x0 as usize..x0 as usize + w]);
                        } else {
                            for (j, o) in dst.iter_mut().enumerate() {
                                *o = src[clip3(0, last_x, x0 + j as i32) as usize];
                            }
                        }
                    }
                    written = true;
                    continue;
                }
                block_inter_prediction(
                    &r.frame.planes[plane],
                    last_x,
                    last_y,
                    sx,
                    sy,
                    stx,
                    sty,
                    w,
                    h,
                    cand.interp_filter,
                    rv,
                    &mut preds[ref_list],
                );
            }
        }
        let ct = self.b.compound_type;
        if ct == COMPOUND_WEDGE && plane == 0 {
            let m = wedge_mask(self.b.mi_size, self.b.wedge_sign, self.b.wedge_index);
            for i in 0..h {
                for j in 0..w {
                    self.f.mask[i * 128 + j] = m[i * w + j];
                }
            }
        } else if ct == COMPOUND_INTRA {
            let size_scale = MAX_SB_SIZE as usize / h.max(w);
            for i in 0..h {
                for j in 0..w {
                    self.f.mask[i * 128 + j] = match self.b.interintra_mode {
                        II_V_PRED => II_WEIGHTS_1D[i * size_scale] as u8,
                        II_H_PRED => II_WEIGHTS_1D[j * size_scale] as u8,
                        II_SMOOTH_PRED => II_WEIGHTS_1D[i.min(j) * size_scale] as u8,
                        _ => 32,
                    };
                }
            }
        } else if ct == COMPOUND_DIFFWTD && plane == 0 {
            let bd = self.f.bit_depth;
            for i in 0..h {
                for j in 0..w {
                    let diff = (preds[0][i * w + j] - preds[1][i * w + j]).abs();
                    let diff = round2(diff, (bd - 8) + rv.post_round as u32);
                    let m = clip3(0, 64, 38 + diff / 16);
                    self.f.mask[i * 128 + j] = if self.b.mask_type != 0 { 64 - m } else { m } as u8;
                }
            }
        }
        let (fwd_weight, bck_weight) = if ct == COMPOUND_DISTANCE {
            self.distance_weights(cand_row, cand_col)
        } else {
            (0, 0)
        };
        let maxv = (1i32 << self.f.bit_depth) - 1;
        let post = rv.post_round as u32;
        if written {
            // Already in the frame.
        } else if !is_compound && !self.b.is_inter_intra {
            let cur = &mut self.f.cur.planes[plane];
            let stride = cur.stride;
            for (i, p) in preds[0][..w * h].chunks_exact(w).enumerate() {
                for (o, &v) in cur.data[(y + i) * stride + x..][..w].iter_mut().zip(p) {
                    *o = v.clamp(0, maxv) as u16;
                }
            }
        } else if ct == COMPOUND_AVERAGE {
            let cur = &mut self.f.cur.planes[plane];
            let stride = cur.stride;
            let (p0, p1) = (&preds[0][..w * h], &preds[1][..w * h]);
            let add = (1 << (1 + post)) >> 1;
            for i in 0..h {
                let row = &mut cur.data[(y + i) * stride + x..][..w];
                let (a, b) = (&p0[i * w..][..w], &p1[i * w..][..w]);
                for j in 0..w {
                    row[j] = ((a[j] + b[j] + add) >> (1 + post)).clamp(0, maxv) as u16;
                }
            }
        } else if ct == COMPOUND_DISTANCE {
            let cur = &mut self.f.cur.planes[plane];
            let stride = cur.stride;
            let (p0, p1) = (&preds[0][..w * h], &preds[1][..w * h]);
            let add = (1 << (4 + post)) >> 1;
            for i in 0..h {
                let row = &mut cur.data[(y + i) * stride + x..][..w];
                let (a, b) = (&p0[i * w..][..w], &p1[i * w..][..w]);
                for j in 0..w {
                    let v = (fwd_weight * a[j] + bck_weight * b[j] + add) >> (4 + post);
                    row[j] = v.clamp(0, maxv) as u16;
                }
            }
        } else {
            self.mask_blend(&preds, plane, x, y, w, h, post, maxv);
        }
        self.pred_bufs = preds;
        if self.b.motion_mode == OBMC {
            self.overlapped_motion_compensation(plane, w, h);
        }
    }

    /// The mask blend process (7.11.3.14).
    #[allow(clippy::too_many_arguments)]
    fn mask_blend(
        &mut self,
        preds: &[Vec<i32>; 2],
        plane: usize,
        dst_x: usize,
        dst_y: usize,
        w: usize,
        h: usize,
        post: u32,
        maxv: i32,
    ) {
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let interintra = self.b.interintra;
        let wedge_interintra = self.b.wedge_interintra;
        let mask = &self.f.mask;
        let cur = &mut self.f.cur.planes[plane];
        for yy in 0..h {
            for xx in 0..w {
                let m = if (sub_x == 0 && sub_y == 0) || (interintra && !wedge_interintra) {
                    mask[yy * 128 + xx] as i32
                } else if sub_x != 0 && sub_y == 0 {
                    round2(
                        mask[yy * 128 + 2 * xx] as i32 + mask[yy * 128 + 2 * xx + 1] as i32,
                        1,
                    )
                } else {
                    round2(
                        mask[2 * yy * 128 + 2 * xx] as i32
                            + mask[2 * yy * 128 + 2 * xx + 1] as i32
                            + mask[(2 * yy + 1) * 128 + 2 * xx] as i32
                            + mask[(2 * yy + 1) * 128 + 2 * xx + 1] as i32,
                        2,
                    )
                };
                if interintra {
                    let pred0 = round2(preds[0][yy * w + xx], post).clamp(0, maxv);
                    let pred1 = cur.get(xx + dst_x, yy + dst_y) as i32;
                    cur.set(
                        xx + dst_x,
                        yy + dst_y,
                        round2(m * pred1 + (64 - m) * pred0, 6) as u16,
                    );
                } else {
                    let pred0 = preds[0][yy * w + xx];
                    let pred1 = preds[1][yy * w + xx];
                    let v = round2(m * pred0 + (64 - m) * pred1, 6 + post).clamp(0, maxv);
                    cur.set(xx + dst_x, yy + dst_y, v as u16);
                }
            }
        }
    }

    /// The distance weights process (7.11.3.15).
    fn distance_weights(&self, cand_row: usize, cand_col: usize) -> (i32, i32) {
        let m = self.mi(cand_row, cand_col);
        let mut dist = [0i32; 2];
        for (ref_list, d) in dist.iter_mut().enumerate() {
            let h = self.f.hdr.order_hints[m.ref_frame[ref_list] as usize];
            *d = clip3(
                0,
                MAX_FRAME_DISTANCE,
                crate::header::get_relative_dist(&self.f.seq, h, self.f.hdr.order_hint).abs(),
            );
        }
        let d0 = dist[1];
        let d1 = dist[0];
        let order = (d0 <= d1) as usize;
        if d0 == 0 || d1 == 0 {
            return (QUANT_DIST_LOOKUP[3][order], QUANT_DIST_LOOKUP[3][1 - order]);
        }
        let mut i = 0;
        while i < 3 {
            let c0 = QUANT_DIST_WEIGHT[i][order];
            let c1 = QUANT_DIST_WEIGHT[i][1 - order];
            if order != 0 {
                if d0 * c0 > d1 * c1 {
                    break;
                }
            } else if d0 * c0 < d1 * c1 {
                break;
            }
            i += 1;
        }
        (QUANT_DIST_LOOKUP[i][order], QUANT_DIST_LOOKUP[i][1 - order])
    }

    /// The overlapped motion compensation process (7.11.3.9).
    fn overlapped_motion_compensation(&mut self, plane: usize, w: usize, h: usize) {
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let ms = self.b.mi_size;
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        if self.b.avail_u && self.f.plane_residual_size(ms, plane) >= BLOCK_8X8 {
            let w4 = NUM_4X4_BLOCKS_WIDE[ms];
            let mut x4 = mi_col;
            let y4 = mi_row;
            let mut n_count = 0;
            let n_limit = 4.min(MI_WIDTH_LOG2[ms]);
            while n_count < n_limit && x4 < self.f.mi_cols.min(mi_col + w4) {
                let cand_row = mi_row - 1;
                let cand_col = x4 | 1;
                let cand_sz = self.mi(cand_row, cand_col).mi_size as usize;
                let step4 = NUM_4X4_BLOCKS_WIDE[cand_sz].clamp(2, 16);
                if self.mi(cand_row, cand_col).ref_frame[0] as i32 > INTRA_FRAME {
                    n_count += 1;
                    let pred_w = w.min((step4 * MI_SIZE) >> sub_x);
                    let pred_h = (h >> 1).min(32 >> sub_y);
                    let mask = obmc_mask(pred_h);
                    self.predict_overlap(
                        plane, cand_row, cand_col, x4, y4, pred_w, pred_h, 0, mask,
                    );
                }
                x4 += step4;
            }
        }
        if self.b.avail_l {
            let h4 = NUM_4X4_BLOCKS_HIGH[ms];
            let x4 = mi_col;
            let mut y4 = mi_row;
            let mut n_count = 0;
            let n_limit = 4.min(MI_HEIGHT_LOG2[ms]);
            while n_count < n_limit && y4 < self.f.mi_rows.min(mi_row + h4) {
                let cand_col = mi_col - 1;
                let cand_row = y4 | 1;
                let cand_sz = self.mi(cand_row, cand_col).mi_size as usize;
                let step4 = NUM_4X4_BLOCKS_HIGH[cand_sz].clamp(2, 16);
                if self.mi(cand_row, cand_col).ref_frame[0] as i32 > INTRA_FRAME {
                    n_count += 1;
                    let pred_w = (w >> 1).min(32 >> sub_x);
                    let pred_h = h.min((step4 * MI_SIZE) >> sub_y);
                    let mask = obmc_mask(pred_w);
                    self.predict_overlap(
                        plane, cand_row, cand_col, x4, y4, pred_w, pred_h, 1, mask,
                    );
                }
                y4 += step4;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_overlap(
        &mut self,
        plane: usize,
        cand_row: usize,
        cand_col: usize,
        x4: usize,
        y4: usize,
        pred_w: usize,
        pred_h: usize,
        pass: usize,
        mask: &[i32],
    ) {
        let cand = *self.mi(cand_row, cand_col);
        let mv = cand.mv[0];
        let ref_idx = self.f.hdr.ref_frame_idx[(cand.ref_frame[0] as i32 - LAST_FRAME) as usize];
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let pred_x = (x4 * 4) >> sub_x;
        let pred_y = (y4 * 4) >> sub_y;
        let r = self.f.refs[ref_idx]
            .clone()
            .expect("reference checked at header");
        let (sx, sy, stx, sty) = self.f.scale_mv(
            plane,
            r.upscaled_width as i32,
            r.frame_height as i32,
            pred_x,
            pred_y,
            mv,
        );
        let last_x = ((r.upscaled_width as i32 + sub_x as i32) >> sub_x) - 1;
        let last_y = ((r.frame_height as i32 + sub_y as i32) >> sub_y) - 1;
        let rv = self.f.rounding_variables(false);
        let mut obmc_pred = vec![0i32; pred_w * pred_h];
        block_inter_prediction(
            &r.frame.planes[plane],
            last_x,
            last_y,
            sx,
            sy,
            stx,
            sty,
            pred_w,
            pred_h,
            cand.interp_filter,
            rv,
            &mut obmc_pred,
        );
        let maxv = (1i32 << self.f.bit_depth) - 1;
        let cur = &mut self.f.cur.planes[plane];
        for i in 0..pred_h {
            for j in 0..pred_w {
                let o = obmc_pred[i * pred_w + j].clamp(0, maxv);
                let m = if pass == 0 { mask[i] } else { mask[j] };
                let c = cur.get(pred_x + j, pred_y + i) as i32;
                cur.set(
                    pred_x + j,
                    pred_y + i,
                    round2(m * c + (64 - m) * o, 6) as u16,
                );
            }
        }
    }

    /// The warp estimation process (7.11.3.8).
    fn warp_estimation(&mut self) {
        let mut a = [[0i64; 2]; 2];
        let mut bx = [0i64; 2];
        let mut by = [0i64; 2];
        let ms = self.b.mi_size;
        let w4 = NUM_4X4_BLOCKS_WIDE[ms] as i64;
        let h4 = NUM_4X4_BLOCKS_HIGH[ms] as i64;
        let mid_y = self.b.mi_row as i64 * 4 + h4 * 2 - 1;
        let mid_x = self.b.mi_col as i64 * 4 + w4 * 2 - 1;
        let suy = mid_y * 8;
        let sux = mid_x * 8;
        let duy = suy + self.b.mv[0][0] as i64;
        let dux = sux + self.b.mv[0][1] as i64;
        let ls = |a: i64, b: i64| ((a * b) >> 2) + (a + b);
        for i in 0..self.b.num_samples {
            let c = self.b.cand_list[i];
            let sy = c[0] as i64 - suy;
            let sx = c[1] as i64 - sux;
            let dy = c[2] as i64 - duy;
            let dx = c[3] as i64 - dux;
            if (sx - dx).abs() < LS_MV_MAX as i64 && (sy - dy).abs() < LS_MV_MAX as i64 {
                a[0][0] += ls(sx, sx) + 8;
                a[0][1] += ls(sx, sy) + 4;
                a[1][1] += ls(sy, sy) + 8;
                bx[0] += ls(sx, dx) + 8;
                bx[1] += ls(sy, dx) + 4;
                by[0] += ls(sx, dy) + 4;
                by[1] += ls(sy, dy) + 8;
            }
        }
        let det = a[0][0] * a[1][1] - a[0][1] * a[0][1];
        self.b.local_valid = det != 0;
        if det == 0 {
            return;
        }
        let (mut div_shift, mut div_factor) = resolve_divisor(det);
        div_shift -= WARPEDMODEL_PREC_BITS as i32;
        if div_shift < 0 {
            div_factor <<= -div_shift;
            div_shift = 0;
        }
        let rd = |v: i128| -> i64 {
            let p = v * div_factor as i128;
            let r = if div_shift == 0 {
                p
            } else if p >= 0 {
                (p + (1i128 << (div_shift - 1))) >> div_shift
            } else {
                -((-p + (1i128 << (div_shift - 1))) >> div_shift)
            };
            r.clamp(i64::MIN as i128, i64::MAX as i128) as i64
        };
        let nondiag = |v: i128| -> i32 {
            rd(v).clamp(
                -WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 + 1,
                WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 - 1,
            ) as i32
        };
        let diag = |v: i128| -> i32 {
            rd(v).clamp(
                (1i64 << WARPEDMODEL_PREC_BITS) - WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 + 1,
                (1i64 << WARPEDMODEL_PREC_BITS) + WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 - 1,
            ) as i32
        };
        let (a00, a01, a11) = (a[0][0] as i128, a[0][1] as i128, a[1][1] as i128);
        let p = &mut self.b.local_warp_params;
        p[2] = diag(a11 * bx[0] as i128 - a01 * bx[1] as i128);
        p[3] = nondiag(-a01 * bx[0] as i128 + a00 * bx[1] as i128);
        p[4] = nondiag(a11 * by[0] as i128 - a01 * by[1] as i128);
        p[5] = diag(-a01 * by[0] as i128 + a00 * by[1] as i128);
        let mvx = self.b.mv[0][1] as i64;
        let mvy = self.b.mv[0][0] as i64;
        let vx = mvx * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x * (p[2] as i64 - (1 << WARPEDMODEL_PREC_BITS)) + mid_y * p[3] as i64);
        let vy = mvy * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x * p[4] as i64 + mid_y * (p[5] as i64 - (1 << WARPEDMODEL_PREC_BITS)));
        p[0] = vx.clamp(
            -WARPEDMODEL_TRANS_CLAMP as i64,
            WARPEDMODEL_TRANS_CLAMP as i64 - 1,
        ) as i32;
        p[1] = vy.clamp(
            -WARPEDMODEL_TRANS_CLAMP as i64,
            WARPEDMODEL_TRANS_CLAMP as i64 - 1,
        ) as i32;
    }

    /// The block warp process (7.11.3.5).
    #[allow(clippy::too_many_arguments)]
    fn block_warp(
        &self,
        warp_params: &[i32; 6],
        plane: usize,
        ref_idx: usize,
        x: usize,
        y: usize,
        i8: usize,
        j8: usize,
        w: usize,
        h: usize,
        pred: &mut [i32],
        rv: RoundingVars,
    ) {
        let r = self.f.refs[ref_idx]
            .as_ref()
            .expect("reference checked at header");
        let refp = &r.frame.planes[plane];
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let last_x = ((r.upscaled_width as i64 + sub_x as i64) >> sub_x) - 1;
        let last_y = ((r.frame_height as i64 + sub_y as i64) >> sub_y) - 1;
        let src_x = ((x + j8 * 8 + 4) << sub_x) as i64;
        let src_y = ((y + i8 * 8 + 4) << sub_y) as i64;
        let wp: [i64; 6] = warp_params.map(|v| v as i64);
        let dst_x = wp[2] * src_x + wp[3] * src_y + wp[0];
        let dst_y = wp[4] * src_x + wp[5] * src_y + wp[1];
        let (_, alpha, beta, gamma, delta) = setup_shear(warp_params);
        let x4 = dst_x >> sub_x;
        let y4 = dst_y >> sub_y;
        let ix4 = x4 >> WARPEDMODEL_PREC_BITS;
        let sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
        let iy4 = y4 >> WARPEDMODEL_PREC_BITS;
        let sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
        let mut intermediate = [[0i32; 8]; 15];
        for i1 in -7i64..8 {
            for i2 in -4i64..4 {
                let sx = sx4 + alpha as i64 * i2 + beta as i64 * i1;
                let offs = (round2_64(sx, WARPEDDIFF_PREC_BITS as u32)
                    + WARPEDPIXEL_PREC_SHIFTS as i64) as usize;
                let mut s = 0i32;
                let yy = (iy4 + i1).clamp(0, last_y) as usize;
                for i3 in 0..8 {
                    let xx = (ix4 + i2 - 3 + i3).clamp(0, last_x) as usize;
                    s += WARPED_FILTERS[offs][i3 as usize] * refp.get(xx, yy) as i32;
                }
                intermediate[(i1 + 7) as usize][(i2 + 4) as usize] = round2(s, rv.round0);
            }
        }
        let lim_i = 4.min(h as i64 - i8 as i64 * 8 - 4);
        let lim_j = 4.min(w as i64 - j8 as i64 * 8 - 4);
        for i1 in -4i64..lim_i {
            for i2 in -4i64..lim_j {
                let sy = sy4 + gamma as i64 * i2 + delta as i64 * i1;
                let offs = (round2_64(sy, WARPEDDIFF_PREC_BITS as u32)
                    + WARPEDPIXEL_PREC_SHIFTS as i64) as usize;
                let mut s = 0i32;
                for i3 in 0..8 {
                    s += WARPED_FILTERS[offs][i3]
                        * intermediate[(i1 + i3 as i64 + 4) as usize][(i2 + 4) as usize];
                }
                let py = (i8 as i64 * 8 + i1 + 4) as usize;
                let px = (j8 as i64 * 8 + i2 + 4) as usize;
                pred[py * w + px] = round2(s, rv.round1);
            }
        }
    }
}

/// `InterRound0`, `InterRound1`, `InterPostRound` (7.11.3.2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundingVars {
    pub(crate) round0: u32,
    pub(crate) round1: u32,
    pub(crate) post_round: i32,
}

impl crate::decoder::FrameCtx {
    /// The rounding variables derivation process (7.11.3.2).
    pub(crate) fn rounding_variables(&self, is_compound: bool) -> RoundingVars {
        let mut round0 = 3;
        let mut round1 = if is_compound { 7 } else { 11 };
        if self.bit_depth == 12 {
            round0 += 2;
        }
        if self.bit_depth == 12 && !is_compound {
            round1 -= 2;
        }
        RoundingVars {
            round0,
            round1,
            post_round: 2 * FILTER_BITS - (round0 + round1) as i32,
        }
    }

    pub(crate) fn plane_ss(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.ssx, self.ssy)
        }
    }

    /// The motion vector scaling process (7.11.3.3): `startX`, `startY`,
    /// `stepX`, `stepY` for a reference of the given upscaled width and
    /// height.
    pub(crate) fn scale_mv(
        &self,
        plane: usize,
        ref_upscaled_width: i32,
        ref_frame_height: i32,
        x: usize,
        y: usize,
        mv: Mv,
    ) -> (i64, i64, i64, i64) {
        let fw = self.hdr.frame_width as i64;
        let fh = self.hdr.frame_height as i64;
        // A reference of the frame's own size has a scale of exactly one;
        // no need to divide to find that out for every block.
        let x_scale = if ref_upscaled_width as i64 == fw {
            1 << REF_SCALE_SHIFT
        } else {
            (((ref_upscaled_width as i64) << REF_SCALE_SHIFT) + fw / 2) / fw
        };
        let y_scale = if ref_frame_height as i64 == fh {
            1 << REF_SCALE_SHIFT
        } else {
            (((ref_frame_height as i64) << REF_SCALE_SHIFT) + fh / 2) / fh
        };
        let (sub_x, sub_y) = self.plane_ss(plane);
        let half_sample = 1i64 << (SUBPEL_BITS - 1);
        let orig_x = ((x as i64) << SUBPEL_BITS) + ((2 * mv[1] as i64) >> sub_x) + half_sample;
        let orig_y = ((y as i64) << SUBPEL_BITS) + ((2 * mv[0] as i64) >> sub_y) + half_sample;
        let base_x = orig_x * x_scale - (half_sample << REF_SCALE_SHIFT);
        let base_y = orig_y * y_scale - (half_sample << REF_SCALE_SHIFT);
        let off = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
        let sh = (REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS) as u32;
        let start_x = round2signed_64(base_x, sh) + off;
        let start_y = round2signed_64(base_y, sh) + off;
        let step_x = round2signed_64(x_scale, (REF_SCALE_SHIFT - SCALE_SUBPEL_BITS) as u32);
        let step_y = round2signed_64(y_scale, (REF_SCALE_SHIFT - SCALE_SUBPEL_BITS) as u32);
        (start_x, start_y, step_x, step_y)
    }
}

/// The block inter prediction process (7.11.3.4).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inter_prediction(
    refp: &PlaneBuf,
    last_x: i32,
    last_y: i32,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    interp_filter: [u8; 2],
    rv: RoundingVars,
    pred: &mut [i32],
) {
    let inter_h = ((((h as i64 - 1) * y_step + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS)
        + 8) as usize;
    let mut filt_x = interp_filter[1] as usize;
    if w <= 4 {
        if filt_x == EIGHTTAP as usize || filt_x == EIGHTTAP_SHARP as usize {
            filt_x = 4;
        } else if filt_x == EIGHTTAP_SMOOTH as usize {
            filt_x = 5;
        }
    }
    let mut filt_y = interp_filter[0] as usize;
    if h <= 4 {
        if filt_y == EIGHTTAP as usize || filt_y == EIGHTTAP_SHARP as usize {
            filt_y = 4;
        } else if filt_y == EIGHTTAP_SMOOTH as usize {
            filt_y = 5;
        }
    }
    // An unscaled reference (every reference of the frame's own size) has
    // its own version. x86-64 keeps the general one, whose row and column
    // filters have AVX2 versions over 32-bit intermediates.
    #[cfg(not(target_arch = "x86_64"))]
    if x_step == 1 << SCALE_SUBPEL_BITS && y_step == 1 << SCALE_SUBPEL_BITS {
        return block_inter_prediction_unscaled(
            refp,
            last_x,
            last_y,
            x,
            y,
            w,
            h,
            [filt_y, filt_x],
            rv,
            pred,
        );
    }
    // The intermediate array: a per-thread buffer, reused.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < inter_h * w {
            buf.resize(inter_h * w, 0);
        }
        block_inter_prediction_with(
            refp,
            last_x,
            last_y,
            x,
            y,
            x_step,
            y_step,
            w,
            h,
            [filt_y, filt_x],
            inter_h,
            rv,
            pred,
            &mut buf[..inter_h * w],
        )
    })
}

/// Block inter prediction from a reference of the frame's own size: the
/// same arithmetic as the general process, with what a filter phase of
/// zero makes of it taken directly. A phase of zero is the filter `128 *
/// sample`, so its pass is a shift (`128 * s` is a multiple of `1 <<
/// round0`, and the intermediate times 128 of `1 << round1`); and with a
/// vertical phase of zero only the `h` rows that filter reads are computed,
/// not the seven around them.
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn block_inter_prediction_unscaled(
    refp: &PlaneBuf,
    last_x: i32,
    last_y: i32,
    x: i64,
    y: i64,
    w: usize,
    h: usize,
    [filt_y, filt_x]: [usize; 2],
    rv: RoundingVars,
    pred: &mut [i32],
) {
    let phase_x = ((x >> 6) & SUBPEL_MASK as i64) as usize;
    let phase_y = (((y & 1023) >> 6) & SUBPEL_MASK as i64) as usize;
    let fx = &SUBPEL_FILTERS[filt_x][phase_x];
    let fy = &SUBPEL_FILTERS[filt_y][phase_y];
    let x0 = (x >> 10) - 3;
    let y0 = (y >> 10) - 3;
    let inside_x = x0 >= 0 && x0 + w as i64 + 7 <= last_x as i64 + 1;
    let mut padded = [0u16; 128 + 8];
    // Row `r` of the intermediate array reads this row of the reference:
    // `w + 7` samples from `x0`, the edges repeated.
    let source = |r: usize, padded: &mut [u16; 128 + 8], out: &mut dyn FnMut(&[u16])| {
        let yy = clip3(0, last_y, (y0 + r as i64) as i32) as usize;
        let row = refp.row(yy);
        if inside_x {
            out(&row[x0 as usize..x0 as usize + w + 7])
        } else {
            for (c, v) in padded[..w + 7].iter_mut().enumerate() {
                *v = row[clip3(0, last_x, (x0 + c as i64) as i32) as usize];
            }
            out(&padded[..w + 7])
        }
    };
    let (round0, round1) = (rv.round0, rv.round1);
    if phase_y == 0 {
        if phase_x == 0 {
            let shift = 14 - round0 - round1;
            for (r, out) in pred[..w * h].chunks_exact_mut(w).enumerate() {
                source(r + 3, &mut padded, &mut |src| {
                    for (o, &s) in out.iter_mut().zip(&src[3..]) {
                        *o = (s as i32) << shift;
                    }
                });
            }
        } else {
            let add = (1 << round1) >> 1;
            let mut line = [0i16; 128];
            for (r, out) in pred[..w * h].chunks_exact_mut(w).enumerate() {
                source(r + 3, &mut padded, &mut |src| {
                    crate::dsp::mc::filter_row16(src, fx, round0, &mut line[..w]);
                });
                for (o, &v) in out.iter_mut().zip(&line[..w]) {
                    *o = (128 * v as i32 + add) >> round1;
                }
            }
        }
        return;
    }
    // The intermediate array: a per-thread buffer, reused.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<i16>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let inter_h = h + 7;
    SCRATCH.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < inter_h * w {
            buf.resize(inter_h * w, 0);
        }
        let intermediate = &mut buf[..inter_h * w];
        for (r, out) in intermediate.chunks_exact_mut(w).enumerate() {
            source(r, &mut padded, &mut |src| {
                if phase_x == 0 {
                    let shift = 7 - round0;
                    for (o, &s) in out.iter_mut().zip(&src[3..]) {
                        *o = (s << shift) as i16;
                    }
                } else {
                    crate::dsp::mc::filter_row16(src, fx, round0, out);
                }
            });
        }
        for (r, out) in pred[..w * h].chunks_exact_mut(w).enumerate() {
            crate::dsp::mc::filter_col16(&intermediate[r * w..], w, fy, round1, out);
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn block_inter_prediction_with(
    refp: &PlaneBuf,
    last_x: i32,
    last_y: i32,
    x: i64,
    y: i64,
    x_step: i64,
    y_step: i64,
    w: usize,
    h: usize,
    [filt_y, filt_x]: [usize; 2],
    inter_h: usize,
    rv: RoundingVars,
    pred: &mut [i32],
    intermediate: &mut [i32],
) {
    if x_step == 1 << SCALE_SUBPEL_BITS && y_step == 1 << SCALE_SUBPEL_BITS {
        // Unscaled: one filter phase per direction for the whole block.
        let fx = &SUBPEL_FILTERS[filt_x][((x >> 6) & SUBPEL_MASK as i64) as usize];
        let fy = &SUBPEL_FILTERS[filt_y][(((y & 1023) >> 6) & SUBPEL_MASK as i64) as usize];
        let x0 = (x >> 10) - 3;
        let inside_x = x0 >= 0 && x0 + w as i64 + 7 <= last_x as i64 + 1;
        let mut padded = [0u16; 128 + 8];
        for r in 0..inter_h {
            let yy = clip3(0, last_y, ((y >> 10) + r as i64 - 3) as i32) as usize;
            let row = refp.row(yy);
            let src: &[u16] = if inside_x {
                &row[x0 as usize..x0 as usize + w + 7]
            } else {
                for (c, v) in padded[..w + 7].iter_mut().enumerate() {
                    *v = row[clip3(0, last_x, (x0 + c as i64) as i32) as usize];
                }
                &padded[..w + 7]
            };
            let out = &mut intermediate[r * w..(r + 1) * w];
            crate::dsp::mc::filter_row(src, fx, rv.round0, out);
        }
        for r in 0..h {
            let out = &mut pred[r * w..(r + 1) * w];
            crate::dsp::mc::filter_col(intermediate, r, w, fy, rv.round1, out);
        }
        return;
    }
    for r in 0..inter_h {
        let yy = clip3(0, last_y, ((y >> 10) + r as i64 - 3) as i32) as usize;
        let row = refp.row(yy);
        for c in 0..w {
            let p = x + x_step * c as i64;
            let f = &SUBPEL_FILTERS[filt_x][((p >> 6) & SUBPEL_MASK as i64) as usize];
            let base = (p >> 10) - 3;
            let mut s = 0i32;
            for t in 0..8 {
                let xx = clip3(0, last_x, (base + t as i64) as i32) as usize;
                s += f[t] * row[xx] as i32;
            }
            intermediate[r * w + c] = round2(s, rv.round0);
        }
    }
    for r in 0..h {
        let p = (y & 1023) + y_step * r as i64;
        let f = &SUBPEL_FILTERS[filt_y][((p >> 6) & SUBPEL_MASK as i64) as usize];
        let base = (p >> 10) as usize;
        for c in 0..w {
            let mut s = 0i32;
            for t in 0..8 {
                s += f[t] * intermediate[(base + t) * w + c];
            }
            pred[r * w + c] = round2(s, rv.round1);
        }
    }
}

/// The setup shear process (7.11.3.6): `warpValid, alpha, beta, gamma,
/// delta`.
pub(crate) fn setup_shear(wp: &[i32; 6]) -> (bool, i32, i32, i32, i32) {
    let alpha0 = clip3(-32768, 32767, wp[2] - (1 << WARPEDMODEL_PREC_BITS));
    let beta0 = clip3(-32768, 32767, wp[3]);
    let (div_shift, div_factor) = resolve_divisor(wp[2] as i64);
    let v = (wp[4] as i64) << WARPEDMODEL_PREC_BITS;
    let gamma0 = round2signed_64(v * div_factor, div_shift as u32).clamp(-32768, 32767) as i32;
    let w = wp[3] as i64 * wp[4] as i64;
    let delta0 = (wp[5] as i64
        - round2signed_64(w * div_factor, div_shift as u32)
        - (1 << WARPEDMODEL_PREC_BITS))
        .clamp(-32768, 32767) as i32;
    let rb = WARP_PARAM_REDUCE_BITS as u32;
    let alpha = round2signed(alpha0, rb) << rb;
    let beta = round2signed(beta0, rb) << rb;
    let gamma = round2signed(gamma0, rb) << rb;
    let delta = round2signed(delta0, rb) << rb;
    let mut valid = true;
    if 4 * alpha.abs() + 7 * beta.abs() >= (1 << WARPEDMODEL_PREC_BITS) {
        valid = false;
    }
    if 4 * gamma.abs() + 4 * delta.abs() >= (1 << WARPEDMODEL_PREC_BITS) {
        valid = false;
    }
    (valid, alpha, beta, gamma, delta)
}

/// The resolve divisor process (7.11.3.7): `(divShift, divFactor)`.
pub(crate) fn resolve_divisor(d: i64) -> (i32, i64) {
    let a = d.unsigned_abs();
    let n = 63 - a.leading_zeros() as i32;
    let e = a as i64 - (1i64 << n);
    let f = if n > DIV_LUT_BITS {
        round2_64(e, (n - DIV_LUT_BITS) as u32)
    } else {
        e << (DIV_LUT_BITS - n)
    };
    let div_shift = n + DIV_LUT_PREC_BITS;
    let factor = DIV_LUT[f as usize] as i64;
    (div_shift, if d < 0 { -factor } else { factor })
}

fn obmc_mask(len: usize) -> &'static [i32] {
    match len {
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        8 => &OBMC_MASK_8,
        16 => &OBMC_MASK_16,
        _ => &OBMC_MASK_32,
    }
}

fn sm_weights(log2: usize) -> &'static [i32] {
    match log2 {
        2 => &SM_WEIGHTS_TX_4X4,
        3 => &SM_WEIGHTS_TX_8X8,
        4 => &SM_WEIGHTS_TX_16X16,
        5 => &SM_WEIGHTS_TX_32X32,
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

/// The intra edge filter strength selection process (7.11.2.9).
#[allow(clippy::if_same_then_else)]
fn intra_edge_filter_strength(w: i32, h: i32, filter_type: bool, delta: i32) -> usize {
    let d = delta.abs();
    let blk_wh = w + h;
    let mut strength = 0;
    if !filter_type {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 12 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

/// The intra edge upsample selection process (7.11.2.10).
fn intra_edge_upsample(w: i32, h: i32, filter_type: bool, delta: i32) -> bool {
    let d = delta.abs();
    let blk_wh = w + h;
    if d <= 0 || d >= 40 {
        false
    } else if !filter_type {
        blk_wh <= 16
    } else {
        blk_wh <= 8
    }
}

/// The intra edge filter process (7.11.2.12) on an edge array (index 0 of
/// the specification's array at `EDGE_OFF`).
fn intra_edge_filter(buf: &mut [i32; EDGE_LEN], sz: usize, strength: usize) {
    if strength == 0 {
        return;
    }
    let mut edge = [0i32; 160];
    for i in 0..sz {
        edge[i] = buf[EDGE_OFF + i - 1];
    }
    for i in 1..sz {
        let mut s = 0;
        for j in 0..5 {
            let k = clip3(0, sz as i32 - 1, i as i32 - 2 + j as i32) as usize;
            s += INTRA_EDGE_KERNEL[strength - 1][j] * edge[k];
        }
        buf[EDGE_OFF + i - 1] = (s + 8) >> 4;
    }
}

/// The intra edge upsample process (7.11.2.11).
fn intra_edge_upsample_apply(buf: &mut [i32; EDGE_LEN], num_px: usize, maxv: i32) {
    let mut dup = [0i32; 64];
    dup[0] = buf[EDGE_OFF - 1];
    for i in -1..num_px as isize {
        dup[(i + 2) as usize] = buf[(EDGE_OFF as isize + i) as usize];
    }
    dup[num_px + 2] = buf[EDGE_OFF + num_px - 1];
    buf[EDGE_OFF - 2] = dup[0];
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        let s = round2(s, 4).clamp(0, maxv);
        buf[EDGE_OFF + 2 * i - 1] = s;
        buf[EDGE_OFF + 2 * i] = dup[i + 2];
    }
}

/// `WedgeMasks`, generated once (7.11.3.11).
struct WedgeMasks {
    /// `[bsize][flipSign][wedge]` -> w*h mask, for sizes with wedges.
    masks: Vec<Vec<u8>>,
}

fn wedge_masks() -> &'static WedgeMasks {
    static M: OnceLock<WedgeMasks> = OnceLock::new();
    M.get_or_init(|| {
        const N: usize = MASK_MASTER_SIZE;
        let mut master = vec![[[0i32; N]; N]; 6];
        const OBL63: usize = 3;
        const VERT: usize = 1;
        const OBL27: usize = 2;
        const OBL117: usize = 4;
        const OBL153: usize = 5;
        const HORZ: usize = 0;
        let w = N;
        let h = N;
        for j in 0..w {
            let mut shift = N as i32 / 4;
            let mut i = 0;
            while i < h {
                master[OBL63][i][j] =
                    WEDGE_MASTER_OBLIQUE_EVEN[clip3(0, N as i32 - 1, j as i32 - shift) as usize];
                shift -= 1;
                master[OBL63][i + 1][j] =
                    WEDGE_MASTER_OBLIQUE_ODD[clip3(0, N as i32 - 1, j as i32 - shift) as usize];
                master[VERT][i][j] = WEDGE_MASTER_VERTICAL[j];
                master[VERT][i + 1][j] = WEDGE_MASTER_VERTICAL[j];
                i += 2;
            }
        }
        for i in 0..h {
            for j in 0..w {
                let msk = master[OBL63][i][j];
                master[OBL27][j][i] = msk;
                master[OBL117][i][w - 1 - j] = 64 - msk;
                master[OBL153][w - 1 - j][i] = 64 - msk;
                master[HORZ][j][i] = master[VERT][i][j];
            }
        }
        let mut masks = vec![Vec::new(); BLOCK_SIZES * 2 * 16];
        for bsize in BLOCK_8X8..BLOCK_SIZES {
            if WEDGE_BITS[bsize] == 0 {
                continue;
            }
            let w = block_width(bsize);
            let h = block_height(bsize);
            for wedge in 0..16 {
                let w4 = NUM_4X4_BLOCKS_WIDE[bsize];
                let h4 = NUM_4X4_BLOCKS_HIGH[bsize];
                let shape = if h4 > w4 {
                    0
                } else if h4 < w4 {
                    1
                } else {
                    2
                };
                let cb = WEDGE_CODEBOOK[shape][wedge];
                let dir = cb[0];
                let xoff = N / 2 - ((cb[1] * w) >> 3);
                let yoff = N / 2 - ((cb[2] * h) >> 3);
                let mut sum = 0;
                for i in 0..w {
                    sum += master[dir][yoff][xoff + i];
                }
                for i in 1..h {
                    sum += master[dir][yoff + i][xoff];
                }
                let avg = (sum + (w + h - 1) as i32 / 2) / (w + h - 1) as i32;
                let flip_sign = (avg < 32) as usize;
                let mut m0 = vec![0u8; w * h];
                let mut m1 = vec![0u8; w * h];
                for i in 0..h {
                    for j in 0..w {
                        let v = master[dir][yoff + i][xoff + j];
                        m0[i * w + j] = v as u8;
                        m1[i * w + j] = (64 - v) as u8;
                    }
                }
                masks[(bsize * 2 + flip_sign) * 16 + wedge] = m0;
                masks[(bsize * 2 + (1 - flip_sign)) * 16 + wedge] = m1;
            }
        }
        WedgeMasks { masks }
    })
}

/// `WedgeMasks[ bsize ][ sign ][ index ]`, row-major `Block_Width` wide.
pub(crate) fn wedge_mask(bsize: usize, sign: usize, index: usize) -> &'static [u8] {
    &wedge_masks().masks[(bsize * 2 + sign) * 16 + index]
}
