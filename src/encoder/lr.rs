//! The encoder's loop restoration search: each unit's filter (none, a
//! Wiener filter fitted to the source, or a self-guided filter with fitted
//! projection weights), and each plane's frame restoration type.
//!
//! Everything is measured with the decoder's own restoration
//! (`postfilter::restore_rect`), over the samples it reads (the stripes'
//! boundary rows come from the deblocked frame), so the error of a choice
//! is the error the decoder's output will have.

use crate::consts::*;
use crate::decoder::FrameCtx;
use crate::decoder::parallel_map;
use crate::decoder::postfilter::{
    LRB, LrBuffers, LrCtx, LrParams, box_filter_rect, lr_stripe, lr_unit_cols, lr_window,
    restore_rect_from_window,
};
use crate::decoder::state::FrameBuf;
use crate::dsp::enc::sse_row;
use crate::tables::*;

/// One unit's restoration, as `read_lr_unit()` codes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrChoice {
    /// `RESTORE_NONE`, `RESTORE_WIENER` or `RESTORE_SGRPROJ`.
    pub(crate) t: u8,
    pub(crate) wiener: [[i32; 3]; 2],
    pub(crate) set: u8,
    pub(crate) xqd: [i32; 2],
}

/// The frame's loop restoration: per plane, `FrameRestorationType` and
/// each unit's choice.
#[derive(Clone, Debug, Default)]
pub(crate) struct LrPlan {
    pub(crate) frame_type: [u8; 3],
    pub(crate) units: [Vec<LrChoice>; 3],
}

/// The self-guided parameter sets tried.
const SGR_SETS_NORMAL: [usize; 6] = [0, 3, 6, 10, 13, 14];
const SGR_SETS_FAST: [usize; 2] = [3, 10];

/// A stripe's part of a unit row: `StripeStartY`, `StripeEndY`, and the
/// first and last rows it restores.
type Stripe = (i32, i32, i32, i32);

/// How hard the search works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effort {
    /// Filters fitted on every other row and column; two self-guided
    /// parameter sets.
    Fast,
    /// Six self-guided parameter sets.
    Normal,
    /// All sixteen.
    Thorough,
}

/// Chooses the restoration of every unit of every plane of `f` (whose
/// header asks for switchable restoration in each, so its unit arrays are
/// set up), on `threads` threads. `cur` is the deblocked frame, `cdef` the
/// CDEF output; `src` the source planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    f: &FrameCtx,
    cur: &FrameBuf,
    cdef: &FrameBuf,
    src: &[Vec<u16>],
    stride: &[usize],
    lambda: f64,
    effort: Effort,
    threads: usize,
) -> LrPlan {
    let mut plan = LrPlan::default();
    let sets: Vec<usize> = match effort {
        Effort::Thorough => (0..16).collect(),
        Effort::Normal => SGR_SETS_NORMAL.to_vec(),
        Effort::Fast => SGR_SETS_FAST.to_vec(),
    };
    let step = if effort == Effort::Fast { 2 } else { 1 };
    let stripes = (f.hdr.frame_height as usize + 8).div_ceil(64);
    // The stripes of each unit row of each plane.
    let mut rows: [Vec<Vec<Stripe>>; 3] = Default::default();
    // Every unit of every plane, one job each.
    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for plane in 0..f.num_planes {
        let lp = &f.lr[plane];
        rows[plane] = vec![Vec::new(); lp.unit_rows];
        for s in 0..stripes {
            if let Some((ss, se, y0, y1, ur)) = lr_stripe(f, plane, s) {
                rows[plane][ur].push((ss, se, y0, y1));
            }
        }
        for ur in 0..lp.unit_rows {
            for uc in 0..lp.unit_cols {
                jobs.push((plane, ur, uc));
            }
        }
    }
    let unit = |&(plane, ur, uc): &(usize, usize, usize)| -> Option<LrChoice> {
        let (sub_x, sub_y) = f.plane_ss(plane);
        let plane_end_x = crate::consts::round2(f.hdr.upscaled_width as i32, sub_x as u32) - 1;
        let plane_end_y = crate::consts::round2(f.hdr.frame_height as i32, sub_y as u32) - 1;
        let (x0, x1) = lr_unit_cols(f, plane, uc);
        if x0 > plane_end_x || rows[plane][ur].is_empty() {
            return None;
        }
        let mut buf = LrBuffers::default();
        let w = (x1 - x0 + 1) as usize;
        // The unit's rectangles, one per stripe.
        let rects: Vec<(LrCtx, i32, usize)> = rows[plane][ur]
            .iter()
            .map(|&(ss, se, y0, y1)| {
                (
                    LrCtx {
                        cur: &cur.planes[plane],
                        cdef: &cdef.planes[plane],
                        stripe_start_y: ss,
                        stripe_end_y: se,
                        plane_end_x,
                        plane_end_y,
                    },
                    y0,
                    (y1 - y0 + 1) as usize,
                )
            })
            .collect();
        let sse_of = |buf: &mut LrBuffers, params: Option<&LrParams>| -> u64 {
            let mut total = 0u64;
            for (ctx, y0, h) in &rects {
                let st = stride[plane];
                match params {
                    Some(p) => {
                        lr_window(ctx, x0, *y0, w, *h, buf);
                        restore_rect_from_window(f, ctx, x0, *y0, w, *h, p, buf);
                        for i in 0..*h {
                            let so = (*y0 as usize + i) * st + x0 as usize;
                            total += sse_row(&buf.out[i * w..(i + 1) * w], &src[plane][so..so + w]);
                        }
                    }
                    None => {
                        let pl = ctx.cdef;
                        for i in 0..*h {
                            let y = *y0 as usize + i;
                            let so = y * st + x0 as usize;
                            let co = y * pl.stride + x0 as usize;
                            total += sse_row(&pl.data[co..co + w], &src[plane][so..so + w]);
                        }
                    }
                }
            }
            total
        };
        let none = sse_of(&mut buf, None);
        let mut best = (none as f64 + lambda, LrChoice::default());
        // Wiener.
        let coef = fit_wiener(plane, &rects, x0, w, src, stride, step, &mut buf);
        let p = LrParams::Wiener(coef);
        let e = sse_of(&mut buf, Some(&p));
        let cost = e as f64 + lambda * if plane == 0 { 40.0 } else { 28.0 };
        if cost < best.0 {
            best = (
                cost,
                LrChoice {
                    t: RESTORE_WIENER,
                    wiener: coef,
                    ..Default::default()
                },
            );
        }
        // Self-guided.
        for &set in &sets {
            let (xqd, e) = fit_sgr(f, set, &rects, x0, w, src, stride, plane, step, &mut buf);
            let cost = e as f64 + lambda * 20.0;
            if cost < best.0 {
                best = (
                    cost,
                    LrChoice {
                        t: RESTORE_SGRPROJ,
                        set: set as u8,
                        xqd,
                        ..Default::default()
                    },
                );
            }
        }
        Some(best.1)
    };
    let results = parallel_map(jobs.len(), threads.max(1), |i| unit(&jobs[i]));
    for plane in 0..f.num_planes {
        let lp = &f.lr[plane];
        plan.units[plane] = vec![LrChoice::default(); lp.unit_rows * lp.unit_cols];
    }
    for (&(plane, ur, uc), choice) in jobs.iter().zip(results) {
        if let Some(c) = choice {
            plan.units[plane][ur * f.lr[plane].unit_cols + uc] = c;
        }
    }
    for plane in 0..f.num_planes {
        let used = |t: u8| plan.units[plane].iter().any(|c| c.t == t);
        plan.frame_type[plane] = match (used(RESTORE_WIENER), used(RESTORE_SGRPROJ)) {
            (false, false) => RESTORE_NONE,
            (true, false) => RESTORE_WIENER,
            (false, true) => RESTORE_SGRPROJ,
            (true, true) => RESTORE_SWITCHABLE,
        };
    }
    plan
}

/// Solves the normal equations `m x = v` (Gaussian elimination with
/// partial pivoting); zeros when singular.
fn solve<const N: usize>(mut m: [[f64; N]; N], mut v: [f64; N]) -> [f64; N] {
    for i in 0..N {
        let piv = (i..N)
            .max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))
            .unwrap_or(i);
        m.swap(i, piv);
        v.swap(i, piv);
        if m[i][i].abs() < 1e-9 {
            return [0.0; N];
        }
        for r in 0..N {
            if r != i {
                let f = m[r][i] / m[i][i];
                for c in i..N {
                    m[r][c] -= f * m[i][c];
                }
                v[r] -= f * v[i];
            }
        }
    }
    std::array::from_fn(|i| v[i] / m[i][i])
}

/// A separable symmetric Wiener filter for the unit, fitted by least
/// squares alternately in each direction, then quantised to the coded
/// ranges: `[vertical, horizontal]` coefficients (outermost first).
#[allow(clippy::too_many_arguments)]
fn fit_wiener(
    plane: usize,
    rects: &[(LrCtx, i32, usize)],
    x0: i32,
    w: usize,
    src: &[Vec<u16>],
    stride: &[usize],
    step: usize,
    buf: &mut LrBuffers,
) -> [[i32; 3]; 2] {
    // Chroma filters have no outermost tap.
    let first = if plane == 0 { 0 } else { 1 };
    let ww = w + 2 * LRB;
    // Coefficients as fractions of 128: index 0 outermost.
    let mut hc = [0f64; 3];
    let mut vc = [0f64; 3];
    // The other direction's filter at the samples the fit reads, each
    // worked out once rather than for each of the seven positions whose
    // line it lies on: `pre[r * ww + c]` (the same sums as before, in the
    // same order, so the same values).
    let mut pre = Vec::new();
    for iter in 0..3 {
        let vertical = iter % 2 == 0;
        let other = if vertical { hc } else { vc };
        let mut m = [[0f64; 3]; 3];
        let mut v = [0f64; 3];
        for (ctx, y0, h) in rects {
            lr_window(ctx, x0, *y0, w, *h, buf);
            let win = &buf.win;
            let at = |r: usize, c: usize| win[r * ww + c] as f64;
            // The other direction at (r, c).
            let filtered = |r: usize, c: usize| {
                let center = at(r, c);
                let mut s = center;
                for k in 0..3 {
                    let d = 3 - k;
                    let (a, b) = if vertical {
                        (at(r, c - d), at(r, c + d))
                    } else {
                        (at(r - d, c), at(r + d, c))
                    };
                    s += other[k] * (a + b - 2.0 * center);
                }
                s
            };
            pre.resize(ww * (*h + 2 * LRB), 0.0);
            if vertical {
                // Every row, the columns of the positions fitted.
                for r in 0..*h + 2 * LRB {
                    for j in (0..w).step_by(step) {
                        pre[r * ww + j + LRB] = filtered(r, j + LRB);
                    }
                }
            } else {
                // The rows of the positions fitted, every column.
                for i in (0..*h).step_by(step) {
                    for c in 0..ww {
                        pre[(i + LRB) * ww + c] = filtered(i + LRB, c);
                    }
                }
            }
            for i in (0..*h).step_by(step) {
                let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
                for j in (0..w).step_by(step) {
                    // The other direction's filter applied first: the
                    // samples along this direction (7 of them).
                    let line: [f64; 7] = std::array::from_fn(|t| {
                        if vertical {
                            pre[(i + t) * ww + j + LRB]
                        } else {
                            pre[(i + LRB) * ww + j + t]
                        }
                    });
                    let target = src[plane][so + j] as f64 - line[3];
                    let d: [f64; 3] =
                        std::array::from_fn(|k| line[k] + line[6 - k] - 2.0 * line[3]);
                    for a in first..3 {
                        for b in first..3 {
                            m[a][b] += d[a] * d[b];
                        }
                        v[a] += d[a] * target;
                    }
                }
            }
        }
        let sol = if first == 0 {
            solve(m, v)
        } else {
            let s2 = solve([[m[1][1], m[1][2]], [m[2][1], m[2][2]]], [v[1], v[2]]);
            [0.0, s2[0], s2[1]]
        };
        if vertical {
            vc = sol;
        } else {
            hc = sol;
        }
    }
    let q = |c: [f64; 3]| -> [i32; 3] {
        std::array::from_fn(|k| {
            if k < first {
                0
            } else {
                ((c[k] * 128.0).round() as i32).clamp(WIENER_TAPS_MIN[k], WIENER_TAPS_MAX[k])
            }
        })
    };
    [q(vc), q(hc)]
}

/// The self-guided projection weights `xqd` for parameter set `set`,
/// fitted by least squares and clamped to the coded ranges (the second
/// derived from the first when the second filter is off, as the decoder
/// derives it).
#[allow(clippy::too_many_arguments)]
fn fit_sgr(
    f: &FrameCtx,
    set: usize,
    rects: &[(LrCtx, i32, usize)],
    x0: i32,
    w: usize,
    src: &[Vec<u16>],
    stride: &[usize],
    plane: usize,
    step: usize,
    buf: &mut LrBuffers,
) -> ([i32; 2], u64) {
    let r0 = SGR_PARAMS[set][0];
    let r1 = SGR_PARAMS[set][2];
    let mut m = [[0f64; 2]; 2];
    let mut v = [0f64; 2];
    // The two filters' outputs over every rectangle, kept for the error.
    let mut flts: Vec<[Vec<i32>; 2]> = Vec::with_capacity(rects.len());
    for (ctx, y0, h) in rects {
        lr_window(ctx, x0, *y0, w, *h, buf);
        for pass in 0..2 {
            box_filter_rect(f, ctx, x0, *y0, w, *h, set, pass, buf);
        }
        flts.push([buf.flt[0].clone(), buf.flt[1].clone()]);
        for i in (0..*h).step_by(step) {
            let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
            for j in (0..w).step_by(step) {
                let u = ((ctx.cdef.get(x0 as usize + j, *y0 as usize + i) as i32)
                    << SGRPROJ_RST_BITS) as f64;
                let a = if r0 != 0 {
                    buf.flt[0][i * w + j] as f64 - u
                } else {
                    0.0
                };
                let b = if r1 != 0 {
                    buf.flt[1][i * w + j] as f64 - u
                } else {
                    0.0
                };
                let t = 128.0 * (((src[plane][so + j] as i32) << SGRPROJ_RST_BITS) as f64 - u);
                m[0][0] += a * a;
                m[0][1] += a * b;
                m[1][1] += b * b;
                v[0] += a * t;
                v[1] += b * t;
            }
        }
    }
    m[1][0] = m[0][1];
    let (w0, w2) = if r0 != 0 && r1 != 0 {
        let s = solve(m, v);
        (s[0], s[1])
    } else if r0 != 0 {
        (if m[0][0] > 0.0 { v[0] / m[0][0] } else { 0.0 }, 0.0)
    } else {
        (0.0, if m[1][1] > 0.0 { v[1] / m[1][1] } else { 0.0 })
    };
    let x0q = if r0 != 0 {
        (w0.round() as i32).clamp(SGRPROJ_XQD_MIN[0], SGRPROJ_XQD_MAX[0])
    } else {
        0
    };
    let x1q = if r1 != 0 {
        ((1 << SGRPROJ_PRJ_BITS) - x0q - w2.round() as i32)
            .clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1])
    } else {
        ((1 << SGRPROJ_PRJ_BITS) - x0q).clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1])
    };
    // The error of the restoration with these weights (7.17.2's
    // projection, as `restore_rect` applies it).
    let (w0, w1) = (x0q, x1q);
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let maxv = (1i32 << f.bit_depth) - 1;
    let mut sse = 0u64;
    for ((ctx, y0, h), flt) in rects.iter().zip(&flts) {
        for i in 0..*h {
            let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
            for j in 0..w {
                let u =
                    (ctx.cdef.get(x0 as usize + j, *y0 as usize + i) as i32) << SGRPROJ_RST_BITS;
                let mut val = w1 * u;
                val += if r0 != 0 {
                    w0 * flt[0][i * w + j]
                } else {
                    w0 * u
                };
                val += if r1 != 0 {
                    w2 * flt[1][i * w + j]
                } else {
                    w2 * u
                };
                let out = round2(val, (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32).clamp(0, maxv);
                let d = out as i64 - src[plane][so + j] as i64;
                sse += (d * d) as u64;
            }
        }
    }
    ([x0q, x1q], sse)
}
