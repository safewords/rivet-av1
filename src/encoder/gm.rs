//! Global motion: a warp of each reference that the frame's GLOBALMV
//! blocks are predicted with (7.10.2.1, 7.11.3.5), estimated by the
//! encoder and coded in the frame header (5.9.24).
//!
//! The estimate is the textbook one. Each textured block of the source is
//! matched in the reference (a full search at a quarter of the size, then
//! refined at half and full size, the sub-sample position from a parabola
//! through the matching costs around the best), which gives a cloud of
//! correspondences; a model is fitted to them by random sample consensus
//! (two correspondences fix a rotation and zoom; the model kept is the one
//! most correspondences agree with, then fitted again by least squares to
//! those that agree); and the model is kept only when it is the motion of
//! most of the frame and explains more of it than no motion does. A model
//! that is nearly a translation is coded as one.

use crate::bits::BitWriter;
use crate::consts::*;
use crate::decoder::predict::setup_shear;

/// The identity model's parameters.
pub(crate) const IDENTITY_PARAMS: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];

/// `global_motion_params()` (5.9.24): each reference's `gm_type` and the
/// parameters its type codes, against `prev` (`PrevGmParams`). The
/// parameters must be on the grid each one is coded on.
pub(crate) fn write_params(w: &mut BitWriter, gm: &[(u32, [i32; 6]); 8], prev: &[[i32; 6]; 8]) {
    for rf in LAST_FRAME as usize..=ALTREF_FRAME as usize {
        let (ty, params) = gm[rf];
        w.flag(ty != IDENTITY); // is_global
        if ty != IDENTITY {
            w.flag(ty == ROTZOOM); // is_rot_zoom
            if ty != ROTZOOM {
                w.flag(ty == TRANSLATION); // is_translation
            }
        }
        let mut param = |idx: usize| {
            let (prec_diff, round, sub, mx) = param_coding(ty, idx);
            let r = (prev[rf][idx] >> prec_diff) - sub;
            let v = (params[idx] - round) >> prec_diff;
            debug_assert_eq!((v << prec_diff) + round, params[idx], "off the grid");
            write_signed_subexp_with_ref(w, -mx, mx + 1, r, v);
        };
        if ty >= ROTZOOM {
            param(2);
            param(3);
            if ty == AFFINE {
                param(4);
                param(5);
            }
        }
        if ty >= TRANSLATION {
            param(0);
            param(1);
        }
    }
}

/// How parameter `idx` of a model of type `ty` is coded (5.9.25, without
/// high-precision vectors): `(precDiff, round, sub, mx)`.
fn param_coding(ty: u32, idx: usize) -> (u32, i32, i32, i32) {
    let (abs_bits, prec_bits) = if idx < 2 {
        if ty == TRANSLATION {
            // allow_high_precision_mv is 0 in the encoder's frames.
            (GM_ABS_TRANS_ONLY_BITS - 1, GM_TRANS_ONLY_PREC_BITS - 1)
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let (round, sub) = if idx % 3 == 2 {
        (1i32 << WARPEDMODEL_PREC_BITS, 1i32 << prec_bits)
    } else {
        (0, 0)
    };
    (prec_diff, round, sub, 1i32 << abs_bits)
}

/// The inverse of `decode_signed_subexp_with_ref( low, high, r )`.
fn write_signed_subexp_with_ref(w: &mut BitWriter, low: i32, high: i32, r: i32, v: i32) {
    let (mx, r, x) = (high - low, r - low, v - low);
    // The position of `x` in the order recentred on `r` (the inverse of
    // inverse_recenter()).
    let recenter = |r: i32, x: i32| {
        if x > 2 * r {
            x
        } else if x >= r {
            2 * (x - r)
        } else {
            2 * (r - x) - 1
        }
    };
    let pv = if (r << 1) <= mx {
        recenter(r, x)
    } else {
        recenter(mx - 1 - r, mx - 1 - x)
    };
    // decode_subexp( mx ), k = 3.
    let (mut i, mut mk, k) = (0, 0, 3);
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if mx <= mk + 3 * a {
            write_ns(w, (mx - mk) as u32, (pv - mk) as u32);
            return;
        }
        let more = pv >= mk + a;
        w.flag(more); // subexp_more_bits
        if !more {
            w.f(b2 as u32, (pv - mk) as u32); // subexp_bits
            return;
        }
        i += 1;
        mk += a;
    }
}

/// `ns( n )` coding `v`.
fn write_ns(w: &mut BitWriter, n: u32, v: u32) {
    if n <= 1 {
        return;
    }
    let wb = 32 - n.leading_zeros();
    let m = (1u32 << wb) - n;
    if v < m {
        w.f(wb - 1, v);
    } else {
        let t = v + m;
        w.f(wb - 1, t >> 1);
        w.f(1, t & 1);
    }
}

/// A luma plane as the estimate reads it: samples clamped to the edges.
struct Plane {
    data: Vec<u16>,
    w: usize,
    h: usize,
}

impl Plane {
    fn from(src: &[u16], stride: usize, w: usize, h: usize) -> Self {
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            data.extend_from_slice(&src[y * stride..y * stride + w]);
        }
        Plane { data, w, h }
    }

    /// Half the size, each sample the mean of four.
    fn half(&self) -> Self {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            let (y0, y1) = ((2 * y).min(self.h - 1), (2 * y + 1).min(self.h - 1));
            for x in 0..w {
                let (x0, x1) = ((2 * x).min(self.w - 1), (2 * x + 1).min(self.w - 1));
                let s = self.data[y0 * self.w + x0] as u32
                    + self.data[y0 * self.w + x1] as u32
                    + self.data[y1 * self.w + x0] as u32
                    + self.data[y1 * self.w + x1] as u32;
                data.push(((s + 2) >> 2) as u16);
            }
        }
        Plane { data, w, h }
    }

    /// SAD of the `n` x `n` block at `(x, y)` of `self` against the one at
    /// `(x + dx, y + dy)` of `r` (clamped to `r`'s edges); `x, y` inside.
    fn sad(&self, r: &Plane, x: usize, y: usize, n: usize, dx: i32, dy: i32) -> u32 {
        let rx = x as i32 + dx;
        let ry = y as i32 + dy;
        if rx >= 0 && ry >= 0 && rx as usize + n <= r.w && ry as usize + n <= r.h {
            let a = &self.data[y * self.w + x..];
            let b = &r.data[ry as usize * r.w + rx as usize..];
            return crate::dsp::enc::sad(a, self.w, b, r.w, n, n, u64::MAX) as u32;
        }
        let mut s = 0u32;
        for i in 0..n {
            let yy = (ry + i as i32).clamp(0, r.h as i32 - 1) as usize;
            for j in 0..n {
                let xx = (rx + j as i32).clamp(0, r.w as i32 - 1) as usize;
                let p = self.data[(y + i) * self.w + x + j] as i32;
                s += (p - r.data[yy * r.w + xx] as i32).unsigned_abs();
            }
        }
        s
    }
}

/// A block's centre and where it lies in the reference, in samples.
#[derive(Clone, Copy, Debug)]
struct Corr {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

/// A model `x' = a x + b y + c`, `y' = d x + e y + f`.
type Affine = [f64; 6];

fn apply(m: &Affine, x: f64, y: f64) -> (f64, f64) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

fn error(m: &Affine, c: &Corr) -> f64 {
    let (px, py) = apply(m, c.x, c.y);
    let (ex, ey) = (px - (c.x + c.dx), py - (c.y + c.dy));
    (ex * ex + ey * ey).sqrt()
}

/// The rotation and zoom (`a = e`, `b = -d`) fitted to `cs` by least
/// squares, coordinates relative to `(ox, oy)` for conditioning.
fn fit_rotzoom(cs: &[Corr]) -> Option<Affine> {
    let n = cs.len() as f64;
    if cs.len() < 2 {
        return None;
    }
    let (mx, my) = (
        cs.iter().map(|c| c.x).sum::<f64>() / n,
        cs.iter().map(|c| c.y).sum::<f64>() / n,
    );
    let (mut sxx, mut sa, mut sb, mut tx, mut ty) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for c in cs {
        let (x, y) = (c.x - mx, c.y - my);
        let (u, v) = (c.x + c.dx, c.y + c.dy);
        sxx += x * x + y * y;
        sa += x * u + y * v;
        sb += y * u - x * v;
        tx += u;
        ty += v;
    }
    if sxx < 1e-6 {
        return None;
    }
    let a = sa / sxx;
    let b = sb / sxx;
    let (tx, ty) = (tx / n, ty / n);
    // u = a (x - mx) + b (y - my) + tx, v = -b (x - mx) + a (y - my) + ty.
    Some([a, b, tx - a * mx - b * my, -b, a, ty + b * mx - a * my])
}

/// The affine model fitted to `cs` by least squares.
fn fit_affine(cs: &[Corr]) -> Option<Affine> {
    if cs.len() < 3 {
        return None;
    }
    let n = cs.len() as f64;
    let (mx, my) = (
        cs.iter().map(|c| c.x).sum::<f64>() / n,
        cs.iter().map(|c| c.y).sum::<f64>() / n,
    );
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    let (mut sxu, mut syu, mut sxv, mut syv, mut su, mut sv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for c in cs {
        let (x, y) = (c.x - mx, c.y - my);
        let (u, v) = (c.x + c.dx, c.y + c.dy);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxu += x * u;
        syu += y * u;
        sxv += x * v;
        syv += y * v;
        su += u;
        sv += v;
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-6 {
        return None;
    }
    let a = (sxu * syy - syu * sxy) / det;
    let b = (syu * sxx - sxu * sxy) / det;
    let d = (sxv * syy - syv * sxy) / det;
    let e = (syv * sxx - sxv * sxy) / det;
    let (u0, v0) = (su / n, sv / n);
    Some([a, b, u0 - a * mx - b * my, d, e, v0 - d * mx - e * my])
}

/// A small deterministic generator for the consensus's samples (the
/// encoder's output does not depend on anything but its input).
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// The correspondences of the textured blocks of `src` in `rf`.
fn correspondences(src: &Plane, rf: &Plane, bit_depth: u32) -> Vec<Corr> {
    let levels = [src.half(), rf.half()];
    let quarter = [levels[0].half(), levels[1].half()];
    // Blocks of 16 samples, 32 above 640 wide; searched to +-32 samples.
    let n = if src.w > 640 { 32 } else { 16 };
    let range = 8i32;
    let shift = bit_depth - 8;
    let mut out = Vec::new();
    let mut by = n;
    while by + 2 * n <= src.h {
        let mut bx = n;
        while bx + 2 * n <= src.w {
            // Texture in both directions: the block's horizontal and
            // vertical gradient energies, each at least a few levels per
            // sample (a block with an edge one way only has no position
            // along it).
            let (mut gx, mut gy) = (0u64, 0u64);
            for i in 0..n {
                let row = &src.data[(by + i) * src.w + bx..][..n + 1];
                let below = &src.data[(by + i + 1) * src.w + bx..][..n];
                for j in 0..n {
                    let d = (row[j + 1] as i64 - row[j] as i64) >> shift;
                    gx += (d * d) as u64;
                    let d = (below[j] as i64 - row[j] as i64) >> shift;
                    gy += (d * d) as u64;
                }
            }
            let area = (n * n) as u64;
            if gx.min(gy) < 16 * area {
                bx += n;
                continue;
            }
            // A quarter size: every vector in range.
            let (qx, qy, qn) = (bx / 4, by / 4, n / 4);
            let mut best = (u32::MAX, 0i32, 0i32);
            for dy in -range..=range {
                for dx in -range..=range {
                    let s = quarter[0].sad(&quarter[1], qx, qy, qn, dx, dy);
                    // Ties to the shorter vector.
                    let s = s * 64 + (dx.abs() + dy.abs()) as u32;
                    if s < best.0 {
                        best = (s, dx, dy);
                    }
                }
            }
            // Half size, then full size: the neighbours of the vector
            // doubled.
            let (mut vx, mut vy) = (best.1, best.2);
            for (lvl, (bxl, byl, nl)) in [(bx / 2, by / 2, n / 2), (bx, by, n)].iter().enumerate() {
                let (a, b) = if lvl == 0 {
                    (&levels[0], &levels[1])
                } else {
                    (src, rf)
                };
                let (cx, cy) = (2 * vx, 2 * vy);
                let mut best = (u32::MAX, cx, cy);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let s = a.sad(b, *bxl, *byl, *nl, cx + dx, cy + dy);
                        if s < best.0 {
                            best = (s, cx + dx, cy + dy);
                        }
                    }
                }
                (vx, vy) = (best.1, best.2);
            }
            // The sub-sample position: a parabola through the costs on
            // each side of the best.
            let c0 = src.sad(rf, bx, by, n, vx, vy) as f64;
            let sub = |m: f64, p: f64| {
                let den = m + p - 2.0 * c0;
                if den > 0.0 {
                    ((m - p) / (2.0 * den)).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let fx = sub(
                src.sad(rf, bx, by, n, vx - 1, vy) as f64,
                src.sad(rf, bx, by, n, vx + 1, vy) as f64,
            );
            let fy = sub(
                src.sad(rf, bx, by, n, vx, vy - 1) as f64,
                src.sad(rf, bx, by, n, vx, vy + 1) as f64,
            );
            out.push(Corr {
                x: bx as f64 + n as f64 / 2.0,
                y: by as f64 + n as f64 / 2.0,
                dx: vx as f64 + fx,
                dy: vy as f64 + fy,
            });
            bx += n;
        }
        by += n;
    }
    out
}

/// The model's parameters on the grid they are coded on, or `None` when
/// they are out of its range.
fn quantise(ty: u32, m: &Affine) -> Option<[i32; 6]> {
    let one = (1i64 << WARPEDMODEL_PREC_BITS) as f64;
    let mut p = IDENTITY_PARAMS;
    let raw: [f64; 6] = if ty == TRANSLATION {
        // The quirk of 7.10.2.1: a translation's first parameter is the
        // vector's row.
        [m[5] * one, m[2] * one, one, 0.0, 0.0, one]
    } else {
        [
            m[2] * one,
            m[5] * one,
            m[0] * one,
            m[1] * one,
            m[3] * one,
            m[4] * one,
        ]
    };
    let idxs: &[usize] = match ty {
        TRANSLATION => &[0, 1],
        ROTZOOM => &[0, 1, 2, 3],
        _ => &[0, 1, 2, 3, 4, 5],
    };
    for &i in idxs {
        let (prec_diff, round, _, mx) = param_coding(ty, i);
        let v = ((raw[i] - round as f64) / (1i64 << prec_diff) as f64).round() as i64;
        if v.abs() > mx as i64 {
            return None;
        }
        p[i] = ((v as i32) << prec_diff) + round;
    }
    if ty == ROTZOOM {
        p[4] = -p[3];
        p[5] = p[2];
    }
    Some(p)
}

/// The model the decoder will apply, from its parameters.
fn dequantise(ty: u32, p: &[i32; 6]) -> Affine {
    let one = (1i64 << WARPEDMODEL_PREC_BITS) as f64;
    if ty == TRANSLATION {
        [1.0, 0.0, p[1] as f64 / one, 0.0, 1.0, p[0] as f64 / one]
    } else {
        [
            p[2] as f64 / one,
            p[3] as f64 / one,
            p[0] as f64 / one,
            p[4] as f64 / one,
            p[5] as f64 / one,
            p[1] as f64 / one,
        ]
    }
}

/// A rotation and zoom about the centre of a `w` x `h` frame, for tests
/// that make every reference warp (`FORCE_GLOBAL`).
pub(crate) fn forced(w: usize, h: usize) -> Option<(u32, [i32; 6])> {
    let (a, b) = (1.0 + 1.0 / 128.0, 1.0 / 256.0);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let m: Affine = [a, b, cx - a * cx - b * cy, -b, a, cy + b * cx - a * cy];
    let p = quantise(ROTZOOM, &m)?;
    setup_shear(&p).0.then_some((ROTZOOM, p))
}

/// The global motion of the frame `src` against the reference `rf` (both
/// luma, `w` x `h`): `(gm_type, gm_params)`, or `None` when no model
/// pays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate(
    src: &[u16],
    src_stride: usize,
    rf: &[u16],
    rf_stride: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
) -> Option<(u32, [i32; 6])> {
    if w < 64 || h < 64 {
        return None;
    }
    let s = Plane::from(src, src_stride, w, h);
    let r = Plane::from(rf, rf_stride, w, h);
    let cs = correspondences(&s, &r, bit_depth);
    if cs.len() < 12 {
        return None;
    }
    // Random sample consensus over rotations and zooms: a correspondence
    // agrees with a model it is within a sample of.
    const THRESHOLD: f64 = 1.0;
    let agree = |m: &Affine| cs.iter().filter(|c| error(m, c) < THRESHOLD).count();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut best: Option<(usize, Affine)> = None;
    for _ in 0..256 {
        let i = rng.below(cs.len());
        let j = rng.below(cs.len());
        if i == j {
            continue;
        }
        let Some(m) = fit_rotzoom(&[cs[i], cs[j]]) else {
            continue;
        };
        let n = agree(&m);
        if best.is_none_or(|b| n > b.0) {
            best = Some((n, m));
        }
    }
    let (_, mut model) = best?;
    // Fitted again to the correspondences that agree, twice.
    for _ in 0..2 {
        let inl: Vec<Corr> = cs
            .iter()
            .copied()
            .filter(|c| error(&model, c) < THRESHOLD)
            .collect();
        model = fit_rotzoom(&inl)?;
    }
    let inliers: Vec<Corr> = cs
        .iter()
        .copied()
        .filter(|c| error(&model, c) < THRESHOLD)
        .collect();
    // An affine model where it explains clearly more.
    let mut ty = ROTZOOM;
    if let Some(a) = fit_affine(&inliers)
        && agree(&a) > agree(&model) + cs.len() / 20
    {
        model = a;
        ty = AFFINE;
    }
    // Nearly a translation (it moves the frame's corners by less than half
    // a sample more than its centre): coded as one.
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (mx, my) = apply(&model, cx, cy);
    let (tx, ty_) = (mx - cx, my - cy);
    let mut spread = 0f64;
    for (x, y) in [
        (0.0, 0.0),
        (w as f64, 0.0),
        (0.0, h as f64),
        (w as f64, h as f64),
    ] {
        let (px, py) = apply(&model, x, y);
        spread = spread.max(((px - x - tx).powi(2) + (py - y - ty_).powi(2)).sqrt());
    }
    let candidate = if spread < 0.5 {
        (TRANSLATION, [1.0, 0.0, tx, 0.0, 1.0, ty_])
    } else {
        (ty, model)
    };
    let params = quantise(candidate.0, &candidate.1)?;
    let coded = dequantise(candidate.0, &params);
    if candidate.0 != TRANSLATION && !setup_shear(&params).0 {
        return None;
    }
    // Kept when it is the motion of most of the frame, moves it, and
    // explains clearly more than no motion.
    let n_model = agree(&coded);
    let still: Affine = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let n_still = agree(&still);
    let (px, py) = apply(&coded, cx, cy);
    let moves = (px - cx).hypot(py - cy) >= 0.5 || spread >= 0.5;
    if 2 * n_model < cs.len() || n_model < n_still + cs.len() / 10 || !moves {
        return None;
    }
    Some((candidate.0, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rotated and zoomed copy of a textured picture gives back the
    /// rotation and zoom.
    #[test]
    fn finds_a_zoom() {
        let (w, h) = (352usize, 288usize);
        let tex = |x: f64, y: f64| {
            128.0
                + 50.0 * (x * 0.21).sin() * (y * 0.17).cos()
                + 40.0 * ((x + 2.0 * y) * 0.09).sin()
                + 30.0 * ((x * 0.05).sin() * 7.0 + y * 0.31).cos()
        };
        let mut rf = vec![0u16; w * h];
        let mut src = vec![0u16; w * h];
        let m: Affine = [1.01, 0.004, -2.0, -0.004, 1.01, 1.5];
        for y in 0..h {
            for x in 0..w {
                rf[y * w + x] = tex(x as f64, y as f64).round().clamp(0.0, 255.0) as u16;
                let (u, v) = apply(&m, x as f64, y as f64);
                src[y * w + x] = tex(u, v).round().clamp(0.0, 255.0) as u16;
            }
        }
        let (ty, p) = estimate(&src, w, &rf, w, w, h, 8).expect("a model");
        assert!(ty == ROTZOOM || ty == AFFINE);
        let got = dequantise(ty, &p);
        for (x, y) in [(0.0, 0.0), (w as f64, h as f64), (100.0, 200.0)] {
            let (a, b) = apply(&got, x, y);
            let (c, d) = apply(&m, x, y);
            assert!((a - c).abs() < 0.5 && (b - d).abs() < 0.5, "{got:?}");
        }
    }
}
