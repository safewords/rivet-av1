//! Film grain estimation (`Config::film_grain`): the encoder codes a
//! denoised picture and signals the grain it took out, for the decoder's
//! film grain synthesis (7.18.3) to put back.
//!
//! For each frame and plane, the noise is measured on 16x16 blocks: each
//! block's variance about its own plane fit (a level and two slopes), or,
//! where that is clearly less, the variance of its difference from the
//! frame before (grain is fresh in every frame, so where the picture holds
//! still the difference is two frames' noise and texture cancels); in each
//! intensity bin the flat blocks' (the median of the lowest quarter)
//! is the noise's strength there, which becomes the scaling function's
//! points. The noise's correlation is fitted on those blocks' residuals by
//! least squares, as the auto-regressive filter (lag 3) the synthesis runs
//! over white Gaussian noise; the scaling is chosen so that the synthesised
//! grain has the measured strength, the strength of the synthesis's own
//! grain for those coefficients measured once by running the same filter
//! on white noise of the strength the synthesis starts from. The plane is
//! denoised with an adaptive local Wiener filter: each sample pulled to its
//! 5x5 neighbourhood's mean by the share of the neighbourhood's variance
//! that is the noise's at its intensity.

use crate::frame::{ChromaFormat, Frame};
use crate::header::FilmGrainParams;
use crate::tables::GAUSSIAN_SEQUENCE;

/// The auto-regressive lag fitted (24 coefficients for luma).
const LAG: i32 = 3;
/// The coefficients' precision: `ar_coeff_shift_minus_6` + 6.
const AR_SHIFT: u32 = 7;
/// Intensity bins of the scaling function (its points, at most 14 for
/// luma and 10 for chroma).
const BINS: usize = 8;
/// Below this noise strength (in 8-bit units) a plane is left as it is.
const MIN_SIGMA: f64 = 0.6;

/// One plane, as `f64`, with its size.
struct Plane {
    w: usize,
    h: usize,
    v: Vec<f64>,
}

impl Plane {
    fn of(frame: &Frame, p: usize) -> Self {
        let pl = frame.planes[p];
        let (w, h) = (pl.width as usize, pl.height as usize);
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                v.push(frame.sample(p, x, y) as f64);
            }
        }
        Plane { w, h, v }
    }
}

/// A 16x16 block's statistics about its own plane fit (a level and two
/// slopes, so that gentle gradients do not count as noise).
struct Block {
    x: usize,
    y: usize,
    mean: f64,
    var: f64,
}

/// The side of the blocks the noise is measured on.
const NB: usize = 16;

/// Every whole 16x16 block's statistics, and its noise: the residual
/// about the block's plane fit (`w` x `h`, zero outside the blocks) — or,
/// where it varies less, the difference from the previous frame `prev`
/// (about its mean, over the square root of 2): grain is fresh in every
/// frame, so where the picture holds still that difference is the noise
/// of two frames, its variance and correlation those of one frame's
/// times two and alike, and texture, which a block's fit would count as
/// noise, cancels.
fn blocks(p: &Plane, prev: Option<&Plane>) -> (Vec<Block>, Vec<f64>) {
    let mut out = Vec::new();
    let mut resid = vec![0f64; p.w * p.h];
    let c = (NB as f64 - 1.0) / 2.0;
    let n = (NB * NB) as f64;
    let sxx = n * (NB * NB - 1) as f64 / 12.0;
    for by in (0..p.h.saturating_sub(NB - 1)).step_by(NB) {
        for bx in (0..p.w.saturating_sub(NB - 1)).step_by(NB) {
            // Least squares of v = a + b x + c y over the block (x and y
            // centred, so the three are independent).
            let (mut s, mut sx, mut sy) = (0.0, 0.0, 0.0);
            for y in 0..NB {
                for x in 0..NB {
                    let v = p.v[(by + y) * p.w + bx + x];
                    s += v;
                    sx += v * (x as f64 - c);
                    sy += v * (y as f64 - c);
                }
            }
            let (a, b, cc) = (s / n, sx / sxx, sy / sxx);
            let mut e = 0.0;
            for y in 0..NB {
                for x in 0..NB {
                    let i = (by + y) * p.w + bx + x;
                    let d = p.v[i] - a - b * (x as f64 - c) - cc * (y as f64 - c);
                    resid[i] = d;
                    e += d * d;
                }
            }
            let mut var = e / (n - 3.0);
            if let Some(q) = prev {
                let (mut t1, mut t2) = (0.0, 0.0);
                for y in 0..NB {
                    for x in 0..NB {
                        let i = (by + y) * p.w + bx + x;
                        let d = p.v[i] - q.v[i];
                        t1 += d;
                        t2 += d * d;
                    }
                }
                let dm = t1 / n;
                let tvar = (t2 / n - dm * dm) * n / (n - 1.0) / 2.0;
                if tvar < 0.7 * var {
                    var = tvar;
                    for y in 0..NB {
                        for x in 0..NB {
                            let i = (by + y) * p.w + bx + x;
                            resid[i] = (p.v[i] - q.v[i] - dm) / std::f64::consts::SQRT_2;
                        }
                    }
                }
            }
            out.push(Block {
                x: bx,
                y: by,
                mean: a,
                var,
            });
        }
    }
    (out, resid)
}

/// The median of the lowest quarter of `v` (sorted in place): the
/// variance of the flat blocks among them, texture not counted.
fn low_median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let q = &v[..v.len().div_ceil(4)];
    q[q.len() / 2]
}

/// The adaptive Wiener filter: each sample moved towards its 5x5
/// neighbourhood's mean by the share of the neighbourhood's variance that
/// is the noise's.
fn denoise(p: &Plane, noise_var: impl Fn(f64) -> f64) -> Vec<f64> {
    let (w, h) = (p.w, p.h);
    // Box sums by integral images.
    let mut s1 = vec![0f64; (w + 1) * (h + 1)];
    let mut s2 = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let (mut r1, mut r2) = (0.0, 0.0);
        for x in 0..w {
            let v = p.v[y * w + x];
            r1 += v;
            r2 += v * v;
            s1[(y + 1) * (w + 1) + x + 1] = s1[y * (w + 1) + x + 1] + r1;
            s2[(y + 1) * (w + 1) + x + 1] = s2[y * (w + 1) + x + 1] + r2;
        }
    }
    let mut out = vec![0f64; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(2), (y + 3).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(2), (x + 3).min(w));
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            let sum = |s: &[f64]| {
                s[y1 * (w + 1) + x1] - s[y0 * (w + 1) + x1] - s[y1 * (w + 1) + x0]
                    + s[y0 * (w + 1) + x0]
            };
            let mean = sum(&s1) / n;
            let var = (sum(&s2) / n - mean * mean).max(0.0);
            let gain = if var > 0.0 {
                ((var - noise_var(mean)) / var).max(0.0)
            } else {
                0.0
            };
            out[y * w + x] = mean + gain * (p.v[y * w + x] - mean);
        }
    }
    out
}

/// The causal neighbours of the synthesis's auto-regressive filter, in its
/// order (rows above, then the row's samples before).
fn ar_offsets() -> Vec<(isize, isize)> {
    let mut o = Vec::new();
    for dr in -LAG..=0 {
        for dc in -LAG..=LAG {
            if dr == 0 && dc == 0 {
                return o;
            }
            o.push((dr as isize, dc as isize));
        }
    }
    o
}

/// The auto-regressive coefficients of the noise `r` (a `w`-wide plane)
/// over the samples `mask` marks: the least-squares fit of each sample to
/// its causal neighbours, quantised to `AR_SHIFT` bits (damped where the
/// filter would not be stable).
fn fit_ar(r: &[f64], w: usize, h: usize, mask: &[bool]) -> Vec<i32> {
    let off = ar_offsets();
    let n = off.len();
    let mut a = vec![0f64; n * n];
    let mut b = vec![0f64; n];
    let mut x = vec![0f64; n];
    let l = LAG as usize;
    for yy in l..h {
        for xx in l..w.saturating_sub(l) {
            if !mask[yy * w + xx] {
                continue;
            }
            for (k, &(dr, dc)) in off.iter().enumerate() {
                x[k] = r[(yy as isize + dr) as usize * w + (xx as isize + dc) as usize];
            }
            let t = r[yy * w + xx];
            for i in 0..n {
                b[i] += x[i] * t;
                for j in 0..n {
                    a[i * n + j] += x[i] * x[j];
                }
            }
        }
    }
    // A little ridge keeps the system well posed on short noise.
    let tr: f64 = (0..n).map(|i| a[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        a[i * n + i] += 1e-3 * tr + 1e-9;
    }
    let c = solve(&mut a, &mut b, n);
    let quantise = |damp: f64| -> Vec<i32> {
        c.iter()
            .map(|v| ((v * damp * (1 << AR_SHIFT) as f64).round() as i32).clamp(-127, 127))
            .collect()
    };
    // An unstable fit (the synthesis's grain growing without bound, then
    // clipped) is damped until it is not.
    let mut damp = 1.0;
    let mut q = quantise(damp);
    while damp > 0.1 && synthesis_sigma(&q) > 8.0 * synthesis_sigma(&[0; 24]) {
        damp *= 0.8;
        q = quantise(damp);
    }
    q
}

/// Solves `a x = b` (`n` x `n`, symmetric positive definite) by Gaussian
/// elimination with partial pivoting.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap_or(col);
        if a[piv * n + col].abs() < 1e-12 {
            continue;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        for i in col + 1..n {
            let f = a[i * n + col] / a[col * n + col];
            for k in col..n {
                a[i * n + k] -= f * a[col * n + k];
            }
            b[i] -= f * b[col];
        }
    }
    let mut x = vec![0f64; n];
    for i in (0..n).rev() {
        let d = a[i * n + i];
        if d.abs() < 1e-12 {
            continue;
        }
        let s: f64 = (i + 1..n).map(|k| a[i * n + k] * x[k]).sum();
        x[i] = (b[i] - s) / d;
    }
    x
}

/// The strength (standard deviation) of the grain the synthesis makes with
/// auto-regressive coefficients `coeffs` before scaling, at 8 bits
/// (`grain_scale_shift` 0): its filter run over white noise of the
/// Gaussian sequence's strength, on a template of the synthesis's size.
fn synthesis_sigma(coeffs: &[i32]) -> f64 {
    let n = GAUSSIAN_SEQUENCE.len() as f64;
    let mean = GAUSSIAN_SEQUENCE.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = GAUSSIAN_SEQUENCE
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    // At 8 bits the sequence is shifted down by 12 - 8.
    let white = var.sqrt() / 16.0;
    let off = ar_offsets();
    let (h, w) = (73usize, 82usize);
    let mut g = vec![0f64; h * w];
    // A fixed pseudo-random Gaussian field (Box-Muller over an LCG).
    let mut seed = 0x2545_F491u32;
    let mut uni = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 8) as f64 + 0.5) / (1u32 << 24) as f64
    };
    let mut total = 0.0;
    let mut count = 0.0;
    for _ in 0..8 {
        for v in g.iter_mut() {
            let (u1, u2) = (uni(), uni());
            *v = white * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        for y in 3..h {
            for x in 3..w - 3 {
                let mut s = 0.0;
                for (k, &(dr, dc)) in off.iter().enumerate() {
                    s += g[(y as isize + dr) as usize * w + (x as isize + dc) as usize]
                        * coeffs[k] as f64;
                }
                g[y * w + x] += s / (1 << AR_SHIFT) as f64;
            }
        }
        // The part the synthesis takes its blocks from.
        for y in 9..h {
            for x in 9..w - 9 {
                total += g[y * w + x] * g[y * w + x];
                count += 1.0;
            }
        }
    }
    (total / count).sqrt()
}

/// What one plane's grain is: its auto-regressive coefficients and the
/// noise strength (8-bit units) at each intensity bin's centre (`None`:
/// too few flat samples there).
struct PlaneGrain {
    coeffs: Vec<i32>,
    sigma: [Option<f64>; BINS],
}

/// Denoises plane `p` of `frame` into `out` (the frame's own layout) and
/// measures the noise it took out; `None` when there is too little noise
/// to be worth synthesising (the plane is then copied as it is).
fn plane_grain(
    frame: &Frame,
    prev: Option<&Frame>,
    p: usize,
    out: &mut Frame,
) -> Option<PlaneGrain> {
    let plane = Plane::of(frame, p);
    let (w, h) = (plane.w, plane.h);
    let scale = (1u32 << (frame.bit_depth - 8)) as f64;
    let prev = prev.map(|f| Plane::of(f, p));
    let (blocks, resid) = blocks(&plane, prev.as_ref());
    if blocks.len() < 8 {
        return None;
    }
    // The noise variance by intensity: in each bin with blocks enough, the
    // flat ones' (sample units).
    let bin_of = |mean: f64| ((mean / scale) as usize * BINS / 256).min(BINS - 1);
    let mut by_bin: Vec<Vec<f64>> = vec![Vec::new(); BINS];
    for b in &blocks {
        by_bin[bin_of(b.mean)].push(b.var);
    }
    let mut var = [None; BINS];
    let mut all: Vec<f64> = blocks.iter().map(|b| b.var).collect();
    let overall = low_median(&mut all);
    // (A bin of few blocks, or of textured ones, held to twice the
    // frame's.)
    for (v, list) in var.iter_mut().zip(by_bin.iter_mut()) {
        if list.len() >= 8 {
            *v = Some(low_median(list).min(2.0 * overall));
        }
    }
    if overall.sqrt() / scale < MIN_SIGMA {
        return None;
    }
    let known: Vec<(usize, f64)> = var
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|v| (i, v)))
        .collect();
    let var_at = |mean: f64| {
        let i = bin_of(mean);
        known
            .iter()
            .min_by_key(|k| k.0.abs_diff(i))
            .map_or(overall, |k| k.1)
    };
    let den = denoise(&plane, var_at);
    let maxv = ((1u32 << frame.bit_depth) - 1) as f64;
    for y in 0..h {
        for x in 0..w {
            let v = den[y * w + x].round().clamp(0.0, maxv) as u16;
            out.set_sample(p, x as u32, y as u32, v);
        }
    }
    // The noise's correlation, from the flat blocks' residuals.
    let mut mask = vec![false; w * h];
    for b in &blocks {
        if b.var <= 1.5 * var_at(b.mean) {
            for y in b.y..b.y + NB {
                mask[y * w + b.x..y * w + b.x + NB].fill(true);
            }
        }
    }
    // (Only samples whose neighbours are in the same block are fitted.)
    let mut inner = mask.clone();
    for y in 0..h {
        for x in 0..w {
            let (bx, by) = (x % NB, y % NB);
            if by < LAG as usize || bx < LAG as usize || bx + LAG as usize >= NB {
                inner[y * w + x] = false;
            }
        }
    }
    let coeffs = fit_ar(&resid, w, h, &inner);
    let mut sigma = [None; BINS];
    for (s, v) in sigma.iter_mut().zip(&var) {
        *s = v.map(|v| v.sqrt() / scale);
    }
    Some(PlaneGrain { coeffs, sigma })
}

/// The scaling function's points for `g` (intensities in 8-bit units, the
/// scaling at each) when the synthesis's grain has strength `base` and
/// the scaling shift is `shift`; empty bins take their neighbours'.
fn points(g: &PlaneGrain, base: f64, shift: u32) -> Vec<(i32, i32)> {
    let known: Vec<(usize, f64)> = g
        .sigma
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|s| (i, s)))
        .collect();
    (0..BINS)
        .map(|i| {
            let s = known
                .iter()
                .min_by_key(|k| k.0.abs_diff(i))
                .map_or(0.0, |k| k.1);
            let centre = ((2 * i + 1) * 256 / (2 * BINS)) as i32;
            let scaling = (s * (1u32 << shift) as f64 / base)
                .round()
                .clamp(0.0, 255.0) as i32;
            (centre, scaling)
        })
        .collect()
}

/// The denoised frame to code, and the film grain parameters that put its
/// noise back (`apply_grain` 0, and the frame as it is, when it has too
/// little); `prev` is the frame shown before it (for the noise's
/// measure), `seed` the frame's `grain_seed`.
pub(crate) fn estimate(
    frame: &Frame,
    prev: Option<&Frame>,
    seed: u16,
    full_range: bool,
) -> (Frame, FilmGrainParams) {
    let mut out = frame.clone();
    let mono = frame.chroma == ChromaFormat::Mono;
    let luma = plane_grain(frame, prev, 0, &mut out);
    let Some(luma) = luma else {
        return (frame.clone(), FilmGrainParams::default());
    };
    let mut chroma: Vec<Option<PlaneGrain>> = if mono {
        Vec::new()
    } else {
        (1..3)
            .map(|p| plane_grain(frame, prev, p, &mut out))
            .collect()
    };
    // At 4:2:0 both chroma planes have grain or neither (6.8.20): one
    // without takes none, scaled to nothing.
    if frame.chroma == ChromaFormat::Yuv420 && chroma.iter().any(|c| c.is_some()) {
        for c in chroma.iter_mut() {
            c.get_or_insert_with(|| PlaneGrain {
                coeffs: vec![0; ar_offsets().len()],
                sigma: [Some(0.0); BINS],
            });
        }
    }
    let mut g = FilmGrainParams {
        apply_grain: true,
        grain_seed: seed as u32,
        update_grain: true,
        ar_coeff_lag: LAG,
        ar_coeff_shift_minus_6: AR_SHIFT - 6,
        grain_scale_shift: 0,
        overlap_flag: true,
        // Studio range: the synthesis keeps the samples in it.
        clip_to_restricted_range: !full_range,
        ..Default::default()
    };
    for (k, &c) in luma.coeffs.iter().enumerate() {
        g.ar_coeffs_y_plus_128[k] = c + 128;
    }
    // The scaling shift: the finest (8 to 11) with every scaling in range.
    let base_y = synthesis_sigma(&luma.coeffs);
    let mut planes: Vec<(&PlaneGrain, f64)> = vec![(&luma, base_y)];
    for c in chroma.iter().flatten() {
        planes.push((c, synthesis_sigma(&c.coeffs)));
    }
    let top = planes
        .iter()
        .flat_map(|(pg, base)| pg.sigma.iter().flatten().map(move |s| s / base))
        .fold(0f64, f64::max);
    let mut shift = 11;
    while shift > 8 && top * (1u32 << shift) as f64 > 255.0 {
        shift -= 1;
    }
    g.grain_scaling_minus_8 = shift - 8;
    let pts = points(&luma, base_y, shift);
    g.num_y_points = pts.len();
    for (i, &(v, s)) in pts.iter().enumerate() {
        g.point_y_value[i] = v;
        g.point_y_scaling[i] = s;
    }
    for (p, c) in chroma.iter().enumerate() {
        let Some(c) = c else { continue };
        let pts = points(c, synthesis_sigma(&c.coeffs), shift);
        // The chroma filters take the co-located luma grain too (the last
        // coefficient): not used, 0.
        let (n, values, scalings, coeffs) = if p == 0 {
            g.cb_mult = 128 + 64;
            g.cb_luma_mult = 128;
            g.cb_offset = 256;
            (
                &mut g.num_cb_points,
                &mut g.point_cb_value,
                &mut g.point_cb_scaling,
                &mut g.ar_coeffs_cb_plus_128,
            )
        } else {
            g.cr_mult = 128 + 64;
            g.cr_luma_mult = 128;
            g.cr_offset = 256;
            (
                &mut g.num_cr_points,
                &mut g.point_cr_value,
                &mut g.point_cr_scaling,
                &mut g.ar_coeffs_cr_plus_128,
            )
        };
        *n = pts.len();
        for (i, &(v, s)) in pts.iter().enumerate() {
            values[i] = v;
            scalings[i] = s;
        }
        for (k, &cf) in c.coeffs.iter().enumerate() {
            coeffs[k] = cf + 128;
        }
        coeffs[c.coeffs.len()] = 128;
    }
    (out, g)
}

/// Writes `film_grain_params()` (5.9.30) for a frame that is shown or
/// showable, with `update_grain` 1.
pub(crate) fn write_params(
    w: &mut crate::bits::BitWriter,
    g: &FilmGrainParams,
    inter: bool,
    mono: bool,
    subsampled_420: bool,
) {
    w.flag(g.apply_grain);
    if !g.apply_grain {
        return;
    }
    w.f(16, g.grain_seed);
    if inter {
        w.flag(true); // update_grain
    }
    w.f(4, g.num_y_points as u32);
    for i in 0..g.num_y_points {
        w.f(8, g.point_y_value[i] as u32);
        w.f(8, g.point_y_scaling[i] as u32);
    }
    if !mono {
        w.flag(g.chroma_scaling_from_luma);
    }
    let chroma_points =
        !(mono || g.chroma_scaling_from_luma || (subsampled_420 && g.num_y_points == 0));
    if chroma_points {
        w.f(4, g.num_cb_points as u32);
        for i in 0..g.num_cb_points {
            w.f(8, g.point_cb_value[i] as u32);
            w.f(8, g.point_cb_scaling[i] as u32);
        }
        w.f(4, g.num_cr_points as u32);
        for i in 0..g.num_cr_points {
            w.f(8, g.point_cr_value[i] as u32);
            w.f(8, g.point_cr_scaling[i] as u32);
        }
    }
    w.f(2, g.grain_scaling_minus_8);
    w.f(2, g.ar_coeff_lag as u32);
    let num_pos_luma = (2 * g.ar_coeff_lag * (g.ar_coeff_lag + 1)) as usize;
    let num_pos_chroma = if g.num_y_points > 0 {
        for i in 0..num_pos_luma {
            w.f(8, g.ar_coeffs_y_plus_128[i] as u32);
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
        for i in 0..num_pos_chroma {
            w.f(8, g.ar_coeffs_cb_plus_128[i] as u32);
        }
    }
    if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
        for i in 0..num_pos_chroma {
            w.f(8, g.ar_coeffs_cr_plus_128[i] as u32);
        }
    }
    w.f(2, g.ar_coeff_shift_minus_6);
    w.f(2, g.grain_scale_shift);
    if g.num_cb_points > 0 {
        w.f(8, g.cb_mult as u32);
        w.f(8, g.cb_luma_mult as u32);
        w.f(9, g.cb_offset as u32);
    }
    if g.num_cr_points > 0 {
        w.f(8, g.cr_mult as u32);
        w.f(8, g.cr_luma_mult as u32);
        w.f(9, g.cr_offset as u32);
    }
    w.flag(g.overlap_flag);
    w.flag(g.clip_to_restricted_range);
}
