//! Forward transforms for the encoder.
//!
//! The specification defines only the inverse transforms. Their 1D kernels
//! (DCT, ADST, identity) have orthogonal basis vectors, so a forward
//! transform is the adjoint of the inverse, normalised: the basis vectors
//! are measured once by running the normative inverse kernels on impulses,
//! and the 2D gain of the inverse (the rectangular scaling and the row and
//! column shifts of 7.13.3) is divided out. The result need not be exact:
//! the decoder reconstructs from the quantised levels with the normative
//! inverse, which the encoder's reconstruction uses too.

use std::sync::OnceLock;

use crate::consts::*;
use crate::dsp::itx::{inverse_1d, kinds};
use crate::tables::{TRANSFORM_ROW_SHIFT, TX_HEIGHT, TX_HEIGHT_LOG2, TX_WIDTH, TX_WIDTH_LOG2};

/// The forward kernel of one 1D inverse kernel: `f[i * n + k]` is the
/// weight of input sample `i` in coefficient `k` (the basis vector of `k`
/// divided by its squared length). The DCT's basis vectors are even or
/// odd about the middle, so it is also kept as two half-size matrices
/// over the sums and differences of mirrored inputs (`fe`, `fo`; made
/// exactly symmetric), and the identity as its diagonal.
struct Basis {
    n: usize,
    kind: u8,
    f: Vec<f32>,
    /// `fe[i * (n / 2) + k]`: input pair `i`, coefficient `2k`.
    fe: Vec<f32>,
    /// `fo[i * (n / 2) + k]`: input pair `i`, coefficient `2k + 1`.
    fo: Vec<f32>,
    diag: Vec<f32>,
    /// The largest weight's magnitude.
    max: f32,
}

fn basis(kind: u8, log2n: u32) -> &'static Basis {
    static CACHE: OnceLock<Vec<Basis>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        let mut v = Vec::new();
        for kind in 0..3u8 {
            for log2n in 0..7u32 {
                let n = 1usize << log2n;
                let valid = log2n >= 2 && (kind != 1 || log2n <= 4) && (kind != 2 || log2n <= 5);
                let mut f = vec![0.0f32; n * n];
                if valid {
                    const A: i32 = 1 << 16;
                    for k in 0..n {
                        let mut t = [0i32; 64];
                        t[k] = A;
                        inverse_1d(kind, log2n, &mut t[..n]);
                        let col: Vec<f64> = t[..n].iter().map(|&x| x as f64 / A as f64).collect();
                        let norm2: f64 = col.iter().map(|x| x * x).sum();
                        for i in 0..n {
                            f[i * n + k] = (col[i] / norm2) as f32;
                        }
                    }
                }
                let half = n / 2;
                let mut fe = vec![0.0f32; half * half];
                let mut fo = vec![0.0f32; half * half];
                if kind == 0 {
                    for i in 0..half {
                        for k in 0..half {
                            let m = n - 1 - i;
                            fe[i * half + k] = 0.5 * (f[i * n + 2 * k] + f[m * n + 2 * k]);
                            fo[i * half + k] = 0.5 * (f[i * n + 2 * k + 1] - f[m * n + 2 * k + 1]);
                        }
                    }
                }
                let diag = (0..n).map(|i| f[i * n + i]).collect();
                let max = f.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                v.push(Basis {
                    n,
                    kind,
                    f,
                    fe,
                    fo,
                    diag,
                    max,
                });
            }
        }
        v
    });
    &all[kind as usize * 7 + log2n as usize]
}

/// The 2D gain of the inverse transform at `tx_sz` (divided out).
fn gain(tx_sz: usize) -> f64 {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let rect = if log2w.abs_diff(log2h) == 1 {
        2896.0 / 4096.0
    } else {
        1.0
    };
    rect / (1u64 << (TRANSFORM_ROW_SHIFT[tx_sz] as u32 + 4)) as f64
}

/// A bound on the magnitude of every coefficient [`forward_2d`] gives for
/// a residual whose absolute values sum to 1.
pub(crate) fn coef_bound(tx_sz: usize, tx_type: usize) -> f64 {
    let (rk, ck) = kinds(tx_type);
    let br = basis(rk, TX_WIDTH_LOG2[tx_sz] as u32);
    let bc = basis(ck, TX_HEIGHT_LOG2[tx_sz] as u32);
    // Slack for the f32 arithmetic.
    1.001 * br.max as f64 * bc.max as f64 / gain(tx_sz)
}

/// The 1D forward transform of `lanes` lines at once: input `i` of line
/// `l` is `x[i * lanes + l]`; coefficient `k` (for `k < keep`) goes to
/// `out[k * lanes + l]`. The column transform as it was made before
/// [`lanes_n`], which the tests hold that to.
#[cfg(test)]
fn forward_lanes(b: &Basis, x: &[f32], lanes: usize, keep: usize, out: &mut [f32]) {
    let n = b.n;
    out[..keep * lanes].fill(0.0);
    match b.kind {
        2 => {
            for k in 0..keep {
                let d = b.diag[k];
                let o = &mut out[k * lanes..(k + 1) * lanes];
                for (a, &v) in o.iter_mut().zip(&x[k * lanes..(k + 1) * lanes]) {
                    *a = v * d;
                }
            }
        }
        0 => {
            let half = n / 2;
            let mut s = [0f32; 64];
            let mut d = [0f32; 64];
            for i in 0..half {
                let lo = &x[i * lanes..(i + 1) * lanes];
                let hi = &x[(n - 1 - i) * lanes..(n - i) * lanes];
                for l in 0..lanes {
                    s[l] = lo[l] + hi[l];
                    d[l] = lo[l] - hi[l];
                }
                let fe = &b.fe[i * half..(i + 1) * half];
                let fo = &b.fo[i * half..(i + 1) * half];
                for k in 0..keep.div_ceil(2) {
                    let w = fe[k];
                    let o = &mut out[2 * k * lanes..(2 * k + 1) * lanes];
                    for (a, &v) in o.iter_mut().zip(&s[..lanes]) {
                        *a += v * w;
                    }
                }
                for k in 0..keep / 2 {
                    let w = fo[k];
                    let o = &mut out[(2 * k + 1) * lanes..(2 * k + 2) * lanes];
                    for (a, &v) in o.iter_mut().zip(&d[..lanes]) {
                        *a += v * w;
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                let xi = &x[i * lanes..(i + 1) * lanes];
                for k in 0..keep {
                    let w = b.f[i * n + k];
                    let o = &mut out[k * lanes..(k + 1) * lanes];
                    for (a, &v) in o.iter_mut().zip(xi) {
                        *a += v * w;
                    }
                }
            }
        }
    }
}

/// The forward 2D transform of a `w` x `h` residual (row-major) for
/// `tx_type` at `tx_sz`, as the coefficients the inverse process expects
/// in `Dequant` (before quantisation), `out[k * w + j]`; only the top-left
/// 32x32, which the syntax can code, is computed. The flipped types flip
/// the residual first, as the reconstruction flips it back (7.13.3).
/// (The encoder quantises the coefficients as they come out:
/// [`forward_quantise`].)
#[cfg(test)]
pub(crate) fn forward_2d(residual: &[i32], tx_sz: usize, tx_type: usize, out: &mut [f64]) {
    SCRATCH.with_borrow_mut(|sc| {
        transform(residual, tx_sz, tx_type, sc);
        let (w, tw, th) = (
            TX_WIDTH[tx_sz],
            TX_WIDTH[tx_sz].min(32),
            TX_HEIGHT[tx_sz].min(32),
        );
        let inv = 1.0 / gain(tx_sz);
        for k in 0..th {
            for j in 0..tw {
                out[k * w + j] = sc.col[k * tw + j] as f64 * inv;
            }
        }
    })
}

/// The dead-zone quantiser of the encoder: a coefficient `c` (as
/// [`forward_2d`] gives it) becomes the level
/// `floor( |c| * r + rounding )`, at most `2^20 - 1`, with the sign of `c`;
/// `r` is the reciprocal of the step (`dqDenom / acQ`, or `dcQ` for the DC
/// coefficient, whose rounding is a half).
#[derive(Clone, Copy)]
pub(crate) struct Quantiser {
    pub(crate) ac_r: f64,
    pub(crate) dc_r: f64,
    /// The AC rounding: one less the dead zone.
    pub(crate) ac_round: f64,
}

/// [`forward_2d`] then the quantiser, the coefficients quantised as they
/// come out of the transform (the same values as quantising `forward_2d`'s
/// output: the same products in the same order) into `levels` (`Quant`'s
/// layout: `levels[k * tw + j]`, the top-left 32x32 at most), and the
/// coefficients themselves into `out` as `forward_2d` leaves them. Returns
/// the end of block: one past the last level that is not zero in the scan
/// whose positions' places `iscan` gives (`iscan[pos]` is the index of
/// `pos` in the scan).
pub(crate) fn forward_quantise(
    residual: &[i32],
    tx_sz: usize,
    tx_type: usize,
    q: Quantiser,
    iscan: &[u16],
    levels: &mut [i32],
    out: &mut [f64],
) -> usize {
    const MAX_LEVEL: f64 = ((1 << 20) - 1) as f64;
    SCRATCH.with_borrow_mut(|sc| {
        transform(residual, tx_sz, tx_type, sc);
        let (w, tw, th) = (
            TX_WIDTH[tx_sz],
            TX_WIDTH[tx_sz].min(32),
            TX_HEIGHT[tx_sz].min(32),
        );
        let inv = 1.0 / gain(tx_sz);
        let mut eob = 0;
        for k in 0..th {
            let col = &sc.col[k * tw..(k + 1) * tw];
            let lv = &mut levels[k * tw..(k + 1) * tw];
            let oc = &mut out[k * w..k * w + tw];
            let is = &iscan[k * tw..(k + 1) * tw];
            for (((o, c), &v), &s) in lv.iter_mut().zip(oc.iter_mut()).zip(col).zip(is) {
                let v = v as f64 * inv;
                *c = v;
                let l = (v.abs() * q.ac_r + q.ac_round).floor().min(MAX_LEVEL) as i32;
                *o = if v < 0.0 { -l } else { l };
                eob = eob.max(if l != 0 { s as usize + 1 } else { 0 });
            }
        }
        // The DC coefficient, the first of every scan, has its own step and
        // rounding: what the loop made of it is replaced.
        let v = out[0];
        let l = (v.abs() * q.dc_r + 0.5).floor().min(MAX_LEVEL) as i32;
        levels[0] = if v < 0.0 { -l } else { l };
        if eob <= 1 {
            // Nothing but the DC coefficient can be in the block.
            eob = (l != 0) as usize;
        }
        eob
    })
}

/// The 2D transform of the residual into `sc.col` (coefficient `k` of
/// column `j` at `col[k * tw + j]`, unscaled), the AVX2 build where there
/// is one.
fn transform(residual: &[i32], tx_sz: usize, tx_type: usize, sc: &mut Scratch) {
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        unsafe { transform_avx2(residual, tx_sz, tx_type, sc) };
        return;
    }
    forward_2d_body(residual, tx_sz, tx_type, sc)
}

/// [`transform`] compiled for AVX2: the same operations in the same
/// order (no fused multiply-adds), so the same results, eight lanes wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transform_avx2(residual: &[i32], tx_sz: usize, tx_type: usize, sc: &mut Scratch) {
    forward_2d_body(residual, tx_sz, tx_type, sc)
}

/// The 1D forward transform of one line `x` (`b.n` inputs) into its first
/// `keep` coefficients: the row transform as it was made one row at a
/// time, which [`forward_2d_body`]'s rows are held to.
#[cfg(test)]
fn forward_line(b: &Basis, x: &[f32], keep: usize, out: &mut [f32]) {
    let n = b.n;
    let out = &mut out[..keep];
    match b.kind {
        2 => {
            for (k, o) in out.iter_mut().enumerate() {
                *o = x[k] * b.diag[k];
            }
        }
        0 => {
            let half = n / 2;
            let (ke, ko) = (keep.div_ceil(2), keep / 2);
            let mut e = [0f32; 32];
            let mut o = [0f32; 32];
            for i in 0..half {
                let (s, d) = (x[i] + x[n - 1 - i], x[i] - x[n - 1 - i]);
                let fe = &b.fe[i * half..i * half + ke];
                for (a, &w) in e[..ke].iter_mut().zip(fe) {
                    *a += s * w;
                }
                let fo = &b.fo[i * half..i * half + ko];
                for (a, &w) in o[..ko].iter_mut().zip(fo) {
                    *a += d * w;
                }
            }
            for k in 0..ke {
                out[2 * k] = e[k];
            }
            for k in 0..ko {
                out[2 * k + 1] = o[k];
            }
        }
        _ => {
            out.fill(0.0);
            for (i, &v) in x[..n].iter().enumerate() {
                let f = &b.f[i * n..i * n + keep];
                for (a, &w) in out.iter_mut().zip(f) {
                    *a += v * w;
                }
            }
        }
    }
}

thread_local! {
    /// Scratch of the 2D transform: the residual by columns, its row
    /// transforms by columns, the row transforms by rows, the columns'.
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        xt: vec![0.0; 64 * 64],
        rt: vec![0.0; 32 * 64],
        t: vec![0.0; 64 * 32],
        col: vec![0.0; 32 * 32],
        sd: vec![0.0; 64 * 64],
    });
}

/// See [`SCRATCH`].
struct Scratch {
    xt: Vec<f32>,
    rt: Vec<f32>,
    t: Vec<f32>,
    col: Vec<f32>,
    /// The DCT's sums and differences of mirrored inputs.
    sd: Vec<f32>,
}

/// The 2D transform. The rows are transformed all at once, as the lanes
/// of [`lanes_n`] (the residual transposed, so that input `i` of every row
/// is a run), then the columns, as [`forward_lanes`] transforms them: each
/// coefficient is the same sum of the same products, in the same order,
/// as [`forward_line`] makes it one row at a time, so the results are
/// those of the row by row transform (a test holds it to that), but the
/// work is in vectors as wide as the block is high rather than as long as
/// a row's coefficients (two, for a 4x4 DCT's even and odd halves). Each
/// size is compiled on its own, its loops' counts known.
#[inline(always)]
fn forward_2d_body(residual: &[i32], tx_sz: usize, tx_type: usize, sc: &mut Scratch) {
    let f = match tx_sz {
        TX_4X4 => sized::<4, 4>,
        TX_8X8 => sized::<8, 8>,
        TX_16X16 => sized::<16, 16>,
        TX_32X32 => sized::<32, 32>,
        TX_64X64 => sized::<64, 64>,
        TX_4X8 => sized::<4, 8>,
        TX_8X4 => sized::<8, 4>,
        TX_8X16 => sized::<8, 16>,
        TX_16X8 => sized::<16, 8>,
        TX_16X32 => sized::<16, 32>,
        TX_32X16 => sized::<32, 16>,
        TX_32X64 => sized::<32, 64>,
        TX_64X32 => sized::<64, 32>,
        TX_4X16 => sized::<4, 16>,
        TX_16X4 => sized::<16, 4>,
        TX_8X32 => sized::<8, 32>,
        TX_32X8 => sized::<32, 8>,
        TX_16X64 => sized::<16, 64>,
        _ => sized::<64, 16>,
    };
    f(residual, tx_type, sc)
}

/// [`forward_2d_body`] for a `W` x `H` block.
#[inline(always)]
fn sized<const W: usize, const H: usize>(residual: &[i32], tx_type: usize, sc: &mut Scratch) {
    let tw = W.min(32);
    let th = H.min(32);
    let (flip_ud, flip_lr) = flips(tx_type);
    let (rk, ck) = kinds(tx_type);
    let br = basis(rk, W.trailing_zeros());
    let bc = basis(ck, H.trailing_zeros());
    debug_assert!(br.n == W && bc.n == H);
    let residual = &residual[..W * H];
    // The residual by columns: input `j` of row `i` at `xt[j * H + i]`.
    let xt = &mut sc.xt[..W * H];
    for i in 0..H {
        let si = if flip_ud { H - 1 - i } else { i };
        let row = &residual[si * W..(si + 1) * W];
        for j in 0..W {
            xt[j * H + i] = row[if flip_lr { W - 1 - j } else { j }] as f32;
        }
    }
    // Rows, the H rows as lanes: coefficient k of row i at rt[k * H + i].
    let rt = &mut sc.rt[..tw * H];
    lanes_n::<W, H>(br, xt, rt, &mut sc.sd);
    // By rows again: coefficient k of row i at t[i * tw + k].
    let t = &mut sc.t[..H * tw];
    for k in 0..tw {
        for i in 0..H {
            t[i * tw + k] = rt[k * H + i];
        }
    }
    // Columns, the tw columns as lanes: coefficient k of column j at
    // col[k * tw + j].
    let col = &mut sc.col[..th * tw];
    if W >= 32 {
        lanes_n::<H, 32>(bc, t, col, &mut sc.sd);
    } else {
        lanes_n::<H, W>(bc, t, col, &mut sc.sd);
    }
}

/// [`forward_lanes`] with `N` inputs (keeping the first 32 coefficients at
/// most) and `L` lanes known where it is compiled. Each coefficient is
/// summed in registers over the inputs, in their order, from zero: the
/// same operations in the same order as `forward_lanes`, whose loops
/// are the other way round.
#[inline(always)]
fn lanes_n<const N: usize, const L: usize>(b: &Basis, x: &[f32], out: &mut [f32], sd: &mut [f32]) {
    let keep = N.min(32);
    let x = &x[..N * L];
    let out = &mut out[..keep * L];
    match b.kind {
        2 => {
            let diag = &b.diag[..N];
            for k in 0..keep {
                let d = diag[k];
                for l in 0..L {
                    out[k * L + l] = x[k * L + l] * d;
                }
            }
        }
        0 => {
            let half = N / 2;
            let (fe, fo) = (&b.fe[..half * half], &b.fo[..half * half]);
            // The sums and differences of the mirrored inputs.
            let (s, d) = sd[..N * L].split_at_mut(half * L);
            for i in 0..half {
                for l in 0..L {
                    let (lo, hi) = (x[i * L + l], x[(N - 1 - i) * L + l]);
                    s[i * L + l] = lo + hi;
                    d[i * L + l] = lo - hi;
                }
            }
            for k in 0..keep.div_ceil(2) {
                let mut acc = [0f32; L];
                for i in 0..half {
                    let w = fe[i * half + k];
                    for l in 0..L {
                        acc[l] += s[i * L + l] * w;
                    }
                }
                out[2 * k * L..(2 * k + 1) * L].copy_from_slice(&acc);
            }
            for k in 0..keep / 2 {
                let mut acc = [0f32; L];
                for i in 0..half {
                    let w = fo[i * half + k];
                    for l in 0..L {
                        acc[l] += d[i * L + l] * w;
                    }
                }
                out[(2 * k + 1) * L..(2 * k + 2) * L].copy_from_slice(&acc);
            }
        }
        _ => {
            let f = &b.f[..N * N];
            for k in 0..keep {
                let mut acc = [0f32; L];
                for i in 0..N {
                    let w = f[i * N + k];
                    for l in 0..L {
                        acc[l] += x[i * L + l] * w;
                    }
                }
                out[k * L..(k + 1) * L].copy_from_slice(&acc);
            }
        }
    }
}

/// Whether `tx_type` flips the residual up and down, left and right.
fn flips(tx_type: usize) -> (bool, bool) {
    (
        matches!(
            tx_type,
            FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
        ),
        matches!(
            tx_type,
            DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform one row at a time (the rows by [`forward_line`]),
    /// as it was before the rows were made lanes.
    fn forward_2d_by_rows(residual: &[i32], tx_sz: usize, tx_type: usize, out: &mut [f64]) {
        let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
        let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
        let (w, h) = (1usize << log2w, 1usize << log2h);
        let (tw, th) = (w.min(32), h.min(32));
        let (flip_ud, flip_lr) = flips(tx_type);
        let (rk, ck) = kinds(tx_type);
        let (br, bc) = (basis(rk, log2w), basis(ck, log2h));
        let mut t = vec![0f32; h * tw];
        let mut col = vec![0f32; th * tw];
        let mut x = [0f32; 64];
        for i in 0..h {
            let si = if flip_ud { h - 1 - i } else { i };
            let row = &residual[si * w..(si + 1) * w];
            for (j, v) in x[..w].iter_mut().enumerate() {
                *v = row[if flip_lr { w - 1 - j } else { j }] as f32;
            }
            forward_line(br, &x[..w], tw, &mut t[i * tw..(i + 1) * tw]);
        }
        forward_lanes(bc, &t, tw, th, &mut col);
        let inv = 1.0 / gain(tx_sz);
        for k in 0..th {
            for j in 0..tw {
                out[k * w + j] = col[k * tw + j] as f64 * inv;
            }
        }
    }

    /// The transform (the AVX2 build of it where there is one) gives
    /// exactly the coefficients of the row by row transform, for every
    /// size and type, on residuals of every range.
    #[test]
    fn matches_the_row_by_row_transform() {
        let mut seed = 1u32;
        for tx_sz in 0..TX_SIZES_ALL {
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            for tx_type in 0..16 {
                let (rk, ck) = kinds(tx_type);
                let ok = |k: u8, n: usize| match k {
                    0 => true,
                    1 => n <= 16,
                    _ => n <= 32,
                };
                if !ok(rk, w) || !ok(ck, h) {
                    continue;
                }
                for range in [1i32, 20, 255, 1023, 4095] {
                    let res: Vec<i32> = (0..w * h)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                            ((seed >> 8) % (2 * range as u32 + 1)) as i32 - range
                        })
                        .collect();
                    let mut a = vec![0.0; w * h];
                    let mut b = vec![0.0; w * h];
                    forward_2d(&res, tx_sz, tx_type, &mut a);
                    forward_2d_by_rows(&res, tx_sz, tx_type, &mut b);
                    for (x, y) in a.iter().zip(&b) {
                        assert_eq!(x.to_bits(), y.to_bits(), "tx {tx_sz} type {tx_type}");
                    }
                }
            }
        }
    }
    /// The quantiser applied as the coefficients come out of the
    /// transform gives the levels and the end of block that quantising
    /// [`forward_2d`]'s coefficients gives.
    #[test]
    fn forward_quantise_matches_quantising_the_transform() {
        use crate::decoder::residual::{inverse_scan, scan_for};
        const MAX_LEVEL: f64 = ((1 << 20) - 1) as f64;
        let mut seed = 5u32;
        for tx_sz in 0..TX_SIZES_ALL {
            let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
            let (tw, th) = (w.min(32), h.min(32));
            for tx_type in [DCT_DCT, ADST_DCT, IDTX, V_DCT, H_DCT, FLIPADST_FLIPADST] {
                let (rk, ck) = kinds(tx_type);
                let ok = |k: u8, n: usize| k == 0 || (k == 1 && n <= 16) || (k == 2 && n <= 32);
                // (The one-dimensional types have scans up to 16x16.)
                let one_d = matches!(tx_type, V_DCT | H_DCT);
                if !ok(rk, w) || !ok(ck, h) || one_d && w.max(h) > 16 {
                    continue;
                }
                for (range, ac_r, dc_r, dz) in [
                    (3, 0.05, 0.04, 0.62),
                    (60, 0.01, 0.02, 0.70),
                    (255, 0.003, 0.1, 0.62),
                ] {
                    let res: Vec<i32> = (0..w * h)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                            ((seed >> 8) % (2 * range as u32 + 1)) as i32 - range
                        })
                        .collect();
                    let mut c = vec![0.0; w * h];
                    forward_2d(&res, tx_sz, tx_type, &mut c);
                    let mut want = vec![0i32; tw * th];
                    for i in 0..th {
                        for j in 0..tw {
                            let v = c[i * w + j];
                            let r = if i + j == 0 { dc_r } else { ac_r };
                            let round = if i + j == 0 { 0.5 } else { 1.0 - dz };
                            let l = (v.abs() * r + round).floor().min(MAX_LEVEL) as i32;
                            want[i * tw + j] = if v < 0.0 { -l } else { l };
                        }
                    }
                    let want_eob = scan_for(tx_sz, tx_type)
                        .iter()
                        .rposition(|&p| want[p as usize] != 0)
                        .map_or(0, |k| k + 1);
                    let mut got = vec![0i32; tw * th];
                    let mut coefs = vec![0.0; w * h];
                    let eob = forward_quantise(
                        &res,
                        tx_sz,
                        tx_type,
                        Quantiser {
                            ac_r,
                            dc_r,
                            ac_round: 1.0 - dz,
                        },
                        inverse_scan(tx_sz, tx_type),
                        &mut got,
                        &mut coefs,
                    );
                    for i in 0..th {
                        assert_eq!(coefs[i * w..i * w + tw], c[i * w..i * w + tw]);
                    }
                    assert_eq!(got, want, "tx {tx_sz} type {tx_type}");
                    assert_eq!(eob, want_eob, "tx {tx_sz} type {tx_type}");
                }
            }
        }
    }

    use crate::dsp::itx::inverse_transform_2d;

    /// The 64-point sizes: a smooth residual (no energy above the 32
    /// lowest frequencies the syntax keeps) survives the round trip.
    #[test]
    fn round_trip_64() {
        for tx_sz in [TX_64X64, TX_32X64, TX_64X32, TX_16X64, TX_64X16] {
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            let res: Vec<i32> = (0..w * h)
                .map(|k| {
                    let (i, j) = ((k / w) as f64, (k % w) as f64);
                    (40.0 * (i / h as f64 * 3.0).sin() + 30.0 * (j / w as f64 * 2.0).cos()) as i32
                })
                .collect();
            let mut c = vec![0.0; w * h];
            forward_2d(&res, tx_sz, DCT_DCT, &mut c);
            let mut dq = vec![0i32; 64 * 64];
            for i in 0..h.min(32) {
                for j in 0..w.min(32) {
                    dq[i * 64 + j] = c[i * w + j].round() as i32;
                }
            }
            let mut out = vec![0i32; w * h];
            inverse_transform_2d(&dq, tx_sz, DCT_DCT, false, 8, &mut out);
            let err: i64 = out
                .iter()
                .zip(&res)
                .map(|(a, b)| ((a - b) as i64).abs())
                .max()
                .unwrap();
            assert!(err <= 3, "tx {tx_sz}: max error {err}");
        }
    }

    /// Forward then the normative inverse returns the residual, near enough.
    #[test]
    fn round_trip() {
        for tx_sz in [
            TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_8X16, TX_16X8, TX_4X8, TX_8X4,
        ] {
            for tx_type in [
                DCT_DCT,
                ADST_ADST,
                ADST_DCT,
                DCT_ADST,
                IDTX,
                FLIPADST_DCT,
                DCT_FLIPADST,
                FLIPADST_FLIPADST,
                V_DCT,
                H_FLIPADST,
            ] {
                if tx_sz == TX_32X32 && tx_type != DCT_DCT && tx_type != IDTX {
                    continue;
                }
                let flip_ud = matches!(
                    tx_type,
                    FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
                );
                let flip_lr = matches!(
                    tx_type,
                    DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
                );
                let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
                let res: Vec<i32> = (0..w * h)
                    .map(|i| ((i * 37 + i / w * 11) % 61) as i32 - 30)
                    .collect();
                let mut c = vec![0.0; w * h];
                forward_2d(&res, tx_sz, tx_type, &mut c);
                let mut dq = vec![0i32; 64 * 64];
                for i in 0..h.min(32) {
                    for j in 0..w.min(32) {
                        dq[i * 64 + j] = c[i * w + j].round() as i32;
                    }
                }
                let mut out = vec![0i32; w * h];
                inverse_transform_2d(&dq, tx_sz, tx_type, false, 8, &mut out);
                // The reconstruction's flips.
                let err: i64 = (0..w * h)
                    .map(|k| {
                        let (i, j) = (k / w, k % w);
                        let si = if flip_ud { h - 1 - i } else { i };
                        let sj = if flip_lr { w - 1 - j } else { j };
                        ((out[si * w + sj] - res[k]) as i64).abs()
                    })
                    .max()
                    .unwrap();
                assert!(err <= 2, "tx {tx_sz} type {tx_type}: max error {err}");
            }
        }
    }
}
