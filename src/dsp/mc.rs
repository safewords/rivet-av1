//! The 8-tap interpolation filters of inter prediction (7.11.3.4) and the
//! blends of compound prediction (7.11.3.1): scalar, and AVX2 versions of
//! the 32-bit filters chosen at run time. The 16-bit filters and the blends
//! have SIMD128 versions in a wasm32 module built with `simd128` and NEON
//! versions on aarch64. All bit-exact.

/// The horizontal filter: `out[c] = Round2(sum_t f[t] * src[c + t],
/// round0)`, `src` holding `out.len() + 7` samples.
#[inline]
pub(crate) fn filter_row(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    debug_assert!(src.len() >= out.len() + 7);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() && out.len() >= 8 {
        // SAFETY: AVX2 is available; lengths checked above.
        unsafe { filter_row_avx2(src, f, round0, out) };
        return;
    }
    filter_row_scalar(src, f, round0, out)
}

/// The vertical filter of output row `r` over the intermediate array
/// (`w` wide): `out[c] = Round2(sum_t f[t] * inter[r + t][c], round1)`.
#[inline]
pub(crate) fn filter_col(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    debug_assert!(inter.len() >= (r + 8) * w && out.len() == w);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() && w >= 8 {
        // SAFETY: AVX2 is available; lengths checked above.
        unsafe { filter_col_avx2(inter, r, w, f, round1, out) };
        return;
    }
    filter_col_scalar(inter, r, w, f, round1, out)
}

pub(crate) fn filter_row_scalar(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    let w = out.len();
    let add = (1 << round0) >> 1;
    if f[3] == 128 {
        for c in 0..w {
            out[c] = (128 * src[c + 3] as i32 + add) >> round0;
        }
        return;
    }
    let mut acc = [0i32; 128];
    let acc = &mut acc[..w];
    for t in 0..8 {
        let ft = f[t];
        let s = &src[t..t + w];
        for c in 0..w {
            acc[c] += ft * s[c] as i32;
        }
    }
    for c in 0..w {
        out[c] = (acc[c] + add) >> round0;
    }
}

pub(crate) fn filter_col_scalar(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    let add = (1 << round1) >> 1;
    if f[3] == 128 {
        let s = &inter[(r + 3) * w..(r + 4) * w];
        for c in 0..w {
            out[c] = (128 * s[c] + add) >> round1;
        }
        return;
    }
    let mut acc = [0i32; 128];
    let acc = &mut acc[..w];
    for t in 0..8 {
        let ft = f[t];
        let s = &inter[(r + t) * w..(r + t + 1) * w];
        for c in 0..w {
            acc[c] += ft * s[c];
        }
    }
    for c in 0..w {
        out[c] = (acc[c] + add) >> round1;
    }
}

/// The taps of a filter that are worth multiplying by: the shortest run of
/// 2, 4, 6 or 8 that holds every tap that is not zero. (Of
/// `Subpel_Filters`, the regular and smooth sets fit six, the sets for
/// blocks 4 wide four, the bilinear one two.)
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Taps {
    /// Which tap of the eight the run starts at.
    pub(crate) first: usize,
    /// How many taps it has.
    pub(crate) len: usize,
    taps: [i16; 8],
}

#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
impl Taps {
    pub(crate) fn new(f: &[i32; 8]) -> Self {
        let lo = f.iter().position(|&t| t != 0).unwrap_or(0);
        let hi = f.iter().rposition(|&t| t != 0).unwrap_or(0);
        let len = (hi - lo + 2) & !1;
        let first = lo.min(8 - len);
        let mut taps = [0i16; 8];
        for (t, v) in taps[..len].iter_mut().enumerate() {
            *v = f[first + t] as i16;
        }
        Taps { first, len, taps }
    }

    /// The run itself.
    #[inline]
    pub(crate) fn taps(&self) -> &[i16] {
        &self.taps[..self.len]
    }
}

/// The horizontal filter of a block's rows into 16-bit intermediates:
/// `out[r][c] = Round2(sum_t taps[t] * src[r][c + t], round0) - bias`. Row
/// `r` of `src` starts at `r * stride` and holds `w + taps.len - 1`
/// samples, the first of them under the first tap kept; row `r` of `out`
/// starts at `r * w`.
///
/// The intermediate of the prediction process fits 16 bits at every bit
/// depth (a sample of at most `BitDepth` bits times taps whose positive
/// part sums to at most 184, shifted down by `round0`: under 2^15; see the
/// test), which halves what the vertical pass reads and lets both passes
/// multiply 16-bit lanes. `bias` is zero for an intermediate; a compound
/// prediction with no vertical phase is this filter's output as it stands,
/// and takes its bias here (see [`Compound`]).
///
/// `eight_bit` says that no sample is over 255, as in a frame of 8 bits.
/// The sum itself then fits 16 bits (see [`taps8_scalar`]), and the SIMD
/// versions never leave 16-bit lanes.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_rows16(
    src: &[u16],
    stride: usize,
    w: usize,
    rows: usize,
    taps: &Taps,
    round0: u32,
    bias: i32,
    eight_bit: bool,
    out: &mut [i16],
) {
    let t = taps.taps();
    if eight_bit && bias == 0 {
        return taps8(src, stride, 1, w, rows, t, round0, out);
    }
    let add = ((1 << round0) >> 1) - (bias << round0);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 4 {
        match t.len() {
            2 => rows16_simd128::<1>(src, stride, w, rows, t, add, round0, out),
            4 => rows16_simd128::<2>(src, stride, w, rows, t, add, round0, out),
            6 => rows16_simd128::<3>(src, stride, w, rows, t, add, round0, out),
            _ => rows16_simd128::<4>(src, stride, w, rows, t, add, round0, out),
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 4 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            match t.len() {
                2 => rows16_neon::<2>(src, stride, w, rows, t, add, round0, out),
                4 => rows16_neon::<4>(src, stride, w, rows, t, add, round0, out),
                6 => rows16_neon::<6>(src, stride, w, rows, t, add, round0, out),
                _ => rows16_neon::<8>(src, stride, w, rows, t, add, round0, out),
            }
        }
        return;
    }
    match t.len() {
        2 => rows16_scalar::<2>(src, stride, w, rows, t, add, round0, out),
        4 => rows16_scalar::<4>(src, stride, w, rows, t, add, round0, out),
        6 => rows16_scalar::<6>(src, stride, w, rows, t, add, round0, out),
        _ => rows16_scalar::<8>(src, stride, w, rows, t, add, round0, out),
    }
}

/// A filter over samples of at most 8 bits, its taps `step` samples apart
/// (1 for the horizontal filter, the stride for a vertical one): `out[r][c]
/// = Round2(sum_t taps[t] * src[r * stride + c + t * step], shift)`, row `r`
/// of `out` at `r * w`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn taps8(
    src: &[u16],
    stride: usize,
    step: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    shift: u32,
    out: &mut [i16],
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 4 {
        match taps.len() {
            2 => taps8_simd128::<2>(src, stride, step, w, rows, taps, shift, out),
            4 => taps8_simd128::<4>(src, stride, step, w, rows, taps, shift, out),
            6 => taps8_simd128::<6>(src, stride, step, w, rows, taps, shift, out),
            _ => taps8_simd128::<8>(src, stride, step, w, rows, taps, shift, out),
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 4 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            match taps.len() {
                2 => taps8_neon::<2>(src, stride, step, w, rows, taps, shift, out),
                4 => taps8_neon::<4>(src, stride, step, w, rows, taps, shift, out),
                6 => taps8_neon::<6>(src, stride, step, w, rows, taps, shift, out),
                _ => taps8_neon::<8>(src, stride, step, w, rows, taps, shift, out),
            }
        }
        return;
    }
    match taps.len() {
        2 => taps8_scalar::<2>(src, stride, step, w, rows, taps, shift, out),
        4 => taps8_scalar::<4>(src, stride, step, w, rows, taps, shift, out),
        6 => taps8_scalar::<6>(src, stride, step, w, rows, taps, shift, out),
        _ => taps8_scalar::<8>(src, stride, step, w, rows, taps, shift, out),
    }
}

/// What the 8-bit filters add to a sum to keep it in an unsigned 16-bit
/// lane.
///
/// The taps of a filter sum to 128, the positive ones to at most 184 and
/// the negative ones to at least -56 (the test goes through the table), so
/// over samples of 0..=255 the sum lies in -14280..=46920. With 16384 and
/// the rounding constant of a shift of up to 7 added it lies in 0..65536:
/// 16-bit lanes that wrap hold it exactly, a logical shift is the
/// arithmetic one, and `16384 >> shift` taken off afterwards leaves
/// `Round2(sum, shift)`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
const LIFT8: i32 = 16384;

/// The scalar version of [`taps8`], in the 16-bit arithmetic the SIMD
/// versions use.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taps8_scalar<const N: usize>(
    src: &[u16],
    stride: usize,
    step: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    shift: u32,
    out: &mut [i16],
) {
    let taps: [u16; N] = std::array::from_fn(|t| taps[t] as u16);
    let add = (LIFT8 + ((1 << shift) >> 1)) as u16;
    let sub = (LIFT8 >> shift) as i16;
    for r in 0..rows {
        let s = &src[r * stride..][..(N - 1) * step + w];
        for (c, o) in out[r * w..][..w].iter_mut().enumerate() {
            let mut acc = add;
            for t in 0..N {
                acc = acc.wrapping_add(taps[t].wrapping_mul(s[c + t * step]));
            }
            *o = (acc >> shift) as i16 - sub;
        }
    }
}

/// The scalar version of [`filter_rows16`] for samples of any depth: `out =
/// (sum + add) >> round0`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rows16_scalar<const N: usize>(
    src: &[u16],
    stride: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    add: i32,
    round0: u32,
    out: &mut [i16],
) {
    let taps: [i32; N] = std::array::from_fn(|t| taps[t] as i32);
    for r in 0..rows {
        let s = &src[r * stride..][..w + N - 1];
        for (c, o) in out[r * w..][..w].iter_mut().enumerate() {
            let mut acc = add;
            for t in 0..N {
                acc += taps[t] * s[c + t] as i32;
            }
            *o = (acc >> round0) as i16;
        }
    }
}

/// The vertical filter over eight rows of 16-bit intermediates, `rows`
/// holding them one after another (`w` wide): `out[c] = Round2(sum_t f[t] *
/// rows[t][c], round1)`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn filter_col16(rows: &[i16], w: usize, f: &[i32; 8], round1: u32, out: &mut [i32]) {
    debug_assert!(rows.len() >= 8 * w && out.len() == w);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 4 {
        filter_col16_simd128(rows, w, f, round1, out);
        return;
    }
    filter_col16_scalar(rows, w, f, round1, out)
}

/// The vertical filter of a compound prediction, which stays in 16 bits:
/// `out[c] = Round2(sum_t f[t] * rows[t][c], round1) - bias` (see
/// [`Compound`] for why that fits).
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn filter_col16_compound(
    rows: &[i16],
    w: usize,
    f: &[i32; 8],
    round1: u32,
    bias: i32,
    out: &mut [i16],
) {
    debug_assert!(rows.len() >= 8 * w && out.len() == w);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 4 {
        filter_col16_compound_simd128(rows, w, f, round1, bias, out);
        return;
    }
    filter_col16_compound_scalar(rows, w, f, round1, bias, out)
}

pub(crate) fn filter_col16_scalar(
    rows: &[i16],
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    let add = (1 << round1) >> 1;
    let rows = &rows[..8 * w];
    for (c, o) in out.iter_mut().enumerate() {
        let mut acc = add;
        for t in 0..8 {
            acc += f[t] * rows[t * w + c] as i32;
        }
        *o = acc >> round1;
    }
}

pub(crate) fn filter_col16_compound_scalar(
    rows: &[i16],
    w: usize,
    f: &[i32; 8],
    round1: u32,
    bias: i32,
    out: &mut [i16],
) {
    let add = ((1 << round1) >> 1) - (bias << round1);
    let rows = &rows[..8 * w];
    for (c, o) in out.iter_mut().enumerate() {
        let mut acc = add;
        for t in 0..8 {
            acc += f[t] * rows[t * w + c] as i32;
        }
        *o = (acc >> round1) as i16;
    }
}

/// How the two predictions of a compound block are kept in 16 bits, and
/// what blending them needs.
///
/// A compound prediction is `Round2(.., InterRound1)` with `InterRound1 =
/// 7`: the sample scaled by `1 << InterPostRound` (16 for 8 and 10 bits, 4
/// for 12), with the filters' overshoot on both sides. At 8 bits that is
/// within -2800..9300. At 10 and 12 bits it reaches -20600..37000, which
/// 16 bits do not hold, so `bias` (8192) is taken off as the value is
/// stored: -28800..28800 (the tests compute the bounds). A blend's weights
/// sum to a constant, so the bias comes back as one term in its rounding
/// constant.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Compound {
    /// What every stored prediction is short of its value.
    pub(crate) bias: i32,
    /// `InterPostRound`.
    pub(crate) post: u32,
    /// The largest sample, `(1 << BitDepth) - 1`.
    pub(crate) maxv: i32,
}

#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
impl Compound {
    /// For a frame of `bit_depth` bits and a compound block's
    /// `InterPostRound`.
    pub(crate) fn new(bit_depth: u32, post: u32) -> Self {
        Compound {
            bias: if bit_depth == 8 { 0 } else { 8192 },
            post,
            maxv: (1 << bit_depth) - 1,
        }
    }
}

/// The average of two predictions that are both whole samples of their
/// references: `(a + b + 1) >> 1`. (Each is the sample times `1 <<
/// InterPostRound`, and `Round2(a + b, 1 + InterPostRound)` of those is
/// this; it needs no clip.)
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn avg_samples(a: &[u16], b: &[u16], dst: &mut [u16]) {
    debug_assert!(a.len() >= dst.len() && b.len() >= dst.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 8 {
        avg_samples_simd128(a, b, dst);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if dst.len() >= 8 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { avg_samples_neon(a, b, dst) };
        return;
    }
    avg_samples_scalar(a, b, dst)
}

/// The average of a compound block's two predictions (7.11.3.1), as
/// samples: `Clip1(Round2(a + b, 1 + InterPostRound))`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn compound_avg(a: &[i16], b: &[i16], c: Compound, dst: &mut [u16]) {
    debug_assert!(a.len() >= dst.len() && b.len() >= dst.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 8 {
        compound_avg_simd128(a, b, c, dst);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if dst.len() >= 8 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { compound_avg_neon(a, b, c, dst) };
        return;
    }
    compound_avg_scalar(a, b, c, dst)
}

pub(crate) fn avg_samples_scalar(a: &[u16], b: &[u16], dst: &mut [u16]) {
    for ((o, &a), &b) in dst.iter_mut().zip(a).zip(b) {
        *o = ((a as u32 + b as u32 + 1) >> 1) as u16;
    }
}

pub(crate) fn compound_avg_scalar(a: &[i16], b: &[i16], c: Compound, dst: &mut [u16]) {
    let shift = 1 + c.post;
    let add = (1 << c.post) + 2 * c.bias;
    for ((o, &a), &b) in dst.iter_mut().zip(a).zip(b) {
        *o = ((a as i32 + b as i32 + add) >> shift).clamp(0, c.maxv) as u16;
    }
}

/// The taps as four pairs of 16-bit lanes, for `i32x4_dot_i16x8`, and
/// which pairs are not both zero.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn tap_pairs(f: &[i32; 8]) -> ([std::arch::wasm32::v128; 4], [bool; 4]) {
    use std::arch::wasm32::*;
    let pair = |k: usize| {
        let (a, b) = (f[2 * k] as i16, f[2 * k + 1] as i16);
        i16x8(a, b, a, b, a, b, a, b)
    };
    (
        [pair(0), pair(1), pair(2), pair(3)],
        std::array::from_fn(|k| f[2 * k] != 0 || f[2 * k + 1] != 0),
    )
}

/// [`taps8`] in 16-bit lanes that wrap, eight outputs at a time (four for a
/// block under eight wide): a multiply and an add per tap.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
fn taps8_simd128<const N: usize>(
    src: &[u16],
    stride: usize,
    step: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    shift: u32,
    out: &mut [i16],
) {
    use std::arch::wasm32::*;
    assert!(taps.len() == N && w >= 4 && rows > 0);
    assert!(src.len() >= (rows - 1) * stride + (N - 1) * step + w && out.len() >= rows * w);
    let coef: [v128; N] = std::array::from_fn(|t| i16x8_splat(taps[t]));
    let add = i16x8_splat((LIFT8 + ((1 << shift) >> 1)) as i16);
    let sub = i16x8_splat((LIFT8 >> shift) as i16);
    for r in 0..rows {
        // SAFETY: row r of each, which the assertions put inside.
        let (sp, op) = unsafe { (src.as_ptr().add(r * stride), out.as_mut_ptr().add(r * w)) };
        // Whole vectors, then one more ending at the last column
        // (overlapping what is done: the same values are stored again).
        let mut c = 0;
        if w >= 8 {
            while c < w {
                let c0 = c.min(w - 8);
                let mut acc = add;
                for t in 0..N {
                    // SAFETY: eight samples under tap t, the last of them
                    // at (rows - 1) * stride + (N - 1) * step + w - 1 or
                    // before.
                    let s = unsafe { v128_load(sp.add(t * step + c0) as *const v128) };
                    acc = i16x8_add(acc, i16x8_mul(s, coef[t]));
                }
                let v = i16x8_sub(u16x8_shr(acc, shift), sub);
                // SAFETY: c0 + 8 <= w.
                unsafe { v128_store(op.add(c0) as *mut v128, v) };
                c = c0 + 8;
            }
        } else {
            while c < w {
                let c0 = c.min(w - 4);
                let mut acc = add;
                for t in 0..N {
                    // SAFETY: four samples under tap t, as above.
                    let s = unsafe { v128_load64_zero(sp.add(t * step + c0) as *const u64) };
                    acc = i16x8_add(acc, i16x8_mul(s, coef[t]));
                }
                let v = i16x8_sub(u16x8_shr(acc, shift), sub);
                // SAFETY: c0 + 4 <= w.
                unsafe { v128_store64_lane::<0>(v, op.add(c0) as *mut u64) };
                c = c0 + 4;
            }
        }
    }
}

/// The horizontal filter over samples of any depth, `(sum + add) >>
/// round0`: taps two by two (`P` pairs), each pair one 16-bit dot product.
/// A 32-bit lane of the source is two neighbouring samples, so the product
/// of the row from an even column gives the even outputs' sums and the row
/// from the next column the odd ones'; the two are put together at the end.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
fn rows16_simd128<const P: usize>(
    src: &[u16],
    stride: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    add: i32,
    round0: u32,
    out: &mut [i16],
) {
    use std::arch::wasm32::*;
    assert!(taps.len() == 2 * P && w >= 4 && rows > 0);
    assert!(src.len() >= (rows - 1) * stride + 2 * P - 1 + w && out.len() >= rows * w);
    let coef: [v128; P] = std::array::from_fn(|k| {
        let (a, b) = (taps[2 * k], taps[2 * k + 1]);
        i16x8(a, b, a, b, a, b, a, b)
    });
    let add = i32x4_splat(add);
    let low = i32x4_splat(0xffff);
    for r in 0..rows {
        // SAFETY: row r of each, which the assertions put inside.
        let (sp, op) = unsafe { (src.as_ptr().add(r * stride), out.as_mut_ptr().add(r * w)) };
        let mut c = 0;
        if w >= 8 {
            while c < w {
                let c0 = c.min(w - 8);
                let (mut even, mut odd) = (add, add);
                for k in 0..P {
                    // SAFETY: eight samples from c0 + 2k and from the
                    // column after; c0 + 2k + 1 + 8 <= w + 2P - 1.
                    let (a, b) = unsafe {
                        (
                            v128_load(sp.add(c0 + 2 * k) as *const v128),
                            v128_load(sp.add(c0 + 2 * k + 1) as *const v128),
                        )
                    };
                    even = i32x4_add(even, i32x4_dot_i16x8(a, coef[k]));
                    odd = i32x4_add(odd, i32x4_dot_i16x8(b, coef[k]));
                }
                // Each result fits 16 bits: the even ones in the low
                // halves of their lanes, the odd ones in the high halves.
                let even = v128_and(i32x4_shr(even, round0), low);
                let v = v128_or(even, i32x4_shl(i32x4_shr(odd, round0), 16));
                // SAFETY: c0 + 8 <= w.
                unsafe { v128_store(op.add(c0) as *mut v128, v) };
                c = c0 + 8;
            }
        } else {
            while c < w {
                let c0 = c.min(w - 4);
                let (mut even, mut odd) = (add, add);
                for k in 0..P {
                    // SAFETY: four samples from c0 + 2k and from the
                    // column after; c0 + 2k + 1 + 4 <= w + 2P - 1.
                    let (a, b) = unsafe {
                        (
                            v128_load64_zero(sp.add(c0 + 2 * k) as *const u64),
                            v128_load64_zero(sp.add(c0 + 2 * k + 1) as *const u64),
                        )
                    };
                    even = i32x4_add(even, i32x4_dot_i16x8(a, coef[k]));
                    odd = i32x4_add(odd, i32x4_dot_i16x8(b, coef[k]));
                }
                let even = v128_and(i32x4_shr(even, round0), low);
                let v = v128_or(even, i32x4_shl(i32x4_shr(odd, round0), 16));
                // SAFETY: c0 + 4 <= w.
                unsafe { v128_store64_lane::<0>(v, op.add(c0) as *mut u64) };
                c = c0 + 4;
            }
        }
    }
}

/// The vertical filter's sums, `(acc + add) >> round1` in 32-bit lanes,
/// eight (or, for a block four wide, four) columns at a time: rows two by
/// two, interleaved as in the horizontal filter. `store(c0, lo, hi)` gets
/// columns `c0..c0 + 4` in `lo` and, when `w >= 8`, the next four in `hi`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn filter_col16_sums_simd128(
    rows: &[i16],
    w: usize,
    f: &[i32; 8],
    add: i32,
    round1: u32,
    mut store: impl FnMut(usize, std::arch::wasm32::v128, std::arch::wasm32::v128),
) {
    use std::arch::wasm32::*;
    assert!(w >= 4 && rows.len() >= 8 * w);
    let (coef, used) = tap_pairs(f);
    let add = i32x4_splat(add);
    let rp = rows.as_ptr();
    if w < 8 {
        // Four columns: the low halves alone. (Blocks are 4, 8, 16, ...
        // wide; anything else below 8 goes four at a time from each end.)
        let mut c = 0;
        while c < w {
            let c0 = c.min(w - 4);
            let mut lo = add;
            for k in 0..4 {
                if !used[k] {
                    continue;
                }
                // SAFETY: four samples of rows 2k and 2k + 1, c0 + 4 <= w.
                let (a, b) = unsafe {
                    (
                        v128_load64_zero(rp.add(2 * k * w + c0) as *const u64),
                        v128_load64_zero(rp.add((2 * k + 1) * w + c0) as *const u64),
                    )
                };
                let l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b);
                lo = i32x4_add(lo, i32x4_dot_i16x8(l, coef[k]));
            }
            let lo = i32x4_shr(lo, round1);
            store(c0, lo, lo);
            c = c0 + 4;
        }
        return;
    }
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - 8);
        let (mut lo, mut hi) = (add, add);
        for k in 0..4 {
            if !used[k] {
                continue;
            }
            // SAFETY: eight samples of rows 2k and 2k + 1, c0 + 8 <= w.
            let (a, b) = unsafe {
                (
                    v128_load(rp.add(2 * k * w + c0) as *const v128),
                    v128_load(rp.add((2 * k + 1) * w + c0) as *const v128),
                )
            };
            let l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b);
            let h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b);
            lo = i32x4_add(lo, i32x4_dot_i16x8(l, coef[k]));
            hi = i32x4_add(hi, i32x4_dot_i16x8(h, coef[k]));
        }
        store(c0, i32x4_shr(lo, round1), i32x4_shr(hi, round1));
        c = c0 + 8;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_col16_simd128(rows: &[i16], w: usize, f: &[i32; 8], round1: u32, out: &mut [i32]) {
    use std::arch::wasm32::*;
    assert!(out.len() == w);
    let op = out.as_mut_ptr();
    filter_col16_sums_simd128(rows, w, f, (1 << round1) >> 1, round1, |c0, lo, hi| {
        // SAFETY: c0 + 4 <= w = out.len(), and c0 + 8 <= w when w >= 8.
        unsafe {
            v128_store(op.add(c0) as *mut v128, lo);
            if w >= 8 {
                v128_store(op.add(c0 + 4) as *mut v128, hi);
            }
        }
    });
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_col16_compound_simd128(
    rows: &[i16],
    w: usize,
    f: &[i32; 8],
    round1: u32,
    bias: i32,
    out: &mut [i16],
) {
    use std::arch::wasm32::*;
    assert!(out.len() == w);
    let op = out.as_mut_ptr();
    let add = ((1 << round1) >> 1) - (bias << round1);
    filter_col16_sums_simd128(rows, w, f, add, round1, |c0, lo, hi| {
        // The sums fit 16 bits (see `Compound`): the narrowing changes
        // nothing.
        let v = i16x8_narrow_i32x4(lo, hi);
        // SAFETY: c0 + 4 <= w = out.len(), and c0 + 8 <= w when w >= 8.
        unsafe {
            if w >= 8 {
                v128_store(op.add(c0) as *mut v128, v);
            } else {
                v128_store64_lane::<0>(v, op.add(c0) as *mut u64);
            }
        }
    });
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn avg_samples_simd128(a: &[u16], b: &[u16], dst: &mut [u16]) {
    use std::arch::wasm32::*;
    let w = dst.len();
    assert!(w >= 8 && a.len() >= w && b.len() >= w);
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - 8);
        // SAFETY: c0 + 8 <= w, which none of the three slices is shorter
        // than.
        unsafe {
            let va = v128_load(ap.add(c0) as *const v128);
            let vb = v128_load(bp.add(c0) as *const v128);
            v128_store(dp.add(c0) as *mut v128, u16x8_avgr(va, vb));
        }
        c = c0 + 8;
    }
}

/// Eight samples at a time, in 16-bit lanes with saturation doing the
/// clip: the sum saturates only where the result is past the largest
/// sample anyway, `max` with `-add` and a saturating `+ add` leave 0..32767,
/// and that shifted down is at most the largest sample at 10 and 12 bits
/// (at 8 it is clipped).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn compound_avg_simd128(a: &[i16], b: &[i16], c: Compound, dst: &mut [u16]) {
    use std::arch::wasm32::*;
    let w = dst.len();
    assert!(w >= 8 && a.len() >= w && b.len() >= w);
    let shift = 1 + c.post;
    let neg_add = i16x8_splat(-((1 << c.post) + 2 * c.bias) as i16);
    let maxv = i16x8_splat(c.maxv as i16);
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    let mut col = 0;
    while col < w {
        let c0 = col.min(w - 8);
        // SAFETY: c0 + 8 <= w, which neither `a` nor `b` is shorter than.
        let (va, vb) = unsafe {
            (
                v128_load(ap.add(c0) as *const v128),
                v128_load(bp.add(c0) as *const v128),
            )
        };
        let s = i16x8_max(i16x8_add_sat(va, vb), neg_add);
        let v = i16x8_min(i16x8_shr(i16x8_sub_sat(s, neg_add), shift), maxv);
        // SAFETY: c0 + 8 <= w = dst.len().
        unsafe { v128_store(dp.add(c0) as *mut v128, v) };
        col = c0 + 8;
    }
}

/// [`taps8`] in 16-bit lanes that wrap, eight outputs at a time (four for a
/// block under eight wide): a multiply-accumulate per tap.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn taps8_neon<const N: usize>(
    src: &[u16],
    stride: usize,
    step: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    shift: u32,
    out: &mut [i16],
) {
    use std::arch::aarch64::*;
    assert!(taps.len() == N && w >= 4 && rows > 0);
    assert!(src.len() >= (rows - 1) * stride + (N - 1) * step + w && out.len() >= rows * w);
    let add = (LIFT8 + ((1 << shift) >> 1)) as u16;
    let sub = (LIFT8 >> shift) as i16;
    // A shift left by a negative count is the shift right.
    let down = -(shift as i16);
    for r in 0..rows {
        // SAFETY: row r of each, which the assertions put inside.
        let (sp, op) = unsafe { (src.as_ptr().add(r * stride), out.as_mut_ptr().add(r * w)) };
        // Whole vectors, then one more ending at the last column
        // (overlapping what is done: the same values are stored again).
        let mut c = 0;
        if w >= 8 {
            while c < w {
                let c0 = c.min(w - 8);
                let mut acc = vdupq_n_u16(add);
                for t in 0..N {
                    // SAFETY: eight samples under tap t, the last of them
                    // at (rows - 1) * stride + (N - 1) * step + w - 1 or
                    // before.
                    let s = unsafe { vld1q_u16(sp.add(t * step + c0)) };
                    acc = vmlaq_n_u16(acc, s, taps[t] as u16);
                }
                let v = vreinterpretq_s16_u16(vshlq_u16(acc, vdupq_n_s16(down)));
                // SAFETY: c0 + 8 <= w.
                unsafe { vst1q_s16(op.add(c0), vsubq_s16(v, vdupq_n_s16(sub))) };
                c = c0 + 8;
            }
        } else {
            while c < w {
                let c0 = c.min(w - 4);
                let mut acc = vdup_n_u16(add);
                for t in 0..N {
                    // SAFETY: four samples under tap t, as above.
                    let s = unsafe { vld1_u16(sp.add(t * step + c0)) };
                    acc = vmla_n_u16(acc, s, taps[t] as u16);
                }
                let v = vreinterpret_s16_u16(vshl_u16(acc, vdup_n_s16(down)));
                // SAFETY: c0 + 4 <= w.
                unsafe { vst1_s16(op.add(c0), vsub_s16(v, vdup_n_s16(sub))) };
                c = c0 + 4;
            }
        }
    }
}

/// The horizontal filter over samples of any depth, `(sum + add) >>
/// round0`: a widening multiply-accumulate per tap into 32-bit lanes,
/// narrowed again (the result fits 16 bits).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows16_neon<const N: usize>(
    src: &[u16],
    stride: usize,
    w: usize,
    rows: usize,
    taps: &[i16],
    add: i32,
    round0: u32,
    out: &mut [i16],
) {
    use std::arch::aarch64::*;
    assert!(taps.len() == N && w >= 4 && rows > 0);
    assert!(src.len() >= (rows - 1) * stride + N - 1 + w && out.len() >= rows * w);
    let down = vdupq_n_s32(-(round0 as i32));
    for r in 0..rows {
        // SAFETY: row r of each, which the assertions put inside.
        let (sp, op) = unsafe { (src.as_ptr().add(r * stride), out.as_mut_ptr().add(r * w)) };
        let mut c = 0;
        if w >= 8 {
            while c < w {
                let c0 = c.min(w - 8);
                let (mut lo, mut hi) = (vdupq_n_s32(add), vdupq_n_s32(add));
                for t in 0..N {
                    // SAFETY: eight samples under tap t; c0 + t + 8 <= w +
                    // N - 1.
                    let s = vreinterpretq_s16_u16(unsafe { vld1q_u16(sp.add(c0 + t)) });
                    lo = vmlal_n_s16(lo, vget_low_s16(s), taps[t]);
                    hi = vmlal_high_n_s16(hi, s, taps[t]);
                }
                let v = vcombine_s16(
                    vmovn_s32(vshlq_s32(lo, down)),
                    vmovn_s32(vshlq_s32(hi, down)),
                );
                // SAFETY: c0 + 8 <= w.
                unsafe { vst1q_s16(op.add(c0), v) };
                c = c0 + 8;
            }
        } else {
            while c < w {
                let c0 = c.min(w - 4);
                let mut lo = vdupq_n_s32(add);
                for t in 0..N {
                    // SAFETY: four samples under tap t; c0 + t + 4 <= w +
                    // N - 1.
                    let s = vreinterpret_s16_u16(unsafe { vld1_u16(sp.add(c0 + t)) });
                    lo = vmlal_n_s16(lo, s, taps[t]);
                }
                // SAFETY: c0 + 4 <= w.
                unsafe { vst1_s16(op.add(c0), vmovn_s32(vshlq_s32(lo, down))) };
                c = c0 + 4;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn avg_samples_neon(a: &[u16], b: &[u16], dst: &mut [u16]) {
    use std::arch::aarch64::*;
    let w = dst.len();
    assert!(w >= 8 && a.len() >= w && b.len() >= w);
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - 8);
        // SAFETY: c0 + 8 <= w, which none of the three slices is shorter
        // than.
        unsafe {
            let v = vrhaddq_u16(vld1q_u16(ap.add(c0)), vld1q_u16(bp.add(c0)));
            vst1q_u16(dp.add(c0), v);
        }
        c = c0 + 8;
    }
}

/// The SIMD128 version's arithmetic: 16-bit lanes, saturation as the clip.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compound_avg_neon(a: &[i16], b: &[i16], c: Compound, dst: &mut [u16]) {
    use std::arch::aarch64::*;
    let w = dst.len();
    assert!(w >= 8 && a.len() >= w && b.len() >= w);
    let shift = vdupq_n_s16(-(1 + c.post as i16));
    let neg_add = vdupq_n_s16(-((1 << c.post) + 2 * c.bias) as i16);
    let maxv = vdupq_n_s16(c.maxv as i16);
    let (ap, bp, dp) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    let mut col = 0;
    while col < w {
        let c0 = col.min(w - 8);
        // SAFETY: c0 + 8 <= w, which neither `a` nor `b` is shorter than.
        let (va, vb) = unsafe { (vld1q_s16(ap.add(c0)), vld1q_s16(bp.add(c0))) };
        let s = vmaxq_s16(vqaddq_s16(va, vb), neg_add);
        let v = vminq_s16(vshlq_s16(vqsubq_s16(s, neg_add), shift), maxv);
        // SAFETY: c0 + 8 <= w = dst.len().
        unsafe { vst1q_u16(dp.add(c0), vreinterpretq_u16_s16(v)) };
        col = c0 + 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx2(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let w = out.len();
    let add = _mm256_set1_epi32((1 << round0) >> 1);
    let shift = _mm_cvtsi32_si128(round0 as i32);
    let taps: [__m256i; 8] = std::array::from_fn(|t| _mm256_set1_epi32(f[t]));
    let sp = src.as_ptr();
    let op = out.as_mut_ptr();
    let mut c = 0;
    // Whole vectors, then one more ending at the last column (overlapping
    // what is done: the same values are stored again).
    while c < w {
        let c0 = c.min(w - 8);
        let mut acc = _mm256_setzero_si256();
        if f[3] == 128 {
            // SAFETY: c0 + 3 + 8 <= w + 7 <= src.len().
            let s = unsafe { _mm_loadu_si128(sp.add(c0 + 3) as *const __m128i) };
            acc = _mm256_slli_epi32(_mm256_cvtepu16_epi32(s), 7);
        } else {
            for (t, tap) in taps.iter().enumerate() {
                // SAFETY: c0 + t + 8 <= w + 7 <= src.len().
                let s = unsafe { _mm_loadu_si128(sp.add(c0 + t) as *const __m128i) };
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_cvtepu16_epi32(s), *tap));
            }
        }
        let v = _mm256_sra_epi32(_mm256_add_epi32(acc, add), shift);
        // SAFETY: c0 + 8 <= w.
        unsafe { _mm256_storeu_si256(op.add(c0) as *mut __m256i, v) };
        c = c0 + 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_col_avx2(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let add = _mm256_set1_epi32((1 << round1) >> 1);
    let shift = _mm_cvtsi32_si128(round1 as i32);
    let taps: [__m256i; 8] = std::array::from_fn(|t| _mm256_set1_epi32(f[t]));
    let ip = inter.as_ptr();
    let op = out.as_mut_ptr();
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - 8);
        let mut acc = _mm256_setzero_si256();
        if f[3] == 128 {
            // SAFETY: row r + 3 of the intermediate array.
            let s = unsafe { _mm256_loadu_si256(ip.add((r + 3) * w + c0) as *const __m256i) };
            acc = _mm256_slli_epi32(s, 7);
        } else {
            for (t, tap) in taps.iter().enumerate() {
                // SAFETY: rows r..r + 8 of the intermediate array.
                let s = unsafe { _mm256_loadu_si256(ip.add((r + t) * w + c0) as *const __m256i) };
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tap));
            }
        }
        let v = _mm256_sra_epi32(_mm256_add_epi32(acc, add), shift);
        // SAFETY: c0 + 8 <= w = out.len().
        unsafe { _mm256_storeu_si256(op.add(c0) as *mut __m256i, v) };
        c = c0 + 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::SUBPEL_FILTERS;

    /// The SIMD filters agree with the scalar ones for every filter and
    /// phase, at several widths, on 8-, 10- and 12-bit samples.
    /// The horizontal filter's result fits 16 bits for any samples: the
    /// extremes are all-maximum samples under the positive taps or under
    /// the negative ones.
    #[test]
    fn intermediates_fit_sixteen_bits() {
        for (bd, round0) in [(8u32, 3u32), (10, 3), (12, 5)] {
            let max = (1i32 << bd) - 1;
            for f in SUBPEL_FILTERS.iter().flatten() {
                let pos: i32 = f.iter().filter(|&&t| t > 0).sum();
                let neg: i32 = f.iter().filter(|&&t| t < 0).sum();
                let add = (1 << round0) >> 1;
                assert!((max * pos + add) >> round0 <= i16::MAX as i32, "{bd} {f:?}");
                assert!((max * neg + add) >> round0 >= i16::MIN as i32, "{bd} {f:?}");
            }
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10, 12] {
            for filt in SUBPEL_FILTERS.iter() {
                for f in filt.iter() {
                    for w in [4usize, 8, 12, 16, 32, 64, 128] {
                        let src: Vec<u16> = (0..w + 7).map(|_| rnd(1 << bd) as u16).collect();
                        let round0 = if bd == 12 { 5 } else { 3 };
                        let mut a = vec![0i32; w];
                        let mut b = vec![0i32; w];
                        filter_row_scalar(&src, f, round0, &mut a);
                        filter_row(&src, f, round0, &mut b);
                        assert_eq!(a, b, "row {bd}-bit w {w} {f:?}");
                        let inter: Vec<i32> = (0..(w * 9))
                            .map(|_| rnd(1 << (bd + 4)) as i32 - (1 << (bd + 2)))
                            .collect();
                        for round1 in [7u32, 11, 9] {
                            filter_col_scalar(&inter, 1, w, f, round1, &mut a);
                            filter_col(&inter, 1, w, f, round1, &mut b);
                            assert_eq!(a, b, "col {bd}-bit w {w} {f:?}");
                        }
                        // The vertical pass over 16-bit intermediates is the
                        // same as over 32-bit ones.
                        let inter16: Vec<i16> = (0..(w * 9))
                            .map(|_| (rnd(1 << (bd + 3)) as i32 - (1 << (bd + 1))) as i16)
                            .collect();
                        let inter32: Vec<i32> = inter16.iter().map(|&v| v as i32).collect();
                        for round1 in [7u32, 11, 9] {
                            filter_col_scalar(&inter32, 1, w, f, round1, &mut a);
                            filter_col16_scalar(&inter16[w..], w, f, round1, &mut b);
                            assert_eq!(a, b, "col16 scalar {bd}-bit w {w} {f:?}");
                            filter_col16(&inter16[w..], w, f, round1, &mut b);
                            assert_eq!(a, b, "col16 {bd}-bit w {w} {f:?}");
                        }
                        // A compound prediction's vertical filter: the same
                        // values less the bias, in 16 bits.
                        let bias = Compound::new(bd, 0).bias;
                        let less_bias =
                            |v: &[i32]| v.iter().map(|&v| (v - bias) as i16).collect::<Vec<_>>();
                        let mut a16 = vec![0i16; w];
                        let mut b16 = vec![0i16; w];
                        // (The intermediates here are the horizontal
                        // filter's range at this depth.)
                        let inter16: Vec<i16> = (0..(w * 8))
                            .map(|_| (rnd(31_000) as i32 - 7_400) as i16)
                            .map(|v| if bd == 8 { v / 4 } else { v })
                            .collect();
                        let inter32: Vec<i32> = inter16.iter().map(|&v| v as i32).collect();
                        filter_col_scalar(&inter32, 0, w, f, 7, &mut a);
                        filter_col16_compound_scalar(&inter16, w, f, 7, bias, &mut a16);
                        assert_eq!(less_bias(&a), a16, "col16 compound scalar {bd}-bit w {w}");
                        filter_col16_compound(&inter16, w, f, 7, bias, &mut b16);
                        assert_eq!(a16, b16, "col16 compound {bd}-bit w {w} {f:?}");
                    }
                }
            }
        }
    }

    /// The range of a compound prediction at each bit depth, as `(min,
    /// max)`: the extremes of the horizontal filter (all-maximum samples
    /// under the positive taps or under the negative ones) put through the
    /// vertical one the same way.
    fn compound_range(bd: u32) -> (i32, i32) {
        let round0 = if bd == 12 { 5 } else { 3 };
        let max = (1i32 << bd) - 1;
        let (mut lo, mut hi) = (0, 0);
        for fx in SUBPEL_FILTERS.iter().flatten() {
            let pos: i32 = fx.iter().filter(|&&t| t > 0).sum();
            let neg: i32 = fx.iter().filter(|&&t| t < 0).sum();
            let h_max = (max * pos + ((1 << round0) >> 1)) >> round0;
            let h_min = (max * neg + ((1 << round0) >> 1)) >> round0;
            for fy in SUBPEL_FILTERS.iter().flatten() {
                let pos: i32 = fy.iter().filter(|&&t| t > 0).sum();
                let neg: i32 = fy.iter().filter(|&&t| t < 0).sum();
                hi = hi.max((h_max * pos + h_min * neg + 64) >> 7);
                lo = lo.min((h_min * pos + h_max * neg + 64) >> 7);
            }
        }
        (lo, hi)
    }

    /// A compound prediction less its bias fits 16 bits at every depth.
    #[test]
    fn compound_predictions_fit_sixteen_bits() {
        for bd in [8u32, 10, 12] {
            let (lo, hi) = compound_range(bd);
            let bias = Compound::new(bd, 0).bias;
            assert!(lo - bias >= i16::MIN as i32, "{bd}-bit: {lo}");
            assert!(hi - bias <= i16::MAX as i32, "{bd}-bit: {hi}");
        }
    }

    /// The blends agree with the specification's arithmetic on 32-bit
    /// predictions, over the whole range a prediction can take, its
    /// extremes together included, at every depth and width.
    #[test]
    fn blends_match_the_specification() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10, 12] {
            let post = if bd == 12 { 2 } else { 4 };
            let c = Compound::new(bd, post);
            let (lo, hi) = compound_range(bd);
            for w in [4usize, 8, 12, 16, 32, 64, 128] {
                for round in 0..40 {
                    // Predictions as the specification has them, and as
                    // they are stored.
                    let mut pick = || match (round, rnd(8)) {
                        (0, _) | (_, 0) => hi,
                        (1, _) | (_, 1) => lo,
                        _ => lo + rnd((hi - lo + 1) as u64) as i32,
                    };
                    let p0: Vec<i32> = (0..w).map(|_| pick()).collect();
                    let p1: Vec<i32> = (0..w).map(|_| pick()).collect();
                    let a: Vec<i16> = p0.iter().map(|&v| (v - c.bias) as i16).collect();
                    let b: Vec<i16> = p1.iter().map(|&v| (v - c.bias) as i16).collect();
                    let spec = |f: &dyn Fn(usize) -> i32| -> Vec<u16> {
                        (0..w).map(|j| f(j).clamp(0, c.maxv) as u16).collect()
                    };
                    let mut out = vec![0u16; w];
                    let mut out_scalar = vec![0u16; w];

                    let want = spec(&|j| (p0[j] + p1[j] + (1 << post)) >> (1 + post));
                    compound_avg_scalar(&a, &b, c, &mut out_scalar);
                    compound_avg(&a, &b, c, &mut out);
                    assert_eq!(want, out_scalar, "avg scalar {bd}-bit w {w}");
                    assert_eq!(want, out, "avg {bd}-bit w {w}");

                    // Whole samples: the predictions are the samples times
                    // `1 << post`.
                    let s0: Vec<u16> = (0..w).map(|_| rnd(1 << bd) as u16).collect();
                    let s1: Vec<u16> = (0..w).map(|_| rnd(1 << bd) as u16).collect();
                    let want = spec(&|j| {
                        let (p0, p1) = ((s0[j] as i32) << post, (s1[j] as i32) << post);
                        (p0 + p1 + (1 << post)) >> (1 + post)
                    });
                    avg_samples_scalar(&s0, &s1, &mut out_scalar);
                    avg_samples(&s0, &s1, &mut out);
                    assert_eq!(want, out_scalar, "whole samples scalar {bd}-bit w {w}");
                    assert_eq!(want, out, "whole samples {bd}-bit w {w}");
                }
            }
        }
    }

    /// What [`LIFT8`] rests on: over 8-bit samples, every filter's sum with
    /// the lift and the rounding constant of any shift up to 7 is a 16-bit
    /// unsigned value, and the lift is a multiple of `1 << shift`.
    #[test]
    fn eight_bit_sums_fit_a_lifted_lane() {
        for f in SUBPEL_FILTERS.iter().flatten() {
            let pos: i32 = f.iter().filter(|&&t| t > 0).sum();
            let neg: i32 = f.iter().filter(|&&t| t < 0).sum();
            assert_eq!(pos + neg, 128, "{f:?}");
            for shift in 0..=7 {
                let round = (1 << shift) >> 1;
                assert!(255 * neg + LIFT8 + round >= 0, "{f:?}");
                assert!(255 * pos + LIFT8 + round <= u16::MAX as i32, "{f:?}");
                assert_eq!(LIFT8 % (1 << shift), 0);
            }
        }
    }

    /// The run of taps kept holds every tap that is not zero, in place.
    #[test]
    fn kept_taps_are_the_filter() {
        for f in SUBPEL_FILTERS.iter().flatten() {
            let t = Taps::new(f);
            assert!(
                matches!(t.len, 2 | 4 | 6 | 8) && t.first + t.len <= 8,
                "{f:?}"
            );
            let mut back = [0i32; 8];
            for (k, &v) in t.taps().iter().enumerate() {
                back[t.first + k] = v as i32;
            }
            assert_eq!(&back, f);
        }
        // The sets for narrow blocks keep four taps, the bilinear one two,
        // the regular and smooth ones six.
        for (set, len) in [(0usize, 6usize), (1, 6), (2, 8), (3, 2), (4, 4), (5, 4)] {
            for f in &SUBPEL_FILTERS[set][1..] {
                assert!(Taps::new(f).len <= len, "{f:?}");
            }
        }
    }

    /// The horizontal filter of a block, every version of it, against the
    /// specification's arithmetic (`Round2` of the 32-bit sum of all eight
    /// taps): every filter and phase, random blocks of every width with
    /// the extreme samples among them, a stride wider than the block, 8, 10
    /// and 12 bits, with and without a compound prediction's bias.
    #[test]
    fn block_rows_match_the_specification() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10, 12] {
            let round0 = if bd == 12 { 5 } else { 3 };
            let max = (1u64 << bd) - 1;
            for (set, filt) in SUBPEL_FILTERS.iter().enumerate() {
                for (phase, f) in filt.iter().enumerate() {
                    for w in [2usize, 4, 8, 16, 32, 64, 128] {
                        let rows = 1 + rnd(5) as usize;
                        let stride = w + 7 + rnd(9) as usize;
                        let kind = rnd(4);
                        let src: Vec<u16> = (0..rows * stride)
                            .map(|_| match (kind, rnd(4)) {
                                (0, _) | (1, 0) => max as u16,
                                (1, 1) => 0,
                                _ => rnd(max + 1) as u16,
                            })
                            .collect();
                        let taps = Taps::new(f);
                        // The kernels read from the first tap kept.
                        let from = &src[taps.first..];
                        for bias in [0, Compound::new(bd, 0).bias] {
                            let mut want = vec![0i16; rows * w];
                            let mut line = vec![0i32; w];
                            for r in 0..rows {
                                filter_row_scalar(
                                    &src[r * stride..][..w + 7],
                                    f,
                                    round0,
                                    &mut line,
                                );
                                for (o, &v) in want[r * w..][..w].iter_mut().zip(&line) {
                                    assert!(
                                        (i16::MIN as i32..=i16::MAX as i32).contains(&(v - bias))
                                    );
                                    *o = (v - bias) as i16;
                                }
                            }
                            let what =
                                format!("{bd}-bit set {set} phase {phase} w {w} bias {bias}");
                            let mut got = vec![0i16; rows * w];
                            filter_rows16(
                                from,
                                stride,
                                w,
                                rows,
                                &taps,
                                round0,
                                bias,
                                bd == 8,
                                &mut got,
                            );
                            assert_eq!(want, got, "{what}");
                            // The version for any depth, on these samples
                            // too.
                            got.fill(0);
                            filter_rows16(
                                from, stride, w, rows, &taps, round0, bias, false, &mut got,
                            );
                            assert_eq!(want, got, "{what}, 32-bit sums");
                            // And the scalar ones, whatever the target has.
                            let add = ((1 << round0) >> 1) - (bias << round0);
                            let t = taps.taps();
                            got.fill(0);
                            match t.len() {
                                2 => rows16_scalar::<2>(
                                    from, stride, w, rows, t, add, round0, &mut got,
                                ),
                                4 => rows16_scalar::<4>(
                                    from, stride, w, rows, t, add, round0, &mut got,
                                ),
                                6 => rows16_scalar::<6>(
                                    from, stride, w, rows, t, add, round0, &mut got,
                                ),
                                _ => rows16_scalar::<8>(
                                    from, stride, w, rows, t, add, round0, &mut got,
                                ),
                            }
                            assert_eq!(want, got, "{what}, scalar");
                            if bd == 8 && bias == 0 {
                                got.fill(0);
                                match t.len() {
                                    2 => taps8_scalar::<2>(
                                        from, stride, 1, w, rows, t, round0, &mut got,
                                    ),
                                    4 => taps8_scalar::<4>(
                                        from, stride, 1, w, rows, t, round0, &mut got,
                                    ),
                                    6 => taps8_scalar::<6>(
                                        from, stride, 1, w, rows, t, round0, &mut got,
                                    ),
                                    _ => taps8_scalar::<8>(
                                        from, stride, 1, w, rows, t, round0, &mut got,
                                    ),
                                }
                                assert_eq!(want, got, "{what}, scalar in 16 bits");
                            }
                        }
                    }
                }
            }
        }
    }
}
