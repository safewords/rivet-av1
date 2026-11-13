//! The 8-tap interpolation filters of inter prediction (7.11.3.4), one
//! row at a time: scalar, and AVX2 versions chosen at run time (NEON
//! comes from the compiler: it is part of the aarch64 baseline and these
//! loops vectorise), bit-exact.

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

/// The horizontal filter into 16-bit intermediates: `out[c] = Round2(sum_t
/// f[t] * src[c + t], round0)`, `src` holding `out.len() + 7` samples.
///
/// The intermediate of the prediction process fits 16 bits at every bit
/// depth (a sample of at most `BitDepth` bits times taps whose positive
/// part sums to at most 184, shifted down by `round0`: under 2^15; see the
/// test), which halves what the vertical pass reads and lets both passes
/// multiply 16-bit lanes.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn filter_row16(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i16]) {
    debug_assert!(src.len() >= out.len() + 7);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if out.len() >= 8 {
        filter_row16_simd128(src, f, round0, out);
        return;
    }
    filter_row16_scalar(src, f, round0, out)
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

pub(crate) fn filter_row16_scalar(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i16]) {
    let add = (1 << round0) >> 1;
    for (o, s) in out.iter_mut().zip(src.windows(8)) {
        let mut acc = add;
        for t in 0..8 {
            acc += f[t] * s[t] as i32;
        }
        *o = (acc >> round0) as i16;
    }
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

/// Eight outputs at a time: taps two by two, the samples under each pair
/// interleaved so that one 16-bit dot product gives four sums.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_row16_simd128(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i16]) {
    use std::arch::wasm32::*;
    let w = out.len();
    assert!(w >= 8 && src.len() >= w + 7);
    let (coef, used) = tap_pairs(f);
    let add = i32x4_splat((1 << round0) >> 1);
    let sp = src.as_ptr();
    let op = out.as_mut_ptr();
    let mut c = 0;
    // Whole vectors, then one more ending at the last column (overlapping
    // what is done: the same values are stored again).
    while c < w {
        let c0 = c.min(w - 8);
        let (mut lo, mut hi) = (add, add);
        for k in 0..4 {
            if !used[k] {
                continue;
            }
            // SAFETY: c0 + 2k + 1 + 8 <= w - 8 + 7 + 8 = w + 7 <= src.len().
            let (a, b) = unsafe {
                (
                    v128_load(sp.add(c0 + 2 * k) as *const v128),
                    v128_load(sp.add(c0 + 2 * k + 1) as *const v128),
                )
            };
            let l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b);
            let h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b);
            lo = i32x4_add(lo, i32x4_dot_i16x8(l, coef[k]));
            hi = i32x4_add(hi, i32x4_dot_i16x8(h, coef[k]));
        }
        let v = i16x8_narrow_i32x4(i32x4_shr(lo, round0), i32x4_shr(hi, round0));
        // SAFETY: c0 + 8 <= w.
        unsafe { v128_store(op.add(c0) as *mut v128, v) };
        c = c0 + 8;
    }
}

/// Eight (or, for a block four wide, four) outputs at a time: rows two by
/// two, interleaved as in the horizontal filter.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_col16_simd128(rows: &[i16], w: usize, f: &[i32; 8], round1: u32, out: &mut [i32]) {
    use std::arch::wasm32::*;
    assert!(w >= 4 && rows.len() >= 8 * w && out.len() == w);
    let (coef, used) = tap_pairs(f);
    let add = i32x4_splat((1 << round1) >> 1);
    let rp = rows.as_ptr();
    let op = out.as_mut_ptr();
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
            // SAFETY: c0 + 4 <= w = out.len().
            unsafe { v128_store(op.add(c0) as *mut v128, i32x4_shr(lo, round1)) };
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
        // SAFETY: c0 + 8 <= w = out.len().
        unsafe {
            v128_store(op.add(c0) as *mut v128, i32x4_shr(lo, round1));
            v128_store(op.add(c0 + 4) as *mut v128, i32x4_shr(hi, round1));
        }
        c = c0 + 8;
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
                        // The 16-bit intermediates: the same values, and the
                        // vertical pass over them the same as over 32-bit ones.
                        let mut a16 = vec![0i16; w];
                        let mut b16 = vec![0i16; w];
                        filter_row16_scalar(&src, f, round0, &mut a16);
                        filter_row16(&src, f, round0, &mut b16);
                        assert_eq!(a16, b16, "row16 {bd}-bit w {w} {f:?}");
                        let wide: Vec<i32> = a16.iter().map(|&v| v as i32).collect();
                        filter_row_scalar(&src, f, round0, &mut a);
                        assert_eq!(wide, a, "row16 {bd}-bit w {w} {f:?} fits 16 bits");
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
                    }
                }
            }
        }
    }
}
