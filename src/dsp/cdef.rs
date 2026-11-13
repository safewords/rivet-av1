//! The CDEF filter of one block (7.15.3), on a bordered window of the
//! deblocked samples: scalar, AVX2 / NEON versions chosen at run time, and
//! a WebAssembly SIMD128 version where the module is built with `simd128`
//! (a wasm module has its features or it does not load: nothing to detect),
//! all bit-exact.

/// A window sample outside the frame (`CdefAvailable` is 0): not a tap.
pub(crate) const NA: i32 = i32::MIN;

/// The window's row stride: an 8x8 block with a border of 2.
pub(crate) const WS: usize = 12;

/// The taps of one block's filter.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Taps {
    pub(crate) pri_str: i32,
    pub(crate) sec_str: i32,
    /// `damping - FloorLog2( strength )`, at least 0.
    pub(crate) pri_adj: i32,
    pub(crate) sec_adj: i32,
    pub(crate) pri_taps: [i32; 2],
    pub(crate) sec_taps: [i32; 2],
    /// Window offsets of the primary taps (k = 0, 1) and the secondary
    /// ones (k, direction - 2 / + 2); each is taken with both signs.
    pub(crate) pri_off: [isize; 2],
    pub(crate) sec_off: [[isize; 2]; 2],
}

/// Filters the `w` x `h` block whose top-left sample is at window
/// position (2, 2) into `res` (row stride `w`).
pub(crate) fn filter(win: &[i32; WS * WS], w: usize, h: usize, t: &Taps, res: &mut [u16; 64]) {
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        unsafe { filter_avx2(win, w, h, t, res) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { filter_neon(win, w, h, t, res) };
        #[allow(clippy::needless_return)]
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        filter_simd128(win, w, h, t, res);
        #[allow(clippy::needless_return)]
        return;
    }
    #[allow(unreachable_code)]
    filter_scalar(win, w, h, t, res)
}

#[inline(always)]
fn constrain(diff: i32, threshold: i32, adj: i32) -> i32 {
    let mag = diff.abs().min((threshold - (diff.abs() >> adj)).max(0));
    if diff < 0 { -mag } else { mag }
}

/// The reference version.
pub(crate) fn filter_scalar(
    win: &[i32; WS * WS],
    w: usize,
    h: usize,
    t: &Taps,
    res: &mut [u16; 64],
) {
    for i in 0..h {
        for j in 0..w {
            let ci = ((i + 2) * WS + j + 2) as isize;
            let x = win[ci as usize];
            let mut sum = 0i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    let p = win[(ci + sign * t.pri_off[k]) as usize];
                    if p != NA {
                        sum += t.pri_taps[k] * constrain(p - x, t.pri_str, t.pri_adj);
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for so in t.sec_off[k] {
                        let s = win[(ci + sign * so) as usize];
                        if s != NA {
                            sum += t.sec_taps[k] * constrain(s - x, t.sec_str, t.sec_adj);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
            res[i * w + j] = v.clamp(min, max) as u16;
        }
    }
}

/// Filters the `w` x `h` block at index `pos` of a plane (`stride` samples a
/// row) into `res` (row stride `w`), reading the plane itself: for a block
/// whose border of 2 is all available, so that no tap needs the window's
/// marking. `t`'s offsets are in samples of the plane.
///
/// A strength of zero contributes nothing, and its taps are not read. That
/// leaves them out of the minimum and maximum the result is clipped to as
/// well, which changes nothing: with one of the two sets of taps alone the
/// result already lies between the sample and the taps used (their weights
/// sum to 12/16, and `constrain` never passes the difference it is given).
pub(crate) fn filter_plane(
    src: &[u16],
    stride: usize,
    pos: usize,
    w: usize,
    h: usize,
    t: &Taps,
    res: &mut [u16; 64],
) {
    // Every tap of every sample is inside the plane.
    assert!(w <= 8 && h <= 8 && pos >= 2 * stride + 2);
    assert!(pos + (h + 1) * stride + w + 2 <= src.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w == 8 || w == 4 {
        filter_plane_simd128(src, stride, pos, w, h, t, res);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w == 8 || w == 4 {
        // SAFETY: NEON is part of the aarch64 baseline; the bounds are
        // asserted above.
        unsafe { filter_plane_neon(src, stride, pos, w, h, t, res) };
        return;
    }
    filter_plane_scalar(src, stride, pos, w, h, t, res)
}

/// Whether [`filter_plane`] is the faster way to filter a block that has
/// its whole border: everywhere but on x86-64 with AVX2, whose version of
/// the window filter ([`filter`]) has no counterpart here yet.
#[inline]
pub(crate) fn plane_filter_pays() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !crate::dsp::avx2()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// [`filter_plane`] with NEON: a row of the block at a time in 16-bit
/// lanes, as the SIMD128 version.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_plane_neon(
    src: &[u16],
    stride: usize,
    pos: usize,
    w: usize,
    h: usize,
    t: &Taps,
    res: &mut [u16; 64],
) {
    use std::arch::aarch64::*;
    assert!((w == 8 || w == 4) && h <= 8 && pos >= 2 * stride + 2);
    assert!(pos + (h + 1) * stride + w + 2 <= src.len());
    let sp = src.as_ptr();
    let wide = w == 8;
    // SAFETY (every call): `i` is a sample of the block or one of its taps,
    // at most two rows and two columns away, and `w` samples are read: all
    // inside `src` by the assertions above.
    let load = |i: isize| unsafe {
        if wide {
            vreinterpretq_s16_u16(vld1q_u16(sp.offset(i)))
        } else {
            let half = vld1_u16(sp.offset(i));
            vreinterpretq_s16_u16(vcombine_u16(half, half))
        }
    };
    let zero = vdupq_n_s16(0);
    let pri_str = vdupq_n_s16(t.pri_str as i16);
    let sec_str = vdupq_n_s16(t.sec_str as i16);
    let pri_shift = vdupq_n_s16(-(t.pri_adj as i16));
    let sec_shift = vdupq_n_s16(-(t.sec_adj as i16));
    for i in 0..h {
        let base = (pos + i * stride) as isize;
        let x = load(base);
        let mut sum = zero;
        let mut mx = x;
        let mut mn = x;
        let mut tap = |off: isize, strv: int16x8_t, shift: int16x8_t, k: i32| {
            let p = load(base + off);
            let d = vsubq_s16(p, x);
            let ad = vabsq_s16(d);
            let shifted = vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(ad), shift));
            let lim = vmaxq_s16(vsubq_s16(strv, shifted), zero);
            let mag = vminq_s16(ad, lim);
            // Negate where the difference is negative.
            let neg = vshrq_n_s16::<15>(d);
            let c = vsubq_s16(veorq_s16(mag, neg), neg);
            sum = vmlaq_n_s16(sum, c, k as i16);
            mx = vmaxq_s16(mx, p);
            mn = vminq_s16(mn, p);
        };
        if t.pri_str != 0 {
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str, pri_shift, t.pri_taps[k]);
                }
            }
        }
        if t.sec_str != 0 {
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str, sec_shift, t.sec_taps[k]);
                    }
                }
            }
        }
        // x + ((8 + sum - (sum < 0)) >> 4): the sign is -1 where it is set.
        let r = vshrq_n_s16::<4>(vaddq_s16(
            vaddq_s16(sum, vdupq_n_s16(8)),
            vshrq_n_s16::<15>(sum),
        ));
        let v = vreinterpretq_u16_s16(vmaxq_s16(vminq_s16(vaddq_s16(x, r), mx), mn));
        // SAFETY: row i of the result, `w` samples, i < h <= 8.
        unsafe {
            let o = res.as_mut_ptr().add(i * w);
            if wide {
                vst1q_u16(o, v);
            } else {
                vst1_u16(o, vget_low_u16(v));
            }
        }
    }
}

/// The reference version of [`filter_plane`].
pub(crate) fn filter_plane_scalar(
    src: &[u16],
    stride: usize,
    pos: usize,
    w: usize,
    h: usize,
    t: &Taps,
    res: &mut [u16; 64],
) {
    for i in 0..h {
        for j in 0..w {
            let ci = (pos + i * stride + j) as isize;
            let x = src[ci as usize] as i32;
            let mut sum = 0i32;
            let mut max = x;
            let mut min = x;
            if t.pri_str != 0 {
                for k in 0..2 {
                    for sign in [-1isize, 1] {
                        let p = src[(ci + sign * t.pri_off[k]) as usize] as i32;
                        sum += t.pri_taps[k] * constrain(p - x, t.pri_str, t.pri_adj);
                        max = max.max(p);
                        min = min.min(p);
                    }
                }
            }
            if t.sec_str != 0 {
                for k in 0..2 {
                    for sign in [-1isize, 1] {
                        for so in t.sec_off[k] {
                            let s = src[(ci + sign * so) as usize] as i32;
                            sum += t.sec_taps[k] * constrain(s - x, t.sec_str, t.sec_adj);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
            res[i * w + j] = v.clamp(min, max) as u16;
        }
    }
}

/// A row of the block at a time in 16-bit lanes (eight samples, or four
/// for a block four wide). Samples are at most 12 bits, a constrained
/// difference at most a strength (240), and the twelve weighted taps sum
/// to well under 2^15.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_plane_simd128(
    src: &[u16],
    stride: usize,
    pos: usize,
    w: usize,
    h: usize,
    t: &Taps,
    res: &mut [u16; 64],
) {
    use std::arch::wasm32::*;
    // The bounds every load below relies on (also checked by the caller).
    assert!((w == 8 || w == 4) && h <= 8 && pos >= 2 * stride + 2);
    assert!(pos + (h + 1) * stride + w + 2 <= src.len());
    let sp = src.as_ptr();
    let wide = w == 8;
    // SAFETY (every call): `i` is a sample of the block or one of its taps,
    // at most two rows and two columns away, and `w` samples are read: all
    // inside `src` by the assertions above.
    let load = |i: isize| unsafe {
        if wide {
            v128_load(sp.offset(i) as *const v128)
        } else {
            v128_load64_zero(sp.offset(i) as *const u64)
        }
    };
    let zero = i16x8_splat(0);
    let pri_str = i16x8_splat(t.pri_str as i16);
    let sec_str = i16x8_splat(t.sec_str as i16);
    let (pri_adj, sec_adj) = (t.pri_adj as u32, t.sec_adj as u32);
    for i in 0..h {
        let base = (pos + i * stride) as isize;
        let x = load(base);
        let mut sum = zero;
        let mut mx = x;
        let mut mn = x;
        let mut tap = |off: isize, strv: v128, adj: u32, k: i32| {
            let p = load(base + off);
            let d = i16x8_sub(p, x);
            let ad = i16x8_abs(d);
            let lim = i16x8_max(i16x8_sub(strv, u16x8_shr(ad, adj)), zero);
            let mag = i16x8_min(ad, lim);
            // Negate where the difference is negative.
            let neg = i16x8_shr(d, 15);
            let c = i16x8_sub(v128_xor(mag, neg), neg);
            sum = i16x8_add(sum, i16x8_mul(c, i16x8_splat(k as i16)));
            mx = i16x8_max(mx, p);
            mn = i16x8_min(mn, p);
        };
        if t.pri_str != 0 {
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str, pri_adj, t.pri_taps[k]);
                }
            }
        }
        if t.sec_str != 0 {
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str, sec_adj, t.sec_taps[k]);
                    }
                }
            }
        }
        // x + ((8 + sum - (sum < 0)) >> 4): the sign is -1 where it is set.
        let r = i16x8_shr(
            i16x8_add(i16x8_add(sum, i16x8_splat(8)), i16x8_shr(sum, 15)),
            4,
        );
        let v = i16x8_max(i16x8_min(i16x8_add(x, r), mx), mn);
        // SAFETY: row i of the result, `w` samples, i < h <= 8.
        unsafe {
            let o = res.as_mut_ptr().add(i * w);
            if wide {
                v128_store(o as *mut v128, v);
            } else {
                v128_store64_lane::<0>(v, o as *mut u64);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_avx2(win: &[i32; WS * WS], w: usize, h: usize, t: &Taps, res: &mut [u16; 64]) {
    use std::arch::x86_64::*;
    if w != 8 && w != 4 {
        return filter_scalar(win, w, h, t, res);
    }
    let p = win.as_ptr();
    let na8 = _mm256_set1_epi32(NA);
    let zero8 = _mm256_setzero_si256();
    let na4 = _mm_set1_epi32(NA);
    let zero4 = _mm_setzero_si128();
    let pri_str8 = _mm256_set1_epi32(t.pri_str);
    let sec_str8 = _mm256_set1_epi32(t.sec_str);
    let pri_adj = _mm_cvtsi32_si128(t.pri_adj);
    let sec_adj = _mm_cvtsi32_si128(t.sec_adj);
    let pri_str4 = _mm_set1_epi32(t.pri_str);
    let sec_str4 = _mm_set1_epi32(t.sec_str);
    for i in 0..h {
        let base = ((i + 2) * WS + 2) as isize;
        if w == 8 {
            // SAFETY (all loads): every offset stays inside the 12x12
            // window (taps reach at most 2 rows / columns out).
            let x = unsafe { _mm256_loadu_si256(p.offset(base) as *const __m256i) };
            let mut sum = zero8;
            let mut mx = x;
            let mut mn = x;
            let mut tap = |off: isize, str8: __m256i, adj: __m128i, k: i32| {
                let v = unsafe { _mm256_loadu_si256(p.offset(base + off) as *const __m256i) };
                let invalid = _mm256_cmpeq_epi32(v, na8);
                let d = _mm256_sub_epi32(v, x);
                let ad = _mm256_abs_epi32(d);
                let lim =
                    _mm256_max_epi32(_mm256_sub_epi32(str8, _mm256_srl_epi32(ad, adj)), zero8);
                let c = _mm256_sign_epi32(_mm256_min_epi32(ad, lim), d);
                let c = _mm256_andnot_si256(invalid, c);
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(c, _mm256_set1_epi32(k)));
                mx = _mm256_max_epi32(mx, v);
                mn = _mm256_min_epi32(mn, _mm256_blendv_epi8(v, x, invalid));
            };
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str8, pri_adj, t.pri_taps[k]);
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str8, sec_adj, t.sec_taps[k]);
                    }
                }
            }
            let neg = _mm256_cmpgt_epi32(zero8, sum);
            let r = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(8)), neg),
                4,
            );
            let v = _mm256_add_epi32(x, r);
            let v = _mm256_max_epi32(_mm256_min_epi32(v, mx), mn);
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256(v, 1);
            let packed = _mm_packus_epi32(lo, hi);
            // SAFETY: row i of 8 within the 64-sample result.
            unsafe { _mm_storeu_si128(res.as_mut_ptr().add(i * 8) as *mut __m128i, packed) };
        } else {
            let x = unsafe { _mm_loadu_si128(p.offset(base) as *const __m128i) };
            let mut sum = zero4;
            let mut mx = x;
            let mut mn = x;
            let mut tap = |off: isize, str4: __m128i, adj: __m128i, k: i32| {
                let v = unsafe { _mm_loadu_si128(p.offset(base + off) as *const __m128i) };
                let invalid = _mm_cmpeq_epi32(v, na4);
                let d = _mm_sub_epi32(v, x);
                let ad = _mm_abs_epi32(d);
                let lim = _mm_max_epi32(_mm_sub_epi32(str4, _mm_srl_epi32(ad, adj)), zero4);
                let c = _mm_sign_epi32(_mm_min_epi32(ad, lim), d);
                let c = _mm_andnot_si128(invalid, c);
                sum = _mm_add_epi32(sum, _mm_mullo_epi32(c, _mm_set1_epi32(k)));
                mx = _mm_max_epi32(mx, v);
                mn = _mm_min_epi32(mn, _mm_blendv_epi8(v, x, invalid));
            };
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str4, pri_adj, t.pri_taps[k]);
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str4, sec_adj, t.sec_taps[k]);
                    }
                }
            }
            let neg = _mm_cmpgt_epi32(zero4, sum);
            let r = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(sum, _mm_set1_epi32(8)), neg), 4);
            let v = _mm_add_epi32(x, r);
            let v = _mm_max_epi32(_mm_min_epi32(v, mx), mn);
            let packed = _mm_packus_epi32(v, v);
            // SAFETY: row i of 4 within the 64-sample result.
            unsafe { _mm_storel_epi64(res.as_mut_ptr().add(i * 4) as *mut __m128i, packed) };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon(win: &[i32; WS * WS], w: usize, h: usize, t: &Taps, res: &mut [u16; 64]) {
    use std::arch::aarch64::*;
    if w != 8 && w != 4 {
        return filter_scalar(win, w, h, t, res);
    }
    let p = win.as_ptr();
    let na = vdupq_n_s32(NA);
    let zero = vdupq_n_s32(0);
    let pri_str = vdupq_n_s32(t.pri_str);
    let sec_str = vdupq_n_s32(t.sec_str);
    let pri_shift = vdupq_n_s32(-t.pri_adj);
    let sec_shift = vdupq_n_s32(-t.sec_adj);
    for i in 0..h {
        for half in 0..w / 4 {
            let base = ((i + 2) * WS + 2 + half * 4) as isize;
            // SAFETY (all loads): inside the 12x12 window.
            let x = unsafe { vld1q_s32(p.offset(base)) };
            let mut sum = zero;
            let mut mx = x;
            let mut mn = x;
            let mut tap = |off: isize, strv: int32x4_t, shift: int32x4_t, k: i32| {
                let v = unsafe { vld1q_s32(p.offset(base + off)) };
                let invalid = vceqq_s32(v, na);
                let d = vsubq_s32(v, x);
                let ad = vabsq_s32(d);
                let shifted = vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(ad), shift));
                let lim = vmaxq_s32(vsubq_s32(strv, shifted), zero);
                let mag = vminq_s32(ad, lim);
                let neg = vcltq_s32(d, zero);
                let c = vbslq_s32(neg, vnegq_s32(mag), mag);
                let c = vbslq_s32(invalid, zero, c);
                sum = vmlaq_s32(sum, c, vdupq_n_s32(k));
                mx = vmaxq_s32(mx, v);
                mn = vminq_s32(mn, vbslq_s32(invalid, x, v));
            };
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str, pri_shift, t.pri_taps[k]);
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str, sec_shift, t.sec_taps[k]);
                    }
                }
            }
            let neg = vreinterpretq_s32_u32(vcltq_s32(sum, zero));
            let r = vshrq_n_s32::<4>(vaddq_s32(vaddq_s32(sum, vdupq_n_s32(8)), neg));
            let v = vaddq_s32(x, r);
            let v = vmaxq_s32(vminq_s32(v, mx), mn);
            let packed = vqmovun_s32(v);
            // SAFETY: four samples of row i within the 64-sample result.
            unsafe { vst1_u16(res.as_mut_ptr().add(i * w + half * 4), packed) };
        }
    }
}

/// Four samples of a row at a time in a `v128`, the NEON version's
/// arithmetic: 32-bit lanes, an unavailable tap's contribution selected
/// away after it is computed.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn filter_simd128(win: &[i32; WS * WS], w: usize, h: usize, t: &Taps, res: &mut [u16; 64]) {
    use std::arch::wasm32::*;
    if w != 8 && w != 4 {
        return filter_scalar(win, w, h, t, res);
    }
    let p = win.as_ptr();
    let na = i32x4_splat(NA);
    let zero = i32x4_splat(0);
    let pri_str = i32x4_splat(t.pri_str);
    let sec_str = i32x4_splat(t.sec_str);
    let (pri_adj, sec_adj) = (t.pri_adj as u32, t.sec_adj as u32);
    for i in 0..h {
        for half in 0..w / 4 {
            let base = ((i + 2) * WS + 2 + half * 4) as isize;
            // SAFETY (all loads): four samples inside the 12x12 window (a
            // block row starts at column 2, a tap is at most two rows and
            // two columns away).
            let x = unsafe { v128_load(p.offset(base) as *const v128) };
            let mut sum = zero;
            let mut mx = x;
            let mut mn = x;
            let mut tap = |off: isize, strv: v128, adj: u32, k: i32| {
                let v = unsafe { v128_load(p.offset(base + off) as *const v128) };
                let invalid = i32x4_eq(v, na);
                let d = i32x4_sub(v, x);
                let ad = i32x4_abs(d);
                let lim = i32x4_max(i32x4_sub(strv, u32x4_shr(ad, adj)), zero);
                let mag = i32x4_min(ad, lim);
                let c = v128_bitselect(i32x4_neg(mag), mag, i32x4_lt(d, zero));
                let c = v128_andnot(c, invalid);
                sum = i32x4_add(sum, i32x4_mul(c, i32x4_splat(k)));
                // NA is the least i32: it never raises the maximum.
                mx = i32x4_max(mx, v);
                mn = i32x4_min(mn, v128_bitselect(x, v, invalid));
            };
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    tap(sign * t.pri_off[k], pri_str, pri_adj, t.pri_taps[k]);
                    for so in t.sec_off[k] {
                        tap(sign * so, sec_str, sec_adj, t.sec_taps[k]);
                    }
                }
            }
            // x + ((8 + sum - (sum < 0)) >> 4): the comparison is -1 where
            // it holds.
            let r = i32x4_shr(
                i32x4_add(i32x4_add(sum, i32x4_splat(8)), i32x4_lt(sum, zero)),
                4,
            );
            let v = i32x4_max(i32x4_min(i32x4_add(x, r), mx), mn);
            let packed = u16x8_narrow_i32x4(v, v);
            // SAFETY: four samples of row i within the 64-sample result.
            unsafe {
                v128_store64_lane::<0>(packed, res.as_mut_ptr().add(i * w + half * 4) as *mut u64)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every version agrees with the scalar one on random windows (with
    /// unavailable samples), strengths and directions, at 8 and 10 bits.
    #[test]
    fn simd_matches_scalar() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for case in 0..4000 {
            let bd = if case % 2 == 0 { 8 } else { 10 };
            let mut win = [0i32; WS * WS];
            let base = rnd(1 << bd) as i32;
            for v in win.iter_mut() {
                *v = (base + rnd(64) as i32 - 32).clamp(0, (1 << bd) - 1);
                if rnd(9) == 0 {
                    *v = rnd(1 << bd) as i32;
                }
            }
            // Unavailable rows / columns at a frame edge.
            match rnd(5) {
                0 => win[..2 * WS].fill(NA),
                1 => (0..WS).for_each(|r| win[r * WS + WS - 1] = NA),
                _ => {}
            }
            let shift = bd - 8;
            let pri_str = (rnd(16) as i32) << shift;
            let sec_str = ([0, 1, 2, 4][rnd(4) as usize]) << shift;
            let damping = 3 + rnd(4) as i32 + shift;
            let dir = rnd(8) as usize;
            let off = |d: usize, k: usize| -> isize {
                crate::tables::CDEF_DIRECTIONS[d][k][0] as isize * WS as isize
                    + crate::tables::CDEF_DIRECTIONS[d][k][1] as isize
            };
            let adj = |s: i32| {
                if s != 0 {
                    (damping - crate::bits::floor_log2(s as u32) as i32).max(0)
                } else {
                    0
                }
            };
            let t = Taps {
                pri_str,
                sec_str,
                pri_adj: adj(pri_str),
                sec_adj: adj(sec_str),
                pri_taps: crate::tables::CDEF_PRI_TAPS[((pri_str >> shift) & 1) as usize],
                sec_taps: crate::tables::CDEF_SEC_TAPS[((pri_str >> shift) & 1) as usize],
                pri_off: [off(dir, 0), off(dir, 1)],
                sec_off: [
                    [off((dir + 6) & 7, 0), off((dir + 2) & 7, 0)],
                    [off((dir + 6) & 7, 1), off((dir + 2) & 7, 1)],
                ],
            };
            for (w, h) in [(8, 8), (4, 4), (4, 8)] {
                let mut a = [0u16; 64];
                let mut b = [0u16; 64];
                filter_scalar(&win, w, h, &t, &mut a);
                filter(&win, w, h, &t, &mut b);
                assert_eq!(a, b, "case {case} {w}x{h} {t:?}");
            }
            // With every sample available, filtering the same samples as a
            // plane (the window's rows are its rows) gives the same block,
            // zero strengths included.
            if win.iter().all(|&v| v != NA) {
                let plane: Vec<u16> = win.iter().map(|&v| v as u16).collect();
                for (w, h) in [(8, 8), (4, 4), (4, 8), (8, 4)] {
                    for (p, s) in [(true, true), (true, false), (false, true), (false, false)] {
                        let mut t = t;
                        if !p {
                            (t.pri_str, t.pri_adj) = (0, 0);
                        }
                        if !s {
                            (t.sec_str, t.sec_adj) = (0, 0);
                        }
                        let mut a = [0u16; 64];
                        let mut b = [0u16; 64];
                        let mut c = [0u16; 64];
                        filter_scalar(&win, w, h, &t, &mut a);
                        filter_plane_scalar(&plane, WS, 2 * WS + 2, w, h, &t, &mut b);
                        filter_plane(&plane, WS, 2 * WS + 2, w, h, &t, &mut c);
                        assert_eq!(a[..w * h], b[..w * h], "plane {case} {w}x{h} {t:?}");
                        assert_eq!(b[..w * h], c[..w * h], "plane simd {case} {w}x{h} {t:?}");
                    }
                }
            }
        }
    }
}
