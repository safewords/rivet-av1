//! The encoder's distortion kernels: sums of absolute differences, of
//! absolute Hadamard-transformed differences (SATD) and of squared
//! differences, between a source block and a prediction or
//! reconstruction. The SIMD versions (AVX2, NEON) compute exactly what
//! the scalar ones do (integer sums; tested against them).

/// A row source of the SATD: samples (`u16`) or a prediction (`i32`).
pub(crate) trait Px: Copy {
    fn get(self) -> i32;
}

impl Px for u16 {
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
}

impl Px for i32 {
    #[inline(always)]
    fn get(self) -> i32 {
        self
    }
}

/// Sum of absolute differences over `h` rows of `w` samples, or
/// `u64::MAX` if it exceeds `limit`. The sum is checked against the limit
/// as it grows, to stop early; how often does not change the result (a
/// partial sum above the limit means the whole one is), so each version
/// checks where it suits it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sad(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u64,
) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() && w >= 16 {
        // SAFETY: AVX2 is available.
        return unsafe { sad_avx2(a, a_stride, b, b_stride, w, h, limit) };
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 4 {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { sad_neon(a, a_stride, b, b_stride, w, h, limit) };
    }
    sad_scalar(a, a_stride, b, b_stride, w, h, limit)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sad_scalar(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u64,
) -> u64 {
    let mut s = 0u64;
    for i in 0..h {
        let ra = &a[i * a_stride..i * a_stride + w];
        let rb = &b[i * b_stride..i * b_stride + w];
        let mut acc = 0u32;
        for (&x, &y) in ra.iter().zip(rb) {
            acc += (x as i32 - y as i32).unsigned_abs();
        }
        s += acc as u64;
        if s > limit {
            return u64::MAX;
        }
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sad_avx2(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u64,
) -> u64 {
    use std::arch::x86_64::*;
    let mut s = 0u64;
    let ones = _mm256_set1_epi16(1);
    for i in 0..h {
        let ra = &a[i * a_stride..i * a_stride + w];
        let rb = &b[i * b_stride..i * b_stride + w];
        let mut acc = _mm256_setzero_si256();
        let mut j = 0;
        while j + 16 <= w {
            // SAFETY: j + 16 <= w, the length of both rows.
            let (x, y) = unsafe {
                (
                    _mm256_loadu_si256(ra.as_ptr().add(j) as *const __m256i),
                    _mm256_loadu_si256(rb.as_ptr().add(j) as *const __m256i),
                )
            };
            let d = _mm256_or_si256(_mm256_subs_epu16(x, y), _mm256_subs_epu16(y, x));
            // Differences of up to 12-bit samples fit madd's signed lanes.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, ones));
            j += 16;
        }
        let mut lanes = [0u32; 8];
        // SAFETY: eight 32-bit lanes into eight u32s.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc) };
        let mut t: u32 = lanes.iter().sum();
        for k in j..w {
            t += (ra[k] as i32 - rb[k] as i32).unsigned_abs();
        }
        s += t as u64;
        if s > limit {
            return u64::MAX;
        }
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sad_neon(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u64,
) -> u64 {
    use std::arch::aarch64::*;
    let mut s = 0u64;
    // Four rows at a time between checks of the limit, the lanes summed
    // once for them (a lane holds at most 4 * 2 * 4095 per 8 samples of a
    // row: no overflow at any width).
    let mut i = 0;
    while i < h {
        let rows = (h - i).min(4);
        let mut acc = vdupq_n_u32(0);
        let mut acc4 = vdup_n_u32(0);
        let mut tail = 0u32;
        for r in i..i + rows {
            let ra = &a[r * a_stride..r * a_stride + w];
            let rb = &b[r * b_stride..r * b_stride + w];
            let mut j = 0;
            while j + 8 <= w {
                // SAFETY: j + 8 <= w, the length of both rows.
                let (x, y) =
                    unsafe { (vld1q_u16(ra.as_ptr().add(j)), vld1q_u16(rb.as_ptr().add(j))) };
                acc = vpadalq_u16(acc, vabdq_u16(x, y));
                j += 8;
            }
            if j + 4 <= w {
                // SAFETY: j + 4 <= w, the length of both rows.
                let (x, y) =
                    unsafe { (vld1_u16(ra.as_ptr().add(j)), vld1_u16(rb.as_ptr().add(j))) };
                acc4 = vpadal_u16(acc4, vabd_u16(x, y));
                j += 4;
            }
            for k in j..w {
                tail += (ra[k] as i32 - rb[k] as i32).unsigned_abs();
            }
        }
        s += (vaddvq_u32(acc) + vaddv_u32(acc4) + tail) as u64;
        if s > limit {
            return u64::MAX;
        }
        i += rows;
    }
    s
}

/// Sum of squared differences of two rows.
pub(crate) fn sse_row(a: &[u16], b: &[u16]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 16 && crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        return unsafe { sse_row_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= 8 {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { sse_row_neon(a, b) };
    }
    sse_row_scalar(a, b)
}

pub(crate) fn sse_row_scalar(a: &[u16], b: &[u16]) -> u64 {
    let mut s = 0u64;
    for (&u, &v) in a.iter().zip(b) {
        let d = u as i64 - v as i64;
        s += (d * d) as u64;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sse_row_avx2(a: &[u16], b: &[u16]) -> u64 {
    use std::arch::x86_64::*;
    let w = a.len().min(b.len());
    let mut acc = _mm256_setzero_si256();
    let mut j = 0;
    while j + 16 <= w {
        // SAFETY: j + 16 <= w, the length of both rows.
        let (x, y) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().add(j) as *const __m256i),
                _mm256_loadu_si256(b.as_ptr().add(j) as *const __m256i),
            )
        };
        // |d| up to 12 bits: d * d + d' * d' fits 32 bits; summed in 64.
        let d = _mm256_or_si256(_mm256_subs_epu16(x, y), _mm256_subs_epu16(y, x));
        let sq = _mm256_madd_epi16(d, d);
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq)));
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq, 1)));
        j += 16;
    }
    let mut lanes = [0u64; 4];
    // SAFETY: four 64-bit lanes into four u64s.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc) };
    lanes.iter().sum::<u64>() + sse_row_scalar(&a[j..w], &b[j..w])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sse_row_neon(a: &[u16], b: &[u16]) -> u64 {
    use std::arch::aarch64::*;
    let w = a.len().min(b.len());
    let mut acc = vdupq_n_u64(0);
    let mut j = 0;
    while j + 8 <= w {
        // SAFETY: j + 8 <= w, the length of both rows.
        let (x, y) = unsafe { (vld1q_u16(a.as_ptr().add(j)), vld1q_u16(b.as_ptr().add(j))) };
        let d = vabdq_u16(x, y);
        let lo = vmull_u16(vget_low_u16(d), vget_low_u16(d));
        let hi = vmull_high_u16(d, d);
        acc = vpadalq_u32(acc, lo);
        acc = vpadalq_u32(acc, hi);
        j += 8;
    }
    vaddvq_u64(acc) + sse_row_scalar(&a[j..w], &b[j..w])
}

/// Sum of absolute 4x4 Hadamard-transformed differences between `a` and
/// `b` over `w` x `h` (multiples of 4), halved.
pub(crate) fn satd<B: Px>(
    a: &[u16],
    a_stride: usize,
    b: &[B],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if w >= 8 && crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        return unsafe { satd_avx2(a, a_stride, b, b_stride, w, h) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline.
        #[allow(clippy::needless_return)]
        return unsafe { satd_neon(a, a_stride, b, b_stride, w, h) };
    }
    #[allow(unreachable_code)]
    satd_scalar(a, a_stride, b, b_stride, w, h)
}

/// One 4x4 block's sum of absolute transformed differences.
#[inline(always)]
fn hadamard4x4(d: &mut [i32; 16]) -> u64 {
    for i in 0..4 {
        let (a0, a1, a2, a3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let (s0, s1, d0, d1) = (a0 + a1, a2 + a3, a0 - a1, a2 - a3);
        d[i * 4] = s0 + s1;
        d[i * 4 + 1] = s0 - s1;
        d[i * 4 + 2] = d0 + d1;
        d[i * 4 + 3] = d0 - d1;
    }
    let mut total = 0u64;
    for j in 0..4 {
        let (a0, a1, a2, a3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
        let (s0, s1, d0, d1) = (a0 + a1, a2 + a3, a0 - a1, a2 - a3);
        total += ((s0 + s1).abs() + (s0 - s1).abs() + (d0 + d1).abs() + (d0 - d1).abs()) as u64;
    }
    total
}

pub(crate) fn satd_scalar<B: Px>(
    a: &[u16],
    a_stride: usize,
    b: &[B],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for i in 0..4 {
                for j in 0..4 {
                    d[i * 4 + j] = a[(by + i) * a_stride + bx + j] as i32
                        - b[(by + i) * b_stride + bx + j].get();
                }
            }
            total += hadamard4x4(&mut d);
        }
    }
    total / 2
}

/// Eight `B`s as 32-bit lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load8<B: Px>(p: &[B]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    debug_assert!(p.len() >= 8);
    // SAFETY: the caller passes at least eight elements; the size of `B`
    // says which they are.
    unsafe {
        if std::mem::size_of::<B>() == 2 {
            _mm256_cvtepu16_epi32(_mm_loadu_si128(p.as_ptr() as *const __m128i))
        } else {
            _mm256_loadu_si256(p.as_ptr() as *const __m256i)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn satd_avx2<B: Px>(
    a: &[u16],
    a_stride: usize,
    b: &[B],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        let mut bx = 0;
        while bx + 8 <= w {
            // Two 4x4 blocks side by side, one per 128-bit lane.
            let mut r = [_mm256_setzero_si256(); 4];
            for (i, ri) in r.iter_mut().enumerate() {
                let ao = (by + i) * a_stride + bx;
                let bo = (by + i) * b_stride + bx;
                // SAFETY: rows hold `w` >= bx + 8 elements.
                *ri = unsafe { _mm256_sub_epi32(load8(&a[ao..ao + 8]), load8(&b[bo..bo + 8])) };
            }
            // Vertical butterflies (across the rows).
            let s0 = _mm256_add_epi32(r[0], r[1]);
            let s1 = _mm256_add_epi32(r[2], r[3]);
            let d0 = _mm256_sub_epi32(r[0], r[1]);
            let d1 = _mm256_sub_epi32(r[2], r[3]);
            let v0 = _mm256_add_epi32(s0, s1);
            let v1 = _mm256_sub_epi32(s0, s1);
            let v2 = _mm256_add_epi32(d0, d1);
            let v3 = _mm256_sub_epi32(d0, d1);
            // Transpose each lane's 4x4.
            let t0 = _mm256_unpacklo_epi32(v0, v1);
            let t1 = _mm256_unpackhi_epi32(v0, v1);
            let t2 = _mm256_unpacklo_epi32(v2, v3);
            let t3 = _mm256_unpackhi_epi32(v2, v3);
            let c0 = _mm256_unpacklo_epi64(t0, t2);
            let c1 = _mm256_unpackhi_epi64(t0, t2);
            let c2 = _mm256_unpacklo_epi64(t1, t3);
            let c3 = _mm256_unpackhi_epi64(t1, t3);
            // Horizontal butterflies, now across registers.
            let s0 = _mm256_add_epi32(c0, c1);
            let s1 = _mm256_add_epi32(c2, c3);
            let d0 = _mm256_sub_epi32(c0, c1);
            let d1 = _mm256_sub_epi32(c2, c3);
            let sum = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_abs_epi32(_mm256_add_epi32(s0, s1)),
                    _mm256_abs_epi32(_mm256_sub_epi32(s0, s1)),
                ),
                _mm256_add_epi32(
                    _mm256_abs_epi32(_mm256_add_epi32(d0, d1)),
                    _mm256_abs_epi32(_mm256_sub_epi32(d0, d1)),
                ),
            );
            // Widened to 64 bits so no block size can overflow.
            acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum)));
            acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum, 1)));
            bx += 8;
        }
        if bx < w {
            // A last 4-wide column.
            let mut d = [0i32; 16];
            for i in 0..4 {
                for j in 0..4 {
                    d[i * 4 + j] = a[(by + i) * a_stride + bx + j] as i32
                        - b[(by + i) * b_stride + bx + j].get();
                }
            }
            total += hadamard4x4(&mut d);
        }
    }
    let mut lanes = [0u64; 4];
    // SAFETY: four 64-bit lanes into four u64s.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc) };
    (total + lanes.iter().sum::<u64>()) / 2
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn satd_neon<B: Px>(
    a: &[u16],
    a_stride: usize,
    b: &[B],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    use std::arch::aarch64::*;
    let load4 = |p: &[B]| -> int32x4_t {
        debug_assert!(p.len() >= 4);
        // SAFETY: four elements; the size of `B` says which they are.
        unsafe {
            if std::mem::size_of::<B>() == 2 {
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p.as_ptr() as *const u16)))
            } else {
                vld1q_s32(p.as_ptr() as *const i32)
            }
        }
    };
    let mut acc = vdupq_n_u64(0);
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut r = [vdupq_n_s32(0); 4];
            for (i, ri) in r.iter_mut().enumerate() {
                let ao = (by + i) * a_stride + bx;
                let bo = (by + i) * b_stride + bx;
                let x = vreinterpretq_s32_u32(vmovl_u16(
                    // SAFETY: the row holds bx + 4 samples.
                    unsafe { vld1_u16(a[ao..ao + 4].as_ptr()) },
                ));
                *ri = vsubq_s32(x, load4(&b[bo..bo + 4]));
            }
            let s0 = vaddq_s32(r[0], r[1]);
            let s1 = vaddq_s32(r[2], r[3]);
            let d0 = vsubq_s32(r[0], r[1]);
            let d1 = vsubq_s32(r[2], r[3]);
            let v0 = vaddq_s32(s0, s1);
            let v1 = vsubq_s32(s0, s1);
            let v2 = vaddq_s32(d0, d1);
            let v3 = vsubq_s32(d0, d1);
            // Transpose the 4x4.
            let t01 = vtrnq_s32(v0, v1);
            let t23 = vtrnq_s32(v2, v3);
            let c0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
            let c1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
            let c2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
            let c3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
            let s0 = vaddq_s32(c0, c1);
            let s1 = vaddq_s32(c2, c3);
            let d0 = vsubq_s32(c0, c1);
            let d1 = vsubq_s32(c2, c3);
            let sum = vaddq_u32(
                vaddq_u32(
                    vreinterpretq_u32_s32(vabsq_s32(vaddq_s32(s0, s1))),
                    vreinterpretq_u32_s32(vabsq_s32(vsubq_s32(s0, s1))),
                ),
                vaddq_u32(
                    vreinterpretq_u32_s32(vabsq_s32(vaddq_s32(d0, d1))),
                    vreinterpretq_u32_s32(vabsq_s32(vsubq_s32(d0, d1))),
                ),
            );
            acc = vpadalq_u32(acc, sum);
        }
    }
    vaddvq_u64(acc) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32, max: u32) -> Vec<u16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                ((s >> 16) % (max + 1)) as u16
            })
            .collect()
    }

    /// Each SIMD kernel against its scalar version, 8- to 12-bit samples,
    /// every block shape.
    #[test]
    fn simd_matches_scalar() {
        for max in [255u32, 1023, 4095] {
            let a = noise(160 * 136, 1 + max, max);
            let b = noise(160 * 136, 7 + max, max);
            let bi: Vec<i32> = b.iter().map(|&v| v as i32 - 3).collect();
            for (w, h) in [
                (4, 4),
                (8, 8),
                (16, 8),
                (12, 4),
                (32, 16),
                (64, 64),
                (128, 128),
                (4, 16),
                (24, 8),
            ] {
                assert_eq!(
                    satd(&a, 160, &b, 136, w, h),
                    satd_scalar(&a, 160, &b, 136, w, h),
                    "satd u16 {w}x{h}"
                );
                assert_eq!(
                    satd(&a, 160, &bi, w, w, h),
                    satd_scalar(&a, 160, &bi, w, w, h),
                    "satd i32 {w}x{h}"
                );
                for limit in [u64::MAX, 1000, 20_000, 0] {
                    assert_eq!(
                        sad(&a, 160, &b, 136, w, h, limit),
                        sad_scalar(&a, 160, &b, 136, w, h, limit),
                        "sad {w}x{h}"
                    );
                }
                assert_eq!(
                    sse_row(&a[..w], &b[3..3 + w]),
                    sse_row_scalar(&a[..w], &b[3..3 + w])
                );
            }
            // The SAD of any rectangle, against limits it passes at
            // different rows.
            for (w, h) in [(4, 1), (8, 3), (12, 7), (20, 5), (6, 9), (128, 64)] {
                for limit in [u64::MAX, 0, 100, 1000, 20_000, 300_000] {
                    assert_eq!(
                        sad(&a, 160, &b, 136, w, h, limit),
                        sad_scalar(&a, 160, &b, 136, w, h, limit),
                        "sad {w}x{h} limit {limit}"
                    );
                }
            }
        }
    }
}
