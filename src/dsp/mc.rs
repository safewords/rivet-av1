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
    pub(crate) const fn new(f: &[i32; 8]) -> Self {
        // The first and the last tap that are not zero (the first one for
        // both where every tap is).
        let (mut lo, mut hi) = (0, 7);
        while lo < 7 && f[lo] == 0 {
            lo += 1;
        }
        while hi > lo && f[hi] == 0 {
            hi -= 1;
        }
        let len = (hi - lo + 2) & !1;
        let first = if lo < 8 - len { lo } else { 8 - len };
        let mut taps = [0i16; 8];
        let mut t = 0;
        while t < len {
            taps[t] = f[first + t] as i16;
            t += 1;
        }
        Taps { first, len, taps }
    }

    /// The taps kept of `Subpel_Filters[set][phase]`, from a table worked
    /// out when the crate is built: a block looks its two filters up.
    #[inline]
    pub(crate) fn of(set: usize, phase: usize) -> &'static Taps {
        &KEPT[set][phase]
    }

    /// The run itself.
    #[inline]
    pub(crate) fn taps(&self) -> &[i16] {
        &self.taps[..self.len]
    }
}

#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
static KEPT: [[Taps; 16]; 6] = {
    let mut kept = [[Taps::new(&[0; 8]); 16]; 6];
    let mut set = 0;
    while set < 6 {
        let mut phase = 0;
        while phase < 16 {
            kept[set][phase] = Taps::new(&crate::tables::SUBPEL_FILTERS[set][phase]);
            phase += 1;
        }
        set += 1;
    }
    kept
};

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

/// A 16-bit value a vertical filter reads: an intermediate, or a sample of
/// a reference (12 bits at most, so the same value in a signed lane).
pub(crate) trait Lane16: Copy {
    /// The value, as the signed 16 bits the SIMD lanes take it for.
    fn value(self) -> i32;
}

impl Lane16 for i16 {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i32
    }
}

impl Lane16 for u16 {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i16 as i32
    }
}

/// The vertical filter of a block: `out[r][c] = (sum_t taps[t] * src[(r +
/// t) * stride + c] + add) >> shift`, a value the caller knows to fit 16
/// bits (an intermediate's rounding shift, less a compound prediction's
/// bias: see [`Compound`]). `src` holds `h + taps.len() - 1` rows, the
/// first of them under the first tap; row `r` of `out` starts at `r * w`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_cols16<T: Lane16>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    taps: &[i16],
    add: i32,
    shift: u32,
    out: &mut [i16],
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 4 {
        match (taps.len(), w >= 8) {
            (2, true) => cols16_simd128::<T, 1, true>(src, stride, w, h, taps, add, shift, out),
            (4, true) => cols16_simd128::<T, 2, true>(src, stride, w, h, taps, add, shift, out),
            (6, true) => cols16_simd128::<T, 3, true>(src, stride, w, h, taps, add, shift, out),
            (_, true) => cols16_simd128::<T, 4, true>(src, stride, w, h, taps, add, shift, out),
            (2, false) => cols16_simd128::<T, 1, false>(src, stride, w, h, taps, add, shift, out),
            (4, false) => cols16_simd128::<T, 2, false>(src, stride, w, h, taps, add, shift, out),
            (6, false) => cols16_simd128::<T, 3, false>(src, stride, w, h, taps, add, shift, out),
            (_, false) => cols16_simd128::<T, 4, false>(src, stride, w, h, taps, add, shift, out),
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 4 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            match taps.len() {
                2 => cols16_neon::<T, 2>(src, stride, w, h, taps, add, shift, out),
                4 => cols16_neon::<T, 4>(src, stride, w, h, taps, add, shift, out),
                6 => cols16_neon::<T, 6>(src, stride, w, h, taps, add, shift, out),
                _ => cols16_neon::<T, 8>(src, stride, w, h, taps, add, shift, out),
            }
        }
        return;
    }
    match taps.len() {
        2 => cols16_scalar::<T, 2>(src, stride, w, h, taps, add, shift, out),
        4 => cols16_scalar::<T, 4>(src, stride, w, h, taps, add, shift, out),
        6 => cols16_scalar::<T, 6>(src, stride, w, h, taps, add, shift, out),
        _ => cols16_scalar::<T, 8>(src, stride, w, h, taps, add, shift, out),
    }
}

/// The scalar version of [`filter_cols16`].
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cols16_scalar<T: Lane16, const N: usize>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    taps: &[i16],
    add: i32,
    shift: u32,
    out: &mut [i16],
) {
    let taps: [i32; N] = std::array::from_fn(|t| taps[t] as i32);
    for r in 0..h {
        let rows: [&[T]; N] = std::array::from_fn(|t| &src[(r + t) * stride..][..w]);
        for (c, o) in out[r * w..][..w].iter_mut().enumerate() {
            let mut acc = add;
            for t in 0..N {
                acc += taps[t] * rows[t][c].value();
            }
            *o = (acc >> shift) as i16;
        }
    }
}

/// `Warped_Filters` in 16 bits, as [`dot8`] takes a filter.
pub(crate) static WARPED_FILTERS16: [[i16; 8]; 193] = {
    let mut filters = [[0i16; 8]; 193];
    let mut i = 0;
    while i < 193 {
        let mut t = 0;
        while t < 8 {
            filters[i][t] = crate::tables::WARPED_FILTERS[i][t] as i16;
            t += 1;
        }
        i += 1;
    }
    filters
};

/// The sum of the first eight of `v`, each times its tap: one output of
/// either pass of the block warp process (7.11.3.5), where the filter
/// changes from one output to the next and a filter over a row at a time
/// has nothing to share.
#[inline(always)]
pub(crate) fn dot8<T: Lane16>(v: &[T], taps: &[i16; 8]) -> i32 {
    let v = &v[..8];
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use std::arch::wasm32::*;
        const { assert!(size_of::<T>() == 2) };
        // SAFETY: eight 16-bit values each.
        let (a, f) = unsafe {
            (
                v128_load(v.as_ptr() as *const v128),
                v128_load(taps.as_ptr() as *const v128),
            )
        };
        let d = i32x4_dot_i16x8(a, f);
        let d = i32x4_add(d, i32x4_shuffle::<2, 3, 0, 1>(d, d));
        let d = i32x4_add(d, i32x4_shuffle::<1, 0, 3, 2>(d, d));
        #[allow(clippy::needless_return)]
        return i32x4_extract_lane::<0>(d);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline.
        #[allow(clippy::needless_return)]
        return unsafe { dot8_neon(v, taps) };
    }
    #[allow(unreachable_code)]
    dot8_scalar(v, taps)
}

/// The reference version of [`dot8`].
pub(crate) fn dot8_scalar<T: Lane16>(v: &[T], taps: &[i16; 8]) -> i32 {
    let mut sum = 0;
    for (&v, &t) in v[..8].iter().zip(taps) {
        sum += v.value() * t as i32;
    }
    sum
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
/// references, a block of them: `(a + b + 1) >> 1`. (Each is the sample
/// times `1 << InterPostRound`, and `Round2(a + b, 1 + InterPostRound)` of
/// those is this; it needs no clip.) Row `r` of each starts at `r` times
/// its stride.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn avg_samples(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 8 {
        avg_samples_simd128(a, a_stride, b, b_stride, w, h, dst, dst_stride);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 8 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { avg_samples_neon(a, a_stride, b, b_stride, w, h, dst, dst_stride) };
        return;
    }
    avg_samples_scalar(a, a_stride, b, b_stride, w, h, dst, dst_stride)
}

/// The average of a compound block's two predictions (7.11.3.1), as
/// samples: `Clip1(Round2(a + b, 1 + InterPostRound))`. Row `r` of each
/// prediction starts at `r * w`, row `r` of `dst` at `r * dst_stride`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn compound_avg(
    a: &[i16],
    b: &[i16],
    c: Compound,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if w >= 8 {
        compound_avg_simd128(a, b, c, w, h, dst, dst_stride);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if w >= 8 {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { compound_avg_neon(a, b, c, w, h, dst, dst_stride) };
        return;
    }
    compound_avg_scalar(a, b, c, w, h, dst, dst_stride)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn avg_samples_scalar(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    for r in 0..h {
        let (a, b) = (&a[r * a_stride..][..w], &b[r * b_stride..][..w]);
        for ((o, &a), &b) in dst[r * dst_stride..][..w].iter_mut().zip(a).zip(b) {
            *o = ((a as u32 + b as u32 + 1) >> 1) as u16;
        }
    }
}

pub(crate) fn compound_avg_scalar(
    a: &[i16],
    b: &[i16],
    c: Compound,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let shift = 1 + c.post;
    let add = (1 << c.post) + 2 * c.bias;
    for r in 0..h {
        let (a, b) = (&a[r * w..][..w], &b[r * w..][..w]);
        for ((o, &a), &b) in dst[r * dst_stride..][..w].iter_mut().zip(a).zip(b) {
            *o = ((a as i32 + b as i32 + add) >> shift).clamp(0, c.maxv) as u16;
        }
    }
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

/// [`filter_cols16`] a strip of eight columns at a time (`WIDE`; four for a
/// block under eight wide), down the block: rows two by two (`P` pairs of
/// taps), a pair of rows interleaved so that one 16-bit dot product gives
/// four columns' sums. Row `r` takes the pairs of rows `r + 2k` and `r +
/// 2k + 1`, which row `r + 2` takes again one place up: the interleaved
/// pairs are kept, in two sets (even and odd rows), and each row of the
/// block loads and interleaves one row more.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
fn cols16_simd128<T: Lane16, const P: usize, const WIDE: bool>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    taps: &[i16],
    add: i32,
    shift: u32,
    out: &mut [i16],
) {
    use std::arch::wasm32::*;
    const { assert!(size_of::<T>() == 2) };
    let lanes = if WIDE { 8 } else { 4 };
    assert!(taps.len() == 2 * P && w >= lanes && h > 0);
    assert!(src.len() >= (h + 2 * P - 2) * stride + w && out.len() >= h * w);
    let coef: [v128; P] = std::array::from_fn(|k| {
        let (a, b) = (taps[2 * k], taps[2 * k + 1]);
        i16x8(a, b, a, b, a, b, a, b)
    });
    let add = i32x4_splat(add);
    // Rows `a` and `b` interleaved: the low four columns and the high four.
    let pair = |a: v128, b: v128| {
        (
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b),
        )
    };
    let zero = i16x8_splat(0);
    // Whole strips, then one more ending at the last column (overlapping
    // what is done: the same values are stored again).
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - lanes);
        // SAFETY: column c0 of the first row of each, c0 + lanes <= w.
        let (sp, op) = unsafe { (src.as_ptr().add(c0), out.as_mut_ptr().add(c0)) };
        // SAFETY (every call): `lanes` values of row i <= h + 2P - 2 from
        // column c0, which the assertion puts inside `src`.
        let row = |i: usize| unsafe {
            if WIDE {
                v128_load(sp.add(i * stride) as *const v128)
            } else {
                v128_load64_zero(sp.add(i * stride) as *const u64)
            }
        };
        let emit = |r: usize, pairs: &[(v128, v128); P]| {
            let (mut lo, mut hi) = (add, add);
            for k in 0..P {
                lo = i32x4_add(lo, i32x4_dot_i16x8(pairs[k].0, coef[k]));
                if WIDE {
                    hi = i32x4_add(hi, i32x4_dot_i16x8(pairs[k].1, coef[k]));
                }
            }
            // The results fit 16 bits: narrowing changes nothing.
            let v = i16x8_narrow_i32x4(i32x4_shr(lo, shift), i32x4_shr(hi, shift));
            // SAFETY: `lanes` values of row r < h of `out` from column c0.
            unsafe {
                if WIDE {
                    v128_store(op.add(r * w) as *mut v128, v);
                } else {
                    v128_store64_lane::<0>(v, op.add(r * w) as *mut u64);
                }
            }
        };
        // `even[k]` is rows r + 2k and r + 2k + 1 for the even row r being
        // filtered, `odd[k]` the same for the odd row after it; `last` is
        // the newest row loaded.
        let mut even = [(zero, zero); P];
        let mut odd = [(zero, zero); P];
        let mut last = row(0);
        for i in 1..2 * P - 1 {
            let next = row(i);
            let set = if i % 2 == 1 { &mut even } else { &mut odd };
            set[(i - 1) / 2] = pair(last, next);
            last = next;
        }
        let mut r = 0;
        while r < h {
            let next = row(r + 2 * P - 1);
            even[P - 1] = pair(last, next);
            last = next;
            emit(r, &even);
            for k in 0..P - 1 {
                even[k] = even[k + 1];
            }
            r += 1;
            if r == h {
                break;
            }
            let next = row(r + 2 * P - 1);
            odd[P - 1] = pair(last, next);
            last = next;
            emit(r, &odd);
            for k in 0..P - 1 {
                odd[k] = odd[k + 1];
            }
            r += 1;
        }
        c = c0 + lanes;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
fn avg_samples_simd128(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    use std::arch::wasm32::*;
    assert!(w >= 8 && h > 0);
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    for r in 0..h {
        // SAFETY: row r of each, which the assertions put inside.
        let (ap, bp, dp) = unsafe {
            (
                a.as_ptr().add(r * a_stride),
                b.as_ptr().add(r * b_stride),
                dst.as_mut_ptr().add(r * dst_stride),
            )
        };
        // Whole vectors, then one more ending at the last column
        // (overlapping what is done: the same values are stored again).
        let mut c = 0;
        while c < w {
            let c0 = c.min(w - 8);
            // SAFETY: c0 + 8 <= w, and every row holds w samples.
            unsafe {
                let va = v128_load(ap.add(c0) as *const v128);
                let vb = v128_load(bp.add(c0) as *const v128);
                v128_store(dp.add(c0) as *mut v128, u16x8_avgr(va, vb));
            }
            c = c0 + 8;
        }
    }
}

/// Eight samples at a time, in 16-bit lanes with saturation doing the
/// clip: the sum saturates only where the result is past the largest
/// sample anyway, `max` with `-add` and a saturating `+ add` leave 0..32767,
/// and that shifted down is at most the largest sample at 10 and 12 bits
/// (at 8 it is clipped).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn compound_avg_simd128(
    a: &[i16],
    b: &[i16],
    c: Compound,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    use std::arch::wasm32::*;
    assert!(w >= 8 && h > 0 && a.len() >= h * w && b.len() >= h * w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    let shift = 1 + c.post;
    let neg_add = i16x8_splat(-((1 << c.post) + 2 * c.bias) as i16);
    let maxv = i16x8_splat(c.maxv as i16);
    for r in 0..h {
        // SAFETY: row r of each, which the assertions put inside.
        let (ap, bp, dp) = unsafe {
            (
                a.as_ptr().add(r * w),
                b.as_ptr().add(r * w),
                dst.as_mut_ptr().add(r * dst_stride),
            )
        };
        let mut col = 0;
        while col < w {
            let c0 = col.min(w - 8);
            // SAFETY: c0 + 8 <= w, and every row holds w values.
            let (va, vb) = unsafe {
                (
                    v128_load(ap.add(c0) as *const v128),
                    v128_load(bp.add(c0) as *const v128),
                )
            };
            let s = i16x8_max(i16x8_add_sat(va, vb), neg_add);
            let v = i16x8_min(i16x8_shr(i16x8_sub_sat(s, neg_add), shift), maxv);
            // SAFETY: c0 + 8 <= w.
            unsafe { v128_store(dp.add(c0) as *mut v128, v) };
            col = c0 + 8;
        }
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
#[inline]
unsafe fn dot8_neon<T: Lane16>(v: &[T], taps: &[i16; 8]) -> i32 {
    use std::arch::aarch64::*;
    const { assert!(size_of::<T>() == 2) };
    assert!(v.len() >= 8);
    // SAFETY: eight 16-bit values each.
    let (a, f) = unsafe {
        (
            vld1q_s16(v.as_ptr() as *const i16),
            vld1q_s16(taps.as_ptr()),
        )
    };
    let lo = vmull_s16(vget_low_s16(a), vget_low_s16(f));
    vaddvq_s32(vmlal_high_s16(lo, a, f))
}

/// [`filter_cols16`] a strip of eight columns at a time (four for a block
/// under eight wide), down the block: the rows under the taps stay in
/// registers, each row of the block loads one more, and every tap is a
/// widening multiply-accumulate into 32-bit lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn cols16_neon<T: Lane16, const N: usize>(
    src: &[T],
    stride: usize,
    w: usize,
    h: usize,
    taps: &[i16],
    add: i32,
    shift: u32,
    out: &mut [i16],
) {
    use std::arch::aarch64::*;
    const { assert!(size_of::<T>() == 2) };
    assert!(taps.len() == N && w >= 4 && h > 0);
    assert!(src.len() >= (h + N - 2) * stride + w && out.len() >= h * w);
    let down = vdupq_n_s32(-(shift as i32));
    // Whole strips, then one more ending at the last column (overlapping
    // what is done: the same values are stored again).
    let mut c = 0;
    if w >= 8 {
        while c < w {
            let c0 = c.min(w - 8);
            // SAFETY: column c0 of the first row of each, c0 + 8 <= w.
            let (sp, op) =
                unsafe { (src.as_ptr().add(c0) as *const i16, out.as_mut_ptr().add(c0)) };
            // SAFETY (every call): eight values of row i <= h + N - 2 from
            // column c0, which the assertion puts inside `src`.
            let row = |i: usize| unsafe { vld1q_s16(sp.add(i * stride)) };
            let mut under = [vdupq_n_s16(0); N];
            for (i, v) in under[..N - 1].iter_mut().enumerate() {
                *v = row(i);
            }
            for r in 0..h {
                under[N - 1] = row(r + N - 1);
                let (mut lo, mut hi) = (vdupq_n_s32(add), vdupq_n_s32(add));
                for t in 0..N {
                    lo = vmlal_n_s16(lo, vget_low_s16(under[t]), taps[t]);
                    hi = vmlal_high_n_s16(hi, under[t], taps[t]);
                }
                // The results fit 16 bits: narrowing changes nothing.
                let v = vcombine_s16(
                    vmovn_s32(vshlq_s32(lo, down)),
                    vmovn_s32(vshlq_s32(hi, down)),
                );
                // SAFETY: eight values of row r < h of `out` from column c0.
                unsafe { vst1q_s16(op.add(r * w), v) };
                for t in 0..N - 1 {
                    under[t] = under[t + 1];
                }
            }
            c = c0 + 8;
        }
    } else {
        while c < w {
            let c0 = c.min(w - 4);
            // SAFETY: column c0 of the first row of each, c0 + 4 <= w.
            let (sp, op) =
                unsafe { (src.as_ptr().add(c0) as *const i16, out.as_mut_ptr().add(c0)) };
            // SAFETY (every call): four values of row i <= h + N - 2 from
            // column c0, which the assertion puts inside `src`.
            let row = |i: usize| unsafe { vld1_s16(sp.add(i * stride)) };
            let mut under = [vdup_n_s16(0); N];
            for (i, v) in under[..N - 1].iter_mut().enumerate() {
                *v = row(i);
            }
            for r in 0..h {
                under[N - 1] = row(r + N - 1);
                let mut lo = vdupq_n_s32(add);
                for t in 0..N {
                    lo = vmlal_n_s16(lo, under[t], taps[t]);
                }
                // SAFETY: four values of row r < h of `out` from column c0.
                unsafe { vst1_s16(op.add(r * w), vmovn_s32(vshlq_s32(lo, down))) };
                for t in 0..N - 1 {
                    under[t] = under[t + 1];
                }
            }
            c = c0 + 4;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn avg_samples_neon(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    assert!(w >= 8 && h > 0);
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    for r in 0..h {
        // SAFETY: row r of each, which the assertions put inside.
        let (ap, bp, dp) = unsafe {
            (
                a.as_ptr().add(r * a_stride),
                b.as_ptr().add(r * b_stride),
                dst.as_mut_ptr().add(r * dst_stride),
            )
        };
        // Whole vectors, then one more ending at the last column
        // (overlapping what is done: the same values are stored again).
        let mut c = 0;
        while c < w {
            let c0 = c.min(w - 8);
            // SAFETY: c0 + 8 <= w, and every row holds w samples.
            unsafe {
                let v = vrhaddq_u16(vld1q_u16(ap.add(c0)), vld1q_u16(bp.add(c0)));
                vst1q_u16(dp.add(c0), v);
            }
            c = c0 + 8;
        }
    }
}

/// The SIMD128 version's arithmetic: 16-bit lanes, saturation as the clip.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compound_avg_neon(
    a: &[i16],
    b: &[i16],
    c: Compound,
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    use std::arch::aarch64::*;
    assert!(w >= 8 && h > 0 && a.len() >= h * w && b.len() >= h * w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    let shift = vdupq_n_s16(-(1 + c.post as i16));
    let neg_add = vdupq_n_s16(-((1 << c.post) + 2 * c.bias) as i16);
    let maxv = vdupq_n_s16(c.maxv as i16);
    for r in 0..h {
        // SAFETY: row r of each, which the assertions put inside.
        let (ap, bp, dp) = unsafe {
            (
                a.as_ptr().add(r * w),
                b.as_ptr().add(r * w),
                dst.as_mut_ptr().add(r * dst_stride),
            )
        };
        let mut col = 0;
        while col < w {
            let c0 = col.min(w - 8);
            // SAFETY: c0 + 8 <= w, and every row holds w values.
            let (va, vb) = unsafe { (vld1q_s16(ap.add(c0)), vld1q_s16(bp.add(c0))) };
            let s = vmaxq_s16(vqaddq_s16(va, vb), neg_add);
            let v = vminq_s16(vshlq_s16(vqsubq_s16(s, neg_add), shift), maxv);
            // SAFETY: c0 + 8 <= w.
            unsafe { vst1q_u16(dp.add(c0), vreinterpretq_u16_s16(v)) };
            col = c0 + 8;
        }
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

    /// The 32-bit filters' SIMD versions agree with the scalar ones for
    /// every filter and phase, at several widths, on 8-, 10- and 12-bit
    /// samples.
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
    /// extremes together included, at every depth and width, on blocks of
    /// one to three rows whose strides differ.
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
            for w in [2usize, 4, 8, 12, 16, 32, 64, 128] {
                for round in 0..40 {
                    let h = 1 + round % 3;
                    let ds = w + rnd(5) as usize;
                    // Predictions as the specification has them, and as
                    // they are stored.
                    let mut pick = || match (round, rnd(8)) {
                        (0, _) | (_, 0) => hi,
                        (1, _) | (_, 1) => lo,
                        _ => lo + rnd((hi - lo + 1) as u64) as i32,
                    };
                    let p0: Vec<i32> = (0..w * h).map(|_| pick()).collect();
                    let p1: Vec<i32> = (0..w * h).map(|_| pick()).collect();
                    let a: Vec<i16> = p0.iter().map(|&v| (v - c.bias) as i16).collect();
                    let b: Vec<i16> = p1.iter().map(|&v| (v - c.bias) as i16).collect();
                    // `f(row, column)` clipped, where the block's samples
                    // go in a destination of stride `ds`; 7 elsewhere.
                    let spec = |f: &dyn Fn(usize, usize) -> i32| -> Vec<u16> {
                        let mut v = vec![7u16; h * ds];
                        for r in 0..h {
                            for j in 0..w {
                                v[r * ds + j] = f(r, j).clamp(0, c.maxv) as u16;
                            }
                        }
                        v
                    };
                    let mut out = vec![7u16; h * ds];
                    let mut out_scalar = vec![7u16; h * ds];

                    let want =
                        spec(&|r, j| (p0[r * w + j] + p1[r * w + j] + (1 << post)) >> (1 + post));
                    compound_avg_scalar(&a, &b, c, w, h, &mut out_scalar, ds);
                    compound_avg(&a, &b, c, w, h, &mut out, ds);
                    assert_eq!(want, out_scalar, "avg scalar {bd}-bit {w}x{h}");
                    assert_eq!(want, out, "avg {bd}-bit {w}x{h}");

                    // Whole samples: the predictions are the samples times
                    // `1 << post`.
                    let (s0s, s1s) = (w + rnd(5) as usize, w + rnd(5) as usize);
                    let s0: Vec<u16> = (0..h * s0s).map(|_| rnd(1 << bd) as u16).collect();
                    let s1: Vec<u16> = (0..h * s1s).map(|_| rnd(1 << bd) as u16).collect();
                    let want = spec(&|r, j| {
                        let p0 = (s0[r * s0s + j] as i32) << post;
                        let p1 = (s1[r * s1s + j] as i32) << post;
                        (p0 + p1 + (1 << post)) >> (1 + post)
                    });
                    out.fill(7);
                    out_scalar.fill(7);
                    avg_samples_scalar(&s0, s0s, &s1, s1s, w, h, &mut out_scalar, ds);
                    avg_samples(&s0, s0s, &s1, s1s, w, h, &mut out, ds);
                    assert_eq!(want, out_scalar, "whole samples scalar {bd}-bit {w}x{h}");
                    assert_eq!(want, out, "whole samples {bd}-bit {w}x{h}");
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
        for (set, filt) in SUBPEL_FILTERS.iter().enumerate() {
            for (phase, f) in filt.iter().enumerate() {
                let t = Taps::new(f);
                assert!(matches!(t.len, 2 | 4 | 6 | 8) && t.first + t.len <= 8);
                let mut back = [0i32; 8];
                for (k, &v) in t.taps().iter().enumerate() {
                    back[t.first + k] = v as i32;
                }
                assert_eq!(&back, f);
                // And the table has it.
                let of = Taps::of(set, phase);
                assert_eq!((of.first, of.taps()), (t.first, t.taps()), "{f:?}");
            }
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

    /// The vertical filter of a block, every version of it, against the
    /// specification's arithmetic (`Round2` of the 32-bit sum of all eight
    /// taps over 32-bit intermediates): every filter and phase, blocks of
    /// every width and several heights, a stride wider than the block,
    /// intermediates over the whole range the horizontal filter gives at 8,
    /// 10 and 12 bits with its extremes among them, as a single prediction
    /// (`InterRound1` of 11 or 9) and as a compound one less its bias.
    #[test]
    fn block_columns_match_the_specification() {
        let mut seed = 0x0f1e_2d3c_4b5a_6978u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10, 12] {
            let round0 = if bd == 12 { 5 } else { 3 };
            let max = (1i32 << bd) - 1;
            // The horizontal filter's range over every filter.
            let (mut lo, mut hi) = (0, 0);
            for f in SUBPEL_FILTERS.iter().flatten() {
                let pos: i32 = f.iter().filter(|&&t| t > 0).sum();
                let neg: i32 = f.iter().filter(|&&t| t < 0).sum();
                hi = hi.max((max * pos + ((1 << round0) >> 1)) >> round0);
                lo = lo.min((max * neg + ((1 << round0) >> 1)) >> round0);
            }
            for (set, filt) in SUBPEL_FILTERS.iter().enumerate() {
                for (phase, f) in filt.iter().enumerate() {
                    for w in [2usize, 4, 8, 16, 32, 64, 128] {
                        let h = [1usize, 2, 3, 4, 7, 8, 16][rnd(7) as usize];
                        let stride = w + rnd(9) as usize;
                        let kind = rnd(4);
                        let inter: Vec<i16> = (0..(h + 7) * stride)
                            .map(|_| match (kind, rnd(4)) {
                                (0, _) | (1, 0) => hi as i16,
                                (1, 1) | (2, _) => lo as i16,
                                _ => (lo + rnd((hi - lo + 1) as u64) as i32) as i16,
                            })
                            .collect();
                        let taps = Taps::new(f);
                        // The kernels read from the first row kept.
                        let from = &inter[taps.first * stride..];
                        let single = if bd == 12 { 9 } else { 11 };
                        for (round1, bias) in [(single, 0), (7u32, Compound::new(bd, 0).bias)] {
                            let mut want = vec![0i16; h * w];
                            let mut wide = vec![0i32; 8 * w];
                            let mut line = vec![0i32; w];
                            for r in 0..h {
                                for t in 0..8 {
                                    for c in 0..w {
                                        wide[t * w + c] = inter[(r + t) * stride + c] as i32;
                                    }
                                }
                                filter_col_scalar(&wide, 0, w, f, round1, &mut line);
                                for (o, &v) in want[r * w..][..w].iter_mut().zip(&line) {
                                    assert!(
                                        (i16::MIN as i32..=i16::MAX as i32).contains(&(v - bias))
                                    );
                                    *o = (v - bias) as i16;
                                }
                            }
                            let what =
                                format!("{bd}-bit set {set} phase {phase} {w}x{h} round1 {round1}");
                            let add = ((1 << round1) >> 1) - (bias << round1);
                            let t = taps.taps();
                            let mut got = vec![0i16; h * w];
                            filter_cols16(from, stride, w, h, t, add, round1, &mut got);
                            assert_eq!(want, got, "{what}");
                            got.fill(0);
                            match t.len() {
                                2 => cols16_scalar::<i16, 2>(
                                    from, stride, w, h, t, add, round1, &mut got,
                                ),
                                4 => cols16_scalar::<i16, 4>(
                                    from, stride, w, h, t, add, round1, &mut got,
                                ),
                                6 => cols16_scalar::<i16, 6>(
                                    from, stride, w, h, t, add, round1, &mut got,
                                ),
                                _ => cols16_scalar::<i16, 8>(
                                    from, stride, w, h, t, add, round1, &mut got,
                                ),
                            }
                            assert_eq!(want, got, "{what}, scalar");
                        }
                    }
                }
            }
        }
    }

    /// A vertical filter with no horizontal one, taken from the
    /// reference's samples themselves, against the specification's two
    /// passes (the horizontal one with the filter of phase 0): both ways of
    /// doing it (the 16-bit sums of 8-bit samples, and the block's vertical
    /// filter over samples), for a single prediction and a compound one.
    #[test]
    fn columns_of_samples_match_the_specification() {
        let mut seed = 0x6a09_e667_f3bc_c908u64;
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
                        let h = [1usize, 2, 3, 4, 7, 8, 16][rnd(7) as usize];
                        let stride = w + rnd(9) as usize;
                        let kind = rnd(4);
                        let src: Vec<u16> = (0..(h + 7) * stride)
                            .map(|_| match (kind, rnd(4)) {
                                (0, _) | (1, 0) => max as u16,
                                (1, 1) => 0,
                                _ => rnd(max + 1) as u16,
                            })
                            .collect();
                        let taps = Taps::new(f);
                        let from = &src[taps.first * stride..];
                        let single = if bd == 12 { 9 } else { 11 };
                        for (round1, bias) in [(single, 0), (7u32, Compound::new(bd, 0).bias)] {
                            // The specification: Round2(128 * s, round0)
                            // for the intermediates, then the vertical
                            // filter.
                            let mut want = vec![0i16; h * w];
                            let mut wide = vec![0i32; 8 * w];
                            let mut line = vec![0i32; w];
                            let mut padded = vec![0u16; w + 7];
                            for r in 0..h {
                                for t in 0..8 {
                                    padded[3..3 + w].copy_from_slice(&src[(r + t) * stride..][..w]);
                                    filter_row_scalar(
                                        &padded,
                                        &SUBPEL_FILTERS[set][0],
                                        round0,
                                        &mut wide[t * w..][..w],
                                    );
                                }
                                filter_col_scalar(&wide, 0, w, f, round1, &mut line);
                                for (o, &v) in want[r * w..][..w].iter_mut().zip(&line) {
                                    *o = (v - bias) as i16;
                                }
                            }
                            let what =
                                format!("{bd}-bit set {set} phase {phase} {w}x{h} round1 {round1}");
                            // The intermediate is the sample shifted up by
                            // 7 - round0, so the shift is that much less.
                            let shift = round1 - (7 - round0);
                            let add = ((1 << shift) >> 1) - (bias << shift);
                            let t = taps.taps();
                            let mut got = vec![0i16; h * w];
                            filter_cols16(from, stride, w, h, t, add, shift, &mut got);
                            assert_eq!(want, got, "{what}");
                            if bd == 8 {
                                assert_eq!(bias, 0);
                                got.fill(0);
                                taps8(from, stride, stride, w, h, t, shift, &mut got);
                                assert_eq!(want, got, "{what}, 16-bit sums");
                                got.fill(0);
                                match t.len() {
                                    2 => taps8_scalar::<2>(
                                        from, stride, stride, w, h, t, shift, &mut got,
                                    ),
                                    4 => taps8_scalar::<4>(
                                        from, stride, stride, w, h, t, shift, &mut got,
                                    ),
                                    6 => taps8_scalar::<6>(
                                        from, stride, stride, w, h, t, shift, &mut got,
                                    ),
                                    _ => taps8_scalar::<8>(
                                        from, stride, stride, w, h, t, shift, &mut got,
                                    ),
                                }
                                assert_eq!(want, got, "{what}, scalar 16-bit sums");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The warp's intermediates fit 16 bits at every depth: the extremes
    /// are all-maximum samples under a filter's positive taps or under its
    /// negative ones.
    #[test]
    fn warp_intermediates_fit_sixteen_bits() {
        use crate::tables::WARPED_FILTERS;
        for (bd, round0) in [(8u32, 3u32), (10, 3), (12, 5)] {
            let max = (1i32 << bd) - 1;
            for f in WARPED_FILTERS.iter() {
                let pos: i32 = f.iter().filter(|&&t| t > 0).sum();
                let neg: i32 = f.iter().filter(|&&t| t < 0).sum();
                let add = (1 << round0) >> 1;
                assert!((max * pos + add) >> round0 <= i16::MAX as i32, "{bd} {f:?}");
                assert!((max * neg + add) >> round0 >= i16::MIN as i32, "{bd} {f:?}");
            }
        }
    }

    /// One output of the warp filters, every version, against the 32-bit
    /// sum over the specification's table: every filter, over samples of
    /// each depth and over intermediates of the whole 16-bit range.
    #[test]
    fn warp_dot_matches_the_specification() {
        use crate::tables::WARPED_FILTERS;
        let mut seed = 0xbb67_ae85_84ca_a73bu64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for (i, f) in WARPED_FILTERS.iter().enumerate() {
            assert_eq!(f.map(|t| t as i16), WARPED_FILTERS16[i]);
            for round in 0..40 {
                let bd = [8u64, 10, 12][round % 3];
                let samples: [u16; 9] = std::array::from_fn(|_| match rnd(6) {
                    0 => (1 << bd) - 1,
                    1 => 0,
                    _ => rnd(1 << bd) as u16,
                });
                let want: i32 = (0..8).map(|t| f[t] * samples[t] as i32).sum();
                assert_eq!(
                    dot8(&samples, &WARPED_FILTERS16[i]),
                    want,
                    "{f:?} {samples:?}"
                );
                assert_eq!(dot8_scalar(&samples, &WARPED_FILTERS16[i]), want);
                let inter: [i16; 9] = std::array::from_fn(|_| match rnd(6) {
                    0 => i16::MAX,
                    1 => i16::MIN,
                    _ => rnd(1 << 16) as u16 as i16,
                });
                let want: i32 = (0..8).map(|t| f[t] * inter[t] as i32).sum();
                assert_eq!(dot8(&inter, &WARPED_FILTERS16[i]), want, "{f:?} {inter:?}");
                assert_eq!(dot8_scalar(&inter, &WARPED_FILTERS16[i]), want);
            }
        }
    }
}
