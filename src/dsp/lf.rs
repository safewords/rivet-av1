//! The loop filter's sample filtering (7.14.6), bit-exact with the
//! one-sample process (tested).
//!
//! `filter_pair` filters eight samples along an edge, two units' worth,
//! where they lie in the plane: 16-bit lanes in WebAssembly SIMD128 and
//! NEON, one kernel for each filter length and edge direction, every lane's
//! mask first and a filter computed only if a lane takes it.
//!
//! `filter_unit` is that for a wasm module without `simd128`: the
//! one-sample process on the plane, each sample reading what its mask and
//! its filter ask for and no more.
//!
//! `filter` takes the four samples along one unit's edge, gathered across
//! the edge into an array: the decisions and every filter computed for each
//! lane without branches, then selected per lane, written lane by lane in
//! 32-bit arithmetic that the compiler turns into vector code (AVX2 where
//! the kernel is compiled for it). The other targets' path.

/// The samples across an edge, four lanes: `v[7 + k]` is the k-th from the
/// edge (q0 at 7, p0 at 6), `v[k][lane]`.
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
pub(crate) type Edge = [[i32; 4]; 14];

/// The edge's filter parameters (7.14.4, 7.14.3).
#[cfg(any(
    test,
    not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub(crate) limit: i32,
    pub(crate) blimit: i32,
    pub(crate) thresh: i32,
    pub(crate) filter_size: usize,
    pub(crate) chroma: bool,
    pub(crate) bit_depth: u32,
}

/// Filters the four lanes of `v` in place.
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
#[inline]
pub(crate) fn filter(v: &mut Edge, p: &Params) {
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        unsafe { filter_avx2(v, p) };
        return;
    }
    filter_lanes(v, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_avx2(v: &mut Edge, p: &Params) {
    filter_lanes(v, p)
}

#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
#[inline(always)]
fn abs4(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    std::array::from_fn(|l| (a[l] - b[l]).abs())
}

#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
#[inline(always)]
fn gt(a: [i32; 4], t: i32) -> [bool; 4] {
    std::array::from_fn(|l| a[l] > t)
}

#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
#[inline(always)]
fn or(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
    std::array::from_fn(|l| a[l] | b[l])
}

/// The lane-by-lane process.
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
#[inline(always)]
pub(crate) fn filter_lanes(v: &mut Edge, p: &Params) {
    let sh = p.bit_depth - 8;
    let (q0, q1, q2, q3) = (v[7], v[8], v[9], v[10]);
    let (p0, p1, p2, p3) = (v[6], v[5], v[4], v[3]);
    let thresh_bd = p.thresh << sh;
    let limit_bd = p.limit << sh;
    let blimit_bd = p.blimit << sh;
    let filter_len = if p.filter_size == 4 {
        4
    } else if p.chroma {
        6
    } else if p.filter_size == 8 {
        8
    } else {
        16
    };
    // The filter mask process (7.14.6.2).
    let hev = or(gt(abs4(p1, p0), thresh_bd), gt(abs4(q1, q0), thresh_bd));
    let a = abs4(p0, q0);
    let b = abs4(p1, q1);
    let mut skip: [bool; 4] = std::array::from_fn(|l| a[l] * 2 + b[l] / 2 > blimit_bd);
    skip = or(
        skip,
        or(gt(abs4(p1, p0), limit_bd), gt(abs4(q1, q0), limit_bd)),
    );
    if filter_len >= 6 {
        skip = or(
            skip,
            or(gt(abs4(p2, p1), limit_bd), gt(abs4(q2, q1), limit_bd)),
        );
    }
    if filter_len >= 8 {
        skip = or(
            skip,
            or(gt(abs4(p3, p2), limit_bd), gt(abs4(q3, q2), limit_bd)),
        );
    }
    if skip == [true; 4] {
        return;
    }
    let t = 1 << sh;
    let mut flat = [false; 4];
    if p.filter_size >= 8 {
        let mut m = or(
            or(gt(abs4(p1, p0), t), gt(abs4(q1, q0), t)),
            or(gt(abs4(p2, p0), t), gt(abs4(q2, q0), t)),
        );
        if filter_len >= 8 {
            m = or(m, or(gt(abs4(p3, p0), t), gt(abs4(q3, q0), t)));
        }
        flat = std::array::from_fn(|l| !m[l]);
    }
    let mut flat2 = [false; 4];
    if p.filter_size >= 16 {
        let (q4, q5, q6) = (v[11], v[12], v[13]);
        let (p4, p5, p6) = (v[2], v[1], v[0]);
        let m = or(
            or(
                or(gt(abs4(p6, p0), t), gt(abs4(q6, q0), t)),
                or(gt(abs4(p5, p0), t), gt(abs4(q5, q0), t)),
            ),
            or(gt(abs4(p4, p0), t), gt(abs4(q4, q0), t)),
        );
        flat2 = std::array::from_fn(|l| !m[l]);
    }
    // The narrow filter (7.14.6.3), every lane.
    let bd = p.bit_depth;
    let lo = -(1 << (bd - 1));
    let hi = (1 << (bd - 1)) - 1;
    let c = |x: i32| x.clamp(lo, hi);
    let off = 0x80 << sh;
    let mut n_out = [[0i32; 4]; 4]; // p1, p0, q0, q1
    for l in 0..4 {
        let ps1 = p1[l] - off;
        let ps0 = p0[l] - off;
        let qs0 = q0[l] - off;
        let qs1 = q1[l] - off;
        let mut fl = if hev[l] { c(ps1 - qs1) } else { 0 };
        fl = c(fl + 3 * (qs0 - ps0));
        let filter1 = c(fl + 4) >> 3;
        let filter2 = c(fl + 3) >> 3;
        n_out[2][l] = c(qs0 - filter1) + off;
        n_out[1][l] = c(ps0 + filter2) + off;
        if hev[l] {
            n_out[0][l] = p1[l];
            n_out[3][l] = q1[l];
        } else {
            let f = (filter1 + 1) >> 1;
            n_out[3][l] = c(qs1 - f) + off;
            n_out[0][l] = c(ps1 + f) + off;
        }
    }
    // The wide filters (7.14.6.4): log2 size 3 (n = 3 luma, 2 chroma) and 4
    // (n = 6), every lane, for the sizes this edge can use.
    let wide = |log2_size: u32, v: &Edge| -> [[i32; 4]; 12] {
        let n: isize = if log2_size == 4 {
            6
        } else if !p.chroma {
            3
        } else {
            2
        };
        let n2: isize = if log2_size == 3 && !p.chroma { 0 } else { 1 };
        let mut out = [[0i32; 4]; 12];
        for i in -n..n {
            let mut acc = [0i32; 4];
            for j in -n..=n {
                let pp = (i + j).clamp(-(n + 1), n);
                let tap = if j.abs() <= n2 { 2 } else { 1 };
                let s = &v[(7 + pp) as usize];
                for l in 0..4 {
                    acc[l] += s[l] * tap;
                }
            }
            let add = 1 << (log2_size - 1);
            for l in 0..4 {
                out[(i + 6) as usize][l] = (acc[l] + add) >> log2_size;
            }
        }
        out
    };
    let w8 = if p.filter_size >= 8 {
        Some(wide(3, v))
    } else {
        None
    };
    let w16 = if p.filter_size >= 16 {
        Some(wide(4, v))
    } else {
        None
    };
    let n8: isize = if p.chroma { 2 } else { 3 };
    for l in 0..4 {
        if skip[l] {
            continue;
        }
        if p.filter_size == 4 || !flat[l] {
            v[5][l] = n_out[0][l];
            v[6][l] = n_out[1][l];
            v[7][l] = n_out[2][l];
            v[8][l] = n_out[3][l];
        } else if p.filter_size == 8 || !flat2[l] {
            let w = w8.as_ref().expect("filter size 8 or more");
            for i in -n8..n8 {
                v[(7 + i) as usize][l] = w[(i + 6) as usize][l];
            }
        } else {
            let w = w16.as_ref().expect("filter size 16");
            for i in -6isize..6 {
                v[(7 + i) as usize][l] = w[(i + 6) as usize][l];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels that filter the plane where it lies
// ---------------------------------------------------------------------------

/// One unit's edge filtered on the plane a sample at a time: the
/// one-sample process (7.14.6) for each of the four samples along the edge
/// from index `pos` (`along` apart), reading the samples across it (`step`
/// apart) as the mask and the filter that the sample takes ask for them,
/// and writing the ones that filter changes. For a build with no vector
/// lanes, where computing every filter for every lane, as `filter` does,
/// is all cost. Bit-exact with it (tested).
///
/// # Safety
///
/// `p` is valid for reading and writing `len` samples, and while the call
/// runs nothing else reads or writes the samples of the four lanes within
/// seven samples of the edge.
#[cfg(any(test, all(target_arch = "wasm32", not(target_feature = "simd128"))))]
pub(crate) unsafe fn filter_unit(
    p: *mut u16,
    len: usize,
    pos: usize,
    step: usize,
    along: usize,
    k: &Params,
) {
    let filter_len = if k.filter_size == 4 {
        4
    } else if k.chroma {
        6
    } else if k.filter_size == 8 {
        8
    } else {
        16
    };
    // The samples on each side that a lane may read.
    let reach = match filter_len {
        4 => 2,
        6 => 3,
        8 => 4,
        _ => 7,
    };
    assert!(pos >= reach * step && pos + 3 * along + (reach - 1) * step < len);
    let sh = k.bit_depth - 8;
    let (limit, blimit, thresh) = (k.limit << sh, k.blimit << sh, k.thresh << sh);
    let one = 1 << sh;
    let max = (1 << k.bit_depth) - 1;
    for lane in 0..4 {
        // SAFETY (both): sample `i` from the edge of this lane, with -reach
        // <= i < reach: inside the plane by the assertion, and this call's
        // alone by the caller's guarantee.
        let at = |i: isize| unsafe { p.add(pos + lane * along).offset(i * step as isize) };
        let get = |i: isize| unsafe { *at(i) as i32 };
        let put = |i: isize, v: i32| unsafe { *at(i) = v as u16 };
        // The filter mask process (7.14.6.2), nearest samples first.
        let (p1, p0, q0, q1) = (get(-2), get(-1), get(0), get(1));
        let near = (p1 - p0).abs().max((q1 - q0).abs());
        if near > limit || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit {
            continue;
        }
        let mut flat = false;
        if filter_len >= 6 {
            let (p2, q2) = (get(-3), get(2));
            if (p2 - p1).abs() > limit || (q2 - q1).abs() > limit {
                continue;
            }
            flat = near <= one && (p2 - p0).abs() <= one && (q2 - q0).abs() <= one;
        }
        if filter_len >= 8 {
            let (p2, p3, q2, q3) = (get(-3), get(-4), get(2), get(3));
            if (p3 - p2).abs() > limit || (q3 - q2).abs() > limit {
                continue;
            }
            flat = flat && (p3 - p0).abs() <= one && (q3 - q0).abs() <= one;
        }
        if !flat {
            // The narrow filter (7.14.6.3). Its samples are taken less
            // `0x80 << (BitDepth - 8)`: the offset cancels in the
            // differences, and with it added back the last clamp is the
            // clamp to the sample range.
            let (lo, hi) = (-(1 << (k.bit_depth - 1)), (1 << (k.bit_depth - 1)) - 1);
            let hev = near > thresh;
            let f = if hev { (p1 - q1).clamp(lo, hi) } else { 0 };
            let f = (f + 3 * (q0 - p0)).clamp(lo, hi);
            let f1 = (f + 4).min(hi) >> 3;
            let f2 = (f + 3).min(hi) >> 3;
            put(0, (q0 - f1).clamp(0, max));
            put(-1, (p0 + f2).clamp(0, max));
            if !hev {
                let f3 = (f1 + 1) >> 1;
                put(1, (q1 - f3).clamp(0, max));
                put(-2, (p1 + f3).clamp(0, max));
            }
            continue;
        }
        // The wide filters (7.14.6.4).
        let mut v = [0i32; 14];
        let n = if filter_len == 6 { 3 } else { 4 };
        for i in -n..n {
            v[(7 + i) as usize] = get(i);
        }
        if filter_len == 16 {
            for i in 4..7 {
                v[(7 + i) as usize] = get(i);
                v[(6 - i) as usize] = get(-1 - i);
            }
        }
        let flat2 = filter_len == 16
            && (0..3).all(|i| (v[i] - p0).abs() <= one && (v[13 - i] - q0).abs() <= one);
        let out = if flat2 {
            wide_one::<6, 1, 4>(&v)
        } else if filter_len == 6 {
            wide_one::<2, 1, 3>(&v)
        } else {
            wide_one::<3, 0, 3>(&v)
        };
        let n = if flat2 { 6 } else { n - 1 };
        for i in -n..n {
            put(i, out[(6 + i) as usize]);
        }
    }
}

/// The wide filter (7.14.6.4) of the `2 * N` samples of one lane nearest
/// the edge, from `v` (`v[7 + k]` is the k-th sample from the edge); sample
/// `i` of the result is at index `i + 6`. `N2` is 1 where the three middle
/// taps count twice, 0 where the middle one does, and `LOG2` is the
/// filter's shift. The first sum is the process's; each one after it is the
/// sum before plus the two samples whose weight rises as the taps move one
/// sample on, less the two whose weight falls.
#[cfg(any(test, all(target_arch = "wasm32", not(target_feature = "simd128"))))]
#[inline(always)]
fn wide_one<const N: isize, const N2: isize, const LOG2: u32>(v: &[i32; 14]) -> [i32; 12] {
    let at = |i: isize| v[(7 + i.clamp(-(N + 1), N)) as usize];
    let mut sum = 1 << (LOG2 - 1);
    for j in -N..=N {
        sum += at(-N + j) * if j.abs() <= N2 { 2 } else { 1 };
    }
    let mut out = [0; 12];
    out[(6 - N) as usize] = sum >> LOG2;
    for i in 1 - N..N {
        sum += at(i + N) + at(i + N2) - at(i - 1 - N) - at(i - 1 - N2);
        out[(i + 6) as usize] = sum >> LOG2;
    }
    out
}

/// The edges of two 4x4 units that follow each other along an edge: eight
/// samples for one kernel call, the first four filtered with `limits[0]`
/// and the last four with `limits[1]`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pair {
    /// `limit`, `blimit` and `thresh` (7.14.4) of each unit.
    pub(crate) limits: [[u8; 3]; 2],
    /// Whether each unit's edge is filtered at all.
    pub(crate) on: [bool; 2],
    /// The filter size (7.14.3) of the units that are filtered: 4, 8 or 16.
    pub(crate) filter_size: usize,
    pub(crate) chroma: bool,
    pub(crate) bit_depth: u32,
}

/// Filters the eight samples along an edge that start at index `pos` of a
/// plane (`stride` samples a row), reading and writing the plane itself.
/// Along a `vertical` edge (pass 0) they are one below the other and the
/// filter runs along their rows; along a horizontal one (pass 1) they are
/// side by side and it runs along their columns. Bit-exact with `filter`
/// on each unit (tested).
///
/// # Safety
///
/// `p` is valid for reading and writing `len` samples, and while the call
/// runs nothing else reads or writes the samples of the eight lanes within
/// eight samples of the edge.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) unsafe fn filter_pair(
    p: *mut u16,
    len: usize,
    pos: usize,
    stride: usize,
    vertical: bool,
    k: &Pair,
) {
    #[cfg(target_arch = "aarch64")]
    use neon as lanes;
    #[cfg(target_arch = "wasm32")]
    use simd128 as lanes;
    // The filter length of 7.14.6.2: how far the mask looks.
    let filter_len = match (k.filter_size, k.chroma) {
        (4, _) => 4,
        (_, true) => 6,
        (8, false) => 8,
        _ => 16,
    };
    // SAFETY: the caller's guarantee is the kernels' (each asserts that what
    // it touches is inside `len`); NEON is part of the aarch64 baseline.
    unsafe {
        match (vertical, filter_len) {
            (false, 4) => lanes::across_rows::<4>(p, len, pos, stride, k),
            (false, 6) => lanes::across_rows::<6>(p, len, pos, stride, k),
            (false, 8) => lanes::across_rows::<8>(p, len, pos, stride, k),
            (false, _) => lanes::across_rows::<16>(p, len, pos, stride, k),
            (true, 4) => lanes::across_columns::<4>(p, len, pos, stride, k),
            (true, 6) => lanes::across_columns::<6>(p, len, pos, stride, k),
            (true, 8) => lanes::across_columns::<8>(p, len, pos, stride, k),
            (true, _) => lanes::across_columns::<16>(p, len, pos, stride, k),
        }
    }
}

/// The in-plane kernels, written once over a vector of eight 16-bit lanes
/// (`V`) and the handful of operations on it that the module they are
/// expanded in provides: `simd128` and `neon`. `$tf` is the attribute
/// that module's functions carry (NEON's `target_feature`).
///
/// 16 bits are enough: samples are at most 12 bits, so the differences,
/// `2 * |p0 - q0| + |p1 - q1| / 2` and the narrow filter's `3 * (q0 - p0)`
/// plus a clamped term stay inside an `i16`, and the widest filter's sum
/// (16 samples and a rounding 8) inside a `u16`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! in_plane_kernels {
    ($(#[$tf:meta])*) => {
        /// A pair's limits as lanes, and the constants of its bit depth.
        #[derive(Clone, Copy)]
        struct Limits {
            limit: V,
            blimit: V,
            thresh: V,
            /// The flatness threshold, `1 << (BitDepth - 8)`.
            one: V,
            /// The range of the narrow filter's signed values, and of a
            /// sample.
            lo: V,
            hi: V,
            max: V,
        }

        $(#[$tf])*
        #[inline]
        fn limits(k: &Pair) -> Limits {
            let sh = k.bit_depth - 8;
            let of = |unit: usize, i: usize| (k.limits[unit][i] as i16) << sh;
            // No lane of a unit that is not filtered passes the mask: the
            // sum of differences `blimit` is compared with is never below 0.
            let blimit = |unit: usize| if k.on[unit] { of(unit, 1) } else { -1 };
            Limits {
                limit: halves(of(0, 0), of(1, 0)),
                blimit: halves(blimit(0), blimit(1)),
                thresh: halves(of(0, 2), of(1, 2)),
                one: splat(1 << sh),
                lo: splat(-(1 << (k.bit_depth - 1))),
                hi: splat((1 << (k.bit_depth - 1)) - 1),
                max: splat(((1i32 << k.bit_depth) - 1) as i16),
            }
        }

        /// The narrow filter (7.14.6.3) of every lane: the new `p1, p0, q0,
        /// q1`. The process works on samples less `0x80 << (BitDepth - 8)`;
        /// that offset cancels in the differences, and the last clamp of
        /// each result, taken with the offset added back, is the clamp to
        /// the sample range. Where `hev` is set the outer two get a
        /// correction of zero, which leaves them as they are.
        $(#[$tf])*
        #[inline]
        fn narrow(p1: V, p0: V, q0: V, q1: V, hev: V, t: &Limits) -> [V; 4] {
            let clamp = |x: V| max(min(x, t.hi), t.lo);
            let sample = |x: V| max(min(x, t.max), splat(0));
            let d = sub(q0, p0);
            let f = clamp(add(and(clamp(sub(p1, q1)), hev), add(add(d, d), d)));
            // `f` is at least `lo`: only the upper clamp can bite.
            let f1 = sar::<3>(min(add(f, splat(4)), t.hi));
            let f2 = sar::<3>(min(add(f, splat(3)), t.hi));
            let f3 = andnot(sar::<1>(add(f1, splat(1))), hev);
            [
                sample(add(p1, f3)),
                sample(add(p0, f2)),
                sample(sub(q0, f1)),
                sample(sub(q1, f3)),
            ]
        }

        /// The wide filter (7.14.6.4) of the `2 * N` samples nearest the
        /// edge, from `v` (`v[7 + k]` is the k-th sample from the edge, as
        /// in `Edge`); sample `i` of the result is at index `i + 6`.
        /// `N2` is 1 where the three middle taps count twice, 0 where the
        /// middle one does, and `LOG2` is the filter's shift.
        ///
        /// The first sum is the process's; each one after it is the sum
        /// before plus the two samples whose weight rises as the taps move
        /// one sample on, less the two whose weight falls.
        $(#[$tf])*
        #[inline]
        fn wide<const N: isize, const N2: isize, const LOG2: i32>(v: &[V; 14]) -> [V; 12] {
            let at = |i: isize| v[(7 + i.clamp(-(N + 1), N)) as usize];
            let mut sum = splat(1 << (LOG2 - 1));
            for j in -N..=N {
                sum = add(sum, at(-N + j));
                if j.abs() <= N2 {
                    sum = add(sum, at(-N + j));
                }
            }
            let mut out = [v[0]; 12];
            out[(6 - N) as usize] = shr::<LOG2>(sum);
            for i in 1 - N..N {
                sum = sub(
                    add(sum, add(at(i + N), at(i + N2))),
                    add(at(i - 1 - N), at(i - 1 - N2)),
                );
                out[(i + 6) as usize] = shr::<LOG2>(sum);
            }
            out
        }

        /// The sample filtering process (7.14.6) of eight lanes with a
        /// filter length of `LEN` (4; 6 for chroma's size 8; 8; 16): the
        /// samples from `v` (as `Edge`; only the ones the length reads
        /// need be there), the result as [`wide`] gives it and how many
        /// samples on each side of the edge it may have changed, or `None`
        /// where no lane passes the mask.
        ///
        /// Every lane's mask comes first, and a filter is computed only if
        /// some lane takes it.
        $(#[$tf])*
        #[inline]
        fn filter_lanes8<const LEN: usize>(v: &[V; 14], t: &Limits) -> Option<([V; 12], usize)> {
            let (p3, p2, p1, p0) = (v[3], v[4], v[5], v[6]);
            let (q0, q1, q2, q3) = (v[7], v[8], v[9], v[10]);
            // The filter mask process (7.14.6.2).
            let near = max(absd(p1, p0), absd(q1, q0));
            let mut steps = near;
            if LEN >= 6 {
                steps = max(steps, max(absd(p2, p1), absd(q2, q1)));
            }
            if LEN >= 8 {
                steps = max(steps, max(absd(p3, p2), absd(q3, q2)));
            }
            let across = absd(p0, q0);
            let skip = or(
                gt(steps, t.limit),
                gt(add(add(across, across), shr::<1>(absd(p1, q1))), t.blimit),
            );
            if all(skip) {
                return None;
            }
            let hev = gt(near, t.thresh);
            // The lanes that take a wide filter.
            let mut flat = splat(0);
            if LEN >= 6 {
                let mut level = max(near, max(absd(p2, p0), absd(q2, q0)));
                if LEN >= 8 {
                    level = max(level, max(absd(p3, p0), absd(q3, q0)));
                }
                flat = andnot(le(level, t.one), skip);
            }
            let mut out: [V; 12] = std::array::from_fn(|i| v[i + 1]);
            let mut reach = 2;
            if !all(or(skip, flat)) {
                let n = narrow(p1, p0, q0, q1, hev, t);
                for (i, n) in n.into_iter().enumerate() {
                    out[4 + i] = select(skip, out[4 + i], n);
                }
            }
            if LEN >= 6 && any(flat) {
                if LEN == 6 {
                    let w = wide::<2, 1, 3>(v);
                    for i in 4..8 {
                        out[i] = select(flat, w[i], out[i]);
                    }
                } else {
                    let w = wide::<3, 0, 3>(v);
                    for i in 3..9 {
                        out[i] = select(flat, w[i], out[i]);
                    }
                    reach = 3;
                }
                if LEN == 16 {
                    let level = max(
                        max(absd(v[2], p0), absd(v[11], q0)),
                        max(
                            max(absd(v[1], p0), absd(v[12], q0)),
                            max(absd(v[0], p0), absd(v[13], q0)),
                        ),
                    );
                    let flat2 = and(flat, le(level, t.one));
                    if any(flat2) {
                        let w = wide::<6, 1, 4>(v);
                        for i in 0..12 {
                            out[i] = select(flat2, w[i], out[i]);
                        }
                        reach = 6;
                    }
                }
            }
            Some((out, reach))
        }

        /// Eight rows of eight samples as eight columns, and back.
        $(#[$tf])*
        #[inline]
        fn transpose(r: [V; 8]) -> [V; 8] {
            let a = [
                zip16_lo(r[0], r[1]),
                zip16_hi(r[0], r[1]),
                zip16_lo(r[2], r[3]),
                zip16_hi(r[2], r[3]),
                zip16_lo(r[4], r[5]),
                zip16_hi(r[4], r[5]),
                zip16_lo(r[6], r[7]),
                zip16_hi(r[6], r[7]),
            ];
            let b = [
                zip32_lo(a[0], a[2]),
                zip32_hi(a[0], a[2]),
                zip32_lo(a[1], a[3]),
                zip32_hi(a[1], a[3]),
                zip32_lo(a[4], a[6]),
                zip32_hi(a[4], a[6]),
                zip32_lo(a[5], a[7]),
                zip32_hi(a[5], a[7]),
            ];
            [
                zip64_lo(b[0], b[4]),
                zip64_hi(b[0], b[4]),
                zip64_lo(b[1], b[5]),
                zip64_hi(b[1], b[5]),
                zip64_lo(b[2], b[6]),
                zip64_hi(b[2], b[6]),
                zip64_lo(b[3], b[7]),
                zip64_hi(b[3], b[7]),
            ]
        }

        /// A horizontal edge: the eight lanes are the columns from `pos`,
        /// the samples across the edge the rows above and below it, each
        /// row one load.
        ///
        /// # Safety
        ///
        /// As `filter_pair`.
        $(#[$tf])*
        pub(super) unsafe fn across_rows<const LEN: usize>(
            p: *mut u16,
            len: usize,
            pos: usize,
            stride: usize,
            k: &Pair,
        ) {
            // The rows the mask and the filters read on each side.
            let rows: usize = match LEN {
                4 => 2,
                6 => 3,
                8 => 4,
                _ => 7,
            };
            assert!(pos >= rows * stride && pos + (rows - 1) * stride + 8 <= len);
            // SAFETY (every use): row `i` from the edge with -rows <= i <
            // rows, eight samples from `pos`'s column: inside the plane by
            // the assertion, and this call's alone by the caller's
            // guarantee.
            let row = |i: isize| unsafe { p.offset(pos as isize + i * stride as isize) };
            let mut v = [splat(0); 14];
            for i in -(rows as isize)..rows as isize {
                v[(7 + i) as usize] = unsafe { load(row(i)) };
            }
            if let Some((out, reach)) = filter_lanes8::<LEN>(&v, &limits(k)) {
                for i in -(reach as isize)..reach as isize {
                    unsafe { store(row(i), out[(6 + i) as usize]) };
                }
            }
        }

        /// A vertical edge: the eight lanes are the rows from `pos` down,
        /// the samples across the edge lie along each row. Eight rows of
        /// eight samples are transposed into eight vectors of one column
        /// each (twice for the 16 samples of the widest filter), filtered,
        /// and transposed back; rows of four when only the narrow filter's
        /// four columns changed.
        ///
        /// # Safety
        ///
        /// As `filter_pair`.
        $(#[$tf])*
        pub(super) unsafe fn across_columns<const LEN: usize>(
            p: *mut u16,
            len: usize,
            pos: usize,
            stride: usize,
            k: &Pair,
        ) {
            let cols: usize = if LEN == 16 { 8 } else { 4 };
            assert!(pos >= cols && pos + 7 * stride + cols <= len);
            // SAFETY (every use): lane `l` < 8 is `l` rows below `pos`, and
            // column `i` from the edge has -cols <= i, with what is loaded
            // or stored from there ending before column `cols`: inside the
            // plane by the assertion, and this call's alone by the caller's
            // guarantee.
            let at = |l: usize, i: isize| unsafe { p.add(pos + l * stride).offset(i) };
            let mut v = [splat(0); 14];
            let far = if LEN == 16 {
                // Columns -8..0 and 0..8; the outermost on each side is
                // not the filter's.
                let left = transpose(std::array::from_fn(|l| unsafe { load(at(l, -8)) }));
                let right = transpose(std::array::from_fn(|l| unsafe { load(at(l, 0)) }));
                v[..7].copy_from_slice(&left[1..]);
                v[7..].copy_from_slice(&right[..7]);
                Some((left, right))
            } else {
                // Columns -4..4.
                let near = transpose(std::array::from_fn(|l| unsafe { load(at(l, -4)) }));
                v[3..11].copy_from_slice(&near);
                None
            };
            let Some((out, reach)) = filter_lanes8::<LEN>(&v, &limits(k)) else {
                return;
            };
            match (reach, far) {
                (6, Some((left, right))) => {
                    // Columns -8..8: the two outermost on each side as they
                    // were.
                    let a = transpose([
                        left[0], left[1], out[0], out[1], out[2], out[3], out[4], out[5],
                    ]);
                    let b = transpose([
                        out[6], out[7], out[8], out[9], out[10], out[11], right[6], right[7],
                    ]);
                    for l in 0..8 {
                        unsafe {
                            store(at(l, -8), a[l]);
                            store(at(l, 0), b[l]);
                        }
                    }
                }
                (3, _) => {
                    // Columns -4..4, the outermost on each side as they
                    // were.
                    let m = transpose([
                        v[3], out[3], out[4], out[5], out[6], out[7], out[8], v[10],
                    ]);
                    for (l, m) in m.into_iter().enumerate() {
                        unsafe { store(at(l, -4), m) };
                    }
                }
                _ => {
                    // Columns -2..2: two rows of four to a vector.
                    let p_lo = zip16_lo(out[4], out[5]);
                    let p_hi = zip16_hi(out[4], out[5]);
                    let q_lo = zip16_lo(out[6], out[7]);
                    let q_hi = zip16_hi(out[6], out[7]);
                    let two_rows = [
                        zip32_lo(p_lo, q_lo),
                        zip32_hi(p_lo, q_lo),
                        zip32_lo(p_hi, q_hi),
                        zip32_hi(p_hi, q_hi),
                    ];
                    for (l, m) in two_rows.into_iter().enumerate() {
                        unsafe {
                            store_lo(at(2 * l, -2), m);
                            store_hi(at(2 * l + 1, -2), m);
                        }
                    }
                }
            }
        }
    };
}

/// The lane operations of the in-plane kernels in WebAssembly SIMD128.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use super::Pair;
    use std::arch::wasm32::*;

    type V = v128;

    #[inline(always)]
    fn splat(x: i16) -> V {
        i16x8_splat(x)
    }
    /// `a` in the first four lanes, `b` in the last four.
    #[inline(always)]
    fn halves(a: i16, b: i16) -> V {
        i16x8(a, a, a, a, b, b, b, b)
    }
    #[inline(always)]
    fn add(a: V, b: V) -> V {
        i16x8_add(a, b)
    }
    #[inline(always)]
    fn sub(a: V, b: V) -> V {
        i16x8_sub(a, b)
    }
    /// `|a - b|` of two samples.
    #[inline(always)]
    fn absd(a: V, b: V) -> V {
        i16x8_abs(i16x8_sub(a, b))
    }
    #[inline(always)]
    fn min(a: V, b: V) -> V {
        i16x8_min(a, b)
    }
    #[inline(always)]
    fn max(a: V, b: V) -> V {
        i16x8_max(a, b)
    }
    /// All ones in the lanes where `a > b`.
    #[inline(always)]
    fn gt(a: V, b: V) -> V {
        i16x8_gt(a, b)
    }
    #[inline(always)]
    fn le(a: V, b: V) -> V {
        i16x8_le(a, b)
    }
    #[inline(always)]
    fn and(a: V, b: V) -> V {
        v128_and(a, b)
    }
    #[inline(always)]
    fn or(a: V, b: V) -> V {
        v128_or(a, b)
    }
    /// `a` and not `b`.
    #[inline(always)]
    fn andnot(a: V, b: V) -> V {
        v128_andnot(a, b)
    }
    /// `a` in the lanes `mask` has set, `b` in the others.
    #[inline(always)]
    fn select(mask: V, a: V, b: V) -> V {
        v128_bitselect(a, b, mask)
    }
    /// An arithmetic shift right.
    #[inline(always)]
    fn sar<const N: i32>(a: V) -> V {
        i16x8_shr(a, N as u32)
    }
    /// A logical shift right.
    #[inline(always)]
    fn shr<const N: i32>(a: V) -> V {
        u16x8_shr(a, N as u32)
    }
    /// Whether any lane of a mask is set.
    #[inline(always)]
    fn any(mask: V) -> bool {
        v128_any_true(mask)
    }
    /// Whether every lane of a mask is set.
    #[inline(always)]
    fn all(mask: V) -> bool {
        i16x8_all_true(mask)
    }
    #[inline(always)]
    fn zip16_lo(a: V, b: V) -> V {
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
    }
    #[inline(always)]
    fn zip16_hi(a: V, b: V) -> V {
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
    }
    #[inline(always)]
    fn zip32_lo(a: V, b: V) -> V {
        i32x4_shuffle::<0, 4, 1, 5>(a, b)
    }
    #[inline(always)]
    fn zip32_hi(a: V, b: V) -> V {
        i32x4_shuffle::<2, 6, 3, 7>(a, b)
    }
    #[inline(always)]
    fn zip64_lo(a: V, b: V) -> V {
        i64x2_shuffle::<0, 2>(a, b)
    }
    #[inline(always)]
    fn zip64_hi(a: V, b: V) -> V {
        i64x2_shuffle::<1, 3>(a, b)
    }
    /// Eight samples from `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for reading eight samples.
    #[inline(always)]
    unsafe fn load(p: *const u16) -> V {
        // SAFETY: the caller's.
        unsafe { v128_load(p as *const v128) }
    }
    /// Eight samples to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing eight samples.
    #[inline(always)]
    unsafe fn store(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { v128_store(p as *mut v128, v) }
    }
    /// The first four lanes to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing four samples.
    #[inline(always)]
    unsafe fn store_lo(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { v128_store64_lane::<0>(v, p as *mut u64) }
    }
    /// The last four lanes to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing four samples.
    #[inline(always)]
    unsafe fn store_hi(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { v128_store64_lane::<1>(v, p as *mut u64) }
    }

    in_plane_kernels!();
}

/// The lane operations of the in-plane kernels in NEON.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Pair;
    use std::arch::aarch64::*;

    type V = int16x8_t;

    #[inline]
    #[target_feature(enable = "neon")]
    fn splat(x: i16) -> V {
        vdupq_n_s16(x)
    }
    /// `a` in the first four lanes, `b` in the last four.
    #[inline]
    #[target_feature(enable = "neon")]
    fn halves(a: i16, b: i16) -> V {
        vcombine_s16(vdup_n_s16(a), vdup_n_s16(b))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn add(a: V, b: V) -> V {
        vaddq_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn sub(a: V, b: V) -> V {
        vsubq_s16(a, b)
    }
    /// `|a - b|` of two samples.
    #[inline]
    #[target_feature(enable = "neon")]
    fn absd(a: V, b: V) -> V {
        vabdq_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn min(a: V, b: V) -> V {
        vminq_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn max(a: V, b: V) -> V {
        vmaxq_s16(a, b)
    }
    /// All ones in the lanes where `a > b`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn gt(a: V, b: V) -> V {
        vreinterpretq_s16_u16(vcgtq_s16(a, b))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn le(a: V, b: V) -> V {
        vreinterpretq_s16_u16(vcleq_s16(a, b))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn and(a: V, b: V) -> V {
        vandq_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn or(a: V, b: V) -> V {
        vorrq_s16(a, b)
    }
    /// `a` and not `b`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn andnot(a: V, b: V) -> V {
        vbicq_s16(a, b)
    }
    /// `a` in the lanes `mask` has set, `b` in the others.
    #[inline]
    #[target_feature(enable = "neon")]
    fn select(mask: V, a: V, b: V) -> V {
        vbslq_s16(vreinterpretq_u16_s16(mask), a, b)
    }
    /// An arithmetic shift right.
    #[inline]
    #[target_feature(enable = "neon")]
    fn sar<const N: i32>(a: V) -> V {
        vshrq_n_s16::<N>(a)
    }
    /// A logical shift right.
    #[inline]
    #[target_feature(enable = "neon")]
    fn shr<const N: i32>(a: V) -> V {
        vreinterpretq_s16_u16(vshrq_n_u16::<N>(vreinterpretq_u16_s16(a)))
    }
    /// Whether any lane of a mask is set.
    #[inline]
    #[target_feature(enable = "neon")]
    fn any(mask: V) -> bool {
        vmaxvq_u16(vreinterpretq_u16_s16(mask)) != 0
    }
    /// Whether every lane of a mask is set.
    #[inline]
    #[target_feature(enable = "neon")]
    fn all(mask: V) -> bool {
        vminvq_u16(vreinterpretq_u16_s16(mask)) != 0
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip16_lo(a: V, b: V) -> V {
        vzip1q_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip16_hi(a: V, b: V) -> V {
        vzip2q_s16(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip32_lo(a: V, b: V) -> V {
        vreinterpretq_s16_s32(vzip1q_s32(
            vreinterpretq_s32_s16(a),
            vreinterpretq_s32_s16(b),
        ))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip32_hi(a: V, b: V) -> V {
        vreinterpretq_s16_s32(vzip2q_s32(
            vreinterpretq_s32_s16(a),
            vreinterpretq_s32_s16(b),
        ))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip64_lo(a: V, b: V) -> V {
        vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(a),
            vreinterpretq_s64_s16(b),
        ))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn zip64_hi(a: V, b: V) -> V {
        vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(a),
            vreinterpretq_s64_s16(b),
        ))
    }
    /// Eight samples from `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for reading eight samples.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const u16) -> V {
        // SAFETY: the caller's.
        unsafe { vreinterpretq_s16_u16(vld1q_u16(p)) }
    }
    /// Eight samples to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing eight samples.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { vst1q_u16(p, vreinterpretq_u16_s16(v)) }
    }
    /// The first four lanes to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing four samples.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store_lo(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { vst1_u16(p, vget_low_u16(vreinterpretq_u16_s16(v))) }
    }
    /// The last four lanes to `p`.
    ///
    /// # Safety
    ///
    /// `p` is valid for writing four samples.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store_hi(p: *mut u16, v: V) {
        // SAFETY: the caller's.
        unsafe { vst1_u16(p, vget_high_u16(vreinterpretq_u16_s16(v))) }
    }

    in_plane_kernels!(#[target_feature(enable = "neon")]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: the one-sample process of 7.14.6, as written.
    fn reference(v: &mut [i32; 14], p: &Params) {
        let sh = p.bit_depth - 8;
        let (q0, q1, q2, q3) = (v[7], v[8], v[9], v[10]);
        let (p0, p1, p2, p3) = (v[6], v[5], v[4], v[3]);
        let thresh_bd = p.thresh << sh;
        let hev_mask = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
        let filter_len = if p.filter_size == 4 {
            4
        } else if p.chroma {
            6
        } else if p.filter_size == 8 {
            8
        } else {
            16
        };
        let limit_bd = p.limit << sh;
        let blimit_bd = p.blimit << sh;
        let mut mask = (p1 - p0).abs() > limit_bd
            || (q1 - q0).abs() > limit_bd
            || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
        if filter_len >= 6 {
            mask |= (p2 - p1).abs() > limit_bd || (q2 - q1).abs() > limit_bd;
        }
        if filter_len >= 8 {
            mask |= (p3 - p2).abs() > limit_bd || (q3 - q2).abs() > limit_bd;
        }
        if mask {
            return;
        }
        let t = 1 << sh;
        let mut flat = false;
        if p.filter_size >= 8 {
            let mut m = (p1 - p0).abs() > t
                || (q1 - q0).abs() > t
                || (p2 - p0).abs() > t
                || (q2 - q0).abs() > t;
            if filter_len >= 8 {
                m |= (p3 - p0).abs() > t || (q3 - q0).abs() > t;
            }
            flat = !m;
        }
        let mut flat2 = false;
        if p.filter_size >= 16 {
            let m = (v[0] - p0).abs() > t
                || (v[13] - q0).abs() > t
                || (v[1] - p0).abs() > t
                || (v[12] - q0).abs() > t
                || (v[2] - p0).abs() > t
                || (v[11] - q0).abs() > t;
            flat2 = !m;
        }
        if p.filter_size == 4 || !flat {
            let bd = p.bit_depth;
            let c = |x: i32| x.clamp(-(1 << (bd - 1)), (1 << (bd - 1)) - 1);
            let off = 0x80 << sh;
            let (ps1, ps0, qs0, qs1) = (p1 - off, p0 - off, q0 - off, q1 - off);
            let mut fl = if hev_mask { c(ps1 - qs1) } else { 0 };
            fl = c(fl + 3 * (qs0 - ps0));
            let filter1 = c(fl + 4) >> 3;
            let filter2 = c(fl + 3) >> 3;
            v[7] = c(qs0 - filter1) + off;
            v[6] = c(ps0 + filter2) + off;
            if !hev_mask {
                let f = (filter1 + 1) >> 1;
                v[8] = c(qs1 - f) + off;
                v[5] = c(ps1 + f) + off;
            }
        } else {
            let log2_size = if p.filter_size == 8 || !flat2 { 3 } else { 4 };
            let n: isize = if log2_size == 4 {
                6
            } else if !p.chroma {
                3
            } else {
                2
            };
            let n2: isize = if log2_size == 3 && !p.chroma { 0 } else { 1 };
            let orig = *v;
            for i in -n..n {
                let mut t = 0;
                for j in -n..=n {
                    let pp = (i + j).clamp(-(n + 1), n);
                    let tap = if j.abs() <= n2 { 2 } else { 1 };
                    t += orig[(7 + pp) as usize] * tap;
                }
                v[(7 + i) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
            }
        }
    }

    /// Every edge kind, random samples near and across the thresholds.
    #[test]
    fn lanes_match_the_process() {
        let mut seed = 0x0bad_5eed_1234_5678u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for case in 0..20000 {
            let bd = [8u32, 10, 12][case % 3];
            let params = Params {
                limit: 1 + rnd(9) as i32,
                blimit: rnd(130) as i32,
                thresh: rnd(4) as i32,
                filter_size: [4, 8, 16, 6][rnd(4) as usize].min(if case % 5 == 0 { 8 } else { 16 }),
                chroma: case % 5 == 0,
                bit_depth: bd,
            };
            let mut p = params;
            if p.chroma {
                p.filter_size = p.filter_size.min(8);
            }
            let base = rnd(1 << bd) as i32;
            let spread = [1u64, 4, 16, 64][rnd(4) as usize] << (bd - 8);
            let mut lanes: Edge = [[0; 4]; 14];
            for k in 0..14 {
                for l in 0..4 {
                    lanes[k][l] =
                        (base + rnd(spread) as i32 - (spread / 2) as i32).clamp(0, (1 << bd) - 1);
                }
            }
            let mut want = lanes;
            for l in 0..4 {
                let mut one: [i32; 14] = std::array::from_fn(|k| lanes[k][l]);
                reference(&mut one, &p);
                for k in 0..14 {
                    want[k][l] = one[k];
                }
            }
            let mut got = lanes;
            filter(&mut got, &p);
            assert_eq!(got, want, "case {case} {p:?}");
        }
    }

    /// The in-plane kernels against the one-sample process: random planes
    /// of smooth areas, steps and noise, an edge of every filter size in
    /// both directions at every depth, the two units with limits of their
    /// own, either of them left unfiltered. The whole plane is compared, so
    /// a sample written that should not have been shows too.
    #[test]
    fn in_plane_kernels_match_the_process() {
        let mut seed = 0x6a09_e667_f3bc_c908u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        const W: usize = 24;
        const H: usize = 24;
        let mut took = [0usize; 4];
        for case in 0..100000 {
            let bd = [8u32, 10, 12][case % 3];
            let chroma = rnd(3) == 0;
            let filter_size = if chroma {
                [4, 8][rnd(2) as usize]
            } else {
                [4, 8, 16][rnd(3) as usize]
            };
            let vertical = rnd(2) == 0;
            let mut limits = [[0u8; 3]; 2];
            for l in &mut limits {
                let lvl = 1 + rnd(63) as i32;
                let limit = 1 + rnd(lvl as u64) as i32;
                *l = [limit as u8, (2 * (lvl + 2) + limit) as u8, (lvl >> 4) as u8];
            }
            let on = [[true, true], [true, false], [false, true]][rnd(3).min(rnd(3)) as usize];
            let params = limits.map(|[limit, blimit, thresh]| Params {
                limit: limit as i32,
                blimit: blimit as i32,
                thresh: thresh as i32,
                filter_size,
                chroma,
                bit_depth: bd,
            });
            // The plane: the edge in the middle, eight samples along it,
            // and on each side of it, for each of the two units, an area of
            // one value and one amount of noise. The far side is the same
            // area, or one a step away (which the wide filters smooth), or
            // an unrelated one.
            let maxv = (1u64 << bd) - 1;
            let mut areas: [(u64, u64); 4] = std::array::from_fn(|_| {
                let noise = [1, 2, 2, 3, 5, 12, 80][rnd(7) as usize] << (bd - 8);
                (rnd(maxv + 1), noise)
            });
            for unit in 0..2 {
                let (base, noise) = areas[2 * unit];
                match rnd(4) {
                    0 => areas[2 * unit + 1] = (base, noise),
                    1 | 2 => {
                        let step = rnd(24) << (bd - 8);
                        let far = if rnd(2) == 0 {
                            base + step
                        } else {
                            base.saturating_sub(step)
                        };
                        areas[2 * unit + 1] = (far.min(maxv), noise);
                    }
                    _ => {}
                }
            }
            let (pos, step, along) = if vertical {
                (8 * W + 12, 1, W)
            } else {
                (12 * W + 8, W, 1)
            };
            let mut plane = vec![0u16; W * H];
            for y in 0..H {
                for x in 0..W {
                    let (across, lane) = if vertical { (x, y) } else { (y, x) };
                    let (base, noise) = areas[2 * (lane / 12) + across / 12];
                    let v = (base + rnd(noise)).saturating_sub(noise / 2).min(maxv);
                    plane[y * W + x] = v as u16;
                }
            }
            let mut want = plane.clone();
            for unit in 0..2 {
                if !on[unit] {
                    continue;
                }
                for lane in 4 * unit..4 * unit + 4 {
                    let at = |k: usize| pos + lane * along + k * step - 7 * step;
                    let mut one: [i32; 14] = std::array::from_fn(|k| plane[at(k)] as i32);
                    reference(&mut one, &params[unit]);
                    let changed = (0..14).filter(|&k| one[k] != plane[at(k)] as i32).count();
                    took[changed.div_ceil(4)] += 1;
                    for (k, v) in one.into_iter().enumerate() {
                        want[at(k)] = v as u16;
                    }
                }
            }
            let check = |kernel: &str, got: &[u16]| {
                if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                    panic!(
                        "case {case}, {kernel}, {:?} {on:?}, vertical {vertical}: \
                         sample ({}, {}) is {}, the process gives {} (it was {})",
                        params,
                        i % W,
                        i / W,
                        got[i],
                        want[i],
                        plane[i]
                    );
                }
            };
            // A sample at a time, each unit by itself.
            let mut got = plane.clone();
            for unit in 0..2 {
                if on[unit] {
                    let pos = pos + 4 * unit * along;
                    // SAFETY: the plane is `got`'s and nothing else has it.
                    unsafe {
                        filter_unit(got.as_mut_ptr(), got.len(), pos, step, along, &params[unit])
                    };
                }
            }
            check("filter_unit", &got);
            // Eight lanes at once.
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "wasm32", target_feature = "simd128")
            ))]
            {
                let pair = Pair {
                    limits,
                    on,
                    filter_size,
                    chroma,
                    bit_depth: bd,
                };
                let mut got = plane.clone();
                // SAFETY: the plane is `got`'s and nothing else has it.
                unsafe { filter_pair(got.as_mut_ptr(), got.len(), pos, W, vertical, &pair) };
                check("filter_pair", &got);
            }
        }
        // Lanes left alone, and lanes through the narrow filter and each
        // wide one (which change at most 4, 8 and 12 samples).
        assert!(took.iter().all(|&n| n > 15000), "{took:?}");
    }

    /// The kernels refuse an edge whose samples are not all inside the
    /// plane, and take one whose samples just are: for every filter length
    /// and direction, the plane that ends with the last sample a lane may
    /// read is filtered, and neither that plane less its last sample nor
    /// the same edge one sample nearer the start is.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn in_plane_kernels_check_their_bounds() {
        const W: usize = 32;
        let refuses =
            |f: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
        let mut failures = Vec::new();
        for (filter_size, chroma, reach) in
            [(4, false, 2), (8, true, 3), (8, false, 4), (16, false, 7)]
        {
            for vertical in [false, true] {
                let (step, along) = if vertical { (1, W) } else { (W, 1) };
                let params = Params {
                    limit: 4,
                    blimit: 40,
                    thresh: 1,
                    filter_size,
                    chroma,
                    bit_depth: 8,
                };
                // A sample at a time: four lanes, `reach` samples on each
                // side of the edge.
                let pos = reach * step;
                let len = pos + 3 * along + (reach - 1) * step + 1;
                let run = |pos: usize, len: usize| {
                    let mut plane = vec![128u16; len];
                    // SAFETY: the plane is this call's, and the kernel
                    // checks `pos` against `len` before it touches it.
                    unsafe { filter_unit(plane.as_mut_ptr(), len, pos, step, along, &params) }
                };
                let ok = !refuses(&|| run(pos, len))
                    && refuses(&|| run(pos, len - 1))
                    && refuses(&|| run(pos - 1, len - 1));
                if !ok {
                    failures.push(format!("filter_unit {filter_size} {chroma} {vertical}"));
                }
                // Eight lanes: whole rows of eight samples across a
                // horizontal edge, whole blocks of eight columns (two
                // blocks for the 16-sample size) across a vertical one.
                #[cfg(any(
                    target_arch = "aarch64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                ))]
                {
                    let pair = Pair {
                        limits: [[4, 40, 1]; 2],
                        on: [true, true],
                        filter_size,
                        chroma,
                        bit_depth: 8,
                    };
                    let columns = if filter_size == 16 { 8 } else { 4 };
                    let (pos, len) = if vertical {
                        (columns, columns + 7 * W + columns)
                    } else {
                        (reach * W, reach * W + (reach - 1) * W + 8)
                    };
                    let run = |pos: usize, len: usize| {
                        let mut plane = vec![128u16; len];
                        // SAFETY: as above.
                        unsafe { filter_pair(plane.as_mut_ptr(), len, pos, W, vertical, &pair) }
                    };
                    let ok = !refuses(&|| run(pos, len))
                        && refuses(&|| run(pos, len - 1))
                        && refuses(&|| run(pos - 1, len - 1));
                    if !ok {
                        failures.push(format!("filter_pair {filter_size} {chroma} {vertical}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }
}
