//! The inverse transforms of 7.13: DCT 4..64, ADST 4/8/16, identity, the
//! Walsh-Hadamard transform, and the 2D inverse transform process, as the
//! butterfly steps of 7.13.2 (bit-exact).

use crate::consts::*;
use crate::tables::{COS128_LOOKUP, TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

#[inline]
fn cos128(angle: i32) -> i64 {
    let a = (angle & 255) as usize;
    (if a <= 64 {
        COS128_LOOKUP[a]
    } else if a <= 128 {
        -COS128_LOOKUP[128 - a]
    } else if a <= 192 {
        -COS128_LOOKUP[a - 128]
    } else {
        COS128_LOOKUP[256 - a]
    }) as i64
}

#[inline]
fn sin128(angle: i32) -> i64 {
    cos128(angle - 64)
}

#[inline]
fn round2_12(x: i64) -> i32 {
    ((x + (1 << 11)) >> 12) as i32
}

/// What the transforms work on: one value (`i32`, the reference, exact
/// for any input), or several independent ones processed together
/// ([`Lanes`], SIMD by way of the compiler).
pub(crate) trait Lane: Copy {
    const N: usize;
    fn zero() -> Self;
    /// `Round2( a * ca + b * cb, 12 )`.
    fn rot(a: Self, ca: i32, b: Self, cb: i32) -> Self;
    /// `Clip( a + b )` and `Clip( a - b )` to `[lo, hi]`.
    fn add_sub_clamp(a: Self, b: Self, lo: i32, hi: i32) -> (Self, Self);
    fn neg(a: Self) -> Self;
    /// `Round2( a * c, 12 )`.
    fn scale12(a: Self, c: i32) -> Self;
    /// `a * c` (c = 2 or 4).
    fn mul(a: Self, c: i32) -> Self;
    fn get(&self, l: usize) -> i32;
    fn set(&mut self, l: usize, v: i32);
}

impl Lane for i32 {
    const N: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        0
    }
    #[inline(always)]
    fn rot(a: Self, ca: i32, b: Self, cb: i32) -> Self {
        round2_12(a as i64 * ca as i64 + b as i64 * cb as i64)
    }
    #[inline(always)]
    fn add_sub_clamp(a: Self, b: Self, lo: i32, hi: i32) -> (Self, Self) {
        let (x, y) = (a as i64, b as i64);
        (
            (x + y).clamp(lo as i64, hi as i64) as i32,
            (x - y).clamp(lo as i64, hi as i64) as i32,
        )
    }
    #[inline(always)]
    fn neg(a: Self) -> Self {
        a.wrapping_neg()
    }
    #[inline(always)]
    fn scale12(a: Self, c: i32) -> Self {
        round2_12(a as i64 * c as i64)
    }
    #[inline(always)]
    fn mul(a: Self, c: i32) -> Self {
        a.wrapping_mul(c)
    }
    #[inline(always)]
    fn get(&self, _l: usize) -> i32 {
        *self
    }
    #[inline(always)]
    fn set(&mut self, _l: usize, v: i32) {
        *self = v;
    }
}

/// Eight independent values, transformed together: written lane by lane
/// in 32-bit arithmetic, which the compiler turns into vector code (AVX2
/// where the kernel is compiled for it, NEON on aarch64). 32 bits are
/// exact when the values stay in the ranges conformance requires at bit
/// depths up to 10 (`8 + BitDepth` bits: products of at most 2^17 * 2^13);
/// [`inverse_transform_2d`] uses it only there.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
pub(crate) struct Lanes(pub(crate) [i32; 8]);

impl Lane for Lanes {
    const N: usize = 8;
    #[inline(always)]
    fn zero() -> Self {
        Lanes([0; 8])
    }
    #[inline(always)]
    fn rot(a: Self, ca: i32, b: Self, cb: i32) -> Self {
        let mut o = [0i32; 8];
        for l in 0..8 {
            o[l] = a.0[l]
                .wrapping_mul(ca)
                .wrapping_add(b.0[l].wrapping_mul(cb))
                .wrapping_add(1 << 11)
                >> 12;
        }
        Lanes(o)
    }
    #[inline(always)]
    fn add_sub_clamp(a: Self, b: Self, lo: i32, hi: i32) -> (Self, Self) {
        let mut x = [0i32; 8];
        let mut y = [0i32; 8];
        for l in 0..8 {
            x[l] = a.0[l].wrapping_add(b.0[l]).clamp(lo, hi);
            y[l] = a.0[l].wrapping_sub(b.0[l]).clamp(lo, hi);
        }
        (Lanes(x), Lanes(y))
    }
    #[inline(always)]
    fn neg(a: Self) -> Self {
        let mut o = [0i32; 8];
        for l in 0..8 {
            o[l] = a.0[l].wrapping_neg();
        }
        Lanes(o)
    }
    #[inline(always)]
    fn scale12(a: Self, c: i32) -> Self {
        let mut o = [0i32; 8];
        for l in 0..8 {
            o[l] = a.0[l].wrapping_mul(c).wrapping_add(1 << 11) >> 12;
        }
        Lanes(o)
    }
    #[inline(always)]
    fn mul(a: Self, c: i32) -> Self {
        let mut o = [0i32; 8];
        for l in 0..8 {
            o[l] = a.0[l].wrapping_mul(c);
        }
        Lanes(o)
    }
    #[inline(always)]
    fn get(&self, l: usize) -> i32 {
        self.0[l]
    }
    #[inline(always)]
    fn set(&mut self, l: usize, v: i32) {
        self.0[l] = v;
    }
}

/// The working array `T` and the butterflies `B` and `H` (7.13.2.1).
struct Tx<'a, L: Lane> {
    t: &'a mut [L],
}

impl<L: Lane> Tx<'_, L> {
    #[inline(always)]
    fn b(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let ta = self.t[a];
        let tb = self.t[b];
        let (c, s) = (cos128(angle) as i32, sin128(angle) as i32);
        let x = L::rot(ta, c, tb, -s);
        let y = L::rot(ta, s, tb, c);
        if flip {
            self.t[a] = y;
            self.t[b] = x;
        } else {
            self.t[a] = x;
            self.t[b] = y;
        }
    }

    #[inline(always)]
    fn h(&mut self, a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let lo = (-(1i64 << (r - 1))).max(i32::MIN as i64) as i32;
        let hi = ((1i64 << (r - 1)) - 1).min(i32::MAX as i64) as i32;
        let (x, y) = L::add_sub_clamp(self.t[a], self.t[b], lo, hi);
        self.t[a] = x;
        self.t[b] = y;
    }

    /// The inverse DCT array permutation (7.13.2.2).
    #[inline(always)]
    fn dct_permute(&mut self, n: u32) {
        let len = 1usize << n;
        let mut copy = [L::zero(); 64];
        copy[..len].copy_from_slice(&self.t[..len]);
        for i in 0..len {
            self.t[i] = copy[brev(n, i)];
        }
    }

    /// The inverse DCT process (7.13.2.3).
    #[inline(always)]
    fn idct(&mut self, n: u32, r: u32) {
        self.dct_permute(n);
        if n == 6 {
            for i in 0..16 {
                self.b(32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.b(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i32) << 3), false);
            }
        }
        if n == 6 {
            for i in 0..16 {
                self.h(32 + i * 2, 33 + i * 2, i & 1 != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.b(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i32) << 4), false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.h(16 + 2 * i, 17 + 2 * i, i & 1 != 0, r);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..2 {
                    self.b(
                        62 - i * 4 - j,
                        33 + i * 4 + j,
                        60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                        true,
                    );
                }
            }
        }
        if n >= 3 {
            for i in 0..2 {
                self.b(4 + i, 7 - i, 56 - 32 * i as i32, false);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.h(8 + 2 * i, 9 + 2 * i, i & 1 != 0, r);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..2 {
                    self.b(
                        30 - 4 * i - j,
                        17 + 4 * i + j,
                        24 + ((j as i32) << 6) + ((1 - i as i32) << 5),
                        true,
                    );
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                for j in 0..2 {
                    self.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1 != 0, r);
                }
            }
        }
        for i in 0..2 {
            self.b(2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
        }
        if n >= 3 {
            for i in 0..2 {
                self.h(4 + 2 * i, 5 + 2 * i, i != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(14 - i, 9 + i, 48 + 64 * i as i32, true);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                for j in 0..2 {
                    self.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..2 {
                for j in 0..4 {
                    self.b(
                        61 - i * 8 - j,
                        34 + i * 8 + j,
                        56 - i as i32 * 32 + ((j >> 1) as i32) * 64,
                        true,
                    );
                }
            }
        }
        for i in 0..2 {
            self.h(i, 3 - i, false, r);
        }
        if n >= 3 {
            self.b(6, 5, 32, true);
        }
        if n >= 4 {
            for i in 0..2 {
                for j in 0..2 {
                    self.h(8 + 4 * i + j, 11 + 4 * i - j, i != 0, r);
                }
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, true);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..4 {
                    self.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n >= 3 {
            for i in 0..4 {
                self.h(i, 7 - i, false, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(13 - i, 10 + i, 32, true);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..4 {
                    self.h(16 + i * 8 + j, 23 + i * 8 - j, i != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            }
        }
        if n >= 4 {
            for i in 0..8 {
                self.h(i, 15 - i, false, r);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(27 - i, 20 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.h(32 + i, 47 - i, false, r);
                self.h(48 + i, 63 - i, true, r);
            }
        }
        if n >= 5 {
            for i in 0..16 {
                self.h(i, 31 - i, false, r);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(55 - i, 40 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..32 {
                self.h(i, 63 - i, false, r);
            }
        }
    }

    /// The inverse ADST input array permutation (7.13.2.4).
    #[inline(always)]
    fn adst_in_permute(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [L::zero(); 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let idx = if i & 1 != 0 { i - 1 } else { n0 - i - 1 };
            self.t[i] = copy[idx];
        }
    }

    /// The inverse ADST output array permutation (7.13.2.5).
    #[inline(always)]
    fn adst_out_permute(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [L::zero(); 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let a = (i >> 3) & 1;
            let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
            let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
            let d = (i & 1) ^ ((i >> 1) & 1);
            let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
            self.t[i] = if i & 1 != 0 {
                L::neg(copy[idx])
            } else {
                copy[idx]
            };
        }
    }

    /// The inverse ADST4 process (7.13.2.6), lane by lane in 64 bits.
    #[inline(always)]
    fn iadst4(&mut self) {
        for l in 0..L::N {
            let mut v = [
                self.t[0].get(l),
                self.t[1].get(l),
                self.t[2].get(l),
                self.t[3].get(l),
            ];
            iadst4(&mut v);
            for (i, x) in v.iter().enumerate() {
                self.t[i].set(l, *x);
            }
        }
    }

    /// The inverse ADST8 process (7.13.2.7).
    #[inline(always)]
    fn iadst8(&mut self, r: u32) {
        self.adst_in_permute(3);
        for i in 0..4 {
            self.b(2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
        }
        for i in 0..4 {
            self.h(i, 4 + i, false, r);
        }
        for i in 0..2 {
            self.b(4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
        }
        for i in 0..2 {
            for j in 0..2 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..2 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_out_permute(3);
    }

    /// The inverse ADST16 process (7.13.2.8).
    #[inline(always)]
    fn iadst16(&mut self, r: u32) {
        self.adst_in_permute(4);
        for i in 0..8 {
            self.b(2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
        }
        for i in 0..8 {
            self.h(i, 8 + i, false, r);
        }
        for i in 0..2 {
            self.b(8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
            self.b(13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
        }
        for i in 0..4 {
            for j in 0..2 {
                self.h(8 * j + i, 4 + 8 * j + i, false, r);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.b(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..4 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_out_permute(4);
    }

    #[inline(always)]
    fn iadst(&mut self, n: u32, r: u32) {
        match n {
            2 => self.iadst4(),
            3 => self.iadst8(r),
            _ => self.iadst16(r),
        }
    }

    /// The inverse identity transforms (7.13.2.11 to 7.13.2.15).
    #[inline(always)]
    fn iidentity(&mut self, n: u32) {
        let len = 1usize << n;
        for v in self.t[..len].iter_mut() {
            *v = match n {
                2 => L::scale12(*v, 5793),
                3 => L::mul(*v, 2),
                4 => L::scale12(*v, 11586),
                _ => L::mul(*v, 4),
            };
        }
    }
}

impl Tx<'_, i32> {
    /// The inverse Walsh-Hadamard transform (7.13.2.10).
    fn iwht(&mut self, shift: u32) {
        let mut a = self.t[0] >> shift;
        let mut c = self.t[1] >> shift;
        let mut d = self.t[2] >> shift;
        let mut b = self.t[3] >> shift;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        self.t[0] = a;
        self.t[1] = b;
        self.t[2] = c;
        self.t[3] = d;
    }
}

/// The inverse ADST4 process (7.13.2.6) on one set of four values.
#[inline(always)]
fn iadst4(v: &mut [i32; 4]) {
    let t: [i64; 4] = [v[0] as i64, v[1] as i64, v[2] as i64, v[3] as i64];
    const SINPI_1_9: i64 = 1321;
    const SINPI_2_9: i64 = 2482;
    const SINPI_3_9: i64 = 3344;
    const SINPI_4_9: i64 = 3803;
    let mut s = [0i64; 7];
    s[0] = SINPI_1_9 * t[0];
    s[1] = SINPI_2_9 * t[0];
    s[2] = SINPI_3_9 * t[1];
    s[3] = SINPI_4_9 * t[2];
    s[4] = SINPI_1_9 * t[2];
    s[5] = SINPI_2_9 * t[3];
    s[6] = SINPI_4_9 * t[3];
    let a7 = t[0] - t[2];
    let b7 = a7 + t[3];
    s[0] += s[3];
    s[1] -= s[4];
    s[3] = s[2];
    s[2] = SINPI_3_9 * b7;
    s[0] += s[5];
    s[1] -= s[6];
    let x0 = s[0] + s[3];
    let x1 = s[1] + s[3];
    let x2 = s[2];
    let x3 = s[0] + s[1] - s[3];
    v[0] = round2_12(x0);
    v[1] = round2_12(x1);
    v[2] = round2_12(x2);
    v[3] = round2_12(x3);
}

/// Which 1D transform a direction uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dct,
    Adst,
    Identity,
}

fn row_kind(tx_type: usize) -> Kind {
    match tx_type {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => Kind::Dct,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | H_ADST | H_FLIPADST => Kind::Adst,
        _ => Kind::Identity,
    }
}

fn col_kind(tx_type: usize) -> Kind {
    match tx_type {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => Kind::Dct,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | V_ADST | V_FLIPADST => Kind::Adst,
        _ => Kind::Identity,
    }
}

fn run(t: &mut [i32], kind: Kind, n: u32, r: u32, lossless: bool, wht_shift: u32) {
    let mut tx = Tx { t };
    if lossless {
        tx.iwht(wht_shift);
        return;
    }
    run_lanes(tx.t, kind, n, r);
}

#[inline(always)]
fn run_lanes<L: Lane>(t: &mut [L], kind: Kind, n: u32, r: u32) {
    let mut tx = Tx { t };
    // Each size on its own. The processes of 7.13.2 are written for any
    // size, and compiled that way they test the size at every stage, find
    // the permutation's indices bit by bit as they run and look their
    // angles up; with the size a constant where each is compiled in, all
    // of that is decided there.
    match (kind, n) {
        (Kind::Dct, 2) => tx.idct(2, r),
        (Kind::Dct, 3) => tx.idct(3, r),
        (Kind::Dct, 4) => tx.idct(4, r),
        (Kind::Dct, 5) => tx.idct(5, r),
        (Kind::Dct, _) => tx.idct(6, r),
        (Kind::Adst, 2) => tx.iadst(2, r),
        (Kind::Adst, 3) => tx.iadst(3, r),
        (Kind::Adst, _) => tx.iadst(4, r),
        (Kind::Identity, 2) => tx.iidentity(2),
        (Kind::Identity, 3) => tx.iidentity(3),
        (Kind::Identity, 4) => tx.iidentity(4),
        (Kind::Identity, _) => tx.iidentity(5),
    }
}

/// One 1D inverse transform, without intermediate clamping (`r` wide
/// enough): 0 DCT, 1 ADST, 2 identity, of length `1 << n`. The encoder
/// derives its forward transforms from these.
pub(crate) fn inverse_1d(kind: u8, n: u32, t: &mut [i32]) {
    let k = match kind {
        0 => Kind::Dct,
        1 => Kind::Adst,
        _ => Kind::Identity,
    };
    run(t, k, n, 30, false, 0);
}

/// Which 1D transforms a type applies: `(rows, columns)`, 0 DCT, 1 ADST,
/// 2 identity.
pub(crate) fn kinds(tx_type: usize) -> (u8, u8) {
    let f = |k: Kind| match k {
        Kind::Dct => 0,
        Kind::Adst => 1,
        Kind::Identity => 2,
    };
    (f(row_kind(tx_type)), f(col_kind(tx_type)))
}

/// The residual of a DCT_DCT block whose only coefficient is the DC one:
/// every column is the same, and it is returned (`h` entries). The 2D
/// process (7.13.3) on that input, with what is zero left out: the one row
/// that is not zero through the row transform, and, its outputs being all
/// equal, one column through the column transform. `None` if the row's
/// outputs are not all equal (they are, for a DCT; this checks rather than
/// assumes) and the caller takes the general path.
pub(crate) fn inverse_dct_dc_only(dc: i32, tx_sz: usize, bit_depth: u32) -> Option<[i32; 64]> {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let (w, h) = (1usize << log2w, 1usize << log2h);
    let row_shift = TRANSFORM_ROW_SHIFT[tx_sz] as u32;
    let row_clamp = bit_depth + 8;
    let col_clamp = (bit_depth + 6).max(16);
    let mut t = [0i32; 64];
    t[0] = if log2w.abs_diff(log2h) == 1 {
        round2_12(dc as i64 * 2896)
    } else {
        dc
    };
    run(&mut t, Kind::Dct, log2w, row_clamp, false, 0);
    if t[1..w].iter().any(|&v| v != t[0]) {
        return None;
    }
    let lo = -(1i32 << (col_clamp - 1));
    let hi = (1i32 << (col_clamp - 1)) - 1;
    let mut c = [0i32; 64];
    c[0] = round2(t[0], row_shift).clamp(lo, hi);
    run(&mut c, Kind::Dct, log2h, col_clamp, false, 0);
    for v in c[..h].iter_mut() {
        *v = round2(*v, 4);
    }
    Some(c)
}

/// The 2D inverse transform process (7.13.3). `dequant` is the
/// `Dequant` array with a row stride of 64 (only its top-left 32x32 can be
/// non-zero); the result is written to `residual` with a row stride of `w`.
pub(crate) fn inverse_transform_2d(
    dequant: &[i32],
    tx_sz: usize,
    tx_type: usize,
    lossless: bool,
    bit_depth: u32,
    residual: &mut [i32],
) {
    if !lossless && bit_depth <= 10 {
        #[cfg(target_arch = "x86_64")]
        if crate::dsp::avx2() {
            // SAFETY: AVX2 is available.
            unsafe { inverse_transform_2d_avx2(dequant, tx_sz, tx_type, bit_depth, residual) };
            return;
        }
        // NEON is part of the aarch64 baseline, and a wasm module built
        // with `simd128` has it: the lanes compile to vectors on both.
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        {
            inverse_transform_2d_lanes(dequant, tx_sz, tx_type, bit_depth, residual);
            return;
        }
    }
    inverse_transform_2d_scalar(dequant, tx_sz, tx_type, lossless, bit_depth, residual)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inverse_transform_2d_avx2(
    dequant: &[i32],
    tx_sz: usize,
    tx_type: usize,
    bit_depth: u32,
    residual: &mut [i32],
) {
    inverse_transform_2d_lanes(dequant, tx_sz, tx_type, bit_depth, residual)
}

/// The 2D inverse transform eight rows, then eight columns, at a time
/// (not lossless; bit depth 10 at most). Bit-exact with the scalar
/// version for the values conformance allows.
#[inline(always)]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn inverse_transform_2d_lanes(
    dequant: &[i32],
    tx_sz: usize,
    tx_type: usize,
    bit_depth: u32,
    residual: &mut [i32],
) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let row_shift = TRANSFORM_ROW_SHIFT[tx_sz] as u32;
    let row_clamp = bit_depth + 8;
    let col_clamp = (bit_depth + 6).max(16);
    let rk = row_kind(tx_type);
    let ck = col_kind(tx_type);
    let rect = log2w.abs_diff(log2h) == 1;
    let nz = w.min(32);
    // Rows, eight at a time (four for the 4-high sizes, one by one).
    let mut i0 = 0;
    while i0 < h {
        if i0 >= 32 {
            residual[i0 * w..h * w].fill(0);
            break;
        }
        if h - i0 < 8 {
            let mut t = [0i32; 64];
            for i in i0..h {
                let src = &dequant[i * 64..i * 64 + 32];
                let out = &mut residual[i * w..(i + 1) * w];
                if src[..nz].iter().all(|&v| v == 0) {
                    out.fill(0);
                    continue;
                }
                t[..w].fill(0);
                t[..nz].copy_from_slice(&src[..nz]);
                if rect {
                    for v in t[..w].iter_mut() {
                        *v = round2_12(*v as i64 * 2896);
                    }
                }
                run_lanes(&mut t[..], rk, log2w, row_clamp);
                for j in 0..w {
                    out[j] = round2(t[j], row_shift);
                }
            }
            break;
        }
        let mut t = [Lanes::zero(); 64];
        let mut any = false;
        for l in 0..8 {
            let src = &dequant[(i0 + l) * 64..(i0 + l) * 64 + nz];
            for (j, &v) in src.iter().enumerate() {
                t[j].0[l] = v;
                any |= v != 0;
            }
        }
        if !any {
            residual[i0 * w..(i0 + 8) * w].fill(0);
            i0 += 8;
            continue;
        }
        if rect {
            for v in t[..w].iter_mut() {
                *v = Lanes::scale12(*v, 2896);
            }
        }
        run_lanes(&mut t[..], rk, log2w, row_clamp);
        let add = (1i32 << row_shift) >> 1;
        for l in 0..8 {
            let out = &mut residual[(i0 + l) * w..(i0 + l + 1) * w];
            for j in 0..w {
                out[j] = t[j].0[l].wrapping_add(add) >> row_shift;
            }
        }
        i0 += 8;
    }
    let lo = -(1i32 << (col_clamp - 1));
    let hi = (1i32 << (col_clamp - 1)) - 1;
    for v in residual[..w * h].iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
    // Columns, eight at a time (four-wide sizes one by one).
    if w < 8 {
        let mut t = [0i32; 64];
        for j in 0..w {
            for i in 0..h {
                t[i] = residual[i * w + j];
            }
            run_lanes(&mut t[..], ck, log2h, col_clamp);
            for i in 0..h {
                residual[i * w + j] = round2(t[i], 4);
            }
        }
        return;
    }
    let mut j0 = 0;
    while j0 < w {
        let mut t = [Lanes::zero(); 64];
        for i in 0..h {
            t[i].0
                .copy_from_slice(&residual[i * w + j0..i * w + j0 + 8]);
        }
        run_lanes(&mut t[..], ck, log2h, col_clamp);
        for i in 0..h {
            let out = &mut residual[i * w + j0..i * w + j0 + 8];
            for l in 0..8 {
                out[l] = t[i].0[l].wrapping_add(8) >> 4;
            }
        }
        j0 += 8;
    }
}

/// The 2D inverse transform process, one row or column at a time: the
/// reference (and the lossless and 12-bit path).
pub(crate) fn inverse_transform_2d_scalar(
    dequant: &[i32],
    tx_sz: usize,
    tx_type: usize,
    lossless: bool,
    bit_depth: u32,
    residual: &mut [i32],
) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let row_shift = if lossless {
        0
    } else {
        TRANSFORM_ROW_SHIFT[tx_sz] as u32
    };
    let col_shift = if lossless { 0 } else { 4 };
    let row_clamp = bit_depth + 8;
    let col_clamp = (bit_depth + 6).max(16);
    let rk = row_kind(tx_type);
    let ck = col_kind(tx_type);
    let rect = log2w.abs_diff(log2h) == 1;
    let mut t = [0i32; 64];
    for i in 0..h {
        let out = &mut residual[i * w..(i + 1) * w];
        if i >= 32 {
            out.fill(0);
            continue;
        }
        let src = &dequant[i * 64..i * 64 + 32];
        let nz = w.min(32);
        if !lossless && src[..nz].iter().all(|&v| v == 0) {
            out.fill(0);
            continue;
        }
        for j in 0..w {
            t[j] = if j < 32 { src[j] } else { 0 };
        }
        if rect {
            for v in t[..w].iter_mut() {
                *v = round2_12(*v as i64 * 2896);
            }
        }
        run(&mut t, rk, log2w, row_clamp, lossless, 2);
        for j in 0..w {
            out[j] = round2(t[j], row_shift);
        }
    }
    let lo = -(1i32 << (col_clamp - 1));
    let hi = (1i32 << (col_clamp - 1)) - 1;
    for v in residual[..w * h].iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
    for j in 0..w {
        for i in 0..h {
            t[i] = residual[i * w + j];
        }
        run(&mut t, ck, log2h, col_clamp, lossless, 0);
        for i in 0..h {
            residual[i * w + j] = round2(t[i], col_shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform eight rows / columns at a time (SIMD where available)
    /// agrees with the reference for every size and type, on coefficients
    /// in the conformance range at 8 and 10 bits.
    /// The DC-only shortcut gives the general process's residual, for
    /// every size and DC values over the whole range at each depth.
    #[test]
    fn dc_only_matches_the_general_process() {
        for bd in [8u32, 10, 12] {
            for tx_sz in 0..TX_SIZES_ALL {
                let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
                let lim = 1i32 << (7 + bd);
                let mut dcs = vec![-lim, lim - 1, 0, 1, -1, 2, -3, 7, 64, -65];
                dcs.extend((0..200).map(|i| (i * 7919 % (2 * lim)) - lim));
                for dc in dcs {
                    let mut dequant = vec![0i32; 64 * 64];
                    dequant[0] = dc;
                    let mut want = vec![0i32; 64 * 64];
                    inverse_transform_2d_scalar(&dequant, tx_sz, DCT_DCT, false, bd, &mut want);
                    let col = inverse_dct_dc_only(dc, tx_sz, bd).expect("a DCT's row is flat");
                    for i in 0..h {
                        for j in 0..w {
                            assert_eq!(want[i * w + j], col[i], "{bd}-bit size {tx_sz} dc {dc}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lanes_match_scalar() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10] {
            for tx_sz in 0..TX_SIZES_ALL {
                for tx_type in 0..16 {
                    let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                    let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
                    let big = w.max(h) >= 64 || (w.max(h) == 32 && w.min(h) >= 16);
                    if big && tx_type != DCT_DCT && tx_type != IDTX {
                        continue;
                    }
                    if w.max(h) == 64 && tx_type != DCT_DCT {
                        continue;
                    }
                    for case in 0..20 {
                        let mut dq = vec![0i32; 64 * 64];
                        let n = 1 + rnd(16) as usize;
                        // A few coefficients, sometimes large, as a
                        // conformant stream may carry.
                        let scale = if case % 4 == 0 {
                            1 << (bd + 3)
                        } else {
                            1 << (bd - 2)
                        };
                        for _ in 0..n {
                            let i = rnd(h.min(32) as u64) as usize;
                            let j = rnd(w.min(32) as u64) as usize;
                            dq[i * 64 + j] = rnd(2 * scale) as i32 - scale as i32;
                        }
                        let mut a = vec![0i32; w * h];
                        let mut b = vec![0i32; w * h];
                        inverse_transform_2d_scalar(&dq, tx_sz, tx_type, false, bd, &mut a);
                        inverse_transform_2d(&dq, tx_sz, tx_type, false, bd, &mut b);
                        assert_eq!(a, b, "{bd}-bit size {tx_sz} type {tx_type} case {case}");
                        inverse_transform_2d_lanes(&dq, tx_sz, tx_type, bd, &mut b);
                        assert_eq!(a, b, "lanes, {bd}-bit size {tx_sz} type {tx_type}");
                    }
                }
            }
        }
    }

    /// The 1D transforms compiled for each size give what the processes of
    /// 7.13.2 give when called with the size as a variable, for every kind
    /// and size, one value at a time and eight.
    #[test]
    fn sizes_match_the_general_processes() {
        /// The dispatch as it was: one body for all sizes.
        #[inline(never)]
        fn general<L: Lane>(t: &mut [L], kind: Kind, n: u32, r: u32) {
            let n = std::hint::black_box(n);
            let mut tx = Tx { t };
            match kind {
                Kind::Dct => tx.idct(n, r),
                Kind::Adst => tx.iadst(n, r),
                Kind::Identity => tx.iidentity(n),
            }
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for (kind, sizes) in [
            (Kind::Dct, 2..=6u32),
            (Kind::Adst, 2..=4),
            (Kind::Identity, 2..=5),
        ] {
            for n in sizes {
                for r in [16u32, 18] {
                    for _ in 0..200 {
                        let mut a = [Lanes::zero(); 64];
                        for v in a[..1 << n].iter_mut() {
                            for l in 0..8 {
                                v.0[l] = rnd(1 << (r - 1)) as i32 - (1 << (r - 2));
                            }
                        }
                        let mut b = a;
                        run_lanes(&mut a[..], kind, n, r);
                        general(&mut b[..], kind, n, r);
                        let name = ["DCT", "ADST", "identity"][kind as usize];
                        for i in 0..1usize << n {
                            assert_eq!(a[i].0, b[i].0, "{name} of {} at {i}", 1 << n);
                        }
                        // One value at a time: the first lane of each.
                        let mut c = [0i32; 64];
                        for (c, a) in c.iter_mut().zip(&b) {
                            *c = a.0[0];
                        }
                        let mut d = c;
                        run_lanes(&mut c[..], kind, n, r);
                        general(&mut d[..], kind, n, r);
                        assert_eq!(c, d, "{name} of {}, one value at a time", 1 << n);
                    }
                }
            }
        }
    }

    /// A DC-only block reconstructs to a flat residual whose value is the
    /// DC scaled through both passes.
    #[test]
    fn dct_dc_is_flat() {
        for tx_sz in 0..TX_SIZES_ALL {
            let mut dq = vec![0i32; 64 * 64];
            dq[0] = 1024;
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            let mut res = vec![0i32; w * h];
            inverse_transform_2d(&dq, tx_sz, DCT_DCT, false, 8, &mut res);
            assert!(res.iter().all(|&v| v == res[0]), "tx {tx_sz}");
            assert!(res[0] > 0);
        }
    }

    /// The lossless WHT of a single DC coefficient is exactly invertible
    /// with the forward transform's scaling: a flat 4x4 block.
    #[test]
    fn wht_dc() {
        let mut dq = vec![0i32; 64 * 64];
        dq[0] = 4 * 16;
        let mut res = vec![0i32; 16];
        inverse_transform_2d(&dq, TX_4X4, DCT_DCT, true, 8, &mut res);
        assert!(res.iter().all(|&v| v == 4), "{res:?}");
    }
}
