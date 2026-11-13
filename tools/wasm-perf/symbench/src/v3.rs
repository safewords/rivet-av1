//! V3: V1's window, with the 2-, 3- and 4-symbol reads written without data-dependent branches:
//! every threshold computed, the symbol counted, the adaptation selected per entry.
//! SIMD = true: the 4-symbol adaptation as one 4-lane vector operation.
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, update_cdf};
use crate::v1;
pub struct D<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool>(v1::D<'a>);

#[inline(always)]
fn thr(r: u32, c: u16, k: u32) -> u32 {
    ((r * ((32768 - c as u32) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * k
}
#[inline(always)]
fn adapt(c: u16, below: bool, rate: u32) -> u16 {
    if below { c - (c >> rate) } else { c + ((32768 - c as u32) >> rate) as u16 }
}
impl<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool> D<'a, SEARCH, ADAPT, BIT> {
    #[inline(always)]
    fn finish(&mut self, prev: u32, cur: u32) {
        let d = &mut self.0;
        d.rng = prev - cur;
        d.dif -= (cur as u64) << 48;
        d.renorm();
    }
    /// The search over a CDF of N - 1 symbols: (symbol, prev, cur).
    #[inline(always)]
    fn search<const N: usize>(&self, cdf: &[u16; N]) -> (usize, u32, u32) {
        let n = N - 1;
        let v = (self.0.dif >> 48) as u32;
        let r = self.0.rng >> 8;
        match SEARCH {
            0 => {
                let mut cur = self.0.rng;
                let mut symbol: usize = 0;
                let mut prev;
                loop {
                    prev = cur;
                    cur = thr(r, cdf[symbol], (n - symbol - 1) as u32);
                    if v >= cur {
                        break;
                    }
                    symbol += 1;
                }
                (symbol, prev, cur)
            }
            1 => {
                let mut t = [0u32; 6];
                t[0] = self.0.rng;
                let mut s = 0;
                for i in 0..n - 1 {
                    t[i + 1] = thr(r, cdf[i], (n - i - 1) as u32);
                    s += (v < t[i + 1]) as usize;
                }
                (s, t[s], t[s + 1])
            }
            _ => {
                // Selects, from the last symbol backwards.
                let mut s = 0;
                let mut cur = 0;
                let mut prev = 0;
                // candidates: symbol i has (prev, cur) = (t[i], t[i+1]); walk i from n-1 down to 0
                let mut next = 0u32; // t[i+1]
                for i in (0..n).rev() {
                    let ti = if i == 0 { self.0.rng } else { thr(r, cdf[i - 1], (n - i) as u32) };
                    // symbol >= i iff i == 0 or v < t[i]
                    if i == n - 1 {
                        s = i; prev = ti; cur = next;
                    }
                    let _ = (ti, next);
                    if i < n - 1 {
                        // symbol is exactly i (so far, scanning downwards) iff !(v < t[i+1])
                        let here = v >= next;
                        s = if here { i } else { s };
                        prev = if here { ti } else { prev };
                        cur = if here { next } else { cur };
                    }
                    next = ti;
                }
                (s, prev, cur)
            }
        }
    }
    #[inline(always)]
    fn update<const N: usize>(&self, cdf: &mut [u16; N], s: usize) {
        if self.0.disable_update {
            return;
        }
        let n = N - 1;
        match ADAPT {
            0 => update_cdf(cdf, s),
            _ => {
                let cnt = cdf[n];
                let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + (31 - (n as u32).leading_zeros()).min(2);
                if ADAPT == 2 && n == 4 {
                    adapt4((&mut cdf[..5]).try_into().unwrap(), s, rate);
                } else {
                    for i in 0..n - 1 {
                        cdf[i] = adapt(cdf[i], s > i, rate);
                    }
                }
                cdf[n] = cnt + (cnt < 32) as u16;
            }
        }
    }
    #[inline(always)]
    fn fixed<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        let (s, prev, cur) = self.search(cdf);
        self.finish(prev, cur);
        self.update(cdf, s);
        s
    }
}
impl<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool> crate::SymDec<'a> for D<'a, SEARCH, ADAPT, BIT> {
    fn new(data: &'a [u8]) -> Self {
        D(v1::D::new(data))
    }
    #[inline(always)]
    fn sym(&mut self, cdf: &mut [u16]) -> usize {
        self.0.sym(cdf)
    }
    #[inline(always)]
    fn sym2(&mut self, cdf: &mut [u16; 3]) -> usize {
        self.fixed(cdf)
    }
    #[inline(always)]
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize {
        self.fixed(cdf)
    }
    #[inline(always)]
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize {
        self.fixed(cdf)
    }
    #[inline(always)]
    fn bit(&mut self) -> u32 {
        if !BIT {
            return self.0.bit();
        }
        let v = (self.0.dif >> 48) as u32;
        let cur = ((self.0.rng >> 8) << 7) + EC_MIN_PROB;
        let bit = v < cur;
        let (prev, c) = if bit { (cur, 0) } else { (self.0.rng, cur) };
        self.finish(prev, c);
        bit as u32
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn adapt4(cdf: &mut [u16; 5], s: usize, rate: u32) {
    use core::arch::wasm32::*;
    // SAFETY: cdf has five entries; the first four are loaded and stored.
    unsafe {
        let c = v128_load64_zero(cdf.as_ptr() as *const u64);
        let below = u16x8_lt(u16x8(0, 1, 2, 3, 0, 0, 0, 0), u16x8_splat(s as u16));
        let dn = u16x8_sub(c, u16x8_shr(c, rate));
        let up = u16x8_add(c, u16x8_shr(u16x8_sub(u16x8_splat(32768), c), rate));
        let o = v128_bitselect(dn, up, below);
        v128_store64_lane::<0>(o, cdf.as_mut_ptr() as *mut u64);
    }
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn adapt4(cdf: &mut [u16; 5], s: usize, rate: u32) {
    use core::arch::aarch64::*;
    // SAFETY: cdf has five entries; the first four are loaded and stored.
    unsafe {
        let c = vld1_u16(cdf.as_ptr());
        let idx = vcreate_u16(0x0003_0002_0001_0000);
        let below = vclt_u16(idx, vdup_n_u16(s as u16));
        let sh = vdup_n_s16(-(rate as i16));
        let dn = vsub_u16(c, vshl_u16(c, sh));
        let up = vadd_u16(c, vshl_u16(vsub_u16(vdup_n_u16(32768), c), sh));
        vst1_u16(cdf.as_mut_ptr(), vbsl_u16(below, dn, up));
    }
}
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
#[inline(always)]
fn adapt4(cdf: &mut [u16; 5], s: usize, rate: u32) {
    cdf[0] = adapt(cdf[0], s > 0, rate);
    cdf[1] = adapt(cdf[1], s > 1, rate);
    cdf[2] = adapt(cdf[2], s > 2, rate);
    let _ = update_cdf;
}
