//! V10: the window, state by value, and the 2-, 3- and 4-symbol reads with no branch on the data at all: every
//! threshold computed, each entry's adaptation chosen by its own comparison (both results computed first), the
//! interval by a tree of selects. `select_unpredictable` so that the compiler keeps them selects.
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, update_cdf};
use core::hint::select_unpredictable as sel;
pub struct D<'a> {
    data: &'a [u8],
    pos: usize,
    dif: u64,
    cnt: i32,
    rng: u32,
}
#[cold]
#[inline(never)]
fn refill_slow(data: &[u8], mut pos: usize, mut dif: u64, mut cnt: i32) -> (usize, u64, i32) {
    while cnt <= 40 {
        let b = data.get(pos).copied().unwrap_or(0);
        dif |= ((b ^ 0xFF) as u64) << (40 - cnt);
        pos += 1;
        cnt += 8;
    }
    (pos, dif, cnt)
}
#[inline(always)]
fn thr(r: u32, c: u16, k: u32) -> u32 {
    ((r * ((32768 - c as u32) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * k
}
/// The entry after adaptation: towards 0 if the symbol is above it, towards 32768 otherwise.
#[inline(always)]
fn adapt(c: u16, above: bool, rate: u32) -> u16 {
    sel(above, c - (c >> rate), c + ((32768 - c as u32) >> rate) as u16)
}
impl<'a> D<'a> {
    #[inline(always)]
    fn refill(&mut self) {
        if let Some(b) = self.data.get(self.pos..self.pos + 8) {
            let w = u64::from_be_bytes(b.try_into().unwrap());
            let nb = (48 - self.cnt) >> 3;
            self.dif |= !w >> (16 + self.cnt);
            self.pos += nb as usize;
            self.cnt += 8 * nb;
        } else {
            (self.pos, self.dif, self.cnt) = refill_slow(self.data, self.pos, self.dif, self.cnt);
        }
    }
    #[inline(always)]
    fn finish(&mut self, prev: u32, cur: u32) {
        self.rng = prev - cur;
        self.dif -= (cur as u64) << 48;
        let d = self.rng.leading_zeros() - 16;
        self.rng <<= d;
        self.dif <<= d;
        self.cnt -= d as i32;
        if self.cnt < 0 {
            self.refill();
        }
    }
}
impl<'a> crate::SymDec<'a> for D<'a> {
    #[inline(always)]
    fn new(data: &'a [u8]) -> Self {
        let mut d = D { data, pos: 0, dif: 0, cnt: -15, rng: 1 << 15 };
        d.refill();
        d
    }
    #[inline(always)]
    fn sym(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let mut cur = self.rng;
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
        self.finish(prev, cur);
        update_cdf(cdf, symbol);
        symbol
    }
    #[inline(always)]
    fn sym2(&mut self, cdf: &mut [u16; 3]) -> usize {
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let a0 = cdf[0];
        let c0 = thr(r, a0, 1);
        let b0 = v < c0;
        self.finish(sel(b0, c0, self.rng), sel(b0, 0, c0));
        let cnt = cdf[2];
        let rate = 4 + (cnt > 15) as u32 + (cnt > 31) as u32;
        cdf[0] = adapt(a0, b0, rate);
        cdf[2] = cnt + (cnt < 32) as u16;
        b0 as usize
    }
    #[inline(always)]
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize {
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let (a0, a1) = (cdf[0], cdf[1]);
        let (c0, c1) = (thr(r, a0, 2), thr(r, a1, 1));
        let (b0, b1) = (v < c0, v < c1);
        self.finish(sel(b0, sel(b1, c1, c0), self.rng), sel(b0, sel(b1, 0, c1), c0));
        let cnt = cdf[3];
        let rate = 4 + (cnt > 15) as u32 + (cnt > 31) as u32;
        cdf[0] = adapt(a0, b0, rate);
        cdf[1] = adapt(a1, b1, rate);
        cdf[3] = cnt + (cnt < 32) as u16;
        b0 as usize + b1 as usize
    }
    #[inline(always)]
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize {
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let (a0, a1, a2) = (cdf[0], cdf[1], cdf[2]);
        let (c0, c1, c2) = (thr(r, a0, 3), thr(r, a1, 2), thr(r, a2, 1));
        let (b0, b1, b2) = (v < c0, v < c1, v < c2);
        let prev = sel(b1, sel(b2, c2, c1), sel(b0, c0, self.rng));
        let cur = sel(b1, sel(b2, 0, c2), sel(b0, c1, c0));
        self.finish(prev, cur);
        let cnt = cdf[4];
        let rate = 5 + (cnt > 15) as u32 + (cnt > 31) as u32;
        cdf[0] = adapt(a0, b0, rate);
        cdf[1] = adapt(a1, b1, rate);
        cdf[2] = adapt(a2, b2, rate);
        cdf[4] = cnt + (cnt < 32) as u16;
        b0 as usize + b1 as usize + b2 as usize
    }
    #[inline(always)]
    fn bit(&mut self) -> u32 {
        let v = (self.dif >> 48) as u32;
        let cur = ((self.rng >> 8) << 7) + EC_MIN_PROB;
        let bit = v < cur;
        self.finish(sel(bit, cur, self.rng), sel(bit, 0, cur));
        bit as u32
    }
}
