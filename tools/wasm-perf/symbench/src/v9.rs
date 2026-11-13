//! V9: the window, and the 2-, 3- and 4-symbol reads as one arm per symbol: the search's branch for a symbol
//! leads to that symbol's renormalisation and to its adaptation with every direction known (no loop over the
//! entries, no second test of the symbol). REGS: the state by value (v8) or through the struct (v1).
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, update_cdf};
pub struct D<'a, const BIT: bool> {
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
/// The adaptation of an N-1 symbol CDF for the symbol S: entries below S towards 0, the others towards 32768.
#[inline(always)]
fn adapt<const N: usize, const S: usize>(cdf: &mut [u16; N]) {
    let n = N - 1;
    let cnt = cdf[n];
    let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + (31 - (n as u32).leading_zeros()).min(2);
    let mut i = 0;
    while i < n - 1 {
        if i < S {
            cdf[i] -= cdf[i] >> rate;
        } else {
            cdf[i] += ((32768 - cdf[i] as u32) >> rate) as u16;
        }
        i += 1;
    }
    cdf[n] = cnt + (cnt < 32) as u16;
}
impl<'a, const BIT: bool> D<'a, BIT> {
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
impl<'a, const BIT: bool> crate::SymDec<'a> for D<'a, BIT> {
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
        let c0 = thr(r, cdf[0], 1);
        if v >= c0 {
            self.finish(self.rng, c0);
            adapt::<3, 0>(cdf);
            return 0;
        }
        self.finish(c0, 0);
        adapt::<3, 1>(cdf);
        1
    }
    #[inline(always)]
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize {
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let c0 = thr(r, cdf[0], 2);
        if v >= c0 {
            self.finish(self.rng, c0);
            adapt::<4, 0>(cdf);
            return 0;
        }
        let c1 = thr(r, cdf[1], 1);
        if v >= c1 {
            self.finish(c0, c1);
            adapt::<4, 1>(cdf);
            return 1;
        }
        self.finish(c1, 0);
        adapt::<4, 2>(cdf);
        2
    }
    #[inline(always)]
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize {
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let c0 = thr(r, cdf[0], 3);
        if v >= c0 {
            self.finish(self.rng, c0);
            adapt::<5, 0>(cdf);
            return 0;
        }
        let c1 = thr(r, cdf[1], 2);
        if v >= c1 {
            self.finish(c0, c1);
            adapt::<5, 1>(cdf);
            return 1;
        }
        let c2 = thr(r, cdf[2], 1);
        if v >= c2 {
            self.finish(c1, c2);
            adapt::<5, 2>(cdf);
            return 2;
        }
        self.finish(c2, 0);
        adapt::<5, 3>(cdf);
        3
    }
    #[inline(always)]
    fn bit(&mut self) -> u32 {
        let v = (self.dif >> 48) as u32;
        let cur = ((self.rng >> 8) << 7) + EC_MIN_PROB;
        if BIT {
            let bit = v < cur;
            let (prev, c) = if bit { (cur, 0) } else { (self.rng, cur) };
            self.finish(prev, c);
            bit as u32
        } else if v >= cur {
            self.finish(self.rng, cur);
            0
        } else {
            self.finish(cur, 0);
            1
        }
    }
}
