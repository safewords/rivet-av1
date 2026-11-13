//! V8: V1's window with the decoder's state held in registers across the loop (nothing in the loop takes its
//! address: the cold refill gets and returns values). SEARCH 0: the loop; 2: selects. ADAPT 0: update_cdf's two
//! loops; 1: `if` per entry; 3: mask arithmetic (no branch the compiler can reintroduce).
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, update_cdf};
pub struct D<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool> {
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
#[inline(always)]
fn adapt(c: u16, below: bool, rate: u32) -> u16 {
    if below { c - (c >> rate) } else { c + ((32768 - c as u32) >> rate) as u16 }
}
/// The same without a condition: towards 0 when `below`, towards 32768 otherwise, as one step on the distance.
#[inline(always)]
fn adapt_mask(c: u16, below: bool, rate: u32) -> u16 {
    let m = (below as u32).wrapping_sub(1); // 0 below, all ones otherwise
    let x = (c as u32 ^ m).wrapping_add(32769 & m); // c, or 32768 - c
    let x = x - (x >> rate);
    (x ^ m).wrapping_add(32769 & m) as u16
}
impl<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool> D<'a, SEARCH, ADAPT, BIT> {
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
    #[inline(always)]
    fn search(&self, cdf: &[u16]) -> (usize, u32, u32) {
        let n = cdf.len() - 1;
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        if SEARCH == 0 {
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
            (symbol, prev, cur)
        } else {
            let (mut s, mut cur, mut prev, mut next) = (0, 0, 0, 0u32);
            for i in (0..n).rev() {
                let ti = if i == 0 { self.rng } else { thr(r, cdf[i - 1], (n - i) as u32) };
                if i == n - 1 {
                    s = i;
                    prev = ti;
                    cur = next;
                } else {
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
    #[inline(always)]
    fn update(&self, cdf: &mut [u16], s: usize) {
        let n = cdf.len() - 1;
        match ADAPT {
            0 => update_cdf(cdf, s),
            _ => {
                let cnt = cdf[n];
                let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + (31 - (n as u32).leading_zeros()).min(2);
                for i in 0..n - 1 {
                    cdf[i] = if ADAPT == 3 { adapt_mask(cdf[i], s > i, rate) } else { adapt(cdf[i], s > i, rate) };
                }
                cdf[n] = cnt + (cnt < 32) as u16;
            }
        }
    }
    #[inline(always)]
    fn read(&mut self, cdf: &mut [u16]) -> usize {
        let (s, prev, cur) = self.search(cdf);
        self.finish(prev, cur);
        self.update(cdf, s);
        s
    }
}
impl<'a, const SEARCH: u8, const ADAPT: u8, const BIT: bool> crate::SymDec<'a> for D<'a, SEARCH, ADAPT, BIT> {
    #[inline(always)]
    fn new(data: &'a [u8]) -> Self {
        let mut d = D { data, pos: 0, dif: 0, cnt: -15, rng: 1 << 15 };
        d.refill();
        d
    }
    #[inline(always)]
    fn sym(&mut self, cdf: &mut [u16]) -> usize {
        // The loop search and update_cdf, as the generic read.
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
        self.read(cdf)
    }
    #[inline(always)]
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize {
        self.read(cdf)
    }
    #[inline(always)]
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize {
        self.read(cdf)
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
