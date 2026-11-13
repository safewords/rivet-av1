//! V1: the same search and adaptation, the value in a 64-bit window: SymbolValue in the top 16 bits,
//! complemented stream bits below it, refilled several bytes at a time.
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, update_cdf};
pub struct D<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub dif: u64,
    pub cnt: i32,
    pub rng: u32,
    pub disable_update: bool,
}
impl<'a> D<'a> {
    #[cold]
    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.cnt <= 40 {
            let b = self.data.get(self.pos).copied().unwrap_or(0);
            self.dif |= ((b ^ 0xFF) as u64) << (40 - self.cnt);
            self.pos += 1;
            self.cnt += 8;
        }
    }
    #[inline(always)]
    pub fn refill(&mut self) {
        // cnt in -15..=-1
        if let Some(b) = self.data.get(self.pos..self.pos + 8) {
            let w = u64::from_be_bytes(b.try_into().unwrap());
            let nb = (48 - self.cnt) >> 3;
            self.dif |= !w >> (16 + self.cnt);
            self.pos += nb as usize;
            self.cnt += 8 * nb;
        } else {
            self.refill_slow();
        }
    }
    #[inline(always)]
    pub fn renorm(&mut self) {
        let d = self.rng.leading_zeros() - 16;
        self.rng <<= d;
        self.dif <<= d;
        self.cnt -= d as i32;
        if self.cnt < 0 {
            self.refill();
        }
    }
    #[inline(never)]
    fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn read_symbol_inline(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let v = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let mut cur = self.rng;
        let mut symbol: usize = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - cdf[symbol] as u32;
            cur = (r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if v >= cur {
                break;
            }
            symbol += 1;
        }
        self.rng = prev - cur;
        self.dif -= (cur as u64) << 48;
        self.renorm();
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
        symbol
    }
}
impl<'a> crate::SymDec<'a> for D<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut d = D { data, pos: 0, dif: 0, cnt: -15, rng: 1 << 15, disable_update: false };
        d.refill();
        d
    }
    #[inline(always)]
    fn sym(&mut self, cdf: &mut [u16]) -> usize {
        self.read_symbol(cdf)
    }
    #[inline(always)]
    fn sym2(&mut self, cdf: &mut [u16; 3]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn sym3(&mut self, cdf: &mut [u16; 4]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn sym4(&mut self, cdf: &mut [u16; 5]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn bit(&mut self) -> u32 {
        let v = (self.dif >> 48) as u32;
        let cur = ((self.rng >> 8) << 7) + EC_MIN_PROB;
        let bit = if v >= cur {
            self.rng -= cur;
            self.dif -= (cur as u64) << 48;
            0
        } else {
            self.rng = cur;
            1
        };
        self.renorm();
        bit
    }
}
