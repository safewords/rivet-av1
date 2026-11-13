//! V0: rivet-av1's SymbolDecoder as it is at 829b1e1.
use crate::enc::{EC_MIN_PROB, EC_PROB_SHIFT, floor_log2, update_cdf};
pub struct D<'a> {
    data: &'a [u8],
    pos: usize,
    value: u32,
    range: u32,
    max_bits: i64,
    disable_update: bool,
}
impl<'a> D<'a> {
    #[inline]
    fn read_bits(&mut self, n: u32) -> u32 {
        let byte = self.pos >> 3;
        let off = (self.pos & 7) as u32;
        let window = match self.data.get(byte..byte + 3) {
            Some(b) => (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32,
            None => {
                let at = |i: usize| self.data.get(byte + i).copied().unwrap_or(0) as u32;
                at(0) << 16 | at(1) << 8 | at(2)
            }
        };
        self.pos += n as usize;
        (window >> (24 - off - n)) & ((1 << n) - 1)
    }
    #[inline(never)]
    fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn read_symbol_inline(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let mut cur = self.range;
        let mut symbol: usize = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - cdf[symbol] as u32;
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if self.value >= cur {
                break;
            }
            symbol += 1;
        }
        self.range = prev - cur;
        self.value -= cur;
        self.renormalize();
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
        symbol
    }
    #[inline]
    fn renormalize(&mut self) {
        let bits = 15 - floor_log2(self.range);
        if bits == 0 {
            return;
        }
        self.range <<= bits;
        let num_bits = (bits as i64).min(self.max_bits.max(0)) as u32;
        let new_data = self.read_bits(num_bits);
        let padded = new_data << (bits - num_bits);
        self.value = padded ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= bits as i64;
    }
}
impl<'a> crate::SymDec<'a> for D<'a> {
    fn new(data: &'a [u8]) -> Self {
        let sz = data.len() as i64;
        let mut d = D { data, pos: 0, value: 0, range: 1 << 15, max_bits: 8 * sz - 15, disable_update: false };
        let num_bits = (sz * 8).min(15) as u32;
        let buf = d.read_bits(num_bits);
        let padded = buf << (15 - num_bits);
        d.value = ((1 << 15) - 1) ^ padded;
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
        let cur = ((self.range >> 8) << 7) + EC_MIN_PROB;
        let bit = if self.value >= cur {
            self.range -= cur;
            self.value -= cur;
            0
        } else {
            self.range = cur;
            1
        };
        self.renormalize();
        bit
    }
}
