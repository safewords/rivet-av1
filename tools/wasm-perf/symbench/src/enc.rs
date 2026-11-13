//! rivet-av1's SymbolEncoder and update_cdf (src/symbol.rs), trimmed of trial coding.
pub const EC_PROB_SHIFT: u32 = 6;
pub const EC_MIN_PROB: u32 = 4;
pub fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}
#[inline]
pub fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    let cnt = cdf[n];
    let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + floor_log2(n as u32).min(2);
    let (below, from) = cdf[..n - 1].split_at_mut(symbol.min(n - 1));
    for c in below {
        *c -= *c >> rate;
    }
    for c in from {
        *c += (((1u32 << 15) - *c as u32) >> rate) as u16;
    }
    if cnt < 32 {
        cdf[n] += 1;
    }
}
pub struct SymbolEncoder {
    out: Vec<u8>,
    low: u64,
    cnt: u32,
    range: u32,
    pub disable_update: bool,
}
impl SymbolEncoder {
    pub fn new() -> Self {
        SymbolEncoder { out: Vec::new(), low: 0, cnt: 15, range: 1 << 15, disable_update: false }
    }
    fn carry(&mut self) {
        let mut i = self.out.len();
        while i > 0 {
            i -= 1;
            self.out[i] = self.out[i].wrapping_add(1);
            if self.out[i] != 0 {
                return;
            }
        }
    }
    pub fn write_symbol(&mut self, cdf: &mut [u16], symbol: usize) {
        let n = cdf.len() - 1;
        let r = self.range;
        let thresh = |s: isize| -> u32 {
            if s < 0 {
                return r;
            }
            let s = s as usize;
            let f = (1u32 << 15) - cdf[s] as u32;
            (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s - 1) as u32
        };
        let prev = thresh(symbol as isize - 1);
        let cur = thresh(symbol as isize);
        self.low += (r - prev) as u64;
        self.range = prev - cur;
        if self.low >> self.cnt != 0 {
            self.low &= (1u64 << self.cnt) - 1;
            self.carry();
        }
        let d = 15 - floor_log2(self.range);
        self.range <<= d;
        self.low <<= d;
        self.cnt += d;
        while self.cnt >= 24 {
            let byte = (self.low >> (self.cnt - 8)) as u8;
            self.out.push(byte);
            self.low &= (1u64 << (self.cnt - 8)) - 1;
            self.cnt -= 8;
        }
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
    }
    pub fn write_bool(&mut self, bit: u32) {
        let mut cdf = [1u16 << 14, 1 << 15, 0];
        let save = self.disable_update;
        self.disable_update = true;
        self.write_symbol(&mut cdf, bit as usize);
        self.disable_update = save;
    }
    pub fn finish(mut self) -> Vec<u8> {
        let half = 1u64 << 14;
        let mut c = if self.low <= half { half } else { (((self.low - half) + (1 << 15) - 1) >> 15 << 15) + half };
        if c >> self.cnt != 0 {
            c &= (1u64 << self.cnt) - 1;
            self.carry();
        }
        let nbits = self.cnt - 14;
        let mut v = c >> 14;
        let mut left = nbits;
        while left >= 8 {
            self.out.push((v >> (left - 8)) as u8);
            left -= 8;
        }
        if left > 0 {
            v &= (1 << left) - 1;
            self.out.push((v << (8 - left)) as u8);
        }
        self.out
    }
}
