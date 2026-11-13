//! rivet-av1's window SymbolDecoder (step 8 form, cold refill on values), SymbolEncoder and update_cdf.
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
#[derive(Clone)]
pub struct Dec<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub dif: u64,
    pub cnt: i32,
    pub range: u32,
    pub disable_update: bool,
}
impl<'a> Dec<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Dec { data, pos: 0, dif: 0, cnt: -15, range: 1 << 15, disable_update: false };
        d.refill();
        d
    }
    #[inline(always)]
    pub fn refill(&mut self) {
        if let Some(b) = self.data.get(self.pos..self.pos + 8) {
            let w = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            let bytes = (48 - self.cnt) >> 3;
            self.dif |= !w >> (16 + self.cnt);
            self.pos += bytes as usize;
            self.cnt += 8 * bytes;
        } else {
            (self.pos, self.dif, self.cnt) = Self::refill_bytes(self.data, self.pos, self.dif, self.cnt);
        }
    }
    #[cold]
    #[inline(never)]
    fn refill_bytes(data: &[u8], mut pos: usize, mut dif: u64, mut cnt: i32) -> (usize, u64, i32) {
        while cnt <= 40 {
            let b = data.get(pos).copied().unwrap_or(0);
            dif |= ((b ^ 0xFF) as u64) << (40 - cnt);
            pos += 1;
            cnt += 8;
        }
        (pos, dif, cnt)
    }
    #[inline(always)]
    pub fn read_symbol_fixed<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        self.read_symbol_inline(cdf)
    }
    #[inline(always)]
    fn read_symbol_inline(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let value = (self.dif >> 48) as u32;
        let r = self.range >> 8;
        let mut cur = self.range;
        let mut symbol: usize = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - cdf[symbol] as u32;
            cur = (r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if value >= cur {
                break;
            }
            symbol += 1;
        }
        self.range = prev - cur;
        self.dif -= (cur as u64) << 48;
        self.renormalize();
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
        symbol
    }
    /// One arm per symbol: the search's exit for a symbol renormalises and adapts for that symbol, known there.
    #[inline(always)]
    pub fn read_symbol_arms<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        let value = (self.dif >> 48) as u32;
        let r = self.range >> 8;
        let mut prev = self.range;
        let mut symbol = 0;
        while symbol < N - 2 {
            let f = (1u32 << 15) - cdf[symbol] as u32;
            let cur = ((r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (N - 2 - symbol) as u32;
            if value >= cur {
                self.range = prev - cur;
                self.dif -= (cur as u64) << 48;
                self.renormalize();
                if !self.disable_update {
                    adapt_fixed(cdf, symbol);
                }
                return symbol;
            }
            prev = cur;
            symbol += 1;
        }
        // The last symbol: its threshold is zero (the CDF ends at 1 << 15).
        debug_assert!(cdf[N - 2] == 1 << 15);
        self.range = prev;
        self.renormalize();
        if !self.disable_update {
            adapt_fixed(cdf, N - 2);
        }
        N - 2
    }
    #[inline(always)]
    
    #[inline(always)]
    pub fn read_bool_select(&mut self) -> u32 {
        let cur = ((self.range >> 8) << 7) + EC_MIN_PROB;
        let one = ((self.dif >> 48) as u32) < cur;
        let (prev, sub) = if one { (cur, 0) } else { (self.range, cur) };
        self.range = prev - sub;
        self.dif -= (sub as u64) << 48;
        self.renormalize();
        one as u32
    }
    /// No branch on the data: every threshold, each entry's adaptation by its own comparison, the interval by
    /// selects. For CDFs of 2, 3 and 4 symbols (N = 3, 4, 5).
    #[inline(always)]
    pub fn read_symbol_sel<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        use core::hint::select_unpredictable as sel;
        let n = N - 1;
        let v = (self.dif >> 48) as u32;
        let r = self.range >> 8;
        let thr = |c: u16, k: u32| ((r * ((32768 - c as u32) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * k;
        let cnt = cdf[n];
        let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + floor_log2(n as u32).min(2);
        let adapt = |c: u16, above: bool| sel(above, c - (c >> rate), c + ((32768 - c as u32) >> rate) as u16);
        let (prev, cur, s);
        match N {
            3 => {
                let a0 = cdf[0];
                let c0 = thr(a0, 1);
                let b0 = v < c0;
                (prev, cur, s) = (sel(b0, c0, self.range), sel(b0, 0, c0), b0 as usize);
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                }
            }
            4 => {
                let (a0, a1) = (cdf[0], cdf[1]);
                let (c0, c1) = (thr(a0, 2), thr(a1, 1));
                let (b0, b1) = (v < c0, v < c1);
                prev = sel(b0, sel(b1, c1, c0), self.range);
                cur = sel(b0, sel(b1, 0, c1), c0);
                s = b0 as usize + b1 as usize;
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                    cdf[1] = adapt(a1, b1);
                }
            }
            _ => {
                let (a0, a1, a2) = (cdf[0], cdf[1], cdf[2]);
                let (c0, c1, c2) = (thr(a0, 3), thr(a1, 2), thr(a2, 1));
                let (b0, b1, b2) = (v < c0, v < c1, v < c2);
                prev = sel(b1, sel(b2, c2, c1), sel(b0, c0, self.range));
                cur = sel(b1, sel(b2, 0, c2), sel(b0, c1, c0));
                s = b0 as usize + b1 as usize + b2 as usize;
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                    cdf[1] = adapt(a1, b1);
                    cdf[2] = adapt(a2, b2);
                }
            }
        }
        if !self.disable_update {
            cdf[n] = cnt + (cnt < 32) as u16;
        }
        self.range = prev - cur;
        self.dif -= (cur as u64) << 48;
        self.renormalize();
        s
    }
    #[inline(always)]
    pub fn read_symbol_sel2<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        use core::hint::select_unpredictable as sel;
        let n = N - 1;
        let v = (self.dif >> 48) as u32;
        let r = self.range >> 8;
        let thr = |c: u16, k: u32| ((r * ((32768 - c as u32) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * k;
        let cnt = cdf[n];
        debug_assert!(cnt <= 32);
        let rate = 3 + (cnt >> 4) as u32 + floor_log2(n as u32).min(2);
        let adapt = |c: u16, above: bool| sel(above, c - (c >> rate), c + ((32768 - c as u32) >> rate) as u16);
        let (prev, cur, s);
        match N {
            3 => {
                let a0 = cdf[0];
                let c0 = thr(a0, 1);
                let b0 = v < c0;
                (prev, cur, s) = (sel(b0, c0, self.range), sel(b0, 0, c0), b0 as usize);
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                }
            }
            4 => {
                let (a0, a1) = (cdf[0], cdf[1]);
                let (c0, c1) = (thr(a0, 2), thr(a1, 1));
                let (b0, b1) = (v < c0, v < c1);
                prev = sel(b0, sel(b1, c1, c0), self.range);
                cur = sel(b0, sel(b1, 0, c1), c0);
                s = b0 as usize + b1 as usize;
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                    cdf[1] = adapt(a1, b1);
                }
            }
            _ => {
                let (a0, a1, a2) = (cdf[0], cdf[1], cdf[2]);
                let (c0, c1, c2) = (thr(a0, 3), thr(a1, 2), thr(a2, 1));
                let (b0, b1, b2) = (v < c0, v < c1, v < c2);
                prev = sel(b1, sel(b2, c2, c1), sel(b0, c0, self.range));
                cur = sel(b1, sel(b2, 0, c2), sel(b0, c1, c0));
                s = b0 as usize + b1 as usize + b2 as usize;
                if !self.disable_update {
                    cdf[0] = adapt(a0, b0);
                    cdf[1] = adapt(a1, b1);
                    cdf[2] = adapt(a2, b2);
                }
            }
        }
        if !self.disable_update {
            cdf[n] = cnt + (cnt < 32) as u16;
        }
        self.range = prev - cur;
        self.dif -= (cur as u64) << 48;
        self.renormalize();
        s
    }
    #[inline(always)]
    pub fn read_bool_sel(&mut self) -> u32 {
        use core::hint::select_unpredictable as sel;
        let cur = ((self.range >> 8) << 7) + EC_MIN_PROB;
        let one = ((self.dif >> 48) as u32) < cur;
        let (prev, sub) = (sel(one, cur, self.range), sel(one, 0, cur));
        self.range = prev - sub;
        self.dif -= (sub as u64) << 48;
        self.renormalize();
        one as u32
    }
    #[inline(always)]
    pub fn renormalize(&mut self) {
        let bits = self.range.leading_zeros() - 16;
        self.range <<= bits;
        self.dif <<= bits;
        self.cnt -= bits as i32;
        if self.cnt < 0 {
            self.refill();
        }
    }
    #[inline(always)]
    pub fn read_bool(&mut self) -> u32 {
        let cur = ((self.range >> 8) << 7) + EC_MIN_PROB;
        let bit = if (self.dif >> 48) as u32 >= cur {
            self.range -= cur;
            self.dif -= (cur as u64) << 48;
            0
        } else {
            self.range = cur;
            1
        };
        self.renormalize();
        bit
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

/// update_cdf for a CDF of N - 1 symbols, where the symbol is known to the compiler at each use.
#[inline(always)]
pub fn adapt_fixed<const N: usize>(cdf: &mut [u16; N], symbol: usize) {
    let n = N - 1;
    let cnt = cdf[n];
    let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + floor_log2(n as u32).min(2);
    let mut i = 0;
    while i < n - 1 {
        if i < symbol {
            cdf[i] -= cdf[i] >> rate;
        } else {
            cdf[i] += (((1u32 << 15) - cdf[i] as u32) >> rate) as u16;
        }
        i += 1;
    }
    cdf[n] = cnt + (cnt < 32) as u16;
}
