//! Variant g: e, with fewer instructions around the symbol reads. The level map holds Min( level, 3 ) (the
//! coeff_base symbol itself), so a context is five loads and four additions and a table; the levels proper go
//! to Quant, where the rare coeff_br context reads them as the reference does; the scan map carries the CDF's
//! index; coeff_br is read out of line; the non-zero bits are gathered in a register.
use crate::common::*;
use crate::sym::Dec;

const LEVEL_STRIDE: usize = 36;
const LEVEL_NEAR: usize = 4 * LEVEL_STRIDE + 1;
const MAP_LPOS: usize = (1 << 11) - 1;
const MAP_CDF_SHIFT: usize = 11;
const MAP_CDF: usize = (1 << 8) - 1;
const MAP_POS_SHIFT: usize = 19;

const fn near_offsets(tx_class: usize) -> [usize; SIG_REF_DIFF_OFFSET_NUM] {
    let mut o = [0; SIG_REF_DIFF_OFFSET_NUM];
    let mut i = 0;
    while i < SIG_REF_DIFF_OFFSET_NUM {
        let d = SIG_REF_DIFF_OFFSET[tx_class][i];
        o[i] = d[0] as usize * LEVEL_STRIDE + d[1] as usize;
        i += 1;
    }
    o
}
const NEAR_2D: [usize; SIG_REF_DIFF_OFFSET_NUM] = near_offsets(TX_CLASS_2D);
const NEAR_HORIZ: [usize; SIG_REF_DIFF_OFFSET_NUM] = near_offsets(TX_CLASS_HORIZ);
const NEAR_VERT: [usize; SIG_REF_DIFF_OFFSET_NUM] = near_offsets(TX_CLASS_VERT);

/// `5 * Min( ( mag + 1 ) >> 1, 4 )`: what the neighbours add to the index of a coeff_base CDF's first entry.
const MAG_CDF: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut mag = 0;
    while mag < 16 {
        let ctx = (mag + 1) >> 1;
        t[mag] = 5 * if ctx < 4 { ctx as u8 } else { 4 };
        mag += 1;
    }
    t
};

fn scan_map(tx_sz: usize, tx_class: usize) -> &'static [u32] {
    static MAPS: std::sync::OnceLock<Vec<Box<[u32]>>> = std::sync::OnceLock::new();
    &MAPS.get_or_init(|| {
        let mut maps = Vec::with_capacity(TX_SIZES_ALL * 3);
        for tx_sz in 0..TX_SIZES_ALL {
            let bwl = TX_WIDTH_LOG2[ADJUSTED_TX_SIZE[tx_sz]];
            for (tx_class, t) in [(TX_CLASS_2D, DCT_DCT), (TX_CLASS_HORIZ, H_DCT), (TX_CLASS_VERT, V_DCT)] {
                let scan = scan_for(tx_sz, t);
                assert!(scan[0] == 0 && scan[1..].iter().all(|&pos| pos != 0));
                let map = scan.iter().map(|&pos| {
                    let pos = pos as usize;
                    let (row, col) = (pos >> bwl, pos & ((1 << bwl) - 1));
                    let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, 0);
                    ((pos << MAP_POS_SHIFT) | (5 * ctx << MAP_CDF_SHIFT) | (row * LEVEL_STRIDE + col)) as u32
                });
                maps.push(map.collect());
            }
        }
        maps
    })[tx_sz * 3 + tx_class]
}

/// What the level pass works on.
struct Pass<'a, 'd> {
    d: Dec<'d>,
    cdf_base: &'a mut [u16; 5 * 42],
    cdf_br: &'a mut [[u16; 5]; 21],
    levels: &'a mut [u8; 36 * 36],
    quant: &'a mut [i32; 1024],
    tx_class: usize,
    bwl: usize,
    height: usize,
}

/// `coeff_br` for a coefficient whose `coeff_base` was 3: the level, 3 to 15. Out of line and on values: it is
/// rare, and the level pass keeps its registers.
#[inline(never)]
fn level_above_base<'d>(
    mut d: Dec<'d>,
    cdf_br: &mut [[u16; 5]; 21],
    quant: &[i32; 1024],
    tx_class: usize,
    bwl: usize,
    height: usize,
    pos: usize,
) -> (Dec<'d>, u32) {
    let width = 1usize << bwl;
    let (row, col) = (pos >> bwl, pos & (width - 1));
    let mut mag = 0u32;
    for o in &MAG_REF_OFFSET_WITH_TX_CLASS[tx_class] {
        let (r, c) = (row + o[0] as usize, col + o[1] as usize);
        if r < height && c < width {
            mag += (quant[(r << bwl) + c] as u32).min(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1);
        }
    }
    let br_ctx = coeff_br_ctx_of(tx_class, row, col, mag);
    let mut level = NUM_BASE_LEVELS + 1;
    for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
        let coeff_br = d.read_symbol_sel2(&mut cdf_br[br_ctx]) as u32;
        level += coeff_br;
        if coeff_br < BR_CDF_SIZE - 1 {
            break;
        }
    }
    (d, level)
}

impl Pass<'_, '_> {
    /// A level whose `coeff_base` symbol (or `coeff_base_eob` plus one) is `base`, stored.
    #[inline(always)]
    fn store(&mut self, e: usize, base: usize) -> u32 {
        self.levels[e & MAP_LPOS] = base as u8;
        let pos = e >> MAP_POS_SHIFT;
        let mut level = base as u32;
        if level > NUM_BASE_LEVELS {
            (self.d, level) =
                level_above_base(self.d.clone(), self.cdf_br, self.quant, self.tx_class, self.bwl, self.height, pos);
        }
        self.quant[pos] = level as i32;
        level
    }

    /// The coefficients between the last and the first, from the end of the scan back.
    #[inline(always)]
    fn between(&mut self, map: &[u32], nonzero: &mut [u64; 16], near_off: [usize; SIG_REF_DIFF_OFFSET_NUM]) {
        let mut word = 0u64;
        for c in (1..map.len() - 1).rev() {
            let e = map[c] as usize;
            let lpos = e & MAP_LPOS;
            let near: &[u8; LEVEL_NEAR] = self.levels[lpos..lpos + LEVEL_NEAR].try_into().unwrap();
            let mag = near_off.iter().map(|&o| near[o] as usize).sum::<usize>();
            let i = MAG_CDF[mag & 15] as usize + ((e >> MAP_CDF_SHIFT) & MAP_CDF);
            let cdf: &mut [u16; 5] = (&mut self.cdf_base[i..i + 5]).try_into().unwrap();
            let base = self.d.read_symbol_sel2(cdf);
            self.store(e, base);
            word |= ((base != 0) as u64) << (c & 63);
            if c & 63 == 0 {
                nonzero[c >> 6] |= word;
                word = 0;
            }
        }
        nonzero[0] |= word;
    }
}

#[inline(never)]
pub fn read(d0: &mut Dec, st: &mut St, b: &Block) -> u32 {
    let (tx_sz, tx_class, ptype, eob) = (b.tx_sz as usize, b.tx_class as usize, b.ptype as usize, b.eob as usize);
    let tx_sz_ctx = tx_sz_ctx_of(tx_sz);
    st.quant[..seg_eob(tx_sz)].fill(0);
    let mut cul_level: u32 = 0;
    let mut dc_category = 0u8;
    let cdf = &mut st.cdfs;
    let adj = ADJUSTED_TX_SIZE[tx_sz];
    let bwl = TX_WIDTH_LOG2[adj];
    let height = TX_HEIGHT[adj];
    let map = &scan_map(tx_sz, tx_class)[..eob];
    let near_off = match tx_class {
        TX_CLASS_2D => NEAR_2D,
        TX_CLASS_HORIZ => NEAR_HORIZ,
        _ => NEAR_VERT,
    };
    st.levels[..(height + 4) * LEVEL_STRIDE].fill(0);
    let cdf_eob = &mut cdf.coeff_base_eob[tx_sz_ctx][ptype];
    let cdf_base = &mut cdf.coeff_base[tx_sz_ctx][ptype];
    let mut nonzero = [0u64; 16];
    let mut p = Pass {
        d: d0.clone(),
        cdf_base: cdf_base.as_flattened_mut().try_into().unwrap(),
        cdf_br: &mut cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype],
        levels: &mut st.levels,
        quant: &mut st.quant,
        tx_class,
        bwl,
        height,
    };
    // The last coefficient: its own CDFs, and never zero.
    let c = eob - 1;
    let ctx = coeff_base_eob_ctx(c, bwl, height);
    let base = p.d.read_symbol_sel2(&mut cdf_eob[ctx]) + 1;
    p.store(map[c] as usize, base);
    nonzero[c >> 6] |= 1 << (c & 63);
    if eob > 2 {
        match tx_class {
            TX_CLASS_2D => p.between(map, &mut nonzero, NEAR_2D),
            TX_CLASS_HORIZ => p.between(map, &mut nonzero, NEAR_HORIZ),
            _ => p.between(map, &mut nonzero, NEAR_VERT),
        }
    }
    // The first, whose context is 0 in a 2D transform.
    if eob > 1 {
        let e = map[0] as usize;
        let near = &p.levels[e & MAP_LPOS..];
        let mag: u32 = near_off.iter().map(|&o| near[o] as u32).sum();
        let ctx = coeff_base_ctx_of(tx_sz, tx_class, 0, 0, mag);
        let cdf: &mut [u16; 5] = (&mut p.cdf_base[5 * ctx..5 * ctx + 5]).try_into().unwrap();
        let base = p.d.read_symbol_sel2(cdf);
        p.store(e, base);
        nonzero[0] |= (base != 0) as u64;
    }
    let Pass { mut d, quant, .. } = p;
    // The signs and the Golomb parts, in scan order: only a coefficient that is not zero has either.
    if nonzero[0] & 1 != 0 {
        let sign = d.read_symbol_sel2(&mut cdf.dc_sign[ptype]) as u32;
        let level = level_golomb(&mut d, quant[0] as u32);
        dc_category = if sign != 0 { 1 } else { 2 };
        let level = level & 0xFFFFF;
        cul_level += level;
        quant[0] = if sign != 0 { -(level as i32) } else { level as i32 };
        nonzero[0] &= !1;
    }
    for (i, word) in nonzero[..=(eob - 1) >> 6].iter().enumerate() {
        let mut word = *word;
        while word != 0 {
            let c = (i << 6) + word.trailing_zeros() as usize;
            word &= word - 1;
            let pos = map[c] as usize >> MAP_POS_SHIFT;
            let sign = d.read_bool_sel();
            let level = level_golomb(&mut d, quant[pos] as u32) & 0xFFFFF;
            cul_level += level;
            quant[pos] = if sign != 0 { -(level as i32) } else { level as i32 };
        }
    }
    *d0 = d;
    let _ = dc_category;
    cul_level.min(63)
}

#[inline(always)]
fn level_golomb(d: &mut Dec, level: u32) -> u32 {
    if level <= NUM_BASE_LEVELS + COEFF_BASE_RANGE {
        return level;
    }
    let mut length = 0;
    loop {
        length += 1;
        if d.read_bool_sel() != 0 {
            break;
        }
        if length > 32 {
            break;
        }
    }
    let mut x: u32 = 1;
    for _ in 0..length - 1 {
        x = (x << 1) | d.read_bool_sel();
    }
    (x as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64).min(i32::MAX as u64) as u32
}
