//! What every variant shares: the recorded blocks, the encoder that turns them back into a stream (with the
//! reader's contexts as residual.rs's reference functions compute them), CDFs and buffers.
pub use crate::consts::*;
pub use crate::tables::*;
use crate::sym::SymbolEncoder;

pub struct Block {
    pub tx_sz: u8,
    pub tx_class: u8,
    pub ptype: u8,
    pub eob: u16,
    pub start: u32,
}
pub struct Work {
    pub blocks: Vec<Block>,
    pub coefs: Vec<i32>,
    pub data: Vec<u8>,
    pub sum: u64,
}
#[derive(Clone)]
pub struct Cdfs {
    pub coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5],
    pub coeff_base: [[[[u16; 5]; 42]; 2]; 5],
    pub coeff_br: [[[[u16; 5]; 21]; 2]; 5],
    pub dc_sign: [[u16; 3]; 2],
}
impl Cdfs {
    pub fn new() -> Self {
        Cdfs {
            coeff_base_eob: [[[[10923, 21845, 32768, 0]; 4]; 2]; 5],
            coeff_base: [[[[8192, 16384, 24576, 32768, 0]; 42]; 2]; 5],
            coeff_br: [[[[8192, 16384, 24576, 32768, 0]; 21]; 2]; 5],
            dc_sign: [[16384, 32768, 0]; 2],
        }
    }
}
pub struct St {
    pub cdfs: Cdfs,
    pub quant: Box<[i32; 1024]>,
    pub levels: Box<[u8; 36 * 36]>,
}
impl St {
    pub fn new() -> Box<St> {
        Box::new(St { cdfs: Cdfs::new(), quant: Box::new([0; 1024]), levels: Box::new([0; 36 * 36]) })
    }
}

pub fn seg_eob(tx_sz: usize) -> usize {
    if tx_sz == TX_16X64 || tx_sz == TX_64X16 { 512 } else { 1024.min(TX_WIDTH[tx_sz] * TX_HEIGHT[tx_sz]) }
}
pub fn tx_sz_ctx_of(tx_sz: usize) -> usize {
    (TX_SIZE_SQR[tx_sz] + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1
}
/// A transform type of the class (the scan depends on the type through its class alone).
pub fn type_of_class(tx_class: usize) -> usize {
    match tx_class {
        TX_CLASS_2D => DCT_DCT,
        TX_CLASS_HORIZ => H_DCT,
        _ => V_DCT,
    }
}
pub fn scan_of(tx_sz: usize, tx_class: usize) -> &'static [u16] {
    scan_for(tx_sz, type_of_class(tx_class))
}

// ---- residual.rs, verbatim ----
pub fn scan_for(tx_sz: usize, t: usize) -> &'static [u16] {
    if tx_sz == TX_16X64 {
        return &DEFAULT_SCAN_16X32;
    }
    if tx_sz == TX_64X16 {
        return &DEFAULT_SCAN_32X16;
    }
    if TX_SIZE_SQR_UP[tx_sz] == TX_64X64 {
        return &DEFAULT_SCAN_32X32;
    }
    if t == IDTX {
        return default_scan(tx_sz);
    }
    let prefer_row = t == V_DCT || t == V_ADST || t == V_FLIPADST;
    let prefer_col = t == H_DCT || t == H_ADST || t == H_FLIPADST;
    if prefer_row {
        match tx_sz {
            TX_4X4 => &MROW_SCAN_4X4,
            TX_4X8 => &MROW_SCAN_4X8,
            TX_8X4 => &MROW_SCAN_8X4,
            TX_8X8 => &MROW_SCAN_8X8,
            TX_8X16 => &MROW_SCAN_8X16,
            TX_16X8 => &MROW_SCAN_16X8,
            TX_16X16 => &MROW_SCAN_16X16,
            TX_4X16 => &MROW_SCAN_4X16,
            _ => &MROW_SCAN_16X4,
        }
    } else if prefer_col {
        match tx_sz {
            TX_4X4 => &MCOL_SCAN_4X4,
            TX_4X8 => &MCOL_SCAN_4X8,
            TX_8X4 => &MCOL_SCAN_8X4,
            TX_8X8 => &MCOL_SCAN_8X8,
            TX_8X16 => &MCOL_SCAN_8X16,
            TX_16X8 => &MCOL_SCAN_16X8,
            TX_16X16 => &MCOL_SCAN_16X16,
            TX_4X16 => &MCOL_SCAN_4X16,
            _ => &MCOL_SCAN_16X4,
        }
    } else {
        default_scan(tx_sz)
    }
}
fn default_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        TX_4X4 => &DEFAULT_SCAN_4X4,
        TX_4X8 => &DEFAULT_SCAN_4X8,
        TX_8X4 => &DEFAULT_SCAN_8X4,
        TX_8X8 => &DEFAULT_SCAN_8X8,
        TX_8X16 => &DEFAULT_SCAN_8X16,
        TX_16X8 => &DEFAULT_SCAN_16X8,
        TX_16X16 => &DEFAULT_SCAN_16X16,
        TX_16X32 => &DEFAULT_SCAN_16X32,
        TX_32X16 => &DEFAULT_SCAN_32X16,
        TX_4X16 => &DEFAULT_SCAN_4X16,
        TX_16X4 => &DEFAULT_SCAN_16X4,
        TX_8X32 => &DEFAULT_SCAN_8X32,
        TX_32X8 => &DEFAULT_SCAN_32X8,
        _ => &DEFAULT_SCAN_32X32,
    }
}
#[inline(always)]
pub fn coeff_base_ctx_of(tx_sz: usize, tx_class: usize, row: usize, col: usize, mag: u32) -> usize {
    let ctx = mag.div_ceil(2).min(4) as usize;
    if tx_class == TX_CLASS_2D {
        if row == 0 && col == 0 {
            return 0;
        }
        return ctx + COEFF_BASE_CTX_OFFSET[tx_sz][row.min(4)][col.min(4)];
    }
    let idx = if tx_class == TX_CLASS_VERT { row } else { col };
    ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
}
#[inline(always)]
pub fn coeff_br_ctx_of(tx_class: usize, row: usize, col: usize, mag: u32) -> usize {
    let mag = mag.div_ceil(2).min(6) as usize;
    if row == 0 && col == 0 {
        mag
    } else if tx_class == 0 {
        if row < 2 && col < 2 { mag + 7 } else { mag + 14 }
    } else if tx_class == 1 {
        if col == 0 { mag + 7 } else { mag + 14 }
    } else if row == 0 {
        mag + 7
    } else {
        mag + 14
    }
}
pub fn coeff_base_eob_ctx(c: usize, bwl: usize, height: usize) -> usize {
    if c == 0 {
        return 0;
    }
    if c <= (height << bwl) / 8 {
        return 1;
    }
    if c <= (height << bwl) / 4 {
        return 2;
    }
    3
}

pub fn load(trace: &[u8]) -> Work {
    let mut blocks = Vec::new();
    let mut coefs = Vec::new();
    let mut i = 0;
    while i < trace.len() {
        let eob = u16::from_le_bytes([trace[i + 3], trace[i + 4]]);
        blocks.push(Block { tx_sz: trace[i], tx_class: trace[i + 1], ptype: trace[i + 2], eob, start: coefs.len() as u32 });
        i += 5;
        for _ in 0..eob {
            coefs.push(i32::from_le_bytes(trace[i..i + 4].try_into().unwrap()));
            i += 4;
        }
    }
    // Encode: the level pass and the sign pass of coeffs(), as residual.rs's reference reader walks them.
    let mut e = SymbolEncoder::new();
    let mut cdf = Cdfs::new();
    let mut sum = 0u64;
    let mut level = vec![0u32; 1024]; // by position: min(|coef|, 15) as read by the level pass
    for b in &blocks {
        let (tx_sz, tx_class, ptype, eob) = (b.tx_sz as usize, b.tx_class as usize, b.ptype as usize, b.eob as usize);
        let scan = scan_of(tx_sz, tx_class);
        let co = &coefs[b.start as usize..][..eob];
        let tsc = tx_sz_ctx_of(tx_sz);
        let adj = ADJUSTED_TX_SIZE[tx_sz];
        let bwl = TX_WIDTH_LOG2[adj];
        let (width, height) = (1usize << bwl, TX_HEIGHT[adj]);
        level[..width * height].fill(0);
        let at = |level: &[u32], r: usize, c: usize| if r < height && c < width { level[(r << bwl) + c] } else { 0 };
        for c in (0..eob).rev() {
            let pos = scan[c] as usize;
            let (row, col) = (pos >> bwl, pos & (width - 1));
            let full = co[c].unsigned_abs();
            let mut lv;
            if c == eob - 1 {
                let ctx = coeff_base_eob_ctx(c, bwl, height);
                lv = full.clamp(1, 3);
                e.write_symbol(&mut cdf.coeff_base_eob[tsc][ptype][ctx], lv as usize - 1);
            } else {
                let mag: u32 = SIG_REF_DIFF_OFFSET[tx_class]
                    .iter()
                    .map(|o| at(&level, row + o[0] as usize, col + o[1] as usize).min(3))
                    .sum();
                let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, mag);
                lv = full.min(3);
                e.write_symbol(&mut cdf.coeff_base[tsc][ptype][ctx], lv as usize);
            }
            if lv > NUM_BASE_LEVELS {
                let mag: u32 = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class]
                    .iter()
                    .map(|o| at(&level, row + o[0] as usize, col + o[1] as usize))
                    .sum();
                let br_ctx = coeff_br_ctx_of(tx_class, row, col, mag);
                for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                    let s = (full.min(15) - lv).min(3);
                    e.write_symbol(&mut cdf.coeff_br[tsc.min(TX_32X32)][ptype][br_ctx], s as usize);
                    lv += s;
                    if s < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }
            level[pos] = lv;
        }
        let mut cul = 0u32;
        for c in 0..eob {
            let pos = scan[c] as usize;
            if co[c] != 0 {
                if c == 0 {
                    e.write_symbol(&mut cdf.dc_sign[ptype], (co[c] < 0) as usize);
                } else {
                    e.write_bool((co[c] < 0) as u32);
                }
            }
            if level[pos] > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
                let gx = co[c].unsigned_abs() - 14;
                let glen = 32 - gx.leading_zeros();
                for l in 1..=glen {
                    e.write_bool((l == glen) as u32);
                }
                for i in (0..glen - 1).rev() {
                    e.write_bool((gx >> i) & 1);
                }
            }
            cul += co[c].unsigned_abs() & 0xFFFFF;
        }
        sum += cul.min(63) as u64;
    }
    Work { blocks, coefs, data: e.finish(), sum }
}
