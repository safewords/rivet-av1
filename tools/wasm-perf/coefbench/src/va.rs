//! Variant a: the level and sign passes of coeffs_decode as they are after step 8 (829b1e1's loops on the
//! window decoder).
use crate::common::*;
use crate::sym::Dec;

pub fn read(d: &mut Dec, st: &mut St, b: &Block) -> u32 {
    let (tx_sz, tx_class, ptype, eob) = (b.tx_sz as usize, b.tx_class as usize, b.ptype as usize, b.eob as usize);
    let tx_sz_ctx = tx_sz_ctx_of(tx_sz);
    st.quant[..seg_eob(tx_sz)].fill(0);
    let mut cul_level: u32 = 0;
    let mut dc_category = 0u8;
    let scan = scan_of(tx_sz, tx_class);
    let cdf = &mut st.cdfs;
    let adj = ADJUSTED_TX_SIZE[tx_sz];
    let bwl = TX_WIDTH_LOG2[adj];
    let height = TX_HEIGHT[adj];
    let width = 1usize << bwl;
    let lstride = width + 4;
    let levels = &mut st.levels[..(height + 4) * lstride];
    levels.fill(0);
    let base_off: [usize; SIG_REF_DIFF_OFFSET_NUM] = std::array::from_fn(|i| {
        let o = SIG_REF_DIFF_OFFSET[tx_class][i];
        o[0] as usize * lstride + o[1] as usize
    });
    let br_off: [usize; 3] = std::array::from_fn(|i| {
        let o = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][i];
        o[0] as usize * lstride + o[1] as usize
    });
    let quant = &mut st.quant;
    let cdf_eob = &mut cdf.coeff_base_eob[tx_sz_ctx][ptype];
    let cdf_base = &mut cdf.coeff_base[tx_sz_ctx][ptype];
    let cdf_br = &mut cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype];
    for c in (0..eob).rev() {
        let pos = scan[c] as usize;
        let (row, col) = (pos >> bwl, pos & (width - 1));
        let lpos = row * lstride + col;
        let mut level;
        if c == eob - 1 {
            let ctx = coeff_base_eob_ctx(c, bwl, height);
            level = d.read_symbol_fixed(&mut cdf_eob[ctx]) as u32 + 1;
        } else {
            let near = &levels[lpos..];
            let mag: u32 = base_off.iter().map(|&o| near[o].min(3) as u32).sum();
            let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, mag);
            level = d.read_symbol_fixed(&mut cdf_base[ctx]) as u32;
        }
        if level > NUM_BASE_LEVELS {
            let near = &levels[lpos..];
            let mag: u32 = br_off.iter().map(|&o| near[o] as u32).sum();
            let br_ctx = coeff_br_ctx_of(tx_class, row, col, mag);
            for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                let coeff_br = d.read_symbol_fixed(&mut cdf_br[br_ctx]) as u32;
                level += coeff_br;
                if coeff_br < BR_CDF_SIZE - 1 {
                    break;
                }
            }
        }
        quant[pos] = level as i32;
        levels[lpos] = level as u8;
    }
    for c in 0..eob {
        let pos = scan[c] as usize;
        let sign = if quant[pos] != 0 {
            if c == 0 { d.read_symbol_fixed(&mut cdf.dc_sign[ptype]) as u32 } else { d.read_bool() }
        } else {
            0
        };
        if quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
            let mut length = 0;
            loop {
                length += 1;
                if d.read_bool() != 0 {
                    break;
                }
                if length > 32 {
                    break;
                }
            }
            let mut x: u32 = 1;
            for _ in 0..length - 1 {
                x = (x << 1) | d.read_bool();
            }
            quant[pos] = (x as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64).min(i32::MAX as u64) as i32;
        }
        if pos == 0 && quant[pos] > 0 {
            dc_category = if sign != 0 { 1 } else { 2 };
        }
        quant[pos] &= 0xFFFFF;
        cul_level += quant[pos] as u32;
        if sign != 0 {
            quant[pos] = -quant[pos];
        }
    }
    let _ = dc_category;
    cul_level.min(63)
}
