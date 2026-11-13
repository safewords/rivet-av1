//! Variant e (b with no branch on the data in the symbol reads): the experiment tree's passes (scan map, decoder in locals, a bit per non-zero coefficient).
use crate::common::*;
use crate::sym::Dec;

#[inline(never)]
pub fn read(d0: &mut Dec, st: &mut St, b: &Block) -> u32 {
    let (tx_sz, tx_class, ptype, eob) = (b.tx_sz as usize, b.tx_class as usize, b.ptype as usize, b.eob as usize);
    let tx_sz_ctx = tx_sz_ctx_of(tx_sz);
    st.quant[..seg_eob(tx_sz)].fill(0);
    let mut cul_level: u32 = 0;
    let mut dc_category = 0u8;
    let cdf = &mut st.cdfs;
    {
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let bwl = TX_WIDTH_LOG2[adj];
            let height = TX_HEIGHT[adj];
            // Where each coefficient of the scan lies and what its position
            // adds to its context, and the padded level map those places
            // are in.
            let map = &scan_map(tx_sz, tx_class)[..eob];
            let near_off = match tx_class {
                TX_CLASS_2D => NEAR_2D,
                TX_CLASS_HORIZ => NEAR_HORIZ,
                _ => NEAR_VERT,
            };
            let levels = &mut *st.levels;
            levels[..(height + 4) * LEVEL_STRIDE].fill(0);
            let cdf_eob = &mut cdf.coeff_base_eob[tx_sz_ctx][ptype];
            let cdf_base = &mut cdf.coeff_base[tx_sz_ctx][ptype];
            let cdf_br = &mut cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype];
            // The symbol decoder in locals for the two passes: nothing in
            // them takes its address, so it stays in registers.
            let mut d = d0.clone();
            // Which coefficients of the scan are not zero, a bit each.
            let mut nonzero = [0u64; 16];
            // The levels, from the last coefficient back. The last one has
            // its own context and is never zero.
            let c = eob - 1;
            let e = map[c] as usize;
            let ctx = coeff_base_eob_ctx(c, bwl, height);
            let level = d.read_symbol_sel(&mut cdf_eob[ctx]) as u32 + 1;
            let level = level_above_base(&mut d, cdf_br, levels, tx_class, bwl, e, level);
            levels[e & MAP_LPOS] = level as u8;
            nonzero[c >> 6] |= 1 << (c & 63);
            // The ones between it and the first: the context is the
            // neighbours' sum and the position's offset.
            if eob > 2 {
                let n = &mut nonzero;
                match tx_class {
                    TX_CLASS_2D => {
                        levels_between(&mut d, cdf_base, cdf_br, levels, map, n, NEAR_2D, 0, bwl)
                    }
                    TX_CLASS_HORIZ => {
                        levels_between(&mut d, cdf_base, cdf_br, levels, map, n, NEAR_HORIZ, 1, bwl)
                    }
                    _ => {
                        levels_between(&mut d, cdf_base, cdf_br, levels, map, n, NEAR_VERT, 2, bwl)
                    }
                }
            }
            // The first, whose context is its own in a 2D transform.
            if eob > 1 {
                let e = map[0] as usize;
                let lpos = e & MAP_LPOS;
                let near = &levels[lpos..];
                let mag: u32 = near_off.iter().map(|&o| near[o].min(3) as u32).sum();
                let pos = e >> MAP_POS_SHIFT;
                let (row, col) = (pos >> bwl, pos & ((1 << bwl) - 1));
                let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, mag);
                let level = d.read_symbol_sel(&mut cdf_base[ctx]) as u32;
                let level = level_above_base(&mut d, cdf_br, levels, tx_class, bwl, e, level);
                levels[lpos] = level as u8;
                nonzero[0] |= (level != 0) as u64;
            }
            // The signs and the Golomb parts, in scan order: only a
            // coefficient that is not zero has either.
            let quant = &mut st.quant;
            if nonzero[0] & 1 != 0 {
                let e = map[0] as usize;
                let sign = d.read_symbol_sel(&mut cdf.dc_sign[ptype]) as u32;
                let level = level_golomb(&mut d, levels[e & MAP_LPOS] as u32);
                dc_category = if sign != 0 { 1 } else { 2 };
                let level = level & 0xFFFFF;
                cul_level += level;
                quant[e >> MAP_POS_SHIFT] = if sign != 0 {
                    -(level as i32)
                } else {
                    level as i32
                };
                nonzero[0] &= !1;
            }
            for (i, word) in nonzero[..=(eob - 1) >> 6].iter().enumerate() {
                let mut word = *word;
                while word != 0 {
                    let c = (i << 6) + word.trailing_zeros() as usize;
                    word &= word - 1;
                    let e = map[c] as usize;
                    let sign = d.read_bool_sel();
                    let level = level_golomb(&mut d, levels[e & MAP_LPOS] as u32) & 0xFFFFF;
                    cul_level += level;
                    quant[e >> MAP_POS_SHIFT] = if sign != 0 {
                        -(level as i32)
                    } else {
                        level as i32
                    };
                }
            }
            *d0 = d;
    }
    let _ = dc_category;
    cul_level.min(63)
}

/// The row stride of the decode-only reader's level map: the widest
/// transform's 32 columns and the 4 of padding the context neighbours
/// reach into.
const LEVEL_STRIDE: usize = 36;
/// How far past a coefficient its context neighbours lie in the level
/// map: four rows down at most.
const LEVEL_NEAR: usize = 4 * LEVEL_STRIDE + 1;

/// A scan map entry (see [`scan_map`]): the coefficient's place in the
/// level map, what its position adds to the context of `coeff_base`, and
/// its position in the block (`scan[c]`).
const MAP_LPOS: usize = (1 << 11) - 1;
const MAP_CTX_SHIFT: usize = 11;
const MAP_CTX: usize = (1 << 6) - 1;
const MAP_POS_SHIFT: usize = 17;

/// Where the neighbours `Sig_Ref_Diff_Offset` names for a transform class
/// lie in the level map, from the coefficient.
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

/// For the decode-only reader: the scan of a transform size and class,
/// with what the reader needs of each position worked out once. Entry `c`
/// holds, for the coefficient `pos = scan[c]` at `(row, col)`: its place
/// `row * LEVEL_STRIDE + col` in the level map (`MAP_LPOS`), the part of
/// `get_coeff_base_ctx()` that depends on the position alone
/// (`MAP_CTX`: the context is that plus `Min( ( mag + 1 ) >> 1, 4 )`, for
/// every coefficient but the first of a 2D transform, whose context is 0),
/// and `pos` itself.
fn scan_map(tx_sz: usize, tx_class: usize) -> &'static [u32] {
    static MAPS: std::sync::OnceLock<Vec<Box<[u32]>>> = std::sync::OnceLock::new();
    &MAPS.get_or_init(|| {
        let mut maps = Vec::with_capacity(TX_SIZES_ALL * 3);
        for tx_sz in 0..TX_SIZES_ALL {
            let bwl = TX_WIDTH_LOG2[ADJUSTED_TX_SIZE[tx_sz]];
            // A type of each class: the scan depends on the type through
            // its class alone.
            for (tx_class, t) in [
                (TX_CLASS_2D, DCT_DCT),
                (TX_CLASS_HORIZ, H_DCT),
                (TX_CLASS_VERT, V_DCT),
            ] {
                let scan = scan_for(tx_sz, t);
                // The first coefficient of every scan is the DC one, which
                // the reader takes the first to be.
                assert!(scan[0] == 0 && scan[1..].iter().all(|&pos| pos != 0));
                let map = scan.iter().map(|&pos| {
                    let pos = pos as usize;
                    let (row, col) = (pos >> bwl, pos & ((1 << bwl) - 1));
                    let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, 0);
                    ((pos << MAP_POS_SHIFT) | (ctx << MAP_CTX_SHIFT) | (row * LEVEL_STRIDE + col))
                        as u32
                });
                maps.push(map.collect());
            }
        }
        maps
    })[tx_sz * 3 + tx_class]
}

/// The levels of the coefficients between the last and the first of a
/// block, from the end of the scan back: `coeff_base` with the context of
/// the five neighbours at `near` (their offsets in the level map, known
/// where this is compiled), and `coeff_br` above it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn levels_between(
    d: &mut Dec,
    cdf_base: &mut [[u16; 5]; 42],
    cdf_br: &mut [[u16; 5]; 21],
    levels: &mut [u8; 36 * 36],
    map: &[u32],
    nonzero: &mut [u64; 16],
    near_off: [usize; SIG_REF_DIFF_OFFSET_NUM],
    tx_class: usize,
    bwl: usize,
) {
    for c in (1..map.len() - 1).rev() {
        let e = map[c] as usize;
        let lpos = e & MAP_LPOS;
        let near: &[u8; LEVEL_NEAR] = levels[lpos..lpos + LEVEL_NEAR].try_into().unwrap();
        let mag: usize = near_off.iter().map(|&o| near[o].min(3) as usize).sum();
        let ctx = mag.div_ceil(2).min(4) + ((e >> MAP_CTX_SHIFT) & MAP_CTX);
        let level = d.read_symbol_sel(&mut cdf_base[ctx]) as u32;
        let level = level_above_base(d, cdf_br, levels, tx_class, bwl, e, level);
        levels[lpos] = level as u8;
        nonzero[c >> 6] |= ((level != 0) as u64) << (c & 63);
    }
}

/// The `coeff_br` part of a level that `coeff_base` left above
/// `NUM_BASE_LEVELS`: at most `NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1`
/// (15) when this is done; the Golomb part comes with the signs.
#[inline(always)]
fn level_above_base(
    d: &mut Dec,
    cdf_br: &mut [[u16; 5]; 21],
    levels: &[u8; 36 * 36],
    tx_class: usize,
    bwl: usize,
    e: usize,
    mut level: u32,
) -> u32 {
    if level > NUM_BASE_LEVELS {
        let pos = e >> MAP_POS_SHIFT;
        let (row, col) = (pos >> bwl, pos & ((1 << bwl) - 1));
        let near = &levels[e & MAP_LPOS..];
        let mag: u32 = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class]
            .iter()
            .map(|o| near[o[0] as usize * LEVEL_STRIDE + o[1] as usize] as u32)
            .sum();
        let br_ctx = coeff_br_ctx_of(tx_class, row, col, mag);
        for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
            let coeff_br = d.read_symbol_sel(&mut cdf_br[br_ctx]) as u32;
            level += coeff_br;
            if coeff_br < BR_CDF_SIZE - 1 {
                break;
            }
        }
    }
    level
}

/// The Golomb part of a level that the symbols left at their most, 15:
/// `golomb_length_bit`s and the bits they count.
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

