//! Times the 2D inverse transform of a rivet-av1 tree (src/dsp/itx.rs taken by path) on a fixed block per size,
//! natively and in wasm. TREE picks the tree at build time (build.rs writes the include paths).
#![allow(dead_code, clippy::all, unused_imports)]
include!(concat!(env!("OUT_DIR"), "/mods.rs"));

pub const SIZES: [usize; 8] = [0, 1, 2, 3, 4, 7, 9, 10]; // 4x4 8x8 16x16 32x32 64x64 8x16 16x32 32x16

fn fill(dq: &mut [i32], tx_sz: usize, full: bool) -> (usize, usize) {
    let w = 1usize << tables::TX_WIDTH_LOG2[tx_sz];
    let h = 1usize << tables::TX_HEIGHT_LOG2[tx_sz];
    let (rows, cols) = if full { (h.min(32), w.min(32)) } else { ((h.min(32) / 4).max(1), (w.min(32) / 4).max(1)) };
    let mut s = 0x2545_f491u32;
    dq.fill(0);
    for i in 0..rows {
        for j in 0..cols {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            dq[i * 64 + j] = (s % 257) as i32 - 128;
        }
    }
    (rows, cols)
}

/// `iters` transforms of size index `si` (full block of coefficients, or its top-left sixteenth); a checksum.
pub fn run(si: usize, full: bool, bounded: bool, iters: u32) -> i32 {
    let tx_sz = SIZES[si];
    let mut dq = vec![0i32; 64 * 64];
    let mut res = vec![0i32; 64 * 64];
    let (rows, cols) = fill(&mut dq, tx_sz, full);
    let mut acc = 0i32;
    for it in 0..iters {
        dq[0] = (it % 64) as i32;
        go(std::hint::black_box(&dq), tx_sz, (rows, cols), bounded, &mut res, 0, 8);
        acc = acc.wrapping_add(res[5]).wrapping_add(res[1]);
    }
    acc
}

#[cfg(has_within)]
fn go(dq: &[i32], tx_sz: usize, within: (usize, usize), bounded: bool, res: &mut [i32], tx_type: usize, bd: u32) {
    if bounded {
        itx::inverse_transform_2d_within(dq, tx_sz, tx_type, bd, within, res)
    } else {
        itx::inverse_transform_2d(dq, tx_sz, tx_type, false, bd, res)
    }
}
#[cfg(not(has_within))]
fn go(dq: &[i32], tx_sz: usize, _within: (usize, usize), _bounded: bool, res: &mut [i32], tx_type: usize, bd: u32) {
    itx::inverse_transform_2d(dq, tx_sz, tx_type, false, bd, res)
}

/// The transform as the target takes it (`inverse_transform_2d`, and `_within` with the tightest and a looser
/// bound where the tree has it) against the scalar reference, for every size and type on coefficients in the
/// conformance range at 8 and 10 bits: the number of cases compared, or 0 at the first difference. The crate's
/// own unit test does this natively; this is the same comparison where the target is wasm32.
pub fn check() -> u32 {
    use consts::*;
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut rnd = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let mut cases = 0;
    for bd in [8u32, 10] {
        for tx_sz in 0..TX_SIZES_ALL {
            for tx_type in 0..16 {
                let w = 1usize << tables::TX_WIDTH_LOG2[tx_sz];
                let h = 1usize << tables::TX_HEIGHT_LOG2[tx_sz];
                let big = w.max(h) >= 64 || (w.max(h) == 32 && w.min(h) >= 16);
                if (big && tx_type != DCT_DCT && tx_type != IDTX) || (w.max(h) == 64 && tx_type != DCT_DCT) {
                    continue;
                }
                for case in 0..40 {
                    let mut dq = vec![0i32; 64 * 64];
                    let n = if case % 8 == 7 { w * h } else { 1 + rnd(16) as usize };
                    let scale = if case % 4 == 0 { 1 << (bd + 3) } else { 1 << (bd - 2) };
                    let (hh, ww) = (h.min(32) as u64, w.min(32) as u64);
                    let (hh, ww) = if case % 3 == 0 { (1 + rnd(hh), 1 + rnd(ww)) } else { (hh, ww) };
                    let (mut rows, mut cols) = (1, 1);
                    for _ in 0..n {
                        let (i, j) = (rnd(hh) as usize, rnd(ww) as usize);
                        dq[i * 64 + j] = rnd(2 * scale) as i32 - scale as i32;
                        if dq[i * 64 + j] != 0 {
                            rows = rows.max(i + 1);
                            cols = cols.max(j + 1);
                        }
                    }
                    let mut a = vec![0i32; w * h];
                    let mut b = vec![0x5a5a5a; w * h];
                    itx::inverse_transform_2d_scalar(&dq, tx_sz, tx_type, false, bd, &mut a);
                    itx::inverse_transform_2d(&dq, tx_sz, tx_type, false, bd, &mut b);
                    if a != b {
                        return 0;
                    }
                    let looser = (rows + rnd((h.min(32) - rows + 1) as u64) as usize, cols + rnd((w.min(32) - cols + 1) as u64) as usize);
                    for within in [(rows, cols), looser] {
                        b.fill(-0x5a5a5a);
                        go(&dq, tx_sz, within, true, &mut b, tx_type, bd);
                        if a != b {
                            return 0;
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    cases
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn check_all() -> u32 {
    check()
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn bench(si: u32, full: u32, bounded: u32, iters: u32) -> i32 {
    run(si as usize, full != 0, bounded != 0, iters)
}
