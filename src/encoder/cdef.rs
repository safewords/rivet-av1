//! The encoder's CDEF search: the strengths a frame signals and the one
//! each 64x64 block uses.
//!
//! On the deblocked reconstruction, each candidate strength pair (primary,
//! secondary) is applied to every 8x8 block CDEF would filter, with the
//! decoder's own filter (`postfilter::cdef_filter_block`) and direction
//! search, and its squared error against the source summed per 64x64
//! block, for luma and chroma separately. Then up to eight
//! (luma, chroma) pairs are chosen greedily to minimise the frame's error
//! plus the cost of signalling an index per 64x64 block, and each block
//! takes its best.

use crate::consts::*;
use crate::decoder::FrameCtx;
use crate::decoder::postfilter::{cdef_direction, cdef_filter_block};
use crate::tables::CDEF_UV_DIR;

/// The CDEF parameters of a frame header (`cdef_params()`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CdefParams {
    /// `cdef_damping_minus_3`.
    pub(crate) damping_minus_3: u32,
    /// `cdef_bits`.
    pub(crate) bits: u32,
    /// `(cdef_y_pri_strength, cdef_y_sec_strength)` per index, the
    /// secondary as coded (3 meaning 4).
    pub(crate) y: Vec<(u32, u32)>,
    pub(crate) uv: Vec<(u32, u32)>,
}

impl CdefParams {
    /// Off: one index, zero strengths.
    pub(crate) fn off() -> Self {
        CdefParams {
            damping_minus_3: 0,
            bits: 0,
            y: vec![(0, 0)],
            uv: vec![(0, 0)],
        }
    }
}

/// With `Tools::cdef_fast`, the luma and chroma strengths measured on
/// every block (off among them).
const FAST_Y: usize = 10;
const FAST_UV: usize = 6;

/// Coded secondary strength to the strength (6.10.14: 3 is 4).
fn sec(coded: u32) -> i32 {
    if coded == 3 { 4 } else { coded as i32 }
}

/// Chooses the frame's CDEF parameters and each 64x64 block's index
/// (`cdef_idx` layout: `(r >> 4) * cdef_stride + (c >> 4)`; -1 where the
/// block codes none). `f.cur` holds the deblocked frame; `src` the source
/// planes (`stride` apart, padded like `f.cur`).
pub(crate) fn search(
    f: &FrameCtx,
    src: &[Vec<u16>],
    stride: &[usize],
    lambda: f64,
    thorough: bool,
    fast: bool,
    threads: usize,
) -> (CdefParams, Vec<i8>) {
    let y_pri: &[u32] = if thorough {
        &[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    } else {
        &[0, 1, 2, 4, 6, 9, 12]
    };
    let y_sec: &[u32] = &[0, 1, 2, 3];
    let uv_pri: &[u32] = if thorough {
        &[0, 1, 2, 3, 4, 6, 8, 11]
    } else {
        &[0, 1, 2, 4, 7]
    };
    let uv_sec: &[u32] = &[0, 1, 2];
    let ycands: Vec<(u32, u32)> = y_pri
        .iter()
        .flat_map(|&p| y_sec.iter().map(move |&s| (p, s)))
        .collect();
    let uvcands: Vec<(u32, u32)> = uv_pri
        .iter()
        .flat_map(|&p| uv_sec.iter().map(move |&s| (p, s)))
        .collect();
    let damping_minus_3 = damping_for(f.hdr.base_q_idx);
    let coeff_shift = f.bit_depth - 8;
    let damping = (damping_minus_3 + 3) as i32;
    // The 64x64 blocks that code an index.
    let fb_rows = f.mi_rows.div_ceil(16);
    let fb_cols = f.mi_cols.div_ceil(16);
    let mut blocks = Vec::new();
    for fr in 0..fb_rows {
        for fc in 0..fb_cols {
            if f.cdef_idx[fr * f.cdef_stride + fc] >= 0 {
                blocks.push((fr, fc));
            }
        }
    }
    let mut table = f.cdef_idx.clone();
    if blocks.is_empty() {
        return (CdefParams::off(), table);
    }
    // Each block's error for each luma and chroma strength, the blocks on
    // `threads` threads.
    let measure = |bis: &[usize], yc: &[(u32, u32)], uvc: &[(u32, u32)]| {
        crate::decoder::parallel_map(bis.len(), threads.max(1), |i| {
            measure_block(
                f,
                src,
                stride,
                blocks[bis[i]],
                yc,
                uvc,
                damping,
                coeff_shift,
            )
        })
    };
    let all: Vec<usize> = (0..blocks.len()).collect();
    let (ycands, uvcands, per_block) = if fast && blocks.len() >= 8 {
        // Two stages: every strength on a quarter of the blocks, then the
        // strengths those chose most often (and off) on the rest.
        let sample: Vec<usize> = all.iter().copied().filter(|b| b % 4 == 0).collect();
        let first = measure(&sample, &ycands, &uvcands);
        let keep = |n: usize, k: usize, rows: Vec<&[u64]>| -> Vec<usize> {
            let mut wins = vec![(0u32, 0u64); n];
            for v in rows {
                let best = (0..n).min_by_key(|&c| v[c]).unwrap_or(0);
                wins[best].0 += 1;
                for (c, w) in wins.iter_mut().enumerate() {
                    w.1 += v[c];
                }
            }
            let mut order: Vec<usize> = (1..n).collect();
            order.sort_by(|&a, &b| wins[b].0.cmp(&wins[a].0).then(wins[a].1.cmp(&wins[b].1)));
            let mut out = vec![0];
            out.extend(order.into_iter().take(k - 1));
            out.sort_unstable();
            out
        };
        let yk = keep(
            ycands.len(),
            FAST_Y,
            first.iter().map(|b| &b.0[..]).collect(),
        );
        let uvk = keep(
            uvcands.len(),
            FAST_UV,
            first.iter().map(|b| &b.1[..]).collect(),
        );
        let yc: Vec<(u32, u32)> = yk.iter().map(|&k| ycands[k]).collect();
        let uvc: Vec<(u32, u32)> = uvk.iter().map(|&k| uvcands[k]).collect();
        let rest: Vec<usize> = all.iter().copied().filter(|b| b % 4 != 0).collect();
        let second = measure(&rest, &yc, &uvc);
        let mut per_block = vec![(Vec::new(), Vec::new()); blocks.len()];
        for (i, &b) in sample.iter().enumerate() {
            per_block[b] = (
                yk.iter().map(|&k| first[i].0[k]).collect(),
                uvk.iter().map(|&k| first[i].1[k]).collect(),
            );
        }
        for (i, &b) in rest.iter().enumerate() {
            per_block[b] = second[i].clone();
        }
        (yc, uvc, per_block)
    } else {
        let per_block = measure(&all, &ycands, &uvcands);
        (ycands, uvcands, per_block)
    };
    let ny = ycands.len();
    let nuv = uvcands.len();
    let ysse: Vec<u64> = per_block.iter().flat_map(|b| b.0.iter().copied()).collect();
    let uvsse: Vec<u64> = per_block.iter().flat_map(|b| b.1.iter().copied()).collect();
    // Greedy choice of up to 2^bits (luma, chroma) pairs, for each bits.
    let nb = blocks.len();
    let best_for = |set: &[(usize, usize)]| -> (f64, Vec<usize>) {
        let mut total = 0f64;
        let mut pick = vec![0usize; nb];
        for b in 0..nb {
            let mut best = u64::MAX;
            for (i, &(yk, uk)) in set.iter().enumerate() {
                let v = ysse[b * ny + yk] + uvsse[b * nuv + uk];
                if v < best {
                    best = v;
                    pick[b] = i;
                }
            }
            total += best as f64;
        }
        (total, pick)
    };
    let mut best: (f64, Vec<(usize, usize)>, Vec<usize>) = (f64::MAX, Vec::new(), Vec::new());
    let mut set: Vec<(usize, usize)> = Vec::new();
    for bits in 0..=3u32 {
        while set.len() < 1 << bits {
            let mut add = (f64::MAX, (0, 0));
            for yk in 0..ny {
                for uk in 0..nuv {
                    if set.contains(&(yk, uk)) {
                        continue;
                    }
                    set.push((yk, uk));
                    let (t, _) = best_for(&set);
                    set.pop();
                    if t < add.0 {
                        add = (t, (yk, uk));
                    }
                }
            }
            set.push(add.1);
        }
        let (t, pick) = best_for(&set);
        let cost = t + lambda * (bits as f64) * nb as f64;
        if cost < best.0 {
            best = (cost, set.clone(), pick);
        }
    }
    let (_, set, pick) = best;
    let bits = set.len().trailing_zeros();
    for (b, &(fr, fc)) in blocks.iter().enumerate() {
        table[fr * f.cdef_stride + fc] = pick[b] as i8;
    }
    (
        CdefParams {
            damping_minus_3,
            bits,
            y: set.iter().map(|&(yk, _)| ycands[yk]).collect(),
            uv: set.iter().map(|&(_, uk)| uvcands[uk]).collect(),
        },
        table,
    )
}

/// One 64x64 block's (`(fr, fc)`) error for each luma and each chroma
/// strength in `ycands` and `uvcands`.
#[allow(clippy::too_many_arguments)]
fn measure_block(
    f: &FrameCtx,
    src: &[Vec<u16>],
    stride: &[usize],
    (fr, fc): (usize, usize),
    ycands: &[(u32, u32)],
    uvcands: &[(u32, u32)],
    damping: i32,
    coeff_shift: u32,
) -> (Vec<u64>, Vec<u64>) {
    let (ny, nuv) = (ycands.len(), uvcands.len());
    let cols = f.ms;
    let mut ysse = vec![0u64; ny];
    let mut uvsse = vec![0u64; nuv];
    let mut res = [0u16; 64];
    for r in (fr * 16..(fr * 16 + 16).min(f.mi_rows)).step_by(2) {
        for c in (fc * 16..(fc * 16 + 16).min(f.mi_cols)).step_by(2) {
            let sk = |rr: usize, cc: usize| f.mi[rr * cols + cc].skip;
            if sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1) {
                continue;
            }
            let (y_dir, var) = cdef_direction(f, r, c);
            // Luma.
            let p0 = &f.cur.planes[0];
            let (x0, y0) = (c * MI_SIZE, r * MI_SIZE);
            let xlim = f.mi_cols * MI_SIZE;
            let ylim = f.mi_rows * MI_SIZE;
            for (k, &(pri, s)) in ycands.iter().enumerate() {
                let mut pri_str = (pri as i32) << coeff_shift;
                let sec_str = sec(s) << coeff_shift;
                let dir = if pri_str == 0 { 0 } else { y_dir };
                let var_str = if (var >> 6) != 0 {
                    (crate::bits::floor_log2((var >> 6) as u32) as i32).min(12)
                } else {
                    0
                };
                pri_str = if var != 0 {
                    (pri_str * (4 + var_str) + 8) >> 4
                } else {
                    0
                };
                let d = damping + coeff_shift as i32;
                cdef_filter_block(
                    p0,
                    x0,
                    y0,
                    8,
                    8,
                    xlim,
                    ylim,
                    pri_str,
                    sec_str,
                    d,
                    dir,
                    coeff_shift,
                    &mut res,
                );
                ysse[k] += block_sse(&res, 8, 8, &src[0], stride[0], x0, y0, f, 0);
            }
            // Chroma.
            if f.num_planes > 1 {
                let (w, h) = (8 >> f.ssx, 8 >> f.ssy);
                let (cx0, cy0) = (x0 >> f.ssx, y0 >> f.ssy);
                let cxlim = xlim >> f.ssx;
                let cylim = ylim >> f.ssy;
                for (k, &(pri, s)) in uvcands.iter().enumerate() {
                    let pri_str = (pri as i32) << coeff_shift;
                    let sec_str = sec(s) << coeff_shift;
                    let dir = if pri_str == 0 {
                        0
                    } else {
                        CDEF_UV_DIR[f.ssx][f.ssy][y_dir]
                    };
                    let d = damping + coeff_shift as i32 - 1;
                    let mut total = 0;
                    for p in 1..3 {
                        cdef_filter_block(
                            &f.cur.planes[p],
                            cx0,
                            cy0,
                            w,
                            h,
                            cxlim,
                            cylim,
                            pri_str,
                            sec_str,
                            d,
                            dir,
                            coeff_shift,
                            &mut res,
                        );
                        total += block_sse(&res, w, h, &src[p], stride[p], cx0, cy0, f, p);
                    }
                    uvsse[k] += total;
                }
            }
        }
    }
    (ysse, uvsse)
}

/// The damping a quantiser gets: stronger filtering at coarser ones.
fn damping_for(qidx: u32) -> u32 {
    (qidx >> 6).min(3)
}

/// Squared error of a filtered block against the source, inside the frame.
#[allow(clippy::too_many_arguments)]
fn block_sse(
    res: &[u16; 64],
    w: usize,
    h: usize,
    src: &[u16],
    stride: usize,
    x0: usize,
    y0: usize,
    f: &FrameCtx,
    plane: usize,
) -> u64 {
    let (ssx, ssy) = f.plane_ss(plane);
    let fw = (f.hdr.frame_width as usize + ssx) >> ssx;
    let fh = (f.hdr.frame_height as usize + ssy) >> ssy;
    let mut s = 0u64;
    for i in 0..h.min(fh.saturating_sub(y0)) {
        for j in 0..w.min(fw.saturating_sub(x0)) {
            let d = res[i * w + j] as i64 - src[(y0 + i) * stride + x0 + j] as i64;
            s += (d * d) as u64;
        }
    }
    s
}
