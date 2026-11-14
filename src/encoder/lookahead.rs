//! Rate control's lookahead: what each frame of a group will cost to
//! predict, measured on the group's sources before it is coded.
//!
//! Each frame is reduced to a half-size luma (8-bit scale). Each 8x8 block
//! of it (16x16 of the frame) is predicted as the encoder might: intra
//! from the source samples above and to its left (DC, vertical or
//! horizontal), and between frames from the sources of the frames it will
//! be predicted from in the group's pyramid (the nearest frames coded
//! before it on either side), each by a small motion search on sums of
//! absolute differences, and from both at once (their average). A block's
//! cost is the SATD of the cheapest of these predictions. Rate control
//! takes a frame's average (`rc::Complexity`): an alt-ref frame predicted
//! across a whole group, or a frame after a scene cut, is priced as what
//! it is before it is coded.

use crate::Frame;
use crate::dsp::enc::{sad, satd};

/// The margin around a half-size plane, and so the farthest a vector
/// reaches past its edge (in half-size samples).
const MARGIN: usize = 32;
/// The block, in half-size samples.
const B: usize = 8;

/// A frame's half-size luma, its size rounded up to whole blocks and
/// padded by `MARGIN` on every side (edges repeated).
pub(crate) struct Plane {
    data: Vec<u16>,
    stride: usize,
    /// Blocks across and down.
    cols: usize,
    rows: usize,
}

impl Plane {
    /// The half-size luma of `frame`.
    pub(crate) fn new(frame: &Frame) -> Plane {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let shift = 2 + frame.bit_depth - 8;
        let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
        let (cols, rows) = (hw.div_ceil(B), hh.div_ceil(B));
        let (stride, height) = (cols * B + 2 * MARGIN, rows * B + 2 * MARGIN);
        let mut data = vec![0u16; stride * height];
        // Each pair of rows (the last repeated when the height is odd)
        // summed, then each pair of columns, into the rows of the blocks.
        let mut sum = vec![0u32; 2 * hw];
        let row = |y: usize, out: &mut [u32]| {
            let pl = frame.planes[0];
            if frame.bit_depth > 8 {
                let at = pl.offset + 2 * y * pl.width as usize;
                let samples = frame.data[at..][..2 * w].as_chunks::<2>().0;
                for (o, b) in out.iter_mut().zip(samples) {
                    *o += u16::from_le_bytes(*b) as u32;
                }
            } else {
                let at = pl.offset + y * pl.width as usize;
                for (o, &b) in out.iter_mut().zip(&frame.data[at..][..w]) {
                    *o += b as u32;
                }
            }
        };
        for hy in 0..hh {
            sum.fill(0);
            row(2 * hy, &mut sum[..w]);
            row((2 * hy + 1).min(h - 1), &mut sum[..w]);
            if w % 2 == 1 {
                sum[w] = sum[w - 1];
            }
            let o = (hy + MARGIN) * stride + MARGIN;
            for (d, s) in data[o..o + hw].iter_mut().zip(sum.as_chunks::<2>().0) {
                *d = ((s[0] + s[1] + (1 << (shift - 1))) >> shift) as u16;
            }
        }
        // The margins, and the blocks past the frame's edge, repeat it.
        for hy in MARGIN..MARGIN + hh {
            let r = &mut data[hy * stride..(hy + 1) * stride];
            let (first, last) = (r[MARGIN], r[MARGIN + hw - 1]);
            r[..MARGIN].fill(first);
            r[MARGIN + hw..].fill(last);
        }
        for hy in 0..height {
            let src = hy.clamp(MARGIN, MARGIN + hh - 1);
            if src != hy {
                data.copy_within(src * stride..(src + 1) * stride, hy * stride);
            }
        }
        Plane {
            data,
            stride,
            cols,
            rows,
        }
    }

    /// The offset of half-size sample `(x, y)` (from the corner of the
    /// blocks; negative reaches into the margin).
    fn at(&self, x: isize, y: isize) -> usize {
        (y + MARGIN as isize) as usize * self.stride + (x + MARGIN as isize) as usize
    }
}

/// A frame's average cost per half-size sample: predicted intra, and the
/// cheapest of that and the predictions from `refs` (none, one or two
/// frames, of the same size).
pub(crate) fn costs(cur: &Plane, refs: &[&Plane]) -> (f64, f64) {
    let n = cur.cols * cur.rows;
    // Each reference's vectors, the next blocks' candidates.
    let mut mvs = vec![(0isize, 0isize); refs.len() * n];
    let (mut intra_t, mut best_t) = (0u64, 0u64);
    let mut pred = [0u16; B * B];
    for by in 0..cur.rows {
        for bx in 0..cur.cols {
            let b = by * cur.cols + bx;
            let (x, y) = ((bx * B) as isize, (by * B) as isize);
            let src = &cur.data[cur.at(x, y)..];
            let intra = intra_cost(cur, x, y, &mut pred);
            intra_t += intra;
            let mut best = intra;
            let mut found = [[0u16; B * B]; 2];
            for (r, rf) in refs.iter().enumerate() {
                let mv = search(cur, rf, x, y, &mvs[r * n..], bx, by, &mut found[r]);
                mvs[r * n + b] = mv;
                best = best.min(satd(src, cur.stride, &found[r], B, B, B));
            }
            if refs.len() == 2 {
                for (p, (a, c)) in pred.iter_mut().zip(found[0].iter().zip(&found[1])) {
                    *p = (a + c).div_ceil(2);
                }
                best = best.min(satd(src, cur.stride, &pred, B, B, B));
            }
            best_t += best;
        }
    }
    let samples = (n * B * B) as f64;
    (intra_t as f64 / samples, best_t as f64 / samples)
}

/// The SATD of the block at `(x, y)` predicted from the samples above and
/// to its left: DC, vertical or horizontal, the cheapest (at the frame's
/// edge, from what there is; at its corner, mid-grey).
fn intra_cost(p: &Plane, x: isize, y: isize, pred: &mut [u16; B * B]) -> u64 {
    let src = &p.data[p.at(x, y)..];
    let top: Option<[u16; B]> =
        (y > 0).then(|| std::array::from_fn(|j| p.data[p.at(x + j as isize, y - 1)]));
    let left: Option<[u16; B]> =
        (x > 0).then(|| std::array::from_fn(|i| p.data[p.at(x - 1, y + i as isize)]));
    let sum = |e: &[u16; B]| e.iter().map(|&v| v as u32).sum::<u32>();
    let dc = match (&top, &left) {
        (Some(t), Some(l)) => (sum(t) + sum(l)).div_ceil(2 * B as u32),
        (Some(e), None) | (None, Some(e)) => sum(e).div_ceil(B as u32),
        (None, None) => 128,
    };
    pred.fill(dc as u16);
    let mut best = satd(src, p.stride, pred, B, B, B);
    if let Some(t) = &top {
        for row in pred.as_chunks_mut::<B>().0 {
            row.copy_from_slice(t);
        }
        best = best.min(satd(src, p.stride, pred, B, B, B));
    }
    if let Some(l) = &left {
        for (row, &v) in pred.as_chunks_mut::<B>().0.iter_mut().zip(l) {
            row.fill(v);
        }
        best = best.min(satd(src, p.stride, pred, B, B, B));
    }
    best
}

/// The vector, in whole half-size samples, of the block at `(x, y)` (block
/// `(bx, by)`) into `rf`: the best by SAD of none and the vectors of the
/// blocks before it (`mvs`), then a diamond narrowing from 4 samples to 1;
/// and in `pred` its prediction, or the prediction half a sample from it
/// across or down (the average of two) when that is better.
#[allow(clippy::too_many_arguments)]
fn search(
    cur: &Plane,
    rf: &Plane,
    x: isize,
    y: isize,
    mvs: &[(isize, isize)],
    bx: usize,
    by: usize,
    pred: &mut [u16; B * B],
) -> (isize, isize) {
    let src = &cur.data[cur.at(x, y)..];
    let m = MARGIN as isize;
    let (xmax, ymax) = (
        (cur.cols * B) as isize - B as isize + m,
        (cur.rows * B) as isize - B as isize + m,
    );
    let inside =
        |v: (isize, isize)| (-m..=xmax).contains(&(x + v.0)) && (-m..=ymax).contains(&(y + v.1));
    let cost = |v: (isize, isize)| {
        let o = rf.at(x + v.0, y + v.1);
        sad(src, cur.stride, &rf.data[o..], rf.stride, B, B, u64::MAX)
    };
    let b = by * cur.cols + bx;
    let mut cands = [(0, 0); 4];
    let mut nc = 1;
    if bx > 0 {
        cands[nc] = mvs[b - 1];
        nc += 1;
    }
    if by > 0 {
        cands[nc] = mvs[b - cur.cols];
        nc += 1;
        if bx + 1 < cur.cols {
            cands[nc] = mvs[b - cur.cols + 1];
            nc += 1;
        }
    }
    let mut mv = (0, 0);
    let mut c = cost(mv);
    for &v in &cands[1..nc] {
        if inside(v) {
            let k = cost(v);
            if k < c {
                (mv, c) = (v, k);
            }
        }
    }
    for step in [4, 2, 1] {
        loop {
            let mut moved = false;
            for d in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                let v = (mv.0 + d.0, mv.1 + d.1);
                if inside(v) {
                    let k = cost(v);
                    if k < c {
                        (mv, c) = (v, k);
                        moved = true;
                    }
                }
            }
            if !moved {
                break;
            }
        }
    }
    let o = rf.at(x + mv.0, y + mv.1);
    for (i, row) in pred.as_chunks_mut::<B>().0.iter_mut().enumerate() {
        row.copy_from_slice(&rf.data[o + i * rf.stride..][..B]);
    }
    let mut half = [0u16; B * B];
    for d in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let v = (mv.0 + d.0, mv.1 + d.1);
        if !inside(v) {
            continue;
        }
        let o2 = rf.at(x + v.0, y + v.1);
        let mut sad = 0u64;
        for i in 0..B {
            let (a, c) = (
                &rf.data[o + i * rf.stride..][..B],
                &rf.data[o2 + i * rf.stride..][..B],
            );
            let s = &src[i * cur.stride..][..B];
            for j in 0..B {
                let p = (a[j] + c[j]).div_ceil(2);
                half[i * B + j] = p;
                sad += s[j].abs_diff(p) as u64;
            }
        }
        if sad < c {
            c = sad;
            *pred = half;
        }
    }
    mv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaFormat;

    /// A textured frame, its luma moved `(dx, dy)` samples, at `bit_depth`.
    fn textured(w: u32, h: u32, dx: i64, dy: i64, bit_depth: u32) -> Frame {
        let mut f = Frame::new(w, h, bit_depth, ChromaFormat::Yuv420);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as i64 + dx, y as i64 + dy);
                // (Smooth gradients and hard edges, as a picture has.)
                let t = 120.0
                    + 50.0 * (u as f64 * 0.21).sin() * (v as f64 * 0.17).cos()
                    + 40.0 * ((u.div_euclid(16) + v.div_euclid(12)) % 2) as f64;
                f.set_sample(0, x, y, (t as u16) << (bit_depth - 8));
            }
        }
        f
    }

    /// The half-size luma is on the 8-bit scale whatever the bit depth,
    /// and frames of any size, odd or smaller than a block, are measured.
    #[test]
    fn planes_at_every_depth_and_size() {
        let (a, b) = (
            Plane::new(&textured(37, 21, 0, 0, 8)),
            Plane::new(&textured(37, 21, 0, 0, 10)),
        );
        assert_eq!(a.data, b.data);
        for (w, h) in [(1, 1), (2, 3), (17, 9), (64, 64)] {
            let p = Plane::new(&textured(w, h, 0, 0, 8));
            let (intra, best) = costs(&p, &[&p]);
            assert!(intra.is_finite() && best <= intra, "{w}x{h}");
        }
    }

    /// A frame moved against its reference, by whole and by half samples
    /// of the half-size luma, is found: predicted from it, it costs a
    /// fraction of what it costs intra, and from both of two references
    /// either side no more than from either.
    #[test]
    fn motion_is_found() {
        let r0 = Plane::new(&textured(128, 96, 0, 0, 8));
        for (dx, dy) in [(6, 4), (-10, 2), (5, -3)] {
            let cur = Plane::new(&textured(128, 96, dx, dy, 8));
            let r1 = Plane::new(&textured(128, 96, 2 * dx, 2 * dy, 8));
            let (intra, one) = costs(&cur, &[&r0]);
            let (_, two) = costs(&cur, &[&r0, &r1]);
            assert!(
                one < 0.5 * intra,
                "({dx}, {dy}): {one} against {intra} intra"
            );
            assert!(two <= one, "({dx}, {dy}): {two} from both, {one} from one");
        }
    }
}
