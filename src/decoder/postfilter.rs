//! The in-loop filters applied when a frame is complete: the loop filter
//! (7.14), CDEF (7.15), super-resolution upscaling (7.16) and loop
//! restoration (7.17).

use crate::bits::floor_log2;
use crate::consts::*;
use crate::decoder::FrameCtx;
use crate::decoder::state::{FrameBuf, Mi, PlaneBuf};
use crate::header::FrameHeader;
use crate::tables::*;

// ---------------------------------------------------------------------------
// Loop filter (7.14)
// ---------------------------------------------------------------------------

/// A plane's samples written by several threads at once, each to a part
/// no other thread reads or writes (rows or columns, as the caller
/// arranges).
#[derive(Clone, Copy)]
pub(crate) struct SharedPlane {
    ptr: *mut u16,
    len: usize,
    pub(crate) stride: usize,
}

// SAFETY: the threads sharing a `SharedPlane` touch disjoint samples (see
// each user), and the plane outlives them (scoped threads).
unsafe impl Send for SharedPlane {}
unsafe impl Sync for SharedPlane {}

impl SharedPlane {
    pub(crate) fn new(p: &mut PlaneBuf) -> Self {
        SharedPlane {
            ptr: p.data.as_mut_ptr(),
            len: p.data.len(),
            stride: p.stride,
        }
    }

    #[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
    #[inline(always)]
    fn get(&self, i: usize) -> u16 {
        assert!(i < self.len);
        // SAFETY: in bounds; no other thread writes this sample.
        unsafe { *self.ptr.add(i) }
    }

    #[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "wasm32"))))]
    #[inline(always)]
    fn set(&self, i: usize, v: u16) {
        assert!(i < self.len);
        // SAFETY: in bounds; no other thread reads or writes this sample.
        unsafe { *self.ptr.add(i) = v }
    }

    /// Row `y`, `x0..x0 + n`, written from `src`.
    #[inline]
    pub(crate) fn write_row(&self, x0: usize, y: usize, src: &[u16]) {
        let o = y * self.stride + x0;
        assert!(o + src.len() <= self.len);
        // SAFETY: in bounds; no other thread reads or writes these samples.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(o), src.len()) }
    }
}

/// The loop filter process (7.14.1), in place on `CurrFrame`, on up to
/// `threads` threads. The first pass
/// (vertical edges) filters along rows, so bands of rows are independent;
/// the second (horizontal edges) along columns, so bands of columns are.
///
/// The specification visits every 4x4 unit of a plane and decides there
/// whether its edge is filtered, how wide and how strongly (7.14.2 to
/// 7.14.5); [`edge_unit`] is that decision, made in the order that costs
/// least (most units are not on a transform edge, and that needs one byte),
/// with the levels of 7.14.4 looked up in [`LfLevels`].
pub(crate) fn loop_filter_threads(f: &mut FrameCtx, threads: usize) {
    let levels = LfLevels::new(&f.hdr);
    for plane in 0..f.num_planes {
        if plane == 0 || f.hdr.loop_filter_level[1 + plane] != 0 {
            let mut pb = std::mem::take(&mut f.cur.planes[plane]);
            let sp = SharedPlane::new(&mut pb);
            let fr: &FrameCtx = f;
            for pass in 0..2 {
                // Bands of 16 mode info units (64 luma samples).
                let extent = if pass == 0 { fr.mi_rows } else { fr.mi_cols };
                let bands = extent.div_ceil(16);
                let band = |b: usize| {
                    let (r0, r1, c0, c1) = if pass == 0 {
                        (b * 16, ((b + 1) * 16).min(fr.mi_rows), 0, fr.mi_cols)
                    } else {
                        (0, fr.mi_rows, b * 16, ((b + 1) * 16).min(fr.mi_cols))
                    };
                    filter_band(fr, &levels, sp, plane, pass, [r0, r1, c0, c1]);
                };
                if threads > 1 {
                    crate::decoder::parallel_map(bands, threads, band);
                } else {
                    (0..bands).for_each(band);
                }
            }
            f.cur.planes[plane] = pb;
        }
    }
}

/// The filter levels of a frame (7.14.4) for a block with no `delta_lf`:
/// they depend on the block's segment, its first reference frame and
/// whether its mode is one of the `mode_deltas` kind, so there are 144 of
/// them for each of the four `i` (luma vertical edges, luma horizontal
/// edges, U, V); and the limits that go with each level.
pub(crate) struct LfLevels {
    /// `[i][segment][ref_frame + 1][mode_type]`.
    lvl: [[[[u8; 2]; 9]; 8]; 4],
    /// `[lvl]`: `limit`, `blimit`, `thresh`.
    limits: [[u8; 3]; 64],
    /// `[pass][tx size]`: a transform's width (pass 0) or height (pass 1)
    /// in 4x4 units, less one. A unit is on a transform edge where its
    /// column or row has none of these bits. (Zero for the sizes there are
    /// not: [`edge_unit`] is asked about those.)
    tx_units: [[u8; 32]; 2],
}

impl LfLevels {
    pub(crate) fn new(h: &FrameHeader) -> Self {
        let mut lvl = [[[[0u8; 2]; 9]; 8]; 4];
        for (i, per_i) in lvl.iter_mut().enumerate() {
            for (segment, per_segment) in per_i.iter_mut().enumerate() {
                for (r, per_ref) in per_segment.iter_mut().enumerate() {
                    for (mode_type, l) in per_ref.iter_mut().enumerate() {
                        *l = filter_level(h, i, segment, r as i32 - 1, mode_type, 0) as u8;
                    }
                }
            }
        }
        let mut limits = [[0u8; 3]; 64];
        for (l, lim) in limits.iter_mut().enumerate() {
            let (limit, blimit, thresh) = filter_limits(h.loop_filter_sharpness, l as i32);
            *lim = [limit as u8, blimit as u8, thresh as u8];
        }
        let mut tx_units = [[0u8; 32]; 2];
        for (tx_sz, (&w, &h)) in TX_WIDTH.iter().zip(&TX_HEIGHT).enumerate() {
            tx_units[0][tx_sz] = (w / MI_SIZE - 1) as u8;
            tx_units[1][tx_sz] = (h / MI_SIZE - 1) as u8;
        }
        LfLevels {
            lvl,
            limits,
            tx_units,
        }
    }

    /// `lvl` of 7.14.4 for the block of `m`.
    #[inline(always)]
    fn level(&self, h: &FrameHeader, m: &Mi, i: usize) -> usize {
        let segment = m.segment_id as usize;
        let rf = m.ref_frame[0] as i32;
        let mode = m.y_mode as usize;
        let mode_type = (mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV) as usize;
        let delta_lf = if h.delta_lf_multi {
            m.delta_lf[i]
        } else {
            m.delta_lf[0]
        };
        if delta_lf == 0 && segment < 8 && (-1..8).contains(&rf) {
            self.lvl[i][segment][(rf + 1) as usize][mode_type] as usize
        } else {
            filter_level(h, i, segment, rf, mode_type, delta_lf as i32) as usize
        }
    }
}

/// `lvl` of the adaptive filter strength process (7.14.4, with the selection
/// process 7.14.5): `i` is 0 and 1 for luma's vertical and horizontal
/// edges, 2 for U and 3 for V.
fn filter_level(
    h: &FrameHeader,
    i: usize,
    segment: usize,
    rf: i32,
    mode_type: usize,
    delta_lf: i32,
) -> i32 {
    let mut lvl_seg = clip3(0, MAX_LOOP_FILTER, delta_lf + h.loop_filter_level[i]);
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if h.segmentation_enabled && h.feature_enabled[segment][feature] {
        lvl_seg = clip3(
            0,
            MAX_LOOP_FILTER,
            h.feature_data[segment][feature] + lvl_seg,
        );
    }
    if h.loop_filter_delta_enabled {
        let n_shift = lvl_seg >> 5;
        if rf == INTRA_FRAME {
            lvl_seg += h.loop_filter_ref_deltas[INTRA_FRAME as usize] << n_shift;
        } else if rf > INTRA_FRAME {
            lvl_seg += (h.loop_filter_ref_deltas[rf as usize] << n_shift)
                + (h.loop_filter_mode_deltas[mode_type] << n_shift);
        }
        lvl_seg = clip3(0, MAX_LOOP_FILTER, lvl_seg);
    }
    lvl_seg
}

/// `limit`, `blimit` and `thresh` of 7.14.4 for a level.
fn filter_limits(sharp: i32, lvl: i32) -> (i32, i32, i32) {
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        clip3(1, 9 - sharp, lvl >> shift)
    } else {
        (lvl >> shift).max(1)
    };
    (limit, 2 * (lvl + 2) + limit, lvl >> 4)
}

/// One 4x4 unit's edge: the filter size of 7.14.3 (0 for an edge that is
/// not filtered) and the level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct EdgeUnit {
    size: u8,
    lvl: u8,
}

const NO_EDGE: EdgeUnit = EdgeUnit { size: 0, lvl: 0 };

/// The edge loop filter process (7.14.2) up to its sample filtering, for
/// the unit at row `pr`, column `pc` of the plane's 4x4 units: whether the
/// unit's left (`pass` 0) or upper (`pass` 1) edge is filtered, and with
/// what. The caller keeps to the units on screen (`pc` at least 1 in pass
/// 0, `pr` at least 1 in pass 1).
#[inline(always)]
fn edge_unit(
    f: &FrameCtx,
    levels: &LfLevels,
    plane: usize,
    pass: usize,
    pr: usize,
    pc: usize,
) -> EdgeUnit {
    let ms = f.ms;
    let txs = &f.lf_tx_sizes[plane];
    let ti = pr * ms + pc;
    let tx_sz = txs[ti] as usize;
    // Transform sizes and block sizes are powers of two.
    let (at, tx_dim) = if pass == 0 {
        (pc * MI_SIZE, TX_WIDTH[tx_sz])
    } else {
        (pr * MI_SIZE, TX_HEIGHT[tx_sz])
    };
    if at & (tx_dim - 1) != 0 {
        // Not a transform edge, as nearly every unit.
        return NO_EDGE;
    }
    let (sub_x, sub_y) = f.plane_ss(plane);
    let row = (pr << sub_y) | sub_y;
    let col = (pc << sub_x) | sub_x;
    let m = &f.mi[row * ms + col];
    if m.skip && m.ref_frame[0] as i32 > INTRA_FRAME {
        // A skipped inter block is filtered on its own edges alone.
        let plane_size = f.plane_residual_size(m.mi_size as usize, plane);
        let block_dim = if pass == 0 {
            block_width(plane_size)
        } else {
            block_height(plane_size)
        };
        if at & (block_dim - 1) != 0 {
            return NO_EDGE;
        }
    }
    // The filter size process (7.14.3).
    let (prev_tx_sz, prev_mi) = if pass == 0 {
        (txs[ti - 1] as usize, row * ms + col - (1 << sub_x))
    } else {
        (txs[ti - ms] as usize, (row - (1 << sub_y)) * ms + col)
    };
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz].min(tx_dim)
    } else {
        TX_HEIGHT[prev_tx_sz].min(tx_dim)
    };
    let size = base_size.min(if plane == 0 { 16 } else { 8 });
    let i = if plane == 0 { pass } else { plane + 1 };
    let mut lvl = levels.level(&f.hdr, m, i);
    if lvl == 0 {
        lvl = levels.level(&f.hdr, &f.mi[prev_mi], i);
    }
    if lvl == 0 {
        return NO_EDGE;
    }
    EdgeUnit {
        size: size as u8,
        lvl: lvl as u8,
    }
}

/// The loop filter of one pass over the mode info rows `r0..r1` and columns
/// `c0..c1` of a plane (`area`), a band no other thread is filtering.
///
/// Units are taken two at a time along their edge (two rows of units for
/// the vertical edges of pass 0, two columns for the horizontal edges of
/// pass 1), which is eight samples: what one call of a filter kernel takes.
/// The pairs lie inside the band, whose first row and column are multiples
/// of 16.
fn filter_band(
    f: &FrameCtx,
    levels: &LfLevels,
    p: SharedPlane,
    plane: usize,
    pass: usize,
    area: [usize; 4],
) {
    let [r0, r1, c0, c1] = area;
    let (sub_x, sub_y) = f.plane_ss(plane);
    // On screen (7.14.2): the unit's luma position is inside the frame.
    let r1 = r1.min((f.hdr.frame_height as usize).div_ceil(MI_SIZE));
    let c1 = c1.min((f.hdr.frame_width as usize).div_ceil(MI_SIZE));
    // The plane's units: every `1 << sub`-th mode info row and column.
    let (pr0, pr1) = (r0 >> sub_y, (r1 + sub_y) >> sub_y);
    let (pc0, pc1) = (c0 >> sub_x, (c1 + sub_x) >> sub_x);
    let stride = p.stride;
    let ctx = FilterCtx {
        p,
        pass,
        limits: &levels.limits,
        chroma: plane != 0,
        bit_depth: f.bit_depth,
    };
    // Most units are not on a transform edge, which their transform size
    // alone says: a row of sizes is scanned for the ones that are, and only
    // those go through `edge_unit`.
    let ms = f.ms;
    let txs = &f.lf_tx_sizes[plane][..];
    let tx_units = &levels.tx_units[pass];
    let on_tx_edge = |at: usize, tx_sz: u8| at & tx_units[(tx_sz & 31) as usize] as usize == 0;
    let unit = |on: bool, pr: usize, pc: usize| {
        if on {
            edge_unit(f, levels, plane, pass, pr, pc)
        } else {
            NO_EDGE
        }
    };
    if pass == 0 {
        let mut pr = pr0;
        while pr < pr1 {
            let second = pr + 1 < pr1;
            let sizes_a = &txs[pr * ms..pr * ms + pc1];
            let sizes_b = &txs[(pr + 1) * ms..(pr + 1) * ms + pc1];
            // No edge at the frame's left.
            for pc in pc0.max(1)..pc1 {
                let on_a = on_tx_edge(pc, sizes_a[pc]);
                let on_b = second && on_tx_edge(pc, sizes_b[pc]);
                if !(on_a || on_b) {
                    continue;
                }
                let a = unit(on_a, pr, pc);
                let b = unit(on_b, pr + 1, pc);
                if a != NO_EDGE || b != NO_EDGE {
                    let pos = pr * MI_SIZE * stride + pc * MI_SIZE;
                    ctx.filter_pair(pos, a, b);
                }
            }
            pr += 2;
        }
    } else {
        // No edge at the frame's top.
        for pr in pr0.max(1)..pr1 {
            let sizes = &txs[pr * ms..pr * ms + pc1];
            let mut pc = pc0;
            while pc < pc1 {
                let on_a = on_tx_edge(pr, sizes[pc]);
                let on_b = pc + 1 < pc1 && on_tx_edge(pr, sizes[pc + 1]);
                if on_a || on_b {
                    let a = unit(on_a, pr, pc);
                    let b = unit(on_b, pr, pc + 1);
                    if a != NO_EDGE || b != NO_EDGE {
                        let pos = pr * MI_SIZE * stride + pc * MI_SIZE;
                        ctx.filter_pair(pos, a, b);
                    }
                }
                pc += 2;
            }
        }
    }
}

/// What the sample filtering of a plane's edges needs besides the edge.
struct FilterCtx<'a> {
    p: SharedPlane,
    pass: usize,
    limits: &'a [[u8; 3]; 64],
    chroma: bool,
    bit_depth: u32,
}

impl FilterCtx<'_> {
    /// Filters two units' edges (7.14.6 for each of their samples): `a` at
    /// sample `pos`, `b` four samples further along the edge.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[inline(always)]
    fn filter_pair(&self, pos: usize, a: EdgeUnit, b: EdgeUnit) {
        if a.size == b.size {
            // The usual case: one kernel call for the eight samples.
            self.filter_lanes8(pos, a.size, [a.lvl, b.lvl], [true, true]);
        } else {
            // A kernel is for one filter size: each unit by itself, the
            // other one's lanes left as they are.
            if a != NO_EDGE {
                self.filter_lanes8(pos, a.size, [a.lvl; 2], [true, false]);
            }
            if b != NO_EDGE {
                self.filter_lanes8(pos, b.size, [b.lvl; 2], [false, true]);
            }
        }
    }

    /// The eight samples along the edge from `pos` through the kernel of
    /// the filter size, each half with the limits of its level if it is
    /// `on`.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[inline(always)]
    fn filter_lanes8(&self, pos: usize, size: u8, lvl: [u8; 2], on: [bool; 2]) {
        let pair = crate::dsp::lf::Pair {
            limits: lvl.map(|l| self.limits[l as usize]),
            on,
            filter_size: size as usize,
            chroma: self.chroma,
            bit_depth: self.bit_depth,
        };
        let p = self.p;
        // SAFETY: `ptr` is the plane's `len` samples. The kernel reads and
        // writes the eight lanes within eight samples of the edge and
        // nothing else, and those are this thread's: the pair lies in its
        // band (see `filter_band`), which is whole rows in pass 0, where
        // the kernel moves along the rows, and whole columns in pass 1,
        // where it moves along the columns.
        unsafe { crate::dsp::lf::filter_pair(p.ptr, p.len, pos, p.stride, self.pass == 0, &pair) };
    }

    /// Filters two units' edges (7.14.6 for each of their samples): `a` at
    /// sample `pos`, `b` four samples further along the edge.
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    #[inline(always)]
    fn filter_pair(&self, pos: usize, a: EdgeUnit, b: EdgeUnit) {
        // `step` is the distance between samples across the edge, `along`
        // between samples along it.
        let (step, along) = if self.pass == 0 {
            (1, self.p.stride)
        } else {
            (self.p.stride, 1)
        };
        if a != NO_EDGE {
            self.filter_unit(pos, step, along, a);
        }
        if b != NO_EDGE {
            self.filter_unit(pos + 4 * along, step, along, b);
        }
    }

    /// The edge's filter parameters for the one-unit filters.
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn params(&self, e: EdgeUnit) -> crate::dsp::lf::Params {
        let [limit, blimit, thresh] = self.limits[e.lvl as usize];
        crate::dsp::lf::Params {
            limit: limit as i32,
            blimit: blimit as i32,
            thresh: thresh as i32,
            filter_size: e.size as usize,
            chroma: self.chroma,
            bit_depth: self.bit_depth,
        }
    }

    /// One unit's four samples along the edge, a sample at a time on the
    /// plane.
    #[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
    #[inline(always)]
    fn filter_unit(&self, pos: usize, step: usize, along: usize, e: EdgeUnit) {
        let p = self.p;
        // SAFETY: `ptr` is the plane's `len` samples. The filter reads and
        // writes the unit's four lanes within seven samples of the edge
        // and nothing else, and those are this thread's: the unit lies in
        // its band, which is whole rows in pass 0, where the filter moves
        // along the rows, and whole columns in pass 1, where it moves
        // along the columns.
        unsafe { crate::dsp::lf::filter_unit(p.ptr, p.len, pos, step, along, &self.params(e)) };
    }

    /// One unit's four samples along the edge through the lane filter: the
    /// samples across the edge gathered, filtered together and written
    /// back.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    fn filter_unit(&self, pos: usize, step: usize, along: usize, e: EdgeUnit) {
        let p = self.p;
        let filter_size = e.size as usize;
        let (base, step, along) = (pos as isize, step as isize, along as isize);
        let reach: isize = if filter_size >= 16 { 7 } else { 4 };
        let mut v: crate::dsp::lf::Edge = [[0; 4]; 14];
        for l in 0..4isize {
            for k in -reach..reach {
                v[(7 + k) as usize][l as usize] =
                    p.get((base + l * along + k * step) as usize) as i32;
            }
        }
        crate::dsp::lf::filter(&mut v, &self.params(e));
        let span: isize = if filter_size >= 16 { 6 } else { 3 };
        for l in 0..4isize {
            for k in -span..span {
                p.set(
                    (base + l * along + k * step) as usize,
                    v[(7 + k) as usize][l as usize] as u16,
                );
            }
        }
    }
}

/// The loop filter as the specification walks it (7.14.1): every unit of
/// every plane through [`edge_loop_filter`], vertical edges first. What
/// [`loop_filter_threads`] is held to.
#[cfg(test)]
fn loop_filter_reference(f: &mut FrameCtx) {
    for plane in 0..f.num_planes {
        if plane == 0 || f.hdr.loop_filter_level[1 + plane] != 0 {
            let mut pb = std::mem::take(&mut f.cur.planes[plane]);
            let sp = SharedPlane::new(&mut pb);
            let row_step = if plane == 0 { 1 } else { 1 << f.ssy };
            let col_step = if plane == 0 { 1 } else { 1 << f.ssx };
            for pass in 0..2 {
                for row in (0..f.mi_rows).step_by(row_step) {
                    for col in (0..f.mi_cols).step_by(col_step) {
                        edge_loop_filter(f, sp, plane, pass, row, col);
                    }
                }
            }
            f.cur.planes[plane] = pb;
        }
    }
}

/// The edge loop filter process (7.14.2), as written: the reference for
/// [`edge_unit`] and the kernels.
#[cfg(test)]
fn edge_loop_filter(
    f: &FrameCtx,
    p: SharedPlane,
    plane: usize,
    pass: usize,
    row: usize,
    col: usize,
) {
    let (sub_x, sub_y) = f.plane_ss(plane);
    let (dx, dy) = if pass == 0 { (1i32, 0i32) } else { (0, 1) };
    let x = col * MI_SIZE;
    let y = row * MI_SIZE;
    let row = row | sub_y;
    let col = col | sub_x;
    let on_screen = !(x >= f.hdr.frame_width as usize
        || y >= f.hdr.frame_height as usize
        || (pass == 0 && x == 0)
        || (pass == 1 && y == 0));
    if !on_screen {
        return;
    }
    let xp = x >> sub_x;
    let yp = y >> sub_y;
    let prev_row = row - ((dy as usize) << sub_y);
    let prev_col = col - ((dx as usize) << sub_x);
    let cols = f.ms;
    let m = f.mi[row * cols + col];
    let mi_size = m.mi_size as usize;
    let tx_sz = f.lf_tx_sizes[plane][(row >> sub_y) * cols + (col >> sub_x)] as usize;
    let plane_size = f.plane_residual_size(mi_size, plane);
    let skip = m.skip;
    let is_intra = m.ref_frame[0] as i32 <= INTRA_FRAME;
    let prev_tx_sz =
        f.lf_tx_sizes[plane][(prev_row >> sub_y) * cols + (prev_col >> sub_x)] as usize;
    let is_block_edge = if pass == 0 {
        xp.is_multiple_of(block_width(plane_size))
    } else {
        yp.is_multiple_of(block_height(plane_size))
    };
    let is_tx_edge = if pass == 0 {
        xp.is_multiple_of(TX_WIDTH[tx_sz])
    } else {
        yp.is_multiple_of(TX_HEIGHT[tx_sz])
    };
    let apply_filter = is_tx_edge && (is_block_edge || !skip || is_intra);
    if !apply_filter {
        return;
    }
    // The filter size process (7.14.3).
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz].min(TX_WIDTH[tx_sz])
    } else {
        TX_HEIGHT[prev_tx_sz].min(TX_HEIGHT[tx_sz])
    };
    let filter_size = if plane == 0 {
        16.min(base_size)
    } else {
        8.min(base_size)
    };
    let (mut lvl, mut limit, mut blimit, mut thresh) = filter_strength(f, row, col, plane, pass);
    if lvl == 0 {
        (lvl, limit, blimit, thresh) = filter_strength(f, prev_row, prev_col, plane, pass);
    }
    if lvl == 0 {
        return;
    }
    // The four samples along the edge, filtered together (7.14.6 for each).
    let stride = p.stride as isize;
    let step = dy as isize * stride + dx as isize;
    let along = dx as isize * stride + dy as isize;
    let base = yp as isize * stride + xp as isize;
    let reach: isize = if filter_size >= 16 { 7 } else { 4 };
    let mut v: crate::dsp::lf::Edge = [[0; 4]; 14];
    for l in 0..4isize {
        for k in -reach..reach {
            v[(7 + k) as usize][l as usize] = p.get((base + l * along + k * step) as usize) as i32;
        }
    }
    let params = crate::dsp::lf::Params {
        limit,
        blimit,
        thresh,
        filter_size,
        chroma: plane != 0,
        bit_depth: f.bit_depth,
    };
    crate::dsp::lf::filter(&mut v, &params);
    let span: isize = if filter_size >= 16 { 6 } else { 3 };
    for l in 0..4isize {
        for k in -span..span {
            p.set(
                (base + l * along + k * step) as usize,
                v[(7 + k) as usize][l as usize] as u16,
            );
        }
    }
}

/// The adaptive filter strength process (7.14.4): `lvl, limit, blimit,
/// thresh`. As written: the reference for [`LfLevels`].
#[cfg(test)]
fn filter_strength(
    f: &FrameCtx,
    row: usize,
    col: usize,
    plane: usize,
    pass: usize,
) -> (i32, i32, i32, i32) {
    let h = &f.hdr;
    let m = &f.mi[row * f.ms + col];
    let segment = m.segment_id as usize;
    let rf = m.ref_frame[0] as i32;
    let mode = m.y_mode as usize;
    let mode_type = (mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV) as usize;
    let delta_lf = if h.delta_lf_multi {
        m.delta_lf[if plane == 0 { pass } else { plane + 1 }] as i32
    } else {
        m.delta_lf[0] as i32
    };
    // The adaptive filter strength selection process (7.14.5).
    let i = if plane == 0 { pass } else { plane + 1 };
    let base_filter_level = clip3(0, MAX_LOOP_FILTER, delta_lf + h.loop_filter_level[i]);
    let mut lvl_seg = base_filter_level;
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if h.segmentation_enabled && h.feature_enabled[segment][feature] {
        lvl_seg = clip3(
            0,
            MAX_LOOP_FILTER,
            h.feature_data[segment][feature] + lvl_seg,
        );
    }
    if h.loop_filter_delta_enabled {
        let n_shift = lvl_seg >> 5;
        if rf == INTRA_FRAME {
            lvl_seg += h.loop_filter_ref_deltas[INTRA_FRAME as usize] << n_shift;
        } else if rf > INTRA_FRAME {
            lvl_seg += (h.loop_filter_ref_deltas[rf as usize] << n_shift)
                + (h.loop_filter_mode_deltas[mode_type] << n_shift);
        }
        lvl_seg = clip3(0, MAX_LOOP_FILTER, lvl_seg);
    }
    let lvl = lvl_seg;
    let sharp = h.loop_filter_sharpness;
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        clip3(1, 9 - sharp, lvl >> shift)
    } else {
        (lvl >> shift).max(1)
    };
    let blimit = 2 * (lvl + 2) + limit;
    let thresh = lvl >> 4;
    (lvl, limit, blimit, thresh)
}

// ---------------------------------------------------------------------------
// CDEF (7.15)
// ---------------------------------------------------------------------------

/// The CDEF process (7.15): `CdefFrame` from `CurrFrame`, on up to
/// `threads` threads (bands of 64 luma rows: each 8x8 block writes only its
/// own samples).
pub(crate) fn cdef(f: &FrameCtx, threads: usize) -> FrameBuf {
    let mut out = f.cur.clone();
    let h = &f.hdr;
    if h.coded_lossless || h.allow_intrabc || !f.seq.enable_cdef {
        return out;
    }
    let planes: Vec<SharedPlane> = out.planes.iter_mut().map(SharedPlane::new).collect();
    let band = |b: usize| {
        let mut r = b * 16;
        while r < ((b + 1) * 16).min(f.mi_rows) {
            let mut c = 0;
            while c < f.mi_cols {
                let idx = f.cdef_idx[(r >> 4) * f.cdef_stride + (c >> 4)];
                if idx >= 0 {
                    cdef_block(f, &planes, r, c, idx as usize);
                }
                c += 2;
            }
            r += 2;
        }
    };
    let bands = f.mi_rows.div_ceil(16);
    if threads > 1 {
        crate::decoder::parallel_map(bands, threads, band);
    } else {
        (0..bands).for_each(band);
    }
    out
}

/// The CDEF block process (7.15.1) for a block whose parameters are set.
fn cdef_block(f: &FrameCtx, out: &[SharedPlane], r: usize, c: usize, idx: usize) {
    let cols = f.ms;
    let sk = |rr: usize, cc: usize| f.mi[rr * cols + cc].skip;
    let skip = sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1);
    if skip {
        return;
    }
    let coeff_shift = f.bit_depth - 8;
    let h = &f.hdr;
    let uv_on =
        f.num_planes > 1 && (h.cdef_uv_pri_strength[idx] != 0 || h.cdef_uv_sec_strength[idx] != 0);
    if h.cdef_y_pri_strength[idx] == 0 && h.cdef_y_sec_strength[idx] == 0 && !uv_on {
        // Every strength zero: the filter leaves the block as it is.
        return;
    }
    // The direction matters to the primary taps alone (a primary strength
    // of zero filters along direction 0, and the variance only scales the
    // luma primary strength).
    let (y_dir, var) = if h.cdef_y_pri_strength[idx] != 0
        || (f.num_planes > 1 && h.cdef_uv_pri_strength[idx] != 0)
    {
        cdef_direction(f, r, c)
    } else {
        (0, 0)
    };
    let mut pri_str = h.cdef_y_pri_strength[idx] << coeff_shift;
    let sec_str = h.cdef_y_sec_strength[idx] << coeff_shift;
    let dir = if pri_str == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) != 0 {
        (floor_log2((var >> 6) as u32) as i32).min(12)
    } else {
        0
    };
    pri_str = if var != 0 {
        (pri_str * (4 + var_str) + 8) >> 4
    } else {
        0
    };
    let damping = h.cdef_damping + coeff_shift as i32;
    if pri_str != 0 || sec_str != 0 {
        cdef_filter(f, out, 0, r, c, pri_str, sec_str, damping, dir);
    }
    if !uv_on {
        return;
    }
    let pri_str = h.cdef_uv_pri_strength[idx] << coeff_shift;
    let sec_str = h.cdef_uv_sec_strength[idx] << coeff_shift;
    let dir = if pri_str == 0 {
        0
    } else {
        CDEF_UV_DIR[f.ssx][f.ssy][y_dir]
    };
    let damping = h.cdef_damping + coeff_shift as i32 - 1;
    cdef_filter(f, out, 1, r, c, pri_str, sec_str, damping, dir);
    cdef_filter(f, out, 2, r, c, pri_str, sec_str, damping, dir);
}

/// The CDEF direction process (7.15.2): `(yDir, var)`.
pub(crate) fn cdef_direction(f: &FrameCtx, r: usize, c: usize) -> (usize, i32) {
    let mut cost = [0i32; 8];
    let mut partial = [[0i32; 15]; 8];
    let x0 = c << MI_SIZE_LOG2;
    let y0 = r << MI_SIZE_LOG2;
    let p = &f.cur.planes[0];
    let sh = f.bit_depth - 8;
    for i in 0..8 {
        let row = &p.data[(y0 + i) * p.stride + x0..][..8];
        for j in 0..8 {
            let x = (row[j] as i32 >> sh) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
        i += 2;
    }
    let mut best_cost = 0;
    let mut y_dir = 0;
    for (i, &cst) in cost.iter().enumerate() {
        if cst > best_cost {
            best_cost = cst;
            y_dir = i;
        }
    }
    let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
    (y_dir, var)
}

/// The CDEF filter process (7.15.3).
#[allow(clippy::too_many_arguments)]
fn cdef_filter(
    f: &FrameCtx,
    out: &[SharedPlane],
    plane: usize,
    r: usize,
    c: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
) {
    let (sub_x, sub_y) = f.plane_ss(plane);
    let x0 = (c * MI_SIZE) >> sub_x;
    let y0 = (r * MI_SIZE) >> sub_y;
    let w = 8 >> sub_x;
    let h = 8 >> sub_y;
    let xlim = (f.mi_cols * MI_SIZE) >> sub_x;
    let ylim = (f.mi_rows * MI_SIZE) >> sub_y;
    let mut res = [0u16; 64];
    cdef_filter_block(
        &f.cur.planes[plane],
        x0,
        y0,
        w,
        h,
        xlim,
        ylim,
        pri_str,
        sec_str,
        damping,
        dir,
        f.bit_depth - 8,
        &mut res,
    );
    for i in 0..h {
        out[plane].write_row(x0, y0 + i, &res[i * w..(i + 1) * w]);
    }
}

/// The CDEF filter process (7.15.3) of one `w` x `h` block of a plane at
/// `(x0, y0)`, reading `src` (samples at or beyond `xlim` / `ylim`, the
/// frame's mode info edge in this plane, are unavailable) and writing the
/// filtered block to `res` (row stride `w`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block(
    src: &PlaneBuf,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
    coeff_shift: u32,
    res: &mut [u16; 64],
) {
    use crate::dsp::cdef::{NA, Taps, WS};
    const B: usize = 2;
    let adj = |s: i32| {
        if s != 0 {
            (damping - floor_log2(s as u32) as i32).max(0)
        } else {
            0
        }
    };
    let d_lo = (dir + 6) & 7;
    let d_hi = (dir + 2) & 7;
    // The taps, their offsets in samples of a buffer `stride` wide: primary
    // (k, +/-), secondary (k, +/-, -2/+2).
    let taps = |stride: usize| {
        let off = |d: usize, k: usize| -> isize {
            CDEF_DIRECTIONS[d][k][0] as isize * stride as isize + CDEF_DIRECTIONS[d][k][1] as isize
        };
        Taps {
            pri_str,
            sec_str,
            pri_adj: adj(pri_str),
            sec_adj: adj(sec_str),
            pri_taps: CDEF_PRI_TAPS[((pri_str >> coeff_shift) & 1) as usize],
            sec_taps: CDEF_SEC_TAPS[((pri_str >> coeff_shift) & 1) as usize],
            pri_off: [off(dir, 0), off(dir, 1)],
            sec_off: [[off(d_lo, 0), off(d_hi, 0)], [off(d_lo, 1), off(d_hi, 1)]],
        }
    };
    if crate::dsp::cdef::plane_filter_pays()
        && x0 >= B
        && y0 >= B
        && x0 + w + B <= xlim.min(src.w)
        && y0 + h + B <= ylim.min(src.h)
    {
        // Every sample the filter reads is available: filter the plane
        // where it lies.
        let pos = y0 * src.stride + x0;
        crate::dsp::cdef::filter_plane(&src.data, src.stride, pos, w, h, &taps(src.stride), res);
        return;
    }
    // The block with a border of 2, unavailable samples marked.
    let mut win = [NA; WS * WS];
    let inside_x = x0 >= B && x0 + w + B <= xlim;
    for i in 0..h + 2 * B {
        let y = y0 as isize + i as isize - B as isize;
        if y < 0 || y as usize >= ylim {
            continue;
        }
        let row = &src.data[y as usize * src.stride..];
        let out = &mut win[i * WS..i * WS + w + 2 * B];
        if inside_x {
            for (o, &v) in out.iter_mut().zip(&row[x0 - B..x0 + w + B]) {
                *o = v as i32;
            }
        } else {
            for (j, o) in out.iter_mut().enumerate() {
                let x = x0 as isize + j as isize - B as isize;
                if x >= 0 && (x as usize) < xlim {
                    *o = row[x as usize] as i32;
                }
            }
        }
    }
    crate::dsp::cdef::filter(&win, w, h, &taps(WS), res);
}

// ---------------------------------------------------------------------------
// Super-resolution (7.16)
// ---------------------------------------------------------------------------

/// The upscaling process (7.16). Returns `None` without super-resolution
/// (the input is the output).
pub(crate) fn upscale(f: &FrameCtx, input: &FrameBuf) -> Option<FrameBuf> {
    if !f.hdr.use_superres {
        return None;
    }
    let maxv = (1i32 << f.bit_depth) - 1;
    let mut out = FrameBuf::default();
    for plane in 0..f.num_planes {
        let (sub_x, sub_y) = f.plane_ss(plane);
        let downscaled_w = round2(f.hdr.frame_width as i32, sub_x as u32) as i64;
        let upscaled_w = round2(f.hdr.upscaled_width as i32, sub_x as u32) as i64;
        let plane_h = round2(f.hdr.frame_height as i32, sub_y as u32) as usize;
        let step_x = ((downscaled_w << SUPERRES_SCALE_BITS) + upscaled_w / 2) / upscaled_w;
        let err = upscaled_w * step_x - (downscaled_w << SUPERRES_SCALE_BITS);
        let mut initial_subpel_x = (-((upscaled_w - downscaled_w) << (SUPERRES_SCALE_BITS - 1))
            + upscaled_w / 2)
            / upscaled_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2;
        initial_subpel_x &= SUPERRES_SCALE_MASK;
        let mi_w = (f.mi_cols >> sub_x) as i64;
        let max_x = mi_w * MI_SIZE as i64 - 1;
        let src = &input.planes[plane];
        let mut dst = PlaneBuf::new(upscaled_w as usize + 32, src.h, 0);
        for y in 0..plane_h {
            let row = src.row(y);
            for x in 0..upscaled_w {
                let src_x = -(1i64 << SUPERRES_SCALE_BITS) + initial_subpel_x + x * step_x;
                let src_x_px = src_x >> SUPERRES_SCALE_BITS;
                let src_x_subpel = ((src_x & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS) as usize;
                let mut sum = 0i32;
                for k in 0..SUPERRES_FILTER_TAPS {
                    let sx =
                        (src_x_px + (k - SUPERRES_FILTER_OFFSET) as i64).clamp(0, max_x) as usize;
                    sum += row[sx] as i32 * UPSCALE_FILTER[src_x_subpel][k as usize];
                }
                dst.set(
                    x as usize,
                    y,
                    round2(sum, FILTER_BITS as u32).clamp(0, maxv) as u16,
                );
            }
        }
        out.planes.push(dst);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Loop restoration (7.17)
// ---------------------------------------------------------------------------

pub(crate) struct LrCtx<'a> {
    pub(crate) cur: &'a PlaneBuf,
    pub(crate) cdef: &'a PlaneBuf,
    pub(crate) stripe_start_y: i32,
    pub(crate) stripe_end_y: i32,
    pub(crate) plane_end_x: i32,
    pub(crate) plane_end_y: i32,
}

impl LrCtx<'_> {
    /// The get source sample process (7.17.6).
    #[inline]
    fn sample(&self, x: i32, y: i32) -> i32 {
        let x = x.min(self.plane_end_x).max(0) as usize;
        let y = y.min(self.plane_end_y).max(0);
        if y < self.stripe_start_y {
            let y = (self.stripe_start_y - 2).max(y) as usize;
            self.cur.get(x, y) as i32
        } else if y > self.stripe_end_y {
            let y = (self.stripe_end_y + 2).min(y) as usize;
            self.cur.get(x, y) as i32
        } else {
            self.cdef.get(x, y as usize) as i32
        }
    }
}

/// The loop restoration process (7.17): `LrFrame` from `UpscaledCurrFrame`
/// and `UpscaledCdefFrame` (taken: it is the starting point of `LrFrame`),
/// on up to `threads` threads.
///
/// The specification filters 4x4 blocks, each with the parameters of its
/// stripe (64 luma rows, offset by 8) and unit; here each stripe of each
/// unit is filtered as one rectangle, from a window of the samples the
/// process reads (`LrCtx::sample`): the same arithmetic, sample for sample.
/// Stripes write only their own rows, so they run in parallel.
pub(crate) fn loop_restoration(
    f: &FrameCtx,
    cur: &FrameBuf,
    cdef: FrameBuf,
    threads: usize,
) -> FrameBuf {
    if !f.hdr.uses_lr {
        return cdef;
    }
    let mut lr = cdef.clone();
    let planes: Vec<SharedPlane> = lr.planes.iter_mut().map(SharedPlane::new).collect();
    let stripes = (f.hdr.frame_height as usize + 8).div_ceil(64);
    let mut jobs = Vec::new();
    for plane in 0..f.num_planes {
        if f.hdr.frame_restoration_type[plane] != RESTORE_NONE {
            jobs.extend((0..stripes).map(|s| (plane, s)));
        }
    }
    let job = |i: usize| {
        let (plane, s) = jobs[i];
        restore_stripe(f, cur, &cdef, planes[plane], plane, s);
    };
    if threads > 1 {
        crate::decoder::parallel_map(jobs.len(), threads, job);
    } else {
        (0..jobs.len()).for_each(job);
    }
    lr
}

/// The geometry of a stripe of a plane: `(stripe_start_y, stripe_end_y,
/// first row, last row, unit row)`, or `None` past the plane.
pub(crate) fn lr_stripe(
    f: &FrameCtx,
    plane: usize,
    s: usize,
) -> Option<(i32, i32, i32, i32, usize)> {
    let (sub_x, sub_y) = f.plane_ss(plane);
    let _ = sub_x;
    let plane_end_y = round2(f.hdr.frame_height as i32, sub_y as u32) - 1;
    let stripe_start_y = (-8 + s as i32 * 64) >> sub_y;
    let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;
    let y0 = stripe_start_y.max(0);
    let y1 = stripe_end_y.min(plane_end_y);
    if y0 > y1 {
        return None;
    }
    let unit_size = f.hdr.loop_restoration_size[plane];
    let unit_rows = f.lr[plane].unit_rows;
    // The unit row of the stripe's first 4x4 block row (7.17.1).
    let luma_row = (s * 64).saturating_sub(8);
    let unit_row = (unit_rows - 1).min(((luma_row + 8) >> sub_y) / unit_size);
    Some((stripe_start_y, stripe_end_y, y0, y1, unit_row))
}

/// The columns `x0..=x1` of unit column `uc` of a plane.
pub(crate) fn lr_unit_cols(f: &FrameCtx, plane: usize, uc: usize) -> (i32, i32) {
    let (sub_x, _) = f.plane_ss(plane);
    let plane_end_x = round2(f.hdr.upscaled_width as i32, sub_x as u32) - 1;
    let unit_size = f.hdr.loop_restoration_size[plane] as i32;
    let x0 = uc as i32 * unit_size;
    let x1 = if uc + 1 == f.lr[plane].unit_cols {
        plane_end_x
    } else {
        ((uc as i32 + 1) * unit_size - 1).min(plane_end_x)
    };
    (x0, x1)
}

fn restore_stripe(
    f: &FrameCtx,
    cur: &FrameBuf,
    cdef: &FrameBuf,
    out: SharedPlane,
    plane: usize,
    s: usize,
) {
    let Some((stripe_start_y, stripe_end_y, y0, y1, unit_row)) = lr_stripe(f, plane, s) else {
        return;
    };
    let (sub_x, sub_y) = f.plane_ss(plane);
    let ctx = LrCtx {
        cur: &cur.planes[plane],
        cdef: &cdef.planes[plane],
        stripe_start_y,
        stripe_end_y,
        plane_end_x: round2(f.hdr.upscaled_width as i32, sub_x as u32) - 1,
        plane_end_y: round2(f.hdr.frame_height as i32, sub_y as u32) - 1,
    };
    let lp = &f.lr[plane];
    let mut buf = LrBuffers::default();
    for uc in 0..lp.unit_cols {
        let (x0, x1) = lr_unit_cols(f, plane, uc);
        if x0 > ctx.plane_end_x {
            continue;
        }
        let uidx = unit_row * lp.unit_cols + uc;
        let params = match lp.lr_type[uidx] {
            RESTORE_WIENER => LrParams::Wiener(lp.wiener[uidx]),
            RESTORE_SGRPROJ => LrParams::Sgr(lp.sgr_set[uidx] as usize, lp.sgr_xqd[uidx]),
            _ => continue,
        };
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        restore_rect(f, &ctx, x0, y0, w, h, &params, &mut buf);
        for i in 0..h {
            out.write_row(x0 as usize, y0 as usize + i, &buf.out[i * w..(i + 1) * w]);
        }
    }
}

/// One unit's restoration parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LrParams {
    /// `LrWiener[ unit ]`: the vertical then horizontal coefficients.
    Wiener([[i32; 3]; 2]),
    /// `LrSgrSet` and `LrSgrXqd`.
    Sgr(usize, [i32; 2]),
}

/// Scratch for [`restore_rect`]; `out` holds the result.
#[derive(Default)]
pub(crate) struct LrBuffers {
    pub(crate) win: Vec<i32>,
    inter: Vec<i32>,
    pub(crate) flt: [Vec<i32>; 2],
    a: Vec<i32>,
    b: Vec<i32>,
    sq: Vec<i64>,
    sum: Vec<i64>,
    /// `sq` and `sum` hold the integral images of `win` as it is.
    integral: bool,
    pub(crate) out: Vec<u16>,
}

/// The border of the sample window: the Wiener taps reach 3 samples, the
/// self-guided boxes 2 plus 1 for the neighbourhood of `A` and `B`.
pub(crate) const LRB: usize = 3;

/// Restores the `w` x `h` rectangle at `(x0, y0)` (inside one stripe and
/// one unit) with `params` into `buf.out` (row stride `w`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn restore_rect(
    f: &FrameCtx,
    ctx: &LrCtx,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    params: &LrParams,
    buf: &mut LrBuffers,
) {
    lr_window(ctx, x0, y0, w, h, buf);
    restore_rect_from_window(f, ctx, x0, y0, w, h, params, buf);
}

/// The samples a `w` x `h` rectangle at `(x0, y0)` reads, with a border of
/// `LRB`, into `buf.win` (row stride `w + 2 * LRB`).
pub(crate) fn lr_window(ctx: &LrCtx, x0: i32, y0: i32, w: usize, h: usize, buf: &mut LrBuffers) {
    let ww = w + 2 * LRB;
    let wh = h + 2 * LRB;
    buf.win.resize(ww * wh, 0);
    buf.integral = false;
    for r in 0..wh {
        let y = y0 + r as i32 - LRB as i32;
        let row = &mut buf.win[r * ww..(r + 1) * ww];
        for (c, v) in row.iter_mut().enumerate() {
            *v = ctx.sample(x0 + c as i32 - LRB as i32, y);
        }
    }
}

/// [`restore_rect`] once [`lr_window`] has filled the window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn restore_rect_from_window(
    f: &FrameCtx,
    ctx: &LrCtx,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    params: &LrParams,
    buf: &mut LrBuffers,
) {
    let ww = w + 2 * LRB;
    let maxv = (1i32 << f.bit_depth) - 1;
    buf.out.resize(w * h, 0);
    match *params {
        LrParams::Wiener(coef) => {
            let rv = f.rounding_variables(false);
            let vfilter = wiener_coefficient(&coef[0]);
            let hfilter = wiener_coefficient(&coef[1]);
            let bd = f.bit_depth as i32;
            let offset = 1i32 << (bd + FILTER_BITS - rv.round0 as i32 - 1);
            let limit = (1i32 << (bd + 1 + FILTER_BITS - rv.round0 as i32)) - 1;
            buf.inter.resize((h + 6) * w, 0);
            for r in 0..h + 6 {
                let src = &buf.win[r * ww..(r + 1) * ww];
                for c in 0..w {
                    let mut s = 0;
                    for t in 0..7 {
                        s += hfilter[t] * src[c + t];
                    }
                    buf.inter[r * w + c] = clip3(-offset, limit - offset, round2(s, rv.round0));
                }
            }
            for r in 0..h {
                for c in 0..w {
                    let mut s = 0;
                    for t in 0..7 {
                        s += vfilter[t] * buf.inter[(r + t) * w + c];
                    }
                    buf.out[r * w + c] = round2(s, rv.round1).clamp(0, maxv) as u16;
                }
            }
        }
        LrParams::Sgr(set, xqd) => {
            for pass in 0..2 {
                box_filter_rect(f, ctx, x0, y0, w, h, set, pass, buf);
            }
            let w0 = xqd[0];
            let w1 = xqd[1];
            let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
            let r0 = SGR_PARAMS[set][0];
            let r1 = SGR_PARAMS[set][2];
            for i in 0..h {
                for j in 0..w {
                    let u =
                        (ctx.cdef.get(x0 as usize + j, y0 as usize + i) as i32) << SGRPROJ_RST_BITS;
                    let mut v = w1 * u;
                    v += if r0 != 0 {
                        w0 * buf.flt[0][i * w + j]
                    } else {
                        w0 * u
                    };
                    v += if r1 != 0 {
                        w2 * buf.flt[1][i * w + j]
                    } else {
                        w2 * u
                    };
                    let s = round2(v, (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32);
                    buf.out[i * w + j] = s.clamp(0, maxv) as u16;
                }
            }
        }
    }
}

/// The box filter process (7.17.3) over a rectangle, from the window in
/// `buf.win`, into `buf.flt[pass]` (row stride `w`). Box sums come from
/// integral images of the window.
/// The integral images of the window (`sum`, `sq`; a first row and column
/// of zeros), shared by every box filter of the window.
fn integral_images(buf: &mut LrBuffers, ww: usize, wh: usize) {
    let iw = ww + 1;
    buf.sq.resize(iw * (wh + 1), 0);
    buf.sum.resize(iw * (wh + 1), 0);
    buf.sq[..iw].fill(0);
    buf.sum[..iw].fill(0);
    for y in 0..wh {
        let mut rs = 0i64;
        let mut rq = 0i64;
        buf.sq[(y + 1) * iw] = 0;
        buf.sum[(y + 1) * iw] = 0;
        for x in 0..ww {
            let c = buf.win[y * ww + x] as i64;
            rs += c;
            rq += c * c;
            buf.sum[(y + 1) * iw + x + 1] = buf.sum[y * iw + x + 1] + rs;
            buf.sq[(y + 1) * iw + x + 1] = buf.sq[y * iw + x + 1] + rq;
        }
    }
    buf.integral = true;
}

/// The box filter's `A` for each clamped `z` (7.17.3): 256 at 255 and
/// above, 1 at 0, else `((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)`.
const SGR_A2: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut z = 0;
    while z < 256 {
        t[z] = if z >= 255 {
            256
        } else if z == 0 {
            1
        } else {
            (((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)) as i32
        };
        z += 1;
    }
    t
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn box_filter_rect(
    f: &FrameCtx,
    ctx: &LrCtx,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    set: usize,
    pass: usize,
    buf: &mut LrBuffers,
) {
    let r = SGR_PARAMS[set][pass * 2];
    buf.flt[pass].resize(w * h, 0);
    if r == 0 {
        return;
    }
    let ww = w + 2 * LRB;
    let wh = h + 2 * LRB;
    // Integral images (one row and column of zeros first).
    let iw = ww + 1;
    if !buf.integral {
        integral_images(buf, ww, wh);
    }
    let eps = SGR_PARAMS[set][pass * 2 + 1];
    let bd = f.bit_depth;
    let n = (2 * r + 1) * (2 * r + 1);
    let n2e = n * n * eps;
    let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let one_over_n = ((1 << SGRPROJ_RECIP_BITS) + n / 2) / n;
    // A and B at rectangle positions -1..h+1 by -1..w+1.
    let aw = w + 2;
    buf.a.resize(aw * (h + 2), 0);
    buf.b.resize(aw * (h + 2), 0);
    let ru = r as usize;
    for i in 0..h + 2 {
        // The first filter weighs only the odd rows' A and B.
        if pass == 0 && (y0 + i as i32 - 1) & 1 == 0 {
            continue;
        }
        // Window row of position i - 1, box rows from it - r to it + r.
        let wy = i + LRB - 1;
        let (ya, yb) = (wy - ru, wy + ru + 1);
        for j in 0..w + 2 {
            let wx = j + LRB - 1;
            let (xa, xb) = (wx - ru, wx + ru + 1);
            let boxed =
                |t: &[i64]| t[yb * iw + xb] - t[ya * iw + xb] - t[yb * iw + xa] + t[ya * iw + xa];
            let a = boxed(&buf.sq);
            let b = boxed(&buf.sum) as i32;
            let a = round2_64(a, 2 * (bd - 8));
            let d = round2(b, bd - 8) as i64;
            let p = (a * n as i64 - d * d).max(0);
            let z = round2_64(p * s as i64, SGRPROJ_MTABLE_BITS as u32);
            let a2 = SGR_A2[z.min(255) as usize];
            let b2 = ((1 << SGRPROJ_SGR_BITS) - a2) as i64 * b as i64 * one_over_n as i64;
            buf.a[i * aw + j] = a2;
            buf.b[i * aw + j] = round2_64(b2, SGRPROJ_RECIP_BITS as u32) as i32;
        }
    }
    for i in 0..h {
        // Row parity of the frame (rectangles start on even rows, as the
        // 4x4 blocks of the process do).
        let yabs = y0 + i as i32;
        let shift = if pass == 0 && (yabs & 1) != 0 { 4 } else { 5 };
        for j in 0..w {
            let mut a = 0i32;
            let mut b = 0i32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let weight = if pass == 0 {
                        if ((yabs + dy) & 1) != 0 {
                            if dx == 0 { 6 } else { 5 }
                        } else {
                            0
                        }
                    } else if dx == 0 || dy == 0 {
                        4
                    } else {
                        3
                    };
                    let k = (i as i32 + 1 + dy) as usize * aw + (j as i32 + 1 + dx) as usize;
                    a += weight * buf.a[k];
                    b += weight * buf.b[k];
                }
            }
            let v = a * ctx.cdef.get(x0 as usize + j, y0 as usize + i) as i32 + b;
            buf.flt[pass][i * w + j] =
                round2(v, (SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS) as u32);
        }
    }
}

/// The Wiener coefficient process (7.17.5).
fn wiener_coefficient(coeff: &[i32; 3]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for i in 0..3 {
        let c = coeff[i];
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// xorshift: `rnd(n)` is below `n`.
    struct Rng(u64);

    impl Rng {
        fn rnd(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }

        fn signed(&mut self, n: i64) -> i32 {
            (self.rnd(2 * n as u64 + 1) as i64 - n) as i32
        }
    }

    /// A frame for the loop filter alone: `w` x `h` luma samples of smooth
    /// areas, steps and noise, tiled with random blocks and transform sizes
    /// (and, with `scramble`, units whose mode info and transform size
    /// belong to no tiling), under a random loop filter header.
    fn random_frame(
        r: &mut Rng,
        (w, h): (usize, usize),
        (ssx, ssy): (usize, usize),
        num_planes: usize,
        bit_depth: u32,
        scramble: bool,
    ) -> FrameCtx {
        // A reduced still picture header, 1x1: the loop filter reads
        // nothing of the sequence header.
        let seq = crate::obu::SequenceHeader::parse(&[0x18, 0, 0, 0, 0, 0], 0).expect("header");
        let mi_cols = 2 * ((w + 7) >> 3);
        let mi_rows = 2 * ((h + 7) >> 3);
        let ms = mi_cols + 32;
        let n = (mi_rows + 32) * ms;
        let mut hdr = FrameHeader {
            frame_width: w as u32,
            frame_height: h as u32,
            upscaled_width: w as u32,
            mi_cols,
            mi_rows,
            ..Default::default()
        };
        let top = [0, 1, 8, 20, 40, 63][r.rnd(6) as usize];
        hdr.loop_filter_level = std::array::from_fn(|_| r.rnd(top + 1) as i32);
        if r.rnd(8) == 0 {
            // Levels that only the deltas raise.
            hdr.loop_filter_level[r.rnd(2) as usize] = 0;
        }
        hdr.loop_filter_sharpness = r.rnd(8) as i32;
        hdr.loop_filter_delta_enabled = r.rnd(2) == 0;
        hdr.loop_filter_ref_deltas =
            std::array::from_fn(|_| r.signed(if top > 8 { 63 } else { 3 }));
        hdr.loop_filter_mode_deltas = std::array::from_fn(|_| r.signed(8));
        hdr.segmentation_enabled = r.rnd(2) == 0;
        for s in 0..8 {
            for feature in 0..8 {
                hdr.feature_enabled[s][feature] = r.rnd(3) == 0;
                hdr.feature_data[s][feature] = r.signed(63);
            }
        }
        hdr.delta_lf_present = r.rnd(3) == 0;
        hdr.delta_lf_multi = hdr.delta_lf_present && r.rnd(2) == 0;
        let mut mi = vec![Mi::default(); n];
        let mut lf_tx_sizes = [vec![0u8; n], vec![0u8; n], vec![0u8; n]];
        // Square blocks of 4 to 64 samples, or their two halves, with a
        // transform size that fits.
        let fitting_tx = |r: &mut Rng, bw: usize, bh: usize| loop {
            let t = r.rnd(19) as usize;
            if TX_WIDTH[t] <= bw && TX_HEIGHT[t] <= bh {
                break t as u8;
            }
        };
        let max_log2 = 1 + r.rnd(4) as usize;
        let mut block = |r: &mut Rng, row: usize, col: usize, mi_size: usize| {
            let (bw4, bh4) = (NUM_4X4_BLOCKS_WIDE[mi_size], NUM_4X4_BLOCKS_HIGH[mi_size]);
            let intra = r.rnd(4) == 0;
            let m = Mi {
                mi_size: mi_size as u8,
                y_mode: if intra {
                    r.rnd(13) as u8
                } else {
                    (NEARESTMV as u64 + r.rnd(12)) as u8
                },
                ref_frame: [if intra { 0 } else { 1 + r.rnd(7) as i8 }, -1],
                skip: r.rnd(2) == 0,
                segment_id: r.rnd(8) as u8,
                delta_lf: if hdr.delta_lf_present && r.rnd(2) == 0 {
                    std::array::from_fn(|_| r.signed(40) as i8)
                } else {
                    [0; 4]
                },
                ..Default::default()
            };
            let tx = [
                fitting_tx(r, 4 * bw4, 4 * bh4),
                fitting_tx(r, ((4 * bw4) >> ssx).max(4), ((4 * bh4) >> ssy).max(4)),
            ];
            for i in 0..bh4 {
                for j in 0..bw4 {
                    let (rr, cc) = (row + i, col + j);
                    mi[rr * ms + cc] = m;
                    lf_tx_sizes[0][rr * ms + cc] = tx[0];
                    for t in &mut lf_tx_sizes[1..] {
                        t[(rr >> ssy) * ms + (cc >> ssx)] = tx[1];
                    }
                }
            }
        };
        let sb = 16;
        for sb_row in (0..mi_rows).step_by(sb) {
            for sb_col in (0..mi_cols).step_by(sb) {
                // Blocks of `1 << log2` units a side, each whole or halved.
                let log2 = r.rnd(max_log2 as u64 + 1) as usize;
                let side = 1usize << log2;
                // BLOCK_4X4, 8X8, 16X16, 32X32, 64X64 are every third size.
                let square = 3 * log2;
                for row in (sb_row..sb_row + sb).step_by(side) {
                    for col in (sb_col..sb_col + sb).step_by(side) {
                        match if ssx == ssy { r.rnd(3) } else { 0 } {
                            // The 2:1 sizes follow their square: tall, wide.
                            1 if log2 > 0 => {
                                block(r, row, col, square - 1);
                                block(r, row + side / 2, col, square - 1);
                            }
                            2 if log2 > 0 => {
                                block(r, row, col, square - 2);
                                block(r, row, col + side / 2, square - 2);
                            }
                            _ => block(r, row, col, square),
                        }
                    }
                }
            }
        }
        if scramble {
            for _ in 0..mi_rows * mi_cols / 6 {
                let i = r.rnd(mi_rows as u64) as usize * ms + r.rnd(mi_cols as u64) as usize;
                match r.rnd(3) {
                    0 => lf_tx_sizes[r.rnd(3) as usize][i] = r.rnd(19) as u8,
                    1 => mi[i].skip = !mi[i].skip,
                    _ => mi[i].segment_id = r.rnd(8) as u8,
                }
            }
        }
        let aw = (mi_cols * MI_SIZE + 127) & !127;
        let ah = (mi_rows * MI_SIZE + 127) & !127;
        let maxv = (1u64 << bit_depth) - 1;
        let mut planes = Vec::new();
        for plane in 0..num_planes {
            let (sx, sy) = if plane == 0 { (0, 0) } else { (ssx, ssy) };
            let mut p = PlaneBuf::new((aw >> sx) + 32, (ah >> sy) + 32, 0);
            // Areas of 16x16 samples around one value each, the noise of a
            // size that sometimes passes every threshold and sometimes none.
            let area = 16;
            let across = p.w.div_ceil(area);
            let bases: Vec<(u64, u64)> = (0..across * p.h.div_ceil(area))
                .map(|_| {
                    let noise = [1, 2, 3, 6, 24, 200][r.rnd(6) as usize] << (bit_depth - 8);
                    (r.rnd(maxv + 1), noise)
                })
                .collect();
            for y in 0..p.h {
                for x in 0..p.w {
                    let (base, noise) = bases[(y / area) * across + x / area];
                    let v = (base + r.rnd(noise)).saturating_sub(noise / 2).min(maxv);
                    p.set(x, y, v as u16);
                }
            }
            planes.push(p);
        }
        FrameCtx {
            seq: Arc::new(seq),
            hdr,
            bit_depth,
            ssx,
            ssy,
            num_planes,
            mi_rows,
            mi_cols,
            ms,
            mi,
            tx_types: Vec::new(),
            lf_tx_sizes,
            palette_colors: [Vec::new(), Vec::new()],
            segment_ids: Vec::new(),
            prev_segment_ids: Arc::new(Vec::new()),
            cdef_idx: Vec::new(),
            cdef_stride: 0,
            lr: Default::default(),
            motion_field: Arc::new(Vec::new()),
            cur: FrameBuf { planes },
            refs: Default::default(),
            cdfs: crate::cdf::CdfContext::new(0),
            saved_cdfs: None,
            mask: Box::new([0; 128 * 128]),
            tile_num: 0,
            tiles_ok: true,
        }
    }

    /// The tables hold what the adaptive filter strength process gives for
    /// every segment, reference frame, mode and `i`, under random headers.
    #[test]
    fn levels_match_the_strength_process() {
        let mut r = Rng(0x1f2e_3d4c_5b6a_7988);
        for case in 0..300 {
            let mut f = random_frame(&mut r, (16, 16), (1, 1), 3, 8, false);
            let levels = LfLevels::new(&f.hdr);
            for segment in 0..8 {
                for rf in 0..8i8 {
                    for y_mode in 0..26u8 {
                        for with_delta in [false, true] {
                            f.mi[0] = Mi {
                                segment_id: segment,
                                ref_frame: [rf, -1],
                                y_mode,
                                delta_lf: if with_delta {
                                    std::array::from_fn(|_| r.signed(63) as i8)
                                } else {
                                    [0; 4]
                                },
                                ..Default::default()
                            };
                            for plane in 0..3 {
                                for pass in 0..2 {
                                    let i = if plane == 0 { pass } else { plane + 1 };
                                    let (lvl, limit, blimit, thresh) =
                                        filter_strength(&f, 0, 0, plane, pass);
                                    let got = levels.level(&f.hdr, &f.mi[0], i);
                                    assert_eq!(got as i32, lvl, "case {case} {:?}", f.mi[0]);
                                    assert_eq!(
                                        levels.limits[got].map(i32::from),
                                        [limit, blimit, thresh],
                                        "case {case} lvl {lvl}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The whole loop filter against the specification's walk with the
    /// one-edge lane filter: random frames of every layout, depth and size
    /// (with partial blocks at the right and bottom), on one thread and on
    /// three.
    #[test]
    fn frames_match_the_reference_walk() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut filtered = 0usize;
        for case in 0..160 {
            let (ss, num_planes) = [((1, 1), 3), ((0, 0), 3), ((1, 0), 3), ((1, 1), 1)][case % 4];
            let bit_depth = [8, 10, 12][case % 3];
            let w = [64, 65, 100, 129, 200, 36][r.rnd(6) as usize];
            let h = [64, 72, 90, 130, 37][r.rnd(5) as usize];
            let scramble = case % 5 == 4;
            let mut f = random_frame(&mut r, (w, h), ss, num_planes, bit_depth, scramble);
            let before = f.cur.clone();
            let mut want = FrameCtx {
                cur: f.cur.clone(),
                ..f.shard()
            };
            want.mi.clone_from(&f.mi);
            want.lf_tx_sizes.clone_from(&f.lf_tx_sizes);
            loop_filter_reference(&mut want);
            // (A wasm test binary has no threads to start.)
            let threads = if case % 2 == 0 || cfg!(target_arch = "wasm32") {
                1
            } else {
                3
            };
            loop_filter_threads(&mut f, threads);
            for plane in 0..num_planes {
                let (got, want) = (&f.cur.planes[plane], &want.cur.planes[plane]);
                if let Some(i) = (0..got.data.len()).find(|&i| got.data[i] != want.data[i]) {
                    panic!(
                        "case {case} ({w}x{h}, {ss:?}, {bit_depth} bits) plane {plane}: \
                         sample ({}, {}) is {}, the reference has {}",
                        i % got.stride,
                        i / got.stride,
                        got.data[i],
                        want.data[i]
                    );
                }
                filtered += (0..got.data.len())
                    .filter(|&i| got.data[i] != before.planes[plane].data[i])
                    .count();
            }
        }
        // The frames are such that the filter has work to do.
        assert!(filtered > 200_000, "{filtered} samples filtered");
    }
}
