//! Residual decoding (5.11.34 to 5.11.39): transform blocks and trees,
//! coefficients and their contexts (8.3.2), transform types, and the
//! reconstruct process (7.12.3).
//!
//! # Two coefficient readers
//!
//! `coeffs()` (5.11.39) is implemented twice.
//!
//! `coeffs_shared` is **the reference**. It is part of the tile walker the
//! encoder runs too: every symbol goes through `Coder`, which reads it
//! when decoding and writes the value the encoder planted when encoding,
//! and every context is derived from `Quant` as the specification writes
//! it. The encoder always uses it, and it is not to be bent for speed.
//!
//! `coeffs_decode` is the same syntax for decoding alone, where most of a
//! decode's symbols are read: no encoder values to carry, the levels kept
//! in a padded byte map so that the contexts are sums without edge tests,
//! the level symbols read with their CDF size known. Decoding uses it.
//!
//! They must never differ, and a test holds them to that:
//! `CoefficientReader::Checked` runs both on every transform block from
//! the same state and panics unless the coefficients, the end-of-block
//! position, every CDF, the symbol decoder's state (bit position included)
//! and the contexts left for later blocks are identical. The vector, fuzz
//! and encoder round-trip tests all decode that way.

use crate::Result;
use crate::consts::*;
use crate::decoder::CoefficientReader;
use crate::decoder::tile::{TileDecoder, find_tx_size};
use crate::dsp::itx::inverse_transform_2d;
use crate::tables::*;

impl TileDecoder<'_, '_> {
    /// `residual()`.
    pub(crate) fn residual(&mut self) -> Result<()> {
        let sb_mask = if self.f.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let ms = self.b.mi_size;
        let width_chunks = (block_width(ms) >> 6).max(1);
        let height_chunks = (block_height(ms) >> 6).max(1);
        let mi_size_chunk = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64
        } else {
            ms
        };
        let _ = sb_mask;
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let mi_row_chunk = self.b.mi_row + (chunk_y << 4);
                let mi_col_chunk = self.b.mi_col + (chunk_x << 4);
                let planes = 1 + if self.b.has_chroma { 2 } else { 0 };
                for plane in 0..planes {
                    let tx_sz = if self.b.lossless {
                        TX_4X4
                    } else {
                        self.get_tx_size(plane, self.b.tx_size)
                    };
                    let step_x = TX_WIDTH[tx_sz] >> 2;
                    let step_y = TX_HEIGHT[tx_sz] >> 2;
                    let plane_sz = self.f.plane_residual_size(mi_size_chunk, plane);
                    let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz];
                    let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz];
                    let (sub_x, sub_y) = self.f.plane_ss(plane);
                    let base_x = (mi_col_chunk >> sub_x) * MI_SIZE;
                    let base_y = (mi_row_chunk >> sub_y) * MI_SIZE;
                    if self.b.is_inter && !self.b.lossless && plane == 0 {
                        self.transform_tree(base_x, base_y, num4x4_w * 4, num4x4_h * 4)?;
                    } else {
                        let base_x_block = (self.b.mi_col >> sub_x) * MI_SIZE;
                        let base_y_block = (self.b.mi_row >> sub_y) * MI_SIZE;
                        let mut y = 0;
                        while y < num4x4_h {
                            let mut x = 0;
                            while x < num4x4_w {
                                self.transform_block(
                                    plane,
                                    base_x_block,
                                    base_y_block,
                                    tx_sz,
                                    x + ((chunk_x << 4) >> sub_x),
                                    y + ((chunk_y << 4) >> sub_y),
                                )?;
                                x += step_x;
                            }
                            y += step_y;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `transform_block()`.
    fn transform_block(
        &mut self,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx_sz: usize,
        x: usize,
        y: usize,
    ) -> Result<()> {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let row = (start_y << sub_y) >> MI_SIZE_LOG2;
        let col = (start_x << sub_x) >> MI_SIZE_LOG2;
        let sb_mask = if self.f.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let sub_block_mi_row = row & sb_mask;
        let sub_block_mi_col = col & sb_mask;
        let step_x = TX_WIDTH[tx_sz] >> MI_SIZE_LOG2;
        let step_y = TX_HEIGHT[tx_sz] >> MI_SIZE_LOG2;
        let max_x = (self.f.mi_cols * MI_SIZE) >> sub_x;
        let max_y = (self.f.mi_rows * MI_SIZE) >> sub_y;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        if !self.b.is_inter {
            if (plane == 0 && self.b.palette_size_y > 0)
                || (plane != 0 && self.b.palette_size_uv > 0)
            {
                self.predict_palette(plane, start_x, start_y, x, y, tx_sz);
            } else {
                let is_cfl = plane > 0 && self.b.uv_mode == UV_CFL_PRED;
                let mode = if plane == 0 {
                    self.b.y_mode
                } else if is_cfl {
                    DC_PRED
                } else {
                    self.b.uv_mode
                };
                let log2w = TX_WIDTH_LOG2[tx_sz];
                let log2h = TX_HEIGHT_LOG2[tx_sz];
                let (avail_l, avail_u) = if plane == 0 {
                    (self.b.avail_l, self.b.avail_u)
                } else {
                    (self.b.avail_l_chroma, self.b.avail_u_chroma)
                };
                let have_above_right = self.block_decoded(
                    plane,
                    (sub_block_mi_row >> sub_y) as isize - 1,
                    ((sub_block_mi_col >> sub_x) + step_x) as isize,
                );
                let have_below_left = self.block_decoded(
                    plane,
                    ((sub_block_mi_row >> sub_y) + step_y) as isize,
                    (sub_block_mi_col >> sub_x) as isize - 1,
                );
                self.predict_intra(
                    plane,
                    start_x,
                    start_y,
                    avail_l || x > 0,
                    avail_u || y > 0,
                    have_above_right,
                    have_below_left,
                    mode,
                    log2w,
                    log2h,
                );
                if is_cfl {
                    self.predict_chroma_from_luma(plane, start_x, start_y, tx_sz);
                }
            }
            if plane == 0 {
                self.b.max_luma_w = start_x + step_x * 4;
                self.b.max_luma_h = start_y + step_y * 4;
            }
        }
        if !self.b.skip {
            let eob = self.coeffs(plane, start_x, start_y, tx_sz);
            if eob > 0 {
                self.reconstruct(plane, start_x, start_y, tx_sz);
            }
        }
        let cols = self.f.ms;
        for i in 0..step_y {
            for j in 0..step_x {
                let rr = (row >> sub_y) + i;
                let cc = (col >> sub_x) + j;
                if rr < self.f.mi_rows + 32 && cc < cols {
                    self.f.lf_tx_sizes[plane][rr * cols + cc] = tx_sz as u8;
                }
                let by = (sub_block_mi_row >> sub_y) + i;
                let bx = (sub_block_mi_col >> sub_x) + j;
                if by < 33 && bx < 33 {
                    self.block_decoded[plane][by + 1][bx + 1] = true;
                }
            }
        }
        Ok(())
    }

    /// `transform_tree()`.
    fn transform_tree(&mut self, start_x: usize, start_y: usize, w: usize, h: usize) -> Result<()> {
        let max_x = self.f.mi_cols * MI_SIZE;
        let max_y = self.f.mi_rows * MI_SIZE;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let row = start_y >> MI_SIZE_LOG2;
        let col = start_x >> MI_SIZE_LOG2;
        let luma_tx_sz = self.mi(row, col).inter_tx_size as usize;
        let luma_w = TX_WIDTH[luma_tx_sz];
        let luma_h = TX_HEIGHT[luma_tx_sz];
        if w <= luma_w && h <= luma_h {
            let tx_sz = find_tx_size(w, h);
            self.transform_block(0, start_x, start_y, tx_sz, 0, 0)?;
        } else if w > h {
            self.transform_tree(start_x, start_y, w / 2, h)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h)?;
        } else if w < h {
            self.transform_tree(start_x, start_y, w, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w, h / 2)?;
        } else {
            self.transform_tree(start_x, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y + h / 2, w / 2, h / 2)?;
        }
        Ok(())
    }

    /// `get_tx_size( plane, txSz )`.
    fn get_tx_size(&self, plane: usize, tx_sz: usize) -> usize {
        if plane == 0 {
            return tx_sz;
        }
        let uv_tx = MAX_TX_SIZE_RECT[self.f.plane_residual_size(self.b.mi_size, plane)];
        if TX_WIDTH[uv_tx] == 64 || TX_HEIGHT[uv_tx] == 64 {
            if TX_WIDTH[uv_tx] == 16 {
                return TX_16X32;
            }
            if TX_HEIGHT[uv_tx] == 16 {
                return TX_32X16;
            }
            return TX_32X32;
        }
        uv_tx
    }

    /// `get_tx_set( txSz )`.
    pub(crate) fn get_tx_set(&self, tx_sz: usize) -> usize {
        let sqr = TX_SIZE_SQR[tx_sz];
        let sqr_up = TX_SIZE_SQR_UP[tx_sz];
        if sqr_up > TX_32X32 {
            return TX_SET_DCTONLY;
        }
        if self.b.is_inter {
            if self.f.hdr.reduced_tx_set || sqr_up == TX_32X32 {
                TX_SET_INTER_3
            } else if sqr == TX_16X16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            }
        } else if sqr_up == TX_32X32 {
            TX_SET_DCTONLY
        } else if self.f.hdr.reduced_tx_set || sqr == TX_16X16 {
            TX_SET_INTRA_2
        } else {
            TX_SET_INTRA_1
        }
    }

    pub(crate) fn is_tx_type_in_set(&self, tx_set: usize, tx_type: usize) -> bool {
        if self.b.is_inter {
            TX_TYPE_IN_SET_INTER[tx_set][tx_type] != 0
        } else {
            TX_TYPE_IN_SET_INTRA[tx_set][tx_type] != 0
        }
    }

    /// `compute_tx_type( plane, txSz, blockX, blockY )`.
    pub(crate) fn compute_tx_type(
        &self,
        plane: usize,
        tx_sz: usize,
        block_x: usize,
        block_y: usize,
    ) -> usize {
        let sqr_up = TX_SIZE_SQR_UP[tx_sz];
        if self.b.lossless || sqr_up > TX_32X32 {
            return DCT_DCT;
        }
        let tx_set = self.get_tx_set(tx_sz);
        let cols = self.f.ms;
        if plane == 0 {
            return self.f.tx_types[block_y * cols + block_x] as usize;
        }
        if self.b.is_inter {
            let x4 = self.b.mi_col.max(block_x << self.f.ssx);
            let y4 = self.b.mi_row.max(block_y << self.f.ssy);
            let tx_type = self.f.tx_types[y4 * cols + x4] as usize;
            if !self.is_tx_type_in_set(tx_set, tx_type) {
                return DCT_DCT;
            }
            return tx_type;
        }
        let tx_type = MODE_TO_TXFM[self.b.uv_mode];
        if !self.is_tx_type_in_set(tx_set, tx_type) {
            return DCT_DCT;
        }
        tx_type
    }

    /// `transform_type( x4, y4, txSz )`.
    fn transform_type(&mut self, x4: usize, y4: usize, tx_sz: usize) {
        let set = self.get_tx_set(tx_sz);
        let qidx = if self.f.hdr.segmentation_enabled {
            crate::header::get_qindex_header(&self.f.hdr, self.b.segment_id)
        } else {
            self.f.hdr.base_q_idx
        };
        let tx_type = if set > 0 && qidx > 0 {
            let sqr = TX_SIZE_SQR[tx_sz];
            if self.b.is_inter {
                if set == TX_SET_INTER_1 {
                    let p = inv_index(&TX_TYPE_INTER_INV_SET1, self.enc_tx_type);
                    let s = self.sd.symbol(&mut self.cdf.inter_tx_type_set1[sqr], p);
                    TX_TYPE_INTER_INV_SET1[s]
                } else if set == TX_SET_INTER_2 {
                    let p = inv_index(&TX_TYPE_INTER_INV_SET2, self.enc_tx_type);
                    let s = self.sd.symbol(&mut self.cdf.inter_tx_type_set2, p);
                    TX_TYPE_INTER_INV_SET2[s]
                } else {
                    let p = inv_index(&TX_TYPE_INTER_INV_SET3, self.enc_tx_type);
                    let s = self.sd.symbol(&mut self.cdf.inter_tx_type_set3[sqr], p);
                    TX_TYPE_INTER_INV_SET3[s]
                }
            } else {
                let intra_dir = if self.b.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[self.b.filter_intra_mode]
                } else {
                    self.b.y_mode
                };
                if set == TX_SET_INTRA_1 {
                    let p = inv_index(&TX_TYPE_INTRA_INV_SET1, self.enc_tx_type);
                    let s = self
                        .sd
                        .symbol(&mut self.cdf.intra_tx_type_set1[sqr][intra_dir], p);
                    TX_TYPE_INTRA_INV_SET1[s]
                } else {
                    let p = inv_index(&TX_TYPE_INTRA_INV_SET2, self.enc_tx_type);
                    let s = self
                        .sd
                        .symbol(&mut self.cdf.intra_tx_type_set2[sqr][intra_dir], p);
                    TX_TYPE_INTRA_INV_SET2[s]
                }
            }
        } else {
            DCT_DCT
        };
        self.set_tx_types(x4, y4, tx_sz, tx_type);
    }

    fn set_tx_types(&mut self, x4: usize, y4: usize, tx_sz: usize, tx_type: usize) {
        let cols = self.f.ms;
        for i in 0..(TX_WIDTH[tx_sz] >> 2) {
            for j in 0..(TX_HEIGHT[tx_sz] >> 2) {
                if y4 + j < self.f.mi_rows + 32 && x4 + i < cols {
                    self.f.tx_types[(y4 + j) * cols + x4 + i] = tx_type as u8;
                }
            }
        }
    }

    fn get_scan(&self, tx_sz: usize) -> &'static [u16] {
        scan_for(tx_sz, self.plane_tx_type)
    }

    /// `coeffs( plane, startX, startY, txSz )`: returns `eob`.
    /// `coeffs()` (5.11.39): reads the coefficients of a transform block
    /// into `Quant` and returns `eob`. See the module documentation for the
    /// two readers behind it.
    pub(crate) fn coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> usize {
        if self.sd.encoding() || self.coeff_reader == CoefficientReader::Reference {
            return self.coeffs_shared(plane, start_x, start_y, tx_sz);
        }
        if self.coeff_reader == CoefficientReader::Fast {
            return self.coeffs_decode(plane, start_x, start_y, tx_sz);
        }
        // Checked: the reference first, then the fast reader from the same
        // state; everything either leaves behind must be the same.
        let (x4, y4) = (start_x >> 2, start_y >> 2);
        let (w4, h4) = (TX_WIDTH[tx_sz] >> 2, TX_HEIGHT[tx_sz] >> 2);
        type Ctx = [Vec<u8>; 4];
        fn contexts(t: &TileDecoder, plane: usize, x: (usize, usize), y: (usize, usize)) -> Ctx {
            [
                t.above_level_ctx[plane][x.0..x.0 + x.1].to_vec(),
                t.above_dc_ctx[plane][x.0..x.0 + x.1].to_vec(),
                t.left_level_ctx[plane][y.0..y.0 + y.1].to_vec(),
                t.left_dc_ctx[plane][y.0..y.0 + y.1].to_vec(),
            ]
        }
        let sd0 = self.sd.dec().clone();
        let cdf0 = self.cdf.clone();
        let ctx0 = contexts(self, plane, (x4, w4), (y4, h4));
        let tx0 = self.plane_tx_type;
        let eob_ref = self.coeffs_shared(plane, start_x, start_y, tx_sz);
        let quant_ref = self.quant.clone();
        let sd_ref = std::mem::replace(self.sd.dec(), sd0);
        let cdf_ref = std::mem::replace(&mut self.cdf, cdf0);
        let ctx_ref = contexts(self, plane, (x4, w4), (y4, h4));
        let tx_ref = std::mem::replace(&mut self.plane_tx_type, tx0);
        self.above_level_ctx[plane][x4..x4 + w4].copy_from_slice(&ctx0[0]);
        self.above_dc_ctx[plane][x4..x4 + w4].copy_from_slice(&ctx0[1]);
        self.left_level_ctx[plane][y4..y4 + h4].copy_from_slice(&ctx0[2]);
        self.left_dc_ctx[plane][y4..y4 + h4].copy_from_slice(&ctx0[3]);
        let eob = self.coeffs_decode(plane, start_x, start_y, tx_sz);
        let same = eob == eob_ref
            && self.quant == quant_ref
            && *self.sd.dec() == sd_ref
            && self.cdf == cdf_ref
            && contexts(self, plane, (x4, w4), (y4, h4)) == ctx_ref
            && self.plane_tx_type == tx_ref;
        assert!(
            same,
            "the decode-only coefficient reader differs from the shared walker: plane {plane}, \
             block at ({start_x}, {start_y}), transform size {tx_sz}, eob {eob} against {eob_ref}"
        );
        eob
    }

    /// The decode-only reader (see the module documentation): `coeffs()`
    /// as `coeffs_shared` reads it, for a decoder alone.
    fn coeffs_decode(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> usize {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = TX_WIDTH[tx_sz] >> 2;
        let h4 = TX_HEIGHT[tx_sz] >> 2;
        let tx_sz_ctx = (TX_SIZE_SQR[tx_sz] + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1;
        let ptype = (plane > 0) as usize;
        let seg_eob: usize = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
            512
        } else {
            1024.min(TX_WIDTH[tx_sz] * TX_HEIGHT[tx_sz])
        };
        self.quant[..seg_eob].fill(0);
        let mut eob = 0usize;
        let mut cul_level: u32 = 0;
        let mut dc_category = 0u8;
        let ctx = self.all_zero_ctx(plane, tx_sz, x4, y4, w4, h4);
        let all_zero = self
            .sd
            .dec()
            .read_symbol(&mut self.cdf.txb_skip[tx_sz_ctx][ctx])
            != 0;
        if all_zero {
            if plane == 0 {
                self.set_tx_types(x4, y4, tx_sz, DCT_DCT);
            }
        } else {
            if plane == 0 {
                self.transform_type(x4, y4, tx_sz);
            }
            self.plane_tx_type = self.compute_tx_type(plane, tx_sz, x4, y4);
            let scan = self.get_scan(tx_sz);
            let eob_multisize = TX_WIDTH_LOG2[tx_sz].min(5) + TX_HEIGHT_LOG2[tx_sz].min(5) - 4;
            let tx_class = get_tx_class(self.plane_tx_type);
            let ectx = if tx_class == TX_CLASS_2D { 0 } else { 1 };
            // The context of the DC sign looks at the neighbouring blocks,
            // not at this one: taken before the decoder is borrowed.
            let dc_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
            let d = self.sd.dec();
            let cdf = &mut *self.cdf;
            let eob_pt = 1 + match eob_multisize {
                0 => d.read_symbol(&mut cdf.eob_pt_16[ptype][ectx]),
                1 => d.read_symbol(&mut cdf.eob_pt_32[ptype][ectx]),
                2 => d.read_symbol(&mut cdf.eob_pt_64[ptype][ectx]),
                3 => d.read_symbol(&mut cdf.eob_pt_128[ptype][ectx]),
                4 => d.read_symbol(&mut cdf.eob_pt_256[ptype][ectx]),
                5 => d.read_symbol(&mut cdf.eob_pt_512[ptype]),
                _ => d.read_symbol(&mut cdf.eob_pt_1024[ptype]),
            };
            eob = if eob_pt < 2 {
                eob_pt
            } else {
                (1 << (eob_pt - 2)) + 1
            };
            let eob_shift = eob_pt as i32 - 3;
            if eob_shift >= 0 {
                if d.read_symbol(&mut cdf.eob_extra[tx_sz_ctx][ptype][eob_pt - 3]) != 0 {
                    eob += 1 << eob_shift;
                }
                let lim = (eob_pt as i32 - 2).max(0);
                for i in 1..lim {
                    if d.read_bool() != 0 {
                        eob += 1 << (lim - 1 - i);
                    }
                }
            }
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let bwl = TX_WIDTH_LOG2[adj];
            let height = TX_HEIGHT[adj];
            // The padded level map and where the context neighbours of
            // this transform class lie in it (all at or after the
            // coefficient, at most four rows or columns away).
            let width = 1usize << bwl;
            let lstride = width + 4;
            let levels = &mut self.levels[..(height + 4) * lstride];
            levels.fill(0);
            let base_off: [usize; SIG_REF_DIFF_OFFSET_NUM] = std::array::from_fn(|i| {
                let o = SIG_REF_DIFF_OFFSET[tx_class][i];
                o[0] as usize * lstride + o[1] as usize
            });
            let br_off: [usize; 3] = std::array::from_fn(|i| {
                let o = MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][i];
                o[0] as usize * lstride + o[1] as usize
            });
            let quant = &mut self.quant;
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
                // At most NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1 = 15 here
                // (the Golomb part comes with the signs, below).
                levels[lpos] = level as u8;
            }
            for c in 0..eob {
                let pos = scan[c] as usize;
                let sign = if quant[pos] != 0 {
                    if c == 0 {
                        d.read_symbol_fixed(&mut cdf.dc_sign[ptype][dc_ctx]) as u32
                    } else {
                        d.read_bool()
                    }
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
                    quant[pos] = (x as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64)
                        .min(i32::MAX as u64) as i32;
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
            cul_level = cul_level.min(63);
        }
        self.above_level_ctx[plane][x4..x4 + w4].fill(cul_level as u8);
        self.above_dc_ctx[plane][x4..x4 + w4].fill(dc_category);
        self.left_level_ctx[plane][y4..y4 + h4].fill(cul_level as u8);
        self.left_dc_ctx[plane][y4..y4 + h4].fill(dc_category);
        eob
    }

    /// The tile walker's reader, shared with the encoder: the reference
    /// (see the module documentation).
    fn coeffs_shared(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> usize {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = TX_WIDTH[tx_sz] >> 2;
        let h4 = TX_HEIGHT[tx_sz] >> 2;
        let tx_sz_ctx = (TX_SIZE_SQR[tx_sz] + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1;
        let ptype = (plane > 0) as usize;
        let seg_eob: usize = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
            512
        } else {
            1024.min(TX_WIDTH[tx_sz] * TX_HEIGHT[tx_sz])
        };
        // Encode mode: the encoder chooses the levels now, the prediction
        // being in place (its search may code this block on trial).
        let mut target_eob = 0usize;
        if self.sd.encoding() {
            target_eob = self.enc_choose_coeffs(plane, start_x, start_y, tx_sz);
        }
        self.quant[..seg_eob].fill(0);
        let mut eob = 0usize;
        let mut cul_level: u32 = 0;
        let mut dc_category = 0u8;
        let ctx = self.all_zero_ctx(plane, tx_sz, x4, y4, w4, h4);
        let all_zero = self.sd.symbol(
            &mut self.cdf.txb_skip[tx_sz_ctx][ctx],
            (target_eob == 0) as usize,
        ) != 0;
        if all_zero {
            if plane == 0 {
                self.set_tx_types(x4, y4, tx_sz, DCT_DCT);
            }
        } else {
            if plane == 0 {
                self.transform_type(x4, y4, tx_sz);
            }
            self.plane_tx_type = self.compute_tx_type(plane, tx_sz, x4, y4);
            let scan = self.get_scan(tx_sz);
            let eob_multisize = TX_WIDTH_LOG2[tx_sz].min(5) + TX_HEIGHT_LOG2[tx_sz].min(5) - 4;
            let tx_class = get_tx_class(self.plane_tx_type);
            let ectx = if tx_class == TX_CLASS_2D { 0 } else { 1 };
            let t_pt = if target_eob <= 2 {
                target_eob
            } else {
                crate::bits::floor_log2(target_eob as u32 - 1) as usize + 2
            };
            let t_rest = if t_pt >= 3 {
                target_eob - ((1 << (t_pt - 2)) + 1)
            } else {
                0
            };
            let p = t_pt.saturating_sub(1);
            let eob_pt = 1 + match eob_multisize {
                0 => self.sd.symbol(&mut self.cdf.eob_pt_16[ptype][ectx], p),
                1 => self.sd.symbol(&mut self.cdf.eob_pt_32[ptype][ectx], p),
                2 => self.sd.symbol(&mut self.cdf.eob_pt_64[ptype][ectx], p),
                3 => self.sd.symbol(&mut self.cdf.eob_pt_128[ptype][ectx], p),
                4 => self.sd.symbol(&mut self.cdf.eob_pt_256[ptype][ectx], p),
                5 => self.sd.symbol(&mut self.cdf.eob_pt_512[ptype], p),
                _ => self.sd.symbol(&mut self.cdf.eob_pt_1024[ptype], p),
            };
            eob = if eob_pt < 2 {
                eob_pt
            } else {
                (1 << (eob_pt - 2)) + 1
            };
            let eob_shift = eob_pt as i32 - 3;
            if eob_shift >= 0 {
                let pe = (t_rest >> eob_shift.max(0)) & 1;
                let eob_extra = self
                    .sd
                    .symbol(&mut self.cdf.eob_extra[tx_sz_ctx][ptype][eob_pt - 3], pe);
                if eob_extra != 0 {
                    eob += 1 << eob_shift;
                }
                let lim = (eob_pt as i32 - 2).max(0);
                for i in 1..lim {
                    let eob_shift = lim - 1 - i;
                    if self.sd.literal(1, ((t_rest >> eob_shift) & 1) as u32) != 0 {
                        eob += 1 << eob_shift;
                    }
                }
            }
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let bwl = TX_WIDTH_LOG2[adj];
            let height = TX_HEIGHT[adj];
            for c in (0..eob).rev() {
                let pos = scan[c] as usize;
                let mut level;
                if c == eob - 1 {
                    let ctx = coeff_base_eob_ctx(c, bwl, height);
                    let p = (self.enc_level(pos).clamp(1, 3) - 1) as usize;
                    level = self
                        .sd
                        .symbol(&mut self.cdf.coeff_base_eob[tx_sz_ctx][ptype][ctx], p)
                        as u32
                        + 1;
                } else {
                    let ctx = self.coeff_base_ctx(tx_sz, tx_class, bwl, height, pos);
                    let p = self.enc_level(pos).min(3) as usize;
                    level = self
                        .sd
                        .symbol(&mut self.cdf.coeff_base[tx_sz_ctx][ptype][ctx], p)
                        as u32;
                }
                if level > NUM_BASE_LEVELS {
                    let br_ctx = self.coeff_br_ctx(tx_sz, tx_class, pos);
                    for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                        let p = (self.enc_level(pos).min(15).saturating_sub(level)).min(3) as usize;
                        let coeff_br = self.sd.symbol(
                            &mut self.cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype][br_ctx],
                            p,
                        ) as u32;
                        level += coeff_br;
                        if coeff_br < BR_CDF_SIZE - 1 {
                            break;
                        }
                    }
                }
                self.quant[pos] = level as i32;
            }
            for c in 0..eob {
                let pos = scan[c] as usize;
                let sign = if self.quant[pos] != 0 {
                    if c == 0 {
                        let ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
                        let p = (self.enc_coef(pos) < 0) as usize;
                        self.sd.symbol(&mut self.cdf.dc_sign[ptype][ctx], p) as u32
                    } else {
                        self.sd.literal(1, (self.enc_coef(pos) < 0) as u32)
                    }
                } else {
                    0
                };
                if self.quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
                    // Encode mode: x = level - 14, coded as Exp-Golomb.
                    let gx = self.enc_level(pos).max(15) - 14;
                    let glen = crate::bits::floor_log2(gx) + 1;
                    let mut length = 0;
                    loop {
                        length += 1;
                        let golomb_length_bit = self.sd.literal(1, (length == glen) as u32);
                        if golomb_length_bit != 0 {
                            break;
                        }
                        if length > 32 {
                            break;
                        }
                    }
                    let mut x: u32 = 1;
                    for i in (0..length - 1).rev() {
                        x = (x << 1) | self.sd.literal(1, (gx >> i) & 1);
                    }
                    self.quant[pos] = (x as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64)
                        .min(i32::MAX as u64) as i32;
                }
                if pos == 0 && self.quant[pos] > 0 {
                    dc_category = if sign != 0 { 1 } else { 2 };
                }
                self.quant[pos] &= 0xFFFFF;
                cul_level += self.quant[pos] as u32;
                if sign != 0 {
                    self.quant[pos] = -self.quant[pos];
                }
            }
            cul_level = cul_level.min(63);
        }
        if eob > 0
            && let Some(e) = self.enc.as_mut()
        {
            e.rdo.nonzero += 1;
        }
        for i in 0..w4 {
            self.above_level_ctx[plane][x4 + i] = cul_level as u8;
            self.above_dc_ctx[plane][x4 + i] = dc_category;
        }
        for i in 0..h4 {
            self.left_level_ctx[plane][y4 + i] = cul_level as u8;
            self.left_dc_ctx[plane][y4 + i] = dc_category;
        }
        eob
    }

    fn all_zero_ctx(
        &self,
        plane: usize,
        tx_sz: usize,
        x4: usize,
        y4: usize,
        w4: usize,
        h4: usize,
    ) -> usize {
        let mut max_x4 = self.f.mi_cols;
        let mut max_y4 = self.f.mi_rows;
        if plane > 0 {
            max_x4 >>= self.f.ssx;
            max_y4 >>= self.f.ssy;
        }
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let bsize = self.f.plane_residual_size(self.b.mi_size, plane);
        let bw = block_width(bsize);
        let bh = block_height(bsize);
        if plane == 0 {
            let mut top = 0u8;
            let mut left = 0u8;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(self.above_level_ctx[plane][x4 + k]);
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(self.left_level_ctx[plane][y4 + k]);
                }
            }
            let (top, left) = (top as usize, left as usize);
            if bw == w && bh == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + (top.max(left) > 3) as usize
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    above |= self.above_level_ctx[plane][x4 + i];
                    above |= self.above_dc_ctx[plane][x4 + i];
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    left |= self.left_level_ctx[plane][y4 + i];
                    left |= self.left_dc_ctx[plane][y4 + i];
                }
            }
            let mut ctx = (above != 0) as usize + (left != 0) as usize;
            ctx += 7;
            if bw * bh > w * h {
                ctx += 3;
            }
            ctx
        }
    }

    fn coeff_base_ctx(
        &self,
        tx_sz: usize,
        tx_class: usize,
        bwl: usize,
        height: usize,
        pos: usize,
    ) -> usize {
        let width = 1usize << bwl;
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0i32;
        for idx in 0..SIG_REF_DIFF_OFFSET_NUM {
            let ref_row = row as i32 + SIG_REF_DIFF_OFFSET[tx_class][idx][0];
            let ref_col = col as i32 + SIG_REF_DIFF_OFFSET[tx_class][idx][1];
            if ref_row >= 0
                && ref_col >= 0
                && (ref_row as usize) < height
                && (ref_col as usize) < width
            {
                mag += self.quant[((ref_row as usize) << bwl) + ref_col as usize]
                    .abs()
                    .min(3);
            }
        }
        let ctx = ((mag + 1) >> 1).min(4) as usize;
        if tx_class == TX_CLASS_2D {
            if row == 0 && col == 0 {
                return 0;
            }
            return ctx + COEFF_BASE_CTX_OFFSET[tx_sz][row.min(4)][col.min(4)];
        }
        let idx = if tx_class == TX_CLASS_VERT { row } else { col };
        ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
    }

    fn coeff_br_ctx(&self, tx_sz: usize, tx_class: usize, pos: usize) -> usize {
        let adj = ADJUSTED_TX_SIZE[tx_sz];
        let bwl = TX_WIDTH_LOG2[adj];
        let txw = TX_WIDTH[adj];
        let txh = TX_HEIGHT[adj];
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0i32;
        for idx in 0..3 {
            let ref_row = row as i32 + MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][0];
            let ref_col = col as i32 + MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][1];
            if ref_row >= 0
                && ref_col >= 0
                && (ref_row as usize) < txh
                && (ref_col as usize) < (1 << bwl)
            {
                mag += self.quant[ref_row as usize * txw + ref_col as usize]
                    .min((COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as i32);
            }
        }
        let mag = ((mag + 1) >> 1).min(6) as usize;
        if pos == 0 {
            mag
        } else if tx_class == 0 {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        } else if tx_class == 1 {
            if col == 0 { mag + 7 } else { mag + 14 }
        } else if row == 0 {
            mag + 7
        } else {
            mag + 14
        }
    }

    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize, w4: usize, h4: usize) -> usize {
        let mut max_x4 = self.f.mi_cols;
        let mut max_y4 = self.f.mi_rows;
        if plane > 0 {
            max_x4 >>= self.f.ssx;
            max_y4 >>= self.f.ssy;
        }
        let mut dc_sign = 0i32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                match self.above_dc_ctx[plane][x4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                match self.left_dc_ctx[plane][y4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        }
    }

    /// The quantiser index of the current block: `get_qindex( 0, segment_id )`.
    pub(crate) fn qindex(&self) -> i32 {
        let h = &self.f.hdr;
        let seg = self.b.segment_id;
        if h.segmentation_enabled && h.feature_enabled[seg][SEG_LVL_ALT_Q] {
            let data = h.feature_data[seg][SEG_LVL_ALT_Q];
            let mut q = h.base_q_idx as i32 + data;
            if h.delta_q_present {
                q = self.current_q_index + data;
            }
            clip3(0, 255, q)
        } else if h.delta_q_present {
            self.current_q_index
        } else {
            h.base_q_idx as i32
        }
    }

    /// The reconstruct process (7.12.3).
    pub(crate) fn reconstruct(&mut self, plane: usize, x: usize, y: usize, tx_sz: usize) {
        let bd = self.f.bit_depth;
        // `dqDenom` is 1, 2 or 4: a shift of the magnitude.
        let dq_shift: u32 = match tx_sz {
            TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 1,
            TX_64X64 | TX_32X64 | TX_64X32 => 2,
            _ => 0,
        };
        let log2w = TX_WIDTH_LOG2[tx_sz];
        let log2h = TX_HEIGHT_LOG2[tx_sz];
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let tw = w.min(32);
        let th = h.min(32);
        let t = self.plane_tx_type;
        let flip_ud = matches!(
            t,
            FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
        );
        let flip_lr = matches!(
            t,
            DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
        );
        let qindex = self.qindex();
        let (dc_delta, ac_delta) = match plane {
            0 => (self.f.hdr.delta_q_y_dc, 0),
            1 => (self.f.hdr.delta_q_u_dc, self.f.hdr.delta_q_u_ac),
            _ => (self.f.hdr.delta_q_v_dc, self.f.hdr.delta_q_v_ac),
        };
        let bdi = ((bd - 8) >> 1) as usize;
        let dc_q = DC_QLOOKUP[bdi][clip3(0, 255, qindex + dc_delta) as usize] as i64;
        let ac_q = AC_QLOOKUP[bdi][clip3(0, 255, qindex + ac_delta) as usize] as i64;
        let qm_level = if self.f.hdr.using_qmatrix {
            self.f.hdr.seg_qm_level[plane][self.b.segment_id]
        } else {
            15
        };
        let use_qm = self.f.hdr.using_qmatrix && t < IDTX && qm_level < 15;
        let lim = 1i64 << (7 + bd);
        let dq = &mut self.dequant;
        let mut ac = false;
        for i in 0..th {
            for j in 0..tw {
                let coef = self.quant[i * tw + j] as i64;
                if coef == 0 {
                    dq[i * 64 + j] = 0;
                    continue;
                }
                ac |= (i | j) != 0;
                let q = if i == 0 && j == 0 { dc_q } else { ac_q };
                let q2 = if use_qm {
                    let m = QUANTIZER_MATRIX[qm_level as usize][(plane > 0) as usize]
                        [QM_OFFSET[tx_sz] + i * tw + j] as i64;
                    round2_64(q * m, 5)
                } else {
                    q
                };
                let v = coef * q2;
                let mag = (v.abs() & 0xFFFFFF) >> dq_shift;
                let v = if v < 0 { -mag } else { mag };
                dq[i * 64 + j] = v.clamp(-lim, lim - 1) as i32;
            }
            for j in tw..64.min(w) {
                dq[i * 64 + j] = 0;
            }
        }
        let maxv = (1i32 << bd) - 1;
        let cur = &mut self.f.cur.planes[plane];
        let stride = cur.stride;
        if !ac
            && t == DCT_DCT
            && !self.b.lossless
            && let Some(col) = crate::dsp::itx::inverse_dct_dc_only(self.dequant[0], tx_sz, bd)
        {
            // Only the DC coefficient: one value for each row of the block.
            for (i, &r) in col[..h].iter().enumerate() {
                for o in cur.data[(y + i) * stride + x..][..w].iter_mut() {
                    *o = (*o as i32 + r).clamp(0, maxv) as u16;
                }
            }
            return;
        }
        inverse_transform_2d(
            &self.dequant[..],
            tx_sz,
            t,
            self.b.lossless,
            bd,
            &mut self.residual[..],
        );
        for (i, res) in self.residual[..w * h].chunks_exact(w).enumerate() {
            let yy = if flip_ud { h - i - 1 } else { i };
            let row = &mut cur.data[(y + yy) * stride + x..][..w];
            if flip_lr {
                for (o, &r) in row.iter_mut().rev().zip(res) {
                    *o = (*o as i32 + r).clamp(0, maxv) as u16;
                }
            } else {
                for (o, &r) in row.iter_mut().zip(res) {
                    *o = (*o as i32 + r).clamp(0, maxv) as u16;
                }
            }
        }
    }
}

/// The index of `v` in a transform type inversion table (encode mode).
fn inv_index(tbl: &[usize], v: usize) -> usize {
    tbl.iter().position(|&t| t == v).unwrap_or(0)
}

/// `get_scan( txSz )` for a block of transform type `t` (`PlaneTxType`).
pub(crate) fn scan_for(tx_sz: usize, t: usize) -> &'static [u16] {
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

pub(crate) fn get_tx_class(tx_type: usize) -> usize {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    }
}

/// `get_coeff_base_ctx()` for a coefficient that is not the last, given
/// `mag`, the sum over its `SIG_REF_DIFF_OFFSET_NUM` neighbours of
/// `Min( Abs( Quant ), 3 )` (the decode-only reader's form of
/// `TileDecoder::coeff_base_ctx`).
#[inline(always)]
fn coeff_base_ctx_of(tx_sz: usize, tx_class: usize, row: usize, col: usize, mag: u32) -> usize {
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

/// The context of `coeff_br`, given `mag`, the sum over its three
/// neighbours of `Min( Quant, COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1 )` (the
/// decode-only reader's form of `TileDecoder::coeff_br_ctx`).
#[inline(always)]
fn coeff_br_ctx_of(tx_class: usize, row: usize, col: usize, mag: u32) -> usize {
    let mag = mag.div_ceil(2).min(6) as usize;
    if row == 0 && col == 0 {
        mag
    } else if tx_class == 0 {
        if row < 2 && col < 2 {
            mag + 7
        } else {
            mag + 14
        }
    } else if tx_class == 1 {
        if col == 0 { mag + 7 } else { mag + 14 }
    } else if row == 0 {
        mag + 7
    } else {
        mag + 14
    }
}

/// `get_coeff_base_ctx()` with `isEob` set, rebased for `coeff_base_eob`.
fn coeff_base_eob_ctx(c: usize, bwl: usize, height: usize) -> usize {
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
