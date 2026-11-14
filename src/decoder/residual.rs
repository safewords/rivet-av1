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
//! it. It is not to be bent for speed.
//!
//! Its levels and signs (`levels_reference`) have an encoder's version,
//! `levels_encode`, which the encoder codes them with: it writes from the
//! levels the encoder chose, with the decode-only reader's scan maps and
//! padded level map for the contexts (below), and straight to the symbol
//! encoder. That is where the encoder's trial coding spends its time.
//! `levels_checked` runs both from the same state and panics unless the
//! symbols (or their cost), the CDFs, `Quant` and what they return are the
//! same; a test encodes with it on every transform block.
//!
//! `coeffs_decode` is the same syntax for decoding alone, where most of a
//! decode's symbols are read: no encoder values to carry; each scan worked
//! out once into where a coefficient lies and which CDF its position gives
//! it; the levels' `coeff_base` symbols kept in a padded byte map so that
//! a context is a sum without edge tests; the level symbols read without a
//! branch on what they are; the signs read for the coefficients that have
//! one, found by a bit each, and each of those dequantised there and
//! then. Decoding uses it.
//!
//! They must never differ, and a test holds them to that:
//! `CoefficientReader::Checked` runs both on every transform block from
//! the same state and panics unless the coefficients, the end-of-block
//! position, every CDF, the symbol decoder's state (bit position included)
//! and the contexts left for later blocks are identical. The vector, fuzz
//! and encoder round-trip tests all decode that way. In that mode the
//! symbol decoder is checked too: the reference reader's symbols, and all
//! the rest of the tile's from its first transform block on, are read by
//! `symbol::ReferenceSymbolDecoder` (8.2 as written) alongside it.

use crate::Result;
use crate::consts::*;
use crate::decoder::CoefficientReader;
use crate::decoder::record::Pass;
use crate::decoder::tile::{TileDecoder, find_tx_size};
use crate::dsp::itx::inverse_transform_2d;
use crate::symbol::{Coder, CostCounter, SymbolSink};
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
        if !self.b.is_inter && self.pass != Pass::Parse {
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
            match self.pass {
                Pass::Both => {
                    let (eob, dequantised) =
                        self.coeffs_dequantised(plane, start_x, start_y, tx_sz);
                    // The encoder reconstructs as a decoder does: what it
                    // codes is then what the decoder outputs, by
                    // construction.
                    if eob > 0 {
                        self.reconstruct_decode(plane, start_x, start_y, tx_sz, eob, dequantised);
                    }
                }
                Pass::Parse => {
                    let (eob, dequantised) =
                        self.coeffs_dequantised(plane, start_x, start_y, tx_sz);
                    self.record_tx(tx_sz, eob, dequantised);
                }
                Pass::Recon => {
                    let (eob, dequantised) = self.replay_tx(tx_sz);
                    if eob > 0 {
                        self.reconstruct_decode(plane, start_x, start_y, tx_sz, eob, dequantised);
                    }
                }
            }
        }
        // The transform sizes are mode info, the parsing walk's to write;
        // `BlockDecoded` is asked by intra prediction alone.
        let cols = self.f.ms;
        let (sizes, decoded) = (self.pass != Pass::Recon, self.pass != Pass::Parse);
        for i in 0..step_y {
            for j in 0..step_x {
                let rr = (row >> sub_y) + i;
                let cc = (col >> sub_x) + j;
                if sizes && rr < self.f.mi_rows + 32 && cc < cols {
                    self.f.lf_tx_sizes[plane][rr * cols + cc] = tx_sz as u8;
                }
                let by = (sub_block_mi_row >> sub_y) + i;
                let bx = (sub_block_mi_col >> sub_x) + j;
                if decoded && by < 33 && bx < 33 {
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
        self.coeffs_dequantised(plane, start_x, start_y, tx_sz).0
    }

    /// [`Self::coeffs`], and, when the decode-only reader read the block
    /// and dequantised it into `Dequant` as it went, the rows and columns
    /// its coefficients reach (see [`Dequantiser::scan`]).
    fn coeffs_dequantised(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> (usize, Option<(usize, usize)>) {
        if self.sd.encoding() || self.coeff_reader == CoefficientReader::Reference {
            return (self.coeffs_shared(plane, start_x, start_y, tx_sz), None);
        }
        if self.coeff_reader == CoefficientReader::Fast {
            return self.coeffs_decode(plane, start_x, start_y, tx_sz);
        }
        // Checked: the reference first, then the fast reader from the same
        // state; everything either leaves behind must be the same. And the
        // reference reads through the reference symbol decoder as well
        // (`Coder::Both`), as everything else in the tile does from here on.
        self.sd.check_against_reference();
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
        let (sd_ref, sd_reference) = self.sd.split(sd0);
        let cdf_ref = std::mem::replace(&mut self.cdf, cdf0);
        let ctx_ref = contexts(self, plane, (x4, w4), (y4, h4));
        let tx_ref = std::mem::replace(&mut self.plane_tx_type, tx0);
        self.above_level_ctx[plane][x4..x4 + w4].copy_from_slice(&ctx0[0]);
        self.above_dc_ctx[plane][x4..x4 + w4].copy_from_slice(&ctx0[1]);
        self.left_level_ctx[plane][y4..y4 + h4].copy_from_slice(&ctx0[2]);
        self.left_dc_ctx[plane][y4..y4 + h4].copy_from_slice(&ctx0[3]);
        let (eob, dequantised) = self.coeffs_decode(plane, start_x, start_y, tx_sz);
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
        self.sd.rejoin(sd_reference);
        (eob, dequantised)
    }

    /// The decode-only reader (see the module documentation): `coeffs()`
    /// as `coeffs_shared` reads it, for a decoder alone.
    fn coeffs_decode(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> (usize, Option<(usize, usize)>) {
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
        let mut dequantised = None;
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
            let eob_multisize = TX_WIDTH_LOG2[tx_sz].min(5) + TX_HEIGHT_LOG2[tx_sz].min(5) - 4;
            let tx_class = get_tx_class(self.plane_tx_type);
            let ectx = if tx_class == TX_CLASS_2D { 0 } else { 1 };
            // The context of the DC sign looks at the neighbouring blocks,
            // not at this one: taken before the decoder is borrowed.
            let dc_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
            // What `reconstruct_decode` dequantises itself, this dequantises
            // as it reads the signs: the coefficients that are not zero are
            // in hand there, one at a time.
            let dequantiser = (!self.b.lossless && self.f.bit_depth <= 10)
                .then(|| self.dequantiser(plane, tx_sz));
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
            // Each coefficient of the scan: where it lies, and the CDF its
            // position alone would give it.
            let map = &scan_map(tx_sz, tx_class)[..eob];
            let near = match tx_class {
                TX_CLASS_2D => NEAR_2D,
                TX_CLASS_HORIZ => NEAR_HORIZ,
                _ => NEAR_VERT,
            };
            self.levels[..(height + 4) * LEVEL_STRIDE].fill(0);
            let cdf_eob = &mut cdf.coeff_base_eob[tx_sz_ctx][ptype];
            // The levels, from the last coefficient back. From here to the
            // signs the symbol decoder is a local of this function, which
            // nothing takes the address of: it stays in registers.
            let mut pass = LevelPass {
                d: d.clone(),
                cdf_base: cdf.coeff_base[tx_sz_ctx][ptype].as_flattened_mut(),
                cdf_br: &mut cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype],
                levels: &mut self.levels,
                quant: &mut self.quant,
                tx_class,
                bwl,
                height,
            };
            // Which coefficients of the scan are not zero, a bit each.
            let mut nonzero = [0u64; 16];
            // The last coefficient has CDFs of its own and is never zero.
            let c = eob - 1;
            let ctx = coeff_base_eob_ctx(c, bwl, height);
            let base = pass.d.read_symbol_fixed(&mut cdf_eob[ctx]) + 1;
            pass.level(map[c] as usize, base);
            nonzero[c >> 6] |= 1 << (c & 63);
            // The ones between it and the first.
            if eob > 2 {
                match tx_class {
                    TX_CLASS_2D => pass.between(map, &mut nonzero, NEAR_2D),
                    TX_CLASS_HORIZ => pass.between(map, &mut nonzero, NEAR_HORIZ),
                    _ => pass.between(map, &mut nonzero, NEAR_VERT),
                }
            }
            // The first, whose context is 0 in a 2D transform whatever its
            // neighbours.
            if eob > 1 {
                let e = map[0] as usize;
                let around = &pass.levels[e & MAP_LPOS..];
                let mag: u32 = near.iter().map(|&o| around[o] as u32).sum();
                let ctx = coeff_base_ctx_of(tx_sz, tx_class, 0, 0, mag);
                let base = pass.d.read_symbol_fixed(cdf_of(pass.cdf_base, 5 * ctx));
                pass.level(e, base);
                nonzero[0] |= (base != 0) as u64;
            }
            let LevelPass { mut d, quant, .. } = pass;
            // The signs and the Golomb parts, in scan order: only a
            // coefficient that is not zero has either, and `Quant` holds
            // its level. Each is dequantised as it is completed.
            let dq = &mut self.dequant;
            if let Some(dequantiser) = &dequantiser {
                dequantiser.clear(dq);
            }
            let mut reach = (1, 1);
            if nonzero[0] & 1 != 0 {
                let sign = d.read_symbol_fixed(&mut cdf.dc_sign[ptype][dc_ctx]);
                let level = level_golomb(&mut d, quant[0] as u32);
                dc_category = if sign != 0 { 1 } else { 2 };
                let level = level & 0xFFFFF;
                cul_level += level;
                quant[0] = if sign != 0 {
                    -(level as i32)
                } else {
                    level as i32
                };
                if let Some(dequantiser) = &dequantiser {
                    dequantiser.put(dq, 0, quant[0], &mut reach);
                }
                nonzero[0] &= !1;
            }
            for (i, word) in nonzero[..=(eob - 1) >> 6].iter().enumerate() {
                let mut word = *word;
                while word != 0 {
                    let c = (i << 6) + word.trailing_zeros() as usize;
                    word &= word - 1;
                    let pos = map[c] as usize >> MAP_POS_SHIFT;
                    let sign = d.read_bool();
                    let level = level_golomb(&mut d, quant[pos] as u32) & 0xFFFFF;
                    cul_level += level;
                    quant[pos] = if sign != 0 {
                        -(level as i32)
                    } else {
                        level as i32
                    };
                    if let Some(dequantiser) = &dequantiser {
                        dequantiser.put(dq, pos, quant[pos], &mut reach);
                    }
                }
            }
            dequantised = dequantiser.is_some().then_some(reach);
            *self.sd.dec() = d;
            cul_level = cul_level.min(63);
        }
        self.above_level_ctx[plane][x4..x4 + w4].fill(cul_level as u8);
        self.above_dc_ctx[plane][x4..x4 + w4].fill(dc_category);
        self.left_level_ctx[plane][y4..y4 + h4].fill(cul_level as u8);
        self.left_dc_ctx[plane][y4..y4 + h4].fill(dc_category);
        (eob, dequantised)
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
            let block = LevelBlock {
                plane,
                tx_sz,
                tx_sz_ctx,
                ptype,
                tx_class,
                bwl,
                height,
                eob,
                at: (x4, y4, w4, h4),
            };
            (cul_level, dc_category) = if !self.sd.encoding() {
                self.levels_reference(&block, scan)
            } else if CHECK_ENCODER_LEVELS.load(std::sync::atomic::Ordering::Relaxed) {
                self.levels_checked(&block, scan)
            } else {
                self.levels_encode(&block)
            };
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

    /// The levels and signs of `coeffs()` (5.11.39), from the last
    /// coefficient back, then the signs and Golomb parts from the first:
    /// the reference reader's, and the encoder's reference. Returns the
    /// sum of the levels (before its `Min( 63, ... )`) and the DC
    /// category.
    fn levels_reference(&mut self, b: &LevelBlock, scan: &[u16]) -> (u32, u8) {
        let LevelBlock {
            plane,
            tx_sz,
            tx_sz_ctx,
            ptype,
            tx_class,
            bwl,
            height,
            eob,
            at: (x4, y4, w4, h4),
        } = *b;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
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
        (cul_level, dc_category)
    }

    /// [`Self::levels_reference`] for the encoder (see the module
    /// documentation): the same symbols with the same CDFs, coded from the
    /// levels the encoder chose (`EncCtx::coefs`).
    fn levels_encode(&mut self, b: &LevelBlock) -> (u32, u8) {
        let LevelBlock {
            plane,
            at: (x4, y4, w4, h4),
            ..
        } = *b;
        // Taken before the coder and the CDFs are borrowed: the context of
        // the DC sign is the neighbours'.
        let dc_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
        let coefs: &[i32; 1024] = &self.enc.as_ref().expect("encode mode").coefs;
        let Coder::Enc(e) = &mut self.sd else {
            unreachable!("the encoder's writer, encoding")
        };
        let at = LevelState {
            cdf: &mut self.cdf,
            quant: &mut self.quant,
            levels: &mut self.levels,
            coefs,
        };
        // Trial coding adds the symbols' costs up, from a table it looks
        // up once.
        if e.counting {
            let mut c = CostCounter::new();
            let r = write_levels(&mut c, b, dc_ctx, at);
            e.bits += c.bits;
            r
        } else {
            write_levels(e, b, dc_ctx, at)
        }
    }

    /// [`Self::levels_encode`] held to [`Self::levels_reference`]: both
    /// from the same state, which must end the same — the symbols coded
    /// (or their cost), the CDFs, `Quant`, what they return — or this
    /// panics. For the tests ([`set_check_encoder_levels`]).
    fn levels_checked(&mut self, b: &LevelBlock, scan: &[u16]) -> (u32, u8) {
        let enc = |t: &Self| match &t.sd {
            Coder::Enc(e) => e.clone(),
            _ => unreachable!("encoding"),
        };
        let (sd0, cdf0, quant0) = (enc(self), self.cdf.clone(), self.quant.clone());
        let r_ref = self.levels_reference(b, scan);
        let (sd_ref, cdf_ref, quant_ref) = (enc(self), self.cdf.clone(), self.quant.clone());
        self.sd = Coder::Enc(sd0);
        self.cdf = cdf0;
        self.quant = quant0;
        let r = self.levels_encode(b);
        let same =
            r == r_ref && enc(self) == sd_ref && self.cdf == cdf_ref && self.quant == quant_ref;
        assert!(
            same,
            "the encoder's level writer differs from the reference: transform size {}, \
             class {}, eob {}",
            b.tx_sz, b.tx_class, b.eob
        );
        r
    }

    pub(crate) fn all_zero_ctx(
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

    pub(crate) fn dc_sign_ctx(
        &self,
        plane: usize,
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

    /// The dequantisation of the current block's plane `plane`, whose
    /// transform type `plane_tx_type` holds.
    fn dequantiser(&self, plane: usize, tx_sz: usize) -> Dequantiser<'static> {
        let bd = self.f.bit_depth;
        let t = self.plane_tx_type;
        let qindex = self.qindex();
        let (dc_delta, ac_delta) = match plane {
            0 => (self.f.hdr.delta_q_y_dc, 0),
            1 => (self.f.hdr.delta_q_u_dc, self.f.hdr.delta_q_u_ac),
            _ => (self.f.hdr.delta_q_v_dc, self.f.hdr.delta_q_v_ac),
        };
        let bdi = ((bd - 8) >> 1) as usize;
        let qm_level = if self.f.hdr.using_qmatrix {
            self.f.hdr.seg_qm_level[plane][self.b.segment_id]
        } else {
            15
        };
        Dequantiser {
            dc_q: DC_QLOOKUP[bdi][clip3(0, 255, qindex + dc_delta) as usize] as u32,
            ac_q: AC_QLOOKUP[bdi][clip3(0, 255, qindex + ac_delta) as usize] as u32,
            qm: (self.f.hdr.using_qmatrix && t < IDTX && qm_level < 15).then(|| {
                &QUANTIZER_MATRIX[qm_level as usize][(plane > 0) as usize][QM_OFFSET[tx_sz]..]
            }),
            shift: match tx_sz {
                TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 1,
                TX_64X64 | TX_32X64 | TX_64X32 => 2,
                _ => 0,
            },
            lim: 1i32 << (7 + bd),
            tw: TX_WIDTH[tx_sz].min(32),
            th: TX_HEIGHT[tx_sz].min(32),
        }
    }

    /// The reconstruct process (7.12.3) for a decoder (and the encoder),
    /// which knows `eob`: what [`Self::reconstruct`] (the reference) does, without the work for coefficients known to be zero. Only the
    /// first `eob` positions of the scan can hold one, so only they are
    /// dequantised (by the decode-only coefficient reader as it read
    /// them, which `dequantised` then says, or here), and the rows and
    /// columns they reach are all the inverse transform is given to
    /// transform.
    #[inline(never)]
    pub(crate) fn reconstruct_decode(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
        eob: usize,
        dequantised: Option<(usize, usize)>,
    ) {
        let bd = self.f.bit_depth;
        if self.b.lossless || bd > 10 {
            // The Walsh-Hadamard transform, and the depth whose
            // intermediates need 64 bits: the reference, as it is.
            return self.reconstruct(plane, x, y, tx_sz);
        }
        let w = 1usize << TX_WIDTH_LOG2[tx_sz];
        let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
        let t = self.plane_tx_type;
        let flip_ud = matches!(
            t,
            FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
        );
        let flip_lr = matches!(
            t,
            DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
        );
        // `Dequant`, unless the coefficient reader left it.
        let (rows, cols) = match dequantised {
            Some(reach) => reach,
            None => {
                let scan = &scan_for(tx_sz, t)[..eob];
                self.dequantiser(plane, tx_sz)
                    .scan(&self.quant, scan, &mut self.dequant)
            }
        };
        let maxv = (1i32 << bd) - 1;
        let cur = &mut self.f.cur.planes[plane];
        let stride = cur.stride;
        if rows == 1
            && cols == 1
            && t == DCT_DCT
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
        crate::dsp::itx::inverse_transform_2d_within(
            &self.dequant[..],
            tx_sz,
            t,
            bd,
            (rows, cols),
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

/// The dequantisation functions of 7.12.3 for one transform block, as
/// [`TileDecoder::reconstruct_decode`] applies them.
struct Dequantiser<'a> {
    /// `dcQ` and `acQ`.
    dc_q: u32,
    ac_q: u32,
    /// The block's quantiser matrix, if it uses one.
    qm: Option<&'a [u8]>,
    /// `dqDenom` as a shift.
    shift: u32,
    /// `1 << ( 7 + BitDepth )`.
    lim: i32,
    /// The width and height of the block's coefficients (32 at most).
    tw: usize,
    th: usize,
}

impl Dequantiser<'_> {
    /// Zeros in `Dequant` (rows of 64) where the block can hold a
    /// coefficient and where the inverse transform may read, which takes
    /// whole groups of eight rows.
    fn clear(&self, dq: &mut [i32; 64 * 64]) {
        for row in dq.as_chunks_mut::<64>().0[..self.th.next_multiple_of(8)].iter_mut() {
            row[..self.tw].fill(0);
        }
    }

    /// The dequantised value of the coefficient `coef` at position `pos`
    /// of the block.
    #[inline(always)]
    fn value(&self, pos: usize, coef: i32) -> i32 {
        use core::hint::select_unpredictable as select;
        let q = select(pos == 0, self.dc_q, self.ac_q);
        let q = match self.qm {
            Some(m) => (q * m[pos] as u32 + 16) >> 5,
            None => q,
        };
        // `( Abs( coef * q ) & 0xFFFFFF ) / dqDenom` with the sign back:
        // the low 24 bits of the product are those of its low 32.
        let mag = ((coef.unsigned_abs().wrapping_mul(q) & 0xFFFFFF) >> self.shift) as i32;
        select(coef < 0, -mag, mag).clamp(-self.lim, self.lim - 1)
    }

    /// Dequantises one coefficient into `Dequant` and widens `reach`, the
    /// rows and columns the block's coefficients reach, to hold it.
    #[inline(always)]
    fn put(&self, dq: &mut [i32; 64 * 64], pos: usize, coef: i32, reach: &mut (usize, usize)) {
        let (i, j) = (pos >> self.tw.trailing_zeros(), pos & (self.tw - 1));
        dq[i * 64 + j] = self.value(pos, coef);
        *reach = (reach.0.max(i + 1), reach.1.max(j + 1));
    }

    /// `Dequant` from the coefficients `quant` holds at the positions of
    /// `scan`, the only ones that can be other than zero. Returns how many
    /// rows and columns the non-zero ones reach (at least 1 each). For a
    /// block the decode-only reader did not dequantise as it read it.
    ///
    /// Every coefficient of the scan is dequantised, zero or not, and the
    /// rows and columns are found by selects: whether a coefficient is
    /// zero cannot be predicted any better than its level could.
    #[inline(never)]
    fn scan(&self, quant: &[i32; 1024], scan: &[u16], dq: &mut [i32; 64 * 64]) -> (usize, usize) {
        use core::hint::select_unpredictable as select;
        let tw = self.tw;
        let bwl = tw.trailing_zeros();
        self.clear(dq);
        let (mut rows, mut cols) = (1, 1);
        for &pos in scan {
            let pos = pos as usize;
            let coef = quant[pos];
            let (i, j) = (pos >> bwl, pos & (tw - 1));
            dq[i * 64 + j] = self.value(pos, coef);
            rows = rows.max(select(coef != 0, i + 1, 0));
            cols = cols.max(select(coef != 0, j + 1, 0));
        }
        (rows, cols)
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

/// The inverse of `get_scan( txSz )` for a block of transform type `t`:
/// entry `pos` is the index of position `pos` in the scan.
pub(crate) fn inverse_scan(tx_sz: usize, t: usize) -> &'static [u16] {
    static MAPS: std::sync::OnceLock<Vec<Box<[u16]>>> = std::sync::OnceLock::new();
    let maps = MAPS.get_or_init(|| {
        let mut maps = Vec::with_capacity(TX_SIZES_ALL * 3);
        for tx_sz in 0..TX_SIZES_ALL {
            // A type of each class: the scan depends on the type through
            // its class alone (see `scan_map`).
            for t in [DCT_DCT, H_DCT, V_DCT] {
                let scan = scan_for(tx_sz, t);
                let mut inv = vec![0u16; scan.len()];
                for (k, &pos) in scan.iter().enumerate() {
                    inv[pos as usize] = k as u16;
                }
                maps.push(inv.into_boxed_slice());
            }
        }
        maps
    });
    let class = match get_tx_class(t) {
        TX_CLASS_2D => 0,
        TX_CLASS_HORIZ => 1,
        _ => 2,
    };
    &maps[tx_sz * 3 + class]
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
pub(crate) fn coeff_base_ctx_of(
    tx_sz: usize,
    tx_class: usize,
    row: usize,
    col: usize,
    mag: u32,
) -> usize {
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
pub(crate) fn coeff_br_ctx_of(tx_class: usize, row: usize, col: usize, mag: u32) -> usize {
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

/// What [`write_levels`] works on: the CDFs, `Quant`, the level map, and
/// the levels the encoder chose.
struct LevelState<'a> {
    cdf: &'a mut crate::cdf::CdfContext,
    quant: &'a mut [i32; 1024],
    levels: &'a mut [u8; 36 * 36],
    coefs: &'a [i32; 1024],
}

/// The body of [`TileDecoder::levels_encode`], for coding or for counting.
fn write_levels<S: SymbolSink>(
    e: &mut S,
    b: &LevelBlock,
    dc_ctx: usize,
    at: LevelState,
) -> (u32, u8) {
    let LevelBlock {
        tx_sz,
        tx_sz_ctx,
        ptype,
        tx_class,
        bwl,
        height,
        eob,
        ..
    } = *b;
    let LevelState {
        cdf,
        quant,
        levels,
        coefs,
    } = at;
    let map = &scan_map(tx_sz, tx_class)[..eob];
    let near = match tx_class {
        TX_CLASS_2D => NEAR_2D,
        TX_CLASS_HORIZ => NEAR_HORIZ,
        _ => NEAR_VERT,
    };
    levels[..(height + 4) * LEVEL_STRIDE].fill(0);
    let cdf_base = cdf.coeff_base[tx_sz_ctx][ptype].as_flattened_mut();
    let cdf_br = &mut cdf.coeff_br[tx_sz_ctx.min(TX_32X32)][ptype];
    // What `coeff_br` adds to a coefficient whose `coeff_base` symbol
    // is `base`, and its level (at most 15: the Golomb part comes with
    // the signs).
    let mut above_base = |e: &mut S, quant: &[i32; 1024], pos: usize, base: u32| {
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
        let target = coefs[pos].unsigned_abs().min(15);
        let mut level = base;
        for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
            let coeff_br = target.saturating_sub(level).min(3);
            e.symbol(&mut cdf_br[br_ctx], coeff_br as usize);
            level += coeff_br;
            if coeff_br < BR_CDF_SIZE - 1 {
                break;
            }
        }
        level
    };
    // The last coefficient, with CDFs of its own; never zero.
    let c = eob - 1;
    let entry = map[c] as usize;
    let pos = entry >> MAP_POS_SHIFT;
    let base = coefs[pos].unsigned_abs().clamp(1, 3);
    let ctx = coeff_base_eob_ctx(c, bwl, height);
    e.symbol(
        &mut cdf.coeff_base_eob[tx_sz_ctx][ptype][ctx],
        base as usize - 1,
    );
    levels[entry & MAP_LPOS] = base as u8;
    quant[pos] = if base > NUM_BASE_LEVELS {
        above_base(e, quant, pos, base)
    } else {
        base
    } as i32;
    // Then back to the first, each with the context of its neighbours
    // (the first's position alone gives its context in a 2D transform,
    // which the scan map says).
    for c in (0..eob - 1).rev() {
        let entry = map[c] as usize;
        let lpos = entry & MAP_LPOS;
        let around: &[u8; LEVEL_NEAR] = levels[lpos..lpos + LEVEL_NEAR]
            .try_into()
            .expect("a slice of that length");
        let mag: usize = near.iter().map(|&o| around[o] as usize).sum();
        let i = if c == 0 {
            5 * coeff_base_ctx_of(tx_sz, tx_class, 0, 0, mag as u32)
        } else {
            MAG_CDF[mag & 15] as usize + ((entry >> MAP_CDF_SHIFT) & MAP_CDF)
        };
        let pos = entry >> MAP_POS_SHIFT;
        let base = coefs[pos].unsigned_abs().min(3);
        e.symbol(cdf_of(cdf_base, i), base as usize);
        levels[lpos] = base as u8;
        quant[pos] = if base > NUM_BASE_LEVELS {
            above_base(e, quant, pos, base)
        } else {
            base
        } as i32;
    }
    // The signs and the Golomb parts, in scan order.
    let (mut cul_level, mut dc_category) = (0u32, 0u8);
    for (c, &entry) in map.iter().enumerate() {
        let pos = entry as usize >> MAP_POS_SHIFT;
        let mut level = quant[pos];
        if level == 0 {
            continue;
        }
        let negative = coefs[pos] < 0;
        if c == 0 {
            e.symbol(&mut cdf.dc_sign[ptype][dc_ctx], negative as usize);
        } else {
            e.bool(negative as u32);
        }
        if level > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
            // x = level - 14, as Exp-Golomb: its length less one in
            // zeros, then its bits from the top.
            let gx = coefs[pos].unsigned_abs().max(15) - 14;
            let glen = crate::bits::floor_log2(gx) + 1;
            for k in 1..=glen {
                e.bool((k == glen) as u32);
            }
            for k in (0..glen - 1).rev() {
                e.bool((gx >> k) & 1);
            }
            level = (gx as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64).min(i32::MAX as u64)
                as i32;
        }
        if pos == 0 {
            dc_category = if negative { 1 } else { 2 };
        }
        level &= 0xFFFFF;
        cul_level += level as u32;
        quant[pos] = if negative { -level } else { level };
    }
    (cul_level, dc_category)
}

/// A transform block whose levels are being coded: what
/// [`TileDecoder::levels_reference`] and its encoder's version take.
#[derive(Clone, Copy)]
struct LevelBlock {
    plane: usize,
    tx_sz: usize,
    tx_sz_ctx: usize,
    ptype: usize,
    tx_class: usize,
    /// `Log2` of the width and the height of the adjusted size.
    bwl: usize,
    height: usize,
    eob: usize,
    /// The block in units of 4 samples: `x4`, `y4`, `w4`, `h4`.
    at: (usize, usize, usize, usize),
}

/// Whether the encoder's level writer runs alongside the reference, each
/// transform block checked against it ([`TileDecoder::levels_checked`]).
static CHECK_ENCODER_LEVELS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Has every encoder in the process check its level writer against the
/// reference, from now on (`true`) or no more: for the tests.
#[cfg(test)]
pub(crate) fn set_check_encoder_levels(on: bool) {
    CHECK_ENCODER_LEVELS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The row stride of the decode-only reader's level map: the widest
/// transform's 32 columns and the 4 of padding the context neighbours
/// reach into.
const LEVEL_STRIDE: usize = 36;
/// How far past a coefficient its context neighbours lie in the level
/// map: four rows down at most.
const LEVEL_NEAR: usize = 4 * LEVEL_STRIDE + 1;

/// A scan map entry (see [`scan_map`]): the coefficient's place in the
/// level map, the index in a block of `coeff_base` CDFs (as one array) of
/// the CDF its position alone gives it, and its position in the block.
const MAP_LPOS: usize = (1 << 11) - 1;
const MAP_CDF_SHIFT: usize = 11;
const MAP_CDF: usize = (1 << 8) - 1;
const MAP_POS_SHIFT: usize = 19;

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

/// `5 * Min( ( mag + 1 ) >> 1, 4 )`: what the neighbours' sum `mag` adds
/// to the index of a coefficient's `coeff_base` CDF (five entries each),
/// for a sum of five values of at most 3.
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

/// For the decode-only reader: the scan of a transform size and class,
/// with what the reader needs of each position worked out once. Entry `c`
/// holds, for the coefficient `pos = scan[c]` at `(row, col)`: its place
/// `row * LEVEL_STRIDE + col` in the level map (`MAP_LPOS`); five times the
/// part of `get_coeff_base_ctx()` that depends on the position alone
/// (`MAP_CDF`: the context is that plus `Min( ( mag + 1 ) >> 1, 4 )` for
/// every coefficient but the first of a 2D transform, whose context is 0);
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
                    ((pos << MAP_POS_SHIFT)
                        | ((5 * ctx) << MAP_CDF_SHIFT)
                        | (row * LEVEL_STRIDE + col)) as u32
                });
                maps.push(map.collect());
            }
        }
        maps
    })[tx_sz * 3 + tx_class]
}

/// The CDF at index `i` of a block of 4-symbol CDFs taken as one array.
#[inline(always)]
fn cdf_of(cdfs: &mut [u16], i: usize) -> &mut [u16; 5] {
    (&mut cdfs[i..i + 5])
        .try_into()
        .expect("a slice of five is an array of five")
}

/// What the decode-only reader's pass over the levels works on.
struct LevelPass<'a, 'd> {
    d: crate::symbol::SymbolDecoder<'d>,
    /// The 42 `coeff_base` CDFs of the block's size and plane type, end to
    /// end.
    cdf_base: &'a mut [u16],
    cdf_br: &'a mut [[u16; 5]; 21],
    /// `Min( level, 3 )` of the coefficients read so far, which is their
    /// `coeff_base` symbol: what the contexts of `coeff_base` sum.
    levels: &'a mut [u8; 36 * 36],
    /// The levels themselves, as the reference keeps them.
    quant: &'a mut [i32; 1024],
    tx_class: usize,
    bwl: usize,
    height: usize,
}

impl LevelPass<'_, '_> {
    /// Stores the level of the coefficient of map entry `e` whose
    /// `coeff_base` symbol (or `coeff_base_eob` plus one) is `base`, after
    /// reading its `coeff_br` part if it has one.
    #[inline(always)]
    fn level(&mut self, e: usize, base: usize) {
        self.levels[e & MAP_LPOS] = base as u8;
        let pos = e >> MAP_POS_SHIFT;
        let mut level = base as u32;
        if level > NUM_BASE_LEVELS {
            (self.d, level) = level_above_base(
                self.d.clone(),
                self.cdf_br,
                self.quant,
                (self.tx_class, self.bwl, self.height),
                pos,
            );
        }
        self.quant[pos] = level as i32;
    }

    /// The coefficients between the last and the first of the block, from
    /// the end of the scan back: `coeff_base` with the context of the five
    /// neighbours at `near` (their offsets in the level map, known where
    /// this is compiled).
    #[inline(always)]
    fn between(
        &mut self,
        map: &[u32],
        nonzero: &mut [u64; 16],
        near: [usize; SIG_REF_DIFF_OFFSET_NUM],
    ) {
        // The bits of the 64 coefficients the pass is in.
        let mut word = 0u64;
        for c in (1..map.len() - 1).rev() {
            let e = map[c] as usize;
            let lpos = e & MAP_LPOS;
            let around: &[u8; LEVEL_NEAR] = self.levels[lpos..lpos + LEVEL_NEAR]
                .try_into()
                .expect("a slice of that length");
            let mag: usize = near.iter().map(|&o| around[o] as usize).sum();
            let i = MAG_CDF[mag & 15] as usize + ((e >> MAP_CDF_SHIFT) & MAP_CDF);
            let base = self.d.read_symbol_fixed(cdf_of(self.cdf_base, i));
            self.level(e, base);
            word |= ((base != 0) as u64) << (c & 63);
            if c & 63 == 0 {
                nonzero[c >> 6] |= word;
                word = 0;
            }
        }
        nonzero[0] |= word;
    }
}

/// `coeff_br` for a coefficient whose `coeff_base` left it above
/// `NUM_BASE_LEVELS`: its level, at most
/// `NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1` (15; the Golomb part comes with
/// the signs). The context is `TileDecoder::coeff_br_ctx`'s, from `Quant`
/// with its edge tests. One coefficient in seventy gets here: it is out of
/// line and takes and returns the symbol decoder by value, so that the
/// level pass keeps it in registers.
#[inline(never)]
fn level_above_base<'d>(
    mut d: crate::symbol::SymbolDecoder<'d>,
    cdf_br: &mut [[u16; 5]; 21],
    quant: &[i32; 1024],
    (tx_class, bwl, height): (usize, usize, usize),
    pos: usize,
) -> (crate::symbol::SymbolDecoder<'d>, u32) {
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
        let coeff_br = d.read_symbol_fixed(&mut cdf_br[br_ctx]) as u32;
        level += coeff_br;
        if coeff_br < BR_CDF_SIZE - 1 {
            break;
        }
    }
    (d, level)
}

/// The Golomb part of a level that the symbols left at their most, 15:
/// `golomb_length_bit`s and the bits they count.
#[inline(always)]
fn level_golomb(d: &mut crate::symbol::SymbolDecoder, level: u32) -> u32 {
    if level <= NUM_BASE_LEVELS + COEFF_BASE_RANGE {
        return level;
    }
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
    (x as u64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64).min(i32::MAX as u64) as u32
}

/// `get_coeff_base_ctx()` with `isEob` set, rebased for `coeff_base_eob`.
pub(crate) fn coeff_base_eob_ctx(c: usize, bwl: usize, height: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoder's level writer against the reference, on every
    /// transform block of encodes that reach every part of it: trial
    /// coding and coding, all the transform classes, levels from 1 to
    /// past the Golomb part's (a fine quantiser on noise), 8 and 10 bits.
    #[test]
    fn encoder_level_writer_matches_the_reference() {
        use crate::{ChromaFormat, Config, Encoder, Frame};
        set_check_encoder_levels(true);
        for (bit_depth, speed, q) in [(8, 1, 8), (8, 4, 60), (8, 6, 140), (10, 8, 30)] {
            let (w, h) = (72u32, 40u32);
            let mut cfg = Config::new(w, h);
            cfg.bit_depth = bit_depth;
            cfg.quantizer = q;
            cfg.speed = speed;
            cfg.tools = crate::Tools::for_speed(speed);
            let mut enc = Encoder::new(cfg);
            let mut seed = 7u32;
            for t in 0..3u32 {
                let mut f = Frame::new(w, h, bit_depth, ChromaFormat::Yuv420);
                for p in 0..3 {
                    let pl = f.planes[p];
                    for y in 0..pl.height {
                        for x in 0..pl.width {
                            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                            let edge = if (x + 2 * t) % 16 < 8 { 40 } else { 200 };
                            let v = (edge + (seed >> 16) % 48) as u16;
                            f.set_sample(p, x, y, v << (bit_depth - 8));
                        }
                    }
                }
                enc.encode(&f).unwrap();
            }
        }
        set_check_encoder_levels(false);
    }

    /// A scan map entry says of its coefficient what the reference
    /// functions say: its position, its place in the level map, and, with
    /// the table of the neighbours' sums, the `coeff_base` context of every
    /// coefficient but the first, for every sum.
    #[test]
    fn scan_map_agrees_with_the_reference_contexts() {
        for tx_sz in 0..TX_SIZES_ALL {
            let bwl = TX_WIDTH_LOG2[ADJUSTED_TX_SIZE[tx_sz]];
            for t in 0..16 {
                let tx_class = get_tx_class(t);
                let scan = scan_for(tx_sz, t);
                let map = scan_map(tx_sz, tx_class);
                assert_eq!(map.len(), scan.len(), "size {tx_sz} type {t}");
                for (c, (&e, &pos)) in map.iter().zip(scan).enumerate() {
                    let (e, pos) = (e as usize, pos as usize);
                    let (row, col) = (pos >> bwl, pos & ((1 << bwl) - 1));
                    assert_eq!(e >> MAP_POS_SHIFT, pos);
                    assert_eq!(e & MAP_LPOS, row * LEVEL_STRIDE + col);
                    assert!((e & MAP_LPOS) + LEVEL_NEAR <= 36 * 36);
                    assert_eq!(c == 0, pos == 0);
                    if c == 0 {
                        continue;
                    }
                    for mag in 0..16u32 {
                        let i = MAG_CDF[mag as usize] as usize + ((e >> MAP_CDF_SHIFT) & MAP_CDF);
                        let ctx = coeff_base_ctx_of(tx_sz, tx_class, row, col, mag);
                        assert_eq!(i, 5 * ctx, "size {tx_sz} type {t} c {c} mag {mag}");
                        assert!(ctx < 42);
                    }
                }
            }
        }
    }

    /// The neighbours of the level map are `Sig_Ref_Diff_Offset`'s, and a
    /// block's padding holds every one of them.
    #[test]
    fn level_map_neighbours() {
        for (tx_class, near) in [NEAR_2D, NEAR_HORIZ, NEAR_VERT].iter().enumerate() {
            for (o, d) in near.iter().zip(&SIG_REF_DIFF_OFFSET[tx_class]) {
                assert_eq!(*o, d[0] as usize * LEVEL_STRIDE + d[1] as usize);
                assert!(d[0] <= 4 && d[1] <= 4 && *o < LEVEL_NEAR);
            }
        }
    }

    /// `Dequantiser::scan` against 7.12.3 written out in 64 bits over the
    /// whole block: the same `Dequant`, and no coefficient outside the
    /// rows and columns it returns, for every size and scan, with and
    /// without a quantiser matrix, levels up to the 20 bits `coeffs()` can
    /// leave.
    #[test]
    fn dequantiser_matches_the_process() {
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for tx_sz in 0..TX_SIZES_ALL {
            let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
            let (tw, th) = (w.min(32), h.min(32));
            let shift: u32 = match tx_sz {
                TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 1,
                TX_64X64 | TX_32X64 | TX_64X32 => 2,
                _ => 0,
            };
            for t in [DCT_DCT, V_DCT, H_DCT, IDTX] {
                let scan = scan_for(tx_sz, t);
                for case in 0..60 {
                    let bd = [8, 10][case % 2];
                    let (dc_q, ac_q) = (4 + rnd(5343) as u32, 4 + rnd(7309) as u32);
                    let qm = (case % 3 == 0).then(|| {
                        &QUANTIZER_MATRIX[rnd(15) as usize][rnd(2) as usize][QM_OFFSET[tx_sz]..]
                    });
                    let eob = 1 + rnd(scan.len() as u64) as usize;
                    let mut quant = Box::new([0i32; 1024]);
                    for &pos in &scan[..eob] {
                        let level = match rnd(8) {
                            0..=3 => 0,
                            4 | 5 => 1 + rnd(3) as i32,
                            6 => rnd(4000) as i32,
                            _ => rnd(1 << 20) as i32,
                        };
                        quant[pos as usize] = if rnd(2) == 0 { level } else { -level };
                    }
                    let lim = 1i32 << (7 + bd);
                    let d = Dequantiser {
                        dc_q,
                        ac_q,
                        qm,
                        shift,
                        lim,
                        tw,
                        th,
                    };
                    let mut dq = Box::new([0x1234_5678i32; 64 * 64]);
                    let (rows, cols) = d.scan(&quant, &scan[..eob], &mut dq);
                    for i in 0..th {
                        for j in 0..tw {
                            let coef = quant[i * tw + j] as i64;
                            let q = if i == 0 && j == 0 { dc_q } else { ac_q } as i64;
                            let q2 = match qm {
                                Some(m) => round2_64(q * m[i * tw + j] as i64, 5),
                                None => q,
                            };
                            let v = coef * q2;
                            let mag = (v.abs() & 0xFFFFFF) >> shift;
                            let v = if v < 0 { -mag } else { mag };
                            let want = v.clamp(-(lim as i64), lim as i64 - 1) as i32;
                            let at = format!("size {tx_sz} type {t} case {case} at ({i}, {j})");
                            assert_eq!(dq[i * 64 + j], want, "{at}");
                            assert!(coef == 0 || (i < rows && j < cols), "{at}: outside");
                        }
                    }
                    // The rows the inverse transform reads past the last.
                    for i in th..th.next_multiple_of(8) {
                        assert!(dq[i * 64..i * 64 + tw].iter().all(|&v| v == 0));
                    }
                    // The same `Dequant` a coefficient at a time, as the
                    // coefficient reader does it for those not zero.
                    let mut one = Box::new([0x0bad_cafei32; 64 * 64]);
                    let mut reach = (1, 1);
                    d.clear(&mut one);
                    for &pos in &scan[..eob] {
                        if quant[pos as usize] != 0 {
                            d.put(&mut one, pos as usize, quant[pos as usize], &mut reach);
                        }
                    }
                    for i in 0..th.next_multiple_of(8) {
                        let row = i * 64..i * 64 + tw;
                        assert_eq!(one[row.clone()], dq[row], "size {tx_sz} row {i}");
                    }
                    assert_eq!(reach, (rows, cols), "size {tx_sz} type {t} case {case}");
                }
            }
        }
    }
}
