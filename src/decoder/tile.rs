//! Tile decoding (5.11): superblocks, partitions, mode info, the CDF
//! selection of 8.3.2, and storing each block's mode info. Prediction and
//! residual decoding are in `predict` and `residual`; motion vector
//! prediction in `mvpred`.

use crate::Result;
use crate::cdf::{CdfContext, palette_color_cdf};
use crate::consts::*;
use crate::decoder::FrameCtx;
use crate::decoder::record::Pass;
use crate::decoder::state::{Mi, Mv};
use crate::symbol::{Coder, SymbolDecoder};
use crate::tables::*;

/// The block-level variables of the syntax (`MiRow`, `YMode`, `RefFrame`,
/// ...), reset for each block.
#[derive(Clone, Default)]
pub(crate) struct Block {
    pub(crate) mi_row: usize,
    pub(crate) mi_col: usize,
    pub(crate) mi_size: usize,
    pub(crate) has_chroma: bool,
    pub(crate) avail_u: bool,
    pub(crate) avail_l: bool,
    pub(crate) avail_u_chroma: bool,
    pub(crate) avail_l_chroma: bool,
    pub(crate) segment_id: usize,
    pub(crate) skip: bool,
    pub(crate) skip_mode: bool,
    pub(crate) is_inter: bool,
    pub(crate) use_intrabc: bool,
    pub(crate) lossless: bool,
    pub(crate) y_mode: usize,
    pub(crate) uv_mode: usize,
    pub(crate) angle_delta_y: i32,
    pub(crate) angle_delta_uv: i32,
    pub(crate) use_filter_intra: bool,
    pub(crate) filter_intra_mode: usize,
    pub(crate) cfl_alpha_u: i32,
    pub(crate) cfl_alpha_v: i32,
    pub(crate) palette_size_y: usize,
    pub(crate) palette_size_uv: usize,
    pub(crate) palette_colors_y: [u16; 8],
    pub(crate) palette_colors_u: [u16; 8],
    pub(crate) palette_colors_v: [u16; 8],
    pub(crate) ref_frame: [i32; 2],
    pub(crate) mv: [Mv; 2],
    pub(crate) pred_mv: [Mv; 2],
    pub(crate) ref_mv_idx: usize,
    pub(crate) interintra: bool,
    pub(crate) interintra_mode: usize,
    pub(crate) wedge_interintra: bool,
    pub(crate) wedge_index: usize,
    pub(crate) wedge_sign: usize,
    pub(crate) mask_type: usize,
    pub(crate) motion_mode: u32,
    pub(crate) compound_type: u32,
    pub(crate) comp_group_idx: u32,
    pub(crate) compound_idx: u32,
    pub(crate) interp_filter: [u32; 2],
    pub(crate) tx_size: usize,
    pub(crate) left_ref_frame: [i32; 2],
    pub(crate) above_ref_frame: [i32; 2],
    pub(crate) left_intra: bool,
    pub(crate) above_intra: bool,
    pub(crate) left_single: bool,
    pub(crate) above_single: bool,
    // Motion vector prediction (7.10.2).
    pub(crate) num_mv_found: usize,
    pub(crate) new_mv_count: usize,
    pub(crate) ref_stack_mv: [[Mv; 2]; 10],
    pub(crate) weight_stack: [u32; 10],
    pub(crate) global_mvs: [Mv; 2],
    pub(crate) found_match: bool,
    pub(crate) close_matches: usize,
    pub(crate) total_matches: usize,
    pub(crate) new_mv_context: usize,
    pub(crate) ref_mv_context: usize,
    pub(crate) zero_mv_context: usize,
    pub(crate) drl_ctx_stack: [usize; 10],
    // Warped motion (7.10.4, 7.11.3.8).
    pub(crate) num_samples: usize,
    pub(crate) num_samples_scanned: usize,
    pub(crate) cand_list: [[i32; 4]; 8],
    pub(crate) local_warp_params: [i32; 6],
    pub(crate) local_valid: bool,
    pub(crate) max_luma_w: usize,
    pub(crate) max_luma_h: usize,
    pub(crate) is_inter_intra: bool,
}

/// The encoder's decisions for one block, planted before the syntax codes
/// them (encode mode). Unused when decoding.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Plan {
    pub(crate) skip: bool,
    /// Skip mode: the block predicted from the frame's two skip mode
    /// references with their nearest vectors, and no residual.
    pub(crate) skip_mode: bool,
    pub(crate) is_inter: bool,
    pub(crate) y_mode: usize,
    pub(crate) uv_mode: usize,
    pub(crate) angle_delta_y: i32,
    pub(crate) angle_delta_uv: i32,
    pub(crate) ref_frame: i32,
    pub(crate) ref_mv_idx: usize,
    pub(crate) mv: [Mv; 2],
    /// The luma transform type, where there is a choice (greedy mode).
    pub(crate) tx_type: usize,
    /// `tx_depth` (intra) or the uniform variable-transform depth (inter)
    /// under `TX_MODE_SELECT`.
    pub(crate) tx_depth: u8,
    /// `CflAlphaU`, `CflAlphaV` (chroma from luma).
    pub(crate) cfl_alpha_u: i32,
    pub(crate) cfl_alpha_v: i32,
    /// The second reference of a compound block (`INTRA_FRAME`: none).
    pub(crate) ref_frame2: i32,
    /// A luma palette: its size (0: none) and colours, ascending.
    pub(crate) palette_n: u8,
    pub(crate) palette_colors: [u16; 8],
    /// Filter intra (`use_filter_intra`) and its mode, for a `DC_PRED`
    /// luma block of 32x32 or less without a palette.
    pub(crate) use_filter_intra: bool,
    pub(crate) filter_intra_mode: u8,
    /// Intra block copy (`use_intrabc`): the block predicted from the
    /// frame's own reconstruction, displaced by `mv[0]` (whole samples).
    pub(crate) use_intrabc: bool,
    /// `interp_filter[ dir ]`, where the frame's filter is switchable.
    pub(crate) interp_filter: [u8; 2],
    /// `motion_mode` (`SIMPLE`, `OBMC`, `LOCALWARP`), where it is coded.
    pub(crate) motion_mode: u8,
    /// Inter-intra: `interintra`, `interintra_mode`, `wedge_interintra`.
    pub(crate) interintra: bool,
    pub(crate) interintra_mode: u8,
    pub(crate) wedge_interintra: bool,
    /// `wedge_index` and `wedge_sign` (wedge compound and wedge
    /// inter-intra), `mask_type` (difference-weighted compound).
    pub(crate) wedge_index: u8,
    pub(crate) wedge_sign: u8,
    pub(crate) mask_type: u8,
    /// The compound prediction of a compound block: one of the `PLAN_*`
    /// below (average, distance-weighted, wedge, difference-weighted).
    pub(crate) compound: u8,
}

/// `Plan::compound`: the average of the two predictions (the default).
pub(crate) const PLAN_AVERAGE: u8 = 0;
/// `Plan::compound`: weighted by the references' distances (`jnt_comp`).
pub(crate) const PLAN_DISTANCE: u8 = 1;
/// `Plan::compound`: a wedge mask (`COMPOUND_WEDGE`).
pub(crate) const PLAN_WEDGE: u8 = 2;
/// `Plan::compound`: a mask from the predictions' difference
/// (`COMPOUND_DIFFWTD`).
pub(crate) const PLAN_DIFFWTD: u8 = 3;

/// The decoder of one tile: the symbol decoder, the tile's CDFs, the
/// above/left contexts, and the block being decoded.
pub(crate) struct TileDecoder<'a, 'b> {
    pub(crate) f: &'a mut FrameCtx,
    pub(crate) sd: Coder<'b>,
    /// The encoder's decisions for the block being coded (encode mode).
    pub(crate) plan: Plan,
    /// The encoder's state (encode mode only).
    pub(crate) enc: Option<Box<crate::encoder::tile::EncCtx>>,
    /// The transform type the encoder chose for the luma block being coded.
    pub(crate) enc_tx_type: usize,
    pub(crate) cdf: Box<CdfContext>,
    pub(crate) intra_frame_y_mode_cdf: [[[u16; 14]; 5]; 5],
    pub(crate) mi_row_start: usize,
    pub(crate) mi_row_end: usize,
    pub(crate) mi_col_start: usize,
    pub(crate) mi_col_end: usize,
    pub(crate) current_q_index: i32,
    pub(crate) delta_lf: [i32; 4],
    pub(crate) read_deltas: bool,
    pub(crate) ref_sgr_xqd: [[i32; 2]; 3],
    pub(crate) ref_lr_wiener: [[[i32; 3]; 2]; 3],
    pub(crate) above_level_ctx: [Vec<u8>; 3],
    pub(crate) above_dc_ctx: [Vec<u8>; 3],
    pub(crate) above_seg_pred_ctx: Vec<u8>,
    pub(crate) left_level_ctx: [Vec<u8>; 3],
    pub(crate) left_dc_ctx: [Vec<u8>; 3],
    pub(crate) left_seg_pred_ctx: Vec<u8>,
    /// `BlockDecoded[ plane ][ y ][ x ]` for x, y in -1..=32, offset by 1.
    pub(crate) block_decoded: [[[bool; 34]; 34]; 3],
    pub(crate) b: Block,
    pub(crate) color_map_y: Box<[[u8; 64]; 64]>,
    pub(crate) color_map_uv: Box<[[u8; 64]; 64]>,
    // Coefficient decoding.
    pub(crate) quant: Box<[i32; 1024]>,
    /// The decode-only coefficient reader's map of the levels read so far
    /// in the transform block (`Min( level, 3 )` each, rows of 36), with
    /// zero columns to the right of each row and four zero rows below, so
    /// that a context neighbour past the block's edge reads as the zero
    /// the specification gives it, without a test.
    pub(crate) levels: Box<[u8; 36 * 36]>,
    /// Which coefficient reader decodes (see `residual.rs`).
    pub(crate) coeff_reader: crate::decoder::CoefficientReader,
    pub(crate) dequant: Box<[i32; 64 * 64]>,
    pub(crate) residual: Box<[i32; 64 * 64]>,
    pub(crate) plane_tx_type: usize,
    /// Inter prediction's `preds`, kept between blocks.
    pub(crate) pred_bufs: [Vec<i32>; 2],
    /// Which part of decoding a block this walker does (see `record.rs`).
    pub(crate) pass: crate::decoder::record::Pass,
    /// The superblock row's record: written by a parsing walk, replayed by
    /// a reconstructing one.
    pub(crate) rec: crate::decoder::record::RowRecord,
    /// Where the replay of `rec` has got to.
    pub(crate) cursor: crate::decoder::record::Cursor,
    /// For a reconstructing walk of one superblock row whose planes hold
    /// that row's samples alone: each plane's first row of the superblock
    /// row, and the row of samples above it (a copy the walk was handed:
    /// the row itself belongs to whoever works on the superblock row
    /// above). Intra prediction's edge is all that reads above the
    /// superblock row. Zero and empty otherwise: every row is in the plane.
    pub(crate) band_top: [usize; 3],
    pub(crate) above_line: [Vec<u16>; 3],
}

impl<'a, 'b> TileDecoder<'a, 'b> {
    pub(crate) fn new(
        f: &'a mut FrameCtx,
        data: &'b [u8],
        tile_row: usize,
        tile_col: usize,
    ) -> Self {
        let sd = Coder::Dec(SymbolDecoder::new(data, f.hdr.disable_cdf_update));
        Self::with_coder(f, sd, tile_row, tile_col)
    }

    /// A tile walker that encodes the encoder's decisions (`enc`).
    pub(crate) fn new_encoder(
        f: &'a mut FrameCtx,
        enc: Box<crate::encoder::tile::EncCtx>,
        tile_row: usize,
        tile_col: usize,
    ) -> Self {
        let sd = Coder::Enc(crate::symbol::SymbolEncoder::new(f.hdr.disable_cdf_update));
        let mut t = Self::with_coder(f, sd, tile_row, tile_col);
        t.enc = Some(enc);
        t
    }

    fn with_coder(f: &'a mut FrameCtx, sd: Coder<'b>, tile_row: usize, tile_col: usize) -> Self {
        let ti = &f.hdr.tile_info;
        let mi_row_start = ti.mi_row_starts[tile_row];
        let mi_row_end = ti.mi_row_starts[tile_row + 1];
        let mi_col_start = ti.mi_col_starts[tile_col];
        let mi_col_end = ti.mi_col_starts[tile_col + 1];
        let cdf = f.cdfs.clone();
        let cols = f.mi_cols + 64;
        let rows = f.mi_rows + 64;
        let current_q_index = f.hdr.base_q_idx as i32;
        TileDecoder {
            f,
            sd,
            plan: Plan::default(),
            enc: None,
            enc_tx_type: 0,
            cdf,
            intra_frame_y_mode_cdf: DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            mi_row_start,
            mi_row_end,
            mi_col_start,
            mi_col_end,
            current_q_index,
            delta_lf: [0; 4],
            read_deltas: false,
            ref_sgr_xqd: [[0; 2]; 3],
            ref_lr_wiener: [[[0; 3]; 2]; 3],
            above_level_ctx: [vec![0; cols], vec![0; cols], vec![0; cols]],
            above_dc_ctx: [vec![0; cols], vec![0; cols], vec![0; cols]],
            above_seg_pred_ctx: vec![0; cols],
            left_level_ctx: [vec![0; rows], vec![0; rows], vec![0; rows]],
            left_dc_ctx: [vec![0; rows], vec![0; rows], vec![0; rows]],
            left_seg_pred_ctx: vec![0; rows],
            block_decoded: [[[false; 34]; 34]; 3],
            b: Block::default(),
            color_map_y: Box::new([[0; 64]; 64]),
            color_map_uv: Box::new([[0; 64]; 64]),
            quant: Box::new([0; 1024]),
            levels: Box::new([0; 36 * 36]),
            coeff_reader: crate::decoder::CoefficientReader::Fast,
            dequant: Box::new([0; 64 * 64]),
            residual: Box::new([0; 64 * 64]),
            plane_tx_type: 0,
            pred_bufs: [Vec::new(), Vec::new()],
            pass: crate::decoder::record::Pass::Both,
            rec: Default::default(),
            cursor: Default::default(),
            band_top: [0; 3],
            above_line: Default::default(),
        }
    }

    #[inline]
    pub(crate) fn mi(&self, r: usize, c: usize) -> &Mi {
        &self.f.mi[r * self.f.ms + c]
    }

    /// `is_inside( r, c )`.
    #[inline]
    pub(crate) fn is_inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    fn ssx(&self) -> usize {
        self.f.ssx
    }

    fn ssy(&self) -> usize {
        self.f.ssy
    }

    /// `decode_tile()`.
    pub(crate) fn decode_tile(&mut self) -> Result<()> {
        self.begin_tile();
        let sb4 = self.sb_size4();
        let mut r = self.mi_row_start;
        while r < self.mi_row_end {
            self.begin_sb_row();
            let mut c = self.mi_col_start;
            while c < self.mi_col_end {
                self.decode_sb(r, c)?;
                c += sb4;
            }
            r += sb4;
        }
        Ok(())
    }

    /// One superblock row of `decode_tile()`, parsed alone (see
    /// `record.rs`): the superblocks of the row at `r` (in 4x4 units),
    /// into the row's record. `begin_tile` comes before the first row.
    pub(crate) fn parse_sb_row(&mut self, r: usize) -> Result<()> {
        debug_assert!(self.pass == Pass::Parse);
        self.rec.clear();
        self.begin_sb_row();
        let sb4 = self.sb_size4();
        let mut c = self.mi_col_start;
        while c < self.mi_col_end {
            self.rec.begin_sb();
            self.decode_sb(r, c)?;
            c += sb4;
        }
        self.rec.finish();
        Ok(())
    }

    /// The superblock size in 4x4 units.
    pub(crate) fn sb_size4(&self) -> usize {
        if self.f.seq.use_128x128_superblock {
            32
        } else {
            16
        }
    }

    /// The state reset at the start of a tile.
    pub(crate) fn begin_tile(&mut self) {
        for p in 0..3 {
            self.above_level_ctx[p].fill(0);
            self.above_dc_ctx[p].fill(0);
        }
        self.above_seg_pred_ctx.fill(0);
        self.delta_lf = [0; 4];
        for plane in 0..self.f.num_planes {
            for pass in 0..2 {
                self.ref_sgr_xqd[plane][pass] = SGRPROJ_XQD_MID[pass];
                for i in 0..3 {
                    self.ref_lr_wiener[plane][pass][i] = WIENER_TAPS_MID[i];
                }
            }
        }
    }

    /// `clear_left_context()`, at the start of a superblock row.
    pub(crate) fn begin_sb_row(&mut self) {
        for p in 0..3 {
            self.left_level_ctx[p].fill(0);
            self.left_dc_ctx[p].fill(0);
        }
        self.left_seg_pred_ctx.fill(0);
    }

    /// One superblock of `decode_tile()`.
    pub(crate) fn decode_sb(&mut self, r: usize, c: usize) -> Result<()> {
        let sb_size = if self.f.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        self.read_deltas = self.f.hdr.delta_q_present;
        if self.sd.encoding() {
            self.enc_begin_sb(r, c);
        }
        self.clear_cdef(r, c);
        self.clear_block_decoded_flags(r, c, NUM_4X4_BLOCKS_WIDE[sb_size]);
        self.read_lr(r, c, sb_size);
        self.decode_partition(r, c, sb_size)
    }

    pub(crate) fn clear_block_decoded_flags(&mut self, r: usize, c: usize, sb_size4: usize) {
        for plane in 0..self.f.num_planes {
            let sub_x = if plane > 0 { self.ssx() } else { 0 };
            let sub_y = if plane > 0 { self.ssy() } else { 0 };
            let sb_width4 = ((self.mi_col_end - c) >> sub_x) as isize;
            let sb_height4 = ((self.mi_row_end - r) >> sub_y) as isize;
            let bd = &mut self.block_decoded[plane];
            for y in -1..=((sb_size4 >> sub_y) as isize) {
                for x in -1..=((sb_size4 >> sub_x) as isize) {
                    let v = if y < 0 && x < sb_width4 {
                        true
                    } else {
                        x < 0 && y < sb_height4
                    };
                    bd[(y + 1) as usize][(x + 1) as usize] = v;
                }
            }
            bd[(sb_size4 >> sub_y) + 1][0] = false;
        }
    }

    /// `BlockDecoded[ plane ][ y ][ x ]`.
    #[inline]
    pub(crate) fn block_decoded(&self, plane: usize, y: isize, x: isize) -> bool {
        self.block_decoded[plane][(y + 1) as usize][(x + 1) as usize]
    }

    fn clear_cdef(&mut self, r: usize, c: usize) {
        let s = self.f.cdef_stride;
        self.f.cdef_idx[(r >> 4) * s + (c >> 4)] = -1;
        if self.f.seq.use_128x128_superblock {
            self.f.cdef_idx[(r >> 4) * s + (c >> 4) + 1] = -1;
            self.f.cdef_idx[((r >> 4) + 1) * s + (c >> 4)] = -1;
            self.f.cdef_idx[((r >> 4) + 1) * s + (c >> 4) + 1] = -1;
        }
    }

    fn read_lr(&mut self, r: usize, c: usize, b_size: usize) {
        if self.f.hdr.allow_intrabc {
            return;
        }
        for plane in 0..self.f.num_planes {
            if self.f.hdr.frame_restoration_type[plane] == RESTORE_NONE {
                continue;
            }
            let (rows, cols) =
                crate::decoder::lr_units_of_sb(self.f, plane, r, c, NUM_4X4_BLOCKS_WIDE[b_size]);
            for unit_row in rows {
                for unit_col in cols.clone() {
                    self.read_lr_unit(plane, unit_row, unit_col);
                }
            }
        }
    }

    fn read_lr_unit(&mut self, plane: usize, unit_row: usize, unit_col: usize) {
        let frt = self.f.hdr.frame_restoration_type[plane];
        let idx = unit_row * self.f.lr[plane].unit_cols + unit_col;
        // The encoder's choice for the unit (encode mode).
        let planned = self
            .enc
            .as_ref()
            .and_then(|e| e.lr_plan.as_ref())
            .and_then(|p| p.units[plane].get(idx).copied())
            .unwrap_or_default();
        let restoration_type = if frt == RESTORE_WIENER {
            let p = (planned.t == RESTORE_WIENER) as usize;
            if self.sd.symbol(&mut self.cdf.use_wiener, p) != 0 {
                RESTORE_WIENER
            } else {
                RESTORE_NONE
            }
        } else if frt == RESTORE_SGRPROJ {
            let p = (planned.t == RESTORE_SGRPROJ) as usize;
            if self.sd.symbol(&mut self.cdf.use_sgrproj, p) != 0 {
                RESTORE_SGRPROJ
            } else {
                RESTORE_NONE
            }
        } else {
            self.sd
                .symbol(&mut self.cdf.restoration_type, planned.t as usize) as u8
        };
        self.f.lr[plane].lr_type[idx] = restoration_type;
        if restoration_type == RESTORE_WIENER {
            for pass in 0..2 {
                let first_coeff = if plane != 0 {
                    self.f.lr[plane].wiener[idx][pass][0] = 0;
                    1
                } else {
                    0
                };
                for j in first_coeff..3 {
                    let min = WIENER_TAPS_MIN[j];
                    let max = WIENER_TAPS_MAX[j];
                    let k = WIENER_TAPS_K[j] as u32;
                    let v = self.decode_signed_subexp_with_ref_bool(
                        min,
                        max + 1,
                        k,
                        self.ref_lr_wiener[plane][pass][j],
                        planned.wiener[pass][j],
                    );
                    self.f.lr[plane].wiener[idx][pass][j] = v;
                    self.ref_lr_wiener[plane][pass][j] = v;
                }
            }
        } else if restoration_type == RESTORE_SGRPROJ {
            let set = self.sd.literal(SGRPROJ_PARAMS_BITS, planned.set as u32) as usize;
            self.f.lr[plane].sgr_set[idx] = set as u8;
            for i in 0..2 {
                let radius = SGR_PARAMS[set][i * 2];
                let min = SGRPROJ_XQD_MIN[i];
                let max = SGRPROJ_XQD_MAX[i];
                let v = if radius != 0 {
                    self.decode_signed_subexp_with_ref_bool(
                        min,
                        max + 1,
                        SGRPROJ_PRJ_SUBEXP_K,
                        self.ref_sgr_xqd[plane][i],
                        planned.xqd[i],
                    )
                } else if i == 1 {
                    clip3(
                        min,
                        max,
                        (1 << SGRPROJ_PRJ_BITS) - self.ref_sgr_xqd[plane][0],
                    )
                } else {
                    0
                };
                self.f.lr[plane].sgr_xqd[idx][i] = v;
                self.ref_sgr_xqd[plane][i] = v;
            }
        }
    }

    /// `decode_signed_subexp_with_ref_bool( low, high, k, r )`; codes
    /// `planned` when encoding.
    fn decode_signed_subexp_with_ref_bool(
        &mut self,
        low: i32,
        high: i32,
        k: u32,
        r: i32,
        planned: i32,
    ) -> i32 {
        let x = self.decode_unsigned_subexp_with_ref_bool(high - low, k, r - low, planned - low);
        x + low
    }

    fn decode_unsigned_subexp_with_ref_bool(
        &mut self,
        mx: i32,
        k: u32,
        r: i32,
        planned: i32,
    ) -> i32 {
        // The planned value's position in the recentred order (encode
        // mode): the inverse of inverse_recenter().
        let recenter = |r: i32, x: i32| {
            if x > 2 * r {
                x
            } else if x >= r {
                2 * (x - r)
            } else {
                2 * (r - x) - 1
            }
        };
        let pv = if (r << 1) <= mx {
            recenter(r, planned)
        } else {
            recenter(mx - 1 - r, mx - 1 - planned)
        };
        let v = self.decode_subexp_bool(mx, k, pv);
        if (r << 1) <= mx {
            crate::header::inverse_recenter(r, v)
        } else {
            mx - 1 - crate::header::inverse_recenter(mx - 1 - r, v)
        }
    }

    fn decode_subexp_bool(&mut self, num_syms: i32, k: u32, planned: i32) -> i32 {
        let mut i = 0u32;
        let mut mk = 0i32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1i32 << b2;
            if num_syms <= mk + 3 * a {
                return self
                    .sd
                    .ns((num_syms - mk) as u32, (planned - mk).max(0) as u32)
                    as i32
                    + mk;
            } else if self.sd.literal(1, (planned >= mk + a) as u32) != 0 {
                i += 1;
                mk += a;
            } else {
                return self.sd.literal(b2, (planned - mk).max(0) as u32) as i32 + mk;
            }
        }
    }

    /// `decode_partition( r, c, bSize )`.
    pub(crate) fn decode_partition(&mut self, r: usize, c: usize, b_size: usize) -> Result<()> {
        if r >= self.f.mi_rows || c >= self.f.mi_cols {
            return Ok(());
        }
        let avail_u = self.is_inside(r as isize - 1, c as isize);
        let avail_l = self.is_inside(r as isize, c as isize - 1);
        let num4x4 = NUM_4X4_BLOCKS_WIDE[b_size];
        let half_block4x4 = num4x4 >> 1;
        let quarter_block4x4 = half_block4x4 >> 1;
        let has_rows = (r + half_block4x4) < self.f.mi_rows;
        let has_cols = (c + half_block4x4) < self.f.mi_cols;
        let planned = if self.sd.encoding() && b_size >= BLOCK_8X8 && (has_rows || has_cols) {
            self.enc_partition(r, c, b_size, has_rows, has_cols)?
        } else {
            0
        };
        let partition = if b_size < BLOCK_8X8 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            let (bsl, ctx) = self.partition_ctx(r, c, b_size, avail_u, avail_l);
            let cdf = partition_cdf(&mut self.cdf, bsl, ctx);
            self.sd.symbol(cdf, planned)
        } else if has_cols {
            let (bsl, ctx) = self.partition_ctx(r, c, b_size, avail_u, avail_l);
            let pc = partition_cdf(&mut self.cdf, bsl, ctx);
            let p = |i: usize| pc[i] as i32 - if i > 0 { pc[i - 1] as i32 } else { 0 };
            let mut psum = p(PARTITION_VERT)
                + p(PARTITION_SPLIT)
                + p(PARTITION_HORZ_A)
                + p(PARTITION_VERT_A)
                + p(PARTITION_VERT_B);
            if b_size != BLOCK_128X128 {
                psum += p(PARTITION_VERT_4);
            }
            let mut cdf = [((1 << 15) - psum) as u16, 1 << 15, 0];
            if self
                .sd
                .symbol(&mut cdf, (planned == PARTITION_SPLIT) as usize)
                != 0
            {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            let (bsl, ctx) = self.partition_ctx(r, c, b_size, avail_u, avail_l);
            let pc = partition_cdf(&mut self.cdf, bsl, ctx);
            let p = |i: usize| pc[i] as i32 - if i > 0 { pc[i - 1] as i32 } else { 0 };
            let mut psum = p(PARTITION_HORZ)
                + p(PARTITION_SPLIT)
                + p(PARTITION_HORZ_A)
                + p(PARTITION_HORZ_B)
                + p(PARTITION_VERT_A);
            if b_size != BLOCK_128X128 {
                psum += p(PARTITION_HORZ_4);
            }
            let mut cdf = [((1 << 15) - psum) as u16, 1 << 15, 0];
            if self
                .sd
                .symbol(&mut cdf, (planned == PARTITION_SPLIT) as usize)
                != 0
            {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        let sub_size = PARTITION_SUBSIZE[partition][b_size];
        let split_size = PARTITION_SUBSIZE[PARTITION_SPLIT][b_size];
        let (h, q) = (half_block4x4, quarter_block4x4);
        match partition {
            PARTITION_NONE => self.decode_block(r, c, sub_size)?,
            PARTITION_HORZ => {
                self.decode_block(r, c, sub_size)?;
                if has_rows {
                    self.decode_block(r + h, c, sub_size)?;
                }
            }
            PARTITION_VERT => {
                self.decode_block(r, c, sub_size)?;
                if has_cols {
                    self.decode_block(r, c + h, sub_size)?;
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(r, c, sub_size)?;
                self.decode_partition(r, c + h, sub_size)?;
                self.decode_partition(r + h, c, sub_size)?;
                self.decode_partition(r + h, c + h, sub_size)?;
            }
            PARTITION_HORZ_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c, sub_size)?;
            }
            PARTITION_HORZ_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_VERT_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r, c + h, sub_size)?;
            }
            PARTITION_VERT_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_HORZ_4 => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + q, c, sub_size)?;
                self.decode_block(r + q * 2, c, sub_size)?;
                if r + q * 3 < self.f.mi_rows {
                    self.decode_block(r + q * 3, c, sub_size)?;
                }
            }
            _ => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + q, sub_size)?;
                self.decode_block(r, c + q * 2, sub_size)?;
                if c + q * 3 < self.f.mi_cols {
                    self.decode_block(r, c + q * 3, sub_size)?;
                }
            }
        }
        Ok(())
    }

    fn partition_ctx(
        &self,
        r: usize,
        c: usize,
        b_size: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> (usize, usize) {
        let bsl = MI_WIDTH_LOG2[b_size];
        let above = avail_u && MI_WIDTH_LOG2[self.mi(r - 1, c).mi_size as usize] < bsl;
        let left = avail_l && MI_HEIGHT_LOG2[self.mi(r, c - 1).mi_size as usize] < bsl;
        (bsl, left as usize * 2 + above as usize)
    }

    /// `decode_block( r, c, subSize )`.
    pub(crate) fn decode_block(&mut self, r: usize, c: usize, sub_size: usize) -> Result<()> {
        self.decode_block_prediction(r, c, sub_size)?;
        self.residual()?;
        let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size];
        let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size];
        let cols = self.f.ms;
        let rows = self.f.mi_rows + 32;
        for y in 0..bh4 {
            if r + y >= rows {
                break;
            }
            for x in 0..bw4 {
                if c + x >= cols {
                    break;
                }
                let b = &self.b;
                let idx = (r + y) * cols + c + x;
                let m = &mut self.f.mi[idx];
                m.is_inter = b.is_inter;
                m.skip_mode = b.skip_mode;
                m.skip = b.skip;
                m.tx_size = b.tx_size as u8;
                m.mi_size = b.mi_size as u8;
                m.segment_id = b.segment_id as u8;
                m.palette_size = [b.palette_size_y as u8, b.palette_size_uv as u8];
                for i in 0..FRAME_LF_COUNT {
                    m.delta_lf[i] = self.delta_lf[i] as i8;
                }
                self.f.segment_ids[idx] = b.segment_id as u8;
                if b.palette_size_y > 0 {
                    self.f.palette_colors[0][idx] = b.palette_colors_y;
                }
                if b.palette_size_uv > 0 {
                    self.f.palette_colors[1][idx] = b.palette_colors_u;
                }
            }
        }
        Ok(())
    }

    /// `decode_block()` up to its prediction: the block's mode info coded
    /// and stored, and the block predicted (the encoder's search measures
    /// a prediction by this alone).
    #[inline]
    pub(crate) fn decode_block_prediction(
        &mut self,
        r: usize,
        c: usize,
        sub_size: usize,
    ) -> Result<()> {
        let ssx = self.ssx();
        let ssy = self.ssy();
        self.b = Block {
            mi_row: r,
            mi_col: c,
            mi_size: sub_size,
            ..Block::default()
        };
        let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size];
        let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size];
        let b = &mut self.b;
        b.has_chroma =
            if (bh4 == 1 && ssy != 0 && (r & 1) == 0) || (bw4 == 1 && ssx != 0 && (c & 1) == 0) {
                false
            } else {
                self.f.num_planes > 1
            };
        // A block whose chroma has no size at this subsampling (a 4x16 at
        // 4:2:2, say): bitstream conformance requires
        // get_plane_residual_size(MiSize, 1) not to be BLOCK_INVALID (6.10.4).
        if self.b.has_chroma && self.f.plane_residual_size(sub_size, 1) == BLOCK_INVALID {
            return Err(crate::Error::bitstream(
                "a block size with no chroma size at this subsampling",
            ));
        }
        let (ri, ci) = (r as isize, c as isize);
        let avail_u = self.is_inside(ri - 1, ci);
        let avail_l = self.is_inside(ri, ci - 1);
        let mut avail_u_chroma = avail_u;
        let mut avail_l_chroma = avail_l;
        if self.b.has_chroma {
            if ssy != 0 && bh4 == 1 {
                avail_u_chroma = self.is_inside(ri - 2, ci);
            }
            if ssx != 0 && bw4 == 1 {
                avail_l_chroma = self.is_inside(ri, ci - 2);
            }
        } else {
            avail_u_chroma = false;
            avail_l_chroma = false;
        }
        let b = &mut self.b;
        b.avail_u = avail_u;
        b.avail_l = avail_l;
        b.avail_u_chroma = avail_u_chroma;
        b.avail_l_chroma = avail_l_chroma;
        b.ref_frame = [INTRA_FRAME, NONE];
        b.interp_filter = [0, 0];
        b.compound_type = COMPOUND_AVERAGE;
        b.compound_idx = 1;
        if self.sd.encoding() {
            self.enc_decide_block()?;
        }
        self.mode_info();
        self.palette_tokens();
        self.read_block_tx_size();
        if self.b.skip {
            self.reset_block_context(bw4, bh4);
        }
        let is_compound = self.b.ref_frame[1] > INTRA_FRAME;
        let cols = self.f.ms;
        let rows = self.f.mi_rows + 32;
        for y in 0..bh4 {
            if r + y >= rows {
                break;
            }
            for x in 0..bw4 {
                if c + x >= cols {
                    break;
                }
                let b = &self.b;
                let m = &mut self.f.mi[(r + y) * cols + c + x];
                m.y_mode = b.y_mode as u8;
                if b.ref_frame[0] == INTRA_FRAME && b.has_chroma {
                    m.uv_mode = b.uv_mode as u8;
                }
                m.ref_frame = [b.ref_frame[0] as i8, b.ref_frame[1] as i8];
                m.written = true;
                if b.is_inter {
                    if !b.use_intrabc {
                        m.comp_group_idx = b.comp_group_idx as u8;
                        m.compound_idx = b.compound_idx as u8;
                    }
                    m.interp_filter = [b.interp_filter[0] as u8, b.interp_filter[1] as u8];
                    m.mv[0] = b.mv[0];
                    if is_compound {
                        m.mv[1] = b.mv[1];
                    }
                }
            }
        }
        if self.pass == Pass::Parse {
            // The prediction is the reconstructing walk's.
            self.record_block();
        } else {
            self.compute_prediction();
        }
        Ok(())
    }

    fn reset_block_context(&mut self, bw4: usize, bh4: usize) {
        let planes = if self.b.has_chroma { 3 } else { 1 };
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        for plane in 0..planes {
            let sub_x = if plane > 0 { self.ssx() } else { 0 };
            let sub_y = if plane > 0 { self.ssy() } else { 0 };
            for i in (mi_col >> sub_x)..((mi_col + bw4) >> sub_x) {
                self.above_level_ctx[plane][i] = 0;
                self.above_dc_ctx[plane][i] = 0;
            }
            for i in (mi_row >> sub_y)..((mi_row + bh4) >> sub_y) {
                self.left_level_ctx[plane][i] = 0;
                self.left_dc_ctx[plane][i] = 0;
            }
        }
    }

    fn mode_info(&mut self) {
        if self.f.hdr.frame_is_intra {
            self.intra_frame_mode_info();
        } else {
            self.inter_frame_mode_info();
        }
    }

    fn intra_frame_mode_info(&mut self) {
        self.b.skip = false;
        if self.f.hdr.seg_id_pre_skip {
            self.intra_segment_id();
        }
        self.b.skip_mode = false;
        self.read_skip();
        if !self.f.hdr.seg_id_pre_skip {
            self.intra_segment_id();
        }
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        self.b.use_intrabc = if self.f.hdr.allow_intrabc {
            self.sd
                .symbol(&mut self.cdf.intrabc, self.plan.use_intrabc as usize)
                != 0
        } else {
            false
        };
        if self.b.use_intrabc {
            let b = &mut self.b;
            b.is_inter = true;
            b.y_mode = DC_PRED;
            b.uv_mode = DC_PRED;
            b.motion_mode = SIMPLE;
            b.compound_type = COMPOUND_AVERAGE;
            b.palette_size_y = 0;
            b.palette_size_uv = 0;
            b.interp_filter = [BILINEAR, BILINEAR];
            self.find_mv_stack(false);
            self.assign_mv(false);
        } else {
            self.b.is_inter = false;
            let above = if self.b.avail_u {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row - 1, self.b.mi_col).y_mode as usize]
            } else {
                0
            };
            let left = if self.b.avail_l {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row, self.b.mi_col - 1).y_mode as usize]
            } else {
                0
            };
            self.b.y_mode = self.sd.symbol(
                &mut self.intra_frame_y_mode_cdf[above][left],
                self.plan.y_mode,
            );
            self.intra_angle_info_y();
            if self.b.has_chroma {
                self.read_uv_mode();
                if self.b.uv_mode == UV_CFL_PRED {
                    self.read_cfl_alphas();
                }
                self.intra_angle_info_uv();
            }
            self.b.palette_size_y = 0;
            self.b.palette_size_uv = 0;
            let ms = self.b.mi_size;
            if ms >= BLOCK_8X8
                && block_width(ms) <= 64
                && block_height(ms) <= 64
                && self.f.hdr.allow_screen_content_tools
            {
                self.palette_mode_info();
            }
            self.filter_intra_mode_info();
        }
    }

    fn read_uv_mode(&mut self) {
        let ms = self.b.mi_size;
        let ssx = self.ssx();
        let ssy = self.ssy();
        let cfl_allowed = if self.b.lossless && SUBSAMPLED_SIZE[ms][ssx][ssy] == BLOCK_4X4 {
            true
        } else {
            !self.b.lossless && block_width(ms).max(block_height(ms)) <= 32
        };
        let y = self.b.y_mode;
        self.b.uv_mode = if cfl_allowed {
            self.sd
                .symbol(&mut self.cdf.uv_mode_cfl_allowed[y], self.plan.uv_mode)
        } else {
            self.sd
                .symbol(&mut self.cdf.uv_mode_cfl_not_allowed[y], self.plan.uv_mode)
        };
    }

    fn intra_segment_id(&mut self) {
        if self.f.hdr.segmentation_enabled {
            self.read_segment_id();
        } else {
            self.b.segment_id = 0;
        }
        self.b.lossless = self.f.hdr.lossless_array[self.b.segment_id];
    }

    #[allow(clippy::if_same_then_else)]
    fn read_segment_id(&mut self) {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        let cols = self.f.ms;
        let prev_ul: i32 = if self.b.avail_u && self.b.avail_l {
            self.f.segment_ids[(r - 1) * cols + c - 1] as i32
        } else {
            -1
        };
        let prev_u: i32 = if self.b.avail_u {
            self.f.segment_ids[(r - 1) * cols + c] as i32
        } else {
            -1
        };
        let prev_l: i32 = if self.b.avail_l {
            self.f.segment_ids[r * cols + c - 1] as i32
        } else {
            -1
        };
        let pred = if prev_u == -1 {
            if prev_l == -1 { 0 } else { prev_l }
        } else if prev_l == -1 {
            prev_u
        } else if prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if self.b.skip {
            self.b.segment_id = pred as usize;
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let s = self.sd.read_symbol(&mut self.cdf.segment_id[ctx]) as i32;
            let max = self.f.hdr.last_active_seg_id as i32 + 1;
            self.b.segment_id = neg_deinterleave(s, pred, max).clamp(0, 7) as usize;
        }
    }

    fn seg_feature_active(&self, feature: usize) -> bool {
        self.f.hdr.segmentation_enabled && self.f.hdr.feature_enabled[self.b.segment_id][feature]
    }

    fn read_skip_mode(&mut self) {
        let ms = self.b.mi_size;
        if self.seg_feature_active(SEG_LVL_SKIP)
            || self.seg_feature_active(SEG_LVL_REF_FRAME)
            || self.seg_feature_active(SEG_LVL_GLOBALMV)
            || !self.f.hdr.skip_mode_present
            || block_width(ms) < 8
            || block_height(ms) < 8
        {
            self.b.skip_mode = false;
        } else {
            let mut ctx = 0;
            if self.b.avail_u {
                ctx += self.mi(self.b.mi_row - 1, self.b.mi_col).skip_mode as usize;
            }
            if self.b.avail_l {
                ctx += self.mi(self.b.mi_row, self.b.mi_col - 1).skip_mode as usize;
            }
            self.b.skip_mode = self
                .sd
                .symbol(&mut self.cdf.skip_mode[ctx], self.plan.skip_mode as usize)
                != 0;
        }
    }

    fn read_skip(&mut self) {
        if self.f.hdr.seg_id_pre_skip && self.seg_feature_active(SEG_LVL_SKIP) {
            self.b.skip = true;
        } else {
            let mut ctx = 0;
            if self.b.avail_u {
                ctx += self.mi(self.b.mi_row - 1, self.b.mi_col).skip as usize;
            }
            if self.b.avail_l {
                ctx += self.mi(self.b.mi_row, self.b.mi_col - 1).skip as usize;
            }
            self.b.skip = self
                .sd
                .symbol(&mut self.cdf.skip[ctx], self.plan.skip as usize)
                != 0;
        }
    }

    fn read_cdef(&mut self) {
        let h = &self.f.hdr;
        if self.b.skip || h.coded_lossless || !self.f.seq.enable_cdef || h.allow_intrabc {
            return;
        }
        let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64];
        let cdef_mask4 = !(cdef_size4 - 1);
        let r = self.b.mi_row & cdef_mask4;
        let c = self.b.mi_col & cdef_mask4;
        let s = self.f.cdef_stride;
        if self.f.cdef_idx[(r >> 4) * s + (c >> 4)] == -1 {
            let planned = self
                .enc
                .as_ref()
                .and_then(|e| e.cdef_table.as_ref())
                .map_or(0, |t| t[(r >> 4) * s + (c >> 4)].max(0) as u32);
            let v = self.sd.literal(self.f.hdr.cdef_bits, planned) as i8;
            let w4 = NUM_4X4_BLOCKS_WIDE[self.b.mi_size];
            let h4 = NUM_4X4_BLOCKS_HIGH[self.b.mi_size];
            let mut i = r;
            while i < r + h4 {
                let mut j = c;
                while j < c + w4 {
                    self.f.cdef_idx[(i >> 4) * s + (j >> 4)] = v;
                    j += cdef_size4;
                }
                i += cdef_size4;
            }
        }
    }

    fn read_delta_qindex(&mut self) {
        let sb_size = if self.f.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        if self.b.mi_size == sb_size && self.b.skip {
            return;
        }
        if self.read_deltas && self.sd.encoding() {
            // The encoder's planned quantiser for this superblock.
            let reduced = self.enc_delta_qindex();
            let abs = reduced.unsigned_abs();
            self.sd
                .symbol(&mut self.cdf.delta_q, abs.min(DELTA_Q_SMALL) as usize);
            if abs >= DELTA_Q_SMALL {
                let rem_bits = 31 - (abs - 1).leading_zeros();
                self.sd.literal(3, rem_bits - 1);
                self.sd.literal(rem_bits, abs - 1 - (1 << rem_bits));
            }
            if abs != 0 {
                self.sd.literal(1, (reduced < 0) as u32);
                self.current_q_index = clip3(
                    1,
                    255,
                    self.current_q_index + (reduced << self.f.hdr.delta_q_res),
                );
            }
        } else if self.read_deltas {
            let mut delta_q_abs = self.sd.read_symbol(&mut self.cdf.delta_q) as i32;
            if delta_q_abs == DELTA_Q_SMALL as i32 {
                let rem_bits = self.sd.read_literal(3) + 1;
                let abs_bits = self.sd.read_literal(rem_bits) as i32;
                delta_q_abs = abs_bits + (1 << rem_bits) + 1;
            }
            if delta_q_abs != 0 {
                let sign = self.sd.read_literal(1);
                let reduced = if sign != 0 { -delta_q_abs } else { delta_q_abs };
                self.current_q_index = clip3(
                    1,
                    255,
                    self.current_q_index + (reduced << self.f.hdr.delta_q_res),
                );
            }
        }
    }

    fn read_delta_lf(&mut self) {
        let sb_size = if self.f.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        if self.b.mi_size == sb_size && self.b.skip {
            return;
        }
        if self.read_deltas && self.f.hdr.delta_lf_present {
            let mut frame_lf_count = 1;
            if self.f.hdr.delta_lf_multi {
                frame_lf_count = if self.f.num_planes > 1 {
                    FRAME_LF_COUNT
                } else {
                    FRAME_LF_COUNT - 2
                };
            }
            for i in 0..frame_lf_count {
                let delta_lf_abs = if self.f.hdr.delta_lf_multi {
                    self.sd.read_symbol(&mut self.cdf.delta_lf_multi[i])
                } else {
                    self.sd.read_symbol(&mut self.cdf.delta_lf)
                } as i32;
                let abs = if delta_lf_abs == DELTA_LF_SMALL as i32 {
                    let n = self.sd.read_literal(3) + 1;
                    let bits = self.sd.read_literal(n) as i32;
                    bits + (1 << n) + 1
                } else {
                    delta_lf_abs
                };
                if abs != 0 {
                    let sign = self.sd.read_literal(1);
                    let reduced = if sign != 0 { -abs } else { abs };
                    self.delta_lf[i] = clip3(
                        -MAX_LOOP_FILTER,
                        MAX_LOOP_FILTER,
                        self.delta_lf[i] + (reduced << self.f.hdr.delta_lf_res),
                    );
                }
            }
        }
    }

    fn intra_angle_info_y(&mut self) {
        self.b.angle_delta_y = 0;
        if self.b.mi_size >= BLOCK_8X8 && is_directional_mode(self.b.y_mode) {
            let pa = (self.plan.angle_delta_y + 3) as usize;
            let v = self
                .sd
                .symbol(&mut self.cdf.angle_delta[self.b.y_mode - V_PRED], pa)
                as i32;
            self.b.angle_delta_y = v - MAX_ANGLE_DELTA;
        }
    }

    fn intra_angle_info_uv(&mut self) {
        self.b.angle_delta_uv = 0;
        if self.b.mi_size >= BLOCK_8X8 && is_directional_mode(self.b.uv_mode) {
            let pa = (self.plan.angle_delta_uv + 3) as usize;
            let v = self
                .sd
                .symbol(&mut self.cdf.angle_delta[self.b.uv_mode - V_PRED], pa)
                as i32;
            self.b.angle_delta_uv = v - MAX_ANGLE_DELTA;
        }
    }

    fn read_cfl_alphas(&mut self) {
        // The planned alphas (encode mode): their signs, jointly, then
        // their magnitudes.
        let sign_of = |a: i32| match a.signum() {
            0 => CFL_SIGN_ZERO,
            -1 => CFL_SIGN_NEG,
            _ => CFL_SIGN_POS,
        };
        let (pu, pv) = (self.plan.cfl_alpha_u, self.plan.cfl_alpha_v);
        let planned_signs = (sign_of(pu) * 3 + sign_of(pv)).saturating_sub(1) as usize;
        let signs = self.sd.symbol(&mut self.cdf.cfl_sign, planned_signs) as u32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        if sign_u != CFL_SIGN_ZERO {
            let ctx = ((sign_u - 1) * 3 + sign_v) as usize;
            let p = (pu.unsigned_abs().max(1) - 1) as usize;
            let a = self.sd.symbol(&mut self.cdf.cfl_alpha[ctx], p) as i32 + 1;
            self.b.cfl_alpha_u = if sign_u == CFL_SIGN_NEG { -a } else { a };
        } else {
            self.b.cfl_alpha_u = 0;
        }
        if sign_v != CFL_SIGN_ZERO {
            let ctx = ((sign_v - 1) * 3 + sign_u) as usize;
            let p = (pv.unsigned_abs().max(1) - 1) as usize;
            let a = self.sd.symbol(&mut self.cdf.cfl_alpha[ctx], p) as i32 + 1;
            self.b.cfl_alpha_v = if sign_v == CFL_SIGN_NEG { -a } else { a };
        } else {
            self.b.cfl_alpha_v = 0;
        }
    }

    fn filter_intra_mode_info(&mut self) {
        self.b.use_filter_intra = false;
        let ms = self.b.mi_size;
        if self.f.seq.enable_filter_intra
            && self.b.y_mode == DC_PRED
            && self.b.palette_size_y == 0
            && block_width(ms).max(block_height(ms)) <= 32
        {
            // (Encode mode: the planned filter.)
            let planned = self.plan.use_filter_intra as usize;
            self.b.use_filter_intra = self.sd.symbol(&mut self.cdf.filter_intra[ms], planned) != 0;
            if self.b.use_filter_intra {
                let mode = self.plan.filter_intra_mode as usize;
                self.b.filter_intra_mode = self.sd.symbol(&mut self.cdf.filter_intra_mode, mode);
            }
        }
    }

    fn palette_mode_info(&mut self) {
        let ms = self.b.mi_size;
        let bsize_ctx = MI_WIDTH_LOG2[ms] + MI_HEIGHT_LOG2[ms] - 2;
        let bit_depth = self.f.bit_depth;
        if self.b.y_mode == DC_PRED {
            let mut ctx = 0;
            if self.b.avail_u && self.mi(self.b.mi_row - 1, self.b.mi_col).palette_size[0] > 0 {
                ctx += 1;
            }
            if self.b.avail_l && self.mi(self.b.mi_row, self.b.mi_col - 1).palette_size[0] > 0 {
                ctx += 1;
            }
            // The encoder's palette (encode mode): coded without the cache,
            // the first colour then the differences.
            let pn = self.plan.palette_n as usize;
            let pc = self.plan.palette_colors;
            let has_palette_y = self.sd.symbol(
                &mut self.cdf.palette_y_mode[bsize_ctx][ctx],
                (pn >= 2) as usize,
            ) != 0;
            if has_palette_y {
                let n = self.sd.symbol(
                    &mut self.cdf.palette_y_size[bsize_ctx],
                    pn.saturating_sub(2),
                ) + 2;
                self.b.palette_size_y = n;
                let cache = self.get_palette_cache(0);
                let mut colors = [0u16; 8];
                let mut idx = 0;
                let mut i = 0;
                while i < cache.len() && idx < n {
                    if self.sd.literal(1, 0) != 0 {
                        colors[idx] = cache[i];
                        idx += 1;
                    }
                    i += 1;
                }
                if idx < n {
                    colors[idx] = self.sd.literal(bit_depth, pc[0] as u32) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < n {
                    let min_bits = bit_depth - 3;
                    let need = (1..pn.max(1))
                        .map(|k| crate::bits::ceil_log2((pc[k] - pc[k - 1]) as u32))
                        .max()
                        .unwrap_or(0);
                    let extra = need.saturating_sub(min_bits).min(3);
                    palette_bits = min_bits + self.sd.literal(2, extra);
                }
                while idx < n {
                    let planned = (pc[idx.min(7)] as i32
                        - pc[idx.saturating_sub(1).min(7)] as i32
                        - 1)
                    .max(0) as u32;
                    let delta = self.sd.literal(palette_bits, planned) as i32 + 1;
                    let v = (colors[idx - 1] as i32 + delta).min((1 << bit_depth) - 1);
                    colors[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v - 1;
                    palette_bits = palette_bits.min(crate::bits::ceil_log2(range.max(0) as u32));
                    idx += 1;
                }
                colors[..n].sort_unstable();
                self.b.palette_colors_y = colors;
            }
        }
        if self.b.has_chroma && self.b.uv_mode == DC_PRED {
            let ctx = (self.b.palette_size_y > 0) as usize;
            let has_palette_uv = self.sd.symbol(&mut self.cdf.palette_uv_mode[ctx], 0) != 0;
            if has_palette_uv {
                let n = self
                    .sd
                    .read_symbol(&mut self.cdf.palette_uv_size[bsize_ctx])
                    + 2;
                self.b.palette_size_uv = n;
                let cache = self.get_palette_cache(1);
                let mut colors = [0u16; 8];
                let mut idx = 0;
                let mut i = 0;
                while i < cache.len() && idx < n {
                    if self.sd.read_literal(1) != 0 {
                        colors[idx] = cache[i];
                        idx += 1;
                    }
                    i += 1;
                }
                if idx < n {
                    colors[idx] = self.sd.read_literal(bit_depth) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < n {
                    let min_bits = bit_depth - 3;
                    palette_bits = min_bits + self.sd.read_literal(2);
                }
                while idx < n {
                    let delta = self.sd.read_literal(palette_bits) as i32;
                    let v = (colors[idx - 1] as i32 + delta).min((1 << bit_depth) - 1);
                    colors[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v;
                    palette_bits = palette_bits.min(crate::bits::ceil_log2(range.max(0) as u32));
                    idx += 1;
                }
                colors[..n].sort_unstable();
                self.b.palette_colors_u = colors;
                let mut v_colors = [0u16; 8];
                if self.sd.read_literal(1) != 0 {
                    let min_bits = bit_depth - 4;
                    let max_val = 1i32 << bit_depth;
                    let bits = min_bits + self.sd.read_literal(2);
                    v_colors[0] = self.sd.read_literal(bit_depth) as u16;
                    for idx in 1..n {
                        let mut delta = self.sd.read_literal(bits) as i32;
                        if delta != 0 && self.sd.read_literal(1) != 0 {
                            delta = -delta;
                        }
                        let mut val = v_colors[idx - 1] as i32 + delta;
                        if val < 0 {
                            val += max_val;
                        }
                        if val >= max_val {
                            val -= max_val;
                        }
                        v_colors[idx] = val.clamp(0, (1 << bit_depth) - 1) as u16;
                    }
                } else {
                    for c in v_colors.iter_mut().take(n) {
                        *c = self.sd.read_literal(bit_depth) as u16;
                    }
                }
                self.b.palette_colors_v = v_colors;
            }
        }
    }

    fn get_palette_cache(&self, plane: usize) -> Vec<u16> {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        let cols = self.f.ms;
        let above_n = if !(r * MI_SIZE).is_multiple_of(64) {
            self.mi(r - 1, c).palette_size[plane] as usize
        } else {
            0
        };
        let left_n = if self.b.avail_l {
            self.mi(r, c - 1).palette_size[plane] as usize
        } else {
            0
        };
        let above = if above_n > 0 {
            self.f.palette_colors[plane][(r - 1) * cols + c]
        } else {
            [0; 8]
        };
        let left = if left_n > 0 {
            self.f.palette_colors[plane][r * cols + c - 1]
        } else {
            [0; 8]
        };
        let (mut ai, mut li) = (0, 0);
        let mut cache: Vec<u16> = Vec::with_capacity(16);
        while ai < above_n && li < left_n {
            let a = above[ai];
            let l = left[li];
            if l < a {
                if cache.last() != Some(&l) {
                    cache.push(l);
                }
                li += 1;
            } else {
                if cache.last() != Some(&a) {
                    cache.push(a);
                }
                ai += 1;
                if l == a {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            let v = above[ai];
            ai += 1;
            if cache.last() != Some(&v) {
                cache.push(v);
            }
        }
        while li < left_n {
            let v = left[li];
            li += 1;
            if cache.last() != Some(&v) {
                cache.push(v);
            }
        }
        cache
    }

    fn palette_tokens(&mut self) {
        let ms = self.b.mi_size;
        let mut block_height = block_height(ms);
        let mut block_width = block_width(ms);
        let mut onscreen_height = block_height.min((self.f.mi_rows - self.b.mi_row) * MI_SIZE);
        let mut onscreen_width = block_width.min((self.f.mi_cols - self.b.mi_col) * MI_SIZE);
        if self.b.palette_size_y > 0 {
            let n = self.b.palette_size_y;
            if self.sd.encoding() {
                self.enc_palette_map(onscreen_width, onscreen_height);
            }
            let p0 = self.enc.as_ref().map_or(0, |e| e.palette_map[0][0]) as u32;
            let v = self.sd.ns(n as u32, p0) as u8;
            self.color_map_y[0][0] = v;
            self.read_color_map(
                true,
                n,
                onscreen_width,
                onscreen_height,
                block_width,
                block_height,
            );
        }
        if self.b.palette_size_uv > 0 {
            let n = self.b.palette_size_uv;
            let v = self.sd.read_ns(n as u32) as u8;
            self.color_map_uv[0][0] = v;
            block_height >>= self.ssy();
            block_width >>= self.ssx();
            onscreen_height >>= self.ssy();
            onscreen_width >>= self.ssx();
            if block_width < 4 {
                block_width += 2;
                onscreen_width += 2;
            }
            if block_height < 4 {
                block_height += 2;
                onscreen_height += 2;
            }
            self.read_color_map(
                false,
                n,
                onscreen_width,
                onscreen_height,
                block_width,
                block_height,
            );
        }
    }

    fn read_color_map(
        &mut self,
        luma: bool,
        n: usize,
        onscreen_width: usize,
        onscreen_height: usize,
        block_width: usize,
        block_height: usize,
    ) {
        for i in 1..(onscreen_height + onscreen_width - 1) {
            let j_hi = i.min(onscreen_width - 1) as isize;
            let j_lo = (i as isize - onscreen_height as isize + 1).max(0);
            let mut j = j_hi;
            while j >= j_lo {
                let ju = j as usize;
                let (order, hash) = {
                    let map = if luma {
                        &self.color_map_y
                    } else {
                        &self.color_map_uv
                    };
                    palette_color_context(map, i - ju, ju, n)
                };
                let ctx = PALETTE_COLOR_CONTEXT[hash as usize] as usize;
                let rows = if luma {
                    &mut self.cdf.palette_y_color
                } else {
                    &mut self.cdf.palette_uv_color
                };
                // The planned index's position in the context's order
                // (encode mode).
                let planned = if luma {
                    self.enc.as_ref().map_or(0, |e| {
                        let want = e.palette_map[i - ju][ju];
                        order.iter().position(|&o| o == want).unwrap_or(0)
                    })
                } else {
                    0
                };
                let s = self.sd.symbol(palette_color_cdf(rows, n, ctx), planned);
                let map = if luma {
                    &mut self.color_map_y
                } else {
                    &mut self.color_map_uv
                };
                map[i - ju][ju] = order[s];
                j -= 1;
            }
        }
        let map = if luma {
            &mut self.color_map_y
        } else {
            &mut self.color_map_uv
        };
        for i in 0..onscreen_height {
            for j in onscreen_width..block_width {
                map[i][j] = map[i][onscreen_width - 1];
            }
        }
        for i in onscreen_height..block_height {
            for j in 0..block_width {
                map[i][j] = map[onscreen_height - 1][j];
            }
        }
    }

    fn read_block_tx_size(&mut self) {
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms];
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms];
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        if self.f.hdr.tx_mode == TX_MODE_SELECT
            && ms > BLOCK_4X4
            && self.b.is_inter
            && !self.b.skip
            && !self.b.lossless
        {
            let max_tx_sz = MAX_TX_SIZE_RECT[ms];
            let tx_w4 = TX_WIDTH[max_tx_sz] / MI_SIZE;
            let tx_h4 = TX_HEIGHT[max_tx_sz] / MI_SIZE;
            let mut row = mi_row;
            while row < mi_row + bh4 {
                let mut col = mi_col;
                while col < mi_col + bw4 {
                    self.read_var_tx_size(row, col, max_tx_sz, 0);
                    col += tx_w4;
                }
                row += tx_h4;
            }
        } else {
            self.read_tx_size(!self.b.skip || !self.b.is_inter);
            let cols = self.f.ms;
            for row in mi_row..mi_row + bh4 {
                for col in mi_col..mi_col + bw4 {
                    self.f.mi[row * cols + col].inter_tx_size = self.b.tx_size as u8;
                }
            }
        }
    }

    fn read_var_tx_size(&mut self, row: usize, col: usize, tx_sz: usize, depth: usize) {
        if row >= self.f.mi_rows || col >= self.f.mi_cols {
            return;
        }
        let txfm_split = if tx_sz == TX_4X4 || depth == MAX_VARTX_DEPTH {
            false
        } else {
            let ctx = self.txfm_split_ctx(row, col, tx_sz);
            let planned = depth < self.plan.tx_depth as usize;
            self.sd
                .symbol(&mut self.cdf.txfm_split[ctx], planned as usize)
                != 0
        };
        let w4 = TX_WIDTH[tx_sz] / MI_SIZE;
        let h4 = TX_HEIGHT[tx_sz] / MI_SIZE;
        if txfm_split {
            let sub = SPLIT_TX_SIZE[tx_sz];
            let step_w = TX_WIDTH[sub] / MI_SIZE;
            let step_h = TX_HEIGHT[sub] / MI_SIZE;
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(row + i, col + j, sub, depth + 1);
                    j += step_w;
                }
                i += step_h;
            }
        } else {
            let cols = self.f.ms;
            for i in 0..h4 {
                for j in 0..w4 {
                    if row + i < self.f.mi_rows + 32 && col + j < cols {
                        self.f.mi[(row + i) * cols + col + j].inter_tx_size = tx_sz as u8;
                    }
                }
            }
            self.b.tx_size = tx_sz;
        }
    }

    fn get_above_tx_width(&self, row: usize, col: usize) -> usize {
        if row == self.b.mi_row {
            if !self.b.avail_u {
                return 64;
            }
            let m = self.mi(row - 1, col);
            if m.skip && m.is_inter {
                return block_width(m.mi_size as usize);
            }
        }
        TX_WIDTH[self.mi(row - 1, col).inter_tx_size as usize]
    }

    fn get_left_tx_height(&self, row: usize, col: usize) -> usize {
        if col == self.b.mi_col {
            if !self.b.avail_l {
                return 64;
            }
            let m = self.mi(row, col - 1);
            if m.skip && m.is_inter {
                return block_height(m.mi_size as usize);
            }
        }
        TX_HEIGHT[self.mi(row, col - 1).inter_tx_size as usize]
    }

    fn txfm_split_ctx(&self, row: usize, col: usize, tx_sz: usize) -> usize {
        let above = (self.get_above_tx_width(row, col) < TX_WIDTH[tx_sz]) as usize;
        let left = (self.get_left_tx_height(row, col) < TX_HEIGHT[tx_sz]) as usize;
        let ms = self.b.mi_size;
        let size = 64.min(block_width(ms).max(block_height(ms)));
        let max_tx_sz = find_tx_size(size, size);
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_sz];
        (tx_sz_sqr_up != max_tx_sz) as usize * 3 + (TX_SIZES - 1 - max_tx_sz) * 6 + above + left
    }

    fn read_tx_size(&mut self, allow_select: bool) {
        if self.b.lossless {
            self.b.tx_size = TX_4X4;
            return;
        }
        let ms = self.b.mi_size;
        let max_rect_tx_size = MAX_TX_SIZE_RECT[ms];
        let max_tx_depth = MAX_TX_DEPTH[ms];
        self.b.tx_size = max_rect_tx_size;
        if ms > BLOCK_4X4 && allow_select && self.f.hdr.tx_mode == TX_MODE_SELECT {
            let max_tx_w = TX_WIDTH[max_rect_tx_size];
            let max_tx_h = TX_HEIGHT[max_rect_tx_size];
            let (r, c) = (self.b.mi_row, self.b.mi_col);
            let above_w = if self.b.avail_u && self.mi(r - 1, c).is_inter {
                block_width(self.mi(r - 1, c).mi_size as usize)
            } else if self.b.avail_u {
                self.get_above_tx_width(r, c)
            } else {
                0
            };
            let left_h = if self.b.avail_l && self.mi(r, c - 1).is_inter {
                block_height(self.mi(r, c - 1).mi_size as usize)
            } else if self.b.avail_l {
                self.get_left_tx_height(r, c)
            } else {
                0
            };
            let ctx = (above_w >= max_tx_w) as usize + (left_h >= max_tx_h) as usize;
            let p = self.plan.tx_depth as usize;
            let tx_depth = match max_tx_depth {
                4 => self.sd.symbol(&mut self.cdf.tx_64x64[ctx], p),
                3 => self.sd.symbol(&mut self.cdf.tx_32x32[ctx], p),
                2 => self.sd.symbol(&mut self.cdf.tx_16x16[ctx], p),
                _ => self.sd.symbol(&mut self.cdf.tx_8x8[ctx], p.min(1)),
            };
            for _ in 0..tx_depth {
                self.b.tx_size = SPLIT_TX_SIZE[self.b.tx_size];
            }
        }
    }

    fn inter_frame_mode_info(&mut self) {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        self.b.use_intrabc = false;
        let b = &self.b;
        let left = if b.avail_l {
            Some(*self.mi(r, c - 1))
        } else {
            None
        };
        let above = if b.avail_u {
            Some(*self.mi(r - 1, c))
        } else {
            None
        };
        let b = &mut self.b;
        b.left_ref_frame[0] = left.map_or(INTRA_FRAME, |m| m.ref_frame[0] as i32);
        b.above_ref_frame[0] = above.map_or(INTRA_FRAME, |m| m.ref_frame[0] as i32);
        b.left_ref_frame[1] = left.map_or(NONE, |m| m.ref_frame[1] as i32);
        b.above_ref_frame[1] = above.map_or(NONE, |m| m.ref_frame[1] as i32);
        b.left_intra = b.left_ref_frame[0] <= INTRA_FRAME;
        b.above_intra = b.above_ref_frame[0] <= INTRA_FRAME;
        b.left_single = b.left_ref_frame[1] <= INTRA_FRAME;
        b.above_single = b.above_ref_frame[1] <= INTRA_FRAME;
        b.skip = false;
        self.inter_segment_id(true);
        self.read_skip_mode();
        if self.b.skip_mode {
            self.b.skip = true;
        } else {
            self.read_skip();
        }
        if !self.f.hdr.seg_id_pre_skip {
            self.inter_segment_id(false);
        }
        self.b.lossless = self.f.hdr.lossless_array[self.b.segment_id];
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.read_is_inter();
        if self.b.is_inter {
            self.inter_block_mode_info();
        } else {
            self.intra_block_mode_info();
        }
    }

    fn inter_segment_id(&mut self, pre_skip: bool) {
        if !self.f.hdr.segmentation_enabled {
            self.b.segment_id = 0;
            return;
        }
        let predicted = self.get_segment_id();
        if self.f.hdr.segmentation_update_map {
            if pre_skip && !self.f.hdr.seg_id_pre_skip {
                self.b.segment_id = 0;
                return;
            }
            if !pre_skip && self.b.skip {
                self.set_seg_pred_ctx(0);
                self.read_segment_id();
                return;
            }
            if self.f.hdr.segmentation_temporal_update {
                let ctx = self.left_seg_pred_ctx[self.b.mi_row] as usize
                    + self.above_seg_pred_ctx[self.b.mi_col] as usize;
                let seg_id_predicted =
                    self.sd.read_symbol(&mut self.cdf.segment_id_predicted[ctx]) as u8;
                if seg_id_predicted != 0 {
                    self.b.segment_id = predicted;
                } else {
                    self.read_segment_id();
                }
                self.set_seg_pred_ctx(seg_id_predicted);
            } else {
                self.read_segment_id();
            }
        } else {
            self.b.segment_id = predicted;
        }
    }

    fn set_seg_pred_ctx(&mut self, v: u8) {
        let ms = self.b.mi_size;
        for i in 0..NUM_4X4_BLOCKS_WIDE[ms] {
            self.above_seg_pred_ctx[self.b.mi_col + i] = v;
        }
        for i in 0..NUM_4X4_BLOCKS_HIGH[ms] {
            self.left_seg_pred_ctx[self.b.mi_row + i] = v;
        }
    }

    fn get_segment_id(&self) -> usize {
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms];
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms];
        let x_mis = (self.f.mi_cols - self.b.mi_col).min(bw4);
        let y_mis = (self.f.mi_rows - self.b.mi_row).min(bh4);
        let mut seg = 7u8;
        for y in 0..y_mis {
            for x in 0..x_mis {
                seg = seg.min(
                    self.f.prev_segment_ids[(self.b.mi_row + y) * self.f.ms + self.b.mi_col + x],
                );
            }
        }
        seg as usize
    }

    fn read_is_inter(&mut self) {
        if self.b.skip_mode {
            self.b.is_inter = true;
        } else if self.seg_feature_active(SEG_LVL_REF_FRAME) {
            self.b.is_inter =
                self.f.hdr.feature_data[self.b.segment_id][SEG_LVL_REF_FRAME] != INTRA_FRAME;
        } else if self.seg_feature_active(SEG_LVL_GLOBALMV) {
            self.b.is_inter = true;
        } else {
            let b = &self.b;
            let ctx = if b.avail_u && b.avail_l {
                if b.left_intra && b.above_intra {
                    3
                } else {
                    (b.left_intra || b.above_intra) as usize
                }
            } else if b.avail_u || b.avail_l {
                2 * (if b.avail_u {
                    b.above_intra
                } else {
                    b.left_intra
                }) as usize
            } else {
                0
            };
            self.b.is_inter = self
                .sd
                .symbol(&mut self.cdf.is_inter[ctx], self.plan.is_inter as usize)
                != 0;
        }
    }

    fn intra_block_mode_info(&mut self) {
        self.b.ref_frame = [INTRA_FRAME, NONE];
        let ctx = SIZE_GROUP[self.b.mi_size];
        self.b.y_mode = self.sd.symbol(&mut self.cdf.y_mode[ctx], self.plan.y_mode);
        self.intra_angle_info_y();
        if self.b.has_chroma {
            self.read_uv_mode();
            if self.b.uv_mode == UV_CFL_PRED {
                self.read_cfl_alphas();
            }
            self.intra_angle_info_uv();
        }
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        let ms = self.b.mi_size;
        if ms >= BLOCK_8X8
            && block_width(ms) <= 64
            && block_height(ms) <= 64
            && self.f.hdr.allow_screen_content_tools
        {
            self.palette_mode_info();
        }
        self.filter_intra_mode_info();
    }

    fn inter_block_mode_info(&mut self) {
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        self.read_ref_frames();
        let is_compound = self.b.ref_frame[1] > INTRA_FRAME;
        self.find_mv_stack(is_compound);
        if self.b.skip_mode {
            self.b.y_mode = NEAREST_NEARESTMV;
        } else if self.seg_feature_active(SEG_LVL_SKIP) || self.seg_feature_active(SEG_LVL_GLOBALMV)
        {
            self.b.y_mode = GLOBALMV;
        } else if is_compound {
            let ctx = COMPOUND_MODE_CTX_MAP[self.b.ref_mv_context >> 1]
                [self.b.new_mv_context.min(COMP_NEWMV_CTXS - 1)];
            let p = self.plan.y_mode.saturating_sub(NEAREST_NEARESTMV);
            self.b.y_mode = NEAREST_NEARESTMV + self.sd.symbol(&mut self.cdf.compound_mode[ctx], p);
        } else {
            let py = self.plan.y_mode;
            let new_mv = self.sd.symbol(
                &mut self.cdf.new_mv[self.b.new_mv_context],
                (py != NEWMV) as usize,
            );
            if new_mv == 0 {
                self.b.y_mode = NEWMV;
            } else {
                let zero_mv = self.sd.symbol(
                    &mut self.cdf.zero_mv[self.b.zero_mv_context],
                    (py != GLOBALMV) as usize,
                );
                if zero_mv == 0 {
                    self.b.y_mode = GLOBALMV;
                } else {
                    let ref_mv = self.sd.symbol(
                        &mut self.cdf.ref_mv[self.b.ref_mv_context],
                        (py == NEARMV) as usize,
                    );
                    self.b.y_mode = if ref_mv == 0 { NEARESTMV } else { NEARMV };
                }
            }
        }
        self.b.ref_mv_idx = 0;
        let y = self.b.y_mode;
        if y == NEWMV || y == NEW_NEWMV {
            for idx in 0..2 {
                if self.b.num_mv_found > idx + 1 {
                    let ctx = self.b.drl_ctx_stack[idx];
                    let drl_mode = self.sd.symbol(
                        &mut self.cdf.drl_mode[ctx],
                        (idx != self.plan.ref_mv_idx) as usize,
                    );
                    if drl_mode == 0 {
                        self.b.ref_mv_idx = idx;
                        break;
                    }
                    self.b.ref_mv_idx = idx + 1;
                }
            }
        } else if has_nearmv(y) {
            self.b.ref_mv_idx = 1;
            for idx in 1..3 {
                if self.b.num_mv_found > idx + 1 {
                    let ctx = self.b.drl_ctx_stack[idx];
                    let drl_mode = self.sd.symbol(
                        &mut self.cdf.drl_mode[ctx],
                        (idx != self.plan.ref_mv_idx) as usize,
                    );
                    if drl_mode == 0 {
                        self.b.ref_mv_idx = idx;
                        break;
                    }
                    self.b.ref_mv_idx = idx + 1;
                }
            }
        }
        self.assign_mv(is_compound);
        self.read_interintra_mode(is_compound);
        self.read_motion_mode(is_compound);
        self.read_compound_type(is_compound);
        if self.f.hdr.interpolation_filter == SWITCHABLE {
            let dirs = if self.f.seq.enable_dual_filter { 2 } else { 1 };
            for dir in 0..dirs {
                if self.needs_interp_filter() {
                    let ctx = self.interp_filter_ctx(dir);
                    let planned = self.plan.interp_filter[dir] as usize;
                    self.b.interp_filter[dir] =
                        self.sd.symbol(&mut self.cdf.interp_filter[ctx], planned) as u32;
                } else {
                    self.b.interp_filter[dir] = EIGHTTAP;
                }
            }
            if !self.f.seq.enable_dual_filter {
                self.b.interp_filter[1] = self.b.interp_filter[0];
            }
        } else {
            self.b.interp_filter = [self.f.hdr.interpolation_filter; 2];
        }
    }

    fn needs_interp_filter(&self) -> bool {
        let ms = self.b.mi_size;
        let large = block_width(ms).min(block_height(ms)) >= 8;
        if self.b.skip_mode || self.b.motion_mode == LOCALWARP {
            false
        } else if large && self.b.y_mode == GLOBALMV {
            self.f.hdr.gm_type[self.b.ref_frame[0] as usize] == TRANSLATION
        } else if large && self.b.y_mode == GLOBAL_GLOBALMV {
            self.f.hdr.gm_type[self.b.ref_frame[0] as usize] == TRANSLATION
                || self.f.hdr.gm_type[self.b.ref_frame[1] as usize] == TRANSLATION
        } else {
            true
        }
    }

    pub(crate) fn interp_filter_ctx(&self, dir: usize) -> usize {
        let b = &self.b;
        let mut ctx = ((dir & 1) * 2 + (b.ref_frame[1] > INTRA_FRAME) as usize) * 4;
        let mut left_type = 3;
        let mut above_type = 3;
        if b.avail_l {
            let m = self.mi(b.mi_row, b.mi_col - 1);
            if m.ref_frame[0] as i32 == b.ref_frame[0] || m.ref_frame[1] as i32 == b.ref_frame[0] {
                left_type = m.interp_filter[dir] as usize;
            }
        }
        if b.avail_u {
            let m = self.mi(b.mi_row - 1, b.mi_col);
            if m.ref_frame[0] as i32 == b.ref_frame[0] || m.ref_frame[1] as i32 == b.ref_frame[0] {
                above_type = m.interp_filter[dir] as usize;
            }
        }
        if left_type == above_type {
            ctx += left_type;
        } else if left_type == 3 {
            ctx += above_type;
        } else if above_type == 3 {
            ctx += left_type;
        } else {
            ctx += 3;
        }
        ctx
    }

    fn count_refs(&self, frame_type: i32) -> usize {
        let b = &self.b;
        let mut c = 0;
        if b.avail_u {
            c += (b.above_ref_frame[0] == frame_type) as usize;
            c += (b.above_ref_frame[1] == frame_type) as usize;
        }
        if b.avail_l {
            c += (b.left_ref_frame[0] == frame_type) as usize;
            c += (b.left_ref_frame[1] == frame_type) as usize;
        }
        c
    }

    fn ref_ctx(&self, a: &[i32], b: &[i32]) -> usize {
        let c0: usize = a.iter().map(|&f| self.count_refs(f)).sum();
        let c1: usize = b.iter().map(|&f| self.count_refs(f)).sum();
        if c0 < c1 {
            0
        } else if c0 == c1 {
            1
        } else {
            2
        }
    }

    fn read_ref_frames(&mut self) {
        if self.b.skip_mode {
            self.b.ref_frame = self.f.hdr.skip_mode_frame;
            return;
        }
        if self.seg_feature_active(SEG_LVL_REF_FRAME) {
            self.b.ref_frame = [
                self.f.hdr.feature_data[self.b.segment_id][SEG_LVL_REF_FRAME],
                NONE,
            ];
            return;
        }
        if self.seg_feature_active(SEG_LVL_SKIP) || self.seg_feature_active(SEG_LVL_GLOBALMV) {
            self.b.ref_frame = [LAST_FRAME, NONE];
            return;
        }
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms];
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms];
        let comp_mode = if self.f.hdr.reference_select && bw4.min(bh4) >= 2 {
            let ctx = self.comp_mode_ctx();
            let p = (self.plan.ref_frame2 > INTRA_FRAME) as usize;
            self.sd.symbol(&mut self.cdf.comp_mode[ctx], p)
        } else {
            0
        };
        let ctx_single_p1 = self.ref_ctx(
            &[LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME],
            &[BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME],
        );
        let ctx_comp_ref = self.ref_ctx(&[LAST_FRAME, LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME]);
        let ctx_comp_ref_p1 = self.ref_ctx(&[LAST_FRAME], &[LAST2_FRAME]);
        let ctx_comp_ref_p2 = self.ref_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME]);
        let ctx_comp_bwdref = self.ref_ctx(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME]);
        let ctx_comp_bwdref_p1 = self.ref_ctx(&[BWDREF_FRAME], &[ALTREF2_FRAME]);
        if comp_mode == 1 {
            let ctx = self.comp_ref_type_ctx();
            // The encoder plans unidirectional pairs (LAST with LAST2,
            // LAST3 or GOLDEN) and bidirectional ones (a forward reference
            // with a backward one).
            let (r1, r2) = (self.plan.ref_frame, self.plan.ref_frame2);
            let comp_ref_type = self.sd.symbol(
                &mut self.cdf.comp_ref_type[ctx],
                if r2 >= BWDREF_FRAME && r1 < BWDREF_FRAME {
                    BIDIR_COMP_REFERENCE
                } else {
                    UNIDIR_COMP_REFERENCE
                } as usize,
            ) as u32;
            if comp_ref_type == UNIDIR_COMP_REFERENCE {
                let uni = self
                    .sd
                    .symbol(&mut self.cdf.uni_comp_ref[ctx_single_p1][0], 0);
                if uni != 0 {
                    self.b.ref_frame = [BWDREF_FRAME, ALTREF_FRAME];
                } else {
                    let ctx1 = self.ref_ctx(&[LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME]);
                    let p1 = self.sd.symbol(
                        &mut self.cdf.uni_comp_ref[ctx1][1],
                        (r2 != LAST2_FRAME) as usize,
                    );
                    if p1 != 0 {
                        let p2 = self.sd.symbol(
                            &mut self.cdf.uni_comp_ref[ctx_comp_ref_p2][2],
                            (r2 == GOLDEN_FRAME) as usize,
                        );
                        self.b.ref_frame = if p2 != 0 {
                            [LAST_FRAME, GOLDEN_FRAME]
                        } else {
                            [LAST_FRAME, LAST3_FRAME]
                        };
                    } else {
                        self.b.ref_frame = [LAST_FRAME, LAST2_FRAME];
                    }
                }
            } else {
                let comp_ref = self.sd.symbol(
                    &mut self.cdf.comp_ref[ctx_comp_ref][0],
                    (r1 >= LAST3_FRAME) as usize,
                );
                if comp_ref == 0 {
                    let p1 = self.sd.symbol(
                        &mut self.cdf.comp_ref[ctx_comp_ref_p1][1],
                        (r1 == LAST2_FRAME) as usize,
                    );
                    self.b.ref_frame[0] = if p1 != 0 { LAST2_FRAME } else { LAST_FRAME };
                } else {
                    let p2 = self.sd.symbol(
                        &mut self.cdf.comp_ref[ctx_comp_ref_p2][2],
                        (r1 == GOLDEN_FRAME) as usize,
                    );
                    self.b.ref_frame[0] = if p2 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                }
                let bwd = self.sd.symbol(
                    &mut self.cdf.comp_bwd_ref[ctx_comp_bwdref][0],
                    (r2 == ALTREF_FRAME) as usize,
                );
                if bwd == 0 {
                    let p1 = self.sd.symbol(
                        &mut self.cdf.comp_bwd_ref[ctx_comp_bwdref_p1][1],
                        (r2 == ALTREF2_FRAME) as usize,
                    );
                    self.b.ref_frame[1] = if p1 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                } else {
                    self.b.ref_frame[1] = ALTREF_FRAME;
                }
            }
        } else {
            let prf = self.plan.ref_frame;
            let p1 = self.sd.symbol(
                &mut self.cdf.single_ref[ctx_single_p1][0],
                (prf >= BWDREF_FRAME) as usize,
            );
            if p1 != 0 {
                let p2 = self.sd.symbol(
                    &mut self.cdf.single_ref[ctx_comp_bwdref][1],
                    (prf == ALTREF_FRAME) as usize,
                );
                if p2 == 0 {
                    let p6 = self.sd.symbol(
                        &mut self.cdf.single_ref[ctx_comp_bwdref_p1][5],
                        (prf == ALTREF2_FRAME) as usize,
                    );
                    self.b.ref_frame[0] = if p6 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                } else {
                    self.b.ref_frame[0] = ALTREF_FRAME;
                }
            } else {
                let p3 = self.sd.symbol(
                    &mut self.cdf.single_ref[ctx_comp_ref][2],
                    (prf >= LAST3_FRAME) as usize,
                );
                if p3 != 0 {
                    let p5 = self.sd.symbol(
                        &mut self.cdf.single_ref[ctx_comp_ref_p2][4],
                        (prf == GOLDEN_FRAME) as usize,
                    );
                    self.b.ref_frame[0] = if p5 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                } else {
                    let p4 = self.sd.symbol(
                        &mut self.cdf.single_ref[ctx_comp_ref_p1][3],
                        (prf == LAST2_FRAME) as usize,
                    );
                    self.b.ref_frame[0] = if p4 != 0 { LAST2_FRAME } else { LAST_FRAME };
                }
            }
            self.b.ref_frame[1] = NONE;
        }
    }

    fn comp_mode_ctx(&self) -> usize {
        let b = &self.b;
        let cb = |f: i32| (BWDREF_FRAME..=ALTREF_FRAME).contains(&f);
        if b.avail_u && b.avail_l {
            if b.above_single && b.left_single {
                (cb(b.above_ref_frame[0]) ^ cb(b.left_ref_frame[0])) as usize
            } else if b.above_single {
                2 + (cb(b.above_ref_frame[0]) || b.above_intra) as usize
            } else if b.left_single {
                2 + (cb(b.left_ref_frame[0]) || b.left_intra) as usize
            } else {
                4
            }
        } else if b.avail_u {
            if b.above_single {
                cb(b.above_ref_frame[0]) as usize
            } else {
                3
            }
        } else if b.avail_l {
            if b.left_single {
                cb(b.left_ref_frame[0]) as usize
            } else {
                3
            }
        } else {
            1
        }
    }

    fn comp_ref_type_ctx(&self) -> usize {
        let b = &self.b;
        let above0 = b.above_ref_frame[0];
        let above1 = b.above_ref_frame[1];
        let left0 = b.left_ref_frame[0];
        let left1 = b.left_ref_frame[1];
        let samedir = |a: i32, c: i32| (a >= BWDREF_FRAME) == (c >= BWDREF_FRAME);
        let above_comp_inter = b.avail_u && !b.above_intra && !b.above_single;
        let left_comp_inter = b.avail_l && !b.left_intra && !b.left_single;
        let above_uni_comp = above_comp_inter && samedir(above0, above1);
        let left_uni_comp = left_comp_inter && samedir(left0, left1);
        if b.avail_u && !b.above_intra && b.avail_l && !b.left_intra {
            let sd = samedir(above0, left0) as usize;
            if !above_comp_inter && !left_comp_inter {
                1 + 2 * sd
            } else if !above_comp_inter {
                if !left_uni_comp { 1 } else { 3 + sd }
            } else if !left_comp_inter {
                if !above_uni_comp { 1 } else { 3 + sd }
            } else if !above_uni_comp && !left_uni_comp {
                0
            } else if !above_uni_comp || !left_uni_comp {
                2
            } else {
                3 + ((above0 == BWDREF_FRAME) == (left0 == BWDREF_FRAME)) as usize
            }
        } else if b.avail_u && b.avail_l {
            if above_comp_inter {
                1 + 2 * above_uni_comp as usize
            } else if left_comp_inter {
                1 + 2 * left_uni_comp as usize
            } else {
                2
            }
        } else if above_comp_inter {
            4 * above_uni_comp as usize
        } else if left_comp_inter {
            4 * left_uni_comp as usize
        } else {
            2
        }
    }

    /// `assign_mv( isCompound )`.
    pub(crate) fn assign_mv(&mut self, is_compound: bool) {
        for i in 0..1 + is_compound as usize {
            let comp_mode = if self.b.use_intrabc {
                NEWMV
            } else {
                get_mode(self.b.y_mode, i)
            };
            if self.b.use_intrabc {
                self.b.pred_mv[0] = self.intrabc_pred_mv();
            } else if comp_mode == GLOBALMV {
                self.b.pred_mv[i] = self.b.global_mvs[i];
            } else {
                let mut pos = if comp_mode == NEARESTMV {
                    0
                } else {
                    self.b.ref_mv_idx
                };
                if comp_mode == NEWMV && self.b.num_mv_found <= 1 {
                    pos = 0;
                }
                self.b.pred_mv[i] = self.b.ref_stack_mv[pos][i];
            }
            if comp_mode == NEWMV {
                self.read_mv(i);
            } else {
                self.b.mv[i] = self.b.pred_mv[i];
            }
        }
    }

    /// `PredMv[ 0 ]` of an intra block copy (assign_mv, 5.11.26): the
    /// first non-zero vector of the stack's first two, else a default one
    /// superblock up (or left, in the tile's first superblock row).
    pub(crate) fn intrabc_pred_mv(&self) -> Mv {
        let mut pred = self.b.ref_stack_mv[0][0];
        if pred == [0, 0] {
            pred = self.b.ref_stack_mv[1][0];
        }
        if pred == [0, 0] {
            let sb_size = if self.f.seq.use_128x128_superblock {
                BLOCK_128X128
            } else {
                BLOCK_64X64
            };
            let sb_size4 = NUM_4X4_BLOCKS_HIGH[sb_size] as i32;
            pred = if (self.b.mi_row as i32) - sb_size4 < self.mi_row_start as i32 {
                [0, -(sb_size4 * MI_SIZE as i32 + INTRABC_DELAY_PIXELS) * 8]
            } else {
                [-(sb_size4 * MI_SIZE as i32 * 8), 0]
            };
        }
        pred
    }

    fn read_mv(&mut self, r: usize) {
        let mut diff_mv = [0i32; 2];
        let mv_ctx = if self.b.use_intrabc {
            MV_INTRABC_CONTEXT
        } else {
            0
        };
        let planned_diff = [
            self.plan.mv[r][0] - self.b.pred_mv[r][0],
            self.plan.mv[r][1] - self.b.pred_mv[r][1],
        ];
        let pj = (((planned_diff[0] != 0) as usize) << 1) | (planned_diff[1] != 0) as usize;
        let mv_joint = self.sd.symbol(&mut self.cdf.mv_joint[mv_ctx], pj) as u32;
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[0] = self.read_mv_component(mv_ctx, 0, planned_diff[0]);
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[1] = self.read_mv_component(mv_ctx, 1, planned_diff[1]);
        }
        self.b.mv[r][0] = self.b.pred_mv[r][0] + diff_mv[0];
        self.b.mv[r][1] = self.b.pred_mv[r][1] + diff_mv[1];
    }

    fn read_mv_component(&mut self, ctx: usize, comp: usize, planned: i32) -> i32 {
        let force_integer_mv = self.f.hdr.force_integer_mv;
        let hp = self.f.hdr.allow_high_precision_mv;
        // The planned difference's decomposition (encode mode).
        let pv = planned.unsigned_abs().max(1) as i32 - 1;
        let p_class = if pv < 16 {
            0
        } else {
            crate::bits::floor_log2(pv as u32) as i32 - 3
        };
        let p_rest = if p_class == 0 {
            pv
        } else {
            pv - (CLASS0_SIZE << (p_class + 2))
        };
        let (p_int, p_fr, p_hp) = (
            (p_rest >> 3) as usize,
            ((p_rest >> 1) & 3) as usize,
            (p_rest & 1) as usize,
        );
        let mv_sign = self
            .sd
            .symbol(&mut self.cdf.mv_sign[ctx][comp], (planned < 0) as usize);
        let mv_class =
            self.sd
                .symbol(&mut self.cdf.mv_class[ctx][comp], p_class as usize) as i32;

        let mag = if mv_class == 0 {
            let class0_bit =
                self.sd
                    .symbol(&mut self.cdf.mv_class0_bit[ctx][comp], p_int) as i32;
            let fr = if force_integer_mv {
                3
            } else {
                self.sd.symbol(
                    &mut self.cdf.mv_class0_fr[ctx][comp][class0_bit as usize],
                    p_fr,
                ) as i32
            };
            let hpv = if hp {
                self.sd.symbol(&mut self.cdf.mv_class0_hp[ctx][comp], p_hp) as i32
            } else {
                1
            };
            ((class0_bit << 3) | (fr << 1) | hpv) + 1
        } else {
            let mut d = 0;
            for i in 0..mv_class as usize {
                let bit = self
                    .sd
                    .symbol(&mut self.cdf.mv_bit[ctx][comp][i], (p_int >> i) & 1)
                    as i32;
                d |= bit << i;
            }
            let mut m = CLASS0_SIZE << (mv_class + 2);
            let fr = if force_integer_mv {
                3
            } else {
                self.sd.symbol(&mut self.cdf.mv_fr[ctx][comp], p_fr) as i32
            };
            let hpv = if hp {
                self.sd.symbol(&mut self.cdf.mv_hp[ctx][comp], p_hp) as i32
            } else {
                1
            };
            m += ((d << 3) | (fr << 1) | hpv) + 1;
            m
        };
        if mv_sign != 0 { -mag } else { mag }
    }

    fn read_interintra_mode(&mut self, is_compound: bool) {
        let ms = self.b.mi_size;
        if !self.b.skip_mode
            && self.f.seq.enable_interintra_compound
            && !is_compound
            && (BLOCK_8X8..=BLOCK_32X32).contains(&ms)
        {
            let ctx = SIZE_GROUP[ms] - 1;
            let p = self.plan;
            self.b.interintra = self
                .sd
                .symbol(&mut self.cdf.inter_intra[ctx], p.interintra as usize)
                != 0;
            if self.b.interintra {
                self.b.interintra_mode = self.sd.symbol(
                    &mut self.cdf.inter_intra_mode[ctx],
                    p.interintra_mode as usize,
                );
                self.b.ref_frame[1] = INTRA_FRAME;
                self.b.angle_delta_y = 0;
                self.b.angle_delta_uv = 0;
                self.b.use_filter_intra = false;
                self.b.wedge_interintra = self.sd.symbol(
                    &mut self.cdf.wedge_inter_intra[ms],
                    p.wedge_interintra as usize,
                ) != 0;
                if self.b.wedge_interintra {
                    self.b.wedge_index = self
                        .sd
                        .symbol(&mut self.cdf.wedge_index[ms], p.wedge_index as usize);
                    self.b.wedge_sign = 0;
                }
            }
        } else {
            self.b.interintra = false;
        }
    }

    fn read_motion_mode(&mut self, is_compound: bool) {
        let ms = self.b.mi_size;
        if self.b.skip_mode || !self.f.hdr.is_motion_mode_switchable {
            self.b.motion_mode = SIMPLE;
            return;
        }
        if block_width(ms).min(block_height(ms)) < 8 {
            self.b.motion_mode = SIMPLE;
            return;
        }
        if !self.f.hdr.force_integer_mv
            && (self.b.y_mode == GLOBALMV || self.b.y_mode == GLOBAL_GLOBALMV)
            && self.f.hdr.gm_type[self.b.ref_frame[0] as usize] > TRANSLATION
        {
            self.b.motion_mode = SIMPLE;
            return;
        }
        if is_compound || self.b.ref_frame[1] == INTRA_FRAME || !self.has_overlappable_candidates()
        {
            self.b.motion_mode = SIMPLE;
            return;
        }
        self.find_warp_samples();
        if self.f.hdr.force_integer_mv
            || self.b.num_samples == 0
            || !self.f.hdr.allow_warped_motion
            || self.f.is_scaled(self.b.ref_frame[0])
        {
            let planned = (self.plan.motion_mode as u32 == OBMC) as usize;
            let use_obmc = self.sd.symbol(&mut self.cdf.use_obmc[ms], planned);
            self.b.motion_mode = if use_obmc != 0 { OBMC } else { SIMPLE };
        } else {
            let planned = self.plan.motion_mode as usize;
            self.b.motion_mode = self.sd.symbol(&mut self.cdf.motion_mode[ms], planned) as u32;
        }
    }

    fn read_compound_type(&mut self, is_compound: bool) {
        self.b.comp_group_idx = 0;
        self.b.compound_idx = 1;
        if self.b.skip_mode {
            self.b.compound_type = COMPOUND_AVERAGE;
            return;
        }
        let ms = self.b.mi_size;
        if is_compound {
            let n = WEDGE_BITS[ms];
            let p = self.plan;
            let masked = p.compound == PLAN_WEDGE || p.compound == PLAN_DIFFWTD;
            if self.f.seq.enable_masked_compound {
                let ctx = self.comp_group_idx_ctx();
                self.b.comp_group_idx = self
                    .sd
                    .symbol(&mut self.cdf.comp_group_idx[ctx], masked as usize)
                    as u32;
            }
            if self.b.comp_group_idx == 0 {
                if self.f.seq.enable_jnt_comp {
                    let ctx = self.compound_idx_ctx();
                    let planned = (p.compound != PLAN_DISTANCE) as usize;
                    self.b.compound_idx =
                        self.sd.symbol(&mut self.cdf.compound_idx[ctx], planned) as u32;
                    self.b.compound_type = if self.b.compound_idx != 0 {
                        COMPOUND_AVERAGE
                    } else {
                        COMPOUND_DISTANCE
                    };
                } else {
                    self.b.compound_type = COMPOUND_AVERAGE;
                }
            } else if n == 0 {
                self.b.compound_type = COMPOUND_DIFFWTD;
            } else {
                let planned = if p.compound == PLAN_WEDGE {
                    COMPOUND_WEDGE
                } else {
                    COMPOUND_DIFFWTD
                };
                self.b.compound_type = self
                    .sd
                    .symbol(&mut self.cdf.compound_type[ms], planned as usize)
                    as u32;
            }
            if self.b.compound_type == COMPOUND_WEDGE {
                self.b.wedge_index = self
                    .sd
                    .symbol(&mut self.cdf.wedge_index[ms], p.wedge_index as usize);
                self.b.wedge_sign = self.sd.literal(1, p.wedge_sign as u32) as usize;
            } else if self.b.compound_type == COMPOUND_DIFFWTD {
                self.b.mask_type = self.sd.literal(1, p.mask_type as u32) as usize;
            }
        } else if self.b.interintra {
            self.b.compound_type = if self.b.wedge_interintra {
                COMPOUND_WEDGE
            } else {
                COMPOUND_INTRA
            };
        } else {
            self.b.compound_type = COMPOUND_AVERAGE;
        }
    }

    fn comp_group_idx_ctx(&self) -> usize {
        let b = &self.b;
        let mut ctx = 0;
        if b.avail_u {
            if !b.above_single {
                ctx += self.mi(b.mi_row - 1, b.mi_col).comp_group_idx as usize;
            } else if b.above_ref_frame[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        if b.avail_l {
            if !b.left_single {
                ctx += self.mi(b.mi_row, b.mi_col - 1).comp_group_idx as usize;
            } else if b.left_ref_frame[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        ctx.min(5)
    }

    fn compound_idx_ctx(&self) -> usize {
        let b = &self.b;
        let h = &self.f.hdr;
        let seq = &self.f.seq;
        let fwd = crate::header::get_relative_dist(
            seq,
            h.order_hints[b.ref_frame[0] as usize],
            h.order_hint,
        )
        .abs();
        let bck = crate::header::get_relative_dist(
            seq,
            h.order_hints[b.ref_frame[1] as usize],
            h.order_hint,
        )
        .abs();
        let mut ctx = if fwd == bck { 3 } else { 0 };
        if b.avail_u {
            if !b.above_single {
                ctx += self.mi(b.mi_row - 1, b.mi_col).compound_idx as usize;
            } else if b.above_ref_frame[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        if b.avail_l {
            if !b.left_single {
                ctx += self.mi(b.mi_row, b.mi_col - 1).compound_idx as usize;
            } else if b.left_ref_frame[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        ctx
    }
}

fn partition_cdf(cdf: &mut CdfContext, bsl: usize, ctx: usize) -> &mut [u16] {
    match bsl {
        1 => &mut cdf.partition_w8[ctx],
        2 => &mut cdf.partition_w16[ctx],
        3 => &mut cdf.partition_w32[ctx],
        4 => &mut cdf.partition_w64[ctx],
        _ => &mut cdf.partition_w128[ctx],
    }
}

/// `neg_deinterleave( diff, ref, max )`.
fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        max - (diff + 1)
    }
}

pub(crate) fn is_directional_mode(mode: usize) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

fn has_nearmv(y: usize) -> bool {
    y == NEARMV || y == NEAR_NEARMV || y == NEAR_NEWMV || y == NEW_NEARMV
}

/// `get_mode( refList )`.
pub(crate) fn get_mode(y_mode: usize, ref_list: usize) -> usize {
    if ref_list == 0 {
        if y_mode < NEAREST_NEARESTMV {
            y_mode
        } else if y_mode == NEW_NEWMV || y_mode == NEW_NEARESTMV || y_mode == NEW_NEARMV {
            NEWMV
        } else if y_mode == NEAREST_NEARESTMV || y_mode == NEAREST_NEWMV {
            NEARESTMV
        } else if y_mode == NEAR_NEARMV || y_mode == NEAR_NEWMV {
            NEARMV
        } else {
            GLOBALMV
        }
    } else if y_mode == NEW_NEWMV || y_mode == NEAREST_NEWMV || y_mode == NEAR_NEWMV {
        NEWMV
    } else if y_mode == NEAREST_NEARESTMV || y_mode == NEW_NEARESTMV {
        NEARESTMV
    } else if y_mode == NEAR_NEARMV || y_mode == NEW_NEARMV {
        NEARMV
    } else {
        GLOBALMV
    }
}

/// `find_tx_size( w, h )`.
pub(crate) fn find_tx_size(w: usize, h: usize) -> usize {
    let mut tx = 0;
    while tx < TX_SIZES_ALL {
        if TX_WIDTH[tx] == w && TX_HEIGHT[tx] == h {
            break;
        }
        tx += 1;
    }
    tx
}

/// `get_palette_color_context()`: the color order and the context hash.
fn palette_color_context(map: &[[u8; 64]; 64], r: usize, c: usize, n: usize) -> ([u8; 8], i32) {
    let mut scores = [0i32; PALETTE_COLORS];
    let mut order = [0u8; 8];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i as u8;
    }
    if c > 0 {
        scores[map[r][c - 1] as usize] += 2;
    }
    if r > 0 && c > 0 {
        scores[map[r - 1][c - 1] as usize] += 1;
    }
    if r > 0 {
        scores[map[r - 1][c] as usize] += 2;
    }
    for i in 0..PALETTE_NUM_NEIGHBORS {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in (i + 1)..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_score = scores[max_idx];
            let max_color_order = order[max_idx];
            let mut k = max_idx;
            while k > i {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
                k -= 1;
            }
            scores[i] = max_score;
            order[i] = max_color_order;
        }
    }
    let mut hash = 0;
    for i in 0..PALETTE_NUM_NEIGHBORS {
        hash += scores[i] * PALETTE_COLOR_HASH_MULTIPLIERS[i];
    }
    (order, hash)
}
