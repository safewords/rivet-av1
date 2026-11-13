//! The adaptive CDF arrays: one [`CdfContext`] per frame context (the
//! frame's CDFs, the "Tile" working copies, the "Saved" copies and the eight
//! reference-slot contexts of 7.20), initialised from the default tables
//! (`init_non_coeff_cdfs`, `init_coeff_cdfs`, 6.8.2).

use crate::tables::*;

/// Anything made of CDF arrays: resets the adaptation counters (the last
/// entry of each array), as `load_cdfs` requires.
pub(crate) trait CdfArray {
    fn clear_counts(&mut self);
}

impl<const N: usize> CdfArray for [u16; N] {
    fn clear_counts(&mut self) {
        self[N - 1] = 0;
    }
}

impl<T: CdfArray, const M: usize> CdfArray for [T; M] {
    fn clear_counts(&mut self) {
        for t in self.iter_mut() {
            t.clear_counts();
        }
    }
}

macro_rules! cdf_context {
    ($( $field:ident : $ty:ty = $init:expr ),* $(,)?) => {
        /// Every CDF array of 6.8.2's `init_non_coeff_cdfs` and
        /// `init_coeff_cdfs`, the coefficient CDFs for one quantiser
        /// context.
        #[derive(Clone, PartialEq)]
        pub(crate) struct CdfContext {
            $( pub(crate) $field: $ty, )*
        }

        impl CdfContext {
            /// The defaults, the coefficient CDFs chosen by `base_q_idx`.
            pub(crate) fn new(base_q_idx: u32) -> Box<Self> {
                let idx = coeff_cdf_q_ctx(base_q_idx);
                let _ = idx;
                Box::new(CdfContext { $( $field: $init(idx), )* })
            }

            fn clear_all_counts(&mut self) {
                $( self.$field.clear_counts(); )*
            }
        }
    };
}

/// The `idx` of `init_coeff_cdfs`.
pub(crate) fn coeff_cdf_q_ctx(base_q_idx: u32) -> usize {
    if base_q_idx <= 20 {
        0
    } else if base_q_idx <= 60 {
        1
    } else if base_q_idx <= 120 {
        2
    } else {
        3
    }
}

fn per_ctx<T: Copy, const N: usize>(t: T) -> [T; N] {
    [t; N]
}

cdf_context! {
    y_mode: [[u16; 14]; 4] = |_| DEFAULT_Y_MODE_CDF,
    uv_mode_cfl_not_allowed: [[u16; 14]; 13] = |_| DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
    uv_mode_cfl_allowed: [[u16; 15]; 13] = |_| DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
    angle_delta: [[u16; 8]; 8] = |_| DEFAULT_ANGLE_DELTA_CDF,
    intrabc: [u16; 3] = |_| DEFAULT_INTRABC_CDF,
    partition_w8: [[u16; 5]; 4] = |_| DEFAULT_PARTITION_W8_CDF,
    partition_w16: [[u16; 11]; 4] = |_| DEFAULT_PARTITION_W16_CDF,
    partition_w32: [[u16; 11]; 4] = |_| DEFAULT_PARTITION_W32_CDF,
    partition_w64: [[u16; 11]; 4] = |_| DEFAULT_PARTITION_W64_CDF,
    partition_w128: [[u16; 9]; 4] = |_| DEFAULT_PARTITION_W128_CDF,
    segment_id: [[u16; 9]; 3] = |_| DEFAULT_SEGMENT_ID_CDF,
    segment_id_predicted: [[u16; 3]; 3] = |_| DEFAULT_SEGMENT_ID_PREDICTED_CDF,
    tx_8x8: [[u16; 3]; 3] = |_| DEFAULT_TX_8X8_CDF,
    tx_16x16: [[u16; 4]; 3] = |_| DEFAULT_TX_16X16_CDF,
    tx_32x32: [[u16; 4]; 3] = |_| DEFAULT_TX_32X32_CDF,
    tx_64x64: [[u16; 4]; 3] = |_| DEFAULT_TX_64X64_CDF,
    txfm_split: [[u16; 3]; 21] = |_| DEFAULT_TXFM_SPLIT_CDF,
    filter_intra_mode: [u16; 6] = |_| DEFAULT_FILTER_INTRA_MODE_CDF,
    filter_intra: [[u16; 3]; 22] = |_| DEFAULT_FILTER_INTRA_CDF,
    interp_filter: [[u16; 4]; 16] = |_| DEFAULT_INTERP_FILTER_CDF,
    motion_mode: [[u16; 4]; 22] = |_| DEFAULT_MOTION_MODE_CDF,
    new_mv: [[u16; 3]; 6] = |_| DEFAULT_NEW_MV_CDF,
    zero_mv: [[u16; 3]; 2] = |_| DEFAULT_ZERO_MV_CDF,
    ref_mv: [[u16; 3]; 6] = |_| DEFAULT_REF_MV_CDF,
    compound_mode: [[u16; 9]; 8] = |_| DEFAULT_COMPOUND_MODE_CDF,
    drl_mode: [[u16; 3]; 3] = |_| DEFAULT_DRL_MODE_CDF,
    is_inter: [[u16; 3]; 4] = |_| DEFAULT_IS_INTER_CDF,
    comp_mode: [[u16; 3]; 5] = |_| DEFAULT_COMP_MODE_CDF,
    skip_mode: [[u16; 3]; 3] = |_| DEFAULT_SKIP_MODE_CDF,
    skip: [[u16; 3]; 3] = |_| DEFAULT_SKIP_CDF,
    comp_ref: [[[u16; 3]; 3]; 3] = |_| DEFAULT_COMP_REF_CDF,
    comp_bwd_ref: [[[u16; 3]; 2]; 3] = |_| DEFAULT_COMP_BWD_REF_CDF,
    single_ref: [[[u16; 3]; 6]; 3] = |_| DEFAULT_SINGLE_REF_CDF,
    // Motion vector CDFs: [MvCtx][comp]...
    mv_joint: [[u16; 5]; 2] = |_| per_ctx(DEFAULT_MV_JOINT_CDF),
    mv_class: [[[u16; 12]; 2]; 2] = |_| per_ctx(DEFAULT_MV_CLASS_CDF),
    mv_class0_bit: [[[u16; 3]; 2]; 2] = |_| per_ctx(per_ctx(DEFAULT_MV_CLASS0_BIT_CDF)),
    mv_fr: [[[u16; 5]; 2]; 2] = |_| per_ctx(DEFAULT_MV_FR_CDF),
    mv_class0_fr: [[[[u16; 5]; 2]; 2]; 2] = |_| per_ctx(DEFAULT_MV_CLASS0_FR_CDF),
    mv_class0_hp: [[[u16; 3]; 2]; 2] = |_| per_ctx(per_ctx(DEFAULT_MV_CLASS0_HP_CDF)),
    mv_sign: [[[u16; 3]; 2]; 2] = |_| per_ctx(per_ctx(DEFAULT_MV_SIGN_CDF)),
    mv_bit: [[[[u16; 3]; 10]; 2]; 2] = |_| per_ctx(per_ctx(DEFAULT_MV_BIT_CDF)),
    mv_hp: [[[u16; 3]; 2]; 2] = |_| per_ctx(per_ctx(DEFAULT_MV_HP_CDF)),
    palette_y_mode: [[[u16; 3]; 3]; 7] = |_| DEFAULT_PALETTE_Y_MODE_CDF,
    palette_uv_mode: [[u16; 3]; 2] = |_| DEFAULT_PALETTE_UV_MODE_CDF,
    palette_y_size: [[u16; 8]; 7] = |_| DEFAULT_PALETTE_Y_SIZE_CDF,
    palette_uv_size: [[u16; 8]; 7] = |_| DEFAULT_PALETTE_UV_SIZE_CDF,
    // [PaletteSize - 2][ctx], each padded to 9 entries; the live part of a
    // row is its first PaletteSize + 1 entries.
    palette_y_color: [[[u16; 9]; 5]; 7] = |_| palette_colors(true),
    palette_uv_color: [[[u16; 9]; 5]; 7] = |_| palette_colors(false),
    delta_q: [u16; 5] = |_| DEFAULT_DELTA_Q_CDF,
    delta_lf: [u16; 5] = |_| DEFAULT_DELTA_LF_CDF,
    delta_lf_multi: [[u16; 5]; 4] = |_| per_ctx(DEFAULT_DELTA_LF_CDF),
    intra_tx_type_set1: [[[u16; 8]; 13]; 2] = |_| DEFAULT_INTRA_TX_TYPE_SET1_CDF,
    intra_tx_type_set2: [[[u16; 6]; 13]; 3] = |_| DEFAULT_INTRA_TX_TYPE_SET2_CDF,
    inter_tx_type_set1: [[u16; 17]; 2] = |_| DEFAULT_INTER_TX_TYPE_SET1_CDF,
    inter_tx_type_set2: [u16; 13] = |_| DEFAULT_INTER_TX_TYPE_SET2_CDF,
    inter_tx_type_set3: [[u16; 3]; 4] = |_| DEFAULT_INTER_TX_TYPE_SET3_CDF,
    use_obmc: [[u16; 3]; 22] = |_| DEFAULT_USE_OBMC_CDF,
    inter_intra: [[u16; 3]; 3] = |_| DEFAULT_INTER_INTRA_CDF,
    comp_ref_type: [[u16; 3]; 5] = |_| DEFAULT_COMP_REF_TYPE_CDF,
    cfl_sign: [u16; 9] = |_| DEFAULT_CFL_SIGN_CDF,
    uni_comp_ref: [[[u16; 3]; 3]; 3] = |_| DEFAULT_UNI_COMP_REF_CDF,
    wedge_inter_intra: [[u16; 3]; 22] = |_| DEFAULT_WEDGE_INTER_INTRA_CDF,
    comp_group_idx: [[u16; 3]; 6] = |_| DEFAULT_COMP_GROUP_IDX_CDF,
    compound_idx: [[u16; 3]; 6] = |_| DEFAULT_COMPOUND_IDX_CDF,
    compound_type: [[u16; 3]; 22] = |_| DEFAULT_COMPOUND_TYPE_CDF,
    inter_intra_mode: [[u16; 5]; 3] = |_| DEFAULT_INTER_INTRA_MODE_CDF,
    wedge_index: [[u16; 17]; 22] = |_| DEFAULT_WEDGE_INDEX_CDF,
    cfl_alpha: [[u16; 17]; 6] = |_| DEFAULT_CFL_ALPHA_CDF,
    use_wiener: [u16; 3] = |_| DEFAULT_USE_WIENER_CDF,
    use_sgrproj: [u16; 3] = |_| DEFAULT_USE_SGRPROJ_CDF,
    restoration_type: [u16; 4] = |_| DEFAULT_RESTORATION_TYPE_CDF,
    // Coefficient CDFs (init_coeff_cdfs).
    txb_skip: [[[u16; 3]; 13]; 5] = |i: usize| DEFAULT_TXB_SKIP_CDF[i],
    eob_pt_16: [[[u16; 6]; 2]; 2] = |i: usize| DEFAULT_EOB_PT_16_CDF[i],
    eob_pt_32: [[[u16; 7]; 2]; 2] = |i: usize| DEFAULT_EOB_PT_32_CDF[i],
    eob_pt_64: [[[u16; 8]; 2]; 2] = |i: usize| DEFAULT_EOB_PT_64_CDF[i],
    eob_pt_128: [[[u16; 9]; 2]; 2] = |i: usize| DEFAULT_EOB_PT_128_CDF[i],
    eob_pt_256: [[[u16; 10]; 2]; 2] = |i: usize| DEFAULT_EOB_PT_256_CDF[i],
    eob_pt_512: [[u16; 11]; 2] = |i: usize| DEFAULT_EOB_PT_512_CDF[i],
    eob_pt_1024: [[u16; 12]; 2] = |i: usize| DEFAULT_EOB_PT_1024_CDF[i],
    eob_extra: [[[[u16; 3]; 9]; 2]; 5] = |i: usize| DEFAULT_EOB_EXTRA_CDF[i],
    dc_sign: [[[u16; 3]; 3]; 2] = |i: usize| DEFAULT_DC_SIGN_CDF[i],
    coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5] = |i: usize| DEFAULT_COEFF_BASE_EOB_CDF[i],
    coeff_base: [[[[u16; 5]; 42]; 2]; 5] = |i: usize| DEFAULT_COEFF_BASE_CDF[i],
    coeff_br: [[[[u16; 5]; 21]; 2]; 5] = |i: usize| DEFAULT_COEFF_BR_CDF[i],
}

impl CdfContext {
    /// `load_cdfs`' counter reset.
    pub(crate) fn clear_counts(&mut self) {
        self.clear_all_counts();
        // The palette color rows are padded: their counter sits at index
        // PaletteSize, not at the end of the row.
        for rows in [&mut self.palette_y_color, &mut self.palette_uv_color] {
            for (n, ctxs) in rows.iter_mut().enumerate() {
                for row in ctxs.iter_mut() {
                    row[n + 2] = 0;
                }
            }
        }
    }
}

fn palette_colors(luma: bool) -> [[[u16; 9]; 5]; 7] {
    let mut out = [[[0u16; 9]; 5]; 7];
    fn put<const N: usize>(dst: &mut [[u16; 9]; 5], src: &[[u16; N]; 5]) {
        for (d, s) in dst.iter_mut().zip(src) {
            d[..N].copy_from_slice(s);
        }
    }
    if luma {
        put(&mut out[0], &DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF);
        put(&mut out[1], &DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF);
        put(&mut out[2], &DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF);
        put(&mut out[3], &DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF);
        put(&mut out[4], &DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF);
        put(&mut out[5], &DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF);
        put(&mut out[6], &DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF);
    } else {
        put(&mut out[0], &DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF);
        put(&mut out[1], &DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF);
        put(&mut out[2], &DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF);
        put(&mut out[3], &DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF);
        put(&mut out[4], &DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF);
        put(&mut out[5], &DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF);
        put(&mut out[6], &DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF);
    }
    out
}

/// The live part of a palette color CDF row: `PaletteSize + 1` entries.
pub(crate) fn palette_color_cdf(
    rows: &mut [[[u16; 9]; 5]; 7],
    size: usize,
    ctx: usize,
) -> &mut [u16] {
    &mut rows[size - 2][ctx][..size + 1]
}
