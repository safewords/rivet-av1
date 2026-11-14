//! The frame header OBU (5.9, 6.8): `uncompressed_header()` and the
//! structures it reads, parsed against the decoder's reference state.

use std::sync::Arc;

use crate::bits::BitReader;
use crate::consts::*;
use crate::obu::SequenceHeader;
use crate::tables::*;
use crate::{Error, Result};

/// Film grain parameters (5.9.30, 6.8.20).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FilmGrainParams {
    pub(crate) apply_grain: bool,
    pub(crate) grain_seed: u32,
    pub(crate) update_grain: bool,
    pub(crate) num_y_points: usize,
    pub(crate) point_y_value: [i32; 16],
    pub(crate) point_y_scaling: [i32; 16],
    pub(crate) chroma_scaling_from_luma: bool,
    pub(crate) num_cb_points: usize,
    pub(crate) point_cb_value: [i32; 16],
    pub(crate) point_cb_scaling: [i32; 16],
    pub(crate) num_cr_points: usize,
    pub(crate) point_cr_value: [i32; 16],
    pub(crate) point_cr_scaling: [i32; 16],
    pub(crate) grain_scaling_minus_8: u32,
    pub(crate) ar_coeff_lag: i32,
    pub(crate) ar_coeffs_y_plus_128: [i32; 24],
    pub(crate) ar_coeffs_cb_plus_128: [i32; 25],
    pub(crate) ar_coeffs_cr_plus_128: [i32; 25],
    pub(crate) ar_coeff_shift_minus_6: u32,
    pub(crate) grain_scale_shift: u32,
    pub(crate) cb_mult: i32,
    pub(crate) cb_luma_mult: i32,
    pub(crate) cb_offset: i32,
    pub(crate) cr_mult: i32,
    pub(crate) cr_luma_mult: i32,
    pub(crate) cr_offset: i32,
    pub(crate) overlap_flag: bool,
    pub(crate) clip_to_restricted_range: bool,
}

/// What `load_previous()` takes from a reference slot, and the slot's
/// geometry for `frame_size_with_refs()` (7.20, 7.21).
#[derive(Debug, Clone)]
pub(crate) struct RefHeaderState {
    pub(crate) frame_id: u32,
    pub(crate) frame_type: u32,
    pub(crate) upscaled_width: u32,
    pub(crate) frame_height: u32,
    pub(crate) render_width: u32,
    pub(crate) render_height: u32,
    pub(crate) loop_filter_ref_deltas: [i32; 8],
    pub(crate) loop_filter_mode_deltas: [i32; 2],
    pub(crate) feature_enabled: [[bool; 8]; 8],
    pub(crate) feature_data: [[i32; 8]; 8],
    pub(crate) gm_params: [[i32; 6]; 8],
    pub(crate) film_grain: FilmGrainParams,
}

/// The reference-slot state that header parsing reads and updates:
/// `RefValid`, `RefOrderHint`, `RefFrameId` and the slots themselves.
#[derive(Clone, Default)]
pub(crate) struct RefState {
    pub(crate) valid: [bool; 8],
    pub(crate) order_hint: [u32; 8],
    pub(crate) slots: [Option<Arc<RefHeaderState>>; 8],
    /// `current_frame_id` of the previous frame (PrevFrameID).
    pub(crate) current_frame_id: u32,
    /// Whether a frame has set `current_frame_id` since the decoder began:
    /// Annex C (C.5) has the first frame's not compared with an
    /// uninitialised PrevFrameID.
    pub(crate) frame_id_seen: bool,
}

/// `tile_info()` (5.9.15).
#[derive(Debug, Clone, Default)]
pub(crate) struct TileInfo {
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) cols_log2: u32,
    pub(crate) rows_log2: u32,
    pub(crate) mi_col_starts: Vec<usize>,
    pub(crate) mi_row_starts: Vec<usize>,
    pub(crate) context_update_tile_id: usize,
    pub(crate) tile_size_bytes: u32,
}

/// `uncompressed_header()`.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameHeader {
    pub(crate) temporal_id: u32,
    pub(crate) spatial_id: u32,
    pub(crate) show_existing_frame: bool,
    pub(crate) frame_to_show_map_idx: usize,
    pub(crate) frame_type: u32,
    pub(crate) frame_is_intra: bool,
    pub(crate) show_frame: bool,
    pub(crate) showable_frame: bool,
    pub(crate) error_resilient_mode: bool,
    pub(crate) disable_cdf_update: bool,
    pub(crate) allow_screen_content_tools: bool,
    pub(crate) force_integer_mv: bool,
    pub(crate) current_frame_id: u32,
    /// PrevFrameID when `current_frame_id` was read, if a frame had set it
    /// before (for the conformance checks of `current_frame_id`).
    pub(crate) prev_frame_id: Option<u32>,
    /// `display_frame_id` of a `show_existing_frame` header, and
    /// `expectedFrameId[ i ]` of an inter frame's references, when the
    /// sequence has frame ids.
    pub(crate) display_frame_id: u32,
    pub(crate) expected_frame_id: [u32; 7],
    /// `frame_refs_short_signaling`: `ref_frame_idx` derived from every
    /// slot's `RefOrderHint` (7.8), not only from the slots it names.
    pub(crate) frame_refs_short_signaling: bool,
    /// `film_grain_params_ref_idx`, when the grain parameters were loaded
    /// from a slot (`update_grain` equal to 0).
    pub(crate) film_grain_params_ref_idx: Option<usize>,
    pub(crate) frame_size_override_flag: bool,
    pub(crate) order_hint: u32,
    pub(crate) primary_ref_frame: usize,
    pub(crate) refresh_frame_flags: u32,
    pub(crate) allow_intrabc: bool,
    pub(crate) ref_frame_idx: [usize; 7],
    pub(crate) allow_high_precision_mv: bool,
    pub(crate) interpolation_filter: u32,
    pub(crate) is_motion_mode_switchable: bool,
    pub(crate) use_ref_frame_mvs: bool,
    /// `OrderHints[ ref ]`, indexed by reference frame (LAST_FRAME..).
    pub(crate) order_hints: [u32; 8],
    pub(crate) ref_frame_sign_bias: [bool; 8],
    pub(crate) disable_frame_end_update_cdf: bool,
    pub(crate) frame_width: u32,
    pub(crate) frame_height: u32,
    pub(crate) upscaled_width: u32,
    pub(crate) render_width: u32,
    pub(crate) render_height: u32,
    pub(crate) use_superres: bool,
    pub(crate) superres_denom: u32,
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) tile_info: TileInfo,
    pub(crate) base_q_idx: u32,
    pub(crate) delta_q_y_dc: i32,
    pub(crate) delta_q_u_dc: i32,
    pub(crate) delta_q_u_ac: i32,
    pub(crate) delta_q_v_dc: i32,
    pub(crate) delta_q_v_ac: i32,
    pub(crate) using_qmatrix: bool,
    pub(crate) qm_y: u32,
    pub(crate) qm_u: u32,
    pub(crate) qm_v: u32,
    pub(crate) segmentation_enabled: bool,
    pub(crate) segmentation_update_map: bool,
    pub(crate) segmentation_temporal_update: bool,
    pub(crate) segmentation_update_data: bool,
    pub(crate) feature_enabled: [[bool; 8]; 8],
    pub(crate) feature_data: [[i32; 8]; 8],
    pub(crate) seg_id_pre_skip: bool,
    pub(crate) last_active_seg_id: usize,
    pub(crate) delta_q_present: bool,
    pub(crate) delta_q_res: u32,
    pub(crate) delta_lf_present: bool,
    pub(crate) delta_lf_res: u32,
    pub(crate) delta_lf_multi: bool,
    pub(crate) coded_lossless: bool,
    pub(crate) all_lossless: bool,
    pub(crate) lossless_array: [bool; 8],
    pub(crate) seg_qm_level: [[u32; 8]; 3],
    pub(crate) loop_filter_level: [i32; 4],
    pub(crate) loop_filter_sharpness: i32,
    pub(crate) loop_filter_delta_enabled: bool,
    pub(crate) loop_filter_ref_deltas: [i32; 8],
    pub(crate) loop_filter_mode_deltas: [i32; 2],
    pub(crate) cdef_damping: i32,
    pub(crate) cdef_bits: u32,
    pub(crate) cdef_y_pri_strength: [i32; 8],
    pub(crate) cdef_y_sec_strength: [i32; 8],
    pub(crate) cdef_uv_pri_strength: [i32; 8],
    pub(crate) cdef_uv_sec_strength: [i32; 8],
    pub(crate) frame_restoration_type: [u8; 3],
    pub(crate) loop_restoration_size: [usize; 3],
    pub(crate) uses_lr: bool,
    pub(crate) tx_mode: u32,
    pub(crate) reference_select: bool,
    pub(crate) skip_mode_present: bool,
    pub(crate) skip_mode_frame: [i32; 2],
    pub(crate) allow_warped_motion: bool,
    pub(crate) reduced_tx_set: bool,
    pub(crate) gm_type: [u32; 8],
    pub(crate) gm_params: [[i32; 6]; 8],
    pub(crate) film_grain: FilmGrainParams,
}

impl FrameHeader {}

/// `get_relative_dist( a, b )` (5.9.3).
pub(crate) fn get_relative_dist(seq: &SequenceHeader, a: u32, b: u32) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

fn default_gm() -> [[i32; 6]; 8] {
    let mut g = [[0i32; 6]; 8];
    for r in g.iter_mut() {
        r[2] = 1 << WARPEDMODEL_PREC_BITS;
        r[5] = 1 << WARPEDMODEL_PREC_BITS;
    }
    g
}

const DEFAULT_REF_DELTAS: [i32; 8] = [1, 0, 0, 0, -1, 0, -1, -1];

struct Parser<'a, 'b> {
    r: &'a mut BitReader<'b>,
    seq: &'a SequenceHeader,
    st: &'a mut RefState,
    h: FrameHeader,
    prev_gm_params: [[i32; 6]; 8],
}

impl FrameHeader {
    /// Parses `uncompressed_header()`, updating `RefValid` and
    /// `RefOrderHint` in `st` as the syntax does. `load_previous()` and
    /// `setup_past_independence()` are applied to the header's loop filter
    /// deltas, segmentation features and previous global motion; the CDF
    /// and segmentation-map side effects are the caller's.
    pub(crate) fn parse(
        r: &mut BitReader,
        seq: &SequenceHeader,
        st: &mut RefState,
        temporal_id: u32,
        spatial_id: u32,
    ) -> Result<FrameHeader> {
        let mut p = Parser {
            r,
            seq,
            st,
            h: FrameHeader {
                temporal_id,
                spatial_id,
                ..Default::default()
            },
            prev_gm_params: default_gm(),
        };
        p.uncompressed_header()?;
        let start_bits = 0;
        let _ = start_bits;
        Ok(p.h)
    }
}

impl Parser<'_, '_> {
    fn slot(&self, i: usize) -> Result<&Arc<RefHeaderState>> {
        self.st.slots[i].as_ref().ok_or_else(|| {
            Error::bitstream(format!(
                "reference slot {i} is empty (RefValid[ ref_frame_idx ] is 0, 6.8.2)"
            ))
        })
    }

    fn uncompressed_header(&mut self) -> Result<()> {
        let seq = self.seq;
        let id_len = if seq.frame_id_numbers_present {
            seq.additional_frame_id_length_minus_1 + seq.delta_frame_id_length_minus_2 + 3
        } else {
            0
        };
        let all_frames = (1u32 << NUM_REF_FRAMES) - 1;
        let h = &mut self.h;
        if seq.reduced_still_picture_header {
            h.show_existing_frame = false;
            h.frame_type = KEY_FRAME;
            h.frame_is_intra = true;
            h.show_frame = true;
            h.showable_frame = false;
        } else {
            h.show_existing_frame = self.r.flag()?;
            if h.show_existing_frame {
                h.frame_to_show_map_idx = self.r.f(3)? as usize;
                if seq.decoder_model_info_present && !seq.equal_picture_interval {
                    self.r.f(seq.frame_presentation_time_length_minus_1 + 1)?;
                }
                h.refresh_frame_flags = 0;
                if seq.frame_id_numbers_present {
                    h.display_frame_id = self.r.f(id_len)?;
                }
                let idx = h.frame_to_show_map_idx;
                let slot = self.st.slots[idx].clone().ok_or_else(|| {
                    Error::bitstream(
                        "show_existing_frame of an empty slot (RefValid[ frame_to_show_map_idx ] \
                             is 0, 6.8.2)",
                    )
                })?;
                let h = &mut self.h;
                h.frame_type = slot.frame_type;
                if h.frame_type == KEY_FRAME {
                    h.refresh_frame_flags = all_frames;
                }
                if seq.film_grain_params_present {
                    h.film_grain = slot.film_grain.clone();
                }
                return Ok(());
            }
            h.frame_type = self.r.f(2)?;
            h.frame_is_intra = h.frame_type == INTRA_ONLY_FRAME || h.frame_type == KEY_FRAME;
            h.show_frame = self.r.flag()?;
            if h.show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
                self.r.f(seq.frame_presentation_time_length_minus_1 + 1)?;
            }
            let h = &mut self.h;
            if h.show_frame {
                h.showable_frame = h.frame_type != KEY_FRAME;
            } else {
                h.showable_frame = self.r.flag()?;
            }
            let h = &mut self.h;
            if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
                h.error_resilient_mode = true;
            } else {
                h.error_resilient_mode = self.r.flag()?;
            }
        }
        let h = &mut self.h;
        if h.frame_type == KEY_FRAME && h.show_frame {
            for i in 0..NUM_REF_FRAMES {
                self.st.valid[i] = false;
                self.st.order_hint[i] = 0;
            }
            for i in 0..REFS_PER_FRAME {
                h.order_hints[LAST_FRAME as usize + i] = 0;
            }
        }
        h.disable_cdf_update = self.r.flag()?;
        let h = &mut self.h;
        h.allow_screen_content_tools =
            if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
                self.r.flag()?
            } else {
                seq.seq_force_screen_content_tools != 0
            };
        let h = &mut self.h;
        if h.allow_screen_content_tools {
            h.force_integer_mv = if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
                self.r.flag()?
            } else {
                seq.seq_force_integer_mv != 0
            };
        } else {
            h.force_integer_mv = false;
        }
        let h = &mut self.h;
        if h.frame_is_intra {
            h.force_integer_mv = true;
        }
        if seq.frame_id_numbers_present {
            // PrevFrameID = current_frame_id, the previous frame's (which
            // the reference update keeps in `st`).
            self.h.prev_frame_id = self.st.frame_id_seen.then_some(self.st.current_frame_id);
            self.h.current_frame_id = self.r.f(id_len)?;
            // mark_ref_frames( idLen )
            let diff_len = seq.delta_frame_id_length_minus_2 + 2;
            let cur = self.h.current_frame_id as i64;
            for i in 0..NUM_REF_FRAMES {
                let Some(slot) = &self.st.slots[i] else {
                    continue;
                };
                let rid = slot.frame_id as i64;
                if cur > (1i64 << diff_len) {
                    if rid > cur || rid < cur - (1i64 << diff_len) {
                        self.st.valid[i] = false;
                    }
                } else if rid > cur && rid < (1i64 << id_len) + cur - (1i64 << diff_len) {
                    self.st.valid[i] = false;
                }
            }
        } else {
            self.h.current_frame_id = 0;
        }
        let h = &mut self.h;
        if h.frame_type == SWITCH_FRAME {
            h.frame_size_override_flag = true;
        } else if seq.reduced_still_picture_header {
            h.frame_size_override_flag = false;
        } else {
            h.frame_size_override_flag = self.r.flag()?;
        }
        let h = &mut self.h;
        h.order_hint = self.r.f(seq.order_hint_bits)?;
        let h = &mut self.h;
        if h.frame_is_intra || h.error_resilient_mode {
            h.primary_ref_frame = PRIMARY_REF_NONE;
        } else {
            h.primary_ref_frame = self.r.f(3)? as usize;
        }
        if seq.decoder_model_info_present {
            let buffer_removal_time_present = self.r.flag()?;
            if buffer_removal_time_present {
                for op in 0..=seq.operating_points_cnt_minus_1 {
                    if seq.decoder_model_present_for_this_op[op] {
                        let idc = seq.operating_point_idc[op];
                        let in_t = (idc >> self.h.temporal_id) & 1;
                        let in_s = (idc >> (self.h.spatial_id + 8)) & 1;
                        if idc == 0 || (in_t != 0 && in_s != 0) {
                            self.r.f(seq.buffer_removal_time_length_minus_1 + 1)?;
                        }
                    }
                }
            }
        }
        let h = &mut self.h;
        h.allow_high_precision_mv = false;
        h.use_ref_frame_mvs = false;
        h.allow_intrabc = false;
        if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
            h.refresh_frame_flags = all_frames;
        } else {
            h.refresh_frame_flags = self.r.f(8)?;
        }
        let h = &mut self.h;
        if (!h.frame_is_intra || h.refresh_frame_flags != all_frames)
            && h.error_resilient_mode
            && seq.enable_order_hint
        {
            for i in 0..NUM_REF_FRAMES {
                let roh = self.r.f(seq.order_hint_bits)?;
                if roh != self.st.order_hint[i] {
                    self.st.valid[i] = false;
                }
            }
        }
        if self.h.frame_is_intra {
            self.frame_size()?;
            self.render_size()?;
            let h = &mut self.h;
            if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
                h.allow_intrabc = self.r.flag()?;
            }
        } else {
            let mut frame_refs_short_signaling = false;
            if seq.enable_order_hint {
                frame_refs_short_signaling = self.r.flag()?;
                self.h.frame_refs_short_signaling = frame_refs_short_signaling;
                if frame_refs_short_signaling {
                    let last = self.r.f(3)? as usize;
                    let gold = self.r.f(3)? as usize;
                    self.set_frame_refs(last, gold);
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !frame_refs_short_signaling {
                    self.h.ref_frame_idx[i] = self.r.f(3)? as usize;
                }
                if seq.frame_id_numbers_present {
                    let delta_frame_id = self.r.f(seq.delta_frame_id_length_minus_2 + 2)? + 1;
                    let ids = 1u64 << id_len;
                    self.h.expected_frame_id[i] = ((self.h.current_frame_id as u64 + ids
                        - delta_frame_id as u64)
                        % ids) as u32;
                }
            }
            for i in 0..REFS_PER_FRAME {
                self.slot(self.h.ref_frame_idx[i])?;
            }
            if self.h.frame_size_override_flag && !self.h.error_resilient_mode {
                self.frame_size_with_refs()?;
            } else {
                self.frame_size()?;
                self.render_size()?;
            }
            let h = &mut self.h;
            if h.force_integer_mv {
                h.allow_high_precision_mv = false;
            } else {
                h.allow_high_precision_mv = self.r.flag()?;
            }
            // read_interpolation_filter()
            let is_filter_switchable = self.r.flag()?;
            self.h.interpolation_filter = if is_filter_switchable {
                SWITCHABLE
            } else {
                self.r.f(2)?
            };
            self.h.is_motion_mode_switchable = self.r.flag()?;
            if self.h.error_resilient_mode || !seq.enable_ref_frame_mvs {
                self.h.use_ref_frame_mvs = false;
            } else {
                self.h.use_ref_frame_mvs = self.r.flag()?;
            }
            for i in 0..REFS_PER_FRAME {
                let ref_frame = LAST_FRAME as usize + i;
                let hint = self.st.order_hint[self.h.ref_frame_idx[i]];
                self.h.order_hints[ref_frame] = hint;
                self.h.ref_frame_sign_bias[ref_frame] =
                    seq.enable_order_hint && get_relative_dist(seq, hint, self.h.order_hint) > 0;
            }
        }
        let h = &mut self.h;
        if seq.reduced_still_picture_header || h.disable_cdf_update {
            h.disable_frame_end_update_cdf = true;
        } else {
            h.disable_frame_end_update_cdf = self.r.flag()?;
        }
        // setup_past_independence() / load_previous()
        if self.h.primary_ref_frame == PRIMARY_REF_NONE {
            self.h.feature_enabled = [[false; 8]; 8];
            self.h.feature_data = [[0; 8]; 8];
            self.prev_gm_params = default_gm();
            self.h.loop_filter_delta_enabled = true;
            self.h.loop_filter_ref_deltas = DEFAULT_REF_DELTAS;
            self.h.loop_filter_mode_deltas = [0; 2];
        } else {
            let prev = self
                .slot(self.h.ref_frame_idx[self.h.primary_ref_frame])?
                .clone();
            self.prev_gm_params = prev.gm_params;
            self.h.loop_filter_ref_deltas = prev.loop_filter_ref_deltas;
            self.h.loop_filter_mode_deltas = prev.loop_filter_mode_deltas;
            self.h.feature_enabled = prev.feature_enabled;
            self.h.feature_data = prev.feature_data;
        }
        self.tile_info()?;
        self.quantization_params()?;
        self.segmentation_params()?;
        // delta_q_params()
        let h = &mut self.h;
        h.delta_q_res = 0;
        h.delta_q_present = false;
        if h.base_q_idx > 0 {
            h.delta_q_present = self.r.flag()?;
        }
        if self.h.delta_q_present {
            self.h.delta_q_res = self.r.f(2)?;
        }
        // delta_lf_params()
        let h = &mut self.h;
        h.delta_lf_present = false;
        h.delta_lf_res = 0;
        h.delta_lf_multi = false;
        if h.delta_q_present {
            if !h.allow_intrabc {
                h.delta_lf_present = self.r.flag()?;
            }
            if self.h.delta_lf_present {
                self.h.delta_lf_res = self.r.f(2)?;
                self.h.delta_lf_multi = self.r.flag()?;
            }
        }
        let h = &mut self.h;
        h.coded_lossless = true;
        for seg in 0..MAX_SEGMENTS {
            let qindex = get_qindex_header(h, seg);
            let lossless = qindex == 0
                && h.delta_q_y_dc == 0
                && h.delta_q_u_ac == 0
                && h.delta_q_u_dc == 0
                && h.delta_q_v_ac == 0
                && h.delta_q_v_dc == 0;
            h.lossless_array[seg] = lossless;
            if !lossless {
                h.coded_lossless = false;
            }
            if h.using_qmatrix {
                if lossless {
                    h.seg_qm_level[0][seg] = 15;
                    h.seg_qm_level[1][seg] = 15;
                    h.seg_qm_level[2][seg] = 15;
                } else {
                    h.seg_qm_level[0][seg] = h.qm_y;
                    h.seg_qm_level[1][seg] = h.qm_u;
                    h.seg_qm_level[2][seg] = h.qm_v;
                }
            }
        }
        h.all_lossless = h.coded_lossless && h.frame_width == h.upscaled_width;
        self.loop_filter_params()?;
        self.cdef_params()?;
        self.lr_params()?;
        // read_tx_mode()
        if self.h.coded_lossless {
            self.h.tx_mode = ONLY_4X4;
        } else {
            self.h.tx_mode = if self.r.flag()? {
                TX_MODE_SELECT
            } else {
                TX_MODE_LARGEST
            };
        }
        // frame_reference_mode()
        self.h.reference_select = if self.h.frame_is_intra {
            false
        } else {
            self.r.flag()?
        };
        self.skip_mode_params()?;
        let h = &mut self.h;
        if h.frame_is_intra || h.error_resilient_mode || !seq.enable_warped_motion {
            h.allow_warped_motion = false;
        } else {
            h.allow_warped_motion = self.r.flag()?;
        }
        self.h.reduced_tx_set = self.r.flag()?;
        self.global_motion_params()?;
        self.film_grain_params()?;
        Ok(())
    }

    /// The set frame refs process (7.8).
    fn set_frame_refs(&mut self, last_frame_idx: usize, gold_frame_idx: usize) {
        let seq = self.seq;
        let mut idx = [-1i32; 7];
        idx[0] = last_frame_idx as i32;
        idx[(GOLDEN_FRAME - LAST_FRAME) as usize] = gold_frame_idx as i32;
        let mut used = [false; 8];
        used[last_frame_idx] = true;
        used[gold_frame_idx] = true;
        let cur_frame_hint = 1i32 << (seq.order_hint_bits - 1);
        let mut shifted = [0i32; 8];
        for (i, s) in shifted.iter_mut().enumerate() {
            *s = cur_frame_hint + get_relative_dist(seq, self.st.order_hint[i], self.h.order_hint);
        }
        // ALTREF: latest backward.
        {
            let mut r = -1i32;
            let mut latest = 0;
            for i in 0..8 {
                let hint = shifted[i];
                if !used[i] && hint >= cur_frame_hint && (r < 0 || hint >= latest) {
                    r = i as i32;
                    latest = hint;
                }
            }
            if r >= 0 {
                idx[(ALTREF_FRAME - LAST_FRAME) as usize] = r;
                used[r as usize] = true;
            }
        }
        // BWDREF then ALTREF2: earliest backward.
        for rf in [BWDREF_FRAME, ALTREF2_FRAME] {
            let mut r = -1i32;
            let mut earliest = 0;
            for i in 0..8 {
                let hint = shifted[i];
                if !used[i] && hint >= cur_frame_hint && (r < 0 || hint < earliest) {
                    r = i as i32;
                    earliest = hint;
                }
            }
            if r >= 0 {
                idx[(rf - LAST_FRAME) as usize] = r;
                used[r as usize] = true;
            }
        }
        // The rest: latest forward, in Ref_Frame_List order.
        for &rf in REF_FRAME_LIST.iter() {
            let k = rf - LAST_FRAME as usize;
            if idx[k] < 0 {
                let mut r = -1i32;
                let mut latest = 0;
                for i in 0..8 {
                    let hint = shifted[i];
                    if !used[i] && hint < cur_frame_hint && (r < 0 || hint >= latest) {
                        r = i as i32;
                        latest = hint;
                    }
                }
                if r >= 0 {
                    idx[k] = r;
                    used[r as usize] = true;
                }
            }
        }
        let mut r = -1i32;
        let mut earliest = 0;
        for (i, &hint) in shifted.iter().enumerate() {
            if r < 0 || hint < earliest {
                r = i as i32;
                earliest = hint;
            }
        }
        for (i, v) in idx.iter().enumerate() {
            self.h.ref_frame_idx[i] = if *v < 0 { r as usize } else { *v as usize };
        }
    }

    fn frame_size(&mut self) -> Result<()> {
        let seq = self.seq;
        if self.h.frame_size_override_flag {
            self.h.frame_width = self.r.f(seq.frame_width_bits)? + 1;
            self.h.frame_height = self.r.f(seq.frame_height_bits)? + 1;
        } else {
            self.h.frame_width = seq.max_frame_width;
            self.h.frame_height = seq.max_frame_height;
        }
        self.superres_params()?;
        self.compute_image_size();
        Ok(())
    }

    fn render_size(&mut self) -> Result<()> {
        if self.r.flag()? {
            self.h.render_width = self.r.f(16)? + 1;
            self.h.render_height = self.r.f(16)? + 1;
        } else {
            self.h.render_width = self.h.upscaled_width;
            self.h.render_height = self.h.frame_height;
        }
        Ok(())
    }

    fn frame_size_with_refs(&mut self) -> Result<()> {
        let mut found = false;
        for i in 0..REFS_PER_FRAME {
            if self.r.flag()? {
                let s = self.slot(self.h.ref_frame_idx[i])?.clone();
                self.h.upscaled_width = s.upscaled_width;
                self.h.frame_width = self.h.upscaled_width;
                self.h.frame_height = s.frame_height;
                self.h.render_width = s.render_width;
                self.h.render_height = s.render_height;
                found = true;
                break;
            }
        }
        if !found {
            self.frame_size()?;
            self.render_size()?;
        } else {
            self.superres_params()?;
            self.compute_image_size();
        }
        Ok(())
    }

    fn superres_params(&mut self) -> Result<()> {
        self.h.use_superres = if self.seq.enable_superres {
            self.r.flag()?
        } else {
            false
        };
        self.h.superres_denom = if self.h.use_superres {
            self.r.f(SUPERRES_DENOM_BITS)? + SUPERRES_DENOM_MIN
        } else {
            SUPERRES_NUM
        };
        self.h.upscaled_width = self.h.frame_width;
        self.h.frame_width = (self.h.upscaled_width * SUPERRES_NUM + self.h.superres_denom / 2)
            / self.h.superres_denom;
        Ok(())
    }

    fn compute_image_size(&mut self) {
        self.h.mi_cols = 2 * ((self.h.frame_width as usize + 7) >> 3);
        self.h.mi_rows = 2 * ((self.h.frame_height as usize + 7) >> 3);
    }

    fn tile_info(&mut self) -> Result<()> {
        let seq = self.seq;
        let mi_cols = self.h.mi_cols as u32;
        let mi_rows = self.h.mi_rows as u32;
        let sb128 = seq.use_128x128_superblock;
        let sb_cols = if sb128 {
            (mi_cols + 31) >> 5
        } else {
            (mi_cols + 15) >> 4
        };
        let sb_rows = if sb128 {
            (mi_rows + 31) >> 5
        } else {
            (mi_rows + 15) >> 4
        };
        let sb_shift = if sb128 { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
        let t = &mut self.h.tile_info;
        t.mi_col_starts.clear();
        t.mi_row_starts.clear();
        let uniform = self.r.flag()?;
        if uniform {
            t.cols_log2 = min_log2_tile_cols;
            while t.cols_log2 < max_log2_tile_cols {
                if self.r.flag()? {
                    t.cols_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_width_sb = (sb_cols + (1 << t.cols_log2) - 1) >> t.cols_log2;
            let mut start = 0;
            while start < sb_cols {
                t.mi_col_starts.push((start << sb_shift) as usize);
                start += tile_width_sb;
            }
            t.mi_col_starts.push(mi_cols as usize);
            t.cols = t.mi_col_starts.len() - 1;
            let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.cols_log2);
            t.rows_log2 = min_log2_tile_rows;
            while t.rows_log2 < max_log2_tile_rows {
                if self.r.flag()? {
                    t.rows_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_height_sb = (sb_rows + (1 << t.rows_log2) - 1) >> t.rows_log2;
            let mut start = 0;
            while start < sb_rows {
                t.mi_row_starts.push((start << sb_shift) as usize);
                start += tile_height_sb;
            }
            t.mi_row_starts.push(mi_rows as usize);
            t.rows = t.mi_row_starts.len() - 1;
        } else {
            let mut widest_tile_sb = 0;
            let mut start = 0;
            while start < sb_cols {
                t.mi_col_starts.push((start << sb_shift) as usize);
                let max_width = (sb_cols - start).min(max_tile_width_sb);
                let size = self.r.ns(max_width)? + 1;
                widest_tile_sb = widest_tile_sb.max(size);
                start += size;
            }
            t.mi_col_starts.push(mi_cols as usize);
            t.cols = t.mi_col_starts.len() - 1;
            t.cols_log2 = tile_log2(1, t.cols as u32);
            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
            let mut start = 0;
            while start < sb_rows {
                t.mi_row_starts.push((start << sb_shift) as usize);
                let max_height = (sb_rows - start).min(max_tile_height_sb);
                let size = self.r.ns(max_height)? + 1;
                start += size;
            }
            t.mi_row_starts.push(mi_rows as usize);
            t.rows = t.mi_row_starts.len() - 1;
            t.rows_log2 = tile_log2(1, t.rows as u32);
        }
        if t.cols_log2 > 0 || t.rows_log2 > 0 {
            t.context_update_tile_id = self.r.f(t.rows_log2 + t.cols_log2)? as usize;
            t.tile_size_bytes = self.r.f(2)? + 1;
        } else {
            t.context_update_tile_id = 0;
            t.tile_size_bytes = 4;
        }
        if t.context_update_tile_id >= t.cols * t.rows {
            return Err(Error::bitstream("context_update_tile_id out of range"));
        }
        Ok(())
    }

    fn read_delta_q(&mut self) -> Result<i32> {
        if self.r.flag()? { self.r.su(7) } else { Ok(0) }
    }

    fn quantization_params(&mut self) -> Result<()> {
        let c = &self.seq.color;
        self.h.base_q_idx = self.r.f(8)?;
        self.h.delta_q_y_dc = self.read_delta_q()?;
        if c.num_planes > 1 {
            let diff_uv_delta = if c.separate_uv_delta_q {
                self.r.flag()?
            } else {
                false
            };
            self.h.delta_q_u_dc = self.read_delta_q()?;
            self.h.delta_q_u_ac = self.read_delta_q()?;
            if diff_uv_delta {
                self.h.delta_q_v_dc = self.read_delta_q()?;
                self.h.delta_q_v_ac = self.read_delta_q()?;
            } else {
                self.h.delta_q_v_dc = self.h.delta_q_u_dc;
                self.h.delta_q_v_ac = self.h.delta_q_u_ac;
            }
        } else {
            self.h.delta_q_u_dc = 0;
            self.h.delta_q_u_ac = 0;
            self.h.delta_q_v_dc = 0;
            self.h.delta_q_v_ac = 0;
        }
        self.h.using_qmatrix = self.r.flag()?;
        if self.h.using_qmatrix {
            self.h.qm_y = self.r.f(4)?;
            self.h.qm_u = self.r.f(4)?;
            self.h.qm_v = if !c.separate_uv_delta_q {
                self.h.qm_u
            } else {
                self.r.f(4)?
            };
        }
        Ok(())
    }

    fn segmentation_params(&mut self) -> Result<()> {
        self.h.segmentation_enabled = self.r.flag()?;
        if self.h.segmentation_enabled {
            if self.h.primary_ref_frame == PRIMARY_REF_NONE {
                self.h.segmentation_update_map = true;
                self.h.segmentation_temporal_update = false;
                self.h.segmentation_update_data = true;
            } else {
                self.h.segmentation_update_map = self.r.flag()?;
                self.h.segmentation_temporal_update = if self.h.segmentation_update_map {
                    self.r.flag()?
                } else {
                    false
                };
                self.h.segmentation_update_data = self.r.flag()?;
            }
            if self.h.segmentation_update_data {
                for i in 0..MAX_SEGMENTS {
                    for j in 0..SEG_LVL_MAX {
                        let enabled = self.r.flag()?;
                        self.h.feature_enabled[i][j] = enabled;
                        let mut clipped = 0;
                        if enabled {
                            let bits = SEGMENTATION_FEATURE_BITS[j] as u32;
                            let limit = SEGMENTATION_FEATURE_MAX[j];
                            if SEGMENTATION_FEATURE_SIGNED[j] == 1 {
                                let v = self.r.su(1 + bits)?;
                                clipped = clip3(-limit, limit, v);
                            } else {
                                let v = self.r.f(bits)? as i32;
                                clipped = clip3(0, limit, v);
                            }
                        }
                        self.h.feature_data[i][j] = clipped;
                    }
                }
            }
        } else {
            self.h.feature_enabled = [[false; 8]; 8];
            self.h.feature_data = [[0; 8]; 8];
            self.h.segmentation_update_map = false;
            self.h.segmentation_temporal_update = false;
            self.h.segmentation_update_data = false;
        }
        self.h.seg_id_pre_skip = false;
        self.h.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if self.h.feature_enabled[i][j] {
                    self.h.last_active_seg_id = i;
                    if j >= SEG_LVL_REF_FRAME {
                        self.h.seg_id_pre_skip = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn loop_filter_params(&mut self) -> Result<()> {
        let h = &mut self.h;
        if h.coded_lossless || h.allow_intrabc {
            h.loop_filter_level[0] = 0;
            h.loop_filter_level[1] = 0;
            h.loop_filter_ref_deltas = DEFAULT_REF_DELTAS;
            h.loop_filter_mode_deltas = [0; 2];
            return Ok(());
        }
        h.loop_filter_level[0] = self.r.f(6)? as i32;
        self.h.loop_filter_level[1] = self.r.f(6)? as i32;
        if self.seq.color.num_planes > 1
            && (self.h.loop_filter_level[0] != 0 || self.h.loop_filter_level[1] != 0)
        {
            self.h.loop_filter_level[2] = self.r.f(6)? as i32;
            self.h.loop_filter_level[3] = self.r.f(6)? as i32;
        }
        self.h.loop_filter_sharpness = self.r.f(3)? as i32;
        self.h.loop_filter_delta_enabled = self.r.flag()?;
        if self.h.loop_filter_delta_enabled && self.r.flag()? {
            for i in 0..TOTAL_REFS_PER_FRAME {
                if self.r.flag()? {
                    self.h.loop_filter_ref_deltas[i] = self.r.su(7)?;
                }
            }
            for i in 0..2 {
                if self.r.flag()? {
                    self.h.loop_filter_mode_deltas[i] = self.r.su(7)?;
                }
            }
        }
        Ok(())
    }

    fn cdef_params(&mut self) -> Result<()> {
        let h = &mut self.h;
        if h.coded_lossless || h.allow_intrabc || !self.seq.enable_cdef {
            h.cdef_bits = 0;
            h.cdef_y_pri_strength[0] = 0;
            h.cdef_y_sec_strength[0] = 0;
            h.cdef_uv_pri_strength[0] = 0;
            h.cdef_uv_sec_strength[0] = 0;
            h.cdef_damping = 3;
            return Ok(());
        }
        h.cdef_damping = self.r.f(2)? as i32 + 3;
        self.h.cdef_bits = self.r.f(2)?;
        for i in 0..(1usize << self.h.cdef_bits) {
            self.h.cdef_y_pri_strength[i] = self.r.f(4)? as i32;
            let mut s = self.r.f(2)? as i32;
            if s == 3 {
                s += 1;
            }
            self.h.cdef_y_sec_strength[i] = s;
            if self.seq.color.num_planes > 1 {
                self.h.cdef_uv_pri_strength[i] = self.r.f(4)? as i32;
                let mut s = self.r.f(2)? as i32;
                if s == 3 {
                    s += 1;
                }
                self.h.cdef_uv_sec_strength[i] = s;
            }
        }
        Ok(())
    }

    fn lr_params(&mut self) -> Result<()> {
        let seq = self.seq;
        let h = &mut self.h;
        if h.all_lossless || h.allow_intrabc || !seq.enable_restoration {
            h.frame_restoration_type = [RESTORE_NONE; 3];
            h.uses_lr = false;
            return Ok(());
        }
        h.uses_lr = false;
        let mut uses_chroma_lr = false;
        for i in 0..seq.color.num_planes {
            let lr_type = self.r.f(2)? as usize;
            let t = REMAP_LR_TYPE[lr_type] as u8;
            self.h.frame_restoration_type[i] = t;
            if t != RESTORE_NONE {
                self.h.uses_lr = true;
                if i > 0 {
                    uses_chroma_lr = true;
                }
            }
        }
        if self.h.uses_lr {
            let mut lr_unit_shift;
            if seq.use_128x128_superblock {
                lr_unit_shift = self.r.f(1)?;
                lr_unit_shift += 1;
            } else {
                lr_unit_shift = self.r.f(1)?;
                if lr_unit_shift != 0 {
                    lr_unit_shift += self.r.f(1)?;
                }
            }
            let s0 = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);
            let lr_uv_shift =
                if seq.color.subsampling_x != 0 && seq.color.subsampling_y != 0 && uses_chroma_lr {
                    self.r.f(1)?
                } else {
                    0
                };
            self.h.loop_restoration_size = [s0, s0 >> lr_uv_shift, s0 >> lr_uv_shift];
        }
        Ok(())
    }

    fn skip_mode_params(&mut self) -> Result<()> {
        let seq = self.seq;
        let h = &mut self.h;
        let mut skip_mode_allowed = false;
        if !(h.frame_is_intra || !h.reference_select || !seq.enable_order_hint) {
            let mut forward_idx = -1i32;
            let mut backward_idx = -1i32;
            let (mut forward_hint, mut backward_hint) = (0u32, 0u32);
            for i in 0..REFS_PER_FRAME {
                let ref_hint = self.st.order_hint[h.ref_frame_idx[i]];
                if get_relative_dist(seq, ref_hint, h.order_hint) < 0 {
                    if forward_idx < 0 || get_relative_dist(seq, ref_hint, forward_hint) > 0 {
                        forward_idx = i as i32;
                        forward_hint = ref_hint;
                    }
                } else if get_relative_dist(seq, ref_hint, h.order_hint) > 0
                    && (backward_idx < 0 || get_relative_dist(seq, ref_hint, backward_hint) < 0)
                {
                    backward_idx = i as i32;
                    backward_hint = ref_hint;
                }
            }
            if forward_idx < 0 {
                skip_mode_allowed = false;
            } else if backward_idx >= 0 {
                skip_mode_allowed = true;
                h.skip_mode_frame = [
                    LAST_FRAME + forward_idx.min(backward_idx),
                    LAST_FRAME + forward_idx.max(backward_idx),
                ];
            } else {
                let mut second_forward_idx = -1i32;
                let mut second_forward_hint = 0u32;
                for i in 0..REFS_PER_FRAME {
                    let ref_hint = self.st.order_hint[h.ref_frame_idx[i]];
                    if get_relative_dist(seq, ref_hint, forward_hint) < 0
                        && (second_forward_idx < 0
                            || get_relative_dist(seq, ref_hint, second_forward_hint) > 0)
                    {
                        second_forward_idx = i as i32;
                        second_forward_hint = ref_hint;
                    }
                }
                if second_forward_idx >= 0 {
                    skip_mode_allowed = true;
                    h.skip_mode_frame = [
                        LAST_FRAME + forward_idx.min(second_forward_idx),
                        LAST_FRAME + forward_idx.max(second_forward_idx),
                    ];
                }
            }
        }
        self.h.skip_mode_present = if skip_mode_allowed {
            self.r.flag()?
        } else {
            false
        };
        Ok(())
    }

    fn global_motion_params(&mut self) -> Result<()> {
        for rf in LAST_FRAME as usize..=ALTREF_FRAME as usize {
            self.h.gm_type[rf] = IDENTITY;
            for i in 0..6 {
                self.h.gm_params[rf][i] = if i % 3 == 2 {
                    1 << WARPEDMODEL_PREC_BITS
                } else {
                    0
                };
            }
        }
        if self.h.frame_is_intra {
            return Ok(());
        }
        for rf in LAST_FRAME as usize..=ALTREF_FRAME as usize {
            let ty = if self.r.flag()? {
                if self.r.flag()? {
                    ROTZOOM
                } else if self.r.flag()? {
                    TRANSLATION
                } else {
                    AFFINE
                }
            } else {
                IDENTITY
            };
            self.h.gm_type[rf] = ty;
            if ty >= ROTZOOM {
                self.read_global_param(ty, rf, 2)?;
                self.read_global_param(ty, rf, 3)?;
                if ty == AFFINE {
                    self.read_global_param(ty, rf, 4)?;
                    self.read_global_param(ty, rf, 5)?;
                } else {
                    self.h.gm_params[rf][4] = -self.h.gm_params[rf][3];
                    self.h.gm_params[rf][5] = self.h.gm_params[rf][2];
                }
            }
            if ty >= TRANSLATION {
                self.read_global_param(ty, rf, 0)?;
                self.read_global_param(ty, rf, 1)?;
            }
        }
        Ok(())
    }

    fn read_global_param(&mut self, ty: u32, rf: usize, idx: usize) -> Result<()> {
        let mut abs_bits = GM_ABS_ALPHA_BITS;
        let mut prec_bits = GM_ALPHA_PREC_BITS;
        if idx < 2 {
            if ty == TRANSLATION {
                let hp = !self.h.allow_high_precision_mv as u32;
                abs_bits = GM_ABS_TRANS_ONLY_BITS - hp;
                prec_bits = GM_TRANS_ONLY_PREC_BITS - hp;
            } else {
                abs_bits = GM_ABS_TRANS_BITS;
                prec_bits = GM_TRANS_PREC_BITS;
            }
        }
        let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
        let round = if idx % 3 == 2 {
            1i32 << WARPEDMODEL_PREC_BITS
        } else {
            0
        };
        let sub = if idx % 3 == 2 { 1i32 << prec_bits } else { 0 };
        let mx = 1i32 << abs_bits;
        let r = (self.prev_gm_params[rf][idx] >> prec_diff) - sub;
        let v = self.decode_signed_subexp_with_ref(-mx, mx + 1, r)?;
        self.h.gm_params[rf][idx] = (v << prec_diff) + round;
        Ok(())
    }

    fn decode_signed_subexp_with_ref(&mut self, low: i32, high: i32, r: i32) -> Result<i32> {
        let x = self.decode_unsigned_subexp_with_ref(high - low, r - low)?;
        Ok(x + low)
    }

    fn decode_unsigned_subexp_with_ref(&mut self, mx: i32, r: i32) -> Result<i32> {
        let v = self.decode_subexp(mx)?;
        if (r << 1) <= mx {
            Ok(inverse_recenter(r, v))
        } else {
            Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
        }
    }

    fn decode_subexp(&mut self, num_syms: i32) -> Result<i32> {
        let mut i = 0;
        let mut mk = 0;
        let k = 3;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1 << b2;
            if num_syms <= mk + 3 * a {
                let v = self.r.ns((num_syms - mk) as u32)? as i32;
                return Ok(v + mk);
            } else if self.r.flag()? {
                i += 1;
                mk += a;
            } else {
                let v = self.r.f(b2 as u32)? as i32;
                return Ok(v + mk);
            }
        }
    }

    fn film_grain_params(&mut self) -> Result<()> {
        let seq = self.seq;
        if !seq.film_grain_params_present || (!self.h.show_frame && !self.h.showable_frame) {
            self.h.film_grain = FilmGrainParams::default();
            return Ok(());
        }
        let mut g = FilmGrainParams {
            apply_grain: self.r.flag()?,
            ..Default::default()
        };
        if !g.apply_grain {
            self.h.film_grain = FilmGrainParams::default();
            return Ok(());
        }
        g.grain_seed = self.r.f(16)?;
        g.update_grain = if self.h.frame_type == INTER_FRAME {
            self.r.flag()?
        } else {
            true
        };
        if !g.update_grain {
            let film_grain_params_ref_idx = self.r.f(3)? as usize;
            self.h.film_grain_params_ref_idx = Some(film_grain_params_ref_idx);
            let temp_seed = g.grain_seed;
            let slot = self.slot(film_grain_params_ref_idx)?;
            g = slot.film_grain.clone();
            g.grain_seed = temp_seed;
            self.h.film_grain = g;
            return Ok(());
        }
        g.num_y_points = self.r.f(4)? as usize;
        if g.num_y_points > 14 {
            return Err(Error::bitstream("num_y_points > 14"));
        }
        for i in 0..g.num_y_points {
            g.point_y_value[i] = self.r.f(8)? as i32;
            g.point_y_scaling[i] = self.r.f(8)? as i32;
        }
        increasing(&g.point_y_value[..g.num_y_points], "point_y_value")?;
        let c = &seq.color;
        g.chroma_scaling_from_luma = if c.mono_chrome { false } else { self.r.flag()? };
        if c.mono_chrome
            || g.chroma_scaling_from_luma
            || (c.subsampling_x == 1 && c.subsampling_y == 1 && g.num_y_points == 0)
        {
            g.num_cb_points = 0;
            g.num_cr_points = 0;
        } else {
            g.num_cb_points = self.r.f(4)? as usize;
            if g.num_cb_points > 10 {
                return Err(Error::bitstream("num_cb_points > 10"));
            }
            for i in 0..g.num_cb_points {
                g.point_cb_value[i] = self.r.f(8)? as i32;
                g.point_cb_scaling[i] = self.r.f(8)? as i32;
            }
            g.num_cr_points = self.r.f(4)? as usize;
            if g.num_cr_points > 10 {
                return Err(Error::bitstream("num_cr_points > 10"));
            }
            for i in 0..g.num_cr_points {
                g.point_cr_value[i] = self.r.f(8)? as i32;
                g.point_cr_scaling[i] = self.r.f(8)? as i32;
            }
            increasing(&g.point_cb_value[..g.num_cb_points], "point_cb_value")?;
            increasing(&g.point_cr_value[..g.num_cr_points], "point_cr_value")?;
        }
        g.grain_scaling_minus_8 = self.r.f(2)?;
        g.ar_coeff_lag = self.r.f(2)? as i32;
        let num_pos_luma = (2 * g.ar_coeff_lag * (g.ar_coeff_lag + 1)) as usize;
        let num_pos_chroma;
        if g.num_y_points != 0 {
            num_pos_chroma = num_pos_luma + 1;
            for i in 0..num_pos_luma {
                g.ar_coeffs_y_plus_128[i] = self.r.f(8)? as i32;
            }
        } else {
            num_pos_chroma = num_pos_luma;
        }
        if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cb_plus_128[i] = self.r.f(8)? as i32;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cr_plus_128[i] = self.r.f(8)? as i32;
            }
        }
        g.ar_coeff_shift_minus_6 = self.r.f(2)?;
        g.grain_scale_shift = self.r.f(2)?;
        if g.num_cb_points != 0 {
            g.cb_mult = self.r.f(8)? as i32;
            g.cb_luma_mult = self.r.f(8)? as i32;
            g.cb_offset = self.r.f(9)? as i32;
        }
        if g.num_cr_points != 0 {
            g.cr_mult = self.r.f(8)? as i32;
            g.cr_luma_mult = self.r.f(8)? as i32;
            g.cr_offset = self.r.f(9)? as i32;
        }
        g.overlap_flag = self.r.flag()?;
        g.clip_to_restricted_range = self.r.flag()?;
        self.h.film_grain = g;
        Ok(())
    }
}

/// `inverse_recenter( r, v )`.
pub(crate) fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `tile_log2( blkSize, target )`.
pub(crate) fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// `get_qindex( 1, segmentId )`: the segment's quantiser index ignoring
/// the block-level delta.
pub(crate) fn get_qindex_header(h: &FrameHeader, segment_id: usize) -> u32 {
    if h.segmentation_enabled && h.feature_enabled[segment_id][SEG_LVL_ALT_Q] {
        let data = h.feature_data[segment_id][SEG_LVL_ALT_Q];
        clip3(0, 255, h.base_q_idx as i32 + data) as u32
    } else {
        h.base_q_idx
    }
}

/// The film grain points' values must increase: bitstream conformance
/// requires each `point_y_value` / `point_cb_value` / `point_cr_value` to be
/// greater than the one before (6.8.20), and the scaling function's
/// interpolation (7.18.3.5) divides by their differences.
fn increasing(values: &[i32], name: &str) -> Result<()> {
    if values.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::bitstream(format!("{name} does not increase")));
    }
    Ok(())
}
