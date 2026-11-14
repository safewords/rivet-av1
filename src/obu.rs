//! OBUs (5.3): splitting a temporal unit into its open bitstream units, and
//! the sequence header OBU (5.5, 6.4).

use crate::bits::{BitReader, BitWriter};
use crate::consts::*;
use crate::{Error, Result};

/// One OBU: its header fields and payload.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Obu<'a> {
    pub(crate) obu_type: u32,
    pub(crate) temporal_id: u32,
    pub(crate) spatial_id: u32,
    pub(crate) has_extension: bool,
    pub(crate) payload: &'a [u8],
}

/// Why [`read_obu`] found no whole OBU.
pub(crate) enum ObuError<'a> {
    /// The OBU's header is there, but `obu_size` says more bytes than
    /// follow: the OBU with the bytes there are (the data was cut short,
    /// or `obu_size` itself is wrong).
    Truncated(Obu<'a>),
    /// Not even the OBU's header reads.
    Malformed(Error),
}

/// The error of a [`ObuError::Truncated`] OBU.
pub(crate) fn truncated() -> Error {
    Error::bitstream("obu_size runs past the end of the data")
}

/// The OBU at the start of `data` (`open_bitstream_unit()`, 5.3.1), and
/// the number of bytes it takes.
pub(crate) fn read_obu(data: &[u8]) -> std::result::Result<(Obu<'_>, usize), ObuError<'_>> {
    let mut r = BitReader::new(data);
    let header = (|| {
        let forbidden = r.f(1)?;
        if forbidden != 0 {
            return Err(Error::bitstream("obu_forbidden_bit set"));
        }
        let obu_type = r.f(4)?;
        let ext = r.flag()?;
        let has_size = r.flag()?;
        let _reserved = r.f(1)?;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if ext {
            temporal_id = r.f(3)?;
            spatial_id = r.f(2)?;
            r.f(3)?;
        }
        let size = if has_size {
            Some(r.leb128()? as usize)
        } else {
            None
        };
        Ok((obu_type, ext, temporal_id, spatial_id, size))
    })();
    let (obu_type, ext, temporal_id, spatial_id, size) = header.map_err(ObuError::Malformed)?;
    let start = r.position() / 8;
    let size = size.unwrap_or(data.len() - start);
    let end = start.checked_add(size).filter(|&e| e <= data.len());
    let obu = Obu {
        obu_type,
        temporal_id,
        spatial_id,
        has_extension: ext,
        payload: &data[start..end.unwrap_or(data.len())],
    };
    match end {
        Some(end) => Ok((obu, end)),
        None => Err(ObuError::Truncated(obu)),
    }
}

/// Writes an OBU with `obu_has_size_field` set and no extension.
pub(crate) fn write_obu(out: &mut Vec<u8>, obu_type: u32, payload: &[u8]) {
    out.push(((obu_type as u8) << 3) | 0b010);
    crate::bits::write_leb128(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// `color_config()` (5.5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ColorConfig {
    pub(crate) bit_depth: u32,
    pub(crate) mono_chrome: bool,
    pub(crate) num_planes: usize,
    pub(crate) color_primaries: u32,
    pub(crate) transfer_characteristics: u32,
    pub(crate) matrix_coefficients: u32,
    pub(crate) color_range: bool,
    pub(crate) subsampling_x: u32,
    pub(crate) subsampling_y: u32,
    pub(crate) chroma_sample_position: u32,
    pub(crate) separate_uv_delta_q: bool,
    pub(crate) color_description_present: bool,
}

/// `sequence_header_obu()` (5.5.1), the fields the decoding process uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceHeader {
    pub(crate) seq_profile: u32,
    pub(crate) still_picture: bool,
    pub(crate) reduced_still_picture_header: bool,
    pub(crate) timing_info_present: bool,
    pub(crate) decoder_model_info_present: bool,
    pub(crate) equal_picture_interval: bool,
    pub(crate) buffer_removal_time_length_minus_1: u32,
    pub(crate) frame_presentation_time_length_minus_1: u32,
    pub(crate) operating_points_cnt_minus_1: usize,
    pub(crate) operating_point_idc: Vec<u32>,
    pub(crate) seq_level_idx: Vec<u32>,
    pub(crate) seq_tier: Vec<u32>,
    pub(crate) decoder_model_present_for_this_op: Vec<bool>,
    /// `timing_info()`: `num_units_in_display_tick`, `time_scale` and,
    /// with `equal_picture_interval`, `num_ticks_per_picture_minus_1 + 1`
    /// (0 without timing information). The level limits on rates (A.3)
    /// are measured with them.
    pub(crate) num_units_in_display_tick: u32,
    pub(crate) time_scale: u32,
    pub(crate) num_ticks_per_picture: u64,
    /// The operating point chosen (`choose_operating_point()`).
    pub(crate) op: usize,
    /// `OperatingPointIdc` of the chosen operating point (0).
    pub(crate) op_idc: u32,
    pub(crate) frame_width_bits: u32,
    pub(crate) frame_height_bits: u32,
    pub(crate) max_frame_width: u32,
    pub(crate) max_frame_height: u32,
    pub(crate) frame_id_numbers_present: bool,
    pub(crate) delta_frame_id_length_minus_2: u32,
    pub(crate) additional_frame_id_length_minus_1: u32,
    pub(crate) use_128x128_superblock: bool,
    pub(crate) enable_filter_intra: bool,
    pub(crate) enable_intra_edge_filter: bool,
    pub(crate) enable_interintra_compound: bool,
    pub(crate) enable_masked_compound: bool,
    pub(crate) enable_warped_motion: bool,
    pub(crate) enable_dual_filter: bool,
    pub(crate) enable_order_hint: bool,
    pub(crate) enable_jnt_comp: bool,
    pub(crate) enable_ref_frame_mvs: bool,
    pub(crate) seq_force_screen_content_tools: u32,
    pub(crate) seq_force_integer_mv: u32,
    pub(crate) order_hint_bits: u32,
    pub(crate) enable_superres: bool,
    pub(crate) enable_cdef: bool,
    pub(crate) enable_restoration: bool,
    pub(crate) color: ColorConfig,
    pub(crate) film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Parses a sequence header, choosing operating point `op` (clamped to
    /// the ones the header declares).
    pub(crate) fn parse(data: &[u8], op: usize) -> Result<Self> {
        let mut r = BitReader::new(data);
        let seq_profile = r.f(3)?;
        if seq_profile > 2 {
            return Err(Error::unsupported(format!("seq_profile {seq_profile}")));
        }
        let still_picture = r.flag()?;
        let reduced = r.flag()?;
        let mut timing_info_present = false;
        let mut decoder_model_info_present = false;
        let mut equal_picture_interval = false;
        let mut buffer_delay_length_minus_1 = 0;
        let mut buffer_removal_time_length_minus_1 = 0;
        let mut frame_presentation_time_length_minus_1 = 0;
        let mut operating_point_idc = vec![0];
        let mut seq_level_idx = vec![0];
        let mut seq_tier = vec![0];
        let (mut num_units_in_display_tick, mut time_scale, mut num_ticks_per_picture) = (0, 0, 0);
        let mut decoder_model_present_for_this_op = vec![false];
        let mut operating_points_cnt_minus_1 = 0;
        if reduced {
            seq_level_idx[0] = r.f(5)?;
        } else {
            timing_info_present = r.flag()?;
            if timing_info_present {
                num_units_in_display_tick = r.f(32)?;
                time_scale = r.f(32)?;
                equal_picture_interval = r.flag()?;
                if equal_picture_interval {
                    num_ticks_per_picture = r.uvlc()? as u64 + 1;
                }
                decoder_model_info_present = r.flag()?;
                if decoder_model_info_present {
                    buffer_delay_length_minus_1 = r.f(5)?;
                    r.f(32)?; // num_units_in_decoding_tick
                    buffer_removal_time_length_minus_1 = r.f(5)?;
                    frame_presentation_time_length_minus_1 = r.f(5)?;
                }
            }
            let initial_display_delay_present = r.flag()?;
            operating_points_cnt_minus_1 = r.f(5)? as usize;
            operating_point_idc = Vec::new();
            seq_level_idx = Vec::new();
            seq_tier = Vec::new();
            decoder_model_present_for_this_op = Vec::new();
            for _ in 0..=operating_points_cnt_minus_1 {
                operating_point_idc.push(r.f(12)?);
                let lvl = r.f(5)?;
                seq_level_idx.push(lvl);
                seq_tier.push(if lvl > 7 { r.f(1)? } else { 0 });
                let mut present = false;
                if decoder_model_info_present {
                    present = r.flag()?;
                    if present {
                        let n = buffer_delay_length_minus_1 + 1;
                        r.f(n)?; // decoder_buffer_delay
                        r.f(n)?; // encoder_buffer_delay
                        r.f(1)?; // low_delay_mode_flag
                    }
                }
                decoder_model_present_for_this_op.push(present);
                if initial_display_delay_present && r.flag()? {
                    r.f(4)?;
                }
            }
        }
        // choose_operating_point()
        let op = op.min(operating_points_cnt_minus_1);
        let op_idc = operating_point_idc[op];
        let frame_width_bits = r.f(4)? + 1;
        let frame_height_bits = r.f(4)? + 1;
        let max_frame_width = r.f(frame_width_bits)? + 1;
        let max_frame_height = r.f(frame_height_bits)? + 1;
        let frame_id_numbers_present = if reduced { false } else { r.flag()? };
        let (mut delta_frame_id_length_minus_2, mut additional_frame_id_length_minus_1) = (0, 0);
        if frame_id_numbers_present {
            delta_frame_id_length_minus_2 = r.f(4)?;
            additional_frame_id_length_minus_1 = r.f(3)?;
        }
        let use_128x128_superblock = r.flag()?;
        let enable_filter_intra = r.flag()?;
        let enable_intra_edge_filter = r.flag()?;
        let mut s = SequenceHeader {
            seq_profile,
            still_picture,
            reduced_still_picture_header: reduced,
            timing_info_present,
            decoder_model_info_present,
            equal_picture_interval,
            buffer_removal_time_length_minus_1,
            frame_presentation_time_length_minus_1,
            operating_points_cnt_minus_1,
            operating_point_idc,
            seq_level_idx,
            seq_tier,
            decoder_model_present_for_this_op,
            num_units_in_display_tick,
            time_scale,
            num_ticks_per_picture,
            op,
            op_idc,
            frame_width_bits,
            frame_height_bits,
            max_frame_width,
            max_frame_height,
            frame_id_numbers_present,
            delta_frame_id_length_minus_2,
            additional_frame_id_length_minus_1,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: false,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS,
            seq_force_integer_mv: SELECT_INTEGER_MV,
            order_hint_bits: 0,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            color: ColorConfig {
                bit_depth: 8,
                mono_chrome: false,
                num_planes: 3,
                color_primaries: CP_UNSPECIFIED,
                transfer_characteristics: TC_UNSPECIFIED,
                matrix_coefficients: MC_UNSPECIFIED,
                color_range: false,
                subsampling_x: 1,
                subsampling_y: 1,
                chroma_sample_position: CSP_UNKNOWN,
                separate_uv_delta_q: false,
                color_description_present: false,
            },
            film_grain_params_present: false,
        };
        if !reduced {
            s.enable_interintra_compound = r.flag()?;
            s.enable_masked_compound = r.flag()?;
            s.enable_warped_motion = r.flag()?;
            s.enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                s.enable_jnt_comp = r.flag()?;
                s.enable_ref_frame_mvs = r.flag()?;
            }
            let seq_choose_screen_content_tools = r.flag()?;
            s.seq_force_screen_content_tools = if seq_choose_screen_content_tools {
                SELECT_SCREEN_CONTENT_TOOLS
            } else {
                r.f(1)?
            };
            s.seq_force_integer_mv = if s.seq_force_screen_content_tools > 0 {
                if r.flag()? {
                    SELECT_INTEGER_MV
                } else {
                    r.f(1)?
                }
            } else {
                SELECT_INTEGER_MV
            };
            if s.enable_order_hint {
                s.order_hint_bits = r.f(3)? + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.color = parse_color_config(&mut r, seq_profile)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }

    /// Writes this header as a sequence header OBU payload (trailing bits
    /// included). Only the fields the encoder sets are written; timing and
    /// decoder model information are not.
    pub(crate) fn write(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.f(3, self.seq_profile);
        w.flag(self.still_picture);
        w.flag(self.reduced_still_picture_header);
        if self.reduced_still_picture_header {
            w.f(5, self.seq_level_idx[0]);
        } else {
            w.flag(false); // timing_info_present_flag
            w.flag(false); // initial_display_delay_present_flag
            w.f(5, 0); // operating_points_cnt_minus_1
            w.f(12, self.operating_point_idc[0]);
            w.f(5, self.seq_level_idx[0]);
            if self.seq_level_idx[0] > 7 {
                w.f(1, 0);
            }
        }
        w.f(4, self.frame_width_bits - 1);
        w.f(4, self.frame_height_bits - 1);
        w.f(self.frame_width_bits, self.max_frame_width - 1);
        w.f(self.frame_height_bits, self.max_frame_height - 1);
        if !self.reduced_still_picture_header {
            w.flag(self.frame_id_numbers_present);
            if self.frame_id_numbers_present {
                w.f(4, self.delta_frame_id_length_minus_2);
                w.f(3, self.additional_frame_id_length_minus_1);
            }
        }
        w.flag(self.use_128x128_superblock);
        w.flag(self.enable_filter_intra);
        w.flag(self.enable_intra_edge_filter);
        if !self.reduced_still_picture_header {
            w.flag(self.enable_interintra_compound);
            w.flag(self.enable_masked_compound);
            w.flag(self.enable_warped_motion);
            w.flag(self.enable_dual_filter);
            w.flag(self.enable_order_hint);
            if self.enable_order_hint {
                w.flag(self.enable_jnt_comp);
                w.flag(self.enable_ref_frame_mvs);
            }
            if self.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
                w.flag(true);
            } else {
                w.flag(false);
                w.f(1, self.seq_force_screen_content_tools);
            }
            if self.seq_force_screen_content_tools > 0 {
                if self.seq_force_integer_mv == SELECT_INTEGER_MV {
                    w.flag(true);
                } else {
                    w.flag(false);
                    w.f(1, self.seq_force_integer_mv);
                }
            }
            if self.enable_order_hint {
                w.f(3, self.order_hint_bits - 1);
            }
        }
        w.flag(self.enable_superres);
        w.flag(self.enable_cdef);
        w.flag(self.enable_restoration);
        let c = &self.color;
        let high = c.bit_depth > 8;
        w.flag(high);
        if self.seq_profile == 2 && high {
            w.flag(c.bit_depth == 12);
        }
        if self.seq_profile != 1 {
            w.flag(c.mono_chrome);
        }
        w.flag(c.color_description_present);
        if c.color_description_present {
            w.f(8, c.color_primaries);
            w.f(8, c.transfer_characteristics);
            w.f(8, c.matrix_coefficients);
        }
        if c.mono_chrome {
            w.flag(c.color_range);
        } else if c.color_primaries == CP_BT_709
            && c.transfer_characteristics == TC_SRGB
            && c.matrix_coefficients == MC_IDENTITY
        {
        } else {
            w.flag(c.color_range);
            if self.seq_profile == 2 && c.bit_depth == 12 {
                w.f(1, c.subsampling_x);
                if c.subsampling_x != 0 {
                    w.f(1, c.subsampling_y);
                }
            }
            if c.subsampling_x != 0 && c.subsampling_y != 0 {
                w.f(2, c.chroma_sample_position);
            }
        }
        if !c.mono_chrome {
            w.flag(c.separate_uv_delta_q);
        }
        w.flag(self.film_grain_params_present);
        w.trailing_bits();
        w.finish()
    }
}

fn parse_color_config(r: &mut BitReader, seq_profile: u32) -> Result<ColorConfig> {
    let high_bitdepth = r.flag()?;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if r.flag()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let mono_chrome = if seq_profile == 1 { false } else { r.flag()? };
    let color_description_present = r.flag()?;
    let (cp, tc, mc) = if color_description_present {
        (r.f(8)?, r.f(8)?, r.f(8)?)
    } else {
        (CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED)
    };
    let mut c = ColorConfig {
        bit_depth,
        mono_chrome,
        num_planes: if mono_chrome { 1 } else { 3 },
        color_primaries: cp,
        transfer_characteristics: tc,
        matrix_coefficients: mc,
        color_range: false,
        subsampling_x: 1,
        subsampling_y: 1,
        chroma_sample_position: CSP_UNKNOWN,
        separate_uv_delta_q: false,
        color_description_present,
    };
    if mono_chrome {
        c.color_range = r.flag()?;
        return Ok(c);
    } else if cp == CP_BT_709 && tc == TC_SRGB && mc == MC_IDENTITY {
        c.color_range = true;
        c.subsampling_x = 0;
        c.subsampling_y = 0;
    } else {
        c.color_range = r.flag()?;
        if seq_profile == 0 {
            c.subsampling_x = 1;
            c.subsampling_y = 1;
        } else if seq_profile == 1 {
            c.subsampling_x = 0;
            c.subsampling_y = 0;
        } else if bit_depth == 12 {
            c.subsampling_x = r.f(1)?;
            c.subsampling_y = if c.subsampling_x != 0 { r.f(1)? } else { 0 };
        } else {
            c.subsampling_x = 1;
            c.subsampling_y = 0;
        }
        if c.subsampling_x != 0 && c.subsampling_y != 0 {
            c.chroma_sample_position = r.f(2)?;
        }
    }
    c.separate_uv_delta_q = r.flag()?;
    Ok(c)
}

/// `metadata_type` of the HDR metadata OBUs (6.7.1).
pub(crate) const METADATA_TYPE_HDR_CLL: u64 = 1;
pub(crate) const METADATA_TYPE_HDR_MDCV: u64 = 2;

/// The payload of a `METADATA_TYPE_HDR_CLL` metadata OBU (5.8.1, 5.8.3).
pub(crate) fn write_hdr_cll(c: &crate::frame::ContentLightLevel) -> Vec<u8> {
    let mut out = Vec::new();
    crate::bits::write_leb128(&mut out, METADATA_TYPE_HDR_CLL);
    let mut w = BitWriter::new();
    w.f(16, c.max_cll as u32);
    w.f(16, c.max_fall as u32);
    w.trailing_bits();
    out.extend_from_slice(&w.finish());
    out
}

/// The payload of a `METADATA_TYPE_HDR_MDCV` metadata OBU (5.8.1, 5.8.4).
pub(crate) fn write_hdr_mdcv(m: &crate::frame::MasteringDisplay) -> Vec<u8> {
    let mut out = Vec::new();
    crate::bits::write_leb128(&mut out, METADATA_TYPE_HDR_MDCV);
    let mut w = BitWriter::new();
    for p in &m.primaries {
        w.f(16, p[0] as u32);
        w.f(16, p[1] as u32);
    }
    w.f(16, m.white_point[0] as u32);
    w.f(16, m.white_point[1] as u32);
    w.f(32, m.luminance_max);
    w.f(32, m.luminance_min);
    w.trailing_bits();
    out.extend_from_slice(&w.finish());
    out
}

/// A metadata OBU the decoder keeps (5.8).
pub(crate) enum Metadata {
    ContentLight(crate::frame::ContentLightLevel),
    MasteringDisplay(crate::frame::MasteringDisplay),
}

/// `metadata_obu()`: the HDR types; others (and ones cut short, which a
/// decoder may ignore as it ignores any metadata) give `None`.
pub(crate) fn parse_metadata(payload: &[u8]) -> Option<Metadata> {
    let mut r = BitReader::new(payload);
    let t = r.leb128().ok()?;
    match t {
        METADATA_TYPE_HDR_CLL => Some(Metadata::ContentLight(crate::frame::ContentLightLevel {
            max_cll: r.f(16).ok()? as u16,
            max_fall: r.f(16).ok()? as u16,
        })),
        METADATA_TYPE_HDR_MDCV => {
            let mut m = crate::frame::MasteringDisplay::default();
            for p in m.primaries.iter_mut() {
                p[0] = r.f(16).ok()? as u16;
                p[1] = r.f(16).ok()? as u16;
            }
            m.white_point = [r.f(16).ok()? as u16, r.f(16).ok()? as u16];
            m.luminance_max = r.f(32).ok()?;
            m.luminance_min = r.f(32).ok()?;
            Some(Metadata::MasteringDisplay(m))
        }
        _ => None,
    }
}
