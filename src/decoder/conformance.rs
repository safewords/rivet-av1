//! Requirements of bitstream conformance that decoding does not need:
//! the frame id rules (6.8.2) and the level limits (A.3).
//!
//! (And `film_grain_params_ref_idx` naming one of the frame's references,
//! 6.8.20, which is of the same kind.) A decoder that is not strict
//! (`Decoder::set_strict`) does not report these, and its output is the same whether a stream keeps them or not.
//! Two of the frame id rules mean more than form, though: a reference
//! whose frame id is not the one the frame expects, or a slot that is no
//! longer valid, is a frame the encoder did not mean, which is how frame
//! ids let a decoder notice a lost frame (their purpose, 6.8.2). Those
//! two the decoder acts on either way: the frame is decoded as damaged
//! (`recovery.rs`), which leaves the output of a stream that keeps them
//! unchanged.

use crate::consts::*;
use crate::header::{FrameHeader, RefState};
use crate::obu::SequenceHeader;

/// A broken requirement: what it is, and whether it says that a frame's
/// reference is not the frame the encoder meant.
pub(crate) struct Violation {
    pub(crate) rule: String,
    pub(crate) lost_reference: bool,
}

impl Violation {
    fn of_form(rule: String) -> Self {
        Violation {
            rule,
            lost_reference: false,
        }
    }

    fn of_reference(rule: String) -> Self {
        Violation {
            rule,
            lost_reference: true,
        }
    }
}

/// `idLen`, the bits of a frame id.
fn id_len(seq: &SequenceHeader) -> u32 {
    seq.additional_frame_id_length_minus_1 + seq.delta_frame_id_length_minus_2 + 3
}

/// The frame id requirements of a frame header (show_existing_frame equal
/// to 0) once parsed against `st`, and the validity of its references:
/// those broken, the rules of the header's own fields first (the first is
/// what a strict decoder reports), then a reference that is not the frame
/// expected (what the decoder acts on either way).
pub(crate) fn frame_ids(seq: &SequenceHeader, h: &FrameHeader, st: &RefState) -> Vec<Violation> {
    let mut out: Vec<Violation> = header_ids(seq, h).into_iter().collect();
    out.extend(references(seq, h, st));
    out
}

/// The rules on a frame header's own frame id fields (and 6.8.20's on
/// `film_grain_params_ref_idx`).
fn header_ids(seq: &SequenceHeader, h: &FrameHeader) -> Option<Violation> {
    if let Some(g) = h.film_grain_params_ref_idx
        && !h.ref_frame_idx.contains(&g)
    {
        return Some(Violation::of_form(format!(
            "film_grain_params_ref_idx {g} is none of the frame's ref_frame_idx (6.8.20)"
        )));
    }
    if !seq.frame_id_numbers_present {
        return None;
    }
    let len = id_len(seq);
    if len > 16 {
        return Some(Violation::of_form(format!(
            "idLen is {len}: the frame ids need more than 16 bits (6.8.2, display_frame_id)"
        )));
    }
    if (h.frame_type != KEY_FRAME || !h.show_frame)
        && let Some(prev) = h.prev_frame_id
    {
        let cur = h.current_frame_id;
        if cur == prev {
            return Some(Violation::of_form(format!(
                "current_frame_id {cur} is equal to PrevFrameID (6.8.2, current_frame_id)"
            )));
        }
        let diff = if cur > prev {
            cur - prev
        } else {
            (1 << len) + cur - prev
        };
        if diff >= 1 << (len - 1) {
            return Some(Violation::of_form(format!(
                "DiffFrameID {diff} (current_frame_id {cur}, PrevFrameID {prev}) is not less \
                 than 1 << ( idLen - 1 ) (6.8.2, current_frame_id)"
            )));
        }
    }
    None
}

/// An inter frame's references: valid, and with frame ids, the frames the
/// header expects.
fn references(seq: &SequenceHeader, h: &FrameHeader, st: &RefState) -> Option<Violation> {
    if h.frame_is_intra {
        return None;
    }
    for i in 0..REFS_PER_FRAME {
        let idx = h.ref_frame_idx[i];
        if !st.valid[idx] {
            return Some(Violation::of_reference(format!(
                "RefValid[ ref_frame_idx[ {i} ] ] is 0 (slot {idx}): the frame refers to a slot \
                 that holds no valid frame (6.8.2, ref_frame_idx)"
            )));
        }
        if seq.frame_id_numbers_present {
            let held = st.slots[idx].as_ref().map_or(0, |s| s.frame_id);
            if held != h.expected_frame_id[i] {
                return Some(Violation::of_reference(format!(
                    "expectedFrameId[ {i} ] is {} but RefFrameId[ ref_frame_idx[ {i} ] ] (slot \
                     {idx}) is {held} (6.8.2, delta_frame_id_minus_1)",
                    h.expected_frame_id[i]
                )));
            }
        }
    }
    None
}

/// The frame id requirements of a `show_existing_frame` header.
pub(crate) fn show_existing_ids(
    seq: &SequenceHeader,
    h: &FrameHeader,
    st: &RefState,
) -> Option<Violation> {
    if !seq.frame_id_numbers_present {
        return None;
    }
    let idx = h.frame_to_show_map_idx;
    if !st.valid[idx] {
        return Some(Violation::of_reference(format!(
            "RefValid[ frame_to_show_map_idx ] is 0 (slot {idx}) when display_frame_id is read \
             (6.8.2, display_frame_id)"
        )));
    }
    let held = st.slots[idx].as_ref().map_or(0, |s| s.frame_id);
    if held != h.display_frame_id {
        return Some(Violation::of_reference(format!(
            "display_frame_id {} does not match RefFrameId[ frame_to_show_map_idx ] (slot {idx}) \
             {held} (6.8.2, display_frame_id)",
            h.display_frame_id
        )));
    }
    let len = id_len(seq);
    if len > 16 {
        return Some(Violation::of_form(format!(
            "idLen is {len}: display_frame_id needs more than 16 bits (6.8.2, display_frame_id)"
        )));
    }
    None
}

/// The variables a level defines (A.3), for the levels the tables define.
struct Level {
    name: &'static str,
    max_pic_size: u64,
    max_h_size: u32,
    max_v_size: u32,
    max_display_rate: u64,
    max_decode_rate: u64,
    max_header_rate: u64,
    max_tiles: u64,
    max_tile_cols: usize,
}

/// The level `seq_level_idx` maps to, if the tables define it (31, the
/// maximum parameters level, has no limits; the indices the tables leave
/// out are levels not yet defined).
fn level(seq_level_idx: u32) -> Option<Level> {
    #[rustfmt::skip]
    let (name, max_pic_size, max_h_size, max_v_size, max_display_rate, max_decode_rate,
         max_header_rate, max_tiles, max_tile_cols) = match seq_level_idx {
        0 => ("2.0", 147_456, 2048, 1152, 4_423_680, 5_529_600, 150, 8, 4),
        1 => ("2.1", 278_784, 2816, 1584, 8_363_520, 10_454_400, 150, 8, 4),
        4 => ("3.0", 665_856, 4352, 2448, 19_975_680, 24_969_600, 150, 16, 6),
        5 => ("3.1", 1_065_024, 5504, 3096, 31_950_720, 39_938_400, 150, 16, 6),
        8 => ("4.0", 2_359_296, 6144, 3456, 70_778_880, 77_856_768, 300, 32, 8),
        9 => ("4.1", 2_359_296, 6144, 3456, 141_557_760, 155_713_536, 300, 32, 8),
        12 => ("5.0", 8_912_896, 8192, 4352, 267_386_880, 273_715_200, 300, 64, 8),
        13 => ("5.1", 8_912_896, 8192, 4352, 534_773_760, 547_430_400, 300, 64, 8),
        14 => ("5.2", 8_912_896, 8192, 4352, 1_069_547_520, 1_094_860_800, 300, 64, 8),
        15 => ("5.3", 8_912_896, 8192, 4352, 1_069_547_520, 1_176_502_272, 300, 64, 8),
        16 => ("6.0", 35_651_584, 16384, 8704, 1_069_547_520, 1_176_502_272, 300, 128, 16),
        17 => ("6.1", 35_651_584, 16384, 8704, 2_139_095_040, 2_189_721_600, 300, 128, 16),
        18 => ("6.2", 35_651_584, 16384, 8704, 4_278_190_080, 4_379_443_200, 300, 128, 16),
        19 => ("6.3", 35_651_584, 16384, 8704, 4_278_190_080, 4_706_009_088, 300, 128, 16),
        _ => return None,
    };
    Some(Level {
        name,
        max_pic_size,
        max_h_size,
        max_v_size,
        max_display_rate,
        max_decode_rate,
        max_header_rate,
        max_tiles,
        max_tile_cols,
    })
}

/// The level of the operating point being decoded.
fn level_of(seq: &SequenceHeader) -> Option<Level> {
    level(*seq.seq_level_idx.get(seq.op)?)
}

/// The limits of A.3 on one frame (show_existing_frame equal to 0): its
/// size and its tiles. The first one broken, named.
pub(crate) fn frame_level(seq: &SequenceHeader, h: &FrameHeader) -> Option<String> {
    let l = level_of(seq)?;
    let lv = l.name;
    let (uw, fw, fh) = (h.upscaled_width, h.frame_width, h.frame_height);
    if uw as u64 * fh as u64 > l.max_pic_size {
        return Some(format!(
            "level {lv}: UpscaledWidth * FrameHeight ({uw} x {fh}) is more than MaxPicSize {} (A.3)",
            l.max_pic_size
        ));
    }
    if uw > l.max_h_size {
        return Some(format!(
            "level {lv}: UpscaledWidth {uw} is more than MaxHSize {} (A.3)",
            l.max_h_size
        ));
    }
    if fh > l.max_v_size {
        return Some(format!(
            "level {lv}: FrameHeight {fh} is more than MaxVSize {} (A.3)",
            l.max_v_size
        ));
    }
    if fw < 16 || fh < 16 {
        return Some(format!(
            "level {lv}: FrameWidth x FrameHeight ({fw} x {fh}) is less than 16 in a dimension (A.3)"
        ));
    }
    let ti = &h.tile_info;
    let tiles = (ti.cols * ti.rows) as u64;
    if tiles > l.max_tiles {
        return Some(format!(
            "level {lv}: NumTiles {tiles} is more than MaxTiles {} (A.3)",
            l.max_tiles
        ));
    }
    if ti.cols > l.max_tile_cols {
        return Some(format!(
            "level {lv}: TileCols {} is more than MaxTileCols {} (A.3)",
            ti.cols, l.max_tile_cols
        ));
    }
    for col in 0..ti.cols {
        let (c0, c1) = (ti.mi_col_starts[col], ti.mi_col_starts[col + 1]);
        let tile_width = ((c1 - c0) * MI_SIZE) as u64;
        let right_most = c1 == h.mi_cols;
        let upscaled = tile_width * h.superres_denom as u64 / SUPERRES_NUM as u64;
        if upscaled > MAX_TILE_WIDTH as u64 {
            return Some(format!(
                "level {lv}: tile column {col}: TileWidth * SuperresDenom / SUPERRES_NUM \
                 ({upscaled}) is more than MAX_TILE_WIDTH (A.3)"
            ));
        }
        let min = if h.use_superres { 128 } else { 64 };
        if !right_most && tile_width < min {
            return Some(format!(
                "level {lv}: tile column {col} is not the right-most and its TileWidth {tile_width} \
                 is less than {min} (A.3)"
            ));
        }
        let cropped = fw as i64 - (c0 * MI_SIZE) as i64;
        if cropped < 8 {
            return Some(format!(
                "level {lv}: tile column {col}: CroppedTileWidth {cropped} is less than 8 (A.3)"
            ));
        }
        for row in 0..ti.rows {
            let (r0, r1) = (ti.mi_row_starts[row], ti.mi_row_starts[row + 1]);
            let tile_height = ((r1 - r0) * MI_SIZE) as u64;
            if tile_width * tile_height > 4096 * 2304 {
                return Some(format!(
                    "level {lv}: tile ({row}, {col}): TileWidth * TileHeight ({tile_width} x \
                     {tile_height}) is more than 4096 * 2304 (A.3)"
                ));
            }
            let cropped = fh as i64 - (r0 * MI_SIZE) as i64;
            if col == 0 && cropped < 8 {
                return Some(format!(
                    "level {lv}: tile row {row}: CroppedTileHeight {cropped} is less than 8 (A.3)"
                ));
            }
        }
    }
    None
}

/// What a temporal unit counts toward the rate limits of A.3.
#[derive(Clone, Copy, Default)]
struct UnitCount {
    /// `UpscaledWidth * FrameHeight` of the frames it shows (each time
    /// shown) and of the frames it decodes.
    displayed: u64,
    decoded: u64,
    /// Frame headers (not copies, not `show_existing_frame`) and tiles.
    headers: u64,
    tiles: u64,
}

/// The rate limits of A.3, measured where the stream says when its
/// temporal units are shown: with `timing_info()` and
/// `equal_picture_interval`, one temporal unit every
/// `num_ticks_per_picture * num_units_in_display_tick / time_scale`
/// seconds. (Without, the times are the decoder model's or the
/// application's, and the rates are not measured.)
///
/// `TotalDisplayLumaSampleRate` is a temporal unit's own (A.3 divides by
/// the time to the next one). The others are taken over each second of
/// display time, the last so many temporal units: `NumFrameHeadersSec`
/// and the tiles a second are counts a second, and
/// `TotalDecodedLumaSampleRate`, which A.3 divides by the time between
/// decoding times, is held to over a second because decoding times are
/// the decoder model's (Annex E): over a second, the frames a stream
/// decodes are what it must decode in a second, wherever the model puts
/// each. Those three are measured once a second of the stream has gone
/// by: a stream's first temporal unit, with its hidden frames, is not a
/// second's worth.
#[derive(Clone, Default)]
pub(crate) struct Rates {
    /// The temporal units of the last second, the current one last, and
    /// how many make a second.
    units: std::collections::VecDeque<UnitCount>,
    per_second: usize,
}

impl Rates {
    /// A new temporal unit begins (a temporal delimiter).
    pub(crate) fn begin_unit(&mut self, seq: Option<&SequenceHeader>) {
        self.units.push_back(UnitCount::default());
        // (At a stream's first temporal delimiter the sequence header is
        // still to come: how many units make a second is known from the
        // first frame on, `count`.)
        if let Some(seq) = seq {
            self.per_second = per_second(seq).unwrap_or(1).max(1) as usize;
        }
        while self.units.len() > self.per_second.max(1) {
            self.units.pop_front();
        }
    }

    /// Counts a frame into the current temporal unit (its header, if it
    /// is one, its tiles, its samples decoded and shown), and tells
    /// whether a limit is now broken.
    pub(crate) fn count(&mut self, seq: &SequenceHeader, h: &FrameHeader) -> Option<String> {
        if self.units.is_empty() {
            self.units.push_back(UnitCount::default());
        }
        self.per_second = per_second(seq).unwrap_or(1).max(1) as usize;
        let samples = h.upscaled_width as u64 * h.frame_height as u64;
        let u = self.units.back_mut().expect("a unit");
        if h.show_existing_frame {
            u.displayed += samples;
        } else {
            u.headers += 1;
            u.tiles += (h.tile_info.cols * h.tile_info.rows) as u64;
            u.decoded += samples;
            if h.show_frame {
                u.displayed += samples;
            }
        }
        let l = level_of(seq)?;
        let lv = l.name;
        // Seconds per temporal unit: ticks * units / time_scale.
        let ticks = seq.num_ticks_per_picture as u128 * seq.num_units_in_display_tick as u128;
        let scale = seq.time_scale as u128;
        if ticks == 0 || scale == 0 || !seq.equal_picture_interval {
            return None;
        }
        let u = *self.units.back().expect("a unit");
        // displayed / (ticks / scale) <= MaxDisplayRate
        if u.displayed as u128 * scale > l.max_display_rate as u128 * ticks {
            return Some(format!(
                "level {lv}: TotalDisplayLumaSampleRate ({} samples in a temporal unit of {ticks} \
                 / {scale} s) is more than MaxDisplayRate {} (A.3)",
                u.displayed, l.max_display_rate
            ));
        }
        // Over the last second: the units in it, n * ticks / scale seconds.
        if self.units.len() < self.per_second {
            return None;
        }
        let n = self.units.len() as u128;
        let span = n * ticks;
        let sum = |f: fn(&UnitCount) -> u64| self.units.iter().map(f).sum::<u64>() as u128;
        let checks: [(&str, u128, u64); 3] = [
            ("NumFrameHeadersSec", sum(|u| u.headers), l.max_header_rate),
            (
                "TotalDecodedLumaSampleRate",
                sum(|u| u.decoded),
                l.max_decode_rate,
            ),
            (
                "the number of tiles a second",
                sum(|u| u.tiles),
                l.max_tiles * 120,
            ),
        ];
        let limit_names = ["MaxHeaderRate", "MaxDecodeRate", "MaxTiles * 120"];
        for ((what, count, max), limit) in checks.into_iter().zip(limit_names) {
            // count / (span / scale) <= max
            if count * scale > max as u128 * span {
                return Some(format!(
                    "level {lv}: {what} ({count} over {n} temporal units of {ticks} / {scale} s) \
                     is more than {limit} {max} (A.3)"
                ));
            }
        }
        None
    }
}

/// How many temporal units make a second of display time (at least one),
/// if the sequence says.
fn per_second(seq: &SequenceHeader) -> Option<u64> {
    let ticks = seq.num_ticks_per_picture * seq.num_units_in_display_tick as u64;
    if ticks == 0 || !seq.equal_picture_interval {
        return None;
    }
    Some(seq.time_scale as u64 / ticks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::TileInfo;

    /// A sequence header (a committed vector's), with frame ids of
    /// `id_len` bits, and the level and timing given.
    fn seq(id_len: u32, level: u32, fps: Option<u32>) -> SequenceHeader {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/av1-1-b8-01-size-66x66.ivf"
        );
        let data = std::fs::read(path).unwrap();
        let unit = crate::ivf::IvfReader::new(&data)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .data;
        let mut rest = unit;
        let mut s = loop {
            let Ok((obu, len)) = crate::obu::read_obu(rest) else {
                panic!("a sequence header");
            };
            if obu.obu_type == OBU_SEQUENCE_HEADER {
                break SequenceHeader::parse(obu.payload, 0).unwrap();
            }
            rest = &rest[len..];
        };
        s.frame_id_numbers_present = true;
        s.delta_frame_id_length_minus_2 = id_len - 3;
        s.additional_frame_id_length_minus_1 = 0;
        s.seq_level_idx = vec![level];
        if let Some(fps) = fps {
            s.timing_info_present = true;
            s.equal_picture_interval = true;
            s.num_units_in_display_tick = 1;
            s.time_scale = fps;
            s.num_ticks_per_picture = 1;
        }
        s
    }

    fn header(w: u32, h: u32, cur: u32, prev: Option<u32>) -> FrameHeader {
        let mut hdr = FrameHeader {
            frame_type: INTER_FRAME,
            show_frame: true,
            current_frame_id: cur,
            prev_frame_id: prev,
            frame_width: w,
            upscaled_width: w,
            frame_height: h,
            mi_cols: 2 * (w as usize).div_ceil(8),
            mi_rows: 2 * (h as usize).div_ceil(8),
            superres_denom: SUPERRES_NUM,
            tile_info: TileInfo {
                cols: 1,
                rows: 1,
                mi_col_starts: vec![0, 2 * (w as usize).div_ceil(8)],
                mi_row_starts: vec![0, 2 * (h as usize).div_ceil(8)],
                ..Default::default()
            },
            ..Default::default()
        };
        // Every reference in slot 0, frame id `cur - 1`, as expected.
        hdr.expected_frame_id = [cur.wrapping_sub(1) & 0xff; 7];
        hdr
    }

    fn refs(id: u32) -> RefState {
        let mut st = RefState {
            valid: [true; 8],
            ..Default::default()
        };
        let slot = std::sync::Arc::new(crate::header::RefHeaderState {
            frame_id: id,
            frame_type: INTER_FRAME,
            upscaled_width: 64,
            frame_height: 64,
            render_width: 64,
            render_height: 64,
            loop_filter_ref_deltas: [0; 8],
            loop_filter_mode_deltas: [0; 2],
            feature_enabled: [[false; 8]; 8],
            feature_data: [[0; 8]; 8],
            gm_params: [[0; 6]; 8],
            film_grain: Default::default(),
        });
        st.slots = std::array::from_fn(|_| Some(slot.clone()));
        st
    }

    fn rule(v: Vec<Violation>) -> String {
        v.into_iter().next().map_or(String::new(), |v| v.rule)
    }

    #[test]
    fn current_frame_id_rules() {
        let s = seq(8, 31, None);
        let ok = header(64, 64, 10, Some(9));
        assert!(frame_ids(&s, &ok, &refs(9)).is_empty());
        // Equal to the previous frame's.
        let same = header(64, 64, 10, Some(10));
        assert!(rule(frame_ids(&s, &same, &refs(9))).contains("PrevFrameID"));
        // Half the id space or more ahead (idLen 8: DiffFrameID < 128).
        let far = header(64, 64, 10 + 128, Some(10));
        let mut st = refs((10 + 127) & 0xff);
        st.slots[0] = refs(137).slots[0].clone();
        let v = frame_ids(&s, &far, &st);
        assert!(rule(v).contains("DiffFrameID 128"));
        let near = header(64, 64, (10 + 127) & 0xff, Some(10));
        assert!(frame_ids(&s, &near, &refs(136)).is_empty());
        // Wrapping: from 250 to 3 is 9 frames on.
        let wrapped = header(64, 64, 3, Some(250));
        assert!(frame_ids(&s, &wrapped, &refs(2)).is_empty());
        // A shown key frame is not held to them.
        let mut key = header(64, 64, 10, Some(10));
        key.frame_type = KEY_FRAME;
        key.frame_is_intra = true;
        assert!(frame_ids(&s, &key, &refs(9)).is_empty());
        // A reference that is not the frame expected, or not valid: a lost
        // reference.
        let v = frame_ids(&s, &ok, &refs(8)).remove(0);
        assert!(v.lost_reference && v.rule.contains("expectedFrameId[ 0 ]"));
        let mut st = refs(9);
        st.valid[0] = false;
        let v = frame_ids(&s, &ok, &st).remove(0);
        assert!(v.lost_reference && v.rule.contains("RefValid"));
        // More than 16 bits of id.
        let s17 = seq(17, 31, None);
        assert!(rule(frame_ids(&s17, &header(64, 64, 10, None), &refs(9))).contains("idLen is 17"));
    }

    #[test]
    fn rates_are_measured_over_a_second() {
        // Level 2.0: MaxDisplayRate 4 423 680, MaxDecodeRate 5 529 600,
        // MaxHeaderRate 150 a second.
        let s = seq(8, 0, Some(30));
        let h = header(416, 240, 1, Some(0));
        let mut r = Rates::default();
        // 30 frames of 416x240 a second: 2 995 200 samples, within.
        for _ in 0..60 {
            r.begin_unit(Some(&s));
            assert_eq!(r.count(&s, &h), None);
        }
        // A temporal unit that shows two: 5 990 400 a second for it, more
        // than MaxDisplayRate.
        r.begin_unit(Some(&s));
        assert_eq!(r.count(&s, &h), None);
        let shown = r.count(&s, &h).unwrap_or_default();
        assert!(shown.contains("MaxDisplayRate"), "{shown}");
        // Hidden frames, one more each unit: 59 frames decoded over the
        // last second is 7.3 M samples a second, more than MaxDecodeRate.
        let mut hidden = h.clone();
        hidden.show_frame = false;
        let mut r = Rates::default();
        let mut broke = None;
        for _ in 0..30 {
            r.begin_unit(Some(&s));
            broke = broke.or(r.count(&s, &hidden)).or(r.count(&s, &h));
        }
        assert!(broke.unwrap_or_default().contains("MaxDecodeRate"));
        // Without timing information nothing is measured.
        let untimed = seq(8, 0, None);
        let mut r = Rates::default();
        for _ in 0..100 {
            r.begin_unit(Some(&untimed));
            for _ in 0..4 {
                assert_eq!(r.count(&untimed, &h), None);
            }
        }
    }

    #[test]
    fn frame_level_limits() {
        let s = seq(8, 0, None);
        assert_eq!(frame_level(&s, &header(416, 240, 1, None)), None);
        let big = rule_of(frame_level(&s, &header(720, 205, 1, None)));
        assert!(big.contains("MaxPicSize 147456"), "{big}");
        let wide = rule_of(frame_level(&s, &header(2049, 16, 1, None)));
        assert!(wide.contains("MaxHSize 2048"), "{wide}");
        let small = rule_of(frame_level(&s, &header(416, 8, 1, None)));
        assert!(small.contains("less than 16"), "{small}");
        // The maximum parameters level has no limits.
        let s31 = seq(8, 31, None);
        assert_eq!(frame_level(&s31, &header(720, 205, 1, None)), None);
    }

    fn rule_of(v: Option<String>) -> String {
        v.unwrap_or_default()
    }
}
