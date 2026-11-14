//! Requirements of conformance that decoding does not need (frame ids,
//! 6.8.2; level limits, A.3): a strict decoder reports each one a stream
//! breaks, naming it, and a decoder that is not strict decodes the stream
//! as it did before these checks. The streams are the encoder's (with
//! frame ids), with header fields rewritten in place to break a rule.
//! (Argon's error streams break each rule too: `tests/argon.rs`. The rules
//! on `current_frame_id` against the previous frame's cannot be broken by
//! rewriting one field, every reference's expected id following from it:
//! the decoder's unit tests and Argon's streams have those.)

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

fn frames(w: u32, h: u32, n: u32) -> Vec<Frame> {
    (0..n)
        .map(|t| {
            let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
            for p in 0..3 {
                let pl = f.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        let v = ((x + 2 * t) * 7 + y * 3 + 40 * p as u32) % 200 + 20;
                        f.set_sample(p, x, y, v as u16);
                    }
                }
            }
            f
        })
        .collect()
}

fn config(w: u32, h: u32) -> Config {
    let mut c = Config::new(w, h);
    c.speed = 8;
    c.tools = av1::Tools::for_speed(8);
    // No screen content tools: the frame header's fields up to the frame
    // ids are then at the places `ID_AT` and `REF0_DELTA_AT` say.
    c.tools.palette = false;
    c.tools.intrabc = false;
    c.frame_ids = true;
    c.altref = 0;
    c
}

fn encode(cfg: Config, n: u32) -> Vec<Vec<u8>> {
    let src = frames(cfg.width, cfg.height, n);
    let mut enc = Encoder::new(cfg);
    let mut units = Vec::new();
    for f in &src {
        let u = enc.encode(f).unwrap();
        if !u.is_empty() {
            units.push(u);
        }
    }
    while let Some(u) = enc.flush().unwrap() {
        units.push(u);
    }
    units
}

/// The OBUs of a unit: type, and where the payload starts and ends.
fn obus(unit: &[u8]) -> Vec<(u8, usize, usize)> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < unit.len() {
        let h = unit[p];
        p += 1 + ((h >> 2) & 1) as usize;
        let (mut size, mut i) = (0usize, 0);
        loop {
            let b = unit[p];
            p += 1;
            size |= ((b & 0x7f) as usize) << (7 * i);
            i += 1;
            if b & 0x80 == 0 {
                break;
            }
        }
        out.push(((h >> 3) & 15, p, p + size));
        p += size;
    }
    out
}

/// The payload of the unit's first OBU of type `t`.
fn payload_at(unit: &[u8], t: u8) -> usize {
    obus(unit).iter().find(|o| o.0 == t).expect("the OBU").1
}

fn get_bits(d: &[u8], at: usize, n: usize) -> u32 {
    (0..n).fold(0, |v, i| {
        let b = at + i;
        v << 1 | ((d[b / 8] >> (7 - b % 8)) & 1) as u32
    })
}

fn set_bits(d: &mut [u8], at: usize, n: usize, v: u32) {
    for i in 0..n {
        let b = at + i;
        let bit = (v >> (n - 1 - i)) & 1;
        d[b / 8] = (d[b / 8] & !(0x80 >> (b % 8))) | ((bit as u8) << (7 - b % 8));
    }
}

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;
/// Bits of a frame id, of a reference's delta (the encoder's).
const ID_BITS: usize = 15;
const DELTA_BITS: usize = 14;
/// Where `current_frame_id` is in a shown inter frame's header:
/// show_existing_frame, frame_type (2), show_frame, error_resilient_mode,
/// disable_cdf_update.
const ID_AT: usize = 6;
/// Where `delta_frame_id_minus_1` of its first reference is: after the id,
/// frame_size_override_flag, order_hint (7), primary_ref_frame (3),
/// refresh_frame_flags (8), frame_refs_short_signaling, ref_frame_idx[ 0 ].
const REF0_DELTA_AT: usize = ID_AT + ID_BITS + 1 + 7 + 3 + 8 + 1 + 3;

/// Decodes `units`: each unit's result.
fn decode(units: &[Vec<u8>], strict: bool) -> Vec<Result<Vec<Frame>, String>> {
    let mut d = Decoder::new();
    d.set_strict(strict);
    units
        .iter()
        .map(|u| d.decode_all(u).map_err(|e| e.to_string()))
        .collect()
}

/// Every unit decodes, strictly.
fn all_ok(units: &[Vec<u8>], what: &str) -> Vec<Vec<Frame>> {
    decode(units, true)
        .into_iter()
        .enumerate()
        .map(|(i, r)| r.unwrap_or_else(|e| panic!("{what}: unit {i}: {e}")))
        .collect()
}

/// The strict decode of `units` fails at unit `at`, with an error that
/// contains `words`, and not before.
fn strict_fails(units: &[Vec<u8>], at: usize, words: &str) {
    for (i, r) in decode(units, true).into_iter().enumerate().take(at + 1) {
        match r {
            Err(e) if i == at => assert!(e.contains(words), "unit {i}: {e} (wanted {words:?})"),
            Err(e) => panic!("unit {i}: {e}"),
            Ok(_) if i == at => panic!("unit {i}: no error (wanted {words:?})"),
            Ok(_) => {}
        }
    }
}

#[test]
fn encoder_frame_ids_conform() {
    let units = encode(config(96, 64), 10);
    let strict = all_ok(&units, "frame ids");
    // Ids or not, the frames are the same.
    let mut plain = config(96, 64);
    plain.frame_ids = false;
    let without = all_ok(&encode(plain, 10), "no frame ids");
    assert_eq!(strict, without);
    let mut groups = config(96, 64);
    groups.altref = 4;
    all_ok(&encode(groups, 10), "frame ids, alt-ref groups");
}

#[test]
fn a_reference_that_is_not_the_frame_expected() {
    let mut units = encode(config(96, 64), 6);
    let last = units.len() - 1;
    let u = &mut units[last];
    let at = payload_at(u, OBU_FRAME);
    let delta = get_bits(&u[at..], REF0_DELTA_AT, DELTA_BITS);
    set_bits(&mut u[at..], REF0_DELTA_AT, DELTA_BITS, delta + 1);
    strict_fails(&units, last, "expectedFrameId[ 0 ]");
    // Not strict: the reference is not the frame the stream meant (one
    // was lost, as far as the decoder can tell), so the frame is not
    // shown as the stream's.
    let lax = decode(&units, false);
    assert_eq!(lax[last], Ok(Vec::new()));
    let mut d = Decoder::new();
    d.set_concealment(true);
    let shown: Vec<_> = units.iter().map(|u| d.decode_all(u).unwrap()).collect();
    assert!(shown[last][0].concealed);
    assert!(shown[..last].iter().all(|f| !f[0].concealed));
}

#[test]
fn display_frame_id_not_the_slot_s() {
    let mut cfg = config(96, 64);
    cfg.altref = 4;
    let mut units = encode(cfg, 10);
    // A unit that shows a hidden frame: a frame header OBU of
    // show_existing_frame, frame_to_show_map_idx (3), display_frame_id.
    let k = units
        .iter()
        .position(|u| {
            obus(u)
                .iter()
                .any(|o| o.0 == OBU_FRAME_HEADER && u[o.1] >> 7 == 1)
        })
        .expect("a show_existing_frame");
    let u = &mut units[k];
    let at = obus(u)
        .iter()
        .find(|o| o.0 == OBU_FRAME_HEADER && u[o.1] >> 7 == 1)
        .unwrap()
        .1;
    let id = get_bits(&u[at..], 4, ID_BITS);
    set_bits(&mut u[at..], 4, ID_BITS, (id + 1) % (1 << ID_BITS));
    strict_fails(&units, k, "display_frame_id");
}

/// Rewrites the sequence headers' `seq_level_idx` (the encoder writes 31,
/// no limits) to 8, level 4.0: five bits after seq_profile (3),
/// still_picture, reduced_still_picture_header, timing_info_present_flag,
/// initial_display_delay_present_flag, operating_points_cnt_minus_1 (5)
/// and operating_point_idc (12). (31 and 8 both have a `seq_tier` after.)
fn at_level_4_0(units: &mut [Vec<u8>]) {
    for u in units.iter_mut() {
        if let Some(o) = obus(u).iter().find(|o| o.0 == OBU_SEQUENCE_HEADER).copied() {
            assert_eq!(get_bits(&u[o.1..], 24, 5), 31);
            set_bits(&mut u[o.1..], 24, 5, 8);
        }
    }
}

#[test]
fn level_limits() {
    // Within level 4.0: conforms.
    let mut units = encode(config(96, 64), 3);
    at_level_4_0(&mut units);
    all_ok(&units, "level 4.0");
    // Narrower than 16: A.3's FrameWidth.
    let mut units = encode(config(8, 64), 2);
    at_level_4_0(&mut units);
    strict_fails(&units, 0, "FrameWidth");
    assert!(decode(&units, false).iter().all(|r| r.is_ok()));
    // 16 tile columns: more than MaxTileCols, 8.
    let mut cfg = config(1024, 64);
    cfg.tile_cols_log2 = 4;
    let mut units = encode(cfg, 2);
    at_level_4_0(&mut units);
    strict_fails(&units, 0, "MaxTileCols 8");
    let lax = decode(&units, false);
    assert!(lax.iter().all(|r| r.as_ref().is_ok_and(|f| f.len() == 1)));
    // Wider than MaxHSize, 6144.
    let mut units = encode(config(6208, 16), 1);
    at_level_4_0(&mut units);
    strict_fails(&units, 0, "MaxHSize 6144");
}
