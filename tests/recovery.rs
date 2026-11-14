//! Recovery from errors: streams with units corrupted, cut short and
//! dropped. Whatever the damage, decoding never panics, no frame that is
//! not exactly the stream's comes out unflagged, and from the first key
//! frame after the damage every unit's result is exactly a clean decode's.
//!
//! The streams are the committed vectors (each played twice, so that the
//! second play's key frame follows the damage) and encoder streams with
//! key frames, alt-ref groups (hidden frames shown later) and frame ids.

mod common;

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

/// The decoder settings a stream is decoded with.
#[derive(Clone, Copy, Debug)]
struct Mode {
    conceal: bool,
    threads: usize,
    flight: usize,
}

const MODES: [Mode; 4] = [
    Mode {
        conceal: false,
        threads: 1,
        flight: 1,
    },
    Mode {
        conceal: true,
        threads: 1,
        flight: 1,
    },
    Mode {
        conceal: true,
        threads: 4,
        flight: 1,
    },
    Mode {
        conceal: false,
        threads: 4,
        flight: 4,
    },
];

type UnitResult = Result<Vec<Frame>, String>;

/// Decodes `units` unit by unit; `loss_before` units are preceded by
/// `signal_loss`.
fn decode(units: &[Vec<u8>], mode: Mode, loss_before: Option<usize>) -> Vec<UnitResult> {
    let mut d = Decoder::new();
    d.set_concealment(mode.conceal);
    d.set_threads(mode.threads);
    d.set_frames_in_flight(mode.flight);
    let mut out = Vec::new();
    for (i, u) in units.iter().enumerate() {
        if loss_before == Some(i) {
            d.signal_loss();
        }
        if mode.flight > 1 {
            d.queue(u);
            while let Some(r) = d.poll() {
                out.push(r.map_err(|e| e.to_string()));
            }
        } else {
            out.push(d.decode_all(u).map_err(|e| e.to_string()));
        }
    }
    while let Some(r) = d.wait() {
        out.push(r.map_err(|e| e.to_string()));
    }
    assert_eq!(out.len(), units.len(), "a result per unit");
    out
}

/// The OBUs of a unit (low-overhead format): type and payload.
fn obus(unit: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < unit.len() {
        let h = unit[p];
        let t = (h >> 3) & 15;
        let ext = (h >> 2) & 1 != 0;
        let has_size = (h >> 1) & 1 != 0;
        p += 1 + ext as usize;
        let size = if has_size {
            let (mut v, mut i) = (0usize, 0);
            loop {
                let Some(&b) = unit.get(p) else {
                    return out;
                };
                p += 1;
                v |= ((b & 0x7f) as usize) << (7 * i);
                i += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            v
        } else {
            unit.len() - p
        };
        out.push((t, &unit[p.min(unit.len())..(p + size).min(unit.len())]));
        p += size;
    }
    out
}

/// Whether a unit's first frame is a shown key frame: a random access
/// point, from which a clean decode needs nothing before.
fn random_access(unit: &[u8]) -> bool {
    for (t, payload) in obus(unit) {
        // OBU_FRAME_HEADER, OBU_FRAME
        if t == 3 || t == 6 {
            let b = payload[0];
            let show_existing = b >> 7 != 0;
            let frame_type = (b >> 5) & 3;
            let show_frame = (b >> 4) & 1 != 0;
            return !show_existing && frame_type == 0 && show_frame;
        }
    }
    false
}

/// Whether a unit (cut short) ends in a frame OBU of a shown inter frame
/// with some of its tile data.
fn cuts_a_shown_inter_frame(unit: &[u8]) -> bool {
    obus(unit).last().is_some_and(|&(t, payload)| {
        t == 6
            && payload.len() > 16
            && payload[0] >> 7 == 0
            && (payload[0] >> 5) & 3 == 1
            && (payload[0] >> 4) & 1 == 1
    })
}

fn same(a: &Frame, b: &Frame) -> bool {
    a.width == b.width && a.height == b.height && a.data == b.data
}

/// Holds `got` (the damaged stream's results, unit by unit) to `clean`
/// (the clean stream's): `clean_of(i)` is the clean unit damaged unit `i`
/// is, `damaged_from` the first damaged unit. Every unflagged frame is a
/// frame of the clean unit; from the first random access point after the
/// damage, every result is the clean one. Returns how many frames came
/// out flagged as concealed.
fn check(
    what: &str,
    units: &[Vec<u8>],
    clean: &[Vec<Frame>],
    got: &[UnitResult],
    clean_of: impl Fn(usize) -> usize,
    damaged_from: usize,
) -> usize {
    let resume = (damaged_from + 1..units.len())
        .find(|&i| random_access(&units[i]))
        .unwrap_or(units.len());
    let mut concealed = 0;
    for (i, r) in got.iter().enumerate() {
        let want = &clean[clean_of(i)];
        if i < damaged_from || i >= resume {
            let frames = r
                .as_ref()
                .unwrap_or_else(|e| panic!("{what}: unit {i} (clean): {e}"));
            assert_eq!(frames.len(), want.len(), "{what}: unit {i}: frames shown");
            for (f, w) in frames.iter().zip(want) {
                assert!(!f.concealed, "{what}: unit {i}: flagged in a clean unit");
                assert!(same(f, w), "{what}: unit {i}: differs from a clean decode");
            }
            continue;
        }
        if let Ok(frames) = r {
            for f in frames {
                if f.concealed {
                    concealed += 1;
                } else {
                    assert!(
                        want.iter().any(|w| same(f, w)),
                        "{what}: unit {i}: an unflagged frame that is not the stream's"
                    );
                }
            }
        }
    }
    concealed
}

/// Every way of damaging unit `k` of `units`, decoded every way, checked.
fn damage_and_check(name: &str, units: &[Vec<u8>], k: usize, seed: u32) {
    let clean: Vec<Vec<Frame>> = decode(units, MODES[0], None)
        .into_iter()
        .map(|r| r.unwrap_or_else(|e| panic!("{name}: clean decode: {e}")))
        .collect();
    let mut rng = seed | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };
    let len = units[k].len();
    // Corrupted: bytes in the second half of the unit (the tile data of
    // its last frame) changed.
    let mut corrupt = units.to_vec();
    for _ in 0..3 {
        let at = len / 2 + next() as usize % (len - len / 2);
        corrupt[k][at] ^= 1 << (next() % 8);
    }
    // Cut short, at two thirds.
    let mut cut = units.to_vec();
    cut[k].truncate(len * 2 / 3);
    // Dropped.
    let mut dropped = units.to_vec();
    dropped.remove(k);
    let mut concealed_outputs: Vec<Vec<UnitResult>> = Vec::new();
    for mode in MODES {
        let what = format!("{name} unit {k} corrupted ({mode:?})");
        let got = decode(&corrupt, mode, None);
        check(&what, &corrupt, &clean, &got, |i| i, k);
        let what = format!("{name} unit {k} cut short ({mode:?})");
        let got = decode(&cut, mode, None);
        let flagged = check(&what, &cut, &clean, &got, |i| i, k);
        if mode.conceal {
            concealed_outputs.push(got.clone());
            if cuts_a_shown_inter_frame(&cut[k]) {
                // The frame cut short comes out, flagged.
                assert!(flagged > 0, "{what}: the cut frame is concealed");
            }
        }
        let what = format!("{name} unit {k} dropped, loss signalled ({mode:?})");
        let back = |i| if i < k { i } else { i + 1 };
        let got = decode(&dropped, mode, Some(k));
        check(&what, &dropped, &clean, &got, back, k);
        // Not signalled: the loss may go unnoticed (a frame predicted from
        // the wrong reference is a valid decode of another stream), but
        // decoding is exact again from the next key frame.
        let got = decode(&dropped, mode, None);
        let resume = (k..dropped.len())
            .find(|&i| random_access(&dropped[i]))
            .unwrap_or(dropped.len());
        for i in resume..dropped.len() {
            let frames = got[i].as_ref().expect("a unit after the key frame");
            assert_eq!(frames.len(), clean[back(i)].len(), "{name}: {i}");
            for (f, w) in frames.iter().zip(&clean[back(i)]) {
                assert!(
                    same(f, w) && !f.concealed,
                    "{name} unit {k} dropped: unit {i}"
                );
            }
        }
    }
    // Concealment is the same on one thread and on four.
    type Key = Vec<Option<Vec<(bool, Vec<u8>)>>>;
    let key = |rs: &Vec<UnitResult>| -> Key {
        rs.iter()
            .map(|r| {
                r.as_ref()
                    .ok()
                    .map(|fs| fs.iter().map(|f| (f.concealed, f.data.clone())).collect())
            })
            .collect()
    };
    assert_eq!(
        key(&concealed_outputs[0]),
        key(&concealed_outputs[1]),
        "{name} unit {k} cut short: concealment by thread count"
    );
}

/// The committed vectors, each played twice: damage in the first play,
/// exact again from the second play's key frame.
#[test]
fn committed_vectors_recover() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for path in common::vectors_in(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let once = common::packets(&path);
        let mut units = once.clone();
        units.extend(once.iter().cloned());
        let n = once.len();
        for (j, k) in [0, 1, n / 2, n - 1].into_iter().enumerate() {
            if k < n && (j == 0 || k > 0) {
                damage_and_check(&name, &units, k, 0x9e37_79b9 ^ (k as u32 * 7919));
            }
        }
    }
}

/// `n` frames of a moving pattern.
fn moving(w: u32, h: u32, n: u32) -> Vec<Frame> {
    (0..n)
        .map(|t| {
            let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
            for p in 0..3 {
                let pl = f.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        let xs = x + if p == 0 { 3 * t } else { t };
                        let v = if p == 0 {
                            let ring = ((xs as f64 - 40.0).hypot(y as f64 - 30.0) / 4.0).sin();
                            (120.0 + 60.0 * ring + ((xs * 5 + y * 3) % 17) as f64) as u16
                        } else {
                            (100 + (xs + 2 * y + 9 * p as u32) % 50) as u16
                        };
                        f.set_sample(p, x, y, v);
                    }
                }
            }
            f
        })
        .collect()
}

/// An encoder stream: one unit a shown frame.
fn encode(cfg: Config, frames: &[Frame]) -> Vec<Vec<u8>> {
    let mut enc = Encoder::new(cfg);
    let mut units = Vec::new();
    for f in frames {
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

fn config(w: u32, h: u32) -> Config {
    let mut c = Config::new(w, h);
    c.speed = 8;
    c.tools = av1::Tools::for_speed(8);
    c.keyframe_interval = 8;
    c
}

/// Encoder streams: frames in order, alt-ref groups (hidden frames, shown
/// later), several tiles, frame ids; damage at each kind of unit.
#[test]
fn encoder_streams_recover() {
    let frames = moving(160, 96, 20);
    let mut plain = config(160, 96);
    plain.altref = 0;
    let mut groups = config(160, 96);
    groups.altref = 4;
    let mut tiles = config(160, 96);
    tiles.altref = 0;
    tiles.tile_cols_log2 = 1;
    let mut ids = config(160, 96);
    ids.frame_ids = true;
    for (name, cfg) in [
        ("plain", plain),
        ("alt-ref groups", groups),
        ("two tiles", tiles),
        ("frame ids", ids),
    ] {
        let units = encode(cfg, &frames);
        assert_eq!(units.len(), frames.len(), "{name}");
        for k in [1, 3, 5, 9, 12] {
            damage_and_check(name, &units, k, 0x1234_5678 ^ k as u32);
        }
    }
}

/// With frame ids a dropped unit is noticed without being signalled: the
/// frames that refer to the frame it held are not output (with
/// concealment, they are flagged), and the decoder is exact again from
/// the next key frame.
#[test]
fn frame_ids_notice_a_lost_reference() {
    let frames = moving(160, 96, 16);
    let mut cfg = config(160, 96);
    cfg.altref = 0;
    cfg.frame_ids = true;
    let units = encode(cfg, &frames);
    let clean: Vec<Vec<Frame>> = decode(&units, MODES[0], None)
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let k = 3;
    let mut dropped = units.clone();
    dropped.remove(k);
    for mode in [MODES[0], MODES[1]] {
        let got = decode(&dropped, mode, None);
        let back = |i| if i < k { i } else { i + 1 };
        check("frame ids", &dropped, &clean, &got, back, k);
        // The unit after the lost one refers to it: noticed, so no error
        // (its tile does not parse with the wrong references' CDFs, which
        // is no news), and not output, or output flagged.
        let next = got[k].as_ref().expect("the loss is noticed");
        assert!(next.iter().all(|f| f.concealed), "{mode:?}");
        assert_eq!(next.len(), mode.conceal as usize, "{mode:?}");
    }
}

/// An inter frame cut short comes out concealed with concealment on: each
/// of its superblocks is the clean decode's (away from its edges, where
/// the post-filters reach across from the superblocks that did not decode)
/// or, from the first that did not decode on, its reference's.
#[test]
fn a_cut_inter_frame_is_concealed() {
    let frames = moving(320, 192, 6);
    let mut cfg = config(320, 192);
    cfg.altref = 0;
    let units = encode(cfg, &frames);
    let clean: Vec<Vec<Frame>> = decode(&units, MODES[0], None)
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let mut cut = units.clone();
    let k = 3;
    let len = cut[k].len();
    cut[k].truncate(len / 2);
    let mut d = Decoder::new();
    d.set_concealment(true);
    for u in &cut[..k] {
        d.decode(u).unwrap();
    }
    let f = d.decode(&cut[k]).unwrap().expect("the frame, concealed");
    assert!(f.concealed && d.is_recovering());
    let (want, prev) = (&clean[k][0], &clean[k - 1][0]);
    assert_eq!((f.width, f.height), (want.width, want.height));
    let block = |fr: &Frame, sx: u32, sy: u32, m: u32| -> Vec<u16> {
        let mut v = Vec::new();
        for y in sy * 64 + m..((sy + 1) * 64).min(fr.height) - m {
            for x in sx * 64 + m..((sx + 1) * 64).min(fr.width) - m {
                v.push(fr.sample(0, x, y));
            }
        }
        v
    };
    let (mut decoded, mut filled) = (0, 0);
    for sy in 0..f.height.div_ceil(64) {
        for sx in 0..f.width.div_ceil(64) {
            if block(&f, sx, sy, 0) == block(prev, sx, sy, 0)
                && block(prev, sx, sy, 0) != block(want, sx, sy, 0)
            {
                filled += 1;
            } else {
                assert_eq!(
                    filled, 0,
                    "superblock ({sx}, {sy}) decoded after one that did not"
                );
                assert_eq!(
                    block(&f, sx, sy, 8),
                    block(want, sx, sy, 8),
                    "superblock ({sx}, {sy})"
                );
                decoded += 1;
            }
        }
    }
    assert!(
        decoded > 0 && filled > 0,
        "{decoded} decoded, {filled} filled in"
    );
    // Without concealment the unit is an error and nothing is shown until
    // the next key frame.
    let mut d = Decoder::new();
    for u in &cut[..k] {
        d.decode(u).unwrap();
    }
    assert!(d.decode(&cut[k]).is_err());
    assert!(d.is_recovering());
    assert_eq!(d.decode(&cut[k + 1]).unwrap(), None);
}

/// A unit of garbage is an error that does not stop the decoder: what
/// follows it is not trusted until the next key frame, and exact from it.
#[test]
fn a_garbage_unit_is_passed_over() {
    let frames = moving(160, 96, 12);
    let mut cfg = config(160, 96);
    cfg.altref = 0;
    let units = encode(cfg, &frames);
    let clean: Vec<Vec<Frame>> = decode(&units, MODES[0], None)
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let mut d = Decoder::new();
    for (i, u) in units.iter().enumerate() {
        if i == 4 {
            assert!(d.decode(&[0xff, 0x13, 0x00, 0x77, 0x42]).is_err());
            assert!(d.decode(&[]).unwrap().is_none());
        }
        let got = d.decode_all(u).unwrap();
        if !(4..8).contains(&i) {
            assert_eq!(got.len(), 1, "unit {i}");
            assert!(same(&got[0], &clean[i][0]), "unit {i}");
        } else {
            assert!(got.is_empty(), "unit {i}: withheld");
        }
    }
    assert!(!d.is_recovering());
}
