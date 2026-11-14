//! The encoder: every temporal unit it writes must decode, with a fresh
//! decoder, to exactly the encoder's reconstruction; quality must track the
//! quantiser. The natural-content source is the decoded frames of a
//! committed test vector (tests/data).

mod common;

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

/// A configuration for the tests: the default speed, or a fast one in
/// unoptimised builds (the property and overflow-checking run), which
/// still exercises the rate-distortion search.
fn cfg_for(w: u32, h: u32) -> Config {
    let mut c = Config::new(w, h);
    if cfg!(debug_assertions) {
        c.speed = 8;
        c.tools = av1::Tools::for_speed(8);
    }
    c
}

fn psnr(a: &Frame, b: &Frame, plane: usize) -> f64 {
    let maxv = ((1u32 << a.bit_depth) - 1) as f64;
    let pl = a.planes[plane];
    let mut se = 0f64;
    for y in 0..pl.height {
        for x in 0..pl.width {
            let d = a.sample(plane, x, y) as f64 - b.sample(plane, x, y) as f64;
            se += d * d;
        }
    }
    let mse = se / (pl.width * pl.height) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (maxv * maxv / mse).log10()
    }
}

/// `n` frames of natural video: the 4 frames of av1-1-b8-05-mv.ivf
/// (352x288), decoded, played forwards then backwards as needed.
fn natural(n: usize) -> Vec<Frame> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/av1-1-b8-05-mv.ivf");
    let mut d = Decoder::new();
    let src: Vec<Frame> = common::packets(&p)
        .iter()
        .filter_map(|pk| d.decode(pk).unwrap())
        .collect();
    let period = 2 * src.len() - 2;
    (0..n)
        .map(|i| {
            let k = i % period;
            src[if k < src.len() { k } else { period - k }].clone()
        })
        .collect()
}

fn synthetic(w: u32, h: u32, t: u32, bit_depth: u32) -> Frame {
    let mut f = Frame::new(w, h, bit_depth, ChromaFormat::Yuv420);
    let shift = bit_depth - 8;
    for p in 0..3 {
        let pl = f.planes[p];
        for y in 0..pl.height {
            for x in 0..pl.width {
                // Luma pans 2 samples a frame, chroma (half resolution) 1.
                let xs = if p == 0 { x + 2 * t } else { x + t };
                let v = if p == 0 {
                    let ring = ((xs as f64 - 20.0).hypot(y as f64 - 14.0) / 3.0).sin() * 50.0;
                    (110.0 + ring + ((xs * 7 + y * 3) % 23) as f64) as u16
                } else {
                    (90 + (xs * 2 + y + p as u32 * 17) % 60) as u16
                };
                f.set_sample(p, x, y, v.min(255) << shift);
            }
        }
    }
    f
}

/// Encodes `frames`; decodes every packet with a fresh decoder and checks
/// it against the encoder's reconstruction. Returns the decoded frames and
/// the packet sizes, in display order (the packets of a group of frames
/// come out late, `Encoder::flush` handing out the last).
fn round_trip(cfg: Config, frames: &[Frame]) -> (Vec<Frame>, Vec<usize>) {
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    dec.set_strict(true);
    // Both coefficient readers on every block, held to each other.
    dec.set_coefficient_reader(av1::decoder::CoefficientReader::Checked);
    let mut out = Vec::new();
    let mut sizes = Vec::new();
    let mut take = |pkt: Vec<u8>, enc: &Encoder| {
        let i = out.len();
        sizes.push(pkt.len());
        let d = dec
            .decode(&pkt)
            .unwrap()
            .expect("every packet shows a frame");
        assert_eq!(
            &d,
            enc.reconstruction().unwrap(),
            "frame {i}: decoded frame differs from the encoder's reconstruction"
        );
        out.push(d);
    };
    for f in frames {
        let pkt = enc.encode(f).unwrap();
        if !pkt.is_empty() {
            take(pkt, &enc);
        }
    }
    while let Some(pkt) = enc.flush().unwrap() {
        take(pkt, &enc);
    }
    assert_eq!(out.len(), frames.len(), "a packet per frame");
    (out, sizes)
}

#[test]
fn key_frames_decode_to_the_reconstruction() {
    let frames: Vec<Frame> = (0..3).map(|t| synthetic(64, 48, t, 8)).collect();
    for q in [20, 100, 200] {
        let mut cfg = cfg_for(64, 48);
        cfg.quantizer = q;
        cfg.keyframe_interval = 1;
        let (dec, _) = round_trip(cfg, &frames);
        assert!(psnr(&frames[0], &dec[0], 0) > 20.0);
    }
}

#[test]
fn odd_sizes() {
    for (w, h) in [(1, 1), (7, 5), (66, 66), (130, 34), (16, 200)] {
        let frames: Vec<Frame> = (0..3).map(|t| synthetic(w, h, t, 8)).collect();
        let mut cfg = cfg_for(w, h);
        cfg.quantizer = 60;
        round_trip(cfg, &frames);
    }
}

#[test]
fn inter_frames_decode_to_the_reconstruction() {
    let frames: Vec<Frame> = (0..6).map(|t| synthetic(96, 64, t, 8)).collect();
    let mut cfg = cfg_for(96, 64);
    cfg.quantizer = 80;
    let (dec, sizes) = round_trip(cfg, &frames);
    // Panning content: inter frames are much cheaper than the key frame.
    assert!(sizes[1..].iter().all(|&s| s < sizes[0]), "{sizes:?}");
    for (s, d) in frames.iter().zip(&dec) {
        assert!(psnr(s, d, 0) > 30.0);
    }
}

#[test]
fn ten_bit() {
    let frames: Vec<Frame> = (0..3).map(|t| synthetic(64, 64, t, 10)).collect();
    let mut cfg = cfg_for(64, 64);
    cfg.bit_depth = 10;
    cfg.quantizer = 90;
    let (dec, _) = round_trip(cfg, &frames);
    assert!(psnr(&frames[2], &dec[2], 0) > 30.0);
}

/// 10-bit PQ (HDR10) and HLG: the colour description and the HDR
/// metadata OBUs reach a fresh decoder, on every frame.
#[test]
fn hdr_signalling_round_trips() {
    use av1::{ColorInfo, ContentLightLevel, HdrMetadata, MasteringDisplay};
    let frames: Vec<Frame> = (0..4).map(|t| synthetic(64, 48, t, 10)).collect();
    let mdcv = MasteringDisplay {
        // BT.2020 primaries, D65, 1000 / 0.005 cd/m2.
        primaries: [[46_396, 19_235], [11_141, 52_429], [8_651, 3_015]],
        white_point: [20_493, 21_561],
        luminance_max: 1000 << 8,
        luminance_min: 82,
    };
    let cll = ContentLightLevel {
        max_cll: 1000,
        max_fall: 400,
    };
    for (tc, hdr) in [
        (
            16,
            HdrMetadata {
                content_light: Some(cll),
                mastering_display: Some(mdcv),
            },
        ),
        (18, HdrMetadata::default()),
    ] {
        let color = ColorInfo {
            color_primaries: 9,
            transfer_characteristics: tc,
            matrix_coefficients: 9,
            full_range: false,
            chroma_sample_position: 2,
        };
        let mut cfg = cfg_for(64, 48);
        cfg.bit_depth = 10;
        cfg.keyframe_interval = 2;
        cfg.color = color;
        cfg.hdr = hdr;
        let (dec, _) = round_trip(cfg, &frames);
        for d in &dec {
            assert_eq!(d.color, color);
            assert_eq!(d.hdr, hdr);
        }
    }
    // The identity matrix needs 4:4:4: refused by name.
    let mut cfg = cfg_for(16, 16);
    cfg.color.matrix_coefficients = 0;
    let err = Encoder::new(cfg)
        .encode(&synthetic(16, 16, 0, 8))
        .unwrap_err();
    assert!(err.to_string().contains("4:4:4"), "{err}");
}

/// `force_keyframe()`: the next frame is a key frame a fresh decoder can
/// start at; the interval restarts from it.
#[test]
fn forced_key_frames() {
    let frames: Vec<Frame> = (0..9).map(|t| synthetic(64, 48, t, 8)).collect();
    let mut cfg = cfg_for(64, 48);
    cfg.keyframe_interval = 4;
    // (A unit a frame as it comes; alt_ref_groups_round_trip forces key
    // frames in groups.)
    cfg.altref = 0;
    let mut enc = Encoder::new(cfg);
    let mut keys = Vec::new();
    let mut packets = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if i == 2 {
            enc.force_keyframe();
        }
        assert_eq!(enc.next_is_keyframe(), [0, 2, 6].contains(&i), "frame {i}");
        packets.push(enc.encode(f).unwrap());
        keys.push(enc.last_was_keyframe());
        // From the forced key on, a decoder that starts there agrees.
        if i >= 2 {
            let mut d = Decoder::new();
            d.set_strict(true);
            let mut last = None;
            for p in &packets[2..] {
                last = d.decode(p).unwrap();
            }
            assert_eq!(last.as_ref(), enc.reconstruction(), "frame {i}");
        }
    }
    assert_eq!(
        keys,
        [true, false, true, false, false, false, true, false, false]
    );
}

/// Several tile columns (asked for, and forced by a frame wider than
/// 4096): every tile decodes to the reconstruction.
#[test]
fn tiles() {
    let frames: Vec<Frame> = (0..3).map(|t| synthetic(256, 64, t, 8)).collect();
    let mut cfg = cfg_for(256, 64);
    cfg.tile_cols_log2 = 2;
    let (dec, sizes) = round_trip(cfg.clone(), &frames);
    assert!(psnr(&frames[1], &dec[1], 0) > 30.0, "{sizes:?}");
    // In parallel: the same stream.
    cfg.threads = 4;
    let (dec4, sizes4) = round_trip(cfg, &frames);
    assert_eq!(sizes, sizes4);
    assert_eq!(dec, dec4);
    let frames: Vec<Frame> = (0..2).map(|t| synthetic(4104, 8, t, 8)).collect();
    let mut cfg = cfg_for(4104, 8);
    cfg.speed = 8;
    cfg.tools = av1::Tools::for_speed(8);
    round_trip(cfg, &frames);
}

/// Every tool of the default speed (loop restoration, CDEF, compound,
/// chroma from luma) on a small natural crop, and palettes on a frame of
/// few colours: decoded to the reconstruction. Cheap enough for the
/// overflow-checked run.
/// Alt-ref groups: the encoder holds frames, then hands out one temporal
/// unit per frame in order (the first of each group carrying the hidden
/// frames of the pyramid coded before it, the hidden ones shown with
/// `show_existing_frame`), each decoding to the reconstruction it reports;
/// key frames forced or by the interval cut the groups, and the stream
/// ends in the middle of one.
#[test]
fn alt_ref_groups_round_trip() {
    let src = natural(13);
    let mut cfg = cfg_for(src[0].width, src[0].height);
    cfg.altref = 8;
    cfg.keyframe_interval = 9;
    cfg.quantizer = 100;
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    dec.set_strict(true);
    let mut shown = 0;
    let mut check = |tu: Vec<u8>, enc: &Encoder| {
        let d = dec.decode(&tu).unwrap().expect("a shown frame per unit");
        assert_eq!(&d, enc.reconstruction().unwrap(), "frame {shown}");
        assert_eq!(enc.last_was_keyframe(), [0, 9, 11].contains(&shown));
        assert!(
            psnr(&src[shown], &d, 0) > 28.0,
            "frame {shown} shown in order"
        );
        shown += 1;
    };
    let mut held = 0;
    for (i, f) in src.iter().enumerate() {
        if i == 11 {
            enc.force_keyframe();
        }
        assert_eq!(enc.next_is_keyframe(), [0, 9, 11].contains(&i), "frame {i}");
        let tu = enc.encode(f).unwrap();
        if tu.is_empty() {
            held += 1;
        } else {
            check(tu, &enc);
        }
    }
    while let Some(tu) = enc.flush().unwrap() {
        check(tu, &enc);
    }
    assert_eq!(shown, src.len());
    assert!(held > 0, "the encoder held frames for its groups");
}

#[test]
fn monochrome() {
    // Luma only (`mono_chrome` = 1, as an AVIF alpha plane is coded), full
    // range: key frames and inter frames, an odd size, every tool the
    // speed brings; each decodes to the reconstruction and stays mono.
    let src = natural(3);
    for (w, h, speed, key) in [(96, 64, 6, 1), (77, 41, 8, 1), (96, 64, 8, 3)] {
        let frames: Vec<Frame> = src
            .iter()
            .map(|f| {
                let mut m = Frame::new(w, h, 8, ChromaFormat::Mono);
                for y in 0..h {
                    for x in 0..w {
                        m.set_sample(0, x, y, f.sample(0, x + 40, y + 30));
                    }
                }
                m
            })
            .collect();
        let mut cfg = Config::new(w, h);
        cfg.monochrome = true;
        cfg.color.full_range = true;
        cfg.quantizer = 60;
        cfg.speed = speed;
        cfg.tools = av1::Tools::for_speed(speed);
        cfg.keyframe_interval = key;
        let (dec, _) = round_trip(cfg, &frames);
        for (a, b) in frames.iter().zip(&dec) {
            assert_eq!(b.chroma, ChromaFormat::Mono);
            assert!(
                b.color.full_range,
                "the sequence header carries color_range 1"
            );
            let q = psnr(a, b, 0);
            assert!(q > 38.0, "{w}x{h} speed {speed}: luma {q:.1} dB");
        }
    }
    // A 4:2:0 frame is refused by a monochrome encoder.
    let mut cfg = Config::new(16, 16);
    cfg.monochrome = true;
    let e = Encoder::new(cfg).encode(&Frame::new(16, 16, 8, ChromaFormat::Yuv420));
    assert!(e.is_err());
}

/// `src` cropped to `w` x `h` (from 40, 30) in another format: luma
/// copied, chroma taken from the 4:2:0 source's nearest sample (finer
/// chroma repeats it), and the samples scaled to `bit_depth`.
fn reformat(src: &Frame, w: u32, h: u32, bit_depth: u32, chroma: ChromaFormat) -> Frame {
    let mut f = Frame::new(w, h, bit_depth, chroma);
    let (sx, sy) = chroma.shifts();
    for p in 0..chroma.num_planes() {
        let pl = f.planes[p];
        let (sx, sy) = if p == 0 { (0, 0) } else { (sx, sy) };
        for y in 0..pl.height {
            for x in 0..pl.width {
                // Luma coordinates, then the source plane's.
                let (lx, ly) = ((x << sx) + 40, (y << sy) + 30);
                let (px, py) = if p == 0 { (lx, ly) } else { (lx >> 1, ly >> 1) };
                let v = src.sample(p, px, py) as u32;
                f.set_sample(
                    p,
                    x,
                    y,
                    (v << (bit_depth - 8) | v >> (16 - bit_depth)) as u16,
                );
            }
        }
    }
    f
}

#[test]
fn profiles_one_and_two() {
    // 4:4:4 (profile 1), 4:2:2 and 12 bits (profile 2), at a speed with
    // the full search and at a fast one: key and inter frames in alt-ref
    // groups, every one decoding to the reconstruction, in its format.
    let src = natural(4);
    for (chroma, bit_depth, speed) in [
        (ChromaFormat::Yuv444, 8, 4),
        (ChromaFormat::Yuv444, 10, 8),
        (ChromaFormat::Yuv422, 8, 6),
        (ChromaFormat::Yuv422, 10, 4),
        (ChromaFormat::Yuv420, 12, 6),
        (ChromaFormat::Yuv422, 12, 8),
        (ChromaFormat::Yuv444, 12, 6),
        (ChromaFormat::Mono, 12, 6),
    ] {
        let (w, h) = (90, 66);
        let frames: Vec<Frame> = src
            .iter()
            .map(|f| reformat(f, w, h, bit_depth, chroma))
            .collect();
        let mut cfg = Config::new(w, h);
        cfg.chroma = chroma;
        cfg.bit_depth = bit_depth;
        cfg.quantizer = 80;
        cfg.speed = if cfg!(debug_assertions) { 8 } else { speed };
        cfg.tools = av1::Tools::for_speed(cfg.speed);
        cfg.altref = 3;
        let (dec, _) = round_trip(cfg, &frames);
        for (a, b) in frames.iter().zip(&dec) {
            assert_eq!((b.chroma, b.bit_depth), (chroma, bit_depth));
            for p in 0..chroma.num_planes() {
                let q = psnr(a, b, p);
                assert!(q > 34.0, "{chroma:?} {bit_depth}-bit plane {p}: {q:.1} dB");
            }
        }
    }
    // RGB (the identity matrix) needs 4:4:4.
    let mut cfg = Config::new(16, 16);
    cfg.color.matrix_coefficients = 0;
    let frame = Frame::new(16, 16, 8, ChromaFormat::Yuv420);
    assert!(Encoder::new(cfg.clone()).encode(&frame).is_err());
    cfg.chroma = ChromaFormat::Yuv444;
    let frame = Frame::new(16, 16, 8, ChromaFormat::Yuv444);
    let (dec, _) = round_trip(cfg, &[frame]);
    assert_eq!(dec[0].color.matrix_coefficients, 0);
}

#[test]
fn lossless_round_trips_sample_exact() {
    // Quantiser 0: every frame decodes to the source itself, key frames
    // and inter frames (in alt-ref groups and in display order), at each
    // bit depth and in each chroma format, at the speeds' searches.
    let src = natural(5);
    for (chroma, bit_depth, speed, altref) in [
        (ChromaFormat::Yuv420, 8, 4, 16),
        (ChromaFormat::Yuv420, 8, 8, 0),
        (ChromaFormat::Yuv420, 10, 6, 16),
        (ChromaFormat::Yuv444, 8, 6, 16),
        (ChromaFormat::Yuv422, 12, 8, 16),
        (ChromaFormat::Mono, 8, 6, 16),
        (ChromaFormat::Yuv420, 8, 10, 0),
    ] {
        let (w, h) = (70, 50);
        let frames: Vec<Frame> = src
            .iter()
            .map(|f| reformat(f, w, h, bit_depth, chroma))
            .collect();
        let mut cfg = Config::new(w, h);
        cfg.chroma = chroma;
        cfg.bit_depth = bit_depth;
        cfg.quantizer = 0;
        cfg.speed = if cfg!(debug_assertions) { 8 } else { speed };
        cfg.tools = av1::Tools::for_speed(cfg.speed);
        cfg.altref = altref;
        let (dec, sizes) = round_trip(cfg, &frames);
        for (i, (a, b)) in frames.iter().zip(&dec).enumerate() {
            assert!(
                a.data == b.data,
                "{chroma:?} {bit_depth}-bit speed {speed}: frame {i} is not the source"
            );
        }
        // Lossless costs more than a fine quantiser, but not absurdly.
        let raw = frames[0].data.len() * frames.len();
        let total: usize = sizes.iter().sum();
        assert!(total < raw, "{total} bytes for {raw} of samples");
    }
}

/// The cropped natural frames at 96x64.
fn natural_96x64(n: usize) -> Vec<Frame> {
    natural(n)
        .iter()
        .map(|f| reformat(f, 96, 64, 8, ChromaFormat::Yuv420))
        .collect()
}

#[test]
fn filter_intra_round_trips() {
    // The recursive filter predictors among the intra candidates: key
    // frames alone and key then inter frames, at the speeds' searches; the
    // stream changes (the filters are chosen) and decodes to the
    // reconstruction.
    let frames = natural_96x64(4);
    for (speed, key) in [(4, 1), (6, 240), (8, 1), (8, 240)] {
        let speed = if cfg!(debug_assertions) { 8 } else { speed };
        let run = |on: bool| {
            let mut cfg = Config::new(96, 64);
            cfg.quantizer = 100;
            cfg.speed = speed;
            cfg.tools = av1::Tools::for_speed(speed);
            cfg.tools.filter_intra = on;
            cfg.keyframe_interval = key;
            round_trip(cfg, &frames)
        };
        let (dec, sizes) = run(true);
        let (_, without) = run(false);
        assert_ne!(
            sizes, without,
            "speed {speed}: filter intra changed nothing"
        );
        for (a, b) in frames.iter().zip(&dec) {
            assert!(psnr(a, b, 0) > 33.0);
        }
    }
}

/// Screen content: lines of "text" (glyphs of two colours on a flat
/// ground, words repeating), scrolling a sample a frame.
fn screen(w: u32, h: u32, n: u32) -> Vec<Frame> {
    (0..n)
        .map(|t| {
            let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
            for y in 0..h {
                for x in 0..w {
                    let (line, row) = ((y + t) / 10, (y + t) % 10);
                    let (cell, col) = (x / 6, x % 6);
                    // A word of three glyphs out of four, from a few words.
                    let word = (line * 7 + cell / 4).wrapping_mul(2654435761) >> 30;
                    let glyph = word * 3 + cell % 4;
                    let on = cell % 4 != 3
                        && row < 7
                        && col < 5
                        && (glyph.wrapping_mul(0x9E37_79B9) >> (row * 5 + col)) & 1 == 1;
                    f.set_sample(0, x, y, if on { 30 } else { 220 });
                }
            }
            for p in 1..3 {
                let pl = f.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        f.set_sample(p, x, y, 128 + (p as u16 - 1) * 20);
                    }
                }
            }
            f
        })
        .collect()
}

#[test]
fn intra_block_copy_round_trips() {
    // Key frames of screen content with intra block copy (blocks copied
    // from earlier in the frame, no in-loop filters), alone and with inter
    // frames after them, at the speeds' searches (the wavefront search on
    // two threads at 6 and 8): every frame decodes to the reconstruction,
    // and the copies are used.
    let frames = screen(160, 96, 3);
    for (speed, key, threads) in [(4, 1, 1), (6, 1, 2), (8, 3, 2), (6, 240, 1)] {
        let speed = if cfg!(debug_assertions) { 8 } else { speed };
        let run = |on: bool| {
            let mut cfg = Config::new(160, 96);
            cfg.quantizer = 80;
            cfg.speed = speed;
            cfg.tools = av1::Tools::for_speed(speed);
            cfg.tools.intrabc = on;
            cfg.keyframe_interval = key;
            cfg.threads = threads;
            round_trip(cfg, &frames)
        };
        let (dec, sizes) = run(true);
        let (_, without) = run(false);
        assert_ne!(
            sizes[0], without[0],
            "speed {speed}: intra block copy changed nothing"
        );
        for (a, b) in frames.iter().zip(&dec) {
            assert!(psnr(a, b, 0) > 30.0);
        }
    }
}

#[test]
fn film_grain_round_trips() {
    // Frames with grain (correlated noise, fresh each frame): coded
    // denoised with the grain signalled, every frame (grain synthesised)
    // decodes to the reconstruction; the grain the decoder synthesises is
    // about as strong as the grain added; and the stream is smaller than
    // the same frames coded with their grain.
    let (w, h) = (192, 128);
    let mut seed = 99u32;
    let mut noise = || {
        // About Gaussian: a sum of four uniforms, unit variance.
        let mut s = 0.0;
        for _ in 0..4 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s += (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5;
        }
        s * 3f64.sqrt()
    };
    let sigma = 4.0;
    // A soft picture (gradients and a slowly moving disc), the grain on it.
    let frames: Vec<Frame> = (0..4u32)
        .map(|t| {
            let mut g = Frame::new(w, h, 8, ChromaFormat::Yuv420);
            for p in 0..3 {
                let pl = g.planes[p];
                let k = if p == 0 { 1.0 } else { 2.0 };
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        let (fx, fy) = (x as f64 * k, y as f64 * k);
                        let d = (fx - 60.0 - 2.0 * t as f64).hypot(fy - 64.0);
                        let v = if p == 0 {
                            60.0 + fx * 0.5 + fy * 0.3 + if d < 30.0 { 50.0 } else { 0.0 }
                        } else {
                            110.0 + fy * 0.1 + p as f64 * 10.0
                        };
                        g.set_sample(p, x, y, v as u16);
                    }
                }
            }
            for p in 0..3 {
                let pl = g.planes[p];
                // White noise smoothed over two samples each way.
                let (pw, ph) = (pl.width as usize, pl.height as usize);
                let white: Vec<f64> = (0..pw * ph).map(|_| noise()).collect();
                let s = if p == 0 { sigma } else { sigma / 2.0 };
                for y in 0..ph {
                    for x in 0..pw {
                        let at = |dx: usize, dy: usize| {
                            white[(y + dy).min(ph - 1) * pw + (x + dx).min(pw - 1)]
                        };
                        let n = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 2.0;
                        let v = g.sample(p, x as u32, y as u32) as f64 + n * s;
                        g.set_sample(p, x as u32, y as u32, v.round().clamp(0.0, 255.0) as u16);
                    }
                }
            }
            g
        })
        .collect();
    let mut cfg = cfg_for(w, h);
    cfg.quantizer = 80;
    cfg.film_grain = true;
    let mut enc = Encoder::new(cfg.clone());
    let mut dec = Decoder::new();
    let mut plain = Decoder::new();
    plain.set_apply_grain(false);
    let mut bytes = 0;
    let mut strength = Vec::new();
    let mut check = |tu: Vec<u8>, enc: &Encoder| {
        bytes += tu.len();
        let d = dec.decode(&tu).unwrap().unwrap();
        assert_eq!(Some(&d), enc.reconstruction());
        let n = plain.decode(&tu).unwrap().unwrap();
        let pl = d.planes[0];
        let mut se = 0.0;
        for y in 0..pl.height {
            for x in 0..pl.width {
                let e = d.sample(0, x, y) as f64 - n.sample(0, x, y) as f64;
                se += e * e;
            }
        }
        strength.push((se / (pl.width * pl.height) as f64).sqrt());
    };
    for f in &frames {
        let tu = enc.encode(f).unwrap();
        if !tu.is_empty() {
            check(tu, &enc);
        }
    }
    while let Some(tu) = enc.flush().unwrap() {
        check(tu, &enc);
    }
    assert_eq!(strength.len(), frames.len());
    for s in &strength {
        assert!(
            (0.5 * sigma..1.5 * sigma).contains(s),
            "synthesised grain of strength {s:.2} for {sigma}"
        );
    }
    cfg.film_grain = false;
    let (_, sizes) = round_trip(cfg, &frames);
    let without: usize = sizes.iter().sum();
    assert!(
        bytes < without,
        "{bytes} bytes with grain synthesis, {without} without"
    );
}

#[test]
fn every_tool_round_trips() {
    let src = natural(4);
    let crop = |f: &Frame| {
        let mut c = Frame::new(96, 64, 8, ChromaFormat::Yuv420);
        for p in 0..3 {
            let pl = c.planes[p];
            for y in 0..pl.height {
                for x in 0..pl.width {
                    c.set_sample(p, x, y, f.sample(p, x + 40, y + 30));
                }
            }
        }
        c
    };
    let frames: Vec<Frame> = src.iter().map(crop).collect();
    let mut cfg = Config::new(96, 64);
    cfg.quantizer = 120;
    round_trip(cfg, &frames);
    // The speed-6 search (wavefront rows on two threads, pruning) with
    // adaptive quantisation (delta_qindex per superblock).
    let mut cfg = Config::new(96, 64);
    cfg.quantizer = 120;
    cfg.speed = 6;
    cfg.tools = av1::Tools::for_speed(6);
    cfg.tools.aq = true;
    cfg.threads = 2;
    round_trip(cfg, &frames);
    // Screen content: text-like glyphs of two colours on a flat ground.
    let screen: Vec<Frame> = (0..2u32)
        .map(|t| {
            let mut f = Frame::new(64, 64, 8, ChromaFormat::Yuv420);
            for y in 0..64u32 {
                for x in 0..64u32 {
                    let glyph = ((x / 6 + y / 9 + t).wrapping_mul(2654435761u32) >> 29) & 1 == 1
                        && x % 6 < 5
                        && y % 9 < 7;
                    f.set_sample(0, x, y, if glyph { 20 } else { 235 });
                }
            }
            for p in 1..3 {
                let pl = f.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        f.set_sample(p, x, y, 128);
                    }
                }
            }
            f
        })
        .collect();
    let mut cfg = Config::new(64, 64);
    cfg.quantizer = 80;
    round_trip(cfg, &screen);
}

/// The inter tools (switchable and dual filters, OBMC, local and global
/// warps, the compound predictions, inter-intra): searched, then each one
/// forced wherever the block can be coded with it, so that each is
/// exercised whether or not the search would take it. Every unit decodes
/// to the reconstruction, and a forced tool changes the stream.
#[test]
fn inter_tools_round_trip() {
    let src = natural(6);
    let crop = |f: &Frame| {
        let mut c = Frame::new(128, 96, 8, ChromaFormat::Yuv420);
        for p in 0..3 {
            let pl = c.planes[p];
            for y in 0..pl.height {
                for x in 0..pl.width {
                    c.set_sample(p, x, y, f.sample(p, x + 40, y + 30));
                }
            }
        }
        c
    };
    let frames: Vec<Frame> = src.iter().map(crop).collect();
    let with_tools = |speed: u32, force: u8| {
        let mut cfg = cfg_for(128, 96);
        cfg.quantizer = 100;
        if speed != cfg.speed {
            cfg.speed = speed;
            cfg.tools = av1::Tools::for_speed(speed);
            cfg.threads = 2;
        }
        let t = &mut cfg.tools;
        t.switchable_filter = true;
        t.dual_filter = true;
        t.obmc = true;
        t.warped_motion = true;
        t.global_motion = true;
        t.masked_compound = true;
        t.jnt_comp = true;
        t.interintra = true;
        t.force_inter_tool = force;
        cfg
    };
    let (_, searched) = round_trip(with_tools(4, 0), &frames);
    round_trip(with_tools(6, 0), &frames);
    for force in [
        av1::encoder::FORCE_FILTER,
        av1::encoder::FORCE_OBMC,
        av1::encoder::FORCE_WARP,
        av1::encoder::FORCE_GLOBAL,
        av1::encoder::FORCE_WEDGE,
        av1::encoder::FORCE_DIFFWTD,
        av1::encoder::FORCE_DISTANCE,
        av1::encoder::FORCE_INTERINTRA,
        av1::encoder::FORCE_WEDGE_INTERINTRA,
    ] {
        let (_, sizes) = round_trip(with_tools(4, force), &frames);
        assert_ne!(sizes, searched, "tool {force} forced changes the stream");
    }
    // The other formats: each tool forced in 4:4:4, 4:2:2 at 10 bits,
    // 4:2:0 at 12 bits and monochrome.
    for (chroma, bit_depth) in [
        (ChromaFormat::Yuv444, 8),
        (ChromaFormat::Yuv422, 10),
        (ChromaFormat::Yuv420, 12),
        (ChromaFormat::Mono, 8),
    ] {
        let frames: Vec<Frame> = src[..4]
            .iter()
            .map(|f| reformat(f, 90, 66, bit_depth, chroma))
            .collect();
        for force in 1..=av1::encoder::FORCE_WEDGE_INTERINTRA {
            let mut cfg = with_tools(4, force);
            cfg.width = 90;
            cfg.height = 66;
            cfg.chroma = chroma;
            cfg.bit_depth = bit_depth;
            cfg.altref = 3;
            round_trip(cfg, &frames);
        }
    }
}

#[test]
fn natural_video_quality_tracks_the_quantiser() {
    let src = natural(6);
    let mut last_psnr = f64::INFINITY;
    let mut last_size = 0usize;
    for q in [30, 90, 160] {
        let mut cfg = cfg_for(src[0].width, src[0].height);
        cfg.quantizer = q;
        let (dec, sizes) = round_trip(cfg, &src);
        let p: f64 = src
            .iter()
            .zip(&dec)
            .map(|(a, b)| psnr(a, b, 0))
            .sum::<f64>()
            / src.len() as f64;
        let size: usize = sizes.iter().sum();
        eprintln!("q {q}: {size} bytes, luma PSNR {p:.2} dB");
        assert!(p < last_psnr, "PSNR must fall as the quantiser rises");
        assert!(
            last_size == 0 || size < last_size,
            "size must fall as the quantiser rises"
        );
        last_psnr = p;
        last_size = size;
    }
}

/// Average-bitrate mode spends its budget: over 24 frames of natural video
/// (a key frame first) coded as they come, and 64 in alt-ref groups,
/// within 8 % of the target (`examples/av1_ratetest.rs`
/// measures 10-second clips of several kinds).
#[test]
fn rate_control_tracks_the_target() {
    let src = natural(64);
    for (target, altref) in [(8_000u64, 0), (30_000, 0), (8_000, 16), (30_000, 16)] {
        // (24 frames a frame at a time; groups of 16 are planned a group
        // at a time, and take longer to settle.)
        let src = if altref == 0 { &src[..24] } else { &src[..] };
        let mut cfg = cfg_for(src[0].width, src[0].height);
        cfg.altref = altref;
        cfg.quantizer = 60;
        cfg.keyframe_interval = 100;
        cfg.target_bits_per_frame = Some(target);
        let (_, sizes) = round_trip(cfg, src);
        let bits: u64 = sizes.iter().map(|&s| s as u64 * 8).sum();
        let ratio = bits as f64 / (target * sizes.len() as u64) as f64;
        eprintln!(
            "rate control: target {target} bits a frame, alt-ref groups {altref}, sizes {sizes:?}, {ratio:.3}x"
        );
        assert!((0.92..1.08).contains(&ratio), "{ratio:.3}x the target");
    }
}

/// The README table: sizes and PSNR at several quantisers on 8 frames of
/// natural video (a key frame then seven inter frames). Run with
/// `cargo test --release --test encode -- --ignored --nocapture`.
#[test]
#[ignore]
fn quality_table() {
    let src = natural(8);
    eprintln!(
        "| quantiser | bytes (8 frames) | key frame | per inter frame | PSNR Y | PSNR U | PSNR V |"
    );
    for q in [20, 50, 90, 130, 170, 210, 250] {
        let mut cfg = cfg_for(src[0].width, src[0].height);
        cfg.quantizer = q;
        let (dec, sizes) = round_trip(cfg, &src);
        let n = src.len() as f64;
        let p = |pl: usize| {
            src.iter()
                .zip(&dec)
                .map(|(a, b)| psnr(a, b, pl))
                .sum::<f64>()
                / n
        };
        let total: usize = sizes.iter().sum();
        let inter = (total - sizes[0]) / (sizes.len() - 1);
        eprintln!(
            "| {q} | {total} | {} | {inter} | {:.2} dB | {:.2} dB | {:.2} dB |",
            sizes[0],
            p(0),
            p(1),
            p(2)
        );
    }
    let mut cfg = cfg_for(src[0].width, src[0].height);
    cfg.quantizer = 90;
    cfg.keyframe_interval = 1;
    let (_, sizes) = round_trip(cfg, &src);
    eprintln!("all-intra at 90: {} bytes", sizes.iter().sum::<usize>());
}
