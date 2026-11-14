//! Encodes raw 8-bit 4:2:0 planar video (I420), or a Y4M file (8- or
//! 10-bit 4:2:0), to an AV1 IVF file.
//!
//! cargo run --release --example av1_ivfenc -- input.yuv WIDTH HEIGHT output.ivf [QUANTIZER] [settings]
//! cargo run --release --example av1_ivfenc -- input.y4m output.ivf [QUANTIZER] [settings]
//!
//! Settings are `name=value`: `speed`, `tiles` (tile columns, log2),
//! `threads`, `frames` (at most this many), `key` (key frame interval),
//! `bitrate` (bits per second at 30 frames/s), `altref` (the alt-ref
//! group, 0 for none), and the `Tools` switches by name (after `speed`).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: ivfenc input.yuv WIDTH HEIGHT output.ivf [QUANTIZER] [name=value...]\n       ivfenc input.y4m output.ivf [QUANTIZER] [name=value...]";
    if args.len() < 3 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let y4m = data.starts_with(b"YUV4MPEG2");
    let (w, h, bd, mut rest, frames): (u32, u32, u32, usize, Vec<&[u8]>) = if y4m {
        let nl = data.iter().position(|&b| b == b'\n').ok_or("Y4M header")?;
        let header = std::str::from_utf8(&data[..nl])?;
        let (mut w, mut h, mut bd) = (0u32, 0u32, 8u32);
        for tok in header.split_whitespace() {
            match tok.as_bytes()[0] {
                b'W' => w = tok[1..].parse()?,
                b'H' => h = tok[1..].parse()?,
                b'C' if tok.contains("p10") => bd = 10,
                _ => {}
            }
        }
        let size = av1::Frame::new(w, h, bd, av1::ChromaFormat::Yuv420)
            .data
            .len();
        let mut pos = nl + 1;
        let mut frames = Vec::new();
        while pos < data.len() {
            let fl = data[pos..]
                .iter()
                .position(|&b| b == b'\n')
                .ok_or("FRAME")?;
            pos += fl + 1;
            frames.push(&data[pos..pos + size]);
            pos += size;
        }
        (w, h, bd, 2, frames)
    } else {
        if args.len() < 5 {
            eprintln!("{usage}");
            std::process::exit(2);
        }
        let w: u32 = args[2].parse()?;
        let h: u32 = args[3].parse()?;
        let size = av1::Frame::new(w, h, 8, av1::ChromaFormat::Yuv420)
            .data
            .len();
        (w, h, 8, 4, data.chunks_exact(size).collect())
    };
    let output = args[rest].clone();
    rest += 1;
    let mut cfg = av1::Config::new(w, h);
    cfg.bit_depth = bd;
    let mut max_frames = usize::MAX;
    for a in &args[rest..] {
        match a.split_once('=') {
            None => cfg.quantizer = a.parse()?,
            Some(("speed", v)) => {
                cfg.speed = v.parse()?;
                cfg.tools = av1::Tools::for_speed(cfg.speed);
            }
            Some(("tiles", v)) => cfg.tile_cols_log2 = v.parse()?,
            Some(("threads", v)) => cfg.threads = v.parse()?,
            Some(("frames", v)) => max_frames = v.parse()?,
            Some(("key", v)) => cfg.keyframe_interval = v.parse()?,
            Some(("altref", v)) => cfg.altref = v.parse()?,
            Some(("bitrate", v)) => {
                // Bits per second at 30 frames/s.
                cfg.target_bits_per_frame = Some(v.parse::<u64>()? / 30);
            }
            Some((k, v)) => {
                if !cfg.tools.set(k, v.parse()?) {
                    return Err(format!("unknown setting {k}").into());
                }
            }
        }
    }
    let mut enc = av1::Encoder::new(cfg);
    let mut out = av1::ivf::IvfWriter::new(w as u16, h as u16, 30, 1);
    let start = std::time::Instant::now();
    let mut n = 0;
    // (With alt-ref groups the encoder holds frames: a temporal unit comes
    // out for each frame, in order, but later.)
    let mut shown = 0u64;
    for chunk in frames.iter().take(max_frames) {
        let mut f = av1::Frame::new(w, h, bd, av1::ChromaFormat::Yuv420);
        f.data.copy_from_slice(chunk);
        let pkt = enc.encode(&f)?;
        if !pkt.is_empty() {
            out.frame(shown, &pkt);
            shown += 1;
        }
        n += 1;
    }
    while let Some(pkt) = enc.flush()? {
        out.frame(shown, &pkt);
        shown += 1;
    }
    let secs = start.elapsed().as_secs_f64();
    let bytes = out.finish();
    eprintln!(
        "{n} frames, {} bytes, {:.2} frames/s ({:.3} MP/s)",
        bytes.len(),
        n as f64 / secs,
        n as f64 * w as f64 * h as f64 / secs / 1e6
    );
    std::fs::write(&output, bytes)?;
    Ok(())
}
