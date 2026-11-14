//! Decodes an AV1 stream (IVF, or the first AV1 track of an MP4) to a Y4M
//! file: the sources of the encoder's rate-distortion measurements
//! (`examples/av1_rdcurve.rs`) come from here.
//!
//! cargo run --release --example av1toy4m -- input.{ivf,mp4} output.y4m [max_frames] [crop WxH]
//!
//! With `crop`, the centre `W`x`H` of every frame is kept.

use std::io::Write;

#[path = "common/mp4.rs"]
mod mp4;
use mp4::mp4_samples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: av1toy4m input.{{ivf,mp4}} output.y4m [max_frames] [crop WxH]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let max: usize = args.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    let crop: Option<(u32, u32)> = args.get(4).map(|s| {
        let (w, h) = s.split_once('x').expect("crop is WxH");
        (w.parse().unwrap(), h.parse().unwrap())
    });
    let packets: Vec<&[u8]> = if data.starts_with(b"DKIF") {
        av1::ivf::IvfReader::new(&data)?
            .map(|p| p.map(|p| p.data))
            .collect::<Result<_, _>>()?
    } else {
        mp4_samples(&data).ok_or("no av01 track")?
    };
    let mut dec = av1::Decoder::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let mut n = 0;
    'outer: for p in packets {
        for f in dec.decode_all(p)? {
            let (w, h) = crop.unwrap_or((f.width, f.height));
            let (x0, y0) = (((f.width - w) / 2) & !1, ((f.height - h) / 2) & !1);
            if n == 0 {
                let info = av1::y4m::Y4mInfo {
                    width: w,
                    height: h,
                    bit_depth: f.bit_depth,
                    chroma: f.chroma,
                };
                out.write_all(info.header().as_bytes())?;
            }
            out.write_all(b"FRAME\n")?;
            let (sx, sy) = f.chroma.shifts();
            for p in 0..f.chroma.num_planes() {
                let (sx, sy) = if p == 0 { (0, 0) } else { (sx, sy) };
                for y in 0..(h + sy) >> sy {
                    for x in 0..(w + sx) >> sx {
                        let v = f.sample(p, (x0 >> sx) + x, (y0 >> sy) + y);
                        if f.bit_depth > 8 {
                            out.write_all(&v.to_le_bytes())?;
                        } else {
                            out.write_all(&[v as u8])?;
                        }
                    }
                }
            }
            n += 1;
            if n >= max {
                break 'outer;
            }
        }
    }
    eprintln!("{n} frames");
    Ok(())
}
