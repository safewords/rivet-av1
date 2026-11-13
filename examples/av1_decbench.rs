//! Decoder throughput: decodes an AV1 stream (IVF, or the AV1 track of an
//! MP4) `runs` times and reports megapixels a second (shown frames times
//! their size, over the wall time of the decode calls alone).
//!
//! cargo run --release --example av1_decbench -- input.{ivf,mp4} [runs] [threads] [frames in flight]

#[path = "common/mp4.rs"]
mod mp4;

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: decbench input.{{ivf,mp4}} [runs] [threads] [frames in flight]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let runs: usize = args.get(2).map_or(3, |s| s.parse().unwrap());
    let threads: usize = args.get(3).map_or(1, |s| s.parse().unwrap());
    let flight: usize = args.get(4).map_or(1, |s| s.parse().unwrap());
    let packets: Vec<&[u8]> = if data.starts_with(b"DKIF") {
        av1::ivf::IvfReader::new(&data)?
            .map(|p| p.map(|p| p.data))
            .collect::<Result<_, _>>()?
    } else {
        mp4::mp4_samples(&data).ok_or("no av01 track")?
    };
    let mut best = f64::MAX;
    let mut pixels = 0u64;
    let mut frames = 0usize;
    for _ in 0..runs {
        let mut dec = av1::Decoder::new();
        dec.set_threads(threads);
        dec.set_frames_in_flight(flight);
        pixels = 0;
        frames = 0;
        let mut count = |shown: Vec<av1::Frame>| {
            for f in shown {
                pixels += f.width as u64 * f.height as u64;
                frames += 1;
            }
        };
        let start = Instant::now();
        if flight > 1 {
            // Units are queued; their frames are taken as they come, and
            // waited for at the end.
            for p in &packets {
                dec.queue(p);
                while let Some(shown) = dec.poll() {
                    count(shown?);
                }
            }
            while let Some(shown) = dec.wait() {
                count(shown?);
            }
        } else {
            for p in &packets {
                count(dec.decode_all(p)?);
            }
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!(
        "{}: {frames} frames, {:.2} MP/s ({:.1} frames/s), best of {runs}, {threads} thread(s), \
         {flight} frame(s) in flight",
        args[1],
        pixels as f64 / best / 1e6,
        frames as f64 / best
    );
    Ok(())
}
