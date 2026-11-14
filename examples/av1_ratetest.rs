//! The rate controller's accuracy: clips coded at average bitrates; prints
//! the rate each run reached against the one asked for, over the whole
//! clip and over its first group (the key frame and the alt-ref group
//! after it: the frames planned before rate control has seen any frame of
//! the clip coded). Every temporal unit is checked to decode (fresh
//! decoder) to exactly the encoder's reconstruction.
//!
//! By default the clips are of four kinds made from the Y4M sources —
//! `static` (one frame), `motion` (a fast pan across a frame), `noise` (a
//! clip with fresh noise in every frame) and `cuts` (a different source
//! every second). With `--files` each source is a clip of its own (played
//! forwards then backwards when it is shorter than asked for);
//! `tools/noisy-y4m.py` makes a small noisy one, the hardest start.
//!
//! `--frames` takes several lengths (short clips: `--frames 24,40`), and
//! `--segment s` cuts every clip into `s`-second segments coded each by an
//! encoder of its own, as a packager coding segments in parallel does
//! (each segment is then a clip that starts cold), one line a segment.
//!
//! ```text
//! cargo run --release --example av1_ratetest -- [--speed 6] [--rates 300,1000] \
//!     [--frames 300] [--fps 30] [--key 240] [--files] [--segment 4] \
//!     [--only name] [--jobs n] clip1.y4m clip2.y4m ...
//! ```

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

fn read_y4m(path: &str, max: usize) -> Vec<Frame> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    av1::y4m::read(&data, max)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .1
}

/// Frame `i` of a clip played forwards then backwards, repeatedly.
fn ping_pong(clip: &[Frame], i: usize) -> Frame {
    let n = clip.len();
    if n == 1 {
        return clip[0].clone();
    }
    let p = i % (2 * n - 2);
    clip[if p < n { p } else { 2 * n - 2 - p }].clone()
}

/// A window of `src` (mirrored past its edges) at `(dx, dy)`.
fn pan(src: &Frame, dx: i64, dy: i64) -> Frame {
    let (w, h) = (src.width, src.height);
    let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
    for p in 0..3 {
        let (pw, ph) = (f.planes[p].width as i64, f.planes[p].height as i64);
        let s = if p == 0 { 1 } else { 2 };
        let fold = |v: i64, n: i64| {
            let m = v.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        };
        for y in 0..ph {
            for x in 0..pw {
                let v = src.sample(p, fold(x + dx / s, pw) as u32, fold(y + dy / s, ph) as u32);
                f.set_sample(p, x as u32, y as u32, v);
            }
        }
    }
    f
}

fn noisy(src: &Frame, seed: u64) -> Frame {
    let mut f = src.clone();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for v in f.data.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let n = ((s >> 40) % 25) as i32 - 12;
        *v = (*v as i32 + n).clamp(0, 255) as u8;
    }
    f
}

/// A clip's frame `i`.
type Maker<'a> = Box<dyn Fn(usize) -> Frame + Sync + 'a>;

/// One encode: a clip (`name`, `make`), its frames `start..start + frames`
/// (a segment of it), at `rate` kb/s.
struct Job {
    clip: usize,
    frames: usize,
    start: usize,
    rate: u64,
}

/// The frames of a group (the encoder's default), and with the key frame
/// before them the first group's temporal units.
const GROUP: usize = 16;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut speed = 6u32;
    let mut rates = vec![300u64, 1000];
    let mut lengths = vec![300usize];
    let mut fps = 30u64;
    let mut key = 240u32;
    let mut files = false;
    let mut segment: Option<usize> = None;
    let mut only: Option<String> = None;
    let mut jobs_n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut clips = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut next = || it.next().expect("a value");
        let list = |v: String| -> Vec<u64> { v.split(',').map(|v| v.parse().unwrap()).collect() };
        match a.as_str() {
            "--speed" => speed = next().parse().unwrap(),
            "--rates" => rates = list(next()),
            "--frames" => lengths = list(next()).into_iter().map(|v| v as usize).collect(),
            "--fps" => fps = next().parse().unwrap(),
            "--key" => key = next().parse().unwrap(),
            "--files" => files = true,
            "--segment" => segment = Some(next().parse().unwrap()),
            "--only" => only = Some(next()),
            "--jobs" => jobs_n = next().parse().unwrap(),
            _ => clips.push(a),
        }
    }
    assert!(!clips.is_empty(), "no clips");
    let longest = *lengths.iter().max().unwrap();
    let sources: Vec<Vec<Frame>> = clips
        .iter()
        .map(|c| read_y4m(c, if files { longest } else { 60 }))
        .collect();
    let names: Vec<String> = clips
        .iter()
        .map(|c| {
            let f = c.rsplit('/').next().unwrap();
            f.strip_suffix(".y4m").unwrap_or(f).to_string()
        })
        .collect();
    let kinds: Vec<(&str, Maker)> = if files {
        sources
            .iter()
            .zip(&names)
            .map(|(s, n)| (n.as_str(), Box::new(|i| ping_pong(s, i)) as Maker))
            .collect()
    } else {
        vec![
            ("static", Box::new(|_| sources[0][0].clone())),
            (
                "motion",
                Box::new(|i| pan(&sources[1 % sources.len()][0], 13 * i as i64, 5 * i as i64)),
            ),
            (
                "noise",
                Box::new(|i| noisy(&ping_pong(&sources[2 % sources.len()], i), i as u64)),
            ),
            (
                "cuts",
                Box::new(|i| {
                    let s = &sources[(i / fps as usize) % sources.len()];
                    ping_pong(s, i)
                }),
            ),
        ]
    };
    let mut jobs = Vec::new();
    for (k, kind) in kinds.iter().enumerate() {
        if only.as_deref().is_some_and(|o| o != kind.0) {
            continue;
        }
        for &n in &lengths {
            for &rate in &rates {
                let seg = segment.map_or(n, |s| s * fps as usize);
                for start in (0..n).step_by(seg) {
                    jobs.push(Job {
                        clip: k,
                        frames: seg.min(n - start),
                        start,
                        rate,
                    });
                }
            }
        }
    }
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs_n.min(jobs.len()) {
            s.spawn(|| {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(j) else { break };
                    let (name, make) = &kinds[job.clip];
                    let first = make(job.start);
                    let mut cfg = Config::new(first.width, first.height);
                    cfg.speed = speed;
                    cfg.tools = av1::Tools::for_speed(speed);
                    cfg.keyframe_interval = key;
                    cfg.target_bits_per_frame = Some(job.rate * 1000 / fps);
                    let mut enc = Encoder::new(cfg);
                    let mut dec = Decoder::new();
                    let mut sizes = Vec::new();
                    let mut qs = Vec::new();
                    // (Each frame's luma PSNR, summed: their average is reported.)
                    let mut psnr = 0f64;
                    let mut take = |tu: Vec<u8>, enc: &Encoder| {
                        let i = job.start + sizes.len();
                        sizes.push(tu.len());
                        let d = dec.decode(&tu).unwrap().expect("a shown frame");
                        let r = enc.reconstruction().unwrap();
                        assert!(
                            d.data == r.data,
                            "{name} frame {}: decoder != reconstruction",
                            sizes.len() - 1
                        );
                        let s = make(i);
                        let n = (s.width * s.height) as usize;
                        let mse = (0..n)
                            .map(|k| {
                                let (x, y) = ((k as u32) % s.width, (k as u32) / s.width);
                                let e = s.sample(0, x, y) as f64 - d.sample(0, x, y) as f64;
                                e * e
                            })
                            .sum::<f64>()
                            / n as f64;
                        psnr += 10.0 * (255.0 * 255.0 / mse.max(1e-10)).log10();
                    };
                    for i in job.start..job.start + job.frames {
                        let tu = enc.encode(&make(i)).unwrap();
                        if !tu.is_empty() {
                            qs.push(enc.quantizer());
                            take(tu, &enc);
                        }
                    }
                    while let Some(tu) = enc.flush().unwrap() {
                        qs.push(enc.quantizer());
                        take(tu, &enc);
                    }
                    assert_eq!(sizes.len(), job.frames, "{name}: a temporal unit per frame");
                    // kb/s over the first `n` frames, against the target.
                    let rate = |n: usize| -> f64 {
                        let bytes: usize = sizes[..n].iter().sum();
                        bytes as f64 * 8.0 * fps as f64 / n as f64 / 1000.0
                    };
                    let dev = |r: f64| (r / job.rate as f64 - 1.0) * 100.0;
                    let all = rate(job.frames);
                    let g = rate(job.frames.min(GROUP + 1));
                    let label = match segment {
                        Some(_) => format!("{name} [{}..{})", job.start, job.start + job.frames),
                        None => format!("{name} x{}", job.frames),
                    };
                    results.lock().unwrap().push((
                        j,
                        format!(
                            "{label:28} {:5} kb/s: {all:8.1} kb/s ({:+6.2} %), first group {g:8.1} ({:+6.2} %), PSNR-Y {:.2} dB, quantiser {}..{}",
                            job.rate,
                            dev(all),
                            dev(g),
                            psnr / job.frames as f64,
                            qs.iter().min().unwrap(),
                            qs.iter().max().unwrap()
                        ),
                    ));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    for (_, line) in r {
        println!("{line}");
    }
}
