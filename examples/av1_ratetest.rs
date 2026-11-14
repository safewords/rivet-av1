//! The rate controller's accuracy: 10-second clips (300 frames at 30
//! frames/s) of four kinds made from Y4M sources — `static` (one frame),
//! `motion` (a fast pan across a frame), `noise` (a clip with fresh noise
//! in every frame) and `cuts` (a different source every second) — each
//! coded at several average bitrates; prints the rate each run reached
//! against the one asked for. Every temporal unit is checked to decode
//! (fresh decoder) to exactly the encoder's reconstruction.
//!
//! ```text
//! cargo run --release --example av1_ratetest -- [--speed 6] [--rates 300,1000] \
//!     [--frames 300] [--key 240] clip1.y4m clip2.y4m ...
//! ```

use std::sync::Mutex;

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

fn read_y4m(path: &str, max: usize) -> Vec<Frame> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let nl = data.iter().position(|&b| b == b'\n').expect("Y4M header");
    let header = std::str::from_utf8(&data[..nl]).unwrap();
    let (mut w, mut h) = (0u32, 0u32);
    for tok in header.split_whitespace() {
        match tok.as_bytes()[0] {
            b'W' => w = tok[1..].parse().unwrap(),
            b'H' => h = tok[1..].parse().unwrap(),
            _ => {}
        }
    }
    let size = (w * h + 2 * w.div_ceil(2) * h.div_ceil(2)) as usize;
    let mut pos = nl + 1;
    let mut out = Vec::new();
    while pos < data.len() && out.len() < max {
        let fl = data[pos..].iter().position(|&b| b == b'\n').unwrap();
        pos += fl + 1;
        let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
        f.data.copy_from_slice(&data[pos..pos + size]);
        pos += size;
        out.push(f);
    }
    out
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

/// A clip kind's frame `i`.
type Maker<'a> = Box<dyn Fn(usize) -> Frame + Sync + 'a>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut speed = 6u32;
    let mut rates = vec![300u64, 1000];
    let mut frames = 300usize;
    let mut key = 240u32;
    let mut only: Option<String> = None;
    let mut clips = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut next = || it.next().expect("a value");
        match a.as_str() {
            "--speed" => speed = next().parse().unwrap(),
            "--rates" => rates = next().split(',').map(|v| v.parse().unwrap()).collect(),
            "--frames" => frames = next().parse().unwrap(),
            "--key" => key = next().parse().unwrap(),
            "--only" => only = Some(next()),
            _ => clips.push(a),
        }
    }
    assert!(!clips.is_empty(), "no clips");
    let sources: Vec<Vec<Frame>> = clips.iter().map(|c| read_y4m(c, 60)).collect();
    let kinds: Vec<(&str, Maker)> = vec![
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
                let s = &sources[(i / 30) % sources.len()];
                ping_pong(s, i)
            }),
        ),
    ];
    let jobs: Vec<(usize, u64)> = (0..kinds.len())
        .flat_map(|k| rates.iter().map(move |&r| (k, r)))
        .filter(|&(k, _)| only.as_deref().is_none_or(|o| o == kinds[k].0))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.len() {
            s.spawn(|| {
                let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (k, rate) = jobs[j];
                let (name, make) = &kinds[k];
                let first = make(0);
                let mut cfg = Config::new(first.width, first.height);
                cfg.speed = speed;
                cfg.tools = av1::Tools::for_speed(speed);
                cfg.keyframe_interval = key;
                cfg.target_bits_per_frame = Some(rate * 1000 / 30);
                let mut enc = Encoder::new(cfg);
                let mut dec = Decoder::new();
                let mut bytes = 0usize;
                let mut qs = Vec::new();
                let mut shown = 0;
                let mut take = |tu: Vec<u8>, enc: &Encoder| {
                    bytes += tu.len();
                    let d = dec.decode(&tu).unwrap().expect("a shown frame");
                    let r = enc.reconstruction().unwrap();
                    assert!(
                        d.data == r.data,
                        "{name} frame {shown}: decoder != reconstruction"
                    );
                    shown += 1;
                };
                for i in 0..frames {
                    let f = make(i);
                    let tu = enc.encode(&f).unwrap();
                    qs.push(enc.quantizer());
                    if !tu.is_empty() {
                        take(tu, &enc);
                    }
                }
                while let Some(tu) = enc.flush().unwrap() {
                    take(tu, &enc);
                }
                assert_eq!(shown, frames, "{name}: a temporal unit per frame");
                let secs = frames as f64 / 30.0;
                let got = bytes as f64 * 8.0 / secs / 1000.0;
                results.lock().unwrap().push((
                    k,
                    rate,
                    format!(
                        "{name:7} {rate:5} kb/s: {got:8.1} kb/s ({:+6.2} %), quantiser {}..{}",
                        (got / rate as f64 - 1.0) * 100.0,
                        qs.iter().min().unwrap(),
                        qs.iter().max().unwrap()
                    ),
                ));
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| (x.0, x.1));
    for (_, _, line) in r {
        println!("{line}");
    }
}
