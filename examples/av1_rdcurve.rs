//! The encoder's rate-distortion measurement harness: encodes Y4M clips at
//! several quantisers, checks that every temporal unit decodes (fresh
//! decoder) to exactly the encoder's reconstruction, and writes one CSV
//! line per clip and quantiser — bytes, PSNR of each plane, and the
//! 6:1:1-weighted YUV PSNR. With `--compare base.csv` it prints the
//! Bjøntegaard delta rate of this run against the base, per clip and on
//! average (negative: fewer bits for the same quality).
//!
//! ```text
//! cargo run --release --example av1_rdcurve -- \
//!     --out new.csv [--compare base.csv] [--frames 10] [--qs 64,96,128,160,192] \
//!     [--jobs n] [--set name=value ...] clip1.y4m clip2.y4m ...
//! ```
//!
//! `--set` changes an encoder setting by name (see `apply` below: the
//! `Config` fields and the `Tools` switches), so one tool's gain is a run
//! with it and a run without it. The encodes (each clip at each quantiser)
//! run in parallel, one per core; `--jobs 1` runs them one at a time, so
//! that the times of one-thread encodes are not those of encodes
//! contending for the memory and the caches. The time of a point is its
//! encoder's alone: the calls of `encode` and `flush`, not the decoding
//! and the measuring that check them.
//! `examples/av1toy4m.rs` makes Y4M clips from AV1 streams.

use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

use av1::{Config, Decoder, Encoder, Frame};

/// A Y4M file's frames (any format the encoder takes).
fn read_y4m(path: &str, max: usize) -> Vec<Frame> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    av1::y4m::read(&data, max)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .1
}

fn sse(a: &Frame, b: &Frame, plane: usize) -> (f64, u64) {
    let pl = a.planes[plane];
    let mut se = 0f64;
    for y in 0..pl.height {
        for x in 0..pl.width {
            let d = a.sample(plane, x, y) as f64 - b.sample(plane, x, y) as f64;
            se += d * d;
        }
    }
    (se, pl.width as u64 * pl.height as u64)
}

fn psnr(se: f64, n: u64, bd: u32) -> f64 {
    let maxv = ((1u32 << bd) - 1) as f64;
    let mse = (se / n as f64).max(1e-10);
    10.0 * (maxv * maxv / mse).log10()
}

/// Sets a configuration field by name.
fn apply(cfg: &mut Config, name: &str, value: &str) {
    let n = || value.parse::<i64>().unwrap();
    match name {
        "keyframe_interval" => cfg.keyframe_interval = n() as u32,
        "search_range" => cfg.search_range = n() as i32,
        "loop_filter" => cfg.loop_filter = Some(n() as u32),
        "speed" => {
            cfg.speed = n() as u32;
            cfg.tools = av1::Tools::for_speed(cfg.speed);
        }
        "tiles" => cfg.tile_cols_log2 = n() as u32,
        "threads" => cfg.threads = n() as usize,
        "altref" => cfg.altref = n() as u32,
        _ => {
            if !cfg.tools.set(name, n() as u32) {
                panic!("unknown setting {name}");
            }
        }
    }
}

struct Point {
    clip: String,
    q: u32,
    bytes: u64,
    psnr: [f64; 4],
    seconds: f64,
}

fn run(path: &str, frames: &[Frame], q: u32, sets: &[(String, String)]) -> Point {
    let mut cfg = Config::new(frames[0].width, frames[0].height);
    cfg.bit_depth = frames[0].bit_depth;
    cfg.chroma = frames[0].chroma;
    cfg.quantizer = q;
    for (k, v) in sets {
        apply(&mut cfg, k, v);
    }
    let bd = cfg.bit_depth;
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    dec.set_strict(true);
    let mut bytes = 0u64;
    let mut se = [0f64; 3];
    let mut n = [0u64; 3];
    let mut seconds = 0f64;
    // (With alt-ref groups the encoder holds frames: a temporal unit comes
    // out for each frame, in order, but later.)
    let mut shown = 0;
    let mut take = |tu: Vec<u8>, enc: &Encoder| {
        bytes += tu.len() as u64;
        let d = dec.decode(&tu).unwrap().expect("a shown frame");
        assert!(
            Some(&d) == enc.reconstruction(),
            "{path} q {q} frame {shown}: the decoder disagrees with the reconstruction"
        );
        for p in 0..d.chroma.num_planes() {
            let (s, c) = sse(&frames[shown], &d, p);
            se[p] += s;
            n[p] += c;
        }
        shown += 1;
    };
    for f in frames {
        let start = Instant::now();
        let tu = enc.encode(f).unwrap();
        seconds += start.elapsed().as_secs_f64();
        if !tu.is_empty() {
            take(tu, &enc);
        }
    }
    loop {
        let start = Instant::now();
        let tu = enc.flush().unwrap();
        seconds += start.elapsed().as_secs_f64();
        let Some(tu) = tu else { break };
        take(tu, &enc);
    }
    assert_eq!(shown, frames.len(), "{path}: a temporal unit per frame");
    // (Monochrome: the chroma PSNRs are luma's.)
    let p: Vec<f64> = (0..3)
        .map(|i| {
            let i = if n[i] == 0 { 0 } else { i };
            psnr(se[i], n[i], bd)
        })
        .collect();
    let yuv = (6.0 * p[0] + p[1] + p[2]) / 8.0;
    Point {
        clip: std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into(),
        q,
        bytes,
        psnr: [p[0], p[1], p[2], yuv],
        seconds,
    }
}

/// Least-squares cubic through `(x, y)`.
fn cubic_fit(x: &[f64], y: &[f64]) -> [f64; 4] {
    let mut a = [[0f64; 5]; 4];
    for (&xi, &yi) in x.iter().zip(y) {
        let p = [1.0, xi, xi * xi, xi * xi * xi];
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] += p[r] * p[c];
            }
            a[r][4] += p[r] * yi;
        }
    }
    for i in 0..4 {
        let piv = (i..4)
            .max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs()))
            .unwrap();
        a.swap(i, piv);
        let pivot = a[i];
        for (r, row) in a.iter_mut().enumerate() {
            if r != i {
                let f = row[i] / pivot[i];
                for (x, p) in row.iter_mut().zip(pivot.iter()).skip(i) {
                    *x -= f * p;
                }
            }
        }
    }
    [
        a[0][4] / a[0][0],
        a[1][4] / a[1][1],
        a[2][4] / a[2][2],
        a[3][4] / a[3][3],
    ]
}

fn integral(c: &[f64; 4], lo: f64, hi: f64) -> f64 {
    let f =
        |x: f64| c[0] * x + c[1] * x * x / 2.0 + c[2] * x.powi(3) / 3.0 + c[3] * x.powi(4) / 4.0;
    f(hi) - f(lo)
}

/// Bjøntegaard delta rate (percent) of `test` against `base`: points
/// (bytes, PSNR); the log rate fitted as a cubic in PSNR.
fn bd_rate(base: &[(f64, f64)], test: &[(f64, f64)]) -> f64 {
    let fit = |pts: &[(f64, f64)]| {
        let x: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let y: Vec<f64> = pts.iter().map(|p| p.0.ln()).collect();
        (
            cubic_fit(&x, &y),
            x.iter().cloned().fold(f64::MAX, f64::min),
            x.iter().cloned().fold(f64::MIN, f64::max),
        )
    };
    let (ca, mina, maxa) = fit(base);
    let (cb, minb, maxb) = fit(test);
    let lo = mina.max(minb);
    let hi = maxa.min(maxb);
    if hi <= lo {
        return f64::NAN;
    }
    let d = (integral(&cb, lo, hi) - integral(&ca, lo, hi)) / (hi - lo);
    (d.exp() - 1.0) * 100.0
}

fn read_csv(path: &str) -> Vec<Point> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<&str> = l.split(',').collect();
            Point {
                clip: v[0].into(),
                q: v[1].parse().unwrap(),
                bytes: v[2].parse().unwrap(),
                psnr: [
                    v[3].parse().unwrap(),
                    v[4].parse().unwrap(),
                    v[5].parse().unwrap(),
                    v[6].parse().unwrap(),
                ],
                seconds: v[7].parse().unwrap(),
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = None;
    let mut compare = None;
    let mut test_csv: Option<String> = None;
    let mut frames_n = 10usize;
    let mut jobs_n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut qs = vec![64u32, 96, 128, 160, 192];
    let mut sets = Vec::new();
    let mut clips = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let mut next = || {
            i += 1;
            args[i].clone()
        };
        match a.as_str() {
            "--out" => out = Some(next()),
            "--compare" => compare = Some(next()),
            "--test" => test_csv = Some(next()),
            "--frames" => frames_n = next().parse().unwrap(),
            "--jobs" => jobs_n = next().parse::<usize>().unwrap().max(1),
            "--qs" => qs = next().split(',').map(|s| s.parse().unwrap()).collect(),
            "--set" => {
                let s = next();
                let (k, v) = s.split_once('=').expect("--set name=value");
                sets.push((k.to_string(), v.to_string()));
            }
            _ => clips.push(a.clone()),
        }
        i += 1;
    }
    if let (Some(t), Some(b)) = (&test_csv, &compare) {
        // Compare two earlier runs.
        report(&read_csv(b), &read_csv(t));
        return;
    }
    let sources: Vec<(String, Vec<Frame>)> = clips
        .iter()
        .map(|c| (c.clone(), read_y4m(c, frames_n)))
        .collect();
    let jobs: Vec<(usize, u32)> = (0..sources.len())
        .flat_map(|c| qs.iter().map(move |&q| (c, q)))
        .collect();
    let results = Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let wall = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs_n.min(jobs.len()) {
            s.spawn(|| {
                loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(c, q)) = jobs.get(j) else { break };
                    let p = run(&sources[c].0, &sources[c].1, q, &sets);
                    results.lock().unwrap().push((j, p));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let points: Vec<Point> = results.into_iter().map(|r| r.1).collect();
    let mut csv = String::from("clip,q,bytes,psnr_y,psnr_u,psnr_v,psnr_yuv,seconds\n");
    for p in &points {
        csv += &format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.3}\n",
            p.clip, p.q, p.bytes, p.psnr[0], p.psnr[1], p.psnr[2], p.psnr[3], p.seconds
        );
    }
    print!("{csv}");
    let total_secs: f64 = points.iter().map(|p| p.seconds).sum();
    eprintln!(
        "encode time {total_secs:.1} s (wall {:.1} s)",
        wall.elapsed().as_secs_f64()
    );
    if let Some(o) = out {
        std::fs::File::create(o)
            .unwrap()
            .write_all(csv.as_bytes())
            .unwrap();
    }
    if let Some(base) = compare {
        report(&read_csv(&base), &points);
    }
}

/// Prints the BD-rate of `points` against `base`, per clip and on average.
fn report(base: &[Point], points: &[Point]) {
    let names: Vec<String> = {
        let mut v: Vec<String> = points.iter().map(|p| p.clip.clone()).collect();
        v.dedup();
        v
    };
    let mut sum = [0f64; 2];
    let mut speed = [0f64; 2];
    eprintln!("clip: BD-rate PSNR-Y, PSNR-YUV (negative = fewer bits)");
    for n in &names {
        let pick = |set: &[Point], k: usize| -> Vec<(f64, f64)> {
            set.iter()
                .filter(|p| &p.clip == n)
                .map(|p| (p.bytes as f64, p.psnr[k]))
                .collect()
        };
        let y = bd_rate(&pick(base, 0), &pick(points, 0));
        let yuv = bd_rate(&pick(base, 3), &pick(points, 3));
        sum[0] += y;
        sum[1] += yuv;
        speed[0] += base
            .iter()
            .filter(|p| &p.clip == n)
            .map(|p| p.seconds)
            .sum::<f64>();
        speed[1] += points
            .iter()
            .filter(|p| &p.clip == n)
            .map(|p| p.seconds)
            .sum::<f64>();
        eprintln!("  {n}: {y:+.2} %  {yuv:+.2} %");
    }
    let k = names.len() as f64;
    eprintln!(
        "average: {:+.2} % (Y)  {:+.2} % (YUV); encode time {:.2}x the base",
        sum[0] / k,
        sum[1] / k,
        speed[1] / speed[0]
    );
}
