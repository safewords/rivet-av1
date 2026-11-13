//! Test support: packet extraction (IVF, Matroska) and the test vector
//! runner. Shared by the integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The frames of the first video track of a Matroska/WebM file, in order.
///
/// Just enough EBML for the test vectors: descends Segment, Cluster and
/// BlockGroup, takes the payload of every SimpleBlock and Block (no
/// lacing), and skips everything else.
pub fn mkv_frames(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    walk(data, &mut out, &mut None);
    out
}

fn vint(data: &[u8], pos: usize, keep_marker: bool) -> Option<(u64, usize)> {
    let first = *data.get(pos)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > data.len() {
        return None;
    }
    let mut v = if keep_marker {
        first as u64
    } else {
        (first as u64) & ((1u64 << (8 - len)) - 1)
    };
    for i in 1..len {
        v = (v << 8) | data[pos + i] as u64;
    }
    Some((v, len))
}

fn walk(data: &[u8], out: &mut Vec<Vec<u8>>, track: &mut Option<u64>) {
    let mut pos = 0;
    while pos < data.len() {
        let Some((id, il)) = vint(data, pos, true) else {
            return;
        };
        let Some((size, sl)) = vint(data, pos + il, false) else {
            return;
        };
        let start = pos + il + sl;
        let unknown = size == (1u64 << (7 * sl)) - 1;
        if !unknown && start + size as usize > data.len() {
            return;
        }
        let end = if unknown {
            data.len()
        } else {
            start + size as usize
        };
        match id {
            0x18538067 | 0x1F43B675 | 0xA0 => walk(&data[start..end], out, track),
            0xA3 | 0xA1 => {
                let b = &data[start..end];
                if let Some((tn, tl)) = vint(b, 0, false) {
                    if track.is_none() {
                        *track = Some(tn);
                    }
                    if Some(tn) == *track && b.len() >= tl + 3 {
                        out.push(b[tl + 3..].to_vec());
                    }
                }
            }
            _ => {}
        }
        pos = end;
    }
}

/// The temporal units of a test vector (IVF or Matroska).
pub fn packets(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).unwrap();
    if data.starts_with(b"DKIF") {
        av1::ivf::IvfReader::new(&data)
            .unwrap()
            .map(|f| f.unwrap().data.to_vec())
            .collect()
    } else {
        mkv_frames(&data)
    }
}

/// The expected MD5s of a vector's shown frames.
pub fn expected_md5s(path: &Path) -> Vec<String> {
    let md5 = PathBuf::from(format!("{}.md5", path.display()));
    std::fs::read_to_string(md5)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect()
}

/// The MD5 of a frame as the vectors' .md5 files hash it: the planes
/// packed (8-bit samples, or 16-bit little-endian above 8 bits); a
/// monochrome frame is hashed with neutral (mid-grey) chroma planes, as a
/// 4:2:0 file would carry it.
pub fn frame_md5(f: &av1::Frame) -> String {
    let mut ctx = md5::Context::new();
    ctx.consume(f.packed());
    if f.chroma == av1::ChromaFormat::Mono {
        let cw = f.width.div_ceil(2) as usize;
        let ch = f.height.div_ceil(2) as usize;
        let mid = 1u16 << (f.bit_depth - 1);
        let plane: Vec<u8> = if f.bit_depth > 8 {
            std::iter::repeat_n(mid.to_le_bytes(), cw * ch)
                .flatten()
                .collect()
        } else {
            vec![mid as u8; cw * ch]
        };
        ctx.consume(&plane);
        ctx.consume(&plane);
    }
    format!("{:x}", ctx.compute())
}

/// Outcome of decoding one vector.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub name: String,
    pub frames: usize,
    pub expected: usize,
    pub matched: usize,
    pub failure: Option<String>,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.failure.is_none() && self.matched == self.expected && self.frames == self.expected
    }
}

/// Decodes a vector and compares every shown frame with its MD5.
pub fn run_vector(path: &Path) -> Outcome {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let expected = expected_md5s(path);
    let mut dec = av1::Decoder::new();
    // Both coefficient readers on every block, held to each other.
    dec.set_coefficient_reader(av1::decoder::CoefficientReader::Checked);
    // AV1_DECODE_THREADS runs the threaded decoder (tiles, post-filters).
    if let Some(n) = std::env::var("AV1_DECODE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        dec.set_threads(n);
    }
    // AV1_FRAMES_IN_FLIGHT decodes with that many frames in flight: units
    // are queued and their results waited for a few units later.
    let flight: usize = std::env::var("AV1_FRAMES_IN_FLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    dec.set_frames_in_flight(flight);
    let mut frames = 0usize;
    let mut matched = 0usize;
    let mut failure = None;
    let packets = packets(path);
    // With frames in flight a unit's frames come out when the units after
    // it have been queued, or at the end.
    let steps = if flight > 1 {
        packets.len() + flight
    } else {
        packets.len()
    };
    for i in 0..steps {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if flight <= 1 {
                return dec.decode(&packets[i]).map(|f| f.into_iter().collect());
            }
            if let Some(p) = packets.get(i) {
                dec.queue(p);
            }
            // At the end, and when units pile up, the oldest is waited
            // for; otherwise it is taken if it is ready.
            let ready = if i + 1 >= packets.len() || dec.queued() > flight {
                dec.wait()
            } else {
                dec.poll()
            };
            // (`decode` gives a unit's last frame: so here.)
            ready.map_or(Ok(Vec::new()), |r| {
                r.map(|mut f| f.pop().into_iter().collect())
            })
        }));
        let out: Vec<av1::Frame> = match out {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                failure = Some(format!("packet {i}: {e}"));
                break;
            }
            Err(_) => {
                failure = Some(format!("packet {i}: panic"));
                break;
            }
        };
        for f in out {
            let sum = frame_md5(&f);
            if failure.is_none() {
                if expected.get(frames) == Some(&sum) {
                    matched += 1;
                } else {
                    failure = Some(format!(
                        "frame {frames} ({}x{}) md5 mismatch",
                        f.width, f.height
                    ));
                }
            }
            frames += 1;
        }
        if failure.is_some() {
            break;
        }
    }
    if failure.is_none() && frames != expected.len() {
        failure = Some(format!(
            "{} frames shown, {} expected",
            frames,
            expected.len()
        ));
    }
    Outcome {
        name,
        frames,
        expected: expected.len(),
        matched,
        failure,
    }
}

/// The directory holding the downloaded vectors: `AV1_VECTOR_DIR`, else
/// tests/vectors.
pub fn vector_dir() -> PathBuf {
    match std::env::var("AV1_VECTOR_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors"),
    }
}

/// The streams in `dir` that have an .md5 next to them, sorted.
pub fn vectors_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "ivf" || e == "mkv" || e == "webm")
        })
        .filter(|p| PathBuf::from(format!("{}.md5", p.display())).exists())
        .collect();
    v.sort();
    v
}

/// Runs vectors on all cores.
pub fn run_all(paths: &[PathBuf]) -> Vec<Outcome> {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let o = run_vector(&paths[i]);
                    results.lock().unwrap().push(o);
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by(|a, b| a.name.cmp(&b.name));
    r
}
