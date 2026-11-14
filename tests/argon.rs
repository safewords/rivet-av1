//! The Argon conformance streams (AOMedia's "Argon Streams AV1", v2.1):
//! for every stream with a reference MD5, the MD5 of all output frames,
//! concatenated as raw planar video, must match.
//!
//! The suite (7 GB, from the AOMedia "AV1 video decoder verification
//! tool" page) is not downloaded by the tests: set `ARGON_DIR` to its
//! unpacked `argon_coveragetool_av1_base_and_extended_profiles_v2.1`
//! directory. `ARGON_FILTER=substr` restricts the run to matching paths;
//! `ARGON_REPORT=file` writes one line per stream; `ARGON_LAYERS=1` adds
//! the per-operating-point (`layers/N`) variants; `ARGON_VERBOSE=1` prints
//! each result as it comes; `AV1_STRICT=1` decodes strictly (every
//! stream conforms, so none may fail a check). Without `ARGON_DIR`
//! the test reports that it skipped.
//!
//! Each stream's `ref_cmd` script gives the options the reference output
//! was made with: the bitstream format (`--annexb`), the operating point
//! (`--oppoint=N`) and whether every layer's frames are output
//! (`--all-layers`). The error directories (streams that each break a
//! requirement of conformance, no reference output) are not run for
//! output; `argon_error_streams` decodes those whose requirement a strict
//! decoder checks and that decoding does not need (frame ids, levels).
//!
//! The large-scale-tile directories (`*_large_scale_tile*`, tile list
//! OBUs, 7.3) were made with AOMedia's `lightfield_tile_list_decoder
//! input output N L`: the frames the first `N` temporal units show are the
//! anchor frames, and each of the next `L` tile lists is decoded; its
//! output, as that tool writes it, is each decoded tile on its own (as a
//! picture of one tile's size, in the order of the list's entries), which
//! is the output frame of 7.3.1 cut at its tiles. `ARGON_FILTER=large`
//! runs only those.

use std::path::{Path, PathBuf};

struct Case {
    stream: PathBuf,
    md5: String,
    annexb: bool,
    oppoint: usize,
    all_layers: bool,
    /// For a large-scale-tile stream: how many temporal units show the
    /// anchor frames, and how many tile lists are decoded after them.
    tile_lists: Option<(usize, usize)>,
    name: String,
}

fn cases(root: &Path) -> Vec<Case> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let dname = dir.file_name().unwrap().to_string_lossy().to_string();
        if dname.contains("error") {
            continue;
        }
        let md5_dir = dir.join("md5_ref");
        let cmd_dir = dir.join("ref_cmd");
        let mut push = |md5_path: PathBuf, cmd_path: PathBuf, label: String| {
            let Ok(md5) = std::fs::read_to_string(&md5_path) else {
                return;
            };
            let Ok(cmd) = std::fs::read_to_string(&cmd_path) else {
                return;
            };
            let Some(line) = cmd
                .lines()
                .find(|l| l.contains("aomdec") || l.contains("lightfield_tile_list_decoder"))
            else {
                return;
            };
            let Some(input) = line
                .split_whitespace()
                .find(|w| w.contains("ARGON_STREAMS_INPUT"))
            else {
                return;
            };
            let file = input.rsplit('/').next().unwrap().trim_matches('"');
            let oppoint = line
                .split_whitespace()
                .find_map(|w| w.strip_prefix("--oppoint="))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            // lightfield_tile_list_decoder input output N L
            let tile_lists = if line.contains("lightfield_tile_list_decoder") {
                let w: Vec<&str> = line.split_whitespace().collect();
                let n = w.get(3).and_then(|v| v.parse().ok());
                let l = w.get(4).and_then(|v| v.parse().ok());
                let (Some(n), Some(l)) = (n, l) else {
                    return;
                };
                Some((n, l))
            } else {
                None
            };
            let stream = dir.join("streams").join(file);
            out.push(Case {
                annexb: line.contains("--annexb") || (tile_lists.is_some() && is_annexb(&stream)),
                tile_lists,
                stream,
                md5: md5.split_whitespace().next().unwrap_or("").to_string(),
                oppoint,
                all_layers: line.contains("--all-layers"),
                name: label,
            });
        };
        if let Ok(rd) = std::fs::read_dir(&md5_dir) {
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect();
            v.sort();
            for p in v {
                let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                let label = format!("{dname}/{stem}");
                push(p, cmd_dir.join(format!("{stem}.sh")), label);
            }
        }
        let layers = std::env::var("ARGON_LAYERS").is_ok();
        if let Ok(rd) = std::fs::read_dir(md5_dir.join("layers"))
            .map_err(|_| ())
            .and_then(|r| if layers { Ok(r) } else { Err(()) })
        {
            let mut layers: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            layers.sort();
            for l in layers {
                let n = l.file_name().unwrap().to_string_lossy().to_string();
                let Ok(rd) = std::fs::read_dir(&l) else {
                    continue;
                };
                let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                v.sort();
                for p in v {
                    let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                    let label = format!("{dname}/layers/{n}/{stem}");
                    push(
                        p,
                        cmd_dir.join("layers").join(&n).join(format!("{stem}.sh")),
                        label,
                    );
                }
            }
        }
    }
    out
}

/// Whether a stream reads as Annex B (5.2's length-delimited format)
/// throughout: the reference command of a large-scale-tile stream does not
/// say, and the suite has them in both formats.
fn is_annexb(stream: &Path) -> bool {
    let Ok(data) = std::fs::read(stream) else {
        return false;
    };
    let Ok(tus) = av1::annexb_temporal_units(&data) else {
        return false;
    };
    // Each temporal unit is frame units, each of OBUs, sizes adding up;
    // a low-overhead stream starts with a temporal delimiter OBU (0x12),
    // which as a size would rarely add up.
    tus.iter().all(|tu| {
        let mut pos = 0;
        while pos < tu.len() {
            let Some((fu, n)) = leb128(&tu[pos..]) else {
                return false;
            };
            pos += n;
            let end = pos + fu;
            if end > tu.len() {
                return false;
            }
            while pos < end {
                let Some((obu, n)) = leb128(&tu[pos..end]) else {
                    return false;
                };
                pos += n + obu;
                if pos > end || obu == 0 {
                    return false;
                }
            }
        }
        true
    })
}

fn leb128(d: &[u8]) -> Option<(usize, usize)> {
    let mut v = 0usize;
    for i in 0..8 {
        let b = *d.get(i)?;
        v |= ((b & 0x7f) as usize) << (7 * i);
        if b & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

/// Decodes a large-scale-tile case as `lightfield_tile_list_decoder`
/// does: the last frame each of the first `anchors` temporal units shows
/// is an anchor frame; after them every frame a temporal unit shows is
/// the output frame of a tile list, and each of the first `lists` is
/// hashed tile by tile.
fn run_tile_lists(c: &Case, units: &[&[u8]], anchors: usize, lists: usize) -> Result<bool, String> {
    let mut dec = av1::Decoder::new();
    dec.set_strict(strict());
    if let Some(n) = std::env::var("AV1_DECODE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        dec.set_threads(n);
    }
    let decode = |dec: &mut av1::Decoder, tu: &[u8]| {
        if c.annexb {
            dec.decode_annexb(tu)
        } else {
            dec.decode_all(tu)
        }
        .map_err(|e| e.to_string())
    };
    let mut frames = Vec::new();
    for tu in units.iter().take(anchors) {
        frames.extend(decode(&mut dec, tu)?.pop());
    }
    dec.set_anchor_frames(&frames);
    let mut out = md5::Context::new();
    let mut done = 0;
    let rest = &units[anchors.min(units.len())..];
    // AV1_FRAMES_IN_FLIGHT: the tile lists are queued, with that many in
    // flight (Annex B units given as section 5 units, which is what
    // `queue` takes).
    let flight: usize = std::env::var("AV1_FRAMES_IN_FLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    if flight > 1 {
        dec.set_frames_in_flight(flight);
        let mut take = |dec: &mut av1::Decoder, r: av1::Result<Vec<av1::Frame>>| {
            for f in r.map_err(|e| e.to_string())? {
                if done < lists {
                    let l = dec
                        .tile_list_layout()
                        .ok_or("a frame that is not a tile list's")?;
                    hash_tiles(&mut out, &f, l);
                    done += 1;
                }
            }
            Ok::<(), String>(())
        };
        for tu in rest {
            let tu = if c.annexb { section5(tu) } else { tu.to_vec() };
            dec.queue(&tu);
            while let Some(r) = dec.poll() {
                take(&mut dec, r)?;
            }
        }
        while let Some(r) = dec.wait() {
            take(&mut dec, r)?;
        }
        return Ok(format!("{:x}", out.compute()) == c.md5);
    }
    for tu in rest {
        if done == lists {
            break;
        }
        for f in decode(&mut dec, tu)? {
            if done == lists {
                break;
            }
            let l = dec
                .tile_list_layout()
                .ok_or("a frame that is not a tile list's")?;
            hash_tiles(&mut out, &f, l);
            done += 1;
        }
    }
    Ok(format!("{:x}", out.compute()) == c.md5)
}

/// Hashes a tile list's output frame tile by tile, as
/// `lightfield_tile_list_decoder` writes it.
fn hash_tiles(out: &mut md5::Context, f: &av1::Frame, l: av1::TileListLayout) {
    let across = (f.width / l.tile_width) as usize;
    for t in 0..l.tiles {
        let x = l.tile_width as usize * (t % across);
        let y = l.tile_height as usize * (t / across);
        for p in 0..f.planes.len() {
            let (sx, sy) = if p == 0 { (0, 0) } else { f.chroma.shifts() };
            let bps = f.bytes_per_sample();
            let pw = f.planes[p].width as usize;
            let (x0, y0) = (x >> sx, y >> sy);
            let w = l.tile_width as usize >> sx;
            let h = l.tile_height as usize >> sy;
            let plane = f.plane(p);
            for row in y0..y0 + h {
                out.consume(&plane[(row * pw + x0) * bps..(row * pw + x0 + w) * bps]);
            }
        }
    }
}

/// An Annex B temporal unit as a section 5 one: each OBU with its size
/// field.
fn section5(tu: &[u8]) -> Vec<u8> {
    let leb = |d: &[u8], p: &mut usize| {
        let mut v = 0usize;
        for i in 0..8 {
            let b = d[*p];
            *p += 1;
            v |= ((b & 0x7f) as usize) << (7 * i);
            if b & 0x80 == 0 {
                break;
            }
        }
        v
    };
    let mut out = Vec::new();
    let mut p = 0;
    while p < tu.len() {
        let end = leb(tu, &mut p) + p;
        while p < end {
            let len = leb(tu, &mut p);
            let obu = &tu[p..p + len];
            p += len;
            let h = obu[0];
            let ext = ((h >> 2) & 1) as usize;
            let mut q = 1 + ext;
            if h & 2 != 0 {
                leb(obu, &mut q);
            }
            out.push(h | 2);
            out.extend_from_slice(&obu[1..1 + ext]);
            let mut size = obu.len() - q;
            loop {
                let b = (size & 0x7f) as u8;
                size >>= 7;
                out.push(if size > 0 { b | 0x80 } else { b });
                if size == 0 {
                    break;
                }
            }
            out.extend_from_slice(&obu[q..]);
        }
    }
    out
}

/// Decodes a case: `Ok(true)` when the output MD5 matches.
fn run(c: &Case) -> Result<bool, String> {
    let data = std::fs::read(&c.stream).map_err(|e| e.to_string())?;
    if let Some((anchors, lists)) = c.tile_lists {
        let units = if c.annexb {
            av1::annexb_temporal_units(&data).map_err(|e| e.to_string())?
        } else {
            split_at_delimiters(&data)
        };
        return run_tile_lists(c, &units, anchors, lists);
    }
    let mut dec = av1::Decoder::new();
    dec.set_strict(strict());
    // AV1_DECODE_THREADS runs the threaded decoder (tiles, post-filters).
    if let Some(n) = std::env::var("AV1_DECODE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        dec.set_threads(n);
    }
    dec.set_operating_point(c.oppoint);
    let mut out = md5::Context::new();
    // Monochrome output as aomdec writes it with --rawvideo: we hash with
    // and without neutral chroma planes and accept either.
    let mut out_mono = md5::Context::new();
    let mut emit = |frames: Vec<av1::Frame>| {
        let frames = if c.all_layers {
            frames
        } else {
            frames.into_iter().last().into_iter().collect()
        };
        for f in frames {
            out.consume(f.packed());
            out_mono.consume(f.packed());
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
                out_mono.consume(&plane);
                out_mono.consume(&plane);
            }
        }
    };
    if c.annexb {
        for tu in av1::annexb_temporal_units(&data).map_err(|e| e.to_string())? {
            emit(dec.decode_annexb(tu).map_err(|e| e.to_string())?);
        }
    } else {
        // Section 5's low-overhead format: temporal units split at the
        // temporal delimiters.
        for tu in split_at_delimiters(&data) {
            emit(dec.decode_all(tu).map_err(|e| e.to_string())?);
        }
    }
    let a = format!("{:x}", out.compute());
    let b = format!("{:x}", out_mono.compute());
    Ok(a == c.md5 || b == c.md5)
}

/// `AV1_STRICT`: decode strictly.
fn strict() -> bool {
    std::env::var("AV1_STRICT").is_ok_and(|v| v != "0")
}

/// Splits a low-overhead OBU stream into temporal units at its temporal
/// delimiter OBUs.
fn split_at_delimiters(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let h = data[pos];
        let ty = (h >> 3) & 15;
        let ext = (h >> 2) & 1;
        let has_size = (h >> 1) & 1;
        if ty == 2 {
            starts.push(pos);
        }
        let mut p = pos + 1 + ext as usize;
        if has_size == 0 {
            break;
        }
        let mut size = 0usize;
        for i in 0..8 {
            let Some(&b) = data.get(p) else {
                return vec![data];
            };
            p += 1;
            size |= ((b & 0x7f) as usize) << (7 * i);
            if b & 0x80 == 0 {
                break;
            }
        }
        pos = p + size;
    }
    if starts.is_empty() || starts[0] != 0 {
        starts.insert(0, 0);
    }
    let mut out = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(data.len());
        out.push(&data[s..e]);
    }
    out
}

#[test]
fn argon() {
    let Ok(root) = std::env::var("ARGON_DIR") else {
        eprintln!("ARGON_DIR not set; Argon conformance streams skipped");
        return;
    };
    let mut cs = cases(Path::new(&root));
    if let Ok(f) = std::env::var("ARGON_FILTER") {
        if f == "large" {
            cs.retain(|c| c.tile_lists.is_some());
        } else {
            cs.retain(|c| c.name.contains(&f));
        }
    }
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= cs.len() {
                        break;
                    }
                    let c = &cs[i];
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(c)));
                    let r = match r {
                        Ok(r) => r,
                        Err(_) => Err("panic".to_string()),
                    };
                    if std::env::var("ARGON_VERBOSE").is_ok() {
                        eprintln!(
                            "{} {}",
                            if matches!(r, Ok(true)) {
                                "PASS"
                            } else {
                                "FAIL"
                            },
                            c.name
                        );
                    }
                    results.lock().unwrap().push((c.name.clone(), r));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut report = String::new();
    let mut by_dir: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (name, r) in &results {
        let dir = name.split('/').next().unwrap().to_string()
            + if name.contains("/layers/") {
                "/layers"
            } else {
                ""
            };
        let e = by_dir.entry(dir).or_default();
        e.1 += 1;
        let line = match r {
            Ok(true) => {
                e.0 += 1;
                format!("PASS {name}\n")
            }
            Ok(false) => format!("FAIL {name}: md5 mismatch\n"),
            Err(m) => format!("FAIL {name}: {m}\n"),
        };
        report.push_str(&line);
    }
    let total_pass: usize = by_dir.values().map(|v| v.0).sum();
    for (d, (p, t)) in &by_dir {
        eprintln!("{d}: {p}/{t}");
    }
    eprintln!("Argon: {total_pass}/{} pass", results.len());
    if let Ok(f) = std::env::var("ARGON_REPORT") {
        std::fs::write(f, report).unwrap();
    }
}

/// The error streams that break a requirement of conformance decoding
/// does not need, each with the requirement (as the suite's stream
/// documentation names it) and words of the error a strict decoder must
/// give for it. A decoder that is not strict decodes them without that
/// error (but where the stream refers to a slot never filled, which
/// nothing can decode).
const ERROR_STREAMS: &[(&str, &str, &str)] = &[
    ("profile0_error", "test101", "MaxTiles"),
    ("profile0_error", "test106", "TileWidth"),
    ("profile0_error", "test110", "FrameHeight"),
    ("profile0_error", "test111", "CroppedTileWidth"),
    ("profile0_error", "test144", "expectedFrameId"),
    (
        "profile0_error",
        "test17",
        "RefValid[ frame_to_show_map_idx ]",
    ),
    ("profile0_error", "test208", "MaxPicSize"),
    ("profile0_error", "test21", "PrevFrameID"),
    ("profile0_error", "test216", "MaxTileCols"),
    ("profile0_error", "test218", "MAX_TILE_WIDTH"),
    ("profile0_error", "test22", "DiffFrameID"),
    ("profile0_error", "test223", "FrameWidth"),
    ("profile0_error", "test226", "CroppedTileHeight"),
    ("profile0_error", "test24", "RefValid[ ref_frame_idx"),
    ("profile0_error", "test244", "display_frame_id"),
    ("profile0_error", "test246", "idLen"),
    ("profile0_error", "test44", "film_grain_params_ref_idx"),
    ("profile0_error", "test95", "MaxHSize"),
    ("profile0_error", "test96", "MaxVSize"),
    ("profile1_error", "test101", "MaxTiles"),
    ("profile1_error", "test132", "idLen"),
    ("profile1_error", "test136", "DiffFrameID"),
    ("profile1_error", "test144", "expectedFrameId"),
    ("profile1_error", "test223", "FrameWidth"),
    ("profile1_error", "test224", "FrameHeight"),
    ("profile1_error", "test225", "CroppedTileWidth"),
    ("profile1_error", "test226", "CroppedTileHeight"),
    ("profile1_error", "test24", "RefValid[ ref_frame_idx"),
    ("profile1_error", "test244", "display_frame_id"),
    ("profile1_error", "test249", "PrevFrameID"),
    ("profile1_error", "test272", "film_grain_params_ref_idx"),
    ("profile1_error", "test330", "MaxTileCols"),
    ("profile1_error", "test332", "MAX_TILE_WIDTH"),
    ("profile1_error", "test334", "TileWidth"),
    (
        "profile1_error",
        "test359",
        "RefValid[ frame_to_show_map_idx ]",
    ),
    ("profile1_error", "test41", "4096 * 2304"),
    ("profile1_error", "test94", "MaxPicSize"),
    ("profile1_error", "test95", "MaxHSize"),
    ("profile1_error", "test96", "MaxVSize"),
    ("profile2_error", "test101", "MaxTiles"),
    ("profile2_error", "test106", "TileWidth"),
    ("profile2_error", "test110", "FrameHeight"),
    ("profile2_error", "test112", "CroppedTileHeight"),
    ("profile2_error", "test136", "DiffFrameID"),
    (
        "profile2_error",
        "test17",
        "RefValid[ frame_to_show_map_idx ]",
    ),
    ("profile2_error", "test208", "MaxPicSize"),
    ("profile2_error", "test209", "MaxHSize"),
    ("profile2_error", "test216", "MaxTileCols"),
    ("profile2_error", "test218", "MAX_TILE_WIDTH"),
    ("profile2_error", "test223", "FrameWidth"),
    ("profile2_error", "test225", "CroppedTileWidth"),
    ("profile2_error", "test24", "RefValid[ ref_frame_idx"),
    ("profile2_error", "test244", "display_frame_id"),
    ("profile2_error", "test246", "idLen"),
    ("profile2_error", "test249", "PrevFrameID"),
    ("profile2_error", "test258", "expectedFrameId"),
    ("profile2_error", "test41", "4096 * 2304"),
    ("profile2_error", "test44", "film_grain_params_ref_idx"),
    ("profile2_error", "test96", "MaxVSize"),
];

/// Decodes an error stream (Annex B or section 5, which the stream says):
/// the errors, in order.
fn errors_of(path: &Path, data: &[u8], strict: bool) -> Vec<String> {
    let mut dec = av1::Decoder::new();
    dec.set_strict(strict);
    if is_annexb(path) {
        return av1::annexb_temporal_units(data)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tu| dec.decode_annexb(tu).err().map(|e| e.to_string()))
            .collect();
    }
    split_at_delimiters(data)
        .into_iter()
        .filter_map(|tu| dec.decode_all(tu).err().map(|e| e.to_string()))
        .collect()
}

#[test]
fn argon_error_streams() {
    let Ok(root) = std::env::var("ARGON_DIR") else {
        eprintln!("ARGON_DIR not set; Argon error streams skipped");
        return;
    };
    let mut failed = Vec::new();
    for &(dir, stream, words) in ERROR_STREAMS {
        let path = Path::new(&root)
            .join(dir)
            .join("streams")
            .join(format!("{stream}.obu"));
        let Ok(data) = std::fs::read(&path) else {
            failed.push(format!("{dir}/{stream}: missing"));
            continue;
        };
        let strict = errors_of(&path, &data, true);
        if !strict.iter().any(|e| e.contains(words)) {
            failed.push(format!(
                "{dir}/{stream}: strict, no error naming {words:?}: {strict:?}"
            ));
        }
        // A reference to a slot no frame was ever put in is not decodable,
        // strict or not (the streams for RefValid do that); the other rules
        // decoding does not need.
        let needed = strict
            .iter()
            .any(|e| e.contains(words) && e.contains("empty"));
        let lax = errors_of(&path, &data, false);
        if !needed && lax.iter().any(|e| e.contains(words)) {
            failed.push(format!("{dir}/{stream}: not strict, yet {lax:?}"));
        }
    }
    for f in &failed {
        eprintln!("FAIL {f}");
    }
    eprintln!(
        "Argon error streams: {}/{} as expected",
        ERROR_STREAMS.len() - failed.len(),
        ERROR_STREAMS.len()
    );
    assert!(failed.is_empty());
}
