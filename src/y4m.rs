//! YUV4MPEG2 (Y4M) files: the raw video the encoder's examples and
//! measurements read, and the decoder's examples write.
//!
//! A file is a header line (`YUV4MPEG2` then space-separated tags: `W`
//! width, `H` height, `C` colour space, and others this module passes
//! over), then frames, each a `FRAME` line and the planes Y, U, V packed
//! one after another — the layout of [`Frame::data`]. The colour spaces
//! read are those of the formats AV1 codes: `420` (and its sitings
//! `420jpeg`, `420mpeg2`, `420paldv`), `422`, `444` and `mono`, at 8 bits,
//! or with a `p10` / `p12` suffix (`mono10`, `mono12` for monochrome) at 10
//! and 12. A header without a `C` tag is 4:2:0 at 8 bits.

use crate::frame::{ChromaFormat, Frame};
use crate::{Error, Result};

/// A Y4M file's geometry: what each of its frames is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Y4mInfo {
    /// Luma width in samples.
    pub width: u32,
    /// Luma height in samples.
    pub height: u32,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
}

impl Y4mInfo {
    /// The `C` tag for this format (`420jpeg` at 8 bits, as most tools
    /// write it, `420p10`, `444p12`, `mono10`, ...).
    pub fn colorspace(&self) -> String {
        let base = match self.chroma {
            ChromaFormat::Yuv420 => "420",
            ChromaFormat::Yuv422 => "422",
            ChromaFormat::Yuv444 => "444",
            ChromaFormat::Mono => "mono",
        };
        match (self.chroma, self.bit_depth) {
            (ChromaFormat::Yuv420, 8) => "420jpeg".into(),
            (_, 8) => base.into(),
            (ChromaFormat::Mono, bd) => format!("mono{bd}"),
            (_, bd) => format!("{base}p{bd}"),
        }
    }

    /// The file's header line, with its newline (30 frames a second,
    /// progressive, square samples).
    pub fn header(&self) -> String {
        format!(
            "YUV4MPEG2 W{} H{} F30:1 Ip A1:1 C{}\n",
            self.width,
            self.height,
            self.colorspace()
        )
    }
}

/// The colour space of a `C` tag (its text after the `C`): chroma sampling
/// and bit depth.
fn parse_colorspace(c: &str) -> Result<(ChromaFormat, u32)> {
    let bad = || Error::invalid(format!("Y4M colour space C{c} is not one AV1 codes"));
    let (chroma, rest) = if let Some(r) = c.strip_prefix("mono") {
        (ChromaFormat::Mono, r)
    } else if let Some(r) = c.strip_prefix("420") {
        (ChromaFormat::Yuv420, r)
    } else if let Some(r) = c.strip_prefix("422") {
        (ChromaFormat::Yuv422, r)
    } else if let Some(r) = c.strip_prefix("444") {
        (ChromaFormat::Yuv444, r)
    } else {
        return Err(bad());
    };
    let bd = match rest {
        "" | "jpeg" | "mpeg2" | "paldv" => 8,
        _ => {
            let digits = rest.strip_prefix('p').unwrap_or(rest);
            match digits.parse::<u32>() {
                Ok(bd @ (8 | 10 | 12)) => bd,
                _ => return Err(bad()),
            }
        }
    };
    Ok((chroma, bd))
}

/// Reads a Y4M file's header: its geometry, and where its first frame
/// starts.
pub fn parse_header(data: &[u8]) -> Result<(Y4mInfo, usize)> {
    if !data.starts_with(b"YUV4MPEG2") {
        return Err(Error::invalid("not a Y4M file"));
    }
    let nl = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| Error::invalid("Y4M header has no end"))?;
    let header =
        std::str::from_utf8(&data[..nl]).map_err(|_| Error::invalid("Y4M header is not text"))?;
    let mut info = Y4mInfo {
        width: 0,
        height: 0,
        bit_depth: 8,
        chroma: ChromaFormat::Yuv420,
    };
    let num = |t: &str| {
        t.parse::<u32>()
            .map_err(|_| Error::invalid(format!("Y4M tag {t}")))
    };
    for tok in header.split_whitespace().skip(1) {
        match tok.as_bytes()[0] {
            b'W' => info.width = num(&tok[1..])?,
            b'H' => info.height = num(&tok[1..])?,
            b'C' => (info.chroma, info.bit_depth) = parse_colorspace(&tok[1..])?,
            _ => {}
        }
    }
    if info.width == 0 || info.height == 0 {
        return Err(Error::invalid("Y4M header has no size"));
    }
    Ok((info, nl + 1))
}

/// Reads up to `max` frames of a Y4M file (all of them with `usize::MAX`).
pub fn read(data: &[u8], max: usize) -> Result<(Y4mInfo, Vec<Frame>)> {
    let (info, mut pos) = parse_header(data)?;
    let size = Frame::new(info.width, info.height, info.bit_depth, info.chroma)
        .data
        .len();
    let mut out = Vec::new();
    while pos < data.len() && out.len() < max {
        if !data[pos..].starts_with(b"FRAME") {
            return Err(Error::invalid("Y4M frame does not start with FRAME"));
        }
        let fl = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| Error::invalid("Y4M frame header has no end"))?;
        pos += fl + 1;
        if pos + size > data.len() {
            return Err(Error::invalid("Y4M file ends inside a frame"));
        }
        let mut f = Frame::new(info.width, info.height, info.bit_depth, info.chroma);
        f.data.copy_from_slice(&data[pos..pos + size]);
        pos += size;
        out.push(f);
    }
    Ok((info, out))
}

/// Appends `frame` to a Y4M file being written (its `FRAME` line and
/// samples); the header comes from [`Y4mInfo::header`].
pub fn write_frame(out: &mut Vec<u8>, frame: &Frame) {
    out.extend_from_slice(b"FRAME\n");
    out.extend_from_slice(&frame.data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_spaces() {
        for (c, chroma, bd) in [
            ("420jpeg", ChromaFormat::Yuv420, 8),
            ("420mpeg2", ChromaFormat::Yuv420, 8),
            ("420", ChromaFormat::Yuv420, 8),
            ("420p10", ChromaFormat::Yuv420, 10),
            ("420p12", ChromaFormat::Yuv420, 12),
            ("422", ChromaFormat::Yuv422, 8),
            ("422p12", ChromaFormat::Yuv422, 12),
            ("444", ChromaFormat::Yuv444, 8),
            ("444p10", ChromaFormat::Yuv444, 10),
            ("mono", ChromaFormat::Mono, 8),
            ("mono12", ChromaFormat::Mono, 12),
        ] {
            assert_eq!(parse_colorspace(c).unwrap(), (chroma, bd), "C{c}");
        }
        assert!(parse_colorspace("411").is_err());
        assert!(parse_colorspace("444p16").is_err());
    }

    #[test]
    fn round_trip() {
        for chroma in [
            ChromaFormat::Mono,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for bd in [8, 10, 12] {
                let info = Y4mInfo {
                    width: 7,
                    height: 5,
                    bit_depth: bd,
                    chroma,
                };
                let mut f = Frame::new(7, 5, bd, chroma);
                for (i, b) in f.data.iter_mut().enumerate() {
                    *b = (i * 7) as u8 & if bd > 8 && i % 2 == 1 { 0x0f } else { 0xff };
                }
                let mut file = info.header().into_bytes();
                write_frame(&mut file, &f);
                write_frame(&mut file, &f);
                let (got, frames) = read(&file, usize::MAX).unwrap();
                assert_eq!(got, info);
                assert_eq!(frames, vec![f.clone(), f]);
            }
        }
    }
}
