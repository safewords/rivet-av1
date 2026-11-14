//! Large scale tile decoding (7.3): a tile list OBU names tiles of coded
//! "camera" frames, each predicted from one of an array of anchor frames
//! the application provides, and its tiles are decoded alone and written
//! side by side into an output frame.
//!
//! The frame header in force (the camera frame header, which comes
//! without tile groups) supplies everything but the tiles: the frame's
//! size and tiling, its quantisers, its CDFs. For each tile list entry the
//! anchor frame it names becomes the frame `ref_frame_idx[ 0 ]` refers to,
//! with the dimensions of the frame being decoded (7.3.1), and the tile
//! is decoded by the decode camera tile process (7.3.2): the tile decoding
//! of the general process, with no post-filters and no reference update.
//! The conformance requirements of 7.3.1 turn off whatever would reach
//! outside the tile (loop filter, CDEF, loop restoration, super-resolution,
//! motion field projection, CDF adaptation) and make every inter block
//! predict from LAST alone.

use std::sync::{Arc, OnceLock};

use super::state::{FrameBuf, Mi, PlaneBuf, RefData};
use super::tile::TileDecoder;
use super::{Decoder, FrameCtx};
use crate::bits::BitReader;
use crate::consts::*;
use crate::frame::{ChromaFormat, Frame};
use crate::{Error, Result};

/// The anchor frames of large scale tile decoding (`AnchorFrames`), as
/// sample planes, and each as a reference frame for the frame header in
/// force, made when a tile first names it.
#[derive(Default)]
pub(crate) struct Anchors {
    frames: Vec<AnchorFrame>,
    /// The layout of the last tile list decoded.
    layout: Option<TileListLayout>,
}

struct AnchorFrame {
    bit_depth: u32,
    chroma: ChromaFormat,
    planes: FrameBuf,
    /// The anchor as `FrameStore[ last ]` of the frame header in force, and
    /// that header's size it was made for.
    as_ref: Option<((usize, usize), Arc<RefData>)>,
}

impl Anchors {
    /// Unpacks `frames` (decoded frames, or the application's own).
    pub(crate) fn set(&mut self, frames: &[Frame]) {
        self.frames = frames
            .iter()
            .map(|f| {
                let wide = f.bit_depth > 8;
                let planes = (0..f.chroma.num_planes())
                    .map(|p| {
                        let pl = f.planes[p];
                        let (w, h) = (pl.width as usize, pl.height as usize);
                        let src = f.plane(p);
                        let data: Vec<u16> = if wide {
                            src.as_chunks::<2>()
                                .0
                                .iter()
                                .map(|&b| u16::from_le_bytes(b))
                                .collect()
                        } else {
                            src.iter().map(|&v| v as u16).collect()
                        };
                        PlaneBuf {
                            data: data.into(),
                            stride: w,
                            w,
                            h,
                        }
                    })
                    .collect();
                AnchorFrame {
                    bit_depth: f.bit_depth,
                    chroma: f.chroma,
                    planes: FrameBuf { planes },
                    as_ref: None,
                }
            })
            .collect();
    }

    /// Anchor `idx` as the reference frame `last` of frame `f` (the steps
    /// of 7.3.1 before the decode camera tile process): its samples, with
    /// `RefUpscaledWidth`, `RefFrameHeight`, `RefMiCols`, `RefMiRows`,
    /// `RefSubsampling` and `RefBitDepth` those of the frame itself. An
    /// anchor smaller than the frame is extended by repeating its last
    /// column and row, which is what a prediction reaching past its edge
    /// would read; a larger one is cut.
    fn reference(&mut self, idx: usize, f: &FrameCtx) -> Result<Arc<RefData>> {
        let a = self.frames.get_mut(idx).ok_or_else(|| {
            Error::bitstream(format!("anchor_frame_idx {idx}: no such anchor frame"))
        })?;
        let chroma = if f.num_planes == 1 {
            ChromaFormat::Mono
        } else {
            ChromaFormat::from_shifts(f.ssx as u32, f.ssy as u32)
        };
        if a.bit_depth != f.bit_depth || a.chroma != chroma {
            return Err(Error::bitstream(
                "anchor frame format differs from the frame's",
            ));
        }
        let w = f.hdr.upscaled_width as usize;
        let h = f.hdr.frame_height as usize;
        if let Some((size, r)) = &a.as_ref
            && *size == (w, h)
        {
            return Ok(r.clone());
        }
        let planes = a
            .planes
            .planes
            .iter()
            .enumerate()
            .map(|(p, src)| {
                let (sx, sy) = if p == 0 { (0, 0) } else { (f.ssx, f.ssy) };
                let (pw, ph) = ((w + sx) >> sx, (h + sy) >> sy);
                if (src.w, src.h) == (pw, ph) {
                    return src.clone();
                }
                let mut out = PlaneBuf::new(pw, ph, 0);
                for y in 0..ph {
                    let row = src.row(y.min(src.h - 1));
                    for x in 0..pw {
                        out.set(x, y, row[x.min(src.w - 1)]);
                    }
                }
                out
            })
            .collect();
        let last = f.hdr.ref_frame_idx[0];
        let slot = f.refs[last].as_deref();
        let r = Arc::new(RefData {
            frame_type: slot.map_or(INTER_FRAME, |s| s.frame_type),
            upscaled_width: w,
            frame_height: h,
            render_width: w as u32,
            render_height: h as u32,
            mi_rows: f.mi_rows,
            mi_cols: f.mi_cols,
            bit_depth: f.bit_depth,
            subsampling_x: f.ssx as u32,
            subsampling_y: f.ssy as u32,
            order_hint: slot.map_or(0, |s| s.order_hint),
            saved_order_hints: slot.map_or([0; 8], |s| s.saved_order_hints),
            syntax: OnceLock::new(),
            frame: OnceLock::from(FrameBuf { planes }),
            pool: None,
        });
        a.as_ref = Some(((w, h), r.clone()));
        Ok(r)
    }
}

/// Where the tiles of a tile list OBU's output frame are (7.3.1): tile
/// `i` of the list (in the order of its entries) covers `tile_width` by
/// `tile_height` luma samples at column `tile_width * (i % across)`, row
/// `tile_height * (i / across)`, `across` being the output frame's width
/// over `tile_width`. Samples of the output frame no tile covers are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileListLayout {
    /// `TileWidth`: luma samples.
    pub tile_width: u32,
    /// `TileHeight`: luma samples (one superblock).
    pub tile_height: u32,
    /// The tiles decoded: `tile_count_minus_1 + 1`.
    pub tiles: usize,
}

impl Decoder {
    /// Sets the anchor frames of large scale tile decoding (7.3,
    /// `AnchorFrames`): the frames a tile list OBU's entries are predicted
    /// from, by index. How an application comes by them is outside the
    /// specification; typically they are the frames a stream shows before
    /// its tile lists (as AOMedia's light field example takes them: the
    /// first so many frames shown). They must have the bit depth and
    /// chroma format of the frames they are used with.
    ///
    /// Each tile list OBU then decodes to one output frame (returned as
    /// a shown frame is): its tiles in raster order, each predicted from
    /// its anchor, with the frame header in force (the camera frame
    /// header, a frame header OBU that has no tile groups).
    pub fn set_anchor_frames(&mut self, frames: &[Frame]) {
        self.anchors.set(frames);
    }

    /// The layout of the output frame of the last tile list OBU decoded.
    pub fn tile_list_layout(&self) -> Option<TileListLayout> {
        self.anchors.layout
    }

    /// `tile_list_obu()` (5.12) and the large scale tile decoding process
    /// (7.3).
    pub(crate) fn tile_list(&mut self, payload: &[u8]) -> Result<()> {
        if let Some((seq, hdr)) = self.pending.take() {
            // A camera frame header that would have been decoded as tasks
            // had it brought a tile group.
            self.frame = Some(super::build_frame(&self.refs, seq, hdr, &self.planes));
        }
        let Some(mut f) = self.frame.take() else {
            return Err(Error::bitstream("tile list without a frame header"));
        };
        let r = self.tile_list_with(&mut f, payload);
        // The frame header stays in force for the tile lists after this
        // one.
        self.frame = Some(f);
        let out = r?;
        match &mut self.flying {
            // Not decoded as a job: the unit goes no further in flight.
            Some(_) => Err(Error::unsupported("a tile list in flight")),
            None => {
                self.shown.push(out);
                Ok(())
            }
        }
    }

    fn tile_list_with(&mut self, f: &mut FrameCtx, payload: &[u8]) -> Result<Frame> {
        let mut r = BitReader::new(payload);
        let out_w_tiles = r.f(8)? as usize + 1;
        let out_h_tiles = r.f(8)? as usize + 1;
        let tile_count = r.f(16)? as usize + 1;
        let ti = f.hdr.tile_info.clone();
        let tile_w = (ti.mi_col_starts[1] - ti.mi_col_starts[0]) * MI_SIZE;
        let tile_h = (ti.mi_row_starts[1] - ti.mi_row_starts[0]) * MI_SIZE;
        if tile_count > out_w_tiles * out_h_tiles {
            return Err(Error::bitstream("more tiles than the output frame holds"));
        }
        let (out_w, out_h) = (out_w_tiles * tile_w, out_h_tiles * tile_h);
        if out_w as u64 * out_h as u64 > self.max_pixels {
            return Err(Error::bitstream(
                "output frame larger than the decoder's pixel limit",
            ));
        }
        let chroma = if f.num_planes == 1 {
            ChromaFormat::Mono
        } else {
            ChromaFormat::from_shifts(f.ssx as u32, f.ssy as u32)
        };
        let mut out = Frame::new(out_w as u32, out_h as u32, f.bit_depth, chroma);
        out.color = super::color_info(&f.seq.color);
        out.hdr = self.hdr;
        let last = f.hdr.ref_frame_idx[0];
        let slot = f.refs[last].clone();
        let mut pos = 4;
        let mut result = Ok(());
        for tile in 0..tile_count {
            let Some(e) = payload.get(pos..pos + 5) else {
                result = Err(Error::bitstream("tile list entry runs past the OBU"));
                break;
            };
            let anchor = e[0] as usize;
            let (row, col) = (e[1] as usize, e[2] as usize);
            let size = ((e[3] as usize) << 8 | e[4] as usize) + 1;
            pos += 5;
            let Some(data) = payload.get(pos..pos + size) else {
                result = Err(Error::bitstream("coded tile data runs past the OBU"));
                break;
            };
            pos += size;
            if row >= ti.rows || col >= ti.cols {
                result = Err(Error::bitstream("anchor tile outside the frame's tiles"));
                break;
            }
            match self.anchors.reference(anchor, f) {
                Ok(a) => f.refs[last] = Some(a),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            if let Err(e) = camera_tile(f, data, row, col, self.coeff_reader, self.strict) {
                result = Err(e);
                break;
            }
            write_tile(
                f,
                &mut out,
                row,
                col,
                tile % out_w_tiles,
                tile / out_w_tiles,
            );
        }
        f.refs[last] = slot;
        self.anchors.layout = Some(TileListLayout {
            tile_width: tile_w as u32,
            tile_height: tile_h as u32,
            tiles: tile_count,
        });
        result.map(|()| out)
    }
}

/// The decode camera tile process (7.3.2): tile (`row`, `col`) of the
/// frame from `data`, into `f`'s samples.
fn camera_tile(
    f: &mut FrameCtx,
    data: &[u8],
    row: usize,
    col: usize,
    coeff_reader: super::CoefficientReader,
    strict: bool,
) -> Result<()> {
    // What an earlier tile list left at this tile's place: nothing of it
    // may be seen (the mode info a later block of a tile reads is gated
    // on its having been written by this tile).
    clear_tile(f, row, col);
    let mut td = TileDecoder::new(f, data, row, col);
    td.coeff_reader = coeff_reader;
    td.decode_tile()?;
    if strict && !td.sd.trailing_ok() {
        return Err(Error::bitstream("tile does not end with its padding"));
    }
    Ok(())
}

/// Resets the mode info of tile (`row`, `col`) (and, on the frame's last
/// tile row or column, the margin past the frame that its blocks may
/// write).
fn clear_tile(f: &mut FrameCtx, row: usize, col: usize) {
    let ti = &f.hdr.tile_info;
    let r0 = ti.mi_row_starts[row];
    let c0 = ti.mi_col_starts[col];
    let r1 = if row + 1 == ti.rows {
        f.mi_rows + 32
    } else {
        ti.mi_row_starts[row + 1]
    };
    let c1 = if col + 1 == ti.cols {
        f.ms
    } else {
        ti.mi_col_starts[col + 1]
    };
    for r in r0..r1 {
        f.mi[r * f.ms + c0..r * f.ms + c1].fill(Mi::default());
    }
}

/// Writes the samples of decoded tile (`row`, `col`) into the output
/// frame at tile position (`dx`, `dy`) (7.3.1: `OutY`, `OutU`, `OutV` at
/// `destX`, `destY`).
fn write_tile(f: &FrameCtx, out: &mut Frame, row: usize, col: usize, dx: usize, dy: usize) {
    let ti = &f.hdr.tile_info;
    let tile_w = (ti.mi_col_starts[1] - ti.mi_col_starts[0]) * MI_SIZE;
    let tile_h = (ti.mi_row_starts[1] - ti.mi_row_starts[0]) * MI_SIZE;
    let x0 = ti.mi_col_starts[col] * MI_SIZE;
    let y0 = ti.mi_row_starts[row] * MI_SIZE;
    let w = (ti.mi_col_starts[col + 1] - ti.mi_col_starts[col]) * MI_SIZE;
    let h = (ti.mi_row_starts[row + 1] - ti.mi_row_starts[row]) * MI_SIZE;
    let wide = out.bit_depth > 8;
    for p in 0..out.planes.len() {
        let (sx, sy) = f.plane_ss(p);
        let pl = out.planes[p];
        let (ow, oh) = (pl.width as usize, pl.height as usize);
        let (dest_x, dest_y) = ((tile_w * dx) >> sx, (tile_h * dy) >> sy);
        let (src_x, src_y) = (x0 >> sx, y0 >> sy);
        let pw = (w >> sx).min(tile_w >> sx).min(ow.saturating_sub(dest_x));
        let ph = (h >> sy).min(tile_h >> sy).min(oh.saturating_sub(dest_y));
        let src = &f.cur.planes[p];
        let dst = out.plane_mut(p);
        for y in 0..ph {
            let s = &src.data[(src_y + y) * src.stride + src_x..][..pw];
            let o = (dest_y + y) * ow + dest_x;
            if wide {
                for (d, &v) in dst[2 * o..2 * (o + pw)]
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(s)
                {
                    *d = v.to_le_bytes();
                }
            } else {
                for (d, &v) in dst[o..o + pw].iter_mut().zip(s) {
                    *d = v as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tile list OBU (with its size field) of one entry: anchor 0, tile
    /// (0, 0), one byte of tile data.
    const TILE_LIST: [u8; 12] = [8 << 3 | 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn a_tile_list_needs_a_frame_header() {
        let mut dec = Decoder::new();
        assert!(matches!(dec.decode(&TILE_LIST), Err(Error::Bitstream(_))));
        assert_eq!(dec.tile_list_layout(), None);
    }
}
