//! The AV1 decoder: OBUs in, frames out (7.1 to 7.21).
//!
//! [`Decoder::decode`] takes a temporal unit (one IVF frame, one Matroska
//! block: a sequence of OBUs in the low-overhead format) and returns the
//! frame it shows. Decoding follows the specification's processes in their
//! order: the frame header, the tiles (mode info, prediction, residual),
//! then the loop filter, CDEF, super-resolution, loop restoration, the
//! reference update and, on output, film grain synthesis.

mod flight;
pub(crate) mod grain;
pub(crate) mod lst;
pub(crate) mod mvpred;
pub(crate) mod pipeline;
pub(crate) mod postfilter;
pub(crate) mod predict;
pub(crate) mod record;
pub(crate) mod residual;
pub(crate) mod shared;
pub(crate) mod state;
pub(crate) mod tile;

use std::sync::Arc;

use crate::bits::BitReader;
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::frame::{ChromaFormat, ColorInfo, Frame, HdrMetadata};
use crate::header::{FrameHeader, RefHeaderState, RefState, get_relative_dist};
use crate::obu::{Metadata, SequenceHeader, split_obus};
use crate::tables::*;
use crate::{Error, Result};
use shared::Shared;
use state::{FrameBuf, Mi, Mv, Pool, RefData};
use tile::TileDecoder;

pub use lst::TileListLayout;

/// Loop restoration parameters of one plane (`LrType`, `LrWiener`,
/// `LrSgrSet`, `LrSgrXqd`).
#[derive(Clone, Default)]
pub(crate) struct LrPlane {
    pub(crate) unit_rows: usize,
    pub(crate) unit_cols: usize,
    pub(crate) lr_type: Vec<u8>,
    pub(crate) wiener: Vec<[[i32; 3]; 2]>,
    pub(crate) sgr_set: Vec<u8>,
    pub(crate) sgr_xqd: Vec<[i32; 2]>,
}

/// The state of the frame being decoded.
pub(crate) struct FrameCtx {
    pub(crate) seq: Arc<SequenceHeader>,
    pub(crate) hdr: FrameHeader,
    pub(crate) bit_depth: u32,
    pub(crate) ssx: usize,
    pub(crate) ssy: usize,
    pub(crate) num_planes: usize,
    pub(crate) mi_rows: usize,
    pub(crate) mi_cols: usize,
    /// Row stride of the per-4x4 arrays: `mi_cols` plus a margin of 32,
    /// which blocks overhanging the frame write into as the specification's
    /// unbounded arrays allow (and later reads may see).
    pub(crate) ms: usize,
    pub(crate) mi: Shared<Mi>,
    pub(crate) tx_types: Vec<u8>,
    pub(crate) lf_tx_sizes: [Shared<u8>; 3],
    pub(crate) palette_colors: [Vec<[u16; 8]>; 2],
    pub(crate) segment_ids: Vec<u8>,
    pub(crate) prev_segment_ids: Arc<Vec<u8>>,
    pub(crate) cdef_idx: Vec<i8>,
    pub(crate) cdef_stride: usize,
    pub(crate) lr: [LrPlane; 3],
    /// `MotionFieldMvs[ ref ]` on the 8x8 grid.
    pub(crate) motion_field: Arc<Vec<Vec<Mv>>>,
    pub(crate) cur: FrameBuf,
    pub(crate) refs: [Option<Arc<RefData>>; 8],
    /// The frame's CDFs (6.8.2).
    pub(crate) cdfs: Box<CdfContext>,
    /// The `Saved` CDFs from tile `context_update_tile_id`.
    pub(crate) saved_cdfs: Option<Box<CdfContext>>,
    /// `Mask` of the inter prediction process, 128 wide.
    pub(crate) mask: Box<[u8; 128 * 128]>,
    /// Tiles decoded so far (`TileNum`).
    pub(crate) tile_num: usize,
    /// Every tile's padding checked out (see `SymbolDecoder::trailing_ok`).
    pub(crate) tiles_ok: bool,
    /// `LrFrame`, when the post-filters ran with the tile's rows
    /// (`pipeline.rs`) and are not left to the decode frame wrapup.
    pub(crate) filtered: Option<FrameBuf>,
    /// Where the frame's sample planes come from, and go back to.
    pub(crate) pool: Arc<Pool>,
}

impl FrameCtx {
    /// A frame state for decoding one tile on its own: the frame's
    /// parameters and references shared, its per-4x4 arrays and sample
    /// buffers fresh (a tile reads nothing another tile of the same frame
    /// writes). [`FrameCtx::merge_tile`] copies the tile back.
    pub(crate) fn shard(&self) -> FrameCtx {
        let n = self.mi.len();
        FrameCtx {
            seq: self.seq.clone(),
            hdr: self.hdr.clone(),
            bit_depth: self.bit_depth,
            ssx: self.ssx,
            ssy: self.ssy,
            num_planes: self.num_planes,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            ms: self.ms,
            mi: vec![Mi::default(); n].into(),
            tx_types: vec![0; n],
            lf_tx_sizes: std::array::from_fn(|_| vec![0; n].into()),
            palette_colors: [
                vec![[0; 8]; self.palette_colors[0].len()],
                vec![[0; 8]; self.palette_colors[1].len()],
            ],
            segment_ids: vec![0; n],
            prev_segment_ids: self.prev_segment_ids.clone(),
            cdef_idx: self.cdef_idx.clone(),
            cdef_stride: self.cdef_stride,
            lr: self.lr.clone(),
            motion_field: self.motion_field.clone(),
            cur: FrameBuf {
                planes: self
                    .cur
                    .planes
                    .iter()
                    .map(|p| self.pool.plane(p.w, p.h, 0))
                    .collect(),
            },
            refs: self.refs.clone(),
            cdfs: self.cdfs.clone(),
            saved_cdfs: None,
            mask: Box::new([0; 128 * 128]),
            tile_num: self.tile_num,
            tiles_ok: true,
            filtered: None,
            pool: self.pool.clone(),
        }
    }

    /// Copies what decoding tile (`tile_row`, `tile_col`) into `shard`
    /// wrote back into this frame state: the tile's mode info, transform
    /// sizes and types, segment ids, CDEF indices, loop restoration units
    /// and samples (with the margins past the frame edge, for the last
    /// tile row and column).
    pub(crate) fn merge_tile(&mut self, shard: &FrameCtx, tile_row: usize, tile_col: usize) {
        let ti = &self.hdr.tile_info;
        let last_row = tile_row + 1 == ti.rows;
        let last_col = tile_col + 1 == ti.cols;
        let r0 = ti.mi_row_starts[tile_row];
        let c0 = ti.mi_col_starts[tile_col];
        let mr1 = ti.mi_row_starts[tile_row + 1];
        let mc1 = ti.mi_col_starts[tile_col + 1];
        self.merge_rect(shard, r0, mr1, c0, mc1, last_row, last_col);
    }

    /// Copies what decoding the superblock-aligned rectangle of rows
    /// `r0..mr1`, columns `c0..mc1` (in 4x4 units) wrote from `shard`;
    /// a rectangle on the frame's last row or column takes the padding
    /// beyond it too.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn merge_rect(
        &mut self,
        shard: &FrameCtx,
        r0: usize,
        mr1: usize,
        c0: usize,
        mc1: usize,
        last_row: bool,
        last_col: bool,
    ) {
        let rows = self.mi_rows + 32;
        let ms = self.ms;
        let r1 = if last_row { rows } else { mr1 };
        let c1 = if last_col { ms } else { mc1 };
        for row in r0..r1 {
            let a = row * ms + c0;
            let b = row * ms + c1;
            self.mi[a..b].copy_from_slice(&shard.mi[a..b]);
            self.tx_types[a..b].copy_from_slice(&shard.tx_types[a..b]);
            self.segment_ids[a..b].copy_from_slice(&shard.segment_ids[a..b]);
            for k in 0..2 {
                if !self.palette_colors[k].is_empty() {
                    self.palette_colors[k][a..b].copy_from_slice(&shard.palette_colors[k][a..b]);
                }
            }
        }
        for p in 0..self.num_planes {
            let (sx, sy) = self.plane_ss(p);
            let pr1 = if last_row { rows } else { mr1 >> sy };
            let pc1 = if last_col { ms } else { mc1 >> sx };
            for row in (r0 >> sy)..pr1 {
                let a = row * ms + (c0 >> sx);
                let b = row * ms + pc1;
                self.lf_tx_sizes[p][a..b].copy_from_slice(&shard.lf_tx_sizes[p][a..b]);
            }
            let dst = &mut self.cur.planes[p];
            let src = &shard.cur.planes[p];
            let x0 = (c0 * MI_SIZE) >> sx;
            let x1 = if last_col {
                dst.w
            } else {
                (mc1 * MI_SIZE) >> sx
            };
            let y0 = (r0 * MI_SIZE) >> sy;
            let y1 = if last_row {
                dst.h
            } else {
                (mr1 * MI_SIZE) >> sy
            };
            for y in y0..y1 {
                let o = y * dst.stride;
                dst.data[o + x0..o + x1].copy_from_slice(&src.data[o + x0..o + x1]);
            }
        }
        // CDEF indices and loop restoration units, per superblock.
        let sb4 = if self.seq.use_128x128_superblock {
            32
        } else {
            16
        };
        let mut r = r0;
        while r < mr1 {
            let mut c = c0;
            while c < mc1 {
                let s = self.cdef_stride;
                for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    if (dr == 1 || dc == 1) && sb4 == 16 {
                        continue;
                    }
                    let i = ((r >> 4) + dr) * s + (c >> 4) + dc;
                    if i < self.cdef_idx.len() {
                        self.cdef_idx[i] = shard.cdef_idx[i];
                    }
                }
                for plane in 0..self.num_planes {
                    if self.hdr.frame_restoration_type[plane] == RESTORE_NONE {
                        continue;
                    }
                    let (units_r, units_c) = lr_units_of_sb(self, plane, r, c, sb4);
                    let l = &mut self.lr[plane];
                    let sl = &shard.lr[plane];
                    for ur in units_r.clone() {
                        for uc in units_c.clone() {
                            let i = ur * l.unit_cols + uc;
                            l.lr_type[i] = sl.lr_type[i];
                            l.wiener[i] = sl.wiener[i];
                            l.sgr_set[i] = sl.sgr_set[i];
                            l.sgr_xqd[i] = sl.sgr_xqd[i];
                        }
                    }
                }
                c += sb4;
            }
            r += sb4;
        }
    }

    /// `get_plane_residual_size( subsize, plane )`.
    #[inline]
    pub(crate) fn plane_residual_size(&self, subsize: usize, plane: usize) -> usize {
        let (sx, sy) = self.plane_ss(plane);
        SUBSAMPLED_SIZE[subsize][sx][sy]
    }

    /// `is_scaled( refFrame )`.
    pub(crate) fn is_scaled(&self, ref_frame: i32) -> bool {
        if ref_frame <= INTRA_FRAME {
            return false;
        }
        let idx = self.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let Some(r) = &self.refs[idx] else {
            return false;
        };
        let fw = self.hdr.frame_width as i64;
        let fh = self.hdr.frame_height as i64;
        let x_scale = (((r.upscaled_width as i64) << REF_SCALE_SHIFT) + fw / 2) / fw;
        let y_scale = (((r.frame_height as i64) << REF_SCALE_SHIFT) + fh / 2) / fh;
        let no_scale = 1i64 << REF_SCALE_SHIFT;
        x_scale != no_scale || y_scale != no_scale
    }
}

/// Which of the two coefficient readers a [`Decoder`] uses (see
/// `residual.rs`): for tests, which hold them to each other.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoefficientReader {
    /// The decode-only reader. What decoding uses.
    #[default]
    Fast,
    /// The tile walker's own reader, the one the encoder also runs: the
    /// reference.
    Reference,
    /// Both, for every transform block, from the same state: a difference
    /// in the coefficients, the end-of-block position, the CDFs, the symbol
    /// decoder's state (its bit position included) or the contexts left for
    /// later blocks panics.
    Checked,
}

/// Decodes AV1 temporal units into frames.
pub struct Decoder {
    pub(crate) seq: Option<Arc<SequenceHeader>>,
    pub(crate) ref_state: RefState,
    pub(crate) refs: [Option<Arc<RefData>>; 8],
    frame: Option<FrameCtx>,
    seen_frame_header: bool,
    pub(crate) shown: Vec<Frame>,
    strict: bool,
    operating_point: usize,
    max_pixels: u64,
    /// The HDR metadata OBUs seen so far in this coded video sequence.
    hdr: HdrMetadata,
    /// Threads for rows, tiles and the post-filters (1: the caller's).
    threads: usize,
    /// Whether output frames get the film grain their stream asks for.
    apply_grain: bool,
    coeff_reader: CoefficientReader,
    /// The frames whose rows are decoded as tasks (`pipeline.rs`). (Before
    /// the threads: dropped first, it sends them home.)
    engine: pipeline::Engine,
    /// The threads, kept from one frame to the next.
    pool: crate::parallel::Pool,
    /// The headers of such a frame, between its frame header and its tile
    /// group (when `frame` is `None`).
    pending: Option<(Arc<SequenceHeader>, FrameHeader)>,
    /// How many frames may be decoded at once (`flight.rs`), the units
    /// queued, and the one being flown, if one is.
    frames_in_flight: usize,
    samples_in_flight: u64,
    units: flight::Units,
    flying: Option<flight::Flight>,
    /// How many units are decoded in place before one is flown again:
    /// counted down, and what the unit being flown asks for when it meets
    /// a frame that is not a job or finds no thread to fly on.
    grounded: usize,
    ground_for: usize,
    /// The anchor frames of large scale tile decoding (`lst.rs`).
    anchors: lst::Anchors,
    /// Sample planes no frame needs any more, for the frames to come.
    pub(crate) planes: Arc<Pool>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder awaiting a sequence header.
    pub fn new() -> Self {
        Decoder {
            seq: None,
            ref_state: RefState::default(),
            refs: Default::default(),
            frame: None,
            seen_frame_header: false,
            shown: Vec::new(),
            strict: false,
            operating_point: 0,
            max_pixels: 1 << 26,
            hdr: HdrMetadata::default(),
            threads: 1,
            apply_grain: true,
            coeff_reader: CoefficientReader::Fast,
            pool: Default::default(),
            engine: Default::default(),
            pending: None,
            frames_in_flight: 1,
            samples_in_flight: u64::MAX,
            units: Default::default(),
            flying: None,
            grounded: 0,
            ground_for: 0,
            anchors: Default::default(),
            planes: Default::default(),
        }
    }

    /// Chooses the coefficient reader; for tests.
    #[doc(hidden)]
    pub fn set_coefficient_reader(&mut self, reader: CoefficientReader) {
        self.coeff_reader = reader;
    }

    /// Frames (upscaled width times height) larger than this are refused
    /// as a bitstream error rather than allocated. The default, 2^26
    /// pixels, admits AV1's largest level (8K).
    pub fn set_max_pixels(&mut self, max: u64) {
        self.max_pixels = max;
    }

    /// Decodes with up to `n` threads: a frame of one tile has its
    /// superblock rows reconstructed behind the parsing of the tile, a
    /// frame of several its tiles in parallel, and the post-filters run by
    /// rows (see `docs/threading.md`). The frames are the same whatever
    /// `n`. 1, the default, decodes on the caller's thread alone; 0 means
    /// one per core.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = if n == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            n
        };
    }

    /// Whether the output process synthesises film grain (7.18.3) on the
    /// frames of a stream that carries grain parameters. On by default, as
    /// the specification has it; off, a frame comes out as it was decoded
    /// (the reference frames, which never carry grain, are the same either
    /// way), which costs less and is what a player without the budget for
    /// grain shows.
    pub fn set_apply_grain(&mut self, apply: bool) {
        self.apply_grain = apply;
    }

    /// Makes the decoder report a tile whose arithmetic-coded data does not
    /// end with the padding `exit_symbol()` requires as a bitstream error.
    /// A conformance check; off by default.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    /// Chooses the operating point (`choose_operating_point()`, 5.5.1):
    /// which layers of a scalable stream to decode. 0, the default, is the
    /// one the stream lists first (normally all layers).
    pub fn set_operating_point(&mut self, op: usize) {
        self.operating_point = op;
    }

    /// The colour description of the sequence header in force, once one
    /// has been seen.
    pub fn color_info(&self) -> Option<ColorInfo> {
        self.seq.as_deref().map(|s| color_info(&s.color))
    }

    /// The HDR metadata (content light level, mastering display) the
    /// stream has carried so far in the current coded video sequence;
    /// every shown [`Frame`] also carries it.
    pub fn hdr_metadata(&self) -> HdrMetadata {
        self.hdr
    }

    /// Decodes a temporal unit in the length-delimited format of Annex B
    /// (without its leading `temporal_unit_size`; see
    /// [`annexb_temporal_units`]) and returns every frame it shows.
    pub fn decode_annexb(&mut self, tu: &[u8]) -> Result<Vec<Frame>> {
        self.nothing_queued()?;
        self.shown.clear();
        let mut pos = 0;
        while pos < tu.len() {
            let (fu_size, n) = read_leb128(&tu[pos..])?;
            pos += n;
            let end = pos
                .checked_add(fu_size)
                .filter(|&e| e <= tu.len())
                .ok_or_else(|| Error::bitstream("frame_unit_size runs past the temporal unit"))?;
            while pos < end {
                let (obu_len, n) = read_leb128(&tu[pos..end])?;
                pos += n;
                let obu_end = pos
                    .checked_add(obu_len)
                    .filter(|&e| e <= end)
                    .ok_or_else(|| Error::bitstream("obu_length runs past the frame unit"))?;
                for obu in split_obus(&tu[pos..obu_end])? {
                    self.decode_obu(
                        obu.obu_type,
                        obu.temporal_id,
                        obu.spatial_id,
                        obu.has_extension,
                        obu.payload,
                    )?;
                }
                pos = obu_end;
            }
        }
        Ok(std::mem::take(&mut self.shown))
    }

    /// Decodes one temporal unit (or any sequence of whole OBUs) and
    /// returns the last frame it shows, if any.
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        Ok(self.decode_all(data)?.pop())
    }

    /// Decodes one temporal unit and returns every frame it shows, in
    /// order (more than one only with spatial layers).
    ///
    /// # Errors
    ///
    /// Besides the unit's own: `Error::InvalidInput` while units given to
    /// [`Decoder::queue`] are waiting for their results to be taken.
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<Frame>> {
        self.nothing_queued()?;
        self.decode_unit(data)
    }

    /// What decoding a unit in place asks first: that no queued unit is
    /// waiting, and that the threads are not out on a session for those.
    fn nothing_queued(&self) -> Result<()> {
        if self.queued() != 0 {
            return Err(Error::InvalidInput(
                "units are queued: take their results (poll, wait) before decoding in place".into(),
            ));
        }
        self.engine.end_session();
        Ok(())
    }

    /// Decodes one temporal unit before it returns: every frame it shows.
    fn decode_unit(&mut self, data: &[u8]) -> Result<Vec<Frame>> {
        self.shown.clear();
        self.decode_obus(data)?;
        Ok(std::mem::take(&mut self.shown))
    }

    /// The OBUs of a temporal unit, in order.
    fn decode_obus(&mut self, data: &[u8]) -> Result<()> {
        for obu in split_obus(data)? {
            self.decode_obu(
                obu.obu_type,
                obu.temporal_id,
                obu.spatial_id,
                obu.has_extension,
                obu.payload,
            )?;
        }
        Ok(())
    }

    fn decode_obu(
        &mut self,
        obu_type: u32,
        temporal_id: u32,
        spatial_id: u32,
        ext: bool,
        payload: &[u8],
    ) -> Result<()> {
        // Parallel work started below runs on this decoder's threads.
        let _entered = self.pool.enter();
        self.decode_obu_entered(obu_type, temporal_id, spatial_id, ext, payload)
    }

    fn decode_obu_entered(
        &mut self,
        obu_type: u32,
        temporal_id: u32,
        spatial_id: u32,
        ext: bool,
        payload: &[u8],
    ) -> Result<()> {
        if obu_type != OBU_SEQUENCE_HEADER
            && obu_type != OBU_TEMPORAL_DELIMITER
            && ext
            && let Some(seq) = &self.seq
        {
            let idc = seq.op_idc;
            if idc != 0 {
                let in_t = (idc >> temporal_id) & 1;
                let in_s = (idc >> (spatial_id + 8)) & 1;
                if in_t == 0 || in_s == 0 {
                    return Ok(());
                }
            }
        }
        match obu_type {
            OBU_SEQUENCE_HEADER => {
                let s = SequenceHeader::parse(payload, self.operating_point)?;
                if self.seq.as_deref() != Some(&s) {
                    // A new sequence: metadata of the old one no longer
                    // applies.
                    self.hdr = HdrMetadata::default();
                }
                self.seq = Some(Arc::new(s));
            }
            OBU_TEMPORAL_DELIMITER => {
                self.seen_frame_header = false;
            }
            OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                if self.seen_frame_header {
                    if obu_type == OBU_FRAME {
                        return Err(Error::bitstream("OBU_FRAME while a frame is in progress"));
                    }
                    // frame_header_copy(): identical to the header in force.
                    return Ok(());
                }
                let seq = self
                    .seq
                    .clone()
                    .ok_or_else(|| Error::bitstream("frame header before any sequence header"))?;
                let mut r = BitReader::new(payload);
                let hdr =
                    FrameHeader::parse(&mut r, &seq, &mut self.ref_state, temporal_id, spatial_id)?;
                if hdr.show_existing_frame {
                    self.show_existing(seq, hdr)?;
                    self.seen_frame_header = false;
                    return Ok(());
                }
                self.seen_frame_header = true;
                if self.threads > 1 && pipeline::applies(&hdr) {
                    // A frame of one tile on several threads: its state is
                    // set up by the task that parses it.
                    self.check_frame(&seq, &hdr)?;
                    self.frame = None;
                    self.pending = Some((seq, hdr));
                } else {
                    if self.flying.is_some() {
                        // Not a frame that flies: the unit is decoded in
                        // place (`flight.rs`), and this error goes no
                        // further.
                        self.ground_for = flight::AFTER_A_FRAME_THAT_IS_NOT_A_JOB;
                        return Err(Error::unsupported("a frame that is decoded in place"));
                    }
                    let f = self.setup_frame(seq, hdr)?;
                    self.pending = None;
                    self.frame = Some(f);
                }
                if obu_type == OBU_FRAME {
                    r.byte_align()?;
                    let pos = r.position() / 8;
                    self.tile_group(&payload[pos..])?;
                }
            }
            OBU_TILE_GROUP => {
                if !self.seen_frame_header || (self.frame.is_none() && self.pending.is_none()) {
                    return Err(Error::bitstream("tile group without a frame header"));
                }
                self.tile_group(payload)?;
            }
            OBU_METADATA => match crate::obu::parse_metadata(payload) {
                Some(Metadata::ContentLight(c)) => self.hdr.content_light = Some(c),
                Some(Metadata::MasteringDisplay(m)) => self.hdr.mastering_display = Some(m),
                None => {}
            },
            OBU_TILE_LIST => self.tile_list(payload)?,
            _ => {}
        }
        Ok(())
    }

    /// Sets up the frame state after `uncompressed_header()`: CDFs, the
    /// previous segment map, the motion field, the sample buffers.
    pub(crate) fn setup_frame(
        &mut self,
        seq: Arc<SequenceHeader>,
        hdr: FrameHeader,
    ) -> Result<FrameCtx> {
        self.check_frame(&seq, &hdr)?;
        Ok(build_frame(&self.refs, seq, hdr, &self.planes))
    }

    /// What can be wrong with a frame once its header is parsed: its size,
    /// and the references it names. Their header state alone says, so
    /// this does not wait for a reference that is still being decoded.
    fn check_frame(&self, seq: &SequenceHeader, hdr: &FrameHeader) -> Result<()> {
        let c = &seq.color;
        if hdr.upscaled_width as u64 * hdr.frame_height as u64 > self.max_pixels {
            return Err(Error::bitstream(
                "frame larger than the decoder's pixel limit",
            ));
        }
        if !hdr.frame_is_intra {
            for i in 0..REFS_PER_FRAME {
                let idx = hdr.ref_frame_idx[i];
                let r = self.refs[idx]
                    .as_ref()
                    .ok_or_else(|| Error::bitstream("inter frame references an empty slot"))?;
                if r.bit_depth != c.bit_depth
                    || r.subsampling_x != c.subsampling_x
                    || r.subsampling_y != c.subsampling_y
                {
                    return Err(Error::bitstream("reference frame format differs"));
                }
                let (fw, fh) = (hdr.frame_width as usize, hdr.frame_height as usize);
                if 2 * fw < r.upscaled_width
                    || 2 * fh < r.frame_height
                    || fw > 16 * r.upscaled_width
                    || fh > 16 * r.frame_height
                {
                    return Err(Error::bitstream("reference frame scale out of range"));
                }
            }
        }
        if hdr.primary_ref_frame != PRIMARY_REF_NONE
            && self.refs[hdr.ref_frame_idx[hdr.primary_ref_frame]].is_none()
        {
            return Err(Error::bitstream(
                "primary_ref_frame refers to an empty slot",
            ));
        }
        Ok(())
    }

    /// `tile_group_obu( sz )` (5.11.1) for a frame of one tile that is
    /// decoded as tasks: the tile is the whole of `data` (no tile start
    /// and end, no tile size), and with it the frame is complete.
    fn tile_group_tasks(
        &mut self,
        seq: Arc<SequenceHeader>,
        hdr: FrameHeader,
        data: &[u8],
    ) -> Result<()> {
        let out = Arc::new(ref_data(&seq, &hdr, Some(&self.planes)));
        let input = pipeline::Input {
            seq: seq.clone(),
            hdr: hdr.clone(),
            data: data.to_vec(),
            coeff_reader: self.coeff_reader,
            ref_order_hints: self.ref_state.order_hint,
            threads: self.threads,
            pool: self.planes.clone(),
        };
        let job = pipeline::Job::new(&self.refs, out.clone(), input);
        if self.flying.is_some() {
            // In flight: the frame is in its slots as of now (7.20), and
            // decoded when its turn comes, by the threads of a session.
            if !self.engine.session(self.threads) {
                // No thread to be had (not yet, perhaps): in place, then,
                // this unit and the next few (the error goes no further).
                self.ground_for = flight::WITHOUT_THREADS;
                return Err(Error::unsupported("no threads for frames in flight"));
            }
            self.engine.wait_for_room(
                self.frames_in_flight,
                self.samples_in_flight,
                hdr.upscaled_width as u64 * hdr.frame_height as u64,
            );
            let id = self.engine.submit(job);
            self.flying
                .as_mut()
                .expect("a unit in flight")
                .jobs
                .push(id);
            self.reference_update(&seq, &hdr, out);
            self.seen_frame_header = false;
            return Ok(());
        }
        match self.engine.decode(job, self.threads) {
            Some(padding_ok) if padding_ok || !self.strict => {
                self.reference_update(&seq, &hdr, out);
                self.seen_frame_header = false;
                Ok(())
            }
            // A tile that does not parse (or, for a strict decoder, does
            // not end as it should): decoded again as one thread decodes
            // it, for the same error and the same state after it.
            _ => {
                self.frame = Some(build_frame(&self.refs, seq, hdr, &self.planes));
                self.tile_group(data)
            }
        }
    }

    /// `tile_group_obu( sz )` (5.11.1).
    fn tile_group(&mut self, data: &[u8]) -> Result<()> {
        if let Some((seq, hdr)) = self.pending.take() {
            return self.tile_group_tasks(seq, hdr, data);
        }
        let f = self.frame.as_mut().expect("checked by the caller");
        let ti = f.hdr.tile_info.clone();
        let num_tiles = ti.cols * ti.rows;
        let mut r = BitReader::new(data);
        let mut tile_start_and_end_present = false;
        if num_tiles > 1 {
            tile_start_and_end_present = r.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present {
            (0, num_tiles - 1)
        } else {
            let bits = ti.cols_log2 + ti.rows_log2;
            (r.f(bits)? as usize, r.f(bits)? as usize)
        };
        if tg_end >= num_tiles || tg_start > tg_end {
            return Err(Error::bitstream("tile group range out of range"));
        }
        r.byte_align()?;
        let mut pos = r.position() / 8;
        let mut tile_data = Vec::with_capacity(tg_end + 1 - tg_start);
        for tile_num in tg_start..=tg_end {
            let last = tile_num == tg_end;
            let tile_size = if last {
                data.len() - pos
            } else {
                let n = ti.tile_size_bytes as usize;
                if pos + n > data.len() {
                    return Err(Error::bitstream("tile size runs past the tile group"));
                }
                let mut v = 0usize;
                for i in 0..n {
                    v |= (data[pos + i] as usize) << (8 * i);
                }
                pos += n;
                v + 1
            };
            if pos + tile_size > data.len() {
                return Err(Error::bitstream("tile data runs past the tile group"));
            }
            tile_data.push((tile_num, &data[pos..pos + tile_size]));
            pos += tile_size;
        }
        let threads = self.threads.min(tile_data.len());
        // Each tile: whether its padding checked out, and its CDFs when it is
        // the one whose CDFs the frame keeps.
        type TileResult = Result<(bool, Option<Box<CdfContext>>)>;
        let coeff_reader = self.coeff_reader;
        let decode_one = |f: &mut FrameCtx, tile_num: usize, d: &[u8]| -> TileResult {
            let mut td = TileDecoder::new(f, d, tile_num / ti.cols, tile_num % ti.cols);
            td.coeff_reader = coeff_reader;
            td.decode_tile()?;
            let ok = td.sd.trailing_ok();
            let saved = if !td.f.hdr.disable_frame_end_update_cdf
                && tile_num == ti.context_update_tile_id
            {
                Some(td.cdf)
            } else {
                None
            };
            Ok((ok, saved))
        };
        let results: Vec<(usize, TileResult)> = if threads > 1 {
            let nums: Vec<usize> = tile_data.iter().map(|t| t.0).collect();
            tiles_in_parallel(f, &nums, threads, |shard, i| {
                let (tile_num, d) = tile_data[i];
                (tile_num, decode_one(shard, tile_num, d))
            })
        } else {
            let mut out = Vec::new();
            for &(tile_num, d) in &tile_data {
                let r = decode_one(f, tile_num, d);
                let stop = r.is_err();
                out.push((tile_num, r));
                if stop {
                    break;
                }
            }
            out
        };
        for (tile_num, r) in results {
            let (ok, saved) = r?;
            if !ok {
                f.tiles_ok = false;
                if self.strict {
                    return Err(Error::bitstream(format!(
                        "tile {tile_num} does not end with its padding"
                    )));
                }
            }
            if saved.is_some() {
                f.saved_cdfs = saved;
            }
            f.tile_num = tile_num + 1;
        }
        if tg_end == num_tiles - 1 {
            let f = self.frame.take().expect("frame in progress");
            self.finish_frame(f)?;
            self.seen_frame_header = false;
        }
        Ok(())
    }

    /// After the last tile: `frame_end_update_cdf()` and the decode frame
    /// wrapup process.
    pub(crate) fn finish_frame(&mut self, mut f: FrameCtx) -> Result<()> {
        if !f.hdr.disable_frame_end_update_cdf
            && let Some(s) = f.saved_cdfs.take()
        {
            // frame_end_update_cdf(); the counters do not matter: every
            // load clears them.
            f.cdfs = s;
        }
        self.decode_frame_wrapup(f)
    }

    /// The decode frame wrapup process (7.4) for a decoded frame.
    fn decode_frame_wrapup(&mut self, mut f: FrameCtx) -> Result<()> {
        let lr = filter_frame(&mut f, self.threads);
        // `CurrFrame`, once the post-filters have made `LrFrame` of it.
        if f.filtered.is_none() {
            f.pool.give_back(std::mem::take(&mut f.cur));
        }
        let data = ref_data(&f.seq, &f.hdr, Some(&f.pool));
        let _ = data
            .syntax
            .set(frame_syntax(&mut f, &self.ref_state.order_hint));
        let _ = data.frame.set(lr);
        self.reference_update(&f.seq, &f.hdr, Arc::new(data));
        Ok(())
    }

    /// The reference frame update process (7.20) for a frame whose slot
    /// contents are `data`, and the output process if it is shown.
    fn reference_update(&mut self, seq: &Arc<SequenceHeader>, h: &FrameHeader, data: Arc<RefData>) {
        let hs = Arc::new(RefHeaderState {
            frame_id: h.current_frame_id,
            frame_type: h.frame_type,
            upscaled_width: h.upscaled_width,
            frame_height: h.frame_height,
            render_width: h.render_width,
            render_height: h.render_height,
            loop_filter_ref_deltas: h.loop_filter_ref_deltas,
            loop_filter_mode_deltas: h.loop_filter_mode_deltas,
            feature_enabled: h.feature_enabled,
            feature_data: h.feature_data,
            gm_params: h.gm_params,
            film_grain: h.film_grain.clone(),
        });
        for i in 0..NUM_REF_FRAMES {
            if (h.refresh_frame_flags >> i) & 1 != 0 {
                self.refs[i] = Some(data.clone());
                self.ref_state.slots[i] = Some(hs.clone());
                self.ref_state.valid[i] = true;
                self.ref_state.order_hint[i] = h.order_hint;
            }
        }
        self.ref_state.current_frame_id = h.current_frame_id;
        if h.show_frame {
            self.show(seq, &data, &h.film_grain);
        }
    }

    /// Shows a frame: the output process (7.18), now, or when the unit
    /// being flown has landed.
    fn show(
        &mut self,
        seq: &Arc<SequenceHeader>,
        data: &Arc<RefData>,
        grain: &crate::header::FilmGrainParams,
    ) {
        match &mut self.flying {
            Some(flight) => flight.shown.push(flight::Shown {
                seq: seq.clone(),
                data: data.clone(),
                grain: grain.clone(),
                hdr: self.hdr,
            }),
            None => {
                let out = self.output(seq, data, grain);
                self.shown.push(out);
            }
        }
    }

    /// `show_existing_frame` (7.4, 7.21).
    fn show_existing(&mut self, seq: Arc<SequenceHeader>, hdr: FrameHeader) -> Result<()> {
        let idx = hdr.frame_to_show_map_idx;
        let data = self.refs[idx]
            .clone()
            .ok_or_else(|| Error::bitstream("show_existing_frame of an empty slot"))?;
        let hs = self.ref_state.slots[idx]
            .clone()
            .ok_or_else(|| Error::bitstream("show_existing_frame of an empty slot"))?;
        if hdr.frame_type == KEY_FRAME {
            // The reference frame loading process, then the update with
            // refresh_frame_flags = allFrames.
            let order_hint = data.order_hint;
            for i in 0..NUM_REF_FRAMES {
                if (hdr.refresh_frame_flags >> i) & 1 != 0 {
                    self.refs[i] = Some(data.clone());
                    self.ref_state.slots[i] = Some(hs.clone());
                    self.ref_state.valid[i] = true;
                    self.ref_state.order_hint[i] = order_hint;
                }
            }
        }
        self.show(&seq, &data, &hdr.film_grain);
        Ok(())
    }

    /// The output process (7.18): the intermediate output and film grain.
    fn output(
        &self,
        seq: &SequenceHeader,
        data: &RefData,
        grain: &crate::header::FilmGrainParams,
    ) -> Frame {
        self.output_with(seq, data, grain, self.hdr)
    }

    /// [`Self::output`] with the HDR metadata `hdr` (the stream's when the
    /// frame was shown).
    fn output_with(
        &self,
        seq: &SequenceHeader,
        data: &RefData,
        grain: &crate::header::FilmGrainParams,
        hdr: HdrMetadata,
    ) -> Frame {
        let w = data.upscaled_width;
        let h = data.frame_height;
        let mono = seq.color.mono_chrome;
        let chroma = if mono {
            ChromaFormat::Mono
        } else {
            ChromaFormat::from_shifts(data.subsampling_x, data.subsampling_y)
        };
        let mut frame = Frame::new(w as u32, h as u32, data.bit_depth, chroma);
        frame.render_width = data.render_width;
        frame.render_height = data.render_height;
        frame.color = color_info(&seq.color);
        frame.hdr = hdr;
        let grain_on = self.apply_grain && seq.film_grain_params_present && grain.apply_grain;
        let mut planes: Vec<Vec<u16>> = Vec::new();
        if grain_on {
            for (p, pl) in data.frame().planes.iter().enumerate() {
                let (sx, sy) = if p == 0 {
                    (0, 0)
                } else {
                    (data.subsampling_x as usize, data.subsampling_y as usize)
                };
                let pw = (w + sx) >> sx;
                let ph = (h + sy) >> sy;
                let mut v = Vec::with_capacity(pw * ph);
                for y in 0..ph {
                    v.extend_from_slice(&pl.row(y)[..pw]);
                }
                planes.push(v);
            }
            grain::apply(
                seq,
                data.bit_depth,
                data.subsampling_x as usize,
                data.subsampling_y as usize,
                w,
                h,
                grain,
                &mut planes,
            );
        }
        let wide = data.bit_depth > 8;
        for p in 0..frame.planes.len() {
            let pl = frame.planes[p];
            let (pw, ph) = (pl.width as usize, pl.height as usize);
            let out = frame.plane_mut(p);
            for y in 0..ph {
                let src: &[u16] = if grain_on {
                    &planes[p][y * pw..(y + 1) * pw]
                } else {
                    &data.frame().planes[p].row(y)[..pw]
                };
                if wide {
                    let row = out[y * pw * 2..(y + 1) * pw * 2].as_chunks_mut::<2>().0;
                    for (o, &v) in row.iter_mut().zip(src) {
                        *o = v.to_le_bytes();
                    }
                } else {
                    for (o, &v) in out[y * pw..(y + 1) * pw].iter_mut().zip(src) {
                        *o = v as u8;
                    }
                }
            }
        }
        frame
    }
}

/// The motion field estimation process (7.9).
fn motion_field_estimation(f: &mut FrameCtx) {
    let w8 = f.mi_cols >> 1;
    let h8 = f.mi_rows >> 1;
    let invalid: Mv = [-1 << 15, -1 << 15];
    f.motion_field = Arc::new(vec![vec![invalid; w8 * h8]; 8]);
    let seq = f.seq.clone();
    let h = f.hdr.clone();
    let last_idx = h.ref_frame_idx[0];
    let cur_gold_order_hint = h.order_hints[GOLDEN_FRAME as usize];
    let last_alt_order_hint = f.refs[last_idx]
        .as_ref()
        .map_or(0, |r| r.saved_order_hints[ALTREF_FRAME as usize]);
    let use_last = last_alt_order_hint != cur_gold_order_hint;
    if use_last {
        project(f, LAST_FRAME, -1);
    }
    let mut ref_stamp = MFMV_STACK_SIZE - 2;
    let dist = |a: u32, b: u32| get_relative_dist(&seq, a, b);
    if dist(h.order_hints[BWDREF_FRAME as usize], h.order_hint) > 0 && project(f, BWDREF_FRAME, 1) {
        ref_stamp -= 1;
    }
    if dist(h.order_hints[ALTREF2_FRAME as usize], h.order_hint) > 0 && project(f, ALTREF2_FRAME, 1)
    {
        ref_stamp -= 1;
    }
    if dist(h.order_hints[ALTREF_FRAME as usize], h.order_hint) > 0
        && ref_stamp >= 0
        && project(f, ALTREF_FRAME, 1)
    {
        ref_stamp -= 1;
    }
    if ref_stamp >= 0 {
        project(f, LAST2_FRAME, -1);
    }
}

/// The projection process (7.9.2).
fn project(f: &mut FrameCtx, src: i32, dst_sign: i32) -> bool {
    let src_idx = f.hdr.ref_frame_idx[(src - LAST_FRAME) as usize];
    let w8 = (f.mi_cols >> 1) as i32;
    let h8 = (f.mi_rows >> 1) as i32;
    let Some(r) = f.refs[src_idx].clone() else {
        return false;
    };
    if r.mi_rows != f.mi_rows
        || r.mi_cols != f.mi_cols
        || r.frame_type == INTRA_ONLY_FRAME
        || r.frame_type == KEY_FRAME
    {
        return false;
    }
    let saved = r.syntax();
    let seq = f.seq.clone();
    let dist = |a: u32, b: u32| get_relative_dist(&seq, a, b);
    let order_hint = f.hdr.order_hint;
    let order_hints = f.hdr.order_hints;
    for y8 in 0..h8 {
        for x8 in 0..w8 {
            let i = (y8 * w8 + x8) as usize;
            let src_ref = saved.saved_ref_frames[i] as i32;
            if src_ref <= INTRA_FRAME {
                continue;
            }
            let ref_to_cur = dist(order_hints[src as usize], order_hint);
            let ref_offset = dist(
                order_hints[src as usize],
                r.saved_order_hints[src_ref as usize],
            );
            let pos_valid = ref_to_cur.abs() <= MAX_FRAME_DISTANCE
                && ref_offset.abs() <= MAX_FRAME_DISTANCE
                && ref_offset > 0;
            if !pos_valid {
                continue;
            }
            let mv = saved.saved_mvs[i];
            let proj_mv = get_mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
            let (pos_y8, vy) = project_pos(y8, proj_mv[0], dst_sign, h8, MAX_OFFSET_HEIGHT);
            let (pos_x8, vx) = project_pos(x8, proj_mv[1], dst_sign, w8, MAX_OFFSET_WIDTH);
            if vy && vx {
                for dst in LAST_FRAME..=ALTREF_FRAME {
                    let ref_to_dst = dist(order_hint, order_hints[dst as usize]);
                    let p = get_mv_projection(mv, ref_to_dst, ref_offset);
                    Arc::make_mut(&mut f.motion_field)[dst as usize]
                        [(pos_y8 * w8 + pos_x8) as usize] = p;
                }
            }
        }
    }
    true
}

/// The motion field motion vector storage process (7.19), kept on the
/// 8x8 grid (row and col odd). `ref_order_hints` is `RefOrderHint` as the
/// frame's header found it.
fn motion_vector_storage(f: &FrameCtx, ref_order_hints: &[u32; 8]) -> (Vec<i8>, Vec<Mv>) {
    let w8 = f.mi_cols >> 1;
    let h8 = f.mi_rows >> 1;
    let mut refs = vec![NONE as i8; w8 * h8];
    let mut mvs = vec![[0i32; 2]; w8 * h8];
    let seq = &f.seq;
    for y8 in 0..h8 {
        for x8 in 0..w8 {
            let row = 2 * y8 + 1;
            let col = 2 * x8 + 1;
            let m = &f.mi[row * f.ms + col];
            for list in 0..2 {
                let r = m.ref_frame[list] as i32;
                if r > INTRA_FRAME {
                    let ref_idx = f.hdr.ref_frame_idx[(r - LAST_FRAME) as usize];
                    let dist = get_relative_dist(seq, ref_order_hints[ref_idx], f.hdr.order_hint);
                    if dist < 0 {
                        let mv = m.mv[list];
                        if mv[0].abs() <= REFMVS_LIMIT && mv[1].abs() <= REFMVS_LIMIT {
                            refs[y8 * w8 + x8] = r as i8;
                            mvs[y8 * w8 + x8] = mv;
                        }
                    }
                }
            }
        }
    }
    (refs, mvs)
}

/// The post-filters of the decode frame wrapup (7.14 to 7.17) on a frame
/// whose tiles are decoded: `LrFrame`. (A frame whose rows were filtered
/// as they were reconstructed brings it along.)
pub(crate) fn filter_frame(f: &mut FrameCtx, threads: usize) -> FrameBuf {
    if let Some(lr) = f.filtered.take() {
        return lr;
    }
    if f.hdr.loop_filter_level[0] != 0 || f.hdr.loop_filter_level[1] != 0 {
        postfilter::loop_filter_threads(f, threads);
    }
    let cdef = postfilter::cdef(f, threads);
    let up_cdef = match postfilter::upscale(f, &cdef) {
        Some(u) => u,
        None => cdef,
    };
    let up_cur_owned = postfilter::upscale(f, &f.cur);
    let up_cur = up_cur_owned.as_ref().unwrap_or(&f.cur);
    postfilter::loop_restoration(f, up_cur, up_cdef, threads)
}

/// What a frame's header says of it as a reference (7.20): a slot's
/// contents before anything of the frame is decoded.
/// `pool` takes the frame's planes back once no slot holds it.
pub(crate) fn ref_data(seq: &SequenceHeader, h: &FrameHeader, pool: Option<&Arc<Pool>>) -> RefData {
    let c = &seq.color;
    RefData {
        frame_type: h.frame_type,
        upscaled_width: h.upscaled_width as usize,
        frame_height: h.frame_height as usize,
        render_width: h.render_width,
        render_height: h.render_height,
        mi_rows: h.mi_rows,
        mi_cols: h.mi_cols,
        bit_depth: c.bit_depth,
        subsampling_x: c.subsampling_x,
        subsampling_y: c.subsampling_y,
        order_hint: h.order_hint,
        saved_order_hints: h.order_hints,
        syntax: Default::default(),
        frame: Default::default(),
        pool: pool.cloned(),
    }
}

/// What a frame's parsing leaves for the frames that refer to it (7.19,
/// 7.20), once its tiles are parsed and `frame_end_update_cdf()` is done:
/// the saved motion vectors, segment map and CDFs. Takes the frame's
/// segment map and CDFs.
pub(crate) fn frame_syntax(f: &mut FrameCtx, ref_order_hints: &[u32; 8]) -> state::RefSyntax {
    let (saved_ref_frames, saved_mvs) = motion_vector_storage(f, ref_order_hints);
    if f.hdr.segmentation_enabled && !f.hdr.segmentation_update_map {
        let prev = f.prev_segment_ids.clone();
        f.segment_ids.copy_from_slice(&prev);
    }
    state::RefSyntax {
        saved_ref_frames,
        saved_mvs,
        saved_segment_ids: std::mem::take(&mut f.segment_ids),
        cdfs: (*f.cdfs).clone(),
    }
}

/// The frame state of a frame whose header checked out
/// (`Decoder::check_frame`): its CDFs and previous segment map from
/// `primary_ref_frame` (`load_cdfs`, `load_previous_segment_ids`), its
/// motion field (7.9), its arrays and sample buffers. `refs` are the
/// reference slots as they were when the header was parsed; the ones the
/// frame names must be parsed (`RefData::syntax`).
pub(crate) fn build_frame(
    refs: &[Option<Arc<RefData>>; 8],
    seq: Arc<SequenceHeader>,
    hdr: FrameHeader,
    pool: &Arc<Pool>,
) -> FrameCtx {
    let c = &seq.color;
    let mi_rows = hdr.mi_rows;
    let mi_cols = hdr.mi_cols;
    let cdfs = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
        CdfContext::new(hdr.base_q_idx)
    } else {
        let idx = hdr.ref_frame_idx[hdr.primary_ref_frame];
        let r = refs[idx].as_ref().expect("checked with the header");
        let mut c = Box::new(r.syntax().cdfs.clone());
        c.clear_counts();
        c
    };
    let ms = mi_cols + 32;
    let n = (mi_rows + 32) * ms;
    let mut prev_segment_ids = vec![0u8; n];
    if hdr.primary_ref_frame != PRIMARY_REF_NONE && hdr.segmentation_enabled {
        let idx = hdr.ref_frame_idx[hdr.primary_ref_frame];
        if let Some(r) = &refs[idx]
            && r.mi_rows == mi_rows
            && r.mi_cols == mi_cols
        {
            prev_segment_ids.copy_from_slice(&r.syntax().saved_segment_ids);
        }
    }
    let ssx = c.subsampling_x as usize;
    let ssy = c.subsampling_y as usize;
    let num_planes = c.num_planes;
    let aw = (mi_cols * MI_SIZE + 127) & !127;
    let ah = (mi_rows * MI_SIZE + 127) & !127;
    let mut planes = vec![pool.plane(aw + 32, ah + 32, 0)];
    for _ in 1..num_planes {
        planes.push(pool.plane((aw >> ssx) + 32, (ah >> ssy) + 32, 0));
    }
    let cdef_stride = (mi_cols >> 4) + 2;
    let screen_content = hdr.allow_screen_content_tools;
    let mut lr: [LrPlane; 3] = Default::default();
    for (plane, l) in lr.iter_mut().enumerate().take(num_planes) {
        if hdr.frame_restoration_type[plane] != RESTORE_NONE {
            let (sx, sy) = if plane == 0 {
                (0, 0)
            } else {
                (ssx as u32, ssy as u32)
            };
            let unit_size = hdr.loop_restoration_size[plane];
            l.unit_rows =
                count_units_in_frame(unit_size, round2(hdr.frame_height as i32, sy) as usize);
            l.unit_cols =
                count_units_in_frame(unit_size, round2(hdr.upscaled_width as i32, sx) as usize);
            let units = l.unit_rows * l.unit_cols;
            l.lr_type = vec![RESTORE_NONE; units];
            l.wiener = vec![[[0; 3]; 2]; units];
            l.sgr_set = vec![0; units];
            l.sgr_xqd = vec![[0; 2]; units];
        }
    }
    let mut f = FrameCtx {
        seq: seq.clone(),
        hdr,
        bit_depth: c.bit_depth,
        ssx,
        ssy,
        num_planes,
        mi_rows,
        mi_cols,
        ms,
        mi: vec![Mi::default(); n].into(),
        tx_types: vec![0; n],
        lf_tx_sizes: std::array::from_fn(|_| vec![0; n].into()),
        // Palettes need screen content tools.
        palette_colors: if screen_content {
            [vec![[0; 8]; n], vec![[0; 8]; n]]
        } else {
            [Vec::new(), Vec::new()]
        },
        segment_ids: vec![0; n],
        prev_segment_ids: Arc::new(prev_segment_ids),
        cdef_idx: vec![-1; cdef_stride * ((mi_rows >> 4) + 2)],
        cdef_stride,
        lr,
        motion_field: Arc::new(Vec::new()),
        cur: FrameBuf { planes },
        refs: refs.clone(),
        cdfs,
        saved_cdfs: None,
        mask: Box::new([0; 128 * 128]),
        tile_num: 0,
        tiles_ok: true,
        filtered: None,
        pool: pool.clone(),
    };
    if f.hdr.use_ref_frame_mvs {
        motion_field_estimation(&mut f);
    } else {
        f.motion_field = Arc::new(vec![Vec::new(); 8]);
    }
    f
}

/// The public colour description of a `color_config()`.
fn color_info(c: &crate::obu::ColorConfig) -> ColorInfo {
    ColorInfo {
        color_primaries: c.color_primaries,
        transfer_characteristics: c.transfer_characteristics,
        matrix_coefficients: c.matrix_coefficients,
        full_range: c.color_range,
        chroma_sample_position: c.chroma_sample_position,
    }
}

/// The loop restoration units superblock `(r, c)` codes (5.11.57):
/// unit rows and columns.
pub(crate) fn lr_units_of_sb(
    f: &FrameCtx,
    plane: usize,
    r: usize,
    c: usize,
    sb4: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let (sub_x, sub_y) = f.plane_ss(plane);
    let unit_size = f.hdr.loop_restoration_size[plane];
    let l = &f.lr[plane];
    let row_start = (r * (MI_SIZE >> sub_y)).div_ceil(unit_size);
    let row_end = l
        .unit_rows
        .min(((r + sb4) * (MI_SIZE >> sub_y)).div_ceil(unit_size));
    let (numerator, denominator) = if f.hdr.use_superres {
        (
            (MI_SIZE >> sub_x) * f.hdr.superres_denom as usize,
            unit_size * SUPERRES_NUM as usize,
        )
    } else {
        (MI_SIZE >> sub_x, unit_size)
    };
    let col_start = (c * numerator).div_ceil(denominator);
    let col_end = l
        .unit_cols
        .min(((c + sb4) * numerator).div_ceil(denominator));
    (row_start..row_end, col_start..col_end)
}

/// Codes (or decodes) the tiles `tiles` of frame `f` on up to `threads`
/// threads (the caller's among them, see [`crate::parallel`]): `job(state,
/// i)` handles `tiles[i]` in a frame state of the worker's own
/// (`FrameCtx::shard`), whose tile region is then copied into `f`. The
/// tiles of a frame are independent, so the result is the same as handling
/// them one after the other in `f`.
pub(crate) fn tiles_in_parallel<T: Send>(
    f: &mut FrameCtx,
    tiles: &[usize],
    threads: usize,
    job: impl Fn(&mut FrameCtx, usize) -> T + Sync,
) -> Vec<T> {
    let cols = f.hdr.tile_info.cols;
    let workers = threads.min(tiles.len()).max(1);
    // One frame state per worker, taken by the worker that runs.
    let shards: Vec<std::sync::Mutex<Option<FrameCtx>>> = (0..workers)
        .map(|_| std::sync::Mutex::new(Some(f.shard())))
        .collect();
    let main = std::sync::Mutex::new(f);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<T>>> =
        tiles.iter().map(|_| std::sync::Mutex::new(None)).collect();
    crate::parallel::run(workers, &|k| {
        let Some(mut shard) = shards[k].lock().expect("frame state").take() else {
            return;
        };
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= tiles.len() {
                break;
            }
            let r = job(&mut shard, i);
            let t = tiles[i];
            main.lock()
                .expect("frame state")
                .merge_tile(&shard, t / cols, t % cols);
            *out[i].lock().expect("result slot") = Some(r);
        }
        shard.pool.give_back(std::mem::take(&mut shard.cur));
    });
    out.into_iter()
        .map(|m| {
            m.into_inner()
                .expect("result slot")
                .expect("every tile ran")
        })
        .collect()
}

/// Runs `job(i)` for `i` in `0..n` on up to `threads` threads (the
/// caller's among them), returning the results in order.
pub(crate) fn parallel_map<T: Send>(
    n: usize,
    threads: usize,
    job: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<T>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    crate::parallel::run(threads.min(n).max(1), &|_| {
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= n {
                break;
            }
            let v = job(i);
            *out[i].lock().expect("result slot") = Some(v);
        }
    });
    out.into_iter()
        .map(|m| m.into_inner().expect("result slot").expect("every job ran"))
        .collect()
}

/// `leb128()` at the start of `d`: the value and its length in bytes.
fn read_leb128(d: &[u8]) -> Result<(usize, usize)> {
    let mut value: u64 = 0;
    for i in 0..8 {
        let b = *d
            .get(i)
            .ok_or_else(|| Error::bitstream("leb128 runs past the data"))? as u64;
        value |= (b & 0x7f) << (i * 7);
        if b & 0x80 == 0 {
            return Ok((value as usize, i + 1));
        }
    }
    Ok((value as usize, 8))
}

/// Splits a whole Annex B bitstream (5.2's length-delimited format) into
/// its temporal units, each without its `temporal_unit_size`.
pub fn annexb_temporal_units(data: &[u8]) -> Result<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (sz, n) = read_leb128(&data[pos..])?;
        pos += n;
        let end = pos
            .checked_add(sz)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| Error::bitstream("temporal_unit_size runs past the data"))?;
        out.push(&data[pos..end]);
        pos = end;
    }
    Ok(out)
}

/// The get MV projection process (7.9.3).
fn get_mv_projection(mv: Mv, numerator: i32, denominator: i32) -> Mv {
    let clipped_den = denominator.min(MAX_FRAME_DISTANCE);
    let clipped_num = clip3(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE, numerator);
    let mut out = [0i32; 2];
    for i in 0..2 {
        let scaled = round2signed(mv[i] * clipped_num * DIV_MULT[clipped_den as usize], 14);
        out[i] = clip3(-(1 << 14) + 1, (1 << 14) - 1, scaled);
    }
    out
}

/// `project()` of the get block position process (7.9.4): the position and
/// whether it is valid.
fn project_pos(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> (i32, bool) {
    let base8 = (v8 >> 3) << 3;
    let offset8 = if delta >= 0 {
        delta >> (3 + 1 + MI_SIZE_LOG2)
    } else {
        -((-delta) >> (3 + 1 + MI_SIZE_LOG2))
    };
    let v = v8 + dst_sign * offset8;
    let valid = !(v < 0 || v >= max8 || v < base8 - max_off8 || v >= base8 + 8 + max_off8);
    (v, valid)
}

/// `count_units_in_frame( unitSize, frameSize )`.
pub(crate) fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}
