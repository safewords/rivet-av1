//! The AV1 encoder: key frames and inter frames.
//!
//! Each frame is coded by the decoder's own tile walker running in encode
//! mode (see `encoder::tile`): the encoder plants its decisions — the
//! partition, modes, motion vector and quantised levels of each block —
//! and the walker codes them with the normative contexts and CDF
//! adaptation while reconstructing exactly as the decoder will. The frame
//! header is written, then parsed back with the decoder's parser, so the
//! two cannot disagree; the in-loop filters and the reference update are
//! the decoder's.
//!
//! Decisions are searched by trial coding (`encoder::rdo`); the in-loop
//! filters' parameters (loop filter levels, CDEF strengths) are searched on
//! a first pass's reconstruction, and the frame is then coded again
//! replaying the first pass's decisions with them.

mod aq;
pub(crate) mod cdef;
pub(crate) mod fwd;
mod gm;
mod gop;
mod grain;
mod inter;
pub(crate) mod lr;
mod rc;
pub(crate) mod rdo;
pub(crate) mod rdoq;
pub(crate) mod tile;
mod wavefront;

use std::sync::Arc;

use crate::bits::{BitReader, BitWriter};
use crate::consts::*;
use crate::decoder::Decoder;
use crate::decoder::tile::TileDecoder;
use crate::frame::{ChromaFormat, ColorInfo, Frame, HdrMetadata};
use crate::header::{FilmGrainParams, FrameHeader};
use crate::obu::{ColorConfig, SequenceHeader, write_obu};
use crate::tables::AC_QLOOKUP;
use crate::{Error, Result};

/// The default [`Config::speed`].
pub const DEFAULT_SPEED: u32 = 4;

/// The default [`Config::altref`]: groups of 16 frames (−1 % BD-rate
/// against groups of 8, and against 32, on the test clips).
pub const DEFAULT_ALTREF: u32 = 16;

/// Encoder settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bits per sample: 8, 10 or 12 (12 is profile 2's).
    pub bit_depth: u32,
    /// The quantiser index, 1 (finest) to 255 (coarsest); the starting
    /// point under rate control. 0 (without rate control) is lossless:
    /// every frame at quantiser index 0, its residuals coded with the
    /// Walsh-Hadamard transform, no in-loop filters, the output the input
    /// sample for sample.
    pub quantizer: u32,
    /// A key frame every this many frames (1: all key frames).
    pub keyframe_interval: u32,
    /// Target bits per frame for rate control; `None` for a fixed
    /// quantiser.
    pub target_bits_per_frame: Option<u64>,
    /// Loop filter level (0 to 63) at `quantizer`; scaled with it.
    pub loop_filter: Option<u32>,
    /// Half-width of the full-pel motion search, in pixels.
    pub search_range: i32,
    /// The colour description written into the sequence header's
    /// `color_config()`: primaries, transfer characteristics and matrix
    /// (ITU-T H.273 code points), range and chroma sample position. All
    /// three code points 2 (unspecified, the default) writes
    /// `color_description_present_flag` = 0.
    pub color: ColorInfo,
    /// HDR metadata, written as metadata OBUs (`METADATA_TYPE_HDR_CLL`,
    /// `METADATA_TYPE_HDR_MDCV`) after the sequence header of every key
    /// frame's temporal unit.
    pub hdr: HdrMetadata,
    /// Encoder effort, 0 (slowest, best) to 10 (fastest): how much of the
    /// rate-distortion search runs (see [`Tools`] for what it switches).
    pub speed: u32,
    /// Tile columns, as a log2 (0: one tile column; clamped to what the
    /// frame allows). Tiles are coded in parallel.
    pub tile_cols_log2: u32,
    /// Which coding tools the encoder uses; `Tools::for_speed(speed)` by
    /// default.
    pub tools: Tools,
    /// Threads coding tiles in parallel (with several tile columns); 1
    /// codes on the caller's thread.
    pub threads: usize,
    /// Alt-ref groups: frames are coded in groups of this many (3 or more,
    /// up to 32; 0 or less than 3: every frame in display order, as it
    /// comes), in a pyramid: the last frame of each group first, not shown
    /// and coded finest, then the frame half way to it, not shown either,
    /// and so on down, each frame predicted from the nearest coded frames
    /// on both sides and coded coarser the fewer frames refer to it; each
    /// hidden frame is shown with a `show_existing_frame` in its turn (see
    /// `encoder::gop`). [`DEFAULT_ALTREF`] by default; used where the
    /// rate-distortion search runs (`Tools::rdo`: speeds 0 to 8).
    ///
    /// The encoder holds a group's frames until the last of them comes:
    /// [`Encoder::encode`] returns an empty vector for the first frames
    /// (the delay is up to this many frames), then a temporal unit a call
    /// in order, and [`Encoder::flush`] the rest. A stream may end anywhere
    /// in a group.
    pub altref: u32,
    /// Luma only (`mono_chrome` = 1): the encoder takes
    /// [`ChromaFormat::Mono`] frames and codes one plane. An AVIF alpha
    /// plane is coded this way (AV1 Image File Format 4: an alpha item's
    /// sequence header has `mono_chrome` = 1). The same as `chroma` =
    /// [`ChromaFormat::Mono`], which it overrides.
    pub monochrome: bool,
    /// The chroma sampling of the frames taken: 4:2:0 (the default) and
    /// monochrome are profile 0's at 8 and 10 bits, 4:4:4 profile 1's, 4:2:2
    /// profile 2's; at 12 bits every format is profile 2's. The sequence
    /// header's `seq_profile` follows from this and `bit_depth`.
    pub chroma: ChromaFormat,
    /// Film grain: each frame is denoised before it is coded, and the
    /// grain taken out is estimated (its strength by intensity and its
    /// spatial correlation, per plane) and signalled in the frame header
    /// (`film_grain_params`) for the decoder to synthesise. Off by
    /// default. The reconstruction (and what a decoder outputs) has the
    /// synthesised grain; it is not the source's grain, so PSNR against
    /// the source falls while the bitrate does.
    pub film_grain: bool,
}

/// The encoder's coding tools, each switchable (for measurement, or to
/// trade quality for speed). [`Tools::for_speed`] gives each speed's set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tools {
    /// Rate-distortion search: partitions and block modes chosen by trial
    /// coding, the rate from the CDFs in force. Off: a variance test and
    /// SATD (the fastest).
    pub rdo: bool,
    /// Horizontal and vertical partitions (searched with `rdo`).
    pub partition_rect: bool,
    /// 8x8 blocks split to 4x4.
    pub partition_4x4: bool,
    /// Leave a block that codes flat (no residual) unsplit.
    pub prune_split: bool,
    /// Try the split first, and the block whole (or halved) only when
    /// the split's sub-blocks stayed whole.
    pub prune_partition: bool,
    /// Try skipping the residual of inter blocks.
    pub rd_skip: bool,
    /// `TX_MODE_SELECT`: transform sizes below the block size, searched.
    pub tx_size: bool,
    /// Luma transform types searched by trial coding (per transform block).
    pub tx_type_rd: bool,
    /// How many luma transform types the search tries (in a fixed order of
    /// usefulness: DCT, ADST, the mixes, identity, the 1D ones, the
    /// flipped ones).
    pub tx_types: u8,
    /// The full transform sets (`reduced_tx_set` = 0).
    pub full_tx_set: bool,
    /// Intra luma modes trialled per block (the best by SATD).
    pub intra_candidates: u8,
    /// Inter modes / vectors trialled per block.
    pub inter_candidates: u8,
    /// CDEF: strengths searched per frame, an index per 64x64 block.
    pub cdef: bool,
    /// The CDEF search tries more strengths.
    pub cdef_thorough: bool,
    /// Loop filter levels searched per frame (else derived from the
    /// quantiser).
    pub lf_search: bool,
    /// Several references: the last two frames and a golden frame (the
    /// key frame, then every 16th frame, coded finer).
    pub multi_ref: bool,
    /// Chroma from luma: a candidate for intra blocks up to 32x32.
    pub cfl: bool,
    /// Loop restoration: a Wiener or self-guided filter per unit, fitted
    /// to the source on the first pass's reconstruction.
    pub restoration: bool,
    /// The restoration search tries every self-guided parameter set.
    pub restoration_thorough: bool,
    /// Compound prediction: the average of LAST and a second reference
    /// (needs `multi_ref`).
    pub compound: bool,
    /// Palettes (screen content tools) on frames that look like screen
    /// content: few colours, sharp edges.
    pub palette: bool,
    /// Filter intra (`enable_filter_intra`): the five recursive filter
    /// predictors of `DC_PRED` luma blocks of 32x32 or less ranked with the
    /// other intra modes by SATD, the best trialled with them. Off at every
    /// speed: on the five test clips −0.13 %, −0.20 % and −0.24 % BD-rate
    /// PSNR-Y at speeds 4, 6 and 8 for about 5 % to 15 % more time (the
    /// five filtered predictions of every small block); in key frames only,
    /// ±0.
    pub filter_intra: bool,
    /// Intra block copy (`allow_intrabc`) in key frames that look like
    /// screen content (as palettes are offered): blocks of 8x8 or more
    /// copied from earlier in the frame, where their source appears again
    /// (found by hashing the source's 8x8 blocks) or where the stack's
    /// vectors point. Such a frame codes no in-loop filters. On where the
    /// search runs (speeds 0 to 8): on a synthetic screen-content clip
    /// −4.0 % to −6.7 % BD-rate at speeds 4 to 8 (−3 % to −12 % all key
    /// frames) for no more time; natural content is not screen content,
    /// and its streams are unchanged.
    pub intrabc: bool,
    /// Motion search by descending square steps from the best of the
    /// predicted, neighbouring and zero vectors (else an exhaustive window
    /// around the predicted one), and a cross-shaped sub-sample refinement.
    pub fast_me: bool,
    /// The transform depths tried with the DCT alone, the transform types
    /// then searched at the best depth only.
    pub fast_tx_refine: bool,
    /// The restoration search fits its filters on a quarter of the
    /// samples and tries fewer self-guided parameter sets.
    pub restoration_fast: bool,
    /// A partition candidate is given up as soon as the blocks it has
    /// coded cost more than the best candidate before it (a lossless
    /// shortcut: such a candidate cannot win).
    pub abandon_trials: bool,
    /// A block of 32x32 or more is not split when coding it whole costs
    /// less than this many hundredths of the Lagrange multiplier per
    /// sample (0: off).
    pub split_threshold: u16,
    /// The same for the smaller blocks.
    pub split_threshold_small: u16,
    /// The residual of inter blocks is tried skipped only for the best
    /// inter candidate (with `rd_skip`).
    pub rd_skip_best: bool,
    /// The deepest transform split searched (0 to 2).
    pub tx_depth_max: u8,
    /// Intra candidates are trialled in inter frames only when their
    /// SATD cost beats the best inter candidate's (else within 1.5x).
    pub prune_intra: bool,
    /// The decisions searched in superblock rows in a wavefront (each row
    /// adapting its own CDFs from the row above's), on `Config::threads`
    /// threads, then the frame coded again with them. The output does not
    /// depend on the thread count.
    pub wavefront: bool,
    /// Adaptive quantisation: superblocks the previous frame predicts well
    /// coded finer, those it does not coarser (`delta_q_present`).
    pub aq: bool,
    /// Rate-distortion optimised quantisation (`encoder::rdoq`): each
    /// coded transform block's levels and end of block chosen by their
    /// coding cost under the CDFs in force, not by the dead-zone rule.
    /// 0: never; 1: only in the blocks coded for real (the search measures
    /// dead-zone levels, with a wider dead zone); 2: in the search's trials
    /// too, but those of the transform type search; 3: everywhere. At
    /// speed 6, 1 is −3.1 % BD-rate PSNR-Y (with the chroma weights of
    /// `rdo::CHROMA_WEIGHT`) for 1.04x the time, 3 −3.7 % for 1.3x; at
    /// speed 4, 3 is −2.7 % for 1.2x and 1 only −0.8 %.
    pub rdoq: u8,
    /// The transform type search stops at the first type (after the DCT)
    /// that does not beat the best before it, and after the DCT when it
    /// codes no coefficients (at speed 6: −0.1 % BD-rate, 0.98x the time;
    /// the types are tried in their order of usefulness, so a type that
    /// loses seldom has a better one behind it).
    pub tx_type_early: bool,
    /// The blocks of a horizontal or vertical partition candidate search
    /// only the references the block whole chose (an intra block whole:
    /// only the nearest). At speed 6 with `partition_rect`: 0.89x the time
    /// of searching every reference in the halves, +0.9 % BD-rate.
    pub rect_refs: bool,
    /// The square blocks tried halved (with `partition_rect`): those
    /// `rect_min` to `rect_max` samples wide. Halving 8x8 blocks changed
    /// no stream of the test clips at speeds 4 and 6 (and cost 1.05x the
    /// time); at speed 4 the 32x32 halves are worth −1.2 % BD-rate for
    /// 1.12x the time, the 64x64 ones −0.6 % for 1.15x.
    pub rect_min: u8,
    /// See `rect_min`.
    pub rect_max: u8,
    /// Switchable interpolation filters: each inter block's filter (regular,
    /// smooth or sharp) chosen by its prediction (`is_filter_switchable`).
    /// The most of the inter tools: with OBMC and local warps, −2.0 % to
    /// −2.2 % BD-rate at speeds 4 to 8 for 1.04x to 1.09x the time (the
    /// README has each tool's share).
    pub switchable_filter: bool,
    /// With `switchable_filter`, a filter for each direction
    /// (`enable_dual_filter`). ±0 at speeds 4 and 6 (−0.3 % at 8) for 5 %
    /// more time: off.
    pub dual_filter: bool,
    /// Overlapped block motion compensation (`motion_mode` OBMC).
    pub obmc: bool,
    /// Local warped motion (`motion_mode` LOCALWARP): the block's warp
    /// fitted to its neighbours' vectors.
    pub warped_motion: bool,
    /// Global motion: a warp (or translation) per reference, estimated by
    /// the encoder from the source and the reference, for GLOBALMV blocks.
    /// ±0 at speed 4, −0.1 % at 6, −0.2 % at 8, for 4 % to 6 % more time:
    /// off.
    pub global_motion: bool,
    /// Wedge and difference-weighted compound (`enable_masked_compound`).
    /// ±0 at speed 4, +0.1 % at 6, −0.2 % at 8, for 5 % to 9 % more time:
    /// off.
    pub masked_compound: bool,
    /// Distance-weighted compound (`enable_jnt_comp`). −0.1 % at speeds 4
    /// and 8 (+0.1 % at 6) for about 1 % more time: on to speed 3.
    pub jnt_comp: bool,
    /// Inter-intra prediction, smooth and wedge
    /// (`enable_interintra_compound`). −0.2 % at speeds 4 and 8, −0.1 % at
    /// 6, for 3 % to 5 % more time: on to speed 3.
    pub interintra: bool,
    /// The inter tools are tried only for blocks whose best inter
    /// candidate costs at least this many hundredths of the Lagrange
    /// multiplier per sample (0: every block).
    pub inter_tool_threshold: u16,
    /// Every chroma intra mode (the thirteen but chroma from luma) ranked
    /// for the block by its SATD, each plane's weighted by the square root
    /// of its `rdo::CHROMA_WEIGHT`, plus its rate under the CDFs in force;
    /// else DC, V, H, smooth and Paeth with a fixed rate. Off: with the
    /// inter tools in, +0.2 % BD-rate PSNR-Y and +0.1 % PSNR-YUV at speed 6
    /// against the five modes (it was −0.5 % and −0.4 % before them).
    pub uv_modes: bool,
    /// How many more chroma modes (the next best by the ranking of
    /// `uv_modes`) are trialled on a block the search chose intra, with
    /// its luma decisions as they are: the preselection's SATD misjudges
    /// what a chroma block's residual costs.
    pub uv_trials: u8,
    /// `uv_trials` in inter frames too, not only in intra frames (at speed
    /// 6, −0.53 % / −0.82 % BD-rate PSNR-Y / PSNR-YUV with `rdoq_uv` 2,
    /// against −0.42 % / −0.66 % in intra frames only, for 1.08x the time
    /// against about none).
    pub uv_trials_inter: bool,
    /// With `rdoq` 1, the chroma transform blocks of the search's trials
    /// are quantised by rate and distortion too (not only the blocks
    /// coded): their levels then follow the chroma weights, as the coded
    /// ones do. 0: never; 1: in intra frames; 2: in every frame.
    pub rdoq_uv: u8,
    /// How many of the inter tools' variants of a block's best inter
    /// candidate (those whose prediction measures cheaper) are trialled.
    pub inter_tool_trials: u8,
    /// Testing: an inter tool taken wherever it can be coded, whatever it
    /// costs (`FORCE_*`; 0: none), so that every tool is exercised.
    pub force_inter_tool: u8,
}

/// `Tools::force_inter_tool`: the sharp filter (smooth across with
/// `dual_filter`).
pub const FORCE_FILTER: u8 = 1;
/// `Tools::force_inter_tool`: OBMC.
pub const FORCE_OBMC: u8 = 2;
/// `Tools::force_inter_tool`: local warped motion.
pub const FORCE_WARP: u8 = 3;
/// `Tools::force_inter_tool`: GLOBALMV with a global warp on every
/// reference (a small rotation and zoom, whatever the motion is).
pub const FORCE_GLOBAL: u8 = 4;
/// `Tools::force_inter_tool`: wedge compound.
pub const FORCE_WEDGE: u8 = 5;
/// `Tools::force_inter_tool`: difference-weighted compound.
pub const FORCE_DIFFWTD: u8 = 6;
/// `Tools::force_inter_tool`: distance-weighted compound.
pub const FORCE_DISTANCE: u8 = 7;
/// `Tools::force_inter_tool`: smooth inter-intra.
pub const FORCE_INTERINTRA: u8 = 8;
/// `Tools::force_inter_tool`: wedge inter-intra.
pub const FORCE_WEDGE_INTERINTRA: u8 = 9;

impl Tools {
    /// The tools a speed setting uses (0 slowest to 10 fastest).
    pub fn for_speed(speed: u32) -> Self {
        let s = speed.min(10);
        Tools {
            rdo: s <= 8,
            partition_rect: s <= 5,
            partition_4x4: s <= 1,
            prune_split: s >= 3,
            prune_partition: false,
            rd_skip: s <= 8,
            tx_size: s <= 8,
            tx_type_rd: s <= 5,
            tx_types: match s {
                0..=1 => 16,
                2..=3 => 7,
                4..=5 => 5,
                _ => 4,
            },
            full_tx_set: s <= 3,
            intra_candidates: match s {
                0..=1 => 5,
                2..=3 => 3,
                _ => 1,
            },
            inter_candidates: match s {
                0..=1 => 5,
                2..=6 => 3,
                _ => 2,
            },
            cdef: s <= 9,
            cdef_thorough: s <= 3,
            lf_search: s <= 9,
            multi_ref: s <= 3,
            cfl: s <= 8,
            restoration: s <= 3,
            restoration_thorough: s <= 2,
            compound: s <= 5,
            palette: s <= 8,
            filter_intra: false,
            intrabc: s <= 8,
            fast_me: s >= 4,
            fast_tx_refine: s >= 4,
            restoration_fast: s >= 4,
            abandon_trials: true,
            split_threshold: match s {
                0..=3 => 0,
                _ => 15,
            },
            // (Speeds 4 and 8 trade 100 for the inter tools' time: with
            // them, 150 is −2.0 % and −1.1 % BD-rate for 1.04x and 1.01x
            // the time.)
            split_threshold_small: match s {
                0..=3 => 0,
                4 | 8 => 150,
                _ => 100,
            },
            rd_skip_best: s >= 4,
            tx_depth_max: if s >= 4 { 1 } else { 2 },
            prune_intra: s >= 4,
            wavefront: s >= 4,
            aq: false,
            rdoq: match s {
                0..=3 => 3,
                4..=8 => 1,
                _ => 0,
            },
            tx_type_early: s >= 4,
            rect_refs: s >= 4,
            rect_min: if s >= 4 { 16 } else { 8 },
            rect_max: if s >= 4 { 16 } else { 128 },
            // The inter tools (`encoder::inter`) where the search runs:
            // see each field for what it was measured to be worth.
            switchable_filter: s <= 8,
            dual_filter: false,
            obmc: s <= 8,
            warped_motion: s <= 8,
            global_motion: false,
            masked_compound: false,
            jnt_comp: s <= 3,
            interintra: s <= 3,
            inter_tool_threshold: 0,
            uv_modes: false,
            // One at speed 4, which has no time to spare: as much gain.
            uv_trials: match s {
                4 => 1,
                0..=8 => 2,
                _ => 0,
            },
            uv_trials_inter: false,
            rdoq_uv: if s <= 8 { 1 } else { 0 },
            // One trial from speed 4: as much gain for less time.
            inter_tool_trials: if s >= 4 { 1 } else { 2 },
            force_inter_tool: 0,
        }
    }

    /// Whether any of the inter tools of `encoder::inter` is searched.
    pub(crate) fn inter_tools(&self) -> bool {
        self.switchable_filter
            || self.obmc
            || self.warped_motion
            || self.masked_compound
            || self.jnt_comp
            || self.interintra
            || self.force_inter_tool != 0
    }

    /// Sets the switch called `name` (a field name; numbers as 0 / 1 or a
    /// count); false if there is none.
    pub fn set(&mut self, name: &str, value: u32) -> bool {
        let b = value != 0;
        match name {
            "rdo" => self.rdo = b,
            "partition_rect" => self.partition_rect = b,
            "partition_4x4" => self.partition_4x4 = b,
            "prune_split" => self.prune_split = b,
            "prune_partition" => self.prune_partition = b,
            "rd_skip" => self.rd_skip = b,
            "tx_size" => self.tx_size = b,
            "tx_type_rd" => self.tx_type_rd = b,
            "tx_types" => self.tx_types = value.min(16) as u8,
            "full_tx_set" => self.full_tx_set = b,
            "intra_candidates" => self.intra_candidates = value.min(13) as u8,
            "inter_candidates" => self.inter_candidates = value.min(16) as u8,
            "cdef" => self.cdef = b,
            "cdef_thorough" => self.cdef_thorough = b,
            "lf_search" => self.lf_search = b,
            "multi_ref" => self.multi_ref = b,
            "cfl" => self.cfl = b,
            "restoration" => self.restoration = b,
            "restoration_thorough" => self.restoration_thorough = b,
            "compound" => self.compound = b,
            "palette" => self.palette = b,
            "filter_intra" => self.filter_intra = b,
            "intrabc" => self.intrabc = b,
            "fast_me" => self.fast_me = b,
            "fast_tx_refine" => self.fast_tx_refine = b,
            "restoration_fast" => self.restoration_fast = b,
            "abandon_trials" => self.abandon_trials = b,
            "split_threshold" => self.split_threshold = value.min(65535) as u16,
            "split_threshold_small" => self.split_threshold_small = value.min(65535) as u16,
            "rd_skip_best" => self.rd_skip_best = b,
            "tx_depth_max" => self.tx_depth_max = value.min(2) as u8,
            "prune_intra" => self.prune_intra = b,
            "wavefront" => self.wavefront = b,
            "aq" => self.aq = b,
            "rdoq" => self.rdoq = value.min(3) as u8,
            "tx_type_early" => self.tx_type_early = b,
            "rect_refs" => self.rect_refs = b,
            "rect_min" => self.rect_min = value.min(128) as u8,
            "rect_max" => self.rect_max = value.min(128) as u8,
            "switchable_filter" => self.switchable_filter = b,
            "dual_filter" => self.dual_filter = b,
            "obmc" => self.obmc = b,
            "warped_motion" => self.warped_motion = b,
            "global_motion" => self.global_motion = b,
            "masked_compound" => self.masked_compound = b,
            "jnt_comp" => self.jnt_comp = b,
            "interintra" => self.interintra = b,
            "inter_tool_threshold" => self.inter_tool_threshold = value.min(65535) as u16,
            "uv_modes" => self.uv_modes = b,
            "uv_trials" => self.uv_trials = value.min(12) as u8,
            "uv_trials_inter" => self.uv_trials_inter = b,
            "rdoq_uv" => self.rdoq_uv = value.min(2) as u8,
            "inter_tool_trials" => self.inter_tool_trials = value.min(255) as u8,
            "force_inter_tool" => self.force_inter_tool = value.min(255) as u8,
            _ => return false,
        }
        true
    }
}

impl Config {
    /// Settings for a `width` x `height` 8-bit 4:2:0 stream at quantiser
    /// 100, speed [`DEFAULT_SPEED`].
    pub fn new(width: u32, height: u32) -> Self {
        Config {
            width,
            height,
            bit_depth: 8,
            quantizer: 100,
            keyframe_interval: 60,
            target_bits_per_frame: None,
            loop_filter: None,
            search_range: 16,
            color: ColorInfo::default(),
            hdr: HdrMetadata::default(),
            speed: DEFAULT_SPEED,
            tile_cols_log2: 0,
            tools: Tools::for_speed(DEFAULT_SPEED),
            threads: 1,
            altref: DEFAULT_ALTREF,
            monochrome: false,
            chroma: ChromaFormat::Yuv420,
            film_grain: false,
        }
    }

    /// The chroma format coded: [`Config::chroma`], or monochrome.
    pub fn chroma_format(&self) -> ChromaFormat {
        if self.monochrome {
            ChromaFormat::Mono
        } else {
            self.chroma
        }
    }

    /// The `seq_profile` this format and bit depth need (6.4.1): 0 for
    /// 4:2:0 and monochrome at 8 and 10 bits, 1 for 4:4:4 at 8 and 10, 2 for
    /// 4:2:2 and for everything at 12 bits.
    pub fn profile(&self) -> u32 {
        match (self.chroma_format(), self.bit_depth) {
            (_, 12) => 2,
            (ChromaFormat::Yuv444, _) => 1,
            (ChromaFormat::Yuv422, _) => 2,
            _ => 0,
        }
    }
}

/// Encodes frames into AV1 temporal units.
pub struct Encoder {
    cfg: Config,
    seq: Arc<SequenceHeader>,
    /// The decoder whose reference state and in-loop filters the encoder
    /// shares; its output is the encoder's reconstruction.
    dec: Decoder,
    /// Frames coded (shown or not).
    frame_num: u64,
    /// Frames shown so far: the display index (order hint) of the next.
    shown: u64,
    /// Frames waiting for their group (alt-ref groups), and whether each
    /// is to be a key frame.
    queue: std::collections::VecDeque<(Frame, bool, FilmGrainParams)>,
    /// Frames taken, and since the last key frame among them.
    received: u64,
    received_since_key: u64,
    /// Temporal units ready to hand out (alt-ref mode), each with its shown
    /// frame's reconstruction and whether it is a key frame.
    ready: std::collections::VecDeque<(Vec<u8>, Option<Frame>, bool)>,
    /// The display indices of the hidden frames not yet shown (alt-ref
    /// groups).
    pending: Vec<u64>,
    /// The display index of the group's anchor: the last key frame or
    /// alt-ref frame shown (alt-ref groups), kept as a reference.
    anchor: u64,
    /// Rate control's plan for the group being coded.
    group_plan: Option<GroupPlan>,
    /// Frames since the last key frame.
    since_key: u64,
    /// The next frame is to be a key frame whatever the interval says.
    force_key: bool,
    /// Whether the last frame encoded was a key frame.
    last_key: bool,
    q: f64,
    /// Average-bitrate rate control (`Config::target_bits_per_frame`).
    rc: Option<rc::RateControl>,
    /// The previous frame's source planes (adaptive quantisation).
    prev_src: Option<Arc<Vec<Vec<u16>>>>,
    recon: Option<Frame>,
    /// The frame number each reference slot holds.
    slot_frame: [u64; NUM_REF_FRAMES],
    /// The pyramid layer of the frame each slot holds (0: a key or alt-ref
    /// frame).
    slot_layer: [u8; NUM_REF_FRAMES],
    /// The slots of `LAST_FRAME` and `LAST2_FRAME`.
    last_slot: usize,
    last2_slot: usize,
    /// The frame being encoded looks like screen content.
    next_screen_content: bool,
    /// The film grain parameters of the frame being encoded.
    next_grain: FilmGrainParams,
    /// The last frame taken (film grain: its noise is measured against
    /// it).
    grain_prev: Option<Frame>,
}

/// The slot the golden frame (the key frame, or a periodic boosted frame)
/// lives in when several references are used.
const GOLDEN_SLOT: usize = 2;

/// Rate control's plan for a group of frames.
struct GroupPlan {
    /// The plan's quantiser.
    q: u32,
    /// The display index of the group's first frame.
    first: u64,
    /// Each frame's layer and complexity, in display order.
    frames: Vec<(u8, rc::Complexity)>,
}

impl GroupPlan {
    /// The layer and complexity of the frame of display index `hint`.
    fn frame(&self, hint: u64) -> (u8, rc::Complexity) {
        self.frames[(hint - self.first) as usize]
    }
}

/// What one frame's header says, as the encoder chooses it.
#[derive(Clone, Debug)]
struct FrameParams {
    key: bool,
    qidx: u32,
    refresh: u8,
    /// `ref_frame_idx[ LAST_FRAME..ALTREF_FRAME ]`.
    ref_slots: [usize; REFS_PER_FRAME],
    /// The references inter prediction may search (distinct frames).
    search_refs: Vec<i32>,
    /// `loop_filter_level[ 0..4 ]`.
    lf: [u32; 4],
    cdef: Option<cdef::CdefParams>,
    /// `FrameRestorationType` per plane, when restoration is on.
    lr: Option<[u8; 3]>,
    /// `reference_select`: compound prediction allowed.
    reference_select: bool,
    /// `allow_screen_content_tools` (palettes).
    screen_content: bool,
    tx_select: bool,
    reduced_tx_set: bool,
    tile_cols_log2: u32,
    /// Adaptive quantisation: each superblock's quantiser.
    aq: Option<Arc<tile::AqMap>>,
    /// `show_frame` (a hidden alt-ref frame is not shown).
    show: bool,
    /// The frame's display index.
    order_hint: u64,
    /// `film_grain_params()`.
    grain: FilmGrainParams,
    /// `allow_intrabc`, and the source's hash for its search.
    intrabc: bool,
    ibc: Option<Arc<tile::IbcHash>>,
    /// Each reference's global motion (`gm_type`, `gm_params`), indexed by
    /// reference frame (`LAST_FRAME..=ALTREF_FRAME`), and the parameters
    /// they are coded against (`PrevGmParams`).
    gm: [(u32, [i32; 6]); 8],
    prev_gm: [[i32; 6]; 8],
}

impl FrameParams {
    /// Whether the in-loop filters are coded: not in a lossless frame
    /// (`CodedLossless`) nor one with intra block copy, whose loop filter,
    /// CDEF and loop restoration parameters the header leaves out (5.9.11,
    /// 5.9.19, 5.9.20).
    fn filters(&self) -> bool {
        self.qidx > 0 && !self.intrabc
    }
}

/// What a coded frame is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Key,
    Inter,
    /// A frame of an alt-ref group: its display index, its layer of the
    /// pyramid (0 the alt-ref frame), whether it is of the group's top
    /// layer (a reference for few frames, if any), and whether it is shown
    /// now or hidden until a `show_existing_frame`.
    Group {
        hint: u64,
        layer: u8,
        top: bool,
        /// The group's top layer.
        depth: u8,
        show: bool,
    },
}

/// What a second coding pass replays.
struct Replay {
    /// The first pass's decision log, per tile.
    logs: Vec<Vec<rdo::Decision>>,
    /// Each 64x64 block's CDEF index.
    cdef: Option<Vec<i8>>,
    /// Each restoration unit's parameters.
    lr: Option<Arc<lr::LrPlan>>,
}

/// The coded frame: its state for the in-loop filters, and its tiles.
struct Coded {
    f: crate::decoder::FrameCtx,
    header: Vec<u8>,
    tiles: Vec<Vec<u8>>,
    logs: Vec<Vec<rdo::Decision>>,
}

impl Encoder {
    /// An encoder with the given settings.
    pub fn new(cfg: Config) -> Self {
        let bits = |v: u32| 32 - (v.max(1) - 1).leading_zeros().min(31);
        let w_bits = bits(cfg.width).max(1);
        let h_bits = bits(cfg.height).max(1);
        let chroma = cfg.chroma_format();
        let (ss_x, ss_y) = chroma.shifts();
        let seq = SequenceHeader {
            seq_profile: cfg.profile(),
            still_picture: false,
            reduced_still_picture_header: false,
            timing_info_present: false,
            decoder_model_info_present: false,
            equal_picture_interval: false,
            buffer_removal_time_length_minus_1: 0,
            frame_presentation_time_length_minus_1: 0,
            operating_points_cnt_minus_1: 0,
            operating_point_idc: vec![0],
            seq_level_idx: vec![31],
            decoder_model_present_for_this_op: vec![false],
            op_idc: 0,
            frame_width_bits: w_bits,
            frame_height_bits: h_bits,
            max_frame_width: cfg.width,
            max_frame_height: cfg.height,
            frame_id_numbers_present: false,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            use_128x128_superblock: false,
            enable_filter_intra: cfg.tools.filter_intra,
            enable_intra_edge_filter: true,
            enable_interintra_compound: cfg.tools.interintra,
            enable_masked_compound: cfg.tools.masked_compound,
            enable_warped_motion: cfg.tools.warped_motion,
            enable_dual_filter: cfg.tools.switchable_filter && cfg.tools.dual_filter,
            enable_order_hint: true,
            enable_jnt_comp: cfg.tools.jnt_comp,
            // The motion field projected from the references (7.9): their
            // vectors as candidates (-1.8 % BD-rate without alt-ref groups,
            // -1.1 % with them).
            enable_ref_frame_mvs: true,
            // Screen content tools chosen per frame when palettes may be
            // used.
            seq_force_screen_content_tools: if cfg.tools.palette || cfg.tools.intrabc {
                SELECT_SCREEN_CONTENT_TOOLS
            } else {
                0
            },
            seq_force_integer_mv: SELECT_INTEGER_MV,
            order_hint_bits: 7,
            enable_superres: false,
            enable_cdef: cfg.tools.cdef,
            enable_restoration: cfg.tools.restoration,
            color: ColorConfig {
                bit_depth: cfg.bit_depth,
                mono_chrome: chroma == ChromaFormat::Mono,
                num_planes: chroma.num_planes(),
                color_primaries: cfg.color.color_primaries,
                transfer_characteristics: cfg.color.transfer_characteristics,
                matrix_coefficients: cfg.color.matrix_coefficients,
                color_range: cfg.color.full_range,
                subsampling_x: ss_x,
                subsampling_y: ss_y,
                chroma_sample_position: cfg.color.chroma_sample_position,
                separate_uv_delta_q: false,
                color_description_present: cfg.color.color_primaries != CP_UNSPECIFIED
                    || cfg.color.transfer_characteristics != TC_UNSPECIFIED
                    || cfg.color.matrix_coefficients != MC_UNSPECIFIED,
            },
            film_grain_params_present: cfg.film_grain,
        };
        let q = cfg.quantizer.clamp(1, 255) as f64;
        let rc = cfg.target_bits_per_frame.map(|t| {
            rc::RateControl::new(
                t,
                cfg.width,
                cfg.height,
                cfg.bit_depth,
                cfg.keyframe_interval,
                cfg.quantizer,
                cfg.altref >= 3 && cfg.tools.rdo,
            )
        });
        let mut dec = Decoder::new();
        dec.set_threads(cfg.threads.max(1));
        Encoder {
            cfg,
            seq: Arc::new(seq),
            dec,
            frame_num: 0,
            shown: 0,
            queue: Default::default(),
            received: 0,
            received_since_key: 0,
            ready: Default::default(),
            pending: Vec::new(),
            anchor: 0,
            group_plan: None,
            since_key: 0,
            force_key: false,
            last_key: false,
            q,
            rc,
            prev_src: None,
            recon: None,
            slot_frame: [0; NUM_REF_FRAMES],
            slot_layer: [0; NUM_REF_FRAMES],
            last_slot: 0,
            last2_slot: 1,
            next_screen_content: false,
            next_grain: Default::default(),
            grain_prev: None,
        }
    }

    /// The reconstruction of the last frame encoded: what a decoder will
    /// output for it.
    pub fn reconstruction(&self) -> Option<&Frame> {
        self.recon.as_ref()
    }

    /// Makes the next frame a key frame (with its sequence header and
    /// metadata OBUs, so the stream can be entered there), whatever the
    /// key frame interval says; the interval restarts from it. Nothing
    /// else is reset: the rate controller carries on.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    /// Whether the next frame given to [`Encoder::encode`] will be a key
    /// frame.
    pub fn next_is_keyframe(&self) -> bool {
        self.force_key
            || self.received == 0
            || self.received_since_key >= self.cfg.keyframe_interval.max(1) as u64
    }

    /// Whether the last frame encoded was a key frame.
    pub fn last_was_keyframe(&self) -> bool {
        self.last_key
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The quantiser index the next frame will use.
    pub fn quantizer(&self) -> u32 {
        self.q.round().clamp(1.0, 255.0) as u32
    }

    /// Takes one frame and returns the next temporal unit (temporal
    /// delimiter, sequence header and metadata on key frames, the frame
    /// OBUs): this frame's, or with alt-ref groups ([`Config::altref`]) an
    /// earlier frame's in order, or an empty vector while the encoder
    /// holds the frames of a group. [`Encoder::reconstruction`] and
    /// [`Encoder::last_was_keyframe`] are of the frame the unit shows.
    pub fn encode(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        let cfg = &self.cfg;
        if frame.width != cfg.width || frame.height != cfg.height {
            return Err(Error::invalid("frame size differs from the configuration"));
        }
        if frame.bit_depth != cfg.bit_depth || frame.chroma != cfg.chroma_format() {
            return Err(Error::invalid(format!(
                "the encoder is configured for {:?} frames of {} bits",
                cfg.chroma_format(),
                cfg.bit_depth
            )));
        }
        if ![8, 10, 12].contains(&cfg.bit_depth) {
            return Err(Error::invalid("bit depth must be 8, 10 or 12"));
        }
        check_color(&cfg.color, cfg.chroma_format())?;
        let key = self.next_is_keyframe();
        self.force_key = false;
        self.received += 1;
        self.received_since_key = if key { 1 } else { self.received_since_key + 1 };
        // Film grain: the frame is coded denoised, its grain signalled
        // (measured here, in display order, against the frame before).
        let (frame, grain) = if self.cfg.film_grain {
            let seed = (self.received.wrapping_mul(7919).wrapping_add(4321) as u16) | 1;
            let prev = self.grain_prev.replace(frame.clone());
            grain::estimate(frame, prev.as_ref(), seed, self.cfg.color.full_range)
        } else {
            (frame.clone(), FilmGrainParams::default())
        };
        if !self.altref_groups() {
            self.next_grain = grain;
            let (obus, recon) = self.code_one(&frame, if key { Role::Key } else { Role::Inter })?;
            let tu = self.temporal_unit(key, &[], &obus);
            self.recon = recon;
            self.last_key = key;
            return Ok(tu);
        }
        self.queue.push_back((frame, key, grain));
        self.pump(false)?;
        Ok(self.pop_ready().unwrap_or_default())
    }

    /// The temporal units of the frames the encoder still holds (alt-ref
    /// groups), one a call, in order; `None` once there are none. The
    /// encoder can take more frames afterwards.
    pub fn flush(&mut self) -> Result<Option<Vec<u8>>> {
        self.pump(true)?;
        Ok(self.pop_ready())
    }

    /// Whether every frame is lossless: quantiser 0, without rate control.
    fn lossless(&self) -> bool {
        self.cfg.quantizer == 0 && self.cfg.target_bits_per_frame.is_none()
    }

    /// Whether frames are coded in alt-ref groups: asked for, and with the
    /// rate-distortion search, which searches the references after a
    /// frame (without it, speeds 9 and 10, groups cost 3 % to 7 %).
    fn altref_groups(&self) -> bool {
        self.cfg.altref >= 3 && self.cfg.tools.rdo
    }

    /// The frames of a group (up to 32: the order hints of a group's
    /// frames stay within half their range of each other).
    fn group_len(&self) -> u64 {
        self.cfg.altref.min(32) as u64
    }

    /// The next ready temporal unit, its reconstruction and key flag made
    /// current.
    fn pop_ready(&mut self) -> Option<Vec<u8>> {
        let (tu, recon, key) = self.ready.pop_front()?;
        self.recon = recon;
        self.last_key = key;
        Some(tu)
    }

    /// Codes the frames held whose group is complete (all of them when
    /// `all`).
    fn pump(&mut self, all: bool) -> Result<()> {
        while let Some(&(_, key, _)) = self.queue.front() {
            if key {
                let (frame, _, grain) = self.queue.pop_front().expect("a frame");
                self.next_grain = grain;
                let (obus, recon) = self.code_one(&frame, Role::Key)?;
                let tu = self.temporal_unit(true, &[], &obus);
                self.ready.push_back((tu, recon, true));
                continue;
            }
            // The group: up to the alt-ref interval, not past the next key
            // frame (held, or not yet taken).
            let interval = self.cfg.keyframe_interval.max(1) as u64;
            let mut g =
                self.group_len()
                    .min(interval.saturating_sub(self.since_key).max(1)) as usize;
            if let Some(k) = self.queue.iter().skip(1).position(|q| q.1) {
                g = g.min(k + 1);
            }
            if self.queue.len() < g && !all {
                break;
            }
            g = g.min(self.queue.len());
            self.code_group(g)?;
        }
        Ok(())
    }

    /// Codes the next `g` frames held as a group (see `gop`): its temporal
    /// units, one a frame, made ready.
    fn code_group(&mut self, g: usize) -> Result<()> {
        let steps = gop::plan(g as u32);
        let top = steps
            .iter()
            .map(|s| match *s {
                gop::Step::Code { layer, .. } => layer,
                gop::Step::Show { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let base = self.shown;
        if self.rc.is_some() {
            self.plan_group(&steps, top, g);
        }
        // Frames not shown go out in the temporal unit of the next one shown.
        let mut hidden = Vec::new();
        for step in steps {
            match step {
                gop::Step::Code { idx, layer, show } => {
                    let at = idx as usize - 1;
                    let frame = self.queue[at].0.clone();
                    self.next_grain = self.queue[at].2.clone();
                    let role = Role::Group {
                        hint: base + at as u64,
                        layer,
                        top: layer == top,
                        depth: top,
                        show,
                    };
                    let (obus, recon) = self.code_one(&frame, role)?;
                    if show {
                        let tu = self.temporal_unit(false, &hidden, &obus);
                        hidden.clear();
                        self.ready.push_back((tu, recon, false));
                    } else {
                        hidden.extend_from_slice(&obus);
                    }
                }
                gop::Step::Show { idx } => {
                    let (obu, recon) = self.show_existing(base + idx as u64 - 1)?;
                    let tu = self.temporal_unit(false, &hidden, &obu);
                    hidden.clear();
                    self.ready.push_back((tu, recon, false));
                }
            }
        }
        self.queue.drain(..g);
        self.group_plan = None;
        Ok(())
    }

    /// Rate control's plan for the group of the next `g` frames held,
    /// coded in `steps` with `top` its top layer: their complexities
    /// measured in display order, then the quantiser.
    fn plan_group(&mut self, steps: &[gop::Step], top: u8, g: usize) {
        let mut layers = vec![top; g];
        for s in steps {
            if let gop::Step::Code { idx, layer, .. } = *s {
                layers[idx as usize - 1] = layer;
            }
        }
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        let mut group = Vec::with_capacity(g);
        for (i, &layer) in layers.iter().enumerate() {
            let (src, stride) = self.source_planes(&self.queue[i].0);
            let rc = self.rc.as_mut().expect("rate control");
            group.push((layer, rc.measure(&src[0], stride[0], w, h, true)));
        }
        self.group_plan = Some(GroupPlan {
            q: 0,
            first: self.shown,
            frames: group,
        });
        self.replan_group();
    }

    /// The group's quantiser planned (again) from rate control's models.
    fn replan_group(&mut self) {
        let bit_depth = self.cfg.bit_depth;
        let (Some(rc), Some(plan)) = (self.rc.as_ref(), self.group_plan.as_mut()) else {
            return;
        };
        let top = plan.frames.iter().map(|g| g.0).max().unwrap_or(0);
        let frame_q = |q: u32, layer: u8| {
            let role = Role::Group {
                hint: 0,
                layer,
                top: layer == top,
                depth: top,
                show: true,
            };
            frame_qidx(q, role, false, true, true, bit_depth)
        };
        plan.q = rc.plan_group(&plan.frames, frame_q);
    }

    /// A temporal unit: the delimiter, on key frames the sequence header and
    /// metadata, then `hidden` (frame OBUs not shown) and `obus`.
    fn temporal_unit(&self, key: bool, hidden: &[u8], obus: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_obu(&mut out, OBU_TEMPORAL_DELIMITER, &[]);
        if key {
            write_obu(&mut out, OBU_SEQUENCE_HEADER, &self.seq.write());
            if let Some(c) = &self.cfg.hdr.content_light {
                write_obu(&mut out, OBU_METADATA, &crate::obu::write_hdr_cll(c));
            }
            if let Some(m) = &self.cfg.hdr.mastering_display {
                write_obu(&mut out, OBU_METADATA, &crate::obu::write_hdr_mdcv(m));
            }
        }
        out.extend_from_slice(hidden);
        out.extend_from_slice(obus);
        out
    }

    /// The frame header OBU that shows the hidden frame of display index
    /// `hint` (`show_existing_frame`), and the frame shown.
    fn show_existing(&mut self, hint: u64) -> Result<(Vec<u8>, Option<Frame>)> {
        let slot = (0..NUM_REF_FRAMES)
            .find(|&s| self.slot_frame[s] == hint)
            .expect("the hidden frame to show is held");
        self.pending.retain(|&h| h != hint);
        let mut w = BitWriter::new();
        w.flag(true); // show_existing_frame
        w.f(3, slot as u32); // frame_to_show_map_idx
        // A frame header OBU ends in trailing_bits() (5.3.1), like every
        // OBU but a frame or tile group.
        w.trailing_bits();
        let mut obu = Vec::new();
        write_obu(&mut obu, OBU_FRAME_HEADER, &w.finish());
        // The encoder's decoder shows it as any decoder will.
        let mut tu = Vec::new();
        write_obu(&mut tu, OBU_TEMPORAL_DELIMITER, &[]);
        tu.extend_from_slice(&obu);
        let mut recon = self.dec.decode(&tu)?;
        if let Some(r) = recon.as_mut() {
            r.hdr = self.cfg.hdr;
        }
        if self.slot_layer[slot] == 0 {
            self.anchor = hint;
        }
        self.shown += 1;
        self.since_key += 1;
        if let Some(rc) = self.rc.as_mut() {
            rc.account_shown(tu.len() as u64 * 8);
        }
        Ok((obu, recon))
    }

    /// Codes one frame and commits it to the reference state: its frame
    /// OBU, and its reconstruction when it is shown.
    fn code_one(&mut self, frame: &Frame, role: Role) -> Result<(Vec<u8>, Option<Frame>)> {
        let key = role == Role::Key;
        if self.cfg.tools.palette || self.cfg.tools.intrabc {
            self.next_screen_content = looks_like_screen_content(frame);
        }
        let seq = self.seq.clone();
        if self.dec.seq.is_none() || key {
            self.dec.seq = Some(seq.clone());
        }
        let (src, stride) = self.source_planes(frame);
        let src = Arc::new(src);
        let tools = self.cfg.tools;
        let golden = role == Role::Inter
            && tools.multi_ref
            && self.since_key.is_multiple_of(GOLDEN_INTERVAL);
        let bit_depth = self.cfg.bit_depth;
        let groups = self.altref_groups();
        let frame_q = move |q: u32| frame_qidx(q, role, golden, groups, true, bit_depth);
        let shown = !matches!(role, Role::Group { show: false, .. });
        // Rate control: the quantiser planned for this frame.
        let mut planned = None;
        if let Some(rc) = self.rc.as_mut() {
            if let (Role::Group { hint, layer, .. }, Some(plan)) = (role, &self.group_plan) {
                // The group's plan.
                let c = plan.frame(hint).1;
                planned = Some(rc.planned(layer, c, plan.q, frame_q(plan.q)));
            } else {
                let class = if key {
                    rc::Class::Key
                } else {
                    rc::Class::Inter
                };
                let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
                let c = rc.measure(&src[0], stride[0], w, h, shown);
                planned = Some(rc.plan(class, c, frame_q));
            }
        }
        let (mut p, mut f, mut obus) =
            self.code_tu(role, &src, &stride, planned.map(|x| x.qidx))?;
        if let (Some(pl), Some(rc)) = (planned.as_mut(), self.rc.as_mut())
            && let Role::Group { hint, layer: 0, .. } = role
            && rc.rescale_layers(pl, obus.len() as u64 * 8)
        {
            // The group's first frame (its alt-ref frame) far from the first
            // guess of the layers' models: the group planned again and the
            // frame coded again.
            self.replan_group();
            let plan = self.group_plan.as_ref().expect("a group plan");
            let rc = self.rc.as_ref().expect("rate control");
            *pl = rc.planned(0, plan.frame(hint).1, plan.q, frame_q(plan.q));
            (p, f, obus) = self.code_tu(role, &src, &stride, Some(pl.qidx))?;
        }
        if let (Some(pl), Some(rc)) = (planned.as_mut(), self.rc.as_mut())
            && let Some(again) = rc.redo(pl, obus.len() as u64 * 8, frame_q)
        {
            // Far from the plan: planned again and coded again.
            *pl = again;
            (p, f, obus) = self.code_tu(role, &src, &stride, Some(pl.qidx))?;
        }
        self.dec.shown.clear();
        self.dec.finish_frame(f)?;
        let mut recon = self.dec.shown.pop();
        if let Some(r) = recon.as_mut() {
            // What a decoder reports after this temporal unit's metadata.
            r.hdr = self.cfg.hdr;
        }
        // The reference slots now hold this frame.
        let layer = match role {
            Role::Group { layer, .. } => layer,
            Role::Key => 0,
            Role::Inter => 1,
        };
        for i in 0..NUM_REF_FRAMES {
            if (p.refresh >> i) & 1 != 0 {
                self.slot_frame[i] = p.order_hint;
                self.slot_layer[i] = layer;
            }
        }
        if key {
            self.last_slot = 0;
            self.last2_slot = 1;
            self.pending.clear();
            self.anchor = p.order_hint;
        } else if let Role::Group { show, .. } = role {
            if !show {
                self.pending.push(p.order_hint);
            }
        } else if tools.multi_ref {
            self.last2_slot = self.last_slot;
            self.last_slot = (p.refresh & !(1 << GOLDEN_SLOT)).trailing_zeros() as usize;
        }
        if shown {
            self.prev_src = Some(src.clone());
            self.shown += 1;
            self.since_key = if key { 1 } else { self.since_key + 1 };
        }
        self.frame_num += 1;
        if let (Some(pl), Some(rc)) = (planned.as_ref(), self.rc.as_mut()) {
            rc.update(
                pl,
                obus.len() as u64 * 8,
                match role {
                    Role::Inter => !golden,
                    Role::Group { top, .. } => top,
                    Role::Key => false,
                },
                shown,
            );
            self.q = rc.base_q as f64;
        }
        Ok((obus, recon))
    }

    /// Codes one frame (both passes and the filter searches) into its
    /// temporal unit, at quantiser `qidx` (else the configured one), without
    /// committing it: the frame state comes back for `finish_frame`.
    fn code_tu(
        &mut self,
        role: Role,
        src: &Arc<Vec<Vec<u16>>>,
        stride: &[usize],
        qidx: Option<u32>,
    ) -> Result<(FrameParams, crate::decoder::FrameCtx, Vec<u8>)> {
        let tools = self.cfg.tools;
        let (src, stride) = (src.clone(), stride.to_vec());
        let mut p = self.frame_params(role, qidx);
        if tools.aq
            && p.qidx > 0
            && let Some(prev) = self.prev_src.as_ref()
        {
            let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
            if let Some((sb_cols, q)) = aq::quantisers(
                &src[0],
                &prev[0],
                stride[0],
                w,
                h,
                self.cfg.bit_depth,
                p.qidx,
            ) {
                let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
                let lambda = q
                    .iter()
                    .map(|&qi| {
                        (
                            0.4 * AC_QLOOKUP[bdi][qi as usize] as f64,
                            rd_lambda(&self.cfg, qi as u32),
                        )
                    })
                    .collect();
                p.aq = Some(Arc::new(tile::AqMap { sb_cols, q, lambda }));
            }
        }
        if p.intrabc {
            let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
            p.ibc = Some(Arc::new(tile::IbcHash::new(&src[0], stride[0], w, h)));
        }
        if !p.key && (tools.global_motion || tools.force_inter_tool == FORCE_GLOBAL) {
            self.global_motion(&mut p, &src, &stride);
        }
        let mut coded = self.code_frame(&p, &src, &stride, None)?;
        let filters = p.filters();
        if (filters && (tools.lf_search || tools.cdef || tools.restoration))
            || coded.tiles.is_empty()
        {
            // The in-loop filters' parameters, chosen on the first pass's
            // reconstruction; then the frame again with them, replaying the
            // first pass's decisions.
            let mut f = coded.f;
            if filters && tools.lf_search && self.cfg.loop_filter.is_none() {
                p.lf = search_lf(&mut f, &src, &stride, p.lf, self.cfg.threads.max(1));
            }
            f.hdr.loop_filter_level = p.lf.map(|l| l as i32);
            if p.lf[0] != 0 || p.lf[1] != 0 {
                crate::decoder::postfilter::loop_filter_threads(&mut f, self.cfg.threads.max(1));
            }
            let mut cdef_table = None;
            let lambda = rd_lambda(&self.cfg, p.qidx);
            if filters && tools.cdef {
                let (params, table) = cdef::search(
                    &f,
                    &src,
                    &stride,
                    lambda,
                    tools.cdef_thorough,
                    self.cfg.threads.max(1),
                );
                // The chosen CDEF, as the decoder will apply it.
                let h = &mut f.hdr;
                h.cdef_damping = params.damping_minus_3 as i32 + 3;
                h.cdef_bits = params.bits;
                for (i, (&(yp, ys), &(up, us))) in params.y.iter().zip(&params.uv).enumerate() {
                    let sec = |s: u32| if s == 3 { 4 } else { s as i32 };
                    h.cdef_y_pri_strength[i] = yp as i32;
                    h.cdef_y_sec_strength[i] = sec(ys);
                    h.cdef_uv_pri_strength[i] = up as i32;
                    h.cdef_uv_sec_strength[i] = sec(us);
                }
                f.cdef_idx.copy_from_slice(&table);
                p.cdef = Some(params);
                cdef_table = Some(table);
            }
            let mut lr_plan = None;
            if filters && tools.restoration {
                let cdef_frame = crate::decoder::postfilter::cdef(&f, self.cfg.threads.max(1));
                let plan = lr::search(
                    &f,
                    &f.cur,
                    &cdef_frame,
                    &src,
                    &stride,
                    lambda,
                    if tools.restoration_thorough {
                        lr::Effort::Thorough
                    } else if tools.restoration_fast {
                        lr::Effort::Fast
                    } else {
                        lr::Effort::Normal
                    },
                    self.cfg.threads.max(1),
                );
                p.lr = Some(plan.frame_type);
                lr_plan = Some(Arc::new(plan));
            }
            let replay = Replay {
                logs: std::mem::take(&mut coded.logs),
                cdef: cdef_table,
                lr: lr_plan,
            };
            coded = self.code_frame(&p, &src, &stride, Some(replay))?;
        }
        let Coded {
            f, header, tiles, ..
        } = coded;
        let mut payload = header;
        let n = tiles.len();
        if n > 1 {
            // tile_start_and_end_present_flag = 0, then byte alignment.
            payload.push(0);
        }
        for (i, t) in tiles.iter().enumerate() {
            if i + 1 < n {
                payload.extend_from_slice(&((t.len() - 1) as u32).to_le_bytes());
            }
            payload.extend_from_slice(t);
        }
        let mut out = Vec::new();
        write_obu(&mut out, OBU_FRAME, &payload);
        Ok((p, f, out))
    }

    /// Each searched reference's global motion (`FrameParams::gm`),
    /// estimated against its reconstruction.
    fn global_motion(&self, p: &mut FrameParams, src: &[Vec<u16>], stride: &[usize]) {
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        for &rf in &p.search_refs {
            let slot = p.ref_slots[(rf - LAST_FRAME) as usize];
            let Some(r) = self.dec.refs[slot].as_ref() else {
                continue;
            };
            let model = if self.cfg.tools.force_inter_tool == FORCE_GLOBAL {
                // A small rotation and zoom about the frame's centre.
                gm::forced(w, h)
            } else {
                let pl = &r.frame().planes[0];
                gm::estimate(
                    &src[0],
                    stride[0],
                    pl.data.as_slice(),
                    pl.stride,
                    w,
                    h,
                    self.cfg.bit_depth,
                )
            };
            if let Some(m) = model {
                p.gm[rf as usize] = m;
            }
        }
    }

    /// The header parameters of the next frame (before the in-loop filter
    /// search).
    fn frame_params(&self, role: Role, planned_q: Option<u32>) -> FrameParams {
        let tools = self.cfg.tools;
        let key = role == Role::Key;
        let order_hint = match role {
            Role::Group { hint, .. } => hint,
            _ => self.shown,
        };
        // (Rate control plans the frame's offset in.)
        let golden = role == Role::Inter
            && tools.multi_ref
            && self.since_key.is_multiple_of(GOLDEN_INTERVAL);
        let qidx = if self.lossless() {
            0
        } else {
            planned_q.unwrap_or_else(|| {
                frame_qidx(
                    self.quantizer(),
                    role,
                    golden,
                    self.altref_groups(),
                    false,
                    self.cfg.bit_depth,
                )
            })
        };
        let intrabc = key && tools.intrabc && tools.rdo && self.next_screen_content;
        let filters = qidx > 0 && !intrabc;
        let mut refresh = 0xFFu8;
        let mut ref_slots = [0usize; REFS_PER_FRAME];
        let mut search_refs = Vec::new();
        if let Role::Group { .. } = role {
            (refresh, ref_slots, search_refs) = self.group_refs(order_hint);
        } else if !key {
            if tools.multi_ref {
                // This frame replaces the older of the last two (not LAST);
                // every GOLDEN_INTERVAL frames it is also the golden frame,
                // coded finer.
                let golden = role == Role::Inter && self.since_key.is_multiple_of(GOLDEN_INTERVAL);
                let target = [0, 1]
                    .into_iter()
                    .filter(|&s| s != self.last_slot)
                    .min_by_key(|&s| self.slot_frame[s])
                    .expect("a slot to refresh");
                refresh = 1 << target;
                if golden {
                    refresh |= 1 << GOLDEN_SLOT;
                }
                ref_slots = [
                    self.last_slot,
                    self.last2_slot,
                    self.last2_slot,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                ];
                search_refs.push(LAST_FRAME);
                let lf = self.slot_frame[self.last_slot];
                let l2 = self.slot_frame[self.last2_slot];
                let g = self.slot_frame[GOLDEN_SLOT];
                if l2 != lf {
                    search_refs.push(LAST2_FRAME);
                }
                if g != lf && g != l2 {
                    search_refs.push(GOLDEN_FRAME);
                }
            } else {
                refresh = 1;
                search_refs.push(LAST_FRAME);
            }
        }
        let lf = self
            .cfg
            .loop_filter
            .unwrap_or_else(|| ((qidx as f64) * 0.18 + 2.0).min(40.0) as u32);
        let chroma_lf = if lf != 0 { lf / 2 + 1 } else { 0 };
        // Compound prediction where the speed has it, and in a group
        // wherever a frame has a reference on each side.
        let bidir = matches!(role, Role::Group { .. })
            && search_refs
                .iter()
                .any(|&r| self.slot_frame[ref_slots[(r - LAST_FRAME) as usize]] > order_hint);
        let reference_select =
            !key && search_refs.len() > 1 && ((tools.compound && tools.multi_ref) || bidir);
        let screen_content = (tools.palette || tools.intrabc) && self.next_screen_content;
        FrameParams {
            key,
            qidx,
            refresh,
            ref_slots,
            search_refs,
            lf: if filters {
                [lf, lf, chroma_lf, chroma_lf]
            } else {
                [0; 4]
            },
            cdef: if tools.cdef && filters {
                Some(cdef::CdefParams::off())
            } else {
                None
            },
            // Switchable in every plane on the first pass, every unit off:
            // the units exist for the search.
            lr: if tools.restoration && filters {
                Some([RESTORE_SWITCHABLE; 3])
            } else {
                None
            },
            reference_select,
            screen_content,
            tx_select: tools.tx_size && tools.rdo && qidx > 0,
            reduced_tx_set: !(tools.full_tx_set && tools.rdo),
            tile_cols_log2: self.cfg.tile_cols_log2,
            aq: None,
            show: !matches!(role, Role::Group { show: false, .. }),
            order_hint,
            intrabc,
            ibc: None,
            grain: self.next_grain.clone(),
            gm: [(IDENTITY, gm::IDENTITY_PARAMS); 8],
            // PrevGmParams: those of LAST's slot (primary_ref_frame).
            prev_gm: if key {
                [gm::IDENTITY_PARAMS; 8]
            } else {
                self.dec.ref_state.slots[ref_slots[0]]
                    .as_ref()
                    .map_or([gm::IDENTITY_PARAMS; 8], |s| s.gm_params)
            },
        }
    }

    /// The source planes, padded to the frame buffers' size by repeating
    /// the last column and row.
    fn source_planes(&self, frame: &Frame) -> (Vec<Vec<u16>>, Vec<usize>) {
        let mi_cols = 2 * ((self.cfg.width as usize + 7) >> 3);
        let mi_rows = 2 * ((self.cfg.height as usize + 7) >> 3);
        let aw = (mi_cols * MI_SIZE + 127) & !127;
        let ah = (mi_rows * MI_SIZE + 127) & !127;
        let mut src = Vec::new();
        let mut stride = Vec::new();
        let (ssx, ssy) = frame.chroma.shifts();
        for p in 0..3 {
            let (sw, sh) = if p == 0 {
                (aw + 32, ah + 32)
            } else {
                ((aw >> ssx) + 32, (ah >> ssy) + 32)
            };
            if p >= frame.chroma.num_planes() {
                // A monochrome frame: no chroma to code (the planes are
                // never read), a flat one keeps the shape.
                src.push(vec![1u16 << (frame.bit_depth - 1); sw * sh]);
                stride.push(sw);
                continue;
            }
            let pl = frame.planes[p];
            let (w, h) = (pl.width as usize, pl.height as usize);
            let mut v = vec![0u16; sw * sh];
            for y in 0..sh {
                let sy = y.min(h - 1);
                let row = &mut v[y * sw..(y + 1) * sw];
                for (x, o) in row.iter_mut().enumerate() {
                    *o = frame.sample(p, x.min(w - 1) as u32, sy as u32);
                }
            }
            src.push(v);
            stride.push(sw);
        }
        (src, stride)
    }

    /// Writes the header for `p`, parses it back with the decoder's parser
    /// (the frame state is then exactly what a decoder sets up) and codes
    /// every tile: searching, or replaying an earlier pass.
    fn code_frame(
        &mut self,
        p: &FrameParams,
        src: &Arc<Vec<Vec<u16>>>,
        stride: &[usize],
        replay: Option<Replay>,
    ) -> Result<Coded> {
        let header = self.write_frame_header(p);
        let seq = self.seq.clone();
        let mut r = BitReader::new(&header);
        let hdr = FrameHeader::parse(&mut r, &seq, &mut self.dec.ref_state, 0, 0)?;
        let mut f = self.dec.setup_frame(seq, hdr)?;
        for rf in LAST_FRAME as usize..=ALTREF_FRAME as usize {
            debug_assert_eq!(
                (f.hdr.gm_type[rf], f.hdr.gm_params[rf]),
                p.gm[rf],
                "global motion of reference {rf} as parsed"
            );
        }
        let ti = f.hdr.tile_info.clone();
        let num_tiles = ti.cols * ti.rows;
        let mut replay = replay;
        let mut jobs: Vec<std::sync::Mutex<Option<Box<tile::EncCtx>>>> = Vec::new();
        for t in 0..num_tiles {
            let mut enc = self.enc_ctx(p, src, stride);
            if let Some(rp) = replay.as_mut() {
                enc.rdo.replay = std::mem::take(&mut rp.logs[t]).into();
                enc.cdef_table = rp.cdef.clone();
                enc.lr_plan = rp.lr.clone();
            }
            jobs.push(std::sync::Mutex::new(Some(enc)));
        }
        // One tile: the bytes, the decision log, the CDFs if it is the tile
        // the frame keeps them from.
        type TileOut = (
            Vec<u8>,
            Vec<rdo::Decision>,
            Option<Box<crate::cdf::CdfContext>>,
        );
        let code_tile = |f: &mut crate::decoder::FrameCtx, t: usize| -> Result<TileOut> {
            let enc = jobs[t].lock().expect("job").take().expect("job");
            let mut td = TileDecoder::new_encoder(f, enc, t / ti.cols, t % ti.cols);
            td.decode_tile()?;
            let saved = if !td.f.hdr.disable_frame_end_update_cdf && t == ti.context_update_tile_id
            {
                Some(td.cdf.clone())
            } else {
                None
            };
            let log = std::mem::take(&mut td.enc.as_mut().expect("encode mode").rdo.log);
            let crate::symbol::Coder::Enc(e) = td.sd else {
                unreachable!("encode mode")
            };
            Ok((e.finish(), log, saved))
        };
        if replay.is_none() && self.cfg.tools.wavefront {
            // The search alone, in a wavefront per tile; the caller codes
            // the frame again with its decisions.
            let mut logs = Vec::with_capacity(num_tiles);
            let threads = self.cfg.threads.max(1);
            for t in 0..num_tiles {
                let make = || self.enc_ctx(p, src, stride);
                logs.push(wavefront::search_tile(
                    &mut f,
                    t / ti.cols,
                    t % ti.cols,
                    threads,
                    &make,
                )?);
            }
            return Ok(Coded {
                f,
                header,
                tiles: Vec::new(),
                logs,
            });
        }
        let threads = self.cfg.threads.max(1).min(num_tiles);
        let outs: Vec<Result<TileOut>> = if threads > 1 {
            // Tiles are independent: each is coded into a frame state of its
            // own, then copied in.
            let nums: Vec<usize> = (0..num_tiles).collect();
            crate::decoder::tiles_in_parallel(&mut f, &nums, threads, code_tile)
        } else {
            (0..num_tiles).map(|t| code_tile(&mut f, t)).collect()
        };
        let mut tiles = Vec::with_capacity(num_tiles);
        let mut logs = Vec::with_capacity(num_tiles);
        for o in outs {
            let (bytes, log, saved) = o?;
            if saved.is_some() {
                f.saved_cdfs = saved;
            }
            tiles.push(bytes);
            logs.push(log);
        }
        Ok(Coded {
            f,
            header,
            tiles,
            logs,
        })
    }

    fn enc_ctx(
        &self,
        p: &FrameParams,
        src: &Arc<Vec<Vec<u16>>>,
        stride: &[usize],
    ) -> Box<tile::EncCtx> {
        let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
        let qstep =
            AC_QLOOKUP[bdi][p.qidx as usize] as f64 / (1 << (self.cfg.bit_depth - 8)) as f64;
        let scale = (1 << (self.cfg.bit_depth - 8)) as f64;
        Box::new(tile::EncCtx {
            src: src.clone(),
            stride: stride.to_vec(),
            coefs: Box::new([0; 1024]),
            lambda: 0.4 * qstep * scale,
            rd_lambda: rd_lambda(&self.cfg, p.qidx),
            refs: p.search_refs.clone(),
            search_range: self.cfg.search_range,
            tools: Tools {
                compound: self.cfg.tools.compound || p.reference_select,
                ..self.cfg.tools
            },
            rdo: Default::default(),
            res: Vec::new(),
            fc: Vec::new(),
            cdef_table: None,
            lr_plan: None,
            palette_map: Box::new([[0; 64]; 64]),
            me_hints: Vec::new(),
            aq: p.aq.clone(),
            ibc: p.ibc.clone(),
            probing: false,
        })
    }

    /// `uncompressed_header()` for the encoder's frames, byte aligned.
    fn write_frame_header(&self, p: &FrameParams) -> Vec<u8> {
        let seq = &self.seq;
        let key = p.key;
        let mut w = BitWriter::new();
        w.flag(false); // show_existing_frame
        w.f(2, if key { KEY_FRAME } else { INTER_FRAME });
        w.flag(p.show); // show_frame
        if !p.show {
            w.flag(true); // showable_frame
        }
        if !key {
            w.flag(false); // error_resilient_mode
        }
        w.flag(false); // disable_cdf_update
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            w.flag(p.screen_content); // allow_screen_content_tools
            if p.screen_content {
                w.flag(false); // force_integer_mv
            }
        }
        w.flag(false); // frame_size_override_flag
        w.f(
            seq.order_hint_bits,
            (p.order_hint & ((1 << seq.order_hint_bits) - 1)) as u32,
        );
        if !key {
            w.f(3, 0); // primary_ref_frame: LAST_FRAME
            w.f(8, p.refresh as u32); // refresh_frame_flags
            w.flag(false); // frame_refs_short_signaling
            for &s in &p.ref_slots {
                w.f(3, s as u32); // ref_frame_idx
            }
            // frame_size_with_refs() is not used (no override): frame_size()
            // has nothing to write, then render_size().
            w.flag(false); // render_and_frame_size_different
            w.flag(false); // allow_high_precision_mv
            let tools = &self.cfg.tools;
            w.flag(tools.switchable_filter); // is_filter_switchable
            if !tools.switchable_filter {
                w.f(2, EIGHTTAP);
            }
            w.flag(tools.obmc || tools.warped_motion); // is_motion_mode_switchable
            if seq.enable_ref_frame_mvs {
                w.flag(true); // use_ref_frame_mvs
            }
        } else {
            w.flag(false); // render_and_frame_size_different
            if p.screen_content {
                w.flag(p.intrabc); // allow_intrabc
            }
        }
        w.flag(false); // disable_frame_end_update_cdf
        // tile_info(): uniform spacing, the tile columns asked for (at
        // least as many as the frame needs).
        let mi_cols = 2 * ((self.cfg.width + 7) >> 3);
        let mi_rows = 2 * ((self.cfg.height + 7) >> 3);
        let sb_cols = (mi_cols + 15) >> 4;
        let sb_rows = (mi_rows + 15) >> 4;
        let tl = crate::header::tile_log2;
        let min_cols = tl(MAX_TILE_WIDTH >> 6, sb_cols);
        let max_cols = tl(1, sb_cols.min(MAX_TILE_COLS));
        let max_rows = tl(1, sb_rows.min(MAX_TILE_ROWS));
        let cols_log2 = p.tile_cols_log2.clamp(min_cols, max_cols);
        w.flag(true); // uniform_tile_spacing_flag
        for _ in min_cols..cols_log2 {
            w.flag(true); // increment_tile_cols_log2
        }
        if cols_log2 < max_cols {
            w.flag(false);
        }
        let min_tiles = min_cols.max(tl(MAX_TILE_AREA >> 12, sb_rows * sb_cols));
        let min_rows = min_tiles.saturating_sub(cols_log2);
        if min_rows < max_rows {
            w.flag(false); // increment_tile_rows_log2
        }
        if cols_log2 + min_rows > 0 {
            w.f(cols_log2 + min_rows, 0); // context_update_tile_id
            w.f(2, 3); // tile_size_bytes_minus_1
        }
        // quantization_params()
        w.f(8, p.qidx);
        let y_dc = luma_dc_delta(p.qidx, self.cfg.bit_depth);
        w.flag(y_dc != 0); // DeltaQYDc
        if y_dc != 0 {
            w.f(7, (y_dc & 0x7F) as u32); // su(1 + 6)
        }
        let chroma = self.seq.color.num_planes > 1;
        if chroma {
            w.flag(false); // DeltaQUDc
            w.flag(false); // DeltaQUAc
        }
        w.flag(false); // using_qmatrix
        w.flag(false); // segmentation_enabled
        if p.qidx > 0 {
            w.flag(p.aq.is_some()); // delta_q_present
            if p.aq.is_some() {
                w.f(2, 0); // delta_q_res
                if !p.intrabc {
                    w.flag(false); // delta_lf_present
                }
            }
        }
        // loop_filter_params(), unless the frame is lossless.
        if p.filters() {
            w.f(6, p.lf[0]);
            w.f(6, p.lf[1]);
            if chroma && (p.lf[0] != 0 || p.lf[1] != 0) {
                w.f(6, p.lf[2]);
                w.f(6, p.lf[3]);
            }
            w.f(3, 0); // loop_filter_sharpness
            w.flag(false); // loop_filter_delta_enabled
        }
        // cdef_params()
        if let Some(c) = &p.cdef {
            w.f(2, c.damping_minus_3);
            w.f(2, c.bits);
            for i in 0..1usize << c.bits {
                w.f(4, c.y[i].0);
                w.f(2, c.y[i].1);
                if chroma {
                    w.f(4, c.uv[i].0);
                    w.f(2, c.uv[i].1);
                }
            }
        }
        // lr_params()
        if let Some(types) = &p.lr {
            let mut uses_lr = false;
            let mut uses_chroma_lr = false;
            for (plane, &t) in types.iter().enumerate().take(self.seq.color.num_planes) {
                // The coded lr_type (the inverse of Remap_Lr_Type).
                w.f(
                    2,
                    match t {
                        RESTORE_NONE => 0,
                        RESTORE_SWITCHABLE => 1,
                        RESTORE_WIENER => 2,
                        _ => 3,
                    },
                );
                if t != RESTORE_NONE {
                    uses_lr = true;
                    uses_chroma_lr |= plane > 0;
                }
            }
            if uses_lr {
                // Units of 64 samples, 128 above 720p.
                let big = self.cfg.width as u64 * self.cfg.height as u64 > 1280 * 720;
                w.flag(big); // lr_unit_shift
                if big {
                    w.flag(false); // lr_unit_extra_shift
                }
                let c = &self.seq.color;
                if uses_chroma_lr && c.subsampling_x != 0 && c.subsampling_y != 0 {
                    w.flag(false); // lr_uv_shift
                }
            }
        }
        // read_tx_mode(): TX_MODE_SELECT when transform sizes are searched
        // (a lossless frame has ONLY_4X4, not coded).
        if p.qidx > 0 {
            w.flag(p.tx_select);
        }
        if !key {
            w.flag(p.reference_select); // reference_select
            if p.reference_select && self.skip_mode_allowed(p) {
                w.flag(true); // skip_mode_present
            }
        }
        if !key && seq.enable_warped_motion {
            w.flag(true); // allow_warped_motion
        }
        w.flag(p.reduced_tx_set); // reduced_tx_set
        if !key {
            gm::write_params(&mut w, &p.gm, &p.prev_gm);
        }
        // film_grain_params(): every frame is shown or showable.
        if seq.film_grain_params_present {
            let c = &seq.color;
            grain::write_params(
                &mut w,
                &p.grain,
                !key,
                c.mono_chrome,
                c.subsampling_x == 1 && c.subsampling_y == 1,
            );
        }
        w.byte_align();
        w.finish()
    }
}

impl Encoder {
    /// The references of the group frame of display index `hint`, and the
    /// slot it replaces: `refresh_frame_flags`, `ref_frame_idx` and the
    /// references searched.
    ///
    /// The frames held before it are LAST, LAST2 and LAST3, nearest first,
    /// and GOLDEN the anchor (the last key or alt-ref frame shown) unless
    /// it is one of those; the frames held after it (the hidden frames of
    /// the layers below) BWDREF, nearest, ALTREF2, and ALTREF, the
    /// farthest. The search takes the nearest on each side first.
    fn group_refs(&self, hint: u64) -> (u8, [usize; REFS_PER_FRAME], Vec<i32>) {
        // One slot for each frame held.
        let mut frames: Vec<(u64, usize)> = Vec::new();
        for s in 0..NUM_REF_FRAMES {
            if !frames.iter().any(|f| f.0 == self.slot_frame[s]) {
                frames.push((self.slot_frame[s], s));
            }
        }
        let mut past: Vec<(u64, usize)> = frames.iter().copied().filter(|f| f.0 < hint).collect();
        past.sort_by_key(|f| std::cmp::Reverse(f.0));
        let mut future: Vec<(u64, usize)> = frames.iter().copied().filter(|f| f.0 > hint).collect();
        future.sort_by_key(|f| f.0);
        let last = past.first().expect("a frame shown before").1;
        let last2 = past.get(1).map_or(last, |f| f.1);
        let last3 = past.get(2).map_or(last2, |f| f.1);
        let golden = match past.iter().find(|f| f.0 == self.anchor) {
            Some(a) if ![last, last2, last3].contains(&a.1) => a.1,
            _ => past.get(3).map_or(last3, |f| f.1),
        };
        let bwd = future.first().map_or(golden, |f| f.1);
        let alt2 = future.get(1).map_or(bwd, |f| f.1);
        let alt = future.last().map_or(bwd, |f| f.1);
        let ref_slots = [last, last2, last3, golden, bwd, alt2, alt];
        let max_refs = group_max_refs(&self.cfg.tools);
        let mut search_refs = Vec::new();
        let mut used = Vec::new();
        for rf in [
            LAST_FRAME,
            BWDREF_FRAME,
            ALTREF_FRAME,
            LAST2_FRAME,
            GOLDEN_FRAME,
            ALTREF2_FRAME,
            LAST3_FRAME,
        ] {
            let s = ref_slots[(rf - LAST_FRAME) as usize];
            if search_refs.len() < max_refs && !used.contains(&s) {
                used.push(s);
                search_refs.push(rf);
            }
        }
        // The slot replaced: a second copy of a frame, else the oldest
        // frame that is neither waiting to be shown nor the anchor.
        let spare = (0..NUM_REF_FRAMES).find(|&s| !frames.iter().any(|f| f.1 == s));
        let target = spare.or_else(|| {
            frames
                .iter()
                .filter(|f| !self.pending.contains(&f.0) && f.0 != self.anchor)
                .min_by_key(|f| f.0)
                .map(|f| f.1)
        });
        let refresh = 1 << target.expect("a slot to refresh");
        (refresh, ref_slots, search_refs)
    }
}

impl Encoder {
    /// Whether the header of `p` lets a frame signal skip mode
    /// (`skipModeAllowed`, 5.9.22): a forward reference and either a
    /// backward one or a second, older forward one.
    fn skip_mode_allowed(&self, p: &FrameParams) -> bool {
        let seq = &self.seq;
        let hint = (p.order_hint & ((1 << seq.order_hint_bits) - 1)) as u32;
        let dist = |a: u32, b: u32| crate::header::get_relative_dist(seq, a, b);
        let hints: Vec<u32> = p
            .ref_slots
            .iter()
            .map(|&s| self.dec.ref_state.order_hint[s])
            .collect();
        let mut forward: Option<u32> = None;
        let mut backward = false;
        for &h in &hints {
            if dist(h, hint) < 0 {
                if forward.is_none_or(|f| dist(h, f) > 0) {
                    forward = Some(h);
                }
            } else if dist(h, hint) > 0 {
                backward = true;
            }
        }
        let Some(fwd) = forward else {
            return false;
        };
        backward || hints.iter().any(|&h| dist(h, fwd) < 0)
    }
}

/// Whether a frame looks like screen content: a good share of its 16x16
/// blocks have at most eight distinct luma values and sharp edges.
fn looks_like_screen_content(frame: &Frame) -> bool {
    let pl = frame.planes[0];
    let (w, h) = (pl.width, pl.height);
    let shift = frame.bit_depth - 8;
    let mut blocks = 0u32;
    let mut screen = 0u32;
    let mut vals = Vec::with_capacity(256);
    for by in (0..h.saturating_sub(15)).step_by(32) {
        for bx in (0..w.saturating_sub(15)).step_by(32) {
            vals.clear();
            for y in by..by + 16 {
                for x in bx..bx + 16 {
                    vals.push(frame.sample(0, x, y));
                }
            }
            vals.sort_unstable();
            let range = (vals[vals.len() - 1] - vals[0]) >> shift;
            vals.dedup();
            blocks += 1;
            if vals.len() <= 8 && range >= 48 {
                screen += 1;
            }
        }
    }
    blocks > 0 && screen * 10 >= blocks
}

/// A frame's quantiser index when the plan's (or the configured one) is
/// `q`: its step scaled by a power of two for its role. Key frames and
/// the alt-ref frames, references for every frame after them, are coded
/// finer; the frames of the top layer of a group, references for none or
/// few, coarser; the layers between, between.
fn frame_qidx(
    q: u32,
    role: Role,
    golden: bool,
    groups: bool,
    planned: bool,
    bit_depth: u32,
) -> u32 {
    let log2 = match role {
        Role::Key if groups => KEY_Q,
        // (Rate control plans key frames finer by its own offset.)
        Role::Key if planned => return q.saturating_sub(golden_boost(q)).max(1),
        Role::Key => KEY_Q_LOW_DELAY,
        Role::Inter if golden => return q.saturating_sub(golden_boost(q)).max(1),
        Role::Inter => return q,
        Role::Group { layer, depth, .. } => {
            ALTREF_Q + (TOP_Q - ALTREF_Q) * layer as f64 / depth.max(1) as f64
        }
    };
    scaled_qidx(q, log2, bit_depth)
}

/// `DeltaQYDc`: luma's DC coded coarser than its AC, by `LUMA_DC_Q`
/// octaves of quantiser step (0 for a lossless frame). A block's DC is
/// mostly predicted well and its error is spread over every sample, so a
/// coarser step costs little; and the bits it saves are taken where they
/// count more.
fn luma_dc_delta(qidx: u32, bit_depth: u32) -> i32 {
    if qidx == 0 || LUMA_DC_Q == 0.0 {
        return 0;
    }
    (scaled_qidx(qidx, LUMA_DC_Q, bit_depth) as i32 - qidx as i32).clamp(-64, 63)
}

/// See [`luma_dc_delta`]: 0.5 measured best of 0.3, 0.5 and 0.8 at speed
/// 6 (−0.2 % BD-rate PSNR-Y and PSNR-YUV; 0.3 −0.2 % and −0.1 %, 0.8 −0.0 %
/// and −0.2 %; finer, −0.3, +0.3 % and +0.4 %).
const LUMA_DC_Q: f64 = 0.5;

/// The quantiser index (1 to 255) whose step is nearest `2^log2` times the
/// step of `q`.
fn scaled_qidx(q: u32, log2: f64, bit_depth: u32) -> u32 {
    // (A plan past 255: see `rc::MAX_GROUP_Q`.)
    let log2 = log2 + q.saturating_sub(255) as f64 / 32.0;
    let q = q.min(255);
    if log2 == 0.0 {
        return q;
    }
    let t = &AC_QLOOKUP[((bit_depth - 8) >> 1) as usize];
    let goal = (t[q.clamp(1, 255) as usize] as f64 * log2.exp2()).ln();
    (1..=255u32)
        .min_by(|&a, &b| {
            let da = ((t[a as usize] as f64).ln() - goal).abs();
            let db = ((t[b as usize] as f64).ln() - goal).abs();
            da.total_cmp(&db)
        })
        .unwrap_or(q)
}

/// The quantiser step of a key frame against the configured one, as a
/// power of two, with alt-ref groups (with rav1e's figure, -1.2 measured
/// best of -0.9, -1.2 and -1.5 on the test clips) and without (-1.0 of
/// -0.6 and -1.0: there it is the reference of every frame up to the
/// next golden one).
const KEY_Q: f64 = -1.2;
const KEY_Q_LOW_DELAY: f64 = -1.0;
/// The same for a group's alt-ref frame (layer 0) and its top layer; the
/// layers between are spaced evenly. (-0.6 of -0.4, -0.6 and -0.8; 0.5 of
/// 0.5, 0.7 and 0.9.)
const ALTREF_Q: f64 = -0.6;
const TOP_Q: f64 = 0.5;

/// How many references a group frame searches: four with several
/// references (the nearest on each side, the farthest after, the second
/// nearest before; three cost +0.6 % BD-rate, five gained 0.3 % for the
/// time), else the nearest on each side.
fn group_max_refs(tools: &Tools) -> usize {
    if tools.multi_ref { 4 } else { 2 }
}

/// How often a boosted golden frame is coded with several references.
const GOLDEN_INTERVAL: u64 = 16;

/// How much finer the golden frame's quantiser is.
fn golden_boost(qidx: u32) -> u32 {
    (qidx / 6).clamp(4, 24)
}

/// The rate-distortion Lagrange multiplier at quantiser `qidx`.
fn rd_lambda(cfg: &Config, qidx: u32) -> f64 {
    let bdi = ((cfg.bit_depth - 8) >> 1) as usize;
    let qstep = AC_QLOOKUP[bdi][qidx as usize] as f64 / (1 << (cfg.bit_depth - 8)) as f64;
    let scale = (1 << (cfg.bit_depth - 8)) as f64;
    let step = qstep * scale / 8.0;
    rd_lambda_factor() * step * step
}

/// The loop filter levels with the least error: scaled versions of the
/// starting ones, each plane's chosen on its own error. `f.cur` is left
/// unfiltered.
fn search_lf(
    f: &mut crate::decoder::FrameCtx,
    src: &[Vec<u16>],
    stride: &[usize],
    start: [u32; 4],
    threads: usize,
) -> [u32; 4] {
    let saved = f.cur.clone();
    let mut best = [(u64::MAX, 0u32); 3];
    let planes = f.num_planes;
    for scale in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5] {
        let lv = |l: u32| ((l as f64 * scale).round() as u32).min(63);
        let levels = [lv(start[0]), lv(start[1]), lv(start[2]), lv(start[3])];
        f.hdr.loop_filter_level = levels.map(|l| l as i32);
        if levels[0] != 0 || levels[1] != 0 {
            crate::decoder::postfilter::loop_filter_threads(f, threads);
        }
        for (p, b) in best.iter_mut().enumerate().take(planes) {
            let e = plane_sse(f, src, stride, p);
            if e < b.0 {
                *b = (e, if p == 0 { levels[0] } else { levels[1 + p] });
            }
        }
        f.cur.planes.clone_from(&saved.planes);
    }
    // Chroma is filtered only when luma is.
    if best[0].1 == 0 {
        return [0; 4];
    }
    [best[0].1, best[0].1, best[1].1, best[2].1]
}

/// Squared error of plane `p` of `f.cur` against the source, inside the
/// frame.
fn plane_sse(f: &crate::decoder::FrameCtx, src: &[Vec<u16>], stride: &[usize], p: usize) -> u64 {
    let (ssx, ssy) = f.plane_ss(p);
    let w = (f.hdr.frame_width as usize + ssx) >> ssx;
    let h = (f.hdr.frame_height as usize + ssy) >> ssy;
    let pl = &f.cur.planes[p];
    let mut s = 0u64;
    for y in 0..h {
        let a = &src[p][y * stride[p]..y * stride[p] + w];
        let b = &pl.data[y * pl.stride..y * pl.stride + w];
        s += crate::dsp::enc::sse_row(a, b);
    }
    s
}

/// Whether the encoder can write this colour description for frames of
/// `chroma`: code points in range, and the identity matrix (RGB, and the
/// sRGB triple that implies it) only with 4:4:4, as 6.4.2 requires.
fn check_color(c: &ColorInfo, chroma: ChromaFormat) -> Result<()> {
    if c.color_primaries > 255 || c.transfer_characteristics > 255 || c.matrix_coefficients > 255 {
        return Err(Error::invalid("colour code points are 8-bit values"));
    }
    if c.chroma_sample_position > 3 {
        return Err(Error::invalid("chroma_sample_position is 0 to 3"));
    }
    if c.matrix_coefficients == MC_IDENTITY && chroma != ChromaFormat::Yuv444 {
        return Err(Error::invalid("the identity matrix (RGB) needs 4:4:4"));
    }
    if c.color_primaries == CP_BT_709
        && c.transfer_characteristics == TC_SRGB
        && c.matrix_coefficients == MC_IDENTITY
        && !c.full_range
    {
        // The sRGB triple implies full range (`color_range` is not coded).
        return Err(Error::invalid(
            "sRGB (BT.709, sRGB, identity) is full range",
        ));
    }
    Ok(())
}

/// The rate-distortion Lagrange multiplier as a multiple of the squared
/// quantiser step (`AV1_RD_LAMBDA` overrides it, for tuning).
fn rd_lambda_factor() -> f64 {
    std::env::var("AV1_RD_LAMBDA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1)
}
