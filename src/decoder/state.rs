//! Frame-level decoder state: sample buffers, the per-4x4 mode info arrays
//! of the specification (`MiSizes`, `YModes`, `RefFrames`, `Mvs`, ...),
//! and what a reference slot keeps (7.20).

use std::sync::OnceLock;

use crate::cdf::CdfContext;
use crate::decoder::shared::Shared;

/// One plane of samples, 16 bits each whatever the bit depth.
///
/// `data` is the whole plane, or (while the rows of a frame are worked on
/// by several threads) the rows of it lent to one of them: either way a
/// sample's index is `y * stride + x`.
#[derive(Clone, Debug, Default)]
pub(crate) struct PlaneBuf {
    pub(crate) data: Shared<u16>,
    pub(crate) stride: usize,
    /// Allocated width and height.
    pub(crate) w: usize,
    pub(crate) h: usize,
}

impl PlaneBuf {
    pub(crate) fn new(w: usize, h: usize, fill: u16) -> Self {
        PlaneBuf {
            data: vec![fill; w * h].into(),
            stride: w,
            w,
            h,
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.stride + x]
    }

    /// A sample of the rows this plane writes: what the reconstruction of
    /// a block reads back of the frame it is writing (`get` reads the rows
    /// before the ones a plane lent by rows may read; see `Shared`).
    #[inline(always)]
    pub(crate) fn mine(&self, x: usize, y: usize) -> u16 {
        *self.data.mine(y * self.stride + x)
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, x: usize, y: usize, v: u16) {
        self.data[y * self.stride + x] = v;
    }

    pub(crate) fn row(&self, y: usize) -> &[u16] {
        &self.data[y * self.stride..y * self.stride + self.w]
    }
}

/// A frame's planes (one for monochrome, three otherwise).
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameBuf {
    pub(crate) planes: Vec<PlaneBuf>,
}

/// A motion vector: `[row, col]` in eighth samples.
pub(crate) type Mv = [i32; 2];

/// The mode info of one 4x4 luma block, as the decoding process stores it
/// for the whole frame (5.11.5 and the arrays it writes).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Mi {
    pub(crate) mi_size: u8,
    pub(crate) y_mode: u8,
    pub(crate) uv_mode: u8,
    pub(crate) ref_frame: [i8; 2],
    pub(crate) mv: [Mv; 2],
    pub(crate) comp_group_idx: u8,
    pub(crate) compound_idx: u8,
    pub(crate) interp_filter: [u8; 2],
    pub(crate) is_inter: bool,
    pub(crate) skip_mode: bool,
    pub(crate) skip: bool,
    pub(crate) tx_size: u8,
    pub(crate) inter_tx_size: u8,
    pub(crate) segment_id: u8,
    pub(crate) palette_size: [u8; 2],
    pub(crate) delta_lf: [i8; 4],
    /// `RefFrames[ row ][ col ]` has been written for this frame.
    pub(crate) written: bool,
}

/// What a reference slot holds beyond the header state (7.20): the
/// decoded frame and the side data later frames read from it.
///
/// A frame is in a slot as soon as its header is parsed, when the decoder
/// has several frames in flight (`docs/threading.md`): what its header
/// says is there from the start, what its parsing leaves (`syntax`) once
/// its last tile is parsed, and its samples (`frame`) once its
/// post-filters are done. A frame that reads one or the other is not
/// started before it is there.
pub(crate) struct RefData {
    pub(crate) frame_type: u32,
    pub(crate) upscaled_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) render_width: u32,
    pub(crate) render_height: u32,
    pub(crate) mi_rows: usize,
    pub(crate) mi_cols: usize,
    pub(crate) bit_depth: u32,
    pub(crate) subsampling_x: u32,
    pub(crate) subsampling_y: u32,
    pub(crate) order_hint: u32,
    /// `SavedOrderHints[ i ][ ref ]`.
    pub(crate) saved_order_hints: [u32; 8],
    pub(crate) syntax: OnceLock<RefSyntax>,
    pub(crate) frame: OnceLock<FrameBuf>,
}

/// What parsing a frame leaves for the frames that refer to it.
pub(crate) struct RefSyntax {
    /// `SavedRefFrames` on the 8x8 grid (the odd 4x4 positions are the
    /// only ones read).
    pub(crate) saved_ref_frames: Vec<i8>,
    /// `SavedMvs` on the 8x8 grid.
    pub(crate) saved_mvs: Vec<Mv>,
    pub(crate) saved_segment_ids: Vec<u8>,
    pub(crate) cdfs: CdfContext,
}

impl RefData {
    /// The decoded frame.
    ///
    /// # Panics
    ///
    /// If it is not complete: a frame was started before its reference.
    #[inline]
    pub(crate) fn frame(&self) -> &FrameBuf {
        self.frame
            .get()
            .expect("a reference frame's samples, before they are complete")
    }

    /// What the frame's parsing left.
    ///
    /// # Panics
    ///
    /// If it is not parsed: a frame was started before its reference.
    #[inline]
    pub(crate) fn syntax(&self) -> &RefSyntax {
        self.syntax
            .get()
            .expect("a reference frame's saved state, before it is parsed")
    }
}
