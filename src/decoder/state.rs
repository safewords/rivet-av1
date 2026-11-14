//! Frame-level decoder state: sample buffers, the per-4x4 mode info arrays
//! of the specification (`MiSizes`, `YModes`, `RefFrames`, `Mvs`, ...),
//! and what a reference slot keeps (7.20).

use std::sync::{Arc, Mutex, OnceLock};

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

/// Sample planes put aside for the frames after: what a frame's planes
/// were once nothing reads them any more (a reference frame no slot holds,
/// `CurrFrame` and `CdefFrame` once the post-filters are through them).
/// The copies the post-filters start from (`CdefFrame` from `CurrFrame`,
/// `LrFrame` from `CdefFrame`) are made into them rather than into new
/// allocations. Planes of a size not seen lately make room for others.
///
/// A plane that starts as zeros (`CurrFrame`) is allocated afresh: zeroing
/// a recycled plane cost more than the system's zeroed pages (4 % to 7 %
/// of the decode, one thread or four), where recycling the copies'
/// planes is neither faster nor slower.
#[derive(Default)]
pub(crate) struct Pool {
    planes: Mutex<Vec<Vec<u16>>>,
}

/// How many planes a pool keeps: four frames' worth, at three planes a
/// frame (each frame takes two copies and gives back two planes besides
/// the reference it may push out of a slot).
const POOL_PLANES: usize = 3 * 4;

impl Pool {
    /// A plane of `w` by `h` samples, each `fill`: a new allocation (see
    /// above).
    pub(crate) fn plane(&self, w: usize, h: usize, fill: u16) -> PlaneBuf {
        let data = vec![fill; w * h];
        PlaneBuf {
            data: data.into(),
            stride: w,
            w,
            h,
        }
    }

    /// A copy of `src` (a whole plane).
    pub(crate) fn copy(&self, src: &PlaneBuf) -> PlaneBuf {
        let data = match self.take(src.data.len()) {
            Some(mut v) => {
                v.copy_from_slice(src.data.as_slice());
                v
            }
            None => src.data.as_slice().to_vec(),
        };
        PlaneBuf {
            data: data.into(),
            stride: src.stride,
            w: src.w,
            h: src.h,
        }
    }

    /// A copy of every plane of `src`.
    pub(crate) fn copy_frame(&self, src: &FrameBuf) -> FrameBuf {
        FrameBuf {
            planes: src.planes.iter().map(|p| self.copy(p)).collect(),
        }
    }

    fn take(&self, len: usize) -> Option<Vec<u16>> {
        let mut planes = self.planes.lock().unwrap_or_else(|e| e.into_inner());
        // The most recently given back first: the likeliest to be in cache.
        let i = planes.iter().rposition(|v| v.len() == len)?;
        Some(planes.swap_remove(i))
    }

    /// Takes back the planes of `frame` that nothing else holds.
    pub(crate) fn give_back(&self, frame: FrameBuf) {
        let vecs: Vec<Vec<u16>> = frame
            .planes
            .into_iter()
            .filter_map(|p| p.data.into_vec())
            .collect();
        if vecs.is_empty() {
            return;
        }
        let mut planes = self.planes.lock().unwrap_or_else(|e| e.into_inner());
        planes.extend(vecs);
        if planes.len() > POOL_PLANES {
            let excess = planes.len() - POOL_PLANES;
            planes.drain(..excess);
        }
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
    /// Where the frame's planes go when no slot holds it any more.
    pub(crate) pool: Option<Arc<Pool>>,
}

impl Drop for RefData {
    fn drop(&mut self) {
        if let (Some(pool), Some(frame)) = (&self.pool, self.frame.take()) {
            pool.give_back(frame);
        }
    }
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
