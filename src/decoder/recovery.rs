//! Recovery from errors: what a decoder does after a frame that does not
//! decode, so that it keeps taking data, never shows a frame that is not
//! exactly the stream's as if it were, and is back to exact output at the
//! first frame that does not depend on what was lost.
//!
//! The specification's own notion is Annex C's: a frame is *processable*
//! when everything but its samples (its syntax, its size, everything the
//! reference update of 7.20 writes but `FrameStore`) is certainly right.
//! The decoder keeps two marks per reference slot to follow that:
//!
//! - **damaged**: the slot's samples, or what its parsing saved for later
//!   frames (motion vectors, segment map, CDFs), may not be the stream's.
//!   A frame that failed is put in its slots with what of it did decode
//!   and the rest filled in ([`conceal`]); a frame predicted from a
//!   damaged slot is damaged too. Header state (sizes, order hints, the
//!   loop filter deltas a slot passes on) is right.
//! - **unknown**: not even the slot's header state is the stream's: the
//!   frame header that would have set it did not parse, or a whole unit
//!   was lost. Unknown slots are damaged, and a frame that derives its
//!   references from every slot's order hint (`frame_refs_short_signaling`,
//!   7.8) is damaged while any slot is unknown, since it may have chosen
//!   the wrong ones.
//!
//! A frame is damaged when it refers to a damaged slot for anything:
//! `ref_frame_idx` for an inter frame (which covers `primary_ref_frame`,
//! the motion field and prediction), `film_grain_params_ref_idx` when its
//! grain parameters are a slot's that is unknown; or, with frame ids,
//! when a reference is not the frame the header expects (the frame that
//! was there was lost: `conformance.rs`). Intra frames refer to nothing,
//! so a key frame, or an intra-only frame for the slots it refreshes, is
//! where exact output starts again (the random access points of 7.5 and
//! Annex C), and a frame refreshing every slot (a shown key frame) clears
//! every mark.
//!
//! A damaged frame is decoded like any other, from whatever its references
//! hold, so that the slots it refreshes hold something of the right size
//! for the frames after it; what changes is its output. By default it is
//! withheld. With concealment on ([`Decoder::set_concealment`]) it is
//! shown with [`Frame::concealed`](crate::Frame) set, as is an inter frame
//! whose tiles failed, its undecoded superblocks filled from its primary
//! reference.

use super::state::{FrameBuf, Mi};
use super::{Decoder, FrameCtx};
use crate::consts::*;
use crate::header::FrameHeader;
use crate::obu::SequenceHeader;
use crate::{Error, Result};

/// The marks of the reference slots (see above), and what else of the
/// stream is not known.
#[derive(Clone, Copy, Default)]
pub(crate) struct Recovery {
    pub(crate) damaged: [bool; NUM_REF_FRAMES],
    pub(crate) unknown: [bool; NUM_REF_FRAMES],
    /// A sequence header did not parse: until one does, the frames after
    /// it may belong to a sequence the decoder does not know.
    pub(crate) seq_lost: bool,
}

impl Recovery {
    /// Nothing about the slots is known any more: data was lost whose
    /// frames, if any, could have refreshed any of them.
    pub(crate) fn lose_all(&mut self) {
        self.damaged = [true; NUM_REF_FRAMES];
        self.unknown = [true; NUM_REF_FRAMES];
    }

    /// The slots a frame that failed before its frame state existed would
    /// have refreshed: their contents and header state are not the
    /// stream's.
    pub(crate) fn lose_slots(&mut self, refresh_frame_flags: u32) {
        for i in 0..NUM_REF_FRAMES {
            if (refresh_frame_flags >> i) & 1 != 0 {
                self.damaged[i] = true;
                self.unknown[i] = true;
            }
        }
    }

    /// Whether a frame with header `h` is damaged by its references.
    pub(crate) fn damages(&self, h: &FrameHeader) -> bool {
        if self.seq_lost {
            return true;
        }
        if let Some(g) = h.film_grain_params_ref_idx
            && self.unknown[g]
        {
            return true;
        }
        if h.frame_is_intra {
            return false;
        }
        h.ref_frame_idx.iter().any(|&i| self.damaged[i])
            || (h.frame_refs_short_signaling && self.unknown.iter().any(|&u| u))
    }
}

impl Decoder {
    /// Makes an inter frame whose tiles fail come out with what did not
    /// decode filled in from its primary reference, flagged
    /// [`Frame::concealed`](crate::Frame), and the frames predicted from
    /// such a frame come out flagged too, until a frame that does not
    /// depend on it (a key frame, at the latest). Off by default: those
    /// frames are not output at all, and the unit whose frame failed
    /// returns its error.
    ///
    /// A frame that fails is put in the reference slots it refreshes
    /// either way (concealed as above, or grey where an intra frame has
    /// nothing to take from), so that decoding carries on.
    pub fn set_concealment(&mut self, conceal: bool) {
        self.conceal = conceal;
    }

    /// Tells the decoder that data was lost between the units it was given
    /// and the next one (a packet a transport dropped, a cut in a file):
    /// from the next unit on, frames are treated as damaged until each
    /// slot they refer to has been refreshed by a frame decoded after the
    /// loss (a key frame refreshes them all). Without this, a lost unit is
    /// noticed only when what comes after it does not parse, or, in a
    /// stream with frame ids, by them.
    pub fn signal_loss(&mut self) {
        self.loss_pending = true;
    }

    /// Whether some reference slot holds a frame that is not exactly the
    /// stream's (after an error, a loss or a damaged sequence header): the
    /// frames that refer to it are withheld, or with concealment flagged,
    /// until a frame that does not depend on it refreshes it.
    pub fn is_recovering(&self) -> bool {
        self.recovery.seq_lost || self.recovery.damaged.iter().any(|&d| d)
    }

    /// A loss signalled before the unit about to be decoded: the frame in
    /// progress, if any, failed, and nothing about the slots is known.
    pub(crate) fn lose(&mut self) {
        if self.frame.is_some() || self.pending.is_some() {
            let _ = self.fail_in_progress(Error::bitstream("data lost"));
        }
        self.recovery.lose_all();
        self.seen_frame_header = false;
        self.discard = true;
    }

    /// Whether the frame of header `h` is damaged (`Recovery::damages`,
    /// or a lost reference its frame ids show), and, for a strict decoder,
    /// the frame id and level requirements it breaks.
    pub(crate) fn assess(&mut self, seq: &SequenceHeader, h: &FrameHeader) -> Result<bool> {
        let mut damaged = self.recovery.damages(h);
        let broken = super::conformance::frame_ids(seq, h, &self.ref_state);
        if self.strict
            && let Some(v) = broken.first()
        {
            return Err(Error::bitstream(v.rule.clone()));
        }
        damaged |= broken.iter().any(|v| v.lost_reference);
        if self.strict {
            if let Some(rule) = super::conformance::frame_level(seq, h) {
                return Err(Error::bitstream(rule));
            }
            if let Some(rule) = self.rates.count(seq, h) {
                return Err(Error::bitstream(rule));
            }
        }
        Ok(damaged)
    }

    /// Whether the frame a `show_existing_frame` header `h` shows is
    /// damaged, and, for a strict decoder, the frame id requirements of
    /// the header.
    pub(crate) fn assess_shown(&mut self, seq: &SequenceHeader, h: &FrameHeader) -> Result<bool> {
        let idx = h.frame_to_show_map_idx;
        let mut damaged = self.recovery.damaged[idx] || self.recovery.seq_lost;
        if let Some(v) = super::conformance::show_existing_ids(seq, h, &self.ref_state) {
            if self.strict {
                return Err(Error::bitstream(v.rule));
            }
            damaged |= v.lost_reference;
        }
        if self.strict
            && let Some(rule) = self.rates.count(seq, h)
        {
            return Err(Error::bitstream(rule));
        }
        Ok(damaged)
    }

    /// A frame whose header parsed failed before it had a frame state (a
    /// reference it may not have, a conformance requirement): the slots it
    /// would have refreshed are lost, and its tile groups are passed over.
    pub(crate) fn fail_before_state(&mut self, h: &FrameHeader) {
        self.recovery.lose_slots(h.refresh_frame_flags);
        // Its frame id is the previous one for the next frame (6.8.2).
        self.ref_state.current_frame_id = h.current_frame_id;
        self.ref_state.frame_id_seen = true;
        self.frame = None;
        self.pending = None;
        self.seen_frame_header = false;
        self.discard = true;
    }

    /// The frame in progress failed with `e` (a tile that did not decode,
    /// tile groups that never came): it is concealed and put in its slots,
    /// damaged, and shown if it is to be and the decoder conceals. The
    /// error comes back, except for an inter frame a concealing decoder
    /// showed in its place (whose flag says what happened), and for a
    /// frame known to be damaged before it failed: its references were
    /// not the stream's, so its parsing (which reads their CDFs, motion
    /// vectors and segment map) failing is no news, and the error that
    /// damaged them was reported where it happened. In flight, the error
    /// goes no further: the unit is decoded in place.
    pub(crate) fn fail_in_progress(&mut self, e: Error) -> Result<()> {
        if self.frame.is_none() && self.pending.is_none() {
            return Ok(());
        }
        if self.flying.is_some() {
            return Err(e);
        }
        if let Some((seq, hdr)) = self.pending.take() {
            self.frame = Some(super::build_frame(&self.refs, seq, hdr, &self.planes));
        }
        let mut f = self.frame.take().expect("a frame in progress");
        self.seen_frame_header = false;
        self.discard = true;
        let inter = !f.hdr.frame_is_intra;
        let known = self.cur_damaged;
        let lr = conceal(&mut f, self.threads);
        self.cur_damaged = true;
        self.wrapup_with(f, lr);
        if known || (self.conceal && inter) {
            Ok(())
        } else {
            Err(e)
        }
    }

    /// At a temporal delimiter, a new frame header or the end of a
    /// temporal unit: a frame still in progress has lost the tile groups
    /// it had to come, and fails. A frame header without any tile group
    /// that refreshes no slot is left in force: that is a large scale tile
    /// stream's camera frame header (7.3), which its tile lists use (and
    /// a lost frame that refreshes nothing leaves nothing to recover).
    pub(crate) fn end_frame_in_progress(&mut self) -> Result<()> {
        let incomplete = match (&self.frame, &self.pending) {
            (Some(f), _) => f.tile_num > 0 || f.hdr.refresh_frame_flags != 0,
            (None, Some((_, h))) => h.refresh_frame_flags != 0,
            (None, None) => false,
        };
        if !incomplete {
            return Ok(());
        }
        self.fail_in_progress(Error::bitstream(
            "a frame's tile groups end before its last tile",
        ))
    }

    /// Whether `payload`, a frame header OBU's, repeats the header of the
    /// frame in progress (`frame_header_copy()`, which must be identical
    /// to it bit for bit) rather than starting a frame of its own (the
    /// frame in progress having lost its last tile groups).
    pub(crate) fn is_header_copy(&self, payload: &[u8]) -> bool {
        let (bytes, bits) = (&self.header_bytes, self.header_bits);
        let whole = bits / 8;
        if payload.len() * 8 < bits || payload[..whole] != bytes[..whole] {
            return false;
        }
        let rest = bits % 8;
        rest == 0 || (payload[whole] ^ bytes[whole]) >> (8 - rest) == 0
    }

    /// Keeps the first `bits` bits of a frame header for
    /// [`Self::is_header_copy`].
    pub(crate) fn keep_header(&mut self, payload: &[u8], bits: usize) {
        self.header_bytes.clear();
        self.header_bytes
            .extend_from_slice(&payload[..bits.div_ceil(8).min(payload.len())]);
        self.header_bits = bits.min(8 * self.header_bytes.len());
    }
}

/// Finishes a frame whose tiles did not all decode: the superblocks that
/// did not (from the one each tile failed in, and every superblock of the
/// tiles never reached) are reset to blocks that predict nothing, the
/// post-filters run over the frame as they would, and then the samples of
/// those superblocks are filled from the frame's primary reference
/// (`LAST_FRAME`'s slot when it has none, as in error-resilient mode),
/// scaled to the frame's size if the reference is of another; an intra
/// frame has none and is filled with grey (Annex C's neutral value,
/// `1 << ( BitDepth - 1 )`). Returns `LrFrame`.
pub(crate) fn conceal(f: &mut FrameCtx, threads: usize) -> FrameBuf {
    let rects = undecoded(f);
    let ms = f.ms;
    for &(r0, r1, c0, c1) in &rects {
        let r1x = if r1 >= f.mi_rows { f.mi_rows + 32 } else { r1 };
        let c1x = if c1 >= f.mi_cols { ms } else { c1 };
        for r in r0..r1x {
            let row = r * ms;
            f.mi[row + c0..row + c1x].fill(Mi::default());
            f.tx_types[row + c0..row + c1x].fill(0);
            f.segment_ids[row + c0..row + c1x].fill(0);
        }
        for p in 0..f.num_planes {
            let (sx, sy) = f.plane_ss(p);
            for r in (r0 >> sy)..(r1x >> sy) {
                let row = r * ms;
                f.lf_tx_sizes[p][row + (c0 >> sx)..row + (c1x >> sx)].fill(0);
            }
        }
        let s = f.cdef_stride;
        for r in (r0 >> 4)..r1.div_ceil(16) {
            for c in (c0 >> 4)..c1.div_ceil(16) {
                if let Some(v) = f.cdef_idx.get_mut(r * s + c) {
                    *v = -1;
                }
            }
        }
    }
    let mut lr = super::filter_frame(f, threads);
    let source = if f.hdr.frame_is_intra {
        None
    } else {
        let k = if f.hdr.primary_ref_frame == PRIMARY_REF_NONE {
            0
        } else {
            f.hdr.primary_ref_frame
        };
        f.refs[f.hdr.ref_frame_idx[k]].clone()
    };
    let (fw, uw, fh) = (
        f.hdr.frame_width as usize,
        f.hdr.upscaled_width as usize,
        f.hdr.frame_height as usize,
    );
    let grey = 1u16 << (f.bit_depth - 1);
    for &(r0, r1, c0, c1) in &rects {
        for p in 0..f.num_planes {
            let (sx, sy) = f.plane_ss(p);
            let (pw, ph) = ((uw + sx) >> sx, (fh + sy) >> sy);
            // Superblock columns are in the frame's coded width; the
            // samples are in the upscaled one.
            let x0 = ((c0 * MI_SIZE * uw / fw) >> sx).min(pw);
            let x1 = ((c1 * MI_SIZE * uw).div_ceil(fw) >> sx).min(pw);
            let y0 = ((r0 * MI_SIZE) >> sy).min(ph);
            let y1 = ((r1 * MI_SIZE) >> sy).min(ph);
            let dst = &mut lr.planes[p];
            match &source {
                Some(r) => {
                    let src = &r.frame().planes[p];
                    let (rw, rh) = ((r.upscaled_width + sx) >> sx, (r.frame_height + sy) >> sy);
                    for y in y0..y1 {
                        let sy_ = (y * rh / ph).min(rh - 1);
                        let row = src.row(sy_);
                        for x in x0..x1 {
                            dst.set(x, y, row[(x * rw / pw).min(rw - 1)]);
                        }
                    }
                }
                None => {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            dst.set(x, y, grey);
                        }
                    }
                }
            }
        }
    }
    lr
}

/// The superblocks of `f` that did not decode, as rectangles of rows
/// `r0..r1` and columns `c0..c1` in 4x4 units: in each tile, those from
/// the one `sbs_done` counts to on.
fn undecoded(f: &FrameCtx) -> Vec<(usize, usize, usize, usize)> {
    let ti = &f.hdr.tile_info;
    let sb4 = if f.seq.use_128x128_superblock { 32 } else { 16 };
    let mut out = Vec::new();
    for t in 0..ti.rows * ti.cols {
        let done = f.sbs_done.get(t).copied().unwrap_or(0);
        let (row, col) = (t / ti.cols, t % ti.cols);
        let (r0, r1) = (ti.mi_row_starts[row], ti.mi_row_starts[row + 1]);
        let (c0, c1) = (ti.mi_col_starts[col], ti.mi_col_starts[col + 1]);
        let mut k = 0;
        let mut r = r0;
        while r < r1 {
            let mut c = c0;
            while c < c1 {
                if k >= done {
                    out.push((r, (r + sb4).min(r1), c, (c + sb4).min(c1)));
                }
                k += 1;
                c += sb4;
            }
            r += sb4;
        }
    }
    out
}
