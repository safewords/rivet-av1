//! What the tile walker asks of the encoder, in encode mode: the encoder's
//! choice for a syntax element the walker is about to code, and the hooks
//! where the encoder decides (`enc_*`, in `encoder::tile`).
//!
//! The walker is the decoder's: decoding, it codes nothing it is told, and
//! each of these is asked only when `Coder::encoding()` (or answers a
//! constant). Without the `encoder` feature they are constants and stubs,
//! and `encoding()` is `false` where it is compiled, so every encoding
//! branch goes and nothing of the encoder is linked: this file and
//! `Coder`'s `Enc` arms are all there is of the feature in the decoder.

use crate::decoder::tile::{LrChoice, TileDecoder};

#[cfg(feature = "encoder")]
impl TileDecoder<'_, '_> {
    /// The encoder's choice for loop restoration unit `idx` of `plane`.
    #[inline]
    pub(crate) fn planned_lr(&self, plane: usize, idx: usize) -> LrChoice {
        self.enc
            .as_ref()
            .and_then(|e| e.lr_plan.as_ref())
            .and_then(|p| p.units[plane].get(idx).copied())
            .unwrap_or_default()
    }

    /// The encoder's `cdef_idx` for the 64x64 block at `i` of `cdef_idx`.
    #[inline]
    pub(crate) fn planned_cdef(&self, i: usize) -> u32 {
        self.enc
            .as_ref()
            .and_then(|e| e.cdef_table.as_ref())
            .map_or(0, |t| t[i].max(0) as u32)
    }

    /// The encoder's luma palette index at row `i`, column `j`.
    #[inline]
    pub(crate) fn planned_palette_y(&self, i: usize, j: usize) -> u8 {
        self.enc.as_ref().map_or(0, |e| e.palette_map[i][j])
    }

    /// Whether the encoder is probing a prediction, which it measures by
    /// its luma alone.
    #[inline]
    pub(crate) fn probing(&self) -> bool {
        self.enc.as_ref().is_some_and(|e| e.probing)
    }

    /// Whether the encoder's transform type search left the transform
    /// block coded already (its levels' symbols counted, it reconstructed).
    #[inline]
    pub(crate) fn enc_tx_coded(&self) -> bool {
        self.enc.as_ref().is_some_and(|e| e.rdo.tx_coded.is_some())
    }

    /// [`Self::enc_tx_coded`], clearing it.
    #[inline]
    pub(crate) fn enc_take_tx_coded(&mut self) -> bool {
        self.enc
            .as_mut()
            .is_some_and(|e| e.rdo.tx_coded.take().is_some())
    }

    /// Counts a transform block with coefficients, for the encoder's search.
    #[inline]
    pub(crate) fn count_nonzero(&mut self) {
        if let Some(e) = self.enc.as_mut() {
            e.rdo.nonzero += 1;
        }
    }
}

#[cfg(not(feature = "encoder"))]
impl TileDecoder<'_, '_> {
    #[inline(always)]
    pub(crate) fn planned_lr(&self, _plane: usize, _idx: usize) -> LrChoice {
        LrChoice::default()
    }

    #[inline(always)]
    pub(crate) fn planned_cdef(&self, _i: usize) -> u32 {
        0
    }

    #[inline(always)]
    pub(crate) fn planned_palette_y(&self, _i: usize, _j: usize) -> u8 {
        0
    }

    #[inline(always)]
    pub(crate) fn probing(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn count_nonzero(&mut self) {}

    #[inline(always)]
    pub(crate) fn enc_prediction_cached(&mut self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn enc_prediction_store(&mut self) {}

    #[inline(always)]
    pub(crate) fn enc_tx_coded(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn enc_take_tx_coded(&mut self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn enc_level(&self, _pos: usize) -> u32 {
        0
    }

    #[inline(always)]
    pub(crate) fn enc_coef(&self, _pos: usize) -> i32 {
        0
    }

    // The hooks, never called: each call is behind `encoding()`.

    pub(crate) fn enc_begin_sb(&mut self, _r: usize, _c: usize) {
        unreachable!()
    }

    pub(crate) fn enc_partition(
        &mut self,
        _r: usize,
        _c: usize,
        _b_size: usize,
        _has_rows: bool,
        _has_cols: bool,
    ) -> crate::Result<usize> {
        unreachable!()
    }

    pub(crate) fn enc_decide_block(&mut self) -> crate::Result<bool> {
        unreachable!()
    }

    pub(crate) fn enc_delta_qindex(&self) -> i32 {
        unreachable!()
    }

    pub(crate) fn enc_palette_map(&mut self, _onscreen_width: usize, _onscreen_height: usize) {
        unreachable!()
    }

    pub(crate) fn enc_choose_coeffs(
        &mut self,
        _plane: usize,
        _start_x: usize,
        _start_y: usize,
        _tx_sz: usize,
    ) -> usize {
        unreachable!()
    }

    pub(crate) fn levels_encoder(
        &mut self,
        _b: &crate::decoder::residual::LevelBlock,
        _scan: &[u16],
    ) -> (u32, u8) {
        unreachable!()
    }
}
