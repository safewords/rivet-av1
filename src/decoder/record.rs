//! What parsing a superblock row leaves for its reconstruction.
//!
//! A tile's symbols are read in order by one thread: every symbol's
//! meaning depends on the CDFs and contexts the ones before it left. What
//! the symbols say to *do* (predict, add a residual) reads no symbol and
//! decides nothing the parsing depends on: the syntax of 5.11 never looks
//! at a sample of `CurrFrame`. So the two can be taken apart
//! (`docs/threading.md`): a parsing walk ([`Pass::Parse`]) that reads the
//! block's mode info and coefficients and writes down what the block's
//! reconstruction needs, and a reconstructing walk ([`Pass::Recon`]) that
//! replays the row from that record on another thread, later. With one
//! thread (and in the encoder) the tile walker does both as it goes
//! ([`Pass::Both`]), as the specification writes it.

use crate::consts::*;
use crate::decoder::tile::{Block, TileDecoder};
use crate::tables::*;

/// Which part of decoding a block the tile walker does.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Pass {
    /// Everything, block by block: `decode_block()` as written.
    #[default]
    Both,
    /// The syntax alone: mode info and coefficients are read (and the
    /// frame's mode info arrays written), nothing is predicted or
    /// reconstructed, and a [`RowRecord`] is written.
    Parse,
    /// Prediction and reconstruction alone, from a [`RowRecord`].
    Recon,
}

/// A block as its reconstruction needs it: the fields of [`Block`] that
/// prediction (7.11) and reconstruction (7.12.3) read, in their ranges.
#[derive(Clone, Copy, Default)]
pub(crate) struct BlockRec {
    mi_row: u16,
    mi_col: u16,
    mi_size: u8,
    has_chroma: bool,
    avail_u: bool,
    avail_l: bool,
    avail_u_chroma: bool,
    avail_l_chroma: bool,
    skip: bool,
    is_inter: bool,
    lossless: bool,
    use_filter_intra: bool,
    interintra: bool,
    wedge_interintra: bool,
    y_mode: u8,
    uv_mode: u8,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    filter_intra_mode: u8,
    cfl_alpha_u: i8,
    cfl_alpha_v: i8,
    palette_size_y: u8,
    palette_size_uv: u8,
    ref_frame: [i8; 2],
    mv: [[i32; 2]; 2],
    interintra_mode: u8,
    wedge_index: u8,
    wedge_sign: u8,
    mask_type: u8,
    motion_mode: u8,
    compound_type: u8,
    tx_size: u8,
    segment_id: u8,
    num_samples: u8,
    /// `CurrentQIndex`, which the reference `reconstruct` dequantises with.
    q_index: u8,
}

/// One transform block's `coeffs()`: `eob`, and for a block that has
/// coefficients its transform type and how they were written to
/// [`RowRecord::coefs`].
#[derive(Clone, Copy)]
pub(crate) struct TxRec {
    eob: u16,
    tx_type: u8,
    /// The rows and columns of `Dequant` the block's coefficients reach,
    /// which were written, row by row; or 0 for a block left to the
    /// reference `reconstruct` (lossless, 12 bits, or the reference
    /// coefficient reader), whose `Quant` was written whole.
    rows: u8,
    cols: u8,
}

/// A parsed superblock row, in decoding order.
#[derive(Default)]
pub(crate) struct RowRecord {
    blocks: Vec<BlockRec>,
    /// The first block of each superblock, and the number of blocks.
    sb_starts: Vec<u32>,
    txs: Vec<TxRec>,
    coefs: Vec<i32>,
    /// Palette colour maps, a block's rows one after the other.
    bytes: Vec<u8>,
    /// Palette colours and warp candidates.
    words: Vec<i32>,
}

impl RowRecord {
    /// Empties the record, keeping its memory.
    pub(crate) fn clear(&mut self) {
        self.blocks.clear();
        self.sb_starts.clear();
        self.txs.clear();
        self.coefs.clear();
        self.bytes.clear();
        self.words.clear();
    }

    /// A superblock begins here (parsing).
    pub(crate) fn begin_sb(&mut self) {
        self.sb_starts.push(self.blocks.len() as u32);
    }

    /// The row is complete (parsing).
    pub(crate) fn finish(&mut self) {
        self.sb_starts.push(self.blocks.len() as u32);
    }

    /// The blocks of superblock `k` of the row.
    pub(crate) fn sb_blocks(&self, k: usize) -> std::ops::Range<usize> {
        self.sb_starts[k] as usize..self.sb_starts[k + 1] as usize
    }
}

/// Where a replay has got to in a [`RowRecord`]'s streams.
#[derive(Clone, Copy, Default)]
pub(crate) struct Cursor {
    tx: usize,
    coef: usize,
    byte: usize,
    word: usize,
}

impl TileDecoder<'_, '_> {
    /// Writes the block just parsed (its mode info, palette and transform
    /// size) to the row's record.
    pub(crate) fn record_block(&mut self) {
        let b = &self.b;
        let rec = &mut self.rec;
        rec.blocks.push(BlockRec {
            mi_row: b.mi_row as u16,
            mi_col: b.mi_col as u16,
            mi_size: b.mi_size as u8,
            has_chroma: b.has_chroma,
            avail_u: b.avail_u,
            avail_l: b.avail_l,
            avail_u_chroma: b.avail_u_chroma,
            avail_l_chroma: b.avail_l_chroma,
            skip: b.skip,
            is_inter: b.is_inter,
            lossless: b.lossless,
            use_filter_intra: b.use_filter_intra,
            interintra: b.interintra,
            wedge_interintra: b.wedge_interintra,
            y_mode: b.y_mode as u8,
            uv_mode: b.uv_mode as u8,
            angle_delta_y: b.angle_delta_y as i8,
            angle_delta_uv: b.angle_delta_uv as i8,
            filter_intra_mode: b.filter_intra_mode as u8,
            cfl_alpha_u: b.cfl_alpha_u as i8,
            cfl_alpha_v: b.cfl_alpha_v as i8,
            palette_size_y: b.palette_size_y as u8,
            palette_size_uv: b.palette_size_uv as u8,
            ref_frame: [b.ref_frame[0] as i8, b.ref_frame[1] as i8],
            mv: b.mv,
            interintra_mode: b.interintra_mode as u8,
            wedge_index: b.wedge_index as u8,
            wedge_sign: b.wedge_sign as u8,
            mask_type: b.mask_type as u8,
            motion_mode: b.motion_mode as u8,
            compound_type: b.compound_type as u8,
            tx_size: b.tx_size as u8,
            segment_id: b.segment_id as u8,
            num_samples: b.num_samples as u8,
            q_index: self.current_q_index as u8,
        });
        let (bw, bh) = (block_width(b.mi_size), block_height(b.mi_size));
        if b.palette_size_y > 0 {
            rec.words
                .extend(b.palette_colors_y.iter().map(|&c| c as i32));
            for row in &self.color_map_y[..bh] {
                rec.bytes.extend_from_slice(&row[..bw]);
            }
        }
        if b.palette_size_uv > 0 {
            rec.words
                .extend(b.palette_colors_u.iter().map(|&c| c as i32));
            rec.words
                .extend(b.palette_colors_v.iter().map(|&c| c as i32));
            let (uw, uh) = palette_uv_size(bw, bh, self.f.ssx, self.f.ssy);
            for row in &self.color_map_uv[..uh] {
                rec.bytes.extend_from_slice(&row[..uw]);
            }
        }
        if b.motion_mode == LOCALWARP {
            for cand in &b.cand_list[..b.num_samples.min(8)] {
                rec.words.extend_from_slice(cand);
            }
        }
    }

    /// Makes block `i` of the row's record the current block, as
    /// [`Self::record_block`] saw it.
    fn load_block(&mut self, i: usize) {
        let r = self.rec.blocks[i];
        self.b = Block {
            mi_row: r.mi_row as usize,
            mi_col: r.mi_col as usize,
            mi_size: r.mi_size as usize,
            has_chroma: r.has_chroma,
            avail_u: r.avail_u,
            avail_l: r.avail_l,
            avail_u_chroma: r.avail_u_chroma,
            avail_l_chroma: r.avail_l_chroma,
            skip: r.skip,
            is_inter: r.is_inter,
            lossless: r.lossless,
            use_filter_intra: r.use_filter_intra,
            interintra: r.interintra,
            wedge_interintra: r.wedge_interintra,
            y_mode: r.y_mode as usize,
            uv_mode: r.uv_mode as usize,
            angle_delta_y: r.angle_delta_y as i32,
            angle_delta_uv: r.angle_delta_uv as i32,
            filter_intra_mode: r.filter_intra_mode as usize,
            cfl_alpha_u: r.cfl_alpha_u as i32,
            cfl_alpha_v: r.cfl_alpha_v as i32,
            palette_size_y: r.palette_size_y as usize,
            palette_size_uv: r.palette_size_uv as usize,
            ref_frame: [r.ref_frame[0] as i32, r.ref_frame[1] as i32],
            mv: r.mv,
            interintra_mode: r.interintra_mode as usize,
            wedge_index: r.wedge_index as usize,
            wedge_sign: r.wedge_sign as usize,
            mask_type: r.mask_type as usize,
            motion_mode: r.motion_mode as u32,
            compound_type: r.compound_type as u32,
            tx_size: r.tx_size as usize,
            segment_id: r.segment_id as usize,
            num_samples: r.num_samples as usize,
            ..Block::default()
        };
        self.current_q_index = r.q_index as i32;
        let rec = &self.rec;
        let cur = &mut self.cursor;
        let (bw, bh) = (block_width(self.b.mi_size), block_height(self.b.mi_size));
        let colors = |cur: &mut Cursor| {
            let mut c = [0u16; 8];
            for (o, &v) in c.iter_mut().zip(&rec.words[cur.word..cur.word + 8]) {
                *o = v as u16;
            }
            cur.word += 8;
            c
        };
        if self.b.palette_size_y > 0 {
            self.b.palette_colors_y = colors(cur);
            for row in self.color_map_y[..bh].iter_mut() {
                row[..bw].copy_from_slice(&rec.bytes[cur.byte..cur.byte + bw]);
                cur.byte += bw;
            }
        }
        if self.b.palette_size_uv > 0 {
            self.b.palette_colors_u = colors(cur);
            self.b.palette_colors_v = colors(cur);
            let (uw, uh) = palette_uv_size(bw, bh, self.f.ssx, self.f.ssy);
            for row in self.color_map_uv[..uh].iter_mut() {
                row[..uw].copy_from_slice(&rec.bytes[cur.byte..cur.byte + uw]);
                cur.byte += uw;
            }
        }
        if self.b.motion_mode == LOCALWARP {
            for cand in self.b.cand_list[..self.b.num_samples.min(8)].iter_mut() {
                cand.copy_from_slice(&rec.words[cur.word..cur.word + 4]);
                cur.word += 4;
            }
        }
    }

    /// Predicts and reconstructs block `i` of the row's record: the second
    /// half of `decode_block()`, the first being what
    /// [`Self::record_block`] followed.
    pub(crate) fn replay_block(&mut self, i: usize) -> crate::Result<()> {
        self.load_block(i);
        self.compute_prediction();
        self.residual()
    }

    /// Starts replaying the row's record from its beginning.
    pub(crate) fn rewind(&mut self) {
        self.cursor = Cursor::default();
    }

    /// Writes the outcome of `coeffs()` for the transform block just
    /// read: `eob`, and the coefficients where
    /// [`Self::reconstruct_decode`] will look for them (`Dequant` over the
    /// rows and columns the reader says they reach, else `Quant`).
    pub(crate) fn record_tx(
        &mut self,
        tx_sz: usize,
        eob: usize,
        dequantised: Option<(usize, usize)>,
    ) {
        let rec = &mut self.rec;
        if eob == 0 {
            rec.txs.push(TxRec {
                eob: 0,
                tx_type: 0,
                rows: 0,
                cols: 0,
            });
            return;
        }
        let (rows, cols) = match dequantised {
            Some((rows, cols)) => {
                for row in self.dequant.as_chunks::<64>().0[..rows].iter() {
                    rec.coefs.extend_from_slice(&row[..cols]);
                }
                (rows, cols)
            }
            None => {
                let n = TX_WIDTH[tx_sz].min(32) * TX_HEIGHT[tx_sz].min(32);
                rec.coefs.extend_from_slice(&self.quant[..n]);
                (0, 0)
            }
        };
        rec.txs.push(TxRec {
            eob: eob as u16,
            tx_type: self.plane_tx_type as u8,
            rows: rows as u8,
            cols: cols as u8,
        });
    }

    /// The next transform block of the row's record, put back where
    /// `coeffs()` left it: returns what `coeffs_dequantised` returned.
    pub(crate) fn replay_tx(&mut self, tx_sz: usize) -> (usize, Option<(usize, usize)>) {
        let t = self.rec.txs[self.cursor.tx];
        self.cursor.tx += 1;
        if t.eob == 0 {
            return (0, None);
        }
        self.plane_tx_type = t.tx_type as usize;
        let (tw, th) = (TX_WIDTH[tx_sz].min(32), TX_HEIGHT[tx_sz].min(32));
        let at = self.cursor.coef;
        if t.rows == 0 {
            let n = tw * th;
            self.quant[..n].copy_from_slice(&self.rec.coefs[at..at + n]);
            self.cursor.coef += n;
            return (t.eob as usize, None);
        }
        let (rows, cols) = (t.rows as usize, t.cols as usize);
        // `Dequant` as the reader's `Dequantiser::clear` and `put`s left
        // it: zero wherever the inverse transform may read (whole groups
        // of eight rows), the coefficients over them.
        let dq = self.dequant.as_chunks_mut::<64>().0;
        for row in dq[..th.next_multiple_of(8)].iter_mut() {
            row[..tw].fill(0);
        }
        for (row, src) in dq[..rows]
            .iter_mut()
            .zip(self.rec.coefs[at..at + rows * cols].chunks_exact(cols))
        {
            row[..cols].copy_from_slice(src);
        }
        self.cursor.coef += rows * cols;
        (t.eob as usize, Some((rows, cols)))
    }
}

/// The size of a block's chroma colour map (`palette_tokens()`'s
/// `blockWidth` and `blockHeight` for the chroma planes).
fn palette_uv_size(bw: usize, bh: usize, ssx: usize, ssy: usize) -> (usize, usize) {
    let (mut uw, mut uh) = (bw >> ssx, bh >> ssy);
    if uw < 4 {
        uw += 2;
    }
    if uh < 4 {
        uh += 2;
    }
    (uw, uh)
}
