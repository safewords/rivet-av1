# Decoding on several threads

How `Decoder::set_threads(n)` uses its threads, and why that is allowed:
the rules below are read off the AV1 specification's own processes (which
arrays a process reads, and which process wrote them), not off another
decoder. Section numbers are the specification's.

The measurements are in [wasm-performance.md](wasm-performance.md),
"Threads".

## What depends on what

### Inside a tile: parsing

The symbols of a tile are read by one thread, in order. Every symbol is
decoded against a CDF that the symbols before it adapted (8.2.6), and which
CDF it is depends on what the blocks above and to the left turned out to
be (8.3.2). There is nothing to split.

What the syntax of a block reads besides the bitstream (5.11.5 to 5.11.58,
with the context derivations of 8.3.2 and motion vector prediction, 7.10):

- the tile's CDFs and its above and left context arrays;
- the mode info arrays of the blocks before it in the tile (`MiSizes`,
  `YModes`, `RefFrames`, `Mvs`, `IsInters`, `SkipModes`, `Skips`,
  `TxSizes`, `InterTxSizes`, `SegmentIds`, `PaletteSizes`,
  `PaletteColors`, ...), only ever at positions `is_inside()` the tile;
- the frame header and what `setup_past_independence` /
  `load_previous()` set up before the first tile: the CDFs and segment
  map of `primary_ref_frame`, and `MotionFieldMvs`, projected (7.9) from
  the `SavedMvs`, `SavedRefFrames` and `SavedOrderHints` of the reference
  frames.

It never reads a sample: `CurrFrame` appears in section 7.11 and later,
not in 5.11 or 8.3. So **the parsing of a block depends on the parsing of
the blocks before it and on nothing its reconstruction does.**

### Inside a tile: reconstruction

What predicting and reconstructing a block (7.11, 7.12.3) reads:

1. Its own mode info and coefficients: the parsing's output.
2. Mode info of other blocks: the block above and the block to the left
   (the intra edge filter type, 7.11.2.8; overlapped motion compensation,
   7.11.3.10), and for the chroma of a block smaller than 8x8 the luma
   blocks of the same 8x8 (`candRow`, `candCol` in 5.11.33). All decoded
   earlier, all in the block's own superblock row or the row of 4x4s
   directly above it.
3. Samples of the frame being decoded, `CurrFrame`:
   - intra prediction's edge (7.11.2): the column to the left, down to
     `y + 2h - 1` only where `haveBelowLft`; the row above, to
     `x + 2w - 1` only where `haveAboveRt`; and the sample above-left.
     `BlockDecoded` (5.11.3) is true below-left only inside the current
     superblock, and above-right at most one 4x4 into the superblock to
     the right, in the row above. With `w <= 64`, the edge of a block in
     superblock `(r, c)` therefore reaches, outside its own superblock
     row, **the last sample row of superblock row `r - 1`, from the last
     sample of superblock `c - 1` to the last sample of superblock
     `c + 1`**, and nothing else;
   - chroma from luma (7.11.5): the block's own luma;
   - intra block copy (7.11.3.2 with `use_intrabc`): any part of the
     frame decoded so far.
4. Samples of reference frames (7.11.3.3, 7.11.3.4, 7.11.3.5): complete
   frames here, which nothing writes any more.

So, intra block copy aside, **superblock `(r, c)` can be reconstructed
once its row is parsed, superblock `(r, c - 1)` is reconstructed, and
superblock `(r - 1, c + 1)` is** (or the row above is complete): a
wavefront. And the only samples it needs from outside its own superblock
row are one line of the row above.

### Tiles

The tiles of a frame read nothing of each other while they are decoded:
every neighbour test is `is_inside()` the tile (5.11.4, 7.10.2), each tile
starts from the frame's CDFs and contexts (`init_symbol`, `clear_above_context`),
and intra edges stop at the tile (`AvailU`, `AvailL`, `haveAboveRt` through
`MiColEnd`). The frame keeps the CDFs of one of them
(`context_update_tile_id`).

## One-tile frames: rows behind the parsing

`src/decoder/pipeline.rs`, taken for a frame of one tile without intra
block copy when the decoder has more than one thread. (Other frames are
decoded as before: tiles in parallel, each into a frame state of its own
that is copied back; a tile's blocks parsed and reconstructed as they
come.)

The tile walker (`TileDecoder`) has three ways of walking
(`src/decoder/record.rs`):

- `Pass::Both`, what `decode_block()` says: mode info, prediction,
  coefficients and reconstruction of each block in turn. One thread, the
  encoder, and frames the pipeline does not take.
- `Pass::Parse`: the syntax alone. It writes the mode info arrays as
  before, and a **record** of the superblock row: for each block the
  fields prediction and reconstruction read (`BlockRec`), its palette and
  warp candidates if it has them, and for each transform block `eob`, the
  transform type and the coefficients as the reader left them
  (dequantised, over the rows and columns they reach; or `Quant` itself
  for the blocks the reference `reconstruct` handles: lossless, 12 bits,
  the reference coefficient reader).
- `Pass::Recon`: the same walk over the record instead of the bitstream:
  `compute_prediction()`, and `residual()` with each `coeffs()` replaced
  by the record's. `predict_*` and `reconstruct_decode` are the same
  functions in all three.

That the two halves give what the whole gives rests on the rules above
and on one fact about the walker itself: nothing between a block's
prediction and the end of its residual reads what `decode_block()` stores
for *that* block after its residual (`IsInters`, `Skips`, `TxSizes`, ...
of its own positions), so it does not matter that for the reconstructing
walk they are already there. The 244 conformance vectors decode
bit-exactly through it (`AV1_DECODE_THREADS`), and `tests/threads.rs`
holds every thread count to one thread.

The threads of a frame (one `parallel::run` for its tile):

- **The calling thread parses**, superblock row after superblock row.
  When a row is complete it gives up its claim to write the row's mode
  info (below) and publishes the row's record.
- **Every thread reconstructs rows** (the calling thread once it has
  parsed the last): rows are taken in order, a thread at a time. Before
  superblock `c` of row `r` a thread waits until row `r - 1` has finished
  superblock `c + 1`, and after each superblock it publishes the last
  sample line of that superblock: the line travels as a copy, under the
  lock the rows' progress is kept under, into the reading thread's own
  buffer (`TileDecoder::above_line`). Intra prediction's edge is the one
  place that reads above the superblock row, and it reads the copy.

So during reconstruction every sample row of the frame is touched by
exactly one thread: the one reconstructing its superblock row.

### What is shared, and how

| what | who writes | who reads |
|---|---|---|
| mode info (`FrameCtx::mi`) | the parsing thread, the rows of the superblock row it is parsing | the parsing thread, any row; a reconstructing thread, its row and the row above, both parsed |
| `CurrFrame` (`FrameCtx::cur`) | a reconstructing thread, the sample rows of its superblock row | the same thread |
| a row's record | the parsing thread, until it publishes it | the one thread that takes it |
| rows' progress, last sample lines, the queue of records | under one mutex | under the same mutex |
| the frame header, the references, `MotionFieldMvs`, the previous segment map | nobody | everybody (`Arc`, or each thread's copy) |
| transform types, segment ids, palette colours, CDEF indices, loop restoration units, the CDFs and contexts | the parsing thread | the parsing thread alone, until the tile is done |

The first two are single allocations whose rows belong to different
threads at the same time. They are [`Shared<T>`](../src/decoder/shared.rs),
the one type with `unsafe` in the scheme, and a small one. A `Shared` is a
*window* on an array: a range of elements it may read and a range it may
write. Indexing checks the range (as indexing a `Vec` checks its length).
A window is made by `Lender::window`, which refuses, by panicking, to make
one that could conflict with a window in existence: the invariant is that
**the write range of every window is disjoint from both ranges of every
other window of the same array**. From it follows that a `&mut T` handed
out is unique and a `&T` handed out is to an element nobody can write,
which is all its `unsafe` blocks need; and a mistake in the schedule
(two tasks given the same rows) stops the decoder with a panic rather
than racing. Windows are registered and dropped under the array's lock,
which also orders one thread's last write before the next thread's first
read of the same elements.

The unit tests of `shared.rs` cover the window arithmetic, every kind of
refused conflict, indexing outside a window, and rows changing hands
between two threads.

### Errors, and never waiting forever

Only parsing reports errors (reconstruction has nothing to refuse: the
syntax fixed every size and position it uses). When a row fails to parse,
the parsing thread marks the frame abandoned and wakes everybody: rows
already being reconstructed stop at their next superblock, no new row is
taken, and the error is returned as the one-thread decoder returns it;
the frame is dropped, as it is there. A thread that panics does the same
on its way out (a drop guard), so the others end instead of waiting for
a row that will never be published.

Every wait is for something another thread is already committed to
doing: a row waits for the row above, which was taken before it; a thread
without a row waits for the parser, which either publishes a row,
finishes or fails. With no other thread at all (a `Runner` that starts
none) the calling thread parses the tile and then reconstructs every row
itself, in order, and waits for nothing.

`tests/threads.rs` decodes the committed vectors on 2, 3, 4, 8 and 16
threads against one thread, repeats threaded decodes many times, and
decodes streams with flipped bits and cut units on 1, 2, 4 and 8 threads
under a watchdog: the same frames and the same errors, and no hang.

### Memory

A frame in the pipeline holds, besides what one thread holds, the records
of the rows parsed and not yet reconstructed (at most the frame's rows:
the dequantised coefficients and about 60 bytes a block), one sample line
per superblock row, and per thread a frame state without arrays and a
tile walker. One frame is in flight at a time: `decode_all` returns when
the frame is complete.

## Where the threads come from

`parallel::run(n, job)` as before: scoped threads of the standard
library, or the embedder's `Runner` (a pool of Web Workers in wasm-av1).
The waits are `std::sync::Mutex` and `Condvar`, which on `wasm32` with
`+atomics` are `memory.atomic.wait32` and `notify`: the calling thread
blocks too, so it must be a Worker there, as it had to be already.

## Ideas known from elsewhere

That decoders run superblock rows in a wavefront and separate entropy
decoding from reconstruction is common knowledge, not this crate's
invention. What is here was worked out from the dependencies above: which
single sample line crosses a superblock row (from 7.11.2 and
`BlockDecoded`), the two-superblock lag (the same derivation the encoder's
wavefront in `src/encoder/wavefront.rs` already used), and what a block's
record must hold (from the fields 7.11 and 7.12.3 read). No other
decoder's source or design notes were read.
