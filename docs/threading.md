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

### The post-filters

Each works on the frame the one before it left (7.14 to 7.17).

**The loop filter** (7.14.1) visits the planes, and in each plane all the
vertical edges of the frame, then all the horizontal ones. An edge's
filter reads and writes samples across the edge only (7.14.6): at most 7
on each side in luma, 3 in chroma, and never past the far side of the two
transform blocks the edge lies between (the filter size is bounded by
both, 7.14.3). A transform block does not cross a superblock row, so:

- the vertical edges of a superblock row touch that row's samples only;
- the horizontal edges *inside* a row touch that row only, and the edge
  along its top touches it and the last 7 (3) sample rows of the row
  above;
- the horizontal edges of two rows never touch the same sample.

The order the specification asks for is therefore kept by taking the
superblock rows in order and doing, for each, its vertical edges and then
its horizontal ones (its top edge among them): when row `r`'s top edge is
filtered, the vertical edges of both rows it touches have been; and when
row `r + 1`'s vertical edges are filtered, nothing has yet touched its
samples. Row `r`'s filter needs row `r` reconstructed (and the mode info
and transform sizes of rows `r - 1` and `r`); it leaves the last 7 (3)
rows of row `r` unfinished until row `r + 1`'s.

**CDEF** (7.15) writes `CdefFrame` from `CurrFrame` as the loop filter
left it. An 8x8 block reads the skip flags and `cdef_idx` of its own
position and `CurrFrame` up to 2 samples around the block (7.15.2,
7.15.3: `Cdef_Directions`), inside the frame. So the CDEF of superblock
row `r` needs `CurrFrame` final in rows `top(r) - 2` to `bottom(r) + 2`:
the loop filter of row `r + 1` done (which finishes row `r`'s last rows
and filters the first rows of row `r + 1`; those are not touched again
unless row `r + 1` is the frame's last and only a few rows high, when
nothing comes after it). Different rows' CDEF write different rows of
`CdefFrame` and can run at once.

**Super-resolution** (7.16) stands between CDEF and loop restoration and
changes the planes' width. Frames that use it are filtered as before, the
whole frame after its last row is reconstructed.

**Loop restoration** (7.17) writes `LrFrame` in stripes of 64 luma rows
that start 8 rows above a multiple of 64 (7.17.1). A stripe reads
`CdefFrame` in its own rows and `CurrFrame` (deblocked, before CDEF) in
the 2 rows above and the 2 below it (7.17.6), and the parameters of the
restoration units it crosses, which the superblock that holds a unit's
first sample coded (5.11.57). So stripe `s` needs CDEF done for the
superblock rows holding luma rows `64 s - 8` to `64 s + 55`; the two
deblocked rows below it are in the last of those superblock rows too
(`64 s + 56` is not a multiple of 64), and final by then. Stripes write
their own rows and can run at once.

### Tiles

The tiles of a frame read nothing of each other while they are decoded:
every neighbour test is `is_inside()` the tile (5.11.4, 7.10.2), each tile
starts from the frame's CDFs and contexts (`init_symbol`, `clear_above_context`),
and intra edges stop at the tile (`AvailU`, `AvailL`, `haveAboveRt` through
`MiColEnd`). The frame keeps the CDFs of one of them
(`context_update_tile_id`).

## One-tile frames: a pipeline of rows

`src/decoder/pipeline.rs`, taken for a frame of one tile without intra
block copy when the decoder has more than one thread. (Other frames are
decoded as before: tiles in parallel, each into a frame state of its own
that is copied back, a tile's blocks parsed and reconstructed as they
come; then each post-filter over the whole frame, in bands.)

### Parsing apart from reconstruction

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

### The threads of a frame

One `parallel::run` for the frame's tile.

**The calling thread parses**, superblock row after superblock row. When
a row is complete it gives up its claim to write the row's mode info and
transform sizes (below), copies the CDEF indices and loop restoration
units the row coded to where the filters will look for them, and
publishes the row's record.

**Every thread takes tasks** (the calling thread too, once it has parsed
the last row). A task is one stage of one row, and is ready when what it
reads is final:

| task | ready when | reads | writes |
|---|---|---|---|
| reconstruct row `r` | row `r` is parsed and rows before it have been taken | its record; mode info of rows `r - 1`, `r`; a copy of the last sample line of row `r - 1` | `CurrFrame`, the sample rows of row `r` |
| deblock row `r` | row `r` and all before it are reconstructed, rows before it deblocked | mode info and transform sizes of rows `r - 1`, `r` | `CurrFrame`, row `r` and the last 8 (4 in chroma) sample rows of row `r - 1` |
| CDEF row `r` | rows to `r + 1` are deblocked (reconstructed, in a frame without loop filter), or `r` is the last | `CurrFrame`, row `r` and 2 sample rows either side; skip flags of row `r`; its CDEF indices | `CdefFrame`, row `r` |
| restore stripe `s` | the superblock rows holding its luma rows are through CDEF (in a frame without CDEF: deblocked, and the row after them too) | `CdefFrame`, its rows; `CurrFrame`, 2 rows either side; its units | `LrFrame`, its rows, in each restored plane |

Among the ready tasks a thread takes the stage furthest along, so that
few rows are in flight. Rows are reconstructed in order and a row is
taken only after the one above it; inside the task, superblock `c` waits
until the row above has finished superblock `c + 1`, and after each
superblock the task publishes the last sample line of that superblock:
the line travels as a copy, under the lock the tasks' bookkeeping is
kept under, into the reading thread's own buffer
(`TileDecoder::above_line`). Intra prediction's edge is the one place
that reads above the superblock row, and it reads the copy. (That copy is
also what lets the row above be deblocked while the row below is still
being predicted from its unfiltered samples.)

The frame's result is, plane by plane, what the last stage that plane
has wrote: `LrFrame` for a restored plane, else `CdefFrame`, else
`CurrFrame`.

### What is shared, and how

| what | who writes | who reads |
|---|---|---|
| mode info (`FrameCtx::mi`), transform sizes (`lf_tx_sizes`) | the parsing thread, the rows of the superblock row it is parsing | the parsing thread, any row; a task, rows already parsed |
| `CurrFrame`, `CdefFrame`, `LrFrame` | a task, the sample rows the table gives it | tasks, the sample rows the table gives them |
| a row's record | the parsing thread, until it publishes it | the one thread that reconstructs the row |
| tasks' bookkeeping, rows' last sample lines, the queue of records, the published CDEF indices and restoration units | under one mutex | under the same mutex |
| the frame header, the references, `MotionFieldMvs`, the previous segment map, the loop filter's level table | nobody | everybody (`Arc`, or each thread's copy) |
| transform types, segment ids, palette colours, the CDFs and contexts, the parser's own CDEF indices and restoration units | the parsing thread | the parsing thread alone, until the tile is done |

The first two rows are single allocations whose parts belong to different
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
read of the same elements. A task takes its windows when it starts and
drops them before it is recorded as done, so the readiness rules of the
table are what keeps the windows from ever conflicting; the windows are
the check on the rules. (The loop filter's and CDEF's kernels address a
plane by pointer, as they did: `SharedPlane` is now made from a window
and reaches exactly the window's samples.)

The unit tests of `shared.rs` cover the window arithmetic, every kind of
refused conflict, indexing outside a window, and rows changing hands
between two threads.

### Errors, and never waiting forever

Only parsing reports errors (reconstruction and the filters have nothing
to refuse: the syntax fixed every size and position they use). When a row
fails to parse, the parsing thread marks the frame abandoned and wakes
everybody: a row being reconstructed stops at its next superblock, no
new task is taken, and the error is returned as the one-thread decoder
returns it; the frame is dropped, as it is there. A thread that panics
does the same on its way out (a drop guard), so the others end instead of
waiting for a row that will never come.

Every wait is for something another thread is committed to doing: a row
waits for the row above, which was taken before it; a thread without a
task waits for the parser (which publishes a row, finishes or fails) or
for a task in hand to finish. If ever nothing were ready with the parser
finished and no task in hand, the wait would never end: that would be a
mistake in the readiness rules, and the thread that finds it panics
instead of waiting. With no other thread at all (a `Runner` that starts
none) the calling thread parses the tile and then carries out every task
itself, in the order the rules allow, and waits for nothing.

`tests/threads.rs` decodes the committed vectors on 2, 3, 4, 8 and 16
threads against one thread, repeats threaded decodes many times, and
decodes streams with flipped bits and cut units on 1, 2, 4 and 8 threads
under a watchdog: the same frames and the same errors, and no hang.

### Memory

A frame in the pipeline holds what one thread's decode holds (`CurrFrame`,
`CdefFrame`, `LrFrame` and the per-4x4 arrays; less, in fact: a frame
without CDEF has no `CdefFrame`, and `LrFrame` only the restored planes)
and besides: the records of the rows parsed and not yet reconstructed (at
most the frame's rows: the dequantised coefficients and about 60 bytes a
block), one sample line per superblock row, and per thread a frame state
without arrays and a tile walker. One frame is in flight at a time:
`decode_all` returns when the frame is complete.

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
wavefront in `src/encoder/wavefront.rs` already used), what a block's
record must hold (from the fields 7.11 and 7.12.3 read), and how far each
filter trails the stage before it (from the samples 7.14.6, 7.15.3 and
7.17.6 read). That post-filters can follow reconstruction row by row is
likewise a known arrangement; the margins here are the ones derived
above. No other decoder's source or design notes were read.
