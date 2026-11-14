# rivet-av1

[![CI](https://github.com/safewords/rivet-av1/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-av1/actions/workflows/ci.yml)

An **AV1** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from the *AV1
Bitstream & Decoding Process Specification* (AOMedia), not translated from
any other implementation. The decoder is bit-exact on **all 244** of
AOMedia's public AV1 test vectors, on all 3 015 of the Argon
conformance streams and all 571 of their large-scale-tile streams (the numbers are
[below](#how-it-is-checked)); the encoder writes key and inter frames that
decode to exactly what it reconstructed.

Written for the **[rivet](https://github.com/safewords/rivet)**
transcoder, whose default output codec is AV1 (it is rivet's software AV1
decoder and encoder). Usable on its own by anything that has
AV1 temporal units (from IVF, WebM / Matroska, MP4, or an Annex B stream)
and wants planar pictures back, or planar pictures and wants AV1.

The decoder implements the whole specification, decodes tiles and runs its
post-filters on several threads, and has SIMD (AVX2, NEON) in its hottest
kernels; the encoder searches its decisions by rate-distortion trial coding,
uses most of the toolbox, codes superblock rows on several threads, and
holds an average bitrate. What is and is not there is listed precisely
below.

Published as `rivet-av1`; **imported as `av1`** (`use av1::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
av1 = { package = "rivet-av1", git = "https://github.com/safewords/rivet-av1", branch = "develop" }
```

## What it decodes

Everything in the specification's general decoding process:

| | |
|---|---|
| **Profiles** | 0, 1 and 2 (Main, High, Professional): 8-, 10- and 12-bit; monochrome, 4:2:0, 4:2:2 and 4:4:4; every colour description |
| **Bitstream** | OBUs in the low-overhead format (section 5) and the length-delimited format of Annex B; sequence headers, temporal delimiters, frame headers and redundant copies, tile groups, frame OBUs; HDR metadata OBUs (content light level, mastering display) reported on every frame, other metadata and padding skipped; operating point selection and layer dropping (scalable streams); frame ids |
| **Frames** | key, inter, intra-only and switch frames; hidden frames and `show_existing_frame` (including key frames shown that way); error-resilient mode; frame size changes, `frame_size_with_refs`, render sizes; reference frame scaling (2:1 down to 1:16) |
| **Syntax** | uniform and explicit tiles (any count), the symbol decoder with CDF adaptation, saving and loading CDFs through the reference slots, `disable_cdf_update` / `disable_frame_end_update_cdf`; segmentation (map, temporal prediction, every feature); delta quantiser and delta loop filter (single and multi); quantiser matrices; lossless |
| **Partitions** | every partition (none, horizontal, vertical, split, the A/B three-way splits, 4-way) from 128x128 superblocks down to 4x4; transform sizes 4x4 to 64x64 including all rectangular sizes, `TX_MODE_SELECT` and inter transform trees |
| **Intra** | all thirteen modes, angle deltas, the intra edge filter and edge upsampling, smooth modes, Paeth, filter intra (recursive), chroma from luma, palette (with the colour cache), intra block copy |
| **Inter** | single and compound prediction from seven references; motion vector prediction with the temporal motion field (projection); NEAREST/NEAR/GLOBAL/NEW and every compound mode, dynamic reference list; regular, smooth, sharp and bilinear filters with dual filters; global motion (translation, rotation-zoom, affine) and local warped motion; OBMC; inter-intra (smooth and wedge); wedge, difference-weighted and distance-weighted compound; skip mode |
| **Residual** | coefficient coding at every size; all sixteen transform types (DCT, ADST, flipped ADST, identity in every combination), the 64-point DCT, the Walsh-Hadamard transform; dequantisation at all bit depths |
| **Filters** | the deblocking loop filter (4-, 6-, 8- and 16-wide), CDEF, super-resolution upscaling, loop restoration (Wiener and self-guided) |
| **Output** | film grain synthesis (the specification's reference process, bit-exact) |

Output is a [`Frame`](src/frame.rs): the visible picture (`UpscaledWidth`
by `FrameHeight`), planes Y, U, V packed one after another, one byte per
sample at 8 bits and little-endian `u16` above — the shape of rivet-vp9's
`Frame`. `Decoder::decode(&temporal_unit)` takes one temporal unit (an IVF
frame or container block) and returns the frame it shows; with spatial
layers that is the last (highest) one, as the specification's output
process recommends, and `Decoder::decode_all` returns every one.
`Decoder::decode_annexb` takes an Annex B temporal unit
(`annexb_temporal_units` splits a stream into them).
`Decoder::set_apply_grain(false)` leaves film grain synthesis out: frames
come out as decoded, for a player that cannot afford the grain.

**Large scale tile decoding** (7.3) too: a tile list OBU names tiles of
coded "camera" frames, each predicted from one of up to 128 anchor frames
the application provides (`Decoder::set_anchor_frames`; typically the
frames the stream shows first, which is how AOMedia's light field example
takes them). With the frame header in force as the camera frame header
(one that comes without tile groups), each tile list decodes to one output
frame, its tiles side by side in the list's order, returned as a shown
frame is; `Decoder::tile_list_layout` says where each tile is.

Not there yet:

- **Speed 4 at rav1e's speed**: 4.4x its time on one thread (speed 6
  1.09x, speed 8 0.71x; above). What is left there is the search itself —
  four references searched exhaustively, every sub-sample candidate
  predicted with the normative filters — which only a search that tries
  less would change.
- **Tile lists in flight**: a tile list is decoded in place, not by a
  decoder with frames in flight (`queue`).
- **A frame-buffer pool for `CurrFrame`.** The copies the post-filters
  start from are made into the planes of frames nothing holds any more,
  but `CurrFrame` is allocated, zeroed, every frame: zeroing a recycled
  plane measured 4 % to 7 % slower than the system's zeroed pages, and
  recycling it without zeroing would need every sample to be shown
  written before it is read. The pool as it is is neither faster nor
  slower (below).
- **Error recovery.** A corrupt frame returns `Error::Bitstream`; there is
  no concealment, so decoding should resume at the next key frame.
  Malformed input has not made it panic under the property tests below,
  and frames larger than 2^26 pixels (adjustable, `set_max_pixels`) are
  refused before allocation.
- Conformance requirements that do not change the output (frame id
  consistency, level limits) are not checked; `set_strict(true)` checks
  each tile's trailing padding.

### Speed

`Decoder::set_threads(n)` decodes on up to `n` threads, and the frames
are the same whatever `n` ([docs/threading.md](docs/threading.md) derives
what may run at the same time from the specification, and says how it is
kept safe):

- A frame of one tile, what video sites serve: one thread parses the
  tile, the others reconstruct its superblock rows behind the parsing
  and deblock, CDEF and restore them behind the reconstruction.
- A frame of several tiles: the tiles in parallel (each worker into a
  frame state of its own, whose tile region is copied back), then each
  post-filter over the frame in bands.
- `Decoder::set_frames_in_flight(n)` with `queue`, `poll` and `wait`:
  up to `n` frames decoded at once, each waiting for the frames it
  refers to, their output a few units behind their input. A one-tile
  stream needs this to use more than two or three threads (a frame's
  parsing is one thread's); every unit's frames or error are those of
  decoding it in place.

The hottest kernels have SIMD versions chosen at run time —
AVX2 on x86-64, NEON on aarch64 — each tested bit-exact against its scalar
version: the CDEF block filter, the 8-tap interpolation filters, the
inverse transforms (eight rows or columns at a time; the 64-bit scalar
transforms remain for 12-bit and lossless) and the loop filter (the four
samples along an edge at once). `AV1_NO_SIMD=1` turns them off.

On `wasm32-unknown-unknown` the crate builds as it is. A module built with
the `simd128` target feature takes a SIMD128 version of the CDEF block
filter and the vector lanes of the inverse transforms (the loop filter and
the interpolation filters are loops the compiler vectorises there). There
is no `std::thread::spawn` on that target — a thread is another instance of
the module on shared memory, which only the page can create — so an
embedder that has threads registers them once with
`av1::parallel::set_runner` (a [`Runner`](src/parallel.rs): typically a
pool of Workers it keeps), and the decoder runs its parallel work through
it instead of on threads of its own (`Runner::start`, optional, is what
frames in flight need of it).

Throughput (`examples/av1_decbench.rs`, best of three, on the machine this was
written on — an AMD Ryzen 9 9950X), on 30-frame streams of camera
footage from this crate's encoder at quantiser 100, one tile column or four:

| stream | before | 1 thread | 4 threads |
|---|---|---|---|
| 1280x720, one tile | 25.9 MP/s | 59.1 MP/s | 74.6 MP/s |
| 1280x720, four tile columns | 25.7 MP/s | 57.0 MP/s | 86.1 MP/s |
| 1920x1080, one tile | 29.9 MP/s | 71.3 MP/s | 95.0 MP/s |
| 1920x1080, four tile columns | 29.6 MP/s | 71.6 MP/s | 114.4 MP/s |

On AOM test vectors (352x288, more of the toolbox, loop
restoration, film grain), one thread: av1-1-b8-02-allintra 7.4 → 12.0 MP/s,
av1-1-b10-00-quantizer-10 8.0 → 16.7, av1-1-b8-23-film_grain-50 11.8 → 25.2.
With the SIMD versions off (`AV1_NO_SIMD=1`) the 1080p one-tile stream
decodes at 34.7 MP/s on one thread.

The first column is the decoder before this work (single-threaded scalar).
Frames are still decoded one after another; no frame-parallel decoding.

The planes the post-filters write (`CdefFrame` copied from `CurrFrame`,
`LrFrame` from `CdefFrame`) come from a pool of the planes that frames no
longer need: reference frames pushed out of every slot, and `CurrFrame`
and `CdefFrame` once they have been filtered. Measured against the
decoder without it, the two builds taking turns (best of 60 decodes each,
megapixels a second, on an Apple silicon machine shared with other work):
neither faster nor slower.

| stream | threads | without | with |
|---|---|---|---|
| mandelbrot 1280x720 (SVT-AV1, CRF 35, 60 frames) | 1 / 4 | 196.5 / 387.1 | 196.5 / 386.4 |
| mandelbrot 1920x1080 | 1 / 4 | 205.9 / 445.6 | 204.5 / 450.2 |
| mandelbrot 1280x720 10-bit | 1 / 4 | 219.8 / 450.0 | 218.1 / 448.8 |
| av1-1-b8-02-allintra | 1 | 23.9 | 23.8 |
| av1-1-b10-00-quantizer-10 | 1 | 39.1 | 38.8 |

Recycling `CurrFrame` too (zeroed, as it must start) was 4 % to 7 %
slower at one thread and at four: zeroing a recycled plane costs more than
the zeroed pages the system hands out. On this machine allocation was not
a cost to begin with (the system time of a decode is about 1 % of it).

## What it encodes

Profiles 0, 1 and 2 — 4:2:0, 4:2:2, 4:4:4 and monochrome
(`Config::chroma`; `Config::monochrome` as an AVIF alpha plane is coded),
8, 10 and 12 bits, the profile following from the two (4:2:0 and
monochrome at 8 and 10 bits profile 0, 4:4:4 profile 1, 4:2:2 and anything
at 12 bits profile 2; RGB, the identity matrix, with 4:4:4) — one temporal
unit per frame shown (in alt-ref groups frames are coded out of display
order, some hidden and shown later), any size from 1x1 up (frames wider
than 4096 are coded in the tile columns they need). [`av1::y4m`](src/y4m.rs)
reads and writes Y4M in each of these formats (`C420`, `C422`, `C444`,
`Cmono`, with `p10` / `p12`, `mono10` / `mono12`), and the examples take
them:

- **Rate-distortion search** (`encoder::rdo`). Every decision — the
  partition of each block (none, horizontal, vertical, split; 4x4 at the
  slowest speeds), the block's modes, its transform depth under
  `TX_MODE_SELECT`, each luma transform block's type, skipping the residual
  — is made by *coding* the candidates with the tile walker in counting
  mode (each symbol's cost read from the CDFs in force, nothing written,
  nothing adapted), measuring the reconstruction's squared error against
  the source, and keeping the least `SSE + lambda * bits` (lambda a fixed
  multiple of the squared quantiser step; the chroma planes' squared error
  weighted 1.5 (U) and 3 (V) against luma's, see below). Region snapshots
  put the state back between candidates; a decision log replays the chosen sub-decisions
  when the caller codes the winner, so nothing is searched twice. The
  candidates come from a cheap preselection: intra modes by SATD (with the
  mode's rate from the CDFs), inter modes per reference — NEARESTMV, the
  NEARMV entries of the reference list, GLOBALMV, and NEWMV with a
  full-sample search then half- and quarter-sample refinement. From speed
  4 the search is pruned: the motion search starts from the best of the
  predicted, stacked and neighbouring vectors (a per-reference grid of the
  vectors found so far) and descends in square steps; a partition
  candidate is abandoned as soon as what it has coded costs more than the
  best before it (lossless, every speed); a block whose whole coding costs
  little per sample (a threshold in units of lambda) is not split;
  transform depths are tried with the DCT and the types searched only at
  the best depth, the types in their order of usefulness until one does
  not beat the best before it; the halves of a block (horizontal and
  vertical partitions, 16x16 blocks only at speeds 4 and 5) search only
  the references the block whole chose.
- **Intra**: all the directional and smooth modes, Paeth; **chroma from
  luma** (alphas fitted to the source); **palettes** (up to eight colours,
  their indices coded in the normative wavefront order) and **intra block
  copy** (`Tools::intrabc`, speeds 0 to 8) on frames that look like screen
  content; **filter intra** (`Tools::filter_intra`, off: below). Intra
  block copy, in key frames: a block of 8x8 or more copied from earlier in
  the frame, from where the stack's vectors point or where its source
  appears again exactly (a hash of the source's 8x8 blocks at every
  position, the nearest matches first), each vector held to the
  specification's `is_mv_valid` (and, in the wavefront search, to what the
  wavefront has reconstructed); such a frame codes no in-loop filters.
  Filter intra: the five recursive filters of `DC_PRED` blocks of 32x32 or
  less ranked with the other modes by SATD and rate.
- **Alt-ref groups** (`Config::altref`, 16 frames by default; speeds 0 to
  8, `encoder::gop`): the frames are coded in a pyramid. The last frame of
  a group first, hidden (`show_frame` = 0), then the frame half way to it,
  hidden, and so on down to stretches of one or two frames, coded as they
  are shown; each hidden frame is shown in its turn by a
  `show_existing_frame` header of a few bytes. Every frame between two
  coded ones refers to the nearest on both sides (`refresh_frame_flags`
  keeps the hidden frames still to be shown, the group's anchor and the
  latest frames; the order hints place them), and each is coded finer the
  more frames refer to it: a key frame at a quantiser step 1.2 octaves
  below the configured quantiser's, a group's alt-ref frame 0.6 below, its
  top layer 0.5 above, the layers between evenly spaced. The encoder holds
  a group's frames until the last of them is given, so the first temporal
  unit comes out a group late (16 frames), then one a frame;
  `Encoder::flush` hands out the rest, and a stream may end anywhere in a
  group. Each group is coded in the call that completes it, so a caller
  that times its calls (a live pipeline) sees one call in 16 take the
  group's time; `Config::altref` = 0 codes every frame as it comes, for
  16 % more bits (below).
- **Inter**: in a group, up to four references searched (the nearest
  frame on each side, the farthest after, the second nearest before; the
  nearest on each side from speed 4), **compound prediction** of LAST with
  a frame after it (and with LAST2, LAST3 or GOLDEN), **skip mode** (the
  nearest frames on each side with their predicted vectors, no residual),
  and the **motion field projected** from the references
  (`use_ref_frame_mvs`: their vectors among the candidates). The block's
  best inter candidate is then tried with the **inter tools**
  (`encoder::inter`): **switchable interpolation filters** (smooth and
  sharp, and with `Tools::dual_filter` a filter per direction), **OBMC**,
  **local warped motion**, **inter-intra** (each smooth mode, then the
  wedge whose blend of the two predictions is nearest the source),
  **distance-weighted**, **difference-weighted** and **wedge compound**;
  and **global motion** (`encoder::gm`: each textured block of the frame
  matched in the reference, coarse to fine, a rotation and zoom — or an
  affine model, or a translation — fitted to the matches by random sample
  consensus, kept when it is the motion of most of the frame) gives the
  GLOBALMV blocks a warp. Each variant is *probed* — its mode info coded
  in counting mode and the block predicted by the decoder's own syntax and
  prediction, its luma SATD plus the multiplier times its bits — and the
  two that probe cheapest of those cheaper than the candidate are
  trialled. Which are on at which speed is below. Without
  groups (`Config::altref` = 0, and speeds 9 and 10), the last two frames
  and a golden frame (the key frame, then every 16th frame coded with a
  finer quantiser) as references, each searched, compound prediction of
  LAST and another reference to speed 3, a key frame a step of 1 octave
  finer.
- **Transforms**: 4x4 to 64x64, every size; the full transform sets at the
  slow speeds (DCT, ADST, flipped ADST, identity and their mixes, chosen per
  transform block by trial coding), the reduced ones otherwise. A dead-zone
  quantiser, then **rate-distortion optimised quantisation**
  (`encoder::rdoq`): a transform block's levels are chosen again in the
  order they are coded, each among its level rounded to the nearest, one
  less and zero, by squared error plus lambda times its rate under the
  CDFs in force and the contexts it will be coded in; then the end of
  block goes where the block costs least (all zero included). At speeds 0
  to 3 in every block the search codes; at 4 to 8 only in the blocks it
  chose (its trials measure the dead-zone levels, with a wider dead zone
  standing in for the sparser levels the block will be coded with), which
  keeps most of the gain for a ninth of the time (below).
- **In-loop filters**, searched on a first pass's reconstruction (the frame
  is then coded again, replaying the first pass's decisions): the **loop
  filter** levels (per plane), **CDEF** (each candidate strength applied
  with the decoder's own filter, eight (luma, chroma) pairs chosen greedily
  against the cost of signalling them, each 64x64 block taking its best)
  and **loop restoration** (speeds 0 to 3; per unit: a Wiener filter
  fitted to the source by least squares, or a self-guided filter with
  fitted projection weights, or none; measured with the decoder's own
  restoration).
- **Quantiser**: fixed (`Config::quantizer`, the `base_q_idx` 1–255 of
  plain inter frames, and in a group the middle of its pyramid, each frame
  offset from it by its role as above), or **average-bitrate rate
  control** to a target number of bits per frame
  (`Config::target_bits_per_frame`): each frame's quantiser is planned over
  the next frames with a rate model per frame class refitted after every
  frame, the complexity measured on the source before coding (so a scene
  cut is priced before it is coded), what earlier frames over- or
  under-spent repaid over 12 frames; in groups, a group at a time, with a
  model for each layer of the pyramid, the complexities measured on the
  group's sources. Over 10-second clips it lands within a few percent of
  the rate (below).
- **Tiles and threads**: `Config::tile_cols_log2` tile columns, coded in
  parallel on `Config::threads` threads (the stream is the same as coding
  them in turn). From speed 4 the decision search of each tile also runs
  in a **wavefront** of superblock rows on the threads (each row two
  superblocks behind the one above, with CDFs of its own; the frame is
  then coded sequentially with the decisions), and the filter searches run
  on them too; the stream does not depend on the thread count.
- **Colour and HDR signalling**: `Config::color` is written into the
  sequence header's `color_config()` (primaries, transfer, matrix, range,
  chroma sample position; `color_description_present_flag` when any code
  point is specified), and `Config::hdr` as `METADATA_TYPE_HDR_CLL` and
  `METADATA_TYPE_HDR_MDCV` metadata OBUs after the sequence header of every
  key frame — 10-bit PQ (HDR10) and HLG. The decoder reports both on every
  `Frame` (`color`, `hdr`) and through `Decoder::color_info` /
  `Decoder::hdr_metadata`.
- **Key frames on demand**: `Encoder::force_keyframe()` makes the next frame
  a key frame (with its sequence header, so a decoder can start there)
  without resetting the encoder; the interval restarts from it.
- **Lossless**: `Config::quantizer` 0 (without rate control) codes every
  frame at quantiser index 0 — every residual through the Walsh-Hadamard
  transform, the frame header without loop filter, CDEF, restoration or
  transform mode — and every frame decodes to the source, sample for
  sample, in every format above. The forward Walsh-Hadamard transform is
  the exact inverse of the specification's inverse (a chain of lifting
  steps, undone in reverse), held to it at 8, 10 and 12 bits; the search
  then minimises bits alone. foreman (352x288, 4 frames, speed 6) codes in
  37 % of its raw size.
- **Film grain** (`Config::film_grain`, off by default; `encoder::grain`):
  each frame is denoised before it is coded and the grain taken out is
  signalled (`film_grain_params`) for the decoder's synthesis to put back.
  The noise is measured per plane on 16x16 blocks, about each block's own
  plane fit or, where that is clearly less, in its difference from the
  frame before (grain is fresh every frame, so where the picture holds
  still that difference is noise and texture cancels): its strength by
  intensity (the flat blocks' in each bin) becomes the scaling points, its
  correlation, fitted by least squares, the lag-3 auto-regressive filter,
  and the scaling is set so that the synthesis's grain for those
  coefficients (measured by running the same filter on white noise) has the
  measured strength. An adaptive local Wiener filter (5x5, by the noise
  at each intensity) denoises. The reconstruction, and what a decoder
  outputs, carries the synthesised grain. On akiyo and foreman with
  synthetic grain added (correlated Gaussian noise of strength 3.2, fresh
  every frame, stronger in the mid-tones; 30 frames, speed 6): 30 % and
  24 % fewer bytes at quantiser 80, 6.5 % and 9.7 % at 140; the synthesised
  grain's strength 2.6 and 3.8 (luma standard deviation, libdav1d's output
  with grain against without); the picture under the grain 1.1 dB (akiyo)
  and 0.0 dB closer to the clean clip than the same frames coded with their
  grain at 80. PSNR against the grainy source falls, as it must: the grain
  synthesised is not the source's.
- The arithmetic encoder is the exact inverse of the decoder's symbol
  decoder (derived from 8.2.6).

How it works: the encoder runs the decoder's own tile walker in encode
mode. Before each syntax element the walker asks the encoder for its
decision (partition, modes, motion vector, the quantised levels of a
transform block — taken with the prediction already in place), then codes
it with exactly the contexts and CDF adaptation it decodes with, and
reconstructs exactly as it decodes. The frame header is written, then
parsed back by the decoder's header parser, and the in-loop filters and
reference update are the decoder's. So the encoder's reconstruction is,
by construction, what a decoder outputs — and `tests/encode.rs` checks
that every temporal unit decoded by a fresh decoder equals it, sample for
sample.

### How well, and how fast

`examples/av1_rdcurve.rs` is the measurement harness: it encodes Y4M clips at
several quantisers, checks every temporal unit against a fresh decoder,
and reports the Bjøntegaard delta rate (BD-rate: the change in bits at the
same PSNR; negative is better) against an earlier run.
`examples/av1toy4m.rs` makes clips from AV1 streams. The figures below are
on four natural clips — three 640x360 crops of camera footage (C012, C003,
C019) and the 352x288 source of `av1-1-b8-02-allintra` — 10 frames each
(a key frame then nine inter frames), quantisers 64, 96, 128, 160 and 192.

Each tool as it went in, BD-rate (PSNR-Y) against the encoder before it:

| step | BD-rate |
|---|---|
| rate-distortion search: partitions (none, split, horizontal, vertical), modes, skip, transform depth and types by trial coding (against the first, greedy encoder) | −45.3 % |
| … the transform types searched only on the chosen candidate, its decisions replayed; lambda tuned | −4.0 %, −1.9 % |
| CDEF search | −7.0 % |
| loop filter level search | −1.3 % |
| LAST2 and golden references | −1.4 % |
| chroma from luma | −0.3 % (−1.8 % PSNR-YUV) |
| loop restoration | −1.1 % |
| compound prediction | −0.8 % |
| palettes (a synthetic screen-content clip; natural clips unchanged) | −69 % |

What each part of the search is worth at speed 4, measured by turning it off
(BD-rate of the encoder without it): horizontal and vertical partitions
+6.8 %, the per-block transform type search +3.1 %, trying the skip flag
+3.4 %, more than one inter candidate +3.4 %, more than one intra candidate
+1.3 %, transform depths +1.4 %.

All together, the default speed (4) is **−50.8 %** BD-rate against the
first encoder. `Config::speed` trades it for time (`Tools::for_speed` is
what each speed switches; times are the encoder's, one thread, on the
same clips; the encoder kernels' SIMD and the abandoned partition
candidates made every speed faster than the first table here, speed 4
1.9x at the same BD-rate):

| speed | BD-rate against speed 4 | time against speed 4 | megapixels/s |
|---|---|---|---|
| 0 | −3.0 % | 2.0x | 0.081 |
| 2 | −1.2 % | 1.44x | 0.11 |
| **4** (default) | — | 1x | 0.16 |
| 5 | +3.3 % | 0.50x | 0.33 |
| 6 | +12.0 % | 0.16x | 1.0 |
| 7 | +19.3 % | 0.11x | 1.4 |
| 8 | +27.3 % | 0.09x | 1.8 |
| 9 (no search; filters searched) | +60.2 % | 0.06x | 2.8 |
| 10 (the first encoder) | +103.6 % | 0.03x | 5.0 |

Speed 6 against the speed 6 before the pruned search: 0.20x the time at
+1.1 % on these clips; on three 1080x720 clips (10 frames, the same
quantisers) 0.17x the time at −1.6 %. At 1080x720 it codes 1.6
megapixels/s on one thread (2.0 frames/s), 2.9 on two, 4.7 on four and
5.6 on eight — the same stream on each.

Against rav1e 0.8.1 (run as a black box, its defaults otherwise) on five
standard clips — foreman, mobile and akiyo (352x288), ducks_take_off and
park_joy (1280x720) — the first 30 frames, quantisers 64 to 192, a key
frame every 240, one thread, the same speed in both (BD-rate of
rivet-av1 against rav1e, PSNR-Y and the 6:1:1 PSNR-YUV; before: develop at
03054ad), at speed 6:

| clip | PSNR-Y before | PSNR-Y | PSNR-YUV before | PSNR-YUV |
|---|---|---|---|---|
| foreman | +36.7 % | +29.6 % | +53.8 % | +45.5 % |
| mobile | +40.2 % | +33.7 % | +71.0 % | +61.3 % |
| akiyo | +43.6 % | +43.9 % | +61.1 % | +58.3 % |
| ducks_take_off | +10.1 % | +5.4 % | +31.2 % | +24.2 % |
| park_joy | +7.6 % | +3.9 % | +18.8 % | +14.8 % |
| **average** | **+27.6 %** | **+23.3 %** | **+47.2 %** | **+40.8 %** |

At speed 8 the average went from +46.7 % / +65.5 % to +37.1 % / +54.4 %.
Against the encoder before, on the same clips: speed 6 −3.1 % PSNR-Y and
−4.0 % PSNR-YUV for 1.04x the time; speed 8 −5.7 % and −6.1 % for 1.05x;
speed 4 −2.7 % and −4.2 % for 1.2x (times interleaved, one encode at a
time). Of speed 6's, weighting chroma alone is +0.1 % and −1.1 %;
rate-distortion optimised quantisation in every trial as well (speeds 0
to 4) would be −3.7 % and −4.7 %, but for 1.3x the time. akiyo, a still
talking head, is the one clip that loses PSNR-Y (+1.4 % at speed 6), to
the chroma weights.

Why PSNR-YUV trails PSNR-Y by so much against rav1e: rav1e codes chroma
with a much finer quantiser than luma — its frame headers carry an AC
delta-q of +5 to −39 for U and −8 to −57 for V, the finer the coarser the
frame (read back from its streams of foreman with rivet-av1's own header
parser) — so at the same luma quality it spends more on chroma and has
1.5 to 2.5 dB more of it in its inter frames. rivet-av1 codes chroma at
the luma quantiser. Finer chroma quantisers move rivet-av1 along the same
trade, not past it (all U and V offsets −8: +1.8 % PSNR-Y, −0.8 %
PSNR-YUV; offsets like rav1e's: +6.9 % and −2.2 %), and so do the
chroma weights in the search (against counting chroma as luma: U and V
3, +4.0 % and −3.6 %; U and V 2, +2.3 % and −3.1 %); U 1.5 and V 3
(+2.0 % and −3.2 %) are where the two measures together gained the
most.

(The speed table, and the comparison with rav1e just above, were measured
with every frame coded in display order, before alt-ref groups.)

**Alt-ref groups, against rav1e.** The same five clips, 30 frames,
quantisers and rav1e (with its own frame reordering); BD-rate of
rivet-av1 against rav1e, PSNR-Y (PSNR-YUV), frames in display order (the
encoder above, with rate-distortion optimised quantisation) and in
alt-ref groups:

| speed | frames in display order | alt-ref groups |
|---|---|---|
| 4 | +11.5 % (+26.5 %) | **−9.3 %** (−3.7 %) |
| 6 | +23.3 % (+40.8 %) | **−0.1 %** (+6.5 %) |
| 8 | +37.1 % (+54.4 %) | **+14.0 %** (+17.9 %) |

Per clip at speed 6:

| clip | frames in display order | alt-ref groups |
|---|---|---|
| foreman | +29.6 % (+45.5 %) | +13.5 % (+20.0 %) |
| mobile | +33.7 % (+61.3 %) | −6.0 % (+2.8 %) |
| akiyo | +43.9 % (+58.3 %) | +5.0 % (+8.4 %) |
| ducks_take_off | +5.4 % (+24.2 %) | −8.9 % (−0.4 %) |
| park_joy | +3.9 % (+14.8 %) | −4.0 % (+1.7 %) |

Against display order: −16.1 % (−21.2 % PSNR-YUV) at speed 6, −14.1 %
(−20.3 %) at speed 8, −15.8 % (−20.7 %) at speed 4. The time, one encode at a
time interleaved (user CPU time, 30 frames at quantiser 128, before the
rate-distortion optimised quantisation, which both sides now have): at
speed 6 0.79x to 1.02x the time of display order (fewer bits to search
and code in the coarser frames, two more references to search), 1.2x to
1.7x rav1e's; at speed 8 1.0x to 1.1x, 0.85x rav1e's; at speed 4 0.9x
to 1.2x, 4.7x to 7.1x rav1e's.

**The same streams, faster.** The encoder's kernels and the work its
search repeated, with every decision left as it was — each of the 75
streams below (three speeds, five clips, five quantisers, 30 frames) is
byte for byte what it was. Seconds of CPU time per clip (the five
quantisers together), one thread, each encode run alone beside the
others' (rivet-av1 before and after interleaved; the best of two runs),
rav1e 0.8.1 at the same speed:

| speed | clip | before | after | after / before | rav1e | before / rav1e | after / rav1e |
|---|---|---|---|---|---|---|---|
| 4 | foreman | 59.2 | 44.6 | 0.75x | 13.4 | 4.41x | 3.32x |
| 4 | mobile | 84.7 | 64.6 | 0.76x | 18.4 | 4.59x | 3.50x |
| 4 | akiyo | 21.8 | 15.8 | 0.72x | 5.8 | 3.77x | 2.72x |
| 4 | ducks_take_off | 830.6 | 622.0 | 0.75x | 130.8 | 6.35x | 4.76x |
| 4 | park_joy | 816.9 | 598.8 | 0.73x | 136.0 | 6.01x | 4.40x |
| **4** | **all** | **1813** | **1346** | **0.74x** | **305** | **5.96x** | **4.42x** |
| 6 | foreman | 9.7 | 7.6 | 0.78x | 8.2 | 1.19x | 0.93x |
| 6 | mobile | 13.9 | 10.7 | 0.77x | 11.6 | 1.20x | 0.93x |
| 6 | akiyo | 4.5 | 3.4 | 0.77x | 3.4 | 1.33x | 1.02x |
| 6 | ducks_take_off | 127.4 | 95.7 | 0.75x | 78.3 | 1.63x | 1.22x |
| 6 | park_joy | 117.5 | 88.4 | 0.75x | 88.2 | 1.33x | 1.00x |
| **6** | **all** | **273** | **206** | **0.75x** | **190** | **1.44x** | **1.09x** |
| 8 | foreman | 5.3 | 4.2 | 0.80x | 7.0 | 0.77x | 0.61x |
| 8 | mobile | 7.2 | 5.7 | 0.79x | 10.1 | 0.72x | 0.57x |
| 8 | akiyo | 2.8 | 2.1 | 0.75x | 2.9 | 0.98x | 0.73x |
| 8 | ducks_take_off | 68.2 | 53.1 | 0.78x | 67.4 | 1.01x | 0.79x |
| 8 | park_joy | 65.3 | 50.6 | 0.77x | 75.4 | 0.87x | 0.67x |
| **8** | **all** | **149** | **116** | **0.78x** | **163** | **0.92x** | **0.71x** |

On all 16 threads of the same machine (an Apple M3 Max; wall-clock
seconds, one encode at a time, both encoders given 16 threads; rav1e
gains little from them here — one tile, 30 frames):

| speed | before | after | after / before | rav1e | after / rav1e |
|---|---|---|---|---|---|
| 4 | 1648 | 1229 | 0.75x | 290 | 4.24x |
| 6 | 77.9 | 62.5 | 0.80x | 179 | 0.35x |
| 8 | 45.3 | 38.3 | 0.84x | 154 | 0.25x |

Speeds 5 and up search in a wavefront of superblock rows and so use the
threads; speed 4 searches a one-tile frame on one thread (only its filter
searches use the others).

Where the time went, and what was done about it (sampled profiles,
park_joy and foreman, speeds 4, 6 and 8; shares of the encoder's time
before):

- **The coefficients' symbols in trial coding** (12 to 25 %: the reference
  reader of `decoder::residual`, every context derived from `Quant` with
  its edge tests, every symbol through `Coder`). The encoder now codes the
  levels and signs with a writer of its own (`levels_encode`): the
  decode-only reader's scan maps and padded level map for the contexts,
  the costs from a table taken once per block; `levels_checked` holds it to
  the reference on every transform block of a test's encodes.
- **The forward transform** (11 to 15 %): its rows are now lanes, as its
  columns were (the same sums in the same order, so the same values,
  tested), each size compiled on its own, and the quantiser applied as the
  coefficients come out, the end of block found from an inverse scan
  (`forward_quantise`). Over half the transforms at speed 4 were of
  residuals already transformed — the search codes a block for each
  candidate that holds it, then again for real at every level above it —
  and a transform block's levels are now kept for the next time the same
  residual is quantised so.
- **Memory cleared for nothing** (5 to 10 %): 8 KB for every intra
  prediction, 4 KB for every quantised block, 2 KB for every inverse
  transform.
- **The motion search** (15 to 45 % at speed 4, with four references):
  NEON SAD for 4-wide blocks, the limit checked every four rows (the result
  does not depend on where), four candidates of the exhaustive search's
  grid measured at once, and candidates past the reference's edge in three
  runs per row instead of a sample at a time.
- **Loop restoration's search** (10 to 25 % at speed 8): the Wiener fit's
  cross-filtered samples worked out once instead of once for each of the
  seven lines they lie on, and the self-guided box filter's neighbourhood
  sums written out per row parity (the decoder's too).
- The encoder reconstructs through the decoder's `reconstruct_decode`
  (from the end of block), the directional intra predictor's tests of the
  angle are out of its loops, and trial coding's symbol costs are inline.

What is left is flat — at speed 6 no function above 9 % (the level
writer's counting, the inverse transform, SATD and SAD, the motion
search's sub-sample predictions). Speed 4 stays 4.4x rav1e's time
because its decisions cost that much: an exhaustive full-sample search of
four references, every sub-sample candidate predicted with the normative
filters, the full transform type search at every depth and
rate-distortion optimised quantisation in every trial; the gap there is in
what the search tries, not in how fast it tries it.

**The tiers, at rav1e's time.** Then each speed's search was set to what
rav1e spends at the same speed: speed 4 searches far less, speed 8 more,
speed 6 a little less and differently (`Tools::for_speed`; each stream
still decodes to the encoder's reconstruction). The same five clips,
30 frames, quantisers 64 to 192, one thread; BD-rate of rivet-av1
against rav1e 0.8.1 at the same speed, PSNR-Y (PSNR-YUV); seconds of
CPU time per clip summed over the five clips, each encode run alone
beside the others, rivet-av1 and rav1e interleaved, the best of three
(speed 4's before: the best of two, scaled by rav1e's time of the same
run):

| speed | BD-rate before | BD-rate | time before / rav1e | time / rav1e | seconds (rav1e) | against before |
|---|---|---|---|---|---|---|
| 4 | −9.3 % (−3.7 %) | **−0.3 %** (+5.1 %) | 4.33x | **0.96x** | 296 (309) | +9.7 % (+9.0 %), 0.22x the time |
| 6 | −0.1 % (+6.5 %) | **−0.5 %** (+5.0 %) | 1.10x | **0.89x** | 170 (191) | +0.2 % (−0.7 %), 0.81x |
| 8 | +14.0 % (+17.9 %) | **−1.2 %** (+4.2 %) | 0.72x | **0.96x** | 157 (163) | −12.1 % (−10.5 %), 1.34x |

Per clip, PSNR-Y (PSNR-YUV) and time against rav1e:

| clip | speed 4 | speed 6 | speed 8 |
|---|---|---|---|
| foreman | +8.6 % (+12.4 %), 0.75x | +9.1 % (+13.6 %), 0.72x | +6.8 % (+11.4 %), 0.78x |
| mobile | −3.4 % (+3.8 %), 0.87x | −3.0 % (+3.4 %), 0.79x | −0.9 % (+4.8 %), 0.85x |
| akiyo | +5.4 % (+8.7 %), 0.55x | +5.1 % (+8.3 %), 0.68x | +2.9 % (+6.2 %), 0.75x |
| ducks_take_off | −8.3 % (−0.4 %), 1.10x | −9.3 % (−1.1 %), 1.04x | −9.6 % (−1.3 %), 1.11x |
| park_joy | −3.8 % (+1.3 %), 0.87x | −4.4 % (+0.7 %), 0.80x | −5.2 % (+0.0 %), 0.87x |

So at rav1e's time rivet-av1 is at parity in PSNR-Y at every tier, and
4 % to 5 % behind in PSNR-YUV: the chroma that rav1e's finer chroma
quantisers buy (above) is the gap that is left, at every speed alike.

What each tier now does (from speed 4 down to speed 8; speeds 0 to 3 are
as they were):

- **speed 4**: the pruned search of speed 6 (the stepped motion search,
  the transform depths with the DCT, rate-distortion optimised
  quantisation in the blocks coded, the wavefront) with the 16x16 blocks
  also tried halved, the halves searching only the references the block
  whole chose; the nearest frame on each side searched in a group; three
  inter candidates trialled and one intra; the transform types searched;
  no loop restoration.
- **speed 6**: the same without the halves and without the transform
  type search.
- **speed 8**: the same with two inter candidates (the transform depths
  are now searched at speed 8 too).
- At speeds 4 to 8 a block is not split when coding it whole costs less
  than 1.0 lambda per sample (0.15 for blocks of 32x32 and more).

What each change was worth where it was measured (BD-rate PSNR-Y
(PSNR-YUV) against the tier without it; times from runs contending with
each other, to a few percent):

| change | where | BD-rate | time |
|---|---|---|---|
| transform depths searched | speed 8 | −8.8 % (−6.7 %) | 1.21x |
| two inter candidates trialled, not one | speed 8 | −3.6 % (−3.5 %) | 1.06x |
| three, not two | speed 6 | −1.6 % (−1.7 %) | 1.06x |
| … not two | speed 8 | −1.8 % (−1.8 %) | 1.08x |
| the split threshold of blocks under 32x32 at 1.0 lambda, not 2.0 | speed 8; speed 6 | −1.8 % (−1.9 %); −1.6 % (−1.9 %) | 1.09x; 1.12x |
| loop restoration | speed 8; speed 6; speed 4 | −1.2 % (−0.9 %); −1.2 % (−1.1 %); −1.2 % (−1.0 %) | 1.10x; 1.10x; 1.03x |
| four references searched, not two | speed 8; speed 6 | −1.2 % (−1.1 %); −1.1 % (−0.9 %) | 1.19x; 1.15x |
| transform types searched | speed 6 | −1.2 % (−0.8 %) | 1.12x |
| the type search stopped at the first type that loses (`tx_type_early`) | speed 6 | −0.1 % (−0.2 %) | 0.98x |
| one intra candidate trialled, not two | speed 6 | +0.2 % (+0.1 %) | 0.99x |
| horizontal and vertical partitions | speed 8; speed 6 | −10.3 % (−9.5 %); −6.6 % (−7.1 %) | 2.10x; 2.49x |
| … their halves searching only the whole block's references (`rect_refs`) | speed 6 | +0.9 % (+0.9 %) | 0.89x |
| … of 8x8 blocks | speeds 4 and 6 | ±0 (the same streams) | 1.05x |
| … of 32x32 blocks | speed 4 | −1.2 % (−1.2 %) | 1.12x |
| … of 64x64 blocks | speed 4 | −0.6 % (−0.6 %) | 1.15x |
| the stepped motion search, not the exhaustive one | the speed 4 before | −0.3 % (−0.3 %) | 0.88x |

The trade at speed 4, against rav1e's speed 4 (times against rav1e's,
from runs contending with each other, to about 5 %):

| speed 4 | BD-rate | time |
|---|---|---|
| before (four references, the exhaustive motion search, every partition, the full transform search at every depth, rate-distortion optimised quantisation in every trial) | −9.3 % (−3.7 %) | 4.4x |
| speed 5 before | −4.5 % (+1.4 %) | 2.0x |
| speed 6's search, halves from 16x16 to 64x64, loop restoration | −3.7 % (+2.0 %) | 1.4x |
| … halves of 16x16 and 32x32 | −2.9 % (+2.6 %) | 1.2x |
| … no loop restoration | −1.8 % (+3.6 %) | 1.2x |
| **… halves of 16x16 only (speed 4)** | **−0.3 % (+5.1 %)** | **0.96x** |

PSNR-Y and the time at once were reached; PSNR-YUV at rav1e's time was
not (above). At speed 8 the next steps up cost more than rav1e's time
leaves: three inter candidates, −3.2 % (+2.2 %) for about 1.03x rav1e's
time; the halves of every block, −7.6 % (−2.5 %) for 2.3x.

**Groups at scene cuts**, measured and left out: a group ended at a scene
cut (a frame whose quarter-size thumbnail, motion-searched against the
one before, costs more than 1.2 times its own deviation from its block
means and 2.5 times the group's other frames: 0.75 at most for the
motion in the test clips, from 1.85 for a cut between any two of them),
the next group starting at the cut. On two clips made of the test clips
cut together (foreman, mobile and akiyo, 20 frames each; ducks_take_off
and park_joy, 20 frames each), speed 6: +5.0 % (+5.1 %) and +1.5 %
(+1.1 %); the cut frame also made a key frame, +3.6 % (−0.1 %) and +4.6 %
(+3.6 %). A pyramid that spans a cut already codes the frames after it
from the alt-ref frame beyond it (which is coded intra where it must be)
and those before it from the anchor; ending the group early only adds a
short group with its finely coded alt-ref frame.

Each step at speed 6, as it went in (before rate-distortion optimised
quantisation), BD-rate PSNR-Y against the step before (the five clips,
or the three 352x288 ones where it says so):

| step | BD-rate |
|---|---|
| the alt-ref groups before (groups of 8, one hidden frame, no quantiser hierarchy), against display order | +1.2 % |
| groups of 16 in a pyramid, the quantiser offset by layer (352x288 clips) | −10.6 % |
| compound prediction of a reference before with one after (the tile walker had coded every compound pair as unidirectional, so a pair with a frame after was read as LAST and LAST3) (352x288 clips) | −11.2 % |
| the motion field projected from the references (352x288 clips) | −1.3 % |
| skip mode | −0.4 % |
| the top layer 0.5 octave coarser (of 0.5, 0.7 and 0.9) | −1.5 % |
| **all, against display order** | **−16.6 %** (−21.5 % PSNR-YUV) |
| … at speed 4; at speed 8 | −16.3 % (−21.3 %); −14.4 % (−20.2 %) |
| display order (`Config::altref` = 0): the finer key frame and the projected motion field | −6.1 % (−8.1 %) |

Measured and left out: groups of 8 (+1.0 % against 16) and of 32 (+1.1 %);
three references searched (+0.6 % against four) and five (−0.3 %, for the
time of another search); four at speed 8 (−1.2 %, 13 % more time); groups
at speeds 9 and 10, which do not search the references after a frame
(+3 % to +7 %); and motion-compensated temporal filtering of the alt-ref
frames (each 16x16 block matched in up to three frames on each side,
samples averaged by how well their neighbourhood matches against the
noise measured in the frame; the alt-ref frame alone or every hidden
frame, several strengths): between −0.2 % and +0.5 %, so not kept. A
chroma delta-q growing with each frame's quantiser (U and V together, −8 %
or −16 % of the frame's quantiser index) trades as it does without
groups: +1.0 % PSNR-Y for −0.6 % PSNR-YUV, +2.1 % for −0.9 %.

**Filter intra and intra block copy**, each against the encoder without
it (BD-rate PSNR-Y / PSNR-YUV; times one encode at a time, interleaved,
best of two or three, quantiser 128):

| tool | speed 4 | speed 6 | speed 8 |
|---|---|---|---|
| filter intra, the five clips | −0.13 % / −0.09 %, 1.13x to 1.21x the time | −0.20 % / −0.23 %, 1.05x to 1.18x | −0.24 % / −0.12 %, 1.03x to 1.09x |
| … in key frames only | | −0.07 % / −0.09 % | +0.02 % / +0.12 % |
| intra block copy, the five clips | ±0 (the same streams: no frame looks like screen content) | ±0 | ±0 |
| intra block copy, a screen-content clip (key frame, 29 inter frames) | −6.7 % / −6.6 % | −4.6 % / −4.6 % | −4.0 % / −4.2 % |
| … all key frames (10) | −11.9 % / −13.0 % | −3.4 % / −6.4 % | −3.2 % / −5.9 % |
| … time on that clip | 0.89x | 0.92x | 1.01x |

The screen-content clip is synthetic (640x360, 30 frames: a code editor
of coloured text in a made-up bitmap font scrolling, a panel of buttons, a
gradient image, a moving pointer). Intra block copy is on at every speed
that searches; filter intra, which gains a fifth of a percent for 5 % to
20 % more time — less than a slower speed gains for it — is off.

**The inter tools.** What each is worth with all the others on (BD-rate
PSNR-Y / PSNR-YUV of the encoder without it against the encoder with all
of them, so positive is what the tool gains; the five clips, 30 frames,
quantisers 64 to 192, one thread), and the time it adds (user CPU time,
foreman and mobile, three quantisers, one encode at a time; the machine
was shared, so ±2 %):

| tool | speed 4 | speed 6 | speed 8 | time 4 / 6 / 8 |
|---|---|---|---|---|
| switchable filters (smooth, sharp) | +0.89 % / +0.83 % | +0.91 % / +0.86 % | +1.02 % / +0.78 % | (a) / 1.04 / 1.09x |
| … a filter per direction (dual) | −0.01 % / −0.03 % | −0.03 % / −0.04 % | +0.31 % / +0.26 % | 1.06x at 6 and 8 |
| OBMC | +0.71 % / +0.65 % | +0.32 % / +0.31 % | +0.57 % / +0.55 % | (a) / 1.01 / 1.02x |
| local warped motion | +0.51 % / +0.45 % | +0.38 % / +0.39 % | +0.52 % / +0.55 % | (a) / 1.01 / 1.00x |
| global motion | −0.01 % / −0.02 % | +0.10 % / +0.11 % | +0.22 % / +0.21 % | 1.06 / 1.04 / 1.04x |
| wedge and difference-weighted compound | +0.01 % / +0.02 % | −0.12 % / −0.14 % | +0.16 % / +0.11 % | 1.09 / 1.05 / 1.06x |
| distance-weighted compound | +0.06 % / +0.08 % | −0.08 % / −0.09 % | +0.09 % / +0.08 % | 1.01 / 1.00 / 1.01x |
| inter-intra (smooth and wedge) | +0.17 % / +0.18 % | +0.09 % / +0.02 % | +0.24 % / +0.23 % | 1.03 / 1.04 / 1.05x |
| **all** (against none) | **−3.08 %** / −2.97 % | **−2.85 %** / −2.85 % | **−4.90 %** / −4.38 % | 1.13 / 1.30 / 1.52x |

Times at speed 4 are foreman's alone; (a) the filters, OBMC and local
warps together, 1.06x.

Each alone at speed 6, against none: the filters −1.13 %, OBMC −0.64 %,
local warps −0.77 %, global motion −0.24 %, wedge and difference-weighted
−0.36 %, distance-weighted −0.35 %, inter-intra −0.45 % (PSNR-Y). On by
default (`Tools::for_speed`): the switchable filters, OBMC and local warps
at speeds 0 to 8, with distance-weighted compound and inter-intra to speed
4 — where the slower speed's next step costs more time for each percent
than they do. Against the encoder before them (same clips, BD-rate
PSNR-Y / PSNR-YUV; time interleaved, every clip, all five quantisers):

| speed | BD-rate | time | against rav1e, before | against rav1e, after | time against rav1e (from the table above times this one's) |
|---|---|---|---|---|---|
| 4 | −3.03 % / −2.93 % | 1.11x | −9.3 % / −3.7 % | −12.1 % / −6.6 % | 4.9x |
| 6 | −2.12 % / −2.09 % | 1.05x | −0.1 % / +6.5 % | −2.3 % / +4.2 % | 1.14x |
| 8 | −3.58 % / −3.22 % | 1.13x | +14.0 % / +17.9 % | +9.5 % / +13.6 % | 0.80x |

Per clip at speed 6 against rav1e (PSNR-Y / PSNR-YUV): foreman +12.1 % /
+18.4 % (from +13.5 % / +20.0 %), mobile −12.6 % / −4.2 %, akiyo +3.8 % /
+7.2 %, ducks_take_off −9.3 % / −0.6 %, park_joy −5.5 % / +0.2 %. The
tools gain about as much in PSNR-YUV as in PSNR-Y — they improve the
prediction of every plane alike — so they close neither foreman's gap
(1.2 % of its 13.5 %) nor the gap between PSNR-Y and PSNR-YUV, which is
in how chroma is quantised (above).

Rate control, `examples/av1_ratetest.rs`: 10-second clips (300 frames,
640x360, a key frame every 240) of a still frame, a fast pan, a clip with
fresh noise in every frame and a different scene every second, at 100,
300 and 1000 kb/s, speed 6, in alt-ref groups (the sources are 640x360
crops of park_joy and ducks_take_off; the camera crops of the table
before are not in the repository):

| clip | 100 kb/s | 300 kb/s | 1000 kb/s |
|---|---|---|---|
| still | +1.5 % | +0.7 % | −67 % (quantiser 1: 326 kb/s is all it can use) |
| pan | +1.6 % | −1.4 % | −1.1 % |
| noise | +0.8 % | −0.5 % | −1.9 % |
| scene cuts | −0.2 % | −1.7 % | −2.3 % |

Speeds 4 and 8 land likewise, within 2.3 %, but for the still clip at
1000 kb/s and two rates at the floor of what a group costs at quantiser
255: the still clip at 100 kb/s (+8.1 % at speed 4, +7.3 % at speed 8) and
the pan at 100 kb/s (+4.4 % at speed 4). Frames coded in display order
land within 1.5 % on the same clips but for the still one (−2.7 % and
−4.2 % at 100 and 300 kb/s).

Not there yet:

- **Groups that follow the content**: a group is 16 frames whatever the
  frames are (cut only by a key frame); a scene cut inside a group, or a
  fast pan the alt-ref frame cannot be predicted across, is coded as if
  it were not there (ending groups at scene cuts cost 1 % to 5 % on cut
  clips: above). No temporal filtering of the alt-ref frames (built,
  measured at ±0, left out: above).
- **PSNR-YUV at rav1e's time**: 4 % to 5 % behind at speeds 4, 6 and 8
  (PSNR-Y at parity), the chroma that rav1e's finer chroma quantisers
  buy; foreman and akiyo trail at every tier, in PSNR-Y too.
- **Adaptive quantisation that pays**: `Tools::aq` codes a quantiser per
  superblock (`delta_qindex`) from a temporal-importance estimate (the
  previous frame's prediction of each superblock), −0.2 % to +0.2 % on the
  test clips; off by default. No segmentation.
- **Chroma quantiser offsets**: the frame header's delta-q for U and V is
  zero. Finer chroma alone trades PSNR-Y for PSNR-YUV (above); it may pay
  together with a quantiser hierarchy, which changes what each frame's
  chroma is worth to the frames after it.
- **Inter tools that pay at speed 6 and up**: global motion, wedge and
  difference-weighted compound, inter-intra and dual filters are built,
  tested and searched, but gain less than their time there (below); a
  vector searched for OBMC or a warp (the block's own vector is kept) and
  a cheaper probe would make them pay.
- **Film grain from a temporal denoiser**: the denoiser is spatial (it
  leaves part of a coarse grain in the picture coded), the strength of the
  grain is measured to within about 20 % (above), and every frame codes its
  parameters afresh (`update_grain` 1) where an unchanged set could be
  taken from a reference frame for a few bits.
- **Intra block copy for blocks under 8x8**, and copies that are not exact
  (the search finds where a block's source appears again exactly, and the
  stack's vectors).
- **Lookahead**: rate control plans from the frames already coded and,
  in groups, the group's sources (a still scene undershoots high rates: it
  cannot use the bits).

## How it is checked

- **AOMedia's AV1 test vectors**: bitstreams published with the MD5 of
  every frame the reference decoder shows. `tools/fetch-vectors.sh`
  downloads the 244 `av1-1-b8-*` and `av1-1-b10-*` streams (about 7 MB)
  and `tests/vectors.rs` decodes each and compares every shown frame.
  **244 of 244 pass**, every frame bit-exact:

  | group | what it exercises | pass |
  |---|---|---|
  | av1-1-b8-00, av1-1-b10-00 | quantiser 0–63, 8- and 10-bit (lossless included) | 128 / 128 |
  | av1-1-b8-01 | frame sizes 16 to 66 and 196 to 226, every combination | 100 / 100 |
  | av1-1-b8-02 | all-intra coding | 1 / 1 |
  | av1-1-b8-03 | frame size up and down, scaled references (Matroska) | 2 / 2 |
  | av1-1-b8-04, 05, 06 | CDF update, motion vectors, motion field projection | 3 / 3 |
  | av1-1-b8-16 | intra-only frames, intra block copy | 1 / 1 |
  | av1-1-b8-22 | temporal and spatial scalability (L1T2, L2T1, L2T2) | 5 / 5 |
  | av1-1-b8-23, av1-1-b10-23 | film grain | 2 / 2 |
  | av1-1-b8-24, av1-1-b10-24 | monochrome | 2 / 2 |

  Twelve small vectors are committed in [`tests/data`](tests/data/README.md)
  so `cargo test` checks real streams without the download.
- **The Argon conformance streams** (Argon Streams AV1 v2.1, AOMedia's
  decoder verification suite: streams designed for coverage of every
  syntax element and decoding process, in Annex B and section 5 formats,
  with reference MD5s of all output). `tests/argon.rs` runs them from a
  local copy (`ARGON_DIR`; 7 GB, not downloaded by the tests), taking each
  stream's options (format, operating point, all layers) from its
  reference command. **All 3 015
  base streams pass**, every output byte matching:

  | group | what it exercises | pass |
  |---|---|---|
  | profile0_core, profile1_core, profile2_core | the coverage streams, Annex B, per profile (0: 8/10-bit 4:2:0 and monochrome; 1: 4:4:4; 2: 4:2:2, 12-bit) | 764 / 764, 731 / 731, 894 / 894 |
  | profileN_core_special, profileN_not_annexb_special | the suite's "special" streams, Annex B and section 5 | 330 / 330 |
  | profileN_not_annexb | section 5 low-overhead streams | 37 / 37 |
  | profileN_stress | the suite's stress streams | 252 / 252 |
  | profile_switching | the suite's profile-switching streams | 7 / 7 |

  The suite also carries, per stream, reference output for each operating
  point (`layers/N`, about 75 000 more decodes of the same streams with
  layers dropped): the operating point 1 set was run, **2 763 / 2 763 pass**
  (`ARGON_LAYERS=1`); the others were not run for time. The
  error-resilience directories (no reference output) are not run.

  The large-scale-tile directories (tile list OBUs, 7.3) have reference
  output made with AOMedia's `lightfield_tile_list_decoder`: the frames
  the first N temporal units show are the anchor frames, and the output of
  each of the next L tile lists is written tile by tile (each tile as a
  picture of its own, in the list's order: the output frame of 7.3.1 cut
  at its tiles, without the samples no tile covers). `tests/argon.rs`
  decodes them that way and hashes the same bytes (`ARGON_FILTER=large`
  runs only these). **All 571 pass**:

  | group | pass |
  |---|---|
  | profile0_large_scale_tile, profile1_large_scale_tile, profile2_large_scale_tile | 140 / 140, 136 / 136, 139 / 139 |
  | profile0_large_scale_tile_special, profile1_large_scale_tile_special, profile2_large_scale_tile_special | 54 / 54, 54 / 54, 48 / 48 |
- **The encoder** (`tests/encode.rs`, and every run of
  `examples/av1_rdcurve.rs`): every temporal unit, decoded by a fresh decoder
  with the padding check on, equals the encoder's reconstruction exactly —
  key frames at several quantisers, inter frames, 10-bit, sizes from 1x1,
  several tile columns coded in parallel (the same stream as in turn), HDR
  signalling, forced key frames, 4:4:4, 4:2:2, 12-bit and monochrome,
  lossless (every frame the source, sample for sample), filter intra,
  intra block copy, film grain (the synthesised grain about as strong as
  the grain the source had); PSNR falls and size falls as the quantiser
  rises; inter frames cost less than the key frame; rate control lands
  near its target. The formats and tools were also decoded by libdav1d
  (through FFmpeg, as a black box) to the same output as this crate's
  decoder, sample for sample, at speeds 4, 6 and 8 (not part of the
  tests).
- **Malformed input** (`tests/fuzz.rs`, proptest): arbitrary bytes, and the
  committed vectors with bits flipped, bytes cut and garbage spliced in,
  decoded in debug builds (overflow checks on) — errors, never a panic.
- **SIMD**: each SIMD kernel (CDEF, the interpolation filters, the inverse
  transforms, the loop filter) against its scalar version on random input,
  on x86-64 (AVX2) in CI and on aarch64 (NEON) by hand (below); `AV1_NO_SIMD=1`
  turns the SIMD versions off. The test vectors and the Argon suite are
  run with the decoder's threads on (`AV1_DECODE_THREADS`).
- **Units**: the arithmetic encoder against the decoder over random symbol
  sequences with adapting CDFs (and the decoder's padding check); the
  forward transforms round-tripping through the normative inverse; the
  inverse DCT on DC-only blocks; the Walsh-Hadamard transform; bit and
  IVF round trips.

### NEON on ARM hardware

CI runs on x86-64 Linux only, so the NEON (aarch64) code paths are not tested
there. They are verified by hand on ARM hardware (an aarch64 Linux machine,
or Apple silicon) after a change to them and before a release:

```sh
cargo test --release
```

The kernel tests compare each NEON kernel with its scalar version on random
input, so the one run covers both.

## Provenance and licensing

Written from the specification's text; **no AV1 implementation's source was
read** — not libaom, not dav1d or rav1d, not rav1e, not SVT-AV1, not
FFmpeg. The tests run none of them either: they use only the
specification-derived checks above, round trips through this crate, and
published conformance bitstreams with their MD5s (data: the AOM test vectors
from AOMedia's public test-data bucket, the committed ones listed with their
source in [`tests/data/README.md`](tests/data/README.md), and the Argon
streams).

The speed work recorded in
[`docs/wasm-performance.md`](docs/wasm-performance.md) does run two of them,
as black boxes: rav1d as the timing reference, and libdav1d (through FFmpeg)
for the MD5s of the benchmark clips in `tools/wasm-perf`. Their source was
not read for it. Some of its optimisations are techniques commonly known
from other AV1 decoders (16-bit prediction intermediates, a DC-only
transform shortcut, a per-frame table of loop filter levels, loop filter
kernels per filter length that work on 16-bit lanes where the samples lie,
subpixel filters that take a block from one window of the reference and
multiply only by a filter's non-zero taps, an arithmetic decoder on a wide
bit window with branch-free reads of small CDFs, a padded level map for
coefficient contexts, a transform compiled per size and bounded by the end
of block; for the threads: entropy decoding apart from reconstruction,
superblock rows in a wavefront, post-filters following reconstruction by
rows, several frames decoded at once with the output delayed); each was
written from this crate's own specification-derived code and held to it
bit for bit (the threading from the dependencies the specification's
processes have on each other, [`docs/threading.md`](docs/threading.md)),
and what that document says about rav1d's internals comes from
profiles and general knowledge, not from its code. The
specification's tables (default CDFs, scans, quantiser lookups and
matrices, filter taps) are data and were transcribed by
[`tools/gen_tables.py`](tools/gen_tables.py) from the specification's own
source text into [`src/tables.rs`](src/tables.rs). See [NOTICE](NOTICE).

Licensed under the Open Encoding Attribution License 1.0
([LICENSE.md](LICENSE.md)): source-available, not open source.
