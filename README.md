# rivet-av1

[![CI](https://github.com/safewords/rivet-av1/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-av1/actions/workflows/ci.yml)

An **AV1** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from the *AV1
Bitstream & Decoding Process Specification* (AOMedia), not translated from
any other implementation. The decoder is bit-exact on **all 244** of
AOMedia's public AV1 test vectors and on all 3 015 of the Argon
conformance streams (the numbers are
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

Not there yet:

- **Large-scale tile decoding** (tile list OBUs, 7.3). Optional for
  conformance; reported as `Error::Unsupported`.
- **A frame-buffer pool.** Every frame allocates its buffers.
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

## What it encodes

Profile 0 (8- or 10-bit 4:2:0, or monochrome with `Config::monochrome` —
as an AVIF alpha plane is coded), one temporal unit per frame shown (in
alt-ref groups frames are coded out of display order, some hidden and shown
later), any size from 1x1 up (frames wider than 4096 are coded in the tile
columns they need):

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
  5 the search is pruned: the motion search starts from the best of the
  predicted, stacked and neighbouring vectors (a per-reference grid of the
  vectors found so far) and descends in square steps; a partition
  candidate is abandoned as soon as what it has coded costs more than the
  best before it (lossless, every speed); a block whose whole coding costs
  little per sample (a threshold in units of lambda, higher for blocks of
  16x16 and less) is not split; transform depths are tried with the DCT
  and the types searched only at the best depth.
- **Intra**: all the directional and smooth modes, Paeth; **chroma from
  luma** (alphas fitted to the source); **palettes** (up to eight colours,
  their indices coded in the normative wavefront order) on frames that look
  like screen content.
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
  nearest on each side from speed 7), **compound prediction** of LAST with
  a frame after it (and with LAST2, LAST3 or GOLDEN), **skip mode** (the
  nearest frames on each side with their predicted vectors, no residual),
  and the **motion field projected** from the references
  (`use_ref_frame_mvs`: their vectors among the candidates). Without
  groups (`Config::altref` = 0, and speeds 9 and 10), the last two frames
  and a golden frame (the key frame, then every 16th frame coded with a
  finer quantiser) as references, each searched, compound prediction of
  LAST and another reference to speed 5, a key frame a step of 1 octave
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
  to 4 in every block the search codes; at 5 to 8 only in the blocks it
  chose (its trials measure the dead-zone levels, with a wider dead zone
  standing in for the sparser levels the block will be coded with), which
  keeps most of the gain for a ninth of the time (below).
- **In-loop filters**, searched on a first pass's reconstruction (the frame
  is then coded again, replaying the first pass's decisions): the **loop
  filter** levels (per plane), **CDEF** (each candidate strength applied
  with the decoder's own filter, eight (luma, chroma) pairs chosen greedily
  against the cost of signalling them, each 64x64 block taking its best)
  and **loop restoration** (per unit: a Wiener filter fitted to the source
  by least squares, or a self-guided filter with fitted projection weights,
  or none; measured with the decoder's own restoration).
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
  them in turn). From speed 5 the decision search of each tile also runs
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
  it were not there. No temporal filtering of the alt-ref frames (built,
  measured at ±0, left out: above).
- **Adaptive quantisation that pays**: `Tools::aq` codes a quantiser per
  superblock (`delta_qindex`) from a temporal-importance estimate (the
  previous frame's prediction of each superblock), −0.2 % to +0.2 % on the
  test clips; off by default. No segmentation.
- **Chroma quantiser offsets**: the frame header's delta-q for U and V is
  zero. Finer chroma alone trades PSNR-Y for PSNR-YUV (above); it may pay
  together with a quantiser hierarchy, which changes what each frame's
  chroma is worth to the frames after it.
- Filter intra, intra block copy, OBMC and warped motion, wedge /
  difference-weighted compound, inter-intra, switchable interpolation
  filters, film grain parameters.
- **Profiles 1 and 2**, 12-bit, monochrome; lossless (quantiser 0 needs the
  forward Walsh-Hadamard transform).
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
  (`ARGON_LAYERS=1`); the others were not run for time. The large-scale-tile directories (tile
  list OBUs, unsupported) and the error-resilience directories (no
  reference output) are not run.
- **The encoder** (`tests/encode.rs`, and every run of
  `examples/av1_rdcurve.rs`): every temporal unit, decoded by a fresh decoder
  with the padding check on, equals the encoder's reconstruction exactly —
  key frames at several quantisers, inter frames, 10-bit, sizes from 1x1,
  several tile columns coded in parallel (the same stream as in turn), HDR
  signalling, forced key frames; PSNR falls and size falls as the quantiser rises;
  inter frames cost less than the key frame; rate control lands near its
  target.
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
