# Decoder speed in WebAssembly: the running table

The decoder is the core of [wasm-av1](https://github.com/safewords/wasm-av1),
whose reference for speed is rav1d (the Rust port of dav1d, built without
assembly, with that project's wasm SIMD128 kernels). This file records each
step taken to close the gap: what changed, and frames per second before and
after.

**How the numbers are taken.** wasm-av1 at `ee33e98` built on this crate
(`--no-default-features --features rivet-av1-pending,container`, fat LTO, no
`wasm-opt`), decoded in Node 26 (V8) on one thread on an Apple M3 Max; the
time inside the decode calls only, median of 7 runs after one warm-up, the
MD5 of every decoded plane checked against libdav1d's in the same run.
Other builds were running on the machine throughout (load average 4 to 12),
so read differences of a few percent as noise. `baseline` is wasm without
SIMD, `simd` is `+simd128`. Native is `aarch64-apple-darwin`, median of 5.

The clips: wasm-av1's 320x180 fixture (libaom), and four made with ffmpeg
and SVT-AV1 (preset 8, CRF 35): `testsrc2` at 1280x720, and `mandelbrot`
with noise at 1280x720 in 8 and 10 bits and at 1920x1080.

## Where it started (`3c2c549`), and rav1d

| clip | rav1d baseline | rav1d simd | baseline | simd | native rav1d | native |
|---|---|---|---|---|---|---|
| fixture 320x180 | 1173 | 1721 | 582 (0.50x) | 843 (0.49x) | 1508 | 974 |
| testsrc2 720p | 176 | 298 | 59.5 (0.34x) | 110 (0.37x) | 231 | 131 |
| mandelbrot 720p | 64.6 | 145 | 30.7 (0.48x) | 61.8 (0.43x) | 84.0 | 72.4 |
| mandelbrot 720p 10-bit | 78.0 | 81.5 | 36.0 (0.46x) | 78.5 (0.96x) | 101 | 79.1 |
| mandelbrot 1080p | 32.7 | 76.2 | 15.2 (0.46x) | 32.3 (0.42x) | 41.6 | 38.4 |

## Steps

Each row is the state after the step: fps, and its ratio to rav1d's.

| step | clip | baseline | simd | native |
|---|---|---|---|---|
| 1. inter prediction from an unscaled reference: 16-bit intermediates, zero phases as shifts, SIMD128 dot products | fixture 320x180 | 628 (0.54x) | 904 (0.53x) | 1043 |
| | testsrc2 720p | 65.1 (0.37x) | 119 (0.40x) | 142 |
| | mandelbrot 720p | 32.3 (0.50x) | 68.2 (0.47x) | 78.5 |
| | mandelbrot 720p 10-bit | 38.4 (0.49x) | 81.4 (1.00x) | 85.6 |
| | mandelbrot 1080p | 16.4 (0.50x) | 34.9 (0.46x) | 42.8 |
| 2. the symbol decoder reads its bits from a 24-bit window, not one at a time | fixture 320x180 | 670 (0.54x) | 979 (0.54x) | 1143 |
|  | testsrc2 720p | 70.4 (0.37x) | 130 (0.40x) | 145 |
|  | mandelbrot 720p | 34.4 (0.50x) | 72.5 (0.47x) | 84.2 |
|  | mandelbrot 720p 10-bit | 41.5 (0.52x) | 91.4 (1.06x) | 108 |
|  | mandelbrot 1080p | 17.5 (0.52x) | 37.5 (0.47x) | 44.5 |
| 3. CDEF without the window: zero strengths skipped, interior blocks filtered in the plane, 16-bit SIMD128 lanes | fixture 320x180 | 683 (0.57x) | 959 (0.56x) | 904 |
|  | testsrc2 720p | 103 (0.58x) | 136 (0.45x) | 157 |
|  | mandelbrot 720p | 43.5 (0.65x) | 80.0 (0.53x) | 70.5 |
|  | mandelbrot 720p 10-bit | 51.5 (0.67x) | 98.2 (1.21x) | 82.6 |
|  | mandelbrot 1080p | 21.8 (0.68x) | 39.8 (0.54x) | 32.9 |
| 4. the in-plane CDEF filter with NEON; AVX2 hosts keep the window filter | fixture 320x180 | 694 (0.59x) | 972 (0.56x) | 1196 |
|  | testsrc2 720p | 102 (0.56x) | 148 (0.49x) | 176 |
|  | mandelbrot 720p | 43.9 (0.65x) | 70.0 (0.46x) | 100 |
|  | mandelbrot 720p 10-bit | 42.2 (0.61x) | 81.6 (1.17x) | 127 |
|  | mandelbrot 1080p | 20.2 (0.78x) | 38.7 (0.60x) | 52.4 |
| 5. reconstruction: whole-sample vectors copied, DC-only blocks, rows instead of samples | fixture 320x180 | 740 (0.61x) | 1046 (0.59x) | 1180 |
|  | testsrc2 720p | 111 (0.65x) | 169 (0.59x) | 157 |
|  | mandelbrot 720p | 46.8 (0.74x) | 83.9 (0.57x) | 97.1 |
|  | mandelbrot 720p 10-bit | 55.4 (0.77x) | 111 (1.42x) | 143 |
|  | mandelbrot 1080p | 22.3 (0.69x) | 45.1 (0.65x) | 52.8 |

### 1. Inter prediction from an unscaled reference

V8's profile of the simd build on mandelbrot 720p had `block_inter_prediction`
at 20 % of the decode. The general process filters `h + 7` rows horizontally
into 32-bit intermediates and then `h` rows vertically, whatever the motion
vector. For a reference of the frame's own size (every reference, unless the
stream scales them) there is one filter phase per direction for the block,
and: the intermediates fit 16 bits at every depth (tested), so both passes
can multiply 16-bit lanes (`i32x4_dot_i16x8`, eight outputs a step); a phase
of zero is a shift, not a filter; and with a vertical phase of zero only the
`h` rows that are read need computing. After: 11 % of the decode. x86-64
keeps the general path, whose filters have AVX2 versions that could not be
measured where this was written.

### 2. The symbol decoder reads its bits from a 24-bit window, not one at a time

The symbol decoder followed the specification bit for bit: every renormalisation read its bits one at a time, each with its own byte index and bounds check, and the CDF update tested for the symbol at every entry. Bits now come out of the three bytes at the read position in one step (zeros past the end, as before), a renormalisation that shifts nothing returns at once, and the update runs as two loops, below the symbol and from it. Worth 1 to 3 %: `read_symbol` is 12 % of the simd decode and most of that is the search and the update, not the bits.

### 3. CDEF without the window: zero strengths skipped, interior blocks filtered in the plane, 16-bit SIMD128 lanes

CDEF was 21 % of the simd decode and 40 % of the baseline one. Every 8x8 block was copied into a 12x12 window of 32-bit values with the unavailable samples marked, then filtered with all twelve taps, whatever its strengths. Now: a block whose primary and secondary strengths are both zero is not touched (the filter is the identity there), and its direction is not searched unless a primary strength needs it; a block whose border of 2 lies inside the frame, which is nearly all of them, is filtered from the plane itself; a strength of zero costs no taps (with one set of taps alone the clip to the taps' range is a no-op, so leaving the unused taps out of it changes nothing: tested against the window filter); and the SIMD128 version takes a row of eight samples in 16-bit lanes instead of four in 32-bit ones. After: 3.6 % of the simd decode, 21 % of the baseline one (where it is the scalar arithmetic itself).

### 4. The in-plane CDEF filter with NEON; AVX2 hosts keep the window filter

The step before sent interior blocks to the in-plane filter on every target, and on aarch64 that was the scalar version in place of the window filter's NEON one: native decoding got slower (mandelbrot 720p 83.5 to 70.5 fps). The in-plane filter now has a NEON version, 16-bit lanes like the SIMD128 one (tested against the window filter natively), and a host with AVX2 keeps the window filter, whose AVX2 version has no in-plane counterpart yet and could not be measured here. Native is now ahead of where it was before the CDEF work. The wasm numbers are unchanged, as they should be.

### 5. Reconstruction: whole-sample vectors copied, DC-only blocks, rows instead of samples

Four things around the block's samples. A single-reference block whose motion vector has no fractional part (most of a still background) had both filters run as `128 * sample` and its result clamped and stored a sample at a time; it is now the reference's rows copied. A transform block with only its DC coefficient, the commonest kind, went through the whole 2D transform; its residual is one value per row, found with one row and one column transform (checked against the full process for every size and depth). The prediction and the residual were written through `PlaneBuf::set`, a bounds check and an index computation per sample; they go a row slice at a time, which the compiler vectorises. And dequantisation divided 64-bit values by `dqDenom` (1, 2 or 4): a shift. `scale_mv` no longer divides to find a scale of one.

## Where it stands after step 6 (`7369fee`)

All fourteen streams of the comparison, median of 7, load average about 5.
`start` is `3c2c549`; the ratio is to rav1d.

| stream | native: rav1d | start | now | wasm baseline: rav1d | start | now | wasm simd: rav1d | start | now |
|---|---|---|---|---|---|---|---|---|---|
| fixture 320x180, 8-bit | 1520 | 1012 | 1252 (0.82x) | 1184 | 606 | 732 (0.62x) | 1771 | 853 | 1060 (0.60x) |
| fixture 160x90, 10-bit | 3466 | 2550 | 3067 (0.88x) | 3005 | 1777 | 2110 (0.70x) | 3046 | 2445 | 2918 (0.96x) |
| fixture 128x72, film grain | 2856 | 2000 | 2250 (0.79x) | 2342 | 1423 | 1594 (0.68x) | 3225 | 1846 | 2063 (0.64x) |
| fixture 177x99, 8-bit | 2266 | 1972 | 2332 (1.03x) | 1990 | 1134 | 1354 (0.68x) | 3394 | 1808 | 2159 (0.64x) |
| fixture 96x64, 4:4:4 | 4347 | 3300 | 3802 (0.87x) | 3813 | 2147 | 2572 (0.67x) | 6129 | 3284 | 4004 (0.65x) |
| fixture 96x64, mono | 10468 | 7600 | 8594 (0.82x) | 12101 | 6529 | 7727 (0.64x) | 16016 | 8487 | 10155 (0.63x) |
| AOM b8-05-mv 352x288 | 203 | 172 | 230 (1.13x) | 156 | 91.2 | 138 (0.88x) | 250 | 154 | 194 (0.77x) |
| AOM b10-00-quantizer-10, 10-bit | 92.8 | 73.4 | 93.2 (1.00x) | 70.3 | 39.4 | 59.6 (0.85x) | 73.2 | 64.5 | 81.1 (1.11x) |
| AOM b8-23-film_grain-50 | 438 | 317 | 401 (0.92x) | 360 | 158 | 224 (0.62x) | 540 | 282 | 347 (0.64x) |
| AOM b8-00-quantizer-20 | 244 | 203 | 247 (1.01x) | 201 | 102 | 150 (0.75x) | 291 | 179 | 218 (0.75x) |
| SVT testsrc2 720p | 249 | 132 | 205 (0.82x) | 187 | 61.7 | 117 (0.63x) | 320 | 113 | 172 (0.54x) |
| SVT mandelbrot 720p | 87.4 | 75.8 | 111 (1.27x) | 67.9 | 31.0 | 48.4 (0.71x) | 152 | 62.9 | 91.1 (0.60x) |
| SVT mandelbrot 720p, 10-bit | 101 | 94.2 | 149 (1.48x) | 78.4 | 37.2 | 60.6 (0.77x) | 85.5 | 80.6 | 125 (1.46x) |
| SVT mandelbrot 1080p | 40.9 | 37.0 | 59.3 (1.45x) | 33.5 | 15.6 | 23.9 (0.71x) | 78.8 | 33.3 | 49.0 (0.62x) |

**Not at parity.** One thread, against rav1d: the baseline build is at 0.62
to 0.88 (it started at 0.33 to 0.61), the simd build at 0.54 to 0.77 on
8-bit streams (0.36 to 0.63) and at 0.96 to 1.46 on 10-bit ones (rav1d's
wasm kernels are 8-bit only). Native aarch64 went from 0.54 to 0.93 of
rav1d to 0.79 to 1.48, and no row got slower. Every stream decodes to the
same MD5 in every build.

Threads are as they were, since nothing here touches them: the simd +
threads build on 1 / 2 / 4 / 8 threads decodes mandelbrot 720p at 90 / 99 /
109 / 107 fps (rav1d: 143 on one, 485 on eight) and its eight-tile version
at 91 / 152 / 221 / 245 (rav1d: 152 and 940).

### What is left, measured

V8's profile of both simd builds on mandelbrot 720p (rav1d built with its
function names for this), as milliseconds a frame:

| part | rav1d | here | gap |
|---|---|---|---|
| symbol decoder and coefficient reading | 1.60 | 3.12 | 1.5 |
| inter prediction | 1.16 | 2.24 | 1.1 |
| loop filter | 0.59 | 1.36 | 0.8 |
| inverse transforms and reconstruction | 1.13 | 1.69 | 0.6 |
| CDEF | 1.16 | 1.03 | ahead |
| everything else | 1.0 | 1.6 | 0.6 |
| a frame | 6.65 | 11.05 | 4.4 |

So, in order of what it would buy:

1. **Entropy decoding, 1.5 ms.** Twice rav1d's time for the same symbols.
   The reader is the specification's, and it is shared with the encoder:
   every symbol goes through `Coder::symbol` with the value the encoder
   would write, every context is derived from the 32-bit `Quant` array with
   bounds tests. rav1d has dav1d's: a 64-bit window refilled a word at a
   time, the search and the adaptation specialised by alphabet size, a
   padded byte map of levels for the contexts. The level map was tried
   here (`coeff_base_ctx` / `coeff_br_ctx` over a padded `u8` map, bit-exact
   on all 244 vectors) and measured between -5 % and +6 % on a machine too
   loaded to tell; it is not on the branch. Doing this properly means a
   decode-only coefficient reader beside the shared walker, which is a
   decision about the encoder's "the decoder's own walker" design.
2. **Inter prediction, 1.1 ms.** The two filters are now SIMD; what remains
   is around them: the prediction goes through a 32-bit buffer and is
   clamped and stored afterwards, each row is fetched through a closure,
   compound averaging and the masks are scalar. Filtering straight into the
   frame for a single reference, and 16-bit SIMD for the compound cases.
3. **Loop filter, 0.8 ms.** Every 4-sample edge is gathered into a 14 x 4
   array of 32-bit values, all filter sizes are computed for every lane, and
   the result is scattered back; the mode info is copied per edge. rav1d
   filters eight edges' worth in 16-bit lanes from the plane. (Skipping the
   write-back when no lane passes the mask was tried: no measurable
   change.)
4. **Inverse transforms, 0.6 ms.** The network is the specification's
   generic one, with its bit-reversal permutation and angles looked up as it
   runs; rav1d has one straight-line function per size and skips what the
   end-of-block position says is zero. (Sending the baseline build through
   the eight-lane path was tried: within noise, and 150 KB larger.)

None of this needs a different sample type: 16-bit lanes hold 8- and
10-bit samples, which is what the CDEF and prediction kernels here already
do. For the baseline build the same list applies with CDEF added back on
top (its scalar arithmetic is a fifth of that decode).

### What every step was held to

`cargo fmt --all --check`, `cargo clippy --all-targets --all-features -D
warnings`, `cargo doc`, `cargo test --release`, the committed vectors and
the runner test on 4 threads, the fuzz and encoder tests in debug, clippy
for wasm32 with and without `simd128`; wasm-av1's tests on this crate (its
six fixtures, natively on 1, 2 and 4 threads); the fixtures, four AOM
vectors and five SVT-AV1 clips through the wasm baseline and simd builds
against rav1d and libdav1d's MD5s; the clips natively on 1 and 4 threads.
From step 6 on, x86-64 as well: clippy and `cargo test --release` with a
locally built std, run under Rosetta, which reports no AVX2, so those runs
take x86-64's scalar paths. At the end: all 244 AOM vectors natively on 1
and 4 threads (244 of 244), and the 242 that are IVF through both wasm
builds against rav1d (242 of 242 agree).

## Steps after the first round

As above: the state after each step, fps and the ratio to rav1d's.

| step | clip | baseline | simd | native |
|---|---|---|---|---|
| 6. a decode-only coefficient reader, held to the shared walker by a test | fixture 320x180 | 738 (0.62x) | 1062 (0.60x) | 1242 |
|  | testsrc2 720p | 115 (0.65x) | 172 (0.59x) | 209 |
|  | mandelbrot 720p | 48.7 (0.74x) | 95.9 (0.66x) | 114 |
|  | mandelbrot 720p 10-bit | 62.0 (0.80x) | 130 (1.55x) | 152 |
|  | mandelbrot 1080p | 24.4 (0.73x) | 50.2 (0.65x) | 61.2 |
| 7. inter prediction: no buffer clearing per block, no closure call per row | fixture 320x180 | 772 (0.62x) | 1107 (0.60x) | 1286 |
|  | testsrc2 720p | 125 (0.66x) | 189 (0.58x) | 223 |
|  | mandelbrot 720p | 50.5 (0.74x) | 98.8 (0.65x) | 120 |
|  | mandelbrot 720p 10-bit | 60.6 (0.79x) | 135 (1.62x) | 160 |
|  | mandelbrot 1080p | 25.3 (0.75x) | 49.9 (0.62x) | 60.5 |

### 6. A decode-only coefficient reader, held to the shared walker by a test

The coefficient reader was the tile walker's, which the encoder runs too: every symbol through `Coder` with the value the encoder would write, every context from the 32-bit `Quant` array with an edge test per neighbour. `coeffs_decode` reads the same syntax for a decoder alone: the levels in a padded byte map so that a context is a sum of five loads, the level symbols read with their CDF size known so that the search and the adaptation are fixed-length loops, an equiprobable bit as the two comparisons it is. The shared reader stays, untouched, as the reference the encoder uses. `CoefficientReader::Checked` runs both on every transform block from the same state and panics unless the coefficients, the end-of-block position, every CDF, the symbol decoder's state and the neighbour contexts are identical; the committed vectors, the 244 AOM vectors, the fuzz tests and the encoder round trips all decode in that mode, and a planted difference (a context clamp of 2 for 3) fails them. Worth 3 to 5 %.

### 7. Inter prediction: no buffer clearing per block, no closure call per row

Two costs around the prediction filters. `predict_inter` cleared and zero-filled its 32-bit prediction buffers for every block and every list, up to 64 KB each, before the filters overwrote all of it; the buffers now only grow. And the unscaled path fetched each reference row through a `&mut dyn FnMut`, a call the compiler cannot see through, once per row; it is a function returning the row. Worth 1 to 3 %, the same in two interleaved runs of each build on a quiet machine. Filtering a single-reference block straight into the frame, with no prediction buffer at all, was written and measured first: bit-exact, and no faster on any stream here (within 1 %), because these streams predict mostly from two references; it is not on the branch.

## Where it stands after step 7 (`c733c23`)

The same fourteen streams, median of 7, load average 3 to 4.

| stream | native: rav1d | start | now | wasm baseline: rav1d | start | now | wasm simd: rav1d | start | now |
|---|---|---|---|---|---|---|---|---|---|
| fixture 320x180, 8-bit | 1604 | 1049 | 1316 (0.82x) | 1261 | 615 | 773 (0.61x) | 1852 | 899 | 1119 (0.60x) |
| fixture 160x90, 10-bit | 3772 | 2769 | 3338 (0.88x) | 3034 | 1792 | 2157 (0.71x) | 2998 | 2454 | 2988 (1.00x) |
| fixture 128x72, film grain | 3025 | 2155 | 2447 (0.81x) | 2356 | 1454 | 1637 (0.70x) | 3227 | 1848 | 2114 (0.65x) |
| fixture 177x99, 8-bit | 2416 | 2148 | 2710 (1.12x) | 1994 | 1198 | 1446 (0.73x) | 3406 | 1917 | 2351 (0.69x) |
| fixture 96x64, 4:4:4 | 4412 | 3358 | 4069 (0.92x) | 3934 | 2169 | 2651 (0.67x) | 6102 | 3312 | 4056 (0.66x) |
| fixture 96x64, mono | 11168 | 7821 | 9138 (0.82x) | 12040 | 6589 | 7628 (0.63x) | 16934 | 8419 | 10420 (0.62x) |
| AOM b8-05-mv 352x288 | 210 | 178 | 248 (1.18x) | 165 | 94.1 | 146 (0.89x) | 268 | 155 | 209 (0.78x) |
| AOM b10-00-quantizer-10, 10-bit | 94.8 | 76.1 | 101 (1.07x) | 73.4 | 39.6 | 62.7 (0.85x) | 77.2 | 66.2 | 83.3 (1.08x) |
| AOM b8-23-film_grain-50 | 442 | 325 | 420 (0.95x) | 342 | 160 | 229 (0.67x) | 512 | 284 | 357 (0.70x) |
| AOM b8-00-quantizer-20 | 248 | 207 | 265 (1.07x) | 206 | 106 | 158 (0.77x) | 299 | 184 | 232 (0.78x) |
| SVT testsrc2 720p | 250 | 135 | 223 (0.89x) | 191 | 62.8 | 126 (0.66x) | 326 | 114 | 190 (0.58x) |
| SVT mandelbrot 720p | 87.7 | 76.8 | 119 (1.36x) | 68.8 | 31.6 | 51.1 (0.74x) | 153 | 63.9 | 99.2 (0.65x) |
| SVT mandelbrot 720p, 10-bit | 103 | 94.7 | 158 (1.53x) | 79.0 | 37.9 | 61.7 (0.78x) | 86.4 | 81.1 | 130 (1.51x) |
| SVT mandelbrot 1080p | 42.7 | 39.1 | 62.7 (1.47x) | 33.8 | 16.1 | 25.3 (0.75x) | 80.0 | 33.3 | 53.4 (0.67x) |

**Still not at parity on 8-bit streams in wasm.** One thread, against
rav1d: baseline 0.61 to 0.89, simd 0.58 to 0.78 on 8-bit and 1.00 to 1.51 on
10-bit, native aarch64 0.81 to 1.53. All 244 AOM vectors pass natively on 1
and 4 threads with both coefficient readers checked against each other;
the 242 IVF ones agree with rav1d in both wasm builds. The simd + threads
build on 1 / 2 / 4 / 8 threads: mandelbrot 720p 99 / 112 / 120 / 121 fps,
its eight-tile version 96 / 167 / 241 / 261.

Steps 6 and 7 bought 3 to 5 % and 1 to 3 %. Two more changes were written
in this round, verified bit-exact and measured to gain nothing (a
single-reference prediction filtered straight into the frame; see
`tools/wasm-perf/tried/`). The trims are used up: on mandelbrot 720p simd
the decode is now about 10.1 ms a frame against rav1d's 6.6, and what
separates them is four kernels' worth of work, each of which rav1d has in
dav1d's form:

| part | rav1d, ms | here, ms | what closing it takes |
|---|---|---|---|
| coefficient reading and the symbol decoder | 1.6 | 2.7 | the arithmetic decoder on a 64-bit window refilled a word at a time; the end-of-block position driving which contexts are computed |
| inter prediction | 1.2 | 2.0 | compound prediction in 16-bit intermediates with a biased range for 10 bits, SIMD for the average, the masks and the distance weights |
| loop filter | 0.6 | 1.4 | a kernel in 16-bit lanes that reads and writes the plane, eight edge units at a time, per filter size, for SIMD128 and NEON together (the scalar gather-filter-scatter stays as the reference) |
| inverse transforms and reconstruction | 1.1 | 1.7 | one straight-line function per size and kind in place of the generic network, and rows and columns past the end-of-block position skipped |
| CDEF | 1.2 | 1.0 | ahead |

Done completely, those would bring the frame to roughly 7.5 to 8 ms: about
0.85 of rav1d, not parity. The last 15 % is everything else being
dav1d-shaped (the mode info walk, motion vector prediction, the frame
buffers), which is not a list of kernels.

## Inter prediction, from the specification's processes

The steps of this section, of "The loop filter" and of "The residual path"
below were made side by side on separate branches from `829b1e1`, so all
three number their steps from 8 and each section's "before" is that commit,
without the others' work.

`090bf70` (after step 7, and not a row above) keeps both predictions of an
averaging compound block in 16 bits. The steps below go on from `829b1e1`
and are inter prediction alone (7.11.3): its filters, how a block's rows
reach them, the blends, the warp. They were written from the
specification's block inter prediction process and this crate's scalar
code for it, and every kernel is held to that arithmetic by a unit test on
random blocks: run natively for the NEON and scalar versions, and for the
SIMD128 ones with the crate's tests built for `wasm32-wasip1` with
`simd128` and run in Node's WASI (`cargo test` cannot run
`wasm32-unknown-unknown`).

Besides fps, each step says what inter prediction costs on mandelbrot
720p, in milliseconds a decoded frame. In wasm that is V8's profile: the
time under `predict_inter`, it and everything it calls. Natively it is a
timer around `predict_inter` (a measuring patch, not on the branch), less
what the timer itself costs. The builds compared take turns and the
figure is the median of the turns; other builds were running on the
machine (load average 8 to 19), so the fps columns move by a few percent
from one table to the next. The native column has its ratio to rav1d too.

| step | clip | baseline | simd | native |
|---|---|---|---|---|
| 8. the horizontal filter a block at a time; 8-bit sums in 16-bit lanes | fixture 320x180 | 746 (0.63x) | 1078 (0.62x) | 1234 (0.81x) |
|  | testsrc2 720p | 126 (0.69x) | 194 (0.62x) | 211 (0.91x) |
|  | mandelbrot 720p | 50.6 (0.78x) | 102 (0.70x) | 116 (1.38x) |
|  | mandelbrot 720p 10-bit | 62.5 (0.83x) | 136 (1.66x) | 162 (1.60x) |
|  | mandelbrot 1080p | 25.0 (0.77x) | 54.5 (0.72x) | 64.9 (1.53x) |
| 9. the vertical filter a block at a time, in column strips | fixture 320x180 | 731 (0.64x) | 1061 (0.63x) | 1248 (0.83x) |
|  | testsrc2 720p | 122 (0.71x) | 188 (0.65x) | 218 (0.94x) |
|  | mandelbrot 720p | 50.4 (0.79x) | 101 (0.71x) | 122 (1.46x) |
|  | mandelbrot 720p 10-bit | 63.8 (0.85x) | 138 (1.70x) | 161 (1.66x) |
|  | mandelbrot 1080p | 25.3 (0.79x) | 54.9 (0.74x) | 65.9 (1.60x) |
| 10. two whole-sample references averaged from windows of them | fixture 320x180 | 762 (0.64x) | 1104 (0.64x) | 1306 (0.86x) |
|  | testsrc2 720p | 129 (0.72x) | 198 (0.64x) | 226 (0.96x) |
|  | mandelbrot 720p | 51.2 (0.80x) | 102 (0.72x) | 131 (1.51x) |
|  | mandelbrot 720p 10-bit | 64.7 (0.86x) | 145 (1.77x) | 176 (1.73x) |
|  | mandelbrot 1080p | 25.9 (0.80x) | 56.6 (0.75x) | 68.9 (1.66x) |
| 11. the block warp reads a window of the reference | fixture 320x180 | 747 (0.64x) | 1082 (0.64x) | 1288 (0.86x) |
|  | testsrc2 720p | 126 (0.72x) | 191 (0.64x) | 215 (0.94x) |
|  | mandelbrot 720p | 51.7 (0.80x) | 104 (0.72x) | 123 (1.50x) |
|  | mandelbrot 720p 10-bit | 63.8 (0.86x) | 139 (1.72x) | 164 (1.70x) |
|  | mandelbrot 1080p | 24.8 (0.79x) | 54.8 (0.74x) | 66.3 (1.62x) |
| 12. the averages take a block | fixture 320x180 | 754 (0.65x) | 1093 (0.65x) | 1289 (0.84x) |
|  | testsrc2 720p | 127 (0.73x) | 195 (0.65x) | 224 (0.96x) |
|  | mandelbrot 720p | 52.1 (0.80x) | 106 (0.73x) | 128 (1.53x) |
|  | mandelbrot 720p 10-bit | 64.1 (0.86x) | 142 (1.74x) | 170 (1.74x) |
|  | mandelbrot 1080p | 25.8 (0.80x) | 56.7 (0.76x) | 68.1 (1.66x) |
| 13. a block's taps looked up, its buffers borrowed once | fixture 320x180 | 753 (0.65x) | 1090 (0.64x) | 1328 (0.86x) |
|  | testsrc2 720p | 127 (0.72x) | 194 (0.65x) | 222 (0.96x) |
|  | mandelbrot 720p | 52.6 (0.81x) | 106 (0.72x) | 127 (1.53x) |
|  | mandelbrot 720p 10-bit | 64.1 (0.87x) | 141 (1.74x) | 173 (1.77x) |
|  | mandelbrot 1080p | 25.8 (0.80x) | 57.0 (0.75x) | 68.7 (1.67x) |
| 14. a warp output as one 16-bit dot product | fixture 320x180 | 754 (0.65x) | 1101 (0.66x) | 1308 (0.87x) |
|  | testsrc2 720p | 126 (0.73x) | 191 (0.65x) | 226 (0.97x) |
|  | mandelbrot 720p | 52.0 (0.81x) | 106 (0.74x) | 127 (1.52x) |
|  | mandelbrot 720p 10-bit | 63.3 (0.86x) | 143 (1.79x) | 171 (1.76x) |
|  | mandelbrot 1080p | 25.7 (0.81x) | 56.5 (0.75x) | 67.6 (1.66x) |
| 15. the scalar filter and mean, four samples to a 64-bit word | fixture 320x180 | 799 (0.65x) | 1158 (0.65x) | 1302 (0.86x) |
|  | testsrc2 720p | 132 (0.75x) | 196 (0.63x) | 225 (0.96x) |
|  | mandelbrot 720p | 53.6 (0.83x) | 105 (0.74x) | 126 (1.51x) |
|  | mandelbrot 720p 10-bit | 64.9 (0.87x) | 143 (1.76x) | 172 (1.76x) |
|  | mandelbrot 1080p | 26.5 (0.83x) | 56.9 (0.76x) | 68.2 (1.66x) |

### 8. The horizontal filter a block at a time, and 8-bit sums in 16-bit lanes

The horizontal filter was called a row at a time, each row fetched through `source_row` (a clamp of the row number, a test for the block's columns lying inside the reference, a slice), and it multiplied by all eight taps in 32-bit lanes: pairs of taps as 16-bit dot products over interleaved samples, with a test per pair for both taps being zero. Now a block's filter is one call. `source_window` gives the samples under the taps as a slice of the reference and its stride, or, for a block that reaches past the reference's edges, a copy with the edges repeated; the kernel runs over the block's rows. It multiplies by the taps that are not zero (`Taps`: six for the regular and smooth filters, four for the filters of blocks 4 wide, two for the bilinear one), which also narrows what has to lie inside the reference.

For a frame of 8 bits the sum itself fits 16 bits. A filter's taps sum to 128, at most 184 of it positive and 56 negative (tested over the table), so over samples of 0..255 the sum lies in -14280..46920, and with 16384 added it is an unsigned 16-bit value. The 8-bit kernel is then one 16-bit multiply and one add per tap for eight outputs (`i16x8_mul` in SIMD128, `vmlaq_n_u16` in NEON), a logical shift, and the 16384 taken off again. For 10 and 12 bits the SIMD128 kernel takes the taps two by two as dot products straight from the row (a 32-bit lane of the row is two neighbouring samples, so the row from column `c` gives the even outputs' sums and the row from `c + 1` the odd ones': no interleaving), and the NEON one is widening multiply-accumulates.

The kernels were timed against each other in V8 on blocks of 8x8 to 64x64 before being wired in: the kernel that was there 0.63 to 0.82 ns a sample, dot products from the row 0.32 to 0.34, the 16-bit sums 0.12 to 0.18. Inter prediction on mandelbrot 720p: simd 1.99 to 1.47 ms a frame, baseline 4.39 to 4.12, native 1.73 to 1.62 (the vertical filter is the larger part there, and it is the next step). Two native rows of the table's run read flat and down (the fixture and 10-bit mandelbrot); measured again over 15 turns they are 1252 to 1265 and 154.5 to 157.1 fps.

### 9. The vertical filter a block at a time, in column strips

The vertical filter was also a call per row. For each row of the block the SIMD128 version loaded the eight rows of intermediates under the taps and interleaved them pair by pair for its dot products: eight loads and eight shuffles for work that the next row but one mostly repeats (row `r` takes the pairs of rows `r + 2k` and `r + 2k + 1`, and row `r + 2` takes all but one of the same pairs). Natively it was scalar. Now it takes a block, eight columns at a time down the block (four for a block under eight wide). The SIMD128 version keeps the interleaved pairs in two sets, for the even rows and the odd ones, so that a row of the block costs one load, one interleave and a dot product per pair of taps: three pairs for the regular and smooth filters, whose first and last taps are zero. The NEON version keeps the rows under the taps in registers and multiplies by widening multiply-accumulates. The horizontal pass before it computes only the rows under the taps that are not zero (`h + 5` rows for those filters, not `h + 7`). In V8 on the same blocks: 0.92 to 1.13 ns a sample before, 0.28 to 0.33 after (widening multiplies on rows kept in registers, with no interleaving, measured 0.30 to 0.36).

Two things came with it. A block with a vertical phase and no horizontal one used to copy the reference's samples, shifted up, into the intermediate array and filter that. The intermediate is the sample shifted up by `7 - InterRound0`, so the vertical filter now runs on the reference's samples themselves with that much less of a shift (at 8 bits it is the horizontal filter's 16-bit sum, its taps a stride apart). And a single prediction comes out of the same function as a compound one, in 16 bits, which it fits (`Round2` by 11, or 9 at 12 bits, where a compound prediction is rounded by 7), and is widened for the general path's 32-bit blends.

Inter prediction on mandelbrot 720p: simd 1.47 to 1.23 ms a frame, baseline 4.12 to 3.49, native 1.62 to 1.05.

### 10. Two whole-sample references averaged from windows of them

Two thirds of the samples that mandelbrot 720p predicts from two references have whole-sample motion vectors into both, and are the rounded mean of the two blocks (`090bf70`). Those blocks still fetched every row of both references through `source_row`, with its clamp and its test for the columns. They take their two windows once (`source_window`, as the filters do), and `source_row` is gone. Inter prediction on mandelbrot 720p: simd 1.23 to 1.18 ms a frame, native 1.05 to 0.95; baseline within noise (3.55 to 3.51).

### 11. The block warp reads a window of the reference

The block warp process (7.11.3.5) read every sample under every tap of its horizontal filters through a clamp of both coordinates and a bounds-checked `get`: 960 of them for an 8x8 block. It takes the 15 rows of 15 samples it reads as a window, once (`source_window`, on every architecture: the reference itself unless the window reaches past an edge), and its filters run over slices of that. Inter prediction on mandelbrot 720p, where 1 % of the predicted samples are warped: simd 1.13 to 1.10 ms a frame, baseline 3.52 to 3.46, native 0.99 to 0.92.

### 12. The averages take a block

The two averages (of two whole-sample references, and of a compound block's two 16-bit predictions) were called a row at a time, each call slicing its three rows and testing the width again. They take the block, with the row loop inside the kernel. Inter prediction on mandelbrot 720p: native 0.92 to 0.86 ms a frame, simd 1.08 to 0.96 over nine turns (1.02 to 1.00 over five before that: within what V8's profile resolves on this machine), baseline within noise. A SIMD128 copy for whole-sample single predictions, in place of the `memory.copy` that `copy_from_slice` compiles to, was written with it and measured to gain nothing (0.93 and 0.95 ms): that time is the reference's and the frame's memory, not the call. It is in `tools/wasm-perf/tried/`.

### 13. A block's taps looked up, its buffers borrowed once

Two costs a block paid before any sample. Its two filters' kept taps were found by scanning the eight taps of each (`Taps::new`, 2 to 4 % of the prediction's time in V8's profile): they come from a table built with the crate (`Taps::of`). And the per-thread buffers (the intermediate array, the 16-bit predictions, the copy of a reference's edge) were three thread-locals, each taken out of its cell and put back around every use: five pairs of those for a compound block, one for every 8x8 of a warped block. They are one structure, borrowed once for a block and handed down. Inter prediction on mandelbrot 720p natively, nine turns: 0.87 to 0.83 ms a frame (0.85 with the table alone). In wasm, where a thread-local is a static, V8's profile does not resolve it (0.94 and 0.97 over seven turns, within its noise here).

### 14. A warp output as one 16-bit dot product

Each output of the warp's two passes has its own filter, so there is no row of outputs to share a tap between; what an output is, though, is eight 16-bit values times eight taps. `dot8` is that as one 16-bit dot product (`i32x4_dot_i16x8` and two adds across the lanes in SIMD128; `vmull_s16`, `vmlal_high_s16` and `vaddvq_s32` in NEON), over `Warped_Filters` narrowed to 16 bits when the crate is built. The intermediates fit 16 bits (tested over the table) and are kept a column at a time, since the vertical pass reads down a column. Inter prediction on mandelbrot 720p: native 0.83 to 0.79 ms a frame over nine turns, simd 0.97 to 0.89 over seven. The scalar version is what x86-64 takes.

### 15. The scalar 8-bit filter and mean, four samples to a 64-bit word

The baseline wasm build (no SIMD) and any target without a SIMD version take the scalar kernels, and LLVM has nothing to vectorise them with there. Two of them now work four samples at a time in a 64-bit word. Four neighbouring samples of a plane are the four 16-bit lanes of the word at their address (the compiler turns the four loads and shifts into one 64-bit load: checked in the wasm it emits), and a tap times that word is the tap times each sample as long as no lane carries into the next. For 8-bit samples that is the arithmetic of step 8 again: the positive taps are summed in one word with the lift (under 65536 a lane), the negative ones in another (at most 14280 a lane, less than the lift, so subtracting it borrows nothing), then one shift, one mask and the lift taken off with each lane's top bit set against a borrow. And the rounded mean of two whole-sample references is `(a | b) - ((a ^ b) >> 1)` in each lane. Inter prediction on mandelbrot 720p in the baseline build: 3.42 to 2.63 ms a frame (3.02 with the filter alone, 2.85 with one bounds test for the samples under all its taps, 2.63 with the mean). The simd and native builds take these only for blocks under eight wide, and do not move.

## Where inter prediction stands after step 15 (`eb42f3c`)

The fourteen streams again, median of 7, the three builds taking turns on
each stream, load average 4 to 6. `before` is `829b1e1`; the ratio is to
rav1d.

| stream | native: rav1d | before | now | wasm baseline: rav1d | before | now | wasm simd: rav1d | before | now |
|---|---|---|---|---|---|---|---|---|---|
| fixture 320x180, 8-bit | 1578 | 1305 | 1369 (0.87x) | 1217 | 766 | 794 (0.65x) | 1771 | 1102 | 1154 (0.65x) |
| fixture 160x90, 10-bit | 3598 | 3279 | 3430 (0.95x) | 2895 | 2099 | 2151 (0.74x) | 2891 | 2844 | 2990 (1.03x) |
| fixture 128x72, film grain | 2947 | 2378 | 2448 (0.83x) | 2388 | 1626 | 1648 (0.69x) | 3349 | 2087 | 2133 (0.64x) |
| fixture 177x99, 8-bit | 2360 | 2675 | 2811 (1.19x) | 1927 | 1429 | 1461 (0.76x) | 3254 | 2287 | 2388 (0.73x) |
| fixture 96x64, 4:4:4 | 4342 | 3936 | 4210 (0.97x) | 3495 | 2400 | 2457 (0.70x) | 5222 | 3594 | 3747 (0.72x) |
| fixture 96x64, mono | 10814 | 8807 | 9185 (0.85x) | 9355 | 6505 | 6668 (0.71x) | 12732 | 8471 | 8840 (0.69x) |
| AOM b8-05-mv 352x288 | 208 | 241 | 277 (1.33x) | 160 | 143 | 153 (0.96x) | 258 | 202 | 230 (0.89x) |
| AOM b10-00-quantizer-10, 10-bit | 90.9 | 97.2 | 105 (1.15x) | 73.3 | 61.1 | 63.0 (0.86x) | 75.6 | 82.3 | 88.9 (1.18x) |
| AOM b8-23-film_grain-50 | 428 | 396 | 420 (0.98x) | 354 | 225 | 229 (0.65x) | 514 | 349 | 366 (0.71x) |
| AOM b8-00-quantizer-20 | 244 | 255 | 270 (1.11x) | 194 | 154 | 158 (0.81x) | 278 | 222 | 235 (0.85x) |
| SVT testsrc2 720p | 245 | 222 | 242 (0.99x) | 182 | 124 | 136 (0.75x) | 311 | 189 | 202 (0.65x) |
| SVT mandelbrot 720p | 85.3 | 115 | 131 (1.54x) | 65.1 | 49.8 | 54.3 (0.83x) | 146 | 96.2 | 108 (0.74x) |
| SVT mandelbrot 720p, 10-bit | 98.4 | 152 | 173 (1.76x) | 77.5 | 63.6 | 67.2 (0.87x) | 84.5 | 135 | 151 (1.79x) |
| SVT mandelbrot 1080p | 42.2 | 62.5 | 69.4 (1.64x) | 33.8 | 26.0 | 27.9 (0.83x) | 80.3 | 55.3 | 60.9 (0.76x) |

Every row is up in all three builds, and every stream decodes to the same
MD5 in every build. What the steps were about, inter prediction on
mandelbrot 720p in milliseconds a decoded frame, `829b1e1` against step 15
with the two builds taking turns (nine turns; seven for baseline):

| build | before | now |
|---|---|---|
| wasm simd | 1.77 | 0.79 |
| wasm baseline | 4.38 | 2.56 |
| native aarch64 | 1.66 | 0.70 |

rav1d's was 1.2 ms in the simd build's profile above. On a machine this
loaded the wasm figure moved between 0.79 and 0.85 from one run of nine
turns to the next (`829b1e1` between 1.77 and 1.99).

What is left of it in the simd build, from V8's profile with each block
kernel kept out of line (taken at step 12, total 1.0 ms; steps 13 and 14
have since taken about 0.1 off the warp and the per-block part): the
8-bit sums of the horizontal filter and of a vertical filter alone 0.24 ms
a frame, the vertical filter over intermediates 0.17 to 0.19, the mean of
two whole-sample references 0.11, the copy of one 0.10, the warp 0.11,
and 0.25 for everything a block does before and between its kernels. By
that clip's block counts the horizontal filter costs about 0.3 ns a
sample there, against 0.12 to 0.14 alone on blocks that sit in the cache.
The likely reason is memory: a block's rows are some twenty separate
lines of a reference that was written a frame ago. The mean and the copy
do nothing but move such rows, and a SIMD128 copy in place of
`memory.copy` changed nothing (`tools/wasm-perf/tried/`). Samples are 16
bits whatever the depth here; a plane of bytes for 8-bit frames would
halve that traffic, and is a change to every stage, not to prediction.

Still on the general path, in 32 bits and scalar: the distance-weighted,
wedge and difference-weighted compounds, inter-intra, overlapped
prediction's blend, and prediction from a scaled reference. None of the
benchmark streams spends time there. x86-64 is as it was but for the warp
(steps 11 and 14, which are not specific to a target): its other paths
keep the 32-bit AVX2 filters, which could not be measured here.

### What came from knowing how decoders are built

No other implementation's source was read for these steps, and the two
earlier attempts at them (the branches `perf-loopfilter` and
`perf-compound`) were not looked at. Some of the ideas are nevertheless
common knowledge about how AV1 decoders are built rather than things the
specification suggests, and were used by deriving them from this crate's
arithmetic and testing them against it: filtering a block from one window
of the reference, with a copy that repeats the edges when the block
reaches past them; multiplying only by the taps that are not zero (the
regular and smooth filters as six taps, the narrow-block ones as four);
doing an 8-bit filter in 16-bit lanes; separate paths for the horizontal
filter alone, the vertical one alone (on the reference's samples) and
both; a vertical filter that goes down a strip of columns keeping its
rows (or interleaved pairs of rows) from one output row to the next; the
warp's outputs as 16-bit dot products over a 16-bit filter table. The
four-samples-to-a-word scalar kernels, the dot products taken straight
from the row for even and odd outputs, the taps table and the single
borrow of the buffers are ordinary programming, not decoder lore.

### What these steps were held to

The list of the first round for every one of them (`tools/wasm-perf/gate.sh`:
fmt; clippy for the host, wasm32 with and without `simd128`, and x86-64;
doc; `cargo test --release`, which with the vectors in `tests/vectors/`
decodes all 244 AOM vectors, natively and for x86-64 under Rosetta; the
committed vectors and the runner test on 4 threads; the fuzz and encoder
tests; wasm-av1's tests on the tree; the 15 streams through both wasm
builds against rav1d and libdav1d; the clips natively on 1 and 4 threads).
At the end: the 244 vectors natively on 1 and 4 threads (244 of 244, with
`AV1_REQUIRE_VECTORS=1`), and the 242 that are IVF through both wasm builds
against rav1d (242 of 242 agree).

Each kernel has a unit test against the specification's arithmetic on
random blocks with the extreme values among them (`dsp::mc`), run natively
and, for SIMD128, as a `wasm32-wasip1` module in Node (`wasi-test.sh`);
bugs were planted in each kernel, 21 in all, to see the tests fail. One
did not fail them and was not a bug: a mask one bit too wide in the scalar
word filter, harmless because every tap of `Subpel_Filters` is even. And
the whole unscaled prediction (the windows, the edges, the zero phases,
the shifts of each depth) is held to the general process, the one the
specification writes, on random references of 8, 10 and 12 bits with
blocks inside, across and beyond every edge (`decoder::predict`); four
planted bugs in the window and the shifts fail it.

`tools/wasm-perf/` has the scripts every number here came from.

## The loop filter

From `829b1e1` (step 7 and the 16-bit average of two references that
followed it). Written from the specification's loop filter process (7.14)
and this crate's own filter, which stays as the reference every kernel here
is tested against; rav1d's source was not read, only its profile.

**Where its time went**, on mandelbrot 720p. The loop filter runs on 30 of
the clip's 60 frames (the others have a level of zero), on chroma in 14 of
those. In them, 12 % of luma's 4x4 units and 22 % of chroma's are on a
transform edge at all, and every one of those is filtered; 90 % of the luma
edges take the 16-sample filter size and 90 % of the chroma ones the
6-sample one. A quarter of the samples along those edges fail the filter
mask; of luma's that pass at size 16, 63 % go through the 13-tap filter.
Natively (aarch64), with the sample filtering and then the carrying of
samples taken out of a build in turn: 60 % of the loop filter's time was the
filter arithmetic (every filter computed for every lane of every edge, then
one selected), 23 % gathering the samples of an edge into a 14 x 4 array of
32-bit values and scattering them back, 16 % deciding which edges to
filter.

The loop filter's own time is in milliseconds a shown frame of mandelbrot
720p: in wasm, its share of V8's profile times the frame time of an
unprofiled run (`tools/wasm-perf/lfprof.sh`); natively, a clock around the
call (`tools/wasm-perf/lf-timing.diff`, not part of the crate). This table
is one run of all the builds after step 11, taking turns (median of 5 in
wasm, of 7 natively); the figures in each step's text are from that step's
own run against the step before, and differ from these by what two runs on
this machine differ by.

| step | loop filter, ms a frame: wasm simd | wasm baseline | native |
|---|---|---|---|
| before (`829b1e1`) | 1.44 | 2.30 | 1.26 |
| 8. edge decisions | 1.31 | 2.19 | 1.10 |
| 9. in the plane, eight 16-bit lanes (SIMD128, NEON) | 0.34 | (as step 8) | 0.26 |
| 10. the baseline build a sample at a time in the plane | (as step 9) | 0.77 | (as step 9) |
| 11. the walk scans the transform sizes | 0.32 | 0.77 | 0.23 |

rav1d's loop filter in the simd build was 0.59 ms a frame in the profile
above.

The state after each step, fps and the ratio to rav1d's (median of 7, load
average 5 to 12, the builds taking turns from step 9 on):

| step | clip | baseline | simd | native |
|---|---|---|---|---|
| before (`829b1e1`), same run as step 8 | fixture 320x180 | 742 (0.63x) | 1055 (0.62x) | 1218 |
|  | testsrc2 720p | 120 (0.70x) | 181 (0.61x) | 203 |
|  | mandelbrot 720p | 48.6 (0.76x) | 94.3 (0.66x) | 111 |
|  | mandelbrot 720p 10-bit | 60.4 (0.82x) | 126 (1.57x) | 148 |
|  | mandelbrot 1080p | 24.1 (0.86x) | 51.0 (0.69x) | 60.8 |
| 8. the loop filter's edge decisions: the transform size first, levels from a table | fixture 320x180 | 750 (0.64x) | 1098 (0.65x) | 1272 |
|  | testsrc2 720p | 119 (0.69x) | 180 (0.61x) | 207 |
|  | mandelbrot 720p | 49.5 (0.77x) | 95.0 (0.67x) | 114 |
|  | mandelbrot 720p 10-bit | 60.8 (0.82x) | 130 (1.61x) | 150 |
|  | mandelbrot 1080p | 24.3 (0.86x) | 50.6 (0.68x) | 61.9 |
| 9. the loop filter in the plane, eight 16-bit lanes at a time (SIMD128, NEON) | fixture 320x180 | 754 (0.64x) | 1418 (0.83x) | 1704 |
|  | testsrc2 720p | 121 (0.69x) | 189 (0.63x) | 220 |
|  | mandelbrot 720p | 50.0 (0.77x) | 108 (0.74x) | 130 |
|  | mandelbrot 720p 10-bit | 61.8 (0.82x) | 140 (1.71x) | 169 |
|  | mandelbrot 1080p | 24.8 (0.77x) | 57.0 (0.75x) | 70.6 |
| 10. the wasm baseline build filters a sample at a time in the plane | fixture 320x180 | 861 (0.76x) | 1366 (0.82x) | 1745 |
|  | testsrc2 720p | 124 (0.71x) | 190 (0.64x) | 233 |
|  | mandelbrot 720p | 53.3 (0.82x) | 108 (0.74x) | 136 |
|  | mandelbrot 720p 10-bit | 63.8 (0.85x) | 139 (1.72x) | 167 |
|  | mandelbrot 1080p | 26.2 (0.81x) | 57.6 (0.76x) | 69.3 |
| 11. the walk scans the transform sizes for the units on a transform edge | fixture 320x180 | 885 (0.77x) | 1404 (0.82x) | 1685 |
|  | testsrc2 720p | 124 (0.71x) | 189 (0.63x) | 216 |
|  | mandelbrot 720p | 53.0 (0.82x) | 107 (0.74x) | 129 |
|  | mandelbrot 720p 10-bit | 63.7 (0.85x) | 139 (1.71x) | 164 |
|  | mandelbrot 1080p | 26.1 (0.81x) | 57.1 (0.76x) | 68.5 |

### 8. The loop filter's edge decisions: the transform size first, levels from a table

The edge loop of 7.14.2 ran as written for every 4x4 unit of every plane in both passes: the unit's mode info loaded, its block size looked up, and only then the question whether it is on a transform edge, which seven units in eight on this clip are not; for the ones that are, the level derived from the header for each unit (7.14.4: the segment's feature, the reference's and the mode's deltas, three clamps). `edge_unit` makes the same decision in the order that costs least: the transform size, one byte, first; the mode info only on a transform edge; the block size only for a skipped inter block, the one case it matters to. The level is a lookup: for a block with no `delta_lf` it depends on the segment, the first reference frame and the kind of mode alone, so the frame's 576 levels and the limits of each of the 64 level values are worked out once (`LfLevels`), and a block with a `delta_lf` goes through the same function the table was filled with. Units are handed on two at a time along their edge, which is what the kernels of the next steps take; the sample filtering itself is unchanged here. The specification's walk and its strength process stay in the tests as the reference: random frames of every layout, depth and odd size (tilings of blocks and transforms, and mode info that belongs to no tiling) are filtered both ways on one thread and on three and compared sample for sample, and ten planted errors (a wrong neighbour, a missing mode delta, an off-by-one at the frame's edge, ...) each fail them. Worth 5 to 13 % of the loop filter, which is 0 to 4 % of a frame and inside the noise on some rows.

### 9. The loop filter in the plane, eight 16-bit lanes at a time (SIMD128, NEON)

Each unit's edge was gathered into a 14 x 4 array of 32-bit values, a bounds-checked load a sample; every filter its size allows was computed for all four lanes and one result selected per lane; the array was scattered back. `filter_pair` filters two units' worth of an edge, eight samples, where they lie. Across a horizontal edge each row is one load of eight 16-bit lanes and each changed row one store. Across a vertical edge eight rows of eight samples (two such blocks for the 16-sample size) are transposed into vectors of one column each, filtered, and transposed back, as rows of four when only the narrow filter's four columns changed. There is a kernel for each filter length (4, chroma's 6, 8, 16) and direction. The mask of all eight lanes comes first, and the kernel returns if none passes; the narrow filter, the 7-tap (chroma's 5-tap) one and the 13-tap one are each computed only if some lane takes them; and a wide filter's sums run from one output to the next, each the one before plus two samples and less two, instead of being added up afresh. The two units keep their own limits, one value in each half of a vector; a unit that is not filtered, or whose size is another kernel's, gets a `blimit` of -1, which no lane passes. 16 bits hold all of it: with samples of at most 12 bits the mask's sum and the narrow filter's `3 * (q0 - p0)` plus a clamped term stay inside an `i16`, and the 13-tap sum (16 samples and the rounding) inside a `u16`.

The kernels are written once, as a macro over two dozen lane operations, and expanded for SIMD128 and for NEON: the same arithmetic in both by construction, so native aarch64 gains with wasm instead of falling behind it. `in_plane_kernels_match_the_process` holds them to the one-sample process of 7.14.6 on 100,000 random edges (every length, direction and depth, the two halves with limits of their own, either half off, the whole plane compared so that a stray write shows), and `frames_match_the_reference_walk` runs them under the walk of step 8. Both run natively, which is the NEON expansion, and as a `wasm32-wasip1` test binary in Node (`tools/wasm-perf/wasi-test.sh`), which is the SIMD128 one in V8. Fourteen planted errors (a rounding constant, a tap, a threshold, a transposition, a store one sample off, the halves' limits swapped, ...) each fail them, and so did a real one on the way: the right half of the 16-sample block loaded four columns off. The loop filter is 4.2 times faster in the simd build and 3.8 times natively; the frame 3 to 30 % (the fixture, a libaom stream filtered on every frame, gains most). x86-64 and the wasm baseline build keep the lane filter, under the new walk.

### 10. The wasm baseline build filters a sample at a time, in the plane

A module without `simd128` has no vector lanes, so the lane filter was the worst of both there: every filter the edge's size allows, computed for all four lanes in scalar 32-bit arithmetic, inside the same gather and scatter. `filter_unit` is the one-sample process of 7.14.6 run on the plane. Each of a unit's four samples reads the samples across the edge as its mask asks for them, the nearest four first, which is enough to drop the quarter that fail; it computes the one filter it takes and writes the samples that filter changes, with a wide filter's sums carried from one output to the next as in step 9. Held to the same reference by the same two tests, natively and as a baseline `wasm32-wasip1` binary in Node; eight planted errors in its arithmetic fail them, and one in the walk's call of it fails the frame test in wasm. The baseline build's loop filter goes from 2.20 to 0.83 ms a frame, the frame 3 to 17 % faster. The other builds are untouched (their rows moved by under 2 %, both ways). Only wasm32 without `simd128` takes this path: on x86-64 the lane filter is compiled to SSE2 or AVX2 and which of the two is faster there could not be measured here, so x86-64 keeps it.

### 11. The walk scans a row of transform sizes for the units on a transform edge

With the kernels in, the walk was half of the loop filter natively (0.12 of 0.25 ms a frame, measured with the kernel call taken out) and the largest single function of it in the simd build's profile. Seven units in eight are not on a transform edge, and `edge_unit` said so only after indexing the transform sizes through their vector, looking the width up in a table of machine words, and being called twice a pair. `filter_band` now takes the row of sizes as a slice (two rows for the vertical edges, whose pairs are one unit above the other) and tests each byte against a 32-entry byte table of the frame's `LfLevels`, a transform's width or height in units less one: a unit is on a transform edge where its column or row has none of those bits. Only the units that pass go through `edge_unit`, which still makes the whole decision, so the scan can only save work, not change a result. Six planted errors in it (the other pass's table, a unit of the pair tested with its neighbour's size or row, ...) fail the frame test. Measured in a run of its own against step 10, median of 7 natively and of 5 in wasm, builds taking turns: the loop filter 0.255 -> 0.224 ms a frame natively, 0.35 -> 0.27 in the simd build, 0.76 -> 0.77 in the baseline build (no change: there the sample filtering is four fifths of it). The run of the table above has 0.256 -> 0.233 natively and 0.34 -> 0.32 in the simd build: a tenth of the loop filter, which is 0.3 % of a frame and does not show in the fps (the step's rows were measured against step 10 in one run: 883 / 1402 / 1667, 124 / 189 / 218, 53.3 / 106 / 128, 63.7 / 138 / 164, 26.2 / 56.9 / 68.6).

## Where the loop filter stands after step 11 (`1dbf90e`)

The fourteen streams, median of 7, the builds taking turns (wasm) and the
three binaries taking turns (native), load average 5 to 8. `before` is
`829b1e1`; the ratio is to rav1d.

| stream | native: rav1d | before | after | wasm baseline: rav1d | before | after | wasm simd: rav1d | before | after |
|---|---|---|---|---|---|---|---|---|---|
| fixture 320x180, 8-bit | 1486 | 1207 | 1626 (1.09x) | 1158 | 732 | 882 (0.76x) | 1688 | 1055 | 1409 (0.83x) |
| fixture 160x90, 10-bit | 3428 | 3058 | 4129 (1.20x) | 2744 | 1997 | 2348 (0.86x) | 2786 | 2752 | 3568 (1.28x) |
| fixture 128x72, film grain | 2837 | 2273 | 2560 (0.90x) | 2298 | 1555 | 1672 (0.73x) | 3169 | 1997 | 2233 (0.70x) |
| fixture 177x99, 8-bit | 2269 | 2565 | 3290 (1.45x) | 1850 | 1376 | 1551 (0.84x) | 3121 | 2201 | 2776 (0.89x) |
| fixture 96x64, 4:4:4 | 4148 | 3826 | 4690 (1.13x) | 3341 | 2322 | 2525 (0.76x) | 4970 | 3381 | 4327 (0.87x) |
| fixture 96x64, mono | 10365 | 8358 | 9604 (0.93x) | 8938 | 6176 | 6792 (0.76x) | 11598 | 8044 | 8832 (0.76x) |
| AOM b8-05-mv 352x288 | 195 | 228 | 243 (1.25x) | 154 | 139 | 141 (0.92x) | 248 | 195 | 204 (0.82x) |
| AOM b10-00-quantizer-10, 10-bit | 89.0 | 95.5 | 106 (1.19x) | 70.1 | 59.0 | 62.5 (0.89x) | 72.3 | 78.9 | 88.4 (1.22x) |
| AOM b8-23-film_grain-50 | 410 | 379 | 443 (1.08x) | 338 | 214 | 236 (0.70x) | 501 | 332 | 386 (0.77x) |
| AOM b8-00-quantizer-20 | 234 | 250 | 283 (1.21x) | 189 | 150 | 157 (0.83x) | 272 | 217 | 241 (0.89x) |
| SVT testsrc2 720p | 228 | 202 | 212 (0.93x) | 172 | 119 | 122 (0.71x) | 293 | 179 | 186 (0.64x) |
| SVT mandelbrot 720p | 83.4 | 114 | 130 (1.56x) | 64.6 | 49.1 | 53.1 (0.82x) | 143 | 95.1 | 107 (0.75x) |
| SVT mandelbrot 720p, 10-bit | 97.5 | 151 | 165 (1.69x) | 73.2 | 58.8 | 61.9 (0.85x) | 80.0 | 125 | 135 (1.69x) |
| SVT mandelbrot 1080p | 41.1 | 61.7 | 68.4 (1.66x) | 32.0 | 24.5 | 26.0 (0.81x) | 74.8 | 51.7 | 56.7 (0.76x) |

The loop filter is 4.5 times faster in the simd build (1.44 to 0.32 ms a
frame of mandelbrot 720p, where rav1d's takes 0.59), 3 times in the baseline
build (2.30 to 0.77) and 5.4 times natively on aarch64 (1.26 to 0.23). No
row of any column got slower. The streams that gain most are wasm-av1's
small libaom fixtures, 10 to 34 % faster in the simd build and 13 to 35 %
natively; the AOM vectors gain 5 to 16 % and 7 to 17 %, and SVT-AV1's clips
4 to 13 % and 5 to 14 % (on mandelbrot 720p the loop filter runs on 30 of
the 60 frames). One thread, against rav1d: the
baseline build is at 0.70 to 0.92 (0.63 to 0.90 before, in the same run),
the simd build at 0.64 to 0.89 on 8-bit streams (0.61 to 0.80) and 1.22 to
1.69 on 10-bit ones, native aarch64 at 0.90 to 1.69 (0.80 to 1.55).

**What it was held to.** Every step went through `tools/wasm-perf/gate.sh`
(the list under "What every step was held to", x86-64 included). The
loop filter has three tests of its own, each against code that is the
specification as written: the level tables against the adaptive filter
strength process, whole random frames against the specification's walk
with the lane filter, and the in-plane kernels against the one-sample
process; a fourth pins the kernels' bounds checks from both sides. They run
natively (the NEON kernels), on x86-64 under Rosetta (the lane filter under
the new walk), and, as a `wasm32-wasip1` test binary in Node, in V8 without
and with `simd128` (the one-sample kernel and the SIMD128 ones:
`tools/wasm-perf/wasi-test.sh`; the bounds test natively only, since a
panic cannot be caught in that binary). Forty-five errors were planted
across the walk, the tables and the kernels, one at a time; all but one
failed a test, and that one, a bound in an assertion that no arithmetic
test can see, is
what the bounds test was then written for (it found that the one-sample
kernel's own check was a sample short for the 8-sample length; fixed, and
never reachable from the walk). At the end, on `1dbf90e`: all 244 AOM
vectors natively on 1 thread and on 4 (244 of 244 both times), and the 242
that are IVF through both wasm builds against rav1d (242 of 242 agree).

**x86-64** has the new walk and its old arithmetic. It could be measured
only under Rosetta, which has no AVX2, so that is the lane filter as SSE2:
the loop filter on mandelbrot 720p went from 2.43 to 1.30 ms a frame there,
the decode from 39.7 to 41.6 fps. Nothing was measured on an AVX2 host.

**What is left in the loop filter**, in the order of what it would buy:

1. x86-64 still gathers each edge into the 32-bit lane filter. The in-plane
   kernels are a macro over some two dozen lane operations; a third set of
   them (SSE2, which every x86-64 has, or AVX2) would give x86-64 the same
   kernels, and Rosetta can hold them to the same tests. Whether they beat
   the AVX2 lane filter on a real AVX2 host has to be measured on one.
2. The walk was half of the loop filter natively before step 11 and is
   still a large part of it. What it costs is per filtered unit: the block's mode info read to find a level that is
   the same for the whole block, a `Pair` built for the kernel, the kernel
   found by a match. A byte per unit written when the block is decoded (the
   level's index, and whether the block is a skipped inter block) would
   take the mode info out of it; that is a change to the tile decoder.
3. The baseline build's 0.77 ms is four fifths scalar arithmetic, most of
   it the 13-tap filter a sample at a time. Four 16-bit lanes in a 64-bit
   word would do a unit's sums at once; the masks and, for vertical edges,
   the transposition eat much of that, and it was not tried.

## The residual path, written from the specification

A second round on two of the four parts above, the symbol decoder with the
coefficient reader and the inverse transforms with reconstruction, under a
stricter rule than the first: nothing but the specification and this
crate's own code as the source, rav1d only as a black box to time. Each
idea that is common knowledge from other decoders is named as such where
it is used, and was derived and checked here against this crate's
reference code, not taken from anywhere.

**How the numbers are taken.** As above (wasm-av1 at `ee33e98`, Node 26,
one thread, the M3 Max), with two differences. The machine was busier
(load average 8 to 20 throughout, other builds running), so every
comparison is of builds run interleaved, clip by clip, median of 7; the
absolute numbers are lower than the tables above for the same code, and
`before` is measured again each time rather than copied. And the two parts
are measured directly: in wasm as V8's profile (200 us sampling) of
`coeffs_decode` with the symbol decoder, and of `itx::*` with
`reconstruct`, as milliseconds a frame on mandelbrot 720p; natively as
timers around `coeffs()` and `reconstruct()` in `transform_block`
(`parts.sh` and `parts-native.sh` in `tools/wasm-perf`). Three
microbenchmarks there decided between variants before anything went into
the decoder: `symbench` (the symbol decoder alone, on the symbols of a real
decode recorded by `record.sh`), `coefbench` (the coefficient reader's two
passes, on the recorded transform blocks) and `itxbench` (the 2D inverse
transform). The steps quote them.

Each row is the state after the step: fps, and its ratio to rav1d's in the
same run. `before` is `829b1e1`. The rows of different steps come from
different runs on a machine whose load kept changing: across steps compare
the ratios, and for what a step bought, its own paragraph, which gives the
step before and after from one run.

| step | clip | baseline | simd | native |
|---|---|---|---|---|
| before | fixture 320x180 | 732 (0.64x) | 1036 (0.60x) | 1198 |
| | testsrc2 720p | 119 (0.69x) | 180 (0.63x) | 195 |
| | mandelbrot 720p | 47.9 (0.75x) | 93.6 (0.67x) | 108 |
| | mandelbrot 720p 10-bit | 59.7 (0.82x) | 123 (1.56x) | 143 |
| | mandelbrot 1080p | 24.0 (0.78x) | 50.4 (0.70x) | 58.8 |
| 8. the symbol decoder keeps its value at the top of a 64-bit window | fixture 320x180 | 736 (0.64x) | 1064 (0.62x) | 1205 |
| | testsrc2 720p | 120 (0.69x) | 187 (0.65x) | 205 |
| | mandelbrot 720p | 49.3 (0.77x) | 97.4 (0.69x) | 114 |
| | mandelbrot 720p 10-bit | 61.0 (0.84x) | 130 (1.65x) | 150 |
| | mandelbrot 1080p | 24.3 (0.79x) | 52.2 (0.72x) | 61.5 |
| 9. coefficient symbols read without a branch on what they are | fixture 320x180 | 760 (0.63x) | 1101 (0.62x) | 1291 |
| | testsrc2 720p | 135 (0.76x) | 211 (0.70x) | 231 |
| | mandelbrot 720p | 51.3 (0.79x) | 104 (0.72x) | 125 |
| | mandelbrot 720p 10-bit | 66.0 (0.88x) | 147 (1.80x) | 166 |
| | mandelbrot 1080p | 25.3 (0.76x) | 55.1 (0.73x) | 65.3 |
| 10. the level pass: each scan worked out once, a map of `coeff_base` symbols, the signs by a bit each | fixture 320x180 | 756 (0.64x) | 1103 (0.64x) | 1295 |
| | testsrc2 720p | 129 (0.75x) | 205 (0.70x) | 242 |
| | mandelbrot 720p | 52.9 (0.81x) | 110 (0.76x) | 135 |
| | mandelbrot 720p 10-bit | 65.0 (0.87x) | 149 (1.84x) | 175 |
| | mandelbrot 1080p | 25.9 (0.80x) | 58.3 (0.78x) | 70.1 |
| 11. the 1D inverse transforms compiled for each size | fixture 320x180 | 767 (0.65x) | 1110 (0.65x) | 1284 |
| | testsrc2 720p | 136 (0.78x) | 205 (0.69x) | 243 |
| | mandelbrot 720p | 55.8 (0.86x) | 116 (0.80x) | 135 |
| | mandelbrot 720p 10-bit | 67.0 (0.92x) | 154 (1.90x) | 171 |
| | mandelbrot 1080p | 27.0 (0.84x) | 58.5 (0.78x) | 71.1 |
| 12. the inverse transforms' lane arithmetic in SIMD128 operations | fixture 320x180 | 793 (0.66x) | 1159 (0.64x) | 1355 |
| | testsrc2 720p | 144 (0.77x) | 225 (0.72x) | 260 |
| | mandelbrot 720p | 57.4 (0.87x) | 122 (0.85x) | 146 |
| | mandelbrot 720p 10-bit | 69.2 (0.90x) | 159 (1.90x) | 185 |
| | mandelbrot 1080p | 27.9 (0.85x) | 63.4 (0.82x) | 74.7 |
| 13. reconstruction from the end-of-block position: the scan dequantised, the transform bounded | fixture 320x180 | 802 (0.64x) | 1189 (0.65x) | 1347 |
| | testsrc2 720p | 150 (0.79x) | 241 (0.73x) | 250 |
| | mandelbrot 720p | 59.0 (0.87x) | 128 (0.84x) | 143 |
| | mandelbrot 720p 10-bit | 67.2 (0.85x) | 157 (1.84x) | 189 |
| | mandelbrot 1080p | 27.4 (0.85x) | 64.3 (0.83x) | 76.0 |
| 14. a coefficient is dequantised where its sign is read | fixture 320x180 | 801 (0.64x) | 1179 (0.65x) | 1401 |
| | testsrc2 720p | 152 (0.81x) | 244 (0.77x) | 282 |
| | mandelbrot 720p | 60.4 (0.88x) | 133 (0.87x) | 157 |
| | mandelbrot 720p 10-bit | 72.8 (0.91x) | 174 (2.02x) | 202 |
| | mandelbrot 1080p | 28.9 (0.85x) | 68.4 (0.87x) | 80.9 |

The two parts, in milliseconds a frame on mandelbrot 720p:

| after step | symbol decoder and coefficients: simd | baseline | native | transforms and reconstruction: simd | baseline | native |
|---|---|---|---|---|---|---|
| before | 2.92 | 2.82 | 2.46 | 1.66 | 2.96 | 0.97 |
| 8 | 2.40 | 2.42 | 1.98 | 1.60 | 2.94 | 0.97 |
| 9 | 2.06 | 2.10 | 1.71 | 1.62 | 2.96 | 0.97 |
| 10 | 1.45 | 1.47 | 1.08 | 1.61 | 2.98 | 0.98 |
| 11 | 1.46 | 1.45 | 1.06 | 1.45 | 2.04 | 0.84 |
| 12 | 1.44 | 1.49 | 1.05 | 1.19 | 2.00 | 0.83 |
| 13 | 1.44 | 1.48 | 1.05 | 0.90 | 1.83 | 0.66 |
| 14 | 1.49 | 1.57 | 1.09 | 0.61 | 1.48 | 0.46 |

### 8. The symbol decoder keeps its value at the top of a 64-bit window

Where the time went first: a frame of mandelbrot 720p is 1020 transform blocks, 124 000 coefficients read (90 000 of them zero) and 34 000 signs, against 12 000 symbols for everything else in the frame. A recording of the symbols of twelve frames, decoded alone, cost 9.3 ns a symbol in wasm and natively: a 4-symbol read 9.5 ns, an equiprobable bit 5.9 ns in wasm and 9.1 natively. Every renormalisation computed a byte position and a bit offset, loaded three bytes with a bounds test, and kept `SymbolMaxBits` in 64 bits, for a shift of one or two bits.

8.2.6 renormalises with `SymbolValue = paddedData ^ ( ( ( SymbolValue + 1 ) << bits ) - 1 )`. The right-hand operand is `SymbolValue << bits` with ones below, so this is the value shifted up with the stream's next bits shifted in complemented (`init_symbol` starts it the same way; the bits past the end are zeros, so ones). `SymbolValue` is then the top of one long number, and since a symbol subtracts from it no more than it holds, nothing below is touched by decoding: the bits to come can sit under it ahead of time. The decoder now holds `SymbolValue` in the top 16 bits of a 64-bit word with up to 48 complemented stream bits below, renormalises with two shifts, and reloads six or seven bytes at once when they run out (about every thirty symbols), a byte at a time only at the end of a tile. The read position and `SymbolMaxBits` are no longer kept: both follow from the count of bits taken in. (A window refilled a word at a time is how arithmetic decoders are commonly built, and it is what the first round guessed rav1d has; what is here is derived from the formula above.)

The decoder as 8.2 writes it stays in the crate as `ReferenceSymbolDecoder`. Tests read mixed symbols with both, from encoded streams and from arbitrary bytes of every length from 0 to 40, and require the same symbol, CDF and state (value, range, position, `SymbolMaxBits`) after each, past the end of the data included. And `CoefficientReader::Checked` now puts the reference symbol decoder beside the window one for everything in the tile from its first transform block on (`Coder::Both`): the shared coefficient reader, the mode info, the motion vectors. Three planted differences were each caught by the committed vectors in that mode: a context clamp of 2 for 3 in the fast coefficient reader, the lowest bit of a refill dropped, `>` for `>=` in `read_bool`.

The recording after the change: 6.9 ns a symbol (4-symbol 7.3, bit 3.2 in wasm and 4.1 natively). Three further variants were measured on it and left out, none of them a gain in wasm: the 4-symbol search without branches (every threshold computed, the symbol counted) is level with the loop natively and in wasm; the adaptation as one vector operation on four lanes is slower everywhere (9.1 against 7.1 ns in wasm simd: the next read of the same CDF waits on a store it cannot forward from); an equiprobable bit without a branch changes nothing in the decoder itself.

### 9. Coefficient symbols read without a branch on what they are

After step 8 a 4-symbol read still cost 7.3 ns, about 29 cycles, for a dependent chain of perhaps ten. The rest is branches. The search stopped at the symbol with a conditional branch per threshold, and the adaptation was two loops whose lengths are the symbol, so each read ended in branches on a value that cannot be predicted: that it cannot is what its bits are for (the recording averages 0.87 bits a symbol, most of them zeros with ones scattered in). About a read in three was mispredicted. Alone that is a pipeline refill; in the coefficient loop it also discards the contexts the processor had already worked out for the next coefficients, which is why the loop cost 11 ns a coefficient around a read that cost 7.

`read_symbol_fixed` for two, three and four symbols (every CDF the coefficient loop reads: `coeff_base`, `coeff_base_eob`, `coeff_br`, `dc_sign`) now has no branch that depends on the symbol. Every threshold of 8.2.6 is computed. They decrease strictly from one symbol to the next (a CDF does not decrease, and the `EC_MIN_PROB` term does), so the symbol is the number of thresholds above `SymbolValue`, and the entries the adaptation moves towards zero, those below the symbol, are exactly those whose threshold is above it: each entry's two possible adaptations are computed and one is selected by that entry's own comparison, and the two ends of the new interval are selected the same way (`core::hint::select_unpredictable`, so that they stay selects). What this takes for granted of a CDF, that it does not decrease and ends at `1 << 15`, is tested for every default coefficient CDF and for the adaptation, which keeps both. The general `read_symbol` is unchanged, and so is everything that is not a coefficient.

On a recording of the level and sign passes of twelve frames (16 000 blocks, 1.6 million coefficients) replayed outside the decoder, in a loop that differs in the reads alone: 12.1 to 9.5 ns a coefficient natively, 12.7 to 11.0 in wasm. In the decoder, from one run: the part 2.37 to 2.06 ms in wasm simd and 1.98 to 1.71 natively; testsrc2 720p 198 to 211 fps simd and 124 to 135 baseline, mandelbrot 720p 101.2 to 103.8 and 50.8 to 51.3, natively 120 to 125. Two variants measured on the way and left out: one arm for each symbol (the search's exit for a symbol doing that symbol's renormalisation and adaptation, so that the adaptation has no loop) is 12 % faster than the loops on the symbols alone and 3 to 5 % in the coefficient loop, where the mispredictions stay; the adaptation without a branch but computed from the symbol after the search (masks) is slower than the loops, because the next read of the same CDF then waits for the whole search.

### 10. The level pass: each scan worked out once, a map of `coeff_base` symbols, the signs by a bit each

With the reads no longer mispredicted, what bounds the level pass is how many instructions surround each read: about 135 a coefficient natively, at four or so a cycle, against a dependent chain of twenty cycles. The pass was rewritten to need fewer, each change measured on the replay of step 9 before it went in:

- **The scan, worked out once.** For each transform size and class a table gives, for every position of the scan, where the coefficient lies in the level map, the index of the `coeff_base` CDF its position alone gives it, and the position: one load for what was a row and column split, two clamps and a three-dimensional table lookup for every coefficient. The tables are built from `scan_for` and `coeff_base_ctx_of` the first time they are needed, and a test holds every entry to those two for every size, type, position and neighbour sum.
- **A map of `coeff_base` symbols.** The padded level map (now with one stride, 36, so that the neighbours' offsets are constants) holds `Min( level, 3 )`, which is the `coeff_base` symbol as read, so a context is five loads, four additions and a 16-entry table. The levels themselves go to `Quant`, where the reference keeps them; the context of `coeff_br` (one coefficient in seventy has one) is computed from there as the reference computes it, edge tests and all, in a function out of line.
- **The symbol decoder in registers.** For the two passes it is a local that nothing takes the address of: the byte-at-a-time refill and the `coeff_br` reader take and return it by value.
- **The signs by a bit each.** The level pass sets a bit for every coefficient that is not zero; the sign pass walks the bits, 34 000 of the 124 000 coefficients of a frame, and no longer tests each coefficient for zero, a test that was mispredicted as often as the symbols were.

A padded level map was measured in the first round at between -5 % and +6 % and left for later; step 6 then put one in. Measured again here, interleaved on the replay: the pass with the padded map of step 6 and searching reads is 14.2 ns a coefficient natively and 16.2 in wasm; this step's map, tables and bits bring that to 12.1 and 12.7, the reads of step 9 to 9.5 and 11.0, the `Min( level, 3 )` map with the CDF index in the table and `coeff_br` out of line to 8.1 and 10.5. A padded map pays, and more once nothing around it is mispredicted.

In the decoder, from one run: the part 2.11 to 1.45 ms in wasm simd, 2.15 to 1.47 in baseline, 1.71 to 1.08 natively; mandelbrot 720p 103.3 to 110.3 fps simd, 51.2 to 52.9 baseline, 124.5 to 135.3 natively. The part is now below the 1.6 ms measured for rav1d's symbol decoder and coefficient reader in the first round. What it still costs is the reads themselves: the dependent chain of an adaptive read is about twenty cycles when the next read uses the same CDF, which it often does.

### 11. The 1D inverse transforms compiled for each size

A frame of mandelbrot 720p reconstructs 890 transform blocks, 367 000 samples; 184 of the blocks are DC only and take step 5's shortcut. The rest go through `run_lanes`, which called the processes of 7.13.2 with the size as a variable. The DCT of 7.13.2.3 is one process for 4 to 64 points, each stage guarded by the size, and compiled once for all sizes it tested `n` before every group of butterflies, worked out the bit-reversed index of every element of the permutation in a loop as it ran, and fetched each rotation's cosine and sine from the table through four comparisons. `run_lanes` now matches on the kind and the size and calls the same process with the size a constant in each arm, so each is compiled with its stages, permutation and angles decided: the "one function per size" that the list above asked for, got without writing any of them. Nothing in the arithmetic is touched; a test runs every kind and size against the process called with the size hidden from the compiler, eight lanes and one value at a time (a 16-point DCT arm calling the 32-point network, planted, fails it and three other tests; an identity arm of the wrong size fails it alone).

The part, from one run: 1.60 to 1.45 ms in wasm simd, 2.95 to 2.04 in baseline (whose transforms are the scalar ones, a row or a column at a time), 0.95 to 0.84 natively. Mandelbrot 720p 110.0 to 116.2 fps simd, 53.2 to 55.8 baseline, 133.5 to 135.0 natively; the simd module is 118 KB larger (3.80 to 3.92 MB), the baseline one 8 KB.

### 12. The inverse transforms' lane arithmetic in SIMD128 operations

The transforms run eight rows or columns at once on `Lanes`, eight 32-bit values whose arithmetic is written lane by lane for the compiler to turn into vectors. It does that for aarch64. For wasm32 it did it badly: the 2D transform alone, the same source timed natively and in wasm simd (`tools/wasm-perf/itxbench`), took 495 and 859 ns for a full 16x16 block, 1675 and 3268 for 32x32, 4915 and 11 111 for 64x64, twice the native time. With `simd128` the five operations of `Lane` (the rotation, the clamped butterfly, negation, the two scalings) are now written as SIMD128 operations on the two halves, the same wrapping 32-bit arithmetic: 646, 2125 and 5990 ns. Native code is the lane-by-lane form as before, which is why this has no NEON twin: nothing changed there, and the rows above agree.

`cargo test` does not run on wasm32, so the comparison with the scalar reference that the unit test makes natively is made in the wasm module too: `itxbench` builds the tree's `dsp/itx.rs` into a module that runs every size and type against `inverse_transform_2d_scalar` (14 960 cases) before it times anything, and `gate.sh` runs it. An unsigned shift planted in the SIMD128 rotation fails it in the simd build alone.

The part, from one run: 1.44 to 1.19 ms in wasm simd; mandelbrot 720p 117.5 to 121.8 fps, testsrc2 720p 213 to 225.

### 13. Reconstruction from the end-of-block position: the scan dequantised, the transform bounded

`reconstruct` walked the whole block twice whatever it held: every one of its positions was tested and dequantised in 64-bit arithmetic (333 000 a frame, for 34 000 coefficients that are not zero), and the inverse transform gathered every group of eight rows to find out whether it was zero, clamped the whole block in a pass of its own between the two directions, and stored zero rows for the column transforms to load. A decoder knows more than that: `coeffs()` returns `eob`, and nothing past the first `eob` positions of the scan is anything but zero.

`reconstruct_decode` is the process for a decoder (`reconstruct` stays, untouched, for the encoder, for lossless blocks and for 12 bits). It dequantises the first `eob` positions of the scan and no others: every one of them, zero or not, with no branch (whether a coefficient is zero is no more predictable than its level was), in 32 bits (the specification keeps the low 24 bits of the product, which are those of its low 32), noting by selects how many rows and columns the coefficients that are not zero reach. `inverse_transform_2d_within` is then given those two numbers: it transforms the groups of rows up to the last that holds a coefficient, clamps each as it stores it, and the column transforms take the rows that were never stored as the zeros they are. That is exact, not an approximation: every step of 7.13.2 maps zeros to zeros. `inverse_transform_2d` itself is now that function with the whole block as its bounds, so there is one eight-lane implementation, and the encoder's results are what they were.

Tests: the bounded transform against the scalar reference for every size and type, with the tightest bounds and with any looser ones, natively and in the wasm modules (a group of rows left out, planted, fails it); the dequantiser against 7.12.3 written out in 64 bits over the whole block, for every size and scan, with and without a quantiser matrix, levels up to 20 bits (a sign taken from the wrong bit, or a column too few reported, fails it).

The part, from one run: 1.20 to 0.90 ms in wasm simd, 1.98 to 1.83 in baseline, 0.84 to 0.66 natively; mandelbrot 720p 123.1 to 128.3 fps simd, natively 137.5 to 142.7; testsrc2 720p 230 to 241 simd. The part is below the 1.1 ms measured for rav1d's in the first round.

Two things were written for this step, verified, measured and left out (`tools/wasm-perf/tried/`). Transforms that know how many of their inputs are zero (the network run on a copy whose zero entries the compiler can see, so that what they feed folds away, with the network's loops expanded so that it can): a 32-point DCT of 8 inputs takes 80 ns for eight lanes against 231, but on these streams the reconstruction got 3 % slower, their blocks being mostly full, and the crate took five times as long to compile. And a 32-bit single-lane path for the baseline build, which has only the scalar transforms: no faster than the 64-bit one (6.4 against 6.1 us for a 32x32 block), since what the baseline build lacks is lanes, not narrower arithmetic.

### 14. A coefficient is dequantised where its sign is read

After step 13 dequantisation was the scan walked once more: 124 000 positions a frame loaded, dequantised and stored for the 34 000 that hold something, 0.27 ms of the simd frame. The coefficient reader's sign pass already visits exactly those 34 000, each with its position and its value in hand as it completes it. It now dequantises each there (`Dequantiser::put`: the same arithmetic as step 13's, one function shared by both) and keeps count of the rows and columns they reach, and `coeffs_decode` hands that to `reconstruct_decode`, which then has only the transform and the sum left to do. A block read by the reference reader alone (`CoefficientReader::Reference`) is dequantised by the walk of step 13, as before. The test of the dequantiser now builds `Dequant` both ways and requires the same array and the same reach (the position split by the wrong dimension, planted, fails it; the DC coefficient dequantised as an AC one fails the committed vectors).

This moves work from one column of the table of parts to the other, so the two are to be read together: from one run, 1.41 + 0.87 = 2.28 to 1.49 + 0.61 = 2.10 ms in wasm simd, 1.47 + 1.79 to 1.57 + 1.48 in baseline, 1.04 + 0.65 = 1.69 to 1.09 + 0.46 = 1.54 natively. Mandelbrot 720p 129.4 to 132.5 fps simd, 59.7 to 60.4 baseline, 153.3 to 157.0 natively; testsrc2 720p 239 to 244 simd, 276 to 282 natively.

## Where the residual path stands after step 14 (`d3e6ac4`)

All the steps in one run, each build beside the others clip by clip, median of 7 (load average 4 to 18 while it ran): fps and the ratio to rav1d's. Wasm simd:

| after step | fixture 320x180 | testsrc2 720p | mandelbrot 720p | mandelbrot 720p 10-bit | mandelbrot 1080p |
|---|---|---|---|---|---|
| rav1d | 1829 | 328 | 155 | 86.5 | 79.3 |
| before | 1144 (0.63x) | 195 (0.59x) | 103 (0.66x) | 140 (1.62x) | 55.5 (0.70x) |
| 8 | 1150 (0.63x) | 205 (0.62x) | 108 (0.70x) | 146 (1.69x) | 57.9 (0.73x) |
| 9 | 1163 (0.64x) | 212 (0.65x) | 111 (0.72x) | 150 (1.74x) | 58.5 (0.74x) |
| 10 | 1175 (0.64x) | 221 (0.67x) | 119 (0.77x) | 157 (1.81x) | 62.1 (0.78x) |
| 11 | 1176 (0.64x) | 224 (0.68x) | 119 (0.77x) | 160 (1.85x) | 63.3 (0.80x) |
| 12 | 1180 (0.65x) | 229 (0.70x) | 123 (0.80x) | 165 (1.91x) | 64.6 (0.81x) |
| 13 | 1191 (0.65x) | 236 (0.72x) | 129 (0.83x) | 168 (1.94x) | 66.6 (0.84x) |
| 14 | 1195 (0.65x) | 241 (0.74x) | 132 (0.85x) | 171 (1.98x) | 68.2 (0.86x) |

Wasm baseline:

| after step | fixture 320x180 | testsrc2 720p | mandelbrot 720p | mandelbrot 720p 10-bit | mandelbrot 1080p |
|---|---|---|---|---|---|
| rav1d | 1253 | 192 | 68.9 | 79.9 | 33.5 |
| before | 786 (0.63x) | 129 (0.67x) | 52.5 (0.76x) | 65.5 (0.82x) | 25.9 (0.77x) |
| 8 | 795 (0.63x) | 132 (0.69x) | 53.6 (0.78x) | 66.4 (0.83x) | 26.5 (0.79x) |
| 9 | 796 (0.64x) | 134 (0.70x) | 54.2 (0.79x) | 67.1 (0.84x) | 26.6 (0.79x) |
| 10 | 802 (0.64x) | 140 (0.73x) | 56.1 (0.81x) | 68.9 (0.86x) | 27.3 (0.81x) |
| 11 | 810 (0.65x) | 146 (0.76x) | 58.7 (0.85x) | 71.9 (0.90x) | 28.7 (0.86x) |
| 12 | 810 (0.65x) | 146 (0.76x) | 58.3 (0.85x) | 71.6 (0.90x) | 28.6 (0.85x) |
| 13 | 808 (0.64x) | 146 (0.76x) | 59.2 (0.86x) | 72.2 (0.90x) | 28.5 (0.85x) |
| 14 | 810 (0.65x) | 150 (0.78x) | 60.3 (0.88x) | 72.1 (0.90x) | 28.7 (0.86x) |

Native aarch64 (rav1d was not built natively in this round):

| after step | fixture 320x180 | testsrc2 720p | mandelbrot 720p | mandelbrot 720p 10-bit | mandelbrot 1080p |
|---|---|---|---|---|---|
| before | 1333 | 228 | 122 | 164 | 65.1 |
| 8 | 1357 | 240 | 130 | 171 | 68.1 |
| 9 | 1358 | 246 | 134 | 178 | 70.3 |
| 10 | 1375 | 267 | 147 | 189 | 75.3 |
| 11 | 1388 | 270 | 149 | 192 | 76.7 |
| 12 | 1390 | 270 | 149 | 191 | 76.8 |
| 13 | 1395 | 276 | 154 | 197 | 78.8 |
| 14 | 1401 | 282 | 157 | 200 | 79.9 |

No native row got slower at any step. Step 12 changes nothing but the simd build: its other rows show what the noise of the run is.

The fourteen streams, `before` (`829b1e1`) and now, in one run of their own (median of 7, load average about 4):

| stream | native: before | now | wasm baseline: rav1d | before | now | wasm simd: rav1d | before | now |
|---|---|---|---|---|---|---|---|---|
| fixture 320x180, 8-bit | 1337 | 1403 (1.05x of before) | 1259 | 788 | 813 (0.65x) | 1838 | 1138 | 1188 (0.65x) |
| fixture 160x90, 10-bit | 3383 | 3508 (1.04x of before) | 3037 | 2182 | 2258 (0.74x) | 3114 | 3050 | 3196 (1.03x) |
| fixture 128x72, film grain | 2438 | 2536 (1.04x of before) | 2521 | 1682 | 1739 (0.69x) | 3582 | 2193 | 2283 (0.64x) |
| fixture 177x99, 8-bit | 2740 | 2972 (1.08x of before) | 2026 | 1478 | 1563 (0.77x) | 3533 | 2433 | 2656 (0.75x) |
| fixture 96x64, 4:4:4 | 4120 | 4343 (1.05x of before) | 3980 | 2652 | 2823 (0.71x) | 6106 | 4165 | 4385 (0.72x) |
| fixture 96x64, mono | 9201 | 9366 (1.02x of before) | 12396 | 7969 | 8098 (0.65x) | 16865 | 10272 | 10795 (0.64x) |
| AOM b8-05-mv 352x288 | 248 | 302 (1.22x of before) | 165 | 147 | 168 (1.02x) | 271 | 209 | 254 (0.94x) |
| AOM b10-00-quantizer-10, 10-bit | 102 | 118 (1.15x of before) | 75.0 | 62.7 | 70.0 (0.93x) | 77.8 | 85.3 | 98.7 (1.27x) |
| AOM b8-23-film_grain-50 | 421 | 461 (1.10x of before) | 364 | 230 | 249 (0.68x) | 544 | 360 | 398 (0.73x) |
| AOM b8-00-quantizer-20 | 265 | 322 (1.22x of before) | 208 | 159 | 182 (0.88x) | 298 | 237 | 285 (0.96x) |
| SVT testsrc2 720p | 228 | 283 (1.24x of before) | 192 | 129 | 152 (0.79x) | 330 | 196 | 246 (0.75x) |
| SVT mandelbrot 720p | 122 | 157 (1.29x of before) | 68.9 | 52.4 | 59.6 (0.87x) | 155 | 102 | 130 (0.84x) |
| SVT mandelbrot 720p, 10-bit | 166 | 202 (1.22x of before) | 78.9 | 64.8 | 72.3 (0.92x) | 86.2 | 138 | 172 (2.00x) |
| SVT mandelbrot 1080p | 65.8 | 80.0 (1.22x of before) | 34.0 | 26.0 | 29.1 (0.86x) | 80.6 | 55.6 | 68.8 (0.85x) |

Against rav1d on one thread the simd build is now at 0.64 to 0.96 on 8-bit streams (it was at 0.59 to 0.80 in this run's `before`) and at 1.03 to 2.00 on 10-bit ones; the baseline build at 0.65 to 1.02 (0.63 to 0.89); native decoding is 1.02 to 1.29 times what it was. The small fixtures move least: little of their time is coefficients.

The two parts on mandelbrot 720p, milliseconds a frame, median of three profiles of each build (and of three timed runs natively):

| | symbol decoder and coefficients | inverse transforms and reconstruction | the two | a frame |
|---|---|---|---|---|
| wasm simd, before | 2.81 | 1.59 | 4.40 | 10.25 |
| wasm simd, now | 1.53 | 0.60 | 2.13 | 7.87 |
| wasm baseline, before | 2.74 | 2.92 | 5.66 | 19.80 |
| wasm baseline, now | 1.56 | 1.46 | 3.02 | 17.15 |
| native, before | 2.36 | 0.93 | 3.29 | 9.14 |
| native, now | 1.07 | 0.45 | 1.52 | 7.33 |

(Dequantisation is in the first column now and was in the second before, step 14.) The first round measured rav1d's two parts at 1.6 and 1.1 ms in a simd frame of 6.6: this crate's are now 1.5 and 0.6, and its frame 7.9. What is left of the difference between the two decoders is in the other parts.

**What every step was held to.** `tools/wasm-perf/gate.sh` as the first round ran it (formatting, clippy for the host, for wasm32 with and without `simd128` and for x86-64, the documentation, `cargo test --release`, which with the AOM vectors in place decodes all 244 of them with `CoefficientReader::Checked`, the committed vectors and the runner on 4 threads, the fuzz and encoder tests in debug, x86-64's tests under Rosetta, wasm-av1's tests on the tree, 15 streams through both wasm builds against rav1d's and libdav1d's MD5s, the clips natively on 1 and 4 threads), and from step 12 on the transform's comparison with the scalar reference inside both wasm modules. Every step passed all of it, with one exception that was not the code: step 14's run could not start the `itxbench` script, whose executable bit had just been cleared to match the other scripts; that line was run again by hand and passed, and `gate.sh` now starts it with `bash`. For each step a difference was planted and seen to fail the test that should catch it; they are named in the steps. At the end, on `d3e6ac4`: the 244 AOM vectors natively with the readers checked against each other and the symbol decoder against its reference, on 1 thread and on 4 (244 of 244, twice), and the 242 that are IVF through both wasm builds beside rav1d's two (the same MD5 from all four builds for all 242).

AVX2 hosts were not measured: no machine here has AVX2. Steps 8 to 10 and 14 are the same code on every architecture. Steps 11 and 13 change what the AVX2 kernel is compiled from (the per-size dispatch, the bounded transform) in the same way as for aarch64; x86-64 was built, linted and tested without AVX2 only.

### What is known from elsewhere

The rule of this round was the specification and this crate as the only sources. These are the ideas in it that are common knowledge of how decoders are built, each derived here and held to this crate's reference code as its step describes: an arithmetic decoder's value kept in a wide window refilled in bulk (step 8; the complemented form is the specification's own `SymbolValue`); a symbol found by counting thresholds instead of searching, and a CDF adapted without branches (step 9); tables worked out once for each scan, and contexts from a padded map of clamped levels (step 10; the map is step 6's); the transforms as one function for each size (step 11, got by compiling the specification's process for each size, not by writing them); work bounded by the end-of-block position, in dequantisation and in the transform (steps 13 and 14); a coefficient dequantised as it is read (step 14). What rav1d does was not looked at for any of them; the README's sentence about techniques commonly known from other decoders names three, and should name these when this is merged.

### What is left in these two parts

- **The reads.** `coeffs_decode` is 1.35 ms of the simd frame and nearly all of it is 124 000 adaptive reads whose dependent chain (the thresholds from the CDF, the comparison, the adaptation stored for the next read of the same CDF) is about twenty cycles. Shorter needs the read itself restructured, for instance the thresholds and the adaptation of a CDF as one vector operation; the adaptation alone as a vector was slower (step 8), and the whole read as one was not written.
- **Everything that is not a coefficient** (12 000 symbols a frame) still goes through the general `read_symbol`, a search loop and two adaptation loops: about 0.1 ms on its own and more inside `decode_block`.
- **The transform arithmetic**, 0.5 ms of the simd frame, is 32-bit lanes, four to a vector. Sixteen-bit lanes would halve the vectors for 8-bit streams, if they can be proved exact for every value conformance allows; that proof was not attempted.
- **The baseline build's transforms** are scalar and stay 1.4 ms: it has no lanes. Transforms that skip what is known to be zero pay there only on sparser streams than these (`tried/partial-transforms.diff`).

## All three together (`develop` after the merges, 2026-11-13)

The loop filter, inter prediction and the residual path were each measured
against `829b1e1` on its own branch. Merged, in one run with rav1d and the
decoder as first ported (`3c2c549`) taking turns with it, one thread, fps,
median of 7 (another benchmark was running on the machine; good to a few
percent):

| clip | simd: rav1d | first | now | baseline: rav1d | first | now |
|---|---|---|---|---|---|---|
| mandelbrot 720p | 154.1 | 64.3 | 178.9 (1.16x) | 69.1 | 31.4 | 73.3 (1.06x) |
| mandelbrot 720p, 8 tiles | 153.9 | 64.0 | 186.0 (1.21x) | 66.5 | 31.1 | 71.0 (1.07x) |
| mandelbrot 1080p | 80.1 | 33.4 | 89.7 (1.12x) | 34.1 | 16.0 | 34.1 (1.00x) |
| mandelbrot 720p, 10-bit | 86.4 | 81.0 | 217.1 (2.51x) | 78.6 | 37.4 | 80.7 (1.03x) |
| testsrc2 720p | 323.6 | 114.4 | 283.0 (0.87x) | 188.0 | 63.1 | 177.0 (0.94x) |

All fifteen gate streams are bit-exact in both builds. This is one thread;
the threads builds still scale as before, which is what is left.

## Threads

Everything above is one thread. With `set_threads(n)` the decoder
parallelised the tiles of a frame and the bands of its post-filters, so a
stream with one tile to a frame, which is what video sites serve, was
parsed and reconstructed on one thread whatever `n`, and eight threads
bought it a fifth. This section is the work on that. The design, and how
what may run at the same time was derived from the specification, is in
[threading.md](threading.md).

**How the numbers are taken.** wasm-av1 at `ee33e98` with
`tools/wasm-perf/wasm-av1-ee33e98-frames-in-flight.diff` on this crate
(its adapter queues each unit and takes frames as they come), the
simd-threads build, in Node 26 with every wasm thread a `worker_threads`
Worker. `threads-table.sh`: the builds and thread counts take turns, a
process each (one warm-up decode, three timed ones, their median), and a
cell is the median of seven such rounds; every decode's MD5 is checked
against libdav1d's. rav1d is its prebuilt simd-threads package (`6f780a0`
in that checkout's manifest) through the same harness. Other work was
running on the machine (an Apple M3 Max, 16 cores; load average 2.5 to
4.5), so read a few percent as noise.

There are two clocks, and they do not agree for a decoder that decodes
while its caller is doing something else:

- **The wall clock** of the whole decode loop, frames taken and not
  hashed (`WALL=1`). What a player gets.
- **The time inside the decode calls alone**, which is what every table
  above uses and what this document's earlier thread figures were. With
  frames in flight the decoder's threads go on decoding while the harness
  hashes the frame it was just handed, and that time is not counted: this
  clock then flatters the decoder (rav1d's frame threads hardly gain by
  it in this harness; this crate's frames in flight gain a great deal).
  It is given because the reference figures were taken with it; **compare
  by the wall clock.**

`before` is `c753ea1`, where this section starts. `one at a time` is the
decoder now through `decode_all` (the adapter built with
`RIVET_FRAMES_IN_FLIGHT=1`): a unit's frames come back from the call that
took the unit. `in flight` has a frame in flight for each thread, the
adapter's default with more than one thread.

Wall clock, fps on 1 / 2 / 4 / 8 threads:

| clip | build | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| mandelbrot 720p | rav1d | 153 | 277 | 398 | 505 |
|  | before | 184 (1.20x) | 209 (0.75x) | 224 (0.56x) | 220 (0.44x) |
|  | one at a time | 185 (1.21x) | 313 (1.13x) | 341 (0.86x) | 338 (0.67x) |
|  | in flight | 185 (1.21x) | 326 (1.18x) | 505 (1.27x) | 680 (1.35x) |
| mandelbrot 1080p | rav1d | 80 | 144 | 213 | 278 |
|  | before | 89 (1.11x) | 104 (0.72x) | 113 (0.53x) | 115 (0.41x) |
|  | one at a time | 89 (1.11x) | 157 (1.09x) | 191 (0.90x) | 189 (0.68x) |
|  | in flight | 90 (1.12x) | 162 (1.12x) | 266 (1.25x) | 369 (1.33x) |
| mandelbrot 720p, 8 tiles | rav1d | 154 | 294 | 542 | 958 |
|  | before | 185 (1.20x) | 292 (0.99x) | 390 (0.72x) | 373 (0.39x) |
|  | one at a time | 186 (1.21x) | 292 (0.99x) | 391 (0.72x) | 371 (0.39x) |
|  | in flight | 185 (1.20x) | 288 (0.98x) | 386 (0.71x) | 372 (0.39x) |

The same runs by the time inside the decode calls:

| clip | build | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| mandelbrot 720p | rav1d | 152 | 276 | 395 | 503 |
|  | before | 182 (1.20x) | 208 (0.75x) | 223 (0.56x) | 222 (0.44x) |
|  | one at a time | 184 (1.21x) | 312 (1.13x) | 341 (0.86x) | 338 (0.67x) |
|  | in flight | 184 (1.21x) | 411 (1.49x) | 832 (2.11x) | 1016 (2.02x) |
| mandelbrot 1080p | rav1d | 80 | 143 | 210 | 277 |
|  | before | 89 (1.11x) | 104 (0.73x) | 112 (0.53x) | 115 (0.42x) |
|  | one at a time | 89 (1.11x) | 156 (1.09x) | 191 (0.91x) | 190 (0.69x) |
|  | in flight | 89 (1.11x) | 205 (1.43x) | 430 (2.05x) | 501 (1.81x) |
| mandelbrot 720p, 8 tiles | rav1d | 153 | 290 | 538 | 952 |
|  | before | 184 (1.20x) | 289 (1.00x) | 391 (0.73x) | 376 (0.39x) |
|  | one at a time | 184 (1.20x) | 290 (1.00x) | 391 (0.73x) | 379 (0.40x) |
|  | in flight | 184 (1.20x) | 288 (0.99x) | 386 (0.72x) | 376 (0.39x) |

Natively (aarch64): wasm-av1's `decode_ivf` by its own clock, which is
the time inside the decode calls (`native-threads-table.sh`, median of 7,
the builds taking turns); rav1d's native build there has no assembly.

| clip | build | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| mandelbrot 720p | rav1d | 88 | 163 | 249 | 328 |
|  | before | 224 (2.55x) | 252 (1.55x) | 267 (1.07x) | 259 (0.79x) |
|  | one at a time | 222 (2.52x) | 377 (2.31x) | 420 (1.69x) | 412 (1.26x) |
|  | in flight | 221 (2.51x) | 562 (3.45x) | 904 (3.63x) | 1542 (4.70x) |
| mandelbrot 1080p | rav1d | 43 | 80 | 128 | 180 |
|  | before | 107 (2.49x) | 124 (1.55x) | 134 (1.05x) | 136 (0.76x) |
|  | one at a time | 106 (2.47x) | 186 (2.33x) | 233 (1.82x) | 231 (1.28x) |
|  | in flight | 106 (2.47x) | 267 (3.34x) | 421 (3.29x) | 614 (3.41x) |
| mandelbrot 720p, 8 tiles | rav1d | 85 | 163 | 309 | 557 |
|  | before | 224 (2.64x) | 332 (2.04x) | 425 (1.38x) | 372 (0.67x) |
|  | one at a time | 223 (2.62x) | 338 (2.07x) | 436 (1.41x) | 404 (0.73x) |
|  | in flight | 222 (2.61x) | 332 (2.04x) | 437 (1.41x) | 394 (0.71x) |

And this crate's own `examples/av1_decbench.rs`, the wall clock of the
decode with nothing else in the loop, best of 5 (not interleaved), one
frame at a time and with as many frames in flight as threads:

| clip | | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| mandelbrot 720p | one at a time | 213 | 371 | 429 | 418 |
| | in flight | 213 | 384 | 617 | 870 |
| mandelbrot 1080p | one at a time | 105 | 185 | 237 | 235 |
| | in flight | 105 | 192 | 326 | 462 |
| mandelbrot 720p, 8 tiles | one at a time | 222 | 342 | 448 | 408 |
| | in flight | 222 | 340 | 446 | 403 |

**Against the target** (at 4 and at 8 threads at least 0.85 of rav1d on
the one-tile clips, one thread unchanged, the eight-tile clip no slower):
by the wall clock, with frames in flight, mandelbrot 720p is at 1.27 of rav1d on 4 threads and 1.35 on 8, and 1080p at 1.25 and 1.33: ahead of it. One frame at a time they are at 0.86 and 0.67, and 0.90 and 0.68: the target takes the frames in flight, and with them a unit's frames come out a few units after the unit went in. One thread is where it was (185 against 184 fps and 90 against 89 in this build; the builds without threads are below). The eight-tile clip is where it was, to within the noise (185 / 292 / 390 / 373 before, 186 / 292 / 391 / 371 one at a time, 185 / 288 / 386 / 372 with frames in flight, where each of its units was tried in flight and then decoded in place; a stream of such frames is no longer tried unit after unit, since `6be8d84`: 185 / 293 / 392 / 378): nothing here makes its frames jobs, and rav1d is 2.6 times as fast on it on 8 threads.

### The steps

1. **Rows behind the parsing** (`53fa752`). The tile walker can parse a
   superblock row into a record and reconstruct a row from one, as well
   as do both; the calling thread parses, the others reconstruct the
   rows in a wavefront two superblocks behind the row above, each taking
   that row's last sample line as a copy. `Shared`, the window type the
   threads hold their rows through. In the wasm simd-threads build by the
   time inside the decode calls (a frame at a time, the two clocks agree
   to a percent or two; these and the next step's are from the runs of
   their day, one timed decode a process, which reads some 8 % lower
   than the tables above), mandelbrot 720p on 1 / 2 / 4 / 8 threads: 168
   / 190 / 203 / 204 to 167 / 254 / 254 / 249; 1080p 82 / 94 / 102 / 104
   to 81 / 131 / 134 / 137.
2. **The post-filters behind the reconstruction** (`0618dee`). Deblock,
   CDEF and loop restoration are tasks of a row (a stripe), ready when
   the rows they read are final; a frame without CDEF or loop restoration
   no longer copies the frame for them. 720p: 167 / 253 / 255 / 250 to
   166 / 277 / 287 / 270; 1080p 81 / 130 / 135 / 137 to 81 / 138 / 158 /
   147. A frame has work for two or three threads and no more: its
   parsing is one thread's, about 1.8 ms of a 4.8 ms frame natively.
3. **Frames in flight** (`906fbe4`). One engine for the tasks of every
   frame in flight; a frame's parsing waits for the frames whose saved
   state it reads and its rows for its references' samples;
   `set_frames_in_flight`, `queue`, `poll`, `wait`, with every unit's
   result that of decoding it in place. With it: the scheduler wakes one
   waiting thread for each task that becomes ready, the one that waited
   last first, where every thread used to be woken at every change
   (which is why eight threads were slower than four, and are not now);
   the decoder keeps its threads between frames (worth nothing measurable
   by itself on this machine, where starting a thread takes some ten
   microseconds, but it is what a session of frames in flight runs on);
   and the windows read a prefix of their array, below. The tables above
   are its before and after.

### One thread

The same decoder on one thread, in the builds without threads (`simd`,
`baseline`), by `bench-turns.mjs`, median of 9:

| clip | simd: rav1d | before | now | baseline: rav1d | before | now |
|---|---|---|---|---|---|---|
| mandelbrot 720p | 153.2 | 183.9 | 183.4 (-0.3 %) | 68.6 | 74.6 | 74.5 (-0.1 %) |
| mandelbrot 720p, 8 tiles | 152.7 | 183.7 | 182.3 (-0.8 %) | 65.4 | 70.6 | 70.5 (-0.1 %) |
| mandelbrot 1080p | 79.8 | 89.1 | 89.1 (+0.0 %) | 34.0 | 34.2 | 34.3 (+0.3 %) |
| mandelbrot 720p, 10-bit | 86.8 | 219.9 | 219.5 (-0.2 %) | 79.9 | 82.1 | 81.8 (-0.4 %) |
| testsrc2 720p | 323.3 | 281.5 | 278.3 (-1.1 %) | 187.0 | 175.2 | 174.7 (-0.3 %) |

The first version of the window type let a window read any range of its
array, and checking an index against a range is a subtraction more than
checking it against a length. That cost about 1 % of a one-thread decode
in the simd build (183.1 against 184.1 fps on mandelbrot 720p, 87.5
against 88.3 on 1080p, 278.6 against 281.4 on testsrc2; a build with the
subtraction taken out of the reads, which is not sound for windows, gave
184.2, 89.0 and 283.3). So a window now reads from the start of its array
to a limit, which is all a reader in the pipeline ever needs, and reads
the rows it writes by name (`Shared::mine`). A variant with the second
check behind a call that returns (179.8 to 174.9 fps on mandelbrot 720p)
is not on the branch: a slow path that comes back costs the compiler what
a panic does not.

### What every step was held to

`tools/wasm-perf/gate.sh` for each step's commit: formatting, clippy for
the host, wasm32 with and without `simd128` and x86-64, the
documentation, `cargo test --release` (the 244 AOM vectors with both
coefficient readers checked against each other among them), the vectors
again on 4 and on 8 threads and, from step 3, with 4 and 8 frames in
flight; the threaded tests (`tests/threads.rs`) and the fuzz tests in
debug, the latter on 4 threads and with frames in flight too; x86-64's
tests under Rosetta; wasm-av1's tests on the tree; 15 streams through
both wasm builds against rav1d's and libdav1d's MD5s; the clips natively
and in the simd-threads build on 1, 2, 4 and 8 threads.

`tests/threads.rs` holds every thread count (2, 3, 4, 8, 16) and every
combination of threads and frames in flight it tries to one thread, unit
by unit: the same frames, the same errors, for the committed vectors and
for those vectors with bits flipped and units cut (24 kinds of damage a
vector, which give an error in more than a quarter of the cases); every
such decode on a thread of its own with two minutes to finish, so that a
hang fails. Looped: `tests/threads.rs` and the registered-runner test 25 times
over with twice the repetitions (`AV1_THREAD_SOAK=2`), while a gate was
running on the same machine: the threaded tests passed every time. The
runner test failed once in the 25: a race in the test itself (it asked
whether the runner's threads were back from a started job before they
could be), which was fixed, and it then passed 60 times of 60. The unit
tests of the pool and of the window type 200 times of 200, after two of
the pool's tests were corrected the same way (they assumed every index
of a job is taken by a different thread, which the pool does not
promise).

At the end, on `906fbe4`: the 244 AOM vectors natively, both coefficient readers
checked against each other, on 1, 2, 3, 4, 8 and 16 threads and with 2,
4 and 8 frames in flight (244 of 244 each time); the 242 of them that are
IVF through the wasm simd-threads build, with frames in flight on 1, 4
and 8 threads and one frame at a time on 4 and 8, against rav1d's build
(`vectors-threads.sh`: the same MD5 for all 242, five times); and
wasm-av1 `main` with its adapter changed as `threading.md` says ("What a
caller changes"; the patch is for that repository, not in this one): its
Rust tests, its Node tests of the four variants on 1, 2, 4 and 8
threads, and its browser tests in Chromium, Firefox and WebKit, on 2, 4
and 8 threads among them.

### Ideas known from elsewhere

That decoders parse apart from reconstructing, run superblock rows in a
wavefront, let the post-filters follow by rows, and decode several frames
at once with the output delayed, is common knowledge of how video
decoders are threaded. None of it was taken from another decoder's
source or design notes: what waits for what was read off the
specification (threading.md, "What depends on what"), and what the
implementation adds of its own is how it keeps that safe (the windows,
which turn a mistake in the schedule into a panic) and how it keeps a
unit's result exact (anything that does not fit, and every error, is
decoded again in place).

### What is left

- **A frame waits for whole reference frames.** Its rows could start when
  the rows of a reference their motion vectors reach are final; the
  frames at the top of a pyramid, which everything refers to, would then
  overlap with the ones below them. The windows already say "up to this
  row".
- **Frames of several tiles are not jobs.** They are decoded in place
  (tiles in parallel into frame states that are copied back, the filters
  in bands), with or without frames in flight: the eight-tile clip is
  where it was, a third of rav1d's speed on eight threads.
- **Memory is allocated per frame.** Every frame's buffers, and a tile
  walker for every row's reconstruction; a frame's setup is 2 % of a
  one-thread decode.
- **One frame at a time is bound by the parsing**, one thread's: two or
  three threads' worth. That is what frames in flight are for, and they
  cost a few units of delay.
- **One thread** is within about 1 % of where it was, and not ahead: a
  window's writes still check a range, and a block still goes through
  the question "which walk is this".
