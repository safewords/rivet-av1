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

`tools/wasm-perf/` has the scripts every number here came from.

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
(`parts.sh` and `parts-native.sh` in `tools/wasm-perf`).

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
