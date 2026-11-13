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
call (`tools/wasm-perf/lf-timing.diff`, not part of the crate), median of 5
runs with the builds taking turns.

| step | loop filter, ms a frame: wasm simd | wasm baseline | native |
|---|---|---|---|
| before (`829b1e1`) | 1.53 | 2.37 | 1.25 |
| 8. edge decisions | 1.45 | 2.22 | 1.09 |
| 9. in the plane, eight 16-bit lanes (SIMD128, NEON) | 0.34 | 2.22 | 0.29 |

The state after each step, fps and the ratio to rav1d's (median of 7, load
average about 10):

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

### 8. The loop filter's edge decisions: the transform size first, levels from a table

The edge loop of 7.14.2 ran as written for every 4x4 unit of every plane in both passes: the unit's mode info loaded, its block size looked up, and only then the question whether it is on a transform edge, which seven units in eight on this clip are not; for the ones that are, the level derived from the header for each unit (7.14.4: the segment's feature, the reference's and the mode's deltas, three clamps). `edge_unit` makes the same decision in the order that costs least: the transform size, one byte, first; the mode info only on a transform edge; the block size only for a skipped inter block, the one case it matters to. The level is a lookup: for a block with no `delta_lf` it depends on the segment, the first reference frame and the kind of mode alone, so the frame's 576 levels and the limits of each of the 64 level values are worked out once (`LfLevels`), and a block with a `delta_lf` goes through the same function the table was filled with. Units are handed on two at a time along their edge, which is what the kernels of the next steps take; the sample filtering itself is unchanged here. The specification's walk and its strength process stay in the tests as the reference: random frames of every layout, depth and odd size (tilings of blocks and transforms, and mode info that belongs to no tiling) are filtered both ways on one thread and on three and compared sample for sample, and ten planted errors (a wrong neighbour, a missing mode delta, an off-by-one at the frame's edge, ...) each fail them. Worth 5 to 13 % of the loop filter, which is 0 to 4 % of a frame and inside the noise on some rows.

### 9. The loop filter in the plane, eight 16-bit lanes at a time (SIMD128, NEON)

Each unit's edge was gathered into a 14 x 4 array of 32-bit values, a bounds-checked load a sample; every filter its size allows was computed for all four lanes and one result selected per lane; the array was scattered back. `filter_pair` filters two units' worth of an edge, eight samples, where they lie. Across a horizontal edge each row is one load of eight 16-bit lanes and each changed row one store. Across a vertical edge eight rows of eight samples (two such blocks for the 16-sample size) are transposed into vectors of one column each, filtered, and transposed back, as rows of four when only the narrow filter's four columns changed. There is a kernel for each filter length (4, chroma's 6, 8, 16) and direction. The mask of all eight lanes comes first, and the kernel returns if none passes; the narrow filter, the 7-tap (chroma's 5-tap) one and the 13-tap one are each computed only if some lane takes them; and a wide filter's sums run from one output to the next, each the one before plus two samples and less two, instead of being added up afresh. The two units keep their own limits, one value in each half of a vector; a unit that is not filtered, or whose size is another kernel's, gets a `blimit` of -1, which no lane passes. 16 bits hold all of it: with samples of at most 12 bits the mask's sum and the narrow filter's `3 * (q0 - p0)` plus a clamped term stay inside an `i16`, and the 13-tap sum (16 samples and the rounding) inside a `u16`.

The kernels are written once, as a macro over two dozen lane operations, and expanded for SIMD128 and for NEON: the same arithmetic in both by construction, so native aarch64 gains with wasm instead of falling behind it. `in_plane_kernels_match_the_process` holds them to the one-sample process of 7.14.6 on 100,000 random edges (every length, direction and depth, the two halves with limits of their own, either half off, the whole plane compared so that a stray write shows), and `frames_match_the_reference_walk` runs them under the walk of step 8. Both run natively, which is the NEON expansion, and as a `wasm32-wasip1` test binary in Node (`tools/wasm-perf/wasi-test.sh`), which is the SIMD128 one in V8. Fourteen planted errors (a rounding constant, a tap, a threshold, a transposition, a store one sample off, the halves' limits swapped, ...) each fail them, and so did a real one on the way: the right half of the 16-sample block loaded four columns off. The loop filter is 4.2 times faster in the simd build and 3.8 times natively; the frame 3 to 30 % (the fixture, a libaom stream filtered on every frame, gains most). x86-64 and the wasm baseline build keep the lane filter, under the new walk.

