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

`tools/wasm-perf/` has the scripts every number here came from.
