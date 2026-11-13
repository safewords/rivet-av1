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
