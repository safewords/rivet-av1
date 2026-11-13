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
