# The harness behind docs/wasm-performance.md

What measured and gated every step of the wasm speed work, kept here so
that it does not live in one session's scratch directory. Not part of the
crate (`tools/` is excluded from the package) and not run by CI.

`WASM_PERF` names a working directory laid out like this:

```
$WASM_PERF/rivet-av1    this repository, the tree under test
$WASM_PERF/wasm-av1     https://github.com/safewords/wasm-av1 at ee33e98 (both decoder cores)
$WASM_PERF/builds/rav1d/pkg   the reference: that checkout's default build (scripts/build.sh)
$WASM_PERF/clips/*.ivf  + .ivf.md5   benchmark clips and libdav1d's MD5 of their planes
$WASM_PERF/vectors/*.ivf             a few of tests/data, for the same table
$WASM_PERF/iter/<label>/pkg          what iter.sh builds
$WASM_PERF/logs
```

The clips were made with ffmpeg (`testsrc2` and `mandelbrot` with
`noise=alls=8:allf=t`, libsvtav1 preset 8, CRF 35, `-g 30`; 1280x720 8- and
10-bit, 1920x1080, and one with `tile-columns=2:tile-rows=1`), their MD5s
with `ffmpeg -c:v libdav1d ... -f rawvideo - | md5`.

- `iter.sh <label>`: wasm-av1 on the working tree (`--config paths=`, the
  `rivet-av1-pending,container` features), baseline and simd, function
  names kept.
- `bench-node.mjs <runs> <label=pkgdir,...> <variants> <ivf...>`: fps of
  the decode calls, median of the runs, and whether every build's MD5
  agrees (and equals the `.md5` beside the clip).
- `bench-threads.mjs`, `thread-shim.mjs`: a threads build in Node, each
  wasm thread a `worker_threads` Worker.
- `bench-native.py <runs> <threads> <decode_ivf>...`: the same natively,
  with wasm-av1's `decode_ivf` example.
- `prof.sh <label> <variant> <clip>`: V8's CPU profile by function.
- `gate.sh <label>`: this crate's gates (fmt, clippy on the host, wasm32
  with and without simd128, and x86-64 with a locally built std under
  Rosetta; doc; the tests), wasm-av1's tests on the tree, bit-exactness of
  15 streams through the wasm builds against rav1d, and the clips natively
  on 1 and 4 threads.

For the residual path (steps 8 on):

- `parts.sh <label> <variant> <clip>`: V8's profile at 200 us, as
  milliseconds a frame of the symbol decoder with the coefficient reader
  and of the inverse transforms with reconstruction.
- `parts-native.sh <label> [tree]`: the same two natively, as timers
  around `coeffs()` and `reconstruct()` in a patched copy of the tree.
- `table.sh <runs> <before> <after> [all]`: rav1d, and two `iter.sh`
  builds, interleaved clip by clip: wasm baseline and simd, then native
  (`native-build.sh <label> <tree>` builds the native side of a tree that
  `gate.sh` did not).
- `plant.sh <file> <old> <new> --test committed_vectors`: the committed
  vectors, which decode with `CoefficientReader::Checked`, on a copy of
  the tree with one planted difference. They are expected to fail.

- `steps-table.sh <runs> <label>...`: rav1d and any number of `iter.sh`
  builds in one interleaved run, wasm and native: the table of all the
  steps at the end of the document.
- `record.sh <clip> [prefix] [frames]`: decodes a clip in a patched copy of
  the tree and records every symbol the reference symbol decoder reads and
  every transform block the reference coefficient reader reads, to
  `$WASM_PERF/trace-symbols.bin` and `trace-blocks.bin`: what the two
  microbenchmarks below replay.
- `symbench/bench.sh [rounds]`: variants of the symbol decoder, each
  derived from `src/symbol.rs`, decoding the recorded symbols (re-encoded
  with this crate's encoder): ns a symbol natively and in both wasm
  builds. Steps 8 and 9 were chosen on it; the variants that lost are
  still in it.
- `coefbench/bench.sh [rounds]`: variants of the decode-only coefficient
  reader's level and sign passes replaying the recorded blocks: ns a
  coefficient. Steps 9 and 10 were chosen on it.
- `itxbench/bench.sh [tree] [label] [check]`: the 2D inverse transform
  alone, natively and in both wasm builds: the comparison with the scalar
  reference for every size and type (the unit test of `dsp/itx.rs`, run
  where the target is wasm32, which `cargo test` does not reach), then ns
  a block. `gate.sh` runs the comparison.

`tried/` holds changes that were written, verified bit-exact and measured
to gain nothing, so that nobody measures them again without a reason:
`direct-to-frame.diff` filters a single-reference prediction straight into
the frame (within 1 % on every stream; they predict mostly from two
references).
`partial-transforms.diff` gives the inverse DCTs versions for inputs that
are zero from the 8th, 16th or 32nd on, by running the network on a copy
whose zeros the compiler sees and expanding its loops (a 32-point DCT of 8
inputs in a third of the time; 3 % slower on the benchmark streams, whose
blocks are mostly full, and five times the compile time).
`narrow-lanes.diff` is a 32-bit one-value-at-a-time transform path for
builds without vectors (no faster than the 64-bit one in wasm baseline).
Both are against `src/dsp/itx.rs` as step 13 left it.
