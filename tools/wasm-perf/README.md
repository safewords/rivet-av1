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
- `bench-turns.mjs`: `bench-node.mjs` with the builds taking turns (every
  round decodes the clip once with each build), so that a change in the
  machine's load during a measurement falls on all of them. The tables of
  the inter prediction steps (8 on) are from this.
- `prof-inter.sh <label> <variant> <clip>`: V8's profile again, as the
  time under `predict_inter` (it and everything it calls) in ms per
  decoded frame; `cmp-inter.sh <variant> <clip> <rounds> <label>...` runs
  it for several builds in turn and gives the medians.
- `native-inter.diff`: the measuring patch behind the native figure for
  the same thing, a timer around `predict_inter` that reports at exit, less
  what the timer costs (apply, build `decode_ivf`, read `net_ms` on
  stderr, and take the patch off again; it is not for the branch).
- `wasi-test.sh [filter]`, `wasi-run.mjs`: the crate's unit tests built for
  `wasm32-wasip1` with `simd128` and run in Node's WASI, which is how the
  SIMD128 kernels themselves are held to the scalar reference (`cargo test`
  cannot run `wasm32-unknown-unknown`). With a filter: `dsp::`,
  `decoder::predict` (a test that starts a thread ends the run there).
- `gate.sh <label>`: this crate's gates (fmt, clippy on the host, wasm32
  with and without simd128, and x86-64 with a locally built std under
  Rosetta; doc; the tests), wasm-av1's tests on the tree, bit-exactness of
  15 streams through the wasm builds against rav1d, and the clips natively
  on 1 and 4 threads.

`tried/` holds changes that were written, verified bit-exact and measured
to gain nothing, so that nobody measures them again without a reason:
`direct-to-frame.diff` filters a single-reference prediction straight into
the frame (within 1 % on every stream; they predict mostly from two
references).
`copy-samples.diff` (against step 12) copies a whole-sample single
prediction with SIMD128 loads and stores instead of the `memory.copy` that
`copy_from_slice` compiles to (inter prediction 0.93 and 0.95 ms a frame
with and without: that time is the reference's and the frame's memory, not
the call).
