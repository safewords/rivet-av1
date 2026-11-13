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
- `bench-turns.mjs`: `bench-node.mjs` with the builds taking turns, a run
  of each in every round, so that other work on the machine falls on all of
  them alike.
- `bench-threads.mjs`, `thread-shim.mjs`: a threads build in Node, each
  wasm thread a `worker_threads` Worker.
- `bench-native.py <runs> <threads> <decode_ivf>...`: the same natively,
  with wasm-av1's `decode_ivf` example.
- `prof.sh <label> <variant> <clip>`: V8's CPU profile by function.
- `lfprof.sh <runs> <clip> <label/variant>...`: the loop filter's
  milliseconds a frame in wasm builds (its share of the profile times the
  frame time of an unprofiled run), the builds taking turns.
- `wasi-test.sh [filter...]`, `wasi-run.mjs`: this crate's unit tests as a
  `wasm32-wasip1` test binary in Node, without and with `simd128`: the
  kernels a wasm build takes, held to their references in V8 (the `dsp::`
  and `postfilter` tests by default; nothing that starts a thread runs
  there).
- `lf-timing.diff`: a clock around the loop filter, for the same figure
  natively. Apply it to the tree, build `decode_ivf`, revert it, and run
  with `AV1_LF_TIME=1`: the last `LFTIME` line is the total.
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
