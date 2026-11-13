#!/bin/bash
# wasi-test.sh [test filter] — the crate's unit tests as a wasm32-wasip1 module built with simd128 and run in
# Node's WASI: the SIMD128 kernels themselves against the scalar reference, which `cargo test` cannot do for
# wasm32-unknown-unknown. Needs `rustup target add wasm32-wasip1`. A failing test ends the run (no unwinding),
# and so does a test that starts a thread: give a filter (`dsp::`, `decoder::predict`).
here="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$here/../.." || exit 1
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings $here/wasi-run.mjs" \
  cargo test --release --lib --target wasm32-wasip1 --target-dir target/wasi-simd -- "$@"
