#!/bin/bash
# native-build.sh label tree — wasm-av1's decode_ivf example on the given rivet-av1 tree, into iter/<label>/decode_ivf.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
cd "$S/wasm-av1" || exit 1
cargo build --release --config "paths=['$2']" --no-default-features --features rivet-av1-pending,container --example decode_ivf 2>&1 | grep -E "^error|Finished" -A5
mkdir -p "$S/iter/$1" && cp target/release/examples/decode_ivf "$S/iter/$1/decode_ivf"
