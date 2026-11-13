#!/bin/bash
# bench.sh [tree] [label] [check] — the 2D inverse transform of a rivet-av1 tree (default: $WASM_PERF/rivet-av1), natively
# and in wasm baseline and simd: first the comparison with the scalar reference for every size and type (the crate's unit
# test, run where the target is wasm32 too), then, unless `check` is given, ns a block for a block full of coefficients and
# for one with coefficients in its top-left sixteenth, each through inverse_transform_2d and inverse_transform_2d_within.
S="${WASM_PERF:?set WASM_PERF to the working directory described in ../README.md}"
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0 TREE="${1:-$S/rivet-av1}"
L=${2:-cur}; M=$3; T="$S/itxbench-target"
echo "            full  full/within  sixteenth  sixteenth/within   (ns a block)"
cargo build --release --target-dir "$T/$L-native" 2>&1 | grep -E "^error" -A12
echo "== native"; "$T/$L-native/release/itxbench" $M || exit 1
for v in baseline simd; do
  f=""; [ $v = simd ] && f="-C target-feature=+simd128"
  RUSTFLAGS="$f" cargo build --release --lib --target wasm32-unknown-unknown --target-dir "$T/$L-wasm-$v" 2>&1 | grep -E "^error" -A12
  echo "== wasm $v"; node run.mjs "$T/$L-wasm-$v/wasm32-unknown-unknown/release/itxbench.wasm" $M || exit 1
done
