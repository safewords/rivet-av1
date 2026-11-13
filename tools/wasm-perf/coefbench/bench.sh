#!/bin/bash
# bench.sh [rounds] — variants of the decode-only coefficient reader's level and sign passes (src/v*.rs) replaying the
# transform blocks `record.sh` recorded ($WASM_PERF/trace-blocks.bin), re-encoded with this crate's encoder and the
# reference context functions: ns a coefficient, natively and in wasm baseline and simd. Every variant is first checked
# to decode the recorded coefficients.
S="${WASM_PERF:?set WASM_PERF to the working directory described in ../README.md}"
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
R=${1:-9}; TR="${TRACE_FILE:-$S/trace-blocks.bin}"; T="$S/coefbench-target"; NV=$(grep -c '=> for b in' src/lib.rs)
cargo build --release --target-dir "$T/native" 2>&1 | grep -E "^error" -A12
echo "== native"; "$T/native/release/coefbench" "$TR" $R
for v in baseline simd; do
  f=""; [ $v = simd ] && f="-C target-feature=+simd128"
  touch src/lib.rs; BLOCKS="$TR" RUSTFLAGS="$f" cargo build --release --lib --target wasm32-unknown-unknown --target-dir "$T/wasm-$v" 2>&1 | grep -E "^error" -A12
  echo "== wasm $v"; node run.mjs "$T/wasm-$v/wasm32-unknown-unknown/release/coefbench.wasm" $R $NV
done
