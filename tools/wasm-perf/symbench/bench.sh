#!/bin/bash
# bench.sh [rounds] — variants of the symbol decoder (src/v*.rs, each derived from this crate's src/symbol.rs) decoding
# the symbols `record.sh` recorded ($WASM_PERF/trace-symbols.bin), re-encoded with this crate's encoder: ns a symbol,
# natively and in wasm baseline and simd. FILTER=4 keeps only the 4-symbol reads, FILTER=1 only the equiprobable bits.
S="${WASM_PERF:?set WASM_PERF to the working directory described in ../README.md}"
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
R=${1:-9}; TR="${TRACE_FILE:-$S/trace-symbols.bin}"; T="$S/symbench-target"; NV=$(grep -c '=> run::<' src/lib.rs)
touch src/lib.rs; cargo build --release --target-dir "$T/native" 2>&1 | grep -E "^error" -A12
echo "== native"; "$T/native/release/symbench" "$TR" $R
for v in baseline simd; do
  f=""; [ $v = simd ] && f="-C target-feature=+simd128"
  touch src/lib.rs; TRACE="$TR" RUSTFLAGS="$f" cargo build --release --lib --target wasm32-unknown-unknown --target-dir "$T/wasm-$v" 2>&1 | grep -E "^error" -A12
  echo "== wasm $v"; node run.mjs "$T/wasm-$v/wasm32-unknown-unknown/release/symbench.wasm" $R $NV
done
