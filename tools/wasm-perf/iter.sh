#!/bin/bash
# iter.sh [label] — build the rivet-av1 core (current rivet-av1 working tree) into wasm baseline + simd
# with function names kept, into iter/<label>/pkg. Prints sizes.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
label="${1:-cur}"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3
cd "$S/wasm-av1" || exit 1
for v in ${VARIANTS:-baseline simd}; do
  flags=""; [ "$v" = simd ] && flags="-C target-feature=+simd128"
  RUSTFLAGS="$flags" cargo build --config "paths=['$S/rivet-av1']" --lib --release --target wasm32-unknown-unknown \
    --no-default-features --features rivet-av1-pending,container --target-dir "target/wasm-$v" 2>&1 | grep -E "^(error|warning: unused)|Finished" -A6 | grep -v homepage | head -30
  rm -rf "$S/iter/$label/pkg/$v"; mkdir -p "$S/iter/$label/pkg/$v"
  wasm-bindgen "target/wasm-$v/wasm32-unknown-unknown/release/wasm_av1.wasm" --target web --out-dir "$S/iter/$label/pkg/$v" --out-name wasm_av1 || exit 1
  echo "$v $(wc -c < "$S/iter/$label/pkg/$v/wasm_av1_bg.wasm") bytes"
done
