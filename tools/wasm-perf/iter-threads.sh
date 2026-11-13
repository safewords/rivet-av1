#!/bin/bash
# iter-threads.sh [label] — wasm-av1 (ee33e98) on the rivet-av1 working tree as the simd-threads variant (atomics, shared
# memory, std rebuilt), into iter/<label>/pkg/simd-threads, with scripts/build.sh's flags. For bench-threads.mjs.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
label="${1:-cur}"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
cd "$S/wasm-av1" || exit 1
LINK="-C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
for v in ${VARIANTS:-simd-threads}; do
  feat="+atomics,+bulk-memory,+mutable-globals"; [ "$v" = simd-threads ] && feat="+simd128,$feat"
  RUSTC_BOOTSTRAP=1 RUSTFLAGS="-C target-feature=$feat $LINK" cargo build --config "paths=['$S/rivet-av1']" --lib --release \
    --target wasm32-unknown-unknown -Zbuild-std=std,panic_abort --no-default-features --features rivet-av1-pending,container \
    --target-dir "target/wasm-$v" 2>&1 | grep -E "^(error|warning: unused)|Finished" -A6 | grep -v homepage | head -30
  rm -rf "$S/iter/$label/pkg/$v"; mkdir -p "$S/iter/$label/pkg/$v"
  wasm-bindgen "target/wasm-$v/wasm32-unknown-unknown/release/wasm_av1.wasm" --target web --out-dir "$S/iter/$label/pkg/$v" --out-name wasm_av1 || exit 1
  echo "$v $(wc -c < "$S/iter/$label/pkg/$v/wasm_av1_bg.wasm") bytes"
done
