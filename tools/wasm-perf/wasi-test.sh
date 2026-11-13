#!/bin/bash
# wasi-test.sh [filter...]: this crate's unit tests as a wasm32-wasip1 binary run in Node (V8), built without and with
# simd128: the kernels a wasm build takes, held to their references where they run. Give test name filters (default:
# dsp:: and postfilter); the tests that start threads cannot run there.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
cd "$S/rivet-av1" || exit 1
[ $# = 0 ] && set -- dsp:: postfilter
rc=0
for v in baseline simd; do
  flags=""; [ "$v" = simd ] && flags="-C target-feature=+simd128"
  out=$(RUSTFLAGS="$flags" cargo test --release --target wasm32-wasip1 --lib --no-run --target-dir "target/wasi-$v" 2>&1)
  f=$(echo "$out" | grep -o 'target/[^)]*\.wasm')
  [ -n "$f" ] || { echo "$out" | tail -20; echo "FAIL $v: no test binary"; rc=1; continue; }
  node --no-warnings "$(dirname "$0")/wasi-run.mjs" "$f" "$@" 2>&1 | grep -E "^test |^test result|panicked|RuntimeError" | sed "s/^/$v: /"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "FAIL $v"; rc=1; }
done
exit $rc
