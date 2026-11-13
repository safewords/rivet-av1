#!/bin/bash
# gate.sh label — rivet-av1's own gates on the working tree, then bit-exactness through wasm-av1
# (native fixtures 1/2/4 threads, wasm baseline + SIMD on fixtures, AOM vectors and SVT clips), then
# a 7-run benchmark of the key clips. Summary on stdout, everything in logs/gate-<label>.log.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
W=$S/wasm-av1
label="$1"; L="$S/logs/gate-$label.log"; : > "$L"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
fail=0
run() { echo "\$ $*" >> "$L"; "$@" >> "$L" 2>&1; rc=$?; echo "exit=$rc" >> "$L"; [ $rc = 0 ] && echo "ok   $*" || { echo "FAIL $*"; fail=1; }; }
cd "$S/rivet-av1"
run cargo fmt --all --check
run cargo clippy --all-targets --all-features -- -D warnings
run cargo doc --no-deps
run cargo test --release
run env AV1_DECODE_THREADS=4 cargo test --release --test committed_vectors --test parallel
run cargo test --test fuzz --test encode
run cargo clippy --target wasm32-unknown-unknown --lib -- -D warnings
run env RUSTFLAGS="-C target-feature=+simd128" cargo clippy --target wasm32-unknown-unknown --lib --target-dir target/wasm-simd -- -D warnings
run env RUSTC_BOOTSTRAP=1 cargo clippy -Zbuild-std=std,panic_abort --target x86_64-apple-darwin --all-targets --all-features --target-dir target/x86 -- -D warnings
run env RUSTC_BOOTSTRAP=1 cargo test -Zbuild-std=std,panic_abort --target x86_64-apple-darwin --release --target-dir target/x86
run "$S/rivet-av1/tools/wasm-perf/itxbench/bench.sh" "$S/rivet-av1" gate check
cd "$W"
run cargo test --release --config "paths=['$S/rivet-av1']" --no-default-features --features rivet-av1-pending,container
cargo build --release --config "paths=['$S/rivet-av1']" --no-default-features --features rivet-av1-pending,container --example decode_ivf >> "$L" 2>&1
mkdir -p "$S/iter/$label" && cp target/release/examples/decode_ivf "$S/iter/$label/decode_ivf"
"$S/iter.sh" "$label" >> "$L" 2>&1
cd "$S"
echo "\$ wasm bit-exactness (rav1d vs this build, baseline + simd, 1 run)" >> "$L"
node bench-node.mjs 1 rav1d=$S/builds/rav1d/pkg,rivet=$S/iter/$label/pkg baseline,simd $W/testdata/*.ivf $S/vectors/*.ivf $S/clips/*.ivf > "$S/logs/gate-$label-md5.txt" 2>&1
cat "$S/logs/gate-$label-md5.txt" >> "$L"
n=$(grep -cE "all = reference|all agree$" "$S/logs/gate-$label-md5.txt"); bad=$(grep -cE "DIFFER|NOT reference" "$S/logs/gate-$label-md5.txt")
[ "$bad" = 0 ] && [ "$n" = 15 ] && echo "ok   wasm baseline+simd: $n/15 streams bit-exact" || { echo "FAIL wasm bit-exactness: $n ok, $bad bad"; fail=1; }
echo "\$ native SVT clips against libdav1d" >> "$L"
nb=0; for f in $S/clips/*.ivf; do for t in 1 4; do m=$("$S/iter/$label/decode_ivf" "$f" --md5 --threads $t); [ "$m" = "$(cat $f.md5)" ] || { nb=1; echo "MISMATCH $f t$t" >> "$L"; }; done; done
[ $nb = 0 ] && echo "ok   native SVT clips, 1 and 4 threads: libdav1d MD5" || { echo "FAIL native SVT clips"; fail=1; }
[ $fail = 0 ] && echo "GATE PASS $label" || echo "GATE FAIL $label"
