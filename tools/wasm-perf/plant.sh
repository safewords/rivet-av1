#!/bin/bash
# plant.sh file 'old' 'new' [test args...] — a copy of the working tree with one planted difference, and the committed vectors
# (which decode with CoefficientReader::Checked) run on it: they are expected to FAIL.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; export PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=3 RUSTUP_TOOLCHAIN=1.100.0
rsync -a --delete --exclude target --exclude tests/vectors --exclude .git "$S/rivet-av1/" "$S/plant-tree/"
cd "$S/plant-tree" || exit 1
python3 -I -c "
import sys
p,a,b=sys.argv[1:4]
s=open(p).read()
assert s.count(a)>=1, 'anchor not found'
open(p,'w').write(s.replace(a,b,1))" "$1" "$2" "$3" || exit 1
shift 3
cargo test --release "$@" 2>&1 | grep -E "^test |panicked|differs|test result|error" | head -12
