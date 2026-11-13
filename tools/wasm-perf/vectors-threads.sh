#!/bin/bash
# vectors-threads.sh <label> [threads...] — every IVF vector of tests/vectors through the simd-threads build
# iter/<label>/pkg at each thread count (default 1 4 8), against rav1d's simd-threads build on one thread: the MD5 of
# all the planes of all the frames must be the same. Prints how many are, and the ones that are not.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
here="$(cd "$(dirname "$0")" && pwd)"
label="$1"; shift; [ $# = 0 ] && set -- 1 4 8
out="$S/logs/vectors-threads-$label"; mkdir -p "$out"
files=("$S"/rivet-av1/tests/vectors/*.ivf)
node "$here/bench-threads.mjs" 1 "$S/builds/rav1d/pkg" simd-threads 1 "${files[@]}" > "$out/rav1d.txt" 2>&1
for t in "$@"; do
  node "$here/bench-threads.mjs" 1 "$S/iter/$label/pkg" simd-threads "$t" "${files[@]}" > "$out/t$t.txt" 2>&1
done
python3 -I - "$out" "$@" <<'PY'
import sys, re, os
out, threads = sys.argv[1], sys.argv[2:]
def md5s(path):
    d = {}
    for line in open(path):
        m = re.match(r'(\S+\.ivf)\s.*?(\d+) frames\s+md5 (?:NOT REFERENCE |= reference)?\s*([0-9a-f]{32})?', line)
        if m: d[m.group(1)] = (m.group(2), m.group(3) or 'reference')
    return d
ref = md5s(os.path.join(out, 'rav1d.txt'))
print(f"rav1d: {len(ref)} vectors")
bad = 0
for t in threads:
    got = md5s(os.path.join(out, f't{t}.txt'))
    same = [f for f in ref if got.get(f) == ref[f]]
    diff = [f for f in ref if got.get(f) != ref[f]]
    print(f"{t} thread(s): {len(same)} of {len(ref)} the same as rav1d")
    for f in diff[:20]: print('  DIFFERS', f, got.get(f), ref[f])
    bad += len(diff)
sys.exit(1 if bad or not ref else 0)
PY
