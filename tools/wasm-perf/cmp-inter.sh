#!/bin/bash
# cmp-inter.sh variant clip rounds label... — prof-inter.sh for each label, the builds taking turns:
# the median, least and most of inter prediction's ms/frame over the rounds, and the decode's fps.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; variant="$1"; clip="$2"; rounds="$3"; shift 3
here="$(cd "$(dirname "$0")" && pwd)"
tmp=$(mktemp -d)
for i in $(seq "$rounds"); do
  for l in "$@"; do "$here/prof-inter.sh" "$l" "$variant" "$clip" 3 | head -2 | tr '\n' ' ' >> "$tmp/$l"; echo >> "$tmp/$l"; done
done
for l in "$@"; do python3 -I - "$l" "$tmp/$l" "$variant" <<'PY'
import sys,re,statistics
l,f,v=sys.argv[1:4]
fps=[];inter=[]
for line in open(f):
    m=re.search(r'([\d.]+) fps.*inter prediction ([\d.]+) ms',line)
    if m: fps.append(float(m.group(1))); inter.append(float(m.group(2)))
print(f"{l:10s} {v:8s} inter {statistics.median(inter):.3f} ms/frame (min {min(inter):.3f} max {max(inter):.3f})   decode {statistics.median(fps):.1f} fps (max {max(fps):.1f})   n={len(fps)}")
PY
done
rm -rf "$tmp"
