#!/bin/bash
# lfprof.sh <runs> <clip> <label/variant>...: the loop filter's ms per shown frame in each wasm build.
# Each run: a V8 CPU profile of bench-node (1 warm-up + 3 timed decodes) gives the loop filter's share of the samples
# (functions named loop_filter / ::lf:: / edge_ / filter_strength / filter_band / FilterCtx / LfLevels; md5, idle, GC and
# "(program)" left out of the total, as prof.sh does); an unprofiled bench-node run (median of 5) gives the frame time.
# Builds are interleaved; the median of the runs is printed.
S="${WASM_PERF:?}"; runs="$1"; clip="$2"; shift 2
for lv in "$@"; do : > "$S/prof/lf-${lv//\//-}.ms"; : > "$S/prof/lf-${lv//\//-}.fps"; done
for i in $(seq "$runs"); do for lv in "$@"; do
  l="${lv%%/*}"; v="${lv##*/}"; d="$S/prof/lfout-$l-$v"; rm -rf "$d"
  node --cpu-prof --cpu-prof-dir="$d" "$S/bench-node.mjs" 3 x="$S/iter/$l/pkg" "$v" "$clip" > /dev/null
  share=$(python3 -I - "$d" <<'PY'
import json,sys,glob,collections,re
f=sorted(glob.glob(sys.argv[1]+'/*.cpuprofile'))[-1]
p=json.load(open(f))
nodes={n['id']:n for n in p['nodes']}
c=collections.Counter()
for s,d in zip(p['samples'],p['timeDeltas']):
    c[nodes[s]['callFrame']['functionName'] or '(anon)']+=d
for k in ('update','(idle)','(garbage collector)','(program)'): c.pop(k,None)
lf=sum(t for n,t in c.items() if re.search(r'loop_filter|::lf::|edge_|filter_strength|filter_band|FilterCtx|filter_level|LfLevels',n))
print(lf/sum(c.values()))
PY
)
  fps=$(node "$S/bench-node.mjs" 5 x="$S/iter/$l/pkg" "$v" "$clip" | tail -1 | grep -o '[0-9.]* fps' | awk '{print $1}')
  echo "$share $fps" | awk '{printf "%.4f\n", $1*1000/$2}' >> "$S/prof/lf-$l-$v.ms"; echo "$fps" >> "$S/prof/lf-$l-$v.fps"
done; done
med() { sort -n "$1" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
for lv in "$@"; do k="${lv//\//-}"; printf "%-22s loop filter ms/frame %s   fps %s\n" "$lv" "$(med "$S/prof/lf-$k.ms")" "$(med "$S/prof/lf-$k.fps")"; done
