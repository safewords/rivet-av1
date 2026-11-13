#!/bin/bash
# prof.sh label variant clip [n] — V8 cpu profile of decoding clip with iter/<label>/pkg/<variant>; top n self-time functions (md5 'update' excluded from the total).
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
d="$S/prof/out-$1-$2"; rm -rf "$d"
node --cpu-prof --cpu-prof-dir="$d" "$S/bench-node.mjs" 3 x="$S/iter/$1/pkg" "$2" "$3" | tail -1
python3 -I - "$d" "${4:-25}" <<'PY'
import json,sys,glob,collections,re
f=sorted(glob.glob(sys.argv[1]+'/*.cpuprofile'))[-1]
p=json.load(open(f))
nodes={n['id']:n for n in p['nodes']}
c=collections.Counter()
for s,d in zip(p['samples'],p['timeDeltas']):
    c[nodes[s]['callFrame']['functionName'] or '(anon)']+=d
for k in ('update','(idle)','(garbage collector)','(program)'): c.pop(k,None)
tot=sum(c.values())
for name,t in c.most_common(int(sys.argv[2])):
    name=re.sub(r'\[[0-9a-f]{16}\]','',name)
    print(f"{100*t/tot:5.1f}%  {name[:120]}")
PY
