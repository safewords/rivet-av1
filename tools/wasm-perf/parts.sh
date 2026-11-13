#!/bin/bash
# parts.sh label variant clip [runs] — V8 cpu profile (200 us sampling) of decoding clip with iter/<label>/pkg/<variant>;
# ms per frame of the two parts this work is about, from the profile's shares and the fps of the same run.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
d="$S/prof/parts-$1-$2"; rm -rf "$d"
line=$(node --cpu-prof --cpu-prof-interval 200 --cpu-prof-dir="$d" "$S/bench-node.mjs" "${4:-5}" x="$S/iter/$1/pkg" "$2" "$3" | tail -1)
python3 -I - "$d" "$line" "${TOP:-0}" <<'PY'
import json,sys,glob,collections,re
f=sorted(glob.glob(sys.argv[1]+'/*.cpuprofile'))[-1]
fps=float(re.search(r'([\d.]+) fps',sys.argv[2]).group(1))
p=json.load(open(f))
nodes={n['id']:n for n in p['nodes']}
c=collections.Counter()
for s,d in zip(p['samples'],p['timeDeltas']):
    c[nodes[s]['callFrame']['functionName'] or '(anon)']+=d
for k in ('update','(idle)','(garbage collector)','(program)'): c.pop(k,None)
tot=sum(c.values()); ms=1000/fps
def grp(name):
    n=re.sub(r'\[[0-9a-f]{16}\]','',name)
    if 'coeffs' in n or 'symbol::' in n or 'SymbolDecoder' in n: return 'entropy+coefficients'
    if 'itx::' in n or 'reconstruct' in n or 'equantis' in n: return 'transforms+reconstruction'
    return None
g=collections.Counter()
for name,t in c.items():
    k=grp(name)
    if k: g[k]+=t
print(sys.argv[2].strip())
print(f"frame {ms:6.2f} ms   entropy+coefficients {ms*g['entropy+coefficients']/tot:5.2f} ms   transforms+reconstruction {ms*g['transforms+reconstruction']/tot:5.2f} ms")
for name,t in c.most_common(int(sys.argv[3])):
    name=re.sub(r'\[[0-9a-f]{16}\]','',name)
    print(f"  {100*t/tot:5.1f}% {ms*t/tot:6.3f} ms {'*' if grp(name) else ' '} {name[:110]}")
PY
