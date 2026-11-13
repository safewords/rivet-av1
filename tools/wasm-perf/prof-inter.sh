#!/bin/bash
# prof-inter.sh label variant clip [runs] [frames] — V8's CPU profile of decoding clip with iter/<label>/pkg/<variant>:
# the time under predict_inter (it and everything it calls) as ms per decoded frame, and the functions it is in.
# ROOT=<regex> names other roots; TOPN=<n> lists more functions.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
runs="${4:-9}"; frames="${5:-60}"
d="$S/prof/inter-$1-$2"; rm -rf "$d"
out=$(node --cpu-prof --cpu-prof-dir="$d" "$S/bench-node.mjs" "$runs" x="$S/iter/$1/pkg" "$2" "$3" | tail -1)
python3 -I - "$d" "$runs" "$frames" "$out" <<'PY'
import json,sys,glob,collections,re,os
f=sorted(glob.glob(sys.argv[1]+'/*.cpuprofile'))[-1]
runs=int(sys.argv[2]); frames=int(sys.argv[3])
root=os.environ.get('ROOT','predict_inter'); topn=int(os.environ.get('TOPN','8'))
p=json.load(open(f))
nodes={n['id']:n for n in p['nodes']}
parent={}
for n in p['nodes']:
    for c in n.get('children',[]): parent[c]=n['id']
selft=collections.Counter()
for s,d in zip(p['samples'],p['timeDeltas']): selft[s]+=d
def under(i):
    while i is not None:
        if re.search(root,nodes[i]['callFrame']['functionName']): return True
        i=parent.get(i)
    return False
n=frames*(runs+1)  # the warm-up pass is profiled too
inter=0; byfn=collections.Counter()
for i,t in selft.items():
    if under(i):
        inter+=t; byfn[re.sub(r'\[[0-9a-f]{16}\]','',nodes[i]['callFrame']['functionName'])]+=t
tot=sum(t for i,t in selft.items() if nodes[i]['callFrame']['functionName'] not in ('update','(idle)','(garbage collector)','(program)'))
print(sys.argv[4].strip())
print(f"inter prediction {inter/1000/n:.3f} ms/frame   (profiled total {tot/1000/n:.2f} ms/frame, {n} frames)")
for k,t in byfn.most_common(topn): print(f"   {t/1000/n:.3f}  {k[:90]}")
PY
