#!/usr/bin/env python3 -I
"""bench-native.py RUNS BIN... : decode fps per file, median and best of RUNS."""
import subprocess, sys, re, statistics, glob, os
S=os.environ['WASM_PERF']
W=S+'/wasm-av1'
runs=int(sys.argv[1]); threads=sys.argv[2]; bins=sys.argv[3:]
files=sorted(glob.glob(W+'/testdata/*.ivf'))+sorted(glob.glob(S+'/vectors/*.ivf'))+sorted(glob.glob(S+'/clips/*.ivf'))
print(f"{'file':38s} " + " ".join(f"{os.path.basename(os.path.dirname(b)):>22s}" for b in bins))
for f in files:
    row=[]; md5s=[]
    for b in bins:
        fps=[]
        for _ in range(runs):
            p=subprocess.run([b,f,'--threads',threads],capture_output=True,text=True)
            m=re.search(r'\(([\d.]+) ms/frame, ([\d.]+) fps\)',p.stderr)
            fps.append(float(m.group(2)))
            md5=p.stdout.strip()
        md5s.append(md5)
        row.append(f"{statistics.median(fps):9.1f} / {max(fps):9.1f}")
    same = 'same md5' if len(set(md5s))==1 else 'MD5 DIFFER'
    print(f"{os.path.basename(f):38s} " + " ".join(f"{r:>22s}" for r in row) + "  " + same)
