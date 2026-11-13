#!/bin/bash
# steps-table.sh runs label... — rav1d and every given iter label in one interleaved run on the five clips of the step
# tables (wasm baseline and simd), then the same labels natively: one consistent table of all the steps.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; runs=$1; shift
T=$S/wasm-av1/testdata
files="$T/testsrc-320x180-8bit.ivf $S/clips/synth-testsrc2-720p.ivf $S/clips/synth-mandelbrot-720p.ivf $S/clips/synth-mandelbrot-720p-10bit.ivf $S/clips/synth-mandelbrot-1080p.ivf"
pk="rav1d=$S/builds/rav1d/pkg"; bins=""
for l in "$@"; do pk="$pk,$l=$S/iter/$l/pkg"; bins="$bins $S/iter/$l/decode_ivf"; done
uptime
node "$S/bench-node.mjs" $runs "$pk" baseline,simd $files
echo "--- native, 1 thread (median fps)"
python3 -I - $runs $(echo $bins | wc -w) $bins $files <<'PY'
import subprocess, sys, re, statistics, os
runs=int(sys.argv[1]); nb=int(sys.argv[2]); bins=sys.argv[3:3+nb]; files=sys.argv[3+nb:]
print(f"{'file':34s} " + " ".join(f"{os.path.basename(os.path.dirname(b)):>8s}" for b in bins))
for f in files:
    fps=[[] for _ in bins]; md5s=set()
    for _ in range(runs):
        for i,b in enumerate(bins):
            p=subprocess.run([b,f,'--threads','1'],capture_output=True,text=True)
            m=re.search(r'\(([\d.]+) ms/frame, ([\d.]+) fps\)',p.stderr)
            fps[i].append(float(m.group(2))); md5s.add(p.stdout.strip())
    print(f"{os.path.basename(f):34s} " + " ".join(f"{statistics.median(x):8.1f}" for x in fps) + ("  same md5" if len(md5s)==1 else "  MD5 DIFFER"))
PY
uptime
