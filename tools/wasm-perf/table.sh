#!/bin/bash
# table.sh runs before after [all] — fps of rav1d, <before> and <after> (iter labels): wasm baseline and simd (median of runs,
# builds interleaved per clip), then native (decode_ivf of before/after, 1 thread). "all": the fourteen streams, else the five.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"; runs=$1; b=$2; a=$3
T=$S/wasm-av1/testdata
if [ "$4" = all ]; then files="$T/testsrc-320x180-8bit.ivf $T/testsrc-160x90-10bit.ivf $T/testsrc-128x72-grain.ivf $T/testsrc-177x99-8bit.ivf $T/testsrc-96x64-444.ivf $T/testsrc-96x64-mono.ivf $S/vectors/av1-1-b8-05-mv.ivf $S/vectors/av1-1-b10-00-quantizer-10.ivf $S/vectors/av1-1-b8-23-film_grain-50.ivf $S/vectors/av1-1-b8-00-quantizer-20.ivf $S/clips/synth-testsrc2-720p.ivf $S/clips/synth-mandelbrot-720p.ivf $S/clips/synth-mandelbrot-720p-10bit.ivf $S/clips/synth-mandelbrot-1080p.ivf"
else files="$T/testsrc-320x180-8bit.ivf $S/clips/synth-testsrc2-720p.ivf $S/clips/synth-mandelbrot-720p.ivf $S/clips/synth-mandelbrot-720p-10bit.ivf $S/clips/synth-mandelbrot-1080p.ivf"; fi
uptime
node "$S/bench-node.mjs" $runs rav1d=$S/builds/rav1d/pkg,$b=$S/iter/$b/pkg,$a=$S/iter/$a/pkg baseline,simd $files
echo "--- native, 1 thread (median / best fps)"
python3 -I - $runs "$S/iter/$b/decode_ivf" "$S/iter/$a/decode_ivf" $files <<'PY'
import subprocess, sys, re, statistics, os
runs=int(sys.argv[1]); bins=sys.argv[2:4]; files=sys.argv[4:]
print(f"{'file':38s} " + " ".join(f"{os.path.basename(os.path.dirname(b)):>22s}" for b in bins))
for f in files:
    fps=[[] for _ in bins]; md5s=set()
    for _ in range(runs):
        for i,b in enumerate(bins):
            p=subprocess.run([b,f,'--threads','1'],capture_output=True,text=True)
            m=re.search(r'\(([\d.]+) ms/frame, ([\d.]+) fps\)',p.stderr)
            fps[i].append(float(m.group(2))); md5s.add(p.stdout.strip())
    print(f"{os.path.basename(f):38s} " + " ".join(f"{statistics.median(x):9.1f} / {max(x):9.1f}" for x in fps) + ("  same md5" if len(md5s)==1 else "  MD5 DIFFER"))
PY
uptime
