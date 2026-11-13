#!/usr/bin/env python3 -I
"""steprow.py TITLE LABEL [PREV]: the step table rows of docs/wasm-performance.md for iter/LABEL: the five clips
through rav1d and iter/LABEL (and iter/PREV, for a before/after in the same run) in both wasm builds (bench-turns,
median of 7) and natively on one thread (decode_ivf, median of 7, the builds interleaved)."""
import subprocess, sys, os, re, statistics
S=os.environ['WASM_PERF']; title, label = sys.argv[1], sys.argv[2]; prev = sys.argv[3] if len(sys.argv) > 3 else None
clips=[('fixture 320x180', S+'/wasm-av1/testdata/testsrc-320x180-8bit.ivf'), ('testsrc2 720p', S+'/clips/synth-testsrc2-720p.ivf'),
       ('mandelbrot 720p', S+'/clips/synth-mandelbrot-720p.ivf'), ('mandelbrot 720p 10-bit', S+'/clips/synth-mandelbrot-720p-10bit.ivf'),
       ('mandelbrot 1080p', S+'/clips/synth-mandelbrot-1080p.ivf')]
pk=f"rav1d={S}/builds/rav1d/pkg,rivet={S}/iter/{label}/pkg" + (f",prev={S}/iter/{prev}/pkg" if prev else "")
out=subprocess.run(['node', S+'/bench-turns.mjs', '7', pk, 'baseline,simd']+[c[1] for c in clips], capture_output=True, text=True).stdout
sys.stderr.write(out)
rows={}
for line in out.splitlines()[1:]:
    f=line.split()[0]; v=[float(x) for x in re.findall(r'([\d.]+) fps', line)]
    rows[f]=(v, 'all = reference' in line or line.rstrip().endswith('all agree'))
def native(b, f):
    p=subprocess.run([b, f, '--threads', '1'], capture_output=True, text=True)
    return float(re.search(r'\(([\d.]+) ms/frame, ([\d.]+) fps\)', p.stderr).group(2))
bins=[f"{S}/iter/{label}/decode_ivf"] + ([f"{S}/iter/{prev}/decode_ivf"] if prev else [])
nat={}
for name, f in clips:
    r={b: [] for b in bins}
    for _ in range(7):
        for b in bins: r[b].append(native(b, f))
    nat[name]=[statistics.median(r[b]) for b in bins]
fmt=lambda x: f"{x:.0f}" if x >= 100 else f"{x:.1f}"
for i,(name, f) in enumerate(clips):
    v, ok = rows[os.path.basename(f)]
    assert ok, name
    cell=lambda mine, ref: f"{fmt(mine)} ({mine/ref:.2f}x)"
    print(f"| {title if i == 0 else ''} | {name} | {cell(v[2], v[0])} | {cell(v[3], v[1])} | {fmt(nat[name][0])} |")
if prev:
    print("\nbefore (same run):")
    for name, f in clips:
        v, _ = rows[os.path.basename(f)]
        print(f"|  | {name} | {fmt(v[4])} ({v[4]/v[0]:.2f}x) | {fmt(v[5])} ({v[5]/v[1]:.2f}x) | {fmt(nat[name][1])} |")
