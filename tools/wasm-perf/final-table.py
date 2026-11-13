#!/usr/bin/env python3 -I
"""final-table.py BEFORE AFTER: the fourteen streams of docs/wasm-performance.md through rav1d, iter/BEFORE and iter/AFTER:
natively on one thread (decode_ivf, median of 7, the three binaries taking turns) and in both wasm builds (bench-turns,
median of 7). A markdown table in the document's layout, ratios to rav1d."""
import subprocess, sys, os, re, statistics
S=os.environ['WASM_PERF']; before, after = sys.argv[1], sys.argv[2]
T=S+'/wasm-av1/testdata/'; V=S+'/vectors/'; C=S+'/clips/'
streams=[('fixture 320x180, 8-bit', T+'testsrc-320x180-8bit.ivf'), ('fixture 160x90, 10-bit', T+'testsrc-160x90-10bit.ivf'),
 ('fixture 128x72, film grain', T+'testsrc-128x72-grain.ivf'), ('fixture 177x99, 8-bit', T+'testsrc-177x99-8bit.ivf'),
 ('fixture 96x64, 4:4:4', T+'testsrc-96x64-444.ivf'), ('fixture 96x64, mono', T+'testsrc-96x64-mono.ivf'),
 ('AOM b8-05-mv 352x288', V+'av1-1-b8-05-mv.ivf'), ('AOM b10-00-quantizer-10, 10-bit', V+'av1-1-b10-00-quantizer-10.ivf'),
 ('AOM b8-23-film_grain-50', V+'av1-1-b8-23-film_grain-50.ivf'), ('AOM b8-00-quantizer-20', V+'av1-1-b8-00-quantizer-20.ivf'),
 ('SVT testsrc2 720p', C+'synth-testsrc2-720p.ivf'), ('SVT mandelbrot 720p', C+'synth-mandelbrot-720p.ivf'),
 ('SVT mandelbrot 720p, 10-bit', C+'synth-mandelbrot-720p-10bit.ivf'), ('SVT mandelbrot 1080p', C+'synth-mandelbrot-1080p.ivf')]
pk=f"rav1d={S}/builds/rav1d/pkg,before={S}/iter/{before}/pkg,after={S}/iter/{after}/pkg"
out=subprocess.run(['node', S+'/bench-turns.mjs', '7', pk, 'baseline,simd']+[s[1] for s in streams], capture_output=True, text=True).stdout
sys.stderr.write(out)
wasm={}
for line in out.splitlines()[1:]:
    v=[float(x) for x in re.findall(r'([\d.]+) fps', line)]
    assert 'all = reference' in line or line.rstrip().endswith('all agree'), line
    wasm[line.split()[0]]=v
def native(b, f):
    p=subprocess.run([b, f, '--threads', '1'], capture_output=True, text=True)
    md5=re.search(r'MD5=(\w+)', p.stdout+p.stderr)
    return float(re.search(r'\(([\d.]+) ms/frame, ([\d.]+) fps\)', p.stderr).group(2)), md5.group(1) if md5 else None
bins=[f"{S}/builds/rav1d/decode_ivf", f"{S}/iter/{before}/decode_ivf", f"{S}/iter/{after}/decode_ivf"]
fmt=lambda x: f"{x:.0f}" if x >= 100 else f"{x:.1f}"
print("| stream | native: rav1d | before | after | wasm baseline: rav1d | before | after | wasm simd: rav1d | before | after |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, f in streams:
    r={b: [] for b in bins}; md5s=set()
    for _ in range(7):
        for b in bins:
            fps, m = native(b, f); r[b].append(fps); md5s.add(m)
    assert len(md5s)==1, (name, md5s)
    n=[statistics.median(r[b]) for b in bins]
    w=wasm[os.path.basename(f)]   # rav1d b, rav1d s, before b, before s, after b, after s
    cell=lambda x, ref: f"{fmt(x)} ({x/ref:.2f}x)"
    print(f"| {name} | {fmt(n[0])} | {fmt(n[1])} | {cell(n[2], n[0])} | {fmt(w[0])} | {fmt(w[2])} | {cell(w[4], w[0])} | {fmt(w[1])} | {fmt(w[3])} | {cell(w[5], w[1])} |")
    sys.stdout.flush()
