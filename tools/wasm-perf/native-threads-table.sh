#!/bin/bash
# native-threads-table.sh ROUNDS "label=decode_ivf ..." "1 2 4 8" clip.ivf... — wasm-av1's decode_ivf builds natively at
# each thread count, taking turns (every round runs each build at each thread count once), median fps of the rounds,
# and whether every run gave the clip's MD5.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
rounds="$1"; builds="$2"; threads="$3"; shift 3
raw="$S/logs/native-threads-table-$$.log"; : > "$raw"
echo "load before: $(uptime | sed 's/.*load averages*: //')"
for clip in "$@"; do
  ref=$(cat "$clip.md5" 2>/dev/null)
  for ((i = 0; i < rounds; i++)); do
    for b in $builds; do
      label="${b%%=*}"; bin="${b#*=}"
      for t in $threads; do
        out=$("$bin" "$clip" --threads "$t" 2>&1)
        fps=$(echo "$out" | sed -n 's/.*(\([0-9.]*\) ms\/frame, \([0-9.]*\) fps).*/\2/p' | tail -1)
        md5=$(echo "$out" | grep -oE '\b[0-9a-f]{32}\b' | tail -1)
        ok=ok; [ -n "$ref" ] && [ "$md5" != "$ref" ] && ok=BAD
        echo "$label $t $(basename "$clip") $fps $ok" >> "$raw"
      done
    done
  done
done
echo "load after: $(uptime | sed 's/.*load averages*: //')"
python3 -I - "$raw" "$builds" "$threads" "$@" <<'PY'
import sys, statistics, os, collections
raw, builds, threads, clips = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4:]
labels = [b.split('=')[0] for b in builds]
fps = collections.defaultdict(list); bad = set()
for line in open(raw):
    p = line.split()
    if len(p) != 5:
        print('UNPARSED', line.strip()); continue
    label, t, clip, f, ok = p
    fps[(clip, label, t)].append(float(f))
    if ok != 'ok': bad.add((clip, label, t))
print('| clip | build | ' + ' | '.join(threads) + ' |'); print('|---|---|' + '---|' * len(threads))
for c in clips:
    c = os.path.basename(c)
    for l in labels:
        cells = []
        for t in threads:
            v = fps.get((c, l, t))
            cells.append('-' if not v else f"{statistics.median(v):.0f}" + (' BAD-MD5' if (c, l, t) in bad else ''))
        print(f"| {c} | {l} | " + ' | '.join(cells) + ' |')
PY
