#!/bin/bash
# threads-table.sh ROUNDS "label=pkgdir[:variant] ..." "1 2 4 8" clip.ivf... — the threads builds in Node (bench-threads.mjs),
# taking turns: every round runs each build at each thread count once (one warm-up decode and one timed decode per
# process), so that other work on the machine falls on all of them alike. Prints the median fps of the rounds per clip,
# build and thread count, and fails a cell whose MD5 is not the reference. Raw lines go to logs/threads-table-<pid>.log.
S="${WASM_PERF:?set WASM_PERF to the working directory described in README.md}"
here="$(cd "$(dirname "$0")" && pwd)"
rounds="$1"; builds="$2"; threads="$3"; shift 3
raw="$S/logs/threads-table-$$.log"; : > "$raw"
echo "load before: $(uptime | sed 's/.*load averages*: //')"
for clip in "$@"; do
  for ((i = 0; i < rounds; i++)); do
    for b in $builds; do
      label="${b%%=*}"; dir="${b#*=}"; variant=simd-threads
      case "$dir" in *:*) variant="${dir##*:}"; dir="${dir%:*}";; esac
      for t in $threads; do
        out=$(node "$here/bench-threads.mjs" 1 "$dir" "$variant" "$t" "$clip" 2>&1 | tail -1)
        echo "$label $t $out" >> "$raw"
      done
    done
  done
done
echo "load after: $(uptime | sed 's/.*load averages*: //')"
python3 -I - "$raw" "$builds" "$threads" "$@" <<'PY'
import sys, re, statistics, os, collections
raw, builds, threads, clips = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4:]
labels = [b.split('=')[0] for b in builds]
fps = collections.defaultdict(list); bad = set()
for line in open(raw):
    m = re.match(r'(\S+) (\d+) (\S+)\s.*?([\d.]+) fps\s+(\d+) frames\s+md5 (.*)', line)
    if not m:
        print('UNPARSED', line.strip()); continue
    label, t, clip, f, frames, md5 = m.groups()
    fps[(clip, label, t)].append(float(f))
    if not md5.startswith('= reference'): bad.add((clip, label, t))
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
