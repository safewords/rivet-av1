// node bench-turns.mjs RUNS label=pkgdir[,label=pkgdir...] variant[,variant] file.ivf...
// bench-node.mjs with the builds taking turns: one warm-up pass of each, then RUNS rounds in which every build decodes
// the file once, so that a burst of other work on the machine falls on all of them and not on one build's seven runs.
// Decode fps (time inside dec.run() only), median of RUNS, and the MD5 of the planes.
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [runsS, pkgsS, variantsS, ...files] = process.argv.slice(2);
const runs = +runsS;
const pkgs = pkgsS.split(',').map((s) => s.split('='));
const variants = variantsS.split(',');
async function load(dir, variant) {
  const d = path.join(dir, variant);
  const mod = await import(pathToFileURL(path.join(d, 'wasm_av1.js')).href + '?' + encodeURIComponent(dir));
  const wasm = await mod.default({ module_or_path: readFileSync(path.join(d, 'wasm_av1_bg.wasm')) });
  return { mod, wasm };
}
function once({ mod, wasm }, ivf) {
  const dec = new mod.Av1Decoder(4);
  dec.setSourceIvf(ivf);
  const md5 = createHash('md5');
  let frames = 0, ms = 0;
  while (!dec.finished()) {
    const t0 = performance.now();
    dec.run();
    ms += performance.now() - t0;
    while (dec.nextFrame()) { md5.update(new Uint8Array(wasm.memory.buffer, dec.framePtr(), dec.frameLen())); frames++; }
  }
  dec.free();
  return { frames, ms, md5: md5.digest('hex') };
}
const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
const rts = {};
for (const [label, dir] of pkgs) for (const v of variants) rts[`${label}/${v}`] = await load(dir, v);
const keys = Object.keys(rts);
console.log('file'.padEnd(34) + keys.map((k) => k.padStart(18)).join('') + '   md5');
for (const f of files) {
  const ivf = readFileSync(f);
  const ref = existsSync(f + '.md5') ? readFileSync(f + '.md5', 'utf8').trim() : existsSync(f.replace(/\.ivf$/, '.ref')) ? readFileSync(f.replace(/\.ivf$/, '.ref'), 'utf8').trim().split(/\s+/)[4] : null;
  let line = path.basename(f).padEnd(34);
  const md5s = new Set();
  const fps = Object.fromEntries(keys.map((k) => [k, []]));
  for (const k of keys) once(rts[k], ivf);
  for (let i = 0; i < runs; i++) for (const k of keys) { const r = once(rts[k], ivf); fps[k].push(r.frames / (r.ms / 1000)); md5s.add(r.md5); }
  for (const k of keys) line += (med(fps[k]).toFixed(1) + ' fps').padStart(18);
  line += '   ' + (md5s.size === 1 ? (ref ? ([...md5s][0] === ref ? 'all = reference' : 'all agree, NOT reference ' + [...md5s][0]) : 'all agree') : 'DIFFER');
  console.log(line);
}
