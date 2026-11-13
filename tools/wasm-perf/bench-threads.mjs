// node bench-threads.mjs RUNS pkgdir variant THREADS file.ivf...
// A threads build driven from Node: the decoder on the main thread (Node lets it block), each wasm thread a
// worker_threads Worker on the shared memory (thread-shim.mjs). Prints fps (median of RUNS) and the planes' MD5.
// The fps is the time inside dec.run() alone, as in bench-node.mjs: a decoder whose threads go on decoding while the
// caller hashes the last frame (rav1d's frame threads; this crate's frames in flight) gets that time for nothing.
// WALL=1 times the whole loop instead, frames taken and not hashed (the MD5 is then of a pass before the timed ones).
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Worker } from 'node:worker_threads';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
const [runsS, dir, variant, threadsS, ...files] = process.argv.slice(2);
const runs = +runsS, threads = +threadsS;
const glue = pathToFileURL(path.join(dir, variant, 'wasm_av1.js')).href;
const mod = await import(glue);
const wasm = await mod.default({ module_or_path: readFileSync(path.join(dir, variant, 'wasm_av1_bg.wasm')) });
let spawned = 0;
globalThis.__wasmAv1SpawnThread = (module, memory, ptr) => {
  spawned++;
  const w = new Worker(path.join(path.dirname(fileURLToPath(import.meta.url)), 'thread-shim.mjs'), { workerData: { glue, module, memory, ptr, stackSize: 2 * 1024 * 1024 } });
  w.on('error', (e) => { console.error('thread died:', e); process.exit(3); });
};
const wall = !!process.env.WALL;
function once(ivf, hash = true) {
  const dec = new mod.Av1Decoder(4, true, threads);
  const applied = dec.threads();
  dec.setSourceIvf(ivf);
  const md5 = createHash('md5');
  let frames = 0, ms = 0;
  const start = performance.now();
  while (!dec.finished()) {
    const t0 = performance.now();
    dec.run();
    ms += performance.now() - t0;
    while (dec.nextFrame()) {
      if (hash) md5.update(new Uint8Array(wasm.memory.buffer, dec.framePtr(), dec.frameLen()).slice());
      frames++;
    }
  }
  if (!hash) ms = performance.now() - start;
  dec.free();
  return { frames, ms, md5: md5.digest('hex'), applied };
}
const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
for (const f of files) {
  const ivf = readFileSync(f);
  const ref = existsSync(f + '.md5') ? readFileSync(f + '.md5', 'utf8').trim() : existsSync(f.replace(/\.ivf$/, '.ref')) ? readFileSync(f.replace(/\.ivf$/, '.ref'), 'utf8').trim().split(/\s+/)[4] : null;
  const first = once(ivf);
  await new Promise((r) => setTimeout(r, 300)); // let the Workers come up before timing
  const fps = []; let r;
  for (let i = 0; i < runs; i++) { r = once(ivf, !wall); fps.push(r.frames / (r.ms / 1000)); }
  if (wall) r.md5 = first.md5;
  console.log(`${path.basename(f).padEnd(34)} ${variant} threads asked ${threads} applied ${r.applied} workers spawned ${spawned}  ${med(fps).toFixed(1).padStart(8)} fps  ${r.frames} frames  md5 ${ref ? (r.md5 === ref ? '= reference' : 'NOT REFERENCE ' + r.md5) : r.md5}`);
}
process.exit(0);
