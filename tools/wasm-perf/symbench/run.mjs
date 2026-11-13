import { readFileSync } from 'node:fs';
const [file, roundsS, nvS] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(file), {});
const n = instance.exports.init();
const nv = +nvS, rounds = +roundsS;
const t = Array.from({ length: nv }, () => []);
for (let v = 0; v < nv; v++) instance.exports.bench(v); // warm-up / tier-up
for (let r = 0; r < rounds; r++) for (let v = 0; v < nv; v++) {
  const t0 = performance.now();
  const ok = instance.exports.bench(v);
  const dt = performance.now() - t0;
  if (!ok) throw new Error('variant ' + v + ' wrong');
  t[v].push(dt * 1e6 / n);
}
for (let v = 0; v < nv; v++) { t[v].sort((a, b) => a - b); console.log(`variant ${v}  min ${t[v][0].toFixed(2)} ns/op  median ${t[v][rounds >> 1].toFixed(2)}`); }
