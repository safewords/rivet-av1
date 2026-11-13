import { readFileSync } from 'node:fs';
const { instance } = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
const cases = instance.exports.check_all();
if (!cases) { console.log('FAIL: the transform differs from the scalar reference'); process.exit(1); }
console.log(cases + ' cases agree with the scalar reference');
if (process.argv[3] === 'check') process.exit(0);
const names = ['4x4', '8x8', '16x16', '32x32', '64x64', '8x16', '16x32', '32x16'];
for (let si = 0; si < names.length; si++) {
  let line = names[si].padStart(6);
  for (const [full, bounded] of [[1, 0], [1, 1], [0, 0], [0, 1]]) {
    const iters = 20000; let best = 1e30;
    instance.exports.bench(si, full, bounded, iters);
    for (let r = 0; r < 5; r++) { const t0 = performance.now(); instance.exports.bench(si, full, bounded, iters); best = Math.min(best, (performance.now() - t0) * 1e6 / iters); }
    line += '  ' + best.toFixed(0).padStart(8);
  }
  console.log(line);
}
