// node wasi-run.mjs test.wasm [args...] — runs a wasm32-wasip1 test binary (cargo's runner for that target).
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: [file, ...args], env: process.env, preopens: { '/': '/' } });
const mod = await WebAssembly.compile(readFileSync(file));
const inst = await WebAssembly.instantiate(mod, wasi.getImportObject());
wasi.start(inst);
