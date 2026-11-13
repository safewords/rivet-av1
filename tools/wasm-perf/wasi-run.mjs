// node wasi-run.mjs test.wasm [args...]: runs a wasm32-wasip1 test binary (cargo test --target wasm32-wasip1 --no-run) under Node's WASI.
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: [file, ...args], env: process.env, preopens: { '.': process.cwd() } });
const module = await WebAssembly.compile(readFileSync(file));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
