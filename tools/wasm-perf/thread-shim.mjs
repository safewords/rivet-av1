// Node stand-in for pkg/thread-worker.js: one wasm thread in a worker_threads Worker.
import { workerData } from 'node:worker_threads';
const { glue, module, memory, ptr, stackSize } = workerData;
const mod = await import(glue);
await mod.default({ module_or_path: module, memory, thread_stack_size: stackSize });
mod.__wasm_av1_thread_entry(ptr);
