// THREADS.md step 4 — run the real SVM engine over a guest window in **shared** wasm linear memory.
// Detects the threads build (imports a shared memory) vs the default build (exports its own), then
// allocs the module bytes + a guest window in that linear memory and runs a guest over a
// `Region::shared` view of the window via `svm_run_shared`. In the threads build the window lives in
// the host's SharedArrayBuffer — the substrate the parallel mode's per-vCPU Workers will execute over.
// Today the run is cooperative (one thread); this proves the integration of steps 1–3 in real wasm.
//
// Usage:  node threads-engine.mjs <module.wasm> [guest.svmbc] [expected]
import { readFileSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const guestPath = process.argv[3] ?? 'corpus/threads.svmbc';
const expect = BigInt(process.argv[4] ?? 4000);

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const sharedImport = WebAssembly.Module.imports(mod).some((i) => i.kind === 'memory');
const memory = sharedImport
  ? new WebAssembly.Memory({ initial: 512, maximum: 16384, shared: true })
  : undefined;
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const buf = () => (memory ?? ex.memory).buffer;
const shared = buf() instanceof SharedArrayBuffer;

const guest = readFileSync(guestPath);
const mp = ex.svm_alloc(guest.length);
new Uint8Array(buf()).set(guest, mp);
const winSize = 1 << 16;
const win = ex.svm_alloc(winSize);
new Uint8Array(buf(), win, winSize).fill(0);

const got = ex.svm_run_shared(mp, guest.length, win, winSize, 0n);
const pass = BigInt(got) === expect;

console.log(`module: ${wasmPath}`);
console.log(`  linear memory: ${buf().byteLength >> 20} MiB  shared=${shared}`);
console.log(`  svm_run_shared(guest over Region::shared window @0x${win.toString(16)}) = ${got}` +
  `  expect ${expect}  ${pass ? 'PASS' : 'FAIL'}`);
console.log(`\n${pass ? 'PASS' : 'FAIL'}: the SVM engine runs over a ${shared ? 'SHARED' : 'private'} ` +
  `wasm linear-memory window`);
process.exit(pass ? 0 : 1);
