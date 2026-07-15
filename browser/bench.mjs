// V8 (Node) timing runner for the **svm-in-wasm** cross-engine row: it times the SVM **bytecode
// engine, compiled to wasm and running on V8**, executing an encoded SVM IR kernel — the same
// LLVM-frontend IR the native `svm-bytecode` row runs (see bench/cross-engine/README.md). So the
// gap between this row and native `svm-bytecode` is exactly the cost of double-sandboxing the
// interpreter inside the wasm host.
//
//   node bench.mjs <svm_browser.wasm> <kernel.svmbc> <func> <sp> <small> <large>
//
// `<kernel.svmbc>` is the whole encoded module (svm-encode form); `<func>` is the index of the kernel
// entry to run (its export index in the translated module); `<sp>` is the frontend entry stack pointer.
// stdout: two lines — "<per_iter_ns>" then "<result@small>" — matching the native/V8/Wasmtime runners'
// parse, the second line a correctness anchor the Rust driver compares against native bytecode.
//
// Methodology mirrors the other runners exactly: per_iter = (min t(large) - min t(small)) / Δn, min
// over reps, after a warmup so V8 tiers the call site up to TurboFan. The encoded module is loaded
// into the wasm linear memory **once**; only the `svm_run_bench` call is timed (decode + bytecode
// compile happen inside it each call, but that fixed per-call cost cancels in the large/small
// subtraction, just as the native bytecode row's per-call compile does).
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';

const [wasmPath, kernelPath, funcS, spS, smallS, largeS] = process.argv.slice(2);
if (!largeS) {
  console.error('usage: node bench.mjs <svm_browser.wasm> <kernel.svmbc> <func> <sp> <small> <large>');
  process.exit(2);
}
const func = Number(funcS);
const sp = BigInt(spS);
const small = Number(smallS), large = Number(largeS);

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, engineImports())).exports;

// Pointers/lengths are usize: i32 (Number) on wasm32, i64 (BigInt) on wasm64. `func` is u32 (Number);
// `sp`/`n`/result are i64 (BigInt).
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));

// Load the encoded module into linear memory once (re-fetch the view: alloc may grow memory).
const bytes = readFileSync(kernelPath);
const ptr = ex.svm_alloc(N(bytes.length));
new Uint8Array(ex.memory.buffer).set(bytes, Number(ptr));
const len = N(bytes.length);

const runN = (n) => ex.svm_run_bench(ptr, len, func, sp, BigInt(n));
const result = (n) => {
  const r = runN(n);
  const st = ex.svm_status();
  if (st !== 0) {
    console.error(`${kernelPath} func ${func}: svm_run_bench status ${st}`);
    process.exit(3);
  }
  return r;
};

const REPS = 10;
const best = (n) => {
  runN(n); // warm up (lets V8 tier the call site up to TurboFan)
  let b = Infinity;
  for (let r = 0; r < REPS; r++) {
    const t = performance.now();
    runN(n);
    const e = performance.now();
    if (e - t < b) b = e - t;
  }
  return b * 1e6; // ms -> ns
};

const perIter = (best(large) - best(small)) / (large - small);
console.log(perIter.toFixed(6));
console.log(result(small).toString());
