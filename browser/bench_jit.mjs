// V8 (Node) timing runner for the **svm-wasmjit** cross-engine row: it times SVM IR **JIT-compiled
// to wasm** (the `svm-wasmjit` emitter) running the same LLVM-frontend IR kernel as `bench.mjs`'s
// svm-bytecode-wasm row — so the two rows, side by side, are the interpreter-in-wasm vs the
// JIT-in-wasm on identical IR (see BROWSER.md § "wasm-JIT tier"). Unlike the interpreter row, the
// module is **compiled once** (like the native svm-jit row) and only the emitted `f{func}` calls are
// timed; a kernel outside the JIT's integer subset exits 4 so the driver marks the row n/a.
//
//   node bench_jit.mjs <svm_browser.wasm> <kernel.svmbc> <func> <sp> <small> <large>
//
// stdout: two lines — "<per_iter_ns>" then "<result@small>" — matching bench.mjs's parse, the second
// line a correctness anchor the Rust driver cross-checks against native bytecode (a mismatch is a
// loud MISCOMPILE, the emitter's differential in the bench itself).
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

const [wasmPath, kernelPath, funcS, spS, smallS, largeS] = process.argv.slice(2);
if (!largeS) {
  console.error('usage: node bench_jit.mjs <svm_browser.wasm> <kernel.svmbc> <func> <sp> <small> <large>');
  process.exit(2);
}
const func = Number(funcS);
const sp = BigInt(spS);
const small = Number(smallS), large = Number(largeS);

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, {})).exports;
if (ex.memory === undefined) {
  console.error('bench_jit expects the plain cdylib (exported, non-shared memory)');
  process.exit(2);
}
const u8 = () => new Uint8Array(ex.memory.buffer);

// Emit the kernel to a NON-SHARED wasm module (shared=0), rooted at `func` as the JIT entry.
const bytes = readFileSync(kernelPath);
const mptr = ex.svm_alloc(bytes.length);
u8().set(bytes, Number(mptr));
if (ex.svm_wasmjit_compile_full(mptr, bytes.length, func, 0) !== 1) {
  ex.svm_dealloc(mptr, bytes.length);
  console.error(`kernel func ${func} not JIT-eligible (outside the integer subset)`);
  process.exit(4); // the driver marks this row n/a
}
ex.svm_dealloc(mptr, bytes.length);
const wptr = Number(ex.svm_wasmjit_ptr()), wlen = ex.svm_wasmjit_len();
const emitted = u8().slice(wptr, wptr + wlen);

// Instantiate the emitted module against the cdylib's OWN linear memory (so an svm_alloc'ed window
// + env cell are addressable in both). `call_interp` runs an interp leaf via the cdylib; a nonzero
// return is a trap (thrown). `env.trap` records the SVM trap code.
let lastTrap = 0;
const emod = await WebAssembly.compile(emitted);
const einst = await WebAssembly.instantiate(emod, {
  env: {
    memory: ex.memory,
    trap: (code) => { lastTrap = code; },
    call_interp: (f, argsPtr) => { if (ex.svm_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
  },
});
const entry = einst.exports[`f${func}`];
if (!entry) { console.error(`emitted module has no f${func}`); process.exit(3); }

// One window big enough for any kernel's declared memory (chase_rand is 4 MiB), and the env cell
// (fuel + cross-tier scratch). A huge positive fuel budget so the per-dispatch debit never runs out
// across all timed reps (the counter is not reset between calls).
const WIN_BYTES = 1 << 24; // 16 MiB
const win = Number(ex.svm_alloc(WIN_BYTES));
ex.svm_wasmjit_init_window(win, WIN_BYTES); // lay the module's data segments into the window
const env = Number(ex.svm_alloc(ex.svm_wasmjit_env_bytes()));
new DataView(ex.memory.buffer).setBigInt64(env, (1n << 61n), true);

// The frontend ABI is `func(sp: i64, n: i32)`; the emitted entry prepends (win, env). `n` is i32.
const runN = (n) => entry(win, env, sp, n);

const REPS = 10;
const best = (n) => {
  runN(n); // warm up (V8 tiers the call site up to TurboFan)
  let b = Infinity;
  for (let r = 0; r < REPS; r++) {
    const t = performance.now();
    runN(n);
    const e = performance.now();
    if (e - t < b) b = e - t;
  }
  return b * 1e6; // ms -> ns
};

let perIter, resultSmall;
try {
  perIter = (best(large) - best(small)) / (large - small);
  resultSmall = BigInt.asIntN(64, BigInt(runN(small)));
} catch (e) {
  console.error(`emitted kernel trapped (lastTrap=${lastTrap}): ${e.message ?? e}`);
  process.exit(3);
}
console.log(perIter.toFixed(6));
console.log(resultSmall.toString());
