// Browser wasm-JIT tier — the JS linker (BROWSER.md § "wasm-JIT tier", slice 2). Given the SVM
// browser cdylib instance and an encoded SVM module, this: (1) asks the cdylib to emit a
// WebAssembly module for it (`svm_wasmjit_compile`), (2) instantiates the emitted bytes against the
// **cdylib's own linear memory** (so an `svm_alloc`ed window + env cell are addressable in both),
// and (3) calls the exported `f{i}(win, env, ...args)` **directly** — no Rust frame in the call
// path, so a guest trap's `unreachable` surfaces here as a catchable WebAssembly.RuntimeError (the
// slice-1 differential model, preserved). Compute-only, single-threaded: threads + a shared window
// are slice 4; tiering/interp-fallback for non-eligible modules is slice 3.
//
// `ex` is the cdylib's exports (must be the NON-threads build, whose `memory` is exported here;
// the threads build imports it and the caller passes that object instead — see `memoryOf`).

// Emitter trap codes — must match browser/src/lib.rs WASMJIT_TRAP_* (re-exported from svm-wasmjit).
export const TRAP_OUT_OF_FUEL = 1;
export const TRAP_MEMORY_FAULT = 2;

// Default per-call dispatcher fuel (debited once per block dispatch — a coarse §5 bound; huge so
// only a genuine runaway trips it).
const DEFAULT_FUEL = 1_000_000_000n;

// The cdylib's linear memory: exported by the plain build, or passed in by the threads build.
function memoryOf(ex, memory) {
  return memory ?? ex.memory;
}

// Compile `moduleBytes` (an encoded SVM module, already in a JS Uint8Array) to a callable JIT
// function, or return null if the module is not JIT-eligible (the host falls back to the
// interpreter). `ex` is the cdylib exports; `memory` overrides `ex.memory` for the threads build.
// Returns `{ call(args, { fuel, winSize }) }` where `call` runs the emitted `f0` and returns
// `{ value: BigInt }` or throws `{ trap: 'out_of_fuel' | 'memory_fault' | 'wasm' }`.
export async function compileJit(ex, moduleBytes, { memory = null } = {}) {
  const mem = memoryOf(ex, memory);
  // Hand the module to the cdylib emitter (reuse an svm_alloc buffer).
  const mptr = ex.svm_alloc(moduleBytes.length);
  new Uint8Array(mem.buffer).set(moduleBytes, Number(mptr));
  const ok = ex.svm_wasmjit_compile(mptr, moduleBytes.length);
  ex.svm_dealloc(mptr, moduleBytes.length);
  if (ok !== 1) return null; // not JIT-eligible → caller keeps the interpreter tier

  // Copy the emitted wasm out of linear memory (a later svm_alloc could move the stash).
  const wptr = Number(ex.svm_wasmjit_ptr());
  const wlen = ex.svm_wasmjit_len();
  const wasm = new Uint8Array(mem.buffer).slice(wptr, wptr + wlen);

  // The env.trap import records the last trap code; a trap is followed by `unreachable`, so the
  // recorded code + the caught RuntimeError together classify the fault.
  let lastTrap = 0;
  const module = await WebAssembly.compile(wasm);
  const instance = await WebAssembly.instantiate(module, {
    env: { memory: mem, trap: (code) => { lastTrap = code; } },
  });
  const f0 = instance.exports.f0;

  return {
    // `args` must already be correctly typed for the SVM entry's params: a JS Number for an i32
    // param, a BigInt for an i64 param (wasm rejects a mismatched JS type at the call).
    call(args = [], { fuel = DEFAULT_FUEL, winSize = 1 << 16 } = {}) {
      // Window + env cell in the cdylib's memory (both addressable by the emitted module).
      const win = Number(ex.svm_alloc(winSize));
      const env = Number(ex.svm_alloc(8));
      new DataView(mem.buffer).setBigInt64(env, BigInt(fuel), true);
      lastTrap = 0;
      try {
        const value = f0(win, env, ...args);
        return { value: BigInt(value) };
      } catch (e) {
        if (!(e instanceof WebAssembly.RuntimeError)) throw e;
        const trap =
          lastTrap === TRAP_OUT_OF_FUEL ? 'out_of_fuel'
          : lastTrap === TRAP_MEMORY_FAULT ? 'memory_fault'
          : 'wasm'; // div0 / overflow / guest `unreachable` — wasm's own trap
        throw { trap, wasmError: e };
      } finally {
        ex.svm_dealloc(win, winSize);
        ex.svm_dealloc(env, 8);
      }
    },
  };
}
