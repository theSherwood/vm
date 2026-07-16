// Single-shot **wasm-JIT module runner** — the run-to-completion twin of `wasmjit-reactor.js` (which
// drives a `tick` per frame). An on-ramp module's whole program is func 0 (`_start`); the cdylib emits
// it and this module compiles + runs `f0(win, env, ...slots)` **once** against the cdylib's own shared
// linear memory, servicing the ~7% cross-tier helpers (Lua/SQLite) through `env.call_interp`. After the
// run, `svm_onramp_jit_run_finish` captures stdout/stderr/exit into the shared slots — read them via the
// usual `svm_stdout_*` / `svm_exit_code` accessors, exactly like the interpreter `svm_run_onramp` path.
//
// Returns the run status (0 = returned, 5 = exited) or throws if `_start` isn't emittable (the caller
// falls back to `svm_run_onramp`). Synchronous guest work (like the interpreter path); only the initial
// `WebAssembly.compile` of the emitted module is async.
export async function runJitModule(ex, memory, moduleBytes, stdinBytes) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Hand the module (+ optional stdin) to the cdylib: decode, outline, grant powerbox, emit `_start`.
  const modP = Number(ex.svm_alloc(moduleBytes.length));
  u8().set(moduleBytes, modP);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.svm_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  const opened = ex.svm_onramp_jit_run_open(modP, moduleBytes.length, stdinP, stdinLen);
  ex.svm_dealloc(modP, moduleBytes.length);
  if (stdinP) ex.svm_dealloc(stdinP, stdinLen);
  if (opened !== 0) {
    throw new Error(`JIT module open failed: status ${ex.svm_status()} (2 = _start not emittable)`);
  }

  // Copy the emitted bytes out (a later svm_alloc could move the stash), read the window base + the
  // powerbox handle slots `_start` takes as params, and the env-cell size.
  const wptr = Number(ex.svm_onramp_jit_run_wasm_ptr());
  const wlen = ex.svm_onramp_jit_run_wasm_len();
  const emitted = u8().slice(wptr, wptr + wlen);
  const win = Number(ex.svm_onramp_jit_run_win_ptr());
  const envBytes = ex.svm_onramp_jit_run_env_bytes();
  const slots = [];
  for (let i = 0, n = ex.svm_onramp_jit_run_slot_count(); i < n; i++) {
    slots.push(ex.svm_onramp_jit_run_slot(i));
  }

  // `env.call_interp` relays each cross-tier call to the cdylib; a nonzero status (exit/trap) throws to
  // unwind the emitted `f0` (the browser's JS import model — `Exit` and real traps both caught below).
  const module = await WebAssembly.compile(emitted);
  const instance = await WebAssembly.instantiate(module, {
    env: {
      memory,
      trap: () => {},
      call_interp: (func, argsPtr) => {
        if (ex.svm_onramp_jit_run_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier stop');
      },
    },
  });
  const f0 = instance.exports.f0;
  if (typeof f0 !== 'function') {
    ex.svm_onramp_jit_run_close();
    throw new Error('emitted module has no f0 export');
  }

  const env = Number(ex.svm_alloc(envBytes));
  new DataView(memory.buffer).setBigInt64(env, 1n << 60n, true); // huge dispatcher-fuel budget
  try {
    // f0(win, env, ...cap-handle slots) — runs `_start` (→ main) to completion on emitted wasm.
    f0(win, env, ...slots);
  } catch {
    // A cross-tier `exit`/trap unwound `f0` (expected for a guest that calls exit); the finish status
    // and `svm_onramp_jit_run_trap_len` distinguish a clean exit from a real trap.
  }
  ex.svm_dealloc(env, envBytes);
  const status = ex.svm_onramp_jit_run_finish(); // capture stdout/stderr/exit into the shared slots
  ex.svm_onramp_jit_run_close();
  return status;
}
