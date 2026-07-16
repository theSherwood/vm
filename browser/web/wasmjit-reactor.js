// Browser wasm-JIT **reactor** driver (BROWSER.md § "wasm-JIT tier", slice 5d). The interpreter
// reactor (`svm_onramp_*`) runs the guest's `tick` on the bytecode engine each frame; this runs the
// guest's **whole `tick` on emitted wasm**, near-natively. The cdylib opens the reactor
// (`svm_onramp_jit_open_fs`: decode, enlarge the window, run `_start` on the interpreter, emit the
// whole `tick`), and this module compiles the emitted bytes and instantiates them against the cdylib's
// **own** (shared) linear memory — so the emitted code addresses the same window the interpreter
// seeded. Each frame the page calls the exported `f{tick}(win, env, sp)` directly (no Rust frame in the
// path, so a guest trap surfaces as a catchable RuntimeError); non-emitted (cross-tier) helpers relay
// their `env.call_interp` back to `svm_onramp_jit_call_interp`, which runs them on the interpreter over
// the same window with the powerbox (so `display`/`keyboard`/`fs`/`exit` resolve). Only ~4 such bounces
// happen per Doom frame — the hot render path is all emitted — so the JS↔Rust boundary is not a factor.

const DEFAULT_FUEL = 1n << 52n; // huge per-frame dispatcher-fuel budget (only a runaway trips it)

// Open a JIT reactor over `moduleBytes`, optionally granting an `fs`-served `wad` under `wadName` (pass
// `wad = null` for guests that need no served file — bounce/life/mandelzoom). `ex`/`memory` are the
// cdylib exports and its shared linear memory. Returns `{ frame(), close() }` or throws on open/emit
// failure (the caller falls back to the interpreter reactor). `frame()` runs one `tick` and returns the
// status (0 = keep going, 5 = the guest exited, else a trap) after stashing the presented frame into
// the `svm_framebuffer_*` slots.
export async function openJitReactor(ex, memory, moduleBytes, wadName, wad) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Hand the module (+ optional WAD) to the cdylib: it decodes, runs `_start`, and emits the whole
  // `tick`. Without a WAD, the plain `svm_onramp_jit_open` (no `fs` cap) is the open path.
  let opened;
  if (wad) {
    const modP = Number(ex.svm_alloc(moduleBytes.length));
    const nameBytes = new TextEncoder().encode(wadName);
    const nameP = Number(ex.svm_alloc(nameBytes.length));
    const wadP = Number(ex.svm_alloc(wad.length));
    {
      const v = u8();
      v.set(moduleBytes, modP);
      v.set(nameBytes, nameP);
      v.set(wad, wadP);
    }
    opened = ex.svm_onramp_jit_open_fs(modP, moduleBytes.length, nameP, nameBytes.length, wadP, wad.length);
    ex.svm_dealloc(modP, moduleBytes.length);
    ex.svm_dealloc(nameP, nameBytes.length);
    ex.svm_dealloc(wadP, wad.length);
  } else {
    const modP = Number(ex.svm_alloc(moduleBytes.length));
    u8().set(moduleBytes, modP);
    opened = ex.svm_onramp_jit_open(modP, moduleBytes.length);
    ex.svm_dealloc(modP, moduleBytes.length);
  }
  if (opened !== 0) {
    throw new Error(`JIT reactor open failed: status ${ex.svm_status()} (2=tick not emittable, 3=trap)`);
  }

  // Copy the emitted `tick` wasm out of linear memory (a later svm_alloc could move the stash), and
  // read the window base / entry sp / tick index / env-cell size the emitted ABI needs.
  const wptr = Number(ex.svm_onramp_jit_wasm_ptr());
  const wlen = ex.svm_onramp_jit_wasm_len();
  const emitted = u8().slice(wptr, wptr + wlen);
  const win = Number(ex.svm_onramp_jit_win_ptr());
  const entrySp = ex.svm_onramp_jit_entry_sp();
  const tick = ex.svm_onramp_jit_tick();
  const envBytes = ex.svm_onramp_jit_env_bytes();

  // `env.call_interp` relays each cross-tier call to the cdylib; a nonzero status (exit/trap) is thrown
  // so it unwinds the emitted `f{tick}` — the page reads the exact status back after the frame call.
  let lastCross = 0;
  const module = await WebAssembly.compile(emitted);
  const instance = await WebAssembly.instantiate(module, {
    env: {
      memory,
      trap: (_code) => {},
      call_interp: (func, argsPtr) => {
        const s = ex.svm_onramp_jit_call_interp(func, argsPtr);
        if (s !== 0) { lastCross = s; throw new Error('cross-tier stop'); }
      },
    },
  });
  const fTick = instance.exports[`f${tick}`];
  if (typeof fTick !== 'function') throw new Error(`emitted module has no f${tick} export`);

  // The env cell (fuel counter + cross-tier scratch) lives in linear memory; refill fuel each frame.
  const env = Number(ex.svm_alloc(envBytes));

  return {
    // Run one frame. Returns 0 (keep going), 5 (guest exited), or a trap status.
    frame() {
      new DataView(memory.buffer).setBigInt64(env, DEFAULT_FUEL, true);
      lastCross = 0;
      let status = 0;
      try {
        fTick(win, env, entrySp);
      } catch (e) {
        // A cross-tier exit/trap threw through the emitted tick, or the emitted code trapped directly
        // (RuntimeError). `lastCross` carries the cdylib's status for a relayed stop; a bare
        // RuntimeError with no relayed status is an emitted-code trap.
        status = lastCross !== 0 ? lastCross : 3;
      }
      // Stash whatever frame the tick presented through `display` into the framebuffer slots.
      ex.svm_onramp_jit_present();
      return status;
    },
    close() {
      ex.svm_dealloc(env, envBytes);
      ex.svm_onramp_jit_close();
    },
  };
}
