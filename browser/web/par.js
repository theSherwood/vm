// Shared page-side Worker orchestration — one SVM guest's vCPUs across real Web Workers over one
// shared `WebAssembly.Memory`. Extracted from `main.js` (THREADS.md 4c-wasm) so the validation page
// and the playground (`play.js`) drive the exact same machinery. The page creates every Worker (no
// nested Workers) and never blocks (a browser bans main-thread `Atomics.wait`); the Workers do all
// the blocking (`worker.js`).

// Resolved relative to this module's URL (not the document root) so the page works both at the
// origin root (local `serve.mjs`) and under a subpath (GitHub Pages serves a project site at
// `/<repo>/`). The deployed site keeps the same `web/` + `target/…` layout, so `../target/…`
// resolves correctly under either base.
const WASM = new URL('../target/wasm32-unknown-unknown/release/svm_browser.wasm', import.meta.url);
const STACK = 1 << 20, SLOT = 16;
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

export async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

// Compile + instantiate the threads wasm build over a fresh shared memory. Requires cross-origin
// isolation (the caller checks `self.crossOriginIsolated` first for a friendlier message).
export async function loadEngine() {
  const module = await WebAssembly.compile(await fetchBytes(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    throw new Error('not a threads build (no imported memory)');
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  return { module, memory, ex };
}

// Build the runner over a loaded engine. The returned `runAcrossWorkers(guest, opts)` runs one
// guest's `thread.spawn`ed / `instantiate`d vCPUs across real Web Workers and resolves
// `{ value, started }`. `guest` is the **encoded module bytes** (a fetched `.svmbc` or an in-browser
// `svm_parse` product). Options:
//   `jit`     ⇒ build the Rust-side §22 powerbox + reserve the JIT dispatch table;
//   `inst`    ⇒ publish the §14 recipe (root `Instantiator` over the window + the optional granted
//               `unit` module bytes) — the root vCPU builds its powerbox from it;
//   `io`      ⇒ publish the 4d shared I/O powerbox (a `Mutex<Host>` in shared memory every vCPU
//               dispatches `cap.call` through; read stdout back via `svm_par_stdout_*` after);
//   none      ⇒ the recipes are explicitly cleared (`svm_par_powerbox_none`) so a plain compute run
//               isn't seeded by a previous run's recipe;
//   `winSize` sizes the shared window; `signal` (an `AbortSignal`) stops the run: every Worker is
//   terminated and the promise rejects. NOTE a stop tears down Workers mid-run — shared state (the
//   I/O powerbox lock, the live-vCPU counter) may be left unusable; reload the page after a stop.
// Either way the page services no authority: JIT/IO are in-Rust, and a §14 `instantiate` event's
// operands are inert integers relayed into a new Worker.
export function makeRunner({ module, memory, ex }) {
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  return async function runAcrossWorkers(guest, { jit = false, jitCodegen = false, inst = false, io = false, tierup = false, unit = null, winSize = 1 << 16, signal = null } = {}) {
    const gptr = ex.svm_par_alloc(guest.length);
    u8().set(guest, gptr);
    if (jit && ex.svm_par_powerbox(gptr, guest.length) !== 1) throw new Error('svm_par_powerbox failed');
    // §22 real-codegen run: like `jit`, but the host-compiled unit's wasm is emitted + stashed, and a
    // guest `Jit.invoke` runs it on emitted wasm (each Worker instantiates the unit — see worker.js).
    if (jitCodegen && ex.svm_par_powerbox_jit_codegen(gptr, guest.length) !== 1) throw new Error('svm_par_powerbox_jit_codegen failed');
    if (io && ex.svm_par_powerbox_io() !== 1) throw new Error('svm_par_powerbox_io failed');
    if (!jit && !jitCodegen && !io && !inst) ex.svm_par_powerbox_none();
    const prog = (jit || jitCodegen) ? ex.svm_par_compile_jit(gptr, guest.length) : ex.svm_par_compile(gptr, guest.length);
    if (prog === 0) throw new Error('module unsupported on the parallel driver (svm_par_compile null)');
    const win = ex.svm_par_alloc(winSize);
    if (inst) {
      let uptr = 0, ulen = 0;
      if (unit) {
        uptr = ex.svm_par_alloc(unit.length);
        u8().set(unit, uptr);
        ulen = unit.length;
      }
      if (ex.svm_par_powerbox_inst(BigInt(winSize), uptr, ulen) !== 1) {
        throw new Error('svm_par_powerbox_inst failed');
      }
    }

    // wasm-JIT tier-up: a shared i32 cell every Worker atomically bumps on each tier-up, so the
    // caller can prove the seam actually fired (a result match alone can't tell "tiered up" from
    // "silently interpreted"). Read back after the run.
    const tierupCell = tierup ? ex.svm_par_alloc(4) : 0;

    const workers = new Set();
    let started = 0;
    try {
      const value = await new Promise((resolve, reject) => {
        if (signal) {
          if (signal.aborted) return reject(new Error('stopped'));
          signal.addEventListener('abort', () => reject(new Error('stopped')), { once: true });
        }
        const startVcpu = (cfg) => {
          started++;
          const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
          workers.add(w);
          w.onmessage = (e) => {
            const m = e.data;
            if (m.kind === 'spawn') {
              // Plain or §14-confined child: relay the message's cfg verbatim (a confined child's
              // message carries its own win/winSize — the carve — overriding the run defaults).
              const { kind, ...cfg2 } = m;
              startVcpu({ role: 'child', ...cfg2 });
            } else if (m.kind === 'done') {
              resolve(BigInt(m.value));
            } else if (m.kind === 'trap' || m.kind === 'fail') {
              reject(new Error(m.why || 'guest trap'));
            }
          };
          w.onerror = (e) => reject(new Error(e.message || 'worker error'));
          // `tierup` + the guest bytes (kept live at `gptr` for the run) let each Worker JIT-compile
          // the guest locally and run eligible compute regions on the emitted wasm (threads slice).
          w.postMessage({ module, memory, prog, win, winSize, tierup, jitCodegen, gptr, glen: guest.length, tierupCell, ...cfg });
        };
        // The root vCPU runs on its own Worker (the page can't Atomics.wait).
        const rootSlot = ex.svm_par_alloc(SLOT);
        const rootStackTop = ex.svm_par_alloc(STACK) + STACK;
        const rootTlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
        startVcpu({ role: 'root', func: 0, slot: rootSlot, stackTop: rootStackTop, tlsBase: rootTlsBase });
      });
      const tierups = tierup ? Atomics.load(new Int32Array(memory.buffer), tierupCell >> 2) : 0;
      return { value, started, tierups };
    } finally {
      for (const w of workers) w.terminate();
    }
  };
}

// Read back the accumulated stdout of the last 4d I/O run (empty string when no I/O powerbox ran).
// `slice` (not `subarray`) copies out of the SharedArrayBuffer — TextDecoder rejects shared views.
export function readParStdout({ memory, ex }) {
  const len = ex.svm_par_stdout_len();
  const u8 = new Uint8Array(memory.buffer);
  return new TextDecoder().decode(u8.slice(ex.svm_par_stdout_ptr(), ex.svm_par_stdout_ptr() + len));
}
