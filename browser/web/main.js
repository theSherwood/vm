// THREADS/BROWSER step 4c-wasm in a REAL browser — the page-side orchestrator. Proves two things run
// in an actual browser (Chromium via Playwright), not just Node:
//   1. the **powerbox** (`svm_run_pb`) — a guest writes to stdout, single-threaded on the page;
//   2. genuine **parallelism** — one guest's `thread.spawn`ed vCPUs run on separate Web Workers over a
//      shared `WebAssembly.Memory` (only available because the server sent COOP/COEP), synchronising
//      via `Atomics` → the 8-vCPU counter kernel returns 4000.
// The page creates every Worker (no nested Workers) and never blocks (a browser bans main-thread
// `Atomics.wait`); the Workers do all the blocking. This mirrors `threads-spawn.mjs` exactly.

const WASM = '/target/wasm32-unknown-unknown/release/svm_browser.wasm';
const STACK = 1 << 20, SLOT = 16;
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += m + '\n'; };
const set = (id, status, text) => { const e = $(id); e.dataset.status = status; e.textContent = text; };

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function main() {
  set('isolated', String(self.crossOriginIsolated), `crossOriginIsolated: ${self.crossOriginIsolated}`);
  if (!self.crossOriginIsolated) {
    set('powerbox', 'fail', 'powerbox: skipped (no cross-origin isolation)');
    set('threads', 'fail', 'threads: skipped (no SharedArrayBuffer)');
    return;
  }

  const module = await WebAssembly.compile(await fetchBytes(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    set('threads', 'fail', 'threads: not a threads build (no imported memory)');
    return;
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;
  log(`module loaded; shared=${memory.buffer instanceof SharedArrayBuffer}; TLS ${tlsSize}B`);

  // --- 1) powerbox smoke (single-threaded, on the page) -------------------------------------------
  try {
    const pb = await fetchBytes('/corpus/pb_hello.svmbc');
    const p = ex.svm_alloc(pb.length);
    u8().set(pb, p);
    ex.svm_run_pb(p, pb.length, 0, 0);
    const status = ex.svm_status();
    // `slice` (not `subarray`) copies out of the SharedArrayBuffer — TextDecoder rejects shared views.
    const out = new TextDecoder().decode(u8().slice(ex.svm_stdout_ptr(), ex.svm_stdout_ptr() + ex.svm_stdout_len()));
    const ok = status === 0 && out === 'hello, powerbox!\n';
    set('powerbox', ok ? 'pass' : 'fail', `powerbox: status=${status} stdout=${JSON.stringify(out)} ${ok ? 'PASS' : 'FAIL'}`);
    log(`powerbox → ${JSON.stringify(out)}`);
  } catch (e) {
    set('powerbox', 'fail', `powerbox: error ${e}`);
  }

  // Run one guest's `thread.spawn`ed vCPUs across real Web Workers over the one shared window; returns
  // `{ value, started }`. `jit` ⇒ build the Rust-side shared powerbox + reserve the JIT dispatch table
  // (the worker loop is identical — §22 JIT is serviced in-Rust, so the page services no new events).
  async function runAcrossWorkers(guestPath, jit) {
    const guest = await fetchBytes(guestPath);
    const gptr = ex.svm_par_alloc(guest.length);
    u8().set(guest, gptr);
    if (jit && ex.svm_par_powerbox(gptr, guest.length) !== 1) throw new Error('svm_par_powerbox failed');
    const prog = jit ? ex.svm_par_compile_jit(gptr, guest.length) : ex.svm_par_compile(gptr, guest.length);
    if (prog === 0) throw new Error('svm_par_compile null');
    const winSize = 1 << 16;
    const win = ex.svm_par_alloc(winSize);

    const workers = new Set();
    let started = 0;
    const done = new Promise((resolve, reject) => {
      const startVcpu = (cfg) => {
        started++;
        const w = new Worker('/web/worker.js', { type: 'module' });
        workers.add(w);
        w.onmessage = (e) => {
          const m = e.data;
          if (m.kind === 'spawn') {
            startVcpu({ role: 'child', func: m.func, sp: m.sp, arg: m.arg, slot: m.slot, stackTop: m.stackTop, tlsBase: m.tlsBase });
          } else if (m.kind === 'done') {
            resolve(BigInt(m.value));
          } else if (m.kind === 'trap' || m.kind === 'fail') {
            reject(new Error(m.why || 'guest trap'));
          }
        };
        w.onerror = (e) => reject(new Error(e.message || 'worker error'));
        w.postMessage({ module, memory, prog, win, winSize, ...cfg });
      };
      // The root vCPU runs on its own Worker (the page can't Atomics.wait).
      const rootSlot = ex.svm_par_alloc(SLOT);
      const rootStackTop = ex.svm_par_alloc(STACK) + STACK;
      const rootTlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      startVcpu({ role: 'root', func: 0, slot: rootSlot, stackTop: rootStackTop, tlsBase: rootTlsBase });
    });
    const value = await done;
    for (const w of workers) w.terminate();
    return { value, started };
  }

  // --- 2) one guest's vCPUs across real Web Workers ----------------------------------------------
  try {
    const t0 = performance.now();
    const { value, started } = await runAcrossWorkers('/corpus/threads.svmbc', false);
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 4000n;
    set('threads', ok ? 'pass' : 'fail',
      `threads: ${started} Workers (1 root + ${started - 1} spawned) → ${value} (want 4000) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`threads → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('threads', 'fail', `threads: error ${e}`);
  }

  // --- 3) §22 guest-JIT across real Web Workers (THREADS.md 4c-domain C2) -------------------------
  // Each worker vCPU `install`s a host-compiled unit into the **shared** Domain and `call_indirect`s
  // its own raced slot — `service(6,7) = 142`, folded to 8 × 142 = 1136. The powerbox is Rust-side
  // (a leaked `Host` in shared memory); JIT is serviced inside `svm_par_run`, so no new page glue.
  try {
    const t0 = performance.now();
    const { value, started } = await runAcrossWorkers('/corpus/threads_jit_install.svmbc', true);
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 1136n;
    set('jit', ok ? 'pass' : 'fail',
      `jit: ${started} Workers each install+call a unit on the shared Domain → ${value} (want 1136) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jit → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('jit', 'fail', `jit: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('threads', 'fail', `fatal: ${e}`); });
