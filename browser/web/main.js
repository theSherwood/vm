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

  // Run one guest's `thread.spawn`ed / `instantiate`d vCPUs across real Web Workers over the one
  // shared window; returns `{ value, started }`. Options: `jit` ⇒ build the Rust-side §22 powerbox +
  // reserve the JIT dispatch table; `inst` ⇒ publish the §14 recipe (root `Instantiator` over the
  // window + the optional granted `unitPath` module) — the root vCPU builds its powerbox from it;
  // `winSize` sizes the window (the §14 kernels declare 1 MiB so their 64 KiB carves stay
  // wasm-page-aligned). Either way the page services no authority: JIT is in-Rust, and a §14
  // `instantiate` event's operands are inert integers relayed into a new Worker.
  async function runAcrossWorkers(guestPath, { jit = false, inst = false, io = false, unitPath = null, winSize = 1 << 16 } = {}) {
    const guest = await fetchBytes(guestPath);
    const gptr = ex.svm_par_alloc(guest.length);
    u8().set(guest, gptr);
    if (jit && ex.svm_par_powerbox(gptr, guest.length) !== 1) throw new Error('svm_par_powerbox failed');
    // 4d: the run's shared I/O powerbox — a Mutex<Host> in shared memory every vCPU dispatches
    // cap.call through; worker host I/O happens in-Rust, the page just reads stdout back after.
    if (io && ex.svm_par_powerbox_io() !== 1) throw new Error('svm_par_powerbox_io failed');
    const prog = jit ? ex.svm_par_compile_jit(gptr, guest.length) : ex.svm_par_compile(gptr, guest.length);
    if (prog === 0) throw new Error('svm_par_compile null');
    const win = ex.svm_par_alloc(winSize);
    if (inst) {
      let uptr = 0, ulen = 0;
      if (unitPath) {
        const unit = await fetchBytes(unitPath);
        uptr = ex.svm_par_alloc(unit.length);
        u8().set(unit, uptr);
        ulen = unit.length;
      }
      if (ex.svm_par_powerbox_inst(BigInt(winSize), uptr, ulen) !== 1) {
        throw new Error('svm_par_powerbox_inst failed');
      }
    }

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
    const { value, started } = await runAcrossWorkers('/corpus/threads.svmbc');
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
    const { value, started } = await runAcrossWorkers('/corpus/threads_jit_install.svmbc', { jit: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 1136n;
    set('jit', ok ? 'pass' : 'fail',
      `jit: ${started} Workers each install+call a unit on the shared Domain → ${value} (want 1136) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jit → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('jit', 'fail', `jit: error ${e}`);
  }

  // --- 4) §14 confined executor children across real Web Workers (4c-domain §14-D2) ---------------
  // Three sub-proofs, each a fresh run over a 1 MiB window with 64 KiB carves:
  //   a. instantiate:  8 confined children (each its own Worker + attenuated powerbox) → 8 × 5 = 40;
  //   b. nested:       each child instantiates a grandchild over its whole carve — VM-in-VM-in-VM
  //                    across THREE Worker generations → 8 × 9 = 72;
  //   c. module:       `instantiate_module` a granted module 8× (compile + push to the shared source
  //                    + data segments materialized, all crossing Workers) → 8 × 75 = 600.
  try {
    const opt = { inst: true, winSize: 1 << 20 };
    const t0 = performance.now();
    const a = await runAcrossWorkers('/corpus/threads_inst.svmbc', opt);
    const b = await runAcrossWorkers('/corpus/threads_inst_nested.svmbc', opt);
    const c = await runAcrossWorkers('/corpus/threads_inst_mod.svmbc',
      { ...opt, unitPath: '/corpus/threads_inst_unit.svmbc' });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = a.value === 40n && b.value === 72n && c.value === 600n;
    set('inst', ok ? 'pass' : 'fail',
      `inst: confined children → ${a.value} (want 40) · nested ×${b.started} Workers → ${b.value} ` +
      `(want 72) · module → ${c.value} (want 600) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`inst → ${a.value}/${b.value}/${c.value} (nested spanned ${b.started} Workers) in ${ms}ms`);
  } catch (e) {
    set('inst', 'fail', `inst: error ${e}`);
  }

  // --- 5) host I/O from worker vCPUs across real Web Workers (THREADS.md 4d) ----------------------
  // 8 worker vCPUs each `cap.call`-write "tick\n" to the run's ONE shared powerbox (a Mutex<Host> in
  // shared memory — dispatch is in-Rust under the lock, no JS in the loop) and bump a shared counter.
  // Result 8 and stdout "tick\n"×8 are schedule-independent; the page reads stdout back afterward.
  try {
    const t0 = performance.now();
    const { value, started } = await runAcrossWorkers('/corpus/threads_io.svmbc', { io: true });
    const ms = (performance.now() - t0).toFixed(0);
    const len = ex.svm_par_stdout_len();
    // `slice` (not `subarray`) copies out of the SharedArrayBuffer — TextDecoder rejects shared views.
    const out = new TextDecoder().decode(u8().slice(ex.svm_par_stdout_ptr(), ex.svm_par_stdout_ptr() + len));
    const ok = value === 8n && out === 'tick\n'.repeat(8);
    set('capio', ok ? 'pass' : 'fail',
      `capio: ${started} Workers → counter ${value} (want 8), stdout ${JSON.stringify(out)} ` +
      `(want 8 × "tick\\n") ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`capio → ${value}, stdout ${len}B across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('capio', 'fail', `capio: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('threads', 'fail', `fatal: ${e}`); });
