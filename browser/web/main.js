// THREADS/BROWSER step 4c-wasm in a REAL browser — the page-side orchestrator. Proves two things run
// in an actual browser (Chromium via Playwright), not just Node:
//   1. the **powerbox** (`svm_run_pb`) — a guest writes to stdout, single-threaded on the page;
//   2. genuine **parallelism** — one guest's `thread.spawn`ed vCPUs run on separate Web Workers over a
//      shared `WebAssembly.Memory` (only available because the server sent COOP/COEP), synchronising
//      via `Atomics` → the 8-vCPU counter kernel returns 4000.
// The Worker orchestration itself lives in `par.js` (shared with the playground, `play.js`); this
// page mirrors `threads-spawn.mjs` exactly.

import { fetchBytes, loadEngine, makeRunner, readParStdout } from '/web/par.js';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += m + '\n'; };
const set = (id, status, text) => { const e = $(id); e.dataset.status = status; e.textContent = text; };

async function main() {
  set('isolated', String(self.crossOriginIsolated), `crossOriginIsolated: ${self.crossOriginIsolated}`);
  if (!self.crossOriginIsolated) {
    set('powerbox', 'fail', 'powerbox: skipped (no cross-origin isolation)');
    set('threads', 'fail', 'threads: skipped (no SharedArrayBuffer)');
    return;
  }

  let eng;
  try {
    eng = await loadEngine();
  } catch (e) {
    set('threads', 'fail', `threads: ${e.message}`);
    return;
  }
  const { memory, ex } = eng;
  const u8 = () => new Uint8Array(memory.buffer);
  log(`module loaded; shared=${memory.buffer instanceof SharedArrayBuffer}; TLS ${ex.__tls_size.value}B`);

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

  const run = makeRunner(eng);
  const runPath = async (guestPath, opts = {}) => {
    const o = { ...opts };
    if (o.unitPath) {
      o.unit = await fetchBytes(o.unitPath);
      delete o.unitPath;
    }
    return run(await fetchBytes(guestPath), o);
  };

  // --- 2) one guest's vCPUs across real Web Workers ----------------------------------------------
  try {
    const t0 = performance.now();
    const { value, started } = await runPath('/corpus/threads.svmbc');
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
    const { value, started } = await runPath('/corpus/threads_jit_install.svmbc', { jit: true });
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
    const a = await runPath('/corpus/threads_inst.svmbc', opt);
    const b = await runPath('/corpus/threads_inst_nested.svmbc', opt);
    const c = await runPath('/corpus/threads_inst_mod.svmbc',
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
    const { value, started } = await runPath('/corpus/threads_io.svmbc', { io: true });
    const ms = (performance.now() - t0).toFixed(0);
    const out = readParStdout(eng);
    const ok = value === 8n && out === 'tick\n'.repeat(8);
    set('capio', ok ? 'pass' : 'fail',
      `capio: ${started} Workers → counter ${value} (want 8), stdout ${JSON.stringify(out)} ` +
      `(want 8 × "tick\\n") ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`capio → ${value}, stdout ${out.length}B across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('capio', 'fail', `capio: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('threads', 'fail', `fatal: ${e}`); });
