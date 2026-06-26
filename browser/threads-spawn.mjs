// THREADS.md step 4c-wasm — the real thing: **one** guest's `thread.spawn`ed vCPUs run on **separate
// Web Workers** (here Node worker_threads, the same SharedArrayBuffer + Atomics primitives a browser
// uses) over the **one** shared linear-memory window. Each Worker runs one vCPU through the engine's
// resumable `Vcpu` API (svm_par_run → an event → the host services it → deliver → run again); the host
// services the events with genuine cross-Worker primitives:
//   * thread.spawn  → ask main to start a new Worker for the child vCPU;
//   * thread.join   → Atomics.wait on the child's completion slot, then read its result;
//   * memory.wait   → Atomics.wait on the futex word in the window;
//   * memory.notify → Atomics.notify on it.
// So this is genuinely parallel (N vCPUs, N OS threads, one shared memory). The native
// `bytecode_vcpu_orchestration.rs` test is its differential oracle. Main creates every Worker (no
// nested Workers); each vCPU runs on a Worker (never the main thread, which can't Atomics.wait).
//
// Usage:  node threads-spawn.mjs <module.wasm> [guest.svmbc] [expected]
import { readFileSync } from 'node:fs';
import { Worker, isMainThread, workerData, parentPort } from 'node:worker_threads';

const WASM = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const GUEST = process.argv[3] ?? 'corpus/threads.svmbc';
const EXPECT = BigInt(process.argv[4] ?? 4000);

const STACK = 1 << 20; // per-Worker stack
const SLOT = 16; // completion slot: [done:i32 @0][result:i64 @8]
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

// Event codes (must match browser/src/lib.rs PAR_*).
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5;

// ---- a single vCPU on this Worker ---------------------------------------------------------------
async function worker() {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase } = workerData;
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  const i32 = new Int32Array(memory.buffer);
  const i64 = new BigInt64Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  const v = role === 'root'
    ? ex.svm_par_root(prog, win, winSize, func)
    : ex.svm_par_child(prog, win, winSize, func, BigInt(sp), BigInt(arg));
  if (v === 0) { parentPort.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  const handles = []; // local spawn handle (index) → child completion slot ptr

  for (;;) {
    const ev = ex.svm_par_run(v);
    if (ev === DONE) {
      const value = ex.svm_par_ev_a(v); // i64 → BigInt
      i64[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32, slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32, slot >> 2); // ...and wake a joiner
      if (role === 'root') parentPort.postMessage({ kind: 'done', value: value.toString() });
      ex.svm_par_free(v);
      return;
    }
    if (ev === TRAP) {
      Atomics.store(i32, slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32, slot >> 2);
      if (role === 'root') parentPort.postMessage({ kind: 'trap' });
      ex.svm_par_free(v);
      return;
    }
    if (ev === SPAWN) {
      const cfunc = Number(ex.svm_par_ev_a(v)), csp = ex.svm_par_ev_b(v), carg = ex.svm_par_ev_c(v);
      // Allocate the child's completion slot + stack + TLS (shared, thread-safe allocator), then ask
      // main to start a Worker for it. We continue immediately with the handle (the child runs async).
      const cslot = ex.svm_par_alloc(SLOT);
      const cstackTop = ex.svm_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      parentPort.postMessage({
        kind: 'spawn', func: cfunc, sp: csp.toString(), arg: carg.toString(),
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.svm_par_deliver_handle(v, handle);
      continue;
    }
    if (ev === JOIN) {
      const handle = Number(ex.svm_par_ev_a(v));
      const cslot = handles[handle];
      Atomics.wait(i32, cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32, cslot >> 2) === 2;
      const result = i64[(cslot + 8) >> 3];
      ex.svm_par_deliver_join(v, result, trapped ? 1 : 0);
      continue;
    }
    if (ev === WAIT) {
      const addr = Number(ex.svm_par_ev_a(v)), expected = Number(BigInt.asIntN(32, ex.svm_par_ev_b(v)));
      const timeoutNs = ex.svm_par_ev_d(v);
      const idx = (win + addr) >> 2;
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(i32, idx, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.svm_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (ev === NOTIFY) {
      const addr = Number(ex.svm_par_ev_a(v)), count = Number(ex.svm_par_ev_b(v));
      const woke = Atomics.notify(i32, (win + addr) >> 2, count);
      ex.svm_par_deliver_code(v, woke);
      continue;
    }
  }
}

// ---- main: compile, carve the window, start the root Worker, fan out child Workers on request ----
async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    console.log('FAIL: not a threads build (module does not import a shared memory)');
    process.exit(1);
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // Compile the guest once → a program pointer shared (read-only) by every Worker.
  const guest = readFileSync(GUEST);
  const gptr = ex.svm_par_alloc(guest.length);
  u8().set(guest, gptr);
  // §22-JIT mode (SVM_JIT=1): build the Rust-side shared powerbox (sets a process-wide static visible
  // to every Worker) and reserve the dispatch table. The worker loop below is unchanged — JIT events
  // are serviced entirely in-Rust against the shared powerbox + Domain (the host never sees them).
  const jitMode = process.env.SVM_JIT === '1';
  if (jitMode && ex.svm_par_powerbox(gptr, guest.length) !== 1) {
    console.log('FAIL: svm_par_powerbox returned 0 (powerbox build failed)'); process.exit(1);
  }
  const prog = jitMode ? ex.svm_par_compile_jit(gptr, guest.length) : ex.svm_par_compile(gptr, guest.length);
  if (prog === 0) { console.log('FAIL: svm_par_compile returned null (decode/unsupported)'); process.exit(1); }

  // The one shared guest window every vCPU runs over.
  const winSize = 1 << 16;
  const win = ex.svm_par_alloc(winSize);

  console.log(`module: ${WASM}  shared=${memory.buffer instanceof SharedArrayBuffer}`);
  console.log(`  prog@0x${prog.toString(16)}  window@0x${win.toString(16)} (${winSize >> 10}KiB)  TLS ${tlsSize}B`);

  const workers = new Set();
  let started = 0;
  const t0 = process.hrtime.bigint();

  const startVcpu = (cfg) => {
    started++;
    const w = new Worker(new URL(import.meta.url), {
      workerData: { module, memory, prog, win, winSize, ...cfg },
    });
    workers.add(w);
    w.on('message', (m) => {
      if (m.kind === 'spawn') {
        // A vCPU asked to spawn a child: start its Worker (slot/stack/TLS already allocated by the parent).
        startVcpu({ role: 'child', func: m.func, sp: m.sp, arg: m.arg, slot: m.slot, stackTop: m.stackTop, tlsBase: m.tlsBase });
      } else if (m.kind === 'done') {
        finish(BigInt(m.value));
      } else if (m.kind === 'trap' || m.kind === 'fail') {
        finish(null, m.why || 'guest trap');
      }
    });
    w.on('error', (e) => finish(null, String(e)));
  };

  let finished = false;
  const finish = (value, err) => {
    if (finished) return;
    finished = true;
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    for (const w of workers) w.terminate();
    const ok = err == null && value === EXPECT;
    console.log(`  vCPUs started: ${started} (1 root + ${started - 1} spawned), ${ms.toFixed(0)} ms`);
    if (err) console.log(`  error: ${err}`);
    else console.log(`  root returned ${value}  expect ${EXPECT}  ${ok ? '✓' : '✗'}`);
    console.log(`\n${ok ? 'PASS' : 'FAIL'}: one guest's vCPUs ran on ${started} separate Workers over ` +
      `one shared memory, synchronising via Atomics (join) ${ok ? '— genuine wasm parallelism' : ''}`);
    process.exit(ok ? 0 : 1);
  };

  // The root vCPU runs on its own Worker (it blocks on join/futex, which the main thread may not do).
  const rootSlot = ex.svm_par_alloc(SLOT);
  const rootStackTop = ex.svm_par_alloc(STACK) + STACK;
  const rootTlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
  startVcpu({ role: 'root', func: 0, slot: rootSlot, stackTop: rootStackTop, tlsBase: rootTlsBase });
}

if (isMainThread) main(); else worker();
