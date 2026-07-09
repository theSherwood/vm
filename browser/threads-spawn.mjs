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
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5, INSTANTIATE = 6, TIERUP = 7;

// ---- a single vCPU on this Worker ---------------------------------------------------------------
async function worker() {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase,
    smod, entry, slog, fuel, tierup, gptr, glen, tierupCell } = workerData;
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  // Views over the shared memory, refreshed when stale: a shared `WebAssembly.Memory` can GROW
  // mid-run (any Worker's in-wasm allocation — e.g. a §14 module compile+push), and views created
  // before a growth don't cover the new region (an Atomics access past the old length throws).
  let i32v = new Int32Array(memory.buffer), i64v = new BigInt64Array(memory.buffer);
  const i32 = () =>
    i32v.byteLength === memory.buffer.byteLength ? i32v : (i32v = new Int32Array(memory.buffer));
  const i64 = () =>
    i64v.byteLength === memory.buffer.byteLength ? i64v : (i64v = new BigInt64Array(memory.buffer));
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // wasm-JIT tier-up (per-Worker JIT): this Worker enables the tier-up bitmap for its instance
  // (`svm_par_enable_jit` emits the tier-up module — a pure leaf reachable only via `thread.spawn`
  // still emits) and instantiates the emitted module against the ONE shared memory. Each Worker
  // instantiates its own (wasm tables aren't shareable across Workers). On TIERUP it runs `f{func}`.
  let emitted = null, envCell = 0;
  if (tierup && ex.svm_par_enable_jit(gptr, glen) === 1) {
    const wptr = Number(ex.svm_wasmjit_ptr()), wlen = ex.svm_wasmjit_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const emod = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {
      env: {
        memory,
        trap: () => {}, // SVM fault; the following `unreachable` throws, caught below as a vCPU trap
        call_interp: (f, a) => { if (ex.svm_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    emitted = emod.exports;
    envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));
  }

  // A §14 'confined' child's `win`/`winSize` are already its carve (the parent's window + the event's
  // offset) — a confined child is just a child with a shifted, smaller window (DESIGN.md §14).
  const v = role === 'root'
    ? ex.svm_par_root(prog, win, winSize, func)
    : role === 'confined'
      ? ex.svm_par_child_confined(prog, win, slog, smod, entry, BigInt(fuel))
      : ex.svm_par_child(prog, win, winSize, func, BigInt(sp), BigInt(arg));
  if (v === 0) { parentPort.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  const handles = []; // local spawn handle (index) → child completion slot ptr

  for (;;) {
    const ev = ex.svm_par_run(v);
    if (ev === DONE) {
      const value = ex.svm_par_ev_a(v); // i64 → BigInt
      i64()[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake a joiner
      if (role === 'root') parentPort.postMessage({ kind: 'done', value: value.toString() });
      ex.svm_par_free(v);
      return;
    }
    if (ev === TRAP) {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32(), slot >> 2);
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
      Atomics.wait(i32(), cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32(), cslot >> 2) === 2;
      const result = i64()[(cslot + 8) >> 3];
      ex.svm_par_deliver_join(v, result, trapped ? 1 : 0);
      continue;
    }
    if (ev === INSTANTIATE) {
      // §14 confined executor child: the engine already validated the carve + built everything
      // authority-bearing; the operands are inert integers we shuttle into a new Worker (whose
      // window IS the carve), joined via the same completion-slot protocol as SPAWN.
      const am = ex.svm_par_ev_a(v); // (module << 32) | entry
      const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
      const carve = Number(ex.svm_par_ev_b(v)), cslog = Number(ex.svm_par_ev_c(v));
      const cfuel = ex.svm_par_ev_d(v); // i64 → BigInt, shuttled verbatim
      const cslot = ex.svm_par_alloc(SLOT);
      const cstackTop = ex.svm_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      parentPort.postMessage({
        kind: 'spawn', role: 'confined', smod: csmod, entry: centry, slog: cslog,
        fuel: cfuel.toString(), win: win + carve, winSize: 1 << cslog,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.svm_par_deliver_handle(v, handle);
      continue;
    }
    if (ev === WAIT) {
      const addr = Number(ex.svm_par_ev_a(v)), expected = Number(BigInt.asIntN(32, ex.svm_par_ev_b(v)));
      const timeoutNs = ex.svm_par_ev_d(v);
      const idx = (win + addr) >> 2;
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(i32(), idx, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.svm_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (ev === NOTIFY) {
      const addr = Number(ex.svm_par_ev_a(v)), count = Number(ex.svm_par_ev_b(v));
      const woke = Atomics.notify(i32(), (win + addr) >> 2, count);
      ex.svm_par_deliver_code(v, woke);
      continue;
    }
    if (ev === TIERUP) {
      // Run the emitted `f{func}(win, env, ...i64 args)` on this Worker instead of interpreting. A
      // trap throws (SVM fault → env.trap + unreachable, or a wasm trap) → surface as a vCPU trap.
      const tfunc = Number(ex.svm_par_ev_a(v));
      const argvPtr = Number(ex.svm_par_tierup_argv_ptr(v)), n = Number(ex.svm_par_tierup_argv_len(v));
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]);
      new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count tier-ups (non-vacuity)
      try {
        const ret = emitted['f' + tfunc](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.svm_par_deliver_tierup(v, rptr, rets.length);
      } catch {
        ex.svm_par_deliver_tierup_trap(v);
      }
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

  // The one shared guest window every vCPU runs over (SVM_WIN sizes it — the §14 kernels declare a
  // 1 MiB window so their 64 KiB carves stay wasm-page-aligned).
  const winSize = Number(process.env.SVM_WIN ?? 1 << 16);
  const win = ex.svm_par_alloc(winSize);

  // 4d I/O mode (SVM_IO=1): publish the run's shared powerbox — a `Mutex<Host>` in shared linear
  // memory every vCPU dispatches `cap.call` through, so worker vCPUs do host I/O with no JS in the
  // loop. Stdout accumulates in the powerbox; main reads it back after the run.
  if (process.env.SVM_IO === '1' && ex.svm_par_powerbox_io() !== 1) {
    console.log('FAIL: svm_par_powerbox_io returned 0'); process.exit(1);
  }
  // §14 mode (SVM_INST=1): publish the run recipe — the root's `Instantiator` spans the window, plus
  // the optional granted module (SVM_INST_UNIT) for `instantiate_module`. The root vCPU builds its own
  // powerbox from it (svm_par_root); confined children build theirs in-engine.
  if (process.env.SVM_INST === '1') {
    let uptr = 0, ulen = 0;
    if (process.env.SVM_INST_UNIT) {
      const unit = readFileSync(process.env.SVM_INST_UNIT);
      uptr = ex.svm_par_alloc(unit.length);
      u8().set(unit, uptr);
      ulen = unit.length;
    }
    if (ex.svm_par_powerbox_inst(BigInt(winSize), uptr, ulen) !== 1) {
      console.log('FAIL: svm_par_powerbox_inst returned 0'); process.exit(1);
    }
  }

  console.log(`module: ${WASM}  shared=${memory.buffer instanceof SharedArrayBuffer}`);
  console.log(`  prog@0x${prog.toString(16)}  window@0x${win.toString(16)} (${winSize >> 10}KiB)  TLS ${tlsSize}B`);

  // wasm-JIT tier-up (SVM_TIERUP=1): each Worker enables the tier-up bitmap from the guest bytes
  // (kept live at `gptr`) and runs eligible compute regions on emitted wasm. The guest still runs on
  // the interpreter — only direct calls to emitted pure leaves tier up.
  const tierup = process.env.SVM_TIERUP === '1';
  // A shared i32 cell every Worker atomically bumps on each tier-up — proves the seam actually fired
  // (a result match alone couldn't distinguish "tiered up" from "silently interpreted").
  const tierupCell = tierup ? ex.svm_par_alloc(4) : 0;

  const workers = new Set();
  let started = 0;
  const t0 = process.hrtime.bigint();

  const startVcpu = (cfg) => {
    started++;
    const w = new Worker(new URL(import.meta.url), {
      workerData: { module, memory, prog, win, winSize, tierup, gptr, glen: guest.length, tierupCell, ...cfg },
    });
    workers.add(w);
    w.on('message', (m) => {
      if (m.kind === 'spawn') {
        // A vCPU asked to spawn a (plain or §14-confined) child: start its Worker with the message's
        // cfg verbatim (slot/stack/TLS already allocated by the parent; a confined child's message
        // carries its own win/winSize — the carve — overriding the run defaults).
        const { kind, ...cfg } = m;
        startVcpu({ role: 'child', ...cfg });
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
    let ok = err == null && value === EXPECT;
    console.log(`  vCPUs started: ${started} (1 root + ${started - 1} spawned), ${ms.toFixed(0)} ms`);
    if (err) console.log(`  error: ${err}`);
    else console.log(`  root returned ${value}  expect ${EXPECT}  ${ok ? '✓' : '✗'}`);
    // Tier-up non-vacuity: with SVM_TIERUP the workers must have actually run emitted regions.
    if (tierup) {
      const tiered = Atomics.load(new Int32Array(memory.buffer), tierupCell >> 2);
      const tieredOk = tiered > 0;
      console.log(`  tier-ups fired: ${tiered}  ${tieredOk ? '✓ (ran emitted wasm)' : '✗ (vacuous — never tiered up)'}`);
      ok = ok && tieredOk;
    }
    // 4d I/O mode: read the shared powerbox's accumulated stdout back and check the expected
    // schedule-independent bytes ("tick\n" × SVM_IO_LINES, default 8).
    if (process.env.SVM_IO === '1') {
      const len = ex.svm_par_stdout_len();
      const out = Buffer.from(u8().slice(ex.svm_par_stdout_ptr(), ex.svm_par_stdout_ptr() + len)).toString();
      const want = 'tick\n'.repeat(Number(process.env.SVM_IO_LINES ?? 8));
      const outOk = out === want;
      console.log(`  stdout: ${JSON.stringify(out)}  ${outOk ? '✓' : `✗ (want ${JSON.stringify(want)})`}`);
      ok = ok && outOk;
    }
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
