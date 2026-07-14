// THREADS/BROWSER step 4c-wasm in a REAL browser — the per-vCPU Web Worker. One guest vCPU runs here
// through the engine's resumable `Vcpu` API (`svm_par_run` → a host-serviced event → deliver → run
// again) over the ONE shared linear memory. This is the browser twin of `threads-spawn.mjs`'s
// `worker()`: the only differences are init delivery (a `postMessage` instead of Node `workerData`)
// and that a spawn request is posted to the page (which creates every Worker — no nested Workers).
//
// The host services events with genuine browser primitives: `thread.join` → `Atomics.wait` on the
// child's completion slot; `memory.wait`/`notify` → `Atomics.wait`/`notify` on the futex word. A Worker
// (not the page) is the only place a browser permits a blocking `Atomics.wait`.

const STACK = 1 << 20; // per-Worker stack
const SLOT = 16; // completion slot: [done:i32 @0][result:i64 @8]
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);
// Event codes — must match browser/src/lib.rs PAR_*.
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5, INSTANTIATE = 6, TIERUP = 7, JIT_INVOKE = 8;

self.onmessage = async (e) => {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase,
    smod, entry, slog, fuel, tierup, gptr, glen, tierupCell, jitCodegen, instCodegen } = e.data;
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  // Views over the shared memory, refreshed when stale: the shared WebAssembly.Memory can GROW
  // mid-run (any Worker's in-wasm allocation — e.g. a §14 module compile+push), and views created
  // before a growth don't cover the new region (an Atomics access past the old length throws).
  let i32v = new Int32Array(memory.buffer), i64v = new BigInt64Array(memory.buffer);
  const i32 = () =>
    i32v.byteLength === memory.buffer.byteLength ? i32v : (i32v = new Int32Array(memory.buffer));
  const i64 = () =>
    i64v.byteLength === memory.buffer.byteLength ? i64v : (i64v = new BigInt64Array(memory.buffer));
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // A §14 'confined' child's `win`/`winSize` are already its carve (the parent's window + the event's
  // offset) — a confined child is just a child with a shifted, smaller window (DESIGN.md §14).
  // wasm-JIT tier-up (threads slice): this Worker enables the tier-up bitmap in this instance —
  // `svm_par_enable_jit` emits the tier-up module (a pure leaf reachable only via `thread.spawn`
  // still emits, since the guest keeps interpreting), stashes its bytes + the decoded module (so a
  // cross-tier leaf's `call_interp` works), and reports whether anything tier-ups. This Worker then
  // instantiates the emitted module against the ONE shared memory (each Worker instantiates its own —
  // wasm tables aren't shareable across Workers). On PAR_TIERUP it calls `f{func}` here.
  let emitted = null, envCell = 0;
  if (tierup && ex.svm_par_enable_jit(gptr, glen) === 1) {
    const wptr = Number(ex.svm_wasmjit_ptr()), wlen = ex.svm_wasmjit_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const emod = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {
      env: {
        memory,
        trap: () => {}, // an SVM-specific fault; the following `unreachable` throws, caught below
        call_interp: (f, argsPtr) => { if (ex.svm_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    emitted = emod.exports;
    envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes())); // fuel counter + cross-tier scratch
  }

  // §22 guest-JIT real codegen (BROWSER.md slice 5): the run's single §22 unit was emitted + stashed
  // once at powerbox setup (svm_par_powerbox_jit_codegen); every Worker instantiates its own instance
  // against the ONE shared memory. On PAR_JIT_INVOKE this Worker runs the emitted `f0(win, env, args)`
  // instead of the interpreter. A `new WebAssembly.Module`/`Instance` here is synchronous (the unit is
  // small) so it needs no await inside the event loop.
  let jitUnit = null, jitEnvCell = 0;
  if (jitCodegen && ex.svm_par_enable_jit_codegen() === 1 && ex.svm_par_jit_unit_wasm_len() > 0) {
    const wptr = Number(ex.svm_par_jit_unit_wasm_ptr()), wlen = ex.svm_par_jit_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const umod = new WebAssembly.Module(bytes);
    const uinst = new WebAssembly.Instance(umod, {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, argsPtr) => { if (ex.svm_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    jitUnit = uinst.exports;
    jitEnvCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));
  }

  // §14 instantiate real codegen (BROWSER.md slice 5): a confined child whose granted-unit entry is
  // fully in-subset runs it on EMITTED WASM here and fills the completion slot the parent joins — no
  // vCPU. The unit's data segments were materialized into the carve by the parent before this event,
  // so `f{entry}(win=carveBase, env, …cap-handle args a pure unit ignores)` reads them. A cap-using
  // entry isn't in-subset (`svm_par_inst_eligible` is 0), so it falls through to the interpreter.
  if (role === 'confined' && instCodegen && ex.svm_par_enable_inst_codegen() === 1
      && ex.svm_par_inst_eligible(entry) === 1) {
    const wptr = Number(ex.svm_par_inst_unit_wasm_ptr()), wlen = ex.svm_par_inst_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const uinst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, a) => { if (ex.svm_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    const envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));
    new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel
    const args = new Array(Number(ex.svm_par_inst_nparams(entry))).fill(0n); // cap handles, ignored
    if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted children (non-vacuity)
    try {
      const ret = uinst.exports['f' + entry](win, envCell, ...args);
      i64()[(slot + 8) >> 3] = BigInt(ret); // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake the joiner
    } catch {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped (the joiner traps on deliver_join)
      Atomics.notify(i32(), slot >> 2);
    }
    return;
  }

  const v = role === 'root'
    ? ex.svm_par_root(prog, win, winSize, func)
    : role === 'confined'
      ? ex.svm_par_child_confined(prog, win, slog, smod, entry, BigInt(fuel))
      : ex.svm_par_child(prog, win, winSize, func, BigInt(sp), BigInt(arg));
  if (v === 0) { self.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  const handles = []; // local spawn handle (index) → child completion slot ptr

  for (;;) {
    // I22 hang site. A host wasm trap escaping `svm_par_run` — `memory access out of bounds`, or
    // `unreachable` from a panic=abort engine panic — unwinds into this async `onmessage`, rejecting
    // it. A Worker's unhandled rejection does NOT fire `Worker.onerror` on the page, so par.js's
    // promise would never settle: the vCPU's DOM item would sit `pending` until the harness's 30s
    // `waitForFunction` times out (the silent-flake signature). Convert it into a structured failure —
    // wake any joiner (a non-root vCPU's completion slot) so a parent's `Atomics.wait` doesn't
    // cascade-hang, then report `fail` with the trap text so the page/harness self-identifies.
    let evc;
    try {
      evc = ex.svm_par_run(v);
    } catch (err) {
      if (role !== 'root') {
        const iv = new Int32Array(memory.buffer);
        Atomics.store(iv, slot >> 2, 2); // 2 = trapped
        Atomics.notify(iv, slot >> 2);
      }
      let why = `vcpu ${role} host trap: ${err && err.message ? err.message : err}`;
      // If the trap was a panic=abort engine panic (surfaces as `unreachable`), the Rust panic hook
      // stashed FILE:LINE + message; the trap left memory intact, so read it back here (I22 (a)).
      try {
        const plen = ex.svm_par_last_panic_len ? ex.svm_par_last_panic_len() : 0;
        if (plen > 0) {
          const p = Number(ex.svm_par_last_panic_ptr());
          why += ` | panic: ${new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + plen))}`;
        }
      } catch { /* accessor absent (older build) or read failed — the trap text alone still ships */ }
      self.postMessage({ kind: 'fail', why });
      return; // don't svm_par_free(v): the instance just trapped; the page terminates this Worker
    }
    if (evc === DONE) {
      const value = ex.svm_par_ev_a(v); // i64 → BigInt
      i64()[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake a joiner
      if (role === 'root') self.postMessage({ kind: 'done', value: value.toString() });
      ex.svm_par_free(v);
      return;
    }
    if (evc === TRAP) {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32(), slot >> 2);
      if (role === 'root') self.postMessage({ kind: 'trap' });
      ex.svm_par_free(v);
      return;
    }
    if (evc === SPAWN) {
      const cfunc = Number(ex.svm_par_ev_a(v)), csp = ex.svm_par_ev_b(v), carg = ex.svm_par_ev_c(v);
      // Allocate the child's completion slot + stack + TLS, then ask the page to start its Worker.
      const cslot = ex.svm_par_alloc(SLOT);
      const cstackTop = ex.svm_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      self.postMessage({
        kind: 'spawn', func: cfunc, sp: csp.toString(), arg: carg.toString(),
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.svm_par_deliver_handle(v, handle);
      continue;
    }
    if (evc === JOIN) {
      const cslot = handles[Number(ex.svm_par_ev_a(v))];
      Atomics.wait(i32(), cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32(), cslot >> 2) === 2;
      ex.svm_par_deliver_join(v, i64()[(cslot + 8) >> 3], trapped ? 1 : 0);
      continue;
    }
    if (evc === INSTANTIATE) {
      // §14 confined executor child (THREADS.md 4c-domain §14-D2): the engine already validated the
      // carve + built everything authority-bearing; the operands are inert integers we shuttle into
      // a new Worker (whose window IS the carve), joined via the same completion-slot protocol.
      const am = ex.svm_par_ev_a(v); // (module << 32) | entry
      const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
      const carve = Number(ex.svm_par_ev_b(v)), cslog = Number(ex.svm_par_ev_c(v));
      const cfuel = ex.svm_par_ev_d(v); // i64 → BigInt, shuttled verbatim
      const cslot = ex.svm_par_alloc(SLOT);
      const cstackTop = ex.svm_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.svm_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      self.postMessage({
        kind: 'spawn', role: 'confined', smod: csmod, entry: centry, slog: cslog,
        fuel: cfuel.toString(), win: win + carve, winSize: 1 << cslog,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.svm_par_deliver_handle(v, handle);
      continue;
    }
    if (evc === WAIT) {
      const addr = Number(ex.svm_par_ev_a(v));
      const expected = Number(BigInt.asIntN(32, ex.svm_par_ev_b(v)));
      const timeoutNs = ex.svm_par_ev_d(v);
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(i32(), (win + addr) >> 2, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.svm_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (evc === NOTIFY) {
      const addr = Number(ex.svm_par_ev_a(v)), count = Number(ex.svm_par_ev_b(v));
      ex.svm_par_deliver_code(v, Atomics.notify(i32(), (win + addr) >> 2, count));
      continue;
    }
    if (evc === TIERUP) {
      // Run the emitted `f{func}(win, env, ...i64 args)` over the shared window instead of
      // interpreting. A trap throws (SVM fault → `env.trap` + `unreachable`, or a wasm trap) — we
      // surface it as a vCPU trap. Otherwise marshal the i64 result slots back to the engine.
      const func = Number(ex.svm_par_ev_a(v));
      const argvPtr = Number(ex.svm_par_tierup_argv_ptr(v)), n = Number(ex.svm_par_tierup_argv_len(v));
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]); // i64 args → BigInt
      new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel; preempt = write < 0
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count tier-ups (non-vacuity)
      try {
        const ret = emitted['f' + func](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.svm_par_deliver_tierup(v, rptr, rets.length);
      } catch {
        ex.svm_par_deliver_tierup_trap(v);
      }
      continue;
    }
    if (evc === JIT_INVOKE) {
      // §22 guest-JIT real codegen: the guest `Jit.invoke`d a unit — run the emitted unit's
      // `f0(win, env, ...args)` over the shared window instead of the interpreter, then deliver its
      // result slots. Args marshal by declared type (i32 → JS Number, i64 → BigInt) so a unit need not
      // be all-i64; results go back as `BigInt(ret)` (the engine re-tags by result type). A trap
      // throws and surfaces as a vCPU trap (as an interp invoke would).
      const argvPtr = Number(ex.svm_par_jit_argv_ptr(v)), n = Number(ex.svm_par_jit_argv_len(v));
      const ptypes = new Uint8Array(memory.buffer, Number(ex.svm_par_jit_param_types_ptr(v)), n);
      const args = [];
      for (let i = 0; i < n; i++) {
        const slot = i64()[(argvPtr >> 3) + i];
        args.push(ptypes[i] === 0 ? Number(BigInt.asIntN(32, slot)) : slot); // 0 = i32, 1 = i64
      }
      new DataView(memory.buffer).setBigInt64(jitEnvCell, 1n << 61n, true); // ample fuel
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted invokes (non-vacuity)
      try {
        const ret = jitUnit['f0'](win, jitEnvCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.svm_par_deliver_jit_invoke(v, rptr, rets.length);
      } catch {
        ex.svm_par_deliver_jit_invoke_trap(v);
      }
      continue;
    }
  }
};
