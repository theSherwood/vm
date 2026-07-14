// THREADS/BROWSER step 4c-wasm in a REAL browser — the page-side orchestrator. Proves two things run
// in an actual browser (Chromium via Playwright), not just Node:
//   1. the **powerbox** (`svm_run_pb`) — a guest writes to stdout, single-threaded on the page;
//   2. genuine **parallelism** — one guest's `thread.spawn`ed vCPUs run on separate Web Workers over a
//      shared `WebAssembly.Memory` (only available because the server sent COOP/COEP), synchronising
//      via `Atomics` → the 8-vCPU counter kernel returns 4000.
// The Worker orchestration itself lives in `par.js` (shared with the playground, `play.js`); this
// page mirrors `threads-spawn.mjs` exactly.

import { fetchBytes, loadEngine, makeRunner, readParStdout } from '/web/par.js';
import { compileJit } from '/web/wasmjit.js';

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

  // --- 6) wasm-JIT tier: SVM IR compiled to wasm, run in-browser (BROWSER.md wasm-JIT slice 2/3c) --
  // The `alu` compute kernel is emitted to a wasm module by the cdylib (`svm_wasmjit_compile`),
  // instantiated against the page's OWN linear memory, and its `f0` called directly on the page
  // (compute-only → no Atomics.wait, so the main thread is fine). Assert it equals the `svm_run`
  // interpreter over an arg sweep, then time a heavy run to show the JIT's win over interp-in-wasm.
  // Then a **mixed-tier** guest (3c): a JITted integer caller whose SIMD leaf runs on the
  // interpreter via `env.call_interp` — same result as the whole-guest interpreter.
  const interpBytes = (bytes, arg) => {
    const p = ex.svm_alloc(bytes.length);
    u8().set(bytes, p);
    const r = ex.svm_run(p, bytes.length, BigInt(arg));
    const st = ex.svm_status();
    ex.svm_dealloc(p, bytes.length);
    if (st !== 0) throw new Error(`svm_run status ${st}`);
    return BigInt.asIntN(64, r);
  };
  // Compile SVM text → encoded module via the cdylib's front end (svm_parse), like the playground.
  const encode = (src) => {
    const s = new TextEncoder().encode(src);
    const p = ex.svm_alloc(s.length);
    u8().set(s, p);
    const ok = ex.svm_parse(p, s.length);
    ex.svm_dealloc(p, s.length);
    const optr = ex.svm_parse_ptr();
    const out = u8().slice(optr, optr + ex.svm_parse_len());
    if (ok !== 1) throw new Error(`parse: ${new TextDecoder().decode(out)}`);
    return out;
  };
  const MIXED_SRC = `
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v9)
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v9 v12
  br block1(v7, v11, v13)
block3(v14: i64):
  return v14
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
}`;
  // call_indirect (wasm-JIT next slice): a loop dispatches to func1 (double) / func2 (+100) by index
  // parity through the emitted funcref table — the same masked index + wasm signature check the
  // interpreter's identity table does. All-in-subset, so it JITs whole-module.
  const DISPATCH_SRC = `
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 1
  v11 = i64.and v9 v10
  v12 = i64.const 1
  v13 = i64.add v11 v12
  v14 = i32.wrap_i64 v13
  v15 = call_indirect (i64) -> (i64) v14 (v9)
  v16 = i64.add v8 v15
  v17 = i64.const 1
  v18 = i64.add v9 v17
  br block1(v7, v16, v18)
block3(v19: i64):
  return v19
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 2
  v2 = i64.mul v0 v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i64.add v0 v1
  return v2
}`;
  // SIMD (wasm-JIT next slice): v128 lane arithmetic + a compare/bitmask reduction + a store→load
  // through the confined window — the emitted 0xFD opcodes and the one 16-byte widened access run
  // in-browser against the page's own memory, matching the interpreter. All-in-subset.
  const SIMD_SRC = `
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = i32.const 3
  v4 = i32x4.splat v3
  v5 = i32x4.add v2 v4
  v6 = i32x4.mul v5 v4
  v7 = i64.const 32
  v128.store v7 v6
  v8 = v128.load v7
  v9 = i32x4.max_s v8 v2
  v10 = i32x4.gt_s v9 v4
  v11 = i8x16.bitmask v10
  v12 = i32x4.extract_lane 0 v9
  v13 = i32.add v12 v11
  v14 = i64.extend_i32_s v13
  return v14
}`;
  try {
    const bytes = await fetchBytes('/corpus/alu.svmbc');
    const jit = await compileJit(ex, bytes, { memory });
    if (!jit) throw new Error('svm_wasmjit_compile refused an in-subset module');
    let eq = true;
    for (const arg of [0n, 1n, 2n, 5n, 1000n, -1n, 100000n]) {
      if (jit.call([arg]).value !== interpBytes(bytes, arg)) { eq = false; break; }
    }
    const N = 5_000_000n;
    const t0 = performance.now();
    const jv = jit.call([N]).value;
    const t1 = performance.now();
    const iv = interpBytes(bytes, N);
    const t2 = performance.now();
    const jitMs = t1 - t0, intMs = t2 - t1;

    // Mixed-tier: the JITted caller sums a SIMD leaf run on the interpreter via env.call_interp.
    const mbytes = encode(MIXED_SRC);
    const mjit = await compileJit(ex, mbytes, { memory });
    let mixEq = mjit !== null;
    if (mjit) for (const arg of [0n, 1n, 2n, 5n, 20n, 100n]) {
      if (mjit.call([arg]).value !== interpBytes(mbytes, arg)) { mixEq = false; break; }
    }

    // call_indirect: the emitted funcref table + masked-index dispatch matches the interpreter.
    const dbytes = encode(DISPATCH_SRC);
    const djit = await compileJit(ex, dbytes, { memory });
    let ciEq = djit !== null;
    if (djit) for (const arg of [0n, 1n, 2n, 5n, 64n, 1000n, -1n]) {
      if (djit.call([arg]).value !== interpBytes(dbytes, arg)) { ciEq = false; break; }
    }

    // SIMD: emitted v128 opcodes + a confined 16-byte access match the interpreter in-browser.
    const sbytes = encode(SIMD_SRC);
    const sjit = await compileJit(ex, sbytes, { memory });
    let simdEq = sjit !== null;
    if (sjit) for (const arg of [0n, 1n, 2n, 5n, 64n, 1000n, -1n, 100000n]) {
      if (sjit.call([arg], { winSize: 1 << 16 }).value !== interpBytes(sbytes, arg)) { simdEq = false; break; }
    }

    const ok = eq && jv === iv && mixEq && ciEq && simdEq;
    set('wasmjit', ok ? 'pass' : 'fail',
      `wasmjit: alu f0 in-browser → ${jv} (interp ${iv}) ${eq && jv === iv ? 'ok' : 'FAIL'} · ` +
      `mixed-tier (JIT caller + interp SIMD leaf) ${mixEq ? 'ok' : 'FAIL'} · ` +
      `call_indirect dispatch ${ciEq ? 'ok' : 'FAIL'} · ` +
      `SIMD v128 ${simdEq ? 'ok' : 'FAIL'} · ` +
      `alu n=${N}: jit ${jitMs.toFixed(1)}ms vs interp ${intMs.toFixed(1)}ms → ${(intMs / jitMs).toFixed(1)}× ` +
      `${ok ? 'PASS' : 'FAIL'}`);
    log(`wasmjit → ${jv}, ${(intMs / jitMs).toFixed(1)}× over the interpreter; mixed-tier ${mixEq ? 'ok' : 'FAIL'}; call_indirect ${ciEq ? 'ok' : 'FAIL'}; SIMD ${simdEq ? 'ok' : 'FAIL'}`);
  } catch (e) {
    set('wasmjit', 'fail', `wasmjit: error ${e}`);
  }

  // --- 7) wasm-JIT tier-up **across real Web Workers** (BROWSER.md § "wasm-JIT tier", per-Worker JIT) --
  // The flagship: the 4000 counter kernel where each spawned worker's compute leaf tiers up onto
  // emitted wasm on *its own* Worker (`compile_module_tierup` emits the leaf even though it's reachable
  // only through `thread.spawn`), while the guest keeps running on the interpreter for spawn/join +
  // atomics. Run it BOTH ways over real Workers: plain (all-interp) and tier-up. Both must return 4000,
  // and the tier-up run must actually fire the seam (counter > 0) — a result match alone couldn't
  // distinguish tiered-up from silently-interpreted.
  try {
    const guest = await fetchBytes('/corpus/threads_tierup.svmbc');
    const t0 = performance.now();
    const plain = await run(guest);
    const tiered = await run(guest, { tierup: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = plain.value === 4000n && tiered.value === 4000n && tiered.tierups > 0;
    set('tierup', ok ? 'pass' : 'fail',
      `tierup: ${tiered.started} Workers · plain → ${plain.value} · tier-up → ${tiered.value} ` +
      `(want 4000, ${tiered.tierups} regions ran on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`tierup → plain ${plain.value} / tier-up ${tiered.value} with ${tiered.tierups} tier-ups across ${tiered.started} Workers in ${ms}ms`);
  } catch (e) {
    set('tierup', 'fail', `tierup: error ${e}`);
  }

  // --- 8) §22 guest-JIT **real codegen** across real Web Workers (BROWSER.md § "wasm-JIT tier", slice 5) --
  // The guest holds a `Jit` cap + a host-compiled unit and `cap.call`s invoke; each worker runs the
  // submitted unit on EMITTED WASM on its own Worker (`svm_par_powerbox_jit_codegen` emits it at setup)
  // instead of the interpreter. Run the all-i64 invoke kernel BOTH ways: interp (`jit: true`) and
  // codegen (`jitCodegen: true`). Both must return 1136, and codegen must actually run the emitted unit
  // (counter > 0) — the §22 analogue of the tier-up proof, one tier deeper (guest-loaded code, not a
  // static function). `install`/`call_indirect` (Model B2) + guest-compiled units stay on the interp.
  try {
    const guest = await fetchBytes('/corpus/threads_jit_invoke_i64.svmbc');
    const t0 = performance.now();
    const codegen = await run(guest, { jitCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    // Ground truth 1136 (= the interpreter, differential-checked in the Node twin threads-spawn.mjs
    // SVM_JIT vs SVM_JIT_CODEGEN); here we also require the seam actually fired (units ran on wasm).
    const ok = codegen.value === 1136n && codegen.tierups > 0;
    set('jitcodegen', ok ? 'pass' : 'fail',
      `jitcodegen: ${codegen.started} Workers each Jit.invoke a unit on emitted wasm → ${codegen.value} ` +
      `(want 1136, ${codegen.tierups} units ran on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jitcodegen → ${codegen.value} with ${codegen.tierups} emitted invokes across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('jitcodegen', 'fail', `jitcodegen: error ${e}`);
  }

  // --- 9) §14 instantiate_module **real codegen** across real Web Workers (BROWSER.md slice 5) ------
  // The root `instantiate_module`s a granted unit 8× as confined children (each its own Worker + carve).
  // With codegen, a child whose unit entry is fully in-subset runs it on EMITTED WASM (the unit
  // "compiles on push") instead of a confined vCPU — reading the "K"=75 the parent materialized into its
  // carve → 8 × 75 = 600. Run it BOTH ways: interp (`inst`) and codegen (`instCodegen`); both 600, and
  // codegen must actually run children on wasm. Cap-using unit entries (nested instantiate) stay interp.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_mod.svmbc');
    const unit = await fetchBytes('/corpus/threads_inst_unit.svmbc');
    const opt = { unit, winSize: 1 << 20 };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, inst: true });
    const codegen = await run(guest, { ...opt, instCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 600n && codegen.value === 600n && codegen.tierups > 0;
    set('instcodegen', ok ? 'pass' : 'fail',
      `instcodegen: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 600, ${codegen.tierups} children ran on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instcodegen → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted children across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instcodegen', 'fail', `instcodegen: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('threads', 'fail', `fatal: ${e}`); });
