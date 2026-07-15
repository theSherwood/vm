// Single-vCPU tier-up validation for the threads slice — no Workers, no page. Drives the resumable
// `Vcpu` FFI (`svm_par_run` → `PAR_TIERUP` → run the emitted `f{func}` → `svm_par_deliver_tierup`)
// exactly as `worker.js` does, but on one thread so it runs fast under Node. Asserts the tier-up run
// is byte-identical to the pure bytecode interpreter (`svm_run`) for the same guest — the same oracle
// the native `vcpu_tierup.rs` test uses, now over the REAL emitted wasm through the real FFI.
//
// Run: node browser/tierup.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { engineImports } from './engine-imports.mjs';

const WASM = fileURLToPath(new URL('./target/wasm32-unknown-unknown/release/svm_browser.wasm', import.meta.url));
const PAR_DONE = 0, PAR_TRAP = 1, PAR_TIERUP = 7;
const STACK = 1 << 20;

// func 0 sums `f1(i)` over 0..5 (bound baked in — `svm_par_root` passes no args); func 1 is the
// pure-compute leaf `f(x)=x*3+7`, the tier-up target. Result = Σ_{i=0}^{4}(3i+7) = 30+35 = 65.
const SRC = `
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  v2 = i64.const 0
  v3 = i64.const 0
  br block1(v1, v3, v2)
block1(v4: i64, v5: i64, v6: i64):
  v7 = i64.lt_s v6 v4
  br_if v7 block2(v4, v5, v6) block3(v5)
block2(v8: i64, v9: i64, v10: i64):
  v11 = call 1 (v10)
  v12 = i64.add v9 v11
  v13 = i64.const 1
  v14 = i64.add v10 v13
  br block1(v8, v12, v14)
block3(v15: i64):
  return v15
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  v3 = i64.const 7
  v4 = i64.add v2 v3
  return v4
}
`;

// func 0 calls the eligible leaf f1 once; f1 divides 100/(x-3) with x=3 → traps. The emitted `f1`
// calls `env.trap` then `unreachable` → a catchable RuntimeError at the JS boundary, which worker.js
// surfaces as a vCPU trap. Mirrors vcpu_tierup.rs SRC_TRAP.
const SRC_TRAP = `
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = call 1 (v1)
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.sub v0 v1
  v3 = i64.const 100
  v4 = i64.div_s v3 v2
  return v4
}
`;

async function main() {
  const bytes = readFileSync(WASM);
  const module = await WebAssembly.compile(bytes);
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    throw new Error('not a threads build (expected imported shared memory)');
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  ex.__stack_pointer.value = Number(ex.svm_par_alloc(STACK)) + STACK;
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(Number(ex.svm_par_alloc(ex.__tls_size.value + ex.__tls_align.value)));
  const u8 = () => new Uint8Array(memory.buffer);

  // Parse text → verified module bytes into a stable guest allocation (the PARSE stash is reused).
  const toGuest = (src) => {
    const enc = new TextEncoder().encode(src);
    const sptr = Number(ex.svm_par_alloc(enc.length));
    u8().set(enc, sptr);
    if (ex.svm_parse(sptr, enc.length) !== 1) {
      const p = Number(ex.svm_parse_ptr()), l = ex.svm_parse_len();
      throw new Error('parse failed: ' + new TextDecoder().decode(u8().slice(p, p + l)));
    }
    const g = u8().slice(Number(ex.svm_parse_ptr()), Number(ex.svm_parse_ptr()) + ex.svm_parse_len());
    const gptr = Number(ex.svm_par_alloc(g.length));
    u8().set(g, gptr);
    return { gptr, glen: g.length };
  };

  // JIT-compile the guest, enable the per-instance eligibility bitmap, instantiate the emitted module
  // against the ONE shared memory, and return a driver that services PAR_TIERUP via `f{func}`.
  const buildDriver = async ({ gptr, glen }) => {
    if (ex.svm_par_enable_jit(gptr, glen) !== 1) throw new Error('svm_par_enable_jit rejected the guest');
    const wptr = Number(ex.svm_wasmjit_ptr()), wlen = ex.svm_wasmjit_len();
    return WebAssembly.instantiate(
      await WebAssembly.compile(u8().slice(wptr, wptr + wlen)),
      { env: {
        memory,
        trap: () => {},
        call_interp: (f, argsPtr) => { if (ex.svm_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      } });
  };

  const winSize = 1 << 16;
  const envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));

  const runTierUp = async ({ gptr, glen }) => {
    const emod = await buildDriver({ gptr, glen });
    const emitted = emod.exports;
    const prog = ex.svm_par_compile(gptr, glen);
    if (prog === 0) throw new Error('svm_par_compile null');
    ex.svm_par_powerbox_none();
    const win = Number(ex.svm_par_alloc(winSize));
    const v = ex.svm_par_root(prog, win, winSize, 0);
    if (v === 0) throw new Error('svm_par_root null');
    let tierups = 0;
    for (;;) {
      const evc = ex.svm_par_run(v);
      if (evc === PAR_DONE) { const r = ex.svm_par_ev_a(v); ex.svm_par_free(v); return { value: r, tierups }; }
      if (evc === PAR_TRAP) { ex.svm_par_free(v); return { trap: true, tierups }; }
      if (evc === PAR_TIERUP) {
        tierups++;
        const func = Number(ex.svm_par_ev_a(v));
        const argvPtr = Number(ex.svm_par_tierup_argv_ptr(v)), an = Number(ex.svm_par_tierup_argv_len(v));
        const args = [];
        for (let i = 0; i < an; i++) args.push(new BigInt64Array(memory.buffer)[(argvPtr >> 3) + i]);
        new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true);
        try {
          const ret = emitted['f' + func](win, envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
          const o64 = new BigInt64Array(memory.buffer);
          for (let i = 0; i < rets.length; i++) o64[(rptr >> 3) + i] = BigInt(rets[i]);
          ex.svm_par_deliver_tierup(v, rptr, rets.length);
        } catch {
          ex.svm_par_deliver_tierup_trap(v);
        }
        continue;
      }
      throw new Error('unexpected event ' + evc);
    }
  };

  // 1) Compute guest: tier-up run must equal the pure interpreter, non-vacuously.
  const g = toGuest(SRC);
  const oracle = ex.svm_run(g.gptr, g.glen, 0n);
  const { value, tierups } = await runTierUp(g);
  console.log(`compute: oracle=${oracle} vcpu=${value} tierups=${tierups}`);
  if (value !== oracle) throw new Error(`FAIL: tier-up run ${value} != interp ${oracle}`);
  if (tierups !== 5) throw new Error(`FAIL: expected 5 tier-ups, got ${tierups} (vacuous — seam not exercised)`);
  if (oracle !== 65n) throw new Error(`FAIL: expected 65, got ${oracle}`);

  // 2) Trapping guest: the emitted leaf traps (div by zero) → vCPU must trap, matching the interp.
  const gt = toGuest(SRC_TRAP);
  const rt = await runTierUp(gt);
  console.log(`trap: vcpu trap=${!!rt.trap} tierups=${rt.tierups}`);
  if (!rt.trap) throw new Error(`FAIL: expected a trap, got value ${rt.value}`);
  if (rt.tierups !== 1) throw new Error(`FAIL: expected 1 tier-up before the trap, got ${rt.tierups}`);

  console.log('OK: single-vCPU tier-up path matches the interpreter oracle (compute + trap)');
}

main().catch((e) => { console.error(e); process.exit(1); });
