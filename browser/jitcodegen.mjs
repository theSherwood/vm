// Single-vCPU §22 guest-JIT **real-codegen** validation (BROWSER.md § "wasm-JIT tier", slice 5) — no
// Workers, no page. A guest holds a `Jit` cap + a host-compiled unit and `cap.call`s invoke; with
// codegen on, the host runs the unit on EMITTED WASM (`f{entry}(win, env, …args)`) instead of the
// interpreter, then delivers the result back (`svm_par_deliver_jit_invoke`). Runs the same guest both
// ways and asserts identical results — the emitted region must match the interpreter (the ground
// truth `service(6,7) = 6*7 + 100 = 142`), the same MISCOMPILE-grade contract every other tier holds.
//
// Run: node browser/jitcodegen.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const WASM = fileURLToPath(new URL('./target/wasm32-unknown-unknown/release/svm_browser.wasm', import.meta.url));
const PAR_DONE = 0, PAR_TRAP = 1, PAR_JIT_INVOKE = 8;
const STACK = 1 << 20;

// Guest `(jit, code) -> (i64)`: invoke the unit with i32 args (6, 7) and return its result. Single
// vCPU — no threads; the invoke is the only host event. The unit `service` is `(i32,i32)->(i32)`, so
// the Worker marshals the args as i32 (JS Numbers) — the ABI generalization this slice adds.
// `service(6,7) = 142`.
const GUEST = `
memory 16
func (i32, i32) -> (i64) {
block0(vjit: i32, vcode: i32):
  vc = i64.extend_i32_u vcode
  va = i32.const 6
  vb = i32.const 7
  vr = cap.call 11 1 (i64, i32, i32) -> (i32) vjit (vc, va, vb)
  vr64 = i64.extend_i32_u vr
  return vr64
}
`;

async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    throw new Error('not a threads build (expected imported shared memory)');
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  ex.__stack_pointer.value = Number(ex.svm_par_alloc(STACK)) + STACK;
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(Number(ex.svm_par_alloc(ex.__tls_size.value + ex.__tls_align.value)));
  const u8 = () => new Uint8Array(memory.buffer);

  const enc = new TextEncoder().encode(GUEST);
  const sptr = Number(ex.svm_par_alloc(enc.length));
  u8().set(enc, sptr);
  if (ex.svm_parse(sptr, enc.length) !== 1) {
    const p = Number(ex.svm_parse_ptr()), l = ex.svm_parse_len();
    throw new Error('parse failed: ' + new TextDecoder().decode(u8().slice(p, p + l)));
  }
  const guest = u8().slice(Number(ex.svm_parse_ptr()), Number(ex.svm_parse_ptr()) + ex.svm_parse_len());
  const gptr = Number(ex.svm_par_alloc(guest.length));
  u8().set(guest, gptr);
  const glen = guest.length;

  // Build the codegen powerbox: grants the Jit cap, host-compiles the all-i64 unit, and emits its
  // wasm. Then instantiate that emitted unit module against the shared memory (the Worker/host does).
  if (ex.svm_par_powerbox_jit_codegen(gptr, glen) !== 1) throw new Error('svm_par_powerbox_jit_codegen failed');
  const wptr = Number(ex.svm_par_jit_unit_wasm_ptr()), wlen = ex.svm_par_jit_unit_wasm_len();
  if (wlen === 0) throw new Error('no emitted unit wasm');
  const unit = await WebAssembly.instantiate(
    await WebAssembly.compile(u8().slice(wptr, wptr + wlen)),
    { env: {
      memory,
      trap: () => {},
      call_interp: (f, a) => { if (ex.svm_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
    } });
  const f = unit.exports; // f0(win, env, ...i64 args) -> i64
  const envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));
  const winSize = 1 << 16;

  const run = (codegen) => {
    ex.svm_par_jit_set_codegen(codegen ? 1 : 0);
    const prog = ex.svm_par_compile_jit(gptr, glen);
    if (prog === 0) throw new Error('svm_par_compile_jit null');
    const win = Number(ex.svm_par_alloc(winSize));
    const v = ex.svm_par_root(prog, win, winSize, 0);
    if (v === 0) throw new Error('svm_par_root null');
    let invokes = 0;
    for (;;) {
      const evc = ex.svm_par_run(v);
      if (evc === PAR_DONE) { const r = ex.svm_par_ev_a(v); ex.svm_par_free(v); return { value: r, invokes }; }
      if (evc === PAR_TRAP) { ex.svm_par_free(v); return { trap: true, invokes }; }
      if (evc === PAR_JIT_INVOKE) {
        invokes++;
        const argvPtr = Number(ex.svm_par_jit_argv_ptr(v)), n = Number(ex.svm_par_jit_argv_len(v));
        const ptypes = new Uint8Array(memory.buffer, Number(ex.svm_par_jit_param_types_ptr(v)), n);
        const args = [];
        for (let i = 0; i < n; i++) {
          const slot = new BigInt64Array(memory.buffer)[(argvPtr >> 3) + i];
          args.push(ptypes[i] === 0 ? Number(BigInt.asIntN(32, slot)) : slot); // 0 = i32, 1 = i64
        }
        new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel
        try {
          const ret = f['f0'](win, envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
          const o64 = new BigInt64Array(memory.buffer);
          for (let i = 0; i < rets.length; i++) o64[(rptr >> 3) + i] = BigInt(rets[i]);
          ex.svm_par_deliver_jit_invoke(v, rptr, rets.length);
        } catch {
          ex.svm_par_deliver_jit_invoke_trap(v);
        }
        continue;
      }
      throw new Error('unexpected event ' + evc);
    }
  };

  // Interp mode services the invoke in-Rust (no PAR_JIT_INVOKE surfaced); codegen runs the emitted
  // unit here. Both must return 142.
  const interp = run(false);
  const codegen = run(true);
  console.log(`interp → ${interp.value} (${interp.invokes} JS invokes) · codegen → ${codegen.value} (${codegen.invokes} JS invokes)`);
  if (interp.value !== 142n) throw new Error(`FAIL: interp expected 142, got ${interp.value}`);
  if (interp.invokes !== 0) throw new Error(`FAIL: interp should surface no JS invoke, got ${interp.invokes}`);
  if (codegen.value !== interp.value) throw new Error(`FAIL: codegen ${codegen.value} != interp ${interp.value}`);
  if (codegen.invokes !== 1) throw new Error(`FAIL: codegen should run the emitted unit once, got ${codegen.invokes}`);
  console.log('OK: §22 Jit.invoke on emitted wasm matches the interpreter (142, 1 emitted invoke)');
}

main().catch((e) => { console.error(e); process.exit(1); });
