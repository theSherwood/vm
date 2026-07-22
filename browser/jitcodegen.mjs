// Single-vCPU §22 guest-JIT **real-codegen** validation (BROWSER.md § "wasm-JIT tier", slice 5) — no
// Workers, no page. A guest holds a `Jit` cap + a host-compiled unit and `cap.call`s invoke; with
// codegen on, the host runs the unit on EMITTED WASM (`f{entry}(win, env, …args)`) instead of the
// interpreter, then delivers the result back. Runs each guest both ways and asserts identical results
// — the emitted region must match the interpreter (ground truth `service(6,7) = 142`), the same
// MISCOMPILE-grade contract every tier holds. Covers **i32** and **f64** unit signatures (the Worker
// marshals each arg/result by its wasm type — the ABI generalization this slice adds).
//
// Run: node browser/jitcodegen.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { engineImports } from './engine-imports.mjs';

const WASM = fileURLToPath(new URL('./target/wasm32-unknown-unknown/release/svm_browser.wasm', import.meta.url));
const PAR_DONE = 0, PAR_TRAP = 1, PAR_JIT_INVOKE = 8;
const STACK = 1 << 16;

// `(jit, code) -> (i64)`: invoke the i32 `service(6,7)=142` and return it. Args marshal as i32 Numbers.
const GUEST_I32 = `
memory 16
func (i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32) {
  vc = i64.extend_i32_u vcode
  va = i32.const 6
  vb = i32.const 7
  vr = cap.call 11 1 (i64, i32, i32) -> (i32) vjit (vc, va, vb)
  vr64 = i64.extend_i32_u vr
  return vr64
  }
}
`;

// `(jit, code) -> (i64)`: invoke the f64 `fservice(6.0,7.0)=142.0`, truncate to 142. Args marshal as
// f64 (slot bits → JS Number), the f64 result back to its bits — the float ABI path.
const GUEST_F64 = `
memory 16
func (i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32) {
  vc = i64.extend_i32_u vcode
  va = f64.const 6.0
  vb = f64.const 7.0
  vr = cap.call 11 1 (i64, f64, f64) -> (f64) vjit (vc, va, vb)
  vi = i64.trunc_f64_s vr
  return vi
  }
}
`;

const _sdv = new DataView(new ArrayBuffer(8));
const jitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot))
  : tc === 1 ? slot
  : tc === 2 ? (_sdv.setInt32(0, Number(BigInt.asIntN(32, slot)), true), _sdv.getFloat32(0, true))
  : (_sdv.setBigInt64(0, slot, true), _sdv.getFloat64(0, true));
const jitRes = (ret, tc) => tc === 0 ? BigInt(ret)
  : tc === 1 ? ret
  : tc === 2 ? (_sdv.setFloat32(0, ret, true), BigInt(_sdv.getUint32(0, true)))
  : (_sdv.setFloat64(0, ret, true), _sdv.getBigInt64(0, true));

async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  ex.__stack_pointer.value = Number(ex.svm_par_alloc(STACK)) + STACK;
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(Number(ex.svm_par_alloc(ex.__tls_size.value + ex.__tls_align.value)));
  const u8 = () => new Uint8Array(memory.buffer);
  const winSize = 1 << 16;

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

  const prove = async (name, guestSrc, serviceKind) => {
    const { gptr, glen } = toGuest(guestSrc);
    ex.svm_par_jit_codegen_service(serviceKind); // 0 = i32 service, 1 = f64 service
    if (ex.svm_par_powerbox_jit_codegen(gptr, glen) !== 1) throw new Error(`${name}: powerbox failed`);
    const wptr = Number(ex.svm_par_jit_unit_wasm_ptr()), wlen = ex.svm_par_jit_unit_wasm_len();
    if (wlen === 0) throw new Error(`${name}: no emitted unit wasm`);
    const unit = await WebAssembly.instantiate(await WebAssembly.compile(u8().slice(wptr, wptr + wlen)), {
      env: { memory, trap: () => {}, call_interp: (f, a) => { if (ex.svm_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); } },
    });
    const f = unit.exports;
    const envCell = Number(ex.svm_par_alloc(ex.svm_wasmjit_env_bytes()));

    const run = (codegen) => {
      ex.svm_par_jit_set_codegen(codegen ? 1 : 0);
      const prog = ex.svm_par_compile_jit(gptr, glen);
      const win = Number(ex.svm_par_alloc(winSize));
      const v = ex.svm_par_root(prog, win, winSize, 0);
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
          for (let i = 0; i < n; i++) args.push(jitArg(new BigInt64Array(memory.buffer)[(argvPtr >> 3) + i], ptypes[i]));
          new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true);
          try {
            const ret = f['f0'](win, envCell, ...args);
            const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
            const rn = Number(ex.svm_par_jit_result_types_len(v));
            const rtypes = new Uint8Array(memory.buffer, Number(ex.svm_par_jit_result_types_ptr(v)), rn);
            const rptr = Number(ex.svm_par_alloc(Math.max(1, rets.length) * 8));
            const o64 = new BigInt64Array(memory.buffer);
            for (let i = 0; i < rets.length; i++) o64[(rptr >> 3) + i] = jitRes(rets[i], rtypes[i]);
            ex.svm_par_deliver_jit_invoke(v, rptr, rets.length);
          } catch {
            ex.svm_par_deliver_jit_invoke_trap(v);
          }
          continue;
        }
        throw new Error('unexpected event ' + evc);
      }
    };

    const interp = run(false), codegen = run(true);
    console.log(`${name}: interp → ${interp.value} (${interp.invokes} JS invokes) · codegen → ${codegen.value} (${codegen.invokes} JS invokes)`);
    if (interp.value !== 142n) throw new Error(`FAIL ${name}: interp expected 142, got ${interp.value}`);
    if (interp.invokes !== 0) throw new Error(`FAIL ${name}: interp surfaced ${interp.invokes} JS invokes`);
    if (codegen.value !== 142n) throw new Error(`FAIL ${name}: codegen ${codegen.value} != 142`);
    if (codegen.invokes !== 1) throw new Error(`FAIL ${name}: codegen ran ${codegen.invokes} emitted invokes`);
  };

  await prove('i32', GUEST_I32, 0);
  await prove('f64', GUEST_F64, 1);
  console.log('OK: §22 Jit.invoke on emitted wasm matches the interpreter for i32 and f64 unit sigs (142)');
}

main().catch((e) => { console.error(e); process.exit(1); });
