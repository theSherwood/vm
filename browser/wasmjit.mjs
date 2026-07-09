// Browser wasm-JIT tier — the Node proof (BROWSER.md § "wasm-JIT tier", slice 2). The fast-iteration
// twin of the Chromium `#wasmjit` page item: it exercises the exact browser path — emit via the
// cdylib's `svm_wasmjit_compile`, instantiate the emitted module against the cdylib's shared linear
// memory, and call `f0` directly — comparing the JIT result to the `svm_run` interpreter (the
// oracle) and confirming the trap kinds line up. Runs on the **threads** wasm32 cdylib (imported
// shared memory, exactly the browser page's build) on Node's WebAssembly; no Playwright, no Workers
// (the compute kernel is single-threaded — genuine multi-Worker JIT is slice 4).
//
// Usage:  node wasmjit.mjs [module.wasm]        (build the threads cdylib first — see browser-test.mjs)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { compileJit } from './web/wasmjit.js';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] ?? join(ROOT, 'target/wasm32-unknown-unknown/release/svm_browser.wasm');

const mod = await WebAssembly.compile(readFileSync(wasmPath));
if (!WebAssembly.Module.imports(mod).some((i) => i.kind === 'memory')) {
  console.log('FAIL: expected the threads cdylib (imported shared memory) — build with the threads flags');
  process.exit(1);
}
// The one shared linear memory both the cdylib and the emitted JIT modules see (matches the page).
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, { env: { memory } });

// Encode an SVM text module through the cdylib's own front end (svm_parse) so this harness needs no
// .svmbc fixtures — it produces exactly the bytes `svm_wasmjit_compile` / `svm_run` consume.
function encode(src) {
  const bytes = new TextEncoder().encode(src);
  const p = ex.svm_alloc(bytes.length);
  new Uint8Array(memory.buffer).set(bytes, Number(p));
  const ok = ex.svm_parse(p, bytes.length);
  ex.svm_dealloc(p, bytes.length);
  const optr = Number(ex.svm_parse_ptr());
  const out = new Uint8Array(memory.buffer).slice(optr, optr + ex.svm_parse_len());
  if (ok !== 1) throw new Error(`parse: ${new TextDecoder().decode(out)}`);
  return out;
}

// The interpreter oracle: decode + run func 0 on the bytecode engine, returning its i64 result.
function interp(moduleBytes, arg) {
  const p = ex.svm_alloc(moduleBytes.length);
  new Uint8Array(memory.buffer).set(moduleBytes, Number(p));
  const r = ex.svm_run(p, moduleBytes.length, BigInt(arg));
  const st = ex.svm_status();
  ex.svm_dealloc(p, moduleBytes.length);
  return { st, r };
}

let failed = false;
const report = (name, ok, detail) => {
  if (!ok) failed = true;
  console.log(`  ${name}: ${ok ? 'PASS' : 'FAIL'}${detail ? ` — ${detail}` : ''}`);
};

// alu i64-LCG: pure compute, one i64 param — the headline JIT-vs-interp equality + speedup probe.
const ALU = `
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}`;

// A store→load through the confined window — proves the emitted mask+guard shares the cdylib memory.
const MEM = `
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}`;

// A guest `unreachable`: the JIT must trap where the interpreter traps (STATUS_TRAP == 3).
const TRAP = `
func (i64) -> (i64) {
block0(v0: i64):
  unreachable
}`;

// Mixed-tier (slice 3c): an integer caller (JITted) sums a SIMD leaf f(i)=2i over 0..n. The leaf
// (v128 internally → out of subset, memory-free) runs on the bytecode interpreter via
// env.call_interp; the whole-guest interp oracle (svm_run) must agree. (Floats are now in-subset,
// so a float leaf would be JITted directly rather than crossing tiers.)
const MIXED = `
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

async function main() {
  console.log(`module: ${wasmPath}\n`);

  // --- equality: JIT result == interpreter result across an arg sweep ---
  const sweep = [0n, 1n, 2n, 5n, 1000n, -1n, -1000n, 100000n];
  for (const [name, src, winSize] of [['alu', ALU, 1 << 16], ['mem', MEM, 1 << 16]]) {
    const bytes = encode(src);
    const jit = await compileJit(ex, bytes, { memory });
    if (!jit) { report(`${name} (eligible)`, false, 'svm_wasmjit_compile refused an in-subset module'); continue; }
    let allEq = true;
    for (const arg of sweep) {
      const { st, r } = interp(bytes, arg);
      if (st !== 0) { allEq = false; break; }
      const got = jit.call([arg], { winSize }).value;
      if (got !== BigInt.asIntN(64, r)) { allEq = false; console.log(`    ${name}(${arg}): jit ${got} != interp ${r}`); break; }
    }
    report(`${name} equality (jit == interp over ${sweep.length} args)`, allEq);
  }

  // --- trap parity: guest unreachable traps both tiers ---
  {
    const bytes = encode(TRAP);
    const jit = await compileJit(ex, bytes, { memory });
    const { st } = interp(bytes, 0);
    let trapped = false;
    try { jit.call([0n]); } catch (e) { trapped = e.trap === 'wasm'; }
    report('trap parity (unreachable)', st === 3 && trapped, `interp status=${st}, jit trapped=${trapped}`);
  }

  // --- mixed-tier (slice 3c): JITted integer caller + interp SIMD leaf via env.call_interp ---
  {
    const bytes = encode(MIXED);
    const jit = await compileJit(ex, bytes, { memory });
    let allEq = jit !== null;
    if (!jit) report('mixed (eligible)', false, 'svm_wasmjit_compile refused a mixed-tier module');
    else for (const arg of sweep) {
      const { st, r } = interp(bytes, arg);
      if (st !== 0) { allEq = false; break; }
      if (jit.call([arg]).value !== BigInt.asIntN(64, r)) {
        allEq = false; console.log(`    mixed(${arg}): jit != interp ${r}`); break;
      }
    }
    report(`mixed-tier equality (JITted caller + interp SIMD leaf over ${sweep.length} args)`, allEq);
  }

  // --- speedup: the whole point. Time a heavy alu loop on both tiers. ---
  {
    const bytes = encode(ALU);
    const jit = await compileJit(ex, bytes, { memory });
    const N = 5_000_000n;
    const t0 = performance.now();
    const ji = jit.call([N]).value;
    const t1 = performance.now();
    const { r } = interp(bytes, N);
    const t2 = performance.now();
    const eq = ji === BigInt.asIntN(64, r);
    const jitMs = t1 - t0, intMs = t2 - t1;
    report(`speedup (alu n=${N})`, eq, `jit ${jitMs.toFixed(1)}ms vs interp ${intMs.toFixed(1)}ms → ${(intMs / jitMs).toFixed(1)}×`);
  }

  console.log(`\n${failed ? 'FAIL' : 'PASS'}: SVM IR JIT-compiled to wasm, run in a wasm host against the ` +
    `cdylib's own memory, matches the bytecode interpreter`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.log(`fatal: ${e.stack ?? e}`); process.exit(1); });
