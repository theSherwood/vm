// Drive the SVM browser-entry wasm module in Node. Usage:
//   node run.mjs <module.wasm> <fixture.svmbc>
// Verifies (a) the no-import smoke anchors via run_guest, and (b) the production svm_run path,
// which decodes an encoded SVM IR module from the scratch buffer and runs it on the bytecode engine.
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const fixturePath = process.argv[3] ?? 'alu.svmbc';

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const imports = WebAssembly.Module.imports(mod);
console.log(`module: ${wasmPath}`);
console.log('imports required:', imports);
const instance = await WebAssembly.instantiate(mod, {});
const ex = instance.exports;

// Pointers/lengths are usize: i32 on wasm32 (Number), i64 on wasm64 (BigInt). Normalize.
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));
const I = (x) => (typeof x === 'bigint' ? x : BigInt(x)); // guest i64 result -> BigInt
console.log(`address width: ${is64 ? 'wasm64 (memory64)' : 'wasm32'}`);

// Allocate `bytes.length` in linear memory and copy `bytes` in; returns the alloc handle.
const load = (bytes) => {
  const ptr = ex.svm_alloc(N(bytes.length));
  new Uint8Array(ex.memory.buffer).set(bytes, Number(ptr)); // re-fetch view (alloc may grow memory)
  return { ptr, len: N(bytes.length), free: () => ex.svm_dealloc(ptr, N(bytes.length)) };
};

let ok = true;
const check = (label, got, expect) => {
  const pass = I(got) === I(expect);
  ok &&= pass;
  console.log(`  ${label} = ${got}  expect ${expect}  ${pass ? 'PASS' : 'FAIL'}`);
};

// (a) embedded smoke kernel, no host imports
console.log('\n[a] run_guest (embedded, no imports):');
check('run_guest(0)', ex.run_guest(0n), 0n);
check('run_guest(1)', ex.run_guest(1n), 1442695040888963407n);

// (b) production path: decode an encoded module from a host-allocated buffer, then run it
console.log('\n[b] svm_run (decode encoded IR + run on bytecode engine):');
const fixture = readFileSync(fixturePath);
const m = load(fixture);
console.log(`  loaded ${fixture.length}-byte encoded module via svm_alloc`);

const run = (arg) => {
  const r = ex.svm_run(m.ptr, m.len, I(arg));
  const st = ex.svm_status();
  if (st !== 0) throw new Error(`svm_run status ${st}`);
  return r;
};
check('svm_run(arg=0)', run(0n), 0n);
check('svm_run(arg=1)', run(1n), 1442695040888963407n);
console.log(`  svm_run(arg=1000) = ${run(1000n)}  (matches native run_guest)`);
m.free();

// fail-closed sentinel: garbage bytes -> decode error, no crash
const junk = load(new Uint8Array([0xff, 0xff, 0xff, 0xff]));
ex.svm_run(junk.ptr, junk.len, I(0));
const garbageStatus = ex.svm_status();
junk.free();
console.log(`  svm_run(garbage) status = ${garbageStatus} (1=DECODE_ERR expected)`);
ok &&= garbageStatus === 1;

console.log(ok ? '\nALL CHECKS PASS' : '\nFAILED');
process.exit(ok ? 0 : 1);
