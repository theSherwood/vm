// V8 (Node) timing runner for the **module-prep tax inside wasm** — the one-time safe-load cost a
// fast-loading demo pays per page load (translation is done at build time). It times the SVM's
// decode + verify + bytecode-compile of a *pre-translated, pre-resolved* `.svmb`, running the SVM's
// own Rust compiled to wasm on V8 (the `svm-browser` cdylib's `svm_prep_bench`). Its ratio to the
// native `prep_svmb` example is the sandbox tax on loading — the counterpart of `bench.mjs`'s
// interpreter-execution tax. See BOOTSPEED.md.
//
//   node bench_prep.mjs <svm_browser.wasm> <module.svmb>
//
// stdout: one line, the best (min) wall-clock ms over the reps.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

const [wasmPath, modPath] = process.argv.slice(2);
if (!modPath) {
  console.error('usage: node bench_prep.mjs <svm_browser.wasm> <module.svmb>');
  process.exit(2);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, {})).exports;
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));

const bytes = readFileSync(modPath);
const ptr = ex.svm_alloc(N(bytes.length));
new Uint8Array(ex.memory.buffer).set(bytes, Number(ptr));
const len = N(bytes.length);

const once = () => {
  const t = performance.now();
  const r = ex.svm_prep_bench(ptr, len);
  const e = performance.now();
  const st = ex.svm_status();
  if (st !== 0) {
    console.error(`svm_prep_bench status ${st} (1=decode 2=unsupported 5=verify)`);
    process.exit(3);
  }
  return { ms: e - t, funcs: Number(r) };
};

const REPS = 6;
let best = Infinity;
let funcs = 0;
for (let r = 0; r < REPS; r++) {
  const o = once();
  funcs = o.funcs;
  if (o.ms < best) best = o.ms;
}
console.error(`module: ${bytes.length} bytes, ${funcs} funcs; best of ${REPS}`);
console.log(best.toFixed(1));
