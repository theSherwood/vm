// V8 (Node) timing runner for the Embench cross-engine harness (see bench/embench/README.md).
// Instantiates a self-contained wasm32 kernel module (exports `run(long n)`, no imports), warms it up
// so TurboFan tiers up, then reports one per-iteration time and the verify result — same methodology
// as the native/SVM drivers: per_iter = (min t(large) - min t(small)) / (large - small), min over reps.
//
//   node run.mjs <kernel.wasm> <small> <large> <verify_n>
// stdout: two lines — "<per_iter_ns>" then "<verify>" (matches the native harness's parse).
import { readFileSync } from 'fs';
import { performance } from 'perf_hooks';

const [file, smallS, largeS, vnS] = process.argv.slice(2);
const small = Number(smallS), large = Number(largeS), vn = Number(vnS);

const mod = new WebAssembly.Module(readFileSync(file));
const imports = WebAssembly.Module.imports(mod);
if (imports.length) {
  console.error(`${file}: unexpected imports: ${imports.map((i) => `${i.module}.${i.name}`).join(',')}`);
  process.exit(2);
}
const run = new WebAssembly.Instance(mod, {}).exports.run;

const REPS = 10;
const best = (n) => {
  run(n); // warm up (lets V8 tier the call site up to TurboFan)
  let b = Infinity;
  for (let r = 0; r < REPS; r++) {
    const t = performance.now();
    run(n);
    const e = performance.now();
    if (e - t < b) b = e - t;
  }
  return b * 1e6; // ms -> ns
};

const perIter = (best(large) - best(small)) / (large - small);
console.log(perIter.toFixed(6));
console.log(run(vn));
