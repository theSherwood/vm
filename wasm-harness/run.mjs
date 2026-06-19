import { readFileSync } from 'node:fs';
const bytes = readFileSync('target/wasm32-unknown-unknown/release/wasm_harness.wasm');
const mod = await WebAssembly.compile(bytes);
console.log('imports required:', WebAssembly.Module.imports(mod));
let inst;
try {
  inst = await WebAssembly.instantiate(mod, {});
} catch (e) {
  console.log('no-import instantiate failed, retrying with stub env:', e.message);
  const env = {};
  for (const i of WebAssembly.Module.imports(mod)) {
    (env[i.module] ??= {})[i.name] = () => { throw new Error('host call '+i.name); };
  }
  inst = await WebAssembly.instantiate(mod, env);
}
const { run_guest } = inst.exports;
const cases = [[0n, 0n], [1n, 1442695040888963407n]];
let ok = true;
for (const [n, expect] of cases) {
  const got = run_guest(n);
  const pass = got === expect;
  ok &&= pass;
  console.log(`run_guest(${n}) = ${got}  expect ${expect}  ${pass ? 'PASS' : 'FAIL'}`);
}
// a few more to show it runs a real loop
for (const n of [2n, 5n, 1000n]) console.log(`run_guest(${n}) = ${run_guest(n)}`);
console.log(ok ? '\nALL ANCHORS PASS' : '\nFAILED');
process.exit(ok ? 0 : 1);
