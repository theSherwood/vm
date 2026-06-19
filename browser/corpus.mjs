// Differential check: the wasm `svm_run` vs the native bytecode engine, over the corpus emitted by
// `gencorpus`. Usage: node corpus.mjs <module.wasm>  (run gencorpus first).
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const corpus = JSON.parse(readFileSync('corpus.json', 'utf8'));

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, {})).exports;
const is64 = typeof ex.svm_buf_cap() === 'bigint';
const N = (x) => (is64 ? BigInt(x) : Number(x)); // usize
console.log(`module: ${wasmPath} (${is64 ? 'wasm64' : 'wasm32'})  imports:`,
  WebAssembly.Module.imports(mod).length);

const bufPtr = Number(ex.svm_buf());
const cap = Number(ex.svm_buf_cap());
let total = 0, fail = 0;

for (const { name, file, nargs, cases } of corpus) {
  const bytes = readFileSync(file);
  if (bytes.length > cap) throw new Error(`${file} > buf cap`);
  let bad = 0;
  for (const { arg, status, value } of cases) {
    new Uint8Array(ex.memory.buffer).set(bytes, bufPtr); // re-seed (engine may dirty the window)
    const got = nargs === 0 ? ex.svm_run0(N(bytes.length))
                            : ex.svm_run(N(bytes.length), BigInt(arg));
    const gotStatus = ex.svm_status();
    const okStatus = gotStatus === status;
    // value only meaningful when status==OK(0)
    const okValue = status !== 0 || BigInt(got) === BigInt(value);
    total++;
    if (!okStatus || !okValue) {
      fail++; bad++;
      console.log(`  FAIL ${name}(${arg}): native {status:${status},value:${value}} ` +
        `wasm {status:${gotStatus},value:${got}}`);
    }
  }
  console.log(`  ${name}: ${cases.length - bad}/${cases.length} match`);
}

console.log(`\n${total - fail}/${total} cases match native  ${fail ? 'FAILED' : 'ALL MATCH'}`);
process.exit(fail ? 1 : 0);
