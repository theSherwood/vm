// Live host-import demo: run the host-backed powerbox (`svm_run_live`) and prove the guest's
// capability calls reach **real wasm imports** — stream writes show up on the host console as they
// happen, and the clock reads host time. Unlike corpus.mjs (deterministic, import-free), this build
// is `--features live` and instantiated WITH a host `svm_host` import object.
//
// Usage: node live.mjs <module-live.wasm>  (build: cargo build --release --lib \
//   --target wasm32-unknown-unknown --features live; then: cargo run --bin gencorpus)
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2]
  ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const guestPath = process.argv[3] ?? 'corpus/live.svmbc';

// A fixed sentinel so we can assert the host clock value flows host → guest → back unchanged.
const CLOCK_NS = 1234567890123n;
const captured = { 0: '', 1: '' }; // stdout / stderr, as the host actually received them

let ex; // set after instantiate, so the import can read the guest's linear memory
const imports = {
  svm_host: {
    // host_write(stream, ptr, len): the guest's bytes live in this module's linear memory.
    host_write(stream, ptr, len) {
      const bytes = new Uint8Array(ex.memory.buffer, Number(ptr), Number(len));
      const text = Buffer.from(bytes).toString();
      captured[stream] = (captured[stream] ?? '') + text;
      process[stream === 1 ? 'stderr' : 'stdout'].write(text); // live to the host console
    },
    host_now_ns: () => CLOCK_NS,
  },
};

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const importNames = WebAssembly.Module.imports(mod).map((i) => `${i.module}.${i.name}`);
console.log(`module: ${wasmPath}  imports: [${importNames.join(', ')}]`);
ex = (await WebAssembly.instantiate(mod, imports)).exports;
const is64 = typeof ex.svm_buf_cap() === 'bigint';
const N = (x) => (is64 ? BigInt(x) : Number(x));

const guest = readFileSync(guestPath);
new Uint8Array(ex.memory.buffer).set(guest, Number(ex.svm_buf()));
console.log('--- guest output (live, via host_write import) ---');
const ret = ex.svm_run_live(N(guest.length));
console.log('--------------------------------------------------');

const status = ex.svm_status();
const okStatus = status === 0;
const okWrite = captured[0] === 'live from wasm!\n';
const okClock = BigInt(ret) === CLOCK_NS; // guest returned the host clock value
console.log(`status=${status} return=${ret} (expect clock ${CLOCK_NS})`);
console.log(`  host received stdout: ${JSON.stringify(captured[0])}`);

const pass = okStatus && okWrite && okClock;
console.log(`\n${pass ? 'PASS' : 'FAIL'}: console write ${okWrite ? '✓' : '✗'}, ` +
  `clock round-trip ${okClock ? '✓' : '✗'}, status ${okStatus ? '✓' : '✗'}`);
process.exit(pass ? 0 : 1);
