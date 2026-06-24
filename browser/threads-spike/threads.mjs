// Genuinely-parallel atomics proof: two Node worker_threads (separate OS threads) each run the wasm
// module's increment loop over ONE shared linear memory. The atomic path must total exactly 2·N; the
// non-atomic path must lose updates under contention (< 2·N) — which is what proves the atomics are
// real hardware atomics on contended shared memory, not a single-threaded artifact.
//
// Usage:  node threads.mjs   (after `cargo +nightly build --release`)
import { readFileSync } from 'node:fs';
import { Worker, isMainThread, workerData, parentPort } from 'node:worker_threads';

const WASM = new URL('./target/wasm32-unknown-unknown/release/threads_spike.wasm', import.meta.url);
const N = 2_000_000; // increments per worker — enough contention to lose updates in the plain path
const WORKERS = 2;

// Each worker instantiates the module over the shared memory it was handed, then hammers the counter.
async function worker() {
  const { module, memory, addr, n, mode } = workerData;
  const { exports } = await WebAssembly.instantiate(module, { env: { memory } });
  (mode === 'atomic' ? exports.add_atomic : exports.add_plain)(addr, n);
  parentPort.postMessage('done');
}

async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  // The single shared heap every instance imports (256 pages = 16 MiB > the 8 MiB counter address).
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 16384, shared: true });
  const { exports } = await WebAssembly.instantiate(module, { env: { memory } });
  const addr = exports.counter_addr();
  console.log(`shared memory: ${memory.buffer.byteLength >> 20} MiB (shared=${memory.buffer instanceof SharedArrayBuffer}), counter @ 0x${addr.toString(16)}`);

  const run = async (mode) => {
    exports.store(addr, 0); // zero the shared counter
    const workers = Array.from({ length: WORKERS }, () =>
      new Promise((resolve, reject) => {
        const w = new Worker(new URL(import.meta.url), { workerData: { module, memory, addr, n: N, mode } });
        w.once('message', () => w.terminate().then(resolve, resolve));
        w.once('error', reject);
      }));
    const t0 = process.hrtime.bigint();
    await Promise.all(workers);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    return { got: exports.load(addr), ms };
  };

  const want = WORKERS * N;
  const a = await run('atomic');
  const p = await run('plain');

  console.log(`\n  atomic: ${a.got} / ${want}  ${a.got === want ? 'EXACT ✓' : 'WRONG ✗'}  (${a.ms.toFixed(0)} ms)`);
  console.log(`  plain:  ${p.got} / ${want}  ${p.got < want ? `raced ✓ (lost ${want - p.got} updates)` : 'no contention observed'}  (${p.ms.toFixed(0)} ms)`);

  // Pass = atomics give the exact total AND the non-atomic path demonstrably raced (proving real
  // parallel contention, so the atomic correctness isn't a fluke of serialized execution).
  const pass = a.got === want && p.got < want;
  console.log(`\n${pass ? 'PASS' : 'FAIL'}: cross-thread shared-memory atomics ${pass ? 'work' : 'do NOT work'} in wasm`);
  process.exit(pass ? 0 : 1);
}

if (isMainThread) main(); else worker();
