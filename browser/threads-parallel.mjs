// THREADS.md step 4b — genuine parallelism: N Node worker_threads (separate OS threads) each run the
// full SVM engine over the ONE shared linear memory, each over its own guest window, **concurrently**.
// Each Worker is bootstrapped the wasm-threads way: its own stack + TLS block (so the engine's deep
// call stacks don't collide), set from addresses the main thread pre-allocated in the shared memory.
//
// Build the engine as a threads module exporting the stack/TLS globals, then run:
//   node threads-parallel.mjs <module.wasm> [guest.svmbc] [expected] [workers]
import { readFileSync } from 'node:fs';
import { Worker, isMainThread, workerData, parentPort } from 'node:worker_threads';
import { engineImports } from './engine-imports.mjs';

const WASM = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const GUEST = process.argv[3] ?? 'corpus/threads.svmbc';
const EXPECT = BigInt(process.argv[4] ?? 4000);
const WORKERS = Number(process.argv[5] ?? 4);
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

// --- worker: bootstrap its own stack + TLS, then run the engine over its window ----------------
async function worker() {
  const { module, memory, modPtr, modLen, winPtr, winSize, stackTop, tlsBase, arg } = workerData;
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  ex.__stack_pointer.value = stackTop; // a private stack region (grows down from the top)
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // a private TLS block
  const got = ex.svm_run_shared(modPtr, modLen, winPtr, winSize, BigInt(arg));
  parentPort.postMessage(String(got));
}

// --- main: load the guest, carve per-Worker windows + stacks + TLS, then fan out ----------------
async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    console.log('FAIL: not a threads build (module does not import a shared memory)');
    process.exit(1);
  }
  const memory = new WebAssembly.Memory({ initial: 1024, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  const u8 = () => new Uint8Array(memory.buffer);

  // Guest bytes are read-only → shared by all Workers. Everything else is per-Worker + disjoint.
  const guest = readFileSync(GUEST);
  const modPtr = ex.svm_alloc(guest.length);
  u8().set(guest, modPtr);
  const winSize = 1 << 16, stackSize = 1 << 20;
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  const jobs = Array.from({ length: WORKERS }, () => {
    const winPtr = ex.svm_alloc(winSize);
    u8().fill(0, winPtr, winPtr + winSize);
    const stackTop = ex.svm_alloc(stackSize) + stackSize;
    const tlsBase = tlsSize > 0 ? roundUp(ex.svm_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
    return { winPtr, winSize, stackTop, tlsBase };
  });

  console.log(`module: ${WASM}  shared=${memory.buffer instanceof SharedArrayBuffer}`);
  console.log(`  ${WORKERS} workers · stack ${stackSize >> 10}KiB · TLS ${tlsSize}B (align ${tlsAlign}) each`);

  const t0 = process.hrtime.bigint();
  const results = await Promise.all(jobs.map((j) =>
    new Promise((resolve, reject) => {
      const w = new Worker(new URL(import.meta.url), {
        workerData: { module, memory, modPtr, modLen: guest.length, arg: 0, ...j },
      });
      w.once('message', (m) => w.terminate().then(() => resolve(BigInt(m)), () => resolve(BigInt(m))));
      w.once('error', reject);
    })));
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;

  const ok = results.every((r) => r === EXPECT);
  results.forEach((r, i) => console.log(`  worker ${i}: ${r}  ${r === EXPECT ? '✓' : '✗ (want ' + EXPECT + ')'}`));
  console.log(`\n${ok ? 'PASS' : 'FAIL'}: ${WORKERS} SVM engine instances ran in parallel over shared ` +
    `memory (${ms.toFixed(0)} ms)`);
  process.exit(ok ? 0 : 1);
}

if (isMainThread) main(); else worker();
