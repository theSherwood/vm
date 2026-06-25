// THREADS.md step 4c-wasm — the foundational unknown for distributing one guest's vCPUs across
// Workers: does the real wasm **blocking futex** (`memory.atomic.wait`/`notify`, called from Rust via
// `core::arch::wasm32`) work across OS threads? A consumer Worker `wait`s on a flag in the ONE shared
// memory; once it is provably parked, a producer writes a payload, sets the flag, and `wake`s it. The
// consumer must be woken (`wake` returns 1, wait code 0) and read the payload — a genuine cross-Worker
// park/wake handoff. We also check the two non-blocking outcomes the engine's futex relies on: the
// not-equal fast path and the timeout path.
//
// Usage:  node threads-futex.mjs   (after `cargo +nightly build --release`)
import { readFileSync } from 'node:fs';
import { Worker, isMainThread, workerData, parentPort } from 'node:worker_threads';

const WASM = new URL('./target/wasm32-unknown-unknown/release/threads_spike.wasm', import.meta.url);
const FLAG = 8 * 1024 * 1024; // the futex word (reuse the spike's high counter address)
const PAYLOAD = FLAG + 64; // the handed-over value, a separate cell
const MAGIC = 987654;

// A worker runs one named export over the shared memory and reports its result. The `consume` op
// posts 'ready' immediately before parking, so the main thread can wait until it is genuinely parked.
async function worker() {
  const { module, memory, op } = workerData;
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  if (op === 'consume') {
    parentPort.postMessage('ready'); // next statement blocks in the futex
    const code = ex.wait_eq(FLAG, 0, -1n); // `-1` = wait forever (woken only by the producer)
    parentPort.postMessage({ code, payload: ex.load(PAYLOAD) });
  } else if (op === 'fast') {
    // The flag is already non-zero ⇒ the wait must return 1 (not-equal) without parking.
    parentPort.postMessage({ code: ex.wait_eq(FLAG, 0, -1n) });
  } else if (op === 'timeout') {
    // No producer ⇒ the wait must time out (code 2) after ~50 ms.
    parentPort.postMessage({ code: ex.wait_eq(FLAG, 0, 50_000_000n) });
  }
}

// Spawn a worker; resolve with its (single) result message. `onReady` fires when it posts 'ready'.
function runWorker(module, memory, op, onReady) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), { workerData: { module, memory, op } });
    w.on('message', (m) => {
      if (m === 'ready') return void (onReady && onReady());
      w.terminate().then(() => resolve(m), () => resolve(m));
    });
    w.once('error', reject);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, { env: { memory } });
  const shared = memory.buffer instanceof SharedArrayBuffer;
  console.log(`shared memory: ${memory.buffer.byteLength >> 20} MiB (shared=${shared}), futex @ 0x${FLAG.toString(16)}`);

  // --- 1) park/wake handoff: wake only AFTER the consumer is provably parked ----------------------
  ex.store(FLAG, 0);
  ex.store(PAYLOAD, 0);
  let ready;
  const readyP = new Promise((r) => (ready = r));
  const consumer = runWorker(module, memory, 'consume', ready);
  await readyP; // the consumer posted 'ready' immediately before `wait_eq`
  await sleep(150); // ...and is now parked in the futex
  ex.store(PAYLOAD, MAGIC); // hand over the payload...
  ex.store(FLAG, 1); // ...flip the flag it is parked on...
  const woke = ex.wake(FLAG, 1); // ...and wake it
  const got = await consumer;
  const handoff = woke === 1 && got.code === 0 && got.payload === MAGIC;
  console.log(`  park/wake: woke ${woke}, consumer code=${got.code} payload=${got.payload}  ` +
    `${handoff ? 'PASS ✓ (genuinely parked, then woken)' : 'FAIL ✗'}`);

  // --- 2) not-equal fast path: flag already set ⇒ wait returns 1 immediately, no park --------------
  ex.store(FLAG, 1);
  const fast = (await runWorker(module, memory, 'fast')).code;
  console.log(`  not-equal fast path: code=${fast}  ${fast === 1 ? 'PASS ✓' : 'FAIL ✗'}`);

  // --- 3) timeout path: no producer ⇒ wait returns 2 ----------------------------------------------
  ex.store(FLAG, 0);
  const to = (await runWorker(module, memory, 'timeout')).code;
  console.log(`  timeout path: code=${to}  ${to === 2 ? 'PASS ✓' : 'FAIL ✗'}`);

  const pass = handoff && fast === 1 && to === 2;
  console.log(`\n${pass ? 'PASS' : 'FAIL'}: the wasm blocking futex (memory.atomic.wait/notify) ` +
    `${pass ? 'works' : 'does NOT work'} across Workers`);
  process.exit(pass ? 0 : 1);
}

if (isMainThread) main(); else worker();
