// V8 (Node) end-to-end check of the **persistent Postgres session across a snapshot/reboot**, driving
// the *shipping wasm artifact* through the exact FFI the browser console uses (`svm_pg_open` /
// `svm_pg_query` / `svm_pg_snapshot` / `svm_pg_close`). The native `pg_snapshot_roundtrip` test proves
// the Rust logic; this proves the same round-trip survives the wasm boundary (reserved-window engine,
// shared linear memory, the snapshot buffer ABI `play.js` reads).
//
//   node pg_snapshot_test.mjs [svm_browser.wasm] [postgres_resolved.svmb] [pgdata.img]
//
// Boots the backend twice (a few seconds each). Exits 0 on success, 1 on any mismatch.
import { readFileSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/postgres_resolved.svmb';
const imgPath = process.argv[4] ?? 'web/assets/pgdata.img';

const mod = await WebAssembly.compile(readFileSync(wasmPath));
// The threads build imports its linear memory as a shared memory (same shape as web/par.js). A plain
// build owns its own memory and ignores the import; pass one unconditionally so both work.
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const ex = (await WebAssembly.instantiate(mod, engineImports(memory))).exports;
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));
const put = (bytes) => {
  const p = ex.svm_alloc(N(bytes.length));
  new Uint8Array(memory.buffer).set(bytes, Number(p));
  return p;
};
const readStdout = () => {
  const p = Number(ex.svm_stdout_ptr());
  const l = Number(ex.svm_stdout_len());
  return p && l ? Buffer.from(new Uint8Array(memory.buffer, p, l)).toString() : '';
};
const fail = (m) => {
  console.error(`FAIL: ${m}`);
  process.exit(1);
};

const modBytes = readFileSync(modPath);
function open(imgBytes) {
  const modP = put(modBytes);
  const imgP = put(imgBytes);
  const rc = ex.svm_pg_open(modP, N(modBytes.length), imgP, N(imgBytes.length));
  ex.svm_dealloc(modP, N(modBytes.length));
  ex.svm_dealloc(imgP, N(imgBytes.length));
  return Number(rc);
}
function query(sql) {
  const b = Buffer.from(sql.endsWith('\n') ? sql : sql + '\n');
  const p = put(b);
  const rc = Number(ex.svm_pg_query(p, N(b.length)));
  ex.svm_dealloc(p, N(b.length));
  return { rc, out: readStdout() };
}

// 1) Boot from the pristine image.
if (open(readFileSync(imgPath)) !== 0) fail(`initial boot: status ${ex.svm_status()}`);
if (!readStdout().includes('backend>')) fail('no prompt after boot');
console.error('booted pristine backend');

// 2) Create + insert a sentinel row.
if (query('CREATE TABLE persist_probe (x int);').rc !== 0) fail(`CREATE: status ${ex.svm_status()}`);
if (query('INSERT INTO persist_probe VALUES (424242);').rc !== 0) fail(`INSERT: status ${ex.svm_status()}`);
console.error('created table + inserted row');

// 3) Snapshot the live data dir, copy the image out, tear the backend down.
if (Number(ex.svm_pg_snapshot()) !== 0) fail('snapshot');
const sp = Number(ex.svm_pg_snapshot_ptr());
const sl = Number(ex.svm_pg_snapshot_len());
if (!sp || !sl) fail('empty snapshot image');
const snapshot = Buffer.from(new Uint8Array(memory.buffer, sp, sl)); // copy before reopen
console.error(`snapshot image ${sl} B`);
ex.svm_pg_close();

// 4) Reboot a fresh backend from the snapshot (recovery runs here).
if (open(snapshot) !== 0) fail(`reboot from snapshot: status ${ex.svm_status()}`);
console.error('rebooted from snapshot');

// 5) The row survives.
const { rc, out } = query('SELECT x FROM persist_probe;');
if (rc !== 0) fail(`SELECT after reboot: status ${ex.svm_status()}`);
if (out.includes('ERROR')) fail(`SELECT errored:\n${out}`);
if (!out.includes('424242')) fail(`row did not survive; SELECT output:\n${out}`);
ex.svm_pg_close();

console.log('OK: Postgres session survived snapshot → close → reboot (row 424242 recovered)');
