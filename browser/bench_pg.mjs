// V8 (Node) driver: **boot PostgreSQL --single inside wasm** and measure it. Loads the SVM bytecode
// engine compiled to wasm (the `svm-browser` cdylib), the pre-translated+resolved Postgres module, and
// the data-image blob; grants an in-memory `fs` cap over the image; feeds the SQL on stdin; prints the
// backend's stdout and the guest-boot-in-wasm wall time — the direct measurement BOOTSPEED.md wanted.
//
//   node bench_pg.mjs <svm_browser.wasm> <postgres_resolved.svmb> <pgdata.img> [sql-file]
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';

const [wasmPath, modPath, imgPath, sqlPath] = process.argv.slice(2);
if (!imgPath) {
  console.error('usage: node bench_pg.mjs <svm_browser.wasm> <module.svmb> <pgdata.img> [sql-file]');
  process.exit(2);
}
const sql = sqlPath
  ? readFileSync(sqlPath)
  : Buffer.from('CREATE TABLE t (x int, s text);\nINSERT INTO t VALUES (1,\'one\');\nINSERT INTO t VALUES (2,\'two\');\nSELECT * FROM t ORDER BY x DESC;\nSELECT count(*), sum(x) FROM t;\n');

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, engineImports())).exports;
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));

// Load module + image + stdin into the wasm linear memory. Alloc all first (alloc can grow memory),
// then copy each through a fresh view.
const modBytes = readFileSync(modPath);
const imgBytes = readFileSync(imgPath);
const modPtr = ex.svm_alloc(N(modBytes.length));
const imgPtr = ex.svm_alloc(N(imgBytes.length));
const inPtr = ex.svm_alloc(N(sql.length));
const put = (ptr, bytes) => new Uint8Array(ex.memory.buffer).set(bytes, Number(ptr));
put(modPtr, modBytes);
put(imgPtr, imgBytes);
put(inPtr, sql);

console.error(`module ${modBytes.length} B, image ${imgBytes.length} B, sql ${sql.length} B — booting…`);
const t0 = performance.now();
const rv = ex.svm_run_pg(modPtr, N(modBytes.length), imgPtr, N(imgBytes.length), inPtr, N(sql.length));
const ms = performance.now() - t0;

const status = ex.svm_status();
const exit = ex.svm_exit_code();
const outLen = Number(ex.svm_stdout_len());
const outPtr = Number(ex.svm_stdout_ptr());
const stdout = Buffer.from(new Uint8Array(ex.memory.buffer, outPtr, outLen)).toString();

console.log('=== stdout ===');
console.log(stdout);
console.log(`=== status=${status} exit=${exit} rv=${rv} boot=${ms.toFixed(0)} ms ===`);
// status: 0=OK 1=decode 2=unsupported 3=trap 4=bad-result 5=EXIT(clean) 6=verify.
process.exit(status === 0 || status === /*EXIT*/ 5 ? 0 : 1);
