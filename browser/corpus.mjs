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
const mem = () => new Uint8Array(ex.memory.buffer); // re-fetch (memory may grow)
const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
const fromHex = (s) => Uint8Array.from(s.match(/../g) ?? [], (h) => parseInt(h, 16));
let total = 0, fail = 0;

// ---- compute corpus: svm_run / svm_run0 vs native -------------------------------------------
for (const { name, file, nargs, cases } of corpus.compute) {
  const bytes = readFileSync(file);
  if (bytes.length > cap) throw new Error(`${file} > buf cap`);
  let bad = 0;
  for (const { arg, status, value } of cases) {
    mem().set(bytes, bufPtr); // re-seed (engine may dirty the window)
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

// ---- powerbox corpus: svm_run_pb (streams/clock/exit) vs native -----------------------------
for (const c of corpus.powerbox ?? []) {
  const bytes = readFileSync(c.file);
  if (bytes.length > cap) throw new Error(`${c.file} > buf cap`);
  mem().set(bytes, bufPtr);
  // Seed stdin into its own scratch buffer.
  const stdin = fromHex(c.stdin);
  mem().set(stdin, Number(ex.svm_stdin_buf()));
  ex.svm_set_stdin_len(N(stdin.length));
  const got = ex.svm_run_pb(N(bytes.length));
  const gotStatus = ex.svm_status();
  // Read captured streams back out of guest memory.
  const rd = (ptr, len) => hex(mem().slice(Number(ptr), Number(ptr) + Number(len)));
  const gotOut = rd(ex.svm_stdout_ptr(), ex.svm_stdout_len());
  const gotErr = rd(ex.svm_stderr_ptr(), ex.svm_stderr_len());
  const gotExit = ex.svm_exit_code();
  const okStatus = gotStatus === c.status;
  const okValue = c.status !== 0 || BigInt(got) === BigInt(c.value);
  const okExit = c.status !== 5 || gotExit === c.exit;
  const okOut = gotOut === c.stdout, okErr = gotErr === c.stderr;
  total++;
  if (!(okStatus && okValue && okExit && okOut && okErr)) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value},exit:${c.exit},` +
      `stdout:${c.stdout},stderr:${c.stderr}} wasm {status:${gotStatus},value:${got},` +
      `exit:${gotExit},stdout:${gotOut},stderr:${gotErr}}`);
  } else {
    const dec = (h) => JSON.stringify(Buffer.from(fromHex(h)).toString());
    const detail = c.status === 5 ? `exit ${gotExit}`
      : gotOut || gotErr ? `out=${dec(gotOut)}${gotErr ? ` err=${dec(gotErr)}` : ''}`
      : `value ${got}`;
    console.log(`  ${c.name}: match (${detail})`);
  }
}

console.log(`\n${total - fail}/${total} cases match native  ${fail ? 'FAILED' : 'ALL MATCH'}`);
process.exit(fail ? 1 : 0);
