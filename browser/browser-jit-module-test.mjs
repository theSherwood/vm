// Real-browser (V8) differential + measurement for the **single-shot module wasm-JIT**: for each
// on-ramp module (hello_c / Lua / SQLite), run the SAME guest through the interpreter (`svm_run_onramp`)
// and the emitted-wasm tier (`runJitModule`), assert stdout is BYTE-IDENTICAL, and print interp-vs-JIT
// timing. V8 (unlike wasmi) compiles Lua/SQLite's huge functions, so this is where the module JIT is
// exercised. Lua/SQLite assets are built on demand (gitignored); absent ones are skipped.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// hello_c is committed; Lua/SQLite are built on demand — only test the ones present.
const CASES = [
  { name: 'hello_c', stdin: '' },
  { name: 'lua_eval', stdin: 'local s=0; for i=1,2000000 do s=s+i end; print(s)\n' },
  { name: 'sqlite_repl', stdin: "CREATE TABLE t(x);\nWITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i<50000) INSERT INTO t SELECT i FROM c;\nSELECT sum(x), count(*) FROM t;\n" },
].filter((c) => existsSync(`${ROOT}/web/assets/${c.name}.svmb`));

const res = await page.evaluate(async (cases) => {
  const par = await import('./par.js');
  const { runJitModule } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const dec = (p, n) => new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  const readStdout = () => dec(Number(eng.ex.svm_stdout_ptr()), eng.ex.svm_stdout_len());

  const out = {};
  for (const { name, stdin } of cases) {
    const bytes = new Uint8Array(await (await fetch(`./assets/${name}.svmb`)).arrayBuffer());
    const stdinBytes = new TextEncoder().encode(stdin);
    // Interpreter oracle
    let interpOut, interpMs;
    {
      const mp = eng.ex.svm_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, mp);
      let sp = 0; if (stdinBytes.length) { sp = eng.ex.svm_alloc(stdinBytes.length); new Uint8Array(eng.memory.buffer).set(stdinBytes, sp); }
      const t0 = performance.now();
      eng.ex.svm_run_onramp(mp, bytes.length, sp, stdinBytes.length);
      interpMs = performance.now() - t0;
      interpOut = readStdout();
      eng.ex.svm_dealloc(mp, bytes.length); if (sp) eng.ex.svm_dealloc(sp, stdinBytes.length);
    }
    // wasm-JIT
    let jitOut, jitMs, err = null, status = null;
    try {
      const t0 = performance.now();
      status = await runJitModule(eng.ex, eng.memory, bytes, stdinBytes);
      jitMs = performance.now() - t0;
      jitOut = readStdout();
    } catch (e) { err = e.message; }
    out[name] = {
      err, status,
      identical: !err && jitOut === interpOut,
      stdout: interpOut.slice(0, 60),
      interpMs: Math.round(interpMs), jitMs: Math.round(jitMs),
      speedup: err ? null : +(interpMs / Math.max(jitMs, 0.01)).toFixed(1),
    };
  }
  return out;
}, CASES);

console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
await browser.close(); server.close();
const names = CASES.map((c) => c.name);
const ok = errors.length === 0 && names.length > 0 && names.every((n) => res[n] && res[n].identical);
for (const n of names) {
  const r = res[n] || {};
  console.log(`  ${n}: ${r.err ? `ERROR ${r.err}` : `stdout≡=${r.identical} · interp ${r.interpMs}ms vs jit ${r.jitMs}ms (${r.speedup}×)`}`);
}
console.log(ok ? 'PASS — module wasm-JIT byte-identical to the interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
