// THREADS/BROWSER step 4c-wasm — the real-browser proof. Starts the COOP/COEP server, launches the
// preinstalled Chromium via Playwright, loads the page, and asserts the on-page results: the browser
// is cross-origin isolated (SharedArrayBuffer available), the powerbox guest printed "hello,
// powerbox!", and one guest's vCPUs ran across real Web Workers to 4000. This closes the gap between
// "runs on Node worker_threads" and "runs in an actual browser" — the thesis BROWSER.md rests on.
// Also drives the **playground** (`web/play.html`) end to end: SVM text typed into the editor,
// parsed/verified in-browser (`svm_parse`), run across Workers in every powerbox mode, plus a
// parse-reject negative.
//
// Usage:  node browser-test.mjs            (after building the threads wasm + gencorpus; see below)
//   RUSTFLAGS="-Ctarget-feature=+atomics,+bulk-memory,+mutable-globals \
//     -Clink-arg=--shared-memory -Clink-arg=--import-memory -Clink-arg=--max-memory=1073741824 \
//     -Clink-arg=--export=__stack_pointer -Clink-arg=--export=__tls_base -Clink-arg=--export=__tls_size \
//     -Clink-arg=--export=__tls_align -Clink-arg=--export=__wasm_init_tls" \
//     cargo +nightly build -Z build-std=std,panic_abort --release --lib --target wasm32-unknown-unknown
//   cargo run --bin gencorpus
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { startServer } from './serve.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));

// Resolve Playwright's `chromium` portably: a normal `playwright` resolution (CI installs it locally /
// `npm i playwright`), falling back to this environment's global install by absolute path.
async function loadChromium() {
  const specs = ['playwright', '/opt/node22/lib/node_modules/playwright/index.js'];
  for (const spec of specs) {
    try {
      const m = await import(spec);
      const chromium = m.chromium ?? m.default?.chromium;
      if (chromium) return chromium;
    } catch {
      /* try the next resolution */
    }
  }
  throw new Error("playwright not found — run `npm i playwright && npx playwright install chromium`");
}
const chromium = await loadChromium();

const { server, port } = await startServer(ROOT);
// `--no-sandbox` only under CI (GitHub sets `CI`): the OS process sandbox is unrelated to what we test
// (cross-origin isolation / SharedArrayBuffer), and CI runners often can't enable it. Local stays sandboxed.
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`  [page] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });

  // Wait until every work item leaves 'pending' (or time out).
  await page.waitForFunction(
    () => ['powerbox', 'threads', 'jit', 'inst', 'capio', 'wasmjit'].every((id) => document.getElementById(id).dataset.status !== 'pending'),
    { timeout: 30_000 },
  );

  const read = (id) => page.$eval(`#${id}`, (e) => ({ status: e.dataset.status, text: e.textContent }));
  const isolated = await read('isolated');
  const powerbox = await read('powerbox');
  const threads = await read('threads');
  const jit = await read('jit');
  const inst = await read('inst');
  const capio = await read('capio');
  const wasmjit = await read('wasmjit');

  console.log(`\n  ${isolated.text}`);
  console.log(`  ${powerbox.text}`);
  console.log(`  ${threads.text}`);
  console.log(`  ${jit.text}`);
  console.log(`  ${inst.text}`);
  console.log(`  ${capio.text}`);
  console.log(`  ${wasmjit.text}\n`);

  const pageOk = isolated.status === 'true' && powerbox.status === 'pass' &&
    threads.status === 'pass' && jit.status === 'pass' && inst.status === 'pass' &&
    capio.status === 'pass' && wasmjit.status === 'pass';

  // --- the playground (play.html): SVM text typed into the page, parsed in-browser, run across ----
  // Workers. Drives the page like a human: pick an example / type source, click Run, read the
  // result + stdout panes. Covers every powerbox mode through the `svm_parse` front end, plus a
  // parse-reject negative (garbage source → an error message, not a hang or a crash).
  const play = await browser.newPage();
  play.on('console', (m) => console.log(`  [play] ${m.text()}`));
  play.on('pageerror', (e) => console.log(`  [play pageerror] ${e.message}`));
  await play.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await play.waitForFunction(() => document.getElementById('state').dataset.state === 'ready',
    { timeout: 30_000 });

  const runPlay = async (example) => {
    if (example) await play.selectOption('#example', example);
    await play.click('#run');
    await play.waitForFunction(
      () => ['done', 'error', 'stopped'].includes(document.getElementById('state').dataset.state),
      { timeout: 30_000 },
    );
    return {
      state: await play.$eval('#state', (e) => e.dataset.state),
      status: await play.$eval('#state', (e) => e.textContent),
      result: await play.$eval('#result', (e) => e.textContent),
      stdout: await play.$eval('#stdout', (e) => e.textContent),
    };
  };

  const checks = [];
  const check = (name, got, wantResult, wantStdout = null) => {
    const ok = got.state === (wantResult === null ? 'error' : 'done') &&
      (wantResult === null || got.result === wantResult) &&
      (wantStdout === null || got.stdout === wantStdout);
    checks.push(ok);
    console.log(`  play/${name}: state=${got.state} result=${JSON.stringify(got.result)} ` +
      `stdout=${JSON.stringify(got.stdout)} ${ok ? 'PASS' : 'FAIL'}`);
    return ok;
  };

  check('hello (io)', await runPlay('hello'), '14', 'hello, world!\n');
  check('threads (plain-after-io)', await runPlay('threads'), '4000');
  check('io ticks', await runPlay('io'), '8', 'tick\n'.repeat(8));
  check('jit (§22)', await runPlay('jit'), '1136');
  check('inst (§14)', await runPlay('inst'), '40');

  // Negative: garbage source must come back as a parse error message (state 'error').
  await play.fill('#src', 'func ( this is not svm text');
  const bad = await runPlay(null);
  const badOk = bad.state === 'error' && bad.status.includes('parse error');
  checks.push(badOk);
  console.log(`  play/parse-reject: state=${bad.state} msg=${JSON.stringify(bad.status)} ` +
    `${badOk ? 'PASS' : 'FAIL'}`);

  // An on-ramp module: a real C guest (`hello.c`) compiled through the LLVM on-ramp and run via
  // `svm_run_onramp` (not the text/`svm_parse` path). Uses the committed `web/assets/hello_c.svmb`.
  // Runs last: selecting it makes the source textarea read-only (it's a binary module, not editable
  // SVM text), which would trip the `page.fill` in the parse-reject check above.
  check('hello (C → SVM, on-ramp module)', await runPlay('hello (C → SVM)'), '0', 'hello, sandbox!\n');

  const ok = pageOk && checks.every(Boolean);
  failed = !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}: SVM runs in a real browser — powerbox + genuine multi-Worker ` +
    `parallelism (incl. §22 guest-JIT on a shared Domain, §14 confined executor children on their ` +
    `own Workers, and 4d host I/O from worker vCPUs through one shared powerbox) over a shared ` +
    `WebAssembly.Memory under cross-origin isolation — plus the playground (SVM text parsed ` +
    `in-browser via svm_parse, run across Workers in every powerbox mode) and the wasm-JIT tier ` +
    `(SVM IR compiled to wasm in-browser, f0 called directly, matching the interpreter)`);
} catch (e) {
  failed = true;
  console.log(`FAIL: ${e.message}`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
