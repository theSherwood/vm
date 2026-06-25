// THREADS/BROWSER step 4c-wasm — the real-browser proof. Starts the COOP/COEP server, launches the
// preinstalled Chromium via Playwright, loads the page, and asserts the on-page results: the browser
// is cross-origin isolated (SharedArrayBuffer available), the powerbox guest printed "hello,
// powerbox!", and one guest's vCPUs ran across real Web Workers to 4000. This closes the gap between
// "runs on Node worker_threads" and "runs in an actual browser" — the thesis BROWSER.md rests on.
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
// Playwright is globally installed in this environment (not in node_modules); import by absolute path.
const { chromium } = (await import('/opt/node22/lib/node_modules/playwright/index.js')).default;

const { server, port } = await startServer(ROOT);
const browser = await chromium.launch(); // uses the preinstalled Chromium (PLAYWRIGHT_BROWSERS_PATH)
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`  [page] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });

  // Wait until both work items leave 'pending' (or time out).
  await page.waitForFunction(
    () => ['powerbox', 'threads'].every((id) => document.getElementById(id).dataset.status !== 'pending'),
    { timeout: 30_000 },
  );

  const read = (id) => page.$eval(`#${id}`, (e) => ({ status: e.dataset.status, text: e.textContent }));
  const isolated = await read('isolated');
  const powerbox = await read('powerbox');
  const threads = await read('threads');

  console.log(`\n  ${isolated.text}`);
  console.log(`  ${powerbox.text}`);
  console.log(`  ${threads.text}\n`);

  const ok = isolated.status === 'true' && powerbox.status === 'pass' && threads.status === 'pass';
  failed = !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}: SVM runs in a real browser — powerbox + genuine multi-Worker ` +
    `parallelism over a shared WebAssembly.Memory under cross-origin isolation`);
} catch (e) {
  failed = true;
  console.log(`FAIL: ${e.message}`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
