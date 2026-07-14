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
  // Keep the pageerror texts (not just print them): I22 is a rare flake where a worker vCPU's
  // `svm_par_run` takes an uncaught host wasm trap (`memory access out of bounds`, or `unreachable`
  // from a panic=abort engine panic). The rejection never reaches the page, so the item hangs
  // `pending` and the wait below times out — with no clue which check tripped. On timeout we dump
  // both the still-`pending` items and these captured messages so the next recurrence self-identifies.
  const pageErrors = [];
  page.on('console', (m) => console.log(`  [page] ${m.text()}`));
  page.on('pageerror', (e) => { pageErrors.push(e.message); console.log(`  [pageerror] ${e.message}`); });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });

  const WORK_IDS = ['powerbox', 'threads', 'jit', 'inst', 'capio', 'wasmjit', 'tierup'];
  // Wait until every work item leaves 'pending' (or time out).
  try {
    await page.waitForFunction(
      (ids) => ids.every((id) => document.getElementById(id).dataset.status !== 'pending'),
      WORK_IDS, { timeout: 30_000 },
    );
  } catch (e) {
    // Timed out ⇒ ≥1 item never left 'pending' (the I22 hang signature). Report which, plus the
    // uncaught pageerror(s) that most likely caused it, before re-throwing into the outer handler.
    const statuses = await page.evaluate(
      (ids) => ids.map((id) => ({ id, status: document.getElementById(id)?.dataset.status ?? 'missing' })),
      WORK_IDS,
    );
    const stuck = statuses.filter((s) => s.status === 'pending').map((s) => s.id);
    console.log(`  [timeout] items still pending: ${stuck.join(', ') || '(none)'}`);
    console.log(`  [timeout] uncaught pageerror(s): ${pageErrors.length ? pageErrors.join(' | ') : '(none captured)'}`);
    throw e;
  }

  const read = (id) => page.$eval(`#${id}`, (e) => ({ status: e.dataset.status, text: e.textContent }));
  const isolated = await read('isolated');
  const powerbox = await read('powerbox');
  const threads = await read('threads');
  const jit = await read('jit');
  const inst = await read('inst');
  const capio = await read('capio');
  const wasmjit = await read('wasmjit');
  const tierup = await read('tierup');

  console.log(`\n  ${isolated.text}`);
  console.log(`  ${powerbox.text}`);
  console.log(`  ${threads.text}`);
  console.log(`  ${jit.text}`);
  console.log(`  ${inst.text}`);
  console.log(`  ${capio.text}`);
  console.log(`  ${wasmjit.text}`);
  console.log(`  ${tierup.text}\n`);

  const pageOk = isolated.status === 'true' && powerbox.status === 'pass' &&
    threads.status === 'pass' && jit.status === 'pass' && inst.status === 'pass' &&
    capio.status === 'pass' && wasmjit.status === 'pass' && tierup.status === 'pass';

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

  // The framebuffer output path (the `display` capability): the gradient guest presents a 128×128
  // RGBA frame, which play.js blits to the canvas. Assert the canvas got the right dimensions and a
  // pixel matching the guest's analytic gradient — R ramps across X, G down Y (top-left ≈ black).
  {
    const grad = await runPlay('gradient (C → framebuffer)');
    const canvas = await play.evaluate(() => {
      const c = document.getElementById('canvas');
      if (c.style.display === 'none' || !c.width || !c.height) return null;
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      const px = (x, y) => Array.from(d.slice((y * c.width + x) * 4, (y * c.width + x) * 4 + 4));
      return { w: c.width, h: c.height, topLeft: px(0, 0), bottomRight: px(c.width - 1, c.height - 1) };
    });
    const gradOk = grad.state === 'done' && canvas && canvas.w === 128 && canvas.h === 128 &&
      canvas.topLeft[0] === 0 && canvas.topLeft[1] === 0 && canvas.topLeft[3] === 255 &&
      canvas.bottomRight[0] === 255 && canvas.bottomRight[1] === 255 && canvas.bottomRight[3] === 255;
    checks.push(gradOk);
    console.log(`  play/gradient-canvas: state=${grad.state} canvas=${JSON.stringify(canvas)} ` +
      `${gradOk ? 'PASS' : 'FAIL'}`);
  }

  // The reactor run model + `keyboard` capability (Doom slice 2): the bounce guest's exported tick()
  // runs once per requestAnimationFrame — it animates (the box moves frame to frame) and accepts arrow
  // keys through the keyboard capability without trapping. (The exact input→motion mapping is pinned
  // deterministically by the native `reactor.rs` test; here we prove the loop runs in a real browser.)
  {
    // Leftmost x of the amber box (255,220,40) on the canvas, or -1 if not found / no frame yet.
    const boxX = () => play.evaluate(() => {
      const c = document.getElementById('canvas');
      if (!c.width || !c.height) return -1;
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      for (let x = 0; x < c.width; x++)
        for (let y = 0; y < c.height; y++) {
          const i = (y * c.width + x) * 4;
          if (d[i] === 255 && d[i + 1] === 220 && d[i + 2] === 40) return x;
        }
      return -1;
    });
    await play.selectOption('#example', 'bounce (interactive — arrow keys)');
    await play.click('#run');
    await play.waitForFunction(() => document.getElementById('state').dataset.state === 'running',
      { timeout: 30_000 });
    // Wait for the first frame (canvas sized to the guest's 160×120), then sample across ~200ms.
    await play.waitForFunction(() => document.getElementById('canvas').width === 160, { timeout: 5000 });
    const w = await play.$eval('#canvas', (c) => c.width);
    const h = await play.$eval('#canvas', (c) => c.height);
    const a = await boxX();
    await play.waitForTimeout(200);
    const b = await boxX();
    // Deliver arrow keys — the guest must poll them and keep running (no trap).
    for (let i = 0; i < 6; i++) await play.keyboard.press('ArrowLeft');
    await play.waitForTimeout(150);
    const c = await boxX();
    const stateRunning = await play.$eval('#state', (e) => e.dataset.state);
    await play.click('#stop');
    await play.waitForFunction(() => document.getElementById('state').dataset.state === 'stopped',
      { timeout: 5000 });
    const bounceOk = w === 160 && h === 120 && a >= 0 && b >= 0 && c >= 0 && a !== b &&
      stateRunning === 'running';
    checks.push(bounceOk);
    console.log(`  play/bounce-reactor: ${w}×${h} boxX a=${a} b=${b} c=${c} (animates=${a !== b}) ` +
      `${bounceOk ? 'PASS' : 'FAIL'}`);
  }

  // Heap persistence (Doom slice 3): Conway's Game of Life runs its grid in the malloc heap above the
  // mapped window. The glider only advances if the reactor persists the whole guest memory frame to
  // frame — so a moving glider in the browser is the end-to-end heap-persistence proof. Sample the
  // glider's bounding-box top-left: 5 live cells always, position advancing over generations.
  {
    // {count, minX, minY} of the amber live cells (255,200,40) on the canvas.
    const cells = () => play.evaluate(() => {
      const c = document.getElementById('canvas');
      if (!c.width || !c.height) return null;
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let n = 0, minx = 1e9, miny = 1e9;
      for (let y = 0; y < c.height; y++)
        for (let x = 0; x < c.width; x++) {
          const i = (y * c.width + x) * 4;
          if (d[i] === 255 && d[i + 1] === 200 && d[i + 2] === 40) { n++; minx = Math.min(minx, x); miny = Math.min(miny, y); }
        }
      return { w: c.width, h: c.height, n, minx, miny };
    });
    await play.selectOption('#example', 'life (Conway — heap persistence)');
    await play.click('#run');
    await play.waitForFunction(() => document.getElementById('state').dataset.state === 'running',
      { timeout: 30_000 });
    await play.waitForFunction(() => document.getElementById('canvas').width === 96, { timeout: 5000 });
    const a = await cells();
    await play.waitForTimeout(300);
    const b = await cells();
    await play.click('#stop');
    await play.waitForFunction(() => document.getElementById('state').dataset.state === 'stopped',
      { timeout: 5000 });
    // 5 live cells (a glider) throughout, and the bounding-box top-left advanced (heap persisted).
    const lifeOk = a && b && a.w === 96 && a.h === 64 && a.n === 5 && b.n === 5 &&
      (b.minx > a.minx || b.miny > a.miny);
    checks.push(lifeOk);
    console.log(`  play/life-reactor: ${a?.w}×${a?.h} glider a=(${a?.minx},${a?.miny}) ` +
      `b=(${b?.minx},${b?.miny}) live=${a?.n}/${b?.n} ${lifeOk ? 'PASS' : 'FAIL'}`);
  }

  const ok = pageOk && checks.every(Boolean);
  failed = !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}: SVM runs in a real browser — powerbox + genuine multi-Worker ` +
    `parallelism (incl. §22 guest-JIT on a shared Domain, §14 confined executor children on their ` +
    `own Workers, and 4d host I/O from worker vCPUs through one shared powerbox) over a shared ` +
    `WebAssembly.Memory under cross-origin isolation — plus the playground (SVM text parsed ` +
    `in-browser via svm_parse, run across Workers in every powerbox mode) and the wasm-JIT tier ` +
    `(SVM IR compiled to wasm in-browser, f0 called directly, matching the interpreter) — including ` +
    `per-Worker JIT tier-up: a threaded guest's compute leaves run on emitted wasm on their own ` +
    `Workers over the shared memory, byte-identical to the all-interpreter run`);
} catch (e) {
  failed = true;
  console.log(`FAIL: ${e.message}`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
