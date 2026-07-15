// Doom frame-rate measurement (BROWSER.md § "wasm-JIT tier", slice 5d): boot Doom in the playground
// in a real Chromium, run it with the "wasm-JIT" toggle OFF (interpreter tick) and ON (emitted tick),
// and read the live FPS the page surfaces. Reports both + the speedup. Run: `node bench_doom_jit.mjs`.
import { startServer } from './serve.mjs';

const DOOM = 'DOOM (1993 — arrow keys, Ctrl fires)';
const RUN_MS = 20000; // per tier: a few seconds boot + a window to sample steady-state fps (V8 warmup)

async function loadChromium() {
  for (const spec of ['playwright', 'playwright-core']) {
    try {
      const m = await import(spec);
      const chromium = m.chromium ?? m.default?.chromium;
      if (chromium) return chromium;
    } catch { /* try next */ }
  }
  throw new Error('playwright not found');
}

const chromium = await loadChromium();
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });

// Sample the page's live fps (the state text reads "running (tier) — N fps · …") over `ms`, returning
// the peak steady-state reading (peak ≈ the tier's real ceiling once boot settles).
async function sampleFps(page, ms) {
  const readings = [];
  const t0 = Date.now();
  let last = '';
  while (Date.now() - t0 < ms) {
    const txt = await page.$eval('#state', (e) => e.textContent).catch(() => '');
    const m = txt.match(/([\d.]+) fps/);
    if (m && txt !== last) { readings.push(parseFloat(m[1])); last = txt; }
    await new Promise((r) => setTimeout(r, 250));
  }
  console.log(`    fps trace: ${readings.map((r) => r.toFixed(1)).join(' ')}`);
  return readings.length ? Math.max(...readings) : 0;
}

async function measure(page, jit) {
  await page.selectOption('#example', DOOM);
  if (jit) await page.check('#jit'); else await page.uncheck('#jit');
  await page.click('#run');
  const fps = await sampleFps(page, RUN_MS);
  await page.click('#stop').catch(() => {});
  await new Promise((r) => setTimeout(r, 500));
  return fps;
}

try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (/JIT|reactor|fps|fail|error/i.test(m.text())) console.log(`  [play] ${m.text()}`); });
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('state').dataset.state === 'ready', { timeout: 30_000 });

  const interp = await measure(page, false);
  const jitted = await measure(page, true);

  console.log(`\n  DOOM frame rate — interpreter: ${interp.toFixed(1)} fps · wasm-JIT: ${jitted.toFixed(1)} fps`);
  if (interp > 0) console.log(`  speedup: ${(jitted / interp).toFixed(1)}×`);
  // The gate is a *material* win — the emitted tick clearly outruns the interpreter. Full-res (640×400)
  // Doom's renderer is memory-bound (every guest access carries the confinement mask), so the ratio is
  // smaller than register-heavy compute kernels' ~16–112×; smoother (30 fps) would come from a lower
  // render resolution — a doomgeneric asset rebuild, not this wiring slice.
  const ok = interp > 0 && jitted > interp * 2 && jitted > 5;
  console.log(ok ? '  RESULT: wasm-JIT materially faster (emitted tick outruns the interpreter)'
    : '  RESULT: no clear win (investigate)');
  process.exitCode = ok ? 0 : 1;
} finally {
  await browser.close();
  server.close();
}
