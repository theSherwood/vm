// Raw wasm-JIT tick() throughput — bypasses requestAnimationFrame (which caps the reactor's display
// loop at the monitor refresh, ~60 Hz) to measure how many Doom frames the emitted tick can actually
// compute per second. A number well above 60 means the display loop is vsync-capped, not compute-bound.
import { startServer } from './serve.mjs';
const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('state').dataset.state === 'ready', { timeout: 30_000 });
  const res = await page.evaluate(async () => {
    const { loadEngine } = await import('/web/par.js');
    const { openJitReactor } = await import('/web/wasmjit-reactor.js');
    const eng = await loadEngine();
    const doom = new Uint8Array(await (await fetch('/web/assets/doom.svmb')).arrayBuffer());
    const wad = new Uint8Array(await (await fetch('/web/assets/doom1.wad')).arrayBuffer());
    const reactor = await openJitReactor(eng.ex, eng.memory, doom, 'doom1.wad', wad);
    for (let i = 0; i < 20; i++) reactor.frame();               // warm up (V8 tier-up)
    const t0 = performance.now(); let n = 0;
    while (performance.now() - t0 < 3000) { reactor.frame(); n++; }
    const secs = (performance.now() - t0) / 1000;
    reactor.close();
    return { tps: n / secs };
  });
  console.log(`\n  raw tick throughput (no rAF): ${res.tps.toFixed(0)} ticks/sec  (${(1000 / res.tps).toFixed(1)} ms/frame)`);
  console.log(res.tps > 60 ? '  → compute exceeds 60 fps; the display loop is vsync-capped (smooth)'
    : '  → below 60 fps; compute-bound (investigate)');
  process.exitCode = res.tps > 60 ? 0 : 1;
} finally {
  await browser.close();
  server.close();
}
