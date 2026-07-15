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
    // Warm up past the cold regime: V8 must baseline- then optimizing-compile the ~1000-function
    // emitted module, and Doom must run its title screen into the attract-mode demo (the heavy 3D
    // render path). 20 frames is too few — the first Chromium launch can still be in liftoff, which
    // is what made this bench occasionally report a cold-start fluke; 300 frames is safely warm.
    for (let i = 0; i < 300; i++) reactor.frame();
    // Per-frame times over ~10s — long enough to span the title→demo→title cycle, so the tail
    // captures the expensive render frames, not just the cheap static menu.
    const times = [];
    const tEnd = performance.now() + 10000;
    while (performance.now() < tEnd) { const a = performance.now(); reactor.frame(); times.push(performance.now() - a); }
    reactor.close();
    times.sort((x, y) => x - y);
    const n = times.length;
    const pct = (p) => times[Math.min(n - 1, Math.floor((p / 100) * n))];
    const mean = times.reduce((s, t) => s + t, 0) / n;
    return { tps: n / (times.reduce((s, t) => s + t, 0) / 1000), meanMs: mean, p50: pct(50), p99: pct(99), max: times[n - 1],
             overVsync: 100 * times.filter((t) => t > 16.7).length / n };
  });
  console.log(`\n  wasm-JIT tick throughput (no rAF): ${res.tps.toFixed(0)} ticks/sec`);
  console.log(`  per-frame ms: mean ${res.meanMs.toFixed(2)}  p50 ${res.p50.toFixed(2)}  p99 ${res.p99.toFixed(2)}  max ${res.max.toFixed(1)}`);
  console.log(`  frames slower than 16.7ms (60fps): ${res.overVsync.toFixed(2)}%`);
  console.log(res.tps > 60 ? '  → compute far exceeds 60 fps; the display loop is vsync-capped (smooth)'
    : '  → below 60 fps; compute-bound (investigate)');
  process.exitCode = res.tps > 60 ? 0 : 1;
} finally {
  await browser.close();
  server.close();
}
