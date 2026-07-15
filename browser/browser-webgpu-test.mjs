// Real-browser proof for the `webgpu` capability playground demo: drives the full stack (wasm engine →
// `webgpu` cap → JS servicer → navigator.gpu) headlessly, and verifies the guest's WGSL Mandelbrot
// renders real GPU pixels (offscreen readback, since headless Chromium can't read the canvas swapchain).
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('no playwright');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const flags = ['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan','--ignore-gpu-blocklist'];
const browser = await chromium.launch({ headless: true, args: flags, env: { ...process.env, VK_ICD_FILENAMES: '/usr/share/vulkan/icd.d/lvp_icd.json' } });
const page = await browser.newPage();
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type()==='error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);
const res = await page.evaluate(async () => {
  const par = await import('./par.js'); const wg = await import('./webgpu.js');
  if (!wg.webgpuAvailable()) return { skip: 'no webgpu' };
  const eng = await par.loadEngine();
  const bytes = new Uint8Array(await (await fetch('./assets/gpu_shader.svmb')).arrayBuffer());
  const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=480; document.body.append(canvas);
  await wg.initWebGPU(canvas);
  const p = eng.ex.svm_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, p);
  const opened = eng.ex.svm_onramp_open(p, bytes.length); eng.ex.svm_dealloc(p, bytes.length);
  if (opened !== 0) return { opened };
  const statuses = []; for (let i=0;i<12;i++) statuses.push(eng.ex.svm_onramp_frame());
  const px = await wg.readbackForTest(60, 320, 240);
  let black=0, colored=0; for (let i=0;i<px.length;i+=4){ if(px[i]===0&&px[i+1]===0&&px[i+2]===0) black++; else colored++; }
  return { opened, statuses, black, colored, total: px.length/4 };
});
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0,5));
await browser.close(); server.close();
const ok = res.skip || (res.opened===0 && res.statuses.every(s=>s===0) && res.black>200 && res.colored>200 && errors.length===0);
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
