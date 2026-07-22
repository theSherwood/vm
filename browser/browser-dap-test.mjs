// Real-browser (V8) end-to-end for the **DAP-over-bytecode** debugger exposed through the wasm FFI
// (DEBUGGING.md, browser slice). Drives a full DAP conversation against the cdylib's `svm_dap_*`
// entry — the same request→messages logic the native `dap.rs` tests exercise, but marshalled through
// wasm from JS: initialize → launch (engine "bytecode") → setBreakpoints → run to the breakpoint →
// stackTrace → variables → continue to termination. Asserts the source frame binds and the loop
// locals read back correctly, proving the debugger runs on the engine the playground ships.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
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

// A tiny SVM guest with a §6 debug section: a source line at the loop body (sum.c:7) and the two loop
// variables mapped to their SSA slots — the same fixture the native DAP tests use.
const LOOP_SUM_DBG = `
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}

debug.file 0 "sum.c"
debug.fname 0 "sum"
debug.loc 0 1 0 0 7 5
debug.var 0 "i" ssa 0 "int"
debug.var 0 "acc" ssa 1 "int"
`;

const res = await page.evaluate(async (src) => {
  const par = await import('./par.js');
  const { createDapClient } = await import('./dap.js');
  const eng = await par.loadEngine();
  const dap = createDapClient(eng.ex, eng.memory);
  const ev = (r, name) => r.events.find((e) => e.event === name);

  // initialize → launch on the bytecode engine → breakpoint on the loop body.
  dap.send('initialize', {});
  const launch = dap.send('launch', {
    programText: src, function: 0, args: [3], engine: 'bytecode',
  });
  const launched = launch.response.success === true;
  const bpReply = dap.send('setBreakpoints', {
    source: { path: '/work/sum.c' },
    breakpoints: [{ line: 7 }],
  });
  const bp0 = bpReply.response.body.breakpoints[0];

  // Run to the breakpoint; read the frame + the loop locals across a couple of iterations.
  const cfg = dap.send('configurationDone', {});
  const firstReason = ev(cfg, 'stopped')?.body?.reason;

  const readLocals = () => {
    const st = dap.send('stackTrace', { threadId: 1 });
    const top = st.response.body.stackFrames[0];
    const sc = dap.send('scopes', { frameId: top.id });
    const vref = sc.response.body.scopes[0].variablesReference;
    const vars = dap.send('variables', { variablesReference: vref }).response.body.variables;
    const map = Object.fromEntries(vars.map((v) => [v.name, v.value]));
    return { name: top.name, line: top.line, i: map.i, acc: map.acc };
  };
  const first = readLocals();

  // Continue through the loop, collecting (i, acc) at each hit, until the guest terminates.
  const iters = [{ i: first.i, acc: first.acc }];
  let terminated = false;
  for (let k = 0; k < 50; k++) {
    const c = dap.send('continue', {});
    if (ev(c, 'terminated')) { terminated = true; break; }
    if (ev(c, 'stopped')) { const l = readLocals(); iters.push({ i: l.i, acc: l.acc }); }
  }

  return {
    launched,
    bpVerified: bp0.verified === true,
    bpLine: bp0.line,
    firstReason,
    frameName: first.name,
    frameLine: first.line,
    iters,
    terminated,
  };
}, LOOP_SUM_DBG);

console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
await browser.close(); server.close();

const ok =
  errors.length === 0 &&
  res.launched &&
  res.bpVerified && res.bpLine === 7 &&
  res.firstReason === 'breakpoint' &&
  res.frameName === '#0 sum' && res.frameLine === 7 &&
  res.iters[0] && res.iters[0].i === '3' && res.iters[0].acc === '0' &&
  res.iters.length >= 3 &&
  res.terminated;

console.log(`  launch(engine=bytecode): ${res.launched}`);
console.log(`  breakpoint sum.c:7 bound: ${res.bpVerified} (line ${res.bpLine})`);
console.log(`  stopped: ${res.firstReason} at ${res.frameName}:${res.frameLine}`);
console.log(`  loop locals (i,acc): ${res.iters.map((x) => `(${x.i},${x.acc})`).join(' ')}`);
console.log(`  terminated: ${res.terminated}`);
console.log(ok ? 'PASS — DAP debugger runs on the bytecode engine in the browser' : 'FAIL');
process.exit(ok ? 0 : 1);
