// Chromium smoke for the playground's per-demo card layout + CodeMirror editor (BROWSER.md §
// playground). Drives the real page: the sidebar lists every demo, each demo is a self-contained card
// (own editor + controls + output), SVM text highlights, a demo runs end-to-end, the editable-module
// stdin path reads its card's editor, parse errors pin the offending line, and Vim mode engages.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   node browser-play-editor-test.mjs
import { startServer } from './serve.mjs';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The Lua/SQLite `.svmb` guests are built by `build-onramp-assets.mjs`, which the CI real-browser job
// doesn't run (only committed assets are present there). So the editable-module stdin check only runs
// when the Lua asset is actually built — otherwise it's SKIPped, not failed.
const HERE = dirname(fileURLToPath(import.meta.url));
const luaBuilt = existsSync(join(HERE, 'web', 'assets', 'lua_eval.svmb'));

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

// A demo card is addressed by its data-demo attribute (the exact EXAMPLES key).
const card = (name) => `[data-demo="${name}"]`;
const runCard = async (page, name, timeout = 20_000) => {
  await page.click(`${card(name)} .run`);
  await page.waitForFunction(
    (sel) => ['done', 'error', 'stopped'].includes(document.querySelector(sel).dataset.state),
    `${card(name)} .state`, { timeout },
  );
};

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.getElementById('engine-state').dataset.state === 'ready',
    { timeout: 30_000 },
  );

  // The sidebar lists every demo, and every editable demo mounted a CodeMirror editor; the Vim keymap
  // (a vendored bundle script) actually loaded.
  const layout = await page.evaluate(() => ({
    demos: document.querySelectorAll('main#demos .demo').length,
    navLinks: document.querySelectorAll('#nav-list .nav-link').length,
    editors: document.querySelectorAll('.CodeMirror').length,
    vim: typeof window.CodeMirror?.keyMap?.vim,
  }));
  layout.demos > 0 && layout.navLinks === layout.demos && layout.editors > 0 && layout.vim === 'object'
    ? ok(`${layout.demos} demo cards, ${layout.navLinks} nav links, ${layout.editors} editors, vim keymap`)
    : fail(`layout: ${JSON.stringify(layout)}`);

  // The hello card is SVM text → the custom mode highlights keywords, opcodes, and types.
  const tok = await page.evaluate((sel) => ({
    kw: !!document.querySelector(`${sel} .cm-keyword`),
    bi: !!document.querySelector(`${sel} .cm-builtin`),
    ty: !!document.querySelector(`${sel} .cm-type`),
  }), card('hello'));
  tok.kw && tok.bi && tok.ty ? ok('SVM syntax highlighting active') : fail(`SVM tokens: ${JSON.stringify(tok)}`);

  // Running the hello card reads its editor and completes (its 14-byte greeting length).
  await runCard(page, 'hello');
  const hello = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    result: document.querySelector(`${sel} .result`).textContent.trim(),
  }), card('hello'));
  hello.state === 'done' && hello.result === '14'
    ? ok('SVM text ran via the editor → 14')
    : fail(`hello run: ${JSON.stringify(hello)}`);

  // The Lua card mounted a Lua-mode editor with the Lua source…
  const lua = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return { mode: cm?.getOption('mode'), hasPrint: (cm?.getValue() || '').includes('print(') };
  }, card('Lua (5.4.7 — write & run)'));
  lua.mode === 'lua' && lua.hasPrint ? ok('Lua card → lua mode') : fail(`Lua card: ${JSON.stringify(lua)}`);

  // …and running it feeds the card's editor contents to the guest as stdin (when the asset is built).
  if (luaBuilt) {
    await runCard(page, 'Lua (5.4.7 — write & run)', 30_000);
    const luaOut = await page.evaluate((sel) => document.querySelector(`${sel} .stdout`).textContent,
      card('Lua (5.4.7 — write & run)'));
    luaOut.includes('Hello from Lua') ? ok('editable-module stdin reads the card editor') : fail(`Lua stdout: ${luaOut.slice(0, 80)}`);
  } else {
    console.log('  SKIP: editable-module stdin (lua_eval.svmb not built — run build-onramp-assets.mjs)');
  }

  // The SQL card mounted a SQL-mode editor.
  const sqlMode = await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`)?.CodeMirror?.getOption('mode'),
    card('SQLite (:memory: — write & run SQL)'));
  sqlMode === 'text/x-sql' ? ok('SQL card → sql mode') : fail(`SQL mode: ${sqlMode}`);

  // A parse error pins the offending line in that card's editor: a bad opcode on line 3 (unique token).
  await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
    'func () -> (i64) {\nblock 0 () {\n  v0 = i64.notanopcode 1\n  return v0\n  }\n}'), card('hello'));
  await runCard(page, 'hello');
  const mark = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`).CodeMirror;
    const info = cm.lineInfo(2); // 0-based line 2 = the bad-opcode line
    return {
      gutter: !!(info.gutterMarkers && info.gutterMarkers['svm-error-gutter']),
      lineClass: (info.bgClass || '').includes('cm-error-line'),
      widget: !!document.querySelector(`${sel} .cm-error-widget`),
    };
  }, card('hello'));
  (mark.gutter && mark.lineClass && mark.widget)
    ? ok('parse error pinned to the right line (gutter + line + inline message)')
    : fail(`error decoration: ${JSON.stringify(mark)}`);
  // Editing clears the decoration.
  await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
    'func () -> (i64) {\nblock 0 () {\n  v0 = i64.const 1\n  return v0\n  }\n}'), card('hello'));
  const cleared = await page.evaluate((sel) => !document.querySelector(`${sel} .cm-error-widget`), card('hello'));
  cleared ? ok('error decoration clears on edit') : fail('error decoration not cleared on edit');

  // Phase 3/4: every JIT-emittable demo exposes the wasm-JIT toggle + a "Prove it" button — both the
  // interactive reactors (per-frame tick) and the run-to-completion modules (whole _start). Running the
  // parity check confirms the interpreter and wasm-JIT tiers are byte-identical.
  const jitCards = await page.evaluate(() =>
    [...document.querySelectorAll('.demo')].filter((d) => d.querySelector('.jit-label')).map((d) => d.dataset.demo));
  const hasReactorJit = jitCards.includes('bounce (interactive — arrow keys)')
    && jitCards.includes('life (Conway — heap persistence)');
  const hasModuleJit = jitCards.includes('hello (C → SVM)')
    && jitCards.includes('SQLite (:memory: — write & run SQL)');
  hasReactorJit && hasModuleJit
    ? ok(`wasm-JIT toggle on ${jitCards.length} demos (reactors + hello/Lua/SQLite modules)`)
    : fail(`jit cards: ${JSON.stringify(jitCards)}`);

  // The hello module card runs end-to-end via runModule (JIT toggle default-on): this exercises the
  // streamed module fetch (download-progress path) and the single-shot module JIT in CI, since
  // hello_c.svmb is committed. Runs before the module parity check so the asset streams fresh (uncached).
  await runCard(page, 'hello (C → SVM)');
  const helloMod = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card('hello (C → SVM)'));
  helloMod.state === 'done' && helloMod.stdout.length > 0
    ? ok(`hello module ran end-to-end (${JSON.stringify(helloMod.stdout.trim().slice(0, 20))})`)
    : fail(`hello module run: ${JSON.stringify(helloMod)}`);

  // Prove interp ≡ JIT on the bounce reactor (committed asset, fast) — framebuffer byte-identical.
  await page.click(`${card('bounce (interactive — arrow keys)')} .prove`);
  await page.waitForFunction(
    (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
    `${card('bounce (interactive — arrow keys)')} .state`, { timeout: 30_000 });
  const parity = await page.evaluate((sel) => document.querySelector(sel).textContent,
    `${card('bounce (interactive — arrow keys)')} .state`);
  parity.includes('interpreter ≡ wasm-JIT') && parity.includes('byte-identical')
    ? ok(`reactor parity proven in-page: ${parity}`)
    : fail(`parity: ${parity}`);

  // Prove interp ≡ JIT on the hello module (committed asset): the whole _start runs on both tiers and
  // the captured stdout is byte-identical (the module twin of the reactor's per-frame parity).
  await page.click(`${card('hello (C → SVM)')} .prove`);
  await page.waitForFunction(
    (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
    `${card('hello (C → SVM)')} .state`, { timeout: 30_000 });
  const modParity = await page.evaluate((sel) => document.querySelector(sel).textContent,
    `${card('hello (C → SVM)')} .state`);
  modParity.includes('interpreter ≡ wasm-JIT') && modParity.includes('byte-identical stdout')
    ? ok(`module parity proven in-page: ${modParity}`)
    : fail(`module parity: ${modParity}`);

  // Touch dpad: every reactor card carries an on-screen dpad (4 arrows + fire/use/enter/esc) so the
  // interactive guests are playable without a physical keyboard. Structural check (CSS gates visibility
  // to touch/narrow screens); pressing a key while no reactor runs is a guarded no-op.
  const dpad = await page.evaluate((sel) => {
    const d = document.querySelector(`${sel} .dpad`);
    return { present: !!d, keys: d ? d.querySelectorAll('.dkey').length : 0 };
  }, card('bounce (interactive — arrow keys)'));
  dpad.present && dpad.keys === 8
    ? ok(`touch dpad on reactor cards (${dpad.keys} keys)`) : fail(`dpad: ${JSON.stringify(dpad)}`);

  // The Vim toggle engages the Vim keymap on the editors (registered + editor holds vim state).
  await page.check('#vim');
  const vim = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return { opt: cm?.getOption('keyMap'), state: !!cm?.state?.vim };
  }, card('hello'));
  vim.opt === 'vim' && vim.state ? ok('vim mode engaged') : fail(`vim: ${JSON.stringify(vim)}`);

  // Phase 4: an edit persists under the card slug and survives a reload; Reset restores the default and
  // clears storage; a Share permalink round-trips the editor contents through the URL hash.
  const sel = card('hello');
  const setCM = (s, v) => page.evaluate(([s, v]) => document.querySelector(`${s} .CodeMirror`).CodeMirror.setValue(v), [s, v]);
  const getCM = (s) => page.evaluate((s) => document.querySelector(`${s} .CodeMirror`).CodeMirror.getValue(), s);
  const waitReady = () => page.waitForFunction(
    () => document.getElementById('engine-state').dataset.state === 'ready', { timeout: 30_000 });

  await setCM(sel, 'PERSIST_SENTINEL');
  const saved = await page.evaluate(() => localStorage.getItem('svm-play:src:hello'));
  saved === 'PERSIST_SENTINEL' ? ok('edit persisted to localStorage') : fail(`persist: ${saved}`);
  await page.reload({ waitUntil: 'load' });
  await waitReady();
  (await getCM(sel)) === 'PERSIST_SENTINEL'
    ? ok('editor restored from localStorage after reload') : fail('editor not restored after reload');
  await page.click(`${sel} .reset`);
  const afterReset = await page.evaluate((s) => ({
    val: document.querySelector(`${s} .CodeMirror`).CodeMirror.getValue(),
    stored: localStorage.getItem('svm-play:src:hello'),
  }), sel);
  (afterReset.val !== 'PERSIST_SENTINEL' && afterReset.val.includes('cap.call') && afterReset.stored === null)
    ? ok('Reset restores the default source and clears storage')
    : fail(`reset: ${JSON.stringify({ v: afterReset.val.slice(0, 30), s: afterReset.stored })}`);

  // Share: the button emits a permalink; navigating to it (with storage cleared) restores the source
  // purely from the `#demo=…&src=…` hash.
  await setCM(sel, 'SHARED_ROUNDTRIP_42');
  await page.click(`${sel} .share`);
  await page.waitForFunction((s) => /#demo=/.test(document.querySelector(`${s} .log`).textContent), sel, { timeout: 5_000 });
  const shareURL = await page.evaluate((s) => {
    const m = document.querySelector(`${s} .log`).textContent.match(/https?:\/\/\S+#demo=\S+/);
    return m ? m[0] : null;
  }, sel);
  if (shareURL && shareURL.includes('demo=hello')) {
    await page.evaluate(() => localStorage.removeItem('svm-play:src:hello')); // prove the hash, not storage
    await page.goto(shareURL, { waitUntil: 'load' });
    await waitReady();
    (await getCM(sel)) === 'SHARED_ROUNDTRIP_42'
      ? ok('share permalink round-trips the editor via the URL hash') : fail('share permalink did not restore');
  } else {
    fail(`share URL not emitted: ${shareURL}`);
  }

  // The DAP debugger card: a breakpoint is pre-placed, Debug pauses on the bytecode engine at the
  // source line (highlighted), the Variables pane shows the loop locals, and Continue advances the loop.
  const dbgCard = card('Debugger (SVM — breakpoints, step, variables)');
  const dbg0 = await page.evaluate((sel) => ({
    hasDebugBtn: !!document.querySelector(`${sel} .debug`),
    bpDots: document.querySelectorAll(`${sel} .cm-bp-marker`).length,
  }), dbgCard);
  dbg0.hasDebugBtn && dbg0.bpDots === 1
    ? ok(`debug card: Debug button + ${dbg0.bpDots} pre-placed breakpoint`)
    : fail(`debug card: ${JSON.stringify(dbg0)}`);

  // Start debugging → it runs to the breakpoint and pauses.
  await page.click(`${dbgCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    dbgCard, { timeout: 20_000 });
  const paused = await page.evaluate((sel) => ({
    active: document.querySelector(`${sel} .dbg`).classList.contains('active'),
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    readonly: document.querySelector(`${sel} .CodeMirror`).CodeMirror.getOption('readOnly'),
  }), dbgCard);
  paused.active && paused.stopLine && /i\s*=\s*5/.test(paused.vars) && /acc\s*=\s*0/.test(paused.vars) && paused.readonly
    ? ok(`debugger paused at the breakpoint — i=5, acc=0, line highlighted`)
    : fail(`debug paused: ${JSON.stringify(paused)}`);

  // Continue once → next loop iteration: acc accumulates (5), i decrements (4).
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /acc\s*=\s*5/.test(document.querySelector(`${sel} .dbg-vars`).textContent),
    dbgCard, { timeout: 10_000 });
  const stepped = await page.evaluate((sel) => document.querySelector(`${sel} .dbg-vars`).textContent, dbgCard);
  /i\s*=\s*4/.test(stepped) && /acc\s*=\s*5/.test(stepped)
    ? ok('Continue advanced the loop — i=4, acc=5')
    : fail(`debug continue: ${stepped}`);

  // Reverse: run backward to the previous breakpoint hit (deterministic replay) — the locals rewind.
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="reverseContinue"]`);
  await page.waitForFunction((sel) => /i\s*=\s*5/.test(document.querySelector(`${sel} .dbg-vars`).textContent),
    dbgCard, { timeout: 10_000 });
  const reversed = await page.evaluate((sel) => document.querySelector(`${sel} .dbg-vars`).textContent, dbgCard);
  /i\s*=\s*5/.test(reversed) && /acc\s*=\s*0/.test(reversed)
    ? ok('Reverse walked back to the previous breakpoint — i=5, acc=0')
    : fail(`reverse: ${reversed}`);

  // Stop ends the session: the panel hides and the editor is writable again.
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="stop"]`);
  const ended = await page.evaluate((sel) => ({
    active: document.querySelector(`${sel} .dbg`).classList.contains('active'),
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    readonly: document.querySelector(`${sel} .CodeMirror`).CodeMirror.getOption('readOnly'),
  }), dbgCard);
  !ended.active && !ended.stopLine && !ended.readonly
    ? ok('Stop ended the debug session (panel hidden, editor writable)')
    : fail(`debug stop: ${JSON.stringify(ended)}`);

  // The watchpoint card: a counter at a fixed window address, named `count` by its `debug` section, so
  // the Variables pane can arm a data breakpoint on it. Debug pauses at the pre-placed loop-body
  // breakpoint; clicking `count`'s ● toggle arms the watch; Continue then stops for the data breakpoint.
  const wpCard = card('Debugger (SVM — watchpoints / data breakpoints)');
  await page.click(`${wpCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    wpCard, { timeout: 20_000 });
  const wpPaused = await page.evaluate((sel) => ({
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    // `count` is memory-located ⇒ its ● toggle is enabled (a watchable data breakpoint target).
    toggleEnabled: !!document.querySelector(`${sel} .dbg-vars button[data-watch="count"]:not([disabled])`),
  }), wpCard);
  wpPaused.toggleEnabled && /count\s*=\s*0/.test(wpPaused.vars)
    ? ok('watchpoint card paused — count=0, ● toggle armable')
    : fail(`watchpoint paused: ${JSON.stringify(wpPaused)}`);

  // Arm the data breakpoint on `count`, then Continue → the loop-body store trips it (reason "data
  // breakpoint"), and the ● shows armed (.on).
  await page.click(`${wpCard} .dbg-vars button[data-watch="count"]`);
  const armed = await page.evaluate((sel) =>
    !!document.querySelector(`${sel} .dbg-vars button[data-watch="count"].on`), wpCard);
  armed ? ok('clicking ● armed the data breakpoint on count') : fail('watch toggle did not arm (.on)');
  await page.click(`${wpCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /data breakpoint/.test(document.querySelector(`${sel} .state`).textContent),
    wpCard, { timeout: 10_000 });
  const tripped = await page.evaluate((sel) => document.querySelector(`${sel} .state`).textContent, wpCard);
  ok(`watchpoint tripped — ${tripped.replace(/\s+/g, ' ').trim()}`);
  await page.click(`${wpCard} .dbg-controls button[data-cmd="stop"]`);

  // The threads card: a thread.spawn guest on the multithreaded scheduled bytecode engine. Debug stops
  // in a worker; the Variables pane grows a thread selector (one chip per live vCPU); selecting another
  // thread focuses its stack without resuming; Continue catches the second worker; the guest finishes.
  const thCard = card('Debugger (SVM — threads)');
  await page.click(`${thCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 20_000 });
  const thPaused = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    return {
      count: chips.length,
      selected: chips.filter((b) => b.classList.contains('sel')).map((b) => b.dataset.thread),
      marked: chips.filter((b) => b.textContent.includes('●')).map((b) => b.dataset.thread),
      vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    };
  }, thCard);
  thPaused.count >= 3 && thPaused.selected.length === 1 && thPaused.selected[0] === thPaused.marked[0]
    ? ok(`threads card paused in a worker — ${thPaused.count} thread chips, stopped one selected + marked ●`)
    : fail(`threads paused: ${JSON.stringify(thPaused)}`);

  // Focus a *different* thread (not the stopped one) → its chip becomes selected, without resuming.
  const otherThread = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    const stopped = chips.find((b) => b.classList.contains('sel'))?.dataset.thread;
    return chips.map((b) => b.dataset.thread).find((t) => t !== stopped);
  }, thCard);
  await page.click(`${thCard} .dbg-threads .thr[data-thread="${otherThread}"]`);
  const switched = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  switched === otherThread
    ? ok(`selecting another thread (${otherThread}) focuses its stack without resuming`)
    : fail(`thread switch: selected ${switched}, wanted ${otherThread}`);

  // Continue → the second worker hits the same breakpoint (a distinct thread), still paused.
  await page.click(`${thCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 10_000 });
  const secondThread = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  ok(`threads card: Continue caught the second worker (thread ${secondThread})`);

  // ◀◀ Reverse → deterministic replay walks *backward* to the previous worker breakpoint (an earlier
  // global turn) — the scheduled engine's reverse debugging, in the panel.
  await page.click(`${thCard} .dbg-controls button[data-cmd="reverseContinue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 10_000 });
  const reversedThread = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  reversedThread && reversedThread !== secondThread
    ? ok(`threads card: Reverse walked back to the earlier worker (thread ${reversedThread})`)
    : fail(`threads reverse: landed on ${reversedThread}, expected the earlier worker (not ${secondThread})`);
  await page.click(`${thCard} .dbg-controls button[data-cmd="stop"]`);

  // The wait/notify card: a futex handoff. The worker parks on atomic.wait until the root's notify
  // wakes it; a breakpoint after the wait fires only once woken — proving wait/notify drive under the
  // debug scheduler. Then Continue finishes the handoff.
  const wnCard = card('Debugger (SVM — wait / notify)');
  await page.click(`${wnCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    wnCard, { timeout: 20_000 });
  const wnStopped = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, wnCard);
  wnStopped && wnStopped !== '1'
    ? ok(`wait/notify card: the worker woke and stopped after the wait (thread ${wnStopped})`)
    : fail(`wait/notify paused: selected ${wnStopped}, expected a worker (not the root, 1)`);
  await page.click(`${wnCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    wnCard, { timeout: 10_000 });
  ok('wait/notify card: the handoff finished after resuming');

  // The fibers card: a generator. A breakpoint inside the fiber fires only once cont.resume switches
  // the debugged continuation into it — the debugger follows into the fiber and highlights its line;
  // Continue runs the suspend/resume handoff to completion.
  const fbCard = card('Debugger (SVM — fibers / generators)');
  await page.click(`${fbCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    fbCard, { timeout: 20_000 });
  const fbPaused = await page.evaluate((sel) => ({
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    // The stop is inside the fiber body (line 19) — the frame header names the line.
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
  }), fbCard);
  fbPaused.stopLine && /line 19\b/.test(fbPaused.vars)
    ? ok('fibers card: cont.resume stepped into the fiber — stopped inside it (line 19)')
    : fail(`fibers paused: ${JSON.stringify(fbPaused)}`);
  await page.click(`${fbCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    fbCard, { timeout: 10_000 });
  ok('fibers card: the generator finished (36) after resuming');

  // The fibers+threads card: two workers each run a generator fiber. A breakpoint inside the fiber body
  // fires on a *worker* vCPU (a thread selector appears; the stopped chip is not the root), proving fibers
  // compose with threads under the scheduled debugger. Continue catches the other worker; the run finishes.
  const ftCard = card('Debugger (SVM — fibers + threads)');
  await page.click(`${ftCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 20_000 });
  const ftPaused = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    const stopped = chips.find((b) => b.classList.contains('sel'))?.dataset.thread;
    return {
      count: chips.length,
      stopped,
      stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
      vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    };
  }, ftCard);
  // ≥3 live chips (root + two workers), the stop is inside the fiber (line 37), and the stopped vCPU is a
  // worker (thread id ≠ 1, the root).
  ftPaused.count >= 3 && ftPaused.stopLine && /line 37\b/.test(ftPaused.vars) && ftPaused.stopped !== '1'
    ? ok(`fibers+threads card: a worker (thread ${ftPaused.stopped}) stopped inside its fiber (line 37)`)
    : fail(`fibers+threads paused: ${JSON.stringify(ftPaused)}`);
  await page.click(`${ftCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 10_000 });
  const ftSecond = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, ftCard);
  ftSecond !== ftPaused.stopped
    ? ok(`fibers+threads card: Continue caught the other worker's fiber (thread ${ftSecond})`)
    : fail(`fibers+threads second stop: same thread ${ftSecond}`);
  await page.click(`${ftCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 10_000 });
  ok('fibers+threads card: the run finished (50) after both workers');

  // Theme picker: selecting "dark" forces <html data-theme="dark"> and persists; a reload keeps it.
  await page.selectOption('#theme', 'dark');
  const themed = await page.evaluate(() => ({
    attr: document.documentElement.dataset.theme,
    stored: localStorage.getItem('svm-play:theme'),
  }));
  themed.attr === 'dark' && themed.stored === 'dark'
    ? ok('theme picker forces + persists dark') : fail(`theme: ${JSON.stringify(themed)}`);
  await page.reload({ waitUntil: 'load' });
  await waitReady();
  const themeAfter = await page.evaluate(() => ({
    attr: document.documentElement.dataset.theme,
    sel: document.getElementById('theme').value,
  }));
  themeAfter.attr === 'dark' && themeAfter.sel === 'dark'
    ? ok('theme preference survives a reload (no flash — set in <head>)') : fail(`theme reload: ${JSON.stringify(themeAfter)}`);
} finally {
  await browser.close();
  server.close();
}

if (failed) {
  console.log('\nplay editor smoke FAILED');
  process.exit(1);
}
console.log('\nplay editor smoke passed');
