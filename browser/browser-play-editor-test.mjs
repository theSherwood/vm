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
    'func () -> (i64) {\nblock0():\n  v0 = i64.notanopcode 1\n  return v0\n}'), card('hello'));
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
    'func () -> (i64) {\nblock0():\n  v0 = i64.const 1\n  return v0\n}'), card('hello'));
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

  // The Vim toggle engages the Vim keymap on the editors (registered + editor holds vim state).
  await page.check('#vim');
  const vim = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return { opt: cm?.getOption('keyMap'), state: !!cm?.state?.vim };
  }, card('hello'));
  vim.opt === 'vim' && vim.state ? ok('vim mode engaged') : fail(`vim: ${JSON.stringify(vim)}`);
} finally {
  await browser.close();
  server.close();
}

if (failed) {
  console.log('\nplay editor smoke FAILED');
  process.exit(1);
}
console.log('\nplay editor smoke passed');
