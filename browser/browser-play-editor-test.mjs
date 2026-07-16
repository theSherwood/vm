// Chromium smoke for the playground's CodeMirror editor (BROWSER.md § playground). Drives the real
// page: the editor mounts, SVM text highlights, a demo runs end-to-end, the language switches per
// example, the editable-module stdin path reads the editor, and Vim mode actually engages. Catches
// the kind of wiring regression a unit test can't — e.g. a mis-pathed vendored script that 404s.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   node browser-play-editor-test.mjs
import { startServer } from './serve.mjs';

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.getElementById('state').dataset.state === 'ready',
    { timeout: 30_000 },
  );

  // Editor mounted; all vendored scripts (incl. the Vim keymap) actually loaded.
  const mounted = await page.evaluate(
    () => !!document.querySelector('.CodeMirror') && typeof window.CodeMirror?.keyMap?.vim === 'object',
  );
  mounted ? ok('CodeMirror mounted, vim keymap registered') : fail('editor/vim scripts did not load');

  // The default (hello) example is SVM text → the custom mode highlights keywords, opcodes, and types.
  const tok = await page.evaluate(() => ({
    kw: !!document.querySelector('.cm-keyword'),
    bi: !!document.querySelector('.cm-builtin'),
    ty: !!document.querySelector('.cm-type'),
  }));
  tok.kw && tok.bi && tok.ty ? ok('SVM syntax highlighting active') : fail(`SVM tokens: ${JSON.stringify(tok)}`);

  // A run reads the editor contents and completes (hello returns its 14-byte greeting length).
  await page.click('#run');
  await page.waitForFunction(
    () => ['done', 'error', 'stopped'].includes(document.getElementById('state').dataset.state),
    { timeout: 20_000 },
  );
  const hello = await page.evaluate(() => ({
    state: document.getElementById('state').dataset.state,
    result: document.getElementById('result').textContent.trim(),
  }));
  hello.state === 'done' && hello.result === '14'
    ? ok('SVM text ran via the editor → 14')
    : fail(`hello run: ${JSON.stringify(hello)}`);

  // Switching to the Lua example loads Lua source under the Lua mode…
  await page.selectOption('#example', 'Lua (5.4.7 — write & run)');
  await page.waitForTimeout(150);
  const lua = await page.evaluate(() => {
    const cm = document.querySelector('.CodeMirror')?.CodeMirror;
    return { mode: cm?.getOption('mode'), hasPrint: (cm?.getValue() || '').includes('print(') };
  });
  lua.mode === 'lua' && lua.hasPrint ? ok('Lua example → lua mode') : fail(`Lua switch: ${JSON.stringify(lua)}`);

  // …and running an editable module feeds the editor contents to the guest as stdin.
  await page.click('#run');
  await page.waitForFunction(
    () => ['done', 'error', 'stopped'].includes(document.getElementById('state').dataset.state),
    { timeout: 30_000 },
  );
  const luaOut = await page.evaluate(() => document.getElementById('stdout').textContent);
  luaOut.includes('Hello from Lua') ? ok('editable-module stdin reads the editor') : fail(`Lua stdout: ${luaOut.slice(0, 80)}`);

  // The SQL example switches the mode again.
  await page.selectOption('#example', 'SQLite (:memory: — write & run SQL)');
  await page.waitForTimeout(150);
  const sqlMode = await page.evaluate(() => document.querySelector('.CodeMirror')?.CodeMirror?.getOption('mode'));
  sqlMode === 'text/x-sql' ? ok('SQL example → sql mode') : fail(`SQL mode: ${sqlMode}`);

  // A parse error pins the offending line: type SVM text with a bad opcode on line 3 (unique token),
  // Run, and the editor gets a gutter marker + highlighted line + inline message on that line.
  await page.selectOption('#example', 'hello');
  await page.waitForTimeout(100);
  await page.evaluate(() => document.querySelector('.CodeMirror').CodeMirror.setValue(
    'func () -> (i64) {\nblock0():\n  v0 = i64.notanopcode 1\n  return v0\n}'));
  await page.click('#run');
  await page.waitForFunction(() => document.getElementById('state').dataset.state === 'error', { timeout: 20_000 });
  const mark = await page.evaluate(() => {
    const cm = document.querySelector('.CodeMirror').CodeMirror;
    const info = cm.lineInfo(2); // 0-based line 2 = the bad-opcode line
    return {
      gutter: !!(info.gutterMarkers && info.gutterMarkers['svm-error-gutter']),
      lineClass: (info.bgClass || '').includes('cm-error-line'),
      widget: !!document.querySelector('.cm-error-widget'),
    };
  });
  (mark.gutter && mark.lineClass && mark.widget)
    ? ok('parse error pinned to the right line (gutter + line + inline message)')
    : fail(`error decoration: ${JSON.stringify(mark)}`);
  // Editing clears the decoration.
  await page.evaluate(() => document.querySelector('.CodeMirror').CodeMirror.setValue('func () -> (i64) {\nblock0():\n  v0 = i64.const 1\n  return v0\n}'));
  const cleared = await page.evaluate(() => !document.querySelector('.cm-error-widget'));
  cleared ? ok('error decoration clears on edit') : fail('error decoration not cleared on edit');

  // The Vim toggle engages the Vim keymap for real (registered + editor holds vim state).
  await page.check('#vim');
  const vim = await page.evaluate(() => {
    const cm = document.querySelector('.CodeMirror')?.CodeMirror;
    return { opt: cm?.getOption('keyMap'), state: !!cm?.state?.vim };
  });
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
