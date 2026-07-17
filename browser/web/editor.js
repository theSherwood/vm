// The playground editor — a thin factory over CodeMirror 5 (vendored as `codemirror.bundle.js`, loaded
// as a classic script in play.html so `window.CodeMirror` exists before this module runs). CodeMirror
// is UI only: it never touches the sandbox or any authority, so an editor library here doesn't enlarge
// the trusted core.
//
// `createEditor` returns one editor **instance** (the playground mounts one per editable demo card);
// each instance carries its own parse-error decoration. `setVimAll` toggles the Vim keymap across all
// of them at once (the sidebar's single "vim" switch).

const CM = window.CodeMirror;

// A minimal highlighting mode for **SVM text** (the CLIF/LLVM-flavored IR form; grammar in the
// `svm-text` crate). Rule order matters — first match wins:
//   - `;` line comments and `"…"` strings first;
//   - dotted opcodes (`i64.const`, `cap.call`, `thread.spawn`, `mem.fill`, `atomic.rmw.add`) before
//     the bare-type rule, so `i64.const` isn't split at `i64`;
//   - structural keywords / terminators, scalar+vector types, `blockN` labels, `vN` SSA values,
//     numbers (decimal + hex), and the `->` signature arrow.
CM.defineSimpleMode('svm', {
  start: [
    { regex: /;.*/, token: 'comment' },
    { regex: /"(?:[^\\"]|\\.)*"/, token: 'string' },
    { regex: /[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+/, token: 'builtin' },
    { regex: /\b(?:memory|data|func|export|type|return_call_indirect|return_call|return|br_if|br|unreachable|call)\b/, token: 'keyword' },
    { regex: /\b(?:i32|i64|f32|f64|v128)\b/, token: 'type' },
    { regex: /\bblock\d+\b/, token: 'def' },
    { regex: /\bv\d+\b/, token: 'variable-2' },
    { regex: /\b0x[0-9a-fA-F]+\b|-?\b\d+\b/, token: 'number' },
    { regex: /->/, token: 'operator' },
  ],
  meta: { lineComment: ';' },
});

// CodeMirror mode string for a demo's declared `lang`. SVM text is the default.
const MODE = { svm: 'svm', lua: 'lua', sql: 'text/x-sql', c: 'text/x-csrc' };

const instances = [];

// Mount a CodeMirror editor over `textarea`, seeded with language `lang` (a key of MODE). Returns an
// instance API: `getValue`, `markError`/`clearError` (parse-error gutter decoration), `refresh`,
// `focus`, `setVim`.
export function createEditor(textarea, lang) {
  const cm = CM.fromTextArea(textarea, {
    lineNumbers: true,
    matchBrackets: true,
    tabSize: 2,
    indentUnit: 2,
    lineWrapping: false,
    mode: MODE[lang] || MODE.svm,
    // A dedicated gutter left of the line numbers for parse/verify error markers.
    gutters: ['svm-error-gutter', 'CodeMirror-linenumbers'],
  });

  // --- per-instance parse/verify error surfacing --------------------------------------------------
  // The engine's errors are plain messages (no source location yet) but consistently quote the
  // offending token in backticks — `unknown opcode \`foo\``. So we pin the line **only when that token
  // occurs on exactly one line** (never guess a wrong line): a red gutter marker + a highlighted line +
  // an inline widget. Absent/ambiguous → decorate nothing, leaving the message to the status line.
  let errorLine = null;
  let errorWidget = null;
  const clearError = () => {
    if (errorLine === null) return;
    cm.setGutterMarker(errorLine, 'svm-error-gutter', null);
    cm.removeLineClass(errorLine, 'background', 'cm-error-line');
    if (errorWidget) { errorWidget.clear(); errorWidget = null; }
    errorLine = null;
  };
  const markError = (message) => {
    clearError();
    const quoted = /`([^`]+)`/.exec(message);
    if (!quoted) return false;
    const needle = quoted[1];
    const lines = cm.getValue().split('\n');
    const hits = [];
    for (let i = 0; i < lines.length && hits.length < 2; i++) {
      if (lines[i].includes(needle)) hits.push(i);
    }
    if (hits.length !== 1) return false;
    errorLine = hits[0];
    const marker = document.createElement('span');
    marker.className = 'cm-error-marker';
    marker.textContent = '●';
    marker.title = message;
    cm.setGutterMarker(errorLine, 'svm-error-gutter', marker);
    cm.addLineClass(errorLine, 'background', 'cm-error-line');
    const widget = document.createElement('div');
    widget.className = 'cm-error-widget';
    widget.textContent = message;
    errorWidget = cm.addLineWidget(errorLine, widget, { coverGutter: false, noHScroll: true });
    return true;
  };
  cm.on('change', clearError); // clear the decoration as soon as the author starts fixing it

  const api = {
    cm,
    getValue: () => cm.getValue(),
    setValue: (v) => cm.setValue(v),
    // Register a listener fired on every edit (used for localStorage persistence). CodeMirror also
    // fires `change` for programmatic setValue, so a caller that sets the value inside its own handler
    // must guard against re-entry (the playground does — see `restoreOrSeed`).
    onChange: (fn) => cm.on('change', fn),
    markError,
    clearError,
    refresh: () => cm.refresh(),
    focus: () => cm.focus(),
    setVim: (on) => cm.setOption('keyMap', on ? 'vim' : 'default'),
  };
  instances.push(api);
  return api;
}

// Toggle the Vim keymap across every mounted editor.
export function setVimAll(on) {
  for (const e of instances) e.setVim(on);
}

// Refresh every editor's layout (CodeMirror needs this after its container is first laid out / resized).
export function refreshAll() {
  for (const e of instances) e.refresh();
}
