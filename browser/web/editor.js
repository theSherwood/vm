// The playground editor — a thin facade over a single CodeMirror 5 instance (vendored as
// `codemirror.bundle.js`, loaded as a classic script in play.html so `window.CodeMirror` exists
// before this module runs). CodeMirror is UI only: it never touches the sandbox or any authority, so
// an editor library here doesn't enlarge the trusted core.
//
// Exposes a small surface the rest of play.js drives: mount once over the existing `<textarea>`, then
// swap the document / language / read-only state per demo, read it back for a run, and toggle Vim.

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

// CodeMirror mode string for a demo's declared `lang` (added per-example in play.js). SVM text is the
// default; the read-only "click Run" notes use the C mode so their `//` lines render as comments.
const MODE = {
  svm: 'svm',
  lua: 'lua',
  sql: 'text/x-sql',
  c: 'text/x-csrc',
  note: 'text/x-csrc',
};

let cm = null;

// Replace `textarea` with a CodeMirror editor. Returns nothing; use the helpers below.
export function mountEditor(textarea) {
  cm = CM.fromTextArea(textarea, {
    lineNumbers: true,
    matchBrackets: true,
    tabSize: 2,
    indentUnit: 2,
    lineWrapping: false,
    mode: MODE.svm,
    // A dedicated gutter left of the line numbers for parse/verify error markers.
    gutters: ['svm-error-gutter', 'CodeMirror-linenumbers'],
  });
  // Clear any error decoration as soon as the author starts fixing it.
  cm.on('change', clearError);
}

// --- parse/verify error surfacing -------------------------------------------------------------------
// The engine's parse/verify errors are plain messages (no source location yet), but they consistently
// quote the offending token in backticks — `unknown opcode \`foo\``, `duplicate block label \`bad\``.
// So we locate the error line **only when that token occurs on exactly one line** (never guess a wrong
// line): a red gutter marker + a highlighted line + an inline widget carrying the message. When the
// token is absent or ambiguous, we decorate nothing and leave the message to the status line — better
// no marker than a misplaced one. (A precise, always-located version wants line info from the parser.)
let errorLine = null;
let errorWidget = null;

// Decorate the editor for parse/verify error `message`. Returns true if it pinned a specific line.
export function markError(message) {
  clearError();
  if (!cm) return false;
  const quoted = /`([^`]+)`/.exec(message);
  if (!quoted) return false;
  const needle = quoted[1];
  const doc = cm.getValue();
  const lines = doc.split('\n');
  const hits = [];
  for (let i = 0; i < lines.length && hits.length < 2; i++) {
    if (lines[i].includes(needle)) hits.push(i);
  }
  if (hits.length !== 1) return false; // absent or ambiguous → don't guess

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
}

// Remove any error decoration.
export function clearError() {
  if (!cm || errorLine === null) return;
  cm.setGutterMarker(errorLine, 'svm-error-gutter', null);
  cm.removeLineClass(errorLine, 'background', 'cm-error-line');
  if (errorWidget) { errorWidget.clear(); errorWidget = null; }
  errorLine = null;
}

// Load `text` into the editor as language `lang` (a key of MODE), read-only or not.
export function setDoc(text, lang, readOnly) {
  cm.setOption('mode', MODE[lang] || MODE.svm);
  cm.setOption('readOnly', !!readOnly);
  // Read-only panes are just notes; dim the gutter so they don't read as editable.
  cm.getWrapperElement().classList.toggle('cm-readonly', !!readOnly);
  cm.setValue(text);
  cm.clearHistory();
}

// The current editor contents (what a Run reads).
export function getDoc() {
  return cm ? cm.getValue() : '';
}

// Enable/disable the Vim keymap.
export function setVim(on) {
  cm.setOption('keyMap', on ? 'vim' : 'default');
}

// Refresh layout (CodeMirror needs this when its container becomes visible or resizes).
export function refresh() {
  if (cm) cm.refresh();
}
