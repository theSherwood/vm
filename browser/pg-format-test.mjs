// Unit test for `web/pg-format.js` — the psql-style table formatter for `postgres --single` output.
// The inputs are real backend captures (tabs and all), so this pins the parser to what the shipping
// backend actually prints. Pure Node, no wasm/browser. Run: node pg-format-test.mjs
import { formatPgOutput } from './web/pg-format.js';

let failed = false;
const eq = (label, got, want) => {
  if (got === want) { console.log(`  ok: ${label}`); return; }
  failed = true;
  console.log(`  FAIL: ${label}\n--- got ---\n${got}\n--- want ---\n${want}\n-----------`);
};

// A two-column SELECT with a text column: header block, then one block per row, `\t----`-separated.
const twoCol = [
  '\t 1: x\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: s\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
  '\t 1: x = "3"\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: s = "three"\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
  '\t 1: x = "2"\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: s = "two"\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
].join('\n');
// int column (typeid 23) right-aligns; text (25) left-aligns; header centered.
eq('two-column SELECT → grid', formatPgOutput(twoCol), [
  ' x |   s   ',
  '---+-------',
  ' 3 | three ',
  ' 2 | two   ',
  '(2 rows)',
].join('\n'));

// Aggregates: count/sum are int8 (20, right), avg is numeric (1700, right); wide numeric header.
const agg = [
  '\t 1: count\t(typeid = 20, len = 8, typmod = -1, byval = t)',
  '\t 2: sum\t(typeid = 20, len = 8, typmod = -1, byval = t)',
  '\t 3: avg\t(typeid = 1700, len = -1, typmod = -1, byval = f)',
  '\t----',
  '\t 1: count = "3"\t(typeid = 20, len = 8, typmod = -1, byval = t)',
  '\t 2: sum = "6"\t(typeid = 20, len = 8, typmod = -1, byval = t)',
  '\t 3: avg = "2.0000000000000000"\t(typeid = 1700, len = -1, typmod = -1, byval = f)',
  '\t----',
].join('\n');
eq('aggregate SELECT → right-aligned numerics', formatPgOutput(agg), [
  ' count | sum |        avg         ',
  '-------+-----+--------------------',
  '     3 |   6 | 2.0000000000000000 ',
  '(1 row)',
].join('\n'));

// Zero rows: header block, then straight to the prompt — psql shows the header + `(0 rows)`.
const empty = [
  '\t 1: x\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t----',
  'backend> ',
].join('\n');
eq('empty result → header + (0 rows)', formatPgOutput(empty), [
  ' x ',
  '---',
  '(0 rows)',
  'backend> ',
].join('\n'));

// A NULL cell prints with no ` = "..."`, exactly like a header line — but position (a row block, not the
// first block) disambiguates it. psql renders NULL as empty.
const withNull = [
  '\t 1: a\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: b\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
  '\t 1: a = "1"\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: b\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
].join('\n');
eq('NULL cell → empty', formatPgOutput(withNull), [
  ' a | b ',
  '---+---',
  ' 1 |   ',
  '(1 row)',
].join('\n'));

// Non-result text (prompt, banner, NOTICE) passes through verbatim, and a table embedded between
// prompts is formatted in place (the real multi-statement Run shape).
const mixed = [
  '',
  'PostgreSQL stand-alone backend 17.5',
  'backend> ',
  'backend> ',
  '\t 1: n\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t----',
  '\t 1: n = "42"\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t----',
  'backend> ',
].join('\n');
eq('mixed stream → only the table is reformatted', formatPgOutput(mixed), [
  '',
  'PostgreSQL stand-alone backend 17.5',
  'backend> ',
  'backend> ',
  ' n  ',
  '----',
  ' 42 ',
  '(1 row)',
  'backend> ',
].join('\n'));

// The real gluing: `backend> ` has no trailing newline, so the first descriptor of a result set arrives
// stuck to the prompt line (`backend> \t 1: x\t(...)`), and consecutive statements pile prompts up
// (`backend> backend> \t 1: ...`). The formatter must peel the prompt(s) off and still build the table.
const glued = [
  'backend> backend> \t 1: x\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: s\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
  '\t 1: x = "3"\t(typeid = 23, len = 4, typmod = -1, byval = t)',
  '\t 2: s = "three"\t(typeid = 25, len = -1, typmod = -1, byval = f)',
  '\t----',
  'backend> ',
].join('\n');
eq('prompt glued to first descriptor → peeled + table built', formatPgOutput(glued), [
  'backend> backend> ',
  ' x |   s   ',
  '---+-------',
  ' 3 | three ',
  '(1 row)',
  'backend> ',
].join('\n'));

// Output with no result blocks is returned byte-for-byte (lossless passthrough).
const plain = 'backend> \nCREATE TABLE\nbackend> ';
eq('no tables → unchanged', formatPgOutput(plain), plain);

console.log(failed ? 'pg-format test: FAILED' : 'pg-format test: PASSED');
process.exit(failed ? 1 : 0);
