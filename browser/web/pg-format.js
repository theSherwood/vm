// Reformat the raw output of `postgres --single` into psql-style aligned tables.
//
// The standalone backend has no libpq frontend, so it prints query results through `printtup`'s debug
// path (`printatt` in src/backend/access/common/printtup.c): a column-descriptor **header block**, then
// one **row block** per tuple, each block terminated by a lone `\t----` line. Every column line is
//
//   \t 1: name = "value"\t(typeid = OID, len = N, typmod = N, byval = t|f)
//
// with the ` = "value"` part **absent** in the header block and for any NULL cell. We recover the column
// names + per-column `typeid` (to right-align numeric columns, as psql does) from the header block and
// the values from the row blocks, and render a grid. Anything that isn't a recognized result block —
// the `backend>` prompt, the banner, `NOTICE`/`ERROR` lines — passes through **verbatim**, so only
// SELECT results change and nothing is ever lost.

// Postgres type OIDs that psql right-aligns (numbers). int2/4/8, oid, float4/8, money, numeric.
const NUMERIC_OIDS = new Set([20, 21, 23, 26, 700, 701, 790, 1700]);
// A `printatt` line: attnum, the "name" or `name = "value"` middle, and the type OID.
const COL_RE = /^\t\s*(\d+): ([\s\S]*)\t\(typeid = (\d+), len = -?\d+, typmod = -?\d+, byval = [tf]\)$/;
// The same descriptor, but allowing prompt text in front of it: the backend prints `backend> ` with **no
// trailing newline**, so the first descriptor of a result set arrives glued onto the prompt line
// (`backend> \t 1: x\t(typeid = ...)`). Split that prefix off onto its own line before parsing.
const COL_TAIL = /^(.*?)(\t\s*\d+: [\s\S]*\t\(typeid = \d+, len = -?\d+, typmod = -?\d+, byval = [tf]\))$/;
const SEP = '\t----';

const parseCol = (line) => {
  const m = line == null ? null : COL_RE.exec(line);
  return m ? { attnum: +m[1], mid: m[2], typeid: +m[3] } : null;
};
const hasValue = (mid) => / = "[\s\S]*"$/.test(mid);
const valueOf = (mid) => {
  const m = /^[\s\S]* = "([\s\S]*)"$/.exec(mid);
  return m ? m[1] : ''; // no ` = "..."` ⇒ NULL ⇒ empty (psql's default null display)
};

const center = (s, w) => {
  const pad = w - s.length;
  const left = Math.floor(pad / 2);
  return ' '.repeat(left) + s + ' '.repeat(pad - left);
};

// Render one result set (header columns + rows) as psql's aligned format.
function renderTable(header, rows) {
  const names = header.map((h) => h.mid);
  const right = header.map((h) => NUMERIC_OIDS.has(h.typeid));
  const width = names.map((n, c) => Math.max(n.length, ...rows.map((r) => r[c].length)));
  const cell = (v, c) => (right[c] ? v.padStart(width[c]) : v.padEnd(width[c]));
  const dataRow = (vals) => ' ' + vals.map((v, c) => cell(v, c)).join(' | ') + ' ';
  const out = [
    ' ' + names.map((n, c) => center(n, width[c])).join(' | ') + ' ',
    width.map((w) => '-'.repeat(w + 2)).join('+'),
    ...rows.map(dataRow),
    `(${rows.length} row${rows.length === 1 ? '' : 's'})`,
  ];
  return out;
}

// Transform a chunk of backend stdout, prettifying every result block and passing everything else
// through unchanged. Safe on partial/odd input: anything that doesn't parse as a full block is emitted
// verbatim, so the worst case is the original raw text.
export function formatPgOutput(raw) {
  // Normalize: peel any prompt/text prefix off a glued descriptor line so every descriptor stands alone
  // (see COL_TAIL). The peeled prefix stays as its own passthrough line, so prompts are preserved.
  const lines = [];
  for (const phys of raw.split('\n')) {
    const m = COL_TAIL.exec(phys);
    if (m && m[1] !== '') {
      lines.push(m[1], m[2]);
    } else {
      lines.push(phys);
    }
  }
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const first = parseCol(lines[i]);
    // A result set starts with a header block: value-less column lines ending in `\t----`.
    if (!first || hasValue(first.mid)) {
      out.push(lines[i]);
      i += 1;
      continue;
    }
    const header = [];
    let j = i;
    while (parseCol(lines[j]) && !hasValue(parseCol(lines[j]).mid)) {
      header.push(parseCol(lines[j]));
      j += 1;
    }
    if (lines[j] !== SEP) {
      // Not a real header (no terminating separator) — emit the first line and rescan from the next.
      out.push(lines[i]);
      i += 1;
      continue;
    }
    j += 1; // consume the header separator
    // Row blocks: each a run of column lines ending in `\t----`. A NULL cell is a value-less line.
    const rows = [];
    let ok = true;
    while (parseCol(lines[j])) {
      const cells = new Array(header.length).fill('');
      let k = j;
      while (parseCol(lines[k])) {
        const c = parseCol(lines[k]);
        if (c.attnum >= 1 && c.attnum <= header.length) cells[c.attnum - 1] = valueOf(c.mid);
        k += 1;
      }
      if (lines[k] !== SEP) {
        ok = false;
        break;
      }
      rows.push(cells);
      j = k + 1;
    }
    if (!ok) {
      // Malformed row block — bail out and emit the header start verbatim rather than guess.
      out.push(lines[i]);
      i += 1;
      continue;
    }
    out.push(...renderTable(header, rows));
    i = j;
  }
  return out.join('\n');
}
