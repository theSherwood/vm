// Build the playground's **on-ramp `.svmb` assets** — real C/C++ guests (Lua, SQLite) compiled
// through `clang -O2 -emit-llvm` and translated by `svm-llvm-translate` into SVM-IR modules the
// browser engine runs via `svm_run_onramp` (see `web/play.js`).
//
// Every asset is translated with **`--host-page 65536`**: a wasm host has 64 KiB pages, so a
// read-only global must not share a host page with the writable data stack (it would fault under
// D40). The native default (16 KiB) is wrong for the browser — see the `svm-llvm` stack-page commit.
//
// Usage:  node build-onramp-assets.mjs           (builds whatever the toolchain + caches allow)
// Needs `clang`/`llvm-dis` on PATH. SQLite/Lua sources are fetched-and-cached (skipped offline).
// Outputs to `web/assets/*.svmb` (gitignored except the tiny committed `hello_c.svmb`).

import { execFileSync } from 'node:child_process';
import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const ASSETS = join(HERE, 'web', 'assets');
const HOST_PAGE = '65536';
mkdirSync(ASSETS, { recursive: true });

// Build the translator once (release), reuse its path.
const TR = join(REPO, 'crates', 'svm-llvm', 'target', 'release', 'svm-llvm-translate');
function ensureTranslator() {
  if (existsSync(TR)) return;
  console.log('building svm-llvm-translate…');
  execFileSync('cargo', ['build', '--release', '--bin', 'svm-llvm-translate'], {
    cwd: join(REPO, 'crates', 'svm-llvm'), stdio: 'inherit',
  });
}

// clang a C source to bitcode, then translate to a 64 KiB-page `.svmb`. Extra clang flags per guest.
function buildC(name, src, includes = [], defines = []) {
  const bc = join(ASSETS, `${name}.bc`);
  const svmb = join(ASSETS, `${name}.svmb`);
  const flags = ['-O2', '-emit-llvm', '-c', '-fno-vectorize', '-fno-slp-vectorize'];
  execFileSync('clang', [...flags, ...defines, ...includes.map((i) => `-I${i}`), src, '-o', bc], { stdio: 'inherit' });
  execFileSync(TR, [bc, '-o', svmb, '--host-page', HOST_PAGE], { stdio: 'inherit' });
  const size = execFileSync('wc', ['-c', svmb]).toString().trim().split(/\s+/)[0];
  console.log(`  ✓ ${name}.svmb (${size} B)`);
}

// Translate an **already-committed** `.bc` fixture to a 64 KiB-page `.svmb` (no clang step — the
// bitcode is a golden input in the tree, e.g. the Lua test fixtures).
function buildBc(name, bcPath) {
  const svmb = join(ASSETS, `${name}.svmb`);
  execFileSync(TR, [bcPath, '-o', svmb, '--host-page', HOST_PAGE], { stdio: 'inherit' });
  const size = execFileSync('wc', ['-c', svmb]).toString().trim().split(/\s+/)[0];
  console.log(`  ✓ ${name}.svmb (${size} B)`);
}

ensureTranslator();

// 1) hello — the tiny always-present example (also committed so the playground works out of the box).
try {
  buildC('hello_c', join(REPO, 'crates', 'svm-run', 'demos', 'hello.c'));
} catch (e) {
  console.log(`  ✗ hello_c: ${e.message}`);
}

// 2) SQLite Phase A — the `:memory:` breadth script printing query results to stdout. Fetch-and-cache
//    the 3.50.2 amalgamation (same version + cache dir the svm-llvm test harness uses); skip offline.
const CACHE = '/tmp/svm_sqlite_cache';
const AMALG = join(CACHE, 'sqlite-amalgamation-3500200');
function ensureAmalgamation() {
  if (existsSync(join(AMALG, 'sqlite3.c'))) return true;
  mkdirSync(CACHE, { recursive: true });
  const zip = join(CACHE, 'amalgamation.zip');
  try {
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', zip,
      'https://sqlite.org/2025/sqlite-amalgamation-3500200.zip'], { stdio: 'inherit' });
    execFileSync('unzip', ['-o', '-q', zip, '-d', CACHE], { stdio: 'inherit' });
    return existsSync(join(AMALG, 'sqlite3.c'));
  } catch {
    return false;
  }
}
if (ensureAmalgamation()) {
  try {
    buildC('sqlite_demo', join(REPO, 'crates', 'svm-run', 'demos', 'sqlite', 'sqlite_demo.c'),
      [AMALG], ['-DSVM_GUEST']);
  } catch (e) {
    console.log(`  ✗ sqlite_demo: ${e.message}`);
  }
} else {
  console.log('  – sqlite_demo skipped (amalgamation fetch failed — offline?)');
}

// 3) Lua (stdlib) — Lua 5.4.7 core + base/string/table/math libraries running a script that print()s
//    string/table/math results. The bitcode is a committed golden fixture (no Lua source needed).
try {
  buildBc('lua_stdlib', join(REPO, 'crates', 'svm-llvm', 'tests', 'fixtures', 'lua', 'lua_stdlib.bc'));
} catch (e) {
  console.log(`  ✗ lua_stdlib: ${e.message}`);
}

console.log('done. Assets in web/assets/. Serve with `node serve.mjs` and open /web/play.html');
