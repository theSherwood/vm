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

ensureTranslator();

// 1) hello — the tiny always-present example (also committed so the playground works out of the box).
try {
  buildC('hello_c', join(REPO, 'crates', 'svm-run', 'demos', 'hello.c'));
} catch (e) {
  console.log(`  ✗ hello_c: ${e.message}`);
}

// 2) SQLite Phase A — the `:memory:` breadth script printing query results to stdout. Needs the
//    amalgamation (the svm-llvm test harness fetches it to /tmp/svm_sqlite_cache).
const AMALG = '/tmp/svm_sqlite_cache/sqlite-amalgamation-3500200';
if (existsSync(join(AMALG, 'sqlite3.c'))) {
  try {
    buildC('sqlite_demo', join(REPO, 'crates', 'svm-run', 'demos', 'sqlite', 'sqlite_demo.c'),
      [AMALG], ['-DSVM_GUEST']);
  } catch (e) {
    console.log(`  ✗ sqlite_demo: ${e.message}`);
  }
} else {
  console.log('  – sqlite_demo skipped (amalgamation not cached; run the svm-llvm sqlite test once)');
}

console.log('done. Assets in web/assets/. Serve with `node serve.mjs` and open /web/play.html');
