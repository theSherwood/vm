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
import { mkdirSync, existsSync, writeFileSync, readFileSync, copyFileSync } from 'node:fs';
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

// 1b) gradient — the framebuffer demo: a C guest renders an RGBA image and presents one frame through
//     the `display` capability; the page blits it to a <canvas>. The output waist Doom will ride.
try {
  buildC('gradient', join(REPO, 'crates', 'svm-run', 'demos', 'display', 'gradient.c'));
} catch (e) {
  console.log(`  ✗ gradient: ${e.message}`);
}

// 1c) bounce — the interactive reactor demo: a C guest whose exported `tick()` the page calls once per
//     requestAnimationFrame, steering a bouncing box with the arrow keys (the `keyboard` cap in, the
//     `display` cap out). The per-frame run model + input waist Doom rides.
try {
  buildC('bounce', join(REPO, 'crates', 'svm-run', 'demos', 'display', 'bounce.c'));
} catch (e) {
  console.log(`  ✗ bounce: ${e.message}`);
}

// 1d) life — Conway's Game of Life over a malloc heap ABOVE the mapped window: the reactor must
//     persist the guest's whole memory (heap included) between frames or the glider freezes. The
//     heap-persistence proof Doom's zone allocator needs.
try {
  buildC('life', join(REPO, 'crates', 'svm-run', 'demos', 'display', 'life.c'));
} catch (e) {
  console.log(`  ✗ life: ${e.message}`);
}

// 1e) mandelzoom — an interactive Mandelbrot zoom: each reactor `tick()` computes a full
//     double-precision Mandelbrot for the current (auto-zooming, arrow-steerable) view on the CPU
//     in-guest and presents it through `display`. Pure f64 + an integer palette — no libm bundled.
try {
  buildC('mandelzoom', join(REPO, 'crates', 'svm-run', 'demos', 'display', 'mandelzoom.c'));
} catch (e) {
  console.log(`  ✗ mandelzoom: ${e.message}`);
}

// 1f) gpu_shader — the GPU demo: the guest ships a WGSL fragment shader through the `webgpu` capability
//     and the browser renders it (a Mandelbrot zoom) each frame on the real GPU via navigator.gpu.
try {
  buildC('gpu_shader', join(REPO, 'crates', 'svm-run', 'demos', 'display', 'gpu_shader.c'));
} catch (e) {
  console.log(`  ✗ gpu_shader: ${e.message}`);
}

// 2) SQLite (interactive) — the unmodified 3.50.2 amalgamation with a driver that reads a SQL script
//    from **stdin** and runs it against an in-memory database, printing each statement's result table.
//    The page pipes the editor's SQL in as stdin. Fetch-and-cache the amalgamation (same version +
//    cache dir the svm-llvm test harness uses); skip offline.
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
    buildC('sqlite_repl', join(REPO, 'crates', 'svm-run', 'demos', 'sqlite', 'sqlite_repl.c'), [AMALG]);
  } catch (e) {
    console.log(`  ✗ sqlite_repl: ${e.message}`);
  }
} else {
  console.log('  – sqlite_repl skipped (amalgamation fetch failed — offline?)');
}

// 3) Lua (interactive) — Lua 5.4.7 core + base/string/table/math/coroutine/io/os libraries + a guest
//    snprintf, with a harness that reads a Lua chunk from **stdin** and runs it. The page pipes the
//    editor's text in as stdin, so the user writes and runs their own Lua. io.write/os.date/coroutine
//    all work; file I/O (io.open) degrades to nil (no fs cap granted). Committed golden fixture
//    (`lua_eval.bc`; no Lua source needed).
try {
  buildBc('lua_eval', join(REPO, 'crates', 'svm-llvm', 'tests', 'fixtures', 'lua', 'lua_eval.bc'));
} catch (e) {
  console.log(`  ✗ lua_eval: ${e.message}`);
}

// 4) Doom (interactive reactor) — doomgeneric through the on-ramp, driven one `tick` per frame over
//    the persistent window; `_start` reads the shareware IWAD through the `fs` capability. Two assets:
//    the module (`demos/doom/{fetch,build}.sh` — id Software's Doom source is fetched-and-built, not
//    vendored) and the freely-distributable shareware `doom1.wad`. The page opens the reactor over the
//    WAD via `svm_onramp_open_fs`. Both are skipped (the playground just omits the example) if the
//    toolchain or a fetch is unavailable — same fail-soft as SQLite offline.
const DOOM = join(REPO, 'crates', 'svm-run', 'demos', 'doom');
const DCACHE = '/tmp/doomgeneric_cache';

// Build doom.svmb via the demo scripts (fetch the sources, then compile+link+translate). Returns the
// built module path, or null if the fetch/build failed (offline, or no clang/llvm-link).
function ensureDoomModule() {
  const built = join(DCACHE, 'bc', 'doom.svmb');
  if (existsSync(built)) return built;
  try {
    execFileSync('sh', [join(DOOM, 'fetch.sh')], { stdio: 'inherit' });
    execFileSync('sh', [join(DOOM, 'build.sh')], { stdio: 'inherit' });
    return existsSync(built) ? built : null;
  } catch (e) {
    console.log(`  ✗ doom build: ${e.message}`);
    return null;
  }
}

// Fetch the shareware doom1.wad (freely redistributable). Returns its path, or null if unavailable.
// Verifies the IWAD magic so a captive-portal HTML page can't masquerade as the WAD.
function ensureWad() {
  const wad = join(DCACHE, 'doom1.wad');
  const ok = (p) => existsSync(p) && readFileSync(p).subarray(0, 4).toString('latin1') === 'IWAD';
  if (ok(wad)) return wad;
  mkdirSync(DCACHE, { recursive: true });
  for (const url of ['https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad']) {
    try {
      execFileSync('curl', ['-sfL', '--max-time', '180', '-o', wad, url], { stdio: 'inherit' });
      if (ok(wad)) return wad;
    } catch { /* try the next mirror */ }
  }
  return null;
}

const doomSvmb = ensureDoomModule();
const doomWad = ensureWad();
if (doomSvmb && doomWad) {
  copyFileSync(doomSvmb, join(ASSETS, 'doom.svmb'));
  copyFileSync(doomWad, join(ASSETS, 'doom1.wad'));
  const mb = (n) => (readFileSync(n).length / (1024 * 1024)).toFixed(2);
  console.log(`  ✓ doom.svmb (${mb(doomSvmb)} MB) + doom1.wad (${mb(doomWad)} MB)`);
} else {
  console.log('  – doom skipped (no toolchain, or the source/WAD fetch failed — offline?)');
}

console.log('done. Assets in web/assets/. Serve with `node serve.mjs` and open /web/play.html');
