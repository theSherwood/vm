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
import { mkdirSync, existsSync, writeFileSync, readFileSync, copyFileSync, rmSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
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

// 2b) QuickJS (interactive) — Bellard's QuickJS 2024-01-13 with a driver (`qjs_repl.c`) that reads a
//     JS program from **stdin**, evaluates it (print/console.log + the completion value), and prints.
//     Multi-TU, mirroring the `demo_quickjs_eval_vs_native` test: the engine + a guest libm (openlibm,
//     for the address-taken Math functions) + the reused printf/strtod/libc shims, `llvm-link`ed into
//     one `.ll`, then translated. Fetched-and-cached (QuickJS from bellard.org, openlibm from GitHub);
//     when the openlibm fetch is unavailable (the Pages pipeline can't reach GitHub) this rebuild is
//     skipped and the **committed** `web/assets/qjs_repl.svmb` is left in place, so the JS playground
//     works out of the box regardless (see `web/assets/.gitignore` whitelist).
const QJS_VER = '2024-01-13';
const QJS_CACHE = '/tmp/svm_quickjs_cache';
const QJS_DIR = join(QJS_CACHE, `quickjs-${QJS_VER}`);
const OL_VER = '0.8.5';
const OL_CACHE = '/tmp/svm_openlibm_cache';
const OL_DIR = join(OL_CACHE, `openlibm-${OL_VER}`);
// The openlibm double set QuickJS's `Math` object takes the address of (kept in sync with the
// svm-llvm test's OPENLIBM_SRCS + QUICKJS_OPENLIBM_EXTRA).
const OPENLIBM_SRCS = [
  'e_log', 'e_log10', 'e_log2', 'e_exp', 's_exp2', 'e_pow', 's_sin', 's_cos', 's_tan',
  'k_sin', 'k_cos', 'k_tan', 'e_rem_pio2', 'k_rem_pio2', 'e_asin', 'e_acos', 's_atan',
  'e_atan2', 'e_sinh', 'e_cosh', 's_tanh', 's_cbrt', 'e_fmod', 's_scalbn', 's_copysign',
  's_fabs', 'k_exp', 's_expm1', 's_asinh', 'e_acosh', 'e_atanh', 's_log1p', 'e_hypot',
  's_floor', 's_ceil', 's_trunc', 'e_sqrt',
];
function ensureQuickJS() {
  if (existsSync(join(QJS_DIR, 'quickjs.c'))) return true;
  mkdirSync(QJS_CACHE, { recursive: true });
  try {
    const tar = join(QJS_CACHE, `quickjs-${QJS_VER}.tar.xz`);
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', tar,
      `https://bellard.org/quickjs/quickjs-${QJS_VER}.tar.xz`], { stdio: 'inherit' });
    execFileSync('tar', ['xf', tar, '-C', QJS_CACHE], { stdio: 'inherit' });
    return existsSync(join(QJS_DIR, 'quickjs.c'));
  } catch { return false; }
}
function ensureOpenlibm() {
  if (existsSync(join(OL_DIR, 'src', 'e_log.c'))) return true;
  mkdirSync(OL_CACHE, { recursive: true });
  try {
    const tgz = join(OL_CACHE, 'openlibm.tar.gz');
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', tgz,
      `https://github.com/JuliaMath/openlibm/archive/refs/tags/v${OL_VER}.tar.gz`], { stdio: 'inherit' });
    execFileSync('tar', ['xf', tgz, '-C', OL_CACHE], { stdio: 'inherit' });
    return existsSync(join(OL_DIR, 'src', 'e_log.c'));
  } catch { return false; }
}
function buildQuickJS() {
  const svmb = join(ASSETS, 'qjs_repl.svmb');
  const demos = join(REPO, 'crates', 'svm-run', 'demos');
  const cflags = ['-O2', '-emit-llvm', '-S', '-c', '-fno-vectorize', '-fno-slp-vectorize',
    '-DNDEBUG', '-D_GNU_SOURCE', `-DCONFIG_VERSION="${QJS_VER}"`, '-DASSEMBLER=0'];
  const incs = [QJS_DIR, OL_DIR, join(OL_DIR, 'include'), join(OL_DIR, 'src'), join(OL_DIR, 'amd64')]
    .map((i) => `-I${i}`);
  const lls = [];
  const cc = (src, tag) => {
    const out = join(ASSETS, `qjs_${tag}.ll`);
    execFileSync('clang', [...cflags, ...incs, src, '-o', out], { stdio: 'inherit' });
    lls.push(out);
  };
  for (const tu of ['quickjs', 'libregexp', 'libunicode', 'cutils', 'libbf']) cc(join(QJS_DIR, `${tu}.c`), tu);
  cc(join(demos, 'quickjs', 'qjs_repl.c'), 'repl');
  cc(join(demos, 'postgres', 'printf_shim.c'), 'printf_shim');
  cc(join(demos, 'strtod', 'strtod.c'), 'strtod');
  cc(join(demos, 'quickjs', 'libc_shim.c'), 'libc_shim');
  for (const s of OPENLIBM_SRCS) cc(join(OL_DIR, 'src', `${s}.c`), s);
  const linked = join(ASSETS, 'qjs_repl_linked.ll');
  execFileSync('llvm-link', ['-S', ...lls, '-o', linked], { stdio: 'inherit' });
  execFileSync(TR, [linked, '-o', svmb, '--host-page', HOST_PAGE], { stdio: 'inherit' });
  const size = execFileSync('wc', ['-c', svmb]).toString().trim().split(/\s+/)[0];
  console.log(`  ✓ qjs_repl.svmb (${size} B)`);
}
if (ensureQuickJS() && ensureOpenlibm()) {
  try {
    buildQuickJS();
  } catch (e) {
    console.log(`  ✗ qjs_repl: ${e.message}`);
  }
} else {
  console.log('  – qjs_repl rebuild skipped (quickjs/openlibm fetch failed) — using committed qjs_repl.svmb');
}

// 3) Lua (interactive) — Lua 5.4.7 core + base/string/table/math/coroutine/io/os libraries + a guest
//    snprintf, with a harness that reads a Lua chunk from **stdin** and runs it. The page pipes the
//    editor's text in as stdin, so the user writes and runs their own Lua. io.write/os.date/coroutine
//    all work; file I/O (io.open) degrades to nil (no fs cap granted). Committed golden fixture
//    (`lua_eval.ll` — the textual `.ll` the reader ingests directly; no Lua source needed).
try {
  buildBc('lua_eval', join(REPO, 'crates', 'svm-llvm', 'tests', 'fixtures', 'lua', 'lua_eval.ll'));
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

// Mirrors for the freely-redistributable shareware IWAD, tried in order. **Several on purpose**: this
// used to be a single slitaz URL, which started 404ing — and because the failure was swallowed, the
// Pages build kept going green while quietly shipping a playground with no Doom (the page 404'd on
// ./assets/doom.svmb). One host disappearing must not cost us the example.
//   - raw.githubusercontent.com serves the canonical shareware v1.9 IWAD (md5 f0cefca4…); it is the
//     same transport `fetch.sh` already falls back to, so it works wherever the sources do.
//   - the rest are the official idgames archive + two of its mirrors, which carry the shareware v1.8
//     IWAD **gzipped**. v1.8 boots and renders the same; its title demo is from an older engine
//     build, which doomgeneric tolerates (it prints instead of `I_Error`-ing on a demo mismatch).
const WAD_MIRRORS = [
  'https://raw.githubusercontent.com/Akbar30Bill/DOOM_wads/master/doom1.wad',
  'https://www.gamers.org/pub/idgames/idstuff/doom/doom-1.8.wad.gz',
  'https://youfailit.net/pub/idgames/idstuff/doom/doom-1.8.wad.gz',
  'https://ftpmirror1.infania.net/pub/idgames/idstuff/doom/doom-1.8.wad.gz',
];

// Fetch the shareware doom1.wad (freely redistributable). Returns its path, or null if every mirror
// is unavailable. Verifies the IWAD magic **after** decompressing, so neither a captive-portal HTML
// page nor a mirror's 404 body can masquerade as the WAD.
function ensureWad() {
  const wad = join(DCACHE, 'doom1.wad');
  const isIwad = (buf) => buf.subarray(0, 4).toString('latin1') === 'IWAD';
  if (existsSync(wad) && isIwad(readFileSync(wad))) return wad;
  mkdirSync(DCACHE, { recursive: true });
  const tmp = join(DCACHE, 'doom1.wad.part');
  for (const url of WAD_MIRRORS) {
    try {
      execFileSync('curl', ['-sfL', '--max-time', '180', '-o', tmp, url], { stdio: 'inherit' });
      const raw = readFileSync(tmp);
      const buf = url.endsWith('.gz') ? gunzipSync(raw) : raw;
      if (!isIwad(buf)) throw new Error(`not an IWAD (magic ${JSON.stringify(buf.subarray(0, 4).toString('latin1'))})`);
      writeFileSync(wad, buf);
      return wad;
    } catch (e) {
      // Say which mirror failed and why — a silent `catch` here is what hid the outage.
      console.log(`    – WAD mirror ${new URL(url).host} unavailable: ${e.message}`);
    } finally {
      rmSync(tmp, { force: true });
    }
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
  // Name the half that failed. The old catch-all ("no toolchain, or the source/WAD fetch failed")
  // was printed even when the module had just built successfully one line above, which sent the
  // WAD-mirror outage looking like a toolchain problem.
  const missing = [!doomSvmb && 'module build', !doomWad && 'doom1.wad fetch'].filter(Boolean).join(' + ');
  console.log(`  – doom skipped (${missing} failed — offline, or no toolchain?)`);
}

console.log('done. Assets in web/assets/. Serve with `node serve.mjs` and open /web/play.html');
