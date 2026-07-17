// The SVM **playground** — the human-facing demo the THREADS/BROWSER work builds toward: type SVM
// text, it parses/verifies/encodes *inside the wasm sandbox* (`svm_parse`), and runs across real Web
// Workers (`par.js`, the same orchestration the validation page uses). The powerbox select picks the
// run's recipe: none (compute only), 4d host I/O (stdout read back onto the page), §22 guest-JIT, or
// a §14 root `Instantiator` (sandboxed children on their own Workers). The page services no
// authority either way — all of it is Rust-side, in shared linear memory.

import { loadEngine, makeRunner, readParStdout } from './par.js';
import { openJitReactor } from './wasmjit-reactor.js';
import { initWebGPU, teardownWebGPU, webgpuAvailable } from './webgpu.js';
import { createEditor, setVimAll, refreshAll } from './editor.js';

const $ = (id) => document.getElementById(id);

// Each example: the SVM text, its powerbox mode, and what to expect. The kernels are the proven
// schedule-independent ones from `gencorpus.rs` (same ground truths the validation page asserts).
const EXAMPLES = {
  hello: {
    mode: 'io',
    desc: 'One vCPU cap.call-writes a greeting through the host-I/O powerbox and returns the byte ' +
      'count (14). stdout comes back onto the page after the run.',
    src: `memory 16
data 0 "hello, world!\\n"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 14
  v3 = cap.call 0 1 (i64, i64) -> (i64) v0(v1, v2)
  return v3
}
`,
  },
  threads: {
    mode: 'plain',
    desc: 'thread.spawn fans 8 vCPUs out — each onto its own real Web Worker — every one ' +
      'atomic.rmw.adds a shared counter 500 times, the root joins them and returns 4000 on every ' +
      'interleaving.',
    src: `memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
`,
  },
  io: {
    mode: 'io',
    desc: '8 worker vCPUs (one Web Worker each) all cap.call-write "tick\\n" through the run\'s ONE ' +
      'shared powerbox and bump a shared counter — result 8, stdout "tick\\n" × 8, on every schedule.',
    src: `memory 16
data 0 "tick\\n"
func (i32) -> (i64) {
block0(v0: i32):
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br block1(v1, vh0)
block1(vi: i64, vhh: i64):
  v2 = i64.const 8
  v3 = i64.lt_u vi v2
  br_if v3 block2(vi, vhh) block3()
block2(vi2: i64, vhh2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vhh2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vhh2)
block3():
  v10 = i64.const 0
  br block4(v10)
block4(vj: i64):
  v11 = i64.const 8
  v12 = i64.lt_u vj v11
  br_if v12 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vh: i64):
  vhandle = i32.wrap_i64 vh
  vptr = i64.const 0
  vlen = i64.const 5
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
}
`,
  },
  jit: {
    mode: 'jit',
    desc: '§22 guest-JIT: 8 worker vCPUs each install a host-compiled unit into the SHARED Domain ' +
      '(a freshly raced dispatch slot) and call_indirect it — service(6,7) = 142, folded to 1136.',
    src: `memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vslot32 = i32.wrap_i64 vslot
  va = i32.const 6
  vb = i32.const 7
  vr = call_indirect (i32, i32) -> (i32) vslot32 (va, vb)
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
}
`,
  },
  inst: {
    mode: 'inst',
    desc: '§14 sandboxing: the root instantiates 8 confined children — each on its OWN Web Worker, ' +
      'confined to a 64 KiB carve of the 1 MiB window with an attenuated powerbox — joins them and ' +
      'sums 8 × 5 = 40.',
    src: `memory 20
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  vone = i64.const 1
  viplus = i64.add vi2 vone
  v64k = i64.const 65536
  voff = i64.mul viplus v64k
  ventry = i64.const 1
  vslog = i64.const 16
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  vinext = i64.add vi2 vone
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  return v1
}
`,
  },

  // ---- on-ramp modules: real C/C++ guests, compiled through clang → svm-llvm and run as a
  //      pre-built .svmb via `svm_run_onramp` (no in-browser parse). Built by
  //      `build-onramp-assets.mjs` at `--host-page 65536` (the wasm page). ------------------------
  'hello (C → SVM)': {
    kind: 'module',
    url: './assets/hello_c.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/hello.c — a C program compiled with stock clang, translated by the ' +
      'LLVM on-ramp, and run through the powerbox: it write(1, …)s a greeting and exits. The output ' +
      'below is the guest’s real stdout.',
  },
  'gradient (C → framebuffer)': {
    kind: 'module',
    url: './assets/gradient.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/display/gradient.c — a C guest renders a 128×128 RGBA image and ' +
      'presents one frame through the `display` capability (resolved by name, like Lua’s io / ' +
      'SQLite’s VFS). The host reads the frame out of guest memory and blits it to the canvas on the ' +
      'right. This is the framebuffer output path the graphical demos (Doom) ride.',
  },
  'bounce (interactive — arrow keys)': {
    kind: 'reactor',
    url: './assets/bounce.svmb',
    jit: true, // tick() emits after cap-call outlining — toggle "wasm-JIT" to run it near-natively
    mode: 'io',
    desc: 'crates/svm-run/demos/display/bounce.c — a C guest whose exported tick() runs one frame. ' +
      'Click Run, then steer the box with the arrow keys: the page calls tick() once per animation ' +
      'frame (the reactor run model), feeding key events in through the `keyboard` capability and ' +
      'blitting the frame it presents through `display`. State persists between frames. This is the ' +
      'interactive per-frame loop + input path Doom rides. Toggle "wasm-JIT" to run the whole tick() ' +
      'on emitted wasm instead of the interpreter. Click Stop to end the loop.',
  },
  'life (Conway — heap persistence)': {
    kind: 'reactor',
    url: './assets/life.svmb',
    jit: true, // tick() emits after cap-call outlining — toggle "wasm-JIT" to run it near-natively
    mode: 'io',
    desc: 'crates/svm-run/demos/display/life.c — Conway’s Game of Life. Its cell grid lives in the ' +
      'malloc heap (which the on-ramp grows above the mapped window — exactly where Doom’s allocator ' +
      'will sit). Each tick computes the next generation from the current one, so the glider only ' +
      'advances if the reactor persists the guest’s whole memory (heap included) between frames. ' +
      'Click Run to watch it evolve; Stop to end. Toggle "wasm-JIT" to run the whole tick() on ' +
      'emitted wasm instead of the interpreter. This is the heap-persistence proof Doom needs.',
  },
  'Mandelbrot zoom (interactive — arrow keys)': {
    kind: 'reactor',
    url: './assets/mandelzoom.svmb',
    jit: true, // f64 tick() emits after cap-call outlining — toggle "wasm-JIT" for a ~24× speedup
    mode: 'io',
    desc: 'crates/svm-run/demos/display/mandelzoom.c — a C guest whose exported tick() computes a ' +
      'full double-precision Mandelbrot for the current view (in the sandbox, on the CPU — no GPU) ' +
      'and presents the RGBA frame through the `display` capability. Click Run: it auto-zooms toward ' +
      'a seahorse valley with a cycling rainbow palette; steer the zoom target with the arrow keys. ' +
      'Every frame is a fresh ~43k-pixel escape-time render; on the wasm interpreter that runs at a ' +
      'few FPS, so toggle "wasm-JIT" to run the whole tick() on emitted wasm — the f64 escape loop ' +
      'then runs near-natively (~24× faster here) and the frame rate jumps (shown live). The compute ' +
      'is all guest code; only the finished frame crosses the capability boundary. Click Stop to end.',
  },
  'GPU: Mandelbrot zoom (WebGPU shader)': {
    kind: 'reactor',
    url: './assets/gpu_shader.svmb',
    mode: 'io',
    webgpu: true,
    desc: 'crates/svm-run/demos/display/gpu_shader.c — a sandboxed C guest ships a WGSL fragment ' +
      'shader once through a `webgpu` capability, then asks the host to present a frame each tick. ' +
      'The Mandelbrot escape-time loop runs on the **GPU** (via the browser’s WebGPU / navigator.gpu), ' +
      'so it stays smooth at 640×480 while zooming into a seahorse valley — only the tiny (frame, w, h) ' +
      'scalars cross the capability boundary per frame, and the guest never holds a GPU pointer. ' +
      'Needs a WebGPU-capable browser (Chrome/Edge, recent Firefox). Click Stop to end.',
  },
  'DOOM (1993 — arrow keys, Ctrl fires)': {
    kind: 'reactor',
    url: './assets/doom.svmb',
    wad: './assets/doom1.wad',
    jit: true, // the whole tick() is wasm-JIT-emittable — the "wasm-JIT" toggle runs it near-natively
    mode: 'io',
    desc: 'Shareware DOOM (via doomgeneric), compiled from id Software’s C through the LLVM on-ramp ' +
      'and run in the sandbox. Click Run: _start reads the IWAD through the `fs` capability and boots ' +
      'Doom’s whole engine, then the page calls the guest’s tick() once per animation frame (the ' +
      'reactor loop), blitting each 320×200 frame it presents through `display`. Arrow keys move, ' +
      'Ctrl fires, Space uses doors/switches, Esc/Enter drive the menus. The zone heap persists in ' +
      'the guest window between frames (slice 3a). Boot takes a few seconds on the wasm interpreter — ' +
      'the renderer is byte-exact to a native build (the §18 differential). Toggle "wasm-JIT" to run ' +
      'the whole tick() on emitted wasm (near-native) instead of the interpreter — it multiplies the ' +
      'frame rate (shown live). Click Stop to end.',
  },
  'Lua (5.4.7 — write & run)': {
    kind: 'module',
    editable: true,
    lang: 'lua',
    url: './assets/lua_eval.svmb',
    mode: 'io',
    desc: 'Lua 5.4.7 — its core (lexer, parser, GC, bytecode VM) plus the base/string/table/math/' +
      'coroutine/io/os libraries, compiled through the LLVM on-ramp. Edit the Lua on the left and ' +
      'click Run: your code is piped to the guest as stdin, evaluated, and its output appears below. ' +
      'Real Lua, running client-side in the sandbox.',
    src: `-- Write Lua here, then click Run.
print("Hello from " .. _VERSION)

-- recursion
local function fib(n) return n < 2 and n or fib(n - 1) + fib(n - 2) end
local out = {}
for i = 1, 10 do out[i] = fib(i) end
print("fib(1..10):", table.concat(out, " "))

-- tables + sort
local t = { 5, 3, 8, 1, 9, 2, 7 }
table.sort(t)
print("sorted:", table.concat(t, ", "))

-- string.format + math
print(string.format("pi ~ %.4f, 255 in hex = 0x%X", math.pi, 255))

-- io.write (stdout via the Stream capability — no trailing newline)
io.write("counting: ")
for i = 1, 5 do io.write(i, " ") end
io.write("\\n")

-- coroutines: a lazy generator
local function squares(n)
  return coroutine.wrap(function()
    for i = 1, n do coroutine.yield(i * i) end
  end)
end
local sq = {}
for v in squares(6) do sq[#sq + 1] = v end
print("squares:", table.concat(sq, " "))
`,
  },
  'SQLite (:memory: — write & run SQL)': {
    kind: 'module',
    editable: true,
    lang: 'sql',
    url: './assets/sqlite_repl.svmb',
    mode: 'io',
    desc: 'The unmodified SQLite 3.50.2 amalgamation (~257k lines of C), compiled through the LLVM ' +
      'on-ramp. Edit the SQL on the left and click Run: it executes against a fresh in-memory ' +
      'database (each Run starts clean) and prints result tables, change counts, and errors below. ' +
      'Real SQLite, running client-side in the sandbox.',
    src: `-- Write SQL here, then click Run. Each Run is a fresh :memory: database.
CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INT);
INSERT INTO users(name, age) VALUES ('Ada', 36), ('Alan', 41), ('Grace', 45), ('Edsger', 40);

SELECT name, age FROM users WHERE age >= 40 ORDER BY age DESC;

SELECT count(*) AS n, avg(age) AS avg_age, max(age) AS oldest FROM users;

-- a recursive CTE: the first 10 Fibonacci numbers
WITH RECURSIVE fib(n, a, b) AS (
  SELECT 1, 0, 1 UNION ALL SELECT n + 1, b, a + b FROM fib WHERE n < 10
)
SELECT n, a AS fib FROM fib;
`,
  },
  'PostgreSQL (17.5 — write & run SQL)': {
    kind: 'pg',
    editable: true,
    lang: 'sql',
    url: './assets/postgres_resolved.svmb',
    image: './assets/pgdata.img',
    mode: 'io',
    desc: 'A whole, unmodified PostgreSQL 17.5 --single backend — ~15,000 functions compiled LLVM → ' +
      'SVM IR, verified, and run on the bytecode interpreter inside wasm. Its data directory is an ' +
      'in-memory image mounted on a capability-scoped filesystem — no host filesystem, network, or ' +
      'ambient authority. Edit the SQL on the left and click Run: each Run is a fresh boot of the ' +
      'backend (a few seconds), so this is real Postgres running client-side in the sandbox. The two ' +
      'large artifacts (a ~20 MB module + a ~40 MB data image) download once.',
    src: `-- Write SQL here, then click Run. Each Run is a fresh postgres --single boot.
CREATE TABLE t (x int, s text);
INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three');
SELECT * FROM t WHERE x > 1 ORDER BY x DESC;
SELECT count(*), sum(x), avg(x) FROM t;
`,
  },
};

// Size the run's shared window from the source's `memory N` declaration (64 KiB minimum — the wasm
// page granularity §14 carves align to; 16 MiB cap keeps a typo from asking for the whole memory).
function winSizeOf(src) {
  const m = /^\s*memory\s+(\d+)/m.exec(src);
  const log2 = Math.min(Math.max(m ? Number(m[1]) : 16, 16), 24);
  return 1 << log2;
}

// ---- per-card run machinery ----------------------------------------------------------------------
// Each demo renders as a self-contained card (its own editor + controls + output). The run functions
// take that card's context `c`, so state never leaks between cards, and only one run is ever active at
// a time (a fresh Run supersedes any running reactor). `eng`/`run` are the shared wasm engine; `broken`
// latches when a threaded run is Stopped mid-flight (shared state may wedge → every card's Run disables).
let eng, run, aborter = null, broken = false;
const cards = [];

const setState = (c, state, text) => { c.el.state.dataset.state = state; c.el.state.textContent = text; };
const logTo = (c, m) => { c.el.log.textContent += m + '\n'; };
const setEngineState = (state, text) => { const e = $('engine-state'); e.dataset.state = state; e.textContent = text; };

// Fetched `.svmb` bytes, cached (a 6 MB SQLite module is worth not re-downloading on every Run).
const moduleCache = new Map();
async function fetchModule(url) {
  if (moduleCache.has(url)) return moduleCache.get(url);
  // Resolve module URLs relative to this script (not the document), so they work under any base path
  // (origin root locally, `/<repo>/` on GitHub Pages).
  const resolved = new URL(url, import.meta.url);
  const r = await fetch(resolved);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  moduleCache.set(url, bytes);
  return bytes;
}

// Blit the framebuffer the last run presented (via the `display` capability) to this card's canvas.
// w/h of 0 ⇒ no frame: hide the canvas. Copies the RGBA out of wasm memory into a fresh
// Uint8ClampedArray (putImageData rejects a SharedArrayBuffer-backed view, and a later alloc could
// detach the buffer). The canvas' intrinsic size is the frame's; CSS scales it up (pixelated).
function presentFrame(c, w, h) {
  const canvas = c.el.canvas;
  if (!w || !h) { canvas.hidden = true; return; }
  const sp = eng.ex.svm_framebuffer_ptr();
  const sl = eng.ex.svm_framebuffer_len();
  const rgba = new Uint8ClampedArray(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  canvas.hidden = false;
}

// Run a pre-built on-ramp module single-shot on the main engine: alloc a buffer, copy the module in,
// `svm_run_onramp` (the fixed §3e powerbox — stdout/stdin/exit/memory), read the captured stdout. No
// Workers (these guests are single-threaded), so it never touches the par.js shared-window path.
async function runModule(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching module…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    logTo(c, `fetch failed: ${e.message}`);
    return;
  }
  logTo(c, `fetched ${ex.url}: ${bytes.length}B module`);
  const p = eng.ex.svm_alloc(bytes.length);
  // An editable module reads the editor text as **stdin** (the guest evaluates it — e.g. Lua). Alloc
  // both buffers *before* filling: svm_alloc may grow (detach) the linear memory, so take one fresh
  // view after all allocations and write into it.
  let stdinP = 0, stdinLen = 0, stdinBytes = null;
  if (ex.editable) {
    stdinBytes = new TextEncoder().encode(c.editor.getValue());
    if (stdinBytes.length > 0) {
      stdinP = eng.ex.svm_alloc(stdinBytes.length);
      stdinLen = stdinBytes.length;
    }
  }
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (stdinP) view.set(stdinBytes, stdinP);
  setState(c, 'running', 'running…');
  const t0 = performance.now();
  const rv = eng.ex.svm_run_onramp(p, bytes.length, stdinP, stdinLen);
  const status = eng.ex.svm_status();
  const sp = eng.ex.svm_stdout_ptr();
  const sl = eng.ex.svm_stdout_len();
  const stdout = new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
  const fbW = eng.ex.svm_framebuffer_width();
  const fbH = eng.ex.svm_framebuffer_height();
  presentFrame(c, fbW, fbH);
  eng.ex.svm_dealloc(p, bytes.length);
  if (stdinP) eng.ex.svm_dealloc(stdinP, stdinLen);
  const ms = (performance.now() - t0).toFixed(0);
  c.el.stdout.textContent = stdout;
  c.el.result.textContent = `${rv}`;
  // 0 = OK, 5 = clean Exit; anything else is a decode error / trap / unsupported.
  if (status === 0 || status === 5) {
    setState(c, 'done', `done · status ${status} · ${ms}ms`);
    logTo(c, `svm_run_onramp → ${rv} (status ${status}) in ${ms}ms`);
  } else {
    setState(c, 'error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `svm_run_onramp status ${status}`);
  }
}

// Boot PostgreSQL `--single` single-shot on the main engine (the `svm_run_pg` entry): fetch the
// pre-translated+resolved module + the data image, feed the editor's SQL as stdin, mount the image on
// an in-memory `fs` cap, run to a queried backend, read the captured stdout. A fresh boot per Run.
async function runPg(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching module + image…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  let modBytes, imgBytes;
  try {
    [modBytes, imgBytes] = await Promise.all([fetchModule(ex.url), fetchModule(ex.image)]);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-pg-assets.mjs\` to stage the Postgres artifacts`);
    logTo(c, `fetch failed: ${e.message}`);
    return;
  }
  logTo(c, `fetched ${ex.url}: ${modBytes.length}B module, ${ex.image}: ${imgBytes.length}B image`);
  const sql = new TextEncoder().encode(c.editor.getValue());
  setState(c, 'running', 'booting postgres… (a few seconds — the whole backend runs in the sandbox)');
  c.el.run.disabled = true;
  // Yield one paint so "booting…" lands before the synchronous, multi-second boot blocks the thread.
  await new Promise((r) => setTimeout(r, 30));
  try {
    // Alloc all three before filling: svm_alloc may grow (detach) the linear memory.
    const modP = eng.ex.svm_alloc(modBytes.length);
    const imgP = eng.ex.svm_alloc(imgBytes.length);
    const inP = sql.length ? eng.ex.svm_alloc(sql.length) : 0;
    const view = new Uint8Array(eng.memory.buffer);
    view.set(modBytes, modP);
    view.set(imgBytes, imgP);
    if (inP) view.set(sql, inP);
    const t0 = performance.now();
    const rv = eng.ex.svm_run_pg(modP, modBytes.length, imgP, imgBytes.length, inP, sql.length);
    const ms = (performance.now() - t0).toFixed(0);
    const status = eng.ex.svm_status();
    const sp = eng.ex.svm_stdout_ptr();
    const sl = eng.ex.svm_stdout_len();
    const stdout = new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
    eng.ex.svm_dealloc(modP, modBytes.length);
    eng.ex.svm_dealloc(imgP, imgBytes.length);
    if (inP) eng.ex.svm_dealloc(inP, sql.length);
    c.el.stdout.textContent = stdout;
    c.el.result.textContent = `${rv}`;
    if (status === 0 || status === 5) {
      setState(c, 'done', `booted + ran in ${ms}ms · status ${status}`);
      logTo(c, `svm_run_pg → ${rv} (status ${status}) in ${ms}ms`);
    } else {
      setState(c, 'error', `boot failed: status ${status} (1=decode 3=trap 6=verify)`);
      logTo(c, `svm_run_pg status ${status}`);
    }
  } catch (e) {
    setState(c, 'error', `run error: ${e.message}`);
    logTo(c, `run error: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
  }
}

// ---- the reactor run model (interactive per-frame guests: bounce, life, Doom) --------------------
// Open a reactor module once, then drive it one `tick` per requestAnimationFrame. Only one reactor
// runs at a time; `activeReactorCard` is the card it belongs to (for teardown + the GPU canvas).
let reactorRAF = null; // the pending requestAnimationFrame id while a reactor loop runs (else null)
let jitReactor = null; // the wasm-JIT reactor driver while a JIT loop runs (else null → interpreter)
let activeReactorCard = null;

// Cancel any running reactor loop and free the guest instance. Safe to call when none is running.
function stopReactor() {
  teardownWebGPU(); // drop any GPU device + the servicer (no-op for non-webgpu reactors)
  if (activeReactorCard) activeReactorCard.el.gpucanvas.hidden = true;
  if (reactorRAF === null) { activeReactorCard = null; return; }
  cancelAnimationFrame(reactorRAF);
  reactorRAF = null;
  if (jitReactor) {
    jitReactor.close();
    jitReactor = null;
  } else {
    eng.ex.svm_onramp_close();
  }
  activeReactorCard = null;
}

async function runReactor(c) {
  const ex = c.ex;
  stopReactor();
  activeReactorCard = c;
  setState(c, 'running', 'fetching module…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  // The "wasm-JIT" toggle runs an emittable reactor's whole tick() on emitted wasm (near-native) rather
  // than the interpreter. Only offered for JIT-capable examples (Doom); falls back if the emit fails.
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    return;
  }
  // A GPU demo: bring up a `navigator.gpu` device + the WebGPU canvas and install the servicer BEFORE
  // the reactor's first `tick`, so the guest's `webgpu` capability calls (set_shader/present) render.
  if (ex.webgpu) {
    if (!webgpuAvailable()) {
      setState(c, 'error', 'no WebGPU in this browser — the GPU demo needs it (try Chrome/Edge)');
      return;
    }
    try {
      c.el.gpucanvas.hidden = false;
      await initWebGPU(c.el.gpucanvas);
      logTo(c, 'WebGPU device ready — the guest ships one WGSL shader; the GPU renders every frame');
    } catch (e) {
      setState(c, 'error', `WebGPU init failed: ${e.message}`);
      c.el.gpucanvas.hidden = true;
      return;
    }
  }
  // Open the reactor: alloc, copy the module in, run _start (decode + grant powerbox). A guest that
  // needs a served file (Doom reads its WAD at _start) is opened with svm_onramp_open_fs, which grants
  // the `fs` capability over the fetched blob; every other reactor guest uses plain svm_onramp_open.
  let wad = null;
  if (ex.wad) {
    try {
      wad = await fetchModule(ex.wad);
    } catch (e) {
      setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate the WAD`);
      return;
    }
    logTo(c, `fetched ${ex.wad}: ${wad.length}B file (served through the fs capability)`);
  }
  setState(c, 'running',
    ex.wad ? `booting DOOM… (reading the WAD, building the renderer — a few seconds)${useJit ? ' [wasm-JIT]' : ''}`
      : 'running…');
  if (useJit) {
    try {
      jitReactor = await openJitReactor(eng.ex, eng.memory, bytes, 'doom1.wad', wad);
      logTo(c, `wasm-JIT reactor opened: ${ex.url} (${bytes.length}B) — tick() runs on emitted wasm`);
    } catch (e) {
      jitReactor = null;
      logTo(c, `wasm-JIT reactor unavailable (${e.message}); falling back to the interpreter`);
    }
  }
  if (!jitReactor) {
    let opened;
    if (ex.wad) {
      const nameBytes = new TextEncoder().encode('doom1.wad');
      const modP = eng.ex.svm_alloc(bytes.length);
      const nameP = eng.ex.svm_alloc(nameBytes.length);
      const wadP = eng.ex.svm_alloc(wad.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(bytes, modP);
      view.set(nameBytes, nameP);
      view.set(wad, wadP);
      opened = eng.ex.svm_onramp_open_fs(modP, bytes.length, nameP, nameBytes.length, wadP, wad.length);
      eng.ex.svm_dealloc(modP, bytes.length);
      eng.ex.svm_dealloc(nameP, nameBytes.length);
      eng.ex.svm_dealloc(wadP, wad.length);
    } else {
      const p = eng.ex.svm_alloc(bytes.length);
      new Uint8Array(eng.memory.buffer).set(bytes, p);
      opened = eng.ex.svm_onramp_open(p, bytes.length);
      eng.ex.svm_dealloc(p, bytes.length);
    }
    if (opened !== 0) {
      setState(c, 'error', `reactor open failed: status ${eng.ex.svm_status()} (2=unsupported 3=trap)`);
      logTo(c, `svm_onramp_open failed: ${opened}`);
      activeReactorCard = null;
      return;
    }
    logTo(c, `reactor opened: ${ex.url} (${bytes.length}B) — arrow keys steer, Stop ends`);
  }
  const tier = jitReactor ? 'wasm-JIT' : 'interpreter';
  setState(c, 'running', `running (${tier}) — arrow keys to steer, Stop to end`);
  c.el.run.disabled = true;
  c.el.stop.disabled = false;
  let frames = 0;
  const t0 = performance.now();
  let fpsFrames = 0;
  let fpsT0 = t0;
  const loop = () => {
    const status = jitReactor ? jitReactor.frame() : eng.ex.svm_onramp_frame();
    presentFrame(c, eng.ex.svm_framebuffer_width(), eng.ex.svm_framebuffer_height());
    frames++;
    fpsFrames++;
    const now = performance.now();
    if (now - fpsT0 >= 1000) {
      const fps = (fpsFrames * 1000 / (now - fpsT0)).toFixed(1);
      setState(c, 'running', `running (${tier}) — ${fps} fps · arrow keys to steer, Stop to end`);
      fpsFrames = 0;
      fpsT0 = now;
    }
    if (status === 0) {
      reactorRAF = requestAnimationFrame(loop);
      return;
    }
    reactorRAF = null;
    let trapDetail = '';
    if (status !== 0 && status !== 5) {
      const n = jitReactor ? eng.ex.svm_onramp_jit_trap_len() : eng.ex.svm_onramp_trap_len();
      if (n > 0) {
        trapDetail = new TextDecoder().decode(
          new Uint8Array(eng.memory.buffer).slice(eng.ex.svm_stdout_ptr(), eng.ex.svm_stdout_ptr() + n));
      }
    }
    if (jitReactor) {
      jitReactor.close();
      jitReactor = null;
    } else {
      eng.ex.svm_onramp_close();
    }
    activeReactorCard = null;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setState(c, status === 5 ? 'done' : 'error',
      status === 5 ? `guest exited after ${frames} frames · ${secs}s`
        : `reactor trapped: status ${status}${trapDetail ? ` (${trapDetail})` : ''}`);
    logTo(c, `reactor stopped (${tier}): status ${status}${trapDetail ? ` ${trapDetail}` : ''} after ${frames} frames in ${secs}s`);
  };
  reactorRAF = requestAnimationFrame(loop);
}

// SVM **text** guests: parse+verify inside the sandbox (`svm_parse`), then run across Workers under the
// card's selected powerbox recipe.
async function runText(c) {
  setState(c, 'running', 'parsing…');
  const src = c.editor.getValue();
  const mode = c.el.mode.value;
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;

  const u8 = () => new Uint8Array(eng.memory.buffer);
  const srcBytes = new TextEncoder().encode(src);
  let guest;
  if (srcBytes.length === 0) {
    setState(c, 'error', 'parse error: empty source');
    return;
  }
  {
    const p = eng.ex.svm_alloc(srcBytes.length);
    u8().set(srcBytes, p);
    const ok = eng.ex.svm_parse(p, srcBytes.length);
    eng.ex.svm_dealloc(p, srcBytes.length);
    const out = u8().slice(eng.ex.svm_parse_ptr(), eng.ex.svm_parse_ptr() + eng.ex.svm_parse_len());
    if (ok !== 1) {
      const msg = new TextDecoder().decode(out);
      setState(c, 'error', msg);
      c.editor.markError(msg); // pin the offending line in the editor when we can locate it
      return;
    }
    guest = out;
  }
  logTo(c, `parsed: ${srcBytes.length}B text → ${guest.length}B module`);

  aborter = new AbortController();
  c.el.run.disabled = true;
  c.el.stop.disabled = false;
  setState(c, 'running', 'running…');
  const opts = {
    jit: mode === 'jit',
    inst: mode === 'inst',
    io: mode === 'io',
    winSize: winSizeOf(src),
    signal: aborter.signal,
  };
  const t0 = performance.now();
  try {
    const { value, started } = await run(guest, opts);
    const ms = (performance.now() - t0).toFixed(0);
    c.el.result.textContent = `${value}`;
    if (mode === 'io') c.el.stdout.textContent = readParStdout(eng);
    setState(c, 'done', `done: ${started} Worker${started === 1 ? '' : 's'} · ${ms}ms`);
    logTo(c, `run → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    if (e.message === 'stopped') {
      // Workers were torn down mid-run; shared state (locks, the live-vCPU counter) may be wedged.
      broken = true;
      setState(c, 'stopped', 'stopped — reload the page to run again');
      logTo(c, 'stopped by user');
      for (const card of cards) card.el.run.disabled = true;
    } else {
      setState(c, 'error', `run error: ${e.message}`);
      logTo(c, `run error: ${e.message}`);
    }
  } finally {
    aborter = null;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
  }
}

// A card's Run: supersede any running reactor, then dispatch by kind.
async function runDemo(c) {
  if (broken) return;
  if (c.editor) c.editor.clearError();
  stopReactor(); // a fresh Run supersedes any running reactor loop
  const ex = c.ex;
  if (ex.kind === 'reactor') return runReactor(c);
  if (ex.kind === 'pg') return runPg(c);
  if (ex.kind === 'module') return runModule(c);
  return runText(c);
}

// A card's Stop: end a running reactor, or abort a running threaded text run.
function stopDemo(c) {
  if (reactorRAF !== null) {
    stopReactor();
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    setState(c, 'stopped', 'stopped');
  } else {
    aborter?.abort();
  }
}

// ---- DOM: build one card per demo + the sidebar --------------------------------------------------
const slug = (name) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

const POWERBOX_MODES = [
  ['plain', 'none (compute only)'],
  ['io', 'host I/O (stdout)'],
  ['jit', 'guest JIT (§22)'],
  ['inst', 'instantiator (§14)'],
];

function buildCard(name, ex) {
  const section = el('section', 'demo');
  section.id = 'demo-' + slug(name);
  section.dataset.demo = name; // stable hook for tests
  section.append(el('h2', 'demo-title', name));
  section.append(el('p', 'desc', ex.desc || ''));

  // SVM text (no `kind`) and editable modules (Lua/SQL/Postgres) get an editor; a fixed C guest or a
  // reactor gets a lightweight read-only note (its "source" is a pre-built binary).
  const editable = !ex.kind || !!ex.editable;
  let editor = null;
  if (editable) {
    const ta = el('textarea');
    ta.value = ex.src || '';
    const wrap = el('div', 'editor');
    wrap.appendChild(ta);
    section.appendChild(wrap);
    editor = createEditor(ta, ex.lang || 'svm');
  } else {
    section.appendChild(el('pre', 'note',
      ex.kind === 'reactor'
        ? `Pre-built on-ramp reactor module (${ex.url}). Click Run — the page calls tick() once per animation frame; the arrow keys steer it through the keyboard capability.`
        : `Pre-built on-ramp module (${ex.url}). Click Run — it executes as a real C/C++ guest via svm_run_onramp; its stdout appears below.`));
  }

  const controls = el('div', 'controls');
  let modeSel = null;
  if (!ex.kind) {
    modeSel = el('select');
    for (const [v, label] of POWERBOX_MODES) {
      const o = el('option', null, label);
      o.value = v;
      modeSel.appendChild(o);
    }
    modeSel.value = ex.mode;
    const l = el('label', null, 'powerbox ');
    l.appendChild(modeSel);
    controls.appendChild(l);
  }
  const runBtn = el('button', 'run', 'Run');
  runBtn.disabled = true;
  const stopBtn = el('button', 'stop', 'Stop');
  stopBtn.disabled = true;
  controls.append(runBtn, stopBtn);
  let jit = null;
  if (ex.jit) {
    const l = el('label', 'jit-label');
    l.title = 'Run the reactor’s tick() on emitted wasm (wasm-JIT tier) instead of the interpreter';
    jit = el('input');
    jit.type = 'checkbox';
    jit.checked = true;
    l.append(jit, ' wasm-JIT');
    controls.appendChild(l);
  }
  const state = el('span', 'state', 'ready');
  state.dataset.state = 'ready';
  controls.appendChild(state);
  section.appendChild(controls);

  const out = el('div', 'output');
  const result = el('pre', 'result');
  const canvas = el('canvas', 'canvas');
  canvas.hidden = true;
  const gpucanvas = el('canvas', 'gpucanvas');
  gpucanvas.hidden = true;
  const stdout = el('pre', 'stdout');
  const logEl = el('pre', 'log');
  out.append(el('strong', null, 'result'), result, canvas, gpucanvas,
    el('strong', null, 'stdout'), stdout, el('strong', null, 'log'), logEl);
  section.appendChild(out);

  const c = {
    name, ex, editor,
    el: { section, state, result, stdout, log: logEl, canvas, gpucanvas, run: runBtn, stop: stopBtn, mode: modeSel, jit },
  };
  runBtn.addEventListener('click', () => runDemo(c));
  stopBtn.addEventListener('click', () => stopDemo(c));
  return c;
}

// The sidebar: one link per demo, scroll-spied so the in-view demo is highlighted, and a global Vim
// toggle. Clicking a link scrolls its card into view.
function buildSidebar() {
  const nav = $('nav-list');
  for (const c of cards) {
    const a = el('a', 'nav-link', c.name);
    a.href = '#' + c.el.section.id;
    a.dataset.target = c.el.section.id;
    nav.appendChild(a);
  }
  // Scroll-spy: highlight the link whose card is nearest the top of the viewport.
  const links = new Map([...nav.querySelectorAll('.nav-link')].map((a) => [a.dataset.target, a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const a = links.get(entry.target.id);
      if (a) a.classList.toggle('active', entry.isIntersecting);
    }
  }, { rootMargin: '-45% 0px -45% 0px' }); // a thin band across the vertical middle
  for (const c of cards) observer.observe(c.el.section);
}

async function main() {
  const demosEl = $('demos');
  for (const [name, ex] of Object.entries(EXAMPLES)) {
    const c = buildCard(name, ex);
    cards.push(c);
    demosEl.appendChild(c.el.section);
  }
  buildSidebar();
  refreshAll(); // lay the editors out now they're in the DOM
  $('vim').addEventListener('change', (e) => setVimAll(e.target.checked));

  // Forward keys to the running reactor guest through the `keyboard` capability (as JS keyCodes — the
  // guest maps them: bounce steers on the arrows; Doom adds Ctrl fire / Space use / Enter·Esc·Tab
  // menus / Shift run / the letter keys). Only while a loop is running. `preventDefault` is limited to
  // the keys whose default would disrupt play (arrows/Space/Tab scroll or move focus), and never fires
  // for a browser shortcut (Ctrl/Meta + a letter — e.g. Ctrl+R), so reload etc. still work.
  const REACTOR_KEYS = new Set([37, 38, 39, 40, 17, 32, 13, 27, 9, 16]);
  for (let k = 65; k <= 90; k++) REACTOR_KEYS.add(k);
  const SWALLOW = new Set([37, 38, 39, 40, 32, 9]);
  const forward = (pressed) => (e) => {
    if (reactorRAF === null || !REACTOR_KEYS.has(e.keyCode)) return;
    if (jitReactor) eng.ex.svm_onramp_jit_key(e.keyCode, pressed);
    else eng.ex.svm_onramp_key(e.keyCode, pressed);
    const shortcut = (e.ctrlKey || e.metaKey) && e.keyCode !== 17;
    if (SWALLOW.has(e.keyCode) && !shortcut) e.preventDefault();
  };
  window.addEventListener('keydown', forward(1));
  window.addEventListener('keyup', forward(0));

  if (!self.crossOriginIsolated) {
    setEngineState('error', 'no cross-origin isolation (SharedArrayBuffer unavailable) — serve via serve.mjs');
    return;
  }
  try {
    eng = await loadEngine();
    run = makeRunner(eng);
  } catch (e) {
    setEngineState('error', `engine load failed: ${e.message}`);
    return;
  }
  for (const c of cards) c.el.run.disabled = false;
  setEngineState('ready', 'engine ready');
}

main().catch((e) => setEngineState('error', `fatal: ${e.message}`));
