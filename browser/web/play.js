// The SVM **playground** — the human-facing demo the THREADS/BROWSER work builds toward: type SVM
// text, it parses/verifies/encodes *inside the wasm sandbox* (`svm_parse`), and runs across real Web
// Workers (`par.js`, the same orchestration the validation page uses). The powerbox select picks the
// run's recipe: none (compute only), 4d host I/O (stdout read back onto the page), §22 guest-JIT, or
// a §14 root `Instantiator` (sandboxed children on their own Workers). The page services no
// authority either way — all of it is Rust-side, in shared linear memory.

import { loadEngine, makeRunner, readParStdout } from './par.js';
import { openJitReactor } from './wasmjit-reactor.js';
import { initWebGPU, teardownWebGPU, webgpuAvailable } from './webgpu.js';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += m + '\n'; };
const setState = (state, text) => { const e = $('state'); e.dataset.state = state; e.textContent = text; };

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
    mode: 'io',
    desc: 'crates/svm-run/demos/display/bounce.c — a C guest whose exported tick() runs one frame. ' +
      'Click Run, then steer the box with the arrow keys: the page calls tick() once per animation ' +
      'frame (the reactor run model), feeding key events in through the `keyboard` capability and ' +
      'blitting the frame it presents through `display`. State persists between frames. This is the ' +
      'interactive per-frame loop + input path Doom rides. Click Stop to end the loop.',
  },
  'life (Conway — heap persistence)': {
    kind: 'reactor',
    url: './assets/life.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/display/life.c — Conway’s Game of Life. Its cell grid lives in the ' +
      'malloc heap (which the on-ramp grows above the mapped window — exactly where Doom’s allocator ' +
      'will sit). Each tick computes the next generation from the current one, so the glider only ' +
      'advances if the reactor persists the guest’s whole memory (heap included) between frames. ' +
      'Click Run to watch it evolve; Stop to end. This is the heap-persistence proof Doom needs.',
  },
  'Mandelbrot zoom (interactive — arrow keys)': {
    kind: 'reactor',
    url: './assets/mandelzoom.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/display/mandelzoom.c — a C guest whose exported tick() computes a ' +
      'full double-precision Mandelbrot for the current view (in the sandbox, on the CPU — no GPU) ' +
      'and presents the RGBA frame through the `display` capability. Click Run: it auto-zooms toward ' +
      'a seahorse valley with a cycling rainbow palette; steer the zoom target with the arrow keys. ' +
      'Every frame is a fresh ~43k-pixel escape-time render on the wasm interpreter, so it runs at a ' +
      'few FPS (like Doom) — the compute is all guest code, only the finished frame crosses the ' +
      'capability boundary. Click Stop to end.',
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
};

// Size the run's shared window from the source's `memory N` declaration (64 KiB minimum — the wasm
// page granularity §14 carves align to; 16 MiB cap keeps a typo from asking for the whole memory).
function winSizeOf(src) {
  const m = /^\s*memory\s+(\d+)/m.exec(src);
  const log2 = Math.min(Math.max(m ? Number(m[1]) : 16, 16), 24);
  return 1 << log2;
}

let eng, run, aborter = null, broken = false;

function loadExample(name) {
  stopReactor(); // switching examples ends any running reactor loop
  const ex = EXAMPLES[name];
  $('mode').value = ex.mode;
  $('desc').textContent = ex.desc;
  // The "wasm-JIT" toggle is only meaningful for a JIT-emittable reactor (Doom): show it there.
  if ($('jitLabel')) $('jitLabel').hidden = !ex.jit;
  if (ex.kind === 'reactor') {
    // A per-frame reactor module: the "source" is binary; click Run to start the loop, arrow keys steer.
    $('src').value =
      `// ${name}\n// A pre-built on-ramp reactor module: ${ex.url}\n// Click Run — the page calls the ` +
      `guest's tick() once per animation frame\n// (svm_onramp_open/frame), and the arrow keys steer ` +
      `it via the keyboard capability.`;
    $('src').readOnly = true;
  } else if (ex.kind === 'module' && !ex.editable) {
    // A pre-built on-ramp module with a fixed program: the "source" is binary, not editable. Show a note.
    $('src').value =
      `// ${name}\n// A pre-built on-ramp module: ${ex.url}\n// Click Run — it executes as a real ` +
      `C/C++ guest via svm_run_onramp,\n// and its stdout appears in the pane on the right.`;
    $('src').readOnly = true;
  } else {
    // Text examples, and **editable** modules (whose source is fed to the guest as stdin), stay editable.
    $('src').value = ex.src;
    $('src').readOnly = false;
  }
}

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

// Blit the framebuffer the last run presented (via the `display` capability) to the canvas. w/h of 0
// ⇒ the guest presented no frame: hide the canvas. Copies the RGBA out of wasm memory into a fresh
// Uint8ClampedArray (putImageData rejects a SharedArrayBuffer-backed view, and a later alloc could
// detach the buffer). The canvas' intrinsic size is the frame's; CSS scales it up (pixelated).
function presentFrame(w, h) {
  const canvas = $('canvas');
  if (!w || !h) { canvas.style.display = 'none'; return; }
  const sp = eng.ex.svm_framebuffer_ptr();
  const sl = eng.ex.svm_framebuffer_len();
  const rgba = new Uint8ClampedArray(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  canvas.style.display = 'block';
  // No per-frame logging: the reactor loop calls this ~60×/second, which would flood the log pane.
}

// Run a pre-built on-ramp module single-shot on the main engine: alloc a buffer, copy the module in,
// `svm_run_onramp` (the fixed §3e powerbox — stdout/stdin/exit/memory), read the captured stdout.
// No Workers (these guests are single-threaded), so it never touches the par.js shared-window path.
async function runModule(ex) {
  setState('running', 'fetching module…');
  $('result').textContent = '';
  $('stdout').textContent = '';
  $('canvas').style.display = 'none';
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState('error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    log(`fetch failed: ${e.message}`);
    return;
  }
  log(`fetched ${ex.url}: ${bytes.length}B module`);
  const p = eng.ex.svm_alloc(bytes.length);
  // An editable module reads the editor text as **stdin** (the guest evaluates it — e.g. Lua). Alloc
  // both buffers *before* filling: svm_alloc may grow (detach) the linear memory, so take one fresh
  // view after all allocations and write into it.
  let stdinP = 0, stdinLen = 0, stdinBytes = null;
  if (ex.editable) {
    stdinBytes = new TextEncoder().encode($('src').value);
    if (stdinBytes.length > 0) {
      stdinP = eng.ex.svm_alloc(stdinBytes.length);
      stdinLen = stdinBytes.length;
    }
  }
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (stdinP) view.set(stdinBytes, stdinP);
  setState('running', 'running…');
  const t0 = performance.now();
  const rv = eng.ex.svm_run_onramp(p, bytes.length, stdinP, stdinLen);
  const status = eng.ex.svm_status();
  const sp = eng.ex.svm_stdout_ptr();
  const sl = eng.ex.svm_stdout_len();
  const stdout = new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
  // A framebuffer the guest presented through the `display` capability (0×0 ⇒ none): blit it to the
  // canvas. Read the RGBA out of wasm memory into a fresh copy — putImageData needs a non-shared
  // Uint8ClampedArray, and the slice also guards against the buffer detaching on a later alloc.
  const fbW = eng.ex.svm_framebuffer_width();
  const fbH = eng.ex.svm_framebuffer_height();
  presentFrame(fbW, fbH);
  eng.ex.svm_dealloc(p, bytes.length);
  if (stdinP) eng.ex.svm_dealloc(stdinP, stdinLen);
  const ms = (performance.now() - t0).toFixed(0);
  $('stdout').textContent = stdout;
  $('result').textContent = `${rv}`;
  // 0 = OK, 5 = clean Exit; anything else is a decode error / trap / unsupported.
  if (status === 0 || status === 5) {
    setState('done', `done · status ${status} · ${ms}ms`);
    log(`svm_run_onramp → ${rv} (status ${status}) in ${ms}ms`);
  } else {
    setState('error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    log(`svm_run_onramp status ${status}`);
  }
}

// ---- the reactor run model (interactive per-frame guests: bounce, eventually Doom) ----------------
// Open a reactor module once, then drive it one `tick` per requestAnimationFrame: each frame the
// guest runs, presents a frame (blitted to the canvas), and drains the key events we forwarded.
let reactorRAF = null; // the pending requestAnimationFrame id while a reactor loop runs (else null)
let jitReactor = null; // the wasm-JIT reactor driver while a JIT loop runs (else null → interpreter)

// Cancel any running reactor loop and free the guest instance. Safe to call when none is running.
function stopReactor() {
  teardownWebGPU(); // drop any GPU device + the servicer (no-op for non-webgpu reactors)
  const gc = $('gpucanvas');
  if (gc) gc.style.display = 'none';
  if (reactorRAF === null) return;
  cancelAnimationFrame(reactorRAF);
  reactorRAF = null;
  if (jitReactor) {
    jitReactor.close();
    jitReactor = null;
  } else {
    eng.ex.svm_onramp_close();
  }
}

async function runReactor(ex) {
  stopReactor();
  setState('running', 'fetching module…');
  $('result').textContent = '';
  $('stdout').textContent = '';
  $('canvas').style.display = 'none';
  // The "wasm-JIT" toggle runs an emittable reactor's whole tick() on emitted wasm (near-native) rather
  // than the interpreter. Only offered for JIT-capable examples (Doom); falls back if the emit fails.
  const useJit = !!(ex.jit && $('jit') && $('jit').checked);
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState('error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    return;
  }
  // A GPU demo: bring up a `navigator.gpu` device + the WebGPU canvas and install the servicer BEFORE
  // the reactor's first `tick`, so the guest's `webgpu` capability calls (set_shader/present) render.
  // Device creation is the only async step — awaited here, off the main-thread reactor loop.
  if (ex.webgpu) {
    if (!webgpuAvailable()) {
      setState('error', 'no WebGPU in this browser — the GPU demo needs it (try Chrome/Edge)');
      return;
    }
    try {
      $('gpucanvas').style.display = 'block';
      await initWebGPU($('gpucanvas'));
      log('WebGPU device ready — the guest ships one WGSL shader; the GPU renders every frame');
    } catch (e) {
      setState('error', `WebGPU init failed: ${e.message}`);
      $('gpucanvas').style.display = 'none';
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
      setState('error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate the WAD`);
      return;
    }
    log(`fetched ${ex.wad}: ${wad.length}B file (served through the fs capability)`);
  }
  setState('running',
    ex.wad ? `booting DOOM… (reading the WAD, building the renderer — a few seconds)${useJit ? ' [wasm-JIT]' : ''}`
      : 'running…');
  if (useJit) {
    // wasm-JIT reactor: the cdylib emits the whole tick(); this JS module compiles + runs it. On any
    // open/emit failure fall back to the interpreter reactor so the demo still plays.
    try {
      jitReactor = await openJitReactor(eng.ex, eng.memory, bytes, 'doom1.wad', wad);
      log(`wasm-JIT reactor opened: ${ex.url} (${bytes.length}B) — tick() runs on emitted wasm`);
    } catch (e) {
      jitReactor = null;
      log(`wasm-JIT reactor unavailable (${e.message}); falling back to the interpreter`);
    }
  }
  if (!jitReactor) {
    let opened;
    if (ex.wad) {
      const nameBytes = new TextEncoder().encode('doom1.wad');
      // Alloc all three buffers BEFORE filling any: svm_alloc may grow (detach) linear memory, so take
      // one fresh view after the last alloc and write into that.
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
      setState('error', `reactor open failed: status ${eng.ex.svm_status()} (2=unsupported 3=trap)`);
      log(`svm_onramp_open failed: ${opened}`);
      return;
    }
    log(`reactor opened: ${ex.url} (${bytes.length}B) — arrow keys steer, Stop ends`);
  }
  const tier = jitReactor ? 'wasm-JIT' : 'interpreter';
  setState('running', `running (${tier}) — arrow keys to steer, Stop to end`);
  $('run').disabled = true;
  $('stop').disabled = false;
  let frames = 0;
  const t0 = performance.now();
  let fpsFrames = 0;
  let fpsT0 = t0;
  const loop = () => {
    // One tick: 0 = keep going, 5 = the guest exited, else a trap. The JIT driver runs the emitted
    // tick and stashes the frame; the interpreter path runs it in-Rust — both fill svm_framebuffer_*.
    const status = jitReactor ? jitReactor.frame() : eng.ex.svm_onramp_frame();
    presentFrame(eng.ex.svm_framebuffer_width(), eng.ex.svm_framebuffer_height());
    frames++;
    fpsFrames++;
    // Surface a live FPS reading each second so the tier's frame rate is visible.
    const now = performance.now();
    if (now - fpsT0 >= 1000) {
      const fps = (fpsFrames * 1000 / (now - fpsT0)).toFixed(1);
      setState('running', `running (${tier}) — ${fps} fps · arrow keys to steer, Stop to end`);
      fpsFrames = 0;
      fpsT0 = now;
    }
    if (status === 0) {
      reactorRAF = requestAnimationFrame(loop);
      return;
    }
    reactorRAF = null;
    // On a trap, read the Trap variant (the diagnostic export stashes it into the stdout buffer) BEFORE
    // close() frees the reactor, so the log says *why* it stopped, not just the status code.
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
    $('run').disabled = broken;
    $('stop').disabled = true;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setState(status === 5 ? 'done' : 'error',
      status === 5 ? `guest exited after ${frames} frames · ${secs}s`
        : `reactor trapped: status ${status}${trapDetail ? ` (${trapDetail})` : ''}`);
    log(`reactor stopped (${tier}): status ${status}${trapDetail ? ` ${trapDetail}` : ''} after ${frames} frames in ${secs}s`);
  };
  reactorRAF = requestAnimationFrame(loop);
}

async function doRun() {
  if (broken) return;
  stopReactor(); // a fresh Run supersedes any running reactor loop
  // A pre-built on-ramp module runs single-shot via svm_run_onramp — no in-browser parse, no Workers.
  const selected = EXAMPLES[$('example').value];
  if (selected?.kind === 'reactor') return runReactor(selected);
  if (selected?.kind === 'module') return runModule(selected);
  // Leave the terminal states synchronously on click, so an observer (the Playwright smoke) that
  // clicks Run and polls for done/error never reads the PREVIOUS run's state.
  setState('running', 'parsing…');
  const src = $('src').value;
  const mode = $('mode').value;
  $('result').textContent = '';
  $('stdout').textContent = '';
  $('canvas').style.display = 'none';

  // 1) front end, inside the sandbox: SVM text → parse → verify → encoded module bytes.
  const u8 = () => new Uint8Array(eng.memory.buffer);
  const srcBytes = new TextEncoder().encode(src);
  let guest;
  if (srcBytes.length === 0) {
    setState('error', 'parse error: empty source');
    return;
  }
  {
    const p = eng.ex.svm_alloc(srcBytes.length);
    u8().set(srcBytes, p);
    const ok = eng.ex.svm_parse(p, srcBytes.length);
    eng.ex.svm_dealloc(p, srcBytes.length);
    // `slice` copies out of the SharedArrayBuffer (the stash may move on the next call).
    const out = u8().slice(eng.ex.svm_parse_ptr(), eng.ex.svm_parse_ptr() + eng.ex.svm_parse_len());
    if (ok !== 1) {
      setState('error', new TextDecoder().decode(out));
      return;
    }
    guest = out;
  }
  log(`parsed: ${srcBytes.length}B text → ${guest.length}B module`);

  // 2) run it across Workers under the selected powerbox recipe.
  aborter = new AbortController();
  $('run').disabled = true;
  $('stop').disabled = false;
  setState('running', 'running…');
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
    $('result').textContent = `${value}`;
    if (mode === 'io') $('stdout').textContent = readParStdout(eng);
    setState('done', `done: ${started} Worker${started === 1 ? '' : 's'} · ${ms}ms`);
    log(`run → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    if (e.message === 'stopped') {
      // Workers were torn down mid-run; shared state (locks, the live-vCPU counter) may be wedged.
      broken = true;
      setState('stopped', 'stopped — reload the page to run again');
      log('stopped by user');
    } else {
      setState('error', `run error: ${e.message}`);
      log(`run error: ${e.message}`);
    }
  } finally {
    aborter = null;
    $('run').disabled = broken;
    $('stop').disabled = true;
  }
}

async function main() {
  for (const name of Object.keys(EXAMPLES)) {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = name;
    $('example').appendChild(o);
  }
  loadExample('hello');
  $('example').addEventListener('change', () => loadExample($('example').value));
  $('run').addEventListener('click', doRun);
  $('stop').addEventListener('click', () => {
    if (reactorRAF !== null) {
      stopReactor();
      $('run').disabled = broken;
      $('stop').disabled = true;
      setState('stopped', 'stopped');
    } else {
      aborter?.abort();
    }
  });
  // Forward keys to a running reactor guest through the `keyboard` capability (as JS keyCodes — the
  // guest maps them: bounce steers on the arrows; Doom adds Ctrl fire / Space use / Enter·Esc·Tab
  // menus / Shift run / the letter keys y·n·etc.). Only while a loop is running. `preventDefault` is
  // limited to the keys whose default would disrupt play (arrows/Space/Tab scroll or move focus), and
  // never fires for a browser shortcut (Ctrl/Meta + a letter — e.g. Ctrl+R), so reload etc. still work.
  const REACTOR_KEYS = new Set([37, 38, 39, 40, 17, 32, 13, 27, 9, 16]); // arrows + Ctrl/Space/Enter/Esc/Tab/Shift
  for (let c = 65; c <= 90; c++) REACTOR_KEYS.add(c); // A–Z (Doom uses y/n and cheat/menu letters)
  const SWALLOW = new Set([37, 38, 39, 40, 32, 9]); // keys whose default (scroll/focus) must be suppressed
  const forward = (pressed) => (e) => {
    if (reactorRAF === null || !REACTOR_KEYS.has(e.keyCode)) return;
    if (jitReactor) eng.ex.svm_onramp_jit_key(e.keyCode, pressed);
    else eng.ex.svm_onramp_key(e.keyCode, pressed);
    const shortcut = (e.ctrlKey || e.metaKey) && e.keyCode !== 17; // leave Ctrl+R etc. to the browser
    if (SWALLOW.has(e.keyCode) && !shortcut) e.preventDefault();
  };
  window.addEventListener('keydown', forward(1));
  window.addEventListener('keyup', forward(0));

  if (!self.crossOriginIsolated) {
    setState('error', 'no cross-origin isolation (SharedArrayBuffer unavailable) — serve via serve.mjs');
    return;
  }
  try {
    eng = await loadEngine();
    run = makeRunner(eng);
  } catch (e) {
    setState('error', `engine load failed: ${e.message}`);
    return;
  }
  log(`engine loaded; shared=${eng.memory.buffer instanceof SharedArrayBuffer}`);
  $('run').disabled = false;
  setState('ready', 'ready');
}

main().catch((e) => setState('error', `fatal: ${e.message}`));
