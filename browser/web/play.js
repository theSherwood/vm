// The SVM **playground** — the human-facing demo the THREADS/BROWSER work builds toward: type SVM
// text, it parses/verifies/encodes *inside the wasm sandbox* (`svm_parse`), and runs across real Web
// Workers (`par.js`, the same orchestration the validation page uses). The powerbox select picks the
// run's recipe: none (compute only), 4d host I/O (stdout read back onto the page), §22 guest-JIT, or
// a §14 root `Instantiator` (sandboxed children on their own Workers). The page services no
// authority either way — all of it is Rust-side, in shared linear memory.

import { loadEngine, makeRunner, readParStdout } from './par.js';
import { openJitReactor } from './wasmjit-reactor.js';
import { runJitModule } from './wasmjit-module.js';
import { createDapClient } from './dap.js';
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

  'Debugger (SVM — breakpoints, step, variables)': {
    debug: true,
    bp: 7, // a breakpoint pre-placed on line 8 (0-based 7), the loop body
    mode: 'plain',
    desc: 'The §DEBUGGING Debug Adapter Protocol debugger, running on the bytecode engine right here ' +
      'in the sandbox — no `debug` section needed. The engine auto-derives a line table and names the ' +
      'SSA values straight from the SVM text, so any program you write here is debuggable. Click the ' +
      'gutter to set/clear breakpoints (one is pre-placed on line 8), then press Debug: it stops at the ' +
      'line, highlights it, and shows the in-scope values (i / acc) in the Variables pane. Step and ' +
      'Continue walk the loop — watch acc accumulate. Run executes it normally (→ 15). Same DAP server ' +
      'VS Code speaks, driven over the wasm FFI.',
    src: `; Sum i = n..1 into acc. Click the gutter to set a breakpoint, then press Debug.
func () -> (i64) {
block0():
  n = i64.const 5
  acc0 = i64.const 0
  br block1(n, acc0)
block1(i: i64, acc: i64):
  sum = i64.add acc i
  one = i64.const 1
  next = i64.sub i one
  br_if next block1(next, sum) block2(sum)
block2(r: i64):
  return r
}
`,
  },

  // ---- on-ramp modules: real C/C++ guests, compiled through clang → svm-llvm and run as a
  //      pre-built .svmb via `svm_run_onramp` (no in-browser parse). Built by
  //      `build-onramp-assets.mjs` at `--host-page 65536` (the wasm page). ------------------------
  'hello (C → SVM)': {
    kind: 'module',
    jit: true, // _start is wasm-JIT-emittable (proven byte-identical by browser-jit-module-test)
    url: './assets/hello_c.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/hello.c — a C program compiled with stock clang, translated by the ' +
      'LLVM on-ramp, and run through the powerbox: it write(1, …)s a greeting and exits. The output ' +
      'below is the guest’s real stdout. Toggle "wasm-JIT" to run the whole program (_start) on ' +
      'emitted wasm instead of the interpreter — "Prove interp ≡ JIT" checks the stdout matches.',
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
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
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
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
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
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
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
    jit: true, // _start is wasm-JIT-emittable (proven byte-identical by browser-jit-module-test)
    editable: true,
    lang: 'lua',
    url: './assets/lua_eval.svmb',
    mode: 'io',
    desc: 'Lua 5.4.7 — its core (lexer, parser, GC, bytecode VM) plus the base/string/table/math/' +
      'coroutine/io/os libraries, compiled through the LLVM on-ramp. Edit the Lua on the left and ' +
      'click Run: your code is piped to the guest as stdin, evaluated, and its output appears below. ' +
      'Real Lua, running client-side in the sandbox. Toggle "wasm-JIT" to run the whole interpreter ' +
      'on emitted wasm (near-native — the ~7% cross-tier helpers bounce to the interpreter); ' +
      '"Prove interp ≡ JIT" checks the stdout is byte-identical on both tiers.',
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
    jit: true, // _start is wasm-JIT-emittable (proven byte-identical by browser-jit-module-test)
    editable: true,
    lang: 'sql',
    url: './assets/sqlite_repl.svmb',
    mode: 'io',
    desc: 'The unmodified SQLite 3.50.2 amalgamation (~257k lines of C), compiled through the LLVM ' +
      'on-ramp. Edit the SQL on the left and click Run: it executes against a fresh in-memory ' +
      'database (each Run starts clean) and prints result tables, change counts, and errors below. ' +
      'Real SQLite, running client-side in the sandbox. Toggle "wasm-JIT" to run the whole engine on ' +
      'emitted wasm (near-native); "Prove interp ≡ JIT" checks the stdout is byte-identical on both tiers.',
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
      'ambient authority. It runs as a live **interactive session**: the first Run boots the backend ' +
      '(a few seconds), then each Run feeds your SQL to the *same* backend on its blocking stdin — so ' +
      'queries after the first are sub-second and state persists across them (a table you CREATE stays ' +
      'for the next query), exactly like psql. **Your database also survives a page reload:** after each ' +
      'query the data directory is snapshotted into your browser (IndexedDB), and the next visit boots ' +
      'from that snapshot — Postgres runs its own crash recovery over it. Run `\\reset` to wipe the ' +
      'saved database and start fresh; Stop just closes the live backend (Run reopens it). The two large ' +
      'artifacts (a ~20 MB module + a ~40 MB image) download once.',
    src: `-- Click Run to send this to the live backend. Run again with new SQL — the session persists
-- (the table below stays for later queries), and only the first Run pays the boot.
-- Your data survives a page reload too: reload, then Run to resume. Type \\reset + Run to wipe it.
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
async function fetchModule(url, onProgress) {
  if (moduleCache.has(url)) return moduleCache.get(url);
  // Resolve module URLs relative to this script (not the document), so they work under any base path
  // (origin root locally, `/<repo>/` on GitHub Pages).
  const resolved = new URL(url, import.meta.url);
  const r = await fetch(resolved);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  // Stream the body so the big downloads (SQLite ~6 MB, Postgres ~20 MB module + ~40 MB image) show
  // progress instead of a silent stall. Falls back to a one-shot read when there's no reader (or no
  // caller watching): Content-Length gives the percent, absent ⇒ a running byte count.
  if (onProgress && r.body && r.body.getReader) {
    const total = Number(r.headers.get('content-length')) || 0;
    const reader = r.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    const bytes = new Uint8Array(received);
    let off = 0;
    for (const ch of chunks) { bytes.set(ch, off); off += ch.length; }
    moduleCache.set(url, bytes);
    return bytes;
  }
  const bytes = new Uint8Array(await r.arrayBuffer());
  moduleCache.set(url, bytes);
  return bytes;
}

// A download-progress callback that reports into a card's status line. `label` is the file being
// fetched; `total` of 0 (no Content-Length) shows a running byte count instead of a percentage.
const fmtMB = (n) => (n / (1 << 20)).toFixed(1);
const onFetchProgress = (c, label) => (received, total) => {
  const pct = total ? ` ${Math.floor((received / total) * 100)}%` : '';
  const of = total ? ` (${fmtMB(received)}/${fmtMB(total)} MB)` : ` (${fmtMB(received)} MB)`;
  setState(c, 'running', `downloading ${label}…${pct}${of}`);
};
const baseName = (url) => url.split('/').pop();

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

// Read the captured stdout stash (a stable region, independent of the module buffer — safe to read
// after the module has been deallocated). Shared by the interpreter and wasm-JIT module paths.
const readModuleStdout = () =>
  new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(
    eng.ex.svm_stdout_ptr(), eng.ex.svm_stdout_ptr() + eng.ex.svm_stdout_len()));

// Run a pre-built on-ramp module single-shot on the interpreter: alloc a buffer, copy the module in
// (plus optional stdin), `svm_run_onramp` (the fixed §3e powerbox — stdout/stdin/exit/memory), read
// the captured stdout, free. Returns { rv, status, stdout }. No Workers (these guests are
// single-threaded), so it never touches the par.js shared-window path.
function moduleInterp(bytes, stdinBytes) {
  // Alloc both buffers *before* filling: svm_alloc may grow (detach) the linear memory, so take one
  // fresh view after all allocations and write into it.
  const p = eng.ex.svm_alloc(bytes.length);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) stdinP = eng.ex.svm_alloc(stdinLen);
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (stdinP) view.set(stdinBytes, stdinP);
  const rv = eng.ex.svm_run_onramp(p, bytes.length, stdinP, stdinLen);
  const status = eng.ex.svm_status();
  const stdout = readModuleStdout();
  eng.ex.svm_dealloc(p, bytes.length);
  if (stdinP) eng.ex.svm_dealloc(stdinP, stdinLen);
  return { rv, status, stdout };
}

// A card's Run for an on-ramp module. The "wasm-JIT" toggle (offered on the emittable guests —
// hello_c/Lua/SQLite) emits the whole `_start` and runs it on wasm near-natively, servicing the ~7%
// cross-tier helpers through the interpreter; it falls back to the interpreter if the module isn't
// emittable (runJitModule throws). Both tiers share the fixed powerbox, so the stdout is identical.
async function runModule(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching module…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  let bytes;
  try {
    bytes = await fetchModule(ex.url, onFetchProgress(c, baseName(ex.url)));
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    logTo(c, `fetch failed: ${e.message}`);
    return;
  }
  logTo(c, `fetched ${ex.url}: ${bytes.length}B module`);
  // An editable module reads the editor text as **stdin** (the guest evaluates it — e.g. Lua).
  let stdinBytes = null;
  if (ex.editable) {
    const enc = new TextEncoder().encode(c.editor.getValue());
    if (enc.length > 0) stdinBytes = enc;
  }
  setState(c, 'running', `running…${useJit ? ' [wasm-JIT]' : ''}`);
  const t0 = performance.now();
  let rv = 0, status, tier = 'interpreter', stdout = '';
  if (useJit) {
    try {
      // Emit `_start` and run it on wasm; svm_onramp_jit_run_finish captures stdout/exit into the
      // shared slots (read back via the usual accessors, exactly like the interpreter path).
      status = await runJitModule(eng.ex, eng.memory, bytes, stdinBytes);
      rv = eng.ex.svm_exit_code();
      stdout = readModuleStdout();
      tier = 'wasm-JIT';
    } catch (e) {
      logTo(c, `wasm-JIT module unavailable (${e.message}); falling back to the interpreter`);
      status = undefined;
    }
  }
  if (status === undefined) {
    const r = moduleInterp(bytes, stdinBytes);
    rv = r.rv; status = r.status; stdout = r.stdout;
    // A framebuffer guest (gradient) presents through the interpreter path; the emittable JIT guests
    // above are stdout-only, so only the interpreter path blits a frame.
    presentFrame(c, eng.ex.svm_framebuffer_width(), eng.ex.svm_framebuffer_height());
  }
  const ms = (performance.now() - t0).toFixed(0);
  c.el.stdout.textContent = stdout;
  c.el.result.textContent = `${rv}`;
  // 0 = OK, 5 = clean Exit; anything else is a decode error / trap / unsupported.
  if (status === 0 || status === 5) {
    setState(c, 'done', `done (${tier}) · status ${status} · ${ms}ms`);
    logTo(c, `module run (${tier}) → ${rv} (status ${status}) in ${ms}ms`);
  } else {
    setState(c, 'error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `module run (${tier}) status ${status}`);
  }
}

// Boot PostgreSQL `--single` single-shot on the main engine (the `svm_run_pg` entry): fetch the
// pre-translated+resolved module + the data image, feed the editor's SQL as stdin, mount the image on
// an in-memory `fs` cap, run to a queried backend, read the captured stdout. A fresh boot per Run.
// Read the engine's captured stdout buffer (the `svm_pg_*` delta, or a `svm_run_pg` full capture).
function readEngineStdout() {
  const p = eng.ex.svm_stdout_ptr();
  const l = eng.ex.svm_stdout_len();
  return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + l));
}

// ---- persistent Postgres storage (IndexedDB) -----------------------------------------------------
// The live backend's data dir is an in-memory `mem_fs`; on its own it evaporates when the page unloads.
// After each query we snapshot that fs (`svm_pg_snapshot` → an `svm_fs` data image) and stash the image
// in IndexedDB; the next session boots from the saved image instead of the pristine one — so a table you
// CREATE (and its rows) survive a full page reload, recovered by Postgres' own startup recovery over the
// snapshot. Keyed per module URL so distinct builds don't collide. All best-effort: any storage failure
// just logs and the session keeps running in memory.
const PG_DB = 'svm-pg';
const PG_STORE = 'sessions';
const pgKey = (c) => c.ex.url;
function pgIdb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(PG_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(PG_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function pgLoad(key) {
  try {
    const db = await pgIdb();
    return await new Promise((resolve, reject) => {
      const r = db.transaction(PG_STORE, 'readonly').objectStore(PG_STORE).get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => reject(r.error);
    });
  } catch {
    return null; // no IndexedDB (private mode, etc.) ⇒ fall back to the pristine image
  }
}
async function pgSave(key, bytes) {
  const db = await pgIdb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PG_STORE, 'readwrite');
    tx.objectStore(PG_STORE).put(bytes, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function pgClear(key) {
  try {
    const db = await pgIdb();
    await new Promise((resolve) => {
      const tx = db.transaction(PG_STORE, 'readwrite');
      tx.objectStore(PG_STORE).delete(key);
      tx.oncomplete = resolve;
      tx.onerror = resolve;
    });
  } catch {
    /* nothing to clear */
  }
}
// Snapshot the live session's data dir and persist it, **coalescing** concurrent saves: at most one IDB
// write is in flight; a query that lands mid-write just marks the card dirty and re-saves once it drains
// (so a burst of queries collapses to one trailing write of the latest state). The snapshot bytes are
// copied straight out of wasm memory before the async write, so a later `memory.grow` can't detach them.
function persistPg(c) {
  if (!c.pgSession) return;
  let bytes;
  try {
    if (eng.ex.svm_pg_snapshot() !== 0) return;
    const p = eng.ex.svm_pg_snapshot_ptr();
    const l = eng.ex.svm_pg_snapshot_len();
    if (!p || !l) return;
    bytes = new Uint8Array(eng.memory.buffer, p, l).slice(); // detach from the wasm buffer
  } catch (e) {
    logTo(c, `snapshot failed: ${e.message}`);
    return;
  }
  if (c.pgSaving) {
    c.pgDirty = true;
    return;
  }
  c.pgSaving = true;
  pgSave(pgKey(c), bytes)
    .catch((e) => logTo(c, `session save failed: ${e.message}`))
    .finally(() => {
      c.pgSaving = false;
      if (c.pgDirty) {
        c.pgDirty = false;
        persistPg(c);
      }
    });
}

// PostgreSQL as a **live interactive session** (the `svm_pg_open`/`_query`/`_close` path): the first Run
// boots one `postgres --single` backend to the `backend>` prompt (a few seconds) and leaves it suspended
// on its blocking stdin; every Run after feeds the editor's SQL to that *same* backend and resumes it to
// the next prompt — so queries are sub-second and state persists across them. The output pane is a
// running transcript; Stop closes the session (`stopDemo`), and the next Run boots fresh.
async function runPg(c) {
  const ex = c.ex;
  c.el.result.textContent = '';
  c.el.canvas.hidden = true;
  // `\reset` (a bare meta-command in the editor): drop the saved database and close any live session, so
  // the next Run boots from the pristine image. The way back to a clean slate once a session persists.
  if (c.editor.getValue().trim() === '\\reset') {
    await pgClear(pgKey(c));
    if (c.pgSession) {
      eng.ex.svm_pg_close();
      c.pgSession = false;
      c.el.stop.disabled = true;
    }
    setState(c, 'done', 'saved database cleared — the next Run boots a fresh cluster');
    logTo(c, 'reset: cleared the saved session');
    c.el.run.disabled = broken;
    return;
  }
  // 1) Open the session on the first Run. Prefer a **saved** snapshot (a prior session's data dir,
  //    persisted in IndexedDB) over the pristine image, so a page reload resumes where you left off.
  if (!c.pgSession) {
    setState(c, 'running', 'fetching module + image…');
    c.el.stdout.textContent = '';
    let modBytes, imgBytes, restored = false;
    try {
      // Sequential (not Promise.all) so the two large downloads report progress one at a time.
      modBytes = await fetchModule(ex.url, onFetchProgress(c, baseName(ex.url)));
      const saved = await pgLoad(pgKey(c));
      if (saved) {
        imgBytes = saved instanceof Uint8Array ? saved : new Uint8Array(saved);
        restored = true;
      } else {
        imgBytes = await fetchModule(ex.image, onFetchProgress(c, baseName(ex.image)));
      }
    } catch (e) {
      setState(c, 'error', `${e.message} — run \`node build-pg-assets.mjs\` to stage the Postgres artifacts`);
      logTo(c, `fetch failed: ${e.message}`);
      return;
    }
    setState(c, 'running', restored
      ? 'restoring your saved database… (first Run only — a few seconds)'
      : 'booting postgres… (first Run only — a few seconds; later queries are instant)');
    c.el.run.disabled = true;
    await new Promise((r) => setTimeout(r, 30)); // let the status paint before the synchronous boot
    try {
      const modP = eng.ex.svm_alloc(modBytes.length);
      const imgP = eng.ex.svm_alloc(imgBytes.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(modBytes, modP);
      view.set(imgBytes, imgP);
      const t0 = performance.now();
      const rc = eng.ex.svm_pg_open(modP, modBytes.length, imgP, imgBytes.length);
      const ms = (performance.now() - t0).toFixed(0);
      eng.ex.svm_dealloc(modP, modBytes.length);
      eng.ex.svm_dealloc(imgP, imgBytes.length);
      c.el.stdout.textContent += readEngineStdout(); // the banner + first prompt
      if (rc !== 0) {
        // A saved image that won't boot is likely corrupt — drop it so the next Run starts clean.
        if (restored) {
          await pgClear(pgKey(c));
          logTo(c, 'saved session failed to boot — cleared it; Run again for a fresh database');
        }
        setState(c, 'error', `boot failed: status ${eng.ex.svm_status()} (1=decode 3=trap 6=verify)`);
        c.el.run.disabled = broken;
        return;
      }
      c.pgSession = true;
      c.el.stop.disabled = false;
      logTo(c, restored ? `svm_pg_open: restored saved session in ${ms}ms` : `svm_pg_open: backend booted in ${ms}ms`);
    } catch (e) {
      setState(c, 'error', `boot error: ${e.message}`);
      c.el.run.disabled = broken;
      return;
    }
  }
  // 2) Send the editor's SQL to the live backend as one query.
  const sql = c.editor.getValue();
  if (!sql.trim()) {
    setState(c, 'done', 'session live — type SQL and Run (state persists across reloads · `\\reset` clears it)');
    c.el.run.disabled = broken;
    return;
  }
  try {
    const text = sql.endsWith('\n') ? sql : sql + '\n';
    const b = new TextEncoder().encode(text);
    const p = eng.ex.svm_alloc(b.length);
    new Uint8Array(eng.memory.buffer).set(b, p);
    const t0 = performance.now();
    const rc = eng.ex.svm_pg_query(p, b.length);
    const ms = (performance.now() - t0).toFixed(0);
    eng.ex.svm_dealloc(p, b.length);
    // Append this query's output delta to the running transcript.
    c.el.stdout.textContent += readEngineStdout();
    c.el.stdout.scrollTop = c.el.stdout.scrollHeight;
    const status = eng.ex.svm_status();
    if (rc === 0) {
      setState(c, 'done', `query ran in ${ms}ms · session live · saved (reload to resume)`);
      logTo(c, `svm_pg_query in ${ms}ms`);
      persistPg(c); // snapshot the (possibly mutated) data dir so it survives a reload
    } else if (status === 5) {
      // The backend exited (e.g. the SQL issued a shutdown) — the session is over. Persist its final
      // state first, so even a clean shutdown is resumable.
      persistPg(c);
      c.pgSession = false;
      c.el.stop.disabled = true;
      setState(c, 'done', 'backend exited — Run reopens your saved database');
    } else {
      setState(c, 'error', `query failed: status ${status}`);
      logTo(c, `svm_pg_query status ${status}`);
    }
  } catch (e) {
    setState(c, 'error', `query error: ${e.message}`);
    logTo(c, `query error: ${e.message}`);
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

// Feed one key event to the running reactor guest through the `keyboard` capability (JS keyCode +
// pressed flag). Shared by the physical-keyboard handler and the on-screen touch dpad; a no-op when no
// reactor loop is running, and routed to whichever tier (interpreter / wasm-JIT) is live.
function sendReactorKey(keyCode, pressed) {
  if (reactorRAF === null) return;
  if (jitReactor) eng.ex.svm_onramp_jit_key(keyCode, pressed);
  else eng.ex.svm_onramp_key(keyCode, pressed);
}

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
    bytes = await fetchModule(ex.url, onFetchProgress(c, baseName(ex.url)));
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
      wad = await fetchModule(ex.wad, onFetchProgress(c, baseName(ex.wad)));
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

// ---- "prove it": the interpreter ≡ wasm-JIT differential, in the page --------------------------------
// The project's core claim is "verified ⇒ the same result on both tiers." For a JIT-emittable reactor,
// prove it live: open the SAME guest on the interpreter and on the wasm-JIT tier, run N frames on each,
// and compare the presented framebuffer byte-for-byte. (This is exactly what browser-jit-reactor-test.mjs
// asserts in CI — surfaced here as a button.)

// FNV-1a over the presented framebuffer, tagged with its dimensions (so a size divergence also shows).
// Copied out of shared memory — a plain view would be a live alias.
function hashFB() {
  const w = eng.ex.svm_framebuffer_width();
  const h = eng.ex.svm_framebuffer_height();
  const p = Number(eng.ex.svm_framebuffer_ptr());
  const px = new Uint8Array(eng.memory.buffer).slice(p, p + w * h * 4);
  let hsh = 0x811c9dc5;
  for (let i = 0; i < px.length; i++) { hsh ^= px[i]; hsh = Math.imul(hsh, 0x01000193) >>> 0; }
  return `${w}x${h}:${(hsh >>> 0).toString(16)}`;
}

// Open the interpreter reactor, run up to `n` frames, hashing each presented frame; close. Synchronous.
function framesInterp(bytes, wad, n) {
  let opened;
  if (wad) {
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
  if (opened !== 0) throw new Error(`interpreter open failed: status ${eng.ex.svm_status()}`);
  const hs = [];
  for (let i = 0; i < n; i++) { if (eng.ex.svm_onramp_frame() !== 0) break; hs.push(hashFB()); }
  eng.ex.svm_onramp_close();
  return hs;
}

// Open the wasm-JIT reactor (throws if the tick isn't emittable), run up to `n` frames, hashing each.
async function framesJit(bytes, wad, n) {
  const r = await openJitReactor(eng.ex, eng.memory, bytes, 'doom1.wad', wad);
  const hs = [];
  for (let i = 0; i < n; i++) { if (r.frame() !== 0) break; hs.push(hashFB()); }
  r.close();
  return hs;
}

async function proveParity(c) {
  if (broken) return;
  stopReactor(); // a parity run supersedes any running reactor loop
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let bytes, wad = null;
  try {
    bytes = await fetchModule(ex.url);
    if (ex.wad) wad = await fetchModule(ex.wad);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  const N = 30;
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter frames block the thread.
    await new Promise((r) => setTimeout(r, 30));
    const interpH = framesInterp(bytes, wad, N);
    const jitH = await framesJit(bytes, wad, N);
    const n = Math.min(interpH.length, jitH.length);
    let mismatch = -1;
    for (let i = 0; i < n; i++) if (interpH[i] !== jitH[i]) { mismatch = i; break; }
    const identical = mismatch === -1 && interpH.length === jitH.length && n > 0;
    if (identical) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical framebuffer across ${n} frames`);
      logTo(c, `parity: ${n} frames byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged at frame ${mismatch} (interp ${interpH.length} / jit ${jitH.length} frames)`);
      logTo(c, `parity: diverged at frame ${mismatch}`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// The module twin of proveParity: run the SAME on-ramp module (with the same editor stdin) on the
// interpreter and on the wasm-JIT tier and assert the captured **stdout** is byte-identical — the
// "verified ⇒ same result on both tiers" claim for a run-to-completion guest (framebuffer demos prove
// it per-frame instead). This is exactly what browser-jit-module-test.mjs asserts in CI.
async function proveModuleParity(c) {
  if (broken) return;
  stopReactor();
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  let stdinBytes = null;
  if (ex.editable) {
    const enc = new TextEncoder().encode(c.editor.getValue());
    if (enc.length > 0) stdinBytes = enc;
  }
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter run blocks the thread.
    await new Promise((r) => setTimeout(r, 30));
    const interp = moduleInterp(bytes, stdinBytes);
    let jitOut;
    try {
      await runJitModule(eng.ex, eng.memory, bytes, stdinBytes);
      jitOut = readModuleStdout();
    } catch (e) {
      setState(c, 'error', `✗ wasm-JIT unavailable: ${e.message}`);
      logTo(c, `parity: JIT emit failed: ${e.message}`);
      return;
    }
    if (interp.stdout === jitOut) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical stdout (${jitOut.length}B)`);
      logTo(c, `parity: ${jitOut.length}B stdout byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged (interp ${interp.stdout.length}B / jit ${jitOut.length}B stdout)`);
      logTo(c, `parity: stdout diverged (interp ${interp.stdout.length}B vs jit ${jitOut.length}B)`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// ---- the DAP debugger (DEBUGGING.md): breakpoints · stepping · variables, on the bytecode engine --
// One debug session at a time. The panel drives the `svm-dap` server (bytecode backend) through the
// `dap.js` client over the wasm FFI: launch the SVM text, run to a breakpoint, highlight the stopped
// source line, and show the paused frame's named locals; Step/Continue advance it. This is the same
// DAP an editor speaks — the playground is just another DAP frontend.
let dapClient = null; // the active DAP client while a session runs (else null)
let dapCard = null; // the card the session belongs to

// The DAP source a breakpoint request targets — the program's own `debug.file 0 "…"` if it declares
// one, else the name the engine's auto debug info uses (svm-text's AUTO_DEBUG_FILE = "source.svm"), so
// breakpoints bind for a hand-written program with no explicit `debug` section.
function dapSourceName(src) {
  const m = /debug\.file\s+0\s+"([^"]+)"/.exec(src);
  return m ? m[1] : 'source.svm';
}

// Push the card's current breakpoint lines (editor 0-based → DAP 1-based) to the server.
function dapSyncBreakpoints(c) {
  const breakpoints = c.editor.breakpointLines().map((l) => ({ line: l + 1 }));
  dapClient.send('setBreakpoints', {
    source: { path: dapSourceName(c.editor.getValue()) },
    breakpoints,
  });
}

// Render the paused frame + its named locals into the card's Variables pane, and highlight the source
// line (frame.line is 1-based; 0 ⇒ an unmapped op, so no highlight).
function dapShowStop(c) {
  const frame = dapClient.send('stackTrace', { threadId: 1 }).response.body.stackFrames[0];
  if (!frame) return;
  if (frame.line > 0) c.editor.setStopLine(frame.line - 1);
  const scope = dapClient.send('scopes', { frameId: frame.id }).response.body.scopes[0];
  const vars = dapClient.send('variables', { variablesReference: scope.variablesReference })
    .response.body.variables;
  const rows = vars
    .map((v) => `<div><span class="bpname">${v.name}</span> = ${v.value}${v.type ? ` <em>${v.type}</em>` : ''}</div>`)
    .join('');
  c.el.dbgVars.innerHTML = `<div>${frame.name} · line ${frame.line}</div>${rows}`;
}

// Handle a resume reply: a `terminated` event ends the session; a `stopped` event pauses (show it).
function dapHandle(c, reply) {
  if (reply.events.some((e) => e.event === 'terminated')) {
    endDebug(c, 'program finished');
    return;
  }
  const stopped = reply.events.find((e) => e.event === 'stopped');
  if (stopped) {
    dapShowStop(c);
    setState(c, 'running', `paused (${stopped.body.reason}) — Step / Continue, Stop to end`);
  }
}

// Start a debug session on the card's current SVM text (on the bytecode engine).
function startDebug(c) {
  if (broken) return;
  stopReactor();
  if (dapCard) endDebug(dapCard, null); // supersede any running session
  const src = c.editor.getValue();
  dapClient = createDapClient(eng.ex, eng.memory);
  dapCard = c;
  c.el.result.textContent = '';
  c.el.dbgVars.innerHTML = '';
  dapClient.send('initialize', {});
  const launch = dapClient.send('launch', { programText: src, function: 0, args: [], engine: 'bytecode' });
  if (!launch.response.success) {
    endDebug(c, null);
    setState(c, 'error', 'debug launch failed — does the program parse and run single-threaded?');
    return;
  }
  dapSyncBreakpoints(c);
  c.editor.setReadOnly(true);
  c.el.dbg.classList.add('active');
  c.el.run.disabled = true;
  logTo(c, 'debug session started (bytecode engine) — running to the first breakpoint');
  dapHandle(c, dapClient.send('configurationDone', {}));
}

// A step verb (continue / next / stepIn / stepOut) on the active session.
function debugStep(c, command) {
  if (dapCard !== c || !dapClient) return;
  c.editor.clearStopLine();
  dapHandle(c, dapClient.send(command, {}));
}

// End the session: disconnect, clear the stop highlight, restore the editor.
function endDebug(c, message) {
  if (!dapClient || dapCard !== c) {
    if (message && c) setState(c, 'done', message);
    return;
  }
  dapClient.send('disconnect', {});
  dapClient = null;
  dapCard = null;
  c.editor.clearStopLine();
  c.editor.setReadOnly(false);
  c.el.dbg.classList.remove('active');
  c.el.run.disabled = broken;
  if (message) setState(c, 'done', message);
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
      for (const card of cards) { card.el.run.disabled = true; if (card.el.prove) card.el.prove.disabled = true; }
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
  if (dapCard) endDebug(dapCard, null); // a fresh Run supersedes any debug session
  stopReactor(); // a fresh Run supersedes any running reactor loop
  const ex = c.ex;
  if (ex.kind === 'reactor') return runReactor(c);
  if (ex.kind === 'pg') return runPg(c);
  if (ex.kind === 'module') return runModule(c);
  return runText(c);
}

// A card's Stop: close a live Postgres session, end a running reactor, or abort a threaded text run.
function stopDemo(c) {
  if (c.pgSession) {
    eng.ex.svm_pg_close();
    c.pgSession = false;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    // Stop closes the *live* backend but keeps the saved snapshot, so Run reopens the same database.
    setState(c, 'stopped', 'session closed — Run reopens your saved database (`\\reset` for a clean one)');
    return;
  }
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

// ---- editor state: persistence + shareable permalinks --------------------------------------------
// Each editable card's source is persisted under its slug so edits survive a reload; "Reset" restores
// the demo's default and drops the saved copy. localStorage is best-effort — a private-mode/quota
// error must never break the page, so every access is guarded.
const STORE_PREFIX = 'svm-play:src:';
const loadSaved = (id) => { try { return localStorage.getItem(STORE_PREFIX + id); } catch { return null; } };
const saveSrc = (id, value, dflt) => {
  try {
    if (value === dflt) localStorage.removeItem(STORE_PREFIX + id); // back to default ⇒ forget it
    else localStorage.setItem(STORE_PREFIX + id, value);
  } catch { /* private mode / quota — persistence is best-effort */ }
};
const clearSaved = (id) => { try { localStorage.removeItem(STORE_PREFIX + id); } catch { /* ignore */ } };

// URL-safe base64 of a UTF-8 string (for the `#src=` permalink payload). Byte-by-byte, not a spread,
// so a large source can't blow the call stack.
function toB64Url(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function fromB64Url(b64) {
  const bin = atob(b64.replace(/-/g, '+').replace(/_/g, '/'));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

// Build a link that reproduces a card's current editor contents: `…/play.html#demo=<slug>&src=<b64url>`.
const buildShareURL = (id, src) =>
  `${location.origin}${location.pathname}#${new URLSearchParams({ demo: id, src: toB64Url(src) }).toString()}`;

// Copy a permalink for this card to the clipboard (falling back to the address bar if the clipboard is
// blocked — e.g. an insecure context or a denied permission).
async function shareCard(c) {
  const url = buildShareURL(c.id, c.editor.getValue());
  try {
    await navigator.clipboard.writeText(url);
    setState(c, 'done', 'link copied to clipboard');
  } catch {
    location.hash = url.slice(url.indexOf('#') + 1);
    setState(c, 'done', 'link in the address bar — copy it');
  }
  logTo(c, url);
}

// Apply a shared editor state from the URL hash (`#demo=<slug>&src=<b64url>`) once at startup: seed the
// target card's editor and scroll it into view. A bare `#demo=<slug>` (no src) just scrolls to it.
function applyHash() {
  if (!location.hash) return;
  let params;
  try { params = new URLSearchParams(location.hash.slice(1)); } catch { return; }
  const id = params.get('demo');
  if (!id) return;
  const c = cards.find((card) => card.id === id);
  if (!c) return;
  const src = params.get('src');
  if (src != null && c.editor) {
    try { c.editor.setValue(fromB64Url(src)); } catch { /* malformed payload — leave the default */ }
  }
  c.el.section.scrollIntoView({ block: 'start' });
}

const POWERBOX_MODES = [
  ['plain', 'none (compute only)'],
  ['io', 'host I/O (stdout)'],
  ['jit', 'guest JIT (§22)'],
  ['inst', 'instantiator (§14)'],
];

function buildCard(name, ex) {
  const id = slug(name);
  const section = el('section', 'demo');
  section.id = 'demo-' + id;
  section.dataset.demo = name; // stable hook for tests
  section.append(el('h2', 'demo-title', name));
  section.append(el('p', 'desc', ex.desc || ''));

  // SVM text (no `kind`) and editable modules (Lua/SQL/Postgres) get an editor; a fixed C guest or a
  // reactor gets a lightweight read-only note (its "source" is a pre-built binary).
  const editable = !ex.kind || !!ex.editable;
  const dflt = ex.src || '';
  let editor = null;
  if (editable) {
    const ta = el('textarea');
    ta.value = dflt;
    const wrap = el('div', 'editor');
    wrap.appendChild(ta);
    section.appendChild(wrap);
    editor = createEditor(ta, ex.lang || 'svm');
    // Restore a previously edited source, then persist every edit under this card's slug.
    const saved = loadSaved(id);
    if (saved != null && saved !== dflt) editor.setValue(saved);
    editor.onChange(() => saveSrc(id, editor.getValue(), dflt));
    // Debug-capable cards: a gutter click toggles a breakpoint (live-synced to an active session), and
    // the demo may pre-place one. (`c` is referenced by the click closure, which only fires post-build.)
    if (ex.debug) {
      if (ex.bp != null) editor.toggleBreakpoint(ex.bp);
      editor.onGutterClick((line) => {
        editor.toggleBreakpoint(line);
        if (dapCard === c) dapSyncBreakpoints(c);
      });
    }
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
  // Editable cards get Reset (restore the demo's default source) + Share (copy a permalink of the
  // current editor contents). A fixed/reactor card has no editable source, so neither applies.
  let resetBtn = null, shareBtn = null;
  if (editable) {
    resetBtn = el('button', 'reset', 'Reset');
    resetBtn.title = 'Restore this demo’s original source';
    shareBtn = el('button', 'share', 'Share');
    shareBtn.title = 'Copy a link that reproduces the current editor contents';
    controls.append(resetBtn, shareBtn);
  }
  // A debug-capable card gets a Debug button (starts a DAP session on the bytecode engine).
  let debugBtn = null;
  if (ex.debug) {
    debugBtn = el('button', 'debug', 'Debug');
    debugBtn.title = 'Debug this SVM program on the bytecode engine — breakpoints, stepping, variables';
    debugBtn.disabled = true;
    controls.appendChild(debugBtn);
  }
  let jit = null;
  let proveBtn = null;
  if (ex.jit) {
    // A reactor emits its per-frame tick(); a module emits the whole _start. The parity check compares
    // the framebuffer (reactor, per frame) or the stdout (module, run-to-completion) accordingly.
    const isModule = ex.kind === 'module';
    const l = el('label', 'jit-label');
    l.title = isModule
      ? 'Run the whole guest (_start) on emitted wasm (wasm-JIT tier) instead of the interpreter'
      : 'Run the reactor’s tick() on emitted wasm (wasm-JIT tier) instead of the interpreter';
    jit = el('input');
    jit.type = 'checkbox';
    jit.checked = true;
    l.append(jit, ' wasm-JIT');
    controls.appendChild(l);
    // "Prove it": run the guest on both tiers and assert the result is byte-identical.
    proveBtn = el('button', 'prove', 'Prove interp ≡ JIT');
    proveBtn.title = isModule
      ? 'Run the guest on the interpreter and the wasm-JIT tier and check stdout is byte-identical'
      : 'Run 30 frames on the interpreter and the wasm-JIT tier and check the framebuffer is byte-identical';
    proveBtn.disabled = true;
    controls.appendChild(proveBtn);
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

  // On-screen dpad for the interactive reactors: arrows steer, plus the action keys Doom's menus/play
  // use. Only rendered for reactor cards; CSS shows it on touch / narrow screens. Each button dispatches
  // the same keyboard-cap event as the physical key (pressed on pointerdown, released on up/leave).
  if (ex.kind === 'reactor') {
    const pad = el('div', 'dpad');
    // [label, JS keyCode] — arrows (37/38/40/39), fire (Ctrl 17), use (Space 32), enter (13), esc (27).
    for (const [label, code] of [['←', 37], ['↑', 38], ['↓', 40], ['→', 39], ['fire', 17], ['use', 32], ['↵', 13], ['esc', 27]]) {
      const b = el('button', 'dkey', label);
      b.type = 'button';
      b.dataset.key = String(code);
      const press = (down) => (ev) => { ev.preventDefault(); sendReactorKey(code, down ? 1 : 0); };
      b.addEventListener('pointerdown', press(true));
      b.addEventListener('pointerup', press(false));
      b.addEventListener('pointerleave', press(false));
      b.addEventListener('pointercancel', press(false));
      pad.appendChild(b);
    }
    section.appendChild(pad);
  }

  // Debugger panel (DAP over the bytecode engine): step controls + a live Variables pane. Hidden until
  // a session pauses (`.dbg.active`). Only built for debug-capable cards.
  let dbg = null, dbgVars = null;
  if (ex.debug) {
    dbg = el('div', 'dbg');
    const dc = el('div', 'dbg-controls');
    const mk = (label, title, cmd) => {
      const b = el('button', null, label);
      b.title = title;
      b.dataset.cmd = cmd || 'stop';
      b.addEventListener('click', () => (cmd ? debugStep(c, cmd) : endDebug(c, 'debug session ended')));
      return b;
    };
    dc.append(
      mk('▶ Continue', 'Run to the next breakpoint', 'continue'),
      mk('⤼ Step Over', 'Step over the next source line', 'next'),
      mk('↳ Step In', 'Step into a call', 'stepIn'),
      mk('↰ Step Out', 'Run to the caller', 'stepOut'),
      mk('■ Stop', 'End the debug session', null),
    );
    dbgVars = el('pre', 'dbg-vars');
    dbg.append(dc, dbgVars);
    section.appendChild(dbg);
  }

  const c = {
    name, ex, editor, id,
    el: { section, state, result, stdout, log: logEl, canvas, gpucanvas, run: runBtn, stop: stopBtn, mode: modeSel, jit, prove: proveBtn, reset: resetBtn, share: shareBtn, debug: debugBtn, dbg, dbgVars },
  };
  runBtn.addEventListener('click', () => runDemo(c));
  if (debugBtn) debugBtn.addEventListener('click', () => startDebug(c));
  stopBtn.addEventListener('click', () => stopDemo(c));
  if (proveBtn) proveBtn.addEventListener('click', () => (c.ex.kind === 'module' ? proveModuleParity : proveParity)(c));
  if (resetBtn) resetBtn.addEventListener('click', () => {
    editor.setValue(dflt);
    clearSaved(id);
    editor.clearError();
    setState(c, 'ready', 'reset to the original source');
  });
  if (shareBtn) shareBtn.addEventListener('click', () => shareCard(c));
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

// Theme picker: the head script already resolved the initial `data-theme` from the stored preference;
// here we seed the sidebar select and keep it live — persisting the choice and re-resolving `auto`
// against the OS as it changes.
function setupTheme() {
  const sel = $('theme');
  let stored = 'auto';
  try { stored = localStorage.getItem('svm-play:theme') || 'auto'; } catch { /* private mode */ }
  sel.value = stored;
  const mq = matchMedia('(prefers-color-scheme: dark)');
  const apply = (pref) => {
    const dark = pref === 'dark' || (pref === 'auto' && mq.matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  };
  sel.addEventListener('change', () => {
    try { localStorage.setItem('svm-play:theme', sel.value); } catch { /* best-effort */ }
    apply(sel.value);
  });
  mq.addEventListener('change', () => { if (sel.value === 'auto') apply('auto'); }); // follow the OS live
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
  applyHash();  // seed a card's editor from a shared #demo=…&src=… permalink, if present
  setupTheme();
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
    sendReactorKey(e.keyCode, pressed);
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
  for (const c of cards) {
    c.el.run.disabled = false;
    if (c.el.prove) c.el.prove.disabled = false;
    if (c.el.debug) c.el.debug.disabled = false;
  }
  setEngineState('ready', 'engine ready');
}

main().catch((e) => setEngineState('error', `fatal: ${e.message}`));
