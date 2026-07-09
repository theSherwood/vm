// The SVM **playground** — the human-facing demo the THREADS/BROWSER work builds toward: type SVM
// text, it parses/verifies/encodes *inside the wasm sandbox* (`svm_parse`), and runs across real Web
// Workers (`par.js`, the same orchestration the validation page uses). The powerbox select picks the
// run's recipe: none (compute only), 4d host I/O (stdout read back onto the page), §22 guest-JIT, or
// a §14 root `Instantiator` (sandboxed children on their own Workers). The page services no
// authority either way — all of it is Rust-side, in shared linear memory.

import { loadEngine, makeRunner, readParStdout } from '/web/par.js';

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
    url: '/web/assets/hello_c.svmb',
    mode: 'io',
    desc: 'crates/svm-run/demos/hello.c — a C program compiled with stock clang, translated by the ' +
      'LLVM on-ramp, and run through the powerbox: it write(1, …)s a greeting and exits. The output ' +
      'below is the guest’s real stdout.',
  },
  'SQLite (Phase A, :memory:)': {
    kind: 'module',
    url: '/web/assets/sqlite_demo.svmb',
    mode: 'io',
    desc: 'The SQLite 3.50.2 amalgamation (~257k lines of C) running a 29-statement breadth script ' +
      'over an in-memory database — DDL, aggregates, GROUP BY, window functions, transactions — its ' +
      'query output printed to stdout, byte-identical to native. Build via build-onramp-assets.mjs.',
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
  const ex = EXAMPLES[name];
  $('mode').value = ex.mode;
  $('desc').textContent = ex.desc;
  if (ex.kind === 'module') {
    // A pre-built on-ramp module: the "source" is binary, not editable SVM text. Show a note.
    $('src').value =
      `// ${name}\n// A pre-built on-ramp module: ${ex.url}\n// Click Run — it executes as a real ` +
      `C/C++ guest via svm_run_onramp,\n// and its stdout appears in the pane on the right.`;
    $('src').readOnly = true;
  } else {
    $('src').value = ex.src;
    $('src').readOnly = false;
  }
}

// Fetched `.svmb` bytes, cached (a 6 MB SQLite module is worth not re-downloading on every Run).
const moduleCache = new Map();
async function fetchModule(url) {
  if (moduleCache.has(url)) return moduleCache.get(url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  moduleCache.set(url, bytes);
  return bytes;
}

// Run a pre-built on-ramp module single-shot on the main engine: alloc a buffer, copy the module in,
// `svm_run_onramp` (the fixed §3e powerbox — stdout/stdin/exit/memory), read the captured stdout.
// No Workers (these guests are single-threaded), so it never touches the par.js shared-window path.
async function runModule(ex) {
  setState('running', 'fetching module…');
  $('result').textContent = '';
  $('stdout').textContent = '';
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState('error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    log(`fetch failed: ${e.message}`);
    return;
  }
  log(`fetched ${ex.url}: ${bytes.length}B module`);
  const u8 = () => new Uint8Array(eng.memory.buffer);
  const p = eng.ex.svm_alloc(bytes.length);
  u8().set(bytes, p); // re-fetch the view: svm_alloc may have grown (detached) the old buffer
  setState('running', 'running…');
  const t0 = performance.now();
  const rv = eng.ex.svm_run_onramp(p, bytes.length, 0, 0);
  const status = eng.ex.svm_status();
  const sp = eng.ex.svm_stdout_ptr();
  const sl = eng.ex.svm_stdout_len();
  const stdout = new TextDecoder().decode(u8().slice(sp, sp + sl));
  eng.ex.svm_dealloc(p, bytes.length);
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

async function doRun() {
  if (broken) return;
  // A pre-built on-ramp module runs single-shot via svm_run_onramp — no in-browser parse, no Workers.
  const selected = EXAMPLES[$('example').value];
  if (selected?.kind === 'module') return runModule(selected);
  // Leave the terminal states synchronously on click, so an observer (the Playwright smoke) that
  // clicks Run and polls for done/error never reads the PREVIOUS run's state.
  setState('running', 'parsing…');
  const src = $('src').value;
  const mode = $('mode').value;
  $('result').textContent = '';
  $('stdout').textContent = '';

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
  $('stop').addEventListener('click', () => aborter?.abort());

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
