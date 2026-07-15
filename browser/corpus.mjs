// Differential check: the wasm exports vs the native bytecode engine, over the corpus emitted by
// `gencorpus`. Usage: node corpus.mjs <module.wasm>  (run gencorpus first).
//
// I/O crosses the boundary through the alloc ABI: the host `svm_alloc`s a buffer, writes the module
// (and stdin), passes the `(ptr, len)` in, reads any captured streams, then `svm_dealloc`s — no
// fixed scratch buffer, so module/stream sizes are unbounded (see the megabyte echo at the end).
import { readFileSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const corpus = JSON.parse(readFileSync('corpus.json', 'utf8'));

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const ex = (await WebAssembly.instantiate(mod, engineImports())).exports;
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x)); // usize ABI value
const mem = () => new Uint8Array(ex.memory.buffer); // re-fetch (memory may grow under us)
const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
const fromHex = (s) => Uint8Array.from(s.match(/../g) ?? [], (h) => parseInt(h, 16));
console.log(`module: ${wasmPath} (${is64 ? 'wasm64' : 'wasm32'})  imports:`,
  WebAssembly.Module.imports(mod).length);

// Allocate `bytes.length` in linear memory, copy `bytes` in, return a handle (empty → null/0).
const load = (bytes) => {
  if (bytes.length === 0) return { ptr: N(0), len: N(0), free() {} };
  const ptr = ex.svm_alloc(N(bytes.length));
  mem().set(bytes, Number(ptr));
  const len = N(bytes.length);
  return { ptr, len, free: () => ex.svm_dealloc(ptr, len) };
};
const readOut = (ptrFn, lenFn) => {
  const ptr = Number(ptrFn()), len = Number(lenFn());
  return len === 0 ? new Uint8Array(0) : mem().slice(ptr, ptr + len);
};

let total = 0, fail = 0;

// ---- compute / fiber corpora: svm_run / svm_run0 vs native ----------------------------------
// Fibers (§12 cont.*) need no powerbox, so they run on the same plain entries as compute.
const runComputeLike = (list) => {
  for (const { name, file, nargs, cases } of list) {
    let bad = 0;
    for (const { arg, status, value } of cases) {
      const m = load(readFileSync(file)); // re-load each case (the engine may dirty the window)
      const got = nargs === 0 ? ex.svm_run0(m.ptr, m.len) : ex.svm_run(m.ptr, m.len, BigInt(arg));
      const gotStatus = ex.svm_status();
      m.free();
      const okStatus = gotStatus === status;
      const okValue = status !== 0 || BigInt(got) === BigInt(value); // value only meaningful when OK
      total++;
      if (!okStatus || !okValue) {
        fail++; bad++;
        console.log(`  FAIL ${name}(${arg}): native {status:${status},value:${value}} ` +
          `wasm {status:${gotStatus},value:${got}}`);
      }
    }
    console.log(`  ${name}: ${cases.length - bad}/${cases.length} match`);
  }
};
runComputeLike(corpus.compute); // includes the fail-closed `unsup` case (STATUS_UNSUPPORTED)
runComputeLike(corpus.fiber ?? []);
runComputeLike(corpus.float ?? []); // scalar f32/f64 — bit-exact (NaN payloads, rounding)
runComputeLike(corpus.tailcall ?? []);
runComputeLike(corpus.simd ?? []);

// ---- powerbox corpus: svm_run_pb (streams/clock/exit) vs native -----------------------------
for (const c of corpus.powerbox ?? []) {
  const m = load(readFileSync(c.file));
  const stdin = load(fromHex(c.stdin));
  const got = ex.svm_run_pb(m.ptr, m.len, stdin.ptr, stdin.len);
  const gotStatus = ex.svm_status();
  const gotOut = hex(readOut(ex.svm_stdout_ptr, ex.svm_stdout_len));
  const gotErr = hex(readOut(ex.svm_stderr_ptr, ex.svm_stderr_len));
  const gotExit = ex.svm_exit_code();
  m.free(); stdin.free();
  const okStatus = gotStatus === c.status;
  const okValue = c.status !== 0 || BigInt(got) === BigInt(c.value);
  const okExit = c.status !== 5 || gotExit === c.exit;
  const okOut = gotOut === c.stdout, okErr = gotErr === c.stderr;
  total++;
  if (!(okStatus && okValue && okExit && okOut && okErr)) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value},exit:${c.exit},` +
      `stdout:${c.stdout},stderr:${c.stderr}} wasm {status:${gotStatus},value:${got},` +
      `exit:${gotExit},stdout:${gotOut},stderr:${gotErr}}`);
  } else {
    const dec = (h) => JSON.stringify(Buffer.from(fromHex(h)).toString());
    const detail = c.status === 5 ? `exit ${gotExit}`
      : gotOut || gotErr ? `out=${dec(gotOut)}${gotErr ? ` err=${dec(gotErr)}` : ''}`
      : `value ${got}`;
    console.log(`  ${c.name}: match (${detail})`);
  }
}

// ---- capture / gc-roots corpora: svm_run_capture (final memory image) vs native -------------
const runCaptureLike = (list, kind) => {
  for (const c of list) {
    const m = load(readFileSync(c.file));
    const init = load(fromHex(c.init));
    const got = ex.svm_run_capture(m.ptr, m.len, init.ptr, init.len, BigInt(c.arg));
    const gotStatus = ex.svm_status();
    const gotSnap = hex(readOut(ex.svm_snapshot_ptr, ex.svm_snapshot_len));
    m.free(); init.free();
    const okStatus = gotStatus === c.status;
    const okValue = c.status !== 0 || BigInt(got) === BigInt(c.value);
    const okSnap = c.status !== 0 || gotSnap === c.snapshot; // image only meaningful when OK
    total++;
    if (!(okStatus && okValue && okSnap)) {
      fail++;
      console.log(`  FAIL ${c.name}(${c.arg}): native {status:${c.status},value:${c.value},` +
        `snap:${c.snapshot}} wasm {status:${gotStatus},value:${got},snap:${gotSnap}}`);
    } else {
      const detail = kind === 'gc' ? `${got} roots, image ${gotSnap.length / 2}B`
        : `final image ${gotSnap.length / 2}B, value ${got}`;
      console.log(`  ${c.name}(${c.arg}): match (${detail})`);
    }
  }
};
runCaptureLike(corpus.capture ?? [], 'capture');
runCaptureLike(corpus.gcroots ?? [], 'gc');

// ---- reflection corpus: svm_run_reflect (§7 cap.self.* over a fixed 3-cap powerbox) ---------
for (const { name, file, cases } of corpus.reflect ?? []) {
  let bad = 0;
  for (const { arg, status, value } of cases) {
    const m = load(readFileSync(file));
    const got = ex.svm_run_reflect(m.ptr, m.len, BigInt(arg));
    const gotStatus = ex.svm_status();
    m.free();
    const ok = gotStatus === status && (status !== 0 || BigInt(got) === BigInt(value));
    total++;
    if (!ok) { fail++; bad++;
      console.log(`  FAIL ${name}(${arg}): native {status:${status},value:${value}} ` +
        `wasm {status:${gotStatus},value:${got}}`);
    }
  }
  console.log(`  ${name}: ${cases.length - bad}/${cases.length} match`);
}

// ---- nested-child corpus: svm_run_nested (§14 confined child domains) vs native -------------
for (const c of corpus.nested ?? []) {
  const m = load(readFileSync(c.file));
  const got = ex.svm_run_nested(m.ptr, m.len);
  const gotStatus = ex.svm_status();
  m.free();
  const okStatus = gotStatus === c.status;
  const okValue = c.status !== 0 || BigInt(got) === BigInt(c.value);
  total++;
  if (!(okStatus && okValue)) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value}} ` +
      `wasm {status:${gotStatus},value:${got}}`);
  } else {
    const detail = c.status === 3 ? 'child trap propagated' : `value ${got}`;
    console.log(`  ${c.name}: match (${detail})`);
  }
}

// ---- guest-JIT corpus: svm_run_jit (§22 install + cross-module call_indirect) vs native -----
for (const c of corpus.jit ?? []) {
  const m = load(readFileSync(c.file));
  const got = ex.svm_run_jit(m.ptr, m.len);
  const gotStatus = ex.svm_status();
  m.free();
  const ok = gotStatus === c.status && (c.status !== 0 || BigInt(got) === BigInt(c.value));
  total++;
  if (!ok) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value}} ` +
      `wasm {status:${gotStatus},value:${got}}`);
  } else {
    console.log(`  ${c.name}: match (${c.status === 3 ? 'freed slot traps' : `value ${got}`})`);
  }
}

// ---- dynamic-linking corpus: svm_run_dynlink (§22 compile_linked symbol resolution) vs native
for (const c of corpus.dynlink ?? []) {
  const m = load(readFileSync(c.file));
  const got = ex.svm_run_dynlink(m.ptr, m.len, c.link);
  const gotStatus = ex.svm_status();
  m.free();
  const ok = gotStatus === c.status && (c.status !== 0 || BigInt(got) === BigInt(c.value));
  total++;
  if (!ok) {
    fail++;
    console.log(`  FAIL ${c.name}(link=${c.link}): native {status:${c.status},value:${c.value}} ` +
      `wasm {status:${gotStatus},value:${got}}`);
  } else {
    console.log(`  ${c.name}(link=${c.link}): match ` +
      `(${c.link ? `import resolved → ${got}` : 'unresolved → fail-closed'})`);
  }
}

// ---- SharedRegion corpus: svm_run_region (§13 host-backed memory aliasing) vs native --------
for (const c of corpus.region ?? []) {
  const m = load(readFileSync(c.file));
  const got = ex.svm_run_region(m.ptr, m.len);
  const gotStatus = ex.svm_status();
  m.free();
  const ok = gotStatus === c.status && (c.status !== 0 || BigInt(got) === BigInt(c.value));
  total++;
  if (!ok) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value}} ` +
      `wasm {status:${gotStatus},value:${got}}`);
  } else {
    console.log(`  ${c.name}: match (value ${got})`);
  }
}

// ---- durability corpus: svm_run_durable (freeze/thaw, IR-driven) vs native -----------------
for (const c of corpus.durable ?? []) {
  const m = load(readFileSync(c.file));
  const win = load(fromHex(c.init));
  const got = ex.svm_run_durable(m.ptr, m.len, win.ptr, win.len, BigInt(c.clock));
  const gotStatus = ex.svm_status();
  const gotSnap = hex(readOut(ex.svm_snapshot_ptr, ex.svm_snapshot_len));
  m.free(); win.free();
  const okStatus = gotStatus === c.status;
  const okValue = c.status !== 0 || BigInt(got) === BigInt(c.value);
  const okSnap = c.status !== 0 || gotSnap === c.snapshot;
  total++;
  if (!(okStatus && okValue && okSnap)) {
    fail++;
    console.log(`  FAIL ${c.name}: native {status:${c.status},value:${c.value},snap#${c.snapshot.length}} ` +
      `wasm {status:${gotStatus},value:${got},snap#${gotSnap.length}}` +
      (gotSnap !== c.snapshot ? ' (snapshot differs)' : ''));
  } else {
    const detail = c.name === 'dur_freeze' ? 'freeze snapshot identical'
      : c.name === 'dur_thaw' ? `thaw reproduced ${got}` : `value ${got}`;
    console.log(`  ${c.name}: match (${detail})`);
  }
}

// ---- alloc-ABI scale check: echo MEGABYTES of stdin → stdout (past the old 1 MiB cap) --------
{
  const SIZE = 2 << 20; // 2 MiB, double the retired fixed-buffer cap
  const input = new Uint8Array(SIZE);
  for (let i = 0; i < SIZE; i++) input[i] = (i * 2654435761) & 0xff; // cheap pseudo-random pattern
  const m = load(readFileSync('corpus/bigecho.svmbc'));
  const stdin = load(input);
  ex.svm_run_pb(m.ptr, m.len, stdin.ptr, stdin.len);
  const gotStatus = ex.svm_status();
  const out = readOut(ex.svm_stdout_ptr, ex.svm_stdout_len);
  m.free(); stdin.free();
  const ok = gotStatus === 0 && out.length === SIZE && Buffer.compare(out, input) === 0;
  total++; if (!ok) fail++;
  console.log(`  bigecho: ${ok ? 'match' : 'FAIL'} (${(SIZE / (1 << 20)).toFixed(0)} MiB echoed ` +
    `through svm_alloc; status ${gotStatus}, out ${out.length}B)`);
}

console.log(`\n${total - fail}/${total} cases match native  ${fail ? 'FAILED' : 'ALL MATCH'}`);
process.exit(fail ? 1 : 0);
