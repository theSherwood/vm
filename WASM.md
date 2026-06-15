# wasm coverage & roadmap (`crates/svm-wasm`)

What the **wasm → IR transpiler** handles, what it doesn't, and the prioritized work to widen
coverage. svm-wasm is a *second frontend* (after chibicc): it takes core wasm and reconstructs SSA
from the stack machine, so the §1a benchmark thesis can be measured on the **same bytes** Wasmtime
runs. It is an **untrusted** frontend — everything it emits is re-verified by `svm-verify`, so a gap
here is a *capability* limit, never a safety one.

**Status: feature-complete for *typical clang/rustc -O2 output*** (63 tests across
`transpile.rs`/`imports.rs`/`simd.rs`/`atomics.rs`/`threads.rs`). Real clang programs + two real C
libraries (jsmn, B-Con SHA-256) run **byte-identical to native**; a real `clang -msimd128 -O2` saxpy
and a wasi-threads parallel kernel run on both backends. `bench --threads`: SVM ~1.35× faster than
Wasmtime+wasi-threads on spawn-heavy, parity on compute.

This file tracks the path to handling **all** of wasm (the full spec + finished proposals, incl.
memory64). Fold completed sections into `DESIGN.md` / drop this file once the actionable gaps close
(the repo convention, cf. the former `SCHEDULING.md`/`AUDIT.md`).

---

## Done

- **MVP core**: full i32/i64/f32/f64 numeric (arith, bitwise, shifts, rotates, clz/ctz/popcnt,
  compares, eqz, float math); all conversions (`wrap`/`extend`/`trunc`/`reinterpret`); all
  loads/stores incl. narrow 8/16/32 s/u; locals (get/set/tee); globals (get/set); the full structured
  control set (block/loop/if/else/br/br_if/br_table/return/unreachable/nop/select/drop) **incl.
  multi-value** block types; `call`; `call_indirect` (§3c type-id check); `memory.size`/`memory.grow`.
- **memory64** — the 64-bit address path.
- **Finished proposals**: sign-extension ops; non-trapping float→int (`trunc_sat`); bulk-memory
  `memory.copy`/`memory.fill`; **fixed-128 SIMD** (a pragmatic ~60-op v128 subset, D58); **threads** —
  full-width (i32/i64) `*.atomic.*` + `atomic.fence`, `shared`+imported memory, and the **wasi-threads**
  `wasi:thread/spawn` → native `thread.spawn` (a synthesized shim + unique-tid slot).
- **Host ABI**: function imports → `cap.call` by the numeric `module`=type_id / `name`=op convention.

---

## Gaps (tracked)

Severity key: **🔴 silent** (transpiles but mis-behaves — fix first), **🟠 host-blocker** (blocks real
programs), **🟡 fail-closed feature** (clean `Unsupported`; widen on demand), **⚪ non-goal**,
**ℹ️ semantic note**.

### 🔴 Not fail-closed — fix first

- [x] **Start section (`(start $f)`) — DONE (runs it).** Was silently ignored (the default `_ => {}`
  section arm), so a module's start function never ran. Now the transpiler remaps each **exported**
  function to a synthesized wrapper that calls `start` then the real export, so `start` runs once
  before the chosen entry (data/element segments are already materialized when the run begins);
  internal `call`s reach the export directly and don't re-run it. The start function is validated
  `() -> ()`. `tests/start.rs` (runs-before-export, param/result threading, runs-once/internal-bypass,
  bad-signature rejection). A non-`(start)` module is byte-identical to before.

### 🟠 Host-ABI — blocks real WASI programs

- [ ] **Non-numeric WASI imports.** The import convention requires decimal `module`/`name`
  (`type_id`/`op`); a real program importing `("wasi_snapshot_preview1", "fd_write")` is a clean
  `Unsupported` (`lib.rs` import parse). Real I/O programs need a **WASI→capability shim** binding the
  named imports to capability `(type_id, op)`s. Arguably the biggest "run real programs" blocker.
  (The `wasi:thread/spawn` import is already special-cased; this generalizes named-import binding.)

### 🟡 Fail-closed feature gaps (clean `Unsupported`)

- [x] **Tail calls** (`return_call` / `return_call_indirect`) — **DONE.** Lower to the IR's
  `Terminator::ReturnCall`/`ReturnCallIndirect`, which both backends execute as **true** tail calls
  (interp replaces the frame; JIT emits Cranelift `return_call`/`return_call_indirect`). Direct tail
  call to a defined function (true tail call), indirect via the §3c table dispatch, and a capability
  import degrades to `cap.call` + `return` (correct, not tail-optimized; wasi:thread/spawn rejected).
  `tests/tailcall.rs`: 200k-deep tail recursion (would overflow a non-tail call), table dispatch,
  mutual even/odd recursion.
- [ ] **Passive data/element segments + the rest of bulk memory**: `memory.init`, `data.drop`,
  `elem.drop`, `table.init`, `table.copy`, `table.fill`, `table.grow`, `table.size`. (Only
  `memory.copy`/`memory.fill` done.) clang `-O2` occasionally emits `memory.init` for large data.
- [ ] **Reference types**: `externref`/`funcref` as values, `table.get`/`set`/`grow`/`size`/`fill`,
  `ref.null`/`ref.func`/`ref.is_null`, typed `select (result t)`. Natural SVM fit: `externref` →
  capability-handle (an i32 host-table index), `funcref` → funcref-index (already powers
  `call_indirect`); the table-mutation ops are the fiddly part. Low audience (C/C++/Rust don't emit it).
- [ ] **SIMD remainder** (~175 of the v128 proposal): lane compares, lane shifts, int min/max/abs, sat
  add/sub, narrow/widen, `all_true`/`bitmask`, dot product, etc. Mechanical breadth over the proven
  5-step pattern (IR variant → verifier lane rule → interp ref → JIT Cranelift → transpiler arm); a few
  ops have no single Cranelift instruction (cf. `i8x16.mul` already bailing on the JIT).
- [ ] **Narrow atomics** (`*.atomic.rmw8`/`rmw16`, `load8_u`/`16_u`/`32_u`, narrow store/cmpxchg). SVM
  atomics are 32/64-bit only (the §3b narrow-integer decision). Lower via a **32-bit CAS-loop emulation**
  in the transpiler (read containing word, splice the sub-word, cmpxchg) — *not* adding i8/i16 to the IR
  (widens the escape-TCB). wasi-libc locks use 32-bit futex words, so pthreads works without this;
  user code with `_Atomic char`/`bool` needs it.
- [ ] **Imported globals & imported tables.** Dynamic-linking / PIC (`__memory_base`/`__table_base`/
  `__stack_pointer` imports). Statically-linked output defines its own, so this only bites `-shared`/PIC.
- [ ] **Relaxed SIMD** (a separate proposal: relaxed madd, relaxed swizzle, …; clang `-mrelaxed-simd`).
- [ ] **Multiple memories** and **multiple tables** (`lib.rs` rejects both).
- [ ] **Typed function references** (the function-references proposal) and **extended const
  expressions** (arithmetic in global/data-offset initializers — currently only constants).
- [ ] **Exception handling**: `try`/`catch`/`catch_all`/`throw`/`rethrow`/`delegate` + tags. Involved:
  within a function it lowers to branches, but cross-frame propagation needs an exception channel (a
  generalization of the existing per-call trap-cell check) — a perf tax + calling-convention change,
  and clang/Rust only emit it under opt-in (`-fwasm-exceptions`). Low ROI.
- [ ] **SVM host-ABI**: imports spanning multiple capability interfaces (one handle threaded today);
  `wasi:thread/spawn` *alongside* capability imports (needs the per-thread handle stash).

### ⚪ Non-goal (by design)

- **wasm GC** (struct/array/i31, managed references). Not more opcodes — a *different execution model*
  (managed heap + a tracing collector in the runtime), which contradicts SVM's linear-memory +
  small-TCB confinement thesis. The languages targeting wasm-GC (Java/Kotlin/Dart/Scheme) are outside
  the C/C++/Rust niche SVM serves. Treat as a permanent non-goal, not "hard but someday."

### ℹ️ Semantic divergence (not a missing feature)

- **OOB access masks, doesn't trap.** SVM confines an out-of-bounds linear-memory access by **masking**
  into the power-of-two window (§4), where wasm **traps**. Not a miscompile for well-behaved programs,
  but a program that *relies on* an OOB trap (conformance tests; defensive trap-probing) diverges. So
  "passes the wasm spec test suite" is not automatic. (This is the documented §1a confinement model,
  not a bug.)

---

## Recommended order (for "handle more real programs")

1. **Start section** 🔴 — kill the silent footgun (cheap; fail-closed at minimum, ideally run it).
2. **WASI import name mapping** 🟠 — unlocks real I/O programs; the single biggest real-program blocker.
3. **Tail calls** 🟡 — common LLVM output, likely near-free (IR terminators exist).
4. **Passive segments + `memory.init`/`table.*` bulk ops** 🟡 — moderate, evidence-driven.
5. **Reference types** 🟡 (externref→handle, funcref→index), then the **SIMD remainder** 🟡 (breadth),
   then **narrow-atomic CAS-loop** 🟡.
6. EH, relaxed SIMD, multiple memories/tables, imported globals/tables — on demand. GC stays ⚪.

Code map: the rejection sites are the `unsup(...)` calls in `crates/svm-wasm/src/lib.rs` (section
parse + the `worker_op` operator catch-all `other => unsup("operator {…}")`); tests live in
`crates/svm-wasm/tests/`.
