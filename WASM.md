# wasm coverage & roadmap (`crates/svm-wasm`)

What the **wasm → IR transpiler** handles, what it doesn't, and the prioritized work to widen
coverage. svm-wasm is a *second frontend* (after chibicc): it takes core wasm and reconstructs SSA
from the stack machine, so the §1a benchmark thesis can be measured on the **same bytes** Wasmtime
runs. It is an **untrusted** frontend — everything it emits is re-verified by `svm-verify`, so a gap
here is a *capability* limit, never a safety one.

**Status: feature-complete for *typical clang/rustc -O2 output*** (79 tests across
`transpile.rs`/`imports.rs`/`simd.rs`/`atomics.rs`/`threads.rs`/`start.rs`/`tailcall.rs`/`bulk.rs`).
Real clang programs + two real C
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
  `memory.copy`/`memory.fill`; **fixed-128 SIMD** (a pragmatic v128 subset, D58 — arith/bitwise/shuffle
  + the **integer lane compares** `iNxM.{eq,ne,lt,gt,le,ge}` s/u → mask, `VIntCmp`); **threads** —
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

### 🟠 Host-ABI — named import binding

- [x] **Named import binding — DONE (single-handle).** A non-numeric import (e.g. a real WASI
  `("wasi_snapshot_preview1", "fd_write")`) now lowers to a §7 `Inst::CallImport "<module>.<name>"`
  (declared in `Module.imports`); the embedder binds the name to a concrete `(type_id, op)` at load via
  `svm_ir::resolve_imports`. The numeric `module`=type_id / `name`=op convention still lowers to an
  inline `cap.call`. svm-wasm stays pure mechanism — it never interprets host semantics. The
  **`svm-wasi`** crate is the worked example: a minimal preview1 shim (`fd_write`/`proc_exit`) as an
  embedder `HostFn` capability (`svm_interp::iface::HOST_FN`, registered with `Host::grant_host_fn` —
  WASI semantics live outside both svm-wasm and the interp TCB), plus a `resolve` policy. A real WASI
  "hello world" runs end-to-end (`crates/svm-wasi/src/lib.rs` tests). WASI's specific fd/clock/random
  *semantics* stay a ⚪ non-goal — the shim is a host-layer subset, not conformant preview1.
- [ ] **Multi-*handle* import binding (remaining).** Still **one** capability handle threaded (a single
  leading param; `has_handle` is module-wide), so all named imports must share one interface — fine for
  WASI (one `HostFn` handle, many ops) but a module spanning **distinct** capability interfaces is
  rejected. Work: thread **N** handles via reserved slots (the chibicc multi-handle powerbox pattern)
  instead of one leading param. (The `wasi:thread/spawn` import remains separately special-cased.)

### 🟡 Fail-closed feature gaps (clean `Unsupported`)

- [x] **Tail calls** (`return_call` / `return_call_indirect`) — **DONE.** Lower to the IR's
  `Terminator::ReturnCall`/`ReturnCallIndirect`, which both backends execute as **true** tail calls
  (interp replaces the frame; JIT emits Cranelift `return_call`/`return_call_indirect`). Direct tail
  call to a defined function (true tail call), indirect via the §3c table dispatch, and a capability
  import degrades to `cap.call` + `return` (correct, not tail-optimized; wasi:thread/spawn rejected).
  `tests/tailcall.rs`: 200k-deep tail recursion (would overflow a non-tail call), table dispatch,
  mutual even/odd recursion.
- [x] **Passive *data* segments + `memory.init`/`data.drop` — DONE.** A passive segment's bytes are
  known at transpile time, so a **constant-offset** `memory.init` (the toolchain's `__wasm_init_memory`
  shape: `src = 0`, `len = seg_len`, a possibly-runtime `dest`) unrolls into chunked const-stores of
  those bytes — reusing the `memory.copy`/`fill` machinery, no IR/runtime change. `data.drop` is a
  no-op (bytes are inlined at the init site). A non-constant `src`/`len` is fail-closed `Unsupported`
  (no runtime passive-data store); a static source-OOB is a clean transpile error. `tests/bulk.rs`
  (passive + active segments, partial range, runtime dest, multi-segment indexing, dynamic-len reject).
- [ ] **The *table* bulk ops + passive *element* segments**: `elem.drop`, `table.init`, `table.copy`,
  `table.fill`, `table.grow`, `table.size`. Need a **mutable runtime table** (a size cell + funcref
  stores into the table region), unlike the memory side. Lower audience — C/Rust WASI output uses
  *active* elem segments (already supported); passive elem + `table.init` shows up mainly with
  reference-types / dynamic linking.
- [ ] **Reference types**: `externref`/`funcref` as values, `table.get`/`set`/`grow`/`size`/`fill`,
  `ref.null`/`ref.func`/`ref.is_null`, typed `select (result t)`. Natural SVM fit: `externref` →
  capability-handle (an i32 host-table index), `funcref` → funcref-index (already powers
  `call_indirect`); the table-mutation ops are the fiddly part. Low audience (C/C++/Rust don't emit it).
- [ ] **SIMD remainder** (~135 of the v128 proposal): lane shifts, int abs, sat add/sub, narrow/widen,
  `all_true`/`bitmask`, dot product, float lane compares, etc. Mechanical breadth over the proven 5-step
  pattern (IR variant → verifier lane rule → interp ref → JIT Cranelift → transpiler arm); a few ops have
  no single Cranelift instruction (cf. `i8x16.mul` already bailing on the JIT).
  - [x] **Integer lane compares — DONE.** `i8x16`/`i16x8`/`i32x4` `{eq,ne,lt,gt,le,ge}` s/u + the
    `i64x2` signed set → a per-lane all-ones/all-zeros mask (`Inst::VIntCmp`, one Cranelift `icmp`).
    `crates/svm/tests/simd.rs` (round-trip + interp==JIT, oracle = Rust's own compares) and
    `svm-wasm/tests/simd.rs` (the wasm bridge + a real `bitselect`-max idiom).
  - [x] **Integer min/max — DONE.** `i8x16`/`i16x8`/`i32x4` `{min,max}_{s,u}` (extends `VIntBinOp` →
    one Cranelift `smin`/`umin`/`smax`/`umax`; `i64x2` has no min/max op, and would not legalize so it
    bails on the JIT alongside `i8x16.mul`). Tests incl. a real lane-wise `clamp` kernel.
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
- [ ] **`wasi:thread/spawn` *alongside* capability imports** (needs the per-thread handle stash). The
  broader "imports spanning multiple capability interfaces" limitation is tracked as the 🟠
  *named multi-capability import binding* item above.

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
2. **Named multi-capability import binding** 🟠 — let the embedder bind symbolic import names to
   `(type_id, op)` and thread N capability handles (the chibicc powerbox pattern). The general
   mechanism for "host provides arbitrary capabilities, guest uses them"; the single biggest
   real-program blocker. (WASI *semantics* stay ⚪ — a host binds its own caps to whatever names.)
3. **Tail calls** 🟡 — common LLVM output, likely near-free (IR terminators exist).
4. **Passive *data* segments + `memory.init`/`data.drop`** 🟡 — DONE. (The *table* bulk ops + passive
   *element* segments remain — they need a mutable runtime table; lower audience.)
5. **Reference types** 🟡 (externref→handle, funcref→index), then the **SIMD remainder** 🟡 (breadth),
   then **narrow-atomic CAS-loop** 🟡.
6. EH, relaxed SIMD, multiple memories/tables, imported globals/tables — on demand. GC stays ⚪.

Code map: the rejection sites are the `unsup(...)` calls in `crates/svm-wasm/src/lib.rs` (section
parse + the `worker_op` operator catch-all `other => unsup("operator {…}")`); tests live in
`crates/svm-wasm/tests/`.
