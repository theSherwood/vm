# wasm coverage & roadmap (`crates/svm-wasm`)

What the **wasm → IR transpiler** handles, what it doesn't, and the prioritized work to widen
coverage. svm-wasm is a *second frontend* (after chibicc): it takes core wasm and reconstructs SSA
from the stack machine, so the §1a benchmark thesis can be measured on the **same bytes** Wasmtime
runs. It is an **untrusted** frontend — everything it emits is re-verified by `svm-verify`, so a gap
here is a *capability* limit, never a safety one.

**Status: feature-complete for *typical clang/rustc -O2 output*** (95 tests across
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
- [ ] **SIMD remainder** (~68 of the v128 proposal):
  extadd/extmul, q15mulr, etc.
  Mechanical breadth over the proven 5-step
  pattern (IR variant → verifier lane rule → interp ref → JIT Cranelift → transpiler arm); a few ops have
  no single Cranelift instruction (cf. `i8x16.mul` already bailing on the JIT).
  - [x] **Integer lane compares — DONE.** `i8x16`/`i16x8`/`i32x4` `{eq,ne,lt,gt,le,ge}` s/u + the
    `i64x2` signed set → a per-lane all-ones/all-zeros mask (`Inst::VIntCmp`, one Cranelift `icmp`).
    `crates/svm/tests/simd.rs` (round-trip + interp==JIT, oracle = Rust's own compares) and
    `svm-wasm/tests/simd.rs` (the wasm bridge + a real `bitselect`-max idiom).
  - [x] **Integer min/max — DONE.** `i8x16`/`i16x8`/`i32x4` `{min,max}_{s,u}` (extends `VIntBinOp` →
    one Cranelift `smin`/`umin`/`smax`/`umax`; `i64x2` has no min/max op, and would not legalize so it
    bails on the JIT alongside `i8x16.mul`). Tests incl. a real lane-wise `clamp` kernel.
  - [x] **Float lane compares — DONE.** `f32x4`/`f64x2` `{eq,ne,lt,gt,le,ge}` → mask (`Inst::VFloatCmp`,
    one Cranelift `fcmp`; ordered, `ne` unordered — matches Rust's float operators, the test oracle,
    incl. NaN). `crates/svm/tests/simd.rs` + `svm-wasm/tests/simd.rs`.
  - [x] **Lane shifts — DONE.** `i8x16`/`i16x8`/`i32x4`/`i64x2` `{shl,shr_s,shr_u}` by a scalar `i32`
    amount mod the lane width (`Inst::VShift`; vector `ishl`/`ushr`/`sshr` — Cranelift legalizes every
    shape incl. `i8x16`). Oracle = Rust's scalar shifts at the lane width.
  - [x] **Integer abs/neg — DONE.** `i8x16`/`i16x8`/`i32x4`/`i64x2` `{abs,neg}` (`Inst::VIntUn`, the
    unary int sibling of `VFloatUn`; vector `iabs`/`ineg`, all shapes legalize incl. `i64x2.abs`).
    Two's-complement wrap (`abs(INT_MIN) == INT_MIN`); oracle = Rust's `wrapping_abs`/`wrapping_neg`.
  - [x] **Boolean reductions — DONE.** `v128.any_true`, `iNxM.all_true`, `iNxM.bitmask` → an `i32`
    (`Inst::VAnyTrue`/`VAllTrue`/`VBitmask`; Cranelift `vany_true`/`vall_true`/`vhigh_bits`). The
    v128→i32 result shape — how vectorized code **branches on a lane compare**. Tests incl. a SIMD
    `memchr` (`eq` + `any_true`) and a move-mask (`lt_s` + `bitmask`).
  - [x] **Saturating add/sub — DONE.** `i8x16`/`i16x8` `{add,sub}_sat_{s,u}` (`Inst::VSatBin`, a
    dedicated family the **verifier restricts to the two narrow shapes** the wasm spec defines — so
    no JIT bail list and the fuzzer can't reach a wide-shape sat; Cranelift `sadd_sat`/… native on
    x86/aarch64). Oracle = Rust's `saturating_add`/`saturating_sub`; tests incl. a pixel-blend idiom.
  - [x] **Lane widening (extend) — DONE.** `i16x8`/`i32x4`/`i64x2`.`extend_{low,high}_*_{s,u}`
    (`Inst::VWiden` + `VShape::narrower`/`wider` helpers; Cranelift `swiden`/`uwiden` low/high). The
    verifier rejects a result shape with no narrower source. Tests across all three width steps.
  - [x] **Lane narrowing — DONE.** `i8x16`/`i16x8`.`narrow_*_{s,u}` — two wide vectors saturated into
    one narrow vector, `a` then `b` (`Inst::VNarrow`; Cranelift `snarrow`/`unarrow`; verifier restricts
    to the two narrow result shapes). Source read as signed, `s`/`u` pick the clamp range; tests incl.
    a clamp-pack idiom.
  - [x] **int↔float / float↔float conversions (all 10) — DONE.**
    `f32x4.convert_i32x4_{s,u}`, `i32x4.trunc_sat_f32x4_{s,u}`, `f32x4.demote_f64x2_zero`,
    `f64x2.promote_low_f32x4`, plus the **f64↔i32** `f64x2.convert_low_i32x4_{s,u}` and
    `i32x4.trunc_sat_f64x2_{s,u}_zero` (`Inst::VConvert`, whole-instruction mnemonics; Cranelift
    `fcvt_from_{s,u}int`/`fcvt_to_{s,u}int_sat`/`fvdemote`/`fvpromote_low`). Rust's `as` casts are the
    oracle — they already match wasm's round-to-nearest + `trunc_sat` (NaN→0, clamp). The four
    lane-count-changing f64↔i32 ops widen/narrow through an `i64x2` intermediate (`swiden_low`/
    `uwiden_low` for convert_low; `snarrow`/`uunarrow` against a zero vector for trunc_sat_zero) —
    the same recipe Cranelift's own wasm frontend uses.
  - [x] **Pseudo-min/max (pmin/pmax) — DONE.** `f32x4`/`f64x2` `{pmin,pmax}` (`Inst::VPMinMax`, a
    float-only family the verifier restricts to the two float shapes). Unlike IEEE `min`/`max` these
    are a one-sided compare-and-select — `pmin(a,b)=b<a?b:a`, `pmax(a,b)=a<b?b:a` — so a NaN operand
    and signed zeros propagate by the `<` rule (what wasm/LLVM want for `fmin`/`fmax` reductions). JIT
    lowers to one `fcmp` + `bitselect`; oracle = that exact select, tests incl. NaN/-0 cases.
  - [x] **Population count (`i8x16.popcnt`) — DONE.** Per-byte popcount (`Inst::VPopcnt`, no shape
    field — `i8x16` is the only shape wasm defines, so the verifier needs no lane rule). JIT lowers
    to a single vector `popcnt` (native `cnt` on aarch64, a byte-shuffle sequence on x86); oracle =
    Rust's `count_ones`.
  - [x] **Unsigned rounding average (`avgr_u`) — DONE.** `i8x16`/`i16x8` `(a+b+1)>>1` per lane
    (`Inst::VAvgr`, a dedicated family the **verifier restricts to the two narrow shapes** — so no
    JIT bail list, like saturating add/sub). JIT lowers to native `avg_round`; oracle computes the
    average wide. Tests incl. a pixel-blend idiom through the wasm bridge.
  - [x] **Dot product (`i32x4.dot_i16x8_s`) — DONE.** Signed dot of adjacent `i16` pairs into `i32`
    lanes — `out[i] = a[2i]·b[2i] + a[2i+1]·b[2i+1]` (`Inst::VDot`, fixed `i16x8`→`i32x4`, no shape
    field). JIT lowers to `swiden_low/high` + `imul` + `iadd_pairwise` (the pairwise-add legalizes
    the lane-count change); oracle = the same pair-sum in Rust, incl. the `i32` overflow corner.
    Tests incl. a DSP-style dot kernel through the wasm bridge.
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
5. **SIMD remainder** 🟡 — **mostly landed** (15 of ~17 op families: compares (int+float), min/max,
   shifts, abs/neg, the boolean reductions, saturating add/sub, widen, narrow, all 10 int↔float /
   float↔float conversions, pmin/pmax, `i8x16.popcnt`, `avgr_u`, dot product). The **tail** remains:
   extadd/extmul and `q15mulr_sat`. Same proven 5-step pattern.
6. **Reference types** 🟡 (externref→handle, funcref→index), then the **narrow-atomic CAS-loop** 🟡.
7. EH, relaxed SIMD, multiple memories/tables, imported globals/tables — on demand. GC stays ⚪.

Code map: the rejection sites are the `unsup(...)` calls in `crates/svm-wasm/src/lib.rs` (section
parse + the `worker_op` operator catch-all `other => unsup("operator {…}")`); tests live in
`crates/svm-wasm/tests/`.
