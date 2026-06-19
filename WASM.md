# wasm coverage & roadmap (`crates/svm-wasm`)

What the **wasm → IR transpiler** handles, what it doesn't, and the prioritized work to widen
coverage. svm-wasm is a *second frontend* (after chibicc): it takes core wasm and reconstructs SSA
from the stack machine, so the §1a benchmark thesis can be measured on the **same bytes** Wasmtime
runs. It is an **untrusted** frontend — everything it emits is re-verified by `svm-verify`, so a gap
here is a *capability* limit, never a safety one.

**Status: feature-complete for *typical clang/rustc -O2 output*** (121 tests across
`transpile.rs`/`imports.rs`/`simd.rs`/`atomics.rs`/`threads.rs`/`start.rs`/`tailcall.rs`/`bulk.rs`/`reftypes.rs`).
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
  `memory.copy`/`memory.fill`; **fixed-128 SIMD** (the complete v128 op set, D58 — all
  arith/bitwise/shuffle/compare/convert/widen/narrow/dot/extmul/q15 lanes + the memory variants
  (splat-load/load-extend/load-zero/load+store-lane) + all of **relaxed SIMD** via deterministic
  lowerings incl. a fused FMA and the signed-i8 relaxed dot);
  **threads** —
  the full `*.atomic.*` set (full-width i32/i64 map 1:1 onto IR atomics; the narrow 8/16-bit forms
  emulate via a 32-bit word-CAS loop) + `atomic.fence`, `shared`+imported memory, and the
  **wasi-threads** `wasi:thread/spawn` → native `thread.spawn` (a synthesized shim + unique-tid slot).
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

### 🟢 Host-ABI — named import binding (DONE)

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
- [x] **Multi-*handle* import binding — DONE.** The transpiler now threads **one handle per distinct
  import interface** (keyed by the wasm `module` string, in first-appearance order) as the leading
  `i32` params of every function — so a module spanning N capability interfaces takes N leading handle
  params (N=0/1 collapse to the no-handle / single-handle cases, byte-identical to before). Each
  `cap.call`/`CallImport` rides its interface's slot handle; the embedder grants one capability per
  interface and passes the handles as the entry's leading args, in slot order. The module string is the
  grouping key because it is known at transpile time for both numeric and §7 named imports. Purely an
  svm-wasm (frontend) change — the IR/interp/JIT already take a per-call handle. The `Lower` prefix
  machinery generalized `handle: Option<ValIdx>` → `handles: Vec<ValIdx>`; the old "one interface only"
  rejection is gone. Differential tests (interp == JIT) cover two distinct interfaces (Clock + Blocking)
  bound to two handles, threaded through both the entry and a cross-function `call`. (The
  `wasi:thread/spawn` import is separately special-cased; capabilities now reach spawned threads via
  the window handle stash — see the threads section.)

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
- [~] **Reference types — core landed; the bulk/dynamic-table ops remain.** Both ref types are an
  **i32 index** in SVM — `funcref` → the §3c function-table index (already powers `call_indirect`),
  `externref` → a §7 capability handle (an opaque host-table index). So the whole "reference" half is
  i32 plumbing over **existing IR** (no new ops, no verifier rules — the table is just i32-granular
  window memory). **Done:** `funcref`/`externref` as values (params/results/locals/globals);
  `ref.null`/`ref.is_null`/`ref.func` (null = the `0xFFFF_FFFF` sentinel the table already uses); typed
  `select (result t)`; `table.get`/`set`/`size`/`fill` (the i32-slot twins of the memory ops);
  declarative `elem` segments (a no-op). OOB indices **mask** into the window (the §1a model, like
  memory — not a trap); the `call_indirect` §3c type-check still guards a forged funcref, and a forged
  externref faults at `cap.call` (authority lives in the host's grant table, not the handle's bits).
  `tests/reftypes.rs` (a vtable via `ref.func`+`table.set`+`call_indirect`, get/set round-trip, fill,
  typed select, externref pass-through — all interp == JIT).
- [ ] **Reference types — remaining**: `table.copy`, `table.init` + passive *element* segments +
  `elem.drop`, `table.grow` (+ a size cell and a growable table region — the layout-invasive bit), and
  **multiple tables** (`table.get`/etc. on a non-zero table). `table.copy`/`init` mirror
  `memory.copy`/`init`; `grow` mirrors `memory.grow`. Lower audience (dynamic-linking / GC-language
  glue; C/C++/Rust compute doesn't emit it).
- [x] **SIMD — DONE (fixed-width v128 *and* relaxed).** The whole wasm SIMD surface transpiles + runs
  on both backends: every arithmetic/lane/convert/shuffle op, the **memory variants** (splat-load,
  load-extend, load-zero, load/store-lane), and the **relaxed-SIMD** extension. Built over the proven
  5-step pattern (IR variant → verifier lane rule → interp ref → JIT Cranelift → transpiler arm); a
  few ops bail on the JIT where Cranelift can't legalize them (`i8x16.mul`, `i64x2` min/max — the
  interp still covers them and wasm never emits them).
  - [x] **SIMD memory variants — DONE.** `v128.load{8,16,32,64}_splat`, `load{8x8,16x4,32x2}_{s,u}`,
    `load{32,64}_zero`, `load/store{8,16,32,64}_lane` (clang `-msimd128` emits these constantly to
    broadcast/gather). No new IR — each composes a scalar `Load`/`Store` with `Splat`/`ReplaceLane`/
    `ExtractLane`/`VWiden`. `svm-wasm/tests/simd.rs` (all shapes, interp == JIT).
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
  - [x] **Extended multiply (`extmul`) — DONE.** All 12 `<wide>.extmul_{low,high}_<src>_{s,u}`
    (`i16x8`←`i8x16`, `i32x4`←`i16x8`, `i64x2`←`i32x4`): widen the low/high half of both operands
    (`Inst::VExtMul`, reusing `VWidenOp` for the half+sign) then `imul` on the wide shape (legalizes
    for every wide shape, incl. `i64x2`). Verifier requires a narrower source. Oracle = the widened
    product (`i128` intermediate); tests pin the half selection with a distinct-lane const.
  - [x] **Extended pairwise add (`extadd_pairwise`) — DONE.** All 4 `<wide>.extadd_pairwise_<src>_
    {s,u}` (`i16x8`←`i8x16`, `i32x4`←`i16x8`): widen every lane and sum adjacent pairs
    (`Inst::VExtAddPairwise`) — JIT lowers to `swiden/uwiden` low+high + `iadd_pairwise`, whose two
    halves' pairwise sums concatenate to `out[i] = w(a[2i]) + w(a[2i+1])`.
  - [x] **Q15 rounding multiply (`i16x8.q15mulr_sat_s`) — DONE.** Signed Q15 fixed-point multiply
    with rounding + saturation — `out[i] = sat_i16((a·b + 0x4000) >> 15)` (`Inst::VQ15MulrSat`, fixed
    `i16x8`). JIT lowers to native `sqmul_round_sat`; oracle = the formula in `i64`, tests incl. the
    `-1.0·-1.0` corner that saturates to `i16::MAX`. Tests incl. a DSP idiom through the wasm bridge.
- [x] **Narrow atomics — DONE.** `*.atomic.{load,store,rmw8/16/32.*,cmpxchg}{8,16}` (and i64's 32-bit
  forms). SVM IR atomics are 32/64-bit only (the §3b narrow-integer decision), so the **8/16-bit**
  forms emulate with a **32-bit word-CAS loop** in the transpiler: align to the containing word, then
  load → splice the sub-word (`(old & ~mask) | (new << shift & mask)`) → `cmpxchg`, retrying until it
  lands (load is loop-free: word-load + shift/mask). The i64 **32-bit** forms are word-sized (a native
  i32 atomic, zero-extended). No IR/TCB change (i8/i16 stay out of the escape-TCB). A naturally-aligned
  narrow access lies in one word (wasm requires it), so the word-CAS is exact; a *misaligned* one isn't
  trapped (the §1a confine-don't-trap stance). `tests/atomics.rs` (sub-word extract, splice preserving
  neighbours, the rmw old-value + wrap contracts, cmpxchg hit/miss, the i64 forms — all interp == JIT)
  + `tests/threads.rs` (8 workers × 1000 `rmw16` increments = exactly 8000, the real atomicity proof
  under contention on both backends).
- [ ] **Imported globals & imported tables.** Dynamic-linking / PIC (`__memory_base`/`__table_base`/
  `__stack_pointer` imports). Statically-linked output defines its own, so this only bites `-shared`/PIC.
- [x] **Relaxed SIMD — DONE (all 20)** (`-mrelaxed-simd`). A separate proposal of ~20 ops whose
  results are **implementation-defined within a spec-allowed set** — they let an engine emit one native
  instruction (x86 FMA, ARM rounding, `blendv`, `pmaddubsw`) without the fix-up sequence deterministic
  SIMD needs to be bit-identical across architectures. SVM ships them via the **deterministic-choice**
  realization: each op lowers to *one* spec-allowed behavior computed identically in both backends, so
  the interp↔JIT differential holds (no value-insensitive exclusion needed yet):
  - `relaxed_madd`/`relaxed_nmadd` (f32x4/f64x2) → a genuine **fused FMA** (`Inst::VFma`; Cranelift
    `fma`, interp `f*::mul_add` — both correctly-rounded, so bit-identical). This is the *one op that
    gets a real speedup* (the usual reason to reach for `-mrelaxed-simd`), and it stays differentiable.
  - `relaxed_min`/`max` → the deterministic `fmin`/`fmax`; `relaxed_trunc_*` (4) → `trunc_sat`;
    `relaxed_laneselect` (4) → `bitselect` (exact for a valid all-0/all-1 mask); `relaxed_swizzle` →
    `swizzle`; `relaxed_q15mulr_s` → `q15mulr_sat`. All alias existing deterministic ops — transpiler
    arms only, no new IR. `crates/svm/tests/simd.rs` (FMA vs `mul_add`, incl. fused≠mul+add cases) +
    `svm-wasm/tests/simd.rs` (the `-mrelaxed-simd` shape on both backends).
  - `relaxed_dot_i8x16_i7x16_s` → `Inst::VDotI8` (the deterministic **signed-i8 dot** → i16, the
    spec-allowed signed-×-signed behavior, not x86's unsigned-×-signed `pmaddubsw`; `swiden` + `imul`
    + `iadd_pairwise`, the same recipe Cranelift's own deterministic-relaxed mode uses). The
    `relaxed_dot_i8x16_i7x16_add_s` variant composes that dot with the existing `extadd_pairwise` +
    `i32x4.add` (no extra IR). `crates/svm/tests/simd.rs` (incl. the i16 wrap corner) +
    `svm-wasm/tests/simd.rs`.
  The deterministic-default rationale, since it's a recurring question:
  - **The base-SIMD "fixups" are op *semantics*, not a determinism tax.** `i32x4.trunc_sat_f32x4_s`
    *means* "saturate out-of-range, NaN→0"; the clamp instructions Cranelift emits on x86 are the cost
    of computing **that operation**, not a surcharge SVM adds. Emitting raw `cvttps2dq` instead
    wouldn't be "faster non-deterministic SIMD" — it would compute a *different function* than the
    bytecode specifies (wrong, not merely unportable). For the **vast majority** of v128 ops
    (arith/bitwise/shift/shuffle/lane/`i*` min-max) there is **zero fixup** — register-to-register,
    bit-identical across hardware for free. The handful with fixups (float→int trunc, float min/max,
    narrow) emit *the same Cranelift lowering Wasmtime ships*, so SVM is at parity with the "as fast as
    wasm" baseline it measures against (D36) — it never pays *more* than the engine it's compared to.
  - **The one genuine speed-vs-determinism knob already defaults to fast.** Float **NaN bit patterns**
    are host-defined in the default mode (fast, matches hardware) and canonicalized only in the opt-in
    deterministic mode (DESIGN §12); the interp↔JIT differential is correspondingly *NaN-insensitive
    per lane* (DESIGN §17 "float lanes are NaN-insensitive in the differential"). So wherever leaving
    a result un-fixed-up is a real (free) choice, the default is *already* speed-first.
  - **What relaxed SIMD would actually cost is JIT trust, not "purity".** The interp↔JIT differential
    is how SVM *trusts its own JIT* — the primary evidence Cranelift didn't miscompile into a
    confinement escape (§18/I4), with the architecture-independent interpreter as the oracle. A native
    relaxed lowering makes interp and JIT legitimately diverge in *value* (not just NaN bits), so that
    evidence evaporates for those ops. That's a security-model cost, in the exact dimension SVM
    competes on vs. "just run Wasmtime."
  - **What's shipped is the deterministic-choice realization** (above): each relaxed op runs one
    spec-allowed behavior, so a `-mrelaxed-simd` binary *runs correctly* and `relaxed_madd` even gets
    the real FMA speedup, all while keeping interp==JIT. No global default was flipped.
  - **Future native-speed opt-in (if ever needed):** for the ops where the deterministic choice is
    slower than the native one (`relaxed_trunc`'s saturation fixup, `relaxed_min/max`'s NaN handling),
    extend the existing *NaN-insensitive-per-lane* differential precedent from "ignore NaN payload
    bits" to "ignore the result of this *op*": emit the fast native lowering, mark only those ops
    **value-insensitive** in the differential, keep the byte-exact oracle for everything else. That
    buys max-speed relaxed SIMD per-op without globally weakening the JIT-trust story — strictly better
    than flipping the default. Not built; the deterministic lowerings cover correctness today.
- [ ] **Multiple memories** and **multiple tables** (`lib.rs` rejects both).
- [ ] **Typed function references** (the function-references proposal) and **extended const
  expressions** (arithmetic in global/data-offset initializers — currently only constants).
- [ ] **Exception handling**: `try`/`catch`/`catch_all`/`throw`/`rethrow`/`delegate` + tags. Involved:
  within a function it lowers to branches, but cross-frame propagation needs an exception channel (a
  generalization of the existing per-call trap-cell check) — a perf tax + calling-convention change,
  and clang/Rust only emit it under opt-in (`-fwasm-exceptions`). Low ROI.
- [x] **`wasi:thread/spawn` *alongside* capability imports — DONE.** A spawned thread reaches the
  module's capabilities via a **window handle stash**: a reserved region of `n_handles` i32 slots just
  past the tid counter. The spawning function holds its capability handles as the multi-handle prefix,
  so `spawn_op` stores them into the stash right before `thread.spawn` (program order → happens-before
  the new thread); the synthesized shim reads them back on the new vCPU and threads them into
  `wasi_thread_start` (which carries the N-handle prefix like every defined function). No runtime
  change — the powerbox/host is already shared across vCPUs (interp `Arc<Mutex<Host>>`, JIT baked-in
  `cap_ctx`), so an i32 handle is valid on any thread. Transpiler-only; `n_handles == 0` is
  byte-identical to the old threads-only shim. `tests/imports.rs`: `n` spawned workers each `cap.call`
  `work(start_arg)` and atomically sum the results to `Σ mix(i)` — interleaving-invariant, so interp's
  M:N executor and the JIT's OS threads agree.

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
2. **Named multi-capability import binding** 🟢 — **DONE.** Named imports bind to `(type_id, op)` at
   load, and the transpiler threads **N capability handles** (one per distinct import interface/module,
   the powerbox pattern). A module can now span arbitrary capability interfaces. (WASI *semantics*
   stay ⚪ — a host binds its own caps to whatever names.)
3. **Tail calls** 🟡 — common LLVM output, likely near-free (IR terminators exist).
4. **Passive *data* segments + `memory.init`/`data.drop`** 🟡 — DONE. (The *table* bulk ops + passive
   *element* segments remain — they need a mutable runtime table; lower audience.)
5. **SIMD** 🟢 — **DONE (entire surface).** The full v128 op set transpiles + runs on both backends:
   compares, min/max, shifts, abs/neg, reductions, saturating add/sub, widen, narrow, all 10
   conversions, pmin/pmax, popcnt, `avgr_u`, dot product, extmul, extadd_pairwise, q15mulr_sat, the
   memory variants (splat-load/load-extend/load-zero/load+store-lane), **and** all 20 relaxed-SIMD ops
   via deterministic lowerings (incl. a fused FMA and the signed-i8 dot).
6. **Narrow-atomic CAS-loop** 🟢 — **DONE.** **Reference types** 🟡 — core landed (values, `ref.*`,
   typed select, `table.get/set/size/fill`); the bulk/dynamic-table ops (`copy`/`init`/`grow`/multi)
   remain.
7. EH, multiple memories/tables, imported globals/tables — on demand. GC stays ⚪.

Code map: the rejection sites are the `unsup(...)` calls in `crates/svm-wasm/src/lib.rs` (section
parse + the `worker_op` operator catch-all `other => unsup("operator {…}")`); tests live in
`crates/svm-wasm/tests/`.
