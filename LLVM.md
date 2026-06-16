# LLVM-bitcode → IR on-ramp (`crates/svm-llvm`) — design & tracking

The plan, the constraints, and the prioritized work for the **third frontend**: an
**ahead-of-time LLVM-bitcode → SVM-IR translator**. This is the big *breadth* play
(D54) — one component buys every LLVM language (C, C++, Rust, Swift, Zig…) as a guest.

This file is the working tracker for the on-ramp, the analog of `WASM.md` for the wasm
bridge. Like that doc, fold completed sections into `DESIGN.md` and drop this file once
the actionable gaps close (the repo convention, cf. the former `WASM.md`/`SCHEDULING.md`).

**Status: Milestone 1 slices A–N (control flow, memory, calls, switch, globals, floats, indirect
calls, struct aggregates, memory intrinsics, by-value aggregates, relocations, libm math, int
min/max+bit intrinsics, and the powerbox libc on-ramp) done — a broad swath of scalar C from
`clang -O2` runs on both backends (44 tests). A **kitchen-sink capstone** exercises everything at
once (structs by-value, a function-pointer table, floats+libm, recursion, loops, an array `memcpy`,
a global array, `switch`, bit intrinsics) and matches **native `cc`** end to end. **Slice N** binds
libc I/O to host capabilities (`write`/`read` → `Stream`, `exit` → `Exit`) via §7 named imports and
a synthesized powerbox `_start`: a real I/O program now runs through the reference powerbox with
**stdout + exit code matching native** (`check_powerbox_vs_native`). Next: more libc surface
(`malloc`/`printf`), then the real-library demo corpus (Lane C).** `crates/svm-llvm` does the **SSA → block-argument
conversion** (LLVM dominance SSA + φ-nodes → SVM's block-local form via liveness; loops/joins/
critical edges, no edge splitting), the integer scalar op set, the **§3d data-stack** (`alloca` →
window frame slots, `load`/`store` incl. narrow widths, `getelementptr` → address arithmetic),
**direct calls with the threaded data-SP** (every function takes a leading `sp`; a call passes
`sp + frame_size`, so recursion is sound), **`switch` → `br_table`**, **global variables**
(low in the window as `data` segments, constants read-only D40, with a stack guard above; a
`@global` ref is its window address), and **`f32`/`f64`** (arithmetic/compare/conversions +
the common float intrinsics, `fmuladd` lowered unfused). Real `clang -O2` programs — popcount/
collatz loops, if-converted select, a stack-array sum, recursive `fib`, a cross-function call, a
dense switch, `even`/`odd` mutual recursion, a const lookup table, a mutable global counter,
indexed string reads, a gapped switch (a global jump table), double arithmetic/compares/
conversions, `fabs`/`floor`, an indirect call through a function pointer, struct field access
(global/array-of-struct/stack), a struct `memcpy` + `memset`, and by-value struct args/returns
(small-coerced + `byval`/`sret`), and pointer-valued global relocations (a function-pointer table,
a struct string-pointer member), libm math calls (`sqrt`/`fmin`), and int min/max + bit intrinsics
(`smax`/`ctlz`/`popcount`) — run **interp == JIT == hand-computed** (44 tests, incl. a kitchen-sink
program checked against native `cc`, plus a `write`/`exit`/`read`-echo powerbox program checked
against native stdout + exit code).
Remaining M1: more libc surface (`malloc`/`printf`), then the
demo Lane C. Section numbers like "§3d"
refer to `DESIGN.md`; "D54" etc. are its Decision Log.

---

## 1. Why LLVM, and why now

The two on-ramps in DESIGN §20 are **LLVM → IR (breadth)** and **wasm → IR (compat)**.
The wasm bridge (`svm-wasm`) is feature-complete for typical `clang`/`rustc -O2` output
(see `WASM.md`); the LLVM bridge is the remaining frontier (HANDOFF §10 "▶ NEXT", D54).

**Thesis (DESIGN §20): we are a strictly better LLVM target than wasm.** The things LLVM
emits naturally and that wasm forces a frontend to *contort* are exactly our §1a edges:

| LLVM emits | wasm forces | SVM gives it natively |
|---|---|---|
| irreducible CFG | relooper / stackify (extra blocks+branches) | native irreducible CFG (D2/§3) |
| 64-bit pointers | wasm32 windowing / wasm64 bounds checks | 64-bit address space + one mask (§4) |
| multiple return values | single result + memory spill | multi-result instructions (§3a) |
| `musttail` tail calls | not in core wasm | first-class `return_call`/`_indirect` (D6) |
| **SSA with φ-nodes** | stackify → consumer SSA reconstruction | **SSA on the wire**; φ → block params (§3a) |

The last row is the cleanest win and the reason the LLVM path is *less* work than the
wasm path in its core: `svm-wasm` had to **reconstruct SSA from a stack machine**; we
**already have SSA** from LLVM and only need to translate it. LLVM φ-nodes map directly
onto our typed block parameters (§3a "no phi nodes"): each `phi` at a block's head becomes
a block parameter, and each predecessor's terminator supplies the matching branch
argument. (Critical edges get split first — standard.)

---

## 2. Decisions already taken (D54) — the frame

- **Untrusted frontend, no TCB cost (§2a).** `svm-llvm` is the same trust class as the
  chibicc fork and `svm-wasm`: it consumes the core crates to *produce* a Module, and is
  **never a dependency of `svm-jit`/`svm-interp`**. Everything it emits is re-verified by
  `svm-verify`, so a translation bug is a **clean error, never an escape**. Adding LLVM
  costs zero escape-TCB — the eBPF lesson generalized (DESIGN §20).
- **Architecture: AOT (HANDOFF §10 / D54).** The translator links libLLVM at build/dev
  time and is **off the runtime path** — it does *not* go into the ~5 MiB JIT binary. We
  ingest already-compiled bitcode; we are not a JIT-time LLVM dependency.
- **Vehicle: a PNaCl-style bitcode translator, not a from-scratch TableGen backend**
  (D54/D49). The cited NaCl/PNaCl lineage — "SSA as a portable sandbox target" — is the
  team-tractable form.
- **Pin a frozen subset.** LLVM bitcode is **not a stable format** (DESIGN §20). We pin a
  specific LLVM version and a legalized subset of constructs we accept, exactly as PNaCl
  did. Anything outside the subset is a hard, fail-closed `Unsupported` error (never
  silent mis-translation) — same discipline as `svm-wasm`'s `unsup(...)`.
- **MVP scope (D54):** the **scalar + memory + call** subset that chibicc already proves
  end-to-end — aggregates via memory, hard-error on vectors and unsupported intrinsics —
  with a differential harness running the existing C demos through *stock LLVM* and
  matching native `clang`.

### Toolchain present in the dev container (confirmed)
- `clang` 18.1.3, `llvm-config` 18.1.3 (`/usr/lib/llvm-18/lib`).
- `libLLVM.so.18.1` present (plus 17/20/21 — we **pin 18**, the `clang` default here).

So the pinned baseline is **LLVM 18**. (Re-pin deliberately, never drift; a bitcode
produced by a different major version is rejected, not best-effort parsed.)

---

## 3. The hard constraints (read before writing any translation)

Three constraints shape every translation decision. The first two are *forced* by settled
design; the chibicc frontend (`frontend/chibicc/codegen_ir.c`) already solves all three
and is the **oracle** for how (see §5).

### 3a. The two-stack split (§3d) — non-negotiable
A pointer to an address-taken object must be a **window offset** so access through it is
masked + MMU-confined (§4). The control stack is **out-of-band** (§5) and not in the
window. Therefore any frontend must place:

| Goes to | What | LLVM source |
|---|---|---|
| **SSA value** (register/spill, out-of-band) | scalars never address-taken | LLVM SSA registers after `mem2reg` |
| **data stack** (in-window, `ptr.add`+load/store) | address-taken locals, aggregates, `alloca`, varargs, `sret` | LLVM `alloca`s that survive `mem2reg` |

**LLVM does the hard half for us.** chibicc allocates *all* locals to memory and we wrote a
reverse SSA-promotion pass to lift scalars out (HANDOFF §3). With LLVM we **run `mem2reg`/
SROA in the ingest pipeline** so the bitcode arrives with scalars already in SSA registers;
the `alloca`s that *remain* are genuinely address-taken → data-stack slots. The two-stack
classification falls out of LLVM's own promotion — no bespoke pass needed.

### 3b. Narrow integers — the wasm tradeoff (§3b note 1, "revisit at the LLVM on-ramp")
SVM SSA value types are **`{i32, i64}`** only; `i8`/`i16` exist only as memory access
widths. LLVM has native `i1`/`i8`/`i16`/`i24`/… So the translator must **collapse narrow
integers to `i32`** and re-emit truncation explicitly — DESIGN §3b names this exact task:
*"the LLVM on-ramp (D54) will need the same discipline when collapsing LLVM's native
`i8`/`i16` to `i32`."*

- `i1` (from `icmp`/`fcmp`, `br` conditions) → `i32` 0/1.
- `i8`/`i16` SSA values → `i32`, with a canonical narrowing at truncating casts and narrow
  stores. **Prefer the existing `extend8_s`/`extend16_s`/`extend32_s` ops** (lowered on
  both backends, §3b recommendation) over shift-pairs — one fuzzable op, no narrow
  arithmetic added to the TCB.
- Non-byte widths (`i24`, `i48`, …, and `iN` bitfield temporaries) → widen to the
  enclosing `i32`/`i64` with masked stores; reject `i128` for the MVP (clean `Unsupported`).
- `_Atomic char/short`: **no IR form** (the one genuine capability gap, §3b note 2). Lower
  via a 32-bit CAS-loop over the enclosing aligned word, exactly as `WASM.md` plans for
  narrow wasm atomics — *not* by adding `i8`/`i16` to the IR.

### 3c. Totality (§3b) — no UB reaches the IR
The IR is **total**: every op is a defined value or a defined trap. LLVM IR has UB
(`poison`/`undef`, OOB GEP, `unreachable`-after-UB). The translator must **resolve LLVM UB
into defined IR**, the same role chibicc plays for C UB: `undef`/`poison` → a defined
constant (0); `udiv`/`sdiv` by zero is already a defined trap in our IR (§3b); `unreachable`
→ `trap`. We are **untrusted for correctness here**, but I4 totality is enforced by the
verifier + IR semantics regardless (§2a), so a mistake is a wrong-answer bug, not an escape.

---

## 4. LLVM IR → SVM IR mapping (the MVP surface)

The MVP target is the subset the chibicc demos already exercise. Mapping sketch (the
"what lands first" contract; details firm up as code lands):

**Types (DESIGN §3d data model, LP64):**
- `i1/i8/i16/i32` → `i32`; `i64` → `i64`; `iN` (other) → widen or reject (see §3b).
- `float` → `f32`; `double` → `f64`; `x86_fp80`/`fp128` → reject (`long double`=f64, §3d).
- pointers (all address spaces) → `i64` window offset (§3a pointer-as-erasable-i64).
- `[N x T]`, `{...}` aggregates → **by memory** (data-stack slot; SysV/§3d layout via the
  module's `DataLayout`). By-value aggregate args/returns → hidden `sret` pointer (D39),
  exactly the chibicc ABI (HANDOFF §2 "By-value aggregates").
- `<N x T>` vectors → **reject for MVP** (`Unsupported`); SIMD is a later pass mirroring
  the §17/D58 `v128` work `svm-wasm` already did.

**Instructions:**
- arithmetic/bitwise/shift (`add`/`sub`/`mul`/`and`/…/`shl`/`lshr`/`ashr`) → the typed
  `iN.*` ops (wrap semantics, shift mod bitwidth — §3b). `nsw`/`nuw`/`exact` flags: ignore
  (we define wrap; the flags only license UB we don't reproduce).
- `icmp`/`fcmp` → the compare ops (→ `i32` 0/1). float `add`/`sub`/`mul`/`div` → `fN.*`.
- `trunc`/`zext`/`sext`/`fptrunc`/`fpext`/`fptosi`/`sitofp`/`bitcast`/`inttoptr`/`ptrtoint`
  → the §3b conversions (`wrap`/`extend`/`trunc_sat`/`reinterpret`/`ptr.from_int`/`to_int`).
- `getelementptr` → `ptr.add` with the byte offset computed from `DataLayout` (constant
  folded where possible; otherwise index-times-stride arithmetic).
- `load`/`store` → typed `{i32,i64,f32,f64}.load/store` + narrow `load8/16`/`store8/16`
  (the access width drives narrow handling, §3b). `align`/`volatile`: alignment is a hint
  (§3b); `volatile` keeps the access in memory (no promotion — moot post-`mem2reg`).
- `alloca` → bump the data-SP (a data-stack slot), §3d / HANDOFF §3.
- `call` → `call` (direct) / `call_indirect` (function pointer, §3c funcref-index dispatch).
  `musttail`/`tail` → `return_call`/`return_call_indirect` (D6, both backends do true tail
  calls — cf. `svm-wasm` `tests/tailcall.rs`).
- `br`/`switch`/`ret`/`unreachable` → `br`/`br_if`/`br_table`/`return`/`trap` terminators.
  `switch` → `br_table` (dense) or a compare chain (sparse), mirroring chibicc `gen_switch`.
- `phi` → **block parameters** (§1; the headline simplification).
- `select` → `select` (branchless, §3b).
- host calls: LLVM has no capability notion — the C-runtime entry (`write`/`exit`/`malloc`
  over `cap.call`, §3b/§3d) is the same powerbox wiring chibicc uses; the translator binds
  the libc surface to capabilities, it does not invent imports.

**Intrinsics (MVP):** `llvm.memcpy`/`memset`/`memmove` → the loop/bulk lowering (cf.
`svm-wasm` `memory.copy`/`fill`); `llvm.lifetime.*`/`llvm.dbg.*`/`llvm.assume` → drop;
`llvm.trap` → `trap`; `llvm.*.with.overflow`, `llvm.ctlz/cttz/ctpop` → the `clz/ctz/popcnt`
ops or expansions. **Everything else → fail-closed `Unsupported`.**

**Ingest pass pipeline (the "legalize to the subset" step, PNaCl `abi-simplify` analog) —
run out-of-process (DECIDED, §8 Q1/Q2):** `clang -O2 -emit-llvm -fno-vectorize
-fno-slp-vectorize` already runs `mem2reg`+SROA, so the bitcode arrives with scalars
promoted to SSA and only address-taken `alloca`s left — *the two-stack split (§3a) for
free* — while `-fno-*-vectorize` keeps SIMD out of the MVP. For anything more (critical-edge
splitting for φ→block-param, intrinsic/`switch` lowering) shell out to `opt -passes=...`.
We **never run an in-process pass manager and never reimplement `mem2reg`** (PNaCl shipped
`pnacl-opt`; same model). This pipeline is where "pin a frozen subset" is enforced in
practice; the translator then ingests the legalized `.bc` read-only (§6).

---

## 5. The oracle & testing strategy — chibicc as the differential anchor

The user's instinct is right: **chibicc is the oracle for this work.** We already have a
proven, known-good path from C to running IR; the LLVM path consumes the *same C demos*, so
we get a three-lane differential with chibicc as the reference for *our IR shape* and native
`clang` as the reference for *C semantics*.

```
                       demos/*.c  (the existing corpus: clay, jsmn, sha256,
                          │        xxhash, tinfl, perlin, regex, heapgrow, …)
        ┌─────────────────┼──────────────────────────┐
        ▼                 ▼                          ▼
  Lane A: native      Lane B: chibicc → IR       Lane C (NEW): clang -emit-llvm
   cc/clang binary       → interp / JIT             → .bc → svm-llvm → IR
   (C-semantics            (proven; the              → interp / JIT
    ground truth)          IR-shape oracle)
        └─────────────────┴──────────────────────────┘
                 all three must produce identical observable output
```

Why this is the strong setup:
- **Lane B pinpoints translation bugs.** When Lane C diverges from native, we already hold
  a known-good IR (Lane B) for the *same source* — diff the two IR modules to localize the
  bug to the translator vs. something downstream. No other frontend had this luxury.
- **The interp↔JIT differential applies for free.** Lane C's output is just IR, so it rides
  the existing escape-oracle and the interp==JIT checks (HANDOFF §8) — a translator bug that
  produces verifier-valid-but-wrong IR is still caught by interp==JIT==native.
- **Reuse the demo harness.** `assert_demo_matches_cc` (`crates/svm-run/tests/run.rs`) and
  the chibicc invocation (`compile_c` in `svm-run/src/main.rs`) are the template; add a
  `clang -emit-llvm -c -o demo.bc demo.c` → `svm-llvm` → run lane, asserting stdout/exit ==
  native, demo by demo. Start with the **simplest demos first** (a `fib`/`calc`), graduate to
  the real libraries (jsmn, sha256, …) exactly as the chibicc rollout did (HANDOFF §2).
- **Generative fuzzing comes later** — first make the fixed corpus green; the translator's
  own fuzzer (round-trip a generated LLVM module, or reuse `irgen` shapes) is a §8-style
  follow-on, not MVP.

`clang`/`llvm-config` are already required-and-present in CI (the wasm + native-cc lanes use
them), so Lane C adds no new build dependency the harness doesn't already have.

---

## 6. Crate & build plan (proposal — confirm before building)

A new workspace crate **`crates/svm-llvm`**, modeled on `svm-wasm`:

- **Deps:** `svm-ir` (produce the Module). **Dev-deps:** `svm-text`/`svm-verify`/
  `svm-interp`/`svm-jit`/`svm-run` (the differential lanes), mirroring `svm-wasm/Cargo.toml`.
- **LLVM ingest binding — `llvm-ir` 0.11.3, feature `llvm-18` (DECIDED, §8 Q1).** It reads
  the legalized `.bc` via `llvm-sys` and hands the translator an **owned, pure-Rust AST**
  (`enum Instruction`), so the translator is a boring pattern-match-and-emit walk — no
  lifetimes, no `unsafe`, no LLVM context juggling (AGENTS.md "boring obvious"). It is *not*
  asked to run passes (legalization is out-of-process, §4), so its read-only nature is no
  loss. The libLLVM link it pulls in (via `llvm-sys`) is **build/dev-time only** and gated
  so it never enters `svm-jit`/`svm-interp` (the D54 "off the runtime path" rule). Fallbacks
  if it bites: `inkwell` (the maintained, version-tracking wrapper — same C-API limits) →
  then a hand-rolled `.ll` parser over `opt -S` output (zero libLLVM link, but a rot-prone
  parser we'd own). See §8 Q1 for why `llvm-ir` won.
- **Output:** verifier-checked IR `Module`, re-verified in tests (untrusted-frontend, §2a).
- **Tests:** `crates/svm-llvm/tests/` — a `translate.rs` (hand-written `.ll` snippets, the
  unit oracle) and the demo differential lane (Lane C above).

This is a proposal; §6 decisions (binding choice, pin mechanics, CI gating of the libLLVM
dep) are the first things to settle with the maintainer before code lands.

---

## 7. Roadmap (MVP → breadth)

Severity/coverage key mirrors `WASM.md`: **🟢 MVP**, **🟡 fail-closed gap (widen on
demand)**, **🟠 real-program blocker**, **⚪ non-goal/deferred**.

### Milestone 0 — scaffold & first light 🟢 — DONE
- [x] Binding decided: `llvm-ir` 0.11.3 / `llvm-18` (§6, §8 Q1). Legalize out-of-process
      via `clang -O2 -emit-llvm` (+ `opt` as needed) (§4, §8 Q2).
- [x] CI gating: `svm-llvm` **excluded from the workspace** (root `Cargo.toml`, alongside
      `fuzz`/`bench`), so `cargo build/test --workspace` never links libLLVM — confirmed via
      `cargo metadata` (svm-llvm is not a member). The cross-OS runtime matrix is untouched.
- [x] `crates/svm-llvm` skeleton: ingest a `.bc` (`translate_bc_path`), walk functions/blocks,
      emit a `Module`, verify it. Builds + links libLLVM dynamically (see §8 Q4 for prereqs).
- [x] First light green: `clang -O2 -emit-llvm` of `return 42`, an `i32` `add` over params, and
      `i64` arithmetic → translate → verify → interp, matching native semantics; plus a
      fail-closed test (float return → clean `Unsupported`). `tests/translate.rs`, 4 tests.
- [ ] Hand-written `.ll` → IR unit tests (`from_ir_path`) — defer to Milestone 1 alongside the
      richer instruction set; the bitcode lane already covers the M0 surface.

### Milestone 1 — the chibicc-proven scalar+memory+call subset 🟢 (the D54 MVP)
**Slice A (DONE) — control flow + scalar SSA.** Multi-block integer functions on both backends.
- [x] φ → block parameters — done as a general **SSA → block-argument conversion** (liveness-based:
      every value live across a block entry becomes a parameter, φ-results included; each branch
      supplies the args). Critical edges need **no splitting** — args are evaluated in the
      predecessor. Loops/joins/back-edges all covered.
- [x] Integer arith/bitwise/`shl`/`lshr`/`ashr`/`udiv`/`sdiv`/`urem`/`srem`; `icmp` (all 10
      predicates); `select`; `i1`/`i8`/`i16`/`i32`/`i64` `trunc`/`zext`/`sext` (narrow-int collapse
      to `i32`, §3b — sign-extend via the shift pair Cranelift folds); `br`/`br_if`/`return`/
      `unreachable`. Tested interp == JIT == hand-computed on real `clang -O2` output (popcount,
      collatz, classify, …). Non-byte widths (`i33`) are a clean `Unsupported`.

**Slice B (DONE) — the §3d data stack (scalar memory).** Address-taken locals via `alloca`.
- [x] `alloca` → a window data-stack frame slot at an `sp`-relative offset (natural-aligned;
      frame size 16-aligned). Dynamic (non-constant count) `alloca` is a clean `Unsupported`.
- [x] `load`/`store` incl. narrow widths (`i8`/`i16` → the `i32`-container load/store ops; narrow
      loads zero-extend, signedness via the following `sext`/`zext`, §3b). Pointers are `i64`.
- [x] `getelementptr` → `i64` address arithmetic: `base + Σ idx·stride` (pointee + array element
      strides from the type sizes), constant indices folded, variable indices `mul`+`add` (index
      sign-extended to `i64`). Struct/vector GEP is a later slice.
- [x] `undef`/`poison`/`null` → defined `0` (totality, §3c); `llvm.lifetime`/`dbg`/`assume`
      intrinsics dropped. Tested on a `clang -O2` stack-array sum/reverse (GEP + store/load over the
      frame), interp == JIT == hand-computed.

**Slice C (DONE) — calls + the threaded data-SP.** Direct calls; per-activation frames.
- [x] Every function takes a leading `sp` parameter (§3d), threaded as block-local index 0 of every
      block (like chibicc's `v0`); each branch passes it through. A direct `call` resolves the
      target by name → IR function index, and passes the callee `sp + frame_size` (so frames never
      overlap; recursion is sound), then the mapped arguments; the result threads back. Tested on
      `clang -O2` recursive `fib` and a `noinline` cross-function call, interp == JIT == hand-computed.

**Slice D (DONE) — `switch`.**
- [x] `switch` → `br_table` (§3b): the `i32` operand is biased by the minimum case value, then
      indexes a target vector spanning `[min, max]` with gaps filled by the default edge; each edge
      carries its destination's block args (computed once per distinct target). Too-sparse switches
      (span > 4096) and i64-operand switches are a clean `Unsupported`. Tested on a dense switch and
      the `even`/`odd` mutual recursion `-O2` lowers onto a switch-loop.

**Slice E (DONE) — global variables + the data-stack guard.**
- [x] Globals laid out **low** in the window (`[DATA_BASE, globals_end)`), each natural-aligned.
      Emitted as IR `data` segments — constants **read-only** (D40), BSS/zero globals just reserve
      space in the zero-init window. A `@global` reference resolves to its window address (a
      constant `i64`); int/array/string/zero initializers serialize to little-endian bytes. Tested
      on a const lookup table, a mutable counter, indexed string reads, and the gapped switch
      (a global jump table).
- [x] **Guard:** the data stack now starts **just above** the globals (`entry_sp = align(globals_end)`)
      and grows up toward the window's mapped top; `mapped` is sized for the globals + a 1 MiB stack
      reserve, and the runtime leaves a faulting guard beyond `mapped` (reserved > mapped). So a
      stack overflow **faults** (§5) instead of corrupting the globals below — tested by a deep
      recursion with a 32 KiB frame that traps on both backends (a shallow call returns).
- [x] **API:** `translate`/`translate_bc_path` now return `Translated { module, entry_sp }` — the
      host/driver invokes the entry with `entry_sp` as its first (`sp`) argument.

**Slice F (DONE) — floats.**
- [x] `f32`/`f64` arithmetic (`fadd`/`fsub`/`fmul`/`fdiv`/`fneg`), `fcmp` (ordered/unordered collapse
      to the SVM op — NaN corner is a documented fidelity gap), `select`, the int↔float conversions
      (`sitofp`/`uitofp`/`fptosi`/`fptoui`, float→int **saturating** per §3b), `fpext`/`fptrunc`,
      `bitcast`, and the common float math intrinsics (`fmuladd`/`fma` lowered **unfused**;
      `sqrt`/`fabs`/`floor`/`ceil`/`trunc`/`rint`/`copysign`/`min`/`maxnum`) lowered inline. Float
      constants and `f32`/`f64` params/results in the slot ABI. Tested on `clang -O2` double
      arithmetic (incl. the `fmuladd` contraction), compares, int↔float, promote/demote, `fabs`/`floor`.

**Slice G (DONE) — indirect calls (funcref §3c).**
- [x] Taking a function's address → its funcref index (`ref.func`, the function-table index)
      widened to the `i64` pointer rep. An indirect `call` (callee is a function-pointer value)
      truncates it back to the `i32` funcref and lowers to `call_indirect <sig>` — the runtime masks
      the index and checks the type-id (§3c). The signature is the callee's function type plus the
      prepended data-SP, so it matches the callee's IR signature. Tested on a `noinline` pointer-
      returning `pick` whose result `run` calls indirectly (a genuine `-O2` `call_indirect`).

**Slice H (DONE) — aggregates (struct memory).**
- [x] Struct layout (x86-64-SysV: natural field alignment + tail padding to the struct's alignment;
      named structs resolved via the type table), **struct GEP** (a constant field index → the
      field's byte offset, descending into the field type — composes with array indices), struct
      `alloca`s (struct-sized, struct-aligned frame slots), and struct global initializers serialized
      with field padding (read-only D40). Tested on a global struct read field-by-field, an
      array-of-structs (`arr[i].field`), and a `volatile` stack struct (store/load via field GEP).
      Covers structs via pointers/locals/globals — **not** the by-value pass/return ABI.

**Slice I (DONE) — memory intrinsics.**
- [x] `llvm.memcpy`/`memmove`/`memset` (constant length) lower to inline **chunked load/stores**
      (widest-first 8/4/2/1, the plan `svm-wasm` uses for `memory.copy`/`fill`). Copies
      **load-all-then-store-all** (overlap-safe → `memcpy` and `memmove` share a path); `memset`
      replicates the fill byte across an `i64` (`val·0x01010101_01010101`) and stores it chunk-wide.
      Variable-length / `> 4 KiB` is a clean `Unsupported` (needs a runtime loop). Also **page-aligned
      the data stack** above the globals (16 KiB) so a stack write never faults on a read-only
      global's page (D40 protects RO segments page-granularly — the bug a struct-`memcpy`-into-stack
      test surfaced). Tested on a struct `memcpy` from a const global and a `memset` fill.

**Slice J (DONE) — by-value aggregate args/returns (`sret`/`byval`).**
- [x] Works with **no dedicated translator code** — the anticipated-gnarly slice turned out free,
      because clang does the x86-64-SysV register-classification *in the IR*: a small struct is
      coerced to scalar register(s) (`{i32,i32}`→`i64`, three-int→`(i64,i32)`, SSE→`double`s) and the
      body packs/unpacks via a stack slot; a large struct passes via a `byval`/`sret` pointer (the
      caller `alloca`s + `memcpy`s + passes the pointer). So slices A–I (scalar params, memory,
      calls, struct GEP, **memcpy** — the actual prerequisite) already cover it. Tested through calls
      so the call-site coercion is exercised: small `byval`/return, two-eightbyte `(i64,i32)`, an SSE
      `(double,double)`, and a large `mkBig` (`sret`) + `sumBig` (`byval`).

**Slice K (DONE) — relocations (pointer-valued global initializers).**
- [x] A global initializer holding a function pointer, `&other_global`, or arithmetic over those
      resolves via a constexpr evaluator (`const_eval`): `GlobalReference` → a data global's address
      or a function's funcref index, plus `ptrtoint`/`inttoptr`/`bitcast`/`trunc`/`sub`/`add`/`mul`.
      The globals layout is **two-phase** — assign every global an address (sizing via `const_size`,
      which matches the serialized length), then serialize each initializer (so a forward/backward
      reference to another global resolves). Tested on a function-pointer table (`{inc,dec}`, called
      indirectly) and a struct with a string-pointer member. (A regression caught here: phase-A
      sizing must use the serialized length, not `type_size(g.ty)`, or the window mis-sizes.)
      *Deferred: `llvm.load.relative` (clang's relative-offset string tables) and GEP-constexprs.*

**Slice L (DONE) — libm math calls.**
- [x] A call to an *external* `sqrt`/`fabs`/`floor`/`ceil`/`trunc`/`rint`/`nearbyint`/`copysign`/
      `fmin`/`fmax` (and the `…f` f32 variants) lowers to the matching SVM float op inline —
      `lower_libm_call`, gated on the name not being a guest-defined function. `round`
      (half-away-from-zero) and transcendentals (`sin`/`cos`/`exp`/`log`/`pow`) have no SVM op, so
      they stay calls (`Unsupported` for now). Tested on `sqrt` and `fmin`.

**Slice M (DONE) — integer min/max + bit intrinsics.**
- [x] `llvm.smax`/`smin`/`umax`/`umin` → `icmp`+`select`; `llvm.ctlz`/`cttz`/`ctpop` →
      `clz`/`ctz`/`popcnt` (the trailing `is_*_poison` `i1` ignored — SVM defines the zero case);
      `llvm.abs` → `select(x<0, -x, x)` (`lower_int_intrinsic`). Tested on `smax` (a `?:` max),
      `ctlz`, `ctpop`, and an `abs`.

**Slice N (DONE) — the powerbox on-ramp (libc → capabilities, "Lane C").**
- [x] A program that does I/O gets a **synthesized powerbox entry** (`_start`, function 0):
      `(stdout, stdin, exit)` `i32` handles (§3e, `is_powerbox_entry`), stored into the **handle
      stash** — the reserved low window `[0, DATA_BASE)`, **page-isolated** from the globals (which
      now start a page up, `STACK_PAGE`, so a read-only global's D40 page-protection never catches
      `_start`'s handle stores). `_start` then calls `main(entry_sp)` and returns its exit code.
- [x] An external libc call bound to a host capability lowers to `Inst::CallImport "<name>"` the
      embedder resolves at load (§7, `default_cap_resolver`): `write`/`read` → `Stream`
      (`(i64 buf, i64 len) -> (i64)`, the POSIX `fd` dropped — the handle selects the endpoint),
      `exit` → `Exit`. The handle is reloaded from the stash at each call site, so it threads through
      arbitrary call depth with no viral parameter. A guest-*defined* function of the same name
      shadows the binding (mirrors the libm rule). `collect_cap_imports` builds the import table;
      `synth_start` builds the entry; `Module.imports` carries them to `resolve_capability_imports`.
- [x] **End-to-end vs native:** `check_powerbox_vs_native` translates → resolves §7 imports →
      verifies → runs through the reference powerbox, asserting **stdout *and* exit code** match the
      native `cc` build. Tests: a `write`+return hello, an `exit(code)`, a stdin→stdout echo loop,
      and a computed stack-buffer string (composing the data frame + I/O).

**Remaining slices.**
- [ ] `llvm.load.relative` (relative-offset string tables); transcendental math (needs a guest libm);
      `llvm.bswap`/`bitreverse`/`fshl`.
- [ ] More libc surface: `malloc`/`free` (→ the `Memory` capability + a guest allocator), `printf`
      (formatting), `argc`/`argv` `main`. The §7 import mechanism (slice N) is the hard part; these
      are additional bindings on it.
- [ ] **Goal: every existing C demo runs byte-identical to native `clang` on Lane C**
      (the same corpus chibicc passes — clay, jsmn, sha256, xxhash, tinfl, perlin, regex,
      heapgrow). This is the D54 "matches native clang" exit criterion.

### Milestone 2 — beyond chibicc's C subset 🟡
- [ ] Tail calls (`musttail` → `return_call`), if any corpus needs it (likely near-free).
- [ ] Real Rust/C++ *without* EH/unwinding: `rustc --emit=llvm-bc` of a `no_std`/panic=abort
      crate; a C++ TU compiled `-fno-exceptions -fno-rtti`. The breadth proof.
- [ ] Narrow-atomic CAS-loop emulation (§3b note 2), on demand.

### Deferred / hard (name them, don't hide them — DESIGN §20) ⚪
- [ ] **C++ exceptions / unwinding** — `invoke`/`landingpad`/`resume` + `.eh_frame` unwind
      tables (the §18 open item). Lower onto §6 stack-switching; perf tax + ABI change. Low
      ROI until a real workload needs it (mirrors `WASM.md`'s EH stance).
- [ ] **`setjmp`/`longjmp`** — onto §6 stack-switching, same machinery as EH.
- [ ] **SIMD** (`<N x T>` vectors) — a later pass mirroring §17/D58 `v128` (the proven
      5-step pattern `svm-wasm` used). Reject cleanly until then.
- [ ] **Full intrinsic coverage** — expand the table in §4 as real programs demand.
- [ ] **`i128`, `x86_fp80`/`fp128`, vector-of-pointers, scalable vectors** — reject.
- [ ] **GC / managed languages** — permanent non-goal (same as wasm-GC in `WASM.md`):
      contradicts the linear-memory + small-TCB thesis. C/C++/Rust/Swift/Zig is the niche.

---

## 8. Decisions & open questions

**Q1 — Ingest binding (DECIDED): `llvm-ir` 0.11.3, feature `llvm-18`.** The decision splits
into two independent sub-questions, and separating them is what makes it clear:
*(a) how to legalize* (Q2) and *(b) how to read the module*. The binding question is only
(b). Because legalization is out-of-process (Q2), the binding never needs an in-process pass
manager — which removes the main reason to reach for the full `inkwell`/`llvm-sys` API. That
leaves "what's the nicest way to *read* a module": `llvm-ir`'s **owned, pure-Rust AST** wins
on translator ergonomics (pattern-match-and-emit, no lifetimes/`unsafe`) — the boring,
obvious code AGENTS.md asks for. Verified for this repo: `llvm-ir` 0.11.3 supports LLVM
9–19, so the **`llvm-18` feature matches our pin**; its only representation gaps are debug
metadata (we drop `llvm.dbg.*` regardless) and a few C-API-only getters (which constrain
`inkwell` equally — both are LLVM-C-API-bound), neither of which touches the scalar+memory+
call MVP. No mature *pure-Rust* bitcode reader exists, so any programmatic read links
libLLVM; D54 sanctions that as a build/dev-time dep (Q4 keeps it off the runtime path).
**Fallback order if `llvm-ir` bites:** `inkwell` (maintained, version-tracking wrapper) →
hand-rolled `.ll` parser over `opt -S` (zero libLLVM link, but a rot-prone parser we own).

**Q2 — Legalization & opt level (DECIDED): out-of-process, `clang -O2 -emit-llvm
-fno-vectorize -fno-slp-vectorize`** (+ `opt -passes=...` for any extra legalization). `-O2`
gives `mem2reg`/SROA (the two-stack split for free, §3a); `-fno-*-vectorize` keeps SIMD out
of the MVP. We never run an in-process pass manager or reimplement `mem2reg` (the PNaCl
`pnacl-opt` model). See §4 "Ingest pass pipeline".

**Q3 — Pin mechanics (open):** how strictly to reject off-version bitcode, and where the
frozen-subset allow-list lives (a single `unsup(...)`-style chokepoint, like `svm-wasm`).

**Q4 — CI gating (DONE for the build story; the CI yaml lane is the remaining piece):**
`svm-llvm` is **excluded from the workspace** (root `Cargo.toml`, with `fuzz`/`bench`), so the
default `cargo build/test --workspace` never resolves or links libLLVM and the cross-OS
runtime matrix (`svm-jit`/`svm-interp` on Linux+macOS+Windows) is untouched (D54 off-the-
runtime-path). The opt-in lane runs `cd crates/svm-llvm && cargo test`. **Build prerequisites
found at first light (document these for the CI lane):**
- **`llvm-18-dev`** (headers + the `.so`), not just `llvm-18`/`libllvm18` (runtime only). The
  runtime package ships `libLLVM.so` but no `llvm-c/*.h`, which `llvm-sys`'s build script needs.
- **Dynamic linking**: distros ship `libLLVM.so` without the static `.a`s (no `libPolly.a`),
  and `llvm-sys` defaults to static. We depend on `llvm-sys` directly with feature
  **`prefer-dynamic`** (feature-unifies onto the `llvm-sys` `llvm-ir` pulls in) so it links the
  dylib. `clang`/`llvm-config` 18 must be on `PATH` (they are in this container + the existing
  wasm/cc CI lanes).

**Q5 — remaining CI yaml (open):** add the Linux-only `svm-llvm` job to `ci.yml` (install
`llvm-18-dev`, `cd crates/svm-llvm && cargo test`); a maintainer one-liner like the §10 miri/
ASan items in HANDOFF.

---

## 9. Code map
- Translator + frozen-subset chokepoint: `crates/svm-llvm/src/lib.rs` — `translate`/
  `translate_bc_path`, `val_type`/`operand_int_ty` (the §3b narrow-int collapse), `BlockCtx`
  (block-local SSA numbering, §3a), and the `unsup(...)` fail-closed chokepoint.
- First-light differential: `crates/svm-llvm/tests/translate.rs` — `compile_to_bc` runs the
  pinned `clang -O2 -emit-llvm` pipeline; `run` does translate→verify→interp.
- Crate config + build prereqs: `crates/svm-llvm/Cargo.toml` (the `llvm-ir`/`llvm-sys`
  `prefer-dynamic` deps); workspace exclusion in the root `Cargo.toml`.
- The oracle to diff against (Lane B): chibicc `frontend/chibicc/codegen_ir.c` + the running
  demos in `demos/` — wired in at Milestone 1.
