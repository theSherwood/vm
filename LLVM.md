# LLVM-bitcode → IR on-ramp (`crates/svm-llvm`) — design & tracking

The plan, the constraints, and the prioritized work for the **third frontend**: an
**ahead-of-time LLVM-bitcode → SVM-IR translator**. This is the big *breadth* play
(D54) — one component buys every LLVM language (C, C++, Rust, Swift, Zig…) as a guest.

This file is the working tracker for the on-ramp, the analog of `WASM.md` for the wasm
bridge. Like that doc, fold completed sections into `DESIGN.md` and drop this file once
the actionable gaps close (the repo convention, cf. the former `WASM.md`/`SCHEDULING.md`).

> **Milestone 1 (the D54 exit criterion) is complete and folded into `DESIGN.md` §20a** — all
> eight chibicc corpus libraries run byte-identical to native `clang`. This file is retained as
> the tracker for the **remaining general-C breadth** (varargs `printf`, `realloc`, wider SIMD,
> libm, …), now pursued **demo-driven** — see "Pending work — demo-driven plan" below. The slice
> log (A–V) is kept as the implementation record until the file is dropped.

**Status: Milestone 1 slices A–V done — the **D54 exit criterion is met**: all **eight corpus
libraries run byte-identical to native `clang`** — B-Con's **SHA-256**, **xxHash**, **stb_perlin**,
**tiny-regex-c**, **jsmn**, **heapgrow**, **miniz/tinfl**, and **clay** (`demo_*_vs_native`, 64 tests).
Slice U fixed a **narrow-signed `icmp`** bug (a signed compare of a zero-extended `i8`/`i16` must
sign-extend first, §3b), landing tinfl. Slice V scalarizes **2-lane 32-bit vectors**
(`<2 x float>`/`<2 x i32>`) to a packed `i64` — they flow through `phi`/`call`/`ret`/`load`/`store` as
an ordinary `i64`, and only the vector ops (`extractelement`/`insertelement`/lane-wise
`fadd`/`shufflevector`) unpack/repack — landing clay (the 8th demo).
A **kitchen-sink capstone** exercises everything at once (structs by-value, a function-pointer table,
floats+libm, recursion, loops, an array `memcpy`, a global array, `switch`, bit intrinsics) and
matches **native `cc`** end to end. **Slice N** binds the raw I/O primitives (`write`/`read` →
`Stream`, `exit` → `Exit`) via §7 named imports + a synthesized powerbox `_start`; **slice O** adds
the non-varargs **stdio** output family (`puts`/`putchar`/`putc`/`fputc`/`fwrite`/`fputs`/`fflush`,
and `clang`'s `printf("…\n")`→`puts` / `printf("%c")`→`putc` lowering); **slice P** adds funnel-shift
rotates (`llvm.fshl`/`fshr` → `rotl`/`rotr`) and **synthesized runtime mem-loop helpers**
(`__svm_memset`/`__svm_memcpy` — the first multi-block helper, for a variable-length `memset`/`memcpy`).
The **demo-driven breadth plan (demos 1–6) is now complete** — `hexdump` (varargs `printf`, slice W),
`sortvec` (`realloc` + signed `%d`, X), `mat4` (`<4 x float>` SIMD, Y), `crc32` (`llvm.bswap`, Z),
`lineedit` (overlap-safe `memmove`, AA), and `raytrace` (transcendental libm bundled as guest code,
AB) all run byte-identical to native.**
`crates/svm-llvm` does the **SSA → block-argument
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
(`smax`/`ctlz`/`popcount`), funnel-shift rotates (`fshl`/`fshr`), and a variable-length `memset`
loop, `ptr`↔`int`, `freeze`, a constexpr GEP, RO/writable page isolation, `llvm.load.relative`, a
`vm_map`-growing `malloc`/`calloc`/`free`, multi-value struct returns, narrow-signed `icmp`, 2-lane
vector scalarization — run **interp == JIT == hand-computed** (64 tests, incl. a kitchen-sink program
checked against native `cc`, `write`/`exit`/`read`-echo and `puts`/`printf`/`putchar`/`fwrite`/`fputs`
powerbox programs, and **all eight corpus demos** — SHA-256 / xxHash / perlin / regex / jsmn /
heapgrow / tinfl / clay — checked against native stdout). The D54 corpus exit criterion is met;
varargs `printf` remains as general-C breadth (no corpus demo needs it). Section numbers like "§3d"
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
- [x] **Dense** `switch` → `br_table` (§3b): the operand is biased by the minimum case value, then
      indexes a target vector spanning `[min, max]` with gaps filled by the default edge; each edge
      carries its destination's block args (computed once per distinct target). Tested on a dense
      switch and the `even`/`odd` mutual recursion `-O2` lowers onto a switch-loop.
- [x] **Sparse** switches (span > 4096 — i64 niche-optimized enum discriminants, real parsers
      dispatching on 4-byte fourccs) → an equality **compare chain** of synthetic blocks
      (`lower_sparse_switch`, threading live-ins/φ args via `aux_blocks`), since a dense `br_table`
      would be astronomically large. Tested by `switch_sparse_*`; also lands stb_image's PNG chunk
      loop (slice BH).

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

**Slice O (DONE) — the stdio output surface.**
- [x] The non-varargs libc output family funnels to `Stream.write` on stdout (`lower_io_call`):
      `puts` (the literal's bytes + a newline — length from the string-literal global, no runtime
      strlen), `putchar`/`putc`/`fputc` (one byte staged through the stash scratch `[12,16)`),
      `fwrite`/`fputs` (a `size×nmemb` slice / a string), `fflush` (a no-op — unbuffered `Stream`).
      The libc `FILE*` stream argument is ignored (the handle is the endpoint). Several libc names
      share one `write` import (`collect_cap_imports` now keys the table by *import* name).
- [x] `clang -O2` rewrites `printf("…\n")` → `puts` and `printf("%c",c)` → `putc`, so format-free
      `printf` rides this path with no varargs. Result fidelity: the *stdout bytes* are exact; the
      return values are best-effort (`putc`→char, `fwrite`→`nmemb`, `puts`/`fputs`/`fflush`→`0`).
- [x] Tests (`check_powerbox_vs_native`): `puts`, two-line `printf` (→`puts`), a `putchar` range
      loop, `fwrite`+`fputs` mixed, and stdio composed with `exit(42)`.

**Slice P (DONE) — funnel shifts + runtime mem-loop helpers (first corpus demo).** Data-driven from
driving B-Con's SHA-256 (`sha_demo.c`) through the on-ramp and closing the two gaps it hit.
- [x] `llvm.fshl`/`fshr` → `rotl`/`rotr` when the two value operands are identical (the rotate idiom
      clang emits for `(x<<n)|(x>>(w-n))`, e.g. SHA-256's `ROTRIGHT`/`ROTLEFT`); `rotl`/`rotr` mask
      the count mod width, so no shift-by-`w` edge case. A general (non-rotate) funnel shift is a
      clean `Unsupported` (`lower_int_intrinsic`).
- [x] A variable-length — or oversized-constant — `llvm.memset`/`memcpy` calls a **synthesized
      runtime loop helper** (`synth_memset`/`synth_memcpy`: a 4-block counted byte loop threading
      `(ptr, …, i)` as block params — the first hand-built multi-block CFG / "mini-libc"), instead of
      erroring. clang's loop-idiom recognizer turns hand-written `mem*` loops *into* these intrinsics
      with a runtime length, so most real code needs it. Helper indices sit after the defined
      functions, fixed before lowering call sites. (Variable-length `memmove` later got its own
      direction-aware `synth_memmove` — slice AA.)
- [x] **Demo:** `demo_sha256_vs_native` runs the whole SHA-256 library (multi-function calls, the
      data stack, a const global table, rotates, the `memset` loop helper, `write`) — digests
      byte-identical to native `clang`. Plus focused `funnel_shift_rotate` and
      `variable_length_memset_loop` unit checks.

**Slice Q (DONE) — more corpus demos + the gaps they revealed.** Data-driven: drove xxHash, perlin,
and tiny-regex-c through the on-ramp. xxHash + perlin needed *no* new code (slices A–P); regex hit a
cluster of small gaps, all now closed.
- [x] `ptrtoint`/`inttoptr` (instruction form): pointers are an `i64` window offset, so this is a
      width adjust (identity at `i64`, `wrap`/`zext` for narrow), never a reinterpret.
- [x] `freeze`: an identity — the IR is total (`undef`/`poison` → defined 0, no poison propagates).
- [x] **Constexpr GEP** (`&".."[k]`, `&g.f`): an interior pointer into a constant aggregate, folded
      to base address + type-walked constant offset (`const_gep_offset`, mirroring `translate_gep`);
      handled both as an operand and inside an initializer.
- [x] **Read-only globals are page-isolated from writable ones** (`globals_layout`): lay writable +
      BSS globals first, page-align, then the read-only ones — so a `const` never shares a
      D40-protected page with a writable/BSS global (a write to which would otherwise fault). This
      was a latent layout bug, now exercised by regex's `static` arrays beside string literals.
- [x] **Demos:** `demo_xxhash_vs_native`, `demo_perlin_vs_native`, `demo_regex_vs_native`, plus a
      focused `ro_and_writable_global_page_isolation` unit check.

**Slice R (DONE) — `llvm.load.relative` (lands jsmn).** clang lowers a constant-returning `switch`
(jsmn's token-type → name) into a **relative lookup table**: `@reltable = [i32 (&str − &reltable)…]`,
and `llvm.load.relative.i64(P, off)` returns `P + sext_i32(*(i32*)(P + off))` — the absolute target.
- [x] `lower_load_relative`: `add` the offset to the base, `load.i32`, `sext` to i64, `add` to the
      base. The table initializer (`trunc(sub(ptrtoint(@str), ptrtoint(@table)))`) already folds via
      `const_eval` (`Trunc`/`Sub`/`PtrToInt`), so no new initializer support was needed.
- [x] **Demo:** `demo_jsmn_vs_native` — a zero-allocation JSON parser, parsing an embedded document
      into a fixed token array and printing each token's type/size/text, byte-identical to native.

**Slice S (DONE) — `malloc`/heap (the §1a sparse address space; lands heapgrow).** `malloc`/`calloc`
lower to a synthesized **bump allocator** (`synth_malloc` → `__svm_malloc(size)`) that grows the heap
into the window's reserved tail by `vm_map`-committing pages on demand via the `Memory` capability.
- [x] A program that allocates gets a **4-handle `_start`** (`stdout, stdin, exit, memory`); the
      powerbox grants `Memory` for a 4-param entry. `_start` stashes the handle and seeds the heap
      (`HEAP_BRK`/`HEAP_TOP` = the window's mapped boundary, the first reserved page).
- [x] `__svm_malloc`: a 3-block CFG — align the request to 16; if it crosses the committed boundary,
      `CallImport "vm_map"(top, page_up(new) − top, RW)` (resolved to `Memory.map`) and advance the
      boundary; publish the new break and return the old. `free` is a no-op; the heap never reuses, so
      freshly-committed (`vm_map`-zeroed) pages make `calloc` ≡ `malloc`. `realloc` stays `Unsupported`.
- [x] **Demo:** `demo_heapgrow_vs_native` — a guest allocating eight 128 KiB blocks (~16× its initial
      window), growing on demand via the `Memory` cap, byte-identical to native. Plus a focused
      `heap_malloc_calloc_free` check (a growth-forcing `malloc` + a zero-reading `calloc`).

**Slice T (DONE) — multi-value struct returns.** A small by-value struct returned in registers (clang
coerces it to e.g. `{ i64, i64 }` / `{ i64, ptr }`, as clay's `*Array_Allocate_Arena` and any C
returning a 2-field struct) maps to an SVM **multi-result** function (§3a).
- [x] `result_types` flattens a small struct return to its scalar fields; a multi-result `call`
      records the aggregate field-wise (`BlockCtx.agg`, value-id → field indices) via `push_multi`;
      `insertvalue`/`extractvalue` build/read it; `ret` returns the fields. Aggregates are assumed not
      to cross block boundaries (clang's register-coercion produces+consumes them in one block; if one
      did, `agg_of` returns `None` → a clean error). Plus `llvm.experimental.noalias.scope.decl`
      dropped. Tested: `multi_value_struct_return` (a `{i64,i64}` return round-trip vs native).

**Slice U (DONE) — narrow-signed `icmp` (lands tinfl).** A signed `icmp` on a **narrow** (`i8`/`i16`)
operand now sign-extends the operand to `i32` first (`emit_ext` with the predicate's signedness),
fixing the §3b hazard where a zero-extended narrow value (e.g. an `i16` load of a *signed*
`mz_int16`) made `< 0` always false.
- [x] Root-caused from tinfl's runtime fault: the Huffman slow-path `do { temp = tree[~temp + bit]; }
      while (temp < 0)` compared a zero-extended `i16` table entry against 0 (always false), so `~temp`
      produced index `-1` → `zext` `0xFFFFFFFF` → `×2` → a bit-33 corrupt back-reference pointer.
- [x] Lands **tinfl** (`demo_tinfl_vs_native`) — miniz inflate, byte-identical to native. Plus a
      focused `narrow_signed_compare` regression (summing negative signed-`short` table entries).

**Slice V (DONE) — 2-lane vectors (`<2 x float>`/`<2 x i32>`); lands clay → the full corpus.** A 2-lane
32-bit vector (clang's `Clay_Vector2`/2D-point coercion) is **scalarized to a packed `i64`** (lane 0 =
bits 0–31, lane 1 = 32–63 — its little-endian image).
- [x] `vec2_lane_ty` recognizes `<2 x float>`/`<2 x i32>`; `val_type`/`type_size`/`load_op`/`store_op`
      map them to `i64`, so the vector flows through `phi`/`call`/`ret`/`load`/`store`/block-params as
      a plain `i64` — *no* liveness/block-param changes. Only the ops unpack/repack: `vec_lane`/
      `vec_pack` (lane-type-aware) drive `extractelement`/`insertelement`, lane-wise
      `fadd`/`fsub`/`fmul`/`fdiv` (`fp_binop`), constant-mask `shufflevector`, and vector constants/
      `zeroinitializer`/`undef`; a `bitcast` between 2-lane vectors is a no-op (same packed `i64`).
- [x] Lands **clay** (`demo_clay_vs_native`) byte-identical to native — UI layout printing render
      commands. Plus a focused `vec2_float_struct` check (a `{float,float}` add coerced to `<2 x float>`).

**Goal — MET: every corpus demo runs byte-identical to native `clang` on Lane C** ✅
(sha256 ✅, xxhash ✅, perlin ✅, regex ✅, jsmn ✅, heapgrow ✅, tinfl ✅, clay ✅). **8 of 8** — the D54
"matches native clang" exit criterion.

## Pending work — demo-driven plan

The corpus is done; the remaining work is **general-C breadth**, pursued the same way that worked
for the corpus: pick a small **real end-to-end demo** (`crates/svm-run/demos/`), drive it through
`clang -O2 → translate → verify → run` vs native, and close exactly the gaps it reveals. Each demo
below is a whole-program, `write`-output C program (its own minimal libc, like the corpus demos) so
it stays a clean differential against a native `cc` build. Ordered by value (printf first — it is
the dominant general-C gap).

| # | Demo (proposed) | Drives (pending item) | Also exercises |
|---|---|---|---|
| 1 ✅ | **`hexdump`** — read stdin, print `%08lx  %02x ×16  \|ascii\|` rows (`demos/hexdump`, slice W) | **varargs `printf`** (unsigned `%u`/`%x`, width, `0`-pad, `l`) — DONE, byte-identical to native | `read`, loops |
| 2 ✅ | **`sortvec`** — `realloc`-doubling int vector + insertion sort, print `%d` 10/line (`demos/sortvec`, slice X) | **`realloc`** (header-sized grow-and-copy) + signed `printf` (`%d`) — DONE, byte-identical to native | `malloc` |
| 3 ✅ | **`mat4`** — 4×4 matrix × vec4 affine transform, print rows (`demos/mat4`, slice Y) | **128-bit SIMD** (`<4 x float>` → native `v128`) — DONE, byte-identical to native | floats, `printf` |
| 4 ✅ | **`crc32`** — CRC-32 over stdin + a big-endian `u32` reader (`demos/crc32`, slice Z) | **`llvm.bswap`** (inline byte reversal) — DONE, byte-identical to native | shifts, `printf` |
| 5 ✅ | **`lineedit`** — read a line, wrap in `[...]` (right shift) + delete middle char (left shift) (`demos/lineedit`, slice AA) | **overlapping `memmove`** (direction-aware runtime loop) — DONE, byte-identical to native | arrays, `read` |
| 6 ✅ | **`raytrace`** — ASCII sphere raytracer: `sqrt` intersection + diffuse/sinusoidal/exp shading (`demos/raytrace`, slice AB) | **transcendental libm** — `sqrt`/`floor` lower to SVM ops; `sin`/`exp` bundled as **guest `libm`** (poly approximations) — DONE, byte-identical to native | floats, `write` |

Notes:
- **`printf` runs in the guest** (per the capability model): a guest-side format engine parses the
  (constant) format string at translate time and lowers each conversion to int→string / float→string
  helpers → `Stream.write`; only the bytes cross the boundary. `%f` pulls in float formatting (defer
  to demo 3/6 if demo 1 stays integer/hex). Non-constant format strings stay `Unsupported`.
- **transcendentals/libm**: a **guest** `libm` (guest code, not a host math capability) — keeps math
  in the sandbox. `sqrt` lowers to the SVM op (slice F); `exp`/`log`/`pow` now have fdlibm
  implementations in `crates/svm-run/demos/libm/libm.c` (a guest def shadows the on-ramp's trap stub);
  `sin`/`cos`/… are the remaining additions. See the "Transcendentals → a guest `libm`" bullet above.
- **`argc`/`argv`**: **DONE** — see *Slice BE* below. `envp`/`getenv`: **DONE** — see *Slice BF*.

**Slice W (DONE) — varargs `printf`, the guest-side format engine (lands `hexdump`).** A
`printf(fmt, …)` with a **constant** format string is parsed at translate time (`parse_format`):
literal runs are written straight from the format global; each conversion lowers to the synthesized
**`__svm_utoa`** (unsigned int → ASCII, a counted divide loop) plus width/zero-padding (a constant
pre-fill of the scratch buffer `[FMT_BUF, FMT_BUF_END)`, then a `max(len,width)` write window) →
`Stream.write`. Covers unsigned `%u`/`%x`, `%c`, `%%`, field width, the `0` flag, and length
modifiers (the LLVM arg carries the real width — `%lx` ⇒ an `i64` arg). All formatting is **guest
code**; only the bytes cross the boundary. Tests: `demo_hexdump_vs_native` (a `hexdump -C` clone, vs
native, with stdin) + `printf_unsigned_formats` (mixed widths/pads/`%lx`/`%c`/`%%`).
- *Deferred:* `%s` (runtime strlen), `%f`/`%g`/`%e` (float formatting), precision/`*`/`-`/`+`/space/`#`,
  non-constant format strings. **(`%s`, the flags, and precision landed — slices BA–BD below; float
  landed later via the exact bignum `dtoa` family — `__svm_dtoa_{fix_big,sci,gen}` — incl. the float
  `0`/`#` flags; only `*` and non-constant formats remain deferred.)**

**Slices BA–BD (DONE) — the `printf` breadth batch (no new demo; the format engine widened, each
checked byte-for-byte vs native `printf`).**
- **BA — `%s`** (runtime `strlen`): a synthesized `__svm_strlen` (counted forward scan) + a
  right-justified field-width pad; the string bytes are written straight from the argument pointer.
  Test: `printf_string_formats`.
- **BB — flags `-`/`+`/space/`#` + zero-padded signed `%d`** (the previously fail-closed sign+pad
  combo): the int formatter is rebuilt as a flag-aware `emit_printf_int_field` — left-justify, forced
  sign, the `0x` alternate-form prefix (suppressed for zero), with the justify/pad *layout* decided at
  translate time and only digit/pad lengths runtime. Test: `printf_flag_formats`.
- **BD — precision**: `.N` parsing; integer **min-digit** precision (`%.Nd`/`%.Nx` — zero-extend the
  digit region; precision disables the `0` flag per C; `%.0` of `0` prints no digits) and string
  **truncating** precision (`%.Ns` → `min(strlen, N)`). Shared `pf_*` helpers (`pf_sign_prefix`/
  `pf_field_layout`/`pf_prefill_pad`). Test: `printf_precision_formats`.
- **Float (`%f`/`%e`/`%g`) — deliberately fail-closed `Unsupported`.** A first `%f` cut (exact-looking
  `f*=10; d=⌊f⌋; f-=d` digit extraction) was **reverted**: `f*10` rounds for full-mantissa fractions,
  so it diverges from glibc at the rounding boundary (e.g. `%.17f` of `0.1` → on-ramp
  `0.10000000000000000` vs native `...001`). Matching glibc byte-for-byte requires *correctly-rounded
  exact decimal conversion* (Dragon4/Ryū-class **big-integer** arithmetic — the fraction numerator
  alone needs > 64 bits for small magnitudes), which an `f64`-arithmetic approximation cannot give
  without silently breaking the on-ramp's byte-exact contract. Deferred to a bignum-backed formatter.
  **(Since landed:** the bignum family — `dtoa_digits` (Dragon4 digit engine) + `__svm_dtoa_sci`
  (`%e`), `__svm_dtoa_gen` (`%g`), `__svm_dtoa_fix_big` (`%f`, no magnitude ceiling) — correctly
  rounded across the whole double range; tests `printf_float_{fixed,fixed_bignum,scientific,general,
  nonfinite,zero_pad,alt_form}`. The float `0` (zero-pad after the sign) and `#` (keep the point /
  trailing zeros) flags landed last, byte-for-byte vs glibc.)**
- *Also still deferred:* `*` (dynamic width/precision) and non-constant format strings.

**Slice BE (DONE) — `argc`/`argv` (the §3e powerbox args buffer).** A `main(int argc, char** argv)`
now works end-to-end. The decision was to keep the powerbox ABI **language-neutral**: the host
delivers arguments as a flat byte blob at a *fixed, known window offset* (`svm_ir::POWERBOX_ARGS_BASE
= 128`, below the globals base), layout `{ argc:u32-LE, envc:u32-LE }` + packed NUL-terminated
strings — exactly DESIGN §3e / D44's "args_buffer at a known window offset". *No* entry-signature
change (`is_powerbox_entry` and handle granting are untouched), so nothing C-specific leaks into the
VM. All `char**` construction lives in the on-ramp:
- **Frontend** (`svm-llvm`): when `main`'s arity is `(sp, argc, argv)`, `synth_start_argv` replaces
  the straight-line `_start` with a 6-block one — same handle-stash/heap-seed prologue, then it reads
  `argc`, walks the `argc` packed strings building `argv[]` (each entry points *into* the blob — no
  copy) with the `argv[argc] == NULL` terminator at the entry SP, parks `main`'s frame a page above,
  and calls `main(main_sp, argc, argv)`. The window now reserves stack whenever this entry is used
  (it dereferences the SP, unlike the no-arg `_start`). `main(void)` is unchanged; a `main(int)` is
  fail-closed. (`main(…, envp)` and `getenv` are now supported — *Slice BF*.)
- **Runner** (`svm-run`): `run_powerbox_with_args` builds the blob from `(args, env)`, seeds it into
  the window's low bytes via the JIT's `init_mem` (applied *before* data segments, which sit at/above
  `POWERBOX_ARGS_END`, so they never overlap), and rejects an over-large or NUL-bearing arg vector.
  The CLI forwards its post-`--` arguments (`svm-run prog.svmb -- a b c`; `argv[0]` = the file name).
  When *no* `-- args` are given it seeds **nothing** (`init_mem` stays `None`, byte-identical to a
  bare run) — both to avoid perturbing the guest's initial state and because the blob is unused by a
  `main(void)`; a program wanting `argc>=1` must be invoked with `--`. The environment is always empty
  unless explicitly supplied — no ambient host env leaks in. Test: `main_argc_argv` (byte-for-byte vs
  native with a controlled `argv`, via the new `check_powerbox_vs_native_args`).

**Slice BF (DONE) — `envp` + `getenv` (the env half of the §3e blob).** The blob already packs the
environment (`{argc, envc}` then the `argc` argv strings followed by the `envc` env strings, each
`KEY=VALUE`), so this slice is **frontend-only** — the runner/blob/test plumbing was built in BE.
- **`main(int, char**, char** envp)`** (arity 4): `synth_start_argv` (now an 11-block CFG when
  `wants_envp`) finishes `argv[]` as before, then runs a mirror loop over the `envc` env strings —
  packed right where the argv walk ended — building a second NULL-terminated `char**` parked just
  above `argv[]`, and calls `main(main_sp, argc, argv, envp)` with the frame a page above *both*
  arrays. A 2-param `main` is still fail-closed. Test: `main_argc_argv_envp` (empty + multi-entry env,
  passed key-sorted to match `std::process::Command`'s `BTreeMap` ordering).
- **`getenv(name)`**: a synthesized `__svm_getenv` helper (gated on `calls_external("getenv")`, which
  also forces a powerbox `_start` since it has no import of its own). It reads the blob in the reserved
  low scratch directly — no `environ` global, no `_start` coupling — so it works at any `main` arity
  and returns `NULL` when the host seeded no env (the window reads `argc==envc==0`). It skips the
  `argc` argv strings, then compares each env key against `name` char-by-char, returning the value
  pointer just past the `=` on a full match landing on `=` (so `"F"` does not match `"FOO=…"`). Test:
  `getenv_lookup` (hit, miss, prefix guard, and the no-env case).

**Slice BG (DONE) — Monocypher crypto KAT (the 64-bit-carry shakedown; ladder #1a).** Loup Vaillant's
Monocypher 4.0.2 (public domain) runs byte-identical to native: **BLAKE2b**, **ChaCha20**, **Poly1305**,
and an **X25519 ECDH** known-answer test (both sides must derive the same shared secret; exit code =
mismatch count). Extends the integer corpus (SHA-256/xxHash/crc32) into modern AEAD + an elliptic curve,
whose 25.5-bit-limb field arithmetic (`i32 × i32 → i64` products + carry) stresses the 64-bit
shift/rotate/multiply paths. One on-ramp fix it forced: **width-changing `shufflevector`** — clang's
auto-vectorizer emits shuffles whose result lane count differs from the input; the `v128` fast-path is now
gated to same-width permutes, with width-changers falling through to the existing generic scalarize path.
Compiled with on-ramp vectorization off (`-fno-vectorize -fno-slp-vectorize`, via the new
`check_demo_vs_native_flags`) — the SIMD-vectorized crypto loops are the §17 lane, not the scalar
arithmetic this targets; the native oracle keeps vectorizing and exact integer crypto agrees. Test:
`demo_monocypher_vs_native`.

**Slice BH (DONE) — stb_image PNG decoder (the real-parser shakedown; ladder #1b).** Sean Barrett's
stb_image (public domain), PNG-only, decodes an embedded 24×24 RGBA PNG byte-identical to native —
exercising the built-in zlib inflate (Huffman + LZ77), the row unfilters (None/Sub/Up/Average/Paeth — the
test image cycles all five), the chunk/CRC walk, and heap traffic through the synthesized
malloc/realloc/free. It rides on the **sparse-switch compare chain** (Slice D, `lower_sparse_switch`)
— the PNG chunk loop switches on 4-byte fourccs (span ≫ 4096) — and forced one new on-ramp intrinsic:
- **`llvm.bitreverse.{i8,i16,i32,i64}`** — the Huffman setup reverses bits; lowered inline via the log-N
  swap network (mirroring the `llvm.bswap` inline lowering). See `bitreverse_intrinsic`.
- Configured `STBI_NO_THREAD_LOCALS` (the incidental `_Thread_local` failure-string → `llvm.threadlocal.address`,
  out of scope) and vectorization off, like BG. Test: `demo_stb_image_vs_native`.

**Slice BI (DONE) — SQLite Phase A, in-memory (the SQL-engine capstone; ladder #5).** The unmodified
**SQLite 3.50.2 amalgamation** (~257k lines, public domain — fetched + cached at test time, not
vendored) runs as a guest **byte-identical to native**: a `:memory:` database executing a
29-statement breadth script — DDL + indexes, recursive-CTE inserts, aggregates, GROUP BY/HAVING,
self-joins, string/CASE/NULL semantics, float output through SQLite's own `%!.15g` (Dekker
double-double), CASTs, **window functions**, date/time, `random()`/`randomblob()`, UPDATE/DELETE +
transactions, `quote()`/blobs, subqueries/EXISTS, `PRAGMA integrity_check`, and a deliberate error.
Design points:
- **Determinism is pinned in a `SQLITE_OS_OTHER=1` VFS** (`demos/sqlite/sqlite_demo.c`): xRandomness
  is a fixed-seed SplitMix64, xCurrentTime/Int64 a fixed instant (2024-01-01Z), xSleep a no-op — so
  `datetime('now')` and `random()` agree between the native and SVM runs. `:memory:` +
  `SQLITE_TEMP_STORE=3` means **no file I/O exists**; xOpen fail-closes (`SQLITE_CANTOPEN`), proving
  no path can even reach for disk. (Phase B replaces exactly this shim with a storage capability.)
- **Needs SQLite ≥ 3.47**: earlier amalgamations carry `long double` literals in `sqlite3FpDecode`
  (`x86_fp80` in the IR — outside the f64 on-ramp); 3.47+ replaced that path with Dekker
  double-double arithmetic, so the build is f64-clean with no source patching.
- **The on-ramp needed exactly two additions**: synthesized `__svm_strcspn` (the `strspn` scan with
  the continue condition inverted) and `__svm_strrchr` (one forward scan, select-tracked last match)
  — everything else (1448 functions, translate ≈ 2.4 s, full script ≈ 3.4 s on the interpreter) rode
  the existing surface: malloc/realloc/free, mem\*/str\* helpers, the guest printf, sparse switches,
  and the varargs ABI (`sqlite3_str_vappendf`).
Test: `demo_sqlite_vs_native` (skips cleanly offline). Follow-ups: Phase B (the storage capability +
a guest VFS bridging `sqlite3_vfs` to it) and, for scale, running SQLite's own SQL-logic scripts.

**Slice BJ (DONE) — SQLite Phase B: disk-backed persistence through the Fs capability (the
north star's second half).** Same amalgamation, but the database is a real **file**, and in the
guest build every byte of it flows through the embedder-granted `fs` capability: a **guest
`sqlite3_vfs`** (`demos/sqlite/sqlite_cap_vfs.c` — the "guest VFS shim" this doc planned) bridges
xOpen/xRead/xWrite/xTruncate/xSync/xFileSize/xDelete/xAccess to `__vm_cap_resolve("fs")` +
`__vm_host_call`, exactly how Lua's `io` runs. The rollback journal (default DELETE mode) is
created, written, **replayed** (an explicit `ROLLBACK` restores 200 rows), and deleted through the
capability. Zero ambient authority: no cap ⇒ `sqlite3_os_init` fails and no filesystem exists.
The test (`demo_sqlite_fs_cap_vs_native`) asserts three directions:
1. **stdout differential** — guest (`mem_fs`) byte-matches the native oracle (stock unix VFS in a
   temp dir) over create → close → reopen → verify;
2. **the capability story** — under `host_fs` the guest's `test.db` really lands on disk (journal
   really deleted), and the **native binary opens the guest-written file** and verifies it
   byte-identically: cross-implementation file-format proof, capability-written;
3. **the reverse** — the guest reads a native-written `test.db` through `host_fs`.
Supporting work: the fs protocol grew **`truncate` (op 7) + `sync` (op 8)** on both backends (the
probe covers shrink-discard/grow-zero-fill/read-only-refusal), and feeding the enlarged probe
through the on-ramp surfaced two chunked-vector gaps clang's SLP can emit — **`freeze <N x iK>`**
(now identity-on-parts, like the scalar/i128/mask freeze arms) and **`bitcast <N x iK> → iM`**
(lane reassembly: mask, shift, OR — the vectorized 4-byte-compare shape). What remains for a
first-class story: promoting the prototype `IoRing`/`Blocking` route vs the dedicated cap decision
(this slice used the direct `HostCap` surface). (The SQL-logic-scripts scale follow-up landed —
slice BK below.)

**Slice BL (DONE) — LMDB: an embedded memory-mapped B-tree in the sandbox (the *second* storage
shape).** SQLite proved the read/write VFS shape; **LMDB** (OpenLDAP's Lightning MDB — the original
mmap'd B-tree that libmdbx later hardened; ~12k lines vs libmdbx's 37k and a fraction of its OS
surface) proves the **memory-mapped** shape, where the data plane *is* the file-backed mapping —
readers walk the B-tree straight out of the map, no per-access host calls. The Fs capability grew a
**file-backed-mmap surface** — `FS_MMAP`/`FS_MSYNC`/`FS_MUNMAP` (ops 9/10/11, both backends): `mmap`
binds a guest window buffer to a file region (copy-in), `msync` flushes a sub-range back, `munmap`
drops it. A guest shim (`demos/lmdb/lmdb_shim.c`) bridges LMDB's `mmap`/`msync`/`pread`/`open`/… to
`__vm_cap_resolve("fs")` + `__vm_host_call`, plus single-thread no-op stubs (pthread/sysconf/uname/…)
for the OS odds-and-ends `MDB_NOLOCK` never exercises — everything else (malloc/mem\*/str\*/printf/…)
the on-ramp already synthesizes. Opened `MDB_WRITEMAP|MDB_NOLOCK|MDB_NOSUBDIR`, so every page (data +
meta) lands in the map, making the copy-in/flush-out emulation coherent (the buffer is the sole
authority). The one on-ramp addition: **`llvm.trap`/`llvm.debugtrap`** (from `__builtin_trap()`) is
now dropped — clang always follows it with an `unreachable` terminator, which already traps. The
test (`demo_lmdb_mmap_cap_vs_native`) asserts three directions, as Phase B did: guest (`mem_fs`)
stdout byte-matches native over fill → delete → reopen → point-lookups + full ordered cursor scan (a
running checksum over the B-tree walk) + stat; under `host_fs` the guest's `data.mdb` lands on disk
and **native LMDB reads the capability-written mmap database** byte-identically; and the guest reads
a native-written one. LMDB is fetched-not-vendored (OpenLDAP license), skips offline. Follow-on: a
dedicated (non-`HostCap`) file-mmap capability + multi-mapping/shared-memory coherence for a
crash-torture-grade story.

**Slice BK (DONE) — SQLite's own test corpus in the sandbox (sqllogictest).** A compact
**sqllogictest runner** as a guest program (`demos/sqlite/sqlite_logictest.c`): record parser
(statement/query, `skipif`/`onlyif`, `halt`, CRLF-tolerant), **reference-exact value formatting**
(NULL / `%d` of the 32-bit int / `%.3f` / `(empty)` / `@`-substitution — the corpus MD5s were
generated by the reference runner, so every byte matters), `rowsort`/`valuesort` (merge sort), and
an embedded RFC-1321 MD5 for the `N values hashing to <md5>` form. Scripts arrive on **stdin** (pure
Phase A — no fs capability), fetched-with-cache from the stable GitHub mirror (the fossil tarball
endpoint rate-limits). Seven scripts — `select1-3` + four `random/*` torture files — hold **~46k
records / ~56k queries**, and every one passes **twice over**: the guest's own summary reports
`failed=0` (SQLite's expected results hold in the sandbox), *and* guest stdout is byte-identical to
the native build of the same runner. Per-record guest cost is small: a 15k-record file runs in
~4-5 s (release). CI runs `select1` by default; `demo_sqlite_logictest_full` (#[ignore]) sweeps all
seven (`cargo test --test translate demo_sqlite_logictest_full -- --ignored`).

**Slice BM (SPIKE) — Postgres `--single`: whole-program bitcode pipeline + gap inventory (the
setjmp + File-capability capstone; ladder #7).** The feasibility spike for the *biggest* real
program on the ladder — "SQLite Phase B at 100×." Establishes the pipeline and, crucially, turns
"integrate Postgres" into a **concrete, quantified gap list** (the point of picking a target is
picking the gap it drives — §"Translator gaps these programs force"). The reproduction lives in
`crates/svm-run/demos/postgres/` (`build_bitcode.sh` + `emit_bc.py` — fetched-not-vendored, PostgreSQL
license). What the spike established:
- **Native oracle works.** Postgres **17.5** builds with `clang-18` (minimal config: no
  icu/ssl/zlib/readline/xml/gssapi), and `postgres --single -D <data> -O -j postgres` reads SQL on
  **stdin** and prints results (`SELECT 1+1 AS two, upper('hi')` → `2` / `HI`). This is the
  differential target, exactly as SQLite/LMDB are validated. (`--single` sheds the whole category-3
  postmaster: no fork-per-connection, no SysV shmem across processes, no listening socket, no
  signals-driven concurrency — one process, one private address space, SQL on stdin.)
- **The on-ramp reader scales to whole-Postgres.** Postgres is **not** a single amalgamation, so the
  pipeline is `-flto`-free per-TU bitcode (`clang -O2 -emit-llvm -fno-vectorize -fno-slp-vectorize`,
  flags lifted verbatim from the makefile's own compile lines) → **`llvm-link`** the exact
  `postgres` link set (833 modules: the backend + `libpgcommon_srv`/`libpgport_srv` + timezone) →
  one **17.8 MB `.bc`** / **78 MB, 1.59 M-line `.ll`** (**14 563** defined functions). The in-house
  textual-`.ll` reader ingests it and **fail-closes cleanly** on the first unsupported construct
  (~19 s, mostly the `llvm-dis` subprocess) — no OOM, no mis-parse. Scale is **not** the blocker.
- **Confirmed non-blockers.** `invoke`/`landingpad`/`resume` = **0** — `--single`'s entire
  `PG_TRY`/`ereport` error model is `sigsetjmp`/`siglongjmp` (**DONE**, all three engines incl. JIT),
  **not** C++ EH; EH stays a C++ concern. Also **0** `x86_fp80`/`fp128`, **0** `thread_local`, **0**
  `llvm.stacksave` (no VLAs survive `-O2`). The named prerequisites (setjmp/longjmp, the Fs
  capability, Dragon4 `%f`/`%g`) are all landed — Postgres is genuinely *next*, not primitive-blocked.
- **The gap list (the deliverable).** Static inventory over the linked module:
  1. **Inline `asm` — 921 sites, but only ~9 distinct templates.** 559 are **empty memory-barrier**
     asm (`""` + `~{memory}` — compiler fences) → **drop** (no-op); `lock;addl $$0,0(%rsp)` /
     `rep;nop` (barrier / PAUSE) → **drop**; `xchgb` / `xaddl` / `xaddq` / `cmpxchgl` / `cmpxchgq`
     (spinlock TAS + `arch-x86.h` atomics) → **atomic-RMW, and single-threaded under `--single` ⇒
     plain load-op-store**; `cpuid` / `popcntq` / `popcntl` (`pg_bitutils` runtime dispatch) →
     recognize `popcnt`→`Popcount`, feature-detect `cpuid`→fixed value (fall to the SW path). A small
     fixed **recognize-and-lower table** — the same shape as the `setjmp`/`memcpy`/`llvm.trap`
     recognizers — not open-ended asm support. **#1 blocker; the biggest single lever.**
  2. **`atomicrmw` — 110 sites** (the `__atomic`/`__sync` generic path). Single-threaded lowering to
     load-op-store; pairs with the asm-atomic recognizer (same lowering, two front doors).
  3. **`i128` — 252 sites** (64×64→128 widening in the `numeric`/aggregate accumulators). Two routes:
     the **config lever** `#undef HAVE_INT128`/`PG_INT128_TYPE` (Postgres ships a pure-64-bit
     `int128.h` fallback — zero translator work), or on-ramp i128-as-`{i64,i64}` emulation. Config first.
  4. **Vectors — ~3.6 k `<N x …>` sites** even under `-fno-vectorize`: mostly `<16 x i8>`/`<N x i8>`
     from **small-struct `memcpy`/`memset` lowering** (+ some `<2 x double>`/`<4 x i32>`). The on-ramp
     scalarizes 2-lane (slice V) and 4-lane float (slice Y); wide **integer** vector *memory* ops
     want a general "scalarize any vector load/store/`memcpy` lane-wise" pass. The **fuzziest** gap —
     needs a width census before scoping; some fall away under `-fno-builtin-mem*`.
  5. **Varargs breadth — 43 `llvm.va_start`** (`elog`/`ereport`/`snprintf` family; **0** `va_arg`
     instrs — clang inlines the SysV `va_arg` as GEP+load). The on-ramp's varargs `printf` covers the
     shape; confirm it holds across Postgres' `appendStringInfo`/`errmsg` sites.
  6. **The OS waist — the fs/syscall shim (runtime, not an IR gap).** Postgres calls raw
     `open`/`pread`/`pwrite`/`fsync`/`ftruncate`/`unlink`/`mkdir`/`opendir`/`readdir`/`stat` +
     `getpid`/`geteuid`/`getpwuid`/`time`/`clock_gettime`/`sysconf` — far more surface than SQLite's
     tidy `sqlite3_vfs`, but the same play: bridge the file ops to the granted **`fs` capability**
     (Phase B / LMDB machinery), deterministically stub the rest (fixed pid/euid/clock, single-thread
     no-ops), and gate the root-check (`geteuid()==0`).
  7. **Data-dir strategy.** `--single` needs an initialized cluster; the cheap first move is `initdb`
     **natively**, then expose the dir read/write through the `fs` cap (mirroring SQLite Phase B,
     where the file is cap-written but the schema is guest-driven) — deferring in-sandbox `initdb`
     (`postgres --boot`, another backend program) to a later slice.
  **Staged plan:** (1) portable-atomics/no-`int128` config + the asm/`atomicrmw`
  recognize-and-lower table → the module translates; (2) the vector-memory scalarization census +
  pass; (3) the fs/syscall shim over the `fs` cap + a pre-`initdb` data dir → boot `postgres
  --single` on a fixed SQL script, byte-identical to native; (4) a `pg_regress` subset. This slice
  is the **map**; the follow-ons are the territory.

**Slice BN (DONE) — inline-asm recognize-and-lower: barriers + `popcnt` (Postgres gap #1, part 1).**
The first translator gap from slice BM. The on-ramp does **not** execute asm — opaque machine code
can't be masked or re-verified, which is the whole §2a sandbox thesis — so a **fixed allowlist**
matches the handful of template strings known C headers emit and re-emits their *semantics* as
ordinary verified IR, failing closed on anything else. Landed:
- **Parser.** `InlineAssembly` now carries the `template`/`constraints` strings (was type-only);
  `call_signature` parses `asm [sideeffect|alignstack|inteldialect|unwind] "<tmpl>", "<cons>"` in the
  callee position (covers `call` **and** `invoke`, which share it), and `skip_arg_attrs` learns
  `elementtype(<ty>)` (the pointee type an indirect `*m` memory operand carries). Inline asm was
  previously unparseable (`expected a call callee, found asm`).
- **Recognizer (`lower_inline_asm`).** Dispatched early in the call chain (an asm callee has no
  `callee_name`): **compiler/memory barriers** (`""`+`~{memory}`, `lock; addl $$0,0(%rsp)`) and the
  **PAUSE** hint (`rep; nop`) → **dropped** — no architectural effect for a single-address-space,
  single-threaded guest (and correct in tail position: the drop falls through to the real `ret`);
  **`popcnt`** (`popcntq`/`popcntl $1,$0`, `pg_bitutils`' fast-path) → the `Popcnt` unary op (as
  `llvm.ctpop`). Any **unrecognized** template is a clean `Unsupported` (§2a fail-closed) — never a
  silent drop. The x86 atomic-RMW templates are deliberately left for the config lever (portable
  atomics emit `atomicrmw`/`cmpxchg` **instructions**, which the on-ramp *already* lowers
  single-threaded — one lowering, not two front doors), and `cpuid` disappears with the same
  popcount-dispatch lever.
- **Tests** (`inline_asm_barriers_dropped`, `inline_asm_popcnt_lowers`,
  `inline_asm_unrecognized_is_fail_closed`) — differential vs native (the asm survives `-O2` into the
  bitcode, verified), plus the fail-closed contract. **279 translate tests green, fmt + clippy clean.**
- **Effect on the capstone:** the full 78 MB Postgres module now **parses past the entire 921-site
  inline-asm surface** — asm is no longer the blocker.

**Slice BO (DONE) — inline-asm atomics + cpuid: the whole Postgres asm surface clears.** Extends the
recognizer (BN) with the remaining templates, so the complete backend module (see BP) translates past
**all 921** inline-asm sites:
- **x86 atomic RMW / CAS** (`arch-x86.h` `pg_atomic_*` + `s_lock.h` TAS): `xchg{b,w,l,q}` →
  `AtomicRmw::Xchg`, `xadd{b,w,l,q}` → `AtomicRmw::Add`, `cmpxchg{l,q}; setz` → `AtomicCmpxchg` +
  the `{old, success}` aggregate. These lower to the **same runtime atomic ops** the on-ramp already
  emits for `atomicrmw`/`cmpxchg` **instructions** — so they are *genuinely atomic* (not a racy
  load-op-store) and need **no single-threaded gate**, superseding BM's framing. Operand roles are
  pinned by asserting the exact constraint signature (`=q,=*m,0,*m` / `={ax},=*m,=q,{ax},r,*m`),
  fail-closed otherwise; narrow (i8/i16) variants route through the existing narrow CAS-loop helpers
  (`uses_narrow_atomic` now also spots a narrow atomic *asm* call so the helper registers).
- **`cpuid`** (`xchgq %rbx; cpuid; xchgq %rbx`, the `pg_bitutils`/`pg_crc32c` feature probe) → an
  all-zero `{eax,ebx,ecx,edx}`, so Postgres takes its **portable software** popcount/CRC paths, which
  compute identical results → still byte-identical to native.
- Test `inline_asm_x86_atomics` mirrors `arch-x86.h`'s exact asm (TAS/fetch-add/CAS), differential vs
  native. **280 translate tests green, fmt + clippy clean.**

**Slice BP (IN PROGRESS) — the complete Postgres module + the external-surface map.** Two findings
that reshape the capstone estimate:
- **The link set must be complete.** An earlier incomplete link (a fragile bitcode-emit step) left
  functions like `hash_numeric` *declared-only*, surfacing as a spurious `constexpr reference to
  @hash_numeric`. Fixed the pipeline (`emit_bc.py` now bumps the source mtime instead of deleting the
  `.o`, so it's idempotent and never corrupts the native tree; a clean rebuild + regenerated
  `objfiles.txt` restores every object): the complete module is **834 modules / 14 730 defined
  functions**, and it now translates cleanly past all asm.
- **The remaining surface is the OS/libc waist, and it is large.** With asm cleared, translation
  fail-closes at the **first undefined external** (`log`). The module has **251 distinct declared-only
  externals**: **libm** (18 — `log`/`exp`/`pow`/`sin`/`cos`/… — transcendentals the SVM has no op for;
  need a **bundled guest libm**, the raytrace "bring-your-own-libm" model, llvm-linked in), **file/OS
  syscalls** (~30 — `open`/`pread`/`pwrite`/`fsync`/`stat`/`mkdir`/`opendir`/`mmap`/… → the **`fs`
  capability** shim, gap #6), **proc/time/signal** (~24 — `getpid`/`geteuid`/`clock_gettime`/
  `sigaction`/`fork`/`kill`/… → deterministic stubs), and **~180 other libc** (`strtod`/`snprintf`/
  `qsort`/`setlocale`/`strftime`/`memmem`/… — some the on-ramp synthesizes, many not yet). **Every one**
  must resolve (synthesized helper / capability / bundled guest code / stub) before the module
  translates, and then verify + the runtime (initdb data dir, storage manager, WAL, single-process
  shmem, catalog bootstrap) must all work. **This is the multi-week bulk of the capstone** — the asm
  slices (BN/BO) were the tractable translator corner; the external waist is the mountain. The map
  above is the plan of record; each category is a follow-on slice.

**Slice BQ (DONE) — the bundled guest libm (Postgres external category #4: the 18 transcendentals).**
The SVM has no transcendental op (only the hardware float ops sqrt/floor/…), so `log`/`exp`/`pow`/
`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`sinh`/`cosh`/`tanh`/`cbrt`/`log10`/`log2`/`exp2`/`fmod`
stay **guest code** — the raytrace "bring-your-own-libm" model, but a *real* libm (openlibm) rather than
poly approximations, `llvm-link`ed into the module. Findings + deliverable:
- **openlibm's double math translates through the on-ramp with zero gaps.** Its C sources carry no
  inline asm (the asm is in separate `amd64/`/`i387/` dirs we don't compile); the 28-file double set
  (entry points + the `k_*`/`e_rem_pio2` kernels + `k_exp`/`expm1`/scaling) compiles to bitcode and
  translates clean. `sqrt`/`fabs` the code calls resolve to on-ramp float ops (no `e_sqrt` needed).
- **Bit-exact differential.** `libm_bundled_vs_native` links the *same* openlibm on **both** sides
  (guest bitcode *and* native oracle — not the system `-lm`), so any last-ulp choice is identical by
  construction and the test isolates the *math translation*. The driver (`demos/postgres/libm_probe.c`)
  FNV-hashes the **raw IEEE bits** of every result and prints hex via `putchar`, so float formatting is
  out of the loop — the on-ramp reproduces openlibm **bit-for-bit** over ~3600 evaluations. Openlibm is
  fetched-not-vendored (BSD); the test skips offline. **282 translate tests green, fmt + clippy clean.**
- **Effect on the capstone:** with libm llvm-linked, the full Postgres module translates past all 18
  transcendentals; the next undefined external is `strchrnul` (the "other libc" category — a synthesized
  string helper).

**Slice BR (DONE) — the fail-closed extern-stub mechanism: the module clears the *entire* external
surface.** The high-leverage lever from slice BP's map. Postgres has ~250 declared externals, but the
vast majority are **dead on the `--single` query path** (network `accept`/`connect`/`epoll_*`, `fork`/
`exec`/`pipe`, `dlopen`, `syslog`, `backtrace`, …) — yet the strict on-ramp fail-closes at *translate*
time on the first one it doesn't handle, so clearing them one-by-one would be ~200 whack-a-mole helpers
for functions that never run. Instead, an **opt-in** `TranslateOptions { stub_unresolved_externs }`
lowers a call to a genuinely-undefined external (one no recognizer/synthesizer/capability handled — i.e.
that *falls through to the fallback* at the direct-call site, so the classification is correct by
construction, no fragile name predicate) to a synthesized **trap stub**: a function whose body is a
single `unreachable`, which traps. This defers the fail-closed from **translate time to run time** —
the whole-program module translates + verifies, and a stub only traps if actually *called* (never an
escape: it is ordinary re-verified IR, §2a). Off by default (a typo'd callee stays a clean
translate-time error for ordinary programs); opt in via `translate_bc_path_with_options` / the
`--stub-externs` CLI flag for a large-program bring-up.
- **Design.** Stubs are minted lazily into a shared `RefCell<StubTable>` threaded through the
  translation (one per distinct name, keyed by name, using the first signature seen — the SVM sig is
  the threaded data-SP + the call's fixed params, matching the `args=[sp,…]` the site builds) and
  appended to `funcs` **last**, after `_start` + defined + helpers, so each lands at its assigned
  `stub_base + ordinal` index. Reuses the existing `synth_trap_stub`. A later mismatched-signature call
  to the same name surfaces as a clean `svm-verify` type-id error, not an escape.
- **Test** `unresolved_extern_stub_opt_in`: strict default fail-closes on `mystery`; opt-in translates
  + verifies + runs to a **clean exit** with the stub gated off (a dead stub is inert); and a *called*
  stub **traps** (`run_powerbox` errors). **283 translate tests green, fmt + clippy clean.**
- **Effect on the capstone — a milestone:** with `--stub-externs`, the full Postgres module (backend +
  libm) translates past the **entire ~250-external OS/libc surface** in one lever. The next gap is no
  longer an external at all — it is the **SIMD vector tail** (~9 per-lane, non-constant-splat vector
  shifts: `ashr <16 x i32>`, `lshr <4 x i64>`/`<4 x i32>`, from explicit SSE/AVX SIMD, dead at runtime
  under the `cpuid`→0 stub but still compiled). That vector category (#4) is the next slice — either
  per-lane scalarization in the on-ramp, or a config lever compiling the SIMD fast-paths out.

**Slice BS (DONE) — per-lane vector shifts + address-taken extern stubs (two Postgres-tail slices).**
Two independent gaps the module hit after BR, each a clean, tested advance:
- **Per-lane (non-constant-splat) vector shifts.** `VShift` (§17) takes one scalar count for *all*
  lanes; clang's usual constant-splat amount maps directly, but real SSE/AVX SIMD (Postgres' `simd.h`)
  emits **per-lane** amounts (`lshr <4 x i32> %a, %amt`, `ashr <16 x i32>`, …). The on-ramp now
  **scalarizes** those: `v128_lane_shift` extracts each lane + its own count (`ExtractLane`), shifts in
  the lane's integer type (`IntBin`), and repacks (`build_v128_from_lanes`) — the shift analog of the
  existing vector funnel-shift path. Wired into **both** the 128-bit `bin` path (`<4 x i32>`/`<2 x i64>`)
  and the wide `wide_int_shift` path (per-`v128`-chunk, e.g. `<8 x i32>`). Restricted to full-width
  lanes (`I32x4`/`I64x2`); a narrow (`i8`/`i16`) variable shift stays fail-closed (clang keeps those as
  constant splats — the native `VShift` handles them). Test `vector_shift_per_lane_amount` (runtime
  `volatile` seed + amounts so nothing constant-folds; all result lanes folded to one `%u`) is
  byte-identical to native across 128-bit + wide shapes.
- **Address-taken extern stubs.** BR stubbed undefined externals at *call sites*; a **function pointer**
  to an undefined extern (a comparator/dispatch-table entry — what blocked Postgres on `@memcmp`, e.g.
  `select …, ptr @string_compare, ptr @memcmp`) reaches the operand resolver as an opaque `ptr` with no
  call to derive a signature from. Fixed: the parser now records each `declare` prototype in
  `func_declarations` (it previously kept the type only in the internal `symbols` map), so
  `StubTable::get_or_insert_extern` recovers the declared signature and mints the funcref stub; the
  `@global` operand resolver returns its index (the address-of counterpart to the call-site path).
  `None` for an undefined *data* global (no funcref — stays fail-closed). Test
  `unresolved_extern_funcptr_stub`: strict fail-closes on the address-taken `mystery`; opt-in resolves
  the funcref and runs clean when the pointer's live target is the real function (an unselected stub is
  inert). **286 translate tests green, fmt + clippy clean.**
- **Effect on the capstone:** the Postgres module now translates past the vector-shift wall **and** the
  address-taken `@memcmp`; the next gap is the **`<4 x i1>` mask type** (`type <4 x i1> (Milestone 1+)`)
  — the SIMD vector tail continues (mask legalization next), then `<4 x i64>` wide-type support.

**Slice BT (DONE) — mask-mask bitwise (`and`/`or`/`xor <N x i1>`): the SIMD "any-match" idiom.** The
`<4 x i1>` blocker from BS. `lower_mask` already scalarized vector compares/select/extract/insert/
shuffle/movemask/`freeze` and (via `lower_vec_int_convert`) `sext`/`zext` of a mask, but **not** the
bitwise *combination* of masks — `or <4 x i1> %m1, %m2`, which is how SIMD folds several comparison
masks into one ("does any of these lanes match?", then `sext` to a full-width vector; Postgres'
`simd.h`). Those `or`/`and`/`xor <N x i1>` fell through `lower_mask` to the scalar `bin` path →
`val_type(<4 x i1>)` → the `type <4 x i1> (Milestone 1+)` error. Now lowered lane-wise
(`lower_mask_bitwise`: `IntBin` per `0`/`1` lane, result bound as scalarized mask lanes) — the mask
analog of the vector int binop. Cross-block-safe for free: the scan classifies *any* `<N x i1>`-typed
value into its `[i32; N]` fan-out `agg_layout`, so a mask combined in one block and consumed in another
threads through block params like any mask. Test `vector_mask_bitwise_any_match` (the `(a==t1) |
(a==t2) | (a==t3)` idiom clang `-O2` folds to `or <4 x i1>` + one `sext` — verified present in the
bitcode) is byte-identical to native. Also repairs a **pre-existing `main` break**: the upstream
`TranslateOptions { stack_page }` addition left one test's struct literal un-reconciled (missing the
field), so the svm-llvm lane didn't compile — fixed with `..Default::default()`. **287 translate tests
green, fmt + clippy clean.** Effect: the Postgres module now translates past the mask type; the next
gap is an unrelated SSA-liveness corner (`value … not available in block`) in a later function.

**Slice BU (DONE) — `freeze` of an aggregate + vector `ctpop`/`reduce` scalarization (the popcount
dispatch tail).** Three small on-ramp fixes that carry the Postgres module through its `pg_popcount_*`
functions, plus a diagnostics win:
- **Function-named errors.** A whole-program error is now prefixed `in `@fn`: …` (a bare "value N not
  available" is opaque across 14 k functions). This is what isolated the liveness gap below.
- **`freeze` of an aggregate.** The BT liveness error (`value … not available in block`, in
  `pg_popcount_masked_choose`) was a `freeze { i32, i32, i32, i32 }` — the `cpuid` result the recognizer
  binds *field-wise* in the `agg` table. The scalar `freeze` arm `ctx.operand`-ed it as a scalar and
  failed. `freeze` of an aggregate is the identity (its fields are already defined, §3c) → rebind the
  same fields. Test `freeze_of_aggregate` (a hand `.ll` `insertvalue`→`freeze`→`extractvalue`, since
  clang rarely emits an aggregate `freeze` from C).
- **Vector `ctpop` on any width.** Only `i8x16` had a native vector `ctpop`; `pg_popcount_slow`'s
  `llvm.ctpop.v2i64` now scalarizes (extract lane → `Popcnt` → repack — zero-extension leaves a
  popcount unchanged, so it is correct for every width).
- **Wide / 2-lane `vector.reduce`.** The horizontal reduce was 128-bit-only; it now also folds a
  **wide** (>128-bit) accumulator lane-wise across chunks (`pg_popcount_avx512`'s `reduce.add.v8i64`)
  and a **2-lane** packed `<2 x i32>` (via `vec_explode`, which covers both non-wide reps).
- Test `vector_ctpop_and_wide_reduce` (a popcount-sum loop clang `-O2` vectorizes into vector `ctpop`
  + a wide reduce) is byte-identical to native on interp + JIT. **289 translate tests green, fmt +
  clippy clean.** Effect: the module now translates past the popcount dispatch; the next gap is
  `pg_popcount_avx512` proper — full AVX-512 (`<64 x i1>` mask registers, `<8 x i64>`,
  `llvm.masked.load`), which is **dead at runtime** under the `cpuid`→0 stub and is best removed by a
  **build-config lever** (compile Postgres without the AVX-512 popcount fast-path) rather than teaching
  the on-ramp the whole AVX-512 surface — the next slice.

**Slice BV (DONE) — the SIMD tail closes via two build-config levers (no on-ramp change).** The
diagnosis from BU was "teach the on-ramp AVX-512"; the reality on inspection was better — almost the
*entire* Postgres SIMD surface is **incidental**, and closes at the source:
- **The flag-ordering bug (the big one).** `emit_bc.py` passed `-fno-vectorize -fno-slp-vectorize` to
  disable clang auto-vectorization — but inserted them *before* the recovered `-O2`. For the
  vectorizer knobs the **last** flag on the line wins, and `-O2` turns the loop/SLP vectorizers back
  on, so auto-vectorization was silently **never disabled**. Scalar C loops (e.g. `gistextractpage`'s
  offset-array copy) were emitted as `<2 x i32>` loads → `zext <2 x i64>` → `getelementptr <2 x i64>`
  gather-GEPs → `<2 x ptr>` stores — the "SIMD tail" that motivated BS/BT/BU. Appending the flags
  *after* `-O2` (`out_toks.extend(EXTRA)`) makes them take effect; the whole auto-vectorized tail
  vanishes (the module's `<N x …>` occurrences drop by ~40%, and the first gap jumps from
  `gistextractpage`'s vector GEP clear across the vector category). The residual **explicit** SIMD
  (SSE4.2 `_mm_crc32`, 128-bit float vectors) still translates via the on-ramp's existing vector
  support (Y/BS/BT/BU) — those slices earn their keep on the real SIMD, just not on the phantom tail.
- **AVX-512 popcount off.** Independent of vectorization (it is explicit `_mm512_*` intrinsic code in
  `pg_popcount_avx512.c`, gated by `USE_AVX512_POPCNT_WITH_RUNTIME_CHECK`). The `configure` autodetect
  is forced to "no" via its own cache vars (`pgac_cv_avx512_popcnt_intrinsics_=no` and the
  `_mavx512vpopcntdq_mavx512bw` variant) — exactly the config a host without AVX-512 produces — so the
  macro is never defined, `PG_POPCNT_OBJS` is empty, and `pg_popcount_avx512` / its `<64 x i1>` body
  leave the link set. Dead anyway under the guest `cpuid`→0 (`pg_popcount_avx512_available()` is
  false, scalar popcount is always chosen), so the native oracle stays a valid differential target.
- **No translator code changed** (so the 289 translate tests are untouched and still green); the
  deliverable is the two demo-build fixes (`build_bitcode.sh` configure args + `emit_bc.py` flag
  order) plus this log. **Effect:** the module clears the entire SIMD + AVX-512 surface and now stops
  at the first **indirect varargs call** (`manifest_process_version` — a `(...)` function pointer; the
  on-ramp marshals *direct* varargs but rejects indirect), the next slice.

**Slice BW (DONE) — indirect varargs call (a `(...)` function pointer).** The on-ramp already
marshaled a *direct* `(...)` call's variadic arguments into the caller's overflow scratch (one 8-byte
slot each) and deposited the area pointer at the callee's frame-0 slot for `va_start`; the only thing
missing for an **indirect** varargs callee was that three spots each special-cased "direct" and bailed
on the pointer form. All three now treat the two callee shapes identically — the marshaling is
byte-for-byte the same, only the call instruction differs:
- **`vararg_call_extra`** (frame-layout pass) dropped its `callee_name(c)?` early-return, so an
  indirect `(...)` call also reserves `VARARG_SCRATCH` (else "varargs call without reserved scratch").
- **The marshaling `fixed` match** dropped its `callee_name(c).is_some()` guard, so the variadic args
  are stored to scratch (not pushed as IR args) and the area pointer is deposited at `callee_sp + 0`,
  for an indirect callee too.
- **`indirect_sig`** stopped rejecting `is_var_arg`: a `(...)` callee's SVM signature is
  `(sp, fixed-params…)` — `param_types` are exactly the fixed params — which is the *same* shape a
  defined `(...)` function lowers to (§varargs), so the `call_indirect` §3c type-id check matches.
- Found as the Postgres `manifest_process_version` gap. Test `varargs_indirect_call`: a `(...)` helper
  reached through a `volatile`-indexed function pointer (so clang can't devirtualize) — byte-identical
  to native on interp + bytecode + JIT, and clang emits it `tail`, exercising `return_call_indirect`
  too.

Two **i128 op lowerings** landed in the same slice (the next two gaps, both the *same* root cause).
The reported `value … not available in block` in `sqrt_var` looked like a liveness bug but was not:
`lower_i128` was missing the op, so the **generic scalar** handler `ctx.operand`-ed an i128 value —
which lives as an `agg` `(lo, hi)` pair, not in `idx_of` — and failed. The `id()` diagnostic (agg/wide
membership) pinned it immediately.
- **`select i128`** — a per-word `Select` on the `(lo, hi)` pairs (clang emits it from a `? :` on a
  128-bit quantity; numeric `sqrt_var`'s Newton's-method inner loop). Without it the generic scalar
  `Select` mishandled the pair.
- **`store i128`** — two i64 stores, lo at the base and hi at base+8, mirroring the existing `load i128`
  layout (numeric `int2_accum`'s `sumX2` accumulator). Added inside the Store effect-arm before the
  scalar `ctx.operand`, so an i128 value never takes the scalar path.
- Test `i128_select_and_store_roundtrip`: a hand `.ll` (clang won't emit `select i128` from simple C —
  it always branches) that selects between two i128s, stores the winner to an `alloca`, loads it back,
  and folds `lo - hi` (asymmetric — catches a swapped half or a mis-picked select); 293 on interp + JIT.
- **292 translate tests green, fmt + clippy clean.** Effect: the module translates past `sqrt_var` and
  `int2_accum` and now stops at a **vector `llvm.bswap`** in `pg_sha256_final` (SHA-256's big-endian
  digest write — the on-ramp scalarizes vector `ctpop`/min/max but not yet vector `bswap`), the next
  slice.

**Slice BX (DONE) — vector `llvm.bswap`; the Postgres module now translates end-to-end.** A 128-bit
vector byte-swap (`<4 x i32>`) reverses the bytes **within each lane** (element-wise, not across the
register). There is no native vector byte-swap op, so scalarize exactly like vector `ctpop`: explode the
lanes, reverse each with the scalar `emit_bswap`, repack via `build_v128_from_lanes`. One arm in the
vector-intrinsic dispatch (an `i8x16` shape is the identity, but `emit_bswap(nbytes=1)` handles it
harmlessly). Test `vector_bswap_128`: a hand `.ll` (a `-O2` bswap loop over-vectorizes to `<16 x i32>`,
which is a *separate* wide-vector-type gap) whose inputs `0x0N000000` byte-swap to lane `N`, folded
`e0*1000 + e1*100 + e2*10 + e3` = 1234 (asymmetric — a swapped byte or lane order fails); 1234 on interp
+ JIT. **293 translate tests green, fmt + clippy clean.**

**★ Milestone:** this was the *last translate gap*. The whole Postgres backend — **832 modules /
14 985 functions** — now translates through the on-ramp with `--stub-externs`, no fail-closed. The
remaining step to a *verified* module: after `svm_ir::resolve_imports` binds the 4 powerbox caps
(`read`/`write`/`exit`/`vm_map` → `cap.call`), `svm-verify` reports one **`TypeMismatch`** (an `i32` fed
where `i64` is expected) in `ExecRenameStmt` — a translator correctness bug, the next slice. (Before
that resolve step a raw `CallImport` is expected and correctly rejected by the verifier §7.) Then the
**runtime** (initdb data dir + `fs` cap, storage manager, WAL, single-process shmem, catalog bootstrap).

**Slice BY (DONE) — aggregate fan-out in the sparse-`switch` chain; the Postgres module now *verifies*.**
The single `svm-verify` error across all 14 985 functions, in `ExecRenameStmt`. Root cause: three places
expand a threaded block param — `block_params` (the param vids), `branch_args` (the args), and
`block_param_types` (the param *types* a synthetic compare-chain block gets in `lower_sparse_switch`) —
and the first two fan out **both** wide vectors *and* aggregates (a flat struct / i128 `(lo,hi)` /
`<N x i1>` mask → one slot per field), but `block_param_types` fanned out only wide vectors. So when
`ExecRenameStmt`'s sparse `switch` threaded a by-value `{i64,i32}` struct through its compare chain,
`block_param_types` contributed **one** placeholder type while `branch_args` supplied **two** args; the
`zip` that types the chain block desynced right after the struct and mistyped every threaded value behind
it → a `TypeMismatch`. One-branch fix: `block_param_types` now fans aggregates out too
(`types.extend_from_slice(ftys)`), matching the other two. Found by driving translate → `resolve_imports`
→ `svm-verify` and bisecting the failing edge with the verifier's own `func_value_types`. Test
`switch_sparse_threads_aggregate` (hand `.ll` — clang's SROA scalarizes a struct before it can be
threaded, so C can't isolate it — verifies + runs; the same `.ll` fails `svm-verify` with a `TypeMismatch`
before the fix). **294 translate tests green, fmt + clippy clean.**

**★★ Milestone:** the whole Postgres backend — **832 modules / 14 985 functions** — now **translates
*and* verifies**: after `resolve_imports` binds the 4 powerbox caps, `svm-verify` passes clean. What
remains is purely the **runtime** — `initdb` (natively) exposed via the `fs` cap; storage manager, WAL,
single-process shmem, catalog bootstrap; real impls for the ~50 externals the query path calls.

**Slice BZ (DONE) — the `fs` capability's metadata + directory surface; the runtime begins.** The
translate/verify frontier is closed, so the work turns to *running* the module — and the first blocker is
that the `fs` cap (`crates/svm-run/src/fs.rs`) could open/read/write/seek **files** but had no way to
**walk a tree**: no `stat`, `mkdir`, `rmdir`, `opendir`/`readdir`. A natively-`initdb`'d cluster is a deep
directory tree (`base/<db>/…`, `global/…`, `pg_wal/…`), and Postgres `stat`s and scans it pervasively at
startup before it can open a single relation. Added ops 14–19: `stat` fills a fixed 72-byte little-endian
`StatBuf` (the `S_IF*` type bits + size + mtime + ino/dev) with **lstat** semantics — a symlink is never
followed, so it can't be used to probe the *type* of something outside the granted root; `mkdir`/`rmdir`;
`opendir` snapshots a directory's immediate entries and `readdir` yields them **sorted**, one per call
(`0` at exhaustion), `closedir` drops the handle. Both backends stay at protocol parity — `mem_fs` models
directories over its flat name table (a path is a directory if it was `mkdir`'d or is a strict prefix of a
file key), `host_fs` walks the real tree — so a differential runs identically on either. Tests
`os_metadata_ops_parity_mem_vs_host` (the same scripted walk returns the same rc sequence + type bits +
size on both backends) and `readdir_is_sorted_and_bounded` (sorted iteration; a too-small buffer fails
closed without consuming the entry). **svm-run suite green, fmt + clippy clean.** Next (gap #11b): a guest
OS-shim binding the file syscalls + proc/time to this cap, then the ~180 remaining pure-libc externs
(stdio `FILE*`, locale, ctype, `strtod`/`snprintf`) byte-exact vs the native oracle.

**Slice CA (DONE) — the guest OS-shim: the file + directory syscalls over the `fs` cap.** Unlike SQLite
(one `sqlite3_vfs` seam), Postgres calls the libc syscall wrappers *directly* all over `fd.c`/`md.c`/
`xlog.c` — `open`/`read`/`pread`/`write`/`pwrite`/`lseek`/`stat`/`fstat`/`lstat`/`access`/`unlink`/
`rename`/`mkdir`/`rmdir`/`ftruncate`/`fsync`/`opendir`/`readdir`/`closedir`/`chdir`/`getcwd` — and in the
whole-program bitcode every one is an undefined external (the guest links no libc). `os_shim.c`
(`crates/svm-run/demos/postgres/`) **defines** them for a guest build, bridging each to
`__vm_cap_resolve("fs")` + `__vm_host_call` (the slice-BZ cap), mapping the C `open` flag bits to the
cap's `FS_O_*`, and filling glibc's `struct stat`/`struct dirent` **by field** (the shim is compiled with
the same headers as the caller, so offsets agree) from the 72-byte `StatBuf` the `FS_STAT` op returns.
`fstat` (no fstat-by-fd op yet) reports an open fd as a regular file sized by `SEEK_END`; `readdir` hands
back `DT_UNKNOWN` so callers `stat` for the type; `chdir`/`getcwd` treat the rooted cap as the working
directory. Differential `demo_pg_oscap_vs_native`: `os_probe.c` drives a deterministic file+dir sequence
(create/write/`stat`/`pread`/sorted dotfile-filtered `readdir`/`rename`/`rmdir`/`ftruncate`) and the guest
byte-matches the native glibc oracle over **both** `mem_fs` and `host_fs` — and the `host_fs` root is
empty afterward (the probe self-cleans), proving real files through real syscalls. **svm-llvm lane green
(fmt + clippy `-D warnings` + the new test).** Next (gap #11c): proc/time/signal stubs, then the ~180
pure-libc externs byte-exact vs native, then storage manager / WAL / shmem / catalog bootstrap.

**Slice CB (DONE) — the guest pure-libc shim begins with ctype.** Companion to `os_shim.c` (syscalls):
`libc_shim.c` (`crates/svm-run/demos/postgres/`) holds the *pure* libc surface — no capability, just
deterministic computation. First inhabitant: **ctype**. glibc's `<ctype.h>` `isalpha`/`isdigit`/… macros
expand to `(*__ctype_b_loc())[c] & _ISbit`, a direct index into a locale table reached through
`__ctype_b_loc`/`__ctype_tolower_loc`/`__ctype_toupper_loc` — undefined externals in the guest, and
Postgres's scanner/parser classify *every* input byte this way, so the SQL front end is dead without
them. The shim provides the **C/POSIX-locale** tables (the locale Postgres bootstraps in) as **static
compile-time literals** — element 128 is code point 0, so `ptr[c]` is valid across glibc's `[-128, 255]`
index range, and there is no runtime initializer to mis-fire (the first cut used lazy init; the guest
read a still-null table pointer and printed garbage — the differential caught it, and precomputed
literals removed the failure mode). Differential `demo_pg_ctype_vs_native`: `ctype_probe.c` prints all
twelve classifications + `tolower`/`toupper` (read straight from the `*_loc` tables, the form Postgres
uses — not the `tolower()` *function*, whose non-constant-`int` path is a separate surface) for every
byte 0..255, and the guest byte-matches the native glibc oracle over the whole range — which pins every
bit of every table. Pure computation, runs on the bare powerbox. **svm-llvm lane green.** Next (gap
#11d): proc/time/signal stubs, then the rest of the pure-libc surface (stdio `FILE*`, locale,
`strtod`/`snprintf`) byte-exact vs native, then storage manager / WAL / shmem / catalog bootstrap.

**Slice CC (DONE) — guest libc: string + integer parsing + proc/time/signal (a bundle).** Three related
groups of the guest runtime, in one PR. **(1) string** — `libc_shim.c` adds the `<string.h>` members the
on-ramp doesn't already synthesize (synthesized already: `strlen`/`strcmp`/`strcpy`/`strchr`/`strrchr`/
`strspn`/`strcspn`/`strpbrk`/`strncmp`/`strcoll`/`memcmp`/`memchr`/`bcmp`): `strcat`/`strncat`/`strncpy`/
`strnlen`/`strstr`/`strchrnul`/`strdup`/`strlcpy`/`strlcat`/`strtok`(`_r`)/`strxfrm`/`strcoll_l`. **(2)
integer parsing** — `strtol`/`strtoul` (+ the glibc-C23 `__isoc23_*` aliases) / `atoi`/`atol` over a
shared core handling sign, base 0-autodetect (`0x`/`0`), whitespace, `endptr`, and `ERANGE` clamp to
`LONG_MAX`/`MIN`/`ULONG_MAX`; `strtod`/`snprintf`/`getenv` were **already** synthesized, so they're not
re-done. **(3) proc/time/signal** — `proc_shim.c` returns the deterministic values a single-user sandbox
backend needs: constant **non-root** identity (`geteuid()==1000`, so Postgres's root guard passes), a
frozen clock (`gettimeofday`/`clock_gettime`/`time` at a fixed epoch), inert signal masks
(`sigaction`/`sigprocmask`/… all succeed and track nothing), no-op `nanosleep`/`setitimer`, unlimited
`getrlimit`, and `abort`/`__assert_fail` → `_exit(134)`. A shared, include-guarded `shim_errno.h` holds
the one `errno` cell all shims write (glibc's `errno` is `*__errno_location()`), so the eventual
all-shims-in-one-TU Postgres build has a single definition. Tests: `demo_pg_string_vs_native` (byte-exact
vs glibc over signs/bases/prefixes/endptr/ERANGE, bounded copies, tokenizing — `(int)`-printed so the
64-bit results and their truncation agree even on the clamped cases) and `demo_pg_procstub` (the guest's
fixed stub report). **svm-llvm lane green (fmt + clippy `-D warnings` + both tests).** The differential
earned its keep again — a missing `__errno_location` (only `os_shim.c` had defined it) trapped the string
guest at runtime until the shared `errno` header. Next (gap #11e): stdio `FILE*`, `strftime`, the `scanf`
family byte-exact vs native, then storage manager / WAL / shmem / catalog bootstrap.

**Slice CD (DONE) — guest libc: file stdio + time + wide-char (a bundle).** **(1) file stdio** —
`stdio_shim.c` layers the buffered `FILE*` surface Postgres actually declares (`fopen`/`freopen`/`fclose`/
`fread`/`fwrite`/`fgetc`/`getc`/`fgets`/`fputc`/`fseek`/`fseeko`/`ftell`/`feof`/`ferror`/`clearerr`/
`fflush`/`fileno`/`setvbuf`/`ungetc`) on `os_shim.c`'s fs-cap syscalls — a `FILE` is just an fd + EOF/
error flags + a one-byte ungetc slot, so it bottoms out in `open`/`read`/`write`/`lseek`/`close`; no
buffering of its own (the cap is the boundary, so `fflush`/`setvbuf` are inert). Notably the externs are
all *file*-oriented — Postgres declares **no** `fprintf`/`fscanf`/`fputs`, so this needs no varargs and
no `stdout`/`stderr` `FILE*` (those must reach the powerbox Stream cap, deferred). A real bug the
differential caught: the shim first passed the cap's `FS_O_*` bits to `open()`, which itself re-maps
`<fcntl.h>` `O_*` → `FS_O_*`, so a `"w"` fopen decoded as `O_RDWR` with no create and every op failed;
fixed by building real `O_*` flags. **(2) time** — `time_shim.c` provides `gmtime`/`gmtime_r`/`localtime`
(the sandbox is UTC, so localtime==gmtime) as pure calendar math (Hinnant civil-from-days) + a `strftime`
format engine (the common `%Y %m %d %e %H %I %M %S %j %w %u %p %a %A %b %B %C` conversions). **(3)
wide-char** — `libc_shim.c` gains the C-locale byte↔wchar identity `mbstowcs`/`wcstombs`. Differentials:
`demo_pg_stdio_vs_native` (write→reopen→`fgets`/`fgetc`/`ungetc`/`fread`/`fseek`/`ftell`/`feof`, byte-exact
over **both** `mem_fs` and `host_fs`) and `demo_pg_time_vs_native` (five epochs incl. two leap days,
TZ-independent conversions, + a wide-char round-trip; pure, on the bare powerbox). **svm-llvm lane green
(fmt + clippy `-D warnings` + both tests).** Next (gap #11f): the stream `FILE*` (`stdout`/`stderr` via an
on-ramp fd-dispatch) and the varargs `fprintf`/`scanf` engines, then storage manager / WAL / shmem /
catalog bootstrap.

**Slice CE (DONE) — stream `FILE*` + the `write`/`read` fd-dispatch (a small on-ramp change).** The
runtime's fd-multiplexing seam: `stdout`/`stderr`/`stdin` must reach the powerbox **`Stream`** cap while
`fopen`'d files reach the **fs** cap — but a guest that *defines* `write`/`read` (the syscall shim must,
to serve file fds) shadows the on-ramp's `Stream` recognizer, and `__vm_host_call` only reaches
`HOST_FN` caps, so the guest had no way to reach the streams. Fix, three coordinated pieces: **(1)
on-ramp** — two new builtins `__vm_stream_write`/`__vm_stream_read`, added as `cap_spec` entries with
`drop_args: 0` (so `lower_io_call` emits the same `Stream.write`/`read` `CallImport` on the stashed
stdout/stdin handle that the `write`/`read` recognizers do, but taking the `(buf, len)` slice as-is), plus
a `register_vm_stream_imports` scan that registers the `write`/`read` imports *even when the guest defines
`write`/`read`* (the reserved builtin names are never shadowed). Purely additive — touches neither the
verifier nor the confinement masking. **(2) guest** — `os_shim.c`'s `write`/`read` fd-dispatch: fds
1/2 → `__vm_stream_write`, fd 0 → `__vm_stream_read`, everything else → the fs cap; `stdio_shim.c` defines
the `stdin`/`stdout`/`stderr` `FILE*` globals (fds 0/1/2). **(3) the collision** — the fs cap allocated
fds from 0, overlapping the stream fds, so a file could land on fd 1 and its writes misroute to stdout;
`alloc_fd` now **reserves 0/1/2** and files start at 3, making the two fd namespaces disjoint (a small
`fs.rs` change, transparent to SQLite/LMDB which treat the fd opaquely). Test `demo_pg_stream_vs_native`:
`stream_probe.c` writes to `stdout` (FILE*) *and* a real file and the guest byte-matches native glibc over
`mem_fs` + `host_fs`. **All 301 translate tests green + svm-run suite + clippy/fmt on both crates** — the
on-ramp change is regression-free. Next (gap #11g): the varargs `fprintf`/`scanf` engines, then the
storage/WAL/shmem/catalog subsystems, then the first real **boot** (relink the shims into the Postgres
bitcode and drive `postgres --single` to its first live trap).

**Slice CF (DONE) — ★ the first boot: the module RUNS.** With all the guest shims `llvm-link`ed into the
whole-program module (`link_shims.sh` builds `postgres_shimmed.bc`; `pg_shims.c` is the combined TU),
Postgres **translates, verifies, and executes real backend startup** — the whole thing runs in ~24 s
(translate included) on the bytecode engine, no JIT-of-15k-functions wall. Driving it (`translate_bc_path_
with_options` with `stub_unresolved_externs: true` → `instantiate` → `run_with_caps` with the `fs` cap on
the natively-`initdb`'d data dir + `SELECT 1+1;` on stdin) surfaced the first **live** fault: a
`MemoryFault` in `save_ps_display_args`, which walks the C **`environ`** global — undefined in the guest
(the powerbox passes env via the §3e args buffer, not the C `environ` vector). Defining `environ` (empty)
cleared it, and the fault advanced to `Unreachable` stub-traps deeper in init. Added the early-startup
libc surface the boot needs: **`environ`**; the C-locale stubs **`setlocale`** (→ `"C"`) / `newlocale` /
`uselocale` / `duplocale` / `freelocale` / **`localeconv`** / **`nl_langinfo`** (`locale_shim.c`); the
**wide-ctype** `iswX`/`towX` family (ASCII classification, C locale — differential-tested
`demo_pg_wctype_vs_native` over every byte 0..255, the `iswX_l` variants forwarding to it); **`getopt`**
(+ its `optarg`/`optind`/… globals); and **`strsignal`**. The differential-clean parts (wctype) are
CI-tested; the rest are boot-support stubs validated by the boot advancing (the boot itself isn't a CI
test — it needs the cached, fetched Postgres bitcode). **svm-llvm lane green (fmt + clippy `-D warnings`;
8 `demo_pg_*` tests).** Next (gap #11h): a **trap-identifying** diagnostic (self-naming stubs or
partial-output-on-trap — the `Unreachable`/`MemoryFault` traps carry no name today, which makes the
remaining chase slow) so each stub-trap is legible, then the storage manager / WAL / single-process
shmem (`mmap`/`shmget`) / catalog bootstrap, plus the varargs `fprintf`/`scanf` engines. `strerror_r`
(GNU `char *` vs POSIX `int`) needs its own `_GNU_SOURCE`-isolated TU, deferred.

**Slice CG (DONE) — the boot diagnostic: guest output on trap.** Slice CF got Postgres *running* but its
`Unreachable`/`MemoryFault` traps carry **no name** — "guest trapped (Unreachable)" told me nothing about
*which* extern or *why*, so the boot chase was blind guessing (I burned real iterations on it). A trapped
program has usually already said what's wrong (a progress line, an `ereport`, an assertion), but the plain
`?` in `run_with_caps` dropped the `Host` — and its captured `stdout`/`stderr` — on the trap path.
`trap_err_with_output` now folds that captured output (tail-bounded) into the trap error, so the trap is
legible. **First payoff, immediately:** re-running the boot, the trap error now reads
`LOG:  could not find a "postgres" to execute` — Postgres's `find_my_exec` failing to resolve its own
binary path (`readlink("/proc/self/exe")` / argv[0]), the concrete gap #11i starts from. A small,
non-breaking `svm-run` change (the trap still returns `Err`, just an informative one — no `Outcome`
variant churn, no caller changes). Test `trap_error_surfaces_guest_output`: a guest writes a marker then
calls an undefined extern (a `--stub-externs` `unreachable` stub), and the trap error must carry the
marker. **svm-run + svm-llvm suites green, fmt + clippy `-D warnings` on both.** Next (gap #11i): resolve
`find_my_exec`, then drive the now-legible boot forward — storage manager / WAL / single-process shmem /
catalog bootstrap, plus the varargs `fprintf`/`scanf` engines.

**Slice CH (DONE) — the varargs `printf` engine (gap #11g/#11j).** The *output* half of the boot's
remaining format surface (the `scanf` input half stays deferred). Slice CE gave the on-ramp the
stream/file fd-dispatch (`stdout`/`stderr` → the powerbox out-`Stream`, `fopen`'d files → the fs cap,
disjoint fd namespaces); this slice adds the runtime `printf`/`fprintf`/`vfprintf`/`vprintf`/`snprintf`/
`sprintf` family Postgres formats its query results and `elog`/`ereport` log lines with — a query result
builds its directives *at runtime*, the path the on-ramp's translate-time *constant*-format engine can't
lower. **Guest code, no translator change** (the CA–CG shim model):
- **`printf_shim.c`** (new) is the runtime `vsnprintf` engine — the exact byte-for-byte-vs-glibc
  formatter from the Lua `string.format` fixture (`lua_fmt_snprintf.c`): integers/strings/chars/pointers/
  `%a` hex-float formatted in C; `%f`/`%e`/`%g` delegate to the on-ramp's correctly-rounded **bignum
  `__vm_fmt_{fix,sci,gen}` dtoa**; full width/precision/flag/length-modifier support. The stream variants
  format into a buffer (heap for >1 KiB output) and `fwrite` it, so they **compose** with slice CE's
  fd-dispatch: `fprintf(stdout/stderr, …)` → the out-`Stream`, `fprintf(file, …)` → the fs cap. Defining
  these **shadows** the on-ramp's constant-format `printf`/`snprintf` synthesis — deliberate: one runtime
  engine, all formats. Linked into the whole-module boot via `pg_shims.c` (`#include "printf_shim.c"`).
- **Differential** `demo_pg_fprintf_vs_native` (`fprintf_probe.c`): the format family across
  `%d`/`%u`/`%x`/`%c`/`%s`/`%f`/`%e`/`%g` + width/precision/flags to **three targets** — `stdout`, a real
  **file** (fs cap, read back and echoed), and `stderr` — byte-identical to the native glibc oracle (which
  folds `stderr` into `stdout` unbuffered to match the guest's single write-through Stream) on **all three
  engines**, over `mem_fs` and `host_fs`. **svm-llvm lane green (9 `demo_pg_*` tests, fmt + clippy
  `-D warnings`).** Next (the format tail): the `fscanf`/`sscanf`/`vsscanf` **input** engine (gap #11i's
  remaining half), then the boot's storage manager / WAL / single-process shmem / catalog bootstrap.

**Slice CI (DONE) — past `find_my_exec` + the single-process IPC collapses.** The CG diagnostic named the
first gap (`could not find a "postgres" to execute`); this slice clears it and the shared-memory setup
right behind it. **`find_my_exec`:** Postgres resolves its own binary from `argv[0]` — with a slash it
`stat`s/`access(X_OK)`es that path directly (both already shimmed over the fs cap), without one it searches
`$PATH`. Driving the boot with a slashed `argv[0]` (`./postgres`) and an executable `postgres` file in the
data dir lets `validate_exec` succeed, and the boot advances past it. **The IPC collapses (`ipc_shim.c`):**
`postgres --single` still stands up its shared-memory segment (buffer pool, lock tables, …) and semaphores
in early startup — but in *one* process there is nothing to *share*. So shared memory is just anonymous
memory (`mmap(MAP_ANONYMOUS)` → `malloc` + zero; `munmap` → `free`), `shmat` returns `(void *)-1` to force
Postgres onto that anonymous-mmap path, and the unnamed POSIX semaphores are uncontended no-ops
(`sem_init`/`post`/`wait`/… → `0`). `madvise`/`mlock`/`posix_fadvise`/`posix_fallocate` are no-ops; the
System V and `shm_open` surfaces collapse likewise. The one collapse with observable behavior — anonymous
`mmap` must return **zeroed, writable, byte-stable** memory — is differential-tested: `mmap_probe.c` +
`demo_pg_mmap_vs_native` map 4 KiB, assert freshly zeroed, write `i*7+3`, read it back, `munmap`, and the
guest output must byte-match native (`mmap_ok=1 zeroed=1 held=1 munmap=0`). The `sem_*`/`shm*` no-ops have
no observable output, so they're exercised by the boot rather than a differential. **Boot state:** past
`find_my_exec` and into shared-memory/semaphore init; the next fault is a **silent** `Unreachable` (no
guest output before it), which the CG diagnostic can't name because nothing was printed — so gap #11k is
**self-naming stub-traps** (make the `--stub-externs` trap carry the missing extern's name) to make the
remaining silent traps legible, then forward through the storage manager / WAL / catalog bootstrap.
**svm-llvm lane green (fmt + clippy `-D warnings`; 9 `demo_pg_*` tests).**

**Slice CJ (DONE) — the varargs `scanf` engine + a real `strtod` (gap #11l).** The *input* twin of CH:
the runtime `sscanf`/`vsscanf`/`fscanf`/`vfscanf`/`scanf`/`vscanf` family Postgres parses config values,
version strings, and numbers with (a format built at runtime — no translate-time analog). **Guest code,
no translator change:**
- **`scanf_shim.c`** (new) drives one **char-source abstraction** — a string (`sscanf`) or a `FILE*`
  (`fscanf`, via `fgetc`/`ungetc`, so it composes with the CE stream/file fd-dispatch) with a single
  pushback slot (scanf only ever un-reads the one char that ended a conversion). Conversions: `d`/`i`/`u`/
  `o`/`x`/`X`, `c`, `s`, `f`/`e`/`g` (+ `a` hex-float), `[scanset]` (with `a-z`/`0-9` **range** expansion),
  `n`, `p`, `%%`, with assignment-suppression `*`, field width, and `h`/`hh`/`l`/`ll`/`L`/`j`/`z`/`t`
  length modifiers; the return value is the assigned-item count (EOF before the first conversion) — glibc
  semantics. Integers accumulate inline (every `va_arg` in the `va_list`-owning function — passing a
  `va_list *` to a helper trips the on-ramp's varargs ABI, the one gotcha this shook out, shared with
  `printf_shim.c`).
- **A real `strtod` for `%f`/`%lf`/`%g`.** The on-ramp's `strtod` is a **trap stub** (it was never a real
  impl — a latent gap this slice surfaced, and one `float8in` hits at boot too), so the guest brings the
  correctly-rounded **bignum `strtod.c`** (`demos/strtod/`, already used by float Lua) — a guest
  definition shadows the stub. `scanf`'s float conversions collect a token and hand it to `strtod`.
- Both are linked into the whole-module boot via `pg_shims.c` (`#include "../strtod/strtod.c"` +
  `"scanf_shim.c"`), so Postgres' own `sscanf`/`strtod` become real there.
- **Differential** `demo_pg_sscanf_vs_native` (`sscanf_probe.c`): the conversions **and** the return-count
  semantics (a scanf differential must check the count, not just the values — partial-match/EOF cases
  included), plus an `fscanf`-from-`stdin` half, byte-identical to native glibc on **all three engines**.
  **svm-llvm lane green (10 `demo_pg_*` tests, fmt + clippy `-D warnings`).** Next: the boot's storage
  manager / WAL / single-process shmem / catalog bootstrap (the `llvm-postgres` thread, past slice CI).

**Slice CK (DONE) — self-naming stub-traps: the silent trap names itself (gap #11k).** The boot's early-init
fault was a **silent** `Unreachable` — no printed output, so the CG output-diagnostic couldn't name it, and
the chase was back to guessing which of ~96 stubbed externs the live path hit. The fix names the stubs so
the trap names itself, **without touching the confinement path**: under `--stub-externs` the on-ramp now
records each stub's `(func idx → missing-extern name)` in the §6 **function-name table**
(`DebugInfo.func_names` — strippable, verifier-ignored, §2a), so the interpreter's trap-time backtrace
resolves its innermost frame (via `svm_interp::func_name`) to the extern's name. The stub body is
*unchanged* — a pure `Unreachable`, never a capability call — so nothing in the sensitive lowering moves.
Test `stub_trap_names_the_extern`: a guest calls an undefined `frobnicate_widget` under `--stub-externs`, a
traced run traps `Unreachable`, and the innermost backtrace frame must resolve to `frobnicate_widget`.
**svm-llvm lane green.** This turned the boot chase into a tight run → read the named frame → close the gap
→ repeat loop, which drove slice CL.

**Slice CL (DONE) — drive the named boot to real logging (a bundle).** With every trap self-naming (CK),
this slice bundles the whole run of gaps between `find_my_exec` and Postgres's first real log output. **One
on-ramp fix:** `__sigsetjmp` (the libc symbol the `sigsetjmp` macro expands to — Postgres' `PG_TRY`) joins
the `setjmp`/`sigsetjmp` recognizer, lowering to the same `SetJmp` core op (test `sigsetjmp_recognized`).
**The rest are guest shims** (`#include`d into `pg_shims.c`), each named by the diagnostic as the boot
advanced: **`realpath`** (canonicalize an existing path to absolute over the fs cap); a small mutable
**environment** (`setenv`/`getenv`/`unsetenv`/`putenv`); **`getpwuid`**/`getpwnam` (the bootstrap
superuser's name — a fixed non-root identity); a deterministic **`random`**/`srandom`/`rand`/`srand`; the
**event-loop backend** (`event_shim.c`: `signalfd`/`epoll_*`/`eventfd` → distinct fake fds + no-op
registration — `latch.c`'s `WAIT_USE_EPOLL` set-up, never blocked in a synchronous single process); real
**`memcpy`/`memmove`/`memset`** (`mem_shim.c`, `-fno-builtin-mem*`) so an *address-taken* one resolves to a
real function instead of the funcref trap stub; **`pow`** (`math_shim.c` — exact binary-exponentiation for
the `pow(10,n)` datetime case); and **`strerror`** + the GNU **`strerror_r`** (the latter in its own
`_GNU_SOURCE`-isolated TU, `strerror_shim.c`, so its `char *` prototype doesn't perturb `__isoc23_*`/
`getrlimit` across the shared shim TU — the split earlier slices deferred). Composes with slice CJ's real
`strtod`/`scanf`. **Payoff:** the boot now runs real backend startup through GUC init and **config-file
processing**, emitting fully-formatted, timestamped Postgres log lines (`… GMT [1] LOG: …` / `FATAL:
configuration file "postgresql.conf" contains errors`) — no trap, a clean `Exited(1)`. It stops on a
legible, *real* Postgres error: it can't read its **timezone data directory** (`…/share/postgresql/timezone`,
an absolute path outside the fs-cap root → the cap correctly denies it). That's a capability/data-provisioning
question (gap #11m), not a shim gap. **svm-llvm lane green (fmt + clippy `-D warnings`; the on-ramp changes
CI-tested — `stub_trap_names_the_extern`, `sigsetjmp_recognized`; the boot-support shims validated by the
boot advancing; the boot itself isn't a CI test — it needs the fetched Postgres bitcode).**

**Slice CM (DONE) — ★ the boot runs backend init, shared memory, and WAL crash recovery (gap #11n).**
Past the timezone stop (CL's #11m), a run→name→fix loop — the CK self-naming diagnostic on the *fast*
bytecode engine — carried the boot all the way through real database startup to a **recovered cluster**:

- **The fs-cap root is the guest filesystem root.** Postgres opens its install tree by *absolute* path
  (`<prefix>/share/postgresql/timezone`, `…/timezonesets`, …) but the `fs` cap is rooted (rejects
  absolute + `..`). `os_shim`'s path wrappers now map a guest-absolute path to cap-relative (strip the
  leading `/`), so "/" *is* the cap root — confinement unchanged (`..` still forbidden), the sandbox just
  provides the install tree under the root. This is the right model for the eventual browser demo (a
  virtual sysroot). `fill_stat` reports every file owned by the sandbox identity (uid/gid 1000, matching
  `proc_shim`'s `geteuid`) so `checkDataDir`'s ownership check passes.
- **Shared memory + the DSM collapse.** The tiny SysV interlock segment now genuinely attaches (`shmat`
  returns real zeroed memory, sized by `shmget`), and — with `dynamic_shared_memory_type = sysv` (a single
  process needs no cross-process POSIX shm) — dynamic-shared-memory segments come through the same
  multi-segment SysV table; `shm_open` fails cleanly (ENOSYS).
- **★ Real counting semaphores — the bug that "hung" the boot.** The semaphore shim was a blanket no-op,
  but Postgres' `PGSemaphoreReset` *drains* with `while (sem_trywait(s) >= 0) ;` — a `sem_trywait` that
  always succeeds spins forever (the boot's apparent hang was this, in `InitProcess`). Now a real counting
  semaphore kept in the `sem_t` (`sem_trywait` fails EAGAIN at zero). **Differential** `demo_pg_sem_vs_native`
  (byte-exact vs glibc's single-process unnamed semaphore).
- **Startup libc odds and ends:** `__isoc23_*` `scanf` aliases (glibc-C23 — `ValidatePgVersion` reads
  `PG_VERSION`); `sync_file_range` (a writeback *hint* → no-op).
- **Demo cluster config** (in the fetched data dir, documented in the demo README): `timezone = GMT`
  (`tzparse`, no tz-data file), `fsync = off` (skip the startup data-dir sync), `shared_buffers = 1MB` +
  `max_connections = 10` (so buffer/shmem init is cheap on the interpreter).

**Payoff:** the boot now runs real backend startup, opens the catalog (`InitPostgres`/relcache), and does
**WAL crash recovery** — its log reads `database system was not properly shut down; automatic recovery in
progress` / `redo starts at 0/147A9F0` / `redo done` / `checkpoint starting: end-of-recovery`. It stops in
the end-of-recovery checkpoint (`CheckPointGuts → ProcessSyncRequests`) on a **`fn0` funcref** — a
hashtable function pointer resolving to index 0 (a real on-ramp funcref bug, not a shim gap — gap #11o,
its own slice). **svm-llvm lane green (fmt + clippy `-D warnings`; 12 `demo_pg_*` incl. the new
`demo_pg_sem_vs_native`).** The boot reaches recovery in ~74 s on the bytecode engine — *boot speed* (a
cold 15k-function boot) is now the gating concern for a browser demo, pointing at snapshot/restore of the
post-boot state rather than cold-booting each time.

**Slice CN (DONE) — ★★★ `SELECT 1+1` → `2`: PostgreSQL runs in the sandbox (gap #11o, the capstone).**
The end-of-recovery checkpoint trapped on a `fn0` funcref (CM's #11o): `ProcessSyncRequests` →
`hash_search_with_hash_value` → an indirect call to index 0. **Root cause** (reproduced by a minimal
*non-devirtualizable* indirect `memcmp` — a runtime-selected + `volatile` funcptr, so clang can't fold the
call back to a direct one): dynahash's `HASH_BLOBS` table stores `hashp->match = memcmp` and calls it
*through the pointer*, but `memcmp` is **only synthesized for direct calls, never defined** — so its
address-taken funcref falls to a fail-closed trap stub. (Slice CL had defined `memcpy`/`memmove`/`memset`
for exactly this, but missed the comparators.) **Fix:** `mem_shim.c` now also defines `memcmp`, `strlen`,
`strcmp`, `strncmp` (the string comparators dynahash / string-keyed tables take the address of), so the
taken address points at a real libc-ABI function; direct calls still fast-path through the synthesizer.
Differential `demo_pg_funcptr_vs_native` (each builtin called through a `volatile` pointer, byte-exact vs
glibc). **Payoff — the whole thing runs:** with the fix, `postgres --single` completes WAL recovery, the
end-of-recovery checkpoint, reaches the interactive `backend>` prompt, **parses + plans + executes
`SELECT 1+1;` and prints the correct result `2` (type `int4`)**, then shuts down cleanly (`Exited(0)`):

```
PostgreSQL stand-alone backend 17.5
backend>  1: ?column? = "2"  (typeid = 23, len = 4, typmod = -1, byval = t)
```

That is ladder-#7 (`DESIGN.md` §"Suggested ladder") delivered: the full PostgreSQL 17.5 backend — 832
modules / ~14 985 functions — **translates, verifies, and runs a real query** on the SVM bytecode engine
under the `fs` + powerbox capabilities, no ambient authority. **svm-llvm lane green (fmt + clippy
`-D warnings`; 13 `demo_pg_*` incl. the new `demo_pg_funcptr_vs_native`).** Next: *boot speed* — a cold
boot is ~100 s on the interpreter, so a browser demo wants **snapshot/restore** of the post-recovery state
(the §durability machinery) rather than a cold boot each time; and widening the SQL surface past the
constant-folded `SELECT` (real tables, DDL, the `scanf`/`fscanf` input path under load).

**Slice CO (DONE) — ★★ a real table round-trip: `CREATE TABLE` / `INSERT` / `SELECT * FROM t` (gap #11p).**
Widening past the constant-folded capstone to a query over a *real* table. Driving `CREATE TABLE t(x int,
s text)` + `INSERT`s + `SELECT * FROM t` through the boot got the DDL/DML working immediately (the heap
read/write/`UPDATE`/`DELETE`/`WHERE` paths all ride the machinery WAL recovery already exercised) — but
`ORDER BY x DESC` **trapped `Unreachable`** at *planning* time. A traced, name-resolving boot run put the
innermost frame at an address-taken `fn0` called from `cost_tuplesort` (`cost_sort` → `create_sort_path`
→ … → `standard_planner`): the planner's sort-cost model computes `N·LOG2(N)` comparisons, and `LOG2(x)`
is `log(x)/log(2)`. **Root cause: `log` was an undefined external → a fail-closed trap stub.** In fact the
*entire* transcendental surface (`log`/`exp`/`sin`/`cos`/`pow`/…) was undefined — `postgres_libm.bc` was
byte-identical to the pre-libm `postgres.linked.bc`; slice BQ built the bundled-openlibm *mechanism* and
its bit-exact differential (`libm_bundled_vs_native`), but it was **never wired into the boot's link
step**. The capstone's `SELECT 1+1` simply never reached a transcendental. **Fix:** `link_shims.sh` now
`llvm-link`s openlibm's double set (the 28-file `OPENLIBM_SRCS` — the 18 transcendentals + kernels) into
`postgres_shimmed.bc`, so `log`/`exp`/`pow`/… are defined guest code (`sqrt`/`fabs`/`ceil` still lower to
on-ramp float ops); the hand-written `pow` shim (`math_shim.c`) is retired in favor of openlibm's bit-exact
`e_pow`. With it the whole round-trip runs clean (`Exited(0)`): `SELECT *`, `count`/`sum`/`avg` (numeric),
**`ORDER BY … DESC`**, `UPDATE … WHERE`, `DELETE … WHERE`, and `GROUP BY … ORDER BY` all return correct
rows. openlibm is fetched-not-vendored (BSD); `link_shims.sh` takes a pre-staged tree (`SVM_OPENLIBM_DIR` /
`/tmp/openlibm`) when github egress is blocked. **svm-llvm lane green (fmt + clippy `-D warnings`; 13
`demo_pg_*`).** Next: still *boot speed* (snapshot/restore), and wider SQL (joins, indexes, subqueries).

**Slice X (DONE) — `realloc` + signed `printf` `%d` (lands `sortvec`).** `__svm_malloc` now writes a
16-byte **size header** before the data (keeping it 16-aligned), so the header survives for
`realloc`. **`__svm_realloc(p, n)`** handles `realloc(NULL,…)` ≡ `malloc`, else `malloc`s `n`, reads
the old size from `p-16`, and `__svm_memcpy`s `min(old, n)` bytes (no overlap — the fresh block sits
above the old). `printf` gains signed `%d`/`%i`: the sign is computed (`-`), the magnitude formatted
via `__svm_utoa`, the `-` written just below the digits and included only when negative; plain and
space-padded fields supported (zero-padded `%d` stays fail-closed — sign+pad ordering). Tests:
`demo_sortvec_vs_native`, `printf_signed_formats`, `realloc_grow_preserves`. (heapgrow/calloc still
pass — the data region stays freshly-`vm_map`-zeroed below the bump.)

**Slice Y (DONE) — 128-bit SIMD (`<4 x float>` → native `v128`); lands `mat4`.** A 4-lane 32-bit
vector maps to SVM's §17 `v128` (vs the 2-lane → packed-`i64`, since `<4 x …>` is 16 bytes): `load`/
`store` → `v128.load`/`store`; `fadd`/`fsub`/`fmul`/`fdiv` → `f32x4` `VFloatBin`; `extractelement`/
`insertelement` → extract/replace lane; `shufflevector` → an `i8x16.shuffle` byte mask (an all-equal
mask is a splat/broadcast); `<4 x …>` constants → `ConstV128`; `llvm.fmuladd.v4f32` → `f32x4` mul+add
(unfused). The `<4 x i32>` shuffle masks are read as constants, not values. Tests:
`demo_mat4_vs_native`, `vec4_float_scale` (a `<4 x float>` by-value arg/return + splat-mul).

**Slice Z (DONE) — `llvm.bswap` (lands `crc32`).** No SVM byte-swap op, so it is synthesized inline:
each source byte `i` is moved to destination byte `nbytes-1-i` via `((v >> 8*i) & 0xff) << 8*(nbytes-1-i)`,
OR-accumulated (`i16`/`i32`/`i64`; `emit_bswap`). Tests: `demo_crc32_vs_native` (CRC-32 + a
`__builtin_bswap32` big-endian reader, with stdin) and `bswap_intrinsic` (bswap32/64 vs native).

**Slice AA (DONE) — overlap-safe `memmove` (lands `lineedit`).** A variable-length (or
oversized-constant) `llvm.memmove` now calls the synthesized **`__svm_memmove(dst, src, len)`** — an
8-block, direction-aware counted byte copy: when `dst <=u src` it copies **forward** (`i = 0…len`),
otherwise **backward** (`i = len…0`), so overlapping shifts are correct in either direction (the one
thing `memcpy`'s load-all-then-store inline path can't do for runtime lengths). The helper is
appended last in the fixed helper-index order (after `realloc`). Constant small `memmove` still
inlines (already overlap-safe). Tests: `demo_lineedit_vs_native` (right+left overlapping shifts, with
stdin) and `memmove_overlap_runtime` (both directions over an 8-byte window vs native).

**Slice AB (DONE) — transcendental libm, bundled as guest code (lands `raytrace`).** No new lowering:
math beyond the SVM float ops (`sin`/`cos`/`exp`/`pow`/…) is supplied *by the program* as ordinary
guest C (polynomial approximations), exactly as the corpus demos bundle their own `memset`. This is
deliberate — it keeps math **in the sandbox** (no host math capability), and it is what makes the
differential clean: native `cc` compiles the *same* guest `libm`, so every value is bit-identical.
The only machine float ops in play already match across backends — `sqrt`/`floor` lower to SVM ops
(slices F/L; IEEE-exact, matching native libm), `fmuladd` is unfused on both sides, and `+−*∕` are
plain IEEE. The `raytrace` demo (one unit sphere, `sqrt` ray-sphere intersection, diffuse +
`g_sin` surface bands + `g_exp` rim falloff, rendered to a char ramp) comes out byte-identical to
native. A note on the harness: native links now pass `-lm` so libm-calling demos link (`sqrt`/`floor`
become real calls at the native build's `-O0`); harmless for the rest. Tests: `demo_raytrace_vs_native`
and `guest_libm_transcendental` (a guest `exp` + the `sqrt` op over a damped wave's RMS).
- *Deferred:* a transcendental as an **external** libm call (e.g. linking the system `sin`) stays
  `Unsupported` — there is no host math capability and no SVM op for it; the program must bring its
  own (this slice). Adding a *bundled* guest `libm` header the on-ramp injects automatically (so
  unmodified code that calls `sin` links against guest poly code) is the natural follow-up.

**Slice AC (DONE) — the `<svm.h>` capability/concurrency/GC builtins (P0+P1+Memory).** The
low-level SVM surface the chibicc frontend exposes through `<svm.h>` (`frontend/chibicc/codegen_ir.c`,
the oracle) now lowers on the LLVM on-ramp too, so a guest *language* emitting LLVM bitcode (e.g. a
JACL runtime) reaches the VM's fibers, threads, atomics, futex, conservative GC roots, direct window
memory management, and capability reflection — **no host math/scheduler capability, just the existing
primitives**. Each builtin is a call to a declared-only `extern` of a fixed name (`lower_vm_builtin`,
gated on the name being external so a guest definition shadows it, like the libc/libm rules) and
mirrors the chibicc lowering exactly:
- **§3e/§4 Memory** — `__vm_map`/`__vm_unmap`/`__vm_protect`/`__vm_page_size` → `CallImport` on the
  stashed `Memory` handle (slot 12; imports `vm_map`/`vm_unmap`/`vm_protect`/`vm_page_size`, resolved
  by `default_cap_resolver`). The synthesized `_start` now grants the 4th (`Memory`) handle when a
  program uses **either** `malloc` **or** a direct Memory builtin — the heap is seeded only for
  `malloc`, so the powerbox **contract/stash layout is unchanged** (existing demos byte-identical).
- **§12 fibers** — `__vm_fiber_new`/`resume`/`suspend` → `cont.new`/`cont.resume`/`suspend` (the
  funcref is `i32.wrap`'d; `resume` stores its `(status, value)` status through `*done`).
- **§12 threads** — `__vm_thread_spawn` (a *direct* funcref → the static `thread.spawn` funcidx) /
  `__vm_thread_join`; **atomics** — `__vm_atomic_{add,load,store}`(`32`)/`cas32` → the `iN.atomic.*`
  ops (seq-cst); **futex** — `__vm_wait32`/`__vm_notify` → `i32.atomic.wait`/`atomic.notify`.
- **§12 per-vCPU TLS** — `__vm_vcpu_tls_get()` / `__vm_vcpu_tls_set(x)` → `vcpu.tls.get`/`vcpu.tls.set`
  (the current vCPU's i64 register, seeded to the dense vCPU id — so `get` doubles as `vcpu.id`).
- **§GC** — `__vm_gc_roots(lo, hi, mask, buf, cap)` → `gc.roots` (conservative root enumeration; `mask`
  is the §GC tagged-pointer payload mask, top-byte-strip only — pass `~0UL` for untagged).
- **§7 reflection** — `__vm_cap(i)` reads the handle stash (`i32.load` at `i*4`); `__vm_cap_count`/
  `__vm_cap_at` → `cap.self.count`/`cap.self.get`.
Tests (`translate.rs`): `vm_memory_map_and_page_size` (map a page at 256 MiB + page-size, end-to-end
on the JIT powerbox), `vm_fibers_generator`, `vm_atomics_single_threaded` (both backends),
`vm_futex_wait_notify`, `vm_threads_atomic_counter` (4×500 = 2000 on the M:N executor),
`vm_gc_roots_smoke`, and `vm_cap_reflection` (8 granted caps) — interpreter-only where the JIT bails
`Unsupported` on fibers/`cap.self` (mirrors the chibicc test split).
- **Stash layout locked to 8 handles (done — follow-up).** The handle stash is now the fixed region
  `[0, HANDLE_REGION_END) = [0, 32)` — one `i32` slot per `VM_CAP_*` index — with the allocator's
  heap state, the `putc` scratch, and the `printf` format buffer all relocated **above** it
  (`HEAP_BRK`=32/`HEAP_TOP`=40/`STASH_SCRATCH`=48; `FMT_BUF` unchanged at 64). So offsets `16/20/24/28`
  are reserved for the AddressSpace/IoRing/Blocking/Jit tail and granting it later needs **no further
  relocation** — the one-time fix that forecloses the recurring "new handle collides with heap state"
  bug. All six demos + the malloc/printf paths re-verified byte-identical (the offsets are referenced
  by named constant, so the move is transparent).
**Slice AD (DONE) — the async-I/O ring (P2; the STW-safe blocking path JACL's GC needs, GC.md §5.2).**
The §9/§12 submit/complete ring now lowers on the on-ramp, so a guest event-loop / work-stealing
runtime built on LLVM bitcode drives many concurrent blocking I/Os from one parked vCPU:
- `__vm_io_submit_async(sq, n, counter)` / `__vm_io_reap(cq, max)` → `CallImport` on the stashed
  `IoRing` handle (slot 5; imports `vm_io_submit_async`/`vm_io_reap` → `IoRing` ops 1/2). The SQE/CQE
  wire format is guest-built in the window; only the ring indices cross the boundary.
- `__vm_blocking_handle()` → a stash read of the `Blocking` handle (slot 6) the guest names in an SQE.
- **`synth_start` generalized to grant a contiguous handle prefix.** It now grants `n_handles` —
  sized to the highest `VM_CAP_*` index the program uses (exit→3, memory→4, ioring→6, blocking→7) —
  and stashes each at `i*4` with a uniform loop (the old 3/4-handle special-case is gone; existing
  I/O and malloc programs still get exactly 3/4, demos byte-identical). This also lights up
  `__vm_cap(i)` for `i ≥ 4`: the tail handles are now stashed, so the generic reader reaches them
  (`vm_cap_index_reaches_tail_handles`: `__vm_cap(6) == __vm_blocking_handle()`).
- Tests: `vm_async_io_runtime` runs `demos/async_io/async_io.c` through Lane C on the interpreter's
  M:N executor + offload pool (the 7-handle powerbox, futex parking, completion-order-invariant total
  Σ mix(0..8)) — interpreter-only, as the JIT async path needs the separate `HostAsyncHooks` harness
  (mirrors the chibicc `run_async_demo` split). 86 translate tests green, fmt + clippy clean.
- *Still deferred:* the §13/§14 `__vm_region_*` (SharedRegion) builtins stay `Unsupported` until a
  workload needs them — the reserved AddressSpace(4) slot is ready.

**Slice AE (DONE) — guest-driven JIT (§22).** The `Jit` capability now lowers on the on-ramp, so a
guest that emits serialized SVM IR at runtime (a language runtime accelerating its own bytecode)
reaches it from LLVM bitcode: `__vm_jit_compile`/`invoke2`/`release`/`install`/`uninstall`/
`compile_linked` → `CallImport` on the stashed `Jit` handle (slot 7; imports `vm_jit_*` → `Jit` ops
0/1/2/3/4/5). A JIT-using program is granted the **full 8-handle powerbox** (`Jit` is the last
`VM_CAP_*` index; `synth_start`'s contiguous prefix now reaches 8, so `run_powerbox` grants `Jit` with
its validator + call_indirect table). The host verifies + Cranelift-compiles the submitted blob into
*this* domain — same window, same powerbox; verification, not isolation, is the boundary (§2a).
- Tests: `vm_jit_builtins_lower_and_grant_full_powerbox` (structural — every builtin → its `Jit`
  `CallImport`, 8-handle entry); `vm_jit_guest_self_jit_demo` runs the real `demos/jit/jit_demo.c`
  through Lane C on the JIT powerbox — the guest emits IR byte-by-byte, `compile`s + `invoke2`s a raw
  unit **and** an `install`ed unit reached via a C function pointer (`call_indirect`), agreeing with
  its own bytecode interpreter on a 49-input grid. The validator's memory-match is exact, so the test
  probes svm-llvm's parent `size_log2` and patches the demo's blob descriptor to it (no magic
  constant). 88 translate tests green, fmt + clippy clean.
**Slice AF (DONE) — SharedRegion (§13/§14); completes the `<svm.h>` surface.** Guest-minted shareable
memory now lowers on the on-ramp: `__vm_region_create(len)` mints a region from the stashed
`AddressSpace` handle (slot 4; import `vm_region_create` → `AddressSpace` op 5) and returns a **region
handle**; `__vm_region_map`/`unmap`/`page_size` then `cap.call` *that* region handle (their first C
arg — not a stash slot; imports → `SharedRegion` ops 0/1/3) to alias the region's bytes into the
window. A region-minting program is granted the **5-handle powerbox** (`synth_start`'s prefix reaches
AddressSpace). This is the magic-ring-buffer / zero-copy parent↔child data plane (DESIGN §13/§14).
- Test: `vm_region_magic_ring_buffer` — a guest mints a 64 KiB region, maps it at two adjacent window
  offsets, and a single 8-byte store straddling the seam wraps tail→head as one contiguous access
  (then `unmap`s), on the real JIT powerbox (true shared-memory aliasing via the host region factory);
  the `'Y'` success marker is checked against stdout. 89 translate tests green, fmt + clippy clean.

The `<svm.h>` capability/concurrency/GC/JIT/region surface is now **complete** — the LLVM on-ramp has
full capability parity with the chibicc frontend. The next frontier is the D54 **breadth proof**
(Milestone 2): the on-ramp consumes *any* LLVM frontend's bitcode, so other languages run with no
translator change beyond what the C corpus proved.

**Slice AG (DONE) — C++ first light (the breadth proof begins).** A freestanding C++ TU compiled
`clang++ -O2 -fno-exceptions -fno-rtti` runs **byte-identical to native `clang++`** through the on-ramp
— the first non-C language. Mostly **free** (the C corpus already covers it): classes, inheritance,
**virtual dispatch** (vtables are function-pointer global initializers, slice K → loaded + `call_indirect`,
slice G), the `this` pointer, mangled names, **templates** (monomorphize to ordinary functions), and
heap **`new`/`delete`** (the program defines `operator new`/`delete` over the guest `malloc`/`free`),
including **virtual destructors** (the deleting-dtor chain through the vtable). The one real gap closed:
- **C++ static initialization (`@llvm.global_ctors`).** clang emits a `_GLOBAL__sub_I_*` runner for
  global objects with non-trivial ctors; the on-ramp jumped straight to `main`, so static init never
  ran. Now `globals_layout` skips the `llvm.*` reserved globals (metadata, not window data) and
  `collect_global_ctors` extracts the ctor funcrefs in priority order; the synthesized `_start` calls
  them (each `(i64 sp) -> ()`) **before** `main`, exactly as native ([basic.start]). A program with
  global ctors now forces a powerbox `_start` even with no other capability use.
- Tests (`translate.rs`): `cpp_virtual_dispatch_first_light`, `cpp_new_delete_virtual_dtor_templates`
  (heap `new`/`delete` + virtual dtor + a template), `cpp_global_constructor_runs_before_main` (a
  side-effecting global ctor prints before `main`) — all vs native `clang++`.

**Slice AH (DONE) — Rust through the on-ramp + non-power-of-two integers (`iN`).** A `no_std`/
`panic=abort` Rust crate now runs **byte-identical to native `rustc`** — the second non-C frontend.
- **Toolchain pin (a §2 "pin, don't drift" decision).** `rustc` bundles its own LLVM, so the bitcode
  version must match our pinned reader (LLVM 18). The container's default `rustc` ships **LLVM 21**
  (rejected by `llvm-ir`'s `llvm-18`), and re-pinning the *reader* to 21 is blocked here (no
  `llvm-21-dev`/`llvm-config-21`; `llvm-ir` tops out at LLVM 19). The resolution is the reverse: pin a
  **Rust 1.81 toolchain (LLVM 18.1)**, which emits bitcode the existing reader accepts — *no re-pin*.
  (CI must `rustup toolchain install 1.81.0`, as it installs `llvm-18-dev` for the bitcode lane.)
- **The one real gap, closed: non-power-of-two integers (`iN`).** `clang`/`rustc -O2` SCEV closes a
  counted loop into a **polynomial with `i33` intermediates** (holding `n·(n-1)·(2n-1)` before a
  magic-constant divide); slice A had deferred these. Now `val_type` maps `iN` (`33..=64`) to an `i64`
  container, `operand` materializes `iN` constants canonicalized, and `bin` **masks the result of the
  de-normalizing ops** (`add`/`sub`/`mul`/`shl`) back to `N` bits — so every `iN` value stays canonical
  (`mod 2ᴺ` = the exact wrap semantics) and downstream `lshr`/`trunc`/unsigned-compare see clean bits
  (the §3b widen-and-mask discipline, generalized from the existing `i8`/`i16` narrow collapse).
  `i128`+ stays a clean `Unsupported`; signed `iN` ops needing a sign-extended container
  (`ashr`/`sdiv`/`srem`/`sext`-to-`iN`/signed `icmp`-`iN`) are not yet emitted by the corpus — add on
  demand. This benefits **optimized C/C++ too** (any frontend's `-O2` produces `iN`).
- Test: `rust_no_std_matches_native` — the same `compute` (a sum-of-squares, which LLVM closes into the
  `i33` polynomial) compiled both as a `no_std` lib (→ bitcode → on-ramp, interp == JIT) and as a
  native std binary (the oracle), agreeing for `n` in `{5, 1000, 46341, 200000, -7}` — values chosen so
  the `i33` intermediate **overflows 33 bits and wraps**, which only the native differential validates
  (interp == JIT alone would agree even on a wrong mask). 93 translate tests green, fmt + clippy clean.

**Slice AI (DONE) — real `core` Rust + panic-path lowering.** Idiomatic `no_std` Rust runs
byte-identical to native `rustc` with **no translator change** beyond the C corpus — a `#[repr(u8)]`
enum dispatched by `match` (→ `switch`/`br_table`), fixed arrays + slice iteration, `Option` + `match`
(niche-optimized), iterator `find`/`map`/`max` with closures, by-value `struct`s + array-of-struct +
field access, and signed `/`/`%`. The one real gap closed:
- **Panic-path lowering (`-C panic=abort`).** A real Rust program is littered with non-elidable panic
  branches (div-by-zero, bounds, overflow) that call `core::panicking::*` — **external** (precompiled
  libcore), which the on-ramp's undefined-call path rejected, blocking essentially all real Rust. Now
  `is_rust_abort_call` recognizes those entry points (`panicking`/`unwrap_failed`/`expect_failed`/
  `slice_index`/`panic_cannot_unwind`) and, since they are `-> !` and always followed by `unreachable`,
  lowers the call to a **trap** (drop it; the trailing `unreachable` already traps — §3b/§5/totality).
  Gated on the name being an *undefined external*, so a guest-defined function of a matching name is a
  real call.
- *Out of scope:* `core::hint::black_box` is an empty inline-`asm!` optimization barrier → inline asm
  stays a clean `Unsupported` (a non-goal); use `read_volatile` to make a value opaque.
- Tests: `rust_core_enum_slice_option`, `rust_core_structs_and_iterators` (idiomatic `core` vs native),
  `rust_panic_path_div_traps_and_runs` (a runtime division whose div-by-zero/overflow panic branches
  now trap, taking the non-panic path to match native). 96 translate tests green, fmt + clippy clean.

**Slice AJ (DONE) — Rust trait objects, slices, `unwrap`.** More idiomatic Rust, byte-identical to
native: **`&dyn Trait` dynamic dispatch** (a vtable load + `call_indirect` per call — the Rust analog
of the C++ vtable path, slice AG: fat pointers + per-type vtable globals + funcref dispatch, all
already covered), **`&[T]` slice arguments** ({ptr,len} across an `#[inline(never)]` call + a
sub-slice), and **`Option::unwrap`** (its panic path traps via slice AI's recognizer). The one gap:
- **Auto-vectorization → SIMD.** `rustc -O2` vectorizes a reduction loop (the slice `sum`) into
  `<N x i32>` + a horizontal reduce — out of the MVP (§17/D58 is the SIMD lane). The Rust bitcode
  helper now disables it (`-C llvm-args=-vectorize-loops=false -vectorize-slp=false`), matching the
  C/C++ lanes' `-fno-*-vectorize`. The native oracle keeps vectorizing — an integer reduction is
  associative, so scalar (on-ramp) and vectorized (native) agree.
- Tests: `rust_trait_object_dispatch`, `rust_slice_argument`, `rust_option_unwrap`. 99 translate tests
  green, fmt + clippy clean.

**Slice AK (DONE) — Rust `alloc` / heap (`Vec` via a guest `#[global_allocator]`).** The headline for
*real* Rust: a `no_std` + `alloc` crate whose `#[global_allocator]` routes to the guest `malloc`/`free`
runs byte-identical to native `rustc`, with `Vec::push` growing the heap (alloc + `memcpy` + free)
through the on-ramp's `vm_map`-growing bump allocator. Because the allocator + `Memory` grant live in
the powerbox `_start` (gated on `main`), the test runs **through the powerbox**: the on-ramp synthesizes
`#[no_mangle] extern "C" fn main` calling `compute()`, the differential compares the `u8` exit/return
code, and a pinned expected value keeps it non-vacuous. Three gaps closed:
- **`alloc` abort lang items.** `alloc::raw_vec::handle_error` / `alloc::alloc::handle_alloc_error`
  (OOM / capacity-overflow, `-> !`, external) join the panic recognizer (slice AI) → trap.
- **Constant `inttoptr`/`ptrtoint` operands.** Rust's `NonNull::dangling()` (an empty `Vec`'s pointer =
  its alignment, e.g. `inttoptr(i64 4)`) folds to its `i64` window value in `operand` (via `const_eval`,
  like the constexpr-GEP path) — not just in global initializers.
- **rustc edition.** The Rust lanes pin `--edition 2021` (the default 2015 mis-resolves `core::`/
  `alloc::` paths in the std oracle).
- Test: `rust_alloc_vec_via_global_allocator` (Σ i² for i in 0..64 = 85344, % 251 = **4**, vs native).
  100 translate tests green, fmt + clippy clean.

**Slice AL (DONE) — `Box` + `String`: a mini expression evaluator (the heap capstone demo).** A
recursive-descent parser over a byte slice builds a **`Box`ed recursive AST** (`enum Expr { Num,
Add(Box,Box), … }` — the canonical use of `Box`), `eval` walks it recursively, and `render` serializes
it back into a heap **`String`** — a tiny interpreter, right at home next to the guest-JIT demo, running
byte-identical to native `rustc` (through the powerbox/alloc harness). Two real gaps closed, both
generally useful (any frontend's `-O2` hits them):
- **`llvm.{u,s}{add,sub,mul}.with.overflow.iN`** (`lower_overflow_intrinsic`) → the wrapping op + a
  computed overflow flag, recorded as a 2-field aggregate `{result, overflow}` (consumed by
  `extractvalue`). Rust's checked capacity/index arithmetic (`Vec`/`String`/`Layout::array`) emits
  these; the flag feeds a branch to `handle_error`/`panicking` (→ trap). Exact formulas (`add`: wrapped
  sum below an operand; `sub`: borrow; signed: sign-disagreement; `mul`: zero-guarded `r/a != b`).
- **`switch` on `i64`** (Rust enum discriminants). `translate_switch` now lowers an `i64` operand to a
  `br_table` by biasing with `min` (`i64`) and **folding the high 32 bits into the index** — an
  out-of-`[0,2^32)` value forces the default — sound for any `i64` (a bare low-32 `br_table` would
  alias far-apart values onto a case). `i128` switches stay `Unsupported`.
- Test: `rust_box_string_expr_evaluator` — `eval("2+3*4-(5-1)*2+10") = 16`, rendered string is 26
  chars, `(16+26) % 251 = 42`, on-ramp == native. 101 translate tests green, fmt + clippy clean.

**Slice AM (DONE) — the Rust capstone: a `jsmn`-style JSON tokenizer (a real `no_std` program).** The
Rust analog of the C corpus's `jsmn` demo: scan a JSON document (`&[u8]`) into a heap `Vec` of typed
tokens (`enum Kind { Obj, Arr, Str, Prim }` + span), handling `\`-escaped strings, whitespace, and bare
primitives, then fold a deterministic digest over the tokens. **Needed zero translator changes** — a
real Rust library runs end to end on the slices already in place (`Vec<struct>` heap + growth, enums,
`&[u8]` scanning, `match` on bytes, enum→int cast), byte-identical to native `rustc`. This is the
breadth-proof capstone: not a unit test of a feature, but a recognizable program, the way `jsmn`/
`sha256`/`clay` validated C beyond the per-feature slices.
- Test: `rust_json_tokenizer_capstone` — 14 tokens over the doc, folded digest `% 251 = 135`, on-ramp
  == native. 102 translate tests green, fmt + clippy clean.

**Slice AP (DONE) — lane-wise vector integer conversions (`zext`/`sext`/`trunc`).** With the vector
legalization landed (I2 / PR #56), the first of the four vector-op classes that block re-enabling
auto-vectorization on the breadth lanes. svm-ir has **no vector-convert op**, so a `<N x iA> → <N x iB>`
widen/narrow **scalarizes**: explode the source into `N` per-lane scalars, convert each in its `i32`/
`i64` container via the same `emit_ext`/`emit_trunc` the scalar path uses, then repack into the
destination representation. The converter (`lower_vec_int_convert` + `vec_explode_int`/`vec_implode_int`/
`build_v128_from_lanes`, dispatched from the `Trunc`/`ZExt`/`SExt` arms when the operand is a vector)
handles **every source↔dest representation pairing**: the packed-`i64` `<2 x i32>`, a single `v128`,
and a legalized wide value (chunks via `ExtractLane`/`Splat`+`ReplaceLane`, tail lanes already scalar).
This lands the 4 conversion-blocked demos (`var_memset`, `revsum`, `heapgrow`, `simd_i16x8`) when
vectorization is on — confirmed by flipping the lanes (9 fails → 5). Float-lane vector conversions stay
`Unsupported` (fail-closed; a later slice with the `<N x i1>` mask work).
- Tests: `simd_conv_{zext_u8_to_i32,sext_i32_to_i64,trunc_i64_to_i32,trunc_to_u8}` — `check_vectorized_vs_native`
  (vectorization *enabled*), each a real `clang -O2` loop verified to emit the conversion, vs native.
  Covers `zext <4 x i8>→<4 x i32>`, `sext <2 x i32>→<2 x i64>`, `trunc <2 x i64>→<2 x i32>`, and
  `trunc <8 x i16>→<8 x i8>` / `<8 x i32>→<8 x i16>`. 125 translate tests green, fmt + clippy clean.
- **Remaining before the breadth-lane flip:** the `<N x i1>` mask machinery landed (slice AS, lands
  `crc32`); `perlin`/`clay` additionally need **float vector conversions** (`fptosi`/`sitofp` on
  `<2 x float>`), **`llvm.fmuladd.v2f32`**, and the **`<2 x i16>`** small-vector case — a follow-on
  slice (AT). The lanes are **all-or-nothing**, so they stay `-fno-*-vectorize` until those land and
  all of `perlin`/`crc32`/`clay`/`xxhash`/`async_io`/the conversion demos pass together.

**Slice AQ (DONE) — auto-vectorized vector rotate (`llvm.fshl.v4i32` / `fshr`).** The second vector-op
class (lands `xxhash` when vectorized). svm-ir's `VShift` takes only a *scalar* shift amount, but the
auto-vectorizer emits **per-lane-varying constant** amounts (xxHash's `<1,7,12,18>`), so the on-ramp
scalarizes the **rotate idiom** (`a == b`, the only funnel-shift form clang emits for `(x<<n)|(x>>(w-n))`)
lane-wise: explode the data + amount lanes (reusing slice AP's `vec_explode_int`), rotate each lane with
the scalar `IntBin Rotl`/`Rotr` (count masked mod lane width — no shift-by-width edge), and repack the
`v128` (`build_v128_from_lanes`). Handled in `lower_int_intrinsic`'s `vec128` branch; a general
(non-rotate) or float-shape funnel shift stays `Unsupported`.
- Test: `simd_vector_rotate_fshl` — a `(x<<13)|(x>>19)` rotate loop (`check_vectorized_vs_native`,
  vectorization enabled, verified to emit `llvm.fshl.v4i32`), vs native. 126 translate tests green.

**Slice AR (DONE) — general constant-mask wide shuffle (`shufflevector`).** The third vector-op class
(lands `vm_async_io_runtime` when vectorized). The single-`v128` `shufflevector` already lowered to
`Inst::Shuffle` (a byte mask); the **legalized wide** path only handled the broadcast (splat) form.
Generalized to **any constant mask**: explode both operands' lanes (`wide_explode_lanes`, shape-generic
`ExtractLane` + tail), gather each result lane from the `operand0 ++ operand1` concatenation per the
mask, and repack via `bind_lanes_as_vector` (a single `v128` when the result is one `v128`'s worth, else
a wide chunks+tail value). Subsumes the old splat case. This is the shape `async_io`'s `<8 x i8>`
byte-reversal (`<7,6,…,0>`) emits. A non-constant mask stays `Unsupported` (fail-closed).
- Test: `simd_wide_shuffle_reverse` — an `__builtin_shufflevector` `<8 x i8>` byte-reverse
  (`check_vectorized_vs_native`), vs native. 127 translate tests green, fmt + clippy clean.

**Slice AS (DONE) — `<N x i1>` boolean-mask machinery (vector `icmp`/`fcmp`/`select`/movemask).** The
fourth vector-op class (lands `crc32` when vectorized). svm-ir has **no first-class `<N x i1>` type**,
so a mask is held **lane-wise** (`mask_lanes`: `N` scalar `0`/`1`s, the `<N x i1>` analog of the `agg`/
`wide_vals` side-tables) and every producer/consumer is scalarized in a new `lower_mask` pass
(dispatched after `lower_wide`): vector `icmp`/`fcmp` → `N` scalar `IntCmp`/`FCmp` (narrow lanes
extended per the predicate's signedness); `select` (mask condition) → per-lane scalar `select` over the
exploded data (`vec_explode`/`vec_implode`, now float-capable — they reinterpret `<2 x float>` lanes);
`extractelement` → the lane; `insertelement`/`shufflevector` → build/permute the lanes; `bitcast … to
iN` (the SIMD **movemask**) → OR the lanes into a bitmap (handles the odd `i4`); `freeze` → identity.
`scan_func` records an `<N x i1>` result with a placeholder type (never used as a scalar). Masks are
block-local (like `agg`); a mask crossing a block edge stays `Unsupported` (fail-closed).
- Test: `simd_mask_icmp_select` — an `(a[i]==b[i]) ? … : …` loop → `icmp eq <4 x i32>` + `select`
  (`check_vectorized_vs_native`), vs native. `crc32` exercises `icmp`+`select`+`extractelement` end to
  end. (`bitcast`-to-`iN` / `insertelement`-splat are exercised by `clay`, slice AT.) 128 translate
  tests green, fmt + clippy clean.

**Slice AT (DONE) — float vector conversions + vec2 FMA + cross-representation shuffle; the breadth
lanes re-enable vectorization (the capstone).** The remaining gaps that blocked `perlin`/`clay`:
- **Float vector conversions** (`fptosi`/`fptoui`/`sitofp`/`uitofp`/`fpext`/`fptrunc`) scalarize lane-
  wise via `lower_vec_fp_convert` (the float analog of slice AP's int converter) — lands perlin's
  `<2 x float>`↔`<2 x i32>` gradient math.
- **`llvm.fmuladd.v2f32`** (and the other `<2 x float>` float intrinsics) scalarize lane-wise and
  repack the packed-`i64` vec2 (`vec_pack`).
- **Cross-representation shuffle**: a `shufflevector` whose operands and result use *different*
  representations (`<2 x float> ++ <2 x float>` → a `<4 x float>` `v128`) falls back to a generic
  explode/gather/repack (`bind_shuffle_result`), beyond the same-shape `Inst::Shuffle` fast path.
- **Capstone:** with all of `conversions`/`rotate`/`shuffle`/`masks` + these landed, the C/C++/Rust
  breadth-lane compile helpers **drop `-fno-*-vectorize` / `-vectorize-*=false`** — every demo now
  translates its real `-O2` auto-vectorized bitcode (SIMD and all), byte-identical to native. The
  fixed-128 chunk legalization (not flag suppression) preserves interp↔JIT/durable determinism.
- Verification: **128 translate tests green with vectorization enabled across every lane** — the C
  corpus (`sha256`/`xxhash`/`perlin`/`crc32`/`clay`/`regex`/`jsmn`/`heapgrow`/…), C++, Rust, the
  powerbox/async/JIT demos, and the focused `simd_*` shape/op-class tests. fmt + clippy clean.

**Slice AU (DONE) — full chibicc-demo parity: every program the C frontend runs now runs through the
on-ramp.** LLVM is the main frontend, so the on-ramp must cover *everything* chibicc does. The corpus
demos already crossed (slices A–AT); this closes the remaining chibicc demos — the two compute demos
and the **five concurrency demos** — with **no translator change** (the `__vm_*` capability/concurrency
surface from slices AC–AF already lowers them):
- **`calc` / `rational`** (`demos/calc.c`, `demos/rational.c`) — added to the native differential
  (`check_demo_vs_native`, byte-identical to `cc`). `calc` is a recursive-descent calculator over a
  **global function-pointer dispatch table** (relocations + `call_indirect`, slices K/G) with
  recursion; `rational` hammers the **by-value-aggregate sret ABI** (D39/slice J) — every op passes
  *and* returns a `struct Rat` by value, including an **indirect struct-returning call** through a
  dispatch table (sret + funcref relocation + struct-valued `call_indirect`, all at once).
- **The concurrency demos** (`work_stealing`, `mn_sched`, `steal_fibers`, `malloc_threads`,
  `async_work_stealing`) — these `#include <pthread.h>`, which is **chibicc's bundled guest libc** (a
  1:1 threading layer over `thread.spawn`/`join` + the futex + atomics). clang compiles them with the
  chibicc include dir on the path (`-I frontend/chibicc/include`), so the pthread shim resolves to the
  `__vm_*` builtins the on-ramp lowers — i.e. **the guest brings its own libc** (the on-ramp's model),
  here reusing chibicc's. They have no native oracle (`__vm_*` / guest fibers have no native symbol),
  so each asserts its **interleaving-invariant total** (the chibicc `c_guest_*` contract, via the LLVM
  frontend): `work_stealing` → 256, `mn_sched` → 1024 (stackful fibers, `cont.*`), `steal_fibers` →
  256 + 121920 (migratable fibers + stack-integrity), `malloc_threads` → 0 (thread-safe allocator, no
  overlap), `async_work_stealing` → Σ mix(0..16). The first four run on the **real powerbox**
  (`run_powerbox_with_deadline`, M:N executor); `async_work_stealing` is interpreter-only (the M:N +
  offload-pool oracle, like `vm_async_io_runtime` — the JIT async path needs the `HostAsyncHooks`
  harness). Tests (`translate.rs`): `demo_calc_vs_native`, `demo_rational_vs_native`,
  `demo_{work_stealing,mn_sched,steal_fibers,malloc_threads}_vs_chibicc`,
  `vm_async_work_stealing_runtime`.
- **`jit_threads`** (`demos/jit/jit_threads.c`) — the **threaded** guest-driven JIT (the threaded
  sibling of the `jit_demo` capstone): `NWORKERS` pthreads each build + `__vm_jit_compile` a distinct
  unit concurrently (several `Jit.compile`s in flight, serialized through the per-domain `Mutex<Host>`
  the powerbox engages for a `thread.spawn`ing guest) and check the native code against a C reference.
  Combines the guest pthread shim with the `Jit` capability + the 8-handle powerbox; prints `0`. Like
  the single-threaded demo it probes svm-llvm's window `size_log2` and patches the blob descriptor.
  Test: `vm_jit_threads_demo`.
- **157 translate tests green, fmt + clippy clean.** The on-ramp now runs **every demo the chibicc
  frontend does** — the breadth frontier is now *bigger* real-world programs (and other-language
  runtimes), not chibicc parity.

**Slice AV (DONE) — computed `goto`: `indirectbr` / `blockaddress` (the interpreter-category unlock).**
The threaded-dispatch idiom (`static void *tbl[] = {&&l0,…}; goto *tbl[op];`) every real bytecode VM is
built on (SQLite's VDBE, Lua, QuickJS) now lowers. clang `-O2` turns `&&label` into `blockaddress`
constants in the dispatch-table global and `goto *p` into an `indirectbr`:
- **The `llvm-ir` gap.** `llvm-ir` 0.11.3 erases a `blockaddress`'s operands (`Constant::BlockAddress`
  is payloadless). The LLVM-C API *does* expose them, so — exactly as `di.rs` does for the debug-info
  graph — `src/blockaddr.rs` re-parses the `.bc` through `llvm-sys` and recovers, per global, the
  blockaddress targets **in DFS order** (the `const_bytes` serialization order, popped positionally —
  the `di.rs` ordinal discipline). Threaded in via `translate_bc_path` like `di`.
- **Lowering.** A `blockaddress(@f, %bb)` → the **index of `%bb` within `@f`** (matching `block_idx`),
  baked into the table global by `const_bytes` (8 LE bytes, pointer width). `indirectbr %p, [dests…]`
  (`translate_indirectbr`) → a `br_table` indexed by that block index over `[0, nblocks)`: each listed
  dest routes to its block; out-of-list / out-of-range (UB) falls to the default (the first dest — a
  defined in-sandbox branch, §3b totality, never taken on well-defined input). `possible_dests` are all
  successors, so liveness threads each target's live-ins as `br_table` edge args (reusing `branch_args`).
- **Tests:** `computed_goto_threaded_interpreter` (a real threaded bytecode VM whose program is derived
  from `n` at runtime — so no dispatch constant-folds — byte-identical to native on **both** backends)
  and `computed_goto_lowers_indirectbr_to_br_table` (structural: recovery finds the table labels, the
  `indirectbr` becomes a `br_table`). **159 translate tests green, fmt + clippy clean.**
- *Follow-up — DONE (slice AW):* an **operand-position** blockaddress — clang's jump-threading threads
  one through a φ (an instruction operand, not a global). `blockaddr.rs` now also walks every defined
  function's φ nodes, keyed positionally `(func_idx, block_idx, phi_ord, incoming_idx)` (φ results /
  blocks are usually unnamed → the `di.rs` ordinal discipline); `branch_args` resolves a φ-incoming
  `blockaddress` to its block-index constant (`BlockCtx::phi_operand`). Test:
  `computed_goto_phi_threaded_blockaddress` — the same threaded VM but with a *constant* first dispatch
  (so clang threads the entry target through a φ), byte-identical to native on both backends, plus
  `computed_goto_phi_recovery_finds_operand_blockaddress` (structural). Computed `goto` is now robust
  enough for a real interpreter (Lua). **161 translate tests green, fmt + clippy clean.**

### Next frontier — real-world programs as correctness indicators

chibicc parity is done (slice AU). The on-ramp ingests *any* LLVM frontend's `-O2` bitcode, so the
next proof is **whole real-world programs** — not feature unit-tests but recognizable software whose
own test suites/known-answers validate the translation. This section is the standing plan for that
push: the selection criteria, the concrete translator gaps these programs force, the target ladder,
and the **SQLite** north star (in-memory *and* disk-backed via the powerbox).

### What makes a strong indicator (selection criteria)
1. **Self-validating** — ships known-answer vectors or its own test suite, so "correct" is a
   differential (native build vs on-ramp, or built-in KAT) rather than a hand-computed guess.
2. **Feature-dense in bug-finding ways** — hits translator corners toy demos don't: irregular control
   flow, the by-value-aggregate/varargs ABI, float formatting, wide integer math.
3. **Low OS surface** — can run in-memory / over embedded inputs. The on-ramp supplies
   `malloc`/string/`printf` (synthesized helpers) and the `__vm_*` capability surface, but **no
   ambient filesystem or sockets** — real I/O must come through a granted powerbox capability (below).

### Translator gaps these programs force (the real value of picking them)
Picking a target is really picking the *gap* it drives to completion. Current status:
- **Computed `goto` → `indirectbr` / `blockaddress` — DONE (slices AV + AW).** The linchpin for the
  interpreter category (SQLite's VDBE, Lua, QuickJS). A `blockaddress(@f, %bb)` lowers to the **index of
  `%bb` within `@f`**; an `indirectbr` lowers to a `br_table` over those indices. `llvm-ir` erases the
  blockaddress operands, so they are recovered via `llvm-sys` (`src/blockaddr.rs`, the `di.rs` pattern) —
  both in **global dispatch tables** (AV) and as **operand-position** φ-threaded blockaddresses clang's
  jump-threading emits (AW). Robust enough for a real interpreter; next interpreter target is **Lua**
  (which also needs `setjmp`/`longjmp`, below).
- **`setjmp`/`longjmp` — DONE on all three engines (slices AX + the JIT sub-slice); C++ EH next on the same substrate.** The
  two new core ops (`Inst::SetJmp`/`LongJmp`) lower from the recognized external `setjmp`/`_setjmp`/
  `sigsetjmp` (returns-twice) and `longjmp`/`siglongjmp` (noreturn) calls (gated on use, so non-users
  pay nothing). **Tree-walker**: `setjmp` snapshots the frame's resume point (block/inst/value-state —
  the value state is captured because `Frame::vals` is *replaced per block*) keyed by the `jmp_buf`
  address in a per-vCPU table; `longjmp` truncates the call stack back to it (intervening frames
  discarded, no cleanup — C has none), restores the frame + data-SP (which rides in `vals[0]`), and
  re-enters with the result set to the long-jump value. **Engine matrix** (kept in sync — never
  divergent): **both interpreters run it** — the tree-walker (snapshots the frame's value state, since
  `vals` is replaced per block) and the **bytecode** engine (no snapshot: its flat per-function register
  layout gives each block distinct slots, so the `setjmp` block's values survive a deeper call in place
  — it only restores the `(module, cur, base, pc)` cursor + pops the activation stack). The **JIT** now
  runs it too (**Option B, DONE** — see "JIT `longjmp`" below): libc `_setjmp`/`_longjmp` called **inline
  from JITted code** with the `jmp_buf` in a host-side per-run table keyed by the guest buffer address
  (no host SP/return-addr leaks into the window); the in-frame `_setjmp` dodges the "helper-frame-is-gone"
  problem, and clang's `-O2` returns-twice spills + Cranelift's caller-saved spills defuse the optimizer
  hazard. Gated to the `setjmp_rt` targets (unix among `fiber_rt`); a module mixing `setjmp` with
  fibers/threads is fail-closed-declined (the interpreters' per-vCPU keying covers it — a documented
  per-fiber-JIT-keying follow-on). Tests: `setjmp_longjmp_round_trip`, `setjmp_longjmp_loop_and_deep_nesting`,
  plus returns-twice stress (`setjmp_value_live_across`, `setjmp_nested_buffers`) — **all three engines ==
  native libc** (the JIT lane now asserts agreement, not a decline). Perf (`examples/onramp_perf.rs`): the
  non-`setjmp` baseline rows are unmoved (gated; byte-identical IR); the JIT runs `setjmp_happy` ~3× and
  `setjmp_longjmp` ~5–6× faster than the tree-walker (real-`setjmp`-class O(1) capture/unwind). ASan gate:
  the `ASan (JIT setjmp/longjmp)` CI lane runs the differential under `-Zsanitizer=address`. **EH next**
  (`invoke`/`landingpad`/`resume` + cleanups) reuses this stack-transfer core; the JIT `longjmp` + EH
  unblock Postgres `--single` and throwing C++ respectively.
- **`%f` / `%g` / `%e` float formatting** — **DONE**: the correctly-rounded exact-decimal (Dragon4,
  big-integer) formatter family landed (`__svm_dtoa_{fix_big,sci,gen}` over the `big_*` primitives),
  exact across the whole double range (1e±300, subnormals, nonfinite), plus the float `0`/`#` flags.
  Remaining printf tail: `*` (dynamic width/precision), non-constant formats, `%a`.
- **`qsort` + comparator function pointers** — the libc callback ABI (an indirect call from synthesized
  libc into guest code); confirm it lands cleanly when a target needs it.
- **`__int128` / `long double` (`x86_fp80`/`fp128`)** — rejected (§7 deferred). SQLite has a few `i128`
  paths; add on demand.
- **Larger libc surface** — `memmem`/`strtod`/`strtol`/`snprintf` family, `qsort_r`, etc. The on-ramp
  synthesizes a growing subset; each real program extends it (or the program brings its own, the model).

### Lua standard library + real stdout — ACHIEVED ✅ (all three engines)
**Goal (met).** Lua 5.4.7 with the **base/`string`/`table`/`math`** libraries open runs a script that
`print`s, and the on-ramp captures the exact **stdout bytes** the guest writes through the
`Stream.write` capability — byte-identical on the tree-walker, bytecode, and JIT, and to a native build.
The first end-to-end Lua producing *real output* (prior Lua tests checked a return value). Exercised:
`print`, `string.upper`/`rep`/`sub`/`#`, `table.sort`/`concat`/`insert`/`remove`, `math.sqrt`/`pi`/
`floor`/`max`/`abs`, `ipairs`, `pairs`, `type`, `tostring`. Fixture `tests/fixtures/lua/lua_stdlib.bc`
(harness + guest-libc shim), test `tests/lua_stdlib.rs`. Translator work it forced: synthesized
`__svm_strncmp`; `fputs`/`puts` of a **non-literal** string now emit a runtime `__svm_strlen`; the guest
shim brings `log10`/`log2`/`tan` (+ fail-closed `acos`/`asin`/`atan2`), `strstr`, and a no-filesystem
`stdio`.

### Lua `string.format` (runtime format engine) — ACHIEVED ✅ (all three engines)
**Goal (met).** `string.format` now works end to end, byte-identical to native on all three engines.
Lua's `str_format` builds its per-directive spec **at runtime** and calls `snprintf` once per directive
— the path the on-ramp's translate-time *constant*-format engine cannot lower (a non-constant or
unsupported-conversion `%a` format fail-closes to the `snprintf_rt` trap so the enclosing function still
lowers). The runtime format engine fills that gap with the **guest-brings-its-own-libc** model rather
than a new translate-time synthesis:

- **Guest runtime `snprintf`** (`tests/fixtures/lua/lua_fmt_snprintf.c`) `llvm-link`ed alongside Lua
  **shadows** the `snprintf_rt` trap. It parses flags/width/precision/length/conversion at runtime and
  formats `d`/`i`/`u`/`o`/`x`/`X`/`c`/`s`/`p` in C (matching glibc). One definition covers both the
  core's *constant* `%lld`/`%.14g` and `string.format`'s runtime directives.
- **Float bridges** `__vm_fmt_{fix,sci,gen}` — three new vm-builtins recognized in `lower_vm_builtin`.
  The guest `snprintf` delegates `f`/`F`/`e`/`E`/`g`/`G` to `extern int __vm_fmt_*(char *out, double,
  int prec, int width, int flags)`; the on-ramp lowers each to the existing correctly-rounded bignum
  **dtoa** helper (`dtoa_fix`/`dtoa_sci`/`dtoa_gen`) writing to float scratch, then `memcpy`s the result
  to the guest `out`. Gated by `uses_fmt_float`, which also forces a powerbox entry (so the float
  scratch is set up) and pulls in `dtoa`/`memcpy`.

Fixture `tests/fixtures/lua/lua_fmt.bc` (core + base/`string`/`table`/`math` + guest shim/libm/strtod +
guest `snprintf`), test `tests/lua_fmt.rs`, asserts the exact **stdout bytes** across
`%d`/`%x`/`%#x`/`%o`, width/precision/flags (`%5d`/`%-5d`/`%05d`/`%+d`/`%10s`/`%.3s`), `%c`, `%.2f`/
``8.3f`/`%+.1f`, `%.3e`/`%g`/`%.10g`, `%q`, and `%%`. (A known edge at the time — `%f` of an extreme
magnitude could differ by one digit, a 128-bit `dtoa_fix` limit — is gone: `__svm_dtoa_fix_big`
replaced that path with the exact bignum engine; `printf_float_fixed_bignum` asserts `1e300` exact.) This unblocks running **Lua's own test suite** (self-validating, the roadmap's gold standard) — the
next Lua slice.

### Lua's own test suite — ACHIEVED ✅ (`math.lua` + `utf8.lua` + 3 more files, all three engines)
**Goal (met).** Five **unmodified files from the official Lua 5.4.7 distribution's own test suite** run
through the whole VM: the dense **`testes/math.lua`** (fixture `lua_math.bc` / `tests/lua_math.rs`), the
full **`testes/utf8.lua`** (`lua_utf8.bc` / `tests/lua_utf8.rs`), and `testes/vararg.lua` +
`testes/bwcoercion.lua` + `testes/pm.lua` (`lua_testsuite.bc` / `tests/lua_testsuite.rs`) — each loaded
as its own chunk under `pcall`, one fresh `lua_State` per file. A Lua test signals failure by raising
(an `assert`); a clean **exit 0** means every `assert` held, **identical to native Lua** (the suite's
own pass/fail contract — each file's harness also runs green natively as the oracle). Same outcome on
the tree-walker, bytecode, and JIT. The files were chosen as self-contained (no `os`/`io`/`debug`/
`coroutine`, no internal `T` module): `math` (all of integer/float arithmetic, conversions, `//`/`%`,
float↔integer order incl. every NaN corner, the transcendentals, `modf`, `string.format`, decimal + hex
float literals, `math.random`), `utf8` (the whole `utf8` library — `char`/`codepoint`/`len`/`offset`/
`codes`/`charpattern`, strict vs. non-strict decoding across all 1–6-byte sequence sizes, surrogate/
overlong rejection, `\u{…}` escapes), `vararg` (`...`/`select`/`table.unpack`), `bwcoercion`
(string↔number bitwise coercions, `_ENV = nil`), and `pm` (the full pattern-matching engine).

**`require`.** `utf8.lua` opens with `local utf8 = require'utf8'` — the first file to need `require`.
Stock `loadlib.c`'s file and C-library searchers need a filesystem/dynamic loader the on-ramp does not
provide (and can never run for an already-loaded module), so the harness installs a **minimal `require`**
returning `_LOADED[name]` — exactly stock `require`'s fast path for a `luaL_requiref`'d library — and the
official file runs unmodified. No translator or libc change was needed: the four fixes below (landed for
`math.lua`) plus the existing shim already cover `utf8.lua` end to end.

**Four real translator/library fixes this forced** (each independently valuable, each gated on all three
engines + native):
- **`fcmp` NaN correctness.** The on-ramp collapsed **ordered vs unordered** float compares to one op
  (a documented fidelity gap). Lua's `luaV_flttointeger` (`n >= -2^63 && n < 2^63`, which clang emits
  as *unordered* forms after the preceding `n != floor(n)`) then accepted a NaN, so `NaN <=
  math.maxinteger` returned **true**. `emit_fcmp` now expands every predicate NaN-exactly (unordered =
  `uno(a,b) | ordered`, `one` = `a<b | a>b`), matching the interpreters' and Cranelift's `FloatCC`
  semantics. Test `fcmp_ordered_unordered_predicates` (all 11 predicates × NaN/ordinary, interp+JIT).
- **Sign-extended narrow signed ops.** A `<i32` value loaded from memory is *zero-extended* (the
  canonical narrow form), so its sign bit is buried at bit `N-1`, not the container's bit 31 — the
  on-ramp's `ashr`/`sdiv`/`srem` on an `i8`/`i16` now sign-extend the operand first (`bin` in
  `lib.rs`). Before the fix `ashr i8 0x80,7` gave `+1` (should be `-1`); since Lua's `testMMMode`
  compiles to exactly `ashr i8 luaP_opmodes[op],7`, `findsetreg` skipped the wrong instruction and
  `getobjname` dropped the operand name from error messages (`number (field 'huge') has no integer
  representation`). `(i8)-6/3` gave the unsigned `83`, not `-2`. Test `narrow_signed_shift_div_rem`.
- **Hex-float `strtod`.** The guest `strtod` now parses **hex floats** (`0x1.8p3`, `0x.ABCDEFp+24`, a
  64-bit hex mantissa needing round-to-nearest-even, `p`-exponent over/underflow, leading-zero fractions
  like `0x.000…0074p4004`, and the malformed `0x`/`0x.`/`0x3.3.3` with glibc `endptr` semantics) — the
  form Lua's own hex-float literals need. Additive (decimal path unchanged); the accuracy grid +
  `demos/strtod` differential cover it.
- **Guest libc gap-fills.** The shim brings real fdlibm `asin`/`acos`/`atan`/`atan2`/`modf`
  (`lua_testsuite_trig.c`) the base libm lacks, and `localeconv` (Lua reads `decimal_point` when
  appending `.0` to an integer-valued float in `tostring`).

**Coroutines — ACHIEVED ✅ (library slice, all three engines).** Lua 5.4 coroutines turned out to need
**no new machinery** — and, notably, **no fibers**. They are *stackless* with respect to the C stack:
each coroutine is a `lua_State` with its own heap-allocated Lua stack, and resume/yield ride the exact
same `luaD_rawrunprotected` / `luaD_throw` (setjmp/longjmp) primitive `pcall` already uses (ldo.c) —
there is no `swapcontext`/`ucontext`/assembly anywhere in Lua's core. So the on-ramp's existing
`SetJmp`/`LongJmp` core ops (proven by every working `pcall`) carry them unchanged. An in-house
differential (`lua_coroutine.bc` / `tests/lua_coroutine.rs`, source `lua_coroutine.lua`) runs green on
all three engines + native: `create`/`resume`/`yield` (multi-value both ways), the
`suspended`/`running`/`normal`/`dead` status transitions, `running`/`isyieldable`, `wrap`, error
propagation, **yield across `pcall`/`xpcall`** (the yieldable-pcall / continuation path),
`coroutine.close` with `<close>` variables, and a producer/consumer pipeline. The `svm-fiber` machinery
(native-stack switching for vCPUs/Workers) is a separate, host-side concern and is deliberately *not*
involved.

**Debug library + official `coroutine.lua` — ACHIEVED ✅ (all three engines).** The unmodified official
`testes/coroutine.lua` now runs green on the tree-walker, bytecode, and JIT + native
(`lua_coroutine_official.bc` / `tests/lua_coroutine_official.rs`), which brought up the **`debug`
library** (`ldblib.c`: `getinfo`/`getlocal`/`setlocal`/`getupvalue`/`setupvalue`/`sethook`/`traceback`,
including debug on a *suspended* coroutine) alongside coroutines. Standalone the internal `T` C-test
library is absent, so the file's own `if not T`/`if T==nil` guards skip the C-API sections; the rest
still drives yields inside every metamethod and `for` iterator, `coroutine.close` with `<close>`
variables, and C-stack-overflow detection. The debug lib needed only one libc gap-fill (`fgets`, for the
never-called interactive `debug.debug()`).

**One reference-oracle change this forced** (no translator/coroutine/debug change): the file's *"infinite
recursion of coroutines"* case relies on Lua's own `LUAI_MAXCCALLS` raising a `pcall`-catchable "C stack
overflow". The production engines reach that self-limit; the tree-walker's reified call stack previously
capped at `MAX_CALL_DEPTH = 256` and tripped first as an *uncatchable* §5 kill. Raising the oracle cap to
`2048` — still well under the durable shadow-reserve frame budget — lets the reference oracle observe the
same catchable error the real engines do (its whole job). Verified regression-free: durable + interp +
`jit_diff` (54) suites and all four prior Lua fixtures stay green.

**io/os + official `files.lua` — ACHIEVED ✅ (all three engines, over a configurable Fs capability).**
The unmodified official `testes/files.lua` runs green on all three engines
(`lua_files.bc` / `tests/lua_files.rs`), bringing up the **io** and **os** libraries — and, more
importantly, the general mechanism they ride on:

- **Host-defined capabilities from C** (two new on-ramp builtins, both generic): `__vm_cap_resolve`
  (§7 `cap.self.resolve` — resolve an embedder-granted capability by *name* at runtime, the
  complement of the fixed 8-slot stash for capabilities granted outside the §3e prefix) and
  `__vm_host_call` (`cap.call HOST_FN op handle(a,b,c,d)` — the bridge to an **embedder-registered**
  `HostFn`, the wasm-import analogue). The translator stays pure mechanism: no fs semantics live in
  it, and *any* embedder capability is now reachable from C. No stash/ABI change.
- **A configurable Fs capability** (`svm_run::fs`, *not* part of the default powerbox): a 7-op
  protocol (open/read/write/seek/close/remove/rename, negative-errno results, window-relative
  buffers) with two interchangeable backends behind it — `mem_fs()` (deterministic in-memory; the
  hermetic test default) and `host_fs(root)` (the **real** filesystem attenuated to a root
  directory; `..`/absolute refused by protocol on both backends). Injected per run via
  `Instance::run_with_caps` — dependency injection at the capability boundary, exactly the wasm
  embedding model; no filesystem authority exists unless the embedder grants it. The C probe
  `fs_probe.c`/`tests/fs_cap.rs` covers the raw protocol incl. attenuation + real-disk assertions.
- **A real guest stdio (FILE) layer** (`lua_files_stdio.c`: mode parsing, `ungetc`, EOF/error flags,
  seek/tell, **setvbuf-honoring write buffering** — files.lua observes full/line/none visibility
  through a second reader — and POSIX-style unlinked `tmpfile`) and a **guest time/date layer**
  (`lua_files_time.c`: proleptic-Gregorian `gmtime`/`mktime`/`strftime`, UTC, exact round-trips).

files.lua runs byte-for-byte unmodified under the suite's own `_port`/`_soft` portability knobs
(skipping `popen`/`os.execute`/huge-data — process spawning is genuinely out of scope). Asserted on
all three engines against `mem_fs`, plus a `host_fs` run proving the same guest drives **real disk
I/O** end to end (temp root left clean). Native oracle: the same core+harness against real libc in a
scratch directory, exit 0.

With that, **every self-contained file in Lua's official suite is covered** (math, utf8, vararg,
bwcoercion, pm, coroutine+debug, files/io/os). Out of scope: `main.lua` (tests the standalone `lua`
binary) and the internal `T`-library C-API sections.

**The suite sweep — ACHIEVED ✅ (21 more official files; 28 of 33 total).** With the full library
surface in place, the remaining candidates were swept in one batch: `tracegc`, `verybig`, `big`,
`gengc`, `goto`, `events`, `code`, `bitwise`, `closure`, `tpack`, `literals`, `errors`, `nextvar`,
`sort`, `db`, `constructs`, `locals`, `cstack`, `strings`, `gc`, `calls` — all **byte-for-byte
unmodified**, bundled as `lua_sweep.bc` / `tests/lua_sweep.rs` (JIT in CI; the interpreter runs are
`#[ignore]`d full-depth gates, long like the extended fuzz; native oracle exit 0). The sweep harness
adds what the wider suite needs: a **real free-list allocator** (the bump arena exhausted under
`gc.lua`'s collector stress), **sibling-module `require`** + the stock preload searcher (suite files
require each other; `bitwise` installs a `bit32` shim via `package.preload`), a faithful
`package.loaded`/`preload`, and `@`-style chunknames (`db.lua` asserts its own `source`). The
bundle's Lua is built with `LUAI_MAXSTACK = 250000` (an edit to `luaconf.h`, Lua's own embedder
porting header, identical on the native side): `locals.lua`'s stack-overflow-with-`<close>` test
costs ~190 B/frame ≈ 93 MiB at the default 1M-slot ceiling, past the reference JIT's 64 MiB window —
the tests only need *a* ceiling to overflow, not that specific one.

**One real translator bug fell out** (strings.lua's longest-number test): the bignum float
formatter's big integers (`BIG_NLIMBS = 40`, sized for a double's exact value ≈ `2^1074`) did not
account for the `10^prec` scaling of fixed formats — `%.99f` of a near-maximum double reaches
`2^1023 · 10^99 ≈ 2^1352` and the digits were **silently truncated** (388 chars instead of 410).
Now 48 limbs (1536 bits), covering every finite double at C-cap precision; scratch layout shifted
(`FMT_*_O`, `FLOAT_SCRATCH_SIZE` 2304 → 2432). The guest `snprintf` also gained the ISO corners the
suite observes: `%p` width, `%a`/`%A` hex-floats (exact form for `%q` round-trips + precision with
round-half-to-even carry), zero-value-at-zero-precision integers, and guest-side `0`/`#`/width
handling for floats.

**The T library + `api.lua` — ACHIEVED ✅ (ltests.c active; internal assertions on).** The whole
core recompiled with `-DLUA_USER_H='"ltests.h"'` — every internal `lua_assert` live (including
`lua_checkmemory`'s GC walks), the failure-injecting `debug_realloc` allocator, debug sizes — and
**`api.lua` runs byte-for-byte unmodified**: `T.testC`'s string-driven interpreter drives raw
`lua_*` call sequences and allocation-failure paths no pure-Lua file can reach. The previously
skipped `if T` sections in `cstack`/`code`/`events`/`gengc`/`errors`/`nextvar`/`locals`/`coroutine`
now run (plus `gc` + `api` in their own bundle — ltests' `warnf` keeps warning state in process
statics, so each bundle is one shared `lua_State`, the official `all.lua` model; cumulative T-mode
memory bounds the reference JIT's 64 MiB window, hence the split). Fixtures `lua_tlib.bc` /
`lua_tapi.bc`, tests `tests/lua_tlib.rs` (JIT gates CI; interpreter runs `#[ignore]`d full-depth).
Native oracle exit 0, including ltests' real `atexit` leak check. One more translator bug fell out:
constexpr **`ptrtoint (ptr @g to i32)` ignored its target width** (raw i64 address into i32
arithmetic, verify TypeMismatch) — fixed + `const_ptrtoint_i32_width` differential.

**The official suite under its own driver — ACHIEVED ✅ (`all.lua`, the capstone).** The unmodified
`testes/all.lua` — the Lua distribution's own test *driver* — runs on the on-ramp: the whole
`testes/` tree is seeded onto the **in-memory Fs**, and `all.lua` `dofile`s each file through its own
`loadfile → string.dump → load` round-trip, `require`s sibling modules, and ends at `final OK !!!`.
It finds and loads every file off the (in-memory) disk via the **real** `luaopen_package`
(`loadlib.c`) searching `package.path` — the minimal-`require` shim is retired for stock
`require`/`package`/`searchpath`/`loadlib` (the ANSI C-library searcher returns `"absent"`, which the
suite guards for). T library active, one shared `lua_State`, the suite's own `_port`/`_soft`/`_nomsg`
knobs → **26 files** run (`main.lua` early-returns under `_port`, `big.lua` under `_soft`), including
`attrib.lua` (real `require`/`searchpath`/`package.config`, asserted `== 27`). `lua_all.bc` /
`tests/lua_all.rs` (JIT gates CI ~20 s; interpreter runs `#[ignore]`d full-depth); native oracle exit
0 incl. ltests' `atexit` leak check. The guest allocator became a **coalescing explicit-free-list**
(dlmalloc-lite, boundary tags): the whole suite in one state peaks ~19 MiB live, and coalescing keeps
the arena high-water near that — the earlier power-of-two-class allocator fragmented ~3× and overran
the reference JIT's 64 MiB window.

So the claim is now the strongest form: **the on-ramp runs Lua 5.4.7's official test suite, as
shipped, driven by the suite's own `all.lua`** — on the JIT, bytecode, and tree-walker. Only
`main.lua` (spawns the standalone `lua` binary; needs `os.execute` of a real process) and `heavy.lua`
(deliberate multi-GiB exhaustion) are outside scope, both of which `all.lua` itself omits from this
configuration.

### Lua with floats — ACHIEVED ✅ (all three engines, end to end)
**Goal (met).** Real Lua 5.4.7 core **`llvm-link`ed with the bundled guest `libm` + guest `strtod`**
runs a *float* script identical on the tree-walker, bytecode, and JIT — and identical to a native build
of the same sources. The script (`tests/fixtures/lua/lua_floats_harness.c`) computes
`(3.14 + 2.0^0.5 + (10.5 % 3.0) + 1.5e3 + 0.25) * 1000.0` → `1506304`, reaching **every new piece of
this work in one run**: the guest **`strtod`** (every numeric literal, in the lexer), the guest
**`pow`** (the `^` operator), and the synthesized **`fmod`** (the `%` operator) — plus
`frexp`/`localeconv`/`snprintf`/`setjmp` referenced by the core. The guest `pow`/`strtod` definitions
**shadow** the on-ramp's would-be trap stubs (the `llvm-link` is all it takes); `fmod`/`frexp`/`localeconv`/
`sqrt`/`ldexp`/the string ops stay undefined and the on-ramp synthesizes/recognizes them. Test
`tests/lua_floats.rs` (committed fixture `lua_floats.bc`); regenerate per the fixtures README. This
closes the loop: the per-function units validate each guest impl bit-exact vs the system, and this
proves they **compose into a real interpreter's float path** end to end.

### Lua first light — ACHIEVED ✅ (all three engines)
**Goal (met).** A pure-compute Lua 5.4.7 script (`local x=0; for i=1,10 do x=x+i end; return x` → 55)
runs through the on-ramp **identical to native Lua on the tree-walker, bytecode, *and* JIT**
(`Returned([I32(55)])` on each). It exercises the real **lexer, parser, code generator, GC, and the
bytecode VM** (computed `goto` dispatch + `setjmp` error handling). The first whole real-world
interpreter on the on-ramp. Reproduce: build `lua_core.bc` (recipe below) → `examples/run_lua.rs`
(`run_lua <bc> [tree-walk|bytecode|jit]`, runs through the powerbox with Memory granted).

**What it took (this branch):** the varargs ABI (keystone), the libc string batch + `abort`, exact
`ldexp`, the cross-block `<N x i1>` mask fix (a real translator bug Lua surfaced), and **fail-closed
stubs** for the libc the integer-only path never executes: `pow`/`fmod`/`frexp`/`strtod`/`snprintf`/
`localeconv`/`__errno_location` trap if called (bit-exact transcendentals need a host-libm decision —
below), and `time()` returns `0` (the `makeseed` RNG seed is result-irrelevant). These stubs translate
so the module lowers and the executed path is fully real; replacing them is what graduates first light
to *general* Lua (float arithmetic, `string.format`, error messages).

**Next (post-first-light) — the remaining-libm slices.** `snprintf` is **DONE** (the number→string
direction, via the printf engine — merged). What's left, each a localized swap of a fail-closed stub
for a real implementation, split by whether it needs a *host-libm decision*:

- **Exact, decision-free (synthesizable bit-for-bit, no libm dependency):**
  - **`frexp`** — **DONE.** `synth_frexp` (5 blocks): pure bit ops extracting exponent + mantissa,
    writing `*e`; bit-exact to glibc incl. the subnormal (`×2^54` renormalize) and special (zero/
    inf/nan → `*e=0`, return `x+x`) paths. Test `libc_frexp_bit_exact` (a grid incl. a subnormal
    and ±inf), all three engines == native.
  - **`fmod`** — **DONE.** `synth_fmod` (28-block CFG): a faithful translation of musl's exact 64-bit
    remainder (special→NaN, `|x|≤|y|` early returns, subnormal-normalize loops for x and y, the
    bit-shift remainder loop, result renormalize + scale). The remainder is mathematically *exact*
    (always representable), so it is bit-for-bit identical to libc with **no `frem` op and no libm
    decision** — the earlier "`pow`/`fmod` need host-libm delegation" framing was imprecise: only the
    *transcendentals* do. Test `libc_fmod_bit_exact` (a 10×8 grid incl. subnormals, a large quotient,
    and the `y==0`/`x==inf`→NaN paths), all three engines == native.
  - **`localeconv`** — **DONE.** `build_locale_data` lays a read-only C-locale `lconv` struct as
    module data (`decimal_point="."`, the other strings `""`, the numeric/monetary `char` fields
    `CHAR_MAX`), and `localeconv()` returns its address (a `synth_const_i64`). No powerbox needed
    (the struct rides in the globals region like the ctype tables). Test `libc_localeconv_c_locale`,
    all three engines == native's C locale.
  - **`__errno_location`** — a writable `errno` int slot. Deferred and **bundled with `strtod`**: it
    needs the powerbox page-0 layout (a writable persistent slot), which entangles it with the
    powerbox test harness, and it is only ever *set* by `strtod` (`ERANGE`) — so it lands where it is
    exercised end to end.
- **Hard but decision-free:** **`strtod`** (string→double) — **DONE, as guest code** (the keystone for
  *float* Lua: every decimal float literal hits it). `crates/svm-run/demos/strtod/strtod.c` is a
  correctly-rounded decimal→`f64` parser. Correctly-rounded is *unique*, so it matches glibc bit-for-
  bit — and the method needs **no precomputed power-of-ten table** (nothing to mis-transcribe): parse
  every significant digit into a big integer, form the exact rational `N/Dn`, and take the nearest
  double by an **exact big-integer division** with round-to-nearest-even (normal / subnormal incl. the
  boundary / overflow→±inf). A guest def shadows the on-ramp's `strtod` trap stub (`llvm-link
  lua_core.bc strtod.bc`). Two gates: `demo_strtod_vs_native` (raw-f64-image + `endptr` differential,
  all three engines == native) and `strtod_guest_correctly_rounded_vs_system` (native-only, bit + `endptr`
  vs the system `strtod` over the hard cases — subnormal halfway ties, the `2^53` boundary, max-double,
  over/underflow, 40-digit strings). *Scope:* decimal only; hex floats (`0x1p4` — Lua's core parses
  these itself), the `inf`/`nan` spellings, and `errno`/`ERANGE` are follow-ups.
- **Transcendentals → a guest `libm` (DECIDED — keep math in the sandbox).** `pow`/`exp`/`log`/
  `sin`/… can't be synthesized bit-exact to a *specific* host libm, and a host math capability would
  leak math out of the sandbox — so they ride as **guest code** (the §"transcendentals" preference,
  the way the raytrace demo bundled `sin`/`exp`). **Started:** `crates/svm-run/demos/libm/libm.c` is a
  small self-contained guest libm — faithful **fdlibm** transcriptions of `exp`, `log`, and **`pow`**
  (genuinely accurate, not poly approximations), using only IEEE `+−*/`, compares, and union word
  access. A guest definition **shadows** the on-ramp's would-be `pow`/`exp`/`log` trap stubs, so
  `llvm-link lua_core.bc libm.bc` makes them real. `pow` reuses `sqrt` (→ the SVM `f64.sqrt` op) and
  `scalbn` (→ `__svm_ldexp`); **`sin`/`cos`** add fdlibm's `__kernel_sin`/`__kernel_cos` + the
  **medium-path** argument reduction (accurate for `|x| ≤ 2²⁰·π/2 ≈ 1.65e6`, the bound where `n·pio2_1`
  stays an exact product — covering all realistic use; the `npio2_hw` fast-path table is dropped since
  always running the cancellation-correction is equally correct, and the full Payne-Hanek
  `__kernel_rem_pio2` table for astronomically large args is the documented future addition).
  Two gates: `demo_libm_vs_native` (raw-f64-image differential, all three engines == native —
  byte-identical *because* the math is guest code, unfused on both lanes; built `-fno-*-vectorize` so
  the on-ramp takes scalar bitcode while the oracle vectorizes, the corpus-demo split) and
  `libm_guest_exp_log_accurate_vs_system` (a native-only `-D`-renamed build vs the system `<math.h>`,
  ≤2 ULP — validates the fdlibm transcription, which a same-source differential cannot).
  **Remaining:** `tan` + inverses/hyperbolics, the Payne-Hanek table for huge `sin`/`cos` args.
  **End to end — DONE:** the guest libm + guest `strtod` `llvm-link`ed into the real Lua 5.4.7 core run
  a float script (`x^0.5`, `x%y`, decimal/scientific literals) **identical on all three engines and to
  a native build** — see *Lua with floats* below.

After the float-number surface lands, `print`/stdlib graduates first-light to a script that does I/O.

**Build recipe (reproducible).** Lua's core (no standard libraries, no `lauxlib`) + a tiny C-API harness
that drives `lua_newstate`/`lua_load`/`lua_pcall`/`lua_tointeger` with its own allocator (`realloc`) and
string reader — this keeps the file-I/O surface (`fopen`/`fread`/`fprintf` from `lauxlib`'s loaders)
*out* of the module:
```
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"          # NOT: the lib*.c, lauxlib, lua.c, luac.c
for f in $CORE harness; do clang -O2 -emit-llvm -c -Ilua-5.4.7/src $f.c -o $f.bc; done
llvm-link *.bc -o lua_core.bc                              # one module → svm-llvm-translate
```
The full `luaL_*` build pulls in ~39 externals (file I/O dominates); the core-only build above is **21
externals + 4 defined-varargs functions** — the true first-light surface.

**Gap inventory (core-only `lua_core.bc`), in dependency order:**
1. **Varargs *definition* ABI — the keystone. DONE (slice 1).** `luaG_runerror` / `luaO_pushfstring` /
   `lua_pushfstring` / `lua_gc` are defined `(...)` functions; clang `-O2` lowers `va_start`/`va_arg`
   into the **System V AMD64 `__va_list_tag` dance** (`gp_offset`/`fp_offset`/`reg_save_area`/
   `overflow_arg_area`). The on-ramp used to reject defined varargs functions outright; it now lowers
   them via an **overflow-only model** (no `reg_save_area` synthesized): a `(...)` def reserves the
   first 8 bytes of its frame (offset 0) for an **incoming overflow-area pointer**; `llvm.va_start`
   writes the tag with `gp_offset=48`/`fp_offset=176` (both ≥ their register thresholds, so clang's
   lowered `va_arg` *always* takes the memory branch — the register path is dead) and
   `overflow_arg_area = *(sp+0)`, `reg_save_area = null`. At a direct `(...)` **call** site the caller
   marshals the variadic args into a contiguous 8-byte-slot frame scratch (`VARARG_SCRATCH`) and
   deposits a pointer to it at `callee_sp + 0`; only the fixed params are passed as IR args (the
   callee signature stays `(sp, fixed…)`, no synthetic param). `va_end`→no-op, `va_copy`→24-byte tag
   copy. Mixed int/SSE args work because the forced memory path lays all args out positionally.
   Limitation: a variadic arg wider than 8 bytes (a 16-byte `v128`/by-value aggregate) is a clean
   `Unsupported` (Lua's are all int/ptr/double/ptrdiff ≤8B). **Tests:** `varargs_int_double_mixed`,
   `varargs_many_and_copy` — all three engines == native cc. **217 translate tests green, fmt+clippy
   clean.**
2. **`snprintf`** (a varargs *call*) — Lua's number→string (`lua_number2str`/`tostringbuff`). **DONE:**
   routed through the shared `printf` format engine (`lower_snprintf`) with output redirected into the
   caller's buffer; reuses the bignum dtoa float formatter + the varargs-call marshaling from (1).
   Merged (PR #155), incl. the page-0 `FMT_BUF` write-protect fix for `snprintf`-into-stack-buffer + `%p`.
3. **`strtod`** (string→double) — numeric-literal parsing in `llex`/`lobject`. **DONE, as guest code**
   (`demos/strtod/strtod.c`): correctly-rounded decimal→`f64` via an exact big-integer division (no
   power-of-ten table); a guest def shadows the trap stub. Bit-identical to glibc; see the "remaining-
   libm slices" section above. Decimal only (hex floats / `inf`/`nan` / `errno` are follow-ups).
4. **Small libc batch** (synthesized byte-loops / recognized intrinsics, like the existing
   `memcmp`/`strlen`). **Done (slice 2):** `strcmp`/`strchr`/`strcoll`(→`strcmp`),
   `strcpy`/`strspn`/`strpbrk` — synthesized `__svm_*` byte loops (the nested-scan pair for
   span/break); `abort`→trap (dropped like the Rust panic lang-items — the following `unreachable`
   traps). Tests `libc_strcmp_strchr_strcoll`, `libc_strcpy_strspn_strpbrk`, `libc_abort_translates`
   — all three engines == native, with `volatile`-loaded operands so clang `-O2` keeps the calls.
   `ldexp`/`scalbn` synthesized as `__svm_ldexp` (the musl `scalbn` two-step-scale algorithm,
   bit-exact to libc incl. overflow→±inf and gradual underflow) — test `libc_ldexp_bit_exact` folds a
   grid of `x`×`n` (extremes included) into a checksum, identical to native on all three engines.
   `fmod`/`frexp` synthesized bit-exact (no `frem` op exists): `__svm_fmod` via musl's exact 64-bit
   long-division remainder loop, `__svm_frexp` the mantissa∈[0.5,1)/exponent split writing `*exp` —
   tests `libc_fmod_bit_exact`/`libc_frexp_bit_exact`, all three engines == native.
   **Remaining:** `pow` (a **deliberate** fail-closed trap stub — bit-exact vs native needs the
   host-libm decision, §8; a program bringing its own `pow` shadows the stub), `localeconv` (a static
   C-locale struct: `decimal_point="."`), `__errno_location` (a fixed window slot — `strtod` sets
   `ERANGE`), `time`→stub/`Clock` cap (RNG seed in `lstate` `makeseed`). Already covered:
   `_setjmp`/`longjmp`, `free`(no-op), `realloc`, `bcmp`→`memcmp`, `strlen`.

   *Varargs bug found via Lua (fixed):* a `(...)` call with **zero** variadic args (Lua's
   `lua_gc(L, what)`) triggered marshaling but `frame_layout` reserved no scratch (0 slots) →
   `Unsupported("varargs call without reserved scratch")`. Fixed by reserving ≥1 slot whenever a
   function makes any varargs call; regression test `varargs_zero_variadic`.

**Cross-block `<N x i1>` masks — DONE (translator bug Lua surfaced).** With the libc batch in place the
Lua-core translation reached the GC `atomic` function and failed with `Unsupported("value N not
available in block")`. Root cause: clang's SLP vectorizer fuses two adjacent byte-tests (a `GCObject`'s
`marked`/`tt` fields) into a `<2 x i8>` load+`and`+`icmp`, producing a **`<2 x i1>` mask** in one block
and `extractelement`-ing its lanes in *successor* blocks. The on-ramp held masks lane-wise in a
**block-local** `mask_lanes` table ("assumed not to cross block boundaries"). Fixed by unifying masks
into the **`agg` side-table** with an `[i32; N]` `agg_layout` (exactly like the i128 `(lo,hi)` pair), so
a mask fans out into per-lane block params and threads across edges via the existing
`block_params`/`branch_args` machinery; `agg_operand` also learned the constant-`<N x i1>` φ-incoming
case. `mask_lanes` is gone. Regression test `cross_block_i1_mask` (hand-written LLVM, all three engines
× 4 seeds). 222 translate tests green. After this, the Lua-core translation advances past `atomic` to
the next libc gap (`pow`).

**Sequencing:** (slice 1 — **DONE**) varargs ABI + standalone differential tests; (slice 2 — in
progress) the small-libc batch; (slice 3) `strtod` + `snprintf`; then the Lua-core differential test
(native exit vs on-ramp exit, all engines). Each slice is independently testable against native. After
slice 1 the Lua-core translation advances past the four varargs defs to the libc surface (first stop:
`ldexp`).

### SQLite — the north star (in-memory, then disk via the powerbox)
SQLite is the gold-standard target (≈600:1 test-to-code ratio, ships as one amalgamation `.c`). Two
phases, both worth doing:
- **Phase A — in-memory (`:memory:`). ACHIEVED ✅ (slice BI).** Built `SQLITE_THREADSAFE=0`,
  `SQLITE_OMIT_LOAD_EXTENSION`, `SQLITE_OS_OTHER=1` (a deterministic guest VFS — fixed PRNG/clock,
  file-open fail-closed); the differential is native-SQLite vs on-ramp-SQLite over the same
  breadth script, **byte-identical**. No filesystem needed — the whole DB lives in `malloc`'d
  pages. The gates named here (computed goto for the VDBE + float formatting) both held: the VDBE
  dispatch and `%!.15g` output work unmodified. The SQL-logic-scripts extension landed as slice BK
  (~46k corpus records green in the sandbox); the capability story is Phase B (slice BJ).
- **Phase B — disk-backed (real persistence through a granted capability). ACHIEVED ✅ (slice BJ):**
  the guest VFS shim over the configurable `fs` capability (`mem_fs`/`host_fs`), native SQLite
  reading a capability-written database file and vice versa. In-memory proves the SQL
  engine; **disk proves the capability story** — that a sandboxed guest can do real, durable I/O
  *only* through explicitly granted authority. SQLite is built for exactly this: its **VFS** (`sqlite3_vfs`)
  is a narrow, swappable I/O shim (`xOpen`/`xRead`/`xWrite`/`xTruncate`/`xSync`/`xFileSize`/locking).
  The plan: a **guest VFS shim** (the SQLite analog of the guest `<pthread.h>` shim — ordinary guest C
  the program brings) that bridges `sqlite3_vfs` to a granted **file/storage capability** delivered
  through the powerbox handle stash, exactly like `Stream`/`Memory`/`IoRing` are today. Concretely:
  - **The capability.** Today's powerbox grants are `stream`/`exit`/`memory`/`address_space`/`io_ring`/
    `blocking`/`jit`/`shared_region`/`clock`/… (`grant_*` on `Host`). There is **no file/storage
    capability yet** — that is the new host-side piece: a positioned-I/O surface (`read_at`/`write_at`/
    `truncate`/`sync`/`size`/advisory-lock) over a host-chosen backing file or block store, granted as
    one more handle in the stash. Two viable shapes: (a) a **dedicated `File`/`Storage` capability**
    (cleanest semantics), or (b) **ride the existing §9/§12 `IoRing` + `Blocking` path** — a VFS that
    `submit_async`s `pread`/`pwrite`/`fsync` as blocking ops onto the offload pool (reuses the async
    machinery slices AD/async demos already exercise; the parked-vCPU completion model is a natural fit
    for SQLite's synchronous file calls). Recommend prototyping over `IoRing`/`Blocking` first (no new
    capability type), then promoting to a first-class `File` cap if the ergonomics warrant.
  - **On-ramp side: already done.** The frontend already lowers `__vm_*` builtins and resolves
    capability imports to stash handles (slices AC–AF) and grants a contiguous handle prefix in the
    synthesized `_start`. A new `File`/storage handle slots into that same mechanism — likely **no
    translator change**, just the host capability + the guest VFS shim + extending `synth_start`'s
    grant prefix to include it.
  - **Why it matters.** It is the end-to-end demonstration of the whole thesis: a real database, real
    durable files, **zero ambient authority** — every byte of disk access flows through a capability the
    embedder explicitly handed over, auditable at the powerbox boundary.

### Candidate targets, grouped by the dimension they prove
- **Self-validating interpreters (densest control-flow + ABI stressors):**
  - **Lua** (reference impl) — runs the *official* `testes/` suite; forces **`setjmp`/`longjmp`** +
    computed goto. The cleanest "second SQLite."
  - **QuickJS** (Bellard) — full JS engine with a **test262** runner; extreme density (NaN-boxing,
    bigint, regex, computed goto). Big lift; little is left unproven if it passes.
  - **mal / chibi-scheme / a tiny Forth** — cheap stepping stones to the same control-flow features.
- **Byte-exact, zero-OS-surface known-answer suites (cheapest high-confidence wins — start here):**
  - **Monocypher** or **fiat-crypto** — modern crypto (ChaCha20/Blake2/X25519/Ed25519) with built-in
    **KAT vectors**; brutal on 64-bit/carry arithmetic + constant-time bit-twiddling. Extends the
    existing sha256/crc32/xxhash family, no OS surface at all. (A TLS stack — **BearSSL/mbedTLS** as
    guest code, handshake vs vectors — fits here too, and is the bring-your-own TLS `curl` needs.)
  - **stb_image** decoding an **embedded** PNG/JPEG → assert exact pixel bytes (real parser + integer
    math, inputs compiled in).
  - **zlib/miniz full roundtrip** (have `tinfl` inflate; add deflate→inflate identity).
- **Deterministic compute / search (cheap codegen & control-flow bug finders, high "wow"):**
  - **A chess `perft`** (micro-Max or a clean perft) — known-answer node counts; recursion + arrays +
    heavy branching. Tiny, catches control-flow/codegen regressions fast.
  - **A SAT/SMT solver — CaDiCaL / MiniSat (C++)** — pure compute, zero OS surface, and self-validating
    in the strongest sense: SAT/UNSAT with **machine-checkable DRAT UNSAT proofs**. "An SMT-class solver
    runs sandboxed" is impressive and trivially hermetic.
  - **Stockfish (C++)** — `bench` is a reproducible node count + perft is known-answer; compute-bound,
    threads + NNUE (int8 SIMD — exercises the vector lanes), and a crowd-pleaser.
  - **FFmpeg, decode-only / in-memory** — codec cores are pure compute, SIMD-heavy, and the **FATE**
    suite is an enormous byte-exact corpus (decode an embedded clip, hash frames).
  - **TinyCC (TCC)** — a C compiler in C that compiles-and-runs; a small self-validating step toward the
    LLVM self-hosting dream (below).
  - **musl `libm` vs its test vectors** — IEEE edge cases; pairs with the `%f`/`%g` formatter work.
- **Real applications with a narrow I/O waist (each *drives a capability* — the capability-story
  proofs):**
  - **SQLite** (the north star, above) — the read/write **VFS** waist → a `File`/`Storage` capability.
  - **libmdbx** (embedded mmap'd B-tree KV store; the hardened LMDB successor) — no server, no network,
    but its data plane **is the memory-mapping itself** (readers read straight from the mmap). Drives a
    distinct **file-backed-mmap** capability ("map this file region into the window"), close to the
    existing AddressSpace/SharedRegion machinery — a *different* storage shape from SQLite's read/write
    VFS, so the two pair to prove both. Has a brutal torture-test suite; likely *easier* than Postgres
    (no setjmp, no server).
  - **Postgres — `--single` (single-user backend), not the multi-process server.** The postmaster
    (fork-per-connection + SysV shmem + signals + a listening socket) is OS surface (category 3); but
    `postgres --single` is one process reading SQL on stdin, collapsing to: the **File** capability (the
    data dir) + **`sigsetjmp`/`siglongjmp`** (its whole `PG_TRY()`/`ereport()` error model). The program
    that *justifies* the setjmp substrate; "SQLite Phase B at 100×." Self-validating via `pg_regress`.
  - **curl / local git** — both drive the **network** frontier. curl *is* the network (category 3) but
    is exactly what a capability is for: controlled, auditable **egress**. Needs a new **socket/connect
    capability** (security-sensitive, high narrative value) + bring-your-own TLS (above); its pluggable
    socket hooks (`CURLOPT_OPENSOCKETFUNCTION`/multi) wire cleanly to a cap. **git/libgit2** splits:
    local object-store/packfile/diff/merge is File-cap only (zlib + SHA-1/256 + diff, self-validating
    against a fixture repo); network clone wants the same socket cap.
- **The "wow" milestone (in progress):** **Doom** (shareware) — fixed-point-heavy real app; needs a
  stubbed framebuffer + an in-memory WAD, but "Doom runs sandboxed through the LLVM on-ramp, in a
  browser tab" is a strong external signal. (Graphical apps proper ride the **WebGPU capability** —
  its own section below.) **Sequenced into four slices** (each a self-contained PR):
  1. **Framebuffer output capability + canvas** — **DONE (slice BN).** A §7 by-name `HostFn`
     capability `display`, resolved like `fs`/`io`: `op 0 = present(ptr, w, h)` copies `w*h*4` RGBA
     bytes out of the guest window. `svm-browser`'s `onramp_exec` captures the last frame into
     `PbOutcome::framebuffer`; the wasm `svm_run_onramp` export exposes it via `svm_framebuffer_{ptr,
     len,width,height}`; `play.js` blits it to a `<canvas>` with one `putImageData`. First guest:
     `crates/svm-run/demos/display/gradient.c` (a deterministic 128×128 RGBA gradient). Proven
     byte-exact natively (`browser/tests/display.rs`, full-frame) and end-to-end in real Chromium
     (`browser-test.mjs` reads the canvas back). The 324 B `gradient.svmb` is committed as a web asset.
  2. **Interactive frame loop + input capability** — **DONE (slice BO).** The **reactor** run model:
     instead of running `main` to completion, the host instantiates once (`svm_onramp_open` runs
     `_start`) and calls the guest's exported `tick` **once per `requestAnimationFrame`**
     (`svm_onramp_frame`), with globals/BSS persisting between frames via the 256 KiB `SNAP_CAP`
     snapshot round-trip — `OnrampReactor` in `browser/src/lib.rs`, the browser twin of `svm-run`'s
     reactor `Session` over `bytecode::compile_and_run_capture_reserved_with_host`. Input rides a new
     by-name `keyboard` `HostFn` cap (`op 0 = poll()` → a packed `(pressed<<16)|keycode` event or
     `-1`; the doomgeneric `DG_GetKey` shape), fed by `svm_onramp_key` from the page's keydown/keyup.
     First guest: `crates/svm-run/demos/display/bounce.c` (a box you steer with the arrow keys).
     Verified deterministically (`browser/tests/reactor.rs`, exact per-pixel box positions + input
     response across frames) and end-to-end in real Chromium (`browser-test.mjs`: the box animates and
     the arrow keys steer it). (Slice 2 persisted only the low `SNAP_CAP` window — see slice 3a, which
     lifted that limit.)
  3a. **Full-window reactor persistence** — **DONE (slice BP).** Slice 2's snapshot round-trip
     persisted only the low 256 KiB (`SNAP_CAP`) and — decisively — `Mem::seed` clamps writes to the
     `mapped` boundary, so a `vm_map`-grown heap (which lives *above* `mapped`, where Doom's zone
     allocator sits) could **never** be round-tripped back. Fixed with a genuinely persistent instance:
     **`bytecode::Reactor`** (`crates/svm-interp/src/bytecode.rs`) holds the guest `Mem` **live** across
     calls — globals, BSS, **and** the grown heap all persist for free because the window is never torn
     down. It calls the private `run` per frame with a cheap fresh `Domain` over the shared compiled
     source (an `Arc` clone); host caps are serviced inline, so I/O guests work. `OnrampReactor` now
     wraps it (the `snap` round-trip is gone). Proof: `crates/svm-run/demos/display/life.c` — Conway's
     Game of Life with its grid in the **malloc heap above the mapped window**; the glider only advances
     if that heap persists. Deterministic (5 live cells throughout, translating (+1,+1) every 4
     generations) — asserted in `browser/tests/reactor.rs` + a `svm-interp` unit test
     (`tests/reactor.rs`, a counter at 293 KiB climbing across calls) + real Chromium (`browser-test.mjs`
     watches the glider advance). This unblocks Doom's memory footprint.
  3b. **doomgeneric translates + boots in the sandbox** — **DONE (slice BR)**;
     `crates/svm-run/demos/doom/`. The platform layer (`doomgeneric_svm.c`, `DG_*` onto the
     `display`/`keyboard` caps + a deterministic frame clock), the reactor entry (`main.c`:
     `doomgeneric_Create` once, `tick` = `doomgeneric_Tick`), a **complete libc shim** (`doom_libc.c`
     for string/ctype/stdlib/`sscanf`/stubs + the reused Lua `lua_files_stdio.c` `fs`-FILE layer and
     `lua_fmt_snprintf.c` printf engine), and `fetch.sh`/`build.sh`. Results: all **79** Doom TUs
     compile clean and `svm-llvm-translate` produces a **797 KB `doom.svmb`** (`main`/`tick` exported)
     with **zero unsupported IR constructs** — no SIMD/`i128`/inline-asm/vector-memory walls, and the
     on-ramp already lowers indirect calls through unprototyped `void (...)` (K&R) function pointers
     (an earlier spike using a *stale* translator binary misreported that as a gap). Driven through the
     slice-3a persistent reactor with the powerbox + `display`/`keyboard`/`fs` (the shareware
     `doom1.wad`), `_start`→`doomgeneric_Create` runs Doom's **entire init** on the bytecode
     interpreter — the real startup log (`Z_Init`… `W_Init: adding doom1.wad`… `DOOM Shareware`…
     `R_Init`… `ST_Init`), reaching the main loop. So the libc shim is correct end-to-end (WAD
     `fread`/`fseek`, `sscanf` config, `printf`, the `malloc` zone, the string set) — **"Doom runs
     sandboxed through the LLVM on-ramp" is proven at the init level.**
  3c. **doomgeneric renders — byte-exact frame-hash differential** — **DONE (slice BS)**. `doom_diff.c`
     is a headless platform whose `DG_DrawFrame` prints an FNV hash of each framebuffer; compiled BOTH
     as the guest (through the on-ramp) and native `cc`, it makes the rendered output a comparable
     frame-hash stream (the §18 oracle, like the SQLite stdout differential). The `svm-run` test
     `doom_diff` runs the guest over an in-memory WAD and asserts its hashes equal native's
     byte-for-byte. Over **200 frames** — the static title *and* the auto-played **demo1 (E1M1)
     gameplay** (64 unique hashes: live BSP/wall/floor/sprite/palette rendering + player movement) —
     guest == native **exactly**, so Doom's whole fixed-point renderer is correct through the on-ramp.
     Key fix: `DG_SleepMs` must *advance the virtual clock* (Doom's `TryRunTics` busy-waits on the
     clock; a no-op spins forever — that spin, not real work, was the earlier "20 B instructions to
     boot"). With it, init + 200 frames run in **~24 s on the release bytecode interpreter, no JIT**.
     (`crates/svm-run/demos/doom/{doom_diff.c,diff.sh}`, `crates/svm-run/tests/doom_diff.rs`.)
  4. **Doom in the playground** — wire the reactor `.svmb` + `doom1.wad` as browser assets, grant the
     `fs` cap in the browser `OnrampReactor`, and drive the reactor loop in `play.js` (loop + keyboard
     already there); build/deploy via `pages.yml`. Correctness is already proven byte-exact (3c); this
     is the visible payoff (the wasm-JIT tier may be wanted for a smooth frame rate).
- **Other-language runtimes** (the breadth thesis, building on the C++/Rust slices AG–AM): a real Rust
  crate (`regex`/`ryu`/`serde_json` `no_std`), a Zig program, a Swift `-enable-experimental-feature
  Embedded` TU — each is "another frontend, no translator change beyond what the corpus proved."

### Suggested ladder (cheap momentum → the capstones)
1. **Monocypher KAT** + **stb_image decode** — **DONE** (slices BG + BH): zero-OS, byte-exact; widen the corpus and shook out
   64-bit/parser bugs fast. A **DPLL SAT solver** now fills the "pure compute, self-validating" slot too —
   **DONE** (`demos/sat/sat.c`, `demo_sat_vs_native`): backtracking search + unit propagation (a branchy,
   array/pointer-heavy, deeply-recursive shape unlike the corpus hashers/parsers), self-validating via
   **planted** random 3-SAT (SAT by construction, the model re-checked against every clause in-guest) and
   **pigeonhole** PHP(3,2)/(4,3) (the classic UNSAT family); byte-identical to native on all three engines.
2. **`indirectbr` / `blockaddress` (computed goto)** — DONE (slices AV + AW: global tables + φ-threaded
   operand-position). Robust for real interpreters.
3. **`setjmp`/`longjmp`** — DONE on the interpreter (slice AX); JIT `longjmp` (native-stack switch) +
   **EH** on top (reuses the stack-transfer core) next.
4. **`%f` / `%g` float formatting** (Dragon4 bignum) — **DONE** (the `__svm_dtoa_*` family + the
   float `0`/`#` flags; see the printf slices above). SQLite/Postgres unblocked on this front.
5. **SQLite Phase A (in-memory)** — **DONE** (slice BI: the 3.50.2 amalgamation byte-identical
   to native over a breadth script; deterministic OS_OTHER VFS; two on-ramp additions).
6. **The storage capability** → **SQLite Phase B** (read/write VFS) and **libmdbx** (file-backed mmap) —
   two distinct storage shapes proving real durable I/O under zero ambient authority.
7. **Postgres `--single`** — the setjmp + File-capability capstone ("SQLite Phase B at 100×").
   **SPIKE DONE (slice BM):** native oracle runs, the whole-program bitcode pipeline is established
   (833 modules → one 78 MB `.ll` the reader ingests + fail-closes on), and the translator gap list
   is quantified — inline-`asm` (~9 templates), `atomicrmw`, `i128`, vector-memory, the fs/syscall
   shim. Config levers (portable atomics, no-`int128`) + a small asm recognize-and-lower table clear
   the top blockers. See `crates/svm-run/demos/postgres/` for the reproduction.
8. **The WebGPU capability** (its own section below) and **the network/egress capability** (curl/git) —
   the remaining capability frontiers.

### `setjmp`/`longjmp` + EH — design & sequencing (the stack-transfer substrate)

**Status: `setjmp`/`longjmp` DONE on all three engines (slice AX + the JIT sub-slice)** — see the gap
bullet above for the landed design + engine matrix, and "JIT `longjmp`" below for the JIT specifics.
**EH** (`invoke`/`landingpad`/`resume` + cleanups) is the remaining sub-slice on this substrate. The
design notes below are retained for the EH follow-on. Drivers: **Postgres** `--single` (its whole
`PG_TRY`/`ereport` model is `sigsetjmp`/`siglongjmp`) and **Lua**.

**Why a core primitive (not a pure-IR transform).** `longjmp` transfers control from deep in the call
stack back to an ancestor `setjmp`, across N intervening frames. The only way to do this *without a
primitive* is to make every call-return check "am I unwinding?" — which **taxes intervening frames**
(the SJLJ-via-return-checks model). To keep intervening frames untaxed (the perf goal), the VM needs a
real **"unwind to a saved stack checkpoint"** primitive — matching the "VM ships primitives" thesis.
Plan: two svm-ir ops (or one checkpoint + one unwind) lowered from the recognized external `setjmp`/
`_setjmp`/`sigsetjmp` (returns-twice) and `longjmp`/`siglongjmp` (noreturn) calls — gated on the program
calling them, like every other capability, so non-users get nothing.

**Interp (tractable — do first).** The interpreter already runs on an **explicit `Vec<Frame>` guest
call stack** with reified continuations (for fibers). So `setjmp(env)` saves `(frame depth, data-SP,
resume pc + result slot)` into the guest `jmp_buf` and returns 0; `longjmp(env, v)` truncates `frames`
back to that depth (the intervening frames discarded with **no per-frame work** — C has no cleanups),
restores the data-SP, and resumes at the saved pc with the `setjmp` result set to `v`. O(1)-ish capture,
O(depth-discarded) unwind, both only when called.

#### JIT `longjmp` — DONE (Option B; `crates/svm-jit/src/setjmp_rt.rs` + the `SetJmp`/`LongJmp` lowering)

Cranelift code runs on the real native control stack, so `longjmp` must restore the native SP to the
`setjmp` point. **Approach — Option B: call libc `_setjmp`/`_longjmp` from JITted code, with the
`jmp_buf` in a host-side table.** Rationale + the rejected alternatives are below.

**What landed.** A per-run host `jmp_buf` table (`setjmp_rt::SetjmpRuntime`, owned by the
`CompiledModule` so its address stays valid for re-runs) keyed by the guest `jmp_buf` window address;
`Inst::SetJmp` lowers to `rt_setjmp_slot` (alloc/find the host slot) then an **inline** `call _setjmp`,
and `Inst::LongJmp` to `rt_setjmp_lookup` (find, or write the `SetjmpFault` trap cell on a miss — a
fail-closed §3b totality check) then an inline `call _longjmp`. The libc `_setjmp`/`_longjmp` addresses
are baked **directly** as the call targets (not via a Rust thunk, whose frame would be gone by
`longjmp` time), mirroring the `cap.call`/fiber-thunk "call-to-baked-host-address" template. Gated to
the `setjmp_rt` cfg (= `fiber_rt && unix`; built in `build.rs`) — Windows `setjmp` is SEH-coupled, a
follow-on, and there the JIT keeps declining and the interpreters cover it. A §14 JIT child using
`setjmp`, and any module mixing `setjmp` with fibers/threads, are **fail-closed-declined** (no per-child
/ per-fiber `setjmp` table yet — the per-run table is keyed only by buffer address, so concurrent
shared-buffer use across native stacks would corrupt; the interpreters' per-vCPU keying covers it). The
returns-twice hazard held up: clang's `-O2` setjmp-crossing spills + Cranelift's caller-saved spills
across the `SetJmp` call site mean live values arrive as loads/stores, not long-lived SSA, and the
guest's data-stack values ride in window memory (preserved across `_longjmp`). Validated by the on-ramp
differential (all three engines == native, incl. value-live-across-setjmp + nested-buffers) and the
`ASan (JIT setjmp/longjmp)` CI lane.

**Why Option B (and not the others) — decision record:**
- **Option B (chosen): libc `_setjmp`/`_longjmp` called inline from JITted code.** Reuses battle-tested
  libc; no custom asm. The one trick that makes it correct: the `_setjmp` call must be emitted **inline in
  the JIT function's own frame** (the frame we long-jump back to) — *not* in a host helper thunk that
  `setjmp`s and returns, because that helper's frame is gone by `longjmp` time (`longjmp` to a returned
  frame is UB). So `SetJmp` lowers to *two* calls: a thunk that hands back the host `jmp_buf` pointer, then
  an **inline `call _setjmp(buf_ptr)`**.
- **Option A (rejected): a new arch-specific in-place context-save/restore asm primitive** (save
  callee-saved + SP + return-addr without switching; one file per ABI like `svm-fiber`'s `switch_*`).
  Strictly more risk (hand-written unsafe asm per ABI) for no benefit over B unless libc `_setjmp` has a
  concrete problem — fall back to A only if B hits a wall.
- **Rejected: the existing `svm-fiber` switch (`jump`/`make`, boost.context `fcontext`).** It is an
  *asymmetric-coroutine* primitive — the captured `fctx` is a transient register block on the *suspended*
  stack, valid only while suspended; the moment you `jump` back and keep running it is reused/stale.
  `setjmp` needs **capture-in-place + keep running**, then `longjmp` from a deeper frame. Doesn't fit.
- **Rejected: per-function interp fallback** (JIT declines `setjmp` modules). It is the current safety net,
  but module-granular: a Lua-as-a-guest would lose JIT speed on its *hot computed-goto loop* just because
  `pcall` uses `setjmp`. Keep it only as the fallback, not the goal.

**Implementation steps (concrete):**
1. **Host-side `jmp_buf` table + thunks** (new, in/next to `crates/svm-jit/src/fiber_rt.rs`, per-run like
   the fiber runtime): `rt_setjmp_slot(ctx, guest_buf_addr) -> *mut JmpBuf` (alloc/find a host slot keyed
   by `(ctx, guest_buf_addr)`) and `rt_setjmp_lookup(ctx, guest_buf_addr) -> *mut JmpBuf` (or trap-marker
   on miss). **The `jmp_buf` is host-allocated and lives in this table — never the guest window** (it
   holds host SP / return-addr / callee-saved; leaking those into guest-readable memory defeats ASLR; the
   guest `jmp_buf` address is *only the key*). This mirrors the interp's per-vCPU `setjmp_points`.
2. **Cranelift lowering** (the JIT supportedness check + codegen): `Inst::SetJmp { buf }` → `slot = call
   rt_setjmp_slot(ctx, operand(buf))`, then **inline** `r = call _setjmp(slot)`, bind `r` (i32) as the
   result. `Inst::LongJmp { buf, val }` → `slot = call rt_setjmp_lookup(ctx, operand(buf))`, then `call
   _longjmp(slot, operand(val))` (noreturn; the trailing `unreachable` is dead). Bake `&_setjmp`/`&_longjmp`
   as call-target constants (declare them `extern "C"`), exactly as `cap.call` bakes `cap_thunk as usize as
   i64` and the fiber thunks bake their addresses — the **call-to-baked-host-address template** is the
   `cap.call` / `cont.*` lowering already in the crate.
3. **Un-bail + gate.** Remove `SetJmp`/`LongJmp` from the JIT's `_ => Err(JitError::Unsupported)` path (the
   supportedness check ~`crates/svm-jit/src/lib.rs:3640`, the `if cfg!(fiber_rt)` block just above lists
   the ops needing the runtime). Gate to the Unix targets first (like `fiber_rt`); the interpreters cover
   the rest. Thread the per-run `jmp_buf` table through the compile/run setup like the fiber runtime is.
4. **Flip the test.** `check_setjmp_vs_native` (`crates/svm-llvm/tests/translate.rs`) currently asserts the
   JIT *declines*; change that to run the JIT and assert it agrees with the interpreters + native.

**The one real hazard — returns-twice vs Cranelift — and why it's largely defused (still must be proven):**
`Inst::SetJmp` is a *call site*, so Cranelift already spills caller-saved values across it; callee-saved
are saved/restored *by `_setjmp`/`_longjmp` themselves*; and clang already spilled every `setjmp`-crossing
local to memory at `-O2` (returns-twice at the LLVM level), so they arrive as loads/stores in the SVM IR,
not long-lived SSA across the `SetJmp`. A value live across the `setjmp` in a callee-saved reg is restored
to its `setjmp`-time value by `_longjmp` — correct for pre-`setjmp` values; values *modified* after
`setjmp` are C-indeterminate anyway. **Verification bar (gating): a native differential on the JIT path
run under ASan** (this is memory-unsafe native-stack manipulation; it gets `svm-fiber`-grade rigor —
landed as the dedicated `ASan (JIT setjmp/longjmp)` CI lane alongside "ASan (svm-fiber switches)", running
`cargo test -p svm-llvm … setjmp` under `-Zsanitizer=address`). Returns-twice stress cases landed: a value
live across the `setjmp` (`setjmp_value_live_across`), nested buffers (`setjmp_nested_buffers`), and the
loop / deep-nesting `longjmp` across several frames (`setjmp_longjmp_loop_and_deep_nesting`).

**Reference implementations to mirror (already landed, byte-identical to native):** the tree-walker
(`crates/svm-interp/src/lib.rs` — `SetJmpPoint`, `setjmp_points`, the `SetJmp`/`LongJmp` arms in
`run_inner`) and the bytecode engine (`crates/svm-interp/src/bytecode.rs` — `ByteSetJmp`, `Op::SetJmp`/
`LongJmp`). The on-ramp lowering is `lower_setjmp_call` in `crates/svm-llvm/src/lib.rs`; the JIT path is `setjmp_rt.rs` + the `SetJmp`/`LongJmp` lowering arms in
`build_clif`. **Current state: all three engines run `setjmp`/`longjmp`** (the JIT via Option B, above),
byte-identical to native, engines in sync, never divergent.

**Returns-twice in SSA.** `setjmp`'s result (0 first time, `v` on `longjmp`) feeds a branch; the
`longjmp` re-enters at the instruction after `setjmp`. The interp models this directly (set the result
slot + resume pc). The `jmp_buf` holds **backend-internal** state (interp frame index vs JIT native SP)
— opaque to the guest, so *observable* behavior matches across backends (the determinism contract is
about stdout/results, not `jmp_buf` bytes); it is transient, not snapshot-portable across backends.

**Perf (measured, not assumed — `examples/onramp_perf.rs`).** Non-users: **0** (gated). `setjmp`-using
functions inherit clang's returns-twice spills (already in the bitcode — the on-ramp adds none) and the
ops cost O(1) capture / O(1)+unwind, paid only when executed; intervening frames untaxed. A real
workload's hot path (Lua's interpreter loop) has no `setjmp` in it — cost is confined to `pcall` entry
and (rare) error unwinds. The harness establishes the non-`setjmp` baseline now; a `setjmp` happy-path
row + a `longjmp`-taken row slot in beside it once the ops land, so the feature's cost is measured.

### GPU via a WebGPU capability (graphical + compute)

Graphical and GPU-compute software rides a **WebGPU capability** — and WebGPU is the *right* abstraction
precisely because it was **designed for the browser sandbox**: it is already capability-shaped. No raw
pointers, no arbitrary memory — everything is validated buffers/textures/bind-groups, shaders are
**WGSL** (validated, no arbitrary memory access), and work is submitted as structured command buffers.
Exposing raw Vulkan/Metal/CUDA would blow the TCB open; WebGPU is the *safe waist* other GPU APIs lack.
Same thesis as the §22 `Jit` cap (§2a): **verification, not raw access, is the boundary** — and here
WebGPU does the validation for you (guest-authored WGSL is safe by construction).

**Architecture (mirrors the §9/§12 IoRing pattern).**
- The host holds the real device — via **`wgpu`** (the Rust WebGPU implementation; fits the codebase).
- The guest is granted a **WebGPU handle** in the powerbox stash (one more `grant_*` on `Host`, slotted
  into `synth_start`'s contiguous prefix exactly like `IoRing`/`Blocking`/`Jit` — likely **no translator
  change**). API calls (create buffer, write, create pipeline, dispatch/submit) cross the boundary as
  **structured, validated commands**; the guest never holds a GPU pointer.
- Only **data** flows back into the window (a compute buffer's contents, a texture's pixels). Async
  readback (`mapAsync` / queue completion) maps onto the existing **IoRing/Blocking + M:N executor** —
  GPU work is the same submit/validate/return-data shape as I/O.

**Demos that prove safe access (headless first — the safety story is complete with no display):**
1. **GPU compute → readback → assert vs CPU** *(the ideal first demo)* — a WGSL compute shader
   over a guest-uploaded buffer, read back and checked against a CPU reference. Proves the whole data
   plane (upload → dispatch → readback) + the validated-buffer model, **with zero windowing**. —
   **DONE.** The new **workspace-excluded `crates/svm-webgpu`** crate holds a `webgpu` [`HostCap`]
   backed by **`wgpu`**: an op protocol (create_buffer / write_buffer / set_shader / dispatch /
   read_buffer) the guest drives over the generic §7 `__vm_cap_resolve("webgpu")` + `__vm_host_call`
   surface — **no translator change** (exactly the doc's prediction; the same shape as the `fs`/LMDB
   host-fn caps). `demos/webgpu/{wgpu_demo.c,wgpu_shim.c}`: a guest uploads a `u32` buffer, sets a WGSL
   kernel, dispatches, reads back, and re-checks the result against a CPU reference **in-guest** — two
   kernels (an in-place `b[i]=b[i]*3+7` single-buffer map, and a two-buffer `c[i]=a[i]*a[i]+i` at
   `@binding(0)`/`@binding(1)`). Test `demo_webgpu_compute` runs it through the on-ramp on **all three
   engines** (tree-walk/bytecode/JIT), asserting the guest's self-check (`ALL MATCH cpu`); no native
   oracle (the cap is SVM-only, like the async/JIT demos). Runs headless with **no physical GPU** via
   **lavapipe** (`force_fallback_adapter` → llvmpipe software Vulkan); the wgpu build is isolated to
   this opt-in crate (like `svm-llvm`), so the default `cargo test` is untouched. **CI note:** the
   webgpu lane installs `mesa-vulkan-drivers`; without an adapter the test **skips cleanly**
   (`adapter_available()`). Follow-ups: promote the cap into `svm-run` behind a feature; the
   offscreen-render (demo 2) + Mandelbrot→PNG (demo 3) slices; richer bind-group/uniform support.
2. **Offscreen render-to-texture → read pixels → hash** — "hello triangle" / a rotating cube to an
   offscreen attachment, pixels compared to a reference image. Proves the render pipeline headlessly.
3. **Compute image filter / Mandelbrot / a Shadertoy-style shader → PNG** (written via the File cap) —
   visually compelling, deterministic, self-validating; a great screenshot artifact. — **DONE.**
   `demos/webgpu/mandelbrot.c`: a WGSL compute shader renders a Mandelbrot into an RGBA buffer, the
   guest reads it back and writes the raw pixels out through the granted **`fs` cap** (compute + fs caps
   **together**), and the host test encodes the PNG. Self-validated **without a fragile hardcoded hash**:
   the imaginary-axis mapping is *center-relative* (`y0 = (py − 119.5)·scale`), so mirrored rows are
   exact IEEE negatives and Mandelbrot conjugate symmetry makes row `py` equal row `H−1−py`
   **bit-for-bit** — a float-implementation-independent invariant — plus an in-set/escaped sanity pair.
   Test `demo_webgpu_mandelbrot` runs it through the on-ramp on **all three engines**, asserts the
   self-check, and encodes the guest-written RGBA to a PNG. (A latent finding: computing `y0` from
   `(py+0.5)/H·range−offset` is *not* bit-symmetric under f32 rounding — the center-relative form is.)
4. **On-screen presentation (a `Surface`/swapchain capability) — defer.** Presenting to a host window is
   the OS-coupled part and a later capability; the **safe-access** narrative is fully told headless.

Riders: any C/C++/Rust app on **wgpu-native** or **Dawn** (the WebGPU C API) — learn-wgpu / WebGPU
samples — plus `wgpu`'s own CTS as a (meta) conformance check.

### Stretch goal — self-hosting: LLVM in the sandbox

The moonshot beyond SQLite: run **LLVM itself** as an SVM guest. Not because LLVM "happens to run,"
but because it closes a **self-hosting, capability-secured toolchain** loop — the sandbox hosting its
own means of production. With LLVM in-guest and the existing §22 `Jit` capability (a guest emits SVM IR
→ the host Cranelift-compiles it), the whole pipeline runs *inside the sandbox*:

```
C/C++/Rust source → clang/LLVM (in-guest) → LLVM IR
                  → svm-llvm translator (in-guest) → SVM IR
                  → Jit capability → native code
```

and the translator is itself Rust (which compiles Rust→LLVM→SVM), so the chain is self-hostable end to
end. The payoff: a compiler-as-a-sandboxed-service, reproducible builds in a box, untrusted code that
can compile *and* run other untrusted code with **zero ambient authority**. SQLite proves "a real
program with real I/O under capabilities"; LLVM proves "the sandbox can host its own toolchain."

**Scoping insight — the backend is (mostly) already ours.** "Run LLVM" is not one thing, and the
single largest/ugliest part of LLVM — the **target backends + MC layer** (SelectionDAG/GlobalISel,
object emission) — is **out of scope**: SVM already has a backend in the `Jit` capability. The target
is LLVM's **front + middle end** (IR data structures, the bitcode reader, the verifier, the `opt` pass
pipeline), lowering to SVM IR and handing it to `Jit`. That deletes roughly the hardest third of LLVM.
Slices, smallest meaningful first (do **not** aim at clang first):
1. **libLLVMCore in isolation** — a program that builds an IR `Module` in memory, runs the
   **verifier**, and prints it. The "hello world of embedding LLVM," dramatically smaller than clang,
   and already proves the thesis.
2. **`llvm-as` / `llvm-dis`** — bitcode ↔ text round-trip.
3. **`opt` with a few passes** — guest-side optimization.
4. *(moonshots)* **`llc`** (if a non-`Jit` codegen path is ever wanted), then **clang** (C/C++ front
   end — millions of lines: the driver, preprocessor, AST, Sema, CodeGen).

**In our favor (de-risks it):**
- **LLVM builds `-fno-exceptions -fno-rtti` by default** (`LLVM_ENABLE_EH/RTTI` off) — it lands on the
  exact C++ subset slice AG proved (classes, vtables, templates→monomorphized, global ctors). The
  on-ramp has *no* EH path, and LLVM needs none.
- **The window reserves `1 << 40` (1 TiB), lazily paged** (`DEFAULT_RESERVED_LOG2`) — address-space
  scale is not the blocker; the heap grows to gigabytes through the `Memory` cap.
- **APFloat** is LLVM's own arbitrary-precision float — self-contained, no libm dependency for the hard
  numerics.
- **Templates monomorphize + global ctors run** (slice AG) — both heavily used (cl::opt registration,
  ManagedStatic).

**The real mountains (honest list):**
1. **The libc++ + libc surface — the dominant cost.** LLVM exercises a *huge* slice of the C++ standard
   library + syscalls (`std::vector`/`map`/`string`, `DenseMap`, `<algorithm>`, `std::error_code`,
   `mmap`, file I/O, `getpagesize`, time, `errno`). Most of libc++ is header-only templates (fine), but
   the non-header parts + the syscall floor need a real port. This is gating infrastructure — and it is
   **independently valuable**, because it is what *any* large C++ program needs (so growing C++/stdlib
   breadth is the honest first rung, see the ladder above).
2. **C++ `thread_local` / TLS** — `@llvm.threadlocal.address` / `.tdata`/`.tbss` is **not lowered yet**
   (the existing `__vm_vcpu_tls` is a different, SVM-specific per-vCPU register). LLVM uses
   `thread_local` for errno/ManagedStatic — a genuinely new feature to build.
3. **File I/O capability** — LLVM reads/writes files: the **same** powerbox `File`/`Storage` capability
   SQLite Phase B needs. Shared prerequisite, not extra.
4. **Translate-time scale** — the `svm-llvm` translator is demo-sized today; a multi-hundred-MB bitcode
   module is a different regime (translator memory + throughput). Needs measurement, possibly streaming.
5. **Stragglers** — a few inline-asm spots (mostly avoidable), possibly computed-goto in clang's
   lexer (the `indirectbr` work above). (`%f`/`%g` exact-decimal formatting is DONE.)

**Why it sequences after SQLite (not a detour).** SQLite and LLVM share their three biggest
prerequisites — the **File/Storage capability**, **float formatting**, and **large-program scaling** —
so SQLite is squarely on the critical path. The one LLVM-specific track SQLite does *not* exercise is
**big C++ + a real chunk of the standard library**; that is the first rung toward LLVM and is worth
pushing in parallel (a substantial real C++ library through the on-ramp). Sequence:
**SQLite (shared infra) ∥ grow C++/stdlib breadth → libLLVMCore build-verify-print → llvm-as/opt →
moonshot llc/clang.**

### Milestone 2 — beyond chibicc's C subset 🟡
- [x] **C++ without EH/RTTI** — first light (slice AG): classes, vtables/virtual dispatch, `new`/`delete`,
      virtual dtors, templates, static init via `@llvm.global_ctors`. Broaden as gaps surface (multiple
      inheritance / `this`-adjusting thunks, references, `static`-local guards, …).
- [x] **Rust** (`no_std`/panic=abort) — runs vs native: `iN` (slice AH), real `core` (enums/slices/
      `Option`/iterators/structs) + panic-path → trap (slice AI), trait objects / `&[T]` args / `unwrap`
      (slice AJ), `alloc`/heap `Vec` (slice AK), **`Box` recursive AST + `String`** via a mini expr
      evaluator (slice AL — + `*.with.overflow` intrinsics and `i64` switches). Auto-vectorization is
      disabled (SIMD is §17); `--edition 2021`. Broaden (`Result`/`?`, `BTreeMap`, generics with
      bounds, `&mut` aliasing) as gaps surface.
- [x] **SIMD / auto-vectorization — full `-O2` ingestion; the breadth lanes vectorize (slices AN–AT).**
      The on-ramp ingests real `-O2`/`-mavx2` auto-vectorized bitcode end to end, so the C/C++/Rust
      breadth lanes dropped `-fno-*-vectorize`. The pipeline:
      - **128-bit lane ops** (slice AN/AO): a `<4 x i32>`/etc. lane op → `v128` (`VIntBin`/`VBitBin`/
        `VFloatBin`), `llvm.vector.reduce.*` → a lane fold, `llvm.{s,u}{min,max}` → `VIntBin`.
      - **Vector legalization** (I2 / PR #56): the fixed-128 SelectionDAG-`LegalizeTypes` analog
        (`wide_vec_layout`/`lower_wide`) splits wider-than-128 / sub-128 vectors into `v128` chunks +
        a scalar tail; all six 128-bit shapes lower. Fixed-128 chunking (not host detection) preserves
        interp↔JIT/durable determinism.
      - **Integer conversions** (slice AP): `zext`/`sext`/`trunc` scalarize lane-wise across every
        representation. **Rotate** (slice AQ): `llvm.fshl`/`fshr` (rotate idiom) scalarize per lane.
        **Shuffle** (slice AR + AT): general constant-mask gather, incl. wide and cross-representation
        (`<2 x float> ++ <2 x float>` → `<4 x float>`). **`<N x i1>` masks** (slice AS): a lane-wise
        scalarized `mask_lanes` rep — vector `icmp`/`fcmp`/`select`/`extractelement`/`insertelement`/
        `bitcast`-to-`iN` (movemask). **Float conversions + vec2 FMA** (slice AT): `fptosi`/`sitofp`/…
        and `fmuladd.v2f32` scalarize lane-wise.
      - Verified by the whole 128-test suite running vectorized (the real C/C++/Rust corpus + focused
        `simd_*` shape/op-class pins). Remaining fail-closed (no corpus need): a *general* (non-rotate)
        funnel shift, a *non-constant* shuffle mask, a wide float `vector.reduce`, a mask crossing a
        block edge.
- [ ] Tail calls (`musttail` → `return_call`), if any corpus needs it (likely near-free).
- [ ] Narrow-atomic CAS-loop emulation (§3b note 2), on demand.
- [ ] Signed-`iN` ops (`ashr`/`sdiv`/`srem`/`sext`-to-`iN`/signed `icmp`-`iN`) — on demand (rare; `-O2`
      uses `i64` for signed div/rem-by-constant, not `iN`).

### Deferred / hard (name them, don't hide them — DESIGN §20) ⚪
- [ ] **`setjmp`/`longjmp` + C++ exceptions/unwinding — PROMOTED to planned substrate** (no longer
      deferred). Both lower onto §6 stack-switching/`cont.*`; build `setjmp`/`longjmp` first (Postgres
      `--single` / Lua force it), then EH (`invoke`/`landingpad`/`resume` + unwind tables, the §18 open
      item) on the same core. Full plan + ordering + drivers in *Next frontier → Translator gaps*.
- [x] **Computed `goto` — `indirectbr` / `blockaddress` — DONE (slices AV + AW).** Both global dispatch
      tables (AV) and operand-position φ-threaded blockaddresses (AW), recovered via `llvm-sys`
      (`blockaddr.rs`), lowered to a `br_table` over block indices. Robust for real interpreters.
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

**Debug-info nuance (the §6/D-DBG-7 waist).** The "debug metadata gap" above is *one-sided*:
`llvm-ir` **does** expose per-instruction `!DILocation` (line/col/file, via `HasDebugLoc`), so the
on-ramp populates the §6 neutral core's **source-line half** from it (`DebugAcc` in
`crates/svm-llvm/src/lib.rs` → `DebugInfo.locs`; DEBUGGING.md slice 24) — making LLVM the *third*
producer to feed the frontend-neutral waist. The **structured DI graph**
(`DILocalVariable`/`DIType`/`llvm.dbg.value`) is missing from `llvm-ir` (`Metadata::from_llvm_ref` is
`unimplemented!`, `MetadataOperand` is payloadless), so the **fallback-reader** decision above is now
realized concretely: `crates/svm-llvm/src/di.rs` walks the DI nodes **directly through `llvm-sys`**
(the LLVM-C debug-info API), re-parsing the `.bc` into its own context. Slice 25 lands the `-O0 -g`
case — every C local is an `alloca` + `dbg.declare`, recovered as a `TypeDef`-typed `VarLoc::Window`
correlated to the IR by *alloca ordinal* (stable across the two parses). The LLVM-C DI API has no
getters for the `baseType`/`elements` edges or the base-type `encoding`, so the type graph is walked
via the generic MDNode-operand bridge at the positional indices LLVM 18 uses (pinned + tested), and
`encoding` is inferred from the C name. Slice 26 adds the `-O2`/`-Og` `dbg.value` case (promoted
scalars, which LLVM solves for free — its intrinsics survive mem2reg/SROA): a `dbg.value` bound to a
function **argument** becomes a `VarLoc::SsaList` over the arg's live range (the arg is ValueId `k`,
threaded as a block parameter, so its block-local index is its position in each block's param list).
At `-Og`/`-O2` most *other* locals are optimized to `poison`/constants, so parameters are the main
recoverable variable; `dbg.value` bindings to instruction-result / φ values (needing a value→ValueId
ordinal correlation and per-pc `SsaLoc.inst`) are a follow-up of limited yield.
**Fallback order if `llvm-ir` bites:** `inkwell` (maintained, version-tracking wrapper) →
hand-rolled `.ll` parser over `opt -S` (zero libLLVM link, but a rot-prone parser we own).

**Q1a — The `llvm-ir` version-lag bottleneck, and the textual-IR exit (forward-looking).** Two of the
on-ramp's limitations share one root — binding to `llvm-ir`'s LLVM-C reader, which is **lossy**
(collapses every integer constant to a `u64`, truncating wide/negative `i128` literals — ISSUES.md I14,
now fail-closed by `wideint`) and **version-locked** (tops out at LLVM 19). The second is the bigger
long-term drag: the reader pins the whole on-ramp to LLVM 18, and forces pinning *old producers* to feed
it — CI installs **rustc 1.81 (LLVM 18.1)** because the container's default `rustc` emits **LLVM 21**
bitcode `llvm-ir` can't read (slice AH). As of this writing rustc is on LLVM **21**, the container's
clang on **18**, `llvm-ir` caps at **19** — a gap that widens every LLVM release (~2/yr). `llvm-ir` is
maintained but slow (last release Feb 2025; commits through Jan 2026; single maintainer), and
forking/vendoring it fixes only the *constant* loss while **inheriting the version ceiling** — reaching
LLVM 21 would mean adding 20/21 support to the fork (real LLVM-C-API work) *and* getting `llvm-21-dev`
into the container. So a fork is a bandaid, not an exit.

The exit already named in the fallback list — **a hand-rolled `.ll` textual reader** (`clang -S` /
`llvm-dis`, zero libLLVM link) — addresses *both* at once: textual IR prints **full-width constants**
(no truncation, no `wideint` guard) and is **far more version-stable** than the bitcode format or the
C/C++ API (new LLVM versions *add* syntax rather than break existing syntax), so we'd ingest 20/21/22 by
reading text instead of chasing a binding. It would also **collapse today's 3–4 libLLVM re-parses**
(`from_bc_path` + `di` + `blockaddr` + `wideint`, each its own context) into **one** text pass carrying
everything (instructions, `!DILocation`, `blockaddress`, full constants), and **remove the libLLVM build
dependency entirely** (the reason `svm-llvm` is excluded from the workspace).

*Tradeoffs, so the next person can weigh it:*
- **Speed.** `.ll` is bigger than `.bc` (≈3–10× bytes) and text-parsing is slower per byte than bitcode
  decode, so the *parse sub-step* is slower — but parse is a small fraction of an **AOT, once-per-program**
  translate (the ~17 k-line emit walk dominates), `clang -S` costs ≈ `clang -c`, and one text pass
  replaces four libLLVM parses. Net end-to-end is plausibly a wash or better; **measure before assuming**.
- **Prior art.** In-process consumers (rustc's codegen, the real backends) use the C++ API on in-memory
  IR — not our cross-process case. Out-of-process / cross-language consumers split: **bind the C API +
  bitcode** (`llvm-ir`, `inkwell` — version-locked + lossy, our exact pain) vs. **parse `.ll` text**
  (e.g. Go's `llir/llvm`, pure-Go, chosen precisely to avoid the CGo/libLLVM binding and tracking LLVM by
  grammar updates). The textual route is the established way to be version-tolerant and dependency-free.
- **Work, vs. forking `llvm-ir`.** Forking is *less upfront* (vendor + ~6-line patch) but doesn't solve
  the version problem and keeps the libLLVM dep + the multi-parse. A textual reader is *more upfront* — a
  tokenizer + parser for the **~52 instruction kinds** the on-ramp handles, plus types / constants / the
  DI metadata it reads — but **bounded by current coverage** (the on-ramp fail-closes on anything it
  doesn't translate, so the parser only needs *what we emit*, and grows incrementally), and **less
  ongoing** for version bumps. By horizon: stay on LLVM 18 → forking is cheaper; track current Rust/clang
  long-term → the textual reader wins (and `llvm-ir` may not even support the version you need).

**Status: IN PROGRESS** (decided to build it now — `llvm-ir` kept biting, both on constants and the
version ceiling). See the handoff block below for the plan + current state.

**Q1b — Textual `.ll` reader: migration plan & handoff.** Replacing the `llvm-ir`/libLLVM binding
with a dependency-free textual-`.ll` reader. Approach, validation, and the staged sequence:

- **Approach.** Keep the ~17k-line translator (`lib.rs`) unchanged; replace *only* the input layer.
  A new `crates/svm-llvm/src/ll/` module: `ast.rs` mirrors the slice of `llvm-ir`'s data model the
  translator consumes (same variant/field names, same `get_type`/`try_get_result`/`named_struct_def`
  methods) but **owned by us** — no FFI, no version lock — with the **I14 fix baked in** (`Constant::Int`
  is a full `u128`). `lex.rs` tokenizes; `parse.rs` is a recursive-descent reader → `ast::Module`. The
  translator's `use llvm_ir::…` becomes `use crate::ll::…` (mechanical). One text pass also absorbs
  what `di.rs`/`blockaddr.rs`/`wideint.rs` re-walk today (4 libLLVM parses → 1, no libLLVM link).
- **Validation oracle = `llvm-ir` itself.** A differential parity check (`assert_ll_parity`): compile
  each test to *both* `.bc` and `.ll`, translate both, assert identical svm-ir. Keep `llvm-ir` as a
  dev-dep until parity holds across the corpus; only then flip the default and drop it. The existing
  215-test suite + this parity check are the regression gate. The translator's matches are the exact
  spec — complete the `Instruction`/`Constant`/`Terminator` enums by compiling `lib.rs` against `crate::ll`.
- **Staging (≈4 PRs).** (1) AST + lexer + core parser + parity harness, no flip. (2) widen to full
  non-`-g` parity (SIMD/AVX, C++ EH, structs, i128, blockaddress — the last two come free in text).
  (3) debug-info metadata (`!DILocation`/`!DISubprogram`/`!DILocalVariable`/`!DIType`) replacing
  `di.rs` — the hardest, separable slice (the 6 `-g` tests + the DAP test). (4) flip + drop `llvm-ir`/
  `llvm-sys`/`di.rs`/`blockaddr.rs`/`wideint.rs` + the rustc-1.81 pin; prove version-tolerance *here*
  by feeding `rustc 1.94`'s LLVM-21 `.ll` (`rustc --emit=llvm-ir`) through the parser. (CI `ci.yml`
  edits need `workflow` scope the bot lacks → manual follow-up.)
- **Current state (branch `claude/ll-translator-swap`, off latest `main`; PR #158).** The lexer +
  parser-core-slice landed via PR #151 (merged to `main`); this branch carries the AST expansion **and
  the translator type-swap + conversion shim** on top. **The translator now consumes the owned `ll`
  AST** — `lib.rs`'s `Instruction`/`Constant`/`Terminator`/`Type`/`Operand`/`Types` are
  `crate::ll::ast::*`, fed on the bitcode path by [`from_llvm_ir`]; the textual reader will feed it
  directly. **All 223 tests pass** (the whole bitcode corpus translates byte-identically through the
  `ll` AST), fmt + clippy (`-D warnings`) green. Done so far:
  - `ast.rs` — **the data model + type system are now complete** (the swap prerequisite, "until the AST
    enums are complete"). The full **`Instruction`** set (all 48 LLVM-18 variants), the full
    **`Terminator`** set (all 12), and the **`Constant`** set incl. the const-expr ops the on-ramp folds,
    each modeling **only the fields the translator reads** (`nuw`/`nsw`/`exact`/calling-conv/attributes
    omitted — the consumed-subset convention, so we avoid `llvm-ir`'s attribute/CC types). The type
    system is the **read-only** `Types` (no interning — structural `PartialEq` on `TypeRef` makes equal
    types compare equal; constructors hand back a fresh `TypeRef`) with `type_of`/`named_struct_def`
    (`NamedStructDef::{Opaque,Defined}`), and faithful **`Typed`** impls for `TypeRef`/`Operand`/
    `Constant`/`ConstantRef`/`Float`/`Instruction` (the per-instruction result-type logic mirrored from
    `llvm-ir`: binop→operand0, conversion→`to_type`, `icmp`/`fcmp`→`i1`/`<N x i1>`, `cmpxchg`→`{ty,i1}`,
    opaque `alloca`/`gep`→`ptr`, `call`→`function_ty` result, …). Plus `Instruction::try_get_result`,
    `Operand::as_constant`, `HasDebugLoc`, a dependency-free `Either`, and minimal `InlineAssembly`/
    `ParameterAttribute`/`Atomicity`/`RMWBinOp`/`LandingPadClause`. The `u128` I14 fix is retained.
  - `lex.rs` — **complete**, fully unit-tested: `%`/`@`/`!` names (numbered/quoted), full-width int +
    decimal/hex float literals (kept as text, so no truncation), strings (escapes left encoded;
    `unescape` decodes), types, punctuation, attribute-group refs, debug-metadata flag-sets.
  - `parse.rs` — **core slice only** (not yet grown to the full AST), 4 unit tests: `define` functions
    (integer types, binary ops, `ret`/`br`/`condbr`/`unreachable`, `%local`/const-int operands),
    skipping top-level cruft (target/datalayout, attribute groups, module metadata, `declare`) via a
    balanced-delimiter scan. The I14 fix is proven end-to-end (`full_width_i128_constant_survives`
    round-trips `i128::MAX`). Out-of-slice constructs fail closed (clean `ParseError`).
  - **`from_llvm_ir.rs` (the bc→ll conversion shim) — DONE.** `translate_impl` now takes a
    `crate::ll::Module`, so the bitcode path decouples: `translate_bc_path` reads an `llvm_ir::Module`
    via `from_bc_path`, then `from_llvm_ir::convert_module` maps it field-for-field to the `ll` AST.
    **Faithful + fail-closed**: it drops the `nuw`/`nsw`/attribute/calling-conv fields the translator
    ignores, and rejects (clean `Unsupported`) anything outside the modeled subset — a non-folded const
    expression (`Xor`/`Shl`/`ICmp`/… as a constant), funclet EH (`catchpad`/`cleanupret`/`callbr`), a
    target-ext type. That is **zero-regression** because `translate_impl` already lowers *every* defined
    function + global initializer eagerly and rejects those same forms via its `const_eval`/`const_bytes`
    `other => unsup` arms, so a program that translates today converts losslessly and one that doesn't
    fails the same way. `llvm-ir`/`llvm-sys` stay normal deps (+ `either`, named directly for the callee
    `Either`); the shim is **transient** — it goes when they're dropped (PR4). The swap also needed a few
    `ll::ast` additions the translator relies on: `Deref` for `TypeRef`/`ConstantRef` (deref coercion),
    `Display` for `Type`/`TypeRef` (error messages), `HasDebugLoc` for `Instruction`; and `Constant::Int`
    being `u128` (I14) forced `as u64` casts at the ~7 size/priority/switch read sites (all ≤64-bit).
  - **`translate_ll_path`/`translate_ll_str` + the `assert_ll_parity` harness — DONE.** `lib.rs` now
    exposes the textual entry (`ll::parse::parse_module` → `translate`), and `tests/translate.rs` has
    `compile_to_ll` (`clang -O2 -emit-llvm -S`) + `assert_ll_parity(name, c_src)`: compile each test C
    to **both** `.bc` and `.ll`, translate both, assert byte-identical svm-ir (`svm_text::print_module`)
    and `entry_sp`. Three parity tests pass on real `clang -O2` output the **core-slice parser already
    handles**: `ll_parity_trivial_add` (header + flagged binop + `ret`), `ll_parity_arith_chain`
    (binop chain with `±` immediates), `ll_parity_bitwise_shifts` (`shl`/`lshr`/`or`). The textual path
    is proven end-to-end through the unified (`ll`-consuming) translator.
  - **Parser — single-block instruction set + multi-block/`phi` landed.** `parse.rs` now covers (each
    with an `assert_ll_parity` test on real `clang -O2` output): integer **+ float binops**,
    **conversions** (`trunc`/`zext`/`sext`/`fptrunc`/…/`bitcast`, with `nneg`/flag skipping),
    **`icmp`/`fcmp`** (full predicate sets + fast-math-flag skipping), **`select`**, **`fneg`/`freeze`**,
    **`phi`**, and **multi-block CFGs**. The multi-block enabler was **LLVM implicit slot numbering**:
    the unnamed counter is shared across (unnamed) params + blocks + value instructions, so `basic_blocks`
    seeds `next_unnamed` at the parameter count — an unlabeled entry then takes `%{nparams}` and the
    phi/branch refs into it match the bitcode reader's `Name`s. Eight parity tests green (`add`, arith
    chain, shifts, `sext`-widen, `icmp`+`zext`, `fadd`, a `for`-loop sum, an `if/else` diamond). Constant
    operands: int (full width) + `true`/`false`; **float constants are deferred** (LLVM's `0x…` hex-float
    image needs exact bit parsing — a small dedicated slice).
  - **Parser — `call` landed.** `parse.rs` now parses `[%d =] [tail] call [fmf] [cconv/ret-attrs]
    <retty>|<fnty> <callee>(<args>) [#N] [,!meta]`: direct `@f` (a `GlobalReference` whose `ty` is the
    **reconstructed** `<ret>(<argtys>)` function type — or the explicit `<ret>(<params>,...)` signature
    for varargs) and indirect `%fp` callees, the **result-less `void` call** (which forced
    `instruction()` to look ahead via `at_call()` before assuming the leading `%dest =`), and dropped
    arg/return attribute lists. Four parity tests green: direct 1-arg, direct 2-arg, `void`, and the
    `llvm.smax` intrinsic (the `a>b?a:b` lowering — the translator turns it back into `icmp`+`select`).
    **Twelve parity tests total.**
  - **Parser — memory landed.** `parse.rs` now parses `alloca [inalloca] <ty> [, <cty> <count>] [, align
    N] [, addrspace(N)]` (implicit count → the `i32 1` operand the bitcode reader materializes), `load
    [volatile] <ty>, ptr <addr> [, align N]`, `store [volatile] <ty> <val>, ptr <addr> [, align N]` (the
    **second result-less** instruction — the `at_call`/`store` lookahead in `instruction()` routes both),
    and `getelementptr [inbounds] [nuw|nusw] <srcty>, ptr <addr> [, <ity> <idx>]…`. Three parity tests:
    `p[i]` (gep+load), `p[i]=v` (gep+store), and a `volatile` local (alloca + volatile load/store + the
    `llvm.lifetime` intrinsics, which the call path already handles). **Fifteen parity tests total.**
  - **Parser — module-level globals landed.** `parse.rs` now parses `@g = [linkage/attrs] [addrspace(N)]
    global|constant <ty> [<init>] [, align N]` into `module.global_vars` (`ty` = the *pointer* type, as
    the bitcode reader records `LLVMTypeOf`; the written content type parses the initializer). New AST
    coverage: `[N x T]` **array types** in `type_()`, a `constant(&ty)` parser (int/`true`/`false`/
    `zeroinitializer`/`null`/`undef`/`poison`/`@g`-refs/`[…]` arrays/`c"…"` byte strings — the `c` prefix
    lexes as its own `Word`), and a `@name → pointee-type` **symbol table** (populated from each global +
    function as parsed) so a `@g` *operand* resolves to a `GlobalReference` carrying the pointee type the
    translator reads (e.g. GEP base). Also fixed `skip_to_toplevel_boundary` to stop before a depth-0
    `@global` (an unbraced `target … = "…"` line was swallowing the globals after it) and added
    `nonnull`/`noalias` to the return/pre-signature attribute skip. Three parity tests: a mutable scalar
    (`@counter`, load/add/store), a `constant [4 x i32]` lookup table (array type + array init + gep), and
    a `c"hi\00"` string returned as `ptr @.str`. **Eighteen parity tests total.**
  - **Parser — named struct types landed.** `parse.rs` now parses `%s = type { … }` / `%s = type opaque`
    top-level defs (→ `module.types.add_named_struct_def`, matching the bitcode reader's
    `all_struct_names` registration), and `type_()` resolves a `%s` **reference** (a `Local` in type
    position) + a literal `{ i32, i32 }` struct type. `skip_to_toplevel_boundary` also now stops before a
    depth-0 `%local` (same fix as `@global` — an unbraced `target … = "…"` line was swallowing the
    `%s = type …` defs after it). One parity test: struct field access (`%struct.P` + `getelementptr
    %struct.P, ptr %p, i64 0, i32 1`). **Nineteen parity tests total.**
  - **Parser — float constants landed.** `value_as_operand`/`constant` now decode `float`/`double`
    literals — decimal (`5.000000e-01`) or the `0x` + 16-hex-digit `double`-image form (used even for
    `float`) — into `Float::Single(f64 as f32)`/`Float::Double(f64)`, exactly matching the bitcode reader
    (`LLVMConstRealGetDouble`, cast to `f32` for single). `half`/`bfloat`/`fp128`/`x86_fp80`/`ppc_fp128`
    stay payload-free AST variants. Three parity tests: `x*0.5f` (decimal single), `x*3.14` (hex-image
    double), and a `constant [3 x double]` global (float aggregate init). **Twenty-two parity tests
    total.** This was the last deferred *primitive* — the parser now covers scalar/pointer/struct/array
    computation + memory + calls + globals end-to-end.
  - **Parser — `switch` + SIMD vectors landed.** `switch <ty> <v>, label %default [ <cty> <c>, label
    %l … ]` (whitespace-separated case entries) is parsed in `terminator()`. Vectors: `<N x T>` (+ packed
    `<{…}>` struct, `<vscale x …>`) types in `type_()`, `<T …>` vector constants, and
    `extractelement`/`insertelement`/`shufflevector` — the shuffle **mask canonicalized to a
    `Constant::Vector` of `i32` indices** (`zeroinitializer`/`poison`/explicit all normalized), exactly
    as the bitcode reader does via `LLVMGetMaskValue` (the translator's `<4×>` path *requires* `Vector`).
    `value_as_operand` was refactored to delegate every non-`%local` value to `constant()`, so
    `poison`/`undef`/`zeroinitializer`/`null`/vector constants all reach operand position. Four parity
    tests: sparse `switch`, `<4 x i32>` add, a splat (insertelement + shuffle), and a
    `llvm.vector.reduce.add` reduction. **Twenty-six parity tests total.**
  - **Parser — constant-expressions landed.** `constant()` now parses folded const-exprs: `getelementptr
    [inbounds] ( <srcty>, ptr <addr>, <idx>… )` → `ConstGetElementPtr`, the conversions
    `trunc`/`zext`/`sext`/`ptrtoint`/`inttoptr`/`bitcast`/`addrspacecast` `( <c> to <ty> )` →
    `Const{…}(ConstUnaryOp)`, and `add`/`sub`/`mul` `( <c0>, <c1> )` → `Const{…}(ConstBinaryOp)`
    (recursive, so nested `ptrtoint(getelementptr(…))` works). `at_constant_start` recognizes these op
    words so a global initializer that *is* a const-expr isn't skipped. Two parity tests: `&arr[3]` (a
    `getelementptr` global init) and `(long)&arr[1]` (`ptrtoint` of a `getelementptr`). **NB:** a
    function-pointer table (`[N x ptr] [ptr @fa, …]`) is rejected by the *translator itself*
    (`Unsupported("constexpr reference to @fa")`) on **both** paths — a translator gap, not a parser one,
    so no parity test covers it. **Twenty-eight parity tests total.**
  - **Parser — C++ exception handling landed.** `invoke [%r =] … to label %ok unwind label %lpad` is
    parsed in `terminator()` (which now takes `next_unnamed` and handles the value-producing terminator's
    `%r =` prefix; `at_terminator` looks past it). The `call`/`invoke` callee+arg grammar is factored into
    a shared `call_signature`. Added the `resume <ty> <v>` terminator, `landingpad <ty> [cleanup]
    [catch|filter <ty> <c>]…` (clauses → opaque `LandingPadClause {}` markers + the `cleanup` flag, as the
    shim maps them), and `extractvalue`/`insertvalue <ty> <agg>, … , <idx>…`. The function `personality`
    clause is skipped (already handled by the pre-`{` attr skip). The harness gained `assert_ll_parity_cpp`
    (`clang++ -O1 -fexceptions`). One parity test: a `try/catch` (invoke + landingpad + extractvalue +
    `__cxa_*`) — **NB** the translator only reserves the EH region when the module has a `main`
    (`need_eh = uses_eh && has_main`), so the test source includes one. **Twenty-nine parity tests total.**
    The instruction set is now broadly saturated for `clang`-emitted C/C++.
  - **Parser — atomics landed.** `atomicrmw [volatile] <op> ptr <a>, <ty> <v> [syncscope("…")]
    <ordering>` (all 15 `RMWBinOp`s), `cmpxchg [weak] [volatile] ptr <a>, <ty> <exp>, <ty> <new>
    <success> <failure>`, `fence <ordering>` (result-less, routed like `store`), and the `load atomic`/
    `store atomic` variants (an `atomic` keyword + trailing `[syncscope] <ordering>` before `, align`).
    Shared `atomicity()`/`mem_ordering()`/`rmw_op()` helpers; no-`syncscope` ⇒ system scope. One parity
    test: `atomic_fetch_add` + `atomic_compare_exchange_strong` + `atomic_load` — **NB** `fence` parses
    but the *translator* doesn't lower it (`Unsupported`), so it's excluded from the test. **Thirty parity
    tests total.**
- **PR2 landed + merged (PR #158).** The instruction set is saturated for `clang -O2`/`clang++` (30
  parity tests). Loose ends remaining, each a small `clang -O2`-shape + `assert_ll_parity` addition:
  **literal aggregate constants** `{ i32 1, i8 2 }`, **`indirectbr` + `blockaddress`** (AST already has
  `Constant::BlockAddress`), **scalable-vector** ops — all rare in `-O2` output.
- **PR3 (in progress, branch `claude/ll-debug-metadata`): debug metadata from `.ll` text, replacing the
  `di.rs` `llvm-sys` walk.**
  - **Slice 3a — source-line half + function names — DONE.** A metadata pre-pass
    (`Parser::collect_di_metadata`, run before `module()` since the `!N` table is emitted *after* the
    functions that `!dbg`-reference it) parses the reduced `DiNode` table (`!DILocation`/`!DIFile`/scope
    nodes via a generic `parse_di_node`/`scan_node_fields` field-scanner). `skip_trailing_metadata`
    captures each instruction's `!dbg !N` into `pending_dbg`; `instruction()` resolves it
    (`DILocation` → `scope.file` → `DIFile`) and attaches a `DebugLoc` via the new
    `Instruction::set_debug_loc`. `!DISubprogram(name:)` feeds a `@linkage → source` **func-names** table,
    exposed by `parse_module_with_debug`; `translate_ll_str` packages it as a `di::LlvmDebug` (the same
    `di` arg the bitcode path uses). Parity gate: **`assert_ll_parity_debug`** (`clang -O2
    -gline-tables-only` — source lines + subprogram names but *no* `DILocalVariable`/`DIType`, so the
    variable/type graph stays out of scope) — **31 parity tests total.** Key gotcha found: the DIFile
    filename is embedded, so both compiles must use the *same* `.c` source path.
  - **Slice 3b — the variable/type graph — DONE.** A new `ll/debug.rs` fully mirrors `di.rs`,
    building the `di::LlvmDebug` `types`/`vars`/`globals` from text so both readers produce a
    byte-identical structured half: the **type interner** (`DIBasicType`→`Base` with the same
    `infer_encoding(name)` heuristic, `DIDerivedType`ptr→`Pointer`, `DICompositeType` array→`Array`
    with `count = size_bits/elem_bits`, struct/union→`Aggregate`; transparent typedef/const resolution;
    cycle-safe placeholder; **functions walked before globals** so the interning order + `TypeId`s
    match), **`read_globals`** (each global's `!dbg` `DIGlobalVariableExpression`→`DIGlobalVariable`),
    **`read_function_vars`** (alloca ordinals for `dbg.declare`→`Window`, argument index for
    `dbg.value`→`Arg`; dedup by `!DILocalVariable` identity), and **lexical-block scope** (a
    `DILexicalBlock`-scoped var → `(decl_line, block_end)` via `compute_block_ends`). The metadata
    pre-pass was generalized to a generic `DiNode { Node { kind, fields } | Tuple }`; `dbg.declare`/
    `dbg.value` `metadata`-operand payloads (dropped by the AST as `MetadataOperand`) are captured in a
    side-channel (`DbgIntrinsic`). Gates: `assert_ll_parity_debug` (`-O0 -g`, with a `debug.type`
    non-triviality guard) + an `-Og -g` variant — five debug parity tests (source lines, func names,
    type graph + globals, `dbg.declare` locals, `dbg.value` args, lexical scope). **`di.rs` is now
    fully replaceable.** (Corrected a real bug mid-slice: the debug harness had used
    `-gline-tables-only`, which emits no type/var graph, so the type/global/local tests were passing
    trivially — switched to `-O0 -g`.)
- **PR4 (in progress, branch `claude/ll-drop-libllvm`): flip the default + drop the libLLVM binding.**
  - **Experiment (done, then reverted): route `translate_bc_path` through `llvm-dis` + the textual
    reader and run the *whole corpus* (267 tests, not just the 35 parity shapes).** Result: the `.ll`
    reader handled **252/267** after the parser-widening below; the mechanism works (out-of-process
    `llvm-dis foo.bc -o -` → `translate_ll_str`, no libLLVM linked). The reverted experiment is the
    template for the real flip. **Remaining 15 gaps** (the flip's to-do list): **10** a top-level
    construct with `=` near the module head (`at: 17`; hits `cpp_eh_*` + `rust_*` — a comdat/alias/ifunc
    or similar `skip_toplevel_item` doesn't handle; pin it down by dumping tokens near the failure);
    **4** `blockaddress(@f, %bb)` + `indirectbr` (the `computed_goto_*` tests — needs the `blockaddr`
    side-reader's payload recovered from text: parse the `blockaddress` constant + thread the
    `BlockAddrs` the translator reads); **1** `i128_wide_constant_fails_closed` (now *obsolete* — the
    textual reader carries full-width `i128`, so this fail-closed test's premise is gone; update it to
    expect success, the whole point of I14). The **forward-reference** globals/functions gap (a `@f`
    used before its def / an external `declare`) was papered over with an opaque-ptr fallback in the
    experiment; the correct fix is a **two-pass symbol collection** (pre-scan `define`/`declare`/`@g =`
    signatures into `symbols` before parsing bodies) — do that in the flip.
  - **DONE this checkpoint — parser widened for real-world (Rust/C++/`llvm-dis`) IR** (all clean, green,
    landable without the flip): the lexer drops `^N = …` ThinLTO **module-summary** lines; `type_()`
    handles `half`/`bfloat`/`fp128`/`x86_fp80`/`ppc_fp128`; `constant()` parses **literal aggregate
    constants** `{ <ty> <c>, … }` / packed `<{…}>` (`ll_parity_struct_constant_global`);
    `skip_pre_signature_attrs` + `param_list` handle **paren-payload attributes**
    (`dereferenceable(N)`/`range(…)`/`byval(<ty>)`, depth-tracked) + the newer attrs
    (`dead_on_unwind`/`writable`/`captures`/`nofpclass`/…); and `block_label` accepts a **quoted**
    `"name":` label. (These knocked the corpus from 215→252 in the experiment.)
  - **DONE — the flip landed; libLLVM is unlinked.** `translate_bc_path` now `llvm-dis`-disassembles
    to `.ll` and routes through the textual reader; **the entire 268-test behavioral corpus passes
    through the pure textual reader** (not just the 35 parity shapes). Closing the gaps required:
    **(a)** a **two-pass symbol collection** (a forward-ref-tolerant pass harvests every global/function/
    `declare` type into `symbols` before the real parse, so a `@name` used before its definition
    resolves) + parsing `declare` signatures; **(b)** the **external-global fix** — an `external
    global/constant <ty>` has *no* initializer, so `at_constant_start` must not mistake the *next*
    top-level `@name` for its value (this was the "top-level `=`" cluster, 10 tests); **(c)**
    **`blockaddress` + `indirectbr` from text** — the `(@f, %bb)` payload the AST drops is captured
    per-global (initializer DFS) and per-φ `(func, block, phi_ord, incoming)`, resolved to block indices,
    and threaded as the `BlockAddrs` the translator reads (`ll::parse::take_block_addrs`); **(d)**
    **global `alias`es** (`@a = alias <fnty>, ptr @b` → `module.global_aliases`, `type_maybe_fn` for the
    alias's function type); **(e)** the obsolete `i128_wide_constant_fails_closed` test rewritten to
    expect success (I14 is *fixed* — the reader carries full-width `i128`; NB the runtime correctness of
    `i128 urem` by a >64-bit *divisor* is a separate pre-existing translator gap, never exercised before
    because the reader fail-closed there). **Deleted:** `from_llvm_ir.rs`, `wideint.rs`, and the
    `llvm-sys` bodies of `di.rs`/`blockaddr.rs` (their data structs stay, filled by `ll::debug`/
    `ll::parse`); **dropped the `llvm-ir`/`llvm-sys`/`either` deps.** `svm-llvm` links **no libLLVM**.
  - **DONE — version tolerance proven; the rustc-1.81 pin dropped from the breadth lane.** The Rust
    on-ramp breadth lane (`rust_no_std_matches_native` and the `no_std`+`alloc` powerbox tests) now
    compiles with the **default** `rustc` (1.94 → **LLVM 21**) via `--emit=llvm-ir` straight to `.ll`
    and routes through `translate_ll_path` — so the lane *is* the version-tolerance proof: a newer LLVM
    than the old `llvm-ir` ceiling (19) flows through the textual reader unchanged. Its `+1.81.0`
    invocations (both `compile_rust_to_ll` and the native oracles) are gone. (The matching CI change —
    swap `llvm-18-dev` → base `llvm-18`, rescope the 1.81 step to peval only — is a **manual
    follow-up**: the bot's token lacks `workflow` scope, so it cannot push `.github/workflows/ci.yml`.
    The exact edit is spelled out in Q5 below.) **The 1.81 pin survives in exactly one place:**
    the multi-crate `peval_*` Futamura probe (`tests/common/mod.rs`), which links the fixture closure
    with `llvm-link-18` and DCEs it with `opt-18` — LLVM-18 CLI tools that can only ingest LLVM-18 IR,
    so the fixture must be built by a version-matched `rustc`. (Those tests auto-skip, not fail, absent
    the toolchain; CI keeps a scoped `rustup toolchain install 1.81.0` step for them.)
    Feeding real LLVM-21 IR surfaced three new-in-LLVM-19/20/21 constructs the reader/translator now
    handle: **(a)** the no-op alloc-shim marker `@…__rust_no_alloc_shim_is_unstable_v2()` (dropped like
    `llvm.lifetime`); **(b)** the `icmp samesign` poison hint (skipped in the parser); **(c)** the
    three-way-compare intrinsics `llvm.scmp`/`llvm.ucmp` (LLVM ≥ 19, what newer `rustc` lowers
    `Ord::cmp` into) — lowered to `(a>b) - (a<b)`, **masked to the result width** so the {-1,0,1} value
    lands in the `switch i8` `br_table` the way the parser's width-masked case constants expect (an
    unmasked sign-extended `-1` hit the `unreachable` default — a `BTreeMap` navigation trap).
  - **Remaining (optional):** convert the C test helpers from `clang -c` + `llvm-dis` to a direct
    `clang -emit-llvm -S` (dropping even the `llvm-dis` runtime dependency for the C-compiled tests —
    the pre-built `.bc` corpora still use `llvm-dis`).

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
runtime-path). The opt-in lane runs `cd crates/svm-llvm && cargo test`. **Build prerequisites —
UPDATED after the textual-reader flip (PR4):**
- **No `-dev` package, no libLLVM link.** The on-ramp reads textual `.ll` with an in-house parser,
  so `llvm-sys`/`llvm-ir` and the `prefer-dynamic` dance are gone. CI installs just base
  **`llvm-18`** + **`clang-18`** — two ordinary build-time *tools*: `clang` (compile the C/C++
  corpus to bitcode) and `llvm-dis` (disassemble it to `.ll`). Neither is linked into the crate.
- **No Rust-toolchain pin.** The Rust lane compiles with the default stable `rustc` (`--emit=llvm-ir`);
  its bundled LLVM (21 here) flows straight through the textual reader — the version-tolerance proof.

**Q5 — CI yaml (DONE):** no job installs `llvm-18-dev` anymore — nothing links libLLVM. The
Linux-only `svm-llvm` job installs base `llvm-18`/`clang-18` (build tools only — `clang` + `llvm-dis`)
and keeps a `rustup toolchain install 1.81.0` step scoped to the multi-crate `peval_*` Futamura probe
alone (its `llvm-link-18`/`opt-18` can only ingest LLVM-18 IR, so that fixture needs a version-matched
`rustc`; the Rust *breadth* lane runs on default stable — the version-tolerance proof). The three
sibling jobs that run `svm-llvm` tests/examples (`asan-jit-setjmp`, `embench-differential`,
`cross-engine-differential`) got the same `llvm-18-dev` → `llvm-18` swap. All landed by maintainer
pushes (the bot token lacks `workflow` scope): the `svm-llvm` lane verified green on CI run 1193 with
the peval probe running (not skipping); embench + cross-engine verified green on the run for
`31bcfcd` (asan-jit-setjmp is schedule-gated, exercised on the nightly).

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

