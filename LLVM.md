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
- **transcendentals/libm**: prefer a **guest** `libm` (the demo or a bundled header supplies
  `sqrt`/`sin`/… as guest code) over any host math capability — keeps math in the sandbox. `sqrt`
  already lowers to the SVM op (slice F); `sin`/`cos`/`exp`/`pow` need guest implementations.
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
  stays fail-closed, needing exact-decimal/bignum formatting.)**

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
- **`setjmp`/`longjmp` — DONE on the interpreter (slice AX); C++ EH next on the same substrate.** The
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
  — it only restores the `(module, cur, base, pc)` cursor + pops the activation stack). The **JIT** bails
  `Unsupported` (the remaining sub-slice — **Option B: call libc `_setjmp`/`_longjmp` from JITted code**,
  `jmp_buf` in a host-side table keyed by the guest buffer address so no host SP/return-addr leaks; the
  in-frame `_setjmp` call dodges the "helper-frame-is-gone" problem, and clang's `-O2` returns-twice
  spills + Cranelift's caller-saved spills largely defuse the optimizer hazard — to be proven by a native
  differential **under ASan**). Until it lands the JIT declines and the interpreters cover it. Tests:
  `setjmp_longjmp_round_trip`, `setjmp_longjmp_loop_and_deep_nesting` (byte-identical to native libc on
  **both** the tree-walker and the bytecode engine, multi-frame unwind + retry loop; the JIT-declines is
  asserted). Perf (`examples/onramp_perf.rs`): the non-`setjmp` baseline rows are unmoved (gated;
  byte-identical IR), `setjmp` capture/`longjmp` unwind cost is O(live-values)/O(frames) and paid only
  when executed (the JIT path will be real-`setjmp`-class O(1)). **EH next** (`invoke`/`landingpad`/
  `resume` + cleanups) reuses this stack-transfer core; the JIT `longjmp` + EH unblock Postgres
  `--single` and throwing C++ respectively.
- **`%f` / `%g` / `%e` float formatting** — deliberately **fail-closed** (LLVM.md printf notes): an
  `f64`-arithmetic approximation diverges from glibc at the rounding boundary; matching byte-for-byte
  needs a **correctly-rounded exact-decimal (Ryū / Dragon4, big-integer)** formatter. SQLite,
  `printf`-heavy programs, and any numeric output force this.
- **`qsort` + comparator function pointers** — the libc callback ABI (an indirect call from synthesized
  libc into guest code); confirm it lands cleanly when a target needs it.
- **`__int128` / `long double` (`x86_fp80`/`fp128`)** — rejected (§7 deferred). SQLite has a few `i128`
  paths; add on demand.
- **Larger libc surface** — `memmem`/`strtod`/`strtol`/`snprintf` family, `qsort_r`, etc. The on-ramp
  synthesizes a growing subset; each real program extends it (or the program brings its own, the model).

### SQLite — the north star (in-memory, then disk via the powerbox)
SQLite is the gold-standard target (≈600:1 test-to-code ratio, ships as one amalgamation `.c`). Two
phases, both worth doing:
- **Phase A — in-memory (`:memory:` / `memdb` VFS).** Build `SQLITE_THREADSAFE=0`,
  `SQLITE_OMIT_LOAD_EXTENSION`, run its SQL logic scripts; the differential is native-SQLite vs
  on-ramp-SQLite over the same script. **No filesystem needed** — SQLite's built-in in-memory VFS
  keeps the whole DB in `malloc`'d pages. This phase is gated by *computed goto* (VDBE) + *float
  formatting* above, not by any I/O capability.
- **Phase B — disk-backed (real persistence through a powerbox capability).** In-memory proves the SQL
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
- **The "wow" milestone (later):** **Doom** (shareware) — fixed-point-heavy real app; needs a stubbed
  framebuffer + an in-memory WAD, but "Doom runs sandboxed through the LLVM on-ramp" is a strong
  external signal. (Graphical apps proper ride the **WebGPU capability** — its own section below.)
- **Other-language runtimes** (the breadth thesis, building on the C++/Rust slices AG–AM): a real Rust
  crate (`regex`/`ryu`/`serde_json` `no_std`), a Zig program, a Swift `-enable-experimental-feature
  Embedded` TU — each is "another frontend, no translator change beyond what the corpus proved."

### Suggested ladder (cheap momentum → the capstones)
1. **Monocypher KAT** + **stb_image decode** — zero-OS, byte-exact; widen the corpus and shake out
   64-bit/parser bugs fast. (A SAT solver / `perft` slot in here too — pure compute, self-validating.)
2. **`indirectbr` / `blockaddress` (computed goto)** — DONE (slices AV + AW: global tables + φ-threaded
   operand-position). Robust for real interpreters.
3. **`setjmp`/`longjmp`** — DONE on the interpreter (slice AX); JIT `longjmp` (native-stack switch) +
   **EH** on top (reuses the stack-transfer core) next.
4. **`%f` / `%g` float formatting** (Ryū/Dragon4) — needed by SQLite/Postgres and broadly.
5. **SQLite Phase A (in-memory)** — the SQL-engine capstone (gated on goto + float-format).
6. **The storage capability** → **SQLite Phase B** (read/write VFS) and **libmdbx** (file-backed mmap) —
   two distinct storage shapes proving real durable I/O under zero ambient authority.
7. **Postgres `--single`** — the setjmp + File-capability capstone ("SQLite Phase B at 100×").
8. **The WebGPU capability** (its own section below) and **the network/egress capability** (curl/git) —
   the remaining capability frontiers.

### `setjmp`/`longjmp` + EH — design & sequencing (the stack-transfer substrate)

**Status: `setjmp`/`longjmp` DONE on the interpreter (slice AX)** — see the gap bullet above for the
landed design + engine matrix. The JIT `longjmp` (native-stack switch) and **EH** are the remaining
sub-slices. The design notes below are retained for the EH follow-on + the JIT path. Drivers:
**Postgres** `--single` (its whole `PG_TRY`/`ereport` model is `sigsetjmp`/`siglongjmp`) and **Lua**.

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

#### JIT `longjmp` — the remaining sub-slice (HANDOFF — design locked, not yet implemented)

Cranelift code runs on the real native control stack, so `longjmp` must restore the native SP to the
`setjmp` point. **Chosen approach — Option B: call libc `_setjmp`/`_longjmp` from JITted code, with the
`jmp_buf` in a host-side table.** Rationale + the rejected alternatives are below; this is the concrete
work for the next session.

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
run under ASan** (this is memory-unsafe native-stack manipulation; it gets `svm-fiber`-grade rigor — the
repo already has an "ASan (svm-fiber switches)" CI lane to extend). Add returns-twice stress cases
(values live across the `setjmp`; nested/loop `longjmp`; `longjmp` across several JIT frames).

**Reference implementations to mirror (already landed, byte-identical to native):** the tree-walker
(`crates/svm-interp/src/lib.rs` — `SetJmpPoint`, `setjmp_points`, the `SetJmp`/`LongJmp` arms in
`run_inner`) and the bytecode engine (`crates/svm-interp/src/bytecode.rs` — `ByteSetJmp`, `Op::SetJmp`/
`LongJmp`). The on-ramp lowering is `lower_setjmp_call` in `crates/svm-llvm/src/lib.rs`. **Current state:
both interpreters run `setjmp`/`longjmp`; the JIT cleanly declines `Unsupported`** — so the branch is a
clean checkpoint to start the JIT pass from, engines in sync, never divergent.

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
   (prefix-sum / matmul / N-body) over a guest-uploaded buffer, read back and checked against a CPU
   reference. Proves the whole data plane (upload → dispatch → readback) + the validated-buffer model,
   **with zero windowing**.
2. **Offscreen render-to-texture → read pixels → hash** — "hello triangle" / a rotating cube to an
   offscreen attachment, pixels compared to a reference image. Proves the render pipeline headlessly.
3. **Compute image filter / Mandelbrot / a Shadertoy-style shader → PNG** (written via the File cap) —
   visually compelling, deterministic, self-validating by pixel hash; a great screenshot artifact.
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
5. **Stragglers** — `%f`/`%g` exact-decimal formatting (textual IR output), a few inline-asm spots
   (mostly avoidable), possibly computed-goto in clang's lexer (the `indirectbr` work above).

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

