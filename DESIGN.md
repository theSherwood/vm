# Sandbox VM — Design Notes

*A WebAssembly alternative: secure, simple, flexible, fast, with real virtual memory.*
*Codename: TBD. Status: living document — expect churn.*

## How to read this doc

Each section is tagged with its current status:

- **[SETTLED]** — agreed in discussion; change only with a reason.
- **[OPEN]** — actively debated; alternatives and open questions listed.
- **[PARKED]** — direction agreed, details deferred.

The Decision Log at the end records *why* settled things were settled, so we
don't relitigate by accident.

---

## 1. Goals & non-goals

**Goals.** A compilation target and sandbox VM that is: secure (capabilities
exposed by the host are the only channel out; escape is impossible), simple
(small, cheaply verifiable core), flexible (a sane target for many source
languages), very fast (near-native, minimal hot-path overhead), and equipped
with real virtual memory.

**Non-goals.** We do **not** try to stop a guest from corrupting *itself*. We
only require that escape is impossible and that self-corruption is, where
practical, *detectable* so the host can kill the guest. Perfect intra-domain
memory safety is a nice-to-have (see §3, §10), not a requirement.

### 1a. Goals — restated as an achievable bar  [SETTLED]

The aspirational framing above ("escape is impossible", "near-native", "very
fast") is the *direction*. The **committed, measurable target** is relative to
WebAssembly, because that bar is concrete and reachable by this team:

> **As secure (for the host) as wasm, faster than wasm on the axes that matter,
> with a simpler and more flexible interface.**

This relativization matters because the absolute claim is not certifiable in our
configuration (non-expert + agent; see §18). "As secure as wasm" *is* certifiable
as a target: it means matching **Wasmtime's** posture, not proving a theorem.

**Security: achievable — and Cranelift is the mechanism, not a liability.**
"As secure as wasm for the host" = "as secure as Wasmtime", whose realized TCB is
*validator + Cranelift codegen + memory-confinement lowering*, with Spectre an
explicit non-guarantee in-process (distrust ⇒ separate process — exactly §2). By
choosing Cranelift (§18) we **share Wasmtime's single most security-critical
component** — the codegen where escape bugs actually live. The TCB *delta* we own
is therefore small and bounded: our verifier (tiny, fuzzable) + our CLIF-with-
masking lowering + our memory/capability plumbing. Caveat: Wasmtime's security is
also years of continuous fuzzing + expert CVE response, so design-equivalence is
not practice-equivalence until the §18 validation plan is executed. The hinge:
**isolate confinement masking into one tiny, separately-fuzzable lowering pass**
with a crisp invariant (every access dominated by a mask of the *final effective
address* into `[0, size)`), differential-tested against the interpreter and
native — see §4, §18.

**Speed: achievable only if we say *where*.** Sharing Cranelift means
**steady-state compute-bound code runs at ≈ Wasmtime speed by construction** — we
cannot out-run the same backend on a tight inner loop, and should not pretend to.
The compute target is therefore **parity with Wasmtime**; the entire speed budget
is spent *around* compute, where wasm is weak:
- **Host-call / I/O-bound:** faster, often substantially — no component-model
  lift/lower marshalling; zero-copy borrow buffers read in place via the page
  table (§7), so a guest region can be handed straight to a device/GPU with no
  copy-out (vs browser wasm's mandatory linear-memory→JS hop); trampolines
  inlinable to ~free; batched async rings (§9, §13).
  *This is the strongest, most defensible win.*
- **64-bit address space:** faster than wasm64 (one AND mask vs explicit bounds
  check) — confirmed across all memory kernels. Against wasm32 confinement is
  **"guard-when-bounded, mask-when-not"**: where the effective address is *provably
  bounded* (the common indexed-array case — `(i & K)*W`), the mask elides and we
  approach wasm32's free guarded access (bench: ~1.2–1.4× wasm32, well under
  wasm64). Where the base is an **unbounded value** — notably the threaded data-SP
  in C-frontend locals (`sp + (i & 255)*8`) — we **mask** (bench `locals_c`: ~2.26×
  wasm32, still under wasm64). Closing that last gap would need wasm32-style
  **32-bit window addressing** (so the address is `< 2^32` by construction); we
  **accept the mask cost instead, keeping the clean 64-bit model** (D50). So the
  honest claim is: **beats wasm64 everywhere; matches wasm32 on bounded offsets and
  pays ~1 mask on unbounded-base accesses.**
- **Startup / JIT latency:** faster — SSA on the wire (no SSA reconstruction);
  decls-before-bodies ⇒ parallel per-function verify+JIT (§3a).
- **Irregular control flow:** marginally faster — native irreducible CFG avoids
  relooper-introduced blocks/branches (§3).

So "faster than wasm" is **interface + 64-bit-memory + startup + control-flow
shape**, *not* raw compute. If raw-compute wins were ever required, the design has
no mechanism for them without changing the backend (losing the security
inheritance) or adding an unsafe mask-elided tier — neither is on the table.

**Interface: the clearest win.** Simpler than WASI + component model + WIT +
lift/lower (ours is scalars + `(ptr,len)` own/borrow buffers + handles, no IDL,
structured data = pure bytes — §7). More flexible than *shipping* wasm: native
irreducible control flow (wasm cannot express it), first-class stack switching as
one primitive, tail calls, multi-return, and an open capability surface vs a fixed
WASI menu (§6).

**Capability differentiators (beyond speed & interface).** The wins above are speed
and interface; the architecture also has things wasm *structurally* lacks, each
already specified in its own section and collected here so the "why not just ship
wasm?" answer lives in one place:
- **Guest-visible, flexible virtual memory** — the guest holds an attenuable
  `AddressSpace` capability (`map`/`unmap`/`protect` within its window, §4/§14),
  not just `memory.grow` on one linear blob: sparse address spaces, lazy/demand
  page supply, and lending sub-ranges out. Large or sparse programs that fight
  wasm's flat linear memory are the target.
- **Nested sandboxes (VM-in-VM) + composition (VM-beside-VM)** — a guest can use an
  `Instantiator` capability to spawn a child domain in a power-of-two **sub-window**
  with an **attenuated** subset of its own capabilities (§13/§14); confinement
  composes to any depth at depth-independent per-access cost. wasm has no native
  *runtime* nesting (only interpreter-in-wasm or link-time component composition),
  so multi-tenant hosts and plugin-in-plugin fall out for free.
- **Lean by exclusion** — no GC, no JS interop, no UTF-16 / `externref` /
  component-IDL surface. This deliberately narrows the market to systems/native
  languages (C/C++/Rust/Zig/Swift) — managed languages are *not* first-class — in
  exchange for a small verifier and ABI, which is the actual product.

Not a differentiator: the *code-generation* TCB. Sharing Cranelift means a codegen
miscompile is an escape exactly as in Wasmtime (the compute-parity point above), so
the small, auditable surface is the **verifier + interface**, not the backend;
shrinking codegen trust would need output verification (Veriwasm-style), an unbuilt
post-MVP aspiration wasm shares.

**Net:** the design supports the restated goals, contingent on three commitments —
(1) state speed as around-compute + compute-parity; (2) confinement is
guard-when-bounded / mask-when-not, masking the *final effective address*;
(3) confinement masking is one isolated, fuzzed unit. All three are folded in
below (§4, §18) and logged (D36–D38).

---

## 2. Threat model & isolation stance  [SETTLED]

- Guests are **hostile**.
- The host must isolate distrusting guests from each other, and must be able to
  *permit* shared memory + multithreading between *cooperating* modules.
- **Spectre is in scope.**

**Accepted compromise.** In-process isolation (masking, MPK, barriers) is
defense-in-depth, **not** a hard Spectre boundary. The robust boundary between
distrusting parties is a separate process. Residual covert-channel leakage is
*managed*, not eliminated. This was a deliberate, accepted tradeoff — see §9.

Chosen isolation tiers: **0** (same address space), **1** (same process /
hardware protection keys) for *cooperating* modules, and **3** (separate
process) as the *distrust* boundary. Tiers 2 and 4 from the original menu are
available but not the default.

---

## 2a. TCB & escape-freedom contract  [SETTLED]

The security spine. Makes the §3b rule-7 line ("verified ⇒ escape impossible")
precise, names exactly what is trusted, and decomposes the escape-freedom claim
into invariants each with an owner and a validation method — so the security work
has a concrete anchor. **Level (D47): a structured-prose contract** — precise
invariants + trust assumptions + the table below, *not* a formal proof. This is the
"as secure as wasm" bar (Wasmtime has no proof either); a mechanized treatment is a
post-MVP audit item (§18), not attempted now.

### The honest contract
"Verified ⇒ escape impossible" is **false as written** — verification is one link,
and the smallest. The true statement:

> **Verified(module) ∧ Correct(JIT) ∧ Correct(runtime/memory-model) ∧
> Correct(host OS + MMU + CPU) ⟹ escape-free.**

The dominant escape-TCB component is the **JIT** (Cranelift + our masking lowering +
our CLIF generation), not the verifier. Stating this is the point: it puts the risk
where it actually lives (§18).

### What "escape-free" buys — the load-bearing theorem
Even a guest with **arbitrary write access to its entire window** (conceded — §1
self-corruption is a non-goal) **cannot escape**, because the three escape-bearing
things are out of reach: memory access is mask-confined (I1); return addresses live
on an out-of-band stack it cannot name (I2/I5); indirect control flow is
table-confined to its own verified functions (I2). Its only authority is the
capabilities it was handed (I3). *That* is what verification + the memory model +
the out-of-band stack buy — not "the verifier makes it safe."

### The TCB, in two tiers
- **Escape-TCB** (a bug breaks the sandbox for everyone): verifier; JIT incl.
  masking lowering; runtime memory-model plumbing (window setup, guard pages,
  mmap/mprotect, signal/fault handlers); handle-table + control-stack management;
  supervisor; and below us the host kernel / MMU / CPU.
- **Authority-TCB** (a bug lets *one capability* misuse/leak *its own* authority,
  but **cannot escape**): the per-capability host handlers. This split is what makes
  host-extensible capabilities (§7) safe to be open-ended — adding a capability adds
  authority-TCB, not escape-TCB, *provided handlers obey the hygiene rules below*.

### Trust boundary & adversary
- **The boundary is verified IR, not the source or frontend** (the eBPF model). A
  malicious/buggy frontend or hand-written adversarial IR is **in scope and
  handled**: if it passes the verifier, escape-freedom holds. The frontend is
  **untrusted for escape** (trusted only for program *correctness*) — so "the
  frontend resolves C UB" (§3b) is a correctness concern; **I4 totality is enforced
  by the verifier + IR semantics regardless of frontend intent.**
- **The adversary controls:** the IR (any verifier-passing module), all window
  memory (arbitrary reads/writes), the timing/sequencing of `cap.call`s, and
  concurrency/data races.
- **The adversary does *not* control:** the generated machine code, the out-of-band
  control stack, or the handle table (all host-owned).

### Invariants × owner × validation (the anchor)
Escape-freedom decomposes into five sub-invariants; their conjunction is the theorem:

| Invariant | Owner | Validated by |
|---|---|---|
| **I1 Memory confinement** — every access ∈ `[base, base+size)` | masking lowering (JIT) | the isolated masking-fuzz unit + differential vs interpreter (§18) |
| **I2 Control-flow integrity** — transfers only to verified entries/blocks/out-of-band returns; indirect calls table-confined | verifier + out-of-band stack (runtime) + table dispatch | verifier fuzzing; CFI self-tests |
| **I3 Capability integrity** — authority only via held handles; host-owned table; no opcode mints authority; fail-closed signature check | verifier (sealing-as-typing) + runtime | verifier fuzzing; forged-index tests |
| **I4 IR totality** — no UB; every op = defined value or defined trap | IR semantics (JIT + interpreter) + verifier | differential JIT-vs-interpreter |
| **I5 Stack integrity** — control stack unreachable by masking; data overflow → guard fault | runtime (placement) + JIT (stack probes) | guard-page + stack-clash tests |

### Scope
- **In scope:** architectural escape; hostile/malformed/adversarial IR; **Spectre**
  (managed, not eliminated — see below).
- **Non-goals / out of scope:** intra-domain self-corruption (§1); covert/timing
  channels (mitigated, residual leak accepted — §9); **availability / DoS (D48): a
  non-goal — bounded by metering (fuel/quota/preemption) + the kill path, contained
  not prevented** (incl. the §17 GPU coarse-preemption weak spot); hardware fault
  injection (rowhammer/voltage — below our trust line, noted not defended); the
  correctness of our own build toolchain producing the JIT (trusted).

### Microarchitectural posture (one precise statement)
I1–I5 prevent **architectural** escape. They do **not** prevent **microarchitectural**
leakage (Spectre, covert channels); that is *mitigated, not eliminated* —
mask-not-branch (I1 doubles as Spectre-v1), retpoline/eIBRS on indirect dispatch,
IBPB/BHB + VERW + L1D flush on distrust-domain switch, no SMT across distrusting
domains (§9). **The robust distrust boundary is a separate process** (§2). In-process
isolation (tiers 0/1) is defense-in-depth, never a hard Spectre boundary.

### Fail-closed
Verification rejects on any error; every runtime check (bounds / type / generation /
guard) traps to §5 detect-and-kill. The system only ever fails toward
"reject/kill the guest," never toward "let it through."

### Handler hygiene (authority-TCB rules)
A capability handler must: treat borrowed guest buffers as **hostile and volatile**
— validate-on-use, copy if stability is needed, **no TOCTOU** (a concurrent guest
thread may mutate a borrowed buffer mid-call, §12/§13); exercise **only the caller's
authority**; and (fast in-process runtime) stay **secret-less** (§9). These rules
are what keep an authority-TCB bug from becoming an escape.

### Residual risk (honest)
Per the contract, escape-freedom rests on JIT/runtime correctness, which **this team
cannot certify** (§18). The mitigations are the differential interpreter-oracle, the
isolated masking-fuzz unit, and verifier fuzzing — strong bug-finders, not proofs.
**Closing the gap to "certified" is the separate post-MVP workstream** (expert
review, fuzzing infra, audit), tracked as open-ended in §18.

---

## 3. Execution model & verification  [SETTLED]

- Typed **SSA** over a **CFG of basic blocks**, with **explicit typed block
  parameters** (no phi nodes).
- **Key discipline:** values never cross block boundaries except as block
  parameters. This removes dominance analysis entirely — verification is a
  single linear forward type-check pass (check operand types; check branch
  arguments match the destination block's declared parameter types). Keeps the
  verifier — which is security-critical TCB — simple.
- **Irreducible control flow is supported natively.** No relooper, no
  structured-control-flow straitjacket. This is a direct target for LLVM-style
  producers and feeds register allocation directly in the JIT.

---

## 3a. Type system, value model & binary encoding  [SETTLED]

Continuation of §3 — the actual IR, its types, and how it serializes. Verification
(security-critical TCB) is the design driver throughout.

### Two-class value model
**Plain data is forgeable because it is confined; capability *indices* are
forgeable too, but inert because authority lives in the per-domain table, not the
index.** The split is defined by what a forged value can do:
- **Plain data** — scalars (`i32`/`i64`/`f32`/`f64`/`v128`) and **pointers**. A
  forged pointer is harmless: every access is masked + MMU-confined to the window
  (§4), so the worst case is the guest corrupting its *own* memory (allowed). So
  plain data is **freely forgeable and interconvertible** (int↔ptr↔float-bits,
  pointer arithmetic, tagging) — non-negotiable, because C/Rust do exactly this.
  `i8`/`i16` exist only as memory access *widths*, not SSA value types (tiny
  lattice → trivial verifier) — the wasm tradeoff; its compromises (frontend
  truncation burden, no narrow-width atomics) + the recommendation are in §3b
  "Narrow integer types".
- **Capabilities** — **handles** and **function references**, a **typed-index**
  class: `handle<Interface>`, `funcref<Sig>`. **The unforgeability is *positional*,
  not value-level.** A handle/funcref value is a plain **index into a per-domain
  table** (the handle table; the function table), so it may live in a register or
  spill to guest memory like any integer — and **forging the bit-pattern is inert
  (D37):** an out-of-range index traps, and an in-range one can only re-select a
  capability *this domain was already granted*, re-checked against its declared
  type at the call. Authority binds to the **table entry**, not the bearer index
  (§7). You populate the table only by grant, attenuation, or capability-call
  result — there is no opcode that *adds an entry* from plain data — but the
  *index* itself is ordinary forgeable data. The `handle<I>`/`funcref<Sig>` type is
  what the verifier tracks so `cap.call`/`call_indirect` are statically typed; it
  is **not** a claim that the value is unforgeable. (This supersedes earlier
  "sealed class / no forge opcode" language: the security comes from the table
  holding only granted authority, so we do **not** rely on sealing the index.)
  - **C function pointers** lower to exactly this: an integer index into the
    function table, dispatched by `call_indirect` with a runtime type check.
    Forging confines to your own table (cannot reach host code). The standard wasm
    limitations follow — funcptr↔dataptr casts and funcptr arithmetic don't carry
    meaning across the boundary — and are accepted (§3b lowering notes).
  - **Index values are backend-local — a differential-oracle caveat (fiber case now
    resolved).** Because a handle/funcref/§12-fiber index is *positional*, its concrete value
    is whatever the backend's table assigns; the **interpreter and JIT may number the same
    logical entity differently** and still each be self-consistent. The known case was
    **`cont.new` fiber handles**: the interp's table held the root computation as fiber-slot 0
    (first handle 1) while the JIT runs the root off-table (first handle 0), and a *forged*
    handle masked to different slots. That was **safe** (numbering is internal; resolution is
    confined to the domain's own table) but made a fiber-handle *value* non-observable to the
    interp↔JIT differential. The D57 3b-i **run-shared fiber registry** (§23)
    unified the namespace — the interp registry, like the JIT, holds only `cont.new`-created
    fibers, so handles are `0, 1, …` on **both** backends, the fiber fuzzer lets handle values
    flow into compared output, and `jit_fibers::fiber_handle_values_match_across_backends`
    pins the absolute numbering. The 3b-ii **JIT shared table** (`fiber_rt::SharedFiberTable`)
    extended the unification to *multi-vCPU* runs: handles allocate from one domain-wide
    namespace and forged handles mask over the same table shape on both backends
    (`jit_threads::fiber_namespace_is_domain_wide`). With 3c (migratable fibers, D57) the
    last staged divergence closed too: a **foreign vCPU's resume** migrates the fiber on
    *both* backends — pinned by `jit_threads::fiber_suspended_on_root_migrates_to_spawned_vcpu`
    and the randomized-migration differential in `fiber_fuzz`.
- **Pointers** are a CHERI-ready-but-erasable refinement of `i64` (§10): off-CHERI
  a no-op; the type exists for the JIT's masking and a future CHERI *host* backend.

**Central safety theorem (what the verifier + runtime buy):** escape is impossible
because (a) the memory model confines all plain-data access (mask of the final
effective address + guard region, JIT-enforced — §4), (b) **the authority tables
contain only granted capabilities, so a forged index is inert** — it traps or
re-selects one of the domain's own grants, type-re-checked at the call (D37),
(c) control transfers are typed + the control stack is out-of-band (§5). The guest
may mangle plain data arbitrarily — including handle/funcref *indices* — and still
cannot escape: the index space is confined to the domain's own tables, and the
tables are the only thing that maps an index to authority. Note (a) is enforced by
**JIT codegen (the masking lowering), not the verifier** — the verifier secures
typing, control flow, and table-index ranges; confinement of memory is the
masking pass (§4, §18). "Verified ⇒ cannot escape" is shorthand for "verified
**and** the masking lowering is correct."

### Binary encoding
Design goal: **decode and verification fuse into one linear forward pass, no
fixups.** Three choices deliver it:
1. **Block-local value numbering.** Within a block, values index sequentially —
   block parameters `0..k-1`, then each instruction's result takes the next index.
   Operands reference *strictly earlier* same-block indices (no intra-block forward
   refs, no fixups). Cross-block dataflow is *only* via block parameters, so a
   value in block A cannot be named in block B — **dominance analysis is impossible
   to need, by construction.**
2. **Up-front block-signature table.** Each function header declares every block's
   parameter types before the instruction streams, so the single pass can check any
   branch (forward, back-edge, loop) against its target's already-known parameters.
3. **Inferred result types + typed opcodes.** `i32.add` vs `f64.mul` are distinct
   opcodes; the verifier computes result types from opcode + operand types and
   stores nothing per instruction. Types appear only where not inferable (block
   params, constants, polymorphic ops).

**Verification per function (single fused pass):** read the block-signature table;
per block, seed a local type vector with its parameter types, walk instructions
(check operand types, append result types), check each terminator's branch args
against the target's declared parameters. Linear, local, no dominance, no fixups.
**Decode and verify are fused** (one pass, one set of bounds/range/type checks) —
minimizes TCB and the window for decoded-but-unchecked state. Every length and
index is bounds-checked; nothing in the stream is trusted.

**Operand references = block-local indices** (LEB128, usually one byte), not
back-references — clearer, maps directly to the verifier's type vector.

**Module structure:** section-based (magic + version; type section incl. interface/
handle-operation signatures; imports = expected capability handles; window decl;
function bodies; data; exports). **All declarations precede all bodies**, so each
function body is independently verifiable + compilable (lazy/parallel JIT).

**Instructions:** typed opcodes; constants inline; loads/stores carry access width
+ address operand (masking is implicit semantics, JIT-inserted); C11 atomics (§12);
control terminators (branch / cond-branch / br_table / return / tail-call / trap /
stack-switch). **Multi-result instructions** allowed (multi-return calls). Capability
invocation: `cap.call (handle, op-index, args…)` → handle type + op-index resolve
the signature from the type section. Stack-switching = control opcodes over a typed
continuation value (§12).

**Size:** larger on the wire than a stack machine — the deliberate cost of being an
SSA target (no producer stackification, no consumer SSA reconstruction, trivial
verifier). Handled by a naturally-compact encoding (LEB128, block-local indices,
inferred types) + standard wire compression (zstd). **No bespoke compression
scheme** (avoid complexity/TCB).

**Text format first.** Define a CLIF/LLVM-IR-flavored text form 1:1 with the binary.
In Phase 1 (§18) chibicc emits text, a tiny assembler produces binary; the text form
is the human/agent debugging interface throughout. Disproportionately valuable for
an agent-driven build.

---

## 3b. IR specification (Phase 1)  [SETTLED]

The concrete spec the verifier + interpreter are built from. **The IR is total:
every operation produces a defined value or a defined trap — there is no undefined
behavior at the IR level.** Source-language UB (C) is resolved by the frontend into
defined IR; the verifier/JIT never reason about UB. This is load-bearing for
security — UB in a sandbox IR would void the escape guarantee.

### Instruction set (MVP)
- **Constants:** `i32.const i64.const f32.const f64.const`.
- **Integer arithmetic** (i32/i64): `add sub mul` (two's-complement **wrap**),
  `div_s div_u rem_s rem_u` (**trap** on /0; `div_s`/`rem_s` trap on INT_MIN/−1),
  `and or xor shl shr_s shr_u rotl rotr` (shift amount mod bitwidth), `clz ctz
  popcnt`.
- **Integer compare** (→ i32 0/1): `eq ne lt_s lt_u le_s le_u gt_s gt_u ge_s ge_u
  eqz`.
- **Float arithmetic** (f32/f64, IEEE 754, no traps): `add sub mul div sqrt min max
  abs neg ceil floor trunc nearest copysign`.
- **Float compare** (→ i32): `eq ne lt le gt ge`.
- **Conversions:** `i64.extend_i32_s/u`, `i32.wrap_i64`, `extend8_s/extend16_s/
  extend32_s` (the narrow sign-extends — *defined here + in `svm-ir`, the interpreter, **and the JIT**
  (`ireduce`→`sextend`); the chibicc frontend still narrows with shifts — see "Narrow integer types" below*);
  `trunc_sat_f→i_s/u` (**saturating default**, deterministic; trapping
  variant available), `convert_i→f_s/u`, `f32.demote/f64.promote`; `reinterpret`
  (i32↔f32, i64↔f64 — bit-level, for NaN-boxing).
- **Pointers:** `ptr.from_int` / `ptr.to_int` (free, no-op off-CHERI — the §10/§3a
  casts), `ptr.add` (offset by integer; lets the JIT/CHERI backend see pointer
  arithmetic).
- **Memory:** `{i32,i64,f32,f64}.load/store`; narrow `load8_s/u load16_s/u load32_s/u`
  + `store8/16/32` (C char/short). Address operand + immediate offset + alignment
  *hint* (unaligned allowed). Confinement masking is implicit (JIT-inserted).
- **Atomics** (C11, §12): atomic load/store at orderings; RMW `add sub and or xor
  exchange cmpxchg` at orderings; `fence`; `wait`/`notify` (futex). *(**Implemented**
  (Phase-4 concurrency, ahead of the original plan): in the IR, interpreter, and JIT,
  alongside the `cont.*` fiber and `thread.spawn`/`thread.join` primitives — see §12, D56.)*
- **Calls** (produce results): `call <func>`, `call_indirect <funcref>` (typed;
  static check, runtime check only for dynamic dispatch), `cap.call <handle>
  <op-index>` (handle type + op-index → signature; async/sync per the operation).
- **Select:** `select <cond> <a> <b>` (branchless, same-typed).
- **Terminators** (exactly one per block): `br <blk>(args)`, `br_if <cond>
  <then>(args) <else>(args)` (two-target, no implicit fallthrough), `br_table
  <idx>[<blk>(args)…]<default>(args)`, `return(vals)`, `return_call` /
  `return_call_indirect` (tail calls), `trap` / `unreachable`.
- **Deferred to their sections:** SIMD vector ops (§17), stack-switch terminators
  for fibers/continuations (§12 — MVP is single-fiber, so stubbed).

### Narrow integer types — the wasm tradeoff  [SETTLED for the MVP; revisit at the LLVM on-ramp]
`char`/`short`/`_Bool` are carried as **`i32` SSA values** (narrow widths exist only on
*memory* — `load8/16`, `store8/16` — and `IntTy` is `{i32, i64}`). This mirrors wasm. So a
**value-level narrowing cast must be lowered explicitly**: the frontend emits `(x<<k)>>k`
(signed), `& 0xFF/0xFFFF` (unsigned), or `x != 0` (`_Bool`) — `gen_convert`/`narrow_to` in
`codegen_ir.c`. (A *missing* truncation here was a real chibicc bug: a same-IR-width cast was a
no-op, so an rvalue `(char)200` kept `200` — only the store width truncated, which hid it behind
`char c = (char)200`. Now fixed + guarded by `c_matches_gcc_narrowing_casts`.)

**Compromises we accept vs. first-class `i8`/`i16` values:**
1. The **truncation burden lives in each frontend's lowering** — a recurring bug class. Containable
   by centralizing it (done for chibicc); the **LLVM on-ramp (D54) will need the same discipline**
   when collapsing LLVM's native `i8`/`i16` to `i32`.
2. **Narrow-width atomics are not expressible** — `IntTy = {i32, i64}`, so `_Atomic char/short`
   RMW/cmpxchg have no IR form (a guest must widen to a 4-byte atomic — which touches adjacent
   bytes — or the libc omits it; today it offers only 32/64-bit atomics). The one genuine
   *capability* gap (vs. a lowering burden); rare, since most atomics are word-sized.
3. Minor IR verbosity at narrowing points — Cranelift folds the shift pair back to a sign-extend,
   so **zero runtime cost**.

**Why not add `i8`/`i16` value types:** it **widens the escape-TCB** — the JIT (the dominant
escape-TCB component, §2a) would have to lower narrow arithmetic/compares/shifts, against the
"small, separately-fuzzable codegen surface" thesis — and proliferates ops across the whole
pipeline (text/encode/verify/interp/JIT/fuzzer), for **marginal benefit**: C integer-promotes
narrow types to `int`, so narrow *arithmetic* almost never occurs; narrowness matters only at
load/store/cast/atomic *boundaries*, which the current model already covers (except atomics).

**Recommendation (revisit only if a concrete need appears — likely the LLVM on-ramp or a
narrow-atomic workload):** keep the `i32`/`i64` model; prefer the cheaper, TCB-preserving fixes
over adding `i8`/`i16`:
- **Use the already-specified `extend8_s`/`extend16_s` ops** so narrowing is one canonical,
  fuzzable op instead of a shift pair. They exist in `svm-ir`, the interpreter, **and the JIT**
  (lowered as `ireduce`→`sextend`; they ride the 4000-seed interp↔JIT differential, pinned by
  `jit_diff::jit_matches_interp_sign_extend_ops`). The chibicc frontend still emits shifts for
  narrowing casts (a future frontend can emit the op directly). This adds *no* narrow-arithmetic
  surface to the TCB.
- For any `_Atomic char/short`, emit a **CAS loop over the enclosing aligned word** in the guest
  libc (the standard lock-free narrow-atomic trick) — zero VM/TCB change — rather than adding
  `IntTy::I8/I16`.


- Traps: integer /0 and signed-overflow div/rem; out-of-window / unmapped /
  wrong-perm access (hardware fault, §4); `trap`/`unreachable`. All traps deliver to
  the host (§5 detect-and-kill); host decides kill vs signal.
- No traps: wrapping integer arithmetic, shifts (mod bitwidth), IEEE float ops
  (produce inf/NaN), saturating float→int.
- **Memory is little-endian.** IEEE-754 binary32/64, round-to-nearest-even. NaN bit
  patterns are host-defined in the default mode (fast, matches hardware) and
  **canonicalized only in deterministic mode** (§12).

### Verifier validity rules (the TCB contract)
Single fused decode+verify forward pass, O(module size):
1. **Structural:** valid magic/version; sections length-bounded; all indices
   (type/func/block/value/handle-op) in range; nothing in the stream trusted.
2. **Per function:** entry block's params = the function signature's params; every
   block ends in exactly one terminator (terminators only at block end); the
   block-signature table covers all referenced blocks.
3. **Per instruction:** every operand value-index < current block-local count
   (defined-earlier, no forward refs); operand types match the opcode **exactly**
   (no implicit coercion); result type(s) appended per opcode.
4. **Branches:** target in range; branch-arg count + types = target block's declared
   parameter types, exactly.
5. **Calls:** arg count/types = callee signature; `call_indirect` funcref type
   matches; `cap.call` op-index in range and arg types = the operation signature.
6. **Capability typing (not sealing):** `handle<I>`/`funcref<Sig>` are tracked as
   *types* so `cap.call`/`call_indirect`/attenuation ops are statically checked, but
   the *value* is an ordinary forgeable table index (D37). The verifier does **not**
   try to prove indices unforgeable — it checks index-carrying opcodes are
   well-typed and that static indices are in range; runtime bounds- + type-checks
   the entry on use. Safety is positional: the table holds only this domain's
   granted authority, so a forged index is inert.
7. **Contract:** verification is *one* conjunct of escape-freedom, not the whole of
   it — the precise statement, the full TCB, the I1–I5 invariants, and the scope are
   in **§2a**. Short form: `Verified ∧ Correct(JIT) ∧ Correct(runtime) ∧
   Correct(host/HW) ⟹ escape-free`; soundness of the JIT/runtime is the separate,
   hard problem (§18).

### Entry & instantiation contract
- A module declares an **entry function** with a fixed signature and the
  **imports** (the capability handle types) it expects as its initial powerbox.
- **Instantiate** = verify (fail closed on any error) → allocate the domain (window
  + handle table) → bind the host-granted initial capabilities into the handle
  table in declared import order → call `entry(handle_0 … handle_n, args_buffer)`.
- Args/env arrive as a buffer (or buffers) through the initial grant.
- **C `main`:** the frontend's entry wrapper initializes the C runtime — `malloc`
  built over the `map` capability (§4), stdio over the console capability — then
  calls `main(argc, argv)`, then invokes the exit capability with the return code.

---

## 3c. Function table & handle table (the index model)  [SETTLED]

Concretizes how `funcref`/`handle` values work, resolving the earlier
sealed-vs-inert ambiguity in favor of **inert typed indices** (D37).

### Unifying model
Two per-domain, **host-owned** tables — the **function table** (code) and the
**handle table** (authority). A `funcref`/`handle` value is a **forgeable integer
index** into one of them; confinement happens **at the use site** (bounds + type
check against the host-owned table), never by sealing the value.

**No guest-visible value needs to be unforgeable.** Every escape vector is closed
at *use* or by *out-of-band storage*:

| Escape vector | Confined by |
|---|---|
| memory access | mask of final effective address (§4) — use-site |
| indirect call `call_indirect` | function-table bounds + type check — use-site |
| capability call `cap.call` | handle-table bounds + type + liveness, **host-owned table** — use-site |
| direct call / branch | static targets, verifier-checked — no runtime value |
| **return** | out-of-band control stack (§5) — not guest-addressable at all |

So the §3a two-class model is really: **forgeable plain data (everything, indices
included), confined at use, plus the out-of-band control stack as the one thing the
guest cannot name.** The linchpin for both tables is **host-ownership** — the guest
holds indices but can never write a table entry, so a forged index only ever
selects among entries the *host* installed. This formally retires §3a's "no opcode
produces a handle from plain data" language: int↔handle and int↔funcref casts are
*allowed* (C needs them — e.g. a C `int fd` that *is* a handle index); safety is the
use-site checks, not value sealing.

### Function table
**Contents:** exactly the domain's own functions (domain-global indices across all
linked modules, §13/§14, assigned at instantiation/link). There are **no imported
host functions** (§7: all host access is `cap.call`), so `call_indirect` cannot
leave guest code and carries **zero host authority** — pure intra-guest control flow.

**Representation** — flat, power-of-two-padded, **AoS** (the two fields are always
read together → one cache line; per AGENTS.md data-oriented design):
```
struct FnEntry { type_id: u32, code: *const u8 }   // host-owned, guest-unwritable
fn_table: [FnEntry; pow2]                           // indexed by function index
```
- `funcref<Sig>` value = a function index (a plain integer).
- `ref.func <funcidx>` → `funcref<Sig_funcidx>` (the index; direct, no check).
- `call <funcidx>(args)` → fully static direct call (verifier checks funcidx + types).
- `call_indirect <Sig>(fref, args)` — **always runtime-checks** (D38, wasm parity;
  JIT devirtualization is a later optimization, not MVP):
  ```
  i = fref & (len-1)                  // mask, not branch → Spectre-v1 safe table load
  trap if fn_table[i].type_id != Sig.id
  call fn_table[i].code (args)        // indirect branch → retpoline / eIBRS (§9)
  ```

**C function pointers** lower with no friction: the pointer *is* the integer index;
storing, comparing, casting to/from `void*`, and building dispatch arrays are
ordinary integer ops in guest memory — **no `table.set/get/grow` needed** (a mutable
array of function pointers is just an array of indices in guest memory). Accepted
casualties (standard wasm): function-pointer *arithmetic*, and casting a *data*
pointer into a callable funcref (a guest-internal JIT) — the latter simply traps at
`call_indirect`, which is correct for a sandbox. **Deferred:** mutable/growable
function tables and `table.*` opcodes — add only if a language demands them.

### Handle table (the powerbox)
**Representation** — flat, pow2-padded, AoS, **host-owned and outside guest-writable
memory** (same trust class as the control stack, §5):
```
struct HandleEntry {
    type_id:    u32,      // interface type
    generation: u32,      // use-after-close detection (D37); index = (generation, slot)
    methods:    *const Vtable, // per-entry dispatch table for this binding (host-owned)
    object:     *mut (),  // host-side capability state — guest NEVER writes this
}
handle_table: [HandleEntry; pow2]   // per domain, shared across its threads
```
**`cap.call <op_index>(h, args)`** — `op_index` is an immediate and the handle's
interface `I` is the static type (so the *signature* is static); the dispatch
**target is per-entry** (D45):
```
j = slot(h) & (len-1)
e = handle_table[j]
trap if e.type_id != I.id            // forged / wrong-type index → inert
trap if e.generation != gen(h)       // closed / revoked → defined trap (D37)
e.methods[op_index](e.object, args)  // dispatch through the binding's vtable
```
Consequences:
- **Dispatch is per-entry, not per-type** (D45). One interface type has many
  implementations — one per handle (the powerbox's `stdout` and a plugin's `stdout`
  are both `handle<Stream>` yet dispatch to different host code), and §14 *requires*
  this: a capability may be **pass-through** or **parent-virtualized**, and the child
  can't tell which. So the general `cap.call` is an **indirect** call through the
  entry's vtable (retpoline / eIBRS, like `call_indirect`, §9). The JIT
  **devirtualizes** it to a direct, inline-able call (§9's "inline-able to ~free")
  when it can prove the binding — e.g. a powerbox import never reassigned — exactly
  the optimization deferred for `call_indirect`. Cross-domain / slow capabilities are
  just a vtable whose entries are trampoline stubs (marshal to supervisor / ring
  submit, §9).
  - **Devirtualization is deferred, and the cost of doing it is the reason (not just
    laziness).** Today `cap.call` lowers to one fixed generic host thunk: marshal args
    into an `i64` slot array, `call_indirect` the thunk, and the **host** does the
    mask + `type_id` + `generation` resolve — the JIT does *no* authority work, so its
    role carries no authority-TCB. Devirtualizing pulls binding-resolution and
    check-elision *into* the escape/authority-TCB codegen, where a miscompile is an
    authority bug (wrong handler / elided liveness check) — the class AGENTS.md says not
    to invite without a concrete demand. It also fights the **compile ⊥ instantiate**
    split (§3a): the binding is set at *instantiation*, after the (parallel/AOT/lazy)
    JIT runs, so devirtualization must either couple codegen to one instantiation
    (losing compile-once-instantiate-many + the startup win), guard-and-deopt (eroding
    the gain), or re-patch at instantiation (complexity). Soundly skipping the
    `generation` re-check is moreover legal only for a **provably-stable, never-revoked**
    handle (powerbox imports), so the general case stays generic regardless. And it
    addresses only *half* the measured cost — the generic `i64`-array arg ABI is separate,
    and an arbitrary **Rust** handler can't be inlined into CLIF, so the realistic ceiling
    is "direct call + register args (~parity)," not free. **Measured (`bench/` hostcall):
    scalar `cap.call` is ~1.24× a wasm import today; the defensible §1a interface win —
    the zero-copy borrow buffer (`hostbuf`, ~1.8× *faster*) — needs none of this.** Revisit
    only if a real workload makes scalar host-call latency a measured bottleneck (D45).
  - **Update — D45 now implemented (opt-in), and it cleared the TCB concern above.** An optional
    `svm_jit::FastCapResolver` lets the embedder hand the JIT a *specialized* host fn for a
    statically-known `(type_id, op)`; the JIT emits a register-to-register direct call to it
    (resolved at compile time, baked), falling back to the generic thunk for any unclaimed op. The
    authority-TCB worry is sidestepped **by construction**: the specialized fns *delegate to the same
    `Host::cap_dispatch_slots`* the generic thunk uses, so the I2 `resolve` (mask + `type_id` +
    `generation`) is unchanged — devirtualization moves only the *boundary* (register args, no
    runtime `(type_id, op)` dispatch), never the authority check. The resolver is gated on arity
    (`n_args`/`n_res`) so an odd-signature `cap.call` can't C-ABI-mismatch the fn. The production
    powerbox (`svm-run`) fast-paths the window-independent hot ops (`Clock.now`, `Blocking.work`).
    **Measured: hostcall ~1.24×→ ≈0.67× (≈1.5× *faster* than a Wasmtime import)** — the register-ABI
    win was larger than the "~parity ceiling" predicted, since it also drops the generic `i64`-array
    marshalling *and* the host-side dispatch. See HANDOFF §10 (Benchmarking "D45") + `jit_diff::fast_cap`.
- A forged handle index is **inert**: it traps (wrong type / dead generation /
  OOB-masked-to-wrong-type) or selects one of *this domain's own* granted type-`I`
  capabilities. The guest never supplies `e.methods`/`e.object` (host memory), so it
  cannot aim a handler at arbitrary code or an arbitrary object — only at
  host-installed grants.

**Attenuation needs no new IR.** `subdir`, `readonly`, `Connector`-narrowing, etc.
are simply **interface operations whose result type is a handle**: the host allocates
a new, more-restricted entry in the *caller's* handle table and returns its index.
Since `cap.call` results can already be handle-typed, attenuation and the initial
powerbox (instantiation fills the first N entries, §3b) reuse the existing mechanism
— zero extra surface.

**Buffer args** (`(ptr,len)` + own/borrow, §7) are validated at the trampoline: the
ptr is a guest window offset, so the trampoline masks/bounds-checks `(ptr,len)`
against the window before the host borrows it in place (§9's "arg bounds-check").

### Verifier delta
- `ref.func f`: `f` in range → result `funcref<Sig_f>`.
- `call_indirect Sig`: operand `funcref`; args match `Sig` params; results = `Sig` results.
- `cap.call op_index`: operand `handle<I>`; `op_index < I.op_count` (static, type
  section); args/results match `I.ops[op_index]` (results may be handle/funcref-typed).
- int↔funcref and int↔handle conversions allowed (plain-data-like) — use-site checks
  carry safety.

---

## 3d. C ABI & frontend lowering (Phase 2)  [SETTLED]

How the chibicc-style frontend lowers C to the IR. Resolves the §18 "C ABI"
checklist item. Two settled decisions — the **out-of-band control stack** (§5) and
**windowed/masked memory** (§4) — *force* most of the ABI's shape; the rest is
chosen for simplicity (AGENTS.md) and wasm-parity (§1a), since the MVP is a
whole-program single module that links to no external platform ABI.

### The forced two-stack split
A pointer to an address-taken local must be a **window offset** (so access through
it is masked + confined, §4). The control stack is **out-of-band** (§5), *not* in
the window — so an address-taken local cannot live there, or its `&` would mask to
the wrong window location. Hence the SafeStack split:

| Stack | Where | Holds | Guest-addressable |
|---|---|---|---|
| **Control stack** | out-of-band (Cranelift-managed machine stack) | return addrs, callee-saved regs, SSA spills | **No** |
| **Data stack** | in the window, per-thread, guard-paged (§5) | address-taken locals, by-value aggregate copies, `alloca`, varargs buffers, `sret` slots | Yes (confined) |

The frontend manages the data stack via a **data-SP** (per-fiber state — see below)
held in `vmctx` (the context the JIT already needs for window base/mask,
handle-table base, function-table base, fuel; register-pinning the data-SP is a
lowering detail). Overflow hits the guard page → fault → §5 detect-and-kill; frames
larger than a guard page emit a stack-probe (stack-clash mitigation).

**This split *is* the fiber model.** A stackful fiber (§12) owns the **pair** of
stacks — control + data — and switching swaps both SPs; the data-SP and
callee-saved are per-fiber, while window base/mask and the table bases are
per-domain (shared, constant). The control stack lives in VA unreachable by guest
masking (CFI) but is **charged to the guest's quota** (§15) so a fiber-bomb OOMs
itself, not the host (§12). Nothing in this ABI dangles across a suspend: all
§3d data (aggregates, `alloca`, varargs, `sret`) lives on the data stack, which
travels with the fiber. The stack-switch must be modeled as a **call-clobbering**
control op (§3b/§6) so Cranelift spills live values to the control stack around it.

### Local classification (address-taken vs SSA)
One frontend pass, justified by the split above:
- **SSA-value local** — a scalar never address-taken and non-`volatile` → an SSA
  value (register / out-of-band spill). Heap overruns cannot corrupt it.
- **Data-stack local** — address-taken, any array/struct/union accessed by pointer,
  `volatile`, or address-escaping → a window data-stack slot with explicit
  `ptr.add`/load/store. Cranelift never sees it as a value.

(chibicc allocates all locals to memory first; we run the reverse — promote
non-address-taken scalars to SSA. This is the pass that matters for speed.)

### Data model & type mapping
- **LP64, little-endian** (§3b): `int`=i32; `long`=`long long`=pointer=`size_t`=8 B;
  `ptrdiff_t`=i64.
- `char` = **signed** i8 (pinned; matches x86-64 / chibicc). `_Bool`=i8 (0/1).
  `short`=i16. `i8`/`i16` are access widths only (§3a); arguments take the usual C
  integer promotions to i32.
- `float`=f32, `double`=f64, **`long double`=f64** (no 80-bit; pinned).
- `enum`=i32 unless declared wider. **Function pointers = funcref indices** (§3c),
  stored as integers in memory.

### Struct/union layout
Adopt the **standard C / x86-64-SysV layout rules** — natural alignment, tail
padding to the struct's alignment, little-endian bitfield packing. chibicc already
implements them and the whole-program MVP needs no external-ABI compatibility, so
"standard and well-understood" beats novel. `sizeof(void*)`=8.

### Calling convention (guest↔guest)
IR signatures are typed; Cranelift assigns machine registers. The C-level mapping:
- **Scalars** (int/float/pointer) → direct typed IR params (with C promotions).
- **By-value aggregates → by hidden pointer everywhere (D39).** All by-value
  struct/union args and returns pass via a caller-allocated copy in the data stack;
  returns use an `sret` hidden first pointer the callee writes through. Only scalars
  pass directly. This is the simplest correct rule and ~wasm parity (clang's wasm
  ABI is essentially this); register-classification (unwrapping small structs) is a
  deferred optimization, not MVP.
- **Varargs** → clang-wasm-style: the caller marshals variadic args
  (default-promoted) into a contiguous **data-stack buffer** and passes a pointer as
  the trailing hidden arg; `va_list` = that pointer, `va_arg` = load + bump. No
  register-save area.

### Globals / statics → data segments
- Initialized globals + string literals → module **data section** (§3a), copied to
  fixed window offsets at instantiation. `&global` = a ptr constant.
- BSS → a zeroed window region (size + offset only; the window is zero-filled).
- **Const globals + string literals → a read-only data segment (D40):** mapped RO
  via the memory capability (`protect`, §4) at instantiation; a write faults →
  §5 detect-and-kill. One extra `protect` call buys cheap self-corruption detection.
- **`_Thread_local` deferred** to when threads land (MVP is single-thread); treated
  as an ordinary global until then.

### `malloc`/`free` = guest code over `map`
Not a VM primitive. The frontend's mini-libc implements `malloc`/`free`/`calloc`/
`realloc` as **guest C** managing a window heap region, grown via the **`map`
capability** (§4), guard-page-bracketed (§5). MVP allocator = simple free-list/bump;
the shipped `<stdlib.h>` allocator now **grows the heap into the reserved-window tail via the
`map` capability on demand** (the early "fixed-size window" bump-within-a-pre-mapped-heap
simplification is superseded — see §4 / §3e).

### Phase-2 C subset (the "compilability proof" target)
- **In:** `alloca`/VLAs (data-SP bump); computed `goto` (native — irreducible CFG,
  §3); the full scalar/aggregate/vararg conventions above.
- **Deferred:** `setjmp`/`longjmp` and C++ EH → lower onto the §12 stack-switch
  primitive (stubbed in Phase 1); `_Thread_local` (with threads).
- **Out:** inline asm; 80-bit `long double`.

---

## 3e. MVP capability set  [SETTLED]

The first concrete interfaces the §3c handle table dispatches and the §3d C runtime
calls (`malloc` over `map`, stdio, `exit`). Resolves the §18 checklist item. Four
interfaces — `Stream`, `Exit`, `Clock`, `Memory` — plus the powerbox layout. (A fifth,
`SharedRegion` (§13), has since landed as a host-granted interface — *aliasing only*; its
`create`/`grant` are a §14 follow-up.)
**These four are not special:** they are ordinary instances of the general,
host-extensible capability mechanism (§7 "Host-defined capabilities &
discoverability") — a host adds new capabilities the same way the runtime provides
these.

### Shared conventions
- **Invocation:** `cap.call <handle> <op-index> args… → results` (§3c); each
  interface is a fixed numbered method table; op-index + interface type are static,
  so the handler is a compile-time-constant direct call.
- **Args (§7 calling convention):** scalars in registers; **buffers as
  `(ptr: i64, len: i64)`**, **borrow-only** in the MVP (host reads/writes in place;
  own/transfer reserved). The trampoline validates `[ptr, ptr+len) ⊆ [0, size)` —
  violation → `-EFAULT` (recoverable guest bug, not an escape; masking keeps it
  in-window regardless).
- **Error model = negative-errno (D42):** each op returns a signed `i64` — `≥ 0` is
  the success value (e.g. byte count), `< 0` is `-errno`. Syscall-shaped (§7), maps
  1:1 onto the C libc shim. Errors **do not trap**; traps stay reserved for
  escape/fatal (§3b).
- **Sync now, async later:** blocking-capable ops (`Stream.read/write`) are
  **synchronous** in the MVP (single fiber → §12 ordinary blocking). The §12
  submit/complete async form is added later **without changing the interface**.
- **§9 cost-ladder placement:** `Clock` → path 2 (vDSO-style read); `Stream` r/w →
  path 4 (direct confined syscall on a granted fd); `Memory` → path 3 (confined
  kernel syscall); `Exit` → path 6 (supervisor teardown, rare).

### Interfaces

**`Stream`** — byte stream (stdin/stdout/stderr now; files/sockets reuse it via §7
attenuation later) — **D43**:
| op | signature | semantics |
|---|---|---|
| 0 | `read(buf, len) -> i64` | bytes read `≥0` (0 = EOF) or `-errno`; borrow (host writes guest buf); blocking-capable |
| 1 | `write(buf, len) -> i64` | bytes written `≥0` or `-errno`; borrow (host reads guest buf); blocking-capable |
| 2 | `close() -> i64` | optional in MVP (exit reclaims all); included for completeness |

**`Exit`** — lifecycle:
| op | signature | semantics |
|---|---|---|
| 0 | `exit(code: i32)` | terminate the domain with `code`; **noreturn** (no results); frontend emits `unreachable` after |

**`Clock`**:
| op | signature | semantics |
|---|---|---|
| 0 | `now(clock_id: i32) -> i64` | nanoseconds; `clock_id` 0 = monotonic, 1 = realtime (Unix epoch); non-blocking |

**`Memory`** (the §14 `AddressSpace` capability, attenuable to a window sub-range;
window-relative, page-aligned offsets):
| op | signature | semantics |
|---|---|---|
| 0 | `map(offset, len, prot: i32) -> i64` | commit pages; `prot` = `READ\|WRITE` (no `EXEC` — guest data is never executed as code, §3c) |
| 1 | `unmap(offset, len) -> i64` | decommit |
| 2 | `protect(offset, len, prot: i32) -> i64` | change perms — backs the D40 read-only const segment |
| 3 | `page_size() -> i64` | host MMU page granularity (the unit `map`/`unmap`/`protect` round to); lets a guest allocator align to the real host page (§4) |

Out-of-range / misaligned → `-EINVAL`. The Phase-3 implementation went **past** the original
fixed-size / eager-mapping simplification: the window is now a *large* reserved VA range
(`DEFAULT_RESERVED_LOG2 = 40`) with **guest-controlled growth** into the reserved tail and
kernel demand paging, so `map`/`unmap` are real (the shipped `<stdlib.h>` `malloc` grows the
guest heap through them) and `protect` backs the RO-data setup.

### Powerbox (instantiation grant)
`entry(h0…h5, args_buffer)`, imports declared in this order (§3b):
```
h0: Stream  (stdin,  readable)   h3: Exit
h1: Stream  (stdout, writable)   h4: Clock
h2: Stream  (stderr, writable)   h5: Memory (the window heap region)
args_buffer: borrowed buffer at a known window offset
```
**`args_buffer` layout (pure bytes, §7):** `{ argc: u32, envc: u32 }` then
`argc + envc` NUL-terminated UTF-8 strings packed in order. The C entry wrapper
scans it to build `argv[]`/`envp[]` on the data stack, calls `main(argc, argv)`,
then `cap.call h3 exit(ret)` (§3d).

### Deferred
`File`/`Directory`/`openAt`, `Connector`/networking (§7), async submit/complete
forms, the own/transfer buffer bit, multi-fiber/TLS clocks, revocation — none block
the Phase-1 core loop.

---

## 4. Memory model  [SETTLED] (some details PARKED)

- Each domain gets a large **reserved virtual-address window** (e.g. 2^40,
  host-configurable). Guest pointers are **offsets into the window**.
- Real **demand-paged virtual memory via the host's MMU**: `map` / `unmap` /
  `protect`, file-backed mappings, guard pages, and COW are **host
  capabilities**, implemented with `mmap` / `mprotect` / `userfaultfd`.
- **No hot-path software bounds checks.** Confinement is **guard-when-bounded,
  mask-when-not** (D36):
  - *Bounded case (wasm32-style, the common one):* when the JIT can prove the
    effective address stays within a small reach (e.g. a 32-bit dynamic offset +
    small immediate), emit **no instruction at all** — a large guard region behind
    the window catches any escape, exactly as Wasmtime does. Zero hot-path cost,
    matching wasm32.
  - *Unbounded case (the 64-bit window):* mask **the final effective address**
    (after folding base + dynamic offset + immediate offset + `ptr.add`) to the
    window width — a single AND. **Masking the final address is load-bearing for
    security:** masking only the offset operand and then adding a large C immediate
    could land past the guard region in a neighbouring window. Overflow/wrap of the
    masked address stays in-window and is mere guest self-corruption (allowed).
  Masking is also Spectre-v1 hardening: a mask is a data dependency, not a branch,
  so it executes on the speculative path too. A guard region backs both cases; any
  out-of-window / unmapped / wrongly-protected access faults to the host.
- **Confinement is one isolated lowering pass.** The masking/guard logic is a
  single, separately-fuzzable JIT component with the invariant *"every memory
  access is dominated by a mask of the final effective address into `[0, size)`,
  or proven bounded with a guard"* — not diffused through general codegen. This is
  the security hinge (§1a, §18): it is the part the verifier does **not** cover, so
  it is fuzzed and differential-tested in isolation against the interpreter and
  native.

**Guard sizing & the 64-bit framing (clarification).** The window is a *64-bit*
address space (e.g. 2^40 above); there is **no 32-bit index type** as in wasm32,
so the guard region behind the window does **not** need to span 2^32. It only has
to cover the largest *immediate offset + access width* the bounded case will trust
without re-proving the full base bound (pages-to-MiB of reach), because the base
itself is either proven in-window or masked. "Bounded" here means *proven by the
JIT's upper-bound analysis*, **not** free-by-type; a genuinely unknown 64-bit
address always masks. Two strengths of elision follow, and they differ in what
they trust:
- *Conservative (no guard reliance):* the whole access `[addr+offset, +width)` is
  proven `< size`, so the unmasked address already equals the masked one. Sound on
  **every** target — it never depends on a fault.
- *Guard-relying:* only the base is proven in-window; an `offset+width` overrun off
  the top is caught by the guard fault. Sound **only where the guard region exists**
  (see Platform support). This is the wasm32-style win, and it is gated on the guard.

*Reconciling "virtual memory" with "fast":* don't emulate an MMU — borrow the
host's. The guest gets genuine paging semantics with zero software translation,
and the bounded window makes escape impossible without per-access checks.

### Platform support  [Linux + macOS + Windows green]

Confinement itself is portable arithmetic (the masking pass, §16/D51); only the
**non-TCB PAL** — VA reserve/commit/protect/release + guard-fault→trap recovery —
differs per OS. `crates/svm-jit/src/mem.rs` is a portable window model over a small
PAL seam, cfg-selected per target. The full test suite — confinement,
detect-and-kill, the Memory cap, the interp↔JIT escape oracles, and (on unix) the
C frontend — runs green on `ubuntu-latest` (x86-64 / 4 KiB), `macos-latest`
(ARM64 / 16 KiB), and `windows-latest` (x86-64 / 4 KiB) in CI.
- **unix (Linux + macOS):** `mmap(PROT_NONE, MAP_NORESERVE)` + `mprotect` + a
  SIGSEGV/SIGBUS handler via the `cc`-built `trap_shim.c` (`sigsetjmp`/
  `siglongjmp`).
- **Windows:** pure-Rust `windows-sys` — `VirtualAlloc(MEM_RESERVE/COMMIT)` +
  `VirtualProtect(PAGE_NOACCESS)` + an `AddVectoredExceptionHandler` guard with
  `RtlCaptureContext` as the longjmp-equivalent recovery (no C shim, so it stays
  cross-`check`-able from Linux). Two gotchas the bring-up surfaced: the x86-64
  `CONTEXT` must be **16-byte aligned** (it holds XMM state stored with aligned
  moves; windows-sys types it `#[repr(C)]` only, so it needs a `repr(align(16))`
  wrapper), and the cap-buffer borrow needs a guest-window view on non-unix too
  (a portable `WindowMem`, else stdio is silent). Guest-driven Memory-cap growth
  (`map`/`unmap`/`protect` via `VirtualProtect`) and zero-overhead `SharedRegion` aliasing
  (`MapViewOfFile3` over placeholder reservations) now work on Windows too — so all three
  platforms are green with no outstanding per-OS follow-up for the MVP memory model.

The guarantee is identical across targets: same confinement, same detect-and-kill,
same elision. (Guard-relying elision is sound only where the guard region exists —
it does on all three.)

**Page size — host-page default (the "pin page size" resolution).** Page
granularity is the **host MMU page**, queried at runtime, *not* a hardcoded 4 KiB:
x86-64 is 4 KiB, Apple Silicon is a fixed 16 KiB (no 4 KiB granule exists
natively), other arm64 vary. All backends agree by querying the same value — the
JIT/`svm-run` via `sysconf`, the `#![forbid(unsafe_code)]` interpreter via the safe
`page_size` crate — so protection, zeroing, and the page map line up page-for-page
on any host, and the interp↔JIT differential is page-size-agnostic. Two
host-specific subtleties the parity work surfaced: (1) `unmap` must **explicitly
zero** the range — `MADV_DONTNEED` releases anonymous backing on Linux but is only
advisory on Darwin; (2) the chibicc frontend emits portable IR and can't know the
host page, so it pins its compile-time layout constants (RO-data isolation,
heap-growth granularity) to the **largest common host page (16 KiB)** — a multiple
of 4 KiB, harmless on 4 KiB hosts, correct on 16 KiB. The guest can also **query**
the page it is being given at runtime — `Memory` capability op 3 `page_size() ->
i64` (the `__vm_page_size` frontend builtin) — so a guest allocator can align to the
*actual* host page and adapt instead of assuming a fixed size; the shipped
`<stdlib.h>` `malloc` caches it for its growth granularity. Pinning a *deterministic*
guest-visible page (decoupled from the host page) for reproducible cross-host
execution is a later refinement.

A window may itself be a power-of-two-aligned **sub-region of a parent window**
(see §14); confinement is then `base + (offset & (size−1))` with `base`/`size`
as instantiation constants, so a sub-window is indistinguishable from a
top-level window to the code inside it, at identical per-access cost.

**[PARKED]** 64-bit host is assumed; `mmap` churn from chatty `map`/`unmap` is
mitigated by batching and/or a software page-table layer. Exact window size
policy and the demand-paging/userfaultfd plumbing are deferred.

---

## 5. Safety partition & detect-and-kill  [SETTLED]

**Incorruptible by guest writes:**
- SSA locals / virtual registers — not addressable.
- Return addresses and saved registers — live on a **host-managed control
  stack, outside guest-addressable memory** (VA outside `[base, base+size)`, so
  guest masking can never produce an address for it — §4). This gives control-flow
  integrity: even with arbitrary heap corruption, the guest cannot forge a return
  address or jump into host code. No ROP into the host. **The control stack is
  *per-fiber*** (§12), not per-thread: each fiber owns a control+data stack pair,
  and a vCPU executes on the current fiber's pair.

**Corruptible but bounded:**
- Heap and the **per-fiber** data stack live in the window, **bracketed by guard
  pages**, so overruns fault rather than silently corrupting neighbors.

**Detection → kill mechanisms:**
- Page faults (the primary trap), stack-guard hits, trapping arithmetic /
  divide-by-zero, an explicit `trap` / `assert` op (for language-level checks
  and sanitizers), and resource metering (fuel / instruction counting + timer
  preemption).
- Optional **hardened/instrumented tier** (shadow memory, software bounds via
  pointer provenance) that can be swapped for the fast tier once a module is
  trusted.

---

## 6. Flexibility primitives  [SETTLED]

- First-class **tail calls**.
- **Multiple return values.**
- **Stack switching / delimited continuations** as a single primitive — async,
  generators, green threads, and exceptions all build on it.
- Indirect calls via **typed function references** (static type checks where
  possible).

---

## 7. Host interface / ABI  [SETTLED] (revocation PARKED)

We resolved this by separating the security core from the surface and the
schema, and keeping each minimal.

### No ambient authority  [SETTLED]
The committed core. Authority is **never** obtained by naming a global (no
`open(path)`, no `connect(addr)`); it is obtained **only by possessing a
capability**, delegated from an initial grant. This is the one property worth
keeping from ocap, it is cheap (a design rule, not machinery), it kills
confused-deputy attacks, and it makes the §9 egress analysis tractable (a
domain's egress = the transitive closure of its granted capabilities).

### Capability-oriented descriptor surface  [SETTLED]
- Handles are per-domain, **table-indexed, non-integer-castable** references.
  Authority binds to the domain's own table, so a stolen bit-pattern is inert (§9).
- **No global syscall namespace and no ambient host functions.** Every
  operation is reached *through a held handle*: "operation N in the method-table
  of handle type T," invocable only if you possess a handle of type T.
- The host hands the guest an **initial set of handles at instantiation** — that
  set *is* the entire authority grant (the "powerbox"); everything else is
  derived by delegation and attenuation.
- Mechanically this is a syscall-style numbered-op interface — what compiler
  backends already emit — so C/Rust/non-OO toolchains target it with no
  impedance mismatch.

### Host-defined capabilities & discoverability  [SETTLED] (late binding + `cap.self` reflection SETTLED; runtime `Resolver` = host-layer)
The set of capabilities is **open and host-extensible by construction** — the VM,
verifier, and TCB enumerate *no* fixed list. The §3e MVP four are ordinary
instances of this mechanism. A capability interface is just **data + host code**:

- **Interface signature** — an ordered list of op signatures (params/results in IR
  types; a result may be `handle<…>`/`funcref<…>` for attenuation). It lives in the
  **guest module's own type section** (§3a), so the verifier statically type-checks
  every `cap.call` with zero host knowledge — self-contained and verifiable.
- **Implementation** — a method table (vtable, §3c) of handler pointers registered
  **host-side**, entirely outside the guest/verifier/TCB.

**A host adds a capability** by (1) publishing the interface signature out-of-band
(a header-like artifact the toolchain agrees on — *tooling*, never spec/TCB, per
"structured data = pure bytes" below) and (2) implementing + registering the
handlers under a **name**. No VM or verifier change. "Expose a custom capability"
and "expose stdio" are the same act.

**Binding happens once, at instantiation** (§3b): a module's `imports` declare the
interfaces it expects = the structural signature (from the type section) + a
**name/tag** for matching. The host's instantiation policy resolves each named
import to a registered implementation (host decides what to grant), allocates a
`HandleEntry` (§3c) with that interface's `type_id` + vtable + host `object`, and
binds it into the powerbox in declared order. Instantiation **validates the
implementation's signature against the import's declared signature** (structural
compare, fail-closed) — type-safety across the boundary **without an IDL**.

**Discoverability — reflection is ambient, acquisition is a capability.** The
**static-binding** baseline stands: the powerbox is fixed at instantiation; the guest
holds exactly what it imported and was granted; each handle is statically typed
`handle<I>`, so its interface and ops are already known; a missing required import
**fails closed** — nothing to discover. Two refinements extend this without weakening
no-ambient-authority, organized by one line — **knowing what you hold is free; obtaining
something new is a capability**:

- **Late binding (instantiation-time) is the general form of the powerbox.** A module
  declares its capability imports by **name** and the host's instantiation policy resolves
  each to a registered implementation + handle ("Binding happens once," above) — so the
  frontend no longer hardcodes a fixed grant set. A C frontend emits `extern` capability
  references; a wasm frontend emits named imports; both resolve against the same
  `name → (type_id, op)` table. The grant set is still **statically bounded per instance**
  (fixed once instantiation completes), so the §9 egress closure is unaffected — only the
  *binding* moved from compile-time to instantiation-time. The **same named-import mechanism
  generalizes to cross-unit linking** — a name may bind to another function (a direct `call`) or a
  table slot (`call_indirect`), not only a capability — which is how in-window dynamic linking
  (`vm_dlopen`) works; see §22.

- **Reflection = an always-available intrinsic** (`cap.self`), read-only over the domain's
  **own** handle table: it returns the count and, per live entry, the `type_id` + op-schema
  of the capabilities **this domain has actually been granted**. This is *not* ambient
  authority and does **not** void §9: reflecting the held set confers nothing (the guest
  could already invoke every one of those handles), adds no edge to the grant graph, and
  reveals exactly the leaves of the egress closure — never anything the host did not grant.
  It reads **live generation state** (revoked handles drop out) and is a per-domain view, so
  a nested child (§14) reflects only its *attenuated* carve — discovery attenuates for free,
  like `AddressSpace.sub`. It earns its keep precisely under **late binding**, where the
  guest may no longer statically know its full set. *Consequence (accepted):* there are **no
  deniable grants** — a guest can always audit its exact authority. We treat that as a
  transparency feature (a domain can always prove its own least-authority footprint), not a
  leak. The boundary is sharp: enumerating **your own** granted set is fine; there is still
  **no "list everything the host could offer" call** — *that* would be ambient authority.

- **Acquisition = a granted `Resolver` capability** (host-layer, outside the TCB). When a
  guest must *obtain* a not-yet-held capability at runtime (plugin host, service mesh), the
  ocap-correct answer is unchanged: a granted registry interface, e.g.
  `lookup(name) -> handle<…>`, that mints handles through the host table (the same
  handle-minting `AddressSpace.sub`/`Instantiator.instantiate`/`Jit.compile` already do).
  You can only discover-to-acquire via a `Resolver` you were granted, you get back only what
  that registry is scoped to offer, and it is just another node in the grant graph — so the
  egress closure still bounds it. (Distinct from the filesystem `Directory` capability
  below.) Not built now.

### Calling convention  [SETTLED]
The whole platform-level ABI is three things:
- **Scalars** (in registers / stack).
- **Buffers** as `(ptr, len)` with an explicit **own/borrow** bit. Borrow = the
  host reads the buffer in place for the call's duration via the page table (§4),
  no copy; own = ownership transfers. *(This also closes the old data-lifetime
  open item: buffers + own/borrow + handles is the entire data model.)*
- **Handles** as table indices.

### Structured data = pure bytes  [SETTLED]
The platform does **not** define an interface-type system or canonical ABI. A
struct is a buffer plus a layout the *interface* agrees on, not one the platform
dictates. No WIT, no lift/lower, no platform IDL — this keeps the TCB tiny and
serves "simple." Rationale: for guest↔host and intra-domain the host can read
guest memory / modules share an address space, so marshalling is unnecessary;
only the cross-domain (separate-address-space) case needs serialization, and
**the cost of marshalling should scale with distrust** (intra-domain zero-copy;
cross-domain validate-then-read).

**Cross-domain channels: DEFERRED.** A higher host layer will provide channels
for cross-domain structured transfer (likely a self-describing, position-
independent, zero-copy format read in place after validation). Not designed at
the VM layer now. A recommended schema/IDL, if any, ships as *tooling*, never as
spec/TCB.

### File & network capability shapes  [SETTLED]
Fine-grained scoping falls straight out of no-ambient-authority + attenuation,
with no interface types:
- **Files at directory granularity.** A `Directory` capability; the only file op
  is `openAt(dir, relpath, mode) -> File | Directory`, host-enforced to never
  traverse outside the subtree (no `..` escape — Capsicum/`openat` semantics).
  Attenuation: `subdir(dir, rel) -> Directory`, `readonly(dir) -> Directory`.
- **Network at host granularity.** A `Connector` capability scoped to a
  destination set; `connect(c) -> Stream`. Attenuation narrows to a tighter
  host/port/CIDR. The host opens the socket; with no ambient network namespace
  the guest reaches only what its connectors permit. DNS is its own capability
  or folded into `connect` with a host-side scope check.

### Still open
- **Revocation** (PARKED): baseline proposal is host-mediated table invalidation
  (host owns the table; revoked entry traps on next use) + generation counters
  for cheap use-after-revoke detection; transitive/membrane revocation only if a
  concrete need appears. Acceptable v1 fallback: capabilities live until close.
- **Cross-domain channel design** (DEFERRED, above).

---

## 8. Isolation & core concepts  [SETTLED]

### Guest concepts are orthogonal to host primitives  [SETTLED]
Three guest-visible concepts, two host primitives, and the mapping between them
is **host policy the guest cannot observe**:

- **Module** — a unit of code + exports. *Not* an execution or isolation entity.
  Multiple modules freely share a domain (this is intra-domain VM-beside-VM, §13).
- **Domain** — the isolation unit: one window (§4) + one handle table. The
  trust / address-space boundary.
- **Thread** — the execution unit: a stack + scheduling entity, running inside a
  domain over its shared memory.
- Host primitives: **OS process** and **OS thread** (plus cores, MMU).

**Mapping (policy, invisible to the guest):**
- A domain maps to **exactly one process**; many cooperating domains may share a
  process (tiers 0/1). A domain never spans processes (threads must share an
  address space).
- A thread maps to a real OS thread, or is **green-multiplexed** via the
  stack-switching primitive (§6). The guest sees "thread," never "OS thread."

This decoupling is what makes nesting (§14) transparent and zero-overhead: a
parent sub-allocates from its own envelope and expresses isolation *intent* via
capability; the host decides the realization.

### Domains, tiers, sharing
- Threads + shared memory are **intra-domain** (cooperating, native speed).
  Distrust is **cross-domain**.
- Tiers: **0** (same address space, mask + MMU — cooperating only), **1** (same
  process, MPK/PKU — fast architectural path + defense-in-depth, *not* a
  Spectre guarantee), **3** (separate process — robust distrust boundary).
- Explicit **memory consistency model**: the C/C++11 model (§12) — specified, not
  implementation-defined, so the JIT maps deterministically.
  Cross-domain atomics over shared memory (§13) are hardware-coherent — the same
  model applies unchanged across the boundary.
- **Per-fiber** out-of-band control stack + per-fiber guard-paged data stack (a
  fiber owns the *pair*, §12); a vCPU/OS-thread runs on the current fiber's pair.
- Per-domain handle namespace, shared across the domain's threads.
- **Cross-domain sharing is explicit** via shared regions (§13). Cross-domain
  pointers are **not portable** (window-relative), so shared data uses
  region-relative offsets or the ABI.

---

## 9. Spectre hardening, scheduling, split host & exfil stance  [SETTLED]

**Hardening contract for generated code & transitions:**
- Mask-not-branch confinement (already in §4).
- Retpolines / eIBRS for indirect-branch control.
- IBPB + BHB flush on domain switch.
- VERW (MDS) and L1D flush (L1TF) on transitions.
- CET shadow stacks.
- `lfence` / `CSDB` placed surgically at host/guest trampolines, not sprinkled.

**Scheduling discipline:**
- Gang-schedule a domain's threads onto a core / core-set (they trust each other).
- **Never co-schedule distrusting domains on SMT siblings** — disable SMT or use
  core scheduling. (Address-space separation alone does not stop MDS/L1TF across
  siblings.)
- Domain transitions are costly (flushes aren't free) → **batch host calls via
  shared-memory command rings** to amortize the tax.

**Post-compromise / exfil model (the accepted compromise).** If a guest does
succeed in reading another domain's secret, impact is bounded by:
1. **Egress** — every capability is a potential exfil channel. Minimize the
   grant set, and reason about colluding *coalitions* of modules, not single
   modules (the effective egress is the union of the coalition's capabilities).
2. **Covert channels** — timing/cache/contention/DVFS/disk/locks. Low-bandwidth,
   hard to fully close; throttle via resource partitioning, quotas, and timing
   normalization; accept a residual leak.
3. **Authority-bearing-ness** — stolen handles are inert (authority binds to the
   domain table, not a bearer token), so **never pattern local authority as
   knowledge of a secret string**.

Protect the **host's own integrity secrets** (canaries, ASLR base, CFI cookies,
sealing keys) hardest — keep them, and the code that mints new authority, *out of
every guest's address space* in the privileged supervisor (see the split host
below). Bandwidth realism: cross-process Spectre is slow and noisy, so it favors
small high-value secrets (keys, tokens), not bulk data — which is what to defend
first.

### Split host & crossing-cost ladder  [SETTLED]

There is no single "host boundary." There are two, with very different costs, and
the cheap one carries almost all traffic.

- **Fast in-process runtime (guest ↔ trusted runtime).** Host code in the guest's
  own address space, reached only via the capability trampoline. It is
  **secret-less** and exercises **only the caller's own authority** — an extension
  of the guest's privileges, not an escalation point. *That* is what makes it safe
  to be fast: a Spectre hit or confused deputy against it yields nothing the guest
  didn't already have. It is Spectre-hardened code (retpoline/eIBRS, masked arg
  handling). Cost: a stack switch + register save/restore + arg bounds-check + a
  table lookup — wasm-import-call territory, **inline-able to ~free** when the JIT
  knows the target. **No microarchitectural flush** (control returns to the *same*
  guest; no distrust-domain switch).
- **Privileged supervisor (guest ↔ privileged / cross-domain).** Out-of-process;
  holds integrity secrets; mints *new* authority; mediates cross-domain. Paid via
  IPC + (where crossing distrust) the flush tax. Kept **rare** — mostly setup —
  and amortized with async rings.

**Crossing-cost ladder (cheapest → most expensive):**
1. **Inlined / in-process compute capability** (GC, codec, math, buffer ops,
   intra-domain call) — trampoline only, often inlined. ~ns, no syscall, no flush.
2. **vDSO-style read** of host-maintained shared state (time, config, counters) —
   a plain load from a host-updated page. ~free.
3. **Map within the window** — trampoline + one *kernel* syscall (mmap/madvise),
   confined to the window. Kernel crossing, no supervisor IPC, no flush.
4. **I/O on an already-granted resource** — if the supervisor set the process up
   with the right fds + a seccomp filter, the in-process runtime issues the
   syscall *directly*. Native syscall speed; the *kernel* enforces confinement and
   existing OS mitigations cover the kernel boundary.
5. **Async ring submission** when brokering is unavoidable — io_uring-shaped
   submit/complete in shared memory; cross per-batch, not per-call.
6. **Supervisor IPC / cross-distrust-domain call** — the expensive one (context
   switch + Spectre flush). Reserved for acquiring new authority and cross-domain
   mediation.

**The flush tax applies only to switching between mutually-distrusting domains** —
and because domains are gang-scheduled, it is paid **once per scheduling quantum**,
amortized over everything the domain does, *not* per host call. Earlier framing
that lumped "host calls" with this tax was wrong: ordinary guest↔host is path 1–4.

**Direct-vs-brokered syscall knob (security/perf dial).** Letting the guest
process issue confined syscalls directly (path 4) is fast but exposes the kernel's
syscall attack surface — exactly what gVisor removes by interception, at a speed
cost. Default to direct + seccomp; broker through the supervisor (gVisor-style)
only for deployments that distrust kernel robustness.

---

## 10. CHERI & hardware spatial safety  [SETTLED — host-hardening only]

**Decision: CHERI is never imposed on the guest value model.** Guest pointers
stay forgeable 64-bit offsets confined by masking + MMU (§4). If CHERI hardware is
present, it is used only for **host-side TCB hardening** (the runtime, supervisor,
and boundary protecting their own integrity) — the guest never sees it.

**How CHERI works (for reference).** A pointer becomes a 128-bit unforgeable
*capability* — address + compressed bounds + permissions — plus a 1-bit
out-of-band **tag** marking "valid capability." Tags are set only by capability
instructions deriving monotonically (narrow bounds / drop perms, never widen) from
an existing valid capability; any integer-domain write clears the tag. Result:
hardware spatial safety + provenance, pointers unforgeable in silicon. (Cambridge/
SRI; ARM **Morello** is the research prototype.)

**Why host-hardening only, not the guest model.**
- A CHERI capability is **128-bit + tagged**, so it **breaks NaN-boxing**
  (pointers don't fit a 64-bit NaN payload; FP/integer ops clear the tag) and
  constrains aggressive pointer tagging (only low-bit address tags with
  capability-aware ops + masking survive; "pointer is a 64-bit int with free bits"
  does not). That taxes exactly the dynamic-language runtimes (JS, LuaJIT, …) we
  want as guests — porting JS engines to CHERI required reworking value
  representation.
- CHERI's main benefit is **intra-guest spatial safety**, which is an explicit
  **non-goal** (§1: self-corruption is allowed, only ideally detectable). So the
  compatibility cost buys a property we don't require.
- Not mainstream; 128-bit pointers add cache/memory pressure.

**Consequences for the IR.** The `ptr` type stays a CHERI-*ready but erasable*
refinement of i64 (§3a): off-CHERI it is a no-op; a future CHERI backend
can use capabilities for *host* code without touching guest semantics. Considered
and rejected for now: a per-guest opt-in CHERI pointer mode (a C/Rust guest electing
hardware bounds checking) — it means two pointer models in the IR + JIT;
over-engineering until something demands it.

**MTE** (ARM memory tagging) remains a more deployable, lower-cost option for
*optional, probabilistic* intra-guest detection in the §5 hardened tier — without
the value-model disruption. Left open as a hardened-tier ingredient, not a
requirement.

---

## 11. Open questions / parked items (consolidated)

- **Revocation** (§7 PARKED): host-mediated invalidation + generation counters
  vs. capabilities-live-until-close for v1.
- **Cross-domain channels** (§7 DEFERRED): host-layer feature; zero-copy
  self-describing format; designed later, above the VM layer.
- **Runtime acquisition `Resolver`** (§7, host-layer — not built): optional
  `lookup(name) -> handle` registry; host-layer, outside the TCB; not ambient
  authority. (Late binding at instantiation and the always-available `cap.self`
  *reflection* intrinsic are now SETTLED in §7 — only runtime *acquisition* of a
  not-yet-held capability remains a deferred host-layer feature.)
- **MTE** (§10): optional probabilistic intra-guest detection in the §5 hardened
  tier (CHERI settled — host-hardening only, never the guest value model).
- **Type system / value model / binary encoding** — now settled in **§3a**.
- **Window-size & paging policy** (§4 PARKED): default size, userfaultfd plumbing,
  mmap-churn mitigation.
- **Supervisor architecture** (§9): split-host model settled; remaining detail is
  exactly which capabilities are fast-path vs supervisor-brokered.
- **Substrate / backends** (§16): commodity-OS vs seL4; whether to adopt seL4's
  capability-derivation-tree revocation (would close the §7 revocation item).
- **SIMD wider widths** (§17/D58): fixed-128 baseline now SETTLED; `v256`/`v512`
  IR types deferred — **blocked by the backend** (Cranelift has no YMM/ZMM register
  class; owning that codegen contradicts D36/D49), not by the design (which holds:
  the differential survives width-agnostic lane semantics). **Revisit when Cranelift
  adds upstream wide vectors**, not on per-kernel demand. Width-hungry work is better
  served by a host SIMD capability (§7/§13) or the GPU broker. Scalable vectors
  (SVE/RVV) rejected. (GPU settled — WebGPU via sandboxed broker.)

---

## 12. Concurrency model  [SETTLED]

**Mechanism, not policy.** The VM provides primitives; each guest runtime builds
its own threading model (1:1, M:N, async/await, goroutines, actors) on top.

### Fibers & vCPUs (the two primitives)
- **Fiber** — a first-class suspendable stack (an application of §6 stack
  switching). Because of the two-stack split (§3d), a fiber owns a **stack pair**:
  an in-window guard-paged **data stack** + an **out-of-band control stack**
  (return addresses/spills, unreachable by guest masking — §5). Create = allocate
  the pair; switch = save/restore callee-saved + swap *both* SPs (native SP → the
  control stack and its spills follow automatically; plus the per-fiber data-SP)
  (~ns, no syscall, no flush). **Free and uncapped, but quota-metered:** the data
  stack is guest memory; the control stack is out-of-band yet its pages are
  **charged against the guest's memory quota** (§15). So a fiber-bomb OOMs *itself*
  (sandbox-safe) — it cannot exhaust *host* memory via out-of-band stacks. The unit
  of *concurrency*. (`setjmp`/`longjmp` and C++ EH lower onto this switch — §3d.)
- **vCPU** — a capability to run on a physical core, granted with a quota from the
  domain's core-set (§9). Each is an OS thread the host scheduler runs. **Capped**
  — real cores, so resource metering + Spectre core-isolation apply. The unit of
  *parallelism*.
- The runtime multiplexes M fibers onto N vCPUs by any policy it likes; the VM
  imposes none. 1:1 is just one fiber per vCPU.
- **Stackful vs stackless is not a fork.** Provide stackful fibers; stackless
  async (Rust/JS/C# state machines) is free codegen on top — needs no VM feature
  and allocates a fiber only when it actually blocks. Stackful fibers serve both.
- *Rejected:* a built-in M:N scheduler — policy lock-in, the double-scheduler
  pathology (guest runtime over VM runtime), trusted complexity. The reason wasm
  ships none. *(We briefly built one anyway — a green-thread M:N executor in the
  JIT — then removed it to honour this line; see **D56**. The lesson is logged so
  it isn't re-attempted.)*
- **Implemented (the concrete primitives):** `cont.new`/`cont.resume`/`suspend`
  (fibers), `thread.spawn`/`thread.join` (a vCPU = **one OS thread**, 1:1), and
  the `wait`/`notify` futex + C11 atomics — in the IR, interpreter, and JIT
  (x86-64 unix). A spawned vCPU runs the guest entry under the §5 detect-and-kill
  guard on its own OS thread; the guest builds any M:N model over these. **No
  scheduler in the VM** (D56). Deterministic verification of all interleavings is
  the interpreter oracle (`run_scheduled` seed-sweep + `explore_all`, a stateless
  **DPOR** model checker with **sleep sets** + **spin-loop parking** — pruning
  independent-op reorderings and busy-wait retries, sound vs an unreduced
  enumerator, §18), against which the real-thread JIT is differential-tested; the
  futex glue is loom-checked.

### Host-call ABI: async-first
- Blocking-capable host calls are **submit/complete** (io_uring-shaped). The
  synchronous blocking *surface* the source language sees is built by the runtime:
  submit, park the fiber, run another, resume on completion.
- **Blocks the fiber, never the domain.** A single-fiber guest (e.g. C) with
  nothing else to run simply sleeps its vCPU → degenerates to ordinary blocking,
  paying nothing for the machinery.
- Non-blocking capabilities (compute, map-within-window) stay plain synchronous
  calls (§9 cost-ladder paths 1–3).
- *Rejected:* sync-first (cripples M:N — one blocking call freezes every fiber on
  the vCPU); both-as-peers ABI (doubles surface/TCB; async-first's penalty on the
  pure-sync case is negligible).

### Unified event-parking
- All blocking = **park a fiber until an event**. Events: `notify` (futex), I/O
  completion, timer, cross-domain/child signal. One composable wait primitive
  ("wait for any of these") — the convergent OS answer (timerfd/signalfd/eventfd →
  epoll → io_uring).
- **`wait`/`notify`** is a futex over the window: `wait(addr, expected, timeout)`
  parks the fiber if `*addr == expected`; `notify(addr, n)` wakes parked fibers.
  Intra-domain mostly userspace; a host futex is needed only when a vCPU has no
  runnable fiber and must actually sleep. Cross-domain notify signals the other
  domain (slow path).

### Memory model
- **C/C++11 model** (relaxed / acquire / release / acq_rel / seq_cst; RMWs; thread
  fences), lowered by the JIT to the host ISA. Adopt wholesale — it is what LLVM
  emits, real runtimes need relaxed, wasm precedent.
- **Sandbox invariant:** a data race corrupts only the guest's own data, never
  escapes (atomics are masked window accesses like any other). Security is
  invariant across model choice; the choice is guest-semantics + perf only.
- *Rejected as default:* SC-only (full barriers everywhere → slow, worse target);
  TSO/stronger (penalizes ARM/RISC-V). DRF-or-trap race detection is an optional
  §5 hardened tier, not the default (TSan-class cost).

### Keeping cores busy under blocking
Three mechanisms; OS-thread cost is **bounded by host-capped constants, never by
fiber count or I/O concurrency**:
1. **Async ring (fiber-parking)** — any op with an async form. Concurrent I/O
   across any number of fibers = **0 blocked OS threads**; one vCPU reaps
   completions. The primary path.
2. **Bounded blocking-offload pool** (K threads; Tokio `spawn_blocking` / Go
   behavior) — for synchronous-only calls (DNS, some FS ops, third-party blocking,
   synchronous host capabilities). Hand off, park the fiber, a pool thread blocks
   and posts the completion. Cost = K threads regardless of blocked-fiber count;
   the (K+1)th call queues.
3. **vCPU overcommit (M>P)** — for page faults only (mid-instruction, can't be
   offloaded; block the running thread). Split **core quota** (caps simultaneous
   *execution* — fairness + Spectre) from **OS-thread count** (may exceed it); a
   blocked thread isn't executing, so another runs on the freed core within quota.
   Bounded by a small multiple of the quota.
- Total OS threads ≤ core-quota + offload-pool-size + fault-overcommit-factor.
- **Lever:** supervisor-brokered blocking → 0 guest-side blocked threads (the
  supervisor's pool absorbs it; batchable round-trip). Direct path-4 syscalls use
  the guest's own offload pool instead.
- *Rejected:* full scheduler activations (kernel upcall on block) — the
  KSE / Windows-UMS complexity graveyard.

### Preemption & scheduling
- Host preempts **vCPUs** via the fuel/epoch timer (§5) — **undisableable**, so
  cross-domain fairness and killing a runaway guest always work.
- **Fiber** preemption is guest policy via fuel-inserted yield points (Go-style
  async preemption); the VM supplies mechanism, not policy.
- Nested (§14): a child's vCPUs are real OS threads the host scheduler runs within
  the parent's quota — *not* pumped by the parent (that would add overhead and
  break zero-overhead nesting).

### Honest caveats
- **Pool/overcommit sizing is a tuning knob:** too small → blocking calls queue
  (latency); too big → memory + context-switch waste. Pathological all-unique,
  no-async-form blocking serializes past the cap — bounded, not escaped.
- Completion reaping is work at high I/O rates → batch-reap from the ring.
- **Page faults block the vCPU** (can't yield mid-instruction); fast for local
  demand paging, slow for parent-virtualized faults (§14) → prefault/pin hot memory.
- **Reentrancy** (a host capability calling back into the guest) runs on the
  calling fiber so fiber-local state stays consistent; a lock held across a
  callback can self-deadlock (guest's problem, sandbox-safe).

### Optional: deterministic mode
Opt-in host policy: single vCPU + SC + capability-mediated inputs → replay /
record-debugging / consensus. Caveat: true determinism is incompatible with
multicore + relaxed atomics and requires scrubbing every nondeterminism source,
so it is effectively single-threaded — a real mode with real constraints, not a
free toggle.

---

## 13. Shared memory  [SETTLED]

One mechanism for every sharing relationship: host↔guest, guest↔guest
(same-process tiers 0/1), guest↔guest (cross-process tier 3), and parent↔child
(§14).

- A **`SharedRegion`** is a host-backed memory object (anonymous, `shm`/`memfd`,
  or file-backed). Operations: create, **map into a window** at some offset,
  unmap, and **grant** the capability to another domain so it can map the same
  object. Granting `SharedRegion` *is* how two domains come to share memory.
- The same physical pages appear in each window, possibly at **different
  offsets**. Loads/stores are ordinary masked window accesses → **zero
  overhead**. No new access path, no per-access dispatch.
- Because offsets differ per window, shared pointers are **region-relative**, not
  window-relative. Same-offset mapping is an optimization when both ends are
  controlled (e.g. nesting).
- Cross-domain atomics work because it is literally the same hardware-coherent
  memory; the §8 consistency model applies unchanged.
- **Security.** Sharing is a capability — you touch only regions granted and
  mapped; the rest of each window stays private. Shared memory between
  *distrusting* domains is a deliberate, scoped hole; treat the shared region as
  hostile input on the receiving side (validate before trusting; cost scales with
  distrust, per §7).
- **Impact on composition (§14):** this is the **data plane**. Capability calls
  are the control plane (low-rate, pays the §9 transition tax); shared-region
  ring buffers are the bulk data path between adjacent or nested domains, with no
  per-message crossing. Generalizes the §9 command rings from host↔guest to
  guest↔guest.

---

## 14. Composition & nesting (VM-beside-VM, VM-in-VM)  [SETTLED]

**Unifying principle: nesting cost is paid at *setup*, not at *runtime*.** Both
memory access and capability dispatch resolve to a single direct operation
regardless of nesting depth, because the indirection is flattened at
instantiation. The hierarchy lives in the *grant graph*, not the *call path*.

### "Host" is a role, not a level
A child's host is whoever registered the handlers for the capabilities the child
holds. When a parent instantiates a child, each child capability resolves at
grant time to either:
- a **pass-through** to the implementation the parent itself holds — dispatches
  in one hop straight to the ultimate handler, **zero added cost at any depth**; or
- the **parent's own handler** — the parent is *virtualizing* that capability,
  costing one extra crossing **only for the intercepted ops** (pay-for-what-you-
  virtualize).

The child cannot tell whether a capability is real or parent-emulated — the
interface is identical. There is no "am I nested?" query by default.

### VM-beside-VM (composition / linking)
- **Intra-domain:** modules share the address space → direct calls + shared
  handles, like dynamic linking. Bulk data via shared regions (§13).
- **Cross-domain:** control plane = capability calls; data plane = shared regions
  (§13). Cross-domain structured transfer = deferred channels (§7).

### VM-in-VM (nesting), transparent & zero-overhead
- A child's **window is a power-of-two sub-region of the parent's window** (§4).
  Confinement `child_base + (offset & (size−1))` is one AND + ADD with constant
  base/size, so the child sees a zero-based space `[0, size)` and **cannot learn
  it is nested**. Composes to any depth — a grandchild's base is still a single
  resolved constant — so **per-access cost is depth-independent**.
- The parent intrinsically sees all child memory (superset); the child sees only
  its slice (masking). One-way transparency in the correct privilege direction.
- **Lending memory to inner VMs** (the explicit ask): carve a sub-window, or
  share a region into it (§13). For *lazy* page supply, the parent registers as
  the **fault handler** for the child's sub-window (userfaultfd-style): mapped
  access stays zero-overhead; the parent is trapped only on faults it chose to
  virtualize.
- **Instantiation primitive:** an **`Instantiator`** capability lets a holder
  spawn a child domain with (a) a sub-window, (b) an attenuated subset of its own
  capabilities, and (c) a resource/core quota. A parent can only sub-allocate
  what it holds (attenuation), so **a child's isolation tier can never exceed the
  parent's** — a tier-3 child requires the *host* to grant a real process; a
  guest cannot manufacture isolation it lacks.

### New primitives introduced here
- **`SharedRegion`** (§13): create / map / unmap / grant.
- **`AddressSpace`** (memory-management) capability, attenuable to a window
  sub-range: `map` / `unmap` / `protect` within scope; can mint a sub-range
  capability for a child.
- **`Instantiator`**: spawn child domain (sub-window + attenuated caps + quota).

### Honest bounds on "zero overhead"
- Power-of-two, aligned sub-windows → a buddy-style carve of the parent window.
- Deep nesting subdivides VA: a real **window-size vs. nesting-depth** tradeoff
  (a 2^40 window nests many levels, but it is finite).
- "Zero slowdown" = zero *marginal, steady-state* cost for pass-through caps and
  already-mapped memory at any depth. You pay one crossing per *interposed*
  capability op and per *virtualized* fault — the best achievable, and the same
  shape as hardware nested virtualization (cheap steady state, cost on exits).

---

## 15. Resource monitoring & metering  [SETTLED]

**Principle: monitoring is reading the meters on capabilities you granted.** Not a
bolted-on API — it falls out of the grant graph. Every meterable resource is
already a capability with a quota (cores → vCPU quota §12; memory → window
sub-range + `AddressSpace`/`SharedRegion` §4/13/14; CPU-time → fuel/epoch §5; I/O →
granted fds/`Connector`s §7; GPU → device capability §17). The party that minted a
child's grants is exactly the party positioned to observe their use.

### Properties (all fall out of the capability model)
- **Authority-bounded.** A parent observes only what it granted — its child's
  usage against the quotas it set, never a sibling's or anything above its own
  grant. "Who may monitor whom" *is* the nesting tree; no extra access control.
- **Recursive for free.** A child that is itself a parent sub-allocates from its
  own quota, so each level sees its own children at full resolution and everything
  deeper as the aggregate it granted. The grant graph viewed as a monitoring tree.
- **Monitoring and control share the object.** The quota is both the limit and the
  readout, so a parent can also act: tighten a quota, revoke a `SharedRegion`,
  cut a fuel budget, or kill the child (the §5 detect-and-kill path, available to
  a parent over its own children via the lifecycle capability).

### Per-resource readouts (all read off structures the parent already owns)
vCPU/core-time + scheduling stats vs quota; resident/mapped memory vs window +
**fault counts** (double as a §5 self-corruption signal); fuel/epoch consumed +
rate (runaway signal); I/O volume/rates on granted fds/`Connector`s; GPU
submission/time vs device quota; capability-table occupancy.

### Push vs pull
- **Pull:** parent reads meters on demand — cheap, no guest involvement.
- **Push:** parent registers thresholds and gets an event on crossing (e.g.
  ">80% memory quota" or "fuel-rate spike"). Rides §12 event-parking (a monitoring
  fiber parks on the threshold event). Primary interface — polling a deep tree is
  wasteful.

### New primitive
- **`Monitor`/`Meter` capability** — attenuable to a subtree; confers *read* access
  to a child's meters + optional threshold-event registration. **Split from
  `Instantiator`** (which confers control), so observation can be delegated (e.g.
  to a metrics-collector guest) without delegating the ability to re-quota or kill.

### Honest caveat
Resource accounting is observable, so it is also a **covert channel** (§9). A child
modulating CPU/memory/fault behavior can signal a colluding sibling that observes
contention; exposing one child's high-resolution meters to *another* widens it. A
parent monitoring its own children is fine (already more privileged); cross-child
high-resolution visibility is a distrust-scaled, deliberate decision.

---

## 16. Substrate options  [OPEN]

The isolation layer should be an **abstraction with multiple backends**, because
our model is capability-based and no-ambient-authority and so maps onto more than
one substrate:

- **Commodity OS backend** (Linux/Windows/macOS): domains → processes; window →
  reserved `mmap` region; confinement → masking + guard pages; cross-domain →
  shm/`memfd` + IPC; granted I/O → seccomp + fd capabilities. Broadest deployment.
- **seL4 backend** (high-assurance / embedded / hypervisor): our concepts map
  almost 1:1 onto seL4 primitives — domain ≈ CSpace+VSpace+TCBs, handle table ≈
  CSpace capabilities, cross-domain ≈ endpoints (fast IPC), window ≈ a VSpace
  built from untyped/frames, `Instantiator` ≈ Retype + cap grant into a child
  CSpace. Bonus: seL4's **capability derivation tree gives us revocation** (the
  §7 parked item) at the OS layer.

**What seL4 does *not* give us:** the VM itself. seL4 isolates *native* code in
address spaces; our verifier, JIT, SSA target, and masking codegen are a userspace
layer we build *on* it, the same on any backend. And seL4's proofs cover the
*kernel*, not our compiler — our biggest TCB risk (verifier+JIT) persists
regardless. Spectre also stays our problem (the functional proofs don't cover
microarchitecture), though seL4's **time-protection** research is an ally, not a
hindrance.

**Framing:** seL4 is the substrate for the *hardware-isolation half* (tier 3,
cross-domain, supervisor, capability bookkeeping); the *software-isolation half*
(tiers 0/1, the verified-bytecode VM) sits above any kernel and needs no seL4.
Frameworks like the seL4 Core Platform / Microkit, CAmkES, or Genode would host
our components.

### Platform abstraction & portability  [OPEN — Linux/macOS first, Windows next]

The escape-critical core is **already portable**: confinement masking (§4) is pure
arithmetic (`svm-mask` is `no_std`, dependency-free), so the security hinge carries no
OS-specific code. Portability concentrates in two **non-TCB** layers, isolated behind a thin
**Platform Abstraction Layer** (PAL) in the runtime/JIT — never in the audited crates:

- **Virtual-memory management** — reserve a large window, commit the backed prefix, guard
  the tail: `mmap(PROT_NONE)`/`mprotect` (Linux/macOS) ↔ `VirtualAlloc`
  (`MEM_RESERVE`→`MEM_COMMIT`) + `PAGE_NOACCESS` (Windows).
- **Trap-catching safety net (§5 detect-and-kill)** — an out-of-window fault → a clean
  `MemoryFault`: POSIX `SIGSEGV`/`SIGBUS` + `sigsetjmp`/`siglongjmp` (Linux/macOS) ↔ Windows
  **VEH/SEH** on `EXCEPTION_ACCESS_VIOLATION`. **macOS caveat:** Mach exceptions
  (`EXC_BAD_ACCESS`) can intercept ahead of BSD signals (the Wasmtime macOS wrinkle).
- **Futex layer (§12 `wait`/`notify`)** — Linux `futex` ↔ macOS `os_sync`/`__ulock` ↔
  Windows `WaitOnAddress`.

**Lever:** we share Cranelift (D36), and **Wasmtime has already solved cross-platform trap
handling + VA management on all three OSes** — same backend, same problem — so the PAL
borrows a proven design rather than inventing one in §18's riskiest area.

**Tier portability is not uniform:** tier-1 **MPK/PKU is Linux/x86-only**; on macOS and
Windows tier 1 degrades to tier 0 (masking + MMU) or tier 3 (separate process). Tiers 0 and
3 are portable. State this so the isolation story is not over-promised off Linux.

**Staging:** Linux + macOS first (the unix path; the `compile_error!` in `svm-jit`
gates genuinely-unsupported targets, not a permanent stance), Windows VEH/SEH next — **now
done**. Window/mask/interp logic is platform-independent; only the PAL is per-OS. Windows
landed as its own milestone — **Phase 3.5 (§18)** — and Linux/Windows/macOS are now kept at
parity by a gating three-OS CI matrix.

---

---

## 17. Acceleration  [SIMD SETTLED (fixed-128), GPU SETTLED]

### SIMD  [SETTLED — fixed-128 baseline; wider widths deferred]
**Decision: a first-class fixed-width `v128` value type with real hardware
codegen (Cranelift → SSE2/NEON), not scalar expansion.** 128-bit is the
guaranteed floor — SSE2 is on every x86-64 and NEON on every aarch64, so a
`v128` op always lowers to one real vector instruction. **Wasm-parity is *not* a
design goal**; wasm v128 was only the lens that surfaced the gap. The op set is
designed for real hardware SIMD on its own terms (build/inspect, integer/float
lane arith, bitwise, shuffle/swizzle), grown **evidence-driven** — an op is added
only when a real kernel emits it (same program-first discipline as the chibicc
and wasm-transpiler op sets). The wasm bridge (§17→`svm-wasm`) maps wasm v128 →
IR v128 and cleanly rejects what it can't map (`Unsupported`), its existing stance.

- **Escape-TCB delta is small and isolated.** Vector arithmetic/lane/shuffle ops
  are register-to-register and add **zero** escape surface — the verifier just
  gains total lane-typing rules. The *only* confinement change is the wider masked
  access: a `v128.load`/`store` masks the **final effective address** so
  `[addr, addr+16)` stays in-window (the same I1 invariant as scalar, 16 bytes
  wide), so `svm-mask` + `fuzz/mask` gain 16-byte-width coverage (D38).
- **Float lanes are NaN-insensitive in the differential.** Per-lane NaN bits
  aren't pinned across backends, so the interp↔JIT v128 differential is
  NaN-insensitive per lane and vector-float modules stay excluded from the
  byte-exact window oracle — the same caveat as scalar floats today.
- **Wider widths (256/512) are feature-detected, deferred — and the blocker is the
  backend, not the design.** MVP ships `v128` plus a **feature-detection hook** (host
  reports supported vector width) so guests/frontends can dispatch; the wider-width IR
  *types* (`v256`/`v512`) are deferred. The design skeleton is already right and the
  verification story holds: a wider type would add only total lane-typing (zero new
  escape surface), the masked load/store just widens to 32/64 B on `svm-mask`'s
  width-parametric guard, and **the differential survives because lane semantics are
  width-agnostic** — the interpreter (exact per-lane) and the JIT (1× wide *or* split
  into 2×/4× `v128`) agree bit-for-bit on integer lanes, and the feature-detect query is
  per-*machine* not per-*backend* so interp and JIT on one host take the same path (only
  the existing float-NaN caveat carries over). **What's actually missing is in Cranelift:
  its x64 backend has no YMM/ZMM register class** (`RegClass::Float` is XMM/128-bit; the
  `has_avx512f`/`use_avx2` predicates only pick better *128-bit* encodings). Emitting a
  native `vpaddd ymm` needs a new register class + lowering rules *in the shared backend*
  — i.e. owning codegen, which **D36/D49 deliberately don't**. So the "native" lowering
  arm is empty until Cranelift grows wide vectors upstream; the "split-to-`v128`" arm
  gives nothing a hand-written `v128` loop didn't. **ROI of guest-emitted wide SIMD is
  therefore low for this project**: (1) we can't capture the throughput without forking
  the backend off the "as fast/secure as wasm" line; (2) it's effectively **x86-only** —
  ARM has no NEON-256 (its wide path is *scalable* SVE, rejected below), so it cuts
  against portability; (3) AVX-512 presence is fragmented (fused off on Intel hybrid
  client; AVX10 mid-consolidation), and many vector kernels are memory-bandwidth-bound
  where width buys ~0. **The higher-ROI homes for width-hungry work are elsewhere:**
  *(a)* a **host-provided vectorized capability** (`dot`/`gemm`/`memcpy`/crypto/codec the
  host implements with its own tuned AVX-512, invoked via `cap.call` + a zero-copy
  `(ptr,len)` borrow, §7/§13) — guest stays portable + sandboxed, host owns the
  fast/unsafe SIMD, zero backend cost to us, the project's exact grain; *(b)* the **GPU
  broker** (below) for genuinely throughput-bound dense numerics. **Revisit trigger:
  Cranelift adding upstream wide-vector support** (then it slots in nearly for free) — not
  merely "a kernel wants it," since that wouldn't lift the backend blocker. **Scalable
  vectors (SVE/RVV) are rejected** (runtime-variable width makes the one security-critical
  unit — the masked-access bounds proof — runtime-variable too; no Cranelift support;
  fragmented HW benefit).

### GPU = WebGPU via a sandboxed broker  [SETTLED]
The VM does *not* execute GPU code, and the guest never touches the driver. GPU
access is a **WebGPU-shaped capability** (`GpuDevice` / `Surface` / queues +
typed buffers/textures/pipelines/bind-groups). Chosen because it is the one GPU
API already designed for hostile guests, and it is fast/safe enough for the
browser — good enough here.

- **Driver = TCB we don't own.** Unlike CPU code (our verifier+JIT, mature
  silicon isolation), the GPU path runs through a vendor's proprietary driver and
  GPU silicon we can't audit. Strategy is therefore *contain + constrain input*,
  not verify.
- **Sandboxed GPU broker** — the driver runs in its own domain (tier 3). The guest
  calls the validated capability; the broker services it. A driver bug lands in a
  sandboxed process, contained by that sandbox + the IOMMU — not the kernel, not
  other guests.
- **Validated API, not raw command buffers** — every call host-validated and
  bounds-checked; no operation can express raw DMA.
- **Host-recompiled shaders** — guest WGSL/SPIR-V is validated, every
  array/buffer/texture access clamped in-bounds, UB stripped, then re-emitted for
  the driver (Tint/Naga-style). This is the GPU analog of our verifier+JIT.
- **HW defense-in-depth** — per-context GPU page tables, IOMMU-fenced DMA,
  mandatory zeroing of new allocations.
- **Async** — submit/fence; the fiber parks on a completion fence (§12). Zero-copy
  with the window (§13) via staging buffers is the IOMMU-sensitive hot path.
- **Rendering-vs-compute lever** — a higher-level draw-list/canvas API (no
  guest shaders) shrinks guest leverage over the driver; expose the most
  conservative API that meets the need. (Driver stays in the TCB either way;
  containment is what bounds it.)
- **Residual risks (accepted):** safety rests on the validator/translator being
  correct, the broker sandbox holding, and the IOMMU being present/correct — then
  a driver bug is *contained, not catastrophic*. Side channels (pixel-timing,
  contention) → §9-style covert-channel posture. **DoS is the honest weak spot**
  (coarse GPU preemption) → meter + timeout + context-kill.

---

## 19. Debugging & observability  [DESIGN — foundations built; debug surfaces staged]

> **Work-breakdown & detailed designs:** `DEBUGGING.md` (workstreams W1–W8, sequencing,
> open decisions). This section stays the canonical *rationale*; that doc is the *plan*.

Good debugging is a **first-class ergonomics goal**, not an afterthought. The architecture
yields three debugging pillars cheaply, plus one that is real work — pursue all three cheap
ones as pillars and stage the expensive one.

**Status (2026-06).** The *architectural premises* these pillars rest on are now built and
cross-platform-validated — the out-of-band control stack + per-fiber two-stack split (§5/§3d,
`svm-fiber`), the deterministic interpreter oracle (§12, `run_scheduled`/`explore_all`),
capabilities (§3c/§7), and SSA promotion (§3d). The *debug surfaces themselves* — a `cap.call`
record log, an interpreter stepping API, the §3a IR debug-info side-table, DWARF/DAP — are not
yet built. So the pillars have moved from "promised" to "grounded but unimplemented"; nothing
built since has invalidated them.

1. **Record/replay & time-travel — nearly free *in deterministic mode*, a genuine
   differentiator.** With no ambient authority (§7), guest nondeterminism enters through
   capabilities; logging `cap.call` inputs/outputs and seeding the deterministic mode (§12)
   yields a fully **replayable** trace — the capability boundary *is* the recording boundary.
   Time-travel (step backward) follows from deterministic replay to any prior point.
   **Caveat (inherited from §12):** "all nondeterminism via capabilities" holds *only in
   single-vCPU/deterministic mode*. Under the now-built concurrency (threads, relaxed atomics,
   work-stealing fibers — §12/§23) shared-memory **race outcomes** are a nondeterminism source
   that bypasses the capability boundary, so faithful replay there must also record schedule +
   memory-order choices — exactly what the interpreter's DPOR `explore_all` already reifies (a
   stronger substrate than plain seed-replay). The `cap.call` I/O log itself is not yet built.
2. **Trustworthy backtraces even after corruption — integrity free, materialization cheap.**
   The out-of-band control stack (D5/§5, now built) holds return addresses the guest cannot
   forge or smash, so a corrupted in-window data stack cannot destroy the trace — the inverse
   of native debugging, where a smashed stack destroys the backtrace. The *integrity* is free;
   *materializing* a backtrace still needs frame/unwind metadata from Cranelift to walk the
   machine stack, and with migratable fibers (D57/§23) the walk is **per-fiber**, not
   per-thread (each fiber owns its control+data pair).
3. **Reference interpreter as a debug engine — cheap; vehicle mature, surface unbuilt.**
   Single-step / breakpoint / watchpoint over a masked, contiguous window is straightforward
   and deterministic with no JIT plumbing; address watchpoints are trivial (the window is one
   buffer). The interpreter (`svm-interp`) is mature and is already the deterministic oracle,
   but exposes only `run*`/`explore_all` — no stepping API yet.
4. **Source-level debugging (the real work, staged).** Preserve source-location +
   variable-location info **frontend → an IR debug-info side-table (§3a) → Cranelift →
   DWARF**, so gdb/lldb and VS Code (via **DAP**) set breakpoints and inspect variables in
   the *source* language. Cranelift already emits DWARF for JIT code (Wasmtime precedent);
   the new piece is threading debug info through *our* IR. **The dependency is still step
   zero:** the IR (`svm-ir`) carries no source-location fields and chibicc's `codegen_ir.c`
   discards them, so the §3a side-table comes first. A cheap intermediate before full
   DWARF/DAP: a source↔IR-position side-table consumed by the interpreter (pillar 3), which
   needs no Cranelift work.

**Debugger = a host-side capability** (an `Inspector`/`Debugger`, shaped like the §15
`Monitor`): it *observes* a guest from outside, so it never widens the guest's authority and
fits the ocap model. (§15's metering *properties* — fuel/quota — exist on `Host` today, but
`Monitor`/`Inspector` is still a **pattern**, not a built type.) Debug info is **tooling,
untrusted for escape** (§2a) — strippable, and the verifier never trusts it.

**Tension to record (it entangles the §3d perf pass — now concrete, not hypothetical):** SSA
promotion (the implemented headline perf win, §3d) gives a promoted local **no memory
address**, so it is not inspectable as a variable. A debug build therefore either **disables
promotion** (locals stay in-window, addressable) or emits **Cranelift value-location lists**
so the debugger finds the register/stack slot — the classic `-O0`-vs-optimized-debug trade,
here tangled with our headline optimization.

---

## 20. Frontends & language on-ramps  [OPEN — strategy settled, vehicle deferred]

chibicc is the **MVP frontend** (§3d); the goal is to be a target for **many** languages.
The enabling principle should be explicit:

- **The IR is the stable target/ABI; frontends are plugins; every frontend is
  untrusted-for-escape (§2a) and re-checked by the verifier.** Adding a language therefore
  costs **no TCB** — the eBPF lesson, generalized.

Two distinct on-ramps (different bets; the design records both, **priority deferred**):

- **LLVM → our IR (breadth):** buys *every LLVM language* (C, C++, Rust, Swift, Zig…) from
  one component. The team-tractable form is a **PNaCl-style LLVM-bitcode→IR translator** (the
  cited NaCl/PNaCl "SSA as a sandbox target" lineage), not a from-scratch TableGen backend.
  Caveat: LLVM bitcode is not a stable format — pin a frozen subset, as PNaCl did.
- **wasm → our IR (compat):** the whole wasm ecosystem, cheaply — but inherits wasm's
  structured/relooped CFG and 32-bit-flavored memory, so it does not showcase our §1a edges.

**Thesis worth stating: we are a strictly better LLVM target than wasm.** Native irreducible
control flow (D2, no relooper), the 64-bit address space, multi-value returns, and
first-class tail calls (D6) are exactly what LLVM emits and what wasm forces a frontend to
contort — a real §1a differentiator.

**Hard parts to name (not hide):** C++ exceptions / unwinding (the §18 unwind-table open
item), `setjmp`/`longjmp` and EH lowered onto §6 stack-switching, intrinsic coverage, and the
non-negotiable **two-stack constraint** (§3d) — any frontend must place address-taken objects
on the in-window data stack, scalars in SSA, control out-of-band, exactly as `codegen_ir.c`
does. Generalizing that discipline to LLVM is the work.

---

## 21. Host/guest boundary: synchrony & nesting cost  [SETTLED — clarification]

Consolidates what §9/§12/§14 imply but never state in one place: **how synchronous the
host/guest (and guest/guest-as-host) boundary is.**

- **One call shape, and it is synchronous.** `cap.call` produces a result (§3b); the MVP caps
  return a synchronous negative-errno `i64` (D42). There is no separate "async instruction."
  **So host↔guest can be entirely synchronous — that is the default.**
- **"Async" is a construction *on top*, not a second mechanism.** The §12 async-first ABI
  applies only to *blocking-capable* ops: such an op returns a **completion handle**
  synchronously and the runtime parks the fiber (§12 event-parking). Non-blocking caps
  (compute/codec/GC/`map`/vDSO-read) are plain synchronous calls (§9 paths 1–3); a single-
  fiber C guest with nothing else to run simply blocks its vCPU, paying nothing.
- **Synchronous in both directions.** Reentrancy (§12): a host handler may call back into
  guest code on the *same fiber* (a `qsort` comparator, a GC callback) — synchronous
  host→guest as well as guest→host.
- **Nesting (§14):** a child capability resolves at grant time to a **pass-through** (one hop
  to the ultimate handler, zero added cost at any depth) or the **parent's own handler**
  (parent virtualizing). A virtualized op runs **synchronously on the child's calling
  fiber** — child `cap.call` → trampoline → parent handler → return — composing to any depth.

**The governing principle:**

> **Synchrony is interface-guaranteed; cost is host policy the guest cannot observe.**

`cap.call` is always synchronous in *shape*, and the child "cannot tell whether a capability
is real or parent-emulated" (§14). Only the *realized cost* differs, gated by **isolation
tier**, not by the interface:

- **Same process (tiers 0/1 — cooperating / nested sub-window):** trampoline + table lookup,
  **inline-able to ~free, no flush** (§9 path 1). Virtualized hops add one trampoline each;
  pass-through hops add nothing → the zero-overhead-nesting steady state.
- **Across distrust (tier 3 — separate process):** the interface stays synchronous in
  *shape*, but is realized as IPC + (crossing distrust) the Spectre flush tax (§9 path 6).
  Keep it cheap by **batching via async shared-memory rings** (§13 / §9 path 5) — which is
  *why* the ABI is async-first: to amortize the **distrust** boundary, not because the cheap
  one needs it.

**Honest caveat:** a synchronous blocking chain across nesting levels (child blocks →
parent-as-host blocks on *its* host → …) blocks the vCPU per level (§12 overcommit), and
parent-virtualized faults are the slow path (§14). Bounded, but it is where synchronous
nesting bites.

---

## 22. Guest-driven JIT (the `Jit` capability)  [SETTLED — built, differentially tested]

**The problem.** Can *guest* code (an interpreter running inside the sandbox) generate SVM IR at
runtime, hand it to the VM to verify and Cranelift-compile, and call into the resulting native code?
This is the "JIT-inside-the-sandbox" problem wasm handles poorly (W^X + immutable modules force a
guest to ship its own interpreter forever, or round-trip to the host for a fresh module). SVM does it
cleanly, because the submit-a-blob boundary already exists and **the verifier is designed to be the
trust hinge for exactly this**. **Built and merged on both backends, differentially identical**
(`crates/svm/tests/jit_cap.rs`, `jit_incremental.rs`, `jit_reentry.rs`, `jit_compaction.rs`).

**Shape (everything crosses `cap.call`, D55).** The guest builds a serialized IR blob (the binary
`svm-encode` format) in its own window, `cap.call`s a granted **`Jit`** capability with a `(ptr, len)`,
and the host reads the blob, runs `decode_module` → `verify_module` **plus the memory-match
precondition** (below), and — only if all pass — Cranelift-compiles it into a long-lived `JITModule`,
returning a `CompiledCode` handle. Adding the capability needed **no verifier/IR/escape-TCB change** —
an interface signature + a host handler (D45/D46).

**Isolation model — same domain, not a nested guest.** The compiled code joins the *submitter's* domain:
same window, same handle table, same authority (§8 — a module is a unit of code, not an isolation unit).
So the trust boundary is **verification, not isolation**: the blob is exactly as powerful as its
submitter, no more (it cannot reach beyond the window or the granted handles) and no less. For running
*untrusted* code with *less* authority, the right tool is the §13/§14 `Instantiator` (a child VM with its
own window and attenuated handles), not `Jit`. **`Jit` adds speed to a guest; it never adds a protection
domain.**

### Capability surface (iface 11)

`cap.call 11 <op> …` on a granted `Jit` domain handle (negative-errno `i64` ABI, D42):

| op | signature | meaning |
| --- | --- | --- |
| 0 `compile` | `(ptr, len) -> code \| -errno` | borrow blob, decode+verify+precondition gate, compile into the domain, mint a `CompiledCode` handle. Fail-closed. |
| 1 `invoke` | `(code, args…) -> results` | run the unit's entry over the live window (raw i64-slot ABI); a trap is **terminal for the domain** (§5). |
| 2 `release` | `(code) -> 0 \| -errno` | revoke the handle (generation bump). Code/slots not freed (see reclaim). |
| 3 `install` | `(code) -> slot \| -errno` | write the unit into the `call_indirect` table's next reserved slot; returns a funcref index (`-ENOSPC` if full). |
| 4 `uninstall` | `(slot) -> 0 \| -errno` | clear an installed slot (reusable; stale calls trap). |
| 5 `compile_linked` | `(ir, ir_len, symtab, symtab_len) -> code \| -errno` | like `compile`, but the unit may carry **unresolved §7 imports** bound by name against the guest-provided symbol table before verify — the dynamic-linking entry (below). Fail-closed. |

C surface (`frontend/chibicc`, `<svm.h>`): `__vm_jit_compile/compile_linked/invoke2/install/uninstall/release`,
the 8th of the fixed chibicc powerbox; the in-guest dynamic-linking loader `vm_dlopen`/`vm_dlsym`/`vm_dlclose`
(`<vm_dl.h>`) sits on top (below). Demos in `crates/svm-run/demos/jit/` (`jit_demo.c` self-JITs a
bytecode interpreter; `jit_threads.c` concurrently compiles from worker threads; `jit_repl.c` is the
auto-compacting REPL driven by `JitSession`; `jit_link.c`/`jit_dlopen.c`/`jit_hotreload.c` link by name).

### The one structural obstacle — the baked function-table mask

Dropping the module at end-of-run is *policy*; the real obstacle to incremental compilation is that the
**function-table mask is baked as a compile-time `iconst` at every `call_indirect` site** (the
Spectre-safe `slot = band(idx, fn_table_mask)` + `type_id` check, invariant I2 — the touchiest
escape-TCB lowering). Add a function that crosses a power-of-two boundary and every already-compiled
site holds the stale mask. Two models design *around* it, not through it:

- **Model A (shipped, the default) — sidestep the mask.** A submitted unit is reached through the host
  `cap.call` thunk, never installed in the shared table, so the mask never moves; incremental
  define/finalize on a live module is a pure Cranelift capability (`cranelift-jit 0.132`). Cross-unit
  hot calls a REPL would pay per-iteration are absorbed by the guest **re-emitting** the callee's IR
  into the new blob (it owns the IR) — a verified *direct* call at native speed, guest policy, invisible
  to the TCB.
- **Model B2 (shipped, `install`) — neutralize the mask by pre-sizing.** Reserve a fixed power-of-two
  table up front (`table_reserve_log2` / `grant_jit_with_table`, identical on both backends), so the mask
  is constant from `t=0`; only *population* is dynamic (empty slots are `PADDING_TYPE_ID` and trap until
  filled). An installed unit *is* a funcref — old code `call_indirect`s it at native speed, with no
  boundary crossing. (B1, a *growable* table that moves base+mask, is rejected — it would edit the
  Spectre-safe dispatch for nothing B2 lacks.)

All four cross-call directions — old→old, old→new (`install`), new→old, new→new — are covered and
differentially pinned. The reference interpreter matches the JIT's code-pointer dispatch with
**module-aware frames** (`Frame.module`: 0 = the program, ≥1 = an installed unit) dispatching through a
shared, live `DomainTable`.

**Type identity = an append-only intern, not a linking subsystem.** Cross-unit `call_indirect` is sound
because a single `CompiledModule` owns the domain's entire id space: the map `FuncType → type_id` is an
**injection, append-only (an id never remaps), and total over participants** (every signature is interned
before code referencing it is lowered). Given that, id-equality coincides *exactly* with the interpreter's
structural equality. The registry is consulted only inside a synchronous `cap.call` and is **never read at
runtime** — a ~10-line auditable function, zero runtime-readable state. Interning is *cleaner* than a
frozen parent-anchored universe: a later `install` can satisfy a site whose signature arrived in a later
unit, matching structural semantics by construction.

### Security argument

The escape contract (§2a) `Verified ∧ Correct(JIT) ∧ Correct(runtime) ⟹ escape-free` extends to
guest-submitted code: it passes the *same* `decode`+`verify` gate before a single instruction reaches
Cranelift, so a malicious/garbage blob is rejected fail-closed. **The one new authority-TCB precondition
that keeps "no escape-TCB change" true:** confinement masking (I1) masks into `[0, size)` for the
*declared* memory; a blob declaring a *larger* memory, compiled against its own size, would get the JIT to
mask into a larger region — an escape, without touching the masking lowering. The fix lives in the
authority-TCB `Jit` handler: **reject any submitted module whose declared memory ≠ the parent window**
(the host then compiles the blob with the parent's base/mask, so I1 confines it exactly as the main
module). The handler also rejects data segments and §12 concurrency ops inside a *submitted unit* (a
JIT'd blob stays single-threaded; the *parent* may be multi-threaded). A per-domain **compile quota**
(`-ENOMEM`) bounds a looping guest. Net: **Model A adds no escape-TCB surface (authority-TCB only); B2
leaves the indirect-call lowering byte-identical and only adds dynamic table population** on top of the
intern.

### Code reclaim — whole-module compaction

`cranelift-jit`'s `JITModule` has **no per-function free**: every incremental `define_extra` consumes the
256 MiB code arena and nothing is returned, so a long REPL that JITs each prompt would leak code until
`-ENOMEM`. `uninstall` frees a *table slot* but not the *code* behind a stale definition. The reclaim is
therefore **whole-module recompaction**: at a quiescent point, rebuild the *live* unit set into a fresh
`CompiledModule` and drop the old one — RAII frees its entire arena. This is amortized-periodic
O(live-set) recompile, stated honestly, not "incremental forever". The primitives add **no escape-TCB
codegen** (compaction only replays `compile`/`define_extra`/`install` into a fresh module): `install_at`
reinstalls a unit at its *exact* old slot (so a held funcref keeps resolving across the swap, including
around `uninstall` gaps); `extra_byte_count()` is the **byte-accurate** occupancy a watermark watches
(it restarts near zero in the fresh module — the visible reclaim); `is_running()` is the quiescence guard
(the guest is suspended *inside* the module during a `cap.call`, so it can never self-trigger compaction).
The embedder driver `recompact_jit` enumerates live units — installed (`installed_slots`) **or** held
through a live handle (`Host::jit_live_units`) — re-defines each, remaps the `Host` unit→native record
(handles name `(domain, unit)`, not a code address, so they keep working), and reproduces occupied slots.
**`JitSession`** is the auto-compacting REPL driver: it owns the long-lived module + a carried window,
re-enters the entry once per prompt (prior prompt's low bytes seed the next so guest state persists), and
auto-compacts once `extra_byte_count()` crosses a byte watermark — at the quiescent point *after* a prompt.
A 30-prompt session that JITs+invokes+releases a fresh unit every prompt is byte-identical with/without
compaction while occupancy stays bounded (`jit_compaction.rs`, and the C-level `jit_repl.c`).

### Concurrency — threaded install + threaded compile, full platform parity

A guest may be multi-threaded while its JIT'd units stay single-threaded. **Threaded `install`** works on
both backends with no aarch64 gap: the interp shares an atomic `DomainTable` (each slot a packed `u64` —
one `Acquire` load to dispatch, one `Release` store to install, no lock), the JIT publishes
release-ordered atomic `FnEntry` writes (same `#[repr(C)]` layout, so `indirect_dispatch` codegen is
byte-identical). Install *visibility* rides the **guest's own** acquire/release on its ready flag, so a
worker observes a completed install even on weakly-ordered targets — no dispatch-side acquire needed; the
atomic `FnEntry` only guarantees a racy reader never sees a *torn* code pointer. **Threaded `compile`**
works via `svm_run::cap_thunk_locked`, which serializes `cap.call` through a per-domain `Mutex<Host>` so
concurrent `Jit.compile`s are sound (`define_extra`'s `&mut` is exclusive) **while execution stays fully
parallel** — the W^X spike proved `ArenaMemoryProvider::finalize` only re-protects *non-finalized*
segments (executing code, always on a finalized segment, is never touched → no stop-the-world), and i-cache
coherence is cranelift's `clear_cache` + `pipeline_flush_mt`; cranelift *appends* to fresh addresses, so no
cross-modifying-code `isb` is needed. The re-entrant "a running unit compiles more" case is handled by
releasing the lock around `Jit.invoke`. The locked thunk engages **only for concurrent modules**
(`Func::uses_concurrency`); single-threaded runs keep the unlocked fast path at zero lock cost, and the
guest `cap.call 11` iface is unchanged. `JitSession` owns the boxed `Mutex<Host>` and re-bakes the locked
thunk on every recompaction, so a multi-threaded guest auto-compacts soundly between prompts. Pinned by
`jit_cap::threaded_{install,compile}_*` + `cross_thread_execute_fresh_code_agrees` + the
`threaded_session_compacts_transparently` capstone, on every `fiber_rt` target incl. aarch64 macOS.

**Deferred (gated on a measured need):** a coarse-lock→sharded-module optimization if parallel-compile
*throughput* is ever shown to matter — a pure internal swap, since the guest iface is unchanged.

### In-window dynamic linking (`vm_dlopen`)  [SETTLED — built, differentially tested]

Loading separately-authored/compiled code units and resolving cross-unit references **by name** —
the foundation for plugins, dynamic class loading in GC'd-language runtimes, a stateful REPL, and
shared runtime libraries. (Absorbed from the former `DYNLINK.md` when the track completed.)

**The model — linking is a source-to-source rewrite, above the TCB.** A unit carries *named
placeholders* (`call.import "f"` with a self-describing op signature, the §7 import generalized from
capabilities to any symbol). The loader resolves each name against a **symbol table** and rewrites the
placeholder into a concrete instruction; by the time the verifier and both backends see the module it
is an ordinary **closed** module — indistinguishable from any frontend's output, and **re-verified**
like it. There is no runtime "linker" and no new IR execution semantics; a mis-link (wrong signature,
missing symbol) is caught by re-verification, never trusted. Two flavors, both built:

- **Static** (`svm_ir::link`, like `ld`) — merge separate units into **one module** before
  compilation: function symbols → a **direct `call`**, data symbols → **constant addresses** in one
  shared, 16-byte-aligned data window. `LinkUnit { module, exports, data_exports, relocations }`;
  per-unit data is relocated by the unit's data base, `FuncIdx` sites are reindexed by its function
  base, and a `DataReloc`/`RelocKind` patches an address constant by *adding* a base to its addend (so
  `&g + 4` works) — no new IR instruction, just a constant edit. Duplicate / cross-namespace symbol
  collisions, bad exports, and unresolved/non-const relocations all fail closed.
- **Dynamic** (`Resolved::Slot` + `compile_linked`, like `dlopen`) — a **separately-compiled** unit
  reaches a function it does not share an index space with through the **shared `call_indirect` table**
  at a runtime-assigned slot. Both directions work: plugin→host and old/loaded→newly-installed.

**The lowering primitive** is `svm_ir::resolve_imports_with`: each `CallImport` rewrites **1:1** (no
value renumbering) to `Cap(ResolvedCap)`→`cap.call` (the §7 case), `Func(FuncIdx)`→a direct `call`
(static), or `Slot(u32)`→`call_indirect <slot>` (dynamic; the import's handle operand must be a
`ConstI32` placeholder, patched to the slot and reused as the index). Unresolved/ill-typed → fail-closed.

**Serializable imports (codec v2).** So a unit can be shipped as a `.so` with *undefined references*,
the binary form (`svm-encode`) now round-trips the §7 **import section** (name + op signature) and the
`call.import` opcode. v1 was always import-free (imports resolved before encoding); decode stays
untrusted-input TCB and re-verification still gates everything.

**Host-assisted resolve (op 5, both backends).** The design question — does the rewrite run guest-side
or host-side — is **settled host-assisted**: `compile_linked` takes a guest-provided **symbol-table
buffer** and resolves *before* verify (`svm_run::jit_resolve_and_validate`: decode → `resolve_imports_with`
→ verify → the §22 precondition gate). Simpler than a guest-side rewriter (none to load/trust) and
equally safe. The symbol table is a small LEB128 wire form (`count`, then per entry `name` + `kind` —
`0`=`Slot(uleb)`, `1`=`Cap(uleb type_id, uleb op)`); its decoder is fail-closed and fuzzed (a new
untrusted-input surface). The `JitValidator` seam carries the symtab bytes so resolution stays in
`svm-run`; the closed `compile` op (0) is just the empty-table case.

**Security argument.** Resolution is rewrite-**then**-verify, so a missing name, a wrong import
signature, or a non-const slot handle fails verification — nothing reaches Cranelift. The symbol
table's *values* are **guest-chosen by design and confer no authority**: a resolved `Slot` lowers to a
`call_indirect`, which is masked + `type_id`-checked at the call exactly like any index the guest
already controls in its own code — binding to a **wrong-typed** slot **traps `IndirectCallType`**, never
a type-confused dispatch; binding to a forged slot is no worse than a forged index the §3c table check
already handles. So host-assisted linking adds **no escape-TCB surface** on top of §22 — it is the
existing intern + dynamic table population, driven by a guest-controlled name map.

**The guest loader (`<vm_dl.h>`).** `vm_dlopen(name, ir, len)` = build the symbol table from a
`name → {slot, code}` registry → `compile_linked` (host re-verifies) → `install` → record; `vm_dlsym`
= the registry lookup (a funcref **slot**); `vm_dlclose` = `uninstall` + drop. Re-opening a name is
**hot reload**: the new version installs at a *new* slot and the registry repoints, so units already
linked keep their old binding (they baked the old slot) while new resolves see the new one. **Why this
beats POSIX `dlopen`:** the "shared object" is serialized SVM IR, **re-verified** on load (a malicious
one can't escape — worst case it corrupts its own window, a §1 non-goal); `dlsym` returns an
**unforgeable funcref slot** (§3c-checked), not a raw pointer; and loading is **capability-gated** (you
need the `Jit` handle, and the bytes arrive through the powerbox — no ambient "load any file").

Tests/demos: `dynlink.rs` (IR-level static link, interp==JIT), `dynlink_runtime.rs` (runtime by-name),
`dynlink_resolve.rs` (host-assisted resolve + the `Cap` kind), `dynlink_repl.rs` (the symbol-table REPL
spec), `dynlink_cap.rs` (the `compile_linked` op + type-confusion trap, differential), `svm-run`
`symtab_tests` (decoder fuzz), and the guest-C `jit_link.c`/`jit_dlopen.c`/`jit_hotreload.c`. **Open
follow-up:** data symbols through `vm_dlsym` (place data via `Memory`+`memory.init` applying a
`DataReloc`, returning an address) — not yet wired; and a GOT/late-binding variant so *old* code can
call *not-yet-loaded* code by name without recompiling the caller.

---

## 23. Scheduling & migratable fibers (D56/D57)  [SETTLED — built, all slices landed]

How the VM exposes concurrency, why, and how **stackful work-stealing over migratable
fibers** was designed, staged, and verified. (Absorbed from the former `SCHEDULING.md`
when the track completed; the build history lives in HANDOFF.md's log.)

### The model: two primitives, nothing more

| Primitive | What it is | Why only the VM can provide it |
|---|---|---|
| **vCPU** (`thread.spawn`/`join`) | one real OS thread, 1:1 (D56) | parallelism across physical cores — not expressible in portable guest code |
| **fiber** (`cont.new`/`resume`/`suspend`) | a *stackful* coroutine that owns a native call stack | switching the native execution stack needs the `svm-fiber` asm stack-switch — the guest's instruction set can't save/restore SP + callee-saved regs and redirect execution mid-function |

Plus the coordination glue that is *also* primitive-minimal: the `wait`/`notify` **futex**
and **C11 atomics** over the shared window. Everything richer — mutexes, channels, M:N
schedulers, work-stealing, async runtimes — is **guest-built** from those (D22/D56:
*primitives, not policy; no scheduler in the VM*).

**"Stackless tasks" are NOT a third primitive.** A stackless task is a function rewritten
as a state machine (a struct of locals + a resume fn with a `switch` on a state field —
exactly how Rust `async`/C++ coroutines lower). It needs **zero** VM support: suspend is
`return`, resume is calling it again. The primitive surface stays at two; stackless is a
guest *pattern*. It is also strictly *less expressive* (function-coloring: it can suspend
only at points in its own transformed body, never across an arbitrary or unmodified call
frame) — fibers are the only way to cooperatively suspend **unmodified real code**, and
they underpin the §14 fault-driven yield (suspending at an arbitrary hardware-fault PC is
inherently stackful).

### Migratable fibers (D57) — what the VM owes, and what it doesn't

The instinct is "build a Chase-Lev work-stealing deque." **The VM builds no deque.** The
work-stealing run-queue is guest code (a deque of fiber handles in guest memory). The VM
owes only the two things migration needs that the guest cannot provide itself:

1. a **shared fiber-handle namespace** — any vCPU can *name* any fiber (one slot table per
   domain on both backends; handles are `0, 1, …` domain-wide, identical across backends);
2. the **single-owner arbiter** — when two workers race to `cont.resume` the same handle,
   exactly one wins and the loser gets a clean `FiberFault`.

So the entire VM-side surface is one shared slot table, each slot carrying an **`Ownership`
word** — no deque, no policy, no scheduler.

**The ownership protocol** (`svm-jit/src/fiber_registry.rs`, loom-model-checked): one
`AtomicU64` per slot packing `(generation, state)` with states `OWNED` (fresh, or pinned
fault-suspended) / `RUNNABLE` (voluntarily suspended — the pool) / `RUNNING` (never
claimable) / `FREE` (returned; generation bumped, so a stale handle's claim fails — the ABA
guard for eventual slot recycling). The live resume path claims via `claim()`
(`OWNED|RUNNABLE → RUNNING`, acquire CAS; loom proves exactly-one-winner from both start
states plus published-context visibility); a voluntary suspend publishes via
`suspend_to_pool` (release), a return `finish`es. The acquire/release pair is the
happens-before edge that makes resuming a native stack *another thread* saved sound.

**As built, per backend:**
- **Interpreter** (the oracle): a fiber is pure data (`Vec<Frame>`), so migration is a safe
  hand-off through the run-shared `FiberRegistry` (one mutex'd table per domain; the mutex
  is the arbiter — observably identical to the CAS). Each vCPU's root computation runs
  off-table (this unified the handle namespace with the JIT, closing the recorded §3a
  divergence). Fiber ops are *visible* ops to the deterministic explorer, recording a
  `MemAccess::Fiber` conflict so DPOR explores both orders of racing fiber ops; the
  spin-detection fingerprint covers chain/frames/parked-root rather than the shared table.
- **JIT**: the domain-shared `fiber_rt::SharedFiberTable` (slots are `Arc`'d; the boxed
  fiber stays address-stable across table growth; the finished fiber's stack is unmapped at
  `finish`). The cross-thread resume calls the **unchanged** `svm-fiber` switch — none of
  the three ABIs carries thread-bound state (SysV/AAPCS64 save only callee-saved registers,
  no TLS/x18; MS-x64 swaps the TEB `StackBase`/`StackLimit`/`DeallocationStack` per switch).
  Per-thread state (`CURRENT_RT` for yielder pairing; the §5 guard recovery) is re-read
  after **every** switch-in, never carried across a suspension. Fixed-mmap stacks are fine:
  migration moves the executing thread, not the stack (Go copies stacks only for growth).

**What stays per-thread:** the resume chain (a worker's current native/eval call stack) and
the JIT `yielders` stack — migration only ever moves a *suspended* fiber (on no chain). A
fiber anywhere in a resume chain is `RUNNING`, so a re-entrant resume loses the claim and
faults (this replaced the per-thread `chain` checks on both backends). **Quota (§15):**
`max_fibers` is per-run/domain (the shared table's slot count) on both backends.
**Compatibility:** a guest that never resumes a foreign fiber sees identical behavior;
migration is opt-in by the guest's scheduler choosing where to resume a handle.

**Staged remainders** (recorded here; tracked in HANDOFF): *slot recycling* +
generation-carrying handles must land on both backends together and need the interp
explorer to record a fiber-return DPOR access first (a `Return` that frees a slot conflicts
with `cont.new` once slots recycle — `finish` already maintains the generation under the
hood); *pinned fault-suspended fibers* (`pin`: `RUNNING → OWNED`, excluded from migration
because `sigjmp_buf`/VEH recovery state is thread-affine) is designed and kept in the
protocol but nothing produces such fibers on the `cont.*` surface yet.

### The honest tension with D56, and the verification story

D56 removed a VM-owned M:N executor specifically because it reintroduced the project's
highest-risk unsafe — fiber migration across OS threads — in the runtime TCB. D57
re-accepts **exactly that risk**, but as a *primitive* (single-owner
resume-from-any-thread; the guest owns all stealing policy), resolving D56's policy
objections while accepting its TCB-risk one with eyes open: **no expert review is available
for the asm/signal seam** (a stated project constraint), and the composition (verified
protocol + real asm switch + per-thread signal recovery) cannot be model-checked. Safety
therefore rests on an **empirical net**, every layer of which is built and green:

1. **Randomized-migration differential** — `fiber_fuzz::generated_migration_schedules_agree_on_interp_and_jit`:
   generated programs whose fibers suspend/resume across sequences of spawned vCPUs,
   deterministic by construction; the safe-Rust interpreter is the oracle; cross-executor
   saved-stack resumes are *counted and asserted*, not assumed.
2. **ASan with real fiber-switch annotations** — `svm-fiber` brackets every switch with
   `__sanitizer_start/finish_switch_fiber` behind the `asan` cargo feature (chained
   `svm/asan → svm-jit/asan → svm-fiber/asan`), re-capturing came-from stack bounds at each
   switch-in so even migrated resumes are tracked; the whole fiber suite runs clean under
   `-Zsanitizer=address`. (The feature link-requires the sanitizer runtime by design.)
3. **Runtime single-owner assert** at the resume seam (`FiberSlot::running_on`): a
   double-claim aborts loudly instead of running one native stack on two threads.
4. **Guard-paged stacks** + per-thread detect-and-kill recovery: a wild/torn switch faults
   cleanly on whichever thread runs the fiber.
5. **Concurrent-steal stress** — `jit_threads::concurrent_fiber_steal_stress`: racing
   workers claim saved stacks under contention, every second resume a guaranteed
   cross-thread migration, schedule-invariant sum.

**Honest residual:** fuzzing *detects*, it does not *prove*; a sufficiently rare
cross-thread race could escape the net. Accepted knowingly as the price of the capability.

### The evidence: three guest schedulers (demos)

1. `demos/mn_sched` — **sharded stackful M:N** (thread-per-core, tasks pinned by choice;
   glommio/seastar shape).
2. `demos/work_stealing` — **work-stealing over stackless tasks** (state-machine structs,
   injector + per-worker deques; tokio shape). No VM change needed — the pre-D57 proof.
3. `demos/steal_fibers` — **work-stealing over stackful fibers** (the D57 capstone): idle
   workers steal *suspended fibers* and resume them on their own OS threads; the task
   yields from inside a nested call frame (inexpressible stackless) with live locals
   carried across every migration, so its return-sum is a stack-integrity check. Both
   invariant totals identical on interp and JIT
   (`c_frontend::c_guest_steal_fibers_demo`, `run::demo_steal_fibers_runs`).

All three are *entirely guest code* over the two primitives — the D56/D57 thesis, proven
three ways.

## 24. Security & correctness audit — record  [CLOSED — all findings fixed]

Audit date **2026-06-10** (register formerly `AUDIT.md`; deleted when every finding
closed). Scope: the escape-TCB (verifier, masking unit, memory substrate, decoder) and the
unsafe-heavy backends (JIT lowering + mask elision + cap ABI, the §14 nesting runtime, the
fiber/thread runtime + the §5 kill-path). Method: four parallel deep-dive reviews + a
direct review of the capability/authority model.

**Verdict: the escape-TCB is sound.** No memory-safety escape, no unsound optimization
(mask elision is provably upper-bound-sound), no arbitrary-code path. All findings
clustered in **availability (host-survivability)** and on-paper-UB/robustness hardening,
not confinement — and all eight are fixed:

| # | Sev | Finding (all ✅ fixed) |
|---|-----|------|
| 1 | MED–HIGH | guest could abort the host by exhausting the 256-slot handle table (`grant` panicked across the `extern "C"` thunk) → fallible `try_grant`, guest-minting sites return `-EMFILE`; pinned by `address_space::minting_past_table_capacity_returns_emfile_not_panic` |
| 2 | MED (on-paper UB) | racy non-atomic Rust writes to the shared `trap_out` cell → `AtomicI64` + `store_trap`/`load_trap` (Relaxed); JIT code keeps its aligned hardware-atomic store |
| 3 | MED (defensive) | JIT nesting validated child size vs clamped size could diverge → one clamped value for both |
| 4 | LOW | cap.call result buffer partially uninitialized on host arity mismatch → zero-filled |
| 5 | LOW | `Mapped` atomic width dispatch treated any non-4 width as 8 → `debug_assert!(width == 4 \|\| 8)` |
| 6 | LOW | `Paged::read_into` debug-overflow on huge `off` → early out-of-range guard (inert, per the `Region` contract) |
| 7 | LOW | decoder `Vec::with_capacity(ndata)` ~40× allocation amplification → incremental growth |
| 8 | LOW | futex `HashMap` entries never pruned → removed at `waiters == 0` |

**Checked and found sound (no action):** the verifier (fail-closed, `forbid(unsafe_code)`,
every bound checked); `svm-mask` `confine` (in-window for all inputs incl. `u64::MAX`;
`Window::sub` can't wrap past its sub-range); the `svm-mem` unsafe contracts; JIT mask
elision (`ub_of`/`in_window` upper-bound-sound, saturating, width-accounted); the cap.call
ABI (buffers sized from the compile-time verified sig); trap propagation (re-checked after
every call/thunk); `call_indirect` (Spectre-safe masked dispatch + type-id check); the
`svm-fiber` switches (register/alignment-complete on all three ABIs; body panic → abort);
the decoder (every count bounded, LEB128 overflow-guarded); the §5 kill-path (epoch-cell
lifetime via `join_all`; parked vCPUs re-check); capability forgery resistance (generation
+ type_id checked under a Spectre-safe masked index; `create_region` capped).

## Prior art / touchstones

- **eBPF** — verified bytecode in a hostile host; helper calls as the only
  escape. Our philosophy, generalized beyond its deliberate restrictions.
- **Cranelift CLIF** — the block-parameter SSA target shape (§3).
- **NaCl / PNaCl** — LLVM-bitcode-as-portable-sandbox-target; closest prior
  attempt at "SSA as a sandbox target."
- **CHERI / Morello** — hardware capabilities for spatial safety (§10).
- **WebAssembly + proposals** — capability imports, the guard-page trick,
  memory64, typed continuations, the component model (interface types).
- **Chrome site isolation; Swivel / "Spectre is here to stay"** — the basis for
  the §2 accepted compromise (process boundary for distrust).
- **Firecracker / KVM microVMs** — near-native nested isolation; the EPT/NPT
  "cheap steady state, cost on exits" cost model we mirror in §14.
- **Capsicum / CloudABI** — direct ancestor of the §7 `Directory`/`Connector`
  capability shapes (openat-from-preopens, no ambient authority).
- **seL4 / capability microkernels** — candidate isolation substrate (§16); the
  formally-verified TCB bar our verifier+JIT is measured against.
- **Singularity (MSIL SIPs) / KeyKOS–EROS–Coyotos** — "language safety *is*
  isolation," and pure capability OSes.
- **vDSO / io_uring; L4 fast IPC; gVisor** — the crossing-cost playbook in §9
  (gVisor as the cautionary slow-path opposite).

---

## 18. Build plan & MVP estimate  [PLANNING]

**Implementation context:** Claude Code implements; a non-expert guides. No deep
JIT/systems expertise on the team. **Host (escape-TCB: verifier, runtime, JIT glue)
= Rust** (Cranelift-native; best-in-class fuzzing via `cargo-fuzz` + `arbitrary`;
memory-safe TCB; compiler-as-safety-net for the agent — D49). **Frontend = a
chibicc-style C compiler in C** (untrusted-for-escape per §2a, so its language
carries no sandbox-safety cost) emitting our IR. Codegen lowers to **Cranelift**
(don't write our own backend). Compile-time tax accepted, mitigated by `cargo check`
+ cached Cranelift builds.

**Why a single speed multiplier misleads.** Agent speedup here is wildly
non-uniform. *Fast* (volume / known patterns): chibicc frontend, IR + encoding,
interpreter, Cranelift glue, capability plumbing, tests. *Slow & risky* (novel,
correctness-critical, systems-fiddly, debug-heavy): verifier soundness,
masking/window/mmap/guard-page/signal plumbing, atomics/concurrency, and deep-bug
debugging. The slow part dominates schedule + risk — and is exactly where the
team has **no expert safety net**.

**Phases (wide error bars):**
- **Phase 1 — Core loop** — IR + encoding + verifier + interpreter; run
  hand-written IR. *~2–6 weeks.*
- **Phase 2 — Compilability proof** — chibicc→IR frontend; real C runs on the
  interpreter. The "it works" milestone; mostly agent-fast. *~1–3 months total.*
- **Phase 3 — Solid MVP** — Cranelift JIT + windowed memory model (masking, mmap,
  guard pages) + capability runtime; real C running fast in a confined window.
  *~6–15 months, median ~9–12, fat tail.* This is where the systems plumbing and
  deep debugging concentrate.
- **Phase 3.5 — Cross-platform parity** — port the runtime to **Windows** and lock
  parity across **Linux / Windows / macOS** from here on. The escape-critical core
  is already portable (confinement masking is pure arithmetic), so only the non-TCB
  **Platform Abstraction Layer** differs (§16/D51): VA management
  (`VirtualAlloc`/`VirtualProtect`), the detect-and-kill safety net (Windows
  **VEH/SEH**; macOS **Mach exceptions**, which can intercept ahead of BSD
  signals), and later the futex layer (`WaitOnAddress`, once §12 concurrency
  lands). **Phase 3.5 is now done:** the JIT once `compile_error!`d off unix with
  Linux-only CI; today it runs on Linux, macOS, and Windows (Windows VEH/SEH was the
  real work) under a **three-OS gating CI matrix** that keeps every *later* phase green
  on all three. Tier-1 MPK stays Linux-only and degrades to tier 0/3 elsewhere —
  parity is of the *portable* tiers, stated honestly. *~1–2 months, gated on a
  solid Phase-3 MVP.*
- **Phase 4 — Deferred (post-MVP), developed against the parity matrix** — full
  concurrency, nesting, shared memory, isolation tiers, Spectre hardening,
  split-host supervisor, monitoring, GPU, revocation. *(**§17 SIMD has landed**:
  fixed-128 `v128` (D58) end-to-end across IR/text/encode/verify/interp/JIT/wasm with
  native Cranelift SSE2/NEON codegen; the only escape-TCB delta is the 16-byte masked
  load/store on `svm-mask`'s width-parametric guard; v128 ops ride the 4000-seed
  interp↔JIT differential, a real clang `-msimd128` saxpy transpiles to verified SIMD
  IR, and the `bench` SIMD kernel is at ~1.0× Wasmtime (compute parity). Wider
  widths `v256`/`v512` deferred (D58). **Concurrency
  primitives have landed early**: fibers `cont.*`, 1:1 `thread.spawn`/`join`, the
  `wait`/`notify` futex + C11 atomics, in IR/interp/JIT across the parity matrix
  (interp everywhere; JIT on x86-64 unix, aarch64 unix, x86-64 Windows) — **no VM
  scheduler**, M:N is guest-built (D56/§12). **§14 nesting** has also landed on both
  backends (sub-windows, the attenuable `AddressSpace`, the `Instantiator` incl.
  recursion + co-fibers + fault-driven yield, separate-module "plugin" children, and
  cross-domain `SharedRegion` `create`/`grant`), as has the **§5 fuel/epoch kill-path**
  (the lowering polls a host-owned interrupt cell, so a watchdog stops a runaway guest
  with `OutOfFuel` — across the root vCPU, sibling vCPUs incl. parked ones, and nested
  children; the interpreter has its per-step fuel counter). Still deferred here: guest
  M:N runtimes as worked examples, the async submit/complete ring (§9/§12), fiber/vCPU
  quota *metering* (the kill path exists; quotas don't), and honoring *weak* memory
  orderings.)*

**The hard ceiling (call it out, don't bury it):** in this configuration
**"appears to work" is reachable; "is actually secure" is not.** The verifier +
masking layer are the entire escape-prevention claim, and a non-expert + a
fluent-but-not-sound agent cannot certify the TCB is trustworthy. Closing that gap
is a **separate post-MVP workstream** needing capability the team lacks: expert
review, serious fuzzing/differential-testing infra, eventually an audit. Treat it
as open-ended, not a byproduct of the build.

**De-risking moves that fit this setup:**
- **Interpreter-as-oracle:** differential-test the JIT against the interpreter on
  a large random corpus — catches codegen bugs without expert eyes.
- **Fuzz the verifier from day one** (invariant: verified ⇒ cannot escape) — the
  one security validation that doesn't need a continuous expert in the loop.
- **Fuzz the confinement-masking lowering as its own unit** (D38) — it is the part
  the verifier does *not* cover and the true escape hinge. Invariant: *every
  generated memory access is dominated by a mask of the final effective address
  into `[0, size)`, or proven bounded behind a guard.* Differential-test masked
  addresses against the interpreter's checked addresses; add a self-test that
  asserts no access instruction reaches a raw, unmasked, unbounded address.
- **Lean on Cranelift** (removes the hardest codegen risk) — and note it *is* the
  security story: sharing Wasmtime's backend is how we are simultaneously
  "as secure as wasm" and "compute-parity with wasm" (§1a, D36).
- **The design doc itself** substitutes for the missing systems-architecture
  experience — keep the agent anchored to it.
- Most-likely tar pits → memory-model/confinement plumbing and anything
  concurrent; worth buying a few hours of expert review there even without a hire.

**Pre-MVP specification checklist** (design → spec transition):
- ✅ Instruction set, trap/UB semantics, FP/endianness, verifier rules, entry &
  instantiation contract — **§3b**.
- ✅ **C ABI (Phase 2 blocker):** two-stack split, address-taken/SSA local split,
  LP64 type mapping, struct layout, by-pointer aggregates, varargs, data segments,
  const RO data, `malloc`/`free` over `map`, Phase-2 C subset — **§3d**. Remaining:
  toolchain/linking (MVP = whole-program single module) is trivial under §3d.
- ✅ **Concrete window params (Phase 3):** resolved — a *large* reserved window
  (`2^40`) with guest-controlled growth + kernel demand paging, host-page default
  (page size queried at runtime, not pinned), final-effective-address masking,
  guard-page detect-and-kill, and real `map`/`unmap`/`protect`/`page_size` — **§4 / §3e**.
- ✅ **Minimal MVP capability set:** `Stream` (stdio), `Exit`, `Clock`, `Memory`
  (`map`/`unmap`/`protect`); negative-errno model; powerbox + args-buffer — **§3e**.
- ✅ **TCB / threat-model writeup:** the honest conjunction contract, escape-TCB vs
  authority-TCB, the I1–I5 invariants × owner × validation table, scope (DoS a
  non-goal), microarch posture, handler hygiene — **§2a**.



| # | Decision | Status | Why |
|---|----------|--------|-----|
| D1 | Block-local typed SSA, no phi, explicit block params | Settled | Linear verifier, no dominance analysis; great producer/consumer target |
| D2 | Native irreducible control flow | Settled | No relooper; direct LLVM target |
| D3 | Reserved VA window + host MMU for virtual memory | Settled | Real paging, zero software translation, bounded → escape-proof |
| D4 | Mask (not branch) for confinement | Settled | Hot-path speed + Spectre-v1 robustness |
| D5 | Control stack out of guest memory | Settled | Control-flow integrity; no ROP into host |
| D6 | Tail calls, multi-return, stack switching | Settled | Broad language coverage |
| D7 | Domain = unit; threads/shared-mem intra-domain; distrust cross-domain | Settled | Matches OS reality; pairs with Spectre scheduling |
| D8 | Tiers 0/1 (cooperating) + 3 (distrust); in-process is defense-in-depth | Settled | Accepted Spectre compromise |
| D9 | No ambient authority; capability-oriented descriptor surface (ops keyed to held handles) | Settled | Security core; cheap; kills confused deputies; makes egress analysis tractable |
| D10 | CHERI provenance | **Open** | Interested; deployment + cost concerns; MTE alternative |
| D11 | Calling convention = scalars + buffers(own/borrow) + handles | Settled | Syscall-shaped; tiny TCB; closes data-lifetime question |
| D12 | Structured data = pure bytes; no platform IDL/canonical ABI | Settled | Component-model complexity not needed (host reads guest mem; shared address space); marshalling cost scales with distrust |
| D13 | File = `Directory`/`openAt` (no `..`); network = scoped `Connector` | Settled | Directory- and host-granular scoping via capability shape + attenuation |
| D14 | Cross-domain channels deferred to a host layer | Deferred | Not needed at VM layer; design later |
| D15 | Revocation model | **Parked** | Host-mediated + generation counters vs. live-until-close |
| D16 | Module ⊥ domain ⊥ thread; mapping to OS process/thread is invisible host policy | Settled | Enables transparent, zero-overhead nesting; domain↔one process |
| D17 | Shared memory = `SharedRegion` mapped into multiple windows; region-relative offsets | Settled | One mechanism for all sharing; zero-overhead masked access; data plane for composition |
| D18 | Nesting cost paid at setup not runtime; pass-through caps + sub-window memory are depth-independent | Settled | Transparent + zero steady-state overhead; cost only where parent interposes |
| D19 | Child window = power-of-two sub-region of parent; `Instantiator` grants sub-window + attenuated caps + quota | Settled | Child can't tell it's nested; tier can't exceed parent's |
| D20 | Split host: secret-less in-process fast runtime + out-of-process privileged supervisor | Settled | Fast where it's safe to be fast; flush tax only at distrust boundaries, amortized per quantum |
| D21 | Direct confined syscalls by default; broker (gVisor-style) only when distrusting the kernel | Settled | Native syscall speed for granted resources; surface-reduction is an opt-in dial |
| D22 | Mechanism-only concurrency: free uncapped fibers + capped vCPU capabilities; runtime builds the model | Settled | Sane target for every threading model; no built-in scheduler / no double-scheduling |
| D23 | Async-first host-call ABI; sync surface built by runtime fiber-parking; blocks fiber not domain | Settled | M:N without head-of-line blocking; C degenerates to ordinary blocking for free |
| D24 | Unified event-parking (futex/completion/timer/signal → one wait); C11 memory model | Settled | Composable waits (the epoll/io_uring convergence); LLVM-native atomics |
| D25 | Blocking: async ring + bounded offload pool + M>P overcommit (faults); no scheduler activations | Settled | OS-thread cost bounded by host-capped constants, not concurrency; avoids activation graveyard |
| D26 | Host preempts vCPUs (undisableable); fiber preemption is guest policy via yield points | Settled | Fairness/killing always work; no VM-imposed fiber scheduler |
| D27 | Optional deterministic mode (single vCPU + SC + cap-mediated inputs) | Settled (opt-in) | Replay/consensus; effectively single-threaded by nature |
| D28 | GPU = WebGPU-shaped capability via a sandboxed driver broker; host-recompiled shaders | Settled | Only GPU API designed for hostile guests; contain-don't-verify the unownable driver TCB |
| D29 | CHERI used only for host-side TCB hardening, never the guest value model; guest pointers stay forgeable 64-bit offsets | Settled | CHERI breaks NaN-boxing/tagging (taxes dynamic-lang guests) for intra-guest safety we treat as a non-goal |
| D30 | Resource monitoring = reading meters on granted capabilities; `Monitor` cap split from `Instantiator`; push thresholds via §12 | Settled | Monitoring tree = grant tree; recursive, authority-bounded; observation delegable without control |
| D31 | Two-class value model: plain data forgeable (confined) + capabilities as **inert typed table indices** (superseded "sealed" framing, see D37) | Settled | The verifier's escape-impossibility theorem; C-compatible pointers; authority binds to the table, not the index |
| D32 | Encoding fuses decode+verify in one forward pass: block-local indices + up-front block-signature table + inferred typed-opcode results | Settled | Cheapest possible verifier (no dominance, no fixups); minimal TCB |
| D33 | Section-based module (decls before bodies); LEB128 + zstd, no bespoke compression; text format 1:1 with binary, text-first for the build | Settled | Independent/parallel function verify+JIT; agent-friendly debugging |
| D34 | IR is total — no UB; every op gives a defined value or a defined trap (source UB resolved by the frontend) | Settled | UB in a sandbox IR would void the escape guarantee |
| D35 | Phase-1 IR spec pinned: instruction set, trap/wrap/saturate semantics, little-endian + IEEE FP, complete verifier rules, entry/instantiation contract | Settled | The concrete spec the verifier+interpreter are built from (§3b) |
| D36 | Goal = relative to wasm: as secure as wasm (host), faster on interface/64-bit/startup with **compute pegged at Wasmtime parity** (shared Cranelift), simpler+more flexible interface | Settled | Absolute "escape impossible" not certifiable by this team; relative bar is reachable and measurable (§1a) |
| D37 | Capabilities are **inert typed table indices**, not a sealed value class; a forged index traps or re-selects an own grant (authority binds to the table entry) | Settled | Removes §3a/§7 contradiction; lets handle/funcref live in registers/memory and lets C function pointers lower to function-table indices |
| D38 | Confinement = **guard-when-bounded, mask-when-not**, masking the **final effective address**, implemented as one isolated separately-fuzzable lowering pass | Settled | Matches wasm32 hot path (zero instructions), beats wasm64; final-address masking closes the large-immediate escape; isolation makes it fuzzable as the security hinge |
| D39 | C ABI: forced **two-stack split** (out-of-band control stack + in-window guard-paged data stack); address-taken→data stack, scalar non-address-taken→SSA; LP64/little-endian; **by-value aggregates by hidden pointer** (sret); clang-wasm-style vararg buffer | Settled | Window+masking (§4) and out-of-band control stack (§5) force the split; by-pointer is simplest-correct and ~wasm parity; whole-program MVP needs no external-ABI match (§3d) |
| D40 | Const globals + string literals in a **read-only data segment** (`protect` at instantiation) | Settled | One extra protect call → writes to const data fault → §5 detect-and-kill; cheap self-corruption detection |
| D41 | A fiber owns a **stack pair** (in-window data stack + out-of-band control stack); stacks are **per-fiber**; the control stack is unreachable by guest masking (CFI) but **charged to the guest's memory quota** (so a fiber-bomb self-OOMs, not the host) | Settled | Reconciles the §3d two-stack split with §12 fibers; keeps both CFI (§5) and "fibers metered/sandbox-safe" (§12/§15); switch swaps both SPs, ~ns |
| D42 | MVP cap ops use a **negative-errno `i64`** result (`≥0` success value, `<0` `-errno`); errors never trap; buffer args are borrow-only `(ptr,len)` validated at the trampoline (`-EFAULT` on overflow) | Settled | Syscall-shaped (§7), 1:1 with the C libc shim; keeps traps reserved for escape/fatal (§3b) |
| D43 | MVP capability set = `Stream` (stdio via 3 handles), `Exit`, `Clock`, `Memory`; stdio reuses one `Stream` interface (not a bespoke Console) so files/sockets compose later | Settled | First concrete handle-table interfaces (§3c) + C-runtime targets (§3d); orthogonal, one interface to verify (§3e) |
| D44 | Powerbox = `entry(stdin, stdout, stderr, exit, clock, memory, args_buffer)`; args buffer = `{argc,envc}` + packed NUL-terminated strings | Settled | Concrete instantiation grant + C `main` wrapper contract (§3b/§3d/§3e) |
| D45 | `cap.call` dispatch is **per-entry** (vtable in the `HandleEntry`), not per-type — generally an indirect call (retpoline/eIBRS), devirtualized to direct/inline when the binding is statically known. **Devirtualization is deferred — cost recorded in §3c** (authority-TCB in codegen, fights compile⊥instantiate, sound only for stable handles, only half the measured cost; scalar `cap.call` ~1.24× wasm but the zero-copy buffer win needs none of it) | Settled (devirt deferred) | Corrects §3c over-claim; one interface type has many implementations per handle, and §14 virtualization (pass-through vs parent-emulated) needs per-handle dispatch; forgery checks unchanged. Deferral is a recorded trade, not an oversight — don't relitigate without a measured workload |
| D46 | Capability set is **open/host-extensible** (interface signature in the module type section + host-registered vtable, bound by named import at instantiation, signature-validated fail-closed); **binding may be late** (names resolved at instantiation; the general form of the powerbox), **reflection over the domain's *own* granted set is an always-available `cap.self` intrinsic** (read-only; authority-neutral), and runtime *acquisition* of a not-yet-held cap is a granted `Resolver` registry deferred to a host layer | Settled | The §3e four are just instances; late binding keeps the per-instance grant set statically bounded; `cap.self` confers nothing (reflection ≠ amplification) so the §9 egress-closure analysis stays intact; only acquisition (`Resolver`) widens the grant graph, and it stays a gated host-layer cap outside the TCB |
| D47 | Escape-freedom is the **conjunction** `Verified ∧ Correct(JIT) ∧ Correct(runtime) ∧ Correct(host/HW)`, not "verified ⇒ safe"; TCB split into **escape-TCB vs authority-TCB**; decomposed into invariants **I1–I5** (owner + validation each); written as a **structured-prose contract**, not a proof | Settled | Puts risk where it lives (JIT dominates, not the verifier); makes host-extensible caps safe (authority-TCB ≠ escape-TCB); anchors the security work; matches the "as secure as wasm" bar (§2a) |
| D48 | **Availability / DoS is a non-goal** — bounded by metering (fuel/quota/preemption) + the kill path, contained not prevented (incl. §17 GPU); hardware fault injection below the trust line; trust boundary is **verified IR**, frontend untrusted for escape (eBPF model) | Settled | Honest scope; avoids claims the metering/preemption story (and GPU) can't back; verifier makes the frontend untrusted for escape (§2a) |
| D49 | Host (escape-TCB) in **Rust**; frontend in **C**; backend **Cranelift** | Settled | Backend is Rust-native (coupled to D36); Rust gives memory-safe TCB + best fuzzing (`arbitrary`) + compiler safety net for an expert-less agent build; frontend's language is safety-irrelevant (§2a), so C/chibicc is free; compile-time tax accepted |
| D50 | **Accept the mask cost on unbounded-base accesses; do not pursue 32-bit window addressing.** Mask elision (§4 guard-when-bounded) covers *provably-bounded* addresses; for an unbounded base (the threaded data-SP in C locals) we keep the single AND mask (`locals_c` ~2.26× wasm32, still < wasm64) rather than lower window addresses as 32-bit | Settled | The 64-bit address space is a core goal (D36/§1a); the only sound way to elide an unbounded-base access is the wasm32 trick (32-bit address arithmetic, address `< 2^32` by construction so it matches the interp and elides) — masking the i64 data-SP alone is un-elidable or diverges from the interp (an escape). That trick caps the elided window at 4 GiB and reworks the frontend's pointer model for one benchmark; not worth trading the clean 64-bit model. Revisit only if a real workload makes the data-SP mask a measured bottleneck |
| D51 | **Portability via a thin non-TCB Platform Abstraction Layer** (VA reserve/commit/protect, guard-fault→trap, futex); confinement masking stays platform-independent; **Linux/macOS first, Windows (VEH/SEH) next**; tier-1 MPK is Linux-only and degrades elsewhere. Scheduled as **Phase 3.5** (§18): port Windows, then hold Linux/Windows/macOS parity via a gating three-OS CI matrix | Open (staged) | The escape hinge is portable arithmetic; only the safety-net/syscalls differ per-OS; Wasmtime already proves the cross-platform path, so lean on it (D36/§18) |
| D52 | **Capability-boundary record/replay** as the primary debugging differentiator: in deterministic mode (§12) nondeterminism enters via capabilities (§7), so logging `cap.call` I/O + seeding that mode gives replayable, time-travel debugging; trustworthy backtraces come free from the out-of-band control stack (§5). **Caveat:** under multicore + relaxed atomics (§12/§23) race outcomes bypass the cap boundary, so faithful replay must also record schedule/memory-order choices (the interp's DPOR `explore_all` already reifies these) | Proposed (premises built; surfaces unbuilt) | Debugging ergonomics are a first-class goal; the ocap boundary is the cheap recording boundary *in single-vCPU mode*; the control stack survives heap corruption |
| D53 | **Debug surfaces = three cheap pillars + staged DWARF:** reference-interpreter stepping/watchpoints, record/replay, and §5 backtraces now; source-level DWARF (frontend→IR debug side-table→Cranelift→DAP/gdb/lldb) staged. Debug info is untrusted tooling (§2a); debug builds **disable §3d promotion or emit value-locations** so locals stay inspectable; debugger is a host-side `Inspector` capability (like §15 `Monitor`). **Status:** premises built (control stack, deterministic interp, SSA promotion); no stepping API, cap.call log, or §3a debug side-table yet — the side-table is step zero for DWARF | Proposed (premises built; surfaces unbuilt) | The cheap pillars fall out of the architecture; DWARF is the real work; promotion-vs-inspectability is a real trade; debugger-as-capability never widens authority |
| D54 | **Frontends are untrusted IR plugins (verifier re-checks all); multi-language via two on-ramps — LLVM-bitcode→IR translator (breadth, PNaCl-style, pinned subset) and wasm→IR bridge (compat) — vehicle priority deferred.** Our IR is a *better LLVM target than wasm* (irreducible CFG, 64-bit, multivalue, tail calls) | Open (strategy settled) | IR-as-stable-ABI makes language breadth a no-TCB-cost effort (§2a); a bitcode translator beats a TableGen backend for an expert-scarce team (D49); the §1a edges are real LLVM-target advantages |
| D55 | **One synchronous `cap.call` shape; async is a runtime construction over blocking-capable ops.** Synchrony is **interface-guaranteed**; **cost is tier-policy** the guest cannot observe: same-process nesting (tiers 0/1) is synchronous and ~free to any depth; cross-process (tier 3) keeps the shape but pays IPC and batches via §13 rings | Settled (clarification) | Unifies §9/§12/§14; the IR has only a synchronous call; "async-first" amortizes the *distrust* boundary, not the common case; matches zero-overhead nesting (§14) |
| D56 | **Concurrency primitives only, no scheduler in the VM (honouring D22).** The VM exposes `cont.*` (fibers), `thread.spawn`/`thread.join` (a vCPU = **one real OS thread**, 1:1), and the `wait`/`notify` futex + C11 atomics — implemented in IR/interp/JIT. The guest runtime builds any M:N model over them. **A built-in M:N green-thread executor was implemented and then removed**: it gave deterministic seeded/exhaustive *JIT* scheduling but reintroduced exactly D22's costs (policy lock-in, the double-scheduler pathology, and the project's highest-risk unsafe — fiber migration across OS threads — in the runtime TCB). Verification keeps what mattered without it: the **interpreter** is the deterministic oracle (`run_scheduled`/`explore_all` exhaust interleavings at instruction granularity — a sound model of preemptive 1:1 threads), the real-thread JIT is differential-tested against it, and the futex glue is loom-checked | Settled (course-correction) | Removes the §12/D22 contradiction the executor introduced; shrinks the TCB; keeps the VM **less** opinionated than wasm on threading (threads are a 1:1 primitive, not a baked scheduler); the deterministic-exploration win lived in the interp oracle all along, not in owning the scheduler |
| D58 | **SIMD = first-class fixed-128 `v128` with real hardware codegen (Cranelift→SSE2/NEON), not scalar expansion; wasm-parity is not a goal.** 128-bit is the guaranteed floor (SSE2/NEON universal), so a `v128` op = one real vector instruction. Op set designed for real hardware SIMD on its own terms and grown **evidence-driven** (an op is added only when a real kernel emits it). Escape-TCB delta is small/isolated: vector arith/lane/shuffle ops add **zero** escape surface (verifier gains total lane-typing only); the lone confinement change is the 16-byte masked `v128.load`/`store` on the final effective address (`svm-mask`+`fuzz/mask` gain 16-byte width, D38). Float lanes are NaN-insensitive in the interp↔JIT differential (NaN bits unpinned across backends → vector-float modules excluded from the byte-exact window oracle, as scalar floats are today). **Wider widths (`v256`/`v512`) feature-detected and DEFERRED — blocked by the backend, not the design.** The design skeleton is right (wider type = total lane-typing only; mask widens to 32/64 B on the width-parametric guard; the differential survives because lane semantics are width-agnostic — interp's exact lanes == JIT's 1×-wide-or-split, and the feature-detect query is per-machine not per-backend). What's missing is in **Cranelift: no YMM/ZMM register class** (`RegClass::Float` = XMM/128-bit; the `avx2`/`avx512` predicates only pick better 128-bit encodings), so a native `vpaddd ymm` needs a new register class + lowering *in the shared backend* = owning codegen, which D36/D49 refuse. The "native" arm is thus empty until Cranelift adds wide vectors upstream; the "split-to-`v128`" arm equals a hand-written `v128` loop. **ROI of guest-emitted wide SIMD is low**: can't capture throughput without forking the backend; **x86-only** (ARM's wide path is scalable SVE, rejected — no NEON-256); AVX-512 fragmented; many kernels memory-bound. **Higher-ROI homes:** a host-provided vectorized capability (host owns tuned AVX-512 behind `cap.call` + zero-copy borrow, §7/§13 — portable guest, zero backend cost) and the GPU broker. **Revisit trigger = Cranelift gaining upstream wide-vector support**, not "a kernel wants it." **Scalable vectors (SVE/RVV) rejected** (runtime-variable width makes the masked-access bounds proof runtime-variable; no backend support; fragmented benefit). | Settled (fixed-128); wider widths deferred | Real hardware SIMD is the goal; wasm was just the lens. Fixed-128 is the portable floor that always lowers to a real instruction with no scalar fallback. Wider widths are deferred not on evidence-discipline grounds but on a hard backend blocker (no Cranelift YMM/ZMM regclass) — owning that codegen contradicts the "as fast/secure as wasm via shared Cranelift" thesis (D36/D49); and width-hungry workloads are better served by a host SIMD capability or the GPU broker. Vector ops are value-only so the security story barely moves — only the masked access widens |
| D57 | **Two concurrency primitives are the floor; "stackless tasks" add none.** vCPU (`thread.spawn`, 1:1) gives parallelism; fiber (`cont.*`) gives suspension of *native* execution. A **stackless task** (a guest-compiled state machine — struct + resume fn + a `switch` on a state field) is a *guest pattern* needing **zero** primitives: its suspend point is the state-machine transition, built from ordinary loads/stores/branches. So guest-built M:N comes in two flavors **today, with no VM change**: *sharded* M:N over **thread-affine** fibers (tasks pinned to their worker), and **work-stealing** M:N over **stackless** tasks (freely movable — moving a struct is a pointer hand-off, safe by construction; over `thread.spawn`+futex+atomics). Stackless is strictly *less expressive* (function-coloring: it can only suspend at points in a transformed body, not across arbitrary/unmodified frames), so fibers stay — they're the only way to cooperatively suspend **unmodified real code** and they underpin the §14 fault-driven yield (suspend at an arbitrary hardware-fault PC is inherently stackful). **Stackful migration over *migratable* fibers is ADOPTED and landed (slices 3a–3c):** it re-accepts D56's deliberately-removed cross-thread-fiber-migration unsafe, but as a **primitive** (the VM enforces a single-owner *resume-from-any-thread* — the loom-verified `Ownership` claim; the guest owns any stealing policy) rather than a VM scheduler — resolving D56's policy-lock-in / double-scheduler objections while accepting its TCB-risk one *with eyes open*: no expert reviewer is available for the asm/signal seam, so safety rests on the **empirical net** (§23 "verification story" — the randomized-migration interp↔JIT differential, ASan with real fiber-switch annotations in `svm-fiber`, a runtime single-owner assert at the resume seam, guard-paged stacks, concurrent-steal stress). Both backends migrate: the interp's run-shared registry (pure-data hand-off, the oracle) and the JIT's domain-shared table over the *unchanged* `svm-fiber` switch (no thread-bound state in any of the three ABIs; MS-x64 swaps the TEB stack fields per switch). Fault-suspended fibers stay pinned (`pin`, staged); slot recycling is staged behind generation-carrying handles on both backends. | Adopted (extends D56; 3a–3c landed) | Pins the primitive count at two and the "no VM scheduler" rule; records the migratable-fiber path honestly as a re-acceptance of a known high-risk unsafe, not a free win — to be earned, not assumed. Full design + verification story + demo evidence in **§23** |
| D59 | **Guest-driven JIT = the `Jit` capability (§22): a guest submits verified IR and gets native code in its *own* domain.** Verification, not isolation, is the trust boundary — a JIT'd unit is exactly as powerful as its submitter (same window/handles), with one new authority-TCB precondition (declared memory ≡ parent window; reject data segments + concurrency ops in a unit) keeping "no escape-TCB change" true. Model A (cap.call trampoline, default) sidesteps the baked function-table mask; Model B2 (`install` into a pre-reserved table) neutralizes it by pre-sizing — both ship, all four cross-call directions differentially pinned. Type identity is an append-only intern (id-equality ≡ structural equality), never read at runtime. Code reclaim is whole-module recompaction (no per-function free in cranelift-jit), driven by `recompact_jit`/`JitSession` on a byte watermark. Threaded `install` + threaded `compile` work with full platform parity (atomic `DomainTable`/`FnEntry`; `cap_thunk_locked` serializes compiles while execution stays parallel; no aarch64 `isb` needed). | Settled (built, differentially tested) | The "JIT inside the sandbox" wasm handles poorly; the submit-a-blob boundary + verifier-as-hinge already existed, so it was authority-TCB-mostly. Model A's worst case is perf/ergonomics (announced by a benchmark), B2's is a host-writes-into-live-table primitive — both earned their place; the sharded-module throughput optimization stays deferred until measured. Full design + security argument + reclaim/concurrency in §22 |
