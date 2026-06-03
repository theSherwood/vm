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
  table (§7); trampolines inlinable to ~free; batched async rings (§9, §13).
  *This is the strongest, most defensible win.*
- **64-bit address space:** faster than wasm64 (one AND mask vs explicit bounds
  check). Against wasm32 it would be a *wash or slightly worse* — so confinement
  is **"guard-when-bounded, mask-when-not"**: use wasm's zero-instruction
  large-guard-region trick where the offset is provably bounded (the wasm32 common
  case, zero added instructions), and mask only the unbounded 64-bit window (§4).
  This matches wasm32 on the hot path and beats wasm64.
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
  lattice → trivial verifier).
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
  extend32_s`; `trunc_sat_f→i_s/u` (**saturating default**, deterministic; trapping
  variant available), `convert_i→f_s/u`, `f32.demote/f64.promote`; `reinterpret`
  (i32↔f32, i64↔f64 — bit-level, for NaN-boxing).
- **Pointers:** `ptr.from_int` / `ptr.to_int` (free, no-op off-CHERI — the §10/§3a
  casts), `ptr.add` (offset by integer; lets the JIT/CHERI backend see pointer
  arithmetic).
- **Memory:** `{i32,i64,f32,f64}.load/store`; narrow `load8_s/u load16_s/u load32_s/u`
  + `store8/16/32` (C char/short). Address operand + immediate offset + alignment
  *hint* (unaligned allowed). Confinement masking is implicit (JIT-inserted).
- **Atomics** (C11, §12): atomic load/store at orderings; RMW `add sub and or xor
  exchange cmpxchg` at orderings; `fence`; `wait`/`notify` (futex).
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

### Trap / numeric / layout semantics
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
7. **Contract:** a module that passes verification + the memory model (§4) + the
   out-of-band control stack (§5) ⟹ **escape is impossible.** (Soundness *of the
   verifier/JIT* is the separate, hard problem — §18.)

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

*Reconciling "virtual memory" with "fast":* don't emulate an MMU — borrow the
host's. The guest gets genuine paging semantics with zero software translation,
and the bounded window makes escape impossible without per-access checks.

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
  stack, outside guest-addressable memory**. This gives control-flow integrity:
  even with arbitrary heap corruption, the guest cannot forge a return address
  or jump into host code. No ROP into the host.

**Corruptible but bounded:**
- Heap and the per-thread data stack live in the window, **bracketed by guard
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
- Per-thread out-of-band control stack; per-thread guard-paged data stack.
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
- **MTE** (§10): optional probabilistic intra-guest detection in the §5 hardened
  tier (CHERI settled — host-hardening only, never the guest value model).
- **Type system / value model / binary encoding** — now settled in **§3a**.
- **Window-size & paging policy** (§4 PARKED): default size, userfaultfd plumbing,
  mmap-churn mitigation.
- **Supervisor architecture** (§9): split-host model settled; remaining detail is
  exactly which capabilities are fast-path vs supervisor-brokered.
- **Substrate / backends** (§16): commodity-OS vs seL4; whether to adopt seL4's
  capability-derivation-tree revocation (would close the §7 revocation item).
- **SIMD** (§17): fixed-128 baseline vs scalable vectors (GPU now settled —
  WebGPU via sandboxed broker).

---

## 12. Concurrency model  [SETTLED]

**Mechanism, not policy.** The VM provides primitives; each guest runtime builds
its own threading model (1:1, M:N, async/await, goroutines, actors) on top.

### Fibers & vCPUs (the two primitives)
- **Fiber** — a first-class suspendable stack (an application of §6 stack
  switching). Create = allocate a stack in the window; switch = userspace register
  save/restore + SP swap (~ns, no syscall, no flush). **Free and uncapped** — it
  is guest memory, already metered by the window (a fiber-bomb OOMs itself,
  sandbox-safe). The unit of *concurrency*.
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
  ships none.

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

---

---

## 17. Acceleration  [SIMD OPEN, GPU SETTLED]

### SIMD  [OPEN]
Fixed-width 128-bit baseline (portable, simple, safe — vector ops touch values,
not memory escape) vs. scalable vectors (SVE/RVV-style, width-agnostic, harder to
JIT/verify). Lean: fixed-128 baseline + feature-detected wider widths.

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
JIT/systems expertise on the team. Frontend = a chibicc-style C compiler emitting
our IR. Codegen lowers to **Cranelift** (don't write our own backend).

**Why a single speed multiplier misleads.** Agent speedup here is wildly
non-uniform. *Fast* (volume / known patterns): chibicc frontend, IR + encoding,
interpreter, Cranelift glue, capability plumbing, tests. *Slow & risky* (novel,
correctness-critical, systems-fiddly, debug-heavy): verifier soundness,
masking/window/mmap/guard-page/signal plumbing, atomics/concurrency, and deep-bug
debugging. The slow part dominates schedule + risk — and is exactly where the
team has **no expert safety net**.

**Phases (wide error bars):**
1. **Core loop** — IR + encoding + verifier + interpreter; run hand-written IR.
   *~2–6 weeks.*
2. **Compilability proof** — chibicc→IR frontend; real C runs on the interpreter.
   The "it works" milestone; mostly agent-fast. *~1–3 months total.*
3. **Solid MVP** — Cranelift JIT + windowed memory model (masking, mmap, guard
   pages) + capability runtime; real C running fast in a confined window.
   *~6–15 months, median ~9–12, fat tail.* This is where the systems plumbing and
   deep debugging concentrate.
4. **Deferred (post-MVP):** full concurrency (fibers/vCPUs/M:N), nesting, shared
   memory, isolation tiers, Spectre hardening, split-host supervisor, monitoring,
   GPU, SIMD, revocation.

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
- ⬜ **C ABI (Phase 2 blocker):** stack-frame / data-stack model; address-taken vs
  SSA-value local split; struct layout + alignment; by-value aggregates; varargs;
  calling convention; globals/statics → data segments. Plus toolchain/linking
  (MVP = whole-program single module) and `malloc`/`free` over the `map` capability.
- ⬜ **Concrete window params (Phase 3):** §4 is parked — MVP simplification =
  fixed-size window, eager mapping, no demand paging; pin page size, masking
  constant, guard-page placement, minimal map/unmap/protect.
- ⬜ **Minimal MVP capability set:** console/stdio, exit, clock, memory (`map`).
- ⬜ **TCB / threat-model writeup:** make the §3b rule-7 contract precise
  ("verified ⇒ invariants ⇒ escape impossible") — anchors the security work.



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
