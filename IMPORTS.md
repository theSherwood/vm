# Imports, exports & the powerbox — the binding redesign

Status: **PROPOSED** (design review). Amends `DESIGN.md` §3a/§3b/§7 and builds on
`PROCESS.md` §6 (`cap.self.attest`). Nothing here changes the escape-TCB posture:
the confinement masking (§4) and the handle-table use-site checks (§3c) are
untouched; every change below is authority-TCB or below.

This document has three maturity levels, deliberately kept distinct:

1. **Core** — build now. A consolidation: it deletes more than it adds.
2. **Designed, build on demand** — settled shape, parked until a consumer exists.
3. **Parked with reasons** — real designs, deliberately not scheduled.

---

## 1. Motivation: five binding conventions, one question

"How does a call site name a granted capability?" is answered five ways today:

1. **Positional powerbox args** — handles as leading entry args, threaded as
   leading `i32` params through *every* function (the wasm transpiler's
   "data-SP trick"; ~73 sites in `svm-wasm`).
2. **The window stash** — handles written into reserved window slots, reloaded
   by frontend code.
3. **The numeric convention** — a wasm import `("42", "7")` meaning
   `type_id 42, op 7`, bound at transpile time.
4. **Named imports rewritten at instantiation** — `resolve_imports` lowers
   `call.import` → `cap.call`, in two flavors (`Resolved::Cap` keeps the handle
   a call-site operand; `Resolved::CapBound` patches a `ConstI32` placeholder).
5. **Runtime name lookup** — `register_cap_name` + `cap.self.resolve` (F7).

Each grew for a real reason (chibicc threading, wasm benchmarking, POSIX
personality, linking, discovery), but the redundancy is accidental complexity,
and path 4 — the rewrite — has costs that have grown teeth:

- **Module identity dies at instantiation.** The verified/hashed/attested bytes
  are not the executed bytes. `CapBound` bakes a *granted handle value* into an
  instruction, so two instantiations of one module have different code.
- **It defeats the compile cache.** The §14 child compile cache (PROCESS.md S1,
  hardened for cross-thread sharing in S1c) keys on the funcs slice's identity
  — `(funcs_ptr, len, entry, size_log2)`. Per-instantiation rewriting
  guarantees misses. The cache's own residual note (O3: "a digest would beat
  the funcs pointer") points the same direction: content-addressed module
  identity, which rewriting destroys.
- **The patcher is attack surface.** `resolve_imports_with` performs ~100 lines
  of def-index mapping and in-place instruction surgery, guarded by a
  re-verification pass, coordinated with frontends via the fragile "handle
  operand must be a `ConstI32` placeholder" convention (`SlotHandleNotConst`).
- **Re-verification per instantiation** is real work in a project with a
  BOOTSPEED.md.

The fix is not new machinery. The design below is built almost entirely from
pieces already in the tree: the powerbox prefix ("instantiation fills the first
N entries", §3b), the host-owned handle table with type/generation checks
(§3c), `cap.self` reflection incl. by-name resolve (D46/F7), and the
`spawn_named_child` grant flow.

### The bake/parameterize rule (context)

The recent S1/S1c work established a trajectory: parameterize instance
*placement* (window base is a runtime arg; the child cache is
position-independent), share the artifact, keep security-critical constants
baked. The sorting rule, made explicit:

| | Low cardinality | Per-instance / unbounded |
|---|---|---|
| **Hot** (every access) | **Bake**: confinement mask, fn-table mask | **Register arg**: `mem_base` (done) |
| **Cold** (per op / host call) | either | **Instance state**: bindings, kill cell, rt addresses |

Import bindings are cold + per-instance: they belong in instance state, not in
rewritten code. (The two masks stay baked: their variation is cheap to handle
by cache-keying/pre-sizing, their runtime cost is the only nonzero one, and
their auditability is the escape-TCB's boringness. If the parameterized-cold
set keeps growing, consolidate it behind one instance-context pointer rather
than accreting leading ABI args.)

---

## 2. Core design  [BUILD NOW]

### 2.1 The import manifest binds to the powerbox prefix — no rewriting

A module's `imports` list is a **manifest**: ordered, named, signature-carrying
declarations. Import `i` binds to **handle-table slot `i`** (the powerbox
prefix). Instantiation resolves each name through the host's policy (the
existing `name → binding` resolver shape), validates the implementation's
signature against the declared one (structural, fail-closed — no IDL), and
fills the slot. The module bytes are **never modified**. Verify once per
module; instantiate many.

```
import 0 "fs"     : Fs      required     ; unbound at instantiation ⇒ spawn fails
import 1 "stdout" : Stream  required
import 2 "log"    : Log     rebindable   ; starts empty; guest may attach later
```

Import modes:

- **`required`** — bound at instantiation, fail-closed (a missing binding
  refuses to start the module — the current `ImportError::Unresolved`
  semantics, kept). The slot is **immutable for the instance's lifetime**.
- **`rebindable`** — declared and typed, starts empty. The guest fills it at
  runtime with `import.attach` (§2.3). Calling through an empty slot traps.

There is deliberately **no `deferred`/`import.bind`** mode (a mid-run
guest→instantiator upcall channel); see §5.2.

### 2.2 One call convention, two addressing modes

`cap.call` disappears as a guest-facing concept. All capability invocation is
`call.import`, with the same static/dynamic split as `call`/`call_indirect`:

```
; static mode: slot is an immediate. Signature comes from the manifest.
; Verifier checks slot < n_imports and op/sig against the declaration, at load.
  v5 = call.import 0.read v1 v2

; dynamic mode: slot from a value, signature carried inline (like call_indirect's ty).
; Runtime check against the entry's type_id + generation — the §3c use-site check.
  v6 = call.import [v7].read (i64,i64)->(i64) v1 v2
```

Both modes dispatch through the same host-owned table with the same masked,
type_id- and generation-checked resolve (§3c) — the security hinge does not
move. The `cap.call` *encoding* remains as the dynamic mode's wire form (or is
kept as an alias during migration); what is retired is the concept of a
separate convention.

Properties:

- **Static mode is statically checked** — stronger than today, where `cap.call`
  carries a self-asserted `sig` and an unbounded `op`. The verifier checks the
  op index and signature against the canonical interface in the manifest (the
  §13-deferred "module-level interface section" check, landed for free).
- **Static-mode slots that are `required` are immutable-per-instance**, so JIT
  devirtualization of them is *always legal* — the "provably-stable,
  never-revoked handle" precondition (§3c) is now declared in the manifest
  instead of proven ad hoc. `rebindable` and dynamic-mode calls take the
  checked path.
- **Manifest-completeness is a verifier-computed bit.** A module with zero
  dynamic-mode call sites can only ever drive its declared interfaces: its
  egress closure is manifest-complete, statically. Tooling reports the bit;
  a host policy may require it for high-assurance slots. Modules that need
  open-world discovery (shells, plugin hosts, brokers) use dynamic mode and
  keep the ocap bound (egress limited by grants). The strong bound is a
  per-module property, not a global straitjacket.

### 2.3 Slots and handles interconvert; objects are arguments

- **`import.handle i` → `i32`** — reify slot `i` as the ordinary packed
  `(generation, slot)` handle value: store it, pass it as an argument, grant it
  to a child, use it in dynamic mode. This replaces every use of the
  `CapBound`/placeholder patching. Nearly free.
- **`import.attach i, vh`** — fill (or refill) `rebindable` slot `i` with the
  capability behind handle `vh`. Type-checked against the slot's declared
  interface, fail-closed. Authority-neutral: it aliases an entry the domain
  already holds into a named position — no new grant-graph edge (the D37
  argument). This is the "negotiate then use" pattern: reflect
  (`cap.self.count/get/resolve` — already built), decide, attach once, then
  uniform static-mode calls.
- **Objects are arguments, interfaces are dispatch.** The dominant pattern for
  runtime-minted objects (fds, regions, children, pipe ends) is already
  "dispatch on the interface capability, pass the object handle as an
  argument" (`Instantiator.join(child)`, the POSIX `HostFn` singleton).
  Handle-typed arguments keep the full §3c protection at the boundary.
  Dynamic mode covers the genuinely heterogeneous case.

### 2.4 Exports (now): unchanged

`export "name" <funcidx>` stays exactly as is — name-addressable entry points,
verifier-checked, ignored by backends. The provider-side export (`impl`
exports) is §3.2.

### 2.5 What gets deleted (end state)

- The numeric `("42","7")` module/name convention in `svm-wasm`.
- Handle threading as leading params of every transpiled function, the
  spawn-shim handle stash, and the powerbox window stash.
- `Resolved::CapBound` and `patch_placeholder` (`SlotHandleNotConst` dies).
- `resolve_imports` as an instantiation step. It survives, shrunk, **in the
  linker only**: `Resolved::Func`/`Slot` are link-time symbol resolution,
  which legitimately produces new module bytes (`link`, `compile_linked`).
  Instantiation never rewrites.
- Re-verification at instantiation (verify once, at load/install).

### 2.6 Security argument

- The escape hinge — masking lowering (§4) — is untouched.
- The authority hinge — the §3c masked, type_id + generation-checked table
  resolve — is untouched and is now the *single* path for both addressing
  modes.
- New verifier surface: a bounds-check of a slot immediate against the manifest
  and an op/sig check against a declared interface — smaller and more static
  than the instruction surgery it replaces. Fuzz alongside the existing
  verifier invariants.
- `import.attach` writes a table entry, but only by aliasing an entry the
  domain already holds, type-checked; a guest still cannot mint authority.
  Table writes remain host-mediated for everything else (grant, wiring,
  revocation).
- Authority conservation is unchanged and structural: a child's table is wired
  exclusively by its parent from things the parent holds; everything
  terminates in root-host grants (D19 generalized).

### 2.7 Migration plan

Phased, each phase a PR ≤ 1000 LOC, differential tests as the net:

| Phase | Content | Notes |
|---|---|---|
| 1 | Executable `call.import` (static mode) in verifier + tree-walker + bytecode + JIT; instantiation-time binding + sig validation; `import.handle`; spec vectors + fuzz | Additive; old paths keep working. `CallImport` already exists in IR/encode/text/verify (as fail-closed reject — flip the arm). Bytecode debug engines inherit via the shared per-op driver; add a `cap_stops` arm + one debug test. **Invariant: `jit_instantiate_cache`/`jit_lifecycle` cache-hit assertions keep passing.** |
| 2 | Dynamic mode (recast `cap.call` as it); `rebindable` + `import.attach`; manifest-completeness bit | |
| 3 | Migrate frontends: `svm-wasm` de-threading (the big one, ~73 sites), `svm-llvm`, chibicc, `svm-posix` off `CapBound` | Each independent |
| 4 | Deletions (§2.5); shrink `resolve_imports` into the linker; docs (`DESIGN.md` §3a/§7 deltas) | Net-negative LOC |

Pre-phase-1 audits: (a) `svm-wasmjit`/browser and DURABILITY for baked-in
"backends never see imports" assumptions; (b) powerbox-prefix slot ordering vs
the snapshot format's `DurableHandle` classification (slot order becomes ABI).

**The deletion phases are tracked work, not eventual cleanup.** The failure
mode of this migration is not a wrong design; it is stalling at phase 1 and
leaving the tree with five conventions instead of four.

---

## 3. Designed now, build on demand  [PROPOSED, consumer-gated]

### 3.1 Binding provenance in `cap.self.attest`

Extends `PROCESS.md` §6 (and shares its status and its sign-off requirements).
The handle table is host-owned, so the TCB knows, per entry, whether its
implementation is **platform-terminated** (host-native vtable) or
**ancestor-terminated** (a trampoline into a guest domain at depth *d*).
Attest — the one non-interposable namespace — additionally reports each
binding's provenance class (a per-slot `import.attest i` or a bulk report).

This is the *only* question the type system deliberately refuses to answer:
interface identity is structural (D59 intern: id-equality ≡ structural
equality), so a parent-implemented in-memory `Fs` is typewise indistinguishable
from the platform's — which is what makes interposition, testing, and
virtualization work. Provenance is therefore the load-bearing honest bit, and
it must live in the non-interposable namespace or it is worthless. Fits §6's
pinned growth criterion: a fact the platform mechanically enforces.

Trust model served (the asymmetry, stated once):

- The **user** (root host) can always fool any guest — the platform is theirs.
- An **untrusted guest cannot fool its nested children**: it can interpose
  every capability but cannot hide that interposition (provenance reads
  ancestor-terminated), cannot alter the child's attest (already pinned as a
  test requirement), and cannot forge window/freeze provenance.
- **User-trusted parents fooling nested guests** is the masquerade design,
  §5.1 — parked.

### 3.2 Provider-side exports (`impl` exports) and wiring

The symmetric half of the manifest: a domain that *implements* an interface.

```
export "main"   func 0                 ; entry point (unchanged)
export "logger" impl Log = func 7      ; func 7: the dispatch (op, args…) -> results
```

An `impl` export is an **offer** — declaring it confers nothing. Authority
moves only when a wiring party (the host, or a parent holding both ends)
connects it to an importer: the host mints, in the importer's table, an entry
whose vtable trampolines into the exporter's dispatch in the exporter's domain
— the same machinery as §22 Model A trampolines and pipe-end aliasing.
Signature check at wiring, structural, fail-closed. This makes "expose a
custom capability" a guest act (today it is host-Rust-only via
`grant_host_fn`), and is what interposition (§3.3) builds on. A binding wired
from an `impl` export is exactly the ancestor-terminated provenance class.

Interfaces are signatures; **implementing one requires zero authority**. A
parent can offer an `Fs` backed purely by its own window, regardless of
whether it holds any platform fs — the child's "fs" then consumes only the
parent's resources. Not a loophole; the point of the model.

### 3.3 Forwarding, wrapping, overriding — one act

A parent instantiating a child supplies `name → binding` for the child's
manifest (the existing `spawn_named_child` flow, given a manifest to bind
against). The four policies are one act with different right-hand sides:

| Policy | Binding | Per-op cost | Child's provenance view |
|---|---|---|---|
| Forward | alias the parent's own entry | zero (dispatches to the terminal implementation) | that of the original binding |
| Wrap | parent's `impl` export calling its real cap | one domain crossing (§14's stated price) | ancestor-terminated |
| Override | any other implementation | one crossing | ancestor-terminated |
| Withhold | nothing | — | `required` ⇒ spawn fails; `rebindable` ⇒ empty |

Child code is byte-identical under all four (the uniform convention is what
makes interposition transparent), and the compile cache keeps hitting.
Forwarding is zero-marginal-cost at any depth via table aliasing — the §14
"cheap steady state" claim, realized without handle-value threading.

### 3.4 Discovery (already built, restated for completeness)

Three tiers of guest access, all over one table:

1. **Declared-required** — manifest slots, fail-closed, immutable, fastest.
2. **Declared-rebindable** — reflect → `import.attach` → static-mode calls.
3. **Undeclared grants** — the parent may grant beyond the manifest;
   `cap.self.count/get/resolve` (+ D46 op-schema reflection) discover them;
   dynamic-mode calls drive them. Reflection stays authority-neutral: only
   what this domain already holds is visible.

The manifest bounds *requirements*, not *reach*. Reach is bounded by grants —
the correct ocap bound.

---

## 4. Interactions with settled decisions

- **D37/D46/D59, §3c**: unchanged and load-bearing throughout (forgeable-index
  inertness, reflection neutrality, structural type intern).
- **D19 / §14**: authority conservation generalized; the §14 "no am-I-nested
  query" amendment is inherited from PROCESS.md §6, not expanded here.
- **§22 / B2**: `call_indirect` pre-sizing untouched; `Resolved::Func`/`Slot`
  stay as the linker's business. The trampoline machinery is reused by §3.2,
  not duplicated.
- **PROCESS.md S1/O3**: this design is what makes the child compile cache's
  key stable under instantiation; a digest key (O3) becomes strictly better.
- **DESIGN.md §3a** ("all declarations precede all bodies"): preserved; the
  manifest is a declaration section.

---

## 5. Parked, with reasons  [PARKED]

### 5.1 Masquerade — user-authorized provenance forgery

Requirement: user-trusted parents may present ancestor-terminated bindings as
platform-terminated to their children (replay harnesses, test rigs,
deterministic simulation); untrusted parents must not.

Design (recorded so future pressure has a shape to argue against):

- **Masquerade is a stamp on the table entry, not a mode on a domain.** A
  blessed parent wiring a binding may stamp it `provenance = platform`. Attest
  *reads stamps*; the platform mechanically enforces who could stamp. Stamps
  **travel with entries** through forwarding/aliasing, so the lie is
  consistent under composition and reflection at any depth — a lie that is
  not closed under forwarding is detectable, hence not a lie.
- **The right to stamp is a capability**, attenuable on four axes:
  *interface set* × *fact set* (binding-provenance vs window/freeze — the
  latter is a categorically stronger lie) × *delegability* × *temporal scope*
  (spawn-time-only vs anytime). Parent-level ("wide") masquerade is simply
  the unattenuated top of the lattice — a deliberate grant, never
  constructible from narrow pieces. Narrow-by-default is host grant policy,
  not mechanism.
- **Default default: spawn-time-only.** Provenance that cannot change after
  first attest is what guests can reason about; re-stamping a live binding
  after the child has made trust decisions is the capability least worth
  granting.
- Framing that squares it with §6's growth criterion: a masquerade grant
  extends the *user's TCB* to include that parent for the stamped scope.
  Attest keeps reporting truthfully relative to the (enlarged) TCB.

Parked because: it is a TCB-membership grant (the riskiest object in this
document — a compromised blessed parent forges within its scope); it amends
the pinned "cannot alter the child's attest report" test; and stamp
propagation through forwarding, snapshot/restore (`DurableHandle`), and
re-granting has real interaction surface that deserves a consumer first. Ship
attest provenance (§3.1) honest-only first; deception is added deliberately,
later, narrow.

Honest limit: the platform controls *reports*, not behavior — a fake fs with
RAM latency is unmaskable via attest and maskable via a stopwatch. Behavioral
convincingness is the parent's problem; timing side channels remain the
host's (§6).

### 5.2 `import.bind` (mid-run binding upcalls) — cut

A `deferred` mode where the guest asks its instantiator to fill a slot mid-run
adds a new guest→host interaction channel whose unique value over
(rebindable + reflection + dynamic mode) is small. Prime directive: refuse
until a concrete consumer demands it. Adding it later is easy; removing it is
not.

### 5.3 Interposition metering — decision needed, no code

When a child calls a parent-implemented binding, the parent's dispatch runs —
on whose fuel/quota? §15's principle ("monitoring is reading the meters on
capabilities you granted") suggests the parent pays (its code, its choice to
interpose — which also prices interposition honestly), but that lets an
adversarial child drain a parent's budget by hammering a wrapped import. The
likely answer is: the crossing charges the *child's* quota, metered visibly to
the parent. To be decided explicitly when §3.2 is built, not inherited from
whatever the trampoline happens to do.

### 5.4 Runtime acquisition (`Resolver`) — unchanged, still host-layer

Open-ended lookup of *never-granted* capabilities stays outside the import
system: a granted registry capability returning ordinary handles (§7's
existing parked position). The import system's dynamic features (rebindable,
dynamic mode, reflection) cover discovery of *granted* capabilities only.

---

## 6. Open questions

1. Powerbox-prefix slot ordering vs the durable-snapshot handle format —
   confirm slot order can be ABI (pre-phase-1 audit).
2. Does anything in `svm-wasmjit`/browser assume import-free modules
   post-load? (pre-phase-1 audit)
3. Wire format for the manifest's interface declarations: reuse the inline
   `FuncType` list per import (status quo shape) vs an interned interface
   section (§13's deferred idea — phase 2 may want it for op-schema checks).
4. `import.attach` concurrency semantics under §12 threads (per-domain table
   already atomic per D59; specify attach's ordering guarantee).
5. Whether dynamic mode keeps the `cap.call` mnemonic on the wire for
   compatibility during migration, or renames at a format bump.
