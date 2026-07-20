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

**Slot-layout rules** (audit-verified, §7):

- **Root modules: `import i` = slot `i` is already the tree's behavior.** The
  name-bound path documents and implements exactly this — `NamedBinding` is
  "slot `i` of the powerbox stash ↔ import `i`" (svm-run lib.rs:3477-3483),
  `grant_caps` grants in import order on a fresh `Host` whose allocator is
  first-free-slot, so slots `0..N` are guaranteed. This design formalizes
  existing behavior rather than introducing new mechanism.
- **Child modules: a reserved prefix.** `spawn_granted_child` /
  `spawn_named_child` auto-grant `Instantiator` into slot 0 and
  `AddressSpace` into slot 1 before any named grants (svm-interp
  lib.rs:11294-11300, 11356-11367) — the same order the child entry already
  receives them as args. A child manifest therefore treats these as
  **implicit leading imports**: its named imports start at slot 2. This is
  documented ABI, checked at wiring; "import i = slot i" is never applied
  naively to child modules.
- **Mem-hooks instrumentation is exclusive with manifest binding.** The
  opt-in `with_mem_hooks` diagnostics path deliberately takes slot 0
  ("hooks first", svm-run lib.rs:3884-3886). Under a manifest it is refused
  (simplest; it is a diagnostics-only mode) — offsetting the manifest by the
  hook-grant count is the fallback if refusing proves too restrictive.

**The op selector — phase-1 scope decision** (audit finding, §7): today's
data model is **one operation per import entry** — `Import { name, sig }`
carries a single signature, and `Inst::CallImport` carries no `op` immediate
(svm-ir lib.rs:1723). Phase 1 keeps exactly that: an import names one
operation (`"fs.read"`, `"fs.open"`, … — the existing `default_cap_resolver`
name shape), `call.import i` invokes it, and instantiation records the
resolved `(type_id, op)` per slot as **per-instance binding state** beside
the handle entry (cold + per-instance — the §1 rule; the dispatch reads it
host-side). Interface-*grouped* imports (`import 0 "fs" : Fs` declaring an
op list, invoked as `call.import 0.read`) require an `op` immediate on the
static form plus manifest interface declarations — that lands with phase 2
alongside open question 3 (the interned interface section), not phase 1.
Examples elsewhere in this document use the interface-grouped surface as the
end state.

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

The headline conventions:

- The numeric `("42","7")` module/name convention in `svm-wasm`.
- Handle threading as leading params of every transpiled function, the
  spawn-shim handle stash, and the powerbox window stash.
- `Resolved::CapBound` and `patch_placeholder` (`SlotHandleNotConst` dies).
- `resolve_imports` as an instantiation step. It survives, shrunk, **in the
  linker only**: `Resolved::Func`/`Slot` are link-time symbol resolution,
  which legitimately produces new module bytes (`link`, `compile_linked`).
  Instantiation never rewrites.
- Re-verification at instantiation (verify once, at load/install).

And the secondary machinery those conventions carried (the full obviation
inventory — each entry exists only to work around the absence of
slot-addressed imports):

- **The synthesized bootstrap prologues**: the `synth_powerbox_start` /
  `synth_powerbox_start_for_imports` / `synth_powerbox_start_with_names`
  family (svm-ir lib.rs:2700-2741) — generated `_start` wrappers that stash
  positional handles or `cap.self.resolve` names at startup. A manifest
  module needs no synthesized prologue: its slots are bound before entry.
- **`powerbox_resolver` and `svm-posix::resolve_bound`** — the
  `CapBound`-producing resolver wrappers (the "general-form powerbox"
  S15 path). Superseded by slot binding in `grant_caps`.
- **The positional entry-args ABI** for capability delivery: the
  `slots: Vec<Value>` leading-`i32`-handle-arguments convention
  (`grant_powerbox_prefix` positional delivery, the fixed §3e 8-slot entry
  contract). Entry functions go back to taking only their real parameters.
  (The *child* positional-args ABI migrates to the reserved-prefix child
  manifest, §2.1.)
- **`Inst::CallImport`'s `handle` operand** becomes vestigial in static
  mode (the slot is the dispatch; no handle value is threaded). Retire the
  field at the next wire-format bump — until then frontends emit a dummy
  and backends ignore it (encode keeps round-tripping).
- **Test/doc surface asserting the old world**: the
  `resolved.imports.is_empty()` / imports-cleared assertions across
  `svm-posix`, `svm-run/tests`, `svm/tests/dynlink*`, `svm-text`'s
  `resolves_to_capcalls_and_clears_imports`, and the c_frontend/c_posix/
  c_shell/powerbox_* test families that pin the stash/positional/by-name
  bootstraps — flipped or deleted with their phase-3 frontend migrations;
  DESIGN.md §7/§3a text, POWERBOX.md F7/S15, FRONTEND.md entry-convention
  claims updated in phase 4.

Deliberately **kept** (not obviated): `cap.self.count/get/resolve` and
`register_cap_name` (discovery tier 3 and the name directory — §3.4);
`grant_host_fn`/`grant_host_fn_region` (host-side capability definition);
the handle table and every §3c use-site check (the mechanism everything
above now routes through).

**Completion gate for phase 4** — grep-clean checks, so "done" is checkable
rather than asserted: no non-linker caller of `resolve_imports*`; no
occurrence of `CapBound`, `patch_placeholder`, `SlotHandleNotConst`,
`synth_powerbox_start`, `powerbox_resolver`, `resolve_bound`,
`NAMED_IMPORT`, `handle_modules`, or `stash_base` outside the linker and
this document's history; `svm-wasm` emits no leading handle params.

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

### 2.7 Implementation plan

Phased, each phase 1–2 PRs ≤ 1000 LOC, differential tests as the net. The
pre-phase-1 audits have been **run** (findings in §7); the file-level items
below incorporate them.

**Phase 1 — executable `call.import` (static, one-op-per-import). Additive;
every legacy path keeps working.** *Status: **landed**. Implementation notes,
recorded where they refined the plan:*

- *Dispatch is the audit's option (a): a reserved `CAP_IMPORT_TYPE_ID`
  pseudo-type_id (the `CAP_SELF_TYPE_ID` precedent) carrying the import index
  as the op; `Host::cap_dispatch_slots` translates it through the
  instantiation-time binding table (`Host::set_import_bindings`). One shared
  implementation; all three backends (and the W1 record/replay tape, which
  sees the translated call) stay in lockstep. The bytecode engine needed no
  new op — `call.import` compiles to the existing generic `Op::CapCall` with
  the sentinel.*
- *Slot-bindable interfaces are allow-listed to the generic-dispatch set
  (`Stream`/`Exit`/`Clock`/`Memory`/`HostFn`); an import naming an
  executor-dispatch interface (`Instantiator`, guest-`Jit`, …) falls back to
  the legacy rewrite (those interfaces are special-cased in each backend's
  eval/compile loop — reaching them through the binding translation is
  phase-2 work).*
- *`instantiate_with_imports` (name↔handle 1:1 by construction) is the
  converted no-rewrite path. `instantiate()` (the fixed §3e preset) stays on
  the rewrite: its `Resolved::Cap` semantics let the call-site operand select
  the endpoint (stdout vs stdin on one `Stream` interface), which
  slot-binding deliberately does not model.*
- *The `decode_verify` fuzz target covers the new surface with no changes:
  the decoder already round-trips manifests, so fuzzer-built manifest
  modules exercise the new verifier arms, and a verified `call.import`
  reaching a bare host is a clean `CapFault` (fail-closed), never a panic.*

- `svm-ir`: **no changes required.** `Inst::CallImport` already carries
  `import`/`sig`/`handle`/`args` (lib.rs:1723); `Effects` already classifies
  it as a full clobber identical to `CapCall` (lib.rs:2339-2342).
- `svm-encode` / `svm-text`: **no changes required.** Both already round-trip
  `Module.imports` and `CallImport` with tests (encode lib.rs:322-331,
  506-518, 1558-1566, 1888-1896; text lib.rs:66-73, 393-398, 1141-1153,
  1841-1878, `imports_round_trip`).
- `svm-verify`: thread `imports: &[Import]` into `verify_func` (one-argument
  change mirroring the existing `&funcs` thread-through, lib.rs:142/161).
  Flip the reject arm (lib.rs:275) to: `import < imports.len()`,
  `sig == imports[import].sig`, then the arg-count/arg-type/result checks
  copied from the `CapCall` arm (lib.rs:245-271). Flip the two
  "unreachable" arms (lib.rs:446, 597). Add manifest validation to
  `verify_module` (which today never inspects `m.imports` at all): unique
  names, well-formed sigs.
- `svm-interp` tree-walker: a `CallImport` arm beside the generic `CapCall`
  arm (lib.rs:6914-6934) — resolve table slot `import` (a `resolve_slot`
  sibling of `resolve`, lib.rs:11232) + the per-slot `(type_id, op)` binding
  record, then the existing `cap_dispatch_slots` tail. Flip `eval_inst`'s
  `Trap::Malformed` arm (lib.rs:7782). One-line `cap_stops` arm
  (lib.rs:414-423) so the debugger stops on it like `CapCall`.
- `svm-interp` bytecode: `Op::CallImport` variant (ops are a Rust enum,
  bytecode.rs:66 — no opcode-space concern), a `compile_inst` arm mirroring
  the generic `Op::CapCall` lowering (bytecode.rs:1217, replacing the
  `return None` reject at :1321), one exec arm beside :7856. Debug engines
  (`DebugRun`/`ScheduledDebugRun`/`debug_advance_fiber`) inherit it through
  the shared `Vm` op driver — audited: they classify only scheduler-seam
  outcomes, not inline ops. One debug test pinning that.
- `svm-jit`: a lowering arm reusing `lower_cap_call`'s thunk call
  (lib.rs:6945-6977) with `(type_id, op)` from the instantiation binding as
  immediates and the handle resolved from slot `import`; remove `CallImport`
  from the support-gate catch-all (lib.rs:4567). Baking `(type_id, op)` is
  phase-1-correct: the child compile cache only caches **empty-powerbox**
  children today (lib.rs:4252-4273), so import-bearing modules are not the
  cached case and the `jit_instantiate_cache`/`jit_lifecycle` cache-hit
  assertions keep passing (audited). Cache sharing *for import-bearing
  modules* arrives when the binding table is threaded (phase 2+, the
  instance-context consolidation of §1).
- `svm-run`: bind in `Instance::grant_caps` (lib.rs:3884-3919) — it already
  iterates imports in declared order on a fresh Host; redirect from
  rewrite/positional-args to slot-filling + recording `(type_id, op)` per
  slot. `resolve_capability_imports` stays as the legacy path (its
  `imports.is_empty()` early-return already makes it a no-op for migrated
  callers). Refuse `with_mem_hooks` + manifest (slot-layout rule, §2.1).
- `svm-spec` + fuzz: vectors for the new verifier arms (valid/invalid import
  idx, sig mismatch, manifest dup names); extend verifier fuzzing to
  manifest-bearing modules.
- `svm-snapshot`/`svm-durable`: **no changes required** (audited: restore
  pins exact `(slot, generation)` via `grant_at`, DURABILITY.md §12.5
  declares slot stability as an invariant). Document the one non-guarantee:
  an empty `rebindable` slot's generation resets across restore (capture
  skips empty slots, lib.rs:10479) — slot ABI unaffected, only a stale
  packed handle value for a previously-attached-then-detached rebindable
  slot loses D37 protection across a freeze.

**Phase 2 — dynamic mode + rebindable + completeness bit.** *Status:
**landed** (except the two "optionally" items, which stay open). Implementation
notes:*

- *`rebindable` is a per-import mode (`ImportMode`, wire format **v4**: a mode
  byte per import entry; text: a `rebindable` suffix). `required` stays the
  default and is immutable-per-instance. A rebindable slot may start empty
  (`HostCap::template(type_id, op)` — declared interface, no grant; calling
  through it `CapFault`s) or start bound and be retargeted.*
- *`import.attach <slot> v<handle>` (opcode 0x63) rebinds a rebindable slot to
  a **held** capability. The new handle must resolve live under the slot's
  declared interface `type_id` (the §3c check) — attach swaps which *object*,
  never which *interface*. Wrong-type/dead handles return `-EINVAL`
  (probeable); structural misuse (out-of-range, non-rebindable target) is
  rejected **statically** by the verifier (`AttachNotRebindable`). Dispatched
  through a second reserved pseudo-type_id (`CAP_IMPORT_ATTACH_TYPE_ID`), so
  all three backends share one host implementation, like phase 1.*
- *`svm_verify::manifest_complete(m)` is the completeness bit: no `cap.call`
  anywhere ⇒ the manifest is the complete egress surface. Reflection does not
  affect the bit (discovery confers nothing without a dispatch).*
- ***OQ5 resolved:** `cap.call` keeps its mnemonic and wire form — it simply
  *is* the dynamic addressing mode. No rename, no migration churn.*
- ***OQ4 resolved:** attach is serialized by the Host lock like every
  capability dispatch; a concurrent `call.import` on the same slot observes
  the old or the new binding atomically (the binding is copied out under the
  lock — no torn read).*
- *Format v4 required updating the five guest-side C emitters in
  `demos/jit/*` (version byte + mode byte per import entry) — the "verified
  bytes are executed bytes" principle applies to guest-emitted blobs too.*

Still open from the phase-2 list: interface-grouped imports (`op` immediate +
interface declarations, open question 3) and the JIT instance-context
threading for import-bearing cache sharing.

**Phase 3 — frontends.** `svm-wasm` de-threading (~73 handle-threading
sites — the largest single item); `svm-llvm`; chibicc; `svm-posix` off
`CapBound`. `svm-wasmjit`: extend `outline_cap_calls` (lib.rs:1138-1181) to
also outline `CallImport` — its tierability classifier already lists it as a
host-boundary op (lib.rs:947-949) — and relax the `emit_module` import-free
assertion (lib.rs:1352-1353) to "no `CallImport` call-site survives in an
*emitted* function" (permit the manifest; capability dispatch already
bounces to the interpreter tier, which phase 1 made import-capable, so the
browser build inherits support).
`svm-posix`/`svm-run` child spawns adopt the reserved-prefix child manifest
rule (§2.1).

*Phase-3 status:*

- *`svm-wasmjit` — **landed.** `outline_cap_calls` outlines `CallImport`
  into the same cross-tier wrapper as `cap.call` (import index baked as an
  immediate); `emit_module` permits the manifest and rejects only an import
  op surviving in an emitted function (`tests/outline_callimport.rs`).*
- *`svm-wasm` — **landed.** Every wasm function import (numeric convention
  and §7 named alike) is one manifest entry `"<module>.<name>"`; a `call`
  lowers to `call.import <slot>` with a dummy handle operand. Deleted: the
  `NAMED_IMPORT` sentinel and the numeric-vs-named split, `handle_modules`,
  the leading-handle-params prefix on every function/block (and the splices
  at direct/indirect/tail call sites), the spawn-shim handle stash (`§12`
  spawn now needs only the tid counter — bindings are host state shared
  across vCPUs), and the start-wrapper handle threading. Embedders migrated
  to `Host::set_import_bindings` (svm-wasm's own differential tests,
  `svm-wasi` — its `bind` helper replaces `resolve_imports` + handle-arg —
  and the bench thunk/fast-resolver, which now map the
  `CAP_IMPORT_TYPE_ID` sentinel dispatch by arity).*
- *`svm-llvm` — **landed.** `_start` synths dropped the by-name resolve
  prologue and the `[0,32)` handle stash (paramless entry, zero prologue
  instructions); every call site's vestigial handle operand is a dummy
  const; `__vm_cap(i)` enumerates via `cap.self.get`;
  `__vm_blocking_handle` resolves its name at the call site (the one true
  handle-value read). The SharedRegion `map`/`unmap`/`page_size` builtins
  moved to **dynamic mode** (`cap.call` on the live region handle, §2.2) —
  only fixed-interface names remain manifest imports.*
- *chibicc — **landed.** Same shape: no resolve prologue, no stash,
  `dummy_handle()` operands, `__vm_cap` via `cap.self.get`, region ops in
  dynamic mode, `__vm_blocking_handle` by-name at the call site.*
- *Runtime/instantiation — **landed.** `svm_run::instantiate`, the CLI, and
  `run_powerbox` keep the manifest for a named powerbox entry and bind each
  slot in `grant_caps` (name → `(type_id, op)` via `default_cap_resolver`,
  handle by interface); the legacy rewrite remains only for positional
  entries (phase-4 deletion). The `cap_thunk`/`cap_thunk_locked` and the
  tree-walker's special `Jit` servicing translate the
  `CAP_IMPORT_TYPE_ID` sentinel through the binding table *before* their
  interface interception (shared macro bodies — one implementation).
  `svm-posix::bind` supersedes `resolve_bound` at its use sites. The
  browser's on-ramp binds manifest modules (`onramp_prepare` keeps the
  rewrite only for legacy positional blobs). **§2.1 child manifests:**
  `ModuleGrant` retains the import list; an op-13 spawn binds the child's
  slots against its granted powerbox on both backends
  (`Host::bind_child_manifest`, the interp inline + the JIT's
  `ChildManifestBinder` hook via `svm_run::child_bind_imports`).*

**Phase 4 — deletions**: the **full §2.5 inventory** — the five headline
conventions *and* the secondary machinery (the `synth_powerbox_start*`
family, `powerbox_resolver`/`resolve_bound`, the positional entry-args ABI,
the vestigial `CallImport` handle operand at the next format bump, the
imports-cleared test/doc assertions) — plus docs (`DESIGN.md` §3a/§7
deltas, POWERBOX.md F7/S15, FRONTEND.md entry conventions). Net-negative
LOC. **Exit criterion: the §2.5 grep-clean completion gate passes** — done
is checked, not asserted. *Status: **landed**, notes:*

- *The gate is a **test**, not a grep run by hand:
  `crates/svm/tests/imports_gate.rs` scans the tree's `.rs`/`.c`/`.h`
  sources and fails on any reappearance of the deleted symbols, and pins
  `resolve_imports_with` call sites to the linker allowlist.
  `patch_placeholder`/`SlotHandleNotConst` survive only as the
  `Resolved::Slot` rewrite's internals — the gate's "outside the linker"
  qualifier realized.*
- *`resolve_imports` (the Cap-only wrapper) went with `CapBound`; the
  linker pass is `resolve_imports_with` (`Cap`/`Func`/`Slot`). The
  `compile_linked` guest symbol table still delivers `Cap` — link-time by
  definition.*
- *`instantiate` / the CLI / `run_powerbox` / the browser on-ramp fail
  closed on an import-bearing module without the powerbox entry shape
  (paramless exported `_start`): instantiation never rewrites, there is no
  legacy fallback left. `instantiate_with_imports` rejects an import naming
  a non-slot-dispatchable interface with a pointer at dynamic mode (§2.2).*
- *The c_shell/c_shell_exec/stage1_posix_spawn suites link their compiled-C
  shims with `Resolved::Cap` (handle-free, link-time symbol resolution) and
  the guests discover their own handles via `cap.self` reflection — §2.3's
  dynamic mode for the `Instantiator` ops, exercised end to end.*
- *The vestigial `CallImport` handle operand is **kept** until the next
  wire-format bump, exactly as §2.5 schedules it (frontends emit a dummy,
  backends ignore it).*

**The deletion phase is tracked work, not eventual cleanup.** The failure
mode of this migration is not a wrong design; it is stalling at phase 1 and
leaving the tree with five conventions instead of four.

---

## 3. Designed now, build on demand  [§3.1/§3.3 landed; §3.2 v1 landed (exporter-domain state pending); §3.4 built]

### 3.1 Binding provenance in `cap.self.attest`

*Status: **landed** (2026-07-20) as `cap.self.provenance(handle) -> i32` — self-namespace
op 5, reached via dynamic dispatch on the reserved `cap.self` id (no new instruction, no
wire change): `0` = platform-terminated, `d ≥ 1` = ancestor-terminated `d` domain
boundaries up (1 where the offer was wired, +1 per §3.3 re-grant hop). A forged/closed
handle is an inert `CapFault`. PROCESS.md §6's growth-criterion list updated to name it.*

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

*Status: **v1 landed** (2026-07-20), with two recorded amendments and a scoped follow-up —
see the "as built" note at the end of this section.*

The symmetric half of the manifest: a domain that *implements* an interface.

```
export "main"   func 0                 ; entry point (unchanged)
export "logger" impl 7 9               ; one funcidx PER OP (amendment: was one dispatch func)
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

**As built (v1, 2026-07-20).** Two amendments to the sketch above, and one scoped follow-up:

1. **One func per op, not one dispatch func.** Guest functions have fixed signatures, so a
   single `(op, args…)` dispatch func cannot be typed for an interface with heterogeneous op
   arities without a padded-i64 marshaling convention that would blind the verifier. An offer
   instead lists one funcidx per op (`svm_ir::ImplExport { name, ops }`, text
   `export "logger" impl 7 9`, wire v5) and **op `i`'s signature IS `funcs[ops[i]]`'s declared
   type** — derived, never asserted, so the verifier and the wiring check are exact. There is
   no nominal interface name (`Log` above was illustrative): interface identity is the
   structural op-signature list, interned per-host to a `type_id`
   (`Host::intern_interface`, id-equality ≡ structural equality — D59 applied to capability
   interfaces; the OQ3 interned-section wire format remains open, this intern is host-side).
2. **v1 executes a wired op as a *pure dispatch*, not in the exporter's domain.** Wiring
   (`Host::wire_impl` + `bound_import_for_impl`, surfaced as
   `svm_run::HostCap::impl_offer`) is exactly as designed: authority moves only at the wiring
   act, signature-checked structurally, fail-closed, and the binding is non-durable. But
   execution runs the op's function as a fresh reference run over the offer's function table —
   **no window, an empty powerbox, a fixed deterministic fuel budget** — inside the one
   generic dispatch all three backends share. The impl computes over its arguments alone
   (implementing an interface requires zero authority; v1 implements one *with* zero
   authority). That covers adapters, validators, policy checks, and test fakes, and keeps the
   three tiers in differential lockstep with a single implementation.
3. **Follow-up: exporter-domain state.** The stateful headline ("an `Fs` backed by the
   parent's own window") needs the op to run over the *exporter's* window and powerbox —
   cross-domain window views, host references, and lock ordering that v1 deliberately does
   not touch. It builds on the same `GuestImplEntry`/binding shape (add the domain reference,
   thread caller fuel) without changing the wire format or the wiring API.

### 3.3 Forwarding, wrapping, overriding — one act

*Status: **landed** (2026-07-20). A wired offer is re-grantable into a §14 child
(`regrant_into_child` adopts the entry under the child's interned id, one provenance hop
deeper), and `bind_child_manifest` binds, per slot: a **named offer grant** first (its
first signature-matching op — structural, fail-closed, a name match with no signature
match never silently binds), then the reference policy, then **withhold** — a
`rebindable` slot starts empty, a `required` slot fails the whole spawn closed
(probeable `-EINVAL` before any child code runs), on both the interpreter's inline spawn
and the JIT's child builders. Forwarding of platform caps is the existing re-grant
(zero-marginal-cost aliasing); a forwarded offer keeps its entry and gains a depth hop.*

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

1. ~~Powerbox-prefix slot ordering vs the durable-snapshot handle format~~ —
   **RESOLVED (audit, §7): slot order is already stable ABI.** Restore
   re-grants via `grant_at(slot, generation, …)` into the exact captured
   slot; DURABILITY.md §12.5 pins "reinstate the same `(slot, generation)`"
   as a hard invariant; the snapshot roundtrip test proves guest-held packed
   handle values survive restore.
2. ~~Does anything in `svm-wasmjit`/browser assume import-free modules
   post-load?~~ — **RESOLVED (audit, §7): yes, one site** — `emit_module`
   hard-rejects non-empty imports (svm-wasmjit lib.rs:1352-1353). Fix scoped
   in phase 3; the browser adds no independent assumption (its `env.*` wasm
   imports are unrelated to SVM capability imports, and capability dispatch
   bounces to the interpreter tier).
3. Wire format for the manifest's interface declarations: reuse the inline
   `FuncType` list per import (status quo shape) vs an interned interface
   section (§13's deferred idea). Phase 2 wants this for interface-grouped
   imports (the `op`-immediate form, §2.1) and op-schema checks.
4. `import.attach` concurrency semantics under §12 threads (the table is
   `Arc<Mutex<Host>>` shared across vCPUs — audited — so attach is
   serialized; specify the ordering guarantee observed by concurrent
   `call.import` on the same slot).
5. Whether dynamic mode keeps the `cap.call` mnemonic on the wire for
   compatibility during migration, or renames at a format bump.
6. Snapshot digest wiring once the rewrite is removed: freeze/restore must be
   handed the same manifest-carrying module on both sides (the existing
   digest gate enforces this; a doc/wiring note, not a codec change — the
   digest becomes per-module instead of per-instantiation, a strict
   improvement).

---

## 7. Tree-audit record (2026-07-19)

Four parallel audits against the tree at `0c1a7c4` (post-#392/#398) verified
the §2 design before the implementation plan above was finalized. Verdict:
**no architectural obstacle; small, mechanical adaptations only.** Key
findings, with the §2 deltas they forced:

| Track | Verdict | Load-bearing findings |
|---|---|---|
| Verifier / encode / text / load pipeline | Works; small adaptations | `verify_func` lacks `imports` in scope — one-arg thread-through (mirrors `&funcs`). Encode + text already round-trip manifests and `call.import`, tested. `verify_module` never inspects `m.imports` today (manifest itself unchecked — phase 1 adds validation). `grant_caps` already grants in import order = the natural binding hook. |
| Runtime dispatch (interp, bytecode, JIT, wasmjit, opt/peval) | Thin adapters everywhere | Effects model + svm-opt already treat `CallImport` identically to `CapCall`; svm-peval is effects-generic. Bytecode debug engines inherit new ops via the shared `Vm` driver. JIT: bake `(type_id,op)` phase-1 (cache unaffected — only empty-powerbox children are cached today); thread the binding for cache sharing later. **Design gap caught: `CallImport` has no `op` immediate → phase 1 is one-op-per-import (§2.1).** |
| Handle table / prefix | Works; one real landmine, fixed in §2.1 | Root path already implements "import i = slot i" (`NamedBinding`). The DESIGN.md "first handle 0 vs 1" divergence is the *fiber* registry (separate table, already unified, D57) — false alarm for imports. **Real landmine: child auto-grants occupy slots 0/1 → reserved-prefix child manifest rule (§2.1).** `with_mem_hooks` steals slot 0 → exclusive with manifests (§2.1). Table is `Arc<Mutex<Host>>` across vCPUs — slot reads thread-safe. |
| Durability / snapshots | Works as-is | Slot indices already stable ABI (`grant_at` pins `(slot, generation)`; DURABILITY.md §12.5 invariant; roundtrip test). Digest currently over post-rewrite bytes — removing the rewrite makes it per-module (improvement). Non-guarantee documented: empty `rebindable` slots reset generation across restore. |
