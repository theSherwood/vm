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

## 3. Designed now, build on demand  [§3.1/§3.3 landed; §3.2 landed; §3.4 built; §3.5 designed — next build slice; §3.6 designed — the end-state execution model, awaits the shell personality]

### 3.1 Binding provenance in `cap.self.attest`

*Status: **landed** (2026-07-20) as `cap.self.provenance(handle) -> i32` — self-namespace
op 5, reached via dynamic dispatch on the reserved `cap.self` id (no new instruction, no
wire change): `0` = platform-terminated, `d ≥ 1` = ancestor-terminated `d` domain
boundaries up (1 where the offer was wired, +1 per §3.3 re-grant hop). A forged/closed
handle is an inert `CapFault`. PROCESS.md §6's growth-criterion list updated to name it.
`manifest_complete()` exempts `cap.call`s on the reserved self-namespace immediate:
completeness measures capability **egress**, and the self namespace is authority-neutral
reflection — a manifest-complete module may query provenance without losing the bit.*

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
   interfaces). *v6 addendum:* the OQ3 **type section** has since landed — a module
   declares its shapes in one index space (`type (sig)` entries + `interface { idx, ... }`
   tuples over them) and each offer names the interface entry it implements
   (`export "n" impl <iface> : <funcidx>...`), verifier-checked exactly; the host-side intern
   remains the runtime identity.
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
3. **Exporter-domain state: landed as *provider instances* (v2, 2026-07-20).** An offer may
   be wired **instanced** (`Host::wire_impl_instance` / `svm_run::HostCap::impl_service`):
   it gets a persistent *provider domain* — a window seeded once from the provider module's
   memory declaration + data segments, plus its own initially-empty powerbox — and every op
   dispatch runs over that state, so it survives across calls (the stateful headline: an
   in-memory `Fs` backed by the provider's own window works). The wirer re-grants real
   capabilities *into* the provider (`Host::grant_impl_cap`, same policy as §14 child
   re-grants) — how a wrap holds the real cap it forwards. Re-granting an instanced offer to
   a child aliases the **same** instance (like a pipe's shared backing), so parent and
   children drive one service. Deadlock-freedom is by construction: a provider can never
   hold an offer, so provider chains are acyclic and the lock order is always
   domain-host → provider. The provider is a *passive service domain* animated only by
   dispatch — there is deliberately no re-entry into live running domains. Remaining
   limits, recorded: per-op fuel is still the fixed deterministic budget (caller-fuel
   threading needs the dispatch-ABI fuel plumbing), and providers hold no offers.

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

### 3.5 Interface-grouped imports  [DESIGNED 2026-07-21 — next build slice]

The type section's recorded next consumer (OQ3), now designed: one slot binds a
whole interface, the op moves to the call site, and the flat one-op import is
the degenerate case of the same mechanism.

**Surface** (consumer side; syntax settled 2026-07-21):

```
type 0 func (i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 interface { read: 0, len: 1 }   ; op names required

import 0 interface "env" "fs" 2 required   ; grouped: one slot, the whole requirement
import 1 func "env" "log" 1 rebindable     ; flat: a singleton requirement

func 0 () -> (i64) {
  block 0 {
    v1 = i64.const 4096
    v2 = i64.const 64
    v3 = call.import 0.read (v1, v2)   ; by name — resolved to the op index at parse
    v4 = call.import 0.1 (v1)          ; or positional — identical wire form
    v5 = call.import [v6] interface 2 . read (v1, v2)  ; dynamic: handle from a
    return v5                          ; value, requirement by type-section ref
  }
}
```

**Text-format rules (settled with the surface):**

- **One declaration grammar:** `keyword index [kind] [names] payload`. Every
  definition — `type`, `import`, `func`, `export`, `block` — carries its
  index as a *checked positional label* (the parser verifies label =
  position; the wire stays positional — pure legibility: greppable
  references, stable diffs, precise errors). Exactly two kind keywords,
  **`func`** and **`interface`**, used identically in `type`, `import`, and
  `export`; the spellings `iface` and `impl` are retired (Rust follows:
  `ImplExport.iface` → `interface`; the `iface::` constants module renames in
  a follow-up).
- **One grouping construct:** `{ }` — function bodies, blocks, interface
  declarations, offer maps; indentation is never significant. One map syntax,
  `{ name: idx, … }` (in `type … interface` the values are type indices; in
  `export … interface` they are func indices — the kind keyword up front says
  which). One signature syntax, `(params) -> (results)`, appearing only in
  `type … func` entries; everything else references a type index.
- **Import names are two-level**: `("env", "fs")` — the first level names the
  provider binding point, the second the bundle (grouped) or op (flat).
  Resolver vocabulary.
- **Index spaces stay per-section** (types / imports / funcs / exports).
  Import index = handle-table slot is ABI (plus the reserved child prefix);
  folding types into the import numbering would put a "subtract the
  definitions above me" step into the slot path for zero semantic gain.
- **There is deliberately no `import type`.** Under structural identity a type
  import is redundant — declaring the shape locally *is* having the type
  (same interned id, D59) — and load-time verification needs concrete shapes.
  Cross-suite shape dedup is the linker's job (it already merges type
  sections, §2.5); the abstract-type use case is served by capability handles.

**Coverage binding — the binding relation (settled 2026-07-21).** A consumer's
interface declaration is a **requirement set**, not a claimed identity: "ops
with these names and these signatures." Binding succeeds iff the provider
**covers** it — every consumer `(name, sig)` has a same-named,
signature-equal op in the provider's interface; extra provider ops are
ignored; a missing name or a signature mismatch fails closed. At bind time
the host computes a per-slot **op remap** (consumer-local index → provider
index) and freezes it; call sites use the consumer's own numbering.

- **Forward compatibility, both directions:** providers evolve by *adding*
  ops without breaking any consumer; consumers declare only what they use and
  tolerate any covering provider. Coverage is name-keyed, so declaration
  *order* never matters for binding (it only assigns consumer-local indices).
- **Names are the binding contract; `type_id` stays shape-only** — the ELF
  split: link by symbol name, run by address. Matching must be by name
  because signature-only subset matching is ambiguous the moment two ops
  share a signature (`read`/`write`). The price, stated plainly: renaming an
  op is a breaking provider change, like renaming an exported symbol. An
  interposer declares the same names as what it wraps, so interposition
  invisibility is unaffected.
- **A flat `func` import is a singleton requirement set** — wasm's per-op
  import is the degenerate case of the same mechanism, not a competing mode.
  Grouped adds what wasm cannot express: one slot, one fail-closed act, one
  attach, one provenance answer, one revocation point for a coherent bundle.
- **Exact matching is the trivial case** (coverage happens to be total); no
  second mechanism exists.
- The coverage walk runs only at **binding acts** — instantiation,
  `import.attach`, child-manifest binding, wiring — once per binding, off the
  hot path. `required` slots stay devirtualizable (the remap is an instance
  constant).

**Semantics:**

- A grouped slot's binding state is `(type_id, handle, remap)`. The static
  form's verifier check resolves `import 0` → `types[2]` → op → signature,
  all at load; execution passes the remapped op through the existing
  `cap_dispatch_slots(CAP_IMPORT_TYPE_ID, slot, op, …)` path (the `op`
  argument exists today and is always `0` — it starts carrying information).
- The dynamic form (`call.import [v] interface k . op`) keeps the **exact-id
  fast path**: `entry.type_id == intern(types[k])` plus generation at the use
  site — the same §3c check, now *expressible for interned interfaces*. This
  closes the recorded encoding gap: `cap.call` needs a compile-time `type_id`
  immediate, which a guest cannot know for a wired offer; a type-section
  reference is resolvable at instantiation. To drive a
  covering-but-not-equal capability discovered at runtime, attach it to a
  rebindable slot — the coverage walk happens once, there — then use static
  calls. `cap.call` stays as the escape hatch for undeclared grants and the
  reserved self namespace.
- **Intern pre-seeding.** Built-in interfaces publish canonical shapes,
  pre-seeded into the per-host intern, so a structurally equal guest
  declaration interns *to the built-in id* (D59 extended across the
  host-native/guest-impl divide). `HOST_FN` is the deliberate exception — its
  semantics are per-registration embedder code with no canonical shape; it
  binds by name through the registry, the (trusted) embedder asserting the
  shape it implements.
- **Reflection gains two authority-neutral ops:** `cap.self.type_id k` —
  intern *this module's* `types[k]`, return the runtime id (exact-shape
  discovery: iterate `cap.self.get`, compare ids) — and `cap.self.covers
  vh, k` — "does the capability behind this handle cover my `types[k]`?"
  (subset discovery; a failed `import.attach` works as the probe even
  without it). `import.attach` on a grouped slot runs the coverage walk in
  one act. `resolve` (by name) is unchanged; D46 schema reflection returns
  op names alongside signatures (both are stored for pre-seeded and guest
  ids alike).
- **Child manifests** (§3.3): a grouped import matches a named grant by one
  name-keyed coverage walk — `bind_child_manifest`'s per-op signature probe
  becomes exactly that walk, computing the remap as it goes.

**Offer exposure — who may wire (settled 2026-07-21).** Declaring an offer
confers nothing (§3.2); *wirable* offers are capabilities, and the initial
holder is the domain that implements them:

- **`export.handle k` → `i32`** (mirror of `import.handle i`): the exporting
  domain reifies its own export entry `k` as an ordinary capability handle.
  Wiring rights propagate only by granting that handle: down to its own
  children at spawn (bind their manifests with it), up to its parent or out
  to a sibling *iff it chooses to send it*.
- **One rule, not two export semantics: bytes are ambient, instances are
  consensual.** Every export — `func` or `interface` — is an inert
  declaration in the module *bytes*, visible to whoever holds the bytes.
  Invoking a `func` export (a spawn entry, the embedder's `call("main")`)
  creates or enters *the caller's own instantiation* — caller's window,
  caller's bindings, caller's fuel; no pre-existing state of any exporting
  domain is involved, because none exists. Likewise, anyone holding the
  bytes (the embedder registry via `wire_impl*`/`impl_service`, or a parent
  holding a `Module` grant) may wire an offer as **its own** provider
  instance — your *code*, but the wirer's authority, seed state, and fuel;
  harmless to the authoring domain. What consent guards is *this domain's*
  instance — the one backed by its aliased imports and accumulated service
  state — and `export.handle` is the only path to it. No guest-reachable op
  harvests it: a parent holding a child's `Instantiator` cannot
  enumerate-and-wire the child's *live* offers. A hostile parent can always
  run your code; it can never reach into your service.
- **Honest limit:** this controls who can *call through* your offers. In
  window-exposed tiers a §14 parent already reads and writes the child's
  entire carve — export protection cannot create confidentiality the memory
  model doesn't provide. That is `attest`'s job: a child that finds itself
  window-exposed to an untrusted parent should refuse to hold secrets at
  all; distrust means separate processes (§1a). The two compose: attest
  tells you which world you are in; `export.handle` keeps offer wiring
  consent-based in the worlds where isolation is real.
- **The `cap` value type — boundary translation (settled 2026-07-21).** A raw
  `i32` handle crossing domains is deliberately inert (it would index the
  *receiver's* table — the forgeability guarantee). Authority crosses only
  where a signature says so: a parameter or result declared `cap` makes the
  host translate at the capability-call / entry-result boundary — resolve in
  the sender's table (must be live), re-grant into the receiver's, substitute
  the receiver-local packed handle. This is the guest↔guest half of §2.3's
  "objects are arguments" (today only trusted built-ins like
  `Instantiator.join` interpret arguments as handles); unmarked integers keep
  the existing inertness. In guest code `cap` is `i32`-width data; only
  boundaries treat it specially.
- **What offer calls run over — as built (v2) vs the end state (§3.6).** As
  built, offers execute over a **passive provider instance**: a second
  window + powerbox distinct from the live run — own lock, provider-pays
  fuel, shared by all offers the domain reifies, its import slots aliasing
  the reifier's bindings as of reification, deadlock-free by construction
  (providers never hold offers; lock order domain → provider). This bought
  exporter-domain execution with no fiber machinery — **implementation
  sequencing, not principle**. The end state is §3.6's **unified model**:
  one world per domain, offers served as handler fibers over the *same*
  window and powerbox as `main`; the separate-instance concept dissolves (a
  provider wired from bytes is simply a domain whose `main` never ran). The
  passive instance approximates the unified model exactly for the common
  provider — a domain with no concurrently active live run — and diverges
  for a domain that serves *while* running; the divergence is documented
  below and is the motivation for unifying.

**The two-world divergence, concretely (interim — dissolves under §3.6).**
The tempting-but-natural version, which the as-built split silently breaks:

```
func 0 () -> (i64) {                ; stats.get — an offer impl
  block 0 { v0 = i64.load ...TICK... return v0 }  ; reads the INSTANCE window
}
func 2 () -> () {                   ; the live loop
  block 0 { ... i64.store TICK, t ... }           ; writes the LIVE window
}
; nothing faults — `get` just reports 0 forever. Two worlds.
```

The workarounds, in preference order:

1. **Be a client of your own service.** The domain attaches its own
   `export.handle` to a rebindable slot and *calls its own offer* to write
   (`self_stats.bump()` per tick) — legal, deadlock-free (lock order is
   domain → provider; the live run is just another caller), and the instance
   becomes the one shared cell both worlds see through the same door. Cost:
   one dispatch per write.
2. **A shared stream/pipe** both sides hold — for flowing data and event
   patterns.
3. **Reify-time seeding** (data segments into the instance) — for config and
   constants.
4. **Split-phase** for request/response *with* the live run: the offer op
   records the request in instance state (or takes a `cap` callback) and
   returns "pending"; the live loop polls and responds. An offer op must
   never block waiting on the live domain — that recreates the exact cycle
   this design removes.

What remains inexpressible even with workarounds: an offer op that must
**block on the live run** (an interposed stdin `read` — split-phase requires
the *caller* to cooperate, and a posix child calling `read(0, …)` expects to
block; transparent interposition cannot demand caller changes), zero-copy
sharing of live-window buffers with the service, and guest services layered
on guest services (the offers-in-providers refusal). These are lifted by
**reactor domains** — §3.6, designed for exactly this; anything shell-like
is its consumer.

**Wire (v7 — one bump, six riders):** `Import` replaces its inline `FuncType`
with `shape: Func(typeidx) | Interface(typeidx)` and gains the second name
level; `TypeEntry::Interface` elements become `(name, typeidx)` pairs (names
required); `CallImport` gains an `op` immediate and **drops the vestigial
handle operand** (whose retirement was already scheduled for "the next wire
bump" — this is it); dynamic mode gains the type-section-reference form;
`export.handle` joins `import.handle`; **`cap` joins the value types**
(`i32`-width in guest code; special only at boundaries). Backend cost is the
op immediate plus one remap load threading through the one generic dispatch;
JIT devirtualization of `required` grouped slots stays legal per §2.2
(immutable binding, bind-time-constant remap).

**Host-side provider, Rust** (embedder registry — sketch):

```rust
let fs_shape = IfaceShape::new()      // op names required, mirroring the text form
    .op("open", sig_open())
    .op("read", sig_read())
    .op("write", sig_write())
    .op("len", sig_len());
Imports::new()
    // host-native: the (trusted) embedder asserts the shape it implements
    .provide("fs", HostCap::iface(&fs_shape, fs_handle))
    // or a guest offer as the provider — shape and names come from the offer
    .provide("fs", HostCap::impl_service(&fs_module, "fs")?)
```

Instantiation runs the coverage walk: every `(name, sig)` the consumer's
declared entry requires must appear in the provided shape — the same
fail-closed check as `wire_impl`, applied at the registry boundary. A
consumer requiring only `{ read, len }` binds against this four-op provider.

**Parent domain with a nested child, svm-ir** (the §3.3 wrap, grouped —
subset consumer):

```
; ---- parent (the provider; here also an interposer) ----
type 0 func (i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 interface { log: 0, flush: 1 }

func 0 (i64, i64) -> (i64) {
  block 0 (v0: i64, v1: i64) {
    ; record, then forward
  }
}
func 1 (i64) -> (i64) { ... }

export 0 interface "log" 2 { log: 0, flush: 1 }
; parent main: reify `export.handle 0`, grant it under the child's import
; name; manifest binding accepts it (wrap/override); an aliased own-binding
; forwards; no grant under the name + `required` ⇒ the spawn fails closed.

; ---- child (the consumer — requires only the op it uses) ----
type 0 func (i64, i64) -> (i64)
type 1 interface { log: 0 }

import 0 interface "parent" "log" 1 required   ; slot 2 at runtime (child prefix)

func 0 (i64, i64) -> (i64) {
  block 0 (v0: i64, v1: i64) {
    v2 = call.import 0.log (v0, v1)    ; by name — or positionally, `0.0`
    return v2
  }
}
```

The child's one-op requirement binds against the parent's two-op offer —
coverage, remap `[0→0]`. The offer flows the other way just as well: a child
reifying its own offer and returning the handle *up* through a `cap`-typed
result is how a child chooses to serve its parent — exposure is always
module→holder, authority flow is per-wiring, consent-based via
`export.handle`. Provenance reports the wrapped binding ancestor-terminated
at depth 1: the child can see *that* it is interposed, never *what* the
interposer does.

**The full triangle** (worked example: P above, M in the middle, C below —
one passthrough down, one wrap down, a *different* offer up):

```
; ---- P: M's parent ----
type 0 func () -> (i64)
type 1 interface { count: 0 }                 ; what P wants FROM M
import 2 interface "m" "metrics" 1 rebindable ; starts EMPTY — filled only if M delivers
; (P also holds a stream "out" and an Instantiator "kids"; decls elided)

  v0 = call.import kids.spawn (...)   ; spawn M; §3.3: forward own "out" to M
  v1 = call.import kids.join (v0)     ; M's entry result is `cap` -> host-translated
  import.attach 2 v1                  ; coverage-checked against P's interface 1
  v2 = call.import 2.count ()         ; P drives M's metrics

; ---- M: the middle domain ----
type 0 func (i64, i64) -> (i64)
type 1 interface { write: 0 }             ; consumes (from P)
type 2 interface { log: 0 }               ; offers DOWN (wrap of write)
type 3 func () -> (i64)
type 4 interface { count: 3 }             ; offers UP (a different capability)

import 0 interface "p" "out"  1 required
import 1 interface "p" "kids" ... required

func 0 (i64, i64) -> (i64) {              ; log impl: bump counter, forward
  block 0 (v0: i64, v1: i64) {
    ; ...increment counter in M's provider-instance memory...
    v2 = call.import 0.write (v0, v1)     ; instance slot aliases M's "out"
    return v2
  }
}
func 1 () -> (i64) { block 0 { ... } }    ; count impl: read the counter

export 0 interface "log"     2 { log: 0 }
export 1 interface "metrics" 4 { count: 1 }
export 2 func "main" 2                    ; M's entry — what spawn runs

func 2 () -> (cap) {                      ; the setup run
  block 0 {
    v0 = import.handle 0                  ; the ORIGINAL stream    (passthrough)
    v1 = export.handle 0                  ; M's log wrap           (for C)
    v2 = export.handle 1                  ; M's metrics            (for P)
    v3 = call.import 1.spawn (...)        ; spawn C
    v4 = call.import 1.grant (v3, v0)     ; C's "out" <- passthrough
    v5 = call.import 1.grant (v3, v1)     ; C's "log" <- the wrap
    return v2                             ; metrics goes UP — only because M returns it
  }
}

; ---- C: M's child ----
type 0 func (i64, i64) -> (i64)
type 1 interface { write: 0 }
type 2 interface { log: 0 }
import 0 interface "m" "out" 1 required   ; passthrough: provenance = the original's
import 1 interface "m" "log" 2 required   ; wrap: provenance = ancestor depth 1

  v2 = call.import 0.write (v0, v1)       ; straight to the original stream
  v3 = call.import 1.log (v0, v1)         ; through M's counter, then the stream
```

Everything load-bearing is visible here: `metrics` reaches P *only* because
M's entry returns it through a `cap`-typed result (an unmarked `i64` would
arrive as an inert number); `log` never flows up, so P has nothing to bind;
C's two imports answer provenance differently (the passthrough keeps the
original's answer; the wrap reports depth 1); and after `main` returns, M's
live run is over but its provider instance keeps serving both C's `log`
calls and P's `count` calls — the counter they share lives in the instance,
not in the finished run.

### 3.6 The unified execution model — one world per domain  [DESIGNED 2026-07-21; subsumes the two-world split when the fiber slice lands]

The end state, settled after review: **a domain is a program over one
world** — one window, one powerbox. `main` runs; if the domain has reified
offers, dispatches run as handler fibers over that *same* world — while
`main` computes, after it returns (the domain persists as a service for as
long as anyone holds its handles), or never (a provider wired from bytes is
a domain whose `main` never ran — setup is data segments, or a designated
init export). There is no second state: what `main` writes, handlers read.
A domain that *wants* isolated service state says so explicitly — spawn a
child and hand out the child's offer. §3.5's passive instance is the landed
interim implementation of exactly the degenerate case (no concurrently
active live run) and dissolves as a concept when this lands; `export.handle`
has one semantics, no mode flag.

What this makes expressible (fatal gaps of the two-world split, the first
fatal to anything shell-like): an offer op that **blocks on live activity**
(an interposed stdin `read` — a posix child calling `read(0, …)` expects to
block, and transparent interposition cannot demand caller-side cooperation,
so split-phase is a non-starter); zero-copy service access to the domain's
buffers; and guest services layered on guest services.

- **Scheduling — explicit service points, per-vCPU, host-run.** The
  scheduler lives host-side (the same machinery that already parks and
  wakes vCPUs at blocking ops — no new TCB category), but *when handlers
  may run* is guest-controlled. Dispatches **queue** (bounded, fail-closed:
  a full queue is a probeable fault at the caller). `svc.wait` and
  `svc.poll` **park the calling fiber** — the vCPU idles only if it has no
  other runnable fiber, ordinary §12 multiplexing. Handler fibers execute
  *only inside `svc.*` windows*, on the vCPU that opened them: a fresh
  dispatch (or a previously parked handler that has since been notified)
  runs to completion or to its next park; the window closes and control
  returns to the loop (`svc.poll`: run all runnable, return; `svc.wait`:
  park until something is runnable, run it, return). A domain's *own*
  parks — `memory.wait`, a blocking platform read — are **not** service
  points: blocking for your own reasons never invites reentrancy, and
  handler interleaving is *greppable* — exactly the `svc.*` lines.
- **One loop or a pool — the guest chooses.** One vCPU in
  `loop { svc.wait(…); own work }` is the classic sequential event loop
  (the shell shape): handlers never overlap, no locks needed, `svc.wait`
  doubling as the multiplex point (park until a dispatch *or* a watched
  capability is ready — waitset detail pinned when the slice is built).
  N vCPUs sitting in `svc.wait` are a worker pool: handlers run in true
  parallel and the guest synchronizes with the same §12 tools it already
  owes its own threads — it opted into shared-memory concurrency by
  spawning vCPUs; handlers are just more work items. Other guest fibers
  in the domain are untouched: they never see handlers except through
  memory.
- **Synchrony is the platform primitive; asynchrony is protocol.** Every
  cross-domain call is synchronous to its caller: the calling *fiber*
  parks until results return (its vCPU multiplexes on). A callee wanting
  async semantics builds it in the interface — accept, record, return
  early, deliver later through a `cap` callback or a pipe (split-phase by
  choice, not by force). The type system deliberately does not encode
  "may park" on ops — the same honesty as posix and wasm; annotate later
  only if a consumer demands it.
- **Termination: `exit` is still `exit`.** A domain ends immediately on
  explicit exit; its outstanding offer handles go stale through the
  generation bump, so callers get a clean probeable `CapFault`, never a
  hang — death is revocation (D37). "Entry returned" keeps the domain
  alive *only if* it reified offers that are still held — the runtime then
  provides the implicit serve loop, dispatches serialized one at a time
  (exactly the passive instance's observable behavior, which is why the
  interim is equivalence there, not approximation). A program that never
  reified an offer ends when `main` returns, precisely as today.
- **Entry naming — no magic names.** The entry is whatever export the
  spawner or wirer designates (true today already); `"main"` for programs
  and `"init"` (or none — data segments may be the whole setup) for
  providers is *convention, not semantics*.
- **Blocking becomes expressible.** A handler parks on guest state
  (`memory.wait`) and the live run wakes it (`memory.notify`), or vice
  versa. The interposed `read` then blocks its *caller* exactly as a
  platform stream read would — the caller's vCPU parks at the host boundary,
  which is already a concept, not new machinery.
- **Re-entry is a new fiber, not a deadlock.** A's fiber calling B parks at
  a suspension point — and parked fibers are exactly what suspension points
  release, so when B's handler calls back into A, the dispatch runs as a
  *fresh* handler fiber A[f2] while A[f1] stays parked. Call **cycles are
  recursion, not deadlock**: A→B→A→B… deepens the fiber/park chain and is
  bounded the way recursion always is — fuel plus the domain's existing
  fiber quota (`max_fibers`); exhaustion faults the innermost call
  (probeable), never hangs. No wait-for graph, no detection machinery,
  nothing added to the TCB.
- **The honest hazard is reentrancy, not deadlock** — the classic event-loop
  footgun: a handler holding a *guest-level* lock (a mutex in guest memory)
  across a cross-domain call self-deadlocks when the re-entrant handler
  tries to take it (`memory.wait` on a cell only the parked fiber will
  release — which it never will, since it is waiting on the caller). That is
  the guest's locking discipline to keep, as in every reentrant system; the
  rule of thumb is the usual one — don't hold guest locks across
  cross-domain calls — and fuel remains the backstop for getting it wrong.
- **Service-on-service unlocks.** The offers-in-providers refusal was a
  property of the interim passive instance; under the unified model handlers
  hold and call whatever the domain holds, including other domains' offers.
  Layered guest services (a shell's pipeline stages, fs-on-blockdev) become
  expressible, with cycles handled as above — recursion, bounded, faulting.
- **Metering:** the domain serves on its own fuel — §5.3 provider-pays
  generalizes unchanged (its code, its choice to serve). A parked caller
  burns nothing while parked.
- **Unchanged:** provenance, attest, coverage binding, the `cap` type, and
  the consent rule — `export.handle` remains the only path to a domain's
  service.

Cost, honestly: fiber-per-dispatch scheduling and caller parking touch the
dispatch path on all three backends — a slice far larger than the passive
instance was, with new cross-domain blocking semantics to
differential-test. Sequencing: the landed passive instance keeps serving
the no-live-run case correctly today and is *behaviorally identical* to the
unified model there, so nothing regresses; the fiber slice replaces it
when its consumer (the shell personality, STAGE1) is ready to drive it, and
the instance concept is deleted rather than kept as a second mode.

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

### 5.3 Interposition metering — RESOLVED: the provider pays (2026-07-20)

When a child calls a parent-implemented binding, the parent's dispatch runs —
on whose fuel/quota? **Decided: the provider pays** — its code, its choice to
interpose, which prices interposition honestly (§15's "monitoring is reading
the meters on capabilities you granted"). The drain-by-hammering concern is
the provider's to manage with the tools it already has: it can track per-child
request rates, rate-limit, or kill the child (`Instantiator.kill`) — the
platform's job is only to meter honestly, not to police usage policy.

*As built:* an instanced provider carries a drainable **fuel reserve**
(wirer-set via `Host::set_impl_fuel_reserve`, read via
`Host::impl_fuel_remaining` — the wirer's meter); each dispatch is funded from
it, capped per-call, and a dry reserve is an inert probeable `CapFault` until
topped up (provider state survives the dry spell). A pure (non-instanced)
offer has no provider domain to drain and keeps the flat per-call cap the
wirer accepted at wiring.

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
3. ~~Wire format for the manifest's interface declarations~~ — **RESOLVED
   (wire v6, 2026-07-20): the single type section landed.** A module declares
   shapes in one index space — `type (params) -> (results)` entries are
   function signatures, `interface { 0, 1 }` entries are tuples of indices to
   `type` entries — so each signature is written once and shared. Every
   `impl` export names the interface entry it implements
   (`export "n" impl <iface> : <funcidx>...`), with both verifiers checking
   the implementation matches exactly (and that interface entries reference
   only `Func` entries — interfaces never nest). Entries are deliberately
   *not* dedup-canonicalized on the wire — identity is structural (D59) and
   the host intern canonicalizes at wiring; the linker merges sections across
   units with index-offset remapping. Imports still carry the inline per-op
   `FuncType` (the status-quo shape); **interface-grouped imports** and
   **type-referencing call sites** (the `op`-immediate form, §2.1) are the
   recorded next consumers of the section and were deliberately not built
   here (they touch the binding tables and all three backends' `call.import`
   lowering — their own slice, when a consumer demands them). *Since
   designed: §3.5 (2026-07-21).*
4. ~~`import.attach` concurrency semantics under §12 threads~~ — **RESOLVED
   (specified):** every capability dispatch — `import.attach` and
   `call.import` alike — executes under the domain's one `Host` lock, so
   attach is **atomic with respect to concurrent calls on the same slot**: a
   concurrent `call.import` observes either the entire old binding or the
   entire new one (`(type_id, op, handle, bound)` read as a unit under the
   lock), never a torn mixture; attaches from different vCPUs serialize in
   lock-acquisition order, and there is no fairness guarantee beyond the
   lock's. This is the guarantee the shared dispatch entry has mechanically
   provided since phase 2; it is now pinned as spec rather than accident.
5. ~~`cap.call` mnemonic at a format bump~~ — **RESOLVED (phase 2, restated):
   `cap.call` keeps its mnemonic and wire form** — it simply *is* dynamic
   mode (dispatch on a live handle value); two bumps (v5, v6) have since
   shipped without renaming it, confirming the decision.
6. ~~Snapshot digest wiring~~ — **RESOLVED (doc note, as designed):**
   freeze/restore must be handed the same manifest-carrying module on both
   sides; the existing digest gate enforces this mechanically (the §4
   `module_digest` covers functions, memory, data, exports, offers, and — as
   of v6 — the interface section), and with the rewrite deleted the digest
   is per-module instead of per-instantiation, a strict improvement. No
   codec change was needed.
7. **Globals (wasm parity) — OPEN, design wanted (2026-07-21).** svm-ir has
   no globals of any kind: state is SSA values, the window, or host-side
   capability state. Wasm uses globals for three distinct jobs, and any
   design should treat them separately rather than import the feature
   wholesale: **(a) linker constants** (`__heap_base`, `__stack_pointer` as
   an immutable base) — today answered by data segments, entry arguments, or
   host-side configuration (posix `grant(heap_base, …)`); **(b)
   embedder-supplied configuration values** at instantiation — today
   squeezed through the same channels, awkwardly; **(c) shared mutable
   registers** between linked instances (the dynamic-linking stack pointer) —
   today a plain memory cell in the shared window, since shared-everything
   linking is same-domain here. Design questions: immutable value imports
   only (spawn-time constants — pure value plumbing, no authority, no new
   state class) vs mutable globals (a register file beside the window — a
   new state class touching all three backends, snapshots, and the digest);
   window-backed (a linker-reserved address — no new machinery, but
   occupies guest address space) vs register-backed (fast and clean, but
   new machinery everywhere); and whether they enter the import/export
   system — if so they should be type-section entries like everything else
   (a `value` case beside `Func`/`Interface`), groupable per §3.5. Recorded
   lean: start with **immutable spawn-time value imports** (covers (a) and
   (b), which is all current frontends need) and add mutable globals only
   when a concrete linking consumer demands them — the prime directive
   applies.

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
