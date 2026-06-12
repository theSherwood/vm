# Guest-Driven JIT — Feasibility & Design

## Context

The question: can **guest** code (e.g. an interpreter running as a guest) generate
SVM IR at runtime, hand it to the VM to be verified and JIT-compiled by Cranelift, and
then call into the resulting native code? This is the classic "JIT-inside-the-sandbox"
problem that WebAssembly handles poorly (W^X + immutable modules force guests to ship
their own slow interpreter or round-trip to the host for a fresh module). If SVM can do
it cleanly, guest interpreters (Lua/JS/Python-style) get a fast path for their hot loops
without leaving the sandbox's security model.

**Status: the Model A MVP is built and merged** (this file began as the feasibility
write-up; it now also records implementation status). Phases 1–4 + the reference handler +
resource hardening are landed and differentially tested:

| Piece | Where |
| --- | --- |
| Long-lived `CompiledModule` split + `define_extra` + append-only type-id intern | `crates/svm-jit/src/lib.rs`; tests `crates/svm/tests/jit_incremental.rs` |
| Re-entrancy-sound `run_raw` + mid-run `invoke_extra` (nested guard) | `crates/svm-jit/src/lib.rs`; tests `crates/svm/tests/jit_reentry.rs` |
| `Jit` capability (iface 11): `compile`/`invoke`/`release`, both backends | interp: `crates/svm-interp/src/lib.rs` (dispatch arm + eval-loop nested-VCpu invoke); native: `crates/svm-run/src/lib.rs` (`cap_thunk` intercept, `jit_blob_validator`, `grant_jit`, `jit_cap_run`) |
| Differential + fuzz suite (results/errnos/traps/final-memory equality) | `crates/svm/tests/jit_cap.rs` |
| Compile quota (`-ENOMEM`, shared gate) | `Host::jit_compile` / `set_jit_quota` |
| C surface (`__vm_jit_compile/invoke2/release`, 8-handle powerbox) + demo | `frontend/chibicc`, `crates/svm-run/demos/jit/` (`cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c`) |
| **new→old** (slice #1): module-aware interpreter frames + explicit dispatch table (the B2 foundation) | `crates/svm-interp/src/lib.rs` (`Frame.module`, `TableSlot`, `dispatch_indirect`, `VCpu::new_invoke`); tests in `jit_cap.rs` |
| **old→new** (B2 install): pre-reserved table + `install` op both backends + `__vm_jit_install` | `svm-jit` (`CompiledModule::install`, `table_reserve_log2`, `DefinedFn`), `svm-interp` (op-3 arm, `grant_jit_with_table`), `svm-run` (`jit_native_op` op 3), `frontend/chibicc`; tests in `jit_cap.rs`/`jit_incremental.rs` |
| **new→new** (B2): an invoked unit calls an installed one (live `fn_table` / invoke-time snapshot) | `svm-interp` (`VCpu::new_invoke` snapshot); test in `jit_cap.rs` |
| **slot reclaim** (B2): `uninstall(slot)` frees a table slot for reuse (both backends) | `svm-jit` (`CompiledModule::uninstall`, `n_real_funcs`), `svm-interp` (op-4 arm), `svm-run`, `frontend/chibicc`; tests in `jit_cap.rs` |
| **code-memory compaction** (§6 reclaim, mechanism): `install_at` (exact-slot reinstall) + `extra_fn_count`/`is_running` (occupancy + quiescence) — whole-module recompaction rebuilds the live set into a fresh module, dropping the old arena | `svm-jit` (`CompiledModule::install_at`/`extra_fn_count`/`is_running`); simulated-REPL test `crates/svm/tests/jit_compaction.rs` |

The W^X spike resolved affirmatively (incremental finalize leaves running code intact —
pinned by tests including finalize *during* a run). Phase 5 (B2 table install) and
compaction-based reclaim remain contingent, per the Recommendation. The design analysis
below is preserved as written.

**Cross-call scope — old↔new both directions land (slices #1 and B2 install).** The capability
supports **new→old** (a submitted unit `call_indirect`s back into the original program's table)
and **old→new** (old code `call_indirect`s a unit it `install`ed). Mechanism:
- **new→old (slice #1):** the JIT always lowered a unit's `call_indirect` against the parent
  `fn_table`; the reference interpreter matches it with **module-aware frames** — each `Frame`
  carries a `module` (0 = the vCPU's own program, ≥ 1 = a guest-compiled unit), direct calls
  stay in the caller's module, and `call_indirect` dispatches through an explicit module-aware
  table (`TableSlot`) built from module 0.
- **old→new (B2 install):** a `Jit.install(code_handle) -> slot` op (iface 11 op 3) writes the
  unit into the dispatch table's next **pre-reserved** padding slot — on the JIT via
  `CompiledModule::install` (the unit's natural entry + interned `type_id` into the `fn_table`);
  on the interpreter by registering the unit as a module and filling the same `TableSlot`. Both
  reserve the table identically (`table_reserve_log2` / `grant_jit_with_table`) and fill from the
  parent funcs count, so the returned slot index agrees. The returned slot is a funcref old code
  (or another unit) `call_indirect`s at native speed.

- **new→new (B2):** an *invoked* unit `call_indirect`s an *installed* one. The JIT's invoked
  code dispatches the live `fn_table` (which `install` writes to); the interpreter gives the
  invoke child a snapshot of the domain table + units at invoke time (`VCpu::new_invoke`).

**All four cross-call directions are covered and differentially pinned:** old→old, old→new
(install), new→old (slice #1), new→new. Tests assert matching results, slot indices, and
fail-closed traps (signature mismatch, `-ENOSPC`). C surface: `__vm_jit_install` (iface 11 op 3).
*Edge case noted for later:* a unit that `install`s **during** its own invocation is seen live
by the JIT but not by the interpreter's invoke-time snapshot — exotic (it requires the invoked
unit to hold the `Jit` handle).

**Verdict up front: yes, it's feasible and a strong architectural fit.** The submit-a-blob
boundary the feature needs already exists, the verifier is *designed* to be the trust hinge
for exactly this, and the only real engineering obstacle is the JIT's current one-shot
compile→run→drop lifecycle — which, as §"The one real obstacle" makes precise, is enforced
not by Cranelift but by a single SVM-specific baked constant (the function-table mask).
No escape-TCB (verifier / IR / masking) change is required for the recommended MVP, **provided
one authority-TCB precondition is enforced** (the submitted module's declared memory must
equal the parent window — see the Security argument). With that check, the MVP is
authority-TCB-only.

**Recommended path: Model A (cap.call trampoline) for all workloads, including REPLs** —
where hot cross-unit calls are absorbed by guest-side IR re-emission rather than a shared
table. Model B2 (persistent shared function table) is specified below and got materially
cheaper after Phase 1 landed its type-id registry as an append-only intern; it is gated on
either a measured megamorphic workload or a real interpreter port finding the shim convention
burdensome — see "Recommendation".

---

## Framing correction (important)

The guest is sandboxed code running *inside* the VM; it cannot call host Rust APIs
(`verify_module`, `compile_and_run`, …) directly. Everything crosses the **`cap.call`**
boundary. So the real shape is:

1. Guest builds a **serialized IR blob** (the `svm-encode` binary format) in its own window.
2. Guest does `cap.call` on a new **`Jit` capability**, passing the blob as a `(ptr, len)` pair.
3. The **host** reads the blob out of guest memory, runs `decode_module` → `verify_module`
   (the security hinge) **plus the memory-match precondition below**, and — only if all pass —
   Cranelift-compiles it into a **long-lived `JITModule`**.
4. The host returns a **handle** to the compiled code; the guest invokes it later.

Every step except #3's "keep the module alive" already exists in the codebase.

---

## How it works today (the machinery we reuse)

**Submitting a blob across `cap.call` — already the universal pattern.**
`Stream.write(buf, len)` and the `Memory` cap (`map`/`unmap`/`protect`) already take a
guest `(ptr, len)`, and the host reads/writes guest memory *in place* with bounds checks —
no copy:
- IR op `CapCall { type_id, op, sig, handle, args }` — `crates/svm-ir/src/lib.rs:1012`.
- Host trampoline `cap_thunk(ctx, mem_base, mem_size, mem_reserved, type_id, op, handle, args, …)`
  — `crates/svm-run/src/lib.rs:34`. `ctx` is `*mut Host`.
- Dispatch `Host::cap_dispatch_slots(...)` — `crates/svm-interp/src/lib.rs:2407`.
- The borrow path `MprotectWindow::read_bytes` / `range_committed` —
  `crates/svm-run/src/lib.rs:112`; the `Stream.write` worked example —
  `crates/svm-interp/src/lib.rs:2508`.
- Returning a fresh handle: `Host::grant(type_id, binding)` mints a `(generation, slot)`
  handle (256-slot table, `CAP_LOG2 = 8`); `resolve` does a **masked** lookup + type +
  generation check (Spectre-v1-safe, ABA-safe) — `crates/svm-interp/src/lib.rs` ~2330 / 2388.

  **Adding a new capability needs no verifier/IR/escape-TCB change** — only an interface
  signature in the module's type section + a host vtable/handler. (DESIGN.md §3c "D45",
  §2a authority- vs escape-TCB.)

**The verify pipeline is the trust hinge and is cheap.**
- `decode_module(bytes) -> Result<Module, DecodeError>` — `crates/svm-encode/src/lib.rs:622`.
  Untrusted-input-facing, fuzzed, fail-closed, never pre-allocates from an untrusted count.
- `verify_module(&Module) -> Result<(), VerifyError>` — `crates/svm-verify/src/lib.rs:85`.
  Single linear O(module-size) pass, `#![forbid(unsafe_code)]`, dep-free (only `svm-ir`),
  fuzzed panic-free. Safe to run on an arbitrary guest-supplied blob.
- `Module { funcs, memory, data }` is self-contained — `crates/svm-ir/src/lib.rs:1209`;
  a second module is just another value you can build at runtime.

**The design already anticipates the surrounding concepts.** DESIGN.md §8 (`lib`-lines
1064–1104): a *Module* is "a unit of code + exports… **multiple modules freely share a
domain**" — not an isolation unit. §13/§14 (1428–1467) specify an **`Instantiator`**
capability for VM-in/-beside-VM. A "compile-this-IR" capability is squarely in that family.

**Cost note (in our favor).** DESIGN.md D45's "~1.24× a wasm import call" is the
*pre-optimization* number; the current measured `cap.call` cost is **≈0.67× a Wasmtime
import** (i.e. cheaper than a wasm import, not dearer). This strengthens Model A's
"boundary-crossing per invocation" tradeoff below — the amortized-to-zero regime is wider
than the original framing implied.

---

## The one real obstacle — the baked function-table mask

The headline obstacle is **not** simply "the JIT drops the module at the end." Dropping is
policy. The *structural* obstacle is what makes incremental compilation non-trivial: the
**function-table mask is baked as a compile-time `iconst` at every `call_indirect` site**.

`compile_and_run*` (`crates/svm-jit/src/lib.rs:313`) builds a `JITModule` over a 256 MiB
reserved arena (`ArenaMemoryProvider::new_with_size(256 << 20)`, `:955`), declares *all*
functions up front (`:960`), defines them, calls `finalize_definitions()` exactly once
(`:1134`), runs, then **`drop`s the module and frees the code** (`:1296`). The function table
is sized `funcs.len().next_power_of_two()` (`:1155`).

The Spectre-safe indirect-call lowering (`indirect_dispatch`, `:2814`) is:

```
slot   = band(uextend(idx), fn_table_mask)   // :2816  — mask, not branch (Spectre-safe, I1/I2)
tid    = load(entry_addr)                      // :2823
trap unless tid == type_id_of(distinct, ty)    // :2827 / :2833
```

`fn_table_mask` is computed as `next_power_of_two(nfuncs) - 1` (`:1892`) and **materialized as
a single `iconst`** at each call site (`:2816`). The same value is captured as `fiber_mask`
(`:997`) inside the fiber runtime. The `cap.call` thunk/ctx and fiber/thread runtime
addresses are likewise baked as compile-time constants (`:979`, `:1015`, `:3000`, `:3012`) —
those are *fine*, because thunk addresses never change.

**Why this single fact organizes the entire design space below.** The moment you add a
function and the count crosses a power-of-two boundary, the correct mask changes — and every
already-compiled `call_indirect` site holds the *old* mask as an immediate. So:

- **Model A sidesteps the mask entirely** — guest-submitted code is reached through the host
  thunk, never installed in the shared table, so the mask never moves. Incremental
  define/finalize on a live module is purely a Cranelift capability question (it supports it),
  with no SVM constant to invalidate.
- **Model B2 neutralizes the mask by pre-sizing** — reserve a fixed power-of-two table up
  front so the mask is constant from `t=0` and stays byte-identical as slots are populated.
- **Model B1 is rejected precisely because it cannot** — a growable table moves both base and
  mask, forcing the most security-sensitive lowering to recompute them per call.

So the obstacle is real but local: keeping `JITModule` alive and finalizing incrementally is
a Cranelift capability (confirmed supported on our pinned `cranelift-jit 0.132.0`,
`Cargo.toml:10`), and the *only* SVM-side blocker — the baked mask — is designed around, not
through, by both recommended models. Because `cap.call` is synchronous, defining a function
mid-run is safe: the host thunk runs *on the guest's stack*, decode/verify/define/finalize the
new code (mprotecting only the new pages — subject to the W^X check in Open Questions), and
returns before resuming the guest.

---

## The proposed design: a `Jit` capability

Grant the guest a `Jit` handle in the powerbox (opt-in, like `Memory`). Operations:

- `op=0 compile(ir_ptr, ir_len) -> handle | -errno`
  Host borrows `[ir_ptr, ir_len)` from the window, `decode_module` + `verify_module` + the
  **memory-match precondition** (declared memory must equal the parent window — see Security);
  on success defines the function(s) into the live `JITModule`, `finalize_definitions()`, and
  `grant`s a `CompiledCode` handle wrapping the new code pointer + signature id. On any
  failure (decode/verify/memory-mismatch/signature-mismatch/arena-exhausted) returns a negative
  errno and installs nothing (fail-closed).
- `op=1 invoke(handle, arg0..argN) -> result` *(only in the cap.call-trampoline model — see below)*.
- `op=2 release(handle)` — drop the slot (generation bump makes the old handle inert).

The compiled code is handed the **same vmctx** as the parent (window base/mask,
`fn_table_base`, the `cap.call` thunk addr). Because the host controls compilation, it bakes
the *parent's* constants into the new function exactly as it does for the main module — so the
new code reads/writes the guest interpreter's own linear memory with zero copy and can itself
`cap.call` back out, with no runtime vmctx pointer needed for the environment (only the call's
own args use the i64-slot ABI). Submitted IR uses the binary `svm-encode` format (the
hardened, fuzzed decode path) — not text.

**Isolation model — same guest, not a nested guest.** The compiled code runs in the *same
domain* as the guest that submitted it: same window, same handle table, same authority. This
is DESIGN.md §8 working as intended — a module is a unit of code, not an isolation unit, and
the JITed blob is simply a second module joining the guest's domain. Two consequences: (1) the
trust boundary is **verification, not isolation** — the blob is exactly as powerful as its
submitter, no more (it cannot reach beyond the window or the guest's granted handles) and no
less (there is no inner sandbox); (2) this is why the memory-match precondition is
load-bearing — a blob declaring a different memory size would be a module claiming a different
domain while running in this one. For evaluating *untrusted* code with *less* authority than
the submitter (e.g. a REPL running foreign user code), the right tool is the §13/§14
`Instantiator` capability (a child VM with its own window and attenuated handles), not `Jit`.
`Jit` adds speed to a guest; it never adds a protection domain.

---

## Invocation models compared (the explicit ask: safety / speed / simplicity)

Both models share the same prerequisite — a **long-lived `JITModule`** owned by the host
(reachable from the thunk's `ctx`) plus the verify-before-compile capability — so starting
with the simpler one wastes no work toward the faster one.

### Model A — invoke via `cap.call` trampoline
The returned handle is called through `cap.call op=invoke`; the host thunk trampolines into
the compiled code.

- **Safety — highest.** No change to the function-table indirect-call dispatch (the
  Spectre-safe `band(idx, mask)` + `type_id` check, invariant I2 — the most security-sensitive
  lowering); the mask never moves because the code is never in the table. The new code is just
  another verified function reached through the *existing* host thunk, which already does trap
  propagation. The only new escape-relevant surfaces are (1) "keep `JITModule` alive + finalize
  incrementally" — shared Cranelift codegen, escape-TCB either way — and (2) the memory-match
  precondition, which lives in the authority-TCB handler. With (2) enforced, no escape-TCB
  *change* is required (see Security). The capability itself is otherwise authority-TCB only.
- **Speed — boundary crossing per *invocation*.** Each call pays the `cap.call` cost (now
  ≈0.67× a Wasmtime import, D45-updated) + marshalling args through the i64-slot ABI. The
  compiled body runs at full native JIT speed *internally*. So this is **excellent when you
  compile a whole hot loop / function and call it once per outer use** (the overhead amortizes
  to ~0), and **poor for a tiny leaf function called millions of times** in a tight guest loop
  (the boundary dominates). Args are limited to the scalar i64-slot ABI; richer state goes
  through the shared window (which the compiled code can already touch).
- **Simplicity — highest.** Reuses the `Stream.write` / `Memory` pattern verbatim: new
  handler + interface signature, a host-owned `JITModule`, the memory-match check, nothing
  else. No table growth, no vmctx reshaping, no verifier work.

**The heterogeneity question (and the shim answer).** Under A, JITed code has no table
index — so a guest-language array mixing old and new procedures cannot be uniform funcrefs,
which looks like a tag-every-callable tax on every guest interpreter. The tax is real but
**compressible to one shim function**, because interpreters' function values are already
polymorphic (Lua: Lua-fn vs C-fn; Python: function/builtin/bound-method) and closure-shaped
(`{code_idx, env}`): the parent module pre-declares `shim(env) → cap.call invoke(env.handle,
…)` in its table at t=0, and a JITed function is represented as `{shim_idx, env carrying the
handle}`. Every call site stays a plain `call_indirect(closure.code_idx, closure.env)` — the
tag collapses into the code pointer, exactly the tiered-VM trick (the function object's entry
point *is* the dispatch). No value-representation change, no call-site branches; the residual
cost is the boundary crossing per call into JITed code (`call_indirect` → shim → `cap.call` →
trampoline). What the shim can't fix: a guest whose callables are *bare* funcrefs with no env
slot (no closure representation at all), and the per-call boundary itself — those are what B2
removes.

### Model B — invoke via direct `call_indirect` (persistent shared table)
Install the compiled function as a slot in the function table so the guest `call_indirect`s
straight into it at near-native speed. Two sub-variants — and the variant choice matters
more than A-vs-B for safety:

**B1 (growable table) — avoid.** A genuinely *growable* table can move its base on realloc
(it's threaded as `fn_table_base` through every call → needs a live vmctx update) and forces
the power-of-two **mask** (baked into every call site as a constant, `:2816`) to be re-derived
from vmctx per indirect call. That edits the Spectre-safe dispatch — the touchiest escape-TCB
spot — for little reason.

**B2 (pre-reserved fixed-size table) — preferred.** Reserve a fixed large power-of-two
table up front (e.g. 2^16 slots ≈ 1 MiB of `FnEntry`). Then:
- The base never moves and the **mask stays a compile-time constant** (the `iconst` at
  `:2816` is `2^16 - 1` from `t=0` and never changes), so the Spectre-safe indirect-call
  *lowering* (`band(idx, mask)` + `type_id` check, invariants I1/I2) is **structurally
  byte-identical to today**. (One provenance caveat: see the type_id note below — the *shape*
  is identical, but the `type_id` constant's *source* changes.)
- Only table *population* becomes dynamic: empty slots already exist as `PADDING_TYPE_ID`
  (`u32::MAX`, `:144`) entries that trap (`:1169`); `compile` fills the next slot and returns
  its index to the guest.
- The fiber/thread runtimes (which capture `funcs.len().next_power_of_two()`, `:997`) simply
  use the reserved length.

Safety / speed / simplicity, for B2:
- **Safety — moderate, bounded.** No change to the indirect-call lowering's *shape*; the new
  surface is "host writes new `FnEntry` slots into a live table + keeps the `JITModule` alive +
  the (small) interning function below." Reviewable, and far smaller than B1.
- **Speed — near-native per call.** A masked table load + type check + indirect branch
  (retpoline/eIBRS), identical to any existing `call_indirect`. **No boundary crossing on
  cross-unit calls** — the decisive property for REPL/shell cross-calling.
- **Simplicity — middle, and most of it already landed.** Long-lived module (shared with A)
  **plus** an `install` op filling reserved slots, a fixed-table cap, and the type-id registry —
  which Phase 1's implementation demystified to an append-only intern (below), already merged.
- **Guest simplicity — the whole point.** An installed function *is* a funcref: mixed arrays
  of old and new procedures are uniform, old code calls everything with plain `call_indirect`,
  and the interpreter tax (even the one-shim version) is zero.

**B's type-identity requirement, demystified: an append-only intern, not a linking subsystem.**
An earlier draft priced this as "the type-identity half of §13 linking" — that framing was
inherited from the general multi-peer-module problem and is wrong for the shape Phase 1
actually landed: a *single* `CompiledModule` owns the domain's entire id space (`distinct`),
ids are baked as immediates at compile time, and the registry is **never read at runtime** —
it is consulted only inside a synchronous `cap.call` while the guest is suspended, and adds
zero runtime-readable state. Dispatch soundness reduces to one property: the map
`FuncType → type_id` is an **injection, stable over time (append-only — an id never remaps),
and total over participants** (every signature at any call site or table slot is interned
before code referencing it is lowered). Given that, id-equality coincides *exactly* with the
interpreter's structural equality — which makes interning **cleaner** than a frozen
parent-anchored universe, not just more expressive: a frozen universe bakes always-trapping
`NO_MATCH_TYPE_ID` into any site whose signature arrives in a later unit (a wart the Phase-3
reference handler would have had to replicate), while interning lets a later install satisfy
it, matching structural semantics by construction. **This landed in Phase 1** (`intern_type` /
`intern_unit_sigs` / `CompiledModule::interned_type_id`, behavior-preserving today, pinned by
`type_ids_are_interned_append_only_across_units`); new signatures — e.g. a guest JIT's arity-
or type-specialized calling conventions for hot callees — are first-class for table dispatch,
not just for intra-unit direct calls.

### Recommendation — Model A; B2 only on measured evidence

**Ship Model A (Phases 1–4) for all workloads, including the REPL. Treat B2 as a contingency
kept cheap by the shared Phase-1 groundwork, not a committed destination.** Three reasons, in
descending weight:

**1. The risk asymmetry is lopsided — though narrower than first priced.** A's worst case is
a *performance/ergonomics* problem (boundary cost on cross-unit calls, the one-shim
convention); B2's worst case is a *security* problem (a host-writes-into-live-table primitive,
plus compaction touching the module lifecycle). A performance problem announces itself in a
benchmark; an escape-TCB mistake announces itself in an audit, or worse. The type-id registry
**no longer counts against B2** — it demystified to an append-only intern with zero
runtime-readable state and landed in Phase 1 (see "demystified" above) — so B2's residual
surface is the `install` write itself, which under the single-threaded MVP (synchronous
`cap.call`, guest suspended, no concurrent reader) has no publication race to reason about.
Still: demand a *demonstrated* need before buying it.

**2. The measured numbers narrow the gap from both ends.** The cap.call correction (≈0.67× a
Wasmtime import, not the pre-optimization 1.24×) shrinks A's penalty — a cross-unit call
through the thunk is single-digit nanoseconds, roughly 5–20× a native `call_indirect`, not an
order-of-magnitude cliff. Meanwhile the reclaim finding (no per-function free → periodic
whole-module compaction, plan step 6) means B2 does not actually deliver "incremental forever"
for the REPL — it delivers amortized-periodic recompile. The gap between "A degrades" and "B2
degrades gracefully" is real but much narrower than naive framing suggests.

**3. A has a guest-side escape hatch for the REPL's hot cross-calls: re-emission.** The
classic objection to A for a REPL — *define helpers early, loop over them later, and every
iteration eats a boundary crossing* — assumes cross-unit calls must cross the boundary. They
don't: the guest *owns the IR* for everything it has compiled. When a new unit hot-calls an
earlier helper, the guest re-emits that helper's IR into the new blob, making the cross-call a
verified *direct* call — full native speed, zero boundary crossings, zero new host surface.
This is selective inlining at the guest layer: not recompile-the-world (O(accumulated program)
per entry), but recompile-the-hot-closure, and the guest — not the host — has the profile
information to decide when. The cost is some code duplication in the arena; the benefit is
that the entire escape hatch is guest policy, invisible to the TCB.

**The decision rule for B2 (Phase 5):** build it when a real guest gives a concrete reason —
a *lower* bar than the original "profiling must prove boundary cost dominates," for two
reasons. First, B2 got cheaper: the registry landed in Phase 1 as the append-only intern, so
the remaining delta is the pre-reserved table size + the `install` op. Second, the case for it
gained a second independent axis beyond perf: **guest implementation complexity** — under A,
uniformity costs every interpreter the shim convention (workable, but a porting tax, and
unavailable to bare-funcref guests); under B2 it costs nothing, because installed functions
are real funcrefs. The perf trigger remains **megamorphic / late-bound call sites** — where
the callee is unknown when the calling unit is compiled, so re-emission has nothing to inline
and every dispatch must cross the boundary or go through a shared table. Either trigger — a
measured megamorphic workload, or a real interpreter port finding the shim burdensome — is
sufficient. Until one fires, B2's `install` surface stays off the books.

Both models share the long-lived-`JITModule` prerequisite, so building A first is never
throwaway work toward B2. If B2 is ever built, **code reclaim is its load-bearing
constraint** — more so than dispatch speed — and it is entangled with the module lifecycle,
not an orthogonal allocator feature; see Open questions.

---

## Implementation plan (phases 1–4 landed — see Status; 5–6 contingent)

Phased; each phase is independently testable and keeps the escape-TCB crates untouched.

1. **Long-lived JITModule (the enabling refactor, no new feature yet).**
   Split today's `compile_and_run*` (`crates/svm-jit/src/lib.rs:313`) into
   `compile(module) -> CompiledModule { jit_module, ids, table, vmctx }` + `run(entry, args)`,
   with `CompiledModule` *owning* the `JITModule` for the whole run (no `drop` at `:1296`).
   Add `CompiledModule::define_extra(funcs: &[Func]) -> Vec<CodePtr>` that declares + defines
   + `finalize_definitions()` incrementally and returns the new code pointers. **Crucially,
   `define_extra` must NOT register the new functions into the existing `fn_table`** — they
   are thunk-reachable only (Model A), so the baked `fn_table_mask` (`:1892`/`:2816`) and
   `fiber_mask` (`:997`) never change. Prove the existing one-shot path is unchanged (it
   becomes `compile().run()`); the whole `jit_diff` differential + escape-oracle suite must
   stay green. *This is the only escape-TCB-adjacent change and the riskiest — it is shared
   codegen, validated by the existing differential.* This phase is also where the W^X spike
   (Open questions) is resolved, since it introduces the first multi-finalize path.

2. **`Jit` capability (Model A).** New `Binding::Jit { module: *mut CompiledModule }` +
   `cap_dispatch_slots` arm (`crates/svm-interp/src/lib.rs:2407`) implementing
   `compile`/`invoke`/`release`: borrow the blob via `read_bytes`, `decode_module` +
   `verify_module` + **the memory-match precondition**, `define_extra`, `grant` a
   `CompiledCode` handle; `invoke` trampolines into the stored code pointer with the i64-slot
   ABI. Grant the cap in `run_powerbox` (`crates/svm-run/src/lib.rs:335`) behind an opt-in flag.

3. **A reference handler in the interpreter** so the differential oracle can model the
   capability (compile = "decode+verify+memory-match, then run on the interpreter"), keeping
   interp↔JIT parity for any guest that uses it. (The memory-match precondition is also what
   keeps verified-bounds and runtime-mask aligned, so the differential stays meaningful.)

4. **A demo guest** (`crates/svm-run/demos/jit/…`): a tiny bytecode interpreter that emits
   SVM IR for a hot loop, `compile`s it, and `invoke`s it — the end-to-end proof, checked
   against a pure-interpreter run of the same program.

5. **Model B2 — contingent** (see the decision rule in the Recommendation: a measured
   megamorphic workload, *or* a real interpreter port finding the shim convention burdensome —
   either trigger suffices). The remaining delta, the registry having already landed in
   Phase 1 (`intern_type` / `interned_type_id`):
   - a **pre-reserved fixed-size power-of-two function table**: pad the parent's table to a
     host-chosen reserved length at compile (empty slots stay `PADDING_TYPE_ID`); the
     indirect-call lowering and its mask `iconst` are byte-identical from `t=0` — and
     `define_extra` already takes the parent's mask as an explicit parameter, so extra units
     inherit it unchanged;
   - an **`install(handle) → slot_index`** op: look the function's signature up in the
     interned registry, write `(type_id, verified code ptr)` into the next `PADDING_TYPE_ID`
     slot, return the index — a funcref the guest `call_indirect`s directly at native speed.
     Under the single-threaded MVP the write is race-free by construction (synchronous
     `cap.call`, guest suspended); a multi-threaded future needs release-ordered publication
     (code pointer first, `type_id` last — a padding slot traps until the id lands);
   - the fiber/thread runtimes use the reserved table length (`:997`).
   Avoid the B1 growable-table variant (moving base + dynamic mask) — it edits the
   Spectre-safe dispatch for no benefit B2 lacks.

6. **Code reclaim** (load-bearing if B2 is built; for Model A — including the A-with-re-emission
   REPL path, whose duplication makes arena pressure *more* likely — the MVP cap below
   suffices until profiling says otherwise). Because
   `cranelift-jit`'s `JITModule` has **no per-function free**, `release` cannot be a drop-in
   allocator swap. The realistic strategy is **periodic whole-module compaction**: track the
   live definition set, and when the arena passes a watermark, recompile the live set into a
   fresh `CompiledModule` and atomically swap (repopulating the B2 table). This reintroduces an
   *amortized-periodic* O(live-program) recompile — better than recompile-per-entry, but not
   free — so the cost model must be stated honestly. MVP for one-shot: cap total compiled bytes
   / return `-ENOMEM`.
   - **Mechanism landed (the reclaim primitives + a simulated-REPL test).** The
     recompaction strategy above is realized against three `CompiledModule` primitives that
     add **no escape-TCB codegen** — compaction only replays the existing
     `compile`/`define_extra`/`install` paths into a fresh module and drops the old one (RAII
     frees its whole arena): `install_at(slot, code, type_id)` reinstalls a unit at its
     **exact** old slot (so a funcref a guest already holds keeps resolving to the same unit
     across the swap — `install` only fills the *next* padding slot, which cannot reproduce a
     history with `uninstall` gaps); `extra_fn_count()` is the monotonic occupancy proxy a
     watermark watches (it restarts near zero in the fresh module — the visible reclaim);
     `is_running()` is the quiescence guard (recompaction is only sound when no run is in
     flight — the guest is suspended *inside* the module being compacted, so a guest op can
     never trigger it; it is embedder-facing, e.g. a REPL driver between prompts).
     `crates/svm/tests/jit_compaction.rs` drives the embedder pattern end-to-end: redefining
     one function 40 times accumulates 40 dead definitions, then recompaction rebuilds only the
     single live definition into a fresh module that behaves identically, keeps its slot, and
     bounds occupancy by the live set, not the history; a second case pins exact-slot
     reproduction across an `uninstall` gap. **Remaining (the guest-facing integration):**
     wiring this through the `svm-run`/`Host` `Jit` path — enumerating the domain's *live*
     units (liveness is the embedder's handle-table policy: a unit dies when its `CompiledCode`
     handle is released *and* it is not installed), remapping `Host` unit→native pointers + the
     handle table across the swap, and driving a multi-prompt REPL over a **persistent window**
     (carrying `final_mem` forward as the next prompt's `init_mem`) with a differential against
     the interpreter. That integration is its own focused slice; the load-bearing mechanism it
     builds on is the part above.

---

## Security argument

The escape-freedom contract (DESIGN.md §2a) is
`Verified(module) ∧ Correct(JIT) ∧ Correct(runtime) ⟹ escape-free`. Guest-submitted code
slots into it: it must pass the *same* `decode_module` + `verify_module` gate as any other
module before a single instruction is compiled, so a malicious/garbage blob is rejected
fail-closed and never reaches Cranelift.

**The one new authority-TCB precondition that makes "no escape-TCB change" true.** Confinement
masking (invariant I1) masks every load/store into `[0, size)`, where `size` is the module's
declared memory; for the main module the window is sized to exactly that
(`run_powerbox:1206`). A guest-submitted blob that declared a *larger* memory and was compiled
against its own declared size would get the JIT to mask into that larger region — an escape,
achieved without touching one line of the masking lowering. The fix lives in the authority-TCB
`Jit` handler, not the JIT: **reject any submitted module whose declared memory ≠ the parent
window** (the host then compiles the blob with the parent's window base/mask, so I1 confines it
to the parent window exactly as for the main module). With that check, the masking lowering is
unchanged, the compiled code runs under the same guard-page/trap machinery as the parent, and
the JIT's choice of mask value remains host-controlled rather than guest-influenced. Enforcing
equality also keeps verified-bounds and runtime-mask aligned, which is what preserves the
interp↔JIT differential (Phase 3).

Beyond that: Model A adds **no** escape-TCB surface (authority-TCB only); Model B2 leaves the
Spectre-safe indirect-call lowering *structurally* byte-identical and only adds dynamic table
*population* on top of the already-landed type-id intern. The intern's soundness obligation is
the registry invariant (injective, append-only-stable, total over participants — then
id-equality ≡ structural equality; see the "demystified" note), discharged by a ~10-line
auditable function that is never read at runtime; the `install` write is the one remaining
reviewable surface (B1's growable table, which we reject, would have edited the dispatch
itself). The capability is opt-in and attenuable like every other powerbox grant.

## Open questions / risks

- **Code reclaim ⇄ module lifecycle (the load-bearing constraint for any long session).**
  *Two distinct pressures.* (1) **Slot** pressure — the fixed `call_indirect` table fills as a
  REPL installs definitions: **addressed** by `uninstall(slot)` (op 4), which frees a slot for
  reuse (a redefinition `uninstall`s the old slot and `install`s the new code, reusing the
  index). (2) **Code-memory** pressure — repeated `compile`s consume the 256 MiB code arena
  (`:955`) with no per-function reclaim in `cranelift-jit`: **mechanism landed; guest-facing
  integration open.** This is *not* an orthogonal allocator feature — with no per-function
  free, reclaim means periodic whole-module compaction (recompile the live set → swap),
  reintroducing amortized-periodic recompile cost (plan step 6). The reclaim *primitives*
  (`install_at` / `extra_fn_count` / `is_running`) and a simulated-REPL test now exist
  (`crates/svm/tests/jit_compaction.rs`); what remains is wiring them through the
  `svm-run`/`Host` `Jit` path (live-unit enumeration, handle remap, a persistent-window REPL
  differential — plan step 6). The recommended A-with-re-emission REPL path *increases* arena
  pressure (duplicated helper bodies), so the MVP byte-cap / `-ENOMEM` backstop is what gets
  profiled first; compaction is the upgrade if real sessions hit the cap.
- **W^X integrity of incremental `finalize_definitions` (escape-relevant, not just
  functional).** Today `finalize_definitions()` is called exactly once (`:1134`), so the
  multi-finalize path is unexercised. The real question is not re-entrancy correctness but
  whether `cranelift-jit 0.132`'s `ArenaMemoryProvider` ever flips *already-finalized* pages
  back to writable during a *later* finalize — a transient W^X violation on running code. The
  single-threaded MVP sidesteps this entirely (the compile thunk runs synchronously on the
  guest stack; nothing else is executing during finalize), which is the *stronger* reason for
  the single-threaded restriction below. For the multi-threaded future this is the gating
  spike — resolve it in Phase 1.
- **vmctx sharing.** The compiled code must observe the parent's window base/mask/thunk so
  masking + callbacks resolve. Because the host controls compilation it bakes the parent's
  constants directly (as for the main module), so no runtime vmctx pointer is needed for the
  environment — only the call's own args use the i64-slot ABI. Confirm in Phase 2.
- **Concurrency.** If the guest uses threads/fibers, compiling mid-run interacts with the
  cooperative scheduler and the W^X question above — MVP restricts the `Jit` cap to
  single-threaded domains, which also makes the transient-W concern non-exploitable.

## Verification approach

- Phase 1: full `cargo test --workspace` + `jit_diff` differential + escape-oracle snapshots
  unchanged (the refactor is behavior-preserving); resolve the W^X spike against the real
  `cranelift-jit 0.132` incremental define→finalize→define cycle.
- Phase 2/3: new differential cases — a guest that compiles+invokes IR must produce identical
  results/traps on the interpreter reference and the JIT; fuzz the `compile` op with arbitrary
  blobs to confirm decode/verify/memory-match reject fail-closed (never compile invalid IR, and
  in particular never compile a blob whose declared memory ≠ the parent window). Include the
  **mixed-dispatch array** case: the guest builds an in-window array of *tagged* callables —
  table indices for original procedures, `CompiledCode` handles for JITed ones — and old code
  iterates it, branching to `call_indirect` or `cap.call invoke` per tag. This is the canonical
  "closures added next to old procedures" REPL shape under Model A (heterogeneous dispatch is
  A's ergonomic cost; uniform funcref arrays are what B2's table installation would buy), and
  the Phase-1 invariant it leans on — extra code is *unreachable* from the table, so the tags
  can never be confused — is already pinned by `parent_call_indirect_cannot_reach_extra_code`.
- Phase 4: `cargo run -p svm-run -- crates/svm-run/demos/jit/…` matches a pure-interpreter run.
