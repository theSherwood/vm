# Guest-Driven JIT — Feasibility & Design

## Context

The question: can **guest** code (e.g. an interpreter running as a guest) generate
SVM IR at runtime, hand it to the VM to be verified and JIT-compiled by Cranelift, and
then call into the resulting native code? This is the classic "JIT-inside-the-sandbox"
problem that WebAssembly handles poorly (W^X + immutable modules force guests to ship
their own slow interpreter or round-trip to the host for a fresh module). If SVM can do
it cleanly, guest interpreters (Lua/JS/Python-style) get a fast path for their hot loops
without leaving the sandbox's security model.

**Deliverable for this task: a design/feasibility write-up + implementation plan only —
no code changes yet.** This file is that write-up.

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
table. Model B2 (persistent shared function table) is specified below but gated on measured
evidence; see "Recommendation".

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
  the type_id registry below." Reviewable, and far smaller than B1.
- **Speed — near-native per call.** A masked table load + type check + indirect branch
  (retpoline/eIBRS), identical to any existing `call_indirect`. **No boundary crossing on
  cross-unit calls** — the decisive property for REPL/shell cross-calling.
- **Simplicity — lower than middle.** Long-lived module (shared with A) **plus** dynamic
  table population, a fixed-table cap, **and the domain-global `type_id` registry below** — the
  registry makes B2 effectively "implement the type-identity half of §13 linking," which is
  more than a fixed-table bolt-on.

**B's one real new requirement: a domain-global signature→`type_id` registry.** Today
`type_id`s are computed *per-module* via `distinct_types`, and the indirect-call check loads
the slot's `type_id` and compares it against `type_id_of(lower.distinct, ty)` (`:2823`/`:2827`).
For a `call_indirect` in a later unit to type-check against a function compiled in an earlier
unit, `type_id`s must be assigned from a persistent per-domain registry. Note this is *not*
a change to the dispatch instruction sequence, but it **does change the provenance of the
`type_id` immediate** fed into the `icmp` — the constant now comes from a cross-unit registry
rather than a module-local distinct set. Same instructions, different (and now escape-relevant)
input. This is not a wart — it is exactly the §13 linking direction ("domain-global indices…
assigned at instantiation/link") — but it should be reviewed as a (small) escape-relevant
change, not waved through as "byte-identical."

### Recommendation — Model A; B2 only on measured evidence

**Ship Model A (Phases 1–4) for all workloads, including the REPL. Treat B2 as a contingency
kept cheap by the shared Phase-1 groundwork, not a committed destination.** Three reasons, in
descending weight:

**1. The risk asymmetry is lopsided.** A's worst case is a *performance* problem (boundary
cost on cross-unit calls); B2's worst case is a *security* problem (the domain-global
`type_id` registry is an escape-relevant provenance change, plus a host-writes-into-live-table
surface, plus compaction touching the module lifecycle). A performance problem announces
itself in a benchmark; an escape-TCB mistake announces itself in an audit, or worse. For a
sandbox whose whole pitch is the §2a contract, demand *demonstrated* need before buying B2's
surface — not a hypothesized workload sketch.

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

**The decision rule for B2 (Phase 5):** instrument the demo REPL under Model A, and start
Phase 5 only on measured evidence that cross-unit boundary cost dominates *and* guest-side
re-emission cannot absorb it. The genuine residual case is **megamorphic / late-bound call
sites** — where the callee is not known when the calling unit is compiled, so re-emission has
nothing to inline and every dispatch must either cross the boundary or go through a shared
table. If profiling shows that shape dominating a real workload, B2 (pre-reserved fixed table
+ global `type_id` registry + compaction-based reclaim) is the right tool, and the analysis
below stands. Until then, its escape-relevant surface stays off the books.

Both models share the long-lived-`JITModule` prerequisite, so building A first is never
throwaway work toward B2. If B2 is ever built, **code reclaim is its load-bearing
constraint** — more so than dispatch speed — and it is entangled with the module lifecycle,
not an orthogonal allocator feature; see Open questions.

---

## Implementation plan (for when we build it; design-doc deliverable lands first)

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

5. **Model B2 — contingent, gated on Phase-4 profiling** (see the decision rule in the
   Recommendation: built only if measured cross-unit boundary cost dominates a real workload
   and guest-side re-emission cannot absorb it, e.g. megamorphic call sites):
   - a **pre-reserved fixed-size power-of-two function table** populated dynamically (empty
     slots stay `PADDING_TYPE_ID`); the indirect-call lowering and its mask `iconst` are left
     byte-identical;
   - a **domain-global signature→`type_id` registry** so cross-unit `call_indirect` type
     checks resolve across separately-compiled units (reviewed as the small escape-relevant
     provenance change it is);
   - the `compile` op returns a **funcref slot index** (not a cap handle), which the guest
     `call_indirect`s directly at native speed;
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
*population* + a per-domain `type_id` registry — a bounded, reviewable surface, with the
understanding that the registry changes the *provenance* of the `type_id` immediate fed into
the dispatch check (B1's growable table, which we reject, would have edited the dispatch
itself). The capability is opt-in and attenuable like every other powerbox grant.

## Open questions / risks

- **Code reclaim ⇄ module lifecycle (the load-bearing constraint for any long session).**
  Repeated `compile`s consume the 256 MiB code arena (`:955`) with no per-function reclaim in
  `cranelift-jit`. This is *not* an orthogonal allocator feature: with no per-function free,
  reclaim means periodic whole-module compaction (recompile live set → swap), reintroducing
  amortized-periodic recompile cost (plan step 6). Note the recommended A-with-re-emission
  REPL path *increases* arena pressure (duplicated helper bodies), so the MVP byte-cap /
  `-ENOMEM` backstop is what gets profiled first; compaction is the upgrade if real sessions
  hit the cap.
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
  in particular never compile a blob whose declared memory ≠ the parent window).
- Phase 4: `cargo run -p svm-run -- crates/svm-run/demos/jit/…` matches a pure-interpreter run.
