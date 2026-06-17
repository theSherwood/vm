# Durable Domains — Snapshot / Restore / Clone

> **Status: Phase 1 + snapshot codec landed; Phases 2–4 ahead.** This file is the
> single source of truth for the *design and implementation status* of durable
> domains. Built so far: the `svm-durable` IR→IR transform (arbitrary single-vCPU
> CFGs), the `svm-interp` handle-table durability primitives (§12.5), and the
> `svm-snapshot` artifact codec (§12 container + window image + handle table + the R5
> identity gate). The master design is `DESIGN.md` (D-notes, §-sections); the project
> status/pickup doc is `HANDOFF.md`. Keep all three in step — if code and a doc
> disagree, fix one of them in the same change (per `AGENTS.md`).
>
> Proposed decision: **D60** (D59 is currently the last). See bottom of file.

A "durable" domain can be quiesced, serialized to `(window pages + prots, shadow
control state, handle table)`, and later restored to bytewise-equivalent execution —
possibly on the other backend, possibly on a different host. The artifact must
survive: a recompile, a Cranelift version bump, ASLR, and JIT↔interp migration
(see §1 for the precise meaning of "survive a recompile" — it is narrower than it
sounds).

---

## 0. Orientation — how this lands on the existing VM

Grounding the proposal in what already exists (verified against the tree):

- **IR shape is ideal for the codec.** `svm-ir` is a flat CFG of block-local typed
  SSA with explicit block params and **no phi nodes** (`crates/svm-ir/src/lib.rs`
  `Block { params, insts, term }`). So resume-point liveness is *free*: a
  continuation block's `params` already are the live set, and the verifier
  (`svm-verify`) does no liveness/dominance analysis — it is a single linear forward
  type pass.
- **Dispatch primitive exists.** `Terminator::BrTable` (verifier-constrained: valid
  well-typed arm or trapping default, checks in `crates/svm-verify/src/lib.rs`) is
  exactly the rewind dispatch we need. No new instruction.
- **Suspension is explicit IR.** `Suspend`, `ContResume`, `ContNew`, and `CapCall`
  are real instructions; `Func::uses_concurrency()` already scans for them.
- **Memory substrate is close.** `svm-mem` owns the window; page protections
  (`PageProt`) and bulk snapshot-read (`read_into`, `SNAP_CAP`) already exist for the
  escape-oracle. **Restore (write pages back + re-establish prots) does not exist yet
  and is new escape-TCB code** — see §6/§9.
- **Nesting is real.** A child window is a power-of-two sub-range of the parent's via
  `Window::sub()` (`crates/svm-mask/src/lib.rs`). This is the §4 subtree.
- **The oracle is production machinery.** `crates/svm/tests/jit_diff.rs` and
  `fuzz/fuzz_targets/diff.rs` already run every program on interp and JIT and assert
  equivalence. The new snapshot property (§7) plugs straight in.
- **Tooling-tier precedent.** `svm-text` is a non-TCB crate depending only on
  `svm-ir`. The transform pass follows the same pattern (+0 TCB).

---

## 1. Goal & non-goals

**Goal.** Capture a running durable domain into a backend-independent,
recompile-survivable artifact and resume it later.

**What "survives a recompile" precisely means** (this needs to be exact — see
Risk R5):
- **Backend recompile / Cranelift bump / ASLR / JIT↔interp:** yes. The suspended
  state is IR-level and references no native address, register, or compiled code.
- **Re-running the *transform* (different block-splitting → different resume-id
  numbering):** **no, not automatically.** The shadow-stack schema is a function of
  the instrumented module's structure. The artifact is therefore *backend-portable*
  but *coupled to a specific instrumented-module identity*. The snapshot format must
  carry the instrumented-module hash; restore requires the same instrumented module.
  (This is asyncify's "can't thaw into a differently-compiled binary.")

**Non-goals.** Snapshotting non-durable domains (they pay nothing — §6). Capturing a
native stack as bytes (dies on relocation/recompile — §2). A built-in scheduler or
M:N runtime (orthogonal; honours D22/D56 — the VM ships *mechanism, not a
scheduler*).

---

## 2. Mechanism — IR-level freeze/thaw (the codec)

The native stack is a continuation in the least durable schema possible: *these exact
addresses in this exact build*. So we never serialize it. Instead a **durable** domain
is compiled through one IR→IR pass that lets each fiber flatten itself into
guest-resident, IR-level state and rebuild itself from it. The native stack remains
the runtime suspension mechanism (scheduling, fault-driven yield, hot suspend); the
transform is only the **codec** for `fiber.freeze` / `fiber.thaw`.

**The transform** (output is ordinary verifier-passing IR; no new instructions):

- **State word** (per vCPU, in-window): `NORMAL | UNWINDING | REWINDING`.
- **Shadow stack** (in-window): per frame, a *resume id* (small int enumerating a
  function's resume points) + the values live across that point.
- **Resume points = block heads.** Split each block after a may-suspend call; the
  continuation block's params *are* the live set (block-local SSA → liveness is
  explicit, no analysis).
- **Unwind:** after each may-suspend call, `if UNWINDING → spill continuation block's
  args to the shadow frame, push resume id, return`. Propagates out to the host.
- **Rewind:** function prologue, `if REWINDING → br_table over resume blocks`; each
  arm reloads its params from the shadow frame and re-issues the in-flight call.
  Dispatch is the existing, verified `br_table`.

**Freeze** = host sets `UNWINDING`, drives every fiber (suspend sites are resume
points) until all native stacks are empty. **Thaw** = restore memory, set
`REWINDING`, re-enter; the stack rebuilds itself through verified code.

**Why not host-side frame capture (annotate existing stacks).** Capture is feasible
(FP-walk + call-site stack maps decode frames into the interp `Frame`). Restore is
not: native re-entry stubs that rebuild each frame *are* asyncify's rewind, but
implemented in `svm-jit` with per-arch unsafe, *outside the differential oracle*. The
transform puts the same logic in verified IR, inside the oracle. (Full comparison: §8.
Note: the rejection rests on "per-arch unsafe outside the oracle," **not** on D56 —
see §8 for why the earlier D56 framing was wrong.)

---

## 3. Security

The shadow stack holds **IR-level tokens only** — never a native address. Adversarial
writes to it reduce to guest-harms-guest, already conceded by the §2a threat model:

| Guest tampering | Outcome |
| --- | --- |
| Forge a resume id | `br_table` is verifier-constrained: lands on a valid, well-typed arm in the same function, or the trapping default. Wrong data or trap — **never a control escape.** |
| Corrupt saved values | Garbage in well-typed slots — a wild store could already do this. |
| Forge the state word | Spurious self-unwind / broken self-rewind — self-DoS. |

This is the `call_indirect` story exactly: the guest already keeps control-adjacent
state (function-table indices, the in-window data stack) in its window, and the answer
is **masked, verified dispatch** — not memory integrity.

**Why this is +0 to the security argument.** Per `DESIGN.md` §4/D38, the escape hinge
is the **confinement-masking lowering** (`svm-mask`), not the verifier. The shadow
stack is ordinary guest memory, so its stores/loads go through that same masked path
as any guest access — the existing hinge already covers it. The verifier still secures
typing/control-flow/index-ranges of the instrumented IR. A transform bug is a
*correctness* bug, never a confinement bug.

Corollary: restore never crosses a trust boundary as structured data (host loads
opaque bytes and calls the entry), unlike host-side frame capture, whose restore path
is a parser over attacker-controlled frames in the host.

---

## 4. Unit of durability

- **Instrumentation unit = the module** (a compile-mode flag). Includes `Jit`-cap
  units (`DESIGN.md` §22): the host runs the pass on submitted IR before
  verification, so guest-driven JIT composes for free.
- **Snapshot unit = the domain, closed over its nesting subtree (§14).** State lives
  in the domain (window, vCPUs/fibers, handle/dispatch tables); a child's window is a
  power-of-two sub-range of the parent's (`Window::sub`), and a fault-suspended child
  can only be drained-then-unwound if its code is instrumented.

**Enforcement (one flag check at instantiate/install):** *a durable domain admits
only freezable modules and may only spawn durable children.* STW quiesces the subtree
as a unit.

**Open edge (R4):** cross-tree sharing (`SharedRegion`, `DESIGN.md` §13; in-flight
durable-sibling comms) forces co-snapshot of the sharing group or journaling at the
shared edge (consistent-cut). Decide as a `SharedRegion` constraint: either a durable
domain can't share outside its subtree, or regions carry a snapshot protocol. This is
the only place the unit-of-durability question has a real design consequence.

---

## 5. STW & the non-instrumented residue

Two states can't be translated and are **drained**, not decoded:

- **Fault-suspended fiber** (parked at an arbitrary PC, demand-paged coroutine):
  supply the page, let it run to the next poll site, then it unwinds.
- **vCPU inside a host `cap.call`:** let the call return, then unwind.

So freeze = cooperative STW: request unwind, wait for quiescence-at-safepoints (the
Go/JVM shape). The drain protocol is **host-side and identical on both backends** — no
codegen, no native-stack decode. Snapshot latency is bounded by the longest host call
**plus the longest poll-free code path** (until back-edge polls land in phase 4 — see
R6); needs a cancellation story for `Blocking.work` before latency guarantees are
tight.

---

## 6. Performance

**Non-durable modules: zero, structurally.** The pass runs only on request; an
uninstrumented module's bytes, verification, and codegen are byte-identical to today.
No always-on safepoint infra, no global regalloc constraint, no metadata sections.

**Durable modules:**

| Cost | When | Estimate |
| --- | --- | --- |
| Poll (load+cmp+branch on state word) | after may-suspend calls; (later) back-edges | epoch-interruption shape; low single-digit %, worst case 10–30% call/loop-dense |
| Code size | cold-path dispatch + spill/reload blocks | +50–100% in instrumented functions (icache, not host binary) |
| Spill/reload | only on actual unwind | snapshot frequency, not exec frequency |

**Key mitigation — may-suspend analysis:** only calls that transitively reach a
`cap.call` (conservatively: any indirect call) get polls; only functions on such paths
get instrumented. *(Phase-1 status: only **directly** `cap.call`-bearing functions are
instrumented — the transitive analysis arrives with call-chain propagation.)*

**Cost unit is the state-word load, not the branch.** In `NORMAL` state the poll
branch is perfectly predicted; the real cost is the `i32.load` of the state word. That
word lives **in the guest window** (a masked load each poll) deliberately — so the
window snapshot captures it for free (§12.0). A register/host-side state word would be
faster but needs separate capture: that's the main perf lever if `NORMAL`-state
overhead ever shows up on `svm-bench`. **Non-durable modules pay none of this** — the
pass is opt-in and no runtime/TCB crate depends on `svm-durable`.

**Caveat on "pure compute untouched":** the conservative rule treats *any indirect
call* as may-suspend, and `call_indirect` is the normal lowering for C function
pointers / vtables. So "untouched" holds for **direct-call** compute (sha256/perlin/
xxhash shapes); function-pointer-heavy C still gets instrumented. The 10–30% worst
case may be more common than "compute is free" implies. *Validate by running the pass
over `svm-bench` + demos — the harness makes this ~a day.*

---

## 7. Backend equivalence

Both backends run the **same instrumented IR**; the suspended representation is
IR-level, so the artifact references no native address, recompiled code, or register
layout. Consequences: snapshots are **backend-portable** (freeze under JIT, thaw under
interp), and the existing generative fuzzer **proves** it via one new property:

> for any valid module and any snapshot point:
> run-to-snapshot → serialize → restore → run-to-end  ≡  uninterrupted run

checked on interp, on JIT, and cross-backend (extends `crates/svm/tests/jit_diff.rs`
and `fuzz/fuzz_targets/diff.rs`). Equivalence is continuously tested, not asserted.
The §5 residue is drained identically on both backends, so no backend ever decodes a
native stack.

---

## 8. Alternatives considered

| Path | Capture | Restore | Complexity lands in |
| --- | --- | --- | --- |
| **Freeze/thaw transform (chosen)** | guest unwinds itself | guest rewinds itself | verified IR, both backends, oracle-checked |
| Annotate existing stacks (B-lite) | FP-walk + call-site maps | native re-entry stubs (≈ asyncify, in JIT) | per-arch unsafe, outside oracle |
| CRIU-lite (pin code arena + stacks) | memcpy | memcpy | host-heap pointer aliasing; same-binary only — not durable in any useful sense |

**Correction to the original draft:** an earlier version of this argument said
host-side capture "re-opens the unsafe class D56 evicted." That is inaccurate. **D56**
removed a *built-in M:N green-thread executor*, whose highest-risk unsafe was *fiber
migration across OS threads in the runtime TCB* — not a per-arch stack-unwind unsafe
class. Moreover **D57 deliberately re-adopted** migratable-fiber unsafe ("with eyes
open") as a primitive. So host-side capture is rejected on its *own* merits —
**per-arch unsafe, outside the differential oracle** — not because D56 forbade it.

Why the transform is *small here specifically*: Binaryen's asyncify is hairy because
of wasm's structured control flow + locals model + interprocedural liveness. This IR
is a flat CFG of block-local SSA with explicit block params, so resume-point liveness
is free, splitting is mechanical, and dispatch reuses `br_table`. The pass is the only
transform-specific work; everything else in §9 is needed by *any* snapshot design.

---

## 9. Implementation plan & status

New non-TCB crate (tooling tier, like `svm-text`) for the pass; thin plumbing
elsewhere. Net ~1.5–3k lines.

- **TCB impact:** the **pass itself is +0 TCB** (tooling tier; an embedder running
  pre-instrumented modules links none of it). **But phase 2 adds a small escape-TCB
  surface** — page+prot *restore* lives in `svm-mem`, which is escape-TCB. Honest
  accounting: +0 TCB for the codec, +small escape-TCB for the restore path (covered
  by the oracle).

**Sizing:** ~4–8 weeks to a v1 (cap.call-boundary snapshots, MVP-powerbox handles,
restore-on-either-backend). Variance concentrates almost entirely in **phase 3**
(concurrency/quiesce vs. the D57 migratable-fiber ownership protocol) and in fuzz
findings; the transform itself is the *most* predictable piece. Phase 1's
predictability should not be read as overall low risk.

**Before phase 1:** write the one-page snapshot-format + handle-durability spec so the
fuzz property has a stable target. The format **must** include the instrumented-module
hash (see §1 / R5). Scope v1 handle durability to re-grantable handles only.

### Phase tracker

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

- **[x] Phase 0 — Spec.** Snapshot-format + handle-durability spec **complete in
  §12** (D-scope/D-hash/D-region resolved). Format carries instrumented-module
  digest; v1 = re-grantable handles only.
- **[~] Phase 1 — Transform + interp round-trip. Go/no-go: PASSED.** The
  freeze→serialize→restore→thaw round-trip works on the real interpreter
  (`crates/svm-durable`, `tests/roundtrip.rs`): an in-window shadow stack + state
  word + `br_table` rewind reconstructs a frozen single-vCPU domain bytewise, and the
  thawed run reloads the saved `cap.call` result rather than re-issuing it.
  - **Landed:** the `svm-durable` tooling-tier crate (+0 TCB, depends only on
    `svm-ir`); the IR→IR transform — now covering **arbitrary single-/multi-block CFGs**
    (branches, loops, joins) with **any number of resume points** across **call chains**
    (leaf `cap.call` reload vs. propagated `Call` re-issue, R8); the §12.7 frame layout;
    round-trip + inert-instrumentation + verifier tests (`tests/roundtrip.rs`,
    `chain.rs`, `multipoint.rs`, `multiblock.rs`), plus the interp (`durable_fuzz`) and
    cross-backend interp-vs-JIT (`durable_jit`) generative properties over a generator
    that emits multi-frame, multi-point, multi-block modules.
  - **Phase-1 transform complete.** The structural extensions (call-chain propagation,
    multiple resume points, multi-block CFGs) plus the **minimal live-set** spill
    (block-local liveness; ~28–40% smaller instrumented IR and up to ~57% less JIT
    compile time on spill-heavy guests, `tests/durable_bench.rs`) are **done**. Out of
    scope and rejected/ignored: `call_indirect` (and indirect tail calls) to may-suspend
    targets; direct tail calls into may-suspend callees; guest linear-memory use (R9).
  - **Hazards introduced by the as-built transform: R8–R11 (§11).** R9 is **placement,
    not isolation**: the durable region is a budget-accounted reserved slice `[0,
    DURABLE_RESERVE)` of the domain's own window (guest memory above it, wasm
    `__heap_base`-style). Memory-using guests work via `transform_module_assume_confined`
    on a cooperating-toolchain contract; corruption is self-contained and fails safe.
    Hard isolation against adversarial guests (guard-paged §12.7) is optional
    defense-in-depth.
  - **Snapshot artifact codec + handle durability landed.** The `svm-interp` handle-table
    durability primitives (§12.5) and the `svm-snapshot` §12 container — header w/ R5
    digest, sparse zero-eliding window image, Section-3 handle table — now give a real
    `freeze → bytes → restore → thaw` on the interpreter (`crates/svm-snapshot`), with the
    §12.6 canonical + identity-gated invariants tested.
- **[~] Phase 2 — JIT parity + real memory snapshot.** Same instrumented IR on JIT (the
  `durable_jit` cross-backend property already holds); **artifact codec done** (above).
  Remaining: `svm-mem` page+prot snapshot/**restore** for protected/large windows (the
  codec's flat zero-eliding image covers the Phase-1 flat window), and routing the codec
  through the cross-backend §7 property. Risk: low (oracle does the work); Windows
  placeholder semantics the known annoyance. *(escape-TCB touch — restore.)*
- **[ ] Phase 3 — STW + multi-vCPU + fiber freeze/thaw.** Cooperative quiesce, drain
  residue, freeze/thaw choreography against the D57 migratable-fiber ownership
  protocol. **Highest risk** — concurrency seam (loom-check, like the futex glue).
- **[ ] Phase 4 — Back-edge polls, handle hardening, CoW clone.** Latency +
  durability quality + cheap clone. Incremental, off critical path.

---

## 10. Clone

Falls out of the same machinery at a quiescent point: window copy (CoW via
`memfd` + `MAP_PRIVATE` for cheap) + dispatch-table rebuild + handle re-grant. No
extra mechanism beyond snapshot/restore.

---

## 11. Risk register / open questions

| # | Risk / question | Where | Status |
| --- | --- | --- | --- |
| R1 | Phase-3 quiesce vs. D57 migratable-fiber single-owner protocol (a fiber may be mid-migration / owned by another OS thread at safepoint request). The crux of the schedule variance. | §5, §9 | open |
| R2 | `Blocking.work` cancellation needed before snapshot-latency guarantees are tight. | §5 | open |
| R3 | escape-TCB growth from the page+prot **restore** path in `svm-mem`. | §6, §9 | open |
| R4 | `SharedRegion` cross-tree sharing: co-snapshot the sharing group, or regions carry a snapshot protocol? Decide as a `SharedRegion` constraint. | §4 | open |
| R5 | Snapshot-format identity: artifact is coupled to the *instrumented-module* hash, not just backend-independent. Must be pinned in the format. | §1, §9 | open |
| R6 | v1 latency bound includes "longest poll-free path" until back-edge polls (phase 4); a tight direct-call compute loop is un-preemptable in v1. | §5, §6 | open |
| R7 | Breadth of instrumentation: "any indirect call = may-suspend" instruments more ordinary C than "compute is free" suggests. Validate on `svm-bench`. | §6 | open |

**Phase-1 implementation hazards** (introduced by the `svm-durable` transform as built;
the transform *fails closed* — out-of-scope shapes return a `TransformError` rather
than miscompiling, so these are latent/extension hazards, not silent-miscompile bugs):

| # | Risk / question | Where | Status |
| --- | --- | --- | --- |
| R8 | **Call-chain propagation landed; deepest-frame assumption resolved.** The transform now instruments any may-suspend function (transitive `cap.call` closure over the direct-call graph) whose single block suspends on one op: a leaf `cap.call` (reload result + flip `NORMAL`) **or** a propagated `Call` (reload pre-call live set + **re-issue the call**, leaving the state `REWINDING` so the callee rewinds). Real multi-frame stacks; only the innermost leaf flips to `NORMAL`. Covered by `tests/chain.rs` (2-/3-level chains, live-value-across-call) and the generator now emits depth-`1..=4` chains, so the interp (`durable_fuzz`) and cross-backend (`durable_jit`) properties exercise it. **Multiple resume points** and **multi-block CFGs** (branches, loops, joins) now land too — each block is split at its suspend ops, branch targets are remapped, and a global `br_table` dispatch routes the thaw (`tests/multipoint.rs`, `tests/multiblock.rs`; the generator emits multi-frame/multi-point/multi-block modules). Out of scope: `call_indirect`/indirect tail calls to may-suspend targets (treated non-suspending); direct tail calls into may-suspend callees (rejected). A chain deeper than the reserve holds traps cleanly on freeze (R9 overflow guard), rather than overflowing. | §2, §12.7, `svm-durable` | addressed (Phase-1 scope) |
| R9 | **Placement, not an isolation boundary — cheap for MVP.** The control state + shadow stack are a reserved low slice `[0, DURABLE_RESERVE)` (one 64 KiB page) of the domain's *own* window; guest memory is `[DURABLE_RESERVE, window)`, part of the same budget-accounted allotment (the wasm shadow-stack / `__heap_base` convention). Because the window is per-domain and runtime-masked, a guest that writes the reserve corrupts only **its own** durability — never another domain or the host — and it **fails safe**: a forged resume id hits the `br_table` default → `Unreachable`; a wild shadow-SP stays masked in-window; the host validates the artifact (module hash) on restore. **MVP path:** `transform_module_assume_confined` instruments memory-using guests on the cooperating-toolchain contract that the guest's data/heap is based at `DURABLE_RESERVE` (`tests/guest_memory.rs` shows guest memory round-tripping). Strict `transform_module` still fails closed (`GuestUsesMemory`) for untrusted modules. **Optional defense-in-depth (not MVP):** hard isolation against an *adversarial* guest — guard-paged per-fiber placement (§12.7) or per-access confinement. The shadow stack now **traps on overflow**: the freeze-path `UNWIND` check refuses a push whose top would cross `DURABLE_RESERVE`, so a too-deep call chain fails safe (a clean trap) instead of growing into guest memory (`tests/overflow.rs`). See **[DECISION D-shadow-overflow]** below for why this lives in the transform rather than a unified backend recursion ceiling. | §12.7, `svm-durable` | mitigated (placement + fail-safe + overflow trap; hard isolation optional) |
| R10 | **No concurrency protection on the in-window control state** (state word, shadow-SP). Fine at single-vCPU; a hazard once fibers/multi-vCPU arrive (relates to R1, but specifically about the control words racing). | §3, §12.7 | open |
| R11 | **Equivalence now fuzzed (Phase-1 scope), both single-backend and cross-backend.** The §7/§12.6 property runs over a generator of **in-scope** durable modules: (a) interpreter-only — *inert in `NORMAL`* (instrumented == un-instrumented) and *round-trip* (freeze→serialize→restore→thaw ≡ uninterrupted, reload-not-reissue) — `crates/svm-durable/tests/durable_fuzz.rs` + libFuzzer `fuzz/fuzz_targets/durable.rs`; (b) cross-backend — interp vs Cranelift JIT agree on the NORMAL result, leave a **byte-identical freeze artifact**, and a JIT thaw of the **interpreter-frozen** artifact under a different host clock reproduces the result — `crates/svm/tests/durable_jit.rs` + libFuzzer `fuzz/fuzz_targets/durable_jit.rs`. Both stable drivers run in CI without nightly. Coverage broadens automatically as the transform generalizes (R8). | §7, §12.6 | addressed (Phase-1 scope) |

**[DECISION D-shadow-overflow — RESOLVED: freeze-path guard in the transform, not a unified backend recursion ceiling.]** The shadow stack mirrors the call stack (one frame per suspended activation), so it can only overflow the reserve if the call stack is very deep. We bound it with a check on the freeze-path `UNWIND` (trap if a push would cross `DURABLE_RESERVE`) rather than forcing both backends to a common call-depth ceiling. Rationale: shadow overflow is a **tooling-tier** concern (`svm-durable`, +0 TCB), and the guard sits on the **cold** freeze path, so it costs nothing on the per-call hot path; unifying the ceiling would mean an **escape-TCB JIT codegen change** (the JIT has no depth counter today — recursion rides the native stack; the interp caps at `MAX_CALL_DEPTH = 256`) with a permanent per-call cost, to fix an edge case. Consequence: a domain recursed deeper than the reserve holds simply **cannot be frozen** (the freeze traps) — a safe, coherent limitation. Cross-backend recursion *determinism* (interp 256 vs JIT native-stack) remains a separate, latent, un-exercised divergence; unifying the ceiling is the deliberate fix to make **on its own merits** if/when it matters (the overflow guard then becomes a redundant cheap backstop).


---

## 12. v1 snapshot format & handle durability (Phase 0 spec)

> **Status: spec'd (Phase 0 complete).** This is the stable target the §7 fuzz
> property is written against. All three open decisions are **RESOLVED** (D-scope,
> D-hash, D-region), flagged inline.

### 12.0 What is and isn't guest state

The transform (§2) keeps the **state word and the shadow stacks in the window**.
So a quiesced durable domain is described almost entirely by its **window image** —
the shadow stacks, spilled live values, and per-vCPU state words are all guest-
resident bytes. At a safepoint every native stack is empty and every register-
resident value has been spilled to a shadow frame (in-window). What remains
*host-side* and must be captured separately is small:

1. the **set** of vCPUs and fibers and their relationships (not their stacks),
2. the §3c **dispatch table** (`DomainTable`, `call_indirect` slots),
3. the **handle table** (`Host::table` — authority, not the resources it names).

**[DECISION D-scope — RESOLVED: guest + authority only.]** A v1 snapshot does *not*
capture host-side resource state — `Host::stdin`/`stdout`/`stderr` buffers,
`clock_ns`, the offload pool, async rings. Restore re-grants the *authority* (the
handle) and the restoring embedder supplies fresh resources behind it. Rationale:
that state is host-environment, not guest, and capturing it would pull arbitrary host
objects into the artifact.

### 12.1 Container

A sectioned binary, LEB128 varints, same conventions as `svm-encode`. Sections are
TLV (`tag: uleb`, `len: uleb`, body) so a restore-side reader can skip unknown tags
(forward-compatible). **Canonical form is required** — sparse entries ascending by
index, no redundant entries, fixed varint widths — so "re-serialize after restore at
the same point is byte-identical" is a plain `==`, which is what the fuzz property
needs.

### 12.2 Section 0 — Header

| Field | Type | Notes |
| --- | --- | --- |
| magic | `b"SVMD"` | SVM-Durable |
| format version | u16 | bump on incompatible change |
| instrumented-module digest | 32 bytes | digest of the `svm-encode` bytes of the **instrumented** module (R5). Restore refuses on mismatch — this is the durability boundary from §1. |
| window geometry | `reserved_log2: u8`, `mapped: u64` | matches `Module::memory` / `svm_mask::Window`; stored for a fail-fast check |
| host page size at capture | u32 | page granularity of §12.3 |
| vCPU count, fiber count | uleb, uleb | sizes §12.4 |

**[DECISION D-hash — RESOLVED: non-cryptographic 256-bit hash, +0 deps.]** Identity =
the encoded instrumented-module bytes; the header stores a 256-bit non-cryptographic
digest of them. This guards *accidental* restore-into-wrong-module mismatch, not an
adversary (a guest can't forge its way past confinement here — §3), so no crypto-hash
dependency is added to the toolchain crate. The digest function is a snapshot-format
detail; pin the exact one in the implementing crate.

### 12.3 Section 1 — Window image (sparse)

Captured at the quiescent point. Sparse over **committed** pages, with zero-page
elision. Per entry:

- `page_index: uleb` (window offset ÷ page)
- `prot: u8` — `Rw=0, Ro=1, Unmapped=2` (mirrors `PageProt`, `svm-interp` `:5962`)
- if `prot ∈ {Rw, Ro}`: page bytes (run-length / zero-eliding to keep it small)

The in-window shadow stacks + state words ride along in this image for free (§12.0).

**[DECISION D-region — RESOLVED: no `PageProt::Backed` in v1.]** §13 `SharedRegion`-aliased pages
name a host backing shared across the nesting tree — that's the cross-tree-sharing
edge (R4). v1 **freeze refuses** if `Mem::has_regions` is set for any domain in the
subtree. (Lifting this is the R4 work: co-snapshot the sharing group.)

*Optimization (not v1):* diff against the post-instantiation image (`Module::data`
segments) instead of storing all committed pages. Correctness doesn't need it.

### 12.4 Section 2 — Control state

Native stacks are gone (drained, §5), so per-vCPU register/stack state is empty by
construction. What's stored:

- **Per vCPU:** logical id + role (root vs `thread.spawn` child). Re-entry on thaw is
  `REWINDING` re-entry; the shadow stack (in-window) drives the rebuild.
- **Per fiber** (`ContNew`'d): its handle value `(generation, slot)` so guest-held
  fiber/funcref handles stay valid across restore; its in-window shadow-stack
  location; and `suspended | runnable` status. The pending `Suspend`/`ContResume`
  value is already spilled in-window at the resume point, so it is *not* stored here.
- **Dispatch table** (`DomainTable`, `:984`): the `call_indirect` slot contents as
  funcref indices (small ints into module funcs). **v1 stores plain module funcrefs
  only**; installed guest-JIT native funcrefs are not durable (consistent with the
  `JitDomain`/`JitCode` exclusion in §12.5).

### 12.5 Section 3 — Handle table (durability classification)

Per **live** slot (`Slot.entry.is_some()`, `svm-interp` `:4427`), sparse:

- `slot_index: uleb`, `generation: u32`, `type_id: u32`, durable binding descriptor.

**Durable (re-grantable) in v1** — entire state is value-typed:

| `Binding` | Stored | Re-grant path |
| --- | --- | --- |
| `Stream(role)` | role | `grant_stream` |
| `Exit` / `Clock` / `Memory` / `Yielder` | — | `grant_exit`/`grant_clock`/`grant_memory`/`grant_yielder` |
| `AddressSpace { base, size }` | base, size | `grant_address_space` |
| `Instantiator { base, size }` | base, size | `grant_instantiator` |

**Not durable in v1** — carry out-of-line host state or native pointers; their
presence in a live, non-drainable state makes the subtree non-snapshottable, so
**freeze refuses** unless they're closed/drained first:

`SharedRegion(u32)` (R4), `Module(u32)`, `IoRing(u32)` (drain residue §5),
`Blocking(u32)` (§5 + cancellation R2), `JitDomain(u32)`, `JitCode{domain,unit}`.

**Generation/slot pinning.** Restore must reinstate the **same `(slot, generation)`**
so guest-held handle values stay valid — the auto-allocating `grant`/`grant_*`
(`:4858`+) advance generation and pick a slot. v1 adds one host helper,
`grant_at(slot, generation, type_id, binding)`, that pins both. (`Host` is not
escape-TCB; the verifier/mask hinge is untouched — §3.)

**Status: Host primitives landed.** `svm-interp` now implements the §12.5 classification
and pinning on `Host` (`crates/svm-interp/tests/handle_durability.rs`):
`capture_durable_handles() -> Result<Vec<DurableHandle>, NonDurableHandle>` (the
re-grantable set in ascending slot order, or a clean refusal naming the first non-durable
slot — freeze is all-or-nothing), `restore_durable_handles` + the `grant_at` pin, and
`handle_capacity()` for the codec's bounds check. The value-typed descriptors
(`DurableBinding`/`DurableHandle`) are public; `Binding` stays private. The byte-level
**Section 3** serialization is now wired into the `svm-snapshot` container (§12.6 below).

### 12.6 Round-trip / equivalence contract

The format exists to make this testable (extends §7, `jit_diff.rs` / `fuzz/diff.rs`):

> freeze → serialize → (drop domain) → restore → run-to-end  ≡  uninterrupted run,
> on interp, on JIT, and cross-backend.

Two derived invariants the fuzzer checks directly:
1. **Canonical:** re-serializing a freshly-restored domain at the same safepoint is
   byte-identical to the original artifact (§12.1).
2. **Identity-gated:** restore against a mismatched instrumented-module digest
   refuses cleanly (never partial state) — R5.

**Status: codec landed (single-vCPU Phase-1 shape).** `svm-snapshot` (tooling-tier, +0
TCB; depends on `svm-ir`/`svm-encode`/`svm-interp`, **not** `svm-durable`) implements the
§12 container: `freeze(module, window, host) -> Vec<u8>` and `restore(artifact, module,
&mut host) -> window`. Header carries the 256-bit non-crypto instrumented-module digest
(D-hash); the window image is sparse with zero-page elision (the shadow state rides along)
and carries **per-page protection** (`Rw`/`Ro`/`Unmapped`, §12.3) — `freeze_with_prots` /
`restore_with_prots`, with the flat `freeze`/`restore` treating the window as all-`Rw`
(`tests/prots.rs`); Section 3 is the handle table. `crates/svm-snapshot/tests/roundtrip.rs` drives the real
freeze→serialize→restore→thaw on the interpreter and asserts both invariants above plus the
non-durable freeze refusal. The **cross-backend** property (`crates/svm/tests/durable_jit.rs`
+ the libFuzzer `durable_jit` target) now runs through the codec too: it serializes each
backend's freeze and asserts a **byte-identical artifact** across interp/JIT, checks the
canonical re-serialize invariant, and thaws the **restored** interpreter artifact on the JIT.
**Capture + re-establish** landed for the interpreter: `run_capture_reserved_with_host_prots`
both **seeds** an initial per-page protection map (restore) and **returns** the post-run map
(freeze) — `CapturedProt` (`Rw`/`Ro`/`Unmapped`/`Backed`) at the fixed `DURABLE_SNAPSHOT_PAGE`
(= codec `PAGE`) granularity. `crates/svm/tests/durable_prot_capture.rs` shows a D40 `readonly`
data segment captured as `Ro` and surviving freeze→restore through the codec (where Phase-1's
flat all-`Rw` image would have lost it), **and** that re-establishing the map on a thawed run
makes a write to a restored `Ro` page fault — while the same window without it writes through. A
`Backed` page maps to a freeze refusal / is skipped on restore (D-region: the embedder re-grants
the region). Still ahead (escape-TCB): the **JIT** side — capturing `GuestWindow` protections and
re-establishing them via `mprotect`/`VirtualProtect` (Windows placeholder semantics); then §12.4
fiber/dispatch control state.

### 12.7 Shadow-frame layout

The transform's spill/reload code and the suspended representation meet here. Two
properties drive the whole design:

- **The shadow stack is in-window**, so the §12.3 window image captures it verbatim.
  The serializer never walks frames — it copies the byte range `[base, shadow_SP)`
  and records the extent. Frame *internals* are re-interpreted only by the same
  instrumented code on thaw, so the frame need only be self-consistent for **rewind**,
  not for a generic external reader.
- **Resume-point liveness is the continuation block's params** (§2), whose types are
  statically known per resume id. So a frame stores *raw value bytes only* — never
  type tags; the resume id selects the layout.

**Stacks per fiber (D39/D41 extended).** A non-durable fiber owns the D41 *pair*:
out-of-band control stack (native, not serialized) + in-window guard-paged data stack
(data-SP). A **durable** fiber owns a *triple* — add an in-window, guard-paged,
quota-charged **shadow stack** (shadow-SP), swapped alongside the others on fiber
switch. The shadow stack is allocated **only under instrumentation**, so non-durable
modules keep the pair and pay nothing (§6).

**Frame format** (grows upward; `shadow_SP` points just past the live top frame):

```
  ┌─ frame base (16-byte aligned) ───────────────────────────┐
  │ live values, packed in continuation-block param order:   │
  │   i32/f32 → 4B   i64/f64 → 8B   v128 → 16B (nat. aligned) │
  │ … pad to keep the resume id in the top word …            │
  │ resume_id : u32        ← always the top 4 bytes of frame  │
  └──────────────────────────────────────────── shadow_SP ───┘
```

`resume_id` lives at a **fixed offset from `shadow_SP` (`−4`)** so rewind can read it
*before* knowing the frame size — which resolves the circularity (frame size depends
on resume id depends on reading the frame). `resume_id = 0` is reserved ("no in-flight
resume"). Frames are 16-byte aligned (v128). Per-resume-id frame size is a transform
compile-time constant; nothing stores it.

**Unwind (freeze), after a may-suspend call, if `UNWINDING`:** push live values, push
`resume_id` on top, `shadow_SP += frame_size(rid)`, `return` (propagates out to host).

**Rewind (thaw), function prologue, if `REWINDING`:**
`rid = load_u32(shadow_SP − 4); br_table rid` → arm reloads its params from the known
offsets, `shadow_SP −= frame_size(rid)`, then:
- if `shadow_SP == base` (this was the deepest frame — the actual safepoint): flip the
  state word to `NORMAL` and continue forward from the resume point;
- else: re-issue the in-flight call (which re-enters the callee, whose own prologue
  sees `REWINDING` and pops the next frame).

**State word** (`NORMAL | UNWINDING | REWINDING`): per-vCPU, in-window (§2); every
poll/prologue reads it. Freeze sets all to `UNWINDING` and drives each fiber to drain
its native stack into its shadow stack; thaw sets `REWINDING` and re-enters.

**Host-side control state (§12.4) per durable fiber** therefore reduces to: the
shadow-stack region's window offset + `shadow_SP` extent (the bytes themselves are in
the window image). Optional integrity aids (a per-frame `func_id` tag checked on pop)
are *recommended in checked builds* but not normative — correctness needs only
`resume_id` + values.

> **Iteration note.** The exact intra-frame padding and the deepest-frame flip are the
> parts most likely to shift once the Phase 1 transform is real; the
> resume-id-at-`SP−4` rule and the in-window triple-stack placement are the load-
> bearing commitments and should be stable.

---

## Proposed decision record

> **D60 (Proposed). Durability via an IR-level freeze/thaw transform, not native-stack
> capture.** Durable domains compile through an opt-in IR→IR pass that flattens fibers
> into guest-resident, verifier-checked control state; snapshots are
> `(window, shadow state, handles)`, backend-portable and surviving a backend
> recompile / Cranelift bump (but coupled to the instrumented-module identity — R5).
> Rejected: host-side frame capture (per-arch unsafe, outside the differential
> oracle) and CRIU-lite (same-binary only). The confinement-masking lowering stays the
> escape hinge (D38); the codec pass adds +0 TCB, the page+prot restore path adds a
> small escape-TCB surface in `svm-mem`; non-durable modules pay nothing.
