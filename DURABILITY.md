# Durable Domains — Snapshot / Restore / Clone

> **Status: Phases 1–2 landed + Phase 3.1 (single-fiber freeze/thaw) complete on the interpreter
> + Phase 3.3 (JIT parity) complete single-vCPU and multi-vCPU freeze + thaw + **slice 3.4 (full
> multi-vCPU scope: nested spawns + child-owned fibers, both backends, snapshot v4)**.** This file is the single source of truth for the *design and
> implementation status* of durable domains. Built so far: the `svm-durable` IR→IR transform
> (arbitrary single-vCPU CFGs **+ the §12 fiber control ops**), the `svm-interp` handle-table
> durability primitives (§12.5) **+ the per-fiber shadow-SP swap / freeze driver (D-fiber-cont
> option A)**, the `svm-snapshot` artifact codec (§12 container + window image + handle table +
> the R5 identity gate **+ Section-2 fiber residue**), and **per-page protection capture +
> re-establish on both backends** (Phase 2). A single-fiber domain round-trips
> `freeze → serialize → restore → thaw` end-to-end. The master design is `DESIGN.md` (D-notes,
> §-sections). Keep this doc and `DESIGN.md` in step — if code and
> a doc disagree, fix one of them in the same change (per `AGENTS.md`).
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

**Measured (interpreter, back-edge polls).** `cargo run --release --example
durable_overhead -p svm-durable` times a transformed loop vs the same loop
un-transformed (steady-state, large/small-`n` subtraction), plus freeze/thaw. On the
**tree-walking interpreter** the always-on back-edge poll costs **~+25–28 ns/iter**,
which is **+50% on a realistic arithmetic-body loop and ~+75% on a minimal-body loop** —
*higher* than the table's "10–30%" estimate, because (a) it's the worst case (a loop-only
back-edge poll, not a call-gated safepoint) and (b) the interpreter's baseline per-op is
already cheap, so a masked window load is a large *relative* add. (On the JIT, where the
baseline op is ~1 ns and the poll lowers to a register-friendly epoch check, the relative
tax should be smaller — that path is not yet measured here.) Freeze/thaw are dominated by
loop execution: freeze runs to the checkpoint with the countdown armed (heavier than the
inert poll) then unwinds + spills; **thaw rewinds the loop header to the checkpoint**, so
thaw cost grows with freeze depth — i.e. **checkpoint at shallow safepoints / loop
boundaries, not deep inside long-running loops.** The serialized image is the full
reserved window (here 256 KiB); the live loop-carried spill is a small prefix.

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
- **[x] Phase 2 — JIT parity + real memory snapshot.** Same instrumented IR on JIT (the
  `durable_jit` cross-backend property holds); **artifact codec done**; **per-page protection
  capture + re-establish landed on both backends, both directions** (§12.6 / `durable_prot_capture.rs`).
  The page-protection story is complete; a `Backed` (§13 shared-region) page stays out of scope
  (D-region). *(escape-TCB touch — restore.)*
- **[~] Phase 3 — STW + multi-vCPU + fiber freeze/thaw.** Cooperative quiesce, drain
  residue, freeze/thaw choreography against the D57 migratable-fiber ownership protocol.
  **Highest risk** — concurrency seam (loom-check, like the futex glue). **Design in §12.8**;
  **D-fiber-cont RESOLVED (option A).** **Sub-phase 3.1 (one interp fiber) is complete** —
  per-fiber shadow regions + shadow-SP swap, both thaw arms, the freeze driver, and the
  Section-2 residue codec give an end-to-end single-fiber `freeze → serialize → restore → thaw`
  on the interpreter (§12.8, `svm-durable/tests/fiber.rs` + `svm-snapshot/tests/roundtrip.rs`).
  Remaining: 3.2 multi-vCPU + per-context layout, 3.3 JIT parity (replicate the swap in the JIT
  fiber-switch path). (Dispatch table is a no-op — §12.4.) Single-vCPU durability is a coherent MVP.
- **[~] Phase 4 — Back-edge polls, handle hardening, CoW clone.** Latency +
  durability quality + cheap clone. Off critical path. **Slice A (async STW) landed:** 4A.1–4A.4
  (back-edge polls, JIT parity, async `request_freeze`, the loom quiesce model) and **4A.5
  per-context shadow-SP → genuinely-concurrent multi-vCPU STW freeze** (`FORMAT_VERSION` 4→5→6;
  the shared shadow-SP word + its lock retired), plus follow-ups **A** (a `thread.join` result
  survives a concurrent freeze), **B.1** (a concurrent child flattens its own fibers), the
  **blocked-in-`thread.join` freeze** (`thread.join` is now a may-suspend re-issue safepoint; a vCPU
  parked in a join unwinds and the thaw re-issues the join), and **B.2** (full nested concurrent
  spawns — a concurrent child can `thread.spawn` a grandchild; the per-OS-thread spawning-task source
  attributes the grandchild's `parent_task` correctly and the thaw rebuilds the nested topology).
  the **blocked-in-`thread.wait`** freeze (futex `atomic.wait` is now a may-suspend re-issue safepoint —
  bounded + fail-closed), and **B.1′** (a concurrent child-fiber caught *mid-resume-chain*, verified
  deterministically). **Next slices (all detailed under "Phase 4 Slice A.5"):**
  (1) **lift the `atomic.wait` thaw fail-closed** — a re-issued wait that would re-park needs the
  concurrent-thaw rework (run frozen waiters as real OS threads) to re-synchronize notify/wait;
  (2) **4A.6** recycled-context async freeze (sparse-residue payoff); **4A.7** parked-vCPU /
  `Blocking.work` latency; and the non-STW Phase-4 items — handle hardening (drainable non-durable
  bindings), CoW clone, `SharedRegion` consistent-cut (R4).

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
| R10 | **No concurrency protection on the in-window control state** (state word, shadow-SP). Fine at single-vCPU; a hazard once fibers/multi-vCPU arrive (relates to R1, but specifically about the control words racing). *Mitigated for slice 3.2.1:* a freeze/thaw run (state ≠ `NORMAL`) is forced **single-worker**, and the runtime swaps both control words per-vCPU per dispatch — so the words are never touched concurrently. A lock-free parallel STW for the shadow-SP is **planned via per-context SP** (4A.5 — each context keeps its SP in its own region, addressed through a runtime-private per-context register, so the shared word and its lock both disappear; `FORMAT_VERSION` 4→5). The state word stays per-context-swapped (only flipped, not accumulated, so it needs no lock). | §3, §12.7 | mitigated (single-worker STW); 4A.5 = lock-free SP |
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
- **Dispatch table** (`DomainTable`, `svm-interp:2002`): **nothing to capture in v1.**
  The table is an *identity* table built deterministically from the module
  (`slot i → (module 0, func i)`), so it is re-created bit-identically on thaw from the
  same instrumented module — like the JIT's `readonly` data segments. The only runtime
  mutation is `install` of guest-JIT native units, which are **non-durable** anyway
  (their `JitDomain`/`JitCode` handles make freeze refuse — §12.5). So a freezable
  domain's table is a pure function of the module; storing it would be redundant.

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
the region). **JIT re-establish parity** also landed:
`svm_jit::compile_and_run_capture_reserved_with_host_prots` takes a `WindowProt` map and applies
it to the freshly-seeded window (`protect_ro` / new `protect_none` via real
`mprotect`/`VirtualProtect`) before the run, so a thawed `Ro`/`Unmapped` page faults on the JIT
exactly as on the interpreter (`durable_prot_capture.rs` asserts both). Note module-defined
`readonly` segments already re-apply on every JIT instantiation; this adds the *runtime*-captured
map. **JIT-side capture** also landed: `Host::capture_window_prots(data, mapped, npages)`
reconstructs the window's protection map from the two host-side sources — the module's `readonly`
data segments (`Ro`) merged with the runtime page-state map (`cap_pages`, populated by Memory-cap
`map`/`unmap`/`protect`) — mirroring the interpreter's `snapshot_prots`. `durable_prot_capture.rs`
asserts interp and JIT capture the **same** map for a readonly-segment module, and that a runtime
`cap_pages` entry overrides the default. So page protections now round-trip on **both** backends,
both directions. The page-protection story is complete.

The §12.4 **fiber control state** now rides along too (Section 2 — the `FrozenFiber` residue,
slice 3.1.5): a freeze flattens each parked fiber's continuation into the window image and records
its residue (slot/funcref/sp/shadow-SP) in a TLV control section (tag 2, elided when there are no
fibers, so no-fiber artifacts stay byte-identical); `restore` re-seeds the `Host`. A single-fiber
domain now round-trips through the real artifact (`crates/svm-snapshot/tests/roundtrip.rs`,
including the §12.6 canonical re-serialize invariant). Remaining Phase-3 control-state work is
**multi-vCPU** (per-context state words) and the **dispatch table** (a module-derived no-op today).

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

### 12.8 Phase-3 design — STW drive + fiber continuations

Phase 1/2 froze a **single vCPU with no live fibers**. Phase 3 adds multi-vCPU and
fibers. Scoping (verified against the tree) found the work concentrates in two places;
the dispatch table is *not* one of them (see §12.4 — it's a module-derived identity
table, re-created on thaw, nothing to capture).

**(a) Stop-the-world is cooperative safepoints, never preemption.** The safepoint is a
**poll site** (the `if state == UNWINDING` after every may-suspend op the transform
already emits). Freeze, Go/JVM-shaped:

1. Host writes `UNWINDING` to the in-window state word(s).
2. Each running vCPU, at its next poll, unwinds one frame into its shadow stack and
   returns to its caller — the native stack peels off frame-by-frame into in-window,
   durable form.
3. Host waits for **quiescence** (every native stack empty), then snapshots.

Two signal carriers already exist: the **state word** is window memory shared by all
the domain's vCPUs (one write is domain-wide; the poll reads it), and the JIT's
**`interrupt`/epoch cell** (`os_thread_rt`, polled at back-edges + function entries,
re-checked every 20 ms by parked threads) is the *promptness* nudge that drives running
OS threads to a safepoint. The §5 residues drain by **running to the next safepoint**:
a vCPU inside a host `cap.call` finishes the call then polls; a fault-suspended fiber
gets its page, runs to its next poll. Both are semantically invisible (a poll site is a
point the program already passes through).

**(b) The crux — making a guest-suspended fiber's continuation durable.** A fiber the
guest `Suspend`ed sits in `RegFiber::Parked(Vec<Frame>)` (interp, `svm-interp:2966`) or
as a **native stack** (`FiberSlot.fiber`, JIT). Neither is durable, and — unlike the §5
residues — it **must not be run forward** (no `ContResume` is coming; advancing it would
execute work the guest never requested). So its parked continuation must become
shadow-stack form *without advancing the computation*. **[DECISION D-fiber-cont —
RESOLVED: option (A).]** Three options were:

- **(A) Fibers always keep their continuation in an in-window shadow stack.** Instrument
  `Suspend`/`ContResume`/`ContNew` so a suspend spills the continuation to the fiber's
  shadow stack; a parked fiber is then *already* durable. Clean, and the only option
  that gives **both backends** durable fibers (preserves the §7 cross-backend property).
  Cost: a real rework of the fiber engine + a per-suspend window-spill on the hot path.
- **(B) Freeze-time conversion by walking the parked frames.** On the interp,
  `RegFiber::Parked(Vec<Frame>)` is structured — the host can translate each `Frame`
  into a shadow frame directly (no guest execution). Cheap, but **interp-only**: the
  JIT's parked continuation is a raw native stack (no stackmaps — §8 rejected that), so
  this *breaks* cross-backend parity for fiber'd domains.
- **(C) Refuse to freeze with any live/parked fiber.** Too restrictive to be useful.

Recommendation: **(A)** — keep the cross-backend invariant the whole feature rests on,
and pay the engine cost. This decision gates the first implementation slice (the two
options are different implementations), so it should be settled before engine code.

**(c) Per-vCPU / per-fiber window layout.** Today the reserve has *one* state word +
*one* shadow stack (`STATE_OFF=0`, `SHADOW_SP_OFF=8`, `SHADOW_BASE=64` — the vCPU-0
layout). Multi-vCPU and ≥1 fiber need the reserve **partitioned per context** (state
word + shadow stack each), with a runtime-maintained "current context" so a poll
reaches the right shadow-SP. §12.7's per-fiber triple-stack already anticipates this.

**(d) Latency caveat (R6 / Phase 4).** Polls land only after may-suspend calls, not at
loop back-edges, so a vCPU in a poll-free compute loop never reaches a safepoint —
freeze hangs until it exits. Bounded-latency STW needs Phase-4 back-edge polls + a
`Blocking.work` cancellation story. A first cut accepts unbounded latency (cooperative
guest).

**Sub-phases.** 3.1 — freeze/thaw **one fiber, single vCPU, interpreter-only** (isolates
"continuation in a shadow stack" from thread coordination). *In progress:* the transform
recognizes `cont.new`/`cont.resume`/`suspend` as may-suspend points and a fiber'd module is
**NORMAL-inert** under instrumentation + verifies (`svm-durable/tests/fiber.rs`); the
**per-fiber shadow-stack layout + shadow-SP swap landed** (slice 3.1.1), the **resumer-side
`cont.resume` thaw arm** re-issues the resume on rewind (slice 3.1.2), and the **fiber-side
`suspend` re-park arm** flips to `NORMAL` and re-executes `suspend` on rewind (slice 3.1.3) —
so **both fiber thaw arms are now wired** (no fiber arm fails closed). The **freeze driver
flattens idle parked fibers** into their shadow regions (slice 3.1.4), and the **end-to-end
single-fiber round-trip works** (slice 3.1.5): freeze exports each flattened fiber's residue
(`svm_interp::FrozenFiber` via the `Host`), and a thaw re-seeds the registry and re-enters the
root under `REWINDING` — the resumer re-issues `cont.resume`, the fiber rewinds and re-parks,
then runs forward to the same result as the uninterrupted run (`svm-durable/tests/fiber.rs`).
The byte-level snapshot **Section-2 codec** lands too: `svm-snapshot` serializes the
`FrozenFiber` residue (slot/funcref/sp/shadow-SP) into a TLV control section (elided when there
are no fibers, so no-fiber artifacts stay byte-identical) and `restore` re-seeds the `Host`, so a
full `freeze → serialize → restore → thaw ≡ uninterrupted` runs through the **real artifact**
(`svm-snapshot/tests/roundtrip.rs`, incl. the §12.6 canonical re-serialize invariant). **3.1 is
complete on the interpreter, and now generatively fuzzed**: `durable_fuzz`'s
`fiber_freeze_thaw_equivalence_over_generated_modules` (+ the `durable_fiber` libFuzzer target)
drive a root+fiber generator — varying suspend counts, values live across each suspend, multi-point
resume/suspend — through the freeze→thaw round-trip (R11).

**3.3 — JIT parity (COMPLETE, single-vCPU).** The JIT already runs the transform's instrumented IR, so the
*thaw arms* (re-issue / re-park) work for free; parity is about porting the *runtime* pieces. Slice
**3.3.1 landed**: the JIT maintains the per-fiber **shadow-SP swap** in `fiber_resume` (which
brackets a fiber's residency — entry swaps in, exit swaps back, so `fiber_suspend` needs no change),
keyed off a `durable` flag + window base armed on the root `FiberRuntime` at entry and a per-`FiberSlot`
saved-SP. Gated by `compile_and_run_capture_reserved_with_host_durable`; tested by
`crates/svm/tests/durable_fibers_jit.rs` (each context routes to its own region, cross-checked
against `svm_interp`'s `SHADOW_*`). Slice **3.3.2 landed**: the JIT **freeze driver**
(`fiber_rt::freeze_drive`, hooked into `run_code_raw` after the root unwinds, gated on the
`UNWINDING` state word) flattens every still-`RUNNABLE` (parked) fiber into its shadow region by
resuming it under `UNWINDING` via the ordinary `fiber_resume` path — its post-suspend poll fires
before any guest code runs, so it unwinds with zero forward progress and its `Fiber` completes. It
runs host-side and unguarded — a flattening fiber touches only the committed reserve, so no guard
page can fault. Tested by a **cross-backend freeze comparison** (`jit_freeze_driver_flattens_a_fiber_matching_interp`):
interp and JIT freeze the same instrumented fiber module into a **byte-identical durable reserve**.
Slice **3.3.3 landed**, closing JIT parity: the freeze driver **exports** a `svm_jit::FrozenFiber`
residue per flattened fiber (entry funcref + data-SP retained in the `FiberSlot` at `cont.new`,
flattened shadow-SP read after), and a thaw **re-seeds** those fibers into the run-shared table
before re-entering under `REWINDING` (`fiber_rt::seed_frozen_fibers` builds each via the shared
`make_fiber`, so a thaw `cont.resume` re-enters its entry → rewinds → re-parks). The durable entry
(`compile_and_run_capture_reserved_with_host_durable`) takes a `seed` and returns the residue.
`durable_fibers_jit.rs` proves both cross-backend directions: interp and JIT freeze a fiber'd domain
to a **byte-identical §12 artifact** (window image + Section-2 residue), and an **interpreter-frozen
fiber artifact** restored through the codec **thaws on the JIT** to the uninterrupted result (107).
So a fiber'd durable domain now freezes, serializes, restores, and thaws **on either backend, in
either direction**.

The **active-resume-chain** gap is now **closed (slice 3.2, both backends)**: a fiber that's
*running* (mid-`cap.call` / mid-propagated-call / mid-nested-resume), not idle-parked, at the freeze
instant unwinds *in place* during the root's run — its base-frame return (interp) / `Complete`
(JIT) happens under `UNWINDING`. Previously such a fiber was marked `Done` and dropped from the
residue (so it couldn't thaw); now, when it actually unwound (its shadow region is non-empty —
`shadow_sp > region base`, which cleanly distinguishes a freeze-unwind from a *genuine* return of a
non-instrumented fiber), it is captured as residue (`Frozen` on the interp) and re-seeded on thaw,
where it rewinds at its in-flight (leaf/propagated/resume) point and runs **forward** — the active
analogue of an idle fiber's re-park. Tested both backends incl. **reload-not-reissue** of the
in-flight `cap.call` (`svm-durable/tests/fiber.rs::active_resume_chain_fiber_freezes_and_thaws`,
`durable_fibers_jit.rs::interp_frozen_active_chain_fiber_thaws_on_the_jit`).

**[~] Slice 3.2.1 — first multi-vCPU freeze/thaw (interp; no live fibers).** A domain whose root
has spawned a `thread.spawn` child freezes mid-run and thaws to equal the uninterrupted run
(`svm-snapshot/tests/roundtrip.rs::multivcpu_freeze_serialize_restore_thaw_through_the_codec`,
`svm-durable/tests/multivcpu.rs`). The choreography is **transform-free** — `svm-durable` only gains
*typing* for `thread.spawn`/`thread.join` (they aren't checkpoints; they're copied verbatim and their
results spill/reload like any scalar). Mechanism:

- **Single-worker freeze/thaw.** The shared control words are touched only under `UNWINDING`/
  `REWINDING`, so the runtime forces `workers = 1` exactly when the window's initial state ≠ `NORMAL`
  (ordinary `NORMAL` runs — incl. `NORMAL` durable — keep full parallelism). This serializes the
  control-word access (closing R10 for this slice without a lock) and lets the runtime swap the
  *current context* per dispatch.
- **Per-vCPU control words.** Each vCPU is context = its **task id** (root = 0), owning shadow region
  `shadow_region_base(task)`. On each dispatch the runtime swaps *both* the state word and the active
  shadow-SP into the shared window for the running vCPU, saving them back on park — settling §12.8(c):
  the state word must be per-context too, because a rewinding vCPU flips it to `NORMAL` after reloading
  and must not disturb a sibling still rewinding.
- **Residue.** Each unwinding child records a `FrozenVCpu` (`task, func, args, shadow_sp`); the root's
  extent is its implicit residue (`Host::frozen_root_sp`) since the shared active-SP word ends up
  holding the *last* child's extent at freeze end. The snapshot control section (tag 2) carries both,
  appended after the fiber residue and only when spawned vCPUs exist — so fiber-only/single-vCPU
  artifacts stay byte-identical (the §12.6 re-freeze invariant still holds).
- **Thaw.** `restore` re-seeds the residue; `drive` re-spawns the frozen children under `REWINDING`
  with their extents restored (they rewind from their frozen point, then run forward) and rebuilds the
  root's join table — because the root's rewind *skips* its prologue `thread.spawn` (the `REWINDING`
  prologue jumps straight to the resume ARM).

Scope/limits (follow-ups): no live fibers (the multi-vCPU + fibers context unification — vCPU contexts
use task ids, fibers use slot+1, which would collide); flat (root-spawned) children only (nested spawns
need per-parent join-table residue); shallow stacks (the unwind guard is reserve-relative, so per-region
capping is unenforced); and `context = task id` bounds a durable domain to ~15 *lifetime* spawns (a dense
live-context allocator is the fix). JIT parity is 3.3's multi-vCPU follow-up.

**[~] Slice 3.2.2 — vCPU + fiber context unification (collision fix; interp).** A durable domain may
now have spawned vCPUs **and** live fibers at once. Fibers keep growing **up** from context 1
(`slot+1`, untouched — preserves cross-backend artifact parity, since the JIT mirrors that formula),
and spawned vCPUs grow **down** from `MAX_SHADOW_CTX`; the two pools share the reserve and a capacity
guard (`fibers + vcpus <= MAX_SHADOW_CTX`) refuses cleanly (`FiberFault`/`ThreadFault`) before they
meet — never an overlap. Contexts stay *derived* from slot/task (reproduced deterministically on thaw),
so no snapshot-format change and no new residue fields. The `FiberRegistry` gained a vCPU-context
counter (`reserve_vcpu_context` hands out the next top-down region under the lock; `cont.new`
cross-checks the combined bound; thaw re-seeds the count). Tested: a root that owns a fiber **and**
spawns a vCPU freezes/thaws correctly, in-memory (`svm-durable/tests/multivcpu.rs`) and through the
codec (the control section carries both fiber and vCPU residue; canonical re-freeze stays
byte-identical — `svm-snapshot/tests/roundtrip.rs`). Cross-backend parity unaffected (fibers unchanged;
the JIT has no multi-vCPU durable path yet).

**[~] vCPU-context recycling (interp) — done.** A spawned vCPU's shadow context is now **freed when
the child genuinely finishes** (not a freeze-unwind, which keeps the region for thaw), so a durable
domain is bounded by *peak concurrent* vCPUs rather than ~15 *lifetime* spawns. The registry tracks
occupancy as a `u16` mask (contexts `1..=MAX_SHADOW_CTX`); `reserve` hands out the highest free context
above the fiber pool, the fiber/vCPU collision guard checks the lowest occupied vCPU context, and a
child's `VCpu.vcpu_ctx` is freed in the scheduler's `Done` path. The thaw is made **gap-tolerant**
(derives each re-spawned child's context from its restored shadow-SP to seed the mask; preserves task
ids for the §12.6 canonical re-freeze; rebuilds `threads[]` sparsely) — but note a *recycled-context
child at freeze* is **not reachable yet**: a freeze-from-start drives every vCPU to `UNWINDING` at t=0
(residue stays dense), and a *mid-run* multi-vCPU freeze needs a true stop-the-world (the
`arm_freeze_after` trigger flips only the running vCPU's per-context state word), which is Phase 4. So
the gap-tolerance is exercised only on the dense path today; the cap-lifting is pinned by
`svm-durable/tests/vcpu_recycle.rs` (20 sequential and 8×2 concurrent spawn/join cycles, both of which
would `ThreadFault` at the 16th lifetime spawn without recycling).

Remaining for 3.2: **nested spawns** + a *spawned* child owning fibers (per-child `freeze_drive`), and
**JIT multi-vCPU parity** (3.3). Then **Phase 4** back-edge polls for bounded-latency (async) freeze —
which also unlocks the mid-run multi-vCPU STW that would make recycled-context freeze/thaw reachable.

##### Slice 3.3 — JIT multi-vCPU durable parity (design)

The interpreter freezes/thaws a domain whose root has `thread.spawn`-ed children; the JIT freezes/thaws
only single-vCPU (fiber) domains today. The gap is **not** a concurrency-model barrier: the interp's
multi-vCPU durable run is itself **single-worker** (it forces `workers = 1` whenever the window state ≠
`NORMAL`), so "multi-vCPU durable" means *a domain with several vCPUs*, frozen/thawed serially — not
vCPUs running concurrently mid-freeze. The JIT just needs the **same serialization**, which it lacks
because it runs `thread.spawn` children as concurrent 1:1 OS threads (`os_thread_rt::run_child`) with no
cooperative dispatch boundary.

**Mechanism (mirror the interp, deferred single-worker):** when the window state ≠ `NORMAL`, the JIT's
`thread_spawn` thunk does **not** start an OS thread. It reserves the child a top-down shadow context (a
`vcpu` occupancy allocator on `SharedFiberTable`, mirroring the interp's `vcpu_mask`) and a completion
cell, *records* the spawn request, and returns the handle — then the child runs **inline after the
spawning vCPU yields** (`Domain::drive_frozen_spawns`, called from `run_inner` once the root has unwound
and its fibers are flattened). This **deferral is load-bearing for byte-identity**: both backends run the
same instrumented IR, so the root unwinds at its first checkpoint *before* it reaches `thread.join`;
running each child only after the root yields reproduces the interp's exact dispatch order (root → root's
fibers → children, in spawn order) and therefore the same side-effect interleaving (e.g. which vCPU reads
the clock first). Running the child *immediately* at the spawn point instead reverses that interleaving
and diverges the frozen window. Each deferred child runs in its own context (point `SHADOW_SP_OFF` at its
region, run the child entry via the existing guarded-range path), captures its flattened extent as a
**`FrozenVCpu`** residue (a JIT mirror of `svm_interp::FrozenVCpu`) when it unwinds under `UNWINDING`, and
publishes its result to its `Done` cell; the last child leaves the active shadow-SP at its own extent,
matching the interp's dispatch-last convention. `NORMAL` durable runs keep concurrent OS threads (matching
the interp's multi-worker `NORMAL`).

**Decomposition:**
- **PR-1 (freeze side) — DONE:** the deferred single-worker path (`defer_spawn` /
  `Domain::drive_frozen_spawns`) + `FrozenVCpu` residue + vCPU-context allocator, exported through
  `compile_and_run_capture_reserved_with_host_durable_mv`. Pinned by `durable_multivcpu_jit`'s
  `jit_freezes_a_spawned_vcpu_matching_interp`: a root+child domain freezes to a **byte-identical durable
  reserve** and a **field-identical `FrozenVCpu` residue** vs the interpreter (the multi-vCPU analog of
  `jit_freeze_driver_flattens_a_fiber_matching_interp`).
- **PR-2 (thaw side) — DONE:** `Domain::thaw_reattach_and_run` re-attaches the frozen children **before**
  the root re-enters under `REWINDING` (the root's rewind skips its prologue `thread.spawn`, reloading
  the recorded handle), rebuilds the join table at each child's handle slot (`task − 1`, padding
  finished/joined gaps), and runs each child inline from its restored extent (rewind → `NORMAL` → run
  forward → publish its result so the root's re-executed `thread.join` resolves); the root's extent +
  `REWINDING` are then restored for its re-entry. The freeze side exports the **root's extent** (`root_sp`,
  separate because the shared active-SP word ends at the last child's). The children run *before* the root
  (rather than the interp's root-parks-on-join dispatch); this is sound because a `REWINDING` vCPU
  **reloads** its recorded side effects, so the serialization order can't change the result (§12.6).
  Pinned by `durable_multivcpu_jit`'s `jit_thaws_its_own_multivcpu_freeze` (JIT freeze → JIT thaw on an
  advanced clock reproduces the uninterrupted result — reloads, not re-issues) and
  `interp_frozen_multivcpu_thaws_on_the_jit` (cross-backend: an interp-frozen domain thaws on the JIT).

Scope mirrors the interp's: flat (root-spawned) children, no nested spawns, no child-owned fibers.

##### Slice 3.4 — finish the multi-vCPU scope (nested spawns + child-owned fibers) (design)

Slices 3.2/3.3 left two asterisks on multi-vCPU durable: **(1) child-owned fibers** (a spawned child
creates/owns `cont.*` fibers) and **(2) nested spawns** (a child itself calls `thread.spawn`). This
slice lifts both, on both backends, keeping the byte-identical cross-backend artifact invariant.

**The structural fact that scopes the work.** The fiber registry / `SharedFiberTable`, the vCPU-context
allocator, and the fiber-slot allocator are all **domain-shared**. A child's `cont.new` fibers land in
the *same* parked set the root's `freeze_drive` walks (`take_parked_for_freeze` is owner-agnostic, keyed
by slot), and a grandchild's context/task-id come from the same domain-global allocators. So a large part
of "child-owned fibers" is *already handled*, and grandchild context/task assignment composes for any
depth. The genuine gaps are narrow:

- **Child-owned fibers — missing:** (interp) a child parked *mid-`cont.resume`* at the freeze instant
  isn't driven into shadow form — the per-vCPU control-word swap in `dispatch` is gated on
  `cur == ROOT_FIBER`, and a spawned vCPU never runs `freeze_drive`; and the freeze block **assigns**
  `host.frozen_fibers = …` (clobbering) where it must **extend**. (JIT) `run_child_inline` runs no
  per-child `freeze_drive`, and the root's drive already ran (before the deferred children), so a fiber a
  child parks is left un-flattened and dropped from the residue. *No snapshot format change* — the fiber
  residue is owner-agnostic; only *who produces* it changes. Thaw is free once freeze emits the residue
  (fibers re-seed densely by slot before any vCPU re-enters).
- **Nested spawns — missing:** `FrozenVCpu` carries no owner, so thaw can't rebuild the join-table
  topology → add a **`parent_task`** field (root's direct children = 0) and bump snapshot
  `FORMAT_VERSION` 3→4 (the one format change in this slice). (JIT) `drive_frozen_spawns` one-shot-drains
  `pending_spawns`, missing a grandchild spawned during a child's inline run → **loop-drain until
  empty**, stamping each grandchild's `parent_task`. Reconcile the join-table model: the interp uses
  **per-vCPU** `threads` (handle = index in the spawning vCPU's table) while the JIT uses one **global**
  `cells` — identical for flat spawns, divergent once nested, so the JIT gains **per-vCPU handle
  namespaces on the durable single-worker path** to keep handle values byte-identical. Thaw groups the
  seed by `parent_task`, processes parents-before-children, and rebuilds each parent's table.

**Staging (interp-oracle-first per stage, cross-backend byte-identity test at each) — ALL DONE:**
- **A — child-owned fibers, interp (DONE):** every vCPU runs its **own** `freeze_drive` (the root's runs
  before the children exist, so it can't flatten a child's fiber); the per-vCPU residue now **extends**
  the shared host list instead of clobbering it. No format bump. (`multivcpu::child_owns_fiber_…`.)
- **B — child-owned fibers, JIT + cross-backend (DONE):** per-child `freeze_drive` in `run_child_inline`
  (the child's runtime is `set_durable_env`-armed so the driver's `Complete` arm records residue),
  drained into the run residue. (`durable_multivcpu_jit::jit_freezes_and_thaws_a_child_owned_fiber_…`.)
- **C — nested spawns, format v4 + interp (DONE):** `FrozenVCpu.parent_task`; `FORMAT_VERSION` 3→4;
  stamp `parent_task` at `thread.spawn`; thaw re-attach rebuilt **parent-first** with per-parent join
  tables (a `BTreeMap<task, VCpu>`, appending each child's handle into its parent's table in spawn
  order). (`multivcpu::nested_spawn_tree_…` + `roundtrip::nested_spawn_tree_…through_the_codec`.)
- **D — nested spawns, JIT + cross-backend (DONE):** the JIT durable path gained **per-vCPU join
  tables** (`Domain::dchildren`, keyed by spawning task, routed by `cur_task`) so a grandchild's guest
  handle is its index in its *parent's* table — byte-identical to the interp. `drive_frozen_spawns`
  **loop-drains** `pending_spawns` (a grandchild deferred during a child's inline run is caught on the
  next BFS batch — matching the interp's runnable order), and a global monotonic `next_task` matches the
  interp's task ids. **Thaw runs children in *descending* task order (children before parents)** so a
  parent's re-executed `thread.join` finds an already-completed child — the JIT can't park-and-resume a
  parent on the single worker (the interp parks it); a `REWINDING` vCPU reloads its effects, so the
  order can't change the result. (`durable_multivcpu_jit::jit_freezes_and_thaws_a_nested_tree_…`.)

The durable path stays strictly single-worker (writers run inline on one OS thread), so no new loom
seam (re-verified). **Out of scope (needs Phase-4 STW):** a *concurrent* mid-run freeze where a grandchild is
mid-compute on its own OS thread (the trigger flips only the running vCPU's word); recycled-context
nested freeze; and a child blocked in a host `Blocking.work` at the freeze instant. This slice targets
the deterministic single-worker paths (freeze-from-start and `arm_freeze_after` at a fiber safepoint).

##### Phase 4 Slice A — back-edge polls + bounded-latency async STW freeze (design)

Closes the R6 latency caveat and the R10 control-word concurrency seam: today a freeze only lands at
*may-suspend* safepoints (`cap.call`/`cont.resume`/`suspend`), so a vCPU in a **poll-free compute loop**
never reaches a safepoint and the freeze hangs; and the freeze run is only safe because it is forced
**single-worker**. This slice adds polls a compute loop can't skip, plus the multi-worker→single-worker
quiesce handshake for a true async stop-the-world.

**Poll mechanism (in the IR transform, not codegen).** `svm-durable`'s `transform_func` emits a
state-word check at every **loop back-edge** (target block id ≤ source — the reducible back-edge
heuristic; irreducible CFGs fall back to all branch terminators) + extends the function-entry prologue
to unwind on `UNWINDING` (not just `REWINDING`). This is the same observe-and-unwind it already emits
after a may-suspend op, at a new site. **IR-level, not per-backend codegen** — so both backends compile
the *same* poll and the byte-identical-artifact invariant (R11) holds automatically, and it stays +0
TCB (vs. putting freeze-relevant control flow into Cranelift lowering). It reads the **per-context state
word** at `STATE_OFF` (already in-window, per-context, snapshot-captured) — *not* the §5 epoch cell
(that traps rather than unwinds and isn't per-context); the epoch cell stays the *promptness* nudge. A
back-edge poll has no in-flight call to reload, so each instrumented back-edge gets a **new resume id**
whose live set is the target block's edge-args (spilled on unwind, reloaded on rewind → jump to the
loop header). Frame format unchanged ⇒ **no `FORMAT_VERSION` bump**. NORMAL-inert: a not-taken,
perfectly-predicted branch whose only cost is the state-word load.

**STW protocol (the R10 loom seam).** A new `request_freeze` controller writes `UNWINDING` to **every**
live per-context state word (vs. `arm_freeze_after`, which flips only the running vCPU's) + sets the
epoch cell so busy JIT OS threads reach their next poll and parked vCPUs re-check. Each worker observes
it at a back-edge, unwinds its native stack into its own in-window shadow region, and parks at base.
The only truly-shared control word is the active shadow-SP (`SHADOW_SP_OFF`); during a multi-worker
quiesce, concurrent workers swapping their context's SP in/out of it need a **lock/atomic** (the new
loom obligation) — strictly **gated to `workers > 1`**, a no-op fast path at `workers == 1` so every
existing deterministic path stays byte-identical. *(4A.5 retires this shared word — and its lock —
by giving each context its **own** shadow-SP word in its own region; see the per-context shadow-SP
design below.)* After all vCPUs quiesce (join barrier), a single
coordinator runs the **existing** freeze-drive/residue/flatten machinery (untouched — it already
assumes single-worker). Residue is **canonically sorted** (ascending context/task) before serialize, so
the quiesce *order* (which races) can't change the artifact (§12.6 canonical invariant preserved).

**Payoff — activates already-shipped future-proofing:** a mid-run async STW after some children have
finished + recycled their contexts produces the first **sparse** residue (exercising the gap-tolerant
thaw / `seed_vcpu_mask`), and freezing a root whose children are **genuinely concurrent OS threads**
mid-compute is the first real concurrent multi-vCPU freeze (vs. the deferred single-worker emulation).

**Staging (interp-oracle-first):** 4A.1 single-vCPU compute-loop freeze via a back-edge poll (interp,
ticked deterministically like `arm_freeze_after`) — *proves the core*; 4A.2 JIT parity (the IR poll
compiles for free → byte-identical artifact); 4A.3 async `request_freeze` (single-vCPU); **4A.4
multi-worker quiesce + active-SP swap sync (LOOM)**; 4A.5 concurrent multi-vCPU STW freeze, JIT — via
**per-context shadow-SP** (`FORMAT_VERSION` 4→5; retires the 4A.4 shared-SP lock; design subsection
below) (LOOM); 4A.6 recycled-context async freeze (sparse-residue payoff); 4A.7 parked-vCPU /
`Blocking.work` latency
(narrows R6/R2 — freeze refuses on an in-flight `Blocking` call; full offload-cancellation deferred).

**Status:** 4A.1–4A.5 + follow-ups **A** and **B.1** + the **blocked-in-`thread.join` freeze** + **B.2**
(full nested concurrent spawns) + the **blocked-in-`thread.wait` freeze** (bounded + fail-closed) +
**B.1′** (concurrent child-fiber *mid-resume-chain*, verified) are **landed** (the first three merged,
all-platform CI green; the rest on `claude/durable-next-slices-tracker`). The remaining queue — **lift
the `atomic.wait` thaw fail-closed** (concurrent-thaw rework: **design + 3-stage plan now written** under
*"Concurrent thaw"* below; stage 1 = per-context thaw-state relocation, `FORMAT_VERSION` v6→v7), then
**4A.6 / 4A.7** — is detailed in the *"Phase 4 Slice A.5 — per-context shadow-SP"* follow-up notes below.

**Out of scope (separate Phase-4 items):** handle hardening (drainable non-durable bindings), CoW clone,
full `Blocking.work` offload cancellation (R2), `SharedRegion` consistent-cut (R4).

##### Phase 4 Slice A.5 design — per-context shadow-SP (retires the shared-SP lock)

4A.4 shipped the **shared** active shadow-SP (`SHADOW_SP_OFF`, window offset 8) guarded by a
`workers > 1` lock during the quiesce swap: it keeps one shared word correct under concurrency by
*serializing* access to it. 4A.5 takes the other branch — it **removes the shared word** instead of
serializing it. Each context gets its **own** shadow-SP, so concurrent children never touch a common
location, the hot-path lock disappears, and the race is dissolved by construction rather than guarded.

*Why a shared word existed at all.* The durable transform emits every spill/reload against the fixed
offset `SHADOW_SP_OFF`, and the runtime makes that one word mean "the running context's SP" by
**swapping** it in/out of the window per dispatch (the R10 single-worker mitigation). That swap is
correct only single-threaded — one context dispatched at a time; concurrent OS-thread children would
each want offset 8 to be *theirs* at once. The per-child shadow *regions* already exist
(`shadow_region_base(ctx)` gives each context a disjoint slice of the reserve); **only the SP *word* is
shared.** The fix relocates each context's SP word into its own region so the transform addresses *its*
SP, never a global one.

- **Where the SP lives (the format change).** Move the SP word from the global `SHADOW_SP_OFF` = 8 to a
  **per-context slot** — e.g. the first 8 bytes of each context's region — with that region's frames
  starting after it. The reserve's global `[0, 64)` header keeps `STATE_OFF`/`ARM_*`; offset 8 is
  retired as the active-SP word. This changes the artifact's window image ⇒ **`FORMAT_VERSION` 4 → 5**,
  applied **uniformly** (single-worker, concurrent, both backends) so the new format stays byte-identical
  cross-backend (R11) and concurrent-vs-single-worker — just not to v4 (a one-time bump).

- **How the transform finds it (the TLS question).** The spill/reload must compute "my context's region
  base" from compiled code. That needs a *runtime-private per-context identity*. We **do** have a
  per-vCPU TLS register (`vcpu.tls.get/set`, §12), implemented consistently on both backends — but it is
  **guest-overwritable** (`vcpu.tls.set`), so it cannot back the shadow-SP: a guest could clobber it and
  corrupt/escape its own shadow stack (a TCB regression). So 4A.5 adds a **sibling runtime-private
  register** — the same per-OS-thread thread-local mechanism as `vcpu_tls`, **seeded by the runtime** per
  dispatch / per child, with **no guest write op** — holding the active context's region base (or its
  dense id, from which the base is derived). The transform lowers the SP access as `[ctx_base + SP_SLOT]`
  where `ctx_base` is read from this register. Both backends resolve it identically: JIT via the
  thread-local (a baked thunk, paid only on the *cold* unwind/rewind path); interp via its dispatch
  state. It stays **IR-level, not per-backend codegen**, so the byte-identical-artifact invariant (R11)
  holds automatically and it remains +0 TCB.

- **What it retires.** With per-context SP there is no shared active-SP word, so the 4A.4 `workers > 1`
  swap-lock is **unnecessary on the unwind path** — concurrent children spill into disjoint SPs with zero
  coordination. The quiesce primitive collapses to its **join** role: `request_freeze` flips every
  context's state word; each worker self-unwinds into its own region+SP at its next back-edge poll and
  parks at base; the coordinator waits on the existing **join barrier** (the loom-verified 4A.4 handshake)
  before running the **untouched** single-worker freeze-drive/residue/flatten machinery. The loom
  obligation narrows from "swap-exclusion + join" to just the join handshake.

- **Staging within 4A.5.** *(i)* introduce the runtime-private per-context register + migrate the durable
  SP to per-context storage on **both** backends; bump `FORMAT_VERSION` 4 → 5; keep single-worker
  freeze/thaw + the `durable_jit` cross-backend fuzz green (**pure refactor + format change, no
  concurrency yet** — fully testable single-worker). *(ii)* spawn real concurrent children on the
  async-STW entry, each self-unwinding into its own SP; coordinator joins via the barrier → existing
  freeze-drive. *(iii)* loom the join; two-concurrent-children byte-identical-to-single-worker test under
  a deterministic trigger; `request_freeze` round-trip; cross-backend.

**Progress (stage i).** *Landed:* (a) the `durable.shadow_base` IR op + a runtime-private per-OS-thread
register (`svm-jit`'s `durable_shadow`, the interp's `run_inner`), mirroring `vcpu.tls.get` but with no
guest write op; (b) a **byte-identical bridge** — the durable transform reads the active context's
shadow-SP **word address** from that register at all four SP sites (dispatch / unwind check / unwind
spill / arm) instead of `ConstI64(SHADOW_SP_OFF)`, with the register still resolving to the shared
`SHADOW_SP_OFF` (= 8), so artifacts are unchanged. This proves the transform → register → both-backends
path end-to-end. (c) **Relocation + format bump landed** — each context's shadow-SP word now lives at
`shadow_region_base(ctx) + 0` (frames at `+8`), `durable.shadow_base` returns that region base on all
three engines (tree-walker, bytecode, Cranelift JIT), the legacy global `SHADOW_SP_OFF` is retired, and
`FORMAT_VERSION` is bumped 4 → 5. Cross-backend byte-identity (`durable_jit`) + every durable suite
green. Stage (i) is **done**; stage (ii) (real concurrent children + the join barrier) is next. The
original site map is retained below for reference:

- **Layout.** Put each context's SP word at **`shadow_region_base(ctx) + 0`** (the region's first 8
  bytes); frames follow at `+ 8`, so `SHADOW_SP_OFF` (global offset 8) is retired. The transform is
  layout-agnostic: `durable.shadow_base` returns `shadow_region_base(active ctx)` and the SP word is at
  `+0`, so no within-region stride constant leaks into `svm-durable`. Empty extent = `region_base + 8`.
- **Register value.** Flip `durable.shadow_base` from `SHADOW_SP_OFF` to `shadow_region_base(active
  ctx)`. *Interp:* resolve in `run_inner` from `(cur == ROOT_FIBER ? vcpu_ctx : cur + 1)` (both in
  scope) — no seed needed. *JIT:* `durable_shadow::seed(region_base)` at each point the active context
  changes — vCPU/child entry (`os_thread_rt`), the dispatch boundary, and both edges of the `fiber_rt`
  resume swap.
- **SP word storage (retarget the existing helpers, +8 init).** Give `durable_get_sp`/`durable_set_sp`
  (interp `Mem`) and `read_shadow_sp`/`write_shadow_sp` (`fiber_rt`) a `region_base` parameter →
  `window[region_base + 0]` (was the fixed offset 8); each call site already knows its context's region
  (`shadow_switch` out/in ctx, the `fiber_rt` resumer/slot, `os_thread_rt` `p.ctx`, the root). Shift
  every SP **init** from `shadow_region_base(ctx)` to `+ 8` (interp `root_shadow_sp` 4644/4707, registry
  `shadow`/`saved_sp` seeds, child `root_shadow_sp` 6004, the `4839` reset; JIT `AtomicU64` 371,
  `root_shadow_sp` 538, `thaw_root_sp` lib.rs 2395, `os_thread_rt` 825), and the **"spilled?" extent
  checks** (e.g. `fiber_rt` `flat_sp > fiber_region_base(slot)` → `+ 8`).
- **Helpers / format.** `init_durable_window` writes the root SP word at `window[SHADOW_BASE] =
  SHADOW_BASE + 8` (offset 8 unused). `svm-snapshot`'s `SHADOW_BASE` residue defaults (root_sp, the
  empty-section path) → `SHADOW_BASE + 8`. Bump `FORMAT_VERSION` 4 → 5. Guard with the `durable_jit`
  cross-backend fuzz (byte-identical interp↔JIT) — the all-or-nothing oracle for this step.

**Stage (ii) — concurrent multi-vCPU STW freeze (JIT). LANDED.** With per-context SP landed, children
unwind *concurrently* into disjoint region words with no shared scratch and no lock. A new entry
(`..._durable_mv_interruptible`) engages the concurrent path; `thread_spawn` reserves a per-context
shadow context for a durable child spawned during NORMAL; `run_child` seeds the durable shadow-base
register and, on a freeze-unwind (UNWINDING + spilled past its frame base), records the child's
`FrozenVCpu` residue. **`join_all` is the coordinator-wait** — the per-context relocation made the 4A.4
barrier's serialization unnecessary, so each child's freeze-unwind simply completes its OS thread and
`join_all` blocks until all have; the concurrent residue is then drained (after the join) so the
snapshot sees a fully-quiesced window. The quiesce barrier is retained as a loom-verified primitive.
Pinned by `crates/svm/tests/durable_concurrent_jit.rs`: a root + two children all freeze mid-loop under
an async `request_freeze` and the thaw reproduces the result (each total in its own guest-memory slot, so
the round-trip is robust to which context froze when). The tests use a **spawn-before-freeze handshake**
(the root signals via a host fn once children are spawned; the controller requests the freeze only then)
— otherwise the async freeze fires before the root's `thread.spawn`s and the children are *deferred* to
the single-worker path. (That handshake also surfaced a real bug, since fixed: a concurrent child never
initialised its region's shadow-SP word, so on a freeze it spilled over the reserve header — `run_child`
now seeds it to the frame base.)

*Follow-up A — `thread.join` result across a concurrent freeze (LANDED).* A concurrent child that
finishes *before* the freeze point delivers its result to the host-side Done cell, which the snapshot
doesn't capture, so the root's later (post-freeze) `thread.join` couldn't resolve on thaw. Now
`run_child` records every completed concurrent child; on a freeze the coordinator turns them into
`completed_result` `FrozenVCpu` residue (`FORMAT_VERSION` 5→6), and the thaw delivers each result into
the spawner's join table **without re-running** the child (its effects are already in the snapshot).
Emitting *all* completed children keeps the per-parent table dense so every handle still resolves.
Pinned by `concurrent_join_result_survives_a_freeze_before_the_join`.

*Follow-up B.1 — concurrent child owns fibers (LANDED).* `run_child` now arms the child's fiber runtime
durable (`set_durable_env`) and, on a freeze-unwind, runs its own `freeze_drive` over its parked fibers
(the concurrent mirror of `run_child_inline`'s), draining the residue into the domain accumulator
(collected after `join_all`). Pinned by `concurrent_child_owns_fiber_through_freeze_thaw`. **Scope:** this
covers a child whose fiber is **cleanly parked** at the freeze point (the test's signal-after-park
handshake).

*Follow-up B.1′ — concurrent child-fiber caught mid-resume-chain (LANDED).* The harder interleaving — a
freeze that lands while the child's fiber is still **active on the resume chain** (not yet suspended) —
is now verified on the concurrent path. The existing machinery already covered it: when the resumed
fiber unwinds under `UNWINDING`, the fiber runtime's `cont.resume` return path records it as active-chain
residue (the same `rt.frozen` path slice 3.2 uses for the root), the child unwinds at its `cont.resume`
re-issue safepoint, and the thaw re-issues that resume to re-enter the fiber, which rewinds to its
in-flight point and runs forward. No code change was needed — it was a verification gap. Pinned
deterministically by `concurrent_child_owns_active_chain_fiber_through_freeze_thaw`: the **fiber itself**
drives the spawn-before-freeze handshake (it signals from inside the chain, then loops `K`), so the async
freeze reliably lands mid-fiber with the child blocked in `cont.resume`; the thaw reproduces the
uninterrupted `K` (root) and `2K` (child's loop + the fiber's own `K`-loop total it suspends). B.1 now
asserts only its *parked* shape and defers the active shape to this test.

*Follow-up B.2 — nested concurrent spawns (done).* A *concurrent* child that itself `thread.spawn`s a
grandchild attributes the grandchild's `parent_task` via a **per-OS-thread spawning-task source**
(`CONCURRENT_SPAWN_TASK`, seeded to the child's task in `run_child`, read in `thread_spawn`) — *not* the
shared `Domain::cur_task`, which only the single-worker inline/thaw paths maintain and which would race
across concurrent spawners. The earlier `IN_CONCURRENT_CHILD` fail-closed guard is retired. During NORMAL
the nested spawn/join resolves through the flat global thread table (dense global handles); on a freeze
each level self-unwinds into its own per-context region and records a `FrozenVCpu` with the correct
`parent_task`, and the thaw's per-parent rebuild (slice 3.4) reconstructs the topology and runs the tree
in descending-task order so each `thread.join` resolves. Pinned by
`nested_concurrent_spawn_returns_grandchild_value` (NORMAL, returns the grandchild's 42 through both
joins) and `nested_concurrent_tree_freezes_and_thaws` (root → child → grandchild, all real OS threads,
caught mid-flight; the grandchild drives the spawn-before-freeze handshake; the thaw reproduces
`K`/`2K`/`3K`). Deferred nested spawns (slice 3.4) and concurrent *flat* spawns already worked.

**Freezing a vCPU blocked in `thread.join` — done.** `thread.join` is now a may-suspend re-issue
safepoint: `compute_may_suspend` counts it (so a "spawn then join" root is instrumented), the transform
classifies it as `SuspendKind::ThreadJoin` (its result is *re-issued* on thaw like `cont.resume`, since
the joined child replays its own side effects on its rewind — §12.6), and the `thread_join` runtime thunk
now returns on observing `UNWINDING` so a vCPU **parked in the join** unwinds at the trailing safepoint
rather than blocking the stop-the-world freeze. On thaw the join is re-issued; because the join has no
in-thread callee to flip the state word (the child rewinds as a *separate* vCPU and the thaw driver
resets the word to `REWINDING` afterward), the join is the globally-deepest frozen frame on its own
thread, so — like a leaf — it flips the state to `NORMAL` itself before re-issuing. Pinned by
`concurrent_freeze_while_root_blocked_in_join` (root parks in the join; the child drives the
spawn-before-freeze handshake so the freeze lands while the root is blocked). Both the running-root path
(follow-up A) and the blocked-root path are covered.

**Freezing a vCPU blocked in `thread.wait` — done (bounded + fail-closed).** The futex `atomic.wait` is
now a may-suspend re-issue safepoint, mirroring `thread.join`: `compute_may_suspend` counts `MemoryWait`,
the transform classifies it as `SuspendKind::MemoryWait` (re-issued on thaw — reload `addr`/`expected`/
`timeout`, flip the state word to `NORMAL`, re-execute), and the `thread_wait` thunk returns on observing
`UNWINDING` (the same `window_is_unwinding` trick the join thunk uses) so a vCPU parked in a futex wait
unwinds at the trailing safepoint instead of hanging the stop-the-world freeze. **Before this, a freeze
requested while any vCPU was parked in a wait would deadlock `join_all`** — the parked thread only woke on
notify / timeout / kill. *Thaw* re-checks the value: a wake that landed as a value change (in the
snapshot, or replayed by another re-run vCPU) resolves the re-issued wait immediately with
`WAIT_NOT_EQUAL` — no re-park, no notifier. A re-issue that would still *park* can't be satisfied on the
single-worker thaw (no concurrent notifier), so it **fails closed** with `ThreadFault` (matching the
interp, which surfaces a guest wait/join-deadlock as `Trap::ThreadFault`) via a `Domain.thawing` flag,
rather than deadlocking. Pinned by `concurrent_freeze_while_root_blocked_in_wait_thaws_when_value_changed`
(child changes the value without notifying → thaw resolves `NOT_EQUAL`) and
`…_fails_closed_on_thaw` (value unchanged → thaw traps `ThreadFault`). **Lifting the fail-closed** — a
re-park resolvable only by reordering — needs the concurrent-thaw rework below.

#### Concurrent thaw — design + staging (lifts the `atomic.wait` fail-closed)

*Why the current thaw can't satisfy a re-parking wait.* `thaw_reattach_and_run` runs the frozen vCPUs
**inline on one worker**, each rewound to completion before the next, in descending-task order. That is
correct for everything *except* a wait that must re-park: an inline waiter has no concurrent vCPU to
`notify` it, so it would deadlock — hence the fail-closed. Resolving it needs the waiter and its notifier
to run **concurrently**, re-synchronizing through the real domain futex exactly as on a fresh run.

*The blocker: the shared global state word.* The durable state word lives at a **single global window
offset** (`STATE_OFF = 0`) — the prologue reads it (`REWINDING` ⇒ rewind-dispatch), every poll reads it
(`UNWINDING` ⇒ unwind), and a re-issue writes it (`NORMAL`). The interp runs single-worker and
*multiplexes* each vCPU's `dstate` through that one word (swapped in when the vCPU runs, saved back when
it yields). The JIT could spawn the frozen vCPUs as real OS threads, but they would **race on the one
word**: vCPU A finishing its rewind flips the global word to `NORMAL` while vCPU B is still rewinding (B
then skips its rewind), and a forward vCPU calling a function would have the callee's prologue read
another vCPU's `REWINDING` and spuriously rewind. So concurrent rewind is impossible while the state word
is global. (The shadow-**SP** word was already relocated per-context in 4A.5 stage i — `shadow_switch`
keeps each context's SP in its own region — and concurrent *freeze* works because all contexts unwind on
the *same* global `UNWINDING` simultaneously; only thaw needs *independent* per-context state.)

*The fix: a per-context thaw-state word (region-relative).* Move the state word from `ConstI64(STATE_OFF)`
to a **shadow-base-relative** load in the transform, so each context reads/writes the state word in *its
own* region (mirroring the SP word). Then each vCPU rewinds against its own word with no cross-talk, and
the JIT can run them as concurrent OS threads. This is the only IR change; because the transform is
**shared**, both backends pick it up — the interp's per-vCPU `dstate` then lives directly in its region
word (its multiplex/`shadow_switch` swap of the state word simplifies away), and cross-backend artifact
equality is preserved at freeze time (the per-context word holds the same value the global word did). It
is a **snapshot-format change** (the state byte moves out of offset 0 into the regions) ⇒ `FORMAT_VERSION`
bump + regenerate the cross-backend equality fixtures. The global `UNWINDING` freeze-trigger can stay a
broadcast (the concurrent-freeze coordinator already visits every live context), or fold into the
per-context word set on all contexts at once — TBD in stage 1.

*Staging (each stage independently lands + tests green):*
1. **Per-context thaw-state relocation (LANDED).** The durable state word is split: the **freeze** state
   (`UNWINDING`) stays at the single global `STATE_OFF` — a freeze is genuinely stop-the-world, so one word
   is the natural broadcast every poll reads (the arm trigger / `request_freeze` are unchanged) — while the
   **thaw** state (`REWINDING`/`NORMAL`) moves *per-context*, into each region at `STATE_IN_REGION_OFF` (8,
   just past the in-region shadow-SP word), addressed via `durable.shadow_base` like the SP word. Each frozen
   vCPU now rewinds against its **own** thaw word, so one finishing (flipping its word to `NORMAL`) can't
   disturb a sibling still `REWINDING` — the prerequisite stage 2 needs to run rewinds concurrently. The
   *inline* serial thaw is kept (no concurrency yet); all freeze/thaw + cross-backend equality + fuzz tests
   stay green. Implementation notes:
   - *Transform:* `Bb::freeze_word_addr` (global) for the `UNWINDING` polls; `Bb::thaw_word_addr`
     (`durable.shadow_base` + `STATE_IN_REGION_OFF`) for the prologue's `REWINDING` dispatch and the
     deepest frame's `NORMAL` re-issue. (Stage 1a centralized these behind one switched helper; 1b split it
     and hardcoded per-context — no flag, git is the revert.)
   - *Layout / format:* the shadow frame-base shifts past the in-region thaw word (`REGION_HEADER_LEN` 8→16,
     8-aligned); `FORMAT_VERSION` 6→7 (a v6 artifact mis-thaws). Both backends shift identically, so
     cross-backend equality holds.
   - *Per-vCPU multiplex (interp):* `dstate` maps across the two words — `durable_load_dstate`/`store_dstate`
     route `REWINDING` to the context's region word and the freeze phases to the global word. Fiber switches
     (`shadow_switch`, and the JIT's fiber resume) **carry** the active thaw phase across the switch, so the
     globally-deepest frame's `NORMAL` flip still propagates back up a `cont.resume` chain (a resumer doesn't
     flip its own word; the carry does on the return switch).
   - *Thaw entry clears the global freeze word:* a frozen artifact left `STATE_OFF = UNWINDING`, but a thaw
     is not a freeze; the runtime (interp `drive`; the `begin_thaw` test helper / JIT driver) resets it to
     `NORMAL` so the rewinding code's polls don't re-unwind. The per-context thaw word carries the
     `REWINDING` phase instead.
2. **Concurrent JIT thaw driver (LANDED).** `thaw_reattach_and_run` now **re-spawns each frozen vCPU on
   its own OS thread** (via `run_child`, mirroring stage ii) instead of the inline serial loop: a child
   carries a `DurableChild.thaw_extent = Some(extent)` so `run_child` starts it `REWINDING` from its
   restored extent against its *own* per-context thaw word (stage 1b), concurrent with its siblings and
   the root; the root re-enters, rewinds, and `run_inner`'s `join_all` joins the children at run end. The
   `Domain.thawing` blanket fail-closed is gone — a re-issued `atomic.wait` parks on the real futex and a
   sibling's re-issued `atomic.notify` wakes it (the `concurrent_freeze_..._thaws_via_sibling_notify`
   test: a producer↔consumer pair frozen mid-rendezvous now resolves). Replacing the blanket fail-closed,
   `futex_wait` gained **peer-aware deadlock detection**: an infinite wait whose `peers_live()` (`live > 1`)
   has gone false can never be satisfied (a parked waiter can't notify itself; a wasm wait returns only on
   notify/timeout), so it fails closed with `ThreadFault` (the interp's join-deadlock) within `KILL_RECHECK`
   of the last peer exiting — general (helps fresh runs too), not thaw-specific. **Join ordering:**
   `thread_join` resolves its per-vCPU table by the per-OS-thread task (`CONCURRENT_SPAWN_TASK`), not the
   shared `cur_task`, so a concurrent child's nested grandchild-join routes correctly. *Known gap:* a
   *mutual* block (two live vCPUs each waiting on the other) isn't caught by the `live`-count heuristic —
   the interp catches it via scheduler quiescence; a future cross-thread quiescence check would close it.
3. **Determinism / equivalence (LANDED).** A concurrent thaw reintroduces real interleaving; §12.6 holds:
   rewind reloads recorded side effects (deterministic, unaffected by order), and only the forward-phase
   re-issued waits/notifies interleave, re-synchronizing to the same value handoff. What landed:
   - *Mutual-block deadlock detection.* The stage-2 `live > 1` heuristic only caught a *lone* waiter (it
     missed a **mutual** block — two live vCPUs each blocked on the other). Replaced with a quiescence
     check: `futex_wait`'s `peers_live` is now `live > parked`, where `Domain.parked` counts vCPUs blocked
     in `atomic.wait` **or** `thread.join` (a `ParkGuard` RAII increments/decrements it at both park
     sites). When every live vCPU is parked (`live == parked`) no notifier can run, so the wait fails
     closed with `ThreadFault` (the interp's join-deadlock) instead of hanging — general (helps fresh runs
     too). Tests: `mutual_wait_block_fails_closed_not_hangs` (cross-wait, no notify ⇒ `ThreadFault`) and
     `mutual_rendezvous_resolves_without_false_deadlock` (cross-notify 2-way barrier ⇒ resolves; the two
     are never parked at once, so the check must not over-fire). `parked` must count **every** site a
     vCPU can block — `atomic.wait`, `thread.join`, **and `join_all`** (run teardown): a vCPU in `join_all`
     has finished its guest code and can never notify, so omitting it let a sibling that unwound (e.g.
     propagating a trap, skipping its own joins) leave a waiter seeing `live > parked` forever (a flaky
     ~1/6 hang until that was fixed).
   - *Interleaving stress (landed).* `concurrent_freeze_thaw_is_deterministic_across_interleavings`
     re-runs the multi-vCPU freeze/thaw 10× across different real OS-thread schedules, asserting the
     uninterrupted oracle each time. It reuses the unchanged `concurrent_freeze` helper (the CI hangs that
     first appeared alongside a 20× version were the mutual-block deadlock bug above, not the loop — each
     iteration's spin-wait `FreezeController` is short-lived, so looping a modest count doesn't starve a
     runner). The helper's controller is left as-is so the per-shape tests keep their proven timing.
   - *Loom model (landed).* `loom_deadlock_detection_resolves_when_last_peer_exits` model-checks the
     quiescence detection: a consumer waits with a modeled live-peer flag; the peer goes non-live + wakes
     under the futex lock; under every interleaving the consumer returns `WAIT_DEADLOCK`, never blocking
     the model. (A loom-only check mirrors the real build's timed re-check, which loom's no-timeout model
     can't poll.)
   - *Remaining follow-up:* a **generated** random multi-vCPU module generator (vs. today's hand-written
     shapes) for deeper fuzz coverage, and the pure join↔join circular-deadlock case (unlikely on a spawn
     DAG; the `live == parked` check catches every *wait*-involved deadlock today).

Original 4A map:

- **Barrier adaptation (loom-verified, but NOT the live path).** The 4A.4 `quiesce_arrive` ran `unwind`
  *under* the `quiesce` lock to serialize the (then-shared) active-SP scratch; with per-context SP that
  serialization is unnecessary, so `unwind` now runs outside the lock and the lock guards only the join.
  The O-A4 loom model is updated to per-context SP words
  (`loom_quiesce_barrier_never_hangs_with_per_context_sp`). **However**, stage (ii) ultimately chose
  `run_inner`'s existing `join_all` as the coordinator-wait (each child's freeze-unwind into its own
  region completes its OS thread; `join_all` blocks until all do), so `arm_quiesce`/`quiesce_arrive`/
  `quiesce_wait_all` are **unused in production** (`cfg(loom)`-only). They are retained as a verified
  primitive for a possible future *park-in-place* quiesce (workers that stop without ending their OS
  thread). The "Concurrent-durable entry + arming" bullet below describes that original barrier-based
  design; the shipped design uses `join_all` instead.
- **Concurrent-durable entry + arming.** A new multi-vCPU+interruptible entry (combining
  `..._durable_mv` and `..._interruptible`) calls `arm_quiesce(runners)` before any worker can observe a
  freeze, engaging `is_concurrent_durable()`. Single-worker paths stay byte-identical (`quiesce == 0`,
  lock untouched).
- **Concurrent children get shadow contexts + seeding.** Today `thread_spawn` only reserves a shadow
  context on the *deferred* (single-worker) path (`defer_spawn` → `reserve_vcpu_context`); the concurrent
  `run_child` path reserves none (non-durable children never freeze). For a concurrent **durable** run,
  `thread_spawn` must `reserve_vcpu_context()` for the child and `run_child` must
  `durable_shadow::seed(shadow_region_base(ctx))` before guest code (NORMAL never reads the SP word, so
  this matters only once a freeze fires).
- **Child freeze-unwind → residue → barrier.** On observing `UNWINDING` at a back-edge poll the child
  unwinds into its own region and returns the placeholder; `run_child` detects the freeze-unwind (window
  `UNWINDING` + child spilled past its frame base), records a `FrozenVCpu` (its task/parent/extent) into
  the shared residue (under the existing lock), and calls `quiesce_arrive(|| {})` — the unwind already
  happened in guest code, so the closure is empty (or records the extent). Coordinated so the child's OS
  thread still completes and joins normally for teardown.
- **Coordinator.** The root observes `UNWINDING`, unwinds, then (as coordinator) `quiesce_wait_all()` and
  runs the **existing** single-worker `freeze_drive` + residue flatten + snapshot — untouched, since by
  then every child has quiesced into its own region. Residue is canonically sorted before serialize
  (§12.6), so the quiesce order can't change the artifact.
- **Test.** Two children enter poll-free compute loops; `request_freeze` (or armed back-edge trigger on
  all) → both unwind concurrently; the artifact is **byte-identical** to the single-worker (deferred)
  freeze of the same program, and round-trips. Extend the O-A4 loom model if new shared state appears.

##### Context recycling plan (next sub-slice)

Today neither backend recycles fiber slots or vCPU contexts (they grow monotonically), so a long-lived
durable domain that churns fibers/threads eventually exhausts the `MAX_SHADOW_CTX` (~15) reserve. Lifting
that needs recycling, which is **only safe with generation-carrying handles** — a freed slot/context can
be reused, so a stale or forged handle to the old occupant must be rejected, not silently aliased to the
new one. This is a **cross-backend** change (both registries + the snapshot format), best done in its own
sequenced slice:

1. **Generation-carrying fiber handles (both backends).** Make a fiber handle `(generation, slot)` and
   validate the generation on `cont.resume`/`thread.join`-style use; bump the generation when a slot is
   freed. Carry the generation in the handle namespace (matching the loom-checked `Ownership` protocol).
   *Behavior-preserving until step 3.*
   - **[~] Interp side done.** The interp registry carries a per-slot generation (`RegState::gens`); a
     guest handle is `(generation << FIBER_GEN_SHIFT) | slot` (slot in the low 16 bits, since
     `MAX_FIBERS = 1<<16`); `cont.resume`'s `claim` rejects a generation mismatch. All generations are 0
     until step 3, so a handle is exactly its slot — byte-identical to before and to the JIT. Pinned by
     `svm-durable/tests/fiber.rs::forged_fiber_generation_is_rejected`. Cross-backend parity unaffected.
   - **[~] JIT side done.** `svm-jit`'s `fiber_rt` `cont.new` now emits `(generation << FIBER_GEN_SHIFT)
     | slot` (`generation()` of the fresh slot — 0) and `cont.resume` claims via the new generation-
     checked `Ownership::claim_gen(handle_gen)` instead of `claim` (which read the generation from the
     current word and so couldn't reject a stale handle). Behavior-preserving at generation 0
     (handle == slot); cross-backend parity verified (`durable_jit`/`durable_fibers_jit` byte-identical).
     Loom re-checked: `loom_claim_gen_is_exclusive_across_threads` (single-owner still holds — the
     generation check only *adds* a reject), plus `claim_gen_rejects_a_stale_generation` /
     `claim_gen_matches_claim_at_generation_zero`. `claim` is retained (`#[allow(dead_code)]`) as the
     ungated primitive + ABA characterization.

   With step 1 complete on both backends, **step 3** (recycle-on-finish) can wire `recycle_owned` on the
   live path: a finished slot's generation is already bumped, the handle carries it, and the claim
   rejects stale handles — so reuse is now ABA-safe. *(Resolved in step 3: the JIT freeze driver's
   `runnable_handles()` now encodes the generation and `fiber_region_base` is only ever passed a real
   resolved slot — never a raw handle — so the freeze/thaw path is generation-correct on both backends.
   The wired JIT `cont.resume` → `resolve` + `claim_gen` ABA guard is pinned end-to-end against the
   interpreter oracle by `jit_fibers.rs::{fiber_forged_generation_faults_identically,
   recycled_slot_generation_guard_agrees}`, and the recycled freeze/thaw path by the step-4 fuzz.)*
2. **[x] Snapshot format: carry generations — done, end-to-end (interp).** `FrozenFiber` (both backends)
   gains a `generation`; the freeze records it (interp `registry.generation(slot)`, JIT
   `slot.own.generation()` read before `finish` bumps it), the control section carries it (format **v2**,
   one uleb per fiber), and `seed_frozen` re-seeds at it (interp `gens[slot]`, JIT
   `Ownership::new_owned_at`) — so a thaw of a recycled fiber re-establishes the generation its guest
   handle expects. With the **mid-run freeze trigger** now in place (below), this is exercised end-to-end:
   `svm-snapshot/tests/roundtrip.rs::recycled_fiber_freeze_serialize_restore_thaw_through_the_codec`
   recycles slot 0 (fiber A finishes → generation 1), parks fiber B there, freezes mid-run, and confirms
   B is flattened + re-seeded at generation 1 and the thaw round-trips (also pinned at the codec leg by
   `fiber_residue_generation_round_trips_through_the_codec`). The JIT leg now matches (step 4): both
   backends armed-freeze the recycled fiber to a byte-identical artifact that thaws on either.
3. **[~] Recycle on finish — done, both backends.** A finished fiber's slot returns to a per-registry
   **min-heap** free list (`free`), and `cont.new` reuses the lowest free slot before growing — so the
   table is bounded by *peak concurrent* fibers, not the lifetime total, lifting the `MAX_SHADOW_CTX`
   cap to *concurrent* fibers. The reused slot keeps its bumped generation (interp keeps `gens[slot]`;
   the JIT replaces the slot's `Ownership` via `new_owned_at(gen)`), so a stale handle to the former
   occupant fails `claim`/`claim_gen` — the ABA guard from step 1. The JIT freeze driver's
   `runnable_handles()` now encodes the generation and `resolve` returns the slot index (so
   `fiber_region_base` uses the real slot, not the raw handle). Pinned by
   `recycling_reuses_a_freed_slot_with_a_bumped_generation` (interp) and the cross-backend `fiber_fuzz`
   churn differential. Two shadow-routing tests were updated so their fibers `suspend` (stay
   concurrently live) rather than return — otherwise the second fiber would reuse the first's freed
   region. *(vCPU-context recycling — a joined `thread.spawn` child's top-down context — is the sibling
   slice, now done; see "vCPU-context recycling" under slice 3.2.2 above.)*
4. **[x] Cross-backend parity + fuzz — done.** The recycled freeze/thaw leg is exercised on both
   backends, both hand-written and **fuzzed**:
   - *Pinned:* `svm/tests/durable_fibers_jit.rs::jit_and_interp_freeze_a_recycled_fiber_identically_and_thaw_on_the_jit`
     arms both backends to freeze a recycled (generation 1) parked fiber at the same safepoint,
     confirms a **byte-identical durable reserve + residue**, and **thaws the artifact on the JIT**.
   - *Fuzzed:* a recycling-churn generator (`durgen::gen_recycle_fiber_module` — recycle a slot 1..=3
     times → the real fiber lands at generation 1..=3, parked, frozen mid-run via `arm_freeze_after`)
     drives two properties: `durgen::fuzz_recycle_fiber_one` (interpreter freeze→thaw equals the
     uninterrupted run, residue carries the bumped generation) over 400 seeds in
     `durable_fuzz.rs`, and `durjit::fuzz_recycle_fiber_one_xbackend` (interp/JIT armed-freeze to a
     byte-identical reserve + §12 artifact, then thaw on the JIT) over 64 seeds in `durable_jit.rs`.
     LibFuzzer targets `durable_recycle` / `durable_recycle_jit` do the heavy continuous run.

### Mid-run freeze trigger ("freeze after N safepoints") — DONE (both backends)

The freeze mechanism unwinds at the first poll that observes `UNWINDING`. The before-start harness sets
that word *before* the run, so it can only freeze at the very **first** safepoint — too early to ever
hold a recycled (generation > 0) parked fiber, which needs a prior fiber-finish (a prior safepoint).

A new **`STATE_ARMED`** state value + an **`ARM_COUNTDOWN_OFF`** window word (in the reserve's unused
`[16, 64)` gap) make the freeze land *mid-run*, deterministically: `arm_freeze_after(win, N)` writes
`ARMED` + countdown `N`; the runtime decrements the countdown at each **fiber safepoint**
(`cont.resume`/`suspend`) and, at 0, promotes the word to `UNWINDING` so that op's trailing poll begins
the freeze. `ARMED` is **transparent** to the instrumented IR — every emitted poll/prologue tests only
`UNWINDING`/`REWINDING`, so an armed run reads as `NORMAL` until promotion, and an *unarmed* run never
touches the countdown (byte-identical to before). The interpreter ticks at its per-op dispatch; the JIT
ticks in the `fiber_resume`/`fiber_suspend` thunks (`window_tick_arm`, gated on `FiberRuntime::durable`).
**Both backends count the same set — the fiber ops, routed through runtime thunks** — so an armed freeze
lands at the same safepoint on each (cross-backend parity, which the recycled round-trip test pins).
`cap.call` is deliberately **not** counted: the JIT's cap.call thunk is host-supplied (no cross-backend
choke), a cap.call freeze is already reachable at the first safepoint, and the production async trigger
covers general mid-run freeze. This also models that production path (an async controller flipping
`UNWINDING` from another OS thread, picked up at the next poll — the existing mechanism already handles
that; what was missing was a *deterministic single-threaded* way to test it). Constants are cross-checked
in `layout_abi.rs`; placement is pinned by `svm-durable/tests/freeze_trigger.rs` (arm-after-N freezes
after exactly N fiber safepoints; arming past the last runs to completion; an unarmed run is untouched).
Note this is the **deterministic test trigger**, not the bounded-latency STW story (Phase 4's back-edge
polls + `Blocking.work` cancellation — see the latency caveat); a poll-free compute loop still won't
reach a safepoint.

**Recycling status — DONE (all four steps, both backends).** Steps 1 + 3 give complete **non-durable**
slot recycling (table bounded by peak concurrent fibers, ABA-guarded by generation-carrying handles);
step 2 carries the generation through the snapshot format; the mid-run freeze trigger makes a recycled
*durable* freeze/thaw round-trip reachable; and step 4 exercises it on both backends, hand-written and
fuzzed (byte-identical artifacts). The arc is complete — no recycling follow-ups remain.

### Fiber handles are `i64` (48-bit generation) — DONE (both backends)

The fiber guest handle widened from `i32` to **`i64`**: 16-bit slot (`MAX_FIBERS = 1<<16`) + **48-bit
generation**. The ABA guard's generation field was only 16 bits while the handle was an `i32`, so a
stale handle to a slot recycled exactly a multiple of 2¹⁶ times could falsely re-match (memory-safe and
domain-local — the wrong *own* continuation — but a real violation of the "stale handles fault"
invariant, and 65536 is small for a forever-running durable service). 48 bits moves wraparound to 2⁴⁸
recycles — unreachable in practice (centuries even at 10⁶ finishes/s).

The change is a type-system change anchored at the verifier (`cont.new` yields `i64`; `cont.resume`'s
handle operand is `i64`; the `status` result stays `i32`), mirrored through **three** value-type
copies — `svm-verify`, the `svm-durable` transform's own `result_types` (used to spill/reload the
handle across suspends), and both backends' runtime/codegen — plus the `FIBER_GEN_MASK` /
`FIBER_HANDLE_GEN_MASK` widening (interp + JIT), the `FrozenFiber.generation` field (`u32`→`u64`), and
the C/LLVM on-ramps (`int64_t` handle; chibicc widens the resume handle, mirroring svm-llvm
`operand_i64`). **Snapshot format bumped to v3:** the residue generation alone is wire-compatible
(`uleb`), but a handle held live across a suspend now spills **8** bytes in the shadow stack instead of
4 — the window-image layout changed, so a v2 artifact would mis-thaw and is rejected. Covered by the
existing fiber/recycling suites (all migrated to `i64` handles): `jit_fibers`, `fiber_fuzz`,
`fiber_migrate`, `durable`/`durable_jit` recycle fuzz, the C-frontend fiber demos, and the LLVM on-ramp.

#### 3.1 implementation plan (next-session pickup)

Done: the transform recognizes the fiber ops + NORMAL-inert (PR #27, branch
`claude/durable-phase3-design`, commit `4403d41`). The remaining 3.1 slices, in order,
each a small reviewable commit on the interpreter only:

1. **[DONE] Per-fiber shadow-stack layout + shadow-SP swap — the runtime maintains the
   swap (D-fiber-cont option A), *not* the transform.** Generalizes the durable reserve from
   one shadow stack to one **per fiber/context**: context `i` owns `[SHADOW_BASE +
   i*SHADOW_STRIDE, +SHADOW_STRIDE)` within `[0, DURABLE_RESERVE)` (root = context 0; fiber
   slot `s` = context `s+1`). The *active* shadow-SP stays at the fixed `SHADOW_SP_OFF`, and
   the **interpreter's `cont.*` execution** save/restores it to/from a per-context saved slot
   on every fiber switch (`shadow_switch` in `svm-interp`, called from the `cont.resume`,
   `suspend`, and fiber base-frame `Return` arms); `cont.new` assigns the new fiber's region
   (refusing — clean `FiberFault` — if the reserve is full). A non-running context's saved-SP
   lives host-side (the root's on `VCpu::root_shadow_sp`, a fiber's in the registry's parallel
   `shadow` table); the running context's live SP is the in-window word the instrumented IR
   maintains.

   **Why option A (not "the transform emits the swap").** The switch knowledge lives in the
   runtime's resume chain. Two of the three switch points (`cont.resume`, `suspend`) are
   visible IR ops, but the third — a fiber's **base-frame `Return`** — is a `Return`
   terminator statically indistinguishable from an ordinary intra-fiber call return, so the
   transform *cannot* emit its swap; only the interp (which knows it is a base return) can.
   Emitting the resume/suspend swaps in IR would also force reconstructing the resumer chain
   in guest memory (the interp already has it). Option A handles all three points, keeps the
   transform simple, and costs only that the JIT must replicate the ~3-site swap in its own
   fiber-switch path in 3.3 (guarded by the cross-backend artifact-equality property). Cost
   acknowledged: the layout constants (`SHADOW_SP_OFF`/`SHADOW_BASE`/`DURABLE_RESERVE`) are
   now duplicated across the TCB `svm-interp` and tooling `svm-durable` — cross-checked by
   `svm-durable/tests/layout_abi.rs` so they can't drift.

   Gated on a domain-level `Host::set_durable` flag (propagated to every vCPU by `drive`), so
   a non-durable fiber run never touches the reserve. Tests: existing single-vCPU durable
   tests still pass (root = context 0); `svm/tests/durable_fibers.rs` proves two fibers get
   **distinct** regions (a host-fn probes the active shadow-SP from inside each context) and
   that a non-durable run leaves the reserve untouched. *Touched only `svm-interp` (the swap +
   region tracking) — the transform is unchanged.* **Open sub-question deferred:** the
   transform's shadow-overflow guard still trips at the global `DURABLE_RESERVE`, not a
   per-region bound, so a fiber recursed past `SHADOW_STRIDE` could grow into a neighbor's
   region before tripping — harmless for shallow fibers, fixed alongside the sizing decision.

2. **[DONE] `Resume` thaw arm** (`cont.resume`, resumer side). Mirrors `SuspendKind::Propagated`'s
   re-issue, but emits `Inst::ContResume { k, arg }` (operands reloaded from the spilled slots —
   `used[k]/used[arg]` already mark them) and threads its **two** results `(status, value)` into
   the continuation; the fail-closed trap arm now applies to `Yield` only. *Touched only
   `svm-durable`.* Tested structurally (`svm-durable/tests/fiber.rs`): the instrumented module
   verifies, stays NORMAL-inert, and gains one re-issued `cont.resume` per resume point while
   `suspend` gains none (its arm is still a bare `Unreachable`). **A full thaw that re-enters a
   suspended fiber is not yet exercisable** — the re-issued resume reconstructs the fiber via the
   fiber's *own* rewind (the `Yield` re-park), which is slice 3.1.3; the round-trip test lands with
   3.1.3–5 (the fiber re-park + freeze driver + snapshot Section-2 fiber metadata).

3. **[DONE] `Yield` thaw arm — re-park** (the novel bit). On thaw of a `suspend` point the arm
   reloads the fiber's frame, pops, **flips the state word to `NORMAL`**, and **re-executes
   `suspend`** — which parks the fiber back (its current rebuilt frames) and hands `value` to
   the resumer, *not* continue forward; the re-executed suspend's result (the next resume's
   value) threads into the continuation exactly as a leaf's reloaded cap.call result does. The
   deepest-frame "flip to NORMAL" lives here (a parked fiber's `suspend` is the globally-deepest
   frozen frame), **not** in the resumer's `Resume` arm — so the resumer regains control already
   in NORMAL. *Turned out transform-only:* the interp's existing `Suspend` handler re-parks via
   the registry, and the 3.1.1 shadow-SP swap routes the SP, so no `RegFiber` change was needed.
   Tested structurally (`svm-durable/tests/fiber.rs`): both fiber arms now re-issue their op and
   the only `Unreachable` blocks left are the per-function forged-id TRAPs. **End-to-end thaw
   still needs 3.1.4–5** (a parked fiber's continuation isn't captured until the freeze driver
   flattens it into its shadow stack and the snapshot records its metadata).

4. **[DONE] Freeze driver — flatten idle parked fibers.** `VCpu::freeze_drive`, hooked into
   `dispatch` right after the root's run returns `Done` while still `UNWINDING` (the registry is
   still alive there, before `mem.take()`). It loops over `RegFiber::Parked` fibers, marking each
   `Frozen` and running its frames as a standalone unwind: the active shadow-SP is pointed at the
   fiber's region base, `cur = ROOT_FIBER` (so the fiber's base-frame return ends the sub-run, not
   a fiber-finish), and a placeholder resume value is delivered (mimicking `cont.resume`; the
   suspend's result slot is inert — the `Yield` arm redelivers it). Because the transform places
   the poll **immediately** after the `suspend`, that poll fires before any guest code runs → zero
   forward progress; the flattened shadow-SP extent is recorded in the registry's `shadow` table.
   The existing capture entry point then snapshots a window that already includes the flattened
   fibers. *Host-side driver, not escape-TCB; single-vCPU (3.1) — a fiber still on an active resume
   chain at freeze, and multi-vCPU STW, are 3.1.5/3.2 follow-ups.* Tested
   (`svm-durable/tests/fiber.rs`): a parked fiber lands a frame in its **own** region (distinct
   from the root's) and unwinds **at its suspend point** (resume id 1) — a precise zero-forward-progress check.

5. **[DONE (interp round-trip); snapshot codec follow-up] End-to-end test + fiber residue.** The
   freeze driver records each flattened fiber as a `svm_interp::FrozenFiber` (slot, entry funcref,
   data-sp, shadow-SP) and hands it back through the `Host`; `freeze_drive` leaves the active
   shadow-SP at the **root's** region so the captured window is thaw-ready. A thaw re-seeds the
   registry (`drive` recreates each as a `Pending` fiber at its dense slot, with its shadow-SP in
   the `shadow` table) and re-enters the root under `REWINDING`: the resumer re-issues
   `cont.resume`, the seeded fiber re-runs its entry → rewinds → re-parks, then forward execution
   completes. `svm-durable/tests/fiber.rs::single_fiber_freeze_thaw_round_trips` proves `freeze →
   (window + residue) → thaw ≡ uninterrupted` (107), and the `durable_fuzz` fiber property fuzzes
   it over a generated root+fiber space (varying suspend counts, live-across-suspend values,
   multi-point resume/suspend). The byte-level **Section-2 codec** (below) carries the residue
   through the real artifact too.

   **[DONE] Section-2 codec.** `svm-snapshot` now carries the fiber residue: `freeze` writes a TLV
   control section (tag 2) of `(slot, funcref, sp, shadow_sp)` per fiber — ascending slot,
   header `fiber_count`-gated, **elided when there are no fibers** so the no-fiber artifact is
   byte-identical to the pre-fiber format — and `restore` decodes it and re-seeds the `Host`
   (`set_frozen_fibers`). `svm-snapshot/tests/roundtrip.rs::fiber_freeze_serialize_restore_thaw_through_the_codec`
   drives the full round-trip through the **serialized artifact** (107) and the §12.6 canonical
   re-serialize invariant. (Per-fiber `generation` is deferred with the JIT shared-registry
   recycling work; interp slots aren't recycled, so slot alone keys the handle today.)

Then 3.2 (multi-vCPU) and 3.3 (JIT parity) as above. **Slice-1 sub-questions, now settled:**
a fiber handle maps to its region by **dense slot index × stride** (`context i = slot+1`,
base `SHADOW_BASE + i*SHADOW_STRIDE`); the resume chain depth needs **no explicit recording**
— each fiber's saved-SP is tracked independently (registry `shadow` table + the root's
`root_shadow_sp`), so per-fiber rewind falls out. **Still open:** per-fiber shadow-stack
*size* + quota accounting (slice 1 uses a provisional 4 KiB `SHADOW_STRIDE`, ~15 contexts),
and with it the per-region overflow bound (the guard still uses the global `DURABLE_RESERVE`
ceiling — see slice 1 above).

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
