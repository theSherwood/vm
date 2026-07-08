# Escape-TCB audit — software stack-check + arena stacks

Blocker #2 of STACK_GUARD_FLIP.md. Adversarial audit of the software stack-overflow guard
(`emit_stack_check` + the ABI-param plumbing) and the arena allocator (`stack_arena.rs`), treated as
escape-TCB code on par with the masking lowering. Scope: can a guest escape its window / corrupt
another fiber via the native control stack once `arena-stacks` + `stack-check` are the default?

**Verdict: no escape vector found.** The mechanism is sound end-to-end. Findings below are
hardening / assurance items; two cheap ones are already fixed in this change.

## What was verified sound (the load-bearing invariants)

1. **Unskippable, guest-uncontrolled.** `emit_stack_check` is emitted in the entry block of *every*
   guest function — the single `build_clif` lowering path (`lib.rs:4340`), after `emit_epoch_check`,
   before the jump to IR block 0. The guest supplies IR, not codegen, so it cannot omit, move, or
   branch around the check. Functions the JIT can't lower fall back to the interpreter, which enforces
   its own `MAX_CALL_DEPTH` — no unchecked native-recursion path exists.

2. **ABI slot is consistent everywhere** (a mis-index would silently make the check read a wrong value
   and go inert — the highest-risk failure). `stack_limit` is param **slot 3** (after
   `mem_base, fn_table_base, trap_out`, before `sret`/user args) in all four places that must agree:
   `sig_from` (def, `lib.rs:3796`), the entry-block params + `limit_var = params[3]` + `pbase`
   (`lib.rs:4251/4276/4334`), `ctx_args` (every call site, `lib.rs:5889`), and the fiber call
   trampoline pass-through (`lib.rs:4515`). Verified by reading all four; the `sret` index and `pbase`
   correctly add `cfg!(stack-check)`.

3. **Per-vCPU-correct by construction.** The limit is a *value* ABI param, constant within a stack's
   call tree, set anew at each fiber entry from `Yielder::stack_low()` (`fiber_rt.rs:680`). There is no
   shared cell and no TLS, so concurrent vCPUs cannot clobber each other's limit. The root and each
   spawned-vCPU top pass `limit = 0` (they run on OS-guarded thread stacks ⇒ check inert). Pinned by
   `stack_check.rs::spawned_vcpu_fiber_recursion_traps_stack_overflow`.

4. **Frames are validated directly.** The check sits *after* the machine prologue's `sub rsp` (entry
   block), so `get_stack_pointer` reflects this frame; `SP - RED_ZONE < limit` traps if the frame
   crossed the low bound. `RED_ZONE` (16 KiB) need only cover the pre-check register pushes (~tens of
   bytes) — it does, by ~100×. `stack_guard_fuzz.rs` empirically confirms clean `StackOverflow` (never
   `MemoryFault`/crash) for frames swept across and past `RED_ZONE`.

5. **`usable_low` is the true low bound on both backends.** Guard-page: `base + guard_page` (the check
   fires ~16 KiB above the `PROT_NONE` page — it never touches). Arena: the slot base itself, so the
   bottom `RED_ZONE` of each 256 KiB slot is a dead-zone margin and writes stay within the slot; the
   neighbour slot below is never reached.

6. **Arena allocator is memory-safe.** Cursor `cur_next < SLOTS_PER_ARENA = 4096`, so `cur_next * SLOT
   < ARENA` — no overflow; slots are SLOT-aligned and in-bounds (the reservation is `ARENA + SLOT`, and
   the base is rounded up to a SLOT boundary, leaving slack). The global `Mutex` serialises alloc/free;
   Rust ownership precludes double-free / aliasing. Reservation/commit failures return `None`
   (recoverable `FiberFault`), never abort.

## Findings

### F1 — [fixed] `probestack` off was an unpinned escape-TCB dependency
Finding 4's soundness for frames larger than `RED_ZONE` relies on `sub rsp` being a pure pointer move
that touches no pages before the entry-block check. If `enable_probestack` were ever on, a large
frame's prologue would page-walk *downward* below the fiber's low bound **before** the check — under
the arena (no guard page) that is silent neighbour-slot corruption, i.e. an escape. It defaults off in
Cranelift 0.132 but was not set explicitly, so a future default flip would open the hole silently.
**Fix:** both ISA flag builders now pin `enable_probestack = false` with a security comment
(`lib.rs`, top-level + child compile).

### F2 — [fixed] Stale docs on the check (misleads a security auditor)
`emit_stack_check` and the `stack_check` module doc described the *removed* increment-2 mechanism — a
global `STACK_LIMIT` cell with `set_limit`/`restore_limit` across the resume seam, and an "atomic load
opaque to the optimizer". The shipped code uses path B (a value ABI param, `limit_var`). **Fix:** docs
rewritten to describe path B and the after-`sub rsp` check position.

### F3 — [resolved by follow-up analysis] Arena has no hardware backstop; the check is the *sole* defense
On the guard-page backend a check hole is caught by the `PROT_NONE` page (a fault, not corruption). The
arena drops that page, so under the flip a check hole would be silent neighbour-fiber corruption. This
is by design (the VMA win requires dropping the page) and is *why* the check is escape-TCB. This audit
initially flagged a "single-frame-larger-than-the-slot" residual and routed it to a compile-time
frame-size backstop (#5). **A follow-up scoping of #5 (STACK_GUARD_FLIP.md) retired that concern:** the
check is sound for *every frame the compiler can realistically emit*. It runs after `sub rsp` with
`enable_probestack` pinned off (F1), so no page is touched before it; the only pre-check writes are the
return-address push + callee-saved spills, ABI-bounded to ≤ ~224 B ≪ `RED_ZONE` (16 KiB). Any frame that
lowers SP below `limit + RED_ZONE` therefore traps *before* its first frame write — regardless of size
(the compare is size-independent), up to a frame large enough to *wrap* the address space (~2^47 B),
which the compiler cannot emit (a mere 264 KiB frame already hangs regalloc). So the arena's sole-defense
posture is safe as-is; the backstop is defense-in-depth against an unreachable case, **not** a flip gate.

### F4 — [not an escape; decide under #4] Un-zeroed slot reuse is intra-guest data residue
A recycled arena slot is not zeroed, so a new fiber's stack initially holds the previous fiber's bytes
— and slots recycle across vCPUs (shared arena). This is **not** a VM escape: all fibers/vCPUs are the
same guest program in one window, one trust domain (the native stack is not the masked guest memory).
It does (a) interact with GC precision — a conservative `gc.roots` scan over a reused slot
over-approximates (sound superset), the open decision in STACK_GUARD_FLIP.md #4, and (b) violate a
guest assumption of zeroed stacks if one exists. **Recommendation:** fold the zero-on-reclaim decision
into #4; document the trust-boundary reasoning either way.

### F5 — [note; out of scope for the fiber flip] Root / vCPU-top JIT recursion stays OS-guarded
Root and spawned-vCPU tops run with `limit = 0` (check inert) on their OS thread stacks by design, so
their deep JIT recursion is bounded only by the OS guard page — which shares the `sigaltstack`
double-fault DoS (STACK_GUARD_FLIP.md "sigaltstack finding"). The flip does not change this (it only
moves *fibers* to the arena+check), so it neither improves nor worsens the root path. Flagged for
completeness: "flip the default" makes *fiber* overflow survivable, not *all* native-stack overflow.

### F6 — [cosmetic] Commit-failure skips a slot index (Windows)
On `commit_slot` failure `cur_next` is already incremented, so that slot address is never handed out or
committed — a minor address-space leak under memory pressure. The recoverable-fault contract holds.
Acceptable.

## Bottom line for the flip

The check + arena are escape-safe as written. Before flipping the default, **F3/#5 (frame-size
backstop) should land with the flip** so the arena's sole-defense posture has no single-frame gap, and
**F4/#4 (GC zeroing)** should be decided. F1/F2 are done. None of these is an escape in the current
(off-by-default) state.
