# Flipping arena stacks + the software guard on by default — decision tracker

Companion to `STACK_GUARD.md` (which covers the *mechanism*). This file tracks the **decision** to
promote `svm-fiber/arena-stacks` + `svm-jit/stack-check` from off-by-default prototypes to the default
fiber model, and the blockers gating it.

Status: **blockers in progress.** Both features are still off by default.

## Why flip

The default fiber backend maps one guarded reservation per fiber (`mmap` + `mprotect(PROT_NONE)`),
costing **2 VMAs/fiber**, so a process hits Linux `vm.max_map_count` (65 530) at **~32 754 concurrent
fibers** — measured, well short of the millions-of-fibers goal. The arena allocator sub-allocates
256 KiB slots from a few large reservations (~0 extra VMAs/fiber) and recycles via a free-list, but it
**drops the per-fiber hardware guard page**, so overflow protection has to move into a software
stack-limit check the JIT emits in every function prologue.

## Perf — measured, not the constraint

| | Default (guard page) | Arena + software check |
|---|---|---|
| Fiber create/finish | 7.2 µs | **0.23 µs (~30×)** |
| Concurrent fiber ceiling | ~32 754 | **2 000 000+** |
| Software check, per call | 0 | **~0.3 ns / ~1 cycle** |
| Software check, on real work | — | **≈0%** (data-independent, OoO-hidden) |
| Worst case | — | ~20%, do-nothing-leaf-call storm only |

Flipping is a strict throughput + scalability win. Throughput does **not** gate the decision.

## Security — mechanism sound, *assurance* is the gate

The check becomes **load-bearing for isolation**: with arena stacks there is no guard between adjacent
fibers, so an unchecked overflow corrupts a *neighbour* fiber's stack — a cross-fiber escape. It
therefore lives in the **escape-TCB**, alongside the masking lowering.

Already escape-grade about the design (see `STACK_GUARD.md`):
- emitted in **every** guest prologue, unconditionally; guest controls neither codegen nor the
  `stack_limit` param → cannot skip or move it;
- **per-vCPU-correct by construction** (path B: value ABI param, no shared cell, no TLS);
- runs **after** the machine `sub rsp`, so each frame is validated directly; pre-check pushes are
  covered by `RED_ZONE`;
- overflow routes through the existing `trap_out` / detect-and-kill path — no new signal surface.

## Blockers (must clear before flipping)

| # | Blocker | Severity | Status |
|---|---------|----------|--------|
| 1 | **Soundness fuzz** — differential: any overflow must trap `StackOverflow`, never write below `usable_low`. Oracle: run the guard-page backend WITH `stack-check`; a `MemoryFault` (or crash) then proves a hole. Sweep frame size across/past `RED_ZONE`. | Blocker (escape-TCB) | **landed** — `tests/stack_guard_fuzz.rs` (needs CI job #3 to run it) |
| 2 | **Audit** the check + arena as an escape-TCB unit (like masking). | Blocker | **done** — `STACK_GUARD_AUDIT.md`: no escape vector; F1 (pin probestack) + F2 (stale docs) fixed; F3→#5, F4→#4 |
| 3 | **CI coverage** — build/test `--features stack-check[,arena-stacks]` on Linux + macOS-aarch64, and give the Windows arena its first runtime coverage. Without it the guard bit-rots while off by default. | Blocker | **mostly done** — `stack-guard` job exists; fuzz runs under it on merge; needs the guard-page-oracle run added (below) |
| 4 | **GC precision** — reused, un-zeroed arena slots add *false roots* to the conservative scan. **NOT a flip-blocker** (audit): soundness is preserved — a superset only *retains* dead objects, never premature-frees; retention is bounded by the reused region below high-water. Root cause is `full_extent()` scanning `[usable_low, top)` for running fibers; the fix is a tighter `[live_sp, top)` scan (bounds already tracked: `parked_extent` exact, `resumer_sp()`/`current_sp()` for running), which makes zeroing moot. Decoupled → GC follow-up. | Follow-up (not gating) | open |
| 5 | **Frame-size backstop** — reject any function whose frame exceeds `SLOT − RED_ZONE`. The audit (F3) **raises this**: under the arena the check is the *sole* defense (no guard-page backstop), so this single-frame gap should close *with* the flip, not after. Cranelift 0.132 doesn't expose `frame_size` (needs a prologue-parse or a Cranelift upgrade). | Low escape-prob, but arena-critical | open |

Items 1–3 are the gate. The audit reclassifies **#4 as a non-gating GC follow-up** (sound-but-imprecise
interim is acceptable) and **#5 as should-land-with-the-flip** (F3 — the arena has no hardware backstop).

## sigaltstack finding (surfaced while building blocker #1)

Building the fuzz's oracle turned up a real gap in the **current default** (guard-page backend, no
software check), independent of the flip:

- `trap_shim.c` sets `SA_ONSTACK` on the SIGSEGV/SIGBUS handler, but **nothing calls `sigaltstack()`**
  anywhere in the tree — the flag is inert. So the handler runs on the *current* stack.
- When a fiber's control stack overflows into its `PROT_NONE` guard page, the CPU is at stack
  exhaustion; the handler tries to push its frame onto the exhausted stack, **double-faults**, and the
  kernel kills the process (`SIGSEGV`, signal 11). Verified empirically: the same overflowing modules
  that trap a clean `StackOverflow` *with* the software check **crash the process** with it off.
- Consequences:
  1. The `TrapKind::StackOverflow` doc's "with the hardware guard page (default), overflow is a
     `MemoryFault`" is **false for stack-exhaustion faults** — it holds only for in-window memory
     faults (handler has ample stack). Doc should be corrected.
  2. **DoS in the current default**: a guest that deep-recurses inside a fiber crashes the whole host
     process (no corruption — the guard page stops the write — but the host dies). Availability, not
     escape.
  3. It **strengthens the flip case**: the software check traps through `trap_out` (no signal, no
     double-fault) and is the *only* path that converts fiber overflow into a survivable trap. Value
     beyond VMA scaling.

Two independent follow-ups fall out (neither blocks the flip, both worth filing):
- **(a)** Install a `sigaltstack` in the trap-handler setup so guard-page stack overflow is survivably
  caught even on the default backend (fixes the DoS above). Small, self-contained.
- **(b)** Correct the `TrapKind::StackOverflow` / `MemoryFault` docs re: stack-exhaustion faults.

## CI: guard-page-oracle run for the fuzz (apply on `main`)

The `stack-guard` job runs `cargo test -p svm-jit --features stack-check,arena-stacks`, which already
exercises `stack_guard_fuzz.rs` on the **arena** backend (proves the check fires on the backend it will
ship with). But the fuzz's ground-truth "wrote below `usable_low`" oracle is the `PROT_NONE` guard
page, which only exists on the **default (non-arena) backend**. Add one step to the `stack-guard` job
so the fuzz also runs there:

```yaml
      # The stack-guard soundness fuzz's hole-oracle is the PROT_NONE guard page, present only on the
      # default (non-arena) backend. The combined-features run above exercises the check on the arena
      # backend it ships with; this run gives it the ground-truth oracle (STACK_GUARD_FLIP.md #1).
      - run: cargo test -p svm-jit --features stack-check --test stack_guard_fuzz
```

(Place it right after the existing `cargo test -p svm-jit --features stack-check,arena-stacks` line.)

## Flip criteria (all true ⇒ promote)

- [ ] Fuzz (#1) lands and runs clean in CI over a sustained seed sweep, including frames past `RED_ZONE`.
- [ ] CI job (#3) builds + runs the guard/arena suites on Linux + macOS-aarch64; Windows arena has runtime coverage.
- [ ] Escape-TCB audit (#2) signs off the check + arena.
- [ ] GC precision (#4) — **not gating**; accepted as sound-but-imprecise interim, tracked as a GC follow-up (tighter running-scan).
- [ ] Frame-size residual (#5) either closed or explicitly accepted with rationale.

Then flip **both** flags together, per platform, and move the check into the always-on verifier-trusted path.
