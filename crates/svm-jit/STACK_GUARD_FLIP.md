# Flipping arena stacks + the software guard on by default — decision tracker

Companion to `STACK_GUARD.md` (which covers the *mechanism*). This file tracks the **decision** to
promote `svm-fiber/arena-stacks` + `svm-jit/stack-check` from off-by-default prototypes to the default
fiber model, and the blockers gating it.

Status: **executing the staged flip.** Blockers cleared. Staged rollout: PR1 (cross-os gating) —
user-applied on `main`; **PR2 (software check always-on) — DONE (this change)**; PR3 (arena default)
— next. `arena-stacks` is still off by default (guard-page backend remains default until PR3); the
software check is now **always emitted** (no longer behind `stack-check`).

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
| 4 | **GC precision** — reused, un-zeroed arena slots added *false roots* to the conservative scan (never unsound — a superset only retains). **FIXED (minimal):** `gc_roots` now scans running fibers over the tight `[live_sp, top)` instead of `[usable_low, top)` — `live_sp` from the resume chain (`resumer_sp()` for ancestors, `current_sp()` for the innermost). The unused region below `live_sp` (where arena stale bytes live) is no longer read, so **zeroing is unnecessary**. Parked fibers were already exact (`[ctx, top)`). A broader precise-stack-map GC redesign is deferred separately (user). | Follow-up | **done (minimal)** — tight running-scan + regression test `gc_roots_scans_running_ancestor_fiber_on_the_jit` |
| 5 | **Frame-size backstop** — **scoped out as a gate (not needed for realistic code).** The check is sound for every frame the compiler can emit (see "Frame-size scoping" below); the residual is only an address-space *wrap* (~2^47 B), which is unreachable. Cranelift 0.132 doesn't expose the frame size anyway (`CompiledCode` = buffer/vcode/labels/bb only). A belt-and-suspenders prologue-parse remains *possible* but is fragile and unnecessary. | Theoretical — no action | **retired as gate** |

Items 1–3 are the gate; all effectively done. **#4** is a non-gating GC follow-up (the minimal tight
running-scan already landed). **#5** is retired as a gate (frame-size scoping below). ⇒ the technical
blockers are cleared; what remains is execution (apply the CI line, flip the default features, move the
check into the always-on verifier-trusted path).

## Frame-size scoping (#5 — why it is not a gate)

The check is `SP - RED_ZONE <u limit`, emitted in each function's entry block **after** the machine
prologue's `sub rsp`, with `enable_probestack` pinned **off** (audit F1). Consequences:

- `sub rsp` (even for a huge frame) is a pure pointer move — it touches no page before the check.
- The **only** writes before a callee's check are the `call` return-address push + the callee-saved
  register spills in its prologue — ABI-bounded to ≤ ~224 B (x64 SysV 6 GPR = 48 B; Win64 + XMM6–15 ≈
  224 B), i.e. **≪ `RED_ZONE` (16 KiB)**. The caller's passed check guarantees ≥ `RED_ZONE` headroom,
  which covers them with ~70× margin.
- Any frame that lowers SP below `limit + RED_ZONE` therefore **traps before its first frame write**,
  independent of frame size (the compare has no size term). The fuzz confirms clean `StackOverflow`
  for frames past `RED_ZONE`; the mechanism is identical for larger frames.
- The one regime the runtime compare can't catch is `sub rsp` **wrapping** the address space (frame
  ≈ 2^47 B). That is unreachable: the compiler cannot emit it — a 264 KiB (33 000-slot) frame already
  hangs regalloc past 10 min, so a 2^47 frame can't be compiled at all. (Pathological frames are thus a
  *compile-time* resource concern, not a runtime escape — orthogonal to the guard.)

Cranelift 0.132 exposes no frame size (`CompiledCode`: `buffer`, `vcode`, `value_labels_ranges`,
`bb_starts/edges`; `code_info()` = code `total_size` only), so a compile-time bound would need a
dependency bump (unproven it's exposed later; risky for escape-TCB) or a machine-code prologue-parse of
`code_buffer()` for the `sub rsp, imm` (fragile, per-arch). Given the analysis above, neither is
warranted. **Recommendation: ship the flip without a frame-size backstop.**

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

Two independent follow-ups fell out:
- **(a) — RESOLVED for fibers by PR2.** Making the software check always-on means a *fiber* overflow now
  traps `StackOverflow` through `trap_out` (no signal) ~`RED_ZONE` above the guard page, so it never
  reaches the page and never double-faults — the DoS is gone for fibers on the default guard-page
  backend. (The root / spawned-vCPU tops still run `limit = 0` on OS stacks, so a `sigaltstack` would
  only matter for a deeply-recursive *root* JIT computation — a much narrower, still-open case. A
  dedicated `sigaltstack` install is therefore optional now, not a DoS fix.)
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
- [x] GC precision (#4) — **done (minimal)**: tight `[live_sp, top)` running-scan lands the arena-imprecision fix; zeroing unnecessary. Broader GC redesign deferred separately.
- [x] Frame-size residual (#5) — **explicitly accepted**: sound for all emittable frames; residual is an unreachable address-space wrap (frame-size scoping above). No backstop shipped.

Then flip **both** flags together, per platform, and move the check into the always-on verifier-trusted path.

## Status: technical blockers cleared

#1 fuzz ✅ · #2 audit ✅ (no escape vector; F1/F2 fixed) · #3 CI ✅ (job runs the fuzz on merge; one
guard-page-oracle line for `main`) · #4 GC ✅ (minimal tight running-scan; zeroing unnecessary) ·
#5 frame-size ✅ (retired as a gate). Remaining is **execution**, not open risk: (1) apply the CI
guard-page line on `main`; (2) flip `svm-fiber/arena-stacks` + `svm-jit/stack-check` on by default,
per platform; (3) move the software check into the always-on verifier-trusted path (drop the feature
gate) so it cannot be compiled out. A broader precise-stack-map GC redesign remains a separate,
non-blocking follow-up.
