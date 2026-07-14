# `gc.roots` register-flush audit + fix

Audit of the JIT conservative root scan (`fiber_rt::gc_roots`) for the "roots in unspilled callee-saved
registers are out of scope" caveat the code long carried as a deferred follow-up.

## Question

Does the ambient native-stack scan miss a live guest heap root held only in a **callee-saved register**
(not spilled to the stack) — and if so, is it a VM escape?

## Findings

**1. Parked fibers and running ancestors are sound.** The fiber switch (`svm-fiber`'s `jump`) pushes
all six SysV callee-saved registers (`rbp rbx r12-r15`; aarch64 `x19-x28`+fp/lr) onto the outgoing
stack, and the scan extents start at the saved SP (`[ctx, top)` parked, `[resumer_sp, top)` ancestor),
so those registers are always captured.

**2. The current running computation had a real gap.** `gc.roots` is a plain call (no switch). The
Tail calling convention **preserves** the SysV callee-saved registers (`(Tail,false) → SYSV_CLOBBERS`),
so Cranelift parks a root that is live across the `gc.roots` call in one of them rather than spilling
it. The call to the host `gc_roots` (default SystemV) also preserves callee-saved registers, so it does
**not** force those roots to the stack — they are captured only if `gc_roots`' *own* compiled body
happens to clobber (and thus prologue-save) that register. That is incidental and not guaranteed —
**confirmed active**: the debug build's `gc_roots` is 808 instructions and references **no** callee-saved
GPR, so it saves none. A/B test (`gc_roots_captures_caller_callee_saved_register_roots_on_the_jit`): a
fiber with 16 pointers live across `gc.roots`, **without** the flush, reports only 13 of 17 roots — four
register-resident roots missed.

**3. It is NOT a VM escape.** A missed root makes the *guest's own* conservative GC free a
still-referenced object — a use-after-free **inside the guest's masked window**, which the guest can
already do to itself. The reported roots are filtered to `[heap_lo, heap_hi)` (guest-window offsets) and
the output buffer is masked/bounds-checked like any store. Memory confinement is untouched. This is a
**soundness bug in the `gc.roots` service** (it promised a conservative superset but under-reported),
not an escape — but it matters for any consumer whose GC relies on the scan.

## Fix

Route `gc.roots` through a **register-flush trampoline** (`fiber_rt::svm_gc_roots_flush`, one
`#[unsafe(naked)]` variant per target) that spills every callee-saved register onto the stack *before*
`gc_roots` runs, so a call-surviving root in any of them lands in the scanned `[current_sp, top)` region
— closing the gap by construction, independent of how `gc_roots` compiles. To keep each trampoline
trivial (flush + call, no per-argument reshuffle — important for the Win64 path with its stack args +
shadow space), the JIT marshals the ten scan arguments into one stack slot (`GcRootsArgs`) and passes a
single pointer, which rides in the first argument register untouched.

Cost is a handful of push/pops **only on a `gc.roots` call** — paid by gc-opting guests; guests that
never scan roots never reach the trampoline.

A/B-verified: with the flush all 17 roots are reported; bypassing it drops the count to 13.
