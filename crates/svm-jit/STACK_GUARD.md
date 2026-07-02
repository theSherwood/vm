# Software stack-overflow guard (production design)

Status: **design + foundational plumbing landed; check wiring is the next increment.**
Feature-gated behind `svm-jit/stack-check` + `svm-fiber/arena-stacks` (both off by default).

## Why

The per-fiber hardware guard page (`stack_unix.rs`: `mmap` + `mprotect(PROT_NONE)`) costs **2 VMAs
per fiber**, so a process hits the Linux `vm.max_map_count` default (65 530) at **~32 700 concurrent
fibers** — measured, well short of the millions-of-fibers goal. The arena allocator (`stack_arena.rs`)
sub-allocates stacks from a few large `mmap`s (~0 extra VMAs/fiber, ~30× faster create/finish — both
measured), but it **drops the hardware guard**. Overflow protection then has to move to a **software
stack-limit check the JIT emits in every function prologue**.

This check becomes **load-bearing for isolation**: with arena stacks there is no guard between
adjacent fibers, so an unchecked overflow corrupts a *neighbour fiber's* stack. It therefore lives in
the **escape-TCB** and must be treated like the masking lowering — always emitted, unskippable,
fuzzed, and audited.

## Measured cost (this box, prototype)

- Check: **~0.3 ns (~1 cycle) per call**; ~20% only on a do-nothing-leaf call storm, **≈0% on any
  function that does real work** (the check is data-independent and OoO-hidden behind the body).
- Arena alloc: **~30× faster** fiber create/finish (7.2 µs → 0.23 µs), concurrency 32 754 → 2 000 000+.

So throughput does not gate the decision; the escape-TCB correctness of the check does.

## The three problems and the chosen solutions

### 1. Mechanism — explicit CLIF (not Cranelift's native `func.stack_limit`)

Cranelift's native `func.stack_limit` emits a prologue check that traps via `ud2` → **SIGILL**, which
the host signal handler (`trap_shim.c`, SIGSEGV/SIGBUS only) does not catch → host crash; and there is
no `VMContext` param to source the limit from. So we **emit the check explicitly in CLIF**, modelled on
`emit_epoch_check` (the §5 interrupt poll), and route overflow through the existing `trap_out` /
detect-and-kill path — no new signal-handler surface. `TrapKind::StackOverflow = 13` is added for it.

### 2. Frame-size soundness — red-zone + post-compile frame bound

The prologue check runs *before* the frame is allocated, and the frame size isn't known at IR-build
time. Solution: check against a **red-zone** `RED_ZONE ≥ any single frame`, i.e. trap if
`SP − RED_ZONE < limit`. If that passes, `SP ≥ limit + RED_ZONE ≥ limit + frame`, so allocating the
frame keeps `SP ≥ limit`; the callee re-checks before *its* frame. Soundness requires **every guest
function's frame ≤ RED_ZONE**, enforced by a **post-compile frame-size bound**: after compiling a
function, if Cranelift reports `frame_size > RED_ZONE`, fail closed (`JitError`) — never emit an
under-guarded function.

### 3. Per-thread limit source — ride alongside `trap_out` (no vmctx, no TLS codegen)

There is no vmctx param, but every compiled function already receives `trap_out: *mut i64` as its 3rd
Tail-ABI param, and it is **threaded into the fiber entry on resume** (`fiber_rt.rs:658`
`call_tramp(code, mem_base, fn_table_base, trap_out, sp, arg)`) — so it is shared root↔fiber within a
vCPU and **separate per spawned vCPU/OS thread**. Widen that host cell from one `i64` to two —
`[trap_code, stack_limit]` — and:

- the prologue loads the limit from `[trap_out + 8]` (one load, like the epoch poll);
- the runtime writes `stack_limit = fiber.usable_low` **across the resume seam** with the same
  save/restore discipline `svm_set_current_fiber` already uses (`fiber_rt.rs:853/858`): set the
  resumed fiber's low bound before `call_tramp`, restore the previous value after (handles nested
  resumes automatically);
- the **root** computation (on the OS thread stack, not an arena slot) keeps `stack_limit = 0`, and
  `SP − RED_ZONE < 0` is never true → the check is inert there (same convention as `epoch_addr == 0`).

This keeps the check per-thread-correct with no ABI param and no TLS access-model codegen.

## Security contract (what makes it escape-grade)

- The JIT emits the check in **every** guest function prologue unconditionally; the guest controls
  neither codegen nor `trap_out`, so it cannot skip or move the check.
- `RED_ZONE` + the post-compile frame bound guarantee no single frame can jump the limit.
- The masked/confined memory model is unchanged; this only adds a native-stack bound.
- Must be fuzzed (a differential: any recursion depth that traps `StackOverflow` under the check must
  not, under any input, have written below `usable_low`) and audited alongside the masking unit.

## Increment plan

1. **[done]** `TrapKind::StackOverflow` + `from_code`; `arena-stacks` allocator; cost-only
   `emit_stack_check` prototype; feature flags; measurements.
2. **[done — this increment]** Functional escape-grade check: `emit_stack_check` now does the
   frame-aware `SP − RED_ZONE < limit` compare and traps `StackOverflow`; the fiber runtime writes each
   resumed fiber's `usable_low` across the resume seam (`fiber_rt.rs`, bracketed with
   `svm_set_current_fiber`) and the root stays `limit = 0` (inert). Test `tests/stack_check.rs`: an
   unbounded-recursion fiber traps `StackOverflow`; a shallow fiber still runs. **Two deviations from
   the plan, deferred to 2b:**
   - *Limit source* is a **process-global** `STACK_LIMIT` cell, not the trap-cell-adjacent word. It is
     set/restored across the resume seam, so it is **correct for one vCPU's cooperative fibers**
     (including nested resumes) but **not** for concurrent multi-vCPU (`thread.spawn`) runs. The
     feature is off by default, so this affects nothing shipped.
   - *Frame-size bound* is **not** enforced: Cranelift 0.132 doesn't expose the final frame size on
     `CompiledCode`, so a function with an unusually large spill frame (> `RED_ZONE`) is an assumption,
     not a guarantee.
2b. **[next]** Make it multi-vCPU-correct + fully sound: move the limit to the trap-cell-adjacent word
   (widen the trap cell to `[trap_code, stack_limit]` at the run-harness sites; prologue loads
   `[trap_out+8]`), and add the frame-size bound (prologue-parse the `sub rsp, N`, or an upstream
   Cranelift API, or the native frame-aware mechanism + a SIGILL handler). Differential vs the interp.
3. **Windows** arena + TEB-field `Stack` surface; **GC** interaction (a reused, un-zeroed slot makes
   the conservative `gc.roots` scan over-approximate — sound but imprecise; decide zero-on-reclaim of
   only the touched high-water region vs accept the over-approximation).
4. Fuzz target for the guard; audit; then consider flipping arena+check on by default per platform.
