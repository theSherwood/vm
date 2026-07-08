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
  neither codegen nor the `stack_limit` param, so it cannot skip or move the check.
- The check runs after the machine prologue's `sub rsp`, so each frame is validated directly; `RED_ZONE`
  covers the pre-check register pushes. A single frame larger than the whole stack is the one residual
  gap (deferred backstop — see below), astronomically unlikely for a real function.
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
2b. **[investigated — plan revised, see below]** Make it multi-vCPU-correct + fully sound.

### 2b findings (reading the code changed the plan)

- **The trap cell is shared across vCPUs, not per-vCPU** (`lib.rs:2862` "shared across vCPU threads —
  every spawned vCPU gets its address via `set_env`"). So the design's §3 "ride alongside `trap_out`"
  approach **does not give per-vCPU correctness** — a widened trap cell would be shared too, clobbered
  between concurrent vCPUs. Retracted.
- **`cranelift-jit` does not support TLS** (`cranelift-jit-0.132/src/backend.rs:448`
  `assert!(!tls, "JIT doesn't yet support TLS")`). So a Cranelift TLS-data global (the usual per-thread
  source) is out. A helper-*call* per prologue would work but costs ~10–20× the ~1-cycle check.
- **The check runs *after* the machine prologue's `sub rsp, frame`** (it's in the entry block, which
  Cranelift emits after the prologue), so `get_stack_pointer` already reflects this function's frame —
  the check **validates each frame directly**. Therefore `RED_ZONE` only needs to cover the prologue's
  pre-check register *pushes* (~tens of bytes), and the post-compile frame-size bound is **not** a
  per-frame necessity — only a backstop against an absurd (> stack-size) single frame. Good news: the
  current check is more sound than §2 claimed, and the frame-size-API blocker is far less important.

### 2b chosen + DONE: path B (per-vCPU ABI param)

Two routes were on the table — **A** (SP-mask + running every vCPU top on a uniform aligned managed
stack) and **B** (thread a per-vCPU limit through the ABI). Analysing A's implications showed it caps
every vCPU's native stack at `SLOT` (256 KiB vs the 8 MB OS stack), and drags in the durable §12.8
root/context rework, the GC root-scan, and the spawn/root run paths — all for no security gain over B.
So **B was chosen and is implemented**, with two refinements that made it cheaper and cleaner than the
original sketch:

- The limit is a **value** i64 param (not a pointer): `stack_limit`, added to `sig_from` right after
  `trap_out`, threaded through every call by `ctx_args` (constant within a stack's call tree), and set
  anew at each context entry. The prologue check is then `SP − RED_ZONE < limit` against a **register**
  — no cell, no TLS, no load. Per-vCPU-correct by construction.
- The fiber's limit is sourced from **`svm_fiber::Yielder::stack_low()`** (svm-fiber now stores the
  control stack's `usable_low` in its `Control` and exposes it) — correct for both stack backends with
  no alignment coupling and no root-on-managed-stack change. The **root** and each **spawned vCPU top**
  pass `limit = 0` (they run on OS-guarded thread stacks ⇒ check inert); only fibers get a real limit.

Touch-points (all cfg-gated on `stack-check`, so the default ABI is byte-identical): `sig_from`, the
entry block params + `pbase` + `sret` index, `Lower.limit_var`, `ctx_args`, `emit_stack_check`,
`build_trampoline` (root ⇒ 0), `build_fiber_call_trampoline` + `FiberCallTramp` (+`stack_limit`), the
`make_fiber` entry (passes `y.stack_low()`), and the `os_thread_rt` thread-entry (⇒ 0). The removed
increment-2 global cell + resume-seam set/restore are gone.

Tests (`tests/stack_check.rs`, feature-gated): single-vCPU unbounded recursion traps `StackOverflow`;
a **spawned vCPU**'s fiber recursion traps `StackOverflow` on its own thread (proves per-vCPU
correctness — a shared cell would clobber); a shallow fiber still runs. Default build/clippy/fmt clean
under `-D warnings`; no fiber/thread-suite regression.

**Still deferred** (not per-vCPU-related): the post-compile frame-size **backstop** against an absurd
(> stack-size) single frame — Cranelift 0.132 doesn't expose the final frame size; the after-`sub rsp`
check position means normal frames are validated directly, so this is a low-priority backstop only.
Note: the SLOT-aligned arena slots (from the prior 2b commit, for path A's SP-mask) are now vestigial
under B — harmless, removable later.
3. **[Windows arena done]** `stack_arena.rs` is cross-platform with **commit-on-demand**: the arena is
   only *reserved* (`mmap`+`MAP_NORESERVE` on unix; `VirtualAlloc(MEM_RESERVE)` on x86-64 Windows), and
   a slot is committed on first hand-out (a no-op on unix; `MEM_COMMIT` on Windows) and stays committed
   when freed, so recycling is a free-list pop. The *commit* footprint therefore tracks peak concurrent
   fibers, not the reserved arena size — so a large arena is cheap and Windows no longer pays a
   whole-arena pagefile charge. Provides the Windows `Stack` surface
   (`base_ptr`/`limit_ptr`/`top` for the TEB seed). Verified with `cargo check --target
   x86_64-pc-windows-gnu --features arena-stacks` under `-D warnings`; **runtime-untested** (no Windows
   host / CI — needs the CI job below). **`#3` exit-code parity done:** the interp's `trap_status` now
   maps `Trap::StackOverflow` to the JIT's `TrapKind::StackOverflow` (13).
   Still open: **GC** interaction (a reused, un-zeroed slot makes the conservative `gc.roots` scan
   over-approximate — sound but imprecise; decide zero-on-reclaim of the touched high-water region vs
   accepting it).
4. **CI (`#1`, deferred):** a job that builds/tests `--features stack-check,arena-stacks` on Linux
   (and macOS-aarch64), so the guard tests + arena run somewhere and the Windows arena gets runtime
   coverage. Then a **fuzz** target for the guard; **audit**; then consider flipping arena+check on by
   default per platform.
5. **Frame-size backstop** (low priority): reject an absurd (> stack-size) single frame — needs the
   Cranelift frame size (not exposed in 0.132) or a prologue parse.
