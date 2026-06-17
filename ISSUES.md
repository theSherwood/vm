# Known Issues & Robustness Gaps

A registry of **known bugs, robustness gaps, and latent hazards** that are understood but not yet
fixed — distinct from the forward-looking design/status docs (`DESIGN.md`, `DURABILITY.md`,
`HANDOFF.md`). An entry here is a deliberately-deferred problem with a recorded root cause and a fix
sketch, so it isn't rediscovered from scratch. When an issue is fixed, move it to the bottom
("Resolved") with the commit/PR, or delete it and note the fix in the relevant design doc.

Severity: **S1** corruption/escape · **S2** guest-triggerable host crash or wrong result · **S3**
robustness/quality · **S4** cosmetic/flake.

---

## Open

_(none)_

---

## Resolved

### I1 — A fiber-stack OS allocation failure aborts the process instead of trapping (S2) — fixed on `claude/fiber-stack-lazy-commit`

**Where:** `crates/svm-fiber/src/stack_windows.rs` / `stack_unix.rs` (`Stack::new`), reached via
`Fiber::new` ← `svm_jit::fiber_rt::{make_fiber, fiber_new, seed_frozen_fibers}` and
`svm_jit::instantiator_rt` (the coroutine child). The interpreter has no analogue: its fibers are
host-side `Pending` entries with no native control stack, so only the JIT allocates here.

**Symptom (was):** under real memory pressure, allocating a fiber's control stack failed, an
`assert!` **panicked**, and because `fiber_new` is an `unsafe extern "C"` thunk (called from JITted
guest code, which cannot unwind) the panic became a **non-unwinding abort** — the whole process died
(`STATUS_STACK_BUFFER_OVERRUN` / `SIGABRT`). First observed as a flaky **Windows CI** failure in the
unrelated `jit_threads` concurrent-fiber stress test (PRs #36, #41): a lingering spawned-vCPU
thread's `cont.new` aborted the test binary.

**Root cause / why it bit Windows first.** The design intends a fiber that can't be created to be a
clean, recoverable `Trap::FiberFault` — the **quota pre-check** (`SharedFiberTable::has_room`)
already delivers that for a fiber *bomb*. But a *genuine OS-allocation failure below the quota* had no
such path: `Stack::new` just `assert!`ed. Compounded by Windows committing eagerly:
`stack_windows.rs` reserved **and committed** the full per-fiber stack (`FIBER_STACK = 1 MiB`,
`MEM_RESERVE | MEM_COMMIT`), so N live fibers cost N MiB of *committed* VA, while the unix `mmap` path
commits lazily on touch. The quota (`MAX_FIBERS = 1 << 16`) × 1 MiB ⇒ a 64 GiB committed ceiling that
does not bound real Windows memory, so `VirtualAlloc` failed long before the quota tripped.

**Fix (landed):**
1. **`Stack::new` and `Fiber::new` are now fallible** (`-> Option<…>`, returning `None` on
   `MAP_FAILED` / null `VirtualAlloc` / guard-`mprotect`/`VirtualProtect` failure, with the partial
   reservation cleaned up). The JIT callers turn `None` into the intended recoverable trap:
   `fiber_new` writes the trap cell + returns `-1` (the existing `FiberFault` path); `make_fiber` and
   `seed_frozen_fibers` propagate it (a thaw re-seed failure skips the root re-entry rather than
   re-entering with missing fibers); the instantiator coroutine returns `CapFault`. No path can abort
   the host on a fiber-stack allocation failure anymore.
2. **Per-fiber control stack reduced 1 MiB → 256 KiB** (`FIBER_STACK` / `CORO_STACK = 1 << 18`),
   cutting committed Windows memory 4× per live fiber and pushing the practical fiber ceiling out
   correspondingly. Still ample for deep guest call chains.

**Why not true kernel-growth lazy commit on Windows (the original fix-sketch point 2):** rejected.
The `svm-jit` `gc.roots` walker scans a *running* fiber's whole usable stack via
`Fiber::full_extent()` → `[usable_low, top)` (a sound conservative superset of its live frames).
Under demand-commit that scan would touch uncommitted pages and fault. Making it safe would need a
committed/high-water bound threaded through the GC scan, and Windows can't be run-tested in this
environment — so the size reduction + fallible alloc (both fully testable, and the latter is the
actual abort cure) were chosen over an untestable, GC-entangled commit-on-fault scheme.

**Verification:** `svm-fiber` + `svm-jit` unit tests, `jit_threads`, and the durable-fiber
freeze/thaw suites pass on unix; `cargo check --target x86_64-pc-windows-gnu -p svm-fiber` compiles
the rewritten Windows path. The recurring Windows `jit_threads` flake's abort mechanism is removed.
