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

### I1 — A fiber-stack OS allocation failure aborts the process instead of trapping (S2)

**Where:** `crates/svm-fiber/src/stack_windows.rs:42` (`assert!(!base.is_null(), "fiber stack
VirtualAlloc failed")`) and its unix twin `crates/svm-fiber/src/stack_unix.rs:35`
(`assert!(base != MAP_FAILED, "fiber stack mmap failed")`), reached via `Fiber::new` ←
`svm_jit::fiber_rt::{make_fiber, fiber_new}` (and the interpreter's analogue).

**Symptom:** under real memory pressure, allocating a fiber's control stack fails, the `assert!`
**panics**, and because `fiber_new` is an `unsafe extern "C"` thunk (called from JITted guest code,
which cannot unwind) the panic becomes a **non-unwinding abort** — the whole process dies
(`STATUS_STACK_BUFFER_OVERRUN` / `SIGABRT`). First observed as a flaky **Windows CI** failure in the
unrelated `jit_threads` concurrent-fiber stress test (PR #36): all tests passed, then a lingering
spawned-vCPU thread's `cont.new` aborted the test binary.

**Root cause / why it bites Windows first.** The design intends a fiber that can't be created to be a
clean, recoverable `Trap::FiberFault` — and the **quota pre-check** (`SharedFiberTable::has_room`)
already delivers that for a fiber *bomb* (too many fibers). But a *genuine OS-allocation failure
below the quota* has no such path: `Stack::new` just `assert!`s. Two compounding factors:
- **Windows commits eagerly.** `stack_windows.rs` reserves **and commits** the full per-fiber stack
  (`FIBER_STACK = 1 MiB`, `MEM_RESERVE | MEM_COMMIT`), so N live fibers cost N MiB of *committed*
  VA. The unix path `mmap`s lazily (reserve; pages commit on touch), so the same fiber count is far
  cheaper. The default fiber quota (`MAX_FIBERS = 1 << 16`) × 1 MiB ⇒ a 64 GiB *committed* ceiling on
  Windows — well past a CI runner's commit limit — so the quota does **not** bound real Windows
  memory, and `VirtualAlloc` fails long before the quota trips.
- **`extern "C"` + panic = abort.** Even where a fault *should* be recoverable, the panic can't
  unwind out of the thunk, so it aborts rather than writing the trap cell.

**Impact.** (a) Robustness/CI flakiness on memory-tight Windows runners. (b) A latent
**guest-triggerable host crash**: a guest creating many fibers (within the configured quota) can
exhaust host commit on Windows and abort the host process, rather than getting a `FiberFault`. Not a
confinement escape (no memory is corrupted), but an availability gap.

**Fix sketch (deferred — touches the `svm-fiber` unsafe substrate, out of scope for the durable
fiber-parity PR where it was found):**
1. Make `Stack::new` **fallible** (`-> Option<Stack>` / `Result`) instead of `assert!`-on-failure;
   thread the failure up so `Fiber::new` / `fiber_new` return the existing `FiberFault` path
   (`fiber_new` already returns `-1` + writes the trap cell on the quota pre-check — reuse it). Same
   on the interpreter side. This converts the abort into the intended recoverable trap on **both**
   backends.
2. Consider lazy commit on Windows (reserve with `MEM_RESERVE`, commit on demand via a guard-page /
   `VirtualAlloc(MEM_COMMIT)` fault handler, or a smaller committed prefix) so the per-fiber cost and
   the effective fiber ceiling match the unix `mmap` model — otherwise the quota's memory bound stays
   wildly off on Windows.

**Workaround today:** none in-process; re-run flaky Windows CI. The fiber quota can be lowered by the
embedder to bound concurrent fibers, but it counts fibers, not committed bytes.
