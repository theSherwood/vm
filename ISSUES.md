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

### I3 — `durable_jit` cross-backend fuzz flakes on Windows CI under cumulative JIT-commit pressure (S4)

**Where:** `crates/svm/tests/durable_jit.rs::freeze_thaw_cross_backend_over_generated_modules`
(the no-nightly cross-backend freeze/thaw driver), via `support/durjit.rs::fuzz_one_xbackend` →
`svm-jit` compile + guest-window commit. Windows runners only.

**Symptom:** intermittently the test binary aborts mid-run with
`memory allocation of 131072 bytes failed` followed by exit code `0xc0000409`
(`STATUS_STACK_BUFFER_OVERRUN`). Observed on PR #70 (a `svm-peval`-only change that cannot touch
this path); the exact base commit was green on the same job, and Linux/macOS always pass — i.e. a
flake, not a regression.

**Root cause.** Each of the 64 seeds JIT-compiles ~3× and commits a fresh guest window, so the
process's *cumulative* committed VA climbs across the run. On a memory-tight Windows runner the
commit limit (`os error 1455`) is reached, and the **next ordinary heap allocation** — here a
128 KiB (`131072`) `Vec`/`Box` — gets a null back. Rust's global-allocator OOM path
(`handle_alloc_error`) then **aborts** the process, which Windows reports as
`STATUS_STACK_BUFFER_OVERRUN`. This is the same Windows eager-commit memory-pressure *family* as
**I1** and shares its abort signature, but a **distinct** site: I1 was the fiber control-stack
`VirtualAlloc` (now fallible → `Trap::FiberFault`); this is a generic heap allocation that cannot be
made to trap gracefully — once commit is exhausted, *some* allocation aborts. The test already
*bounds* the pressure (seed count capped at 64; the heavier recycled variant is
`#[cfg(not(windows))]`-gated) — that mitigation is just still marginal on the tightest runners.

**Fix sketch (deferred — re-run clears it):**
1. Reduce the Windows blast radius further: lower the seed count behind `#[cfg(windows)]` (e.g. 32),
   or drop the JIT window reservation size for this driver so each commit costs less VA.
2. Reclaim VA between seeds — free/unmap each compiled blob + guest window before the next seed
   instead of letting them accumulate for the whole test (the libFuzzer target does the heavy run
   anyway, so the in-tree smoke needn't hold every artifact live).
3. Or split the driver so each seed (or small batch) runs in its own process, capping peak commit.

Until then, treat a `STATUS_STACK_BUFFER_OVERRUN` / `os error 1455` abort in this specific test on
Windows as a flake: re-run the failed job (`rerun_failed_jobs`).

---

### I4 — Windows JIT trap-time backtrace covers memory faults but not explicit-check traps (S3) — on `claude/debug-jit-backtrace`

**Where:** `crates/svm-jit/src/lib.rs` (`trap_capture_addr()` returns `0` on non-unix, so `emit_trap`
bakes no explicit-trap capture call), `crates/svm-jit/src/trap_shim.c` (the unix-only
`svm_capture_explicit_trap`).

**Update (memory faults: fixed on Windows).** The Windows Vectored Exception Handler now captures the
trap-time backtrace for **memory faults**, mirroring the unix SIGSEGV/SIGBUS path: `mem.rs`'s windows
`pal::veh` reads the faulting `Rip`/`Rbp` from `EXCEPTION_POINTERS->ContextRecord` and walks the
frame-pointer chain (a Rust `walk_fp_chain`) into a thread-local before restoring the recovery context;
the windows `pal::take_trap_frame` reads it. So `last_trap_backtrace()` + the kill message now carry
source frames for a Windows guard fault (covered cross-platform by
`memfault_kill_message_carries_a_source_backtrace` in `svm-run`'s `run.rs`).

**Still open: explicit-check traps on Windows.** Div/rem-by-zero, `unreachable`, `OutOfFuel`, and
indirect-call-type traps store a `TrapKind` and return — there is no signal/exception to capture from, so
on unix the lowering bakes a `call svm_capture_explicit_trap` at the trap site (`trap_capture_addr()`).
On Windows that address is `0`, so these still produce an **empty** backtrace (correct `TrapKind` + kill,
no frames). Not a correctness or escape hazard. (The `trap_kill_message_carries_a_source_backtrace` test —
div-by-zero — keeps its source-line assertion under `#[cfg(unix)]` for this reason.)

**Why it isn't a quick patch (two concrete blockers, found on attempt):**
1. **Recovering the innermost frame without `__builtin_frame_address`.** The unix helper uses
   `__builtin_frame_address(0)` to find its own frame → the trapping fn's `rbp` *and* the trap-site
   return address (`[my_fp+8]`). **MSVC has no `__builtin_frame_address`.** Cranelift's
   `get_frame_pointer` (confirmed present in cranelift-codegen 0.132 x64) can hand the helper the guest
   fn's `rbp` as an argument — but walking from *that* yields only the **caller** chain; the trapping
   function's own line is lost. Recovering it needs the helper's return address (`_ReturnAddress()` on
   MSVC / `__builtin_return_address(0)` on GCC), which pulls the helper back into C.
2. **Cross-language capture state.** Windows memory faults capture into **Rust** thread-locals (the VEH
   is Rust, `mem.rs` windows `pal`); the unix explicit helper writes **C** thread-locals (`trap_shim.c`).
   A C explicit-trap helper on Windows would write a location the Windows `take_trap_frame` (which reads
   the Rust thread-locals) never sees. Unifying them is a capture-state refactor, not a patch.

**Refined fix (a proper slice, not a quick win):** unify the capture state in Rust (one thread-local set
read by `take_trap_frame` on both platforms; the unix C signal handler stores via a small async-signal-
safe `extern "C"` Rust shim), and make the explicit-trap helper take the frame pointer from
`get_frame_pointer` + the trap site from `_ReturnAddress`/`__builtin_return_address`. Then `emit_trap`
bakes `call <helper>(get_frame_pointer())` on **all** targets (de-special-casing unix too). **Test:**
un-gate `trap_kill_message_carries_a_source_backtrace` on Windows; validate on the `windows-latest` CI
job.

---

### I5 — JIT/interp trap backtraces are not labeled with the trapping fiber (S4) — on `claude/debug-jit-backtrace`

**Where:** the trap-time backtrace capture sites — `crates/svm-jit/src/trap_shim.c` (the SIGSEGV/BUS
handler + `svm_capture_explicit_trap`), `crates/svm-jit/src/mem.rs` (the windows VEH), and the §14
coroutine/fiber runtime (`fiber_rt.rs`).

**Is:** a trap-time backtrace (`last_trap_backtrace` / `run_traced`) gives the correct guest **frames**
regardless of which fiber/coroutine was running when the trap fired — the frame-pointer walk works on
whatever stack the trap is on, and Stage 3 already collects a spawned vCPU's capture into the `Domain`.
What's missing is a **fiber-id label** (DEBUGGING.md §5 W3 Stage 3 "names the right fiber under
work-stealing migration"): the backtrace doesn't say *which* §23/D57 migratable fiber the frames belong
to. Pure cosmetics — the frames themselves are right.

**Why it isn't a quick patch:** the capture runs in the low-level handlers (C signal handler, Rust VEH,
the explicit-trap helper), none of which have the running fiber's identity to hand. `fiber_rt::current()`
returns the thread-local `*mut FiberRuntime` but not a stable handle, and a fiber migrates across worker
threads, so the id must be read at capture time, not reconstructed after. Threading a "current fiber
handle" thread-local that the capture sites can cheaply read is the work.

**Fix sketch:** maintain a per-thread "current fiber handle" cell (set on each `cont.resume`/suspend
switch in `fiber_rt`), read it at capture time into the trap-frame thread-local alongside `pc`/`rets`,
and surface it (e.g. `JitFrameLoc`-adjacent or a `last_trap_fiber()` accessor) for the kill message.

---

_(I1 below is open-adjacent — its abort mechanism is fixed, but I3 is the residual same-family flake.
I2 resolved below.)_

---

## Resolved

### I2 — LLVM on-ramp now ingests auto-vectorized output wider than 128 bits (vector legalization landed) (S3) — fixed on `claude/dreamy-newton-ni7epv`

**Where:** `crates/svm-llvm/src/lib.rs` — vector type recognition (`vec_lane_shape`/`vec128_shape`/
`wide_vec_layout`, `val_type`/`type_size`/`type_align`), the `lower_wide` legalization pass + its
`BlockCtx` helpers, and the block-boundary fan-out in `translate_block`/`branch_args`.

**Was:** translating a `clang -O2`-vectorized program fail-closed with
`Error::Unsupported("type <16 x i32> (Milestone 1+)")` (or `<16 x i64>`, `<4 x i64>`, `<8 x i8>`,
`<2 x i64>`, `<16 x i8>`, etc.). The on-ramp mapped only `<4 x {i32,float}>` (and the 2-lane → packed
`i64` case) to a `v128` and rejected every other shape, because svm-ir's SIMD type is a fixed-128-bit
`v128` (§17/D58) while LLVM's `-O2`/SLP vectorizer emits arbitrary-width "virtual" vectors on the
assumption the backend's `LegalizeTypes` pass will split them. The on-ramp had no such pass.

**Fix (landed, the §17 fixed-128 SelectionDAG-`LegalizeTypes` analog — the chunk width is fixed at
128 bits, never host-detected, to preserve the interp↔JIT/durable-fiber determinism contract):**

1. **128-bit shapes generalized** (fix-sketch step 2): a single `vec_lane_shape`/`vec128_shape`
   recognizer maps any 16-byte LLVM vector to its `VShape`, threaded through every 128-bit lowering
   site, replacing the `i32x4`/`f32x4`-only helpers. svm-ir/verify/jit/interp already supported all
   six `VShape`s, so this was frontend-only. Now `i8x16`/`i16x8`/`i64x2`/`f64x2` all work.
2. **Wide / sub-128 legalization** (fix-sketch step 1): `wide_vec_layout` splits a `<N×T>` into
   `full_chunks` 16-byte `v128`s + `tail_lanes` scalar lanes; `lower_wide` (dispatched at the top of
   `translate_inst`) rewrites each wide op per-chunk + per-tail — load/store, int/float lane arith,
   bitwise, lane min/max, horizontal `vector.reduce.*`, extract/insert, constants, and the broadcast
   (splat) `shufflevector`. A wide value is held as `wide_vals[vid] = [chunks…, tail…]`, mirroring the
   `agg` multi-value pattern.
3. **Cross-block fan-out**: a wide value that crosses a block edge (a vectorized loop's accumulator
   carried across the backedge as a wide phi) expands into `K = chunks + tail` consecutive block
   params, supplied as `K` branch args on every edge (`translate_block`/`branch_args`).

**Follow-ons (now landed, slices AP–AT — the breadth lanes re-enabled vectorization):** vector integer
+ float **conversions** (lane-wise scalarize), **rotate** (`llvm.fshl`/`fshr`), **general cross-chunk +
cross-representation shuffles**, and `<N x i1>` **masks** (vector `icmp`/`fcmp`/`select`/`extractelement`/
`bitcast`-movemask, held lane-wise). The C/C++/Rust breadth lanes now compile **without**
`-fno-*-vectorize` and translate their real `-O2` SIMD output. Still fail-closed (no corpus need yet):
a *general* (non-rotate) funnel shift, a *non-constant* shuffle mask, `llvm.masked.*` (gather/scatter/
masked load-store), wide-vector **function params/returns**, and a mask crossing a block edge.

**Verification:** `cargo test -p svm-llvm --test translate` (115 pass). New tests cover every 128-bit
shape, the wide splitter (`<8 x i32>`/`<4 x i64>` chunks, `<8 x i8>` all-tail), a real loop-carried
wide phi (verified `phi <8 x i32>` in the IR), and two **capstones ingesting genuine `-O2 -mavx2`
auto-vectorized bitcode** (a `<8 x i32>` reduction and an elementwise kernel) byte-identical to the
native scalar oracle on both the interpreter and the JIT.

---

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
