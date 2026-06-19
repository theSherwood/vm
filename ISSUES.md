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

### I4 — Rare macOS-CI `SIGABRT` in the `svm-wasm` threaded-import test (S4, surface reduced) — `claude/vcpu-context-recycling`

**Where:** `crates/svm-wasm/tests/imports.rs::spawn_alongside_capability_import` — a `wasi:thread-spawn`
module that spawns 6 OS-thread workers, each doing a `Blocking` `cap.call` + `i64.atomic.rmw.add`, with
the root parking on `memory.atomic.wait32` until they finish. Runs on the JIT via
`svm_jit::compile_and_run_with_host`.

**Symptom (observed once):** on PR #72's first slice-3.3 CI run, the `build · test (macos-latest)` job's
`imports` binary aborted with `signal: 6, SIGABRT`. Tests run in parallel, so the abort surfaced after
a *sibling* test (`import_handle_threads_through_call_indirect`) had already printed `ok`; the only test
in that binary still running — and the only one using real OS threads + futex wait/notify — is
`spawn_alongside_capability_import`. **Not reproduced:** it passed on the very next run (same commit
range) and passes repeatedly on Linux (5/5 stress), and macOS cannot be run in this environment, so the
root cause is not pinned.

**Suspected cause / mitigation (landed).** Slice 3.3 (multi-vCPU durable) began creating the
`SharedFiberTable` for `uses_fibers || uses_threads` (the durable vCPU-context allocator lives on it).
A `.map` over that table *incidentally* also built the **root vCPU's `FiberRuntime` and published it as
`CURRENT_RT`** for a thread-only module — behavior it never had pre-3.3. A fiber-free module never
resumes a fiber, so that runtime is dead weight, but it changed the threaded run's setup/teardown
surface on the spawning thread. The table-vs-runtime split is now fixed: the **table** stays present for
`uses_threads` (needed by the allocator), but the **runtime** is built only for `uses_fibers` (so a
thread-only run again publishes no `CURRENT_RT` and allocates no idle runtime). This *reduces the change
surface back to the pre-3.3 path* but is **not a confirmed cure** — the abort was never reproduced, so it
may be a pre-existing macOS-runner flake (e.g. futex/thread teardown timing, or runner memory pressure)
unrelated to the runtime. Severity is provisional `S4` pending a root cause; if a reproduction shows a
real abort path it should be re-classified.

**Next step if it recurs:** capture the macOS core/backtrace (the `imports` binary under
`RUST_BACKTRACE=full`, ideally `--test-threads=1` to localize which test aborts), and check whether it
is in futex park/teardown (`os_thread_rt::{thread_wait,thread_notify,join_all}`) or the guard/signal
path — distinct from the now-removed root-runtime delta and from the resolved I1 (fiber-stack alloc).

---

### I6 — JIT/interp trap backtraces are not labeled with the trapping fiber (S4) — on `claude/debug-jit-backtrace`

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

---

_(I1 below is open-adjacent — its abort mechanism is fixed, but I3/I4 are residual same-family CI-abort
flakes. I2 resolved below.)_

---

## Resolved

### I5 — Windows JIT trap-time backtrace now covers explicit-check traps too (S3) — fixed on `claude/dap-function-names`

Covered memory faults but not explicit-check traps (div/rem-by-zero, `unreachable`, `OutOfFuel`, indirect-call-type): the unix explicit-trap helper used `__builtin_frame_address` (MSVC lacks it) and the capture state was split (unix C thread-locals vs the windows VEH's Rust ones). **Fixed:** a cross-platform `crates/svm-jit/src/trap_capture.c` (compiled on unix + windows) holds the capture state + frame-pointer walk + the explicit-trap helper; `emit_trap` bakes `call svm_capture_explicit_trap(get_frame_pointer())` on every target — the trapping frame pointer comes from Cranelift `get_frame_pointer` (no `__builtin_frame_address`) and the trap-site return address from `_ReturnAddress` (MSVC) / `__builtin_return_address` (GCC). The unix signal handler feeds `svm_store_trap_frame`; the windows VEH keeps its Rust memory-fault capture and `take_trap_frame` falls back to the C capture for explicit traps. **Confirmed green on the `windows-latest` (MSVC) CI job** — `trap_kill_message_carries_a_source_backtrace` (div) is un-gated there. (build.rs `try_compile`s the shim, tolerating a missing mingw cross-compiler on the non-linking windows-gnu cross-check.)

---

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
