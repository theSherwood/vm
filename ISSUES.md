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

### I5 — Windows JIT trap-time backtrace covers memory faults but not explicit-check traps (S3) — **FIX LANDED** on `claude/dap-function-names` (pending `windows-latest` CI confirmation)

**Fix (landed, the refined-fix design below):** the trap-time capture state + frame-pointer walk +
explicit-trap helper moved into a new cross-platform `crates/svm-jit/src/trap_capture.c` (compiled on
unix **and** windows). `emit_trap` now bakes `call svm_capture_explicit_trap(get_frame_pointer())` on
every target — the trapping frame pointer is threaded in via Cranelift `get_frame_pointer` (so MSVC's
missing `__builtin_frame_address` is sidestepped), and the trap-site return address comes from
`_ReturnAddress()` (MSVC) / `__builtin_return_address(0)` (GCC). The unix signal handler and the windows
VEH both feed the shared capture (the handler via `svm_store_trap_frame`; the VEH keeps its Rust
memory-fault capture and the windows `take_trap_frame` falls back to the C `svm_take_trap_frame` for
explicit traps). The `trap_kill_message_carries_a_source_backtrace` test (div-by-zero) is now un-gated
on Windows. Unix validated locally; windows-gnu compiles; **MSVC runtime is validated by the
`windows-latest` CI job** — move this entry to Resolved once that job is green. _Original report below._

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

_(I1 below is open-adjacent — its abort mechanism is fixed, but I3/I4 are residual same-family CI-abort
flakes. I2 resolved below.)_
### I7 — Rare deadlock/hang in the work-stealing fiber demos (CI flake) (S3)

**Where:** the guest-built work-stealing schedulers run end-to-end through the `svm-run` binary —
`crates/svm-run/demos/work_stealing/work_stealing.c` (stackless tasks) and
`crates/svm-run/demos/steal_fibers/steal_fibers.c` (D57 stackful, migratable fibers stolen across
real OS threads) — and their product-path smoke tests `demo_work_stealing_runs` /
`demo_steal_fibers_runs` in `crates/svm-run/tests/run.rs`. The deadlock is in the
scheduler/fiber-stealing path (guest scheduler logic and/or the host `os_thread_rt` + fiber-steal
runtime), not in the demos' I/O.

**Symptom:** the demo process occasionally **never terminates** — the guest's worker threads wedge
with no forward progress, so the test's `Command::…output()` blocks indefinitely. Observed once on
the **Linux x86_64** CI `check` job (run 27778162761, the `cargo test --workspace` step), which hung
>1 h until the run was cancelled. It is **rare**: 0 hangs in 48 local back-to-back runs of both
demos, and the suite passed cleanly on other runs.

**Why only Linux CI sees it:** both tests are gated `#[cfg(all(unix, target_arch = "x86_64"))]`.
`macos-latest` is arm64 and `windows-latest` is non-unix, so **both skip these demos** — the Linux
x86_64 `check` job is the only CI lane that runs them, so a hang there shows up as a single stuck
job while every other job is green.

**Root cause (hypothesis, not yet confirmed):** a timing-dependent liveness bug — most likely a
lost-wakeup / missed-notification race between the steal path and the park/unpark of idle worker
threads (or in the guest scheduler's termination detection), exposed only under a particular
interleaving. Needs root-causing from a stuck instance (attach `gdb`/`lldb` and dump all thread
backtraces, or add steal/park tracing). The fiber/work-stealing **runtime is not modified** by the
argc/argv work (PR #66).

**Sensitivity clue (PR #66):** the race is sharp enough that a *tiny startup perturbation* flips it
from rare to frequent. PR #66 originally had the `svm-run` CLI seed the §3e args buffer (a few-byte
`init_mem` memcpy during window setup, before the guest runs) for **every** program, including these
`main(void)` demos. That harmless, never-read seeding — only a few microseconds of extra setup —
took the hang from "0 in ~50 sequential runs" to **reliable on the first iteration** under
`cargo test --test run --test-threads=8` (parallel load). Reverting to *not* seeding when there are
no actual program args (so a bare run is byte-identical to before) restored the rare baseline (≥6
clean parallel iterations). So whatever the root cause, it is acutely sensitive to worker-thread
start timing — a strong hint for a park/unpark or steal-loop wakeup race.

**Fix sketch:**
1. Root-cause via thread backtraces of a hung process (reproduce by looping the `svm-run` binary on
   the demo until it wedges, then attach a debugger) — confirm whether the stall is in the host
   steal/park runtime or the guest scheduler, and fix the wakeup race.
2. Interim blast-radius mitigation (independent of the root cause): the runner already honours
   `SVM_DEADLINE_MS` (§5 detect-and-kill); have the demo smoke tests run the `svm-run` subprocess
   under a deadline / `timeout` so a hang **fails fast** instead of blocking CI for hours, and add a
   `timeout-minutes:` to the CI `check` job (it currently has none, so a wedged job sits until
   GitHub's 6 h default).

---

### I8 — svm-jit/Cranelift auto-vectorizes only to **128-bit** SIMD, ~2× behind native AVX2/AVX-512 on wide-vectorizable loops (S3) — `claude/svm-jit-alu-simd`

**Where:** the LLVM on-ramp's vector legalization (`crates/svm-llvm/src/lib.rs` `wide_vec_layout`/
`lower_wide`, the §17 fixed-128 `LegalizeTypes` analog) → svm-ir's fixed-128-bit `v128` (§17/D58) →
`svm-jit` lowering each `v128` to one SSE/NEON 128-bit op.

**Symptom.** A reduction (`vadd`: `s += k ^ seed`) compiled `clang -O2 -mavx2` runs ~2× slower on
svm-jit than the native binary, because the on-ramp splits LLVM's wide `<8 x i32>`/`<16 x i32>` vectors
into **128-bit chunks** (4×i32) and svm-jit emits 128-bit `paddd`/etc., while native uses 256-bit `ymm`
(AVX2) or 512-bit `zmm` (AVX-512). So the SVM stack *does* vectorize (contrary to my earlier bench
claim — see below), but at SSE width.

**Measured, across the wasm comparison columns (ns/iter, the same C kernels):**

| kernel | native AVX2 | wasm32 V8 | wasm32 Wasmtime | **svm-jit** | bytecode |
|---|---|---|---|---|---|
| `xorshift` (scalar serial) | 1.73 | 1.99 | 2.04 | **1.59** | 58.3 |
| `vadd` (vectorizable)      | 0.042 | 0.098 | 0.147 | **0.083** | 37.6 |

The key context: **wasm SIMD is itself fixed at 128 bits** (the spec's `v128`), so V8 and Wasmtime are
*also* ~2–3.5× behind native AVX2 on `vadd` — the exact same cap svm-jit has. **svm-jit is not behind
wasm; it is ahead** (0.083 vs V8 0.098 / Wasmtime 0.147), and on the representative *scalar* kernel it
is the fastest engine measured, beating native. So relative to the VM's actual peer (wasm), there is no
SIMD deficit — only native AVX2/AVX-512 leads, by the determinism-bound 128→256/512 margin.

**Root cause — deliberate, not a miss.** The chunk width is fixed at 128 bits and **never
host-detected**, to preserve the interp↔JIT↔durable-fiber **determinism contract** (a frozen vector
register file must replay identically on any host, and the tree-walker oracle is scalar-128). Widening
to the host's native vector width would make results/snapshots host-dependent. So this is a
throughput-vs-determinism tradeoff, not a codegen bug. (Vector *support* itself — all six `VShape`s +
wide/sub-128 legalization — already landed; see Resolved **I2**.)

**Benchmark caveat that exaggerated it.** My `bench/cross-engine` SVM driver compiled the kernels with
`-fno-vectorize -fno-slp-vectorize` (following the stale LLVM.md §4 "MVP" pipeline note), which keeps
SIMD out **entirely** → the SVM rows looked *scalar*, not merely 128-bit. With vectorization enabled
the on-ramp emits `v128` IR and svm-jit lowers it to real SIMD. Two measurement hazards make the win
hard to see in that harness: (a) `vsum`'s known-content array gets **closed-form-folded** by Cranelift
(the opaque-pointer barrier doesn't survive LLVM→SVM), and (b) `svm_jit::compile_and_run` recompiles
per call, so a fast vectorized loop is swamped by compile jitter unless timed via `CompiledModule`
(compile once, run many).

**Fix sketch (deferred — needs a decision):**
1. **Doc/bench:** drop `-fno-*-vectorize` from the on-ramp invocation in the bench (and LLVM.md §4) so
   the SVM rows show the real 128-bit-SIMD number, not scalar; measure with a non-foldable kernel via
   `CompiledModule` (compile-once).
2. **Throughput (optional, contract change):** an *opt-in*, non-default "fast/non-deterministic" mode
   that legalizes to the host vector width (256/512) — only for runs that don't need
   freeze/thaw/oracle determinism. Default stays fixed-128.

---

### I9 — svm-jit lacks LCG/geometric **recurrence strength-reduction**, so a pure `a = a*M + c` loop is ~8× native (S4) — `claude/svm-jit-alu-simd`

**Where:** `svm-jit` (Cranelift) loop codegen, vs `clang`'s x86 backend.

**Symptom.** The `alu` benchmark kernel (`a = a*1103515245 + 12345 + i`) runs ~1.9 ns/iter on svm-jit
vs ~0.24 ns/iter native — an ~8× gap that *looks* like an svm-jit deficiency.

**Root cause — a clang-specific optimization on a pathological kernel, not a general gap.** clang's
backend recognizes the linear-congruential recurrence and **collapses 4 unrolled steps into a single
multiply by `M^4`** (observed: the native loop is one `imul $0xee067f11` — `M^4 mod 2^32` — per 4
iterations, with the per-step constants folded into additive terms). The on-ramp ingests clang's
*mid-end* IR, which is unrolled 4× but **not** collapsed (4 separate `i32.mul`), and Cranelift doesn't
do the collapse either → svm-jit runs 4 muls / 4 iters at multiply latency. **This is the only kernel
where svm-jit trails native**: on serial loops clang *can't* collapse, svm-jit **matches or beats**
native — measured `xorshift` 1.61 vs 1.74 ns, `muldep` 1.28 vs 1.52 ns (svm-jit faster). LCG-shaped
hot loops are rare in real code, so this is low priority.

**Fix sketch (deferred):**
1. **Don't chase it in svm-jit** — recurrence strength-reduction is a niche backend optimization;
   implementing it in Cranelift/the on-ramp is high-effort, low-yield.
2. **Benchmark hygiene:** the `alu` kernel is unrepresentative (it rewards clang's collapse). Report a
   non-collapsible scalar kernel (e.g. `xorshift`) as the headline scalar-throughput number, where
   svm-jit ≈ native, and keep `alu` only as a "clang recurrence-collapse" demonstrator.

---

### I13 — `<2 x i16>` "no-redundant-load" vector pattern **miscompiles** (soundness) — found via Embench `edn`/`fir_no_red_ld`

**Where:** `crates/svm-llvm/src/lib.rs` — the sub-128 / all-tail vector handling for **2-lane 16-bit
vectors** (`<2 x i16>`, and the `<4 x i16>` they're shuffled from): `vec_explode`/`vec_implode`,
`ShuffleVector`/`InsertElement` in the normal path, and a cross-block `<2 x i16>` phi.

**Symptom (S2 — silent miscompile, not fail-closed).** With I11 landed, Embench `edn` now *translates*
but returns a wrong answer: `verify_benchmark` = 1 native vs 0 on **all three** SVM engines (so the bug
is in translation, not an engine). Bisected to **`fir_no_red_ld`** (the FIR variant that carries a
loaded sample across iterations to avoid a redundant load); `fir` and `jpegdct` — which use the *same*
`<4 x i16>` deinterleave shuffles — are correct, so it's specific to this pattern, not the shape alone.
`edn` compiled `-fno-vectorize` is correct, confirming it's the vectorized translation.

**The pattern (from `fir_no_red_ld`'s `-O2` IR).** A `<2 x i16>` carried across the loop backedge:
`insertelement <2 x i16> poison, i16 %s, 1` (lane 0 left undef) → loop phi `<2 x i16>` →
`shufflevector <2 x i16> %prev, <2 x i16> %cur, <i32 1, i32 2>` (recombine: take lane 1 of the carried
value, lane 0 of the new) → `sext <2 x i16> → <2 x i32>` → `mul` → `sext → <2 x i64>` →
`add` → `llvm.vector.reduce.add.v2i64`. `<2 x i16>` is **not** a `vec2` (that's `<2 x i32>`/`<2 x float>`
only) and not a `v128`, so it takes the all-tail `wide_vec_layout` path (0 chunks, 2 tail lanes of an
`i16x8` shape). Something in that path — most likely how a single i16 tail lane survives
`insertelement`/`shufflevector`/the cross-block phi (width/sign of the lane "container"), or the
explode→implode round-trip for the recombine shuffle — drops or corrupts a lane.

**Impact.** This is the on-ramp's worst failure mode — a **silent miscompile** that violates the
fail-closed contract. It is **pre-existing and independent of I11**: `fir_no_red_ld` uses no wide
shifts, so it would miscompile on `main` too once reached; I11 merely lets the *whole* `edn` translate
far enough to hit it. Narrow in practice (this exact carried-`<2 x i16>` shape), but any program with it
miscompiles. The `embench` driver now flags `edn` as **MISCOMPILE (excluded)** rather than counting it.

**Fix sketch (needs care — soundness):** reproduce minimally (the bisection harness: include
`src/edn/libedn.c`, call only `fir_no_red_ld` on a seeded buffer, diff interp vs native — both via
`bench/embench`), then dump the **SVM IR** the translator emits for it and compare against the scalar
form. Likely either (a) fix the i16 tail-lane representation through `insertelement`/`shufflevector`/phi
so a 16-bit lane round-trips losslessly, or (b) if a clean fix is out of reach, **fail-close** the
specific carried-`<2 x i16>` shuffle/phi shape (return `Unsupported`) so it never miscompiles — but
*without* regressing the working `<4 x i16>` shuffles in `fir`/`jpegdct`. A differential fuzz over small
i16 vectors (interp vs JIT vs a scalar oracle) would catch the whole class.

---

## Resolved

### I11 — on-ramp fail-closed on auto-vectorized **wide vector shifts** (`shl`/`lshr`/`ashr` on `<8 x i32>`) (S3) — fixed on `claude/perf-i11-i12`

**Was:** a plain `clang -O2 -mavx2` (or `-O2` with interleave) program whose vectorizer emits a wide
integer shift — e.g. Embench `edn`'s `lshr <8 x i32> v, <i32 15, …>` — fail-closed at translate with
`Unsupported("type <8 x i32> …")`. The I2 legalization split wide loads/stores/arith/reductions/
conversions into `v128` chunks, but `lower_wide` had **no arm for shifts**, so a wide `Shl`/`LShr`/`AShr`
fell through to the normal `bin()` path, which only handles a single `v128` and rejected the 256-bit type.

**Fix (landed):** a `wide_shift` helper (mirroring `wide_int_binop`) splits a wide constant-splat shift
into one `VShift` per `v128` chunk + a scalar shift per tail lane, dispatched from new
`I::Shl`/`I::LShr`/`I::AShr` arms in `lower_wide`. The amount is taken from the constant splat (the shape
the auto-vectorizer emits; a non-uniform amount stays fail-closed, as in the v128 path). Verified by
`simd_autovec_avx2_wide_shifts` in `tests/translate.rs` (interp == JIT == native on a mixed
logical/arithmetic `<8 x i32>` shift) and a 10-op wide-op isolation sweep (shifts/sext/zext/trunc/
reduction/i16 — all bit-exact).

**Note:** this unblocked `edn`'s *shift* op, but `edn` as a whole still fails — it additionally trips
the **I13** `<2 x i16>` miscompile in `fir_no_red_ld`. (Separately, the on-ramp has no `memcmp`/`bcmp`
builtin — `clang` emits those for array compares; the Embench wrapper supplies them in-module with
`-fno-builtin-memcmp/-bcmp`. Providing them as on-ramp builtins, like `memcpy`/`memset`, is a small
coverage win.)

---

### I12 — the §9/D45 `cap.call` fast path left ~9× on the table for cheap caps by re-entering the generic host dispatch (S4) — fixed on `claude/perf-i11-i12`

**Was:** `cap_call` first reported the JIT generic and "fast" (`fast_cap_resolver`) paths as **within
~2%** — but that was a *benchmark artifact*: the probe's `cap.call` passed a stray arg, so it didn't
match the resolver's claimed `(CLOCK, 0, n_args=0, ...)` and silently ran the generic path *both* times.
With a correct **0-arg** `Clock.now()` call the fast path was already **~1.7×** generic (53→31 ns,
the JIT-side marshalling saving) — but the host side still re-entered `Host::cap_dispatch_slots`, which
for a cheap cap is dominated by the per-call `Vec` result allocation + the W1 record/replay gate.

**Fix (landed):** a new `Host::fast_clock_now(handle) -> Option<Result<i64, Trap>>` (svm-interp) does
the authority check (`resolve`, identical to the generic path — a forged/closed/wrong-type handle is an
inert `CapFault`) and the read+advance **inline**, returning the `i64` with no `Vec`. It returns `None`
when a W1 record/replay tape is active, so `svm_run::fast_clock_now` falls back to the full
`cap_dispatch_slots` and the clock crossing is still taped/served faithfully (the clock is a recorded
nondeterministic input). Net: `Clock.now()` on the fast path drops **31 → 5.7 ns** (a further ~5.5×),
so the fast path is now **~9× cheaper than generic** end-to-end.

**Verification.** `cap_call` now shows jit-generic ≈ 54 ns vs jit-fast ≈ 5.7 ns. New
`crates/svm-run/tests/fast_cap.rs` pins interp == generic-JIT == fast-JIT on a 0-arg clock delta and
that a forged handle still faults; the interp↔JIT differential (`svm/tests/jit_diff.rs`, 54),
`jit_quota` (fast-resolver path), and all `svm-run`/`svm-durable` clock tests stay green. (`Blocking.work`
still uses the shared `fast_dispatch` — it's arg-bearing and rarer; same inline treatment is a future
follow-up if it shows up hot.)

---

### I10 — ordinary `clang -O2` auto-vectorized loops hit two narrow holes in the vector breadth (S3) — fixed on `claude/bench-alu-hygiene`

**Where:** `crates/svm-jit/src/lib.rs` (v128 lane-arith lowering) and `crates/svm-llvm/src/lib.rs`
(vector integer-op translation in `bin`).

**Was.** A plain `clang -O2` program (vectorization on — *not* hand-written SIMD) fail-closed when the
loop vectorizer turned a common scalar loop into vector ops the I2 breadth didn't cover:

1. **`i8x16.mul` — svm-jit `Unsupported("instruction")`.** A byte-array fill like
   `for (i) buf[i] = i*31 + 7;` (`unsigned char buf[256]`) vectorizes to a `<16 x i8>` body whose
   multiply becomes `i8x16.mul`. svm-jit lowered `v128.load/store/const`, `i8x16.add/extract_lane`, and
   `i32x4`/`i64x2` multiply — but **not the 8-bit packed multiply** (x86 has no `PMULLB`). Translation
   *succeeded*; only the JIT lowering was missing.
2. **vector integer shifts — on-ramp `Unsupported("vector integer op ShrU (only add/sub/mul/and/or/xor)")`.**
   A bit-twiddling loop like a table-driven CRC (`c = (c & 1) ? P ^ (c >> 1) : (c >> 1)`) vectorizes to
   `lshr <4 x i32>`, and the on-ramp's vector lane-arith set omitted **`shl`/`lshr`/`ashr`**, so it
   fail-closed at *translate*.

**Fix (landed, both in the I2 style):**
1. **`i8x16.mul` lowering in svm-jit** (`Inst::VIntBin` with `VShape::I8x16`): widen each half to
   `i16x8` (`uwiden_low`/`uwiden_high`), multiply (the low byte of an `i16` product equals the low byte
   of the `i8` product, sign-independent), mask each product to its low byte, then pack the two halves
   back with unsigned-saturating narrow (`unarrow` — every lane ≤ 0xFF, so nothing saturates: an exact
   low-byte truncation matching the interp's wrapping mul). Removed from the JIT's `Unsupported`
   pre-check. The interpreters already implemented `i8x16.mul`, so they needed no change.
2. **Vector `shl`/`lshr`/`ashr` in the on-ramp** (`bin`'s `vec128_shape` path): a `const_splat_int`
   helper recognizes a constant-splat shift amount (`<i32 k, …>`, the shape `clang -O2` emits for
   `v >> k`) and emits `Inst::VShift { shape, op: Shl/ShrU/ShrS, .. }` (svm-ir/verify/jit/interp already
   support `VShift` for every shape; the JIT lets Cranelift legalize even `i8x16`'s no-native-per-byte
   shift). A non-constant-splat amount still fail-closes (no corpus need yet).

**Verification.** New `cargo test -p svm` (`diff_i8x16_mul`, interp↔JIT differential) and
`cargo test -p svm-llvm --test translate` (`simd_i8x16_mul_load_store`, `simd_i32x4_const_shifts`) pin
both fixes against the native oracle. End-to-end, `corpus_diff.rs`'s `fnv` (case 1) and `crc32`
(case 2) now translate + run **vectorized** (NOVEC workaround removed) bit-identical across tree-walk,
bytecode, JIT, and native — `fnv`/`crc32` both land at ~1.03× native.

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
