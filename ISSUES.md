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

### I2 — LLVM on-ramp cannot ingest auto-vectorized output wider than 128 bits (no vector legalization) (S3)

**Where:** `crates/svm-llvm/src/lib.rs` — `val_type` / `type_size` (vector type recognition) and the
vector lowering paths (`bin` vec branch, `lower_int_intrinsic` vec branch, `lower_vector_reduce`,
`is_vec4`). Reached when translating bitcode compiled with auto-vectorization **enabled** (the
`check_vectorized_vs_native` test harness; the C/C++/Rust breadth lanes deliberately pass
`-fno-*-vectorize` / `-vectorize-loops=false`, so they don't hit it).

**Symptom:** translating a `clang -O2`-vectorized program fails **fail-closed** with
`Error::Unsupported("type <16 x i32> (Milestone 1+)")` (or `<16 x i64>`, `<4 x i64>`, `<8 x i8>`,
`<2 x i64>`, `<16 x i8>`, etc.). No wrong result, no escape — the verifier never sees bad IR; the
translator just refuses. Observed when probing whether the breadth lanes could re-enable vectorization
(slices AN/AO): ~7 corpus tests surfaced these shapes.

**Root cause:** svm-ir's SIMD type is a **fixed-128-bit `v128`** (§17/D58). LLVM's `-O2` loop/SLP
vectorizer emits **arbitrary-width "virtual" vectors** — both wider than 128 bits (`<16 x i32>` =
512-bit, from the chosen vectorization factor × interleave/unroll) and assorted sub-/non-`i32x4`
128-bit shapes — on the assumption that the target backend's **type-legalization** pass (LLVM
SelectionDAG `LegalizeTypes`) will split/scalarize them into legal-width chunks before instruction
selection. The on-ramp has no such pass: it maps `<4 x {i32,float}>` (and 2-lane → packed `i64`)
directly and rejects everything else. So it can ingest *controlled* 128-bit `i32x4` vector code
(slices AN/AO: lane add/sub/mul/and/or/xor/min-max + `llvm.vector.reduce.*.v4i32`) but not the
mixed/oversized shapes real auto-vec produces.

**Current posture (intended):** the C/C++/Rust breadth lanes keep `-fno-*-vectorize`, so they hand the
on-ramp scalar IR and run byte-identical to native. This is the correct fail-closed stance — the
on-ramp never silently mis-translates a vector it can't represent (§2a: a gap is a clean error, never
an escape). `i32x4` auto-vec *is* supported and tested for code that produces only that shape.

**Fix sketch (deferred — it's its own project):**
1. **A vector-legalization pass in `svm-llvm`** (the SelectionDAG `LegalizeTypes` analog): split a
   `<N x T>` whose byte-width > 16 into `ceil(N·sizeof T / 16)` `v128` chunks (and a scalar tail),
   rewriting each vector op as per-chunk ops; scalarize widths with no clean split. This is the real
   unblock and the bulk of the work. Alternatively, run `opt`/`llc`-style legalization **out of
   process** (the PNaCl `pnacl-abi-simplify` model the on-ramp already uses for `mem2reg`), if a pass
   exists that lowers vectors to ≤128-bit without selecting machine code.
2. **Generalize the 128-bit shapes first** (bounded, independent): a `is_vec128` recognizer +
   shape-aware lowering for `i8x16`/`i16x8`/`i64x2` (and the narrow `<8 x i8>` etc. via widen), so all
   legal-width shapes work even before the splitter lands.
3. **Cheap experiment worth trying before (1):** `clang -mprefer-vector-width=128
   -mllvm -force-vector-interleave=1` may constrain the vectorizer to 128-bit, un-interleaved output —
   if it removes the wide shapes, much real auto-vec code would translate with only step (2). Unproven.

**Why not now:** the breadth proof (C/C++/Rust running real programs byte-identical to native) is
complete with vectorization off; full auto-vec ingestion is a perf/fidelity nicety gated on a
substantial legalization pass, not a correctness blocker. Tracked here so it isn't rediscovered.

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
