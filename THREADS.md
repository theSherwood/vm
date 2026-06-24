# THREADS.md — genuinely multithreaded SVM in wasm

Tracks making SVM-in-wasm run with **real parallelism** (multiple OS threads / Web Workers over one
shared memory), *without* losing the cooperative single-worker model we already have. Companion to
`BROWSER.md` (the interpreter-as-wasm-guest port, which today runs concurrency cooperatively on one
thread). Living doc: update the **Plan** tracker as work lands; fold into `DESIGN.md` once it closes.

---

## Goal & shape

Today the wasm port runs guest `thread.spawn`/`join` + futex + atomics on the bytecode engine's
**cooperative `drive`** — correct concurrency *semantics*, but multiplexed onto **one** OS thread
(M:1, no parallelism). We proved that works (the 4000-counter kernel). The goal is to *also* offer a
**parallel** backend that uses N cores, as a **host decision behind a clean API** — keeping
cooperative as the default.

### Two execution modes, one guest

The guest module is **byte-identical** under both modes — same `thread.spawn`/`wait`/`notify` +
atomics, same semantics. Only the *execution strategy* differs, selected **host-side**:

| | **Cooperative** (default) | **Parallel** (opt-in) |
|---|---|---|
| vCPUs | all time-share one thread (`drive`) | one **Web Worker per vCPU** (1:1) |
| futex | in-process `wait_waiters` queue | `memory.atomic.wait`/`notify` |
| memory | private linear memory | **shared** linear memory |
| host imports | none | a Worker-spawn capability |
| determinism | yes (replayable, the **oracle**) | no (real races) |
| runs where | anywhere | needs `SharedArrayBuffer` (COOP/COEP) |

**Both stay, deliberately.** Cooperative is not legacy: it's (1) the **deterministic oracle** the
parallel backend is differential-tested against, (2) the **universal deployment** for contexts that
can't enable cross-origin isolation, and (3) the basis for replay/time-travel.

### Why this respects D56

The parallel backend is **1:1** — one Worker per vCPU — which is exactly D56's "a vCPU = one OS
thread" primitive realized in wasm. We do **not** reintroduce the removed M:N in-VM scheduler; the
host's Worker pool is the runtime. Cooperative is just "all vCPUs on one thread." Neither bakes a
scheduler into the VM.

### The clean API

The split lives where it costs nothing: the **guest ABI is unchanged**, so the new surface is purely
host-facing — an **executor seam** (`Cooperative` | `Parallel`, the same shape as the tree-walker's
`SchedRef::Real` vs `Det`) the host selects at the run entry. wasm makes shared memory a *module-level*
property, so in practice:

- **`svm_browser.wasm`** (today) — import-free, **cooperative-only**, runs anywhere.
- **`svm_browser_threads.wasm`** (`--features threads`) — shared memory + `+atomics` + a Worker-spawn
  import; the host picks cooperative **or** parallel at run time within it. (Cooperative still works
  here — and *must* use the queue futex, since a lone Worker can't `atomic.wait` on itself.)

---

## Plan

- [x] **Step 1 — shared-memory atomics spike (`browser/threads-spike/`).** De-risk the foundational
  unknown: do Rust→wasm shared-memory atomics work across OS threads? **Yes.** A tiny `no_std`
  module imports one host-owned shared memory; two Node `worker_threads` each run 2,000,000
  increments of a single shared cell. Result: **atomic → exactly 4,000,000** (`i32.atomic.rmw.add`
  across two OS threads on contended memory), and the **non-atomic path lost ~1.4M updates** —
  proving the workers genuinely ran in parallel, so the atomic correctness isn't serialized luck.
- [x] **Step 2 — `Region::Shared` svm-mem backing.** A new `Region` variant over **caller-owned**
  memory (`unsafe fn Region::shared(base, size)`), with the *same* raw-pointer hardware atomics as
  the unix `Mapped` mmap backing — but borrowed, not owned, and available on **every** target (so it
  spans the wasm shared linear memory). The atomic/byte/word bodies lower to `core::sync::atomic`
  (→ `i32`/`i64.atomic.rmw` under wasm `+atomics`). Verified natively (the substrate stand-in for the
  wasm Worker pool): **8 OS threads racing one counter through `&Region::Shared` land on the exact
  total**, the new `differential_shared_vs_paged_fuzz` gates `Shared` against the safe `Paged` model
  byte-for-byte across 20k random ops (so it can't drift from `Mapped`), and both pass under **Miri**
  (no UB / data race / provenance error). Compiles clean for `wasm32`; `Mapped` left untouched (zero
  regression to the existing TCB). The engine doesn't use it yet — that's Steps 3–4.
- [x] **Step 3 — engine over a caller-owned shared window (the substrate→engine bridge).** Reading
  the engine reshaped this step: the bytecode `drive` *is* the cooperative executor (a one-thread
  `tasks` loop), so a "Cooperative | Parallel executor seam" isn't a trait swapped inside `drive` —
  the parallel executor is a *different driver* (per-Worker), which folds into Step 4 alongside the
  host mode selection. The real prerequisite, done here, is letting the engine **run over the shared
  backing**: `Mem::with_reservation_over(Arc<Region>)` + the public `bytecode::compile_and_run_capture_over`
  take a caller-built `Region::shared` window instead of an engine-`mmap`ped one. The engine stays
  `#![forbid(unsafe_code)]` (it accepts a pre-built `Arc<Region>`; the `unsafe` borrow lives in the
  embedder's `Region::shared`); `Region` is now re-exported from `svm-interp`. Verified
  (`bytecode_shared_window.rs`): a compute+memory guest over the caller-owned window is **byte-identical**
  (result + final image) to the engine's own backing and its writes land in the caller's buffer, and
  the 8-vCPU `thread.spawn`+atomics+futex kernel runs cooperatively over the shared window → **4000**.
  This is the exact window the parallel mode will run every Worker over.
- [x] **Step 4a — the engine runs over **shared wasm linear memory** (real-wasm integration).** The
  whole point of steps 1–3, now proven in the actual artifact: the **full** SVM engine (not the spike)
  builds as a wasm **threads module** — `+atomics`/`+bulk-memory`/`+mutable-globals` ·
  `--shared-memory --import-memory` · `build-std=std,panic_abort` — so svm-interp/svm-mem with all
  their `Mutex`/`RwLock`/`Arc` compile and instantiate over a host `SharedArrayBuffer` (a major
  de-risk: ~54 s, clean). New export `svm_run_shared(mod, len, win_ptr, win_size, arg)` runs a guest
  over a `Region::shared` window the **host** carves out of that shared linear memory (Step 3's
  `compile_and_run_capture_over`). Verified (`threads-engine.mjs`): the 8-vCPU `thread.spawn`+atomics+
  futex kernel runs over a window **in the SharedArrayBuffer** → **4000**. Stateless (no `static mut`),
  so two Workers over disjoint windows won't race on ABI globals. Still **cooperative** (one thread) —
  this is the substrate + window the parallel driver distributes; the default build stays import-free
  (185/185 differential intact).
- [x] **Step 4b — genuine multi-core parallelism in wasm (independent domains).** Real `worker_threads`
  (separate OS threads) each run the **full SVM engine** over the **one shared** `SharedArrayBuffer`,
  each over its own guest window — **concurrently**. The hard wasm-threads hurdle is solved: each
  Worker is bootstrapped with its **own stack + TLS block** (export `__stack_pointer` / `__tls_*` /
  `__wasm_init_tls`; the main thread pre-allocates the per-Worker stacks+TLS in shared memory so a
  Worker never touches the shared default stack before its own is set). Verified
  (`threads-parallel.mjs`): **4 and 8 Workers** each run the 8-vCPU `thread.spawn`+atomics+futex kernel
  (so 8×8 = 64 vCPUs total) over disjoint windows → every Worker returns **4000**, robust across
  repeats. So SVM runs genuinely in parallel across cores in wasm — N programs, N threads, one shared
  memory. (The runs are *independent* — each Worker its own window; one guest's `thread.spawn` vCPUs
  fanned across Workers is step 4c.)
- [x] **Step 4c — the shared-memory `thread.spawn` parallel driver (native slice).** The host-selected
  `Parallel` mode now exists: `bytecode::drive_parallel` runs **one** guest's `thread.spawn`ed vCPUs on
  **separate OS threads** (the native stand-in for per-vCPU Workers) sharing **one** `Region::shared`
  window — genuine cross-core `thread.spawn`/`join` + hardware `atomic.*`, not a single-thread
  interleaving. Each vCPU runs on its own scoped `std::thread` over a `fork_for_thread` view of the
  shared backing (so the `Arc<Region>` bytes + address space are shared, real atomics); `std::thread::scope`
  borrows the `&Domain` (now `Sync`) and a `ThreadRegistry` (`Mutex<HashMap>` + `Condvar`) into each
  child, with the registry serving the cross-thread `thread.join` rendezvous (handle→id, value-or-trap
  delivery, the `MAX_VCPUS` anti-bomb gate). The root runs on the calling thread and `join`s via the
  condvar — never `atomic.wait`, sidestepping the browser main-thread-wait wrinkle. Differential-tested
  against the cooperative oracle (`bytecode_parallel.rs`): the 8-vCPU counter kernel → **4000** and a
  join-value kernel → **46**, both **byte-identical** to `compile_and_run_capture` and **stable across
  50 real-race repeats** (a wrong driver would be flaky). New public entry
  `compile_and_run_capture_over_parallel` (the `Parallel` sibling of `compile_and_run_capture_over`).
- [x] **Step 4c-futex — the cross-thread `memory.wait`/`notify`.** The parallel driver now services the
  **full threads model**, not just spawn/join: a native `Futex` (a per-address parked-token queue under
  one bucket lock — the std-sync analogue of a kernel futex bucket, with the compare-and-park done under
  the lock so a `notify` can't slip in and lose a wakeup) backs guest `memory.wait`/`notify` across the
  real OS threads. In real wasm this role is `memory.atomic.wait`/`notify` directly; here it serves the
  cooperative oracle's same semantics for genuinely parallel vCPUs (the not-equal fast path, the parked
  path, and the timeout path). Differential-tested (`bytecode_parallel.rs`): the futex-handoff kernel →
  **987654** (consumer either parks-and-is-woken or wins the not-equal race) and an 8-worker **barrier**
  where the **root** genuinely parks until the last worker `notify`s it → **8**, both matching the oracle
  and stable across 100/50 real-race repeats. So `Parallel` now covers `thread.spawn`/`join` +
  `memory.wait`/`notify` + atomics + compute — the complete pure-threads model — genuinely in parallel.
  Remaining fail-closed (need a `&mut Domain`/shared powerbox): §14 `instantiate`, §22 JIT install, and a
  shared locked `Host` for `cap.call`; plus running the driver's vCPUs as real wasm Workers (per-Worker
  stack/TLS from 4b) — the follow-ons below.

### Known wrinkles (surfaced for later steps)

- **Main thread can't `atomic.wait`** (it traps in browsers) — the root vCPU must run on a Worker, or
  poll non-blockingly.
- **Per-thread stack + TLS** — each Worker needs its own `__stack_pointer` / `__wasm_init_tls` block;
  the spike sidesteps this with register-only functions, but a real runtime must set it up.
- **Thread-safe ABI** — the cdylib's `static mut` scratch/state globals race under shared memory; they
  must become per-Worker (TLS) or per-instance.
- **Data init once** — under `--shared-memory` only the first instance may run memory init; workers
  set up TLS/stack and skip it.

---

## Reproduce

```sh
rustup toolchain install nightly -c rust-src

# Step 1 — shared-memory atomics across OS threads (tiny no_std spike)
cd browser/threads-spike
cargo +nightly build --release   # flags baked into .cargo/config.toml (shared mem + atomics)
node threads.mjs                 # two worker_threads → atomic EXACT 4,000,000; plain races
cd ..

# Step 2/3 — the substrate + engine bridge (native, in CI)
cargo test -p svm-mem shared                          # Region::Shared cross-thread atomics + fuzz
cargo test -p svm --test bytecode_shared_window       # engine over a caller-owned shared window
cargo test -p svm --test bytecode_parallel            # 4c: parallel driver (real OS threads) vs oracle

# Step 4a/4b — the FULL engine as a wasm threads module, run over a SharedArrayBuffer window.
# The `--export=__stack_pointer/__tls_*/__wasm_init_tls` are the per-Worker bootstrap hooks (4b).
cargo run --bin gencorpus                             # → corpus/threads.svmbc
RUSTFLAGS="-Ctarget-feature=+atomics,+bulk-memory,+mutable-globals \
  -Clink-arg=--shared-memory -Clink-arg=--import-memory -Clink-arg=--max-memory=1073741824 \
  -Clink-arg=--export=__stack_pointer -Clink-arg=--export=__tls_base \
  -Clink-arg=--export=__tls_size -Clink-arg=--export=__tls_align -Clink-arg=--export=__wasm_init_tls" \
  cargo +nightly build -Z build-std=std,panic_abort --release --lib --target wasm32-unknown-unknown
W=target/wasm32-unknown-unknown/release/svm_browser.wasm
node threads-engine.mjs   "$W" corpus/threads.svmbc 4000      # 4a: engine over a shared-mem window
node threads-parallel.mjs "$W" corpus/threads.svmbc 4000 8    # 4b: 8 Workers, real parallelism → 4000
```

The threads-build flags (`+atomics` · `--shared-memory --import-memory --max-memory` · `build-std`)
are the reusable core; the spike's `.cargo/config.toml` and the Step-4a command share them.
