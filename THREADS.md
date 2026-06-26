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
- [x] **Step 4c-miri — race/UB verification of the parallel driver.** `svm-interp/tests/parallel_miri.rs`
  drives the **real interpreter**'s concurrent atomic + non-atomic accesses over one `Region::shared`
  backing across genuine OS threads, exercising the whole parallel machinery (per-thread `fork_for_thread`
  views, the cross-thread `Futex` park/wake, the spawn/join registry). **Miri reports no data race, UB, or
  provenance error** (4-vCPU counter → 200; 2-thread futex handoff → 987654). The test lives in `svm-interp`
  (small dep set + `svm-text` dev-dep) so Miri can build it — the `svm` integration crate pulls in the
  Cranelift JIT Miri can't compile. So the genuinely-parallel substrate is now both differentially correct
  *and* memory-model-clean.

#### Remaining follow-ons (each its own project)

- [x] **4c-host — a shared `Host` for `cap.call` under parallelism.** Done by **mirroring the
  tree-walker**, which already shares one `Arc<Mutex<Host>>` across a run's vCPUs and locks it *per
  `cap.call`* (compute between calls is lock-free). The bytecode engine assumed exclusive `&mut Host`;
  a small `HostCell { Excl(&mut Host) | Shared(&Mutex<Host>) }` with one `with(|h| …)` accessor now
  threads through `resume`/`RunCtx`/every caller (commit A, zero behavior change — the full suite stays
  green). The parallel driver then shares one `Mutex<Host>` across its vCPU threads (`HostCell::Shared`,
  commit B), so a spawned vCPU's `cap.call` dispatches on the **same** powerbox, taking the lock only
  for its own dispatch — host I/O from workers works while compute/atomics/futex stay parallel. New
  entry `compile_and_run_capture_over_parallel_with_host`. **Determinism is preserved as a mode choice,
  not lost:** cooperative (the default) stays the deterministic oracle (it already shares one host, in
  fixed order); parallel is the opt-in whose order-sensitive stateful-cap interleaving races, as real
  threads do. Differential-tested (`bytecode_parallel_caps.rs`): 8 worker vCPUs each `cap.call`-write
  the same line + bump a shared counter → result + (schedule-independent) stdout byte-identical to the
  oracle across 50 real-race repeats; Miri (`parallel_miri.rs`) confirms the shared-host access is
  race/UB-free. (Scope: caps `cap_dispatch_slots` handles — streams/clock/exit/reflection — over the
  native driver; the wasm-Worker shared-`Host` and order-sensitive-cap demos are follow-ons.)
- [ ] **4c-domain — §14 `instantiate` / §22 JIT install in parallel** *(§22 landed: A/B/C1 done;
  C2 browser glue + §14 remain)*. These mutate the `Domain` (`&mut`), which the parallel driver
  shares `&`-immutably, so they fail closed today. **Motivation:** web *interpreter playgrounds* — a
  guest that JITs/`eval`s user code (§22) or sandboxes sub-programs (§14) **and** runs in parallel.
  **Design (chosen: full-unify, mirroring the tree-walker's proven `DomainTable`).** The bytecode engine's
  `Domain { mods: Vec<Compiled>, table: Vec<Option<(u32,u32)>> }` (mutated by `&mut`) becomes:

  ```rust
  struct SharedTable {                       // mirrors tree-walker `DomainTable`
      slots: Box<[AtomicU64]>,               // dispatch: 1 Acquire load; install: Release store (pack_slot)
      units: Mutex<Vec<Arc<Compiled>>>,      // installed §22 units; touched only on install / cache-miss
  }
  struct Domain { primary: Arc<Compiled>, table: SharedTable }   // used by root, §14 child, AND coroutine
  ```

  Reuse the crate-private `super::{pack_slot, unpack_slot, TableSlot, TABLE_EMPTY, INVOKE_MODULE}`.
  `install(&self, Arc<Compiled>)`/`uninstall(&self,…)` go interior-mutable (so a shared `&Domain` can
  install). Dispatch is lock-free: `table.slot(i)` Acquire-loads, pairing with the install Release.

  The realised shapes (in `bytecode.rs`): `SharedSlots { slots: Box<[AtomicU64]> }` (the dispatch
  table), `ModuleSource { mods: Mutex<Vec<Arc<Compiled>>> }` (the module store), and
  `Domain { source: Arc<ModuleSource>, table: SharedSlots }` with interior-mutable `install`/`uninstall`.

  **Plan (one big suite-gated commit for A, then B/C):**
  - [x] **A — convert the engine to the unified `Domain`** (behavior-preserving; the whole `bytecode_*`
    suite gated it). `resume`'s hot loop: `c: &Compiled` → `c: Arc<Compiled>` (clone on the rare
    module-crossing), resolved via a per-vCPU snapshot cache refreshed on a miss; `mods: &[Compiled]`
    param → `(source: &ModuleSource, table: &SharedSlots)`. Updated every `resume` caller, `step_vcpu`,
    `drive`'s `JitInstall`/`JitUninstall` arms, `run_invoke`, child/coro construction, and the
    `vm_trap_bt`/`cur_ir_pc` helpers. Dispatch gains a `Relaxed`/`Acquire` load — ~free on x86/ARM;
    child/coro pay an uncontended atomic/`Arc` cost. Cooperative order (hence determinism) unchanged.
  - [x] **B — un-fail-close `JitInstall`/`JitUninstall`/`JitInvoke` in `drive_parallel`**, dispatching
    to the shared `Domain` (`install`/`uninstall`/`push` interior-mutable; an invoked unit runs over
    the shared powerbox). Proven by `bytecode_parallel_jit.rs` (8 vCPUs concurrently invoke a pure
    unit, and concurrently `install` + `call_indirect` their own raced slot → counter byte-identical
    to the cooperative oracle, 50 runs) and `parallel_jit_miri.rs` (race/UB/provenance-clean under Miri).
  - [x] **C1 — un-fail-close the resumable `Vcpu`**: `Jit.install`/`uninstall`/`invoke` surface as
    host-serviced `VcpuEvent`s (the host resolves the unit from the powerbox, `deliver_jit_*` hands it
    back; the vCPU acts on the shared `Domain`). `VcpuProgram::compile_with_jit_table` reserves the
    granted dispatch table. Proven by `bytecode_vcpu_orchestration_jit.rs` (a `std::thread` host —
    the native model of the JS/Worker host — drives 8 vCPUs invoking / installing on the shared domain
    to the oracle's result). Invoked units run over the vCPU's deny-all powerbox (a `cap.call`ing unit
    is out of scope, like every host capability on this path).
  - [ ] **C2 — the browser glue**: wire `svm_par_*` so a guest JITs **across Web Workers**, with a
    Node/Chromium test. Open design question: where the powerbox lives across Workers (Rust-side in
    shared linear memory vs JS-side) and how a `JitInstall`/`Invoke` event triggers unit resolution
    over the wasm boundary. Until then the browser path fail-closes JIT (`svm_par_run` → `PAR_TRAP`).
  - [ ] **§14 `instantiate` in parallel** — a follow-on after §22 (children get their own confined
    `Domain`; the cooperative driver's single `extra_envs` vec doesn't map onto per-thread vCPUs —
    separate slice).
- [x] **4c-wasm — the driver's vCPUs as real wasm Workers (the browser payoff).** Done: **one** guest's
  `thread.spawn`ed vCPUs now run on **separate Workers** (Node `worker_threads` here — the same
  `SharedArrayBuffer` + `Atomics` a browser uses) over the **one** shared linear-memory window, genuinely
  in parallel. Built in three de-risked slices:
  - **The cross-Worker blocking futex** (`browser/threads-spike/threads-futex.mjs`): proved
    `memory.atomic.wait`/`notify` works across OS threads from Rust (`core::arch::wasm32`) — park/wake,
    not-equal, and timeout paths — the foundational unknown, mirroring how Step 1 de-risked atomics.
  - **A resumable per-vCPU API** in `svm-interp` (`VcpuProgram` + `Vcpu` + `VcpuEvent`): platform-agnostic,
    no threads/FFI — `run` advances one vCPU until a host-serviced event (`Spawn`/`Join`/`Wait`/`Notify`),
    the host services it and `deliver_*`s the result. Proven natively by `bytecode_vcpu_orchestration.rs`
    (a `std::thread` host — the native model of the JS host — drives the counter kernel to 4000 and a
    futex handoff to 987654), which is the wasm driver's **differential oracle**.
  - **The wasm driver** (`browser/src/lib.rs` `svm_par_*` C-ABI + `browser/threads-spawn.mjs`): each vCPU
    runs on its own Worker through the resumable API; the host services `thread.spawn` → start a Worker,
    `thread.join` → `Atomics.wait` on the child's completion slot, `memory.wait`/`notify` → `Atomics` on
    the futex word. The 4b per-Worker stack/TLS bootstrap and the "main can't `atomic.wait`" wrinkle are
    both handled (every vCPU runs on a Worker; main only fans out). Verified: the counter kernel runs on
    **9 Workers** (1 root + 8 spawned) → **4000**, and the futex kernel on **2 Workers** → **987654**,
    stable across repeats.

  Remaining `4c-host` / `4c-domain` (`cap.call` under a shared `Host`; §14/§22 domain-mutating events) are
  unchanged below — orthogonal to the threading model, each its own project.

### Known wrinkles — all resolved in `4c-wasm`

- **Main thread can't `atomic.wait`** (it traps in browsers) — *resolved*: every vCPU (including the
  root) runs on a Worker; `threads-spawn.mjs`'s main thread only compiles, carves the window, and fans
  out Workers — it never blocks.
- **Per-thread stack + TLS** — *resolved*: each vCPU's Worker sets its own `__stack_pointer` +
  `__wasm_init_tls` (the 4b bootstrap); a spawner `svm_par_alloc`s the child's stack/TLS before asking
  main to start it.
- **Thread-safe ABI** — *resolved*: the parallel `svm_par_*` path is stateless (no `static mut`); each
  vCPU's state is a heap `Box` in the shared linear memory, and the `VcpuProgram` is shared read-only by
  pointer. (The thread-safe shared allocator was de-risked by 4b.)
- **Data init once** — *resolved*: only the root `Vcpu::new_root` seeds + data-initialises the window; a
  `Vcpu::new_child` shares it without re-seeding.

---

## Reproduce

```sh
rustup toolchain install nightly -c rust-src

# Step 1 — shared-memory atomics across OS threads (tiny no_std spike)
cd browser/threads-spike
cargo +nightly build --release   # flags baked into .cargo/config.toml (shared mem + atomics)
node threads.mjs                 # two worker_threads → atomic EXACT 4,000,000; plain races
cd ..

# Step 2/3/4c — the substrate, engine bridge, and parallel drivers (native, in CI)
cargo test -p svm-mem shared                          # Region::Shared cross-thread atomics + fuzz
cargo test -p svm --test bytecode_shared_window       # engine over a caller-owned shared window
cargo test -p svm --test bytecode_parallel            # 4c: native parallel driver vs oracle
cargo test -p svm --test bytecode_parallel_caps       # 4c-host: shared-powerbox cap.call vs oracle
cargo test -p svm --test bytecode_vcpu_orchestration  # 4c-wasm: resumable Vcpu API, host-orchestrated
cargo test -p svm --test bytecode_parallel_jit            # 4c-domain B: §22 JIT in drive_parallel vs oracle
cargo test -p svm --test bytecode_vcpu_orchestration_jit  # 4c-domain C1: §22 JIT via resumable Vcpu vs oracle
cargo +nightly miri test -p svm-interp --test parallel_miri       # 4c: parallel driver + shared host race-free
cargo +nightly miri test -p svm-interp --test parallel_jit_miri   # 4c-domain B: §22 JIT shared-Domain race-free

# Step 1-futex / 4c-wasm — the cross-Worker blocking futex (tiny spike)
cd browser/threads-spike && cargo +nightly build --release && node threads-futex.mjs && cd ..

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
node threads-parallel.mjs "$W" corpus/threads.svmbc 4000 8    # 4b: 8 Workers, independent domains → 4000
node threads-spawn.mjs    "$W" corpus/threads.svmbc 4000      # 4c-wasm: ONE guest's vCPUs across Workers
node threads-spawn.mjs    "$W" corpus/futex.svmbc   987654    # 4c-wasm: the cross-Worker futex handoff
node browser-test.mjs                                        # 4c-wasm in a REAL browser (Chromium): → 4000
```

The threads-build flags (`+atomics` · `--shared-memory --import-memory --max-memory` · `build-std`)
are the reusable core; the spike's `.cargo/config.toml` and the Step-4a command share them.
