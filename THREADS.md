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
- [ ] **Step 3 — the executor seam.** `Cooperative` | `Parallel` behind one interface
  (`spawn`/`wait`/`notify`); the threads build runs either, host-selected, cooperative default.
- [ ] **Step 4 — `thread.spawn` → Worker-spawn import + atomic futex**, then differential-test the
  parallel backend against the cooperative oracle on the existing corpus.

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

## Reproduce (step 1)

```sh
rustup toolchain install nightly -c rust-src
cd browser/threads-spike
cargo +nightly build --release   # flags baked into .cargo/config.toml (shared mem + atomics)
node threads.mjs                 # two worker_threads → atomic EXACT 4,000,000; plain races
```

The build recipe (in `.cargo/config.toml`) is the reusable core for the real threads build:
`+atomics,+bulk-memory,+mutable-globals` · `--shared-memory --max-memory=… --import-memory` ·
`--no-entry` · `build-std` (core must be recompiled for `+atomics`).
