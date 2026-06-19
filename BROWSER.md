# BROWSER.md — the interpreter as a wasm guest (run SVM in the browser)

Tracks the design and implementation of compiling the SVM **bytecode interpreter**
(`crates/svm-interp/src/bytecode.rs`) to a wasm target so an SVM guest can run **inside a browser**
(or any wasm host).

> **Not to be confused with `WASM.md`.** That doc is the *inbound* `svm-wasm` frontend
> (wasm bytes → SVM IR). This doc is the *outbound* direction: the interpreter itself compiled
> *to* wasm. The two never meet — a guest could even be wasm transpiled by `svm-wasm`, then run by
> this interpreter-in-wasm.

It is a living document: update the **Status** and the **Phase tracker** as work lands. Fold into
`DESIGN.md` and drop this file once the work closes (repo convention, cf. `WASM.md`).

---

## Why

The interpreter is a self-contained, `#![forbid(unsafe_code)]`, cooperatively-scheduled execution
engine whose only sandbox is SVM's own masking + guard-page confinement (`svm-mask`/`svm-mem`).
Compiling it to wasm makes it **embeddable in the browser** with zero native dependencies — ship one
`.wasm`, run SVM guests client-side. The payoff is portability/embeddability, **not** added
isolation: inside wasm you are double-sandboxing (the host's wasm sandbox over SVM's own).

## Target: `wasm64`, not `wasm32`

SVM addresses are `u64` end-to-end — `svm-mask` confines into `[0, 1<<reserved_log2)` and every
`svm-mem` offset is a `u64`. `wasm64` (memory64) is therefore the production target: `wasm32` would
force a `u64`→32-bit narrowing at every guest access. `wasm32-unknown-unknown` remains useful as a
quick, flag-free Node-runnable smoke target for **compute-only** guests (no/small guest memory),
where the address width is immaterial.

## Why the bytecode engine (not the tree-walker)

`run`/`run_with_host` (the tree-walker) is the reference **oracle**, scheduled by a production
`Scheduler` that uses real OS threads and wall-clock timers — hostile to wasm. The **bytecode
engine** is reached via `run_fast`/`run_with_host_fast` and is the right target:

- **No platform surface in `bytecode.rs`** — no `std::thread`, no `Instant`, no `page_size`.
- **Its own single-OS-thread cooperative scheduler** — `drive` (`bytecode.rs:2377`) multiplexes
  guest **threads** (`thread.spawn`/`join` + `memory.wait`/`notify`), **fibers** (`step_vcpu`,
  `bytecode.rs:1916`), **coroutines**, and §14 executor children cooperatively on one OS thread over
  one shared `Mem`. A `wait` with no runnable task fires the earliest timeout **deterministically**
  (`bytecode.rs:2471`) — no wall-clock.
- **A ready-made embedder entry** — `compile_and_run_capture_reserved_with_host`
  (`bytecode.rs:1202`) takes an embedder `&mut Host` + init-memory image + reservation, runs, and
  returns a `Capture` (results **and** the final memory snapshot). This is the browser entry point;
  no new public API is required.
- **Clean deps** — `svm-ir`, `svm-mask` (`#![no_std]`), `svm-mem` (non-unix `Paged` fallback),
  `svm-verify` (pure), `page_size`. No `svm-fiber` (asm stack-switch — fibers here are
  continuation-based) and no `svm-jit` (Cranelift). Nothing architecture-specific is dragged in.

---

## Status

**Viability: PROVEN.** Two facts, both reproduced (not just argued):

1. **Compiles to `wasm64` unmodified.** `svm-interp` builds clean for `wasm64-unknown-unknown` via
   `-Z build-std` (a ~26 MB rlib). The `std::thread`/`Instant`/`page_size` references *compile* on
   wasm (they exist as symbols); they are a **runtime** concern only, and the bytecode engine's
   cooperative `drive` never invokes real OS threads — so it sidesteps them. cfg-gating the
   tree-walker `Scheduler` for wasm is therefore **dead-code cleanup, not a correctness blocker**.

2. **Executes correctly in a wasm sandbox.** The throwaway `wasm-harness/` crate runs a guest
   through `bytecode::compile_and_run`, compiles to wasm, and runs in Node/V8 with **zero host
   imports** (`imports required: []` — a fully self-contained sandbox). The hand-derived anchors
   `run_guest(0) == 0` and `run_guest(1) == 1442695040888963407` pass, exercising loops, i64
   wrapping arithmetic, conditional branches, and SSA block-arg copies.

### Reproduce

```sh
# (1) wasm64 library compile (nightly + rust-src for build-std)
rustup toolchain install nightly -c rust-src
cargo +nightly build -Z build-std=std,panic_abort \
  -p svm-interp --target wasm64-unknown-unknown

# (2) end-to-end execution in a wasm sandbox (wasm32, Node-runnable without flags)
rustup target add wasm32-unknown-unknown
cd wasm-harness
cargo build --release --target wasm32-unknown-unknown
node run.mjs        # asserts the correctness anchors above
```

`wasm-harness/` is a **throwaway** spike (detached `[workspace]`, source committed, build artifacts
git-ignored) — kept as a reproducible viability artifact, **not** part of the production build.

---

## Decisions

- **Fallback policy → fail-closed (v1).** When `compile_module` returns `None` (rare seams:
  instantiate-mixed-with-fibers, multi-fiber durable freeze), the wasm entry returns a clean
  `Unsupported`-style trap rather than dropping to the tree-walker's threaded `Scheduler`. So the
  tree-walker `Scheduler` is purely cfg-gated *out* of wasm — no cooperative-fallback porting.
  (Non-durable guest threads still run on the engine's cooperative `drive`; only *durable* `thread.*`
  is refused, by `compile_and_run_capture_reserved_with_host` itself.)
- **Host capabilities → compute-only first.** v1 supplies a deny-all `Host` (empty powerbox, any
  `cap.call` is inert). Browser-backed capabilities (console/IO/clock bound to JS) are deferred.

---

## Non-portable surface (all in `lib.rs`; bytecode path uses none of it)

Compile-clean today; gate behind `cfg(not(target_family = "wasm"))` for a clean production build:

1. **Tree-walker production `Scheduler`** — `available_parallelism`, `JoinHandle`/`thread::spawn`,
   and its `Instant` timer uses.
2. **Blocking-offload host pool** — `OffloadPool`/`AsyncState` (a `std::thread` pool +
   `thread::sleep`); stub to run inline on wasm (`AsyncState::mix` is already deterministic).
3. **`page_size` crate** — `host_page_size()` / `region_page_granularity()`; gate to a constant
   (65536) under wasm.

(`svm-mem`/`svm-mask`/`svm-verify` need no work: Paged fallback; `#![no_std]`; pure logic.)

---

## Phase tracker

- [x] **Spike — viability.** wasm64 compile + Node execution of a guest, correctness anchors green.
- [ ] **cfg-gate `lib.rs` for wasm** — `Scheduler`, `OffloadPool`, `std::thread`/`Instant` imports;
  `page_size` → constant. Native build/tests unaffected.
- [ ] **wasm64 entry crate** — a `cdylib` exporting an entry over
  `compile_and_run_capture_reserved_with_host`; on `compile_module == None`, return a clean
  `Unsupported` trap (fail-closed). Supply a deny-all `Host` (compute-only v1).
- [ ] **Browser load + differential check** — load the `wasm64` module (memory64) and run a guest —
  incl. one doing a guest `thread.spawn` (cooperative `drive`) — asserting byte-identical
  results/memory to the native bytecode engine.
- [ ] **Host powerbox (follow-up)** — design the browser-backed capability set (console/IO/clock).

## Verification

- **Builds:** the two `cargo build` lines under **Reproduce** (wasm64 via build-std; wasm32 smoke).
- **No semantic drift natively:** the bytecode↔tree-walker exact-equality harnesses
  (`crates/svm/tests/bytecode_diff.rs` + the `bytecode_{caps,fibers,threads,coroutines,instantiate,
  tailcall,debug,durable,dynlink}.rs` suite) must stay green after the cfg-gating — proving the port
  didn't disturb engine semantics.
- **Runs in a wasm host:** `node wasm-harness/run.mjs` (compute-only); later, a memory64 load with a
  byte-identical differential check against native.
- **Confinement intact:** `svm-mask` property/fuzz tests compile and pass unchanged.
