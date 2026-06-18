# INTERP_PERF.md — Reference-interpreter performance

The reference interpreter (`crates/svm-interp`) is the escape-TCB **oracle**: the JIT is
differentially tested against it, so it must stay total, panic-free, `#![forbid(unsafe_code)]`, and
deterministically detect-and-trap. It is also the metered, debuggable, cooperatively-scheduled
execution engine (fuel, breakpoints/watchpoints, fibers/threads, durability). All of that is per-op
work a raw JIT never does — so it will always be slower — but it had been **far** slower than it
needs to be. This document tracks the work to close that gap, and the design constraints that bound
it.

It is a living document: update the **Status** table and the **Phase tracker** as work lands.

---

## Status

Benchmark: `cargo test -p svm --release --test interp_perf -- --nocapture --ignored`
(three hand-written kernels run through interp / JIT, plus a CPython reference for the same
computation via `tests/interp_perf.py`). Numbers are ns per loop iteration on the dev box; treat as
ratios, not absolutes (the build machine is noisy — the bench takes best-of-N with a big−small
subtraction).

| kernel              | interp (origin) | interp (now) | JIT   | CPython | interp/JIT | vs CPython     |
|---------------------|-----------------|--------------|-------|---------|------------|----------------|
| alu recurrence      | ~319            | ~66          | ~1.6  | ~91     | ~42×       | 1.4× faster    |
| call/return loop    | ~252            | ~78          | ~1.0  | ~56     | ~75×       | 1.4× slower    |
| memory load/store   | (added later)   | ~152         | ~0.33 | ~44     | ~467× *    | 3.4× slower    |

\* The JIT *elides* the kernel's redundant store/load (dead-access elimination), so 467× overstates
the structural gap; a non-redundant memory workload would show the JIT doing real masked accesses
(~2–4 ns) for ~40–75×. The memory path is nonetheless the widest real gap (see "Why memory is
special").

**Goal.** Compute-bound code (alu/calls) into the ~10–25× band (competitive with a good bytecode
interpreter, faster than CPython across the board); memory as close to its software-checked floor as
the safety model allows.

---

## Why the interpreter is slow (diagnosis)

Per-op cost on the hot path, roughly in order of impact found so far:

1. **IR-walking, not bytecode.** The engine walks the SSA IR data structure: a `Vec<Block>`, each a
   `Vec<Inst>`, operands referenced by block-local `ValIdx` indexed into a per-frame `Vec`. Every op
   pays bounds-checked `Vec` indexing (`frames[top]`, `block.insts[i]`, `vals[idx]`) and re-reads the
   instruction's type/width from the enum. A real bytecode interpreter compiles to a flat
   instruction array with **pre-resolved operand offsets**, a single instruction pointer, and a
   value **register array**, eliminating most of that plumbing.

2. **Per-op metering / scheduling / debug seam.** Every op pays: a fuel `checked_sub`, a
   preemption-budget check, a `memop`/visibility check, and a `debug.is_some()` gate. A JIT pays
   none of these. Some is reducible (e.g. charge fuel at back-edges, not per op — see Phase 3
   constraints).

3. **Value width & dispatch-call overhead.** *(largely addressed — see Completed.)* The 24-byte
   `Value` enum became a 16-byte raw `Reg` slot, and the hottest ops were lifted out of a
   non-inlined `eval_inst` call into the dispatch loop.

4. **`Arc<[Func]>` reclone per block entry.** *(addressed — see Completed.)* The module resolution
   atomically refcount-bumped on every branch/back-edge.

### Why memory is special

A guest load/store must be **confined** (address masked into the window + bounds) and
**protection-checked** (page mapped? writable?) before the access. The JIT gets this *for free from
the MMU*: the window is mapped with guard pages, the address is masked in 1–2 instructions, and an
out-of-bounds access faults in hardware. The interpreter does the checks **in software**, per
access, deliberately: it is `#![forbid(unsafe_code)]` and is the reference that must deterministically
detect-and-trap (it cannot lean on SIGSEGV/guard-page tricks the way the JIT does). So memory has a
hard software floor (~a mask + a mapped/writable bit-test, ~5 cycles) well above the JIT's ~1
instruction. We can approach that floor; we cannot reach the JIT.

---

## Constraints / invariants (do not regress)

- **Oracle fidelity.** Behavior must stay byte-identical on verified modules: same results, same
  trap kinds, same final memory window. The differential suite is the spec.
- **Totality & safety.** No panics on any input (verified or not); `#![forbid(unsafe_code)]` stays.
- **Public API unchanged.** `run*`/`Inspector` keep returning `Value`; conversions happen only at
  the API / capability / debugger boundaries.
- **Seams preserved.** Fuel metering, deterministic-explorer preemption (`budget`/`memop`), the
  debug seam (breakpoints/watchpoints/stepping keyed by `IrPc = (module, func, block, inst)`),
  fibers/coroutines, threads, durability (freeze/thaw), and capability calls must all keep working.
  This is the hard part of any dispatch rewrite — the new execution model must still expose every
  seam.
- **Determinism.** Fuel/scheduling changes must not make the interp diverge from the JIT on a
  verified module (e.g. a fuel change that turns a completing run into `OutOfFuel` is a divergence).

---

## Completed work (PR #52, branch `claude/interp-perf`)

Each landed against the full oracle (jit_diff, the generative interp-vs-JIT fuzzers, escape_oracle,
durable/fiber/concurrent/dynlink suites, debug). Cumulative: alu ~319 → ~66 ns (~5×).

- **Allocation-free hot-loop branching** — reuse a scratch buffer for block-arg edges (ping-pong)
  instead of a fresh `Vec` per taken branch.
- **eval_inst dispatch + typed operand reads** — fold the no-result stores into the main match;
  read operands as the op's static type instead of copying a whole `Value`.
- **Allocation-free common return** — gather results into a reusable buffer, copy into the caller.
- **Tier-1 raw-slot value model** — `Frame.vals: Vec<Reg>` (16-byte POD: scalar bits in `lo`, v128
  in `lo`/`hi`) replacing the 24-byte `Value` enum; op-directed reads; boundary conversions only at
  API / cap / debugger. Debugger value-typing reuses `svm_verify::func_value_types` (single source
  of truth).
- **Fast-path dispatch for pure ops** — the hottest ops (`Const*`, `IntBin`, `IntCmp`, then the
  float/convert/select set, then `Load`/`Store`) dispatch directly in the eval loop, reusing the
  shared semantic helpers, instead of paying the `eval_inst` call. (This was the largest single win
  for compute-bound code.)
- **Module-resolution cache** — resolve `Arc<[Func]>` once per module change, not per block entry.
- **Benchmark + CPython reference** — `interp_perf` now prints interp / JIT / CPython per kernel.

---

## Plan: bytecode-dispatch rewrite

The remaining structural win is to stop walking the IR and instead **compile each function once into
a flat, operand-resolved bytecode** and interpret that. The whole thing is staged so every phase
lands green on its own and is individually measurable; we stop/relate to ROI at each boundary.

Open design question threaded through all phases: **how far to flatten.** Two viable shapes:
- **(A) Per-block compiled op array**, keeping the `(block, inst)` structure and the `'frames`
  loop. Operands pre-resolved to slot offsets; result slots precomputed; branch targets resolved.
  The debug `IrPc` maps 1:1 to `(block, op-index)`. *Lower risk — preserves every seam's shape.*
- **(B) Fully flat bytecode** with a single instruction pointer across blocks, threaded dispatch,
  and a PC→`IrPc` side table for the debugger. *Higher ceiling, higher risk (every seam must be
  re-expressed against a linear PC).*

Recommendation: do **(A) first** (it captures the operand-resolution and dispatch wins while keeping
the seams intact), then evaluate (B) as a stretch once (A)'s ceiling is measured.

### Phase 0 — contained wins · ✅ DONE
See "Completed work". Got alu to ~5× of origin; exhausted the cheap, in-place wins.

### Phase 1 — compile pass + per-block bytecode (shape A)

> **ROI spike (done — `crates/svm/tests/bytecode_spike.rs`):** a self-contained flat-bytecode
> compiler+executor measured **~3.5× faster** on the ALU kernel (62.5 → 17.8 ns/iter) and **~3.0×**
> on the call/return kernel (78.7 → 26.0 ns/iter) than the tree-walker, *keeping the per-op fuel
> check, under `forbid(unsafe)`*. The call path uses **register windows** (one big register file, each
> activation a `[base, base+nslots)` window — no per-call allocation, no `Arc` clone, no `frames[top]`
> indexing); at 26 ns it would be ~2× faster than CPython on calls (vs ~1.4× slower today). The win
> comes from a flat op
> array (no `frames[top]` indexing, no per-block re-resolution), a preallocated **global-slot**
> register file (each SSA value a function-wide slot → no per-edge `Vec`/swap, no `push`), branches
> copying straight into the target block's param slots, and a small dispatch enum. The integrated
> version must use 16-byte `Reg` and keep *all* seams, so it'll land higher than 17.9 ns — but even at
> 2× the spike this is a large, clearly worthwhile win (it revises the earlier ≤1.8× guess up). The
> global-slot model is the main departure from today's per-frame `Vec<Reg>` and is what the real
> compiler must adopt.

- Add a `compile` step: per function, a cached `Program` of per-block compiled ops. Each op carries
  pre-resolved operand **slot offsets**, its result slot, and (for terminators) resolved block
  targets. Built once per run (indexed by `FuncIdx`), reusing `svm_verify` types for slot widths.
- Execute the compiled ops in the existing `'frames`/block loop; the inner per-op work becomes
  "read pre-resolved slots → compute → write result slot", no `ValIdx` decode, no per-op type
  re-derivation.
- Keep all seams unchanged (`IrPc` ↔ `(block, op-index)`).
- **De-risking:** before switching execution over, add a test harness that compiles + runs the new
  path and asserts result/trap/memory equality against the tree-walker on the generator corpus.
- **Success:** full oracle green; measurable drop on alu/call kernels; no API change.

### Phase 2 — memory-op specialization + software fast-path
- **[done] Lock-free `check_prot` fast path.** `check_prot` took a `RwLock` *read* guard on **every**
  access just to test `prot.is_empty()`. Added a monotonic `Mem::prot_dirty` flag, set once at the
  `space_write` choke point (the only path that mutates the address space — `map`/`unmap`/`protect`,
  §13 region alias, demand/supply paging). While clear (the common case: no syscalls, no coroutines,
  no regions) an in-prefix access skips the lock entirely. Also hoisted the per-byte `has_regions`
  check out of `read_le`/`write_le`. Benefits the **default tree-walker** (and the bytecode engine),
  not just the compiled path. Measured on the tree-walker memory kernel: ~176 → ~147 ns (~17%).
  All oracle suites byte-identical (jit_diff, escape_oracle, shared_region, address_space,
  durable_prot_capture, concurrent_escape_fuzz, dpor, coroutine, threads, simd).
- Width-specialized load/store handlers in the compiled form; drop the `Value`↔slot round-trip at
  the `Mem` boundary (store raw slot bits; load returns slot bits directly).
- Inline the common-case confinement: a single mask + a mapped/writable bit-test, falling back to
  the full `confine_checked`/`check_prot` path on the cold/edge cases (RO pages, unmapped tail,
  aliased/§13 regions, atomics alignment). Keep the exact trap semantics.
- **Success:** memory kernel drops toward the software floor; escape_oracle + shared_region +
  address_space still byte-identical.

### Phase 3 — per-op seam overhead
- Move fuel accounting to **back-edges/calls** instead of per op (still bounds every loop, so
  termination is guaranteed) — *only if* it can be shown not to change verified-module
  trap-vs-complete outcomes vs the JIT. If that can't be guaranteed cheaply, keep per-op fuel.
- Hoist/curry the debug-seam and preemption checks out of the inner loop for the common
  (undebugged, real-pool) case.
- **Success:** lower fixed per-op cost on all kernels; determinism and debug behavior preserved.

### Phase 4 — stretch: full flat bytecode (shape B)
- Single instruction pointer, threaded/tail-call dispatch, value register array with minimized
  bounds checks, PC→`IrPc` table for the debugger.
- Only if Phases 1–3 leave meaningful headroom and the ROI justifies re-expressing the seams.

---

## Validation strategy (every phase)

- Full differential oracle must stay green: `jit_diff`, `jit_fuzz`, `fiber_fuzz`, `concurrent_fuzz`,
  `concurrent_escape_fuzz`, `escape_oracle`, `shared_region`, `durable_jit`, `durable_fibers_jit`,
  `dynlink`, `address_space`, `cap_self`, `fuzz_smoke`, `debug`, and the `svm-interp` unit tests.
- `fmt` + `clippy` clean; workspace builds; `#![forbid(unsafe_code)]` intact.
- Benchmark A/B on the same machine (multi-run, since the box is noisy) — record deltas here.
- Land in small, individually-green, bisectable commits (the Tier-1 slot rewrite was one big change
  and sprawled; bytecode work must not repeat that).

---

## Risks

- **Seam re-integration** (esp. debug `IrPc` mapping and fiber/durability stack switching) is the
  main source of subtle bugs — favor shape (A), and gate Phase 1 on a tree-walker-vs-bytecode
  equality harness.
- **Compile-time cost** of the per-run compile pass must stay negligible vs. execution (cache per
  run; most entry funcs run long enough to amortize — but a tiny function called once shouldn't
  regress; measure).
- **Determinism vs. the JIT** on fuel/scheduling changes (Phase 3) — treat any verified-module
  divergence as a hard stop.

---

## Phase tracker

- [x] **Phase 0** — contained in-place wins (PR #52). alu ~319 → ~66 ns (~5×).
- [~] **Phase 1** — compile pass + resolved bytecode + equality harness.
  - [x] ROI spike (`bytecode_spike.rs`): ~3.5× ALU, ~3.0× call.
  - [x] **Slice 1b** — production compiler + register-window executor (`svm-interp/src/bytecode.rs`,
        scalar + memory + direct-call subset) + equality harness (`crates/svm/tests/bytecode_diff.rs`,
        exact-equality on 4000 generated modules + kernels). Standalone `compile_and_run` path, not
        yet the default. Perf vs the tree-walker: alu 1.46×, call 1.76×, mem 1.13× (uses 16-byte
        `Reg` + per-op fuel, so below the raw-`i64` spike; slot narrowing + mem fast-path are later).
  - [x] **Slice 1c-a** — op coverage: SIMD/`v128`/fence long tail delegated to `eval_inst` (reuse,
        no re-implementation), run against each block's sub-window so no operand remap is needed.
        Harness coverage of the generated corpus rose to ~1114/4000 (28%); the rest is
        `call_indirect` / host / fiber / thread / cap programs (later slices). Still non-default.
  - [x] **Slice 1c-b** — `call_indirect` through module 0's natural function table (slot `i` ⇒ func
        `i`, power-of-two padding traps; resolved signature type-checked against the call site, a
        forged/mistyped slot is an inert `IndirectCallType` trap — same semantics as
        `dispatch_indirect`). Self-contained only (no `install`/`invoke` cross-module units — those
        need the shared `DomainTable` + scheduler, a later slice). Harness coverage rose to
        ~1770/4000 (44%), all bit-identical. Still non-default.
  - [ ] **Slice 1c** — make bytecode the default production path, with the tree-walker **demoted to
        the test-only differential oracle** (not retired — its simplicity is its value; both JIT and
        bytecode are checked against it in the test build). Decision recorded 2026-06-18: we accept a
        permanent two-interpreter maintenance cost (every future seam change lands in both) in
        exchange for a fast production interpreter. The seam-heavy work needs **new kinds of
        equality harness** (ordering / state-shape / snapshot equality, not just return-value
        equality), since fiber/scheduler/debug/durability parity is about *how* a run unfolds, not
        only its result. Decomposed into bisectable sub-slices:
    - [x] **1c-1** — reify the continuation: `bytecode::run` split into `Vm { regs, stack, cur,
          base, pc, scratch }` + `Vm::new`/`Vm::resume`. The flat analogue of the tree-walker's
          `Vec<Frame>`; holding it as data (not host-stack frames) is the prerequisite for every
          suspension seam. Behavior unchanged (existing harness green); perf-neutral (hot cursor
          kept in locals — ratios alu 1.49× / call 1.90× / mem 1.16×, in line with pre-refactor).
    - [x] **1c-2** — suspension seam: `Vm::resume` now takes an op `budget` and returns
          `Outcome::{Done, Suspended}` (trap = the `Err` arm); on `Suspended` it persists the cursor
          into `self` at the op boundary, so a later `resume` continues exactly where it paused. The
          production `run` passes an unlimited budget (the predicted branch is free — ratios alu
          1.64× / call 2.07× / mem 1.16×). New "interrupt-anywhere" harness
          (`bytecode_suspend_resume_preserves_result`): slicing the run at every op boundary
          (slice = 1/3/17) is bit-identical to running straight through, across the generated corpus.
          This is the machinery the scheduler/blocking-op/debug-stop seams drive; wiring it to an
          actual scheduler is 1c-4.
    - [x] **1c-3** — debug seam: `pc → {block, inst}` reverse map (`Program::src`) so `IrPc`,
          breakpoints, and stepping report tree-walker-identical locations. Harness
          `bytecode_debug.rs` (location trace == tree-walker `Inspector` `seek` sequence).
    - [x] **1c-4** — wire as a fast path: new `run_fast` / `run_with_host_fast` route eligible
          modules through the bytecode engine (`compile_and_run` returns `None` for any
          seam-requiring op, so eligibility is automatic) and fall back to the tree-walker `run`
          otherwise. **`run` itself is unchanged** — it stays the reference oracle the JIT and the
          bytecode engine are both diffed against (the refined strategy: tree-walker = test-only
          oracle, *kept not retired*). The umbrella `svm::run_text` now uses `run_fast`. New harness
          `run_fast_matches_run_on_generated_modules` (covers routing + fallback); full `svm` suite
          (58 binaries incl. `jit_diff`/`fiber_fuzz`/`concurrent_fuzz`/`dynlink`) green. Production
          guest execution is the JIT; the interpreter's role is oracle / escape-TCB checker, so this
          speeds the interpreter-only and differential paths without touching the oracle.
    - [ ] **1c-5** — **the seam rewrite** (decision 2026-06-18): re-express `run_inner`'s seam layer
          against the `Vm` so capability / fiber / thread / cross-module guests run on bytecode too,
          not just fall back. Driven **TDD-style** — each seam slice builds its verification harness
          *first* (the random corpus doesn't emit these ops, so we author targeted modules + the
          ordering/state-shape oracle the seam needs, then make bytecode match the tree-walker). The
          `Vm` becomes a first-class schedulable/parkable continuation alongside `VCpu`. Planned
          slices, in dependency order (refined once the seam inventory lands):
        - [x] **1c-5a** — synchronous host/capability seam. `Op::CapCall` drives the generic
              powerbox path via the *same* reusable `host.cap_dispatch_slots` the tree-walker's
              generic `CapCall` arm uses (handle i32, args/results i64 slots, results re-typed by
              `sig.results`); `host` is threaded through `Vm::resume` / `run`, and a new
              `compile_and_run_with_host` is what `run_with_host_fast` now calls. The executor/fiber
              capability variants (`Instantiator`/`Yielder`/`JIT`/`SharedRegion` op 4) are rejected by
              the compiler → tree-walker fallback. Also covers the synchronous §7 reflection ops
              `cap.self.count` / `cap.self.get` (reuse `host.self_dispatch`). New TDD harness
              `bytecode_caps.rs` (hand-authored host-fn modules: sum-args, op-selector, chained,
              in-loop, forged-handle-traps, self-count, self-get) — all bit-identical to
              `run_with_host`; `.expect(Some)` gates that bytecode actually drove it (didn't fall back).
        - [x] **1c-5b** — §12 **fibers** (`cont.new` / `cont.resume` / `suspend`), cooperative
              continuation switching. Reordered ahead of threads because it is **single-vCPU and
              inline-driven** (no M:N pool, no DPOR), so it builds directly on the 1c-2 suspend/resume
              machinery. `Outcome` gained `ContNew`/`ContResume`/`FiberSuspend`; the per-op loop
              escapes to a new `drive` loop that owns the fiber registry (`FiberState`) + resume
              `chain` (parked resumers, each with its `Vm` and the `cont.resume` result slot) and
              switches the active `Vm` — the bytecode analogue of `run_inner`'s `cont.*` arms. Fiber
              entry resolves through the natural table + `fiber_sig` (forged/mistyped → `FiberFault`);
              `run`/`compile_and_run_sliced` now share `drive` (budget unifies 1c-2 slicing). New TDD
              harness `bytecode_fibers.rs` (run-to-completion, return-status, suspend round-trip,
              multi-suspend loop, forged-resume fault, root-suspend fault) — all bit-identical to
              `run`. **Migration** (a fiber resumed on a *different* vCPU) needs the thread pool, so it
              rides on 1c-5c.
        - [x] **1c-5c** — threads (`thread.spawn`/`join` + `memory.wait`/`notify`). Key insight from
              the oracle study: concurrent oracle programs are **interleaving-invariant**, so the
              bytecode engine needs a *correct* scheduler, not DPOR/M:N replication. `drive` became a
              **cooperative single-threaded scheduler** over `VTask`s (the per-vCPU fiber world) all
              sharing one `Mem` (single-threaded ⇒ shared memory is trivially consistent;
              `fork_for_thread` confirmed the tree-walker shares the backing via `Arc`). New
              `Outcome::Thread*`/`Memory*` escape `Vm::resume` to the scheduler via `step_vcpu`; join
              parks on a child, `notify`/child-completion wakes, a stuck set advances a logical clock
              to the next `wait` deadline (else deadlock → `ThreadFault`, matching the explorer); the
              run ends when the **root** vCPU completes (trap propagates through `join`). Lowest-index
              scheduling keeps it deterministic. New TDD harness `bytecode_threads.rs` (tiny atomic=2,
              8×500 atomic counter=4000, futex handoff=987654 exercising wait/notify, forged-join
              fault) — bit-identical to `run`. **Fiber migration** (run-shared registry) is deferred:
              modules using *both* threads and fibers are compile-rejected (→ fallback) for now.
        - [x] **1c-5d** — §14 **coroutines** (`Instantiator.spawn_coroutine`/`resume` + `Yielder.yield`),
              the cooperative nesting round-trip. `spawn_coroutine` carves a confined child window via
              `Mem::nested_view(abs_base, size_log2)` (shared backing, fresh page-protection) and gives
              the child a Yielder-only powerbox; `resume` drives that child **inline** (`resume_coro`,
              like `run_inner`'s recursion) over the child's own `mem`/`host` until `CoYield`/`Done`;
              `yield` escapes as `Outcome::CoYield`. Cap authority (`resolve_instantiator` /
              `resolve_yielder`) is checked in `Vm::resume`, so a forged/ungranted handle is an inert
              `CapFault` in place; because a coroutine child holds only a Yielder, its own
              spawn/resume CapFault (no recursion needed). New TDD harness `bytecode_coroutines.rs`
              (the coroutine.rs round-trip = 1_001_329, forged-resume fault) bit-identical to
              `run_with_host`. Deferred (rare, complex, ~0 corpus): `instantiate`/`join` executor
              children, demand-paging / fault-yield (`CoFault`), and the module-spawning variants
              (ops 5/6/7). Coroutine modules are single-vCPU (no fibers/threads) by compile-rejection.
        - [x] **1c-5e** — cross-module §22 units. **Decision (post-clear):** since the tree-walker is
              oracle-only, bytecode is the real fallback when the JIT backend isn't viable, so a guest
              holding the `Jit` cap must get guest-JIT on bytecode too (no production fall-back path).
            - [x] **5e-1** — multi-module foundation + `install`/`uninstall` + cross-module
                  `call_indirect`. The engine became multi-module: a `Domain { mods, table }` (module 0
                  = primary, `mods[k≥1]` = installed units; runtime dispatch table replacing the
                  compile-time natural table). `Vm` activations carry a `module`, re-bound only at
                  cross-module call/return so the per-op hot loop is unchanged. `compile`/
                  `compile_linked` (JIT ops 0/5) ride the generic `cap_dispatch_slots` (free);
                  `install`/`uninstall` (ops 3/4) escape to `drive` (owns the mutable `Domain`):
                  install compiles the unit to bytecode + fills a padding slot, uninstall clears one.
                  Coroutine children keep their own natural table (no installed units), matching the
                  tree-walker. New harness `bytecode_dynlink.rs` (install→call_indirect = 142;
                  uninstall→call_indirect traps `IndirectCallType`) bit-identical to `run_with_host`.
                  **Known gap:** a unit using an op the bytecode engine can't lower traps `Malformed`
                  (no mid-run fall-back) — same coverage edge as a top-level module.
            - [x] **5e-2** — `Jit.invoke` (op 1): `run_invoke` runs the unit's entry synchronously as
                  a transient module over the shared window/powerbox + shared dispatch table (so the
                  unit's `call_indirect` reaches installed units), concurrency-free (park/spawn/yield/
                  re-install → inert `CapFault`, matching the tree-walker); args/results marshal via the
                  i64-slot ABI. New harness case `invoke_unit_that_calls_installed_unit_agrees`
                  (install A, invoke B where B calls A → 14, the §22 new→new path) bit-identical to
                  `run_with_host`. `run_fast` now routes install/invoke guests to bytecode.
            - [x] **5e-3** — tail calls (`return_call`/`return_call_indirect`): reuse the current
                  activation window (no stack growth, O(1) deep tail recursion), staying in-module for
                  direct / dispatching the runtime table for indirect. New harness
                  `bytecode_tailcall.rs` (factorial accumulator, 100k-deep recursion, indirect with a
                  type-mismatch trap) bit-identical to `run`. The generator *does* emit tail calls, so
                  corpus coverage rose to **3978/4000 (99.45%)** (the rest is the deferred
                  `instantiate`/`join` executor children, `gc.roots`, `call.import`, demand coroutines).
        - [x] **1c-5f** — fiber **migration**: the fiber registry moved out of `VTask` into a
              **run-shared** `Vec<FiberState>` owned by `drive` (one domain-wide handle namespace),
              passed to `step_vcpu`; only the resume `chain` stays per-vCPU. A fiber created/suspended
              on one vCPU is now claimable on another (cooperative ⇒ claim is trivially exclusive;
              claiming a fiber Running in another vCPU's chain is `FiberFault`, matching the
              tree-walker). Lifts the thread+fiber compile rejection. Harness: the `MIGRATE` pattern
              (fiber suspended on root, resumed on a spawned thread → 75) bit-identical to `run`.
        - [x] **1c-5g** — §14 **executor children** (`Instantiator.instantiate` / `join`, ops 0/1):
              a child runs on the cooperative scheduler (unlike an inline coroutine), confined to a
              power-of-two sub-window (`nested_view` over the **shared** backing) with an attenuated
              powerbox (an `Instantiator` + an `AddressSpace`, each over `[0, child_size)`) and a
              `quota` fuel sub-budget. Each scheduler task now carries an `env: Option<usize>`: `None`
              = the shared domain (root + its `thread.spawn` siblings); `Some(k)` = a confined
              `ChildEnv { mem, host, table, fuel }` (a fresh **natural** dispatch table — no installed
              §22 units, like the tree-walker's `DomainTable::new(&cfuncs, 0)`). `step_vcpu` takes a
              bundled `RunCtx { table, fuel, mem, host }` selected per task, so the per-op hot loop is
              untouched. `instantiate` validates the entry sig + carve in `drive` (the task-set owner),
              builds the child, and registers it in the spawner's `threads` namespace; `join` reuses
              the §12 thread-join machinery (`InstJoin` checks the cap authority, then emits
              `Outcome::ThreadJoin`). A thread spawned by a confined child inherits its `env` (shares
              its window). `compile_module` reclassifies ops 0/1 as scheduler-driven (not coroutine),
              so instantiate now composes with threads/coroutines; only instantiate+`cont.*` fibers is
              still rejected (the run-shared fiber registry would leak across the child domain → tree
              walker fall-back). New harness `bytecode_instantiate.rs` (shared-backing round-trip →
              42123, depth-2 nesting → 77, two-arg child driving its own `AddressSpace.unmap` → 0,
              out-of-range carve → −EINVAL, child trap propagating through `join`) bit-identical to
              `run_with_host`. Still deferred: the separate-**module** coroutine variants (ops 6/7)
              and demand/fault-yield coroutines (op 4).
        - [x] **1c-5h** — §14 separate-**module** executor child (`Instantiator.instantiate_module`,
              op 5): the "plugin-in-plugin" story. The host grants the parent a `Module` capability
              (iface 8); op 5 takes that handle as its first arg and spawns a confined child running
              **that** verified module (not the holder's program). The driver resolves the grant,
              `compile_module`s it (a module using an op the engine can't lower is a `Malformed` trap —
              the one place a guest program outruns coverage, no fallback mid-run, as for
              `Jit.install`), pushes it into `dom.mods`, and runs the child over a natural table mapping
              into *its* module index (`build_table_for`). The carve must **equal** the module's
              declared memory (§14 transparency), and the module's **data segments materialize** into
              the carve at spawn (written through the shared backing). Reuses op 1 `join` unchanged.
              New harness `bytecode_separate_module.rs` (a foreign 64 KiB module with a `"VM"` data
              segment → 1086, marker visible to the parent → 1_086_000_007, carve≠declared-memory →
              −EINVAL, forged module handle → CapFault) bit-identical to `run_with_host`. Still
              deferred: the module **coroutine** variants (ops 6/7) and demand/fault-yield (op 4).
        - [x] **1c-5i** — §14 separate-**module** *coroutine* (`Instantiator.spawn_coroutine_module`,
              op 6): the inline-coroutine analogue of op 5. The spawn escapes to the driver (it must
              compile + push the granted module into `dom.mods`), which builds a `Coro` over it and
              registers it in the spawner's coroutine set; thereafter it is `resume`d **inline** like
              any coroutine. `Coro` gained a `table` field (its natural dispatch table — module 0 for
              op 2, its own pushed index for op 6; the `vm.module` selects the program), so
              `resume_coro` no longer hard-codes module 0. Data segments materialize into the carve and
              the carve must equal the module's declared memory, as for op 5. New case in
              `bytecode_separate_module.rs` (a foreign coroutine module yielding 100 / 210 then
              returning 1019 → 1_001_329) bit-identical to `run_with_host`. Still deferred: demand
              variants (ops 4/7, fault-yield paging).
        - [x] **1c-5j** — §14 **demand (fault-driven-yield) coroutines** (`spawn_demand_coroutine`
              op 4, `spawn_demand_coroutine_module` op 7): completes §14 (ops 0–7). A demand child
              starts with its whole window **unmapped** (`Mem::demand_page`), so an in-window access to
              an unsupplied page is a *recoverable* fault that suspends to the parent (status `FAULTED`
              = 2, value = the fault address) instead of trapping; the parent supplies the page
              (`Mem::supply_page`, keeping the bytes) and resumes, and the child's rewound access
              re-executes and reads it (the userfaultfd-style lazy-paging model). **The "rewind the
              faulting op" needs no hot-loop change**: a demand coroutine is stepped one op at a time
              (`budget = 1`) in `resume_coro`, so the budget boundary persists the cursor *at* the next
              op before running it — when that op faults, the cursor already points at it, so the next
              `resume` (after the parent supplies the page) retries exactly that access (the access
              checks protection before any effect, so re-running is side-effect-clean). `Coro` gained
              `fault_yields` / `faulted_page`; `CoStop` gained `Fault`; the `resume` op supplies the
              page (not delivering a yield value) when `faulted_page` is set. New harness
              `bytecode_demand_coroutine.rs` (op-4 fault→supply→read round-trip → 2_001_123, fault
              address → 65536, op-7 lazy module data supply → 2_101_086) bit-identical to
              `run_with_host`. **§14 is now fully covered on the bytecode engine.**
        - [x] **1c-3** — debug seam: a `pc → {block, inst}` reverse map (`Program::src`) so the engine
              reports tree-walker-identical [`IrPc`] locations for stepping/breakpoints. Built at
              compile time parallel to the op stream: each instruction op carries its `(block, inst)`;
              the one terminator op per block is `None` (non-steppable — the tree-walker's `before_op`
              stops only at instructions, never terminators, and its logical clock ticks once per
              instruction). `Vm::cur_ir_pc` reads it; `bytecode::ir_trace` single-steps (`budget = 1`,
              one op per `resume`) recording each instruction location. New harness `bytecode_debug.rs`
              asserts the bytecode location trace is **identical** to driving the tree-walker
              `Inspector` with `seek(0), seek(1), …` (which enumerates executed-instruction locations) —
              across straight line, branches, loops (back-edges revisit `IrPc`s), cross-frame calls,
              and a trap — plus result equality. (Full Inspector parity — watchpoints, backtrace,
              cap-call stops, time-travel — is follow-on; this lands the location model.)
    - [ ] **1c-6** — durability seam: capture/restore a `Vm` across a coroutine yield. New harness:
          snapshot equality.
- [~] **Phase 2** — memory-op specialization + software fast-path.
  - [x] Lock-free `check_prot` fast path (`prot_dirty` flag) + `read_le`/`write_le` `has_regions`
        hoist. Tree-walker memory kernel ~176 → ~147 ns (~17%); all oracle suites byte-identical.
- [ ] **Phase 3** — per-op seam overhead (fuel-at-back-edges if provably safe; debug/preempt hoist).
- [ ] **Phase 4** — (stretch) fully flat bytecode + threaded dispatch.
