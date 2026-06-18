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
- [ ] **Phase 1** — compile pass + per-block resolved bytecode (shape A) + equality harness.
- [ ] **Phase 2** — memory-op specialization + software fast-path.
- [ ] **Phase 3** — per-op seam overhead (fuel-at-back-edges if provably safe; debug/preempt hoist).
- [ ] **Phase 4** — (stretch) fully flat bytecode + threaded dispatch.
