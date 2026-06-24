# Debugging & observability — work scoping

Status: **scoping draft**, written 2026-06-15. Branch: `claude/charming-johnson-pmlsnr`.

This document is the **work-breakdown and detailed design** for the debugging effort. The
**rationale and decisions** live in `DESIGN.md` §19 (pillars), §12/§23 (concurrency), §5/§3d
(the two-stack split), §15 (metering/`Monitor`), §18 (the model checker), §2a (debug info is
untrusted-for-escape). This doc does not restate those; it scopes the *work* each pillar
implies — current substrate, design sketch, API surface, dependencies, open questions, effort,
and acceptance — and proposes a build sequence. Keep §19 as the canonical "why"; keep this as
the canonical "what/how/when."

---

## 0. Framing

§19 names four pillars. The 2026-06 reassessment (this branch) established that their
*architectural premises are built and cross-platform-validated* — the out-of-band control
stack + per-fiber two-stack split (§5/§3d, `svm-fiber`), the deterministic interpreter oracle
(§12/§18, `run_scheduled`/`explore_all`), capabilities (§3c/§7), and SSA promotion (§3d) — but
the *debug surfaces themselves* are not. So this is not green-field design; it is wiring known
surfaces onto substrate that already exists and is tested.

Design invariants every workstream inherits (do not relitigate; see §19/§2a):

- **Debugger = a host-side capability** (an `Inspector`/`Debugger`, shaped like §15 `Monitor`):
  it *observes* from outside, never widens the guest's authority.
- **Debug info is tooling, untrusted for escape** (§2a): strippable; the verifier never trusts
  it; corrupt/malicious debug info can degrade the debugging experience but **cannot** break
  confinement.
- **The interpreter is the debug engine; the JIT is the production engine.** The interpreter
  owns the scheduler (M:N green threads on one OS thread), which is what makes deterministic,
  controllable multithread debugging possible at all. Surfaces land on the interpreter first;
  the JIT path is differential-checked against it.

---

## 1. Current-state snapshot

| Capability | State | Where |
|---|---|---|
| Out-of-band per-fiber control stack (CFI; backtrace *integrity*) | **Built** | §5/§3d, `svm-fiber`, `fiber_rt` |
| Deterministic scheduled replay (seed) | **Built** | `svm-interp` `run_scheduled` |
| Exhaustive DPOR model checker (all interleavings) | **Built** | `svm-interp` `explore_all` (+ `_bruteforce` oracle) |
| Interp↔JIT differential testing of concurrency | **Built** | `jit_fuzz.rs`, `concurrent_fuzz.rs`, `fiber_fuzz.rs` |
| SSA promotion (the inspectability-tension source) | **Built** | §3d, frontend promote pass |
| Fuel/quota metering *properties* | **Built** | `Host::set_quota`/`quota`, §15 |
| `cap.call` I/O record log (`CapTape`) — input caps `Clock` + stdin `read` + **any host-fn** (slots **and** buffer writes); replayed for faithful `seek` | **Built — W1 slices 2, 5** | `svm-interp` `Host::record_caps` / `CapTape` / `RecordingMem` |
| Schedule record log (`SchedTape`) — capture a live interleaving as a replayable plan; seeded schedule fuzzing | **Built — W1 slice 4** (interp; SC ⇒ schedule *is* memory order) | `svm-interp` `Inspector::sched_tape` / `attach_scheduled_seeded` |
| W7 model-check → replayable witness (find a failing interleaving, reproduce it) | **Built — slice 1** | `svm-interp` `find_schedule` / `replay_schedule` / `Witness` |
| W1 time-travel — `seek(t)` / `step_back`: single-threaded (op `clock`) **and** multithreaded (global `turn`); faithful via `CapTape` | **Built — slices 1–3** | `svm-interp` `Inspector::seek` / `turn` / `step_back` |
| Interpreter stepping / breakpoint / watchpoint / cap.call stop / backtrace / value+window read | **Built — slices 1–3** | `svm-interp` `Inspector` (single-threaded) |
| Multithreaded debugging — fixed-schedule `thread.spawn` guest, per-thread breakpoints, replay a failing interleaving, inspect any thread (`select_task`), time-travel to a global turn | **Built — Milestone B slices 1–3** | `svm-interp` `Inspector::attach_scheduled` / `SchedDriver` |
| Source-level debugging — chibicc `-g` → `debug.var` + `debug.loc` → named locals & `file:line` (interpreter `source_loc` nearest-preceding) | **Built — W4 slices 5–6** | `codegen_ir.c`, `svm-text`, `svm-interp` |
| Backtrace *materialization* (unwind tables → frames) | **Built** — gdb-facing DWARF CFI (`.debug_frame`) + a host-side fiber-rooted walk of a *suspended* fiber (W5 JIT/DWARF Stage 4), **and** the always-on **trap-time** backtrace: a JIT trap (memory fault *or* explicit check) symbolizes its stack into `last_trap_backtrace()`, folded into the host kill message (W3 Stages 0–3, unix) | §5, `svm-jit` `trap_backtrace`/`fiber_backtrace`/`dwarf` |
| Debug-info ABI (frontend-neutral IR waist; source locs + var locs + structured types) | **Built — neutral core + structured `TypeRef` table (text **and** binary); chibicc `-g` emits the full waist; **wasm ingests embedded DWARF (source lines + vars + aggregate/pointer/array types)**; **LLVM ingests `!DILocation` source lines + (`-O0`) `dbg.declare` variables/types via `llvm-sys`** — three independent producers on both halves; DAP consumes types** (D-DBG-7/§6) | `svm-ir` `DebugInfo`/`TypeDef`, `svm-text`, `svm-encode`, `svm-interp`, `codegen_ir.c`, `svm-wasm`, `svm-llvm` |
| DAP server (interpreter-backed: source breakpoints + **conditions**, **data breakpoints** (watchpoints, incl. cross-thread), frames, locals, **source-line** stepping (in/over/out), **reverse debugging** (single + multithreaded), **multithreaded** per-thread stacks, **`evaluate`** expressions/hover incl. **member/index/arrow** (`a.b`, `arr[i]`, `p->x`), **Variables-pane struct/array/pointer expansion**) | **Built — W5 slices 1–6 + W4 slices 8, 10, 11** | `svm-dap` (`DapServer` / `expr` / `run_stdio`) |
| DWARF emission (gdb/lldb on JIT native code) | **Built — W5 JIT/DWARF tier, Stages 0–4** — source-line breakpoints, `print` of register **and** spilled variables, `bt` across guest frames, type DIEs, GDB JIT registration, and a fiber-rooted backtrace; all confirmed under gdb 15.1 (Stage 5 DAP-over-JIT + guest-window-memory var forms deferred) | `svm-jit` `dwarf`/`gdb`/`symbolize`/`var_locations` |
| `Inspector`/`Monitor` capability *type* | **Missing** (pattern only) | — |
| DRF-or-trap hardened race-detection tier | **Missing** (designed, §12) | — |

---

## 1b. Cross-engine debug parity

The project has **three execution engines** but only **two debug modalities**, so "parity" means
different things depending on which pair you compare:

- **Tree-walker** (`svm-interp`, the default `run`/`Inspector` path) — the **reference debug engine and
  correctness oracle**. The full `Inspector` surface (breakpoints + conditions, watchpoints, in/over/out
  + reverse stepping, time-travel, multithreaded, the whole DAP server) runs here. When you debug *via
  DAP*, you are always on the tree-walker.
- **Bytecode** (`svm-interp::bytecode`, the perf rewrite reached by `run_with_host_fast`) — preserves the
  `IrPc = (block, inst)` debug seam *by construction* (`Program::src` reverse map). `bytecode::ir_trace`
  reproduces the tree-walker's `seek(0,1,…)` stepping-location sequence op-for-op, but only for
  **single-vCPU, seam-free** runs; it **falls back to the tree-walker** for watchpoints, real
  breakpoint-stops, concurrency/coroutines, and time-travel.
- **JIT** (`svm-jit`) — a *different modality*: it emits **DWARF** for **gdb/lldb** (W5 Stages 0–4) and an
  always-on trap-time source backtrace; it does **not** use the `Inspector`, and DAP-over-JIT (Stage 5)
  is deferred. The unifying mechanism that keeps all three agreeing on *where in the source* a program
  point is, is the §6 debug-info **narrow waist** (W4): every engine consumes the *same* neutral
  `DebugInfo` (`debug.loc`/`debug.var`/types).

**What is already protected by tests:**
- *Bytecode ↔ tree-walker* — `bytecode_debug.rs` (stepping/breakpoint **locations** op-for-op:
  straight-line/branch/loop/call/trap, results too) + `bytecode_diff.rs` and ~15 per-feature
  `bytecode_*` suites (value/trap/memory exact-equality, tree-walker as oracle).
- *JIT debug emission* — `jit_srcloc.rs` (`debug.loc` → `symbolize` → DWARF `.debug_line` round-trip),
  `jit_trap_backtrace.rs`/`interp_trap_backtrace.rs`/`jit_per_fiber_trap.rs` (source backtraces),
  in-crate `symbolize` tests, the `gdb_attach` example (gdb 15.1).
- *Interp ↔ JIT value/trap parity* — a large cross-engine suite (`simd`, `fiber_fuzz`, `jit_*`,
  `multivcpu_trap_origin`, `dynlink`, plus svm-llvm `cross_engine`/`corpus_diff`). This is **result**
  parity, not **debug** parity.

### Known gaps (the regression risks this section tracks)

- **G1 — direct cross-engine *source-location* parity assertion. ✅ Landed (`crates/svm/tests/debug_parity.rs`).**
  Compiles one `-g` module and checks the tree-walker's `source_loc`, the bytecode engine's `ir_trace`
  location, **and** the JIT's `src_ranges`/`symbolize` agree on the same op→line mapping (straight-line,
  loop with repeated lines, branch). The two interpreters must match **op-for-op**; the JIT's line set
  matches exactly (full-coverage fixtures) or is a superset (a branch's not-taken arm stays mapped).
  A drift in any single engine's debug-info threading now fails here. One **legitimate** divergence is
  pinned (`jit_elides_const_only_source_line`): a line that belongs only to a folded single-use `const`
  is stepped by both interpreters but has no JIT machine-code range — compiled code has no instruction
  for a materialized immediate — while the invariant "the JIT never maps a line the interpreters don't
  step" still holds.
- **G2 — bytecode variable-value parity + the delegation boundary. ✅ Landed (`debug_parity.rs` +
  `bytecode::ir_window_trace`/`ir_value_trace`).** A source variable holds the **same value on both
  engines at every step**, proven through the real debugger APIs:
  - *SSA-located* — `compile_func` gives each value a **stable, unique slot** (a "global slot per value";
    *no* register reuse/coalescing), so a promoted scalar is directly inspectable on the bytecode engine
    (`regs[base + i]` typed by `func_value_types` — the same storage the tree-walker's `read_ir_value`
    reads). `ssa_var_value_parity_per_step` compares the per-step block-local value vectors op-for-op and
    checks `read_var(a)`/`read_var(b)` against the bytecode slots. The bytecode tier is therefore **fully
    inspectable, not precluded** (the earlier "precluded by design" framing was wrong — it is unbuilt as
    a DAP backend, not blocked).
  - *Window-located* — both engines drive the one `Mem`, so `window_var_value_parity_per_step` reads `x`
    via `var_addr`/`read_var` at each `seek(t)` and matches the bytecode engine's per-step window snapshot.
  - *Delegation boundary* — `bytecode_debug_trace_declines_outside_single_vcpu_scope` pins that the
    single-vCPU debug-trace returns `None` for a thread-spawner (so a debugger falls back to the
    tree-walker / Milestone-B scheduled Inspector); a later flip to `Some` is a boundary change to notice.
  - *Residual:* watchpoints/conditional breakpoints/time-travel on the bytecode path stay delegated (no
    independent implementation to regress); the per-step SSA reader is single-block-scoped (slot index ==
    block-local index there — multi-block adds the per-block slot base).
- **G3 — DAP-level cross-engine parity. ⏳ Foundation landed; full backend still feature work.** The DAP
  server drives the `Inspector` (tree-walker) and nothing else, so the *full* gap — a DAP conversation
  replayed against a second backend — needs that backend *built*, not just tested: **DAP-over-JIT** is W5
  Stage 5 (unbuilt; open design fork — drive the JIT under DAP via `int3`/single-step, *or*
  interpreter-steps-JIT-runs), and **DAP-over-bytecode** needs the bytecode debug primitive wired into
  the DAP server. The first prerequisite of the bytecode path **landed**: `bytecode::DebugRun` — a
  resumable debug session (`run_to` a breakpoint, stopping *before* the op like the tree-walker; `value`
  reads a block-local SSA value at the stop) — the engine-level control+inspection a DAP-over-bytecode
  backend wires into. Two runtime parity tests prove the half G1/G2's one-shot traces don't:
  `breakpoint_runtime_parity_across_loop_iterations` (the tree-walker `Inspector` and `DebugRun`, driven
  through the same loop-body breakpoint, report identical stop locations and inspected `(i, acc)` at
  **every hit** — resume included — and the same result) and `breakpoint_runtime_parity_across_call_frames`
  (stopped *inside a callee*, both report the same **call-stack depth, per-frame location, and per-frame
  locals** — the `stackTrace`/`scopes`/`variables` surface, now matched cross-frame via `DebugRun`'s
  per-function slot metadata). `DebugRun` also has the **stepping verbs** (`step`/`step_over`/`step_out`,
  mirroring the tree-walker's `step_to_depth`); `stepping_parity_over_and_out_at_a_call` checks that
  step-over (run a call to completion) and step-out (return from a frame) land at the same op and agree
  on the call result. So the bytecode engine now has the full **forward-debug** primitive surface —
  breakpoints, cross-frame inspection, backtrace, stepping — all parity-tested against the tree-walker.
  *Remaining, and scope-bounded:* the `svm-dap` server calls 18 distinct `Inspector` methods;
  `DebugRun` covers the forward-debug subset, but the rest — `seek`/`step_back` (reverse debugging),
  `set_watchpoint` (data breakpoints), `select_task`/`threads`/`stopped_task` (multithreading) — are
  **outside the bytecode engine's single-vCPU debug scope** (it delegates those to the tree-walker, per
  G2). So a DAP-over-bytecode backend can only ever cover the *forward-debug subset*; wiring it in is a
  server-side backend seam (abstract that subset, add a name→`VarLoc` variable read on `DebugRun`) whose
  *correctness* is already guaranteed by the engine-level parity here — it is feature plumbing for users,
  not a parity risk. The JIT path is Stage 5 (separate). The *static* source-map half is covered
  transitively by **G1**.

---

## 2. Workstreams

Eight workstreams (W1–W8). Dependency graph (→ = "depends on"):

```
W8 Inspector/Monitor host capability  ─┬─→ used by W2, W1, W3
W2 Interpreter step/break/watch       ─┼─→ W1 (replay drives stepping)
W1 Record/replay (cap + schedule log) ─┘
W3 Backtrace materialization          ── (independent; needs Cranelift unwind)
W4 §3a IR debug-info side-table       ──→ W5
W5 DWARF + DAP                         (← W4, W3 unwind, debug-build mode)
W6 Debug-build mode (promotion off / value-locations)  ──→ W5
W7 Concurrency-debug surfacing (explore_all UX) ── (← W2 optional)
```

The graph above is **functional** coupling ("A needs B to run"); honor it by ordering. The more
dangerous coupling is **design-time** — shared representations (§2a) — which ordering does *not*
solve. Recommended order is in §11. Each workstream below follows the same shape.

---

## 2a. Cross-workstream coupling — the shared "debug core"

The §2 graph captures *functional* dependency (A needs B to work). It misses *design-time*
coupling: places where two workstreams independently touch the **same representation**, so
freezing one's version first forces the other to rework. Ordering cannot fix this; only deciding
the representation **once, up front** can. Six such representations form a small shared **debug
core** — each is a half-page data-model decision even though the implementations behind them are
large and staged. That asymmetry is the argument: co-design the *vocabulary* first, iterate the
*bodies* against it.

| # | Shared representation | Consumed by | Rework if designed per-workstream |
|---|---|---|---|
| **S1** | **Location model** — naming "where in the program" (IR-PC + granularity: per-op vs per-statement) | W2, W3, W4, W5 | W2 picks an ad-hoc PC; W4 needs finer granularity → all breakpoint/frame addresses change |
| **S2** | **Value-location model (`VarLoc`)** — where source var X lives at PC P (window slot / SSA value / promoted) | W2, W4, W5, W6 | W2 builds window-slot-only "read local" → can't express promoted SSA values → inspect API reworked |
| **S3** | **Logical-time / position clock** — the monotonic coordinate `seek` targets | W1, W2, W7 | W1 uses cap-call count, W2 needs op count → no shared seek; time-travel + step-back don't compose |
| **S4** | **Interpreter instrumentation seam** — the per-step / per-memop hook in the hot loop | W1, W2, W3, W7 | each bolts a parallel loop variant → conflicting hot paths, untestable. **The biggest pinch** |
| **S5** | **Inspector control/session model** — stop-the-world vs observe-running vs many-runs | W1, W2, W3, W7 (home: W8) | W8 shaped only for synchronous stepping → W7 "explore many" + W1 "replay tape" don't fit → W8 reshaped |
| **S6** | **Cranelift debug-emission layer** — enabling unwind/debug info in the JIT | W3, W5 | two Cranelift config paths for the same emission → duplication + drift |

Plus two cross-cutting **invariants** (constraints, not representations) every workstream
inherits:

- **S7 — observe-only / behavior-preserving + strippable.** Interpreter hooks (W1/W2/W3/W7)
  must not perturb scheduling or values: the **interp↔JIT differential** (the core testing
  discipline) must hold with instrumentation off-path, and a debugger that changes the schedule
  **hides the heisenbug**. Sanctioned exception: W2 *driving* a chosen deterministic schedule via
  `run_scheduled` (control, not perturbation). All debug artifacts (W1 tape, W4 section, W5
  DWARF) are strippable and untrusted-for-escape (§2a).
- **S8 — metering-pause semantics.** Stopping at a breakpoint (W2) collides with §5's
  **undisableable** fuel/epoch preemption (a runaway guest must always die). A guest stopped at a
  breakpoint must not be fuel-killed, without reopening the runaway hole → a "metering paused
  while stopped" state W8/W2 must define against §15/§5 (see D-DBG-6).

**Genuinely separable** (iterate freely, low coupling): **W7 surfacing** (the functions exist),
**W1 sequential `CapTape`** (couples to the cap ABI, not other debug work), **W6** (decoupled
from the interpreter path — the interpreter holds SSA values explicitly, so Milestone A needs no
debug-build mode), and the **DRF-or-trap tier** (standalone).

**Conclusion.** The workstreams are *not* cleanly separable, but full co-design is unnecessary.
Fix S1–S6 and decide S7/S8 in a thin **debug-core design pass** (Milestone 0, §11) before writing
W2/W4; then the workstream bodies iterate independently against the frozen core.

---

## 3. W1 — Record/replay & time-travel (the multithreaded centerpiece)

**Pillar 1.** Goal: capture a guest run as a compact, deterministically **replayable** trace,
so a failure — including a multithreaded one — can be re-run identically and stepped backward.

**Current substrate.** Two halves of replay already exist and are tested:
- *Capability boundary.* No ambient authority (§7): in single-vCPU/deterministic mode all
  nondeterminism enters through `cap.call`. Logging those inputs/outputs is the whole recording
  surface for the sequential case — the boundary already exists; only the log does not.
- *Schedule + memory order.* For true multicore, race outcomes bypass the cap boundary. But the
  DPOR explorer **already reifies exactly the choices replay needs** — `explore_all` runs each
  schedule from "a planned sequence of scheduling choices" at memory-op granularity, recording a
  per-step `MemAccess`. `run_scheduled(seed)` is already a deterministic, reproducible single
  schedule. So the schedule-recording machinery is built; it is not yet *exposed as a record/log
  artifact* a host can capture from a live run and feed back.

**Design sketch.**
1. **`CapTape`** — an append-only log of `(handle, iface, op, args-bytes, result-bytes,
   logical-time)` records, written by the `cap.call` trampoline behind a host flag. Buffer args
   are borrow-only `(ptr,len)` today (D42); the tape snapshots the *bytes that crossed*, in both
   directions, so replay needs no live host. Strictly host-side ⇒ untrusted-for-escape (§2a).
2. **`SchedTape`** — for concurrent runs, the ordered sequence of scheduling decisions at each
   visible op (the `plan` vector `explore_all`/`run_scheduled` already consume), plus the
   memory-order resolution of each racing access. For the interpreter this is a direct dump of
   the explorer's `trace`; for the JIT it requires interposing on visible ops (the expensive
   part — see open questions).
3. **Replay** = re-run the interpreter feeding `CapTape` for cap results and `SchedTape` for
   scheduling, asserting the live trace matches. Time-travel (step backward) = replay from the
   nearest checkpoint to `t-1` (stateless re-execution, as `explore_all` already does; optional
   periodic snapshots bound the cost).

**API surface (host).** `Host::record(tape: &mut CapTape)` / `Host::replay(tape: &CapTape,
sched: Option<&SchedTape>)`; an `Inspector` (W8) exposes `seek(logical_time)` for time-travel.

**Dependencies.** Sequential replay: none beyond the tape. Multicore JIT replay: W8 (capture
surface) and JIT visible-op interposition. Time-travel UX: W2 (stepping) and W8.

**Open questions.**
- *JIT schedule capture cost.* Recording every visible memory op on the real-thread JIT is
  TSan-class overhead. Options: (a) record only on the interpreter and treat the JIT as
  verify-only (cheapest, matches "debug on the interpreter"); (b) a coarse logical-clock /
  vector-clock record at sync ops only (sound for DRF programs, lossy under races); (c) gate
  full capture behind the §12 DRF-or-trap hardened tier (W-future). **Recommend (a)** for v1.
- *Tape size / checkpoint cadence* for long REPL-style runs (interacts with `JitSession`
  compaction, §22).
- *Non-determinism leaks to audit*: clock reads, RNG, address-space layout, uninitialized
  reads. Deterministic mode (§12/D27) already enumerates the scrub list; reuse it.

**Effort / risk.** Sequential `CapTape` + interpreter replay: **moderate, low risk** — the
boundary and the deterministic engine both exist. Multicore JIT replay: **high, real risk** —
this is where the cost lives; keep it out of v1.

**Acceptance.** A recorded sequential run replays to a byte-identical cap-I/O trace and
identical final window; a recorded interpreter-scheduled concurrent run replays to the same
outcome; `seek(t)` returns the guest state at logical time `t`.

**Built — slice 1 (time-travel via stateless re-execution).** `Inspector::seek(t)` re-executes a
single-threaded run from `clock 0` to logical time `t` and pauses there, restoring the exact frame
state (pc + live SSA values) so `backtrace`/`read_*` show the guest as it was; `step_back()` is
`seek(clock-1)`. Implemented with the §18 explorer's own trick — the attach inputs (funcs, func,
args, fuel, memory, data) are kept as cheap-to-clone `Arc`s in a `SeekInit`, and a `seek_target`
on `DebugCtx` fast-forwards a fresh re-run past breakpoints to `t`. This is **exact for a
deterministic guest** (the common algorithmic-debugging case); the re-run uses a fresh empty
powerbox, so a guest whose `cap.call`s carry real side effects or nondeterminism needs the
`CapTape` (next slice) to seek faithfully. Tests (`debug.rs`): out-of-order `seek` restores the
exact frame state recorded while stepping forward; `step_back` decrements the clock; `seek(0)` then
resume reproduces the result; seek-past-end finishes.

**Built — slice 2 (`CapTape`: record/replay the nondeterministic cap inputs).** A run now tapes the
capability **inputs** crossing into the guest (`Inspector::cap_tape() -> CapTape` of `CapRecord`s),
and `seek` re-executes against a fresh powerbox seeded to **replay** that tape — so time-travel is
faithful even when the guest's result depends on a host input a fresh re-run couldn't reproduce.
The hook is the single `cap_dispatch_slots` chokepoint: it records/serves only the *nondeterministic
input* caps (`is_recorded_input`: `Clock` op 0, `Stream` op 0 = stdin `read`, and **any host-fn**
`iface::HOST_FN` — the embedder's escape hatch for RNG / a real clock / external I/O, whose closure
is *gone* on the fresh replay powerbox, so only the tape can reproduce it), leaving deterministic /
structural caps (`Memory` ops, `SharedRegion`, `Stream` *write*) to re-run live on the fresh powerbox
— which reproduces them exactly, so they need no tape. Both directions cross: a
`CapRecord` keeps the result slots **and** the bytes a buffer-filling input wrote into the guest
window (captured via a `RecordingMem` `GuestMem` wrapper that logs `write_bytes`), re-applied on
replay. Replay verifies each served crossing matches the live `(type_id, op, handle, args)`
(divergence detection). Tests (`debug.rs`): a guest summing two `Clock` reads (host clock seeded to
1000) replays to `2001` after `seek(0)` (vs `1` on a fresh clock); a guest reading 2 stdin bytes
into its buffer and returning `buf[0]` replays to `'H'` (72) — the captured buffer write re-applied
— vs `0` on a fresh empty-stdin host; and a guest summing two reads of a stateful host-fn (an
incrementing counter) replays to `201` even though the closure no longer exists on the fresh
powerbox — so only the tape could have carried it.

**Built — slice 3 (scheduled-mode seek: multithreaded time-travel).** `seek(t)` now also time-travels
a **multithreaded** (`attach_scheduled`) run — the W1↔Milestone-B unification. The coordinate is the
**global scheduler turn** (`Inspector::turn()`), one per visible-op decision across all threads — the
plan index, the only coordinate that names a whole-program instant (per-vCPU clocks diverge). `seek(t)`
rebuilds the run and replays the fixed plan for exactly `t` turns, landing at a **global snapshot**:
no thread is "stopped", but every thread is inspectable via `threads()`/`select_task()`/`backtrace`
(the focus defaults to the thread that ran turn `t`). Mechanism: the re-entrant `SchedDriver` gained a
`turn_limit` (stop at the turn boundary — no held vCPU), and a `suppress_stops` flag on `DebugShared`
fast-forwards past breakpoints during the replay; `step_back` decrements `turn()` (vs the op `clock`
single-threaded). Because the plan pins the interleaving and the `CapTape` replays the inputs, the
snapshot at turn `t` — including whatever the guest's own userland scheduler had done by then — is
exact and reproducible. Test (`debug_threads.rs`): a witness-pinned racy run seeks to a mid-turn
global snapshot (reproducible across repeats), `seek(0)` + resume reproduces the raced outcome, and
`step_back` walks the global turn down by one.

**Built — slice 4 (`SchedTape`: capture a live interleaving + schedule fuzzing).** The schedule a
scheduled run actually executed is now a first-class artifact: `Inspector::sched_tape() -> Vec<u64>`
returns the ordered `TaskId` choice at each visible-op decision (a direct dump of the explorer's
`trace`). Under sequential consistency the schedule *is* the memory order, so this fully pins the
run — `attach_scheduled(tape)` replays the exact interleaving deterministically, making any run a
portable, shareable repro. To make capture worthwhile, `attach_scheduled_seeded(seed)` drives a
**random** fine-grained interleaving (one random runnable thread per turn) — schedule fuzzing — so
different seeds explore different interleavings (and surface different race outcomes); a found
failure's `sched_tape` replays it. The randomization is a seed on `Dpor`'s schedule-*extension*
(past the plan); the explorer leaves it unset so its DPOR backtracking stays deterministic, and the
choices still land in `trace`, so a seeded run replays from either its seed (`seek`) or its captured
tape. Tests (`debug_threads.rs`): across 64 seeds fuzzing surfaces both the lost update (1) and the
correct total (2), each run's `sched_tape` replays to the identical outcome and interleaving, and
`seek(0)` reproduces a seeded run.

**Built — slice 5 (host-fn input caps).** `is_recorded_input` now also tapes `iface::HOST_FN`, the
embedder's general escape hatch — so RNG, a real wall clock, or external I/O exposed as a host-fn
records/replays like `Clock`/stdin (slots + any guest-window writes via `RecordingMem`). This is the
sharpest demonstration of why a tape is needed at all: the host-fn *closure* is not present on the
fresh replay powerbox, so re-execution **cannot** reproduce it any other way (test: a stateful
counter host-fn replays to `201` after `seek(0)`).

**Snapshot/checkpoint cadence (the W1 performance piece) — built for single-threaded runs.** `seek(t)`
used to re-execute from time 0, so it was O(t) and repeated `step_back` O(t²). It now keeps a
**checkpoint ladder**: during a `seek` replay the `Inspector` drives the sole vCPU in stride-sized
chunks (every `SEEK_CHECKPOINT_STRIDE` ops, via the existing `seek_target` — *no hot-loop change*) and
snapshots the vCPU at each boundary, so a later `seek`/`step_back` restarts from the nearest snapshot
(`clock ≤ t`) and replays only the tail — a backward sweep drops from O(t²) to ~O(t·stride). A
checkpoint is `frames.clone()` + window bytes (`Mem::snapshot`/`seed`) + the host's restorable
substate (`stdout`/`stderr`/`clock_ns`/`stdin_pos`/cap-replay cursor + record); the shared `VCpu`/`Host`
structure is rebuilt by `fresh_single_root`, so neither needs to be `Clone`. Captured only for the
**root-only, non-fiber, non-durable, no-installed-units, simple-memory** subset where `frames` + window
bytes fully determine the continuation (and the host carries no §13/§14/§22 residue a restore would
drop) — `VCpu::checkpointable` + `Host::checkpoint_safe`; anything richer turns checkpointing off and
falls back to the (correct) replay-from-0. *Tests (`debug_checkpoints.rs`):* a **warm** Inspector
(ladder populated, so `seek` restores) is asserted state-identical — result, paused location, clock,
and window bytes — to a **cold** one (replays from 0) across checkpoint-stride boundaries, a backward
sweep, and one-at-a-time `step_back`. *Still open:* **multithreaded** (`turn`-coordinate) checkpoints,
dirty-page-tracked window copies (today's snapshot is the full mapped prefix), RNG via a dedicated
iface (vs a host-fn), and capturing a `SchedTape`/`CapTape` from a *JIT* execution (the interpreter is
the debug engine by design, so this is lower priority).

---

## 4. W2 — Interpreter stepping / breakpoints / watchpoints

**Pillar 3.** Goal: single-step, breakpoint, and watchpoint over a guest on the interpreter,
**concurrency-aware** (per-fiber/per-vCPU), deterministic, with no JIT plumbing.

**Current substrate.** The interpreter executes at op granularity already (the DPOR explorer
runs at `memop`/`quantum = 1`), holds an explicit reified call stack (`Vec<Frame>`, see §23 —
"a fiber is pure data"), and owns the scheduler. Watchpoints are trivial because guest memory
is one contiguous masked window buffer — "break when any thread touches `addr`" is a single
range check in the load/store path.

**Design sketch.** A `Debug` execution policy alongside the existing `Policy::Dpor` /
deterministic / normal modes:
- **Breakpoints**: a set of `(func, block, op-index)` or IR-PC values; the step loop checks
  before executing. Cheap.
- **Watchpoints**: `(addr, len, RW)` ranges checked in the masked load/store helpers in
  `svm-mem`; fires with the offending vCPU/fiber id. Address watchpoints are the headline win.
- **Stepping**: step-op / step-over (skip to matching frame depth) / step-out using the reified
  frame stack; **per-fiber** because each fiber is a separate `Vec<Frame>`.
- **Concurrency control**: because the interpreter owns the M:N scheduler, the debugger can
  *freeze a chosen interleaving* — step one fiber while others are parked, or replay a specific
  schedule from `run_scheduled`/`explore_all` and break inside it. This is the differentiator a
  native multithread debugger cannot offer.

**API surface.** Through the `Inspector` capability (W8): `set_breakpoint`, `set_watchpoint`,
`step{,_over,_out}`, `continue`, `select_fiber(id)`, `stack(fiber)`, `read_window(addr,len)`,
`read_local(slot)`. Returns IR-level locations until W4 maps them to source.

**Dependencies.** W8 (the capability shell). Optional W1 (step a recorded interleaving). W4 to
show source rather than IR positions.

**Open questions.** Inspecting **SSA-promoted locals** at IR level — a promoted scalar has no
window slot (the §3d/§19 tension). For interpreter debugging this is softer than for the JIT:
the interpreter holds SSA values in its own value table, so it *can* surface them by IR value-id
even when the JIT couldn't. Decide whether to expose IR-value inspection now and defer
source-variable mapping to W4/W6.

**Effort / risk.** **Moderate, low risk.** Pure interpreter work on existing structures;
forbid(unsafe) preserved. No backend or ABI changes.

**Acceptance.** Set a breakpoint and a write-watchpoint on a concurrent guest; run under
`run_scheduled`; the debugger stops at the right op on the right fiber; stepping advances one
fiber while others stay parked; window + IR-value reads are correct.

**Built — Milestone B slice 1 (multithreaded stepping under a fixed schedule).** `Inspector::
attach_scheduled(m, func, args, fuel, schedule)` drives a `thread.spawn` guest cooperatively on
one OS thread under a fixed, reproducible `schedule` — an empty `Vec` for the deterministic
default order, or a `Witness::plan` from W7 [`find_schedule`] to **step a specific (e.g. failing)
interleaving**. The enabling refactor: (1) the per-op debug seam's breakpoint/watchpoint set moved
into a run-shared `DebugShared` (`Arc<Mutex>`) so a breakpoint fires in *whichever* thread reaches
it — `clock`/`step_target` stay per-vCPU; `thread.spawn` children inherit the shared set; (2) the
cooperative scheduler loop became a **re-entrant `SchedDriver`** that pauses on a debug stop
(holding the interrupted vCPU's turn intact) and resumes without re-deciding the schedule —
`run_with_policy` is now its non-pausing wrapper, so the model-checker path is unchanged.
`stopped_task()` reports which thread is paused. `threads()` lists every live thread and
`select_task(id)` focuses read-inspection (`backtrace`/`read_var`/`read_window`/`clock`) on **any**
of them — found wherever the scheduler parks it (runnable / join / wait / spin) — so you can stand
at a breakpoint in one thread and examine another's stack; the focus resets on the next resume,
and stepping always drives the *stopped* thread. Tests (`debug_threads.rs`): a worker breakpoint
fires once per spawned thread (distinct vCPUs); a W7 race witness replays to the lost update (1)
*under the debugger*; single-stepping advances the stopped thread one op; `select_task` inspects a
second live thread mid-stop and switches back. **Watchpoints are cross-thread** — they live in the
same run-shared `DebugShared` and every vCPU's per-op seam checks them, so a window-range watch fires
in whichever thread touches the range, reporting that thread + the confined address + read/write
(`cross_thread_watchpoints.rs`): a write-watch on a contended counter fires before *each* worker's
store (attributed per thread) and a read-watch before each worker's *and* the root's load; while
stopped at a watch you can read the value the thread is about to write and `select_task` another
thread's stack; a watch on an untouched range never fires. **Debug stops compose with a fixed
`find_schedule` witness-plan replay** too: a breakpoint or watch can pause *within* the exact failing
interleaving and resume without desyncing the plan — so you can **catch a lost-update race in the act**,
watching the contended address under the witness and reading the stale value each thread is about to
write (`debug_witness_stepping.rs`). The enabling fix: the one-visible-op-per-turn *yield* now precedes
the debug seam in `run_inner`, so a stop at a budget-exhausted visible op fires at the **start of its
own turn** (on the next pick) instead of running the op inside the previous turn — which had collapsed
two turns into one and left the plan's next `TaskId` no longer runnable (`debug_assert` desync); the
DPOR explorer (no debug seam) and the single-threaded debugger (unbounded budget) are unaffected, gated
by `dpor.rs` matching the brute-force oracle. *Not yet:* W1 record/replay is the rest of Milestone B.

---

## 5. W3 — Trustworthy backtrace materialization

**Pillar 2.** Goal: turn the *integrity* guarantee (return addresses the guest can't smash,
already built) into an actual rendered backtrace, per fiber, even after heap corruption.

**Current substrate.** Integrity is done: out-of-band control stack (§5), per-fiber control+data
pair (§3d/§23), three `svm-fiber` ABIs. On the **interpreter** a backtrace is already free — the
reified `Vec<Frame>` *is* the call stack; W2 exposes it directly. The missing piece is the
**JIT**, where the control stack is the Cranelift-managed machine stack and walking it needs
frame/unwind metadata.

**Design sketch.**
- *Interpreter*: expose `stack(fiber) -> Vec<FrameInfo>` from the existing frames (essentially
  free; fold into W2/W8).
- *JIT*: have Cranelift emit unwind info (it already does for its own exception/backtrace
  support — Wasmtime precedent) and walk the out-of-band stack from a trap/inspection point,
  per fiber. With migratable fibers (§23/D57) the walk is rooted at the fiber's saved SP, not
  the OS thread's — so the unwinder takes a fiber handle, not a thread.

**API surface.** `Inspector::backtrace(fiber) -> Vec<FrameInfo { func, ir_pc, source? }>`.
Source frames appear once W4 lands.

**Dependencies.** Interpreter path: W2/W8 only. JIT path: Cranelift unwind-info emission; reuse
the `fiber_rt` saved-SP. Symbolization to source: W4.

**Open questions.** Whether to materialize JIT backtraces eagerly at trap time (cheap, the
detect-and-kill path already has the context) or lazily via the `Inspector`. Trap-time capture
is likely the higher-value first step (it improves every existing `Trap`/kill message).

**Effort / risk.** Interpreter: **trivial.** JIT trap-time backtrace: **moderate.** Full
fiber-rooted JIT unwinding from an arbitrary inspection point: **moderate-high.**

**Acceptance.** A guest that corrupts its data stack still produces a correct per-fiber
backtrace; the trace names the right fibers under work-stealing migration.

### Scoping — JIT trap-time backtrace (always-on source frames on a kill)

**Goal.** When a JIT'd guest **traps** (memory fault, div/rem-by-zero, `unreachable`, indirect-call
type, …), capture and symbolize the guest call stack into the kill/`Trapped` report — so *every*
trap message carries a `file:line` backtrace, with **no debugger and no `-g`-only gate** (the §6
debug info enriches it but symbolization works from the Stage 1 address map regardless).

**Why this, why now.** It is the host-facing complement to the gdb tooling just shipped: it reuses
the exact primitives that tier built — `CompiledModule::symbolize(pc)` (the machine-pc → source map,
Stage 1) and the frame-pointer-chain walk (the `preserve_frame_pointers` rbp frames the
`.debug_frame` CFI / `fiber_backtrace` already rely on) — pointed at a trap site instead of a parked
fiber. It closes Pillar 2 (W3) on the JIT, and it has a **built-in oracle**: the interpreter
reifies its call stack as `Vec<Frame>`, so the JIT trap backtrace is differential-testable against it.

**Substrate / the wrinkle.** Two trap families capture differently:
- **Memory faults** (§4/§5 guard page → `trap_shim.c`'s SIGSEGV/SIGBUS handler): at the fault the
  guest stack is **fully intact**, and the handler's `ucontext` carries the faulting `rip`/`rbp`. The
  capture point is *in the handler, before `siglongjmp`* — async-signal-safe (stack reads + a
  thread-local buffer write); symbolize later in normal context.
- **Explicit-check traps** (div/rem/`unreachable`/indirect-call-type): the lowered check stores a
  `TrapKind` and `return`s, **unwinding the native stack** via the trap-propagate path — so the
  frames are gone by the time the host observes the trap. These need the PC/frames captured *at* the
  check (or accumulated along the propagate path), a harder, separate step.

**The wrinkle that bit, and the fix.** The first cut captured only `(rip, rbp)` in the handler and
walked the `rbp` chain *on the host* after the fault. That is **unsound**: `siglongjmp` unwinds back
onto the *same* stack the post-fault host code (and the symbolization itself) then reuses, so the
dead guest frames are overwritten before the host can read them — the walk read clobbered slots and
flaked (`fp − sp` was ~100 bytes; the guest frames overlapped the live host frames). The walk must
therefore happen **entirely in the handler**, while the stack is intact: it chases the chain and
stashes the faulting `pc` + each frame's raw return address; the host only *symbolizes* (pure
arithmetic, no stack reads). The handler walk moves *up* toward the stack base (away from the
low-address guard the fault sits near) and is bounded — aligned, non-null, strictly-increasing
links, a span backstop, a 64-frame cap — so a corrupt chain terminates, async-signal-safely.

**Stages (each independently shippable, differential-tested vs the interpreter):**
- [x] **Stage 0 — the symbolize-and-classify core.** `symbolize_capture(pc, rets, sym)` (pure): the
  innermost frame from `pc` (the faulting instruction, symbolized directly), then one per captured
  return address (callers at `ret - 1`, inside the call), stopping at the first non-guest frame;
  adjacent duplicates collapse. (The frame-pointer *walk* itself lives in `trap_shim.c`'s handler —
  see the wrinkle above.) Unit-tested with a synthetic capture + a fake symbolizer.
- [x] **Stage 1 — memory-fault backtrace.** The `svm_handler` walks the chain and stashes
  `pc + rets[]`; `mem::take_trap_frame` reads them; `CompiledModule::trap_backtrace` symbolizes them
  into a `Vec<JitFrameLoc>` published by `last_trap_backtrace()` after every `run`. *Tests
  (`jit_trap_backtrace.rs`, unix):* an out-of-bounds store reports a backtrace naming the faulting
  function + line; a callee fault walks the caller chain (`[func1@store, func0@call]`); a
  non-trapping run leaves it empty. Stable under parallel + serial runs.
- [x] **Differential oracle (the built-in check this slice was premised on).** The interpreter
  reifies its call stack as `Vec<Frame>` and leaves it intact on a trap (returns `Err` without
  unwinding), so an `Inspector` driven to the trap exposes the trap-time stack via `backtrace()` — no
  interpreter changes. A test asserts the JIT's `last_trap_backtrace()` equals the interpreter's
  frames `(func, file, line)`, innermost first, across both trap families × single/multi-frame. This
  is the project's standard interp↔JIT differential validation, now closed for trap backtraces.
- [x] **Interpreter symmetry (the plain-run counterpart).** The trap-time backtrace was previously
  reachable on the interpreter only via the `Inspector` (stepping); `run_traced` /
  `run_with_host_traced` now return it on the plain run path too — the M:N executor snapshots the
  trapping vCPU's live `Vec<Frame>` into its run outcome, and a free `source_loc(m, pc)` resolves each
  `IrPc` to source (the symmetric analog of the JIT's `JitFrameLoc`). Both engines now report *where*
  a guest trapped from their ordinary run entry, which also lets the interp↔JIT differential fuzzer
  pinpoint divergences. *Tests (`interp_trap_backtrace.rs`, all platforms):* a store fault names the
  store; a callee div-by-zero walks the caller chain; a clean run is empty.
- [x] **Bytecode-engine symmetry (the third backend).** The Phase-1b bytecode engine (`run_fast` /
  `run_with_host_fast`, the production interpreter path) previously produced *no* backtrace on a trap —
  the one engine of three (tree-walker, bytecode, JIT) without one. `run_with_host_fast_traced` /
  `run_fast_traced` close that: the engine reifies its continuation as data (the flat register-window
  call stack), so on a trap `vm_trap_bt` walks the active `Vm` cursor + suspended caller windows into
  the *same* `Vec<IrPc>` (innermost first) the tree-walker snapshots from `Vec<Frame>` — driven one op
  at a time (the single-vCPU debug seam, bit-identical to run-to-completion) so the cursor still points
  at the faulting op. Single-vCPU, seam-free scope (S4); a durable host / out-of-subset op / concurrency
  seam falls back to `run_with_host_traced`, so a backtrace is never dropped. The `IrPc`s are
  **bit-identical** to the tree-walker's, including its quirks: the live frame's `inst` advances past
  the op for every trap but `OutOfFuel` (the tree-walker does `inst += 1` before eval), a caller's
  `inst` sits past its call, and a terminator trap (`unreachable`) reports the block's instruction
  count. *Tests:* `bytecode_traced.rs` (memory fault, single/multi-frame div, `unreachable` incl. an
  empty block, `OutOfFuel`, clean run); and `bytecode_diff.rs` asserts `IrPc`-equality with
  `run_traced` on **every trapping generated module** — the standard randomized oracle, now guarding
  backtrace parity going forward.
- [x] **Stage 2 — explicit-check trap backtrace.** An explicit trap has no signal — the lowered check
  stores its kind and `return`s, unwinding the guest frames — so the capture happens *at the trap
  site*: `emit_trap` (the single origin every explicit trap routes through — div/rem, `unreachable`,
  `OutOfFuel`, indirect-call-type) emits a `call svm_capture_explicit_trap` before the store+return,
  gated on `-g`. The helper walks the frame-pointer chain from its caller (the trapping guest frame,
  found via `__builtin_frame_address`) into the *same* thread-local the signal path uses, so the host
  symbolizes both identically; the trap site rides in `rets[0]` (symbolized at `ret − 1`, like every
  caller) since it's a return address. The current op's `SourceLoc` is in effect at `emit_trap`, so
  the line is precise for ops that carry a `debug.loc` (div/rem, `unreachable`); `OutOfFuel`/the
  back-edge checks attribute the right *function* but an approximate line. *Tests:* a div-by-zero
  names the div line; a callee div-by-zero walks the caller chain `[func1@div, func0@call]`.
- [x] **Stage 3 — vCPU attribution + the kill message.** Two halves: *(a)* a trap on a **spawned
  vCPU** captured into that worker's `trap_shim.c` thread-local, which dies with the worker — so the
  dying worker now hands its capture to the run-scoped `Domain` (last-wins, matching the last-wins
  trap cell), and the run thread reconciles it with its own capture after `join_all` (the root's own
  trap takes precedence, the common single-vCPU case). *(b)* the host's kill message now carries the
  backtrace: `powerbox_compile_run` plumbs `last_trap_backtrace()` out and `run_powerbox*` appends a
  `file:line:col (fn N)` block to the `guest trapped (…)` error. *Tests:* a `thread.spawn`ed worker's
  div-by-zero is attributed to the worker's div line; a powerbox div-by-zero's kill message names the
  source.
- [x] **Stage 4 — per-fiber attribution under migration (§23/D57).** The capture was vCPU-thread-rooted,
  so a work-stealing-migrated fiber (which may resume on a different vCPU thread than it suspended on)
  couldn't be *named*. Now the fiber runtime publishes the **running fiber handle** into a shared
  `trap_capture.c` thread-local across the resume seam (`svm_set_current_fiber`, save/restore-bracketed
  around `(*fib).resume` exactly like the durable shadow-SP swap — stack-disciplined for nested
  resumes), and every capture path stashes it: unix memory-fault (`svm_store_trap_frame`) + explicit
  (`svm_capture_explicit_trap`) read it directly, the Windows VEH snapshots it (`svm_current_fiber`).
  It rides through `take_trap_frame` → the `Domain` handoff → `CompiledModule::last_trap_fiber()`
  (`Some(handle)` for a fiber, `Some(-1)` for the root, `None` on a clean run), and the kill message
  names it (`… [fiber N] …`). Captured *at the trap instant*, so migration can't misattribute it — the
  thread no longer identifies the fiber, but the published handle does. The **interpreter** reports the
  same attribution (it's single-OS-thread M:N, so it always knows the running fiber): `run_traced` now
  returns the trapping fiber as a third field (`Outcome::trap_fiber`, `trap_fiber_of` = the trapping
  vCPU's `cur` fiber handle or `-1`). Fiber handles are cross-backend-identical (`(generation <<
  FIBER_GEN_SHIFT) | slot`), so the two engines must agree — which is the **interp↔JIT differential**
  that validates the JIT's at-the-trap-instant capture against the oracle. *Tests*
  (`jit_per_fiber_trap.rs`, unix — interp vs JIT for each): a div-by-zero in a resumed fiber → that
  fiber's handle (and still names the div line); a **nested** resume (root → A → B) → the innermost
  fiber B, pinning the resume-seam save/restore discipline; a root trap → `-1`; a clean run → `None`.
  (The JIT capture is `-g`-gated, so a trapping differential program carries `-g`; the interpreter
  reifies frames regardless.)
- [x] **Stage 5 — multi-vCPU trap origin (the interpreter side of Stage 3).** A trap on a
  `thread.spawn`ed worker propagates to its `thread.join`er as a bare `Err(Trap)` — the parent re-traps
  with *its* frames at the join — so the interpreter's run outcome named the *join site*, not where the
  guest actually trapped (the JIT already reported the origin via the `Domain` capture handoff). The
  interpreter now matches via a run-shared **first-wins trap-origin cell** (`Sched::trap_origin`): the
  first vCPU to trap on its own op records its backtrace + fiber, and `run_traced` reports that instead
  of the root's own outcome — the interpreter counterpart of the JIT's `root_trap_cap.or(worker_trap_cap)`.
  First-wins also makes the reported `(backtrace, fiber)` self-consistent (from a single trap event)
  under racing concurrent traps, rather than two independently last-wins fields. *Tests*
  (`multivcpu_trap_origin.rs`, interp↔JIT differential): a worker div-by-zero names the worker's origin
  line (not the join site), and the interpreter and JIT agree on the origin frame + fiber. *Remaining:*
  picking the *exact* killing trap when several race simultaneously (first-wins on the interp,
  last-wins on the JIT — both report *a* valid trapping frame's chain, not necessarily the same one).

**Trust/TCB.** Pure host-side observability, off the runtime hot path and the escape-TCB: a wrong
backtrace mis-renders a kill message, never affects confinement. The capture stays async-signal-safe
in the fault handler (no allocation — a fixed thread-local buffer; symbolization is deferred to the
host in normal context).

**Progress:** Stages 0–3 **done** + the interp↔JIT **differential oracle** wired — the JIT trap-time
backtrace is live for both memory faults *and* explicit-check traps, attributed across spawned vCPUs,
folded into the host's kill message, and validated frame-for-frame against the interpreter. **Memory
faults now capture cross-platform** — unix via the SIGSEGV/SIGBUS handler, Windows via the Vectored
Exception Handler (it walks the faulting `CONTEXT`'s `Rbp` chain in `mem.rs`, validated by
`memfault_kill_message_carries_a_source_backtrace` on the `windows-latest` CI job). This closes Pillar
2 (W3) on the JIT.

**Function names** (the readability finish-up): the §6 debug-info waist gained a `func → name` table
(`debug_info.func_names`, text `debug.fname <func> "<name>"`, binary-encoded). **All three frontends
populate it under `-g`** — chibicc emits `debug.fname` per function; svm-wasm reads each
`DW_TAG_subprogram` `DW_AT_name` (mapped to its IR function by PC range); svm-llvm reads each
`DISubprogram` source name (correlated to the IR function index by linkage name). Threaded through
every backtrace renderer — `JitFrameLoc::func_name`, the interpreter's free `func_name(m, func)`, the
kill message (`#0 file:line:col in compute`), and gdb's DWARF `DW_AT_name` + ELF `.symtab` — so frames
read `in compute` instead of `(fn 0)`/`fn0`. Empty ⇒ the `fn{N}` fallback.

Trap-time backtraces are now **fully cross-platform**: the capture state + frame-pointer walk +
explicit-trap helper live in a shared `trap_capture.c`, and `emit_trap` threads the trapping frame
pointer in via Cranelift `get_frame_pointer` (sidestepping MSVC's missing `__builtin_frame_address`),
so div-by-zero / `unreachable` / `OutOfFuel` / indirect-call-type traps capture on **unix and Windows**
(ISSUES I5 — landed, pending `windows-latest` CI confirmation). Remaining: per-fiber naming under
work-stealing migration (ISSUES I6, S4 cosmetic).

---

## 6. W4 — debug-info ABI: a frontend-neutral narrow waist (D-DBG-7)

**Pillar 4, step zero.** Goal: carry source `(file, line, col)` and variable→location info from
**any** frontend through the IR, so every later tool (interpreter source view, DWARF, DAP) has
something to symbolize against — without baking in one frontend's debug model.

**The reframe — do not design around chibicc.** Three frontends are in scope (§20), and two of
the three already carry rich, DWARF-shaped debug info:

| Frontend | Debug info it carries | Shape |
|---|---|---|
| **chibicc** | AST `Token` (file/line/col) | bespoke, minimal |
| **LLVM** (D54) | `!DILocation`, `!DILocalVariable`, `llvm.dbg.value`, DI type graph | DWARF-shaped, rich; **already solves optimized variable locations** |
| **wasm** | embedded DWARF (`.debug_*`) + name section | literally DWARF |

chibicc is the outlier, so designing the waist around its token model would be backwards.

**The principle.** Debug info crosses the IR boundary like everything else, so it follows §20's
*"IR is the stable ABI; frontends are plugins"* — a **narrow waist**: a small mandatory
frontend-neutral **core** plus an optional opaque per-producer **rich blob**.

- **Neutral core (mandatory; every frontend populates it during lowering)** — only what *our*
  tools need:
  - `IrPc → SourceLoc { file_id, line, col? }` — keyed on S1's `IrPc`, stored as **ranges**.
  - `VarInfo { name, ty: TypeRef, scope: IrPc-range, loc: VarLoc }` — `VarLoc` is exactly **S2**
    (`Window{off}` | `Ssa(LocList)` | `Machine(..)`).
  - a `files` table + a **neutral** `TypeRef { name, encoding: Signed|Unsigned|Float|Bool|Ptr|
    Aggregate|Opaque, size }` — enough to render primitives and "opaque struct of N bytes at
    addr"; full structure lives in the rich blob.
  This is everything the **interpreter stepper** (pillar 3) and **backtraces** (pillar 2) need.
- **Rich blob (optional, opaque, per-producer)** — `{ producer: Chibicc|Llvm|Wasm|Other(str),
  bytes }`, strippable, that the **middle never parses**. Carries full DWARF DIEs / LLVM DI
  metadata / language type graphs. Only **W5 (DWARF emit)** and a **DAP server** consume it,
  host-side. The verifier and interpreter ignore it (§2a, untrusted tooling).

**Why the waist beats the alternatives.**
- *vs. a chibicc-shaped table* — forces LLVM/wasm to down-convert richer info into a C-flavored
  schema (fidelity loss), and gets reworked the moment a non-C language appears.
- *vs. "just use DWARF as the interchange"* — our `IrPc` is **structural** (`module/func/block/
  inst`); DWARF line programs assume a **linear** PC. That forces a structural model into a
  linear one at the IR and back out for the JIT, and makes the interpreter parse DWARF for basic
  stepping. The waist keeps DWARF an *output* (W5) and an *input* (ingest), never the waist;
  faithful DWARF re-emit for LLVM/wasm comes from carrying their original blob and having W5
  *prefer* it.

**Load-bearing insight.** The `IrPc → source` map **must be built during lowering**, by each
frontend — only the lowering step knows "this source position produced these IR ops." It can't
be reconstructed downstream. So the neutral core is the mandatory interchange every frontend
populates as it lowers; the rich blob is optional pass-through.

**Per-frontend mapping.**
- *chibicc* — emits only the neutral core from its `Token`s. No DWARF generation; the toy
  frontend stays toy. Sufficient for C.
- *LLVM* — `!DILocation` → `SourceLoc`; **`llvm.dbg.value` → `VarLoc` location lists** (LLVM
  already computes post-optimization variable locations, so **S2's promotion-vs-inspectability
  problem is solved for free on the LLVM path** — its dbg intrinsics survive mem2reg/SROA).
  DI type graph → rich blob.
- *wasm* — remap the embedded DWARF line program (wasm-offset → `IrPc`) into the neutral core
  during translation; pass the original `.debug_*` through as the blob (W5 mostly relocates).

**Storage.** A new **strippable binary section** + text-format syntax (1:1 text↔binary per D33);
absent ⇒ no debug info, zero cost; verifier ignores it (§2a). `Module::debug_info:
Option<DebugInfo>`; helpers `loc_of(ir_pc)`, `vars_in_scope(ir_pc)`. Consumed first by W2
(interpreter source view), then W5 (DWARF/DAP).

**Build order.** Implement **chibicc populating the neutral core first** — it's the MVP and
validates the waist end-to-end against the interpreter stepper (W2) — but **freeze the schema
now with LLVM/wasm in mind** (file table; **optional** columns, since wasm DWARF often lacks
them; `IrPc` ranges; neutral `TypeRef`; producer-tagged opaque blob; room for an `inlined_at`
chain, since LLVM/DWARF carry inlining).

**Dependencies.** Builds on S1 (`IrPc`) + S2 (`VarLoc`), both settled. Blocks W5 and the
source-level half of W2.

**Open questions.**
- *Core loc granularity*: per-op (precise, larger) vs per-statement (smaller). Start
  per-statement ranges.
- *Rich-blob versioning*: the blob is producer+version tagged (LLVM bitcode/DWARF versions
  drift); W5/DAP refuse an unknown version rather than mis-render.
- *Text-format ergonomics*: neutral core in the text IR (agent-friendly, D33); rich blob
  binary-only (it is opaque bytes). Lean that way.

**Effort / risk.** **Moderate** for the neutral core + chibicc (new IR section + encode/decode +
verifier skip + frontend threading + text syntax; `svm-ir`, `svm-encode`, `svm-text`,
`svm-verify` skip, `codegen_ir.c`). Low *risk* (additive, strippable, no TCB). The LLVM/wasm
ingest sides are scoped by their own on-ramps (LLVM.md/WASM.md), targeting this waist.

**Acceptance.** A C program compiled with `--emit-ir -g` round-trips its neutral core through
text and binary; the interpreter prints the source line + a named local for any `IrPc`; stripping
the debug section yields a byte-identical-minus-debug module that runs identically; the schema has
a place for an LLVM/wasm rich blob without change (a `producer` tag + opaque bytes round-trips).

---

## 7. W5 — DWARF emission + DAP server

**Pillar 4, the real work.** Goal: gdb/lldb and VS Code (via the Debug Adapter Protocol) set
source breakpoints and inspect source variables on JIT'd guest code.

**Current substrate.** Cranelift already emits DWARF for JIT code (Wasmtime precedent); the new
piece is mapping **our** IR debug-info (W4) into Cranelift's debug-info inputs and serving DAP.

**Design sketch.** Cranelift `ir::SourceLoc` per instruction from W4's `locs`; value-location
lists for variables (needs W6 for promoted locals); assemble DWARF line + variable programs over
the JIT'd blob; a thin **DAP server** translating DAP requests onto the `Inspector` (W8) +
DWARF. Reuse the interpreter (W2) as the *stepping engine behind DAP* for a source-level
experience without solving optimized-code inspection first.

**Dependencies.** W4 (the neutral core + the per-producer rich blob — for LLVM/wasm guests W5
prefers the native DWARF blob and mostly relocates addresses; for chibicc it synthesizes DWARF
from the core, D-DBG-7), W6 (promoted-local locations), W3 (frame/unwind for call-stack
requests), W8 (the `Inspector` DAP binds to).

**Open questions.** Two-engine DAP (interpreter for stepping/inspection, JIT for speed) vs
JIT-only DWARF. The interpreter-backed DAP is far cheaper and sidesteps the optimized-debug
problem — **recommend interpreter-backed DAP first**, JIT/DWARF as a later tier.

**Effort / risk.** **High.** The largest workstream; explicitly staged. Defer until W2/W4 prove
the source-level loop on the interpreter.

**Acceptance.** Set a breakpoint in VS Code on a `.c` line; it binds; hitting it shows the
source frame and inspectable locals.

**Built — slice 1 (interpreter-backed DAP server).** A new `svm-dap` crate translates Debug Adapter
Protocol requests onto the `Inspector` — so the **interpreter is the stepping engine** and source
mapping comes straight from the §6/W4 debug info, with **no DWARF and no JIT** (the doc's recommended
first tier; optimized-code inspection is sidestepped entirely). `DapServer::handle(request) ->
[messages]` covers the acceptance loop: `initialize` (+ `initialized` event), `launch` (parse IR,
attach the `Inspector`), `setBreakpoints` (source line → IR pc via a reverse `(file,line)→pc` index
over `debug.loc`, snapping forward to the next line with code), `configurationDone`,
`threads`/`stackTrace`/`scopes`/`variables` (frames carry the source location; locals enumerate
`debug.var` and resolve through `read_var`), `continue`/`next`/`stepIn`/`stepOut`, `disconnect`, and
the `stopped`/`terminated` events. JSON is hand-rolled (no serde — matching the workspace's
dependency ethos); `run_stdio` is the `Content-Length`-framed wire loop a real client (VS Code)
connects to, and the `svm-dap` binary is the server. Test (`dap.rs`): a scripted conversation sets a
breakpoint on `sum.c:7`, hits it, and reads back the source frame plus `i = 3` / `acc = 0` — the
acceptance, no editor needed.

**Built — slice 2 (reverse debugging over DAP).** The W5 server now exposes the W1 time-travel
engine to the editor: `initialize` advertises `supportsStepBack`, so VS Code enables its reverse
controls. `stepBack` calls `Inspector::step_back` (reverse single-step); `reverseContinue` runs
*backward* to the previous breakpoint — found by re-executing from time 0, remembering the last stop
strictly before the current op `clock`, and `seek`ing there (else rewinding to the start). Test
(`dap.rs`): a loop-body breakpoint hits three times (`i` = 3, 2, 1) and `reverseContinue` walks
backward through the hits — `i=1,acc=5` → `i=2,acc=3` → `i=3,acc=0` → entry — with the locals
correct at each, then `stepBack` reverse-single-steps. Reverse debugging in an editor is a genuine
differentiator (few debuggers implement DAP's `stepBack`/`reverseContinue`), here free from the
deterministic interpreter + `seek`.

**Built — slice 3 (multithreaded DAP).** A scheduled `launch` (arg `schedule: [tids]` for a fixed
interleaving / witness, `schedule: []` for the deterministic default, or `seed: N` to fuzz) drives a
`thread.spawn` guest, and the editor sees every vCPU as a **DAP thread** (`threads` → `Inspector::
threads`, DAP id = vCPU id + 1). The `stopped` event names **which thread** hit the breakpoint
(`stopped_task`), `stackTrace(threadId)` focuses that thread (`select_task`) and reports *its* stack,
and `variables` reads the right thread's frame — DAP frame references encode `(thread, frame)` so a
client switching threads never reads the wrong one. Test (`dap.rs`): two workers over a shared
counter, a source breakpoint at `worker.c:4` fires in one worker (the `stopped` event names it, not
the root), `threads` lists root + workers, the worker's `stackTrace`/`variables` show `worker.c:4`
with `delta = 1`, and `continue` stops in the *other* worker. So the headline multithread debugger —
per-thread breakpoints, thread selection, deterministic interleavings — is now usable from an editor.

**Built — slice 4 (`evaluate` + multithreaded reverse-continue).** `evaluate` resolves a watch /
hover / REPL expression — slice 1 a bare source-variable name in the given frame, read through
`read_var` (advertised via `supportsEvaluateForHovers`; an unknown name fails so the client shows
"not available"); richer expressions are a follow-up. And `reverseContinue` now picks its time-travel
coordinate by mode — the global scheduler `turn` when multithreaded, the op `clock` single-threaded —
so reverse debugging works for concurrent guests too. Tests (`dap.rs`): `evaluate("i")`/`("acc")`
return `3`/`0` at the loop breakpoint (`"nope"` fails); and a two-worker run's `reverseContinue`
walks back through the per-thread breakpoint hits, then to the start.

**Built — slice 5 (step-over / step-out).** Stepping is now call-depth-aware via the reified frame
stack (W2). A new `Inspector::step_over` runs any call the current op makes to completion instead of
descending (stops at the next op in the *same* frame), and `step_out` runs until the current
function returns (stops at the op in the caller it returned to). Both are one depth-aware primitive:
the per-op seam stops at the first op (after stepping off the current one) whose call depth is `<=`
a target — current depth for over, current − 1 for out — read from `frames.len()`. DAP `next`/
`stepOut` map to these (and `stepIn` stays single-op = descend). Tests: at the Inspector level
(`debug.rs`) `step_over` stays in the caller and runs the callee, `step`+`step_out` descend then
return; at the DAP level (`dap.rs`) `next` on a call line lands on the next source line in one frame
(no descent).

**Built — slice 6 (source-line stepping + expression evaluator + conditional breakpoints).** Two
editor-facing refinements, both pure DAP/interpreter-side (no ABI change):
- **Source-line stepping.** Op-step stays the interpreter primitive (and the behavior with no debug
  info — IR-level debugging); but with debug info the DAP `next`/`stepIn` now op-step *until the
  frame's source line changes*, so the editor advances a line at a time rather than stuttering
  op-by-op across one C line. (`stepOut` already lands in the caller.) A safety op-cap guards against
  unmapped code.
- **Scalar expression evaluator** (`svm-dap::expr`, ~one screen, hand-rolled): integer literals,
  frame variables, `()`, unary `- ! ~`, and the C arithmetic/bitwise/comparison/logical binops with
  C precedence; values are `i64`. It powers a richer `evaluate` (watch / hover / REPL — a bare
  variable keeps its typed form, anything else evaluates to an integer) **and conditional
  breakpoints**: DAP `setBreakpoints` `condition`s are stored per pc and `continue`/`configurationDone`
  transparently skip a breakpoint whose condition is zero (`supportsConditionalBreakpoints`).
  Tests (`dap.rs`): one `next` advances a two-op source line to the next line; a `i == 1` conditional
  breakpoint skips the i=3/i=2 loop iterations and stops at i=1; `evaluate("i * 2 + acc")` = 6,
  `"i / acc"` (acc=0) fails cleanly. Evaluator unit tests in `expr`.

**Built — structured `TypeRef` ABI + producer (W4 slice 7).** The debug-info waist now carries a
structured type table — `DebugInfo::types: Vec<TypeDef>` (`Base{enc,size}` | `Pointer{pointee}` |
`Array{elem,count}` | `Aggregate{size,fields:[{name,offset,ty}]}` | `Opaque{size}`), referenced
by a `VarInfo::type_id: Option<TypeId>` — so a variable's **layout** (struct field offsets, array
strides, pointee) crosses the IR, not just a render name. Text round-trips it (`debug.type` /
`debug.field`, and an optional trailing id on `debug.var`); the `ty` name string stays for the
scalar common case and back-compat (a var with no `type_id` is name-only). chibicc `-g` emits it:
`intern_type` walks each named local's C `Type`, interning by pointer identity (which bounds the
`struct Node { struct Node *next; }` cycle) plus structural dedup for base scalars, and records
field offsets / element counts / pointees. Tests: chibicc emits a `struct`/array/pointer with the
right offsets and a self-consistent table (`c_frontend.rs`); the table round-trips through text
(`debug.rs`).

**Built — DAP Variables-pane aggregate expansion (W4 slice 8, the first consumer of the type
table).** A struct/array local now shows in the editor's Variables pane with an expand triangle:
the DAP server hands back a nonzero `variablesReference` (a `Place` = focused thread + window
address + `TypeId`), and a `variables` request on it enumerates the struct's fields / the array's
elements — each itself expandable if aggregate, scalar leaves read straight from the thread's
window and formatted per their `TypeDef` (signed/unsigned/float/`_Bool`/pointer-as-hex). Addresses
come from the frame's data-SP (`read_ir_value(frame, 0)`) plus the `Window` offset; `seek`/resume
clear the place table (the addresses go stale). Test (`dap.rs`): a guest fills a `struct {int x,y}`
and an `int[3]`, and the scripted conversation expands `p` → `x=11, y=22` and `row` → `[0]=100,
[1]=200, [2]=300`. The `evaluate` member/index half landed in slice 10 and pointer-deref
expansion in slice 11; richer render names (the C tag) in slice 12 (all below).

**Built — `evaluate` member / index / arrow access (W4 slice 10, the consumer half completed).**
`evaluate` now resolves `a.b`, `arr[i]`, and `p->x` (and combinations like `p.x + arr[i]`,
`pp[0].y`) over the structured types. The scalar `expr` evaluator grew a frontend-agnostic
[`expr::Resolver`] trait + a [`Value`] = `Int | Place{addr, type_id}`: parsing/precedence stay in
`expr` (now with postfix `.`/`->`/`[]`), while the *semantics* (name lookup, member/index/deref,
integer coercion) are a callback the caller implements. `svm-dap`'s `EvalEnv` is that callback —
pure address arithmetic over `TypeDef` + window reads, no `svm-ir`/frontend types in `expr`
itself. `eval_int` (conditional breakpoints) is unchanged, now a thin wrapper over the same core.
Tests (`dap.rs`): member/index/mixed arithmetic over a struct+array, and `->`/pointer-indexing
through a pointer; bad accesses (`p.nope`, `p.x.y`, `pp->x->y`) fail cleanly. (Pointer-deref
*expansion* landed in slice 11, richer render names in slice 12, and floats / short-circuit in
slice 13.)

**Built — frontend-neutrality cleanup (W4 slice 9).** Scalar read widths now come from the
structured type's `size` (`scalar_width` → `TypeDef.size`), not the variable's *name*; the old
C-name heuristic (`ty_width`) survives only as the fallback for name-only / legacy debug info with
no `type_id`. That removes the one remaining C-specific assumption from the normal consumer path —
an audit of `svm-interp` + `svm-dap` finds **zero** chibicc/frontend references and `ty_width` as
the lone C-name site. A standing **neutrality test** (`dap.rs`,
`dap_inspects_a_non_c_frontend_by_structured_layout_only`) inspects a debug section with *non-C*
type names (`i32`, `Pair`, an `x.rs` file) and asserts the interpreter + DAP read and expand it
correctly using only the structured layout — locking the consumer's frontend-agnosticism into CI
(the D-DBG-7 waist holds: consumers depend only on the neutral `DebugInfo`, never on a producer).

**Built — reverse-continue honors conditional breakpoints (W5).** `reverseContinue` now skips a
breakpoint hit whose `condition` is false when walking *backward*, identical to forward `continue`,
so it lands on the previous hit that actually fires (not just any breakpoint pc). The condition
check moved to a shared `Session::condition_holds` used by both directions. Test (`dap.rs`): with
`i != 2` over a loop hitting `i=3,2,1`, reverse-continuing from the `i=1` stop skips the
false-condition `i=2` hit back to `i=3`. (Float expressions and short-circuit `&&`/`||` in `evaluate`
landed earlier — see `expr.rs`.)

**Built — lexical-scope resolution of shadowed variables (§6).** A source variable now carries an
optional `scope` — a `(start_line, end_line)` source-line range — so a name with multiple
declarations (an inner-block redeclaration, or a local shadowing a global) resolves to the binding
**in scope at the stopped pc**, not just the first declared. The consumer maps the stopped pc → its
source line and, among same-name candidates, keeps those whose scope covers it, choosing the
innermost (largest `start_line`; a function-wide `None` is outermost) — in `read_var`/`var_addr`
(so `evaluate`/watch/conditions follow) and the DAP Variables pane (one entry per name, the visible
shadow). chibicc emits it: each local's declaration line plus its enclosing block's closing-`}` line
(stamped at scope-exit in the parser; a parameter/top-level local stays function-wide). The waist
threads the field through text (`debug.var … scope <s> <e>`), binary, and the schema. Source-line
ranges (not `IrPc` ranges) because a frontend knows source spans at parse time and the consumer
already maps a pc to a line. Test (`c_frontend.rs`): a C function with an inner-block `int x`
shadowing an outer `int x` reads `x = 105` inside the block and `x = 6` after it.

**All three producers now emit scopes.** The **wasm** DWARF reader recovers a variable's enclosing
`DW_TAG_lexical_block` `[low_pc, high_pc)` and maps it — via the line table the source-line half
already parses — to a `(decl_line, block_last_line)` scope; the **LLVM** DI reader reads each
`DILocalVariable`'s `DILexicalBlock` scope and, since `DILexicalBlock` carries no end line, derives
it from the `!dbg` `DILocation`s of the instructions in the block (walking the location's scope
chain). Both correlate by the same source-line `scope` field the consumer resolves — so a clang
guest (wasm or `-O0` LLVM) with an inner-block `int x` shadowing an outer one resolves to the inner
shadow inside the block and the outer after it, exactly as chibicc does (the LLVM lane verifies it at
runtime; the wasm lane structurally, since a bare `WindowVia` frame read needs `__stack_pointer`
setup the direct call doesn't do). *Not yet:* the JIT/DWARF tier for gdb/lldb on native code — scoped
below.

### Scoping — the JIT/DWARF tier (gdb/lldb on native JIT'd code)

**This is the W5 "real work" the interpreter-backed tiers above deliberately deferred** — a
cross-session workstream tracked here. Goal: a developer attaches **gdb or lldb** (or VS Code via
DAP) to a process running JIT'd guest code, sets a breakpoint on a `.c`/source line, it binds, and
hitting it shows the source frame, a backtrace, and inspectable source variables — at native speed.

**Why now / what changed.** The §6 waist is complete across all three frontends (source lines,
structured types, variables incl. `SsaList`/`WindowVia`/`Fixed`, lexical scopes, globals). That is
the **load-bearing simplification for this tier**: the JIT/DWARF emitter has *one* uniform input —
the neutral `DebugInfo` — to **synthesize** DWARF from, rather than transforming three different
native blobs (the original §7 sketch's "prefer the native blob, relocate addresses" is moot because
the JIT's addresses are *machine* pcs, so even a carried blob's line program and location
expressions must be rewritten anyway). So: synthesize DWARF from the §6 core uniformly; the native
DWARF blobs become an optional later fidelity enhancement, not a prerequisite.

**Current substrate (what exists).** `svm-jit` compiles IR → machine code via `cranelift-jit`
(`JITModule`), one `cranelift_frontend::FunctionBuilder` per function. Cranelift already provides the
two hooks this tier needs: `func.set_srcloc(inst, SourceLoc)` (an opaque `u32` we own, attached per
instruction → carried into the compiled address map) and `ValueLabelsRanges = HashMap<ValueLabel,
Vec<ValueLocRange>>` (value-location lists — *the* W6-JIT substrate, a value's
register/stack-slot over a machine-pc range). It does **not** yet set srclocs, label values, emit
unwind info, or produce any DWARF. The project **hand-rolls DWARF parsing** (`svm-wasm`'s
`dwarf_line.rs`/`dwarf_info.rs`), so the ethos is to **hand-roll the writer** (the inverse) rather
than add `gimli`; revisit only if loclist encoding gets heavy. No `gimli`/`object` deps today.

**Dependencies pulled in.** This tier *is* where the long-deferred **W3-JIT** (Cranelift unwind info
+ fiber-rooted stack walk for backtraces, §5/§23) and **W6-JIT** (Cranelift value-location lists →
DWARF variable locations) finally land — they are stages here, not separate workstreams.

**Staged plan (slices — update status as they land):**

- [x] **Stage 0 — SourceLoc threading (foundation). — Built.** `svm-jit`'s `lower_block` stamps each
  emitted op with a `cranelift SourceLoc` = its `debug_info.locs` index, via a `(func,block,inst) →
  index` map (`SrcLocMap`) built in `compile` only when the module carries `-g` (threaded through
  the `Lower` struct, so the non-debug path is byte-identical). No debugger yet.
- [x] **Stage 1 — JIT address→source map + `symbolize`. — Built.** After `finalize`, each function's
  captured `MachSrcLoc` ranges (relative offsets, filtered to non-default) resolve against its
  `get_finalized_function` base into a sorted, disjoint `Vec<SrcRange>` (`[lo,hi)` machine address →
  `func`/`file`/`line`/`col`); `CompiledModule::symbolize(pc) -> Option<JitFrameLoc>` binary-searches
  it. Test (`jit_srcloc.rs`): a `-g` compute module's three body lines all map to non-empty machine
  ranges, `symbolize(range.lo)` round-trips line+file, an unmapped pc → `None`, and a non-`-g` build
  has an empty map. *Trap symbolization* (mapping a live trap pc, which the explicit-check traps
  don't capture today, vs the memory-fault signal pc) folds into Stage 4 / W3-JIT — the map and
  `symbolize` are the substrate.
- [x] **Stage 2 — `.debug_line` + `.debug_info` synthesis + GDB JIT registration (line-level
  gdb/lldb). — built.**
  - [x] **2a — `.debug_line` synthesis.** A hand-rolled DWARF v4/DWARF32 line-program emitter
    (`svm-jit`'s `dwarf` module, the inverse of `dwarf_line`) turns the Stage 1 `SrcRange` map into a
    `.debug_line` section — one self-contained sequence per range (`set_address(lo)` → set
    file/col/line → `copy` → `set_address(hi)` → `end_sequence`), so gaps never bleed a line into
    the next. `CompiledModule::debug_line_section()` exposes it. Test (`jit_srcloc.rs`): the emitted
    bytes **round-trip through `svm_wasm::dwarf_line::parse`** and reconstruct the exact
    machine-address → (file, line) map; a non-`-g` module emits nothing.
  - [x] **2b — `.debug_info`.** `dwarf::debug_info` emits a CU DIE + a `DW_TAG_subprogram` per
    function (synthesized `fnN` name, `DW_AT_low_pc` + `DW_AT_high_pc` offset form) with a matching
    `.debug_abbrev`; `CompiledModule::debug_info_sections()` derives each function's `[low_pc,
    high_pc)` as the span of its source-mapped ranges. Test (`jit_srcloc.rs`): the pair
    **round-trips through `svm_wasm::dwarf_info::parse`** to one subprogram whose `low_pc`/`high_pc`
    match the function's machine extent.
  - [x] **2c — in-memory ELF + GDB JIT registration.** `svm-jit`'s `gdb` module hand-rolls a minimal
    ELF64 (`build_elf`): an `SHT_NOBITS` `.text` whose `sh_addr` is the *live* code address (gdb
    reads the bytes from the inferior), the three `.debug_*` sections, and a `.symtab`/`.strtab`
    naming one `STT_FUNC` per function at its real `[lo, hi)`. `CompiledModule::elf_object()` builds
    it; `register_with_gdb()` wraps it in a `jit_code_entry`, links it onto `__jit_debug_descriptor`,
    and calls the `#[no_mangle]` `__jit_debug_register_code` hook (the symbols gdb knows by name),
    returning an RAII `GdbRegistration` that **unregisters on drop**. *Acceptance:* gdb/lldb binds a
    source-line breakpoint and shows the source frame — **the headline milestone** (the gdb-attach
    step is manual, not CI). CI-testable parts done (`jit_srcloc.rs`): the ELF re-parses and its
    embedded `.debug_line`/`.debug_info` **round-trip through the readers** out of the wrapper, the
    DWARF carries real finalized-code addresses, and register/drop drive the descriptor
    linked-list + `action_flag` (`JIT_REGISTER_FN` → `JIT_UNREGISTER_FN`) as gdb expects. The CU
    DIE carries `DW_AT_stmt_list` (→ `.debug_line` offset 0) — without it gdb loads the function but
    no source lines, so a `break file.c:N` never binds. ✅ **Manual acceptance confirmed:** under gdb
    15.1, `break compute.c:3` binds to the live JIT'd address (`fn0+3`) and stops there when the code
    runs (`Breakpoint 1, fn0 () at compute.c:3`); see the `gdb_attach` example
    (`crates/svm/examples/gdb_attach.rs`) for the repro harness + the exact `gdb --batch` invocation.
    *Effort: high.*
- [~] **Stage 3 — W6-JIT value locations + DWARF variables (inspect source vars). — register vars
  done; memory/CFA forms pending Stage 4.** The §6 core already carries the inputs (`DebugInfo::vars:
  Vec<VarInfo>` with a neutral `VarLoc`, and `DebugInfo::types: Vec<TypeDef>`), so this is purely a
  JIT-side *emit* problem — no IR/text change. ✅ **`print x` confirmed under gdb 15.1** (a JIT'd
  register-resident local reads back with its value + type).
  - [x] **3a — value-location substrate (W6-JIT core). — built.** Gated on `-g` vars: `compile`
    assigns each SSA-resident source var (`VarLoc::{Ssa, SsaList}`) a `ValueLabel`, calls Cranelift's
    `func.collect_debug_info()` per function, and `lower_block` `set_val_label`s the CLIF value backing
    it (block-local `SsaLoc{block,value}` indexes the JIT's per-block value map directly). After
    `define_function` it reads `CompiledCode::value_labels_ranges`, translates each `LabelValueLoc`
    (`Reg` → DWARF regnum via `isa.map_regalloc_reg_to_dwarf`, `CFAOffset` kept) and, post-finalize,
    resolves the offsets to absolute machine pcs — exposed as `CompiledModule::var_locations() ->
    [VarMachineInfo{func, name, ranges:[VarRange{lo,hi,VarMachineLoc}]}]`. Non-`-g` codegen stays
    byte-identical (no `collect_debug_info`, no labels). Tests (`jit_srcloc.rs`): a tracked local lives
    in a register over the expected code span, a folded var has empty ranges (the faithful
    `<optimized out>`), and a module without `debug.var` tracks nothing; the JIT/fiber/SIMD/c_frontend
    differential suites confirm the non-`-g` path is unchanged. (Memory forms `Window`/`WindowVia`/
    `Fixed` carry no value label — they're a DWARF memory expression in 3c.)
  - [x] **3b — `DW_TAG_*_type` DIEs. — built.** `dwarf::debug_info` now also emits the §6 `TypeDef`
    graph as type DIEs in the CU (the inverse of `dwarf_info`'s type reader): `Base` →
    `DW_TAG_base_type` (`DW_ATE_*` encoding + byte_size), `Pointer` → `DW_TAG_pointer_type`, `Array` →
    `DW_TAG_array_type` + a `DW_TAG_subrange_type` child carrying the count, `Aggregate` →
    `DW_TAG_structure_type` + `DW_TAG_member` children, `Opaque` → an empty `structure_type`.
    Inter-type references (`pointee`/`elem`/field `ty`) are CU-relative `DW_FORM_ref4`s resolved by a
    fixup pass once each type DIE's offset is known. `CompiledModule` carries the `TypeDef`s;
    `debug_info_sections()` emits them ahead of the subprograms. Tests (`jit_srcloc.rs`): the base/
    pointer/array/struct graph **round-trips through `svm_wasm::dwarf_info::parse`** with every
    `DW_AT_type` ref resolving to the right DIE, and binutils `readelf --debug-dump=info` parses it
    cleanly (refs shown as `<0x18>` → the `int` DIE). The 2b subprogram round-trip is unchanged.
  - [x] **3c — `DW_TAG_variable` DIEs + register locations. — built.** `dwarf::debug_info` now emits
    each tracked source variable as a `DW_TAG_variable` child of its subprogram (with proper
    per-subprogram null termination): `DW_AT_name`, `DW_AT_type` (a `DW_FORM_ref4` into the 3b type
    DIEs) and, for register-resident ranges, a `DW_AT_location` → a DWARF v4 `.debug_loc` location
    list (a base-address-selection entry pinning base 0 so the `[lo, hi)` are absolute, then one
    `DW_OP_reg{N}` entry per Stage-3a range). Four abbrev variants cover `{type?} × {location?}` (a
    folded var with no live range omits `DW_AT_location` ⇒ gdb shows `<optimized out>`). The ELF
    gains a `.debug_loc` section. Tests (`jit_srcloc.rs`): the loclist encodes exactly what
    `var_locations()` resolved and the DIE tree still parses; `readelf --debug-dump=loc` shows
    `DW_OP_reg4 (rsi)` over the right range. **CFA-relative ranges and the `Window`/`WindowVia`/
    `Fixed` memory forms are deferred** — they need a frame base (`DW_AT_frame_base =
    DW_OP_call_frame_cfa`) and the window-base register, which arrive with Stage 4's unwind/CFI.
  - [x] **3d — gdb acceptance. — confirmed.** Under gdb 15.1, breaking in JIT'd code and
    `print t` yields `$1 = 11` with `whatis t` → `int` (and `info locals` shows it) — gdb reads the
    source variable straight from its register location. Repro: the `gdb_attach` example (now carries
    a `debug.var`). Vars the optimizer dropped correctly show `<optimized out>` (faithful native
    behavior — the interpreter tier stays the always-faithful inspection fallback). *Effort (whole
    stage): high.*
- [x] **Stage 4 — W3-JIT unwind / backtrace. — built.** DWARF CFI (`.debug_frame`) so gdb unwinds
  JIT'd frames (`bt`) and computes the CFA (4a); spilled variables as `DW_OP_fbreg` against that CFA
  (4b); and a host-side fiber-rooted backtrace of a suspended fiber's guest stack (4c). ✅ **`bt`
  across guest frames with source and `print` of a spilled variable both confirmed under gdb 15.1;
  the per-fiber walk is CI-tested.**
  - [x] **4a — `.debug_frame` CFI + `DW_AT_frame_base`. — built.** `dwarf::debug_frame` hand-rolls a
    `.debug_frame` (one DWARF v4 CIE with the steady-state frame-pointer rules — CFA = `rbp+16`,
    return address at CFA−8, saved `rbp` at CFA−16, valid because `preserve_frame_pointers=true` gives
    every function a uniform `rbp` frame — plus one FDE per function over its `[lo, hi)`); the GDB-JIT
    ELF gains a `.debug_loc`+`.debug_frame` (now 10 sections), and every subprogram carries
    `DW_AT_frame_base = DW_OP_call_frame_cfa`. ✅ **Confirmed under gdb 15.1:** breaking in a callee
    `fn1` and running `bt` walks the **guest call stack with source** — `#0 fn1 () at two.c:9` /
    `#1 … in fn0 () at two.c:3` — across a real guest `call`. CI (`jit_srcloc.rs`): the CIE rules +
    one-FDE-per-function structure validated byte-wise; `readelf --debug-dump=frames` parses it
    (`DW_CFA_def_cfa: r6 (rbp) ofs 16`, FDE over the function). (The steady-state CIE is inexact only
    in the 1–2-instruction prologue window — fine for body breakpoints; precise per-prologue CFI would
    need Cranelift's private instruction list or `gimli`. The unwind stops at the host boundary: the
    buffer-ABI trampoline has no FDE, which is expected — the *guest* stack is what unwinds.)
  - [x] **4b — spilled (CFA-relative) variables as `DW_OP_fbreg`. — built.** With a frame base now
    defined (4a), `emit_loclist` turns a `VarMachineLoc::CfaOffset(off)` range into a `DW_OP_fbreg
    <off>` location-list entry — `frame_base + off = CFA + off`, the spill slot Cranelift stored the
    value to — while register ranges stay `DW_OP_reg`. ✅ **Confirmed under gdb 15.1:** a variable the
    regalloc spilled to the stack `print`s its value (`$1 = 30`, `whatis` → `int`) by reading the
    CFA-relative slot. Test (`jit_srcloc.rs`): a fixture of many `call` results kept live across a
    final call deterministically forces spills, and every variable range — register *and* spill — is
    reconstructed verbatim in `.debug_loc` (`DW_OP_fbreg` for the `CfaOffset` ranges). **The guest
    *window*-memory forms (`Window`/`WindowVia`/`Fixed`) remain deferred:** guest memory has no
    compile-time/frame-independent address (the window base is a per-run pointer; globals have no
    frame), so they need a different mechanism (a runtime base registered with gdb, or a JIT reader
    plugin) rather than static DWARF.
  - [x] **4c — fiber-rooted backtrace. — built.** `CompiledModule::fiber_backtrace(handle)` walks a
    **suspended fiber's** out-of-band control stack — rooted at the fiber *handle* (§23/D57 migratable
    fibers), not the OS thread. The fiber runtime exposes the parked stack via
    `SharedFiberTable::with_parked_stack` (the slot lock held, only for a fiber not running on a vCPU;
    `Fiber::ctx` is the saved SP, `parked_extent()` gives `[ctx, top)`); the walk conservatively scans
    that region low→high (innermost first) and symbolizes every word that lands in this module's guest
    code — robust to the host runtime glue between the guest frames and the suspend switch (the
    GC-root-scan precedent), no fragile frame-pointer chase. Test (`jit_fibers.rs`): a fiber entry
    calls a helper that `suspend`s; the root resumes once and returns, leaving it parked; the
    backtrace is exactly `[helper @ fib.c:9 (innermost), entry @ fib.c:5]`, and a module with no
    created fiber yields an empty walk. *Effort (whole stage): med–high.*
- [ ] **Stage 5 — DAP-over-JIT (optional; editor parity at native speed).** Either drive the JIT
  under `svm-dap` (breakpoints via software `int3` patching or single-step over DWARF line
  boundaries) so VS Code debugs native-speed code, **or** keep the interpreter as the DAP stepping
  engine and use the JIT only for speed (the two-engine question below). *Effort: high.*

**Key decisions / open questions (to resolve as stages land):**
- *DWARF writer:* hand-roll (ethos; reuses the `dwarf_line`/`dwarf_info` shapes) vs `gimli` write
  (standard but a new dep). Lean hand-roll for `.debug_line`/`.debug_info`; reconsider for
  `.debug_loclists` if the encoding gets heavy.
- *ELF builder:* a minimal hand-rolled in-memory ELF (a handful of section headers) vs the `object`
  crate. Lean minimal-hand-rolled — only `.text` + a few `.debug_*` sections are needed.
- *Registration target:* the **GDB JIT interface** (gdb + lldb, full source/DWARF) is the goal;
  `/tmp/perf-<pid>.map` (symbols only) and `perf` jitdump are cheaper symbol-only side options if a
  quick win is wanted before Stage 2.
- *Two-engine DAP (Stage 5):* interpreter-for-stepping + JIT-for-speed (cheap, already have the
  interpreter tier, sidesteps optimized-debug) vs JIT-only DWARF/DAP. The original §7 recommendation
  was interpreter-first (done); Stage 5 is where JIT-native stepping is decided.
- *Trust/TCB:* DWARF and the JIT registration are **host-side tooling**, untrusted-for-escape (§2a)
  like the rest of the waist — a malformed DWARF mis-renders, never escapes. Keep it off the
  verifier/runtime hot path.

**Progress:** Stages 0–1 **built** (source-loc threading + `symbolize`); **Stage 2 built** — 2a/2b
(the `.debug_line` and `.debug_info`/`.debug_abbrev` emitters) plus 2c (the in-memory ELF wrapper +
GDB JIT registration: `gdb::build_elf` / `CompiledModule::{elf_object, register_with_gdb}` and the
`__jit_debug_descriptor`/`__jit_debug_register_code` interface), all round-tripped through the
readers / asserted against the descriptor state in CI. The "set a breakpoint on a `.c` line in gdb"
milestone is **confirmed** — a real gdb 15.1 binds `break compute.c:3` to the live JIT'd address and
stops there (repro: the `gdb_attach` example). **Next: Stage 3** — W6-JIT value locations +
`DW_TAG_variable` DIEs (inspect source vars in gdb). Each stage is independently shippable.

---

## 8. W6 — Debug-build mode (the promotion ⊥ inspectability trade)

**Cross-cutting.** Goal: make locals inspectable as *source variables* despite SSA promotion
(§3d/§19), via the classic `-O0`-vs-optimized-debug switch.

**Design sketch.** A frontend/build flag with two strategies (both recorded in §19):
- **`-Og`/disable-promotion**: keep address-taken-or-debug locals in the window (addressable,
  trivially inspectable) at a speed cost. Simplest; pairs with W4 `VarLoc::WindowSlot`.
- **value-location lists**: keep promotion, emit Cranelift value-location lists so the debugger
  finds the register/stack slot. Harder; required for debugging *optimized* code.

**Dependencies.** W4 (`VarLoc`). Consumed by W2 (interpreter inspection) and W5 (DWARF vars).

**Open questions.** Whether the interpreter (which holds SSA values explicitly) makes the
disable-promotion mode unnecessary *for interpreter debugging* — likely yes, which lets us ship
source-variable inspection on the interpreter (W2+W4) before solving value-location lists.

**Effort / risk.** Disable-promotion: **low.** Value-location lists: **high** (couples to W5).

**Acceptance.** In debug-build mode every source local is inspectable by name on the
interpreter; in optimized mode value-location lists resolve the live ones, unavailable ones are
honestly reported as `<optimized out>`.

---

## 9. W7 — Concurrency-debugging surfacing

**Mostly built; needs ergonomics.** Goal: expose the model checker and scheduled replay as
first-class debugging tools, and stage the runtime race detector.

**Current substrate.** `explore_all` (exhaustive DPOR; returns `Exhaustive { outcomes,
schedules, complete }`), `explore_all_bruteforce` (the soundness oracle), `run_scheduled(seed)`.
These are *test-suite* entry points today, not a user-facing debugging surface.

**Design sketch.**
- **CLI/Inspector verbs** over the existing functions: "check this entry for races/deadlocks/
  assertion failures across all schedules" (`explore_all` + report which schedule produced a bad
  outcome), and "replay schedule N / seed S" (`run_scheduled`) to drop into W2 stepping inside a
  failing interleaving. The schedule that produced a bad `outcome` is reconstructable from the
  explorer's plan — surface it as a replayable handle.
- **DRF-or-trap hardened tier** (§12, designed-not-built): an optional §5 instrumented JIT tier
  that traps on a data race at runtime (TSan-class cost) — runtime detection on the fast path,
  complementary to `explore_all`'s exhaustive interpreter exploration. Separate, later track.

**Dependencies.** Surfacing: none (functions exist) + optionally W2 (step the bad schedule) and
W1 (persist it). DRF-tier: substantial standalone JIT work.

**Open questions.** What the "bad schedule → replayable handle" artifact is (a `SchedTape`,
W1) and how to present a minimal failing interleaving (DPOR already visits ~one schedule per
Mazurkiewicz trace, so the witness is already near-minimal).

**Effort / risk.** Surfacing the built checker: **low.** DRF-or-trap tier: **high**, deferred.

**Acceptance.** A one-command "model-check this concurrent entry" reports the set of outcomes
and, on a failure, hands back a schedule that `run_scheduled` reproduces and W2 can step.

**Built — slice 1 (witness find + replay).** `svm-interp` now exposes the model checker as a
debugging tool: `find_schedule(m, func, args, fuel, max, pred) -> Option<Witness>` model-checks
across interleavings (DPOR) and returns the **first** schedule whose outcome matches `pred`
(deadlock / trap / specific bad result) as a replayable `Witness { plan, outcome,
schedule_index }`; `replay_schedule(m, func, args, fuel, &plan)` re-runs that exact interleaving
deterministically (the W7 → W1 bridge). Implemented by extracting the DPOR loop into a private
`explore_core` with an `on_outcome(idx, &outcome, &plan)` callback that `explore_all` (collect the
outcome set) and `find_schedule` (capture a witness, stop early) share; the witness is the
executed `trace` tids, and replay forces them through `Dpor::pick`. Tests (`concurrency_debug.rs`,
the racy lost-update counter): a `→ 1` witness is found and replays deterministically 5×, the
serial `→ 2` witness replays, and an impossible outcome returns `None`. Full workspace green.

*Not yet:* CLI verbs over these functions; stepping a witness interleaving (needs the
multithreaded `Inspector` — Milestone B, the `Policy` scheduler seam); the DRF-or-trap tier.

---

## 10. W8 — `Inspector` / `Monitor` host capability

**Cross-cutting shell.** Goal: the host-side capability object every other workstream's surface
hangs off — shaped like §15 `Monitor`, observe-only, never widening guest authority.

**Current substrate.** The §15 metering *properties* exist (`Host::set_quota`/`quota`, fuel),
but no `Monitor`/`Inspector` *type*. The `Host` already exposes a rich `grant_*` surface and an
async-notify hook — the right place to anchor an observer.

**Design sketch.** An `Inspector` host object holding a reference/handle to a guest domain,
exposing the read-only/control verbs W1–W3/W7 need (breakpoints, watchpoints, step, backtrace,
read window/locals, record/replay control, model-check/replay). It is **not** a guest-callable
capability by default — it is a *host* capability (the embedder/debugger holds it), consistent
with "debugger observes from outside." Nesting (§14) makes a parent a natural debugger of a
child.

**Dependencies.** None upstream; it is the integration point. Build the shell first so W1/W2/W3
land verbs onto it incrementally.

**Open questions.** Whether an `Inspector` is ever delegated *into* a guest (self-debugging /
guest-built tooling) — allowed by the ocap model (it grants no new authority) but out of scope
for v1. Revocation interacts with §7's parked revocation item.

**Effort / risk.** **Low-moderate** for the shell; grows with the verbs it carries.

**Acceptance.** A host can attach an `Inspector` to a running interpreter domain and drive W2
verbs through it; attaching/detaching never changes guest-observable behavior.

---

## 11. Recommended sequencing

Two tracks; the **cheap, interpreter-rooted source+stepping loop** first, the **expensive
production-grade pieces** staged behind proof on the interpreter. Gated on a **Milestone 0**
design pass that fixes the shared debug core (§2a) so the bodies don't pinch.

**Milestone 0 — debug-core design pass (paper, little/no code):** decide S1 (location model),
S2 (`VarLoc`), S3 (logical-time clock), S4 (interpreter instrumentation seam), S5 (Inspector
control model), S6 (Cranelift emission layer), and the S7/S8 invariants; resolve the
cross-cutting decisions (§12) that gate them (D-DBG-3/4/6 especially). Cheap, and it prevents
the rework §2a identifies. **S1–S5 are drafted in §13** (S4/S5 pinned S1/S3; S2 follows from the
frontend's local classification); only S6 (JIT-tier) remains.

**Milestone A — "debug a single-threaded guest on the interpreter" (cheap, high value):**
1. **W8 shell** (the capability to hang verbs on).
2. **W2 stepping/breakpoints/watchpoints** (IR-level) + interpreter backtrace (W3a).
3. **W4 §3a debug-info side-table** + frontend threading.
4. Wire W4 into W2 → **source-level stepping on the interpreter** (no DWARF yet).

**Milestone B — "debug a multithreaded guest" (the headline differentiator):**
5. **W7 surfacing** of `explore_all`/`run_scheduled` as debugging verbs.
6. **W1 sequential `CapTape`** record/replay + time-travel `seek` on the interpreter.
7. Schedule-replay handle (W1 `SchedTape` from the explorer) → step a failing interleaving.

**Milestone C — "production-grade, staged":**
8. **W3 JIT trap-time backtraces** (improves every kill message).
9. **W6 debug-build mode** + **W5 interpreter-backed DAP** (VS Code), then JIT/DWARF tier.
10. **DRF-or-trap** hardened tier (W7, standalone).

Rationale: Milestone A delivers a usable source debugger entirely on existing interpreter
structures with no backend/ABI risk; Milestone B exploits the already-built DPOR/replay
substrate that makes *multithread* debugging this project's standout capability; Milestone C is
the genuinely expensive, deferrable production tooling.

---

## 12. Open cross-cutting decisions

- **D-DBG-1 — JIT schedule capture (W1):** record on the interpreter only (recommend) vs
  vector-clock at sync ops vs full capture behind the DRF tier. Determines whether multicore
  *production* replay is in scope or interpreter-replay is the supported story.
- **D-DBG-2 — DAP engine (W5):** interpreter-backed DAP first (recommend) vs JIT/DWARF first.
- **D-DBG-3 — debug-build default (W6):** disable-promotion `-Og` as the default debug build
  (recommend) with value-location lists as the optimized-debug tier.
- **D-DBG-4 — debug-info location (W4):** text-first per D33 (recommend) vs binary-only.
- **D-DBG-5 — `Inspector` delegation (W8):** host-only for v1 (recommend); guest-delegable
  self-debugging deferred.
- **D-DBG-6 — metering-pause semantics (S8/W2/W8):** how a guest stopped at a breakpoint avoids
  §5's undisableable fuel/epoch kill without reopening the runaway-guest hole. Options: a
  host-only "inspector-paused" state that stops the fuel clock only while an `Inspector` holds
  the guest (recommend), vs. a wall-clock grace that still bounds total stopped time.
- **D-DBG-7 — debug info is a frontend-neutral IR waist (W4): SETTLED (design).** A **narrow
  waist** at the IR: a small mandatory neutral core (`IrPc→SourceLoc`, `VarInfo`+`VarLoc`,
  neutral `TypeRef`) every frontend populates *during lowering*, plus an optional **opaque
  per-producer rich blob** (DWARF/LLVM-DI) the middle never parses and only W5/DAP consume.
  Rejected: a chibicc-shaped table (forces lossy down-conversion from LLVM/wasm), and DWARF-as-
  interchange (forces our structural `IrPc` into linear DWARF + makes the interpreter parse
  DWARF). chibicc emits only the core; LLVM/wasm map their DWARF-shaped info in (LLVM's
  `dbg.value` solves S2 promotion-vs-inspectability for free) and pass their native blob through.
  Full design + per-frontend mapping in §6. Implement chibicc-core first; freeze the schema now.

When these are settled, fold the resolved ones into `DESIGN.md` §19 / the decision log as
`D54+` so DESIGN stays the canonical record. (D-DBG-7's waist is the debug-info analog of D33's
"IR is the stable target" — worth a DESIGN §19/§20 cross-reference when it lands.)

---

## 13. Milestone-0 designs — S4, S5 (+ S1/S3 pinned, S2)

Detailed pass of the shared-core items (§2a) on the **interpreter path**. Grounded in the
interpreter as built (`crates/svm-interp/src/lib.rs`) and the frontend (`codegen_ir.c`); line
refs are to the state on this branch. Designing the two highest-leverage items (S4, S5)
**pinned S1 and S3** as a consequence (see "Cascade"), and S2 (`VarLoc`) follows from the
frontend's existing local classification — so **five of the six core items are settled here**;
only S6 (Cranelift value-locations) remains, and it is JIT-tier (W5/W6).

### S4 — interpreter instrumentation seam

**Key finding: the seam already exists, hard-wired to DPOR.** The interpreter has two extension
points the debug hooks should *widen* rather than replace:

1. **Scheduler seam** — `run_with_policy` (`lib.rs:1691`) + the `Policy` enum (`lib.rs:1675`)
   already choose *which vCPU, what quantum*. `Policy::Dpor(plan)` / `Seeded` are already
   plan/seed-driven schedule control.
2. **Per-op seam** — inside `run_inner` (`lib.rs:2396-2424`), the `memop`/`is_visible`/`acc`/
   `budget` block is already a per-op "observe this op, record what it touched, optionally
   yield" hook — just bound to DPOR and visible-ops-only.

**Design.**
- Generalize `VCpu::memop: bool` (`lib.rs:2081`) → `obs: ObsMode ∈ {Off, Memop, Debug}`.
  `Off` = today's hot path byte-for-byte (the `else` at 2416); `Memop` = today's DPOR; `Debug`
  = consult a probe per op. The single-discriminant gate is the shape the loop already pays, so
  **S7** (behavior-preserving, differential-safe) holds — the differential harness runs `Off`.
- **Per-op hook** at the existing decision point (2402-2424): before executing, build the
  context the loop already has in scope — `cx = { vcpu_id, fiber: cur, ir_pc: (module, func,
  block, inst), mem }` — and call `probe.before_op(cx) -> Flow`. `Flow::Run` continues;
  `Flow::Pause(reason)` returns a **new `Inner::Pause(Stop)` → `Step::Pause`** variant, sibling
  to the existing `Inner::Yield` (2405). A `VCpu` is already a self-contained, movable
  continuation (Frames hold no borrows, `lib.rs:922`), so "pause" = "stop pumping, hand the
  VCpu back."
- **Watchpoints** reuse `access_of` (`lib.rs:496`) — it already computes the confined address
  for visible ops; extend to loads/stores generally and range-check the watch set. "Break when
  any fiber writes `addr`" is one check in the masked access path (the window is one buffer).
- **Schedule record/replay** is a new `Policy` variant — `Policy::Record(&mut SchedTape)` /
  `Policy::Replay(&SchedTape)` — structurally identical to `Dpor(plan)`. No new seam; W1's
  schedule tape and W7's replay both ride `Policy`.
- **S8 (metering-pause)** falls out: `step(fuel)?` (`lib.rs:2423`) is the only fuel decrement
  and it is *inside* the pump. A paused guest is one the driver isn't pumping, so fuel can't
  advance and the undisableable preemption (a scheduler-loop property) still governs every
  *unpaused* guest — the host-only "inspector-paused" state of D-DBG-6, with no hole.

### S5 — Inspector control/session model

**Key constraint: a driver, not a callback** — forced by the interpreter being single-OS-thread
cooperative (`run_with_policy` pumps vCPUs, returning at Yield/Park/Done). The `Inspector` *owns
and pumps* the run, regaining control at stop points:

```
inspector.run_until_stop() -> Stop { reason, fiber, ir_pc }
    reason ∈ { Breakpoint, Watchpoint{addr}, Step, CapCall, Trap, Exit, SchedulePoint }
// stopped: backtrace(fiber) / read_window(a,len) / read_ir_value(id) / locals()
// loop: run_until_stop()
```

One verb subsumes all four control models S5 had to span:
- **W2 stepping** — probe pauses after one op / at a breakpoint.
- **W1 record** — probe logs a `CapTape`/`SchedTape`; stop only at `Exit`.
- **W1 replay / time-travel** — built with `Policy::Replay(tape)` + `CapTape`; `seek(t)` =
  stateless re-run from the nearest checkpoint to logical time `t` (what `explore_all` already
  does per schedule).
- **W3 read** — when stopped, read frames/window/values.
- **W7 many-runs** — a higher verb `model_check() -> Exhaustive` wraps `explore_all`; on a bad
  outcome it returns a `SchedTape` the *same* Inspector can `replay()` then step (the
  W7→W1→W2 bridge in one object).

**Honest boundary: the driver model is interpreter-only.** The JIT runs real OS threads and
can't be pumped op-by-op, so it gets a thinner, separate `JitInspector` profile — attach +
read-at-stop (trap-time backtrace W3, async-notify observation), point-in-time only, no
stepping. DAP (W5) binds to the interpreter `Inspector`; the JIT profile is for production trap
diagnostics. This *is* "interpreter is the debug engine, JIT is production," made concrete.

The session is host-side and **observe-only** (S5/S7): it holds run state, never guest
authority; attach/detach under `ObsMode::Off` is behavior-identical.

### Cascade — S4/S5 determine S1 and S3

- **S1 (location model)** = `IrPc { module, func, block, inst }`, per-op granularity — exactly
  the tuple the per-op hook has in scope (`lib.rs:920-936`). Source mapping stays deferred to W4.
- **S3 (logical-time clock)** = the probe's monotonic **event index** — because `seek`,
  step-back, and `SchedTape` keys must all reference the same stream the probe emits.

So S1, S3, S4, S5 are settled by this pass; with **S2 drafted below**, the only remaining
Milestone-0 item is **S6 (Cranelift value-locations)**, which is JIT-tier (W5/W6).

### S2 — value-location model (`VarLoc`)

**The frontend already classifies every local two ways** at lowering (`codegen_ir.c`,
`is_promoted(v) = v->offset < 0`, line 189) — S2 only fixes how that classification is
*recorded* for inspection:

- **Memory local** — address-taken, narrow (`char`/`short`/`_Bool`), or array/struct/union → a
  window **data-stack slot** at `sp + offset` (non-negative `offset`). Address = the data-SP
  (block param `v0`) `+ offset`, **constant over the function**.
- **Promoted local** — never-address-taken full-width scalar → a real **SSA value**, threaded as
  block parameter `v(s+1)` of every block; the slot's current value is tracked per block in
  `curval[s]` and is **reassigned within a block** on each write (`codegen_ir.c:174-185`).

Two interpreter shapes, plus a deferred JIT shape:

```
enum VarLoc {
    Window { off: i32 },     // addr = vals[0] (data-SP) + off; constant over the function
    Ssa(LocList),            // promoted: PC-keyed list of which SSA value holds the slot
    // Machine(CraneliftValueLocList)   // JIT-optimized; deferred to S6/W5
}
struct LocList(Vec<(IrPcRange, ValueIdx)>);   // IrPcRange keys on S1's IrPc
```

**Resolution at an `IrPc` is trivial on the interpreter**, which holds every block-local SSA
value by index in `Frame::vals` (`lib.rs:938`):
- `Window{off}` → read `Frame::vals[0]` (data-SP), add `off`, read `len` window bytes. Constant.
- `Ssa(loclist)` → find the range covering the `IrPc`, read `Frame::vals[value_idx]`. **No
  Cranelift machinery, even in optimized builds.**

Three consequences:
1. **List, not a single id** — a promoted slot's SSA value changes within a block (each write
   makes a fresh value, updates `curval[s]`), so "where is `x` at this op" varies by PC — the
   DWARF location-list problem. The frontend *already computes* `curval[s]` per op, so emitting
   the list records what it knows; W4 packages it, S2 fixes only the shape.
2. **The interpreter sidesteps W6** — `Ssa` resolves straight from `Frame::vals`, so
   interpreter source-variable inspection needs **no** debug-build / disable-promotion mode
   (confirming the W2/W6 note). W6 and `VarLoc::Machine` matter only for debugging
   *JIT-optimized* code, where a promoted local is in a Cranelift register/stack slot (S6/W5;
   honest `<optimized out>` where unavailable).
3. **The common case is the easy one** — only never-address-taken full-width scalars hit `Ssa`;
   every struct/array/union and every address-taken or narrow local is `Window`. "Inspect this
   struct" is always the constant-location path; location-list complexity is confined to the
   minority of hot promoted scalars.

This pins S2 against S1 (`IrPcRange` keys on `IrPc`) and **closes the interpreter-path core:
S1–S5 settled, only S6 (JIT-tier) remains.**

### Built — Milestone A slice 1 (`svm-interp::Inspector`)

First implementation landed against these designs (`crates/svm-interp/src/lib.rs`, tests in
`crates/svm/tests/debug.rs`):

- **S4 seam** — `VCpu` gained `debug: Option<Box<DebugCtx>>`; the per-op hook in `run_inner`
  consults `DebugCtx::before_op(IrPc)` and returns the new `Inner::Pause`/`Step::Pause` on a
  hit. `None` is the untouched hot path (S7); the scheduler/coroutine paths assert the pause is
  unreachable (only an `Inspector`-driven vCPU carries `debug`).
- **S5 driver** — `Inspector::attach` → `run_until_stop` / `step`, with `set_breakpoint` /
  `clear_breakpoint`, `backtrace`, `read_ir_value` (S2 `Ssa` resolution straight from
  `Frame::vals`), and `read_window` (S2 `Window` resolution via a new `Mem::read_window`).
- **S1/S3 confirmed in code** — `IrPc { module, func, block, inst }`; `clock` = ops executed
  (non-terminator granularity — terminators live in `Block::terminator`, not `insts`, so they
  are not step points).

Five tests cover run-to-completion transparency, per-iteration breakpoints with value reads, a
single-step that advances exactly one op + ticks the clock, a two-frame backtrace inside a
callee, and a window read-back. Full workspace suite stays green.

**Slice 2 — watchpoints.** `set_watchpoint(addr, len, WatchKind)` / `clear_watchpoint(id)` watch
a window range for `Read`/`Write`/`ReadWrite` accesses, reported as `StopReason::Watchpoint {
addr, write }` *before* the accessing op (step once to apply it). The hit test reuses
`access_of` — the same confined-range analysis the DPOR explorer uses — computed in the hot loop
**only when a watch is armed** (it confines, so it isn't free); breakpoints/stepping skip it.
Because the window is one contiguous buffer, a watch catches every code path with no per-op
instrumentation. Four more tests: stop-before-store then step-applies, read/write kind filtering,
clear, and non-overlapping range. Workspace green, clippy clean.

**Slice 3 — capability-using guests + the cap.call boundary stop.** `Inspector::attach_with_host`
takes a caller-prepared `Host` (the powerbox): `grant_*` the capabilities, pass their handles in
`args`, and debug a real capability-using guest (§3c/§3e). `host()` locks the (uncontended-while-
paused) powerbox to read effects — captured stdout, clock, grants. `set_cap_call_stops(true)`
pauses *before* every `cap.call` with `StopReason::CapCall { type_id, op }` (the handle/args are
live; `step` to perform it) — the §7 host boundary and the future W1 record/replay hook (S5). The
module must be import-resolved (`svm_run::resolve_capability_imports`) per the new named-import
model on `main`; the interpreter runs only concrete `cap.call`s. Three tests: end-to-end stdout
capture, a boundary stop with the effect deferred until `step`, and the toggle defaulting off.

**Slice 4 — W4 debug-info waist, neutral core (text).** The frontend-neutral waist (D-DBG-7/§6)
landed as `svm_ir::DebugInfo { files, locs, vars }` on `Module::debug_info: Option<DebugInfo>`,
with `VarLoc ∈ { Window{off}, Ssa{value} }` (= S2). The **text** form round-trips it
(`debug.file` / `debug.loc` / `debug.var` directives, `svm-text`); the binary form stays
debug-stripped for now (like the import-free rule — a follow-up). The verifier never reads it
(§2a). The `Inspector` consumes it: `source_loc(IrPc) -> SourceLoc`, source-enriched `backtrace`
frames, and `read_var(frame, name, width) -> VarValue` (the W4→S2 bridge — `Ssa` reads
`Frame::vals`, `Window` reads `data-SP + off`). Three tests: source location + named-var reads at
a breakpoint, text round-trip (incl. a window var), and the no-debug path. *Slice-1 limits (noted
for later):* `VarInfo` is function-scoped with a single `VarLoc` (proper SSA vars need S2's
per-block `LocList`); `ty` is a render-name string (structured `TypeRef` later); no per-producer
rich blob yet. **Frontend emission is separate:** chibicc populating the core from its `Token`s,
and the LLVM/wasm ingest sides, are their own slices targeting this same waist.

**Slice 5 — chibicc `-g` emission (the producer side, end-to-end).** `chibicc --emit-ir -g`
now emits the §6 waist from real C: a `debug.var` per named local mapping its C name to a
`VarLoc`. At first, **`-g` was `-Og`** — it disabled SSA promotion (the W6 / §19 debuggable-vs-
optimized trade), so every local kept a *stable* window data-stack slot resolving function-wide as
`VarLoc::Window{off}` (promoted scalars would need S2's per-block `LocList`). **Superseded by slice
17**, which keeps promotion and emits a location list instead — debugging the optimized build. `main.c` gains a 2-line `-g` flag
(intercepted before chibicc's generic `-g*`-ignore block); everything else is in `codegen_ir.c`
(the project's "ours" file). End-to-end test (`c_frontend.rs`, behind the unix toolchain gate):
compile C → parse → `Inspector` reads `s`, `t`, `a`, `b` by their **C names** with correct
values. Default (non-`-g`) output is unchanged, so all 78 existing c_frontend tests are
untouched. `debug.loc` (per-op source lines) needs per-op inst counting in the emitter — a later
slice.

**Slice 6 — `debug.loc` source lines (chibicc).** `chibicc -g` now also emits `debug.loc`
rows, so breakpoints and backtraces resolve to `file:line`, not just named variables. The
emitter routes every IR line through one sink (`cg(...)`) that counts `block.insts` entries (a
two-space-prefixed, non-terminator line), so source locations key on `(func, block, inst)`
*exactly* — no fragile heuristics. Locations are recorded per statement (in `gen_stmt`), and the
interpreter's `source_loc` uses **nearest-preceding within `(func, block)`** (DWARF line-table
semantics). `backtrace` frames carry the resolved `SourceLoc`. Tests (`c_frontend.rs`): a
breakpoint at the last block op resolves to the `return` line, and multi-block control flow
(a `for` loop) maps each block to its source line. The full source-level loop — *set a breakpoint,
see the line and the named locals* — now works end-to-end on real C.

**Slice 7 — structured `TypeRef` ABI + chibicc emission.** The debug-info waist gained a
structured type table (`DebugInfo::types`, referenced by `VarInfo::type_id`), so a variable
carries its **layout** — struct field offsets, array element + count, pointer pointee — and not
just a render-name string. `TypeDef ∈ { Base, Pointer, Array, Aggregate{fields}, Opaque }`; the
text form adds `debug.type` / `debug.field` and an optional trailing id on `debug.var`, with the
old name-only `debug.var` still valid (back-compat: `type_id = None`). chibicc `-g` interns each
named local's C `Type` (by pointer identity to bound recursive aggregates, plus structural dedup
of base scalars) and emits the table. This is the **producer + ABI** half (the §7 W5 note). The
binary section was text-only until slice 14 (below); render names were generic until slice 12
(below) gave them the C tag.

**Slice 8 — DAP Variables-pane aggregate expansion (first consumer of the type table).** A
struct/array local expands in the editor: the server hands back a nonzero `variablesReference`
naming a `Place` (thread + window address + `TypeId`), and a `variables` request on it lists the
fields/elements — recursively for nested aggregates, scalar leaves read from the window and
formatted per `TypeDef`. Full design in §7.

**Slice 10 — `evaluate` member / index / arrow.** Completes the `evaluate` half: `evaluate` resolves
`a.b` / `arr[i]` / `p->x` (and mixes like `p.x + arr[i]`) over the structured types. The `expr`
evaluator grew a frontend-agnostic `Resolver` trait + a `Value` (`Int | Place`); `svm-dap`'s
`EvalEnv` implements the navigation as address arithmetic over `TypeDef` + window reads. Full
design in §7.

**Slice 11 — pointer-deref expansion (Variables pane).** A pointer local/field is now expandable:
its row shows the address (`0x…`, read from the window) and expands to a single synthetic `*`
child — the pointee at the dereferenced address, itself expandable if it's a struct/array/pointer.
A null pointer shows a `<null>` leaf, an unreadable one `<unreadable>` (lazy, so a self-referential
`struct Node { Node *next; }` walks one level per click). `is_expandable` now covers `Pointer`, and
an expandable row's summary comes from a memory-aware `place_summary`. Test (`dap.rs`): a `pp`
→ `struct Point *` expands `pp` (`0x410`) → `*` (`{...}`) → `x=7, y=9`.

**Slice 12 — richer type render names.** Derived types now render readably: `struct Point` (the C
tag, carried on a new `Type::tag` field in chibicc), `int *`, `int[4]` — built compositely in
`dbg_typename`, so both the `debug.var` name and the type-table name show the type a programmer
would write rather than a bare kind. Anonymous structs stay `"struct"`.

**Slice 13 — `evaluate` floats + short-circuit `&&`/`||`.** The `expr` evaluator gained float
literals (`1.5`, `2e3`), a `Value::Float`, and C numeric promotion (an internal `Num`: arithmetic
and comparisons run in `f64` when either side is float; `% << >> & | ^ ~` stay integer-only).
`Resolver::coerce_int` became `load` (a scalar `Place` reads as `Float` for a float base type, else
`Int`). `&&`/`||` now **short-circuit**: the dead operand is parsed (for position) but evaluated
with a `live = false` flag that suppresses resolver calls and errors — so `b != 0 && a/b > 0` no
longer trips integer division-by-zero. The bare-variable `evaluate`/Variables scalar path also now
formats window bytes through the structured type, so a `double` reads as `2.5`, not its bit
pattern. Tests: `expr` unit tests (promotion, short-circuit) + a DAP test over a window `double`.

**Slice 14 — binary serialization of the debug section.** `svm-encode` now encodes/decodes the
full `DebugInfo` (files, locs, the structured type table, vars) as a strippable section appended
after the funcs, so debug info survives the binary IR form, not just text. It's **append-only and
back-compatible**: a module with no debug info encodes byte-identically to before (the decoder
treats "no bytes after the funcs" as `None`), so existing blobs and `svm-snapshot` digests are
unchanged — verified by a prefix assertion. The decoder keeps the untrusted-input discipline
(bounded counts, UTF-8-checked strings, validated discriminants → typed `DecodeError`s) and the
verifier still ignores the section (§2a). Tests (`svm-encode`): a module exercising every
`TypeDef` variant + `VarLoc` + `type_id` round-trips; the no-debug case stays a byte-prefix; a
corrupted section fails to decode without panicking.

**Slice 15 — wasm DWARF → `debug.loc` (the second producer; frontend-neutrality demonstrated).**
`svm-wasm` now ingests a guest's embedded DWARF `.debug_line` and maps it onto the **same neutral
`DebugInfo` waist** chibicc populates — so a *second*, independent frontend produces source
locations the interpreter/DAP consume unchanged. A small hand-rolled DWARF v2–v4 line-program
reader (`dwarf_line.rs`, no `gimli` dep — matching the crate's lean ethos) decodes the
`(address → file, line, col)` rows; the translator threads each wasm operator's byte offset and
records where its first IR instruction lands, so a line row's code-relative address resolves to an
IR pc (`func, block, inst`). The wasm-DWARF address convention (offsets relative to the code
section content) is handled by subtracting the code section's start. Best-effort and strippable:
malformed DWARF ⇒ no debug info, and the verifier ignores the section (§2a). Test (`debug_line.rs`):
a clang-built `dline.c` wasm fixture transpiles with the C source's body lines (2, 3) mapped to
in-range IR pcs, and still verifies. *Scope:* source lines only — variable ingest needs the
per-block `LocList` (wasm/LLVM locals are SSA values that vary by block), and the LLVM bitcode
metadata path is its own slice.

**Slice 16 — per-block `LocList` ABI + interpreter resolution (S2, the location-list case).**
`VarLoc` gained an `SsaList(Vec<SsaLoc>)` variant: a DWARF-style location list where each entry
(`block, inst, value`) says "from this pc onward within the block, the var is this block-local SSA
value". The interpreter's `read_var` resolves it **nearest-preceding within the stopped block**
(like `source_loc`), so a promoted scalar whose holding value changes — across a block boundary
(its block-param index differs per block) or mid-block (a reassignment makes a fresh SSA value) —
reads correctly at any stopped pc. This is the representation SSA-valued locals need: it unblocks
**wasm/LLVM variable ingest** and lets **chibicc debug without `-Og`**. The text (`debug.var …
ssalist <n> <b> <i> <v>…`) and binary forms round-trip it; `Window`/`Ssa` are unchanged
(back-compat). The DAP `EvalEnv` resolves an `SsaList` name through `read_var` at the frame pc.
Tests: an interpreter test reads `x` at three pcs (block boundary + intra-block transition) getting
the right value each time; text + binary round-trips. *This is the ABI + consumer; producers
emitting location lists (chibicc promotion, wasm locals) are follow-up slices, per the slice-4→5
pattern.*

**Slice 17 — chibicc emits location lists (debug the optimized build; no more `-Og`).** `-g` no
longer disables SSA promotion (superseding slice 5's `-g = -Og`): a promoted scalar now keeps its
SSA value and `-g` emits a **location list** (`debug.var … ssalist …`) recording the holding
block-local value as it changes — across blocks (a fresh block parameter each block, e.g. a loop's
iteration variable) and on each write. The emitter records every `set_curval` (block entry, the
entry block's param/zero-init bindings, and writes) as a `(func, slot, block, inst, value)` and
emits the list; memory locals still emit `win`. So you debug the **actually-optimized** code — the
value-location-list tier of the W6/§19 trade — and the interpreter resolves a var to the right SSA
value at each pc. (Honest optimized-debug consequence: a variable assigned by a block's *last* op
is only live at the following step point, since the interpreter breaks before an op, not at the
terminator.) Tests (`c_frontend.rs`): named locals read by C name at a breakpoint (now via
`ssalist`), source-line mapping, structured types, and a **loop accumulator** whose `(i, acc)` read
correctly across iterations — chibicc debugging promoted code end-to-end.

**Slice 18 — per-producer rich blob (the §6 waist's opaque half).** `DebugInfo` gained
`blobs: Vec<ProducerBlob{producer, bytes}>` — a frontend's **native** debug info carried through the
IR verbatim (the middle never parses it; only a future DWARF/DI re-emitter (W5) will; the verifier
ignores it, §2a). Text (`debug.blob "<producer>" "<escaped bytes>"`, reusing the data-segment byte
escaping) and binary forms round-trip it (incl. non-UTF-8 / NUL bytes); the no-debug encoding stays
a byte-prefix. The **wasm** producer now passes every embedded `.debug_*` section through as a blob
(tagged by section name), so a guest's full DWARF survives transpilation — and `.debug_info` (the
variable-bearing section the core doesn't yet parse) is preserved for later. This freezes the §6
schema's rich-blob slot, as the doc recommended. Tests: wasm carries `.debug_info`/`.debug_line`
blobs verbatim; text + binary round-trips; a truncated debug section errors without panicking.

**Slice 19 — wasm `.debug_info` DIE reader (variable-ingest foundation).** A small hand-rolled
DWARF v2–v4 (DWARF32) `.debug_info` reader (`dwarf_info.rs`, no `gimli` — matching the line reader)
parses `.debug_abbrev` (abbreviation tables) + `.debug_info` (the DIE tree) + `.debug_str`,
recovering per `DW_TAG_subprogram`: its PC range, its **frame base** (a wasm local, from
`DW_OP_WASM_location 0x0 <n>`), and its parameter/variable children — each a `(name, DW_OP_fbreg
offset, type DIE)` — plus `DW_TAG_base_type` DIEs by offset. Best-effort (malformed/unsupported ⇒
no vars; the verifier ignores it, §2a). This grounds the variable-ingest location model against real
clang output: the wasm DWARF describes a var as a **C-frame memory** location `(frame_base + fbreg)`
= a window address whose base is a *runtime* wasm-local value, so resolving it needs the local's SSA
value per pc (an `SsaList`-style lookup) plus the offset — a forthcoming `VarLoc` variant. Test
(`debug_line.rs`): the reader recovers `add`'s frame-base local 4 and its `a/b/s` at fbreg +12/+8/+4
with the `int` base type, from the committed fixture.

**Slice 20 — `VarLoc::WindowVia` (the wasm/DWARF location model).** A variable in **window memory at
a runtime base + offset**: `WindowVia { base: Vec<SsaLoc>, off }` resolves `base` as a location list
per pc (nearest-preceding within the block, like `SsaList`) to a frame value, reads that as a
window address, adds `off`, and reads `width` bytes. This is the `DW_OP_fbreg <off>` case from slice
19 — the frame base is a wasm local (an SSA value here), not a fixed `data-SP` slot. (`Window{off}`
is the special case where the base is always frame value 0.) The interpreter `read_var` resolves it
(refactored to share the loclist lookup with `SsaList`); text (`debug.var … winvia <n> <b> <i> <v>…
<off>`) and binary forms round-trip it; DAP's `evaluate` reads such a name through the Inspector.
Tests: an interpreter test stores a value at `data-SP+8` and reads it back through a `winvia` whose
base is value 0; text + binary round-trips (incl. a negative offset).

**Slice 21 — wasm variable ingest, end-to-end (a second producer feeds the *variable* half).** A
clang-compiled wasm guest's source variables now land in the §6 waist as named `debug.var`s the
interpreter/DAP read by name. Two pieces: (a) the lowering records each **wasm local's SSA value per
pc** (`(local, block, inst, value)` at block-entry re-threading + `local.set`/`tee`) — the
`SsaList` a frame-pointer local needs; (b) `build_debug_info` parses `.debug_info`
([`dwarf_info`]), matches each subprogram to its IR function by PC range (via the op offsets), and
for each variable emits a `VarLoc::WindowVia { base = the frame-base local's recorded `SsaList`, off
= DW_OP_fbreg }` plus a structured `TypeRef` from the `DW_TAG_base_type`. So the DWARF "C-frame
memory at `frame_base + fbreg`" resolves at runtime to the actual window address (the frame pointer's
value per pc, + offset). Test (`debug_line.rs`): the fixture's `a/b/s` are ingested as `WindowVia`
vars at fbreg +12/+8/+4 with the `int` type, into the right IR function, and the module still
verifies — **frontend neutrality demonstrated for variables, not just source lines**.

**Slice 22 — DAP treats `WindowVia` as a first-class window location.** A new
`Inspector::var_addr(frame, name)` resolves a memory-located var (`Window` *or* `WindowVia`) to its
window address at the stopped pc (factored out of `read_var`, sharing the `loclist_value`
nearest-preceding lookup). DAP now uses it uniformly: a `WindowVia` aggregate **expands** in the
Variables pane (struct fields / array elements / pointer deref), `evaluate` returns a **typed
`Place`** for it (so `p.x` / `arr[i]` / `p->x` work over wasm vars), and the scalar Variables path
formats through the structured type (a window `double`/pointer reads typed, not as raw bytes). So a
wasm guest's variables — ingested as `WindowVia` in slice 21 — are now fully inspectable, the same
as chibicc's. Test (`dap.rs`): a `WindowVia` `struct {int x,y}` expands to `x=10, y=20` and
`evaluate("p.x + p.y") = 30`.

**Slice 23 — wasm aggregate / pointer / array type ingest (the structured-type half).** The
`.debug_info` reader is now a proper **DIE-*tree* walk** (a depth stack tracks open
subprogram/struct/array DIEs) instead of a flat base-type pass, and recovers the full structured-type
graph: `DW_TAG_structure_type`/`union_type` + `DW_TAG_member` (name + `DW_AT_data_member_location`),
`DW_TAG_pointer_type` (pointee ref), `DW_TAG_array_type` + `DW_TAG_subrange_type` (`DW_AT_count` /
`DW_AT_upper_bound + 1`), and `DW_TAG_typedef`/`const`/`volatile` (transparent aliases). A recursive,
cycle-safe `intern_type` (reserves the `TypeId` with an `Opaque` placeholder before recursing, so a
`struct Point *` whose pointee is `struct Point` terminates) lowers each `DwarfType` into the §6
`TypeDef` graph — `Aggregate{fields}`, `Pointer{pointee}`, `Array{elem,count}`, `Base` — with names
like `struct Point *` / `int[3]`. So a wasm guest's `struct`/array/pointer locals expand in the DAP
Variables pane the same as chibicc's. Test (`debug_line.rs`, new `agg_clang.wasm` fixture): `struct
Point p` ingests as an `Aggregate{x@0,y@4,size 8}`, `int row[3]` as `Array{count 3}`, and `struct
Point *pp` as a `Pointer` whose pointee is the same aggregate — all `WindowVia` into the C frame.

**Slice 24 — LLVM `!DILocation` → the waist (a *third* producer feeds the source-line half).** The
AOT LLVM-bitcode on-ramp (`svm-llvm`) now populates the §6 neutral core's **source-line half**: each
LLVM instruction's `!DILocation` (via `llvm-ir`'s `HasDebugLoc`) is keyed onto the SVM `(func,
block, inst)` pc it lowered to, with a deduped `files` table — a `DebugAcc` threaded through
`translate_func`/`translate_block` (each defined function's final index is `base + i`, accounting
for the synthesized `_start`). A non-`-g` build carries no debug section (byte-identical to before).
So **three independent frontends** — chibicc tokens, wasm DWARF, and now LLVM DI — feed the *same*
frontend-neutral core, the decisive cross-check that the waist isn't coupled to any one frontend.
Tests (`translate.rs`): a clang `-Og -g` statement chain maps several distinct source lines, each to
an in-range IR pc, and module verification is unaffected (debug info is escape-irrelevant); a
non-`-g` build has `debug_info: None`.

**Slice 25 — LLVM *variable / type* ingest via a direct `llvm-sys` DI walk (`dbg.declare` →
`Window`).** The pinned `llvm-ir` 0.11.3 leaves the structured metadata graph unimplemented
(`Metadata::from_llvm_ref` is `unimplemented!`, `MetadataOperand` is payloadless), so a new
[`svm-llvm::di`] module reads the DI graph **directly through `llvm-sys`** (the fallback reader
`LLVM.md` §8 sanctioned), re-parsing the same `.bc` into its own context and walking it. At `-O0
-g` every C local is an `alloca` + `llvm.dbg.declare(addr, !DILocalVariable, !DIExpression)`; the
reader recovers each variable's **name + structured type** (a recursive, cycle-safe `intern_type`
over `DIBasicType`/`DICompositeType`/`DIDerivedType` → the §6 `TypeDef` graph — `Base`/`Aggregate`/
`Array`/`Pointer`, transparent typedef/const/volatile, array `count = size/elem_size`, base
`encoding` inferred from the C name since the LLVM-C API exposes no getter) and **correlates it to
the IR by alloca *ordinal*** — the Nth alloca in textual order is stable across the `llvm-sys` parse
and the translator's own walk, so it resolves to the alloca's data-stack frame slot → a
`VarLoc::Window`. So a `-O0 -g` LLVM guest's `struct`/array/pointer locals are now ingested with
full structured types, the LLVM analog of the wasm slice-23 ingest. Tests (`translate.rs`): a
`struct Point`/`int[3]`/`struct Point *` fixture ingests with the right field offsets and pointee,
verifies, and — the correlation lock — its values **read back correctly at runtime** (`p.x`, `p.y`,
`row[0]` through the interpreter's `Window` reads at a breakpoint).

**Slice 26 — LLVM `-O2`/`-Og` `llvm.dbg.value` → `SsaList` (promoted variables, the optimized
case).** At higher opt levels mem2reg/SROA promote scalars, so their debug locations are
`llvm.dbg.value(ssavalue, var, expr)` bindings rather than `dbg.declare`+alloca — the case where
**LLVM solves S2's promotion-vs-inspectability for free** (its intrinsics survive optimization). The
`di` reader now also recovers `dbg.value` bindings to a function **argument** (the stable-SSA case);
the translator emits a `VarLoc::SsaList` over the argument's live range — the argument is ValueId
`k`, threaded as a block parameter wherever it's live, so its block-local value index is simply its
position in that block's param list (one `SsaLoc` per such block, effective from block entry — no
per-pc instruction plumbing needed). So an optimized LLVM build's **function parameters** are named,
typed, and inspectable via the same location-list machinery chibicc and wasm use. The honest reality
of `-Og`/`-O2` is that most *other* locals are optimized to `poison`/constants (no recoverable
location — the reader skips those); a clang loop accumulator, e.g., is folded to closed form. Tests
(`translate.rs`): a `-Og` argument ingests as a multi-entry `SsaList` with the `int` type, verifies,
and **reads back its value at runtime** through the list at a breakpoint. *Not yet:* `dbg.value`
bindings to non-argument SSA values (instruction results / φ / loop-carried), which need the
value→ValueId ordinal correlation and the per-pc `SsaLoc.inst` position — a follow-up, of limited
yield at `-Og` since those values are often optimized away anyway.

(Everything else is built: the chibicc producer incl. optimized-build location lists; both DAP
consumers over `Window` *and* `WindowVia`; binary serialization; the wasm producer's source lines,
DWARF pass-through, named variables, and aggregate/pointer/array types; and the LLVM producer's
source lines, `-O0` `dbg.declare` variable/type ingest, and `-O2`/`-Og` `dbg.value` argument
location lists. The W4 debug-info waist is exercised end-to-end by **three** independent frontends —
all three on the source-line half, all three on the variable+type half.)

**Slice 27 — the LLVM producer driven through the *DAP server* end-to-end.** The prior LLVM slices
proved the waist at the interpreter-`Inspector` level; this one closes the loop through the **actual
W5 DAP consumer**. A real LLVM-bitcode → SVM-IR translation (with its §6 debug info) is serialized to
text (`print_module` — the debug info survives the round-trip the DAP server launches from) and
driven over the Debug Adapter Protocol: a source breakpoint **binds by line** to the recorded clang
path, the Variables pane **expands the LLVM-ingested `struct`** (a nonzero `variablesReference` →
`x`/`y` members), and **`evaluate("p.x + p.y")`** reads members over the structured type. So the LLVM
frontend's debug output is fully DAP-inspectable, the same as chibicc's and wasm's — the §6 waist is
*frontend-neutral all the way to the debugger UI*, not just at ingest. Test (`dap_over_llvm.rs`, a
new `svm-dap` dev-dep on the LLVM-only lane): a `-O0 -g` `struct Point` guest stops at its `return`
line and expands to `x=5, y=6`, with `evaluate` yielding `11`.

**Slice 28 — module-scoped globals (a schema-level waist extension across all consumers).** Source
**global** variables don't fit the function-scoped, data-SP-relative local model, so the §6 waist
gains two small, frontend-neutral primitives: a `VarLoc::Fixed { addr }` (an *absolute* window
address, not `data-SP + off`) and a `GLOBAL_SCOPE` sentinel `VarInfo::func` (`u32::MAX`, visible in
*every* frame). These thread through the whole stack — `svm-ir` (the variant + sentinel), `svm-encode`
(tag 4), `svm-text` (`debug.var global "<n>" fixed <addr>`, in both the parser **and** the
header-prescan), `svm-interp` (`read_var`/`var_addr` resolve `Fixed`; a `var_in_scope` helper ORs the
sentinel into every lookup), and `svm-dap` (globals show in each frame's Variables pane and resolve in
`evaluate`). The LLVM producer then reads each global's `!dbg` `DIGlobalVariableExpression` (via a
direct `llvm-sys` walk — `LLVMGlobalCopyAllMetadata` → `DIGlobalVariable`, op 1 = name / op 3 = type)
and correlates it by **symbol name** to `globals_layout`'s window address → a `Fixed` global with its
structured type. So a C guest's `int counter`/`struct P origin` are inspectable by name in any frame.
Because the primitives live in the neutral core, **chibicc and wasm get globals for free** the moment
they emit them. Tests: the schema round-trips a `Fixed`/`global` var through text **and** binary; an
LLVM `-O0 -g` guest ingests `counter` (a `Fixed` int reading its data-segment `7`) and `origin` (a
`Fixed` `struct P`), and over DAP a global **appears in the Variables pane** and **`evaluate("total")`
reads `42`**. (Lexical-scope narrowing — a `static`/local shadowing a same-name binding resolving to
the right one — landed later via the `scope` field; see the W5 "Built — lexical-scope resolution"
note.)

**Slice 29 — chibicc emits globals (the second producer of the slice-28 primitive).** Slice 28 put
the global primitives (`VarLoc::Fixed` + `GLOBAL_SCOPE`) in the *neutral core* precisely so every
frontend gets them; this cashes that in for **chibicc**, the project's primary C frontend. Under
`-g`, `codegen_ir.c` now emits a `debug.var global "<name>" fixed <addr>` for each source global —
a named, non-function definition (skipping `extern` decls and anonymous `.L` string literals) at the
fixed window offset `layout_globals` already assigned it, with its structured type interned
alongside the per-function locals. So a C guest's globals are inspectable by name **in every frame**,
exactly as the LLVM path's are — the *same* §6 `Fixed`/`GLOBAL_SCOPE` machinery driven by a second,
independent producer (no consumer changes: the interpreter/DAP already resolve it from slice 28).
Test (`c_frontend.rs`, the chibicc lane): an `int counter`/`struct P origin` guest emits the
`global … fixed` lines and `counter` reads its data-segment `7` at a breakpoint inside `bump`. (The
wasm DWARF producer emitting globals — `DW_TAG_variable` at module scope with a `DW_OP_addr`
location — is the natural third-producer follow-up.)

**Slice 30 — the wasm DWARF producer emits globals (the third producer, completing the trifecta).**
The `dwarf_info` reader now also recovers a **CU-level `DW_TAG_variable` located by `DW_OP_addr`** (a
fixed linear-memory address, no frame base) as a `DwarfGlobal { name, addr, type_ref }`, distinct
from the in-subprogram `DW_OP_fbreg` locals. Since the wasm→IR model maps **linear memory directly
to the window**, that `DW_OP_addr` *is* the window address, so the producer emits a `GLOBAL_SCOPE`
`VarLoc::Fixed` var (visible in every frame) with its structured type — again with **no consumer
changes** (the slice-28 interpreter/DAP path already resolves it). So all **three** frontends —
chibicc, LLVM, and the wasm DWARF on-ramp — now emit module-scoped globals through the *same* §6
`Fixed`/`GLOBAL_SCOPE` machinery, the strongest possible demonstration that the waist's global
support is genuinely frontend-neutral. Tests (`debug_line.rs`, new `global_clang.wasm` fixture): a
clang wasm guest's `int counter` / `struct Point origin` ingest as `Fixed` globals with their types,
and `counter` **reads its data-segment `7` at runtime** at a breakpoint inside `bump` (confirming the
`DW_OP_addr` → window-address mapping). The `Fixed`/`GLOBAL_SCOPE` primitive is now exercised
end-to-end by every producer and both consumers.

### Open questions (S4/S5/S2)

- *Probe dispatch in the hot loop*: monomorphized generic (`Probe` type param on `run_inner`)
  vs `&mut dyn Probe` behind the `ObsMode::Debug` gate. Lean generic so `Off`/`Memop` keep the
  current codegen; revisit if it bloats `run_inner`.
- *Checkpoint cadence for `seek`*: pure stateless re-run (cheapest, O(t) per seek) vs periodic
  window snapshots (bounds seek cost, costs memory). Start stateless; add snapshots if REPL-scale
  traces need it (interacts with §22 `JitSession`).
- *`SchedulePoint` stops*: whether the Inspector exposes scheduler-seam decisions as stoppable
  events (useful for "step the scheduler") or only op-seam events. Default op-seam; gate
  scheduler stepping behind a flag.
- *`LocList` granularity (S2)*: per-op entries (precise, larger) vs coalesced PC ranges
  (smaller; recompute the boundaries where `curval[s]` changes). Start coalesced — the frontend
  knows exactly where each slot's value changes.
- *Uninitialized / out-of-scope (S2)*: a `LocList` gap (no range covers the `IrPc`) reports
  `<not yet live>`; reuse the same honest-unavailable path as JIT `<optimized out>`.
