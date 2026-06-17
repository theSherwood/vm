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
| Backtrace *materialization* (unwind tables → frames) | **Missing** | needs Cranelift unwind info |
| Debug-info ABI (frontend-neutral IR waist; source locs + var locs + structured types) | **Built — neutral core + structured `TypeRef` table (text); chibicc `-g` emits `debug.var` + `debug.loc` + `debug.type`/`debug.field`** (D-DBG-7/§6; binary section pending; DAP consumer of types pending) | `svm-ir` `DebugInfo`/`TypeDef`, `svm-text`, `svm-interp`, `codegen_ir.c` |
| DAP server (interpreter-backed: source breakpoints + **conditions**, frames, locals, **source-line** stepping (in/over/out), **reverse debugging** (single + multithreaded), **multithreaded** per-thread stacks, **`evaluate`** expressions/hover) | **Built — W5 slices 1–6** | `svm-dap` (`DapServer` / `expr` / `run_stdio`) |
| DWARF emission (gdb/lldb on JIT native code) | **Missing** | needs the S6 Cranelift debug layer |
| `Inspector`/`Monitor` capability *type* | **Missing** (pattern only) | — |
| DRF-or-trap hardened race-detection tier | **Missing** (designed, §12) | — |

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

**Not yet — snapshot/checkpoint cadence (the remaining W1 performance piece).** `seek(t)`
re-executes from time 0, so it is O(t) and repeated `step_back` is O(t²). Bounding that needs
periodic state snapshots to restart from the nearest checkpoint — but the interpreter's state is not
cheaply snapshottable: `VCpu`/`Host` aren't `Clone`, and guest memory is a **shared** `Arc<Region>`
(`Mem::fork_for_thread` shares bytes, it doesn't copy them), so a checkpoint needs a deep window
copy (ideally dirty-page-tracked) plus a host-substate snapshot (output buffers, `clock_ns`,
`stdin_pos`, the cap-replay cursor). That is a dedicated effort, not a small slice; correctness
(seek already works) is unaffected — only long-run navigation cost. Also still open: RNG via a
dedicated iface (vs a host-fn), and capturing a `SchedTape`/`CapTape` from a *JIT* execution (the
interpreter is the debug engine by design, so this is lower priority).

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
second live thread mid-stop and switches back. *Not yet:* watchpoints across threads as a headline
test, and stepping that crosses a scheduler decision point (step-over-a-spawn). Those, plus W1
record/replay, are the rest of Milestone B.

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
(`debug.rs`). *Not yet (the consumer half):* wiring this into DAP — struct/array expansion in the
Variables pane (nested `variablesReference`) and `a.b` / `arr[i]` in `evaluate` (the expression
*parser* is already in place) — plus richer render names (the C tag, e.g. `"struct Point"`).

**Not yet (other W5 gaps).** Float expressions and short-circuit `&&`/`||` in `evaluate`;
conditional breakpoints are honored by forward `continue` but not yet by `reverseContinue` (it
stops at any breakpoint pc); and the JIT/DWARF tier for gdb/lldb on native code.

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
`VarLoc`. Crucially, **`-g` is `-Og`** — it disables SSA promotion (the W6 / §19 debuggable-vs-
optimized trade), so every local keeps a *stable* window data-stack slot and resolves
function-wide as `VarLoc::Window{off}`. (Promoted scalars would need S2's per-block `LocList`
since their value index varies by block/PC — deferred.) `main.c` gains a 2-line `-g` flag
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
of base scalars) and emits the table. This is the **producer + ABI** half (the §7 W5 note); the
**consumer** half — DAP Variables-pane struct/array expansion and `a.b` / `arr[i]` in `evaluate`
— is the next slice (the expression parser is already in place). The binary section is still
stripped (`svm-encode` sets `debug_info: None`), and render names are still generic (`"struct"`,
not `"struct Point"` — the C tag isn't on chibicc's `Type`).

**Not yet (next slices):** wire the structured types into DAP (Variables expansion + member/index
`evaluate`); the LLVM/wasm `debug.loc` + type ingest sides; binary serialization of the debug
section; richer type render names; and the remaining W4 refinement (per-block `LocList` so
promoted scalars are debuggable without `-Og`, per-producer rich blob). (Multithreaded debugging
and the interpreter-backed DAP server are already built — Milestone B and W5 slices 1–6.)

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
