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
| `cap.call` I/O record log | **Missing** | — |
| Schedule / memory-order record log (multicore replay) | **Missing** (substrate in DPOR) | — |
| Interpreter stepping / breakpoint / watchpoint / backtrace / value+window read | **Built — slices 1–2** | `svm-interp` `Inspector` (single-threaded; multithread pending) |
| Backtrace *materialization* (unwind tables → frames) | **Missing** | needs Cranelift unwind info |
| `§3a` IR debug-info side-table (source locations in IR) | **Missing** (IR has no loc fields) | `svm-ir`, frontend `codegen_ir.c` |
| DWARF emission + DAP server | **Missing** | — |
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

## 6. W4 — §3a IR debug-info side-table (source locations)

**Pillar 4, step zero.** Goal: carry source `(file, line, col)` and variable→location info from
the frontend through the IR so every later tool (interpreter source view, DWARF, DAP) has
something to symbolize against.

**Current substrate.** None — and this is the gating gap. The IR (`svm-ir`) has **no**
source-location fields, and chibicc's `codegen_ir.c` discards locations. The binary format
(§3a) is section-based with a deferred "module-level type/interface section" already noted, so a
new **strippable debug section** fits the format without touching the hot path.

**Design sketch.**
- A **side-table**, not inline ops: `DebugInfo { files: Vec<String>, locs: Map<IrPc,
  SourceLoc>, vars: Vec<VarInfo { name, ty, scope, location: VarLoc }> }`, where `VarLoc` is
  `WindowSlot(off)` | `SsaValue(id)` | `Promoted(then resolved per build mode, W6)`.
- A new **strippable binary section** + text-format syntax (1:1 text↔binary per D33); absent ⇒
  no debug info, zero cost; verifier ignores it (§2a, untrusted-for-escape).
- **Frontend**: chibicc's AST already carries `Token`/line info; thread it into `codegen_ir.c`
  emit calls. Keep the chibicc diff minimal (the project rule — only `codegen_ir.c` is ours).

**API surface.** `Module::debug_info: Option<DebugInfo>`; helpers `loc_of(ir_pc)`,
`vars_in_scope(ir_pc)`. Consumed first by W2 (interpreter source view), then W5 (DWARF).

**Dependencies.** None upstream. Blocks W5 and the source-level half of W2.

**Open questions.**
- *Granularity*: per-op locs (precise, larger) vs per-statement (smaller). Start per-statement.
- *Variable scoping* with SSA promotion (W6 tension) — `VarLoc` (designed in §13/S2) expresses
  this: `Window{off}` for memory locals, a PC-keyed `Ssa(LocList)` for promoted ones.
- *Text-format ergonomics*: keep debug info in the text IR (agent-friendly, D33) vs binary-only.
  Lean text-first for the build, per D33.

**Effort / risk.** **Moderate.** New IR section + encode/decode + verifier skip + frontend
threading + text syntax. Low *risk* (additive, strippable, no TCB), but touches several crates
(`svm-ir`, `svm-encode`, `svm-text`, `svm-verify` skip, `frontend/chibicc/codegen_ir.c`).

**Acceptance.** A C program compiled with `--emit-ir -g` round-trips source locations through
text and binary; the interpreter can print the source line for any IR-PC; stripping the section
yields a byte-identical-minus-debug module that runs identically.

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

**Dependencies.** W4 (info), W6 (promoted-local locations), W3 (frame/unwind for call-stack
requests), W8 (the `Inspector` DAP binds to).

**Open questions.** Two-engine DAP (interpreter for stepping/inspection, JIT for speed) vs
JIT-only DWARF. The interpreter-backed DAP is far cheaper and sidesteps the optimized-debug
problem — **recommend interpreter-backed DAP first**, JIT/DWARF as a later tier.

**Effort / risk.** **High.** The largest workstream; explicitly staged. Defer until W2/W4 prove
the source-level loop on the interpreter.

**Acceptance.** Set a breakpoint in VS Code on a `.c` line; it binds; hitting it shows the
source frame and inspectable locals.

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

When these are settled, fold the resolved ones into `DESIGN.md` §19 / the decision log as
`D54+` so DESIGN stays the canonical record.

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

**Not yet (next slices):** capability-using guests (grant into the Inspector's `Host`),
multithreaded debugging (the `Policy` scheduler seam, Milestone B), and the source mapping (W4).

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
