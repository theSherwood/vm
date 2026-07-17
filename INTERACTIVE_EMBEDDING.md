# INTERACTIVE_EMBEDDING.md — the interactive-embedder surface (browser-first)

Status: **scoping doc**, written 2026-07-17. No code in this PR — this is a work-breakdown for
the capability surface below.

An *interactive embedder* is a host that drives a guest **step by step and inspects it
between steps**: debugger frontends, educational programming environments, REPLs and
playgrounds, profiling/visualization tools. Natively, SVM already serves them — the
tree-walker's `Inspector` (`svm-interp`) has stepping, breakpoints, watchpoints, time-travel,
and a DAP server (`svm-dap`) on top (`DEBUGGING.md`). **In the browser it does not**: the
browser build (`browser/`, `BROWSER.md`) runs the bytecode engine through run-to-completion
entries (`svm_run*`), with no interactive surface at all.

This doc scopes the workstreams that close that gap, plus a few adjacent host/tooling
capabilities interactive embedders keep needing. Requirements are stated embedder-neutrally;
several prospective consumers (e.g. educational debugger frontends) want this surface, and
nothing here couples SVM to any one of them. Acceptance is against SVM's own oracles — the
native `Inspector` and the differential house style — not any consumer's test suite.

Design invariants inherited from `DEBUGGING.md` §0 (do not relitigate): the debugger is a
host-side observer that never widens guest authority; debug info is tooling, untrusted for
escape; the interpreter tier is the debug engine.

---

## Current substrate (what this builds on, not rebuilds)

| Piece | State | Where |
|---|---|---|
| Full interactive debug surface, native | Built | `svm-interp` `Inspector`, `svm-dap` (`DEBUGGING.md`) |
| Bytecode-engine debug seam: op-for-op stepping-location + per-step window/SSA-value traces (single-vCPU, seam-free) | Built, **batch-shaped** | `bytecode.rs` `ir_trace`/`ir_window_trace`/`ir_value_trace` (`crates/svm-interp/src/bytecode.rs:3003/:3045/:3101`) |
| Bytecode values inspectable: stable, unique slot per SSA value (`regs[base + i]`, typed by `func_value_types`; no reuse/coalescing), parity-proven vs the tree-walker | Built | `DEBUGGING.md` §1b G2, `crates/svm/tests/debug_parity.rs` |
| Single-op stepping bit-identical to run-to-completion (`budget = 1`) | Built | `bytecode.rs:1391/:2997` |
| Deterministic, self-contained browser `Host` (streams accumulate, stdin is a buffer, `Clock` is a counter) | Built | `BROWSER.md` § Decisions |
| Host-serviced vCPU events (spill frame → host services → `deliver_*` resumes) | Built (pattern) | `bytecode.rs:1842ff` (`VcpuEvent`, tier-up path) |
| Cooperative multi-vCPU `drive` + deterministic timeout selection | Built | `bytecode.rs:4623` |
| Memory-access instrumentation pass (observe/veto every guest memory op, zero-cost-when-off, all backends) | Built natively | `HOOKS.md`, `Instance::with_mem_hooks` (`crates/svm-run/src/lib.rs:3615`) |
| Source-level debug info waist (`debug.loc`/`debug.var`/types), chibicc `-g` | Built | `svm-ir` `DebugInfo`, W4 in `DEBUGGING.md` |
| `display` / `keyboard` / `fs` browser capabilities | Built | `browser/src/lib.rs` (~:1831), `demos/doom/` |

The key prior finding (`DEBUGGING.md` §1b): *the bytecode tier is fully inspectable, not
precluded — it is unbuilt as a DAP backend, not blocked.* Everything below is wiring, not
research.

---

## W1 — Interactive debug sessions on the bytecode engine (the critical path)

**Need.** An embedder must be able to: step one op / one source line (into/over/out), run
until a breakpoint/watchpoint/fuel bound, pause, read the PC and source location, read frames
+ locals + arbitrary window bytes (and write bytes), step **backward**, and `seek` to an
arbitrary step index — synchronously, from JS, against the browser cdylib.

**Today.** The `ir_trace` family is trace-after-the-fact (run fully, return the sequence), not
interactive; the cdylib exports are run-to-completion. The full `Inspector` lives on the
tree-walker, which is excluded from the wasm build (fail-closed — its `Scheduler` uses OS
threads/`Instant`; `BROWSER.md` § Decisions).

**Direction.**
1. A **resumable debug-session object** over the bytecode engine: own the `Vcpu` + `Mem`,
   execute with `budget = 1` per call (already bit-identical to run-to-completion), expose
   `IrPc`, slots, and the window. Single-vCPU, seam-free scope first — exactly `ir_trace`'s
   scope.
2. **Time-travel v1 by replay**: the browser `Host` is deterministic and self-contained, so
   `seek(t)` = re-run from 0 with the same inputs; cache periodic window+slots snapshots so a
   seek costs O(snapshot interval). `step_back` = `seek(t−1)`. An undo-log can come later if
   replay-cost ever matters; it changes nothing observable.
3. **Breakpoints/watchpoints** as step-loop checks: source breakpoints via `debug.loc`;
   watchpoints via the W3 hook pass or a per-step window diff — whichever is simplest that
   meets acceptance.
4. **cdylib ABI** (same `svm_alloc` conventions as existing entries):
   `svm_dbg_new(module, stdin, caps) → session`, `svm_dbg_step / step_back / run_until`,
   `svm_dbg_pc / source_loc / step_count / seek`, `svm_dbg_read_reg / read_var / read_window
   / write_window`, `svm_dbg_frames_json`, breakpoint/watchpoint set/clear/list. Fuel bounds
   every `run_until`.
5. **Threads follow-on**: multi-vCPU debug rides the cooperative `drive` with a deterministic
   scheduler and a global turn counter (the `Inspector::turn` shape). Not in the v1 slice.

**Acceptance.** A Node/Chromium test compiles a `-g` C program and drives: step to a source
line → hit a breakpoint → read a local → `seek` back 10 steps → re-read (value differs) →
step forward to reconvergence — with every stepped location and read value matching the
native tree-walker `Inspector` on the same program (extend the `debug_parity.rs` pattern
through the wasm ABI). A watchpoint fires at the same step index as the native `Inspector`.
Fuel stops a runaway `run_until`.

## W2 — Machine-state view (rides on W1)

**Need.** Debugger UIs want a "machine panel": a register file, a program counter, a stack
pointer, and SIMD lanes — real machine state, not a display fiction.

**Today.** The bytecode engine *is* a register machine with stable typed slots (§1b G2). The
chibicc frontend threads a data-stack pointer through calls; frames with
spilled/address-taken locals live at real window addresses. `v128` is a first-class value
type.

**Direction.**
- **v1 (with W1):** expose the current frame's slot file (filtered: `debug.var`-named values
  + recently-written), `IrPc` as the PC, the data-stack pointer as SP (frame base as FP), and
  lane-rendered `v128` slots. Pure ABI + view work; the state already exists.
- **v2 (optional follow-on):** an opt-in **finite register file** compile mode in
  `compile_func`: cap slots at a small named set (e.g. 16), spill excess to the data stack
  (visible in the window), pass leading args in designated registers. Naming should be
  RISC-flavored (`a0–a7`/`ra`/`sp`/`t*`): SVM IR is a load/store machine whose compares
  produce values — there are no flags, so borrowing a flags-ISA's names would misdescribe the
  machine. Differentially tested against the unconstrained mode (house style). This makes
  register scarcity, spilling, and calling conventions *observable* — useful to any embedder
  that teaches or visualizes them.

**Acceptance (v1).** For a program with named locals: at every step, exposed slot values
equal the tree-walker's `read_var` (the `ssa_var_value_parity_per_step` pattern, driven
through the wasm ABI). SP visibly moves across call/return; a `v128` local renders its lanes.

## W3 — Memory-access scoring in the browser

**Need.** Profiling/visualization embedders want the guest's memory-access stream: cache and
locality models, heat maps, access ordering — without touching the engine.

**Today.** The `HOOKS.md` pass fires an embedder hook around every guest memory op, identical
across backends, zero-cost when off — with cache/page-fault scoring as a named use case. It
is wired natively (`Instance::with_mem_hooks`); whether it is reachable from the browser
cdylib is unverified.

**Direction.** (1) Confirm the hook pass runs on the bytecode engine under wasm; add a
hook-install flag to the W1 session. (2) Ship access-stream consumers (e.g. a small L1/L2
cache model with per-run counters and a line-state dump) **host-side in the cdylib** as
tooling — models stay out of the engine and out of the TCB.

**Acceptance.** A strided-vs-sequential access pair of guests shows the expected miss-count
ordering, and browser counters match the native run of the same hook stream.

## W4 — Blocking-input suspend/resume

**Need.** Interactive guests read input that does not exist yet (a REPL prompt, a stdin-driven
program). The embedder needs the run to **suspend** when input is exhausted, surface that to
JS, and **resume** when it supplies bytes — instead of EOF-and-done.

**Today.** The browser `Host`'s stdin is a pre-supplied buffer; a read past the end is EOF.
The engine already has the right seam: `VcpuEvent` spills the frame for host-serviced events
and resumes via a `deliver_*` call (the tier-up path).

**Direction.** A `WaitingForInput`-style outcome on the W1 session (and optionally the plain
run entries): when the stdin capability's `read` finds no bytes, suspend the vCPU via the
`VcpuEvent` pattern and return a distinct status; `svm_dbg_provide_stdin(ptr, len)` appends
and resumes. Provided bytes join the run's deterministic input record (the `CapTape` idea from
`DEBUGGING.md` W1), so a later `seek` replays them faithfully without re-suspending.

**Acceptance.** A prompt-loop C guest round-trips two provided inputs from a test page;
`seek(0)` + re-run replays both, byte-identically, with no new suspensions.

## W5 — In-browser frontend (C source → module, client-side)

**Need.** Interactive embedders want the full edit-compile-run loop client-side: source text
in, verified module out, no server round-trip, sub-second warm compiles.

**Today.** `frontend/chibicc` runs natively only. This is already tracked as `BROWSER.md`'s
"real-language playground tab" open item ("pre-compiled modules first, in-wasm compilation
later"); the playground's `svm_parse` (text IR → verify → encode inside the cdylib) shows the
in-wasm pattern.

**Direction.** chibicc is plain C99 with modest libc needs; compile it to wasm as a
**separate** module the embedder's worker calls (`--emit-ir` + the encoder: C source in,
`.svmb` out), keeping the Rust cdylib untouched. Always emit `-g` — the W1 surface depends on
debug info. (Running chibicc as an SVM guest over `fs` is a nice later dogfood, not the first
slice.) Details belong to the `BROWSER.md` item; this doc adds the requirement that the
compile path emits debug info and the W6 compile metrics.

**Acceptance.** In Chromium: source → verified module → runs on a W1 session, no server;
warm compile of a few-hundred-line program well under a second.

## W6 — Small host/tooling items

- **Compile metrics from the frontend.** Emit per-file node/size counts alongside
  `--emit-ir` output (a walk at emit time). Embedders use these for complexity budgets and
  UI display; SVM cost: a small report, no new machinery.
- **Deterministic-scheduler seed exposure.** The cooperative scheduler's seed should be
  get/settable through the browser ABI so embedders can reproduce and vary interleavings
  (pairs with the W1 threads follow-on; the native `attach_scheduled_seeded` already exists).
- **`display` frame-query op.** A capability op that answers simple predicates over the last
  presented frame (e.g. count of pixels matching an RGBA value) so embedders can assert on
  visual output without reading the whole frame back per query. A few lines in the cdylib
  host next to `present`.
- **Window memory-map introspection.** A JSON description of the window layout — data-segment
  placements, guest heap extent, data-stack region, capability-mapped regions — derived from
  module + Memory-capability state. Read-only tooling over existing state.
- **Design note — time-travel is global-turn.** Multithreaded `seek`/`step_back` targets a
  global turn counter (the `Inspector::turn` model). Rolling back one thread independently
  while others stand still is not meaningful under shared memory and is a **non-goal**.

## Non-goals

- Consumer-side integration (any embedder's UI, worker glue, content, or test suites).
- DAP-over-the-browser-build (the native `svm-dap` stays the DAP story; the W1 ABI is
  lower-level and JS-shaped).
- Porting the tree-walker (and its OS-thread `Scheduler`) to wasm — the bytecode engine is
  the browser debug tier, per the fail-closed decision in `BROWSER.md`.
- Matching any particular consumer's legacy machine model (register names, flags registers,
  fixed address layouts). W2 exposes SVM's real machine state; a finite-register *mode* is
  the one concession, and it is SVM-shaped.

## Suggested slice order

1. **W1 spike** — single-vCPU interactive step + source breakpoint exported from the cdylib,
   driven by a throwaway page, parity-checked against the native `Inspector`. De-risks
   everything; all else stacks on it.
2. **W1 time-travel + watchpoints**, then **W2 v1** (same ABI).
3. **W4** (small, high leverage for interactive embedders).
4. **W5** + the **W6 compile metrics** (closes the client-side edit-compile-run loop).
5. **W3**, remaining **W6**, **W1 threads** (+ seed exposure).
6. **W2 v2** (finite register file) — optional, demand-driven.

Each slice lands with tests gating CI, differential against the tree-walker wherever both
observe the same program, per `AGENTS.md`.
