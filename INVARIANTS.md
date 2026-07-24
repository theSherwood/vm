# Invariants

The design rules that answer "is this change allowed?" — read this before working on
anything. Each invariant is a constraint the whole tree already obeys; a change that breaks
one is wrong until the invariant itself is deliberately renegotiated with the owner (record
the renegotiation here, dated). Keep this list short: an invariant earns its place by
rejecting real proposals, not by describing the code.

## 1. Small trustworthy core

Every line is potential TCB. Prefer the boring, obvious implementation; no abstraction,
configurability, or cleverness until something concrete demands it. When in doubt, do less.
*Violated by:* any change justified by "we might need it" rather than a failing test, a
measured regression, or a named consumer. (AGENTS.md prime directive; DESIGN.md §1.)

## 2. Confinement is the masking lowering

Memory safety for the host rests on one pass: every guest access is masked to `[0, size)` or
proven bounded. The verifier secures typing, control flow, and index ranges — **not** memory.
The target is "as secure as Wasmtime"; in-process isolation is not a Spectre boundary.
*Violated by:* any feature that adds emitted-code or window-access surface outside the
masking regime — new lowerings are suspect by default; prefer reusing an existing guarded
seam (as the JIT serve loop reused `invoke_extra`). (DESIGN.md §1a/§4; the fuzzed hinge.)

## 3. Authority moves only down the grant graph

Every capability transfer is mediated by an authority holding both ends: spawn grants and
`child_offer` re-grants. No peer discovery, no self-mint transfer channels, no registries,
no ambient names. The one sanctioned residue: a domain offering its *own* export down its
own grant graph. *Violated by:* any path where a domain reaches a capability its ancestors
never granted. (Owner decision 2026-07-23; IMPORTS.md §3.3/§3.6, PROCESS.md §4.)

## 4. Host = mechanism, guest = policy

The host's inter-domain layer is a waiter table, wake plumbing, and lifecycle cleanup —
never scheduling policy. Concretely: FIFO queues, wake-all (the host never picks a winner;
guests race through the admission lock), work-stealing, guest-stated deadlines, no
priorities, no fairness classes, no timeslicing. Scheduling *policy* lives in guest code:
guest-driven fibers (D22), parent-as-scheduler coroutines, worker-domain sharding over the
grant graph. The host holds the waiter table only because it alone can deliver lifecycle
cleanup (death-is-revocation must find parked callers). *Violated by:* any host feature
keyed on caller identity, priority, or ordering beyond FIFO — e.g. per-caller fairness
belongs in guest patterns, not the substrate. (Owner decision 2026-07-24; ISSUES.md I38/I39.)

## 5. Errors are values; traps are for forgery

Fallible operations return negative errnos, probeable on the caller's own error path.
Traps stay reserved for what can never be legitimate: forged handles (a generation never
issued), typing violations on live handles, and escape-adjacent faults. Cancellation is a
value: revocation completes calls with an errno whether the caller was parked mid-call or
calls after — a lifecycle event is never a domain-killing surprise. *Violated by:* any new
trap reachable from a benign race or another party's lifecycle action. (D42; I41.)

## 6. One world per domain

A domain's handlers, threads, and fibers share one window, one powerbox, one fuel budget.
A handler trap is terminal for the domain — never resume over half-mutated state. Safety is
serial-by-default with explicit opt-in ladders (multi-consumer serving, threading) whose
cost — the threading discipline — is the guest's stated choice. *Violated by:* partial-state
recovery, transactional handler worlds, or implicit parallelism. (IMPORTS.md §3.6; I37/I39.)

## 7. Re-execution is recovery

Parks rewind their frames, so a wake — spurious, racing, or post-thaw — simply re-executes
the parked op, which re-drains, re-parks, or re-derives its own waiter state. Calls that
cross a freeze or revocation boundary are **re-issued** (O10): at-least-once delivery,
idempotence is the personality's problem. Recovery never replays captured scheduler state.
*Violated by:* recovery designs that carry waiter/scheduler records in snapshots, or
exactly-once claims. (§3.6 rewound parks; PROCESS.md O10; DURABILITY.md §13.)

## 8. Control plane ≠ data plane

Service calls carry shell-frequency control traffic — single-slot scalar replies. Bulk data
rides `SharedRegion` rings the guests own. *Violated by:* widening the dispatch/reply ABI to
carry payloads, or any hot path routed through handlers. (F6; I39; the c_shell rings.)

## 9. The interpreter is the oracle; decline, never diverge

The tree-walker defines semantics; fast backends run only what they can run identically and
**decline the rest** (compile vetoes, routing folds — one shared predicate, one definition)
back to the oracle. Anything a backend or a step can't handle refuses probeably or falls
back — it never runs wrong and never hangs where refusal is possible. Differential tests
gate every backend feature. *Violated by:* a fast-backend feature without an oracle
counterpart, a second copy of a veto predicate, or silent divergence documented as a quirk.
(DESIGN.md §18; the serve-qualification veto.)

## 10. Identity is structural

Interface and type identity is the interned shape (D59) — never a nominal name or registry.
The one honest non-structural bit — who terminates a capability — lives in the
non-interposable attest/provenance namespace, so a parent can interpose everything but
cannot hide that it did. *Violated by:* nominal type registries, or trust decisions keyed on
names rather than provenance. (D59; IMPORTS.md §3.1.)
