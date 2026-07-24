# Known Issues & Robustness Gaps

A registry of **known bugs, robustness gaps, and latent hazards** that are understood but not yet
fixed — distinct from the forward-looking design/status docs (`DESIGN.md`, `DURABILITY.md`).
An entry here is a deliberately-deferred problem with a recorded root cause and a fix
sketch, so it isn't rediscovered from scratch. When an issue is fixed, move it to the bottom
("Resolved") with the commit/PR, or delete it and note the fix in the relevant design doc.

Severity: **S1** corruption/escape · **S2** guest-triggerable host crash or wrong result · **S3**
robustness/quality · **S4** cosmetic/flake.

---

## Open

> **I36–I40 (2026-07-23):** the §3.6 serving-substrate review, recorded at the owner's request
> after a design walkthrough. Two further items from the same review were **already tracked** and
> are not duplicated here: fiber-level `svc.wait`/`Join` parks (TODO.md §3.6 residue,
> "Join-in-fiber parks") and durability × serving (TODO.md "Durable event-parks" + PROCESS.md O10).
> Verdict from the review: none of these needs a different design — the model is the actor model
> (domain = actor, svc queue = mailbox, one world = actor state) — but I36 is a promoted work item
> and I37/I38 need their idioms documented so they're chosen, not stumbled into.

### I36 — a serving module runs its ENTIRE program on the tree-walk oracle: `module_serves` folds the fast backends away (S3, **promoted 2026-07-23 — owner: the cliff is not acceptable**)

**Where:** the §3.6 parity decision (IMPORTS.md): the serve loop (`svc.wait`/`svc.poll` + handler
admission) exists only in the tree-walk eval loop; the bytecode and JIT entries detect a serving
module (`module_serves`) and fall back to the oracle **for the whole module** — compute included.
One impl-export handler costs a personality its fast backend everywhere.

**Why it's a gap, not a design flaw:** nothing in the model precludes native serve loops — the
JIT already has the pieces (fiber runtime, futex thunks, call trampolines, host-side queue). The
fold was the correct differential-first baseline; it was parked "awaiting benchmark evidence,"
and the owner's verdict supersedes that: a parent-as-kernel personality (jacl) is exactly a
serving domain that needs its compute fast.

**Fix sketch (staged):** (1) bytecode serve loop — the same rewind-driven state machine
(`serve_run`/`handler_parks`/`serve_count`) in the bytecode dispatch loop, sharing the Host queue
and Sched wake paths; differential vs the tree-walk. (2) JIT serve loop — `svc.wait` as a thunk
parking on the domain queue (condvar keyed like the futex table), handlers launched as fibers via
the existing call trampoline, handler parks riding the S1c shared-futex machinery. The oracle
fold stays as the differential baseline, not the shipped path.

**Bytecode map + slice-1 design (2026-07-24).** The gap is wider than the fold comment reads:
`compile_inst` declines the *whole* §3.6 surface — svc ops 9/10 (`bytecode.rs:1230`), every
`Instantiator` op past 7 via the catch-all (`:1223` — so granted spawns 8/11/13 and `child_offer`
14 also fall back), and a caller's `cap.call` on a `LiveImpl` handle reaches generic dispatch and
refuses. What the engine already has is the right substrate, in cooperative form: `drive` (a
deterministic cooperative scheduler over `TaskSlot { vt, threads, env, state }` with
`TaskState::{Runnable, BlockedJoin, BlockedWait, Done}` and a logical clock), a **run-shared
fiber registry** (`FiberState`, D57 migration), confined `ChildEnv`s for §14 children, and —
crucially — all §3.6 *state* already lives on the shared `Host` (`svc_queue`, `svc_results`,
tickets, `svc_handler_func`), so enqueue/settle/reply plumbing is reused verbatim; only the
scheduling is engine-local. The staged plan: (a) serve-loop core, (b) caller-side
`TaskState::BlockedTicket` parking, (c) the wake paths in `drive`, (d) granted spawns (8/11/13)
so a serving *child* spawned from bytecode runs native too.

**Slice 1 BUILT (2026-07-24) — the `svc.poll` serve-loop core, native on bytecode.** A serving
module now compiles when a **qualification veto** admits it: any park-capable seam anywhere in
the module (futex waits/threads, fibers, coroutines, nested instantiate, setjmp/longjmp — a
`longjmp` out of a handler would unwind past the serve linkage — blocking stream reads,
spawn-bound imports, gc.roots) still declines to the tree-walk oracle, whose serve arm has the
fiber-park machinery, so a native handler always runs to completion or traps. `Op::SvcPoll` is
the tree-walk serve arm's rewind state machine in register-window form: an admitted handler
activation's return linkage re-enters the op (pc un-advanced) with its result in `dst`, the
re-execution settles it into the ticket's completion cell (no cross-domain caller can be
ticket-parked in this engine yet, so the reply always rides the cell — the tree-walker's
unclaimed-result path), and the drained-queue execution delivers the served count. Arity
mismatches errno inline and serving continues; a handler trap is terminal (one world), matching
the oracle. Pinned by `svm-interp/tests/bytecode_svc.rs`: cross-entry equality on the slice-2
corpus scenarios, a full-queue (64-dispatch) native drain, and `compile_module` is-Some/is-None
pins so the differential can never silently degrade into re-testing the fallback. Remaining, in
order: **slice 2** — `svc.wait` + its waker topology in `drive` (needs an enqueuer: caller-side
ticket parking and/or the granted spawns, which stay declined); **slice 3** — the JIT serve
loop.

**Slice 2 BUILT (2026-07-24) — `svc.wait`, caller-side parking, and `child_offer`: the whole
caller ↔ servicer round-trip native on bytecode.** `Op::SvcPoll` grew the wait form (CAP_SELF
op 10): an empty queue with no progress persists the cursor AT the op (a wake re-executes the
whole drain — the tree-walker's rewound park) and surfaces `Outcome::SvcWait`, which `drive`
parks as `TaskState::BlockedSvc`. The caller side rides three new pieces: (1) `cap.call` on a
handle probes `live_impl_of` first — a live-callee hit enqueues on the callee's `Host` (its
lock only) and surfaces `Outcome::LiveCall { ticket, callee, dst }`; `drive` wakes any
`BlockedSvc` task of the callee's domain (the tree-walker's `svc_wake`) and parks the caller as
`TaskState::BlockedTicket`; a full callee queue is the probeable `-EAGAIN`. (2) A settle-wake
scan at the top of the pick loop claims settled completion cells (`svc_results.remove`) into
parked callers' `dst` — the tree-walker's cap_reply preference in cooperative form. (3)
`Instantiator` op 14 (`child_offer`) mints a live offer over a spawned child's export:
`offer_shape` from the callee's module (its lock, fetched before the wirer wires — the
tree-walker's lock order), `wire_live_impl` into the parent's table, bad handle/export a
probeable `-EINVAL`. To make the callee reachable, `ChildEnv.host` became `Arc<Mutex<Host>>`
(the same shape the tree-walker's live bindings hold, so `wire_live_impl`/`live_impl_of` are
reused verbatim) and both spawn arms set the child's `self_module` (op-0 same-module children
clone the parent's; op-5 grants carry their own). The serve loop's home-module guard
generalized from `module != 0` to `module != self.home` (a `Vm` field: 0 for the primary, the
pushed unit index for a separate-module child) — the slice-1 pin was the *reason* for the
guard (handlers resolve against the domain's `self_module`, so serving from any other unit
would index the wrong program table), and a spawned serving child IS its own home. The
non-scheduler drivers (single-vCPU `Vcpu::run`, `run_vcpu_parallel`) fail closed
(`ThreadFault`) on the new stops, and an unwakeable park is the scheduler's existing deadlock
`ThreadFault` — fail-closed where the tree-walker's richer waker set (timers, cross-process)
would hang differently; the differential never runs hang cases. Pinned in `bytecode_svc.rs`:
the separate-module corpus round-trip (op-5 spawn → op-14 mint → live call parks → `svc.wait`
serves → settle-wake → join = 142) with is-Some compile pins on BOTH modules, and
`svc.wait`-with-queued-work ≡ `svc.poll` progress semantics. Remaining: **slice 3** — the JIT
serve loop; then the granted spawns 8/11/13 (still declined → tree-walk).

**Slice 3 BUILT (2026-07-24) — the JIT serve-loop core: `svc.poll`/`svc.wait` native, the fold
narrowed to what still needs the oracle.** The shape is embedder-side, not new lowering: the ops
already reach svm-run's `cap_thunk` through the generic `cap.call` path, so the thunk grew a
`serve_native` arm (CAP_SELF 9/10 intercept, like the iface-11 `Jit` intercept beside it) that
pops the Host's `svc_queue` and invokes each handler's compiled code **over the live window**
via the pre-existing `invoke_extra` re-entry seam — the same mid-`cap.call` guest-invocation
machinery the guest-driven `Jit` capability uses, nested detect-and-kill included. svm-jit's
only contribution: `CompiledModule::compile` now emits a **buffer-ABI trampoline per
impl-export handler** (the same `build_trampoline` the entry gets — any arity, no per-signature
ABI) exposed as `handler_tramp(fidx)`, and the module pointer is registered on the Host around
each run (`set_serve_native_ctx` — a root slot, since a serving module need not hold a `Jit`
grant). Semantics mirror the oracle and the bytecode `Op::SvcPoll` exactly: arity-mismatch →
inline `-EINVAL` settle, serving continues; handler trap → the run's trap cell, terminal
(one world); drained queue → the served count; `svc.wait` with no progress → fail-closed
`ThreadFault` (no enqueuer can exist mid-run while the op-14 fold stands — the bytecode drive's
deterministic-deadlock answer). Replies always ride the completion cells (no ticket-parked
caller exists on this backend yet). **Routing** (`svm-run`): the `module_serves` fold narrowed
to `module_serves && !serve_qualifies` — `serve_qualifies` is the bytecode compile veto's
svc-qualification predicate, extracted (`scan_seams`) and exported from
`svm_interp::bytecode`, so both fast backends admit exactly the same serving modules (one
definition, no drift). Still folding: op-14 offer mints (caller-side wiring, the next JIT
slice), park-capable serving modules, and the concurrent path (`cap_thunk_locked` answers svc
ops `-EINVAL` — a serve-qualified module has no thread ops, so it never routes there; the
guard exists so a stray dispatch can't self-deadlock under the lock). Pinned by
`svm/tests/jit_svc.rs`: tree-walk ↔ JIT differential on the slice-1 corpus scenarios
(results, completion cells, drain-once, **byte-identical final memory** — the escape-oracle),
2042/[7,12] headline pins, the `svc.wait` fail-closed pin, and `serve_qualifies` is-true/
is-false routing pins; `svc_parity.rs` (the op-14 program) stays green on the fold. Remaining:
**JIT caller side** — op-14 `child_offer` + live-call enqueue/park over persistent child Hosts
(the nursery currently frees a granted child's Host when it returns; a serving child must
outlive its spawn behind an `Arc<Mutex<Host>>`-equivalent), then the bytecode granted spawns
8/11/13.

### I37 — a handler trap kills the whole serving domain: total blast radius per bad request (S3)

**Where:** §3.6 handlers run over the domain's **one world** (same window/powerbox/fuel), so a
trap in any handler is terminal for the domain and every in-flight dispatch — any client that
finds a crashing input in any handler takes the service down for everyone. Death-is-revocation
keeps the failure *clean* (parked callers wake with a probeable errno; nothing hangs), but the
blast radius is the domain.

**Why "continue after trap" is not the fix:** the world may be half-mutated at the trap point;
resuming the serve loop over corrupted state would be unsound. Trap-is-terminal is forced by
one-world semantics, which is also what makes handlers race-free without locks.

**Fix (idioms, not substrate):** the actor-model answers, both already expressible —
(1) **supervision**: the parent `join`/`poll`s its serving child and respawns it on death (all
primitives exist; a documented pattern, optionally a personality-level respawn helper);
(2) **isolation granularity is domain granularity**: put risky handlers in worker child domains
the server spawns (pay-for-what-you-isolate). Action: document both as THE pattern in
IMPORTS.md/PROCESS.md so personalities choose their blast radius deliberately.

**Supervision mechanics correction (2026-07-24):** `join` of a trapped child **re-raises the
child's trap in the joiner** (interp `Pending::Join` → `out.result?`; JIT `join` →
`*trap_out = trap`) — so a naive supervisor that joins a crashed worker dies with it. The
supervision idiom is **`poll` → status `2` (trapped, non-propagating) → `detach` + respawn**;
`join` only after `poll` reports a clean return. Any supervision pattern doc must lead with this.

**Escalation options if the domain-fatal default proves too sharp** (recorded for the future,
none built): (a) **poison-drain** — on a handler trap, errno the trapped dispatch's caller *and*
every queued/parked dispatch, refuse new work, exit cleanly: converts a blast into an orderly
shutdown with no execution over torn state (errno plumbing is host-side); cheapest real
softening. (b) **opt-in resilient mode** — the trap kills only the handler fiber, its caller
gets an errno, the loop continues: VM-sound (confinement holds regardless — torn state is a
guest-consistency risk the domain explicitly accepts, with crash-only handler discipline);
interp-side cheap (drop the fiber's frames), JIT-side sensitive (the detect-and-kill guard must
unwind to the serve frame instead of the domain — trap-shim/guard machinery, escape-TCB-adjacent)
plus a leaked-resource sweep for the dead handler's tickets (cf. I40). (c) durability (see the
TODO.md durable-serving row): thaw-from-snapshot turns domain death into state rollback — the
complementary answer rather than a trap-scoping one.

### I38 — the servicer cannot shed or shape load: no per-client fairness, no admission control beyond one global quota (S3)

**Where:** the svc queue is one bounded FIFO per domain; the only backpressure is queue-full and
the fiber quota at admission (`EAGAIN`). A single chatty client with a live offer can keep the
queue full and starve every sibling into `EAGAIN`; the servicer cannot cancel a stuck parked
handler, deadline a dispatch, or distinguish callers. Caller-side timeouts (racing fibers +
revocation-unparks, O1) protect *callers* — nothing protects the *servicer* beyond provider-pays
fuel caps.

**Boundary:** mid-flight handler cancellation is unsound for the same one-world reason as I37 —
load control must live at **admission**, where nothing has mutated yet.

**Fix sketch:** per-caller (or per-offer) bounded sub-queues with round-robin admission — the
enqueue path already knows the caller's identity (ticket/domain); plus an optional timed
`svc.wait` so an idle-but-scheduled servicer can run its own housekeeping. Parked-handler
discipline stays guest-side (handlers use timed waits). Small, additive, no model change.

### I39 — handler execution is serialized: one domain's dispatches never use more than one core (S3, latent hazard — a constraint to keep documented, not a bug)

**Where:** concurrency in the serve loop comes only from handler *parks*; a CPU-bound handler
blocks every other dispatch until it finishes or parks. This is the flip side of the race-freedom
guarantee (one world, no locks) and matches F6's scoping of guest-served calls to
**shell-frequency control traffic**. The hazard is latent: someone routes a hot path or a data
plane through handlers and discovers the ceiling in production.

**Fix (pattern, not substrate):** shard state across worker domains (the parent introduces
clients to N workers — the grant graph is the load balancer), and keep bulk data on the
`SharedRegion` ring plane, never in handler args. Action: state the ceiling and both patterns
explicitly next to F6 so the constraint is designed around, not tripped over.

**Resolution path (owner-agreed 2026-07-24) — serial by default, an opt-in ladder up:** the
serialization is a *serve-loop* property, not a one-world property (the substrate already has
real parallelism over one window via `thread.spawn` + atomics/futexes). The ladder:
(1) *available today* — handler-internal parallelism: a handler `thread.spawn`s workers and
rendezvouses on a futex (`atomic.wait` fiber-parks correctly in handlers, slice 5b), so the loop
keeps serving while the handler's compute uses other cores; the Join-in-fiber residue is the
rough edge to smooth. (2) *available today* — shard across worker domains when state partitions.
(3) *substrate extension, sequenced AFTER I36* — **multi-consumer `svc.wait`**: N spawned server
vCPUs each park on the domain queue (`svc_waiters` becomes multi-waiter per key; queue pops are
already host-locked; per-vCPU serve state needs no sharing; near-free on the native JIT loop).
The cost is semantic and must be pinned in the differential before the JIT loop exists: handlers
in a multi-server domain are threaded code (atomics/locks discipline — the same opt-in contract
as `thread.spawn` generally, per D22), and the woken-before-admissions/completion ordering
guarantees become per-worker. A domain that spawns one server keeps today's lock-free semantics
untouched. Transactional per-dispatch worlds were considered and rejected (fights flat memory +
the JIT's raw stores; guest-visible aborts).

### I41 — revocation is observably inconsistent: a *parked* call through a revoked handle completes with an errno, a *fresh* call traps the domain (S3) — found 2026-07-24 answering "can a trap be triggered by a simple revocation?"

**Where:** yes, it can — and it's the most likely non-bug trap in a long-running server. D37
makes a revoked handle indistinguishable from a forged one (the slot's generation bumps; "any
later `cap.call` on it traps", `Host::close`), so a server whose grantor revokes *anything* it
holds dies on its next use of that handle. But §3.6 slice 1 (revocation-unparks) already broke
the revoked≡forged equivalence for the *parked* case: a fiber parked in a call through the
revoked handle wakes with a **negative errno** — the in-code comment says it outright:
"the call completes with the negative errno … no trap, no kill; **cancellation is a value**"
(`Pending::CapResult`). So the same lifecycle event is a value if you were mid-call and a
domain-killing trap if you call a moment later. There is no guest-side defense: reflection
can't check-then-use atomically (TOCTOU).

**Fix sketch — graceful revocation (tombstones):** distinguish *revoked-once-valid* from
*never-existed* in the holder's table (a tombstone binding, or a generation→revoked side map):
use of a tombstoned handle returns a probeable `-EREVOKED`-style errno (consistent with the
unpark path — cancellation is a value); a forged handle (dead generation, no tombstone) still
traps. Costs to weigh deliberately: tombstone storage until a slot-reuse policy exists, and the
D37 anti-probing property — which revocation-unparks has already half-surrendered, so the
tombstone *completes* an inconsistency rather than creating one. This pairs with I37: it removes
the dominant benign trigger before any trap-scoping mechanism is considered.

**BUILT (2026-07-24).** Better than the sketch: **no tombstone storage at all** — a slot's
generation advances only at (re)grant (`try_grant`), so every generation `1..=current` was once
a live handle, and a dead-but-issued generation IS the tombstone (`Host::handle_revoked`; once
the full-width counter wraps past the handle's generation bits, every masked generation has
genuinely been issued, so the check degrades exactly as `resolve`'s own masked ABA acceptance
does). A `cap.call` through such a handle completes with **`CAP_REVOKED` (`-EBADF`)** — the
*same* errno the slice-1 revocation-unpark delivers, so cancellation is a value whether the
caller was parked mid-call or calls a moment later. Still traps: a forged generation (never
issued — D37's real target) and a **wrong-type use of a live handle** (`handle_revoked` is
false for live handles, so typing discipline is untouched). One seam covers all three backends
(the single `resolve` site at the top of `cap_dispatch_slots_inner`), plus the D45 `Clock.now`
fast path (`fast_clock_now` answers the identical errno, so the JIT's fast-cap route can't
diverge). Pinned by `svm/tests/revocation_errno.rs`: revoked → `-9009` on tree-walk/bytecode/
JIT (the JIT case exercising the fast path), forged → `CapFault` on all three, live-wrong-type
→ still `CapFault`.

### I40 — an unclaimed svc reply outlives a dead caller: `svc_results` entries are never garbage-collected (S4)

**Where:** a completed dispatch whose caller didn't (or can't) claim the reply parks the value in
`Host::svc_results` keyed by ticket. If the caller died between enqueue and claim, nothing sweeps
the entry — a long-lived serving domain accumulates orphaned tickets. Bounded by call volume, not
by live state.

**Fix sketch:** sweep a caller's outstanding tickets on its death/revocation (the
death-is-revocation path already visits the waiter structures), or bound the map with an LRU/TTL.
Small; suitable as a rider on any §3.6 residue slice.

### I35 — chibicc miscompile (unreduced): an indexed array store through a post-incremented counter inside a capability-enumeration loop read back zeros (S3) — seen 2026-07-23, building the c_shell `__stage` ring runner

**Where:** guest C compiled by the chibicc frontend (`--child-entry`). The `__stage` filter
runner's grant-discovery loop originally read

```c
int regs[2]; int nregs = 0;
int n = __vm_cap_count();
for (int i = 0; i < n; i++) {
  int t = 0;
  int h = __vm_cap_at(i, &t);
  if (t == 4 && nregs < 2) regs[nregs++] = h;
}
```

and `regs[0]`/`regs[1]` later read back **0** (both) even though `nregs` correctly reached 2 and
an *inline* re-enumeration in the same function saw the right handles/types — so the powerbox and
`cap.self.get` are fine; the `regs[nregs++] = h` stores are what went missing. A **minimal**
probe (straight-line `a[n++] = 7; a[n++] = 9;` in a `--child-entry` `main`) compiles *correctly*
(the emitted IR increments and indexes right), so the bug needs more of the surrounding shape —
suspects: the loop back-edge interaction with the promote-scalars pass on `nregs`, the
address-taken `&t` neighbor, or the local (frame-relocated) array in a `main(argc, argv)` child.
Not reduced further.

**Workaround (in-tree):** the runner uses explicit slot picks (`if (nregs == 0) regs[0] = h; else
if (nregs == 1) regs[1] = h; nregs = nregs + 1;`) on `static` storage — see
`crates/svm/tests/c_shell.rs` (`STAGE_RUNNER_MAIN`), which carries a pointer to this entry.

**Fix sketch:** reduce by re-adding the original shape piecewise (loop, `&t`, local vs static
array, `--child-entry` argv frame) against the emitted IR diff; the defect is frontend-only
(codegen_ir.c), no VM/TCB involvement.

### I34 — CI flake: `apt-get install gcc-mingw-w64-x86-64` stalled ~29 min on the `fiber-scaling (stack-check + arena-stacks)` job until the run was cancelled (S4) — seen 2026-07-23, PR #422 run 30027500683

**Where:** the ubuntu-latest job's mingw cross-toolchain install step (for the
`x86_64-pc-windows-gnu` cross-clippy). The sibling `build · test · fmt · clippy` job ran the
**same step in the same run** in ~12.5 min (also slow, but completing) — so this is an apt
mirror/runner stall, not a tree change (the job's compile+test steps had all passed).

**Also observed on the same PR (separate root cause, fixed in-tree):** the windows-latest
`cargo test --workspace` hung >30 min because the new `concurrent_stages.rs` fixtures gave
children 32 KiB windows while the Windows §13 map granule is the 64 KiB allocation
granularity — the region map refused probeably, the ring landed in each child's private
anonymous pages, and the consumer's futex loop polled forever (no iteration cap). Fixed by
sizing child windows to 128 KiB (map `len = granule` queried at run time, portable across
4 K/16 K/64 K granule platforms) and adding a timeout-count **bail** to every wait loop so
any future rendezvous regression fails loudly in seconds instead of hanging a runner.

**Action if the apt stall recurs:** cache the mingw toolchain (Swatinem-style or a
pre-built container) or add a step-level `timeout-minutes` so the job fails fast and
re-runs instead of burning the runner budget.

**Same-day sibling (2026-07-23, run 30032025837):** the `real-browser` job's "Install
Playwright + Chromium" step stalled >30 min (24 s – 3 min on every prior run) — an
npm/CDN download hang, before any tree code runs. Third distinct infra fetch-stall of
the day (apt mingw, runner-loss mid-link, npm). The pattern generalizes the mitigation:
**every network-fetch step in CI should carry a `timeout-minutes`** so a wedged mirror
fails-fast into a re-run instead of pinning a runner for the 6-hour default; caching
(Playwright browser cache keyed on the package version, like the Postgres inputs the
same job already caches) removes the fetch entirely from the steady state. **Timeouts
applied** in `.github/workflows_src/ci.yml` (the editable mirror — owner copies over):
apt mingw ×2 (15 min) + Playwright install (10 min); the cache half remains open.

### I33 — `jit_killpath_stops_runaway_child` flaked under full-workspace parallel load (S4) — **RESOLVED** (2026-07-22): the kill-path escape in the JIT `join` returned a clean `0` instead of propagating `OutOfFuel`; original report below

**Where:** `crates/svm/tests/jit_killpath.rs::jit_killpath_stops_runaway_child`, during a full
`cargo test --workspace` on a loaded machine (the phase-3 IMPORTS migration's local gate). 4/5 of
the binary's tests passed; this one failed exactly once.

**Not reproducible in isolation:** the single test passed 3/3 re-runs and the whole `jit_killpath`
binary passed 5/5 consecutive runs immediately afterward on the same tree. The failing run's
machine was near its disk quota and running many test binaries concurrently, so the likely
mechanism is timing pressure on the kill path (a watchdog/deadline racing a deliberately-runaway
child), the same runner-pressure family as I3/I4.

**Action if it recurs:** capture the panic message (it scrolled out of the grep-filtered log this
time — re-run with the failure line preserved), and consider serializing the binary's tests like
`imports.rs` does (ISSUES.md I4 pattern) or widening the kill deadline under load.

**Recurred in CI (macos-latest), 2026-07-20** — PR #407, head `de02a9d0` (`build · test
(macos-latest)` job 88426158563), same test, this time with the panic captured: expected
`Trapped(OutOfFuel)`, got `Returned([0])` — the runaway child *completed* before the watchdog's
interrupt landed, i.e. the deadline raced the kill signal exactly as hypothesized. Unrelated diff
(the failing head touches only svm-llvm test fixtures).

**Third sighting + mitigation landed, 2026-07-20 (PR #407):** recurred again in a local full
`cargo test --workspace` (same test, passed 3/3 in isolation immediately after). Three sightings
in one day across two platforms, all under parallel load, none in isolation → applied the
prescribed I4-pattern mitigation: `jit_killpath.rs` now serializes its five tests behind a
process-local mutex (every test there races a wall-clock watchdog against runaway guest code, so
sibling-test CPU contention distorts exactly what it measures). Leave this issue **open** until
the serialized binary holds green across several full-workspace/CI runs; if it flakes *while
serialized*, the deadline itself is too tight — widen it next.

**Fourth sighting — flaked *while serialized*, 2026-07-22 (PR #414):** recurred on `build ·
test (macos-latest)` (run 29937973810, head `3b60cd00`), same test, same captured signature
(expected `Trapped(OutOfFuel)`, got `Returned([0])` — child completed before the interrupt
landed). The serialization mitigation is in place, so per the note above this is now the
"deadline too tight under load" signal, not sibling-test contention. Unrelated diff (the head is
the capability-model `iface::→cap_id` rename + intern pre-seeding — no JIT-fuel/kill-path code
touched; that run's Linux main gate, both macOS fiber jobs, and all differential/browser jobs
passed, and a local full `cargo test --workspace` was green). **Next action is now pinned:** widen
the runaway-child kill deadline (or raise the child's fuel headroom so the watchdog reliably wins
the race on a loaded runner) — a focused, security-reviewed change on its own PR, since it tunes a
TCB-adjacent fuel/kill assertion.

**ROOT CAUSE FOUND + RESOLVED (2026-07-22).** The four prior sightings all read this as a *timing*
flake ("the child completed before the interrupt landed / the deadline is too tight"), but that
model is wrong: the runaway child (`PARENT_WITH_RUNAWAY_CHILD`'s func 1) is a **true infinite loop**
— it can never "complete", so a widened deadline / more child fuel would not have helped. The real
mechanism is a **race in the JIT `join`** (`crates/svm-jit/src/instantiator_rt.rs::join`). The
non-durable child now runs **asynchronously on its own OS thread** (PROCESS.md S1c), so the parent
*parks* in `join` on the child's completion cell. When the watchdog fires the kill, two paths race:
(a) the child's own epoch poll trips `OutOfFuel` and fills its completion cell → `join` breaks with
that trap → `Trapped(OutOfFuel)` ✓; **(b)** the parent's `join` loop observes `epoch_fired` *first*
and **returned a bare `0` with `*trap_out` unset**, on the assumption the parent "traps `OutOfFuel`
at its next epoch poll." That assumption holds for a *spinning* caller (a back-edge poll follows),
but the parent module does `cap.call join` then `return` with **no** intervening back-edge or
function-entry — so there is no next poll, and the `0` flows straight out as a clean `Returned([0])`.
Under parallel load the parent-side escape (b) wins the race more often, hence "only under load,
never in isolation." This was a genuine (if narrow) kill-path correctness gap — a host firing the
§5 interrupt on a parked-in-`join` parent would silently see `Returned` instead of `OutOfFuel` — not
merely a test-timing artifact.

**Fix.** `join`'s `epoch_fired` escape now sets `*trap_out = TrapKind::OutOfFuel` before returning
(one line + the SAFETY/why comment), so the kill surfaces as `Trapped(OutOfFuel)` regardless of
whether a subsequent guest poll exists — making *both* race outcomes (a) and (b) deterministic. The
child bakes the same interrupt cell and unwinds on its own; it is joined at run teardown
(`join_children`), so nothing leaks or hangs. Only the **kill** branch changed — the sibling freeze
escape (`window_is_unwinding`) in `os_thread_rt::thread_join` still returns `0` deliberately (a
freeze must reach the guest's safepoint, not trap). Verified: `jit_killpath` 5/5 green, then the
runaway-child test alone **60/60** serial + **36/36** across 6 parallel rounds under saturated CPU
load (the condition that used to flake it); `jit_killpath_threads`, `jit_instantiator`,
`jit_instantiate_{cache,granted,named}`, and `jit_coroutine` all green; clippy clean. The
`jit_killpath.rs` test serialization from the third sighting is kept (harmless — the other four
tests there still race their own watchdogs). The `os_thread_rt::thread_join` kill escape has the
same latent shape but is not exercised by any current test (both thread tests spin after join, so a
back-edge poll always follows); left as-is, noted here for the next time it matters.


### I30 — Rare Linux-CI linker crash: `rust-lld` dies with SIGBUS while linking `svm-jit` test binaries (S4) — seen on the `build · test · fmt · clippy` job (2026-07-18)

**Where:** the gating `build · test · fmt · clippy` job (ubuntu-latest), during `cargo test --workspace`'s
**link** step for `svm-jit`'s test binaries (`bulk_mem`, `bench`, `specialize`) and `svm-capi` (lib test).

**Symptom.** The bundled LLVM linker crashes mid-link:

```
collect2: fatal error: ld terminated with signal 7 [Bus error], core dumped
  ... rust-lld ... libLLVM ... llvm::parallelFor(...) ...
error: could not compile `svm-jit` (test "bulk_mem") due to 1 previous error
```

with an LLVM crash backtrace (a `PLEASE submit a bug report to llvm-project` note). Exit 101.

**Why it's a flake, not our code.** A SIGBUS *inside the linker* is a runner-level fault (a truncated
`mmap`/page-in of an object file under memory/disk pressure — `svm-jit` pulls in the large Cranelift +
Wasmtime rlibs, the heaviest link in the tree), not a miscompile. The failing run's only change vs. the
prior green run was a `.mjs` file in the **detached** `browser` workspace, which cannot affect
main-workspace linking; every other job compiling the same workspace (windows, macOS, real-browser)
linked fine on the same commit. Distinct from the macOS-launch SIGBUS entry below (that one crashes a
*test binary at launch*; this crashes the *linker at build time*, on Linux).

**Fix sketch.** Transient — re-run the job (a fresh commit / "Re-run failed jobs" clears it). If it
recurs, reduce link-time memory: cap the linker's parallelism or split the heaviest test binaries. Log
recurrences here to judge whether it needs a durable mitigation vs. staying a re-run-and-move-on flake.

**Recurrence (2026-07-23, PR #422 run 30030308082):** same job, harder death — the runner was lost
48 s into `cargo test --workspace` (step stuck "in_progress", job concluded `failure`, **no logs ever
uploaded**, likely the OOM-killer taking the runner agent during the parallel link phase). Same
commit's windows/macOS/miri/llvm jobs all green, and the identical job was fully green on the parent
commit 19 minutes earlier with only a test-fixture resize + docs in between. Second sighting —
if a third lands, take the durable mitigation (cap link parallelism / split `svm-jit` test bins)
rather than re-running.

**Third sighting (2026-07-23, run 30034429088) — durable mitigation prepared, blocked on token
scope.** Identical death 51 s into the same step on the immediate retry (code-identical tree; the
interleaved run between the two deaths passed in 8 min — an alternating pass/die pattern consistent
with OOM raciness under runner neighbor pressure). UI note: the job *name* contains "fmt", so the
PR checks list reads as a fmt failure — the fmt/clippy/build steps were green; the death is in the
test step's link phase. Per the rule above the fix is capping the gating job's test-build
parallelism — change ci.yml line `- run: cargo test --workspace` (the `check` job) to
`- run: cargo test --workspace -j 2` — bounding concurrent heavy links (the memory peak; the step
is warm-cache dominated, so the wall-clock cost is small). **The CI token cannot push workflow
files** (`refusing to allow an OAuth App to ... without workflow scope`), so the edit lives in
**`.github/workflows_src/ci.yml`** (the editable mirror — see its README; the owner copies the
directory over `.github/workflows/`). If a fourth death lands *with* the cap, the next escalation
is splitting the heaviest `svm-jit` test binaries.

### I25 — QuickJS BigInt (`libbf`) is miscompiled through the LLVM on-ramp: wrong results / hangs (S2) — **RESOLVED** (2026-07-18) — found by the QuickJS breadth harness (2026-07-17)

**Where:** the LLVM on-ramp on Bellard's QuickJS 2024-01-13, the `libbf` bignum path (BigInt).
`crates/svm-run/demos/quickjs/` — the engine otherwise runs a wide slice of JS byte-identical to
native (`demo_quickjs_breadth_vs_native`: regex, generators, `try`/`catch`, `Map`/`Set`, closures,
destructuring, `JSON`, `Object`/`Array` methods, `Date`, integer `Math`). **BigInt is the one known
JS-surface gap** and is deliberately excluded from the breadth demo.

**Symptom.** Even a trivial BigInt is wrong, so this is a fundamental `libbf` op, not a complex one:
- `(7n).toString()` → `"128000000000000000008"` (should be `"7"`); `(2n+3n).toString()` →
  `"96000000000000000008"` (should be `"5"`) — the `~2^value`-scale garbage smells like a mantissa/
  exponent normalization gone wrong (libbf stores value as mantissa × 2^exp).
- `(6n*7n)` **hangs** (a non-terminating loop, presumably a normalization/carry loop fed a wrong
  limb count).

**Ruled out.** The primitives libbf leans on are individually **correct** on the on-ramp (verified
guest == native over a focused probe): `__builtin_clzll`/`ctzll`, 64-bit variable shifts, and the
128-bit multiply (`(__int128)a*b` → lo/hi). So the bug is a subtler libbf-specific pattern (candidate
suspects: `bf_set_si`/`bf_normalize` bit-length + shift, `bf_ftoa` base conversion, a struct-layout /
`memcpy` of the `bf_t` limb array, or an `add/sub_overflow` carry idiom), not one of those ops.

**Root cause — the translator dropped the high 64 bits of a large i128 constant.** Bisected exactly as
the fix sketch below suggested (`bf_probe.c` linking `libbf.c` + `cutils.c`): `bf_set_si`/`bf_add`/
`bf_mul`/`bf_ftoa` are all correct, but the BigInt *literal* path (`bf_atof`, string → bf_t) gave
garbage — `(7n)` parsed to `128000000000000000008` with `expn=67`. Narrowed through `bf_atof` →
`bf_mul_pow_radix` (the `T·radix^expn` finalizer, wrong only on the **negative** exponent / division
branch) → `bf_div` → `mp_div1norm` → **`udiv1norm`** (the Granlund–Montgomery divide-by-invariant).
Its `a = (a1<<64|a0) − q·d − d` folds `2^126` as an i128 constant subtrahend; `i128_parts` (and the
i128-φ path) split an i128 constant into `(lo, hi)` with **`hi` hardcoded to `0`**, silently dropping
every bit ≥ 2⁶⁴. So `2^126 − x` came out with a zero high limb → wrong quotient → wrong BigInt. (The
stale comment claimed `llvm-ir` held the value in a `u64`; the textual `.ll` reader holds the full
`u128`.) The 128-bit *multiply* was fine because it never took a large i128 **constant** operand.

**Fix.** `i128_parts` and the aggregate-φ i128-constant case now split out **both** limbs
(`lo = value as i64`, `hi = (value >> 64) as i64`). Regression `i128_large_constant_operand` (hand-C,
interp+JIT: `2^126 − x` vs a `u128` oracle) fails on the old code, passes now; and
`demo_quickjs_bigint_vs_native` (`#[ignore]`d, wall-clock) diffs the real `libbf` BigInt path
(literal `toString`, `+`, `*`) byte-for-byte against native. BigInt is no longer a JS-surface gap.

**Fix sketch (as pursued).** Reproduce headlessly by linking `libbf.c` alone with a tiny driver that
calls `bf_set_si` + `bf_ftoa` (base 10) and diffing guest vs native — that isolates set/normalize/format
from the JS layer. Then bisect the first miscompiling `bf_*` primitive.

### I23 — svm-jit miscompiles some rustc-emitted bitcode: an in-bounds heap `Vec` access faults / returns garbage (S2) — **RESOLVED** (2026-07-16; re-verified + extended 2026-07-18) — found by the `bench/rustbench` real-program harness (2026-07-14)

**Where:** the LLVM on-ramp + `svm-jit` on **rustc**-produced bitcode (rustc 1.81, LLVM 18 — the
version the on-ramp's `llvm-dis` reads). Not seen on any `clang`-produced module; the five other
`rustbench` workloads (hashmap/vm/sort/parse/base64) cross-check identical to native and Wasmtime.

**Symptom.** A tiny, fully in-bounds program traps with `MemoryFault` where native returns the right
answer — the confinement faults on a *legitimate* access (so a bad address computation, not an
overrun). Minimal reproducer (prepend `bench/rustbench/prelude.rs` for the bump allocator/panic
handler):

```rust
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let v = vec![3i32; 100];          // 400 bytes in a 32 MiB arena — nowhere near the guard
    let mut h = 0i64;
    for _ in 0..n { for &d in v.iter() { h = h.wrapping_add(d as i64); } }
    h
}
// native run(3) = 900 ; svm-jit run(10) = Trapped(MemoryFault)
```

Independent of element type (`u8`/`i32`/`i64` all fault) and opt level (`-O0`..`-O2`). The `bfs`
workload (kept in `workloads/bfs.rs`, disabled in the driver's `WORKLOADS` — grep `I23`) hits a
variant that **returns garbage** (`8160438656660` vs `881260`) instead of trapping. **Distinct from
I21**: that is a bulk-op span *overrunning* `mapped`; this is a small in-bounds access.

**Root cause — two distinct `svm-llvm` *translation* bugs** (not svm-jit codegen: all three backends —
tree-walker, bytecode, JIT — reproduced identically, so the IR itself was wrong). Both are
opaque-pointer / auto-vectorizer patterns that clang happened not to emit but rustc does:

1. **Constexpr-GEP stride ignored the source element type** (the *fault*). `const_gep_offset` strode
   index 0 by the *pointee* type of the base `GlobalReference` instead of the GEP's own source element
   type. With opaque pointers `getelementptr (i8, ptr @HEAP, i64 8)` strides by `i8` (⇒ +8), but the
   evaluator scaled 8 by `sizeof(@HEAP)` (the 32 MiB `[33554432 x i8]`) ⇒ a 256 MiB offset, far past
   the window ⇒ `MemoryFault`. Fix: `ConstGetElementPtr` now carries `source_element_type` (parsed,
   previously dropped) and `const_gep_offset` strides by it — mirroring the instruction-form
   `translate_gep`.

2. **2-lane vector min/max compared the packed word, not per lane** (the *garbage*). A `<2 x i32>`
   packs into a scalar `i64` (lane 0 low). `llvm.smax`/`smin`/`umax`/`umin` on it fell through to the
   *scalar* i64 min/max path, comparing the whole 64-bit word: `smax(<-1, 3>, 0)` kept the `-1` lane
   because `0x0000_0003_FFFF_FFFF` is positive, and `zext`ing that lane gave `4294967295` — the source
   of the huge sums. LLVM produces this from the auto-vectorized `if d > 0 { h += d }` clamp
   reduction (`smax(d, 0)` over an `i32` slice with negatives). Fix: the intrinsic lowering now
   scalarizes a 2-lane operand per lane (explode → scalar min/max → repack), like the 128-bit path.

**Resolved (2026-07-16).** Both fixed in `crates/svm-llvm`; regression tests
`vec2_minmax_per_lane` + `constexpr_gep_i8_element_stride` (hand-`.ll`, interp+JIT) added. The `bfs`
workload now cross-checks byte-identical to native (`881260`) on all three backends and is **re-enabled**
in the driver's `WORKLOADS`.

**Re-verified + extended (2026-07-18).** Re-ran the full `rustbench` suite on the modern default
toolchain (rustc 1.94 / LLVM 21): all six workloads (bfs included) cross-check byte-identical to native
— the two original bugs stay fixed. To hunt for *sibling* rustc/auto-vectorizer patterns, added a fast
correctness-only harness **`bench/src/bin/rustprobe.rs`** (rustc `--emit=llvm-ir` → `svm-llvm` →
`svm-jit`, vs a native build, over a spread of `n`) with eight diverse stress workloads
(`bench/rustbench/probes/`: reductions, checked/saturating arith, bit intrinsics, slice adapters,
floats, structs+enums, wide auto-vectorized math, static tables). It surfaced **two new translate gaps**
— both clean fail-closes (the on-ramp refused rather than miscompiling), now fixed in `crates/svm-llvm`:
1. **LLVM 21's constant-splat shorthand `<N x T> splat (T v)`** (rustc emits it pervasively from
   min/max clamps + elementwise ops; clang's textual output doesn't) — the `.ll` reader now parses it,
   expanding to the explicit `<T v, …>` vector. Regression `vec2_splat_constant`.
2. **2-lane `<2 x i32>` `llvm.abs`** wasn't scalarized (only the min/max siblings were, from the
   original I23 fix) — now explode → per-lane `select(x<0,-x,x)` → repack. Regression `vec2_abs_per_lane`.
All eight probes now cross-check byte-identical; **no new wrong-value/trap miscompile was found.**

### I24 — the LLVM on-ramp was effectively pinned to LLVM 18 — **RESOLVED** (2026-07-17): the textual reader is version-tolerant; consumers feed `.ll` text and skip `llvm-dis`

**Resolution.** The pin was never in the reader — it was the **`llvm-dis` bitcode step**. Our in-house
`.ll` reader already parses modern textual IR: validated by translating the five `bench/rustbench`
workloads emitted by **rustc 1.94 (LLVM 21)** — all parse, verify, and run **byte-identical to native**
on all three backends (e.g. `bfs` = 881260). So the fix is to route around `llvm-dis`:
- **`svm-llvm`** — `translate_ll_path` (already public) reads `.ll` text with no `llvm-dis` and no
  version coupling; `translate_bc_path` now shells the **newest** `llvm-dis` on `PATH` (`best_llvm_dis`,
  newest-first probe; a newer tool reads older bitcode too; `SVM_LLVM_DIS` overrides) instead of a
  hardcoded one, so a `.bc` from a newer producer works wherever a matching `llvm-dis` is installed.
- **`bench/rustbench`** — the svm-jit lane now emits `--emit=llvm-ir` (textual) and reads it via
  `translate_ll_path`; the `+1.81.0` pin is gone (all lanes use the **system default** rustc;
  `SVM_RUSTBENCH_RUSTC` overrides). Confirmed running under rustc 1.94/LLVM 21 with correctness green.

The `ast.rs` "pin is LLVM 18" note is corrected to "version-tolerant (validated LLVM 18–22)". Original
report below.

**Where (orig):** `svm_llvm::translate_bc_path` reads a module by shelling `llvm-dis` (LLVM **18** — the CI
`svm-llvm` job installs `llvm-18`/`clang-18`) to disassemble `.bc` → textual `.ll`, then parses the
`.ll` with the in-house reader.

**Symptom.** Bitcode from any LLVM ≥ 19 producer fails at disassembly, e.g. from current stable rustc
(1.94 → LLVM 21): `llvm-dis: error: Unknown attribute kind (102) (Producer: 'LLVM21.1.8-rust-1.94.1'
Reader: 'LLVM 18.1.3')`. So the on-ramp only consumes LLVM-18-or-older bitcode. This is why `rustbench`
pins **rustc 1.81** (the last LLVM-18 Rust release) for its svm-jit lane, and why any consumer must be
held to an LLVM-18 toolchain.

**Impact.** A maintenance drag that worsens as LLVM advances: new Rust/clang can't feed the on-ramp
without an old-toolchain pin, and the pin ages out of support. Not a correctness or escape issue —
purely which producers the frontend accepts.

**Fix sketch.** Options, cheapest first: (a) bump the on-ramp's build tools to a newer LLVM
(`llvm-dis`/`clang`) and confirm the `.ll` reader parses the newer textual IR — the reader is
in-house, so the surface to re-check is the new attributes/opcodes, not a libLLVM link; (b) make the
`.ll` reader forward-tolerant (skip unknown function/param attributes, which are semantically inert
for the subset we translate) so it survives minor IR drift; (c) if staying on 18, document the pin
prominently (a `translate_bc_path` version check with a clear error beats a raw `llvm-dis` failure).
Track the LLVM version as an explicit, bumped dependency rather than an implicit `apt` default.

### I3 — Windows CI memory-pressure aborts under `cargo test --workspace` (S3) — **FIX LANDED & MERGED** (audit PRs, 2026-07-08); **holding** — green on all 6 post-fix nightlies (Jul 9–14), not yet proven eliminated (see Confirmation below)

**Where:** `crates/svm/tests/durable_jit.rs::freeze_thaw_cross_backend_over_generated_modules`
(the no-nightly cross-backend freeze/thaw driver), via `support/durjit.rs::fuzz_one_xbackend` →
`svm-jit` compile + guest-window commit. Windows runners only.

**Symptom:** intermittently the test binary aborts mid-run with
`memory allocation of 131072 bytes failed` followed by exit code `0xc0000409`
(`STATUS_STACK_BUFFER_OVERRUN`). Observed on PR #70 (a `svm-peval`-only change that cannot touch
this path); the exact base commit was green on the same job, and Linux/macOS always pass — i.e. a
flake, not a regression.

**Root cause.** Each of the 64 seeds JIT-compiles ~3× and commits a fresh guest window, so the
process's *cumulative* committed VA climbs across the run. On a memory-tight Windows runner the
commit limit (`os error 1455`) is reached, and the **next ordinary heap allocation** — here a
128 KiB (`131072`) `Vec`/`Box` — gets a null back. Rust's global-allocator OOM path
(`handle_alloc_error`) then **aborts** the process, which Windows reports as
`STATUS_STACK_BUFFER_OVERRUN`. This is the same Windows eager-commit memory-pressure *family* as
**I1** and shares its abort signature, but a **distinct** site: I1 was the fiber control-stack
`VirtualAlloc` (now fallible → `Trap::FiberFault`); this is a generic heap allocation that cannot be
made to trap gracefully — once commit is exhausted, *some* allocation aborts. The test already
*bounds* the pressure (seed count capped at 64; the heavier recycled variant is
`#[cfg(not(windows))]`-gated) — that mitigation is just still marginal on the tightest runners.

**Fix sketch (deferred — re-run clears it):**
1. Reduce the Windows blast radius further: lower the seed count behind `#[cfg(windows)]` (e.g. 32),
   or drop the JIT window reservation size for this driver so each commit costs less VA.
2. Reclaim VA between seeds — free/unmap each compiled blob + guest window before the next seed
   instead of letting them accumulate for the whole test (the libFuzzer target does the heavy run
   anyway, so the in-tree smoke needn't hold every artifact live).
3. Or split the driver so each seed (or small batch) runs in its own process, capping peak commit.

Until then, treat a `STATUS_STACK_BUFFER_OVERRUN` / `os error 1455` abort in this specific test on
Windows as a flake: re-run the failed job (`rerun_failed_jobs`).

**Scope update (2026-07-08 CI-flakiness audit over runs Jun 3 – Jul 8).** This entry is written
against `durable_jit`, but the same Windows memory-pressure family is the repo's **#1 CI failure by
far** and hits at least five other test binaries. Observed in the run history:

- `jit_fuzz` (`jit_matches_interp_on_generated_modules`): the most frequent single offender — the
  256 KiB/128 KiB alloc-abort (`0xc0000409`) killed main pushes 27078313769, 27230183986,
  27231558406, 27343150519, 27573684058, 28162141664, nightly 28575211654, plus one explicit
  `window commit failed (err 1455)` (27225507614).
- `fiber_fuzz` (`generated_migration_schedules_agree_on_interp_and_jit`): "fiber stack VirtualAlloc
  failed" (`svm-fiber/src/stack_windows.rs:42`) — runs 27584519722, 27568759548.
- `jit_threads`: svm-vcpu worker threads panic "fiber stack VirtualAlloc failed" in
  `fiber_rt::fiber_new` (a **nounwind** path, so the panic is an instant process abort that kills the
  whole binary) — runs 27716659364, 27713453924.
- `jit_diff`: thread stack overflows `0xc00000fd` in `return_call_indirect`/`rem_s_int_min_neg_one`
  (28166517444) — same pressure, different symptom.
- `durable_jit` itself: 27585086455 (heap alloc), 27581152487 (`window commit failed (err 0)`),
  27583202387 (`freeze_thaw_cross_backend_over_generated_modules` seed-panic that cleared on retry).

Frequency: 6 of the 6 fail→pass re-runs in the audit window were this family; 15 of 104 PR CI
failures failed **only** the `build · test (windows-latest)` job with every other lane green; ~10
main-push failures. **Escalation signal:** run 27716659364 (`claude/durable-active-resume-chain`,
commit `e549ea6`) failed identically on **both** attempts — at that commit the exhaustion was
reproducible, not transient. Severity should be treated as **S3** now (it is the dominant
PR-blocking failure and consumes a manual re-run each time), even though each incident is S4.

Additional fix levers beyond the sketch above (they apply to the whole family, not just
`durable_jit`): cap `cargo test` parallelism on Windows (`--test-threads` / `-j`) so concurrent
binaries don't stack their commit charge; shrink the per-window reservation/commit sizes under
`cfg(windows)` in test drivers; make `fiber_rt::fiber_new`'s allocation-failure path report/unwind
instead of nounwind-aborting the whole test binary (turns a process kill into one failed test); and
consider a larger runner or explicit pagefile bump for the windows lane. (The `fiber_new` item
was already delivered by I1's fallible `Stack::new`, landed Jun 19 — all "fiber stack VirtualAlloc
failed" abort sightings above pre-date it.)

**ROOT CAUSE FOUND (2026-07-08): the JIT leaked its entire code arena — 256 MiB of
eagerly-committed VA — on every compile.** cranelift-jit deliberately *leaks* all code memory when
a `JITModule` is dropped (its `Memory::drop` `mem::forget`s every allocation so stale `fn`
pointers can never fault); reclaiming requires the explicit unsafe `free_memory()`, which
`svm-jit` never called — a comment even asserted the opposite ("`JITModule` frees its executable
memory on drop"). Both compile paths install a 256 MiB `ArenaMemoryProvider` (the
i32-relocation-overflow mitigation), and on Windows the region crate allocates it
`MEM_RESERVE | MEM_COMMIT` (noted in cranelift's own `arena.rs`) — so **every JIT compile
permanently charged 256 MiB against the system commit limit**. A fuzz/differential loop pins the
runner's commit ceiling within dozens of compiles; from then on the arena alloc fails (silently
falling back to the small system provider — itself leaked on drop), *unrelated* heap allocations
abort (`memory allocation of N bytes failed` → `0xc0000409`, killing the whole test binary),
fiber-stack `VirtualAlloc`s return null, and window commits fail `os error 1455` — every symptom
in this family, including the "different test binaries, same abort" spread above. On Linux/macOS,
overcommit hid the identical leak as unbounded VA growth: measured at **+4.9 GiB of address space
over 50 differential iterations** before the fix, **0 MiB** after.

**Fix (landed on this branch):** `OwnedJit` — the `JITModule` owners (`CompiledModule`,
`ChildCode`) now call cranelift's `free_memory()` on drop. Sound because both structs already pin
the lifetime contract "nothing that points into the code may outlive the struct" (the module
field is declared/dropped last, after the runtimes/tables/trampolines whose addresses are baked
into the code). Regression-pinned by `crates/svm/tests/jit_code_memory.rs` (Linux: VA growth over
a 50-iteration compile loop must stay < 512 MiB; the Windows commit exhaustion is the same leak
seen through eager commit charging).

**After windows-lane confirmation:** re-test and lift the mitigation caps in the "skips & caps"
inventory (the reduced Windows iteration counts, and the `#[cfg(not(windows))]` recycled
cross-backend fuzz — its cranelift PC-relative-drift rationale was *also* this leak accumulating
address-space distance between arenas). Watch whether I15 (`pal::release` fragment flake) and the
`jit_diff` thread stack overflows disappear with the pressure gone. Also watch the nightly ASan
lane: freeing on drop turns any latent stale-pointer use (previously masked by the leak) into a
reported use-after-free instead of silent luck.

**Confirmation (2026-07-14, follow-up detection).** The fix merged to `main` 2026-07-08 (audit PRs
#172/#179/#181/#185). The **last observed I3 abort was the Jul 2 nightly** (28575211654): `build ·
test (windows-latest)` died at `jit_fuzz-…​.exe (exit code: 0xc0000409, STATUS_STACK_BUFFER_OVERRUN)`
— the canonical signature. Since the fix, the `windows-latest` lane has been **green on all six
nightlies (Jul 9–14)** and there were **no `windows-latest` re-runs** across the sampled PR/push runs
(Jul 2–13; the only re-runs in-window were I22 `real-browser`). Consistent with the fix holding — but
I3 was ~14 % intermittent (15/104 PR runs), and a single nightly/day is weak coverage, so this is
**"holding, not proven eliminated."** Keep watching before lifting the Windows mitigation caps below;
downgrade S3→resolved only after a wider clean sample (e.g. a few weeks of PR windows lanes).

---

### I4 — Rare macOS-CI `SIGABRT` in the `svm-wasm` threaded-import test (S4, surface reduced) — `claude/vcpu-context-recycling`

**Where:** `crates/svm-wasm/tests/imports.rs::spawn_alongside_capability_import` — a `wasi:thread-spawn`
module that spawns 6 OS-thread workers, each doing a `Blocking` `cap.call` + `i64.atomic.rmw.add`, with
the root parking on `memory.atomic.wait32` until they finish. Runs on the JIT via
`svm_jit::compile_and_run_with_host`.

**Symptom (observed twice):** on PR #72's first slice-3.3 CI run, the `build · test (macos-latest)` job's
`imports` binary aborted with `signal: 6, SIGABRT`. Tests run in parallel, so the abort surfaced after
a *sibling* test (`import_handle_threads_through_call_indirect`) had already printed `ok`; the only test
in that binary still running — and the only one using real OS threads + futex wait/notify — is
`spawn_alongside_capability_import`. **Recurred** on PR #92 (run #887 attempt 1, commit `4d45f97`), an
exports-only change that touches no threading code: identical signature (`signal: 6, SIGABRT` in the
`imports` binary after the same sibling test's `ok`), macOS-only — Linux *and* Windows ran the same
`cargo test --workspace` green in that very run, and a plain re-run of just the macOS job (attempt 2)
passed. **Not reproduced deterministically:** it has always cleared on the next run, and macOS cannot be
run in this environment, so the root cause is not pinned.

**Suspected cause / mitigation (landed, now confirmed NOT a cure).** Slice 3.3 (multi-vCPU durable) began
creating the `SharedFiberTable` for `uses_fibers || uses_threads` (the durable vCPU-context allocator
lives on it). A `.map` over that table *incidentally* also built the **root vCPU's `FiberRuntime` and
published it as `CURRENT_RT`** for a thread-only module — behavior it never had pre-3.3. A fiber-free
module never resumes a fiber, so that runtime is dead weight, but it changed the threaded run's
setup/teardown surface on the spawning thread. The table-vs-runtime split was fixed in I4's original
slice: the **table** stays present for `uses_threads` (needed by the allocator), but the **runtime** is
built only for `uses_fibers`. The **PR-#92 recurrence post-fix rules this delta out** — the abort
reappeared with the runtime split already in place, on a change that cannot touch the threading path. So
the cause is a **pre-existing macOS-runner flake** in real-thread futex park/notify/teardown (or runner
memory pressure), not the slice-3.3 runtime delta. Severity stays `S4` (transient, re-run clears it).

**Next step if it recurs:** capture the macOS core/backtrace (the `imports` binary under
`RUST_BACKTRACE=full`, ideally `--test-threads=1` to localize which test aborts), and check whether it
is in futex park/teardown (`os_thread_rt::{thread_wait,thread_notify,join_all}`) or the guard/signal
path — distinct from the now-removed root-runtime delta and from the resolved I1 (fiber-stack alloc).
If it keeps tripping unrelated PRs' CI, the cheap unblock (until root-caused) is to de-flake the test
itself — serialize it (`--test-threads=1` for the `imports` binary, or a process-global lock so the
6-worker spawn doesn't overlap other tests) or lengthen the `memory.atomic.wait32` timeout — rather than
re-running the whole macOS job by hand each time.

**Sighting update (2026-07-08 CI-flakiness audit).** More macOS-only occurrences than the two above:
run 28183991685 (Jun 25, the PR #126 merge push to main) — the `imports.rs` binary died `SIGABRT`
after 8/9 tests passed, same signature; and three more macOS-`cargo test` attempt-1 failures that
cleared on plain re-run of the same SHA (runs 28019319661, 27835056463; 28069421356 is the PR #92
recurrence already recorded above). Four further PR runs failed **only** the macOS job with all
other lanes green (27687656906, 27776754171, 27778073561, 27837565343 — failing test not
re-verified per-run). macOS is the #2 flake source after I3; the de-flake sketch above (serialize
the `imports` binary) is now worth doing rather than deferring.

**Mitigation landed (2026-07-08, `claude/ci-flakiness-audit-fw9023`):** the de-flake sketch's
process-global lock — every test in `imports.rs` now takes a shared `serial()` mutex, so the
6-worker threaded test has the process to itself and a recurrence is localized to the single test
that held the lock (the interleaving that blocked attribution is gone). Root cause remains open;
if it recurs *serialized*, capture the core/backtrace per the next-step note above. Two things may
also make it vanish outright: I3's code-arena leak fix (memory pressure was one suspected trigger)
and the serialization itself (scheduler contention was the other).

**No recurrence since serialization (2026-07-14 audit).** Swept **60 main + 30 PR CI runs** spanning
2026-07-09 → 07-14 (the full window since the `serial()` mitigation landed 07-08): **zero** occurrences
of the I4 signature (macOS `SIGABRT` in `imports.rs`) on any lane. The only failures in that window
were unrelated — a browser-lane flake (**I22**), a review branch's own WIP breakage (`escape_oracle` +
`fmt`), and cancelled duplicate-trigger runs. Encouraging but not proof-of-cure: I4 was always
low-frequency (~8 sightings over *weeks*), so a clean ~6-day window is consistent with both "fixed by
serialization + I3's memory fix" and "hasn't rolled the dice enough." Keep open with a watch; treat as
likely-resolved. Downgrade to close only after a longer clean window (or a captured core if it recurs).

---

### I24 — Rare macOS-CI `Bus error: 10` (SIGBUS) at a test-binary launch under `cargo test --workspace` (S4)

<!-- Renumbered from I21 → I24 (2026-07-15): the I21 number was already held by the
     bulk-memory divergence (now Resolved, below), and I23 is the rustc-bitcode
     miscompile above; this entry collided, so it takes the next free id, I24. -->


**Where:** `build · test (macos-latest)`. Observed on PR #202 (run 28986379444, a durable
nested-freeze `svm-interp`/`svm-snapshot` change): after `tests/c_frontend.rs` passed 71/71, the
harness printed `Running tests/cap_self.rs` and immediately died —
`…​.sh: line 1: 25515 Bus error: 10   cargo test --workspace`, exit code 138 (128 + SIGBUS 10). **No
test in `cap_self.rs` ran** (no `test …` line, no `test result`); the crash is at the binary's launch,
before any test body.

**Why a flake, not a regression.** `cap_self.rs` is the §7 capability-reflection suite
(`count`/`get`/`resolve`/`label`) — no threads, no durable freeze, and nothing the PR's diff touches.
The **same** `cargo test --workspace` ran green on Linux (`build · test · fmt · clippy`, where
`cap_self` passed) and on `build · test (windows-latest)` in that very run; `cargo test -p svm --test
cap_self` passes locally (7/7). macOS-only, unrelated binary, clears on re-run — the same
**macOS-runner-crash family as I4**, but a distinct signature: SIGBUS (not SIGABRT), a *non-threaded*
binary, and a crash *at launch* rather than mid-run after a sibling's `ok`. That points away from I4's
real-thread futex-teardown hypothesis and toward a transient runner fault (a page-in/`mmap` SIGBUS, or
a bad static-init/dylib map on the shared runner) during test-binary startup.

**Not reproduced deterministically** (macOS can't be run-tested in this environment). **Next step if it
recurs:** capture the macOS core/backtrace for the `cap_self` binary's launch, and check whether it
tracks memory pressure like I3/I4 (it followed a large `c_frontend` binary). Until root-caused, treat a
`Bus error: 10` / exit 138 at a test-binary launch on the macOS job as a flake and re-run it. If it
keeps tripping unrelated PRs, the cheap unblock is the I4-style mitigation (or making the macOS
`cross-os` lane non-gating, as its comment already contemplates).

---

### I22 — Rare `real-browser` (Chromium/Playwright) flake: a worker vCPU traps (OOB / `unreachable` panic) (S4) — **FIXED (2026-07-15): double-free race on shared codegen stashes → emit-once-per-run under a spin-lock; verified green in real Chromium (retry + liveness backstop retained as defense-in-depth)**

**Where:** the `real-browser (Chromium via Playwright)` CI job — `browser/browser-test.mjs` driving
`web/index.html` + `web/play.html` in a headless Chromium under COOP/COEP. The wasm module is the
shared-memory THREADS build (`-Z build-std`, `+atomics`, imported/shared `WebAssembly.Memory`), so
every on-page check runs real vCPUs across Web Workers over one shared linear memory.

**Symptom:** intermittently one on-page assertion fails with a wasm **out-of-bounds memory access**
(a `RuntimeError` surfaced via the page's `pageerror`/`console` hooks) instead of its expected
result, so `browser-test.mjs` exits non-zero and the job goes red. Observed on **PR #229** (the
on-ramp-in-playground work, run 29048631247): the diff added editable-module plumbing and page
assets and could not touch the shared-window/Worker path the failing check exercises. It **passed
locally** on the same commit, every other lane was green, and it **cleared on a plain re-run**
(`rerun_failed_jobs`) — the classic flake signature. The exact trap site/offset was **not
captured**: the attempt-1 logs rolled off once the passing re-run replaced them, so which of the
`powerbox`/`threads`/`jit`/`inst`/`capio`/`wasmjit` (or `play/*`) checks tripped is not yet pinned.

**Why a flake, not a regression:** a shared `WebAssembly.Memory` grown (detached) by one Worker
while another holds a stale typed-array **view** is a known Chromium-timing hazard on this stack —
an `svm_alloc` that grows the memory invalidates every previously-taken `Uint8Array(buffer)` view,
and a Worker reading through a stale view (or racing the grow) reads past the new bounds. Under a
loaded CI runner the interleaving that exposes it is rare and non-deterministic; local single-machine
runs and re-runs almost never hit it. This is the browser analogue of the I3/I4 "shared-runner load
makes a rare interleaving surface" family, not a codegen or verifier defect.

**Fix sketch (deferred — re-run clears it):**
1. **Capture first.** Make `browser-test.mjs` dump, on any `pageerror`, the failing check id + the
   `RuntimeError` message/stack and (if reachable) the memory `byteLength` at failure, so the next
   recurrence self-identifies which check and whether a grow/detach preceded it — today we can't tell
   which assertion tripped.
2. **Harden the view discipline** in the page glue (`play.js`, `par.js`): take a **fresh**
   `new Uint8Array(eng.memory.buffer)` after *every* call that can grow memory (`svm_alloc`, run
   entry), never cache a view across an alloc — `runModule` already does this for the single-shot
   path; audit the Worker/`par.js` shared-window path for a cached view held across a grow.
3. If it keeps tripping unrelated PRs before root-cause, treat a wasm OOB in `real-browser` that
   passes locally as a flake and re-run the job; consider making `real-browser` non-gating only as a
   last resort (it is the sole real-Chromium proof, so keep it gating if at all possible).

**Sighting update (2026-07-13 CI-flakiness detection, runs Jul 2–13).** This is the **most frequent
flake in the window** — **4 occurrences in 5 days** (Jul 8–12), all on the `real-browser` job's
"Build threads module + run in Chromium" step, each a `[pageerror] …` followed by `FAIL:
page.waitForFunction: Timeout 30000ms exceeded` (exit 1). Three were PR re-runs that **failed on
attempt 1 and passed unchanged on attempt 2** — the textbook flake signature — and one struck the
nightly `schedule` lane (which is never re-run, so it just sat red):

- run **28973194295** att1 (Jul 8, PR, `claude/charming-johnson-pmlsnr`) — `memory access out of bounds`; att2 green.
- run **29042617187** att1 (Jul 9, PR, `claude/peaceful-lamport-vuz65e`) — `memory access out of bounds`; att2 green.
- run **29048631247** att1 (Jul 9, PR #229, above) — `memory access out of bounds`; att2 green.
- run **29186787532** (Jul 12, **nightly on `main`**) — **`[pageerror] unreachable`**, same timeout; sat red (nightly is not re-run). `real-browser` was green on the Jul 9/10/11/13 nightlies, so this is non-deterministic, not a regression.

**New information vs. the original report:** (a) the page-error is **not OOB-only** — the Jul 12
nightly tripped a wasm **`unreachable`** trap with the identical downstream symptom, so the entry's
"out-of-bounds" framing should be read as *"any guest trap surfaced via `pageerror`"* (consistent
with the stale-view/grow-detach hypothesis: a Worker reading through a detached view can land on any
trap, not just OOB). (b) It now hits the **nightly `main` lane**, not just PRs. (c) Frequency is high
enough (3 of the window's PR-blocking re-runs, plus a red nightly) that although each incident is
S4, `real-browser` is now a **recurring gating-flake** worth prioritising fix-sketch step 1 (capture
the failing check id + `RuntimeError` on `pageerror`) — the attempt-1 diagnostics are still rolling
off before anyone can pin the check, exactly as noted above, so we still cannot say which on-page
assertion trips.

**Investigation (2026-07-14): the failure mechanism, and why we can't tell which check.** Traced the
page glue. Two facts pin the mechanism:
- Every one of the 7 index-page items (`web/main.js`) runs inside a `try { … } catch { set(id,
  'fail', …) }`, so a trap on the **page** thread produces a clean `fail`, never a `pending` timeout.
  The observed symptom is always a **timeout** (an item stuck `pending`) ⇒ the trap is in a **Worker**.
- In `web/worker.js` the vCPU event loop called `ex.svm_par_run(v)` with **no guard**. A host-level
  wasm trap there — `memory access out of bounds`, or `unreachable` (which is what a `panic=abort`
  engine panic lowers to, matching the Jul 12 `[pageerror] unreachable` variant) — unwinds into the
  `async onmessage`, rejecting it. **A Worker's unhandled promise rejection does not fire
  `Worker.onerror` on the page**, so `par.js`'s per-vCPU promise never settles: `main.js`'s `await
  run(...)` hangs, the item sits `pending`, and the harness's 30 s `waitForFunction` times out with
  only a bare `[pageerror]` and no check id. (The tier-up call one branch over *is* already
  `try/catch`-wrapped → `svm_par_deliver_tierup_trap`, which is why tier-up traps report cleanly —
  confirming the unguarded `svm_par_run` as the escape.)

So I22 is **two problems**: (a) a rare shared-memory race in the engine that makes `svm_par_run`
occasionally trap/panic under a loaded runner (the deep root cause — still open, needs a captured
instance), and (b) a **diagnostics/robustness gap** that turns (a) into a silent, unattributable 30 s
hang — which is precisely why the fix-sketch's "capture first" has never had anything to capture.

**Landed (2026-07-14, first step — targets (b), the capture gap; low-risk, glue-only, no TCB):**
- `web/worker.js`: guard the `svm_par_run(v)` call. On a host trap, wake any joiner (store `2`=trapped
  into a non-root vCPU's completion slot + `Atomics.notify`, so a parent's `Atomics.wait` doesn't
  cascade-hang) and `postMessage({kind:'fail', why})`. `par.js` already maps `fail` → promise reject
  → `main.js` marks the item `fail` **with the trap text**, converting the silent hang into a named,
  diagnosable failure.
- `browser-test.mjs`: retain the `pageerror` texts and, on the first `waitForFunction` timeout, dump
  **which items are still `pending`** plus the captured pageerror(s) before failing — so even a hang
  that slips past the guard self-identifies the stuck check.

These do not change the passing path and cannot fix the underlying race; they make the **next**
recurrence name the check + carry the `RuntimeError`, which is the prerequisite for root-causing (a).
Not yet exercised in a real browser here (needs the `-Z build-std` threads wasm + Chromium); the
next CI `real-browser` failure — or a local threads build — is the validation.

**Root-cause (a) — investigation so far (2026-07-14).** Working the engine side (`browser/src/lib.rs`):

- **The `unreachable` variant is an engine *panic*, not a masked guest trap.** The crate is
  `panic = "abort"` (`browser/Cargo.toml`), which lowers every Rust panic to a wasm `unreachable`.
  So the Jul 12 nightly's `[pageerror] unreachable` is an engine-internal invariant violation
  (`unwrap`/slice-index/`debug_assert`) hit under a concurrent interleaving — a *different, more
  informative* signal than the `memory access out of bounds` variant (a corrupted/racy pointer or
  index producing an actual OOB linear-memory access). Both point at **shared mutable engine state**
  touched by `svm_par_run` while other Worker vCPUs run over the one shared memory.
- **The shared allocator is a *deprioritised* lead.** `svm_par_alloc` is just the Rust global
  allocator (`std::alloc::alloc_zeroed`, 16-aligned), whose dlmalloc control block lives in the
  shared linear memory — so concurrent allocs from different Worker instances *could* race. But
  THREADS.md 4b explicitly states "the thread-safe shared allocator was de-risked by 4b", and the
  demo passes thousands of times, so this is not the prime suspect without evidence. Candidate shared
  state to audit first is the cross-Worker engine bookkeeping reached from `svm_par_run`: the §22
  `Domain`/`ModuleSource`, the 4d `Mutex<Host>` powerbox, the completion-slot/join protocol, and the
  tier-up cross-instance state — anywhere a rare ordering leaves an index/pointer inconsistent.
- **Can't go further without a captured instance.** The precise panic site / OOB offset has never
  been captured (attempt-1 logs roll off; a bare `unreachable` carries no location). That is the gate.

**Landed (2026-07-14, second step — the capture enabler for (a); diagnostic-only, native-compiled):**
`browser/src/lib.rs` installs a **panic hook** (once, wasm-only via `cfg(target_arch = "wasm32")`, so
native/`#[should_panic]` test output is untouched) that formats the panic's `FILE:LINE:COL` + message
into a static buffer in linear memory, exposed by `svm_par_last_panic_ptr()`/`svm_par_last_panic_len()`.
A wasm `unreachable` unwinds to the host but leaves memory intact, so `worker.js`'s new trap handler
reads the buffer **after** catching the trap and appends `| panic: panicked at FILE:LINE: MESSAGE` to
the `fail` reason. Net effect: the next `unreachable` recurrence reports the **exact Rust source
location** instead of a bare `unreachable`, turning (a) from "unobservable" into "one recurrence away
from a stack-precise fix". Compiles natively under `-D warnings`; **not** yet exercised on the wasm
threads build (same validation path as the first step). The hook is alloc-free (formats into a stack
buffer); the accessors are read-only.

**Sighting update (2026-07-14, post-diagnostics) — the first recurrence since the two diagnostic steps
landed.** run **29337399591** att1 (Jul 14, PR #255, `claude/peaceful-lamport-vuz65e`) — `[pageerror]
unreachable`, then the harness dumped `[timeout] items still pending: tierup, jitcodegen, instcodegen`
before the 30 s `waitForFunction` timeout; att2 green on the unchanged commit. Two notes:
- **Fifth PR-side occurrence, and the second `unreachable` variant** (first on a PR — previously only
  the Jul 12 nightly), reconfirming "any guest trap surfaced via `pageerror`", not OOB-only.
- **Partial validation of the 2026-07-14 fixes.** The `browser-test.mjs` **pending-items dump (step b)
  fired** — this is the first recurrence to *name* the stuck checks (`tierup`/`jitcodegen`/`instcodegen`,
  the index-page JIT items). But the `worker.js` guard + panic-hook did **not** surface a named
  `fail | panic: FILE:LINE` — the symptom is still a bare `[pageerror] unreachable` with those items
  merely *pending*, not `fail`. So the wedge is not a caught trap in the guarded `svm_par_run` compute
  path; the next capture pass should check whether the diagnostics build was actually in this run's
  base and, if so, why the tier-up/codegen items hang without routing through the guard (a Worker
  promise that rejects on a path the guard doesn't wrap would still leave the page item `pending`).
- **Immediately reconfirmed — sixth occurrence, on a *docs-only* PR** (run **29343104313** att1, Jul 14,
  PR #260, `claude/peaceful-lamport-vuz65e` — this very entry): `[pageerror] unreachable`, pending items
  `jitcodegen, instcodegen`; att2 green. A change touching only `ISSUES.md` cannot affect the browser
  build, so this is **diff-independent beyond any doubt**. Across the two `unreachable` sightings the
  stuck items are consistently the **codegen** checks (`jitcodegen`/`instcodegen`, `tierup` in one),
  narrowing the Worker wedge to the JIT **codegen** path.
- **Seventh occurrence (2026-07-14, on the §22 float-codegen PR #256).** Same signature again:
  `[pageerror] unreachable`, pending `jitcodegen, instcodegen`, 30 s timeout; att2 green on the
  unchanged commit, local Chromium green repeatedly (i32 → 1136, f64 → 1136, both on emitted wasm).
  The added f64 codegen item churns more Workers per run, so the codegen-path race surfaced a touch
  more often — the same double-free wedge diagnosed below, not a float-path bug. (PR #256 now carries
  the root-cause fix directly, via the merge of the 2026-07-15 worker.js full-body guard.)

**ROOT CAUSE FOUND (2026-07-15) — a double-free race on the shared codegen stashes** — which answers
the sighting update's open question (why the codegen items hang without routing through the guard) and
is fixed by wrapping the *whole* worker handler, not just `svm_par_run`. The diagnostics paid off. Four `real-browser` re-runs on Jul 14 PR CI (runs 29346033162, 29343104313, 29337767633,
29337399591 — all att1 fail → att2 pass) now self-identify the stuck check (main.js runs items
sequentially, so the **first** still-`pending` item is the culprit; the rest never start):

| run | `[pageerror]` | first stuck item |
|---|---|---|
| 87129853255 | `memory access out of bounds` | **`inst`** (§14 confined children) |
| 87119735304 | `unreachable` (panic) | **`jitcodegen`** (§22 guest-JIT real codegen) |
| 87100018744 | `unreachable` (panic) | **`tierup`** (wasm-JIT tier-up) |

Those three items are exactly the three that call a per-Worker `svm_par_enable_*` setup function —
`svm_par_enable_jit` (tierup), `svm_par_enable_jit_codegen` (jitcodegen), `svm_par_enable_inst_codegen`
(inst/instcodegen). Each **emits wasm and `stash()`es it into a `static mut`** (`JIT_UNIT_WASM`,
`INST_UNIT_WASM` / `INST_ELIGIBLE`, and the tier-up stash). `stash()` (`lib.rs`) does
`std::alloc::dealloc(old_ptr)` then `Box::into_raw(new)`. The SAFETY comments call these stashes
"single-threaded per instance" — **that is the bug**: a plain (non-`#[thread_local]`) Rust `static`
lives in the **shared** linear memory at one fixed address, so every Worker instance sees the *same*
stash. Each Worker runs `svm_par_enable_*` in its own setup, concurrently, over one shared memory ⇒
two Workers read the same `old_ptr` and both `dealloc` it ⇒ **double-free / use-after-free** on the
shared allocator ⇒ heap corruption ⇒ a later `memory access out of bounds`, or a Rust panic
(`unreachable`) — matching both observed variants. Rare because the window (two Workers in
`enable_*` at once) is narrow; load-dependent for the same reason. The allocator being thread-safe
(THREADS.md 4b) does not help — a double-free of the *same pointer* is a logic error above any
allocator.

**Mitigation LANDED (2026-07-15) — stops the PR bleeding + guarantees diagnosability; engine fix
deferred (needs a real-browser build to verify, which this environment can't run):**
1. `browser-test.mjs`: retry the index page up to **3×** (reload between), logging every retry loudly
   (`[I22 retry] …`) so the flake stays visible per AGENTS.md. It clears on reload every time it's been
   seen, so this self-heals CI without a manual `rerun_failed_jobs`; a *real* regression fails all 3
   and stays red.
2. `browser/web/worker.js`: wrap the **whole** vCPU handler in a liveness backstop (not just the
   already-guarded `svm_par_run` loop) so a trap in the `enable_*`/instantiate/`svm_par_child*` setup
   can never silently hang the page — it wakes any joiner (fills the completion slot the parent
   `Atomics.wait`s on) and reports `fail` with the captured panic location.
3. `browser/src/lib.rs`: install the panic-capture hook at the top of the three `svm_par_enable_*`
   functions too (not only `svm_par_run`), so a *setup-time* panic reports its `FILE:LINE` instead of a
   bare `unreachable`.

**Recommended engine fix (follow-up, verify in a real browser):** stop the per-Worker re-emit race.
Either (a) emit each unit **once on the page** (single-threaded, before spawning Workers) and have the
Workers *read* the shared stash behind an `Acquire` that pairs with the page's `Release` — the
"per-instance" premise is false, so the page's stash is already visible to every Worker; or (b) guard
each `enable_*` emit+stash with a lock so the dealloc/realloc is serialized (each pointer freed once).
`#[thread_local]` would be the natural expression of the original intent but is **not** available: the
`wasm32-differential` CI job builds this crate on **stable**, so a `#![feature(thread_local)]` would
break it.

**ENGINE FIX LANDED + VERIFIED IN REAL CHROMIUM (2026-07-15).** Took approach (b), the localized one
(`browser/src/lib.rs`): **emit each codegen unit exactly once per run.** Every run's page-side powerbox
publisher (`svm_par_powerbox` / `_jit_codegen` / `_io` / `_inst` / `_none` — exactly one per run, all
single-threaded before any Worker spawns) bumps a `PAR_RUN_GEN`; each of the three `svm_par_enable_*`
now runs its emit under a **spin-lock** (`CODEGEN_LOCK`) and only if it hasn't already run for the
current generation — later Workers skip the emit and reuse the shared stash (identical bytes either
way). So each stash is written **once per run and never freed mid-run**, killing the double-free/UAF at
the source; the Workers' reads of the emitted bytes are stable, and the lock's `Acquire`/`Release`
makes the first Worker's write visible to the rest. A spin-lock (not a `Mutex`) so the page's own
`enable_jit_codegen` call — on the main thread inside `svm_par_powerbox_jit_codegen` — can never hit a
forbidden main-thread `Atomics.wait`; it is always uncontended (previous run's Workers are terminated
before the next run publishes), so it acquires without spinning. No new import, no ABI change, builds
on **stable** (`wasm32-differential`) and nightly alike.

*Verified locally in real Chromium* (nightly `-Z build-std` threads build + Playwright, the same lane
as CI `real-browser`): the full `browser-test.mjs` passes green — all nine index items incl.
`inst`/`tierup`/`jitcodegen`/`instcodegen` (the three flake culprits) PASS, byte-identical to the
interpreter, with **no `[I22 retry]`** triggered. Native `cargo check -D warnings` clean. (A large-N
before/after stress loop was attempted but the sandbox's browser subprocess launching degraded after
~40 launches; the functional green run on the real build + the principled once-per-run serialization
of the proven double-free are the evidence.) The retry + liveness backstop from the mitigation stay in
as defense-in-depth.

---

### I6 — JIT/interp trap backtraces are not labeled with the trapping fiber (S4) — on `claude/debug-jit-backtrace`

**Where:** the trap-time backtrace capture sites — `crates/svm-jit/src/trap_shim.c` (the SIGSEGV/BUS
handler + `svm_capture_explicit_trap`), `crates/svm-jit/src/mem.rs` (the windows VEH), and the §14
coroutine/fiber runtime (`fiber_rt.rs`).

**Is:** a trap-time backtrace (`last_trap_backtrace` / `run_traced`) gives the correct guest **frames**
regardless of which fiber/coroutine was running when the trap fired — the frame-pointer walk works on
whatever stack the trap is on, and Stage 3 already collects a spawned vCPU's capture into the `Domain`.
What's missing is a **fiber-id label** (DEBUGGING.md §5 W3 Stage 3 "names the right fiber under
work-stealing migration"): the backtrace doesn't say *which* §23/D57 migratable fiber the frames belong
to. Pure cosmetics — the frames themselves are right.

**Why it isn't a quick patch:** the capture runs in the low-level handlers (C signal handler, Rust VEH,
the explicit-trap helper), none of which have the running fiber's identity to hand. `fiber_rt::current()`
returns the thread-local `*mut FiberRuntime` but not a stable handle, and a fiber migrates across worker
threads, so the id must be read at capture time, not reconstructed after. Threading a "current fiber
handle" thread-local that the capture sites can cheaply read is the work.

**Fix sketch:** maintain a per-thread "current fiber handle" cell (set on each `cont.resume`/suspend
switch in `fiber_rt`), read it at capture time into the trap-frame thread-local alongside `pc`/`rets`,
and surface it (e.g. `JitFrameLoc`-adjacent or a `last_trap_fiber()` accessor) for the kill message.

---

_(I1 below is open-adjacent — its abort mechanism is fixed, but I3/I4 are residual same-family CI-abort
flakes. I2 resolved below.)_
### I7 — Rare deadlock/hang in the work-stealing fiber demos (CI flake) (S3) — **fail-fast + diagnostics LANDED** (`claude/charming-johnson-pmlsnr`); root cause still open (awaiting a captured wedge)

**Where:** the guest-built work-stealing schedulers run end-to-end through the `svm-run` binary —
`crates/svm-run/demos/work_stealing/work_stealing.c` (stackless tasks) and
`crates/svm-run/demos/steal_fibers/steal_fibers.c` (D57 stackful, migratable fibers stolen across
real OS threads) — and their product-path smoke tests `demo_work_stealing_runs` /
`demo_steal_fibers_runs` in `crates/svm-run/tests/run.rs`. The deadlock is in the
scheduler/fiber-stealing path (guest scheduler logic and/or the host `os_thread_rt` + fiber-steal
runtime), not in the demos' I/O.

**Symptom:** the demo process occasionally **never terminates** — the guest's worker threads wedge
with no forward progress, so the test's `Command::…output()` blocks indefinitely. Observed once on
the **Linux x86_64** CI `check` job (run 27778162761, the `cargo test --workspace` step), which hung
>1 h until the run was cancelled. It is **rare**: 0 hangs in 48 local back-to-back runs of both
demos, and the suite passed cleanly on other runs.

**Why only Linux CI sees it:** both tests are gated `#[cfg(all(unix, target_arch = "x86_64"))]`.
`macos-latest` is arm64 and `windows-latest` is non-unix, so **both skip these demos** — the Linux
x86_64 `check` job is the only CI lane that runs them, so a hang there shows up as a single stuck
job while every other job is green.

**Root cause (hypothesis, not yet confirmed):** a timing-dependent liveness bug — most likely a
lost-wakeup / missed-notification race between the steal path and the park/unpark of idle worker
threads (or in the guest scheduler's termination detection), exposed only under a particular
interleaving. Needs root-causing from a stuck instance (attach `gdb`/`lldb` and dump all thread
backtraces, or add steal/park tracing). The fiber/work-stealing **runtime is not modified** by the
argc/argv work (PR #66).

**Sensitivity clue (PR #66):** the race is sharp enough that a *tiny startup perturbation* flips it
from rare to frequent. PR #66 originally had the `svm-run` CLI seed the §3e args buffer (a few-byte
`init_mem` memcpy during window setup, before the guest runs) for **every** program, including these
`main(void)` demos. That harmless, never-read seeding — only a few microseconds of extra setup —
took the hang from "0 in ~50 sequential runs" to **reliable on the first iteration** under
`cargo test --test run --test-threads=8` (parallel load). Reverting to *not* seeding when there are
no actual program args (so a bare run is byte-identical to before) restored the rare baseline (≥6
clean parallel iterations). So whatever the root cause, it is acutely sensitive to worker-thread
start timing — a strong hint for a park/unpark or steal-loop wakeup race.

**Investigation (this session — narrowed, not reproduced).** Reviewed every primitive on the demos'
path and could **not** reproduce a wedge nor find a defect by inspection:
- **Guest scheduler logic is hang-free by construction.** *Both* demos **busy-spin** the worker loop
  (`while (atomic_load(&g_remaining) > 0) { …; if (!t) continue; }`) — they do **not** park idle
  workers, so the "park/unpark of idle workers" in the original hypothesis isn't even a code path here.
  `g_total`/`g_returns`/`g_remaining` are interleaving-invariant: every task is stepped exactly `STEPS`
  times and is, on each iteration, either completed (decrement) or re-pushed — no task is dropped or
  double-counted, so `g_remaining` always reaches 0 and every worker then exits. A *resume* bug would
  surface as a wrong total or a `FiberFault` **trap** (non-zero exit), **not** a hang.
- **The only blocking points are sound / loom-verified.** The guest `pthread_mutex` is a 2-state
  futex lock whose `__vm_wait32` re-checks the word **under the futex lock** (the classic
  unlock-between-cas-and-wait race cannot lose a wakeup — and the host `futex_wait` holds that lock
  across `still_eq()` + `waiters++` + `cv.wait`, so a `notify` can't slip in between). `futex_wait`/
  `futex_notify`, the fiber single-owner `Ownership::claim`/`suspend_to_pool` migration arbiter, and
  `thread_join`/`run_child` (set-state-under-lock + `notify_all`) are all textbook-correct and several
  are **loom-verified** (`loom_wait_notify_never_hangs`, `fiber_registry`). The §5 signal/`siglongjmp`
  guard is **not exercised** by a fault-free demo run.
- **Not reproducible here.** ~24 000 demo runs total — 800 (8-way) + 3 600 **pinned to one core**
  (`taskset -c 0`, maximal startup-interleaving pressure) + 20 000 (8-way, both demos, with a
  gdb-dumping watchdog) — plus **60 full `run.rs`-suite parallel iterations** (the CI load profile):
  **0 hangs, 0 wrong outputs.** Consistent with the once-ever CI sighting (~1e-3–1e-4/run) — the
  residual risk lives in something loom can't model (the cross-thread native stack switch, or runner
  memory-pressure/scheduler pathology, the same I3/I4 family), or it was an environmental fluke.

**Fix sketch:**
1. *(LANDED — fail-fast + diagnostics)* The demo smoke tests now run through `run_demo_failfast`
   (`crates/svm-run/tests/run.rs`): the `svm-run` subprocess gets `SVM_DEADLINE_MS=30000` (so a
   *guest-side* wedge — spinning **or** futex-parked, since `KILL_RECHECK` wakes a parked vCPU — is
   §5 detect-and-killed and exits non-zero with the kill diagnostic), **plus** a 90 s host-side
   process timeout backstop that, on expiry, **best-effort `gdb -p` dumps every thread's backtrace**
   (the root-cause data this entry asks for) and SIGKILLs the child. A healthy run is milliseconds, so
   neither bound trips normally (verified: all `run.rs` green, ~1 s). **Net: a recurrence can no
   longer hang the named tests, and it self-captures the thread dump** needed to finish the root cause.
   The CI `check` (30) / `cross-os` (45) jobs also carry a `timeout-minutes:` backstop now, so any
   *other* unforeseen `cargo test --workspace` hang fails in minutes instead of GitHub's 6 h default.
2. *(still open — needs a captured wedge)* Pin the root cause from the next dump (CI or a longer local
   soak): if a worker is parked in `pthread_cond_wait`/futex at capture time it's a lost-wakeup in the
   mutex/futex layer; if all workers are spinning in JIT code (`??` frames) with `g_remaining > 0` it's
   a guest termination-detection / steal-loop livelock; if the stall is host-side (a Rust frame in
   `os_thread_rt`/`fiber_rt`) it's the migration/teardown path. Then fix the specific race.

**Sighting update (2026-07-08 CI-flakiness audit).** A second wedge was found in the run history,
predating the fail-fast landing: run 27778162761 (Jun 18, `claude/llvm-c-breadth`, commit `d3360b4`)
— the ubuntu `check` job's `cargo test --workspace` sat wedged for **54 minutes** (17:41→18:35)
until manually cancelled; the re-run was also cancelled by a superseding push, so no diagnostics
were captured. That makes ~2 sightings in ~1,200 runs, consistent with the 1e-3–1e-4 estimate. The
`timeout-minutes` + `run_demo_failfast` backstops landed after this occurrence; the next recurrence
should self-capture the thread dump.

---

### I8 — svm-jit/Cranelift auto-vectorizes only to **128-bit** SIMD, ~2× behind native AVX2/AVX-512 on wide-vectorizable loops (S3) — `claude/svm-jit-alu-simd`

**Where:** the LLVM on-ramp's vector legalization (`crates/svm-llvm/src/lib.rs` `wide_vec_layout`/
`lower_wide`, the §17 fixed-128 `LegalizeTypes` analog) → svm-ir's fixed-128-bit `v128` (§17/D58) →
`svm-jit` lowering each `v128` to one SSE/NEON 128-bit op.

**Symptom.** A reduction (`vadd`: `s += k ^ seed`) compiled `clang -O2 -mavx2` runs ~2× slower on
svm-jit than the native binary, because the on-ramp splits LLVM's wide `<8 x i32>`/`<16 x i32>` vectors
into **128-bit chunks** (4×i32) and svm-jit emits 128-bit `paddd`/etc., while native uses 256-bit `ymm`
(AVX2) or 512-bit `zmm` (AVX-512). So the SVM stack *does* vectorize (contrary to my earlier bench
claim — see below), but at SSE width.

**Measured (ns/iter, same C kernels, one machine; svm-jit timed *compile-once* — see the bench fix
below). wasm is disambiguated into the full matrix — {wasm32, wasm64} × {V8/TurboFan, Wasmtime/Cranelift}
— because the *backend* is the whole story:**

| kernel | native AVX2 (256b) | wasm32 V8 | wasm64 V8 | wasm32 Wasmtime | wasm64 Wasmtime | **svm-jit** | bytecode | tree-walk |
|---|---|---|---|---|---|---|---|---|
| `xorshift` (scalar serial) | 1.69 | 1.92 | 1.92 | 1.99 | 1.99 | **1.63** | 62.4 | 108.2 |
| `vadd` (vectorizable)      | 0.041 | 0.096 | 0.096 | 0.147 | 0.147 | **0.18** | 47.5 | 52.5 |

(wasm32 ≈ wasm64 within noise on both engines — the memory model doesn't move compute throughput here.
Wasmtime's *Pulley* interpreter tier, measured but omitted, is ~16 / ~7 ns — an interpreter, not a peer
of the JITs.)

**Scalar: no deficit** — svm-jit (1.63) *beats* every engine including native (1.69).
**Vectorized: it's the backend, not svm-jit.** The matrix makes this clear: **Wasmtime uses Cranelift —
the same backend as svm-jit** — and lands `vadd` at 0.147, right next to svm-jit's 0.18 (the ~1.2×
residual is on-ramp reduction shape + the bench's per-run window alloc). **V8/TurboFan**, also 128-bit,
is ~2× faster than *both* Cranelift engines (0.096). So the vectorized gap splits cleanly:
- **~2× width** (native AVX2 256-bit vs everyone else's 128-bit) — the determinism / opt-in-mode story.
- **~2× backend** (Cranelift vs TurboFan vectorization quality) — and svm-jit ≈ Wasmtime, i.e. **svm-jit
  is already at the Cranelift ceiling**.

(This *corrects* an earlier note here that claimed svm-jit *beat* wasm on `vadd` at 0.083 — that lumped
"wasm" as V8 only, predates the compile-once timing fix, and isn't reproducible.)

**Is the residual 128-bit gap actionable? No — it's upstream Cranelift.** That svm-jit ≈ Wasmtime (same
backend) is the proof: `opt_level` is already `"speed"`, and the on-ramp emits a minimal clean
translation (clang's 2-accumulator unroll → one SSE op per lane op, no redundant moves). The ~2× vs V8
is Cranelift's vector instruction selection/scheduling, which **D36/D49 deliberately don't own** — the
same "we don't fork the backend" boundary as the wide-vector blocker. (`-O3` shrinks it a little via
better-scheduled IR, but using a *different* `-O` for the SVM rows than native/wasm would make the
comparison dishonest — the very thing the bench fix below removes.)

**Root cause — deliberate, not a miss.** The chunk width is fixed at 128 bits and **never
host-detected**, to preserve the interp↔JIT↔durable-fiber **determinism contract** (a frozen vector
register file must replay identically on any host, and the tree-walker oracle is scalar-128). Widening
to the host's native vector width would make results/snapshots host-dependent. So this is a
throughput-vs-determinism tradeoff, not a codegen bug. (Vector *support* itself — all six `VShape`s +
wide/sub-128 legalization — already landed; see Resolved **I2**.)

**Benchmark caveat that exaggerated it.** My `bench/cross-engine` SVM driver compiled the kernels with
`-fno-vectorize -fno-slp-vectorize` (following the stale LLVM.md §4 "MVP" pipeline note), which keeps
SIMD out **entirely** → the SVM rows looked *scalar*, not merely 128-bit. With vectorization enabled
the on-ramp emits `v128` IR and svm-jit lowers it to real SIMD. Two measurement hazards make the win
hard to see in that harness: (a) `vsum`'s known-content array gets **closed-form-folded** by Cranelift
(the opaque-pointer barrier doesn't survive LLVM→SVM), and (b) `svm_jit::compile_and_run` recompiles
per call, so a fast vectorized loop is swamped by compile jitter unless timed via `CompiledModule`
(compile once, run many).

**Fix sketch:**
1. **Doc/bench — LANDED.** The bench already vectorizes (`-fno-*-vectorize` gone) and `vsum`→`vadd` is
   fold-resistant (runtime seed, no array). The remaining hazard — `svm_jit::compile_and_run` recompiling
   per call, whose ~5–6 ms jitter swamped the ~0.1 ms vectorized signal even through the large/small
   subtraction — is fixed: a new `svm_jit::compile(m, func) -> CompiledModule` (compile once, run many)
   drives the JIT row in `examples/cross_engine.rs`. `vadd` now reports a clean ~0.18 ns/iter (≈0.5
   cycle/element) — the honest 128-bit-SIMD number. (A wider `-mavx2 <8 x i32>` also legalizes + runs
   correctly now via the two-chunk I2/I11 path, but the chunks stay 128-bit so it adds no throughput; the
   bench keeps `-O2`/one-v128 to make the width comparison clean.)
2. **Throughput — accepted as a future opt-in mode, gated on Cranelift.** A host-dependent
   (non-deterministic) SIMD mode that legalizes to the host vector width (256/512) is now a
   product-sanctioned direction (DESIGN.md §17): default stays fixed-128/deterministic, the mode is opt-in
   for runs that don't need replay/freeze-thaw/oracle. The blocker is **not** determinism (explicitly
   waived for that mode) but the backend — Cranelift's x64 has no YMM/ZMM register class, so there's
   nothing to lower host-native ops to. Revisit when Cranelift grows upstream wide-vector support; until
   then width-hungry work uses a host vectorized capability (§7/§13) or the GPU broker.

---

### I9 — svm-jit lacks LCG/geometric **recurrence strength-reduction**, so a pure `a = a*M + c` loop is ~8× native (S4) — `claude/svm-jit-alu-simd`

**Where:** `svm-jit` (Cranelift) loop codegen, vs `clang`'s x86 backend.

**Symptom.** The `alu` benchmark kernel (`a = a*1103515245 + 12345 + i`) runs ~1.9 ns/iter on svm-jit
vs ~0.24 ns/iter native — an ~8× gap that *looks* like an svm-jit deficiency.

**Root cause — a clang-specific optimization on a pathological kernel, not a general gap.** clang's
backend recognizes the linear-congruential recurrence and **collapses 4 unrolled steps into a single
multiply by `M^4`** (observed: the native loop is one `imul $0xee067f11` — `M^4 mod 2^32` — per 4
iterations, with the per-step constants folded into additive terms). The on-ramp ingests clang's
*mid-end* IR, which is unrolled 4× but **not** collapsed (4 separate `i32.mul`), and Cranelift doesn't
do the collapse either → svm-jit runs 4 muls / 4 iters at multiply latency. **This is the only kernel
where svm-jit trails native**: on serial loops clang *can't* collapse, svm-jit **matches or beats**
native — measured `xorshift` 1.61 vs 1.74 ns, `muldep` 1.28 vs 1.52 ns (svm-jit faster). LCG-shaped
hot loops are rare in real code, so this is low priority.

**Fix sketch (deferred):**
1. **Don't chase it in svm-jit** — recurrence strength-reduction is a niche backend optimization;
   implementing it in Cranelift/the on-ramp is high-effort, low-yield.
2. **Benchmark hygiene:** the `alu` kernel is unrepresentative (it rewards clang's collapse). Report a
   non-collapsible scalar kernel (e.g. `xorshift`) as the headline scalar-throughput number, where
   svm-jit ≈ native, and keep `alu` only as a "clang recurrence-collapse" demonstrator.

---

### I14 — on-ramp has no 128-bit integer (`__int128` / `i128`) support (S3) — found via Embench `aha-mont64`

**Symptom.** A `clang -O2` program that uses `__int128` fail-closes at translate with
`Unsupported("integer width i128 (i128+ unsupported)")`. Found via Embench `aha-mont64`, whose
`mulul64` does a 64×64→128 widening multiply (`(unsigned __int128)u * v`, then `>>64` / truncate for the
hi/lo halves) — clang lowers it to `zext i64→i128`, `mul i128`, `lshr 128, 64`, `trunc i128→i64`.

**Where.** There is **no 128-bit integer anywhere in the stack**: `svm-ir`'s scalar value model is
`I32 | I64 | F32 | F64 | V128` and the interpreter's `Value` enum matches it. The on-ramp rejects
`bits > 64` in `crates/svm-llvm/src/lib.rs` (`val_type`, ~line 1029), with the same wall in switch
lowering (`switch on i128`), the load/store width tags, and constant materialization. Integer widths
33–63 are handled today by living in an `i64` and masking after de-normalizing ops; 128 genuinely needs
a second word.

**Status (stopgap landed — `aha-mont64` only).** The `embench` example (`examples/embench.rs`) compiles
`aha-mont64` with **`-U__SIZEOF_INT128__`** (applied to *both* the native and SVM builds so the
differential stays honest). `mont64.c` has a `#ifdef __SIZEOF_INT128__` guard with a pure-64-bit fallback
`mulul64`, so undefining the macro routes it to code the on-ramp handles. (The fallback then exposed a
*separate, unrelated* gap — a constant-amount non-rotate funnel shift `fshl.i64(hi, lo, 1)` from
`modul64`'s double-word shift — which is now lowered in `lower_int_intrinsic`; see
`tests/translate.rs::funnel_shift_general_const`.) With both, `aha-mont64` translates and verifies
`OK (all engines = native, verify=1)`. The i128 piece is a **benchmark-harness workaround, not an engine
capability**: any `__int128` program without such a fallback still fails closed (which is correct —
fail-closed, never miscompile).

**Fix sketch (three tiers, by scope):**
1. *(landed)* Harness sidestep: `-U__SIZEOF_INT128__` for kernels with a 64-bit fallback. Zero engine
   work; gets `aha-mont64` green. Not a capability.
2. **Pattern-match the widening multiply** *(LANDED — `claude/onramp-i128`)*: the on-ramp now recognizes
   the idiom (`zext i64 → mul i128 → lshr 64 → trunc`) and lowers it to 64-bit ops without ever
   materializing a 128-bit value — `lower_i128_idiom` tracks each i128 SSA value symbolically (`Zext` /
   `WideMul` / `Hi`) and emits a concrete op only at the `trunc`: `mul` for a product's low half, an inline
   schoolbook `emit_umulhi` for its high half (the engine has no scalar high-multiply primitive, so the
   32×32 expansion is emitted in IR — self-contained in `svm-llvm`, no new op across the stack). Covers
   `aha-mont64`'s `mulul64` and the overwhelming majority of real `__int128` use (bignum, fixed-point,
   hashing, mulhi). Anything beyond the idiom — a full i128 `add`/`sub`/variable-shift, or an `xor`/`and`/
   `or i128` (which clang folds `(u128)…` bitwise combinations into) — still fails closed, never miscompiles.
   Tests: `translate.rs::{i128_widening_mul_hi, i128_widening_mul_lo_and_hi}`, bit-exact (interp == JIT) vs a
   `u128` oracle. *(The `embench` example still keeps `-U__SIZEOF_INT128__` for `aha-mont64`: `modul64`'s
   `__int128` **variable** shift is outside this idiom, so a full-kernel `__int128` build needs more than
   tier 2 — removing the sidestep should be validated against a real Embench checkout.)*
3. **General i128 legalization** *(LANDED — `claude/onramp-i128-tier3`, supersedes tier 2)*: every i128
   SSA value is now a materialized `(lo, hi)` i64 pair — the unified `agg`-pair representation already
   used by `load i128` / `icmp i128`. `lower_i128` lowers each op to 64-bit ops over the parts:
   `zext`/`sext` (any source ≤ 64) / `trunc`, `and`/`or`/`xor`, `add`/`sub` (carry/borrow via an
   unsigned-overflow compare), `mul` (the schoolbook 64×64 with `emit_umulhi`), double-word
   `shl`/`lshr`/`ashr` by a **runtime** amount (branchless via `Select`: within-word part + cross-word
   carry guarded for `m==0` + an `n≥64` word move + sign fill for `ashr`), and `icmp` **all predicates**
   (`hi <strict> | (hi == & lo <op_u>)`). i128 **function params/returns** ride clang's `{i64,i64}` ABI
   split through the existing `agg` machinery. Tests (`translate.rs::i128_*`): add/sub carry, full
   128×128 mul + bitwise, variable shifts across `[0,128)`, all compare predicates, and param/return —
   each **bit-exact, interp == JIT, vs a native `i128`/`u128` oracle`.
4. **Cross-block i128** *(LANDED — `claude/charming-johnson-pmlsnr`)*: an i128 SSA value now registers an
   `[i64, i64]` `agg_layout` (like a flat 2-field struct), so its `(lo, hi)` pair **fans out as two
   block params over an edge** — a **loop-carried `phi i128`** / live-across value — via the existing
   struct-φ machinery (`block_params`/`branch_args`), not just same-block. `agg_operand` also
   materializes a **constant i128 φ incoming** (`phi i128 [0, entry], …`) as `(lo, 0)`. Tests
   (`translate.rs`): `i128_cross_block_loop_accumulator` (an i128 LCG accumulator across a backedge,
   constant-0 entry) and `i128_cross_block_fib_pair` (two i128 φs — a Fibonacci pair — crossing
   together), both bit-exact interp == JIT vs a `u128` oracle.
5. **i128 div/rem** *(LANDED — `claude/charming-johnson-pmlsnr`)*: `udiv`/`sdiv`/`urem`/`srem i128` (clang
   keeps these as IR ops at `-O2`; the `__divti3`-family libcall is a *backend* lowering the on-ramp
   never sees) now lower to a synthesized **`__svm_udivmod128`** helper — a binary long-division loop
   over the `(lo, hi)` pair returning quotient **and** remainder in one pass (the first arithmetic synth
   helper, alongside `__svm_memcpy`/`__svm_utoa`). Division by zero **traps** (`DivByZero`, matching the
   scalar `i64` divide). Signed forms reuse it: the lowering abs-es the operands and re-signs (quotient
   negative iff signs differ; remainder takes the dividend's sign — C truncation toward zero). A
   `freeze i128` (clang emits it on the `udiv`/`urem` operands) is now an identity on the pair. Tests
   (`translate.rs`): `i128_udiv_urem` (small/large/high-word-divisor/divisor>dividend) and
   `i128_sdiv_srem` (all four sign combinations), each bit-exact interp == JIT vs a native `i128`/`u128`
   oracle.

6. **Wide / negative i128 constants — fail-closed guard** *(LANDED — `claude/charming-johnson-pmlsnr`;
   this was first a silent-miscompile soundness bug)*. `llvm-ir` 0.11.3 reads every integer constant
   through `LLVMConstIntGetZExtValue`, a **`u64`** — for a `bits > 64` literal it **silently truncates**
   to the low 64 bits on a *no-asserts* libLLVM (Ubuntu's `llvm-18` is `--assertion-mode OFF`; an
   asserts build would instead abort). The on-ramp then materialized `(low64, 0)`, **miscompiling** any
   i128 literal outside `[0, 2⁶⁴)` — verified: `x % (2⁶⁴+1)` ran as `x % 1 = 0`. (An earlier revision of
   this entry wrongly said it "fails the parse"; that only holds on an asserts-enabled LLVM.) The
   truncation is irreversible by the time we hold the AST, so the fix is a **fail-closed guard**
   ([`wideint`], an `llvm-sys` re-walk like [`blockaddr`]/[`di`]): a module holding an i128 constant
   `≥ 2⁶⁴` / negative is rejected with a clean `Unsupported` — never a miscompile. Constants in
   `[0, 2⁶⁴)` (incl. the loop-carried-φ entry `0`) round-trip from the exact low word and still run.
   Tests (`translate.rs`): `i128_wide_constant_fails_closed`, `i128_small_constant_still_runs`.

   *Supporting* (not just rejecting) wide constants would need the high word, i.e. patching `llvm-ir` —
   considered (a ~6-line vendored fork works) but **rejected as not worth vendoring ~12 k lines** of a
   third-party crate for a rare case; the guard restores soundness in ~80 lines of our own code. If wide
   i128 literals ever show up in real corpora, revisit the fork. With this, **i128 is feature-complete**
   in the on-ramp modulo that fail-closed case.

7. **Wide constants — fixed at the root** *(LANDED — PR #169, the textual-reader flip; LLVM.md §8
   Q1b PR4)*. The on-ramp now reads **textual `.ll`** with an in-house parser, and text carries
   integer constants at full width — so a `≥ 2⁶⁴` / negative i128 literal parses exactly and
   translates instead of fail-closing. The [`wideint`] guard and the `llvm-ir` dependency it
   compensated for are **deleted**; `i128_wide_constant_fails_closed` became
   `i128_wide_constant_now_translates`. (One pre-existing, newly *reachable* translator gap noted
   there: the runtime correctness of `i128 urem` by a >64-bit constant *divisor* — never exercised
   while the reader fail-closed on such constants.) With this, **I14 is fully resolved** at the
   input layer.

---

### I16 — libFuzzer `diff` target crashes on 1–4-byte inputs (S2 until triaged) — **TRIAGED: harness-level, not an escape; FIX LANDED & MERGED (2026-07-08); CONFIRMED** (green on 6 post-fix nightlies Jul 9–14 + deterministic replay)

**Where:** nightly `cargo-fuzz (escape-TCB targets)` job, target `diff`
(`fuzz/fuzz_targets/diff.rs`).

**Symptom:** libFuzzer "deadly signal" on tiny inputs, six separate nightly/dispatch runs across
the audit window — each found a *different* crashing input, so this is being re-found nightly, not
a single cached artifact: Jun 11 (27334653221) input `[0x54]`; Jun 14 (27493229934)
`[0x79,0x7C,0x00,0x02]`; Jun 15 dispatch (27563212001) `[0xAD,0xA9,0xAC]`; Jun 19 (27815739473)
`[0xE8,0x01,0xDE,0xCD]`; Jul 2 (28575211654) `[0x2A,0x93,0x00]`; Jul 4 (28701938264)
`[0x00,0x71,0x04,0x1C]`. Crash artifacts were written to `fuzz/artifacts/diff/` on each failed run
(e.g. `crash-9149fee…` on 27563212001). Nightlies Jul 5–8 were green, but fuzzing is
nondeterministic — absence of a crash is not evidence of a fix, and no commit in that window claims
one.

**Why S2-classified for now:** the fuzz lane exists precisely because these are **escape-TCB**
surfaces. A deadly signal (not an rss/timeout) reachable from a ≤4-byte input in the diff path is
presumptively a guest-triggerable host crash until triaged down.

**Triage (2026-07-08).** Reproduced on stable via `Gen::from_bytes` + `fuzz_one` (the same path the
target drives): the Jun 19 / Jul 2 / Jul 4 inputs still crashed; Jun 11 / Jun 14 no longer
reproduce (the byte→module mapping drifts as the generator evolves). **Root cause — a JIT
compile-time rejection of a verifier-valid module, not a guest-triggerable host crash.** Each
crashing input generates a `cap.call` to the Instantiator interface (type_id 6, ops 5/6/7 —
`instantiate_module` / `spawn[_demand]_coroutine_module`) whose declared sig has fewer args than
the op's contract. The verifier checks args against the *declared* sig only (it knows nothing of
host-iface shapes), but `svm-jit`'s `lower_instantiator` dispatches on `op` statically and indexed
the missing args at compile time → `JitError::Malformed` → the differential's "JIT failed to
compile a verified module" panic → libFuzzer "deadly signal". The interpreter, by contrast,
resolves the handle at runtime and CapFaults (the generated handle is garbage). So the S2 concern
is retired: no memory unsafety, no interp/JIT *result* divergence — but any real guest module with
such a call would run on the interpreter and fail to compile on the JIT, which is still a
backend-parity bug.

**Fix (landed on this branch):** `lower_instantiator` now validates the declared `(op, sig)` shape
against each op's contract (arg-prefix types + exact result types); any mismatch — including an
unknown op, matching the interpreter's default arm — lowers to an unconditional **runtime
CapFault** instead of failing the compile, with zero-value placeholders keeping the verifier's
value accounting for the (dead) rest of the block. All six recorded inputs are pinned in
`jit_fuzz.rs::DIFF_REGRESSIONS`, so the stable CI sweep replays them on every PR and the nightly
stops re-discovering them. Confirm by watching the next few nightly `fuzz(diff)` runs stay green.

**Confirmation (2026-07-14, follow-up detection).** Fix merged to `main` 2026-07-08 (`dd370eb`, audit
PR #172). The **last `fuzz(diff)` failure was Jul 4** (28701938264, `[0x00,0x71,0x04,0x1C]`) — before
the fix. Since the merge the `cargo-fuzz (diff)` lane has been **green on all six nightlies
(Jul 9–14)**. Stronger than fuzzing luck: the root cause (a compile-time rejection of a
verifier-valid `cap.call` shape) is *fixed*, and all six historical inputs are pinned in
`DIFF_REGRESSIONS` so the stable per-PR sweep now covers them deterministically. **Treating I16 as
confirmed resolved** — the S2 escape concern was already retired at triage; the residual JIT/interp
parity fix now has 6 clean nightlies + deterministic replay behind it.

---

### I17 — nightly bench lane red ~every night: cold/wasmtime rows drift past any tolerance (S4) — **FIX LANDED** on `claude/ci-flakiness-audit-fw9023` (cold row now info-only; baseline regen still pending)

**Where:** nightly `bench regression check (non-gating)` job — `bench … --check baseline.txt --tol 0.4`.

**Symptom:** 24 of the 25 failed nightlies in Jun 4 – Jul 4 include this job failing, always the
same shape: **cold-start** and **wasmtime** ratio rows exceed the 40 % tolerance (`alu` +72–92 %,
`memsum` +82–88 %, `scatter` +89–93 %, `alu_c` +44–54 %, `locals_c` +43–50 %, `hostcall` +38–41 %,
`hostbuf` +40 %), with magnitudes drifting upward over the month, while compute ratios stay in
tolerance — and several kernels (`simd`, `float`, `calli`, `cache`, `irreducible`) report
**MISSING** from the baseline entirely. `baseline.txt` was last regenerated Jun 19 (PR #86) and the
cold/wasmtime columns have drifted continuously since. The job is `continue-on-error`, so it never
blocks — but a lane that is red every night by construction can no longer flag a *real* gross
regression (its stated purpose), and it pads every nightly failure report.

**Fix:** regenerate `bench/baseline.txt` on the current bench machine including the missing
kernels; consider excluding the cold/wasmtime columns from `--check` (or giving them their own,
wider tolerance) — cold-start wall-clock on shared runners is exactly the noise the 40 % tol was
supposed to absorb, and empirically it does not.

**Landed (2026-07-08):** the second half — `check_baseline` now treats `cold/wasmtime` as
**info-only** (printed with its drift, marked `high (info-only)`, never fails the check): it
measures runner generation + external-wasmtime version drift, not our codegen, and it was the sole
gating-failure cause in all 24 red bench nights. The same-run svm/wasm compute ratios (the
machine-portable signal the baseline header itself calls the tracked one) still gate. **Still
pending:** regenerate `baseline.txt` on the designated bench machine so the five MISSING kernels
(`simd`, `float`, `calli`, `cache`, `irreducible`) get rows — MISSING never gated, but those
kernels currently have no regression tracking at all.

**Info-only half confirmed (2026-07-14 follow-up detection):** the fix merged 2026-07-08 12:59; the
Jul 8 nightly ran at 09:30 (before the merge) and still failed on the cold/wasmtime rows, but the
**Jul 9 nightly (29011551854) was fully green** — the first all-green nightly in the history and
direct proof the info-only change stopped the cold/wasmtime rows from gating. (Jul 10–14 bench reds
are the *unrelated* ambiguous-binary break below, not a tolerance failure.)

**Follow-up (2026-07-13 CI-flakiness detection): the bench lane is now red for a *different*,
deterministic reason — the `--tol` landing above never runs.** Since the Jul 10 nightly the `bench`
job fails **before executing any benchmark**, at the `cargo run` invocation itself:

```
error: `cargo run` could not determine which binary to run. Use the `--bin` option to specify a
binary, or the `default-run` manifest key.
available binaries: bench-vs-wasmtime, confine
```

Observed every night Jul 10–13 (runs 29086218690, 29146664268, 29186787532, 29242756076). Root
cause: PR #225 (`bench: reliable confinement-cost harness`, merged Jul 9) added a **second** binary
`bench/src/bin/confine.rs` alongside the existing `[[bin]] bench-vs-wasmtime` (`src/main.rs`). The
`ci.yml` bench step runs a bare `cargo run --release -- --check baseline.txt --tol 0.4` with no
`--bin`, and the crate has no `default-run`, so cargo now refuses. This is **deterministic, not a
flake** — but it fully **masks I17**: the lane dies before it can print any ratio, so neither the
cold/wasmtime info-only rows nor the gating compute ratios are produced (the Jul 9 nightly, the last
before #225, was the window's only fully-green nightly). Non-gating (`continue-on-error`), so it
doesn't block merges, but the nightly perf signal is currently dead. **Fix (one line):** add
`default-run = "bench-vs-wasmtime"` to `bench/Cargo.toml`'s `[package]`, or pass
`--bin bench-vs-wasmtime` in the `ci.yml` bench step.

**Fixed (2026-07-14):** added `default-run = "bench-vs-wasmtime"` to `bench/Cargo.toml`. Chose the
manifest key over an `--bin` in `ci.yml` because it repairs the **documented bare `cargo run`**
everywhere (the crate header + local workflow, not just the one CI line) and leaves `ci.yml` untouched
(bot pushes lack `workflow` scope — see I18). The confinement probe stays reachable as `cargo run
--bin confine`. Verified locally: the bare `cargo run --release -- --check …` that previously errored
instantly now resolves to the harness and proceeds to build (`cargo metadata` reports
`default_run = bench-vs-wasmtime`). The nightly `bench` lane will again reach the `--check` compare —
so I17's *actual* signal (the same-run compute ratios) resumes gating, and the cold/wasmtime info-only
drift resumes printing. The remaining I17 item is unchanged: regenerate `baseline.txt` so the five
MISSING kernels regain rows.

---

### I18 — CI transients: crates.io network resets and rolling-nightly toolchain breakage (S4)

Two environmental failure classes from the audit window, recorded so recurrences are recognized
instead of re-investigated:

1. **crates.io download reset.** Run 28253766023 attempt 1 (Jun 26, `embench differential` job,
   step "build the in-process Wasmtime runner"): `download of 3/s/syn failed … curl [56] Recv
   failure: Connection reset by peer` → exit 101; re-run of the same SHA passed. Any job doing a
   cold `cargo build`/`cargo install` can hit this.
   *Mitigation:* jobs already use lockfiles + `Swatinem/rust-cache`; add `CARGO_NET_RETRY=10` (and
   `CARGO_HTTP_TIMEOUT=60`) to the workflow `env:` so cargo itself rides out resets.
2. **`cargo install cargo-fuzz --locked` broken by the rolling nightly.** Jun 4–9 (runs
   26940471925, 27004283086, 27056872718, 27087106040, 27193280846) all 3–4 fuzz matrix jobs failed
   before fuzzing started: cargo-fuzz 0.13.1's locked `rustix 0.36.5` stopped compiling on the new
   nightly (`rustc_layout_scalar_valid_range_*` became reserved). Self-resolved upstream by Jun 11 —
   five nights of **zero fuzz coverage, silently**.
   *Mitigation:* pin the fuzz job's nightly to a dated toolchain (bumped deliberately), or cache
   the built `cargo-fuzz` binary keyed on that date, so lane health doesn't depend on
   `nightly-latest × crates.io` compiling at 07:00 UTC.

**Patch prepared (2026-07-08, attached to the audit PR):** both mitigations —
`CARGO_NET_RETRY=10` + `CARGO_HTTP_TIMEOUT=60` in the workflow-global `env:`, and the fuzz job's
toolchain pinned to `nightly-2026-07-01` (a deliberate-bump pin; the fuzz *targets* need nightly
features, not the newest nightly — the other nightly lanes keep the rolling channel). The change
touches `.github/workflows/ci.yml`, which bot tokens cannot push (no `workflow` scope) — a
maintainer needs to `git apply` the patch from the PR. Move to Resolved once applied and a few
nightlies confirm. If the dated toolchain ever lacks a component the job needs, bump the date
rather than reverting to the channel.

3. **Runner disk-full during `apt-get install` of the mingw-w64 Windows cross-toolchain.** Run
   29508205769 (Jul 16, `build · test · fmt · clippy` job, dependency-install step, before any
   build/test ran): `dpkg … cannot copy extracted data … failed to write (No space left on device)`
   while unpacking `gcc-mingw-w64-x86-64-*` → exit 100. Purely the runner's ephemeral disk filling
   during toolchain install; not a code failure (the same SHA is fmt/clippy/test-clean locally).
   Re-running on a fresh runner clears it.
   *Mitigation:* free space before the apt step (e.g. the standard
   `jlumbroso/free-disk-space` action or `rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android`),
   or install only the mingw packages actually needed. Workflow-file change (`workflow` scope), so a
   maintainer applies it.

### I26 — GitHub Pages deploy silently drops any playground asset not matched by `web/*.js` / `web/*.html`; nothing checks the published site (S3) — surfaced when the CodeMirror editor 404'd in production (2026-07-16)

**Where:** `.github/workflows/pages.yml` → the "assemble site" step. It hand-copies `web/*.html`
`web/*.js` (plus `web/assets/*.svmb`, the WAD, and the one wasm engine path) into `_site`, then
uploads that. Anything else under `web/` — a subdirectory, a `.css`, any file not on those two globs —
is never copied into the deployed site.

**Symptom.** #335 vendored CodeMirror under `web/vendor/…` (subdirectories + `.css`). Local dev
(`serve.mjs` serves all of `web/`) and the Chromium CI test (same server) were green, but the
**deployed** site 404'd every editor file and `editor.js` threw `Cannot read properties of undefined
(reading 'defineSimpleMode')`. The deploy path has **no automated check**, so "works locally + passes
the browser CI test" still shipped a broken production playground.

**Worked around (PR #340):** collapse the editor into a single top-level `web/codemirror.bundle.js`
(matched by the existing `web/*.js` copy) that also injects its CSS. That clears the immediate outage
but not the class of bug — the next asset added under a subdirectory or with a new extension will
silently 404 again.

**Fix sketch (needs `workflow` scope, so a maintainer applies it):** either (a) copy `web/`
**recursively** into `_site/web/` (`cp -r web "$SITE/"`, pruning anything that shouldn't ship) instead
of globbing two extensions; or (b) add a post-assemble gate that scans `play.html` / `index.html` for
every `<script src>` / `<link href>` / module `import` and fails the job if the referenced file is
absent from `_site`. (b) is the general guard — it turns a missing asset into a red deploy instead of
a published broken page.

### I27 — the playground editor smoke isn't gating CI; it exists but isn't invoked (S4) — `browser-play-editor-test.mjs`, added with the editor (2026-07-16)

**Where:** `.github/workflows/ci.yml`, the `real-browser` job. `browser/browser-play-editor-test.mjs`
(#335) drives `play.html` in Chromium — editor mount, SVM highlighting, a run, per-example language
switch, the parse-error gutter, and Vim — and *did* catch a mis-pathed vendored script (a `vim.js`
404) during development. It is **not** wired into CI: the automation that pushes these branches lacks
GitHub's `workflow` scope, so it cannot modify `.github/workflows/`.

**Fix (a maintainer with `workflow` scope):** add one line to the `real-browser` job, right after
`node browser-jit-reactor-test.mjs`:

```yaml
          node browser-play-editor-test.mjs
```

Until then, run it locally: `node browser/browser-play-editor-test.mjs` (after the wasm32 threads
module is built, per the job's build step).

### I28 — the Pages deploy rebuilds on-ramp assets that no test exercises, so an on-ramp/ABI change silently breaks every large demo (S3) — surfaced by the by-name `_start` grant break (2026-07-16)

**Where:** `pages.yml`'s `build on-ramp assets` step runs `build-onramp-assets.mjs`, which **rebuilds**
DOOM / Lua / SQLite / the GPU shader from current source at deploy time (they're gitignored). But the
only Chromium CI tests — `browser-test.mjs` / `browser-jit-reactor-test.mjs` — drive the **committed**
`.svmb` assets (hello_c/gradient/bounce/life/mandelzoom), never a freshly-built one.

**Symptom.** When the on-ramp switched to a by-name `_start` (322527c / S15) while the browser host
still granted the powerbox positionally, every freshly-rebuilt asset trapped (`status 3`) but the
committed assets kept working — so CI stayed green and the break shipped to the deployed playground.
The immediate case is fixed (PR #345, by-name grant), but the *class* of gap remains: any future
on-ramp/embedding-ABI change can re-break the large demos undetected.

**Fix sketch (needs `workflow` scope):** add a CI step that **builds a by-name on-ramp asset and runs
it** — either run `build-onramp-assets.mjs` (at least `hello_c`: fast, no SQLite fetch / DOOM build) and
drive it through the playground in Chromium, or gate a native `onramp_exec` test over a freshly-built
(not committed) fixture. Pairs with I26/I27 — all three are "the deploy/rebuild path has no automated
check." A cheaper partial guard already exists but doesn't gate: `browser/tests/onramp.rs`'s fixture is
now regenerated by-name, so `cargo test` in `browser/` catches the grant path — but `browser/` is a
**detached workspace** the main `cargo test --workspace` skips, so it needs its own CI lane to bite.

### I29 — the browser on-ramp host still carries the legacy **positional** `_start` grant path; dropping it needs the on-ramp to emit by-name for every guest (S4) — noted while fixing the by-name grant (2026-07-16)

**Where:** `grant_onramp_caps` (`browser/src/lib.rs`) supports **both** on-ramp entry forms — the S15
by-name paramless `_start` and a legacy **positional** one (its first `arity` handles passed as args).
The browser can't drop the positional path unilaterally: the current on-ramp still emits a positional
`_start` for some guests — `gradient`/`bounce`/`mandelzoom` translate to an arity-1 func 0 — so a
by-name-only host would trap them (arity mismatch / unresolved caps).

**Not yet root-caused:** why the on-ramp emits a paramless `_start` for `hello`/`life` (arity 0) but a
positional arity-1 one for `gradient`/`bounce`/`mandelzoom` is unclear — likely tied to which/how many
capabilities the guest imports, or a `main(argc, argv)` vs `main(void)` signature. Worth confirming
before the change below.

**Fix (svm-llvm, off-workspace lane):** make the on-ramp's `synth_*_start` emit the **by-name**
paramless entry (the S15 `synth_powerbox_start` / `synth_powerbox_start_for_imports` path) for *every*
guest, regardless of cap count or `main` signature. Once every emitted guest is by-name, drop the
positional branch from `grant_onramp_caps` and the `arity > 5` guard in `onramp_exec`, collapsing the
host to a single by-name grant; regenerate the committed `.svmb` fixtures/assets so they're by-name
too. `svm-run`'s `grant_caps` (which also still keeps a positional branch) can drop it in the same pass.

---

## Platform-coverage skips & caps — inventory (2026-07-08 audit)

Every place the suite deliberately runs *less* on some platform to dodge the failure families
above. Each is a tracked coverage hole: when the underlying issue (I3/I4/I7) is fixed, the cap
should be lifted; until then this is what Windows/macOS are **not** testing.

**Windows-reduced iteration counts (all motivated by the I3 commit-limit family):**

| Site | Windows | Elsewhere |
|---|---|---|
| `crates/svm/tests/jit_fuzz.rs:43` (JIT↔interp differential sweep) | 500 seeds | 4000 |
| `crates/svm/tests/fiber_fuzz.rs:331` (migration-schedule fuzz) | 400 iters | 1500 |
| `crates/svm/tests/fiber_fuzz.rs:462` | 80 iters | 250 |
| `crates/svm/tests/jit_threads.rs:576` (thread-spawn reps) | 10 reps | 30 |
| `crates/svm/tests/concurrent_escape_fuzz.rs:153` (concurrent escape programs) | 40 | 150 |
| `crates/svm/tests/durable_jit.rs` (cross-backend seeds, bounded per I3) | 64 | 64 |

**Windows-excluded tests:**

- `crates/svm/tests/durable_jit.rs:39` —
  `recycled_fiber_freeze_thaw_cross_backend_over_generated_modules` is `#[cfg(not(windows))]`
  (cranelift PC-relative relocation overflows `i32` under cumulative JIT allocation drift; see the
  in-file comment). Windows keeps partial coverage via the hand-written recycled test + the no-JIT
  400-seed interp fuzz, but has **no recycled cross-backend JIT fuzz** at all.

**Linux-only tests (`cfg(all(unix, target_arch = "x86_64"))`) — Windows *and* macOS skip these:**

- `crates/svm-run/tests/run.rs` (~4 sites, from :141) — the work-stealing fiber demos (the I7
  surface). Only the ubuntu `check` lane ever runs them.
- `crates/svm/tests/c_frontend.rs` (~4 tests, from :1900) — chibicc-built C end-to-end runs.
- `crates/svm-llvm/tests/translate.rs` (~10 sites, e.g. :2632–:2765, :3964–:4163) — the
  setjmp/longjmp-family and other JIT-adjacent on-ramp tests.

**Whole-crate platform holes:**

- `crates/svm-llvm` is **excluded from the root workspace** (root `Cargo.toml` `exclude`), so the
  `cross-os` jobs' `cargo test --workspace` never builds or tests it — the on-ramp has **zero
  Windows/macOS coverage** by design (its CI job is Linux-only; the harness shells out to
  Linux-installed LLVM 18 tools).
- `crates/svm-llvm` tests auto-skip at runtime when tools are absent (`tests/common/mod.rs:14`
  guard; ~30 `eprintln!("note: skipping …")` sites across `translate.rs`, `snprintf.rs`,
  `llvm_alias.rs`, `dap_over_llvm.rs`): missing `clang`/`cc`/`llvm-as-18` ⇒ silent skip; missing
  `rustc +1.81.0`/`llvm-link-18`/`opt-18` ⇒ the `peval_futamura`/`peval_jit`/`peval_in_sandbox`
  probes skip (documented in `ci.yml`). **Risk:** if a CI setup step silently stops installing a
  tool, these tests all "pass" while testing nothing — worth a canary assertion in the svm-llvm CI
  job that the expected tools were actually found. **Canary landed (2026-07-08):**
  `crates/svm-llvm/tests/ci_tool_canary.rs` — on Linux CI (`CI` env set) it asserts every tool the
  auto-skips probe for is runnable, naming the missing ones; a no-op locally so contributor
  machines stay unburdened.

**CI-workflow-level scoping (`.github/workflows/ci.yml`):**

- `fuzz`, `bench`, `ASan (svm-fiber)`, `TSan (svm-mem)`, `ASan (JIT setjmp/longjmp)` run **only** on
  `schedule`/`workflow_dispatch` — PRs get no sanitizer or fuzz coverage (accepted trade-off, but it
  means I16-class bugs land first and are found nightly).
- `cargo-audit` is gated off `pull_request` (deliberate, documented in-file).
- `loom`, `miri`, wasm32/wasm64 differentials, `browser-real`, `embench`, `cross-engine` are
  ubuntu-only lanes.
- The windows-**gnu** target gets `cargo check` + `clippy` only (no test execution); windows-MSVC
  tests run in `cross-os`.
- `bench` is `continue-on-error` (non-gating) — see I17 for why that lane is currently signal-free.
- Runtime capability gating: ~10 JIT test sites early-return when `svm_jit::fiber_supported()` is
  false (`jit_instantiator.rs`, `jit_killpath.rs`, `jit_trap_backtrace.rs`,
  `jit_separate_module.rs`, …) — correct-by-construction platform gating (single source of truth);
  `jit_diff.rs:831` asserts the gate matches the platform so silent regressions of the gate itself
  are caught (that assertion itself failed once on Windows: run 27225054386, Jun 9 — worth a look
  if it recurs).

**In-product mitigations that paper over runner pressure (fine, but they mask I3's frequency):**

- `crates/svm-jit/src/mem.rs:608-721` — bounded retry (6×, ~0.3 s backoff) on
  `ERROR_COMMITMENT_LIMIT` in the Windows commit path.
- `miri` job disables weak-memory emulation (`-Zmiri-disable-weak-memory-emulation`, documented
  Miri bug); ASan lanes run `detect_leaks=0` (documented intentional leak).

---

## Resolved

### I25 — wasm-JIT emitter rejected `return_call` (tail calls), silently dropping tail-call-ending functions to the interpreter (S3) — **fixed** (2026-07-15) — found profiling the Doom wasm-JIT reactor

**Was:** `crates/svm-wasmjit`'s `term_in_subset` accepted only `Br | BrIf | BrTable | Return |
Unreachable`, so any function whose IR ended in a `return_call` / `return_call_indirect` terminator was
classified out of the JIT subset — and it, plus its *entire direct-call subtree*, ran on the bytecode
interpreter instead of emitted wasm. Silent (the fallback is correct, just slow) and pervasive: at
`-O2` LLVM turns *any* function whose last statement is a call into a `return_call` (sibling-call
optimization). The Doom reactor measured **~88% of each frame interpreted** because its single hottest
function, `I_FinishUpdate` (the per-pixel scale + swizzle), ends in a call and was thus fully
interpreted along with its subtree.

**Fix (`crates/svm-wasmjit/src/lib.rs`):** lower `return_call` / `return_call_indirect` as the ordinary
call sequence (direct / cross-tier via `env.call_interp` / indirect through the identity table) leaving
the callee's results on the operand stack, followed by `return`. A tail call's callee results equal the
caller's results (the verifier guarantees it), so the stack matches the emitted function's declared
return type — a plain call + return, no wasm frame reuse, which is all the semantics require.
`term_in_subset` now accepts both; `func_uses_indirect` and the `needs_table`/type scans account for
`return_call_indirect` (it reuses the trampoline-populated table). Proven by `tests/differential.rs::
tailcall` (entry tail-calls a helper that tail-**recurses** — same-tier + self) and
`tests/cross_tier.rs::cross_tier_tail_call` (a `return_call` to a `v128`-gated cross-tier callee over
the shared window). Measured: Doom's whole `tick` now emits with **no guest changes** — ~7 fps
(interpreter) → ~47 fps (wasm-JIT), and ~50 fps with the display cap.call also isolated. This retires
the per-guest `-fno-optimize-sibling-calls` workaround and helps every on-ramp guest.

### I21 — JIT bulk-memory ops diverge from the interpreter on spans overrunning `mapped` inside the reservation (S2) — **fixed** (2026-07-15) — found by the SPEC.md slice-5 window-model suite

**Was:** `svm-jit`'s D62 bulk lowering (`confine_span` + the `memcpy`/`memmove`/`memset` libcall)
only bounded the span against **`reserved`**, delegating the `[mapped, reserved)` guard hole to
the libcall's own accesses. That leaked two interp↔JIT divergences (§3 parity, **not** an escape —
every byte stayed in `[0, reserved)`): (1) libc `memcpy`/`memmove` **short-circuit `dst == src`**,
so a self-copy overrunning `mapped` never faulted (the trap was *lost*); (2) the libcall wrote a
prefix before hitting the guard, so a faulting run differed from the interpreter's
fault-before-any-write. The interpreter's `Mem::check_prot_span` validates every page up front, so
it faulted correctly in both cases.

**Fix (`probe_span`, `svm-jit`):** before each bulk libcall, emit a guarded 1-byte read of each
span's **last** byte (`confine_span` base `+ len − 1`, guarded by `len != 0`). It faults
`MemoryFault` on the `PROT_NONE` guard exactly where the interpreter faults — **up front** (no
partial write) and **independent of the `dst == src` short-circuit**. The probe reads the *live*
page tables (a guard-page access), so it honors guest `grow` — unlike the compile-time
`Lower::mapped`, an explicit compare against which would over-fault a grown window. For the
contiguous backed prefix (the production model + the spec window) last-byte-in-bounds ⟺
whole-span-in-bounds, matching the interpreter's own `last < mapped` fast path. The slice-5 spec
suite (`crates/svm/tests/spec_mem.rs`) drops its JIT-leg carve-out and now pins the JIT window on
**faulting** runs too (untouched), byte-identical to the interpreters across the full boundary
lattice.

**Residual (documented, narrow):** a bulk op whose span straddles a guest-created **interior**
hole (`unmap`) or a read-only page mid-span is not caught by a last-byte *read* probe — the same
contiguity boundary the interpreter's fast path assumes. Left to a per-page probe loop if a real
workload ever needs bulk ops over a deliberately-punched window; not reachable from the spec
window or any current corpus.

### I5 — Windows JIT trap-time backtrace covers memory faults but not explicit-check traps (S3) — **resolved** (windows-latest confirmed green)

**Confirmed (2026-07-08):** the entry's own resolution criterion — a green `windows-latest`
`cargo test --workspace` with the un-gated `trap_kill_message_carries_a_source_backtrace`
(`crates/svm-run/tests/run.rs`, plain `#[test]`, no cfg gate) — has been met repeatedly since the
fix landed; most recently run 28967660183 (main @ `7b72216`, `build · test (windows-latest)`
green). MSVC runtime is validated. _Original entry below._

**Fix (landed, the refined-fix design below):** the trap-time capture state + frame-pointer walk +
explicit-trap helper moved into a new cross-platform `crates/svm-jit/src/trap_capture.c` (compiled on
unix **and** windows). `emit_trap` now bakes `call svm_capture_explicit_trap(get_frame_pointer())` on
every target — the trapping frame pointer is threaded in via Cranelift `get_frame_pointer` (so MSVC's
missing `__builtin_frame_address` is sidestepped), and the trap-site return address comes from
`_ReturnAddress()` (MSVC) / `__builtin_return_address(0)` (GCC). The unix signal handler and the windows
VEH both feed the shared capture (the handler via `svm_store_trap_frame`; the VEH keeps its Rust
memory-fault capture and the windows `take_trap_frame` falls back to the C `svm_take_trap_frame` for
explicit traps). The `trap_kill_message_carries_a_source_backtrace` test (div-by-zero) is now un-gated
on Windows. Unix validated locally; windows-gnu compiles; **MSVC runtime is validated by the
`windows-latest` CI job** — move this entry to Resolved once that job is green. _Original report below._

**Where:** `crates/svm-jit/src/lib.rs` (`trap_capture_addr()` returns `0` on non-unix, so `emit_trap`
bakes no explicit-trap capture call), `crates/svm-jit/src/trap_shim.c` (the unix-only
`svm_capture_explicit_trap`).

**Update (memory faults: fixed on Windows).** The Windows Vectored Exception Handler now captures the
trap-time backtrace for **memory faults**, mirroring the unix SIGSEGV/SIGBUS path: `mem.rs`'s windows
`pal::veh` reads the faulting `Rip`/`Rbp` from `EXCEPTION_POINTERS->ContextRecord` and walks the
frame-pointer chain (a Rust `walk_fp_chain`) into a thread-local before restoring the recovery context;
the windows `pal::take_trap_frame` reads it. So `last_trap_backtrace()` + the kill message now carry
source frames for a Windows guard fault (covered cross-platform by
`memfault_kill_message_carries_a_source_backtrace` in `svm-run`'s `run.rs`).

**Still open: explicit-check traps on Windows.** Div/rem-by-zero, `unreachable`, `OutOfFuel`, and
indirect-call-type traps store a `TrapKind` and return — there is no signal/exception to capture from, so
on unix the lowering bakes a `call svm_capture_explicit_trap` at the trap site (`trap_capture_addr()`).
On Windows that address is `0`, so these still produce an **empty** backtrace (correct `TrapKind` + kill,
no frames). Not a correctness or escape hazard. (The `trap_kill_message_carries_a_source_backtrace` test —
div-by-zero — keeps its source-line assertion under `#[cfg(unix)]` for this reason.)

**Why it isn't a quick patch (two concrete blockers, found on attempt):**
1. **Recovering the innermost frame without `__builtin_frame_address`.** The unix helper uses
   `__builtin_frame_address(0)` to find its own frame → the trapping fn's `rbp` *and* the trap-site
   return address (`[my_fp+8]`). **MSVC has no `__builtin_frame_address`.** Cranelift's
   `get_frame_pointer` (confirmed present in cranelift-codegen 0.132 x64) can hand the helper the guest
   fn's `rbp` as an argument — but walking from *that* yields only the **caller** chain; the trapping
   function's own line is lost. Recovering it needs the helper's return address (`_ReturnAddress()` on
   MSVC / `__builtin_return_address(0)` on GCC), which pulls the helper back into C.
2. **Cross-language capture state.** Windows memory faults capture into **Rust** thread-locals (the VEH
   is Rust, `mem.rs` windows `pal`); the unix explicit helper writes **C** thread-locals (`trap_shim.c`).
   A C explicit-trap helper on Windows would write a location the Windows `take_trap_frame` (which reads
   the Rust thread-locals) never sees. Unifying them is a capture-state refactor, not a patch.

**Refined fix (a proper slice, not a quick win):** unify the capture state in Rust (one thread-local set
read by `take_trap_frame` on both platforms; the unix C signal handler stores via a small async-signal-
safe `extern "C"` Rust shim), and make the explicit-trap helper take the frame pointer from
`get_frame_pointer` + the trap site from `_ReturnAddress`/`__builtin_return_address`. Then `emit_trap`
bakes `call <helper>(get_frame_pointer())` on **all** targets (de-special-casing unix too). **Test:**
un-gate `trap_kill_message_carries_a_source_backtrace` on Windows; validate on the `windows-latest` CI
job.

---

### I15 — Windows `pal::release` placeholder-fragment leak assertion flake (S4) — **resolved** (was already fixed before filing)

**Where:** `crates/svm-jit/src/mem.rs` lib test
`mem::tests::pal_release_frees_all_placeholder_fragments_no_leak`, Windows only.

**Symptom (observed once):** run 27291252672 attempt 1 (Jun 10, a push to main, commit `c29e07c`)
failed with `pal::release leaked 69632 bytes of the placeholder reservation (fragments past the
first not freed)`. A plain re-run of the same SHA passed; every other job in the run was green.

**Resolution.** Filed from the 2026-07-08 CI audit with a suspected split/coalesce bug — but the
real cause was already root-caused and fixed on **Jun 19** (`3dfb15e`, before the audit): a **false
positive in the test itself**. The no-leak check releases its reservation and then walks the VA
range asserting every byte is `MEM_FREE`; cargo runs unit tests in parallel, so a *sibling* test's
fresh reservation could land inside the just-freed range mid-walk and read as a "leak". The fix
serializes the reserving PAL tests behind `PAL_TEST_LOCK` (`mem.rs::tests`). The one recorded
sighting (Jun 10) predates the fix; none since. No production `pal::release` bug existed.

### I19 — TSan lane never ran: svm-mem doctests broke the build with a `-Zsanitizer` ABI mismatch (S4) — **fixed**

15 consecutive nightlies Jun 16–30 (27606473990 → 28430367633): the `TSan (svm-mem concurrency)`
job failed at build — rustdoc compiled the svm-mem **doctests** without `-Zsanitizer=thread`
against TSan-built deps ("mixing `-Zsanitizer` will cause an ABI mismatch", 18 errors). A toolchain
change around Jun 16 turned the mismatch into a hard error; before that the job passed. Net effect:
**no TSan coverage at all for two weeks** while the job showed generic red. Fixed by scoping the
job to `--tests` (commit `2197c7a`, Jun 30); nightlies green from Jul 1. Alternative had it recurred:
matching `RUSTDOCFLAGS`.

### I20 — ASan (JIT setjmp/longjmp) lane never ran: `package ID specification 'svm-llvm' did not match any packages` (S4) — **fixed**

6 consecutive nightlies Jun 25–30 (28156456664 → 28430367633): the job invoked cargo with
`-p svm-llvm` from the root workspace, which **excludes** `crates/svm-llvm`, so cargo errored
before building anything — no ASan coverage of the setjmp path those nights. Fixed by invoking via
`--manifest-path crates/svm-llvm/Cargo.toml --tests` (commit `2197c7a`, Jun 30); green from Jul 1. Lesson recorded
in the skips inventory above: lanes that fail during *setup* look like test failures but are
coverage gaps.

### I13 — `<2 x i32>` (packed-`i64`) lane arithmetic miscompiled (soundness, S2) — found via Embench `edn`/`fir_no_red_ld` — **fixed**

**Was:** Embench `edn`'s `fir_no_red_ld` ("no-redundant-load" FIR) carries a `<2 x i16>` across the loop
and auto-vectorizes its deinterleaved widening multiply to **`<2 x i32>` lane arithmetic**. `edn`
translated but returned a wrong answer (`verify_benchmark` = 1 native vs 0 on **all three** SVM engines —
so a translation bug, not an engine bug). Pre-existing and independent of I11; I11 merely let the *whole*
`edn` translate far enough to reach it.

**Root cause.** A 2-lane 32-bit vector (`<2 x i32>`/`<2 x float>`) is the one vector shape the on-ramp
carries *packed into an `i64`* (lane 0 = low 32 bits, lane 1 = high 32 bits) rather than a `v128` or a
legalized chunk+tail. Integer arithmetic on it fell through `bin` to a **single `i64` `IntBin`** on that
packed image — which is **not lane-wise**: `mul` mixes the lanes (the low product's carry and the
lane0×lane1 cross term corrupt lane 1), and `add`/`sub`/`shl`/`lshr`/`ashr` carry/shift across the 32-bit
lane boundary. (The earlier bisection fingered the carried-`<2 x i16>` φ because that φ is what forces
clang to *keep* the `<2 x i32>` shape — but the corruption was the `<2 x i32>` `mul`, not the i16 tail
lane or the φ fan-out, both of which round-trip correctly.)

**Fix (landed):** `bin` now lowers `<2 x i32>` integer arithmetic **lane-wise** — explode the packed
`i64` to its two `i32` lanes (`vec_explode`), apply the scalar `IntBin` per lane, repack (`vec_pack`).
The bitwise `and`/`or`/`xor` would be lane-safe even packed, but the path is uniform. The narrow φ
fail-close stopgap (a guard in `translate_function` that rejected a carried tiny all-tail sub-32-bit
vector) is **removed** — the pattern now translates correctly.

**Tests (`translate.rs`):** `simd_vec2_i32_carried_widening_mul_i13` compiles the real `fir_no_red_ld`
kernel and asserts the full **64-bit** checksum is bit-exact vs the native `cc` oracle on interp **and**
JIT (for two seeds); `simd_vec2_i32_lane_arith_add_shift_i13` covers `add`/`sub`/`shl` on an explicit
`vector_size(8)` `<2 x i32>` with lane values large enough that a packed-`i64` op would visibly corrupt
the high lane. End-to-end, Embench `edn` now reports `OK (all engines = native, verify=1)` in the
`embench` example.

### I11 — on-ramp fail-closed on auto-vectorized **wide vector shifts** (`shl`/`lshr`/`ashr` on `<8 x i32>`) (S3) — fixed on `claude/perf-i11-i12`

**Was:** a plain `clang -O2 -mavx2` (or `-O2` with interleave) program whose vectorizer emits a wide
integer shift — e.g. Embench `edn`'s `lshr <8 x i32> v, <i32 15, …>` — fail-closed at translate with
`Unsupported("type <8 x i32> …")`. The I2 legalization split wide loads/stores/arith/reductions/
conversions into `v128` chunks, but `lower_wide` had **no arm for shifts**, so a wide `Shl`/`LShr`/`AShr`
fell through to the normal `bin()` path, which only handles a single `v128` and rejected the 256-bit type.

**Fix (landed):** a `wide_shift` helper (mirroring `wide_int_binop`) splits a wide constant-splat shift
into one `VShift` per `v128` chunk + a scalar shift per tail lane, dispatched from new
`I::Shl`/`I::LShr`/`I::AShr` arms in `lower_wide`. The amount is taken from the constant splat (the shape
the auto-vectorizer emits; a non-uniform amount stays fail-closed, as in the v128 path). Verified by
`simd_autovec_avx2_wide_shifts` in `tests/translate.rs` (interp == JIT == native on a mixed
logical/arithmetic `<8 x i32>` shift) and a 10-op wide-op isolation sweep (shifts/sext/zext/trunc/
reduction/i16 — all bit-exact).

**Note:** this unblocked `edn`'s *shift* op, but `edn` as a whole still fails — it additionally trips
the **I13** `<2 x i16>` miscompile in `fir_no_red_ld`. (Separately, the on-ramp has no `memcmp`/`bcmp`
builtin — `clang` emits those for array compares; the Embench wrapper supplies them in-module with
`-fno-builtin-memcmp/-bcmp`. Providing them as on-ramp builtins, like `memcpy`/`memset`, is a small
coverage win.)

---

### I12 — the §9/D45 `cap.call` fast path left ~9× on the table for cheap caps by re-entering the generic host dispatch (S4) — fixed on `claude/perf-i11-i12`

**Was:** `cap_call` first reported the JIT generic and "fast" (`fast_cap_resolver`) paths as **within
~2%** — but that was a *benchmark artifact*: the probe's `cap.call` passed a stray arg, so it didn't
match the resolver's claimed `(CLOCK, 0, n_args=0, ...)` and silently ran the generic path *both* times.
With a correct **0-arg** `Clock.now()` call the fast path was already **~1.7×** generic (53→31 ns,
the JIT-side marshalling saving) — but the host side still re-entered `Host::cap_dispatch_slots`, which
for a cheap cap is dominated by the per-call `Vec` result allocation + the W1 record/replay gate.

**Fix (landed):** a new `Host::fast_clock_now(handle) -> Option<Result<i64, Trap>>` (svm-interp) does
the authority check (`resolve`, identical to the generic path — a forged/closed/wrong-type handle is an
inert `CapFault`) and the read+advance **inline**, returning the `i64` with no `Vec`. It returns `None`
when a W1 record/replay tape is active, so `svm_run::fast_clock_now` falls back to the full
`cap_dispatch_slots` and the clock crossing is still taped/served faithfully (the clock is a recorded
nondeterministic input). Net: `Clock.now()` on the fast path drops **31 → 5.7 ns** (a further ~5.5×),
so the fast path is now **~9× cheaper than generic** end-to-end.

**Verification.** `cap_call` now shows jit-generic ≈ 54 ns vs jit-fast ≈ 5.7 ns. New
`crates/svm-run/tests/fast_cap.rs` pins interp == generic-JIT == fast-JIT on a 0-arg clock delta and
that a forged handle still faults; the interp↔JIT differential (`svm/tests/jit_diff.rs`, 54),
`jit_quota` (fast-resolver path), and all `svm-run`/`svm-durable` clock tests stay green. (`Blocking.work`
still uses the shared `fast_dispatch` — it's arg-bearing and rarer; same inline treatment is a future
follow-up if it shows up hot.)

---

### I10 — ordinary `clang -O2` auto-vectorized loops hit two narrow holes in the vector breadth (S3) — fixed on `claude/bench-alu-hygiene`

**Where:** `crates/svm-jit/src/lib.rs` (v128 lane-arith lowering) and `crates/svm-llvm/src/lib.rs`
(vector integer-op translation in `bin`).

**Was.** A plain `clang -O2` program (vectorization on — *not* hand-written SIMD) fail-closed when the
loop vectorizer turned a common scalar loop into vector ops the I2 breadth didn't cover:

1. **`i8x16.mul` — svm-jit `Unsupported("instruction")`.** A byte-array fill like
   `for (i) buf[i] = i*31 + 7;` (`unsigned char buf[256]`) vectorizes to a `<16 x i8>` body whose
   multiply becomes `i8x16.mul`. svm-jit lowered `v128.load/store/const`, `i8x16.add/extract_lane`, and
   `i32x4`/`i64x2` multiply — but **not the 8-bit packed multiply** (x86 has no `PMULLB`). Translation
   *succeeded*; only the JIT lowering was missing.
2. **vector integer shifts — on-ramp `Unsupported("vector integer op ShrU (only add/sub/mul/and/or/xor)")`.**
   A bit-twiddling loop like a table-driven CRC (`c = (c & 1) ? P ^ (c >> 1) : (c >> 1)`) vectorizes to
   `lshr <4 x i32>`, and the on-ramp's vector lane-arith set omitted **`shl`/`lshr`/`ashr`**, so it
   fail-closed at *translate*.

**Fix (landed, both in the I2 style):**
1. **`i8x16.mul` lowering in svm-jit** (`Inst::VIntBin` with `VShape::I8x16`): widen each half to
   `i16x8` (`uwiden_low`/`uwiden_high`), multiply (the low byte of an `i16` product equals the low byte
   of the `i8` product, sign-independent), mask each product to its low byte, then pack the two halves
   back with unsigned-saturating narrow (`unarrow` — every lane ≤ 0xFF, so nothing saturates: an exact
   low-byte truncation matching the interp's wrapping mul). Removed from the JIT's `Unsupported`
   pre-check. The interpreters already implemented `i8x16.mul`, so they needed no change.
2. **Vector `shl`/`lshr`/`ashr` in the on-ramp** (`bin`'s `vec128_shape` path): a `const_splat_int`
   helper recognizes a constant-splat shift amount (`<i32 k, …>`, the shape `clang -O2` emits for
   `v >> k`) and emits `Inst::VShift { shape, op: Shl/ShrU/ShrS, .. }` (svm-ir/verify/jit/interp already
   support `VShift` for every shape; the JIT lets Cranelift legalize even `i8x16`'s no-native-per-byte
   shift). A non-constant-splat amount still fail-closes (no corpus need yet).

**Verification.** New `cargo test -p svm` (`diff_i8x16_mul`, interp↔JIT differential) and
`cargo test -p svm-llvm --test translate` (`simd_i8x16_mul_load_store`, `simd_i32x4_const_shifts`) pin
both fixes against the native oracle. End-to-end, `corpus_diff.rs`'s `fnv` (case 1) and `crc32`
(case 2) now translate + run **vectorized** (NOVEC workaround removed) bit-identical across tree-walk,
bytecode, JIT, and native — `fnv`/`crc32` both land at ~1.03× native.

---



### I2 — LLVM on-ramp now ingests auto-vectorized output wider than 128 bits (vector legalization landed) (S3) — fixed on `claude/dreamy-newton-ni7epv`

**Where:** `crates/svm-llvm/src/lib.rs` — vector type recognition (`vec_lane_shape`/`vec128_shape`/
`wide_vec_layout`, `val_type`/`type_size`/`type_align`), the `lower_wide` legalization pass + its
`BlockCtx` helpers, and the block-boundary fan-out in `translate_block`/`branch_args`.

**Was:** translating a `clang -O2`-vectorized program fail-closed with
`Error::Unsupported("type <16 x i32> (Milestone 1+)")` (or `<16 x i64>`, `<4 x i64>`, `<8 x i8>`,
`<2 x i64>`, `<16 x i8>`, etc.). The on-ramp mapped only `<4 x {i32,float}>` (and the 2-lane → packed
`i64` case) to a `v128` and rejected every other shape, because svm-ir's SIMD type is a fixed-128-bit
`v128` (§17/D58) while LLVM's `-O2`/SLP vectorizer emits arbitrary-width "virtual" vectors on the
assumption the backend's `LegalizeTypes` pass will split them. The on-ramp had no such pass.

**Fix (landed, the §17 fixed-128 SelectionDAG-`LegalizeTypes` analog — the chunk width is fixed at
128 bits, never host-detected, to preserve the interp↔JIT/durable-fiber determinism contract):**

1. **128-bit shapes generalized** (fix-sketch step 2): a single `vec_lane_shape`/`vec128_shape`
   recognizer maps any 16-byte LLVM vector to its `VShape`, threaded through every 128-bit lowering
   site, replacing the `i32x4`/`f32x4`-only helpers. svm-ir/verify/jit/interp already supported all
   six `VShape`s, so this was frontend-only. Now `i8x16`/`i16x8`/`i64x2`/`f64x2` all work.
2. **Wide / sub-128 legalization** (fix-sketch step 1): `wide_vec_layout` splits a `<N×T>` into
   `full_chunks` 16-byte `v128`s + `tail_lanes` scalar lanes; `lower_wide` (dispatched at the top of
   `translate_inst`) rewrites each wide op per-chunk + per-tail — load/store, int/float lane arith,
   bitwise, lane min/max, horizontal `vector.reduce.*`, extract/insert, constants, and the broadcast
   (splat) `shufflevector`. A wide value is held as `wide_vals[vid] = [chunks…, tail…]`, mirroring the
   `agg` multi-value pattern.
3. **Cross-block fan-out**: a wide value that crosses a block edge (a vectorized loop's accumulator
   carried across the backedge as a wide phi) expands into `K = chunks + tail` consecutive block
   params, supplied as `K` branch args on every edge (`translate_block`/`branch_args`).

**Follow-ons (now landed, slices AP–AT — the breadth lanes re-enabled vectorization):** vector integer
+ float **conversions** (lane-wise scalarize), **rotate** (`llvm.fshl`/`fshr`), **general cross-chunk +
cross-representation shuffles**, and `<N x i1>` **masks** (vector `icmp`/`fcmp`/`select`/`extractelement`/
`bitcast`-movemask, held lane-wise). The C/C++/Rust breadth lanes now compile **without**
`-fno-*-vectorize` and translate their real `-O2` SIMD output. Still fail-closed (no corpus need yet):
a *general* (non-rotate) funnel shift, a *non-constant* shuffle mask, `llvm.masked.*` (gather/scatter/
masked load-store), wide-vector **function params/returns**, and a mask crossing a block edge.

**Verification:** `cargo test -p svm-llvm --test translate` (115 pass). New tests cover every 128-bit
shape, the wide splitter (`<8 x i32>`/`<4 x i64>` chunks, `<8 x i8>` all-tail), a real loop-carried
wide phi (verified `phi <8 x i32>` in the IR), and two **capstones ingesting genuine `-O2 -mavx2`
auto-vectorized bitcode** (a `<8 x i32>` reduction and an elementwise kernel) byte-identical to the
native scalar oracle on both the interpreter and the JIT.

---

### I1 — A fiber-stack OS allocation failure aborts the process instead of trapping (S2) — fixed on `claude/fiber-stack-lazy-commit`

**Where:** `crates/svm-fiber/src/stack_windows.rs` / `stack_unix.rs` (`Stack::new`), reached via
`Fiber::new` ← `svm_jit::fiber_rt::{make_fiber, fiber_new, seed_frozen_fibers}` and
`svm_jit::instantiator_rt` (the coroutine child). The interpreter has no analogue: its fibers are
host-side `Pending` entries with no native control stack, so only the JIT allocates here.

**Symptom (was):** under real memory pressure, allocating a fiber's control stack failed, an
`assert!` **panicked**, and because `fiber_new` is an `unsafe extern "C"` thunk (called from JITted
guest code, which cannot unwind) the panic became a **non-unwinding abort** — the whole process died
(`STATUS_STACK_BUFFER_OVERRUN` / `SIGABRT`). First observed as a flaky **Windows CI** failure in the
unrelated `jit_threads` concurrent-fiber stress test (PRs #36, #41): a lingering spawned-vCPU
thread's `cont.new` aborted the test binary.

**Root cause / why it bit Windows first.** The design intends a fiber that can't be created to be a
clean, recoverable `Trap::FiberFault` — the **quota pre-check** (`SharedFiberTable::has_room`)
already delivers that for a fiber *bomb*. But a *genuine OS-allocation failure below the quota* had no
such path: `Stack::new` just `assert!`ed. Compounded by Windows committing eagerly:
`stack_windows.rs` reserved **and committed** the full per-fiber stack (`FIBER_STACK = 1 MiB`,
`MEM_RESERVE | MEM_COMMIT`), so N live fibers cost N MiB of *committed* VA, while the unix `mmap` path
commits lazily on touch. The quota (`MAX_FIBERS = 1 << 16`) × 1 MiB ⇒ a 64 GiB committed ceiling that
does not bound real Windows memory, so `VirtualAlloc` failed long before the quota tripped.

**Fix (landed):**
1. **`Stack::new` and `Fiber::new` are now fallible** (`-> Option<…>`, returning `None` on
   `MAP_FAILED` / null `VirtualAlloc` / guard-`mprotect`/`VirtualProtect` failure, with the partial
   reservation cleaned up). The JIT callers turn `None` into the intended recoverable trap:
   `fiber_new` writes the trap cell + returns `-1` (the existing `FiberFault` path); `make_fiber` and
   `seed_frozen_fibers` propagate it (a thaw re-seed failure skips the root re-entry rather than
   re-entering with missing fibers); the instantiator coroutine returns `CapFault`. No path can abort
   the host on a fiber-stack allocation failure anymore.
2. **Per-fiber control stack reduced 1 MiB → 256 KiB** (`FIBER_STACK` / `CORO_STACK = 1 << 18`),
   cutting committed Windows memory 4× per live fiber and pushing the practical fiber ceiling out
   correspondingly. Still ample for deep guest call chains.

**Why not true kernel-growth lazy commit on Windows (the original fix-sketch point 2):** rejected.
The `svm-jit` `gc.roots` walker scans a *running* fiber's whole usable stack via
`Fiber::full_extent()` → `[usable_low, top)` (a sound conservative superset of its live frames).
Under demand-commit that scan would touch uncommitted pages and fault. Making it safe would need a
committed/high-water bound threaded through the GC scan, and Windows can't be run-tested in this
environment — so the size reduction + fallible alloc (both fully testable, and the latter is the
actual abort cure) were chosen over an untestable, GC-entangled commit-on-fault scheme.

**Verification:** `svm-fiber` + `svm-jit` unit tests, `jit_threads`, and the durable-fiber
freeze/thaw suites pass on unix; `cargo check --target x86_64-pc-windows-gnu -p svm-fiber` compiles
the rewritten Windows path. The recurring Windows `jit_threads` flake's abort mechanism is removed.
