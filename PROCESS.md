# Process substrate & OS personalities — processes over domains

> **Status: PROPOSED — design draft v2 (red-team findings folded in), nothing built.** Working tracker for a **process
> abstraction** over the machinery that already exists (§14 nesting, §12 concurrency
> primitives, §13 shared regions, §7 host-extensible capabilities, `DURABILITY.md`
> snapshot/clone). v2 supersedes the earlier POSIX-flavored draft on this branch: the
> design is now split into an **OS-neutral substrate** (few, orthogonal, capability-shaped
> primitives) and **personalities** (POSIX first — it exists to run Bash/BusyBox — with
> other OS-like layers explicitly intended later). Like `WASM.md`/`THREADS.md`, fold
> settled parts into `DESIGN.md` and drop this file when the gaps close.
>
> Proposed decision: **D63** (D62 is currently the last). See bottom of file.

The one-sentence design: **the substrate offers process primitives, not process policy —
exactly as D56 offers concurrency primitives, not a scheduler.** A process is a domain;
every "dial" (memory visibility, interface style, IPC, accounting) is a choice of *which
capability you pass* or *which verb you call*, never a mode flag; POSIX semantics (fds,
fork-returns-twice, signals, errno) live in a guest-library personality that the substrate
never learns about.

---

## 0. Orientation — what already exists (verified against the tree)

- **Domains-as-processes in embryo.** §14 `Instantiator` (iface 6), 8 ops on all three
  engines: `instantiate` (0) / `join` (1) / `spawn_coroutine`+`resume` (2/3) /
  `spawn_demand_coroutine` (4) / the `*_module` variants (5/6/7). A child is its own
  domain: power-of-two carve, own powerbox + dispatch table, fuel quota sub-allocated
  from the parent's, recursion to any depth. Note ops 0/2/4 are *already* the
  concurrent / coroutine / demand-paged interface dial — bundled into op variants.
- **exec exists**: `Module` (iface 8) — host-verified code a guest may instantiate and
  nothing else. **wait exists**: `join` parks only the calling fiber.
- **Guest-serviced capability, in one instance**: the coroutine `Yielder` (iface 7) —
  child parks, parent services, child resumes. The bytecode engine's `VcpuEvent`
  orchestration seam has the same shape host-side.
- **fork's machinery exists, unexposed**: durable freeze → serialize → restore → thaw
  round-trips on both backends including nested subtrees (snapshot v4,
  `durable_nesting.rs`); `DURABILITY.md` §10: clone "falls out at a quiescent point."
  CoW clone is Phase-4, unlanded.
- **Memory-authority objects exist**: `AddressSpace` (iface 5, attenuable to sub-ranges,
  can mint a child's), `SharedRegion` (§13), demand paging (parent-as-pager).
- **Async exists**: `IoRing` (iface 9) — batched deferred `cap.call`s with offload.
- **The capability seam is open by construction** (§7/D46): interfaces are data + host
  code, bound by name at instantiation; `cap.self` reflection is an always-available,
  **runtime-resolved** intrinsic; acquisition is a granted `Resolver`.

Named gaps (each already a "follow-up" in code comments or docs):

1. **Children are born destitute** — powerbox = `Instantiator` + `AddressSpace` only;
   *"pass-through of the parent's other handles is a follow-up"* (`svm-interp`,
   instantiate op 0). No I/O of any kind.
2. **Guest-serviceable capabilities exist only as the special-cased `Yielder`** — there
   is no general way for one domain to implement a capability interface for another.
3. **No lifecycle ops** — no poll, no parent-initiated kill (§15 names the design), no
   detach.
4. **JIT children run synchronously at `instantiate`** (`instantiator_rt.rs` header;
   park-only-the-calling-fiber is its named follow-up). Spawn cost was a full Cranelift
   recompile per spawn — **now cached** (S1, done): `compile_child` bakes only the *size*
   mask and the window **base is a runtime arg**, so one compiled child is
   position-independent and a per-carve cache reuses it across offsets (my earlier note
   that the JIT "bakes `sub_base` into code" was wrong — corrected).
5. **No clone entry point**; no dynamic module loading (bytes → verified `Module`); `fs`
   cap has no directory ops.

---

## 1. Requirements

1. **Process-grade isolation.** Sibling domains mutually invisible (mask-enforced —
   already true); optionally, domains whose memory **no ancestor below the platform
   host** can read.
2. **Self-similarity under nesting.** A domain can *be the OS* for its children —
   pass through, attenuate, or fully virtualize their world — and behavior is identical
   at every depth. Children cannot distinguish a parent-implemented capability from a
   platform one (§14 "host is a role, not a level", taken as a hard requirement).
3. **OS-agnostic.** The substrate encodes no POSIX (no fds, no fork-returns-twice, no
   signal numbers, no errno beyond the D42 convention). Different OS-like layers with
   different semantics must be layerable per-project — and per-*subtree* (a guest may run
   its children under a different personality than its own).
4. **Orthogonal dials, not modes.** Memory-visibility-to-parent, parent↔child interface
   style (sync / coroutine / concurrent), peer IPC style, and resource accounting vary
   independently, without a combinatorial op set.
5. **Attestable protection.** A domain can learn, from an unforgeable source, whether any
   host other than the platform host holds read authority over its memory — so it can
   refuse to handle secrets under a hostile nested host.
6. **Resources attenuate along the grant graph, but accounting is delegable.** A child
   can never exceed what its ancestors granted; yet a child must be able to have a new
   domain created **charged to its parent's budget** (with the parent's consent), not
   carved from its own.

---

## 2. The shape — substrate vs personalities  [PROPOSED]

Two layers above the (unchanged) core VM:

- **Substrate** — OS-neutral primitives: `Domain`, `Endpoint`, window sources, `Budget`,
  plus the existing `Module`, `SharedRegion`, futex, `IoRing`, durable clone. Exit is
  `(i64, trap kind)`; no channels, no signals, no fds. All of it ordinary §7 capability
  interfaces — **authority-TCB, not escape-TCB** (§2a): a substrate handler bug misuses
  its own authority; it cannot escape.
- **Personalities** — guest libraries + capability recipes. POSIX is the first client
  (fd table, `fork`/`exec`/`wait`, signals-as-doorbell, pipes — enough for Bash/BusyBox).
  Sketched seconds, to keep the substrate honest: an actor personality
  (spawn/link/monitor/mailboxes ≈ `start`+`poll`/`kill`+async endpoints) and a
  deterministic-dataflow personality for durable pipelines. Personalities are recipes:
  `posix_spawn` = "nested window + my budget + `start` + endpoints named
  stdin/stdout/stderr". The substrate never learns a recipe.

The discipline (prime directive): **design for two personalities, build for one.**
Substrate ops land only as the POSIX personality needs them — but named and placed so the
second personality never forces a re-layering.

### The dials, mapped

| dial | how it's chosen | mechanism |
|---|---|---|
| memory visible to parent? | **which window source you pass** to `create` | `AddressSpace` sub-range (nested carve — visible superset, §14) vs. platform window minter (detached — opaque) |
| parent↔child interface | **which verb you drive with** | `call` (sync) / `resume` (coroutine) / `start`+`join` (concurrent) |
| peer / cousin IPC | same mechanism as everything else | an `Endpoint` (sync `cap.call` or async via `IoRing`) granted to both parties |
| who is charged | **which `Budget` you pass** to `create` | budget lineage, not requester identity |
| who implements my world | invisible to the child | every granted cap is host-, ancestor-, or peer-served — indistinguishable (§14) |

No flags. Adding a dial setting never adds an op.

---

## 3. `Domain` — the process object  [PROPOSED]

A domain is created **inert** and assembled, then driven. All ops D42-shaped
(negative-errno `i64`, borrow-only `(ptr,len)` args, fail-closed):

```
create(module, window, budget)      -> domain | -errno     (born suspended, empty powerbox
                                                             + Domain/AddressSpace over itself)
grant(domain, name_ptr, name_len, handle) -> 0 | -errno    (pre-start only; re-grants MY handle
                                                             into its table under `name`;
                                                             child finds it via cap.self.resolve)
start(domain, entry)                -> 0 | -errno           (schedule concurrently; §12 primitives)
call(domain, entry, args…)          -> result | -errno      (synchronous: park caller till return)
resume(domain, val)                 -> status/val           (coroutine step; existing op-3 semantics)
join(domain)                        -> result               (park calling fiber; existing op 1)
poll(domain)                        -> 0 running | 1 returned | 2 trapped   (never parks)
kill(domain)                        -> 0 | -errno           (idempotent; §5 detect-and-kill on the
                                                             child's grant subtree; joiner sees Killed)
detach(domain)                      -> 0 | -errno           (drop claim; auto-reap on completion)
clone(domain)                       -> new domain | -errno  (domain must be parked at a suspension
                                                             point and durable-instrumented; §7 below)
```

Notes:

- **create-suspended + `grant` + seed-then-`start` replaces v1's spawn-time grant list.**
  Simpler (no record format to parse/fuzz — each grant is one call), and it dissolves the
  proc-block seeding-order problem: parent creates, seeds argv/env into the (nested)
  window or a granted region, grants caps, starts.
- The interface-style dial is **per use, not per domain**: the same domain can be
  `call`ed for setup, then `start`ed. Ops 0/2/4 of today's `Instantiator` become recipes
  over one object; **the existing ops stay** (built, tested, CI-gated) as the nested fast
  path until a second consumer motivates unifying — no churn for aesthetics.
- Scheduling honors **D56/D22**: `start` maps onto the existing vCPU primitives (1:1 OS
  thread on the JIT, the deterministic oracle on the interpreter); the substrate adds no
  scheduler and no policy. Fairness across many children = the existing fuel/epoch
  machinery, measured not assumed.
- Exit is `(i64, trap kind)`, un-flattened. POSIX's `$?` 8-bit convention is personality.
- **Teardown is specified, not implied**: a domain's completion (return, trap, kill)
  drops its handle table; drop notifications reach reference-counted resources (an
  endpoint with no remaining servicer fails its parked callers with `-ECONNRESET`-shaped
  errno rather than hanging them). This is the close-on-exit rule that makes pipelines
  terminate.

---

## 4. `Endpoint` — the one communication primitive  [PROPOSED]

A guest-serviceable capability interface: the general form of the `Yielder`, and the
keystone of self-similarity (requirement 2). Whoever holds the **serve end** implements
the interface; whoever holds a **client handle** just sees an ordinary capability.

```
mint(sig)                -> (serve_end, client_template)   (interface signature per §7 —
                                                            declared in the type section,
                                                            verifier-checked at every cap.call)
serve(serve_end, buf)    -> (caller, op, args)             (park until a call arrives)
reply(serve_end, caller, result)  -> 0 | -errno            (resume that caller)
```

- **Client side is invisible**: a client's `cap.call` on an endpoint-backed handle parks
  its fiber until `reply` — indistinguishable from a host-implemented cap. This *is*
  §14's "the parent's own handler / pay-for-what-you-virtualize," made a mintable object
  rather than a special case. A personality is now implementable by **any** guest for its
  children — parent-as-POSIX-kernel, parent-as-pager, sibling-as-service.
- **Async is not a second mechanism**: a client submits the same call through the
  existing `IoRing`; sync vs async is per-call.
- **v1 scope is deliberately microkernel-lesson-sized**: synchronous rendezvous +
  kill-safe cancellation only. Kill of a parked *client* → the servicer's eventual
  `reply` is inert (generation-checked, like every stale handle). Kill of a *servicer* →
  parked clients fail with an errno (see §3 teardown). No reply forwarding, no call
  timeouts, no priority — deferred until a personality demonstrably needs them.
- **Wire discipline — scalars only; the data plane is explicit.** D42's borrow-only
  `(ptr,len)` args assume the handler is the *host* (validated trampoline into any guest
  window); a **guest** servicer cannot dereference a detached caller's window at all.
  Nested parent↔child gets the data plane free (§14 superset); detached/sibling endpoints
  carry scalars and move bulk data through a `SharedRegion` established at grant time.
  Runtime copy-in/copy-out is **rejected**: it would reintroduce exactly the lift/lower
  marshalling tax §1a defines this VM against. Consequence, stated: sibling IPC is the
  identical *interface*, not the identical *data plane*.
- **Budget flow: the servicer pays** for servicing fuel (a caller pays only its own call
  overhead). Named consequence: a client can spend its servicer's fuel by calling in a
  loop — a *liveness* exposure inside an existing trust relationship, bounded by the
  caller's own fuel and by personality-level rate limits; never an isolation break.
  (Fuel donation à la seL4 MCS is deliberately not attempted.)
- **Rendezvous order is fixed FIFO.** One pinned, non-configurable policy — required for
  the deterministic oracle to mean anything. ("No policy in the substrate" is precisely:
  no *configurable* policy.)
- **Deadlock is unowned in v1.** Call cycles across the grant graph (A→B→C→A) wedge all
  parties; there are no timeouts (the L4 lesson: timeouts are their own tar pit). The
  escape hatch is `kill` plus a supervisor reading §15 meters (fuel flatline = wedge).
  Timeouts, if ever, are personality policy.
- **Implementation direction — library first (the D56 lesson).** D56 removed the in-VM
  M:N executor as the project's highest-risk unsafe; cross-domain fiber rendezvous is the
  same risk class. So the first build is an endpoint **library** over `SharedRegion` +
  futex (both exist), with runtime support only for handle transfer — falling back to
  runtime rendezvous only where the library measurably can't reach (cross-domain futex
  keys — the O2 spike, promoted to first in the tracker).

### S0 spike results — the library-endpoint path is viable  [DONE]

The go/no-go for "endpoints as a library over shared memory + futex" was: does futex
`wait`/`notify` rendezvous *across domains* on shared backing? Findings, grounded in the
code and pinned by `crates/svm/tests/futex_cross_domain.rs`:

- **Nested (parent ↔ child): confirmed, by construction.** The §12 futex key is the
  *confined absolute* backing address (`Mem::confine_checked` → `window.base() + offset`),
  the scheduler queues waiters under that exact key (`Scheduler::notify` /
  `wait_waiters`), and a nested child shares the parent's backing `Arc`
  (`Mem::nested_view`) and executor. So a parent at `carve_off + a` and its child at `a`
  name the **same** key and rendezvous. Pinned both directions (deterministic
  parent-parks/child-notifies; racy inverse accepting both futex-legal outcomes). This is
  the load-bearing case — a parent implementing a personality for its children — and it
  works with **zero new runtime machinery**.
- **Sibling (region-aliased): value-coherent, but wakeup needs canonical keys.** Two
  siblings aliasing one §13 `SharedRegion` share *bytes* fine (backed-page reads/writes
  route through the shared `back` region — already proven cross-domain by
  `region_grant.rs`), so a ring buffer's **data plane** works sibling↔sibling today. But
  each sibling maps the region at its *own* window offset, so the same region byte has a
  *different* absolute address in each — hence a *different* futex key, and a `notify` on
  one alias won't wake a waiter parked on the other. The `wait`/`notify` **wakeup** is the
  only gap. Fix (scoped to S9, not the spike): when the waited/notified page is `Backed`,
  key on `(backing identity, region offset)` instead of the window-absolute address — the
  exact distinction Linux draws between `FUTEX_PRIVATE` (VA-keyed) and shared futexes
  (page-keyed). A small, localized change in `prepare_wait`/`confine_for_notify` + the
  scheduler key type; **no confinement-hinge contact** (a futex key is a rendezvous
  coordinate, not a bounds check — a wrong key misses a wakeup, it cannot escape).

**Conclusion:** library-first endpoints are viable now for the nested (personality) case;
the sibling case is data-plane-ready and needs only the canonical-key wakeup, folded into
S9. No runtime rendezvous engine is required for either — the D56-shaped outcome the
design hoped for.

### F6 — endpoint RTT budget (model; measured numbers land with S9)

An endpoint is *guest-served*, so a call costs a rendezvous, not a trampoline. The
ordering — and it must be stated so it never silently contradicts §1a's host-call speed
pitch:

| path | crossing cost (model) | notes |
|---|---|---|
| host `cap.call` | 1 trampoline, **0 fiber switches** | the §1a fast path; a host handler runs on the caller's stack |
| guest-served endpoint RTT (nested) | **≥ 2 fiber switches + 1 futex rendezvous** | caller parks → servicer wakes, runs, replies → caller wakes |
| guest-served endpoint RTT (sibling) | as nested, once canonical keys land | + the region-map setup, one-time |
| Linux syscall (reference) | ~0.1–1 µs | the thing personalities are compared against |

Order of magnitude, not a promise: the full nested rendezvous cycle in the S0 test
(parent parks on `wait` → child `notify` → wake → `join`) completes in **well under a
millisecond** (both spike tests report `0.00s` run time) — i.e. endpoint syscalls sit in
the sub-microsecond-to-microsecond regime, fine for **shell-frequency** control calls
(`fork`/`wait`/`open`), and decisively the **wrong** structure for per-byte pipe
throughput (that rides `SharedRegion` bulk data, not per-byte endpoint calls). The
standing mitigation is §14's **pay-for-what-you-virtualize**: a pass-through capability is
a host `cap.call` (top row) — you only pay the endpoint RTT for capabilities a parent
*chooses* to interpose. Real measured figures (endpoint RTT vs `cap.call` vs syscall, all
three engines) are produced with the S9 endpoint implementation and replace this model.

### S1 spike results — JIT children are position-independent  [compile cache DONE]

The JIT re-compiles a §14 child as a top-level guest over its own window; the open worry
(F6/gap-4) was that this bakes the carve into the code, so every spawn recompiles. It
does **not**: `compile_child` bakes only the *size* mask (`& (2^size_log2 − 1)`) and the
window **base is a runtime argument** to `run_guarded`. So one compiled child runs at
**any** carve offset, and a per-carve compile cache reuses it across offsets — not merely
across repeated same-slot spawns. Built and pinned by `crates/svm/tests/jit_instantiate_cache.rs`:
the same `(module, entry, size)` spawned at two different offsets JIT-compiles **once**
(`svm_jit::child_compiles()` advances by 1), and each spawn still runs correctly confined
to its carve. Scope: **non-durable** children (the shell-applet common case) are cached;
durable/nesting children keep the per-call `compile_child_and_run` path (their baked
per-child nursery makes the code un-shareable) — a small, deliberate exclusion, not a gap.
This also *helps* async children: cached code is read-only executable, so the same blob
can back N concurrent OS-thread children.

### S1 remaining — async children: the architecture, corrected by integration  [design]

The JIT already has a **1:1 OS-thread executor** (`os_thread_rt.rs`, D56/§12): each
`thread.spawn` is a real OS thread over the shared window with hardware atomics, and the §5
kill-path reaches parked siblings (`KILL_RECHECK`). So "async children" reuses that, not new
concurrency machinery: `instantiate` spawns the (cached) child on its own OS thread and
returns; `join` parks the calling fiber on the child's completion cell (as `thread.join`
does). This is what makes a pipeline work — child A blocks on a pipe/futex while child B
(another thread) unblocks it — where synchronous-at-`instantiate` deadlocks (the parent is
stuck inside `instantiate(A)` and never reaches `instantiate(B)`). Sequential spawn/wait
(stage 1) works on the synchronous path today, so this precedes stage-2 pipelines, not
stage-1.

**The load-bearing finding (why the obvious design is a confinement bug).** The tempting
shortcut — since S1 proved child code is position-independent — is to run the child
*in-place in the parent's window* (base = `parent_base + carve_off`), so parent and child
share bytes live and the S0 futex rendezvous "just works" on the JIT too. **It does not, and
it is unsafe.** JIT confinement is D38 *check + clamp* (`& (reserved−1)`), and the clamp
confines the **offset** to `[0, child_size)` — but a width-`w` access at the top of the carve
reaches up to `base + child_size + (w−1)`, which the separate-window model catches with a
**trailing guard page**. Densely-packed carves have *no* guard page between them, so an
in-place child could write up to `w−1` bytes past its carve into the **neighbouring carve or
the parent** — a real break of carve isolation (not a host escape — still inside the parent
window — but it destroys sibling mutual-invisibility and the detached/confidential model).
**This is exactly why the JIT runs each child in its own separately-guarded window**, and it
stands: implicit carve-superset sharing is a *nested-synchronous convenience* (seed argv,
read results at `join`), **never** the concurrency channel.

**So the concurrency & communication plane is explicit `SharedRegion` + canonical-key
futex — the same mechanism as siblings (S0), for the same reason.** A JIT child runs in its
own guarded window, so parent↔child *live* rendezvous can't ride implicit carve addresses
(different allocations) any more than two siblings can; both go through a `SharedRegion`
mapped into each, with the futex keyed on the region's canonical `(backing, region_off)` —
not the window-absolute address. The interpreter's `PageProt::Backed { region, region_off }`
already carries that identity; the work is keying the scheduler's futex map (and the JIT's)
on it without colliding with anonymous absolute-address keys — a **cross-backend futex
key-space change** (S9's canonical-key item), which this finding **promotes onto the async
critical path**: it gates JIT concurrent parent↔child, not just sibling pipes.

**Revised async-children plan (own-window + explicit channels):**
1. **Canonical-key futex** (was S9): key `Backed` pages on `(backing, region_off)` on both
   backends. Unblocks the comm plane for siblings *and* JIT concurrency. Testable by
   extending `futex_cross_domain.rs` to two siblings sharing a region.
2. **OS-thread children on the JIT**: `instantiate` spawns the cached child in its **own
   guarded window** (safe — keeps the trailing guard) on an `os_thread_rt` thread; `join`
   parks on the completion cell; the child polls the parent's epoch cell (kill-path already
   wired). Copy-in seeds argv at spawn; the child's live channel is a granted `SharedRegion`,
   not the copied carve.
3. **Interp parity check**: the interpreter's parent↔child futex already works via shared
   backing (S0) — but that is the *nested-synchronous* superset, not a portable channel; the
   portable (backend-agnostic) pipe rides the `SharedRegion` path from step 1 on both engines.

The one thing the corrected model does **not** need is any change to the D38 confinement
lowering — the security hinge stays untouched; children keep their own guarded windows.

---

## 5. Window sources & `Budget` — visibility and accounting as arguments  [PROPOSED]

**Visibility = window provenance.** `create`'s `window` argument is a handle to a window
object, and *who holds authority over its backing* is the whole visibility story:

- **Nested**: a sub-range minted from my `AddressSpace` (machinery exists). I see the
  child (§14 superset) — the hypervisor relationship: free argv seeding, parent-as-pager,
  subtree freeze works **today**. Geometric (power-of-two) attenuation, mask-enforced.
- **Detached**: a window minted by a **platform window-minter capability** — an ordinary
  granted authority (the D46 `Resolver`-shaped acquisition pattern: you can mint detached
  windows only if someone granted you that). No ancestor below the minter holds read
  authority. Numeric quota, host-enforced at mint.
- Demand-paged sits between, and honestly: **pager authority is read authority** — a
  domain whose pages are supplied by its parent is visible to it. `attest` (§6) reports
  this.

Trade-offs stated once, honestly: detached subtrees make freeze/clone a **multi-window
snapshot** — new `DURABILITY.md` work (nested subtrees freeze today); nested carves
subdivide parent VA (real in the browser's wasm32 window). Projects choose per child; a
shell would plausibly run coreutils detached and its own helper coroutines nested.
(Carve *geometry* no longer costs the JIT recompiles — S1's cache is position-independent,
§4 S1 results — so JIT compile-cache pressure is no longer a reason to prefer detached.)

**Accounting = budget lineage.** `Budget` is §15's principle — "every meterable resource
is already a capability with a quota" — promoted to a passable, splittable object:

```
split(budget, fuel, mem, spawn)  -> sub_budget | -errno    (attenuation: sub ≤ remaining)
read(budget)                     -> remaining/spent          (§15 monitoring readout)
```

A domain's consumption charges the budget it was created with. Requirement 6 then costs
**zero mechanism**: budgets always attenuate along the grant graph (a child can never
exceed its ancestors — D19's invariant, kept), and "child asks parent to create a process
charged to the parent" is either (a) the parent pre-split a budget to the child, or (b)
the child calls a **spawn endpoint the parent serves**, and the parent creates the domain
against its own budget — its consent is that it services the call. Genode's quota-transfer
model, reached via two primitives we need anyway.

### Faults — the security trap is terminal; the memory fault is a capability event

Two different things surface as "SIGSEGV" and the design splits them:

- **Confinement violations (out-of-window)** — the D38/§4/§5 path (guard page or
  explicit check → cold trap → detect-and-kill). **Terminal, always**: making this
  resumable would put feature pressure on the most security-sensitive lowering in the
  tree, for the benefit of a guest probing its confinement. Post-mortem observation is
  covered (trap kind + backtrace to the parent — `jit_trap_backtrace`).
- **In-window memory-management faults (unmapped / protected page)** — already a
  *resumable event* on both backends: a demand-paged child's fault suspends the fiber
  with the fault address (`SUSP_FAULT`, interp + `instantiator_rt.rs`), the pager
  supplies the page, and **resume retries the access**. Retry-on-resume is the trick:
  precise fault handling with no per-access deoptimization metadata — the tax that
  makes in-band SIGSEGV handlers expensive in a JIT, and which Cranelift won't sell
  cheaply.

The substrate generalization: **pager authority over a region is a grantable
capability** — to the parent (built today), to a sibling service, or to a designated
fiber of the *same domain* (self-paging: the faulting fiber suspends, the handler fiber
maps/protects via its own `AddressSpace`, the faulted fiber resumes and retries; the
handler fiber must not fault on the region it pages — the discipline POSIX handlers
already need, here enforced by structure). That covers the legitimate SIGSEGV-handler
uses — lazy mapping, GC write barriers over `PageProt`, guard-page stack growth — in
capability style. Not offered: a handler that runs *on the faulting context* and mutates
it (`ucontext` fiddling, `siglongjmp` out of a handler) — the deopt-metadata case stays
out. Core wasm has no pager events at all; this is a place the design is strictly ahead.

---

## 6. `cap.self.attest` — the trust anchor  [PROPOSED — amends one §14 sentence]

Requirement 5 hits the attestation regress: if everything is interposable (§4), a hostile
nested host virtualizes the attestation capability too and lies. No trust bootstraps from
inside pure virtualization. The minimal fix is **one reserved, non-interposable namespace,
always platform-terminated** — and it already exists: `cap.self` is a D46 *intrinsic*,
runtime-resolved, never a handle-table entry, so no parent can interpose it by
construction. Add one read-only op:

```
cap.self.attest() -> { isolation_tier,                      (§2: 0 / 1 / 3)
                       window_provenance,                    (which authorities hold
                                                              map/read/pager rights over
                                                              my backing: platform-only,
                                                              or ancestor-held)
                       freeze_authority }                    (who may snapshot me —
                                                              a snapshot IS a read)
```

- **Read-only report, no negotiation.** A domain that dislikes the answer exits before
  touching secrets. Fits D46's `cap.self` contract exactly: reflection confers nothing,
  adds no grant-graph edge, and extends the stated "no deniable grants" transparency
  principle from *authority* to *exposure*.
- **Durability and confidentiality are in direct tension — a per-domain choice.**
  Transparent freeze means a domain *cannot observe* being snapshotted, and the artifact
  is a complete read of its window. So "no ancestor can read my memory" is false for any
  domain an ancestor can freeze — which is every nested durable child today. Hence
  `freeze_authority` in the report, and the rule: a domain may be **confidential**
  (freezable by nobody below the platform) or **ancestor-durable**, not both. Pick per
  domain.
- **Attest covers computation, not provisioning.** Every capability a domain holds came
  through its (possibly hostile) creator, so "fetch my secret over my secure channel" is
  MITM-able regardless of a clean report — the classic TEE lesson. v1 deliberately claims
  only: *confidentiality of computation over data the domain was created with or
  derives*. Attested secret provisioning (a platform-terminated channel) would grow the
  non-interposable surface and is explicitly out of scope until a real consumer forces
  the argument.
- **Honest limits, recorded**: attest cannot protect you from your creator having chosen
  your initial state or code (nothing can); and per §2, tiers 0/1 are never a Spectre
  boundary — a domain requiring protection from a *distrusted* host must see `tier 3` in
  the report or refuse, which is exactly §14's "a tier-3 child requires the host to grant
  a real process." Timing side channels (scheduling, fuel drip, a granted — hence fakeable
  — clock) remain the host's.
- **The friction, named**: §14 currently says *"There is no 'am I nested?' query by
  default."* Attest is deliberately such a query (provenance reveals nesting). Proposed
  amendment: the default stands — no ambient nesting query, and a virtualized capability
  remains indistinguishable — but a domain may **opt in** to the one platform-vouched
  provenance report, because self-protection from hostile nested hosts is impossible
  without a trust anchor. This is a change to settled §14 text and is called out for
  exactly that reason (change settled things only with a reason — this is the reason).
  The carve-out must stay **tiny**: every op added to the non-interposable namespace
  erodes the self-similarity the rest of the design exists to provide. The growth
  criterion, pinned now so future pressure has a rule to argue against: the namespace
  admits only **facts the platform mechanically enforces** — never services, never
  channels. `attest` and the existing reflection are the whole list.

---

## 7. clone & fork — parked domains only  [PROPOSED; PARKED until §10 stage 3]

The endpoint model dissolves v1's fork inversion (shells fork *themselves*, but a parent
clones a *child*): **every endpoint call is a park at a durable suspension point**
(`cap.call` is precisely the `svm-durable` transform's suspension unit). So:

- `clone(domain)` requires the target to be **parked and durable-instrumented**; it
  captures the domain (existing freeze machinery), restores into a fresh window (same
  source kind), and re-grants the same pass-through handles (policy hook for anything
  fancier). Both copies resume from the same park.
- **POSIX `fork` is personality sugar**: the libc's `fork()` is a call on a spawn/fork
  endpoint the domain's personality-provider serves; the servicer clones the parked
  caller and replies differently to each copy. Fork-returns-twice is a *reply value*,
  not a substrate concept. Mechanically this needs one endpoint feature beyond §4's v1:
  `clone` of a caller parked on a pending call **duplicates the pending call** and hands
  the servicer a second reply token, so it replies once per copy. The supervisor pattern
  (shell runs under a tiny init domain) falls out of "someone serves your fork endpoint"
  — no special architecture.
- **Cost/coverage, honestly**: v1 clone is a full window copy (CoW rides `DURABILITY.md`
  Phase 4, not blocked on). The caller must be durable-instrumented, and the transform
  today treats `call_indirect` to may-suspend targets as out of scope (R8) — Bash
  dispatches builtins through function-pointer tables, so **R8 closure is on fork's
  critical path**, tracked as its own slice. **Replay-fork** (deterministically re-execute
  from `create` with recorded inputs up to the park) is kept only as a *niche* option for
  short-lived deterministic domains — for a long-lived interactive shell it is O(session)
  with full input recording, so it is **not** a credible R8 escape hatch. Personalities
  that never fork (most) pay nothing.

---

## 8. Alignment with DESIGN.md — the check

Aligned (by section):

- **§7/D46**: every substrate object is an ordinary open-set capability interface —
  signature in the type section, verifier-checked `cap.call`, named binding, fail-closed.
  `Endpoint` adds guest implementations of the same interface shape; the client-side
  contract is unchanged.
- **§2a**: the substrate is **authority-TCB** — handlers can misuse their own authority,
  never escape. No verifier, masking, or confinement change anywhere in this file.
- **§14/D19**: nested mode is untouched; attenuation invariants (child ≤ parent, tier ≤
  parent) hold in both window modes; "host is a role" is promoted from principle to
  mintable mechanism (`Endpoint`). One sentence amended (the nesting query — §6, flagged).
- **§15**: `Budget` is its "quota is both the limit and the readout" made passable;
  monitoring stays reading-the-meters-you-granted.
- **D56/D22**: no scheduler, no policy — D63 is D56's move applied to processes.
  Personalities are "the guest runtime builds any model over the primitives," verbatim.
- **D42**: errno/borrow-only conventions throughout. **D13/Capsicum**: window/budget
  attenuation-by-shape matches `Directory`/`Connector`.

Frictions, named rather than hidden:

| # | friction | disposition |
|---|---|---|
| F1 | `attest` vs §14's "no am-I-nested query by default" | deliberate, narrow amendment (§6); needs sign-off with D63 |
| F2 | non-interposable namespace erodes self-similarity if it grows | hard rule: reflection + attest are the entire list |
| F3 | D19 bundles window+caps+quota in `Instantiator`; substrate factors them | generalization, not contradiction — invariants kept; ops 0–7 remain as the nested recipe |
| F4 | "protection from hostile hosts" overpromises at tier 0/1 (Spectre, §2) | attest reports tier; distrust still means tier 3 — no new claim |
| F5 | prime directive vs. abstraction-before-demand | build only what the POSIX personality needs; the factoring is naming/placement, not speculative code |
| F6 | guest-*served* calls are ≥2 fiber switches — much slower than host cap calls; must not silently contradict §1a's host-call speed pitch | **RTT budget model delivered (§4 S0 results)**: nested rendezvous is sub-ms, endpoint syscalls sit at shell-frequency scale, per-byte pipe throughput rides `SharedRegion` bulk data not endpoint calls; pass-through (don't virtualize what you don't need) keeps un-interposed caps at host-`cap.call` speed. Measured figures land with S9 |
| F7 | endpoints re-enter the risk class D56 removed (cross-domain fiber rendezvous, the project's highest-risk unsafe) | library-first implementation over SharedRegion+futex (§4); runtime rendezvous only where the library measurably can't reach |

---

## 9. POSIX personality (first client)  [sketch]

Guest libc + a capability recipe set; the substrate never learns any of it:

- **fd table**: int fd → handle map in guest memory; `dup2`/redirection are table edits.
- **pipes**: a channel capability *of the personality* — either a host-served
  stream-pair cap (svm-run library, park-on-empty/full via endpoint semantics) or a
  guest ring in a `SharedRegion` + futex (durable-friendly: buffer bytes are window
  bytes, so frozen pipelines round-trip — the host-served variant's buffers are host
  state that D-scope snapshots drop). Resolve via O2 which lands first.
- **spawn/exec**: `create`(Module, window, budget) + `grant`(stdin/stdout/stderr/fs/…) +
  seed argv/env + `start`. PATH is a name→`Module` map the shell holds; dynamic loading
  (bytes → verified `Module`) needs a **`ModuleLoader` capability** (host verify — the
  verifier is a cheap linear pass; natural, fail-closed) — required for `cc x.c && ./a.out`
  workflows, optional for a coreutils shell.
- **wait/`$?`/signals**: `join`/`poll` + exit flattening; `kill` for SIGKILL; a reserved
  doorbell word (guest memory + futex) for cooperative `trap` checks between commands —
  matching how shells actually poll traps. The full delivery ladder is below.
- **fork**: §7. BusyBox `ash`/`hush` (fork-less NOMMU designs) before Bash.

### POSIX process-model coverage — an honest census

What the substrate can recreate, graded. "Faithful" = a program using it cannot tell.

**Faithful:**

| POSIX | realization |
|---|---|
| `posix_spawn` / the `fork`+`exec` pattern | `create`/`grant`/seed/`start` — covers the large majority of real-world fork sites |
| `waitpid` (blocking + `WNOHANG`), exit codes | `join`/`poll` + `$?` flattening |
| `kill(pid, SIGKILL)` | `kill` on a held handle |
| argv / environ / cwd | proc ABI |
| pipes, `dup2`, redirection, here-docs | fd table + pipe cap |
| fd inheritance | explicit grants — i.e. `O_CLOEXEC`-by-default, the modern best practice |
| rlimits (`CPU`/`AS`/`NPROC`) | fuel / memory / spawn budgets |
| zombies & reaping | `join`/`detach` with auto-reap — leak-free by construction |
| `mmap(MAP_SHARED)`, SysV/POSIX shm + semaphores | `SharedRegion` + futex (personality lib) |
| orphan reparenting to init | supervisor personality |
| `ptrace` / `strace` | nested-window visibility (`svm-dap` exists) + endpoint interposition of the cap set — *stronger* than POSIX |

**Faithful with caveats:**

| POSIX | caveat |
|---|---|
| `fork` proper | clone-of-parked via the fork endpoint: needs a durable-instrumented build (R8 on the critical path), full window copy until CoW, and clones **all** vCPUs (forkall) where POSIX forks only the calling thread — benign in practice, since POSIX itself restricts post-fork threaded code to async-signal-safe calls |
| shell `trap` (INT/TERM/EXIT) | doorbell word checked at command boundaries — the *same* delivery points Bash itself uses (L0 below); compute-bound code waits on L2 |
| `SIGCHLD` | reap-by-`poll` (L1/L2 below add async delivery when built) |
| `EINTR` / signals while blocked | interruptible parks — wake-with-interrupted-status; a small runtime slice (L1 below), not yet built |
| async handlers (`SIGTERM`, `setitimer`) | safepoint-injected delivery (L2 below): poll-granularity latency, bounded once Phase-4 back-edge polls land (`DURABILITY.md` R6) — the JVM/Go/wasmtime-epoch norm |
| catching memory faults | in-window unmapped/protected-page faults are **pager events with retry precision** (§5 faults) — *stronger* than wasm, where every trap is terminal; confinement faults stay terminal by design |
| `getpid`, pid files | personality-local pids; no cross-tree pid meaning |
| job control (`fg`/`bg`/`kill %1`) | personality bookkeeping over held handles; see the SIGSTOP gap below |
| `exec` in place | spawn + transfer the pid label + exit — observable only to a peer inspecting window identity |
| `select`/`poll`/`epoll` | needs one readiness convention across channel caps (futex word / `IoRing` completions) — design work, feasible |

**Absent (deliberate, or genuinely hard):**

| POSIX | why |
|---|---|
| `SIGSTOP`/`SIGCONT` (Ctrl-Z) | no stop/continue op yet; the L2 safepoint machinery is the natural carrier (stop = park at the next poll) — open O12 |
| instruction-granularity signal delivery | the one thing never offered: interrupting arbitrary compute between two instructions — the part of POSIX signals POSIX itself fences off behind async-signal-safety |
| handler mutates the faulting context | `ucontext` fiddling / `siglongjmp` out of a fault handler — needs per-access deopt metadata in the JIT; the §5 retry-on-resume pager model covers the legitimate uses instead |
| ambient `kill(pid)` / `pkill` / global `ps` | refused on purpose — you kill what you hold; enumeration is §15's own-subtree-only |
| uids, setuid, permission bits | replaced by capability attenuation; uid-checking programs get stubs |
| CoW-fork efficiency (the Redis-BGSAVE pattern) | until Phase-4 CoW clone |

Bottom line: for the shell / coreutils / build-tool corpus this is on the order of ~90%
of the process model *as actually used*. With the signals ladder below, the true misses
shrink to instruction-granularity delivery and handler-mutates-context — both things
managed runtimes universally dropped — plus ambient authority, refused deliberately.
Calibration: Cygwin ran Bash for decades on strictly worse primitives (fork by
re-exec-and-copy over Win32); everything here is cleaner than that. On signals and
faults the design lands *stronger* than wasm, where traps are terminal and there are no
pager events at all.

### Signals — the delivery ladder

"No preemptive signals" is imprecise. The VM has interruption points that exist for
other reasons — the §5 kill poll, fuel slices, the durable async-STW redirect, every
park — and signal delivery is only a question of which action fires at them:

- **L0 — doorbell (guest convention, ships with the shell).** A word the guest checks at
  its own boundaries. Bash's `trap` is *natively* this model (traps are delivered at
  command boundaries), so shell semantics are exact, not approximated. Zero VM change.
- **L1 — interruptible parks (small runtime slice).** A parked call (`join`, endpoint,
  pipe) wakes with an interrupted status instead of its result; the libc runs pending
  handlers and re-issues, or returns `EINTR`. Parks are runtime state that already
  delivers several outcome kinds — this is the signals-while-blocked half of POSIX.
- **L2 — safepoint-injected handlers (rides existing polls).** At a kill-poll / fuel
  check, the slow path redirects the fiber into a registered guest handler, then resumes
  at the poll site — the same interrupt-at-safepoint pattern the §5 kill path and the
  durable async STW already use, with a non-lethal action. The hot path pays nothing
  new. Latency = poll granularity, and the bound is exactly `DURABILITY.md` R6 (tight
  direct-call loops are poll-free until Phase-4 back-edge polls) — signals inherit the
  kill/freeze latency work for free. This is the JVM-safepoint / Go-preemption /
  wasmtime-epoch consensus: handlers run only at consistent states, a *saner* contract
  than POSIX's async-signal-safety minefield, not a weaker one.
- **Never — instruction-granularity interruption** of arbitrary compute. The residue is
  the part of POSIX that POSIX itself fences off.

L0 ships with the shell; L1/L2 are parked (S13) until a personality claims a consumer.

## 10. The validation ladder

Unchanged in substance from v1, restated against the substrate:

- **Stage 0 — no processes**: Bash/ash as a pure interpreter (`sh -c`, builtins). Needs
  only fs directory ops (`FS_STAT`/`FS_READDIR`/`FS_MKDIR` — embedder-tier protocol
  extension) + the existing port model. Proves nothing about this file; unblocks demos.
- **Stage 1 — spawn/wait/exec**: `Domain` create/grant/start/join/poll/kill + BusyBox
  applets as `Module` grants. Prerequisites: JIT async children + compile cache.
- **Stage 2 — pipes/IPC**: **host-served** pipe cap + fd table — **no endpoints
  required** (red-team: the shell must not be hostage to the hardest machinery).
  `ls | grep x > out` byte-identical to native on all three engines.
- **Stage 2.5 — the interposition gate**: the BusyBox suite passes **unmodified** under
  a *guest-implemented* virtualizing-fs personality (a parent serves the child's `fs`
  via endpoints). This is the self-similarity thesis made a CI gate — the keystone
  primitive ships with a real consumer, not just synthetic tests.
- **Stage 3 — fork**: §7 clone (R8 closure). Full Bash: subshells, `$( )`, `&`. The
  capstone — the demo wasm needed a spec fork (WASIX) to approximate.

## 11. Testing

- Every substrate op: interp ↔ bytecode ↔ JIT differential incl. errno paths and trap
  kinds (repo standard).
- **Endpoint semantics fuzz**: generated call/serve/reply/kill interleavings on the
  deterministic oracle (`run_scheduled`/`explore_all`), JIT differentially against it;
  kill-during-park at every point; no hangs, no stale-handle confusion (generation
  checks), teardown always releases parked peers.
- **Grant/teardown soundness**: kill/exit at every lifecycle point; refcounted resources
  observe drops exactly once.
- **Attest cannot be spoofed**: a maximally-virtualizing parent (all caps
  endpoint-served) cannot alter the child's `attest` report — pinned as a test, since it
  is the security claim.
- **Freeze × processes**: durable subtree frozen with domains parked on endpoints
  round-trips (nested mode); detached-subtree freeze explicitly out of v1 scope, tested
  to *fail closed*.
- Stage demos gate CI like the Lua fixtures.

## 12. Plan tracker

| # | Slice | Depends on | Status |
|---|---|---|---|
| S0 | **Spikes first**: cross-domain futex on shared backing (O2) → library-endpoint feasibility; endpoint-RTT budget table (F6) | — | **done** — nested futex confirmed (`futex_cross_domain.rs`); sibling wakeup-key gap characterized + fix scoped to S9; F6 RTT model in §4 |
| S1a | JIT per-carve compile cache | — | **done** (`jit_instantiate_cache.rs`; position-independent, one compile per `(module,entry,size)`) |
| S1b | Canonical-key futex — key `Backed` pages on `(backing, region_off)` | — | **interp done** (`futex_region_canonical.rs`, negative-checked; `FutexKey::{Anon,Region}` in both schedulers). JIT half lands with S1c (JIT threads share one window today, so no aliasing gap until concurrent §14 children exist) |
| S1c | OS-thread children on the JIT: `instantiate` spawns the cached child in its **own guarded window** (keeps the trailing guard — in-place would break carve isolation), `join` parks on the completion cell; live channel is a granted `SharedRegion`, not the copied carve | S1b | todo |
| S2 | Grant capabilities into a child's powerbox ("children born destitute" fix). **Done (interp): (a)** `instantiate_granted` (op 8) — single coordinate-free cap (`Stream`/`Exit`/`Clock`) as the child's 3rd entry arg (`instantiate_granted.rs`); **(b)** `instantiate_named` (op 11) — a **multi-cap grant list** (`grants_n` × 16-byte `{name_off,name_len,handle,flags}` records) re-granted **by name**, child discovers via `cap.self.resolve` (`instantiate_named.rs`). stdout/stderr grants share the parent's sink (stdio inheritance); non-copyable caps refused (`CapFault`). Remaining: create-suspended/start split + teardown/refcount + JIT parity | — | **grant list + names done** |
| S3 | Lifecycle: `poll`/`kill`/`detach` (+ per-child kill cell on JIT) | S1 | **poll + detach done** (interp; `Instantiator` ops 9/10, `lifecycle_poll_detach.rs`): `poll` is a non-destructive 0/1/2 status probe (the `WNOHANG`/`SIGCHLD` reap probe), `detach` drops the join claim (child runs on, parent never blocks). `kill` remains — needs a per-child §5 interrupt on the M:N executor (run-wide interrupt exists; per-child does not) |
| S4 | fs dir ops (**done** — landed with the Postgres `initdb` work: `FS_STAT`/`MKDIR`/`RMDIR`/`OPENDIR`/`READDIR`/`CLOSEDIR`); POSIX personality lib: fd table, **host-served** pipe, proc ABI | S2 | fs ops done; personality lib todo |
| S5 | `Budget` split/read; detached window minter behind a granted authority | S2 | todo |
| S6 | `cap.self.attest` incl. freeze authority (+ the §14 amendment PR into DESIGN.md) | S5 | todo |
| S7 | BusyBox port; stage-1/2 demo gates (**endpoint-free**) | S3,S4 | todo |
| S8 | Bash stage-0 port (interpreter-only; autoconf cross-config, `--noediting`) | fs ops | todo |
| S9 | `Endpoint` v1 — library over SharedRegion+futex (S0: viable); canonical-key futex now lands earlier (S1b); kill-safe cancel; FIFO | S0,S1b,S3 | todo |
| S10 | **Interposition gate** (stage 2.5): guest-implemented virtualizing-fs personality runs the BusyBox suite unmodified | S9 | todo |
| S11 | R8 closure (`call_indirect` durable coverage); `clone` of parked domains (full-copy) + fork endpoint with duplicated reply token | S9, durable | todo |
| S12 | Bash stage-3; suite subset as CI gate | S8,S11 | todo |
| S13 | CoW clone; detached-subtree freeze; `ModuleLoader`; async endpoints over IoRing; signals L1 (interruptible parks) + L2 (safepoint handlers, rides Phase-4 back-edge polls) + stop/continue (O12); pager-authority generalization incl. self-paging | S11 | parked |
| S14 | Second personality (actor-model sketch) — the design-for-two check, build only when wanted | S9,S5 | parked |

## 13. Risk register / open questions

| # | Risk / question | Where | Status |
|---|---|---|---|
| O1 | Endpoint servicer DoS (never replies): callers park forever. v1 answer: your servicer is in your grant chain — you trusted it; a personality may add timeouts. Is that acceptable for cross-*sibling* endpoints? | §4 | open |
| O2 | Pipe substrate: host-served (simple, buffers lost on freeze) vs guest ring + futex (durable). **Spike done (§4 S0 results):** nested futex rendezvous works today (pinned by `futex_cross_domain.rs`); sibling aliases are value-coherent but need region-canonical futex keys for wakeup (Linux shared-futex analogue) — no confinement-hinge contact. Integration (O15) promoted this to **S1b** (it also gates JIT concurrent parent↔child) | §9 | resolved — nested confirmed; sibling/JIT fix is S1b |
| O3 | JIT compile-cache: **built** with key `(funcs ptr, n_funcs, entry, size_log2)` — carve base is *absent* (position-independent reuse), so it hits across offsets, not just repeated same-slot spawns. Residual: a robust separate-module identity (a digest would beat the funcs pointer, though the run-lifetime grant contract makes a stale-pointer collision impossible within a run); cache eviction if a long shell session accumulates many distinct applets | §4, §5 | mostly resolved |
| O15 | **In-place shared-window children are unsafe** (§4 S1 finding): D38 clamp confines the *offset*, but a width-`w` access at a carve's top reaches `w−1` bytes past it — caught by the per-window trailing guard page, which densely-packed carves lack. So a concurrent JIT child must keep its **own guarded window**; the live parent↔child channel is a `SharedRegion` + canonical-key futex (S1b), never implicit carve addresses. No D38 change. Decided; the alternative (per-carve guard pages) is rejected as wasteful and layout-invasive | §4 | resolved (own-window + SharedRegion) |
| O4 | Detached windows on the browser/wasm32 host: pool sizing inside one wasm memory | §5 | open |
| O5 | `attest` provenance granularity: report authority *set* or just tier + platform-only bit? Smaller is better (F2) | §6 | open |
| O6 | Multi-window (detached) subtree freeze: consistent cut across windows — interacts with `DURABILITY.md` R4 | §5, §11 | open |
| O7 | Replay-fork viability: input recording cost for a real shell; where the determinism boundary sits (host caps replayed vs re-run) | §7 | open — cheap spike |
| O8 | Budget resource vector: fuel + mem + spawn now; handle-table slots? endpoint count? Keep the vector short | §5 | open |
| O9 | Durable instrumentation overhead on a shell-sized module (`DURABILITY.md` R7) — fork's tax, measured on Bash | §7 | open |
| O10 | Endpoint × freeze consistent cut: a frozen domain parked on a call whose servicer is *outside* the cut — the pending call is neither host state (D-scope re-supply) nor captured guest state. v1 rule: freeze-boundary calls are **re-issued** on thaw; idempotence is the personality's problem. Validate against reload-not-reissue (R8/R11 machinery) | §4, §11 | open |
| O11 | `clone` captures all vCPUs (forkall) vs POSIX calling-thread-only fork — benign for shells (POSIX post-fork threaded code is async-signal-safe-only anyway); pin the divergence in the personality doc | §7 | open |
| O12 | No stop/continue (SIGSTOP / Ctrl-Z): the L2 safepoint redirect is the natural carrier (stop = park at the next poll instead of running a handler) — fold into the signals ladder rather than mint a bespoke op? | §9 | open |
| O13 | Signals L1/L2 are designed, not built: until they land, parked calls are uninterruptible short of kill and compute-bound code sees no delivery — scope the POSIX personality's claims to L0 meanwhile | §9 | open |
| O14 | Attest's `freeze_authority` field requires freeze authority to be *explicit* — today subtree-freeze authority is implicit in nesting; plumbing needed before the report can be truthful | §6 | open |

---

**[PROPOSED DECISION D63 — process primitives, not process policy (the D56 move, applied
to processes).]** The substrate is four capability-shaped objects — `Domain`
(create-suspended / grant / call / resume / start / join / poll / kill / detach / clone),
`Endpoint` (mintable guest-serviceable interfaces — §14's virtualization as mechanism),
window sources (visibility = provenance: nested `AddressSpace` carve vs. platform-minted
detached window), and `Budget` (§15's quota as a passable, splittable object) — riding
existing machinery (`Module`, `SharedRegion`, futex, `IoRing`, D60 durable clone). Every
process "dial" is an argument or a verb, never a mode flag; OS semantics (POSIX for
Bash/BusyBox first; others by design) are guest-library personalities the substrate never
learns. One deliberate amendment to settled text: `cap.self` gains a read-only,
non-interposable `attest` (isolation tier + window provenance) as the sole trust anchor
against hostile nested hosts — amending §14's "no am-I-nested query" with an opt-in
exception, bounded by the hard rule that reflection + attest are the *entire*
non-interposable surface. **No new instructions; no verifier or confinement-lowering
change; all substrate code is authority-TCB (§2a).** Rationale: the factoring maps
one-to-one onto settled decisions (D19 attenuation, D42 ABI, D46 open capabilities, D56
primitives-not-policy, §15 quotas-as-capabilities), so the marginal cost is capability
plumbing; orthogonality-via-arguments avoids the mode-matrix op explosion; and the
staged shell (BusyBox → Bash) remains the differential-testable exit criterion.
