# Process substrate & OS personalities — processes over domains

> **Status: PROPOSED — design draft v2, nothing built.** Working tracker for a **process
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
   park-only-the-calling-fiber is its named follow-up), and the JIT **recompiles the
   child per carve geometry** (mask/`sub_base` baked into code) — spawn cost on the JIT
   is a module compile unless cached.
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

A guest-servicedable capability interface: the general form of the `Yielder`, and the
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
- Bulk data rides `SharedRegion`s granted alongside; the endpoint carries scalars
  (D42 borrow-only discipline unchanged).

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
  authority. Numeric quota, host-enforced at mint. Standard geometry also makes the JIT's
  per-carve compile cache actually hit (gap 4 in §0).
- Demand-paged sits between, and honestly: **pager authority is read authority** — a
  domain whose pages are supplied by its parent is visible to it. `attest` (§6) reports
  this.

Trade-offs stated once, honestly: detached subtrees make freeze/clone a **multi-window
snapshot** — new `DURABILITY.md` work (nested subtrees freeze today); nested carves
subdivide parent VA (real in the browser's wasm32 window) and vary carve geometry (JIT
compile cache pressure). Projects choose per child; a shell would plausibly run coreutils
detached and its own helper coroutines nested.

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
                       window_provenance }                   (which authorities hold
                                                              map/read/pager rights over
                                                              my backing: platform-only,
                                                              or ancestor-held)
```

- **Read-only report, no negotiation.** A domain that dislikes the answer exits before
  touching secrets. Fits D46's `cap.self` contract exactly: reflection confers nothing,
  adds no grant-graph edge, and extends the stated "no deniable grants" transparency
  principle from *authority* to *exposure*.
- **Honest limits, recorded**: attest cannot protect you from your creator having chosen
  your initial state (nothing can); and per §2, tiers 0/1 are never a Spectre boundary —
  a domain requiring protection from a *distrusted* host must see `tier 3` in the report
  or refuse, which is exactly §14's "a tier-3 child requires the host to grant a real
  process."
- **The friction, named**: §14 currently says *"There is no 'am I nested?' query by
  default."* Attest is deliberately such a query (provenance reveals nesting). Proposed
  amendment: the default stands — no ambient nesting query, and a virtualized capability
  remains indistinguishable — but a domain may **opt in** to the one platform-vouched
  provenance report, because self-protection from hostile nested hosts is impossible
  without a trust anchor. This is a change to settled §14 text and is called out for
  exactly that reason (change settled things only with a reason — this is the reason).
  The carve-out must stay **tiny**: every op added to the non-interposable namespace
  erodes the self-similarity the rest of the design exists to provide. `attest` and the
  existing reflection are the whole list.

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
  not a substrate concept. The supervisor pattern (shell runs under a tiny init domain)
  falls out of "someone serves your fork endpoint" — no special architecture.
- **Cost/coverage, honestly**: v1 clone is a full window copy (CoW rides `DURABILITY.md`
  Phase 4, not blocked on). The caller must be durable-instrumented, and the transform
  today treats `call_indirect` to may-suspend targets as out of scope (R8) — Bash
  dispatches builtins through function-pointer tables, so **R8 closure is fork's real
  dependency**, tracked as its own slice. Fallback worth keeping alive: **replay-fork** —
  the cooperative engine is deterministic, so a clone can be produced by re-executing
  from `create` with recorded inputs up to the park; zero snapshot machinery, O(execution)
  cost, no instrumentation requirement. Personalities that never fork (most) pay nothing.

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
  matching how shells actually poll traps. Catchable async signals stay out until a real
  script demands them.
- **fork**: §7. BusyBox `ash`/`hush` (fork-less NOMMU designs) before Bash.

## 10. The validation ladder

Unchanged in substance from v1, restated against the substrate:

- **Stage 0 — no processes**: Bash/ash as a pure interpreter (`sh -c`, builtins). Needs
  only fs directory ops (`FS_STAT`/`FS_READDIR`/`FS_MKDIR` — embedder-tier protocol
  extension) + the existing port model. Proves nothing about this file; unblocks demos.
- **Stage 1 — spawn/wait/exec**: `Domain` create/grant/start/join/poll/kill + BusyBox
  applets as `Module` grants. Prerequisites: JIT async children + compile cache.
- **Stage 2 — pipes/IPC**: `Endpoint` v1 + the personality pipe + fd table.
  `ls | grep x > out` byte-identical to native on all three engines.
- **Stage 3 — fork**: §7 clone (R8 closure or replay-fork). Full Bash: subshells,
  `$( )`, `&`. The capstone — and the demo wasm needed a spec fork (WASIX) to approximate.

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
| S0 | JIT async children (park-only-calling-fiber) + per-carve compile cache | — | todo |
| S1 | `Domain.grant` + create-suspended/start split; child `cap.self.resolve` names; teardown/refcount rules | — | todo |
| S2 | Lifecycle: `poll`/`kill`/`detach` (+ per-child kill cell on JIT) | S0 | todo |
| S3 | `Endpoint` v1 (mint/serve/reply, sync rendezvous, kill-safe cancel) on all three engines | S0 | todo |
| S4 | `Budget` split/read; detached window minter behind a granted authority | S1 | todo |
| S5 | `cap.self.attest` (+ the §14 amendment PR into DESIGN.md) | S4 | todo |
| S6 | fs dir ops; POSIX personality lib: fd table, pipe (decide via O2), proc ABI | S1,S3 | todo |
| S7 | BusyBox port; stage-1/2 demo gates | S2,S6 | todo |
| S8 | Bash stage-0 port (interpreter-only) | fs ops | todo |
| S9 | `clone` of parked domains (full-copy) + fork endpoint pattern; R8 closure **or** replay-fork spike | S3, durable | todo |
| S10 | Bash stage-3; suite subset as CI gate | S8,S9 | todo |
| S11 | CoW clone; detached-subtree freeze; `ModuleLoader`; async endpoints over IoRing | S9 | parked |
| S12 | Second personality (actor-model sketch) — the design-for-two check, build only when wanted | S3,S4 | parked |

## 13. Risk register / open questions

| # | Risk / question | Where | Status |
|---|---|---|---|
| O1 | Endpoint servicer DoS (never replies): callers park forever. v1 answer: your servicer is in your grant chain — you trusted it; a personality may add timeouts. Is that acceptable for cross-*sibling* endpoints? | §4 | open |
| O2 | Pipe substrate: host-served (simple, buffers lost on freeze) vs guest ring + futex (durable, needs cross-domain futex on shared backing — verify keys work) | §9 | open — spike early |
| O3 | JIT compile-cache key: (module digest, carve base, size_log2) — hit rates under a real shell; detached-standard-geometry as the mitigation | §5, gap 4 | open |
| O4 | Detached windows on the browser/wasm32 host: pool sizing inside one wasm memory | §5 | open |
| O5 | `attest` provenance granularity: report authority *set* or just tier + platform-only bit? Smaller is better (F2) | §6 | open |
| O6 | Multi-window (detached) subtree freeze: consistent cut across windows — interacts with `DURABILITY.md` R4 | §5, §11 | open |
| O7 | Replay-fork viability: input recording cost for a real shell; where the determinism boundary sits (host caps replayed vs re-run) | §7 | open — cheap spike |
| O8 | Budget resource vector: fuel + mem + spawn now; handle-table slots? endpoint count? Keep the vector short | §5 | open |
| O9 | Durable instrumentation overhead on a shell-sized module (`DURABILITY.md` R7) — fork's tax, measured on Bash | §7 | open |

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
