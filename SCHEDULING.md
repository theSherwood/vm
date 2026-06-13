# Scheduling & concurrency primitives (D56 / D57)

How the VM exposes concurrency, why, and the roadmap for guest-built M:N
schedulers — including the **migratable-fiber** evolution and its honest cost.

## The model: two primitives, nothing more

The VM ships exactly **two** concurrency primitives, and intends to keep it that way:

| Primitive | What it is | Why only the VM can provide it |
|---|---|---|
| **vCPU** (`thread.spawn`/`join`) | one real OS thread, 1:1 (D56) | parallelism across physical cores — not expressible in portable guest code |
| **fiber** (`cont.new`/`resume`/`suspend`) | a *stackful* coroutine that owns a native call stack | switching the native execution stack needs the `svm-fiber` asm stack-switch — the guest's instruction set can't save/restore SP + callee-saved regs and redirect execution mid-function |

Plus the coordination glue that is *also* primitive-minimal: the `wait`/`notify`
**futex** and **C11 atomics** over the shared window.

Everything richer — mutexes, channels, M:N schedulers, work-stealing, async
runtimes — is **guest-built** from those. That is the D22/D56 thesis: *primitives,
not policy; no scheduler in the VM.*

### "Stackless tasks" are NOT a third primitive

A **stackless task** is a function rewritten as a **state machine**: a struct
holding its locals, plus a resume function with a `switch` on a state field.
"Suspend" is `return`; "resume" is calling it again with the saved state. This is
exactly how Rust `async` / C++ coroutines lower. It needs **zero** VM support — it
is ordinary loads/stores/branches the guest already has; the VM never sees a new
opcode. So the primitive surface stays at **two**. (This is the resolution to "are
we growing a third primitive?" — no, stackless is a guest *pattern*.)

## Two flavors of guest-built M:N — both work *today*, no VM change

|  | **Sharded** (over fibers) | **Work-stealing** (over stackless tasks) |
|---|---|---|
| Task representation | a stackful fiber (`cont.*`) | a state-machine struct (guest data) |
| Migration across cores | **yes** since D57 3c — `cont.resume` from any vCPU claims + migrates the fiber (the sharded *demo* keeps tasks pinned by choice) | **yes** — moving a struct is a pointer hand-off |
| Load balancing | none (thread-per-core) | dynamic (steal from busy workers) |
| Safety of migration | the `Ownership` claim (loom-verified) + the 3c empirical net | safe by construction (it's just data) |
| Can suspend *unmodified* code | **yes** (suspends the whole native stack) | **no** — function-coloring: only at transformed points |
| Real-world analogue | glommio, seastar, redpanda (thread-per-core) | tokio, Go-style work-stealing |
| VM change required | none | none |

**Why both exist:** stackless is *strictly less expressive* (the function-coloring
problem — a stackless task can only suspend at points in its own transformed body,
never across an arbitrary or unmodified call frame). Fibers are the only way to
cooperatively suspend **unmodified real code** (a recursive parser, a library you
can't touch) and they underpin the **§14 fault-driven yield** (suspending at an
arbitrary hardware-fault PC is inherently stackful — there is no state-machine
suspend point there). Conversely, stackless is the only safe way to *migrate* a
task for work-stealing without moving native stacks. They are complementary; the VM
offers both substrates and the guest picks.

## Fibers are migratable on both backends (3a–3c landed)

The per-vCPU fiber tables are gone on both backends: the interp's run-shared
`FiberRegistry` (step 3b-i — pure-data `Vec<Frame>` hand-off, the oracle) and the
JIT's domain-shared `fiber_rt::SharedFiberTable` (3b-ii storage + the live
`Ownership` word; 3c lifted the affinity gate) give one handle namespace and a
single-owner claim per slot, so **`cont.resume` on any vCPU migrates a suspended
fiber to that vCPU's OS thread** — exactly one racing claimant wins; a loser gets
a clean `FiberFault`. Thread-per-core affinity (glommio/seastar-style) remains
available as *guest policy* — a scheduler that only resumes its own fibers gets
exactly the old locality — but it is no longer a VM constraint.

## Migratable fibers → stackful work-stealing (D57)

**Status: ADOPTED — the primitive landed (slices 3a–3c); Demo 3 (a guest stackful
work-stealing scheduler over it) is the remaining roadmap item.** The outcome is
work-stealing M:N over *stackful* fibers — Go-class scheduling for **arbitrary
unmodified compiled code** inside a confinement sandbox, strictly more than either
flavor above. Go is the decade-long, planet-scale existence proof of stackful
work-stealing. It carried a real, eyes-open cost, accounted below.

### The honest tension with D56

D56 *removed* a VM-owned M:N executor specifically because it reintroduced "the
project's highest-risk unsafe — **fiber migration across OS threads** — in the
runtime TCB." Migratable fibers re-accept **exactly that risk**. The difference
from what D56 removed:

- D56's executor was a **VM scheduler** (policy lock-in + the double-scheduler
  pathology). Migratable fibers are a **primitive**: the VM enforces only
  *single-owner resume-from-any-thread*; the **guest** owns the work-stealing deque
  and stealing policy. → resolves D56's *policy* objections.
- It does **not** resolve D56's *TCB-risk* objection: cross-thread native-stack
  resumption is back in the trust boundary.

So this is a deliberate re-acceptance of a known risk, justified by the capability
it unlocks — **to be earned with verification + review, not assumed.**

### Design sketch

1. **Shared, transferable fiber registry** replacing the per-thread tables, with an
   **atomic single-owner protocol** (states: `running` / `runnable` / `owned`). A
   fiber is stealable only when **voluntarily suspended** (in a runnable pool owned
   by no thread); the currently-running fiber is never stealable. Stealing =
   **atomic pop / CAS** from the pool; the loser backs off. *This invariant — "one
   stack, exactly one thread" → "exactly one CAS wins" — is pure atomics and is
   loom-verifiable*, like the futex.
2. **Fault-suspended fibers stay pinned.** A fiber suspended *from inside the
   §5 fault handler* (the §14 demand-paging case, `SA_NODEFER` + suspend-during-
   fault) carries thread-affine recovery state (`sigjmp_buf`/VEH `CONTEXT`) and is
   **excluded** from the steal pool. Voluntary-suspension fibers (the 99% case)
   carry only thread-agnostic register context and are stealable.
3. **The asm switch barely changes.** Resuming a fiber's saved context from another
   thread is the same `svm-fiber` instruction sequence; unix has no thread-bound
   state in it, and Win64 already swaps the TEB `StackBase`/`StackLimit`/
   `DeallocationStack` per switch (so "this stack is now active on *this* thread" is
   already handled). Fixed-mmap stacks are fine — migration moves the executing
   thread, not the stack (Go copies stacks only for *growth*).

### Verification story (and its limit)

The ownership protocol is **loom-verifiable** — that is where the dangerous
invariant lives, and it is done (step 3a). The **composition** of (verified protocol) +
(real asm switch) + (per-thread signal recovery) cannot be exhaustively model-checked
(asm + signals are outside loom/TSan) — the *same* residual caveat that already applies
to today's fiber+thread code, exercised harder.

**No expert review is available for the asm/signal seam** (a stated project constraint),
so safety for step 3c rests entirely on an **empirical net** designed to make every
plausible failure mode *loud and detectable* rather than silent corruption. *(Status: 3c is
landed and every net layer below is built and green — see §5's 3c entry for the concrete
artifacts, including the ASan fiber-switch annotations that made layer 2 actually runnable.)* Two facts make
this a reasonable — not reckless — posture:

- The cross-thread resume introduces **no new assembly**: it calls the *same* `svm-fiber`
  stack-switch already in production for `thread.spawn`'d vCPUs and per-vCPU fibers (unix
  has no thread-bound state in the switch; Win64 already swaps the TEB stack fields per
  switch). The delta is *which thread* calls it; the instruction sequence is unchanged.
- The project **already** trusts that asm via differential + stress, not TSan (TSan cannot
  instrument JITted code). So this extends the existing bar; it does not invent a new one.

**The net (each layer turns a class of silent failure into a detected one):**

1. **Differential randomized-schedule fuzzer.** The interpreter is **safe Rust — it cannot
   corrupt memory** — and is the oracle. A generator emits guest work-stealing-over-fibers
   schedules with randomized migration decisions; each runs on interp **and** JIT and must
   agree on the result and never crash. Any JIT-only divergence or fault is an asm-seam bug.
   Thousands of seeds as a stable CI test + a libFuzzer `diff` target (the §18 methodology
   the whole JIT already uses).
2. **Sanitizer job (ASan).** The *runtime glue* around the switch — the shared registry, the
   `Box<Fiber>` lifetimes, `CURRENT_RT`, the yielders — is **Rust and is ASan-instrumentable**
   (only the JITted guest body is opaque). A switch that corrupts a stack or frees a live
   fiber surfaces as an ASan report. Run the stress suite under ASan in a dedicated job.
3. **Runtime single-owner assertion.** The `Ownership` CAS already guarantees exclusivity;
   assert it *at the resume seam* so a double-resume **panics loudly** (→ detect-and-kill)
   instead of running one stack on two threads.
4. **Guard-page detection.** `svm-fiber` stacks are guard-paged, so a wild/torn switch faults
   into the §5 handler (a clean kill) rather than scribbling silently.
5. **Soak.** Many workers × many fibers × many migrations, repeated — CI-bounded, plus a
   longer nightly run.

**Honest residual:** fuzzing *detects*, it does not *prove*. A sufficiently rare cross-thread
race could still escape the net. The layers above are chosen so that the *likely* failure modes
are caught and made non-silent; we accept the residual knowingly, as the price of the
capability, and gate landing 3c on the net being green and the stress genuinely exercising
migration (asserted, not assumed — e.g. counting observed cross-thread resumes).

## Demo roadmap

1. **Demo 1 — sharded stackful M:N** *(done — `demos/mn_sched`)*. A guest cooperative scheduler
   over `cont.*`: N `thread.spawn` workers, each round-robining its own pool of
   fiber tasks that yield; a shared atomic aggregate. Proves fibers + threads +
   atomics compose into a real M:N runtime, and establishes the scheduler machinery
   (run queue, resume-until-suspend, park/unpark) the later versions reuse. Honest
   about affinity (tasks pinned per worker).
2. **Demo 2 — stackless work-stealing M:N** *(done — `demos/work_stealing`)*. Tasks as guest
   state machines, queues over atomics — cross-core load balancing with
   **no VM change**. Proves the harder claim and is the natural lead-in to the async
   I/O ring (B).
3. **Migratable-fiber primitive + Demo 3 — stackful work-stealing** *(done — the primitive via
   slices 3a–3c below; the demo is `demos/steal_fibers`)*: a guest work-stealing scheduler
   whose tasks are **fibers** — an idle worker steals a *suspended fiber* from a sibling's deque
   and resumes it on its own OS thread, the task yielding from inside a nested call frame
   (inexpressible stackless) with live locals carried across every migration. Both invariant
   totals (`256` work units, `121920` return-sum — the stack-integrity check) print identically
   on interp (the migrating-registry oracle) and JIT (real cross-thread switches):
   `c_frontend::c_guest_steal_fibers_demo` + `run::demo_steal_fibers_runs`. The VM contributed
   exactly the namespace + arbiter; every scheduling decision is guest C.
   - **Step 3a — the single-owner ownership protocol, loom-verified (DONE).** The
     load-bearing atomic state machine (`OWNED`/`RUNNABLE`/`RUNNING`/`FREE`) is built
     in isolation in `crates/svm-jit/src/fiber_registry.rs`: a fiber is stealable only
     while `RUNNABLE` (voluntarily suspended, ownerless), and a steal is a single
     `RUNNABLE → RUNNING` CAS, so **exactly one** thread can ever claim it (acquire on
     the claim / release on the suspend publishes the saved context). This is the whole
     safety argument of migration — "one native stack, exactly one thread" — reduced to
     pure atomics and **loom-model-checked** (`loom_single_owner_steal_is_exclusive`
     proves exactly-one-winner across every interleaving + the acquire/release
     visibility; `loom_running_fiber_is_never_stealable`), plus a mutation check
     confirming a non-CAS steal makes loom find a double-claim. The ownership word is
     **generation-tagged** (`(generation, state)` packed into one `AtomicU64`, the
     generation bumped on `finish`) so the shared registry can **reuse slots** without
     the classic **ABA hazard**: a stealer holding a stale `(slot, gen)` after the slot
     was finished and reused for a different fiber finds its `try_steal(gen)` CAS fail —
     pinned by a deterministic reuse-cycle unit test (dropping the bump defeats it). It
     is **not yet wired into the live runtime** — that integration (a shared registry
     replacing the per-thread tables) and the cross-thread asm resume (design sketch #3,
     expert-review gated) are the remaining steps, specified in
     "Integration design (steps 3b–3c)" below. Verifying the dangerous invariants first
     is the "earn the risk" discipline this feature demands.

## Integration design (steps 3b–3c)

The full plan for wiring the verified ownership protocol (step 3a, above) into the live
runtime. Written for review **before** any escape-TCB/asm code, because steps 3b–3c
re-accept the cross-thread native-stack-resume unsafe D56 removed.

### 0. The reframing: the VM owes a *namespace + arbiter*, not a scheduler

The instinct is "build a Chase-Lev work-stealing deque." **The VM builds no deque.** D56/D57
is *primitives, not policy*: the work-stealing run-queue is **guest code** (a deque of fiber
handles in guest memory, exactly as the stackless `demos/work_stealing` already does for
state-machine tasks). The VM owes only two things migration needs that the guest cannot
provide itself:

1. a **shared fiber-handle namespace** — any worker (vCPU) can *name* any fiber, vs. today's
   per-vCPU tables where a handle indexes only its creator's table; and
2. the **single-owner arbiter** — when two workers race to `cont.resume` the same handle,
   exactly one wins and the other gets a clean `FiberFault`. That arbiter is the step-3a
   `Ownership` CAS (`RUNNABLE → RUNNING`), already loom-verified.

So the entire VM-side surface is **one shared slot table, each slot carrying an `Ownership`
word** — no deque, no new policy, no scheduler. This shrinks the risk surface dramatically and
keeps the "VM ships mechanism" thesis intact.

### 1. Handle = `(slot, generation)`; one shared registry per run

Replace the per-vCPU/per-runtime fiber tables with **one registry shared by all vCPUs of a
run** (an `Arc`), mirroring the threaded-install refactor that already made the JIT
`call_indirect` table (the interp's `DomainTable`) a shared atomic structure (DESIGN §22).
A `cont.new` handle becomes the pair **`(slot index, generation)`** packed into the i32/i64
the guest already holds; `cont.resume`/`suspend` carry it. The generation is the step-3a ABA
guard: it lets the registry **reuse** a finished fiber's slot (so a long work-stealing session
doesn't leak slots up to `MAX_FIBERS`) while a stale handle to the previous occupant fails
closed.

**A shared registry also *unifies the handle namespace* across backends — closing a known
interp↔JIT divergence (do this here, not before).** Today the two backends number fiber handles
differently: the interp keeps the root computation as fiber-slot 0, the JIT runs the root
off-table, so the same fiber is handle `N` on the interp and `N−1` on the JIT (and a forged
handle masks to different slots). It is **safe** — numbering is internal, resolution confined to
the domain's own table (DESIGN §3a) — and **harmless in production** (a single backend is
self-consistent), but it is a real hole in the interp≡JIT differential: a fiber-handle *value*
that flows into observable output diverges. The fiber fuzzer works around it (resumes only
genuine handles); the durable record is in DESIGN §3a. The fix is *this* slice — a shared
registry hands both backends the **same** `(slot, generation)` for a fiber, so handle values
match and the differential can observe them. Deliberately **not** fixed as a standalone
pre-change: aligning the numbering touches `resolve_fiber`/`chain`/`live_frames`, the exact code
3b-i restructures, so doing it separately would duplicate the risk on soon-replaced code.

### 2. Op → ownership transition

| Guest op | Registry action | `Ownership` transition |
|---|---|---|
| `cont.new(funcref)` | allocate or recycle a slot for a `Pending` fiber | `new_owned` / `recycle_owned` → `OWNED` |
| `cont.resume(h)` on **any** worker | claim the fiber to run it here | `begin_owned` (`OWNED→RUNNING`, owner) or `try_steal` (`RUNNABLE→RUNNING`, migrated); **lose ⇒ `FiberFault`** (running elsewhere, or stale generation) |
| `suspend` (voluntary) | publish to the shared pool — now migratable | `suspend_to_pool` (`RUNNING→RUNNABLE`) |
| `suspend` **inside the §5/§14 fault handler** | keep thread-affine (recovery state is bound to this thread) | `pin` (`RUNNING→OWNED`) — **excluded from migration** |
| fiber returns | free the slot for reuse | `finish` (`RUNNING→FREE`, generation bumped) |

This is the whole protocol: the existing `resolve_fiber` "already running / dead" `FiberFault`
check *becomes* the lost-CAS path, so the guest-visible error model is unchanged.

### 3. What stays per-thread (unchanged)

The **resume chain** (the worker's current native/eval call stack) and the JIT `yielders`
stack are per-running-worker — migration only ever moves a *suspended* fiber (on no chain).
The re-entrant-resume aliasing guard (`chain.contains`) stays a per-worker check. `CURRENT_RT`
becomes "the running worker's context," with the fiber *table* lifted out to the shared
registry.

### 4. Interp integration (step 3b-i) — safe, the oracle — ✅ DONE

A fiber in the interp is **pure data — `Vec<Frame>`** (`crates/svm-interp`), so migration is a
*data hand-off*, exactly like a stackless task; **no `unsafe`, no asm.** As built: the
per-`VCpu` `fibers: Vec<Fiber>` is now the run-shared **`FiberRegistry`** — one mutex'd slot
table per domain (root + `thread.spawn` children; nested §14 children get their own), whose
states mirror the verified protocol (`Pending` ≈ OWNED, `Parked` ≈ RUNNABLE, `Running` ≈
RUNNING — incl. parked mid-resume ancestors, never claimable — `Done` ≈ FREE-unrecycled) with
the mutex as the claim arbiter (the sanctioned safe-Rust stand-in; the lock-free `Ownership`
word is the JIT's, 3b-ii/3c). `cont.resume` on **any** vCPU claims the fiber (exactly one
racing claimant wins; a loser gets the clean `FiberFault`); `suspend` publishes it back, and
each vCPU's **root runs off-table** — which *unified the handle namespace* with the JIT
(handles `0, 1, …` on both backends; pinned by
`jit_fibers::fiber_handle_values_match_across_backends`, observed generatively by the fiber
fuzzer whose handle values now flow into compared output). Slots are deliberately **not
recycled** yet (matches both backends' historical tables; recycling + the generation tag land
with 3b-ii so the backends adopt one policy together). The `max_fibers` quota is now
**per-run** (§6 below). The explorer/oracle crux landed with it: `cont.new`/`cont.resume`/
`suspend` are **visible ops** recording a `MemAccess::Fiber` that conflicts with itself, so
DPOR explores both orders of racing fiber ops (cross-checked against the unreduced
brute-force enumerator), and the spin-detection fingerprint covers the chain/frames/parked
root instead of the now-shared table (sound: a vCPU's chain-held slots can't change within
its own turn, and every fiber op visibly changes its fingerprint). Tests:
`svm/tests/fiber_migrate.rs` (mid-life migration across vCPUs on the real pool + seeded +
exhaustive explorers, racing resumes ⇒ exactly one winner, the no-race foreign-claim control,
and the per-run quota pin).

### 5. JIT integration (steps 3b-ii, 3c) — the review-gated seam

- **3b-ii (storage refactor, affinity preserved) — ✅ DONE.** The per-thread
  `fibers: Vec<Option<Box<Fiber>>>` is now **`fiber_rt::SharedFiberTable`** — one slot table per
  domain (root + spawned vCPUs), each slot carrying the loom-verified [`Ownership`] word *live*:
  `cont.resume` claims via `begin_owned` (a re-entrant/racing/finished claim **loses and
  faults**, replacing the old per-thread `chain` check), a voluntary suspend re-parks via `pin`
  (thread-affine — deliberately *not* `suspend_to_pool` yet), and a return `finish`es the slot
  (generation bumped under the hood; the box is dropped, unmapping its stack). **Affinity is the
  slot's `owner` token:** a foreign vCPU's resume is a clean `FiberFault` — pinned, alongside the
  interp's migrating semantics, by `jit_threads::foreign_vcpu_resume_faults_on_jit_until_3c`
  (the test 3c flips into a differential). What this aligned with the interp registry: the
  **domain-wide handle namespace** incl. multi-vCPU (`fiber_namespace_is_domain_wide`), the
  **per-domain §15 fiber quota** (`jit_quota::jit_fiber_quota_spans_vcpus`), and forged-handle
  masking over one table shape. The fiber's body closure now reads `CURRENT_RT` dynamically
  (not captured) so its yielder pairing targets whichever thread resumes it — behavior-neutral
  under affinity, the 3c prep. Slot recycling + generation-carrying handles stay deferred to a
  later slice on both backends together (the interp explorer needs a fiber-return DPOR access
  first; `finish` already maintains the generation).
- **3c (the asm seam — empirical-net gated) — ✅ DONE.** A worker resumes fibers whose native
  stacks other workers created/suspended. As built: `cont.resume` claims via the new
  loom-verified [`Ownership::claim`] (fresh `OWNED` *or* pooled `RUNNABLE` → `RUNNING`; the
  acquire pairs with the suspender's `suspend_to_pool` release, so the saved stack context is
  fully visible cross-thread), a voluntary suspend publishes back to the pool, and per-thread
  state is **re-read after every switch-in, never carried across a suspension** — the fiber
  body closure and `fiber_suspend` re-read `CURRENT_RT` post-switch so yielder pairing targets
  the *resuming* thread (the two latent landmines this slice defused). Design sketch #3 held:
  the `svm-fiber` switch is the same instruction sequence (verified: SysV/AAPCS64 save only
  callee-saved registers, no TLS/x18; MS-x64 swaps the TEB stack fields per switch), and the
  re-entrancy discipline is preserved (no lock or `&mut` crosses a switch; only the
  address-stable boxed fiber, exclusive via the claim). **The empirical net, delivered:**
  (1) `fiber_fuzz::generated_migration_schedules_agree_on_interp_and_jit` — randomized
  migration schedules, interp (the safe oracle) ≡ JIT, with cross-executor saved-stack resumes
  *counted and asserted*; (2) ASan is now genuinely runnable on fiber stacks — `svm-fiber`
  gained real `__sanitizer_{start,finish}_switch_fiber` annotations behind the `asan` feature
  (`RUSTFLAGS=-Zsanitizer=address cargo +nightly test -Zbuild-std --target
  x86_64-unknown-linux-gnu -p svm --features asan --test jit_fibers --test jit_threads --test
  fiber_fuzz`), whole fiber net clean; (3) a **runtime single-owner assert** at the resume seam
  (`FiberSlot::running_on` — a double-claim aborts loudly instead of running one stack on two
  threads); (4) guard-paged stacks fault into detect-and-kill on whichever thread runs the
  fiber; (5) concurrent-steal stress (`jit_threads::concurrent_fiber_steal_stress` — 16 fibers
  × 2 rounds × 4 racing workers, every second resume a guaranteed cross-thread migration of a
  saved stack, schedule-invariant sum). The former staged-divergence pin flipped into the
  migration differential (`fiber_suspended_on_root_migrates_to_spawned_vcpu`). Honest residual
  (unchanged from the stated decision): fuzzing detects, it does not prove.

### 6. Quota, accounting, compatibility

- **Quota (§15):** `max_fibers` moves from per-vCPU to **per-run** (the shared registry's slot
  count); the spawn-bomb ceiling still trips a `FiberFault`. `live_frames` accounting (the
  per-vCPU fiber-frame total) is computed over the chain the worker is actually running, which
  is unchanged.
- **Backward-compatible / additive:** a guest that never resumes a *foreign* fiber sees
  identical behavior; single-threaded runs are untouched (a lone vCPU owns everything, never
  steals). Migration is opt-in by the guest's own scheduler choosing to resume a handle on a
  different worker.

### 7. Test plan

- Ownership CAS + ABA: **loom + deterministic** (step 3a, done).
- Shared registry claim/recycle: unit tests; loom if the slot table is made lock-free.
- **Interp cross-vCPU resume:** a guest *stackful* work-stealing scheduler (Demo 3 — `mn_sched`
  re-pointed at a shared handle pool) with an interleaving-invariant total, proving a fiber
  created on one worker runs to completion on another. Safe, exhaustively schedulable by the
  interp oracle (`explore_all`).
- **Differential interp↔JIT** on that program — the JIT asm seam validated against the safe
  interp semantics.
- Stress (many workers, many migrations) + the documented expert-review of the asm seam.

### 8. Staging (each its own reviewed slice)

1. **3a — ownership protocol (loom-verified).** ✅ Done.
2. **3b-i — interp shared registry + cross-vCPU resume.** ✅ Done (see §4 above): safe-Rust
   registry, unified handle namespace, racing-resume arbiter, explorer/DPOR integration.
3. **3b-ii — JIT shared registry, affinity preserved.** ✅ Done (see §5 above): domain-shared
   `SharedFiberTable` with the `Ownership` word live, unified multi-vCPU namespace + per-domain
   quota + masking; slot *recycling*/generation-carrying handles deferred (needs the interp
   explorer's fiber-return DPOR access; the generation already ticks under the hood).
4. **3c — JIT cross-thread asm resume.** ✅ Done (see §5): `claim` + `suspend_to_pool` live,
   post-switch `CURRENT_RT` re-reads, the five-layer empirical net delivered and green.
5. **Demo 3 — guest stackful work-stealing**, differential interp ≡ JIT. ✅ Done
   (`demos/steal_fibers` — see the demo roadmap above). **The D57 track is complete.**
