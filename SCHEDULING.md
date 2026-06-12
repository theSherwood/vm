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
| Migration across cores | **no** — fibers are thread-affine | **yes** — moving a struct is a pointer hand-off |
| Load balancing | none (thread-per-core) | dynamic (steal from busy workers) |
| Safety of migration | n/a (no migration) | safe by construction (it's just data) |
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

## Why fibers are thread-affine today

Each OS-thread vCPU builds its **own** `FiberRuntime` (JIT: `CURRENT_RT`
thread-local + a per-runtime `fibers` table; interp: a per-`VCpu` `fibers: Vec`).
A fiber handle indexes the *creating* thread's table, so a fiber can only be
resumed on the worker that created it. This is deliberate and good: zero locking on
the fiber table, cache locality — the thread-per-core architecture chosen on
purpose by glommio/seastar, not merely "the easy option."

## Proposed evolution: migratable fibers → stackful work-stealing

**Status: Proposed (D57), not adopted.** The ideal outcome is work-stealing M:N
over *stackful* fibers — Go-class scheduling for **arbitrary unmodified compiled
code** inside a confinement sandbox, strictly more than either flavor above. It is
**feasible and safely so** — Go is the decade-long, planet-scale existence proof of
stackful work-stealing. But it carries a real, eyes-open cost.

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
invariant lives. The **composition** of (verified protocol) + (real asm switch) +
(per-thread signal recovery) cannot be exhaustively model-checked (asm + signals are
outside loom/TSan) — the *same* residual caveat that already applies to today's
fiber+thread code, exercised harder. Mitigation: heavy stress, the interp↔JIT
differential oracle, and expert review of that seam.

## Demo roadmap

1. **Demo 1 — sharded stackful M:N** *(in progress)*. A guest cooperative scheduler
   over `cont.*`: N `thread.spawn` workers, each round-robining its own pool of
   fiber tasks that yield; a shared atomic aggregate. Proves fibers + threads +
   atomics compose into a real M:N runtime, and establishes the scheduler machinery
   (run queue, resume-until-suspend, park/unpark) the later versions reuse. Honest
   about affinity (tasks pinned per worker).
2. **Demo 2 — stackless work-stealing M:N**. Tasks as guest state machines, a
   Chase-Lev deque over atomics, futex park/unpark — cross-core load balancing with
   **no VM change**. Proves the harder claim and is the natural lead-in to the async
   I/O ring (B).
3. **Migratable-fiber primitive + Demo 3 — stackful work-stealing**. The D57
   evolution: the shared registry + ownership protocol (loom-verified) + Demo 1
   re-pointed at a shared steal pool. Done deliberately, with review, only after 1–2
   establish the baseline and the value is concrete.
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
     replacing the per-thread tables, over a Chase-Lev steal deque) and the cross-thread
     asm resume (design sketch #3, expert-review gated) are the remaining steps.
     Verifying the dangerous invariants first is the "earn the risk" discipline this
     feature demands.
