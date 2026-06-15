# GC.md — svm ⇄ guest GC support contract (conservative, non-moving)

> Status: draft RFC, converged between the svm and JACL sides. Expect iteration.
> Scope: what svm provides so a **guest** can run its own garbage collector. svm
> implements **no GC and no object model** — only fiber **root enumeration**.
> Cross-refs are to `DESIGN.md` sections (§).

## 1. Model

A guest (the motivating case is JACL) runs a **non-moving, conservative mark-sweep**
collector over a heap it owns in its linear window (grown via the existing `Memory`
capability, §4/§3e). The guest keeps its own block/line maps for object-start and
interior-pointer resolution. Roots are **conservative**: the guest exposes raw C
pointers, so no value is tag-filterable — svm must report candidate root words as
raw data and let the guest decide what is a pointer.

svm contributes exactly one thing the guest cannot do itself: enumerate the roots
that live on **control stacks + saved-register blocks**. Everything else is guest
policy.

### Division of labor

| Owner | Responsibility |
|---|---|
| **Guest** | heap + allocator, mark/sweep, object/line maps, the M:N scheduler, world-stop coordination, scanning its own **in-window data stack + heap** (both already guest-addressable, §3d). |
| **svm** | enumeration of roots on **control stacks + saved-register blocks** — out-of-band, unnameable by guest masking (§3d/§5), so the *only* place the guest cannot reach. |

## 2. Stop-the-world: cooperative, guest-coordinated (no async preemption)

svm provides **no** preemptive, any-PC stop with register capture, and will not —
it would add escape-TCB (§2a) and cuts against the deliberate cooperative-preemption
design (§12: "the VM supplies mechanism, not policy"; the signal/guard machinery is
detect-and-kill, it does not pause-and-introspect). World-stop is therefore built by
the guest from primitives that already exist.

**Coordinate the N vCPUs, not the M fibers.** The GC *is* the scheduler: "stop the
world" = stop resuming user fibers and let running ones reach their safepoints and
`suspend`. A vCPU only re-enters its scheduler loop (where it can park) *after* its
current fiber suspended/returned, at which point that fiber's roots are flushed to
its control stack (`suspend` is call-clobbering, §3b/§6). So:

> **all N vCPUs parked ⟹ no fiber running ⟹ every fiber scannable.**

Quiescing the N threads quiesces the M fibers as a consequence — there is no
per-fiber handshake.

**The handshake is pure guest code** over window atomics + `memory.wait`/`memory.notify`
(§12) — Go-style STW:

1. The collector vCPU bumps a shared atomic `gc_epoch` (release store to a window
   address).
2. Each mutator fiber polls `gc_epoch` at **safepoints the guest compiler inserts**
   (loop back-edges + call sites). On a mismatch it `suspend`s to its vCPU's scheduler.
3. Each vCPU's scheduler, seeing `gc_epoch` changed, **parks** instead of resuming
   more work (`parked++`; `notify(&parked)`; `wait` until done).
4. The collector waits for `parked == N-1`, scans (§3), signals done, and `notify`s
   the parked vCPUs to resume.

**No new svm world-stop primitive is required.** At most an optional thin `quiesce`
helper wrapping the futex barrier; it is a convenience, not a mechanism.

## 3. The one new svm primitive: range-filtered root enumeration

A capability-gated op the trusted GC calls during STW:

```
gc.roots(heap_lo, heap_hi) -> buffer<u64>      // candidate root values
// (or roots_begin / roots_next, if streaming is preferred)
```

`heap_lo`/`heap_hi` are **window offsets** bounding the guest heap; svm walks every
fiber's live control-stack extent (incl. the saved-register block the switch routine
writes on suspend, §3d), and returns the words that fall inside `[heap_lo, heap_hi)`.

### Required properties

1. **Range-filtered.** Return only in-window candidate words. Out-of-window words —
   host JIT return addresses (code arena), saved frame pointers, host pointers — are
   filtered **inside svm** and never cross the boundary. No host ASLR/layout leaks.
   This is what keeps the control-stack *read* inside svm's TCB and makes option-B
   ("let the GC read control stacks") safe.
2. **Value-only, deduplicated.** A non-moving collector needs *reachability, not
   positions*. Return a deduped list of candidate values; **never expose a raw
   control-stack view** and never expose word locations. This collapses "enumerate"
   and "read" into one safe op.
3. **Coverage = every non-mutating fiber, including the caller.** See §3.1.
4. **Mechanism, not policy.** svm does only the window range-check; the guest does
   object-start validation, interior-pointer resolution, and marking.
5. **Representation.** `heap_lo`/`heap_hi` and every returned candidate are **window
   offsets** — the guest's confined pointer representation (its "raw C pointers"),
   consistent in and out.

### 3.1. Who gets scanned (the coverage invariant)

> svm scans every fiber **not actively executing guest mutator code** = all **parked**
> fibers **+ the caller of `gc.roots`**. Under STW (all other vCPUs parked) that is
> *every* fiber.

The caller is covered because `gc.roots` is a `cap.call`, which is **call-clobbering**:
when the handler runs, the calling (collector) fiber is frozen at the call site with
all live-across-call roots already spilled to its control stack. svm walks the caller
from the `gc.roots` call frame down, exactly like a parked fiber. So **the collector
never reads its own out-of-band control stack** (it cannot) — svm reads it on the
collector's behalf, and *asking is what made it safe to read*: the call is the
collector's own safepoint.

The collector handles by itself only its **in-window GC structures** (mark stack,
work list, remembered set), which are guest-addressable anyway. There is no stack-root
gap. (Interp realization: the collector's own frames sit in the running vCPU's
`Vec<Frame>` chain and are walked like any other fiber's.)

### 3.2. Backend uniformity is *semantic*, not bit-exact — and that is sound

`gc.roots` is defined **semantically** ("the in-range candidate words of every
non-mutating fiber") and **realized differently per backend** (§3d two-stack model):

- **JIT:** raw native control-stack words in `sp..base`, incl. spills, padding, saved
  registers.
- **Interp:** the typed `Value`s in each live frame's value vector (no machine
  registers exist).

Therefore the candidate set **legitimately differs across backends** — the JIT carries
more conservative false-positives than the interp. For a **non-moving** collector this
is sound *over-approximation*: a falsely-retained object is never freed and never moves,
so a correct program cannot observe the difference.

**Consequence to state loudly:** GC heap occupancy is **NOT** part of the bit-exact
interp↔JIT differential oracle — only program *output* is. Do not later try to make the
two backends retain identically; that would be "fixing" a non-bug and is impossible in
general (the interp has no spill words to match).

### 3.3. Soundness preconditions (caller obligations svm cannot cheaply enforce)

- `gc.roots` is sound **only under STW** — svm does the range-check; the guest
  guarantees no concurrent mutation. Optional cheap sanity assert: svm may refuse if
  any *other* vCPU of the domain currently holds a RUNNING fiber.
- It is **authority-TCB, not escape-TCB** (§2a): only the powerbox holder can call it;
  a forged handle is inert (§3c). It cannot break escape-freedom.

## 4. Already guaranteed — document, no work

- **Heap stability.** The window base/mask are per-domain instantiation constants
  (§3d); JIT code-arena compaction (§22) compacts the *code arena*, never the guest
  window. The GC may rely on stable pointers under a running guest.
- **Frame pointers.** Already preserved (mandatory for Cranelift x64 tail calls). The
  frame-walk forward-compat ask needs no action.

## 5. Guest-side obligations (part of the contract)

1. **Safepoint polls** at loop back-edges + call sites — no un-polled tight loops, or
   STW stalls (svm offers no async escape hatch, by design). Piggyback the `gc_epoch`
   poll on the same sites used for svm's epoch/kill-path check (§5) — one poll, not two.
2. **Blocking host ops use async-form capabilities** (which park the fiber → scannable).
   A fiber must never sit in a long synchronous host `cap.call` during STW.
3. **No reentrant guest execution while stopped** (a host capability that calls back
   into guest code, §12 reentrancy, must not run a mutator during STW).
4. The guest scans its own **data stack + heap**; it relies on svm only for
   control-stack / saved-register roots.

## 6. Forward-compat: precise GC later (cheap to reserve, expensive to retrofit)

- **Do now:** reserve a distinct **`ref` `ValType`** — opaque 64-bit, lowers as `i64`;
  threaded through `ir`/`text`/`encode`/`verify`/`interp`/`jit`; round-trip fuzzers
  accept it; **zero codegen/perf delta** (it is an `i64` to the backend). This prevents
  a format break when precise stack maps arrive.
- **Defer until precise GC is committed:** Cranelift **stack maps** + per-PC
  value-location metadata + a GC-safe call ABI. Frame pointers already exist;
  conservative GC needs none of these. Drivers to revisit: deterministic heap state for
  the model checker, or evacuation/defrag for long-lived heaps.

## 7. Build order

1. **This RFC** (`GC.md`) — the contract above. No code-behavior change.
2. Reserve `ref` `ValType` — cheap, uncontroversial, no runtime effect.
3. `gc.roots` range-filtered enumeration capability (interp + JIT, backend-uniform
   *semantics* per §3.2). The largest piece.

No world-stop primitive, no register capture, no stack-map work.

## 8. What is explicitly out of scope / rejected

- Preemptive any-PC stop-the-world with `mcontext` register capture. Unnecessary under
  the cooperative model (§2) and would add escape-TCB.
- Any svm object model, allocator, write barrier, or moving/evacuating collector.
- Exposing raw control-stack views or word locations to the guest (§3 keeps the read
  inside svm and returns values only).
