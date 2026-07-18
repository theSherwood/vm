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
pointers, so by default svm reports candidate root words as raw data and lets the guest
decide what is a pointer (`mask = ~0`, §3). For a guest that tags pointers in the **high
byte** (`(tag << 56) | offset`), `gc.roots` takes an optional **payload mask** — constrained
to clear only the top byte — that strips the tag to the bare offset before the range test, so
tagged roots are recovered without the guest re-scanning (§3).

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
helper wrapping the futex barrier; it is a convenience, not a mechanism — and it is **pure
guest code over existing ops** (`i32.atomic.load.acquire` / `store.release` /
`i32.atomic.wait` / `atomic.notify`), so svm ships it only as a **tested reference**, not an
API. svm can wrap only the collector↔vCPU barrier; the safepoint poll + `suspend` (step 2)
lives in the guest scheduler. svm exposes **no vCPU-count intrinsic**, so the guest passes `N`;
it can, however, read the **current vCPU's id** ambiently via `vcpu.tls.get` (§12 — the per-vCPU
TLS register, seeded to a dense id), which is what per-CPU GC state (mark stacks, allocator
caches) indexes by, correct even after a fiber migrates between vCPUs.

### 2.1. Reference barrier (the optional `quiesce` helper)

The reference below is exercised end-to-end on both backends in
`crates/svm/tests/gc_quiesce.rs` (parametric in `N`, run for N=2 and N=4) — one STW cycle; a
long-lived guest wraps the mutator body in its scheduler loop and re-arms `EPOCH` per cycle.
Window slots are i32 at fixed offsets; each flag has a **single writer**, so no atomic
read-modify-write is needed. `EPOCH`/`RELEASE`/`STOPPED`/`VIOLATION` are scalars;
`parked[i]`/`work[i]` are per-mutator.

```
// mutator vCPU i (the guest scheduler's safepoint + park path):
loop {                                 // bounded work; the loop top is a safepoint
  // … run guest work; work[i]++ …
  if load.acquire(STOPPED) == 1: store.release(VIOLATION, 1);   // must never fire
  if load.acquire(EPOCH) != 0 { break; }                        // a stop was requested
}
store.release(parked[i], 1);           // publish "parked"
notify(parked[i], 1);                  // wake the collector's wait
while load.acquire(RELEASE) != 1:      // block until released
  wait(RELEASE, <last value seen != 1>, timeout);
// resumed → return (or, in a long-lived guest, loop back to the work phase)

// collector vCPU (after spawning the N-1 mutators):
store.release(EPOCH, 1); notify(EPOCH, N);         // request stop, wake early waiters
for i in mutators:                                 // barrier: parked == N-1
  while load.acquire(parked[i]) != 1: wait(parked[i], 0, timeout);
store.release(STOPPED, 1);                          // — world is stopped —
// … call gc.roots(…) (§3), mark, sweep …
store.release(STOPPED, 0);
store.release(RELEASE, 1); notify(RELEASE, N);     // resume everyone
join(mutators);
```

Soundness rests on the §2 invariant: a mutator only reaches its park path *after* its current
fiber suspended, so `parked == N-1` ⟹ no fiber is running ⟹ every fiber is scannable. The
test additionally proves **mutual exclusion**: a mutator sets `VIOLATION` if it ever observes
`STOPPED == 1`, and because all mutators are blocked in `wait(RELEASE, …)` before the collector
raises `STOPPED`, `VIOLATION` stays 0.

## 3. The one new svm primitive: range-filtered root enumeration

An **ambient introspection op** (not a capability — see §3.0) the GC calls during STW. It
writes the candidate words into a guest-provided buffer and returns the total count:

```
gc.roots(heap_lo, heap_hi, mask, buf, cap) -> count   // count = total candidates found
//   masks each scanned word (m = w & mask), then writes min(count, cap) distinct masked
//   words, ascending, as u64 at byte offset `buf`; count > cap ⇒ retry with a larger buffer
```

`heap_lo`/`heap_hi` are **window offsets** bounding the guest heap; svm walks every
fiber's live control-stack extent (incl. the saved-register block the switch routine
writes on suspend, §3d) plus the caller's own live frames (§3.1). Each scanned word `w` is
**masked** (`m = w & mask`), and the masked value `m` is what is range-tested and emitted; svm
returns the deduplicated `m` that fall inside `[heap_lo, heap_hi)`. `mask = ~0` is the untagged
case (emit raw words); a guest tagging pointers in the high byte passes `0x00FF_FFFF_FFFF_FFFF`
to strip the tag and recover the bare offset. **`mask` is constrained to clear only the top
byte** (`mask | 0xFF00_0000_0000_0000 == ~0`); see property 1.

### 3.0. Why ambient, not a capability (decision)

The earlier draft proposed a capability-gated handle. We chose an **ambient IR op** instead —
the same family as `cont.*`/`suspend`, and authority-neutral like `cap.self` reflection:

- **It conveys ~no authority.** Every word it returns is an in-window value the guest's own
  heap already encodes; out-of-window words (host return addresses, frame pointers, host
  pointers) are filtered *inside* svm and never returned (§3, property 1). It cannot read
  out-of-window memory and writes only to a guest-provided, masked buffer — **zero
  escape-TCB**, nothing to gate.
- **svm treats a domain as one trust principal.** Capabilities gate the inter-domain / host
  boundary, not code *within* a domain. Gating `gc.roots` would protect a boundary that does
  not exist intra-domain.
- **Mechanism reality.** Reaching the fiber registry requires special-casing in the execution
  loop either way (a generic `HostFn` capability handler cannot see fibers), so a real
  capability would add `Binding`/grant/handle-validation plumbing for no security gain.

The one honest caveat: unlike `cap.self`, `gc.roots` does read *control-stack* words the guest
cannot otherwise name — but filtered to the guest's own heap range (and the payload `mask` is
top-byte-only so it cannot fold a host word into that range — property 1), so no host/ASLR leak
and no cross-principal boundary is crossed.

### Required properties

1. **Range-filtered (and mask-constrained against host leaks).** Return only in-window
   *masked* candidate words. Out-of-window words — host JIT return addresses (code arena),
   saved frame pointers, host pointers — are filtered **inside svm** and never cross the
   boundary. No host ASLR/layout leaks. This is what keeps the control-stack *read* inside
   svm's TCB and makes option-B ("let the GC read control stacks") safe.
   The payload `mask` must not weaken this: an unconstrained mask could fold a host pointer
   *into* the window (e.g. keep only low bits, so a return address's low 24 bits land in
   `[heap_lo, heap_hi)`), leaking host-address bits past the filter. So svm **constrains the
   mask to clear only the top byte** — the low 56 bits must be all-ones,
   `mask | 0xFF00_0000_0000_0000 == ~0`. A canonical host pointer (`< 2^56`, true for the
   user-space VAs the high-byte tag scheme assumes) is then never reduced → it stays large and
   is excluded by the range filter. The constraint is enforced **statically by the verifier**
   for a constant mask and **defensively at runtime** (a trap) on both backends for any mask.
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

The caller is covered because `gc.roots` is a **call-clobbering** control op (like
`cont.resume`/`suspend`): when it executes, the calling (collector) fiber's live-across
roots are already spilled to its control stack (JIT) / present in its current frames
(interp). svm walks the caller from the `gc.roots` site down, exactly like a parked fiber.
So **the collector never reads its own out-of-band control stack** (it cannot) — svm reads
it on the collector's behalf, and *asking is what made it safe to read*: the op is the
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

The newer tiers keep the same shape (BROWSER.md): the **bytecode engine** realizes `gc.roots` on
its own tier — it scans the whole vCPU continuation (active frames, resume-chain ancestors, parked
fibers, suspended coroutines; `svm-interp/src/bytecode.rs`), and a module combining `gc.roots` with
`thread.spawn` falls back to the reference interp, since only the calling vCPU's continuation is
scannable there. The **wasm-JIT** tier does not realize it at all: `gc.roots` is outside its v1
subset, so `gc.roots`-bearing functions fail closed onto the interpreter (natively the op thunks
into a runtime stack-walk; on wasm even a thunk cannot see JITted locals).

### 3.3. Soundness preconditions (caller obligations svm cannot cheaply enforce)

- `gc.roots` is sound **only under STW** — svm does the range-check; the guest
  guarantees no concurrent mutation. The cheap sanity assert is enforced on the JIT:
  the scan refuses (`FiberFault`) if any *other* vCPU of the domain currently holds a
  RUNNING fiber (`svm-jit/src/fiber_rt.rs`); the interp scans the shared registry
  without it.
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

1. **This RFC** (`GC.md`) — the contract above. No code-behavior change. *(done)*
2. Reserve `ref` `ValType` — cheap, uncontroversial, no runtime effect. *(done — opaque i64,
   threaded through ir/text/encode/jit/interp; round-trip + interp/JIT identity tests.)*
3. `gc.roots` ambient range-filtered enumeration op. *(interp done — registry + caller-frame
   walk, range-filter, dedup, buffer write; functional + round-trip tests.)*
4. **`gc.roots` on the JIT** (the intricate, unsafe piece). *(done — a conservative native-stack
   walk.)* New `svm-fiber` accessors expose a parked fiber's `[ctx, top)` saved extent and a
   running fiber's `[usable_low, top)` superset; a baked thunk reads `CURRENT_RT`'s
   `SharedFiberTable` and scans raw control-stack words, filtering to `[heap_lo, heap_hi)` and
   writing the deduped result to the (confined) guest buffer. **Spilled-only contract:** every
   region scanned has its roots already flushed to memory — parked fibers (the suspend spilled
   their registers), running resume-chain ancestors + the calling fiber (whole-stack superset),
   and the root computation's OS-stack frames `[root_low, root_entry_sp)`. Roots a caller holds
   *only* in unspilled callee-saved registers of its own frame were a real gap — the Tail ABI
   *preserves* those registers across the call, so the thunk boundary did **not** force the caller
   to spill them (audited and A/B-demonstrated: 13 of 17 roots reported without the fix). Closed by
   construction: the JIT now reaches the scan only through a **register-flush trampoline**
   (`fiber_rt::svm_gc_roots_flush`, one naked variant per target) that spills every callee-saved
   register onto the scanned stack before the walk runs (`crates/svm-jit/GC_ROOTS_AUDIT.md`; A/B
   test in `crates/svm/tests/gc_roots.rs`). Scanning a running fiber's / another vCPU's stack assumes
   a stop-the-world safepoint, exactly as the interpreter scans the shared registry — and the JIT
   enforces the §3.3 sanity assert (refuses, `FiberFault`, if another vCPU holds a RUNNING fiber).
   Backend-uniform
   *semantics* per §3.2 (a sound superset of the live roots, not bit-identical candidate sets).
5. **`gc.roots` tagged-pointer payload mask** — a 5th operand `mask` (top-byte-strip only) so a guest
   tagging pointers in the high byte recovers bare offsets; `mask = ~0` is the untagged default
   (§1, §3 property 1). *(done — threaded through ir/encode/text/verify/interp/jit/llvm; verifier +
   runtime reject a fold-down mask.)*
6. **Reference STW `quiesce` barrier** (§2.1) — pure-guest futex barrier, shipped only as a both-backends
   test (`crates/svm/tests/gc_quiesce.rs`), not an svm API. *(done.)*

No world-stop primitive, no register capture, no stack-map work.

## 8. What is explicitly out of scope / rejected

- Preemptive any-PC stop-the-world with `mcontext` register capture. Unnecessary under
  the cooperative model (§2) and would add escape-TCB.
- Any svm object model, allocator, write barrier, or moving/evacuating collector.
- Exposing raw control-stack views or word locations to the guest (§3 keeps the read
  inside svm and returns values only).
