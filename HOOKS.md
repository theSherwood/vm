# HOOKS.md — memory-access instrumentation hooks

Design + implementation tracker for **embedder hooks around basic ops** (memory reads/writes
first). Companion to `DESIGN.md` (§3 parity invariant, §4 memory model, §15 monitoring, §19
debugging). Status lives at the bottom.

## 1. Problem & requirements

Consumers want to observe (and possibly veto) a guest's memory accesses — e.g. a memory-safety
validator, or an educational platform that estimates cache misses / page faults from the access
trace and scores student programs on it.

Hard requirement: **zero performance cost for programs that do not opt in.** Given the project's
posture (small trustworthy core; the confinement paths are the most sensitive code in the tree),
"zero" should be *structural* — provable by "nothing changed" — not a benchmark argument.

## 2. Design: instrument the module, not the engines

A hooked run executes a **rewritten module**: an IR-to-IR pass
(`svm_opt::instrument::instrument_mem_hooks`) inserts, ahead of every guest memory op, a
`cap.call` to an embedder-bound hook capability carrying the event kind and access coordinates.
The engines are untouched — an instrumented module is an ordinary module.

Why this shape won over the alternatives:

- **Zero-cost is exact.** A program that doesn't opt in runs a byte-for-byte unchanged module on
  byte-for-byte unchanged engines. There is no `Option<hooks>` branch in the bytecode `match`
  loop, no second codegen mode in the JIT's `mask_addr`, nothing new in `Mem::load_scalar`'s
  fast path.
- **Parity is inherited, not rebuilt.** The §3 invariant (tree-walker == bytecode == JIT) applies
  to the instrumented module like any other, so the *event stream* is backend-identical by
  construction. The JIT's bounds-elision drops only the check, never the access, so elision
  cannot drop events; bulk ops are one IR op on every backend, so span events match too.
- **The JIT works on day one** — and is the fastest hooked configuration — without touching the
  §4 confinement lowering.
- **Veto is free.** The hook is a host capability handler; `Err(Trap)` aborts the run with
  ordinary, backend-identical cap-trap semantics.
- **Untrusted for escape.** The pass follows the `specialize_module` precedent: rewrite, then
  **re-verify** (fail-closed). A pass bug is a clean verify error, never an escape.

Approaches considered and rejected (see the review history for the full survey): runtime
`Option<Hooks>` branches in the hot paths (a permanent ~0.5–3% interpreter tax, and unacceptable
on JIT array kernels); monomorphizing the interpreters over a `Hooks` type parameter (infects
`Vm`/scheduler/fibers, doubles interpreter codegen, makes the oracle generic); a JIT trampoline
mode at the `mask_addr` sites (a second mode through the most sensitive code, forever);
PROT_NONE/fault-handler tricks (page-granular, ~µs/event, entangles hook faults with
trap-confinement); extending `Inspector` watchpoints (stop/resume tops out orders of magnitude
below tracing).

## 3. Event contract

Hooks fire **pre-access, pre-confinement-check**: the final event of a faulting run is the
*attempted* faulting access, and the trace prefix is backend-identical.

Event kinds (the `op` immediate of the inserted `cap.call`; constants in
`svm_opt::instrument::mem_hook_op`, decoded to `svm_run::MemEvent`):

| kind | op | args |
|---|---|---|
| `Load` (incl. v128, width 16) | 0 | `[effective_addr, width]` |
| `Store` (incl. v128) | 1 | `[effective_addr, width]` |
| `AtomicLoad` | 2 | `[effective_addr, width]` |
| `AtomicStore` | 3 | `[effective_addr, width]` |
| `AtomicRmw` (one event, not load+store) | 4 | `[effective_addr, width]` |
| `AtomicCmpxchg` (fires whether or not it swaps) | 5 | `[effective_addr, width]` |
| `Copy` (`mem.copy` **and** `mem.move`) | 6 | `[dst, src, len]` |
| `Fill` (`mem.fill`) | 7 | `[dst, len]` |

- `effective_addr` = base operand + immediate offset, materialized as a wrapping i64 add — the
  same address fold both backends confine.
- Bulk ops are **one event per op** carrying the span; consumers (e.g. a cache model) expand
  spans themselves. This is backend-identical by construction; per-line synthesis inside the VM
  would push expansion policy into the TCB.
- Hooks **observe and may veto** (`Err(Trap)`); they cannot rewrite values or addresses.
  Override semantics would be a fourth source of behavior every backend must replicate
  bit-exactly — out of scope until a concrete consumer demonstrates the need.

**Deliberately not reported** (runtime-internal or host-side traffic, not guest data ops): futex
`atomic.wait`/`notify` word touches; `setjmp`/`longjmp` jmp_buf traffic; `gc.roots` scans;
`cap.self.resolve`/`label` name/label buffers; host-side `GuestMem` access from other capability
handlers; and accesses a frontend's SSA promotion removed before the IR existed (§3d — the trace
is of the post-promotion module; a consumer wanting source-faithful traces disables promotion in
the frontend, the §19 trade).

Known semantic shifts of a hooked run (per-module, not per-backend): the instrumented module
executes more instructions, so **fuel consumption and `Inspector` clock coordinates differ** from
the pristine module (`MemHookStats::inserted_insts` lets an embedder scale `Limits::fuel`), and
`debug_info` is dropped (its `(func, block, inst)` positions would be stale).

## 4. Implementation map

- **Pass**: `crates/svm-opt/src/instrument.rs` — `instrument_mem_hooks(&Module, MemHookSpec)
  -> (Module, MemHookStats)`. Pure, `no_std`, exhaustive block-local renumbering via the
  operand remapper (`map_operands`/`map_term_operands`). `MemHookSpec { type_id, handle }` keeps
  svm-opt free of svm-interp; svm-run supplies `iface::HOST_FN`.
- **Embedder API**: `crates/svm-run/src/lib.rs` — `Instance::with_mem_hooks(make)` instruments,
  re-verifies (fail-closed), and stores the handler factory; `MemEvent` / `MemHookFn` are the
  public surface. The factory builds a fresh handler per host (`run_diff` grants two hosts);
  shared consumer state goes behind an `Arc` in the closure. `Instance::mem_hook_stats()` exposes
  the pass's inserted-op count for fuel scaling.
- **C ABI**: `crates/svm-capi` — `svm_instance_with_mem_hooks(instance, hook, ctx)` consumes the
  instance and returns a hooked one; the callback gets a flattened `SvmMemEvent { kind, addr, src,
  size }` (the `SVM_MEM_*` kinds) and returns non-zero to veto. Trampolines into
  `Instance::with_mem_hooks` exactly as `svm_imports_provide_host_fn` does for host-fns. Declared
  in `include/svm.h`.
- **Handle binding**: `cap.call` needs a handle constant at instrument time. Grants are
  deterministic, so `with_mem_hooks` discovers the value with a scratch first-grant on a fresh
  `Host`, bakes it, and `grant_caps` grants the hook **first** on every run's fresh host
  (asserted fail-closed). Covers `run`/`run_with_caps`/`run_diff`/`call("_start")`/`Session`.
  Bare-kernel exports (`run_kernel_diff`) run hostless on the JIT and are rejected for hooked
  instances with a clear error.
- **Tests**:
  - `crates/svm-opt/src/instrument.rs` unit tests — renumbering + re-verify; pristine ==
    instrumented result with the expected trace (reference interpreter); an
    exhaustiveness gate cross-checking the hooked set against `Inst::effects()` so a future
    guest memory op cannot silently go untraced.
  - `crates/svm/tests/mem_hooks_diff.rs` — the three-backend gate: identical event streams
    (incl. v128 through the bytecode engine's `Op::Eval` fallback), unperturbed outcomes,
    faulting trace ends at the attempted access, veto aborts identically everywhere, and a
    multi-vCPU (`thread.spawn`) guest is observed without crashing (§6 findings).
  - `crates/svm-capi/src/abi_tests.rs` — the C ABI observe + veto over all three backends.
- **Worked example**: `crates/svm-run/examples/mem_hooks_cache_model.rs` — the educational use
  case made concrete: a direct-mapped cache + first-touch page-fault model driven by the event
  stream, scoring a cache-friendly vs a cache-hostile guest (identical 1024 loads, ~7.8× cycle
  estimate gap). `cargo run -p svm-run --release --example mem_hooks_cache_model`.

## 5. Zero-cost accounting

Nothing under `crates/svm-interp`, `crates/svm-jit`, `crates/svm-mask`, or `crates/svm-mem`
changed for this feature — the un-opted path is the same machine code as before, by
construction. The only touched crates are `svm-opt` (a new, never-called-by-default pass),
`svm-run` (a `None` hooks field consulted at grant time, off the per-op path), and tests.
Benchmark A/B against `bench/baseline.txt` is still worth running on any commit that later
touches an engine file; for this change the diff itself is the proof.

Hooked-run cost is **measured** by the overhead probe (`bench/`, `cargo run --release --bin
hooks`): a counting hook (one relaxed `fetch_add` per event) on the store+load mem kernel,
subtraction-isolated, min-of-5. First recorded numbers (2026-07-16, dev container — absolute ns
are machine-dependent; watch the trend, not the value):

| backend | pristine/iter | hooked/iter | overhead/event | hooked events/s |
|---|---|---|---|---|
| TreeWalk | 121.6 ns | 267.0 ns | 72.7 ns | 7.5 M/s |
| Bytecode | 109.1 ns | 199.3 ns | 45.1 ns | 10.0 M/s |
| Jit | 0.3 ns | 102.5 ns | 51.1 ns | 19.5 M/s |

Right in the design's estimated band, and the JIT is the fastest hooked configuration as
predicted (its per-event cost is the host-call boundary; the consumer's own handler is on top).
Adequate for scoring a student program (≤10⁹ accesses in seconds-to-minutes); not aimed at
>10⁷–10⁸ events/s tracing — that remains the P4 trigger.

## 6. Status & follow-ups

- [x] P0 — event vocabulary + firing contract (this doc, §3).
- [x] P1 — instrumentation pass + re-verify + unit tests (`svm-opt`).
- [x] P2 — `Instance::with_mem_hooks` + deterministic handle grant (`svm-run`).
- [x] P3 — three-backend trace parity gate (`crates/svm/tests/mem_hooks_diff.rs`).
- [x] C ABI surface (`svm-capi`): `svm_instance_with_mem_hooks(instance, hook, ctx)` +
      `SvmMemEvent`/`SvmMemHook` in `include/svm.h`, observe + veto exercised in `abi_tests.rs`.
- [x] Published hooked-run overhead number in the benchmark harness — `bench/`'s `hooks` bin
      (`cargo run --release --bin hooks`), hooked vs pristine on all three backends through the
      real `Instance::with_mem_hooks` path; first numbers in §5. `Instance::mem_hook_stats()`
      now exposes the pass's inserted-op count for fuel scaling.
- [x] Multi-vCPU hooked runs **investigated** (finding, not a new mechanism): a `thread.spawn`
      guest runs hooked without crashing, and a spawned vCPU's accesses *are* observed. The run's
      powerbox `Host` is shared across vCPUs via `Arc<Mutex<Host>>` (svm-interp lib.rs ~1881), so
      the single hook handler is invoked **serialized under the host lock** — no data race, and a
      handler capturing plain state is sound. The one caveat is that **cross-vCPU event order is
      schedule-dependent** (within a vCPU it is ordered; `join`/atomics impose happens-before). So
      hooks are *supported* under threads; a consumer needing per-vCPU attribution or a
      deterministic order is the only reason to add a vCPU id to the event — deferred until asked.
      Pinned by the multi-vCPU case in `mem_hooks_diff.rs`.
- [ ] **P4, only on demonstrated need** (>10⁷ events/s native tracing, or "must observe the
      pristine module's fuel behavior"): native interpreter seam — fold a hooks flag into `Mem`'s
      existing `prot_dirty` gate so the scalar fast paths stay instruction-identical and hooked
      runs take the documented-identical cold path; tree-walker firing joins the existing debug
      seam; the bytecode engine compiles `Op::*Hooked` variants (including the `Op::Eval`
      fallback). A JIT trampoline in `mask_addr` stays rejected.
