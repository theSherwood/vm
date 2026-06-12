# Handoff — Guest-driven JIT (`Jit` capability, Model A + B2)

Pick-up notes for a fresh session. Written 2026-06-12. Branch: **`main`** (this work commits
straight to `main`; remote `theSherwood/vm`). Everything below is committed and the full
workspace suite is green (`cargo test --workspace` → 65 test binaries pass).

The **design + status doc** is `JIT.md` (read it first — this file is the operational
pick-up companion). DESIGN.md section refs (§2a, §3c, §4, §8, §13) are the VM's contracts.

---

## 1. What this is (30 seconds)

Guest code (e.g. an interpreter running *inside* the sandbox) can, at runtime: build serialized
SVM IR in its own window, hand the blob to a new **`Jit` capability** (iface 11) across the
`cap.call` boundary, and have the host **verify** it (the same `decode_module` + `verify_module`
gate every module passes) and **Cranelift-compile** it into the *same domain* (same window, same
powerbox — a module is not an isolation unit, DESIGN §8). The compiled code is then reached by
`invoke` (a trampoline call) or, once `install`ed into the `call_indirect` table, as a first-class
funcref. "JIT inside the sandbox," with verification — not isolation — as the trust boundary.

**It is built and working on both backends, differentially identical.** All four cross-call
directions work: old→old, old→new, new→old, new→new. The reference **interpreter** is the
correctness oracle; the **Cranelift JIT** is the production backend; every `Jit` operation is
asserted to produce identical results / errnos / traps / final-memory on both
(`crates/svm/tests/jit_cap.rs`).

---

## 2. The `Jit` capability surface (iface 11)

`cap.call 11 <op> …` on a granted `Jit` domain handle:

| op | signature | meaning |
| --- | --- | --- |
| 0 `compile` | `(ptr, len) -> code_handle \| -errno` | borrow blob from window, decode+verify+precondition gate, compile into the domain; mint a `CompiledCode` handle (iface 12). Fail-closed: nothing installed on any failure. |
| 1 `invoke` | `(code_handle, args…) -> results` | run the unit's entry (`funcs[0]`) over the **live window**; raw i64-slot ABI; traps **terminal for the domain**. |
| 2 `release` | `(code_handle) -> 0 \| -errno` | revoke the handle (generation bump). Code/slots not freed. |
| 3 `install` | `(code_handle) -> slot \| -errno` | write the unit into the `call_indirect` table's next reserved padding slot; returns a funcref index. `-ENOSPC` if full. |
| 4 `uninstall` | `(slot) -> 0 \| -errno` | clear an installed slot (reusable; stale calls trap). Guards real-function slots (`-EINVAL`). |

**The security hinge** (`svm_run::jit_blob_validator`, injected into the `Host` as a
`JitValidator` fn so `svm-interp` keeps its tiny dep set and *both backends run the identical
gate*): `decode_module` → `verify_module` → **memory-match precondition** (the blob's declared
memory must equal the parent window — else the JIT would mask to a different size; an escape) →
reject data segments → reject §12 concurrency ops inside a *submitted unit* (`Func::uses_concurrency`
— a JIT'd blob stays single-threaded; the **parent** guest may now be multi-threaded, §6 #2).
All failures `-EINVAL`. A per-domain **compile quota** (`-ENOMEM`, `Host::jit_compile` /
`set_jit_quota`) bounds a looping guest.

C surface (`frontend/chibicc`, `<svm.h>`): `__vm_jit_compile/invoke2/install/uninstall/release`.
**ABI gotcha:** this frontend threads the data-stack pointer as every function's hidden first
param, so a unit called via a **C function pointer** must be shaped `(i64 sp, A, B) -> T`
(leaf ignores `sp`); `__vm_jit_invoke2` is the *raw* `(i64,i64)->(i64)` shape with no `sp`.

---

## 3. Architecture & the key idea per backend

**The enabling refactor (Phase 1):** the JIT's old one-shot `compile→run→drop` became a
long-lived `CompiledModule` that owns the `JITModule` across runs
(`crates/svm-jit/src/lib.rs`): `compile()` + `run()` (or `run_raw()` for the re-entrant
capability path) + `define_extra(funcs) -> Vec<DefinedFn>` (incremental compile into the live
module) + `install`/`uninstall`. The one-shot `compile_and_run*` entry points are now thin
wrappers and still back the existing test harness.

**The single structural obstacle** is the **`fn_table` mask**, baked as an `iconst` at every
`call_indirect` site (`fn_table_mask`). Model A sidesteps it (`invoke` is thunk-reached, never in
the table). Model B2 **pre-reserves** the table (`table_reserve_log2` / `grant_jit_with_table`,
*identical on both backends*) so the mask is constant from `t=0` and `install` fills padding
without moving it. Slot indices agree because both fill from the parent's function count.

**Type identity (the thing that makes cross-unit `call_indirect` sound):** an **append-only
intern** registry (`intern_type` / `intern_unit_sigs` / `interned_type_id`), so id-equality ≡
structural equality across every unit a module ever compiles. The JIT checks the baked id; the
interpreter checks structural equality directly — these coincide by construction. This is the
load-bearing Phase-1 groundwork that made B2 cheap.

**Re-entrancy (Phase 2a):** `compile`/`install` mutate the live module *mid-run* while the guest
is suspended in its synchronous `cap.call`. `run_code_raw` keeps **no** Rust reference into the
`CompiledModule` alive across the guarded call, and the capability uses `run_raw(this: *mut
CompiledModule, …)` so a handler can re-derive `&mut *this` safely. `invoke_extra` runs an extra
trampoline over the *live* window under a nested detect-and-kill guard (`run_guarded_range`).

**Cross-module execution in the reference interpreter (slice #1 — the conceptual heart):** the
interp resolved functions by index into one `funcs` array; the JIT uses code pointers. To match
new→old/new→new, each `Frame` gained a `module` tag (0 = the vCPU's own program, `≥ 1` = an
installed unit, `INVOKE_MODULE` = the transient unit a `Jit.invoke` is running). Direct calls stay
in the caller's module; `call_indirect`/`return_call_indirect` dispatch through a **shared, live
`DomainTable`** (atomic slots + the writer-locked installed `units`), `Arc`-shared by every vCPU —
the reference mirror of the JIT's one `fn_table` (§6 #2; this replaced the old per-`VCpu`
`Vec<TableSlot>` that diverged on threaded install).
- **Dispatch is lock-free:** a slot is a single `u64` word read with an `Acquire` atomic load;
  `install`/`uninstall` publish a `Release` store; `units` is append-only behind a writer-only
  `Mutex` with a per-vCPU lock-free local clone (re-synced on a miss). `resolve_module` resolves a
  frame's/slot's module; the type-check borrows (no `Arc` clone). Module 0 dispatch touches neither
  lock nor `units`.
- `Jit.invoke` runs a unit as a nested inline-driven `VCpu` (`VCpu::new_invoke`) that **shares** the
  parent's `Arc<DomainTable>` (no longer a snapshot), so an invoked unit reaches the original program
  (new→old), already-installed units (new→new), **and** units installed *during* its own invocation;
  its own transient unit is `INVOKE_MODULE`, kept out of the shared `units`. The JIT gets the same
  from the live `fn_table` (now atomic `FnEntry` for threaded-install safety).

---

## 4. Where things live

- `crates/svm-jit/src/lib.rs` — `CompiledModule` (`compile`/`run`/`run_raw`/`define_extra`/
  `invoke_extra`/`install`/`install_at`/`uninstall`/`installed_slots`/`extra_fn_count`/`is_running`),
  `DefinedFn`, `intern_*`, `n_real_funcs`, `table_reserve_log2`, and the **atomic `FnEntry`**
  (`AtomicU32` type_id + `AtomicU64` code, release-ordered publish — threaded-install safety, §6 #2).
  The escape-TCB-adjacent crate.
- `crates/svm-interp/src/lib.rs` — `Frame.module`, the **shared `DomainTable`** (atomic slots +
  writer-locked `units`) / `resolve_module` / `dispatch_indirect` / `INVOKE_MODULE`, `VCpu::new`
  (takes the shared `Arc<DomainTable>`) / `new_invoke` (shares it), the eval-loop `Jit` op arms
  (invoke=1, install=3, uninstall=4 in the loop; compile=0/release=2 in generic dispatch
  `Binding::JitDomain`), `Host` Jit state (`grant_jit`/`grant_jit_with_table`/`jit_compile`/
  `jit_unit_*`/`jit_unit_count`/`jit_live_units`/`set_jit_native_ctx`), `iface::JIT`=11 /
  `JIT_CODE`=12, the `JitValidator` seam, `domain_table_tests`.
- `crates/svm-run/src/lib.rs` — `cap_thunk` intercepts iface 11 → `jit_native_op`; **`cap_thunk_locked`**
  (the per-domain `Mutex<Host>` serialized thunk for a concurrent guest, with invoke-release
  re-entrancy) + `jit_invoke_locked`; `jit_blob_validator` (the canonical gate); `grant_jit`;
  `jit_cap_run` (locked path when `uses_concurrency`); `run_powerbox`/`powerbox_compile_run` (CLI,
  same locked branch); `recompact_jit` + `recompact_into` (compaction); `JitSession` (auto-compacting
  REPL driver, owns a boxed `Mutex<Host>`, concurrency-capable).
- `crates/svm-ir/src/lib.rs` — `Func::uses_concurrency`.
- `frontend/chibicc/codegen_ir.c` + `include/svm.h` — the `__vm_jit_*` builtins (JIT_SLOT=28,
  8-handle powerbox).
- `crates/svm-run/demos/jit/` — `jit_demo.c`: a bytecode interpreter that JITs itself, both via
  `invoke` and via `install` + a C funcptr. `cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c`.

---

## 5. Test surface (run these to know you didn't break it)

- `cargo test -p svm --test jit_incremental` — Phase-1 `CompiledModule` / `define_extra` / intern /
  JIT-level `install`. (JIT-only.)
- `cargo test -p svm --test jit_reentry` — mid-run `define_extra`/`invoke_extra` (incl. the
  strongest W^X case: `finalize_definitions` while parent code is live on the stack).
- `cargo test -p svm --test jit_compaction` — §6 code-memory reclaim: the simulated-REPL
  recompaction mechanism (`install_at`/`installed_slots`/`extra_fn_count`), the
  embedder-integrated `recompact_jit` driver (persistent-window REPL transparency + reclaim,
  live invoke-only carry), **and** the auto-compacting `JitSession` (watermark policy, guest-driven
  `cap.call`-end-to-end session).
- `cargo test -p svm --test jit_cap` — **the differential suite** (interp ≡ JIT): compile/invoke,
  all 4 cross-call directions, install/uninstall, **threaded install** (a worker dispatching a
  post-spawn install) **+ threaded compile** (concurrent `Jit.compile` from a worker — §6 #2),
  garbage/memory-mismatch/data/concurrency rejection, quota, traps, a deterministic blob fuzz. This
  is the one that matters most.
- `cargo test -p svm --test c_frontend c_guest_jit_demo` — the C demo end-to-end on both backends.
- Always finish with `cargo test --workspace` + `cargo clippy -p svm-jit -p svm-interp -p svm-run`.

The differential harness is `jit_cap.rs::diff_run` / `diff_run_t(…, table_log2)` — runs the same
guest program + Host setup + blob on interp (`run_capture_reserved_with_host`) and JIT
(`jit_cap_run`) and asserts equal results/traps/final-memory. To add a B2 (install) case, use
`diff_run_t` with a non-zero `table_log2`.

---

## 6. Remaining work (prioritized, with honest scoping)

The user's core asks are **done** (old↔new both directions, install, slot reclaim). What's open:

1. **Code-memory compaction reclaim** — *the* valuable item; **mechanism + embedder integration +
   auto-trigger landed; only polish remains.** Slot reclaim (`uninstall`) shipped, but repeated `compile`s
   still consume the 256 MiB code arena with no per-function free in `cranelift-jit`. Reclaim =
   periodic **whole-module compaction** (recompile the live set into a fresh `CompiledModule`,
   swap). Key constraint: it can **only** run at a *quiescent* point — the guest is suspended
   *inside* the very module being compacted, so the guest can't trigger it mid-`cap.call`. So it
   is **embedder-facing** (a REPL driver between prompts).
   - **Mechanism (no escape-TCB codegen — only replays `compile`/`define_extra`/`install`):**
     `CompiledModule::install_at(slot, code, type_id)` reinstalls a unit at its **exact** old slot
     (a funcref the guest holds keeps resolving across the swap; `install` only fills the *next*
     padding slot, which can't reproduce an `uninstall`-gap history); `installed_slots()` exposes
     the occupied `(slot, code, type_id)`; `extra_fn_count()` is the occupancy proxy a watermark
     watches (restarts near zero in the fresh module — the visible reclaim); `is_running()` is the
     quiescence guard.
   - **Embedder driver:** `svm_run::recompact_jit(base, entry, reserved_log2, table_reserve_log2,
     host, domain, old)` rebuilds the domain's *live* code into a fresh module. It carries every
     unit still reachable — occupying a slot (`installed_slots`) **or** held through a live
     `CompiledCode` handle (`Host::jit_live_units` / `jit_unit_count`) — re-`define_extra`'ing it,
     remapping the `Host` unit→native record (existing handles name `(domain, unit)` not a code
     address, so they keep working), and reproducing occupied slots via `install_at`; a unit
     neither installed nor live-handled is dead and dropped (the reclaim). The handle table needs
     no edit.
   - **Tests (`crates/svm/tests/jit_compaction.rs`):** the pure-mechanism simulated REPL
     (redefine 40× → recompact to 1 live def, exact-slot across a gap) **plus** two
     embedder-integrated cases driving the real `Host` tracking (the
     `jit_compile`/`set_jit_unit_native`/`install` sequence `jit_native_op` runs from a guest
     `cap.call`): a 20-prompt persistent-window REPL is byte-identical with/without compaction
     while occupancy stays bounded, and a live invoke-only unit survives the swap with its
     trampoline remapped. Oracle is compacting-JIT vs non-compacting-JIT (a multi-run interp↔JIT
     differential needs the interp to persist installs *across runs* — the shared `DomainTable` from
     #2 now makes that possible, but the harness still builds a fresh table per `run`; single-run
     correctness stays differential in `jit_cap.rs`).
   - **Auto-trigger:** `svm_run::JitSession` is the persistent REPL driver — owns the long-lived
     `CompiledModule` + carried window, `run_prompt` re-enters once per prompt (prior prompt's low
     bytes seed the next so guest state persists), and **auto-compacts** at a watermark on
     `extra_fn_count()` at the quiescent point *after* a prompt (the only sound place — the guest
     is suspended *inside* the module during a `cap.call`). `seed_window`/`compact`/`occupancy`/
     `compactions` round it out. `jit_session_auto_compacts_transparently` drives a 30-prompt
     guest that `cap.call`-compiles/invokes/releases a fresh unit each prompt (real `cap.call`s
     end-to-end): identical results+window with auto-compaction off vs on, occupancy bounded at the
     watermark.
   - **Byte-accurate watermark — done.** `CompiledModule::extra_byte_count()` sums the actual
     machine-code bytes of every `define_extra`'d function + trampoline (read from
     `ctx.compiled_code().code_buffer().len()` at finalize, before `clear_context`; the dominant term
     in arena consumption, alignment/rodata excluded so it slightly under-counts). `JitSession`'s
     watermark and `occupancy()` are now in **code bytes** (the count-based `extra_fn_count()` stays
     for callers who want it); the `jit_compaction` session tests derive the watermark from a
     one-prompt byte probe so they're robust to per-platform code sizes.
   - **C-level guest REPL demo — done.** `demos/jit/jit_repl.c`: a prompt body that `__vm_jit_compile`s
     a fresh unit, invokes + releases it, and accumulates into a **BSS** global (no `data` segment, so
     it carries across the session's per-prompt window reseed). Driven by `JitSession` in
     `c_frontend::c_guest_jit_repl_compacts` (30 prompts, watermark off vs on): identical results **and**
     stdout transcript while the on-run's byte occupancy stays bounded by the live set — a long C REPL
     that JITs every prompt and never exhausts the arena. (Not a standalone `cargo run` showcase like
     `jit_demo`/`jit_threads` — compaction is an embedder-between-prompts operation, so the embedder
     `JitSession` is the driver; run standalone it executes one prompt.) The `-ENOMEM` byte-cap backstop
     bounds the arena regardless.

2. **Threaded install + install-during-own-invocation + threaded compile** — **all landed, full
   platform parity** (x86-64 Linux/Windows, aarch64 macOS); only a throughput optimization and a C
   demo remain (see the end of this item). The root cause was a *structural mismatch*: the interp modeled the
   dispatch table as per-`VCpu` owned state (a `Vec` rebuilt at `thread.spawn`, snapshotted at
   `Jit.invoke`), while the JIT has one `fn_table` shared by every thread — so a post-spawn (or
   during-own-invocation) install was visible to the JIT but not the interp.
   - **Interp — shared atomic `DomainTable`:** the table is now `Arc`-shared by every vCPU, making
     the two backends structurally isomorphic. The feared "lock on the hot path" was a false
     constraint — each slot is a single packed `u64` word, so dispatch is one **`Acquire` load**
     (free on x86) and `install`/`uninstall` one **`Release` store**, no lock, no torn read (table
     pre-reserved). `units` is append-only behind a writer-only `Mutex`; readers keep a lock-free
     local clone re-synced on a miss, so module-0 dispatch touches neither lock nor `units`, and the
     type-check resolves by borrow (no `Arc` clone). A `Jit.invoke` unit runs as a transient
     `INVOKE_MODULE` kept out of the shared `units` (no collision with an install it performs on
     itself). **Perf:** `svm-bench interp_ci` ~79.7→~80.8 µs best-of (+1.4%, within ~3% noise).
     Unit-tested (`domain_table_tests`).
   - **JIT — atomic `FnEntry`:** `install`/`uninstall`/`install_at` now publish **release-ordered
     atomic** writes (`FnEntry { type_id: AtomicU32, code: AtomicU64 }`, same `#[repr(C)]` layout, so
     `indirect_dispatch` codegen is **byte-identical** — no runtime regression) with `&self`
     (interior mutability, since the running generated code reads the table through raw pointers).
     `code` first then `type_id` (the ready field), so a reader observing the unit's `type_id` sees
     its `code`. The fiber/thread funcref dispatch reads via the atomic accessors.
   - **End-to-end differential (landed):** `jit_cap::threaded_install_agrees_across_backends` — main
     compiles, spawns a worker, then installs + signals readiness via a guest atomic; the worker
     `call_indirect`s the post-spawn install → both backends return 52 (non-flaky 12/12). Compile (the
     only `finalize`) is before the spawn, so install is the lone concurrent table op. The test runs
     on **all `fiber_rt` targets** (x86-64 unix, aarch64 unix/macOS, x86-64 Windows) — full platform
     parity (see below).
   - **Platform parity (no aarch64 gap).** The install's *visibility* rides the **guest's own**
     acquire/release on its ready flag (install stores → `ready` store-release → `ready` load-acquire
     → the worker's dispatch loads), so the worker's `type_id`+`code` loads observe the completed
     install on a weakly-ordered target too — the dispatch's own load order is irrelevant, so **no
     acquire-on-`type_id` codegen change is needed**. The atomic `FnEntry` is for a *different*,
     platform-uniform property: a **racy** guest (a worker dispatching a slot concurrently being
     reinstalled without synchronizing) still reads a *complete* code pointer (old or new — both
     valid `AtomicU64` values), never a torn/half-written pointer (no wild jump → no escape); a racy
     outcome is the guest's own bug and is contained.
   - **Threaded *compile* — spike done; no stop-the-world needed.** A worker calling `Jit.compile`
     while others run hits `finalize_definitions` under live threads. The spike (source analysis of
     cranelift-jit 0.132 + a concurrent stress test, `jit_incremental::
     concurrent_finalize_does_not_disturb_running_code`) settled the W^X question:
     - **Page protection: safe by construction.** `ArenaMemoryProvider::finalize` only `mprotect`s
       *non-finalized* segments (`Segment::finalize` early-returns on `finalized`; the allocate
       `set_rw` resize path skips finalized segments). Executing code always lives on a finalized
       segment, so finalize/allocate **never touch a running page** — no transient W^X, no
       stop-the-world. The stress test (a sibling hammering a finalized leaf through millions of
       calls across 400 `define_extra`/finalize cycles, returning 42, non-flaky) corroborates it.
     - **I-cache cross-core: handled, one macOS caveat.** finalize does `clear_cache` (aarch64:
       `ic ivau` broadcast + `dsb ish` + `isb`) + `pipeline_flush_mt`, which on Linux aarch64 is
       `membarrier(SYNC_CORE)` (broadcasts an `isb` to every core); x86 is coherent (no-op). On
       **aarch64 macOS** `pipeline_flush_mt` is a no-op, so a busy-spinning executing core's own
       `isb` isn't guaranteed — that one target needs the executing thread to context-synchronize
       at a safepoint (a brief quiesce, *not* a global stop-the-world).
   - **Threaded *compile* — MVP landed (a real target language needs it).** The spike's path,
     built: `svm_run::cap_thunk_locked` serializes `cap.call` through a **per-domain `Mutex<Host>`**
     so concurrent compiles are sound — `define_extra`'s `&mut *cm` is exclusive because the lock
     serializes all cap.calls, while *execution stays fully parallel* (the spike's Finding #1: a
     finalize never touches a running page). The confirmed **re-entrant** case ("running units
     compile more") is handled: `Jit.invoke` resolves the unit under the lock, **releases**, then
     trampolines, so an invoked unit may itself compile on the same thread without self-deadlock and
     other threads keep progressing. Every other op is host-side only (Instantiator/fibers re-enter
     via their own runtimes, not `cap_thunk`), so holding the lock across a delegate to the unlocked
     `cap_thunk` is deadlock-free. **`jit_cap_run` engages the locked thunk only when the module uses
     concurrency** (`Func::uses_concurrency`); a single-threaded guest keeps the unlocked
     `cap_thunk` + raw `*mut Host` path **verbatim — zero lock cost**. The guest-facing `cap.call 11`
     iface is **unchanged**, so the serialization is an internal detail swappable later (sharded
     modules for parallel-compile throughput) without breaking guest software. Tests
     `jit_cap::threaded_compile_agrees_across_backends` (main + worker concurrently compile+invoke →
     134) and `threaded_compile_loop_stress_agrees` (≈20 overlapping compiles → 515), both
     differential, non-flaky, on all `fiber_rt` targets. The interp already serialized via its
     `Arc<Mutex<Host>>`, so it needed no change.
   - **Threaded-compile follow-ups (done / resolved):** (a) **CLI wiring — done.** `run_powerbox`
     now routes a concurrent guest through `cap_thunk_locked` + a `Mutex<Host>` (single-threaded keeps
     the fast path); validated by the existing concurrent C demos (`c_guest_work_stealing_demo`,
     `c_guest_thread_safe_malloc`, …), which do concurrent cap.calls and pass on the locked path. (b)
     **No aarch64 `isb` needed — resolved.** The earlier concern was over-cautious: cranelift-jit
     *appends* new functions to fresh arena addresses and never modifies executing code in place, so
     an executing core has no stale prefetch — the cross-modifying-code `isb` is for *in-place*
     modification, which never happens; `clear_cache`'s `ic ivau` + `dsb ish` make the new bytes
     visible to a sibling's fetch. Pinned by `jit_cap::cross_thread_execute_fresh_code_agrees` (worker
     spawned *first*, then main compiles+installs, worker executes the fresh code → 52), green on all
     `fiber_rt` targets incl. aarch64 macOS. (install-during-own-invocation is covered structurally by
     the interp's `INVOKE_MODULE` + shared table.)
   - **Compaction + multithreaded — done.** `JitSession` now **owns** the `Host` behind a boxed
     `Mutex` (stable address) and bakes `cap_thunk_locked`, so a multi-threaded guest's workers can
     `cap.call` (incl. threaded `Jit.compile`) *and* the session auto-compacts between prompts (a
     quiescent point), re-baking the **same** locked thunk so the next multi-threaded prompt stays
     sound. The thunk-agnostic keep-set rebuild is factored into `pub recompact_into(fresh, host,
     domain, old)`; `recompact_jit` stays the single-threaded standalone primitive (a concurrent
     custom driver replicates `JitSession`'s `Mutex<Host>` + `cap_thunk_locked` + `recompact_into`
     pattern). `JitSession::new` now takes `Host` by value; `run_prompt`/`compact` drop the host
     param; `into_host` recovers it. Pinned by `jit_compaction::threaded_session_compacts_transparently`
     (every prompt spawns a worker that concurrently `Jit.compile`s; identical with/without
     compaction; occupancy bounded). A single-threaded session pays only an uncontended lock per
     `cap.call` (negligible for an interactive driver; the perf path `jit_cap_run` stays unlocked).
   - **Threaded-compile remainders (narrow):** (a) a coarse-lock→**fine-grained / sharded-module**
     optimization if parallel-compile *throughput* is ever measured to matter — deliberately deferred
     (no demonstrated need; the guest `cap.call 11` iface is unchanged, so it's a pure internal swap).
     (b) **C-level threaded-compile demo — done, *differential*.** `demos/jit/jit_threads.c`: `NWORKERS`
     guest threads each build IR for a distinct unit and `__vm_jit_compile` it (several `Jit.compile`s in
     flight, serialized through the per-domain `Mutex<Host>` since the guest `thread.spawn`s), invoke the
     native code, and check it against a C reference — prints `0` mismatches. Pinned two ways: the
     product-path smoke test `run::demo_jit_threads_runs` (through the `svm-run` binary's locked thunk),
     **and** a full interp≡JIT differential `c_frontend::c_guest_jit_threads_demo`. The latter needed
     `run_c_full` to become **concurrency-aware**: a guest whose module `uses_concurrency()` now drives the
     JIT side through `cap_thunk_locked` over a `Mutex<Host>` (single-threaded guests keep the unlocked
     raw-`*mut Host` fast path verbatim), mirroring `run_powerbox` — so any genuinely-concurrent C JIT
     feature is now differentially testable, not just demonstrable. (The IR-level
     `jit_cap::threaded_compile_agrees_across_backends` + `…_loop_stress_agrees` remain the race/stress pin.)

**Recommendation:** #1 (compaction) and #2 (threaded **install** + threaded **compile**) are both
landed end-to-end — install with full platform parity, compile via the per-domain serialized thunk
(single-threaded paths untouched), CLI wired, the cross-thread-execute case confirmed needing no
`isb`, and **compaction works for multithreaded guests** (`JitSession` owns the `Mutex<Host>`). The
only open threaded-compile item is the throughput optimization (gated on a measured need).

Also nice-to-have, low priority: a guest convention/helper so emitting C-ABI units (the `sp`-first
shape) for `install` is less manual; today the demo hand-rolls it.

---

## 7. Conventions & gotchas

- **Commit messages:** end with the session line; **avoid backticks/`$()`/parens in `-m`** (the
  shell expands them — bit me twice). Use `git commit -F <file>` for multi-line bodies.
- **Format/lint every commit:** `cargo fmt --all` then clippy on the touched crates.
- Adding a `CompiledModule::compile` parameter ripples to ~7 call sites (3 in svm-run, 3 tests, the
  one-shot `run_inner`) — grep `CompiledModule::compile(`.
- `Jit.invoke` op is **eval-loop-serviced** in the interp (it runs guest code); `install`/`uninstall`
  too (they mutate the table). `compile`/`release` are generic-dispatch. On the JIT all five are
  intercepted in `cap_thunk` before generic Host dispatch.
- Module ids (interp): module 0 = the guest program; module `k` = `units[k-1]`. `install` pushes a
  unit as a new module and points a `TableSlot` at it. The JIT has no "module" concept (native
  pointers); only the **slot index** must agree across backends.
- **Testing a *concurrent* C guest differentially:** `c_frontend::run_c_full` is now concurrency-aware
  — a module that `uses_concurrency()` drives the JIT side through `cap_thunk_locked` over a
  `Mutex<Host>` (so worker `cap.call`s, incl. threaded `Jit.compile`, don't race), while a
  single-threaded guest keeps the unlocked raw-`*mut Host` fast path. So a threaded C demo can go
  straight through `run_c_full` for a real interp≡JIT differential; you do **not** need a bespoke
  harness. A single run is still a weak *race* detector, though — keep the stress/repetition pin at the
  IR level (`jit_cap::threaded_compile_loop_stress_agrees`).
