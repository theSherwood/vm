# Handoff — Guest-driven JIT (`Jit` capability, Model A + B2)

Pick-up notes for a fresh session. Written 2026-06-12. Branch: **`main`** (this work commits
straight to `main`; remote `theSherwood/vm`). Everything below is committed and the full
workspace suite is green (`cargo test --workspace` → 64 test binaries pass).

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
reject data segments → reject §12 concurrency ops (`Func::uses_concurrency`, single-threaded MVP).
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
new→old/new→new, each `Frame` gained a `module` tag (0 = the vCPU's own program, ≥1 = a
guest-compiled unit in `units[k-1]`). Direct calls stay in the caller's module;
`call_indirect`/`return_call_indirect` dispatch through an explicit module-aware table
(`TableSlot` + `dispatch_indirect`), the reference mirror of the JIT's `fn_table`.
- **Borrow gotcha (important if you touch the eval loop):** `units`/`table` are bound `&mut` from
  the `VCpu` destructure so `install`/`uninstall` can mutate them mid-loop; the running frame's
  module functions are cloned into a local `Arc` (`cur_funcs`) each iteration so neither `units`
  nor `table` is borrowed across the instruction loop. Do **not** reintroduce a long-lived borrow
  of them.
- `Jit.invoke` runs a unit as a nested inline-driven `VCpu` (`VCpu::new_invoke`) that takes a
  **snapshot** of the domain's `units` + `table`, so an invoked unit reaches the original program
  (new→old) and already-installed units (new→new). The JIT gets this for free (invoked native
  code dispatches the live `fn_table`).

---

## 4. Where things live

- `crates/svm-jit/src/lib.rs` — `CompiledModule` (`compile`/`run`/`run_raw`/`define_extra`/
  `invoke_extra`/`install`/`uninstall`), `DefinedFn`, `intern_*`, `n_real_funcs`,
  `table_reserve_log2`. The escape-TCB-adjacent crate.
- `crates/svm-interp/src/lib.rs` — `Frame.module`, `TableSlot`/`build_table`/`dispatch_indirect`,
  `VCpu::new_invoke`, the eval-loop `Jit` op arms (invoke=1, install=3, uninstall=4 serviced in
  the loop; compile=0/release=2 in generic dispatch `Binding::JitDomain`), `Host` Jit state
  (`grant_jit`/`grant_jit_with_table`/`jit_compile`/`jit_unit_*`/`set_jit_native_ctx`),
  `iface::JIT`=11 / `JIT_CODE`=12, the `JitValidator` seam.
- `crates/svm-run/src/lib.rs` — `cap_thunk` intercepts iface 11 → `jit_native_op` (native
  compile/invoke/install/uninstall over the registered live `CompiledModule`); `jit_blob_validator`
  (the canonical gate); `grant_jit(host, m, table_log2)`; `jit_cap_run` (drives the JIT with the
  module registered for mid-run re-entry).
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
- `cargo test -p svm --test jit_cap` — **the differential suite** (interp ≡ JIT): compile/invoke,
  all 4 cross-call directions, install/uninstall, garbage/memory-mismatch/data/concurrency
  rejection, quota, traps, a deterministic blob fuzz. This is the one that matters most.
- `cargo test -p svm --test c_frontend c_guest_jit_demo` — the C demo end-to-end on both backends.
- Always finish with `cargo test --workspace` + `cargo clippy -p svm-jit -p svm-interp -p svm-run`.

The differential harness is `jit_cap.rs::diff_run` / `diff_run_t(…, table_log2)` — runs the same
guest program + Host setup + blob on interp (`run_capture_reserved_with_host`) and JIT
(`jit_cap_run`) and asserts equal results/traps/final-memory. To add a B2 (install) case, use
`diff_run_t` with a non-zero `table_log2`.

---

## 6. Remaining work (prioritized, with honest scoping)

The user's core asks are **done** (old↔new both directions, install, slot reclaim). What's open:

1. **Code-memory compaction reclaim** — *the* valuable item; **substantial.** Slot reclaim
   (`uninstall`) shipped, but repeated `compile`s still consume the 256 MiB code arena with no
   per-function free in `cranelift-jit`. Reclaim = periodic **whole-module compaction** (recompile
   the live set into a fresh `CompiledModule`, swap). Key constraint: it can **only** run at a
   *quiescent* point — the guest is suspended *inside* the very module being compacted, so the
   guest can't trigger it mid-`cap.call`. So it is **embedder-facing** (a REPL driver between
   prompts), realized as a `CompiledModule` API (enumerate live units → recompile → swap, remapping
   handles/slots), not a guest op. Recommend building this as its own focused effort + a
   simulated-REPL test. The `-ENOMEM` byte-cap backstop is what bounds the arena until then.

2. **Threaded install** + **install-during-own-invocation** — *one* refactor, **niche + risky.**
   Today the interp's dispatch table is per-`VCpu` (owned `Vec`); spawned threads get a snapshot,
   so a *post-spawn* install isn't seen by workers (the JIT, sharing the live `fn_table`, would
   see it → divergence). Making it differentially faithful needs the table behind an
   `Arc<Mutex>`, i.e. a **lock on the reference interp's hot `call_indirect` path** (used by every
   concurrency/DPOR test) — the riskiest change in this area for niche value. Gate on a real need
   for threaded guests using `install`. The same refactor also closes the
   install-during-own-invocation edge case (currently: JIT live, interp snapshot stale — exotic;
   needs the invoked unit to hold the `Jit` handle).

**Recommendation:** do #1 (compaction) as a focused effort when REPL arena pressure is real; treat
#2/#3 as a separate, carefully-reviewed shared-table refactor gated on demonstrated need. Don't
half-bake the shared-table change.

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
