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

### I74 — wasm-JIT confines against the **compile-time** `mapped`, so an access into `vm_map`-grown memory faults on the JIT where the interpreter allows it (S3, correctness-of-tier — masking hinge; opened 2026-08-07, root-caused; runtime decline already keeps results correct, PR #665)

**What (root-caused).** The wasm tier's confinement (`emit_confine`, `svm-wasm-jit` `lib.rs:~2792`)
bounds-checks every access as `eff > mapped - width ⇒ MemoryFault`, where `mapped = 1 << size_log2` is
the module's **declared** memory size, **baked in at emit time** (`lib.rs:1985`). But a guest grows the
live mapped region at runtime via `vm_map` (ADDRESS_SPACE op 0). After a grow, an access into
`[declared_mapped, grown_mapped)` **faults on the JIT** (stale compile-time bound) while the
interpreter — which tracks the live `Window::mapped` (`svm-mask`) — admits it. Confirmed on the
`open_memstream` card: `memory 18` (256 KiB declared), the guest's `malloc` calls `vm_map` to extend
the heap, and the emitted allocator (`IR func 3` = `wasm-function[5]`) traps `TRAP_MEMORY_FAULT`
(env.trap code `2`) writing the allocation header past 256 KiB. The interpreter prints correctly.

**Why the existing exclusion misses it.** `map`/`unmap`/`protect` are *deliberately* out-of-subset for
the mask-only tier (`is_nested_leaf_cap`, `lib.rs:784`) — but only in their `Inst::CapCall` form. The
on-ramp emits heap growth as `Inst::CallSym "vm_map"` (the `<svm.h>` `__vm_map` builtin,
`svm-llvm:512`), and the CallSym outliner (`lib.rs:~1560`) rewrites **every** symbolic call into an
emittable cross-tier leaf wrapper unconditionally, so the page-state-changing `vm_map` slips into the
emitted subset.

**Correctness is already handled — this is a *which-tier* / efficiency gap, not a wrong-result bug.**
The runner reports the trap as `STATUS_TRAP` and the caller declines the whole run to the interpreter
oracle (PR #665), so the output is correct. The cost is that a program which grows memory runs on the
interpreter instead of the wasm-JIT.

**Why not a simple veto.** A static "module contains `vm_map` ⇒ decline" veto **regresses working
guests**: `qjs_repl.svmb` contains `vm_map` (heap-growth path) yet runs fully on the JIT today (its
workload never grows past the declared size, so it never faults). Vetoing on static presence would drop
qjs/SQLite from the JIT (losing the ~7× speedup). Confining against `reserved` instead would be
**fail-open** — the JIT would sail through unmapped `[mapped, reserved)` pages the interpreter faults
(the exact divergence `lib.rs:786` rejects). Neither is acceptable.

**The correct fix is the masking hinge (owner territory, INVARIANT 2).** Confine against the **live**
`mapped` size read at runtime (e.g. from the env cell the emitted code already holds — no new import,
per the CONSOLIDATION §0 yardstick — kept in sync by the `vm_map` cross-tier handler), instead of the
emit-time constant. This is the fuzzed confinement lowering ("the most sensitive code in the tree"), so
it needs owner sign-off and its own masking-fuzz coverage of the dynamic-`mapped` case before landing.
Until then the runtime decline (PR #665) is the fail-closed resting point. Severity raised S4→S3: it is
a real interpreter-vs-JIT semantic divergence (masked today only because the JIT declines on the
resulting fault).

### I73 — punt-inside-a-fiber: the fast backends blocked the vCPU inline where the tree-walk oracle parks the fiber — **CONVERGED 2026-08-07, all three engines** (S3; opened by FIBER_PARK.md F1, closed by F2+F3 on the same arc)

**F2** converged the bytecode cooperative driver (`FiberState::CapParked` + the ordered
`drain_cap_parked`; `fiber_punt_diff.rs` pins all four F1 kernels TreeWalk ≡ Bytecode bit-exact).
**F3** converged the Cranelift JIT: `cap_thunk`/`cap_thunk_locked` route a fiber's punt through
the pending face and park the FIBER (`fiber_cap_wait` over the `Completions` fiber cells — the
ordered drain lives inside the store's one lock — plus the `fiber_rt` event-park seam the futex
thunk already used); `svm/tests/fiber_punt_jit.rs` pins all four kernels across all three
engines, statuses/values/delivery-order alike. The enumerated **inline-by-design** postures that
remain are not debt: the bytecode parallel and browser-`Vcpu` drivers (the I45 posture — fiber
delivery through the real cross-thread futex is its own slice), the debug drivers (sanctioned
whole-vCPU tiering, invariant 9 observability corollary), the §22 invoke leaves (seam-free
atomic), confined `instantiate` children on the cooperative driver (their completions live on
their own host), and durable runs everywhere (`freeze_drive` has no cap-park re-derivation).

**What.** F1 extended the §3.6 slice-5a fiber-park contract to punted host calls: on the
tree-walk oracle, a `Pending` dispatch inside a fiber now unwinds `FIBER_PARKED (3)` to its
resumer and the pool completion wakes the fiber (`completion_waiters` × `Waiter::Fiber`, the
ordered drain). The bytecode cooperative driver and the Cranelift JIT still run the pre-F1
semantics — the punt blocks the vCPU inline (bytecode: `comps.wait(id)` at its punt site; JIT:
`cap_thunk`/`cap_thunk_locked` wait on the thunk). A resumer polling a fiber that punts sees
`FIBER_PARKED` on the oracle but `FIBER_RETURNED`-with-result on the fast backends.

**Why tracked, not fixed here.** A punt is a per-call runtime decision (`OffloadOutcome`), so a
compile veto cannot see it — decline is unavailable and parity is the only convergence
(invariant 9). Each fast backend mirrors its existing slice-5a futex fiber-park shape keyed by
completion id: **F2** (bytecode `FiberState` arm + `drive` idle wake on completions) and **F3**
(Cranelift JIT per-fiber completion cell + the `fiber_rt` event-park seam) are the plan, on the
same PR arc. Until then the divergence is witnessable only by a punt-inside-a-fiber kernel: the
generic differentials generate no fiber kernels (verified — `bytecode_diff`/`jit_diff` carry
none), and the I45 rule extends — any harness pointed at the fast backends must avoid
punt-inside-fiber kernels until F3 closes this. Cross-engine pins land with F2/F3
(`fiber_timed_wait.rs` `pin_all` pattern).

**Scope notes (the deliberate non-divergences).** Root-context punts are unchanged everywhere
(vCPU park, guest-invisible). Durable callers never fiber-park on punts (the predicate's
`!durable` — `freeze_drive` fails closed on unwoken cap parks, pinned in `fiber_parks.rs`).
The explorer has no powerbox, so no punt can occur there (the `SchedRef::Real` gate is
structural). The I45 secondary drivers gain the same row by reference.

### I72 — `fiber-scaling features (macos-latest)` failed at *action download* with `Service Unavailable` (S4, GitHub Actions infra flake) — recorded 2026-08-06 on PR #642

Run 31119236467: the job died in **Prepare all required actions** — `Failed to resolve action
download info. Error: Service Unavailable`, retried twice (21 s, 17 s) then `##[error]Service
Unavailable`. No project code ran; the commit was test-only (`lua_futamura_call_arith` + ISSUES).
GitHub's action-registry was transiently down. Sibling of I67/I70 (GitHub-hosted infra flakes);
rerun-once policy applies — a fresh push re-triggers and clears it. No code fix.

> **I36–I40 (2026-07-23):** the §3.6 serving-substrate review, recorded at the owner's request
> after a design walkthrough. Two further items from the same review were **already tracked** and
> are not duplicated here: fiber-level `svc.wait`/`Join` parks (TODO.md §3.6 residue,
> "Join-in-fiber parks") and durability × serving (TODO.md "Durable event-parks" + PROCESS.md O10).
> Verdict from the review: none of these needs a different design — the model is the actor model
> (domain = actor, svc queue = mailbox, one world = actor state) — but I36 is a promoted work item
> and I37/I38 need their idioms documented so they're chosen, not stumbled into.
>
> **I36 update (2026-08-07):** the fork substrate (`clone_caller`/`reap`) was part of this fold.
> Progress: (a) the parity matrix is now **honest** — the process/serve/fork ops are their own
> `OPS_PARITY.md` family classified per-backend, no longer hidden in the `cap.call` row; (b) **native
> bytecode fork landed** — `clone_caller`/`reap` run on the bytecode cooperative serve driver (twin =
> the parked caller's `Vm` cloned over `fork_private` + `fork_powerbox`; `reap` via a `BlockedReap`
> state + `forked_twins`), pinned bit-for-bit against the oracle. The per-backend veto split keeps
> Cranelift folding fork. So fork is now ✅ on tree-walk + bytecode, 🚧 on Cranelift (the remaining
> slice), ⛔ on the wasm-JIT leaf. Track + Cranelift plan in FORK.md §9.

### I71 — peval precall/poscall call projection gaps — **ALL FACETS FIXED** (S3) — recorded 2026-08-06, closed 2026-08-07: **(a)** result-feeds-arithmetic (`lua_futamura_call_arith`), **(b)** nested 2-frame (`lua_futamura_call_nested` — root cause a `CallInfo` overlay collision, `CI_SIZE` 104 vs the real 64-byte stride, not an engine bug), **(c)** sequential distinct callees sharing one cached `CallInfo` (`lua_futamura_call_seq` — per-site `LuaSite::pins` on the shared node's `func`/`savedpc`); plus the call-bearing loop now **executes** (`lua_futamura_call_loop_exec`). All config/test — zero engine changes.

PR #637 landed projecting *through* a Lua-to-Lua call (`SpecConfig::precall_model` + `PoscallModel`):
slice 1 (call-in) and slice 2 (return) run end-to-end and diff stdout for `print(add(40,2))`, and
slice 3 shows a call-bearing loop `for i=1,5 do x=add(x,3) end` **rolls** — but slice 3's check is
**structural only** (`br_table=0`, a back-edge, bounded blocks); the residual keeps its cut call-outs
so it is not self-contained and was never *executed*. That masked this gap.

**Symptom.** When a Lua call's result is consumed by **arithmetic in the caller** (not by a C call like
`print`, which is opaque to tags), the executed projection fails:
- **single call** `print(add(2,3) * 10)` → `SpecError::Unsupported`, reached while inlining an unnamed
  3-block arithmetic helper (source `f661`, block 1 ip 0 in the current `lua_eval.ll`) — an
  `OP_MUL`/integer-arith slow-path callee that isn't in the cut set and hits an unsupported access.
- **nested** `local function inner(x) return x+1 end local function outer(y) return inner(y)*2 end
  print(outer(10))` → `Budget` (divergence): the fold walks off a frame's code into garbage and inlines
  the metamethod cluster (`luaT_trybinTM`/`callbinTM`/`luaT_gettmbyobj`/`luaV_equalobj`/…). Both
  full-reload and `selective` diverge, so the 2-frame case is a *caller-resume* problem, not a tag one.

**Root cause — traced 2026-08-06 (follow-up investigation).** The original read of facet (a) ("cut one
`OP_MUL` helper") was too shallow; the real chain is two layered problems.

*Layer 1 — the cut set is missing the allocation/GC/string primitives.* `f661` is not an arith helper;
it is the **arena allocator grow path** (`l_alloc → f663 → f661`), whose `CallImport import:3` is the
host `vm_map`. It is reached because `l_alloc` (`f531`, no direct callers) is the `frealloc` pointer,
devirtualized by `indirect_targets_cap` and projected. Cutting `l_alloc` alone just moves the wall: its
`Dyn` result makes "did the alloc succeed?" dynamic, so the fold explores the **alloc-failure cold
subtree** — `luaM_realloc_ → luaC_fullgc` (emergency GC) → retry → `fread`/`fwrite`/`strstr` and a
second `l_alloc.777` clone. These are all stateful host-backed machinery that must be **cut like the GC
barriers already are**. The needed additions (confirmed): `luaM_realloc_`, `luaM_saferealloc_`,
`luaM_malloc_`, `luaM_growaux_`, `luaS_resize`, `luaS_newlstr`, `luaC_fullgc`. With those cut,
**`print(add(2,3))` (no arithmetic on the result) projects and executes correctly** — so the cut-set is
necessary and, for the plain-return case, sufficient.

*Layer 2 (the actual root) — the proto **constant pool (`proto->k`) was not overlaid.* With the cut-set
in place, `print(add(2,3) * 10)` still fell into `OP_MMBINK` → `luaT_trybinassocTM` → `luaD_throw`
because the `OP_MULK` fast-path guard `ttisinteger(v1) && ttisinteger(v2)` didn't fold. **A prior
"caller `base` goes dynamic" reading was a red herring** — instrumenting `eval_load` showed the register
operand `v1 = R[B]` *does* load at a constant base and resolves to the pinned `VNUMINT` tag. The operand
that stayed dynamic was `v2 = KC(i)` — the **constant `10`**, read from `proto->k[C]`. The overlays
covered the proto struct and the code/upvalue arrays but **not the `k` array it points to**, so the
constant's tag was unresolved and the guard couldn't reduce. `print(add(40,2))` never hit this because
small integer literals compile to `OP_LOADI` **immediates** (baked into the instruction word), never
touching `k`. Fix: overlay each frame's `proto->k` (`PROTO_K = 56`, `PROTO_SIZEK = 20`, `16 * sizek`
bytes) alongside the code array. The `k` pool is static program data — it belongs in the const image
exactly like the bytecode.

**Resolution (facet a).** `crates/svm-llvm/tests/lua_futamura_call_arith.rs` projects
`print(add(2,3) * 10)` and executes it **byte-identical to the interpreter (`50`)**, 172 residual
blocks, no deopt — a call result flowing into `OP_MULK`, the exact shape a real call-bearing loop needs
(`x = add(x, k)`). Both ingredients are config the `SpecConfig` API already supports (no engine change):
the `k`-pool overlay + the alloc/GC/string cut-set.

**Facet (b) — nested 2-frame (`outer(y) = inner(y) * 2`) — FIXED 2026-08-07**
(`lua_futamura_call_nested`, un-`#[ignore]`d, embeds + stdout-diffs `22`). Three ingredients, all
config, **no engine change** — and the earlier "engine-level, callee base is dynamic" reading was
**wrong**:

1. Facet (a)'s per-frame `proto->k` overlays + alloc/GC/string cut-set.
2. The **upvalue-machinery cut** (`luaF_findupval`/`luaF_closeupval`/`luaF_close`/`luaF_initupvals`)
   — `outer` captures `inner`, so `OP_CLOSURE` walks/mutates the open-upvalue list.
3. **The real root cause: a `CallInfo` overlay collision.** Adjacent ci nodes sit **64 bytes apart**,
   but the tests sliced each ci overlay at `CI_SIZE = 104` — so `outer`'s ci overlay (captured at
   outer's dispatch, *before* inner's ci exists) swallowed `inner`'s ci header with stale zeros.
   `inner.ci->func`/`savedpc` seeded as `0x0` (caught via `eval_load` seed traces: `ILOAD
   eff=inner.ci … seed cval=0x0`), inner's frame dispatched garbage, and the fold walked into the
   metamethod/GC cold subtree (`Budget`) — masquerading as a dynamic-base engine bug. Fix:
   `CI_SIZE = 64` (the allocation stride; every field the fold reads — func +0, previous +16,
   savedpc, trap, callstatus +62 — lies below 64) + a non-overlap assert. Harmonized across all
   call-model tests (`lua_futamura_call{,_arith,_loop,_nested,_reroot}`), all green.

**Shift-the-root (2026-08-07, `lua_futamura_call_reroot`)** — the experiment that exposed the
collision also validated the **per-function residual architecture**: the same nested program projects
cleanly when *rooted at `outer`'s frame* (outer's ci as the const entry arg, state overlaid from
outer's dispatch-moment window). 72 blocks, `br_table = 0`, no metamethod calls — and because Lua's
inline `RETURN1` fast path folds as const stores, `(10+1)*2` reduces to a **literal `ConstI64(22)`**
in the residual. So any frame of a call tree can be a projection root handling its own body + one
call boundary; depth-N inlining is an optimization, not a prerequisite for end-to-end peval.

**Facet (c) — FIXED 2026-08-07** (`lua_futamura_call_seq`, embeds + stdout-diffs `25`). Two
sequential distinct callees (`add`, `mul`) reuse one cached `CallInfo` (`main_ci->next` — the test
asserts the shared address). The fix needed **no engine change**: `LuaSite::pins` is already a
generic `(address, value)` cell list and mem cells shadow the overlay seed, so each call site pins
the shared node's per-callee fields itself — `ci->func` (that site's `ra`) and `ci->savedpc` (that
callee's code start, offset discovered per occupancy) — while ONE overlay carries the fields common
to both occupancies (previous, callstatus, trap). One shared `selective` entry covers both integer
returns.

**Call-bearing loop, executed (2026-08-07, `lua_futamura_call_loop_exec`)** — closes slice 3's
"structural-only" caveat from the other side: an **entry-rooted** projection of
`for i=1,5 do x=add(x,3) end print(x)` unrolls the constant trip count (five bodies through one
shared `OP_CALL` site/frame node), embeds wholesale, and prints `15` byte-identically. Together with
`lua_futamura_call_loop` (safepoint-rooted, rolls) the mechanism is shown to both roll *and* run;
the remaining un-executed artifact is the rolled residual itself, which needs mid-loop entry
stitching (same follow-up as the reroot residual).

**Stitching slice 1 (2026-08-07, `lua_futamura_call_stitch`)** — that follow-up's first execution:
**two composed residuals run one program** (`print(add(2,3) * 10)` → `50`, byte-identical). The
caller residual (entry-rooted) folds main's prologue, emits the real `luaD_precall` cut, and at the
callee's `startfunc` block — a [`deopt_targets`] entry — spills and **tail-calls the continuation**;
the continuation residual (add-rooted, the reroot shape, `() -> ()` per the deopt-handler
convention) resumes purely from the written-back window, runs `add`, main's post-call arithmetic,
`print`, and exits. Zero engine changes: the deopt machinery *is* a continuation-transfer mechanism
— the handler is simply pointed at a specialized continuation instead of the baseline interpreter
(pass B runs on a module padded with a `() -> ()` placeholder at a known index; the continuation's
entry replaces it after both passes, valid because both carries are identity-indexed). One capture
subtlety: the caller residual still needs the **callee proto overlaid** — main's `OP_CLOSURE` reads
`proto->p[bx]->sizeupvalues` to bound the upvalue-init loop, even though the fold never enters the
callee's body.

**Stitching slice 2 (2026-08-07, `lua_futamura_call_stitch_chain`)** — the composition **chains**: a
continuation residual is itself an ordinary projection that stitches onward at *its* next call
boundary. Three residuals, two stitches, one program (`print(add(2,3) * mul(4,5))` → `100`,
byte-identical): entry-rooted B → add-rooted A1 (runs add, main's middle, stitches at mul's
`startfunc`) → mul-rooted A2 (runs mul, the register-register `OP_MUL`, `print`, exit). The two
sequential callees share one cached `CallInfo`, so this composes with the facet-(c) per-site pins;
the module is padded with two placeholders and every pass carries identity indices. This is the
per-function-residual execution model working recursively; the remaining step to a *deployable*
stitch is the guard (closure-identity check with baseline fallback — today the transfer is
unconditional, valid only because the captured run is deterministic).

**Guarded stitch + float returns + tail calls (2026-08-07)** — three more of the named follow-ups:

- *Guarded stitch* (`lua_futamura_call_stitch_guard`) — the deployable shape, **no engine surface**:
  the stitch handler is a hand-built `() -> ()` function (what a stitch compiler would emit) that
  loads the callee closure pointer from the spilled window, compares against the expected constant,
  and tail-calls the continuation on a match or the **pristine baseline `luaV_execute`** (with
  materialized `(sp, L, ci)`) on a mismatch. Both paths proven by *poisoning the arm that must not
  be taken*: match variant (fallback poisoned) and mismatch variant (continuation poisoned, real
  baseline resumes from the spilled window at the callee frame and completes the program) both print
  byte-identically.
- *Float returns* (`lua_futamura_call_float`) — `PoscallModel::selective` generalizes beyond
  integers: `print(half(5) + 0.25)` with the return tag **discovered at capture** (read from the
  result slot at print's precall break, `0x13`), float fast-path folds, `2.75` byte-identical.
- *Tail calls* (`lua_futamura_call_tail`) — `return f(x)` via `luaD_pretailcall`, the frame-REUSE
  shape, via a new small engine model (`SpecConfig::pretailcall_model` / `PretailcallModel`,
  mirroring the precall model): at a matching `ra` the cut's int result binds negative (the Lua
  "frame moved" arm folds), `L->ci` stays on the reused node, and per-occupancy pins install the
  moved closure slot + callee `savedpc` (the facet-(c) pinning applied to frame reuse — the test
  asserts both occupancies share the node *and* the `func` slot). `print(outer(10))` where
  `outer(y) = return inner(y)` → `11` byte-identical.

**Capture automation (2026-08-07, `tests/futamura/mod.rs` + `lua_futamura_auto`)** — the pipeline:
Lua script in → verified residual out, with **zero per-program configuration**. One profiling run
with breakpoints on the precall/pretailcall branch blocks and the dispatch `br_table` yields an
event stream; a small state machine classifies sites (Lua / C / tail, loops deduped), captures each
callee occupancy at its own dispatch, discovers return tags **depth-qualified** (read at the first
event whose `previous`-chain depth drops below the callee's — correct across nesting), and derives
the uniform per-site pins, per-occupancy `CallInfo` overlays (with the I71(b) collision assert),
transitive proto-struct overlays (even *uncalled* definitions — `OP_CLOSURE` bounds its upvalue
loop by `proto->p[bx]->sizeupvalues`), the cut lists, and the tail/precall models. Seven programs
pass byte-identical through the one driver, including a combined nested+tail+shared-node+loop
program. The automation also flushed out a real latent engine bug: **a tail call's moved arguments
were stale in the fold** (the move happens inside the opaque cut; the hand test had passed by
value-coincidence — `inner(y)` moves `y` onto a slot that already held `y`). Fix: `TailSite::args`
reloads each moved value dynamic with its captured tag pinned (the selective-reload discipline at
the call-in side). Known limits, documented in the driver: one return tag per frame node (mixing
int/float returns through the same node needs savedpc-keyed selective), no C tail calls, no rolling
safepoint roots, no multi-value returns.

**First benchmarks (2026-08-07, `lua_futamura_bench`, `#[ignore]`d; run `--release -- --ignored`)** —
the arc's first timing numbers, with the compile confound handled by differential-N (the baseline
module is fixed across trip counts, so `(T(N₂)−T(N₁))/(N₂−N₁)` cancels JIT compile + parse exactly):
baseline `luaV_execute` costs **≈ 93 ns per iteration** for `x = add(x,3)` + forloop (20M-iteration
delta); whole-program at N=200 is at **parity** (both ≈ 0.95 s, dominated by JIT-compiling the
~690-function module); specialization is linear at ≈ 48.5 residual blocks / ≈ 0.6 ms per unrolled
iteration. The pinned-down economics: **unrolled specialization can never pay per-run** (~0.6 ms/iter
to build vs ~93 ns/iter to interpret) — the payoff shape requires build-once/run-many with a
compiled-module cache (no such API today: `run_powerbox` compiles per call, `Instance` holds only
the `Module`) or **rolled** loops (N-independent residual size). This quantifies why rolled +
stitched + cached is the deployment target, and confirms the pipeline costs nothing at runtime
(residual parity with baseline end-to-end).

**Execution-vs-execution number (2026-08-07, `rolled_residual_vs_interpreter_per_iteration`)** — the
payoff the arc was pointing at, now measured directly. A rolled residual runs standalone with a
*dynamic* trip count, so its module is N-independent and the differential-N trick cancels compile on
*both* sides — the first true execution-only comparison. On a pure `x = x + 3` loop the interpreter
costs **≈ 17.6 ns/iter** and the dispatch-folded rolled residual **≈ 1.53 ns/iter**: an **11.5×
per-iteration execution speedup**. (The 93 ns/iter baseline above is heavier because it pays a full
Lua `add(x,3)` call each iteration; the two baselines differ by exactly that call overhead.) With the
interpreter's decode+dispatch removed, the rolled residual runs an order of magnitude faster per
iteration — confirming rolled loops are the shape that actually pays, not just the shape that avoids
the unroll blow-up.

**Compiled-module cache (2026-08-07, `svm_run::PowerboxProgram`)** — the build-once/run-many API the
benchmark economics called for, now built. `run_powerbox` JIT-compiles the whole module on every
call; `PowerboxProgram::compile(module)` pays the Cranelift codegen **once** and `run(stdin)` reuses
the native code per input. Each run stays as independent as a fresh `run_powerbox` — `run_raw`
allocates a new guest window and re-applies data segments every call, and the boxed host (whose
stable address is the baked `cap.call` ctx) is reset to a fresh powerbox each run — verified
byte-identical to `run_powerbox` by the `powerbox_program` differential tests. Scope: the
single-threaded compute powerbox (the `cap_thunk` + raw-`*mut Host` + `fast_cap_resolver` non-locked
arm, fiber hosting stood up exactly as the one-shot does); concurrent / tree-walk-folding modules are
refused with a pointer to `run_powerbox`. Measured on the ~690-function Lua module
(`powerbox_program_amortizes_compile`): a batch of N=8 runs drops from **≈ 775 ms/run** (per-call
compile) to **≈ 145 ms/run** (compile once) — a **5.3× batch speedup** that grows with N toward the
compile:exec ratio. This is what turns the 11.5× per-iteration execution win into an end-to-end win:
a built residual (or the bare interpreter) amortizes its compile across a batch instead of re-paying
it every invocation.

**Auto-rolled residual, slice 1 (2026-08-07, `lua_futamura_auto_rolled`)** — the rolled residual (the
shape that delivers the 11.5×) built with **zero hardcoded register offsets**. The hand path
(`lua_futamura_rolled`) hardcodes the FORLOOP counter (R[3]) and carried registers (x=R[1], i=R[2]);
this discovers them by observing the running VM. Key finding from the probe: `ci->savedpc` is synced
*lazily* (it sticks at the loop-body pc while the body re-executes), so it is not a per-bytecode PC —
but the **loop body is the savedpc value with the longest run of consecutive dispatch hits**, its
first hit is a clean resume safepoint, and the **carried cells are the frame registers that vary
across the loop region** (prologue hits excluded, else pre-FORPREP garbage corrupts the diff); the
trip counter is the carried cell whose loop-region series is monotone-decreasing. Auto-discovery
recovers exactly the hand set {R[1],R[2],R[3]} (plus a harmless internal loop register), and
specializing with the discovered cells rolls the real `luaV_execute` **841 → 53 blocks, br_table=0,
4 dynamic params** — identical roll to the hand build — correct across a trip sweep on tree-walk and
JIT. This is a standalone test (doesn't touch the shared `futamura::auto` driver yet); folding it in,
and identifying the accumulator generically for write-back, are the next slices toward an end-to-end
auto-rolled win.

**De-overfitting: discovery on a second interpreter (2026-08-07, `peval_second_interp`)** — evidence
the runtime cell-discovery is *not* Lua-shaped. The same technique (observe the dispatch block, find
loop-carried cells as the registers that vary across hits, pick the counter as the one that
monotone-decreases from its peak) is pointed at a **structurally different** interpreter: a C
register-machine VM (`clang -O2 → svm-IR`) whose registers are a **global array** (`long reg[16]`, not
frame-relative), values are **untagged** `long` (no `TValue` tag), and whose program counter is a
**local in a register** (not `ci->savedpc` in memory). Discovery recovers exactly its counter (R0) and
accumulator (R1), excluding the invariant R2, with only ONE target-specific input — *where the
registers live* (the `reg` symbol's address from the on-ramp `data_symbols`; for Lua, `ci->func+16`).
That register region is the first field of a real `TargetDesc`. One genuine structural finding: the
regVM's pc-in-a-register means it **cannot be resumed from a memory snapshot** the way Lua can, so it
entry-roots rather than safepoint-roots — cell discovery transfers, but *rooting strategy* is
target-specific. Next: extract the `TargetDesc` abstraction so the Lua and regVM capture paths share
one parameterized driver, and (harder) a second interpreter that itself needs safepoint rooting.

**De-overfitting: the shared `TargetDesc` driver (2026-08-07, `tests/peval_capture/mod.rs`)** — "same
technique, parallel code" is now "same code, two interpreters". The near-identical Lua and regVM
discovery code was extracted into one interpreter-agnostic `discover()`, parameterized by a static
`TargetDesc` (register stride, count, optional value-tag filter) and a per-run `Located` (register
base, optional in-memory pc address). One unified predicate reproduces both prior behaviours: a
carried cell is a register that takes **≥3 distinct values** over the loop region (Lua's "varies in
the loop region" and the regVM's "distinct ≥ 3 over the whole stream" collapse to this), the safepoint
is the longest in-memory-pc run when a pc address is given (Lua) or hit 0 otherwise (the regVM's
register-pc), and the counter is peak-based monotone-decreasing. Both `lua_futamura_auto_rolled`
(841→53 blocks, rolled + correct) and `peval_second_interp` (R0/R1 discovered, counter=R0) now call
the shared driver and stay green. The Lua↔regVM difference is now entirely data: two structs, no
branching in the driver.

**De-overfitting: the SAFEPOINT roll on a second non-Lua interpreter (2026-08-07,
`peval_struct_interp`)** — the strong generality test. The regVM (`peval_second_interp`) keeps its pc
in a machine register, so it entry-roots and never exercises the hard Lua machinery. This third
interpreter does: a C struct-VM `VM { long pc; long r[16]; }` at a fixed address, with `resume(prog)`
a pure dispatch loop that **resumes from the in-memory `vm.pc`** (like Lua's `ci->savedpc`) and
`run(prog, n)` setting it up. The **unchanged shared `discover`** finds the loop safepoint via the
in-memory pc and the carried cells (`r0`/`r1`) in the struct; specializing `resume` rooted at that
safepoint **rolls** the real dispatch loop (9→24 blocks, br_table folded, 2 dynamic params) and
matches the interpreter across a trip sweep on tree-walk + JIT. So the full Lua-style safepoint +
dynamic-cell roll — not just cell discovery — works on a layout that shares nothing with Lua (struct
fields vs frame `TValue`s vs a bare global array). Two incidental findings: this VM writes `vm.pc`
every bytecode (no *lazy*-savedpc stickiness like Lua), so the safepoint is a body-entered hit
(the same "in-flight iteration" the Lua rolled residual has) — which is why correctness is anchored to
the interpreter itself, not a hand formula; and `resume` had to be `noinline` or `-O2` folds it into
`run` and the dispatch never appears as its own function.

**Zero-config auto-rolled, folded into the driver + end-to-end number (2026-08-07,
`tests/lua_rolled/mod.rs`, `lua_futamura_auto_rolled`, `auto_rolled_zero_config_vs_interpreter`)** —
the rolled residual (the 11.5× shape) is now built by a reusable driver function `auto_rolled(m,
script)` on top of the shared `peval_capture::discover`: script in → rolled residual + run/verify
metadata out, no hardcoded offsets. The accumulator is identified generically as the lowest-address
carried cell that is not the trip counter (the first local declared before the loop; documented as
the common-shape heuristic). `lua_futamura_auto_rolled` now calls it (recovers x=R1/i=R2/counter=R3,
841→53 blocks, correct across a sweep). The end-to-end benchmark measures the whole pipeline with no
hand-config: profile + discover + specialize is one **≈ 0.3 s build** (the cost the compiled-module
cache amortizes), and the resulting residual runs at **≈ 1.5–1.7 ns/iter** — matching the hand-built
rolled residual, i.e. **≈ 6–11×** faster than the interpreter (ratio swings only because the
interpreter baseline is noisy on shared runners, ≈ 10–18 ns/iter; the residual side is stable). This
closes the arc: a Lua chunk in, a measured per-iteration speedup out, zero hand-configuration.
Remaining generality gaps for `auto_rolled` (documented in the module): dataflow-based accumulator
identification for chunks whose result is not the first local, and post-loop code that calls out
(e.g. `print`) rather than a clean `return`.

**More-real programs + the result-register decode (2026-08-07, `lua_real_programs`)** — pushed the
zero-config pipeline past `x = x + 3` onto genuine numeric loops whose body uses the loop variable and
does real arithmetic. A gallery (`#[ignore]`d bench) runs each and prints the result **plus the
per-iteration speed of the interpreter and of the residual**: `sum 1..n` (=1275), `sum of squares`
(=42925), `polynomial i²−i` (=41650), `powers of two` (=2⁵⁰), and **fibonacci** (=fib(51)=20365011074,
two accumulators + a swap). Residuals ≈ **1–3 ns/iter** vs the interpreter's ≈ 9–26 ns/iter — **≈
8–15×**, all byte-correct. This retired the "lowest carried cell = accumulator" heuristic: the result
cell is now decoded from the chunk's `RETURN`/`RETURN1` bytecode (its `A` operand, +1 for the
VARARGPREP frame shift), which is correct for multi-accumulator chunks like fibonacci (returns `b`,
not the lowest cell). Two frontier walls found and documented: **`%` and `//` hit `Unsupported`** in
the specializer (the div/mod divide-by-zero cold path the auto config doesn't yet deopt — bitwise `&`
works, so it's div/mod-specific; modulo is the highest-value next unlock), and a **conditionally-
updated accumulator** (`if i>25 then s=s+i`) is mis-discovered when it stays constant through the short
observation window. Everything that rolls is straight-line integer arithmetic per iteration.

### I70 — `real-browser` CI job: the `Install Playwright + Chromium` step times out at 10 min because the Azure apt mirror serves `--with-deps` font packages at ~35 KB/s (S4, flaky CI infra) — recorded 2026-08-06 on PR #639

Run 31106929611 (attempt 1): `npm exec playwright install --with-deps chromium` spent the
entire 10-minute step budget inside `apt-get install` of the recommended font set —
`fonts-ipafont-gothic` (3.5 MB) alone took 100 s from `azure.archive.ubuntu.com` — and the
step's `timeout-minutes: 10` killed it before any project code compiled or ran. Sibling of
I67 (same apt-mirror substrate, different failure spelling: slow instead of 403). Rerun-once
policy applies (I53/I66/I67 class). A robust fix, if it recurs: cache
`~/.cache/ms-playwright` keyed on the Playwright version (`actions/cache`), and drop
`--with-deps` in favor of a one-time `playwright install-deps` layer or the runner image's
preinstalled Chromium — the deps apt set is what's exposed to the mirror. Workflow edits go
through `.github/workflows_src/` (session tokens lack the `workflow` scope).

### I69 — `threaded_offers::concurrent_callers_share_one_threaded_instance` intermittently loses one caller's increment under full-suite parallel load (S4→S2, real lost-caller window in the threaded-offer admission) — recorded 2026-08-06 on PR #634 — **FIX LANDED 2026-08-06** (`claude/i69-issues-luifte`)

**Symptom.** Under `cargo test -p svm-interp` with the full suite running in parallel, the test
fails with `left: Ok([I64(10)]) right: Ok([I64(21)])` — one of the two spawned callers' writes
(10 / 11) did not land in the shared instance window. 8/8 green when the binary runs alone;
observed once (Linux, local, suite under load). The file was untouched by the change under test
(§3d.1 record migration), so this is pre-existing.

**Root cause (not a lost *update* — a lost *caller*).** The earlier "lost read-modify-write /
`drive_arc_shared`" reading was wrong: the two writes are to **distinct** cells (16, 24) with no
byte-level race, and the host-side `drive_arc_shared` tier isn't on this test's path (it goes
`thread.spawn` → guest `cap.call` → the **eval-loop** admission). The real defect is in that
eval-loop arm (`crates/svm-interp/src/lib.rs`): admission opens with `state_.try_lock()`, and the
`Err(WouldBlock)` arm answered `-EAGAIN` **before** the policy was even checked. Two concurrent
callers on distinct vCPUs collide on that brief snapshot lock; the loser is handed a spurious
`-EAGAIN`, so its `store` handler **never runs** and its cell stays `0`. The thread's return value
(`-11`) is discarded by the consumer, so the only visible effect is the short sum (`10 + 0 = 10`).
This violates the `Threaded` contract (DESIGN.md §12a: **no admission gate**) — the `-EAGAIN`-on-held
-lock is correct *only* for the `single`/`drive_arc` tiers.

**Fix.** For `entry_.policy == OfferPolicy::Threaded`, the `WouldBlock` arm now **blocks** to acquire
the state lock instead of refusing. Bounded and deadlock-free: a threaded provider's critical section
holds `state_` only to fork the window + clone the powerbox cell and drops it **before** the handler
runs (it never spans a sub-run — a threaded offer is never durable, so the long-holding 3a fallback
can't apply, and the lock is released before any nested `cap.call`). Every other tier keeps the 3a
`-EAGAIN` semantics. Regression: `concurrent_callers_never_lose_an_admission` fans out 10 concurrent
threaded callers and sums every cell; it failed **8/8** (short sums 35–53 vs 55) before the fix and
passes **8/8** after (plus the full `svm-interp` suite, fmt, clippy green). Reclassified S4→S2 — a
silently-dropped guest dispatch is a wrong result, not a flake.

### I68 — `fork_manager` guest-fork race: intermittent HANG or wrong result in the FORK.md fork/join path (S3, real race — reclassified from S4 after recurrence + Linux repro) — recorded 2026-08-05 on PR #627, updated 2026-08-06 on PR #629 — **FIX LANDED 2026-08-06** (`claude/i68-issue-fix-sxghke`)

**FIX LANDED (2026-08-06) — not a substrate race: the three fork tests didn't honor FORK.md §8.6's
serve/park-race contract.** Reproduced on Linux (~1 hang in 150, ~1 wrong-result in 400) and diagnosed
with a scheduler-quiesce dump + a fork/reply/park event trace. The `-EAGAIN` on the serve/park race
(`svc_enqueue` wakes the server *before* the caller registers its `CapReply` waiter, so a handler can
run `clone_caller` before the caller is parked and there is no parked vCPU to clone) is **by design**
(FORK.md §8.6) — the contract is a **long-lived server** plus a **guest that retries** `while ((pid =
fork()) < 0)`, exactly what the reference `clone_caller.rs::SRC_FORK_PID` does. All three
manager-topology tests violated it two ways:

- **The HANG.** Every server (`func 1`) was a **single-shot** `svc.wait` (`svc.wait; return`), not the
  `svc.wait` **loop** its own doc comment claimed. When the guest's first `fork()` lost the serve/park
  race, the server replied `-EAGAIN`, then **exited**; the guest's retry enqueued on the now-dead server
  and parked in `ticket_waiters` forever (workers idle, `live > 0`, `shutdown` never set → the 45-min CI
  cancel). Trace: `FORK miss-none (2,0)` → `REPLY stash (2,0)=-11` → `PARK early-hit (2,0)=-11` →
  `FINISH id=1` (server exits) → `PARK insert (2,1)` (retry strands). **Fix:** loop the server
  (`br 1(); … br 1()`), matching `SRC_FORK_PID` and the doc.
- **The WRONG RESULT (`left: [I64(-11)]`).** `fork_import`'s IR guest and `c_fork`'s C guest
  (`slot = fork();`) called `fork()` **once, with no retry**, so a raced first attempt returned `-EAGAIN`
  straight to the caller. (`fork_manager`'s guest already retried — which is why it only ever hung, never
  returned wrong.) **Fix:** add the retry — `br_if vforkfail` in the IR guest, `while ((slot = fork()) <
  0);` in the C guest (the realistic shell idiom, §8.6).

Test-only change (no `crates/svm-interp` edit — the substrate's `-EAGAIN` is correct). After: all three
pass **1000/1000** under the same stress that reproduced both failure modes; `clone_caller.rs` (7 tests)
stays green. Note this refines the I53 sighting that claimed these three "do not flake" — they did, via
the missing-retry / single-shot-server path, not the root-is-forker artifact I53 tracked.

**Update 2026-08-06 (PR #629, head 45fc6a19):** recurred on macOS (same signature — 60s
warning, silent until the ~45 min workflow cancel, orphan `fork_manager` terminated), so
rerun-once is exhausted. Local stress on Linux **reproduces it**: ~1 hang in 150 runs
(10s timeout) and, separately, ~1 outright test FAILURE (exit 101 — wrong result) in 400
runs — the same race manifesting two ways. Not macOS-specific and not OS fork(2): the
"real libc fork" is FORK.md's guest-level `fork_twin` (vCPU duplication on the M:N
scheduler), so this is an order-sensitivity/lost-wakeup race in the fork/join/memfs-share
path, the I52/I53 family. **Escalation:** a dedicated slice with a loom model of
`fork_twin`'s park/wake (the parallel-scheduler loom harness exists; add the fork shape),
plus the per-test timeout so a hang fails in seconds. Until then this test can hang any
`build · test` leg — treat a `fork_manager` orphan in the cancel log as this issue.

Observed once on head 258a0182: the test printed the 60-second warning, then nothing until
the job was cancelled at ~45 min and the orphaned `fork_manager` test process was terminated
in cleanup. Same run: the test passed on Linux and Windows; locally it completes in
milliseconds. The head's diff (§3a/§3b record-spawn) is behavior-neutral on the legacy spawn
paths this test uses (record locals are `None`; the new `fits` conjunct short-circuits without
locking), so this is the macOS fork(2)+threads flake family (I52/I53): a real fork on a loaded
runner with live sibling threads can deadlock the child (lock held across fork). Rerun-once
policy applies. If it recurs: audit locks held at `fork_twin`'s fork point on macOS
(pthread_atfork discipline or single-threaded fork window), and consider a per-test timeout so
a hang fails fast instead of eating the job's 45-minute budget.

### I67 — `svm-llvm` CI job: `apt-get update` dies on unrelated `packages.microsoft.com` 403 before any Rust runs (S4, flaky CI infra) — recorded 2026-08-05 on PR #627 — **FIX STAGED 2026-08-07** (`.github/workflows_src/ci.yml`, awaiting owner copy-over)

**Fix staged (2026-08-07, `claude/ci-flakiness-git-hooks-c8xg3t`).** Applied the robust fix below to
**every** `apt-get update` step, not just `svm-llvm` — the same unused `microsoft*`/`azure*` sources sit
on every ubuntu runner, so the 403 could fell any of the 10 apt sites (mingw cross lanes, the `clang`
reference lanes, all `llvm-18` blocks). Each now runs
`sudo rm -f /etc/apt/sources.list.d/microsoft* /etc/apt/sources.list.d/azure*` before `apt-get update`
(glob + `-f`, so it's a no-op if the files are absent — resilient to the deb822 `azure-cli.sources`
vs. legacy `.list` naming drift). Staged in `workflows_src` per the CI-editing protocol; the
`workflows-in-sync` guard stays red until the owner copies it over. Move to Resolved after the copy-over
lands green.

The Linux-only `svm-llvm` job installs LLVM/clang via apt; on a GitHub runner the
`apt-get update` step failed with `403 Forbidden` from `packages.microsoft.com`
(azure-cli / ubuntu-prod repos, unrelated to anything we install) and the job died with
exit 100 before the checkout was even compiled. Pure runner-image/mirror flake — no tree
change can affect it. Rerun-once policy applies (like I53/I66). A robust fix, if it recurs:
prefix the apt step with `sudo rm /etc/apt/sources.list.d/microsoft-prod.list
azure-cli.sources || true` (the job needs nothing from those repos), via
`.github/workflows_src/` per the CI-editing protocol.

### I64 — the binary format's v9→v10 bump carries a one-version decoder compatibility window; retire it when the committed `.svmb` assets regenerate (S3, tracked debt) — recorded 2026-08-05, CALLS.md 7.4 (PR #612)

v10 added the impl-export policy byte (CALLS.md 7.4). The bump surfaced that the format's
exact-current-VERSION rule predated **stored artifacts**: 17 committed `.svmb` blobs
(`browser/web/assets/` + `browser/tests/fixtures/` — chibicc, QuickJS, the canvas demos, the
stage-1 shell) are v9 and regenerate only through their heavyweight LLVM on-ramps
(`build_chibicc_svmb.sh`, `prep_svmb`, `build-pg-assets.mjs`). The decoder therefore accepts
**v9 and v10** (a v9 impl-export has no policy byte and reads as `single` — what every v9-era
offer was); v8 and below stay rejected. Pinned by
`svm-encode::a_v9_blob_with_an_impl_export_decodes_as_single` (bytes derived from the v10
encoder, so the pin tracks the real layout). **Close-out:** regenerate the 17 assets at ≥v10,
drop the `v != 9` arm and the pin, restore exact-version. Hand-rolled guest emitters
(`demos/jit/*.c`) were bumped to v10 in the same PR — grep `format v` before any future bump.

### I65 — `real-browser` `instnested` one-off: `vcpu confined setup/host trap: memory access out of bounds` (S4, flaky CI — single occurrence) — recorded 2026-08-05 on PR #612

Failed once on run 31022849148 (commit `7f8a4813`, the CALLS.md 7.3 push); the identical suite
passed on the same code plus 7.4 (run 31025127104) and on main at the shared base. No local
reproduction attempted (the scenario needs the Playwright harness). If it recurs, suspect the
§14-on-emitted-wasm worker setup path (the main-side `696c83e5` fuel-global change is nearby);
one data point is not a diagnosis. Logged per the AGENTS.md flakiness rule.

### I66 — macOS `jit_poll_observes_a_concurrently_running_child` is timing-gated: a loaded runner finishes the 20M-iteration child before the parent's first poll (S4, flaky CI) — recorded 2026-08-05 on PR #612

Failed on run 31021473236 (commit `b682562f`) with `got Returned([11])` — the child ran to
completion before the first poll, so the "still running" observation raced. The assertion is
inherently timing-based (`crates/svm/tests/jit_lifecycle.rs:191`); a slow/loaded macOS runner
loses the race legitimately. Rerun-once policy applies (like I53). A robust fix would gate the
child on a rendezvous the parent releases *after* its first poll rather than a spin count.

### I61 — `wasm_transpile` panicked ("not a func") on a GC composite type that the up-front validator accepts (S2, nightly fuzz red) — **FIX LANDED 2026-08-04** (`claude/nightly-ci-failures-u9y1ly`)

**Symptom.** Nightly `cargo-fuzz (all targets) (wasm_transpile)` has failed every run since 2026-07-25
(last green 2026-07-24). The minimized crash is a 13-byte module —
`[0,97,115,109, 1,0,0,0, 1,3,1, 0x5f, 0]` = `\0asm` + version + a type section declaring one
zero-field GC **struct** (`0x5f`) — that aborts with `thread panicked at
wasmparser-.../types.rs: not a func`.

**Root cause.** `svm_wasm::transpile` guards with `wasmparser::Validator::new().validate_all()`, whose
**default** `WasmFeatures` now enable the stabilized GC proposal, so a `struct`/`array`/`cont`
composite type validates cleanly. The type-section loop then assumed every sub-type was a function and
called `SubType::unwrap_func()`, which `panic!("not a func")`s on any non-func composite. We only lower
function types, so this is a fail-*open* on hostile-but-valid input — exactly what the fuzz target is
the standing net for.

**Fix.** Match `sub.composite_type.inner` and bail closed with `Error::Unsupported("non-function GC
composite type")` for the non-`Func` arms instead of `unwrap_func()`. Regression test
`gc_struct_type_is_unsupported_not_panic` replays the exact crash bytes and asserts a clean
`Unsupported`, not a panic. (This latent hole surfaced the moment the fuzzer first generated a GC
composite type; no transpile code regressed on 2026-07-24.)

### I60 — the vendored-`svm` submodule is a *linked worktree* of the parent repo's `.git`, so a `git gc` in one checkout can prune unpushed commits authored in the other's detached HEAD (S3, tooling hazard — cost real committed work this run) — recorded 2026-07-31 on the JACL SVM-backend migration

**Symptom.** JACL vendors this repo at `jacl_impl/vendor/svm` as a **linked git worktree** sharing the
parent's object store (`jacl_impl/vendor/svm/.git` is a `gitdir:` pointer into
`jacl_impl/.git/worktrees/svm`, not an independent clone). While work was committed on the vendored
worktree's **detached HEAD** but not yet pushed to a branch ref, a concurrent `git gc`/auto-gc running
against the *shared* object store pruned the commit as unreachable: it vanished from `reflog` and
`fsck --lost-found`, forcing every change to be re-applied from scratch against an advanced `main`.
This is the direct cause of how messy the SVM-backend-migration run was.

**Root cause.** A detached-HEAD commit is reachable only from `HEAD` in *that* worktree. Auto-gc
triggered from another worktree (or the parent) enumerates roots across the shared store; a
per-worktree `HEAD` reflog is honored, but the window between committing and the reflog/ref being
durable — plus `gc --prune=now`-style aggressive settings — let an unreferenced tip be collected.
Nothing in the tree is *wrong*; this is a property of `git worktree` + detached HEAD + shared gc that
bites the vendored-submodule development model specifically.

**Mitigation / working rule (no code fix — a discipline note).** When developing in the vendored
worktree: (1) **never work on a detached HEAD** — `git checkout -B <branch>` immediately so every
commit is anchored to a ref that gc treats as a root; (2) **push early** so `origin/<branch>` is a
durable root even if the local ref is lost; (3) avoid `git gc`/`git worktree prune` against the shared
store while another worktree holds unpushed detached commits; (4) `git config gc.auto 0` in the
vendored worktree during a long editing session is the cheap belt-and-suspenders. Recorded so the next
agent anchors to a branch first instead of rediscovering the loss.

### I59 — the detached `browser` workspace duplicates `svm_ir`/`svm_interp` when several test targets are built together, producing transient `E0308 "multiple different versions of crate"` (S4, build flake) — recorded 2026-07-31 wiring the JACL on-ramp playground

**Symptom.** Building/running more than one `browser/tests/*.rs` target in a single `cargo test`
invocation intermittently fails to compile with `E0308 mismatched types: … expected struct
svm_ir::Module (…), found struct svm_ir::Module (…)` — the *same* type reported as two different
crate instances. Building the targets **one at a time** (`cargo test --test <one>`) resolves it every
time; the tree is not miscompiled.

**Root cause (same family as I55/I18).** `browser/` is a **separate workspace** (its own `Cargo.lock`)
that depends on the main-workspace crates by path. Under some target/feature-unification orderings when
multiple test binaries are compiled together, Cargo instantiates `svm_ir`/`svm_interp` twice (distinct
`-Cmetadata`), and a value produced by one instance can't satisfy a signature from the other. It's a
resolver/unification nondeterminism, not a code defect — hence transient and per-invocation.

**Mitigation / fix sketch.** Immediate: run browser test targets individually in CI and locally (the
`real-browser`/wasm gates already do this per-`.mjs`; the native `browser/tests/*.rs` are the exposed
surface). Durable options, in order of preference: fold new native browser tests into a **single** test
binary (as I3/I30 did for the heavy `svm` links — prefer extending an existing target over adding a
new one), or fold `browser` back into the main workspace so there is one dependency instantiation.
Logged per the CLAUDE.md "log flakiness early" rule so it isn't re-diagnosed from scratch.

### I58 — the on-ramp compiler-guest hard-codes a 4 KiB `vm_map` alignment, so it faults on any host whose software page size is coarser (S3, latent — wasm was the trigger this run; 16 KiB native hosts are the next) — recorded 2026-07-31, the JACL on-ramp heap fix (`059c1b4a`)

**Symptom.** The self-hosted JACL compiler-guest (`jacl_compiler.svmb`), which grows its heap with
`vm_map`s as it stages macros, `MemoryFault`ed mid-compile **only on wasm32** (native ran clean). Root
cause was a page-size mismatch: the guest maps its heap in **4 KiB** increments, but the interpreter's
software prot-map granularity (`host_page_size()` in `crates/svm-interp/src/lib.rs`) was **64 KiB** on
wasm — a `map` of one 4 KiB guest page left the rest of the enclosing 64 KiB host page unmapped, so the
guest's next store past 4 KiB hit an uncommitted page and trapped. Reproduced natively by forcing
`host_page_size() = 64 KiB` (the guest then faults identically).

**The applied fix is a host-side patch, not the root fix.** `059c1b4a` sets `host_page_size() = 4096`
on `#[cfg(target_family = "wasm")]` (native keeps `page_size::get()` so the interpreter's
uncommitted-tail faulting still *agrees with the native JIT's `PROT_NONE`* — the interp≡JIT
differential; see the demand-paging note below). That makes the software prot-map granularity match
what the guest assumes on wasm, which is why the tour now compiles in-browser. But it only papers over
the mismatch where we hit it.

**The real latent bug (unfixed).** The guest assuming a fixed 4 KiB page is wrong on **any** host whose
page size is coarser than 4 KiB — notably **16 KiB-page native hosts** (Apple-silicon macOS, some
ARM64 Linux configs), where `page_size::get()` returns 16384 and the identical fault is reachable
*natively*, with no wasm involved. We simply haven't run the growing compiler-guest on such a host yet.

**Fix sketch (guest-side, page-size-agnostic).** Have the compiler-guest import a `vm_page_size`
capability (or read it from the powerbox at start-up) and **align its `map` requests up to the
runtime's actual page size** instead of assuming 4 KiB — round both the base and the length to
`vm_page_size` before each grow. Then drop the wasm `host_page_size()` override and let the guest be
correct on 4 K/16 K/64 K hosts uniformly. This is the small, boring fix; the host override is the
stopgap that unblocked the browser tour.

**Why not just demand-page the interpreter's tail.** Considered and rejected this run: making the
interpreter lazily commit the uncommitted tail of a mapped region (so a 4 KiB `map` implicitly covers
its enclosing host page) would make the **interpreter stop faulting where the native Cranelift JIT
still faults** (the JIT guards uncommitted pages with `PROT_NONE`; `confine_checked` deliberately
matches that). That divergence breaks the interp≡JIT differential — the escape oracle — so the guest
aligning to the true page size is the correct layer to fix, not the confinement path.

### I57 — `provision-nimony.sh` failed the `nim end-to-end` job with `fatal: destination path 'nimony' already exists and is not an empty directory` on any warm nimony cache (S4, CI break) — surfaced 2026-07-31 on PR #560, **FIXED same day** (`claude/bash-svm-viability-uqj6ts`)

**Symptom.** On PR #560 (fork capstone — touches only `svm-interp`, one new `svm` test, and `FORK.md`,
nothing in the nim path) the `nim end-to-end (real toolchain → svm)` job failed in the provisioning
step, immediately after a cache hit restored `nimony/bin`: `scripts/ci/provision-nimony.sh` then did a
`git clone` into `nimony/` and aborted — `fatal: destination path 'nimony' already exists and is not an
empty directory` (exit 128). Recurred on every re-trigger (the branch cache stayed warm), confirming it
is **deterministic on a cache hit**, not a race.

**Root cause.** The `actions/cache` step caches `nimony/bin` + `~/.cache/nim` — the built output, **not**
`nimony/.git`. So on a warm cache `nimony/` exists (from the restored `bin/`) but is not a git checkout,
the `if [ ! -d nimony/.git ]` guard passes, and `git clone nimony` aborts on the non-empty target. Cold
cache = `nimony/` absent = clone into empty dir succeeds; hence it only bit once the key went warm.

**Fix (applied).** `provision-nimony.sh` now inits a repo in place and fetches instead of cloning:
`mkdir -p nimony; git -C nimony init; git -C nimony remote add origin …; git -C nimony fetch
--filter=blob:none origin` (all branches, to mirror the old clone's reachability so the pinned SHA is
present), then the existing `git -C nimony checkout --detach "$NIMONY_REF"`. Works whether `nimony/` is
absent (cold) or holds a restored `bin/` (warm). Editing the script also rotates the cache key
(`hashFiles('scripts/ci/provision-nimony.sh')`), so the fix runs cold first, then warm on the new key.

### I56 — `pg-reload-test.mjs` intermittently fails the `real-browser` job at `page.reload` (`Timeout 30000ms exceeded` waiting for `load`) (S4, flaky CI) — surfaced 2026-07-31 on PR #558, hardened same day

**Symptom.** On PR #558 (a `browser/src/lib.rs` rename + a native-test comment + docs — nothing on the
Postgres/reload path) the `real-browser` job failed **after** every other real-browser check passed
(`browser-test.mjs`, `browser-jit-reactor-test.mjs`, `browser-play-editor-test.mjs`, the chibicc
bench): `pg-reload-test.mjs` got through boot + the 41 MB IndexedDB save, then
`page.reload: Timeout 30000ms exceeded · waiting for navigation until "load"`. Unrelated to the diff.

**Root cause.** The step waited on the `load` event, which doesn't fire until **every** sub-resource
re-settles — including the ~2.2 MB engine wasm and the ~4.3 MB `qjs_repl.svmb` asset re-fetched from
the COOP/COEP server after a hard reload. Under CI runner load that occasionally exceeds Playwright's
30 s default; the app itself was fine (the engine-ready gate and the persistence assertions never ran).

**Hardened here.** `page.reload({ waitUntil: 'load' })` → `{ waitUntil: 'domcontentloaded', timeout:
60_000 }`. The subsequent `waitEngine` (waits for `engine-state=ready`) plus the `SELECT 919191` /
"restored" checks remain the real gate, so relaxing the *navigation* wait cannot mask a genuine reload
or persistence regression — it only drops the over-strict full-`load` dependency. If it still recurs,
the next lever is a one-shot reload retry, or serving the two big assets with a warm cache across the
reload.

**Recurred 2026-08-04 on PR #588** (another browser-inert diff — `svm-wasm-jit` opt-in paths + docs;
every other real-browser check passed, again including the full wasm-JIT/§14/§22 suite and the chibicc
bench): the **hardened** wait also lapsed — `page.reload: Timeout 60000ms exceeded · waiting for
navigation until "domcontentloaded"` right after the 41 MB IndexedDB save. Applied the next lever named
above: `pg-reload-test.mjs` now does a **one-shot reload retry** on timeout (safe — the snapshot is
already durable before the reload, and `waitEngine` + the SELECT/`restored` assertions remain the real
gate). If it recurs *through the retry*, the remaining lever is warm-caching the two big assets across
the reload.

### I55 — the `browser` crate is a separate workspace, so a cross-crate rename left `main` un-buildable on wasm32 for a day before any gate caught it (S3, build-gate gap) — surfaced 2026-07-31, fixed same day

**Symptom.** `cargo +nightly build -Z build-std … --target wasm32-unknown-unknown` of `browser`
(`svm-browser` lib) failed to compile on `main`: `browser/src/lib.rs` still called
`Host::grant_host_fn` / `cap_id::HOST_FN`, both renamed to `grant_host_proc` / `HOST_PROC` by the
mechanical-rename commit `3f7957b` ("CALLS.md increment 1"). That commit updated `svm-run`,
`svm-interp`, `svm-posix`, and the standalone `svm-llvm`, but **not** `browser` — the two access-sink
call sites (added by the concurrent W3-in-the-browser debug work) drifted.

**Root cause — the gate gap, not the rename.** `browser/` is its **own cargo workspace**
(`browser/Cargo.toml`), so the per-PR `build · test · fmt · clippy` job — which runs
`cargo build/test --workspace` on the **root** workspace — never compiles it. The only lane that
builds `svm-browser` is the **nightly `real-browser`** job (`ci.yml`), so a root-workspace rename that
misses `browser/src` compiles green on the PR and only goes red the next night (or never, if that
lane is quiet). A path-dependency on the renamed crate means `browser` always resolves the *new*
source, so the break is guaranteed, not version-masked.

**Fixed here** (mechanical: `grant_host_fn`→`grant_host_proc` ×2, `cap_id::HOST_FN`→`HOST_PROC`);
verified by a full wasm32 build + the `browser-play-editor-test.mjs` real-Chromium gate (all green).
**Gap remains (the real S3):** nothing per-PR compiles the `browser` crate. Fix sketch — add a cheap
`cargo check -p svm-browser --target wasm32-unknown-unknown` (no `build-std` needed for a type-check)
to the per-PR lane, or a `workspaces-in-sync` guard that greps `browser/src` for renamed symbols. Until
then, any cross-crate rename must grep `browser/` by hand.

### I54 — `c_shell::stage0_shell_ring_pipeline_status_and_early_exit` intermittently fails the Linux `build · test · fmt · clippy` gate (S4, flaky CI) — surfaced 2026-07-30 on PR #545

**Symptom.** On PR #545 (a **`crates/svm-leng`-only** change — zero lines in `crates/svm` or the
shell/pipeline/bytecode-engine path) the `build · test · fmt · clippy` job failed with
`crates/svm/tests/c_shell.rs:333` — `assertion left == right failed: bytecode (browser engine)
output must match interp`. The bytecode-engine output carried **two extra leading lines**
(`one\nthree\n`) versus interp — i.e. a **ring-pipeline output-ordering** difference, not a wrong
result. **Unrelated to the diff** and **flaky, not deterministic**: the test passed **3/3** locally
back-to-back on the same commit.

**Family.** Same `build · test · fmt · clippy` gate and the same scheduling-nondeterminism shape as
I52/I53 — a ring pipeline's stage-completion interleaving is order-sensitive, and under CI load the
bytecode (browser) engine and the interp can drain the ring in a different order, so the early-exit
status lines and the piped stdout appear interleaved differently. Fix sketch: make the ring-pipeline
early-exit drain deterministic (flush/join the pipeline's stages in a fixed order before reading the
combined output), or have the test compare a set/sorted view of the lines rather than raw byte order.
Low priority (S4) until it recurs; logged now so it isn't rediscovered from scratch.

### I53 — `clone_caller::clone_caller_forks_the_caller_into_a_twin_that_returns_the_second_reply` (and `pid_mode_…`) intermittently fail the Linux `build · test · fmt · clippy` gate (S4, flaky CI) — surfaced 2026-07-30 on PR #539 — **FIX LANDED 2026-08-04** (`claude/i53-flake-issues-9wnzzw`)

**Symptom.** On PR #539 (a **docs + `#[ignore]`d-test + `workflows_src` only** change — zero lines in
`crates/svm-interp`) the `build · test · fmt · clippy` job failed with
`crates/svm-interp/tests/clone_caller.rs:255` — `assertion left == right failed: two i64 writes
reached the shared sink, left: 8, right: 16`. The expected single reply-write (8) was doubled (16),
i.e. the cloned twin's reply landed twice / a dedup lost a race. **Unrelated to the diff** (the PR
touches nothing in the svc/fiber/clone path) and **flaky, not deterministic**: the same test passed
**6/6** locally back-to-back on the same commit.

**Family.** Same crate and failure shape as I52 (`svc_serve_chain`) — the fork/serve scheduler path's
nondeterminism under CI load. Likely the same lost-wakeup/ordering class I52 root-caused; the
`clone_caller` twin-reply ordering wants the same fail-fast + wakeup-ordering scrutiny. Fix sketch:
audit `clone_caller`'s out-of-band reply injection for a reply that can be delivered to both the
original and the twin (or observed twice at the shared sink) when the two resume in the losing order.
Low priority (S4) until it recurs; logged now so it isn't rediscovered from scratch.

**Sighting 2026-07-31** (local, slice-8 branch `claude/interactive-embedding-review-tqgjri`): one
failure in a full `cargo test -p svm-interp` sweep, then **6/6 passes** rerunning the suite alone on
the same commit (a debug-path-only diff — nothing in the svc/fiber/clone path). Recurred, so it's
climbing the "until it recurs" bar; same shape, no new information beyond load-sensitivity.

**Sighting 2026-07-31 #2 + corrected root cause** (windows-latest, PR #564 — a fork slice-2 diff that
touches only `bind_child_manifest`, *not* the `clone_caller` path, which uses `cap.self.resolve`/direct
`cap.call`, no named imports). Same `clone_caller.rs:258` `left: 8, right: 16`. The earlier note read
this as the reply "doubled" — that is backwards: the assert is `assert_eq!(bytes.len(), 16, …)`, so
`left = actual = 8` (only **one** i64 write landed) and `right = expected = 16`. **The twin's write is
missing, not doubled.** Root cause: in *this test* the forking caller **is the root** (task 0), so
`run_with_host` returns the instant the original resumes past the fork and returns `100` — the orphan
**twin fiber may not be scheduled to its write before the run ends**, so the sink holds only the
original's 8 bytes. The manager-topology fork tests (`fork_manager` / `fork_import` / `c_fork`) do **not**
flake here — a manager `join`s the guest and the scheduler drains the twin before the overall run ends,
so all three deterministically observe 16 bytes. **Fix sketch (revised):** give this test a parent that
outlives the twin (the manager shape), or drain ready fibers before `run_with_host` returns, rather than
the original "dedup a double-delivered reply" guess (there is no double delivery). Still S4 — the real
fork path is deterministic; this is a root-is-the-forker test artifact.

**Sighting + corrected diagnosis 2026-07-31** (CI, PR #562 — a `svm-dap`-only diff): third
recurrence, same assert. **The original write-up misread the assert**: in `assert_eq!(bytes.len(),
16, …)` the failing `left: 8` is the *actual* — the shared sink held **one** reply, not a doubled
one. This is a **lost write**, the opposite failure: one of the two forked copies (original or
twin) didn't get its 8-byte reply into the shared stdout before the run finished. Revised
hypothesis: the twin `fork_parked_caller` creates is **detached** — the fixture's guest joins the
svc child but nothing joins the twin, so the twin's final `Stream` write races the root's
run teardown; under CI load the run returns first and the write is dropped with the task. Fix
direction accordingly: either run teardown drains still-runnable forked twins before returning
(engine semantics — decide against FORK.md §8), or the fixture joins the twin (its handle is
delivered to the handler) so the test stops encoding the race. The doubling/dedup sketch above is
withdrawn.

**Related manifestation 2026-07-31 — the `-EAGAIN`/`-ECHILD` serve-race in fork + wait (FORK.md
§8.6).** The same enqueue-before-park window has a second face on the servicer side, surfaced building
`reap` (`wait(pid)`): `svc_enqueue` makes a dispatch visible and wakes the server *before* the caller
registers its `CapReply` waiter, so a servicer that runs `clone_caller`/`reap` can find **no parked
caller** (`ticket_waiters` miss). `fork` already answers this with `-EAGAIN` (pid mode); `reap` must
answer it the same way (`ReapOutcome::Retry` → `-EAGAIN`) and **not** confuse it with a genuine
unknown-pid `-ECHILD`. The fix is guest-side and realistic: the `fork_then_wait` test retries both
`fork` and `wait` on `-EAGAIN` (`while ((s = wait(pid)) < 0);`), which converges and makes it **stable
0/50 under the full parallel `clone_caller` suite** where a one-shot form flaked ~1/15 with `left:
[-10]` (a raced `wait` mis-reported `-ECHILD`, or a raced `fork` handed the parent a bogus pid). This
is the same root class as I53 — not a new bug — and the retry idiom is the standing mitigation for the
serve/park race on **any** handler that needs its parked caller. Fully closing it would require the
serve protocol to register the caller's waiter before the dispatch is servable (still S4).

**Sighting 2026-07-31** (CI, `build · test (windows-latest)` on PR #571 — a `svm-dap` + browser-only
diff, nothing in the svc/fork path): `pid_mode_replies_the_twins_task_id_to_the_parent_and_zero_to_the_child`
got `left: [I64(-11)]` (a raced `fork` handed the parent `-EAGAIN` instead of the twin's `TaskId 3`) —
exactly the `-EAGAIN` fork-race face documented above, now on the `pid_mode` test rather than the
retry-guarded `fork_then_wait`. Unlike `fork_then_wait`, the `pid_mode` fixture does **not** retry on
`-EAGAIN`, so it flakes directly under Windows CI load; the standing mitigation (guest-side retry) or
the deterministic-serve fix would cover it. Same I53 root class, unrelated to the diff.

**Sighting 2026-08-04** (local, CALLS.md 4b.3 branch `claude/domain-communication-patterns-ypdwz0` — a
promotion/teardown diff touching nothing in the fork/clone path): one failure of
`pid_mode_replies_the_twins_task_id_to_the_parent_and_zero_to_the_child` in a full parallel
`cargo test -p svm-interp` sweep, then **passed on the immediate full re-run and 3/3 isolated** on the
same commit. Confirms the flake reproduces under **local** parallel load too (not only Windows CI),
and is load-sensitive not diff-sensitive. Same I53 root class; no new information beyond that.

**FIX LANDED (2026-08-04) — the two tests now use the FORK.md §8.6 `reap` idiom, closing BOTH serve/park
races; no product change (the real fork path was always deterministic under a `wait`ing parent — this was
a test-fixture race).** First **reproduced** it deterministically to confirm the diagnosis: hammering the
`clone_caller` binary (`--test-threads=8` under CPU load) failed **5/200** in exactly the two documented
faces — the two-reply test `left: 8, right: 16` (×3) and pid-mode `left: [I64(-11)]` (×2). Two distinct
races, per the sightings above:

- **Teardown-drain (both tests).** A fork twin is minted straight onto `runnable` with **no handle in
  anyone's table** — a detached daemon — and INVARIANT #6 says root completion abandons daemons
  (post-teardown sibling effects unspecified). The old fixtures had root `join` only the *original* `C`,
  so the twin's stdout write raced teardown and was dropped (`left: 8`). **Fix:** the original **reaps**
  its twin (`wait` verb, op 1 → `reap`, func 3 — the sibling `fork_then_wait` already used) before
  returning, so the run cannot end until the twin has finished (and thus written). The twin's
  deterministic `TaskId` is `3` (root 0, S 1, C 2, twin 3 — failed forks never consume an id, so the
  *winning* fork always mints 3).
- **Fork-park (pid mode).** `fork` can lose the enqueue-before-park window (server drains the dispatch
  before `C` registers its `CapReply` waiter); pid mode answers `-EAGAIN`, so `C` retries
  `while ((pid = fork()) < 0)`. This is what the `left: [-11]` sighting was.
- **Fork-park (two-reply).** The explicit two-reply `clone_caller(reply_orig, reply_twin)` has **no
  `-EAGAIN` signal** — on the lost race it *silently degrades to a single reply with no twin*. So the
  guest detects it structurally: after the `reply_orig` (100) return, `reap(3)` answers `-ECHILD` (no
  such twin) and `C` **re-forks**; a live twin instead makes `reap` retryable-`-EAGAIN` or delivers its
  status, never `-ECHILD`. (This second face was invisible in the original two-reply fixture and is why
  a reap-only fix still flaked ~6/400 until the re-fork was added.)

**Verified robust:** the rewritten tests passed **1100/1100** hammered iterations (`--test-threads=8`,
with and without background CPU contention) where the originals failed ~2.5%; full `cargo test -p
svm-interp` + `fmt` + `clippy` green. Interp-only, like every fork test. This retires the whole I53
family, including the `pid_mode` `-EAGAIN`/`-ECHILD` face and the §8.6 serve-race note recorded above —
the standing mitigation (guest-side retry) is now baked into the fixtures themselves.

**Sighting 2026-08-05 — the fix's blind spot: `crates/svm/tests/fork_manager.rs`** (CI, `build ·
test · fmt · clippy` on PR #607, run 30988982256 — a **peval-only** diff touching `svm-peval` /
`svm-run` / `svm-llvm` tests, nothing in the svc/fork path). `a_guest_forks_with_real_libc_and_both_copies_write_through_the_shared_memfs`
got `left: [I64(-11)]` vs `right: [I64(3)]` — the same `-EAGAIN` fork-park face, now on the
manager-topology **real-libc** test. This is exactly the gap the 2026-08-04 fix left open: that
rewrite baked the guest-side retry into the `svm-interp` fixtures only, but `fork_manager` lives in
the **`svm`** crate and its hand-written-IR guest (func 3) calls `fork()` **one-shot**
(`vr = cap.call 268435456 0 … vfork (varg)`, no `while (vr < 0)` loop), so it still flakes directly
under CI load. An earlier note (Sighting 2026-07-31 #2) wrongly generalized "the manager-topology
fork tests do **not** flake" — true for the *teardown-drain* face (a `join`ing manager drains the
twin) but **not** for the fork-park `-EAGAIN` face, which fires at the `fork()` call itself before any
twin exists. **Fix (deferred — not in this peval PR):** give func 3's guest the same `-EAGAIN` retry
loop the I53 rewrite gave the interp fixtures; deterministic-safe because failed forks never consume a
task id, so the winning fork still mints twin id 3. Unrelated to the diff; reran the job to unblock.

**FIX LANDED (2026-08-05, `claude/flaky-test-issues-3vdppg`) — the blind spot is closed.** Func 3's
guest now wraps its `fork()` in the `while ((vr = fork()) < 0)` retry loop the I53 rewrite baked into
the interp fixtures: the single `cap.call … vfork` moved into its own block that re-branches to itself
on `i64.lt_s vr 0` (an `-EAGAIN`/`-11` fork-park loss) and falls through to the write/return only on a
non-negative result. Deterministic-safe exactly as the sighting predicted — a lost fork never mints a
task id, so the winning fork still returns twin id 3 to the original and 0 to the twin, and the
sorted-stdout assertion (`{0, 3}`) is unchanged. The manager still `join`s the guest, so the
teardown-drain face stays covered; this only adds the fork-park retry that face never needed. Verified
green + hammered **60/60** (`fork_manager --test-threads=8` under load) where the one-shot form flaked
under CI contention. This retires the last documented face of the I53 family.

### I52 — `svc_serve_chain::a_handler_forwarding_to_another_server_completes` intermittently hangs the `build · test` job (macOS + Windows) to the timeout ceiling (S4, flaky CI hang) — surfaced 2026-07-29 on PR #504 — **ROOT-CAUSED & FIXED 2026-07-29** (fail-fast watchdog + the underlying lost-wakeup; `claude/ci-flakiness-review-fix-3xrmgg`)

**Symptom.** On PR #504 (a `svm-dap`/browser-only change) both `build · test (macos-latest)` and
`build · test (windows-latest)` were **cancelled** (not failed) after ~45 min: the single test in
`crates/svm-interp/tests/svc_serve_chain.rs`, `a_handler_forwarding_to_another_server_completes`,
logged "has been running for over 60 seconds" and never completed (macOS killed an orphan
`svc_serve_chain` process at cleanup). **Intermittent and unrelated to the diff** — Windows *passed*
this test in an earlier run of the same PR; the change touches only the debug adapter and browser
tests, nothing in the svc/fiber path. The Linux `build · test · fmt · clippy` lane (same test) went
green both times.

**Recurrences.** Same hang on **PR #509** (`windows-latest`, run 30470197189, chibicc `--emit-object`
cross-TU change — no serve-loop code; cancelled ~35 min) and on **PR #510** (`macos-latest`, the
posix→LLVM-on-ramp bridge — zero lines in `crates/svm-interp`; see the I50 recurrence note). Every
other job (Linux gate, `svm-llvm`, all wasm/differential lanes) was green in each run. Definitively a
base-branch flake, not the PR diff, in every sighting. *(This entry absorbs a duplicate that was also
filed as "I52" for the PR #509 sighting — the two were the same flake; consolidated 2026-07-29.)*

**Family.** Same svc-handler-forwarding × park surface as **I44** (freeze-on-quiesce, fixed
2026-07-24) and I40/I41 — a serve-chain rendezvous that can wedge under parallel-test load on the
slower/serialized CI runners.

**Where.** `crates/svm-interp/tests/svc_serve_chain.rs` (the forwarding-chain rendezvous), exercised
through the `svm-interp` svc serve loop.

**Root cause (definitive, 2026-07-29) — a lost-wakeup in the `svc.wait` re-park.** The chain is
`root → C1.fwd → C2.leaf`. A §3.6 handler runs as a **fiber of its serving vCPU** (slice 5b), so
`C1`'s `fwd` handler calling `leaf` **fiber-parks** and its serve loop moves on. When `C2` replies to
that handler's ticket, `Scheduler::cap_reply_or_stash` (Fiber arm) does two things under the
scheduler lock: `registry.wake_blocked(slot)` (marks the handler fiber `ParkedOn { woken: true }`)
**and** `svc_wake(C1)` (re-admit `C1`'s serve loop so it re-executes `svc.wait` and re-claims the
woken handler). The serve loop's `svc.wait` re-execution re-claims woken handlers from
`handler_parks` **before** it decides to park — but the `Blocked::SvcWait` park handler's
compare-and-park recheck tested **only `svc_queue.is_empty()`**, never "did one of my handler fibers
just get woken?". So if the reply lands in the window *after* the serve loop's `handler_parks` claim
saw the handler still-blocked and *before* the vCPU registered in `svc_waiters`, the `svc_wake` finds
no parked consumer and is **dropped**; the vCPU then parks with a woken-but-unresumed handler and
nothing left to wake it — `root` and both serve loops strand, and `worker_loop` sleeps on its idle
condvar forever (hence the 45-min ceiling; the multi-worker driver has no quiescence-deadlock
detector, unlike the deterministic explorer). Timing-sensitive, which is why it surfaced only on the
slower/serialized macOS + Windows runners under `cargo test --workspace` load. **Reproduced on Linux**
this review by widening that exact window with a temporary sleep — a stall-detector then fired
deterministically (all vCPUs parked, nothing runnable, no timers).

**FIX LANDED (2026-07-29) — observe the woken handler in the re-park.** The `Blocked::SvcWait` park
handler now parks only when the queue is empty **and** none of this vCPU's `handler_parks` slots is
already woken (`FiberRegistry::slot_woken`, a non-consuming peek). Because the reply holds the
scheduler lock across `wake_blocked` + `svc_wake` and this recheck reads under the same lock, the two
are serialized: either the recheck runs first and parks *before* the reply's `svc_wake` (which then
finds it in `svc_waiters` and wakes it), or the reply runs first and the recheck observes the woken
handler and re-admits instead of parking. The window is closed on every architecture (all
synchronization rides the scheduler `Mutex`, not memory-ordering assumptions). Each vCPU checks only
its own `handler_parks`, so the guard is correct for a multi-consumer domain too. The common park
path (empty queue, no woken handler) is unchanged.

**Fail-fast watchdog RETAINED (from PR #512).** The test still runs under a 60 s wall-clock watchdog
(`run_chain_with_watchdog`), converting any *future* re-strand into a fast red instead of a 45-min
runner burn. Defense-in-depth behind the real fix, not the fix itself.

**Residual hardening (optional, not blocking).** The multi-worker `drive`/`worker_loop` still lacks a
genuine quiescence-deadlock detector — if any *other* strand class ever arises it would hang rather
than `ThreadFault`. The deterministic explorer already does this ("`live > 0` but quiescent: a
join-deadlock"); generalizing it to the M:N driver (nothing runnable, no timers, no async-ring job in
flight, every worker simultaneously idle with `live > 0` ⇒ shut down with `ThreadFault`) would make
the watchdog redundant and cover the whole `svc_serve_chain`/`durable_concurrent_jit`/`serve.rs`
family. Deferred: it is delicate (false-positive risk of aborting a correct run), and the specific
bug that caused this flake is now fixed.

### I51 — bytecode `vcpu.tls` is per-`Vm`, not per-vCPU: a fiber that migrates across workers reads a stale TLS word (S3, multi-worker only) — recorded 2026-07-28 landing the JACL-in-browser `vcpu.tls` lowering

**Symptom.** `vcpu.tls.get` must return the word of the vCPU *currently executing* the op
(§12; `svm-ir` `Inst::VcpuTlsGet` — "after a fiber migrates between vCPUs `get` returns the
new vCPU's word"). On the **bytecode engine** the TLS word lives on the running `Vm`
(`bytecode.rs` `Vm::tls`), seeded to the vCPU's dense id and re-seeded for a `thread.spawn`ed
task. A **fiber** (`cont.new`) is a separate `Vm` whose `tls` defaults to `0`, so a fiber
running on worker N reads `0` instead of N's word; and a fiber that migrates from worker A to
worker B (D57 / work-stealing) keeps A's word instead of reading B's. The tree-walker is
correct here (its `tls` is on the vCPU it steps). A guest whose per-CPU allocator keys on
`vcpu.tls` (e.g. JACL's `heap_gc.c` per-worker regions) would allocate into the wrong worker's
arena under real multi-worker execution.

**Not browser-affecting.** The browser tier is single-OS-thread cooperative with one worker
(JACL `POOL_WORKERS=1`), so every read is a faithful `0` — this is latent only for native
multi-worker (`drive_parallel`, or cooperative `drive` with spawned workers running fibers).
That is why it was deferred rather than fixed on the JACL-in-browser path.

**Where.** `crates/svm-interp/src/bytecode.rs`: `Op::VcpuTlsGet`/`VcpuTlsSet` read/write
`self.tls` (the active `Vm`); the seed is on `Vm::new` + `drive`'s `Spawn` arm.

**Fix sketch.** Make the TLS word per-vCPU rather than per-`Vm`: hold it on the vCPU/task
(`VTask`) and have `Op::VcpuTlsGet`/`Set` operate on *that* word — thread a `&mut i64` for the
executing vCPU's TLS through `Vm::resume` (~10 call sites) so it is correct regardless of which
fiber is active and survives fiber switches within a `step_vcpu`. Cover with a cooperative
multi-worker differential test (spawn workers, set distinct TLS per worker, migrate a fiber,
assert `get` tracks the hosting worker) matched to the tree-walker.

### I49 — the playground's `chibicc.svmb` was never committed, so the C-compiler card 404'd (S3) — the I26/I42 asset-shipping class again — surfaced 2026-07-27 (`fetch ./assets/chibicc.svmb: 404`) — **FIX LANDED** (`claude/chibicc-playground-status-7n6eh0`)

**Symptom.** The playground's "C compiler (chibicc → SVM)" card fails with
`fetch ./assets/chibicc.svmb: 404 — run \`node build-onramp-assets.mjs\` to generate it`.

**Where.** `browser/.gitignore` ignores `/web/assets/*.svmb` with explicit `!` un-ignore exceptions
for every committed playground asset (`hello_c`, `gradient`, `bounce`, `life`, `mandelzoom`,
`qjs_repl`) — but **not `chibicc.svmb`**. Step-5 landed the card + the `build-onramp-assets.mjs`
wiring but never committed the artifact, and that build is **fail-soft** (skipped when clang/llvm-18
is absent or the build hiccups). So the card only worked if the deploy-time build happened to
succeed; there was no committed fallback, and any failure silently shipped a 404 — locally the card
never worked out of the box at all.

**This is the I26/I28/I42 class** (a Pages deploy shipping a playground missing an asset, fail-soft
build masking it). **Fix:** commit `chibicc.svmb` in-tree behind a `!/web/assets/chibicc.svmb`
gitignore exception, exactly like `qjs_repl.svmb` — the build script still rebuilds it in place when
the toolchain is present. Built asset: 392786 bytes, 333 funcs, verifies + bytecode-compiles;
compiles a C source to SVM IR on the bytecode engine (the browser's engine).

**Class guard added (2026-07-27):** `browser/check-play-assets.mjs`, driven from `web/play.js` (the
single source of truth for referenced assets), wired into two workflows (`workflows_src`, pending
copy-over): a `playground-assets` PR job asserts every referenced asset is committed or declared
deploy-built (catches a card referencing an unaccounted asset — the static half of this class), and
a `pages.yml` `--site` step asserts every referenced asset is actually present in the assembled
`_site` before publish (catches a fail-soft build dropping a required asset — the I26/I42 half),
with a `MAY_BE_ABSENT` carve-out for DOOM's externally-mirrored WAD. New cards are covered
automatically. This closes the residual guard gap I26 named.

### I50 — CI flake: the `durable_concurrent_jit` binary fails on macOS in two modes (S4) — seen 2026-07-27, PR #455 runs 30263159591 + 30266760876

**Symptom — two distinct nondeterministic modes on `build · test (macos-latest)`, same binary:**
- **Mode A — SIGSEGV on exit** (run 30263159591, commit `9de4926`): all 15 tests in
  `durable_concurrent_jit` print `... ok`, then the process crashes on teardown —
  `process didn't exit successfully: durable_concurrent_jit-… (signal: 11, SIGSEGV: invalid memory
  reference)`. A test-harness/teardown fault, not an assertion.
- **Mode B — wake-race assertion** (run 30266760876, commit `5b8faa4`):
  `durable_concurrent_jit.rs:1048` `assertion left == right failed: the root's re-issued atomic.wait
  parked and was woken by the sibling's re-issued notify on thaw` (`left: 1, right: 0`) — the
  sibling's re-issued `notify` didn't land the wake before the assertion read the counter.

**Why it's a base-branch flake, not the PR (definitive).** PR #455 (the Tcl on-ramp target) touches
only `svm-llvm` (workspace-excluded), `crates/svm-run/demos/tcl/` (C/sh/md, not compiled by this
suite), `browser/`, and docs — **zero lines in `crates/svm`**. Mode A failed on the PR's *first*
commit, which adds only `demos/tcl/` + an `#[ignore]`d test + docs and therefore *cannot* affect
`crates/svm`; **20 of the 21 jobs were green** on that commit. Two different failure modes on
commits that don't touch the code ⇒ a pre-existing macOS-flaky binary in the branch base (the #450
merge), same family as **I44** (freeze × parked `svc.wait`).

**Action.** Re-run the macOS job (a flake — expect green). Don't churn CI chasing it from an
unrelated PR. Owner-side follow-up (the durability subsystem, not a Tcl-PR concern): Mode B is a
thaw-time wake-ordering hole the I44 clamp doesn't cover (sibling-notify-on-thaw); Mode A is a macOS
teardown SIGSEGV in the `durable_concurrent_jit` harness — both want a macOS-hammer repro
(`stress-ng`-style loop over the full binary) to localize, as I44 needed the full-binary hammer.

**Recurrence 2026-07-29 (PR #510, the posix→LLVM-on-ramp bridge).** `build · test (macos-latest)`
hung in `svm-interp/tests/svc_serve_chain.rs::a_handler_forwarding_to_another_server_completes`
("running for over 60 seconds" → job cancelled at the 45-min cap); the other 20 jobs were green
(Linux `build · test · fmt · clippy`, `svm-llvm`, windows, all differentials). Same I44 family (a
service-point handler forwarding — the fixed root→C1.fwd→C2.leaf deadlock's macOS scheduling twin,
now intermittent). Definitively base-branch, not the PR: #510 touches only `svm-posix`/`svm-run`/
`svm-llvm`/docs — **zero lines in `crates/svm-interp`**, so that test binary is byte-identical to
main. Re-ran (expect green).

### I45 — `megabench` example's `chase`/`chase_rand`/`fnv`/`fma`/`vsum` kernels no longer parse (S4) — surfaced 2026-07-25 measuring bytecode-vs-JIT — **FIX LANDED** (PR #444)

`cargo run --release --example megabench -p svm` panics after the first four kernels
(alu/call/call_indirect/mem) with `ParseError("expected RBrace, found Ident(\"binit\")")` at
`megabench.rs:33` — the `chase_src` generator emits a token the text frontend no longer accepts, so
the memory-latency (`chase`/`chase_rand`), FNV, FMA, and vector-sum rows are silently lost. Not
CI-gating (it's a dev example, not a gate), but it blinds the cross-engine A/B exactly where the
INTERP_PERF Phase-5 work needs memory + SIMD + pointer-chase coverage. **Fix is the Phase-5 prereq**
(INTERP_PERF.md "Phase 5"): repair the kernel sources to current text syntax before measuring 5a/5c.

### I44 — freeze-on-quiesce could fire multi-worker and strand a subset in `svc.wait` (S2, intermittent CI hang) — **FIX LANDED 2026-07-24** (§13.4 4c-bis branch)

**Symptom.** PR #437's `build · test · fmt · clippy` + `build · test (macos)` jobs both hit their
`timeout-minutes` ceilings (cancelled, not failed): `a_nested_two_server_subtree_freezes_on_quiesce
_and_thaws_still_serving` hung indefinitely. Reproduced locally ~2/12 only when the whole `serve.rs`
binary ran (all tests in parallel threads, under load) — never in isolation, which is why the
per-crate gate missed it.

**Root cause.** The durable "serialize onto one worker" clamp is gated on `durable_load_dstate(0)
!= NORMAL`. A **freeze-on-quiesce** run starts `NORMAL` (it runs normally, then freezes the instant
it would block on `svc.wait` parks), so the clamp didn't catch it. A subtree that spawns a child
(2+ live vCPUs) then spawns a *second* worker thread (`maybe_spawn_worker`), and the freeze-on-
quiesce trigger — a whole-run stop-the-world that must observe *every* domain parked atomically —
could fire on one worker while a domain was still executing on the other: it drained the parked
subset, consumed its one-shot arm, and left the rest parked in `svc.wait` forever.

**Fix.** Extend the clamp: a freeze-on-quiesce-armed run (`durable_freeze_on_quiesce()`) also
serializes onto one worker, so the quiesce check sees the true global parked state. Hammered 20/20
clean on the full binary. (Lesson for the harness: per-crate `cargo test -p X` runs one binary's
tests in parallel but doesn't reproduce cross-binary load; the flake needed the full-binary hammer.)

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

**Teardown now uniform (2026-07-24):** the domain-fatal rule this issue assumes is enforced
identically on all engines as of the domain-lifetime decision (DESIGN.md §12 "Domain lifetime
& teardown", INVARIANTS #6): exit/trap tears the whole domain down immediately — parked
threads included — and root completion ends the batch run with daemons abandoned. Previously
the interpreter only surfaced a spawned thread's trap at `thread.join` and the JIT never woke
parked waiters to observe a trap (the jacl timed-wait regression, whose reported JIT/interp
divergence was this teardown gap, not the timed wait itself). Recorded residues of that
build, none consumer-blocking: the bytecode **`drive_parallel`** opt-in scoped-threads mode
still rides the `MAX_WAIT` clamp for abandoned waiters (no committed fixture exercises it);
the debug-stepping bytecode scheduler and browser vCPU orchestration keep pre-decision
semantics (observability tier); the JIT's shared trap cell can in principle be reset by a
still-running sibling's successful `cap.call` in the teardown window (a pre-existing hazard
class for real traps too); and `ticket_waiters`' bare per-callee ticket keying predates the
teardown sweep and is inherited by it.

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

**Timed `svc.wait` BUILT (2026-07-24)** as part of the I39 multi-consumer rung (it turned out
to be that rung's prerequisite — the consumer wind-down primitive; see the I39 rung-3 block for
the as-built): op 10's optional single arg is a timeout in ns; a deadline that fires with
nothing served returns `0`. Oracle-only (the fast backends' serve veto treats the timed form
as a park seam). The sub-queue/fairness half of this issue remains open.

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

**Rung 3 BUILT on the oracle (2026-07-24) — multi-consumer `svc.wait`, plus the wind-down it
forced and a latent settle race it flushed out.** Three pieces:

*(a) Multi-waiter `svc_waiters` + wake-all.* Exactly the sketched substrate change:
`Sched::svc_waiters` became multi-waiter per domain key (a `Vec` of parked vCPUs — the old
single-slot map silently **displaced** a second parker, dropping a live vCPU: a latent hang for
any svc+threads module on the oracle), and a wake re-admits **all** of a domain's parked
consumers. Wake-all is the deliberately boring form: the wake path knows only the domain key,
never which vCPU owns a parked handler (`handler_parks` is per-vCPU), admission is race-free
under the powerbox lock, and a consumer that finds nothing runnable re-parks via its rewound
`svc.wait`.

*(b) Timed `svc.wait` (the I38 sketch, pulled in as the wind-down primitive).* Hammering the
first test draft proved multi-consumer is unusable without it: consumers **work-steal** (any
sibling may serve every dispatch), so a spare consumer parked in an untimed `svc.wait` can
never exit — it stranded the child's `thread.join` and hung the run. Op 10 now takes an
optional single arg (timeout in ns; `< 0`/absent = forever, today's form byte-identical): the
park registers a deadline in a new `Sched::svc_timers` heap; a fire re-admits the still-parked
consumer with `Pending::SvcTimeout`, whose rewound `svc.wait` admits anything that raced the
timer and then returns its count — `0` on a pure timeout — instead of re-parking. Timed form
is **oracle-only**: `serve_qualifies`/the bytecode compile veto treat it as a park seam (both
fast backends decline the module), and the JIT cap-thunk intercept lets it fall through to the
generic probeable `-EINVAL`.

*(c) A pre-existing settle/park TOCTOU (slice 5b), found by the hammer.* The serve settle was
two-step: `cap_reply` (miss — caller not parked yet), scheduler lock released, then the cell
insert under a separate powerbox lock. A caller could park exactly in between — its park-time
cell probe empty, its `ticket_waiters` entry never woken (no second reply ever comes) —
stranded forever with the value in the cell. Multi-consumer's spurious wakes widened the
window, but the race is reachable single-consumer too. Fix: `Scheduler::cap_reply_or_stash` —
wake-or-stash under ONE scheduler lock (lock order scheduler→powerbox, matching the park
handler and the fiber early-probe), used by all three serve-arm settle sites (result, arity
`-EINVAL`, quota `-EAGAIN`).

Pinned in `svm-interp/tests/svc_multi_consumer.rs`: two pollers split one queue (counts sum,
pure-handler cells exact); a pure timed-wait timeout returns 0 (fast entry declines and falls
back identically); and two timed-`svc.wait` consumers inside a §14 serving child serve a live
caller's three sequential calls (`add`, `add`, `finish`-sets-the-flag wind-down protocol)
across repeated interleavings — hammered 60× clean where the old map/race hung within ~5 runs.
The fast backends' serve veto still declines svc+thread modules (pinned), so the oracle is the
only backend running these shapes — the "pinned in the differential before the JIT loop"
prerequisite is met; the native serve loops pick the rung up when a consumer demands it. The
threaded-handler discipline (atomics/locks, per D22) is the opt-in contract; the two-pollers
test's handler is pure for exactly that reason.

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

### I40 — an unclaimed svc reply outlives a dead caller: `svc_results` entries are never garbage-collected (S4) — **FIX LANDED 2026-07-29** (`claude/ci-flakiness-review-fix-3xrmgg`)

**Where:** a completed dispatch whose caller didn't (or can't) claim the reply parks the value in
`Host::svc_results` keyed by ticket. If the caller died between enqueue and claim, nothing sweeps
the entry — a long-lived serving domain accumulates orphaned tickets. Bounded by call volume, not
by live state. The teardown code named it outright ("the callee's eventual reply finds no waiter and
stashes, **harmlessly**") — harmless for correctness, but an unbounded slow leak as child callers
come and go against a surviving server.

**FIX LANDED — drop the reply at its stash site instead of sweeping the callee.** Rather than chase
`svc_results` (per-callee, keyed by bare ticket, with no caller attribution — nothing to sweep *by*),
the fix records the *caller's* death and drops the reply when it arrives. A scheduler-side
`Sched::orphan_tickets: BTreeSet<(callee_id, ticket)>` is populated wherever a caller with an
in-flight call is reaped — `teardown_domain`'s caller sweep (the parked case, the one the old comment
described) and the `CapReply` park gate (the rarer enqueue→park-window case) — and consumed at
`Scheduler::cap_reply_or_stash`'s stash arm: a recorded ticket is dropped, not stashed. Tickets are
unique per run (monotone `svc_next_ticket`), so a recorded key can **never** collide with a live
dispatch — the fix cannot drop a live caller's reply. `orphan_tickets` is self-bounded: a callee that
itself dies sweeps its own entries (their replies can't come), and each entry is removed the moment
its reply lands. Scheduler-local — **no** change to the public `Host` API or the durability snapshot
format. Scope is the tree-walker scheduler; the JIT/embedder serve loop has no cross-domain
ticket-parked callers yet (I36 slice 3), so no orphan arises there. Pinned by
`svm-interp` unit tests `orphan_reply_tests`: the consume invariant (dead-caller reply dropped, live
reply stashed, cross-callee key isolation) and the populate + self-GC path (`teardown_domain` records
a dying caller's outstanding ticket and sweeps orphans awaiting the now-dead domain).

### I35 — NOT a miscompile: a `--child-entry` `main`'s argv-relocated frame was rounded up to the next 16 KiB page and collided with a SharedRegion the program mapped there; a local array on that frame read back garbage (S3) — seen 2026-07-23, building the c_shell `__stage` ring runner — **FIX LANDED 2026-07-27** (`claude/chibicc-playground-status-7n6eh0`)

**The original diagnosis was wrong.** It was filed as "the indexed post-increment store
`regs[nregs++] = h` miscompiles." It does not — chibicc emits correct IR for that store. Proof
(reproductions, 2026-07-27): the identical loop in a **powerbox** `main(int, char**)` (a *local*
array + post-increment) runs correctly on both engines; the same program spawned as a `--child-entry`
child in a carve **that matches its declared window** also runs correctly; and in the failing runner
both the post-increment form **and** an explicit-slot-pick form fail — so the store shape was never
the variable. Storage class was: `static` worked, `local` failed.

**Real root cause (frontend, `codegen_ir.c` `emit_start`).** For a `main(int, char**)`, `_start`
builds `argv[]` at `data_end` and relocates `main`'s frame above it. It rounded `main_sp` **up to the
next full `POWERBOX_ARGS_END` (16 KiB) page**. The `__stage` runner's writable globals (a
`window_pin_[50000]` pad that forces its declared window to 256 KiB) push `data_end` to 114688, so
`main_sp` rounded to **exactly 131072** — the offset the runner then `__vm_region_map`s its ring into
(`upper half of the 256 KiB window`). So a **local** `regs[]` lived on the frame at 131072+, aliasing
the ring, and read back garbage once the ring was mapped/used; a **static** `regs[]` sat in the low
globals region, clear of the frame, and survived — which is exactly why the static workaround worked.
chibicc's IR was correct throughout; the frame was merely parked on top of a region the program owns.

**Fix:** relocate `main`'s frame **just past the argv array (16-byte aligned)** instead of rounding up
a full page — the array is only `(argc+1)*8` bytes and the frame grows upward away from it, so 16-byte
alignment is all the ABI needs (`data_end` is already page-isolated from read-only globals, so D40
still holds). This keeps the frame adjacent to `data_end`, below whatever offset a program maps
regions at. The runner's grant-discovery loop is restored to its natural shape (local array +
`regs[nregs++] = h`) in `crates/svm-run/demos/shell/stage_runner_main.c` (the `__stage` runner
`crates/svm/tests/c_shell.rs` `include_str!`s) and now regression-guards the fix (it fails if the
full-page rounding returns). Validated: full `c_frontend` + `c_shell` + `stage1_*` suites green.

**Latent parallel (not changed):** `svm-llvm`'s `synth_start_argv` (the on-ramp frontend) does the
same page-alignment, deliberately tied to D40. It is not currently triggered — on-ramp `main`s don't
map SharedRegions at colliding offsets — and its page-alignment is redundant-but-safe (its data stack
is already page-isolated). Left untouched to keep this fix off the on-ramp artifact path; align it too
if a child-entry on-ramp program ever hits the same collision.

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

**Recurred 2026-07-28 (run 30395295933, PR #488):** the `real-browser` "Install Playwright +
Chromium" step hit its **10-min cap** — this time the stall was the `--with-deps` **apt font
download** (`Fetched 21.1 MB in 9min 55s (35.4 kB/s)`, a wedged Azure mirror), not the npm/CDN
half. The timeout-minutes mitigation worked as intended (failed fast into a re-run instead of
pinning the runner); every other job on the commit was green and the change was a pure
`svm-interp` scheduler edit with no browser surface. Reinforces the still-open residual: **cache
or pre-provision the apt font deps** (or split `--with-deps` off the timed step) so a slow distro
mirror can't eat the budget. Cleared by a re-run.

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

**Sightings 4–5 (2026-07-24, PR #427 runs 30089414778 + 30091022655) — WITH the `-j 2` cap;
escalation taken.** Both runs died the identical death (~56–59 s into `cargo test --workspace
-j 2`, step frozen "in_progress", runner agent lost, logs never uploaded, fmt/clippy/build green;
main green on the same day) — and the branch had added **two new heavy-link `svm` test binaries**
(`jit_svc.rs`, `revocation_errno.rs`, each pulling svm-run → svm-jit → Cranelift), which raised
the concurrent-link peak past whatever headroom the cap had left. Two-pronged escalation:
(1) **in-tree** — the new tests were merged into existing binaries (`jit_cap.rs`, `pipeline.rs`),
so the branch adds zero new link targets; hold that line — prefer extending an existing heavy
test binary over adding a new one. (2) **durable, pending owner copy-over** — the `check` job
gains `CARGO_PROFILE_TEST_DEBUG: "0"` in `.github/workflows_src/ci.yml`: debug info for the
Cranelift/Wasmtime-sized dep graph is the dominant per-link memory term; dropping it keeps
symbol-name backtraces while cutting link memory by multiples. If a death lands with BOTH in
place, the remaining lever is `-j 1` on the test step (or self-hosted/larger runners).

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

### I42 — Rare macOS-CI `Bus error: 10` (SIGBUS) at a test-binary launch under `cargo test --workspace` (S4)

<!-- Renumbered I24 → I42 (2026-07-24): I24 collided with the (now-retired) LLVM-version-pin
     issue that had also held I24, so this open entry takes the next free id, I42. (It had earlier
     moved I21 → I24 on 2026-07-15, which only relocated the collision.) -->


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

**Follow-up (2026-08-04): the bench lane was red for yet another *deterministic* reason — the harness
no longer compiled, so the perf signal has been dead again (not drift).** The nightly `bench` job
failed at `cargo build` with `error[E0061]: this function takes 12 arguments but 11 were supplied`
at `bench/src/threads.rs:71` — `svm_jit::CompiledModule::compile` gained a 10th parameter
`fast_resolver: Option<FastCapResolver>` (the §9/D45 hot-`cap.call` specializer), but the bench
caller was never updated. Same root class as the "standalone workspaces" section below: `bench/` is
its own cargo workspace, so `cargo build --workspace` (the usual pre-push check) never compiles it and
the break sailed through — this time from an **svm-jit** API change, not svm-interp/svm-run. **Fixed:**
added the missing `None, // fast_resolver` argument (the in-tree idiom at 7 other call sites); verified
the bench workspace builds clean (`cargo build --release`, 4m09s). The lane's `--check` compare is once
again reachable. Baseline-regen for the five MISSING kernels remains the one open I17 item.

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

4. **GitHub archive download served a non-gzip response.** Run 30108936792 job 89533205009
   (Jul 24, `embench differential` job, setup step): `curl -sSL …/embench-iot/…/master.tar.gz |
   tar xz` failed with `gzip: stdin: not in gzip format` → exit 2, before any repo code ran —
   codeload returned an error/rate-limit page instead of the tarball (the `-sS` flags hide the
   HTTP status and `curl | tar` can't check it). Re-run clears it.
   *Mitigation:* add `--retry 5 --retry-all-errors -f` to the curl (fail on HTTP errors and let
   curl retry), or cache the embench checkout keyed on a pinned ref instead of re-fetching
   `master` every run (pinning also removes a reproducibility hole). Workflow-file change
   (`workflow` scope), so a maintainer applies it — mirrored in `.github/workflows_src/`.

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
paramless entry for *every* guest, regardless of cap count or `main` signature. (The S15
`synth_powerbox_start*` family this originally named was deleted in IMPORTS.md phase 4; the
frontend-neutral public equivalent is now `svm_ir::synth_manifest_start`, and svm-llvm's private
`synth_start` is already paramless.) Once every emitted guest is by-name, drop the
positional branch from `grant_onramp_caps` and the `arity > 5` guard in `onramp_exec`, collapsing the
host to a single by-name grant; regenerate the committed `.svmb` fixtures/assets so they're by-name
too. `svm-run`'s `grant_caps` (which also still keeps a positional branch) can drop it in the same pass.

**Partly done (2026-07-24, `claude/doom-asset-generation-6zi7k6`):** the "regenerate the committed
fixtures/assets" half is now current. `hello_c`/`hello_onramp` (309 → 111 B) and `life` (1644 → 1376 B)
had drifted from what `svm-llvm-translate --host-page 65536` emits today; `gradient`/`bounce`/
`mandelzoom`/`fsread` rebuild byte-identical. The drift was **body encoding only** — imports and
exports are identical across old and new (`write` / `vm_map`; `_start`, `main`, `+ tick`), so this says
nothing either way about the paramless-vs-positional question above, which is still open. The
regenerated pair passes `onramp`, `reactor`, `shared_reactor`, and `reactor_fs`. `web/assets/qjs_repl.svmb`
was stale too (4319380 B vs the pages run's 4318992 B) and is now regenerated as well, once I43 made
openlibm fetchable again.

### I42 — the Doom example vanished from the published playground: its single WAD mirror started 404ing, and every layer swallowed it (S3) — surfaced 2026-07-24 by `fetch ./assets/doom.svmb: 404` in production — **FIX LANDED** (`claude/doom-asset-generation-6zi7k6`)

**Root cause retired (2026-07-27):** the shareware `doom1.wad` is now **vendored in-tree**
(`crates/svm-run/demos/doom/doom1.wad`, v1.9, md5 `f0cefca49926d00903cf57551d901abe`) and staged
directly — no WAD fetch, so no mirror can drop it. `build-onramp-assets.mjs`'s `ensureWad()`/`WAD_MIRRORS`
are gone. The `--site` reachability gate (I49) now *requires* `doom1.wad` (removed from
`MAY_BE_ABSENT`); only `doom.svmb` (id's engine source, still fetched-and-built) stays fail-soft. The
mirror-list fix below is the prior, superseded mitigation.

**Where:** `browser/build-onramp-assets.mjs` → `ensureWad()`. The shareware IWAD was fetched from a
**single** URL, `https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad`, which now returns
**404** (verified 2026-07-24). `curl -sfL` is silent, the `catch` was empty, and the loop had exactly
one mirror — so the outage produced no output at all.

**Symptom.** The playground's Doom example 404'd on `./assets/doom.svmb` in production, while the
`pages` workflow stayed **green** on every run. The Doom *module* builds fine — the pages log shows
`built /tmp/doomgeneric_cache/bc/doom.svmb (784303 bytes); exports: main 65 / tick 66` — but
`copyFileSync` into `web/assets/` is gated on `doomSvmb && doomWad`, so a missing WAD dropped the
module too. The one line printed was the catch-all
`– doom skipped (no toolchain, or the source/WAD fetch failed — offline?)`, immediately after a
successful module build, which pointed diagnosis at the toolchain rather than at a dead mirror.

**This is the I26/I28 class again** — a Pages deploy that ships a playground missing an asset without
ever going red. I26 was a copy-glob dropping files; I28 was an untested asset; this is a *build input*
disappearing. Same failure mode: local dev has a warm `/tmp/doomgeneric_cache`, so nobody sees it.

**Fixed:** `ensureWad()` now tries **four** mirrors and reports each failure with host + reason:
`raw.githubusercontent.com/Akbar30Bill/DOOM_wads` (canonical shareware v1.9, md5
`f0cefca49926d00903cf57551d901abe` — the same transport `fetch.sh` already falls back to), plus the
official idgames archive and two of its mirrors (`gamers.org`, `youfailit.net`, `ftpmirror1.infania.net`)
carrying the shareware v1.8 IWAD **gzipped** (decompressed in-process via `node:zlib`). The IWAD magic
is checked **after** decompression, so a 404 body or captive-portal page still can't masquerade as the
WAD. The skip line now names which half failed (`module build` vs `doom1.wad fetch`). Verified from a
cold cache: `✓ doom.svmb (0.75 MB) + doom1.wad (4.00 MB)`, and `doom_reactor` boots and renders
300/300 frames (into demo1 gameplay) over the fetched WAD.

**Residual (not fixed here):** the skip is still **fail-soft by design** — an offline build must be
able to omit Doom rather than fail — so four dead mirrors would once again ship a Doom-less playground
green. The general guard is I26's fix sketch (b): a post-assemble gate that fails the deploy when a
`play.js` example's asset is absent from `_site`. That needs `workflow` scope, so it goes through
`.github/workflows_src/`. A cheaper stopgap is an env gate (`SVM_REQUIRE_DOOM=1`) that turns the skip
into a hard error in the Pages job only.

### I43 — openlibm was fetched from a **single** GitHub archive URL, and that endpoint is gated on some networks — the third instance of the one-source-fetch class (S3) — surfaced 2026-07-24 while regenerating `qjs_repl.svmb` — **FIX LANDED** (`claude/doom-asset-generation-6zi7k6`)

**Where:** three independent sites, all with the same single URL
`https://github.com/JuliaMath/openlibm/archive/refs/tags/v$VER.tar.gz`, all sharing the
`/tmp/svm_openlibm_cache` tree:
`ensureOpenlibm()` (`browser/build-onramp-assets.mjs`), `fetch_openlibm()`
(`crates/svm-llvm/tests/translate.rs`), and `crates/svm-run/demos/postgres/link_shims.sh`.

**Symptom.** GitHub's **archive** endpoint answers **403** on networks where `github.com` git and
`raw.githubusercontent.com` are both fine — so this is not "offline", it is one endpoint being gated.
Every consumer misread it as offline and degraded: the QuickJS rebuild skipped silently (leaving a
stale committed `qjs_repl.svmb`), the `libm_bundled_vs_native` differential skipped, and `link_shims.sh`
hard-failed with `OPENLIBM FETCH FAILED`.

**This has happened before.** Same shape as I42 (doom's one WAD mirror 404ing), and
`crates/svm-run/demos/doom/fetch.sh` *already* documents this exact endpoint split — "the GitHub
archive tarball (fast; what CI uses), else a per-file fetch from raw.githubusercontent.com (works
where the archive host is gated)". openlibm simply never got the same treatment.

**Mirrors — what actually works** (probed 2026-07-24 from a gated sandbox):

| source | result |
|---|---|
| `github.com/.../archive/refs/tags/v0.8.5.tar.gz` | **403** (gated; works in CI) |
| `codeload.github.com`, `api.github.com` tarball/tree | 403 |
| `git clone --depth 1 --branch v0.8.5 https://github.com/JuliaMath/openlibm` | **works** |
| `raw.githubusercontent.com/JuliaMath/openlibm/v0.8.5/<path>` | **works** (per-file) |
| `cdn.jsdelivr.net/gh/JuliaMath/openlibm@v0.8.5/<path>` | **works** (per-file, byte-identical) |
| `data.jsdelivr.com/v1/packages/gh/JuliaMath/openlibm@v0.8.5` | **works** (file listing, for a per-file walk) |
| Debian pool `openlibm_0.7.0+dfsg.orig.tar.xz` | reachable but **unusable** — wrong version, DFSG-stripped |
| Gentoo distfiles, `cache.julialang.org`, archive.org | unreachable / absent |

**Fixed:** all three sites now fall back to a **shallow tag clone** (`git clone --depth 1 --branch
v$VER`) when the archive fails, and each says which mirror failed and why instead of swallowing it.
The clone is tag-pinned to the same commit (`v0.8.5` = `db24332`), so the sources are identical to the
archive's; unlike a per-file walk it needs no file list kept in sync with whichever sources a given
consumer compiles (37 for QuickJS, 18 for the Postgres differential).

**Verification:** with the archive gated and the cache wiped, `build-onramp-assets.mjs` takes the clone
path and emits `qjs_repl.svmb` at **4318992 B** — byte-for-byte the size the pages run produces, i.e.
the mirror reproduces the archive build exactly. The module runs: openlibm-backed `Math.sqrt(2)` →
`1.4142135623730951` and `Math.log(Math.E)` → `1`, which is the precise surface openlibm supplies.

**Residual:** per-file `raw.githubusercontent.com` / jsDelivr (the table above) is the known next lever
if git-over-https is ever gated too — that is the shape `demos/doom/fetch.sh` already implements.

---

### I43 — child `poll` (op 9) is a **WNOHANG** probe; its *terminal* status is backend-portable — **RESOLVED 2026-07-28**: contract clarified + differentially pinned (`claude/svm-ir-wasm-comparison-y0lkbn`)

**What.** A crashing command must not crash the shell, so the shell reaches for `poll` (op 9: is the
spawned child done yet?) instead of an unconditional `join`. The original concern: `poll` immediately
after a *synchronous* spawn reads differently across backends — the tree-walk interpreter runs the child
**lazily** (M:N scheduler defers it), so an immediate `poll` reads `0` (still running); the JIT runs it
**eagerly** on its own OS thread, so an immediate `poll` may already read `1`/`2`.

**Resolution — this is the *defined* semantics of a non-blocking probe, not a divergence.** `poll` is
**WNOHANG**: `0` ("not done yet") is a valid answer at any time on any backend — a caller does not
control how many `0`s it sees before the child is scheduled to completion. Making the interpreter
"eager" *cannot* converge the immediate poll for the deterministic single-worker configs anyway (the
JIT is a 1:1 OS-thread executor; a single-worker interp run has no thread to run the child ahead of the
parent). The **portable idiom** is to loop `poll` (yielding the worker between probes) until it is
non-zero; the **terminal** value that loop reaches is identical across backends — `1` for a returning
child, `2` for a trapping one. That terminal convergence is now pinned by
`crates/svm/tests/lifecycle_poll_convergence.rs` (returning-child and trapping-child cases, interp vs
JIT), where before it lived only in this note and the interp⇄JIT fuzzer never exercised the poll+spawn
shape.

**Still open (separate, guest-personality):** the `$?` = 128 + signal exit-code mapping for a
signal-killed child is a shell/guest convention, not a substrate contract — tracked under STAGE1.md
crash-handling, out of the substrate `poll` scope resolved here.

**Owner / plan:** RESOLVED as a contract clarification + differential pin. STAGE1.md "Known caveat"
updated to point here.

---

### I45 — fiber futex waits on the **secondary bytecode drivers** still park the whole vCPU (S3) — split out of the jacl timed-wait fix (2026-07-25)

**What.** The jacl timed-wait fix aligned the three svm-run backends on the §3.6 slice-5a
fiber-park contract for `memory.wait` inside a fiber (tree-walk scheduler, bytecode *cooperative*
driver `drive`, Cranelift JIT futex thunk — pinned by `svm/tests/fiber_timed_wait.rs`). Two
bytecode entry paths outside that routing still run the **pre-5a whole-vCPU-park** semantics: the
opt-in **parallel** driver (`compile_and_run_capture_over_parallel*` — the wait blocks the OS
thread with the fiber active) and the browser **`Vcpu`** session driver (the wait surfaces as a
whole-vCPU `VcpuEvent::Wait`). The resumer never sees `FIBER_PARKED` there; the wait resolves
inside the resume. The debug drivers (`ScheduledDebugRun`) share the vCPU-park shape, which for a
*tool* is sanctioned tiering (invariant 9 observability corollary); the deterministic explorer's
whole-vCPU park is likewise deliberate (the `fiber_park!` gate on `SchedRef::Real`, documented in
`fiber_timed_wait.rs`'s module doc).

**Why tracked, not fixed here.** Neither path is reachable from `svm_run::Instance::run` (the
consumer surface the regression was reported against), no existing test puts a wait inside a
fiber on them, and extending the fiber-park routing into the parallel driver means fiber-waiter
delivery through its real cross-thread futex — its own slice, not a bolt-on. Invariant 9 wants
the divergence enumerated rather than silently normalized: this is the entry.

**Plan.** Fold fiber-level wait parks into the parallel driver's futex (mirroring the JIT's
`FutexEntry::fibers` cells) when a consumer reaches for fibers + waits on that path; the `Vcpu`
driver should follow the cooperative driver's `WaitParked` shape. Until then, fail-closed is not
required (the semantics are the historical ones), but any differential harness pointed at those
entries must avoid wait-inside-fiber kernels.

### I46 — `opt_sccp` fuzz oracle false-positived on any NaN result (S4, nightly fuzz red) — **FIX LANDED 2026-07-26** (`claude/nightly-ci-failures-1u64o2`)

**Symptom.** Nightly `cargo-fuzz (all targets) (opt_sccp)` failed both 2026-07-25 and 2026-07-26:
`assertion left == right failed: optimize_module changed observable behavior`, with `left` and
`right` printing **identically** (`Ok([F32(NaN), F64(-1.146…e-155)])`). The fuzzer had just
generated its first NaN-producing module (input `[255, 96, 43, 0]`), so this is a latent oracle
bug surfacing, not a code regression — the 2026-07-24 nightly was green.

**Root cause.** The target compared the two interpreter results with `assert_eq!` on
`Result<Vec<Value>, Trap>`. `Value` derives `PartialEq`, so its `F32`/`F64` arms use IEEE equality
where **`NaN != NaN`** — the equality fails whenever a result contains a NaN, even when the
optimizer preserved behavior exactly (identical `Debug` output confirms it). The IR pins no NaN
bit-pattern (SCCP may legally reshape a NaN payload), so bit-exact NaN comparison would be wrong
anyway.

**Fix.** Compare NaN-aware, reusing `irgen::values_equal` — the same equivalence the JIT
differential (`jit_fuzz`) already uses: bit-exact for non-NaN scalars, all-NaN-equal for NaN.
`values_equal` promoted to `pub` (the `#![allow(dead_code)]` module already tolerates
per-includer unused subsets). Verified by replaying the crash input through the real
optimize+interp pipeline: the input does produce a NaN, the bare `==` differs (control), and the
new comparison holds.

### I47 — text round-trip dropped a present-but-empty `debug_info`, breaking `parse ∘ print = id` (S3, nightly fuzz red) — **FIX LANDED 2026-07-26** (`claude/nightly-ci-failures-1u64o2`)

**Symptom.** Nightly `cargo-fuzz (all targets) (roundtrip)` failed 2026-07-26 (input
`SVM\0` + zeroed section, `[83, 86, 77, 0, 8, 0, …]`): `text round-trip changed the IR`, with
`left` = `Some(DebugInfo { files: [], locs: [], … })` and `right` = `None`.

**Root cause.** `decode_module` returned `Some(decode_debug_info(...))` whenever any bytes trailed
the funcs, so a zeroed (all-count-0) debug section decoded to `Some(DebugInfo::default())`. The
binary round-trip (`decode ∘ encode`) preserved that `Some(empty)`, but the text printer emits
nothing for empty debug info, so `parse ∘ print` collapsed it to `None` — the two round-trips
disagreed on a value that carries no information.

**Fix.** Canonicalize an all-empty debug section to `None` at decode (`if di ==
DebugInfo::default() { None }`), giving "no debug info" a single in-memory representation. Both
round-trips now agree; existing `debug_info_round_trips_through_binary` /
`no_debug_info_is_back_compatible` unit tests still pass, and the exact crash input was verified to
normalize and round-trip through both the binary and text paths.

### I48 — no **blocking** `cont.resume`: a guest with only a parked fiber has to busy-spin the poll (S3, ergonomics) — raised 2026-07-25; **BUILT 2026-08-07** as `cont.resume.block` (DESIGN.md §12)

**What.** After the §3.6 slice-5a fiber-park contract (svm PR #442), a `memory.wait` inside a
fiber parks the *fiber* and the resumer polls it with `cont.resume`, seeing `FIBER_PARKED (3)`
until the event fires. A guest whose **only** pending work is one parked fiber had nothing to do
between polls, so it busy-spun `cont.resume` (jacl's shape). No primitive said "idle this vCPU
until the parked fiber is due or woken."

**Invariant-1 renegotiation (owner, 2026-08-07).** This was deferred on INVARIANTS #1 ("no
machinery without a demonstrated need — wait for a second consumer or a measured cost"). The
owner explicitly requested the primitive be built, which *is* the demonstrated need the invariant
gates on; invariant 1 is satisfied, not weakened (no standing rule changed). Recorded here dated,
per the INVARIANTS.md preamble.

**BUILT — `cont.resume.block`, an advisory op (DESIGN.md §12).** A new opcode (`0xBF`) that is
**identical to `cont.resume`** — same `(status, value)` result, same 0/1 statuses — except the
runtime **may idle the resuming vCPU** on the resumed fiber's own registered waiter instead of
returning the `FIBER_PARKED (3)` poll status. Advisory is the whole trick: returning
`FIBER_PARKED` is a *conforming* implementation, so a guest still loops for completion exactly as
with `cont.resume`, and the design's hard problems dissolve —
- only the **tree-walk M:N oracle** idles (park into `svc_waiters` keyed on the fiber's domain,
  the `Blocked::OfferPark` shape — woken for free by the `svc_wake_locked` every fiber-wake
  already calls; swept by teardown; the idle deadline already includes the fiber's timer). The
  interception is at both `fiber_park!` (switch-in → fiber parks → idle) and the `cont.resume`
  `StillParked` re-poll, keyed by a `run_inner`-local marker on the resumed fiber slot;
- the **bytecode** and **Cranelift JIT** backends and the **deterministic explorer** alias it to
  `cont.resume` (return `FIBER_PARKED`) — no idle core needed, **no compile veto, nothing
  de-JITed**, invariant 9 holds by definition;
- **freeze / durability**: gated on `!durable`, so a durable run takes the `FIBER_PARKED`
  downgrade (guest loops) and freeze-on-quiesce is untouched (the F1 precedent);
- **teardown (#440)**: the idle resumer parks through `park_gate` into `svc_waiters`, swept by
  `teardown_domain`/`teardown_run` on exit/trap/freeze exactly like `OfferPark`.

Pins: `svm-interp/tests/blocking_resume.rs` (oracle idle-on-timer with a **fuel** proof it does
not spin, cross-vCPU notify wake, sibling-trap frees a blocked resumer);
`svm/tests/fiber_blocking_resume.rs` (advisory conformance — the looping form agrees
TreeWalk ≡ Bytecode ≡ Cranelift JIT). jacl swaps one opcode in its existing poll loop.

**PARITY on the production backends (2026-08-08).** #672 idled only the oracle (the test tier); prod
runs on svm-jit + wasm-jit, so the idle was extended to the tiers that matter:
- **Bytecode (+ wasm-jit)** — the cooperative `drive` idles the resumer's task as
  `TaskState::BlockedOnFiber` (rewind the resume op; a top-of-loop scan re-runs it when the fiber
  wakes via idle-timer / notify / cap-drain), burning zero fuel. wasm-jit folds fibers to this
  driver (`DriveMode::InterpDriven`), so it is covered for free. Scoped to `drive` via a
  `cooperative` flag on `step_vcpu`.
- **Cranelift JIT** — `fiber_resume_block` parks the resumer's **OS thread** on `Domain.futex_cv`
  and re-resumes, woken by any `notify`/teardown broadcast and bounded by the `KILL_RECHECK`
  re-poll (so a timed wait's deadline, a kill, a freeze, or teardown is observed with **no timer
  thread**). A fiber-using module now always builds the `Domain` (the thunk takes its pointer).
  Correctness does not depend on the broadcast — the bounded re-poll guarantees progress — so
  there is no lost-wakeup race beyond the futex-core primitives already loom-checked; a bespoke
  fiber-modeling loom test is out of scope (the loom model doesn't model fibers).

Now pinned on **all three backends** in `svm/tests/fiber_blocking_resume.rs`: a **no-loop**
`cont.resume.block` of a timed-wait fiber returns the fiber's real result (`102`) — a spinning alias
would return the transient `300` — plus a cross-vCPU **notify** wake, both across TreeWalk +
Bytecode + Cranelift JIT.

**Remaining follow-up (small):** the bytecode/JIT **OS-thread-parallel** driver variants
(`drive_parallel`, the single-vCPU `Vcpu::run`) still take the advisory `FIBER_PARKED` downgrade —
the same OS-thread-park mechanism, applied to those driver entry points.

---

### I49 — the **serve chain** deadlock (a handler that calls another server) — ticket-namespace collision, **FIX LANDED 2026-07-28**; grandchild serve-capture also fixed — surfaced building the §13.4 4d follow-up

**What.** Three threads split out of the "nested holder" 4d follow-up (a *child* C1 holding a live
`child_offer` cap onto a *grandchild* C2, durably frozen and re-linked on thaw):

1. **FIXED — depth-2 serve-state keying.** A live-spawned nested child never stamped its own
   `parent_task` at the op-0 `instantiate_module` spawn (the field's own destructure comment stated
   the intent — "a spawned child stamps its own id as the grandchild's parent below" — but the
   assignment was missing). A **grandchild** therefore defaulted to `parent_task = 0` (the root), so
   its `FrozenChildState` keyed `(0, slot)` mismatched its `FrozenNested` `(C1, slot)` on thaw, and
   it restored via the fresh-grant path **without its serve module** (first offer call → `EAGAIN`).
   One line at the spawn fixes it (direct children unchanged; root id is `0`). Pinned by
   `svm-durable/tests/serve.rs::a_three_level_nested_server_subtree_keys_the_grandchild_serve_state_to_its_real_parent`.

2. **FIXED — the serve-chain deadlock (root cause: a ticket-namespace collision).** The "handler
   forwards to another server" observable dead-locked, and reproduces with **zero durability** (a
   general serving-correctness bug, not thaw-specific — a front-end server delegating to a back-end
   is the common jacl shape). Root cause: `Sched::ticket_waiters` was keyed by the **bare dispatch
   ticket**, but tickets are per-callee-domain (each host's `svc_next_ticket` starts at 0). In
   `root → C1.fwd → C2.leaf`, root parked on C1's ticket 0, then C1's handler parked on C2's ticket
   0 — **overwriting root's waiter at key 0** — so when C1's handler returned, its reply to ticket 0
   found no waiter and stashed, stranding root forever (traced: `reply … -> STASH (no waiter)`). Fix:
   key `ticket_waiters` by `(callee domain id, ticket)`; every park/reply/teardown site threads the
   callee domain (all teardown tickets are dispatches *to* the dying domain, so its key is in scope).
   Pinned by `svm-interp/tests/svc_serve_chain.rs` (root → C1.fwd → C2.leaf(7) → 107; hung before,
   passes in 0.00s after).

3. **FIXED — the nested re-link generalization.** With the deadlock gone, the thaw re-link was
   generalized from root-only to **every holder**, keyed by the `(holder task, join slot)` edge
   (root-direct children key on `(id, slot)`, a grandchild on `(its parent-child cid, slot)`), so a
   child C1's durable cap onto a grandchild C2 re-links on thaw. Pinned end to end by
   `svm-durable/tests/serve.rs::a_nested_holder_freezes_and_thaws_with_the_grandchild_cap_relinked`:
   freeze the three-level cap-holding subtree, thaw, seed a dispatch into the root's queue, and the
   root drives `fwd(7) → C1 forwards leaf(7)` through the re-linked grandchild cap → **107** (fails
   with `-11`/`EAGAIN` if the nested edge is not re-linked; the test also confirms the serve chain
   no longer hangs). Remaining from the same 4d note: the wire/child-regrant **sibling-provenance**
   durable name (a live cap with no §14 child behind it).

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

## Standalone workspaces not covered by `cargo build --workspace` (rename hazard)

`browser/` (svm-browser), `bench/`, `browser/wt/`, and `fuzz/` are **separate cargo
workspaces**, not members of the root workspace (like `crates/svm-llvm/`). A `cargo build
--workspace` — the usual local pre-push check — does **not** compile them, so a cross-cutting
change to any public API they consume — `svm-interp`/`svm-run` (e.g. `host_fn`→`host_proc`,
2026-07-30) **or `svm-jit`** (e.g. the `CompiledModule::compile` `fast_resolver` param, 2026-08-04,
which broke `bench/`) — builds clean locally yet breaks a CI lane that only compiles the standalone
tree. This has recurred 4× now. **When changing any public `svm-interp`/`svm-run`/`svm-jit` signature,
also grep + build the standalone trees:** `browser`, `bench`, `browser/wt`, `fuzz`
(`cargo build --manifest-path <tree>/Cargo.toml`). A CI job that fast-checks these on every PR (not
only the expensive full browser build / non-gating nightly bench) would close the gap.
