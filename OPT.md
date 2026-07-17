# OPT.md — `svm-opt`, the generic svm-IR AOT optimizer

Tracks building a **powerful, generic IR→IR AOT optimizer** that runs **outside svm (host-side
tooling) and inside svm (guest-side, in-sandbox)** — the successor to the cleanup optimizer that
lives in `svm-peval` today (`optimize_module`, DESIGN.md §20c). Companion to `DESIGN.md`
§20a/§20c and `PEVAL_BENCH.md`. Living doc: update the **Plan** tracker as work lands; fold into
`DESIGN.md` once it closes.

---

## Goal & shape

`svm-peval` already contains a generic, closed-module, semantics-preserving optimizer — constant
folding (bit-exact with the interpreter, floats + all v128 lane ops), branch resolution,
dead-block elimination, dead-value elimination, block merging, dead block-param elimination, and
copy-prop/algebraic identities, iterated to fixpoint. The goal here is to grow that into a real
AOT optimizer — global (cross-block) and interprocedural passes — **without** touching the wire
IR, the verifier, or the escape-TCB.

The value is *not* uniform across backends, and that shapes which passes are worth writing:

| Payoff | Why |
|---|---|
| **Cross-function transforms** | Cranelift optimizes per function; nothing in the pipeline inlines, devirtualizes constant-index `call_indirect`, or drops dead functions. Highest leverage for the JIT path. |
| **Interpreter backends** | The tree-walker and bytecode engine run the IR as-is — every pass pays off directly. The bytecode engine is the *only* execution path in the browser build (§21). |
| **Compile time** | Smaller IR JITs faster (peval corpus: residuals compile 4–13× faster). Guest-side `Jit.compile` is metered, so shrinking IR before compiling is a real in-sandbox win. |
| **Peval residual quality** | The specializer leans on the generic optimizer for cleanup; cases like `stack: chained expr` (73% of interpreter bytes kept) show headroom for GVN / load-forwarding. |
| **Per-function scalar cleanup (JIT path)** | *Low* — Cranelift's mid-end already does GVN/CSE, store-to-load forwarding, const materialization at `opt_level=speed`. Worth having for the interp paths, not a JIT-path win. |

## Posture: untrusted-for-escape, +0 TCB  [SETTLED — inherited from §20a/§20c]

Identical to the LLVM on-ramp and the specializer: the optimizer is a pure `Module → Module`
transform whose output is **re-verified** (`svm_verify::verify_module`) before it runs. An
optimizer bug is a clean verify error or a miscompile caught by the differential oracle — never
an escape. The verifier, the confinement-masking lowering, and the wire format do not change.

## Inside *and* outside svm  [SETTLED — recipe proven by svm-peval]

The guest-side story costs nothing new if the crate follows the rules `svm-peval` already
satisfies:

- `#![forbid(unsafe_code)]`, `no_std + alloc` (test harness gets `std`), dependency-free except
  the optional `libm` feature that travels with the float folds.
- Pure transform: no I/O, no time, no randomness — also required for determinism (differential
  testing, reproducible builds). Flat arrays / `BTreeMap`, no `HashMap`.
- **Bounded work**: every superlinear or growth-producing pass (inliner, unroller) takes an
  explicit budget via `OptConfig` — a guest pays metered fuel to run the optimizer, and code
  growth costs metered `Jit.compile` time.

Host-side AOT flow: `decode → optimize → verify → encode` (already wired in `svm-run` behind the
`optimize` flag). Guest-side: `optimize → encode → §22 Jit.compile → invoke`, exactly the peval
demos' shape.

## The equivalence contract (what a pass may do)  [SETTLED — this is the spec]

The bar is the §3 parity invariant, applied to `optimized` vs `original` on the reference
interpreter: **same results, same trap kind, same final memory window** — for every input. The
consequences are the legality rules every pass must obey:

1. **Trapping ops are effects.** Loads can trap out-of-window; div/rem-by-zero and signed
   `INT_MIN/-1` trap deterministically. So: no deleting a dead load, no hoisting a load out of a
   loop, no reordering two potentially-trapping ops — *unless* an analysis proves the op cannot
   trap (e.g. a range analysis proving in-bounds).
2. **Store-to-load forwarding is sound** (same effective address: the store already proved the
   access in-window), and so is redundant-load elimination across a dominating identical load.
3. **Dead-store elimination is illegal** under "same final memory window." If wanted, it needs an
   opt-in caller contract ("this region is scratch"), the same shape as `SpecConfig`'s
   constant-memory / rename-region promises: a false promise is a miscompile, never an escape.
4. **Threads/fibers (§12 C/C++11 model):** any atomic, fence, `call`, `cap.call`, or fiber/thread
   op is a full clobber for memory-value tracking; non-atomic load caching is sound only between
   clobbers.
5. **Function indices are addressable** (`ref.func`, `call_indirect` over the identity table):
   dead-function elimination may only drop functions that are unexported *and* provably
   unreferenced by any table-addressable path.

All of this is encoded once, in a shared **per-`Inst` effects table** (pure / can-trap /
reads-mem / writes-mem / host-observable / control), living next to the `Inst` enum in `svm-ir`
so new instructions *must* classify. Every pass consults the table; no pass carries its own
opinion about purity. (Today this exists piecemeal as `is_removable_if_dead` + scattered match
arms.)

## Architecture  [SETTLED — decision recorded below]

**Crate:** new `crates/svm-opt`, depending only on `svm-ir` (workspace member, same manifest
shape as peval). The existing `optimize_module` + fold/remap machinery *moves* there;
`svm-peval` depends on `svm-opt` for residual cleanup and stays about specialization.

**The block-local-SSA question.** The wire IR's key discipline — values are block-local,
cross-block dataflow only via block params, no dominance analysis anywhere (§3) — is what keeps
the verifier a linear pass, and is untouchable. But global passes (GVN, LICM, SCCP) need to move
and track values across blocks, which in wire form means threading block params through every
intermediate block per pass. Decision: **the optimizer converts each function into a private,
optimizer-internal conventional SSA form** (function-global value numbering, block params as
phis; flat structs-of-arrays, integer indices, arena-per-function per the DOD rules), runs its
passes there, and lowers back to block-local form in one well-tested boundary pass (out-of-SSA
param threading). The wire format and verifier never see it. Same shape as the JIT (Cranelift
converts to its own form too); one boundary to fuzz instead of per-pass renumbering cleverness.
The round-trip is differentially fuzzed as a **no-op** (convert → lower, no passes) before any
pass uses it.

**Debug info:** `optimize_module` currently drops `debug_info`. v1 keeps that behavior but makes
it explicit (`-O` drops debug info, documented); threading line maps through transforms is a
tracked enhancement, not a blocker.

## Testing, fuzzing, benchmarks (the CLAUDE.md obligations)

- **Differential oracle, every pass:** `optimized(args) == original(args)` on the reference
  interpreter — results, trap kind, *and final memory window* — plus randomized module
  generation stressing each new pass (the `tests/optimize.rs` harness extends).
- **Fuzz from the first pass:** a `fuzz/` target — arbitrary module → verify → optimize → must
  **re-verify** and behave identically. The internal-SSA round-trip gets its own no-op fuzz
  target first.
- **Bench from the first pass:** an optimizer on/off axis in the PEVAL_BENCH report and the
  Wasmtime-relative harness (size, compile time, run time per backend), so regressions are one
  commit old. Log flakiness to `ISSUES.md`.

## Plan

- [x] **Phase 0 — carve-out.** Created `crates/svm-opt` (`no_std + alloc`, `forbid(unsafe_code)`,
  optional `libm`); moved `optimize_module` + fold/remap machinery out of `svm-peval`; re-pointed
  `svm-peval` / `svm-run`. Pure refactor, no behavior change; all existing tests green.
- [ ] **Phase 1 — infrastructure.**
  - [x] **(a) The per-`Inst` effects/trap table in `svm-ir`** ([`Inst::effects`] → [`Effects`]:
    `can_trap` / `reads_mem` / `writes_mem` / `side_effect`; `is_pure` + `removable_if_dead`
    derived). Exhaustive match, no wildcard, so a new `Inst` variant must be classified to compile.
    The DVE pass's `is_removable_if_dead` now delegates to it — one purity oracle. Behavior-preserving
    (the classification reproduces the old whitelist exactly; harness unchanged) and unit-tested by
    category in `svm-ir`.
  - [x] **(b) CFG utilities** (`svm_opt::cfg`): `Cfg` (pred/succ adjacency), `postorder`/`rpo`,
    iterative Cooper–Harvey–Kennedy `dominators` (correct on irreducible CFGs), Tarjan `sccs`, and
    irreducible-aware `loop_headers` (flags *every* entry of a multi-entry cycle). All traversals are
    iterative (no host-stack recursion) and fuzz-safe (out-of-range targets ignored, not panicked).
    `term_successors` now delegates to `cfg::successors`. Unit-tested incl. an irreducible two-entry
    loop, self-loop, and unreachable block.
  - [x] **(c) The internal conventional-SSA form** (`svm_opt::ssa`): `to_ssa` renames block-local
    operands to a function-global `Value` space (via the exhaustive `map_operands`), tracking each
    value's `Def` site; `from_ssa` is its exact inverse. The round-trip is the **identity**
    (`from_ssa(to_ssa(f)) == f`), pinned by hand-built shapes (params/`br_table`/loops/multi-result
    `call`), an interpreter behavioral check, a 5000-case randomized structural test, and the
    `opt_ssa_roundtrip` cargo-fuzz target (reuses `irgen` → round-trip identity + interp-vs-JIT
    differential on the lowered module). Cross-block-use lowering (param threading) is deferred to the
    first Phase 2 pass that needs it — today every value is used only in its defining block.
  - [x] (d) `OptConfig` (pass toggles) — landed once something concrete demanded it: the **ablation
    benchmark** (`OPT_BENCH.md`), which needs to disable one pass at a time to attribute a size/speed
    delta to it. `OptConfig { sccp, reassociate, gvn, licm, local_cse, jump_thread }` (all default
    on) + `optimize_module_with` / `optimize_func_with`; `optimize_module` is the `all()` pipeline
    unchanged. Only the six global/analysis passes toggle — the always-on intra-block canonicalization
    is the shared substrate and the honest "no optimization" baseline. Budgets stay deferred to the
    first budgeted pass (inliner / unroller), where a size/fuel budget actually bites.
- [x] **Phase 2 — global scalar passes.**
  - [x] **SCCP** (`svm_opt::sccp`): sparse conditional constant propagation on the internal SSA
    form — a `Top ⊒ Const ⊒ Bottom` lattice propagated across the CFG (through block-parameter phis
    and loops) together with per-edge executability, so a value is only marked varying on account of
    edges that can actually be taken. The transfer function reuses `try_fold` (interpreter-exact), so
    a trapping/effectful op is never marked constant; the rewrite materializes constants and resolves
    constant branches, then the existing fixpoint prunes/DCEs/merges. Runs first in `optimize_func`.
    Because it only materializes constants, lowering needs no block-param threading. Differential +
    structural tests (`tests/sccp.rs`: multi-pred const param, cross-block branch resolution,
    loop-invariant fold, trapping-div-not-folded) and the `opt_sccp` cargo-fuzz target (gen → verify
    → optimize → **re-verify** + interp differential). The existing optimize/peval harnesses (which
    now run through SCCP) stay green.
  - [x] **Local CSE** (`local_cse` in the `optimize_func` fixpoint): intra-block common-subexpression
    elimination — a *pure* instruction (per the Phase 1a effects table) whose op and operands match an
    earlier one is rewritten to that result and left for DCE. Operands are canonicalized to their CSE
    roots first, so equal expressions from equal subexpressions collapse too. Only pure ops qualify
    (a load/atomic/call may trap, read changing memory, or have effects — never CSE'd); single-block,
    so the earlier def trivially dominates, no param threading. Tests (`tests/cse.rs`): redundant add
    deduped, nested equal subexpressions, and identical loads **not** deduped. Covered by the
    `opt_sccp` fuzz target (now exercises the whole `optimize_module` pipeline).
  - [x] **Global GVN** (`svm_opt::gvn`): value-number **congruence** (a block parameter is congruent
    to the value passed to it when every predecessor agrees), iterated to a fixpoint, so a
    recomputation at a **multi-predecessor join** is recognized as redundant even though its operands
    are fresh parameters — the case `merge_blocks` + `local_cse` cannot reach. A congruent value whose
    definition **dominates** the redundant one is threaded to it by adding block parameters + edge args
    along the intervening edges (the internal SSA form's first cross-block-use lowering); new params
    are typed via the verifier's `func_value_types`, so the threaded IR re-verifies. Only pure ops get
    a shared number (loads/atomics/calls stay unique). Runs before the per-function cleanup. Tests
    (`tests/gvn.rs`): diamond-join redundancy, a derived two-level expression across a diamond, impure
    loads at a join **not** deduped, and a 600-case randomized branchy-DAG differential (behavior
    preserved + optimizer demonstrably firing); also covered by the `opt_sccp` fuzz target (whole
    pipeline).
  - [x] **Branch & select simplification** (in `resolve_term` / `forward_to_operand`): a
    `br_if`/`br_table` whose targets all coincide (same block *and* args) becomes an unconditional
    `br` (the selector computation dies for DCE, and the now-single-predecessor target merges); a
    `select` with equal arms folds to a copy. Both are no-renumber and compound with SCCP/GVN, which
    routinely emit such degenerate branches/selects. Tests (`tests/simplify.rs`): coincident
    `br_if`→`br` (+ dead condition removed + block merged), coincident `br_table`→`br`, equal-arm
    `select`→copy.
  - [x] **Instcombine peepholes** (`try_fold` + `reassociate`): integer **self-comparison** folds
    (`x==x`/`x<=x`/`x>=x` → 1, `x!=x`/`x<x`/`x>x` → 0; integer only — floats are `FCmp`); and
    **constant reassociation** `(x OP c1) OP c2 → x OP (c1 OP c2)` for associative+commutative ops
    (Add/Mul/And/Or/Xor), which shrinks constant chains an op at a time and exposes CSE (two paths that
    reassociate to `x+8` then share one op). Tests in `tests/peephole.rs`.
  - [x] **LICM** (`svm_opt::licm`): hoists pure, non-trapping loop-invariant computations to the
    loop preheader. Invariance is computed iteratively through block-param phis (a value defined
    outside the loop is invariant; a loop parameter is invariant when every incoming arg is invariant
    or is the parameter itself — the archetypal `x` passed unchanged around the back edge; a pure op is
    invariant when its operands are). An invariant op is cloned into the preheader (operands rewritten
    to preheader-valid values — a dominating value as is, an invariant header param's entry arg, or a
    loop-body **constant rematerialized** in the preheader) and its result threaded back in
    (`crate::thread`), leaving the original for DCE. A **hoist cost model** keeps the threading honest: a
    bare constant is never hoisted (free to recompute — threading one out is pure overhead), and an
    invariant's constant operands are re-emitted in the preheader (`Threader::emit`) rather than threaded,
    so a worthwhile hoist still fires without dragging a constant around the loop (measured: size-neutral
    on hoist-free loops, smaller output on real-invariant loops — see `OPT_BENCH.md`). Sound by
    construction (pure+non-trapping speculated above the loop) and conservative on shape (reducible
    single-header loops with a unique preheader only). Reuses the shared `vn` / `thread` modules.
    Tests: invariant op hoisted out of every loop (SCC check) + variant op stays, behavior preserved; a
    bare invariant constant is *not* hoisted, while an invariant op over a loop-body constant still is
    (its constant rematerialized out of the loop); covered by the `opt_sccp` fuzz target (whole pipeline).
  - [x] **Jump threading** (`jump_thread` in the `optimize_func` fixpoint): redirects an edge that
    reaches an **empty conditional forwarder** — a block with no instructions whose `br_if`/`br_table`
    tests one of its own parameters — straight to the resolved target when the predecessor passes a
    constant for that selector parameter. This is the correlated-branch pattern (`if c { … } ; if c
    { … }`) that SCCP structurally cannot catch, because the forwarder's selector parameter meets a
    *different* constant on each incoming edge (so its lattice value is not constant). Sound because
    the forwarder has no instructions (no defs, no effects): entering it only selects a branch, so
    threading past it with the same argument values is observationally identical; the resolved edge's
    args are forwarder-parameter indices, mapped back through the predecessor's edge args to values
    valid there. The fixpoint's prune then drops any forwarder left with no predecessors, and re-runs
    threading so multi-hop chains collapse a hop per iteration. Tests (`tests/jump_thread.rs`):
    correlated branch threaded + forwarder eliminated, behavior preserved; covered by the `opt_sccp`
    fuzz target (whole pipeline, gen → verify → optimize → re-verify + interp differential).
- [ ] **Phase 3 — interprocedural.** Budgeted inliner; constant-index `call_indirect` /
  `ref.func` devirtualization through the identity table; dead-function elimination
  (export/table-aware, rule 5 above).
  - [x] **Dead-function elimination** (`svm_opt::interproc::dead_func_elim`): the first module-level
    pass. Call-graph reachability closure from the roots (entry `func 0` + every named export) over
    `call`/`return_call`/`thread.spawn`/`ref.func` edges (the static-funcidx sites
    `svm_ir::offset_func_indices` enumerates); survivors renumbered densely, every funcidx reference +
    export target remapped (names preserved). Because a funcref **equals its funcidx** (identity table)
    and can be a plain `ConstI32`, an indirect dispatch could reach any function, so the pass bails to
    the identity while `call_indirect`/`return_call_indirect`/`cont.new` is present — devirtualization
    (below) removes those first, then DFE applies. `OptConfig.dfe` toggle (default on), runs once after
    the per-function passes; output re-verified; debug info dropped on any removal. Tests in
    `tests/interproc.rs`; covered by the `opt_sccp` fuzz target (whole pipeline).
  - [x] **Budgeted direct-call inliner** (`svm_opt::interproc::inline_calls`): splices a small callee
    into a direct `call` site. A **single-block, straight-line** callee is substituted in place — its
    params bind to the call's args, its instruction results take fresh caller-local indices where the
    call was, and the call's result forwards to the callee's returned values (renumbered through the
    shared `map_operands` map). A **multi-block callee** (internal control flow) is spliced by
    **CFG surgery** (`inline_multi_block_call`): the caller block is split at the call, the callee's
    blocks are appended (targets shifted, each `return` → branch to a fresh **continuation** block that
    receives the return values), and each **captured** pre-call value used after the call is threaded
    through the callee — appended as a pass-through parameter to every callee block and carried along
    every edge (including back edges of loops in the callee) to the continuation. This over-threads;
    the always-on dead-block-parameter cleanup prunes the params that weren't needed. Tail-call callee
    exits are excluded (a separate transform). Module-wide instruction budget + `MAX_CALLEE_INSTS`
    (total, across all callee blocks) size guard + direct-self-recursion skip bound growth and
    guarantee termination. `OptConfig.inline` toggle (default on), runs first at module scope so the
    per-function passes fold through the inlined bodies and DFE sweeps the now-uncalled leaf — the
    end-to-end interprocedural story. Tests in `tests/interproc.rs` (single-block leaf inlined +
    DFE-removed, live code across the call site renumbered; a multi-block `abs` callee inlined with a
    captured value threaded through the join; a callee with a **loop** threaded around its back edge;
    full-pipeline devirt→inline→DFE), all re-verified; the peval differential suite + `opt_sccp` fuzz
    target now exercise it on real residuals.
  - [x] **Constant-funcref devirtualization** (`svm_opt::interproc::devirtualize`): a
    `call_indirect`/`return_call_indirect` whose `idx` is a compile-time-constant funcref (a
    `ref.func k`, or an in-range `ConstI32 k` — a funcref is a plain `i32`, the identity table) and
    `funcs[k]`'s signature matches `ty` → rewritten **in place** to a direct `call`/`return_call`
    (matching signatures ⇒ matching result arity ⇒ no renumbering). The sig check is load-bearing: a
    mismatched/out-of-range index is left as an indirect call so it still *traps* identically rather
    than silently calling the wrong function. `OptConfig.devirt` toggle (default on), runs before
    inlining so a devirtualized call becomes an inlining candidate — and, with the indirect dispatch
    gone, DFE's gate lifts. Tests in `tests/interproc.rs` (devirt → inline → DFE end-to-end; a
    signature mismatch is left to trap); peval differential + `opt_sccp` fuzz cover the pipeline. This
    completes the Phase 3 trio (**devirt + inliner + DFE**); multi-block-callee inlining (below) has
    since landed too.
- [ ] **Phase 4 — memory passes.** Redundant-load elimination + store-to-load forwarding over
  the effects table (clobber rules 2/4); opt-in scratch-region contract for DSE; simple range
  analysis so LICM/DCE can touch provably in-bounds loads.
  - [x] **Intra-block redundant-load elimination + store-to-load forwarding** (`mem_forward` in the
    `optimize_func` fixpoint): scanning a block forward, a value is *available* at a location keyed by
    `(address SSA value, offset, load op)` once a load reads it or a full-width matching store writes
    it; a later identical `Load` with no intervening clobber is **removed** and its result forwarded
    (renumbering like `dce_block` — the pass carries its own safety argument since general DCE keeps
    loads as possible traps). Minimal, sound alias model: same location ⇔ same address *value* + offset
    + op; **any** memory write or side effect (store / atomic / `mem.copy`/`fill` / call, via the
    effects table) clobbers the whole availability map, and a plain store re-establishes only the cell
    it wrote. Runs right after `local_cse` so recomputed addresses are already unified. Sound on value
    *and* traps (the earlier same-address, same-width access proved the address in-bounds; `align` is a
    hint). Only full-width same-type store→load pairs forward (`i32/i64/f32/f64`); narrowing/cross-type
    are excluded. A store **clobbers precisely**: it drops cached cells off a *different* base value
    (may alias) or the *same* base with an **overlapping** byte range, and **keeps same-base cells at
    disjoint offsets** — sound under `svm_mask`'s trap-confinement, where two admitted accesses off one
    base differ by exactly their offset gap (an out-of-range address traps, never wraps to alias), so
    disjoint offset ranges are disjoint bytes (the common struct-field pattern). An atomic / `mem.copy`
    / call still clobbers the whole map. **`v128` load/store** participate too (keyed by a distinct
    16-byte discriminator; a `v128.store` forwards to a `v128.load`), with cross-width overlap against
    scalar cells handled by the same byte-range check. `OptConfig.mem` toggle (default on). Tests in
    `tests/memopt.rs` (forwarded store; redundant load across a pure op; the aliasing `a==b` may-alias
    store that must block forwarding; a disjoint-offset store that must *not*; an overlapping-offset
    store that must; `v128` store→load forwarding); the peval differential suite + `opt_sccp` fuzz
    cover the pipeline.
  - [x] **Cross-block redundant-load elimination** (`svm_opt::load_elim`) — the memory analogue of
    GVN. A load whose location a **dominating** access (an earlier load, or a matching store)
    established, with **no memory write on any path between**, is removed and its result forwarded —
    threaded across blocks by `crate::thread::Threader`, exactly as GVN threads a congruent dominating
    value. "Same location" is `(address value-number, offset, op)` — the address by `crate::vn`
    congruence (block-local SSA never shares an operand id). Alias model across blocks is **precise for
    stores**: a between-path store does *not* clobber when it is provably disjoint — the same base
    value-number with a non-overlapping byte range — reusing `mem_forward`'s intra-block reasoning
    (sound under trap-confinement: two admitted accesses off one base differ by their offset gap, an
    out-of-range address traps rather than wrapping to alias). A store off a *different* base
    value-number, or any other write / side effect with unknown reach (atomic / `mem.copy`/`fill` /
    call), still clobbers. The between region must be **acyclic** (loop-carried loads deferred) so the
    source runs once before the load and the partial-block clobber checks (after the source, before the
    load) are valid.
    Sound on value *and* traps (the dominating same-width access proved the address in-bounds). The load
    is removed with a block-local rebuild (general DCE keeps loads as possible traps). Runs per-function
    after GVN (addresses already congruent); `OptConfig.load_elim` toggle (default on). Tests
    (`tests/load_elim.rs`): sequential + diamond-join forwarding, a between-store that must block it
    (checked at `addr==other` where a missed clobber would miscompile), an adversarial loop with a
    per-iteration store that must **not** be eliminated, and the alias-precision boundary — a
    disjoint-offset store in an arm still forwards, an overlapping-offset store blocks it; the whole
    pipeline (18-case randomized
    differential, 46 peval differentials, `opt_sccp` fuzz) exercises it. **Next:** loop-invariant load
    hoisting and an opt-in scratch-region contract for dead-store elimination (DSE needs a
    private-region guarantee to stay sound under shared-memory threads).
- [ ] **Phase 5 — close out.** In-sandbox demo (guest runs `svm-opt` on a module and JITs the
  result, peval-demo shape); PEVAL_BENCH + Wasmtime-relative numbers with the optimizer on;
  fold the settled design into `DESIGN.md` §20 and retire this doc to a tracker stub.
  - [x] **Per-pass ablation harness** (`crates/svm-peval/tests/opt_bench.rs`, report in
    `OPT_BENCH.md`): leave-one-out over the six togglable passes on a corpus of a realistic
    specialization residual + pass-targeted micro-modules, reporting encoded size and both JIT and
    interpreter run time. Every variant is re-verified + interp-differential-tested, so the size test
    is also a correctness/size guard. First findings: the JIT's own optimizer washes out svm-opt's
    scalar passes (native run time flat), so their JIT-path value is size/compile; on the interpreter
    the full pipeline is ~1.3× on an invariant-heavy loop, LICM the dominant run-time pass (~1.17×),
    while LICM/GVN *cost* static size — motivating a hoist cost model. Broaden the corpus (more
    realistic residuals, branch-heavy shapes) and add Wasmtime-relative numbers next.

### Benchmark follow-ups (from `OPT_BENCH.md`, PR #337)

The first ablation surfaced concrete next steps, tracked here so they aren't lost:

- [x] **LICM hoist cost model (constants).** The clearest actionable finding: on a loop with *nothing
  worth hoisting* (the tight sum loop), LICM still hoisted the loop's constants, adding block-parameter
  threading that grew static size for no run-time gain. A constant is free to recompute, so hoisting one
  only threads a copy around the loop — pure overhead. LICM now (1) **never hoists a bare constant**, and
  (2) when a worthwhile invariant *uses* a loop-body constant, **rematerializes that constant in the
  preheader** instead of threading it in (`Threader::emit`). Result (`opt_size_ablation`): on the
  hoist-free reg-sum loop LICM's size delta is now **+0** (was −10) — it hoists nothing useless; on the
  real-invariant loops the LICM win is preserved *and* output shrinks (heavy-invariant loop 100→94 B,
  licm+cse 77→74 B) because invariant const operands no longer thread. Removing LICM on the heavy loop is
  still ~1.4× (it remains the dominant interp-path run-time pass). A GVN variant of the same cost concern
  is deferred until a corpus case demands it.
- [x] **Broaden the ablation corpus + measure the full pass set.** The harness now leaves out **all
  eleven** togglable passes (was six) and adds two cases so the Phase-3/4 passes have a shape that
  exercises them: a **memory** case (a redundant same-address load for `mem`, a diamond-join reload
  for `load_elim`) and an **interproc** case (a constant `call_indirect` → `devirt` → `inline` → `dfe`,
  nearly halving the module). `OPT_BENCH.md` regenerated; the interprocedural passes turn out to be the
  biggest *size* wins in the corpus. (Still open: more branch/const-prop-heavy shapes to sharpen SCCP.)
- [ ] **Multi-run statistics in the harness.** Single-run numbers show visible variance (one JIT row
  read 2× its neighbors). Report medians + spread over several runs before treating any delta as load-
  bearing.
- [ ] **Wasmtime-relative numbers** (also under Phase 5): measure optimized-residual run time against
  Wasmtime on the same workloads, per DESIGN.md §1a — the "measured relative to wasm/Wasmtime" bar.
- [ ] **Note for Phase 3/4 targeting.** Because the JIT backend (`opt_level="speed"`) already does its
  own GVN/CSE/LICM, scalar passes are ~JIT-run-time-neutral; the higher-leverage host-JIT wins are the
  cross-boundary transforms Cranelift *cannot* see — inlining (Phase 3) and memory passes (Phase 4).

### Enhancements (tracked, not gating)

- [x] **Multi-block-callee inlining.** Landed (`inline_multi_block_call`). A callee with internal
  control flow is spliced by CFG surgery: split the caller block at the call, append the callee's
  blocks (targets shifted, `return`s → branch to a fresh continuation block that receives the results),
  and thread each value live across the call site through the callee — appended as a pass-through
  block parameter to every callee block and carried along every edge (loops' back edges included) to
  the continuation, with the dead-block-parameter cleanup pruning the over-threaded params. Termination
  is the same instruction budget + total-insts size guard + self-recursion skip as the single-block
  path. (Chose uniform over-threading + cleanup over selective `crate::thread::make_available` — same
  result, simpler code.)
- [ ] Debug-info (line map) preservation through transforms.
- [ ] Loop unrolling / peeling under `OptConfig` budgets.
- [ ] Interprocedural constant propagation (beyond what inlining exposes).

## Open questions

- ~~Where exactly the effects table lives: `svm-ir` vs `svm-opt`.~~ **Resolved (Phase 1a):** it
  lives in `svm-ir` next to `Inst`, as `Inst::effects() -> Effects`, so the exhaustive match forces
  every new instruction to be classified before it compiles.
- Whether `svm-run` grows a standalone optimize-only mode (`decode → optimize → verify →
  encode`, no specialization) for AOT pipelines — cheap, probably Phase 0/2 boundary.
- How much of the specializer's CFG-cleanup entanglement moves vs stays: `specialize` calls the
  optimizer for residual cleanup; the seam is the `svm-opt` public API. Settle during Phase 0.
