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
  - [ ] (d) `OptConfig` (budgets, pass toggles) — **deferred to Phase 2**. Per the prime directive
    (no configurability until something concrete demands it), this lands with the first budgeted pass
    (the inliner / unroller), where a size/fuel budget actually bites; adding toggles now, with only
    the always-on cleanup passes, would be speculative surface.
- [ ] **Phase 2 — global scalar passes.**
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
  - [ ] jump threading, LICM for pure non-trapping ops. Each with differential + fuzz.
- [ ] **Phase 3 — interprocedural.** Budgeted inliner; constant-index `call_indirect` /
  `ref.func` devirtualization through the identity table; dead-function elimination
  (export/table-aware, rule 5 above).
- [ ] **Phase 4 — memory passes.** Redundant-load elimination + store-to-load forwarding over
  the effects table (clobber rules 2/4); opt-in scratch-region contract for DSE; simple range
  analysis so LICM/DCE can touch provably in-bounds loads.
- [ ] **Phase 5 — close out.** In-sandbox demo (guest runs `svm-opt` on a module and JITs the
  result, peval-demo shape); PEVAL_BENCH + Wasmtime-relative numbers with the optimizer on;
  fold the settled design into `DESIGN.md` §20 and retire this doc to a tracker stub.

### Enhancements (tracked, not gating)

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
