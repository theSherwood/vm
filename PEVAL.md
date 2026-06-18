# Partial evaluation / Futamura projection over the IR (`PEVAL.md`) — viability & design

A viability assessment, in the working-tracker style of `LLVM.md` / `WASM.md` /
`INTERP_PERF.md`: can we do the **first Futamura projection** over the SVM IR — take an
**interpreter** expressed in our IR, partial-evaluate it against a fixed input script,
and get a **compiler** for that script's language out the other side? The wasm precedent
is real (**weval**), and the answer here is **yes, and the IR is unusually well-suited
to it** — with the engine itself being the substantial work.

> **Status: ASSESSMENT ONLY — no code yet.** This file scopes the idea, the wasm
> precedent, why SVM is well-positioned, what is missing, the recommended (host-side,
> offline) architecture, and a staged prototype path. Fold into `DESIGN.md` and drop
> this file once a prototype lands and the actionable gaps close (the repo convention,
> cf. the former `WASM.md`/`SCHEDULING.md`).

---

## 1. What & why

The **first Futamura projection**: given an interpreter `interp(program, input)` and a
specializer `spec`, the residual `spec(interp, program)` is a *compiled* version of
`program` — `spec(interp, program)(input) ≡ interp(program, input)`, but with the
interpreter's dispatch loop, opcode decode, and program-walking folded away. Specializing
the *specializer itself* gives the second (a standalone compiler) and third (a
compiler-generator) projections; those are out of scope here.

**The use case is guest interpreters — full stop.** Specializing *our own* reference
interpreter buys little: `svm-interp` is deliberately the escape-TCB **oracle / debug
engine**, not the performance path, and the **JIT is already the production engine** that
lowers IR → native (see `INTERP_PERF.md`). The compelling target is **guest-level
language runtimes**. Today a guest that wants to run JS / Python / Lisp / a DSL either:

- ships an *interpreter* (compiled to our IR via the C frontend or the LLVM on-ramp) and
  eats interpreter overhead forever, or
- hand-writes a *JIT* for its language on top of the §22 guest-driven `Jit` capability —
  which works (`crates/svm-run/demos/jit/jit_demo.c`) but is real, per-language work.

A weval-style partial evaluator gives the guest-language author a **third option: write
only the interpreter, get the compiler for free.** That is precisely weval's value
proposition for SpiderMonkey, and it is the goal of this work.

---

## 2. The wasm precedent — weval

**weval** is Chris Fallin's WebAssembly partial evaluator. It implements the first
Futamura projection on wasm: an interpreter compiled to wasm, annotated with intrinsics,
is specialized against a fixed bytecode program to produce a specialized wasm function.
It is the engine behind AOT-compiling the **SpiderMonkey** JS interpreter to wasm.

Two properties of weval matter for us:

1. **It is built on `waffle`, an SSA wasm-to-wasm IR.** weval's single hardest first
   step is *reconstructing SSA* from wasm's stack machine before it can do anything —
   constant propagation and value-flow need SSA. (We pay none of this; see §3.)
2. **The interpreter must be annotated.** weval intrinsics mark what is constant and
   bound specialization: `assume_const_memory` (the bytecode region is constant at
   spec time), `push_context` / `pop_context` / `update_context` (the "context" is
   essentially the interpreter's program counter — this drives *polyvariance*: one
   specialized residual block per bytecode position, and it bounds termination),
   `specialize_value`, `flush_to_mem`. The hard engineering is then symbolic execution
   over constant memory + context-driven block specialization + renaming the
   interpreter's value-stack/locals out of memory into SSA so the residual is
   straight-line.

The bar weval sets, and the vocabulary ("constant memory", "context", "polyvariance",
"memory renaming"), carry over directly.

---

## 3. Why SVM is unusually well-positioned

Relative to wasm/weval, the IR removes or pre-pays several of the hard steps:

- **SSA-on-the-wire.** The IR is typed SSA over a CFG with **explicit typed block
  parameters, no phi nodes** (`crates/svm-ir/src/lib.rs`: `Module` / `Func` / `Block` /
  `Inst` / `Terminator`; `DESIGN.md` §3, §3a). weval's hardest first step — SSA
  reconstruction from a stack machine — *does not exist for us.* `DESIGN.md` already
  sells this as the §1a win over wasm ("SSA on the wire — no SSA reconstruction").
- **Total / deterministic semantics.** All arithmetic is defined; div-by-zero and
  `INT_MIN/-1` **trap** (not UB), shifts are mod width, everything wraps two's-complement
  (svm-ir `IntBin`/`IntCmp`; `DESIGN.md` §3b). There is **no undefined behavior to
  reason around**, which makes constant folding and branch resolution sound and simple.
- **IR→IR rewriting is already a thing we do.** `svm_ir::resolve_imports_with`
  (`crates/svm-ir/src/lib.rs:2467` — a 1:1, no-renumber instruction rewrite) and
  `svm_ir::link` (`:2602` — merge units, relocate data, reindex `FuncIdx`). A
  specializer is the same *kind* of pass, just deeper.
- **"Untrusted transform + re-verify = zero escape-TCB" is an established, blessed
  pattern.** The LLVM on-ramp (`crates/svm-llvm`, `DESIGN.md` §20a) translates
  *untrusted* bitcode → IR and is safe **only because its output is re-verified**. A
  partial evaluator inherits the exact same posture: the specializer can be as buggy as
  it likes — a bad residual is a clean `verify_module` error, **never an escape**. This
  is the single most important enabler: it lets the PE engine live entirely *outside*
  the escape-TCB.
- **A hand-written first projection already ships.** §22's `jit_demo.c` is a guest
  interpreter that emits specialized IR for its own bytecode and JITs it through the
  `Jit` capability. The PE engine *automates the emission step* that demo does by hand —
  the back half (verify → compile → run, differentially pinned interp == JIT) already
  exists and is unchanged.

---

## 4. What is missing (the actual work)

- **The partial-evaluation engine.** There are **zero** optimization/transformation
  passes in `crates/svm-ir/src/` today — no constant propagation, no constant folding,
  no DCE, no branch resolution, no inlining. All of it must be built.
- **Symbolic "constant memory".** The interpreter's bytecode/script lives in the masked
  memory window; PE must treat a designated region as **constant at spec time** so that
  loads of opcodes/operands fold to constants and the dispatch `br_table` resolves to a
  single edge. (weval's `assume_const_memory`.)
- **Spec-time markers.** We need an equivalent of weval's intrinsics — to mark the
  constant region and to delimit the **context** (the PC) that drives polyvariant block
  specialization and bounds termination. These can be recognized as IR annotations or,
  more ergonomically, surfaced at the C-frontend level (a `<svm_peval.h>` of no-op
  intrinsics the frontend lowers to markers).
- **Value-stack memory renaming.** Because of the §3d two-stack split, the interpreter's
  operand stack and locals live **in the window**, not in SSA. To get good residual code
  the engine must lift those slots into SSA values during specialization. This is the
  genuinely hard part — the same problem weval solves — and the main correctness risk.

**What can *not* be specialized away (by design, and fine):** confinement masking (I1)
and `call_indirect` / `cap.call` type+generation checks (I2/I3) remain in residual code.
They are positional, use-site security, not interpreter overhead — exactly like wasm
bounds checks surviving weval.

---

## 5. Recommended architecture — host-side, offline (`svm-peval`)

A new crate that **mirrors `svm-llvm`'s stance exactly**: untrusted-for-escape, output
re-verified, off the cross-OS runtime path.

```
Module
  + interpreter entry fn
  + designation of the constant inputs (the script / bytecode region)
        │
        ▼
  svm-peval   ──  const-prop + symbolic constant-memory
                + context(PC)-driven polyvariant block specialization
                + value-stack memory renaming (slots → SSA)
                + branch resolution + DCE + fold
        │
        ▼
  specialized Module
        │
        ▼
  svm_verify::verify_module        ← the safety net (PE is untrusted-for-escape)
        │
        ▼
  existing JIT / interpreter       ← unchanged back half
```

- **Reuse:** the verifier (`crates/svm-verify`), the text format
  (`crates/svm-text::print_module`) for golden tests, and the existing **differential
  oracle** (`interp == JIT`) as the correctness spec — the residual must satisfy
  `specialized(input) == interp(script, input)` on both backends.
- **Why offline-first:** lowest risk, fully reuses the re-verify safety net, and keeps
  the large new engine out of both the runtime and the guest. It directly serves the
  goal — a guest-language author runs `svm-peval` ahead of time on their interpreter +
  the user's script and ships a fast residual module.

### Alternative — guest-side, on the §22 `Jit` capability
Ship the PE engine *inside* the sandbox; the guest emits a specialized Module at runtime
and compiles it via `Jit`. Higher ceiling for **dynamic** languages (specialize on
runtime-observed types / hot bytecode, recompile on invalidation — the SpiderMonkey IC
story). But it is a large engine to write/carry in guest C, and it gains nothing for the
offline case. **Recommendation: prove it offline first (host-side), then port the engine
guest-side once it's trusted** — the IR it produces is identical, so the back half and
the correctness oracle are shared.

---

## 6. Staged prototype path (when we build it)

Each stage lands green on its own and is independently measurable (the `INTERP_PERF.md`
discipline). Every stage asserts `residual == interp == JIT` on the differential oracle.

- **Stage 0 — generic IR→IR optimizer.** Constant folding + branch resolution + DCE on
  *closed* modules, no interpreter notion yet. Proves the rewrite → re-verify → run loop
  end to end and gives a reusable pass kit. (Builds on the `resolve_imports_with`/`link`
  rewrite shape.)
- **Stage 1 — constant memory + a toy interpreter.** Add spec-time markers; treat a
  designated data region as constant; specialize a *toy accumulator-bytecode*
  interpreter against a fixed program; assert the residual matches interp/JIT.
- **Stage 2 — polyvariance + memory renaming.** Context(PC)-driven specialization with
  termination control, plus lifting the interpreter's value-stack/locals into SSA →
  straight-line residual per bytecode position. This is the weval-equivalent core.
- **Stage 3 — JIT + benchmark.** Feed the residual through the existing JIT; measure
  against the unspecialized interpreter loop. The `INTERP_PERF` kernels and a small real
  guest interpreter (e.g. the `jit_demo` bytecode machine) are the measuring sticks.

---

## 7. Risks & honest bounds

- **Polyvariance / termination.** Without a disciplined context (PC) the specializer can
  diverge or explode in code size. The context mechanism (weval's lesson) is mandatory,
  not optional.
- **Memory-renaming soundness.** Lifting in-window stack slots to SSA must be provably
  equivalent (aliasing, address-taken slots, traps mid-sequence). This is the highest
  correctness risk; the re-verify gate catches *escapes* but not *miscompiles* — the
  differential oracle is the real guard, so corpus breadth matters.
- **Compile-time cost.** PE is offline here, so this is tolerable, but a runaway
  specializer on a large interpreter is a real failure mode; bound it.
- **ROI honesty.** This pays off for **guest interpreters**, not for our own engine (the
  JIT already exists). The win is *new capability for guest-language authors*, not a
  speedup of anything we ship today. Worth stating plainly so the effort is scoped
  against that and not oversold.

---

## 8. Verdict

**Viable, and the IR is a notably good substrate** — SSA-on-the-wire, total semantics,
existing IR→IR rewrite infra, and the re-verify safety net mean SVM skips weval's hardest
prerequisite (SSA reconstruction) and gets escape-safety for free. The work that remains
is the partial-evaluation engine itself (constant memory, polyvariant specialization,
value-stack renaming), which is substantial but well-trodden ground thanks to weval. The
recommended first step is a **host-side, offline `svm-peval`** crate built in the four
stages above, with the existing verifier + differential oracle as its safety net and
correctness spec — squarely in service of the goal: **giving guest interpreters a
compiler for free.**
