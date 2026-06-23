# Partial evaluation / Futamura projection (`PEVAL.md`) — tracker

The **design lives in `DESIGN.md` §20c** (the partial-evaluation on-ramp). This file is the working
tracker for the **remaining slices**, in the repo convention (cf. the former `WASM.md`/`SCHEDULING.md`):
it is dropped once the actionable slices (1–2 below) close.

**Status: BUILT** — first Futamura projection, host-side/offline. `crates/svm-peval` is a pure
`Module → Module` transform, untrusted-for-escape (re-verified), with the differential oracle
(residual == interp == JIT) as its correctness spec.

## Done

- **Generic IR→IR optimizer** — constant folding (integer **and scalar float**), branch resolution,
  dead-block / dead-value elim, block merging, dead block-param elim, and **copy propagation +
  algebraic identities** (constant-condition `select`, `x+0`/`x*1`/`x<<0`/`x&-1`/…, and absorbing
  forms `x*0`/`x&0`/`x|-1`/`x-x`/`x^x`/`x%1`), iterated to a fixpoint. `tests/optimize.rs`.
- **Stage 1 — specialize**: online polyvariant symbolic execution; constant-memory reads fold, the
  dispatch `br_table` resolves, the interpreter loop unrolls. `tests/specialize.rs`.
- **Constant memory = caller contract** (`SpecConfig`): readonly data segment (default), arbitrary
  `const_regions`, or explicit `const_overlays`.
- **Stage 2 — value-stack renaming**: constant-address stores/loads in a private region lifted into
  SSA and elided, incl. **narrow `i8`/`i16` cells** (sign/zero re-extension) and a coexisting dynamic
  heap (`rename_is_private`).
- **Cross-function `call` inlining**: a straight-line fast path (static control flow, recursion
  unrolling) plus **CFG inlining for dynamic control flow** (data-dependent branches, loops, nested
  calls, tail calls) — the context is a symbolic call stack; one residual function still comes out.
- **Scalar float constant folding** (`f32`/`f64`): arithmetic, compares, fused multiply-add,
  float↔int conversions, reinterpret/demote/promote casts — bit-for-bit the interpreter (NaN/±0/ties),
  a trapping `trunc` folds only in range.
- **v128 (SIMD) constant folding** — *all* pure lane ops: splat, extract/replace, lane int+float
  arithmetic / compares / shifts, bitwise (and/or/xor/andnot/not/bitselect), shuffle, swizzle, FMA,
  **plus the exotic ops** (saturating add/sub, widen/narrow, lane int↔float convert, dot/dot-i8,
  ext-mul, ext-add-pairwise, pmin/pmax, avgr, popcnt, q15, any/all-true, bitmask) — each bit-for-bit
  the interpreter (peval mirrors `svm-interp`'s `simd_*` helpers). `tests/specialize.rs`
  (`folds_v128_exotic_lane_ops`, a differential oracle on `Value::V128` bytes incl. NaN lanes).
- **Indirect-call specialization**: a `call_indirect` / `return_call_indirect` (and `ref.func`) whose
  index resolves to a *constant, in-range, signature-matching* function is resolved through the
  identity module-0 table to the concrete callee and inlined like a direct call (incl. into a
  dynamic-CF callee, and via a funcref loaded from constant memory — the handler-table shape). A
  dynamic / out-of-range / mismatched index returns `Unsupported`.
- **CLI / pipeline integration**: `svm-run --specialize` exposes `specialize → verify → run/AOT`
  from the command line (`--arg`, `--const-region`, `--rename[-private]`, `--no-optimize`, and
  `-o`/`--emit-text`/`--run-args`) — usable without writing Rust. `svm_run::specialize_module` is the
  reusable library entry. `crates/svm-run/tests/specialize_cli.rs`.
- **Residual-call mode (outlining)** (`SpecConfig::outline_calls`, `svm-run --outline`): instead of
  inlining, each `(callee, arg pattern)` is specialized to a shared residual function and called — a
  multi-function residual that bounds code growth and specializes **dynamic-depth recursion** (a
  finite self-recursive residual where inlining would diverge). Composes with a rename region (see
  next bullet).
- **Selective outlining** (`SpecConfig::selective_outline`): inline straight-line and *bounded*
  recursion as usual, and outline **only an unbounded-recursion back-edge** — a call re-entering an
  activation already live on the stack with the same argument pattern. The residual is then a *tight*
  recursive function with its leaves and structure folded in, instead of one tiny function per call
  site (full `outline_calls`). Each frame carries a recursion signature (the entry argument pattern,
  empty outside selective mode, so the inline / full-outline memo keys are untouched); a back-edge is
  cut by the function-level outline memo, everything else by ordinary CFG inlining. On the Lisp `fib`
  demo this takes the residual from 13 functions to **2** and the same-backend JIT win from 2.3× to
  **~15×**. `tests/specialize.rs` (`selective_outlining_inlines_leaves_and_outlines_recursion`).
- **Outlining + renaming together**: the renamed region's live abstract cells are threaded across a
  residual call — passed in as extra arguments, returned as extra results — so the operand stack stays
  in SSA across an outlined (or selectively-outlined) call instead of forcing the region into real
  memory. The driver builds callees eagerly depth-first so a callee's out-cell signature is known
  before its `call` is emitted; the out-cell set is fixed at the first return and required to match at
  every other (mismatch / recursion-through-a-region / outlined tail-call-with-live-cells fail
  closed). `tests/specialize.rs` (`outlining_threads_a_renamed_cell_across_a_call`),
  `tests/bench.rs` (`outline_rename_threads_operand_stack_through_helpers`).
- **AOT pipeline** (`tests/aot.rs`).
- **End-to-end demo on a real interpreter** (`crates/svm-llvm/tests/peval_demo.rs`): a Brainfuck
  interpreter **written in C**, compiled `clang -O2 → LLVM → svm-llvm → svm-IR`, then specialized
  against a fixed BF program (the program is a runtime pointer clang can't fold, declared constant to
  the specializer — weval's real use case). The generic 21-block interpreter folds to a **5-block**
  compiled program (1484 → 176 bytes, 8.4× smaller); on a 2M-iteration workload the same-backend
  specialization win is **~16× (JIT)** and the end-to-end interpreted→compiled-native is **~1600×**.
  Proves the projection on frontend-emitted IR, not just hand-written toy interpreters.
- **Second demo — a recursive Lisp/Scheme tree-walker** (`crates/svm-llvm/tests/lisp_demo.rs`): the
  same on-ramp (C interpreter, `clang -O2 → svm-llvm`, opaque program pointer) on a *recursive*
  AST-walking evaluator (`let`/`if`/arithmetic/variables + guest functions), exercising **both**
  residual strategies. An **expression** program (a finite AST) fully **inlines**: the whole
  3-function/16-block tree-walker collapses to a **single 4-block straight-line formula** — the
  dispatch `switch`, node decode, and AST all gone. A **recursive** program (`fib` defined in the AST,
  dynamic depth) uses **selective outlining**: the leaves/structure inline and only the self-call
  outlines, folding into a **tight 2-function self-recursive residual** — fib(32) is **~145× (JIT)** /
  **~2100×** end-to-end over the interpreted interpreter. Two practical findings it surfaced: (1)
  clang's tail-recursion elimination loopifies the evaluator and turns `if` into a `select` of node
  indices — a *dynamic* index that defeats dispatch folding — so the demo compiles with
  `-fno-optimize-sibling-calls`; (2) a *counted* host loop (`for i in 0..n`) is unrolled by online PE
  (its induction variable looks constant each step), so the guest's only foldable looping construct is
  recursion — which is exactly where outlining earns its keep.
- **Benchmarking** — a corpus of harnesses (`size_corpus`, `gain_spectrum`, `roi_futamura_loop`,
  `fuzz_specialization_*` in `svm-peval`'s `tests/bench.rs`; `peval_corpus` in `svm-llvm`) plus a
  regenerable consolidated report. See the **Benchmarking** section below and
  [`PEVAL_BENCH.md`](PEVAL_BENCH.md). Headline: on the sum-loop, ~3.6× (interp backend) / ~7× (JIT,
  run-time only after the compile-once timing fix) specialization win, thousands× end-to-end
  interpreted-interpreter→compiled-native.

## Remaining slices (ranked by ROI)

1. **Guest-side engine (§22 `Jit` capability)** — ship the specializer inside the sandbox for
   dynamic-language IC-style recompilation (guests recompile themselves). Highest ceiling, very large
   effort (on-device re-verify, determinism/TCB review) — a project, not a slice.

## Benchmarking

**Regenerable report: [`PEVAL_BENCH.md`](PEVAL_BENCH.md)** — run
`python3 scripts/peval_bench_report.py` to rebuild it. The script runs the CSV-emitting benches in
`svm-peval` and `svm-llvm` (set `SVM_BENCH_CSV=1` to emit `CSV,<bench>,<case>,<metric>,<value>` rows)
and renders one consolidated markdown table; timings are JIT, compile-once/run-many, single-run and
machine-dependent (the report records the host).

Benches feeding it:
- `tests/bench.rs` (`svm-peval`): `size_corpus` (size across toy shapes, also a size-regression
  guard), `gain_spectrum` (the overhead-bound→work-bound run-time gradient on toy loops), and
  `roi_futamura_loop` (end-to-end Futamura on the sum-loop: ~3.6× interp / ~7× JIT specialization
  win, thousands× interpreted-interpreter→compiled-native).
- `tests/peval_corpus.rs` (`svm-llvm`): the real clang-compiled BF + Lisp interpreters across a range
  of guest programs — size, PE time, JIT-compile time, and run-time speedup.
- `tests/bench.rs::fuzz_specialization_*` (`svm-peval`): the differential oracle
  (interp == interp == jit) over random programs across four interpreter shapes; the bail surface
  (Budget / Unsupported / nonterminating) is reported and verified legitimate.

Extend the corpus with new shapes as slices land, so each one's size/speed effect is measured, not
assumed.

**Non-goals** (the engine correctly bails, not pending work): effectful / multi-result ops — atomics,
fibers/threads, host `cap.call` / imports — cannot be folded soundly.

Drop this file once the actionable slices (1–2) close.
