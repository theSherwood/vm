# Partial evaluation / Futamura projection (`PEVAL.md`) — tracker

The **design lives in `DESIGN.md` §20c** (the partial-evaluation on-ramp). This file is the working
tracker for the **remaining slices**, in the repo convention (cf. the former `WASM.md`/`SCHEDULING.md`):
it is dropped once the actionable slices (1–3 below) close.

**Status: BUILT** — first Futamura projection, host-side/offline. `crates/svm-peval` is a pure
`Module → Module` transform, untrusted-for-escape (re-verified), with the differential oracle
(residual == interp == JIT) as its correctness spec.

## Done

- **Generic IR→IR optimizer** — constant folding (integer **and scalar float**), branch resolution,
  dead-block / dead-value elim, block merging, dead block-param elim, to a fixpoint. `tests/optimize.rs`.
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
- **Indirect-call specialization**: a `call_indirect` / `return_call_indirect` (and `ref.func`) whose
  index resolves to a *constant, in-range, signature-matching* function is resolved through the
  identity module-0 table to the concrete callee and inlined like a direct call (incl. into a
  dynamic-CF callee, and via a funcref loaded from constant memory — the handler-table shape). A
  dynamic / out-of-range / mismatched index returns `Unsupported`.
- **AOT pipeline** (`tests/aot.rs`) and **ROI** (`tests/bench.rs`, `#[ignore]`).

## Remaining slices (ranked by ROI)

1. **CLI / pipeline integration** — expose `specialize → verify → run/AOT` from `svm-run` so the
   feature is usable without writing Rust. Low–medium effort; best effort-to-value ratio.
2. **Optimizer copy-forwarding + light algebraic identities** — add a value-forward/copy op (Stage 0
   has none today), tightening `select`/param folding and residual width; cheap identities (`x+0`,
   `x*1`, `x*2→shl`). Low effort, incremental (partly redundant with the JIT/LLVM backend).
3. **Residual-call mode (bounded interprocedural, shared)** — specialize a callee once and emit a
   shared residual *call* (live memory cells threaded as params) instead of always inlining, to bound
   code growth / avoid the block budget on large programs. Medium–large effort; matters at scale.
4. **v128 (SIMD) constant folding** — fold the 128-bit lane ops (a separate, larger lane/shape/shuffle
   surface with its own `Known` representation). Medium–large effort.
5. **Guest-side engine (§22 `Jit` capability)** — ship the specializer inside the sandbox for
   dynamic-language IC-style recompilation (guests recompile themselves). Highest ceiling, very large
   effort (on-device re-verify, determinism/TCB review) — a project, not a slice.

**Non-goals** (the engine correctly bails, not pending work): effectful / multi-result ops — atomics,
fibers/threads, host `cap.call` / imports — cannot be folded soundly.

Drop this file once the actionable slices (1–3) close.
