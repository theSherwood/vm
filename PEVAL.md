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
  finite self-recursive residual where inlining would diverge). Requires no rename region.
- **AOT pipeline** (`tests/aot.rs`).
- **Benchmarking** (`tests/bench.rs`): `size_corpus` (a normal test, also a size-regression guard)
  reports blocks / insts / encoded `.svmb` bytes for interpreter → residual → optimized across four
  shapes (register machine: constant / straight-line / runtime-loop, plus a renamed stack machine) —
  e.g. the constant program shrinks 236→44 bytes, the runtime-loop residual drops the whole dispatch
  table (272→131 bytes) while keeping a compiled loop. `roi_futamura_loop` (`#[ignore]`) reports
  speed + size: ~3.6× (interp backend) / ~3.3× (JIT) specialization win, ~2780× end-to-end
  interpreted→compiled-native on the sum-loop. Print with `--nocapture` (`--ignored` for the timing).

## Remaining slices (ranked by ROI)

1. **v128 (SIMD) constant folding** — fold the 128-bit lane ops (a separate, larger lane/shape/shuffle
   surface with its own `Known` representation). Medium–large effort.
2. **Outlining + renaming together** — thread the renamed region's live abstract cells across a
   residual call (as extra params/results), so outlining works even with a rename region (today it
   requires `rename = None`). Medium effort; only needed for very large renamed interpreters.
3. **Guest-side engine (§22 `Jit` capability)** — ship the specializer inside the sandbox for
   dynamic-language IC-style recompilation (guests recompile themselves). Highest ceiling, very large
   effort (on-device re-verify, determinism/TCB review) — a project, not a slice.

## Benchmarking

Built — see the `tests/bench.rs` bullet under **Done**: `size_corpus` reports residual program size
(blocks / insts / `.svmb` bytes) across a corpus of interpreter shapes and guards against size
regressions; `roi_futamura_loop` reports size + speed on the runtime-loop workload. Extend the corpus
with new shapes as slices land, so each one's size/speed effect is measured, not assumed.

**Non-goals** (the engine correctly bails, not pending work): effectful / multi-result ops — atomics,
fibers/threads, host `cap.call` / imports — cannot be folded soundly.

Drop this file once the actionable slices (1–2) close.
