# Partial evaluation / Futamura projection (`PEVAL.md`) — tracker

The **design is folded into `DESIGN.md` §20c** (the partial-evaluation on-ramp). This file is
the working tracker for the **remaining gaps**, in the repo convention (cf. the former
`WASM.md`/`SCHEDULING.md`): it is dropped once the actionable gaps close.

**Status: BUILT — first Futamura projection, host-side/offline, Stages 0–2 + AOT + ROI.**
`crates/svm-peval` is a pure `Module → Module` transform, untrusted-for-escape (re-verified),
with the differential oracle (residual == interp == JIT) as its correctness spec.

## Done

- **Generic IR→IR optimizer** — constant folding (matches interp arithmetic exactly; trapping
  ops preserved), branch resolution, dead-block elim, dead-value elim (exhaustive operand
  remapper), block merging, dead block-param elim; iterated to a fixpoint. `tests/optimize.rs`
  (incl. a randomized differential fuzz).
- **Stage 1 — first Futamura projection** (`specialize`): online polyvariant symbolic execution;
  constant-memory program reads fold, dispatch `br_table` resolves, the interpreter loop unrolls.
  `tests/specialize.rs` (toy accumulator interpreter → compiled residual; static/dynamic branch).
- **Constant memory = caller contract** (`SpecConfig`): readonly segment (default), arbitrary
  `const_regions`, or explicit `const_overlays`. No readonly requirement.
- **Stage 2 — value-stack renaming**: a private window range's constant-address full-width
  stores/loads are lifted into SSA and elided; `rename_is_private` lets a dynamic heap coexist.
- **Wider value-op coverage**: float/SIMD arithmetic, casts, conversions, pointer ops emitted
  faithfully into the residual (not folded, but dispatch still removed).
- **Cross-function `call` (inlining)** — a direct `Call` (and a `return_call` tail call) is inlined
  at the call site: the callee's CFG is traced in the caller's context, sharing the same abstract
  memory (so a callee that folds constant memory or touches the renamed operand stack behaves as if
  written inline), and the call disappears. The callee is traced straight-line — its branches must
  resolve statically, so static recursion unrolls (bounded by an inline-fuel budget); a callee that
  stays dynamically-branching returns `Unsupported`. `tests/specialize.rs` (leaf helper, unrolled
  static recursion, tail call, a call-threaded interpreter whose helper calls fold away, and the
  dynamic-branch boundary).
- **AOT pipeline** (`tests/aot.rs`): `specialize → verify → encode_module → decode_module →
  run/JIT`, all agreeing — the shippable-artifact path.
- **ROI** (`tests/bench.rs`, `#[ignore]`): specialization ~5–6× on either backend; end-to-end
  interpreted→compiled-native ~470× on a lean register machine.

## Open gaps (the reason this file still exists)

- **Cross-function `call` with *dynamic* control flow** — the inliner (above) traces a callee
  straight-line, so a callee with a data-dependent branch that survives specialization returns
  `Unsupported`. Inlining its CFG as residual blocks (splitting the caller block at the call and
  threading live values through the callee) is the remaining step; indirect calls
  (`call_indirect`) are also still out of scope.
- **Narrow (`i8`/`i16`) renamed cells** — char/short locals kept *on the renamed operand stack*
  (sub-word abstract cells with extension + partial-overlap handling). Word-width already works;
  narrow memory outside the rename region already works (residual / readonly fold).
- **Float/SIMD constant folding** — float/SIMD ops pass through unfolded (the abstract domain
  tracks integer constants only). NaN/rounding fidelity is the reason to be deliberate here.
- **Guest-side engine** — porting the specializer inside the sandbox on the §22 `Jit` capability
  (dynamic-language IC-style recompilation). The residual IR + back half are shared; deferred.

Drop this file once dynamic-control-flow `call` + narrow cells land (the rest are enhancements, not
gaps).
