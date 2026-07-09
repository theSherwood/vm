# AGENTS.md

Working agreement for agents (and humans) building this project. Keep it short;
keep it followed. The full design lives in `DESIGN.md`.

## Prime directive: keep it simple

This is a sandbox VM whose entire value is a **small, trustworthy core**. Every
line is potential TCB. Prefer the boring, obvious implementation. Don't add
abstraction, configurability, or cleverness until something concrete demands it.
If a change makes the verifier or the confinement path harder to read, it is
probably wrong. When in doubt, do less.

## Tests, fuzzing, benchmarks — early, not eventually

- **Tests from the first commit.** Every component lands with tests. The
  interpreter is the oracle: differential-test the JIT against it (D-notes in
  `DESIGN.md` §18). Tests should gate the CI.
- **Fuzz from day one.** Two invariants get fuzzed continuously:
  1. *verified ⇒ cannot escape* (fuzz the verifier),
  2. *every memory access is masked to `[0, size)` or proven bounded* (fuzz the
     confinement-masking lowering as its own unit — it is the security hinge, §4).
- **Benchmark as soon as there's anything to run.** Stand up a benchmark harness
  early and watch it over time; we are measured *relative to wasm/Wasmtime*
  (`DESIGN.md` §1a). Catch regressions when they're one commit old, not one
  release old.
  **Update ISSUES.md with any flaky CI problems.** Catch and log flakiness early so that we have visibility and can track a fix.

## Performance philosophy: data-oriented design

Most of our speed comes from **reducing allocation and improving cache locality**,
not from micro-optimizing hot code. Default to:

- **Flat data structures.** Prefer arrays / structs-of-arrays over trees of
  pointer-chasing nodes. Index with integers, not pointers, where it keeps things
  flat and relocatable.
- **Arenas / bump allocation.** Allocate per-phase (per-module, per-function)
  into arenas and free in one shot. Avoid per-node heap allocation and avoid
  scattered ownership.
- **Few, predictable passes over contiguous memory.** The decode+verify design is
  a single linear forward pass for a reason — keep that shape elsewhere too.
- Measure before optimizing beyond this; the benchmark harness is the arbiter.

## Security posture (the bar we hold)

- Target is **"as secure as wasm for the host"** — i.e. as secure as Wasmtime, not
  a proof of escape-impossibility (`DESIGN.md` §1a).
- The verifier secures typing, control flow, and index ranges. **Memory
  confinement is the masking lowering, not the verifier** — treat that pass as the
  most sensitive code in the tree.
- In-process isolation is defense-in-depth, **not** a Spectre boundary; distrust
  means separate processes.
