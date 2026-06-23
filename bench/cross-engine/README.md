# Cross-engine micro-benchmarks

Compares the SVM backends against native, WebAssembly, JavaScript, and Python on the same compute
kernels, to place the bytecode engine and JIT on an absolute scale. **One C source** (`kernels.c`)
feeds every engine — including the SVM ones, which run IR produced by the **real LLVM frontend**
(`clang -emit-llvm` → `svm-llvm`), not hand-written IR.

## Engines

| engine | how |
|---|---|
| `native` | `clang -O2` (C kernels), timed in-process |
| `wasm32` / `wasm64` | `clang --target=wasm{32,64}` → run on Node/V8 (TurboFan, warmed up) |
| `js(v8)` | the same kernels as pure JavaScript on V8 |
| `svm-jit` | this repo's Cranelift JIT (`svm_jit::compile_and_run`), on **LLVM-frontend** IR |
| `svm-bytecode` | this repo's bytecode engine (`bytecode::compile_and_run`), on LLVM-frontend IR |
| `svm-tree-walk` | this repo's tree-walking oracle (`svm_interp::run`), on LLVM-frontend IR |
| `python` | CPython 3 |
| `wasm32/64(wasmtime)` | the same wasm on Wasmtime (Cranelift, like `svm-jit`) — in-process via `wasmtime-rs/`, or via the `wasmtime` CLI with `wasmtime_bench.py` |

The SVM rows come from compiling `kernels.c` with `clang -O2 -emit-llvm -c` (the D54 on-ramp's
legalized subset, **vectorization on**), translating the bitcode to SVM IR with
`svm_llvm::translate_bc_path`, and timing each kernel on the three engines — so they reflect IR the
toolchain actually produces. (Driver: `crates/svm-llvm/examples/cross_engine.rs`.)

## Kernels

Each loops `n` times in **int32** arithmetic. Loops are fold-resistant *by construction* — an
`a = a*1103515245 + 12345 + i` **i32-LCG** recurrence (multiplicative, so `clang`'s SCEV can't
closed-form it) or data-dependent loads — rather than inline-asm barriers, because the LLVM→SVM
on-ramp rejects inline asm. i32 throughout so JS can match via `Math.imul`.

- `alu` — the bare i32-LCG recurrence. **A demonstrator, not the headline scalar number:** clang's
  backend collapses this recurrence (4 steps → one multiply by `M⁴`), which svm-jit doesn't, so it reads
  ~8× native. It's the *only* kernel where svm-jit trails native (see ISSUES.md I9) — `xorshift` is the
  fair scalar number.
- `xorshift` — a serial scalar hash (`a ^= a<<13; a ^= a>>17; a += i`) clang **can't** strength-reduce.
  The representative scalar-throughput kernel: svm-jit ≈ native here.
- `call` — each iteration calls a `noinline` LCG `step` (a real call/return).
- `call_indirect` — same `step` dispatched through an opaque function pointer.
- `mem` — `cell = a; a = LCG(cell, i)` — a store→load the optimizers **forward and delete** (so this is
  a store→load-*forwarding* probe: it separates engines that forward/DCE memory — jit/native/wasm → ~0.3 ns
  — from interpreters that execute both ops → ~50 ns; it does **not** measure the memory path).
- `chase` — a **dependent-load pointer chase**: `idx = mem[idx]`, `size = 4096` (16 KiB, L1). Each load's
  *address* is the previous load's *value*, so the access is strictly serial — it can't be forwarded,
  hoisted, vectorized, or unrolled-for-ILP. This is the honest cross-engine memory-load kernel; the chain
  uses a constant stride (prefetcher-friendly), so it measures the engine's **load-issue / load-use path**.
- `chase_rand` — same chase, but `size = 1<<20` (4 MiB, L3) and the chain is a **full-period LCG
  permutation** (`(i*1103515245+12345) & mask`), which **defeats the hardware prefetcher** and exposes real
  **cache/DRAM latency**. On compiled engines every backend converges to the same number (they're all
  bottlenecked on the memory hierarchy, not codegen).

The chase chains are rebuilt **inside** the timed function — a fixed `O(size)` prelude that cancels in the
large/small-`n` subtraction — so no reliance on language-specific init / wasm start functions.

Three more kernels go beyond the synthetic micros:

- `fnv` — **FNV-1a-32** over a 4 KiB byte buffer, hashing `n` bytes (wrapping). A realistic
  byte-processing inner loop (byte-load + xor + mul + branch) whose hash chain is **serial** (so it
  can't be vectorized or folded). A fairer "composite" workload than the single-op micros.
- `fma` — a scalar **f64 FMA recurrence** `acc = acc*C + D`. Covers the floating-point path (everything
  else is integer); the serial FP dependency is latency-bound and not vectorizable. Returns `trunc(acc)`
  so every engine still returns an `i32`.
- `vadd` — a **vectorizable** reduction `s += (k ^ seed)` with no array (seed runtime, so it can't be
  folded; nothing to fall out of bounds). Auto-vectorizing backends collapse it to vector adds: **native
  uses AVX2 (256-bit), while wasm (the `v128` spec) and svm-jit (the on-ramp's determinism-fixed 128-bit
  legalization) use 128-bit SIMD** — so native leads `vadd` ~2×, and svm-jit lands right with the wasm
  engines (see ISSUES.md I8). The interpreters stay scalar. (Replaces the old `vsum`, whose known-content
  array let Cranelift fold the loop to a bogus ~0 on svm-jit and read out of bounds at large `n`.) The
  Wasmtime *CLI* driver omits `vadd` — its ~7 ms process spawn can't resolve a sub-0.1 ns/iter kernel
  (use the in-process `wasmtime-rs/` driver).

## Methodology

- **Per-iteration isolation:** `(time(large n) − time(small n)) / Δn`, cancelling fixed per-run cost
  (frame setup, JIT compile, V8 warmup, and the chase/array preludes). **Min over reps** rejects noise.
  The non-SVM drivers use `n = 1000 → 201000`; the SVM driver uses `n = 1000 → 2_000_000` because
  `svm_jit::compile_and_run` recompiles each call, so the run must dominate compile jitter.
- **No closed-form folding:** the kernels resist folding by construction (i32-LCG / data-dependent
  loads), so native, wasm, and SVM all honestly execute every iteration — no inline-asm barriers (which
  the LLVM→SVM on-ramp rejects).

## Compile latency & engine break-even

The table above is *steady-state* throughput — `n` is sized huge precisely so the JIT's per-call
recompile washes out. That hides a first-order property: **time-to-first-result** and **how many
iterations a workload must run before a JIT (or even a bytecode) compile pays for itself** versus just
tree-walking. The `compile_latency` driver measures it (same LLVM-frontend IR, same kernels):

```sh
cd crates/svm-llvm && cargo run --release --example compile_latency
```

It reports, per kernel: one-time **translate** (LLVM bitcode → SVM IR), each engine's **cold** cost
(the `n → 0` intercept of `T(n) = cold + n·iter` — the fixed per-`compile_and_run` cost), the
steady-state **per-iter**, and the **break-even** iteration counts (`cold + n·iter` crossover between
engines, i.e. compile once + run `n` times).

Indicative shape of the result (absolute numbers are machine-dependent):

- **JIT cold ≈ 5–6 ms** (whole-module Cranelift codegen — `compile_and_run` recompiles *every* call,
  so it's ~constant across kernels regardless of entry) **+ ~3.6 ms translate ≈ ~9 ms to first
  result.** **Bytecode cold ≈ 30 µs** (~160× cheaper to start); **tree-walk cold ≈ 0** (pure frame
  setup).
- **Break-even ≈ 10⁵–10⁶ iterations**: below that the bytecode engine is the right tier; the JIT's
  per-iter win (~1–2 ns vs ~30–70 ns) only repays its compile past ~75k–450k iterations.
- This is the dominant inefficiency the steady-state table omits, and it directly motivates a
  **compiled-module cache** (compile once, reuse across invocations) — today every `compile_and_run`
  recompiles from scratch.

Caveat: a kernel with a large in-function *prelude* (e.g. `fnv`'s 4 KiB buffer fill) inflates the
interpreters' `cold` (the prelude is fixed per-call work, not compile); the JIT runs that prelude in
microseconds so its `cold` stays ~pure compile.


## Run

```sh
bench/cross-engine/run.sh        # prints engine,kernel,ns_per_iter CSV
```

Needs `clang`, `node`, `python3`; the SVM rows additionally need **libLLVM-18** (for `svm-llvm`), and
`run.sh` skips them with a note if it's absent. (`crates/svm/examples/megabench.rs` is a separate
hand-written-IR variant that needs no libLLVM, with its own simpler kernels — not part of this table.)

To also compare against **Wasmtime** (Cranelift, like `svm-jit`), run it against the wasm modules
`run.sh` built. Two drivers:

```sh
# accurate, in-process (covers every kernel incl. vsum; directly comparable to the in-process V8 rows):
( cd bench/cross-engine/wasmtime-rs && cargo build --release )   # one-time: fetches + builds Wasmtime
bench/cross-engine/wasmtime-rs/target/release/wasmtime-bench bench/cross-engine/k{32,64}.wasm

# lightweight, via the wasmtime CLI (no crate build), large/small-n subtraction over subprocess spawns.
# Omits vsum: the ~7 ms process overhead can't resolve a sub-0.1 ns/iter kernel, and the large n it
# needs would read past vsum's (unwrapped) array. Use the in-process driver for vsum.
WASMTIME=/path/to/wasmtime python3 bench/cross-engine/wasmtime_bench.py
```
