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

The SVM rows come from compiling `kernels.c` with `clang -O2 -emit-llvm -c -fno-vectorize
-fno-slp-vectorize` (the D54 on-ramp's legalized subset), translating the bitcode to SVM IR with
`svm_llvm::translate_bc_path`, and timing each kernel on the three engines — so they reflect IR the
toolchain actually produces. (Driver: `crates/svm-llvm/examples/cross_engine.rs`.)

## Kernels

Each loops `n` times in **int32** arithmetic. Loops are fold-resistant *by construction* — an
`a = a*1103515245 + 12345 + i` **i32-LCG** recurrence (multiplicative, so `clang`'s SCEV can't
closed-form it), data-dependent loads, or opaque pointers — rather than inline-asm barriers, because
the LLVM→SVM on-ramp rejects inline asm. i32 throughout so JS can match via `Math.imul`.

- `alu` — the bare i32-LCG recurrence.
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
- `vsum` — a contiguous **i32 reduction** `sum += arr[k]` over a 1 MiB array. A *vectorizable* loop:
  auto-vectorizing backends (native AVX, wasm SIMD via `-msimd128`) collapse it to vector adds, while a
  scalar backend (the interpreters) stays scalar — exposing the vectorization gap. **Omitted from the
  SVM rows**: the on-ramp legalizes with `-fno-vectorize` (the MVP is scalar) *and* the opaque-pointer
  barrier doesn't survive LLVM→SVM→Cranelift, so the reduction folds to a bogus ~0 ns. Valid only for
  `n ≤ 262144` (unwrapped affine sweep); the in-process drivers use `n = 201000`, and the Wasmtime CLI
  driver omits it (its ~7 ms process overhead can't resolve a sub-0.1 ns/iter kernel anyway).

## Methodology

- **Per-iteration isolation:** `(time(large n) − time(small n)) / Δn`, cancelling fixed per-run cost
  (frame setup, JIT compile, V8 warmup, and the chase/array preludes). **Min over reps** rejects noise.
  The non-SVM drivers use `n = 1000 → 201000`; the SVM driver uses `n = 1000 → 2_000_000` because
  `svm_jit::compile_and_run` recompiles each call, so the run must dominate compile jitter.
- **No closed-form folding:** the kernels resist folding by construction (i32-LCG / data-dependent
  loads), so native, wasm, and SVM all honestly execute every iteration — no inline-asm barriers (which
  the LLVM→SVM on-ramp rejects).

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
