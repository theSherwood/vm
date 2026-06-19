# Cross-engine micro-benchmarks

Compares the SVM backends against native, WebAssembly, JavaScript, and Python on the same compute
kernels, to place the bytecode engine and JIT on an absolute scale.

## Engines

| engine | how |
|---|---|
| `native` | `clang -O2` (C kernels), timed in-process |
| `wasm32` / `wasm64` | `clang --target=wasm{32,64}` → run on Node/V8 (TurboFan, warmed up) |
| `js(v8)` | the same kernels as pure JavaScript on V8 |
| `svm-jit` | this repo's Cranelift JIT (`svm_jit::compile_and_run`) |
| `svm-bytecode` | this repo's bytecode engine (`bytecode::compile_and_run`) |
| `svm-tree-walk` | this repo's tree-walking oracle (`svm_interp::run`) |
| `python` | CPython 3 |
| `wasm32/64(wasmtime)` | the same wasm on Wasmtime (Cranelift, like `svm-jit`) — in-process via `wasmtime-rs/`, or via the `wasmtime` CLI with `wasmtime_bench.py` |

## Kernels

Each loops `n` times in **int32** arithmetic (matching the SVM kernels' i32 ops):

- `alu` — `acc += n; n -= 1` (scalar/branch recurrence)
- `call` — each iteration calls a leaf `+1` function
- `call_indirect` — same, dispatched through a function pointer / table
- `mem` — `store acc; acc = load + 1` at a *fixed* address each iteration. **Degenerate on purpose**:
  every optimizing engine forwards the store into the load and deletes the access, so this is really a
  store→load-*forwarding* probe — it separates engines that forward/DCE memory (jit/native/wasm → ~0.3 ns)
  from those that don't (interpreters → 70+ ns). It does **not** measure the memory path.
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
  scalar backend (the SVM JIT's Cranelift, and the interpreters) stays scalar — exposing the
  vectorization gap. Valid only for `n ≤ 262144` (the array isn't wrapped, so the loads stay a clean
  affine sweep the vectorizer can analyze); the in-process drivers use `n = 201000`, and the Wasmtime
  CLI driver omits `vsum` (its ~7 ms process overhead can't resolve a sub-0.1 ns/iter kernel anyway).

## Methodology

- **Per-iteration isolation:** `(time(n=201000) − time(n=1000)) / 200000`, cancelling fixed per-run
  cost (frame setup, JIT compile, V8 warmup). **Min over reps** rejects scheduler noise.
- **No closed-form folding:** the C kernels carry a zero-instruction barrier (`DNO`) so `clang` can't
  fold e.g. `alu` into `n(n+1)/2`; native and wasm execute every iteration. The barrier emits no
  instructions, so codegen is otherwise unaffected.
- **`mem` is plain (non-volatile):** every engine's optimizer treats the redundant store→load
  naturally — interpreters execute both ops, JITs may forward — matching the SVM IR, which doesn't
  forbid forwarding.

## Run

```sh
bench/cross-engine/run.sh        # prints engine,kernel,ns_per_iter CSV
```

Needs `clang`, `node`, `python3`. The SVM rows come from the `megabench` example
(`crates/svm/examples/megabench.rs`), built in release.

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
