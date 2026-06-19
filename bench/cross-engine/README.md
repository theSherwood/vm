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

## Kernels

Each loops `n` times in **int32** arithmetic (matching the SVM kernels' i32 ops):

- `alu` — `acc += n; n -= 1` (scalar/branch recurrence)
- `call` — each iteration calls a leaf `+1` function
- `call_indirect` — same, dispatched through a function pointer / table
- `mem` — `store acc; acc = load + 1` at a fixed address each iteration

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
