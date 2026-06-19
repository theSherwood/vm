#!/usr/bin/env bash
# Cross-engine micro-benchmark runner: native (clang -O2), wasm32 + wasm64 (clang → Node/V8), pure JS
# (V8), the three SVM engines (jit / bytecode / tree-walk) via the real LLVM frontend, and CPython.
# Each engine reports `engine,kernel,ns_per_iter` for the same kernels, with per-iteration compute
# isolated by large/small-`n` subtraction and taken as the min over reps.
#
# Methodology notes:
#   * ONE C source (kernels.c) feeds native, wasm, AND svm — the SVM rows come from compiling that
#     same C through `clang -O2 -emit-llvm` → svm-llvm (the D54 on-ramp), so the SVM IR is what the
#     toolchain actually produces, not hand-written.
#   * All kernels do i32 arithmetic; loops are fold-resistant *by construction* (multiplicative
#     i32-LCG recurrences, data-dependent loads) rather than inline-asm barriers, so the same source
#     survives the LLVM→SVM on-ramp (which rejects inline asm).
#   * `alu` is a *demonstrator* (clang collapses its LCG recurrence → ~8x native; svm-jit doesn't);
#     `xorshift` is the representative scalar-throughput kernel (svm-jit ≈ native). `mem` forwards a
#     store→load (compilers delete it; interpreters execute it). `vadd` is a vectorizable reduction:
#     native uses AVX2 (-mavx2, 256-bit), wasm + svm-jit use 128-bit SIMD (the wasm v128 spec / the
#     on-ramp's determinism-fixed legalization), so native leads vadd by ~2x and svm-jit ≈ wasm.
#
# Requires: clang, node, python3; the SVM rows additionally need libLLVM-18 (for svm-llvm). Run:
#   bench/cross-engine/run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)

echo "engine,kernel,ns_per_iter"

# --- native + wasm (clang) ---
# native uses -mavx2 so `vadd` shows native's real 256-bit SIMD width; wasm/svm are 128-bit.
clang -O2 -mavx2 -c kernels.c -o kernels.o
clang -O2 -mavx2 native_bench.c kernels.o -o native_bench
./native_bench
clang --target=wasm32 -O2 -msimd128 -nostdlib -Wl,--no-entry -Wl,--export-all -o k32.wasm kernels.c
clang --target=wasm64 -O2 -msimd128 -nostdlib -Wl,--no-entry -Wl,--export-all -o k64.wasm kernels.c
node wasmrun.mjs k32.wasm k64.wasm
node js.mjs

# --- SVM engines via the real LLVM frontend (clang → bitcode → svm-llvm → SVM IR) ---
# svm-llvm links libLLVM-18 and is excluded from the workspace, so it builds independently.
if ! ( cd "$ROOT/crates/svm-llvm" && cargo run --release --quiet --example cross_engine ); then
  echo "note: SVM rows skipped (svm-llvm needs libLLVM-18 + clang)" >&2
fi

# --- CPython ---
python3 bench.py
