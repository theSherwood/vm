#!/usr/bin/env bash
# Cross-engine micro-benchmark runner: native (clang -O2), wasm32 + wasm64 (clang → Node/V8),
# pure JS (V8), the three SVM engines (jit / bytecode / tree-walk), and CPython. Each engine reports
# `engine,kernel,ns_per_iter` for the same four kernels (alu, call, call_indirect, mem), with
# per-iteration compute isolated by large/small-`n` subtraction and taken as the min over reps.
#
# Methodology notes:
#   * All kernels do int32 arithmetic, mirroring the SVM kernels' i32 ops.
#   * The C kernels carry a zero-instruction optimization barrier (DNO) so the compiler can't
#     closed-form-fold the loop — native AND wasm honestly execute all n iterations.
#   * `mem` uses plain (non-volatile) memory, so each engine's optimizer treats the store→load
#     naturally (interpreters execute both ops; JITs may forward) — matching the SVM IR.
#
# Requires: clang, node, python3, and a release build of the `megabench` example. Run from repo root:
#   bench/cross-engine/run.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)

echo "engine,kernel,ns_per_iter"

# --- native + wasm (clang) ---
clang -O2 -c kernels.c -o kernels.o
clang -O2 native_bench.c kernels.o -o native_bench
./native_bench
clang --target=wasm32 -O2 -nostdlib -Wl,--no-entry -Wl,--export-all -o k32.wasm kernels.c
clang --target=wasm64 -O2 -nostdlib -Wl,--no-entry -Wl,--export-all -o k64.wasm kernels.c
node wasmrun.mjs k32.wasm k64.wasm
node js.mjs

# --- SVM engines (jit / bytecode / tree-walk) ---
( cd "$ROOT" && cargo run --release --quiet --example megabench -p svm )

# --- CPython ---
python3 bench.py
