# Op × backend parity matrix

**Generated — do not edit by hand.** Regenerate with `cargo run -p svm-parity` after changing the manifest (`crates/svm-parity/src/`). This file is the human-readable view of the exhaustive, test-checked classifier in `svm-parity`; the conformance test (`crates/svm-parity/tests/conformance.rs`) pins every non-skipped row against what the backends actually compile.

Backends (DESIGN.md §3): the tree-walk interpreter is the **oracle** (defines observable behavior); the bytecode interpreter is held bit-exact against it; the Cranelift and wasm JITs are fail-closed accelerators that fold their non-subset back to the oracle (INVARIANTS.md #9).

## Legend

- ✅ **Full** — runs the op, observable behavior identical to the oracle (modulo the deliberately unpinned float-NaN bits / backend-local handle indices; DESIGN §3/§3a).
- ⛔ **Declines (parity not expected)** — folds to the oracle by design: a wasm-JIT concurrency/cap/fiber op (leaf accelerator), or a lowering with no target counterpart.
- 🚧 **Not yet (parity not achieved)** — a real gap this backend could close but hasn't.
- 🔶 **Conditional** — Full where a build/target cfg holds, Declines elsewhere (the note names the condition).

**459 ops.** Across the two JIT columns: 855 ✅ Full · 49 ⛔ Declines · 3 🚧 Not-yet · 11 🔶 Conditional.

## scalar integer

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `i32.const` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.div_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.div_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.rem_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.rem_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.and` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.or` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.xor` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.rotl` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.rotr` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.lt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.le_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.gt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.ge_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.clz` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.ctz` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.popcnt` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.extend8_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.extend16_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.extend32_s` | ✅ | ✅ | ✅ | ⛔ | i32.extend32_s is the identity; no wasm opcode |
| `i32.eqz` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.const` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.div_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.div_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.rem_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.rem_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.and` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.or` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.xor` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.rotl` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.rotr` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.lt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.le_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.gt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.ge_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.clz` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.ctz` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.popcnt` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.extend8_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.extend16_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.extend32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.eqz` | ✅ | ✅ | ✅ | ✅ |  |
| `select` | ✅ | ✅ | ✅ | ✅ |  |

## scalar float

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `f32.const` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.add` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.div` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.min` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.max` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.copysign` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.sqrt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.ceil` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.floor` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.trunc` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.nearest` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.lt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.le` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.gt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.ge` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.fma` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `f64.const` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.add` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.div` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.min` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.max` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.copysign` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.sqrt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.ceil` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.floor` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.trunc` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.nearest` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.lt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.le` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.gt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.ge` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.fma` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |

## conversions

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `i64.extend_i32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.extend_i32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.wrap_i64` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_sat_f32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_f32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_sat_f32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_f32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_sat_f32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_f32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_sat_f32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_f32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_sat_f64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_f64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_sat_f64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.trunc_f64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_sat_f64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_f64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_sat_f64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.trunc_f64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.convert_i32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.convert_i32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.convert_i64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.convert_i64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.convert_i32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.convert_i32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.convert_i64_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.convert_i64_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.demote_f64` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.promote_f32` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.reinterpret_i32` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.reinterpret_f32` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.reinterpret_i64` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.reinterpret_f64` | ✅ | ✅ | ✅ | ✅ |  |

## memory

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `i32.load` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.load` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.load` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.load8_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.load8_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.load16_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.load16_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load8_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load8_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load16_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load16_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load32_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.load32_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.store` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.store` | ✅ | ✅ | ✅ | ✅ |  |
| `f32.store` | ✅ | ✅ | ✅ | ✅ |  |
| `f64.store` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.store8` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.store16` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.store8` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.store16` | ✅ | ✅ | ✅ | ✅ |  |
| `i64.store32` | ✅ | ✅ | ✅ | ✅ |  |
| `mem.copy` | ✅ | ✅ | ✅ | ✅ |  |
| `mem.move` | ✅ | ✅ | ✅ | ✅ |  |
| `mem.fill` | ✅ | ✅ | ✅ | ✅ |  |

## atomics

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `i32.atomic.load` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.store` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.add` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.sub` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.and` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.or` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.xor` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.rmw.xchg` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i32.atomic.cmpxchg` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.load` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.store` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.add` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.sub` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.and` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.or` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.xor` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.rmw.xchg` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `i64.atomic.cmpxchg` | ✅ | ✅ | ✅ | ✅ | single-threaded lowering (concurrency-free module) |
| `atomic.fence` | ✅ | ✅ | ✅ | ✅ |  |
| `i32.atomic.wait` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `atomic.notify` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |

## simd v128

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `v128.const` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.load` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.store` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.splat` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.extract_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.replace_lane` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.mul` | ✅ | ✅ | ✅ | ⛔ | no i8x16.mul opcode in wasm |
| `i8x16.min_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.min_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.max_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.max_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.lt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.gt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.le_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.ge_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.all_true` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.bitmask` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.min_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.min_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.max_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.max_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.lt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.gt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.le_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.ge_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.all_true` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.bitmask` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.min_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.min_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.max_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.max_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.lt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.gt_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.le_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.ge_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.all_true` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.bitmask` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.add` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.min_s` | ✅ | ✅ | ✅ | ⛔ | no i64x2 min/max op in wasm |
| `i64x2.min_u` | ✅ | ✅ | ✅ | ⛔ | no i64x2 min/max op in wasm |
| `i64x2.max_s` | ✅ | ✅ | ✅ | ⛔ | no i64x2 min/max op in wasm |
| `i64x2.max_u` | ✅ | ✅ | ✅ | ⛔ | no i64x2 min/max op in wasm |
| `i64x2.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.lt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.lt_u` | ✅ | ✅ | ✅ | ⛔ | no unsigned i64x2 compare in wasm |
| `i64x2.gt_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.gt_u` | ✅ | ✅ | ✅ | ⛔ | no unsigned i64x2 compare in wasm |
| `i64x2.le_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.le_u` | ✅ | ✅ | ✅ | ⛔ | no unsigned i64x2 compare in wasm |
| `i64x2.ge_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.ge_u` | ✅ | ✅ | ✅ | ⛔ | no unsigned i64x2 compare in wasm |
| `i64x2.shl` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.shr_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.shr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.all_true` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.bitmask` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.add_sat_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.add_sat_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.sub_sat_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.sub_sat_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.avgr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.add_sat_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.add_sat_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.sub_sat_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.sub_sat_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.avgr_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extend_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extmul_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extend_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extmul_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extend_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extmul_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extend_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extmul_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extend_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extmul_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extend_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extmul_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extend_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extmul_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extend_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extmul_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extend_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extmul_low_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extend_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extmul_low_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extend_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extmul_high_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extend_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i64x2.extmul_high_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.narrow_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.narrow_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.narrow_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.narrow_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extadd_pairwise_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.extadd_pairwise_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extadd_pairwise_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.extadd_pairwise_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.convert_i32x4_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.convert_i32x4_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.trunc_sat_f32x4_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.trunc_sat_f32x4_u` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.demote_f64x2_zero` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.promote_low_f32x4` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.convert_low_i32x4_s` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.convert_low_i32x4_u` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.trunc_sat_f64x2_s_zero` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.trunc_sat_f64x2_u_zero` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.add` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.div` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.min` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.max` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.sqrt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.ceil` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.floor` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.trunc` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.nearest` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.lt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.gt` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.le` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.ge` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.pmin` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.pmax` | ✅ | ✅ | ✅ | ✅ |  |
| `f32x4.relaxed_madd` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `f32x4.relaxed_nmadd` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `f64x2.add` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.sub` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.mul` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.div` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.min` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.max` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.abs` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.neg` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.sqrt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.ceil` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.floor` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.trunc` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.nearest` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.eq` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.ne` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.lt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.gt` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.le` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.ge` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.pmin` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.pmax` | ✅ | ✅ | ✅ | ✅ |  |
| `f64x2.relaxed_madd` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `f64x2.relaxed_nmadd` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `v128.and` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.or` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.xor` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.andnot` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.not` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.any_true` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.popcnt` | ✅ | ✅ | ✅ | ✅ |  |
| `v128.bitselect` | ✅ | ✅ | ✅ | ✅ |  |
| `i32x4.dot_i16x8_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i16x8.dot_i8x16_s` | ✅ | ✅ | ✅ | ⛔ | no core-wasm opcode (relaxed-SIMD / scalar-fma only) |
| `i16x8.q15mulr_sat_s` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.shuffle` | ✅ | ✅ | ✅ | ✅ |  |
| `i8x16.swizzle` | ✅ | ✅ | ✅ | ✅ |  |

## calls & control

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `ref.func` | ✅ | ✅ | ✅ | ✅ |  |
| `call` | ✅ | ✅ | ✅ | ✅ |  |
| `call_indirect` | ✅ | ✅ | ✅ | ✅ |  |

## capabilities & reflection

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `cap.call` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `vcpu.tls.get` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `vcpu.tls.set` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `export.handle` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `import.attach` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `cap.self.type_id` | ✅ | ✅ | ✅ | ⛔ | host cap/handle op — serviced by the oracle, not emitted |
| `cap.self.covers` | ✅ | ✅ | ✅ | ⛔ | host cap/handle op — serviced by the oracle, not emitted |
| `call.import` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `call.import.dyn` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `call.sym` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `durable.shadow_base` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |

## process, serve & fork

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `instantiate` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `join` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `instantiate_module` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `child_offer` | ✅ | ✅ | ✅ | ⛔ | leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `svc.poll` | ✅ | ✅ | ✅ | ⛔ | native serve-loop core (svc.poll/svc.wait) for a serve-qualified module; else folds to the oracle; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `svc.wait` | ✅ | ✅ | ✅ | ⛔ | native serve-loop core (svc.poll/svc.wait) for a serve-qualified module; else folds to the oracle; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `clone_caller` | ✅ | ✅ | 🚧 | ⛔ | native on tree-walk + bytecode; Cranelift still folds (serve loop in svm-run, native-frame twin) — the next slice (FORK.md §9.1); leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `reap` | ✅ | ✅ | 🚧 | ⛔ | native on tree-walk + bytecode; Cranelift still folds (serve loop in svm-run, native-frame twin) — the next slice (FORK.md §9.1); leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `fuel.remaining` | ✅ | 🚧 | ✅ | ⛔ | declines the module (folds to the oracle) rather than adding a native op; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `exec_module` | ✅ | 🚧 | 🚧 | ⛔ | eval-loop-only image-replace (Step::Exec); the fast tiers decline the module and fold to the oracle (FORK.md §8.6); leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |

## fibers, threads & non-local control

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `cont.new` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `cont.resume` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `cont.resume.block` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `suspend` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `thread.spawn` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `thread.join` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `gc.roots` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `setjmp` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (setjmp_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |
| `longjmp` | ✅ | ✅ | 🔶 | ⛔ | Full on x86-64-unix (setjmp_rt); Declines to the interp elsewhere; leaf accelerator: folds to the bytecode interp underneath (DESIGN §3) |

## terminators

| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |
|----|:----:|:----:|:----:|:----:|-------|
| `return` | ✅ | ✅ | ✅ | ✅ |  |
| `unreachable` | ✅ | ✅ | ✅ | ✅ |  |
| `br` | ✅ | ✅ | ✅ | ✅ |  |
| `br_if` | ✅ | ✅ | ✅ | ✅ |  |
| `br_table` | ✅ | ✅ | ✅ | ✅ |  |
| `return_call` | ✅ | ✅ | ✅ | ✅ |  |
| `return_call_indirect` | ✅ | ✅ | ✅ | ✅ |  |

