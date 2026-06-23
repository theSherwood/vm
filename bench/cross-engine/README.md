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
| `wasm32/64(pulley)` | the same wasm on Wasmtime's **Pulley** portable bytecode *interpreter* (`target("pulley64")`) — the interpreter-tier baseline for `svm-bytecode`, in-process via `wasmtime-rs/` |

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

## Interpreter-tier absolute scale

The cross-engine table places the *JIT* (`svm-jit`) next to native/V8/Wasmtime-Cranelift. To place the
*interpreters* (`svm-bytecode`, `svm-tree-walk`) on an absolute scale, the table includes two
interpreter baselines that execute the **same compiled bytecode** of the same C — **Pulley**
(Wasmtime's production portable bytecode interpreter, added via `wasmtime-rs/`) and CPython (`python`,
though that interprets a Python *transliteration*, not the same compiled IR). Pulley is the fair
apples-to-apples reference: a mature switch/tail-call bytecode loop, same input, same in-process
methodology.

Indicative shape (per-iter ns; absolute numbers machine-dependent):

| kernel | `svm-tree-walk` | `svm-bytecode` | `wasm(pulley)` |
|---|--:|--:|--:|
| alu | ~51 | ~28 | ~3 |
| xorshift | ~70 | ~37 | ~14 |
| call | ~153 | ~59 | ~19 |
| fnv | ~115 | ~74 | ~7 |
| fma | ~31 | ~16 | ~9 |

Reading: **`svm-bytecode` is ~2× faster than the tree-walker** (the bytecode tier earns its keep), but
still **~3–9× behind Pulley** — real headroom in the bytecode dispatch loop (instruction encoding /
dispatch overhead), not an algorithmic gap. Both SVM interpreters are the same order of magnitude as a
production wasm interpreter, and both are ~20–50× off the JIT (cf. the steady-state table) — which is
why the JIT exists and why the break-even analysis above matters for tier selection.

## End-to-end real programs

The kernels above are single-algorithm loops. To see how the stack holds up on **whole, branchy
programs** — not tight arithmetic — `end_to_end` runs four self-contained workloads through the LLVM
frontend across all four engines, and **cross-checks every engine's result bit-exact against native**
(so it doubles as a whole-stack differential test):

```sh
cd crates/svm-llvm && cargo run --release --example end_to_end
```

- `json` — build a small JSON object from a seed, tokenize it (string/number scan + nesting depth), sum
  the integers — a realistic parser inner loop.
- `dfa` — count substrings matching `[a-z]+@[a-z]+\.[a-z]+` via a hand-coded scanner (regex/lexer shape).
- `lz` — an LZ77-style compressor (longest-match search in a sliding window) — memory + match-search heavy.
- `vm` — a tiny stack-machine **bytecode interpreter** running a generated program (an interpreter on
  the SVM).

Result: **all four engines return bit-identical to native**, and **svm-jit lands ~1.4–2.6× native
(geomean ~1.7×)** on these control-flow-heavy programs — a touch behind its ~1.3× on the straight-line
`corpus_diff` kernels, as expected where branchy code favors clang's mature backend over Cranelift. The
interpreters run these programs ~30–110× (bytecode) / ~40–155× (tree-walk) slower than the JIT — the
same tier ordering the micro-benchmarks show, now confirmed on realistic code. (The JIT uses a large
per-engine iteration count so its per-call recompile washes out; the interpreters use smaller counts —
per-iter is normalized, so the columns stay comparable.)

## Footprint — code size & memory

Every section above measures *speed*; `footprint` measures *size*, the other axis for a VM meant to
host many sandboxes. It deliberately runs from the **deployed-sandbox perspective**: the LLVM frontend
is an **AOT tool** (`svm-llvm-translate` → SVM IR), and the IR — not the compiler — is what travels to
the sandbox, so the probe lives in `svm-run` and **links no libLLVM** (verified: `ldd` shows none). It
takes an AOT-produced `.svmb`/`.svm` and reports each representation's size plus the peak process RSS to
build + hold each engine's artifact (re-exec'd per engine):

```sh
# AOT (the only place libLLVM is used — produces the shippable IR):
clang -O2 -emit-llvm -c bench/cross-engine/kernels.c -o /tmp/k.bc
( cd crates/svm-llvm && cargo run --release --bin svm-llvm-translate -- /tmp/k.bc -o /tmp/k.svmb )
# Runtime probe (no libLLVM — the real sandbox view):
cargo run -p svm-run --release --example footprint -- /tmp/k.svmb
```

Indicative (machine-dependent; ~11 functions, ~680 IR instructions):

| representation | size | vs IR |
|---|--:|--:|
| IR (encoded) | ~3.5 KB | 1.00× |
| bytecode (threaded `Vec<Op>`) | ~750 ops | ~1.1 ops/instruction |
| JIT (native code) | ~3.2 KB | **~0.9×** |

Reading: **the JIT'd native code is roughly the *same size* as the encoded IR** (~0.9×) — compact, not
bloated; bytecode expands the IR ~1.1 ops/instruction. Memory (RSS, libLLVM-free): a runtime process
baseline is **~2.6 MB**; holding the module (tree-walk) or its bytecode adds **~0**; a **JIT compile
adds ~3.4 MB resident** (Cranelift's working set) for this small module. So per-guest steady-state
memory is dominated by the *JIT compiler's working set*, not the emitted code — another argument (with
the compile-latency section) for a compiled-module cache and for running short-lived / many guests on
the bytecode tier. **None of the ~50 MB libLLVM the frontend carries is in the sandbox** — it's AOT
only. (New `svm_jit::CompiledModule::code_byte_count` and `svm_interp::bytecode::Compiled::op_count`
accessors expose the exact sizes.)

## Capability-call (host-boundary) overhead

`cap.call` is how a guest reaches the host (I/O, clock, spawn, and the durable safepoint path), yet
it's absent from the compute tables. `cap_call` times a loop of one `cap.call` to the cheapest cap (the
clock read) minus an identical no-cap loop, isolating the boundary cost on each engine — the JIT
through the real `svm_run::cap_thunk` (§9 marshalling + indirect dispatch), the interpreters through the
in-process `Host`:

```sh
cd crates/svm-llvm && cargo run --release --example cap_call
```

Indicative for the clock cap: **~49 ns/call JIT-generic, ~48 ns JIT-fast (D45), ~49 ns bytecode,
~66 ns tree-walk.** The cost is **roughly engine-independent** — it's the host boundary, not the engine
— so on the JIT a `cap.call` is **~50× a normal in-VM op** (~1 ns): cap-chatty / IO-heavy workloads are
boundary-bound, not compute-bound.

The driver times the JIT *both* ways — the generic `cap_thunk` and the §9/D45 devirtualized fast
resolver `run_powerbox` wires by default — and they come out **within ~2%**. That's a finding, not a
win: D45 removes the JIT-side arg marshalling but the fast handler still re-enters the *same*
`Host::cap_dispatch_slots`, which is essentially the entire cost for a cheap cap. Making the hot
fast-handlers do their work inline (skipping the general dispatch) is the concrete low-hanging perf
lever the slices surfaced — see **ISSUES.md I12**. (Window-touching caps — real I/O, spawn — stay on
the generic path and add their own work on top.)

## Parallel scaling — many isolated guests at once

The SVM exists to host many sandboxed guests; `parallel` measures whether *W* concurrent guests deliver
~*W*× throughput. Each OS thread runs its **own** guest (the JIT a private `CompiledModule`, the
interpreters the shared read-only `&Module`) with no shared mutable runtime state, so it probes the
runtime for hidden contention (global locks, allocator, false sharing) that would cap multi-tenancy. The
kernel is a serial `xorshift64*` (pure ALU, no memory) so only the runtime — not memory bandwidth —
could limit scaling. Runtime-only (no libLLVM):

```sh
cargo run -p svm-run --release --example parallel
```

Indicative (machine-dependent; a 4-physical-core host), scaling = `throughput(W) / (W · throughput(1))`:

| threads | jit | bytecode | tree-walk |
|---|--:|--:|--:|
| 1 | 100% | 100% | 100% |
| 2 | 99% | 100% | 100% |
| 4 (= cores) | **97%** | 96% | 100% |
| 8 (oversubscribed) | 50% | 45% | 48% |

Reading: **all three engines scale ~linearly to the physical-core count** — *W* guests ≈ *W*× aggregate
throughput, i.e. the runtime adds no contention to multi-tenant execution (no global JIT/interp lock,
no allocator hot path on the steady-state run). The plateau past the core count is ordinary
SMT/oversubscription, not a runtime bottleneck.

### Intra-guest scaling (one guest, many threads)

The companion axis: a **single** guest that spawns *W* worker threads via `thread.spawn` and joins them
— does the runtime spread *one* sandbox across cores? On the JIT, `thread.spawn` maps to a real OS
thread per guest thread (`os_thread_rt`), so this is genuine parallel execution; the interpreters
model-check thread interleavings on one OS thread, so they're excluded. Driven through `run_powerbox`
(the concurrent path serializes host access via a per-domain `Mutex<Host>`); runtime-only, no libLLVM:

```sh
cargo run -p svm-run --release --example thread_scaling
```

Indicative (4-physical-core host), `efficiency = slope(1)/slope(W)`, `speedup = W·efficiency`:

| threads | speedup | efficiency |
|---|--:|--:|
| 2 | 1.97× | 98% |
| 4 (= cores) | **3.88×** | 97% |
| 8 (oversubscribed) | 3.83× | 48% |

Reading: **one guest gets near-linear speedup to the physical-core count** (3.88× at 4 threads) — the
§12 scheduler / `Mutex<Host>` concurrent path spreads a single sandbox's threads across cores with no
meaningful serialization, despite every host call funneling through the per-domain lock (these workers
are compute-only, so they don't contend on it; cap-heavy guests would). Single-thread aggregate
(~480 Miter/s) is a touch below the independent-guest figure (~534), the cost of the concurrent path's
locking + spawn/join — paid once, not per iteration.

## Embench-IoT — externally-comparable kernels

Everything above uses *our* kernels; `embench` runs the recognized **Embench-IoT** embedded suite
through the LLVM frontend across native + all three SVM engines, for numbers comparable to published
Embench results. The source isn't vendored (mixed per-benchmark licenses) — point it at a checkout
(`bench/embench/wrapper.c` `#include`s each kernel and exposes `long run(long n)` →
`verify_benchmark`'s strict pass/fail, used as both the timed kernel and the cross-engine oracle):

```sh
curl -sSL https://github.com/embench/embench-iot/archive/refs/heads/master.tar.gz | tar xz -C /tmp
EMBENCH=/tmp/embench-iot-master cargo run -p svm-llvm --release --example embench
```

Indicative (svm-jit ÷ native; **every engine bit-exact = native, `verify`=1**):

| benchmark | ratio |
|---|--:|
| nsichneu *(big state machine)* | ~1.0× |
| crc32 | ~1.1× |
| nettle-sha256 | ~1.5× |
| ud *(LU decomposition)* | ~1.6× |
| matmult-int | ~3.9× |

**geomean ~1.6× native over 5 kernels**, all bit-identical across tree-walk / bytecode / JIT / native.
The tail is the familiar one: `matmult-int` vectorizes, so it pays the 128-bit-vs-AVX2 SIMD-width gap
(ISSUES I8), exactly like `corpus_diff`'s `matmul8`. Honestly skipped (real on-ramp coverage gaps, not
silent): `edn` (a wide-vector legalization edge — `<8 x i32>` in a context the I2 pass doesn't cover)
and `aha-mont64` (`i128` 64×64→128 Montgomery multiply). This is real third-party code — so the suite
doubles as a whole-stack differential test on programs we didn't write.


## Run

```sh
bench/cross-engine/run.sh        # prints engine,kernel,ns_per_iter CSV
```

Needs `clang`, `node`, `python3`; the SVM rows additionally need **libLLVM-18** (for `svm-llvm`), and
`run.sh` skips them with a note if it's absent. (`crates/svm/examples/megabench.rs` is a separate
hand-written-IR variant that needs no libLLVM, with its own simpler kernels — not part of this table.)

To also compare against **Wasmtime** (Cranelift JIT, like `svm-jit`) and **Pulley** (its bytecode
interpreter, the `svm-bytecode` baseline), run it against the wasm modules `run.sh` built. Two drivers:

```sh
# accurate, in-process (covers every kernel; emits both wasm{32,64}(wasmtime) Cranelift rows and
# wasm{32,64}(pulley) interpreter rows; directly comparable to the in-process V8 / SVM rows):
( cd bench/cross-engine/wasmtime-rs && cargo build --release )   # one-time: fetches + builds Wasmtime
bench/cross-engine/wasmtime-rs/target/release/wasmtime-bench bench/cross-engine/k{32,64}.wasm

# lightweight, via the wasmtime CLI (no crate build), large/small-n subtraction over subprocess spawns.
# Omits vsum: the ~7 ms process overhead can't resolve a sub-0.1 ns/iter kernel, and the large n it
# needs would read past vsum's (unwrapped) array. Use the in-process driver for vsum.
WASMTIME=/path/to/wasmtime python3 bench/cross-engine/wasmtime_bench.py
```
