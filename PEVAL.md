# Partial evaluation / Futamura projection (`PEVAL.md`) — tracker

The **design lives in `DESIGN.md` §20c** (the partial-evaluation on-ramp). This file is the working
tracker for the **remaining slices**, in the repo convention (cf. the former `WASM.md`/`SCHEDULING.md`):
it is dropped once the actionable slices (1–2 below) close.

**Status: BUILT** — first Futamura projection, host-side/offline. `crates/svm-peval` is a pure
`Module → Module` transform, untrusted-for-escape (re-verified), with the differential oracle
(residual == interp == JIT) as its correctness spec.

## Done

- **Generic IR→IR optimizer** — constant folding (integer **and scalar float**), branch resolution,
  dead-block / dead-value elim, block merging, dead block-param elim, and **copy propagation +
  algebraic identities** (constant-condition `select`, `x+0`/`x*1`/`x<<0`/`x&-1`/…, and absorbing
  forms `x*0`/`x&0`/`x|-1`/`x-x`/`x^x`/`x%1`), iterated to a fixpoint. `tests/optimize.rs`.
- **Stage 1 — specialize**: online polyvariant symbolic execution; constant-memory reads fold, the
  dispatch `br_table` resolves, the interpreter loop unrolls. `tests/specialize.rs`.
- **Constant memory = caller contract** (`SpecConfig`): readonly data segment (default), arbitrary
  `const_regions`, or explicit `const_overlays`.
- **Stage 2 — value-stack renaming**: constant-address stores/loads in a private region lifted into
  SSA and elided, incl. **narrow `i8`/`i16` cells** (sign/zero re-extension) and a coexisting dynamic
  heap (`rename_is_private`).
- **Cross-function `call` inlining**: a straight-line fast path (static control flow, recursion
  unrolling) plus **CFG inlining for dynamic control flow** (data-dependent branches, loops, nested
  calls, tail calls) — the context is a symbolic call stack; one residual function still comes out.
- **Scalar float constant folding** (`f32`/`f64`): arithmetic, compares, fused multiply-add,
  float↔int conversions, reinterpret/demote/promote casts — bit-for-bit the interpreter (NaN/±0/ties),
  a trapping `trunc` folds only in range.
- **v128 (SIMD) constant folding** — *all* pure lane ops: splat, extract/replace, lane int+float
  arithmetic / compares / shifts, bitwise (and/or/xor/andnot/not/bitselect), shuffle, swizzle, FMA,
  **plus the exotic ops** (saturating add/sub, widen/narrow, lane int↔float convert, dot/dot-i8,
  ext-mul, ext-add-pairwise, pmin/pmax, avgr, popcnt, q15, any/all-true, bitmask) — each bit-for-bit
  the interpreter (peval mirrors `svm-interp`'s `simd_*` helpers). `tests/specialize.rs`
  (`folds_v128_exotic_lane_ops`, a differential oracle on `Value::V128` bytes incl. NaN lanes).
- **Indirect-call specialization**: a `call_indirect` / `return_call_indirect` (and `ref.func`) whose
  index resolves to a *constant, in-range, signature-matching* function is resolved through the
  identity module-0 table to the concrete callee and inlined like a direct call (incl. into a
  dynamic-CF callee, and via a funcref loaded from constant memory — the handler-table shape). A
  dynamic / out-of-range / mismatched index returns `Unsupported`.
- **CLI / pipeline integration**: `svm-run --specialize` exposes `specialize → verify → run/AOT`
  from the command line (`--arg`, `--const-region`, `--rename[-private]`, `--no-optimize`, and
  `-o`/`--emit-text`/`--run-args`) — usable without writing Rust. `svm_run::specialize_module` is the
  reusable library entry. `crates/svm-run/tests/specialize_cli.rs`.
- **Residual-call mode (outlining)** (`SpecConfig::outline_calls`, `svm-run --outline`): instead of
  inlining, each `(callee, arg pattern)` is specialized to a shared residual function and called — a
  multi-function residual that bounds code growth and specializes **dynamic-depth recursion** (a
  finite self-recursive residual where inlining would diverge). Composes with a rename region (see
  next bullet).
- **Selective outlining** (`SpecConfig::selective_outline`): inline straight-line and *bounded*
  recursion as usual, and outline **only an unbounded-recursion back-edge** — a call re-entering an
  activation already live on the stack with the same argument pattern. The residual is then a *tight*
  recursive function with its leaves and structure folded in, instead of one tiny function per call
  site (full `outline_calls`). Each frame carries a recursion signature (the entry argument pattern,
  empty outside selective mode, so the inline / full-outline memo keys are untouched); a back-edge is
  cut by the function-level outline memo, everything else by ordinary CFG inlining. On the Lisp `fib`
  demo this takes the residual from 13 functions to **2** and the same-backend JIT win from 2.3× to
  **~15×**. `tests/specialize.rs` (`selective_outlining_inlines_leaves_and_outlines_recursion`).
- **Outlining + renaming together**: the renamed region's live abstract cells are threaded across a
  residual call — passed in as extra arguments, returned as extra results — so the operand stack stays
  in SSA across an outlined (or selectively-outlined) call instead of forcing the region into real
  memory. The driver builds callees eagerly depth-first so a callee's out-cell signature is known
  before its `call` is emitted; the out-cell set is fixed at the first return and required to match at
  every other (mismatch / recursion-through-a-region / outlined tail-call-with-live-cells fail
  closed). `tests/specialize.rs` (`outlining_threads_a_renamed_cell_across_a_call`),
  `tests/bench.rs` (`outline_rename_threads_operand_stack_through_helpers`).
- **AOT pipeline** (`tests/aot.rs`).
- **End-to-end demo on a real interpreter** (`crates/svm-llvm/tests/peval_demo.rs`): a Brainfuck
  interpreter **written in C**, compiled `clang -O2 → LLVM → svm-llvm → svm-IR`, then specialized
  against a fixed BF program (the program is a runtime pointer clang can't fold, declared constant to
  the specializer — weval's real use case). The generic 21-block interpreter folds to a **5-block**
  compiled program (1484 → 176 bytes, 8.4× smaller); on a 2M-iteration workload the same-backend
  specialization win is **~16× (JIT)** and the end-to-end interpreted→compiled-native is **~1600×**.
  Proves the projection on frontend-emitted IR, not just hand-written toy interpreters.
- **Second demo — a recursive Lisp/Scheme tree-walker** (`crates/svm-llvm/tests/lisp_demo.rs`): the
  same on-ramp (C interpreter, `clang -O2 → svm-llvm`, opaque program pointer) on a *recursive*
  AST-walking evaluator (`let`/`if`/arithmetic/variables + guest functions), exercising **both**
  residual strategies. An **expression** program (a finite AST) fully **inlines**: the whole
  3-function/16-block tree-walker collapses to a **single 4-block straight-line formula** — the
  dispatch `switch`, node decode, and AST all gone. A **recursive** program (`fib` defined in the AST,
  dynamic depth) uses **selective outlining**: the leaves/structure inline and only the self-call
  outlines, folding into a **tight 2-function self-recursive residual** — fib(32) is **~145× (JIT)** /
  **~2100×** end-to-end over the interpreted interpreter. Two practical findings it surfaced: (1)
  clang's tail-recursion elimination loopifies the evaluator and turns `if` into a `select` of node
  indices — a *dynamic* index that defeats dispatch folding — so the demo compiles with
  `-fno-optimize-sibling-calls`; (2) a *counted* host loop (`for i in 0..n`) is unrolled by online PE
  (its induction variable looks constant each step), so the guest's only foldable looping construct is
  recursion — which is exactly where outlining earns its keep.
- **Benchmarking** — a corpus of harnesses (`size_corpus`, `gain_spectrum`, `roi_futamura_loop`,
  `fuzz_specialization_*` in `svm-peval`'s `tests/bench.rs`; `peval_corpus` in `svm-llvm`) plus a
  regenerable consolidated report. See the **Benchmarking** section below and
  [`PEVAL_BENCH.md`](PEVAL_BENCH.md). Headline: on the sum-loop, ~3.6× (interp backend) / ~7× (JIT,
  run-time only after the compile-once timing fix) specialization win, thousands× end-to-end
  interpreted-interpreter→compiled-native.

## Remaining work

### Guest-side specialization — the plan9 substrate goal

**Goal.** svm is meant to be the substrate for a plan9-like OS where freely-shared programs run in
nested sandboxes. Part of that vision: **from within svm, a process can specialize a script and get a
residual that runs in svm** — e.g. take a JS source + a QuickJS interpreter (both as svm-IR) and
partial-evaluate them into a smaller/faster residual that runs *without* the interpreter, then share
that residual over the wire / run it in a nested sandbox. The classic first Futamura projection, but
performed *in-sandbox by an ordinary program*, not host-side/offline.

**Explicit non-goal: no speculative / V8-style engine.** No type-feedback, inline caches, speculation,
or deoptimization guards. We only want the **sound** projection we already have (constants that are
*provably* constant get folded). For a dynamic language this means the residual removes the
parser/bytecode-dispatch/decode (invariant given the program) but **keeps** the dynamic value runtime
(GC, boxing, type-dispatch on runtime values) — a real but bounded win, not native speed. That is
accepted; chasing the work-bound part is what needs speculation, and we're not doing it.

**Security/architecture — already settled, no new escape-TCB.** The §22 `Jit` capability is built: a
guest hands the VM a serialized IR blob, the VM `decode → verify → compile → invoke`s it, and **the
trust hinge is verification, not the producer**. So the specializer is *just another untrusted
program*: however buggy, its residual is re-verified before a single instruction runs, a bad residual
only hurts the guest that ran it (confined to its own sandbox), and guest-side specialization adds
**zero escape-TCB surface** (DESIGN §22). This fits the plan9 ethos: anyone can write/share/improve a
specializer; running someone else's is safe by construction. The "run the residual" half is done; the
"residual IR + back half" are shared with the host-side engine (DESIGN §20c).

**The actual enabler: run `svm-peval` as an svm-IR program — i.e. the Rust→svm-IR on-ramp.** To
specialize from within svm, the specializer must itself be svm-IR. Status of the pieces:

- **Rust→svm-IR exists and is tested.** `crates/svm-llvm` is a generic *LLVM-bitcode → svm-IR*
  translator ("one component buys every LLVM language", `LLVM.md` / D54), not C-specific. The Milestone-2
  test `crates/svm-llvm/tests/translate.rs` compiles a `#![no_std]` / `panic=abort` Rust crate with
  **`rustc +1.81.0`** and runs it `interp == JIT == native rustc` — including `-O2` auto-vectorized SIMD.
  - **Toolchain pin:** the bitcode version must match the pinned reader (LLVM 18). Default `rustc`
    (1.94 / LLVM 21) is rejected; **`rustc +1.81.0` (LLVM 18.1)** is accepted. CI installs `1.81.0` for
    this lane. (`rustup toolchain install 1.81.0` to run it locally; the test skips without it.)
  - **Guest constraints:** `#![no_std]` (no OS) + `panic=abort` (no EH/unwinding → "lowers like C").
- **Gap to running `svm-peval` (each bounded, none greenfield):**
  1. ~~**`core + alloc`**~~ **DONE.** A `no_std` Rust program with a `#[global_allocator]` over the guest
     `malloc`/`free` (the `vm_map`-growing bump allocator) now runs end-to-end through the on-ramp:
     `rust_core_alloc_heap_matches_native` in `crates/svm-llvm/tests/translate.rs` builds a growing
     `Vec<u64>` (many `RawVec` reallocs → `malloc`/`free` churn → heap growth) + a `Box`, sums on the
     heap, and prints — byte-identical to the same program built as a native `std` binary. The full
     `alloc` stack (`RawVec`, `__rust_alloc`/`__rust_dealloc`/`__rust_realloc`, `Box`) lowers with no
     translator change beyond the C heap path.
  2. ~~**`HashMap` → `BTreeMap`**~~ **DONE.** `svm-peval`'s memo maps and `svm-ir`'s linker symbol
     tables are `BTreeMap` now; `Known`/`Frame` gained `Ord`. (`HashMap` is `std`-only.)
  3. ~~**`no_std`-ify `svm-peval`** (compile half)~~ **DONE.** `svm-ir`, `svm-verify`, and `svm-peval`
     are `#![cfg_attr(not(test), no_std)]` + `alloc` (their own test harnesses still get `std`;
     dependents are unaffected, they bring their own `std`). Float folds route through `libm`
     (`sqrt`/`ceil`/`floor`/`trunc`/round-ties-even/`fma`/`abs`/`copysign` — all not in `core` on the
     pinned `rustc 1.81`; correctly-rounded/exact so bit-identical, proven by the differential fuzz).
     Also made 1.81-clean (`is_none_or` → explicit match). **Result: the three crates compile to
     `no_std`/`panic=abort` LLVM-18 bitcode on the pinned toolchain.** The translator gaps (next bullet)
     are now the wall, *not* compilation.
  4. **Wire residual → §22 `Jit.compile`** to run/share it in-sandbox (Milestone 3).

### Milestone 2 — translation status: **BLOCKED at the on-ramp (the actual remaining work).**

The specializer **compiles** to `no_std` LLVM-18 bitcode (above), but does **not yet translate**
through `svm-llvm`. Probing the merged bitcode (`cargo +1.81.0 build --release` with
`RUSTFLAGS=--emit=llvm-bc`, `llvm-link-18`, `opt-18 internalize,globaldce` down to the closure reachable
from `specialize`, then `svm-llvm-translate`) surfaces an ordered list of gaps — and the root causes are
**not the specializer's own code** (it uses no i128, no inline-asm, no exotic intrinsics):

1. **`Unsupported("inline-asm call")`** — `libm`'s `fma` does runtime CPU-feature detection
   (`core::core_arch::x86::_xgetbv`, `cpuid`) to pick a hardware vs software path. Inline asm has no
   svm-IR meaning; the on-ramp rejects it.
2. **`Unsupported("integer width i128 …")`** — `libm`'s correctly-rounded `fma` software fallback uses
   extended-precision integer math (`support::big::u256`, `narrowing_div`, `widen_mul`). svm-IR has no
   i128; the on-ramp would need to **legalize i128 → pairs of i64** (add/sub/mul/cmp/shift/ctlz/cttz).
3. **Structural, behind those: core/alloc runtime symbols absent from the bitcode.** `core::panicking::*`
   (`panic_fmt`, bounds-check, `unwrap_failed`), `cell::panic_already_borrowed`,
   `alloc::alloc::handle_alloc_error`, `raw_vec::handle_error`, `__rust_alloc`/`dealloc`/`realloc`,
   `bcmp`, `String::clone` — these are **non-generic** library functions that live in the *precompiled
   `core`/`alloc` rlibs*, so they are never emitted as bitcode. (Generic code — `Vec`, `BTreeMap`, slice
   ops — *is* monomorphized into the crate bitcode, which is why Milestone 1 worked.) Getting them into
   bitcode needs **`-Z build-std`** (nightly) — which fights the 1.81/LLVM-18 pin — *or* the on-ramp must
   **stub them** (panics → `trap`; `__rust_alloc`/`realloc`/`dealloc` → the synthesized `malloc`/`free`;
   `bcmp` → `memcmp`; `handle_alloc_error` → `trap`).

**Two root causes, two directions:**
- **`libm` (gaps 1–2).** It's pulled in *only* by the float folds. Cheapest unblock: in the in-svm
  (`no_std`) build, **don't fold the `libm`-requiring float ops** (`fma`/`sqrt`/… pass through unfolded
  — sound: they just run at runtime in the residual). That deletes the inline-asm + i128 surface
  entirely. The "right" fix is i128 legalization in the on-ramp (wanted anyway for general Rust).
- **core/alloc runtime symbols (gap 3) — the fundamental one.** Independent of `libm`; *any* non-trivial
  Rust hits panics/bounds-checks/allocator shims. The on-ramp needs a **runtime-symbol shim layer**
  (panics→trap, `__rust_*`→`malloc`/`free`, `bcmp`→`memcmp`), the Rust analogue of how it already
  synthesizes libc. This — plus i128 — *is* the "on-ramp coverage at scale" work the milestone
  predicted (the same class as QuickJS), now **enumerated concretely** instead of hypothesized.

So Milestone 2's compile half is **done**; the translate half reduces to a **scoped on-ramp feature
list** (i128 legalization + a Rust runtime-symbol shim), which is the next chunk of work.

**Why not `std`.** `std` is Rust's OS-abstraction layer (`core` + `alloc` + a `std::sys::<target>`
platform backend for files/threads/time/net/startup). svm has none of those as ambient services (it
has capabilities), and rustc has no `std::sys::svm` backend, so a `std` build drags in unresolved
syscall/libc externs (or inline-asm `syscall`), `panic=unwind` EH (`invoke`/`landingpad` + libunwind),
pthreads/TLS, the libc allocator, and `lang_start` OS startup — none of which the on-ramp can map.
Supporting `std` = **porting `std` to svm** (a `std::sys::svm` backend over svm caps, à la WASI/Redox)
— a separate, larger **OS-personality workstream**, gated on svm exposing the host caps `std` needs. It
is **not needed for the specializer** (a pure `Module → Module` transform: no I/O, threads, or time —
`core + alloc` suffices), only for running *general* `std`-using Rust as guests.

**Related but separate workstream:** translating large real interpreters (QuickJS) to svm-IR is blocked
on `svm-llvm` coverage (`setjmp`/`longjmp`, scale), tracked in `LLVM.md`, not here.

**Milestones (smallest first):**
1. ~~`core + alloc` through the Rust on-ramp~~ **DONE** — heap-allocating `no_std` program, on-ramp
   stdout byte-identical to native (`rust_core_alloc_heap_matches_native`).
2. `no_std`-ify `svm-peval` —
   - **compile half: DONE** — the three crates compile to `no_std`/`panic=abort` LLVM-18 bitcode on
     `rustc 1.81` (`BTreeMap`, `libm` float folds, `not(test)` no_std, 1.81-clean).
   - **translate half: BLOCKED** on a scoped on-ramp feature list (i128 legalization + a Rust
     runtime-symbol shim for panics/`__rust_alloc`/`bcmp`; `libm` inline-asm+i128 avoidable by not
     folding libm-requiring float ops in the in-svm build). See "translation status" above.
3. End-to-end in-sandbox demo: a guest specializes a toy interpreter against a script and runs the
   residual via the §22 `Jit` cap (alongside `crates/svm-run/demos/jit/`). *Gated on Milestone 2's
   translate half.*

## Benchmarking

**Regenerable report: [`PEVAL_BENCH.md`](PEVAL_BENCH.md)** — run
`python3 scripts/peval_bench_report.py` to rebuild it. The script runs the CSV-emitting benches in
`svm-peval` and `svm-llvm` (set `SVM_BENCH_CSV=1` to emit `CSV,<bench>,<case>,<metric>,<value>` rows)
and renders one consolidated markdown table; timings are JIT, compile-once/run-many, single-run and
machine-dependent (the report records the host).

Benches feeding it:
- `tests/bench.rs` (`svm-peval`): `size_corpus` (size across toy shapes, also a size-regression
  guard), `gain_spectrum` (the overhead-bound→work-bound run-time gradient on toy loops), and
  `roi_futamura_loop` (end-to-end Futamura on the sum-loop: ~3.6× interp / ~7× JIT specialization
  win, thousands× interpreted-interpreter→compiled-native).
- `tests/peval_corpus.rs` (`svm-llvm`): the real clang-compiled BF + Lisp interpreters across a range
  of guest programs — size, PE time, JIT-compile time, and run-time speedup.
- `tests/bench.rs::fuzz_specialization_*` (`svm-peval`): the differential oracle
  (interp == interp == jit) over random programs across four interpreter shapes; the bail surface
  (Budget / Unsupported / nonterminating) is reported and verified legitimate.

Extend the corpus with new shapes as slices land, so each one's size/speed effect is measured, not
assumed.

**Non-goals** (the engine correctly bails, not pending work): effectful / multi-result ops — atomics,
fibers/threads, host `cap.call` / imports — cannot be folded soundly.

Drop this file once the actionable slices (1–2) close.
