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

### Milestone 2 — translation status: **in progress, gaps falling one by one.**

The specializer **compiles** to `no_std` LLVM-18 bitcode (above). To find what it takes to *translate*,
we probe the merged bitcode: `cargo +1.81.0 build --release` (`--ignore-rust-version`,
`default-features=false`) with `RUSTFLAGS=--emit=llvm-bc`, then `llvm-link-18`, then `opt-18
internalize,globaldce` down to the closure reachable from a powerbox `main` that builds a tiny module
and calls `specialize`, then `svm-llvm-translate`. Each gap, in the order hit, and its disposition:

1. ✅ **inline-asm** — came from `libm`'s `fma` doing x86 CPU-feature detection. **Cleared:** the
   libm-requiring float folds (`sqrt`/`ceil`/`floor`/`trunc`/round-ties-even/`fma`) are now behind the
   default-on `libm-floats` feature; the in-svm build turns it off and leaves those ops unfolded
   (sound passthrough). `abs`/`copysign` moved to pure-`core` bit ops (still fold everywhere). x86
   inline-asm is fundamentally untranslatable, so avoiding libm is the right call, not a shortcut.
2. ✅ **i128** — came from the specializer's *own* exotic SIMD folds (saturating add/sub, narrow,
   ext-mul, ext-add-pairwise), which used `i128` for wide intermediates. **Cleared:** rewritten to
   `i64`/`u64` (wasm SIMD lane results cap at 64 bits; unsigned 32→64 ext-mul uses `u64`). Bit-identical
   — exotic-ops test + thorough fuzz green. The on-ramp i128 dependency is gone.
3. ✅ **translator panic** — `translate_switch` computed `max - min + 1` in `i64`, which **overflowed**
   on a switch whose cases straddle the i64 range (a niche-discriminant match). **Fixed** to compute the
   span in `i128` → clean `Unsupported` instead of a panic.
4. ✅ **sparse switch** — Rust's **niche-optimized enum layouts** (e.g.
   `drop_in_place::<Option<(u32, Option<Vec<(u64,u32)>>)>>`, the specializer's memo value type) match the
   discriminant with `i64::MIN`-ish sentinels: dense in *cases*, astronomically sparse in *span*, so a
   `br_table` can't represent it. **Implemented** `lower_sparse_switch` in `svm-llvm` — an equality
   **compare chain** of synthetic blocks (appended after the real blocks, so existing indices are
   unchanged) that thread the data-SP, the operand, and every case/default target's branch args
   (computed once in the switch block's context, where φ/live-in resolution is valid). Heavily tested:
   four new differential tests (`switch_sparse_*`) covering i64 cases at `i64::MIN/MAX`, threaded
   live-ins + a φ successor, a 5-block chain, and the i32 path — all `interp == JIT`; full svm-llvm
   suite (178 tests) green, dense `br_table` path untouched. The specializer probe now translates past
   the switch.
5. ✅ **non-power-of-two integer memory + extend** — Rust niche layouts produce odd-width integer
   accesses (e.g. a 7-byte `i56` discriminant field). **Implemented:** `load iN` (33–63) reads the
   enclosing `i64` and masks; `store iN` writes exactly `ceil(N/8)` bytes (byte-exact, never clobbering
   an adjacent field); `emit_ext` sign/zero-extends a 33–63-bit source in `i64` (it previously
   mis-applied the i32 extend ops). Tested round-trip (unsigned + signed ±) on interp+JIT.
6. ✅ **saturating arithmetic** — `llvm.{u,s}{add,sub}.sat` (i32/i64, Rust `saturating_*`) → wrapping
   op + `select` clamp.
7. ✅ **saturating float→int casts** — `llvm.fpto{si,ui}.sat` (Rust float `as` casts) → svm-IR
   `FToISat`. (6+7 covered by `rust_saturating_and_fp_sat_casts_match_native`: 22 values byte-identical
   to native, incl. overflow/underflow/NaN.)
8. ✅ **vector popcount** — `llvm.ctpop.v16i8` → `Inst::VPopcnt`.
9. ✅ **i128 from a 16-byte struct-eq coalesce** — `-O2` compares `Known`'s `[u8;16]` payload as a
   single `load i128` + `icmp eq/ne i128`. Held as a **pair of i64 halves** in the aggregate
   side-table (load → two i64 loads; eq/ne → compare the halves). Same-block only.
10. ✅ **`memcmp`/`bcmp`** — Rust slice/`[u8]` equality and `BTreeMap` key ordering call these; the
    on-ramp synthesized `memcpy`/`memset`/`memmove` but not `memcmp`. Added `__svm_memcmp` (a counted
    unsigned byte compare → `0`-if-equal-else-signed-difference), backing both `memcmp` and `bcmp`.
11. ✅ **unordered/ordered float compare** — `fcmp uno`/`ord` (NaN tests from Rust float code) have no
    single svm-ir op; expanded as `uno = (a!=a)|(b!=b)`, `ord = (a==a)&(b==b)` (`true`/`false` → const).

### Milestone 2 translate half — **DONE for the `specialize` closure.** ✅

With gaps 1–11 cleared, the probe **translates end-to-end with no `Unsupported`**, and the result
**verifies**: the statically-reachable closure of `specialize()` — **102 functions** spanning
`svm-peval` + `svm-ir` + `svm-verify` + the `core`/`alloc` monomorphizations — lowers to a valid svm-IR
module (`svm_run::resolve_capability_imports` → `svm_verify::verify_module` both pass). So every
legalization above produces *sound* IR, not merely non-erroring output. The specializer **is**
translatable to svm-IR.

*Scope/caveats.* (a) The closure is the **static** call graph from a powerbox `main` that calls
`specialize` on a trivial module; it covers `specialize`'s machinery comprehensively (globaldce keeps
all statically-reachable functions, input-independent), but a future change that pulls in a genuinely
new code path could surface a new construct. (b) The pipeline is still the manual probe
(`rustc +1.81` `--emit=llvm-bc` → `llvm-link-18` → `opt-18 internalize,globaldce` →
`svm-llvm-translate`); folding it into an in-repo build is its own task. (c) The earlier worry about
core/alloc *panic* runtime symbols was retired — they're shimmed to `trap` (`is_rust_abort_call`), and
the allocator shims resolved through the synthesized `malloc`/`free`. **Next: Milestone 3** — actually
*run* the translated residual in-sandbox via the §22 `Jit` capability.

### Milestone 3 — the specializer **runs in-sandbox and produces the right answer.** ✅

The run pipeline works end to end: **build → translate → verify → execute**. A powerbox `main` that
builds a trivial module (`() -> i32` returning `42`), calls `svm_peval::specialize(&m, 0, &[])`, and
returns `residual.funcs.len()` translates, verifies, **runs** on the reference interpreter, and returns
the **correct** `1` (a one-function residual). The in-sandbox specializer is real *and* correct.

**The corruption bug — found and fixed: the on-ramp sized an empty struct (a Rust ZST) as 1 byte, not
0.** `struct_layout` in `svm-llvm` clamped a struct's total size with `off.max(1)`. A `Vec`/`RawVec`
carries the zero-sized `alloc::alloc::Global` allocator marker (`%"alloc::alloc::Global" = type {}`), so
the clamp inflated every `RawVec` by a byte → a 24-byte `Vec` laid out as **32** with `len` shifted from
offset 16 to **24**. LLVM lays `type {}` out as **0** bytes — that is the layout every `getelementptr`
is computed against — so every field offset/element stride through a `Vec`-bearing struct desynced from
the GEPs. Concretely, an indexed `module.funcs[i].params.len()` read the **outer** `funcs.len()` (the
byte at the wrong offset), so `specialize_with_config`'s `args.len() != f.params.len()` compared `0` to
a garbage `1` → `ArityMismatch`; the same garbage flowing into a `Vec`/alloc capacity is what made a
later `malloc` over-allocate → NULL → `handle_alloc_error` → `MemoryFault` (the symptom after the
slice-panic shim let it advance further). One root cause, both symptoms. **Fix:** `struct_layout`
returns `off` (an empty struct is size 0). Guarded by `rust_zst_struct_field_layout_matches_native` in
`crates/svm-llvm/tests/translate.rs` — a `no_std` `Vec<Inner>` with `Inner { data: Vec<u64>, tag: u64 }`
indexed and summed, byte-identical to native (it fails without the fix). A flat `Vec<u64>` (the existing
heap/BTreeMap tests) never exercised a ZST-bearing *element*, which is why it slipped through.

**One more translate gap cleared en route:** `core::cell::panic_already_borrowed` /
`…_already_mutably_borrowed` — the `-> !` cold lang items the specializer's `RefCell<OutlineState>`
borrows pull in — are now shimmed to `trap` (`is_rust_abort_call`), like the slice/alloc panic family.

**The manual probe is now an in-repo test (the pipeline slice — DONE).**
`crates/svm-llvm/tests/peval_in_sandbox.rs` (`peval_specialize_runs_in_sandbox_and_matches_host`)
folds the whole dance into one auto-skipping test: it builds the in-repo fixture
`tests/fixtures/peval_probe` (a `no_std` powerbox crate depending on `svm-peval`
`default-features = false` + `svm-ir`) to LLVM-18 bitcode with `rustc +1.81`, `llvm-link-18`s the
dependency closure, `opt-18 internalize,globaldce`s down to the powerbox `main`, then
`translate_bc_path` → `resolve_capability_imports` → `verify_module` → `run_powerbox`. The fixture
builds a small foldable module (`() -> i32` = `21 * 2`), calls `specialize`, and prints the residual
summary (`funcs`/`blocks`/`insts`); the test asserts it equals the **same** specialization run
host-side (a differential: in-sandbox == host — both fold to a single `const 42`). This
regression-proofs the *whole* ≈100-function `specialize` closure end-to-end in-sandbox, not just the
unit ZST-layout test. It skips cleanly when `rustc +1.81.0` / `llvm-link-18` / `opt-18` are absent.

**The end-to-end §22 `Jit` capstone — DONE.** `crates/svm-llvm/tests/peval_jit.rs`
(`peval_guest_specializes_and_jits_in_sandbox`, fixture `tests/fixtures/peval_jit`) closes the loop:
a `no_std` powerbox guest, *entirely in-sandbox*, builds a module (`entry(a,b) → helper(a,b) =
a*3 + b*5 + 7`), **specializes** it with `svm-peval` (inlining the call + folding the constants into
one function), **encodes** the residual with `svm-encode`, submits the blob to the §22 `Jit`
capability (`__vm_jit_compile`), and **invokes** the Cranelift-compiled residual (`__vm_jit_invoke2`)
over an input grid against a plain-Rust oracle — `0` mismatches. This is the guest-side Futamura loop
the whole lane was built for: **specialize → encode → Jit.compile → invoke**, no host involvement
beyond the capability the powerbox already grants; verification (not the producer) is the trust hinge.
Two enablers landed with it:
- **`svm-encode` → `no_std`** (the encoder must run in-sandbox to serialize the residual).
- **On-ramp resolves LLVM function aliases** (gap #12). Identical-code folding (LLVM's pass *and*
  Rust's cross-crate dedup) collapses byte-identical function bodies — e.g. svm-ir's
  `VIntUnOp::index` and `VPMinMaxOp::index`, both 2-variant enum→byte — into one definition plus an
  `@x = alias … @y`. The on-ramp built `name2idx` only from `define`s, so a `call`/`ref.func` to an
  alias looked like an undefined external. Now each function alias is registered under its aliasee's
  index (fixpoint for alias→alias chains; data-global aliases skipped). `svm-llvm:alias_target_name`.
- *Memory-match note:* the `Jit.compile` gate requires the residual's `memory.size_log2` to equal the
  guest's own window (and **no** data segments). The window is layout-dependent, so the test reads it
  off the translated guest and passes it as `argv[1]`; the guest stamps the module's memory descriptor
  with it. The residual carries no data segments (the program is folded into the IR, not a segment).

**How to reproduce by hand (for debugging a new translate gap on this lane).**
- *Full probe* (now `peval_in_sandbox.rs`): a cargo crate depending on `svm-peval` (`default-features =
  false`) + `svm-ir`, whose `main` builds a module and calls `specialize`. Build to bitcode with
  `RUSTFLAGS=--emit=llvm-bc cargo +1.81.0 build --release --ignore-rust-version`, then
  `llvm-link-18 target/release/deps/*.bc`, then `opt-18 -passes=internalize,globaldce
  -internalize-public-api-list=main,malloc,free`, then `translate_bc_path` →
  `svm_run::resolve_capability_imports` → `verify_module` → `run_powerbox`.
- *Fast isolations* (single crate, no `llvm-link`): `compile_rust_to_bc` → `translate_bc_path` →
  `resolve_capability_imports` → `run_powerbox`, exactly as the `rust_*` tests do. To localize a layout
  bug, dump the translated module with `svm_text::print_module` and diff a `getelementptr`'s LLVM byte
  offsets (`llvm-dis-18`) against the emitted `add`/`mul` strides — they must agree.

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
   - **translate half: DONE** — eleven on-ramp gaps cleared (inline-asm, i128 in SIMD folds,
     switch-span overflow panic, sparse-switch compare chain, i56 memory + extend, saturating
     arithmetic, fp-sat casts, vector ctpop, i128 struct-eq, `memcmp`/`bcmp` helper, fcmp uno/ord),
     each tested in `svm-llvm`. The reachable `specialize` closure (102 funcs) now **translates and
     verifies**. See "Milestone 2 translate half — DONE" above.
3. **Milestone 3: DONE** — the specializer **runs in-sandbox and returns the correct answer** (translate
   → verify → execute). The corruption bug was the on-ramp sizing an empty struct (Rust ZST, e.g.
   `Vec`/`RawVec`'s `Global` marker) as 1 byte instead of 0, desyncing every `Vec`-bearing struct's
   field offsets from LLVM's GEPs; fixed in `struct_layout`, guarded by
   `rust_zst_struct_field_layout_matches_native`. See "Milestone 3" above.
   - **in-repo pipeline: DONE** — the manual probe is now `peval_in_sandbox.rs`
     (`peval_specialize_runs_in_sandbox_and_matches_host`): builds the `tests/fixtures/peval_probe`
     crate → bitcode → link → `globaldce` → translate → verify → run, asserting the in-sandbox residual
     equals the host specialization. Regression-proofs the whole closure; auto-skips without the
     `rustc 1.81`/`llvm-18` toolchain. See "Milestone 3" above.
   - **§22 `Jit` capstone: DONE** — `peval_jit.rs` (`peval_guest_specializes_and_jits_in_sandbox`):
     a guest specializes a module, encodes the residual, and `Jit.compile`s + invokes it **all
     in-sandbox**, checked against an oracle. Enablers: `svm-encode` → `no_std`, and the on-ramp now
     resolves LLVM function aliases (gap #12, identical-code folding). See "Milestone 3" above.
   **Milestone 3 is complete.** Possible follow-ups: a richer Futamura flavor (specialize a small
   interpreter against a program in a const-overlay so a dispatch loop folds away), and folding the
   `rustc+1.81 → llvm-link → opt → translate` build into a reusable in-repo harness shared by the two
   `peval_*` tests instead of duplicated per fixture.

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
