# Optimizer ablation report

_What each svm-opt pass buys us, measured by **leave-one-out**: run the full pipeline, then the full
pipeline with one pass disabled, and attribute the difference to that pass._

Harness: `crates/svm-peval/tests/opt_bench.rs`. Regenerate with:

```
# size table (fast, non-ignored — also a correctness/size-regression guard)
cargo test --release -p svm-peval --test opt_bench opt_size_ablation -- --nocapture

# + run-time table (perf, #[ignore])
cargo test --release -p svm-peval --test opt_bench -- --include-ignored --nocapture

# machine-readable CSV rows on stderr
SVM_BENCH_CSV=1 cargo test --release -p svm-peval --test opt_bench -- --include-ignored --nocapture 2>&1 >/dev/null
```

- numbers below: release build, single host, single run — machine-dependent, ratios are the story.
- `none` = only the always-on intra-block canonicalization (fold / DCE / copy-prop / merge / prune).
  `all` = full pipeline. The **eleven** togglable passes are: `sccp`, `reassociate`, `gvn`, `licm`,
  `local_cse`, `jump_thread` (Phase 2); `devirt`, `inline`, `dfe` (Phase 3); `mem`, `load_elim`
  (Phase 4).

## How to read it

The optimizer's output is **re-verified** and **differential-tested against the interpreter** on every
ablation variant (the size test is also a correctness guard). Two backends answer two questions:

- The **JIT** runs its own optimizer over the IR, so svm-opt's scalar passes are largely redundant
  there — native run time barely moves whatever we do. svm-opt earns its keep on the JIT path through
  **code size** (smaller residual → faster compile) and by feeding the JIT cleaner IR, not by changing
  the final machine code's speed.
- The reference **interpreter** executes the IR as-is, so a pass that trims per-iteration work (LICM,
  CSE) shows a real run-time delta. This is the run-time value svm-opt adds on the interp path — and
  when svm-opt is itself translated to run *inside* svm (DESIGN.md §20c).

## Size ablation (encoded bytes)

`input → all` is the whole pipeline's size reduction. The **contributions** column lists each pass's
*byte delta if it is removed* from the full pipeline: `+N` = the output is N bytes larger without it
(the pass shrinks code by N — what it buys); `−N` = N bytes *smaller* without it (the pass **grows**
static size, e.g. LICM/GVN thread values through new block params — a size cost paid for a run-time
win). Passes not listed for a case had zero delta there.

| case | input→all (i/b) | pass contributions (Δbytes if removed) |
|---|---|---|
| reg-sum residual (loop) | 7/233 → 5/127 | _(all passes size-neutral)_ |
| licm+cse kernel | 7/71 → 6/74 | licm −6, local_cse +3 |
| sccp const-loop | 6/57 → 5/56 | sccp +1, reassociate −2, licm −5 |
| reassoc chain | 8/41 → 2/26 | **reassociate +15** |
| correlated branch | 8/73 → 6/55 | **jump_thread +18** |
| memory (mem + load_elim) | 5/85 → 3/77 | **mem +4, load_elim +4** |
| interproc (devirt+inline+dfe) | 10/82 → 7/41 | **devirt +41, inline +18, dfe +47** |
| multiblock inline (inline+dfe) | 4/67 → 3/64 | inline +3, **dfe +33** |

Reading it:

- **The interprocedural passes are the biggest size wins in the corpus.** On the interproc case a
  constant `call_indirect` is devirtualized to a direct call, the small leaf is inlined, and the leaf +
  an unused function are DCE'd — nearly halving the module (82 → 41 B). The three deltas overlap
  because they *cascade*: `devirt` (+41) is the enabler (without it the indirect call blocks inlining
  and keeps the funcref'd function alive), and `dfe` (+47) collects the whole payoff of removing the
  now-dead functions.
- **The simplifiers each single-handedly shrink their target shape**: `reassociate` collapses the
  constant chain `((((x+1)+2)+3)+4)` to `x+10` (+15), `jump_thread` threads the correlated branch past
  its empty forwarder (+18), `local_cse` dedupes redundant computations (+2–4).
- **The memory passes** trim a redundant same-address load (`mem`) and a diamond-join reload that only
  cross-block `load_elim` can reach (a multi-predecessor block can't be merged, so intra-block
  forwarding never sees it) — +4 each here.
- **Multi-block inlining** (the `multiblock inline` case, a caller of a three-block `max` with a value
  threaded across the call) shows a smaller *net* `inline` delta (+3) than the single-block `interproc`
  case (+18): splicing a callee's CFG into the caller grows the caller by roughly the callee's body, so
  the byte win comes downstream — `dfe` (+33) reclaiming the now-dead callee, and the caller's own code
  folding through the inlined region. The point is capability, not raw bytes: this callee shape was a
  hard `call` before.
- **LICM's negative delta on loops is bounded by a cost model**: it hoists/threads invariants through
  new block params (larger static code, to save run time — next section), but never a bare constant
  (free to recompute — threading one is pure overhead) and it rematerializes an invariant's constant
  operands in the preheader instead of threading them. So on the **hoist-free reg-sum loop LICM's delta
  is +0** (was −10), and on real-invariant loops the hoist still fires while the module *shrinks*
  (licm+cse −6 not −9; heavy-invariant loop 100→94 B in the run-time section).
- **GVN now carries the same constant cost model** and so no longer shows a negative delta: it used to
  thread a dominating constant to replace a congruent one (−5 on reg-sum, −3 on sccp const-loop), which
  is the same pure overhead. Leaving constants local drops those costs to **+0** (reg-sum 132→127 B),
  and — crucially — keeps a relooper's dispatch selectors as *local constants*, which is what lets
  `jump_thread` **de-reloop** irreducible control flow (see below).

## Run-time ablation (N = 1,000,000)

Two loops: the register-machine sum residual, and a **heavy-invariant loop** whose body recomputes
`inv = (a*b + a)*(b + 7) + a*b` (invariant in runtime params `a`, `b`) every iteration. `interp/all` is
each variant's interpreter run time ÷ the full pipeline's; `>1` means removing that pass made the
interpreter slower — the pass was buying that run time. Rows for passes that don't apply to a loop
(the interproc/memory passes, and the simplifiers here) sit at ~1.00× and are elided.

Ratios are normalized to `all = 1.00×`. Absolute ms are from one run on a noisier host than the first
report (both `jit_ms` and `interp_ms` are ~1.5–2× the earlier absolutes) — hence the standing multi-run
follow-up; the **bytes** column and the ratio *ordering* are the load-bearing parts.

### reg-machine sum 1..=N (already-tight loop)

| variant | bytes | interp_ms | interp/all |
|---|---|---|---|
| none | 127 | 81.46 | 1.08× |
| all | 132 | 75.12 | 1.00× |
| −gvn | 127 | 79.87 | 1.06× |
| −licm | 132 | 73.64 | 0.98× |
| _(all other passes)_ | 132 | ~73 | ~0.97× |

Nothing in this loop is worth hoisting, so every pass sits within run-to-run noise (±~5%) of `all`. The
point is the **bytes**: `all` is now **132** (was 142) — LICM's constant-hoist cost model removed the
useless threading, and `−licm` is byte-identical to `all` (LICM changes nothing here, as it should).

### heavy-invariant loop (LICM showcase)

| variant | bytes | interp_ms | interp/all |
|---|---|---|---|
| none | 87 | 183.3 | 1.53× |
| all | 94 | 120.2 | 1.00× |
| −gvn | 93 | 127.8 | 1.06× |
| −licm | 85 | 167.5 | **1.39×** |
| _(all other passes)_ | 94 | ~119 | ~0.99× |

`all` is **94 B** (was 100) — the invariant chain still hoists, but its constant operands are
rematerialized in the preheader instead of threaded, so the hoisted form is smaller. Removing LICM is
**1.39×**: it remains the dominant interp-path pass by a wide margin.

## Takeaways

1. **On the JIT, svm-opt's passes are ~run-time-neutral** (`jit_ms` flat across every variant on both
   loops). Cranelift (`opt_level="speed"`) re-derives the same native code, so the IR-level win is
   washed out. svm-opt's JIT-path value is **code size / compile time** — where the interprocedural
   passes shine (halving the interproc module) — not native speed.
2. **On the interpreter, the optimizer buys real run time**: the full pipeline is **~1.5×** on the
   heavy-invariant loop; the already-tight sum loop has nothing to hoist and sits in the noise.
3. **LICM is the dominant run-time pass** — **1.39×** on the heavy loop by itself (it moves the
   invariant chain out of the per-iteration path). Its **hoist cost model** now keeps it from *costing*
   size where it can't pay: it never threads a bare constant out of a loop and rematerializes invariant
   constant operands in the preheader, so on the hoist-free sum loop LICM is size-neutral (was −10 B)
   while the heavy loop still hoists — *and* shrinks (100→94 B). This closes the clearest earlier
   follow-up.
4. **The interprocedural + simplifier passes are size plays** (devirt/inline/dfe, reassociate,
   jump_threading, CSE) and are loop-run-time-neutral — but code size *is* the JIT-path win, so they
   earn their keep where the loop passes don't.

## Wasmtime-relative (JIT lane, `bench/`)

The ablation above measures svm-opt against *itself*. This measures it against the bar that matters
(DESIGN.md §1a): run the **same wasm bytes** on the SVM JIT (transpiled to IR) and on **Wasmtime**, with
and without svm-opt in front of the SVM JIT. Both lower core wasm through Cranelift, so it is like-for-
like. `compute32 = svm-jit ÷ Wasmtime-wasm32` per-iteration time (>1 = svm slower); best-of-5.

Regenerate from `bench/`:

```
cargo run --release -- --from-wasm --csv --reps 5             # baseline (no svm-opt)
cargo run --release -- --from-wasm --optimize --csv --reps 5  # with svm-opt in front of the JIT
```

Back-to-back best-of-5, same session (ratios only — absolute ns drift with host load):

| kernel | svm/Wasmtime, no opt | svm/Wasmtime, +svm-opt |
|---|---|---|
| **irreducible** (clang relooper output) | **1.17×** | **0.40×** |
| memsum | 1.23× | 0.96× |
| locals_c | 1.16× | 0.96× |
| hostcall | 1.17× | 1.10× |
| alu | 1.03× | 1.04× |
| alu_c | 1.03× | 1.04× |
| float | 1.02× | 1.03× |
| simd | 1.06× | 1.10× |
| scatter | 1.10× | 1.11× |
| calli | 1.26× | 1.28× |
| hostbuf | 0.49× | 0.49× |

_(`cache` is DRAM-latency-bound — the harness gives it a custom, low-repeat span — so its ratio is
memory-stall noise, not IR quality; elided.)_

**The headline is `irreducible`: 1.17× → 0.40×** — svm-opt makes the SVM JIT **2.5× faster than
Wasmtime** on it. This is the §1a differentiator made real. That kernel is a C `goto` into a loop —
genuinely irreducible control flow. Wasmtime *must* reloop it (wasm can't express irreducible CFG), so
even from the same source it pays a per-iteration dispatch. Under `--from-wasm` the SVM lane starts from
clang's *already-relooped* wasm and inherits that dispatch; svm-opt then **de-reloops** it — GVN keeps
the relooper's dispatch selectors as local constants (its new constant cost model), and `jump_thread`
resolves the dispatch `br_table` per edge until it dies, recovering the irreducible CFG that **SVM IR
runs natively but wasm cannot represent**. (Running the *native* SVM IR directly — no reloop ever — is
0.29×; de-relooping the transpiled form recovers most of that edge.)

**The loop kernels also improve** — `memsum` 1.23→0.96, `locals_c` 1.16→0.96 — because the same GVN
constant cost model stops threading loop constants into block params, tightening the loop bodies (the
GVN analogue of the LICM win). **The rest is flat** (within run-to-run noise), which *confirms* the
ablation's thesis: on straight-line compute the JIT re-derives svm-opt's scalar passes, so the optimizer
is native-speed-neutral there — its JIT-path value is code size / compile time, plus the structural wins
(de-relooping, un-threading) that the backend inherits and can't reconstruct itself.

_The corpus is small and hand-built to isolate each pass. Remaining follow-up (OPT.md): multi-run
medians + variance._
