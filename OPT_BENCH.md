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
| reg-sum residual (loop) | 7/233 → 5/142 | gvn −5, licm −10, local_cse +2 |
| licm+cse kernel | 7/71 → 6/77 | licm −9, local_cse +3 |
| sccp const-loop | 6/57 → 4/63 | sccp +1, gvn −6, licm −9, local_cse +4 |
| reassoc chain | 8/41 → 2/26 | **reassociate +15** |
| correlated branch | 8/73 → 6/55 | **jump_thread +18** |
| memory (mem + load_elim) | 5/85 → 3/77 | **mem +4, load_elim +4** |
| interproc (devirt+inline+dfe) | 10/82 → 7/41 | **devirt +41, inline +18, dfe +47** |

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
- **LICM and GVN carry *negative* size deltas** on loops: they hoist/thread invariants through new
  block params, which is larger static code — deliberately, to save run time (next section).

## Run-time ablation (N = 1,000,000)

Two loops: the register-machine sum residual, and a **heavy-invariant loop** whose body recomputes
`inv = (a*b + a)*(b + 7) + a*b` (invariant in runtime params `a`, `b`) every iteration. `interp/all` is
each variant's interpreter run time ÷ the full pipeline's; `>1` means removing that pass made the
interpreter slower — the pass was buying that run time. Rows for passes that don't apply to a loop
(the interproc/memory passes, and the simplifiers here) sit at ~1.00× and are elided.

### reg-machine sum 1..=N (already-tight loop)

| variant | bytes | jit_ms | interp_ms | interp/all |
|---|---|---|---|---|
| none | 127 | 0.620 | 65.76 | 1.10× |
| all | 142 | 0.622 | 59.68 | 1.00× |
| −gvn | 137 | 0.624 | 56.40 | 0.94× |
| −licm | 132 | 0.630 | 60.95 | 1.02× |
| _(sccp / reassoc / cse / jump_thread / devirt / inline / dfe / mem / load_elim)_ | 142 | ~0.62 | ~59 | ~1.00× |

### heavy-invariant loop (LICM showcase)

| variant | bytes | jit_ms | interp_ms | interp/all |
|---|---|---|---|---|
| none | 87 | 0.589 | 103.15 | 1.32× |
| all | 100 | 0.589 | 81.07 | 1.00× |
| −gvn | 97 | 0.592 | 76.64 | 0.98× |
| −licm | 85 | 0.589 | 90.69 | **1.16×** |
| _(all other passes)_ | 100 | ~0.59 | ~78–79 | ~1.00× |

## Takeaways

1. **On the JIT, svm-opt's passes are ~run-time-neutral** (`jit_ms` flat ~0.6 ms across every variant on
   both loops). Cranelift (`opt_level="speed"`) re-derives the same native code, so the IR-level win is
   washed out. svm-opt's JIT-path value is **code size / compile time** — where the interprocedural
   passes shine (halving the interproc module) — not native speed.
2. **On the interpreter, the optimizer buys real run time**: the full pipeline is **1.32×** on the
   heavy-invariant loop and **1.10×** on the already-tight sum loop.
3. **LICM is the dominant run-time pass** — **1.16×** on the heavy loop by itself (it moves the five-op
   invariant chain out of the per-iteration path). It's a pure run-time optimization: it *costs* static
   size (the negative deltas) and only pays off when there's real invariant work to hoist. On the tight
   sum loop, which has nothing worth hoisting, LICM/GVN slightly *hurt* both size and interp time — the
   signal for a **hoist cost model** (only hoist when the loop body warrants it), still the clearest
   actionable follow-up.
4. **The interprocedural + simplifier passes are size plays** (devirt/inline/dfe, reassociate,
   jump_threading, CSE) and are loop-run-time-neutral — but code size *is* the JIT-path win, so they
   earn their keep where the loop passes don't.

_The corpus is small and hand-built to isolate each pass. Remaining follow-ups (OPT.md): a LICM/GVN
hoist cost model, multi-run medians + variance, and Wasmtime-relative numbers._
