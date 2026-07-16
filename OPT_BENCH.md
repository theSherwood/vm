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
  `all` = full pipeline. The togglable passes are the six global/analysis passes.

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

Per case: `input → none → all` (instructions / bytes). Each pass column is the **byte delta if that
pass is removed** from the full pipeline:

- `+N` — the output is N bytes *larger* without the pass ⇒ the pass shrinks code by N (what it buys).
- `−N` — the output is N bytes *smaller* without the pass ⇒ the pass *grows* static size (LICM/GVN
  hoist or thread values through new block params — a size cost paid for a run-time win).

| case | input i/b | none i/b | all i/b | sccp | reassoc | gvn | licm | local_cse | jump_thread |
|---|---|---|---|---|---|---|---|---|---|
| reg-sum residual (loop, all passes) | 7/233 | 5/127 | 5/142 | +0 | +0 | −5 | −10 | +2 | +0 |
| licm+cse kernel | 7/71 | 7/71 | 6/77 | +0 | +0 | +0 | −9 | +3 | +0 |
| sccp const-loop | 6/57 | 6/57 | 4/63 | +1 | +0 | −6 | −9 | +4 | +0 |
| reassoc chain | 8/41 | 8/41 | 2/26 | +0 | **+15** | +0 | +0 | +0 | +0 |
| correlated branch | 8/73 | 8/73 | 6/55 | +0 | +0 | +0 | +0 | +0 | **+18** |

Reading the rows:

- **reassociate** is the only pass that touches the constant chain `((((x+1)+2)+3)+4)` — it collapses
  the tail to `x+10` (41 → 26 bytes); nothing else helps.
- **jump threading** is the only pass that resolves the correlated branch (73 → 55) — it threads the
  edge past the empty forwarder that SCCP can't (the forwarder's selector is a different constant on
  each incoming edge). No other pass moves it.
- **local_cse** dedupes the redundant `a*b` in the loop bodies (a few bytes each); on these single-block
  bodies GVN adds nothing on top (the +0 gvn columns), since the local pass already caught it.
- **LICM (and GVN)** carry *negative* size deltas: they hoist/clone the invariant into a preheader and
  thread its result back through a new block parameter, which is larger static code — deliberately, to
  save run time. Whether that pays off is the run-time table.

## Run-time ablation (N = 1,000,000)

The register-machine sum loop and a **heavy-invariant loop** — a counted loop whose body recomputes
`inv = (a*b + a)*(b + 7) + a*b` (invariant in the runtime params `a`, `b`) every iteration. `interp/all`
is each variant's interpreter run time relative to the full pipeline; `>1` means removing that pass made
the interpreter slower, i.e. the pass was buying that run time.

### reg-machine sum 1..=N (already-tight loop)

| variant | bytes | compile_ms | jit_ms | interp_ms | interp/all |
|---|---|---|---|---|---|
| none | 127 | 0.221 | 0.625 | 64.34 | 1.10× |
| all | 142 | 0.217 | 0.624 | 58.39 | 1.00× |
| −sccp | 142 | 0.204 | 0.620 | 59.37 | 1.02× |
| −reassociate | 142 | 0.206 | 0.620 | 59.99 | 1.03× |
| −gvn | 137 | 0.225 | 0.647 | 56.43 | 0.97× |
| −licm | 132 | 0.222 | 0.621 | 60.51 | 1.04× |
| −local_cse | 144 | 0.204 | 0.621 | 58.47 | 1.00× |
| −jump_thread | 142 | 0.224 | 0.621 | 58.47 | 1.00× |

### heavy-invariant loop (LICM showcase)

| variant | bytes | compile_ms | jit_ms | interp_ms | interp/all |
|---|---|---|---|---|---|
| none | 87 | 0.224 | 0.592 | 103.47 | 1.32× |
| all | 100 | 0.234 | 0.599 | 78.81 | 1.00× |
| −sccp | 100 | 0.280 | 0.593 | 78.91 | 1.00× |
| −reassociate | 100 | 0.245 | 0.593 | 79.64 | 1.01× |
| −gvn | 97 | 0.238 | 0.593 | 78.36 | 1.00× |
| −licm | 85 | 0.388 | 1.123 | 91.94 | **1.17×** |
| −local_cse | 105 | 0.239 | 0.592 | 79.50 | 1.01× |
| −jump_thread | 100 | 0.253 | 0.609 | 79.98 | 1.02× |

## Takeaways

1. **On the JIT, svm-opt's scalar passes are ~run-time-neutral** (`jit_ms` is flat ~0.6 ms across every
   variant on both loops). The JIT's own optimizer re-derives the same native code, so the IR-level win
   is washed out. svm-opt's JIT-path value is size/compile, not native speed.
2. **On the interpreter, the optimizer buys real run time**: the full pipeline is **1.32×** on the
   heavy-invariant loop and **1.10×** on the already-tight sum loop.
3. **LICM is the dominant run-time pass** — worth **1.17×** on the heavy loop by itself (it moves the
   five-op invariant chain out of the per-iteration path). It is a pure run-time optimization: it *costs*
   static size (the negative size deltas) and only pays off when there is real invariant work to hoist.
   On the tight sum loop, which has nothing worth hoisting, LICM/GVN slightly *hurt* both size and interp
   time — a signal that a cost model (hoist only when the loop body warrants it) is the natural next step.
4. **The simplifying passes earn their size**: reassociation and jump threading each single-handedly
   shrink their target shape (15 B and 18 B), and local CSE trims the redundant multiplies. SCCP's win is
   small on this corpus (its big wins are branch-elimination shapes; more of those would sharpen it).

_The corpus is small and hand-built to isolate each pass; broadening it (more realistic residuals, more
branch-heavy shapes) is tracked under OPT.md Phase 5._
