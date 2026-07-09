# Trap-confinement: wrap → trap memory-model change

**Goal.** Change SVM's guest-memory confinement from **wrap-confinement** (an out-of-window
access is masked back into the window: `base + ((addr+offset) & (reserved-1))`) to
**trap-confinement** (an out-of-bounds access raises `Trap::MemoryFault` at the offending
access). This must be consistent across **every** engine so the §18 escape oracle
(interp↔JIT↔wasm final-memory + trap agreement) still holds.

**Why.** (1) UX — a clear fault at the exact OOB access instead of silent aliasing to another
in-window byte. (2) Perf — the mask's `AND` sits on the load's address-latency path; a bounds
check is off it (load issues speculatively past a predicted-not-taken branch). Measured on the
JIT: edn −9%, picojpeg −8% (matmult +5%). See PR #175 (the JIT-only spike that proved the win and
hit this exact semantics wall).

## The semantics, precisely

For an access `[addr+offset, addr+offset+width)` in a window with backed extent `mapped`
(and absolute `base`):

- **Old (wrap):** `rel = (addr+offset) & (reserved-1)`; if `rel+width ≤ mapped` → access at
  `base+rel`, else fault. (An OOB address whose masked value lands in `[0,mapped)` aliases to
  that byte — the behavior the escape oracle pins down.)
- **New (trap):** if `addr+offset+width ≤ mapped` → access at `base+addr+offset` (no masking),
  else `Trap::MemoryFault`. No aliasing; every OOB access traps.

The window keeps its `reserved` (power-of-two) reservation + guard tail as defense-in-depth, but
the bounds check traps before the guard is ever reached. `base + (addr+offset)` cannot overflow
because a passing check implies `addr+offset < mapped ≤ reserved` and `base+reserved ≤ 2^64`.

## Components (checklist)

- [x] **`svm-mask`** — the shared spec. `Window::checked` / `confine` are bounds checks (no
      `& mask`); property tests + all doc comments rewritten; `mask` fuzz target postconditions
      updated for trap semantics.
- [x] **`svm-interp` (tree-walk)** — `confine_checked` is a bounds check with checked arithmetic
      (rejects u64-overflow, keeps the `reserved` bound so grown tail pages still reach
      `check_prot`); `window.checked` fast-path sites flip automatically; sub-window test now
      asserts a fault (not a wrap). Doc comments refreshed.
- [x] **`svm-interp::bytecode`** — same `Mem` access path; covered by the above.
- [x] **`svm-jit`** — cherry-picked the bounds-check + cold-trap lowering, **corrected to bound
      against `reserved`** (not `mapped`) so it mirrors interp `confine_checked`+`check_prot` and
      preserves `memory.grow`; extended to §14 sub-windows (bounds-check then shift by `sub_base`).
- [x] **`svm-wasm`** — no code change (a transpiler; emits Load/Store IR that the engines now
      trap-confine). Its OOB accesses now trap like real wasm. Stale "SVM masks rather than
      bounds-checks" comments updated; residual per-region confinement difference documented.
- [x] **`svm-mem`** — no change needed (backed `Region` only ever sees a confined offset; the
      guard/fault model is unchanged — verified via the C-frontend growth tests).
- [x] **Escape-oracle tests** — rewrote `escape_oracle.rs` (top-level, far-address, reserved-tail,
      sub-window), `concurrent_escape.rs`, `concurrent_escape_fuzz.rs`, and the nesting tests
      (`instantiator.rs`, `bytecode_instantiate.rs`, `jit_instantiator.rs`) for trap semantics;
      kept in-window agreement coverage. Full svm suite green (768).
- [x] **Generator** (`irgen.rs`) — no logic change (already draws arbitrary OOB addresses; both
      backends now trap-agree, verified by `jit_fuzz`/`jit_diff`/`bytecode_diff`). Comments updated.
- [x] **Docs** — DESIGN §1a/§2a/§4/§18, invariant I1/I5, escape-vector table, C-ABI, the D38/D39
      decision notes; `fuzz/mask.rs`; `svm-mask`/`svm-interp`/`svm-wasm`/`svm-jit` headers.

## ⚠️ Security tradeoff surfaced — RESOLVED: check + clamp

The earlier mask model masked the address with a single **AND** — a *data dependency* that also
executes on the speculative path, i.e. it doubled as **Spectre-v1 hardening** (DESIGN §4 explicitly
claimed this). Trap-confinement's bounds-check branch does **not** have that property: the very
mechanism that buys the perf win (the load issuing speculatively past a predicted-not-taken branch)
*is* the Spectre-v1 exposure.

**Decision: keep the bounds check for the architectural trap and re-emit the old AND as a
speculative clamp on the address feeding the access** (`& (reserved−1)`; architecturally a no-op
for any access that passes the check, since `addr+offset < reserved` — the power-of-two
reservation is what makes the clamp this cheap). A misspeculated OOB access is confined to
`[0, reserved)` exactly as under wrap-confinement; the trap semantics and the §18 escape oracle
are untouched (the interpreter needs no change — it doesn't speculate).

**Why not Wasmtime's cmov guard.** Four lowerings were measured on identical LLVM-frontend IR
(confinement-dense kernels, every access through an unprovable base; `SVM_CONFINE` spike,
ns/iter, min over 5 interleaved rounds, spread ≤2%):

| kernel | mask (old wrap) | check (trap only) | check+AND (adopted) | check+cmov (Wasmtime-style) |
|---|---|---|---|---|
| chase2 (serial load chain) | 2.12 | 1.83 | 2.29 | 2.86 |
| dot (2 loads/iter, ILP) | 0.98 | 1.36 | 1.81 | 2.35 |
| matmul (i64, dense inner) | 9.4µs | +61% | +94% | +120% |
| bytes / stream / fnv (slack) | — | ±0% | ±0% | ±0% |

The cmov (`select_spectre_guard`, what Wasmtime's `heap_access_spectre_mitigation` emits for
bounds-checked heaps) puts `cmp→cmov` on the address-latency path — one op *more* than the AND —
and loses to check+AND on every memory-dense kernel by 20–50%. The AND clamp is available to SVM
(and not to Wasmtime) because the reservation is a power of two.

**The honest cost:** the clamp puts the AND back on the address-latency path, so the check-only
lowering's memory-bound win (edn −9%, picojpeg −8% on the JIT spike, PR #175) is spent buying
back the old speculative confinement — net ≈ old-mask performance plus the check's small
throughput cost on load-dense loops (these dense kernels amplify it ~4–8× vs the embench-scale
±5–9%). Trap-confinement's yield is therefore **semantics** (clean `MemoryFault` at the offending
access, wasm-parity UX), not raw speed. No unhardened check-only tier is offered: one fuzzed
configuration (D38); revisit only if a real deployment justifies trading Spectre-v1 confinement
for single-digit %.

**Placed against Wasmtime** (`bench/`, both engines at default hardening — Wasmtime's
`heap_access_spectre_mitigation` cmov is on by default for its bounds-checked heaps; `--csv
--reps 7`, best-of, svm÷wasm ratios; `mask` = old model, `both` = adopted check+clamp):

| kernel | svm/w32 mask → both | svm/w64 mask → both |
|---|---|---|
| memsum | 0.91 → 0.92 | 0.51 → 0.52 |
| scatter | 0.93 → 0.94 | 0.48 → 0.48 |
| cache | 0.96 → 1.28 (svm ns +6.7%) | 0.64 → 0.75 |
| locals_c (threaded data-SP, the check+clamp case) | 2.20 → **1.81** | 1.24 → **1.02** |
| alu / alu_c / float / simd / calli | ~1.0 ± 3% (unchanged) | ~1.0 ± 3% (unchanged) |

Adoption moved the suite vs the old mask model by ±0–3% everywhere except `cache` (+6.7%, the
check's branch on a miss-bound loop) and `locals_c` (**−18%, faster** — the C-locals
store→load path improves under check+clamp), which lands at **parity with hardened wasm64**
(1.02×), the honest same-width Cranelift-vs-Cranelift comparison. The remaining >1× vs wasm32
(`locals_c` 1.81×) is the known 32-bit-index structural gap (§1a/D50), not the clamp.

## Progress log

- (start) Branch `claude/trap-confinement` off main. Mapped every confinement call site; wrote
  this doc. Starting bottom-up: `svm-mask` spec first.
- Part 1: `svm-mask` + `svm-interp` (both engines) → trap. Committed.
- Part 2: `svm-jit` bounds-check (corrected to bound against `reserved` for growth) + all
  escape-oracle / nesting / concurrent test suites rewritten. Full svm suite green. Committed.
- Part 3: `svm-wasm` comments, `fuzz/mask.rs`, and DESIGN.md updated for trap-confinement.
  Surfaced the Spectre-v1 tradeoff (above).
