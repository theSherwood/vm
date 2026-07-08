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

## ⚠️ Security tradeoff surfaced (needs a decision)

The earlier mask model masked the address with a single **AND** — a *data dependency* that also
executes on the speculative path, i.e. it doubled as **Spectre-v1 hardening** (DESIGN §4 explicitly
claimed this). Trap-confinement's bounds-check branch does **not** have that property: the very
mechanism that buys the perf win (the load issuing speculatively past a predicted-not-taken branch)
*is* the Spectre-v1 exposure. This matches what native wasm engines that bounds-check accept, but it
is a real change to a security property the old design advertised. I documented it honestly in §4;
if Spectre-v1 hardening must be preserved, it needs a separate mechanism (an index mask on the
speculative path, or a fence) — flagged for your call.

## Progress log

- (start) Branch `claude/trap-confinement` off main. Mapped every confinement call site; wrote
  this doc. Starting bottom-up: `svm-mask` spec first.
- Part 1: `svm-mask` + `svm-interp` (both engines) → trap. Committed.
- Part 2: `svm-jit` bounds-check (corrected to bound against `reserved` for growth) + all
  escape-oracle / nesting / concurrent test suites rewritten. Full svm suite green. Committed.
- Part 3: `svm-wasm` comments, `fuzz/mask.rs`, and DESIGN.md updated for trap-confinement.
  Surfaced the Spectre-v1 tradeoff (above).
