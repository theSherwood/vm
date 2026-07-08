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

- [ ] **`svm-mask`** — the shared spec. `Window::checked` / `confine` become bounds checks (no
      `& mask`). Update the crate's property tests + the `mask` fuzz target postconditions.
- [ ] **`svm-interp` (tree-walk)** — `load_scalar`/`store_scalar`/`load_v128`/`store_v128` call
      `window.checked`; the `confine` site (`lib.rs:11202`). Trap on OOB.
- [ ] **`svm-interp::bytecode`** — same access path (shares the `Mem`/`Window` helpers).
- [ ] **`svm-jit`** — already done on `claude/jit-bounds-confine` (bounds check + cold trap,
      width threaded through `mask_addr`). Re-apply here.
- [ ] **`svm-wasm`** — currently masks *deliberately* to match wrap ("the documented confinement
      difference"). Trap-semantics lets it drop the mask and use wasm's native bounds-trap;
      window can be exact-sized. (Simplification.)
- [ ] **`svm-mem`** — the backed `Region` already only sees a confined offset; likely no change,
      but verify the guard/fault model still matches.
- [ ] **Escape-oracle tests** (`crates/svm/tests/escape_oracle.rs`) — rewrite the 3 wrap-
      confinement cases (`out_of_window_store_confines_identically`,
      `far_address_and_offset_fold_into_window`, `reserved_tail_access_faults_identically`) to
      assert trap-confinement.
- [ ] **Generator** (`crates/svm/tests/support/irgen.rs`) — the interp/JIT diff must agree on the
      new trap; verify OOB-generating inputs still converge (both trap).
- [ ] **Docs** — DESIGN §4/§18, invariant I1, the `svm-jit`/`svm-wasm`/`svm-interp` headers that
      describe wrap-confinement.

## Progress log

- (start) Branch `claude/trap-confinement` off main. Mapped every confinement call site; wrote
  this doc. Starting bottom-up: `svm-mask` spec first.
