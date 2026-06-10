# Security & Correctness Audit — findings register

Audit date: **2026-06-10**. Scope: the escape-TCB (verifier, masking unit, memory
substrate, decoder) and the unsafe-heavy backends (JIT lowering + mask elision + cap
ABI, the §14 nesting runtime, the fiber/thread runtime + the §5 kill-path). Method:
four parallel deep-dive reviews + a direct review of the capability/authority model.

**Verdict: the escape-TCB is sound.** No memory-safety escape, no unsound optimization
(mask elision is provably upper-bound-sound), no arbitrary-code path. Findings cluster in
**availability (host-survivability)** and **on-paper-UB / robustness hardening**, not
confinement. This register tracks every finding to closure.

## Status table

| # | Sev | Area | Status |
|---|-----|------|--------|
| 1 | MED–HIGH | Capability model — guest can abort the host by exhausting the handle table | ✅ **FIXED** (`eadd460`) |
| 2 | MED (on-paper UB) | Thread runtime — racy non-atomic writes to the shared `trap_out` cell | ✅ **FIXED** (`eadd460`) |
| 3 | MED (defensive) | JIT nesting — validated child size vs clamped child size can diverge | ✅ **FIXED** (this commit) |
| 4 | LOW | cap.call ABI — result buffer partially uninitialized on a sig/host arity mismatch | ✅ **FIXED** (this commit) |
| 5 | LOW | memory substrate — `Mapped` atomic width dispatch treats any non-4 width as 8 | ✅ **FIXED** (this commit) |
| 6 | LOW | memory substrate — `Paged::read_into` can debug-overflow on out-of-range `off` | ✅ **FIXED** (this commit) |
| 7 | LOW | decoder — `Vec::with_capacity(ndata)` ~40× allocation amplification | ✅ **FIXED** (this commit) |
| 8 | LOW | thread runtime — futex `HashMap` entries never pruned | ✅ **FIXED** (this commit) |

**All audit findings are now closed.** Remaining hardening beyond this register (e.g. per-domain
quota metering, §15) is tracked in HANDOFF/DESIGN, not here.

---

## Fixed

### 1 — Guest can abort the host by exhausting the handle table  *(MED–HIGH)* ✅
**Was:** the handle table is fixed at `CAP = 256` slots and `Host::grant` panicked
(`.expect("handle table full")`). `AddressSpace.sub`, `create_region`, and the cross-domain
`SharedRegion.grant` mint handles, are guest-callable in a loop, and there is no
guest-reachable close op — so after ~250 mints `grant` panics, and on the JIT path that
unwinds across the `extern "C"` `cap_thunk` (no `catch_unwind`) → **process abort**. Not an
escape, but it broke "a guest can never crash the host."
**Fix:** `Host::grant` split into a fallible `try_grant` (returns `None` when full) and an
infallible `grant` retained only for host-controlled powerbox setup (bounded by construction).
The three guest-minting sites now surface `None` as **`-EMFILE`** (-24) like every other
fallible cap op. `try_grant_shared_region_backed` checks for a free slot before registering a
backing (no leak on a full table).
`crates/svm-interp/src/lib.rs` (`try_grant`/`grant`, `try_grant_shared_region_backed`, the
`sub`/`create_region`/cross-domain-`grant` sites). Pinned by
`address_space::minting_past_table_capacity_returns_emfile_not_panic` (differential — confirms
the JIT path returns -24, not an abort across the thunk).
**Follow-up:** the mint *count* should also be quota-metered (§15) — tracked under future
quota work, not this register.

### 2 — Racy non-atomic writes to the shared `trap_out` cell  *(MED, on-paper UB)* ✅
**Was:** the run's single trap cell is shared across all vCPU threads; multiple
concurrently-dying vCPUs (and the root) wrote it via a non-atomic `*mut i64` store
(`os_thread_rt.rs`), a formal data race (benign on real hardware — aligned 8-byte stores
don't tear — but UB in the Rust abstract machine).
**Fix:** `run_inner`'s shared trap cell is now an `AtomicI64`; every **Rust** access goes
through `Relaxed` atomics (`store_trap`/`load_trap` helpers in `os_thread_rt.rs`, and the
`run_inner` read-back/fault-store). The JIT continues to write the same cell via an aligned
`i64` store in emitted code — a hardware-atomic store that is foreign machine code from Rust's
model, so the Rust-side atomics make the cell race-free. Single-threaded child runs
(`compile_child_and_run`) keep a plain `i64` (no sharing). Verified against `jit_threads`,
`jit_killpath_threads`, and the loom model check.

---

## Resolved (#3–8) — fixes applied

Each "Fix:" below describes what was implemented.

### 3 — JIT child-size: validated vs clamped value diverge  *(MED, defensive)* ✅
`instantiator_rt.rs` validates `child_size = 1 << size_log2` **unclamped** against the parent
carve, but `lib.rs` builds/seeds/copies-back the child window with `1 << size_log2.min(MAX_JIT_WINDOW_LOG2)`
(2^26). Not exploitable today — any `size_log2 > 26` fails `child ≤ parent ≤ 2^26` first — but
the two paths should share **one** clamped value so the invariant is local, not cross-module.
**Fix:** compute the clamped child size once and use it in both validation and window setup.

### 4 — cap.call result buffer partially uninitialized on arity mismatch  *(LOW)* ✅
If a host op returns fewer results than the IR sig declares, the JIT reads back trailing result
slots that hold stack garbage (`lib.rs` result read-back / `svm-run` `zip`). Bounded to the
JIT-owned `n_res*8` stack slot (no OOB) and the verifier pins sig↔host arity, so it's a
differential-correctness corner, not a safety issue.
**Fix:** zero-fill the result buffer before the thunk call, or `debug_assert!` host arity ==
sig arity.

### 5 — `Mapped` atomic width dispatch treats any non-4 width as 8  *(LOW)* ✅
`crates/svm-mem/src/lib.rs` — `match width { 4 => u32, _ => u64 }`. Only callers pass 4/8
(`atomic_width`), so unreachable, but a future odd width would silently do an 8-byte access.
**Fix:** `debug_assert!(width == 4 || width == 8)` at the `Region::atomic_*` entry.

### 6 — `Paged::read_into`/`zero` can debug-overflow on out-of-range `off`  *(LOW)* ✅
`crates/svm-mem/src/lib.rs` — `off + k` / `off + len` are debug-`+`; a huge `off` would
overflow-panic before the `o >= size` break, instead of being inert. Unreachable today
(callers pre-confine), but the public `Region` API documents "out-of-range reads zero."
**Fix:** early `off >= size` guard or `saturating_add`, matching `byte()`'s inert contract.

### 7 — Decoder `Vec::with_capacity(ndata)` amplification  *(LOW)* ✅
`crates/svm-encode/src/lib.rs` — `ndata` is bounded by remaining bytes (so not a true OOM),
but `with_capacity` pre-reserves ~40 B/element (~40× the blob). It's the only untrusted-count
pre-allocation; every other collection grows incrementally.
**Fix:** reserve incrementally, or cap the pre-reservation.

### 8 — Futex `HashMap` entries never pruned  *(LOW)* ✅
`crates/svm-jit/src/os_thread_rt.rs` — `futex_wait` does `entry(key).or_default()` and the
waiter-count decrement never removes a zeroed entry. Bounded by the confined window size (keys
are confined guest addresses), so not unbounded.
**Fix:** remove an entry when its `waiters` hits 0.

---

## Checked and found sound (no action)

- **Verifier** — fail-closed; every type/result/arity/branch-target/data-segment-bound checked
  with `checked_add`; forged `call_indirect`/`ref.func` indices inert by construction; no panic
  on hostile-but-decodable input; `#![forbid(unsafe_code)]`.
- **Masking (`svm-mask`)** — `confine` keeps the result in-window for *all* inputs (`addr=u64::MAX`,
  `size=1`, 2^63 window); the access **width** is bounded by `checked_add`; `Window::sub`
  reserved-aligns the base so a child can't wrap past its sub-range / into a sibling / out of the
  parent.
- **`svm-mem` unsafe** — `Mapped` preconditions met by the caller contract (alignment enforced
  upstream, ranges clamped); `unsafe impl Send/Sync` justified for the shared-window model.
- **JIT mask elision (`ub_of`/`in_window`)** — upper-bound-sound: block-local values reset per
  block, all block-param / loop-carried / call / `sp` values arrive `UB_TOP` (always masked),
  every rule saturates on overflow, width accounted for — an elided address is bit-identical to
  the masked one.
- **cap.call ABI** — arg/result buffers sized from the compile-time-fixed verified sig (not guest
  runtime data); null/0 handled.
- **Trap propagation** — re-checked after every call/cap.call/thunk; trapping ops return from a
  dead side block; the §5 kill-path composes cleanly.
- **`call_indirect`** — Spectre-safe masked table dispatch + structural type-id check.
- **Fiber switches (`svm-fiber`)** — register- and alignment-complete on all three ABIs (incl.
  AArch64 `d8–d15` and the Win64 TEB stack fields); guard-paged stacks; body panic → abort (no
  unwinding across a switch).
- **Decoder** — every count bounded by remaining bytes; LEB128 overflow-guarded; no silent
  truncation (`try_from` guards); guaranteed termination.
- **§5 kill-path** — epoch-cell lifetime guaranteed by `join_all` before teardown; spinning **and**
  parked vCPUs (futex/join) re-check and unwind; no `join_all` hang.
- **Capability forgery resistance** — `resolve` checks generation **and** type_id with a
  Spectre-safe masked slot index; region/module lookups bounds-checked; `create_region`
  allocation capped (256 MiB).
