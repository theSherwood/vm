//! **Concurrent escape-oracle** (`DESIGN.md` §4/§18, extended to §12 threads).
//!
//! The single-threaded escape-oracle (`escape_oracle.rs`, `jit_fuzz`) proves *"verified ⇒ cannot
//! escape"* by byte-comparing the final guest window across the interpreter (the masking reference)
//! and the JIT. The §12 concurrency work grew the escape-TCB — a shared `svm-mem` `Region`, raw
//! hardware atomics, the per-thread JIT runtime + its `mem_base` threading — without extending that
//! oracle. These cases close the gap: a **spawned thread** accessing an **out-of-window** address
//! must be confined into `[0, reserved)` *identically on both backends*, i.e. confinement still holds
//! when the access happens off the root vCPU.
//!
//! Determinism (so the window byte-compare is sound despite nondeterministic scheduling): every
//! shared access is a **commutative** `atomic.rmw.add` (interleaving-invariant final value) or a
//! **disjoint** per-thread plain write; thread handles are kept in SSA values, never the window, so
//! the compared bytes are a pure function of the program, not the schedule or the backend's handle
//! numbering. The non-vacuous element is the **out-of-window address**: a thread-context masking bug
//! diverges the windows (lands at the wrong in-window slot) or escapes (guard fault → JIT traps).
//!
//! Gated to the targets where the JIT runs threads (`svm_fiber::supported()`); the interpreter runs
//! threads everywhere, so off those targets there is no second backend to compare against.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use svm_interp::run_capture_reserved;
use svm_jit::{compile_and_run_capture_reserved, JitOutcome};

/// Run a threaded module on both backends with `init` seeding a fully-mapped `1 << size_log2`-byte
/// window, join all vCPUs, and return both final-window snapshots — asserting both ran to completion
/// (no escape surfaced as a trap) and agree on the entry result. `reserved_log2 = 0` ⇒ fully mapped
/// (`reserved == mapped`), so an out-of-window address **wraps** back in (the mask), the behaviour
/// these cases pin.
fn both_windows_threaded(src: &str, init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 50_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[], &mut fuel, init, 0);
    let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[], init, 0).expect("jit");
    assert!(ir.is_ok(), "interp trapped (concurrent escape?): {ir:?}");
    assert!(
        matches!(jo, JitOutcome::Returned(_)),
        "jit did not return (concurrent escape?): {jo:?}"
    );
    (imem, jmem)
}

/// Four worker threads each `atomic.rmw.add 1` a shared counter **100×**, but the counter lives at an
/// **out-of-window** address (`65544`, just past the 64 KiB window). The §4 final-address mask must
/// confine every one of those 400 atomic accesses to `65544 & 65535 = 8` — *from the spawned threads*
/// — identically on both backends. The total (400) is interleaving-invariant (atomic add commutes),
/// so the windows must be byte-identical: all zero except the i64 counter at offset 8.
#[test]
fn concurrent_atomic_to_out_of_window_addr_confines() {
    // counter address = 65536 + 8 = 65544 → masks to offset 8 (i64, occupies [8,16)).
    let src = "\
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.const 100
  v2 = thread.spawn 1 v0 v1
  v3 = thread.spawn 1 v0 v1
  v4 = thread.spawn 1 v0 v1
  v5 = thread.spawn 1 v0 v1
  v6 = thread.join v2
  v7 = thread.join v3
  v8 = thread.join v4
  v9 = thread.join v5
  v10 = i64.const 65544
  v11 = i64.atomic.load v10
  return v11
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 65544
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
";
    let init = vec![0u8; 65536];
    let (imem, jmem) = both_windows_threaded(src, &init);
    assert_eq!(
        imem, jmem,
        "concurrent escape-oracle: interp/JIT windows diverge (thread-context masking?)"
    );
    // The counter (i64, little-endian 400) confined to offset 8 — and *only* offset 8.
    let counter = u64::from_le_bytes(imem[8..16].try_into().unwrap());
    assert_eq!(counter, 400, "out-of-window atomic counter wrong/escaped");
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        2, // 400 = 0x0190 → two non-zero bytes at offsets 8 and 9
        "a concurrent access landed outside the confined counter slot"
    );
}

/// Four threads each write a fixed marker (`0xAA`) with a plain (non-atomic) store to its **own**
/// out-of-window address handed in as `arg`; the four addresses mask to four **disjoint** in-window
/// slots, so there is no race and the final window is deterministic. This exercises confinement of
/// *plain* stores issued from spawned threads (the atomic case above covers the atomic path) — each
/// store must land at `arg & 65535` and nowhere else.
#[test]
fn concurrent_disjoint_plain_stores_confine() {
    // Targets 65536+0/8/16/24 → mask to disjoint offsets 0/8/16/24; each gets the 0xAA marker.
    let src = "\
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.const 65536
  v2 = thread.spawn 1 v0 v1
  v3 = i64.const 65544
  v4 = thread.spawn 1 v0 v3
  v5 = i64.const 65552
  v6 = thread.spawn 1 v0 v5
  v7 = i64.const 65560
  v8 = thread.spawn 1 v0 v7
  v9 = thread.join v2
  v10 = thread.join v4
  v11 = thread.join v6
  v12 = thread.join v8
  v13 = i64.const 0
  return v13
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i32.const 170
  i32.store8 v0 v1
  v2 = i64.const 0
  return v2
}
";
    let init = vec![0u8; 65536];
    let (imem, jmem) = both_windows_threaded(src, &init);
    assert_eq!(
        imem, jmem,
        "concurrent escape-oracle: interp/JIT windows diverge on disjoint plain stores"
    );
    for slot in [0usize, 8, 16, 24] {
        assert_eq!(
            imem[slot], 0xAA,
            "a thread's out-of-window plain store did not confine to slot {slot}"
        );
    }
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        4,
        "a plain store from a thread escaped its confined slot"
    );
}

/// **Concurrent tail-fault oracle** (§4 decoupled `reserved`/`mapped`, §5 detect-and-kill, from a
/// thread). A spawned worker accesses an address in the **reserved-but-unmapped tail** (`1<<20`, well
/// past the 64 KiB backed window and past any host page so it works on 4 KiB *and* 16 KiB hosts). The
/// contrast pins the I1 change *and* that it holds off the root vCPU:
/// - **Fully mapped** (`reserved == mapped`): `1<<20` masks to offset 0 → wraps in → the worker
///   completes and the run returns.
/// - **`reserved > mapped`** (2^24): the tail address is in `[mapped, reserved)` → an uncommitted
///   access that must **detect-and-kill** — the worker faults, the trap propagates through `join`, and
///   the whole run traps `MemoryFault` on both backends. A thread-context bug that *wrapped* instead
///   (escaping into the wrong/uncommitted page) would let the run complete here.
///
/// Unix only (the reserved-tail fault path; matches the single-threaded `escape_oracle.rs` cases),
/// but page-size-independent so it runs on both x86-64 and aarch64.
#[cfg(unix)]
#[test]
fn concurrent_tail_access_detect_and_kills_from_thread() {
    let src = "\
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  v3 = i64.const 0
  return v3
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v1 = i64.const 1048576
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 0
  return v3
}
";
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let init = vec![0u8; 65536];

    // Fully mapped: the tail address wraps in (1<<20 & 65535 == 0) → the worker completes.
    let mut fuel = 50_000_000u64;
    let (ir0, _) = run_capture_reserved(&m, 0, &[], &mut fuel, &init, 0);
    let (jo0, _) = compile_and_run_capture_reserved(&m, 0, &[], &init, 0).expect("jit");
    assert!(
        ir0.is_ok(),
        "interp should complete under a fully-mapped window: {ir0:?}"
    );
    assert!(
        matches!(jo0, JitOutcome::Returned(_)),
        "jit should complete under a fully-mapped window: {jo0:?}"
    );

    // Reserved (2^24) > mapped (2^16): the worker's tail access must detect-and-kill on both backends.
    let mut fuel = 50_000_000u64;
    let (ir1, _) = run_capture_reserved(&m, 0, &[], &mut fuel, &init, 24);
    let (jo1, _) = compile_and_run_capture_reserved(&m, 0, &[], &init, 24).expect("jit");
    assert!(
        ir1.is_err(),
        "interp did not fault on the thread's out-of-mapped tail access: {ir1:?}"
    );
    assert!(
        matches!(jo1, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the thread's tail access: {jo1:?}"
    );
}
