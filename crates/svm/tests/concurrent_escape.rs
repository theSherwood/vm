//! **Concurrent escape-oracle** (`DESIGN.md` §4/§18, extended to §12 threads).
//!
//! The single-threaded escape-oracle (`escape_oracle.rs`, `jit_fuzz`) proves *"verified ⇒ cannot
//! escape"* by byte-comparing the final guest window across the interpreter (the confinement
//! reference) and the JIT. The §12 concurrency work grew the escape-TCB — a shared `svm-mem`
//! `Region`, raw hardware atomics, the per-thread JIT runtime + its `mem_base` threading — without
//! extending that oracle. These cases close the gap under **trap-confinement**: a **spawned thread**
//! that accesses an **out-of-window** address must **detect-and-kill** (fault) *identically on both
//! backends* — confinement holds off the root vCPU — while in-window shared accesses from threads
//! stay confined and leave byte-identical windows.
//!
//! Determinism (so the window byte-compare is sound despite nondeterministic scheduling): every
//! shared access is a **commutative** `atomic.rmw.add` (interleaving-invariant final value) or a
//! **disjoint** per-thread plain write; thread handles are kept in SSA values, never the window, so
//! the compared bytes are a pure function of the program, not the schedule or the backend's handle
//! numbering. A fault is likewise deterministic: an out-of-window address faults on the worker's
//! first access, and the trap propagates through `join` to the whole run on both backends.
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
/// (`reserved == mapped`). Used by the in-window cases (shared accesses that stay confined).
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

/// Four worker threads each `atomic.rmw.add 1` a shared counter **100×** at an **in-window** address
/// (offset 8). Confinement must route every one of those 400 atomic accesses *from the spawned
/// threads* to offset 8 identically on both backends. The total (400) is interleaving-invariant
/// (atomic add commutes), so the windows must be byte-identical: all zero except the i64 counter at
/// offset 8. (The out-of-window atomic case is [`concurrent_out_of_window_atomic_faults_from_thread`].)
#[test]
fn concurrent_atomic_shared_counter_agrees() {
    // counter address = 8 (i64, occupies [8,16)), well inside the 64 KiB window.
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
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
  v10 = i64.const 8
  v11 = i64.atomic.load v10
  return v11
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 3() 2(v1)
}
block 2 (v4: i64) {
  v5 = i64.const 8
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 0
  return v10
  }
}
";
    let init = vec![0u8; 65536];
    let (imem, jmem) = both_windows_threaded(src, &init);
    assert_eq!(
        imem, jmem,
        "concurrent escape-oracle: interp/JIT windows diverge (thread-context confinement?)"
    );
    // The counter (i64, little-endian 400) at offset 8 — and *only* offset 8.
    let counter = u64::from_le_bytes(imem[8..16].try_into().unwrap());
    assert_eq!(counter, 400, "shared atomic counter wrong/escaped");
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        2, // 400 = 0x0190 → two non-zero bytes at offsets 8 and 9
        "a concurrent access landed outside the shared counter slot"
    );
}

/// Four threads each write a fixed marker (`0xAA`) with a plain (non-atomic) store to its **own**
/// in-window address handed in as `arg`; the four addresses are **disjoint**, so there is no race
/// and the final window is deterministic. This exercises confinement of *plain* stores issued from
/// spawned threads (the atomic case above covers the atomic path) — each store must land at `arg`
/// and nowhere else, identically on both backends.
#[test]
fn concurrent_disjoint_plain_stores_confine() {
    // Targets in-window offsets 0/8/16/24; each gets the 0xAA marker.
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 0
  v2 = thread.spawn 1 v0 v1
  v3 = i64.const 8
  v4 = thread.spawn 1 v0 v3
  v5 = i64.const 16
  v6 = thread.spawn 1 v0 v5
  v7 = i64.const 24
  v8 = thread.spawn 1 v0 v7
  v9 = thread.join v2
  v10 = thread.join v4
  v11 = thread.join v6
  v12 = thread.join v8
  v13 = i64.const 0
  return v13
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i32.const 170
  i32.store8 v0 v1
  v2 = i64.const 0
  return v2
  }
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
            "a thread's plain store did not land at slot {slot}"
        );
    }
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        4,
        "a plain store from a thread escaped its slot"
    );
}

/// Trap-confinement from a thread (§4/§5 detect-and-kill): four workers each `atomic.rmw.add` a
/// counter at an **out-of-window** address (`65544`, just past the 64 KiB window). Under
/// trap-confinement that access **faults** on the worker's first iteration; the trap propagates
/// through `join` and the whole run traps `MemoryFault` on both backends (the old model masked
/// `65544 & 65535 = 8` and completed). A thread-context confinement bug that let the access through
/// would instead complete the run.
#[test]
fn concurrent_out_of_window_atomic_faults_from_thread() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
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
  v10 = i64.const 0
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 3() 2(v1)
}
block 2 (v4: i64) {
  v5 = i64.const 65544
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 0
  return v10
  }
}
";
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let init = vec![0u8; 65536];
    let mut fuel = 50_000_000u64;
    let (ir, _) = run_capture_reserved(&m, 0, &[], &mut fuel, &init, 0);
    let (jo, _) = compile_and_run_capture_reserved(&m, 0, &[], &init, 0).expect("jit");
    assert!(
        ir.is_err(),
        "interp did not fault on the thread's out-of-window atomic: {ir:?}"
    );
    assert!(
        matches!(jo, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the thread's out-of-window atomic: {jo:?}"
    );
}

/// **Concurrent tail-fault oracle** (§4 decoupled `reserved`/`mapped`, §5 detect-and-kill, from a
/// thread). A spawned worker accesses an address past the backed window (`1<<20`, well past the
/// 64 KiB `mapped` and past any host page, so it works on 4 KiB *and* 16 KiB hosts). Under
/// trap-confinement the guest-visible bound is `mapped` in **both** reservation configs, so the
/// worker faults either way — proving the fault holds off the root vCPU and does not depend on the
/// reservation. The trap propagates through `join` and the whole run traps `MemoryFault` on both
/// backends. A thread-context bug that *wrapped* instead would let the run complete.
///
/// Unix only (the reserved-tail fault path; matches the single-threaded `escape_oracle.rs` cases),
/// but page-size-independent so it runs on both x86-64 and aarch64.
#[cfg(unix)]
#[test]
fn concurrent_tail_access_detect_and_kills_from_thread() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  v3 = i64.const 0
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v1 = i64.const 1048576
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 0
  return v3
  }
}
";
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let init = vec![0u8; 65536];

    // Both reservation configs bound the guest to `[0, mapped)`, so `1<<20` (past mapped) faults
    // either way — fully mapped (`reserved == mapped`) and reserved (2^24) > mapped.
    for reserved_log2 in [0u8, 24] {
        let mut fuel = 50_000_000u64;
        let (ir, _) = run_capture_reserved(&m, 0, &[], &mut fuel, &init, reserved_log2);
        let (jo, _) =
            compile_and_run_capture_reserved(&m, 0, &[], &init, reserved_log2).expect("jit");
        assert!(
            ir.is_err(),
            "interp did not fault on the thread's out-of-mapped access (reserved_log2={reserved_log2}): {ir:?}"
        );
        assert!(
            matches!(jo, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
            "jit did not detect-and-kill the thread's access (reserved_log2={reserved_log2}): {jo:?}"
        );
    }
}
