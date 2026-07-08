//! Escape-oracle plumbing tests (`DESIGN.md` §4/§18): the generative differential
//! (`jit_fuzz`/`diff`) also byte-compares the final guest window across the interpreter
//! and JIT for float-free modules — realizing the §18 move *"verified ⇒ cannot escape"* at
//! the module level (the `fuzz/mask` unit proves the confinement arithmetic in isolation).
//! The broad coverage is the seed loop; these hand-written cases pin the mechanism down under
//! **trap-confinement**: that an **out-of-window access faults at the offending access** —
//! identically on both backends, with no aliasing back into the window — and that the capture
//! path reflects guest stores.

use svm_interp::{run_capture, run_capture_reserved, run_capture_sub, Value};
use svm_jit::{
    compile_and_run_capture, compile_and_run_capture_reserved, compile_and_run_capture_sub,
    JitOutcome,
};

/// True on a 4 KiB-page host. A couple of `reserved_*` cases below hardcode the mapped/tail
/// boundary at address 4096 (`memory 12`); on a 16 KiB-page host (macOS ARM) the JIT rounds the
/// 4 KiB `mapped` up to one 16 KiB host page, so that address is still backed and the "tail
/// access faults" premise no longer holds. Those cases skip there; the guard itself is covered on
/// 16 KiB by the `svm-jit` PAL conformance test (64 KiB / 1 MiB windows, host-page-aligned).
#[cfg(unix)]
fn host_4k() -> bool {
    // SAFETY: sysconf is always safe; _SC_PAGESIZE is positive.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) == 4096 }
}

/// Parse + verify a module, then run it on both backends with `init` seeding the window; return
/// both final-window snapshots (asserting both ran to completion and agree on the result). Used by
/// the success cases (an in-window access, or a no-op) where both backends complete.
fn both_windows(src: &str, init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I32(0)], &mut fuel, init, 0);
    let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[0i64], init, 0).expect("jit");
    assert!(ir.is_ok(), "interp trapped: {ir:?}");
    assert!(
        matches!(jo, JitOutcome::Returned(_)),
        "jit did not return: {jo:?}"
    );
    (imem, jmem)
}

/// Like [`both_windows`], but returns each backend's **trap disposition** (interpreter-trapped,
/// JIT outcome) alongside the two snapshots — the harness for the trap-confinement cases, which
/// assert both backends fault on the same out-of-window access. (Memory is not compared on a trap:
/// a faulted store never writes, and capture-on-trap is not a guaranteed byte-for-byte snapshot.)
fn both_windows_disposition(src: &str, init: &[u8]) -> (bool, JitOutcome, Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I32(0)], &mut fuel, init, 0);
    let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[0i64], init, 0).expect("jit");
    (ir.is_err(), jo, imem, jmem)
}

#[test]
fn out_of_window_store_faults_identically() {
    // Window = 2^8 = 256 bytes. A store to 261 is out of the window; under trap-confinement it
    // raises `MemoryFault` *at the offending access* on both backends (no aliasing to 261 & 255 = 5),
    // leaving the window untouched.
    let src = "\
memory 8
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 261
  v2 = i32.const 171
  i32.store8 v1 v2
  v3 = i32.const 0
  return v3
}
";
    let (it, jo, imem, _jmem) = both_windows_disposition(src, &[0u8; 256]);
    assert!(it, "interp did not fault on the out-of-window store");
    assert!(
        matches!(jo, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the out-of-window store: {jo:?}"
    );
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        0,
        "a faulted store must not write the window"
    );
}

#[test]
fn far_address_and_offset_fault_identically() {
    // A huge base plus a folded immediate offset is out of the window; under trap-confinement it
    // faults on both backends (the old model masked it to an in-window byte). base = i64::MAX,
    // offset 8.
    let src = "\
memory 8
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 9223372036854775807
  v2 = i32.const 200
  i32.store8 v1 v2 offset=8
  v3 = i32.const 0
  return v3
}
";
    let (it, jo, imem, _jmem) = both_windows_disposition(src, &[0u8; 256]);
    assert!(it, "interp did not fault on the far out-of-window store");
    assert!(
        matches!(jo, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the far store: {jo:?}"
    );
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        0,
        "a faulted store must not write the window"
    );
}

#[test]
fn seed_survives_when_untouched() {
    // A no-op body must leave the seeded window exactly as provided, on both backends — so
    // a real divergence later can't hide behind a zeroed window.
    let init: Vec<u8> = (0..256)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let src = "\
memory 8
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
";
    let (imem, jmem) = both_windows(src, &init);
    assert_eq!(imem, init, "interp did not preserve the seeded window");
    assert_eq!(jmem, init, "jit did not preserve the seeded window");
}

/// The JIT elides the bounds check when the address is *provably* in-window (the §1a
/// "check-when-not" path). This pins that the elided path stays confined: `addr = (n & 7)*8`
/// is provably ≤ 56 in a 256-byte window, so the check is dropped — and for adversarial `n`
/// (incl. negative / i64::MAX, whose low bits still confine via `& 7`) the interpreter and the
/// JIT must leave an identical window and land at the same slot (never faulting, never escaping).
#[test]
fn elided_bounded_address_confines() {
    let src = "\
memory 8
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 7
  v2 = i64.and v0 v1
  v3 = i64.const 8
  v4 = i64.mul v2 v3
  v5 = i64.const 171
  i64.store v4 v5
  v6 = i64.load v4
  return v6
}
";
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    for n in [0i64, 5, 0x12345, -1, i64::MAX, 9_999_999] {
        let init = [0u8; 256];
        let mut fuel = 1_000_000u64;
        let (ir, imem) = run_capture(&m, 0, &[Value::I64(n)], &mut fuel, &init);
        let (jo, jmem) = compile_and_run_capture(&m, 0, &[n], &init).expect("jit");
        assert_eq!(ir.ok(), Some(vec![Value::I64(171)]), "interp result n={n}");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[171]),
            "jit {jo:?} n={n}"
        );
        assert_eq!(imem, jmem, "elided-address windows diverge at n={n}");
        let slot = ((n as u64 & 7) * 8) as usize;
        assert_eq!(imem[slot], 171, "store landed at wrong slot for n={n}");
    }
}

/// Run both backends with a host **reservation** (`reserved_log2`): only the declared `memory`
/// bytes are backed inside the larger reserved range. Returns whether each backend trapped plus the
/// two window snapshots, so a test can assert they agree on the trap disposition *and* the final
/// memory (the escape-oracle under the decoupled `reserved`/`mapped` model, §4). `n` is the entry
/// arg (an `i64`); `init` must be `1 << size_log2` bytes (the JIT snapshots the whole backed window,
/// so the seed length sets the compared extent).
#[cfg(unix)] // only the `cfg(unix)` reserved-tail tests use it (windows runs them once svm-run's
             // Memory-cap path is ported — Phase 3.5); avoids a dead-code warning on windows.
fn both_reserved(
    src: &str,
    init: &[u8],
    reserved_log2: u8,
    n: i64,
) -> (bool, JitOutcome, Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I64(n)], &mut fuel, init, reserved_log2);
    let (jo, jmem) =
        compile_and_run_capture_reserved(&m, 0, &[n], init, reserved_log2).expect("jit");
    (ir.is_err(), jo, imem, jmem)
}

/// Trap-confinement (§4): an access past the backed `mapped` prefix **faults** on both backends —
/// whether it lands in a decoupled reserved-but-unmapped tail (`reserved > mapped`) or one byte past
/// a fully-mapped window (`reserved == mapped`). Trap-confinement bounds the guest to `[0, mapped)`
/// in *both* configs, so the reservation is now purely internal defense-in-depth (it no longer
/// changes the guest-visible bound). This pins that the same out-of-`mapped` store faults either way.
#[cfg(unix)]
#[test]
fn reserved_tail_access_faults_identically() {
    if !host_4k() {
        return; // hardcodes the 4 KiB mapped/tail boundary (address 4096); see `host_4k`.
    }
    // memory 12 = 4 KiB backed (exactly one page). Store one byte at address 4096 — the first
    // byte *past* the mapped window.
    let src = "\
memory 12
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 4096
  v2 = i32.const 7
  i32.store8 v1 v2
  v3 = i32.const 0
  return v3
}
";
    // Fully mapped (reserved == mapped == 4 KiB): 4096 is one past the top ⇒ fault on both backends
    // (trap-confinement — the old model wrapped 4096 & 4095 = 0 in).
    let (it0, jo0, _im0, _jm0) = both_reserved(src, &[0u8; 4096], 0, 0);
    assert!(
        it0,
        "interp should fault one byte past a fully-mapped window"
    );
    assert!(
        matches!(jo0, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the past-the-top access (fully mapped): {jo0:?}"
    );

    // Reserved (2^24) > mapped (2^12): 4096 is in the unmapped tail ⇒ fault on both backends.
    let (it1, jo1, _im1, _jm1) = both_reserved(src, &[0u8; 4096], 24, 0);
    assert!(
        it1,
        "interp did not fault on the out-of-mapped (tail) access"
    );
    assert!(
        matches!(jo1, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the tail access: {jo1:?}"
    );
}

/// Under `reserved > mapped`, an access that stays within the backed `mapped` prefix still
/// succeeds and leaves byte-identical windows on both backends — the reservation only changes
/// what happens *outside* `mapped`, not in-window behaviour.
#[cfg(unix)]
#[test]
fn reserved_in_mapped_access_matches() {
    // Address = (n & 511) * 8, provably ≤ 4088 < mapped (4 KiB) — always in the backed prefix.
    let src = "\
memory 12
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 511
  v2 = i64.and v0 v1
  v3 = i64.const 8
  v4 = i64.mul v2 v3
  v5 = i32.const 99
  i32.store8 v4 v5
  v6 = i32.const 0
  return v6
}
";
    for n in [0i64, 1, 7, 511, 512, i64::MAX, -1] {
        let (it, jo, imem, jmem) = both_reserved(src, &[0u8; 4096], 24, n);
        assert!(!it, "interp faulted on an in-mapped access, n={n}");
        assert!(matches!(jo, JitOutcome::Returned(_)), "jit n={n}: {jo:?}");
        assert_eq!(imem, jmem, "in-mapped windows diverge at n={n}");
        let slot = ((n as u64 & 511) * 8) as usize;
        assert_eq!(imem[slot], 99, "store landed at wrong slot for n={n}");
    }
}

/// Run both backends with the guest confined to a §14 **nested sub-window** `[base, base+size)`
/// of a fully-backed parent of `parent_bytes` (size = the module's declared `memory`). `init`
/// seeds the whole parent; the two returned snapshots are the whole parent window, plus each
/// backend's trap disposition. This is the **sub-window escape-oracle**: the lowering shifts an
/// in-bounds child offset by `base` (matching `svm_mask::Window::sub`) and faults out-of-child
/// accesses, so byte-comparing the whole parent proves the child stayed in its slice on both
/// backends — the riskiest claim of §14 nesting #1.
fn both_sub(
    src: &str,
    init: &[u8],
    base: u64,
    parent_bytes: u64,
) -> (bool, JitOutcome, Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_sub(&m, 0, &[Value::I32(0)], &mut fuel, init, base, parent_bytes);
    let (jo, jmem) =
        compile_and_run_capture_sub(&m, 0, &[0i64], init, base, parent_bytes).expect("jit");
    (ir.is_err(), jo, imem, jmem)
}

/// §14 nesting #1: a guest runs over a 4 KiB child window placed at offset 64 KiB inside a 128 KiB
/// parent. Its **in-window** stores must land only inside the child's slice `[64 KiB, 64 KiB + 4 KiB)`,
/// shifted there by `base`, identically on both backends, leaving every other parent byte exactly as
/// seeded. The lowering computes `mem_base + base + (addr+offset)` once the bounds check proves the
/// access in-child; this differential proves the `+ base` shift is right (and confines) on both the
/// interpreter and the JIT. (Out-of-child faulting is [`sub_window_out_of_child_faults`].)
#[test]
fn sub_window_confines_child_to_its_slice() {
    const PARENT: u64 = 128 << 10; // 128 KiB
    const BASE: u64 = 64 << 10; // size-aligned 64 KiB offset
    const SIZE: u64 = 4096; // memory 12 → a 4 KiB child window
    let src = "\
memory 12
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 261
  v2 = i32.const 171
  i32.store8 v1 v2
  v3 = i64.const 1000
  v4 = i32.const 200
  i32.store8 v3 v4
  v5 = i64.const 4095
  v6 = i32.const 99
  i32.store8 v5 v6
  v7 = i32.const 0
  return v7
}
";
    // Seed the whole parent with a non-zero pattern so an escaped write (or a divergent read)
    // outside the child's slice is observable, not hidden behind zeros.
    let init: Vec<u8> = (0..PARENT)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let (it, jo, imem, jmem) = both_sub(src, &init, BASE, PARENT);
    assert!(!it, "interp faulted on an in-child store");
    assert!(matches!(jo, JitOutcome::Returned(_)), "jit: {jo:?}");

    assert_eq!(imem.len(), PARENT as usize, "snapshot is the whole parent");
    assert_eq!(
        imem, jmem,
        "sub-window escape-oracle: interp/JIT parents diverge"
    );

    // Every byte outside the child's slice is untouched (the seed survives) — confinement.
    for i in 0..PARENT {
        if !(BASE..BASE + SIZE).contains(&i) {
            assert_eq!(
                imem[i as usize], init[i as usize],
                "parent byte {i} escaped"
            );
        }
    }
    // The three in-window stores landed at their child offsets, shifted into the slice by `base`.
    assert_eq!(imem[(BASE + 261) as usize], 171, "store @261 misplaced");
    assert_eq!(imem[(BASE + 1000) as usize], 200, "store @1000 misplaced");
    assert_eq!(imem[(BASE + 4095) as usize], 99, "store @4095 misplaced");
}

/// §14 nesting #1, the confinement half: a store past the top of the child window (`4101` in a
/// 4 KiB child) **faults** on both backends — it is not aliased back into the slice — and every
/// parent byte outside the slice survives untouched. Pairs with
/// [`sub_window_confines_child_to_its_slice`] (the in-window half).
#[test]
fn sub_window_out_of_child_faults() {
    const PARENT: u64 = 128 << 10; // 128 KiB
    const BASE: u64 = 64 << 10; // size-aligned 64 KiB offset
    const SIZE: u64 = 4096; // memory 12 → a 4 KiB child window
    let src = "\
memory 12
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 4101
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i32.const 0
  return v3
}
";
    let init: Vec<u8> = (0..PARENT)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let (it, jo, imem, _jmem) = both_sub(src, &init, BASE, PARENT);
    assert!(it, "interp did not fault on the out-of-child store");
    assert!(
        matches!(jo, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the out-of-child store: {jo:?}"
    );
    // Nothing outside the child's slice was touched (the faulted store wrote nothing).
    for i in 0..PARENT {
        if !(BASE..BASE + SIZE).contains(&i) {
            assert_eq!(
                imem[i as usize], init[i as usize],
                "parent byte {i} escaped"
            );
        }
    }
}

/// Detect-and-kill (§4/§5): a store that overruns the top of the window is caught as a clean
/// `MemoryFault` — the host survives, no crash. Under trap-confinement the bounds check fires
/// *before* the access (an 8-byte store at 65532 needs `[65532, 65540) ⊆ [0, 65536)`, which fails);
/// the hardware guard page behind `mapped` remains as defense-in-depth for any elided path.
#[cfg(unix)]
#[test]
fn guard_page_fault_is_detect_and_kill() {
    // memory 16 = 64 KiB. An 8-byte store at 65532 writes [65532,65540), overrunning the 64 KiB top.
    let src = "\
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
}
";
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let out = svm_jit::compile_and_run(&m, 0, &[0]).expect("jit");
    assert!(
        matches!(out, JitOutcome::Trapped(svm_jit::TrapKind::MemoryFault)),
        "expected a caught MemoryFault, got {out:?}"
    );
}
