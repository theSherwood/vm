//! Escape-oracle plumbing tests (`DESIGN.md` §4/§18): the generative differential
//! (`jit_fuzz`/`diff`) now also byte-compares the final guest window across the interpreter
//! and JIT for float-free modules — realizing the §18 move *"verified ⇒ cannot escape"* at
//! the module level (the `fuzz/mask` unit already proves the masking arithmetic in
//! isolation). The broad coverage is the seed loop; these hand-written cases pin the
//! mechanism down: that an **out-of-window address is confined to the same in-window byte**
//! on both backends, and that the capture path reflects guest stores.

use svm_interp::{run_capture, Value};
use svm_jit::{compile_and_run_capture, JitOutcome};

/// Parse + verify a module, then run it on both backends with `init` seeding the window;
/// return both final-window snapshots (asserting both ran to completion and agree on the
/// result). This is exactly the escape-oracle contract the fuzzer asserts.
fn both_windows(src: &str, init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture(&m, 0, &[Value::I32(0)], &mut fuel, init);
    let (jo, jmem) = compile_and_run_capture(&m, 0, &[0i64], init).expect("jit");
    assert!(ir.is_ok(), "interp trapped: {ir:?}");
    assert!(
        matches!(jo, JitOutcome::Returned(_)),
        "jit did not return: {jo:?}"
    );
    (imem, jmem)
}

#[test]
fn out_of_window_store_confines_identically() {
    // Window = 2^8 = 256 bytes. A store to 261 must alias to 261 & 255 = 5 on *both*
    // backends (the §4 mask), land there, and leave nothing else touched.
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
    let (imem, jmem) = both_windows(src, &[0u8; 256]);
    assert_eq!(imem, jmem, "escape-oracle: interp/JIT windows diverge");
    assert_eq!(
        imem[5], 171,
        "out-of-window store did not confine to offset 5"
    );
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        1,
        "spurious writes"
    );
}

#[test]
fn far_address_and_offset_fold_into_window() {
    // A huge base plus a folded immediate offset must still mask to an in-window byte,
    // identically on both backends. base = i64::MAX (0x7fff..ff), offset 8 → wraps.
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
    let (imem, jmem) = both_windows(src, &[0u8; 256]);
    assert_eq!(imem, jmem, "escape-oracle: interp/JIT windows diverge");
    // (i64::MAX + 8) & 255 = (0x...07 + 8) & 0xff = 7. The exact byte matters less than the
    // two backends agreeing on it — that agreement *is* the confinement property.
    assert_eq!(imem.iter().filter(|&&b| b == 200).count(), 1, "store lost");
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

/// The JIT elides the confinement mask when the address is *provably* in-window (the §1a
/// "mask-when-not" path). This pins that the elided path stays confined: `addr = (n & 7)*8`
/// is provably ≤ 56 in a 256-byte window, so the mask is dropped — yet for adversarial `n`
/// (incl. negative / i64::MAX, whose low bits still confine via `& 7`) the interpreter (which
/// always masks) and the JIT must still leave an identical window and land at the same slot.
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

/// Detect-and-kill (§4/§5): a store that overruns the top of the window must fault into the
/// guard page and be caught as a clean MemoryFault — the host survives, no crash. (Unix
/// only: other targets have no hardware guard and the masked access reads a heap margin.)
#[cfg(unix)]
#[test]
fn guard_page_fault_is_detect_and_kill() {
    // memory 16 = 64 KiB (page-aligned ⇒ guard page begins exactly at the top). An 8-byte
    // store at 65532 writes [65532,65540), crossing into the guard page at 65536.
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
