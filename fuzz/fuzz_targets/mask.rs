//! libFuzzer target for the **confinement masking unit** (`DESIGN.md` §4, I1; §14; §18).
//!
//! The masking unit is the part of escape-freedom the verifier does *not* cover, so
//! it is fuzzed in isolation against its crisp invariant: for any inputs,
//! `Window::checked` is total (never panics) and, when it returns `Some(a)`,
//! `a == confine(addr, offset)` and `[a, a+width) ⊆ [base, base+mapped) ⊆ [base, base+reserved)`.
//! All three constructors are driven — the fully-mapped form ([`Window::new`]), the decoupled
//! reserved/mapped form ([`Window::with_mapped`], §4), and the **§14 nested sub-window**
//! ([`Window::sub`], at an arbitrary `base`) — so the confinement-stays-in-its-(sub-)region property
//! is fuzzed for nested children too.
//!
//! Run: `cargo +nightly fuzz run mask`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_mask::Window;

/// Assert the crisp invariant for one window (top-level or §14 sub-window): `confine` folds the
/// effective address into this window's region `[base, base+reserved)` as `base + (x & mask)`, and
/// `checked` either rejects or returns that confined address within the backed `[base, base+mapped)`.
/// Total — never panics.
fn check(w: Window, addr: u64, offset: u64, width: u32) {
    let base = w.base();
    let reserved = w.reserved();
    let mapped = w.mapped();
    assert!(mapped <= reserved, "mapped must not exceed reserved");
    assert_eq!(base & (reserved - 1), 0, "base must be size-aligned");
    assert!(
        base <= u64::MAX - (reserved - 1),
        "base + (reserved-1) must not overflow"
    );

    let rel = addr.wrapping_add(offset) & (reserved - 1); // child-relative, in [0, reserved)
    let a = w.confine(addr, offset);
    assert_eq!(a, base + rel, "confine = base + masked offset");
    // The decisive property: the confined address stays inside this window's (sub-)region.
    assert!(a >= base, "confined address fell below its window base");
    assert!(a - base < reserved, "confined address reached/passed its window top");

    match w.checked(addr, offset, width) {
        Some(c) => {
            assert_eq!(c, a, "checked must return the confined address");
            assert!(c >= base, "checked address below base");
            assert!(
                (c - base) + width as u64 <= mapped,
                "confined access escaped the backed [base, base+mapped)"
            );
        }
        None => assert!(
            rel + width as u64 > mapped,
            "faulted on an in-mapped access"
        ),
    }
}

fuzz_target!(|data: &[u8]| {
    // Derive the inputs from the fuzz bytes (pad short inputs with zeros).
    let mut b = [0u8; 34];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let addr = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let offset = u64::from_le_bytes(b[8..16].try_into().unwrap());
    // 1..=8 scalar widths, plus 16 for the §17 `v128` access (D58) — the masking unit is
    // width-parametric, so the wider SIMD load/store rides the same confinement invariant.
    let width = match b[16] % 9 {
        8 => 16,
        k => (k as u32) + 1,
    };
    let reserved_log2 = b[17]; // any byte, incl. out-of-range (Window clamps)
    let mapped = u64::from_le_bytes(b[18..26].try_into().unwrap()); // clamped to reserved
    let base = u64::from_le_bytes(b[26..34].try_into().unwrap()); // clamped size-aligned by `sub`

    // Fully-mapped form (mapped == reserved == size), the decoupled form, and a §14 sub-window at an
    // arbitrary base — all must keep every access confined to their own (sub-)region.
    check(Window::new(reserved_log2), addr, offset, width);
    check(Window::with_mapped(reserved_log2, mapped), addr, offset, width);
    check(Window::sub(base, reserved_log2, mapped), addr, offset, width);
});
