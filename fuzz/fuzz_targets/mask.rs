//! libFuzzer target for the **confinement masking unit** (`DESIGN.md` §4, I1; §18).
//!
//! The masking unit is the part of escape-freedom the verifier does *not* cover, so
//! it is fuzzed in isolation against its crisp invariant: for any inputs,
//! `Window::checked` is total (never panics) and, when it returns `Some(base)`,
//! `base == confine(addr, offset)` and `[base, base+width) ⊆ [0, mapped) ⊆ [0, reserved)`.
//! Both the fully-mapped form ([`Window::new`]) and the decoupled reserved/mapped form
//! ([`Window::with_mapped`], the large-reserved-window + guard model, §4) are driven.
//!
//! Run: `cargo +nightly fuzz run mask`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_mask::Window;

/// Assert the crisp invariant for one window: `confine` is exactly the mask into
/// `[0, reserved)`, and `checked` either rejects or returns that masked base within the
/// backed `[0, mapped)`. Total — never panics.
fn check(w: Window, addr: u64, offset: u64, width: u32) {
    let reserved = w.reserved();
    let mapped = w.mapped();
    assert!(mapped <= reserved, "mapped must not exceed reserved");
    let expected = addr.wrapping_add(offset) & (reserved - 1);

    // `confine` is exactly the documented mask, always within the reserved domain.
    assert_eq!(w.confine(addr, offset), expected);
    assert!(expected < reserved);

    match w.checked(addr, offset, width) {
        Some(base) => {
            assert_eq!(base, expected, "base must be the masked final address");
            assert!(
                base + width as u64 <= mapped,
                "confined access escaped the mapped region"
            );
        }
        None => assert!(
            expected + width as u64 > mapped,
            "faulted on a fully-mapped access"
        ),
    }
}

fuzz_target!(|data: &[u8]| {
    // Derive the inputs from the fuzz bytes (pad short inputs with zeros).
    let mut b = [0u8; 26];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let addr = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let offset = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let width = (b[16] % 8) as u32 + 1; // 1..=8
    let reserved_log2 = b[17]; // any byte, incl. out-of-range (Window clamps)
    let mapped = u64::from_le_bytes(b[18..26].try_into().unwrap()); // clamped to reserved

    // Fully-mapped form (mapped == reserved == size) and the decoupled form.
    check(Window::new(reserved_log2), addr, offset, width);
    check(Window::with_mapped(reserved_log2, mapped), addr, offset, width);
});
