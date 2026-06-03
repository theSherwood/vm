//! libFuzzer target for the **confinement masking unit** (`DESIGN.md` §4, I1; §18).
//!
//! The masking unit is the part of escape-freedom the verifier does *not* cover, so
//! it is fuzzed in isolation against its crisp invariant: for any inputs,
//! `Window::checked` is total (never panics) and, when it returns `Some(base)`,
//! `base == confine(addr, offset)` and `[base, base+width) ⊆ [0, size)`.
//!
//! Run: `cargo +nightly fuzz run mask`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_mask::Window;

fuzz_target!(|data: &[u8]| {
    // Derive the inputs from the fuzz bytes (pad short inputs with zeros).
    let mut b = [0u8; 21];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let addr = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let offset = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let width = (b[16] % 8) as u32 + 1; // 1..=8
    let size_log2 = b[17]; // any byte, incl. out-of-range (Window clamps)

    let w = Window::new(size_log2);
    let size = w.size();
    let expected = addr.wrapping_add(offset) & (size - 1);

    // `confine` is exactly the documented mask, always in-window.
    assert_eq!(w.confine(addr, offset), expected);
    assert!(expected < size);

    match w.checked(addr, offset, width) {
        Some(base) => {
            assert_eq!(base, expected, "base must be the masked final address");
            assert!(
                base + width as u64 <= size,
                "confined access escaped the window"
            );
        }
        None => assert!(
            expected + width as u64 > size,
            "faulted on an in-window access"
        ),
    }
});
