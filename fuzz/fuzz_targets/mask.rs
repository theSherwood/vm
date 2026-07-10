//! libFuzzer target for the **trap-confinement unit** (`DESIGN.md` §4, I1; §14; §18).
//!
//! The confinement unit is the part of escape-freedom the verifier does *not* cover, so
//! it is fuzzed in isolation against its crisp invariant: for any inputs,
//! `Window::checked` is total (never panics) and returns `Some(a)` **iff**
//! `[addr+offset, addr+offset+width) ⊆ [0, mapped)`, in which case `a == confine(addr, offset)`
//! (the raw, *unmasked* `base + addr + offset`) and `[a, a+width) ⊆ [base, base+mapped) ⊆
//! [base, base+reserved)`. Every out-of-window access **faults** (`None`) — no aliasing back in.
//! All three constructors are driven — the fully-mapped form ([`Window::new`]), the decoupled
//! reserved/mapped form ([`Window::with_mapped`], §4), and the **§14 nested sub-window**
//! ([`Window::sub`], at an arbitrary `base`) — so the confinement-stays-in-its-(sub-)region property
//! is fuzzed for nested children too.
//!
//! Run: `cargo +nightly fuzz run mask`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_mask::Window;

/// Assert the crisp invariant for one window (top-level or §14 sub-window) under **trap-confinement**:
/// `confine` is the raw effective address `base + addr + offset` (no masking), and `checked` admits an
/// access iff it lies fully within the backed `[base, base+mapped)`, returning that address, and
/// otherwise faults. Total — never panics.
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

    // `confine` is the raw (unmasked) effective address — trap-confinement, no `& mask`.
    let a = w.confine(addr, offset);
    assert_eq!(
        a,
        base.wrapping_add(addr.wrapping_add(offset)),
        "confine = base + addr + offset"
    );

    // In bounds iff `[addr+offset, addr+offset+width) ⊆ [0, mapped)`, computed overflow-free.
    let in_bounds = addr
        .checked_add(offset)
        .and_then(|e| e.checked_add(width as u64))
        .is_some_and(|end| end <= mapped);

    match w.checked(addr, offset, width) {
        Some(c) => {
            assert!(in_bounds, "admitted an out-of-window access");
            assert_eq!(c, a, "checked must return the confined address");
            assert!(c >= base, "checked address below base");
            assert!(
                (c - base) + width as u64 <= mapped,
                "confined access escaped the backed [base, base+mapped)"
            );
        }
        None => assert!(!in_bounds, "faulted on an in-mapped access"),
    }
}

/// Fuzz the **bulk-memory span confinement** predicate (D62): the JIT's `confine_span` traps a copy
/// unless the whole span `[ptr, ptr+len)` lies in `[0, reserved)`, computed *without* overflowing via
/// the two sub-checks `len > reserved` (before the `reserved - len` subtraction can wrap) and
/// `ptr > reserved - len`, gated by `len != 0` (a zero-length op is an in-bounds no-op). This asserts
/// that overflow-avoiding formula matches a clean `u128` oracle for every input — the arithmetic is
/// the subtle, security-critical part (a wrong subcheck would admit an out-of-window bulk copy).
fn check_span(ptr: u64, len: u64, reserved: u64) {
    // The exact formula emitted by `svm_jit::confine_span` (kept in sync with it).
    let jit_oob = len != 0 && (len > reserved || ptr > reserved.wrapping_sub(len));
    // Independent oracle in u128 (cannot overflow): the span escapes iff it is non-empty and its end
    // exceeds the reservation.
    let oracle_oob = len != 0 && (ptr as u128 + len as u128) > reserved as u128;
    assert_eq!(
        jit_oob, oracle_oob,
        "span-confinement OOB mismatch: ptr={ptr} len={len} reserved={reserved}"
    );
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

    // Bulk-memory span confinement (D62): drive the same reservation the scalar check uses. `reserved`
    // is a power of two ≤ 2^63; `offset` doubles as the second span length so both a small and a large
    // length are exercised against `addr`.
    let reserved = Window::new(reserved_log2).reserved();
    check_span(addr, mapped, reserved);
    check_span(addr, offset, reserved);
    check_span(offset, addr, reserved);
});
