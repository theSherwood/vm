//! Confinement masking — the **isolated, separately-fuzzable security unit**
//! (`DESIGN.md` §4, invariant I1; §18). This is the part of escape-freedom the
//! *verifier does not cover*: memory safety is enforced here (and, in production,
//! by the JIT lowering that must match this reference), not by typing.
//!
//! The whole crate is one idea — *confine a guest address into its window* — kept
//! deliberately tiny and dependency-free so it can be audited and fuzzed on its own.
//!
//! ## The crisp invariant (§4)
//! A window is `size = 1 << size_log2` bytes, a power of two. Confinement is the
//! single operation
//! ```text
//! confine(addr, offset) = (addr + offset) & (size - 1)
//! ```
//! Masking the **final effective address** (after folding the immediate `offset`)
//! is load-bearing: masking only the operand and *then* adding a large immediate
//! could land past the guard region in a neighbouring window. A multi-byte access
//! that would cross the top of the window is rejected ([`Window::checked`] returns
//! `None`), modelling the guard region that backs every window.
//!
//! **Totality:** every function here is total and panic-free for *all* inputs
//! (any `addr`/`offset`/`width`/`size_log2`), so the unit is safe to drive from a
//! fuzzer. Overflow/wrap of the masked address stays in-window and is mere guest
//! self-corruption (allowed). See `Window::checked` for the post-condition that the
//! property tests and the `mask` fuzz target assert.
#![forbid(unsafe_code)]
#![no_std]

/// A confined linear-memory window: `1 << size_log2` bytes, power-of-two sized so
/// confinement is a single bitmask. Construct with [`Window::new`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Window {
    // Invariant: `<= 63`, so `1u64 << size_log2` never overflows. Private + clamped
    // in `new` so a `Window` can never name a non-representable size.
    size_log2: u8,
}

impl Window {
    /// A window of `1 << size_log2` bytes. `size_log2` is clamped to `63`
    /// defensively — a verified module always has `size_log2 < 64`, but this unit
    /// must stay total even on unverified (fuzzed) input.
    #[inline]
    pub fn new(size_log2: u8) -> Window {
        Window {
            size_log2: if size_log2 > 63 { 63 } else { size_log2 },
        }
    }

    /// Window size in bytes (`1 << size_log2`, always `>= 1`).
    #[inline]
    pub fn size(self) -> u64 {
        1u64 << self.size_log2
    }

    /// The confinement mask (`size - 1`).
    #[inline]
    pub fn mask(self) -> u64 {
        self.size() - 1
    }

    /// Confine the **final effective address** into `[0, size)`:
    /// `(addr + offset) & (size - 1)`, with wrapping add. This is the load-bearing
    /// operation (§4); the result is always a valid offset into the window.
    #[inline]
    pub fn confine(self, addr: u64, offset: u64) -> u64 {
        addr.wrapping_add(offset) & self.mask()
    }

    /// Confine, then guard-check a `width`-byte access. Returns the in-window base
    /// offset, or `None` if `[base, base+width)` would cross the top of the window
    /// (the guard-region fault).
    ///
    /// Post-condition (asserted by the property tests / `mask` fuzz target): if this
    /// returns `Some(base)` then
    /// `base == confine(addr, offset)` **and** `base + width <= size`,
    /// hence `[base, base + width) ⊆ [0, size)`.
    #[inline]
    pub fn checked(self, addr: u64, offset: u64, width: u32) -> Option<u64> {
        let base = self.confine(addr, offset);
        match base.checked_add(width as u64) {
            Some(end) if end <= self.size() => Some(base),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic PRNG (SplitMix64) so property tests need no dev-deps —
    /// the escape-TCB crates stay dependency-free.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// The core property: `checked` either rejects, or returns an in-window base
    /// equal to the masked final address — for arbitrary inputs, never panicking.
    #[test]
    fn checked_confines_or_faults() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for _ in 0..2_000_000 {
            let addr = rng.next();
            let offset = rng.next();
            let width = (rng.next() % 8 + 1) as u32; // 1..=8
            let size_log2 = (rng.next() % 66) as u8; // include out-of-range (clamped)

            let w = Window::new(size_log2);
            let size = w.size();
            let expected_base = addr.wrapping_add(offset) & (size - 1);

            // `confine` is exactly the documented mask.
            assert_eq!(w.confine(addr, offset), expected_base);

            match w.checked(addr, offset, width) {
                Some(base) => {
                    assert_eq!(base, expected_base, "base must be the masked address");
                    assert!(base + width as u64 <= size, "access escaped the window");
                }
                None => {
                    // The only reason to fault is a width-overrun past the top.
                    assert!(
                        expected_base + width as u64 > size,
                        "faulted on an in-window access"
                    );
                }
            }
        }
    }

    #[test]
    fn boundary_cases() {
        let w = Window::new(16); // 64 KiB
        let size = w.size();
        // An aligned 8-byte load at the last full slot is fine.
        assert_eq!(w.checked(size - 8, 0, 8), Some(size - 8));
        // One byte further crosses the top -> fault.
        assert_eq!(w.checked(size - 7, 0, 8), None);
        // A single byte at the very last address is fine.
        assert_eq!(w.checked(size - 1, 0, 1), Some(size - 1));
        // An out-of-window address aliases back in (the I1 property).
        assert_eq!(w.checked(size + 8, 0, 4), Some(8));
        // Folding the immediate offset participates in the mask.
        assert_eq!(w.checked(size - 4, 8, 4), Some(4));
    }

    #[test]
    fn degenerate_one_byte_window() {
        let w = Window::new(0); // size 1, mask 0
        assert_eq!(w.size(), 1);
        assert_eq!(w.checked(12345, 0, 1), Some(0)); // everything aliases to 0
        assert_eq!(w.checked(0, 0, 2), None); // 2 bytes never fit
    }

    #[test]
    fn largest_window_does_not_overflow() {
        let w = Window::new(63);
        assert_eq!(w.size(), 1u64 << 63);
        // Near the top of a 2^63 window: an access that fits, and one that doesn't.
        assert_eq!(w.checked((1u64 << 63) - 8, 0, 8), Some((1u64 << 63) - 8));
        assert_eq!(w.checked((1u64 << 63) - 1, 0, 2), None);
        // size_log2 over the max is clamped, not a shift-overflow panic.
        assert_eq!(Window::new(200).size(), 1u64 << 63);
    }
}
