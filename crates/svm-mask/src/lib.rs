//! Confinement masking — the **isolated, separately-fuzzable security unit**
//! (`DESIGN.md` §4, invariant I1; §18). This is the part of escape-freedom the
//! *verifier does not cover*: memory safety is enforced here (and, in production,
//! by the JIT lowering that must match this reference), not by typing.
//!
//! The whole crate is one idea — *confine a guest address into its window* — kept
//! deliberately tiny and dependency-free so it can be audited and fuzzed on its own.
//!
//! ## The crisp invariant (§4)
//! A window has two extents (decoupled for the "guard-when-bounded" perf model, §4):
//! - **`reserved`** = `1 << reserved_log2` bytes, a power of two — the **mask domain**.
//!   Confinement masks every address into `[0, reserved)`; this is what the JIT's mask
//!   constant (`reserved − 1`) confines to, so no access can name an address outside the
//!   reserved virtual range.
//! - **`mapped`** ≤ `reserved` bytes — the **backed** region `[0, mapped)`. The range
//!   `[mapped, reserved)` is reserved-but-unmapped (a `PROT_NONE` guard region in
//!   production), so an access that lands there **faults** rather than touching memory.
//!
//! Confinement is the single masking operation (with `base = 0` for a top-level window)
//! ```text
//! confine(addr, offset) = base + ((addr + offset) & (reserved - 1))
//! ```
//! Masking the **final effective address** (after folding the immediate `offset`)
//! is load-bearing: masking only the operand and *then* adding a large immediate
//! could land past the guard region in a neighbouring window. An access whose confined
//! address + width is not fully within the backed region is rejected ([`Window::checked`]
//! returns `None`), modelling the guard region that backs every window — both a width-overrun
//! off the top and (once `mapped < reserved`) a landing in the unmapped tail.
//!
//! ## Nesting (§14)
//! A window can be a power-of-two-aligned **sub-region** of an enclosing window — a parent grants a
//! child the slice `[base, base + reserved)` ([`Window::sub`]). Confinement then folds the child
//! offset into exactly that slice (`base + (x & (reserved-1))`), so a child access can **never leave
//! its sub-region** — hence never reach the parent's other regions or escape the parent window, at
//! identical per-access cost. `base` is `reserved`-aligned, so the add is also `base | (x & mask)`.
//!
//! A **fully-mapped** window (`mapped == reserved`, the historical case — [`Window::new`])
//! collapses both extents to one and behaves exactly as before: `confine` masks to `size`
//! and `checked` faults only on a width-overrun past the top. The decoupled form
//! ([`Window::with_mapped`]) is the substrate for the large reserved window + guard the
//! perf model needs; no in-tree caller uses it yet, so this split is behaviour-preserving.
//!
//! **Totality:** every function here is total and panic-free for *all* inputs
//! (any `addr`/`offset`/`width`/`reserved_log2`/`mapped`), so the unit is safe to drive
//! from a fuzzer. Overflow/wrap of the masked address stays within `reserved` and is mere
//! guest self-corruption (allowed). See `Window::checked` for the post-condition that the
//! property tests and the `mask` fuzz target assert.
#![forbid(unsafe_code)]
#![no_std]

/// A confined linear-memory window with a power-of-two **mask domain** (`reserved`) and a
/// backed sub-extent (`mapped` ≤ `reserved`). Confinement masks into `[0, reserved)`; an
/// access outside `[0, mapped)` faults. Construct fully-mapped with [`Window::new`], or
/// decoupled with [`Window::with_mapped`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Window {
    // Absolute offset of this window's low byte within the enclosing address space (the parent
    // backing for a §14 sub-window; `0` for a top-level window). Invariant: **size-aligned**
    // (`base & (reserved-1) == 0`) and `base + reserved <= 2^64`, so `base + (x & mask)` confines
    // into `[base, base+reserved)` without overflow. Clamped in `sub`.
    base: u64,
    // Invariant: `<= 63`, so `1u64 << reserved_log2` never overflows. Private + clamped in
    // the constructors so a `Window` can never name a non-representable reserved size.
    reserved_log2: u8,
    // Invariant: `<= reserved()` (the backed prefix). Clamped in the constructors so the
    // invariant holds even on fuzzed input; `[mapped, reserved)` is the unmapped tail.
    mapped: u64,
}

impl Window {
    /// A **fully-mapped** window of `1 << size_log2` bytes (`mapped == reserved`, the
    /// historical shape). `size_log2` is clamped to `63` defensively — a verified module
    /// always has `size_log2 < 64`, but this unit must stay total even on unverified
    /// (fuzzed) input.
    #[inline]
    pub fn new(size_log2: u8) -> Window {
        let reserved_log2 = if size_log2 > 63 { 63 } else { size_log2 };
        Window {
            base: 0,
            reserved_log2,
            mapped: 1u64 << reserved_log2,
        }
    }

    /// A window whose **mask domain** is `1 << reserved_log2` bytes but whose **backed**
    /// region is only the prefix `[0, mapped)`. `reserved_log2` is clamped to `63` and
    /// `mapped` is clamped to `reserved()` (preserving `mapped <= reserved`), so the
    /// constructor is total on any input. This is the decoupled form the large-reserved-
    /// window + guard perf model uses (§4); confinement still masks into `[0, reserved)`,
    /// but an access landing in the unmapped tail `[mapped, reserved)` now faults.
    #[inline]
    pub fn with_mapped(reserved_log2: u8, mapped: u64) -> Window {
        let reserved_log2 = if reserved_log2 > 63 {
            63
        } else {
            reserved_log2
        };
        let reserved = 1u64 << reserved_log2;
        Window {
            base: 0,
            reserved_log2,
            mapped: if mapped > reserved { reserved } else { mapped },
        }
    }

    /// A **sub-window** at absolute offset `base` — a power-of-two sub-region of an enclosing
    /// window (a §14 nested child: the parent grants a `1 << reserved_log2`-byte slice with `mapped`
    /// backed). Confinement maps a child offset `x` to `base + (x & (reserved-1))`, so every child
    /// access lands in `[base, base + reserved)` and can therefore **never leave its sub-region** —
    /// hence never reach the parent's other regions or escape the parent window. `base` is clamped
    /// **size-aligned** (the power-of-two-aligned sub-region §4/§14 requires) which also makes
    /// `base + reserved <= 2^64`, so the unit stays total + overflow-free on any input;
    /// `reserved_log2`/`mapped` are clamped as in [`Window::with_mapped`]. A `base == 0` sub-window
    /// is exactly a [`Window::with_mapped`] window.
    #[inline]
    pub fn sub(base: u64, reserved_log2: u8, mapped: u64) -> Window {
        let reserved_log2 = if reserved_log2 > 63 {
            63
        } else {
            reserved_log2
        };
        let reserved = 1u64 << reserved_log2;
        // `!mask == 2^64 - reserved` (for `mask = reserved - 1`): `base & !mask` is therefore both
        // size-aligned *and* `<= 2^64 - reserved`, so `base + reserved` never overflows.
        let base = base & !(reserved - 1);
        Window {
            base,
            reserved_log2,
            mapped: if mapped > reserved { reserved } else { mapped },
        }
    }

    /// The absolute offset of this window's low byte (`0` for a top-level window; the parent-relative
    /// base for a §14 sub-window). The confined address is `base() + (masked child offset)`.
    #[inline]
    pub fn base(self) -> u64 {
        self.base
    }

    /// The **mask domain** in bytes (`1 << reserved_log2`, always `>= 1`): confinement
    /// masks every address into `[0, reserved)`.
    #[inline]
    pub fn reserved(self) -> u64 {
        1u64 << self.reserved_log2
    }

    /// The **backed** extent in bytes (`<= reserved()`): accesses outside `[0, mapped)`
    /// fault. Equal to [`Window::reserved`] for a fully-mapped [`Window::new`] window.
    #[inline]
    pub fn mapped(self) -> u64 {
        self.mapped
    }

    /// Window size in bytes — an alias for [`Window::reserved`] (the mask domain), retained
    /// for callers that predate the `reserved`/`mapped` split. For a fully-mapped window
    /// this is also the backed extent.
    #[inline]
    pub fn size(self) -> u64 {
        self.reserved()
    }

    /// The confinement mask (`reserved - 1`).
    #[inline]
    pub fn mask(self) -> u64 {
        self.reserved() - 1
    }

    /// Confine the **final effective address** into the window `[base, base + reserved)`:
    /// `base + ((addr + offset) & (reserved - 1))`, with wrapping add. This is the load-bearing
    /// operation (§4/§14); the result is always within this window's reserved range — for a
    /// top-level window (`base == 0`) that is `[0, reserved)`, for a sub-window it is the granted
    /// sub-region, so a child can never name an address outside it. (It may still land in the
    /// unmapped tail — [`Window::checked`] is what enforces the `mapped` bound.)
    #[inline]
    pub fn confine(self, addr: u64, offset: u64) -> u64 {
        self.base
            .wrapping_add(addr.wrapping_add(offset) & self.mask())
    }

    /// Confine, then guard-check a `width`-byte access against the **backed** region.
    /// Returns the confined absolute address, or `None` if the access is not fully within the
    /// backed region — i.e. a width-overrun off the top, or (when `mapped < reserved`) a landing in
    /// the unmapped tail. Both model the guard-region fault.
    ///
    /// Post-condition (asserted by the property tests / `mask` fuzz target): if this returns
    /// `Some(a)` then `a == confine(addr, offset)`, the child-relative offset `a - base` satisfies
    /// `(a - base) + width <= mapped`, and hence
    /// `[a, a + width) ⊆ [base, base + mapped) ⊆ [base, base + reserved)`.
    #[inline]
    pub fn checked(self, addr: u64, offset: u64, width: u32) -> Option<u64> {
        // The masked *child-relative* offset, in `[0, reserved)`; the guard check is against the
        // backed `[0, mapped)`. On success the returned address is absolute (`base + rel`).
        let rel = addr.wrapping_add(offset) & self.mask();
        match rel.checked_add(width as u64) {
            Some(end) if end <= self.mapped => Some(self.base.wrapping_add(rel)),
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

    /// The decoupled form: confinement still masks into `[0, reserved)`, but `checked`
    /// faults whenever the masked base + width leaves the backed `[0, mapped)` — including
    /// a landing in the unmapped tail `[mapped, reserved)`. Total, never panics.
    #[test]
    fn checked_confines_or_faults_with_mapped() {
        let mut rng = Rng(0x0FED_CBA9_8765_4321);
        for _ in 0..2_000_000 {
            let addr = rng.next();
            let offset = rng.next();
            let width = (rng.next() % 8 + 1) as u32; // 1..=8
            let reserved_log2 = (rng.next() % 66) as u8; // include out-of-range (clamped)
                                                         // A `mapped` biased to land both below and at/above `reserved` (clamped).
            let reserved = 1u64 << (reserved_log2.min(63));
            let mapped_in = rng.next() % (reserved.saturating_mul(2).max(1));

            let w = Window::with_mapped(reserved_log2, mapped_in);
            // Invariants the constructor must establish on any input.
            assert!(
                w.mapped() <= w.reserved(),
                "mapped must not exceed reserved"
            );
            assert_eq!(w.reserved(), reserved);
            assert_eq!(w.mapped(), mapped_in.min(reserved));

            let expected_base = addr.wrapping_add(offset) & (reserved - 1);
            // `confine` masks into the reserved domain regardless of `mapped`.
            assert_eq!(w.confine(addr, offset), expected_base);
            assert!(expected_base < reserved);

            match w.checked(addr, offset, width) {
                Some(base) => {
                    assert_eq!(base, expected_base, "base must be the masked address");
                    assert!(
                        base + width as u64 <= w.mapped(),
                        "access escaped the mapped region"
                    );
                }
                None => assert!(
                    expected_base + width as u64 > w.mapped(),
                    "faulted on a fully-mapped access"
                ),
            }
        }
    }

    #[test]
    fn unmapped_tail_faults() {
        // reserved = 64 KiB (mask domain), but only the low 256 bytes are backed.
        let w = Window::with_mapped(16, 256);
        assert_eq!(w.reserved(), 1 << 16);
        assert_eq!(w.mapped(), 256);
        assert_eq!(w.size(), 1 << 16); // alias = reserved (the mask domain)
        assert_eq!(w.mask(), (1 << 16) - 1);

        // In the backed prefix: confines and passes the guard.
        assert_eq!(w.checked(0, 0, 8), Some(0));
        assert_eq!(w.checked(248, 0, 8), Some(248)); // last fully-backed 8-byte slot
                                                     // Crossing the top of the backed region faults, even though it is well within
                                                     // `reserved` (this is the new behaviour the split enables).
        assert_eq!(w.checked(252, 0, 8), None);
        // Confinement still masks into `reserved`; an address in the unmapped tail is a
        // valid *masked* offset but faults the guard check.
        assert_eq!(w.confine(1000, 0), 1000);
        assert_eq!(w.checked(1000, 0, 1), None);
        // An out-of-reserved address still aliases into `[0, reserved)` (I1), then faults
        // because it lands in the unmapped tail.
        assert_eq!(w.confine((1 << 16) + 1000, 0), 1000);
        assert_eq!(w.checked((1 << 16) + 1000, 0, 1), None);
    }

    #[test]
    fn fully_mapped_matches_new() {
        // `with_mapped(n, 1<<n)` is exactly `new(n)` — the historical fully-mapped shape.
        for n in [0u8, 1, 8, 16, 63] {
            let a = Window::new(n);
            let b = Window::with_mapped(n, 1u64 << n);
            assert_eq!(a, b);
            assert_eq!(a.mapped(), a.reserved());
        }
        // `mapped` over-large is clamped down to `reserved`, recovering the fully-mapped form.
        assert_eq!(Window::with_mapped(16, u64::MAX), Window::new(16));
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

    /// The §14 **nesting** property: a sub-window confines every access into *its own* sub-region
    /// `[base, base + reserved)` — never below `base`, never at/above the top — for arbitrary inputs,
    /// and `checked` admits only accesses fully within the backed `[base, base + mapped)`. This is
    /// exactly what makes a nested child unable to reach the parent's other memory or escape the
    /// parent window. Total + panic-free on any input (drives the same way the `mask` fuzz target
    /// does).
    #[test]
    fn sub_window_confines_within_subregion() {
        let mut rng = Rng(0xCAFE_F00D_1234_5678);
        for _ in 0..2_000_000 {
            let base_in = rng.next();
            let addr = rng.next();
            let offset = rng.next();
            let width = (rng.next() % 8 + 1) as u32; // 1..=8
            let reserved_log2 = (rng.next() % 66) as u8; // include out-of-range (clamped)
            let reserved = 1u64 << reserved_log2.min(63);
            let mask = reserved - 1;
            let mapped_in = rng.next() % reserved.saturating_mul(2).max(1);

            let w = Window::sub(base_in, reserved_log2, mapped_in);
            let base = w.base();
            // Constructor invariants on any input.
            assert_eq!(
                base,
                base_in & !mask,
                "base size-aligned to the clamped reserved"
            );
            assert_eq!(base & mask, 0, "base must be size-aligned");
            assert!(
                base <= u64::MAX - mask,
                "base + (reserved-1) must not overflow"
            );
            assert_eq!(w.reserved(), reserved);
            assert!(w.mapped() <= reserved);

            let rel = addr.wrapping_add(offset) & mask; // child-relative, in [0, reserved)
            let a = w.confine(addr, offset);
            assert_eq!(a, base + rel, "confine = base + masked offset");
            // The decisive property: the confined address stays inside this sub-window's region.
            assert!(a >= base, "sub-window access fell below its sub-region");
            assert!(
                a - base < reserved,
                "sub-window access reached/passed its sub-region top"
            );

            match w.checked(addr, offset, width) {
                Some(c) => {
                    assert_eq!(c, a, "checked must return the confined address");
                    assert!(c >= base, "checked address below base");
                    assert!(
                        (c - base) + width as u64 <= w.mapped(),
                        "checked access left the backed [base, base+mapped)"
                    );
                }
                None => assert!(
                    rel + width as u64 > w.mapped(),
                    "faulted on an in-mapped access"
                ),
            }
        }

        // A `base == 0` sub-window is exactly the fully-mapped/decoupled top-level window.
        assert_eq!(Window::sub(0, 16, 1 << 16), Window::new(16));
        assert_eq!(Window::sub(0, 16, 256), Window::with_mapped(16, 256));
        // Concrete child: a 4 KiB window granted at offset 64 KiB in the parent.
        let child = Window::sub(1 << 16, 12, 1 << 12);
        assert_eq!(child.base(), 1 << 16);
        assert_eq!(child.confine(0, 0), 1 << 16); // child offset 0 → absolute 64 KiB
        assert_eq!(child.confine((1 << 12) + 8, 0), (1 << 16) + 8); // wraps within the child
        assert_eq!(child.checked(4088, 0, 8), Some((1 << 16) + 4088)); // last backed slot
        assert_eq!(child.checked(4092, 0, 8), None); // overruns the child's top → fault
    }
}
