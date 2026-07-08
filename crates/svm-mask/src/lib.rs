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
//! - **`mapped`** = the **backed** region `[0, mapped)`, the bytes a guest may actually touch.
//!   **Trap-confinement** admits a `width`-byte access iff `[addr+offset, addr+offset+width)`
//!   lies fully within `[0, mapped)`; anything else raises `Trap::MemoryFault` at the offending
//!   access. There is **no masking** — an out-of-bounds address is never aliased back into the
//!   window, it faults.
//! - **`reserved`** = `1 << reserved_log2` bytes, a power of two ≥ `mapped` — the virtual
//!   **reservation**. `[mapped, reserved)` is reserved-but-unmapped (a `PROT_NONE` guard region
//!   in production): defense-in-depth behind the bounds check, never reached on a passing access.
//!
//! The effective address is the plain fold of the immediate `offset` into the operand
//! ```text
//! confine(addr, offset) = base + (addr + offset)   // no `& mask`
//! ```
//! but the address is only *used* once [`Window::checked`] has confirmed the whole
//! `[addr+offset, addr+offset+width)` span is within `[0, mapped)`; otherwise the caller faults.
//! Bounding the **final effective address** (after folding the immediate `offset`) is
//! load-bearing: a wild `addr+offset` yields a wild address that the check rejects, so no access
//! can name a byte outside its backed region. `base + (addr+offset)` cannot overflow on a
//! passing access because it implies `addr+offset < mapped ≤ reserved` and `base+reserved ≤ 2^64`.
//!
//! ## Nesting (§14)
//! A window can be a power-of-two-aligned **sub-region** of an enclosing window — a parent grants a
//! child the slice `[base, base + reserved)` ([`Window::sub`]). Trap-confinement admits a child
//! access iff it lies within `[base, base + mapped)` and otherwise faults, so a child access can
//! **never leave its sub-region** — hence never reach the parent's other regions or escape the
//! parent window. `base` is `reserved`-aligned, so `[base, base+reserved) ⊆` the parent's slice.
//!
//! A **fully-mapped** window (`mapped == reserved`, the historical case — [`Window::new`])
//! collapses both extents to one: `checked` admits an access iff it fits within `size` and
//! faults otherwise. The decoupled form ([`Window::with_mapped`]) is the substrate for the
//! large reserved window + guard the perf model needs.
//!
//! **Totality:** every function here is total and panic-free for *all* inputs
//! (any `addr`/`offset`/`width`/`reserved_log2`/`mapped`), so the unit is safe to drive
//! from a fuzzer. `confine`'s add wraps (mere guest self-corruption, allowed) and yields an
//! address `checked` would reject. See `Window::checked` for the post-condition that the
//! property tests and the `mask` fuzz target assert.
#![forbid(unsafe_code)]
#![no_std]

/// A trap-confined linear-memory window with a **backed** extent (`mapped`) and a power-of-two
/// virtual **reservation** (`reserved` ≥ `mapped`). An access is admitted iff it lies fully
/// within `[0, mapped)` and otherwise faults — no masking, no aliasing. Construct fully-mapped
/// with [`Window::new`], or decoupled with [`Window::with_mapped`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Window {
    // Absolute offset of this window's low byte within the enclosing address space (the parent
    // backing for a §14 sub-window; `0` for a top-level window). Invariant: **size-aligned**
    // (`base & (reserved-1) == 0`) and `base + reserved <= 2^64`, so a bounded access
    // `base + (addr+offset)` (with `addr+offset < mapped <= reserved`) stays in `[base, base+reserved)`
    // without overflow. Clamped in `sub`.
    base: u64,
    // Invariant: `<= 63`, so `1u64 << reserved_log2` never overflows. Private + clamped in
    // the constructors so a `Window` can never name a non-representable reserved size.
    reserved_log2: u8,
    // Invariant: `<= reserved()` (the backed prefix). Clamped in the constructors so the
    // invariant holds even on fuzzed input; `[mapped, reserved)` is the unmapped guard tail.
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

    /// A window whose virtual **reservation** is `1 << reserved_log2` bytes but whose **backed**
    /// region is only the prefix `[0, mapped)`. `reserved_log2` is clamped to `63` and
    /// `mapped` is clamped to `reserved()` (preserving `mapped <= reserved`), so the
    /// constructor is total on any input. This is the decoupled form the large-reserved-
    /// window + guard perf model uses (§4); trap-confinement admits an access iff it lies within
    /// `[0, mapped)`, so an access landing in the unmapped tail `[mapped, reserved)` faults.
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
    /// backed). Trap-confinement maps an in-bounds child offset `x` to `base + x` and faults any
    /// access outside `[0, mapped)`, so every admitted child access lands in `[base, base + mapped)`
    /// and can therefore **never leave its sub-region** — hence never reach the parent's other regions
    /// or escape the parent window. `base` is clamped
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
    /// base for a §14 sub-window). The confined address is `base() + (addr + offset)`.
    #[inline]
    pub fn base(self) -> u64 {
        self.base
    }

    /// The virtual **reservation** in bytes (`1 << reserved_log2`, always `>= 1`, `>= mapped`):
    /// the power-of-two address range reserved for this window; `[mapped, reserved)` is the guard.
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

    /// Window size in bytes — an alias for [`Window::reserved`] (the reservation), retained
    /// for callers that predate the `reserved`/`mapped` split. For a fully-mapped window
    /// this is also the backed extent.
    #[inline]
    pub fn size(self) -> u64 {
        self.reserved()
    }

    /// The reservation mask (`reserved - 1`). Retained for callers that align to the
    /// power-of-two reservation; trap-confinement no longer uses it to mask accesses.
    #[inline]
    pub fn mask(self) -> u64 {
        self.reserved() - 1
    }

    /// The raw absolute effective address `base + addr + offset` — **no confinement** (this is
    /// **trap-confinement**, not the old wrap model: there is no `& mask`). Callers that access
    /// memory must go through [`Window::checked`], which bounds-checks and rejects an out-of-window
    /// access; `confine` is only the address arithmetic for an access already known/about to be
    /// bounded. The add wraps (a wild `addr+offset` yields a wild address that `checked` rejects).
    #[inline]
    pub fn confine(self, addr: u64, offset: u64) -> u64 {
        self.base.wrapping_add(addr.wrapping_add(offset))
    }

    /// **Trap-confinement** (§4/§14/§18): a `width`-byte access is allowed iff
    /// `[addr+offset, addr+offset+width) ⊆ [0, mapped)` — the backed region — with **no masking**.
    /// Returns the absolute address `base + addr + offset` on success, or `None` (⇒ the caller
    /// raises `MemoryFault`) for any out-of-window address. Unlike the old wrap model, an
    /// out-of-window address is **never** aliased back into the window; it faults.
    ///
    /// Post-condition (asserted by the property tests / `mask` fuzz target): if this returns
    /// `Some(a)` then `a == base + addr + offset`, `(a - base) + width <= mapped`, and hence
    /// `[a, a + width) ⊆ [base, base + mapped) ⊆ [base, base + reserved)` — so a verified guest
    /// cannot name an address outside its backed region. The `checked_add`s make the bound itself
    /// overflow-free.
    #[inline]
    pub fn checked(self, addr: u64, offset: u64, width: u32) -> Option<u64> {
        match addr
            .checked_add(offset)
            .and_then(|e| e.checked_add(width as u64))
        {
            Some(end) if end <= self.mapped => {
                Some(self.base.wrapping_add(addr.wrapping_add(offset)))
            }
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

    /// The core property (**trap-confinement**): `checked` either faults, or returns
    /// `base + addr + offset` for an access fully within `[0, mapped)` — no aliasing, never panics.
    #[test]
    fn checked_confines_or_faults() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for _ in 0..2_000_000 {
            let addr = rng.next();
            let offset = rng.next();
            // 1..=8 scalar widths, plus 16 for the §17 `v128` access (D58) — width-parametric, so the
            // wider SIMD access is covered by the same invariant.
            let width = match rng.next() % 9 {
                8 => 16,
                k => (k + 1) as u32,
            };
            let size_log2 = (rng.next() % 66) as u8; // include out-of-range (clamped)

            let w = Window::new(size_log2);
            let size = w.size(); // fully-mapped: mapped == reserved == size
                                 // In bounds iff `addr+offset+width <= size`, computed overflow-free.
            let in_bounds = addr
                .checked_add(offset)
                .and_then(|e| e.checked_add(width as u64))
                .is_some_and(|end| end <= size);

            // `confine` is the raw (unmasked) address.
            assert_eq!(w.confine(addr, offset), addr.wrapping_add(offset));

            match w.checked(addr, offset, width) {
                Some(a) => {
                    assert!(in_bounds, "admitted an out-of-window access");
                    assert_eq!(
                        a,
                        addr.wrapping_add(offset),
                        "checked = addr + offset (base 0)"
                    );
                    assert!(a + width as u64 <= size, "access escaped the window");
                }
                None => assert!(in_bounds == false, "faulted on an in-window access"),
            }
        }
    }

    /// With a decoupled backed extent: `checked` admits an access iff it lies fully within
    /// `[0, mapped)` (unmasked) — anything past `mapped`, including the unmapped tail, faults.
    #[test]
    fn checked_confines_or_faults_with_mapped() {
        let mut rng = Rng(0x0FED_CBA9_8765_4321);
        for _ in 0..2_000_000 {
            let addr = rng.next();
            let offset = rng.next();
            let width = match rng.next() % 9 {
                8 => 16, // §17 `v128` width (D58)
                k => (k + 1) as u32,
            };
            let reserved_log2 = (rng.next() % 66) as u8; // include out-of-range (clamped)
            let reserved = 1u64 << (reserved_log2.min(63));
            let mapped_in = rng.next() % (reserved.saturating_mul(2).max(1));

            let w = Window::with_mapped(reserved_log2, mapped_in);
            assert!(
                w.mapped() <= w.reserved(),
                "mapped must not exceed reserved"
            );
            assert_eq!(w.reserved(), reserved);
            assert_eq!(w.mapped(), mapped_in.min(reserved));

            let in_bounds = addr
                .checked_add(offset)
                .and_then(|e| e.checked_add(width as u64))
                .is_some_and(|end| end <= w.mapped());
            assert_eq!(w.confine(addr, offset), addr.wrapping_add(offset));

            match w.checked(addr, offset, width) {
                Some(a) => {
                    assert!(in_bounds, "admitted an access past mapped");
                    assert_eq!(a, addr.wrapping_add(offset));
                    assert!(a + width as u64 <= w.mapped(), "access escaped mapped");
                }
                None => assert!(in_bounds == false, "faulted on a fully-mapped access"),
            }
        }
    }

    #[test]
    fn out_of_mapped_faults() {
        // reserved = 64 KiB (reservation), but only the low 256 bytes are backed.
        let w = Window::with_mapped(16, 256);
        assert_eq!(w.reserved(), 1 << 16);
        assert_eq!(w.mapped(), 256);

        // In the backed prefix: passes.
        assert_eq!(w.checked(0, 0, 8), Some(0));
        assert_eq!(w.checked(248, 0, 8), Some(248)); // last fully-backed 8-byte slot
                                                     // Crossing the top of the backed region faults — no aliasing.
        assert_eq!(w.checked(252, 0, 8), None);
        // Any address at/above `mapped` faults; `confine` is the raw (unmasked) address.
        assert_eq!(w.confine(1000, 0), 1000);
        assert_eq!(w.checked(1000, 0, 1), None);
        // A far out-of-reservation address is **not** aliased back in — it faults, and `confine`
        // returns the raw address (trap-confinement, not wrap).
        assert_eq!(w.confine((1 << 16) + 1000, 0), (1 << 16) + 1000);
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
        let w = Window::new(16); // 64 KiB, fully mapped
        let size = w.size();
        // An aligned 8-byte load at the last full slot is fine.
        assert_eq!(w.checked(size - 8, 0, 8), Some(size - 8));
        // One byte further crosses the top -> fault.
        assert_eq!(w.checked(size - 7, 0, 8), None);
        // A single byte at the very last address is fine.
        assert_eq!(w.checked(size - 1, 0, 1), Some(size - 1));
        // An out-of-window address faults (no aliasing).
        assert_eq!(w.checked(size + 8, 0, 4), None);
        // The immediate offset pushes the access past the top -> fault.
        assert_eq!(w.checked(size - 4, 8, 4), None);
    }

    #[test]
    fn degenerate_one_byte_window() {
        let w = Window::new(0); // size 1
        assert_eq!(w.size(), 1);
        assert_eq!(w.checked(0, 0, 1), Some(0)); // the one valid byte
        assert_eq!(w.checked(1, 0, 1), None); // one past the top faults (no aliasing)
        assert_eq!(w.checked(12345, 0, 1), None); // far OOB faults
        assert_eq!(w.checked(0, 0, 2), None); // 2 bytes never fit
    }

    #[test]
    fn largest_window_does_not_overflow() {
        let w = Window::new(63);
        assert_eq!(w.size(), 1u64 << 63);
        // Near the top of a 2^63 window: an access that fits, and one that doesn't.
        assert_eq!(w.checked((1u64 << 63) - 8, 0, 8), Some((1u64 << 63) - 8));
        assert_eq!(w.checked((1u64 << 63) - 1, 0, 2), None);
        // A wild address whose `addr+offset` overflows u64 faults (checked_add rejects it).
        assert_eq!(w.checked(u64::MAX, 8, 1), None);
        // size_log2 over the max is clamped, not a shift-overflow panic.
        assert_eq!(Window::new(200).size(), 1u64 << 63);
    }

    /// The §14 **nesting** property under trap-confinement: `checked` admits an access iff it lies
    /// fully within the backed `[base, base + mapped)` (unmasked), returning `base + addr + offset`;
    /// anything else faults. A nested child therefore can neither reach the parent's other memory
    /// nor escape the parent window. Total + panic-free on any input.
    #[test]
    fn sub_window_confines_within_subregion() {
        let mut rng = Rng(0xCAFE_F00D_1234_5678);
        for _ in 0..2_000_000 {
            let base_in = rng.next();
            let addr = rng.next();
            let offset = rng.next();
            let width = match rng.next() % 9 {
                8 => 16, // §17 `v128` width (D58)
                k => (k + 1) as u32,
            };
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

            let in_bounds = addr
                .checked_add(offset)
                .and_then(|e| e.checked_add(width as u64))
                .is_some_and(|end| end <= w.mapped());

            match w.checked(addr, offset, width) {
                Some(c) => {
                    assert!(
                        in_bounds,
                        "admitted an access past the child's backed region"
                    );
                    assert_eq!(
                        c,
                        base.wrapping_add(addr.wrapping_add(offset)),
                        "checked = base + addr + offset"
                    );
                    // Decisive: the access stays inside this sub-window's backed slice.
                    assert!(c >= base, "sub-window access fell below its sub-region");
                    assert!(
                        (c - base) + width as u64 <= w.mapped(),
                        "access left [base, base+mapped)"
                    );
                }
                None => assert!(in_bounds == false, "faulted on an in-mapped access"),
            }
        }

        // A `base == 0` sub-window is exactly the fully-mapped/decoupled top-level window.
        assert_eq!(Window::sub(0, 16, 1 << 16), Window::new(16));
        assert_eq!(Window::sub(0, 16, 256), Window::with_mapped(16, 256));
        // Concrete child: a 4 KiB window granted at offset 64 KiB in the parent.
        let child = Window::sub(1 << 16, 12, 1 << 12);
        assert_eq!(child.base(), 1 << 16);
        assert_eq!(child.confine(0, 0), 1 << 16); // child offset 0 → absolute 64 KiB
        assert_eq!(child.confine((1 << 12) + 8, 0), (1 << 16) + (1 << 12) + 8); // raw (unmasked)
        assert_eq!(child.checked(4088, 0, 8), Some((1 << 16) + 4088)); // last backed slot
        assert_eq!(child.checked(4092, 0, 8), None); // overruns the child's top → fault
        assert_eq!(child.checked(1 << 12, 0, 1), None); // one past the backed top → fault (no alias)
    }
}
