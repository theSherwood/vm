//! The shared guest-memory substrate (`DESIGN.md` §12/§13).
//!
//! A [`Region`] is the backing store for a guest window's **anonymous** pages: a flat,
//! demand-zeroed, raw-addressable byte range of the window's *reserved* size. Confinement/masking
//! (§4) is the caller's job ([`svm_mask::Window`]); a `Region` only sees an already-confined offset
//! and bounds-checks it against its own length as defense-in-depth.
//!
//! Why a separate crate: the parallel interpreter (and later the JIT) must run **multiple OS-thread
//! vCPUs over one shared region with real hardware atomics** (§12). A shared raw region + real
//! width-4/8 atomics inherently need `unsafe`, but the interpreter is the `#![forbid(unsafe_code)]`
//! reference oracle. So all the `unsafe` lives *here*, behind a safe API, and is audited/fuzzed in
//! isolation — exactly the role [`svm_mask`] plays for masking.
//!
//! Two backings:
//! - **`Mapped`** (unix): one anonymous `mmap` of the reserved size (lazy: pages cost nothing until
//!   touched, then the kernel zero-fills). Page-aligned, so **real** `AtomicU32`/`AtomicU64` ops
//!   (the §12 hardware atomics the JIT already emits) are sound on it. The substrate parallel
//!   execution runs on.
//! - **`Paged`** (non-unix, or a reservation too large to `mmap`): a `BTreeMap` of zeroed pages.
//!   Single-threaded-only — `Vec<u8>` pages aren't width-aligned, so its "atomics" are plain
//!   value-correct ops, identical to the non-atomic op under a single thread.

use std::collections::BTreeMap;

/// The six read-modify-write operations (§12), mirrored from `svm_ir::AtomicRmwOp` without taking a
/// dependency on the IR crate (this crate sits below it). Each returns the **old** value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

/// A guest window's anonymous-page backing store. See the crate docs for the two variants.
pub enum Region {
    /// Unix: one demand-zeroed anonymous `mmap` of `[0, len)` (page-rounded). Real atomics.
    #[cfg(unix)]
    Mapped(Mapped),
    /// Portable fallback: zeroed pages in a map. Single-threaded; "atomics" are plain ops.
    Paged(Paged),
}

impl Region {
    /// A region addressing `[0, size)` bytes, all reading as zero until written. `page` is the
    /// host page granularity (the unit [`Region::zero`] re-zeroes and the `Paged` chunk size).
    ///
    /// On unix a feasible `size` is `mmap`-backed (the shared substrate); a `size` too large to map
    /// — or any non-unix target — falls back to the paged backing.
    pub fn new(size: u64, page: u64) -> Region {
        #[cfg(unix)]
        {
            if size > 0 {
                if let Some(m) = Mapped::new(size, page) {
                    return Region::Mapped(m);
                }
            }
        }
        Region::Paged(Paged::new(size, page))
    }

    /// The addressable length `[0, size)`.
    pub fn len(&self) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.size,
            Region::Paged(p) => p.size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read one byte; an untouched address reads zero. Out of range reads zero (the caller already
    /// confined into `[0, len)`; this is belt-and-suspenders).
    pub fn byte(&self, off: u64) -> u8 {
        if off >= self.len() {
            return 0;
        }
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.byte(off),
            Region::Paged(p) => p.byte(off),
        }
    }

    /// Write one byte. Out-of-range writes are dropped (the caller confines first).
    pub fn set_byte(&mut self, off: u64, b: u8) {
        if off >= self.len() {
            return;
        }
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.set_byte(off, b),
            Region::Paged(p) => p.set_byte(off, b),
        }
    }

    /// Reset `[off, off+len)` to zero (the `map`/`unmap` "fresh page" semantics). Range is clamped
    /// to `[0, size)`.
    pub fn zero(&mut self, off: u64, len: u64) {
        let len = clamp_len(off, len, self.len());
        if len == 0 {
            return;
        }
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.zero(off, len),
            Region::Paged(p) => p.zero(off, len),
        }
    }

    /// Copy `[off, off+out.len())` into `out` (zero past the touched extent / region end). Used for
    /// the escape-oracle window snapshot, which can span the whole mapped extent — so the mmap
    /// backing bulk-copies rather than dispatching per byte.
    pub fn read_into(&self, off: u64, out: &mut [u8]) {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.read_into(off, out),
            Region::Paged(p) => {
                for (k, slot) in out.iter_mut().enumerate() {
                    *slot = p.byte(off + k as u64);
                }
            }
        }
    }

    /// `width`-byte (4 or 8) sequentially-consistent atomic load (§12). The caller guarantees
    /// natural alignment and in-window bounds.
    pub fn atomic_load(&self, off: u64, width: u32) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_load(off, width),
            Region::Paged(p) => p.plain_load(off, width),
        }
    }

    /// `width`-byte seq-cst atomic store.
    pub fn atomic_store(&mut self, off: u64, width: u32, val: u64) {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_store(off, width, val),
            Region::Paged(p) => p.plain_store(off, width, val),
        }
    }

    /// `width`-byte seq-cst read-modify-write; returns the **old** value.
    pub fn atomic_rmw(&mut self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_rmw(off, width, op, val),
            Region::Paged(p) => p.plain_rmw(off, width, op, val),
        }
    }

    /// `width`-byte seq-cst compare-exchange: store `replacement` iff the current value equals
    /// `expected`; always return the **old** value.
    pub fn atomic_cmpxchg(&mut self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_cmpxchg(off, width, expected, replacement),
            Region::Paged(p) => p.plain_cmpxchg(off, width, expected, replacement),
        }
    }
}

/// Apply an [`RmwOp`] to `(old, v)` truncated to `width` bytes — the value math shared by both
/// backings (the `Mapped` path uses it only for `Paged`-parity in tests; live `Mapped` RMWs use the
/// hardware `fetch_*`).
fn rmw_apply(op: RmwOp, old: u64, v: u64, width: u32) -> u64 {
    let m = width_mask(width);
    let r = match op {
        RmwOp::Add => old.wrapping_add(v),
        RmwOp::Sub => old.wrapping_sub(v),
        RmwOp::And => old & v,
        RmwOp::Or => old | v,
        RmwOp::Xor => old ^ v,
        RmwOp::Xchg => v,
    };
    r & m
}

fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Bytes of `[off, off+len)` that lie within `[0, size)`.
fn clamp_len(off: u64, len: u64, size: u64) -> u64 {
    if off >= size {
        0
    } else {
        len.min(size - off)
    }
}

// ============================== unix: the mmap-backed shared region ==============================

#[cfg(unix)]
pub use mapped::Mapped;

#[cfg(unix)]
mod mapped {
    use super::RmwOp;
    use core::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};

    /// One anonymous `mmap` of `[0, size)` (rounded up to `map_len`). The base is page-aligned, so
    /// any naturally-aligned 4/8-byte access is hardware-atomic-able.
    pub struct Mapped {
        base: *mut u8,
        pub(super) size: u64,
        map_len: usize,
    }

    impl Mapped {
        pub(super) fn new(size: u64, page: u64) -> Option<Mapped> {
            let page = (page as usize).max(1);
            let map_len = round_up(size as usize, page);
            // SAFETY: a fresh anonymous lazy reservation; `MAP_NORESERVE` so a large `size` costs
            // only virtual address space until pages are touched (then kernel-zeroed). Null/MAP_FAILED
            // is handled below (→ caller falls back to the paged backing).
            let base = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    map_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if base == libc::MAP_FAILED || base.is_null() {
                return None;
            }
            Some(Mapped {
                base: base as *mut u8,
                size,
                map_len,
            })
        }

        #[inline]
        fn ptr(&self, off: u64) -> *mut u8 {
            // SAFETY: callers go through `Region`, which bounds `off < size <= map_len`.
            unsafe { self.base.add(off as usize) }
        }

        pub(super) fn byte(&self, off: u64) -> u8 {
            // SAFETY: `off < size`; the page is mapped RW (lazily zero-filled on first touch).
            unsafe { self.ptr(off).read() }
        }

        pub(super) fn set_byte(&mut self, off: u64, b: u8) {
            // SAFETY: as `byte`; `&mut self` rules out concurrent access this turn.
            unsafe { self.ptr(off).write(b) }
        }

        pub(super) fn zero(&mut self, off: u64, len: u64) {
            // SAFETY: `[off, off+len)` is within `[0, size)` (clamped by the caller).
            unsafe { core::ptr::write_bytes(self.ptr(off), 0, len as usize) }
        }

        pub(super) fn read_into(&self, off: u64, out: &mut [u8]) {
            // Copy the in-range prefix in one shot; anything past `size` stays whatever `out` held
            // (the caller zeroes `out` first).
            let avail = self.size.saturating_sub(off) as usize;
            let n = avail.min(out.len());
            if n == 0 {
                return;
            }
            // SAFETY: `[off, off+n)` is within `[0, size)`; `out[..n]` is a distinct caller buffer.
            unsafe { core::ptr::copy_nonoverlapping(self.ptr(off), out.as_mut_ptr(), n) }
        }

        pub(super) fn atomic_load(&self, off: u64, width: u32) -> u64 {
            // SAFETY: caller guarantees `off` is `width`-aligned and in-bounds; base is page-aligned,
            // so `base+off` is `width`-aligned → a valid `AtomicU32`/`U64` location.
            unsafe {
                match width {
                    4 => AtomicU32::from_ptr(self.ptr(off) as *mut u32).load(SeqCst) as u64,
                    _ => AtomicU64::from_ptr(self.ptr(off) as *mut u64).load(SeqCst),
                }
            }
        }

        pub(super) fn atomic_store(&mut self, off: u64, width: u32, val: u64) {
            // SAFETY: aligned + in-bounds as in `atomic_load`.
            unsafe {
                match width {
                    4 => AtomicU32::from_ptr(self.ptr(off) as *mut u32).store(val as u32, SeqCst),
                    _ => AtomicU64::from_ptr(self.ptr(off) as *mut u64).store(val, SeqCst),
                }
            }
        }

        pub(super) fn atomic_rmw(&mut self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
            // SAFETY: aligned + in-bounds as in `atomic_load`.
            unsafe {
                match width {
                    4 => {
                        let a = AtomicU32::from_ptr(self.ptr(off) as *mut u32);
                        let v = val as u32;
                        let old = match op {
                            RmwOp::Add => a.fetch_add(v, SeqCst),
                            RmwOp::Sub => a.fetch_sub(v, SeqCst),
                            RmwOp::And => a.fetch_and(v, SeqCst),
                            RmwOp::Or => a.fetch_or(v, SeqCst),
                            RmwOp::Xor => a.fetch_xor(v, SeqCst),
                            RmwOp::Xchg => a.swap(v, SeqCst),
                        };
                        old as u64
                    }
                    _ => {
                        let a = AtomicU64::from_ptr(self.ptr(off) as *mut u64);
                        match op {
                            RmwOp::Add => a.fetch_add(val, SeqCst),
                            RmwOp::Sub => a.fetch_sub(val, SeqCst),
                            RmwOp::And => a.fetch_and(val, SeqCst),
                            RmwOp::Or => a.fetch_or(val, SeqCst),
                            RmwOp::Xor => a.fetch_xor(val, SeqCst),
                            RmwOp::Xchg => a.swap(val, SeqCst),
                        }
                    }
                }
            }
        }

        pub(super) fn atomic_cmpxchg(
            &mut self,
            off: u64,
            width: u32,
            expected: u64,
            replacement: u64,
        ) -> u64 {
            // SAFETY: aligned + in-bounds as in `atomic_load`. `compare_exchange` returns the prior
            // value in both the `Ok` (swapped) and `Err` (unchanged) arms.
            unsafe {
                match width {
                    4 => {
                        let a = AtomicU32::from_ptr(self.ptr(off) as *mut u32);
                        match a.compare_exchange(
                            expected as u32,
                            replacement as u32,
                            SeqCst,
                            SeqCst,
                        ) {
                            Ok(old) | Err(old) => old as u64,
                        }
                    }
                    _ => {
                        let a = AtomicU64::from_ptr(self.ptr(off) as *mut u64);
                        match a.compare_exchange(expected, replacement, SeqCst, SeqCst) {
                            Ok(old) | Err(old) => old,
                        }
                    }
                }
            }
        }
    }

    impl Drop for Mapped {
        fn drop(&mut self) {
            // SAFETY: releasing exactly the reservation created in `new`.
            unsafe {
                libc::munmap(self.base as *mut libc::c_void, self.map_len);
            }
        }
    }

    fn round_up(n: usize, align: usize) -> usize {
        (n + align - 1) & !(align - 1)
    }
}

// ========================= portable fallback: paged, single-threaded =========================

/// The portable backing: zeroed `page`-sized chunks in a `BTreeMap`, committed on first write. Used
/// on non-unix targets and for reservations too large to `mmap`. Single-threaded only — its pages
/// aren't width-aligned, so its "atomics" are plain value-correct ops (equal to the non-atomic op
/// under one thread; the JIT/`Mapped` path provides true atomicity once threads exist).
pub struct Paged {
    size: u64,
    page: u64,
    pages: BTreeMap<u64, Vec<u8>>,
}

impl Paged {
    fn new(size: u64, page: u64) -> Paged {
        Paged {
            size,
            page: page.max(1),
            pages: BTreeMap::new(),
        }
    }

    fn byte(&self, off: u64) -> u8 {
        let idx = (off % self.page) as usize;
        self.pages.get(&(off / self.page)).map_or(0, |p| p[idx])
    }

    fn set_byte(&mut self, off: u64, b: u8) {
        let idx = (off % self.page) as usize;
        let page = self.page as usize;
        self.pages
            .entry(off / self.page)
            .or_insert_with(|| vec![0u8; page])[idx] = b;
    }

    fn zero(&mut self, off: u64, len: u64) {
        // Whole pages of the range are dropped (an absent page reads zero); partial edges are
        // overwritten byte-wise.
        let mut o = off;
        let end = off + len;
        while o < end {
            let page = o / self.page;
            let page_start = page * self.page;
            let page_end = page_start + self.page;
            if o == page_start && end >= page_end {
                self.pages.remove(&page);
                o = page_end;
            } else {
                let stop = end.min(page_end);
                for b in o..stop {
                    self.set_byte(b, 0);
                }
                o = stop;
            }
        }
    }

    fn plain_load(&self, off: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            raw |= (self.byte(off + k) as u64) << (8 * k);
        }
        raw
    }

    fn plain_store(&mut self, off: u64, width: u32, val: u64) {
        for k in 0..width as u64 {
            self.set_byte(off + k, (val >> (8 * k)) as u8);
        }
    }

    fn plain_rmw(&mut self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        let old = self.plain_load(off, width);
        self.plain_store(off, width, rmw_apply(op, old, val, width));
        old
    }

    fn plain_cmpxchg(&mut self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        let old = self.plain_load(off, width);
        if old == (expected & width_mask(width)) {
            self.plain_store(off, width, replacement);
        }
        old
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn each_region(size: u64, page: u64, mut f: impl FnMut(Region)) {
        f(Region::new(size, page)); // platform default (mmap on unix)
        f(Region::Paged(Paged::new(size, page))); // force the portable path
    }

    #[test]
    fn byte_rw_and_zero_default() {
        each_region(1 << 16, 4096, |mut r| {
            assert_eq!(r.byte(10), 0);
            r.set_byte(10, 0xAB);
            assert_eq!(r.byte(10), 0xAB);
            r.zero(0, 4096);
            assert_eq!(r.byte(10), 0);
        });
    }

    #[test]
    fn out_of_range_is_inert() {
        let mut r = Region::new(4096, 4096);
        r.set_byte(1 << 20, 1); // ignored
        assert_eq!(r.byte(1 << 20), 0);
    }

    #[test]
    fn atomics_value_semantics() {
        each_region(1 << 16, 4096, |mut r| {
            r.atomic_store(8, 8, 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_rmw(8, 8, RmwOp::Add, 1), 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7789);
            // cmpxchg miss leaves it; hit swaps it.
            assert_eq!(r.atomic_cmpxchg(8, 8, 0, 7), 0x1122_3344_5566_7789);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7789);
            assert_eq!(
                r.atomic_cmpxchg(8, 8, 0x1122_3344_5566_7789, 7),
                0x1122_3344_5566_7789
            );
            assert_eq!(r.atomic_load(8, 8), 7);
            // 32-bit width truncates.
            r.atomic_store(16, 4, 0xDEAD_BEEF);
            assert_eq!(r.atomic_load(16, 4), 0xDEAD_BEEF);
            assert_eq!(r.atomic_rmw(16, 4, RmwOp::Xchg, 1), 0xDEAD_BEEF);
        });
    }

    #[test]
    fn read_into_spans_pages() {
        each_region(1 << 16, 4096, |mut r| {
            r.set_byte(4095, 1);
            r.set_byte(4096, 2);
            let mut out = [0u8; 4];
            r.read_into(4094, &mut out);
            assert_eq!(out, [0, 1, 2, 0]);
        });
    }
}
