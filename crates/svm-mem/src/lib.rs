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
//! ## Sharing across vCPUs
//!
//! A `Region` is [`Send`] + [`Sync`]: several vCPU threads hold `&Region` and run over the *one*
//! guest memory image — that shared image is what makes them threads of one guest rather than
//! isolated programs. Every accessor therefore takes `&self`. `Region` itself adds **no** locking or
//! ordering policy beyond what each op needs to be language-level sound; the concurrency *semantics*
//! (the memory model, scheduling, `wait`/`notify`) live above it. What each op guarantees:
//! - **atomic ops** (`atomic_*`) — real seq-cst hardware atomics; the sound primitive for concurrent
//!   access to a *shared* location.
//! - **single-byte plain ops** (`byte`/`set_byte`) — relaxed atomics, so even a same-byte race is
//!   *defined* (no UB), just unordered (the guest's responsibility, per the §12 C11-style model).
//! - **bulk ops** (`zero`/`read_into`) — control-plane (`map`/`unmap`/snapshot); they assume no
//!   concurrent access to *their own range*, which holds for steady-state guest execution.
//!
//! Beyond that, a guest data race corrupts only the guest's own confined memory and can never escape
//! the window (§12) — masking + bounds still gate every access.
//!
//! Two backings:
//! - **`Mapped`** (unix): one anonymous `mmap` of the reserved size (lazy: pages cost nothing until
//!   touched, then the kernel zero-fills). Page-aligned, so **real** `AtomicU32`/`AtomicU64` ops
//!   (the §12 hardware atomics the JIT already emits) are sound on it. The substrate parallel
//!   execution runs on.
//! - **`Paged`** (non-unix, or a reservation too large to `mmap`): a `BTreeMap` of zeroed pages
//!   behind a `Mutex`. Correct but serialized — the portable fallback, not the parallel substrate.

use std::collections::BTreeMap;
use std::sync::Mutex;

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

/// A guest window's anonymous-page backing store. See the crate docs for the two variants and the
/// sharing contract. All accessors take `&self`: a `Region` is shared by reference across vCPUs.
pub enum Region {
    /// Unix: one demand-zeroed anonymous `mmap` of `[0, len)` (page-rounded). Real atomics.
    #[cfg(unix)]
    Mapped(Mapped),
    /// Portable fallback: zeroed pages in a `Mutex`-guarded map (serialized, not the parallel path).
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
    pub fn set_byte(&self, off: u64, b: u8) {
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
    /// to `[0, size)`. Control-plane: assumes no concurrent access to the range.
    pub fn zero(&self, off: u64, len: u64) {
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
            Region::Paged(p) => p.read_into(off, out),
        }
    }

    /// `width`-byte (4 or 8) sequentially-consistent atomic load (§12). The caller guarantees
    /// natural alignment and in-window bounds.
    pub fn atomic_load(&self, off: u64, width: u32) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_load(off, width),
            Region::Paged(p) => p.atomic_load(off, width),
        }
    }

    /// `width`-byte seq-cst atomic store.
    pub fn atomic_store(&self, off: u64, width: u32, val: u64) {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_store(off, width, val),
            Region::Paged(p) => p.atomic_store(off, width, val),
        }
    }

    /// `width`-byte seq-cst read-modify-write; returns the **old** value.
    pub fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_rmw(off, width, op, val),
            Region::Paged(p) => p.atomic_rmw(off, width, op, val),
        }
    }

    /// `width`-byte seq-cst compare-exchange: store `replacement` iff the current value equals
    /// `expected`; always return the **old** value.
    pub fn atomic_cmpxchg(&self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => m.atomic_cmpxchg(off, width, expected, replacement),
            Region::Paged(p) => p.atomic_cmpxchg(off, width, expected, replacement),
        }
    }
}

/// Apply an [`RmwOp`] to `(old, v)` truncated to `width` bytes — the value math the `Paged` backing
/// uses (the `Mapped` path uses the hardware `fetch_*` instead).
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
    use core::sync::atomic::{
        AtomicU32, AtomicU64, AtomicU8,
        Ordering::{Relaxed, SeqCst},
    };

    /// One anonymous `mmap` of `[0, size)` (rounded up to `map_len`). The base is page-aligned, so
    /// any naturally-aligned 4/8-byte access is hardware-atomic-able.
    pub struct Mapped {
        base: *mut u8,
        pub(super) size: u64,
        map_len: usize,
    }

    // SAFETY: `Mapped` is the only non-`Send`/`Sync`-by-default piece of `Region` (it holds a raw
    // `*mut u8`). Sharing it across vCPU threads is sound under the contract documented on the crate:
    //   * the mmap is a process-wide reservation owned by this value (freed once on `Drop`), so
    //     moving it between threads (`Send`) transfers nothing thread-local;
    //   * concurrent access through `&Mapped` (`Sync`) is either a real hardware atomic (`atomic_*`,
    //     seq-cst) or a relaxed-atomic single byte (`byte`/`set_byte`) — both *defined* under races,
    //     never UB. Bulk `zero`/`read_into` are control-plane and not raced against live access.
    // Every offset is bounds-checked by `Region` before reaching here, and masking (§4) confines the
    // address upstream, so no access can leave `[0, size)` — a guest race can corrupt only the
    // guest's own memory, never escape the window (§12).
    unsafe impl Send for Mapped {}
    unsafe impl Sync for Mapped {}

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
            // Relaxed atomic so a concurrent same-byte write is defined, not UB. On x86 this is a
            // plain `mov`. SAFETY: `off < size`; a `*mut u8` is trivially 1-aligned for `AtomicU8`.
            unsafe { AtomicU8::from_ptr(self.ptr(off)).load(Relaxed) }
        }

        pub(super) fn set_byte(&self, off: u64, b: u8) {
            // SAFETY: as `byte`.
            unsafe { AtomicU8::from_ptr(self.ptr(off)).store(b, Relaxed) }
        }

        pub(super) fn zero(&self, off: u64, len: u64) {
            // SAFETY: `[off, off+len)` is within `[0, size)` (clamped by the caller). Control-plane:
            // not concurrent with live access to the range.
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

        pub(super) fn atomic_store(&self, off: u64, width: u32, val: u64) {
            // SAFETY: aligned + in-bounds as in `atomic_load`.
            unsafe {
                match width {
                    4 => AtomicU32::from_ptr(self.ptr(off) as *mut u32).store(val as u32, SeqCst),
                    _ => AtomicU64::from_ptr(self.ptr(off) as *mut u64).store(val, SeqCst),
                }
            }
        }

        pub(super) fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
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
            &self,
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
                        match a.compare_exchange(expected as u32, replacement as u32, SeqCst, SeqCst)
                        {
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

// ========================= portable fallback: paged, Mutex-serialized =========================

/// The portable backing: zeroed `page`-sized chunks in a `BTreeMap`, committed on first write, all
/// behind a `Mutex`. Used on non-unix targets and for reservations too large to `mmap`. Correct
/// under sharing but fully serialized — the fallback, not the parallel substrate (which is `Mapped`).
pub struct Paged {
    size: u64,
    page: u64,
    pages: Mutex<BTreeMap<u64, Vec<u8>>>,
}

impl Paged {
    fn new(size: u64, page: u64) -> Paged {
        Paged {
            size,
            page: page.max(1),
            pages: Mutex::new(BTreeMap::new()),
        }
    }

    /// Lock the page map, recovering from a poisoned lock (our ops never panic while holding it, so
    /// the data is always consistent) rather than propagating the panic.
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Vec<u8>>> {
        self.pages.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn byte(&self, off: u64) -> u8 {
        let idx = (off % self.page) as usize;
        self.lock().get(&(off / self.page)).map_or(0, |p| p[idx])
    }

    fn set_byte(&self, off: u64, b: u8) {
        let page = self.page as usize;
        let idx = (off % self.page) as usize;
        let key = off / self.page;
        self.lock().entry(key).or_insert_with(|| vec![0u8; page])[idx] = b;
    }

    fn zero(&self, off: u64, len: u64) {
        let mut map = self.lock();
        // Whole pages of the range are dropped (an absent page reads zero); partial edges are
        // overwritten byte-wise.
        let mut o = off;
        let end = off + len;
        let page_sz = self.page as usize;
        while o < end {
            let key = o / self.page;
            let page_start = key * self.page;
            let page_end = page_start + self.page;
            if o == page_start && end >= page_end {
                map.remove(&key);
                o = page_end;
            } else {
                let stop = end.min(page_end);
                let p = map.entry(key).or_insert_with(|| vec![0u8; page_sz]);
                for b in o..stop {
                    p[(b % self.page) as usize] = 0;
                }
                o = stop;
            }
        }
    }

    fn read_into(&self, off: u64, out: &mut [u8]) {
        let map = self.lock();
        for (k, slot) in out.iter_mut().enumerate() {
            let o = off + k as u64;
            if o >= self.size {
                break;
            }
            let idx = (o % self.page) as usize;
            *slot = map.get(&(o / self.page)).map_or(0, |p| p[idx]);
        }
    }

    // The atomic ops hold the lock across the whole read-modify-write, so they are atomic with
    // respect to one another (true atomicity vs. other backings comes from `Mapped`).
    fn load_locked(map: &BTreeMap<u64, Vec<u8>>, page: u64, off: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            let o = off + k;
            let idx = (o % page) as usize;
            let b = map.get(&(o / page)).map_or(0, |p| p[idx]);
            raw |= (b as u64) << (8 * k);
        }
        raw
    }

    fn store_locked(map: &mut BTreeMap<u64, Vec<u8>>, page: u64, off: u64, width: u32, val: u64) {
        let page_sz = page as usize;
        for k in 0..width as u64 {
            let o = off + k;
            let idx = (o % page) as usize;
            map.entry(o / page).or_insert_with(|| vec![0u8; page_sz])[idx] = (val >> (8 * k)) as u8;
        }
    }

    fn atomic_load(&self, off: u64, width: u32) -> u64 {
        Self::load_locked(&self.lock(), self.page, off, width)
    }

    fn atomic_store(&self, off: u64, width: u32, val: u64) {
        Self::store_locked(&mut self.lock(), self.page, off, width, val);
    }

    fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        let mut map = self.lock();
        let old = Self::load_locked(&map, self.page, off, width);
        Self::store_locked(&mut map, self.page, off, width, rmw_apply(op, old, val, width));
        old
    }

    fn atomic_cmpxchg(&self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        let mut map = self.lock();
        let old = Self::load_locked(&map, self.page, off, width);
        if old == (expected & width_mask(width)) {
            Self::store_locked(&mut map, self.page, off, width, replacement);
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
        each_region(1 << 16, 4096, |r| {
            assert_eq!(r.byte(10), 0);
            r.set_byte(10, 0xAB);
            assert_eq!(r.byte(10), 0xAB);
            r.zero(0, 4096);
            assert_eq!(r.byte(10), 0);
        });
    }

    #[test]
    fn out_of_range_is_inert() {
        let r = Region::new(4096, 4096);
        r.set_byte(1 << 20, 1); // ignored
        assert_eq!(r.byte(1 << 20), 0);
    }

    #[test]
    fn atomics_value_semantics() {
        each_region(1 << 16, 4096, |r| {
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
        each_region(1 << 16, 4096, |r| {
            r.set_byte(4095, 1);
            r.set_byte(4096, 2);
            let mut out = [0u8; 4];
            r.read_into(4094, &mut out);
            assert_eq!(out, [0, 1, 2, 0]);
        });
    }

    #[test]
    fn region_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Region>();
    }

    /// The headline Phase-2 capability: many OS threads sharing `&Region` and racing on one atomic
    /// counter still land on the exact total — i.e. the atomic RMWs are genuinely atomic across
    /// threads over the shared substrate, not just value-correct single-threaded.
    #[test]
    fn shared_atomic_counter_across_threads() {
        const THREADS: u64 = 8;
        const ITERS: u64 = 20_000;
        let r = Region::new(1 << 16, 4096);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        r.atomic_rmw(0, 8, RmwOp::Add, 1);
                    }
                });
            }
        });
        assert_eq!(r.atomic_load(0, 8), THREADS * ITERS);
    }

    /// Non-atomic sharing too: threads writing *disjoint* byte ranges through one `&Region` all land
    /// (no data race — distinct addresses), proving the shared image is one backing, not per-thread.
    #[test]
    fn shared_disjoint_plain_writes() {
        const THREADS: u64 = 8;
        const SPAN: u64 = 1024;
        let r = Region::new(1 << 16, 4096);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let r = &r;
                s.spawn(move || {
                    let v = (t as u8).wrapping_add(1);
                    for i in 0..SPAN {
                        r.set_byte(t * SPAN + i, v);
                    }
                });
            }
        });
        for t in 0..THREADS {
            let v = (t as u8).wrapping_add(1);
            assert_eq!(r.byte(t * SPAN), v);
            assert_eq!(r.byte(t * SPAN + SPAN - 1), v);
        }
    }
}
