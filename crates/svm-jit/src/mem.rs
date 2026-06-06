//! Guest-window allocation + detect-and-kill trap recovery (`DESIGN.md` §4/§5), over a small
//! **Platform Abstraction Layer** (PAL, §16/D51).
//!
//! The window is a large **reserved** virtual range whose backed prefix `[0, mapped)` is committed
//! read/write and whose tail `[mapped, reserved)` plus a trailing guard page are inaccessible. The
//! masking lowering confines every access into `[0, reserved)` (the JIT mask const = `reserved−1`),
//! so an access lands either in the backed prefix or in the inaccessible tail/guard — where it
//! **faults**, and the fault is caught and turned into a clean [`TrapKind::MemoryFault`] instead of
//! corrupting the host (§5 detect-and-kill). This fires on a width-overrun at the very top of the
//! window or — defense-in-depth — a masking/elision bug the guard caught.
//!
//! Everything platform-specific is the small [`pal`] module: reserve / commit / protect / release
//! virtual address space, and run a call under a guard that converts an in-window fault into a
//! caught return. The window *model* (page rounding, the `mapped`/`total` bookkeeping, the RW
//! slice) is portable and shared. Today there is a **unix** PAL (`mmap` + `mprotect(PROT_NONE)` +
//! a SIGSEGV/SIGBUS handler via `trap_shim.c`); the **windows** PAL (`VirtualAlloc` +
//! `VirtualProtect(PAGE_NOACCESS)` + a Vectored Exception Handler) is the next leg (§4 "Platform
//! support", Phase 3.5). The confinement arithmetic is identical on every target, so the
//! interpreter↔JIT differential (§18) is the cross-platform conformance oracle.

use crate::TrapKind;
use core::ffi::c_void;

/// The compiled entry trampoline ABI (see `build_trampoline`): `(args, results, mem_base,
/// fn_table_base, trap_out)`. The 4th pointer is opaque here (`FnEntry*` to the JIT).
type Entry = extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64);

/// Protection the PAL applies to a committed range of the window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prot {
    /// Readable + writable (the backed prefix; restored before a snapshot read).
    Rw,
    /// Read-only (the D40 const data segment; a later write faults).
    Ro,
}

#[inline]
fn round_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

// ============================ portable window model (over the PAL) ============================

/// A guest window: `mapped` readable/writable bytes (the backed prefix `[0, mapped)`) inside a
/// larger `reserved` virtual range whose tail `[mapped, reserved)` plus one trailing guard page are
/// inaccessible (so an access past `mapped` faults → §4/§5 detect-and-kill). A fully-mapped window
/// is just `reserved == mapped` (the historical single-extent case). Built on the [`pal`] below.
pub(crate) struct GuestWindow {
    base: *mut u8,
    mapped: usize, // backed RW bytes `[0, mapped)` (the logical window the guest declared)
    total: usize,  // full reservation length: reserved (page-rounded) + one guard page
}

impl GuestWindow {
    /// Reserve `reserved` bytes (page-rounded) + a guard page as inaccessible, then commit the
    /// `mapped` backed prefix read/write. `reserved` is raised to at least `mapped`.
    ///
    /// For the inaccessible tail's fault boundary to agree with the interpreter's byte-exact
    /// `mapped` bound, `mapped` must be page-aligned whenever `reserved > mapped` — true for any
    /// `size_log2 >= 12`, which every caller of the decoupled form satisfies.
    pub(crate) fn new(mapped: usize, reserved: usize) -> GuestWindow {
        if mapped == 0 {
            return GuestWindow {
                base: std::ptr::null_mut(),
                mapped: 0,
                total: 0,
            };
        }
        let page = pal::page_size();
        let reserved = reserved.max(mapped);
        let rw = round_up(mapped, page);
        let total = round_up(reserved, page) + page; // reserved + one guard page
                                                     // SAFETY: a fresh inaccessible reservation (a huge `reserved` costs only virtual address
                                                     // space until pages are committed/touched). Checked non-null below.
        let base = unsafe { pal::reserve(total) };
        assert!(!base.is_null(), "svm-jit: window reserve failed");
        // SAFETY: commit the backed prefix `[0, rw)` read/write; the tail + guard stay inaccessible
        // so any access past `mapped` faults.
        unsafe { pal::commit_rw(base, rw) };
        GuestWindow {
            base,
            mapped,
            total,
        }
    }

    /// The logical (backed) window `[0, mapped)`, readable/writable (freshly committed pages are
    /// zeroed). Bytes in the reserved-but-inaccessible tail are not part of this slice.
    pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
        if self.mapped == 0 {
            return &mut [];
        }
        // SAFETY: `[base, base+mapped)` is committed RW for the window's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.mapped) }
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    /// Re-enable read+write on the whole backed region `[0, mapped)`. The guest may have changed
    /// page protections through the `Memory` capability (`unmap`→inaccessible, `protect`→read-only),
    /// so a snapshot read of the window could otherwise fault *outside* the guarded call and crash
    /// the host. Idempotent; a no-op cost when nothing changed.
    pub(crate) fn restore_rw(&self) {
        if self.mapped == 0 {
            return;
        }
        let rw = round_up(self.mapped, pal::page_size());
        // SAFETY: `[base, base+rw)` is the window's backed region, owned for its lifetime.
        unsafe { pal::protect(self.base, rw, Prot::Rw) };
    }

    /// Map the whole pages touched by `[offset, offset+len)` **read-only** — the D40 const data
    /// segment (§3a/§4): a later guest write to them faults into the guarded range. The data must
    /// already be written (this only changes protection). A producer keeps RO data on its own
    /// pages; protection is page-granular, so a shared page would over-protect.
    pub(crate) fn protect_ro(&self, offset: u64, len: u64) {
        if self.mapped == 0 || len == 0 {
            return;
        }
        let page = pal::page_size();
        let start = (offset as usize / page) * page;
        let end = round_up((offset + len) as usize, page);
        // SAFETY: `[base+start, base+end)` lies within the backed region (the caller bounds
        // `offset+len <= mapped`, which is page-rounded up to `rw`), owned for the lifetime.
        unsafe { pal::protect(self.base.add(start), end - start, Prot::Ro) };
    }

    /// The address range a fault must land in to be attributed to this window (the whole
    /// reservation, so the inaccessible tail + guard page are covered). `(0, 0)` when there is no
    /// window.
    fn fault_range(&self) -> (usize, usize) {
        if self.mapped == 0 {
            (0, 0)
        } else {
            (self.base as usize, self.base as usize + self.total)
        }
    }
}

impl Drop for GuestWindow {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: releasing exactly the reservation we created.
            unsafe { pal::release(self.base, self.total) };
        }
    }
}

/// Run the JIT entry `code` under the guard. Returns `true` if a fault in the window's guarded
/// range was caught and unwound (→ the caller reports `MemoryFault`).
///
/// # Safety
/// `code` must be the finalized trampoline with the [`Entry`] signature, and the pointers must
/// satisfy its contract (valid for the call, outliving it).
pub(crate) unsafe fn run_guarded(
    window: &GuestWindow,
    code: *const u8,
    args: *const i64,
    results: *mut i64,
    mem_base: *mut u8,
    fn_table: *const c_void,
    trap_cell: *mut i64,
) -> bool {
    pal::install_guard();
    let (lo, hi) = window.fault_range();
    let f: Entry = std::mem::transmute(code);
    pal::run_guarded(f, args, results, mem_base, fn_table, trap_cell, lo, hi)
}

/// The trap code a caught guard fault reports.
pub(crate) const FAULT_TRAP: i64 = TrapKind::MemoryFault as i64;

// ================================= PAL: unix ==================================================
// `mmap` reservation + `mprotect` (PROT_NONE tail/guard) + a SIGSEGV/SIGBUS handler that
// `siglongjmp`s out of an in-window fault (the C shim `trap_shim.c`, for sound setjmp/longjmp).
#[cfg(unix)]
mod pal {
    use super::{Entry, Prot};
    use core::ffi::c_void;
    use std::sync::Once;

    extern "C" {
        fn svm_install_trap_handler();
        #[allow(clippy::too_many_arguments)]
        fn svm_run_guarded(
            f: Entry,
            a: *const i64,
            r: *mut i64,
            m: *mut u8,
            t: *const c_void,
            tc: *mut i64,
            lo: usize,
            hi: usize,
        ) -> i32;
    }

    pub(super) fn page_size() -> usize {
        // SAFETY: sysconf is always safe to call; _SC_PAGESIZE returns a positive size.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    }

    /// Reserve `total` bytes of inaccessible virtual address space (no commit; `MAP_NORESERVE` so a
    /// huge reservation costs only VA). Returns the base, or null on failure.
    ///
    /// # Safety
    /// Returns a fresh mapping owned by the caller until [`release`].
    pub(super) unsafe fn reserve(total: usize) -> *mut u8 {
        let p = libc::mmap(
            std::ptr::null_mut(),
            total,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        );
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p as *mut u8
        }
    }

    /// Commit `[base, base+len)` read/write.
    ///
    /// # Safety
    /// `[base, base+len)` must lie within a reservation from [`reserve`].
    pub(super) unsafe fn commit_rw(base: *mut u8, len: usize) {
        let rc = libc::mprotect(base as *mut c_void, len, libc::PROT_READ | libc::PROT_WRITE);
        assert!(rc == 0, "svm-jit: window commit failed");
    }

    /// Set the protection of a committed range `[base, base+len)`.
    ///
    /// # Safety
    /// `[base, base+len)` must be a committed sub-range of a window reservation.
    pub(super) unsafe fn protect(base: *mut u8, len: usize, prot: Prot) {
        let p = match prot {
            Prot::Rw => libc::PROT_READ | libc::PROT_WRITE,
            Prot::Ro => libc::PROT_READ,
        };
        libc::mprotect(base as *mut c_void, len, p);
    }

    /// Release a whole reservation from [`reserve`].
    ///
    /// # Safety
    /// `base`/`total` must be exactly a mapping returned by [`reserve`].
    pub(super) unsafe fn release(base: *mut u8, total: usize) {
        libc::munmap(base as *mut c_void, total);
    }

    static INSTALL: Once = Once::new();

    pub(super) fn install_guard() {
        // SAFETY: installs the process-wide SIGSEGV/SIGBUS handler exactly once.
        INSTALL.call_once(|| unsafe { svm_install_trap_handler() });
    }

    /// Run `f` under the installed handler; `true` if an in-`[lo,hi)` fault was caught.
    ///
    /// # Safety
    /// `f` and its pointer args must honour the [`Entry`] contract for the call.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_guarded(
        f: Entry,
        a: *const i64,
        r: *mut i64,
        m: *mut u8,
        t: *const c_void,
        tc: *mut i64,
        lo: usize,
        hi: usize,
    ) -> bool {
        svm_run_guarded(f, a, r, m, t, tc, lo, hi) != 0
    }
}

// ============================ PAL: unsupported targets ========================================
// The escape guarantee (and the guard the §4 elision leans on) needs a guard-page + fault-recovery
// PAL. unix has one (above); **windows** (`VirtualAlloc` + `VirtualProtect(PAGE_NOACCESS)` + a
// Vectored Exception Handler) is the Phase-3.5 leg. Any *other* target (no-MMU / wasm) is
// unsupported and refuses to build rather than weaken the guarantee.
#[cfg(not(unix))]
mod pal {
    use super::{Entry, Prot};
    use core::ffi::c_void;

    compile_error!(
        "svm-jit needs a guard-page Platform Abstraction Layer: unix (mmap + mprotect(PROT_NONE) \
         + a SIGSEGV/SIGBUS handler) exists; windows (VirtualAlloc + VirtualProtect(PAGE_NOACCESS) \
         + a Vectored Exception Handler) is the Phase-3.5 leg and not built yet; other targets \
         (no-MMU / wasm) are unsupported. See DESIGN.md §4 \"Platform support\"."
    );

    // Stubs so the portable code's references resolve to that single, clear error rather than a
    // pile of "unresolved name" follow-ons.
    pub(super) fn page_size() -> usize {
        4096
    }
    pub(super) unsafe fn reserve(_total: usize) -> *mut u8 {
        core::ptr::null_mut()
    }
    pub(super) unsafe fn commit_rw(_base: *mut u8, _len: usize) {}
    pub(super) unsafe fn protect(_base: *mut u8, _len: usize, _prot: Prot) {}
    pub(super) unsafe fn release(_base: *mut u8, _total: usize) {}
    pub(super) fn install_guard() {}
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_guarded(
        _f: Entry,
        _a: *const i64,
        _r: *mut i64,
        _m: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
        _lo: usize,
        _hi: usize,
    ) -> bool {
        false
    }
}

// ============================ PAL conformance test ============================================
// Platform-agnostic: it drives the window model + the guard PAL directly (no JIT), so the *same*
// test validates each platform's PAL — unix now, windows on CI once that leg lands.
#[cfg(test)]
mod tests {
    use super::*;

    // Entry-shaped probes (the trampoline ABI). They do a single volatile read at a fixed window
    // offset; if it faults inside the guarded range the guard unwinds and `run_guarded` reports it.
    extern "C" fn read_at_0(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // SAFETY: offset 0 is in the committed prefix for the test's window.
        unsafe {
            let _ = core::ptr::read_volatile(mem);
        }
    }
    extern "C" fn read_in_tail(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // 512 KiB lands in the reserved-but-inaccessible tail of the test's window → faults.
        // SAFETY: the read is expected to fault; the guard catches it (or the test fails loudly).
        unsafe {
            let _ = core::ptr::read_volatile(mem.add(512 << 10));
        }
    }

    fn guarded(
        win: &GuestWindow,
        f: extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64),
    ) -> bool {
        let mut tc = 0i64;
        // SAFETY: `f` honours the Entry ABI; `mem_base` is this window's base.
        unsafe {
            run_guarded(
                win,
                f as *const u8,
                std::ptr::null(),
                std::ptr::null_mut(),
                win.base(),
                std::ptr::null(),
                &mut tc,
            )
        }
    }

    #[test]
    fn pal_guard_catches_tail_fault_not_in_window() {
        // 64 KiB committed inside a 1 MiB reservation: offset 0 is live, 512 KiB is in the tail.
        let win = GuestWindow::new(64 << 10, 1 << 20);
        assert!(
            !guarded(&win, read_at_0),
            "an in-window read must complete without a guard fault"
        );
        assert!(
            guarded(&win, read_in_tail),
            "a read in the reserved-but-inaccessible tail must be caught by the guard"
        );
    }

    #[test]
    fn pal_window_is_writable_and_released() {
        let mut win = GuestWindow::new(8 << 10, 64 << 10);
        let w = win.rw_mut();
        w[0] = 0xAB;
        w[(8 << 10) - 1] = 0xCD;
        assert_eq!(win.rw_mut()[0], 0xAB);
        assert_eq!(win.rw_mut()[(8 << 10) - 1], 0xCD);
        // dropped here — `release` must not fault.
    }
}
