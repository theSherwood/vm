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
//! slice) is portable and shared. There is a **unix** PAL (`mmap` + `mprotect(PROT_NONE)` + a
//! SIGSEGV/SIGBUS handler via `trap_shim.c`) and a **windows** PAL (`VirtualAlloc2` placeholder
//! reservation + `VirtualProtect(PAGE_NOACCESS)` + a Vectored Exception Handler). The confinement
//! arithmetic is identical on every target, so the interpreter↔JIT differential (§18) is the
//! cross-platform conformance oracle.

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

    /// Commit + read the low `snap` bytes of the window for the escape-oracle snapshot — including
    /// reserved-**tail** pages the guest grew (or left `unmap`-ed / RO) via the Memory cap. `commit_rw`
    /// makes an uncommitted reserved page readable-as-zero and re-asserts RW on a `NOACCESS`/RO page,
    /// so the host read can't fault outside the guarded call. `snap` is clamped to the reservation
    /// (excluding the trailing guard page). Used only by the `_with_host` capture; the common path
    /// reads `[0, mapped)` directly.
    pub(crate) fn read_low(&self, snap: usize) -> Vec<u8> {
        if self.mapped == 0 || snap == 0 {
            return Vec::new();
        }
        let max = self.total - pal::page_size(); // everything but the trailing guard page
        let snap = snap.min(max);
        let commit = round_up(snap, pal::page_size()).min(max);
        // SAFETY: `[base, base+commit)` lies in the reservation (≤ total − guard); `commit_rw` makes
        // it committed + RW, so the subsequent `[0, snap)` read is in-bounds and cannot fault.
        unsafe {
            pal::commit_rw(self.base, commit);
            std::slice::from_raw_parts(self.base, snap).to_vec()
        }
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

/// Windows-only: make `[base, base+len)` committed read/write across this window's **placeholder**
/// reservation — idempotent (see the windows `pal::commit_rw`). Exposed so `svm-run`'s production
/// `MprotectWindow` Memory-cap backend, which operates on this *same* window, can commit/grow tail
/// pages without re-implementing the placeholder split/replace dance (a plain `VirtualAlloc(
/// MEM_COMMIT)` fails on a placeholder). On unix the equivalent is a plain `mprotect`, done inline.
///
/// # Safety
/// `[base, base+len)` must lie within the JIT guest-window reservation that produced `base`.
#[cfg(windows)]
pub unsafe fn win_commit_rw(base: *mut u8, len: usize) {
    pal::commit_rw(base, len)
}

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

// ================================= PAL: windows ===============================================
// `VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)` reservation + `VirtualProtect` (PAGE_NOACCESS tail/guard)
// + a **Vectored Exception Handler** that, on an in-window access violation, restores a captured
// context to unwind out of the fault — the windows analogue of unix's signal + siglongjmp. A
// **placeholder** reservation (rather than a plain `MEM_RESERVE`) is what lets `svm-run`'s §13
// `SharedRegion` path alias a shared section into a fixed window sub-range via
// `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` (issue #1). Pure Rust via `windows-sys` (no C shim), so
// the path stays `cargo check/clippy --target *-windows-*`-able on a non-windows host.
//
// Validated locally by cross-compiling to `x86_64-pc-windows-msvc` (`cargo-xwin`) and running the
// suite under **wine** (the PAL conformance test + the interp↔JIT differential, incl. §13
// `SharedRegion` aliasing); the gating runtime check remains the `cross-os` `windows-latest` CI run.
#[cfg(windows)]
mod pal {
    use super::{Entry, Prot};
    use core::cell::Cell;
    use core::ffi::c_void;
    use std::sync::Once;
    use windows_sys::Win32::Foundation::{GetLastError, HANDLE};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, RtlCaptureContext, CONTEXT, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
        MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MEM_COMMIT, MEM_MAPPED,
        MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
        MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY,
        PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `GetLastError()` as an `i64`, for thread-into-panic-messages debuggability (CI has no debugger,
    /// so a failing Win32 call names its error code in the test output).
    fn last_error() -> i64 {
        // SAFETY: GetLastError reads thread-local state; always safe.
        unsafe { GetLastError() as i64 }
    }

    // VEH return codes + the access-violation status (kept local to avoid version-specific paths).
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const STATUS_ACCESS_VIOLATION: i32 = 0xC000_0005u32 as i32;

    pub(super) fn page_size() -> usize {
        // SAFETY: GetSystemInfo only writes the out-param; always safe.
        let mut si: SYSTEM_INFO = unsafe { core::mem::zeroed() };
        unsafe { GetSystemInfo(&mut si) };
        let p = si.dwPageSize as usize;
        if p == 0 {
            4096
        } else {
            p
        }
    }

    /// Reserve `total` bytes of inaccessible VA as a **placeholder** (`VirtualAlloc2` with
    /// `MEM_RESERVE_PLACEHOLDER`). A placeholder — rather than a plain `MEM_RESERVE` — is what lets a
    /// later `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` alias a shared section into a *fixed* sub-range
    /// of this window (the §13 `SharedRegion` path in `svm-run`); a plain reservation cannot host a
    /// fixed-address view. The reservation costs only address space until a sub-range is committed.
    ///
    /// # Safety
    /// Returns a fresh reservation owned by the caller until [`release`].
    pub(super) unsafe fn reserve(total: usize) -> *mut u8 {
        VirtualAlloc2(
            0 as HANDLE,
            core::ptr::null(),
            total,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            core::ptr::null_mut(),
            0,
        ) as *mut u8
    }

    /// Make `[base, base+len)` committed + read/write — **idempotently**, across a placeholder
    /// reservation. A *placeholder* sub-range is split out (`MEM_PRESERVE_PLACEHOLDER`) to its exact
    /// bounds and replace-committed (`MEM_REPLACE_PLACEHOLDER`, RW, OS-zero-filled); an already-
    /// *committed* sub-range (private memory or a mapped §13 view) is merely re-asserted read/write
    /// with `VirtualProtect`, so its live contents are **never re-zeroed**. Walking the region map
    /// with `VirtualQuery` is what makes this safe to call on overlapping / growing ranges — the
    /// snapshot path ([`read_low`](super::GuestWindow::read_low)) re-commits a prefix that is already
    /// live, and a plain `VirtualAlloc(MEM_COMMIT)` cannot commit a placeholder at all (it fails).
    ///
    /// # Safety
    /// `[base, base+len)` must lie within a reservation from [`reserve`].
    pub(super) unsafe fn commit_rw(base: *mut u8, len: usize) {
        let end = base as usize + len;
        let mut addr = base as usize;
        while addr < end {
            let mut mbi: MEMORY_BASIC_INFORMATION = core::mem::zeroed();
            let n = VirtualQuery(
                addr as *const c_void,
                &mut mbi,
                core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            );
            assert!(
                n != 0,
                "svm-jit: VirtualQuery failed (err {})",
                last_error()
            );
            let region_base = mbi.BaseAddress as usize;
            let region_end = region_base + mbi.RegionSize;
            let a = addr.max(region_base);
            let b = end.min(region_end);
            if mbi.State == MEM_COMMIT {
                // Already committed (private or a mapped view): re-assert RW, keep contents.
                let mut old: PAGE_PROTECTION_FLAGS = 0;
                VirtualProtect((a as *const c_void) as _, b - a, PAGE_READWRITE, &mut old);
            } else {
                // Placeholder (`MEM_RESERVE`): carve the exact sub-range, then replace-commit it RW.
                if a != region_base || b != region_end {
                    let ok = VirtualFree(
                        a as *mut c_void,
                        b - a,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    );
                    assert!(
                        ok != 0,
                        "svm-jit: placeholder split failed (err {})",
                        last_error()
                    );
                }
                let p = VirtualAlloc2(
                    0 as HANDLE,
                    a as *const c_void,
                    b - a,
                    MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    core::ptr::null_mut(),
                    0,
                );
                assert!(
                    !p.is_null(),
                    "svm-jit: window commit failed (err {})",
                    last_error()
                );
            }
            addr = b;
        }
    }

    /// Set the protection of a committed range `[base, base+len)`.
    ///
    /// # Safety
    /// `[base, base+len)` must be a committed sub-range of a window reservation.
    pub(super) unsafe fn protect(base: *mut u8, len: usize, prot: Prot) {
        let flags: PAGE_PROTECTION_FLAGS = match prot {
            Prot::Rw => PAGE_READWRITE,
            Prot::Ro => PAGE_READONLY,
        };
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        VirtualProtect(base as *const c_void, len, flags, &mut old);
    }

    /// Release a whole reservation from [`reserve`] — **every fragment of it**. After [`commit_rw`]
    /// (and the §13 `map_region` view path in `svm-run`) have split the original placeholder into
    /// independent sub-allocations — committed-private regions, leftover placeholders, mapped views —
    /// a single `VirtualFree(base, 0, MEM_RELEASE)` frees only the *first* fragment and **leaks the
    /// rest**: grown-tail committed pages accumulate as commit charge across many window teardowns
    /// until an allocation fails (the intermittent `windows-latest` `jit_fuzz` OOM/fastfail). So walk
    /// `[base, base+total)` with `VirtualQuery` and tear down each region — unmap shared-section views,
    /// release placeholders / private commits. Defensive: teardown must never fault, so individual
    /// Win32 failures are ignored (the address space is reclaimed wholesale at process exit anyway).
    ///
    /// # Safety
    /// `base`/`total` must be exactly a reservation returned by [`reserve`].
    pub(super) unsafe fn release(base: *mut u8, total: usize) {
        let end = base as usize + total;
        let mut addr = base as usize;
        let proc = GetCurrentProcess();
        while addr < end {
            let mut mbi: MEMORY_BASIC_INFORMATION = core::mem::zeroed();
            if VirtualQuery(
                addr as *const c_void,
                &mut mbi,
                core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            ) == 0
            {
                break;
            }
            let region_base = mbi.BaseAddress as usize;
            let next = region_base + mbi.RegionSize;
            if mbi.Type == MEM_MAPPED {
                // A §13 shared-section view: unmap it back to a placeholder, then release that.
                UnmapViewOfFile2(
                    proc,
                    MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: region_base as *mut c_void,
                    },
                    MEM_PRESERVE_PLACEHOLDER,
                );
                VirtualFree(region_base as *mut c_void, 0, MEM_RELEASE);
            } else if mbi.State == MEM_COMMIT || mbi.State == MEM_RESERVE {
                // A committed-private region or a leftover placeholder: free the whole fragment.
                VirtualFree(mbi.AllocationBase, 0, MEM_RELEASE);
            }
            addr = if next > addr {
                next
            } else {
                addr + page_size()
            };
        }
    }

    // ---- the guard: AddVectoredExceptionHandler + RtlCaptureContext (longjmp-equivalent) --------
    // `CONTEXT` must be **16-byte aligned** on x86-64: it embeds XMM (`M128A`) save area that
    // `RtlCaptureContext` writes with *aligned* SSE stores (`movaps`). The real Win32 header declares
    // it `__declspec(align(16))`, but windows-sys types it `#[repr(C)]` only — so a bare stack local
    // is merely 8-byte aligned, and when it lands at an 8-mod-16 address `RtlCaptureContext` faults
    // (`STATUS_ACCESS_VIOLATION`) *inside the capture itself*, before the guest runs. Wrap it to
    // restore the ABI-required alignment.
    #[repr(C, align(16))]
    struct AlignedContext(CONTEXT);

    #[derive(Clone, Copy)]
    struct Frame {
        ctx: *const CONTEXT, // captured recovery context (a stack local of `run_guarded`)
        lo: usize,
        hi: usize,
    }
    thread_local! {
        // The active guarded call's window range + recovery context (None ⇒ no guarded call).
        static GUARD: Cell<Option<Frame>> = const { Cell::new(None) };
        // Set by the VEH before it restores the context, read after RtlCaptureContext returns.
        static TRIPPED: Cell<bool> = const { Cell::new(false) };
    }

    unsafe extern "system" fn veh(ep: *mut EXCEPTION_POINTERS) -> i32 {
        let ep = &*ep;
        let rec = &*ep.ExceptionRecord;
        if rec.ExceptionCode == STATUS_ACCESS_VIOLATION {
            // ExceptionInformation[1] is the faulting address for an access violation.
            let addr = rec.ExceptionInformation[1];
            if let Some(f) = GUARD.with(|g| g.get()) {
                if addr >= f.lo && addr < f.hi {
                    TRIPPED.with(|t| t.set(true));
                    // Restore the captured context → resume right after RtlCaptureContext in
                    // `run_guarded` (the unix siglongjmp analogue). The abandoned JIT frames hold no
                    // Rust destructors.
                    core::ptr::copy_nonoverlapping(f.ctx, ep.ContextRecord, 1);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    static INSTALL: Once = Once::new();
    pub(super) fn install_guard() {
        INSTALL.call_once(|| {
            // first = 1 → our handler runs before any previously-registered one.
            let h = unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
            assert!(!h.is_null(), "svm-jit: AddVectoredExceptionHandler failed");
        });
    }

    /// Run `f` under the handler; `true` if an in-`[lo,hi)` access violation was caught.
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
        let mut saved = AlignedContext(core::mem::zeroed());
        // Capture the recovery point. On a guard fault the VEH copies `saved` over the fault context,
        // so execution resumes *here* with TRIPPED set — the longjmp-equivalent return.
        RtlCaptureContext(&mut saved.0);
        if TRIPPED.with(|x| x.replace(false)) {
            GUARD.with(|g| g.set(None));
            return true;
        }
        GUARD.with(|g| {
            g.set(Some(Frame {
                ctx: &saved.0,
                lo,
                hi,
            }))
        });
        f(a, r, m, t, tc);
        GUARD.with(|g| g.set(None));
        false
    }
}

// ============================ PAL: unsupported targets ========================================
// The escape guarantee (and the guard the §4 elision leans on) needs a guard-page + fault-recovery
// PAL. unix and windows have one (above); any *other* target (no-MMU / wasm) is unsupported and
// refuses to build rather than weaken the guarantee.
#[cfg(not(any(unix, windows)))]
mod pal {
    use super::{Entry, Prot};
    use core::ffi::c_void;

    compile_error!(
        "svm-jit needs a guard-page Platform Abstraction Layer: unix (mmap + mprotect(PROT_NONE) + \
         a SIGSEGV/SIGBUS handler) and windows (VirtualAlloc + VirtualProtect + a Vectored \
         Exception Handler) exist; other targets (no-MMU / wasm) are unsupported. See DESIGN.md §4 \
         \"Platform support\"."
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
