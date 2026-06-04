//! Guest-window allocation + detect-and-kill trap recovery (`DESIGN.md` §4/§5).
//!
//! On unix the window is `mmap`'d with a trailing **`PROT_NONE` guard page**, and the JIT
//! entry runs under a SIGSEGV/SIGBUS handler (see `trap_shim.c`): a fault inside the
//! window's guarded range unwinds back out of the call as a [`TrapKind::MemoryFault`]
//! instead of corrupting the host (§5 detect-and-kill). The masking lowering already
//! confines every access to `[0, size)`, so this fires on a width-overrun at the very top
//! of the window or — defense-in-depth — a masking/elision bug the guard caught.
//!
//! Non-unix hosts (Windows, no-MMU) are **not supported yet** and the crate refuses to build
//! there (`compile_error!`, below). Without the guard we cannot make the same escape
//! guarantee, and guard-*relying* mask elision (§4 "guard-when-bounded") would be real
//! host-memory UB. The Windows equivalent — `VirtualAlloc(MEM_RESERVE/COMMIT)` +
//! `VirtualProtect(PAGE_NOACCESS)` + a Vectored Exception Handler — is TODO; the goal is to
//! run *identically* on Windows once it lands (see `DESIGN.md` §4 "Platform support").

use crate::TrapKind;
use core::ffi::c_void;

/// The compiled entry trampoline ABI (see `build_trampoline`): `(args, results, mem_base,
/// fn_table_base, trap_out)`. The 4th pointer is opaque here (`FnEntry*` to the JIT).
type Entry = extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64);

// ---- unix: mmap'd window + PROT_NONE guard page + signal-based recovery ----------------
#[cfg(unix)]
mod imp {
    use super::*;
    use std::sync::Once;

    extern "C" {
        fn svm_install_trap_handler();
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

    fn page_size() -> usize {
        // SAFETY: sysconf is always safe to call; _SC_PAGESIZE returns a positive size.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    }

    fn round_up(n: usize, align: usize) -> usize {
        (n + align - 1) & !(align - 1)
    }

    /// An `mmap`'d guest window: `rw` readable/writable bytes (≥ `size`, page-rounded)
    /// followed by one `PROT_NONE` guard page. `size` is the logical window `[0, size)`.
    pub(crate) struct GuestWindow {
        base: *mut u8,
        size: usize,
        total: usize, // rw + guard page (the full mapping length)
    }

    impl GuestWindow {
        pub(crate) fn new(size: usize) -> GuestWindow {
            if size == 0 {
                return GuestWindow {
                    base: std::ptr::null_mut(),
                    size: 0,
                    total: 0,
                };
            }
            let page = page_size();
            let rw = round_up(size, page);
            let total = rw + page; // one guard page after the rw region
                                   // SAFETY: a fresh anonymous mapping; checked against MAP_FAILED below.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    total,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(base != libc::MAP_FAILED, "svm-jit: window mmap failed");
            let base = base as *mut u8;
            // SAFETY: protect the trailing page so any access past `rw` faults.
            let rc = unsafe { libc::mprotect(base.add(rw) as *mut c_void, page, libc::PROT_NONE) };
            assert!(rc == 0, "svm-jit: guard-page mprotect failed");
            GuestWindow { base, size, total }
        }

        /// The logical window `[0, size)`, readable/writable (anonymous mmap is zeroed).
        pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
            if self.size == 0 {
                return &mut [];
            }
            // SAFETY: `[base, base+size)` is mapped RW for the window's lifetime.
            unsafe { std::slice::from_raw_parts_mut(self.base, self.size) }
        }

        pub(crate) fn base(&self) -> *mut u8 {
            self.base
        }

        /// The address range a fault must land in to be attributed to this window (the whole
        /// mapping, so the guard page is covered). `(0, 0)` when there is no window.
        fn fault_range(&self) -> (usize, usize) {
            if self.size == 0 {
                (0, 0)
            } else {
                (self.base as usize, self.base as usize + self.total)
            }
        }
    }

    impl Drop for GuestWindow {
        fn drop(&mut self) {
            if !self.base.is_null() {
                // SAFETY: unmapping exactly the mapping we created.
                unsafe { libc::munmap(self.base as *mut c_void, self.total) };
            }
        }
    }

    static INSTALL: Once = Once::new();

    /// Run the JIT entry `code` under the guard handler. Returns `true` if a fault in the
    /// window's guarded range was caught and unwound (→ the caller reports `MemoryFault`).
    ///
    /// # Safety
    /// `code` must be the finalized trampoline with the [`Entry`] signature, and the
    /// pointers must satisfy its contract (valid for the call, outliving it).
    pub(crate) unsafe fn run_guarded(
        window: &GuestWindow,
        code: *const u8,
        args: *const i64,
        results: *mut i64,
        mem_base: *mut u8,
        fn_table: *const c_void,
        trap_cell: *mut i64,
    ) -> bool {
        INSTALL.call_once(|| svm_install_trap_handler());
        let (lo, hi) = window.fault_range();
        let f: Entry = std::mem::transmute(code);
        svm_run_guarded(f, args, results, mem_base, fn_table, trap_cell, lo, hi) != 0
    }
}

// ---- non-unix: unsupported until the guard path is built (see DESIGN.md §4) ------------
//
// Confinement's escape guarantee (and the guard the elision leans on) is the unix
// `PROT_NONE` guard page + SIGSEGV/SIGBUS handler above. On Windows / other non-unix hosts
// we cannot yet make that guarantee — and guard-relying elision would be host-memory UB
// without the guard — so we refuse to build rather than run with weaker guarantees. The
// equivalent (`VirtualAlloc(MEM_RESERVE/COMMIT)` + `VirtualProtect(PAGE_NOACCESS)` + a
// Vectored Exception Handler) is TODO; the goal is to run *identically* on Windows.
#[cfg(not(unix))]
mod imp {
    use super::*;

    compile_error!(
        "svm-jit supports only unix hosts for now: memory confinement relies on PROT_NONE \
         guard pages + a SIGSEGV/SIGBUS handler (see mem.rs / trap_shim.c), so we refuse to \
         build/run on Windows or other non-unix targets rather than weaken the escape \
         guarantee. The equivalent (VirtualAlloc + VirtualProtect(PAGE_NOACCESS) + a Vectored \
         Exception Handler) is not built yet — see DESIGN.md §4 \"Platform support\". \
         Goal: run identically on Windows once that lands."
    );

    // Unreachable: the `compile_error!` above aborts every non-unix build. These stubs exist
    // only so the `pub(crate) use imp::{…}` below resolves to that single, clear error rather
    // than a pile of "unresolved import" follow-on errors.
    pub(crate) struct GuestWindow;
    impl GuestWindow {
        pub(crate) fn new(_size: usize) -> GuestWindow {
            GuestWindow
        }
        pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
            &mut []
        }
        pub(crate) fn base(&self) -> *mut u8 {
            core::ptr::null_mut()
        }
    }

    pub(crate) unsafe fn run_guarded(
        _window: &GuestWindow,
        _code: *const u8,
        _args: *const i64,
        _results: *mut i64,
        _mem_base: *mut u8,
        _fn_table: *const c_void,
        _trap_cell: *mut i64,
    ) -> bool {
        false
    }
}

pub(crate) use imp::{run_guarded, GuestWindow};

/// The trap code a caught guard fault reports.
pub(crate) const FAULT_TRAP: i64 = TrapKind::MemoryFault as i64;
