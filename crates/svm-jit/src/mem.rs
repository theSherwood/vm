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

    /// An `mmap`'d guest window: `mapped` readable/writable bytes (the backed prefix,
    /// `[0, mapped)`) inside a larger `reserved` virtual range whose tail `[mapped, reserved)`
    /// plus one trailing `PROT_NONE` guard page are unmapped. Confinement masks every address
    /// into `[0, reserved)` (the JIT mask const = `reserved − 1`), so an access lands either in
    /// the backed prefix or in the unmapped tail/guard — where it faults (§4/§5 detect-and-kill).
    /// A fully-mapped window is just `reserved == mapped` (the historical single-extent case).
    pub(crate) struct GuestWindow {
        base: *mut u8,
        mapped: usize, // backed RW bytes `[0, mapped)` (the logical window the guest declared)
        total: usize,  // full mapping length: reserved (page-rounded) + one guard page
    }

    impl GuestWindow {
        /// Reserve `reserved` bytes (page-rounded) + a guard page as `PROT_NONE`, then make the
        /// `mapped` backed prefix readable/writable. `reserved` is raised to at least `mapped`.
        ///
        /// For the unmapped tail's fault boundary to agree with the interpreter's byte-exact
        /// `mapped` bound, `mapped` must be page-aligned whenever `reserved > mapped` — true for
        /// any `size_log2 >= 12`, which every caller of the decoupled form satisfies.
        pub(crate) fn new(mapped: usize, reserved: usize) -> GuestWindow {
            if mapped == 0 {
                return GuestWindow {
                    base: std::ptr::null_mut(),
                    mapped: 0,
                    total: 0,
                };
            }
            let page = page_size();
            let reserved = reserved.max(mapped);
            let rw = round_up(mapped, page);
            let total = round_up(reserved, page) + page; // reserved + one guard page
                                                         // SAFETY: a fresh anonymous reservation; MAP_NORESERVE so a huge `reserved` costs
                                                         // only virtual address space (no commit) until pages are touched. Checked below.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    total,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert!(base != libc::MAP_FAILED, "svm-jit: window mmap failed");
            let base = base as *mut u8;
            // SAFETY: make the backed prefix `[0, rw)` readable/writable; the tail + guard stay
            // PROT_NONE so any access past `mapped` faults.
            let rc = unsafe {
                libc::mprotect(base as *mut c_void, rw, libc::PROT_READ | libc::PROT_WRITE)
            };
            assert!(rc == 0, "svm-jit: window mprotect failed");
            GuestWindow {
                base,
                mapped,
                total,
            }
        }

        /// The logical (backed) window `[0, mapped)`, readable/writable (anonymous mmap is
        /// zeroed). Bytes in the reserved-but-unmapped tail are not part of this slice.
        pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
            if self.mapped == 0 {
                return &mut [];
            }
            // SAFETY: `[base, base+mapped)` is mapped RW for the window's lifetime.
            unsafe { std::slice::from_raw_parts_mut(self.base, self.mapped) }
        }

        pub(crate) fn base(&self) -> *mut u8 {
            self.base
        }

        /// Re-enable read+write on the whole backed region `[0, mapped)`. The guest may have
        /// changed page protections through the `Memory` capability (`unmap`→`PROT_NONE`,
        /// `protect`→read-only), so a snapshot read of the window could otherwise fault *outside*
        /// the guarded call and crash the host. Idempotent; a no-op cost when nothing changed.
        pub(crate) fn restore_rw(&self) {
            if self.mapped == 0 {
                return;
            }
            let page = page_size();
            let rw = round_up(self.mapped, page);
            // SAFETY: `[base, base+rw)` is the window's backed region, owned for its lifetime.
            unsafe {
                libc::mprotect(
                    self.base as *mut c_void,
                    rw,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
        }

        /// Map the whole pages touched by `[offset, offset+len)` **read-only** — the D40 const
        /// data segment (§3a/§4): a later guest write to them faults into the guarded range. The
        /// data must already be written (this only changes protection). A producer keeps RO data
        /// on its own pages; protection is page-granular, so a shared page would over-protect.
        pub(crate) fn protect_ro(&self, offset: u64, len: u64) {
            if self.mapped == 0 || len == 0 {
                return;
            }
            let page = page_size();
            let start = (offset as usize / page) * page;
            let end = round_up((offset + len) as usize, page);
            // SAFETY: `[base+start, base+end)` lies within the backed region (the caller bounds
            // `offset+len <= mapped`, which is page-rounded up to `rw`), owned for the lifetime.
            unsafe {
                libc::mprotect(
                    self.base.add(start) as *mut c_void,
                    end - start,
                    libc::PROT_READ,
                )
            };
        }

        /// The address range a fault must land in to be attributed to this window (the whole
        /// reservation, so the unmapped tail + guard page are covered). `(0, 0)` when there is
        /// no window.
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
        pub(crate) fn new(_mapped: usize, _reserved: usize) -> GuestWindow {
            GuestWindow
        }
        pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
            &mut []
        }
        pub(crate) fn base(&self) -> *mut u8 {
            core::ptr::null_mut()
        }
        pub(crate) fn restore_rw(&self) {}
        pub(crate) fn protect_ro(&self, _offset: u64, _len: u64) {}
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
