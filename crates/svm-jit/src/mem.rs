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
    /// Inaccessible — any access faults into the guard (a durable-restored `Unmapped` page).
    None,
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

    /// A `size`-byte window whose pages start **inaccessible** (plus the usual guard page): the §14
    /// demand-paged child window. Every access faults until [`commit_range`](Self::commit_range)
    /// supplies the page (fault-driven yield). `mapped` is the logical size, so confinement and
    /// [`fault_range`](Self::fault_range) see the full window — but `[0, mapped)` is **not**
    /// committed, so `rw_mut`/whole-window reads must not be used; the §14 nesting runtime tracks
    /// committed pages and touches only those.
    #[cfg(fiber_rt)]
    pub(crate) fn new_uncommitted(size: usize) -> GuestWindow {
        if size == 0 {
            return GuestWindow {
                base: std::ptr::null_mut(),
                mapped: 0,
                total: 0,
            };
        }
        let page = pal::page_size();
        let total = round_up(size, page) + page; // window + one guard page, all inaccessible
                                                 // SAFETY: a fresh inaccessible reservation; checked non-null below.
        let base = unsafe { pal::reserve(total) };
        assert!(!base.is_null(), "svm-jit: window reserve failed");
        GuestWindow {
            base,
            mapped: size,
            total,
        }
    }

    /// Commit `[offset, offset+len)` (page-rounded by the PAL) read/write — the §14 lazy-paging
    /// page supply. Freshly committed pages read zero; the nesting runtime then copies the parent's
    /// bytes in. Idempotent on already-committed pages.
    ///
    /// # Safety
    /// `[offset, offset+len)` must lie within the window's reservation (excluding the guard page).
    #[cfg(fiber_rt)]
    pub(crate) unsafe fn commit_range(&self, offset: usize, len: usize) {
        pal::commit_rw(self.base.add(offset), len);
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

    /// Map the whole pages touched by `[offset, offset+len)` **inaccessible** — a durable-restored
    /// `Unmapped` page (DURABILITY.md §12.3): any later access faults into the guarded range,
    /// exactly as on the frozen guest. The snapshot's `read_low` re-commits it RW before reading.
    pub(crate) fn protect_none(&self, offset: u64, len: u64) {
        if self.mapped == 0 || len == 0 {
            return;
        }
        let page = pal::page_size();
        let start = (offset as usize / page) * page;
        let end = round_up((offset + len) as usize, page);
        // SAFETY: as `protect_ro` — `[base+start, base+end)` is within the backed region.
        unsafe { pal::protect(self.base.add(start), end - start, Prot::None) };
    }

    /// The address range a fault must land in to be attributed to this window (the whole
    /// reservation, so the inaccessible tail + guard page are covered). `(0, 0)` when there is no
    /// window.
    pub(crate) fn fault_range(&self) -> (usize, usize) {
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

/// Install the guard on the calling (worker) thread. Idempotent; the handler is process-wide but its
/// recovery state is thread-local, so each worker arms independently.
#[cfg(fiber_rt)]
pub(crate) fn install_guard() {
    pal::install_guard();
}

/// The trap stack the guard handler captured at the most recent caught memory fault on this thread,
/// read-and-cleared (§5 W3 trap-time backtrace): the faulting `pc` and the frame-pointer chain's raw
/// return addresses (the handler walked them while the guest stack was intact). `None` if nothing
/// was captured (no fault since the last take, or an arch the handler doesn't decode). The host
/// symbolizes them via [`CompiledModule::trap_backtrace`].
pub(crate) fn take_trap_frame() -> Option<(usize, Vec<usize>)> {
    pal::take_trap_frame()
}

/// Run an `Entry`-shaped `code` with faults in `[lo, hi)` caught (detect-and-kill), for a window
/// fault range obtained from [`GuestWindow::fault_range`]. Used to run a fiber resume on a worker (a
/// guest memory fault inside the fiber unwinds back here — the fiber stack is abandoned, the domain is
/// being killed) and by `CompiledModule::invoke_extra` to nest a recovery inside an in-flight run
/// (re-entrant, like the §14 child path). Returns `true` if a guarded fault was caught.
///
/// # Safety
/// `code` must honour the [`Entry`] ABI and its pointer args must be valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_guarded_range(
    code: *const u8,
    args: *const i64,
    results: *mut i64,
    mem_base: *mut u8,
    fn_table: *const c_void,
    trap_cell: *mut i64,
    lo: usize,
    hi: usize,
) -> bool {
    let f: Entry = std::mem::transmute(code);
    pal::run_guarded(f, args, results, mem_base, fn_table, trap_cell, lo, hi)
}

/// The trap code a caught guard fault reports.
pub(crate) const FAULT_TRAP: i64 = TrapKind::MemoryFault as i64;

/// The host page size — the §14 demand-paging granularity (what one fault supplies).
#[cfg(fiber_rt)]
pub(crate) fn page_size() -> usize {
    pal::page_size()
}

/// A snapshot of this thread's **detect-and-kill recovery state** (the armed window range + recovery
/// point), for §14 co-fibers: a coroutine child arms its own guard on its own fiber stack, and a
/// suspend switches away mid-call — so the parent swaps the whole recovery state around every switch
/// (save the child's at suspend-return, restore its own; reinstall the child's at the next resume).
/// A fresh snapshot is *disarmed*. Same-thread only: the captured recovery point (sigjmp_buf /
/// CONTEXT pointer) is only meaningful on the thread that captured it — the §14 nesting runtime
/// enforces same-thread resume.
#[cfg(fiber_rt)]
pub(crate) struct GuardState(guard_imp::State);

#[cfg(fiber_rt)]
impl GuardState {
    /// A fresh, disarmed state (restoring it disables fault recovery until something re-arms).
    pub(crate) fn new() -> GuardState {
        GuardState(guard_imp::new())
    }
}

/// Capture this thread's current recovery state into `s`.
#[cfg(fiber_rt)]
pub(crate) fn guard_save(s: &mut GuardState) {
    guard_imp::save(&mut s.0)
}

/// Install `s` as this thread's recovery state.
#[cfg(fiber_rt)]
pub(crate) fn guard_restore(s: &GuardState) {
    guard_imp::restore(&s.0)
}

// unix: the state lives in the C shim (sigjmp_buf has C-side size/alignment), as an opaque
// calloc'd blob (all-zero ⇒ disarmed).
#[cfg(all(unix, fiber_rt))]
mod guard_imp {
    use core::ffi::c_void;
    extern "C" {
        fn svm_guard_box() -> *mut c_void;
        fn svm_guard_unbox(p: *mut c_void);
        fn svm_guard_save(p: *mut c_void);
        fn svm_guard_restore(p: *const c_void);
    }
    pub(super) struct State(*mut c_void);
    pub(super) fn new() -> State {
        // SAFETY: allocates a zeroed C blob; checked non-null.
        let p = unsafe { svm_guard_box() };
        assert!(!p.is_null(), "svm-jit: guard-state allocation failed");
        State(p)
    }
    pub(super) fn save(s: &mut State) {
        // SAFETY: `s.0` is a live blob from `svm_guard_box`.
        unsafe { svm_guard_save(s.0) }
    }
    pub(super) fn restore(s: &State) {
        // SAFETY: as above; restoring only writes this thread's recovery thread-locals.
        unsafe { svm_guard_restore(s.0) }
    }
    impl Drop for State {
        fn drop(&mut self) {
            // SAFETY: exactly the blob `svm_guard_box` returned, freed once.
            unsafe { svm_guard_unbox(self.0) }
        }
    }
}

// windows: the state is the thread-local VEH guard frame (the recovery CONTEXT pointer + range).
#[cfg(all(windows, fiber_rt))]
mod guard_imp {
    pub(super) struct State(super::pal::GuardSnap);
    pub(super) fn new() -> State {
        State(super::pal::GuardSnap::disarmed())
    }
    pub(super) fn save(s: &mut State) {
        s.0 = super::pal::guard_snapshot();
    }
    pub(super) fn restore(s: &State) {
        super::pal::guard_install(&s.0)
    }
}

/// A §14 **demand-fault** callback: called by the fault handler (signal/VEH context!) for a fault
/// inside the registered demand range. Return nonzero to re-execute the faulting access (the
/// callback suspended to the parent, which supplied the page), zero to fall through to
/// detect-and-kill. Must be async-signal-safe up to the stack switch (touch only the given ctx).
#[cfg(fiber_rt)]
pub(crate) type DemandCb = unsafe extern "C" fn(addr: usize, ctx: *mut c_void) -> i32;

/// Register this thread's §14 demand range `[lo, hi)` + callback — the *recoverable* fault window of
/// the demand-paged coroutine child about to run. One registration per thread (coroutines don't
/// nest); the caller pairs it with [`clear_demand`] around every switch into the child.
///
/// # Safety
/// `cb`/`ctx` must stay valid until [`clear_demand`]; the handler will call them from fault context.
#[cfg(fiber_rt)]
pub(crate) unsafe fn set_demand(lo: usize, hi: usize, cb: DemandCb, ctx: *mut c_void) {
    demand_imp::set(lo, hi, cb, ctx)
}

/// Clear this thread's §14 demand registration (faults in the range become detect-and-kill again).
#[cfg(fiber_rt)]
pub(crate) fn clear_demand() {
    demand_imp::clear()
}

#[cfg(all(unix, fiber_rt))]
mod demand_imp {
    use core::ffi::c_void;
    extern "C" {
        fn svm_set_demand(lo: usize, hi: usize, cb: super::DemandCb, ctx: *mut c_void);
        fn svm_clear_demand();
    }
    pub(super) unsafe fn set(lo: usize, hi: usize, cb: super::DemandCb, ctx: *mut c_void) {
        svm_set_demand(lo, hi, cb, ctx)
    }
    pub(super) fn clear() {
        // SAFETY: clears this thread's registration thread-locals; always safe.
        unsafe { svm_clear_demand() }
    }
}

#[cfg(all(windows, fiber_rt))]
mod demand_imp {
    use core::ffi::c_void;
    pub(super) unsafe fn set(lo: usize, hi: usize, cb: super::DemandCb, ctx: *mut c_void) {
        super::pal::DEMAND.with(|d| d.set(Some((lo, hi, cb, ctx))));
    }
    pub(super) fn clear() {
        super::pal::DEMAND.with(|d| d.set(None));
    }
}

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
        fn svm_take_trap_frame(pc: *mut usize, rets: *mut usize, max: i32) -> i32;
    }

    /// Read and clear the trap stack the SIGSEGV/SIGBUS handler captured at the most recent caught
    /// guard fault (§5 W3 trap-time backtrace): the faulting `pc` and the frame-pointer chain's raw
    /// return addresses. `None` if none was captured. Must match `SVM_TRAP_MAXFRAMES` in the shim.
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>)> {
        const MAX: usize = 64;
        let mut pc = 0usize;
        let mut rets = [0usize; MAX];
        // SAFETY: reads+clears the shim's thread-locals into `pc` + the `MAX`-slot `rets` buffer.
        let n = unsafe { svm_take_trap_frame(&mut pc, rets.as_mut_ptr(), MAX as i32) };
        (n >= 0).then(|| (pc, rets[..n as usize].to_vec()))
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
            Prot::None => libc::PROT_NONE,
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

    /// `ERROR_COMMITMENT_LIMIT` — the system-wide commit charge (RAM + page file) is momentarily
    /// exhausted. Unlike unix `mmap` (overcommit), Windows charges every committed page up front, so a
    /// tight CI page file under churn (e.g. the 4000-window fuzz loop) can transiently hit this. It is
    /// **transient** (clears as other allocations free / the page file grows), so the commit path
    /// retries it with backoff rather than failing the run.
    const ERROR_COMMITMENT_LIMIT: i64 = 1455;
    /// Bounded retries on a transient commit-limit (backoff 5·2^k ms ⇒ ~0.3 s worst case, then give up
    /// loudly). Enough to ride out a momentary spike without ever hanging on a genuine exhaustion.
    const COMMIT_RETRIES: u32 = 6;

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
                // Replace-commit the placeholder sub-range RW. `MEM_REPLACE_PLACEHOLDER` is atomic
                // (success → ptr, failure → null leaving the placeholder intact), so retrying a
                // transient `ERROR_COMMITMENT_LIMIT` is safe.
                let mut p = core::ptr::null_mut();
                for attempt in 0..COMMIT_RETRIES {
                    p = VirtualAlloc2(
                        0 as HANDLE,
                        a as *const c_void,
                        b - a,
                        MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                        PAGE_READWRITE,
                        core::ptr::null_mut(),
                        0,
                    );
                    if !p.is_null() || last_error() != ERROR_COMMITMENT_LIMIT {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5u64 << attempt));
                }
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
            Prot::None => PAGE_NOACCESS,
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
        // §5 W3 trap-time backtrace: the faulting `(pc, frame-pointer-chain return addresses)` the VEH
        // captures from the access-violation `CONTEXT` *before* it overwrites that context with the
        // recovery one. Read + cleared by `take_trap_frame`. A fixed buffer (no allocation in the VEH).
        static TRAP_VALID: Cell<bool> = const { Cell::new(false) };
        static TRAP_PC: Cell<usize> = const { Cell::new(0) };
        static TRAP_RETS: Cell<[usize; TRAP_MAXFRAMES]> = const { Cell::new([0; TRAP_MAXFRAMES]) };
        static TRAP_N: Cell<usize> = const { Cell::new(0) };
    }

    /// Max trap-backtrace frames captured (matches the unix shim's `SVM_TRAP_MAXFRAMES`).
    const TRAP_MAXFRAMES: usize = 64;

    /// Walk the frame-pointer chain from `fp` into `out`, returning the count — the Rust analog of the
    /// unix shim's `svm_walk_fp_chain` (DEBUGGING.md §5/W3). The JIT's `preserve_frame_pointers` gives
    /// every guest frame a `{ saved_fp, ret_addr }` record (`*fp` = caller's saved fp, `*(fp+1)` = its
    /// return address). Bounded — aligned, non-null, strictly-increasing links, a span backstop, the
    /// frame cap — so a corrupt chain terminates instead of looping or reading wild memory.
    ///
    /// # Safety
    /// `fp` must be a live frame pointer of guest JIT code (the faulting `CONTEXT`'s `Rbp`); reads the
    /// intact-at-fault guest stack. Called only from the VEH, before anything unwinds.
    unsafe fn walk_fp_chain(fp: usize, out: &mut [usize; TRAP_MAXFRAMES]) -> usize {
        const SPAN: usize = 8 * 1024 * 1024; // don't chase a corrupt chain off the stack
        let align = core::mem::size_of::<usize>() - 1;
        let (mut cur, start) = (fp, fp);
        let mut n = 0;
        while n < TRAP_MAXFRAMES
            && cur != 0
            && cur & align == 0
            && cur >= start
            && cur - start < SPAN
        {
            let next = *(cur as *const usize);
            let ret = *((cur + core::mem::size_of::<usize>()) as *const usize);
            out[n] = ret;
            n += 1;
            if next <= cur {
                break; // frame pointers grow toward the base; a non-increasing link is the end
            }
            cur = next;
        }
        n
    }

    /// Capture the trap-time backtrace from the faulting `CONTEXT` (§5/W3): the faulting `Rip` is the
    /// innermost frame (symbolized directly by the host), and the `Rbp` chain gives the callers. Called
    /// from the VEH while the guest stack is still intact, before the recovery context is restored.
    ///
    /// # Safety
    /// `ctx` is the live faulting context for an in-window access violation in guest JIT code.
    unsafe fn capture_trap_frame(ctx: &CONTEXT) {
        let mut rets = [0usize; TRAP_MAXFRAMES];
        let n = walk_fp_chain(ctx.Rbp as usize, &mut rets);
        TRAP_PC.with(|c| c.set(ctx.Rip as usize));
        TRAP_RETS.with(|c| c.set(rets));
        TRAP_N.with(|c| c.set(n));
        TRAP_VALID.with(|c| c.set(true));
    }

    #[cfg(fiber_rt)]
    thread_local! {
        // The §14 demand-fault registration `(lo, hi, callback, ctx)` — the *recoverable* fault
        // window of the demand-paged coroutine child currently running on this thread (see
        // `mem::set_demand`). Checked by the VEH before detect-and-kill.
        pub(super) static DEMAND: Cell<Option<(usize, usize, super::DemandCb, *mut c_void)>> =
            const { Cell::new(None) };
    }

    unsafe extern "system" fn veh(ep: *mut EXCEPTION_POINTERS) -> i32 {
        let ep = &*ep;
        let rec = &*ep.ExceptionRecord;
        if rec.ExceptionCode == STATUS_ACCESS_VIOLATION {
            // ExceptionInformation[1] is the faulting address for an access violation.
            let addr = rec.ExceptionInformation[1];
            // §14 fault-driven yield: a fault in the registered demand range is *recoverable* (it
            // lies inside the armed child window, so this check precedes detect-and-kill). The
            // callback suspends the child's fiber to its parent from this VEH frame (on the child's
            // fiber stack, live across the suspension); when the parent resumes, the callback
            // returns and CONTINUE_EXECUTION re-runs the faulting access on the supplied page.
            #[cfg(fiber_rt)]
            if let Some((lo, hi, cb, ctx)) = DEMAND.with(|d| d.get()) {
                if addr >= lo && addr < hi && cb(addr, ctx) != 0 {
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
            if let Some(f) = GUARD.with(|g| g.get()) {
                if addr >= f.lo && addr < f.hi {
                    TRIPPED.with(|t| t.set(true));
                    // §5 W3: capture the trap-time backtrace from the faulting context *before* the
                    // `copy_nonoverlapping` below overwrites it with the recovery context.
                    capture_trap_frame(&*ep.ContextRecord);
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

    /// A snapshot of the thread's guard frame (§14 co-fibers): the parent swaps this around every
    /// fiber switch so the child's armed frame survives a suspend and the parent's is reinstated.
    /// The captured `Frame::ctx` points into a `run_guarded` stack frame — on a *fiber* stack for a
    /// child, which stays live while the fiber is suspended.
    #[cfg(fiber_rt)]
    pub(super) struct GuardSnap(Option<Frame>);
    #[cfg(fiber_rt)]
    impl GuardSnap {
        pub(super) fn disarmed() -> GuardSnap {
            GuardSnap(None)
        }
    }
    #[cfg(fiber_rt)]
    pub(super) fn guard_snapshot() -> GuardSnap {
        GuardSnap(GUARD.with(|g| g.get()))
    }
    #[cfg(fiber_rt)]
    pub(super) fn guard_install(s: &GuardSnap) {
        GUARD.with(|g| g.set(s.0))
    }

    static INSTALL: Once = Once::new();
    pub(super) fn install_guard() {
        INSTALL.call_once(|| {
            // first = 1 → our handler runs before any previously-registered one.
            let h = unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
            assert!(!h.is_null(), "svm-jit: AddVectoredExceptionHandler failed");
        });
    }

    /// Read and clear the trap stack the VEH captured at the most recent caught **memory fault**
    /// (§5/W3): the faulting `pc` + the frame-pointer chain's return addresses. `None` if none was
    /// captured. (Explicit-check traps — div-by-zero etc. — are not yet captured on Windows: that
    /// path needs the trap-site frame pointer, and MSVC has no `__builtin_frame_address`; see ISSUES
    /// I3. So a Windows explicit trap still yields an empty backtrace.)
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>)> {
        if !TRAP_VALID.with(|c| c.get()) {
            return None;
        }
        TRAP_VALID.with(|c| c.set(false));
        let pc = TRAP_PC.with(|c| c.get());
        let n = TRAP_N.with(|c| c.get());
        let rets = TRAP_RETS.with(|c| c.get());
        Some((pc, rets[..n].to_vec()))
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
        // Save the caller's (possibly-armed parent) guard frame to restore on exit — re-entrant so a
        // §14 child guest can run (in its own window) inside the parent's guarded call. A child fault
        // resumes at the child's `saved` context; the parent's frame is restored afterwards intact.
        let prev = GUARD.with(|g| g.replace(None));
        let mut saved = AlignedContext(core::mem::zeroed());
        // Capture the recovery point. On a guard fault the VEH copies `saved` over the fault context,
        // so execution resumes *here* with TRIPPED set — the longjmp-equivalent return.
        RtlCaptureContext(&mut saved.0);
        if TRIPPED.with(|x| x.replace(false)) {
            GUARD.with(|g| g.set(prev)); // restore the parent's frame; report the caught fault
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
        GUARD.with(|g| g.set(prev)); // ran to completion; restore the parent's frame
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
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>)> {
        None
    }
}

// ============================ PAL conformance test ============================================
// Platform-agnostic: it drives the window model + the guard PAL directly (no JIT), so the *same*
// test validates each platform's PAL — unix now, windows on CI once that leg lands.
#[cfg(test)]
mod tests {
    use super::*;

    // These PAL tests each reserve a window in the *same* process. On Windows the no-leak check
    // (`pal_release_frees_all_placeholder_fragments_no_leak`) releases its reservation and then walks
    // that VA range asserting every byte is `MEM_FREE` — but a *sibling* test's fresh reservation can
    // land in the just-freed range during the walk (cargo runs unit tests in parallel), reading as a
    // false "leak" (the intermittent `windows-latest` failure). Serialize the reserving tests so no
    // two hold overlapping-lifetime reservations across that walk. (Harmless on unix.)
    static PAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn pal_test_guard() -> std::sync::MutexGuard<'static, ()> {
        PAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
        let _serial = pal_test_guard();
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
        let _serial = pal_test_guard();
        let mut win = GuestWindow::new(8 << 10, 64 << 10);
        let w = win.rw_mut();
        w[0] = 0xAB;
        w[(8 << 10) - 1] = 0xCD;
        assert_eq!(win.rw_mut()[0], 0xAB);
        assert_eq!(win.rw_mut()[(8 << 10) - 1], 0xCD);
        // dropped here — `release` must not fault.
    }

    // The windows placeholder allocator splits its reservation into independent fragments —
    // committed-private regions (the prefix and grown-tail pages) and leftover placeholders (plus
    // shared-section views in `svm-run`'s §13 path). Teardown (`pal::release`) must free **every**
    // fragment: a single `VirtualFree(base, 0, MEM_RELEASE)` frees only the *first* and leaks the
    // rest, whose commit charge accumulated across `jit_fuzz`'s teardowns until an allocation failed
    // (the intermittent `windows-latest` crash). This pins the no-leak contract: fragment a
    // reservation the way production does, release it, then `VirtualQuery` the original range and
    // assert not one byte remains mapped/committed/reserved. (Non-vacuous: the pre-fix single-release
    // leaks all-but-the-first fragment here.)
    #[cfg(windows)]
    #[test]
    fn pal_release_frees_all_placeholder_fragments_no_leak() {
        use windows_sys::Win32::System::Memory::{
            VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE,
        };

        let _serial = pal_test_guard();
        let page = pal::page_size();
        let total = 64 * page; // room for several disjoint fragments
                               // SAFETY: a fresh placeholder reservation, released below.
        let base = unsafe { pal::reserve(total) };
        assert!(!base.is_null(), "reserve failed");
        // Fragment it the way production does: commit the prefix, then two *non-adjacent* tail pages
        // (each an interior split + replace-commit out of the big placeholder), so the reservation is
        // now a mix of committed-private regions and leftover placeholders.
        // SAFETY: every range lies within `[base, base+total)`.
        unsafe {
            pal::commit_rw(base, page); // prefix
            pal::commit_rw(base.add(20 * page), page); // a "grown" tail page
            pal::commit_rw(base.add(40 * page), page); // another, non-adjacent
        }
        // SAFETY: release exactly the reservation created above.
        unsafe { pal::release(base, total) };

        // Walk the original range: every region must now be `MEM_FREE` (nothing leaked). Nothing
        // allocates between the release and this walk, so the freed VA is not reused under us.
        let lo = base as usize;
        let hi = lo + total;
        let mut addr = lo;
        let mut leaked = 0usize;
        while addr < hi {
            let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
            // SAFETY: `VirtualQuery` accepts any address and only writes `mbi`.
            let n = unsafe {
                VirtualQuery(
                    addr as *const c_void,
                    &mut mbi,
                    core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if n == 0 {
                break;
            }
            let region_end = mbi.BaseAddress as usize + mbi.RegionSize;
            let overlap = region_end
                .min(hi)
                .saturating_sub(addr.max(mbi.BaseAddress as usize));
            if mbi.State != MEM_FREE {
                leaked += overlap;
            }
            addr = if region_end > addr {
                region_end
            } else {
                addr + page
            };
        }
        assert_eq!(
            leaked, 0,
            "pal::release leaked {leaked} bytes of the placeholder reservation \
             (fragments past the first not freed)"
        );
    }
}
