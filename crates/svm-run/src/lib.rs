//! `svm-run` — the **embedding runtime**: instantiate a verified module with the MVP powerbox
//! (§3e) and run it on the Cranelift JIT, returning its outcome and the bytes it wrote.
//!
//! This is the single, reusable host glue — the `cap.call` trampoline ([`cap_thunk`]) plus the
//! powerbox grant ([`run_powerbox`]) — that was previously copy-pasted across the JIT test
//! harnesses (`c_frontend.rs`, `jit_diff.rs`). The `svm-run` **CLI** is a thin wrapper over it.
//!
//! It is *not* escape-TCB: the verifier (run before this) is what makes a module safe to run;
//! this crate only wires the host capabilities a guest is granted. A guest that traps
//! (out-of-window fault, `unreachable`, …) is **detect-and-killed** (§5) — surfaced here as an
//! `Err`, never undefined behaviour in the host.

use core::ffi::c_void;

use svm_interp::{GuestMem, Host, RegionBacking, StreamRole, Trap};
// `SharedBacking`'s methods + its `ShmBacking` impl are only used by the unix shared-mapping path.
#[cfg(unix)]
use svm_interp::SharedBacking;
use svm_ir::{Module, ValType};

// Re-export the value type so embedders (and the CLI) need not also depend on `svm-interp`.
pub use svm_interp::Value;
use svm_jit::{compile_and_run, compile_and_run_with_host, JitOutcome, TrapKind, EXIT_CODE};

/// The host trampoline bridging the JIT's [`svm_jit::CapThunk`] ABI (§9) to the reference
/// [`Host`]'s capability dispatch — the host code a real embedder supplies. One shared copy.
///
/// # Safety
/// Honours the `CapThunk` contract: `ctx` is a live `*mut Host`; `args`/`results` are valid for
/// `n_args`/`n_results`; `mem_base` (when non-null) is the guest window with `mem_size` backed
/// bytes inside a `mem_reserved` reservation; `trap_out` is writable. The trap cell is encoded as
/// the JIT expects: `0` = ok, a [`TrapKind`] for a fault, or `EXIT_CODE | (code << 32)` for `Exit`.
pub unsafe extern "C" fn cap_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let host = &mut *(ctx as *mut Host);
    // The JIT passes a null args/results pointer when the count is 0; `from_raw_parts` requires a
    // non-null (aligned) pointer even for an empty slice, so use `&[]` in that case (UB otherwise).
    let arg_slots = if n_args == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args, n_args as usize)
    };
    // The guest window with a real hardware-protected Memory capability (`map`/`unmap`/`protect`,
    // incl. growth into the reserved tail): `mprotect` on unix, `VirtualProtect`/`VirtualAlloc` on
    // windows — the same software-page-map model, only the syscalls differ.
    #[cfg(any(unix, windows))]
    let mut wm = MprotectWindow::new(mem_base, mem_size, mem_reserved);
    #[cfg(any(unix, windows))]
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };
    // Any other target has no window backend (the JIT, `svm-jit`, does not build there anyway).
    #[cfg(not(any(unix, windows)))]
    let gm: Option<&mut dyn GuestMem> = {
        let _ = (mem_base, mem_size, mem_reserved);
        None
    };
    match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
        Ok(res) => {
            if n_results != 0 {
                let out = std::slice::from_raw_parts_mut(results, n_results as usize);
                for (o, r) in out.iter_mut().zip(res) {
                    *o = r;
                }
            }
            *trap_out = 0;
        }
        Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
        Err(_) => *trap_out = TrapKind::CapFault as i64,
    }
}

/// The **host** page size: the protection granularity for `map`/`unmap`/`protect`, matching the
/// interpreter (`svm_interp`) and the JIT (`svm-jit`) on the same host so all three agree
/// page-for-page (§4 "pin page size", host-page default). `sysconf(_SC_PAGESIZE)` on unix,
/// `GetSystemInfo` on windows.
#[cfg(unix)]
fn host_page_size() -> u64 {
    // SAFETY: sysconf is always safe; _SC_PAGESIZE is positive.
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        p if p > 0 => p as u64,
        _ => 4096,
    }
}
#[cfg(windows)]
fn host_page_size() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: GetSystemInfo only writes its out-param; always safe.
    let mut si: SYSTEM_INFO = unsafe { core::mem::zeroed() };
    unsafe { GetSystemInfo(&mut si) };
    match si.dwPageSize as u64 {
        0 => 4096,
        p => p,
    }
}

/// A [`GuestMem`] over the JIT's guest window whose `map`/`unmap`/`protect` (the Memory capability,
/// §3e) are backed by **real hardware page protection** on the window pages (`mprotect` on unix,
/// `VirtualAlloc`/`VirtualProtect` on windows), mirrored by a software page-state map. The mirror
/// lets cap-buffer borrows (§7) **fail closed** (`-EFAULT`) on an unmapped/RO page instead of
/// faulting the host outside the guarded call, and bounds growth to the reserved mask domain —
/// keeping this backend bit-identical to the interpreter's paged `Mem` (the §18 oracle, enforced by
/// `jit_diff`'s differential). The page-map model is portable; only the three hardware primitives
/// (`hw_commit_rw`/`hw_apply`/`hw_release_hint`) differ per OS.
///
/// # Safety
/// `base` must point at the JIT guest window: `[base, base+mapped)` initially RW and the whole
/// `[base, base+reserved)` a live inaccessible/RW reservation owned for the call's duration.
#[cfg(any(unix, windows))]
pub struct MprotectWindow {
    base: *mut u8,
    mapped: u64,
    reserved: u64,
    /// Host page size (`host_page_size()`), the protection granularity (matches `svm_interp`).
    page: u64,
    /// Page index ⇒ explicit state; absent ⇒ region default (rw in `[0, mapped)`, unmapped in the
    /// reserved tail). Mirrors `svm_interp`'s page map so the two backends agree page-for-page.
    prot: std::collections::BTreeMap<u64, PageState>,
}

#[cfg(any(unix, windows))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Rw,
    Ro,
    Unmapped,
}

#[cfg(any(unix, windows))]
impl MprotectWindow {
    /// Wrap the JIT window `[base, base+mapped)` (backed) inside a `reserved` mask domain.
    pub fn new(base: *mut u8, mapped: u64, reserved: u64) -> MprotectWindow {
        MprotectWindow {
            base,
            mapped,
            reserved: reserved.max(mapped),
            page: host_page_size(),
            prot: std::collections::BTreeMap::new(),
        }
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed — the
    /// same default rule as the interpreter (`svm_interp::Mem::page_access`).
    fn page_access(&self, page: u64) -> Option<bool> {
        match self.prot.get(&page) {
            Some(PageState::Rw) => Some(true),
            Some(PageState::Ro) => Some(false),
            Some(PageState::Unmapped) => None,
            None => (page * self.page < self.mapped).then_some(true),
        }
    }

    /// Every page of `[ptr, ptr+len)` is committed (and writable when `write`), within
    /// `[0, reserved)` — the §7 borrow check, mirroring `svm_interp`.
    fn range_committed(&self, ptr: u64, len: u64, write: bool) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        if end > self.reserved {
            return false;
        }
        if len == 0 {
            return true;
        }
        (ptr / self.page..=(end - 1) / self.page)
            .all(|p| matches!(self.page_access(p), Some(w) if w || !write))
    }

    /// Validate a `map`/`unmap`/`protect` range and return its inclusive page-index span, or
    /// `-EINVAL` (page-aligned offset, non-zero len, within `[0, reserved)`) — matching the
    /// interpreter's `prot_pages` (growth into the reserved tail is allowed).
    fn prot_pages(&self, offset: u64, len: u64) -> Result<std::ops::RangeInclusive<u64>, i64> {
        const EINVAL: i64 = -22;
        if len == 0 || !offset.is_multiple_of(self.page) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.reserved {
            return Err(EINVAL);
        }
        Ok((offset / self.page)..=((end - 1) / self.page))
    }

    /// Update one page's software state from cap `prot` bits, mirroring `svm_interp::set_prot`:
    /// a read-write page is left absent in the prefix, explicit `Rw` in the reserved tail.
    fn set_prot(&mut self, page: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        if prot & PROT_WRITE != 0 {
            if page * self.page < self.mapped {
                self.prot.remove(&page);
            } else {
                self.prot.insert(page, PageState::Rw);
            }
        } else if prot & PROT_READ != 0 {
            self.prot.insert(page, PageState::Ro);
        } else {
            self.prot.insert(page, PageState::Unmapped);
        }
    }

    // ---- the three hardware primitives (the only per-OS part) -----------------------------------
    // All take a **page-aligned** `[off, off+len)` already validated `⊆ reserved` by `prot_pages`.

    /// Make `[off, off+len)` **committed and read-write** (so a following zero-fill / protection
    /// change lands). On unix the reservation is `MAP_NORESERVE`, so `mprotect(RW)` suffices and the
    /// kernel demand-zeroes; on windows the tail is reserved-but-uncommitted, so `VirtualAlloc(
    /// MEM_COMMIT)` is required (it zero-fills only *newly* committed pages — callers zero explicitly
    /// when they need it).
    #[cfg(unix)]
    fn hw_commit_rw(&self, off: u64, len: u64) {
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::mprotect(
                self.base.add(off as usize) as *mut c_void,
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
            );
        }
    }
    #[cfg(windows)]
    fn hw_commit_rw(&self, off: u64, len: u64) {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, PAGE_READWRITE};
        // SAFETY: `[base+off, +len)` is within the reservation (validated); committing an already-
        // committed page is a no-op that re-asserts RW without zeroing live contents.
        unsafe {
            VirtualAlloc(
                self.base.add(off as usize) as *const c_void,
                len as usize,
                MEM_COMMIT,
                PAGE_READWRITE,
            );
        }
    }

    /// Apply cap `prot` bits (`0` none / `1` read / `3` read-write) to the committed `[off, off+len)`
    /// without touching its contents — `mprotect` on unix, `VirtualProtect` on windows. `none` maps
    /// to `PROT_NONE`/`PAGE_NOACCESS` (the page stays committed but faults on access).
    #[cfg(unix)]
    fn hw_apply(&self, off: u64, len: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        let hw = if prot & PROT_WRITE != 0 {
            libc::PROT_READ | libc::PROT_WRITE
        } else if prot & PROT_READ != 0 {
            libc::PROT_READ
        } else {
            libc::PROT_NONE
        };
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::mprotect(self.base.add(off as usize) as *mut c_void, len as usize, hw);
        }
    }
    #[cfg(windows)]
    fn hw_apply(&self, off: u64, len: u64, prot: i32) {
        use windows_sys::Win32::System::Memory::{
            VirtualProtect, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE,
        };
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        let flags: PAGE_PROTECTION_FLAGS = if prot & PROT_WRITE != 0 {
            PAGE_READWRITE
        } else if prot & PROT_READ != 0 {
            PAGE_READONLY
        } else {
            PAGE_NOACCESS
        };
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        // SAFETY: `[base+off, +len)` is committed (callers `hw_commit_rw` first) and in-reservation.
        unsafe {
            VirtualProtect(
                self.base.add(off as usize) as *const c_void,
                len as usize,
                flags,
                &mut old,
            );
        }
    }

    /// Hint the OS to drop the physical backing of the now-inaccessible `[off, off+len)` (a pure
    /// memory-footprint optimization, *after* the range has been zeroed + protected `none`). `unmap`
    /// semantics ("re-`map` reads zero") are already guaranteed by the explicit zero, so this need
    /// not be exact: `MADV_DONTNEED` on unix; a no-op on windows (the pages stay committed-but-
    /// `NOACCESS`, which keeps the snapshot's `restore_rw` able to read the backed prefix).
    #[cfg(unix)]
    fn hw_release_hint(&self, off: u64, len: u64) {
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::madvise(
                self.base.add(off as usize) as *mut c_void,
                len as usize,
                libc::MADV_DONTNEED,
            );
        }
    }
    #[cfg(windows)]
    fn hw_release_hint(&self, _off: u64, _len: u64) {}
}

#[cfg(any(unix, windows))]
impl GuestMem for MprotectWindow {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.range_committed(ptr, len, false) {
            return None;
        }
        // SAFETY: range_committed proved every page mapped+readable and `[ptr,ptr+len) ⊆ reserved`.
        let w = unsafe { std::slice::from_raw_parts(self.base, self.reserved as usize) };
        Some(w[ptr as usize..(ptr + len) as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        if !self.range_committed(ptr, data.len() as u64, true) {
            return None;
        }
        // SAFETY: range_committed proved every page mapped+writable and the range ⊆ reserved.
        let w = unsafe { std::slice::from_raw_parts_mut(self.base, self.reserved as usize) };
        w[ptr as usize..ptr as usize + data.len()].copy_from_slice(data);
        Some(())
    }
    /// §3e op 0 `map`: (re)commit the **whole pages** covering `[offset,offset+len)` with `prot`,
    /// zero-filled — including **growth** into the reserved tail. The commit/zero/protect span the
    /// page range, not the raw `[offset, len)`, so the zeroing is page-granular and matches the
    /// interpreter's per-page `Mem::map` on any host page size (on a 16 KiB host, `len` may be a
    /// fraction of a page).
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        // Commit + make RW so the zero-fill lands, zero (a fresh commit reads zero), then apply the
        // requested protection.
        self.hw_commit_rw(start, plen);
        // SAFETY: the page range is RW and within the reserved mapping (validated).
        unsafe { std::ptr::write_bytes(self.base.add(start as usize), 0, plen as usize) };
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_apply(start, plen, prot);
        0
    }
    /// §3e op 1 `unmap`: decommit the **whole pages** covering `[offset,offset+len)` — any access
    /// faults, and a re-`map` reads zero. Operates on the page range (page-granular work needs whole
    /// pages) to match `Mem::unmap`.
    ///
    /// We **explicitly zero** the range so a later re-`map` reads zero on every platform: on Linux
    /// `MADV_DONTNEED` alone would suffice (next fault returns a fresh zero page), but Darwin treats
    /// it as advisory (stale bytes survive) and windows keeps the page committed — so the zero is what
    /// makes them all agree, and `hw_release_hint` is then a pure footprint optimization.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        // Commit + make RW, zero it, hint the OS to drop the backing, then protect NONE so any later
        // access faults (detect-and-kill).
        self.hw_commit_rw(start, plen);
        // SAFETY: the page range is RW and within the reserved mapping (validated).
        unsafe { std::ptr::write_bytes(self.base.add(start as usize), 0, plen as usize) };
        self.hw_release_hint(start, plen);
        self.hw_apply(start, plen, 0 /* none */);
        for page in pages {
            self.prot.insert(page, PageState::Unmapped);
        }
        0
    }
    /// §3e op 2 `protect`: change protection without touching backing (the D40 RO mechanism). The
    /// page is committed first (a no-op on already-committed pages; on windows it makes a never-mapped
    /// reserved tail page addressable, matching the interpreter's "absent page reads zero" model)
    /// **without** zeroing live contents, then the protection is applied.
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        self.hw_commit_rw(start, plen);
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_apply(start, plen, prot);
        0
    }
    /// §3e op 3 `page_size`: the hardware protection granularity (`self.page` = the host page) —
    /// the unit `map`/`unmap`/`protect` round to, matching the interpreter's `Mem::page_size` on the
    /// same host so the two backends agree.
    fn page_size(&self) -> i64 {
        self.page as i64
    }

    /// §13 op 0 `map`: alias a `SharedRegion` into the window with a **real shared mapping** —
    /// `mmap(MAP_SHARED | MAP_FIXED)` of the region's `os_fd` over `[win_off, win_off+len)`, so two
    /// mappings of the same region (here, or in another window) name the *same* physical pages: true
    /// hardware aliasing with zero per-access overhead (§13). The mapping persists in the window's
    /// reservation across `cap.call`s — this `MprotectWindow` is rebuilt per call, but the OS mapping
    /// and the region fd (owned by the `Host`'s backing) are not. Validation mirrors the interpreter's
    /// `Mem::map_region`. Wired on Linux (`memfd`); macOS/windows are a follow-up (→ `-EINVAL`).
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        _region: u32,
        backing: RegionBacking,
    ) -> i64 {
        const EINVAL: i64 = -22;
        #[cfg(unix)]
        {
            const PROT_READ: i32 = 1;
            const PROT_WRITE: i32 = 2;
            let pages = match self.prot_pages(win_off, len) {
                Ok(p) => p,
                Err(e) => return e,
            };
            if !region_off.is_multiple_of(self.page) || prot & PROT_READ == 0 {
                return EINVAL;
            }
            match region_off.checked_add(len) {
                Some(end) if end <= backing.size() => {}
                _ => return EINVAL,
            }
            let Some(fd) = backing.os_fd() else {
                return EINVAL;
            };
            let writable = prot & PROT_WRITE != 0;
            let start = *pages.start() * self.page;
            // Whole-page span covering `[win_off, win_off+len)`. The region fd is page-rounded ≥ this,
            // so `region_off + plen` never maps past EOF (no SIGBUS); bytes past the logical region
            // size read zero on both backends.
            let plen = (*pages.end() + 1 - *pages.start()) * self.page;
            let hw = if writable {
                libc::PROT_READ | libc::PROT_WRITE
            } else {
                libc::PROT_READ
            };
            // SAFETY: `[base+start, +plen) ⊆` the reserved window (validated by `prot_pages`).
            // `MAP_FIXED` replaces those reserved pages with a shared mapping of the region fd at
            // `region_off`; the fd outlives the run (held by the Host's backing).
            let p = unsafe {
                libc::mmap(
                    self.base.add(start as usize) as *mut c_void,
                    plen as usize,
                    hw,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    region_off as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                return EINVAL;
            }
            // Mirror the software page state (committed; RW or RO) for in-call §7 borrow checks.
            let state = if writable {
                PageState::Rw
            } else {
                PageState::Ro
            };
            for page in pages {
                self.prot.insert(page, state);
            }
            0
        }
        // TODO(§13 windows, issue #1): wire real shared mappings via placeholder reservations
        // (`VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)` + `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`).
        // Until then SharedRegion `map` is unsupported on windows; pinned by the `#[cfg(windows)]`
        // test in `svm/tests/shared_region.rs`.
        #[cfg(not(unix))]
        {
            let _ = (win_off, region_off, len, prot, backing);
            EINVAL
        }
    }
}

/// Create a fresh anonymous, `cap`-byte OS shared-memory fd: `memfd_create` on Linux, an immediately-
/// `shm_unlink`ed POSIX `shm_open` object on other unix (macOS). The fd keeps the (unlinked) object
/// alive; closing it reclaims the memory. Sized with `ftruncate` so a window `mmap` of whole pages
/// never faults past EOF.
#[cfg(unix)]
fn create_region_fd(cap: usize) -> std::io::Result<std::os::fd::OwnedFd> {
    use std::os::fd::{FromRawFd, OwnedFd};
    #[cfg(target_os = "linux")]
    // SAFETY: a valid NUL-terminated name; returns a fresh owned fd or -1.
    let raw = unsafe { libc::memfd_create(c"svm_region".as_ptr(), libc::MFD_CLOEXEC) };
    #[cfg(not(target_os = "linux"))]
    let raw = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        // A short unique name (POSIX shm names are length-capped): "/svm<pid·seq in hex>".
        let uniq = ((std::process::id() as u64) << 24) ^ SEQ.fetch_add(1, Ordering::Relaxed);
        let name = format!("/svm{uniq:x}\0");
        // SAFETY: a valid NUL-terminated name; O_EXCL so we own a fresh object, or -1.
        let raw = unsafe {
            libc::shm_open(
                name.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_int,
            )
        };
        if raw >= 0 {
            // Unlink now: the open fd keeps the object usable; it's anonymous + auto-reclaimed on close.
            // SAFETY: `name` is the just-created object's NUL-terminated name.
            unsafe { libc::shm_unlink(name.as_ptr() as *const libc::c_char) };
        }
        raw
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `raw` is a fresh owned fd.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: sizing the just-created object (before any mmap), per the once-only ftruncate rule.
    if unsafe { libc::ftruncate(raw, cap as libc::off_t) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(fd)
}

/// A §13 `SharedRegion` backing over a real OS shared-memory object (`memfd`/`shm`), whose `os_fd` a
/// window `mmap`s `MAP_SHARED` for true hardware aliasing. The fd is also mapped once into the host
/// process so `read_byte`/`write_byte` work (e.g. if an interpreter `Mem` uses this backing); in the
/// JIT differential the guest's loads/stores go straight through the window's shared mapping. Unix
/// only; windows (`CreateFileMapping` + placeholder reservations) is a follow-up.
#[cfg(unix)]
struct ShmBacking {
    fd: std::os::fd::OwnedFd,
    ptr: *mut u8,
    cap: usize, // page-rounded mapping length (the fd size)
    len: usize, // logical region size the guest sees
}

#[cfg(unix)]
impl ShmBacking {
    fn new(len: usize) -> std::io::Result<ShmBacking> {
        use std::os::fd::AsRawFd;
        let page = host_page_size() as usize;
        let cap = len.max(1).div_ceil(page) * page;
        let fd = create_region_fd(cap)?;
        // SAFETY: map the whole object shared into the host (for `read_byte`/`write_byte`).
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                cap,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(ShmBacking {
            fd,
            ptr: p as *mut u8,
            cap,
            len,
        })
    }
}

#[cfg(unix)]
impl SharedBacking for ShmBacking {
    fn size(&self) -> u64 {
        self.len as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) }
        } else {
            0
        }
    }
    fn write_byte(&self, off: u64, b: u8) {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) = b }
        }
    }
    fn os_fd(&self) -> Option<i32> {
        use std::os::fd::AsRawFd;
        Some(self.fd.as_raw_fd())
    }
}

#[cfg(unix)]
impl Drop for ShmBacking {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`cap` are the host mapping from `new`; the fd is closed by `OwnedFd`.
        unsafe { libc::munmap(self.ptr as *mut c_void, self.cap) };
    }
}

/// Create a §13 `SharedRegion` backing over a fresh `len`-byte OS shared-memory object — install it
/// with [`svm_interp::Host::grant_shared_region_backed`] so the JIT can `mmap` it for real aliasing.
#[cfg(unix)]
pub fn new_shared_region(len: usize) -> RegionBacking {
    std::rc::Rc::new(ShmBacking::new(len).expect("create shared region"))
}

/// How a guest program ended: its entry returned values, or it invoked `Exit(code)` (§3e).
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Returned(Vec<Value>),
    Exited(i32),
}

/// The result of running a program through the powerbox: how it ended, plus the bytes it wrote
/// to stdout/stderr via the `Stream` capabilities.
#[derive(Debug, Clone)]
pub struct Run {
    pub outcome: Outcome,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The frontend's powerbox entry shape (function 0): the three `i32` handles
/// `_start(stdout, stdin, exit)`, or four `_start(stdout, stdin, exit, memory)` once the program
/// uses the Memory capability (a guest heap that grows via `map`, §3e/§4). A module whose entry
/// matches either is a runnable *program*; anything else is a bare kernel (run with [`run_kernel`]).
pub fn is_powerbox_entry(module: &Module) -> bool {
    matches!(
        module.funcs.first().map(|f| f.params.as_slice()),
        Some([ValType::I32, ValType::I32, ValType::I32])
            | Some([ValType::I32, ValType::I32, ValType::I32, ValType::I32])
    )
}

fn typed(t: ValType, v: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(v as i32),
        ValType::I64 => Value::I64(v),
        ValType::F32 => Value::F32(f32::from_bits(v as u32)),
        ValType::F64 => Value::F64(f64::from_bits(v as u64)),
    }
}

/// Run `module`'s entry (function 0) on the JIT under the MVP powerbox (§3e): a writable
/// `stdout`, a readable `stdin` seeded from `stdin`, and `Exit` — the three handles the
/// frontend's `_start` expects, granted in declared order. Returns the outcome and captured
/// output. `Err` if the (already-verified) module fails to JIT-compile, or if the guest
/// **traps** (detect-and-kill, §5) — the guest can never corrupt the host.
pub fn run_powerbox(module: &Module, stdin: &[u8]) -> Result<Run, String> {
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    // Grant in the powerbox's declared import order: stdout, stdin, exit, then Memory if the
    // entry takes a 4th handle (§3e / D44) — so a `map`-growing guest heap has a handle to call.
    let wants_memory = matches!(
        module.funcs.first().map(|f| f.params.len()),
        Some(n) if n >= 4
    );
    let mut slots = vec![
        host.grant_stream(StreamRole::Out) as i64,
        host.grant_stream(StreamRole::In) as i64,
        host.grant_exit() as i64,
    ];
    if wants_memory {
        slots.push(host.grant_memory() as i64);
    }
    let jit = compile_and_run_with_host(
        module,
        0,
        &slots,
        cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
    .map_err(|e| format!("JIT compile failed: {e:?}"))?;

    let outcome = match jit {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Outcome::Returned(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Outcome::Exited(code),
        JitOutcome::Trapped(kind) => {
            return Err(format!("guest trapped ({kind:?}) — detect-and-kill (§5)"))
        }
    };
    Ok(Run {
        outcome,
        stdout: host.stdout,
        stderr: host.stderr,
    })
}

/// Run a bare (non-powerbox) kernel — `module`'s entry on the JIT with `args` and no host
/// capabilities — returning its typed result values. For hand-written IR that is a pure
/// function rather than a program (e.g. the benchmark kernels). `Err` on compile failure,
/// a guest trap, or an `Exit` (a kernel should not call one).
pub fn run_kernel(module: &Module, args: &[i64]) -> Result<Vec<Value>, String> {
    match compile_and_run(module, 0, args).map_err(|e| format!("JIT compile failed: {e:?}"))? {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Ok(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Err(format!("kernel called Exit({code})")),
        JitOutcome::Trapped(kind) => Err(format!("kernel trapped ({kind:?})")),
    }
}
