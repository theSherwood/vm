//! Guard-paged control stacks on **Windows** (`VirtualAlloc` + a `PAGE_NOACCESS` overflow guard).
//!
//! OS-specific counterpart to `stack_unix.rs`. Beyond the guard page, the Windows switch must keep the
//! TEB `StackBase`/`StackLimit`/`DeallocationStack` fields in step with the active fiber stack on every
//! switch (so SEH dispatch and stack-overflow detection follow it), so this also exposes
//! `base_ptr`/`limit_ptr`/`top` for `make` to seed those TEB slots.

use core::ffi::c_void;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
    PAGE_PROTECTION_FLAGS, PAGE_READWRITE,
};

/// x86-64 Windows page size (always 4 KiB on this arch).
const PAGE: usize = 4096;

/// A guard-paged control stack for one fiber/green thread. The lowest page is `PAGE_NOACCESS`, so an
/// overflow faults (detect-and-kill, §5) instead of silently smashing adjacent memory. Freed on drop.
pub struct Stack {
    base: *mut u8,
    len: usize,
}

// SAFETY: same contract as the unix `Stack` — an owned VA reservation (a pointer + length); only the
// thread currently running on it touches the bytes, and moving the handle between threads is sound.
unsafe impl Send for Stack {}

impl Stack {
    /// Allocate a stack with at least `size` usable bytes (rounded up to whole pages), plus one
    /// `PAGE_NOACCESS` guard page below it.
    pub fn new(size: usize) -> Stack {
        let usable = size.max(PAGE).div_ceil(PAGE) * PAGE;
        let len = usable + PAGE; // + one guard page at the low end
        unsafe {
            // SAFETY: standard VA reservation; we check the result and own it.
            let base = VirtualAlloc(
                core::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as *mut u8;
            assert!(!base.is_null(), "fiber stack VirtualAlloc failed");
            // Guard the lowest page: the stack grows down toward it, so an overflow hits PAGE_NOACCESS.
            let mut old: PAGE_PROTECTION_FLAGS = 0;
            let ok = VirtualProtect(base as *const c_void, PAGE, PAGE_NOACCESS, &mut old);
            assert!(ok != 0, "fiber stack guard VirtualProtect failed");
            Stack { base, len }
        }
    }

    /// The lowest *usable* address (just above the guard page) — the conservative low bound for a
    /// GC stack scan of a **running** fiber (see the unix counterpart). Same address as
    /// [`Self::limit_ptr`], named for the scanner's intent.
    pub fn usable_low(&self) -> *const u8 {
        // SAFETY: within the reservation; not dereferenced here.
        unsafe { self.base.add(PAGE) as *const u8 }
    }

    /// The usable region's low address + size (above the guard page) — the stack bounds handed to
    /// AddressSanitizer's fiber-switch annotations (`feature = "asan"`).
    #[cfg(feature = "asan")]
    pub fn usable(&self) -> (*const u8, usize) {
        // SAFETY: within the reservation; not dereferenced.
        (unsafe { self.base.add(PAGE) as *const u8 }, self.len - PAGE)
    }

    /// The top of the stack (highest address, exclusive) — pass to `make`; also TEB `StackBase`.
    pub fn top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end of our own reservation.
        unsafe { self.base.add(self.len) }
    }

    /// The lowest *usable* address (above the guard page) — TEB `StackLimit`.
    pub fn limit_ptr(&self) -> *mut u8 {
        // SAFETY: within the reservation.
        unsafe { self.base.add(PAGE) }
    }

    /// The allocation base — TEB `DeallocationStack`.
    pub fn base_ptr(&self) -> *mut u8 {
        self.base
    }

    /// The usable address range `[low, high)` (above the guard page), for tests.
    #[cfg(test)]
    pub fn usable_range(&self) -> (usize, usize) {
        (self.base as usize + PAGE, self.base as usize + self.len)
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        // SAFETY: `base` is exactly what `VirtualAlloc` returned; `MEM_RELEASE` wants size 0.
        unsafe {
            VirtualFree(self.base as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
