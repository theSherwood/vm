//! Guard-paged control stacks on **unix** (anonymous `mmap` + a `PROT_NONE` overflow guard).
//!
//! This is *OS*-specific, not *arch*-specific: the same stack backs both the x86-64 and the aarch64
//! register switch. The Windows analogue lives in `stack_windows.rs`.

/// A guard-paged, mmap'd control stack for one fiber/green thread. The lowest page is `PROT_NONE`, so
/// an overflow faults (detect-and-kill, §5) instead of silently smashing adjacent memory. Freed on
/// drop.
pub struct Stack {
    base: *mut u8,
    len: usize,
}

// SAFETY: `Stack` is just an owned mmap region (a pointer + length); moving it between threads is
// sound, and the bytes are only ever touched by whichever thread is currently running on it.
unsafe impl Send for Stack {}

impl Stack {
    /// Allocate a stack with at least `size` usable bytes (rounded up to whole pages), plus one
    /// `PROT_NONE` guard page below it.
    pub fn new(size: usize) -> Stack {
        // SAFETY: standard anonymous mmap; we check for MAP_FAILED and own the result.
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let usable = size.max(page).div_ceil(page) * page;
            let len = usable + page; // + one guard page
            let base = libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(base != libc::MAP_FAILED, "fiber stack mmap failed");
            // Guard the lowest page: the stack grows down toward it, so an overflow hits PROT_NONE.
            assert!(
                libc::mprotect(base, page, libc::PROT_NONE) == 0,
                "fiber stack guard mprotect failed"
            );
            Stack {
                base: base as *mut u8,
                len,
            }
        }
    }

    /// The top of the stack (highest address, exclusive) — pass to `make`.
    pub fn top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end of our own allocation.
        unsafe { self.base.add(self.len) }
    }

    /// The usable address range `[low, high)` (above the guard page), for tests/asserts that a fiber
    /// is really running on this stack.
    #[cfg(test)]
    pub fn usable_range(&self) -> (usize, usize) {
        // SAFETY: arithmetic within the allocation; not dereferenced.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        (self.base as usize + page, self.base as usize + self.len)
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        // SAFETY: `base`/`len` are exactly what we mmap'd.
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.len);
        }
    }
}
