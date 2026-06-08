//! x86-64 SysV implementation of the stack-switch primitive.
//!
//! This follows the well-trodden `boost.context` `fcontext` design: a context is just a saved stack
//! pointer; switching pushes the six callee-saved registers (`rbp rbx r12 r13 r14 r15`), stores the
//! old `rsp`, loads the new one, pops the callee-saved set, and `ret`s into the other side. The two
//! transferred words (the "from" context and a `u64` payload) ride in `rax`/`rdx`, which is how the
//! SysV ABI returns a two-word `#[repr(C)]` struct — so [`jump`] returns a [`Transfer`].

use core::arch::naked_asm;

/// What a [`jump`] hands the side it switches *into*: the context to jump back to (`fctx`, i.e. the
/// stack pointer the caller was suspended at) and the `u64` payload it passed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Transfer {
    /// The resumer's saved context — pass this to [`jump`] to switch back to it.
    pub fctx: *mut u8,
    /// The `u64` the resumer passed across the switch.
    pub data: u64,
}

/// The entry point of a freshly [`make`]d context. It receives the [`Transfer`] from the first
/// [`jump`] into it (so it knows who resumed it and any payload) and **must never return** — when its
/// work is done it [`jump`]s back to a resumer for the last time. (If it ever fell through, the
/// trampoline traps via `ud2`.)
pub type Entry = extern "C" fn(Transfer) -> !;

/// Switch to the context `to`, passing `data`. Returns once someone switches *back* to us, yielding
/// their context and payload.
///
/// # Safety
/// `to` must be a context produced by [`make`] (and not yet finished) or one returned as
/// `Transfer::fctx` from a prior switch, whose backing [`Stack`] is still alive. Switching to a stale
/// or finished context is undefined behavior.
#[unsafe(naked)]
pub unsafe extern "C" fn jump(to: *mut u8, data: u64) -> Transfer {
    // SysV: rdi = `to`, rsi = `data`; returns Transfer in rax (fctx) : rdx (data).
    naked_asm!(
        // Save callee-saved registers onto the current (outgoing) stack.
        "push rbp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        // rax := our stack pointer — the context the other side will use to switch back to us.
        "mov rax, rsp",
        // Switch to the target stack.
        "mov rsp, rdi",
        // Restore the target's callee-saved registers (mirror of the push order).
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        // Payload rides in rdx; rax already holds the "from" context. `ret` resumes the target.
        "mov rdx, rsi",
        "ret",
    )
}

/// First-resume trampoline: a freshly [`make`]d stack `ret`s here. It moves the incoming `Transfer`
/// (in rax:rdx, per [`jump`]) into the SysV argument registers, aligns the stack, and calls the entry
/// function (whose pointer [`make`] parked in the `r12` slot). The entry never returns; `ud2` makes a
/// bug loud instead of silent.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() {
    naked_asm!(
        "mov rdi, rax", // Transfer.fctx
        "mov rsi, rdx", // Transfer.data
        "and rsp, -16", // 16-byte align, then `call` pushes 8 → entry sees the SysV %16==8 it expects
        "call r12",     // entry(Transfer)
        "ud2",
    )
}

/// Lay out the fresh stack `stack_top` (its highest address, exclusive) so the first [`jump`] into the
/// returned context begins executing `entry`.
///
/// The image, low → high from the returned pointer, is the six callee-saved slots that [`jump`] will
/// `pop` (with `entry` parked in the `r12` slot for the trampoline to `call`) followed by the
/// trampoline return address:
///
/// ```text
///   [r15=0][r14=0][r13=0][r12=entry][rbx=0][rbp=0][ret=trampoline]
///    ^ returned context pointer
/// ```
///
/// # Safety
/// `stack_top` must be the top of a live, writable, suitably sized stack (e.g. [`Stack::top`]).
pub unsafe fn make(stack_top: *mut u8, entry: Entry) -> *mut u8 {
    // 16-align the base, then push the seven 8-byte slots top-down.
    let mut sp = (stack_top as usize) & !15usize;
    let mut push = |v: u64| {
        sp -= 8;
        // SAFETY: `sp` stays within the caller-provided stack region.
        unsafe { core::ptr::write(sp as *mut u64, v) };
    };
    push(trampoline as *const () as u64); // return address → trampoline on first resume
    push(0); // rbp
    push(0); // rbx
    push(entry as *const () as u64); // r12  (trampoline `call r12`)
    push(0); // r13
    push(0); // r14
    push(0); // r15  ← sp ends here; `jump` pops r15 first
    sp as *mut u8
}

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

    /// The top of the stack (highest address, exclusive) — pass to [`make`].
    pub fn top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end of our own allocation.
        unsafe { self.base.add(self.len) }
    }

    /// The usable address range `[low, high)` (above the guard page), for tests/asserts that a fiber
    /// is really running on this stack.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A coroutine that keeps a running total: each resume adds the payload and yields the new total;
    /// a `u64::MAX` payload is the "finish" sentinel.
    extern "C" fn summer(mut t: Transfer) -> ! {
        let mut total: u64 = 0;
        loop {
            if t.data == u64::MAX {
                // Final switch back to the resumer; never resumed again.
                // SAFETY: `t.fctx` is the live resumer context.
                unsafe { jump(t.fctx, total) };
                unreachable!("resumed after finishing");
            }
            total = total.wrapping_add(t.data);
            // SAFETY: switch back to the resumer, who holds our stack alive.
            t = unsafe { jump(t.fctx, total) };
        }
    }

    #[test]
    fn switch_roundtrip_accumulates() {
        let stack = Stack::new(64 * 1024);
        let mut ctx = unsafe { make(stack.top(), summer) };
        for (payload, want) in [(5u64, 5u64), (10, 15), (100, 115), (1, 116)] {
            let t = unsafe { jump(ctx, payload) };
            ctx = t.fctx;
            assert_eq!(t.data, want, "after adding {payload}");
        }
        let t = unsafe { jump(ctx, u64::MAX) };
        assert_eq!(t.data, 116, "final total");
    }

    /// The coroutine reports the address of one of its locals; it must lie inside the fiber stack's
    /// usable range, proving execution really moved off the caller's stack.
    extern "C" fn report_sp(t: Transfer) -> ! {
        let local: u64 = 0xABCD;
        let addr = &local as *const u64 as u64;
        core::hint::black_box(local);
        unsafe { jump(t.fctx, addr) };
        unreachable!()
    }

    #[test]
    fn runs_on_the_fiber_stack() {
        let stack = Stack::new(64 * 1024);
        let (lo, hi) = stack.usable_range();
        let ctx = unsafe { make(stack.top(), report_sp) };
        let t = unsafe { jump(ctx, 0) };
        let addr = t.data as usize;
        assert!(
            (lo..hi).contains(&addr),
            "local at {addr:#x} not in fiber stack [{lo:#x}, {hi:#x})"
        );
    }

    /// Deep recursion on the fiber stack: a recursive Fibonacci proves the switched-to stack supports
    /// ordinary nested frames, not just a single leaf.
    extern "C" fn fibber(t: Transfer) -> ! {
        fn fib(n: u64) -> u64 {
            if n < 2 {
                n
            } else {
                fib(n - 1) + fib(n - 2)
            }
        }
        let r = fib(t.data);
        unsafe { jump(t.fctx, r) };
        unreachable!()
    }

    #[test]
    fn recursion_on_fiber_stack() {
        let stack = Stack::new(256 * 1024);
        let ctx = unsafe { make(stack.top(), fibber) };
        let t = unsafe { jump(ctx, 25) };
        assert_eq!(t.data, 75025); // fib(25)
    }

    /// Stress: many back-and-forth switches must stay consistent (no register/stack drift).
    #[test]
    fn many_switches_are_stable() {
        let stack = Stack::new(64 * 1024);
        let mut ctx = unsafe { make(stack.top(), summer) };
        let mut expect = 0u64;
        for i in 0..100_000u64 {
            expect = expect.wrapping_add(i | 1); // never u64::MAX
            let t = unsafe { jump(ctx, i | 1) };
            ctx = t.fctx;
            assert_eq!(t.data, expect);
        }
    }

    /// Two independent fibers interleaved through the same resumer keep separate state.
    #[test]
    fn two_fibers_independent() {
        let sa = Stack::new(64 * 1024);
        let sb = Stack::new(64 * 1024);
        let mut a = unsafe { make(sa.top(), summer) };
        let mut b = unsafe { make(sb.top(), summer) };
        let ta = unsafe { jump(a, 3) };
        a = ta.fctx;
        let tb = unsafe { jump(b, 7) };
        b = tb.fctx;
        let ta = unsafe { jump(a, 4) };
        a = ta.fctx;
        assert_eq!(ta.data, 7); // 3 + 4
        let tb = unsafe { jump(b, 70) };
        b = tb.fctx;
        assert_eq!(tb.data, 77); // 7 + 70
        let _ = (a, b);
    }
}
