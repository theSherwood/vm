//! x86-64 **SysV** register/stack switch (Linux, macOS-x86-64, other unix).
//!
//! This follows the well-trodden `boost.context` `fcontext` design: a context is just a saved stack
//! pointer; switching pushes the six callee-saved registers (`rbp rbx r12 r13 r14 r15`), stores the
//! old `rsp`, loads the new one, pops the callee-saved set, and `ret`s into the other side. The two
//! transferred words (the "from" context and a `u64` payload) ride in `rax`/`rdx`, which is how the
//! SysV ABI returns a two-word `#[repr(C)]` struct — so [`jump`] returns a [`Transfer`].
//!
//! The guard-paged control [`Stack`](crate::stack::Stack) is OS-specific and lives in `stack_unix.rs`.

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
/// `Transfer::fctx` from a prior switch, whose backing [`Stack`](crate::stack::Stack) is still alive.
/// Switching to a stale or finished context is undefined behavior.
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

/// First-resume trampoline: a freshly [`make`]d stack `ret`s here. It moves the incoming [`Transfer`]
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
/// `stack` must be a live, writable, suitably sized control stack.
pub unsafe fn make(stack: &crate::stack::Stack, entry: Entry) -> *mut u8 {
    // 16-align the base, then push the seven 8-byte slots top-down.
    let mut sp = (stack.top() as usize) & !15usize;
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
