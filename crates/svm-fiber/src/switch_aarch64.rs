//! aarch64 **AAPCS64** register/stack switch (Linux / macOS arm64, other arm64 unix).
//!
//! Same `boost.context` `fcontext` model as the other switches, and the cleanest of the three: the
//! 16-byte [`Transfer`] is a small integer composite, so AAPCS64 passes/returns it in `x0:x1` (no
//! hidden-pointer dance), and there is no TEB to keep in step (the unix mmap stack + the signal-based
//! detect-and-kill guard are ISA-neutral). A switch saves the callee-saved GP set `x19`–`x30` and the
//! callee-saved FP halves `d8`–`d15`, swaps `sp`, and `ret`s through the restored `x30`.
//!
//! macOS note: `x18` is the reserved platform register — we never touch it. Rust's
//! `aarch64-apple-darwin` is plain arm64 (not arm64e), so no pointer-authentication signing is needed.
//!
//! The guard-paged control [`Stack`](crate::stack::Stack) is the shared unix one (`stack_unix.rs`).

use core::arch::naked_asm;

/// What a [`jump`] hands the side it switches *into*: the context to jump back to and the payload.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Transfer {
    /// The resumer's saved context — pass this to [`jump`] to switch back to it.
    pub fctx: *mut u8,
    /// The `u64` the resumer passed across the switch.
    pub data: u64,
}

/// The entry point of a freshly [`make`]d context; receives the first [`Transfer`] and never returns.
pub type Entry = extern "C" fn(Transfer) -> !;

// Saved-context frame layout (offsets from the saved sp), 0xa0 bytes:
//   0x00 x19 0x08 x20  0x10 x21 0x18 x22  0x20 x23 0x28 x24  0x30 x25 0x38 x26
//   0x40 x27 0x48 x28  0x50 x29 0x58 x30  0x60 d8  0x68 d9  ... 0x98 d15
// The switched-in side restores from it and `ret`s through the restored x30.

/// Switch to the context `to`, passing `data`. Returns once someone switches *back* to us.
///
/// # Safety
/// `to` must be a live context from [`make`] (not yet finished) or a `Transfer::fctx` from a prior
/// switch whose backing [`Stack`](crate::stack::Stack) is still alive.
#[unsafe(naked)]
pub unsafe extern "C" fn jump(to: *mut u8, data: u64) -> Transfer {
    // AAPCS64: x0 = `to`, x1 = `data`; returns Transfer in x0 (fctx) : x1 (data).
    naked_asm!(
        "sub sp, sp, #0xa0",
        // save callee-saved GP (x19..x30) + FP (d8..d15)
        "stp x19, x20, [sp, #0x00]",
        "stp x21, x22, [sp, #0x10]",
        "stp x23, x24, [sp, #0x20]",
        "stp x25, x26, [sp, #0x30]",
        "stp x27, x28, [sp, #0x40]",
        "stp x29, x30, [sp, #0x50]",
        "stp d8, d9, [sp, #0x60]",
        "stp d10, d11, [sp, #0x70]",
        "stp d12, d13, [sp, #0x80]",
        "stp d14, d15, [sp, #0x90]",
        // x2 = our context pointer (to hand back as Transfer.fctx); survives the switch (scratch).
        "mov x2, sp",
        // switch to the target stack
        "mov sp, x0",
        // restore the target's callee-saved regs
        "ldp x19, x20, [sp, #0x00]",
        "ldp x21, x22, [sp, #0x10]",
        "ldp x23, x24, [sp, #0x20]",
        "ldp x25, x26, [sp, #0x30]",
        "ldp x27, x28, [sp, #0x40]",
        "ldp x29, x30, [sp, #0x50]",
        "ldp d8, d9, [sp, #0x60]",
        "ldp d10, d11, [sp, #0x70]",
        "ldp d12, d13, [sp, #0x80]",
        "ldp d14, d15, [sp, #0x90]",
        "add sp, sp, #0xa0",
        // Transfer{fctx = our ctx (x2), data = the data we were passed (x1, untouched)}.
        "mov x0, x2",
        "ret",
    )
}

/// First-resume trampoline. The fresh context `ret`s here with `x0`/`x1` = the incoming [`Transfer`]
/// and `x19` = the entry fn (both seeded by [`make`]). AAPCS64 passes the 16-byte struct by value in
/// `x0:x1`, so just call `entry`.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() {
    naked_asm!(
        "blr x19", // entry(Transfer{x0, x1})
        "brk #1",  // entry must never return
    )
}

/// Lay out the fresh stack so the first [`jump`] into the returned context begins executing `entry`.
///
/// # Safety
/// `stack` must be a live, writable control stack.
pub unsafe fn make(stack: &crate::stack::Stack, entry: Entry) -> *mut u8 {
    let top = stack.top() as usize;
    let ctx = (top & !0xf) - 0xa0; // 16-aligned; `jump` restores from here then `add sp, #0xa0`
    unsafe {
        // SAFETY: every offset stays within the caller-provided stack region.
        let w = |off: usize, v: u64| core::ptr::write((ctx + off) as *mut u64, v);
        w(0x00, entry as *const () as u64); // x19 = entry (trampoline `blr x19`)
        let mut off = 0x08;
        while off < 0xa0 {
            w(off, 0); // x20..x28, x29, x30, d8..d15 (x30 overwritten next)
            off += 8;
        }
        w(0x58, trampoline as *const () as u64); // x30 = trampoline (first-resume return)
    }
    ctx as *mut u8
}
