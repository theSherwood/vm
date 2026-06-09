//! x86-64 **Windows (MS x64)** register/stack switch.
//!
//! Same `boost.context` `fcontext` model as the SysV switch, but the MS x64 ABI forces three
//! differences, each handled here:
//!
//! 1. **`Transfer` is returned via a hidden pointer (sret), not `rax:rdx`.** A 16-byte struct return
//!    means the caller passes the result slot's address in `rcx`, then `to` in `rdx`, `data` in `r8`.
//!    `jump` must save that `rcx` in the saved context so the *other* side, when resumed, can fill its
//!    own caller's slot — so each context carries its own sret pointer.
//! 2. **More callee-saved state:** the MS x64 callee-saved set adds `rdi`/`rsi` **and `xmm6`–`xmm15`**
//!    (plus we preserve MXCSR + the x87 control word).
//! 3. **TEB stack fields.** Windows keeps the active stack's bounds in the TEB
//!    (`StackBase`/`StackLimit`/`DeallocationStack`); SEH dispatch and stack-overflow detection read
//!    them. Running guest code on a fiber stack while the TEB still describes the OS-thread stack would
//!    break exception delivery (our detect-and-kill VEH guard included), so every switch swaps those
//!    three TEB fields along with the registers. `make` seeds them from the new stack's bounds.
//!
//! The guard-paged control [`Stack`](crate::stack::Stack) is OS-specific and lives in `stack_windows.rs`.

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

// Saved-context frame layout (offsets from the saved stack pointer). The frame is 0x110 bytes; the
// switched-in side restores from it and `ret`s to `[ctx + 0x110]`:
//
//   0x00 r12   0x08 r13   0x10 r14   0x18 r15
//   0x20 rdi   0x28 rsi   0x30 rbx   0x38 rbp
//   0x40 rcx (this context's sret pointer)
//   0x48 TEB.StackBase   0x50 TEB.StackLimit   0x58 TEB.DeallocationStack
//   0x60 MXCSR(4) 0x64 x87 control word(2)
//   0x70..0x100 xmm6..xmm15 (movups, no alignment requirement)
//   0x110 return address
//
// TEB lives at gs:[0x30]; within it StackBase = +0x08, StackLimit = +0x10,
// DeallocationStack = +0x1478 (x64).

/// Switch to the context `to`, passing `data`. Returns once someone switches *back* to us.
///
/// # Safety
/// `to` must be a live context from [`make`] (not yet finished) or a `Transfer::fctx` from a prior
/// switch whose backing [`Stack`](crate::stack::Stack) is still alive.
#[unsafe(naked)]
pub unsafe extern "C" fn jump(to: *mut u8, data: u64) -> Transfer {
    // MS x64 + sret: rcx = &Transfer return slot, rdx = `to`, r8 = `data`.
    naked_asm!(
        "sub rsp, 0x110",
        // --- save our callee-saved GP regs + our sret pointer ---
        "mov [rsp + 0x00], r12",
        "mov [rsp + 0x08], r13",
        "mov [rsp + 0x10], r14",
        "mov [rsp + 0x18], r15",
        "mov [rsp + 0x20], rdi",
        "mov [rsp + 0x28], rsi",
        "mov [rsp + 0x30], rbx",
        "mov [rsp + 0x38], rbp",
        "mov [rsp + 0x40], rcx",
        // --- save our TEB stack fields ---
        "mov r10, gs:[0x30]",
        "mov rax, [r10 + 0x08]",
        "mov [rsp + 0x48], rax",
        "mov rax, [r10 + 0x10]",
        "mov [rsp + 0x50], rax",
        "mov rax, [r10 + 0x1478]",
        "mov [rsp + 0x58], rax",
        // --- save FP control + xmm6..xmm15 ---
        "stmxcsr [rsp + 0x60]",
        "fnstcw [rsp + 0x64]",
        "movups [rsp + 0x70], xmm6",
        "movups [rsp + 0x80], xmm7",
        "movups [rsp + 0x90], xmm8",
        "movups [rsp + 0xA0], xmm9",
        "movups [rsp + 0xB0], xmm10",
        "movups [rsp + 0xC0], xmm11",
        "movups [rsp + 0xD0], xmm12",
        "movups [rsp + 0xE0], xmm13",
        "movups [rsp + 0xF0], xmm14",
        "movups [rsp + 0x100], xmm15",
        // our context pointer (to hand back as Transfer.fctx)
        "mov rax, rsp",
        // --- switch to the target stack ---
        "mov rsp, rdx",
        // --- restore the target's GP regs + its sret pointer ---
        "mov r12, [rsp + 0x00]",
        "mov r13, [rsp + 0x08]",
        "mov r14, [rsp + 0x10]",
        "mov r15, [rsp + 0x18]",
        "mov rdi, [rsp + 0x20]",
        "mov rsi, [rsp + 0x28]",
        "mov rbx, [rsp + 0x30]",
        "mov rbp, [rsp + 0x38]",
        "mov rcx, [rsp + 0x40]",
        // --- restore the target's TEB stack fields ---
        "mov r10, gs:[0x30]",
        "mov r9, [rsp + 0x48]",
        "mov [r10 + 0x08], r9",
        "mov r9, [rsp + 0x50]",
        "mov [r10 + 0x10], r9",
        "mov r9, [rsp + 0x58]",
        "mov [r10 + 0x1478], r9",
        // --- restore FP control + xmm ---
        "ldmxcsr [rsp + 0x60]",
        "fldcw [rsp + 0x64]",
        "movups xmm6, [rsp + 0x70]",
        "movups xmm7, [rsp + 0x80]",
        "movups xmm8, [rsp + 0x90]",
        "movups xmm9, [rsp + 0xA0]",
        "movups xmm10, [rsp + 0xB0]",
        "movups xmm11, [rsp + 0xC0]",
        "movups xmm12, [rsp + 0xD0]",
        "movups xmm13, [rsp + 0xE0]",
        "movups xmm14, [rsp + 0xF0]",
        "movups xmm15, [rsp + 0x100]",
        // --- fill the target's Transfer slot: {fctx = our ctx (rax), data = r8} ---
        "mov [rcx + 0x00], rax",
        "mov [rcx + 0x08], r8",
        "mov rax, rcx", // sret convention: return the result-slot pointer
        "add rsp, 0x110",
        "ret",
    )
}

/// First-resume trampoline. The fresh context's `ret` lands here with `rax` = the `Transfer*` that
/// `jump` just filled (the scratch slot `make` parked) and `r12` = the entry fn. MS x64 passes a
/// 16-byte struct argument **by pointer in rcx**, so hand `rax` straight to `entry`.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() {
    naked_asm!(
        "mov rcx, rax", // Transfer* argument
        "and rsp, -16", // 16-align, then reserve shadow space
        "sub rsp, 0x20",
        "call r12", // entry(Transfer)
        "ud2",
    )
}

/// Lay out the fresh stack so the first [`jump`] into the returned context begins executing `entry`.
///
/// # Safety
/// `stack` must be a live, writable control stack.
pub unsafe fn make(stack: &crate::stack::Stack, entry: Entry) -> *mut u8 {
    let top = stack.top() as usize;
    let base = stack.base_ptr() as usize;
    let limit = stack.limit_ptr() as usize;
    // A 16-byte Transfer scratch just below the top (the first `jump` fills it; the trampoline passes
    // its address to `entry`). The context frame sits below the scratch.
    let scratch = (top & !0xf) - 16;
    let ctx = scratch - 8 - 0x110; // scratch-8 is the return-address slot (= ctx + 0x110)
    unsafe {
        // SAFETY: every offset stays within the caller-provided stack region.
        let w = |off: usize, v: u64| core::ptr::write((ctx + off) as *mut u64, v);
        w(0x00, entry as *const () as u64); // r12 = entry (trampoline `call r12`)
        w(0x08, 0); // r13
        w(0x10, 0); // r14
        w(0x18, 0); // r15
        w(0x20, 0); // rdi
        w(0x28, 0); // rsi
        w(0x30, 0); // rbx
        w(0x38, 0); // rbp
        w(0x40, scratch as u64); // rcx = &scratch Transfer
        w(0x48, top as u64); // TEB.StackBase
        w(0x50, limit as u64); // TEB.StackLimit
        w(0x58, base as u64); // TEB.DeallocationStack
        core::ptr::write((ctx + 0x60) as *mut u32, 0x0000_1F80); // default MXCSR
        core::ptr::write((ctx + 0x64) as *mut u16, 0x037F); // default x87 control word
        let mut off = 0x70;
        while off < 0x110 {
            w(off, 0); // xmm6..xmm15 zeroed
            off += 8;
        }
        w(0x110, trampoline as *const () as u64); // return address → trampoline on first resume
    }
    ctx as *mut u8
}
