//! Host-side `setjmp`/`longjmp` runtime for the JIT — **Option B** (LLVM.md §"JIT `longjmp`").
//!
//! The JIT lowers [`Inst::SetJmp`]/[`Inst::LongJmp`] to libc `_setjmp`/`_longjmp` called **inline in
//! the guest JIT frame** — the very frame `longjmp` returns to — with the `jmp_buf` held **host-side**
//! in this per-run table, keyed by the guest `jmp_buf` window address. The host `jmp_buf` holds the
//! native stack pointer / return address / callee-saved registers; it MUST NOT live in the guest
//! window (leaking those defeats ASLR — the guest address is only the *key*). This mirrors the
//! interpreters' per-vCPU `setjmp_points` map (`svm-interp`'s `SetJmpPoint` / `ByteSetJmp`).
//!
//! **The one correctness invariant.** `_setjmp` is emitted inline in the guest frame, never wrapped in
//! a Rust thunk that `setjmp`s and returns — that helper's frame would be gone by `longjmp` time, and
//! `longjmp` into a returned frame is UB. So only the *slot* alloc/lookup are host thunks
//! ([`rt_setjmp_slot`]/[`rt_setjmp_lookup`]); they merely return a pointer and their own frames
//! returning is harmless. The `SetJmp` lowering is therefore *two* calls — `rt_setjmp_slot` then an
//! inline `_setjmp(slot)`; `LongJmp` is `rt_setjmp_lookup` then an inline `_longjmp(slot, val)`.
//!
//! Unix-only (the `setjmp_rt` cfg = `fiber_rt && unix`): `libc` is a unix-only dependency here and the
//! plain `_setjmp`/`_longjmp` symbols are unix; Windows `setjmp` is SEH-coupled (a follow-on).

#![cfg(setjmp_rt)]

use crate::TrapKind;
use std::collections::BTreeMap;
use std::os::raw::{c_int, c_void};
use std::sync::Mutex;

extern "C" {
    /// libc `_setjmp`/`_longjmp` — the **non**-signal-mask variants (no `sigprocmask` save/restore),
    /// matching the interpreters' semantics. Real exported symbols in glibc/musl. Their addresses are
    /// baked into the `SetJmp`/`LongJmp` sites and called by `call_indirect` *inline in the guest
    /// frame* (so `_setjmp` saves that frame's SP and `_longjmp` returns into it).
    fn _setjmp(env: *mut c_void) -> c_int;
    fn _longjmp(env: *mut c_void, val: c_int) -> !;
}

/// Address of libc `_setjmp`, baked into each `SetJmp` site as the inline call target.
pub(crate) fn setjmp_addr() -> i64 {
    _setjmp as *const () as i64
}

/// Address of libc `_longjmp`, baked into each `LongJmp` site as the inline call target.
pub(crate) fn longjmp_addr() -> i64 {
    _longjmp as *const () as i64
}

/// One host-owned libc `jmp_buf`, over-aligned and generously sized (glibc/musl `jmp_buf` ≈ 200 B; the
/// 512 B here covers every gated target with margin). `Box`ed so its address is **stable** across table
/// growth — the inline `_setjmp` writes into this fixed location and a later `_longjmp` reads it back.
#[repr(C, align(16))]
struct JmpBufCell([u8; 512]);

impl JmpBufCell {
    fn boxed() -> Box<JmpBufCell> {
        Box::new(JmpBufCell([0u8; 512]))
    }
    fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr() as *mut c_void
    }
}

/// Per-run `setjmp` checkpoint table: guest `jmp_buf` window address → host `jmp_buf` storage. Stood up
/// only when the module uses `setjmp` (`module_uses_setjmp`), one per run; its stable address is baked
/// into the run's `SetJmp`/`LongJmp` sites (`SetjmpEnv` in `lib.rs`) and the table is owned by the
/// `CompiledModule`, so it lives exactly as long as the code can run.
///
/// Shared across the run's OS threads (a `thread.spawn`ed guest may also `setjmp`), so guarded by a
/// `Mutex`. The lock is held **only** to take/insert the slot pointer — released before the inline
/// `_setjmp`/`_longjmp` runs — and the `Box`ed cell's address is stable, so no lock is ever held across
/// the native-stack operation.
pub(crate) struct SetjmpRuntime {
    slots: Mutex<BTreeMap<u64, Box<JmpBufCell>>>,
}

impl SetjmpRuntime {
    pub(crate) fn new() -> SetjmpRuntime {
        SetjmpRuntime {
            slots: Mutex::new(BTreeMap::new()),
        }
    }
}

/// `rt_setjmp_slot(rt, guest_buf)` — return the stable host `jmp_buf` pointer for the guest buffer at
/// window address `guest_buf`, allocating it on the first `setjmp` to that address (a re-`setjmp` to the
/// same buffer reuses and overwrites the slot, matching the interpreters). The returned pointer is then
/// handed to an **inline** `_setjmp` in the guest frame. Infallible (allocation aborts on OOM, as any
/// `Box`).
///
/// # Safety
/// `rt` is the per-run [`SetjmpRuntime`] address baked into the site by the lowering; valid for the run.
pub(crate) unsafe extern "C" fn rt_setjmp_slot(
    rt: *mut SetjmpRuntime,
    guest_buf: u64,
) -> *mut c_void {
    let rt = &*rt;
    let mut slots = rt.slots.lock().unwrap_or_else(|e| e.into_inner());
    let cell = slots.entry(guest_buf).or_insert_with(JmpBufCell::boxed);
    cell.as_ptr()
}

/// `rt_setjmp_lookup(rt, guest_buf, trap_out)` — find the host `jmp_buf` for `guest_buf` (set by a prior
/// `setjmp`). On a **miss** — a stale/forged token, or a buffer never `setjmp`'d — write the
/// [`TrapKind::SetjmpFault`] marker into `*trap_out` and return null; the JIT's trap-propagate check
/// then bails *before* the (skipped) `_longjmp`, matching the interpreters' `Trap::Malformed`
/// (§3b totality). Never `_longjmp`s into a missing slot.
///
/// # Safety
/// `rt` and `trap_out` are valid for the run (baked / threaded by the lowering).
pub(crate) unsafe extern "C" fn rt_setjmp_lookup(
    rt: *mut SetjmpRuntime,
    guest_buf: u64,
    trap_out: u64,
) -> *mut c_void {
    let rt = &*rt;
    let slots = rt.slots.lock().unwrap_or_else(|e| e.into_inner());
    match slots.get(&guest_buf) {
        Some(cell) => cell.as_ptr(),
        None => {
            *(trap_out as *mut i64) = TrapKind::SetjmpFault as i64;
            core::ptr::null_mut()
        }
    }
}
