//! The JIT host side of the §14 **`Instantiator`** capability — VM-in-VM nesting. A guest holding an
//! `Instantiator` `instantiate`s a child confined to a power-of-two sub-window of its own window and
//! `join`s it. Unlike the interpreter (which spawns a child vCPU on its M:N executor), the JIT bakes
//! confinement into machine code, so a child confined to a *different* sub-window needs its own
//! compilation — "**nesting cost is paid at setup, not at runtime**" (§14): [`instantiate`] re-compiles
//! the child entry with the child's `mask`/`sub_base` ([`crate::compile_child_and_run`]) and runs it
//! over the **parent's live window** (so the parent intrinsically sees the child's writes — the §14
//! superset), under the caller's already-installed detect-and-kill guard.
//!
//! Authority lives in the host capability table (the same `Host` the interpreter uses): `instantiate`
//! resolves its `Instantiator` handle through the run's `cap.call` thunk (op 0 → the carve range
//! `[base, base+size)`), so a forged/wrong handle is an inert `CapFault` exactly as for any cap. The
//! child gets an **empty powerbox** for now (an inert `cap.call`); attenuated child caps + recursion +
//! "park only the calling fiber" (vs. today's synchronous run-at-`instantiate`) are follow-ups.

use crate::{CapThunk, TrapKind};
use std::sync::Mutex;
use svm_ir::{Func, FuncIdx};

/// Negative-errno an out-of-range carve returns (matches the interpreter's `EINVAL`, §3e D42).
const EINVAL: i64 = -22;

/// One spawned child's outcome: its `i64` result and trap cell (`0` = clean), plus whether it has
/// been `join`ed (a second join is inert — `ThreadFault`, matching the interpreter).
#[derive(Clone, Copy)]
struct Child {
    result: i64,
    trap: i64,
    joined: bool,
}

/// The per-run §14 nesting runtime, baked into the module's `Instantiator` `cap.call` sites. Holds
/// what compiling + running a child needs: the module's functions, the run's `cap.call` thunk/ctx
/// (to resolve an `Instantiator` handle's authority), and — supplied post-finalize via [`set_env`] —
/// the live window's detect-and-kill fault range. Children run synchronously at `instantiate` and
/// their outcomes are stashed for `join`.
pub(crate) struct Nursery {
    funcs: std::sync::Arc<[Func]>,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    children: Mutex<Vec<Child>>,
}

// SAFETY: the raw `cap_ctx` is the run's host pointer, valid for the whole run; the `Nursery` is
// only ever used on the run's threads while that host (and window) are alive. The interior tables
// are `Mutex`-guarded. (A child runs synchronously on the calling thread today, so there is in fact
// no cross-thread sharing yet; the bounds keep the door open for concurrent children later.)
unsafe impl Send for Nursery {}
unsafe impl Sync for Nursery {}

impl Nursery {
    pub(crate) fn new(
        funcs: std::sync::Arc<[Func]>,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
    ) -> Nursery {
        Nursery {
            funcs,
            cap_thunk,
            cap_ctx,
            children: Mutex::new(Vec::new()),
        }
    }

    /// Resolve `handle` as this domain's `Instantiator` via the run's `cap.call` thunk, returning its
    /// carve range `[base, base+size)`. `None` (and `*trap_out` set) for a forged/closed/wrong handle.
    unsafe fn resolve(&self, mem_base: u64, handle: i32, trap_out: *mut i64) -> Option<(u64, u64)> {
        let mut out = [0i64; 2];
        // op 0 on an `Instantiator` binding returns `[base, size]` (see `cap_dispatch_slots`); a bad
        // handle sets `*trap_out` to a `CapFault` and we propagate by returning `None`.
        (self.cap_thunk)(
            self.cap_ctx,
            mem_base as *mut u8,
            0,
            0,
            svm_ir_iface_instantiator(),
            0,
            handle,
            core::ptr::null(),
            0,
            out.as_mut_ptr(),
            out.len() as u64,
            trap_out,
        );
        if unsafe { *trap_out } != 0 {
            return None;
        }
        Some((out[0] as u64, out[1] as u64))
    }
}

/// The `Instantiator` interface id (§3e), kept in lockstep with `svm_interp::iface::INSTANTIATOR`.
/// (`svm-jit` does not depend on `svm-interp`; the host dispatch on the other side checks the same
/// constant, and the cross-backend tests pin them equal.)
#[inline]
fn svm_ir_iface_instantiator() -> u32 {
    6
}

/// `instantiate(handle, entry, off, size_log2, fuel) -> child_handle` — the §14 nesting op. Resolves
/// the holder's carve range, validates the requested power-of-two sub-window fits within it
/// (`-EINVAL` otherwise), then **re-compiles** the child entry confined to `[base+off, …+2^size_log2)`
/// and runs it over the parent's live window (`mem_base`), stashing its outcome for `join`. Returns a
/// child handle (a table index), or `-EINVAL`. A child that cannot be compiled (e.g. it uses §12
/// fibers/threads, unsupported for a JIT child today) sets `*trap_out` to a `CapFault`.
///
/// # Safety
/// Called from JIT'd code with `rt` the baked [`Nursery`], `mem_base` the live parent window base, and
/// `trap_out` the run's trap cell. All must be valid for the call (the JIT lowering guarantees it).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate(
    rt: *const Nursery,
    mem_base: u64,
    handle: i32,
    entry: i64,
    off: i64,
    size_log2: i64,
    _fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };

    // The carve must be a power-of-two-aligned sub-window within `[0, size)` — a child can only get
    // what the holder sub-allocates (§14/D19). Bad entry index / size / alignment ⇒ `-EINVAL`.
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
        && (entry as usize) < rt.funcs.len();
    if !fits {
        return EINVAL as i32;
    }

    // The child entry takes its starter caps as `i64` args; with an empty powerbox today they are
    // unused, so pass zeros of the right arity (the entry is a fixed `(i64[, i64]) -> i64`).
    let nargs = rt.funcs[entry as usize].params.len();
    let args = vec![0i64; nargs];

    // Re-compile the child as a top-level guest over its own window, seeded from the parent's
    // sub-region `[base+off, … + child_size)` and copied back on completion (the §14 superset).
    let outcome = crate::compile_child_and_run(
        &rt.funcs,
        entry as FuncIdx,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        &args,
    );
    let (result, trap) = match outcome {
        Ok(rt) => rt,
        Err(_) => {
            // A child we cannot compile (fibers/threads, or a backend error) is a CapFault, not a
            // silent success — the guest learns its nesting request was refused.
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };

    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    children.push(Child {
        result,
        trap,
        joined: false,
    });
    (children.len() - 1) as i32
}

/// `join(child_handle) -> result` — block on the child's completion (it already ran synchronously at
/// `instantiate` today) and return its `i64` result, propagating a child trap as the parent's
/// (`*trap_out`). A forged / already-joined handle is inert (a `CapFault`), matching the interpreter's
/// once-only join.
///
/// # Safety
/// As [`instantiate`]: `rt`/`trap_out` are the baked nursery + run trap cell, valid for the call.
pub(crate) unsafe extern "C" fn join(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i64 {
    let rt = &*rt;
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = handle as usize;
    match children.get_mut(slot) {
        Some(c) if !c.joined => {
            c.joined = true;
            if c.trap != 0 {
                *trap_out = c.trap; // a child trap propagates to the parent on join
                0
            } else {
                c.result
            }
        }
        _ => {
            *trap_out = TrapKind::CapFault as i64; // forged or already-joined handle
            0
        }
    }
}
