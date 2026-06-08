//! JIT fiber runtime (§12) — the host-side state and `extern "C"` thunks that let JITted code create,
//! resume, and suspend stackful fibers via [`svm_fiber`].
//!
//! The JIT lowers `cont.new`/`cont.resume`/`suspend` to indirect calls to the three thunks here
//! (passing `mem_base`/`fn_table_base`/`trap_out` from the threaded context, exactly like `cap.call`).
//! A fiber's body runs **JITted guest code** on its own native control stack: because Rust cannot call
//! the guest `Tail` calling convention directly, the body goes through a small generated CLIF
//! *call-trampoline* ([`FiberCallTramp`]) that `call_indirect`s the guest entry. A `suspend` from deep
//! within that guest code switches the whole native stack back to the resumer — the §3d two-stack
//! model in action.
//!
//! **Single-threaded / cooperative:** fibers (unlike `thread.spawn`, not yet in the JIT) never run
//! concurrently — exactly one is on the stack at a time. So the runtime's `Vec`s are touched only by
//! the one running OS thread, and the reentrancy (a resumed fiber calling back in to create/resume/
//! suspend) is sound via raw-pointer field access: no `&mut FiberRuntime` is ever held across a switch
//! (only a `*mut Fiber` to the boxed, address-stable fiber being resumed), and the per-fiber control
//! `chain` rejects re-entrant resume of an already-running fiber, so each `Fiber`'s `&mut` is exclusive.

use crate::{FnEntry, TrapKind};
use std::cell::Cell;
use svm_fiber::{Fiber, State, Yielder};

thread_local! {
    /// The fiber runtime of the computation currently running on this OS thread — the standalone root,
    /// or the vCPU a scheduler is resuming. The `cont.*` thunks read it, so **each vCPU has its own
    /// fiber table** and threads + fibers compose (a threaded module can use `cont.*` per vCPU). Null
    /// between resumes / when no fiber-capable computation is running.
    static CURRENT_RT: Cell<*mut FiberRuntime> = const { Cell::new(std::ptr::null_mut()) };
}

/// Publish the running computation's fiber runtime; returns the previous value to restore afterward.
/// Set by the standalone entry path and by each scheduler around a vCPU resume.
pub(crate) fn set_current(rt: *mut FiberRuntime) -> *mut FiberRuntime {
    CURRENT_RT.with(|c| c.replace(rt))
}

fn current() -> *mut FiberRuntime {
    CURRENT_RT.with(|c| c.get())
}

/// Max concurrently-allocated fibers per run (matches the interpreter's `MAX_FIBERS`): an anti-bomb
/// ceiling so a fiber-bomb traps (`FiberFault`) instead of exhausting host memory.
const MAX_FIBERS: usize = 1 << 16;

/// Per-fiber control-stack size (the out-of-band native stack, guard-paged by `svm-fiber`). 1 MiB of
/// reserved VA, committed on demand — cheap even with many fibers.
const FIBER_STACK: usize = 1 << 20;

/// The generated CLIF call-trampoline: `extern "C"` on the outside (callable from Rust), it
/// `call_indirect`s a guest fiber entry (`Tail` ABI `(mem_base, fn_table_base, trap_out, sp, arg) ->
/// i64`). One trampoline serves every fiber since all fiber entries share that signature (§12).
pub(crate) type FiberCallTramp = extern "C" fn(
    code: u64,
    mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
    sp: u64,
    arg: u64,
) -> u64;

/// Host-side fiber table + switch bookkeeping for one JIT run.
pub(crate) struct FiberRuntime {
    /// The fiber table; the handle a `cont.new` returns is an index here. `Box` so a fiber's address
    /// is stable across table growth (a `cont.new` from inside a running fiber may grow this).
    fibers: Vec<Option<Box<Fiber>>>,
    /// Handles currently on the resume chain — used to reject re-entrant resume of a running fiber
    /// (which would alias its `&mut`), matching the interpreter's `chain.contains` check.
    chain: Vec<usize>,
    /// The running fibers' `Yielder`s (same depth as `chain`); `suspend` switches via the top one.
    yielders: Vec<*const Yielder>,
    /// The generated call-trampoline address (filled in after the module is finalized).
    call_tramp: Option<FiberCallTramp>,
    /// The structural type id every fiber entry must have (`(i64 sp, i64 arg) -> i64`), checked at
    /// first resume against the funcref's table slot — a forged/wrong-type funcref traps there.
    fiber_type_id: u32,
    /// `next_pow2(nfuncs) - 1`, to mask a funcref into the function table.
    fn_table_mask: u64,
}

impl FiberRuntime {
    pub(crate) fn new(fiber_type_id: u32, fn_table_mask: u64) -> FiberRuntime {
        FiberRuntime {
            fibers: Vec::new(),
            chain: Vec::new(),
            yielders: Vec::new(),
            call_tramp: None,
            fiber_type_id,
            fn_table_mask,
        }
    }

    /// Record the finalized call-trampoline address (must be set before any fiber runs).
    pub(crate) fn set_call_tramp(&mut self, t: FiberCallTramp) {
        self.call_tramp = Some(t);
    }
}

/// Write `FiberFault` into the host trap cell (the JIT propagates it after the thunk returns).
///
/// # Safety
/// `trap_out` is the live `*mut i64` trap cell threaded from the call site.
unsafe fn fault(trap_out: u64) {
    *(trap_out as *mut i64) = TrapKind::FiberFault as i64;
}

// Reentrancy discipline (all three thunks): a running fiber may call back in (create/resume/suspend),
// so no `&mut FiberRuntime` is ever held across a stack switch — `&mut *rt` is taken only in short
// scopes that end *before* `resume`/`suspend`, and only a `*mut Fiber` (to an address-stable boxed
// fiber) crosses the switch. The `chain` re-entrancy check makes each fiber's `&mut` exclusive, and
// fibers live in separate heap allocations from the table, so a re-entrant `cont.new` growing the
// table never aliases a fiber being resumed.

/// `cont.new` thunk: allocate a suspended fiber that, on first resume, calls guest `funcref(sp, arg)`.
/// Returns the fiber handle (table index), or traps (`-1`) on a fiber-bomb.
///
/// # Safety
/// `fn_table_base`/`trap_out` are the threaded context. The running vCPU's fiber runtime is read from
/// the [`CURRENT_RT`] thread-local. The funcref is resolved (and type-checked) lazily on first resume,
/// matching the interpreter.
pub(crate) unsafe extern "C" fn fiber_new(
    mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
    funcref: i32,
    sp: u64,
) -> i32 {
    let rt = current();
    if rt.is_null() {
        fault(trap_out);
        return -1;
    }
    let (mask, type_id, call_tramp) = {
        let rt = &mut *rt;
        if rt.fibers.len() >= MAX_FIBERS {
            fault(trap_out);
            return -1;
        }
        (
            rt.fn_table_mask,
            rt.fiber_type_id,
            rt.call_tramp
                .expect("call-trampoline set before any fiber runs"),
        )
    };
    let rt_ptr: *mut FiberRuntime = rt;

    let fiber = Fiber::new(FIBER_STACK, move |y: &Yielder, arg: u64| -> u64 {
        // SAFETY: the fiber only runs while `rt` is alive (during the guarded run); each `&mut *rt`
        // here is momentary and single-threaded.
        unsafe {
            (*rt_ptr).yielders.push(y as *const Yielder);
            // Resolve + type-check the funcref now (first resume), like the interpreter.
            let slot = (funcref as u32 as usize) & (mask as usize);
            let entry = (fn_table_base as *const FnEntry).add(slot);
            let result = if (*entry).type_id != type_id {
                fault(trap_out);
                0u64
            } else {
                call_tramp((*entry).code, mem_base, fn_table_base, trap_out, sp, arg)
            };
            (*rt_ptr).yielders.pop();
            result
        }
    });

    let rt = &mut *rt;
    rt.fibers.push(Some(Box::new(fiber)));
    (rt.fibers.len() - 1) as i32
}

/// `cont.resume` thunk: switch into fiber `handle`, delivering `arg`; writes `*status_out` (0 =
/// suspended, 1 = returned) and returns the fiber's yielded/returned value. A forged / out-of-range /
/// already-running / finished handle traps (`FiberFault`), matching the interpreter.
///
/// # Safety
/// `status_out`/`trap_out` are live `*mut i64` cells. The running vCPU's runtime is [`CURRENT_RT`].
pub(crate) unsafe extern "C" fn fiber_resume(
    handle: i32,
    arg: i64,
    status_out: *mut i64,
    trap_out: u64,
) -> i64 {
    let rt = current();
    if rt.is_null() {
        fault(trap_out);
        *status_out = 1;
        return 0;
    }
    // Phase 1: validate + extract a raw fiber pointer, releasing the `&mut *rt` before the switch.
    let fib: *mut Fiber = {
        let rt = &mut *rt;
        let nfib = rt.fibers.len();
        let mask = if nfib == 0 {
            0
        } else {
            nfib.next_power_of_two() - 1
        };
        let slot = (handle as u32 as usize) & mask;
        if slot >= nfib || rt.chain.contains(&slot) {
            fault(trap_out);
            *status_out = 1;
            return 0;
        }
        match &mut rt.fibers[slot] {
            Some(b) if !b.is_done() => {
                rt.chain.push(slot);
                &mut **b as *mut Fiber
            }
            _ => {
                fault(trap_out);
                *status_out = 1;
                return 0;
            }
        }
    };
    // Phase 2: the switch (may reenter the runtime) — no `&mut *rt` held.
    let st = (*fib).resume(arg as u64);
    // Phase 3: pop the chain under a fresh short borrow.
    {
        let rt = &mut *rt;
        rt.chain.pop();
    }
    match st {
        State::Yielded(v) => {
            *status_out = 0;
            v as i64
        }
        State::Complete(v) => {
            *status_out = 1;
            v as i64
        }
    }
}

/// `suspend` thunk: hand `value` back to the resumer and return the next resume's `arg`. Suspending
/// with no running fiber (the root computation) traps (`FiberFault`).
///
/// # Safety
/// `trap_out` is the live trap cell. The running vCPU's runtime is read from [`CURRENT_RT`].
pub(crate) unsafe extern "C" fn fiber_suspend(value: i64, trap_out: u64) -> i64 {
    let rt = current();
    if rt.is_null() {
        fault(trap_out);
        return 0;
    }
    // pop-before-switch / push-after keeps the yielder stack consistent so a resumer reached by the
    // switch sees *its* yielder on top.
    let y = {
        let rt = &mut *rt;
        match rt.yielders.pop() {
            Some(y) => y,
            None => {
                fault(trap_out); // root computation cannot suspend
                return 0;
            }
        }
    };
    let r = (*y).suspend(value as u64);
    {
        let rt = &mut *rt;
        rt.yielders.push(y);
    }
    r as i64
}
