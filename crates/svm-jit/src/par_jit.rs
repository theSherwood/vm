//! Real-fiber + JIT integration for the parallel executor (§12, part 4 step 2b): a [`par::Task`] that
//! runs JITted guest code on an `svm-fiber` stack, the `thread.spawn`/`thread.join` thunks for the
//! parallel mode, and the **current-vCPU thread-local** that connects them.
//!
//! Per-worker detect-and-kill: each `fiber.resume` runs under the §5 guard ([`mem::run_guarded_range`])
//! so a guest memory fault on any worker `siglongjmp`s back out of the resume (the fiber stack is
//! abandoned — the domain is being killed); the worker then sets the trap cell + requests pool
//! shutdown. The guard's recovery state is thread-local, so workers arm independently.
//!
//! Unlike a `cont.resume` chain, threads never resume one another, so the running vCPU is just a
//! per-OS-thread fact — a `thread_local`. A thunk called from guest code reads it to find the running
//! vCPU's `Yielder` (to suspend) and block slot. Lock discipline lives in `par`: a thunk locks the
//! scheduler only briefly and never across the suspend.

use crate::fiber_rt::FiberCallTramp;
use crate::par::{Shared, Step, Task};
use crate::{mem, FnEntry, TrapKind};
use std::cell::Cell;
use std::sync::Arc;
use svm_fiber::{Fiber, State, Yielder};

/// Per-vCPU control-stack size (guard-paged by `svm-fiber`).
const FIBER_STACK: usize = 1 << 20;

/// Max concurrently-live vCPUs (anti-bomb; matches the interpreter / cooperative scheduler).
const MAX_VCPUS: usize = 1 << 16;

/// Why a thunk suspended the running fiber back to its worker.
enum Block {
    Join(usize),
    Wait { key: u64, deadline: u64 },
}

/// The per-run constants every vCPU needs to call guest code / spawn children — copied into each
/// child's [`Ctx`] (they are constant for the whole run).
#[derive(Clone, Copy)]
struct Env {
    mem_base: u64,
    fn_table_base: u64,
    trap_out: *mut i64,
    call_tramp: FiberCallTramp,
    fault_lo: usize,
    fault_hi: usize,
}

/// The running vCPU's context, reached by the `thread.*` thunks via [`CURRENT`].
struct Ctx {
    /// Set by the fiber body at entry — how a blocking thunk suspends this vCPU back to its worker.
    yielder: *const Yielder,
    /// Set by a thunk just before it suspends; read by [`FiberTask::run`] after the fiber yields.
    block: Option<Block>,
    env: Env,
}

thread_local! {
    /// The vCPU currently running on this worker (null between resumes); the `thread.*` thunks read it.
    static CURRENT: Cell<*mut Ctx> = const { Cell::new(std::ptr::null_mut()) };
}

/// What a vCPU's fiber does when first resumed — run the root entry, or a spawned thread entry.
enum Body {
    /// The guest entry (`main`), via its buffer-ABI trampoline; results go to `results`.
    Root {
        entry_code: *const u8,
        args: *const i64,
        results: *mut i64,
    },
    /// A spawned thread entry `code(sp, arg)` via the shared call-trampoline.
    Child { code: u64, sp: u64, arg: u64 },
}

/// A green-thread vCPU: an `svm-fiber` running JITted guest code, plus its [`Ctx`] (kept at a stable
/// address so the thread-local can point at it while running).
struct FiberTask {
    fiber: Box<Fiber>,
    ctx: Box<Ctx>,
}

// SAFETY: a `FiberTask` is only ever run by one worker at a time (the `par` core takes it out of its
// slot to run it, and a re-entrant resume of the same vCPU is impossible). Its raw pointers refer to
// the run's shared window / trap cell (valid for the whole run) and its own fiber's `Yielder` (set per
// resume on the running worker), so moving a *parked* task between workers is sound — same contract as
// `svm_fiber::Fiber: Send`.
unsafe impl Send for FiberTask {}

/// In/out cell smuggled through the `Entry`-shaped [`fiber_resume_entry`] so the guarded runner can
/// drive `fiber.resume` (and a fault can `longjmp` past it).
struct ResumeCall {
    fiber: *mut Fiber,
    resume_val: u64,
    out: Out,
}
enum Out {
    Yield,
    Done(u64),
}

/// `Entry`-shaped shim: resume the fiber named by `a` (a `*mut ResumeCall`). Reusing the existing
/// `Entry` ABI lets us run it under `mem::run_guarded_range` without a new C shim.
extern "C" fn fiber_resume_entry(
    a: *const i64,
    _r: *mut i64,
    _m: *mut u8,
    _t: *const core::ffi::c_void,
    _tc: *mut i64,
) {
    // SAFETY: `a` is the `&mut ResumeCall` we passed to the guarded runner; the fiber is live.
    unsafe {
        let rc = a as *mut ResumeCall;
        let st = (*(*rc).fiber).resume((*rc).resume_val);
        (*rc).out = match st {
            State::Yielded(_) => Out::Yield,
            State::Complete(v) => Out::Done(v),
        };
    }
}

impl Task for FiberTask {
    fn run(&mut self, resume_val: i64, shared: &Arc<Shared>) -> Step {
        self.ctx.block = None;
        let ctx_ptr: *mut Ctx = &mut *self.ctx;
        let (lo, hi) = (self.ctx.env.fault_lo, self.ctx.env.fault_hi);
        let trap_out = self.ctx.env.trap_out;

        let mut rc = ResumeCall {
            fiber: &mut *self.fiber,
            resume_val: resume_val as u64,
            out: Out::Yield,
        };
        // Run the resume under this worker's guard, with the running vCPU published in the thread-local.
        let prev = CURRENT.with(|c| c.replace(ctx_ptr));
        // SAFETY: `fiber_resume_entry` honours the Entry ABI; `rc` outlives the call; a guest fault in
        // `[lo,hi)` unwinds back here (the fiber stack is abandoned — the domain is being killed).
        let faulted = unsafe {
            mem::run_guarded_range(
                fiber_resume_entry as *const () as *const u8,
                &mut rc as *mut ResumeCall as *const i64,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                lo,
                hi,
            )
        };
        CURRENT.with(|c| c.set(prev));

        if faulted {
            // SAFETY: `trap_out` is the run's live trap cell.
            unsafe { *trap_out = TrapKind::MemoryFault as i64 };
            shared.request_shutdown();
            return Step::Done(0);
        }
        // A non-memory trap (DivByZero, etc.) set the cell from JITted code: tear the pool down too.
        // SAFETY: live trap cell.
        if unsafe { *trap_out } != 0 {
            shared.request_shutdown();
            return Step::Done(0);
        }
        match rc.out {
            Out::Done(v) => Step::Done(v as i64),
            Out::Yield => match self.ctx.block.take() {
                Some(Block::Join(child)) => Step::Join(child),
                Some(Block::Wait { key, deadline }) => Step::Wait { key, deadline },
                // A bare yield with no recorded block shouldn't happen for threads; treat as done(0).
                None => Step::Done(0),
            },
        }
    }
}

/// Build a vCPU fiber for `body` with context `env`. The fiber body publishes its `Yielder` into the
/// `Ctx` at entry (so thunks can suspend it), then runs the guest code.
fn make_task(env: Env, body: Body) -> FiberTask {
    let mut ctx = Box::new(Ctx {
        yielder: std::ptr::null(),
        block: None,
        env,
    });
    let ctx_ptr: *mut Ctx = &mut *ctx;
    let fiber = Fiber::new(FIBER_STACK, move |y: &Yielder, _arg: u64| -> u64 {
        // SAFETY: `ctx_ptr` is this task's stable `Ctx`, live for the fiber's whole life.
        unsafe { (*ctx_ptr).yielder = y as *const Yielder };
        match body {
            Body::Root {
                entry_code,
                args,
                results,
            } => {
                // SAFETY: `entry_code` is the finalized buffer-ABI entry trampoline (Entry shape).
                let entry: crate::EntryTramp = unsafe { std::mem::transmute(entry_code) };
                entry(
                    args,
                    results,
                    env.mem_base as *mut u8,
                    env.fn_table_base as *const core::ffi::c_void,
                    env.trap_out,
                );
                0
            }
            Body::Child { code, sp, arg } => (env.call_tramp)(
                code,
                env.mem_base,
                env.fn_table_base,
                env.trap_out as u64,
                sp,
                arg,
            ),
        }
    });
    FiberTask {
        fiber: Box::new(fiber),
        ctx,
    }
}

/// Spawn the root vCPU (the guest entry) into `shared` and run the pool on `workers` threads. Called
/// by `run_inner` for the parallel mode; returns once the run finishes (root's value is in `results`).
///
/// # Safety
/// All pointers must outlive the call; `shared`'s address must be the one baked into the JITted
/// `thread.*` thunks. The caller installs the guard and keeps the window/module/buffers alive.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_root(
    shared: &Arc<Shared>,
    workers: usize,
    entry_code: *const u8,
    args: *const i64,
    results: *mut i64,
    mem_base: u64,
    fn_table_base: u64,
    trap_out: *mut i64,
    call_tramp: FiberCallTramp,
    fault: (usize, usize),
) {
    let env = Env {
        mem_base,
        fn_table_base,
        trap_out,
        call_tramp,
        fault_lo: fault.0,
        fault_hi: fault.1,
    };
    let root = make_task(
        env,
        Body::Root {
            entry_code,
            args,
            results,
        },
    );
    crate::par::run_pool(shared, workers, Box::new(root));
}

/// Read the running vCPU's env (the per-run constants), or `None` outside a resume.
fn current_env() -> Option<Env> {
    let p = CURRENT.with(|c| c.get());
    if p.is_null() {
        None
    } else {
        // SAFETY: set by `FiberTask::run` for the duration of the resume.
        Some(unsafe { (*p).env })
    }
}

/// `thread.spawn` thunk (parallel mode): start a new vCPU running `funcs[func_idx](sp, arg)` and
/// return its `i32` handle; traps (`ThreadFault`, `-1`) on a thread-bomb.
///
/// # Safety
/// `sched` is the run's live `par::Shared`; the other args are the threaded context.
pub(crate) unsafe extern "C" fn thread_spawn(
    sched: *const Shared,
    _mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
    func_idx: u32,
    sp: u64,
    arg: u64,
) -> i32 {
    let shared = &*sched;
    if shared.task_count() >= MAX_VCPUS {
        *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
        return -1;
    }
    let env = current_env().expect("thread.spawn outside a vCPU");
    // Resolve the entry code from the function table (compile-time index, valid).
    let entry = (fn_table_base as *const FnEntry).add(func_idx as usize);
    let code = (*entry).code;
    let child = make_task(env, Body::Child { code, sp, arg });
    shared.spawn(Box::new(child)) as i32
}

/// `thread.join` thunk (parallel mode): block until vCPU `handle` finishes and return its `i64`
/// result. A forged / out-of-range handle traps (`ThreadFault`). The handle is masked into the vCPU
/// table like a capability handle, so a forged one is inert.
///
/// # Safety
/// `sched` is the run's live `par::Shared`; `trap_out` is the live trap cell.
pub(crate) unsafe extern "C" fn thread_join(
    sched: *const Shared,
    handle: i32,
    trap_out: u64,
) -> i64 {
    let shared = &*sched;
    let n = shared.task_count();
    let mask = if n == 0 { 0 } else { n.next_power_of_two() - 1 };
    let slot = (handle as u32 as usize) & mask;
    if slot >= n {
        *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
        return 0;
    }
    let ctx = CURRENT.with(|c| c.get());
    if ctx.is_null() {
        *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
        return 0;
    }
    loop {
        if let Some(r) = shared.result_of(slot) {
            return r;
        }
        // Record the block and suspend back to the worker (which parks us until `slot` completes).
        (*ctx).block = Some(Block::Join(slot));
        let yielder = (*ctx).yielder;
        (*yielder).suspend(0);
        // Resumed: loop and re-check the result.
    }
}

/// The low `width`-byte mask (`width` ∈ {1,2,4,8}).
fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Read the `width`-byte value at physical address `phys` (the lowering's alignment guard ensures it
/// is aligned).
///
/// # Safety
/// `phys` points at `width` readable bytes in the guest window.
unsafe fn read_phys(phys: u64, width: u32) -> u64 {
    match width {
        1 => *(phys as *const u8) as u64,
        2 => *(phys as *const u16) as u64,
        4 => *(phys as *const u32) as u64,
        _ => *(phys as *const u64),
    }
}

/// `<ty>.atomic.wait` thunk (parallel mode): if the `width`-byte value at confined `phys` still equals
/// `expected`, park the running vCPU on `phys` until a notify or `timeout` ns elapse (`< 0` = forever,
/// fired only at pool quiescence). Returns the `i32` status (woken / not-equal / timed-out). Spurious
/// wakeups are spec-allowed, so a single suspend (no internal re-check loop) is correct.
///
/// # Safety
/// `sched` is the run's live `par::Shared`; `phys` points at `width` readable guest bytes.
pub(crate) unsafe extern "C" fn thread_wait(
    sched: *const Shared,
    phys: u64,
    expected: u64,
    width: u32,
    timeout: i64,
) -> i32 {
    let mask = width_mask(width);
    if read_phys(phys, width) & mask != expected & mask {
        return 1; // WAIT_NOT_EQUAL
    }
    let ctx = CURRENT.with(|c| c.get());
    if ctx.is_null() {
        return crate::par::WAIT_TIMED_OUT as i32;
    }
    let deadline = if timeout < 0 {
        u64::MAX
    } else {
        timeout as u64
    };
    (*ctx).block = Some(Block::Wait {
        key: phys,
        deadline,
    });
    let yielder = (*ctx).yielder;
    let _ = sched; // (the worker parks us via the `Step::Wait`; `sched` kept for symmetry)
                   // Suspend; the worker resumes us with the wake status (WAIT_WOKEN / WAIT_TIMED_OUT) as resume_val.
    (*yielder).suspend(0) as i32
}

/// `atomic.notify` thunk (parallel mode): wake up to `count` vCPUs parked on confined `phys`; returns
/// the `i32` count woken.
///
/// # Safety
/// `sched` is the run's live `par::Shared`.
pub(crate) unsafe extern "C" fn thread_notify(sched: *const Shared, phys: u64, count: i32) -> i32 {
    (*sched).notify(phys, count.max(0) as u32) as i32
}
