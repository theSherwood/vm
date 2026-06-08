//! Cooperative green-thread **scheduler core** for the JIT (§12) — the algorithm behind
//! `thread.spawn`/`thread.join`, built on the [`svm_fiber`] stack switch.
//!
//! A vCPU is a green thread: a [`Fiber`] running a guest thread-entry. The scheduler keeps a runnable
//! queue and drives one vCPU at a time (cooperative / single OS thread for now — true multi-core
//! workers are a later step); a vCPU runs until it **blocks** (`thread.join` on an unfinished child)
//! or finishes. Blocking suspends the vCPU's fiber back to the scheduler loop (a stack switch); when
//! the awaited child completes, its waiter is moved back to the runnable queue and resumed, picking up
//! right where it left off. This is exactly the interpreter's M:N model, executed on native stacks.
//!
//! This module is the backend-agnostic core: a vCPU body is any closure, so it is unit-tested here
//! with plain Rust bodies that call back in via [`spawn`]/[`join`]. The JIT wires it up by making a
//! body call the compiled guest entry (the fiber call-trampoline) and lowering `thread.spawn`/
//! `thread.join` to [`spawn`]/[`join`].
//!
//! **Reentrancy** (a running vCPU calls back in to spawn/join): identical discipline to `fiber_rt` —
//! no `&mut Sched` is ever held across a stack switch (only a `*mut Fiber` to an address-stable boxed
//! fiber crosses it), and short field borrows derived from the raw `*mut Sched` end before the switch.
//! Single OS thread, so the `Vec`s are never touched concurrently.

use crate::fiber_rt::FiberCallTramp;
use crate::{FnEntry, TrapKind};
use std::collections::VecDeque;
use svm_fiber::{Fiber, State, Yielder};

/// Max concurrently-live vCPUs per run (matches the interpreter's `MAX_VCPUS`): an anti-bomb ceiling
/// so a thread-bomb traps (`ThreadFault`) instead of exhausting host memory.
const MAX_VCPUS: usize = 1 << 16;

/// `<ty>.atomic.wait` status results (§12, matching the interpreter / wasm): woken by a notify, the
/// value did not equal `expected` (no wait), or timed out.
const WAIT_WOKEN: i32 = 0;
const WAIT_NOT_EQUAL: i32 = 1;
const WAIT_TIMED_OUT: i32 = 2;

/// A guest thread body: receives a [`Ctx`] (its scheduler handle, for nested spawn/join) and returns
/// the thread's `i64` result. `'static` so it can live in the heap-allocated fiber.
pub(crate) type Body = Box<dyn FnOnce(&Ctx) -> i64 + 'static>;

/// Why a vCPU suspended back to the scheduler.
enum Block {
    /// Waiting for child vCPU (task id) to finish.
    Join(usize),
    /// Parked on a futex `key` until a notify wakes it or the logical-clock `deadline` fires.
    Wait { key: u64, deadline: u64 },
}

/// One green thread.
struct VCpu {
    fiber: Box<Fiber>,
    /// The running fiber's `Yielder` (set by the body wrapper on first run) — how a blocking op
    /// suspends this vCPU back to the scheduler.
    yielder: *const Yielder,
    done: bool,
    result: i64,
    /// Whether some `join` has already consumed this vCPU's result (a re-join is inert).
    joined: bool,
    /// The value the scheduler delivers on the next resume (a wait status for a woken/timed-out
    /// waiter; `0` otherwise).
    resume_val: u64,
}

/// The cooperative scheduler: a fiber table + runnable queue + join parking.
pub(crate) struct Sched {
    vcpus: Vec<Option<VCpu>>,
    runnable: VecDeque<usize>,
    /// `(child, parent)` pairs: `parent` is parked until `child` finishes.
    join_waiters: Vec<(usize, usize)>,
    /// `(key, deadline, tid)` futex waiters, parked until a notify on `key` or the `deadline`.
    wait_waiters: Vec<(u64, u64, usize)>,
    /// Logical clock (ns), advanced only to fire the earliest `wait` deadline when nothing is
    /// runnable — so a run is a pure function of its schedule (matching the interpreter's `DetSched`).
    clock: u64,
    /// The currently running vCPU (valid while a fiber is resumed).
    cur: usize,
    /// Set by a blocking thunk on the running vCPU's stack just before it suspends; read by the loop.
    block: Option<Block>,
    /// Per-vCPU control-stack size.
    stack: usize,
    /// The generated call-trampoline used to invoke a JITted thread entry (`None` for the unit tests,
    /// whose vCPU bodies are plain Rust closures). Filled in after the module is finalized.
    call_tramp: Option<FiberCallTramp>,
    /// The run's host trap cell: when a vCPU trap sets it, the scheduler stops (the domain is killed).
    /// `null` in the unit tests (no JITted code → never set).
    trap_cell: *mut i64,
}

/// What a vCPU body uses to spawn children and join them (the cooperative-scheduler handle). Wraps the
/// raw scheduler pointer + this vCPU's identity; methods are the safe surface over the reentrant
/// `spawn`/`join` free functions.
pub(crate) struct Ctx {
    sched: *mut Sched,
    tid: usize,
    yielder: *const Yielder,
}

impl Ctx {
    /// Spawn a child vCPU running `body`; returns its handle (task id).
    pub(crate) fn spawn(&self, body: Body) -> usize {
        // SAFETY: single-threaded; `sched` is the live scheduler driving us.
        unsafe { spawn(self.sched, body) }
    }

    /// Join child `handle`: block until it finishes, then return its result. A forged / out-of-range
    /// / already-joined handle is inert → `Err`.
    pub(crate) fn join(&self, handle: usize) -> Result<i64, ()> {
        // SAFETY: single-threaded; `sched`/`yielder` are live for this vCPU.
        unsafe { join(self.sched, self.tid, self.yielder, handle) }
    }

    /// Park on futex `key` until a notify or `deadline` (logical clock); returns the wake status.
    /// (The value-compare a real `atomic.wait` does lives in the JIT thunk, not here.)
    pub(crate) fn wait(&self, key: u64, deadline: u64) -> i32 {
        // SAFETY: single-threaded; `sched`/`yielder` are live for this vCPU.
        unsafe { wait_park(self.sched, key, deadline, self.yielder) }
    }

    /// Wake up to `count` vCPUs parked on `key`; returns how many.
    pub(crate) fn notify(&self, key: u64, count: u32) -> u32 {
        // SAFETY: single-threaded; `sched` is live.
        unsafe { notify(self.sched, key, count) }
    }
}

impl Sched {
    pub(crate) fn new(stack: usize) -> Sched {
        Sched {
            vcpus: Vec::new(),
            runnable: VecDeque::new(),
            join_waiters: Vec::new(),
            wait_waiters: Vec::new(),
            clock: 0,
            cur: 0,
            block: None,
            stack,
            call_tramp: None,
            trap_cell: std::ptr::null_mut(),
        }
    }

    /// Record the JITted thread-entry call-trampoline (set before any vCPU runs).
    pub(crate) fn set_call_tramp(&mut self, t: FiberCallTramp) {
        self.call_tramp = Some(t);
    }

    /// Point the scheduler at the run's host trap cell, so it stops as soon as a vCPU traps.
    pub(crate) fn set_trap_cell(&mut self, tc: *mut i64) {
        self.trap_cell = tc;
    }

    /// Whether a vCPU has tripped the trap cell (the domain is being killed).
    fn trapped(&self) -> bool {
        // SAFETY: `trap_cell` is null (tests) or the live run trap cell.
        !self.trap_cell.is_null() && unsafe { *self.trap_cell } != 0
    }

    /// Spawn the root vCPU running `body`, drive the schedule to completion, and return the root's
    /// result (or `None` on a guest deadlock — nothing runnable but vCPUs still live).
    pub(crate) fn run(&mut self, body: Body) -> Option<i64> {
        let self_ptr: *mut Sched = self;
        // SAFETY: single-threaded driver; see the module reentrancy note.
        let root = unsafe { spawn(self_ptr, body) };
        loop {
            let tid = match self.runnable.pop_front() {
                Some(t) => t,
                None => {
                    // Nothing runnable: fire the earliest `wait` deadline (timeout), or stop (all done
                    // / a genuine guest deadlock with no waiters).
                    let Some(idx) =
                        (0..self.wait_waiters.len()).min_by_key(|&i| self.wait_waiters[i].1)
                    else {
                        break;
                    };
                    let (_, deadline, w) = self.wait_waiters.remove(idx);
                    self.clock = self.clock.max(deadline);
                    if let Some(v) = &mut self.vcpus[w] {
                        v.resume_val = WAIT_TIMED_OUT as u64;
                    }
                    self.runnable.push_back(w);
                    continue;
                }
            };
            self.cur = tid;
            // The value to deliver on resume (a wait status for a woken/timed-out waiter; else 0).
            let resume_val = self.vcpus[tid].as_ref().map_or(0, |v| v.resume_val);
            // Extract a raw fiber pointer and release the `vcpus` borrow before the switch, so a
            // re-entrant `spawn` may grow the table without aliasing the resumed fiber.
            let fib: *mut Fiber = match &mut self.vcpus[tid] {
                Some(v) if !v.fiber.is_done() => {
                    v.resume_val = 0;
                    &mut *v.fiber as *mut Fiber
                }
                _ => continue,
            };
            // SAFETY: `fib` is an address-stable boxed fiber; only one vCPU runs at a time.
            let st = unsafe { (*fib).resume(resume_val) };
            // A vCPU trap kills the whole domain: stop scheduling (remaining fibers are abandoned).
            if self.trapped() {
                break;
            }
            match st {
                State::Complete(_) => { /* completion bookkeeping ran in the body wrapper */ }
                State::Yielded(_) => match self.block.take() {
                    Some(Block::Join(child)) => {
                        let child_done = matches!(&self.vcpus[child], Some(v) if v.done);
                        if child_done {
                            self.runnable.push_back(tid); // result is ready; re-run to collect it
                        } else {
                            self.join_waiters.push((child, tid));
                        }
                    }
                    Some(Block::Wait { key, deadline }) => {
                        self.wait_waiters.push((key, deadline, tid));
                    }
                    None => self.runnable.push_back(tid), // a plain cooperative yield
                },
            }
        }
        self.vcpus[root].as_ref().map(|v| v.result)
    }
}

/// Spawn a vCPU running `body`, enqueue it, and return its handle.
///
/// # Safety
/// `s` is the live scheduler. Single-threaded; field borrows are momentary.
unsafe fn spawn(s: *mut Sched, body: Body) -> usize {
    let (tid, stack) = {
        let s = &mut *s;
        let tid = s.vcpus.len();
        s.vcpus.push(None);
        s.runnable.push_back(tid);
        (tid, s.stack)
    };
    let s_ptr: *mut Sched = s;
    let fiber = Fiber::new(stack, move |y: &Yielder, _arg: u64| -> u64 {
        // SAFETY: runs only while the scheduler is alive; momentary single-threaded field access.
        unsafe {
            // Register our yielder so blocking ops can suspend us.
            {
                let s = &mut *s_ptr;
                if let Some(v) = &mut s.vcpus[tid] {
                    v.yielder = y as *const Yielder;
                }
            }
            let ctx = Ctx {
                sched: s_ptr,
                tid,
                yielder: y as *const Yielder,
            };
            let result = body(&ctx);
            // Completion: record the result and wake anyone joining us.
            complete(s_ptr, tid, result);
        }
        0
    });
    {
        let s = &mut *s;
        s.vcpus[tid] = Some(VCpu {
            fiber: Box::new(fiber),
            yielder: std::ptr::null(),
            done: false,
            result: 0,
            joined: false,
            resume_val: 0,
        });
    }
    tid
}

/// Mark vCPU `tid` finished with `result` and move its join-waiters back to the runnable queue.
///
/// # Safety
/// `s` is the live scheduler; single-threaded.
unsafe fn complete(s: *mut Sched, tid: usize, result: i64) {
    let s = &mut *s;
    if let Some(v) = &mut s.vcpus[tid] {
        v.done = true;
        v.result = result;
    }
    let mut i = 0;
    while i < s.join_waiters.len() {
        if s.join_waiters[i].0 == tid {
            let (_, parent) = s.join_waiters.remove(i);
            s.runnable.push_back(parent);
        } else {
            i += 1;
        }
    }
}

/// Join child `handle` from the running vCPU `cur`: return its result if finished, else block (suspend
/// to the scheduler) until it is. A forged / out-of-range / already-joined handle is inert → `Err`.
///
/// # Safety
/// `s`/`yielder` are live for the running vCPU. Single-threaded; no `&mut Sched` held across suspend.
unsafe fn join(
    s: *mut Sched,
    _cur: usize,
    yielder: *const Yielder,
    handle: usize,
) -> Result<i64, ()> {
    loop {
        // Phase 1: short borrow to check state / arm the block, ending before any suspend.
        let ready = {
            let s = &mut *s;
            match s.vcpus.get_mut(handle) {
                Some(Some(v)) if v.done && !v.joined => {
                    v.joined = true;
                    Some(v.result)
                }
                Some(Some(v)) if !v.done && !v.joined => {
                    s.block = Some(Block::Join(handle));
                    None
                }
                _ => return Err(()), // forged / out of range / already joined
            }
        };
        match ready {
            Some(r) => return Ok(r),
            None => {
                // Phase 2: suspend to the scheduler; resumes when the child completes, then re-check.
                (*yielder).suspend(0);
            }
        }
    }
}

/// Park the running vCPU on futex `key` until a notify or the `deadline` fires; returns the wake
/// status (`WAIT_WOKEN` / `WAIT_TIMED_OUT`) the scheduler delivers on resume.
///
/// # Safety
/// `s`/`yielder` are live for the running vCPU. Single-threaded; no `&mut Sched` held across suspend.
unsafe fn wait_park(s: *mut Sched, key: u64, deadline: u64, yielder: *const Yielder) -> i32 {
    {
        let s = &mut *s;
        s.block = Some(Block::Wait { key, deadline });
    }
    // Suspend; the scheduler resumes us with the wake status as the resume value.
    (*yielder).suspend(0) as i32
}

/// Wake up to `count` vCPUs parked on futex `key` (in insertion order); returns how many were woken.
///
/// # Safety
/// `s` is the run's live scheduler. Single-threaded.
unsafe fn notify(s: *mut Sched, key: u64, count: u32) -> u32 {
    let s = &mut *s;
    let mut woken = 0u32;
    let mut i = 0;
    while woken < count && i < s.wait_waiters.len() {
        if s.wait_waiters[i].0 == key {
            let (_, _, tid) = s.wait_waiters.remove(i);
            if let Some(v) = &mut s.vcpus[tid] {
                v.resume_val = WAIT_WOKEN as u64;
            }
            s.runnable.push_back(tid);
            woken += 1;
        } else {
            i += 1;
        }
    }
    woken
}

/// The low `width`-byte mask (`width` ∈ {1,2,4,8}).
fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Read the `width`-byte value at physical address `phys` (the guard guarantees alignment).
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

// ===== JIT-facing thunks (called from JITted guest code; addresses baked in as constants) =====

/// `thread.spawn` thunk: start a new vCPU running JITted `funcs[func_idx](sp, arg)` over the shared
/// window, and return its `i32` handle. Traps (`ThreadFault`, `-1`) on a thread-bomb. `func` is a
/// compile-time index, so it indexes the function table directly (no masking).
///
/// # Safety
/// `s` is the run's live scheduler; `fn_table_base`/`trap_out` are the threaded context.
pub(crate) unsafe extern "C" fn thread_spawn(
    s: *mut Sched,
    mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
    func_idx: u32,
    sp: u64,
    arg: u64,
) -> i32 {
    let call_tramp = {
        let s = &mut *s;
        if s.vcpus.len() >= MAX_VCPUS {
            *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
            return -1;
        }
        s.call_tramp
            .expect("call-trampoline set before any vCPU runs")
    };
    // Resolve the entry's code pointer from the function table (func_idx is a valid module index).
    let entry = (fn_table_base as *const FnEntry).add(func_idx as usize);
    let code = (*entry).code;
    let body: Body = Box::new(move |_c: &Ctx| -> i64 {
        call_tramp(code, mem_base, fn_table_base, trap_out, sp, arg) as i64
    });
    spawn(s, body) as i32
}

/// `thread.join` thunk: block until vCPU `handle` finishes and return its `i64` result. A forged /
/// out-of-range / already-joined handle traps (`ThreadFault`). The handle is masked into the vCPU
/// table like a capability handle (§3c), so a forged handle is inert.
///
/// # Safety
/// `s` is the run's live scheduler; `trap_out` is the live trap cell.
pub(crate) unsafe extern "C" fn thread_join(s: *mut Sched, handle: i32, trap_out: u64) -> i64 {
    let (cur, yielder, slot, in_range) = {
        let s = &*s;
        let n = s.vcpus.len();
        let mask = if n == 0 { 0 } else { n.next_power_of_two() - 1 };
        let slot = (handle as u32 as usize) & mask;
        let yielder = s
            .vcpus
            .get(s.cur)
            .and_then(|v| v.as_ref())
            .map(|v| v.yielder)
            .unwrap_or(std::ptr::null());
        (s.cur, yielder, slot, slot < n)
    };
    if !in_range || yielder.is_null() {
        *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
        return 0;
    }
    match join(s, cur, yielder, slot) {
        Ok(v) => v,
        Err(()) => {
            *(trap_out as *mut i64) = TrapKind::ThreadFault as i64;
            0
        }
    }
}

/// `<ty>.atomic.wait` thunk: if the `width`-byte value at confined address `phys` still equals
/// `expected`, park the running vCPU on `phys` until a notify or `timeout` ns elapse (`< 0` = forever).
/// Returns the `i32` status (`WAIT_WOKEN` / `WAIT_NOT_EQUAL` / `WAIT_TIMED_OUT`).
///
/// # Safety
/// `s` is the run's live scheduler; `phys` points at `width` readable guest bytes (alignment guarded
/// by the lowering).
pub(crate) unsafe extern "C" fn thread_wait(
    s: *mut Sched,
    phys: u64,
    expected: u64,
    width: u32,
    timeout: i64,
) -> i32 {
    let mask = width_mask(width);
    if read_phys(phys, width) & mask != expected & mask {
        return WAIT_NOT_EQUAL;
    }
    let (yielder, deadline) = {
        let s = &mut *s;
        let yielder = s
            .vcpus
            .get(s.cur)
            .and_then(|v| v.as_ref())
            .map(|v| v.yielder)
            .unwrap_or(std::ptr::null());
        let deadline = if timeout < 0 {
            u64::MAX // "forever": only fired if the run would otherwise deadlock
        } else {
            s.clock.saturating_add(timeout as u64)
        };
        (yielder, deadline)
    };
    if yielder.is_null() {
        return WAIT_TIMED_OUT;
    }
    wait_park(s, phys, deadline, yielder)
}

/// `atomic.notify` thunk: wake up to `count` vCPUs parked on confined address `phys`; returns the
/// `i32` count woken. Accesses no memory.
///
/// # Safety
/// `s` is the run's live scheduler.
pub(crate) unsafe extern "C" fn thread_notify(s: *mut Sched, phys: u64, count: i32) -> i32 {
    notify(s, phys, count.max(0) as u32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const STACK: usize = 64 * 1024;

    /// Root spawns three children that each return a value; root joins all and sums them. Join must
    /// block (the children run only after the root yields by joining), and every result must arrive.
    #[test]
    fn spawn_join_sum() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            let mut total = 0i64;
            let mut handles = Vec::new();
            for k in 1..=3i64 {
                handles.push(ctx.spawn(Box::new(move |_c: &Ctx| k * 10)));
            }
            for h in handles {
                total += ctx.join(h).unwrap();
            }
            total
        }));
        assert_eq!(r, Some(60)); // 10 + 20 + 30
    }

    /// Nested spawn: a child spawns a grandchild and returns grandchild_result + 1; the root joins the
    /// child. Exercises spawning + joining from *within* a running vCPU (the reentrant path).
    #[test]
    fn nested_spawn_join() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            let child = ctx.spawn(Box::new(|c: &Ctx| {
                let g = c.spawn(Box::new(|_c: &Ctx| 41));
                c.join(g).unwrap() + 1
            }));
            ctx.join(child).unwrap()
        }));
        assert_eq!(r, Some(42));
    }

    /// Joining a child that is spawned but not yet run must block the parent and resume it with the
    /// child's result — and joining an out-of-range handle is inert (`Err`).
    #[test]
    fn join_blocks_and_forged_handle_is_inert() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            assert!(ctx.join(9999).is_err(), "forged handle must be inert");
            let h = ctx.spawn(Box::new(|_c: &Ctx| 7));
            let first = ctx.join(h).unwrap();
            // A second join of the same (now-consumed) handle is inert.
            assert!(ctx.join(h).is_err(), "re-join must be inert");
            first
        }));
        assert_eq!(r, Some(7));
    }

    /// Many children interleave correctly and each returns its own value.
    #[test]
    fn many_children_independent() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            let handles: Vec<usize> = (0..16)
                .map(|k| ctx.spawn(Box::new(move |_c: &Ctx| (k * k) as i64)))
                .collect();
            handles.into_iter().map(|h| ctx.join(h).unwrap()).sum()
        }));
        let want: i64 = (0..16i64).map(|k| k * k).sum();
        assert_eq!(r, Some(want));
    }

    const KEY: u64 = 0xF00D;

    /// A waiter blocks on a futex key; a separate notifier wakes it. The waiter must resume with
    /// `WAIT_WOKEN`. (Drives the block→notify path the cooperative JIT can't reach root-first.)
    #[test]
    fn wait_then_notify_wakes() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            let waiter = ctx.spawn(Box::new(|c: &Ctx| c.wait(KEY, u64::MAX) as i64));
            let notifier = ctx.spawn(Box::new(|c: &Ctx| c.notify(KEY, 1) as i64));
            let woke = ctx.join(notifier).unwrap();
            let status = ctx.join(waiter).unwrap();
            woke * 10 + status // 1 woken, status 0 → 10
        }));
        assert_eq!(r, Some(10));
    }

    /// A waiter that is never notified times out once nothing else can run (`WAIT_TIMED_OUT`), the
    /// logical clock advancing to its deadline.
    #[test]
    fn wait_times_out() {
        let mut s = Sched::new(STACK);
        let r = s.run(Box::new(|ctx: &Ctx| {
            let waiter = ctx.spawn(Box::new(|c: &Ctx| c.wait(KEY, 1_000) as i64));
            ctx.join(waiter).unwrap()
        }));
        assert_eq!(r, Some(WAIT_TIMED_OUT as i64));
    }
}
