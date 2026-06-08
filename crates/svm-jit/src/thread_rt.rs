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

use std::collections::VecDeque;
use svm_fiber::{Fiber, State, Yielder};

/// A guest thread body: receives a [`Ctx`] (its scheduler handle, for nested spawn/join) and returns
/// the thread's `i64` result. `'static` so it can live in the heap-allocated fiber.
pub(crate) type Body = Box<dyn FnOnce(&Ctx) -> i64 + 'static>;

/// Why a vCPU suspended back to the scheduler.
enum Block {
    /// Waiting for child vCPU (task id) to finish.
    Join(usize),
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
}

/// The cooperative scheduler: a fiber table + runnable queue + join parking.
pub(crate) struct Sched {
    vcpus: Vec<Option<VCpu>>,
    runnable: VecDeque<usize>,
    /// `(child, parent)` pairs: `parent` is parked until `child` finishes.
    join_waiters: Vec<(usize, usize)>,
    /// The currently running vCPU (valid while a fiber is resumed).
    cur: usize,
    /// Set by a blocking thunk on the running vCPU's stack just before it suspends; read by the loop.
    block: Option<Block>,
    /// Per-vCPU control-stack size.
    stack: usize,
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
}

impl Sched {
    pub(crate) fn new(stack: usize) -> Sched {
        Sched {
            vcpus: Vec::new(),
            runnable: VecDeque::new(),
            join_waiters: Vec::new(),
            cur: 0,
            block: None,
            stack,
        }
    }

    /// Spawn the root vCPU running `body`, drive the schedule to completion, and return the root's
    /// result (or `None` on a guest deadlock — nothing runnable but vCPUs still live).
    pub(crate) fn run(&mut self, body: Body) -> Option<i64> {
        let self_ptr: *mut Sched = self;
        // SAFETY: single-threaded driver; see the module reentrancy note.
        let root = unsafe { spawn(self_ptr, body) };
        loop {
            let Some(tid) = self.runnable.pop_front() else {
                break;
            };
            self.cur = tid;
            // Extract a raw fiber pointer and release the `vcpus` borrow before the switch, so a
            // re-entrant `spawn` may grow the table without aliasing the resumed fiber.
            let fib: *mut Fiber = match &mut self.vcpus[tid] {
                Some(v) if !v.fiber.is_done() => &mut *v.fiber as *mut Fiber,
                _ => continue,
            };
            // SAFETY: `fib` is an address-stable boxed fiber; only one vCPU runs at a time.
            let st = unsafe { (*fib).resume(0) };
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
}
