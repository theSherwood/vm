//! Parallel (multi-core) executor core for the JIT thread scheduler (§12) — `M` green-thread vCPUs
//! across `N` OS-thread workers.
//!
//! This is the concurrent *protocol*: a `Mutex`-guarded runnable queue + join parking + completion
//! wakeups, driven by a pool of worker threads. The verification-critical invariant is the **lock
//! discipline**: a worker holds the lock only to *pick* a task and to *handle its result*, never
//! across **running** the task — because a running task (a vCPU) calls back in (spawn/join), which
//! re-takes the lock, so holding it across a run would deadlock. (Same shape as the interpreter's real
//! `Scheduler`.)
//!
//! Because fibers/JIT can't run under `loom` (loom controls `loom::thread`, not native stack
//! switches), the core is generic over an abstract [`Task`]: the real backend resumes a `Fiber`
//! (step 2b), while the tests here use **mock tasks**, so `loom` can exhaustively explore the
//! worker/queue/wake races — the part that's genuinely hard to get right. Sync primitives are
//! `loom::sync` under `--cfg loom`, else `std::sync`.
//!
//! Scope today: `spawn` + `join` (the wakeup protocol). `wait`/`notify` parking layers on next.

#![allow(dead_code)] // wired into the JIT execution path in step 2b

use std::collections::VecDeque;
use std::sync::Arc; // `Arc`'s refcount isn't the race of interest; loom only swaps the rest below.

#[cfg(loom)]
use loom::sync::{Condvar, Mutex};
#[cfg(loom)]
use loom::thread;
#[cfg(not(loom))]
use std::sync::{Condvar, Mutex};
#[cfg(not(loom))]
use std::thread;

/// What running a [`Task`] yields: it finished with a result, or it blocked joining child vCPU `tid`.
pub(crate) enum Step {
    Done(i64),
    Join(usize),
}

/// A schedulable green thread. The real backend (step 2b) resumes a `Fiber`; the tests use mocks.
/// `run` executes until the next block/completion; it may call `sched` to spawn children or read
/// results (those lock internally — `run` itself is called with **no** lock held).
pub(crate) trait Task: Send {
    fn run(&mut self, resume_val: i64, sched: &Arc<Shared>) -> Step;
}

struct Inner {
    /// vCPU ids ready to run.
    runnable: VecDeque<usize>,
    /// Each vCPU's task: `Some` while parked/pending, `None` while a worker is running it (taken out
    /// so no two workers touch it — the ownership hand-off that makes migration sound).
    tasks: Vec<Option<Box<dyn Task>>>,
    /// Completed results, by vCPU id.
    results: Vec<Option<i64>>,
    /// `(child, parent)` — `parent` is parked until `child` finishes.
    join_waiters: Vec<(usize, usize)>,
    /// Value delivered to a vCPU on its next run (a wake status; 0 otherwise).
    resume_val: Vec<i64>,
    /// vCPUs not yet finished. When this hits 0 the run is done and idle workers are released.
    live: usize,
    /// Tear the pool down early — a guest trap kills the whole domain (workers stop without finishing
    /// the remaining vCPUs).
    shutdown: bool,
}

/// The lock guard type (distinct between `loom::sync` and `std::sync`).
#[cfg(loom)]
type Guard<'a> = loom::sync::MutexGuard<'a, Inner>;
#[cfg(not(loom))]
type Guard<'a> = std::sync::MutexGuard<'a, Inner>;

/// The shared scheduler: state behind a `Mutex`, plus a `Condvar` workers block on for work.
pub(crate) struct Shared {
    inner: Mutex<Inner>,
    cvar: Condvar,
}

impl Shared {
    pub(crate) fn new() -> Arc<Shared> {
        Arc::new(Shared {
            inner: Mutex::new(Inner {
                runnable: VecDeque::new(),
                tasks: Vec::new(),
                results: Vec::new(),
                join_waiters: Vec::new(),
                resume_val: Vec::new(),
                live: 0,
                shutdown: false,
            }),
            cvar: Condvar::new(),
        })
    }

    fn lock(&self) -> Guard<'_> {
        self.inner.lock().unwrap()
    }

    /// Tear the pool down (a guest trap): wake every worker so they exit.
    pub(crate) fn request_shutdown(&self) {
        self.lock().shutdown = true;
        self.cvar.notify_all();
    }

    /// Spawn a new vCPU running `task`; returns its id. Wakes one idle worker. Callable from a running
    /// task (locks internally; the caller holds no lock).
    pub(crate) fn spawn(&self, task: Box<dyn Task>) -> usize {
        let mut g = self.lock();
        let tid = g.tasks.len();
        g.tasks.push(Some(task));
        g.results.push(None);
        g.resume_val.push(0);
        g.live += 1;
        g.runnable.push_back(tid);
        drop(g);
        self.cvar.notify_one();
        tid
    }

    /// Number of vCPUs created so far (for the spawn anti-bomb cap) and the live-bound check.
    pub(crate) fn task_count(&self) -> usize {
        self.lock().tasks.len()
    }

    /// The result of vCPU `tid`, if it has finished.
    pub(crate) fn result_of(&self, tid: usize) -> Option<i64> {
        self.lock().results.get(tid).copied().flatten()
    }

    /// One worker thread: pick a runnable vCPU, run it **without the lock**, then re-lock to park it
    /// (blocked) or record its result and wake its joiners (done). Exits when all vCPUs are done.
    /// (`this` is a by-value arg, not `self`, because `loom::sync::Arc` can't be a `self` type.)
    fn worker(this: Arc<Shared>) {
        loop {
            // ---- pick a runnable vCPU (or exit when the run is finished) ----
            let (tid, mut task) = {
                let mut g = this.lock();
                loop {
                    if g.shutdown {
                        return; // domain killed (a trap); stop.
                    }
                    if let Some(tid) = g.runnable.pop_front() {
                        // Take the task out of its slot — exclusive ownership while we run it.
                        let task = g.tasks[tid].take().expect("runnable vCPU has its task");
                        break (tid, task);
                    }
                    if g.live == 0 {
                        // All done: release the other idle workers and exit.
                        this.cvar.notify_all();
                        return;
                    }
                    g = this.cvar.wait(g).unwrap();
                }
            };

            let resume_val = this.lock().resume_val[tid];
            // ---- run the vCPU with NO lock held (it may re-enter spawn/join, or trap) ----
            let step = task.run(resume_val, &this);

            // ---- handle the outcome ----
            let mut g = this.lock();
            if g.shutdown {
                // A trap (this task or another) is tearing the pool down; stop (this task's slot is
                // already empty — it's abandoned, like the rest).
                drop(g);
                this.cvar.notify_all();
                return;
            }
            match step {
                Step::Done(r) => {
                    g.results[tid] = Some(r);
                    g.live -= 1;
                    // Wake everyone joining this vCPU.
                    let mut i = 0;
                    while i < g.join_waiters.len() {
                        if g.join_waiters[i].0 == tid {
                            let (_, parent) = g.join_waiters.remove(i);
                            g.runnable.push_back(parent);
                        } else {
                            i += 1;
                        }
                    }
                    drop(g);
                    // Wake workers for the newly-runnable parents (and idlers to exit if live==0).
                    this.cvar.notify_all();
                }
                Step::Join(child) => {
                    if g.results[child].is_some() {
                        // Already done: re-run immediately to collect the result.
                        g.tasks[tid] = Some(task);
                        g.runnable.push_back(tid);
                        drop(g);
                        this.cvar.notify_one();
                    } else {
                        // Park: put the task back in its slot and record the join edge.
                        g.tasks[tid] = Some(task);
                        g.join_waiters.push((child, tid));
                    }
                }
            }
        }
    }
}

/// Drive `root` (vCPU 0) to completion on `workers` OS-thread workers over a **pre-created** `shared`
/// (so its address can be baked into the JITted `thread.*` thunks before this is called); returns the
/// root's result (or `None` if the run deadlocked / was torn down by a trap).
pub(crate) fn run_pool(shared: &Arc<Shared>, workers: usize, root: Box<dyn Task>) -> Option<i64> {
    let root_id = shared.spawn(root);
    let handles: Vec<_> = (0..workers.max(1))
        .map(|_| {
            let s = Arc::clone(shared);
            thread::spawn(move || Shared::worker(s))
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let result = shared.lock().results[root_id];
    result
}

/// Convenience for the tests: create a fresh scheduler and run it.
#[cfg(test)]
pub(crate) fn run(workers: usize, root: Box<dyn Task>) -> Option<i64> {
    run_pool(&Shared::new(), workers, root)
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    /// A leaf vCPU that finishes immediately with a value.
    struct Leaf(i64);
    impl Task for Leaf {
        fn run(&mut self, _rv: i64, _s: &Arc<Shared>) -> Step {
            Step::Done(self.0)
        }
    }

    /// A vCPU that spawns `vals.len()` leaf children, joins them all, and returns their sum.
    struct Parent {
        vals: Vec<i64>,
        handles: Vec<usize>,
        sum: i64,
        next: usize,
        spawned: bool,
    }
    impl Parent {
        fn new(vals: Vec<i64>) -> Box<Parent> {
            Box::new(Parent {
                vals,
                handles: Vec::new(),
                sum: 0,
                next: 0,
                spawned: false,
            })
        }
    }
    impl Task for Parent {
        fn run(&mut self, _rv: i64, sched: &Arc<Shared>) -> Step {
            if !self.spawned {
                for &v in &self.vals {
                    self.handles.push(sched.spawn(Box::new(Leaf(v))));
                }
                self.spawned = true;
            }
            // Collect every child that is ready; block on the first that isn't.
            while self.next < self.handles.len() {
                let h = self.handles[self.next];
                match sched.result_of(h) {
                    Some(r) => {
                        self.sum += r;
                        self.next += 1;
                    }
                    None => return Step::Join(h),
                }
            }
            Step::Done(self.sum)
        }
    }

    /// Real OS threads, many runs: the parallel pool must always compute the exact sum (no lost task,
    /// no lost wakeup, no double-run).
    #[test]
    fn parallel_pool_sums_children() {
        for _ in 0..2000 {
            let vals: Vec<i64> = (1..=8).collect();
            let want: i64 = vals.iter().sum();
            let got = run(4, Parent::new(vals));
            assert_eq!(got, Some(want));
        }
    }

    /// Deeper nesting across the pool: a parent whose children themselves have children.
    #[test]
    fn parallel_pool_nested() {
        struct Nest {
            depth: u32,
            child: Option<usize>,
        }
        impl Task for Nest {
            fn run(&mut self, _rv: i64, sched: &Arc<Shared>) -> Step {
                if self.depth == 0 {
                    return Step::Done(1);
                }
                match self.child {
                    None => {
                        let c = sched.spawn(Box::new(Nest {
                            depth: self.depth - 1,
                            child: None,
                        }));
                        self.child = Some(c);
                        Step::Join(c)
                    }
                    Some(c) => match sched.result_of(c) {
                        Some(r) => Step::Done(r + 1),
                        None => Step::Join(c),
                    },
                }
            }
        }
        for _ in 0..1000 {
            let got = run(
                4,
                Box::new(Nest {
                    depth: 8,
                    child: None,
                }),
            );
            assert_eq!(got, Some(9)); // 1 + depth
        }
    }
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;

    /// Explore the worker/queue/wake interleavings with mock tasks: a root spawns two leaf children
    /// and joins both on a 2-worker pool. loom drives the schedules of the two worker threads (lock
    /// order, condvar wait/notify, the complete-vs-join race) and the result must always be the exact
    /// sum — proving no lost task and no lost wakeup. A **preemption bound** keeps the condvar-heavy
    /// worker-pool tractable (loom's recommended practice — finds essentially all real concurrency bugs
    /// while bounding the state space).
    #[test]
    fn loom_spawn_join_no_lost_wakeup() {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(2);
        builder.check(|| {
            struct Leaf(i64);
            impl Task for Leaf {
                fn run(&mut self, _rv: i64, _s: &Arc<Shared>) -> Step {
                    Step::Done(self.0)
                }
            }
            struct Root {
                h: Vec<usize>,
                sum: i64,
                next: usize,
                spawned: bool,
            }
            impl Task for Root {
                fn run(&mut self, _rv: i64, sched: &Arc<Shared>) -> Step {
                    if !self.spawned {
                        self.h.push(sched.spawn(Box::new(Leaf(10))));
                        self.h.push(sched.spawn(Box::new(Leaf(20))));
                        self.spawned = true;
                    }
                    while self.next < self.h.len() {
                        match sched.result_of(self.h[self.next]) {
                            Some(r) => {
                                self.sum += r;
                                self.next += 1;
                            }
                            None => return Step::Join(self.h[self.next]),
                        }
                    }
                    Step::Done(self.sum)
                }
            }
            let got = run(
                2,
                Box::new(Root {
                    h: Vec::new(),
                    sum: 0,
                    next: 0,
                    spawned: false,
                }),
            );
            assert_eq!(got, Some(30));
        });
    }
}
